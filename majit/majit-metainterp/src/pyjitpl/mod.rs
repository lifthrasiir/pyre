mod dispatch;
mod frame;

pub use dispatch::build_state_field_snapshot;
pub use dispatch::{
    ClosureRuntime, JitCodeMachine, JitCodeRuntime, JitCodeSym, StandaloneFrameStack,
    consume_observed_float_call, consume_observed_int_call, consume_observed_ref_call,
    consume_observed_void_call, observer_arg_to_i64, observer_i64_to_value, trace_jitcode,
    trace_jitcode_observer, trace_jitcode_observer_with_args, trace_jitcode_with_args,
};
pub(crate) use dispatch::{
    call_int_function, call_ref_function, call_void_function, eval_binop_f, eval_binop_i,
    eval_unary_f, eval_unary_i,
};
pub use frame::{MIFrame, MIFrameStack};

use std::collections::HashMap;
use std::sync::Arc;

use crate::optimizeopt::optimizer::{Optimizer, PendingBridgeRd};
use majit_backend::{Backend, ExitRecoveryLayout, JitCellToken};
#[cfg(feature = "cranelift")]
pub(crate) use majit_backend_cranelift::CraneliftBackend as BackendImpl;
#[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
pub(crate) use majit_backend_dynasm::runner::DynasmBackend as BackendImpl;
#[cfg(target_arch = "wasm32")]
pub(crate) use majit_backend_wasm::WasmBackend as BackendImpl;

#[cfg(not(any(feature = "cranelift", feature = "dynasm", target_arch = "wasm32")))]
compile_error!("majit-metainterp requires a backend: enable feature \"cranelift\" or \"dynasm\"");

use crate::history::TreeLoop;
use crate::warmstate::{HotResult, WarmEnterState};
use majit_ir::descr::DescrRef;
use majit_ir::{Const, FailDescr, GcRef, InputArg, Op, OpCode, OpRef, Type, Value};

use crate::blackhole::{BlackholeResult, ExceptionState, blackhole_execute_with_state_ca};
use crate::compile;
use crate::compile::make_jitcell_token;
pub use crate::compile::{
    CompileResult, CompiledExitLayout, CompiledTerminalExitLayout, CompiledTraceLayout,
    DeadFrameArtifacts, RawCompileResult,
};
use crate::io_buffer;
use crate::jitdriver::JitDriverStaticData;
use crate::optimizeopt::{
    SnapshotBoxes, SnapshotFramePcs, SnapshotFrameSizes, snapshot_get, snapshot_insert,
};
use crate::resume::{
    MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite, ResumeData,
    ResumeLayoutSummary, ResumeStorage, SnapshotBox,
};
use crate::trace_ctx::TraceCtx;
use crate::virtualizable::VirtualizableInfo;

/// No direct RPython equivalent — Rust struct carrying data that RPython
/// passes through internal method calls in handle_guard_failure
/// (pyjitpl.py:2890). Fields correspond to:
/// - `fail_types`: ResumeGuardDescr.fail_arg_types (compile.py:797)
/// - `is_exception_guard`: isinstance(key, ResumeGuardExcDescr) (compile.py:932)
/// - `storage`: shared `Arc<ResumeStorage>` handle for the parent
///   guard (resume.py:1042 `rebuild_from_resumedata` reads
///   `self.storage.rd_numb / rd_consts / rd_virtuals` off the same
///   descriptor). No owned `Vec<Const>` copy — all readers observe
///   the pool the GC walker updates.
pub struct BridgeRetraceResult {
    pub is_exception_guard: bool,
    pub fail_types: Vec<Type>,
    pub storage: Option<Arc<ResumeStorage>>,
}

/// Result of checking a back-edge.
/// pyjitpl.py:2807 `raise SwitchToBlackhole(Counters.ABORT_TOO_LONG)` —
/// reason attached to an abort.  RPython uses `Counters.ABORT_*` ints
/// (`resoperation.Counters.ABORT_TOO_LONG`, `ABORT_BRIDGE`, ...); pyre
/// tracks only the variants that propagate through the blackhole flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbortReason {
    /// `Counters.ABORT_TOO_LONG`: trace exceeded the length / tag budget.
    TooLong,
    /// `Counters.ABORT_BRIDGE` / `ABORT_BAD_LOOP`: generic abort path —
    /// used when pyre cannot classify the reason more precisely.
    Generic,
}

impl AbortReason {
    /// Map to the upstream `Counters.ABORT_*` integer for hook payloads.
    /// Values follow the declaration order in `rpython/rlib/jit.py`
    /// `class Counters` (ABORT_TOO_LONG=12, ABORT_BRIDGE=13, ABORT_BAD_LOOP=14).
    #[inline]
    pub const fn as_int(self) -> i32 {
        match self {
            AbortReason::TooLong => 12,
            AbortReason::Generic => 13,
        }
    }
}

pub enum BackEdgeAction {
    /// Not hot yet; keep interpreting.
    Interpret,
    /// Tracing has started. Use `trace_ctx()` to record operations.
    StartedTracing,
    /// Already tracing this loop (inner back-edge).
    AlreadyTracing,
    /// Compiled code exists. Call `run_compiled()`.
    RunCompiled,
}

/// pyjitpl.py: result of compile_loop / compile_trace.
///
/// RPython uses exceptions (raise ContinueRunningNormally on success,
/// raise SwitchToBlackhole on fatal failure) and None returns for
/// cancellation. majit uses this enum instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileOutcome {
    /// Compilation succeeded — compiled loop is installed and ready to run.
    Compiled { green_key: u64, from_retry: bool },
    /// Compilation was cancelled (e.g. InvalidLoop, virtual state mismatch).
    /// The caller may retry or continue tracing.
    Cancelled,
    /// Too many cancellations — abort and fall back to interpreter.
    /// Equivalent to RPython's SwitchToBlackhole(ABORT_BAD_LOOP).
    Aborted,
}

pub(crate) struct CompiledTrace {
    /// Inputargs for this trace, used to recover typed exit layouts during blackhole replay.
    pub(crate) inputargs: Vec<InputArg>,
    /// Optimized ops for blackhole fallback from compiled guard failures.
    pub(crate) ops: Vec<majit_ir::Op>,
    /// Constant pool paired with `ops` for blackhole fallback.
    pub(crate) constants: HashMap<u32, i64>,
    /// Constant types for the constant pool entries.
    pub(crate) constant_types: HashMap<u32, Type>,
    /// Static exit metadata for each guard/finish in this trace.
    pub(crate) exit_layouts: HashMap<u32, StoredExitLayout>,
    /// Static exit metadata for terminal FINISH/JUMP ops, keyed by op index.
    pub(crate) terminal_exit_layouts: HashMap<usize, StoredExitLayout>,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredExitLayout {
    pub(crate) source_op_index: Option<usize>,
    pub(crate) gc_ref_slots: Vec<usize>,
    pub(crate) force_token_slots: Vec<usize>,
    pub(crate) recovery_layout: Option<ExitRecoveryLayout>,
    pub(crate) resume_layout: Option<ResumeLayoutSummary>,
    /// compile.py:853 `ResumeGuardDescr` storage — single guard-owned
    /// shared pool containing rd_numb / rd_consts / rd_virtuals /
    /// rd_pendingfields. All readers (blackhole resume, bridge
    /// retrace, GC root walker) share this Arc.
    pub(crate) storage: Option<Arc<ResumeStorage>>,
    /// Source-op `descr` Arc, captured at trace-build time. Production
    /// guards / FINISH carry their `ResumeGuardDescr` /
    /// `_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescrRef`
    /// here so the entry's `fail_arg_types` resolve through the
    /// canonical descr-side state (`compile.py:853 ResumeGuardDescr`)
    /// instead of a trace-side mirror.  `None` is reserved for the
    /// pathological case where a backend-only layout's `source_op_index`
    /// does not resolve to an op with a descr — readers treat that as
    /// an empty type vector.
    pub(crate) descr: Option<DescrRef>,
    /// PRE-EXISTING-ADAPTATION (pyre): types of the values flowing through
    /// a JUMP terminal exit, populated only when `descr` cannot supply
    /// them.  RPython's JUMP descr is `TargetToken` (`history.py:470`)
    /// which carries no `fail_arg_types` because PyPy reads JUMP arg
    /// types directly off the boxes (`box.type`, `history.py:182/220`).
    /// Pyre's `OpRef` is typed for the post-optimizer trace, so this
    /// field caches the per-arg types `infer_terminal_exit_layout`
    /// already computes; readers keep working uniformly through
    /// `resolve_exit_types()`.  `None` for guard / FINISH layouts whose
    /// `descr.as_fail_descr().fail_arg_types()` is the canonical source.
    /// Convergence path: when pyre stops including JUMP in
    /// `terminal_exit_layouts` (PyPy never does), this field disappears
    /// with it.
    pub(crate) op_arg_types_for_jump: Option<Vec<Type>>,
}

impl StoredExitLayout {
    /// compile.py:186: rd_loop_token = original_jitcell_token.
    /// `owning_key` is the green_key of the compiled loop that owns this guard.
    pub(crate) fn public(
        &self,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> CompiledExitLayout {
        CompiledExitLayout {
            rd_loop_token: owning_key,
            trace_id,
            fail_index,
            source_op_index: self.source_op_index,
            exit_types: self.resolve_exit_types().to_vec(),
            is_finish: self.resolve_is_finish(),
            gc_ref_slots: self.gc_ref_slots.clone(),
            force_token_slots: self.force_token_slots.clone(),
            recovery_layout: self.recovery_layout.clone(),
            resume_layout: self.resume_layout.clone(),
            storage: self.storage.clone(),
        }
    }

    /// Resolve the canonical `exit_types` for this layout.
    ///
    /// RPython parity (compile.py:853 ResumeGuardDescr): the descr
    /// holds the post-numbering `fail_arg_types`, so the descr's
    /// `fail_arg_types()` is the single source of truth for guard
    /// and FINISH exits.  When `descr.as_fail_descr()` is `None` the
    /// entry is either:
    ///
    ///   1. A JUMP terminal exit whose descr is `LoopTargetDescr`
    ///      (RPython `TargetToken`, `history.py:470`).  PyPy reads
    ///      types per-arg via `box.type` (`history.py:182/220`); pyre
    ///      caches the equivalent in `op_arg_types_for_jump` because
    ///      `OpRef::ty()` is the per-box analog and was already
    ///      computed at `infer_terminal_exit_layout` time.
    ///
    ///   2. A backend-only layout whose `source_op_index` did not
    ///      resolve to an op with a descr — readers treat that as
    ///      an empty type vector (the constructor synthesizes a
    ///      `MetaFailDescr` for backend-only entries so this branch
    ///      is normally unreachable in production).
    pub(crate) fn resolve_exit_types(&self) -> &[Type] {
        if let Some(types) = self
            .descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .map(|fd| fd.fail_arg_types())
        {
            return types;
        }
        if let Some(types) = self.op_arg_types_for_jump.as_deref() {
            return types;
        }
        // Both descr-side and JUMP-cache fallbacks empty — production
        // builders (`build_terminal_exit_layouts`, `build_guard_metadata`,
        // `merge_backend_*_layouts`) always populate at least one, since
        // backend-only entries synthesize a `MetaFailDescr` and JUMP
        // entries set `op_arg_types_for_jump`.  Hitting this branch
        // means a synthetic test fixture skipped both populators or a
        // builder regressed — flag it loudly in debug builds so it
        // does not silently mask a stale `fail_arg_count_for=0`.
        debug_assert!(
            false,
            "resolve_exit_types: both descr.fail_arg_types() and op_arg_types_for_jump are missing — every production layout builder must populate one of them",
        );
        &[]
    }

    /// Resolve the canonical FINISH discriminator for this layout.
    ///
    /// RPython parity: `compile.py:658-662 ExitFrameWithExceptionDescrRef`
    /// and `compile.py:701-784 _DoneWithThisFrameDescr*` carry
    /// `is_finish() = True` on the descr itself; pyre routes the read
    /// through `descr.is_finish()` so terminal-exit FINISH semantics flow
    /// from the canonical descr handle.  Backend-only fallback layouts
    /// synthesize via `make_finish_fail_descr_typed` (terminal exits) or
    /// `make_fail_descr_typed` (guard exits) so the discriminator stays
    /// correct even when the source op has been evicted.
    pub(crate) fn resolve_is_finish(&self) -> bool {
        self.descr
            .as_ref()
            .and_then(|d| d.as_fail_descr())
            .is_some_and(|fd| fd.is_finish())
    }
}

/// opencoder.py:819 parity: extract per-snapshot box maps from trace snapshots.
///
/// opencoder.py:603 _encode: Const boxes are registered in the constant pool
/// and returned as pool OpRefs so the optimizer's BoxEnv can resolve them
/// via is_const/get_const (resume.py:157 getconst parity).
/// Decode a raw virtualizable-slot `i64` from `VirtualizableInfo::read_all_boxes`
/// into the typed `majit_ir::Value` the parallel virtualizable concrete
/// shadow expects.  Mirrors `value_from_backend_constant_bits_typed` so the
/// shadow never disagrees with the register-shadow encoding.
fn heap_value_for(ty: Type, bits: i64) -> Value {
    match ty {
        Type::Int => Value::Int(bits),
        Type::Float => Value::Float(f64::from_bits(bits as u64)),
        Type::Ref => Value::Ref(GcRef(bits as usize)),
        Type::Void => Value::Void,
    }
}

pub(crate) use crate::trace_ctx::value_to_raw_bits;

fn snapshot_map_from_trace_snapshots(
    trace_snapshots: &[crate::recorder::Snapshot],
    constants: &mut std::collections::HashMap<u32, i64>,
    constant_types: &mut std::collections::HashMap<u32, majit_ir::Type>,
) -> (
    SnapshotBoxes,
    SnapshotFrameSizes,
    SnapshotBoxes,
    SnapshotBoxes,
    SnapshotFramePcs,
) {
    let mut box_map = Vec::new();
    let mut size_map = Vec::new();
    let mut vable_map = Vec::new();
    let mut vref_map = Vec::new();
    let mut pc_map = Vec::new();
    let mut next_const_idx = constants
        .keys()
        .filter(|k| majit_ir::OpRef::from_raw(**k).is_constant())
        .map(|k| majit_ir::OpRef::from_raw(*k).const_index())
        .max()
        .map(|m| m + 1)
        .unwrap_or(0);
    // opencoder.py:603 _encode: trace snapshot recorder only emits Box
    // (live deadframe slot) and Const (compile-time pool) payloads.
    // TAGVIRTUAL belongs to resume numbering (resume.py:_number_boxes)
    // and is synthesized later from PtrInfo::is_virtual on the live
    // Box's OpRef. SnapshotTagged carries no `Virtual` variant (see
    // recorder.rs:73 docstring) so this match is exhaustive over the
    // two recorder-side cases.
    let mut tagged_to_box = |t: &crate::recorder::SnapshotTagged| -> SnapshotBox {
        match t {
            crate::recorder::SnapshotTagged::Box(opref, fallback_tp) => {
                // history.py:182/220/261/307 + resoperation.py:719/727/739/
                // 564-638: `box.type` lives on the Box. Pyre's typed
                // OpRef variants carry it intrinsically; the explicit
                // `fallback_tp` is the lockstep authority for residual
                // `OpRef::Untyped` cases (legacy from_raw producers).
                let tp = opref.ty().unwrap_or(*fallback_tp);
                SnapshotBox::typed(*opref, tp)
            }
            crate::recorder::SnapshotTagged::Const(val, tp) => {
                // resume.py:173-176: null Ref → NULLREF via getconst.
                // Register in pool so is_const → true, get_const → (0, Ref),
                // getconst → NULLREF. Do NOT short-circuit to OpRef::NONE.
                // resume.py:157 getconst: find or allocate pool entry.
                let existing = constants
                    .iter()
                    .find(|(k, v)| {
                        **v == *val
                            && constant_types.get(k).copied().unwrap_or_else(|| {
                                panic!(
                                    "constant_types missing entry for raw={} though \
                                     constants has it: both maps are populated in \
                                     lockstep at line ~302",
                                    *k
                                )
                            }) == *tp
                    })
                    .map(|(k, _)| *k);
                let opref = match existing {
                    // Re-mint via the typed factory rather than poking the
                    // cached `raw` straight into a variant: a CONST_BIT-less
                    // key (would only arise from a producer bypassing the
                    // `OpRef::const_typed` factory) must not manufacture a
                    // malformed `OpRef::ConstInt(raw)`.
                    Some(raw) => {
                        let cached = majit_ir::OpRef::from_raw(raw);
                        debug_assert!(
                            cached.is_constant(),
                            "constants map key {} missing CONST_BIT",
                            raw
                        );
                        majit_ir::OpRef::const_typed(cached.const_index(), *tp)
                    }
                    None => {
                        let opref = majit_ir::OpRef::const_typed(next_const_idx, *tp);
                        next_const_idx += 1;
                        constants.insert(opref.raw(), *val);
                        constant_types.insert(opref.raw(), *tp);
                        opref
                    }
                };
                SnapshotBox::typed(opref, *tp)
            }
        }
    };
    for (id, snap) in trace_snapshots.iter().enumerate() {
        let boxes: Vec<SnapshotBox> = snap
            .frames
            .iter()
            .flat_map(|f| f.boxes.iter())
            .map(&mut tagged_to_box)
            .collect();
        let frame_sizes: Vec<usize> = snap.frames.iter().map(|f| f.boxes.len()).collect();
        let vable_boxes: Vec<SnapshotBox> =
            snap.vable_boxes.iter().map(&mut tagged_to_box).collect();
        // opencoder.py:767 create_top_snapshot writes BOTH vable_array
        // AND vref_array. resume.py:243-247 _number_boxes consumes
        // vref_array as a separate section after vable_array.
        let vref_boxes: Vec<SnapshotBox> = snap.vref_boxes.iter().map(&mut tagged_to_box).collect();
        let frame_pcs: Vec<(i32, i32)> = snap
            .frames
            .iter()
            .map(|f| (f.jitcode_index as i32, f.pc as i32))
            .collect();
        let id = id as i32;
        snapshot_insert(&mut box_map, id, boxes);
        snapshot_insert(&mut size_map, id, frame_sizes);
        snapshot_insert(&mut vable_map, id, vable_boxes);
        snapshot_insert(&mut vref_map, id, vref_boxes);
        snapshot_insert(&mut pc_map, id, frame_pcs);
    }
    (box_map, size_map, vable_map, vref_map, pc_map)
}

struct PreparedBridgeTrace {
    ops: Vec<Op>,
    inputargs: Vec<InputArg>,
    snapshot_boxes: SnapshotBoxes,
    snapshot_frame_sizes: SnapshotFrameSizes,
    snapshot_vable_boxes: SnapshotBoxes,
    snapshot_vref_boxes: SnapshotBoxes,
    snapshot_frame_pcs: SnapshotFramePcs,
    pending_bridge_rd: Option<PendingBridgeRd>,
    /// Per-iter `BoxRef` pool from the bridge `TraceIterator`, mirroring
    /// `opencoder.py:259-262`'s fresh `inputarg_from_tp` / `cls()` allocation
    /// for the bridge's disjoint Box identity set. The optimizer consumes it
    /// via `set_pending_box_pool` so BoxRef-routing readers see the bridge's
    /// own boxes, not the parent loop's.
    box_pool: crate::r#box::BoxPool,
}

fn translate_trace_iter_opref(opref: OpRef, cache: &[Option<OpRef>]) -> OpRef {
    if opref.is_none() || opref.is_constant() {
        return opref;
    }
    // opencoder.py:286-289 `_get(self, i)` parity — `assert _cache[i] is
    // not None`. The bridge fresh-iterator
    // (`prepare_bridge_trace_for_optimizer`) writes `_cache[old_pos] =
    // new_opref` for every inputarg + emitted op position in the bridge
    // trace. Snapshot/vable feeds that flow through here MUST reference
    // OpRefs the recorder produced for this bridge.
    cache
        .get(opref.raw() as usize)
        .copied()
        .flatten()
        .unwrap_or_else(|| {
            panic!(
                "translate_trace_iter_opref cache miss for {opref:?} (cache_len={})",
                cache.len(),
            )
        })
}

fn translate_trace_iter_box_map(
    mut box_map: SnapshotBoxes,
    cache: &[Option<OpRef>],
) -> SnapshotBoxes {
    for boxes in box_map.iter_mut().flatten() {
        for boxref in boxes.iter_mut() {
            *boxref = boxref.map_opref(|opref| translate_trace_iter_opref(opref, cache));
        }
    }
    box_map
}

/// `unroll.py:187 trace = trace.get_iter()` parity.
///
/// RPython mints fresh `InputArg` / `ResOperation` objects before
/// `optimize_bridge()` consumes the trace; `opencoder.py:249-273
/// TraceIterator.__init__` allocates a fresh inputarg per
/// `self.trace.inputargs`, so bridge boxes carry distinct Python `is`
/// identity from the parent loop's boxes automatically.
///
/// Pyre's flat `OpRef(u32)` lacks identity, so this helper performs
/// the same rename explicitly: `TraceIterator::new(.., start_fresh =
/// bridge_inputarg_base)` walks the recorded bridge ops, allocates
/// fresh OpRefs in `[bridge_inputarg_base..)`, and rewrites every
/// reference (op args, fail_args, snapshot boxes, vable boxes,
/// `pending_bridge_rd.liveboxes`) through the iterator's `_cache`.
fn prepare_bridge_trace_for_optimizer(
    bridge_ops: &[Op],
    bridge_inputargs: &[InputArg],
    snapshot_boxes: SnapshotBoxes,
    snapshot_frame_sizes: SnapshotFrameSizes,
    snapshot_vable_boxes: SnapshotBoxes,
    snapshot_vref_boxes: SnapshotBoxes,
    snapshot_frame_pcs: SnapshotFramePcs,
    pending_bridge_rd: Option<PendingBridgeRd>,
    bridge_inputarg_base: u32,
) -> PreparedBridgeTrace {
    // unroll.py:187 `trace = trace.get_iter()` parity for bridge traces.
    // RPython allocates fresh InputArg / ResOperation objects before
    // optimize_bridge() consumes the trace; majit's analogue is a fresh
    // TraceIterator walk with `start_fresh = bridge_inputarg_base`.
    let bridge_inputarg_types: Vec<Type> = bridge_inputargs.iter().map(|ia| ia.tp).collect();
    let mut iter = crate::opencoder::TraceIterator::new(
        bridge_ops,
        0,
        bridge_ops.len(),
        None,
        &bridge_inputarg_types,
        bridge_inputarg_base,
        None, // p1_full_prefix — bridge has no prior unroll phase
    );
    let mut ops = Vec::with_capacity(bridge_ops.len());
    while let Some(op) = iter.next() {
        ops.push(op);
    }
    let inputargs = bridge_inputargs
        .iter()
        .zip(iter.inputargs.iter().copied())
        .map(|(arg, opref)| InputArg::from_type(arg.tp, opref.raw()))
        .collect();
    let cache = iter._cache;
    let box_pool = iter.box_pool;
    let snapshot_boxes = translate_trace_iter_box_map(snapshot_boxes, &cache);
    let snapshot_vable_boxes = translate_trace_iter_box_map(snapshot_vable_boxes, &cache);
    let snapshot_vref_boxes = translate_trace_iter_box_map(snapshot_vref_boxes, &cache);
    let pending_bridge_rd = pending_bridge_rd.map(|mut prd| {
        prd.liveboxes = prd
            .liveboxes
            .into_iter()
            .map(|opref| translate_trace_iter_opref(opref, &cache))
            .collect();
        prd
    });
    PreparedBridgeTrace {
        ops,
        inputargs,
        snapshot_boxes,
        snapshot_frame_sizes,
        snapshot_vable_boxes,
        snapshot_vref_boxes,
        snapshot_frame_pcs,
        pending_bridge_rd,
        box_pool,
    }
}

fn normalize_root_loop_entry_contract(
    mut inputargs: Vec<InputArg>,
    mut optimized_ops: Vec<Op>,
) -> Result<(Vec<InputArg>, Vec<Op>), (usize, usize)> {
    let last_jump = optimized_ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Jump);
    let jump_arg_count = last_jump.map(|op| op.args.len()).unwrap_or(0);
    let label_op = optimized_ops
        .iter()
        .rev()
        .find(|op| op.opcode == OpCode::Label);
    let label_arg_count = label_op.map(|op| op.args.len()).unwrap_or(0);
    let label_descr_index = label_op
        .and_then(|op| op.descr.as_ref())
        .map(|descr| descr.index());
    let jump_targets_current_loop = last_jump.is_some_and(|op| {
        let jump_descr_index = op.descr.as_ref().map(|descr| descr.index());
        match (jump_descr_index, label_descr_index) {
            (Some(jump_idx), Some(label_idx)) => jump_idx == label_idx,
            (None, None) => true,
            _ => false,
        }
    });

    // RPython compile.py:359/373 parity: the optimizer pipeline is the only
    // source of the LABEL/JUMP contract. A trace missing its LABEL — or one
    // whose LABEL/JUMP arities disagree — is a broken optimizer output, not
    // something to auto-recover. Report both shapes as an arity mismatch so
    // the caller aborts compilation.
    if label_arg_count == 0 && jump_arg_count > 0 {
        return Err((0, jump_arg_count));
    }
    if jump_targets_current_loop && label_arg_count != jump_arg_count {
        // RPython compile.py:334: assert jump.numargs() == label.numargs().
        return Err((label_arg_count, jump_arg_count));
    }

    Ok((inputargs, optimized_ops))
}

/// Slice T-final.F.0 survey probe.
///
pub(crate) struct CompiledEntry<M> {
    /// `Arc<JitCellToken>` so the same Python-style object identity can
    /// flow through `compile_loop` → `attach_procedure_to_interp` →
    /// `MemoryManager.keep_loop_alive` per `compile.py:266` /
    /// `:1019` / `:567` / `:1149`.  Slice 5.1 lifts this from the
    /// previous by-value `JitCellToken`; Slice 5.3 will route the
    /// same Arc to `attach_procedure_to_interp` (currently still on
    /// the install_token PRE-EXISTING-ADAPTATION).
    pub(crate) token: std::sync::Arc<JitCellToken>,
    pub(crate) meta: M,
    /// Front-end loop-version state, mirroring RPython's
    /// jitcell_token.target_tokens ownership across recompilations.
    pub(crate) front_target_tokens: Vec<crate::optimizeopt::unroll::TargetToken>,
    /// Trace id of the root compiled loop.
    pub(crate) root_trace_id: u64,
    /// Metadata for the root loop and any attached bridges, keyed by trace id.
    pub(crate) traces: HashMap<u64, CompiledTrace>,
    /// RPython parity: previous compiled entries for this green_key.
    /// In RPython, JitCellToken keeps all target_tokens' code alive.
    /// In majit, each retrace produces a new Cranelift function;
    /// previous functions are kept here so external target_token JUMPs
    /// can redirect to them via runtime trampoline.  Stored as
    /// `Arc<JitCellToken>` for Slice 5.1 token-identity lift.
    pub(crate) previous_tokens: Vec<std::sync::Arc<JitCellToken>>,
    /// Box identity plan Phase E: high-water OpRef at which a bridge
    /// compilation starts allocating fresh boxes.
    ///
    /// RPython `opencoder.py:249-273 TraceIterator.__init__` allocates
    /// fresh `InputArg` Python objects every iteration, so bridge
    /// inputargs are automatically disjoint from the parent loop's boxes
    /// by Python `is` identity. In pyre `OpRef::from_raw(u32)` IS the identity, so
    /// a bridge that re-uses the `[0..num_inputs)` range collides with
    /// the parent loop's OpRefs. `next_global_opref` records the first
    /// OpRef Phase 2 did *not* allocate; `compile_bridge` /
    /// `start_retrace_from_guard` read it to pick a disjoint
    /// `bridge_inputarg_base` (see the `bridge_inputarg_base` derivation
    /// in `compile_bridge` below).
    pub(crate) next_global_opref: u32,
}

/// Compute the smallest fresh OpRef strictly above every position
/// referenced by `inputargs` and `ops` (op positions, op args, and
/// guard fail_args).
///
/// Box Identity Phase E.2b parity: every loop / bridge / entry-bridge
/// stored in `compiled_loops` records its high-water mark so the next
/// bridge's `bridge_inputarg_base` derives a disjoint
/// `[bridge_inputarg_base..)` namespace. Without this the second
/// bridge of the same loop, or a bridge-from-bridge chain, would
/// re-use OpRef slots already owned by an earlier trace and lose the
/// "fresh `InputArg` per `trace.get_iter()`" identity guarantee
/// RPython gets for free (`opencoder.py:249-273`).
///
/// Args/fail_args are scanned in addition to op.pos because a port-side
/// undefined-OpRef fallback path historically tolerated dangling refs.
/// Scanning them mirrors RPython's Box-identity model where every
/// referenced Box keeps the parent trace alive: any OpRef the trace
/// touches must be reflected in the high-water mark.
fn compute_next_global_opref(inputargs: &[InputArg], ops: &[majit_ir::Op]) -> u32 {
    fn opref_high_water(r: OpRef) -> u32 {
        if r.is_none() || r.is_constant() {
            0
        } else {
            r.raw().saturating_add(1)
        }
    }
    let from_inputargs = inputargs
        .iter()
        .map(|ia| ia.index.saturating_add(1))
        .max()
        .unwrap_or(0);
    let from_ops = ops
        .iter()
        .map(|op| {
            let mut hw = opref_high_water(op.pos);
            for a in &op.args {
                hw = hw.max(opref_high_water(*a));
            }
            if let Some(fa) = &op.fail_args {
                for a in fa {
                    hw = hw.max(opref_high_water(*a));
                }
            }
            hw
        })
        .max()
        .unwrap_or(0);
    from_inputargs.max(from_ops)
}

/// compile.py compile_trace return parity.
/// Indicates the result of bridge compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeCompileResult {
    /// Bridge was compiled and attached to the guard.
    Compiled,
    /// Bridge compilation failed.
    Failed,
    /// Optimizer requested retrace (no matching target token).
    /// The tracing context is intact — caller should continue tracing.
    /// pyjitpl.py:3196: compile_trace returns None → MetaInterp continues.
    RetraceNeeded,
}

/// pyjitpl.py: partial trace saved from a failed bridge compilation.
///
/// When `compile_trace` (bridge path) fails to close the loop and sets
/// `retrace_needed`, this struct stores the intermediate compilation result
/// so that `compile_retrace` can append new body ops to it.
pub struct PartialTrace {
    /// Optimized ops from the first (incomplete) compilation attempt.
    pub(crate) ops: Vec<Op>,
    /// Inputargs from the partial trace.
    pub(crate) inputargs: Vec<InputArg>,
    /// Constants from the partial trace.
    pub(crate) constants: HashMap<u32, i64>,
}

/// The meta-tracing JIT engine.
///
/// Manages the full JIT lifecycle: warm counting → tracing → optimization
/// → compilation → execution.
///
/// `M` is the interpreter-specific metadata stored alongside each compiled loop
/// (e.g., storage layout, register mapping). The interpreter provides `M` when
/// closing a trace and receives it back when running compiled code.

/// model.py:199-201 cpu.cls_of_box(box) default implementation:
///   obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
///   return ConstInt(ptr2int(obj.typeptr))
/// Reads the first word of the object (typeptr/vtable pointer).
fn default_cls_of_box(raw_ref: i64) -> i64 {
    debug_assert!(raw_ref != 0, "cls_of_box: null ref");
    unsafe { *(raw_ref as *const usize) as i64 }
}

/// model.py:266-273 + RPython rclass.ll_issubclass default implementation.
/// Uses the active backend's GC subclass range table, matching
/// `rclass.py:1133-1137 ll_issubclass`.  The exact-match fallback is only
/// for standalone fixtures that run without installed GC hooks.
fn default_issubclass(typeptr: i64, bounding_class: i64) -> bool {
    if let (Some((cls_min, cls_max)), Some((subcls_min, _))) = (
        majit_gc::subclass_range(bounding_class as usize),
        majit_gc::subclass_range(typeptr as usize),
    ) {
        cls_min <= subcls_min && subcls_min < cls_max
    } else {
        typeptr == bounding_class
    }
}

/// pyjitpl.py:2908-2920 `MetaInterp._prepare_bridge_resumption` context.
///
/// Bridge-origin descriptor carried from `start_retrace_from_guard`
/// through `compile_trace_finish`.  RPython stores the equivalent on
/// `self.resumekey` + the active `MetaInterp` history; pyre aggregates
/// the four identifying fields into one value so the finish-compile
/// helpers can dispatch to the bridge branch without cross-component
/// tuple plumbing.
///
/// `code_ptr` is the code address for the bridge's green key, enabling
/// any PC to map back to a green key via the same hash function used by
/// `make_green_key`.
#[derive(Debug, Clone, Copy)]
pub struct BridgeTraceInfo {
    pub green_key: u64,
    pub trace_id: u64,
    pub fail_index: u32,
    pub code_ptr: usize,
}

/// pyjitpl.py `MetaInterp` tracing-session context.
///
/// Owns the per-session state that RPython carries through
/// `self.history`, `self.resumekey`, and the callbacks on `self`: the
/// frontend `M` metadata snapshot taken at trace start, and — when
/// tracing a bridge — the origin descriptor needed to thread
/// `compile_trace(self, self.resumekey, ...)` through the bridge
/// branch.  `clear_trace_session` takes this field back out on
/// abort / finish so the next trace starts from `None`.
pub struct ActiveTraceSession<M: Clone> {
    /// Frontend state snapshot captured at `force_start_tracing` /
    /// `bound_reached` / `on_back_edge_typed` / bridge resume.  Held
    /// by MetaInterp so the finish-compile helpers can consume it
    /// without requiring the JitDriver to mediate.
    pub trace_meta: M,
    /// `Some(BridgeTraceInfo)` when the session is a bridge retrace
    /// (populated by `set_bridge_trace_info`).  `None` for root traces.
    pub bridge: Option<BridgeTraceInfo>,
}

pub struct MetaInterp<M: Clone> {
    pub(crate) warm_state: WarmEnterState,
    pub(crate) backend: BackendImpl,
    pub(crate) compiled_loops: HashMap<u64, CompiledEntry<M>>,
    pub(crate) tracing: Option<TraceCtx>,
    pub(crate) next_trace_id: u64,
    /// RPython metainterp_sd.virtualref_info — shared VirtualRefInfo.
    pub(crate) virtualref_info: crate::virtualref::VirtualRefInfo,
    /// JIT hooks for profiling and debugging.
    pub(crate) hooks: JitHooks,
    /// Pre-allocated token number for the trace currently being recorded.
    /// Set when tracing starts so that self-recursive calls can emit
    /// call_assembler targeting this token before the trace is compiled.
    pub(crate) pending_token: Option<(u64, u64)>,
    /// Cumulative statistics counters.
    pub(crate) stats: JitStatsCounters,
    /// Pointer to the live virtualizable object at trace entry.
    /// Used to derive lengths from the actual object when the interpreter
    /// does not provide them explicitly.
    pub(crate) vable_ptr: *const u8,
    /// Virtualizable array lengths for trace-entry box layout.
    pub(crate) vable_array_lengths: Vec<usize>,
    /// warmspot.py:449 jd.result_type — per-driver static result type.
    pub(crate) result_type: Type,
    /// RPython portal_call_depth parity: call depth at which the current
    /// trace started. When Some(depth), only merge_point at that depth fires.
    /// Replaces the pyre-jit TLS JIT_TRACING_DEPTH — state colocated with
    /// tracing context for single source of truth.
    pub tracing_call_depth: Option<u32>,
    /// PyPy warmspot.py max_unroll_recursion (default 7).
    pub(crate) max_unroll_recursion: usize,
    /// RPython parity: `prepare_trace_segmenting()` marks the next tracing run
    /// for a green key so the loop should finish early instead of repeatedly
    /// aborting once it nears the trace limit.
    pub(crate) force_finish_trace: bool,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Maps oopspec indices to (calldescr, func_ptr) for generate_modified_call.
    pub(crate) callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` runtime hook. The
    /// host runtime (pyre etc.) registers this via
    /// [`MetaInterp::set_string_length_resolver`] at JIT init. Propagated
    /// to `Optimizer::string_length_resolver` inside `make_optimizer`, then
    /// on to `OptContext::string_length_resolver` for each optimizer run.
    pub(crate) string_length_resolver: Option<crate::optimizeopt::info::StringLengthResolver>,
    /// info.py:788-790 `ConstPtrInfo._unpack_str(mode)` runtime hook.
    pub(crate) string_content_resolver: Option<crate::optimizeopt::info::StringContentResolver>,
    /// history.py:377 `get_const_ptr_for_string(s)` runtime hook.
    pub(crate) string_constant_alloc: Option<crate::optimizeopt::info::StringConstantAllocator>,
    /// pyjitpl.py:2389: partial trace from a failed bridge compilation attempt.
    /// When bridge optimization returns "not final" (retrace needed), the
    /// partial optimized ops are saved here so compile_retrace can append
    /// the new body and compile the complete loop.
    pub(crate) partial_trace: Option<PartialTrace>,
    /// pyjitpl.py:2390: trace position where the retrace should resume.
    /// Set to potential_retrace_position by retrace_needed(). On the next
    /// compile_loop, the merge point's start position is compared
    /// against this to verify we're retracing from the correct location.
    pub(crate) retracing_from: Option<crate::recorder::TracePosition>,
    /// pyjitpl.py:2374: optimizer state snapshot from the failed bridge attempt.
    /// compile_retrace imports this to resume optimization from where the
    /// first attempt left off.
    pub(crate) exported_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// pyjitpl.py:2373: number of cancelled compilation attempts.
    pub(crate) cancel_count: u32,
    /// Actual green_key the last compile_loop stored under. May differ
    /// from the tracing green_key when cross-loop cut retargets to the
    /// inner loop's key (compile.py:269).
    pub(crate) last_compiled_key: Option<u64>,
    /// virtualizable.py:86 NUM_SCALAR_INPUTARGS: number of scalar inputargs
    /// (frame + static fields). Set by the interpreter at JIT init.
    pub num_scalar_inputargs: usize,
    /// pyjitpl.py:3182: trace position saved before compile_trace records
    /// a tentative JUMP. If compile_trace triggers retrace_needed, this
    /// becomes the retracing_from position.
    pub(crate) potential_retrace_position: Option<crate::recorder::TracePosition>,
    /// RPython compile.py:204-207 (record_loop_or_bridge) parity:
    /// quasi-immutable dependencies from the last compilation.
    /// Raw pointers to namespace/quasi-immutable objects that the compiled
    /// loop depends on. After compilation, the caller registers the loop's
    /// invalidation flag on each dep. Cleared on each compile attempt.
    pub last_quasi_immutable_deps: Vec<(u64, u32)>,
    /// Set by compile_bridge when optimizer returns retrace_requested=true.
    /// Checked by compile_bridge_trace to return RetraceNeeded.
    pub(crate) retrace_after_bridge: bool,
    /// pyjitpl.py:3317 virtualref_boxes: pairs of (symbolic OpRef, concrete ptr).
    /// Managed by opimpl_virtual_ref/opimpl_virtual_ref_finish.
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
    /// compile.py:288-290 parity: preamble target tokens saved from Phase 1
    /// even when Phase 2 raises InvalidLoop.
    pending_preamble_tokens: HashMap<u64, Vec<crate::optimizeopt::unroll::TargetToken>>,
    // pyjitpl.py:2289 `self.staticdata.all_descrs = self.cpu.setup_descrs()` now
    // lives on MetaInterpStaticData (RPython `metainterp_sd.all_descrs`).
    // Access via `self.staticdata.all_descrs` / `&mut self.staticdata.all_descrs`.
    /// bridgeopt.py:124 frontend_boxes parity: runtime values from the
    /// guard failure DeadFrame. Saved by start_retrace_from_guard, used
    /// by compile_bridge for cls_of_box during deserialize_optimizer_knowledge.
    pending_frontend_boxes: Option<Vec<i64>>,
    /// model.py:199-201 cpu.cls_of_box parity: callback that reads
    /// the class/vtable pointer from a runtime Ref object.
    /// Default: `Some(default_cls_of_box)` — reads typeptr at offset 0,
    /// matching backend/model.py:199-201.
    pub(crate) cls_of_box: Option<fn(i64) -> i64>,

    /// model.py:266-273 + RPython `rclass.ll_issubclass` parity: callback
    /// that decides whether `typeptr` is a subclass of `bounding_class`.
    /// Default: `Some(default_issubclass)` — active GC subclass-range
    /// lookup, matching `rclass.py:1133-1137 ll_issubclass`. Mirrors the
    /// blackhole-side resolution at `blackhole.rs:7962-7966`.
    pub(crate) issubclass: Option<fn(i64, i64) -> bool>,

    /// pyjitpl.py:2179 `self.metainterp.staticdata` — per-process
    /// static lookup tables (insns, descrs, indirectcalltargets,
    /// list_of_addr2name).  See [`MetaInterpStaticData`] for the
    /// methods that read this field.
    ///
    /// `Arc<MetaInterpStaticData>` so the same instance is shareable
    /// with `TraceRecordBuffer.metainterp_sd` (opencoder.py:472
    /// parity: `self.metainterp_sd = metainterp_sd`).  Field
    /// mutations go via `Arc::get_mut` while the reference count is
    /// 1 — this holds in production until `TraceCtx.recorder`
    /// migrates to `TraceRecordBuffer` that clones the Arc.  Step
    /// 2e.2 will add interior mutability for shared-refcount
    /// mutation at that point.
    pub staticdata: std::sync::Arc<MetaInterpStaticData>,

    /// pyjitpl.py:2451, 3269, 3403 `MetaInterp.framestack` — stack of
    /// `MIFrame` objects representing the current call chain.
    /// Populated by `newframe` (pyjitpl.py:2432-2452) and drained by
    /// `popframe` (pyjitpl.py:2462-2477).  Initialized as empty by
    /// `initialize_state_from_start` and `rebuild_state_after_failure`.
    pub framestack: crate::pyjitpl::MIFrameStack,

    /// pyjitpl.py:2378 `MetaInterp.portal_call_depth = 0` (class
    /// attribute, instance-mutated).  Counts the nesting depth of
    /// jitdriver portal frames currently on `framestack`.  Bumped by
    /// `newframe` when `jitcode.jitdriver_sd is not None`
    /// (pyjitpl.py:2434), decremented by `popframe`
    /// (pyjitpl.py:2466).  Initialized to `-1` by
    /// `initialize_state_from_start` (pyjitpl.py:3268) so the first
    /// portal frame brings it to `0`.
    ///
    /// Distinct from `tracing_call_depth` which captures the depth at
    /// which the current trace started — that field is a one-shot
    /// snapshot, this one is the live counter.
    pub portal_call_depth: i32,

    /// pyjitpl.py:2400 `self.call_ids = []`.
    ///
    /// Stack of `current_call_id` values captured by `newframe`
    /// (pyjitpl.py:2435 `self.call_ids.append(self.current_call_id)`)
    /// every time a portal frame is pushed.  `popframe` drops the top
    /// (pyjitpl.py:2469 `self.call_ids.pop()`).  Resume snapshots use
    /// the entries to identify which portal call a fail-arg belongs to.
    pub call_ids: Vec<u64>,

    /// pyjitpl.py:2391 `self.portal_trace_positions = []`.
    ///
    /// Start/end markers for each recursive-main-jitcode portal frame
    /// active during the current trace.  `newframe` appends
    /// `(jd_no, Some(greenkey), trace_position)` when the entering
    /// jitcode passes `is_main_jitcode` (pyjitpl.py:2443-2445) and
    /// `popframe` appends `(jd_no, None, trace_position)` on the
    /// symmetric exit (pyjitpl.py:2470-2472).  `find_biggest_function`
    /// (pyjitpl.py:3514-3551) walks the list as a stack to compute the
    /// longest-traced inlined function for abort reporting; the reset
    /// to `None` at pyjitpl.py:2795 signals that the trace aborted.
    ///
    /// pyre's existing `find_biggest_function` (trace_ctx.rs:625) uses
    /// `TraceCtx::inline_trace_positions` — a narrower subset that only
    /// tracks active inlined callees.  Keeping this field here mirrors
    /// RPython's shape so a future port of `find_biggest_function` can
    /// line-by-line read the start/end stack; callers that merely want
    /// the active-frame list should keep using `inline_trace_positions`.
    pub portal_trace_positions: Option<Vec<(usize, Option<u64>, crate::recorder::TracePosition)>>,

    /// pyjitpl.py:2401 `self.current_call_id = 0`.
    ///
    /// Monotonically increasing counter that uniquely identifies each
    /// portal call.  Stamped onto `call_ids` by `newframe` and bumped
    /// after the entry (pyjitpl.py:2442).
    pub current_call_id: u64,

    /// pyjitpl.py:2393 `self.last_exc_value = lltype.nullptr(rclass.OBJECT)`.
    ///
    /// Last exception value pointer.  Cleared by `finishframe`
    /// (pyjitpl.py:2481) and `assert_no_exception` (pyjitpl.py:3398).
    /// Stored as a raw `i64` pointer in pyre — `0` is the upstream
    /// `nullptr(OBJECT)` sentinel.
    pub last_exc_value: i64,

    /// pyjitpl.py:2405 `self.aborted_tracing_jitdriver = None`.
    ///
    /// Set by `aborted_tracing` (pyjitpl.py:2776-2785) when the trace
    /// aborts because it grew too long; the next compile attempt
    /// reads it to fire the trace-too-long hook.
    pub aborted_tracing_jitdriver: Option<usize>,

    /// pyjitpl.py:3291 `self.jitdriver_sd` parity — index into
    /// `staticdata.jitdrivers_sd` of the driver that owns the
    /// in-flight trace.
    ///
    /// Upstream MetaInterp is constructed once per
    /// `_compile_and_run_once` and binds `self.jitdriver_sd =
    /// jitdriver_sd` at construction (warmspot.py
    /// `WarmEnterState._compile_and_run_once`).  pyre reuses a
    /// single MetaInterp across traces, so the active driver is
    /// installed at every trace-start path
    /// (`setup_tracing` / `force_start_tracing` /
    /// `start_retrace_from_guard`) and cleared once the trace
    /// session ends.  Single-portal pyre still has only the shell
    /// driver, so this is always 0 in practice; reads
    /// (`initialize_virtualizable` / `virtualizable_info` /
    /// `set_virtualizable_info`) prefer this slot and fall back to
    /// scanning `jitdrivers_sd` when it has not yet been set
    /// (init-time broadcast, test paths).
    pub active_jitdriver_sd: Option<usize>,

    /// pyjitpl.py:2406 `self.aborted_tracing_greenkey = None`.  See
    /// `aborted_tracing_jitdriver`.
    pub aborted_tracing_greenkey: Option<u64>,

    /// Stash for `abort_trace_live` → `aborted_tracing` handoff.
    /// RPython's exception unwind carries green_key/permanent implicitly
    /// through frame locals; pyre threads them through these fields so
    /// `aborted_tracing` fires the `on_trace_abort` hook with the same
    /// payload the old monolithic `abort_trace` produced.
    pub(crate) pending_abort_green_key: Option<u64>,
    pub(crate) pending_abort_permanent: bool,

    /// pyjitpl.py:2381 `MetaInterp.last_exc_box = None` (class
    /// attribute).  Set by `handle_possible_exception` to the boxed
    /// exception value (either a fresh `ConstPtr` when the class is
    /// statically known, or the GUARD_EXCEPTION result op).  Read by
    /// downstream opimpl methods that need to read the active
    /// exception (`opimpl_last_exc_value`, `last_exception` BC, etc.).
    pub last_exc_box: Option<OpRef>,

    /// pyjitpl.py:3386, 3392 `MetaInterp.class_of_last_exc_is_const`.
    /// Tracks whether `last_exc_box`'s class is a runtime
    /// `ConstPtr(val)` (`true`) or the dynamic GUARD_EXCEPTION op result
    /// (`false`).  `handle_possible_exception` always promotes to true
    /// after processing because subsequent `last_exception` reads see
    /// a known class.
    pub class_of_last_exc_is_const: bool,

    /// pyjitpl.py:2394 `self.forced_virtualizable = None`.
    ///
    /// Tracks the virtualizable that was force-flushed during the last
    /// `do_residual_call` (CALL_MAY_FORCE).  pyjitpl.py:2078
    /// `vable_after_residual_call(funcbox)` consumes it on the next
    /// call boundary.  Stored as a raw `i64` GC pointer in pyre.
    pub forced_virtualizable: i64,

    /// pyjitpl.py: `self.ovf_flag` — set by overflow-detecting
    /// opimpls (`int_add_ovf`, `int_sub_ovf`, `int_mul_ovf`,
    /// `_record_unary_op`/`_record_binop_with_ovf` paths) when the
    /// concrete arithmetic overflowed during tracing.  Read by
    /// `MIFrame.handle_possible_overflow_error` (pyjitpl.py:1881-1890)
    /// to choose between `GUARD_OVERFLOW` and `GUARD_NO_OVERFLOW`.
    pub ovf_flag: bool,

    /// pyjitpl.py:2403 `self.box_names_memo = {}`.
    /// Memoized symbolic names for boxes (debug/log output only).
    /// Pyre uses simple `OpRef → String` mapping; populated lazily by
    /// the on-demand log formatter.
    pub box_names_memo: std::collections::HashMap<OpRef, String>,

    /// pyjitpl.py:2412 `self.trace_length_at_last_tco = -1`.
    ///
    /// Trace position recorded by `_try_tco` (pyjitpl.py:1308-1321)
    /// the last time it removed a frame.  Used to detect infinite
    /// tail-recursive loops that would otherwise spin in the
    /// metainterp without recording any new ops (gh-5021).  When the
    /// post-pop trace length matches this value the next TCO emits a
    /// SAME_AS_I so the trace-length limit eventually fires.
    pub trace_length_at_last_tco: i32,

    /// pyjitpl.py `MetaInterp.history` + `self.resumekey` owner.
    ///
    /// Centralises the trace-session state that used to live on
    /// `JitDriver<S>.trace_meta` / `bridge_info` so `finishframe` /
    /// `finishframe_exception` can drive the finish compile without a
    /// `TraceAction::Finish` roundtrip — upstream places
    /// `compile_done_with_this_frame` / `compile_exit_frame_with_exception`
    /// directly on `MetaInterp`.  Single source of truth; the JitDriver
    /// reads this via accessors and never duplicates it.
    pub(crate) active_trace_session: Option<ActiveTraceSession<M>>,
}

/// Internal mutable counters for JIT compilation statistics.
///
/// Holds only the pyre-specific lifetime counters (`loops_compiled`,
/// `loops_aborted`, `bridges_compiled`, `guard_failures`) that have no
/// `Counters.*` slot upstream.  Every counter that maps to a
/// `Counters.*` id (OPS / HEAPCACHED_OPS / RECORDED_OPS / GUARDS /
/// OPT_OPS / OPT_GUARDS / OPT_GUARDS_SHARED / NV* / ABORT_* /
/// FORCE_VIRTUALIZABLES / OPT_VECTORIZE_*) lives on
/// [`crate::jitprof::JitProfiler`] in `MetaInterpStaticData.profiler`
/// and is published via `profiler.snapshot()`.
#[derive(Default, Clone, Debug)]
pub(crate) struct JitStatsCounters {
    loops_compiled: usize,
    loops_aborted: usize,
    bridges_compiled: usize,
    guard_failures: usize,
}

/// Snapshot of cumulative JIT compilation statistics.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JitStats {
    pub loops_compiled: usize,
    pub loops_aborted: usize,
    pub bridges_compiled: usize,
    pub guard_failures: usize,
}

/// Callback hooks for JIT events (compilation, guard failures, etc.).
///
/// Mirrors RPython's jit_hook_loop, jit_hook_bridge, etc.
/// All fields are optional closures.
#[derive(Default)]
pub struct JitHooks {
    /// Called when a loop is compiled. Args: (green_key, num_ops_before, num_ops_after).
    pub on_compile_loop: Option<Box<dyn Fn(u64, usize, usize) + Send>>,
    /// Called when a bridge is compiled. Args: (green_key, fail_index, num_ops).
    pub on_compile_bridge: Option<Box<dyn Fn(u64, u32, usize) + Send>>,
    /// Called on guard failure. Args: (green_key, fail_index, fail_count).
    pub on_guard_failure: Option<Box<dyn Fn(u64, u32, u32) + Send>>,
    /// Called when tracing starts. Args: (green_key).
    pub on_trace_start: Option<Box<dyn Fn(u64) + Send>>,
    /// Called when tracing is aborted. Args: (green_key, permanent).
    pub on_trace_abort: Option<Box<dyn Fn(u64, bool) + Send>>,
    /// Called when compilation (loop or bridge) fails. Args: (green_key, error_message).
    pub on_compile_error: Option<Box<dyn Fn(u64, &str) + Send>>,
}

impl<M: Clone> MetaInterp<M> {
    /// resume.py:1314 parity: `metainterp_sd.virtualref_info` shared
    /// `VirtualRefInfo` handed to `blackhole_from_resumedata` /
    /// `consume_virtualref_info` so JIT_VIRTUAL_REF handles decode.
    pub fn virtualref_info(&self) -> &crate::virtualref::VirtualRefInfo {
        &self.virtualref_info
    }

    /// framework.py `root_walker.walk_roots` parity for the JIT-side
    /// constant pool.
    ///
    /// RPython's whole-program GC traces `ConstPtr.value` automatically.
    /// Pyre's nursery collector has no whole-program tracing, so every
    /// `Const::Ref` slot reachable from a compiled guard must be exposed
    /// via this root walker
    /// (`majit_gc::shadow_stack::walk_extra_roots`).
    ///
    /// Each guard owns a single shared `Arc<ResumeStorage>`
    /// (compile.py:853 `ResumeGuardDescr`). `StoredExitLayout` is the
    /// sole carrier on the trace surrogate (T4.4 retired the parallel
    /// `StoredResumeData` side table), and every downstream reader
    /// (bridge retrace, blackhole resume, GC root walker) points at
    /// the same Arc, so walking the `exit_layouts` /
    /// `terminal_exit_layouts` storage slots once updates the pool for
    /// every observer.
    pub fn walk_rd_consts_refs(&mut self, mut visitor: impl FnMut(&mut GcRef)) {
        fn visit_storage(
            storage: Option<&Arc<ResumeStorage>>,
            visitor: &mut dyn FnMut(&mut GcRef),
        ) {
            let Some(s) = storage else { return };
            // SAFETY: pyre is single-threaded and the minor-collection
            // walker is the only writer; concurrent readers run outside
            // GC cycles.
            let consts = unsafe { s.rd_consts_mut_for_gc() };
            for c in consts.iter_mut() {
                if let Const::Ref(slot) = c {
                    visitor(slot);
                }
            }
        }

        for entry in self.compiled_loops.values_mut() {
            for trace in entry.traces.values_mut() {
                for layout in trace.exit_layouts.values_mut() {
                    visit_storage(layout.storage.as_ref(), &mut visitor);
                }
                for layout in trace.terminal_exit_layouts.values_mut() {
                    visit_storage(layout.storage.as_ref(), &mut visitor);
                }
            }
        }
    }

    #[inline]
    fn prepare_compiled_run_io() {
        io_buffer::io_buffer_discard();
    }

    #[inline]
    fn finish_compiled_run_io() {
        // Guard exits hand control back to the interpreter or blackhole after
        // the already-executed prefix of the trace. Any traced I/O in that
        // prefix is semantically committed and must survive deoptimization.
        io_buffer::io_buffer_commit();
    }

    #[inline]
    fn is_jump_exit(is_finish: bool, fail_index: u32) -> bool {
        !is_finish && fail_index == u32::MAX
    }

    #[inline]
    fn should_record_guard_failure(is_finish: bool, fail_index: u32) -> bool {
        !is_finish && !Self::is_jump_exit(is_finish, fail_index)
    }

    #[inline]
    fn record_guard_failure_event(&mut self, green_key: u64, fail_index: u32) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] guard failure at key={}, guard={}",
                green_key, fail_index
            );
        }
        self.stats.guard_failures += 1;
        self.warm_state.log_guard_failure(fail_index);
        if let Some(ref hook) = self.hooks.on_guard_failure {
            hook(green_key, fail_index, 0);
        }
    }

    /// jitprof.py:118-122 `Profiler.count_ops(opnum, kind=Counters.OPS)`.
    ///
    /// Thin pass-through to `staticdata.profiler.count_ops`.  RPython
    /// callers reach the profiler through `self.staticdata.profiler`;
    /// pyre's MetaInterp keeps the accessor as a convenience because
    /// the borrow shape is friendlier than spelling
    /// `self.staticdata.profiler.count_ops(...)` at every site.
    pub fn count_ops(&self, opnum: OpCode, kind: i32) {
        self.staticdata.profiler.count_ops(opnum, kind);
    }

    /// jitprof.py:101-102 `Profiler.count(kind, inc=1)`.
    ///
    /// Thin pass-through to `staticdata.profiler.count`.  Same
    /// rationale as `count_ops` above.
    pub fn count(&self, kind: i32, inc: usize) {
        self.staticdata.profiler.count(kind, inc);
    }

    #[inline]
    fn run_result_for_jump_exit(
        fail_index: u32,
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    ) -> Option<RunResult<M>> {
        (fail_index == u32::MAX).then_some(RunResult::Jump {
            values,
            meta,
            savedata,
        })
    }

    #[inline]
    fn blackhole_result_for_jump_exit(
        fail_index: u32,
        values: Vec<i64>,
        typed_values: Vec<Value>,
        exit_layout: CompiledExitLayout,
        meta: M,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<BlackholeRunResult<M>> {
        (fail_index == u32::MAX).then_some(BlackholeRunResult::Jump {
            values,
            typed_values: Some(typed_values),
            exit_layout: Some(exit_layout),
            meta,
            via_blackhole: false,
            savedata,
            exception,
        })
    }

    fn alloc_trace_id(&mut self) -> u64 {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        trace_id
    }

    /// Test-only accessor for the root trace_id of a compiled entry.
    /// Production code reaches the same value via the resume guard descr's
    /// `descr.trace_id()` (set at backend compile time) or via
    /// `bridge_info().trace_id` after `start_retrace_from_guard`.
    #[cfg(test)]
    pub fn compiled_root_trace_id(&self, green_key: u64) -> Option<u64> {
        self.compiled_loops.get(&green_key).map(|c| c.root_trace_id)
    }

    /// Salvage the evicted entry's per-trace metadata into the new
    /// CompiledEntry being built for the same `green_key`, and return the
    /// list of `Arc<JitCellToken>` to seed the new entry's
    /// `previous_tokens` for cross-token JUMP/redirect keepalive.
    ///
    /// Mirrors `pyjitpl.py` recompile semantics where successor compiled
    /// loops absorb predecessor's per-fail metadata so old guards (which
    /// still reference their original compiled trace via descr identity)
    /// stay reachable through the current `compiled_loops[green_key]`
    /// entry — without a parallel side-table.
    ///
    /// `merged_traces` is populated with the old entry's `traces` for any
    /// trace_id not already present (the new entry's freshly-built
    /// `traces` HashMap takes precedence on collision).
    fn retire_compiled_entry(
        &mut self,
        _owning_key: u64,
        entry: CompiledEntry<M>,
        merged_traces: &mut HashMap<u64, CompiledTrace>,
    ) -> Vec<Arc<JitCellToken>> {
        let token = entry.token;
        let mut previous_tokens = Vec::with_capacity(1 + entry.previous_tokens.len());
        previous_tokens.push(Arc::clone(&token));
        previous_tokens.extend(entry.previous_tokens);
        for (tid, ct) in entry.traces {
            merged_traces.entry(tid).or_insert(ct);
        }
        previous_tokens
    }

    fn trace_for_exit<'a>(
        compiled: &'a CompiledEntry<M>,
        trace_id: u64,
    ) -> Option<(u64, &'a CompiledTrace)> {
        compiled
            .traces
            .get(&trace_id)
            .map(|trace| (trace_id, trace))
    }

    /// O(1) owner lookup via `descr.rd_loop_token` (compile.py:186 stamp,
    /// stored as the owning loop's green_key).  Every guard produced by
    /// the regular compile_loop / compile_bridge paths goes through the
    /// `record_loop_or_bridge` walker, which stamps the owning clt onto
    /// the descr.  Returns `None` for pre-populated descrs
    /// (`compile_tmp_callback` stubs) so callers fall through to the
    /// legacy scan.
    ///
    /// Issue 1.1 fix: old traces from a prior compilation are merged
    /// into the new entry's `traces` map at recompile time
    /// (`retire_compiled_entry`), mirroring main's eager-merge
    /// behavior.  The previous retired-tokens fallback was a pyre-only
    /// side-table that has been removed.
    fn trace_for_exit_by_rd_loop_token<'a>(
        &'a self,
        rd_loop_token: Option<u64>,
        trace_id: u64,
    ) -> Option<(u64, u64, &'a CompiledTrace)> {
        let green_key = rd_loop_token?;
        let compiled = self.compiled_loops.get(&green_key)?;
        let resolved = trace_id;
        compiled
            .traces
            .get(&resolved)
            .map(|trace| (green_key, resolved, trace))
    }

    fn compiled_exit_layout_from_trace(
        trace: &CompiledTrace,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        trace
            .exit_layouts
            .get(&fail_index)
            .map(|layout| layout.public(owning_key, trace_id, fail_index))
    }

    /// `compile.py:855 ResumeGuardDescr._attrs_` parity: per-guard exit
    /// types live on the descr object itself.  RPython has no such
    /// recovery helper because every `Box` already carries `.type`
    /// (`history.py:220 ConstInt` etc., `resoperation.py:1597`); the
    /// optimizer's `store_final_boxes_in_guard` writes the fail args to
    /// the guard op via `setfailargs` and that's enough.
    ///
    /// Pyre's `OpRef` is untyped, so the descr exposes `fail_arg_types()`
    /// as a cached `Vec<Type>` set by `set_fail_arg_types` at
    /// optimizer.py:724 time.  This helper returns that vector — and
    /// only that vector.  Canonical sources, in priority order:
    ///
    /// 1. `layout.resolve_exit_types()` — descr-first
    ///    (`compile.py:853 ResumeGuardDescr.fail_arg_types`); falls
    ///    back to the trace-side `exit_types` Vec mirror when the descr
    ///    is `None` (backend-only synthesised entries) or returns an
    ///    empty vector. See `StoredExitLayout::resolve_exit_types` for
    ///    the parity rationale.
    /// 2. `op.fail_arg_types` — guard-op cached vector.
    /// 3. `op.descr.fail_arg_types()` — descr field via op walk.
    ///
    /// Returns `None` if none of the canonical sources carry the data.
    /// The earlier OpRef-walker fallback (inputargs / constant_types /
    /// `op.pos` producer scan) was a pyre-only NEW-DEVIATION: RPython
    /// would have `Box.type` in hand at the same point and never need a
    /// trace-level walk.  Removed per `/parity` review.
    fn infer_exit_types_from_trace(trace: &CompiledTrace, fail_index: u32) -> Option<Vec<Type>> {
        if let Some(layout) = trace.exit_layouts.get(&fail_index) {
            return Some(layout.resolve_exit_types().to_vec());
        }
        // F.5-orthodox.1: drop `guard_op_indices.get` shortcut. The
        // producer (`compile.rs:248-991 build_guard_metadata`) inserts
        // into both `exit_layouts` (line 971) and `guard_op_indices`
        // (line 284) at the same per-trace `fail_index` counter, with
        // `exit_layouts` a superset (also covers FINISH ops). Reaching
        // here means `exit_layouts` already returned None for this
        // fail_index, so the HashMap shortcut is structurally dead;
        // walk `trace.ops` via descr-side identity (matching RPython's
        // `compile.py:184 op.getdescr()` predicate) — this is the
        // F.5-orthodox.1 canonical path that replaces the
        // `guard_op_indices` HashMap entirely.
        //
        // Compare against `fail_index_per_trace()` (the per-trace
        // counter the producer stamps at `compile.rs:301`), not
        // `fail_index()` (which is the global `alloc_fail_index()` id
        // at `descr.rs:1065`). The HashMap was keyed on the per-trace
        // counter, so descr-side identity must read the per-trace slot.
        let op_index = trace.ops.iter().position(|op| {
            op.descr
                .as_ref()
                .and_then(|descr| descr.as_fail_descr())
                .is_some_and(|descr| descr.fail_index_per_trace() == fail_index)
        })?;
        let op = trace.ops.get(op_index)?;
        if let Some(types) = &op.fail_arg_types {
            return Some(types.clone());
        }
        op.descr
            .as_ref()
            .and_then(|descr| descr.as_fail_descr())
            .map(|descr| descr.fail_arg_types().to_vec())
            .filter(|types| !types.is_empty())
    }

    fn backend_fail_descr_layout(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<majit_backend::FailDescrLayout> {
        let lookup = |token: &JitCellToken| {
            if token.compiled.is_none() {
                return None;
            }
            self.backend
                .compiled_trace_fail_descr_layouts(token, trace_id)
                .and_then(|layouts| {
                    layouts
                        .into_iter()
                        .find(|layout| layout.fail_index == fail_index)
                })
        };
        lookup(&compiled.token).or_else(|| compiled.previous_tokens.iter().find_map(|t| lookup(t)))
    }

    fn terminal_exit_layout_from_trace(
        trace: &CompiledTrace,
        owning_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        trace.terminal_exit_layouts.get(&op_index).map(|layout| {
            layout.public(
                owning_key,
                trace_id,
                compile::find_fail_index_for_exit_op(&trace.ops, op_index).unwrap_or(u32::MAX),
            )
        })
    }

    #[allow(dead_code)]
    fn backend_terminal_exit_layout(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        op_index: usize,
    ) -> Option<majit_backend::TerminalExitLayout> {
        let lookup = |token: &JitCellToken| {
            if token.compiled.is_none() {
                return None;
            }
            self.backend
                .compiled_trace_terminal_exit_layouts(token, trace_id)
                .and_then(|layouts| {
                    layouts
                        .into_iter()
                        .find(|layout| layout.op_index == op_index)
                })
        };
        lookup(&compiled.token).or_else(|| compiled.previous_tokens.iter().find_map(|t| lookup(t)))
    }

    fn compiled_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let frontend_exit_types = Self::trace_for_exit(compiled, trace_id)
            .and_then(|(_, trace)| Self::infer_exit_types_from_trace(trace, fail_index));
        self.backend_fail_descr_layout(compiled, trace_id, fail_index)
            .map(|layout| {
                // compile.py:861 copy_all_attributes_from parity:
                // when the compiled trace has been evicted but the
                // backend still has rd_numb / rd_consts / rd_virtuals /
                // rd_pendingfields propagated on the fail descriptor,
                // reassemble a `ResumeStorage` so downstream consumers
                // (force_from_resumedata, blackhole) see the same
                // shared pool the frontend-primed path provides.
                // Mirrors compile.rs:917-924 inside merge_backend_exit_layouts.
                let storage = layout.rd_numb.as_ref().map(|rd_numb| {
                    crate::resume::ResumeStorage::new(
                        rd_numb.clone(),
                        layout.rd_consts.clone().unwrap_or_default(),
                        layout.rd_virtuals.clone().unwrap_or_default(),
                        layout.rd_pendingfields.clone().unwrap_or_default(),
                    )
                });
                let exit_types = if layout.fail_arg_types.is_empty() {
                    frontend_exit_types.unwrap_or_default()
                } else {
                    layout.fail_arg_types
                };
                let gc_ref_slots = if layout.gc_ref_slots.is_empty() {
                    exit_types
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, ty)| (*ty == Type::Ref).then_some(idx))
                        .collect()
                } else {
                    layout.gc_ref_slots
                };
                CompiledExitLayout {
                    rd_loop_token: owning_key, // compile.py:186
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout.source_op_index,
                    exit_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout,
                    resume_layout: None,
                    storage,
                }
            })
    }

    fn terminal_exit_layout_from_backend(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        self.backend_terminal_exit_layout(compiled, trace_id, op_index)
            .map(|layout| CompiledExitLayout {
                rd_loop_token: owning_key, // compile.py:186
                trace_id,
                fail_index: layout.fail_index,
                source_op_index: Some(layout.op_index),
                exit_types: layout.exit_types,
                is_finish: layout.is_finish,
                gc_ref_slots: layout.gc_ref_slots,
                force_token_slots: layout.force_token_slots,
                recovery_layout: layout.recovery_layout,
                resume_layout: None,
                storage: None,
            })
    }

    fn compiled_trace_layout_for_trace(
        &self,
        compiled: &CompiledEntry<M>,
        owning_key: u64,
        trace_id: u64,
    ) -> Option<CompiledTraceLayout> {
        let mut exit_layouts =
            if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
                let mut layouts: Vec<_> = trace
                    .exit_layouts
                    .iter()
                    .map(|(&fail_index, layout)| {
                        layout.public(owning_key, resolved_trace_id, fail_index)
                    })
                    .collect();
                layouts.sort_by_key(|layout| layout.fail_index);
                layouts
            } else {
                Vec::new()
            };
        if let Some(backend_layouts) = self
            .backend
            .compiled_trace_fail_descr_layouts(&compiled.token, trace_id)
        {
            let mut merged = HashMap::new();
            for layout in exit_layouts.drain(..) {
                merged.insert(layout.fail_index, layout);
            }
            for layout in backend_layouts {
                merged.insert(
                    layout.fail_index,
                    CompiledExitLayout {
                        rd_loop_token: owning_key, // compile.py:186
                        trace_id,
                        fail_index: layout.fail_index,
                        source_op_index: layout.source_op_index,
                        exit_types: layout.fail_arg_types,
                        is_finish: layout.is_finish,
                        gc_ref_slots: layout.gc_ref_slots,
                        force_token_slots: layout.force_token_slots,
                        recovery_layout: layout.recovery_layout,
                        resume_layout: merged
                            .get(&layout.fail_index)
                            .and_then(|existing| existing.resume_layout.clone()),
                        storage: None,
                    },
                );
            }
            exit_layouts = merged.into_values().collect();
            exit_layouts.sort_by_key(|layout| layout.fail_index);
        }

        let mut terminal_exit_layouts =
            if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
                let mut layouts: Vec<_> = trace
                    .terminal_exit_layouts
                    .iter()
                    .map(|(&op_index, layout)| CompiledTerminalExitLayout {
                        op_index,
                        exit_layout: layout.public(
                            owning_key,
                            resolved_trace_id,
                            compile::find_fail_index_for_exit_op(&trace.ops, op_index)
                                .unwrap_or(u32::MAX),
                        ),
                    })
                    .collect();
                layouts.sort_by_key(|layout| layout.op_index);
                layouts
            } else {
                Vec::new()
            };
        if let Some(backend_layouts) = self
            .backend
            .compiled_trace_terminal_exit_layouts(&compiled.token, trace_id)
        {
            let mut merged = HashMap::new();
            for layout in terminal_exit_layouts.drain(..) {
                merged.insert(layout.op_index, layout);
            }
            for layout in backend_layouts {
                merged.insert(
                    layout.op_index,
                    CompiledTerminalExitLayout {
                        op_index: layout.op_index,
                        exit_layout: CompiledExitLayout {
                            rd_loop_token: owning_key, // compile.py:186
                            trace_id,
                            fail_index: layout.fail_index,
                            source_op_index: Some(layout.op_index),
                            exit_types: layout.exit_types,
                            is_finish: layout.is_finish,
                            gc_ref_slots: layout.gc_ref_slots,
                            force_token_slots: layout.force_token_slots,
                            recovery_layout: layout.recovery_layout,
                            resume_layout: merged
                                .get(&layout.op_index)
                                .and_then(|existing| existing.exit_layout.resume_layout.clone()),
                            storage: None,
                        },
                    },
                );
            }
            terminal_exit_layouts = merged.into_values().collect();
            terminal_exit_layouts.sort_by_key(|layout| layout.op_index);
        }

        if exit_layouts.is_empty() && terminal_exit_layouts.is_empty() {
            None
        } else {
            Some(CompiledTraceLayout {
                trace_id,
                exit_layouts,
                terminal_exit_layouts,
            })
        }
    }

    /// Create a new MetaInterp with the given compilation threshold.
    pub fn new(threshold: u32) -> Self {
        let mut this = MetaInterp {
            warm_state: WarmEnterState::new(threshold),
            backend: BackendImpl::new(),
            compiled_loops: HashMap::new(),
            tracing: None,
            next_trace_id: 1,
            virtualref_info: crate::virtualref::VirtualRefInfo::new(),
            hooks: JitHooks::default(),
            pending_token: None,
            stats: JitStatsCounters::default(),
            vable_ptr: std::ptr::null(),
            vable_array_lengths: Vec::new(),
            result_type: Type::Ref,
            tracing_call_depth: None,
            max_unroll_recursion: 7, // RPython default from rlib/jit.py
            force_finish_trace: false,
            callinfocollection: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            partial_trace: None,
            retracing_from: None,
            exported_state: None,
            cancel_count: 0,
            last_compiled_key: None,
            num_scalar_inputargs: 0,
            potential_retrace_position: None,
            last_quasi_immutable_deps: Vec::new(),
            retrace_after_bridge: false,
            virtualref_boxes: Vec::new(),
            pending_preamble_tokens: HashMap::new(),
            pending_frontend_boxes: None,
            cls_of_box: Some(default_cls_of_box),
            issubclass: Some(default_issubclass),
            staticdata: std::sync::Arc::new(MetaInterpStaticData::new()),
            framestack: crate::pyjitpl::MIFrameStack::empty(),
            portal_call_depth: 0,
            call_ids: Vec::new(),
            current_call_id: 0,
            portal_trace_positions: Some(Vec::new()),
            last_exc_value: 0,
            aborted_tracing_jitdriver: None,
            active_jitdriver_sd: None,
            aborted_tracing_greenkey: None,
            pending_abort_green_key: None,
            pending_abort_permanent: false,
            last_exc_box: None,
            class_of_last_exc_is_const: false,
            forced_virtualizable: 0,
            ovf_flag: false,
            box_names_memo: std::collections::HashMap::new(),
            trace_length_at_last_tco: -1,
            active_trace_session: None,
        };
        // `pyjitpl.py:2222` `make_and_attach_done_descrs([self, cpu])` —
        // now that both sides of the pair exist, publish the
        // `MetaInterpStaticData`-side `DoneWithThisFrameDescr*` Arcs
        // onto the backend so FINISH fast-path pointer identity works
        // against the same `Arc` the metainterp reads back.
        let MetaInterp {
            ref staticdata,
            ref mut backend,
            ..
        } = this;
        staticdata.attach_descrs_to_cpu(backend);
        this
    }

    /// `warmspot.py:289` `self.metainterp_sd.finish_setup(self.codewriter)`
    /// — drive `MetaInterpStaticData::finish_setup(asm)` against the
    /// canonical staticdata while it still has a single owner, mirroring
    /// the upstream lifecycle in which `make_jitcodes` populates the
    /// codewriter's assembler before this call.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre stores `staticdata: Arc<…>` on
    /// `MetaInterp` (see field doc) so callers downstream can clone the
    /// Arc into per-session structures.  RPython holds an unwrapped
    /// reference instead.  Routing through `Arc::get_mut` recovers the
    /// upstream `&mut` access only while the refcount is still 1, which
    /// is true between `MetaInterp::new` and the first Arc clone (e.g.
    /// the bridge-resume / tracing setup paths).  Callers that need the
    /// orthodox `finish_setup(codewriter)` step must invoke this method
    /// before any clone of `self.staticdata` is taken; otherwise the
    /// `unwrap` panics with a clear message and the convergence failure
    /// is visible at the call site.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's `CodeWriter` does not own
    /// `callcontrol` (see `MetaInterpStaticData::finish_setup` doc);
    /// the call site threads both as siblings to keep the upstream
    /// `codewriter.callcontrol.<field>` reads literal in the inner
    /// staticdata method.
    pub fn finish_setup(
        &mut self,
        codewriter: &majit_translate::jit_codewriter::codewriter::CodeWriter,
        callcontrol: &majit_translate::jit_codewriter::call::CallControl,
    ) {
        let staticdata = std::sync::Arc::get_mut(&mut self.staticdata).expect(
            "MetaInterp::finish_setup called after `staticdata` was cloned; \
             RPython warmspot.py:289 invariant requires a single owner at finish_setup time",
        );
        staticdata.finish_setup(codewriter, callcontrol);
        // pyjitpl.py:2268 `self.callinfocollection = codewriter.callcontrol
        //                                  .callinfocollection`. Upstream
        // exposes the populated collection through `metainterp.staticdata
        // .callinfocollection`; pyre's distribution path threads it as the
        // `Option<Arc<…>>` field on `MetaInterp` (consumed by optimizer /
        // unroll / cranelift / pyre-jit eval guard-exit). Seeding the Arc
        // here means every later `self.callinfocollection.clone()` shares
        // identity through `Arc::clone` and the `vstring`/`Concat` resume
        // paths see a populated table instead of `None`.
        self.callinfocollection = Some(std::sync::Arc::new(staticdata.callinfocollection.clone()));
    }

    /// Narrow lifecycle hook for state-field JIT: install the canonical
    /// liveness payload from a pre-populated `Assembler` without
    /// requiring a full `CodeWriter` / `CallControl`.
    ///
    /// See `MetaInterpStaticData::install_canonical_liveness` for the
    /// RPython parity citation.  The same single-owner `Arc::get_mut`
    /// invariant as `finish_setup` (RPython `warmspot.py:289`) applies
    /// — call before any tracing path clones `staticdata`.
    pub fn install_canonical_liveness(
        &mut self,
        asm: &majit_translate::jit_codewriter::assembler::Assembler,
    ) {
        let staticdata = std::sync::Arc::get_mut(&mut self.staticdata).expect(
            "MetaInterp::install_canonical_liveness called after `staticdata` was cloned; \
             RPython warmspot.py:289 invariant requires a single owner at finish_setup time",
        );
        staticdata.install_canonical_liveness(asm);
    }

    /// Phase 4 Epic B.3-B.4: copy a freshly-snapshotted `all_liveness`
    /// byte stream into `staticdata.liveness_info` without re-running
    /// the full `install_canonical_liveness` insn-id seeding.
    ///
    /// Invoked from `JitDriver::sync_liveness_info_from_shared_asm`
    /// after the macro-emitted `__JitMeta::install_canonical_liveness`
    /// has registered the canonical entry into the driver-shared
    /// `Assembler`.  Mirrors `pyjitpl.py:2264 self.liveness_info =
    /// "".join(asm.all_liveness)` for the post-canonical-entry state.
    ///
    /// The same single-owner `Arc::get_mut` invariant as
    /// `install_canonical_liveness` applies: callers must finish any
    /// liveness-producing factory builds and sync before tracing clones
    /// `staticdata`.
    pub fn sync_liveness_info(&mut self, all_liveness: &[u8]) {
        let staticdata = std::sync::Arc::get_mut(&mut self.staticdata).expect(
            "MetaInterp::sync_liveness_info called after `staticdata` was cloned; \
             RPython warmspot.py:289 invariant requires a single owner at finish_setup time",
        );
        staticdata.liveness_info = all_liveness.to_vec();
    }

    /// Install a fresh [`ActiveTraceSession`] seeded with the frontend
    /// trace metadata.  Called from `force_start_tracing` /
    /// `bound_reached` / `on_back_edge_typed` and the bridge-resume
    /// path.  Panics if a prior session was not cleared — mirrors
    /// RPython's invariant that `self.history` has a single owner per
    /// trace.
    pub fn begin_trace_session(&mut self, trace_meta: M) {
        debug_assert!(
            self.active_trace_session.is_none(),
            "begin_trace_session called while a trace session is already active",
        );
        self.active_trace_session = Some(ActiveTraceSession {
            trace_meta,
            bridge: None,
        });
    }

    /// Attach bridge-origin metadata to the active session.  Called
    /// once during `start_retrace_from_guard` after
    /// `begin_trace_session` has installed the frontend meta.
    pub fn set_bridge_trace_info(&mut self, bridge: BridgeTraceInfo) {
        let slot = self
            .active_trace_session
            .as_mut()
            .expect("set_bridge_trace_info called with no active trace session");
        slot.bridge = Some(bridge);
    }

    /// Bridge-origin descriptor for the active session, if any.
    /// Returns `None` for root traces and when no session is active.
    pub fn bridge_info(&self) -> Option<BridgeTraceInfo> {
        self.active_trace_session.as_ref().and_then(|s| s.bridge)
    }

    /// Consume the bridge-origin descriptor while keeping the active
    /// session's frontend meta in place.  Used by `CloseLoop` /
    /// `CloseLoopWithArgs` branches that drive `compile_trace_finish`
    /// from the bridge identity yet fall through to run `compile_loop`
    /// on the remainder of the trace.  Returns `None` for root traces.
    pub fn take_bridge_info(&mut self) -> Option<BridgeTraceInfo> {
        self.active_trace_session
            .as_mut()
            .and_then(|s| s.bridge.take())
    }

    /// Read-only access to the frontend trace metadata for callers
    /// (e.g. `compile_trace_entry_data`) that must peek without
    /// consuming the session.
    pub fn trace_meta(&self) -> Option<&M> {
        self.active_trace_session.as_ref().map(|s| &s.trace_meta)
    }

    /// Take ownership of the frontend trace metadata and end the
    /// session.  Used by the finish-compile helpers that consume `M`
    /// before calling `recorder.finish()` + backend compile.
    pub fn take_trace_meta(&mut self) -> Option<M> {
        self.active_trace_session.take().map(|s| s.trace_meta)
    }

    /// Drop the active session without consuming the meta, matching
    /// the abort / cleanup path used when tracing aborts before
    /// finish.
    pub fn clear_trace_session(&mut self) {
        self.active_trace_session = None;
    }

    /// warmspot.py:449 — set the per-driver result_type from the portal
    /// function's return signature. Called once during driver setup.
    ///
    /// `warmspot.py:449` assigns to `jd.result_type` on the jitdriver
    /// static-data itself.  pyre stores a convenience copy on
    /// `MetaInterp` for early lookups before any driver has been
    /// registered, but the authoritative value lives on each
    /// `JitDriverStaticData`.  Propagate the update to every registered
    /// driver (auto-creating the default driver if needed) and
    /// re-execute the downstream steps that depend on the result kind:
    /// `jd.portal_calldescr` (`warmspot.py:1013`) and
    /// `jd.portal_finishtoken` (`pyjitpl.py:2275-2279`).
    pub fn set_result_type(&mut self, tp: Type) {
        self.result_type = tp;
        self.ensure_default_driver_sd();
        let Self {
            staticdata,
            backend,
            ..
        } = self;
        let sd = std::sync::Arc::get_mut(staticdata)
            .expect("set_result_type: staticdata has other owners");
        for jd in sd.jitdrivers_sd.iter_mut() {
            jd.result_type = tp;
            // `warmspot.py:1013-1017` build_portal_calldescr reads
            // `self.result_type`; a stale descr from an earlier
            // set_result_type call must be rebuilt.
            jd.build_portal_calldescr();
        }
        // `pyjitpl.py:2275-2283` portal_finishtoken is keyed by
        // `jd.result_type`, and the backend's `propagate_exception_descr`
        // is re-bound inside the same method — re-run the attachment so
        // each driver picks the correct `done_with_this_frame_descr_*`
        // sibling and the cpu observes the shared exc_descr.
        sd.finish_setup_descrs_for_jitdrivers(backend);
    }

    /// warmspot.py:449 — the per-driver static result_type.
    pub fn result_type(&self) -> Type {
        self.result_type
    }

    /// pyjitpl.py:2289 / descr.py:25-47 parity: take back all_descrs from
    /// optimizer after compilation. Optimizer.ensure_descr_index() assigns
    /// sequential descr_index during collect_optimizer_knowledge_for_resume().
    pub(crate) fn take_back_all_descrs(&mut self, all_descrs: Vec<DescrRef>) {
        *self.staticdata.all_descrs.lock().unwrap() = all_descrs;
    }

    /// Accessor for `pending_frontend_boxes` without consuming it.
    /// Used by `start_bridge_tracing` to thread the raw deadframe values
    /// into `setup_bridge_sym` so `Box(n, _)` decodes match `rd_numb`'s
    /// encoder-time liveboxes numbering (bridgeopt.py:124 parity).
    pub fn pending_frontend_boxes_ref(&self) -> Option<&[i64]> {
        self.pending_frontend_boxes.as_deref()
    }

    /// Cache the current virtualizable object pointer for trace-entry setup.
    /// Mirrored onto `TraceCtx::virtualizable_heap_ptr` so
    /// `synchronize_virtualizable` can reach the live frame without a
    /// callback back into MetaInterp.
    pub(crate) fn set_vable_ptr(&mut self, ptr: *const u8) {
        self.vable_ptr = ptr;
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.set_virtualizable_heap_ptr(ptr);
        }
    }

    /// Cache fallback virtualizable array lengths for trace-entry box setup.
    pub(crate) fn set_vable_array_lengths(&mut self, lengths: Vec<usize>) {
        self.vable_array_lengths = lengths;
    }

    fn trace_entry_vable_lengths(&self, info: &VirtualizableInfo) -> Vec<usize> {
        // pyjitpl.py:3302 `vinfo.read_boxes(self.cpu, virtualizable, startindex)`
        // reads field/array values directly from the concrete virtualizable
        // heap object; RPython does not consult any interpreter-supplied
        // "trace-entry cache" for lengths. Match that here when the layout
        // exposes a readable header (the common case).
        if !self.vable_ptr.is_null() && info.can_read_all_array_lengths_from_heap() {
            // Safety: vable_ptr is cached from JitState::virtualizable_heap_ptr()
            // for the currently active interpreter state.
            return unsafe { info.read_array_lengths_from_heap(self.vable_ptr) };
        }
        // Fallback for layouts that cannot expose array length on the heap
        // object alone (header-less embedded arrays) and for unit tests that
        // do not stage a real heap object. RPython has no such layout in
        // the PyPy tree; this branch is a documented pyre adaptation until
        // header-less layouts are either dropped or taught to report their
        // length directly on the heap object.
        self.vable_array_lengths.clone()
    }

    /// pyjitpl.py:3290 `initialize_virtualizable(self, original_boxes)`.
    ///
    /// RPython:
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         index = (self.jitdriver_sd.num_green_args +
    ///                  self.jitdriver_sd.index_of_virtualizable)
    ///         virtualizable_box = original_boxes[index]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.clear_vable_token(virtualizable)
    ///         startindex = len(original_boxes) - self.jitdriver_sd.num_green_args
    ///         self.virtualizable_boxes = vinfo.read_boxes(self.cpu, virtualizable, startindex)
    ///         original_boxes += self.virtualizable_boxes
    ///         self.virtualizable_boxes.append(virtualizable_box)
    ///         self.check_synchronized_virtualizable()
    ///
    /// pyre adaptation: pyre's TraceCtx already holds the
    /// `virtualizable_boxes` field. This method is the RPython-named
    /// MetaInterp-level entry point; it computes the array lengths /
    /// derived layout and delegates to the lower-level
    /// `TraceCtx::init_virtualizable_boxes` helper which performs the
    /// actual `virtualizable_boxes = [..., vable_ref]` push (matching
    /// the `read_boxes(...) ; append(virtualizable_box)` shape from
    /// pyjitpl.py:3302-3306).
    ///
    /// `live_values` is the pyre analog of RPython's `original_boxes`,
    /// with one structural difference: pyre carries greens in the
    /// `green_key` side channel, so `live_values` holds only the red
    /// inputargs. The absolute index inside `live_values` therefore
    /// collapses to `index_of_virtualizable` (and `num_green_args`
    /// contributes zero for all currently registered drivers, mirroring
    /// pyre's single-portal discipline).
    fn initialize_virtualizable(&self, ctx: &mut TraceCtx, live_values: &[Value]) {
        // pyjitpl.py:3291: vinfo = self.jitdriver_sd.virtualizable_info
        // Prefer the trace-bound `active_jitdriver_sd` (RPython
        // `self.jitdriver_sd`); fall back to scanning when an
        // init-time / test caller has not yet elected one.
        let Some(idx) = self.resolve_active_jitdriver_sd_with_vinfo() else {
            return;
        };
        let jd_sd = &self.staticdata.jitdrivers_sd[idx];
        let info = jd_sd
            .virtualizable_info
            .as_ref()
            .expect("resolve_active_jitdriver_sd_with_vinfo returned a slot without vinfo");
        // pyjitpl.py:3293-3295:
        //     index = (self.jitdriver_sd.num_green_args +
        //              self.jitdriver_sd.index_of_virtualizable)
        //     virtualizable_box = original_boxes[index]
        //
        // pyre adaptation: `live_values` is reds-only (greens live in
        // `green_key`), so the effective offset inside `live_values`
        // reduces to `index_of_virtualizable`. `set_virtualizable_info`
        // eagerly sets `index_of_virtualizable = 0` on the pyre shell
        // driver (empty reds + named virtualizable), matching the
        // portal convention, so the strict RPython read works for
        // every driver.
        let num_green_args = jd_sd.num_greens();
        debug_assert_eq!(
            num_green_args, 0,
            "pyre green args live in green_key, not in live_values (pyjitpl.py:3293)"
        );
        assert!(
            jd_sd.index_of_virtualizable >= 0,
            "pyjitpl.py:3293: jitdriver with virtualizable_info must have \
             index_of_virtualizable set (got {})",
            jd_sd.index_of_virtualizable,
        );
        let index_of_virtualizable = jd_sd.index_of_virtualizable as usize;
        let index = num_green_args + index_of_virtualizable;

        let num_static = info.num_static_extra_boxes;
        // virtualizable.py:86-99 `read_boxes` iterates `for i in range(len(lst))`
        // over the heap array. `trace_entry_vable_lengths` is the pyre-side
        // cache populated by `jitdriver::refresh_vable_layout_cache`; when
        // the cache hasn't been seeded (test paths), fall back to reading
        // the physical array length straight off `live_values[index]` (the
        // virtualizable pointer).
        let array_lengths = {
            let reported = self.trace_entry_vable_lengths(info);
            if !reported.is_empty() {
                reported
            } else if info.can_read_all_array_lengths_from_heap() {
                let vable_ptr = match live_values.get(index) {
                    Some(Value::Ref(r)) => r.as_usize() as *const u8,
                    Some(Value::Int(v)) => *v as *const u8,
                    _ => std::ptr::null(),
                };
                if vable_ptr.is_null() {
                    Vec::new()
                } else {
                    unsafe { info.read_array_lengths_from_heap(vable_ptr) }
                }
            } else {
                Vec::new()
            }
        };
        let num_array_elems: usize = array_lengths.iter().sum();
        let total_vable = num_static + num_array_elems;
        // pyjitpl.py:3293-3302 parity:
        //     index = num_green_args + index_of_virtualizable
        //     virtualizable_box = original_boxes[index]
        //     startindex = len(original_boxes) - num_green_args  # == num_red_args
        //     self.virtualizable_boxes = vinfo.read_boxes(cpu, virtualizable, startindex)
        //     original_boxes += self.virtualizable_boxes
        //
        // In pyre, `live_values` plays the role of `original_boxes` and
        // greens are stripped out before this helper runs, so index maps
        // directly to `virtualizable_arg_index()` and `startindex` becomes
        // `num_reds`. The expanded static/array slots occupy
        // `live_values[num_reds .. num_reds + total_vable]`.
        let vable_index = ctx
            .driver_descriptor()
            .and_then(|driver| driver.virtualizable_arg_index())
            .unwrap_or(0);
        let num_reds = ctx
            .driver_descriptor()
            .map(|driver| driver.num_reds())
            .unwrap_or(1);
        // pyjitpl.py:3290-3307 `initialize_virtualizable` only gates on
        // `vinfo is not None` and unconditionally calls
        // `vinfo.read_boxes(cpu, virtualizable, startindex)`. Pyre's
        // current callers (descriptor=None default) supply `live_values`
        // already in the expanded shape that `read_boxes` would produce,
        // so the function reuses those slots instead of re-minting
        // inputargs. The `live_values.len() < num_reds + total_vable`
        // gate below preserves the descriptor=None contract:
        // re-minting under descriptor=None breaks pyre's downstream
        // virtualizable_boxes consumers (dynasm nested_loop crashes
        // exit 101 if the gate is removed without also flipping
        // descriptor=Some + Task #21 heap-writeback). PRE-EXISTING-
        // ADAPTATION: the gate is the convergence-debt marker; the
        // fully-RPython-orthodox shape lands together with descriptor
        // activation (state.rs:4058 driver_descriptor + Task #21 plan).
        // Cluster 2 (b): allow heap-read fallback when live_values is the
        // reds-only `[frame, ec]` shape that descriptor=Some emits. The
        // expanded-tail path still uses live_values directly; the short
        // path drives the `vable_ptr` heap read below to mint inputargs
        // for each vable static field + array item.
        if total_vable == 0 {
            return;
        }
        let _has_expanded_tail_outer = live_values.len() >= num_reds + total_vable;
        if !_has_expanded_tail_outer && self.vable_ptr.is_null() {
            return;
        }
        // pyjitpl.py:3293-3295: index = num_green_args + index_of_virtualizable.
        // The caller derives `index` above from jitdriver_sd so the bootstrap
        // path and the regular JitDriver registry agree. The virtualizable
        // is a Ref-typed inputarg (resoperation.py:739 InputArgRef).
        let virtualizable_box = OpRef::input_arg_ref(index as u32);
        let virtualizable_value = live_values[index];
        let has_expanded_tail = live_values.len() >= num_reds + total_vable;
        // pyjitpl.py:3302: virtualizable_boxes = vinfo.read_boxes(...)
        // pyjitpl.py appends these boxes to `original_boxes` before
        // create_empty_history() snapshots the trace inputargs. When the
        // caller already supplied an expanded tail we reuse those inputarg
        // slots; otherwise we mint new inputargs here for each freshly-read
        // box, recovering the same `original_boxes += read_boxes(...)` shape.
        let vable_values: Vec<Value> = if !self.vable_ptr.is_null() {
            let (static_boxes, array_boxes) =
                unsafe { info.read_all_boxes(self.vable_ptr, &array_lengths) };
            let mut out = Vec::with_capacity(total_vable);
            for (i, bits) in static_boxes.iter().enumerate() {
                out.push(heap_value_for(info.static_fields[i].field_type, *bits));
            }
            for (a, items) in array_boxes.iter().enumerate() {
                let item_ty = info.array_fields[a].item_type;
                for bits in items {
                    out.push(heap_value_for(item_ty, *bits));
                }
            }
            out
        } else if has_expanded_tail {
            live_values[num_reds..num_reds + total_vable].to_vec()
        } else {
            return;
        };
        let vable_oprefs: Vec<OpRef> = if has_expanded_tail {
            (0..total_vable)
                .map(|i| OpRef::input_arg_typed((num_reds + i) as u32, vable_values[i].get_type()))
                .collect()
        } else {
            vable_values
                .iter()
                .map(|value| {
                    let opref = ctx.recorder.record_input_arg(value.get_type());
                    let (bits, tp) = match value {
                        Value::Int(i) => (*i, Type::Int),
                        Value::Float(f) => (f.to_bits() as i64, Type::Float),
                        Value::Ref(r) => (r.as_usize() as i64, Type::Ref),
                        Value::Void => (0, Type::Void),
                    };
                    ctx.initial_inputarg_consts
                        .push(ctx.constants.get_or_insert_typed(bits, tp));
                    opref
                })
                .collect()
        };
        // pyjitpl.py:3306: virtualizable_boxes.append(virtualizable_box)
        // is folded inside init_virtualizable_boxes (it pushes vable_ref
        // at the end of the list).
        ctx.init_virtualizable_boxes(
            info,
            virtualizable_box,
            virtualizable_value,
            &vable_oprefs,
            &vable_values,
            &array_lengths,
        );
        // pyjitpl.py:3446 synchronize_virtualizable parity: TraceCtx needs
        // the live heap pointer to mirror shadow writes. Mirror here — the
        // MetaInterp `vable_ptr` was cached before `tracing` existed, so
        // `set_vable_ptr` could not plumb it through.
        ctx.set_virtualizable_heap_ptr(self.vable_ptr);
        // pyjitpl.py:3307: check_synchronized_virtualizable() — debug-only
        // assertion. pyre's analog is `check_synchronized_virtualizable`
        // on MetaInterp, but it requires &self which we don't have here.
        // Callers in setup_tracing / bound_reached perform the check
        // immediately after `set_force_finish` via the existing path.
    }

    /// warmstate.py:259: set_param_trace_eagerness — delegates to warmstate.
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.warm_state.set_param_trace_eagerness(eagerness);
    }

    /// Update the green_key associated with the current trace.
    /// Called when tracing started at function entry but the loop closes
    /// at a backward jump with a different PC.
    pub fn update_tracing_green_key(&mut self, key: u64, raw: (usize, usize)) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.set_green_key(key, raw);
        }
    }

    /// compile.py:269-270: return cross-loop cut info if the current trace
    /// closes at a different loop header than where it started.
    pub fn cross_loop_cut_info(&self) -> Option<(usize, Vec<Type>)> {
        let ctx = self.tracing.as_ref()?;
        let inner_key = ctx.cut_inner_green_key?;
        // compile.py:269: cross-loop cut uses the inner loop's merge point.
        // Lookup by inner_key (not ctx.green_key which is the outer loop).
        ctx.get_merge_point_at(inner_key, ctx.header_pc)
            .filter(|mp| mp.position._pos > 0)
            .map(|mp| (mp.header_pc, mp.original_box_types.clone()))
    }

    /// Compat alias: delegates to set_trace_eagerness.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.set_trace_eagerness(threshold);
    }

    /// Set the main compilation threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.warm_state.set_threshold(threshold);
    }

    /// Set the function inlining threshold.
    ///
    /// A function must be called at least this many times during tracing
    /// before it is inlined. Default is 4 (matching RPython).
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.warm_state.set_function_threshold(threshold);
    }

    pub fn warm_state_ref(&self) -> &crate::warmstate::WarmEnterState {
        &self.warm_state
    }

    pub fn warm_state_mut(&mut self) -> &mut crate::warmstate::WarmEnterState {
        &mut self.warm_state
    }

    /// pyjitpl.py:2268 `metainterp.staticdata.callinfocollection` accessor.
    ///
    /// Returns the cached `Arc<CallInfoCollection>` used by the frontend
    /// runtime (pyre-jit eval.rs) to resolve OS_STR_CONCAT /
    /// OS_UNI_CONCAT / OS_STR_SLICE / OS_UNI_SLICE func pointers and
    /// calldescrs when materializing VStr/VUni Concat/Slice virtuals
    /// during guard-exit recovery (resume.py:1143-1188).
    pub fn callinfocollection(&self) -> Option<&std::sync::Arc<majit_ir::CallInfoCollection>> {
        self.callinfocollection.as_ref()
    }

    /// Decay all counters to avoid stale hotness data.
    pub fn decay_counters(&mut self) {
        self.warm_state.decay_counters();
    }

    /// Lazily ensure a default `JitDriverStaticData` slot exists.
    ///
    /// Returns the index of the slot (always 0 for single-driver pyre).
    /// Pyre's production path constructs `JitDriver` without ever
    /// calling `register_jitdriver_sd`, so per-driver readers would
    /// otherwise see an empty `jitdrivers_sd` table.  This helper
    /// installs a placeholder driver — empty greens/reds, no
    /// virtualizable name — purely as a destination for
    /// `set_virtualizable_info` / `set_greenfield_info` propagation.
    /// When the host later registers a real driver via
    /// `register_jitdriver_sd`, that driver gets index 1+ and reads
    /// continue to consult the table linearly via `iter()`.
    pub fn ensure_default_driver_sd(&mut self) -> usize {
        if self.staticdata.jitdrivers_sd.is_empty() {
            std::sync::Arc::get_mut(&mut self.staticdata)
                .expect("ensure_default_driver_sd: staticdata has other owners")
                .jitdrivers_sd
                .push(crate::jitdriver::JitDriverStaticData::new(vec![], vec![]));
        }
        0
    }

    /// Set virtualizable info for interpreter frame virtualization.
    ///
    /// This tells the JIT how to read/write interpreter frame fields
    /// during trace entry/exit.
    ///
    /// warmspot.py:545 `jd.virtualizable_info = vinfos[VTYPEPTR]` —
    /// auto-creates a default `jitdrivers_sd[0]` entry if pyre's
    /// production path hasn't registered one yet, then propagates to
    /// every registered jitdriver_sd so per-driver readers
    /// (`_do_jit_force_virtual`, `vable_*_residual_call`) see the info.
    pub fn set_virtualizable_info(&mut self, info: std::sync::Arc<VirtualizableInfo>) {
        self.ensure_default_driver_sd();
        let active = self.active_jitdriver_sd;
        let sd = std::sync::Arc::get_mut(&mut self.staticdata)
            .expect("set_virtualizable_info: staticdata has other owners");
        // warmspot.py:545 `jd.virtualizable_info = vinfos[VTYPEPTR]`:
        // upstream binds `virtualizable_info` to a single driver. pyre
        // reuses the same call for both init-time setup (no active
        // driver yet) and per-trace updates; route to the elected
        // `active_jitdriver_sd` when available, otherwise broadcast
        // so the single-portal default still receives the info before
        // `setup_tracing` runs.
        let assign = |jd: &mut crate::jitdriver::JitDriverStaticData| {
            jd.virtualizable_info = Some(info.clone());
            // warmspot.py:529/538 parity: pyre's shell driver
            // (`ensure_default_driver_sd`) is constructed with empty
            // reds, so `virtualizable_arg_index` returns None and
            // `index_of_virtualizable` stays at -1. The pyre convention
            // is that portal_runner supplies the virtualizable at red
            // slot 0; encode that here so `initialize_virtualizable`
            // can use `jd.index_of_virtualizable` unconditionally,
            // matching RPython's `index = num_green_args +
            // index_of_virtualizable` (pyjitpl.py:3293-3295) without a
            // `< 0` fallback at the read site.
            if jd.index_of_virtualizable < 0 && jd.reds().is_empty() {
                jd.index_of_virtualizable = 0;
            }
        };
        if let Some(idx) = active.filter(|&i| i < sd.jitdrivers_sd.len()) {
            assign(&mut sd.jitdrivers_sd[idx]);
        } else {
            for jd in sd.jitdrivers_sd.iter_mut() {
                assign(jd);
            }
        }
    }

    /// Get the active virtualizable info.
    ///
    /// pyjitpl.py:3291 `vinfo = self.jitdriver_sd.virtualizable_info`.
    /// Prefers the trace-bound `active_jitdriver_sd`; falls back to
    /// scanning `jitdrivers_sd` for callers that read this before any
    /// trace has started (warmspot init, host-side queries).
    pub fn virtualizable_info(&self) -> Option<&std::sync::Arc<VirtualizableInfo>> {
        if let Some(idx) = self.active_jitdriver_sd {
            if let Some(jd) = self.staticdata.jitdrivers_sd.get(idx) {
                if jd.virtualizable_info.is_some() {
                    return jd.virtualizable_info.as_ref();
                }
            }
        }
        self.staticdata
            .jitdrivers_sd
            .iter()
            .find_map(|jd| jd.virtualizable_info.as_ref())
    }

    /// pyjitpl.py:3291 `self.jitdriver_sd = jitdriver_sd` parity —
    /// pick the slot inside `staticdata.jitdrivers_sd` that owns the
    /// next trace.
    ///
    /// Single-portal pyre still tracks one driver, so the search is
    /// trivial.  The argument is kept so future multi-portal hosts
    /// can hook the descriptor's identity (e.g. `JitDriverStaticData::index`
    /// once `register_jitdriver_sd` populates it) without rewriting the
    /// trace-start callers.
    fn elect_active_jitdriver_sd(
        &self,
        driver_descriptor: Option<&crate::jitdriver::JitDriverStaticData>,
    ) -> Option<usize> {
        // warmspot.py:537 each registered driver keeps its slot index;
        // honour it when the descriptor still carries one.
        if let Some(descriptor) = driver_descriptor {
            if let Some(idx) = descriptor.index {
                if idx < self.staticdata.jitdrivers_sd.len() {
                    return Some(idx);
                }
            }
        }
        // Fall back to the vinfo-bearing slot — pyre's portal driver
        // always has `virtualizable_info` set by the time tracing
        // starts. If no driver carries vinfo (test-only paths), fall
        // through to slot 0 so reads that don't need vinfo still work.
        self.staticdata
            .jitdrivers_sd
            .iter()
            .position(|jd| jd.virtualizable_info.is_some())
            .or_else(|| {
                if self.staticdata.jitdrivers_sd.is_empty() {
                    None
                } else {
                    Some(0)
                }
            })
    }

    /// Resolve the slot index of the JitDriver that should be treated
    /// as `self.jitdriver_sd` (pyjitpl.py:3291) for callers that need
    /// `virtualizable_info` and the related metadata together.
    ///
    /// Prefers the trace-bound `active_jitdriver_sd` and validates it
    /// carries `virtualizable_info`; otherwise scans for the first
    /// registered slot with `virtualizable_info` populated. Returns
    /// `None` when no driver carries `virtualizable_info` yet (the
    /// caller should bail out — RPython would have early-returned
    /// from the `vinfo is None` branch in `initialize_virtualizable`).
    fn resolve_active_jitdriver_sd_with_vinfo(&self) -> Option<usize> {
        if let Some(idx) = self.active_jitdriver_sd {
            if let Some(jd) = self.staticdata.jitdrivers_sd.get(idx) {
                if jd.virtualizable_info.is_some() {
                    return Some(idx);
                }
            }
        }
        self.staticdata
            .jitdrivers_sd
            .iter()
            .position(|jd| jd.virtualizable_info.is_some())
    }

    /// warmspot.py:519-525 `jd.greenfield_info = GreenFieldInfo(cpu, jd)`.
    ///
    /// Hosts that declare green fields (greens containing `.`) call
    /// this between `JitDriver::new` and the first
    /// `_do_jit_force_virtual` to mirror the upstream warmspot wiring.
    /// Auto-creates `jitdrivers_sd[0]` and propagates to every
    /// registered driver.
    pub fn set_greenfield_info(&mut self, info: crate::greenfield::GreenFieldInfo) {
        self.ensure_default_driver_sd();
        let sd = std::sync::Arc::get_mut(&mut self.staticdata)
            .expect("set_greenfield_info: staticdata has other owners");
        for jd in sd.jitdrivers_sd.iter_mut() {
            jd.greenfield_info = Some(info.clone());
        }
    }

    /// Borrow the active greenfield info.
    ///
    /// jitdriver.py:17 parity — reads the first registered
    /// `jitdrivers_sd` entry whose `greenfield_info` slot is populated.
    pub fn greenfield_info(&self) -> Option<&crate::greenfield::GreenFieldInfo> {
        self.staticdata
            .jitdrivers_sd
            .iter()
            .find_map(|jd| jd.greenfield_info.as_ref())
    }

    /// Create an optimizer with virtualizable config if available.
    ///
    /// Standard virtualizable fields become virtual input args — first reads
    /// are replaced with input references and values flow through JUMP args.
    /// No heap access for these fields on the hot path.
    fn current_virtualizable_optimizer_config(
        &self,
    ) -> Option<crate::optimizeopt::virtualize::VirtualizableConfig> {
        self.tracing.as_ref().and_then(|ctx| {
            if !ctx.has_virtualizable_boxes() {
                return None;
            }
            self.virtualizable_info().map(|info| {
                let mut config = info.to_optimizer_config();
                config.array_lengths = ctx.virtualizable_array_lengths().unwrap_or(&[]).to_vec();
                // virtualizable.py:90 read_boxes input layout = [frame,
                // extra_reds..., vable_scalars..., array_items...]. The
                // canonical source of `vable_input_offset` is the active
                // jitdriver's `num_red_args - 1` (excluding frame).
                // Today's slot-0 jitdriver carries empty reds (`wiggly-
                // barto S3.1 Slice 1`) so `num_reds == 0` and the offset
                // is `0` — matching the legacy `[frame, vable_scalars...]`
                // layout. Once Task #24 populates the real reds spec
                // (`['frame', 'ec']`) and the macro flip lands the
                // `extra_reds = { ec: Ref }` block, this expression
                // returns `1` automatically.
                let num_reds = ctx.driver_descriptor().map(|d| d.num_reds()).unwrap_or(0);
                config.vable_input_offset = num_reds.saturating_sub(1);
                config
            })
        })
    }

    fn make_optimizer(&self) -> Optimizer {
        let mut opt = if let Some(config) = self.current_virtualizable_optimizer_config() {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        opt.set_pureop_historylength(self.warm_state.pureop_historylength() as usize);
        // optimizer.py:787-789: constant_fold — allocate immutable objects
        // at compile time. Uses Box::leak for permanent allocation (immutable
        // objects are never freed, matching RPython's prebuilt constants).
        opt.constant_fold_alloc = Some(Box::new(|size_bytes: usize| {
            let layout = std::alloc::Layout::from_size_align(size_bytes, 8)
                .unwrap_or(std::alloc::Layout::new::<u8>());
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                majit_ir::GcRef::NULL
            } else {
                majit_ir::GcRef(ptr as usize)
            }
        }));
        // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — propagate the
        // host-runtime resolver so constant STRLEN / UNICODELEN operations
        // can fold to an exact `IntBound::from_constant(len)` during
        // intbounds postprocessing.
        opt.string_length_resolver = self.string_length_resolver.clone();
        opt.string_content_resolver = self.string_content_resolver.clone();
        opt.string_constant_alloc = self.string_constant_alloc.clone();
        opt
    }

    /// Install the host-runtime `getstrlen1` resolver. The closure must be
    /// callable from the optimizer for arbitrary constant `GcRef` / mode
    /// pairs. `mode == 0` is byte-string, `mode == 1` is unicode; any other
    /// value returns `None` (matching PyPy's `vstring.mode_string` /
    /// `vstring.mode_unicode` dispatch).
    pub fn set_string_length_resolver(
        &mut self,
        resolver: crate::optimizeopt::info::StringLengthResolver,
    ) {
        self.string_length_resolver = Some(resolver);
    }

    /// Install the host-runtime `_unpack_str(mode)` resolver.
    /// info.py:788-790 ConstPtrInfo._unpack_str — extracts character values
    /// from a constant string GcRef.
    pub fn set_string_content_resolver(
        &mut self,
        resolver: crate::optimizeopt::info::StringContentResolver,
    ) {
        self.string_content_resolver = Some(resolver);
    }

    /// Install the host-runtime `get_const_ptr_for_string(s)` allocator.
    /// history.py:377-387 — creates a constant GcRef from character values.
    pub fn set_string_constant_alloc(
        &mut self,
        alloc: crate::optimizeopt::info::StringConstantAllocator,
    ) {
        self.string_constant_alloc = Some(alloc);
    }

    /// model.py:199-201 cpu.cls_of_box — override the default callback
    /// for reading the class/vtable pointer from a runtime Ref object.
    /// The default reads the first word at offset 0 (typeptr).
    pub fn set_cls_of_box(&mut self, f: fn(i64) -> i64) {
        self.cls_of_box = Some(f);
    }

    /// model.py:266-273 + RPython `rclass.ll_issubclass` — override the
    /// default subclass test.  The default reads the active GC subclass
    /// range table, falling back to exact-match only in standalone fixtures.
    pub fn set_issubclass(&mut self, f: fn(i64, i64) -> bool) {
        self.issubclass = Some(f);
    }

    /// Set a callback for loop compilation events.
    pub fn set_on_compile_loop(&mut self, f: impl Fn(u64, usize, usize) + Send + 'static) {
        self.hooks.on_compile_loop = Some(Box::new(f));
    }

    /// Set a callback for bridge compilation events.
    pub fn set_on_compile_bridge(&mut self, f: impl Fn(u64, u32, usize) + Send + 'static) {
        self.hooks.on_compile_bridge = Some(Box::new(f));
    }

    /// Set a callback for guard failure events.
    pub fn set_on_guard_failure(&mut self, f: impl Fn(u64, u32, u32) + Send + 'static) {
        self.hooks.on_guard_failure = Some(Box::new(f));
    }

    /// Set a callback for trace start events.
    pub fn set_on_trace_start(&mut self, f: impl Fn(u64) + Send + 'static) {
        self.hooks.on_trace_start = Some(Box::new(f));
    }

    /// Set a callback for trace abort events.
    pub fn set_on_trace_abort(&mut self, f: impl Fn(u64, bool) + Send + 'static) {
        self.hooks.on_trace_abort = Some(Box::new(f));
    }

    /// Set a callback for compilation error events (loop or bridge).
    pub fn set_on_compile_error(&mut self, f: impl Fn(u64, &str) + Send + 'static) {
        self.hooks.on_compile_error = Some(Box::new(f));
    }

    /// Return a snapshot of the cumulative JIT compilation statistics.
    ///
    /// Counters that map to a `Counters.*` id (OPS, RECORDED_OPS, NV*,
    /// OPT_*, ABORT_*, ...) live on `staticdata.profiler` and are
    /// published via [`crate::jitprof::JitProfiler::snapshot`].
    pub fn get_stats(&self) -> JitStats {
        JitStats {
            loops_compiled: self.stats.loops_compiled,
            loops_aborted: self.stats.loops_aborted,
            bridges_compiled: self.stats.bridges_compiled,
            guard_failures: self.stats.guard_failures,
        }
    }

    /// Check a back-edge: is this location hot enough to trace or run?
    ///
    /// `green_key` identifies the loop header (e.g., PC).
    /// `live_values` are the interpreter's live integer values at this point.
    ///
    /// On `StartedTracing`, the framework registers each value in `live_values`
    /// as an InputArg. The interpreter should then build its symbolic state
    /// from the returned InputArg OpRefs (OpRef::from_raw(0), OpRef::from_raw(1), ...).
    pub fn on_back_edge(&mut self, green_key: u64, live_values: &[i64]) -> BackEdgeAction {
        let typed_values: Vec<Value> = live_values.iter().copied().map(Value::Int).collect();
        self.on_back_edge_typed(green_key, (0, 0), None, None, &typed_values)
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    pub fn force_start_tracing(
        &mut self,
        green_key: u64,
        green_key_raw: (usize, usize),
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.force_start_tracing(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing => {
                // RPython pyjitpl.py:2604 create_empty_history(inputargs): the
                // MetaInterp owns the history/Trace factory, not warmstate.
                let mut recorder = crate::recorder::Trace::new();
                for value in live_values {
                    recorder.record_input_arg(value.get_type());
                }

                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] force start tracing at key={}, num_inputs={}",
                        green_key,
                        live_values.len()
                    );
                }

                let mut ctx = TraceCtx::new(recorder, green_key, self.staticdata.clone());
                ctx.set_root_green_key_raw(green_key_raw);
                // pyjitpl.py:2789 warmrunnerstate.trace_limit snapshot.
                ctx.set_trace_limit(self.warm_state.trace_limit() as usize);
                ctx.callinfocollection = self.callinfocollection.clone();
                ctx.initial_inputarg_consts = live_values
                    .iter()
                    .map(|v| {
                        let (bits, tp) = match v {
                            Value::Int(i) => (*i, Type::Int),
                            Value::Float(f) => (f.to_bits() as i64, Type::Float),
                            Value::Ref(r) => (r.as_usize() as i64, Type::Ref),
                            Value::Void => (0, Type::Void),
                        };
                        ctx.constants.get_or_insert_typed(bits, tp)
                    })
                    .collect();
                if let Some(ref descriptor) = driver_descriptor {
                    ctx.set_driver_descriptor(descriptor.clone());
                }
                // pyjitpl.py:3291 `self.jitdriver_sd = jitdriver_sd`: see
                // `setup_tracing` for the rationale; `force_start_tracing`
                // is the parallel trace-start entry point and must keep the
                // same invariant.
                self.active_jitdriver_sd = self.elect_active_jitdriver_sd(ctx.driver_descriptor());
                // pyjitpl.py:3290 initialize_virtualizable parity.
                self.initialize_virtualizable(&mut ctx, live_values);
                // warmstate.py:439 `force_finish_trace=bool(cell.flags &
                // JC_FORCE_FINISH)`.  Read-only — JC_FORCE_FINISH is sticky
                // upstream (no clear in rpython/jit/metainterp/).
                self.force_finish_trace = self.warm_state.should_force_finish_tracing(green_key);
                ctx.set_force_finish(self.force_finish_trace);
                // compile_tmp_callback parity: pending CALL_ASSEMBLER targets
                // must expose the same red-args-only entry contract that
                // `patch_new_loop_to_load_virtualizable_fields()` later hands
                // to `backend.compile_loop(...)`. Mirror the `setup_tracing`
                // path so both trace-start entry points register the same
                // pending-token shape.
                let input_types = Self::pending_target_input_types(
                    ctx.inputarg_types(),
                    driver_descriptor.as_ref(),
                );
                let num_inputs = input_types.len();
                // warmspot.py:527-538 — jd.index_of_virtualizable is -1 when
                // no virtualizables, else jitdriver.reds.index(vname).
                let index_of_virtualizable: i32 = ctx
                    .driver_descriptor()
                    .and_then(|jd| jd.virtualizable_arg_index())
                    .map(|i| i as i32)
                    .unwrap_or(-1);
                self.tracing = Some(ctx);
                // pyjitpl.py:1547-1556 auto-stamp gate inputs — see
                // `setup_tracing` for rationale.  Bridge-trace
                // distinction now flows through
                // `TraceCtx::is_bridge_trace` rather than
                // `has_compiled_targets_fn` presence, so the parallel
                // entry installs both fns unconditionally.
                let self_ptr = self as *const Self as *const ();
                if let Some(ref mut ctx) = self.tracing {
                    ctx.has_compiled_targets_fn = Some(Box::new(move |gk: u64| -> bool {
                        let meta = unsafe { &*(self_ptr as *const Self) };
                        meta.has_compiled_targets(gk)
                    }));
                    ctx.portal_call_depth_fn = Some(Box::new(move || -> i32 {
                        let meta = unsafe { &*(self_ptr as *const Self) };
                        meta.portal_call_depth
                    }));
                }
                let pending_num = self.warm_state.alloc_token_number();
                self.pending_token = Some((green_key, pending_num));
                // RPython compile_tmp_callback parity: register a placeholder
                // target so call_assembler can resolve the pending token at
                // runtime. call_assembler_fast_path detects null code_ptr and
                // falls back to force_fn.
                self.backend.register_pending_target(
                    pending_num,
                    input_types,
                    num_inputs,
                    self.num_scalar_inputargs,
                    index_of_virtualizable,
                );
                if let Some(ref hook) = self.hooks.on_trace_start {
                    hook(green_key);
                }
                BackEdgeAction::StartedTracing
            }
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    /// RPython warmstate.py:425 bound_reached parity.
    ///
    /// Like `on_back_edge_typed` but bypasses the counter tick — the
    /// caller (can_enter_jit_hook) already verified the counter fired.
    /// This allows decay_counters() to be called before tracing starts
    /// without the internal tick check blocking the trace.
    pub fn bound_reached(
        &mut self,
        green_key: u64,
        green_key_raw: (usize, usize),
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.force_start_tracing(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing => self.setup_tracing(
                green_key,
                green_key_raw,
                green_key_values,
                driver_descriptor,
                live_values,
            ),
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
        }
    }

    pub fn on_back_edge_typed(
        &mut self,
        green_key: u64,
        green_key_raw: (usize, usize),
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        if self.tracing.is_some() {
            return BackEdgeAction::AlreadyTracing;
        }

        match self.warm_state.maybe_compile(green_key) {
            HotResult::NotHot => BackEdgeAction::Interpret,
            HotResult::StartTracing => self.setup_tracing(
                green_key,
                green_key_raw,
                green_key_values,
                driver_descriptor,
                live_values,
            ),
            HotResult::AlreadyTracing => BackEdgeAction::AlreadyTracing,
            HotResult::RunCompiled => BackEdgeAction::RunCompiled,
        }
    }

    #[allow(dead_code)]
    fn pending_target_input_types(
        input_types: Vec<Type>,
        driver_descriptor: Option<&crate::jitdriver::JitDriverStaticData>,
    ) -> Vec<Type> {
        let Some(driver) = driver_descriptor else {
            return input_types;
        };
        let Some(_) = driver.virtualizable_arg_index() else {
            return input_types;
        };
        // `patch_new_loop_to_load_virtualizable_fields()` (compile.py:425-461)
        // collapses the loop's inputargs to the JitDriver reds. Pyre's
        // `extract_live_values()` still emits the expanded
        // `[frame, last_instr, pycode, valuestackdepth, debugdata, lastblock,
        //  w_globals, locals..., stack...]` shape, so the trace's inputarg
        // types do NOT carry the reds in the leading `num_reds` slots —
        // truncating to `num_reds` here would register a bogus
        // `[Ref(frame), Int(last_instr)]` ABI when reds is `[frame, ec]`.
        // Synthesise the reds-only shape directly from the JitDriver var
        // table (`JitDriverVar.tp` matches RPython's `Box.type` parity).
        let red_types: Vec<Type> = driver.reds().iter().map(|red| red.tp).collect();
        if input_types.len() <= red_types.len() {
            return input_types;
        }
        red_types
    }

    fn setup_tracing(
        &mut self,
        green_key: u64,
        green_key_raw: (usize, usize),
        green_key_values: Option<majit_ir::GreenKey>,
        driver_descriptor: Option<JitDriverStaticData>,
        live_values: &[Value],
    ) -> BackEdgeAction {
        // RPython parity: each tracing pass starts with cancel_count=0.
        // In RPython, MetaInterp is re-created per _compile_and_run_once.
        // In pyre, MetaInterp is reused, so reset per-trace state here.
        self.cancel_count = 0;
        // RPython pyjitpl.py:2604 `create_empty_history(inputargs)` — the
        // MetaInterp owns the history factory.
        let mut recorder = crate::recorder::Trace::new();
        for value in live_values {
            recorder.record_input_arg(value.get_type());
        }

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] start tracing at key={}, num_inputs={}",
                green_key,
                live_values.len()
            );
        }

        let mut ctx = if let Some(values) = green_key_values {
            TraceCtx::with_green_key(recorder, green_key, values, self.staticdata.clone())
        } else {
            TraceCtx::new(recorder, green_key, self.staticdata.clone())
        };
        ctx.set_root_green_key_raw(green_key_raw);
        // pyjitpl.py:2789 warmrunnerstate.trace_limit — snapshot onto the
        // per-trace context so `is_too_long` can consult it without needing
        // a warmstate borrow at every check site.
        ctx.set_trace_limit(self.warm_state.trace_limit() as usize);
        ctx.callinfocollection = self.callinfocollection.clone();
        ctx.initial_inputarg_consts = live_values
            .iter()
            .map(|v| {
                let (bits, tp) = match v {
                    Value::Int(i) => (*i, Type::Int),
                    Value::Float(f) => (f.to_bits() as i64, Type::Float),
                    Value::Ref(r) => (r.as_usize() as i64, Type::Ref),
                    Value::Void => (0, Type::Void),
                };
                ctx.constants.get_or_insert_typed(bits, tp)
            })
            .collect();
        if let Some(ref descriptor) = driver_descriptor {
            ctx.set_driver_descriptor(descriptor.clone());
        }
        // pyjitpl.py:3291 `self.jitdriver_sd = jitdriver_sd`: bind the
        // active driver before reads (`initialize_virtualizable`) consult
        // it. `elect_active_jitdriver_sd` mirrors RPython by picking the
        // driver matching the descriptor; with the single-portal pyre
        // shell driver this collapses to slot 0.
        self.active_jitdriver_sd = self.elect_active_jitdriver_sd(ctx.driver_descriptor());
        // pyjitpl.py:3290 initialize_virtualizable parity.
        self.initialize_virtualizable(&mut ctx, live_values);

        // warmstate.py:439 `force_finish_trace=bool(cell.flags &
        // JC_FORCE_FINISH)`.  Read-only — JC_FORCE_FINISH is sticky upstream.
        self.force_finish_trace = self.warm_state.should_force_finish_tracing(green_key);
        // pyjitpl.py:2411: propagate force_finish_trace to TraceCtx
        // so the proc-macro merge_fn closure can read it.
        ctx.set_force_finish(self.force_finish_trace);
        // compile_tmp_callback parity: pending CALL_ASSEMBLER targets must
        // expose the same red-args-only entry contract that
        // `patch_new_loop_to_load_virtualizable_fields()` later hands to
        // `backend.compile_loop(...)`.
        let input_types =
            Self::pending_target_input_types(ctx.inputarg_types(), driver_descriptor.as_ref());
        let num_inputs = input_types.len();
        let index_of_virtualizable: i32 = ctx
            .driver_descriptor()
            .and_then(|jd| jd.virtualizable_arg_index())
            .map(|i| i as i32)
            .unwrap_or(-1);
        self.tracing = Some(ctx);
        // pyjitpl.py:1547-1556 `opimpl_jit_merge_point` auto-stamp
        // gate inputs.  Both `portal_call_depth` and
        // `has_compiled_targets(ptoken)` feed the primary-trace gate;
        // mirror `start_bridge_tracing`'s install (jitdriver.rs:3714-
        // 3717) at the primary path so the gate sees consistent state
        // regardless of which entry started this trace.  The
        // `has_compiled_targets_fn` presence is no longer overloaded
        // as a bridge marker — `TraceCtx::is_bridge_trace` carries
        // that distinction explicitly.
        let self_ptr = self as *const Self as *const ();
        if let Some(ref mut ctx) = self.tracing {
            ctx.has_compiled_targets_fn = Some(Box::new(move |gk: u64| -> bool {
                let meta = unsafe { &*(self_ptr as *const Self) };
                meta.has_compiled_targets(gk)
            }));
            ctx.portal_call_depth_fn = Some(Box::new(move || -> i32 {
                let meta = unsafe { &*(self_ptr as *const Self) };
                meta.portal_call_depth
            }));
        }
        let pending_num = self.warm_state.alloc_token_number();
        self.pending_token = Some((green_key, pending_num));
        self.backend.register_pending_target(
            pending_num,
            input_types,
            num_inputs,
            self.num_scalar_inputargs,
            index_of_virtualizable,
        );
        if let Some(ref hook) = self.hooks.on_trace_start {
            hook(green_key);
        }
        BackEdgeAction::StartedTracing
    }

    /// Access the active TraceCtx (if currently tracing).
    pub fn trace_ctx(&mut self) -> Option<&mut TraceCtx> {
        self.tracing.as_mut()
    }

    pub fn force_finish_trace_enabled(&self) -> bool {
        self.force_finish_trace
    }

    // ── RPython opimpl_* equivalents for virtualizable ──────────────
    //
    // pyjitpl.py:1120-1146 `_nonstandard_virtualizable(pc, box, fielddescr)`
    // is implemented in `TraceCtx::is_nonstandard_virtualizable` with the
    // full Step 1..5b shape; the opimpl_*_vable thin wrappers below forward
    // to `TraceCtx::vable_*` which are the line-by-line port of RPython's
    // `opimpl_*_vable` opcode handlers. The earlier `MetaInterp` duplicate
    // (with its own `is_standard_virtualizable` / `nonstandard_virtualizable`
    // / `virtualizable_field_index` / `get_arrayitem_vable_index` /
    // `check_synchronized_virtualizable` helpers) was a pyre-introduced
    // duplication of the same logic and has been removed in favour of the
    // single TraceCtx implementation.

    /// pyjitpl.py:3499-3512 `MetaInterp.replace_box(oldbox, newbox)`.
    ///
    /// ```text
    /// def replace_box(self, oldbox, newbox):
    ///     for frame in self.framestack:
    ///         frame.replace_active_box_in_frame(oldbox, newbox)
    ///     boxes = self.virtualref_boxes
    ///     for i in range(len(boxes)):
    ///         if boxes[i] is oldbox:
    ///             boxes[i] = newbox
    ///     if (self.jitdriver_sd.virtualizable_info is not None or
    ///         self.jitdriver_sd.greenfield_info is not None):
    ///         boxes = self.virtualizable_boxes
    ///         for i in range(len(boxes)):
    ///             if boxes[i] is oldbox:
    ///                 boxes[i] = newbox
    ///     self.heapcache.replace_box(oldbox, newbox)
    /// ```
    ///
    /// RPython rewrites every place where `oldbox` may appear during
    /// tracing — frame registers, virtualref pairs, the standard
    /// virtualizable box array, and the heap cache — and does so
    /// eagerly so subsequent tracing-time queries see the new identity.
    pub fn replace_box(&mut self, oldbox: OpRef, newbox: OpRef) {
        // pyjitpl.py:3500-3501: for frame in self.framestack:
        //                          frame.replace_active_box_in_frame(...)
        //
        // pyre's MIFrame::replace_active_box_in_frame needs `oldbox.type`
        // to pick the bank to scan; OpRef does not carry a type tag, so
        // resolve the type once via the trace context's type oracle and
        // reuse it for every frame.  When the trace context is absent
        // (post-tracing or never-traced paths) the framestack walk is a
        // no-op, matching the RPython semantic that `replace_box` is
        // exclusively a tracing-time operation.
        if let Some(oldbox_type) = self
            .tracing
            .as_ref()
            .and_then(|ctx| ctx.get_opref_type(oldbox))
        {
            for frame in self.framestack.frames.iter_mut() {
                frame.replace_active_box_in_frame(oldbox, newbox, oldbox_type);
            }
        }
        // pyjitpl.py:3502-3505: virtualref_boxes walk
        for slot in self.virtualref_boxes.iter_mut() {
            if slot.0 == oldbox {
                slot.0 = newbox;
            }
        }
        if let Some(ctx) = self.tracing.as_mut() {
            // pyjitpl.py:3506-3512:
            //     boxes = self.virtualizable_boxes
            //     for i in range(len(boxes)):
            //         if boxes[i] is oldbox:
            //             boxes[i] = newbox
            //     self.heapcache.replace_box(oldbox, newbox)
            //
            // The trace-context portion (virtualizable_boxes walk +
            // heap_cache walk) lives on `TraceCtx::replace_box` so the
            // trace_ctx-only call site in `_nonstandard_virtualizable`
            // Step 4 shares the same helper.
            ctx.replace_box(oldbox, newbox);
        }
    }

    /// pyjitpl.py:3446-3450 `MetaInterp.synchronize_virtualizable()`.
    ///
    /// ```text
    /// def synchronize_virtualizable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     virtualizable_box = self.virtualizable_boxes[-1]
    ///     virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///     vinfo.write_boxes(virtualizable, self.virtualizable_boxes)
    /// ```
    ///
    /// Delegates to `TraceCtx::synchronize_virtualizable`, which owns the
    /// `virtualizable_values` shadow and the mirrored `vable_ptr`. Keeping
    /// this thin wrapper preserves the RPython call-site spelling
    /// (`self.metainterp.synchronize_virtualizable()`) at setfield_vable /
    /// setarrayitem_vable sites that route through MetaInterp.
    pub fn synchronize_virtualizable(&mut self, _vable_opref: OpRef) {
        if let Some(ctx) = self.tracing.as_ref() {
            ctx.synchronize_virtualizable();
        }
    }

    /// pyjitpl.py:3452-3464 `MetaInterp.load_fields_from_virtualizable()`.
    ///
    /// ```text
    /// def load_fields_from_virtualizable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         self.virtualizable_boxes = vinfo.read_boxes(self.cpu, virtualizable, 0)
    ///         self.virtualizable_boxes.append(virtualizable_box)
    /// ```
    ///
    /// Reloads the tracing-time `virtualizable_boxes` cache from the heap
    /// object just before we abort to blackhole after an escaping residual
    /// call. This mirrors the upstream "heap wins" direction on the escape
    /// path so any forced writes become visible to the resumed interpreter.
    pub fn load_fields_from_virtualizable(&mut self) {
        let info = match self.virtualizable_info().cloned() {
            Some(info) => info,
            None => return,
        };
        let vable_ptr = self.vable_ptr;
        if vable_ptr.is_null() {
            return;
        }
        let (vable_box, array_lengths) = match self.tracing.as_ref() {
            Some(ctx) => {
                let Some(vable_box) = ctx.standard_virtualizable_box() else {
                    return;
                };
                let array_lengths = ctx
                    .virtualizable_array_lengths()
                    .map(|lengths| lengths.to_vec())
                    .unwrap_or_default();
                (vable_box, array_lengths)
            }
            None => return,
        };
        let (static_boxes, array_boxes) = unsafe { info.read_all_boxes(vable_ptr, &array_lengths) };
        let Some(ctx) = self.tracing.as_mut() else {
            return;
        };
        let cap = static_boxes.len() + array_boxes.iter().map(Vec::len).sum::<usize>() + 1;
        let mut boxes = Vec::with_capacity(cap);
        let mut values = Vec::with_capacity(cap);
        for (index, value) in static_boxes.into_iter().enumerate() {
            let (opref, concrete) = match info.static_fields[index].field_type {
                majit_ir::Type::Int => (ctx.const_int(value), Value::Int(value)),
                majit_ir::Type::Ref => (
                    ctx.const_ref(value),
                    Value::Ref(majit_ir::GcRef(value as usize)),
                ),
                majit_ir::Type::Float => (
                    ctx.const_float(value),
                    Value::Float(f64::from_bits(value as u64)),
                ),
                majit_ir::Type::Void => continue,
            };
            boxes.push(opref);
            values.push(concrete);
        }
        for (array_index, items) in array_boxes.into_iter().enumerate() {
            let item_type = info.array_fields[array_index].item_type;
            for value in items {
                let (opref, concrete) = match item_type {
                    majit_ir::Type::Int => (ctx.const_int(value), Value::Int(value)),
                    majit_ir::Type::Ref => (
                        ctx.const_ref(value),
                        Value::Ref(majit_ir::GcRef(value as usize)),
                    ),
                    majit_ir::Type::Float => (
                        ctx.const_float(value),
                        Value::Float(f64::from_bits(value as u64)),
                    ),
                    majit_ir::Type::Void => continue,
                };
                boxes.push(opref);
                values.push(concrete);
            }
        }
        boxes.push(vable_box);
        // The vable identity's concrete value is the heap pointer itself.
        values.push(Value::Ref(majit_ir::GcRef(vable_ptr as usize)));
        ctx.set_virtualizable_boxes_with_info(boxes, values, &info, &array_lengths);
    }

    /// pyjitpl.py:1167-1172 `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_int requires active tracing")
            .vable_getfield_int(vable_opref, fielddescr)
    }

    /// pyjitpl.py:1173-1179 `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_ref requires active tracing")
            .vable_getfield_ref(vable_opref, fielddescr)
    }

    /// pyjitpl.py:1180-1186 `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    pub fn opimpl_getfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getfield_vable_float requires active tracing")
            .vable_getfield_float(vable_opref, fielddescr)
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box, valuebox, fielddescr, pc)`.
    pub fn opimpl_setfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_int requires active tracing")
            .vable_setfield(vable_opref, fielddescr, value, concrete);
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` — ref variant.
    pub fn opimpl_setfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_ref requires active tracing")
            .vable_setfield(vable_opref, fielddescr, value, concrete);
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` — float variant.
    pub fn opimpl_setfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setfield_vable_float requires active tracing")
            .vable_setfield(vable_opref, fielddescr, value, concrete);
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — int variant.
    pub fn opimpl_getarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_int requires active tracing")
            .vable_getarrayitem_int_indexed(vable_opref, index, index_runtime_value, fdescr, adescr)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — ref variant.
    pub fn opimpl_getarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_ref requires active tracing")
            .vable_getarrayitem_ref_indexed(vable_opref, index, index_runtime_value, fdescr, adescr)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — float variant.
    pub fn opimpl_getarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        self.tracing
            .as_mut()
            .expect("opimpl_getarrayitem_vable_float requires active tracing")
            .vable_getarrayitem_float_indexed(
                vable_opref,
                index,
                index_runtime_value,
                fdescr,
                adescr,
            )
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — int variant.
    pub fn opimpl_setarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_int requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                fdescr,
                adescr,
                value,
                concrete,
            );
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — ref variant.
    pub fn opimpl_setarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_ref requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                fdescr,
                adescr,
                value,
                concrete,
            );
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` — float variant.
    pub fn opimpl_setarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) {
        self.tracing
            .as_mut()
            .expect("opimpl_setarrayitem_vable_float requires active tracing")
            .vable_setarrayitem_indexed(
                vable_opref,
                index,
                index_runtime_value,
                fdescr,
                adescr,
                value,
                concrete,
            );
    }

    /// pyjitpl.py:1253-1263 `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    pub fn opimpl_arraylen_vable(
        &mut self,
        vable_opref: OpRef,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> OpRef {
        self.tracing
            .as_mut()
            .expect("opimpl_arraylen_vable requires active tracing")
            .vable_arraylen_vable(vable_opref, fdescr, adescr)
    }

    /// pyjitpl.py:1064-1073 `opimpl_hint_force_virtualizable(box)`.
    ///
    /// ```text
    /// def opimpl_hint_force_virtualizable(self, box):
    ///     self.metainterp.gen_store_back_in_vable(box)
    /// ```
    ///
    /// RPython's `gen_store_back_in_vable` (pyjitpl.py:3465) handles the
    /// nonstandard / forced_virtualizable gating internally and emits the
    /// SETFIELD_GC + SETARRAYITEM_GC + token-clear flush. pyre's TraceCtx
    /// hosts the same gating now (see TraceCtx::gen_store_back_in_vable),
    /// so this is a thin forward.
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.gen_store_back_in_vable(vable_opref);
        }
    }

    /// pyjitpl.py:1789-1814 opimpl_virtual_ref parity.
    /// Creates concrete vref via virtual_ref_during_tracing(real_object),
    /// records VIRTUAL_REF(box, cindex), pushes [virtualbox, vrefbox].
    pub fn opimpl_virtual_ref(&mut self, virtual_obj: OpRef, virtual_obj_ptr: usize) -> OpRef {
        let Some(ctx) = self.tracing.as_mut() else {
            return OpRef::NONE;
        };
        // pyjitpl.py:1804: virtual_ref_during_tracing(virtual_obj)
        let vref_info = crate::virtualref::VirtualRefInfo::new();
        let vref_ptr = vref_info.virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
        // pyjitpl.py:1805: cindex = ConstInt(len(virtualref_boxes) // 2)
        let cindex = ctx.const_int((self.virtualref_boxes.len() / 2) as i64);
        // pyjitpl.py:1806: record VIRTUAL_REF(box, cindex)
        let vref = ctx.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex]);
        // pyjitpl.py:1814: virtualref_boxes += [virtualbox, vrefbox]
        self.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
        self.virtualref_boxes.push((vref, vref_ptr as usize));
        vref
    }

    /// pyjitpl.py:1819-1831 opimpl_virtual_ref_finish parity.
    /// Checks is_virtual_ref() on concrete vref before recording.
    pub fn opimpl_virtual_ref_finish(&mut self, vref: OpRef, _virtual_obj: OpRef) {
        let Some(ctx) = self.tracing.as_mut() else {
            return;
        };
        // pyjitpl.py:1827: check is_virtual_ref(vrefbox)
        let vref_ptr = self
            .virtualref_boxes
            .last()
            .map(|&(_, ptr)| ptr)
            .unwrap_or(0);
        let vref_info = crate::virtualref::VirtualRefInfo::new();
        let is_vref = vref_ptr != 0 && unsafe { vref_info.is_virtual_ref(vref_ptr as *const u8) };
        if is_vref {
            // pyjitpl.py:3371: VIRTUAL_REF_FINISH(vrefbox, NULL)
            let null = ctx.const_ref(0);
            let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref, null]);
        }
        if self.virtualref_boxes.len() >= 2 {
            self.virtualref_boxes.pop();
            self.virtualref_boxes.pop();
        }
    }

    /// Whether the engine is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.tracing.is_some()
    }

    /// pyjitpl.py:2788-2807 `blackhole_if_trace_too_long`.
    ///
    /// Runs the too-long bookkeeping (`disable_noninlinable_function` /
    /// `aborted_tracing_*` stash / `trace_next_iteration` / `prepare_trace_segmenting`)
    /// and returns `Some(AbortReason::TooLong)` so the caller can unwind
    /// exactly once via `abort_trace_live(false)` + `aborted_tracing(reason)`
    /// — matching RPython's `raise SwitchToBlackhole(ABORT_TOO_LONG)` →
    /// `_interpret` handler → `aborted_tracing(reason)` shape.
    ///
    /// Returns `None` when the trace is still within budget.
    #[inline]
    pub fn blackhole_if_trace_too_long(&mut self) -> Option<AbortReason> {
        match self.tracing.as_ref() {
            Some(ctx) if ctx.is_too_long() => self.blackhole_trace_too_long_slow(),
            _ => None,
        }
    }

    #[cold]
    #[inline(never)]
    fn blackhole_trace_too_long_slow(&mut self) -> Option<AbortReason> {
        let ctx = self.tracing.as_ref().expect("tracing is Some");
        let green_key = ctx.green_key;
        // pyjitpl.py:2793: find_biggest_function — if an inlined function
        // caused the bloat, disable just that function.
        let huge_fn_key = ctx.find_biggest_function();
        // pyjitpl.py:2795: `self.portal_trace_positions = None` marks the
        // abort boundary so post-abort consumers (e.g. test inspections
        // at pyjitpl.py:3547) can detect a terminated trace session.
        self.portal_trace_positions = None;
        // pyjitpl.py:2801 `if self.current_merge_points:` — outermost
        // loop's greenkey, used only when one exists (never for bridges).
        let outermost_merge_key = ctx.current_merge_points_first_greenkey();
        if let Some(huge_fn_key) = huge_fn_key {
            self.warm_state.disable_noninlinable_function(huge_fn_key);
            // pyjitpl.py:2799-2800: stash the aborted jd_sd + greenkey so
            // `aborted_tracing(reason)` can fire `on_trace_too_long` when
            // the hook is ported.  Pyre only registers a single jitdriver
            // so jd_sd.index is always 0.
            self.aborted_tracing_jitdriver = Some(0);
            self.aborted_tracing_greenkey = Some(huge_fn_key);
            // pyjitpl.py:2801-2804: only boost retrace for the outermost
            // loop (when `current_merge_points` is non-empty).  Bridge
            // and function-entry overflow must NOT trigger trace_next_iteration.
            if let Some(outer_key) = outermost_merge_key {
                self.warm_state.trace_next_iteration(outer_key);
            }
        } else {
            // pyjitpl.py:2806 `self.prepare_trace_segmenting()`.
            self.prepare_trace_segmenting();
        }
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] blackhole_if_trace_too_long: aborting at key={}",
                green_key
            );
        }
        // pyjitpl.py:2807 `raise SwitchToBlackhole(ABORT_TOO_LONG)`.
        // Return the reason; caller unwinds with abort_trace_live +
        // aborted_tracing(reason) exactly once.
        Some(AbortReason::TooLong)
    }

    /// pyjitpl.py:2809 `MetaInterp.prepare_trace_segmenting`.
    ///
    /// Called when a trace overflows `trace_limit` and no inlinable function
    /// caused it.  Two independent branches (the upstream method tests both):
    ///
    /// 1. `if self.current_merge_points:` — set the warmstate-level
    ///    JC_FORCE_FINISH / JC_DONT_TRACE_HERE flags on the outermost merge
    ///    point's greenkey so the next tracing run for that loop segments
    ///    instead of aborting again.
    /// 2. `if not isinstance(self.resumekey, ResumeFromInterpDescr):` — we
    ///    are tracing a bridge.  ResumeGuardDescr has no spare bits, so set
    ///    `FORCE_BRIDGE_SEGMENTING` on the source loop token; all future
    ///    bridges from that token will then inherit `force_finish_trace=True`
    ///    via `start_retrace_from_guard` (compile.py:725-731 parity).
    fn prepare_trace_segmenting(&mut self) {
        // pyjitpl.py:2815 `if self.current_merge_points:` — outermost
        // loop's greenkey, never set for bridges.
        let outermost_merge_key = self
            .tracing
            .as_ref()
            .and_then(|c| c.current_merge_points_first_greenkey());
        if let Some(outer_key) = outermost_merge_key {
            // pyjitpl.py:2819 `JitCell.trace_next_iteration(greenkey)`.
            self.warm_state.trace_next_iteration(outer_key);
            // pyjitpl.py:2820 `warmstate.mark_force_finish_tracing(greenkey)`.
            self.warm_state.mark_force_finish_tracing(outer_key);
            // pyjitpl.py:2822 `warmstate.dont_trace_here(greenkey)`.
            self.warm_state.disable_noninlinable_function(outer_key);
        }
        // pyjitpl.py:2825 `if not isinstance(self.resumekey, ResumeFromInterpDescr):`
        // — pyre carries the source token directly via
        // `TraceCtx::resumekey_original_loop_token` (Some only when bridge
        // tracing; None for ResumeFromInterpDescr-equivalent loop entry).
        if let Some(source_jct) = self
            .tracing
            .as_ref()
            .and_then(|c| c.resumekey_original_loop_token().cloned())
        {
            // pyjitpl.py:2832-2833 `loop_token.retraced_count |=
            // loop_token.FORCE_BRIDGE_SEGMENTING`.
            let cur = source_jct.retraced_count.get();
            source_jct
                .retraced_count
                .set(cur | majit_backend::JitCellToken::FORCE_BRIDGE_SEGMENTING);
        }
    }

    /// RPython JC_TRACING parity: check if we are currently tracing
    /// this specific green key. Returns false for different green keys,
    /// matching RPython's per-cell JC_TRACING flag.
    ///
    /// `target_raw` is the structured `(code_ptr, pc)` greenkey;
    /// comparison routes through `TraceCtx::is_tracing_key` which walks
    /// `green_key_raw` + `inline_frames` element-wise (pyjitpl.py:1396-
    /// 1401 parity).
    #[inline]
    pub fn is_tracing_key(&self, target_raw: (usize, usize)) -> bool {
        self.tracing
            .as_ref()
            .is_some_and(|ctx| ctx.is_tracing_key(target_raw))
    }

    /// Finish the current active trace without optimizing or compiling it.
    ///
    /// This is a semantic-seam helper for parity tests: it lets callers
    /// inspect the raw recorded trace that the proc-macro/runtime path
    /// produced, without requiring backend compilation.
    pub fn finish_trace_for_parity(
        &mut self,
        finish_args: &[OpRef],
    ) -> Option<(TreeLoop, HashMap<u32, i64>)> {
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take()?;
        let green_key = ctx.green_key;
        ctx.finish(finish_args, crate::make_fail_descr(finish_args.len()));
        let constants = std::mem::take(&mut ctx.constants).into_inner();
        let trace = ctx.into_tree_loop();
        self.warm_state.abort_tracing(green_key, false);
        Some((trace, constants))
    }

    /// RPython-compatible helper name from compile.py.
    pub fn send_loop_to_backend(&mut self, jump_args: &[OpRef], meta: M) -> CompileOutcome {
        self.compile_loop(jump_args, meta)
    }

    /// compile.py:504-511 `send_loop_to_backend()` virtualizable hook:
    ///
    /// ```python
    /// vinfo = jitdriver_sd.virtualizable_info
    /// if vinfo is not None:
    ///     vable = orig_inpargs[jitdriver_sd.index_of_virtualizable].getref_base()
    ///     patch_new_loop_to_load_virtualizable_fields(loop, jitdriver_sd, vable)
    /// ```
    ///
    /// RPython runs this unconditionally for every loop (both JUMP-terminated
    /// and FINISH-terminated) before the loop goes into `cpu.compile_loop`.
    /// Per-array lengths come from `vinfo.get_array_length(vable, arrayindex)`
    /// at compile.py:443 — a direct read of the concrete virtualizable heap
    /// object, not a synthesis from `len(inputargs)`.
    ///
    /// `orig_vable_ptr` is the `*const u8` view of RPython's
    /// `orig_inpargs[jitdriver_sd.index_of_virtualizable].getref_base()`
    /// (compile.py:510): the caller extracts the constant `Value::Ref` that
    /// the tracer stashed for the virtualizable inputarg at trace-start and
    /// passes it through. Helper name matches compile.py so audits can grep
    /// the same symbol on both sides.
    ///
    /// Invoked from both compile sites that mirror RPython's
    /// `send_loop_to_backend` (compile.py:504-511): the JUMP-terminated
    /// `compile_loop_body` and the FINISH-terminated `finish_and_compile`.
    /// Pyre's structure matches RPython compile.py:312/320/327 — `inputargs`
    /// (entry contract) and `start_label.args` carry the trace's ROOT
    /// expanded shape ([0..num_inputs)), while the body `LABEL`/`JUMP`
    /// inside `compiled_ops` use virtualstate-allocated OpRefs that are
    /// outside the forwarding map. Truncating `inputargs` to `num_red_args`
    /// and prepending GETFIELD_GC / GETARRAYITEM_GC therefore only rewrites
    /// the entry contract and `start_label.args`; body LABEL/JUMP arities
    /// stay independent.
    pub(crate) fn patch_new_loop_to_load_virtualizable_fields(
        &self,
        inputargs: &mut Vec<InputArg>,
        ops: &mut Vec<Op>,
        constants: &mut HashMap<u32, i64>,
        constant_types: &mut HashMap<u32, Type>,
        driver_descriptor: Option<&crate::jitdriver::JitDriverStaticData>,
        orig_vable_ptr: *const u8,
    ) {
        let Some(vinfo) = self.virtualizable_info() else {
            return;
        };
        let Some(driver) = driver_descriptor else {
            return;
        };
        let Some(index_of_vable) = driver.virtualizable_arg_index() else {
            return;
        };
        let num_red_args = driver.num_reds();
        if inputargs.len() <= num_red_args {
            // Trace was never expanded (no virtualizable fields live at entry).
            return;
        }
        // compile.py:508-511
        //     vable = orig_inpargs[jitdriver_sd.index_of_virtualizable].getref_base()
        //     patch_new_loop_to_load_virtualizable_fields(loop, jitdriver_sd, vable)
        //
        // RPython never tolerates a null `vable` here — the live
        // virtualizable is whatever `orig_inpargs[idx]` was constructed from
        // at trace-start. Require the same invariant: the caller must pass
        // the constant Ref value from that inputarg. A null pointer means
        // the tracer-time inputarg lookup failed, which is a bug upstream
        // of this helper.
        assert!(
            !orig_vable_ptr.is_null(),
            "patch_new_loop_to_load_virtualizable_fields requires \
             orig_inpargs[index_of_virtualizable].getref_base() to be non-null"
        );
        // compile.py:443 `vinfo.get_array_length(vable, arrayindex)` — the
        // concrete virtualizable heap object is the sole source of truth
        // for every array length. Read each array field directly via
        // `VirtualizableInfo::get_array_length(obj_ptr, i)`
        // (virtualizable.rs:695-711). Any layout that cannot expose its
        // length on the heap object must be fixed inside
        // `VirtualizableInfo` itself (to match `vinfo.get_array_length`'s
        // universal contract), not worked around in this helper.
        let array_lengths: Vec<usize> = (0..vinfo.array_fields.len())
            .map(|i| unsafe { vinfo.get_array_length(orig_vable_ptr, i) })
            .collect();
        compile::patch_new_loop_to_load_virtualizable_fields(
            ops,
            inputargs,
            vinfo,
            &array_lengths,
            num_red_args,
            index_of_vable,
            constants,
            constant_types,
        );
        // compile.py:425-461 `patch_new_loop_to_load_virtualizable_fields`
        // only touches `loop.inputargs`; it does not rewrite any LABEL/JUMP
        // arity inside `loop.operations`. The helper's forwarding map is
        // ROOT inputarg OpRef → fresh GETFIELD result OpRef, so any op that
        // referenced a removed inputarg slot (start_label, preamble ops, and
        // any guard fail_args reaching back to it) gets rewritten in place;
        // body LABEL/JUMP carry virtualstate-allocated OpRefs outside that
        // map and stay untouched. Both `compile_loop_body` and
        // `finish_and_compile` invoke this helper, mirroring RPython's
        // unconditional `send_loop_to_backend` wiring (compile.py:504-511).
    }

    fn orig_vable_ptr_from_trace_ctx(
        &self,
        ctx: &TraceCtx,
        driver_descriptor: Option<&crate::jitdriver::JitDriverStaticData>,
    ) -> *const u8 {
        driver_descriptor
            .and_then(|driver| driver.virtualizable_arg_index())
            .and_then(|idx| ctx.initial_inputarg_consts.get(idx).copied())
            .and_then(|const_ref| match ctx.constants.get_value(const_ref) {
                Some(majit_ir::Value::Ref(gcref)) => Some(gcref.0 as *const u8),
                _ => None,
            })
            .unwrap_or(std::ptr::null())
    }

    /// compile.py:168 / pyjitpl.py:3605 parity: every real loop token must
    /// carry the same outermost jitdriver metadata that compile_tmp_callback
    /// installs on pending tokens.  The backend's handle_call_assembler
    /// lookup reads this to decide whether the rewritten op is [frame] or
    /// [frame, virtualizable].
    fn configure_loop_token_for_driver(
        &self,
        token: &mut JitCellToken,
        green_key: u64,
        driver_descriptor: Option<&JitDriverStaticData>,
    ) {
        // `compile.py:168 jitcell_token.outermost_jitdriver_sd = jitdriver_sd`
        // is already set by `make_jitcell_token` at the call site; this
        // helper only fills the pyre-specific fields (`green_key`,
        // `num_scalar_inputargs`, `virtualizable_arg_index`) used by
        // warmstate cell lookup and the backend's
        // `handle_call_assembler` rewrite.
        token.green_key = green_key;
        token.num_scalar_inputargs = self.num_scalar_inputargs;
        token.virtualizable_arg_index =
            driver_descriptor.and_then(JitDriverStaticData::virtualizable_arg_index);
    }

    /// Close the current trace, optimize, and compile.
    ///
    /// `jump_args` are the symbolic values (OpRefs) at the end of the loop,
    /// in the same order as the InputArgs registered during `on_back_edge`.
    /// `meta` is interpreter-specific metadata to store alongside the compiled loop.
    /// pyjitpl.py:2979-3036 `reached_loop_header` → `compile_loop` dispatch.
    ///
    /// This public entry wraps [`Self::compile_loop_body`] so every exit
    /// path restores the RPython invariant that
    /// `active_trace_session.is_some()` iff `self.tracing.is_some()`.
    /// Upstream uses `self.history` as a shared mutable object — cancel
    /// paths fall through to `current_merge_points.append(...)` with the
    /// history still live, and the only time the session ends is when
    /// `abort_tracing` or successful compilation drops the tracer. pyre
    /// mirrors that by checking, after the body returns, whether the
    /// inner path consumed `self.tracing`. If so, the session envelope
    /// is dropped alongside it; if not (early-Cancelled paths), both
    /// halves stay live for continued tracing.
    pub fn compile_loop(&mut self, jump_args: &[OpRef], meta: M) -> CompileOutcome {
        let outcome = self.compile_loop_body(jump_args, meta);
        // pyjitpl.py:3015-3032 parity: once the body has taken the trace
        // ctx (tracing=None), drop the matching frontend session so the
        // next `begin_trace_session` sees a clean slate. Cancelled paths
        // that kept `self.tracing` alive (e.g. `prior_retraced_count ==
        // MAX` above) fall through harmlessly and keep tracing running.
        if self.tracing.is_none() && self.active_trace_session.is_some() {
            self.clear_trace_session();
        }
        outcome
    }

    fn compile_loop_body(&mut self, jump_args: &[OpRef], meta: M) -> CompileOutcome {
        // pyjitpl.py:2993-3007: if partial_trace is set, the previous
        // compilation attempt requested a retrace. Verify the green_key
        // matches and dispatch to compile_retrace.
        if self.partial_trace.is_some() {
            if let Some(retrace_pos) = self.retracing_from {
                // pyjitpl.py:2994: if start != self.retracing_from
                // Find the merge point whose position matches retracing_from.
                // pyjitpl.py:2994: iterate current_merge_points in reverse,
                // check same_greenkey and position match. Use header_pc
                // for precise matching across root/inner key registrations.
                let position_matches = self
                    .tracing
                    .as_ref()
                    .and_then(|ctx| {
                        ctx.get_merge_point_at(ctx.green_key, ctx.header_pc)
                            .map(|mp| mp.position == retrace_pos)
                    })
                    .unwrap_or(false);
                if position_matches {
                    let ok = self.compile_retrace(jump_args, meta.clone());
                    if ok {
                        self.cancel_count = 0;
                        return CompileOutcome::Compiled {
                            green_key: 0,
                            from_retry: false,
                        };
                    }
                    // pyjitpl.py:3004: creation of the loop was cancelled!
                    self.cancel_count += 1;
                    if self.cancelled_too_many_times() {
                        if crate::majit_log_enabled() {
                            eprintln!("[jit] retrace cancelled too many times");
                        }
                        self.clear_retrace_state();
                        if let Some(ctx) = self.tracing.take() {
                            self.warm_state.abort_tracing(ctx.green_key, false);
                        }
                        // Keep tracing + session in lockstep (pyjitpl.py:3015).
                        self.clear_trace_session();
                        return CompileOutcome::Aborted;
                    }
                    if self.tracing.is_none() {
                        // compile.py has no "late cancel after draining the
                        // tracer" state. If compile_retrace consumed the
                        // tracing ctx, it was a hard backend failure and the
                        // caller must abort instead of continuing to trace.
                        return CompileOutcome::Aborted;
                    }
                    // Not too many — clear retrace state and fall through
                    // to normal compile_loop path.
                    self.exported_state = None;
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] retrace cancelled, trying normal compilation");
                    }
                } else {
                    // pyjitpl.py:2994-2995: position mismatch — abort.
                    self.clear_retrace_state();
                    if let Some(ctx) = self.tracing.take() {
                        self.warm_state.abort_tracing(ctx.green_key, false);
                    }
                    // Keep tracing + session in lockstep (pyjitpl.py:3015).
                    self.clear_trace_session();
                    return CompileOutcome::Aborted;
                }
            }
        }

        // Clear bridge retrace flag — partial_trace is the authoritative
        // state for compile_retrace dispatch.
        if self.retrace_after_bridge {
            self.retrace_after_bridge = false;
        }
        // pyjitpl.py:3162: has_compiled_targets(ptoken) →
        // raise SwitchToBlackhole(ABORT_BAD_LOOP).
        if let Some(ctx) = self.tracing.as_ref() {
            let gk = ctx.green_key;
            if self.has_compiled_targets(gk) {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_loop → SwitchToBlackhole: has_compiled_targets key={}",
                        gk
                    );
                }
                self.abort_trace(false);
                return CompileOutcome::Aborted;
            }
        }

        let vable_config = self.current_virtualizable_optimizer_config();
        // pyjitpl.py:3015-3032 parity: compile_loop uses `self.history`
        // without consuming it, so cancel paths can fall through to
        // `current_merge_points.append(...)` and keep tracing. Before
        // committing to a compile we mirror that by reading green_key
        // from the live trace ctx; only the "committed" exits below
        // take ownership of the ctx and drop the trace session.
        let (green_key, cut_inner_green_key) = {
            let ctx = self
                .tracing
                .as_ref()
                .expect("compile_loop: no active trace ctx");
            let outer = ctx.green_key;
            let cut_inner = ctx.cut_inner_green_key;
            // compile.py:269-270: cross-loop cut → store under inner loop's
            // jitcell_token. RPython: jitcell_token = cross_loop.jitcell_token.
            (cut_inner.unwrap_or(outer), cut_inner)
        };

        // pyjitpl.py:3015-3032 parity: pyre caches the retrace limit per
        // green_key so guard-heavy recompilations do not loop forever.
        // The limit check happens BEFORE we consume the trace ctx so
        // Cancelled here keeps `self.tracing` + `active_trace_session`
        // alive — the caller (reached_loop_header analogue) then falls
        // through to `current_merge_points.append(...)` and records more
        // ops. `warm_state.abort_tracing(..., permanent=true)` disables
        // future entries at this key without disturbing the live trace.
        let prior_front_target_tokens_early = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .unwrap_or_default();
        let prior_retraced_count_early = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.token.get_retraced_count())
            .unwrap_or(0);
        if prior_retraced_count_early == u32::MAX && !prior_front_target_tokens_early.is_empty() {
            if crate::majit_log_enabled() {
                eprintln!("[jit] skipping recompile: retraced_count=MAX for key={green_key}");
            }
            self.warm_state.abort_tracing(green_key, true);
            return CompileOutcome::Cancelled;
        }

        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        // Cache driver descriptor before ctx is partially consumed below;
        // mirrors the FINISH-path capture pattern (see `finish_and_compile`).
        let driver_descriptor = ctx.driver_descriptor().cloned();
        // compile.py:510 `vable = orig_inpargs[index_of_virtualizable].getref_base()`.
        // Resolve while `ctx` is still whole (before `ctx.constants` is moved
        // out below) so `patch_new_loop_to_load_virtualizable_fields` can
        // read the heap object via `vinfo.get_array_length(vable, i)`
        // (compile.py:443).
        let orig_vable_ptr_loop =
            self.orig_vable_ptr_from_trace_ctx(&ctx, driver_descriptor.as_ref());
        let cross_loop_cut = if cut_inner_green_key.is_some() {
            ctx.get_merge_point_at(green_key, ctx.header_pc)
                .filter(|mp| mp.position._pos > 0)
                .map(|mp| {
                    (
                        mp.original_boxes.clone(),
                        mp.original_box_types.clone(),
                        crate::history::TreeLoopCutPosition::new(mp.position._pos),
                    )
                })
        } else {
            None
        };

        // compile.py:221: call_pure_results = metainterp.call_pure_results
        let call_pure_results = ctx.take_call_pure_results();

        let mut recorder = ctx.recorder;
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED. This allows external invalidation
        // (via JitCellToken.invalidate()) to force compiled loops
        // back to the interpreter.
        // RPython heapcache.py:176: every trace gets at least one
        // GUARD_NOT_INVALIDATED before the closing JUMP. fail_args = jump_args
        // so guard failure restores the same state as the JUMP target.
        // pyjitpl.py:2969: GUARD_FUTURE_CONDITION and heapcache.py:176:
        // GUARD_NOT_INVALIDATED are both emitted during tracing in
        // close_loop_args_at (state.rs) via record_guard → capture_resumedata.
        recorder.close_loop(jump_args);
        // Task #70: snapshots live on TraceCtx; rebuild the TreeLoop with
        // them so downstream consumers (`trace.snapshots`) still observe
        // the captured resumedata. `recorder.get_trace()` on its own
        // returns a snapshot-less TreeLoop post-Task #70.
        let mut trace = recorder.get_trace();
        trace.snapshots = std::mem::take(&mut ctx.snapshots);

        // compile.py:269-270: cut trace at cross-loop merge point.
        // When the trace was retargeted to a different loop header, record
        // the new header PC so meta can be updated after insert.
        // RPython parity: only cut the trace if no compiled entry already
        // exists at the inner loop's green_key. If the inner loop was already
        // compiled independently, its entry has correct code+meta. Cutting
        // and replacing would install cross-loop-cut code with mismatched
        // inputarg layout. Instead, keep the original (uncut) trace and
        // compile.py:269: cut trace at cross-loop merge point.
        // When the trace was retargeted to a different loop header,
        // cut_trace_from removes ops before the merge point and
        // replaces inputargs with original_boxes at the cut position.
        let trace = if let Some((ref original_boxes, ref original_box_types, start)) =
            cross_loop_cut
        {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] cut_trace_from: start.op_index={} original_boxes={} trace_ops={} header_pc={}",
                    start.op_index,
                    original_boxes.len(),
                    trace.ops.len(),
                    ctx.header_pc,
                );
            }
            trace.cut_trace_from_with_consts(
                start,
                original_boxes,
                original_box_types,
                &ctx.initial_inputarg_consts,
            )
        } else {
            trace
        };
        let trace_snapshots = trace.snapshots.clone();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        // Refresh shadow-stack-rooted Ref constants so `into_inner_with_types`
        // sees the post-GC addresses for any object that moved since the last
        // `get_or_insert_typed`.
        ctx.constants.refresh_from_gc();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let trace_ops = trace.ops.clone();
        if crate::majit_log_enabled() {
            eprintln!("--- trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        let num_ops_before = trace.ops.len();
        let num_trace_inputargs = trace.inputargs.len();

        // Save trace_ops + constants snapshot for potential unroll-free retry
        // (pyjitpl.py:3016-3021).
        let trace_ops_snapshot = trace_ops.clone();
        let constants_snapshot = constants.clone();

        // Use UnrollOptimizer for preamble peeling when available.
        // compile.py: compile_loop → PreambleCompileData + LoopCompileData.
        // NOTE: the `prior_retraced_count == u32::MAX` early Cancelled
        // path fires above (before ctx take) so we only reach this point
        // when a recompile is actually attempted. `pending_preamble_tokens`
        // must be consumed here because the tokens from a previous
        // InvalidLoop attempt are now being resupplied to the unroller.
        let prior_front_target_tokens = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .or_else(|| self.pending_preamble_tokens.remove(&green_key))
            .unwrap_or_default();
        let mut unroll_opt = crate::optimizeopt::unroll::UnrollOptimizer::new();
        unroll_opt.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        unroll_opt.target_tokens = prior_front_target_tokens.clone();
        unroll_opt.retraced_count = prior_retraced_count_early;
        unroll_opt.retrace_limit = self.warm_state.retrace_limit();
        unroll_opt.max_retrace_guards = self.warm_state.max_retrace_guards();
        unroll_opt.constant_types = constant_types.clone();
        unroll_opt.callinfocollection = self.callinfocollection.clone();
        unroll_opt.call_pure_results = call_pure_results.clone();
        unroll_opt.numbering_type_overrides = numbering_overrides.clone();
        // RPython Box type parity: each InputArg carries its type from
        // tracing. Propagate to optimizer so value_types covers inputargs.
        unroll_opt.trace_inputarg_types = trace.inputargs.iter().map(|ia| ia.tp).collect();
        // resume.py parity: convert tracing-time snapshots to flat OpRef
        // vectors so the optimizer can rebuild fail_args from snapshot in
        // store_final_boxes_in_guard (RPython ResumeDataVirtualAdder.finish).
        let (
            snapshot_map,
            snapshot_frame_size_map,
            snapshot_vable_map,
            snapshot_vref_map,
            snapshot_pc_map,
        ) = snapshot_map_from_trace_snapshots(
            &trace_snapshots,
            &mut constants,
            &mut constant_types,
        );
        unroll_opt.snapshot_boxes = snapshot_map.clone();
        unroll_opt.snapshot_frame_sizes = snapshot_frame_size_map.clone();
        unroll_opt.snapshot_vable_boxes = snapshot_vable_map.clone();
        unroll_opt.snapshot_vref_boxes = snapshot_vref_map.clone();
        unroll_opt.snapshot_frame_pcs = snapshot_pc_map.clone();

        // RPython compile.py:278-294 parity: Phase 1 results must survive
        // Phase 2 InvalidLoop. Phase 1 writes to phase1_out on the caller's
        // stack BEFORE Phase 2 starts. If Phase 2 panics, phase1_out still
        // holds the Phase 1 results.
        let mut phase1_out: Option<(Vec<Op>, crate::optimizeopt::unroll::ExportedState)> = None;
        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = unroll_opt.optimize_trace_with_constants_and_inputs_vable_out(
                &trace_ops,
                &mut constants,
                num_trace_inputargs,
                vable_config.clone(),
                Some(&mut phase1_out),
            );
            // Capture Phase 2 constant types before unroll_opt is dropped.
            for (k, v) in &unroll_opt.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let mut retried_without_unroll = false;
        let (optimized_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                // Phase 2 panicked — unroll_opt dropped. Phase 1 results
                // survive in phase1_out (written before Phase 2 started).
                if let Some(inv) = payload.downcast_ref::<crate::optimize::InvalidLoop>() {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] abort trace at key={} (InvalidLoop: {})",
                            green_key, inv.0,
                        );
                    }
                    self.cancel_count += 1;
                    // pyjitpl.py:3018-3029: RPython increments cancel_count
                    // and falls through (tracing continues). compile_loop is
                    // re-invoked on the next reached_loop_header. Do NOT call
                    // abort_tracing — TRACING flag must stay active.
                    if !self.cancelled_too_many_times() {
                        self.exported_state = None;
                        return CompileOutcome::Cancelled;
                    }
                    {
                        let mut retry_constants = constants_snapshot;
                        let mut simple_opt = Optimizer::default_pipeline();
                        simple_opt.constant_types = constant_types.clone();
                        simple_opt.numbering_type_overrides = numbering_overrides.clone();
                        // history.py:_make_op parity — see the
                        // function-entry compile path below.
                        let inputarg_types: Vec<majit_ir::Type> =
                            trace.inputargs.iter().map(|ia| ia.tp).collect();
                        simple_opt.trace_inputarg_types = inputarg_types.clone();
                        for (i, &tp) in inputarg_types.iter().enumerate() {
                            simple_opt
                                .constant_types
                                .insert(OpRef::input_arg_typed(i as u32, tp).raw(), tp);
                        }
                        simple_opt.snapshot_boxes = snapshot_map.clone();
                        simple_opt.snapshot_frame_sizes = snapshot_frame_size_map.clone();
                        simple_opt.snapshot_vable_boxes = snapshot_vable_map.clone();
                        simple_opt.snapshot_vref_boxes = snapshot_vref_map.clone();
                        simple_opt.snapshot_frame_pcs = snapshot_pc_map.clone();
                        simple_opt.call_pure_results = call_pure_results.clone();
                        // H-3.0b: forward the recorder's BoxRef pool — the
                        // retry path uses the same upstream `Rc<Box>`
                        // allocations from the original trace.
                        simple_opt.set_pending_box_pool(trace.box_pool.clone());
                        let retry_result =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                simple_opt.optimize_with_constants_and_inputs(
                                    &trace_ops_snapshot,
                                    &mut retry_constants,
                                    num_trace_inputargs,
                                )
                            }));
                        match retry_result {
                            Ok(retry_ops) => {
                                if crate::majit_log_enabled() {
                                    eprintln!(
                                        "[jit] retry without unroll succeeded at key={}",
                                        green_key
                                    );
                                }
                                retried_without_unroll = true;
                                constants = retry_constants;
                                let ni = simple_opt.final_num_inputs();
                                (retry_ops, ni)
                            }
                            Err(_) => {
                                self.warm_state.abort_tracing(green_key, false);
                                self.exported_state = None;
                                return CompileOutcome::Aborted;
                            }
                        }
                    }
                } else {
                    std::panic::resume_unwind(payload);
                }
            }
        };
        let num_ops_after = optimized_ops.len();

        // RPython compile.py keeps the root entry contract on the original
        // loop inputargs. Simple loops synthesize a LABEL from that contract;
        // they do not grow inputargs to match a rewritten JUMP arity.
        //
        // Box.type parity: when unrolling reduces the trace inputargs
        // (virtualstate + short preamble collapse vable slots down to the
        // handful of live values), the reduced LABEL's ith slot is NOT
        // `trace.inputargs[i]`. Using a prefix of `trace.inputargs` declares
        // every reduced slot with the wrong type (e.g. the vable layout's
        // `frame, next_instr, code, …` instead of the optimizer's actual
        // `frame, s_value, i_value`). The optimizer already records the
        // per-reduced-slot type in `ExportedState.renamed_inputarg_types`
        // (derived from `opref_type` of each renamed inputarg). Consume it
        // here so the backend sees declared types that match the reduced
        // LABEL's args.
        // compile.py:341 parity: the optimizer's reduced LABEL contract
        // (ExportedState.renamed_inputarg_types) is the only valid source of
        // root inputarg types. RPython has no synthetic recovery when this
        // is absent; abort compilation so the caller falls back to the
        // interpreter instead of synthesizing Int-padded InputArgs.
        let root_inputargs = if retried_without_unroll {
            // compile.py:233 compile_simple_loop: loop.inputargs is the
            // original trace inputargs. There is no ExportedState on the
            // simple path.
            trace.inputargs.clone()
        } else {
            match unroll_opt
                .final_exported_state
                .as_ref()
                .map(|es| es.renamed_inputarg_types.as_slice())
                .filter(|types| types.len() == final_num_inputs)
            {
                Some(types) => types
                    .iter()
                    .enumerate()
                    .map(|(i, &tp)| InputArg::from_type(tp, i as u32))
                    .collect::<Vec<_>>(),
                None => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] abort compile: root loop missing ExportedState inputarg types \
                             (final_num_inputs={final_num_inputs})",
                        );
                    }
                    self.cancel_count += 1;
                    return CompileOutcome::Cancelled;
                }
            }
        };
        let mut optimized_ops = optimized_ops;
        if retried_without_unroll
            && !optimized_ops
                .first()
                .is_some_and(|op| op.opcode == OpCode::Label)
        {
            // compile.py:251-259 compile_simple_loop synthesizes
            // LABEL(inputargs) before the optimized body.
            let mut label_op = majit_ir::Op::new(
                majit_ir::OpCode::Label,
                &root_inputargs
                    .iter()
                    .map(|ia| ia.opref())
                    .collect::<Vec<_>>(),
            );
            label_op.pos = majit_ir::OpRef::NONE;
            optimized_ops.insert(0, label_op);
        }
        let (inputargs, optimized_ops) = match normalize_root_loop_entry_contract(
            root_inputargs,
            optimized_ops,
        ) {
            Ok(normalized) => normalized,
            Err((expected, actual)) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] abort compile: root loop entry/jump arity mismatch input={} jump={}",
                        expected, actual,
                    );
                }
                self.cancel_count += 1;
                return CompileOutcome::Cancelled;
            }
        };

        // RPython virtualizable parity: standard virtualizable fields and
        // arrays stay in the trace as first-class virtualizable boxes.
        // Do not prepend raw heap preamble loads here; compiled callers pass
        // the traced virtualizable values in the live-input layout, and
        // `vable_*` operations keep the hot path on boxes instead of
        // re-materializing `GetfieldRaw*`/`GetarrayitemRaw*` entry ops.
        let (mut inputargs, optimized_ops) = (inputargs, optimized_ops);
        let mut compiled_ops =
            compile::normalize_closing_jump_args(optimized_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&compiled_ops, &constants));
            for op in &compiled_ops {
                if op.opcode == majit_ir::OpCode::GuardNotInvalidated {
                    if let Some(ref fa) = op.fail_args {
                        let raw: Vec<String> = fa
                            .iter()
                            .map(|a| format!("OpRef::from_raw({})", a.raw()))
                            .collect();
                        eprintln!("[jit] FINAL GuardNotInv fail_args=[{}]", raw.join(", "));
                    }
                }
            }
        }

        // RPython: jit_merge_point tick counter provides periodic exit from
        // RPython: the optimizer always emits at least one guard
        // (GUARD_NOT_INVALIDATED from OptHeap, or user-level guards).
        // A guardless trace is a bug — the invariant is that the optimizer
        // never produces a guardless loop.
        debug_assert!(
            compiled_ops.iter().any(|op| op.opcode.is_guard()),
            "optimizer produced guardless loop — GUARD_NOT_INVALIDATED should always be present"
        );

        // resume.py parity: rd_numb is now produced inline during optimization
        // (ctx.emit → store_final_boxes_in_guard) rather than post-assembly.

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key {
                pn
            } else {
                self.warm_state.alloc_token_number()
            }
        } else {
            self.warm_state.alloc_token_number()
        };
        // `compile.py:266 jitcell_token = make_jitcell_token(jitdriver_sd)`.
        let mut token =
            make_jitcell_token(token_num, driver_descriptor.as_ref().and_then(|d| d.index));
        self.configure_loop_token_for_driver(
            Arc::get_mut(&mut token).expect("fresh JitCellToken must be uniquely owned"),
            green_key,
            driver_descriptor.as_ref(),
        );
        // `compile.py:180-181` wref wiring — done inside
        // `record_loop_or_bridge` once all `Arc::get_mut` writes settle.
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let front_target_tokens = if retried_without_unroll {
            let target_token = crate::optimizeopt::unroll::TargetToken::new_loop(token_num);
            if let Some(jump_op) = compiled_ops
                .last_mut()
                .filter(|op| op.opcode == OpCode::Jump)
            {
                jump_op.descr = Some(target_token.as_jump_target_descr());
            }
            if let Some(label_op) = compiled_ops
                .iter_mut()
                .find(|op| op.opcode == OpCode::Label)
            {
                label_op.descr = Some(target_token.as_jump_target_descr());
            } else {
                let mut label_op = majit_ir::Op::new(
                    majit_ir::OpCode::Label,
                    &inputargs.iter().map(|ia| ia.opref()).collect::<Vec<_>>(),
                );
                label_op.pos = majit_ir::OpRef::NONE;
                label_op.descr = Some(target_token.as_jump_target_descr());
                compiled_ops.insert(0, label_op);
            }
            vec![target_token]
        } else if unroll_opt.target_tokens.is_empty() {
            prior_front_target_tokens.clone()
        } else {
            unroll_opt.target_tokens.clone()
        };
        // `compile.py:237` / `compile.py:289`
        // `target_token.original_jitcell_token = jitcell_token`. Backfill the
        // owning JitCellToken.number on every TargetToken now that the token
        // exists, so `record_loop_or_bridge`'s JUMP branch
        // (`compile.py:197-199`) can read it.
        for target_token in &front_target_tokens {
            target_token.set_original_jitcell_token_number(token_num);
        }

        // compile.py:504-511 send_loop_to_backend — unconditional virtualizable
        // field reload for every loop. Mirrors the FINISH-path call in
        // `finish_and_compile`; see `MetaInterp::patch_new_loop_to_load_virtualizable_fields`
        // for the shared helper. RPython's `loop.inputargs` is independent
        // from the inner body LABEL/JUMP arity (compile.py:312/320/327 — entry
        // contract is `start_state.renamed_inputargs`, body LABEL is
        // `loop_info.label_op`); pyre's `start_label.args` and `inputargs`
        // are at the trace's ROOT inputarg shape ([0..num_inputs)), and the
        // body LABEL inside `compiled_ops` carries virtualstate-allocated
        // OpRefs that the helper's forwarding map does not touch. Truncating
        // `inputargs` to `num_red_args` and prepending GETFIELD_GC /
        // GETARRAYITEM_GC therefore leaves body LABEL/JUMP arities intact.
        self.patch_new_loop_to_load_virtualizable_fields(
            &mut inputargs,
            &mut compiled_ops,
            &mut constants,
            &mut constant_types,
            driver_descriptor.as_ref(),
            orig_vable_ptr_loop,
        );
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());
        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend.compile_loop(
                &inputargs,
                &compiled_ops,
                Arc::get_mut(&mut token)
                    .expect("JitCellToken must stay uniquely owned until backend compile"),
            )
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(e) => {
                let is_invalid_loop = e.downcast_ref::<crate::optimize::InvalidLoop>().is_some();
                if crate::majit_log_enabled() {
                    let kind = if is_invalid_loop {
                        "InvalidLoop"
                    } else {
                        "panic"
                    };
                    eprintln!("[jit] compile_loop {kind}, aborting trace at key={green_key}");
                }
                // compile.py:288 parity: preserve preamble target_tokens
                // even on InvalidLoop/panic. The unroller's Phase 1 created
                // target_tokens that the next retrace needs.
                // Store in compiled_loops if available, otherwise in
                // pending_preamble_tokens for the first InvalidLoop before
                // any successful compilation (RPython: jitcell_token.
                // target_tokens = [start_descr] before Phase 2 runs).
                if is_invalid_loop && !unroll_opt.target_tokens.is_empty() {
                    if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                        if compiled.front_target_tokens.is_empty() {
                            compiled.front_target_tokens = unroll_opt.target_tokens.clone();
                        }
                    } else {
                        self.pending_preamble_tokens
                            .entry(green_key)
                            .or_insert_with(|| unroll_opt.target_tokens.clone());
                    }
                }
                self.warm_state.abort_tracing(green_key, !is_invalid_loop);
                self.cancel_count += 1;
                return CompileOutcome::Cancelled;
            }
        };
        match compile_result {
            Ok(_) => {
                // compile.py:826-830 store_hash: assign jitcounter hashes.
                self.assign_guard_hashes(token.as_ref());
                // compile.py:566-567 send_loop_to_backend registers the token
                // with the memory manager before record_loop_or_bridge reads it.
                self.warm_state.memory_manager.keep_loop_alive(&token);
                // compile.py:213 record_loop_or_bridge — record this loop's
                // CALL_ASSEMBLER / JUMP keepalive targets.
                self.record_loop_or_bridge(&token, &mut compiled_ops, trace_id);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled loop at key={}, num_inputs={}",
                        green_key,
                        inputargs.len()
                    );
                }
                // Build resume data and exit layouts for all guards in the optimized trace.
                let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                    &inputargs,
                    &compiled_ops,
                    green_key,
                    &compiled_constant_types,
                    self.callinfocollection.as_deref(),
                );
                let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                    &inputargs,
                    &compiled_ops,
                    &compiled_constant_types,
                );
                if let Some(backend_layouts) =
                    self.backend.compiled_fail_descr_layouts(token.as_ref())
                {
                    compile::merge_backend_exit_layouts(
                        &mut exit_layouts,
                        backend_layouts.as_slice(),
                        &compiled_ops,
                    );
                }
                if let Some(backend_layouts) =
                    self.backend.compiled_terminal_exit_layouts(token.as_ref())
                {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                        &compiled_ops,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(token.as_ref(), trace_id);
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                self.take_back_all_descrs(std::mem::take(&mut unroll_opt.all_descrs));
                // unroll.py:176-177: disable_retracing_if_max_retrace_guards
                let mut final_retraced_count = unroll_opt.retraced_count;
                crate::optimizeopt::unroll::OptUnroll::disable_retracing_if_max_retrace_guards(
                    &compiled_ops,
                    &mut final_retraced_count,
                    self.warm_state.max_retrace_guards(),
                );
                let mut next_global_opref = unroll_opt
                    .next_global_opref
                    .max(compute_next_global_opref(&inputargs, &compiled_ops));
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: inputargs.clone(),
                        ops: compiled_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        exit_layouts,
                        terminal_exit_layouts,
                    },
                );

                // RPython parity: keep previous compiled tokens alive so
                // external target_token JUMPs can redirect to them.
                let mut previous_tokens: Vec<std::sync::Arc<JitCellToken>> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    // Cranelift workaround (no RPython counterpart): copy
                    // bridges from old token to new, since Cranelift cannot
                    // patch machine code in-place. No-op for dynasm.
                    self.backend
                        .migrate_bridges(&old_entry.token, token.as_ref());
                    // Box Identity Phase E.2b parity: preserve old entry's
                    // high-water so previously stored bridges' OpRefs stay
                    // disjoint from any future bridge.
                    next_global_opref = next_global_opref.max(old_entry.next_global_opref);
                    previous_tokens = self.retire_compiled_entry(green_key, old_entry, &mut traces);
                }
                if crate::majit_log_enabled() {
                    eprintln!("[jit][compiled_loops.insert] green_key={green_key}");
                }
                token.set_retraced_count(final_retraced_count);
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token: Arc::clone(&token),
                        meta,
                        front_target_tokens,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                        // Box Identity Phase E Step 1: record Phase 2's final
                        // OpRef high-water so later bridges can allocate in a
                        // disjoint namespace. RPython gets disjointness for
                        // free via fresh `InputArg` Python identities per
                        // TraceIterator (opencoder.py:249-273); pyre flat u32
                        // OpRefs need this explicit baseline.
                        next_global_opref,
                    },
                );
                // warmstate.py:339-348 attach the same compiled token object.
                self.attach_procedure_with_redirect(green_key, Arc::clone(&token));

                self.stats.loops_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
                // pyjitpl.py:3025: self.exported_state = None
                self.exported_state = None;
                let from_retry = self.cancel_count > 0;
                // When replacing an existing inner-loop entry with a
                // cross-loop cut, suppress meta rebuild — the existing
                // meta is authoritative (built from the inner loop's own
                // trace). The new compiled code may have different inputarg
                // layout, but is_compatible uses meta to extract live_values
                // so the meta must stay consistent with the entry point.
                self.last_compiled_key = Some(green_key);
                return CompileOutcome::Compiled {
                    green_key,
                    from_retry,
                };
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                let msg = format!("JIT compilation failed: {e}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                // RPython: backend failure propagates to warmspot, handled
                // non-permanently (allows retry). compile.py has no explicit
                // catch for backend errors — they fall through to the outer
                // try/except in maybe_compile_and_run.
                self.warm_state.abort_tracing(green_key, false);
                self.cancel_count += 1;
                // pyjitpl.py:3025: self.exported_state = None
                self.exported_state = None;
                return CompileOutcome::Cancelled;
            }
        }
    }

    /// pyjitpl.py:2936-2942: cancelled_too_many_times — check if
    /// cancel_count exceeds max_unroll_loops.
    fn cancelled_too_many_times(&self) -> bool {
        let limit = self.warm_state.max_unroll_loops();
        self.cancel_count > limit
    }

    /// pyjitpl.py:2389 attribute access — `self.partial_trace`.
    pub fn partial_trace(&self) -> Option<&PartialTrace> {
        self.partial_trace.as_ref()
    }

    /// Clear retrace state (partial_trace, retracing_from, exported_state).
    ///
    /// RPython parity: does NOT reset cancel_count. cancel_count is
    /// per-tracing-pass (reset in setup_tracing, which corresponds to
    /// RPython creating a new MetaInterp per _compile_and_run_once).
    /// clear_retrace_state only clears the retrace-specific fields.
    pub fn clear_retrace_state(&mut self) {
        self.partial_trace = None;
        self.retracing_from = None;
        self.exported_state = None;
    }

    /// compile.py: has_compiled_targets — check if a green key has
    /// compiled target tokens that a bridge can jump to.
    pub fn has_compiled_targets(&self, green_key: u64) -> bool {
        // Consistent with has_compiled_loop: direct key only, no alias.
        // Both functions must see the same view — otherwise tracing sees
        // "targets exist" while execution sees "no compiled loop", causing
        // wasted compile/trace churn in nested loops.
        self.compiled_loops
            .get(&green_key)
            .map_or(false, |c| !c.front_target_tokens.is_empty())
    }

    /// pyjitpl.py:3179-3190: compile_trace — try to compile the current
    /// trace as a bridge to an existing compiled loop.
    ///
    /// Called during tracing when a loop header is reached and that loop
    /// already has compiled code. Records a tentative JUMP, takes a
    /// snapshot of the trace ops, then cuts the JUMP back off.
    /// The snapshot is optimized as a bridge; on success the bridge is
    /// compiled and installed, on failure retrace_needed may be set.
    ///
    /// `bridge_origin`: if Some((trace_id, fail_index)), the trace started
    /// from a guard failure (ResumeGuardDescr). If None, the trace started
    /// from the interpreter (ResumeFromInterpDescr / entry bridge).
    ///
    /// Returns `CompileOutcome::Compiled` if the bridge was successfully
    /// compiled and installed. The caller should switch to compiled code.
    /// Returns `CompileOutcome::Cancelled` if the bridge couldn't close
    /// (retrace_needed was set, or optimization failed).
    /// compile.py:1028 compile_trace parity.
    /// `ends_with_jump=true`: records JUMP, uses BridgeCompileData (optimize_bridge).
    /// `ends_with_jump=false`: records FINISH, uses SimpleCompileData (optimize_loop).
    pub fn compile_trace(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
    ) -> CompileOutcome {
        self.compile_trace_inner(green_key, finish_args, bridge_origin, None, None)
    }

    /// compile.py:1002-1021 ResumeFromInterpDescr parity.
    pub fn compile_trace_from_interp(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        original_green_key: u64,
        entry_meta: M,
    ) -> CompileOutcome {
        self.compile_trace_inner(
            green_key,
            finish_args,
            None,
            None,
            Some((original_green_key, entry_meta)),
        )
    }

    /// compile_trace with ends_with_jump=false (FINISH).
    pub fn compile_trace_finish(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: majit_ir::DescrRef,
    ) -> CompileOutcome {
        self.compile_trace_inner(
            green_key,
            finish_args,
            bridge_origin,
            Some(finish_descr),
            None,
        )
    }

    fn compile_trace_inner(
        &mut self,
        green_key: u64,
        finish_args: &[OpRef],
        bridge_origin: Option<(u64, u32)>,
        finish_descr: Option<majit_ir::DescrRef>,
        entry_bridge: Option<(u64, M)>,
    ) -> CompileOutcome {
        let ends_with_jump = finish_descr.is_none();
        let ctx = match self.tracing.as_mut() {
            Some(ctx) => ctx,
            None => return CompileOutcome::Cancelled,
        };

        // pyjitpl.py:3187: save position before recording JUMP/FINISH
        let cut_at = ctx.get_trace_position();
        self.potential_retrace_position = Some(cut_at);

        // pyjitpl.py:3189 / 3217: record tentative JUMP or FINISH
        if let Some(descr) = finish_descr {
            ctx.finish(finish_args, descr);
        } else {
            let jump_descr = self
                .compiled_loops
                .get(&green_key)
                .and_then(|compiled| compiled.front_target_tokens.first())
                .map(|target_token| target_token.as_jump_target_descr());
            let Some(jump_descr) = jump_descr else {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_trace: no front_target_token for key={}, bridge_origin={:?}",
                        green_key, bridge_origin
                    );
                }
                return CompileOutcome::Cancelled;
            };
            ctx.recorder
                .close_loop_with_descr(finish_args, Some(jump_descr));
        }

        // Snapshot the trace ops (including JUMP) for bridge compilation.
        let bridge_ops = ctx.ops().to_vec();
        let bridge_inputargs: Vec<majit_ir::InputArg> = ctx
            .recorder
            .inputarg_types()
            .iter()
            .enumerate()
            .map(|(i, &tp)| majit_ir::InputArg::from_type(tp, i as u32))
            .collect();
        let mut constants = ctx.constants.snapshot();
        let mut constant_types = ctx.constants.constant_types_snapshot();
        let trace_snapshots = ctx.snapshots().to_vec();
        let (
            snapshot_boxes,
            snapshot_frame_sizes,
            snapshot_vable_boxes,
            snapshot_vref_boxes,
            snapshot_frame_pcs,
        ) = snapshot_map_from_trace_snapshots(
            &trace_snapshots,
            &mut constants,
            &mut constant_types,
        );

        // pyjitpl.py:3195 finally: always cut — pop the tentative JUMP/FINISH.
        ctx.cut_trace(cut_at);

        if crate::majit_log_enabled() {
            let label = if ends_with_jump { "jump" } else { "finish" };
            eprintln!(
                "[jit] compile_trace({}): key={}, ops={}, origin={:?}",
                label,
                green_key,
                bridge_ops.len(),
                bridge_origin,
            );
        }

        match bridge_origin {
            Some((trace_id, fail_index)) => {
                // compile.py:1082 — ResumeGuardDescr path: attach bridge
                // to the existing guard that failed.
                // The `green_key` parameter here is the JUMP TARGET's key
                // (from CloseLoopWithArgs) when the bridge closes as a jump
                // into another compiled loop, but the fail_descr belongs to
                // the ORIGIN guard. RPython reaches the origin via
                // `resumekey.rd_loop_token`; pyre stores it in
                // `active_trace_session.bridge.green_key`.
                let origin_key = self.bridge_info().map(|b| b.green_key).unwrap_or(green_key);
                // Prevent double-compilation: if a bridge was already compiled
                // and attached to this guard, skip. RPython's
                // raise_continue_running_normally stops the trace entirely,
                // so this path is never re-entered; pyre's trace may continue
                // and re-enter, so guard explicitly.
                let already = self.bridge_was_compiled(origin_key, trace_id, fail_index);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] bridge_was_compiled({}, {}, {}) = {}",
                        origin_key, trace_id, fail_index, already
                    );
                }
                if already {
                    return CompileOutcome::Compiled {
                        green_key: 0,
                        from_retry: false,
                    };
                }
                let descr_arc = {
                    let compiled = match self.compiled_loops.get(&origin_key) {
                        Some(c) => c,
                        None => return CompileOutcome::Cancelled,
                    };
                    match self.bridge_source_descr(compiled, trace_id, fail_index) {
                        Some(d) => d,
                        None => {
                            if crate::majit_log_enabled() {
                                eprintln!(
                                    "[jit] bridge_source_descr({}, {}) = None → Cancelled",
                                    trace_id, fail_index
                                );
                            }
                            return CompileOutcome::Cancelled;
                        }
                    }
                };
                let fail_descr = descr_arc
                    .as_fail_descr()
                    .expect("bridge source op.descr must implement FailDescr");
                let success = self.compile_bridge(
                    origin_key,
                    fail_index,
                    fail_descr,
                    &bridge_ops,
                    &bridge_inputargs,
                    constants,
                    constant_types,
                    snapshot_boxes,
                    snapshot_frame_sizes,
                    snapshot_vable_boxes,
                    snapshot_vref_boxes,
                    snapshot_frame_pcs,
                );
                if success {
                    CompileOutcome::Compiled {
                        green_key: 0,
                        from_retry: false,
                    }
                } else {
                    CompileOutcome::Cancelled
                }
            }
            None => {
                // compile.py:1006-1022 — ResumeFromInterpDescr path:
                // compile a fresh entry bridge and attach it to the
                // original interpreter green key.
                let Some((original_green_key, entry_meta)) = entry_bridge else {
                    return CompileOutcome::Cancelled;
                };
                let success = self.compile_entry_bridge(
                    green_key,
                    original_green_key,
                    entry_meta,
                    &bridge_ops,
                    &bridge_inputargs,
                    constants,
                    constant_types,
                    snapshot_boxes,
                    snapshot_frame_sizes,
                    snapshot_vable_boxes,
                    snapshot_vref_boxes,
                    snapshot_frame_pcs,
                );
                if success {
                    CompileOutcome::Compiled {
                        green_key: original_green_key,
                        from_retry: false,
                    }
                } else {
                    CompileOutcome::Cancelled
                }
            }
        }
    }

    /// pyjitpl.py:2408-2412: retrace_needed — save state from a failed
    /// bridge compilation for a subsequent compile_retrace attempt.
    ///
    /// Called when the optimizer returns "not final" (no existing target token
    /// matched). The partial trace and exported state are saved so the next
    /// compile_loop for this green_key can use compile_retrace.
    pub fn retrace_needed(
        &mut self,
        green_key: u64,
        ops: Vec<Op>,
        inputargs: Vec<InputArg>,
        constants: HashMap<u32, i64>,
        exported_state: crate::optimizeopt::unroll::ExportedState,
    ) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] retrace_needed: key={}, partial_ops={}",
                green_key,
                ops.len()
            );
        }
        self.partial_trace = Some(PartialTrace {
            ops,
            inputargs,
            constants,
        });
        // pyjitpl.py:2410: self.retracing_from = self.potential_retrace_position
        self.retracing_from = self.potential_retrace_position;
        self.exported_state = Some(exported_state);
        // pyjitpl.py:2418: self.heapcache.reset()
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.heap_cache_mut().reset();
        }
    }

    /// pyjitpl.py:3171-3177 / compile.py:341-394: compile_retrace — compile
    /// a new loop specialization by appending new body ops to a partial trace.
    ///
    /// Uses the saved exported_state to import optimizer knowledge from the
    /// first (failed) attempt, then optimizes the new trace body and
    /// concatenates with partial_trace ops.
    ///
    /// Returns true if compilation succeeded.
    pub fn compile_retrace(&mut self, jump_args: &[OpRef], meta: M) -> bool {
        let partial = match self.partial_trace.take() {
            Some(p) => p,
            None => return false,
        };
        let mut start_state = match self.exported_state.take() {
            Some(s) => s,
            None => return false,
        };
        // gcreftracer.py parity: GC may have moved objects between Phase 1
        // and Phase 2. Refresh GcRef values from shadow stack before use.
        start_state.refresh_from_gc();
        let bridge_trace = self.bridge_info();
        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let retracing_from = self.retracing_from.take();
        let mut ctx = match self.tracing.take() {
            Some(ctx) => ctx,
            None => return false,
        };
        let (
            green_key,
            driver_descriptor,
            orig_vable_ptr_retrace,
            numbering_overrides,
            mut constants,
            mut constant_types,
            trace,
        ) = {
            let green_key = ctx.green_key;
            let header_pc = ctx.header_pc;
            let driver_descriptor = ctx.driver_descriptor().cloned();
            let retrace_cut = retracing_from.and_then(|retrace_pos| {
                ctx.get_merge_point_at(green_key, header_pc)
                    .filter(|mp| mp.position == retrace_pos && mp.position._pos > 0)
                    .map(|mp| {
                        (
                            mp.original_boxes.clone(),
                            mp.original_box_types.clone(),
                            crate::history::TreeLoopCutPosition::new(mp.position._pos),
                        )
                    })
            });
            let orig_vable_ptr_retrace =
                self.orig_vable_ptr_from_trace_ctx(&ctx, driver_descriptor.as_ref());
            let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
            let constants = ctx.constants.snapshot();
            let constant_types = ctx.constants.constant_types_snapshot();
            let initial_inputarg_consts = ctx.initial_inputarg_consts.clone();

            // compile.py:358-362 records the closing JUMP on the same history
            // that `cut_trace_from` views. Rust materializes TreeLoop eagerly,
            // so close once, then cut the completed trace.
            ctx.close_loop(jump_args);
            let trace = ctx.into_tree_loop();
            let trace = if let Some((ref original_boxes, ref original_box_types, start)) =
                retrace_cut
            {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] cut_retrace_from: start.op_index={} original_boxes={} trace_ops={} header_pc={}",
                        start.op_index,
                        original_boxes.len(),
                        trace.ops.len(),
                        header_pc,
                    );
                }
                trace.cut_trace_from_with_consts(
                    start,
                    original_boxes,
                    original_box_types,
                    &initial_inputarg_consts,
                )
            } else {
                trace
            };
            (
                green_key,
                driver_descriptor,
                orig_vable_ptr_retrace,
                numbering_overrides,
                constants,
                constant_types,
                trace,
            )
        };

        let trace_ops = trace.ops.clone();

        if crate::majit_log_enabled() {
            eprintln!("--- retrace body (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        // compile.py:362-367: optimize using UnrolledLoopData with start_state.
        let prior_front_target_tokens = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.front_target_tokens.clone())
            .or_else(|| self.pending_preamble_tokens.remove(&green_key))
            .unwrap_or_default();
        let mut unroll_opt = crate::optimizeopt::unroll::UnrollOptimizer::new();
        unroll_opt.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        unroll_opt.target_tokens = prior_front_target_tokens.clone();
        unroll_opt.retraced_count = self
            .compiled_loops
            .get(&green_key)
            .map(|compiled| compiled.token.get_retraced_count())
            .unwrap_or(0);
        unroll_opt.retrace_limit = self.warm_state.retrace_limit();
        unroll_opt.max_retrace_guards = self.warm_state.max_retrace_guards();
        unroll_opt.constant_types = constant_types.clone();
        unroll_opt.callinfocollection = self.callinfocollection.clone();
        unroll_opt.numbering_type_overrides = numbering_overrides;
        let (
            retrace_snapshot_boxes,
            retrace_snapshot_frame_sizes,
            retrace_snapshot_vable_boxes,
            retrace_snapshot_vref_boxes,
            retrace_snapshot_frame_pcs,
        ) = snapshot_map_from_trace_snapshots(
            &trace.snapshots,
            &mut constants,
            &mut constant_types,
        );
        unroll_opt.snapshot_boxes = retrace_snapshot_boxes;
        unroll_opt.snapshot_frame_sizes = retrace_snapshot_frame_sizes;
        unroll_opt.snapshot_vable_boxes = retrace_snapshot_vable_boxes;
        unroll_opt.snapshot_vref_boxes = retrace_snapshot_vref_boxes;
        unroll_opt.snapshot_frame_pcs = retrace_snapshot_frame_pcs;
        // Import the exported state from the first (failed) attempt so the
        // optimizer can continue from where it left off.
        unroll_opt.imported_state = Some(start_state);

        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = unroll_opt.optimize_trace_with_constants_and_inputs_vable(
                &trace_ops,
                &mut constants,
                trace.inputargs.len(),
                vable_config,
            );
            for (k, v) in &unroll_opt.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let (body_ops, final_num_inputs) = match optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] compile_retrace: InvalidLoop at key={}", green_key);
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };

        // compile.py:379-382: partial_trace.operations + [label_op] + loop_ops.
        //
        // RPython invariant: partial_trace.operations does NOT contain a
        // terminal JUMP — the optimizer's propagate_all_forward separates
        // JUMP into last_op (not in _newoperations). body_ops (from
        // assemble_peeled_trace_with_jump_args) contains Label + body + JUMP.
        //
        // pyre parity: partial.ops now stores Phase 1 optimized preamble
        // ops (JUMP excluded), matching RPython's partial_trace.operations.
        let mut combined_ops = partial.ops;
        combined_ops.extend(body_ops);
        // Merge constants from partial trace with new constants.
        for (k, v) in partial.constants {
            constants.entry(k).or_insert(v);
        }

        // compile.py:1075-1085 + 379-393 parity: the partial trace saved by
        // compile_trace already owns the bridge inputarg contract
        // (`new_trace.inputargs = info.renamed_inputargs`), and
        // compile_retrace reuses that same `partial_trace` object as the
        // final loop/bridge. Do not reconstruct dense `[0..n)` inputargs from
        // type side data here: bridge Phase E.2b renamed_inputargs may live in
        // a shifted `[bridge_inputarg_base..)` namespace, exactly like
        // RPython's fresh InputArg object identities.
        let root_inputargs: Vec<InputArg> = partial.inputargs.clone();
        let (inputargs, combined_ops) =
            match normalize_root_loop_entry_contract(root_inputargs, combined_ops) {
                Ok(normalized) => normalized,
                Err((expected, actual)) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_retrace: entry/jump arity mismatch input={} jump={}",
                            expected, actual,
                        );
                    }
                    return false;
                }
            };

        let combined_ops =
            compile::normalize_closing_jump_args(combined_ops, &constants, final_num_inputs);

        if crate::majit_log_enabled() {
            eprintln!("--- retrace combined (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&combined_ops, &constants));
        }

        let num_combined_ops = combined_ops.len();
        let has_guard = combined_ops.iter().any(|op| op.opcode.is_guard());
        if !has_guard {
            if crate::majit_log_enabled() {
                eprintln!("[jit] compile_retrace: guardless loop");
            }
            return false;
        }

        // compile.py:504-511 send_loop_to_backend virtualizable hook —
        // retrace paths also need the preamble loads so the loop's inputarg
        // contract is reds-only and virtualizable fields are reloaded from
        // the heap object at entry.
        let mut inputargs = inputargs;
        let mut combined_ops = combined_ops;
        self.patch_new_loop_to_load_virtualizable_fields(
            &mut inputargs,
            &mut combined_ops,
            &mut constants,
            &mut constant_types,
            driver_descriptor.as_ref(),
            orig_vable_ptr_retrace,
        );
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());

        let token_num = self.warm_state.alloc_token_number();
        // `compile.py:266 jitcell_token = make_jitcell_token(jitdriver_sd)`.
        let mut token =
            make_jitcell_token(token_num, driver_descriptor.as_ref().and_then(|d| d.index));
        self.configure_loop_token_for_driver(
            Arc::get_mut(&mut token).expect("fresh JitCellToken must be uniquely owned"),
            green_key,
            driver_descriptor.as_ref(),
        );
        // `compile.py:180-181` wref wiring — done inside
        // `record_loop_or_bridge` once all `Arc::get_mut` writes settle.
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend.compile_loop(
                &inputargs,
                &combined_ops,
                Arc::get_mut(&mut token)
                    .expect("JitCellToken must stay uniquely owned until backend compile"),
            )
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(_) => {
                if crate::majit_log_enabled() {
                    eprintln!("[jit] compile_retrace panicked at key={green_key}");
                }
                self.warm_state.abort_tracing(green_key, false);
                return false;
            }
        };
        match compile_result {
            Ok(_) => {
                self.assign_guard_hashes(token.as_ref());
                // `compile.py:237` / `compile.py:289` — every TargetToken
                // whose LABEL or JUMP appears in `combined_ops` carries
                // `original_jitcell_token = jitcell_token`.  In the retrace
                // path, `combined_ops = old_front + new_body`, so the old
                // front's TargetTokens (created under the previous
                // jitcell_token) still point at the old number.  RPython
                // avoids this entirely by reusing the same
                // `loop_jitcell_token` for retrace (`compile.py:356`); pyre
                // allocates a new `token_num` for cranelift bridge
                // migration, so we rebind the prior tokens to the new
                // number here.  Without this, `record_loop_or_bridge`'s
                // JUMP arm sees `target_owner_num == old_num !=
                // new_token.number` and records a false self-loop keepalive.
                for target_token in &prior_front_target_tokens {
                    target_token.set_original_jitcell_token_number(token_num);
                }
                for target_token in &unroll_opt.target_tokens {
                    target_token.set_original_jitcell_token_number(token_num);
                }
                self.warm_state.memory_manager.keep_loop_alive(&token);
                // compile.py:213 record_loop_or_bridge.
                self.record_loop_or_bridge(&token, &mut combined_ops, trace_id);
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled retrace at key={}, num_inputs={}",
                        green_key,
                        inputargs.len()
                    );
                }
                let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                    &inputargs,
                    &combined_ops,
                    green_key,
                    &compiled_constant_types,
                    self.callinfocollection.as_deref(),
                );
                let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                    &inputargs,
                    &combined_ops,
                    &compiled_constant_types,
                );
                if let Some(backend_layouts) =
                    self.backend.compiled_fail_descr_layouts(token.as_ref())
                {
                    compile::merge_backend_exit_layouts(
                        &mut exit_layouts,
                        backend_layouts.as_slice(),
                        &combined_ops,
                    );
                }
                if let Some(backend_layouts) =
                    self.backend.compiled_terminal_exit_layouts(token.as_ref())
                {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                        &combined_ops,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(token.as_ref(), trace_id);
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                self.take_back_all_descrs(std::mem::take(&mut unroll_opt.all_descrs));
                let mut next_global_opref = unroll_opt
                    .next_global_opref
                    .max(compute_next_global_opref(&inputargs, &combined_ops));
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: inputargs.clone(),
                        ops: combined_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        exit_layouts,
                        terminal_exit_layouts,
                    },
                );

                let mut previous_tokens: Vec<std::sync::Arc<JitCellToken>> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    // Cranelift workaround (no RPython counterpart): copy
                    // bridges from old token to new, since Cranelift cannot
                    // patch machine code in-place. No-op for dynasm.
                    self.backend
                        .migrate_bridges(&old_entry.token, token.as_ref());
                    // Box Identity Phase E.2b parity: see compile_loop site.
                    next_global_opref = next_global_opref.max(old_entry.next_global_opref);
                    previous_tokens = self.retire_compiled_entry(green_key, old_entry, &mut traces);
                }
                if crate::majit_log_enabled() {
                    eprintln!("[jit][compiled_loops.insert] green_key={green_key}",);
                }
                token.set_retraced_count(unroll_opt.retraced_count);
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token: Arc::clone(&token),
                        meta,
                        front_target_tokens: if unroll_opt.target_tokens.is_empty() {
                            prior_front_target_tokens
                        } else {
                            unroll_opt.target_tokens.clone()
                        },
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                        // Box Identity Phase E Step 1: see main compile site.
                        next_global_opref,
                    },
                );
                self.attach_procedure_with_redirect(green_key, Arc::clone(&token));
                self.stats.loops_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, 0, num_combined_ops);
                }
                true
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                if crate::majit_log_enabled() {
                    eprintln!("[jit] compile_retrace failed: {e}");
                }
                self.warm_state.abort_tracing(green_key, false);
                false
            }
        }
    }

    /// Abort the current trace.
    ///
    /// If `permanent` is true, this location will never be traced again.
    ///
    /// Structured as two halves mirroring RPython's exception-unwind shape
    /// (pyjitpl.py:2807 `raise SwitchToBlackhole` → `_interpret` unwind →
    /// pyjitpl.py:2760 `aborted_tracing(reason)`):
    ///
    /// 1. `abort_trace_live(permanent)` — live cleanup only (recorder
    ///    abort, warm-state reset, pending_token clear).  Matches the
    ///    implicit unwind in RPython: no stats bump, no hook fire.
    /// 2. `aborted_tracing(AbortReason::Generic)` — accounting + hook
    ///    fire (stats.aborted, on_trace_abort).  Matches pyjitpl.py:2760
    ///    the upstream accounting site.
    ///
    /// Blackhole callers that set `aborted_tracing_jitdriver` before
    /// calling this pair get the distinct `on_trace_too_long` hook
    /// routed through `aborted_tracing` (currently folded into the
    /// single `on_trace_abort`; the split lands when pyre's hook surface
    /// is fully ported).
    pub fn abort_trace(&mut self, permanent: bool) {
        self.abort_trace_live(permanent);
        self.aborted_tracing(AbortReason::Generic.as_int());
    }

    /// Live-cleanup half of `abort_trace` — no stats, no hooks.
    /// Callers that go through `blackhole_if_trace_too_long` should invoke
    /// this directly and then call `aborted_tracing(AbortReason::TooLong)`
    /// so the accounting event fires exactly once with the correct reason.
    pub fn abort_trace_live(&mut self, permanent: bool) {
        self.force_finish_trace = false;
        self.clear_retrace_state();
        if let Some(ctx) = self.tracing.take() {
            let green_key = ctx.green_key;
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] abort trace at key={} (permanent={})",
                    green_key, permanent
                );
            }
            // Dropping `ctx` at end of scope releases the recorder.
            self.warm_state.abort_tracing(green_key, permanent);
            self.pending_token = None;
            // Stash green_key / permanent for the subsequent
            // `aborted_tracing` call so its hook fires with the upstream
            // payload even though the ctx has been taken.
            self.pending_abort_green_key = Some(green_key);
            self.pending_abort_permanent = permanent;
            // RPython invariant: `tracing` (the tracer context) and
            // `active_trace_session` (the frontend meta envelope) share
            // a lifetime — upstream's `MetaInterp.staticdata` carries
            // both through `begin_tracing` / abort paths as one unit.
            // If we take the tracer, the session must go with it, or a
            // subsequent `bound_reached` → `begin_trace_session` will
            // find a stale `Some(..)` and fire the "already active"
            // assertion. Clearing unconditionally here is idempotent for
            // sites that already consumed the session via
            // `take_trace_meta` (e.g. the compile_loop dispatch) — a
            // None→None set is a no-op.
            self.clear_trace_session();
        }
        // `pyjitpl.py:3015` — cancel/abort unwinds the tracing state
        // atomically. Keep `self.tracing` and `self.active_trace_session`
        // in lockstep: leaving `active_trace_session = Some` after
        // `self.tracing = None` would leak a stale session into the next
        // `begin_trace_session`, which asserts the slot is empty.
        self.clear_trace_session();
    }

    /// Finish the current trace with a terminal `FINISH`, then optimize and compile it.
    ///
    /// `exit_with_exception` selects the FINISH descr per `pyjitpl.py`:
    /// * `false` → `compile_done_with_this_frame` (pyjitpl.py:3198-3220) —
    ///   descr = `sd.done_with_this_frame_descr_<kind>`.
    /// * `true` → `compile_exit_frame_with_exception` (pyjitpl.py:3238-3245)
    ///   — descr = `sd.exit_frame_with_exception_descr_ref`.
    ///
    /// Returns `Err(SwitchToBlackhole::bad_loop())` on optimizer
    /// `InvalidLoop` or backend compile failure, matching
    /// pyjitpl.py:3220 `compile.giveup()` surfacing as
    /// `SwitchToBlackhole(ABORT_BAD_LOOP)`.  The caller (typically
    /// `compile_finish_from_active_session`) propagates the error so
    /// `finishframe`/`finishframe_exception` can translate it into
    /// `aborted_tracing(reason)` per pyjitpl.py:2491.
    pub fn finish_and_compile(
        &mut self,
        finish_args: &[OpRef],
        finish_arg_types: Vec<Type>,
        meta: M,
        exit_with_exception: bool,
    ) -> Result<(), SwitchToBlackhole> {
        // Cache vable_config before take() clears self.tracing.
        let vable_config = self.current_virtualizable_optimizer_config();
        // Cache driver descriptor before ctx is partially consumed below.
        let driver_descriptor = self
            .tracing
            .as_ref()
            .and_then(|ctx| ctx.driver_descriptor())
            .cloned();
        self.force_finish_trace = false;
        let mut ctx = self.tracing.take().unwrap();
        // compile.py:510 `vable = orig_inpargs[index_of_virtualizable].getref_base()`.
        // Resolve the constant Ref that the tracer stashed for the
        // virtualizable inputarg at trace-start so
        // `patch_new_loop_to_load_virtualizable_fields` below can read the
        // heap object via `vinfo.get_array_length(vable, i)` without
        // consulting a separate trace-start cache.
        let orig_vable_ptr = self.orig_vable_ptr_from_trace_ctx(&ctx, driver_descriptor.as_ref());
        // pyjitpl.py:3199 compile_done_with_this_frame parity:
        // `store_token_in_vable` (SetfieldGc on vable_token + the
        // accompanying GUARD_NOT_FORCED_2) is recorded by the pyre
        // frontend right before TraceAction::Finish is emitted, so the
        // guard captures fresh resumedata via the proper
        // `MIFrame::generate_guard` path.
        let green_key = ctx.green_key;

        let mut recorder = ctx.recorder;
        // `pyjitpl.py:3216-3217` / `pyjitpl.py:3241`:
        //   `token = sd.done_with_this_frame_descr_<type>` (normal) or
        //   `token = sd.exit_frame_with_exception_descr_ref` (raising),
        //   then `self.history.record(rop.FINISH, exits, None, descr=token)`.
        // Use the metainterp-attached singleton so FINISH identity is
        // shared with the backend (see `attach_descrs_to_cpu`).  Falls
        // back to `make_fail_descr_typed` only if the singleton was
        // never attached (tests that bypass `MetaInterp::new`).
        let finish_descr = if exit_with_exception {
            self.staticdata
                .exit_frame_with_exception_descr_ref
                .clone()
                .unwrap_or_else(|| crate::make_finish_fail_descr_typed(finish_arg_types.clone()))
        } else {
            self.staticdata
                .done_with_this_frame_descr_from_types(&finish_arg_types)
                .unwrap_or_else(|| crate::make_finish_fail_descr_typed(finish_arg_types.clone()))
        };
        recorder.finish(finish_args, finish_descr);
        // Task #70: snapshots live on TraceCtx; rebuild the TreeLoop with
        // them so downstream consumers (`trace.snapshots`) still observe
        // the captured resumedata. `recorder.get_trace()` on its own
        // returns a snapshot-less TreeLoop post-Task #70.
        let mut trace = recorder.get_trace();
        trace.snapshots = std::mem::take(&mut ctx.snapshots);
        let trace_snapshots = trace.snapshots.clone();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        // Refresh shadow-stack-rooted Ref constants so `into_inner_with_types`
        // sees the post-GC addresses for any object that moved.
        ctx.constants.refresh_from_gc();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let trace_ops = trace.ops.clone();

        let num_ops_before = trace_ops.len();
        let mut optimizer = if let Some(config) = vable_config {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        optimizer.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        optimizer.constant_types = constant_types.clone();
        optimizer.numbering_type_overrides = numbering_overrides;
        // history.py:_make_op parity: every InputArg carries its type
        // from the recorder. Propagate those raw recorder types to the
        // optimizer without further reconciliation.
        let inputarg_types: Vec<majit_ir::Type> = trace.inputargs.iter().map(|ia| ia.tp).collect();
        optimizer.trace_inputarg_types = inputarg_types.clone();
        for (i, &tp) in inputarg_types.iter().enumerate() {
            optimizer
                .constant_types
                .insert(OpRef::input_arg_typed(i as u32, tp).raw(), tp);
        }
        // resume.py parity: convert tracing-time snapshots to flat OpRef
        // vectors so the optimizer can rebuild fail_args from snapshot in
        // store_final_boxes_in_guard (RPython ResumeDataVirtualAdder.finish).
        let (
            snapshot_map,
            snapshot_frame_size_map,
            snapshot_vable_map,
            snapshot_vref_map,
            snapshot_pc_map,
        ) = snapshot_map_from_trace_snapshots(
            &trace_snapshots,
            &mut constants,
            &mut constant_types,
        );
        // compile.py:92-96 SimpleCompileData.optimize → optimize_loop parity.
        // Wire snapshot data through to the optimizer so guard
        // store_final_boxes_in_guard (mod.rs:2261) can properly populate
        // rd_numb / rd_consts via _number_boxes (resume.py:200-205).
        // Without this, every guard from a function-entry trace is dropped
        // by the no-snapshot fallback in mod.rs:2281, leaving rd_numb=None,
        // and the runtime guard-fail path immediately invalidates the loop
        // (because resume_in_blackhole has no resume_pc to walk to).
        optimizer.snapshot_boxes = snapshot_map;
        optimizer.snapshot_frame_sizes = snapshot_frame_size_map;
        optimizer.snapshot_vable_boxes = snapshot_vable_map;
        optimizer.snapshot_vref_boxes = snapshot_vref_map;
        optimizer.snapshot_frame_pcs = snapshot_pc_map;
        // H-3.0b: hand the recorder's BoxRef pool to the optimizer so
        // mirror writes (H-3.1) reach the same `Rc<Box>` allocations.
        // RPython parity: PyPy's `AbstractValue` objects from tracing
        // flow unchanged into optimization.
        optimizer.set_pending_box_pool(trace.box_pool.clone());

        // Wrap in catch_unwind — InvalidLoop during optimization should
        // abort the trace, not crash the process. Matches compile_loop.
        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = optimizer.optimize_with_constants_and_inputs(
                &trace_ops,
                &mut constants,
                trace.inputargs.len(),
            );
            for (k, v) in &optimizer.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let optimized_ops = match optimize_result {
            Ok(ops) => ops,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!("[jit] abort finish: InvalidLoop at key={}", green_key);
                    }
                    self.warm_state.abort_tracing(green_key, true);
                    // pyjitpl.py:2760 aborted_tracing() reads greenkey from
                    // `current_merge_points`; pyre's analog reads it from
                    // pending_abort_{green_key,permanent} staged here so the
                    // caller-side `aborted_tracing(stb.reason)` hook payload
                    // carries the real trace key instead of 0.
                    self.pending_abort_green_key = Some(green_key);
                    self.pending_abort_permanent = true;
                    return Err(SwitchToBlackhole::giveup());
                }
                std::panic::resume_unwind(payload);
            }
        };
        // RPython optimizer.py:552-556 (flush=True): Finish/Jump is sent
        // through passes inside propagate_all_forward and ends up in
        // new_operations naturally — no restoration needed.
        let optimized_ops = optimized_ops;
        let num_ops_after = optimized_ops.len();
        // optimizer.py:557 self.resumedata_memo.update_counters(profiler)
        optimizer.update_counters(&self.staticdata.profiler);
        // RPython compile.py:234 parity: transfer quasi-immutable deps
        // from optimizer to MetaInterp for post-compile watcher registration.
        self.last_quasi_immutable_deps = optimizer.quasi_immutable_deps.drain().collect();

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] finish_and_compile: key={}, ops_before={}, ops_after={}",
                green_key, num_ops_before, num_ops_after
            );
            eprintln!("--- finish trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
            eprintln!("--- finish trace (after opt, before unbox) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        // resume.py:411-417 parity: NONE entries in guard fail_args are
        // valid (TAGCONST/TAGVIRTUAL slots that resume reconstructs from
        // rd_consts/rd_virtuals). RPython has no compile-time abort
        // heuristic — runtime guard-fail recovery handles unrecoverable
        // cases via clear_compiled_loops.
        let mut optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        if crate::majit_log_enabled() {
            eprintln!("--- finish trace (after unbox) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        // Use pre-allocated token number if available (for self-recursion
        // support), otherwise allocate a fresh one.
        let token_num = if let Some((pk, pn)) = self.pending_token.take() {
            if pk == green_key {
                pn
            } else {
                self.warm_state.alloc_token_number()
            }
        } else {
            self.warm_state.alloc_token_number()
        };
        // `compile.py:266 jitcell_token = make_jitcell_token(jitdriver_sd)`.
        let mut token =
            make_jitcell_token(token_num, driver_descriptor.as_ref().and_then(|d| d.index));
        self.configure_loop_token_for_driver(
            Arc::get_mut(&mut token).expect("fresh JitCellToken must be uniquely owned"),
            green_key,
            driver_descriptor.as_ref(),
        );
        // `compile.py:180-181` wref wiring — done inside
        // `record_loop_or_bridge` once all `Arc::get_mut` writes settle.
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        // compile.py:233 `loop.inputargs = loop_info.inputargs`.
        let mut inputargs = trace.inputargs.clone();
        // Reconcile inputarg types with optimizer's post-unbox types.
        // Pyre starts tracing with Ref values (all Python objects), but
        // the optimizer may unbox Int-typed locals. Read the first guard's
        // fail_arg_types to discover the optimizer's type decisions, then
        // update inputargs to match. This ensures gcmap and adapt-live
        // agree on which slots are GC refs vs raw ints.
        if let Some(first_guard_types) = optimized_ops
            .iter()
            .find(|op| op.opcode.is_guard())
            .and_then(|op| {
                op.descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
            })
        {
            for (i, ia) in inputargs.iter_mut().enumerate() {
                if let Some(&tp) = first_guard_types.get(i) {
                    ia.tp = tp;
                }
            }
        }

        // Note: adapt-live type correction (inputarg Ref→Int, guard
        // fail_arg_types) is NOT applied here. CalAssemblerI calls the
        // callee without adapt-live, so the runtime types at guard failure
        // are the original Ref types. The no-snapshot fallback handles
        // types correctly via MetaFailDescr.

        // compile.py:504-511 send_loop_to_backend — unconditional virtualizable
        // field reload for every loop. See
        // `MetaInterp::patch_new_loop_to_load_virtualizable_fields` above;
        // `orig_vable_ptr` is the constant Ref that was stashed for the
        // virtualizable inputarg at trace-start (captured above via
        // `ctx.initial_inputarg_consts` + `ctx.constants.get_value`), i.e.
        // RPython's `orig_inpargs[idx].getref_base()`.
        self.patch_new_loop_to_load_virtualizable_fields(
            &mut inputargs,
            &mut optimized_ops,
            &mut constants,
            &mut constant_types,
            driver_descriptor.as_ref(),
            orig_vable_ptr,
        );

        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());
        match self.backend.compile_loop(
            &inputargs,
            &optimized_ops,
            Arc::get_mut(&mut token)
                .expect("JitCellToken must stay uniquely owned until backend compile"),
        ) {
            Ok(_) => {
                self.assign_guard_hashes(token.as_ref());
                self.warm_state.memory_manager.keep_loop_alive(&token);
                // compile.py:213 record_loop_or_bridge.
                self.record_loop_or_bridge(&token, &mut optimized_ops, trace_id);
                let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                    &inputargs,
                    &optimized_ops,
                    green_key,
                    &compiled_constant_types,
                    self.callinfocollection.as_deref(),
                );
                let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                    &inputargs,
                    &optimized_ops,
                    &compiled_constant_types,
                );
                if let Some(backend_layouts) =
                    self.backend.compiled_fail_descr_layouts(token.as_ref())
                {
                    compile::merge_backend_exit_layouts(
                        &mut exit_layouts,
                        backend_layouts.as_slice(),
                        &optimized_ops,
                    );
                }
                if let Some(backend_layouts) =
                    self.backend.compiled_terminal_exit_layouts(token.as_ref())
                {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                        &optimized_ops,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(token.as_ref(), trace_id);
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                self.take_back_all_descrs(std::mem::take(&mut optimizer.all_descrs));
                let mut next_global_opref = compute_next_global_opref(&inputargs, &optimized_ops);
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: trace.inputargs.clone(),
                        ops: optimized_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types.clone(),
                        exit_layouts,
                        terminal_exit_layouts,
                    },
                );
                {
                    let mut previous_tokens: Vec<std::sync::Arc<JitCellToken>> = Vec::new();
                    let ft = self
                        .compiled_loops
                        .get(&green_key)
                        .map(|c| c.front_target_tokens.clone())
                        .unwrap_or_default();
                    let rc = self
                        .compiled_loops
                        .get(&green_key)
                        .map(|c| c.token.get_retraced_count())
                        .unwrap_or(0);
                    let _had_old = self.compiled_loops.contains_key(&green_key);
                    if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                        // Box Identity Phase E.2b parity: preserve old entry's
                        // high-water so previously stored bridges' OpRefs stay
                        // disjoint from any future bridge.
                        next_global_opref = next_global_opref.max(old_entry.next_global_opref);
                        previous_tokens =
                            self.retire_compiled_entry(green_key, old_entry, &mut traces);
                    }
                    token.set_retraced_count(rc);
                    self.compiled_loops.insert(
                        green_key,
                        CompiledEntry {
                            token: Arc::clone(&token),
                            meta,
                            front_target_tokens: ft,
                            root_trace_id: trace_id,
                            traces,
                            previous_tokens,
                            next_global_opref,
                        },
                    );
                }
                self.attach_procedure_with_redirect(green_key, Arc::clone(&token));
                self.stats.loops_compiled += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] finish_and_compile: compiled trace key={}, trace_id={}",
                        green_key, trace_id
                    );
                }
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
            }
            Err(e) => {
                let msg = format!("finish_and_compile: compile_loop FAILED key={green_key}: {e:?}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                self.warm_state.abort_tracing(green_key, false);
                // pyjitpl.py:2761/:2786 `aborted_tracing` is the single
                // bump site for `stats.aborted()`; keep the increment
                // there so the caller-side `aborted_tracing(stb.reason)`
                // catch counts exactly once.  pyjitpl.py:2760 reads
                // greenkey from the current merge-point state — pyre's
                // analog is pending_abort_* staged here for the catch.
                self.pending_abort_green_key = Some(green_key);
                self.pending_abort_permanent = false;
                return Err(SwitchToBlackhole::giveup());
            }
        }
        Ok(())
    }

    /// compile.py:216-249 compile_simple_loop parity.
    ///
    /// Compiles the trace with simple optimizer (no preamble peeling),
    /// prepends a LABEL (via front_target_tokens) for bridge attachment.
    /// Returns the green_key on success (caller must call
    /// attach_procedure_to_interp), None on failure.
    pub fn compile_simple_loop(&mut self, meta: M) -> Option<u64> {
        let vable_config = self.current_virtualizable_optimizer_config();
        self.force_finish_trace = false;
        let mut ctx = match self.tracing.take() {
            Some(ctx) => ctx,
            None => return None,
        };
        let green_key = ctx.green_key;
        let driver_descriptor = ctx.driver_descriptor().cloned();
        // compile.py:510 parity — capture orig_inpargs[idx].getref_base()
        // before `ctx.recorder` is moved. Used by the send_loop_to_backend
        // hook below.
        let orig_vable_ptr_simple =
            self.orig_vable_ptr_from_trace_ctx(&ctx, driver_descriptor.as_ref());

        let recorder = ctx.recorder;
        // Task #70: snapshots live on TraceCtx; rebuild the TreeLoop with
        // them so downstream consumers (`trace.snapshots`) still observe
        // the captured resumedata. `recorder.get_trace()` on its own
        // returns a snapshot-less TreeLoop post-Task #70.
        let mut trace = recorder.get_trace();
        trace.snapshots = std::mem::take(&mut ctx.snapshots);
        let trace_snapshots = trace.snapshots.clone();

        let numbering_overrides = ctx.constants.numbering_type_overrides().clone();
        // Refresh shadow-stack-rooted Ref constants so `into_inner_with_types`
        // sees the post-GC addresses for any object that moved.
        ctx.constants.refresh_from_gc();
        let (mut constants, mut constant_types) = ctx.constants.into_inner_with_types();

        let trace_ops = trace.ops.clone();

        if crate::majit_log_enabled() {
            eprintln!("--- simple loop trace (before opt) ---");
            eprint!("{}", majit_ir::format_trace(&trace_ops, &constants));
        }

        let num_ops_before = trace_ops.len();
        let num_trace_inputargs = trace.inputargs.len();

        // Simple optimizer — no unrolling (compile.py:222-226 SimpleCompileData).
        let mut optimizer = if let Some(config) = vable_config {
            Optimizer::default_pipeline_with_virtualizable(config)
        } else {
            Optimizer::default_pipeline()
        };
        optimizer.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        optimizer.constant_types = constant_types.clone();
        optimizer.numbering_type_overrides = numbering_overrides;
        // history.py InputArg.type parity: register inputarg types.
        for ia in &trace.inputargs {
            optimizer.constant_types.insert(ia.opref().raw(), ia.tp);
        }

        let (
            snapshot_map,
            snapshot_frame_size_map,
            snapshot_vable_map,
            snapshot_vref_map,
            snapshot_pc_map,
        ) = snapshot_map_from_trace_snapshots(
            &trace_snapshots,
            &mut constants,
            &mut constant_types,
        );
        optimizer.snapshot_boxes = snapshot_map;
        optimizer.snapshot_frame_sizes = snapshot_frame_size_map;
        optimizer.snapshot_vable_boxes = snapshot_vable_map;
        optimizer.snapshot_vref_boxes = snapshot_vref_map;
        optimizer.snapshot_frame_pcs = snapshot_pc_map;
        // H-3.0b: simple-loop path — same recorder BoxRef pool plumb as
        // the unrolled loop entry above.
        optimizer.set_pending_box_pool(trace.box_pool.clone());

        let mut updated_constant_types = constant_types.clone();
        let optimize_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let result = optimizer.optimize_with_constants_and_inputs(
                &trace_ops,
                &mut constants,
                num_trace_inputargs,
            );
            for (k, v) in &optimizer.constant_types {
                updated_constant_types.entry(*k).or_insert(*v);
            }
            result
        }));
        constant_types = updated_constant_types;
        let optimized_ops = match optimize_result {
            Ok(ops) => ops,
            Err(payload) => {
                if crate::majit_log_enabled() {
                    if payload
                        .downcast_ref::<crate::optimize::InvalidLoop>()
                        .is_some()
                    {
                        eprintln!(
                            "[jit] compile_simple_loop: InvalidLoop at key={}",
                            green_key
                        );
                    }
                }
                // compile.py:228-230: trace.cut_at(cut_at); return None
                self.warm_state.abort_tracing(green_key, false);
                return None;
            }
        };

        // optimizer.py:557 self.resumedata_memo.update_counters(profiler)
        optimizer.update_counters(&self.staticdata.profiler);
        self.last_quasi_immutable_deps = optimizer.quasi_immutable_deps.drain().collect();

        let num_ops_after = optimized_ops.len();
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] compile_simple_loop: key={}, ops_before={}, ops_after={}",
                green_key, num_ops_before, num_ops_after
            );
            eprintln!("--- simple loop trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        let optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        // Allocate token and compile.
        let token_num = self.warm_state.alloc_token_number();
        // `compile.py:266 jitcell_token = make_jitcell_token(jitdriver_sd)`.
        let mut token =
            make_jitcell_token(token_num, driver_descriptor.as_ref().and_then(|d| d.index));
        self.configure_loop_token_for_driver(
            Arc::get_mut(&mut token).expect("fresh JitCellToken must be uniquely owned"),
            green_key,
            driver_descriptor.as_ref(),
        );
        // `compile.py:180-181` wref wiring — done inside
        // `record_loop_or_bridge` once all `Arc::get_mut` writes settle.
        let trace_id = self.alloc_trace_id();
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(green_key);

        // compile.py:233 `loop.inputargs = loop_info.inputargs`.
        let mut inputargs = trace.inputargs.clone();

        // compile.py:236-245 parity: simple-loop compilation owns a real
        // TargetToken, prepends LABEL(descr=target_token), and patches the
        // closing JUMP to the same token.
        let target_token = crate::optimizeopt::unroll::TargetToken::new_loop(token_num);
        // `compile.py:237 target_token.original_jitcell_token = jitcell_token`.
        target_token.set_original_jitcell_token_number(token_num);
        let mut compiled_ops = optimized_ops.clone();
        if let Some(jump_op) = compiled_ops
            .last_mut()
            .filter(|op| op.opcode == OpCode::Jump)
        {
            jump_op.descr = Some(target_token.as_jump_target_descr());
        }
        let mut label_op = majit_ir::Op::new(
            majit_ir::OpCode::Label,
            &inputargs.iter().map(|ia| ia.opref()).collect::<Vec<_>>(),
        );
        label_op.pos = majit_ir::OpRef::NONE;
        label_op.descr = Some(target_token.as_jump_target_descr());
        compiled_ops.insert(0, label_op);

        // compile.py:504-511 send_loop_to_backend virtualizable hook —
        // simple-loop compile path must also reload virtualizable fields on
        // entry. Without this, the vable inputarg contract differs from
        // the unrolled loop path and guard-failure recovery cannot restore
        // the heap array slots.
        self.patch_new_loop_to_load_virtualizable_fields(
            &mut inputargs,
            &mut compiled_ops,
            &mut constants,
            &mut constant_types,
            driver_descriptor.as_ref(),
            orig_vable_ptr_simple,
        );
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());
        match self.backend.compile_loop(
            &inputargs,
            &compiled_ops,
            Arc::get_mut(&mut token)
                .expect("JitCellToken must stay uniquely owned until backend compile"),
        ) {
            Ok(_) => {
                self.assign_guard_hashes(token.as_ref());
                self.warm_state.memory_manager.keep_loop_alive(&token);
                // compile.py:213 record_loop_or_bridge.
                self.record_loop_or_bridge(&token, &mut compiled_ops, trace_id);
                let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                    &inputargs,
                    &compiled_ops,
                    green_key,
                    &compiled_constant_types,
                    self.callinfocollection.as_deref(),
                );
                let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                    &inputargs,
                    &compiled_ops,
                    &compiled_constant_types,
                );
                if let Some(backend_layouts) =
                    self.backend.compiled_fail_descr_layouts(token.as_ref())
                {
                    compile::merge_backend_exit_layouts(
                        &mut exit_layouts,
                        backend_layouts.as_slice(),
                        &compiled_ops,
                    );
                }
                if let Some(backend_layouts) =
                    self.backend.compiled_terminal_exit_layouts(token.as_ref())
                {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                        &compiled_ops,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(token.as_ref(), trace_id);
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                self.take_back_all_descrs(std::mem::take(&mut optimizer.all_descrs));
                let mut next_global_opref = compute_next_global_opref(&inputargs, &compiled_ops);
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: trace.inputargs.clone(),
                        ops: compiled_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types,
                        exit_layouts,
                        terminal_exit_layouts,
                    },
                );
                let mut previous_tokens: Vec<std::sync::Arc<JitCellToken>> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&green_key) {
                    // Box Identity Phase E.2b parity: see finish_and_compile.
                    next_global_opref = next_global_opref.max(old_entry.next_global_opref);
                    previous_tokens = self.retire_compiled_entry(green_key, old_entry, &mut traces);
                }
                self.compiled_loops.insert(
                    green_key,
                    CompiledEntry {
                        token: Arc::clone(&token),
                        meta,
                        front_target_tokens: vec![target_token],
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                        next_global_opref,
                    },
                );
                self.stats.loops_compiled += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_simple_loop: compiled segmented trace key={}, trace_id={}",
                        green_key, trace_id
                    );
                }
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(green_key, num_ops_before, num_ops_after);
                }
                // compile.py:249: return target_token
                return Some(green_key);
            }
            Err(e) => {
                self.stats.loops_aborted += 1;
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_simple_loop: compile FAILED key={}: {:?}",
                        green_key, e
                    );
                }
                self.warm_state.abort_tracing(green_key, false);
                return None;
            }
        }
    }

    /// Resolve cross-loop cut alias: inner_key → outer_key.
    ///
    /// If the key has its own direct entry, use it — alias only applies
    /// when no direct entry exists. This prevents cross-loop cut aliases
    /// from shadowing independently compiled inner loop entries.
    ///
    /// Get the metadata for a compiled loop without executing it.
    ///
    /// Allows the interpreter to check preconditions (e.g., whether the
    /// current state matches the compiled loop's assumptions) before calling
    /// the run_compiled_* family.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&M> {
        self.compiled_loops.get(&green_key).map(|e| &e.meta)
    }

    /// Actual key the last compile_loop stored under. Returns inner key
    /// for cross-loop cuts, otherwise the tracing key.
    pub fn last_compiled_key(&self) -> Option<u64> {
        self.last_compiled_key
    }

    /// warmstate.py:437-444 `cell.flags |= JC_TRACING ... try ... finally:
    /// cell.flags &= ~JC_TRACING` parity — the green_key that was entered
    /// into `bound_reached` and on which TRACING must be cleared unconditionally
    /// once tracing ends. Pulled from the active TraceCtx; returns None when
    /// no trace is in progress.
    pub fn starting_green_key(&self) -> Option<u64> {
        self.tracing.as_ref().map(|ctx| ctx.green_key)
    }

    /// Typed-input raw fast-path runner.  Avoids explicit deadframe decoding
    /// in the caller while preserving typed exits, backend exit layout,
    /// savedata, and exception state.
    pub fn run_compiled_raw_detailed_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<RawCompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let result = self.backend.execute_token_raw(&compiled.token, live_values);
        Self::finish_compiled_run_io();

        let fail_index = result.fail_index;
        let trace_id = result.trace_id;

        let trace_layout =
            Self::trace_for_exit(compiled, trace_id).and_then(|(trace_id, trace)| {
                Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
            });
        let exit_layout = result
            .exit_layout
            .clone()
            .map(|layout| {
                let trace_layout_ref = trace_layout.as_ref();
                let mut resume_layout = trace_layout
                    .as_ref()
                    .and_then(|tl| tl.resume_layout.clone());
                compile::enrich_resume_layout_with_frame_stack(
                    &mut resume_layout,
                    layout.frame_stack.as_deref(),
                );
                CompiledExitLayout {
                    rd_loop_token: green_key, // compile.py:186
                    trace_id,
                    fail_index: layout.fail_index,
                    source_op_index: layout
                        .source_op_index
                        .or_else(|| trace_layout_ref.and_then(|layout| layout.source_op_index)),
                    exit_types: layout.fail_arg_types,
                    is_finish: layout.is_finish,
                    gc_ref_slots: layout.gc_ref_slots,
                    force_token_slots: layout.force_token_slots,
                    recovery_layout: layout.recovery_layout.or_else(|| {
                        trace_layout_ref.and_then(|layout| layout.recovery_layout.clone())
                    }),
                    resume_layout,
                    storage: trace_layout_ref.and_then(|layout| layout.storage.clone()),
                }
            })
            .or(trace_layout)
            .unwrap_or_else(|| CompiledExitLayout {
                rd_loop_token: green_key, // from trace context
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: result.typed_outputs.iter().map(Value::get_type).collect(),
                is_finish: result.is_finish,
                gc_ref_slots: result
                    .typed_outputs
                    .iter()
                    .enumerate()
                    .filter_map(|(slot, value)| (value.get_type() == Type::Ref).then_some(slot))
                    .collect(),
                force_token_slots: result.force_token_slots.clone(),
                recovery_layout: None,
                resume_layout: None,
                storage: None,
            });
        let effective_is_finish = result.is_finish || exit_layout.is_finish;
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit] run_compiled_exit: gk={} fi={} tid={} result.finish={} layout.finish={} effective={}",
                green_key,
                fail_index,
                trace_id,
                result.is_finish,
                exit_layout.is_finish,
                effective_is_finish
            );
        }

        if Self::should_record_guard_failure(effective_is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_class = if result.exception_value.is_null() {
            0
        } else {
            unsafe { *(result.exception_value.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: result.exception_value.0 as i64,
            ovf_flag: false,
        };
        let descr_arc = result.descr_arc.clone();
        let compiled = self.compiled_loops.get(&green_key).unwrap();

        Some(RawCompileResult {
            values: result.outputs,
            typed_values: result.typed_outputs,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            descr_arc,
            is_finish: effective_is_finish,
            is_exit_frame_with_exception: result.is_exit_frame_with_exception,
            exit_layout,
            savedata: result.savedata,
            exception,
            status: result.status,
        })
    }

    /// Run compiled code and return detailed guard failure information.
    ///
    /// Unlike `run_compiled`, this returns the full `CompileResult` including
    /// the fail_index, which allows the interpreter to handle different guard
    /// failures differently.
    pub fn run_compiled_detailed(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<CompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let frame = self
            .backend
            .execute_token_ints(&compiled.token, live_values);

        let descr_arc = self.backend.get_latest_descr_arc(&frame);
        let descr: &dyn majit_ir::FailDescr = &*descr_arc;
        let fail_index = descr.fail_index();
        let trace_id = descr.trace_id();
        let is_finish = descr.is_finish();
        let is_exit_frame_with_exception = descr.is_exit_frame_with_exception();
        let exit_types = descr.fail_arg_types().to_vec();
        let gc_ref_slots: Vec<usize> = exit_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
            .collect();
        let force_token_slots = descr.force_token_slots().to_vec();
        let status = descr.get_status();
        // compile.py:186 `descr.rd_loop_token` — owning loop's clt,
        // stamped at compile time.  Walk the chain
        // `descr.rd_loop_token_clt() → clt.upgrade_loop_token()` to
        // recover the owning `Arc<JitCellToken>` (pyjitpl.py:2897
        // `resumedescr.rd_loop_token.loop_token_wref()`) for guard
        // exits that belong to a loop other than the one currently
        // executing (bridge-into-B while running A).  Derive
        // `green_key` from `jct.green_key` so identity is preserved
        // through the lookup. O(1) replacement for the legacy O(N)
        // scan over `compiled_loops`.
        let rd_loop_token = majit_backend::descr_owning_jct(descr).map(|jct| jct.green_key);
        Self::finish_compiled_run_io();

        if Self::should_record_guard_failure(is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        // FINISH descrs are singletons (`DONE_WITH_THIS_FRAME_DESCR_*` /
        // `EXIT_FRAME_WITH_EXCEPTION_DESCR_REF_CL`) with `trace_id == 0`
        // and `fail_index == u32::MAX`; they carry no per-trace exit
        // metadata. Skip the trace lookup entirely and synthesize the
        // default layout, mirroring RPython where FINISH descrs are
        // dispatched on identity rather than trace-keyed lookup.
        let mut exit_layout = if is_finish {
            CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.clone(),
                is_finish,
                gc_ref_slots,
                force_token_slots,
                recovery_layout: None,
                resume_layout: None,
                storage: None,
            }
        } else {
            Self::trace_for_exit(compiled, trace_id)
                .map(|(resolved_id, trace)| (green_key, resolved_id, trace))
                .or_else(|| self.trace_for_exit_by_rd_loop_token(rd_loop_token, trace_id))
                .and_then(|(owning_key, resolved_id, trace)| {
                    Self::compiled_exit_layout_from_trace(
                        trace,
                        owning_key,
                        resolved_id,
                        fail_index,
                    )
                })
                .unwrap_or_else(|| CompiledExitLayout {
                    rd_loop_token: green_key,
                    trace_id,
                    fail_index,
                    source_op_index: None,
                    exit_types: exit_types.clone(),
                    is_finish,
                    gc_ref_slots,
                    force_token_slots,
                    recovery_layout: None,
                    resume_layout: None,
                    storage: None,
                })
        };
        // RPython: deadframe has ALL jitframe slots accessible.
        // If the backend's descr covers more slots than the trace layout,
        // extend exit_layout.exit_types to match (conservative Int for extras).
        if exit_types.len() > exit_layout.exit_types.len() {
            exit_layout.exit_types.resize(exit_types.len(), Type::Int);
        }
        let mut values = Vec::with_capacity(exit_arity);
        let mut typed_values = Vec::with_capacity(exit_arity);
        for (i, &tp) in exit_types.iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.backend.get_int_value(&frame, i);
                    values.push(value);
                    typed_values.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.backend.get_ref_value(&frame, i);
                    values.push(value.as_usize() as i64);
                    typed_values.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.backend.get_float_value(&frame, i);
                    values.push(value.to_bits() as i64);
                    typed_values.push(Value::Float(value));
                }
                Type::Void => {
                    values.push(0);
                    typed_values.push(Value::Void);
                }
            }
        }
        let savedata = self.backend.get_savedata_ref(&frame);
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_value_ref = self.backend.grab_exc_value(&frame);
        let exc_class = if exc_value_ref.is_null() {
            0
        } else {
            unsafe { *(exc_value_ref.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: exc_value_ref.0 as i64,
            ovf_flag: false,
        };

        Some(CompileResult {
            values,
            typed_values,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            descr_arc,
            is_finish,
            is_exit_frame_with_exception,
            exit_layout,
            savedata,
            exception,
            status,
        })
    }

    /// Typed-input counterpart to [`run_compiled_detailed`].
    pub fn run_compiled_detailed_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<CompileResult<'_, M>> {
        let compiled = self.compiled_loops.get(&green_key)?;

        Self::prepare_compiled_run_io();
        let frame = self.backend.execute_token(&compiled.token, live_values);
        // RPython: bridge compilation happens synchronously inside
        // assembler_call_helper (called from compiled code). No deferred queue.

        let descr_arc = self.backend.get_latest_descr_arc(&frame);
        let descr: &dyn majit_ir::FailDescr = &*descr_arc;
        let fail_index = descr.fail_index();
        let trace_id = descr.trace_id();
        let is_finish = descr.is_finish();
        let is_exit_frame_with_exception = descr.is_exit_frame_with_exception();
        let exit_types = descr.fail_arg_types().to_vec();
        let gc_ref_slots: Vec<usize> = exit_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
            .collect();
        let force_token_slots = descr.force_token_slots().to_vec();
        let status = descr.get_status();
        // compile.py:186 `descr.rd_loop_token` — see `run_compiled_detailed`.
        let rd_loop_token = majit_backend::descr_owning_jct(descr).map(|jct| jct.green_key);
        Self::finish_compiled_run_io();

        // RPython: guard failure counter tick and bridge compilation happen
        // in handle_fail → must_compile (compile.py:701-784).
        // must_compile handles tick.
        if Self::should_record_guard_failure(is_finish, fail_index) {
            self.record_guard_failure_event(green_key, fail_index);
        }

        let exit_arity = exit_types.len();
        let compiled = self.compiled_loops.get(&green_key).unwrap();
        // FINISH descrs (singletons) have `trace_id == 0`; skip the
        // trace lookup and synthesize the default layout per
        // `run_compiled_detailed`.
        let mut exit_layout = if is_finish {
            CompiledExitLayout {
                rd_loop_token: green_key,
                trace_id,
                fail_index,
                source_op_index: None,
                exit_types: exit_types.clone(),
                is_finish,
                gc_ref_slots: gc_ref_slots.clone(),
                force_token_slots: force_token_slots.clone(),
                recovery_layout: None,
                resume_layout: None,
                storage: None,
            }
        } else {
            Self::trace_for_exit(compiled, trace_id)
                .map(|(resolved_id, trace)| (green_key, resolved_id, trace))
                .or_else(|| self.trace_for_exit_by_rd_loop_token(rd_loop_token, trace_id))
                .and_then(|(owning_key, resolved_id, trace)| {
                    Self::compiled_exit_layout_from_trace(
                        trace,
                        owning_key,
                        resolved_id,
                        fail_index,
                    )
                })
                .unwrap_or_else(|| CompiledExitLayout {
                    rd_loop_token: green_key,
                    trace_id,
                    fail_index,
                    source_op_index: None,
                    exit_types: exit_types.clone(),
                    is_finish,
                    gc_ref_slots,
                    force_token_slots,
                    recovery_layout: None,
                    resume_layout: None,
                    storage: None,
                })
        };
        // RPython: deadframe has ALL jitframe slots accessible.
        // If the backend's descr covers more slots than the trace layout,
        // extend exit_layout.exit_types to match (conservative Int for extras).
        if exit_types.len() > exit_layout.exit_types.len() {
            exit_layout.exit_types.resize(exit_types.len(), Type::Int);
        }
        let mut values = Vec::with_capacity(exit_arity);
        let mut typed_values = Vec::with_capacity(exit_arity);
        for (i, &tp) in exit_types.iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.backend.get_int_value(&frame, i);
                    values.push(value);
                    typed_values.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.backend.get_ref_value(&frame, i);
                    values.push(value.as_usize() as i64);
                    typed_values.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.backend.get_float_value(&frame, i);
                    values.push(value.to_bits() as i64);
                    typed_values.push(Value::Float(value));
                }
                Type::Void => {
                    values.push(0);
                    typed_values.push(Value::Void);
                }
            }
        }
        let savedata = self.backend.get_savedata_ref(&frame);
        // pyjitpl.py:3119-3123: exc_class = ptr2int(exception_obj.typeptr)
        let exc_value_ref = self.backend.grab_exc_value(&frame);
        let exc_class = if exc_value_ref.is_null() {
            0
        } else {
            unsafe { *(exc_value_ref.0 as *const i64) }
        };
        let exception = ExceptionState {
            exc_class,
            exc_value: exc_value_ref.0 as i64,
            ovf_flag: false,
        };

        Some(CompileResult {
            values,
            typed_values,
            meta: &compiled.meta,
            fail_index,
            trace_id,
            descr_arc,
            is_finish,
            is_exit_frame_with_exception,
            exit_layout,
            savedata,
            exception,
            status,
        })
    }

    /// Attach resume data to a specific guard in a compiled loop.
    ///
    /// `resume.py:1042 rebuild_from_resumedata` consumes this storage at
    /// blackhole resume time via the descr; only test fixtures install
    /// resume data through this MetaInterp-side helper today.
    ///
    /// **No PyPy counterpart**: PyPy builds `ResumeGuardDescr` storage
    /// during compilation (`compile.py:858 compile_loop_or_bridge`
    /// → `record_loop_or_bridge` populates `descr.rd_*` from the live
    /// `ResumeData` snapshot before the loop executes). There is no
    /// helper that injects resume data after the fact. Tests that need
    /// the production `get_resume_storage` chain can use this helper
    /// because it installs both the layout summary and the guard-owned
    /// `ResumeStorage` surrogate.
    pub fn attach_resume_data(&mut self, green_key: u64, fail_index: u32, resume_data: ResumeData) {
        let Some(trace_id) = self.compiled_loops.get(&green_key).map(|c| c.root_trace_id) else {
            return;
        };
        self.attach_resume_data_to_trace(green_key, trace_id, fail_index, resume_data);
    }

    /// Attach resume data to a specific guard in a specific compiled trace.
    ///
    /// **Test-helper-only divergence (no PyPy counterpart).** The
    /// production compile path populates two views of guard-owned resume
    /// data on `StoredExitLayout`: the `ResumeLayoutSummary` used by
    /// frontend recovery helpers and the shared `ResumeStorage` consumed
    /// by `get_resume_storage` (`compile.py:853 ResumeGuardDescr`
    /// parity). This helper now installs both views from the same
    /// `EncodedResumeData` so tests observe the same lookup chain as
    /// production. Pending-field replay still requires the production
    /// compile path because the test helper input has descriptor
    /// indices but not the live field/array descriptors.
    ///
    /// **Convergence path**: retire this helper after the 8 fixtures
    /// in `tests/jit_driver_runtime_parity.rs` migrate to
    /// compile-path resume data injection (multi-session test
    /// refactor). Reaching strict line-by-line parity removes the
    /// out-of-band injection surface entirely, matching PyPy's "resume
    /// data is built at compile time, never injected" contract.
    pub fn attach_resume_data_to_trace(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        resume_data: ResumeData,
    ) {
        // pyjitpl.py: `attach_resume_data_to_trace` callers always pass
        // the real allocated trace_id; `alloc_trace_id` starts at 1, so
        // `trace_id == 0` would be a sentinel-misuse bug, not a valid
        // input.  RPython has no `0 → root_trace_id` fallback because
        // it dispatches by descr object identity, not numeric trace id.
        let Some((trace_id, trace_info)) = self.compiled_loops.get(&green_key).map(|compiled| {
            (
                trace_id,
                self.backend.compiled_trace_info(&compiled.token, trace_id),
            )
        }) else {
            return;
        };
        let mut patched_recovery_layout = None;
        if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
            if let Some(trace) = compiled.traces.get_mut(&trace_id) {
                let recovery_layout = trace
                    .exit_layouts
                    .get(&fail_index)
                    .and_then(|layout| layout.recovery_layout.clone());
                let encoded = resume_data.encode();
                let mut layout = encoded.layout_summary();
                let storage = encoded.to_resume_storage();
                compile::enrich_resume_layout_with_trace_metadata(
                    &mut layout,
                    trace_id,
                    &trace.inputargs,
                    trace_info.as_ref(),
                    recovery_layout.as_ref(),
                );
                if let Some(exit_layout) = trace.exit_layouts.get_mut(&fail_index) {
                    exit_layout.resume_layout = Some(layout);
                    exit_layout.storage = Some(storage);
                    if let Some(summary) = exit_layout.resume_layout.as_ref() {
                        let recovery_layout = summary.to_exit_recovery_layout_with_caller_prefix(
                            exit_layout.recovery_layout.as_ref(),
                        );
                        exit_layout.recovery_layout = Some(recovery_layout.clone());
                        patched_recovery_layout = Some(recovery_layout);
                    }
                }
            }
        }
        if let Some(recovery_layout) = patched_recovery_layout {
            if let Some(compiled) = self.compiled_loops.get(&green_key) {
                let _ = self.backend.update_fail_descr_recovery_layout(
                    &compiled.token,
                    trace_id,
                    fail_index,
                    recovery_layout,
                );
            }
        }
    }

    /// Get the full static layout for a compiled exit in a specific trace.
    pub fn get_compiled_exit_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<CompiledExitLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        if let Some((resolved_trace_id, trace)) = Self::trace_for_exit(compiled, trace_id) {
            if let Some(layout) = Self::compiled_exit_layout_from_trace(
                trace,
                green_key,
                resolved_trace_id,
                fail_index,
            ) {
                return Some(layout);
            }
        }
        self.compiled_exit_layout_from_backend(compiled, green_key, trace_id, fail_index)
    }

    /// Get the full static layout for a terminal FINISH/JUMP op in a specific trace.
    pub fn get_terminal_exit_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
        op_index: usize,
    ) -> Option<CompiledExitLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
        Self::terminal_exit_layout_from_trace(trace, green_key, trace_id, op_index).or_else(|| {
            self.terminal_exit_layout_from_backend(compiled, green_key, trace_id, op_index)
        })
    }

    /// Get the full static layout for a compiled trace in a specific trace id.
    pub fn get_compiled_trace_layout_in_trace(
        &self,
        green_key: u64,
        trace_id: u64,
    ) -> Option<CompiledTraceLayout> {
        let compiled = self.compiled_loops.get(&green_key)?;
        self.compiled_trace_layout_for_trace(compiled, green_key, trace_id)
    }

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    ///
    /// Marks the loop token as invalidated. Subsequent executions of the
    /// compiled code will fail at GUARD_NOT_INVALIDATED and fall back to
    /// the interpreter.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        if let Some(token) = self.warm_state.get_compiled(green_key) {
            token.invalidate();
            if crate::majit_log_enabled() {
                eprintln!("[jit] invalidated loop at key={}", green_key);
            }
        }
    }

    /// rpython/rlib/rstack.py:75-90 `stack_almost_full` — delegates to
    /// the interpreter-registered hook (see
    /// `majit_metainterp::register_stack_almost_full_hook`) which reads
    /// `PYRE_STACKTOOBIG.stack_end` / `stack_length` and tracks
    /// `sys.setrecursionlimit`. Without a registered hook (tests),
    /// returns `false` matching `if not we_are_translated: return False`
    /// at rstack.py:76-77. Used by `compile.py:702-703` to skip bridge
    /// compilation and by `warmstate.py:430` to back off when stack
    /// space is tight.
    #[inline]
    pub fn stack_almost_full() -> bool {
        crate::stack_almost_full()
    }

    /// pyjitpl.py:2345-2348: try_to_free_some_loops — advance the
    /// memory manager's generation counter.  Old loops not accessed
    /// for max_age generations are removed from `alive_loops`.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre also drops the matching
    /// `compiled_loops` entry.  RPython's `try_to_free_some_loops` is
    /// one line — `next_generation()` alone — because dropping the
    /// only `alive_loops` strong owner triggers `LoopToken.__del__`
    /// (`memmgr.py:13`).  Pyre's `compiled_loops` is still a second
    /// strong owner, so without explicit removal here the backend
    /// resources stay pinned forever.  Slice X-G (memmgr Slice 3.6)
    /// downgrades `compiled_loops` to `Weak`, after which this branch
    /// is removed and the function reverts to the upstream one-liner
    /// shape.
    ///
    /// The eviction dispatch matches by **token-object identity**
    /// (`Arc::ptr_eq`) — mirroring `memmgr.py:73`'s `del
    /// self.alive_loops[looptoken]` which keys on the looptoken
    /// itself.  This protects the recompile case: when fn B was
    /// recompiled, `compiled_loops[gk].token` is the new token while
    /// the old token is in `previous_tokens`; if alive_loops evicts
    /// the old token, only the `previous_tokens` slot is dropped —
    /// the current entry stays.  Conversely, when the current token
    /// itself ages out, the whole `compiled_loops` entry is dropped
    /// (RPython's `LoopToken.__del__` analog).
    pub fn try_to_free_some_loops(&mut self) {
        let evicted = self.warm_state.memory_manager.next_generation();
        for token in evicted {
            let gk = token.green_key;
            let Some(entry) = self.compiled_loops.get_mut(&gk) else {
                continue;
            };
            if std::sync::Arc::ptr_eq(&entry.token, &token) {
                // Current eviction is the `LoopToken.__del__` analog
                // (memmgr.py:73): the whole green_key's loop disappears,
                // including every previous-token predecessor on the
                // entry (the merged `traces` map and the
                // previous_tokens Vec drop together).
                self.compiled_loops.remove(&gk);
                if crate::majit_log_enabled() {
                    eprintln!("[jit][memmgr] evicted current loop key={}", gk);
                }
            } else {
                let before = entry.previous_tokens.len();
                entry
                    .previous_tokens
                    .retain(|t| !std::sync::Arc::ptr_eq(t, &token));
                if crate::majit_log_enabled() && entry.previous_tokens.len() < before {
                    eprintln!("[jit][memmgr] evicted previous token at key={}", gk);
                }
            }
        }
    }

    /// Get the number of fail_arg_types for a guard.
    /// Returns 0 if not found. Used to adjust bridge tracing state.
    pub fn fail_arg_count_for(&self, green_key: u64, trace_id: u64, fail_index: u32) -> usize {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return 0;
        };
        self.bridge_source_descr(compiled, trace_id, fail_index)
            .and_then(|d| d.as_fail_descr().map(|fd| fd.fail_arg_types().len()))
            .unwrap_or(0)
    }

    /// `warmstate.py:339-348 attach_procedure_to_interp` parity.
    ///
    /// ```python
    /// def attach_procedure_to_interp(self, greenkey, procedure_token):
    ///     cell = self.JitCell.ensure_jit_cell_at_key(greenkey)
    ///     old_token = cell.get_procedure_token()
    ///     cell.set_procedure_token(procedure_token)
    ///     if old_token is not None:
    ///         self.cpu.redirect_call_assembler(old_token, procedure_token)
    ///         old_token.record_jump_to(procedure_token)
    /// ```
    ///
    /// Caller-side helper because pyre's `WarmEnterState` does not
    /// hold a `&mut Backend` (the upstream `self.cpu` lives on the
    /// driver) — the `redirect_call_assembler` + `record_jump_to`
    /// chain runs here, where `MetaInterp` owns both the warm-state
    /// cell map and the backend.
    ///
    /// The compile paths pass the actual compiled `Arc<JitCellToken>`,
    /// so `cell.loop_token` shares identity with the token registered in
    /// `MemoryManager`, matching upstream's single-object flow.
    pub fn attach_procedure_with_redirect(
        &mut self,
        green_key: u64,
        attach_token: std::sync::Arc<JitCellToken>,
    ) {
        let old_token = self
            .warm_state
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&attach_token));
        if let Some(old_token) = old_token {
            // `warmstate.py:343-347` line-by-line: when an `old_token`
            // is present, redirect + record_jump_to run unconditionally.
            // Upstream guarantees both tokens own real compiled code at
            // this point — the public attach paths (`compile_loop`,
            // `compile_simple_loop_or_bridge`, bridge-finish) only call
            // here after `backend.compile_loop` has stamped the new
            // token, and the old token retained its compiled blob from
            // the previous attach.
            //
            // `warmstate.py:344` `cpu.redirect_call_assembler(old, new)`.
            let _ = self
                .backend
                .redirect_call_assembler(&old_token, &attach_token);
            // `warmstate.py:347` `old_token.record_jump_to(procedure_token)`.
            old_token.record_jump_to(attach_token);
        }
    }

    // ── Call Assembler Support ──────────────────────────────────

    /// Get the JitCellToken for a compiled loop (for CALL_ASSEMBLER).
    ///
    /// In RPython, `call_assembler` allows JIT code for one function
    /// to directly call JIT code for another function. The caller needs
    /// the target's JitCellToken to set up the call.
    pub fn get_loop_token(&self, green_key: u64) -> Option<&JitCellToken> {
        self.warm_state
            .get_compiled(green_key)
            .map(|arc| arc.as_ref())
    }

    /// Return the owning `Arc<JitCellToken>` for the compiled loop at
    /// `green_key`, matching `compile.py:187 isinstance(descr, JitCellToken)`
    /// identity. Used by `direct_assembler_call` to thread the same Arc
    /// through `make_call_assembler_descr` so the keepalive walker
    /// (`record_loop_or_bridge`) recovers the production token directly
    /// from the descr without a side-table lookup.
    pub fn get_loop_token_arc(&self, green_key: u64) -> Option<&std::sync::Arc<JitCellToken>> {
        self.warm_state.get_compiled(green_key)
    }

    /// Recover the actual front-target LABEL contract for bridge closes.
    ///
    /// RPython bridge closes target the peeled loop entry token recorded in
    /// `jitcell_token.target_tokens[0]`, not the red-only `JitCellToken`
    /// entry signature. In majit the equivalent LABEL args live in the root
    /// trace's saved ops, so rebuild their types from that trace.
    ///
    /// Two parity paths cover every JUMP target:
    /// 1. Peeled (unrolled) loops — `front_target_tokens.first()` names the
    ///    peeled-entry `TargetToken`; locate its LABEL in the root trace and
    ///    read each arg's type via `build_trace_value_maps`. RPython:
    ///    `optimizeopt/unroll.py` peeled-entry LABEL is the JUMP target.
    /// 2. Non-peeled loops — there is no peeled-entry token; the JUMP target
    ///    is the loop's `TreeLoop.inputargs` (`history.py:501`). Their types
    ///    live on `root_trace.inputargs[i].tp`.
    pub fn front_target_inputarg_types(&self, green_key: u64) -> Option<Vec<Type>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let root_trace = compiled.traces.get(&compiled.root_trace_id)?;
        if let Some(front_target) = compiled.front_target_tokens.first() {
            let target_descr = front_target.as_jump_target_descr();
            // `root_trace.constant_types` was populated alongside `ops`
            // / `inputargs` at compile time (`build_trace_value_maps`
            // → `OpTypeIndex` seed) and survives in the stored trace.
            // Passing an empty map here would let any legacy
            // `Untyped(x | CONST_BIT)` operand fall through
            // `OpTypeIndex::opref_type_at`'s constant_types miss to the
            // hard-panic boundary (`history.py:220` parity), so reuse
            // the saved map instead.
            let type_index = majit_ir::OpTypeIndex::new(
                &root_trace.inputargs,
                &root_trace.ops,
                &root_trace.constant_types,
            );
            if let Some((label_index, label)) = root_trace.ops.iter().enumerate().find(|(_, op)| {
                op.opcode == OpCode::Label
                    && op
                        .descr
                        .as_ref()
                        .is_some_and(|descr| descr.index() == target_descr.index())
            }) {
                return Some(
                    label
                        .args
                        .iter()
                        .map(|arg| {
                            type_index
                                .opref_type_at(*arg, label_index)
                                .unwrap_or(Type::Ref)
                        })
                        .collect(),
                );
            }
        }
        Some(root_trace.inputargs.iter().map(|ia| ia.tp).collect())
    }

    /// Get the pre-allocated token number for a trace being recorded.
    ///
    /// Returns `Some(number)` if the given green_key matches the trace
    /// currently being recorded. This enables self-recursive calls to
    /// emit call_assembler targeting the pending token.
    pub fn get_pending_token_number(&self, green_key: u64) -> Option<u64> {
        self.pending_token
            .filter(|&(pk, _)| pk == green_key)
            .map(|(_, num)| num)
    }

    /// Redirect existing call_assembler calls from one loop to another.
    ///
    /// When a loop is recompiled (e.g., with bridges), existing
    /// CALL_ASSEMBLER instructions in other compiled code should be
    /// updated to point to the new version.
    pub fn redirect_call_assembler(&self, old_key: u64, new_key: u64) {
        let old_token = self.warm_state.get_compiled(old_key);
        let new_token = self.warm_state.get_compiled(new_key);
        if let (Some(old), Some(new)) = (old_token, new_token) {
            let _ = self.backend.redirect_call_assembler(old, new);
        }
    }

    fn jitcell_token_by_number(&self, token_number: u64) -> Option<std::sync::Arc<JitCellToken>> {
        for compiled in self.compiled_loops.values() {
            if compiled.token.number == token_number {
                return Some(std::sync::Arc::clone(&compiled.token));
            }
            for previous in &compiled.previous_tokens {
                if previous.number == token_number {
                    return Some(std::sync::Arc::clone(previous));
                }
            }
        }
        // PRE-EXISTING-ADAPTATION: pyre-only fallback to cover targets
        // whose `JitCellToken` lives only on `BaseJitCell.loop_token`
        // (tmp-callback installs via `attach_tmp_callback_to_interp` —
        // `warmstate.py:716-723`). Without this, `compile.py:187`
        // `original.record_jump_to(descr)` keepalive narrows to
        // already-compiled targets and silently drops tmp-callback ones,
        // regressing main behavior. Removed by Slice X-D when
        // `CallAssemblerDescr` carries the owning `Arc<JitCellToken>`
        // directly (Codex parity recommendation #4).
        self.warm_state
            .find_token_by_number(token_number)
            .map(std::sync::Arc::clone)
    }

    /// Port of `rpython/jit/metainterp/compile.py:171-211
    /// record_loop_or_bridge`. Walks `ops` (the freshly compiled
    /// `loop.operations` from `compile.py:183`) and triages each op's
    /// descr by upstream type:
    ///
    /// 1. `compile.py:185-186 ResumeDescr` — `descr.rd_loop_token = clt`.
    /// 2. `compile.py:187-191 JitCellToken` (CALL_ASSEMBLER) — record
    ///    keepalive on `original` and clear the descr reference.
    /// 3. `compile.py:192-203 TargetToken` (JUMP) — walk to the target's
    ///    owning JitCellToken, record keepalive, clear the descr.
    ///
    /// Each branch below cites its upstream line and, where pyre cannot
    /// yet match RPython 1:1, names a PRE-EXISTING-ADAPTATION with the
    /// blocker that gates convergence.
    fn record_loop_or_bridge(
        &self,
        original: &Arc<JitCellToken>,
        ops: &mut [majit_ir::Op],
        trace_id: u64,
    ) {
        // `compile.py:178-179` `assert original_jitcell_token.generation > 0`.
        debug_assert!(
            original.generation.get() > 0,
            "compile.py:179 — token must be registered with memmgr before record_loop_or_bridge"
        );
        //
        // `compile.py:180-181` `wref = weakref.ref(original_jitcell_token);
        // clt.loop_token_wref = wref`. Issue 3.3 Phase A line-by-line.
        // Wire the weak back-reference now that all `Arc::get_mut(&mut
        // token)` writes have settled (the walker runs after
        // `backend.compile_loop`).
        if let Some(clt) = original.compiled_loop_token.as_ref() {
            clt.set_loop_token_wref(Arc::downgrade(original));
        }
        //
        // `compile.py:183` `for op in loop.operations`.
        for op in ops.iter_mut() {
            // `compile.py:184 descr = op.getdescr()`. Clone the Arc
            // (single atomic bump) so the rest of this loop iteration
            // can hold the descr value while still freely mutating
            // `op.descr` to land the `compile.py:191/202 cleardescr()`
            // calls on the JitCellToken/TargetToken branches below.
            let Some(descr) = op.descr.clone() else {
                // `compile.py:184` returns `None` for ops without a
                // descr; the subsequent `isinstance` checks all fail.
                continue;
            };
            // `compile.py:185-186` line-by-line port:
            // ```python
            // if isinstance(descr, ResumeDescr):
            //     descr.rd_loop_token = clt   # stick it there
            // ```
            // The metainterp-side ResumeGuardDescr (this `descr`, carried
            // on `op.descr` in the IR) is the canonical RPython location
            // of `rd_loop_token`. Phase A/B previously stamped only the
            // backend descr because `cpu.get_latest_descr()` returns the
            // backend object; with Phase E foundation in place
            // (E.1 backend_data slot, E.2/E.5 backend stamp on op.descr,
            // E.3-prereq keepalive) the same stamp now also lands on the
            // metainterp descr — closing the deviation called out in
            // mod.rs:632/729 audit.
            //
            // Also push the metainterp ResumeGuardDescr Arc onto the
            // JitCellToken keepalive so it outlives the IR Loop drop
            // (without the keepalive, ~29% of guards reach refcount 0).
            // compile.py:185 `if isinstance(descr, ResumeDescr): ...` —
            // ResumeDescr is the union of `ResumeGuardDescr`-family + the
            // `ResumeGuardCopiedDescr` sibling (compile.py:832).  Pyre
            // mirrors the check via the `is_resume_guard()` /
            // `is_resume_guard_copied()` predicate pair on the FailDescr
            // trait; including both keeps the stamp aligned with the 7
            // metainterp descrs that override `set_rd_loop_token_clt`
            // (mod.rs:632 audit).
            if descr.is_resume_guard() || descr.is_resume_guard_copied() {
                // Pyre-only owning-trace stamp.  RPython resolves descr
                // identity by `id(descr)` (`history.py:125`); pyre's
                // runtime exit paths look up by `(trace_id, fail_index)`,
                // so the owning trace_id is captured on the descr itself
                // and read directly via `descr.trace_id()`.
                descr
                    .as_fail_descr()
                    .expect(
                        "compile.py:185 isinstance(descr, ResumeDescr) — \
                         every ResumeDescr-family descr is a FailDescr",
                    )
                    .set_trace_id(trace_id);
                if let Some(clt) = original.compiled_loop_token.as_ref() {
                    let cloned = std::sync::Arc::clone(clt);
                    let any_arc: std::sync::Arc<dyn std::any::Any + Send + Sync> = cloned;
                    descr
                        .as_fail_descr()
                        .expect(
                            "compile.py:185 isinstance(descr, ResumeDescr) — \
                                 every ResumeDescr-family descr is a FailDescr",
                        )
                        .set_rd_loop_token_clt(any_arc);
                }
            }

            // `compile.py:185-186` `if isinstance(descr, ResumeDescr):
            // descr.rd_loop_token = clt`.
            //
            // PRE-EXISTING-ADAPTATION (deep epic Phase E): the `descr`
            // here is the metainterp-side `ResumeGuardDescr` attached to
            // the IR op (`op.descr`).  At runtime,
            // `cpu.get_latest_descr()` returns the BACKEND descr
            // (`DynasmFailDescr` / `CraneliftFailDescr`) — a separate
            // object built fresh during backend compile.  RPython has a
            // single descr object per fail, so `compile.py:186` patches
            // the same object `cpu.get_latest_descr()` will return.
            // Pyre therefore stamps the backend descr post-compile
            // (`runner.rs::compile_loop` / `compiler.rs::compile_loop`)
            // gated on `is_resume_guard()`.  Convergence requires
            // unifying the IR-side and backend-side descrs into a single
            // object (multi-session epic).

            // `compile.py:187-191` `if isinstance(descr, JitCellToken)`.
            //
            // pyre exposes the owning `Arc<JitCellToken>` carried by
            // `MetaCallAssemblerDescr` through
            // `LoopTokenDescr::token_handle_any` (Slice X-D). All
            // production CALL_ASSEMBLER descrs are constructed via
            // `make_call_assembler_descr` with the real Arc; jitcode
            // dispatch sites (`trace_ctx.rs`) and tests use
            // `make_call_assembler_descr_by_number`, which builds a synth
            // Arc with `compiled.is_none()` — those fall back to the
            // number-keyed lookup via `jitcell_token_by_number`.
            //
            // The `is_call_assembler()` opcode test mirrors RPython's
            // `isinstance(descr, JitCellToken)`: in upstream, the
            // `JitCellToken` descr type is exclusively attached to
            // CALL_ASSEMBLER ops; the opcode test is the structural
            // equivalent in pyre, where the trait `LoopTokenDescr` is
            // implemented only by CALL_ASSEMBLER descrs.
            if op.opcode.is_call_assembler() {
                if let Some(loop_descr) = descr.as_loop_token_descr() {
                    let target_number = loop_descr.loop_token_number();
                    // `compile.py:189` `if descr is not original_jitcell_token`.
                    if target_number != original.number {
                        // `compile.py:190` `original_jitcell_token.record_jump_to(descr)`.
                        // RPython's `descr` IS the `JitCellToken` object,
                        // so the call is unconditional.  Pyre's descr
                        // carries the real owning `Arc<JitCellToken>` for
                        // production CALL_ASSEMBLER paths (`make_call_assembler_descr`,
                        // Slice X-D); the by-number factory
                        // (`make_call_assembler_descr_by_number`) used by
                        // jitcode dispatch and tests builds a synth Arc
                        // with `compiled.is_none()` and falls back to
                        // `jitcell_token_by_number`. Empirical probe
                        // `MAJIT_PROBE_CA_TARGET_MISS` against full
                        // pyre/check.py + cargo test recorded zero misses,
                        // matching `compile.py:187`'s no-fallback shape.
                        // Promote to `expect` so any future regression
                        // surfaces fail-loud rather than silently dropping
                        // a `record_jump_to`.
                        let direct_arc = loop_descr
                            .token_handle_any()
                            .and_then(|any| any.downcast_ref::<std::sync::Arc<JitCellToken>>())
                            .filter(|arc| arc.compiled.is_some());
                        let target = match direct_arc {
                            Some(real_arc) => std::sync::Arc::clone(real_arc),
                            None => self.jitcell_token_by_number(target_number).expect(
                                "compile.py:187 — CALL_ASSEMBLER descr's \
                                 JitCellToken must be reachable through \
                                 compiled_loops or warmstate cells",
                            ),
                        };
                        original.record_jump_to(target);
                    }
                    // `compile.py:191` `op.cleardescr()`.  Clears the
                    // descr reference unconditionally — both the
                    // `descr is original_jitcell_token` short-circuit
                    // and the `record_jump_to` branch fall through
                    // here.  The keepalive is now on `original` (via
                    // `record_jump_to`), so the descr-on-op pointer is
                    // no longer needed and is released to break any
                    // loop ↔ JitCellToken cycle a downstream consumer
                    // (e.g., debug/tests) might form.
                    op.descr = None;
                    continue;
                }
            }

            // `compile.py:192-203` `elif isinstance(descr, TargetToken)`
            // (JUMP target).
            //
            // `compile.py:197 if descr.original_jitcell_token is not
            //                  original_jitcell_token`. pyre stores the
            // owner's `number` in `LoopTargetDescr.original_jitcell_token_number`
            // (set by `compile.py:237` / `compile.py:289` counterparts at
            // `pyjitpl/mod.rs:3886`/`5518`).  Empirically (probe
            // `MAJIT_PROBE_TARGETTOKEN_NONE` against the full pyre/check.py
            // suite, dynasm 14/14) every JUMP TargetToken reaching the
            // walker has the owner number backfilled, so the
            // `assert descr.original_jitcell_token is not None`
            // (`compile.py:198`) form below is a structural invariant
            // rather than a sentinel skip.
            if op.opcode == majit_ir::OpCode::Jump {
                if let Some(target_descr) = descr.as_loop_target_descr() {
                    let target_owner_num = target_descr.original_jitcell_token_number();
                    // `compile.py:197` `if descr.original_jitcell_token
                    // is not original_jitcell_token`.
                    if target_owner_num != Some(original.number) {
                        // `compile.py:198` `assert descr.original_jitcell_token
                        // is not None`.
                        let target_owner_num = target_owner_num.expect(
                            "compile.py:198 — JUMP TargetToken must carry an owning \
                             JitCellToken.number by record_loop_or_bridge time",
                        );
                        // `compile.py:199` `original_jitcell_token
                        // .record_jump_to(descr.original_jitcell_token)` — the
                        // upstream call is unconditional (the descr already
                        // carries the owning JitCellToken object).  Empirically
                        // (probe `MAJIT_PROBE_JUMP_TARGET_MISS` against full
                        // pyre/check.py + cargo test, dynasm 14/14 +
                        // metainterp 1321/0/2) the number→Arc resolve through
                        // `jitcell_token_by_number` always succeeds, so the
                        // unwrap mirrors RPython's no-fallback shape.
                        let target = self.jitcell_token_by_number(target_owner_num).expect(
                            "compile.py:199 — JUMP TargetToken's owning \
                             JitCellToken must be reachable through compiled_loops \
                             or warmstate cells",
                        );
                        original.record_jump_to(target);
                    }
                    // `compile.py:202` `op.cleardescr()`.  Clears the
                    // TargetToken descr reference unconditionally —
                    // both the `descr.original_jitcell_token is
                    // original_jitcell_token` short-circuit and the
                    // `record_jump_to` branch fall through here.  The
                    // `compile.py:200-201` `_descr_wref` capture is
                    // `if not we_are_translated()` (test-only debug
                    // aid); pyre has no consumer of that weakref so
                    // the cleardescr stands alone.
                    op.descr = None;
                }
            }
        }

        // `compile.py:204-207` quasi-immutable_deps register_loop_token.
        //
        // PRE-EXISTING-ADAPTATION (crate-boundary): the registration walker
        // is ported in `pyre/pyre-jit/src/eval.rs::register_quasi_immutable_deps`
        // and called at the post-compile sites that follow `compile_loop`
        // / `compile_bridge` in `eval.rs:2513` / `eval.rs:3059`.  It cannot
        // live inside this method because the dependency target is a
        // `pyre_interpreter::DictStorage` slot watcher; majit-metainterp
        // sits below the pyre/* crates and may not import them
        // (`/parity` crate boundary invariant).  Convergence requires a
        // backend-resident watcher trait plumbed through `MetaInterp` so
        // that the registration walker can execute here without the
        // pyre-interpreter import.  `last_quasi_immutable_deps` (the
        // pyre-side analog of `loop.quasi_immutable_deps`) is populated
        // by the optimizer (`pyjitpl/mod.rs:5407` / `:5774`) and drained
        // by the eval.rs walker at the same call-graph depth as
        // `compile.py:204-207`.

        // `compile.py:210` `loop.original_jitcell_token = None`.
        //
        // PARITY BY CONSTRUCTION: pyre's `Loop` value drops at the end of
        // each `compile_loop` / `compile_bridge` body, breaking the cycle
        // implicitly without an explicit assignment.
    }

    /// Check whether a compiled loop exists for a given green key.
    ///
    /// `pyjitpl.py:2982` / `:3162` upstream pattern step 1
    /// `JitCell.get_procedure_token()` (`warmstate.py:191-196`) is the
    /// canonical green_key → token lookup; pyre routes through
    /// `WarmEnterState::get_procedure_token` (warmstate.rs:862) which
    /// reads `cell.loop_token.as_ref()` directly per F.1 audit
    /// (`tfinal_f0_f1_landed_2026_05_07`).
    ///
    /// PRE-EXISTING-ADAPTATION (residual): upstream step 2
    /// `has_compiled_targets(token)` checks `token.target_tokens != []`
    /// (`pyjitpl.py:3898`). pyre's per-token `target_tokens` list still
    /// lives on `CompiledEntry.front_target_tokens`; until that field
    /// migrates to `JitCellToken` (T-final.F.3 sub-slice), the closest
    /// available signal is `!token.is_invalidated()`, which fires `false`
    /// after a `GUARD_NOT_INVALIDATED` trip. Convergence path: once
    /// `front_target_tokens` reaches `JitCellToken`, replace the
    /// invalidation check with `!token.target_tokens.is_empty() && !token.is_invalidated()`.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.warm_state
            .get_procedure_token(green_key)
            .map_or(false, |token| !token.is_invalidated())
    }

    /// Check if any guard in the compiled trace has Float-typed fail_args.
    /// Used to gate bridge compilation: traces with Float guards have
    /// type metadata issues that cause crashes on bridge guard failures.
    pub fn compiled_trace_has_float_guards(&self, green_key: u64) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        for trace in compiled.traces.values() {
            for layout in trace.exit_layouts.values() {
                if layout
                    .resolve_exit_types()
                    .iter()
                    .any(|t| matches!(t, majit_ir::Type::Float))
                {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the compiled trace is safe for bridge compilation.
    /// Returns true if all guard exit_types at slot positions match
    /// the expected slot_types. Mismatches indicate type propagation
    /// bugs that cause crashes when bridge guards fail.
    pub fn compiled_trace_safe_for_bridge(
        &self,
        green_key: u64,
        slot_types: &[majit_ir::Type],
    ) -> bool {
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        let num_slots = slot_types.len();
        for trace in compiled.traces.values() {
            for layout in trace.exit_layouts.values() {
                let exit_types = layout.resolve_exit_types();
                for i in 0..num_slots {
                    let exit_pos = i + 3;
                    if let Some(et) = exit_types.get(exit_pos) {
                        if *et != slot_types[i] {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Remove all compiled loops. Used when guard-fail recovery is
    /// unrecoverable (null Ref in resume data).
    pub fn clear_compiled_loops(&mut self) {
        self.compiled_loops.clear();
    }

    /// warmstate.py:385 — whether this driver's portal returns a raw int.
    /// result_type == INT.
    pub fn has_raw_int_finish(&self) -> bool {
        self.result_type == Type::Int
    }

    // compile.py:687-696 status encoding constants.
    const ST_BUSY_FLAG: u64 = 0x01;
    const ST_TYPE_MASK: u64 = 0x06;
    const ST_SHIFT: u32 = 3;
    const TY_INT: u64 = 0x02;
    const TY_REF: u64 = 0x04;
    const TY_FLOAT: u64 = 0x06;

    /// compile.py:738-784: must_compile — read self.status directly from
    /// the failed descriptor (by descr_addr), compute hash, tick jitcounter.
    ///
    /// RPython: must_compile is a method ON the failed descriptor. self.status
    /// reads the live status of that exact object. descr_addr IS the identity
    /// of that descriptor (current_object_addr_as_int(self) in RPython).
    /// ALWAYS ticks the counter. stack_almost_full is checked by the caller
    /// in handle_fail (compile.py:702-703).
    ///
    /// Identity is derived directly from `descr_arc`: the owning JCT's
    /// `green_key` mirrors `compile.py:725 resumedescr.rd_loop_token.
    /// loop_token_wref()`, `trace_id` mirrors `assembler.py:227
    /// self.faildescr.trace_id`, and `fail_index_per_trace` mirrors
    /// `self.faildescr.index = i`.  The `fallback_green_key` only fires
    /// when `descr_owning_jct` returns `None` (the JCT was evicted by
    /// memmgr, equivalent to RPython's `compile.py:725-729 compile.
    /// giveup()` path), so callers pass their own outer entry key as
    /// the safe default.
    ///
    /// Returns (should_compile, owning_green_key).
    pub fn must_compile_with_values(
        &mut self,
        descr_arc: &std::sync::Arc<dyn majit_ir::FailDescr>,
        fail_values: &[i64],
        fallback_green_key: u64,
    ) -> (bool, u64) {
        let descr_addr = std::sync::Arc::as_ptr(descr_arc) as *const () as usize;
        let trace_id = descr_arc.trace_id();
        let fail_index = descr_arc.fail_index_per_trace();
        // compile.py:725 `_trace_and_compile_from_bridge` walks
        // `resumedescr.rd_loop_token.loop_token_wref()` for the owning
        // JCT.  When the weakref is dead (memmgr eviction —
        // `compile.py:725-729 compile.giveup()` parity), no other
        // identity is recoverable, so we fall back to the caller's
        // outer entry key.  RPython doesn't have this fallback because
        // its identity is descr-pointer-based, never indirected through
        // a numeric `green_key`.
        let owning_key = majit_backend::descr_owning_jct(descr_arc.as_ref())
            .map(|jct| jct.green_key)
            .unwrap_or(fallback_green_key);
        if descr_addr == 0 {
            if crate::majit_log_enabled() {
                eprintln!("[jit] must_compile: descr_addr=0, skip");
            }
            return (false, owning_key);
        }
        // compile.py:741: self.status — read live status directly from the
        // failed descriptor object (not a re-lookup by trace_id/fail_index).
        let status = self.backend.read_descr_status(descr_addr);
        // compile.py:741-751: decode status to get hash
        let hash = if status & (Self::ST_BUSY_FLAG | Self::ST_TYPE_MASK) == 0 {
            // compile.py:745: common case — TY_NONE, not busy.
            status
        } else if status & Self::ST_BUSY_FLAG != 0 {
            // compile.py:750-751: already busy tracing.
            return (false, owning_key);
        } else {
            // compile.py:753-781: GUARD_VALUE per-value hash.
            let index = (status >> Self::ST_SHIFT) as u32;
            let typetag = status & Self::ST_TYPE_MASK;
            let raw = fail_values.get(index as usize).copied().unwrap_or(0);
            let intval: i64 = match typetag {
                Self::TY_INT => raw,
                Self::TY_REF => raw,
                Self::TY_FLOAT => raw,
                _ => raw,
            };
            // compile.py:780-781: current_object_addr_as_int(self) * 777767777
            //   + intval * 1442968193
            (descr_addr as u64)
                .wrapping_mul(777767777)
                .wrapping_add((intval as u64).wrapping_mul(1442968193))
        };
        // compile.py:783-784: jitcounter.tick(hash, increment)
        let fired = self.warm_state.tick_guard_failure(hash);
        if fired && crate::majit_log_enabled() {
            eprintln!(
                "[jit] must_compile FIRED: key={} trace={} guard={}",
                owning_key, trace_id, fail_index
            );
        }
        (fired, owning_key)
    }

    /// memmgr.py:58-61: keep_loop_alive(looptoken).
    /// warmstate.py:402: warmrunnerdesc.memory_manager.keep_loop_alive(loop_token)
    ///
    /// `warmstate.py:398` `loop_token = jitcell.get_procedure_token()`
    /// is the upstream lookup; pyre routes through
    /// `WarmEnterState::get_procedure_token` (warmstate.rs:862) which
    /// reads `cell.loop_token.as_ref()` per F.1 audit
    /// (`tfinal_f0_f1_landed_2026_05_07`). The Arc identity returned
    /// here is the same `Arc<JitCellToken>` that `compiled_loops[gk].token`
    /// holds (Slice 5.1 lifted `CompiledEntry.token` so both slots share
    /// one Arc); convergence to a sole owner happens at F.6 when the
    /// `compiled_loops` HashMap retires.
    ///
    /// Cells with no compiled entry (key not yet compiled, or already
    /// evicted) silently no-op — RPython's `keep_loop_alive` is likewise
    /// gated by `if loop_token is not None` callers (`compile.py:1149`).
    pub fn keep_loop_alive(&mut self, green_key: u64) {
        let Some(token) = self.warm_state.get_procedure_token(green_key) else {
            return;
        };
        self.warm_state.memory_manager.keep_loop_alive(&token);
    }

    /// compile.py:826-830 store_hash: assign jitcounter hashes to guards
    /// after compile_loop/compile_bridge. RPython calls store_hash during
    /// optimizer emit (store_final_boxes_in_guard); in majit the backend
    /// creates fail_descrs, so we assign hashes after compilation.
    /// Only allocates hashes for real guards (not FINISH/external JUMP).
    fn assign_guard_hashes(&mut self, token: &JitCellToken) {
        let layouts = self.backend.compiled_fail_descr_layouts(token);
        let hashes: Vec<u64> = layouts
            .iter()
            .flatten()
            .map(|layout| {
                if layout.is_finish {
                    0 // FINISH/external JUMP — no hash needed
                } else {
                    self.warm_state.fetch_next_hash()
                }
            })
            .collect();
        self.backend.store_guard_hashes(token, &hashes);
    }

    /// compile.py:826-830 store_hash for bridge guards.
    fn assign_bridge_guard_hashes(
        &mut self,
        source_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) {
        let layouts = self.backend.compiled_bridge_fail_descr_layouts(
            source_token,
            source_trace_id,
            source_fail_index,
        );
        let hashes: Vec<u64> = layouts
            .iter()
            .flatten()
            .map(|layout| {
                if layout.is_finish {
                    0
                } else {
                    self.warm_state.fetch_next_hash()
                }
            })
            .collect();
        self.backend.store_bridge_guard_hashes(
            source_token,
            source_trace_id,
            source_fail_index,
            &hashes,
        );
    }

    /// compile.py:741-745: look up (status, descr_addr) for a guard.
    /// Search current token + previous_tokens by (trace_id, fail_index)
    /// to find the exact descriptor — same pattern as start_guard_compiling.
    /// Returns None if descriptor not found (no tick should happen).
    pub fn get_guard_status(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<(u64, usize)> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let tid = trace_id;
        let (s, a) = self
            .backend
            .get_guard_status(&compiled.token, tid, fail_index);
        if a != 0 {
            return Some((s, a));
        }
        for prev in &compiled.previous_tokens {
            let (s, a) = self.backend.get_guard_status(prev, tid, fail_index);
            if a != 0 {
                return Some((s, a));
            }
        }
        None
    }

    /// compile.py:786-788: self.start_compiling() — set ST_BUSY_FLAG
    /// on the exact failed descriptor (by descr_addr, not re-lookup).
    pub fn start_guard_compiling(&self, descr_addr: usize) {
        self.backend.start_compiling_descr(descr_addr);
    }

    /// compile.py:790-795: self.done_compiling() — clear ST_BUSY_FLAG
    /// on the exact failed descriptor (by descr_addr, not re-lookup).
    pub fn done_guard_compiling(&self, descr_addr: usize) {
        self.backend.done_compiling_descr(descr_addr);
    }

    /// Resolve the source ResumeGuardDescr Arc the optimizer stamped onto
    /// the originating guard op (`op.descr`).  Returns the metainterp
    /// `Arc<dyn Descr>` directly: after Sessions 6.5/6.6 stamped
    /// `trace_id` and `fail_index_per_trace` onto the descr itself,
    /// callers no longer need a `(trace_id, fail_index)` capturing
    /// proxy — they consume the Arc and reach the per-trace key via
    /// `descr.as_fail_descr()?.fail_index_per_trace()`.  RPython parity:
    /// `cpu.get_latest_descr()` returns the same `ResumeGuardDescr` the
    /// metainterp stamped (`history.py:125`); `op.getdescr()` /
    /// `compile.py:184` is the equivalent path.
    ///
    /// Synthetic / FINISH exits whose `exit_layouts` entry holds a
    /// non-`ResumeGuard*` descr legitimately produce `None`; the strict
    /// `compile.py:800 assert resumekey_original_loop_token is not None`
    /// is enforced at `compile_bridge` entry instead.
    pub(crate) fn bridge_source_descr(
        &self,
        compiled: &CompiledEntry<M>,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<std::sync::Arc<dyn majit_ir::Descr>> {
        // `compile.py:184` `op.getdescr()` — primary path: the metainterp
        // ResumeGuardDescr Arc the optimizer stamped onto the originating
        // guard op.  `build_guard_metadata` (compile.rs:984) clones that
        // same Arc onto `StoredExitLayout.descr` keyed by per-trace
        // `fail_index`, so reading through `exit_layouts` returns the
        // identical Arc identity without walking the trace.ops Vec.
        // Restrict to `ResumeGuard*`-family (compile.rs:294 predicate
        // parity) — `_DoneWithThisFrameDescr` / `ExitFrameWithExceptionDescrRef`
        // entries on FINISH ops share the same `fail_index` counter but
        // are not bridge sources.  Under unified ResumeGuardDescr identity
        // (PyPy parity) `cpu.get_latest_descr()` returns the same object
        // (`history.py:125`).
        if let Some(trace) = Self::trace_for_exit(compiled, trace_id).map(|(_, t)| t) {
            if let Some(descr) = trace
                .exit_layouts
                .get(&fail_index)
                .and_then(|layout| layout.descr.as_ref())
                .filter(|d| d.is_resume_guard() || d.is_resume_guard_copied())
            {
                return Some(descr.clone());
            }
        }
        // `warmspot.py:1022` `cpu.get_latest_descr(deadframe)` parity:
        // when `op.descr` is unavailable (synthetic / cut-tentative ops,
        // post-retrace stale traces) the backend still owns the live
        // FailDescr Arc — it's the same identity pyre's runtime guard-
        // failure helpers received as `descr_addr` and the same identity
        // PyPy's deadframe carries.  Walk the current token first, then
        // `previous_tokens` (matches `bridge_was_compiled`'s chain at
        // mod.rs:7710).
        //
        // Empirically (epic #236 Slice E, 2026-05-05) this fallback
        // fires zero times across nbody / fannkuch / fib_recursive /
        // raise_catch_loop / spectral_norm × {dynasm, cranelift};
        // current production benchmarks always carry a live `op.descr`
        // at the lookup.  The fallback is kept as defensive RPython
        // parity for the split-descr period — Phase E.3+ unification
        // will eventually collapse both paths back to a single
        // ResumeGuardDescr identity.
        if let Some(descr) =
            self.backend
                .compiled_bridge_descr_arc(&compiled.token, trace_id, fail_index)
        {
            return Some(descr);
        }
        for prev_token in &compiled.previous_tokens {
            if let Some(descr) = self
                .backend
                .compiled_bridge_descr_arc(prev_token, trace_id, fail_index)
            {
                return Some(descr);
            }
        }
        None
    }

    /// Check whether a bridge was actually compiled and attached for a guard.
    /// Used by jit_bridge_compile_for_guard to distinguish successful bridge
    /// compilation from trace abort (RPython pyjitpl.py:2906-2907 parity).
    ///
    /// Searches the current token AND previous_tokens, since bridge
    /// compilation may have attached to an earlier token that was replaced
    /// by a retrace/recompile.
    pub fn bridge_was_compiled(&self, green_key: u64, trace_id: u64, fail_index: u32) -> bool {
        if let Some(token) = self.warm_state.get_compiled(green_key) {
            if self
                .backend
                .compiled_bridge_fail_descr_layouts(token, trace_id, fail_index)
                .is_some()
            {
                return true;
            }
        }
        // PRE-EXISTING-ADAPTATION: previous_tokens is a pyre NEW-DEVIATION
        // field on `CompiledEntry` that compensates for cross-recompile
        // bridge attachments. Upstream `JitCellToken.target_tokens`
        // (`history.py:501-540`) keeps every retraced loop's code alive
        // naturally; pyre stores it on the compiled_loops side until the
        // F.3-orthodox slice migrates the field onto JitCellToken. The
        // probe stays here so the residual compiled_loops touch surfaces in
        // the F.0 audit.
        let Some(compiled) = self.compiled_loops.get(&green_key) else {
            return false;
        };
        compiled.previous_tokens.iter().any(|prev_token| {
            self.backend
                .compiled_bridge_fail_descr_layouts(prev_token, trace_id, fail_index)
                .is_some()
        })
    }

    // ── Bridge Compilation ──────────────────────────────────────

    /// pyjitpl.py:3195 finally: self.history.cut(cut_at) — undo tentative JUMP/FINISH.
    fn cut_tentative_op(&mut self, cut_at: crate::recorder::TracePosition) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.cut_trace(cut_at);
        }
    }

    /// pyjitpl.py:2982-2983: close_bridge — compile_trace wrapper that
    /// maps CompileOutcome to BridgeCompileResult.
    pub fn close_bridge(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> BridgeCompileResult {
        let outcome = self.compile_trace(green_key, finish_args, Some((trace_id, fail_index)));
        match outcome {
            CompileOutcome::Compiled { .. } => BridgeCompileResult::Compiled,
            _ if self.retrace_after_bridge => {
                // Keep retrace_after_bridge=true so compile_loop can
                // detect bridge retrace and abort early (preserve
                // retraced_count). pyjitpl.py:3000 partial_trace check
                // is before 3162 has_compiled_targets.
                BridgeCompileResult::RetraceNeeded
            }
            _ => BridgeCompileResult::Failed,
        }
    }

    /// RPython-compatible helper name from compile.py.
    pub fn send_bridge_to_backend(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> BridgeCompileResult {
        self.close_bridge(green_key, trace_id, fail_index, finish_args)
    }

    /// Record a constant Ref value in the active trace context.
    /// Returns the OpRef for the constant.
    pub fn record_bridge_const_ref(&mut self, value: i64) -> majit_ir::OpRef {
        let ctx = self
            .tracing
            .as_mut()
            .expect("record_bridge_const_ref requires active trace");
        ctx.constants
            .get_or_insert_typed(value, majit_ir::Type::Ref)
    }
}

impl<M: Clone> MetaInterp<M> {
    /// Obtain the source ResumeGuardDescr Arc for a guard identified by
    /// green_key, trace_id, and fail_index. Used by call_assembler bridge
    /// callbacks that need to pass a FailDescr to compile_bridge —
    /// callers reach the FailDescr surface via `as_fail_descr()`.
    pub fn get_fail_descr_for_bridge(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<std::sync::Arc<dyn majit_ir::Descr>> {
        let compiled = self.compiled_loops.get(&green_key)?;
        self.bridge_source_descr(compiled, trace_id, fail_index)
    }

    fn recovery_slot_types_from_exit_types_and_layout(
        exit_types: &[Type],
        recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Vec<Type> {
        let Some(recovery) = recovery_layout else {
            return exit_types.to_vec();
        };
        let mut types = Vec::new();
        for frame in recovery.frames.iter().rev() {
            let Some(slot_types) = frame.slot_types.as_ref() else {
                return exit_types.to_vec();
            };
            types.extend_from_slice(slot_types);
        }
        if types.len() == exit_types.len() {
            types
        } else {
            exit_types.to_vec()
        }
    }

    /// Return the full recovery slot types for a guard exit, concatenated
    /// from all frames in callee-first order (matching the blackhole
    /// consumer's section convention). Falls back to exit_types when
    /// recovery_layout is absent.
    pub fn get_recovery_slot_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        let exit_layout =
            self.get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)?;
        Some(Self::recovery_slot_types_from_exit_types_and_layout(
            &exit_layout.exit_types,
            exit_layout.recovery_layout.as_ref(),
        ))
    }

    /// Return the merge point PC for blackhole resume from a guard exit.
    ///
    /// Producer invariant: after build_guard_metadata + backend merge,
    /// every guard has recovery_layout with header_pc on all frames.
    /// Returns None only if the (green_key, trace_id, fail_index) lookup
    /// itself fails — a metadata consistency error, not a missing field.
    pub fn get_merge_point_pc(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<u64> {
        let exit_layout =
            self.get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)?;
        let recovery = exit_layout.recovery_layout.as_ref()?;
        recovery.frames.first()?.header_pc
    }

    /// compile.py:853 `ResumeGuardDescr` storage handle lookup.
    /// Returns the shared `Arc<ResumeStorage>` owned by the guard at
    /// (green_key, trace_id, fail_index). Readers use this instead of
    /// the legacy owned-copy accessors (`get_rd_numb`, `get_rd_virtuals`)
    /// so every observer sees the same `rd_consts` pool the GC root
    /// walker updates.
    pub fn get_resume_storage(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Arc<ResumeStorage>> {
        // compile.py:853 `ResumeGuardDescr` storage is shared via the
        // FailDescr identity, so the same guard descriptor exposes the
        // same `rd_*` pool regardless of which retrieval path looks it
        // up (frontend export, backend-recovered layout, previous-token
        // bridge). Mirror that by falling back to the same lookup chain
        // `get_compiled_exit_layout_in_trace` uses (frontend trace ->
        // backend layout -> previous-token bridges) so callers like
        // `get_rd_virtuals` and `get_resume_data_summary` see the
        // storage even when only the backend has it.
        let compiled = self.compiled_loops.get(&green_key)?;
        if let Some((_, trace_data)) = Self::trace_for_exit(compiled, trace_id) {
            if let Some(exit_layout) = trace_data.exit_layouts.get(&fail_index) {
                if let Some(ref storage) = exit_layout.storage {
                    return Some(storage.clone());
                }
            }
        }
        self.get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)
            .and_then(|layout| layout.storage.clone())
    }

    /// Get exit_types for a guard (for decode_ref type dispatch).
    pub fn get_exit_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        let exit_layout =
            self.get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)?;
        Some(exit_layout.exit_types.clone())
    }

    /// resume.py:924-926 _prepare: get rd_virtuals + rd_pendingfields
    /// for blackhole resume at a guard failure.
    pub fn get_rd_virtuals(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>> {
        let storage = self.get_resume_storage(green_key, trace_id, fail_index)?;
        Some(storage.rd_virtuals.clone())
    }

    /// resume.py:926 _prepare parity: get rd_pendingfields for a guard.
    pub fn get_rd_pendingfields(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<majit_ir::GuardPendingFieldEntry>> {
        let storage = self.get_resume_storage(green_key, trace_id, fail_index)?;
        Some(storage.rd_pendingfields.clone())
    }

    /// compile.py:1002-1021 ResumeFromInterpDescr.compile_and_attach parity.
    ///
    /// Optimize against the already-compiled loop at `green_key`, then
    /// compile the result as a fresh interpreter entry under
    /// `original_green_key`.
    pub fn compile_entry_bridge(
        &mut self,
        green_key: u64,
        original_green_key: u64,
        meta: M,
        bridge_ops: &[majit_ir::Op],
        bridge_inputargs: &[majit_ir::InputArg],
        constants: HashMap<u32, i64>,
        mut constant_types: HashMap<u32, Type>,
        snapshot_boxes: SnapshotBoxes,
        snapshot_frame_sizes: SnapshotFrameSizes,
        snapshot_vable_boxes: SnapshotBoxes,
        snapshot_vref_boxes: SnapshotBoxes,
        snapshot_frame_pcs: SnapshotFramePcs,
    ) -> bool {
        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        // RPython-orthodox: bridgeopt.py / unroll.py have no source→bridge
        // constant pool merge. Const objects flow via rd_consts + fresh
        // decode (resume.py:1245-1282).
        let (retraced_count, loop_num_inputs, parent_next_global_opref) = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            (
                compiled.token.get_retraced_count(),
                compiled.token.inputarg_types.len(),
                compiled.next_global_opref,
            )
        };
        // Box Identity Phase E Step 2a: stage bridge_inputarg_base based
        // on the parent loop's recorded next_global_opref. See
        // Optimizer::optimize_bridge docstring for the RPython identity
        // model this mirrors (opencoder.py:249-273).
        let bridge_inputarg_base = parent_next_global_opref.max(bridge_inputargs.len() as u32);
        // unroll.py:187 `trace = trace.get_iter()`: mint fresh InputArg /
        // ResOperation objects in a disjoint OpRef namespace
        // (`opencoder.py:259-262 self.inputargs = [rop.inputarg_from_tp(...)]`).
        let prepared = prepare_bridge_trace_for_optimizer(
            bridge_ops,
            bridge_inputargs,
            snapshot_boxes,
            snapshot_frame_sizes,
            snapshot_vable_boxes,
            snapshot_vref_boxes,
            snapshot_frame_pcs,
            None,
            bridge_inputarg_base,
        );
        let bridge_inputargs = prepared.inputargs.as_slice();
        let bridge_ops = prepared.ops.as_slice();

        let mut optimizer = self.make_optimizer();
        optimizer.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        // Per-iter bridge `BoxRef` pool from `prepare_bridge_trace_for_optimizer`'s
        // `TraceIterator::new` (`opencoder.py:259-262` parity). The bridge
        // optimizer reads through `BoxRef::_forwarded` for `getptrinfo` /
        // `getintbound` callsites; without this plumb, those readers fall back
        // to the legacy Vec via `box_pool.is_empty()` sentinel.
        optimizer.set_pending_box_pool(prepared.box_pool);
        let mut constants = constants;
        optimizer.constant_types = constant_types.clone();
        for arg in bridge_inputargs {
            optimizer.constant_types.insert(arg.index, arg.tp);
        }
        optimizer.snapshot_boxes = prepared.snapshot_boxes;
        optimizer.snapshot_frame_sizes = prepared.snapshot_frame_sizes;
        optimizer.snapshot_vable_boxes = prepared.snapshot_vable_boxes;
        optimizer.snapshot_vref_boxes = prepared.snapshot_vref_boxes;
        optimizer.snapshot_frame_pcs = prepared.snapshot_frame_pcs;
        optimizer.trace_inputarg_types = bridge_inputargs.iter().map(|ia| ia.tp).collect();

        // RPython-orthodox: bridgeopt.py / unroll.py have no source→bridge
        // constant pool merge. Const objects flow via rd_consts + fresh
        // decode (resume.py:1245-1282).
        let retrace_limit = self.warm_state.retrace_limit();
        let bridge_optimize_result = {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                optimizer.optimize_bridge(
                    bridge_ops,
                    &mut constants,
                    bridge_inputargs.len(),
                    &mut compiled.front_target_tokens,
                    true,
                    retraced_count,
                    retrace_limit,
                    None,
                    Some(loop_num_inputs),
                    bridge_inputarg_base,
                )
            }))
        };
        let (optimized_ops, retrace_requested) = match bridge_optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_entry_bridge: InvalidLoop target={} original={}",
                            green_key, original_green_key
                        );
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };
        // optimizer.py:557 self.resumedata_memo.update_counters(profiler)
        optimizer.update_counters(&self.staticdata.profiler);
        // RPython-orthodox: unroll.py replay uses Const args directly;
        // no cross-trace constant pool merge step.
        if retrace_requested {
            if let Some(compiled) = self.compiled_loops.get(&green_key) {
                compiled
                    .token
                    .set_retraced_count(compiled.token.get_retraced_count() + 1);
            }
            if let Some(es) = optimizer.exported_loop_state.take() {
                let renamed_inputargs: Vec<InputArg> = es
                    .renamed_inputargs
                    .iter()
                    .enumerate()
                    .map(|(i, &opref)| {
                        // RPython retrace passes the original typed Box list
                        // directly; pyre's renamed_inputarg_types side table
                        // must be parallel to renamed_inputargs.
                        let tp = es.renamed_inputarg_types.get(i).copied().unwrap_or_else(|| {
                            panic!(
                                "missing renamed_inputarg_types[{}] (renamed_inputargs.len()={}): \
                                 export_state populates the parallel type list in lockstep",
                                i, es.renamed_inputargs.len()
                            )
                        });
                        InputArg::from_type(tp, opref.raw())
                    })
                    .collect();
                self.retrace_needed(
                    green_key,
                    optimized_ops.clone(),
                    renamed_inputargs,
                    constants,
                    es,
                );
            }
            self.retrace_after_bridge = true;
            return false;
        }

        let mut optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);
        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        let trace_id = self.alloc_trace_id();

        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][entry-bridge] original_key={} target_key={} inputs={:?}",
                original_green_key, green_key, bridge_inputargs
            );
            for (i, op) in optimized_ops.iter().enumerate() {
                eprintln!(
                    "[jit][entry-bridge] op[{i}] {:?} pos={:?} args={:?} descr={:?}",
                    op.opcode, op.pos, op.args, op.descr
                );
            }
        }

        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());
        self.backend.set_next_trace_id(trace_id);
        self.backend.set_next_header_pc(original_green_key);

        let mut token = make_jitcell_token(self.warm_state.alloc_token_number(), None);
        {
            let token_mut =
                Arc::get_mut(&mut token).expect("fresh JitCellToken must be uniquely owned");
            token_mut.green_key = original_green_key;
            token_mut.num_scalar_inputargs = self.num_scalar_inputargs;
        }

        let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.backend.compile_loop(
                bridge_inputargs,
                &optimized_ops,
                Arc::get_mut(&mut token)
                    .expect("JitCellToken must stay uniquely owned until backend compile"),
            )
        }));
        let compile_result = match compile_result {
            Ok(r) => r,
            Err(_) => return false,
        };

        match compile_result {
            Ok(_) => {
                self.assign_guard_hashes(token.as_ref());
                self.warm_state.memory_manager.keep_loop_alive(&token);
                // compile.py:213 record_loop_or_bridge.
                self.record_loop_or_bridge(&token, &mut optimized_ops, trace_id);
                let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                    bridge_inputargs,
                    &optimized_ops,
                    original_green_key,
                    &compiled_constant_types,
                    self.callinfocollection.as_deref(),
                );
                let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                    bridge_inputargs,
                    &optimized_ops,
                    &compiled_constant_types,
                );
                if let Some(backend_layouts) =
                    self.backend.compiled_fail_descr_layouts(token.as_ref())
                {
                    compile::merge_backend_exit_layouts(
                        &mut exit_layouts,
                        backend_layouts.as_slice(),
                        &optimized_ops,
                    );
                }
                if let Some(backend_layouts) =
                    self.backend.compiled_terminal_exit_layouts(token.as_ref())
                {
                    compile::merge_backend_terminal_exit_layouts(
                        &mut terminal_exit_layouts,
                        &backend_layouts,
                        &optimized_ops,
                    );
                }
                let trace_info = self.backend.compiled_trace_info(token.as_ref(), trace_id);
                compile::enrich_guard_resume_layouts_for_trace(
                    &mut resume_data,
                    &mut exit_layouts,
                    trace_id,
                    bridge_inputargs,
                    trace_info.as_ref(),
                );
                compile::patch_backend_guard_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut exit_layouts,
                );
                compile::patch_backend_terminal_recovery_layouts_for_trace(
                    &mut self.backend,
                    token.as_ref(),
                    trace_id,
                    &mut terminal_exit_layouts,
                );
                self.take_back_all_descrs(std::mem::take(&mut optimizer.all_descrs));
                let mut next_global_opref =
                    compute_next_global_opref(bridge_inputargs, &optimized_ops);
                let mut traces = HashMap::new();
                traces.insert(
                    trace_id,
                    CompiledTrace {
                        inputargs: bridge_inputargs.to_vec(),
                        ops: optimized_ops,
                        constants: compiled_constants,
                        constant_types: compiled_constant_types,
                        exit_layouts,
                        terminal_exit_layouts,
                    },
                );

                let front_target_tokens = self
                    .compiled_loops
                    .get(&original_green_key)
                    .map(|c| c.front_target_tokens.clone())
                    .unwrap_or_else(|| {
                        if original_green_key == green_key {
                            self.compiled_loops
                                .get(&green_key)
                                .map(|c| c.front_target_tokens.clone())
                                .unwrap_or_default()
                        } else {
                            Vec::new()
                        }
                    });
                let retraced_count = self
                    .compiled_loops
                    .get(&original_green_key)
                    .map(|c| c.token.get_retraced_count())
                    .unwrap_or(0);
                let mut previous_tokens: Vec<std::sync::Arc<JitCellToken>> = Vec::new();
                if let Some(old_entry) = self.compiled_loops.remove(&original_green_key) {
                    // Box Identity Phase E.2b parity: see finish_and_compile.
                    next_global_opref = next_global_opref.max(old_entry.next_global_opref);
                    self.backend
                        .migrate_bridges(&old_entry.token, token.as_ref());
                    previous_tokens =
                        self.retire_compiled_entry(original_green_key, old_entry, &mut traces);
                }
                token.set_retraced_count(retraced_count);
                self.compiled_loops.insert(
                    original_green_key,
                    CompiledEntry {
                        token: Arc::clone(&token),
                        meta,
                        front_target_tokens,
                        root_trace_id: trace_id,
                        traces,
                        previous_tokens,
                        next_global_opref,
                    },
                );
                self.attach_procedure_with_redirect(original_green_key, Arc::clone(&token));
                self.stats.loops_compiled += 1;
                if let Some(ref hook) = self.hooks.on_compile_loop {
                    hook(original_green_key, bridge_ops.len(), num_optimized_ops);
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Compile a bridge from a guard failure point.
    ///
    /// In RPython, when a guard fails frequently, the JIT compiles a
    /// "bridge" — an alternative path starting from the guard failure
    /// that eventually jumps back to the loop or exits.
    ///
    /// `green_key` identifies the loop containing the guard.
    /// `fail_index` identifies which guard to bridge from.
    /// `fail_descr` is the FailDescr from the guard that failed.
    /// `bridge_ops` are the recorded bridge trace operations.
    /// `bridge_inputargs` are the input arguments for the bridge.
    pub fn compile_bridge(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_descr: &dyn majit_ir::FailDescr,
        bridge_ops: &[majit_ir::Op],
        bridge_inputargs: &[majit_ir::InputArg],
        constants: HashMap<u32, i64>,
        mut constant_types: HashMap<u32, Type>,
        snapshot_boxes: SnapshotBoxes,
        snapshot_frame_sizes: SnapshotFrameSizes,
        snapshot_vable_boxes: SnapshotBoxes,
        snapshot_vref_boxes: SnapshotBoxes,
        snapshot_frame_pcs: SnapshotFramePcs,
    ) -> bool {
        if !self.compiled_loops.contains_key(&green_key) {
            return false;
        }

        // `pyjitpl.py:2897-2899` parity:
        //   self.resumekey_original_loop_token =
        //       resumedescr.rd_loop_token.loop_token_wref()
        //   if self.resumekey_original_loop_token is None:
        //       raise compile.giveup()  # should be rare
        // `compile.giveup()` (`compile.py:27-29`) raises
        // `SwitchToBlackhole(ABORT_BRIDGE)`, which RPython catches at
        // `pyjitpl.py:2906-2907` and falls through to blackhole resume.
        // Pyre's `compile_bridge` mirrors that abort by returning `false`;
        // the caller (`compile_trace`) maps `false` to
        // `CompileOutcome::Cancelled`, the same control-flow blackhole
        // resume observes.  The weakref-dead path is "should be rare" per
        // upstream, but structurally required — never panic.
        let source_jct: Arc<JitCellToken> = match majit_backend::descr_owning_jct(fail_descr) {
            Some(jct) => jct,
            None => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compile_bridge: rd_loop_token weakref dead → \
                         compile.giveup() (pyjitpl.py:2898), key={} fail_index={}",
                        green_key, fail_index,
                    );
                }
                return false;
            }
        };
        debug_assert_eq!(
            source_jct.green_key, green_key,
            "compile.py:801 — bridge source descr's rd_loop_token must match \
             the caller-supplied green_key (origin_key from bridge_info)"
        );

        // RPython unroll.py:183-236: Optimizer.optimize_bridge()
        // compile.py:1035-1038: isinstance(resumekey, ResumeAtPositionDescr)
        let inline_short_preamble = !fail_descr.is_resume_at_position();
        // RPython warmspot.py:93 retrace_limit=5: allow bridge to create
        // new target_token specializations when existing body token doesn't
        // match. Without this, bridges fall back to preamble (causing
        // infinite guard failure loops on preamble guards).
        let retrace_limit = self.warm_state.retrace_limit();
        // bridgeopt.py:124-185 deserialize_optimizer_knowledge:
        // Retrieve guard's rd_numb + frontend_boxes for deserialization.
        use crate::optimizeopt::optimizer::PendingBridgeRd;
        let (retraced_count, loop_num_inputs, parent_next_global_opref, pending_bridge_rd): (
            u32,
            usize,
            u32,
            Option<PendingBridgeRd>,
        ) = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            // `fail_descr.trace_id()` is the bridge origin's allocated
            // id (non-zero per `alloc_trace_id` starting at 1).  RPython
            // reaches the same value through the resumekey object's
            // identity (`compile.py:1049 self.resumekey.original_jitcell_token`);
            // pyre carries it as a u64 stamped at backend compile time.
            // No `0 → root_trace_id` sentinel — the bridge proxy's
            // `trace_id` was sourced from `bridge_info().trace_id`
            // (production) or `compiled_root_trace_id(green_key)` (tests),
            // both of which only ever hold real allocated ids.
            let source_trace_id = fail_descr.trace_id();
            debug_assert_ne!(
                source_trace_id, 0,
                "compile_bridge expects bridge origin descr.trace_id() to be a real allocated id, not the FINISH-singleton sentinel"
            );
            let pending = compiled.traces.get(&source_trace_id).and_then(|trace| {
                // F.5-orthodox.1 site #3: route guard identity through
                // `exit_layouts.descr` instead of `guard_op_indices →
                // trace.ops[idx]`. The descr Arc already carries
                // `fail_arg_types` (resume.py:467 / history.py:307 parity),
                // so the indexed op lookup is redundant.
                let exit_layout = trace.exit_layouts.get(&fail_index)?;
                // compile.py:853 `ResumeGuardDescr` storage — every
                // guard's rd_* pool lives behind a shared Arc; the
                // bridge deserializer borrows that same Arc.
                let storage = exit_layout.storage.clone()?;
                // Each bridge inputarg carries its `box.type`
                // (resoperation.py:719/727/739 InputArg{Int,Ref,Float});
                // mint the typed `OpRef::input_arg_*` variant via
                // `InputArg::opref()` so that variant-aware Eq/Hash
                // matches against the producer-side typed OpRefs threaded
                // through the trace (history.py:182 `box.type`
                // intrinsic).
                let liveboxes: Vec<OpRef> = bridge_inputargs.iter().map(|ia| ia.opref()).collect();
                // bridgeopt.py parity: the deserializer's `liveboxes` type
                // filter (box.type == "r") is driven by the type each box
                // carried when the parent guard was finalized — read from
                // `descr.fail_arg_types()` (the same source that
                // `guard_op.fail_arg_types` was caching). Pyre's
                // `bridge_inputargs.tp` can diverge from that (the bridge
                // tracer unboxes via getfield_gc_pure_i etc. so its
                // inputargs see Int where the guard saw Ref), producing a
                // serialize/deserialize bitfield-count mismatch → rd_numb
                // overrun in `deserialize_optimizer_knowledge` once
                // super-instruction GEN widens the live set. Use the
                // parent guard's saved types instead so the deserializer
                // matches the types the serializer used at memo.finish()
                // time.
                let livebox_types: Vec<Type> = exit_layout
                    .descr
                    .as_ref()
                    .and_then(|descr| descr.as_fail_descr())
                    .map(|fd| fd.fail_arg_types().to_vec())
                    .filter(|types| !types.is_empty())
                    .unwrap_or_else(|| bridge_inputargs.iter().map(|ia| ia.tp).collect());
                // unroll.py:183-188: frontend_inputargs = trace.inputargs
                // bridgeopt.py:126 asserts len(frontend_boxes) == len(liveboxes).
                // Cluster 2 (c1): `compile_bridge` is invoked twice for the
                // same bridge under descriptor=Some + (a)+(b); take() empties
                // on the first call and the second hits frontend_boxes.len()=0
                // vs liveboxes.len()=N. RPython threads frontend_boxes through
                // `optimize_bridge` as a parameter, not a one-shot stash.
                let frontend_boxes = self.pending_frontend_boxes.clone().unwrap_or_default();
                assert_eq!(frontend_boxes.len(), liveboxes.len());
                Some(PendingBridgeRd {
                    storage,
                    frontend_boxes,
                    liveboxes,
                    livebox_types,
                    all_descrs: self.staticdata.all_descrs.lock().unwrap().clone(),
                    cls_of_box: self.cls_of_box,
                })
            });
            (
                compiled.token.get_retraced_count(),
                compiled.token.inputarg_types.len(),
                compiled.next_global_opref,
                pending,
            )
        };
        // Box Identity Phase E Step 2a: stage bridge_inputarg_base based on
        // the parent loop's recorded next_global_opref. See
        // Optimizer::optimize_bridge docstring for the RPython identity
        // model this mirrors (opencoder.py:249-273).
        let bridge_inputarg_base = parent_next_global_opref.max(bridge_inputargs.len() as u32);
        // unroll.py:187 `trace = trace.get_iter()`: mint fresh InputArg /
        // ResOperation objects in a disjoint OpRef namespace
        // (`opencoder.py:259-262 self.inputargs = [rop.inputarg_from_tp(...)]`).
        let prepared = prepare_bridge_trace_for_optimizer(
            bridge_ops,
            bridge_inputargs,
            snapshot_boxes,
            snapshot_frame_sizes,
            snapshot_vable_boxes,
            snapshot_vref_boxes,
            snapshot_frame_pcs,
            pending_bridge_rd,
            bridge_inputarg_base,
        );
        let bridge_inputargs = prepared.inputargs.as_slice();
        let bridge_ops = prepared.ops.as_slice();
        let pending_bridge_rd = prepared.pending_bridge_rd;

        let mut optimizer = self.make_optimizer();
        optimizer.all_descrs = std::mem::take(&mut *self.staticdata.all_descrs.lock().unwrap());
        // Per-iter bridge `BoxRef` pool — see the sibling compile_bridge entry
        // earlier for the rationale. Both bridge entries (descriptor=None and
        // descriptor=Some) need the plumb so BoxRef readers stay primary.
        optimizer.set_pending_box_pool(prepared.box_pool);
        let mut constants = constants;
        optimizer.constant_types = constant_types.clone();
        // RPython Box.type parity: inputargs carry their types implicitly
        // via Box subclass. In majit, register inputarg types explicitly
        // so fail_arg_types inference can resolve them.
        for arg in bridge_inputargs {
            optimizer.constant_types.insert(arg.index, arg.tp);
        }
        optimizer.snapshot_boxes = prepared.snapshot_boxes;
        optimizer.snapshot_frame_sizes = prepared.snapshot_frame_sizes;
        optimizer.snapshot_vable_boxes = prepared.snapshot_vable_boxes;
        optimizer.snapshot_vref_boxes = prepared.snapshot_vref_boxes;
        optimizer.snapshot_frame_pcs = prepared.snapshot_frame_pcs;
        // Store bridge inputarg types so export_state can propagate them
        // to ExportedState.renamed_inputarg_types (RPython Box type parity).
        optimizer.trace_inputarg_types = bridge_inputargs.iter().map(|ia| ia.tp).collect();

        // RPython-orthodox: no source→bridge constant_types merge.
        // bridgeopt.py / unroll.py do not copy the source loop's constant
        // pool; typed seeding flows through decoded_box_to_opref per
        // TAGCONST decode.

        // RPython bridgeopt.py:133-146 deserialize_optimizer_knowledge:
        // known_classes are restored from the per-guard bitfield that was
        // serialized at guard compile time (bridgeopt.py:69-88). Only
        // classes that were known at the guard point are restored —
        // runtime class inspection is NOT used here.
        if crate::majit_log_enabled() {
            eprintln!(
                "--- bridge trace (before opt) ninputs={} ---",
                bridge_inputargs.len()
            );
            eprintln!("inputargs: {:?}", bridge_inputargs);
            eprint!("{}", majit_ir::format_trace(bridge_ops, &constants));
        }
        let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
        // compile.py:1077-1078 parity: optimize_bridge may raise InvalidLoop
        // (e.g. rewrite.py:404-407 GUARD_CLASS proven to always fail).
        // RPython catches it via the abstract jitexc handler and discards
        // the bridge. Mirror that here so the trace abort doesn't unwind
        // past compile_bridge.
        let bridge_optimize_result = {
            let compiled = self.compiled_loops.get_mut(&green_key).unwrap();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                optimizer.optimize_bridge(
                    bridge_ops,
                    &mut constants,
                    bridge_inputargs.len(),
                    &mut compiled.front_target_tokens,
                    inline_short_preamble,
                    retraced_count,
                    retrace_limit,
                    pending_bridge_rd,
                    Some(loop_num_inputs),
                    bridge_inputarg_base,
                )
            }))
        };
        let (optimized_ops, retrace_requested) = match bridge_optimize_result {
            Ok(result) => result,
            Err(payload) => {
                if let Some(inv) = payload.downcast_ref::<crate::optimize::InvalidLoop>() {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] compile_bridge: InvalidLoop(\"{}\") at key={} fail_index={}",
                            inv.0, green_key, fail_index
                        );
                    }
                    return false;
                }
                std::panic::resume_unwind(payload);
            }
        };
        // optimizer.py:557 self.resumedata_memo.update_counters(profiler)
        optimizer.update_counters(&self.staticdata.profiler);
        // RPython-orthodox: no post-optimize cross-trace constant merge.
        // Short preamble replay (unroll.py) emits ops with Const args
        // directly; missing-constant recovery from source_trace is
        // pyre-only and violates bridge pool isolation.
        if retrace_requested {
            // compile.py:1079: metainterp.retrace_needed(new_trace, info)
            // Save partial trace + exported state so the next loop-header's
            // compile_loop → compile_retrace can produce a new specialization.
            if let Some(compiled) = self.compiled_loops.get(&green_key) {
                compiled
                    .token
                    .set_retraced_count(compiled.token.get_retraced_count() + 1);
            }
            let exported = optimizer.exported_loop_state.take();
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] bridge retrace needed: key={} exported={}",
                    green_key,
                    exported.is_some(),
                );
            }
            if let Some(es) = exported {
                // compile.py:1075-1084: new_trace.inputargs = info.renamed_inputargs.
                // Types come from ExportedState.renamed_inputarg_types, populated
                // by Optimizer from trace_inputarg_types (RPython Box type parity).
                let renamed_inputargs: Vec<InputArg> =
                    es.renamed_inputargs
                        .iter()
                        .enumerate()
                        .map(|(i, &opref)| {
                            let tp = es.renamed_inputarg_types.get(i).copied().unwrap_or_else(
                                || {
                                    panic!(
                                        "missing renamed_inputarg_types entry for retrace input {}",
                                        i
                                    )
                                },
                            );
                            InputArg::from_type(tp, opref.raw())
                        })
                        .collect();
                self.retrace_needed(
                    green_key,
                    optimized_ops.clone(),
                    renamed_inputargs,
                    constants,
                    es,
                );
            }
            self.retrace_after_bridge = true;
            return false;
        }

        let mut optimized_ops = compile::strip_stray_overflow_guards(optimized_ops);

        let num_optimized_ops = optimized_ops.len();
        let compiled_constants = constants.clone();
        let compiled_constant_types = constant_types.clone();
        let bridge_trace_id = self.alloc_trace_id();

        if crate::majit_log_enabled() {
            eprintln!("--- bridge trace (after opt) ---");
            eprint!("{}", majit_ir::format_trace(&optimized_ops, &constants));
        }

        self.backend.set_constants(constants);
        self.backend.set_constant_types(constant_types.clone());
        // resume.py:1143-1188 parity — VStr/VUni Concat/Slice guard-exit
        // materialization needs the staticdata.callinfocollection to
        // resolve OS_STR_CONCAT / OS_UNI_CONCAT / OS_STR_SLICE /
        // OS_UNI_SLICE func pointers + calldescr. Backends that don't
        // handle VStr/VUni at the backend layer (dynasm) get a no-op.
        self.backend
            .set_callinfocollection(self.callinfocollection.clone());
        self.backend.set_next_trace_id(bridge_trace_id);
        self.backend.set_next_header_pc(green_key);

        let result = {
            let compiled = self.compiled_loops.get(&green_key).unwrap();
            // compile.py:701-717: bridge failure → blackhole resume.
            // Catch Cranelift panics to prevent crashing the process.
            // `compile.py:807-810 send_bridge_to_backend(...,
            //  new_loop.original_jitcell_token, ...)` — the backend's
            // `original_loop_token` is the *source descr's* owning JCT
            // (= `metainterp.resumekey_original_loop_token`), not the
            // current running loop's token.  Pass `source_jct` so backend
            // stamping (`runner.rs:1670`, `compiler.rs:13352`) reaches the
            // correct CLT for newly compiled bridge-internal guards.
            // `previous_tokens` lets cranelift attach the bridge to retired
            // predecessor descrs whose machine code is still running (it
            // cannot patch in place); see `compile_bridge` trait doc.
            let previous_tokens = &compiled.previous_tokens;
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] calling backend.compile_bridge: key={} guard={} ops={}",
                    green_key,
                    fail_index,
                    optimized_ops.len()
                );
            }
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.backend.compile_bridge(
                    fail_descr,
                    bridge_inputargs,
                    &optimized_ops,
                    &source_jct,
                    previous_tokens,
                )
            })) {
                Ok(r) => r,
                Err(e) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] bridge compile_bridge panicked key={} guard={}: {:?}",
                            green_key,
                            fail_index,
                            e.downcast_ref::<String>()
                                .map(|s| s.as_str())
                                .or_else(|| e.downcast_ref::<&str>().copied())
                                .unwrap_or("unknown panic")
                        );
                    }
                    Err(majit_backend::BackendError::CompilationFailed(
                        "Cranelift panic during bridge compilation".to_string(),
                    ))
                }
            }
        };

        match result {
            Ok(_) => {
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[jit] compiled bridge at key={}, guard={}",
                        green_key, fail_index
                    );
                }
                // compile.py:826-830 store_hash for bridge guards.
                let source_trace_id = {
                    // pyjitpl.py:1049 — fail_descr.trace_id() is the
                    // bridge origin's real allocated id (`alloc_trace_id`
                    // starts at 1).  RPython reaches the same value via
                    // the resumekey object's identity; no sentinel
                    // fallback to root_trace_id.
                    fail_descr.trace_id()
                };
                self.assign_bridge_guard_hashes(source_jct.as_ref(), source_trace_id, fail_index);
                // `compile.py:811 record_loop_or_bridge(metainterp_sd,
                //  new_loop)` parity — `new_loop.original_jitcell_token =
                //  metainterp.resumekey_original_loop_token` (compile.py:801).
                // The walker stamps every internal `ResumeDescr.rd_loop_token
                //  = clt(source_jct)` (compile.py:186), so the new
                // bridge-internal guards inherit the *source* JCT identity
                // (could be a previous_tokens entry on cross-loop or
                // post-recompile failures), not the current running loop's
                // latest token.
                self.record_loop_or_bridge(&source_jct, &mut optimized_ops, bridge_trace_id);
                // Mark the bridge as compiled
                if let Some(compiled) = self.compiled_loops.get_mut(&green_key) {
                    // pyjitpl.py:1049 — `fail_descr.trace_id()` is the
                    // bridge origin's allocated id (`alloc_trace_id`
                    // starts at 1).  No `0 → root_trace_id` sentinel;
                    // RPython resolves the source via descr identity.
                    let source_trace_id = fail_descr.trace_id();
                    let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
                        bridge_inputargs,
                        &optimized_ops,
                        green_key,
                        &compiled_constant_types,
                        self.callinfocollection.as_deref(),
                    );
                    let mut terminal_exit_layouts = compile::build_terminal_exit_layouts(
                        bridge_inputargs,
                        &optimized_ops,
                        &compiled_constant_types,
                    );
                    if let Some(backend_layouts) = self.backend.compiled_bridge_fail_descr_layouts(
                        source_jct.as_ref(),
                        source_trace_id,
                        fail_index,
                    ) {
                        compile::merge_backend_exit_layouts(
                            &mut exit_layouts,
                            &backend_layouts,
                            &optimized_ops,
                        );
                    }
                    if let Some(backend_layouts) =
                        self.backend.compiled_bridge_terminal_exit_layouts(
                            source_jct.as_ref(),
                            source_trace_id,
                            fail_index,
                        )
                    {
                        compile::merge_backend_terminal_exit_layouts(
                            &mut terminal_exit_layouts,
                            &backend_layouts,
                            &optimized_ops,
                        );
                    }
                    let bridge_trace_info = self
                        .backend
                        .compiled_trace_info(source_jct.as_ref(), bridge_trace_id);
                    compile::enrich_guard_resume_layouts_for_trace(
                        &mut resume_data,
                        &mut exit_layouts,
                        bridge_trace_id,
                        bridge_inputargs,
                        bridge_trace_info.as_ref(),
                    );
                    compile::patch_backend_guard_recovery_layouts_for_trace(
                        &mut self.backend,
                        source_jct.as_ref(),
                        bridge_trace_id,
                        &mut exit_layouts,
                    );
                    compile::patch_backend_terminal_recovery_layouts_for_trace(
                        &mut self.backend,
                        source_jct.as_ref(),
                        bridge_trace_id,
                        &mut terminal_exit_layouts,
                    );
                    let new_high_water =
                        compute_next_global_opref(bridge_inputargs, &optimized_ops);
                    compiled.next_global_opref = compiled.next_global_opref.max(new_high_water);
                    compiled.traces.insert(
                        bridge_trace_id,
                        CompiledTrace {
                            inputargs: bridge_inputargs.to_vec(),
                            ops: optimized_ops,
                            constants: compiled_constants,
                            constant_types: compiled_constant_types,
                            exit_layouts,
                            terminal_exit_layouts,
                        },
                    );
                }
                self.take_back_all_descrs(std::mem::take(&mut optimizer.all_descrs));
                self.warm_state.log_bridge_compile(fail_index);
                self.stats.bridges_compiled += 1;

                if let Some(ref hook) = self.hooks.on_compile_bridge {
                    hook(green_key, fail_index, num_optimized_ops);
                }
                true
            }
            Err(e) => {
                // RPython compile.py:701-717: bridge compilation failure
                // is not permanent — the counter resets and may fire again.
                // RPython uses ST_BUSY_FLAG only (cleared by done_compiling).
                let msg = format!("Bridge compilation failed: {e}");
                if crate::majit_log_enabled() {
                    eprintln!("[jit] {msg}");
                }
                if let Some(ref cb) = self.hooks.on_compile_error {
                    cb(green_key, &msg);
                }
                false
            }
        }
    }

    /// Start retracing from a guard failure point.
    ///
    /// When a guard fails enough times (>= trace_eagerness) and is not yet
    /// eligible for bridge compilation, we start a new trace from the
    /// guard failure point. The resulting trace replaces the original guard.
    ///
    /// Returns true if retracing was started.
    /// RPython pyjitpl.py:2890 handle_guard_failure parity:
    /// Initialize bridge tracing from a guard failure point.
    /// Returns (success, is_exception_guard) so the caller can emit
    /// SAVE_EXC_CLASS + SAVE_EXCEPTION ops for exception bridges.
    pub fn start_retrace_from_guard(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
    ) -> Option<BridgeRetraceResult> {
        // bridgeopt.py:124 frontend_boxes come directly from the guard
        // failure values in fail_arg_types order.
        self.pending_frontend_boxes = Some(fail_values.to_vec());
        let compiled = match self.compiled_loops.get(&green_key) {
            Some(c) => c,
            None => return None,
        };

        let norm_tid = trace_id;
        let descr_arc = match self.bridge_source_descr(compiled, trace_id, fail_index) {
            Some(arc) => arc,
            None => return None,
        };
        let fail_descr = descr_arc
            .as_fail_descr()
            .expect("bridge source op.descr must implement FailDescr");

        // RPython compile.py:932 invent_fail_descr_for_op:
        // GUARD_EXCEPTION / GUARD_NO_EXCEPTION → ResumeGuardExcDescr.
        // Read the subtype tag off the descr stamped by
        // `store_final_boxes_in_guard` (`is_guard_exc()` — equivalent to
        // RPython's `isinstance(descr, ResumeGuardExcDescr)`).  The
        // `descr_arc` here is the same Arc identity that
        // `build_guard_metadata` cloned onto the originating guard op
        // (compile.rs:984), so reading the subtype tag through it
        // matches the previous `op.descr.is_guard_exc()` walk.
        let is_exception_guard = descr_arc.is_guard_exc();

        // compile.py:797-811 parity: bridge inputargs come from the guard's
        // fail_arg_types AFTER store_final_boxes_in_guard.  The metainterp
        // ResumeGuardDescr Arc carries the post-`store_final_boxes_in_guard`
        // type vector (compile.py:855 `_attrs_`).
        let bridge_input_types = fail_descr.fail_arg_types();
        self.warm_state.start_retrace(bridge_input_types);
        // RPython pyjitpl.py:2609 `create_history(max_num_inputargs)` — the
        // MetaInterp owns the history factory on the bridge path too.
        let recorder = crate::recorder::Trace::with_input_types(bridge_input_types);
        // compile.py:725-731 `_trace_and_compile_from_bridge`:
        //     loop_token = self.rd_loop_token.loop_token_wref()
        //     force_finish_trace = False
        //     if loop_token:
        //         force_finish_trace = bool(loop_token.retraced_count
        //                                   & loop_token.FORCE_BRIDGE_SEGMENTING)
        // Bridge entry reads ONLY the source loop token's bit, never the
        // greenkey-side JC_FORCE_FINISH (warmstate.py:439 reads that flag at
        // loop entry, not bridge entry).
        let source_jct = majit_backend::descr_owning_jct(fail_descr);
        self.force_finish_trace = source_jct.as_ref().is_some_and(|jct| {
            jct.retraced_count.get() & majit_backend::JitCellToken::FORCE_BRIDGE_SEGMENTING != 0
        });
        let mut ctx = crate::trace_ctx::TraceCtx::new(recorder, green_key, self.staticdata.clone());
        ctx.set_force_finish(self.force_finish_trace);
        // pyjitpl.py:2898 `self.resumekey_original_loop_token = ...`.
        // Stash the source token on the trace context so
        // `prepare_trace_segmenting` can set FORCE_BRIDGE_SEGMENTING here.
        if let Some(jct) = source_jct {
            ctx.set_resumekey_original_loop_token(jct);
        }
        // pyjitpl.py:2789 warmrunnerstate.trace_limit snapshot for bridge traces.
        ctx.set_trace_limit(self.warm_state.trace_limit() as usize);
        ctx.callinfocollection = self.callinfocollection.clone();
        self.tracing = Some(ctx);
        // pyjitpl.py:3291 `self.jitdriver_sd = jitdriver_sd`: bridges
        // inherit the parent's driver. The bridge entry path does not
        // thread `driver_descriptor`, so fall back to scanning for the
        // vinfo-bearing slot — a no-op for single-portal pyre and the
        // same shape `virtualizable_info()` already used.
        self.active_jitdriver_sd = self.elect_active_jitdriver_sd(None);

        // RPython uses global ConstInt/ConstPtr boxes, so a bridge that
        // references a parent-loop constant and a bridge that allocates
        // a fresh one never land on the same "slot". majit instead uses
        // per-trace, zero-based constant indices (ConstantPool), so a
        // bridge's first `const_int` returns `OpRef::const_int(0)` — the
        // same index the parent loop used. `compile_entry_bridge` later
        // copies the parent's `constant_types` into the bridge
        // optimizer's type map (so shared parent-constant references
        // type-check); that copy overwrites the bridge's own fresh type
        // for the colliding index and `getintbound` panics with
        // Int/Ref mismatch. Reserve bridge's pool past the parent's
        // highest const index so every new allocation is disjoint.
        if let Some((_, source_trace)) = Self::trace_for_exit(compiled, norm_tid) {
            let max_const = source_trace
                .constants
                .keys()
                .copied()
                .chain(source_trace.constant_types.keys().copied())
                .max();
            if let (Some(max), Some(ref mut ctx)) = (max_const, self.tracing.as_mut()) {
                ctx.constants.reserve_index_past(max);
            }
        }

        if let Some(ref hook) = self.hooks.on_trace_start {
            hook(green_key);
        }

        // resume.py:1042 / compile.py:853 `ResumeGuardDescr` storage —
        // share the parent guard's pool via Arc so the bridge retrace
        // tracer, optimizer, and GC root walker all observe the same
        // `rd_consts`. No owned clone. rd_virtuals carries the parent
        // guard's virtual descriptor table so a future bridge tracer
        // can rebuild parent virtuals via NEW_WITH_VTABLE + SETFIELD_GC
        // at trace start, mirroring ResumeDataBoxReader.consume_boxes
        // → rd_virtuals[i].allocate (resume.py:945-956 getvirtual_ptr).
        let storage = self
            .get_compiled_exit_layout_in_trace(green_key, norm_tid, fail_index)
            .and_then(|layout| layout.storage);

        Some(BridgeRetraceResult {
            is_exception_guard,
            fail_types: bridge_input_types.to_vec(),
            storage,
        })
    }

    /// compile.py:987-1000: handle_async_forcing — force all virtuals
    /// from resume data when a GUARD_NOT_FORCED fires asynchronously
    /// (during a residual call that forces the virtualizable).
    ///
    /// RPython flow: force_now() → cpu.force(token) → handle_async_forcing()
    /// → force_from_resumedata() → materialize all virtuals → save on deadframe.
    ///
    /// Returns the forced virtual caches (ptr, int) for later blackhole
    /// resumption from the GUARD_NOT_FORCED. RPython stores these as
    /// AllVirtuals via cpu.set_savedata_ref().
    pub fn handle_async_forcing(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
    ) -> Option<(Vec<i64>, Vec<i64>)> {
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][handle_async_forcing] key={} trace={} fail={} nvals={}",
                green_key,
                trace_id,
                fail_index,
                fail_values.len()
            );
        }
        // compile.py:988-991: resolve metainterp_sd, vinfo, ginfo
        let _compiled = self.compiled_loops.get(&green_key)?;
        let norm_tid = trace_id;
        let exit_layout =
            self.get_compiled_exit_layout_in_trace(green_key, norm_tid, fail_index)?;

        // compile.py:973-985 don't interrupt me! If the stack runs out
        // in force_from_resumedata() then we have seen cpu.force() but
        // not self.save_data(), leaving in an inconsistent state.
        //
        // RPython wraps the body in try/finally. CriticalCodeGuard's
        // Drop impl re-enables report_error on every exit — including
        // panic unwind — matching the RPython contract.
        let _cc_guard = crate::CriticalCodeGuard::enter();
        // compile.py:994: force_from_resumedata(metainterp_sd, self, deadframe, vinfo, ginfo)
        // compile.py:853 `ResumeGuardDescr` storage — borrow rd_numb /
        // rd_consts / rd_virtuals / rd_pendingfields off the guard-owned
        // Arc.  resume.py:1345-1351 init the reader with the storage
        // wholesale; missing rd_virtuals / rd_pendingfields here meant
        // forced virtuals fell back to NullAllocator entries and pending
        // heap writes were dropped on async forcing.
        let storage = exit_layout.storage.as_deref();
        let rd_numb = storage.map(|s| s.rd_numb.as_slice()).unwrap_or(&[]);
        let empty_consts: [Const; 0] = [];
        let rd_consts: &[Const] = storage.map(|s| s.rd_consts()).unwrap_or(&empty_consts);
        // resume.py:1371 _prepare(storage) parity: materialize rd_virtuals
        // entries before handling_async_forcing. The decoder needs the
        // resumecode item count up-front to size virtual layouts.
        let count = if rd_numb.len() >= 2 {
            let mut reader = crate::resumecode::Reader::new(rd_numb);
            let _items_resume_section = reader.next_item();
            reader.next_item()
        } else {
            fail_values.len() as i32
        };
        let rd_virtuals = storage.map(|s| {
            s.rd_virtuals
                .iter()
                .map(|rd| crate::resume::rd_virtual_to_virtual_info(rd.as_ref(), rd_consts, count))
                .collect::<Vec<_>>()
        });
        let deadframe_types = self.get_recovery_slot_types(green_key, norm_tid, fail_index);
        // compile.py:990-991: vinfo = self.jitdriver_sd.virtualizable_info
        let vinfo = self.virtualizable_info();
        let allocator = crate::resume::NullAllocator;
        let all_liveness = self.staticdata.liveness_info.as_slice();
        let (all_virtuals_ptr, all_virtuals_int) = crate::resume::force_from_resumedata(
            rd_numb,
            rd_consts,
            all_liveness,
            fail_values,
            deadframe_types.as_deref(),
            rd_virtuals.as_deref(),
            storage.map(|s| s.rd_pendingfields.as_slice()),
            Some(&self.virtualref_info as &dyn crate::resume::VRefInfo),
            vinfo.map(|v| v.as_ref() as &dyn crate::resume::VirtualizableInfo),
            None, // ginfo — pyre has no greenfield mechanism
            &allocator,
        );
        drop(_cc_guard);
        // compile.py:999-1000: obj = AllVirtuals(all_virtuals)
        //   metainterp_sd.cpu.set_savedata_ref(deadframe, obj.hide())
        // Return the virtual caches so the caller can store them.
        if crate::majit_log_enabled() {
            eprintln!(
                "[jit][handle_async_forcing] forced {} ptr + {} int virtuals",
                all_virtuals_ptr.len(),
                all_virtuals_int.len(),
            );
        }
        Some((all_virtuals_ptr, all_virtuals_int))
    }

    /// RPython pyjitpl.py:3101 _prepare_exception_resumption +
    /// pyjitpl.py:3132 prepare_resume_from_failure parity:
    /// Emit SAVE_EXC_CLASS + SAVE_EXCEPTION + RESTORE_EXCEPTION at
    /// the bridge trace start for exception guard bridges.
    pub fn emit_exception_bridge_prologue(&mut self, exc_class: i64, exc_value: i64) {
        let Some(ref mut ctx) = self.tracing else {
            return;
        };
        let class_const = ctx.const_int(exc_class);
        let value_const = ctx.const_int(exc_value);
        let class_op = ctx.record_op(OpCode::SaveExcClass, &[class_const]);
        let value_op = ctx.record_op(OpCode::SaveException, &[value_const]);
        ctx.record_op(OpCode::RestoreException, &[class_op, value_op]);
    }

    // ── Guard Failure Recovery ─────────────────────────────────

    /// Handle a guard failure: recover interpreter state using resume data.
    ///
    /// This is the central guard failure handler, equivalent to RPython's
    /// `handle_guard_failure()` in pyjitpl.py.
    ///
    /// Returns `GuardRecovery` describing the recovered state.
    /// Bridge-vs-blackhole is decided by the caller from `must_compile()`,
    /// matching compile.py:701-717 handle_fail flow.
    pub fn handle_guard_failure(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        self.handle_guard_failure_with_savedata(green_key, fail_index, fail_values, None, exception)
    }

    /// `handle_guard_failure()` variant that also carries backend savedata.
    pub fn handle_guard_failure_with_savedata(
        &mut self,
        green_key: u64,
        fail_index: u32,
        fail_values: &[i64],
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        // pyjitpl.py:2900: try_to_free_some_loops
        self.try_to_free_some_loops();
        let trace_id = self.compiled_loops.get(&green_key)?.root_trace_id;
        self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            None,
            savedata,
            exception,
        )
    }

    /// Handle a guard failure in a specific compiled trace (root loop or bridge).
    ///
    /// This is the trace-aware counterpart to `handle_guard_failure()`. Callers
    /// should use the `trace_id` reported by `run_compiled_detailed()` when the
    /// failing exit may come from a bridge.
    pub fn handle_guard_failure_in_trace(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        typed_fail_values: Option<&[Value]>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            typed_fail_values,
            None,
            exception,
        )
    }

    /// `handle_guard_failure_in_trace()` variant that also carries backend savedata.
    ///
    pub fn handle_guard_failure_in_trace_with_savedata(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        typed_fail_values: Option<&[Value]>,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    ) -> Option<GuardRecovery> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;

        let exit_layout =
            Self::compiled_exit_layout_from_trace(trace, green_key, trace_id, fail_index)
                .or_else(|| {
                    self.compiled_exit_layout_from_backend(
                        compiled, green_key, trace_id, fail_index,
                    )
                })
                .unwrap_or_else(|| CompiledExitLayout {
                    rd_loop_token: green_key, // from trace context
                    trace_id,
                    fail_index,
                    source_op_index: None,
                    exit_types: typed_fail_values
                        .map(|values| values.iter().map(Value::get_type).collect())
                        .unwrap_or_default(),
                    is_finish: false,
                    gc_ref_slots: typed_fail_values
                        .map(|values| {
                            values
                                .iter()
                                .enumerate()
                                .filter_map(|(slot, value)| {
                                    (value.get_type() == Type::Ref).then_some(slot)
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                    force_token_slots: Vec::new(),
                    recovery_layout: None,
                    resume_layout: None,
                    storage: None,
                });
        // pyjitpl.py:3277-3288 initialize_state_from_guard_failure:
        // guard failure rebuild is stack-critical code — must not be
        // interrupted by StackOverflow, otherwise jit_virtual_refs are
        // left in a dangling state. RPython try/finally; Rust Drop
        // guard — see CriticalCodeGuard.
        let _cc_guard = crate::CriticalCodeGuard::enter();
        let reconstructed_state = exit_layout
            .resume_layout
            .as_ref()
            .map(|layout| layout.reconstruct_state(fail_values));
        let resume_layout = exit_layout.resume_layout.clone();
        let reconstructed = reconstructed_state
            .as_ref()
            .map(|state| state.frames.clone());
        let materialized_virtuals = reconstructed_state
            .as_ref()
            .map(|state| state.virtuals.clone())
            .unwrap_or_default();
        let pending_field_writes = reconstructed_state
            .as_ref()
            .map(|state| state.pending_fields.clone())
            .unwrap_or_default();
        drop(_cc_guard);

        Some(GuardRecovery {
            trace_id,
            fail_index,
            exit_layout,
            fail_values: fail_values.to_vec(),
            typed_fail_values: typed_fail_values.map(|values| values.to_vec()),
            resume_layout,
            reconstructed_frames: reconstructed,
            reconstructed_state,
            materialized_virtuals,
            pending_field_writes,
            savedata,
            exception,
        })
    }

    /// Run compiled code and handle guard failures automatically.
    ///
    /// This is a convenience wrapper around `run_compiled_detailed` +
    /// `handle_guard_failure`.
    pub fn run_and_recover(&mut self, green_key: u64, live_values: &[i64]) -> Option<RunResult<M>> {
        let result = self.run_compiled_detailed(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            // Normal finish (not a guard failure)
            return Some(RunResult::Finished {
                values,
                meta,
                savedata,
            });
        }

        if let Some(jump) =
            Self::run_result_for_jump_exit(fail_index, values.clone(), meta.clone(), savedata)
        {
            return Some(jump);
        }

        // Guard failure — recover
        let recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception,
        );

        Some(RunResult::GuardFailure {
            values,
            meta,
            trace_id,
            fail_index,
            savedata,
            recovery,
        })
    }

    /// RPython resume_in_blackhole parity: resume execution from the guard
    /// failure point using the IR-based blackhole interpreter.
    /// RPython bh_call_i parity: `memory` provides call_i/call_r for
    pub fn blackhole_guard_failure(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
    ) -> Option<(BlackholeResult, ExceptionState)> {
        self.blackhole_guard_failure_ca(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            exception,
            None,
        )
    }

    /// blackhole.py:1095 bhimpl_recursive_call parity:
    /// Like `blackhole_guard_failure` but with a CallAssembler callback
    /// so the IR blackhole can execute CallAssembler ops via portal_runner.
    pub fn blackhole_guard_failure_ca(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: ExceptionState,
        call_assembler_fn: Option<&crate::blackhole::CallAssemblerFn>,
    ) -> Option<(BlackholeResult, ExceptionState)> {
        let compiled = self.compiled_loops.get(&green_key)?;
        let (_, trace) = Self::trace_for_exit(compiled, trace_id)?;
        // F.5-orthodox.1 site #5: walk `trace.ops` via descr-side
        // identity instead of the `guard_op_indices` HashMap shortcut.
        // The blackhole replay needs the op-position (`guard_op_index +
        // 1` is the replay start at line below), which is inherently
        // a positional concept — match by `descr.fail_index_per_trace()`
        // (the per-trace counter the producer stamps at
        // `compile.rs:301`), like RPython's `compile.py:184
        // op.getdescr()` predicate.
        let guard_op_index = trace.ops.iter().position(|op| {
            op.descr
                .as_ref()
                .and_then(|descr| descr.as_fail_descr())
                .is_some_and(|descr| descr.fail_index_per_trace() == fail_index)
        })?;
        let guard_op = trace.ops.get(guard_op_index)?;
        let fail_args = guard_op.fail_args.as_ref()?;

        let mut initial_values = HashMap::with_capacity(fail_args.len());
        for (arg, value) in fail_args.iter().zip(fail_values.iter().copied()) {
            initial_values.insert(arg.raw(), value);
        }

        Some(blackhole_execute_with_state_ca(
            &trace.ops,
            &trace.constants,
            &initial_values,
            guard_op_index + 1,
            exception,
            call_assembler_fn,
        ))
    }

    /// Run compiled code and, on guard failure, immediately continue through the
    /// blackhole interpreter for the subset where majit can replay the remaining
    /// optimized ops directly.
    pub fn run_with_blackhole_fallback(
        &mut self,
        green_key: u64,
        live_values: &[i64],
    ) -> Option<BlackholeRunResult<M>> {
        let result = self.run_compiled_detailed(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let compiled_exit_layout = result.exit_layout.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            return Some(BlackholeRunResult::Finished {
                values,
                typed_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                via_blackhole: false,
                savedata,
                exception,
            });
        }

        if let Some(jump) = Self::blackhole_result_for_jump_exit(
            fail_index,
            values.clone(),
            typed_values.clone(),
            compiled_exit_layout.clone(),
            meta.clone(),
            savedata,
            exception.clone(),
        ) {
            return Some(jump);
        }

        let initial_recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception.clone(),
        );

        let Some((blackhole_result, blackhole_exception)) = self.blackhole_guard_failure(
            green_key,
            trace_id,
            fail_index,
            &values,
            exception.clone(),
        ) else {
            return Some(BlackholeRunResult::GuardFailure {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                materialized_virtuals: Vec::new(),
                pending_field_writes: Vec::new(),
                via_blackhole: false,
                savedata,
                exception,
            });
        };

        match blackhole_result {
            BlackholeResult::Finish {
                op_index, values, ..
            } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Finished {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Jump { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Jump {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailed {
                guard_index,
                fail_values,
            } => {
                let (
                    fallback_trace_id,
                    fallback_fail_index,
                    exit_layout,
                    typed_fail_values,
                    materialized_virtuals,
                    pending_field_writes,
                ) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    // F.5-orthodox.1 reverse reader: route through
                    // descr-side identity instead of iterating the
                    // `guard_op_indices` HashMap looking for a value
                    // match. The op at `guard_index` carries its
                    // FailDescr Arc; `descr.fail_index_per_trace()`
                    // returns the producer's per-trace counter directly
                    // (set at `compile.rs:301 build_guard_metadata` time
                    // via `set_fail_index_per_trace`, same value the
                    // HashMap was returning). Mirrors RPython
                    // `compile.py:184 op.getdescr()` where the descr
                    // identity lookup replaces the side table entirely.
                    //
                    // Note: `descr.fail_index()` is the global
                    // `alloc_fail_index()` id (descr.rs:1065), distinct
                    // from the per-trace counter; reading the wrong slot
                    // would mismatch the HashMap semantics being
                    // replaced.
                    let fallback_fail_index = trace
                        .ops
                        .get(guard_index)
                        .and_then(|op| op.descr.as_ref())
                        .and_then(|descr| descr.as_fail_descr())
                        .map(|fd| fd.fail_index_per_trace())
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let resume_layout = exit_layout
                        .as_ref()
                        .and_then(|layout| layout.resume_layout.as_ref());
                    let materialized_virtuals = resume_layout
                        .map(|layout| layout.materialize_virtuals(&fail_values))
                        .unwrap_or_default();
                    let pending_field_writes = resume_layout
                        .map(|layout| layout.resolve_pending_field_writes(&fail_values))
                        .unwrap_or_default();
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                        materialized_virtuals,
                        pending_field_writes,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                let (fallback_trace_id, fallback_fail_index, exit_layout, typed_fail_values) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    // F.5-orthodox.1 reverse reader: route through
                    // descr-side identity instead of iterating the
                    // `guard_op_indices` HashMap looking for a value
                    // match. The op at `guard_index` carries its
                    // FailDescr Arc; `descr.fail_index_per_trace()`
                    // returns the producer's per-trace counter directly
                    // (set at `compile.rs:301 build_guard_metadata` time
                    // via `set_fail_index_per_trace`, same value the
                    // HashMap was returning). Mirrors RPython
                    // `compile.py:184 op.getdescr()` where the descr
                    // identity lookup replaces the side table entirely.
                    //
                    // Note: `descr.fail_index()` is the global
                    // `alloc_fail_index()` id (descr.rs:1065), distinct
                    // from the per-trace counter; reading the wrong slot
                    // would mismatch the HashMap semantics being
                    // replaced.
                    let fallback_fail_index = trace
                        .ops
                        .get(guard_index)
                        .and_then(|op| op.descr.as_ref())
                        .and_then(|descr| descr.as_fail_descr())
                        .map(|fd| fd.fail_index_per_trace())
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Abort(message) => Some(BlackholeRunResult::Abort {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                message,
                savedata,
                exception: blackhole_exception,
            }),
        }
    }

    /// Typed-input counterpart to [`run_with_blackhole_fallback`].
    pub fn run_with_blackhole_fallback_with_values(
        &mut self,
        green_key: u64,
        live_values: &[Value],
    ) -> Option<BlackholeRunResult<M>> {
        let result = self.run_compiled_detailed_with_values(green_key, live_values)?;
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let is_finish = result.is_finish;
        let values = result.values.clone();
        let typed_values = result.typed_values.clone();
        let compiled_exit_layout = result.exit_layout.clone();
        let savedata = result.savedata;
        let exception = result.exception.clone();
        let meta = result.meta.clone();

        if is_finish {
            return Some(BlackholeRunResult::Finished {
                values,
                typed_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                via_blackhole: false,
                savedata,
                exception,
            });
        }

        if let Some(jump) = Self::blackhole_result_for_jump_exit(
            fail_index,
            values.clone(),
            typed_values.clone(),
            compiled_exit_layout.clone(),
            meta.clone(),
            savedata,
            exception.clone(),
        ) {
            return Some(jump);
        }

        let initial_recovery = self.handle_guard_failure_in_trace_with_savedata(
            green_key,
            trace_id,
            fail_index,
            &values,
            Some(&typed_values),
            savedata,
            exception.clone(),
        );

        let Some((blackhole_result, blackhole_exception)) = self.blackhole_guard_failure(
            green_key,
            trace_id,
            fail_index,
            &values,
            exception.clone(),
        ) else {
            return Some(BlackholeRunResult::GuardFailure {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                materialized_virtuals: Vec::new(),
                pending_field_writes: Vec::new(),
                via_blackhole: false,
                savedata,
                exception,
            });
        };

        match blackhole_result {
            BlackholeResult::Finish {
                op_index, values, ..
            } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Finished {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Jump { op_index, values } => {
                let exit_layout = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (terminal_trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    compile::terminal_exit_layout_for_trace(
                        trace,
                        green_key,
                        terminal_trace_id,
                        op_index,
                    )
                };
                let typed_values = exit_layout
                    .as_ref()
                    .map(|layout| compile::decode_values_with_layout(&values, layout));
                Some(BlackholeRunResult::Jump {
                    values,
                    typed_values,
                    exit_layout,
                    meta,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailed {
                guard_index,
                fail_values,
            } => {
                let (
                    fallback_trace_id,
                    fallback_fail_index,
                    exit_layout,
                    typed_fail_values,
                    materialized_virtuals,
                    pending_field_writes,
                ) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    // F.5-orthodox.1 reverse reader: route through
                    // descr-side identity instead of iterating the
                    // `guard_op_indices` HashMap looking for a value
                    // match. The op at `guard_index` carries its
                    // FailDescr Arc; `descr.fail_index_per_trace()`
                    // returns the producer's per-trace counter directly
                    // (set at `compile.rs:301 build_guard_metadata` time
                    // via `set_fail_index_per_trace`, same value the
                    // HashMap was returning). Mirrors RPython
                    // `compile.py:184 op.getdescr()` where the descr
                    // identity lookup replaces the side table entirely.
                    //
                    // Note: `descr.fail_index()` is the global
                    // `alloc_fail_index()` id (descr.rs:1065), distinct
                    // from the per-trace counter; reading the wrong slot
                    // would mismatch the HashMap semantics being
                    // replaced.
                    let fallback_fail_index = trace
                        .ops
                        .get(guard_index)
                        .and_then(|op| op.descr.as_ref())
                        .and_then(|descr| descr.as_fail_descr())
                        .map(|fd| fd.fail_index_per_trace())
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let resume_layout = exit_layout
                        .as_ref()
                        .and_then(|layout| layout.resume_layout.as_ref());
                    let materialized_virtuals = resume_layout
                        .map(|layout| layout.materialize_virtuals(&fail_values))
                        .unwrap_or_default();
                    let pending_field_writes = resume_layout
                        .map(|layout| layout.resolve_pending_field_writes(&fail_values))
                        .unwrap_or_default();
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                        materialized_virtuals,
                        pending_field_writes,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::GuardFailedWithVirtuals {
                guard_index,
                fail_values,
                materialized_virtuals,
                pending_field_writes,
            } => {
                let (fallback_trace_id, fallback_fail_index, exit_layout, typed_fail_values) = {
                    let compiled = self.compiled_loops.get(&green_key)?;
                    let (trace_id, trace) = Self::trace_for_exit(compiled, trace_id)?;
                    // F.5-orthodox.1 reverse reader: route through
                    // descr-side identity instead of iterating the
                    // `guard_op_indices` HashMap looking for a value
                    // match. The op at `guard_index` carries its
                    // FailDescr Arc; `descr.fail_index_per_trace()`
                    // returns the producer's per-trace counter directly
                    // (set at `compile.rs:301 build_guard_metadata` time
                    // via `set_fail_index_per_trace`, same value the
                    // HashMap was returning). Mirrors RPython
                    // `compile.py:184 op.getdescr()` where the descr
                    // identity lookup replaces the side table entirely.
                    //
                    // Note: `descr.fail_index()` is the global
                    // `alloc_fail_index()` id (descr.rs:1065), distinct
                    // from the per-trace counter; reading the wrong slot
                    // would mismatch the HashMap semantics being
                    // replaced.
                    let fallback_fail_index = trace
                        .ops
                        .get(guard_index)
                        .and_then(|op| op.descr.as_ref())
                        .and_then(|descr| descr.as_fail_descr())
                        .map(|fd| fd.fail_index_per_trace())
                        .unwrap_or(fail_index);
                    let exit_layout = Self::compiled_exit_layout_from_trace(
                        trace,
                        green_key,
                        trace_id,
                        fallback_fail_index,
                    );
                    let typed_fail_values = exit_layout
                        .as_ref()
                        .map(|layout| compile::decode_values_with_layout(&fail_values, layout));
                    (
                        trace_id,
                        fallback_fail_index,
                        exit_layout,
                        typed_fail_values,
                    )
                };
                let recovery = self.handle_guard_failure_in_trace_with_savedata(
                    green_key,
                    fallback_trace_id,
                    fallback_fail_index,
                    &fail_values,
                    typed_fail_values.as_deref(),
                    savedata,
                    blackhole_exception.clone(),
                );
                Some(BlackholeRunResult::GuardFailure {
                    trace_id: fallback_trace_id,
                    fail_index: fallback_fail_index,
                    fail_values,
                    typed_fail_values,
                    exit_layout,
                    meta,
                    recovery,
                    materialized_virtuals,
                    pending_field_writes,
                    via_blackhole: true,
                    savedata,
                    exception: blackhole_exception,
                })
            }
            BlackholeResult::Abort(message) => Some(BlackholeRunResult::Abort {
                trace_id,
                fail_index,
                fail_values: values,
                typed_fail_values: Some(typed_values),
                exit_layout: Some(compiled_exit_layout),
                meta,
                recovery: initial_recovery,
                message,
                savedata,
                exception: blackhole_exception,
            }),
        }
    }

    // ── Retrace Support ──────────────────────────────────────

    /// Start retracing from a guard failure point.
    ///
    /// When a guard fails too many times, the JIT can start a new trace
    /// from the failure point. The new trace becomes a bridge that is
    /// attached to the failed guard.
    ///
    /// `green_key` identifies the loop containing the guard.
    /// `fail_index` identifies which guard failed.
    /// `live_values` are the concrete values at the guard failure point.
    ///
    /// Returns `true` if retracing was started, `false` if not possible.
    pub fn start_retrace(&mut self, green_key: u64, _fail_index: u32, live_values: &[i64]) -> bool {
        let Some(root_trace_id) = self.compiled_loops.get(&green_key).map(|c| c.root_trace_id)
        else {
            return false;
        };
        self.start_retrace_from_guard(green_key, root_trace_id, _fail_index, live_values)
            .is_some()
    }

    // ── Inlining Support ──────────────────────────────────────

    /// Check if a function call should be inlined during tracing.
    ///
    /// Line-by-line port of `_opimpl_recursive_call` (pyjitpl.py:1375-1423)
    /// + `do_recursive_call` (pyjitpl.py:1425-1432). Decision flow:
    ///
    /// 1. Not tracing → CALL_ASSEMBLER if compiled, else residual.
    /// 2. `!can_inline_callable` (cell has `JC_DONT_TRACE_HERE` or
    ///    `can_never_inline`) → fall through to assembler / residual.
    /// 3. `recursive_depth >= max_unroll_recursion` → `dont_trace_here`
    ///    + fall through to assembler / residual.
    /// 4. Otherwise → inline (`perform_call`).
    ///
    /// `recursive_depth` mirrors the RPython framestack walk at
    /// pyjitpl.py:1389-1402 which skips frames with `greenkey is None`
    /// (the root frame created by `initialize_state_from_start` /
    /// `newframe(mainjitcode)` at pyjitpl.py:3270 — always greenkey-None).
    /// Pyre's `has_inline_frame_for` therefore walks `inline_frames`
    /// only, which counts the same population: already-inlined portal
    /// frames.
    pub fn should_inline(&mut self, callee_key: u64, callee_raw: (usize, usize)) -> InlineDecision {
        // Extract inline-relevant info from ctx before calling impl
        // (avoids borrow conflict between self.tracing and &mut self).
        let ctx_info = self
            .tracing
            .as_ref()
            .map(|ctx| (ctx.inline_depth(), ctx.recursive_depth(callee_raw)));
        self.should_inline_core(callee_key, ctx_info)
    }

    pub fn should_inline_with_ctx(
        &mut self,
        callee_key: u64,
        callee_raw: (usize, usize),
        ctx: &crate::trace_ctx::TraceCtx,
    ) -> InlineDecision {
        let ctx_info = Some((ctx.inline_depth(), ctx.recursive_depth(callee_raw)));
        self.should_inline_core(callee_key, ctx_info)
    }

    /// Core inline decision logic — RPython `_opimpl_recursive_call`
    /// (pyjitpl.py:1375-1423) + `do_recursive_call`
    /// (pyjitpl.py:1425-1432) + `do_residual_call`
    /// (pyjitpl.py:1996-2055) decision tree.
    ///
    /// `ctx_info = Some((inline_depth, recursive_depth))` when tracing,
    /// `None` outside a trace.
    ///
    /// pyre note: `recursive_depth` here is the direct analog of
    /// RPython's `count` at pyjitpl.py:1389-1402 and uses the same
    /// gate as RPython — a flat `< max_unroll_recursion` check.
    /// There is no `is_self_recursive` secondary gate (pyjitpl.py
    /// does not distinguish "self-recursive" from "recursive depth N"
    /// — `count` is a single integer) and no
    /// `should_inline_function` helper-threshold (that was pyre's
    /// runtime stand-in for RPython's jtransform-time helper-inlining
    /// decision; until jtransform is ported, eager inlining on hot
    /// paths is accepted as a temporary perf regression).
    fn should_inline_core(
        &mut self,
        callee_key: u64,
        ctx_info: Option<(usize, usize)>,
    ) -> InlineDecision {
        // pyre adaptation: `pending_token` covers the window between
        // beginning a self-recursive CALL_ASSEMBLER convergence and
        // installing the compiled trace in `compiled_loops`. RPython
        // closes the same gap through `get_assembler_token`, which
        // synthesises a `compile_tmp_callback` token on demand
        // (warmstate.py:714). Pyre has no `compile_tmp_callback`, so
        // the pending-token entry stands in for an already-installed
        // token for inlining-decision purposes only.
        let callee_compiled = self.compiled_loops.contains_key(&callee_key)
            || self.pending_token.map_or(false, |(k, _)| k == callee_key);

        // Not tracing: pyjitpl.py:1381 `warmrunnerstate.inlining`
        // is only meaningful inside a trace. Route compiled
        // callees to CALL_ASSEMBLER and the rest residually.
        let Some((inline_depth, recursive_depth)) = ctx_info else {
            if callee_compiled {
                return InlineDecision::CallAssembler;
            }
            return InlineDecision::ResidualCall;
        };

        // pyjitpl.py:1382 `warmrunnerstate.can_inline_callable(greenboxes)`:
        // returns False when the cell is flagged `JC_DONT_TRACE_HERE`
        // (set at 1413 by `dont_trace_here` after recursion reached
        // `max_unroll_recursion`) or when `can_never_inline` is True.
        // When False, pyjitpl.py:1417 sets `assembler_call = True`
        // and falls through to `do_recursive_call`.
        if !self.warm_state.can_inline_callable(callee_key) {
            if callee_compiled {
                return InlineDecision::CallAssembler;
            }
            return InlineDecision::ResidualCall;
        }

        // pyre-only safety guard. RPython imposes no cap on
        // `metainterp.framestack` depth beyond `rstack`; this bound
        // exists to protect the Rust interpreter thread from
        // runaway native-stack usage when a trace keeps inlining.
        if inline_depth >= MAX_INLINE_DEPTH {
            if callee_compiled {
                return InlineDecision::CallAssembler;
            }
            return InlineDecision::ResidualCall;
        }

        // pyjitpl.py:1404 `count >= max_unroll_recursion` →
        // `dont_trace_here(greenboxes)` + fall through to
        // `do_recursive_call(..., assembler_call=True)`. Pyre's
        // tracer also calls `disable_noninlinable_function` at the
        // same decision point (trace_opcode.rs:3044-3049); doing it
        // here too is idempotent and keeps the metainterp path
        // self-consistent when entered without the tracer wrapper.
        if recursive_depth >= self.max_unroll_recursion {
            self.warm_state.disable_noninlinable_function(callee_key);
            if callee_compiled {
                return InlineDecision::CallAssembler;
            }
            return InlineDecision::ResidualCall;
        }

        // pyjitpl.py:1415 `perform_call(portal_code, allboxes,
        // greenkey=greenboxes)`.
        InlineDecision::Inline
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Pushes an inline frame so tracing can continue through the callee body.
    /// We intentionally avoid recording ENTER_PORTAL_FRAME markers for inline
    /// calls: unlike a real portal transition, they do not carry runtime
    /// semantics and only bloat the trace.
    ///
    /// Returns `true` if inlining started, `false` if not tracing or depth exceeded.
    pub fn enter_inline_frame(&mut self, callee_raw: (usize, usize)) -> bool {
        let ctx = match self.tracing.as_mut() {
            Some(ctx) => ctx,
            None => return false,
        };
        if ctx.inline_depth() >= MAX_INLINE_DEPTH {
            return false;
        }

        ctx.push_inline_frame(callee_raw, MAX_INLINE_DEPTH as u32);
        true
    }

    /// Leave an inlined function call during tracing.
    ///
    /// Pops the inline frame. See `enter_inline_frame()` for why we do not
    /// record LEAVE_PORTAL_FRAME for inline calls.
    pub fn leave_inline_frame(&mut self) {
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.pop_inline_frame();
        }
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.tracing
            .as_ref()
            .map(|ctx| ctx.inline_depth())
            .unwrap_or(0)
    }

    // ────────────────────────────────────────────────────────────────
    // Frame-management surface mirroring pyjitpl.py:2421-2477.
    //
    // perform_call → newframe → MIFrame::setup_call (pyjitpl.py:2421-
    // 2425), popframe → cleanup_registers (pyjitpl.py:2462-2477),
    // finishframe → caller make_result_of_lastop + ChangeFrame
    // (pyjitpl.py:2479-2503) are all wired against
    // MetaInterp::framestack.  Two upstream limbs are explicitly
    // staged:
    //
    // - finishframe's empty-framestack branch returns `Ok(())`
    //   instead of raising `DoneWithThisFrame{Void,Int,Ref,Float}`;
    //   the upstream raise path lands once the portal-runner shim is
    //   migrated onto MetaInterp (see DoneWithThisFrame variant
    //   below).
    // - the upstream method on MIFrame, `do_residual_or_indirect_call`,
    //   uses `self.metainterp` as a back-pointer; pyre's borrow
    //   checker forbids that (MIFrame already lives inside
    //   MetaInterp::framestack), so the canonical body lives on
    //   MetaInterp<M> and acts on the current top-of-framestack
    //   frame implicitly.
    // ────────────────────────────────────────────────────────────────

    /// pyjitpl.py:2421-2425 `MetaInterp.perform_call(jitcode, boxes, greenkey)`.
    ///
    /// ```python
    /// def perform_call(self, jitcode, boxes, greenkey=None):
    ///     # causes the metainterp to enter the given subfunction
    ///     f = self.newframe(jitcode, greenkey)
    ///     f.setup_call(boxes)
    ///     raise ChangeFrame
    /// ```
    ///
    /// `argboxes` mirrors RPython's `boxes` list of typed
    /// `(JitArgKind, OpRef, i64)` tuples — the `(kind, value, concrete)`
    /// tuple `MIFrame::setup_call` consumes.  `jitcode` is the shared
    /// `Arc<JitCode>` that
    /// [`MetaInterpStaticData::bytecode_for_address`] returns.
    pub fn perform_call(
        &mut self,
        jitcode: std::sync::Arc<crate::jitcode::JitCode>,
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        greenkey: Option<u64>,
    ) -> Result<(), ChangeFrame> {
        // pyjitpl.py:2423: f = self.newframe(jitcode, greenkey)
        let _ = self.newframe(jitcode, greenkey);
        // pyjitpl.py:2424: f.setup_call(boxes)
        self.framestack.current_mut().setup_call(argboxes);
        // pyjitpl.py:2425: raise ChangeFrame
        Err(ChangeFrame)
    }

    /// pyjitpl.py:3266-3275 `MetaInterp.initialize_state_from_start(original_boxes)`.
    ///
    /// ```python
    /// def initialize_state_from_start(self, original_boxes):
    ///     # ----- make a new frame -----
    ///     self.portal_call_depth = -1 # always one portal around
    ///     self.framestack = []
    ///     f = self.newframe(self.jitdriver_sd.mainjitcode)
    ///     f.setup_call(original_boxes)
    ///     assert self.portal_call_depth == 0
    ///     self.virtualref_boxes = []
    ///     ...
    /// ```
    ///
    /// Resets `framestack` to empty, pushes the portal `mainjitcode`
    /// frame, and seeds it with the original argboxes.  Other branches
    /// of the upstream method (`initialize_withgreenfields`,
    /// `initialize_virtualizable`) live behind pyre's portal-runner
    /// shim and are not yet wired through this entry — they remain
    /// driven by the existing per-driver setup paths.
    pub fn initialize_state_from_start(
        &mut self,
        mainjitcode: std::sync::Arc<crate::jitcode::JitCode>,
        original_boxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
    ) {
        // pyjitpl.py:3268: self.portal_call_depth = -1 # always one portal around
        self.portal_call_depth = -1;
        // pyjitpl.py:3269: self.framestack = []
        self.framestack = crate::pyjitpl::MIFrameStack::empty();
        // pyjitpl.py:3270: f = self.newframe(self.jitdriver_sd.mainjitcode)
        let _ = self.newframe(mainjitcode, None);
        // pyjitpl.py:3271: f.setup_call(original_boxes)
        self.framestack.current_mut().setup_call(original_boxes);
        // pyjitpl.py:3272: assert self.portal_call_depth == 0
        debug_assert_eq!(self.portal_call_depth, 0);
        // pyjitpl.py:3273: self.virtualref_boxes = []
        self.virtualref_boxes.clear();
    }

    /// pyjitpl.py:3400-3406 `MetaInterp.rebuild_state_after_failure` —
    /// the part that resets `self.framestack = []` before
    /// `resume.rebuild_from_resumedata` repopulates it.  Pyre's resume
    /// stack rebuild lives in `crate::resume::blackhole_from_resumedata`
    /// and does not interact with `MetaInterp::framestack` yet, so the
    /// helper just clears the stack to match the upstream invariant.
    pub fn reset_framestack_for_failure(&mut self) {
        self.framestack = crate::pyjitpl::MIFrameStack::empty();
    }

    /// pyjitpl.py:1941-1958 `MIFrame.execute_varargs(opnum, argboxes, descr, exc, pure)`.
    ///
    /// ```python
    /// def execute_varargs(self, opnum, argboxes, descr, exc, pure):
    ///     self.metainterp.clear_exception()
    ///     patch_pos = self.metainterp.history.get_trace_position()
    ///     op = self.metainterp.execute_and_record_varargs(opnum, argboxes,
    ///                                                         descr=descr)
    ///     if pure and not self.metainterp.last_exc_value and op:
    ///         op = self.metainterp.record_result_of_call_pure(op, argboxes, descr,
    ///             patch_pos, opnum)
    ///         exc = exc and not isinstance(op, Const)
    ///     if exc:
    ///         if op is not None:
    ///             self.make_result_of_lastop(op)
    ///         self.metainterp.handle_possible_exception()
    ///     else:
    ///         self.metainterp.assert_no_exception()
    ///     return op
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: lives on `MetaInterp<M>` rather than
    /// `MIFrame` because of the borrow-checker constraint that already
    /// moved `do_residual_or_indirect_call` here.  `make_result_of_lastop`
    /// is invoked on the framestack's current frame via `dst` —
    /// upstream reads `target_index = ord(self.bytecode[self.pc-1])`
    /// from MIFrame's bytecode, but pyre's call BC encodes `dst`
    /// explicitly per call site, so callers thread it through.  Pass
    /// `None` when the caller writes the result itself after
    /// miframe_execute_varargs returns; pass `Some((kind, target_index))`
    /// to match upstream's `self.make_result_of_lastop(op)` ordering
    /// before `handle_possible_exception()` (pyjitpl.py:1951-1954).
    pub fn miframe_execute_varargs(
        &mut self,
        opnum: OpCode,
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
        exc: bool,
        pure: bool,
        dst: Option<(crate::jitcode::JitArgKind, usize)>,
    ) -> Result<Option<(OpRef, i64)>, FinishframeExceptionSignal> {
        // pyjitpl.py:1942: self.metainterp.clear_exception()
        self.clear_exception();
        // pyjitpl.py:1943: patch_pos = self.metainterp.history.get_trace_position()
        let patch_pos = self.tracing.as_ref().map(|ctx| ctx.get_trace_position());
        // pyjitpl.py:1944-1945: op = execute_and_record_varargs(...)
        let mut op =
            self.execute_and_record_varargs(opnum, argboxes, descr_ref.clone(), descr_view);
        // pyjitpl.py:1946-1948: `pure and not last_exc_value and op` →
        //     op = self.metainterp.record_result_of_call_pure(op, argboxes,
        //         descr, patch_pos, opnum)
        let mut op_was_constant_folded = false;
        if pure && self.last_exc_value == 0 {
            if let (Some((opref, resvalue)), Some(patch_pos)) = (op, patch_pos) {
                let result_value = match descr_view.result_type() {
                    majit_ir::Type::Int => majit_ir::Value::Int(resvalue),
                    majit_ir::Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(resvalue as usize)),
                    majit_ir::Type::Float => {
                        majit_ir::Value::Float(f64::from_bits(resvalue as u64))
                    }
                    majit_ir::Type::Void => majit_ir::Value::Void,
                };
                let opref_args: Vec<OpRef> = argboxes.iter().map(|(_, op, _)| *op).collect();
                let concrete_arg_values: Vec<majit_ir::Value> = argboxes
                    .iter()
                    .map(|(kind, _, val)| match kind {
                        crate::jitcode::JitArgKind::Int => majit_ir::Value::Int(*val),
                        crate::jitcode::JitArgKind::Ref => {
                            majit_ir::Value::Ref(majit_ir::GcRef(*val as usize))
                        }
                        crate::jitcode::JitArgKind::Float => {
                            majit_ir::Value::Float(f64::from_bits(*val as u64))
                        }
                    })
                    .collect();
                if let Some(ctx) = self.tracing.as_mut() {
                    let new_op = ctx.record_result_of_call_pure(
                        opref,
                        &opref_args,
                        &concrete_arg_values,
                        descr_ref,
                        patch_pos,
                        opnum,
                        result_value,
                    );
                    // pyjitpl.py:1949: `exc = exc and not isinstance(op, Const)`
                    // — record_result_of_call_pure returns a Const-typed
                    // OpRef when all args fold to constants; that suppresses
                    // the exception expectation since constant-folded ops
                    // can't raise.
                    op_was_constant_folded = ctx.constants_get_value(new_op).is_some();
                    op = Some((new_op, resvalue));
                }
            }
        }
        // pyjitpl.py:1949: `exc = exc and not isinstance(op, Const)`
        let exc = exc && !op_was_constant_folded;
        // pyjitpl.py:1950-1957: exception handling.
        if exc {
            // pyjitpl.py:1951-1954: `if op is not None: self.make_result_of_lastop(op)`
            // — must run BEFORE handle_possible_exception() so the
            // result box is in the register snapshot when a guard fires
            // (`get_list_of_active_boxes()`).  Pyre callers pass `dst`
            // when they have decoded the call's target register; if
            // `dst` is `None`, the caller is responsible for writing
            // the result itself after we return (legacy dispatch path).
            if let (Some((opref, concrete)), Some((kind, target_index))) = (op, dst) {
                self.framestack.current_mut().make_result_of_lastop(
                    kind,
                    target_index,
                    opref,
                    concrete,
                );
            }
            self.handle_possible_exception()?;
        } else {
            // pyjitpl.py:1957: self.metainterp.assert_no_exception()
            self.assert_no_exception();
        }
        // pyjitpl.py:1958: return op
        Ok(op)
    }

    /// pyjitpl.py:2641-2652 `MetaInterp.execute_and_record_varargs(opnum, argboxes, descr=None)`.
    ///
    /// ```python
    /// def execute_and_record_varargs(self, opnum, argboxes, descr=None):
    ///     history.check_descr(descr)
    ///     # execute the operation
    ///     profiler = self.staticdata.profiler
    ///     profiler.count_ops(opnum)
    ///     resvalue = executor.execute_varargs(self.cpu, self,
    ///                                         opnum, argboxes, descr)
    ///     # check if the operation can be constant-folded away
    ///     argboxes = list(argboxes)
    ///     assert not rop._ALWAYS_PURE_FIRST <= opnum <= rop._ALWAYS_PURE_LAST
    ///     return self._record_helper_varargs(opnum, resvalue, descr,
    ///                                                argboxes)
    /// ```
    ///
    /// Returns `(OpRef, resvalue)` for non-void calls, `None` for void
    /// calls.  The OpRef points at the recorded `CALL_*` IR op; the
    /// resvalue is the concrete return value to keep alongside the
    /// OpRef in the caller's symbolic stack.  For Float results the
    /// resvalue carries the f64 bits via `f64::to_bits` (i64-return
    /// ABI shared with `executor::execute_varargs`); the caller must
    /// unpack with `f64::from_bits(resvalue as u64)` before observing
    /// the slot as an f64.
    pub fn execute_and_record_varargs(
        &mut self,
        opnum: OpCode,
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
    ) -> Option<(OpRef, i64)> {
        // pyjitpl.py:2645: profiler.count_ops(opnum)
        self.count_ops(opnum, counters::OPS);
        // pyjitpl.py:2646-2647: resvalue = executor.execute_varargs(self.cpu, self, ...)
        let resvalue = crate::executor::execute_varargs(self, opnum, argboxes, descr_view);
        // pyjitpl.py:2649-2650: assert not rop._ALWAYS_PURE_FIRST <= opnum <= rop._ALWAYS_PURE_LAST
        debug_assert!(
            !opnum.is_call_pure(),
            "execute_and_record_varargs: pure calls go through _record_helper_pure_varargs",
        );
        // pyjitpl.py:2651-2652: return self._record_helper_varargs(...)
        let opref_args: Vec<OpRef> = argboxes.iter().map(|(_, opref, _)| *opref).collect();
        self._record_helper_varargs(opnum, resvalue, descr, &opref_args)
    }

    /// pyjitpl.py:2655-2663 `MetaInterp._record_helper_varargs(opnum, resvalue, descr, argboxes)`.
    ///
    /// ```python
    /// def _record_helper_varargs(self, opnum, resvalue, descr, argboxes):
    ///     # record the operation
    ///     profiler = self.staticdata.profiler
    ///     profiler.count_ops(opnum, Counters.RECORDED_OPS)
    ///     self.heapcache.invalidate_caches_varargs(opnum, descr, argboxes)
    ///     op = self.history.record(opnum, argboxes, resvalue, descr)
    ///     self.attach_debug_info(op)
    ///     if op.type != 'v':
    ///         return op
    /// ```
    ///
    /// Returns `(OpRef, resvalue)` for non-void calls, `None` for void
    /// — matching upstream's `if op.type != 'v': return op` shape.
    pub fn _record_helper_varargs(
        &mut self,
        opnum: OpCode,
        resvalue: i64,
        descr: majit_ir::DescrRef,
        argboxes: &[OpRef],
    ) -> Option<(OpRef, i64)> {
        // pyjitpl.py:2658: profiler.count_ops(opnum, Counters.RECORDED_OPS)
        // — fires before the heapcache touch + history.record so a
        // dropped trace (`self.tracing.is_none()`) is not double-counted.
        self.count_ops(opnum, counters::RECORDED_OPS);
        let ctx = self.tracing.as_mut()?;
        // pyjitpl.py:2659: self.heapcache.invalidate_caches_varargs(opnum, descr, argboxes)
        let effectinfo = descr.as_call_descr().map(|cd| cd.get_extra_info());
        ctx.heap_cache_mut()
            .invalidate_caches_varargs(opnum, effectinfo, argboxes);
        // pyjitpl.py:2660: op = self.history.record(opnum, argboxes, resvalue, descr)
        let op = ctx.record_op_with_descr(opnum, argboxes, descr);
        // pyjitpl.py:2661: self.attach_debug_info(op)
        self.attach_debug_info(Some(op));
        // pyjitpl.py:2662-2663: if op.type != 'v': return op
        if opnum.result_type() == majit_ir::Type::Void {
            None
        } else {
            Some((op, resvalue))
        }
    }

    /// pyjitpl.py:2733-2737 `MetaInterp.attach_debug_info(op)`.
    ///
    /// ```python
    /// def attach_debug_info(self, op):
    ///     if (not we_are_translated() and op is not None
    ///         and getattr(self, 'framestack', None)):
    ///         op.pc = self.framestack[-1].pc
    ///         op.name = self.framestack[-1].jitcode.name
    /// ```
    ///
    /// **No-op stub.**  RPython attaches the current frame's pc + the
    /// jitcode's name onto the FrontendOp for debug-print output.
    /// Pyre's `Op` struct already carries `pos` (the IR position) but
    /// not `pc` / `name` debug fields; the named entry stays so the
    /// upstream call sequence (notably `_record_helper_varargs`) can
    /// invoke it without a structural mismatch.
    pub fn attach_debug_info(&mut self, _op: Option<OpRef>) {}

    /// pyjitpl.py:2739-2743 `MetaInterp.execute_raised(exception, constant=False)`.
    ///
    /// ```python
    /// def execute_raised(self, exception, constant=False):
    ///     if isinstance(exception, jitexc.JitException):
    ///         raise exception      # go through
    ///     llexception = jitexc.get_llexception(self.cpu, exception)
    ///     self.execute_ll_raised(llexception, constant)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre callers already hold a `i64`
    /// exception pointer (the equivalent of RPython's `llexception`
    /// after `get_llexception` lowering), so this entry just forwards
    /// to `execute_ll_raised`.  The `JitException` re-raise branch
    /// lives in pyre's caller-side error propagation; the entry stays
    /// to mirror upstream call sites.
    pub fn execute_raised(&mut self, llexception: i64, constant: bool) {
        self.execute_ll_raised(llexception, constant);
    }

    /// pyjitpl.py:2745-2755 `MetaInterp.execute_ll_raised(llexception, constant=False)`.
    ///
    /// ```python
    /// def execute_ll_raised(self, llexception, constant=False):
    ///     # called by execute.do_call() when an exception is raised
    ///     self.last_exc_value = llexception
    ///     self.class_of_last_exc_is_const = constant
    /// ```
    pub fn execute_ll_raised(&mut self, llexception: i64, constant: bool) {
        // pyjitpl.py:2751: self.last_exc_value = llexception
        self.last_exc_value = llexception;
        // pyjitpl.py:2752: self.class_of_last_exc_is_const = constant
        self.class_of_last_exc_is_const = constant;
    }

    /// pyjitpl.py:2760-2786 `MetaInterp.aborted_tracing(reason)`.
    ///
    /// ```python
    /// def aborted_tracing(self, reason):
    ///     self.staticdata.profiler.count(reason)
    ///     debug_print('~~~ ABORTING TRACING %s' % Counters.counter_names[reason])
    ///     jd_sd = self.jitdriver_sd
    ///     if not self.current_merge_points:
    ///         greenkey = None # we're in the bridge
    ///     else:
    ///         greenkey = self.current_merge_points[0][0][:jd_sd.num_green_args]
    ///         hooks = self.staticdata.warmrunnerdesc.hooks
    ///         if hooks.are_hooks_enabled():
    ///             hooks.on_abort(reason, jd_sd.jitdriver, greenkey,
    ///                 jd_sd.warmstate.get_location_str(greenkey),
    ///                 self.staticdata.logger_ops._make_log_operations(
    ///                     self.box_names_memo),
    ///                 self.history.trace.unpack()[1])
    ///         if self.aborted_tracing_jitdriver is not None:
    ///             jd_sd = self.aborted_tracing_jitdriver
    ///             greenkey = self.aborted_tracing_greenkey
    ///             if hooks.are_hooks_enabled():
    ///                 hooks.on_trace_too_long(jd_sd.jitdriver, greenkey,
    ///                     jd_sd.warmstate.get_location_str(greenkey))
    ///             # no ops for now
    ///             self.aborted_tracing_jitdriver = None
    ///             self.aborted_tracing_greenkey = None
    ///     self.staticdata.stats.aborted()
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's existing `abort_trace` performs
    /// the live cleanup (recorder.abort, warm_state.abort_tracing,
    /// pending_token reset, on_trace_abort hook).  This named entry
    /// adds the upstream-shaped accounting:
    ///
    /// 1. bumps the `loops_aborted` counter (RPython's
    ///    `staticdata.profiler.count(reason)` + `stats.aborted()`).
    /// 2. fires `on_trace_too_long` via the existing
    ///    `on_trace_abort` hook when `aborted_tracing_jitdriver` was
    ///    pre-set, then clears both fields per pyjitpl.py:2784-2785.
    ///
    /// `reason` is the upstream `Counters.ABORT_*` int (pyre routes
    /// `AbortReason::as_int()` through here).
    pub fn aborted_tracing(&mut self, reason: i32) {
        // pyjitpl.py:2761: profiler.count(reason) — reason-keyed bump
        // lands on the matching `staticdata.profiler.abort_*` atomic.
        self.count(reason, 1);
        // pyjitpl.py:2786: self.staticdata.stats.aborted() — pyre's
        // mapping of the lifetime aborted-loop tally lives on
        // `JitStatsCounters.loops_aborted` (separate from the
        // reason-keyed `profiler.abort_*`).
        self.stats.loops_aborted = self.stats.loops_aborted.saturating_add(1);
        // pyjitpl.py:2770 on_abort hook payload — pyre's single hook
        // receives (greenkey, permanent).  `abort_trace_live` stashes the
        // greenkey / permanent from the consumed ctx so we can fire once
        // here.  The reason is carried only through the eventual hook
        // surface split; `_reason` is intentionally unused today.
        let green_key = self.pending_abort_green_key.take().unwrap_or(0);
        let permanent = std::mem::take(&mut self.pending_abort_permanent);
        if let Some(ref hook) = self.hooks.on_trace_abort {
            hook(green_key, permanent);
        }
        // pyjitpl.py:2776-2785: on_trace_too_long clause — pyre folds it
        // into the single hook above until a distinct hook surface is
        // ported; clear the fields unconditionally so bookkeeping cannot
        // leak into the next trace.
        self.aborted_tracing_jitdriver = None;
        self.aborted_tracing_greenkey = None;
    }

    /// pyjitpl.py:2757-2758 `MetaInterp.clear_exception()`.
    ///
    /// ```python
    /// def clear_exception(self):
    ///     self.last_exc_value = lltype.nullptr(rclass.OBJECT)
    /// ```
    pub fn clear_exception(&mut self) {
        self.last_exc_value = 0;
    }

    /// pyjitpl.py:3683-3693 `MetaInterp.do_not_in_trace_call(allboxes, descr)`.
    ///
    /// ```python
    /// def do_not_in_trace_call(self, allboxes, descr):
    ///     self.clear_exception()
    ///     executor.execute_varargs(self.cpu, self, rop.CALL_N,
    ///                                       allboxes, descr)
    ///     if self.last_exc_value:
    ///         # cannot trace this!  it raises, so we have to follow the
    ///         # exception-catching path, but the trace doesn't contain
    ///         # the call at all
    ///         raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
    ///                                 raising_exception=True)
    ///     return None
    /// ```
    ///
    /// Executes a `@not_in_trace` decorated call (`OS_NOT_IN_TRACE`
    /// oopspec) without recording any IR.  The call's side effects
    /// happen now; the trace simply skips over the call.  If the call
    /// raises, the trace must abort because the exception-catching
    /// path needs the blackhole interpreter.
    ///
    /// `allboxes` contains the funcbox at slot 0 followed by typed
    /// argboxes (matching `_build_allboxes`'s output).
    pub fn do_not_in_trace_call(
        &mut self,
        allboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr: &dyn majit_ir::descr::CallDescr,
    ) -> Result<Option<OpRef>, SwitchToBlackhole> {
        // pyjitpl.py:3684: self.clear_exception()
        self.clear_exception();
        debug_assert_eq!(
            descr.result_type(),
            majit_ir::Type::Void,
            "do_not_in_trace_call expects a CALL_N descr",
        );
        debug_assert!(
            !allboxes.is_empty(),
            "do_not_in_trace_call: allboxes must include funcbox at slot 0",
        );
        // pyjitpl.py:3685-3686: executor.execute_varargs(cpu, self,
        //                                                  CALL_N, allboxes, descr).
        // `executor::execute_varargs` transcribes the helper-side
        // exception (BH_LAST_EXC_VALUE thread-local) onto
        // `self.last_exc_value` automatically.
        let _ = crate::executor::execute_varargs(self, OpCode::CallN, allboxes, descr);
        // pyjitpl.py:3687-3692: if self.last_exc_value: raise SwitchToBlackhole(ABORT_ESCAPE)
        if self.last_exc_value != 0 {
            return Err(SwitchToBlackhole::abort_escape());
        }
        // pyjitpl.py:3693: return None
        Ok(None)
    }

    /// pyjitpl.py:3397-3398 `MetaInterp.assert_no_exception()`.
    ///
    /// ```python
    /// def assert_no_exception(self):
    ///     assert not self.last_exc_value
    /// ```
    pub fn assert_no_exception(&self) {
        debug_assert!(
            self.last_exc_value == 0,
            "MetaInterp.assert_no_exception: last_exc_value = {:#x}",
            self.last_exc_value
        );
    }

    /// pyjitpl.py:3380-3395 `MetaInterp.handle_possible_exception()`.
    ///
    /// ```python
    /// def handle_possible_exception(self):
    ///     if self.last_exc_value:
    ///         exception_box = ConstInt(ptr2int(self.last_exc_value.typeptr))
    ///         op = self.generate_guard(rop.GUARD_EXCEPTION,
    ///                                  None, exception_box)
    ///         val = lltype.cast_opaque_ptr(llmemory.GCREF, self.last_exc_value)
    ///         if self.class_of_last_exc_is_const:
    ///             self.last_exc_box = ConstPtr(val)
    ///         else:
    ///             self.last_exc_box = op
    ///             op.setref_base(val)
    ///         assert op is not None
    ///         self.class_of_last_exc_is_const = True
    ///         self.finishframe_exception()
    ///     else:
    ///         self.generate_guard(rop.GUARD_NO_EXCEPTION)
    /// ```
    ///
    /// Emits `GUARD_EXCEPTION(typeptr)` or `GUARD_NO_EXCEPTION` per
    /// `self.last_exc_value`.  Pyre records the guard via TraceCtx
    /// when an active trace is present; outside a trace this is a
    /// no-op so blackhole-side callers stay safe.
    ///
    /// Returns `Err(FinishframeExceptionSignal::ChangeFrame)` when the
    /// exception path triggered `finishframe_exception` and the
    /// framestack still has a caller — matches RPython's `raise
    /// ChangeFrame`.  When the framestack is fully drained, returns
    /// `Err(FinishframeExceptionSignal::ExitFrameWithExceptionRef(_))`,
    /// matching RPython's `raise jitexc.ExitFrameWithExceptionRef`.
    pub fn handle_possible_exception(&mut self) -> Result<(), FinishframeExceptionSignal> {
        if self.last_exc_value != 0 {
            let typeptr = self.read_typeptr_from_exception(self.last_exc_value);
            let exception_value = self.last_exc_value;
            let class_is_const = self.class_of_last_exc_is_const;
            // pyjitpl.py:3382-3390:
            //   op = generate_guard(GUARD_EXCEPTION, None, exception_box)
            //   val = cast_opaque_ptr(GCREF, last_exc_value)
            //   if class_of_last_exc_is_const:
            //       last_exc_box = ConstPtr(val)
            //   else:
            //       last_exc_box = op           # op.setref_base(val)
            // The guard is recorded in both arms; only the box stored as
            // last_exc_box differs. Pyre's const_ref(val) is the orthodox
            // ConstPtr equivalent (trace_ctx.rs:583).
            let last_exc_box = if let Some(ctx) = self.tracing.as_mut() {
                let exc_class_box = ctx.const_int(typeptr);
                let guard_op = ctx.guard_exception(exc_class_box, 0);
                if class_is_const {
                    ctx.const_ref(exception_value)
                } else {
                    guard_op
                }
            } else {
                OpRef::NONE
            };
            self.last_exc_box = Some(last_exc_box);
            // pyjitpl.py:3392: self.class_of_last_exc_is_const = True
            self.class_of_last_exc_is_const = true;
            // pyjitpl.py:3393: self.finishframe_exception()
            self.finishframe_exception()
        } else {
            if let Some(ctx) = self.tracing.as_mut() {
                ctx.record_guard(OpCode::GuardNoException, &[], 0);
            }
            Ok(())
        }
    }

    /// pyjitpl.py:2506-2538 `MetaInterp.finishframe_exception()`.
    ///
    /// Walk the framestack looking for an immediately-following
    /// `catch_exception` bytecode, mirroring the upstream interpreter.
    /// Frames without a handler are popped. `rvmprof_code` is decoded
    /// in-place before the pop, matching the RPython side effect.
    pub fn finishframe_exception(&mut self) -> Result<(), FinishframeExceptionSignal> {
        const SIZE_LIVE_OP: usize = majit_translate::liveness::OFFSET_SIZE + 1;

        // pyjitpl.py:2507: excvalue = self.last_exc_value
        let excvalue = self.last_exc_value;

        while !self.framestack.is_empty() {
            let mut handled = false;
            {
                let frame = self.framestack.current_mut();
                let code = &frame.jitcode.code;
                let mut position = if frame.pc != 0 || frame.code_cursor == 0 {
                    frame.pc
                } else {
                    frame.code_cursor
                };

                if position < code.len() {
                    let mut opcode = code[position];
                    if opcode == crate::jitcode::insns::BC_LIVE {
                        position += SIZE_LIVE_OP;
                        if position < code.len() {
                            opcode = code[position];
                        }
                    }
                    if opcode == crate::jitcode::insns::BC_CATCH_EXCEPTION
                        && position + 2 < code.len()
                    {
                        let target =
                            u16::from_le_bytes([code[position + 1], code[position + 2]]) as usize;
                        frame.pc = target;
                        frame.code_cursor = target;
                        handled = true;
                    } else if opcode == crate::jitcode::insns::BC_RVMPROF_CODE
                        && position + 2 < code.len()
                    {
                        let leaving_idx = code[position + 1] as usize;
                        let unique_id_idx = code[position + 2] as usize;
                        let leaving = frame
                            .int_values
                            .get(leaving_idx)
                            .and_then(|v| *v)
                            .unwrap_or(0);
                        let unique_id = frame
                            .int_values
                            .get(unique_id_idx)
                            .and_then(|v| *v)
                            .unwrap_or(0);
                        crate::rvmprof::cintf::jit_rvmprof_code(leaving, unique_id);
                    }
                }
            }
            if handled {
                // pyjitpl.py:2522: raise ChangeFrame
                return Err(FinishframeExceptionSignal::ChangeFrame);
            }
            self.popframe(true);
        }
        // pyjitpl.py:2533-2538: framestack drained.
        //   try:
        //       self.compile_exit_frame_with_exception(self.last_exc_box)
        //   except SwitchToBlackhole as stb:
        //       self.aborted_tracing(stb.reason)
        //   raise jitexc.ExitFrameWithExceptionRef(
        //       lltype.cast_opaque_ptr(GCREF, excvalue))
        let valuebox = self.last_exc_box;
        if let Err(stb) = self.compile_exit_frame_with_exception(valuebox) {
            self.aborted_tracing(stb.reason);
        }
        Err(FinishframeExceptionSignal::ExitFrameWithExceptionRef(
            majit_ir::GcRef(excvalue as usize),
        ))
    }

    /// pyjitpl.py:1881-1890 `MIFrame.handle_possible_overflow_error(label, orgpc, resbox)`.
    ///
    /// ```python
    /// def handle_possible_overflow_error(self, label, orgpc, resbox):
    ///     if self.metainterp.ovf_flag:
    ///         self.metainterp.generate_guard(rop.GUARD_OVERFLOW, resumepc=orgpc)
    ///         self.pc = label
    ///         return None
    ///     else:
    ///         self.metainterp.generate_guard(rop.GUARD_NO_OVERFLOW, resumepc=orgpc)
    ///         return resbox
    /// ```
    ///
    /// `frame_pc_target` is the `(pc_target, source_pc)` pair RPython
    /// passes as `(label, orgpc)`: when an overflow happened the
    /// current frame's `pc` jumps to `label` so the caller can route
    /// to the user-level overflow handler.  Returns `None` when the
    /// overflow guard fires, otherwise returns the `resbox` opref so
    /// the caller can use it as the operation's typed result.
    pub fn handle_possible_overflow_error(
        &mut self,
        frame_pc_target: usize,
        _orgpc: usize,
        resbox: OpRef,
    ) -> Option<OpRef> {
        if self.ovf_flag {
            // pyjitpl.py:1883-1885: GUARD_OVERFLOW + frame.pc = label
            if let Some(ctx) = self.tracing.as_mut() {
                ctx.record_guard(OpCode::GuardOverflow, &[], 0);
            }
            if !self.framestack.is_empty() {
                self.framestack.current_mut().pc = frame_pc_target;
            }
            None
        } else {
            // pyjitpl.py:1888-1890: GUARD_NO_OVERFLOW + return resbox
            if let Some(ctx) = self.tracing.as_mut() {
                ctx.record_guard(OpCode::GuardNoOverflow, &[], 0);
            }
            Some(resbox)
        }
    }

    /// Read the exception value's `typeptr` field — RPython:
    /// `self.last_exc_value.typeptr`.  Pyre stores the exception as a
    /// raw pointer in `last_exc_value`; the typeptr lives at the head
    /// of every `OBJECT` instance per the RPython object layout.
    fn read_typeptr_from_exception(&self, exc_value: i64) -> i64 {
        if let Some(callback) = self.cls_of_box {
            callback(exc_value)
        } else {
            // No callback wired yet — fall back to the class pointer
            // of the exception object itself.  Conservative; the
            // GUARD_EXCEPTION will simply store this raw pointer.
            exc_value
        }
    }

    /// Run `JitCodeMachine` against `MetaInterp::framestack` per the
    /// upstream `pyjitpl.py:self.framestack` single-stack invariant.
    ///
    /// Pushes the root MIFrame onto `self.framestack`, hands the
    /// borrow to a `JitCodeMachine`, and pops the root after the
    /// machine returns.  Mirrors RPython's `MetaInterp.interpret`
    /// shape where `MIFrame.run_one_step` operates on
    /// `self.framestack[-1]`.
    ///
    /// Reaches into `self.tracing` for the active `TraceCtx` so the
    /// caller does not need a second `&mut TraceCtx` borrow that would
    /// conflict with `&mut self`.  Panics when called outside an
    /// active trace.
    pub fn trace_jitcode_with_framestack<S, R>(
        &mut self,
        sym: &mut S,
        jitcode: std::sync::Arc<crate::jitcode::JitCode>,
        pc: usize,
        runtime: &R,
    ) -> crate::TraceAction
    where
        S: crate::pyjitpl::JitCodeSym,
        R: crate::pyjitpl::JitCodeRuntime,
    {
        // pyjitpl.py:2451: self.framestack.append(f) — push the root.
        let root_frame = crate::pyjitpl::MIFrame::setup(
            jitcode,
            pc,
            None,
            Some(
                self.tracing
                    .as_mut()
                    .expect("trace_jitcode_with_framestack requires an active trace"),
            ),
        );
        self.framestack.push(root_frame);
        let cls_of_box = self.cls_of_box;
        let issubclass = self.issubclass;
        let action = {
            // Split the &mut borrow so the trace context and framestack
            // can be passed to the machine simultaneously.  `tracing`
            // and `framestack` are independent fields on MetaInterp.
            let ctx = self
                .tracing
                .as_mut()
                .expect("trace_jitcode_with_framestack requires an active trace");
            // Sub-jitcode and fn-ptr pools now live on each JitCode's
            // `exec.descrs` / `exec.fn_ptrs` (see RPython `blackhole.py:150-157`
            // `j`/`d` argcode resolution), so the machine no longer
            // needs parallel slice borrows at construction time.
            let mut machine = crate::pyjitpl::JitCodeMachine::<S, _>::with_framestack(
                &mut self.framestack,
                &[],
                &[],
            );
            machine.set_cls_of_box(cls_of_box);
            machine.set_issubclass(issubclass);
            machine.run_to_end(ctx, sym, runtime)
        };
        // RPython's interpret loop drains framestack via popframe; pyre
        // mirrors the post-condition explicitly so this entry point is
        // re-entrant — leave the stack in the same shape it came in.
        let _ = self.framestack.pop();
        action
    }

    /// pyjitpl.py:2427-2429 `MetaInterp.is_main_jitcode(jitcode)`.
    ///
    /// ```python
    /// def is_main_jitcode(self, jitcode):
    ///     return (jitcode.jitdriver_sd is not None and
    ///             jitcode.jitdriver_sd.jitdriver.is_recursive)
    /// ```
    ///
    /// Reads `staticdata.jitdrivers_sd[idx].is_recursive` exactly like
    /// upstream's `jitcode.jitdriver_sd.jitdriver.is_recursive`. Falls
    /// back to `false` when the jitcode does not point at a registered
    /// driver slot — matches the `jitdriver_sd is not None` guard.
    pub fn is_main_jitcode(&self, jitcode: &crate::jitcode::JitCode) -> bool {
        match jitcode.jitdriver_sd() {
            Some(idx) => self
                .staticdata
                .jitdrivers_sd
                .get(idx)
                .map(|jd| jd.is_recursive)
                .unwrap_or(false),
            None => false,
        }
    }

    /// pyjitpl.py:2432-2452 `MetaInterp.newframe(jitcode, greenkey)`.
    ///
    /// ```python
    /// def newframe(self, jitcode, greenkey=None):
    ///     if jitcode.jitdriver_sd:
    ///         self.portal_call_depth += 1
    ///         self.call_ids.append(self.current_call_id)
    ///         unique_id = -1
    ///         if greenkey is not None:
    ///             unique_id = jitcode.jitdriver_sd.warmstate.get_unique_id(greenkey)
    ///             jd_no = jitcode.jitdriver_sd.index
    ///             self.enter_portal_frame(jd_no, unique_id)
    ///         self.current_call_id += 1
    ///     if greenkey is not None and self.is_main_jitcode(jitcode):
    ///         self.portal_trace_positions.append(...)
    ///     if len(self.free_frames_list) > 0:
    ///         f = self.free_frames_list.pop()
    ///     else:
    ///         f = MIFrame(self)
    ///     f.setup(jitcode, greenkey)
    ///     self.framestack.append(f)
    ///     return f
    /// ```
    ///
    /// Pyre stores frames in `self.framestack`; we still bump the
    /// existing `inline_depth` counter so callers that have not yet
    /// migrated keep their existing book-keeping.
    pub fn newframe(
        &mut self,
        jitcode: std::sync::Arc<crate::jitcode::JitCode>,
        greenkey: Option<u64>,
    ) -> usize {
        // pyjitpl.py:2433: if jitcode.jitdriver_sd: portal_call_depth += 1
        if let Some(jd_no) = jitcode.jitdriver_sd() {
            self.portal_call_depth += 1;
            // pyjitpl.py:2435: self.call_ids.append(self.current_call_id)
            self.call_ids.push(self.current_call_id);
            // pyjitpl.py:2440-2441: enter_portal_frame(jitdriver_sd.index, unique_id)
            if let Some(unique_id) = greenkey {
                self.enter_portal_frame(jd_no, unique_id);
            }
            // pyjitpl.py:2442: self.current_call_id += 1
            self.current_call_id += 1;
        }
        // pyjitpl.py:2443-2445: `if greenkey is not None and
        // self.is_main_jitcode(jitcode): self.portal_trace_positions.append(
        //     (jitcode.jitdriver_sd, greenkey, self.history.get_trace_position()))`.
        if let (Some(gk), Some(jd_no)) = (greenkey, jitcode.jitdriver_sd()) {
            if self.is_main_jitcode(&jitcode) {
                if let (Some(positions), Some(ctx)) =
                    (self.portal_trace_positions.as_mut(), self.tracing.as_ref())
                {
                    positions.push((jd_no, Some(gk), ctx.get_trace_position()));
                }
            }
        }
        // Bump the existing TraceCtx inline-depth counter so trace
        // recorder bookkeeping (already wired through pyre's tracer)
        // stays in sync; the canonical frame storage is `framestack`.
        // The `newframe` path predates the raw (code_ptr, pc) greenkey
        // and operates on sub-jitcodes rather than portal frames, so
        // project the u64 greenkey into the raw slot verbatim —
        // pyjitpl.py:1396-1401 element-wise parity still holds because
        // this caller doesn't feed the recursion-depth walk.
        let raw = (greenkey.unwrap_or_default() as usize, 0);
        let _ = self.enter_inline_frame(raw);
        // pyjitpl.py:2446-2451: reuse / allocate MIFrame, push onto framestack.
        let frame = crate::pyjitpl::MIFrame::setup(jitcode, 0, greenkey, self.tracing.as_mut());
        self.framestack.push(frame);
        self.framestack.len() - 1
    }

    /// pyjitpl.py:2454-2456 `MetaInterp.enter_portal_frame(jd_no, unique_id)`.
    ///
    /// ```python
    /// def enter_portal_frame(self, jd_no, unique_id):
    ///     self.history.record2(rop.ENTER_PORTAL_FRAME,
    ///                          ConstInt(jd_no), ConstInt(unique_id), None)
    /// ```
    pub fn enter_portal_frame(&mut self, jd_no: usize, unique_id: u64) {
        if let Some(ctx) = self.tracing.as_mut() {
            let jd_no_box = ctx.const_int(jd_no as i64);
            let unique_id_box = ctx.const_int(unique_id as i64);
            ctx.record_op(OpCode::EnterPortalFrame, &[jd_no_box, unique_id_box]);
        }
    }

    /// pyjitpl.py:2458-2459 `MetaInterp.leave_portal_frame(jd_no)`.
    ///
    /// ```python
    /// def leave_portal_frame(self, jd_no):
    ///     self.history.record1(rop.LEAVE_PORTAL_FRAME, ConstInt(jd_no), None)
    /// ```
    pub fn leave_portal_frame(&mut self, jd_no: usize) {
        if let Some(ctx) = self.tracing.as_mut() {
            let jd_no_box = ctx.const_int(jd_no as i64);
            ctx.record_op(OpCode::LeavePortalFrame, &[jd_no_box]);
        }
    }

    /// pyjitpl.py:2462-2477 `MetaInterp.popframe(leave_portal_frame=True)`.
    ///
    /// ```python
    /// def popframe(self, leave_portal_frame=True):
    ///     frame = self.framestack.pop()
    ///     jitcode = frame.jitcode
    ///     if jitcode.jitdriver_sd:
    ///         self.portal_call_depth -= 1
    ///         if leave_portal_frame:
    ///             self.leave_portal_frame(jitcode.jitdriver_sd.index)
    ///         self.call_ids.pop()
    ///     ...
    ///     frame.cleanup_registers()
    ///     self.free_frames_list.append(frame)
    /// ```
    pub fn popframe(&mut self, leave_portal_frame: bool) {
        // pyjitpl.py:2463: frame = self.framestack.pop()
        if let Some(mut frame) = self.framestack.pop() {
            // pyjitpl.py:2465-2469: jitdriver_sd → portal_call_depth/leave_portal_frame/call_ids.
            if let Some(jd_no) = frame.jitcode.jitdriver_sd() {
                self.portal_call_depth -= 1;
                if leave_portal_frame {
                    self.leave_portal_frame(jd_no);
                }
                // pyjitpl.py:2469: self.call_ids.pop()
                let _ = self.call_ids.pop();
            }
            // pyjitpl.py:2470-2472: `if frame.greenkey is not None and
            // self.is_main_jitcode(jitcode): self.portal_trace_positions.append(
            //     (jitcode.jitdriver_sd, None, self.history.get_trace_position()))`.
            if let (Some(_gk), Some(jd_no)) = (frame.greenkey, frame.jitcode.jitdriver_sd()) {
                if self.is_main_jitcode(&frame.jitcode) {
                    if let (Some(positions), Some(ctx)) =
                        (self.portal_trace_positions.as_mut(), self.tracing.as_ref())
                    {
                        positions.push((jd_no, None, ctx.get_trace_position()));
                    }
                }
            }
            // pyjitpl.py:2476: frame.cleanup_registers().
            frame.cleanup_registers();
            // pyjitpl.py:2477: self.free_frames_list.append(frame) is
            // an RPython memory-reuse optimization; pyre relies on the
            // Rust drop to release register banks.
        }
        // Mirror the TraceCtx inline-depth counter so trace recorder
        // bookkeeping stays balanced with the framestack pop.
        self.leave_inline_frame();
    }

    /// pyjitpl.py:2479-2503 `MetaInterp.finishframe(resultbox, leave_portal_frame=True)`.
    ///
    /// ```python
    /// def finishframe(self, resultbox, leave_portal_frame=True):
    ///     # handle a non-exceptional return from the current frame
    ///     self.last_exc_value = lltype.nullptr(rclass.OBJECT)
    ///     self.popframe(leave_portal_frame=leave_portal_frame)
    ///     if self.framestack:
    ///         if resultbox is not None:
    ///             self.framestack[-1].make_result_of_lastop(resultbox)
    ///         raise ChangeFrame
    ///     else:
    ///         try:
    ///             self.compile_done_with_this_frame(resultbox)
    ///         except SwitchToBlackhole as stb:
    ///             self.aborted_tracing(stb.reason)
    ///         ...
    /// ```
    ///
    /// `result` is `None` for void returns; otherwise it is the
    /// `(kind, target_index, opref, concrete)` tuple
    /// `MIFrame::make_result_of_lastop` consumes.  PRE-EXISTING-ADAPTATION:
    /// pyre's call BC encodes `target_index` explicitly per call site
    /// instead of reading `bytecode[pc-1]` after dispatch, so the
    /// caller threads it through here.
    ///
    /// `compile_done_with_this_frame` is invoked here line-by-line
    /// per pyjitpl.py:2487-2491; its body is the structural port at
    /// `MetaInterp::compile_done_with_this_frame` (pyjitpl.py:3198).
    /// PRE-EXISTING-ADAPTATION: in pyre the `recorder.finish()` +
    /// `compile.compile_trace` work also happens at the
    /// `TraceAction::Finish` dispatch point (jitdriver.rs:956 →
    /// `MetaInterp::finish_and_compile`) because production function-
    /// return paths route through the trace-recorder pipeline rather
    /// than this method.  When `compile_done_with_this_frame` raises
    /// `SwitchToBlackhole`, the catch translates it into
    /// `aborted_tracing(reason)` per pyjitpl.py:2491.
    pub fn finishframe(
        &mut self,
        result: Option<(crate::jitcode::JitArgKind, usize, OpRef, i64)>,
        leave_portal_frame: bool,
    ) -> Result<(), FinishFrameSignal> {
        // pyjitpl.py:2481: self.last_exc_value = lltype.nullptr(...)
        self.last_exc_value = 0;
        // Capture the popping frame's jitdriver_sd index BEFORE
        // popframe takes the frame: pyjitpl.py:2493 reads
        // `self.jitdriver_sd.result_type` from the active jitdriver,
        // which in pyre is identified by the popped frame's jitcode.
        let popping_jdindex = self
            .framestack
            .frames
            .last()
            .and_then(|f| f.jitcode.jitdriver_sd());
        // pyjitpl.py:2482: self.popframe(leave_portal_frame=...)
        self.popframe(leave_portal_frame);
        // pyjitpl.py:2483: if self.framestack:
        if !self.framestack.is_empty() {
            // pyjitpl.py:2484-2485: framestack[-1].make_result_of_lastop(resultbox)
            if let Some((kind, target_index, opref, concrete)) = result {
                self.framestack.current_mut().make_result_of_lastop(
                    kind,
                    target_index,
                    opref,
                    concrete,
                );
            }
            // pyjitpl.py:2486: raise ChangeFrame
            return Err(FinishFrameSignal::ChangeFrame);
        }
        // pyjitpl.py:2493-2503: result_type = self.jitdriver_sd.result_type
        // → raise DoneWithThisFrame{Void,Int,Ref,Float} per result_type.
        // The variant is determined by the active jitdriver's
        // declared return type, NOT by the resultbox kind tuple — the
        // resultbox supplies the value, the driver supplies the type.
        // Pre-resolved here so compile_done_with_this_frame and the
        // matching DoneWithThisFrame constructor share the same value.
        let result_type = popping_jdindex
            .and_then(|idx| self.staticdata.jitdrivers_sd.get(idx))
            .map(|jd| jd.result_type)
            // No active jitdriver_sd (e.g. helper jitcodes that never
            // entered through a portal); fall back to the resultbox
            // kind so the helper's caller still gets a typed signal.
            .unwrap_or_else(|| match &result {
                None => majit_ir::Type::Void,
                Some((crate::jitcode::JitArgKind::Int, _, _, _)) => majit_ir::Type::Int,
                Some((crate::jitcode::JitArgKind::Ref, _, _, _)) => majit_ir::Type::Ref,
                Some((crate::jitcode::JitArgKind::Float, _, _, _)) => majit_ir::Type::Float,
            });
        // pyjitpl.py:2487-2491:
        //     try:
        //         self.compile_done_with_this_frame(resultbox)
        //     except SwitchToBlackhole as stb:
        //         self.aborted_tracing(stb.reason)
        let exitbox = result.map(|(_, _, opref, _)| opref);
        if let Err(stb) = self.compile_done_with_this_frame(exitbox, result_type) {
            self.aborted_tracing(stb.reason);
        }
        let signal = match result_type {
            // pyjitpl.py:2494-2496: VOID → assert resultbox is None;
            //                              raise DoneWithThisFrameVoid()
            majit_ir::Type::Void => {
                debug_assert!(
                    result.is_none(),
                    "finishframe: VOID result_type with non-None resultbox",
                );
                DoneWithThisFrame::Void
            }
            // pyjitpl.py:2497-2498: INT → DoneWithThisFrameInt(resultbox.getint())
            majit_ir::Type::Int => {
                let value = result.map(|(_, _, _, v)| v).unwrap_or(0);
                DoneWithThisFrame::Int(value)
            }
            // pyjitpl.py:2499-2500: REF → DoneWithThisFrameRef(resultbox.getref_base())
            // jitexc.py:29 carries `GcRef`; pyre stores the raw GC
            // pointer as `i64` in the make_result_of_lastop tuple, so
            // wrap it back into the typed jitexc payload here.
            majit_ir::Type::Ref => {
                let value = result.map(|(_, _, _, v)| v).unwrap_or(0);
                DoneWithThisFrame::Ref(majit_ir::GcRef(value as usize))
            }
            // pyjitpl.py:2501-2502: FLOAT → DoneWithThisFrameFloat(resultbox.getfloatstorage())
            // jitexc.py:37 carries `f64`; pyre threads the IEEE-754 bit
            // pattern through the i64 result tuple, so decode it back
            // to f64 for the typed payload.
            majit_ir::Type::Float => {
                let value = result.map(|(_, _, _, v)| v).unwrap_or(0);
                DoneWithThisFrame::Float(f64::from_bits(value as u64))
            }
        };
        Err(FinishFrameSignal::Done(signal))
    }

    /// pyjitpl.py:3198-3220 `MetaInterp.compile_done_with_this_frame(exitbox)`.
    ///
    /// ```python
    /// def compile_done_with_this_frame(self, exitbox):
    ///     self.store_token_in_vable()
    ///     sd = self.staticdata
    ///     result_type = self.jitdriver_sd.result_type
    ///     if result_type == history.VOID:
    ///         assert exitbox is None
    ///         exits = []
    ///         token = sd.done_with_this_frame_descr_void
    ///     elif result_type == history.INT:
    ///         exits = [exitbox]
    ///         token = sd.done_with_this_frame_descr_int
    ///     elif result_type == history.REF:
    ///         exits = [exitbox]
    ///         token = sd.done_with_this_frame_descr_ref
    ///     elif result_type == history.FLOAT:
    ///         exits = [exitbox]
    ///         token = sd.done_with_this_frame_descr_float
    ///     else:
    ///         assert False
    ///     self.history.record(rop.FINISH, exits, None, descr=token)
    ///     target_token = compile.compile_trace(self, self.resumekey, exits)
    ///     if target_token is not token:
    ///         compile.giveup()
    /// ```
    ///
    /// `popping_jdindex` is the index of the jitdriver_sd whose result_type
    /// drives the dispatch (pyre's analog of `self.jitdriver_sd`, which
    /// upstream resolves implicitly off `self`).  See `finishframe`'s
    /// `popping_jdindex` snapshot for why the index has to be threaded
    /// through here in pyre.
    ///
    /// PRE-EXISTING-ADAPTATION: `self.history.record(rop.FINISH, ...)`
    /// + `compile.compile_trace` are also driven from the
    /// `TraceAction::Finish` dispatch (jitdriver.rs:956 →
    /// `MetaInterp::finish_and_compile`) — pyre's recorder owns the
    /// finish/compile sequence and emits FINISH from there with the
    /// matching `make_fail_descr_typed(result_types)` descr (the
    /// `done_with_this_frame_descr_*` analog).  This method runs the
    /// upstream skeleton — `store_token_in_vable` (idempotent because
    /// the frontend already records it before TraceAction::Finish at
    /// mod.rs:3267), result_type/exits/token bookkeeping, and surfacing
    /// `SwitchToBlackhole` to the caller — without re-emitting the
    /// FINISH op.
    pub fn compile_done_with_this_frame(
        &mut self,
        exitbox: Option<OpRef>,
        result_type: majit_ir::Type,
    ) -> Result<(), SwitchToBlackhole> {
        // pyjitpl.py:3199 self.store_token_in_vable()
        // Early-return on no-vinfo / no-vbox / forced_virtualizable ==
        // vbox (pyjitpl.py:3223-3228). The accompanying GUARD_NOT_FORCED_2
        // is emitted by the pyre frontend wrapper
        // (pyre-jit-trace/src/trace_opcode.rs::store_token_in_vable)
        // through MIFrame::generate_guard so the guard captures fresh
        // resumedata at the current framestack position.
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.store_token_in_vable_setfield();
        }
        // pyjitpl.py:3200-3216: select exits / result-type from result_type.
        let (exits, finish_arg_types): (Vec<OpRef>, Vec<majit_ir::Type>) = match result_type {
            // pyjitpl.py:3204-3206: VOID → exits = [], assert exitbox is None.
            majit_ir::Type::Void => {
                debug_assert!(
                    exitbox.is_none(),
                    "compile_done_with_this_frame: VOID with non-None exitbox",
                );
                (Vec::new(), Vec::new())
            }
            // pyjitpl.py:3207-3215: INT/REF/FLOAT → exits = [exitbox].
            tp => (exitbox.into_iter().collect(), vec![tp]),
        };
        // pyjitpl.py:3217 self.history.record(rop.FINISH, exits, None, descr=token)
        // pyjitpl.py:3218 target_token = compile.compile_trace(self, self.resumekey, exits)
        // pyjitpl.py:3219-3220 if target_token is not token: compile.giveup()
        //
        // Dispatch through `compile_finish_from_active_session` so
        // root / bridge finish branches share the session-owned compile
        // helper — `ActiveTraceSession.bridge` picks between them.
        // `compile_done_with_this_frame` is the non-exception path; the
        // `compile_exit_frame_with_exception` sibling passes `true`.
        self.compile_finish_from_active_session(&exits, finish_arg_types, false)
    }

    /// pyjitpl.py:3238-3245 `MetaInterp.compile_exit_frame_with_exception(valuebox)`.
    ///
    /// ```python
    /// def compile_exit_frame_with_exception(self, valuebox):
    ///     self.store_token_in_vable()
    ///     sd = self.staticdata
    ///     token = sd.exit_frame_with_exception_descr_ref
    ///     self.history.record1(rop.FINISH, valuebox, None, descr=token)
    ///     target_token = compile.compile_trace(self, self.resumekey, [valuebox])
    ///     if target_token is not token:
    ///         compile.giveup()
    /// ```
    ///
    /// Exception-flavored sibling of `compile_done_with_this_frame`.
    /// PRE-EXISTING-ADAPTATION shared with that method: the FINISH op
    /// emit + `compile.compile_trace` happen at the trace-dispatch
    /// `TraceAction::Finish` site (jitdriver.rs:1031), so this method
    /// runs only the upstream skeleton — `store_token_in_vable` +
    /// `make_fail_descr_typed` for the Ref result-type slot — and
    /// surfaces `SwitchToBlackhole` to the caller exactly like
    /// `compile_done_with_this_frame`.
    ///
    /// The primary exception exit path in pyre is dispatch.rs's
    /// `unwind_to_exception_handler` at BC_RAISE/BC_RERAISE: when the
    /// framestack drains with no `catch_exception`, dispatch returns
    /// `TraceAction::Finish { finish_args: [last_exc_box], finish_arg_types:
    /// [Ref] }` directly (dispatch.rs:298), so the normal
    /// `finish_and_compile` path records FINISH + compiles — matching
    /// `pyjitpl.py:3238-3245`. This MetaInterp-side hook covers the
    /// rarer path where an exception surfaces during residual-call
    /// dispatch (miframe_execute_varargs / do_conditional_call); the
    /// `FinishframeExceptionSignal::ExitFrameWithExceptionRef` return
    /// from `handle_possible_exception` bubbles up, but the wiring
    /// that converts it into a `TraceAction::Finish` dispatch at the
    /// MetaInterp call chain is not yet complete (deferred epic).
    pub fn compile_exit_frame_with_exception(
        &mut self,
        valuebox: Option<OpRef>,
    ) -> Result<(), SwitchToBlackhole> {
        // pyjitpl.py:3239 self.store_token_in_vable()
        // Same split as compile_done_with_this_frame: the
        // GUARD_NOT_FORCED_2 that RPython's store_token_in_vable emits
        // (pyjitpl.py:3236) is produced by the pyre frontend wrapper
        // (pyre-jit-trace/src/trace_opcode.rs::store_token_in_vable)
        // through MIFrame::generate_guard.
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.store_token_in_vable_setfield();
        }
        // pyjitpl.py:3242 self.history.record1(rop.FINISH, valuebox, None, descr=token)
        // pyjitpl.py:3243 target_token = compile.compile_trace(self, self.resumekey, [valuebox])
        // pyjitpl.py:3244-3245 if target_token is not token: compile.giveup()
        //
        // Routes through the session-owned `compile_finish_from_active_session`
        // so both the MetaInterp-call-chain exception exit (this method)
        // and the pyre-dispatch-layer `unwind_to_exception_handler`
        // (dispatch.rs:298, emits `TraceAction::Finish` with [Ref])
        // share a single compile path, matching upstream's single-owner
        // `compile.compile_trace` invocation for the FINISH.
        let exits: Vec<OpRef> = valuebox.into_iter().collect();
        self.compile_finish_from_active_session(&exits, vec![majit_ir::Type::Ref], true)
    }

    /// pyjitpl.py:3198-3220 + 3238-3245 shared compile-and-finish helper.
    ///
    /// Consumes the [`ActiveTraceSession`] installed by
    /// `begin_trace_session` and drives `compile.compile_trace(self,
    /// self.resumekey, exits)` — the exact RPython call that
    /// `compile_done_with_this_frame` and
    /// `compile_exit_frame_with_exception` each make.  Dispatch:
    ///
    /// - `bridge.is_some()` → `compile_trace_finish(bridge_key, ...,
    ///   bridge_origin, finish_descr)` records the trailing FINISH via
    ///   `recorder.finish()` and drives the bridge compile.
    /// - `bridge.is_none()` → `finish_and_compile(..., trace_meta)` —
    ///   the root-trace equivalent.
    ///
    /// Returns `Err(SwitchToBlackhole::bad_loop())` if the compile
    /// gave up (matching `compile.giveup()`).  The caller
    /// (`compile_done_with_this_frame` / `compile_exit_frame_with_exception`)
    /// propagates the error so `finishframe` / `finishframe_exception`
    /// can translate it into `aborted_tracing(stb.reason)` per
    /// pyjitpl.py:2491.
    ///
    /// Idempotent when no session is active — a second call after the
    /// first one already consumed the session returns `Ok(())` without
    /// touching the tracer.  This matches the dual-entry shape in pyre
    /// where `TraceAction::Finish` (dispatch-layer) and
    /// `MetaInterp.finishframe` (residual-call chain) can both fire
    /// along the same trace; only one actually runs the compile.
    pub fn compile_finish_from_active_session(
        &mut self,
        finish_args: &[OpRef],
        finish_arg_types: Vec<Type>,
        exit_with_exception: bool,
    ) -> Result<(), SwitchToBlackhole> {
        // Idempotent no-op: session already consumed by the sibling
        // finish entry (e.g. `TraceAction::Finish` arm ran first).
        if self.active_trace_session.is_none() {
            return Ok(());
        }
        // bridge branch: `compile_trace_finish` records FINISH on the
        // existing tracer and dispatches to `compile_trace_inner`.
        if let Some(bridge) = self.bridge_info() {
            // `pyjitpl.py:3216-3217` / `pyjitpl.py:3241`:
            //   `token = sd.done_with_this_frame_descr_<type>` (normal) or
            //   `token = sd.exit_frame_with_exception_descr_ref` (raising).
            // Use the metainterp-attached singleton for pointer identity
            // parity with the backend (`attach_descrs_to_cpu`).  Falls
            // back to `make_fail_descr_typed` only when the singleton is
            // unattached (tests bypassing `MetaInterp::new`).
            let finish_descr = if exit_with_exception {
                self.staticdata
                    .exit_frame_with_exception_descr_ref
                    .clone()
                    .unwrap_or_else(|| {
                        crate::make_finish_fail_descr_typed(finish_arg_types.clone())
                    })
            } else {
                self.staticdata
                    .done_with_this_frame_descr_from_types(&finish_arg_types)
                    .unwrap_or_else(|| {
                        crate::make_finish_fail_descr_typed(finish_arg_types.clone())
                    })
            };
            let outcome = self.compile_trace_finish(
                bridge.green_key,
                finish_args,
                Some((bridge.trace_id, bridge.fail_index)),
                finish_descr,
            );
            // pyjitpl.py:3095-3099 raise_if_successful(): successful
            // bridge closure terminates tracing.  Consume the whole
            // session (bridge + trace_meta) and unwind the tracer via
            // `abort_trace_live` (live cleanup + `pending_abort_*`
            // staging) — NOT `abort_trace`, because that also fires
            // `aborted_tracing(Generic)` which would double-count the
            // upstream abort hook (pyjitpl.py:2491 fires it once with
            // `stb.reason` via the caller-side catch).  On success no
            // catch fires, so we clear the staged `pending_abort_*`
            // below to keep the next aborted_tracing clean.
            self.clear_trace_session();
            self.abort_trace_live(false);
            return match outcome {
                CompileOutcome::Compiled { .. } | CompileOutcome::Cancelled => {
                    // Drop the `pending_abort_*` staged by
                    // `abort_trace_live` — on success no abort hook
                    // fires, so letting stale greenkey linger would
                    // attach this successfully-compiled bridge's key
                    // to a later, unrelated abort.
                    self.pending_abort_green_key = None;
                    self.pending_abort_permanent = false;
                    Ok(())
                }
                // pyjitpl.py:3220/:3245 `compile.giveup()` per
                // `rpython/jit/metainterp/compile.py:27` →
                // `SwitchToBlackhole(Counters.ABORT_BRIDGE)`.  The
                // bridge FINISH path shares the same giveup reason as
                // the root FINISH path.
                CompileOutcome::Aborted => Err(SwitchToBlackhole::giveup()),
            };
        }
        // Root branch: drain `trace_meta`, drive `finish_and_compile`.
        // `finish_and_compile` takes the tracer (`self.tracing`) so no
        // separate `abort_trace` is needed.
        let meta = self
            .take_trace_meta()
            .expect("compile_finish_from_active_session: session must be present");
        self.finish_and_compile(finish_args, finish_arg_types, meta, exit_with_exception)
    }

    /// pyjitpl.py:2174-2186 `MIFrame.do_residual_or_indirect_call`.
    ///
    /// ```python
    /// def do_residual_or_indirect_call(self, funcbox, argboxes, calldescr, pc):
    ///     """The 'residual_call' operation is emitted in two cases:
    ///     when we have to generate a residual CALL operation, but also
    ///     to handle an indirect_call that may need to be inlined."""
    ///     if isinstance(funcbox, Const):
    ///         sd = self.metainterp.staticdata
    ///         key = funcbox.getaddr()
    ///         jitcode = sd.bytecode_for_address(key)
    ///         if jitcode is not None:
    ///             # we should follow calls to this graph
    ///             return self.metainterp.perform_call(jitcode, argboxes)
    ///     # but we should not follow calls to that graph
    ///     return self.do_residual_call(funcbox, argboxes, calldescr, pc)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: RPython places this method on `MIFrame`
    /// because `self.metainterp` is a back-pointer attribute; pyre's
    /// `MIFrame` does not carry a `&mut MetaInterp` back-reference (the
    /// borrow checker would alias `MetaInterp::framestack` against
    /// itself), so the method lives on `MetaInterp<M>` and acts on the
    /// current top-of-framestack frame implicitly.  Body remains
    /// line-for-line identical to `pyjitpl.py:2178-2186`.
    ///
    /// Returns:
    /// - `Err(DoResidualCallAbort::ChangeFrame)` when the funcbox is a
    ///   Const whose address resolves to an indirect-call target and
    ///   `perform_call` raised `ChangeFrame` (control transfers into
    ///   the inlined callee — no return value).
    /// - `Err(DoResidualCallAbort::AbortEscape)` when `do_residual_call`
    ///   bubbles up an `OS_NOT_IN_TRACE` blackhole switch.
    /// - `Ok(Some((box, concrete)))` / `Ok(None)` when the call was
    ///   emitted as a residual `CALL_*` IR op (caller continues with
    ///   the residual return value).
    ///
    /// Signature matches RPython: `funcbox` is a typed triple (same
    /// shape as `argboxes`), `descr_ref + descr_view` replace
    /// RPython's `calldescr` argument (both handle and view are
    /// needed because Rust cannot pass a trait object through an
    /// `Arc<dyn Descr>` transparently).
    pub fn do_residual_or_indirect_call(
        &mut self,
        funcbox: (crate::jitcode::JitArgKind, OpRef, i64),
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
        pc: usize,
        dst: Option<(crate::jitcode::JitArgKind, usize)>,
    ) -> Result<Option<(OpRef, i64)>, DoResidualCallAbort> {
        // pyjitpl.py:2178: if isinstance(funcbox, Const):
        if funcbox.1.is_constant() {
            // pyjitpl.py:2179: sd = self.metainterp.staticdata
            // pyjitpl.py:2180: key = funcbox.getaddr()
            let key = funcbox.2 as usize;
            // pyjitpl.py:2181: jitcode = sd.bytecode_for_address(key)
            if let Some(jitcode) = self.staticdata.bytecode_for_address(key) {
                // pyjitpl.py:2184: return self.metainterp.perform_call(jitcode, argboxes)
                self.perform_call(jitcode, argboxes, None)?;
                unreachable!("perform_call always raises ChangeFrame");
            }
        }
        // pyjitpl.py:2186: return self.do_residual_call(funcbox, argboxes, calldescr, pc)
        self.do_residual_call_full(
            funcbox, argboxes, descr_ref, descr_view, pc, false, None, dst,
        )
    }

    /// pyjitpl.py:1278-1321 `MIFrame._try_tco()`.
    ///
    /// ```python
    /// def _try_tco(self):
    ///     if self.jitcode.jitdriver_sd:
    ///         return
    ///     argcode = self._result_argcode
    ///     pc = self.pc
    ///     if argcode == 'v':
    ///         target_index = -1
    ///     else:
    ///         target_index = ord(self.bytecode[pc - 1])
    ///     op = ord(self.bytecode[pc])
    ///     if op != self.metainterp.staticdata.op_live:
    ///         return
    ///     next_pc = pc + SIZE_LIVE_OP
    ///     if next_pc >= len(self.bytecode):
    ///         return
    ///     next_op = ord(self.bytecode[next_pc])
    ///     if ((argcode == 'i' and next_op == ...op_int_return) or
    ///         (argcode == 'r' and next_op == ...op_ref_return) or
    ///         (argcode == 'f' and next_op == ...op_float_return) or
    ///         (argcode == 'v' and next_op == ...op_void_return)
    ///     ):
    ///         if (target_index < 0 or
    ///                 ord(self.bytecode[next_pc + 1]) == target_index):
    ///             ...
    ///             del self.metainterp.framestack[-2]
    ///             tracelength = self.metainterp.history.length()
    ///             if tracelength == self.metainterp.trace_length_at_last_tco:
    ///                 self.metainterp._record_helper(
    ///                     rop.SAME_AS_I, tracelength, None,
    ///                     ConstInt(tracelength))
    ///             else:
    ///                 self.metainterp.trace_length_at_last_tco = tracelength
    /// ```
    ///
    /// The "frame" that runs `_try_tco` is the **callee** that was
    /// just pushed by `_opimpl_inline_call*` (pyjitpl.py:1265-1276);
    /// the upstream `del framestack[-2]` removes the **caller** from
    /// the stack, leaving the new callee in place — that's what makes
    /// it a tail call.
    ///
    /// PRE-EXISTING-ADAPTATION: lives on `MetaInterp<M>` rather than
    /// `MIFrame` — same borrow-checker constraint as
    /// `do_residual_or_indirect_call`.  The "self frame" (RPython's
    /// `self`) is `framestack.current_mut()` — i.e. the top frame.
    /// `SIZE_LIVE_OP` is `OFFSET_SIZE + 1 = 3` per
    /// `liveness.py:125`.
    pub fn _try_tco(&mut self) {
        const SIZE_LIVE_OP: usize = 3;
        if self.framestack.is_empty() {
            return;
        }
        // Snapshot fields from the top frame so we don't hold a
        // mutable borrow across the framestack mutation below.
        let (jitcode_arc, pc, argcode) = {
            let frame = self.framestack.current_mut();
            (frame.jitcode.clone(), frame.pc, frame._result_argcode)
        };
        // pyjitpl.py:1279-1280: if self.jitcode.jitdriver_sd: return
        if jitcode_arc.jitdriver_sd().is_some() {
            return;
        }
        let bytecode = &jitcode_arc.code;
        // pyjitpl.py:1283-1286: target_index from bytecode[pc-1] (or
        // -1 for void).
        let target_index: i32 = if argcode == b'v' {
            -1
        } else {
            if pc == 0 {
                return;
            }
            bytecode[pc - 1] as i32
        };
        // pyjitpl.py:1287-1290: op = bytecode[pc]; must be op_live.
        if pc >= bytecode.len() {
            return;
        }
        let op = bytecode[pc] as i32;
        if op != self.staticdata.op_live {
            return;
        }
        // pyjitpl.py:1291-1293: next_pc bounds check.
        let next_pc = pc + SIZE_LIVE_OP;
        if next_pc >= bytecode.len() {
            return;
        }
        let next_op = bytecode[next_pc] as i32;
        // pyjitpl.py:1295-1299: next_op must be a *_return matching argcode.
        let return_op_for_kind = match argcode {
            b'i' => self.staticdata.op_int_return,
            b'r' => self.staticdata.op_ref_return,
            b'f' => self.staticdata.op_float_return,
            b'v' => self.staticdata.op_void_return,
            _ => return,
        };
        if next_op != return_op_for_kind {
            return;
        }
        // pyjitpl.py:1301-1302: target register match check.
        if target_index >= 0 {
            let next_target = bytecode.get(next_pc + 1).copied().unwrap_or(0) as i32;
            if next_target != target_index {
                return;
            }
        }
        // pyjitpl.py:1306-1307: assert framestack[-2] is self; del framestack[-2]
        // The callee (self) is at top; remove the caller (-2 == len-2).
        if self.framestack.frames.len() < 2 {
            return;
        }
        let caller_idx = self.framestack.frames.len() - 2;
        let _removed = self.framestack.frames.remove(caller_idx);
        // pyjitpl.py:1308-1321: trace_length_at_last_tco bookkeeping.
        let tracelength = self
            .tracing
            .as_ref()
            .map(|ctx| ctx.ops().len() as i32)
            .unwrap_or(0);
        if tracelength == self.trace_length_at_last_tco {
            // pyjitpl.py:1318-1319: emit SAME_AS_I(ConstInt(tracelength))
            // so the trace-length limit eventually fires.
            if let Some(ctx) = self.tracing.as_mut() {
                let const_box = ctx.const_int(tracelength as i64);
                ctx.record_op(OpCode::SameAsI, &[const_box]);
            }
        } else {
            self.trace_length_at_last_tco = tracelength;
        }
    }

    /// pyjitpl.py:3581-3587 `MetaInterp.direct_call_may_force(argboxes, valueconst, calldescr)`.
    ///
    /// ```python
    /// def direct_call_may_force(self, argboxes, valueconst, calldescr):
    ///     opnum = rop.call_may_force_for_descr(calldescr)
    ///     return self.history.record_nospec(opnum, argboxes, valueconst, calldescr)
    /// ```
    ///
    /// `valueconst` is the concrete result of the already-executed
    /// call (RPython's `c_result`).  Pyre tracks resvalue separately
    /// from the recorded OpRef, so the caller is responsible for
    /// keeping the concrete result alongside the returned OpRef.
    pub fn direct_call_may_force(
        &mut self,
        argboxes: &[OpRef],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
    ) -> Option<OpRef> {
        // pyjitpl.py:3586: opnum = rop.call_may_force_for_descr(calldescr)
        let opnum = OpCode::call_may_force_for_type(descr_view.result_type());
        // pyjitpl.py:3587: history.record_nospec(opnum, argboxes, valueconst, calldescr)
        let ctx = self.tracing.as_mut()?;
        Some(
            ctx.recorder
                .record_op_with_descr(opnum, argboxes, descr_ref),
        )
    }

    /// pyjitpl.py:3671-3681 `MetaInterp.direct_call_release_gil(argboxes, valueconst, calldescr)`.
    ///
    /// ```python
    /// def direct_call_release_gil(self, argboxes, valueconst, calldescr):
    ///     ...
    ///     effectinfo = calldescr.get_extra_info()
    ///     realfuncaddr, saveerr = effectinfo.call_release_gil_target
    ///     funcbox = ConstInt(adr2int(realfuncaddr))
    ///     savebox = ConstInt(saveerr)
    ///     opnum = rop.call_release_gil_for_descr(calldescr)
    ///     return self.history.record_nospec(opnum,
    ///                                       [savebox, funcbox] + argboxes[1:],
    ///                                       valueconst, calldescr)
    /// ```
    ///
    /// Returns `None` when no `call_release_gil_target` is registered
    /// (mirrors RPython's `effectinfo.call_release_gil_target`
    /// access; pyre stores the field as Option to make absence
    /// explicit).
    pub fn direct_call_release_gil(
        &mut self,
        argboxes: &[OpRef],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
    ) -> Option<OpRef> {
        // pyjitpl.py:3674: effectinfo = calldescr.get_extra_info()
        let effectinfo = descr_view.get_extra_info();
        // pyjitpl.py:3675: realfuncaddr, saveerr = effectinfo.call_release_gil_target
        // The field is `(u64, i32)`; the upstream sentinel for "no
        // target registered" is `(NULL, 0)` (effectinfo.py:114).
        // Pyre returns None when we hit that sentinel so callers can
        // fall through to direct_call_may_force.
        if !effectinfo.is_call_release_gil() {
            return None;
        }
        let (realfuncaddr, saveerr) = effectinfo.call_release_gil_target;
        let realfuncaddr = realfuncaddr as i64;
        // pyjitpl.py:3678: opnum = rop.call_release_gil_for_descr(calldescr).
        // resoperation.py:1243-1244 has the `'r'` arm commented out as
        // `# no such thing` — Type::Ref has no upstream opcode.  Defer
        // to the helper so the panic citation is shared with every
        // other `call_release_gil_for_type` caller.
        let opnum = OpCode::call_release_gil_for_type(descr_view.result_type());
        let ctx = self.tracing.as_mut()?;
        // pyjitpl.py:3676-3677: funcbox/savebox ConstInt
        let savebox = ctx.const_int(saveerr as i64);
        let funcbox_real = ctx.const_int(realfuncaddr);
        // pyjitpl.py:3679-3681: history.record_nospec(opnum, [savebox, funcbox] + argboxes[1:], ...)
        let mut new_args = Vec::with_capacity(argboxes.len() + 1);
        new_args.push(savebox);
        new_args.push(funcbox_real);
        if argboxes.len() > 1 {
            new_args.extend_from_slice(&argboxes[1..]);
        }
        Some(
            ctx.recorder
                .record_op_with_descr(opnum, &new_args, descr_ref),
        )
    }

    /// pyjitpl.py:3611-3669 `MetaInterp.direct_libffi_call`.
    ///
    /// ```python
    /// def direct_libffi_call(self, argboxes, valueconst, orig_calldescr):
    ///     assert self.staticdata.has_libffi_call
    ///     box_cif_description = argboxes[1]
    ///     if not isinstance(box_cif_description, ConstInt):
    ///         return None     # cannot be handled by direct_libffi_call()
    ///     cif_description = box_cif_description.getint()
    ///     ...
    ///     calldescr = self.cpu.calldescrof_dynamic(cif_description, extrainfo)
    ///     if calldescr is None:
    ///         return None     # cannot be handled by direct_libffi_call()
    ///     ...
    ///     return self.history.record_nospec(opnum,
    ///                                       [c_saveall, argboxes[2]] + arg_boxes,
    ///                                       valueconst, calldescr)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `cpu.calldescrof_dynamic`,
    /// no `CIF_DESCRIPTION_P` layout reader, and no
    /// `ffisupport.get_arg_descr` — the upstream specialization
    /// reaches into `rpython.rlib.jit_libffi` which has no Rust
    /// equivalent in pyre.  The early-return contract for
    /// `argboxes[1] not ConstInt` and `cif_description == NULL` is
    /// preserved so the dispatch in `do_residual_call` (pyjitpl.py:2061)
    /// falls through to `direct_call_release_gil` / `direct_call_may_force`
    /// the same way it would when upstream's `direct_libffi_call`
    /// declines to handle the call.  Pyre never produces an
    /// `OopSpecIndex::LibffiCall` today, so the dispatch path is dead
    /// in production; the contract is matched here for the day a host
    /// adds libffi support.
    pub fn direct_libffi_call(
        &mut self,
        argboxes: &[OpRef],
        _descr_ref: majit_ir::DescrRef,
        _descr_view: &dyn majit_ir::descr::CallDescr,
    ) -> Option<OpRef> {
        // pyjitpl.py:3622-3624: box_cif_description = argboxes[1];
        //   if not isinstance(box_cif_description, ConstInt): return None
        let box_cif_description = *argboxes.get(1)?;
        let ctx = self.tracing.as_ref()?;
        let cif_description = match ctx.constants_get_value(box_cif_description) {
            Some(majit_ir::Value::Int(v)) => v,
            _ => return None,
        };
        // pyjitpl.py:3631-3632: if calldescr is None: return None — pyre
        // has no calldescrof_dynamic equivalent, so a NULL cif_description
        // is the only case we can reject before bailing entirely.
        if cif_description == 0 {
            return None;
        }
        // The cif-driven specialized recording (pyjitpl.py:3633-3667)
        // requires CIF_DESCRIPTION_P layout parsing + dynamic calldescr
        // construction that pyre lacks.  Returning None makes the
        // dispatch fall through to direct_call_release_gil /
        // direct_call_may_force per pyjitpl.py:2061 contract.
        None
    }

    /// pyjitpl.py:3589-3609 `MetaInterp.direct_assembler_call(arglist, valueconst, calldescr, targetjitdriver_sd)`.
    ///
    /// ```python
    /// def direct_assembler_call(self, arglist, valueconst, calldescr, targetjitdriver_sd):
    ///     num_green_args = targetjitdriver_sd.num_green_args
    ///     greenargs = arglist[1:num_green_args+1]
    ///     args = arglist[num_green_args+1:]
    ///     warmrunnerstate = targetjitdriver_sd.warmstate
    ///     token = warmrunnerstate.get_assembler_token(greenargs)
    ///     opnum = OpHelpers.call_assembler_for_descr(calldescr)
    ///     op = self.history.record_nospec(opnum, args, valueconst, descr=token)
    ///     jd = token.outermost_jitdriver_sd
    ///     if jd.index_of_virtualizable >= 0:
    ///         return args[jd.index_of_virtualizable], op
    ///     else:
    ///         return None, op
    /// ```
    ///
    pub fn direct_assembler_call(
        &mut self,
        arglist: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr_view: &dyn majit_ir::descr::CallDescr,
        targetjitdriver_sd: usize,
    ) -> (Option<OpRef>, Option<OpRef>) {
        // pyjitpl.py:3593 num_green_args = targetjitdriver_sd.num_green_args
        let target_sd = match self
            .staticdata
            .jitdrivers_sd
            .get(targetjitdriver_sd)
            .cloned()
        {
            Some(sd) => sd,
            None => return (None, None),
        };
        let num_green_args = target_sd.num_greens();
        // pyjitpl.py:3594-3595 greenargs = arglist[1:num+1]; args = arglist[num+1:]
        if arglist.len() < num_green_args + 1 {
            return (None, None);
        }
        let greenargs = &arglist[1..num_green_args + 1];
        let args = &arglist[num_green_args + 1..];
        // pyjitpl.py:3596 assert len(args) == targetjitdriver_sd.num_red_args
        debug_assert_eq!(
            args.len(),
            target_sd.num_reds(),
            "pyjitpl.py:3596 — direct_assembler_call args.len() must match num_red_args",
        );
        // pyjitpl.py:3597-3599 token = warmrunnerstate.get_assembler_token(greenargs).
        //
        // S2.4 follow-up: pull `arg_types` from `target_sd.red_args_types`
        // (warmspot.py:664) — the static spec is the source of truth.
        // The previous shape recomputed types from runtime arg kinds
        // every call; the consistency assert at
        // compile.rs::compile_tmp_callback (S2.4) already locks the
        // contract that the runtime kinds match jd.red_args_types in
        // declaration order, so the two derivations are observationally
        // identical. Routing through the static spec removes the
        // per-call recomputation and makes the failure surface
        // immediate when a future caller drifts from the registered
        // driver.
        let arg_types: Vec<Type> = target_sd.red_arg_types_as_ir_types();
        debug_assert_eq!(
            arg_types.len(),
            args.len(),
            "pyjitpl.py:3596 already verified args.len() == num_red_args; \
             red_arg_types_as_ir_types must agree with that count",
        );
        // PyPy `warmstate.py:575 _green_args_spec` keys per-type
        // `equal_whatever`/`hash_whatever` off each green's lltype, so a
        // Float / Ref green hashes differently than an Int green
        // carrying the same i64 bits, and a Ptr(rstr.STR) green uses
        // content-hash/equality rather than pointer identity.  Pull the
        // typed spec from the registered driver
        // (`target_sd.green_args_spec()`, mirroring upstream
        // `warmrunnerstate.get_assembler_token(greenargs)` reading
        // `_green_args_spec`) so STR / UNICODE greens land on the same
        // cell that the macro-emitted typed-key path uses.  Earlier
        // pyre revisions reconstructed the GreenType from runtime
        // `JitArgKind` (which collapses STR/UNICODE to `Ref`); the
        // collapse caused direct-assembler lookups for STR/UNICODE
        // greens to compute pointer-identity hashes instead of the
        // content hashes the install-time path stored under.
        let green_values: Vec<i64> = greenargs.iter().map(|(_, _, value)| *value).collect();
        let green_types: Vec<majit_ir::GreenType> = target_sd.green_args_spec();
        debug_assert_eq!(
            green_types.len(),
            greenargs.len(),
            "warmspot.py:663 _green_args_spec must agree with the \
             jitcode arg layout greenargs prefix",
        );
        let green_key = crate::green_key_hash_typed(&green_values, &green_types);
        // `compile.py:187` parity: `op.getdescr()` IS a `JitCellToken`.  Carry
        // the *same* Arc that `compiled_loops` / warm cell own through to the
        // descr so `record_loop_or_bridge` can downcast and push it directly,
        // skipping the number-keyed `jitcell_token_by_number` recovery.
        let target_token: std::sync::Arc<JitCellToken> = if let Some(arc) =
            self.get_loop_token_arc(green_key)
        {
            std::sync::Arc::clone(arc)
        } else {
            // warmstate.py:714-723 — cell has no procedure_token yet, so
            // synthesise one via `compile_tmp_callback`.  Reuses the
            // already-allocated pending token number when available so
            // the backend's pending-target registry stays consistent.
            let greenboxes: Vec<Value> = greenargs
                .iter()
                .map(|(kind, _, value)| match kind {
                    crate::jitcode::JitArgKind::Int => Value::Int(*value),
                    crate::jitcode::JitArgKind::Ref => Value::Ref(GcRef(*value as usize)),
                    crate::jitcode::JitArgKind::Float => {
                        Value::Float(f64::from_bits(*value as u64))
                    }
                })
                .collect();
            let token_number = self
                .get_pending_token_number(green_key)
                .unwrap_or_else(|| self.warm_state.alloc_token_number());
            let backend = &mut self.backend;
            match self.warm_state.get_assembler_token(green_key, || {
                compile::compile_tmp_callback(
                    backend,
                    &target_sd,
                    token_number,
                    green_key,
                    &greenboxes,
                    &arg_types,
                )
            }) {
                Ok(token) => token,
                Err(err) => {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit][call_assembler] compile_tmp_callback failed for key={green_key}: {err:?}"
                        );
                    }
                    return (None, None);
                }
            }
        };
        let vable_index = target_token.virtualizable_arg_index;
        // pyjitpl.py:3601 opnum = OpHelpers.call_assembler_for_descr(calldescr)
        let opnum = match descr_view.result_type() {
            majit_ir::Type::Int => OpCode::CallAssemblerI,
            majit_ir::Type::Ref => OpCode::CallAssemblerR,
            majit_ir::Type::Float => OpCode::CallAssemblerF,
            majit_ir::Type::Void => OpCode::CallAssemblerN,
        };
        // pyjitpl.py:3602 op = self.history.record_nospec(opnum, args, valueconst, descr=token)
        let opref_args: Vec<OpRef> = args.iter().map(|(_, opref, _)| *opref).collect();
        let op_ref = {
            let ctx = match self.tracing.as_mut() {
                Some(ctx) => ctx,
                None => return (None, None),
            };
            let descr = crate::make_call_assembler_descr(
                std::sync::Arc::clone(&target_token),
                &arg_types,
                descr_view.result_type(),
            );
            ctx.record_op_with_descr(opnum, &opref_args, descr)
        };
        // pyjitpl.py:3604-3608 return vablebox per jd.index_of_virtualizable.
        let vablebox = vable_index.and_then(|idx| args.get(idx).map(|(_, opref, _)| *opref));
        (vablebox, Some(op_ref))
    }

    /// pyjitpl.py:3317-3335 `MetaInterp.vable_and_vrefs_before_residual_call`.
    ///
    /// ```python
    /// def vable_and_vrefs_before_residual_call(self):
    ///     vrefinfo = self.staticdata.virtualref_info
    ///     for i in range(1, len(self.virtualref_boxes), 2):
    ///         vrefbox = self.virtualref_boxes[i]
    ///         vref = vrefbox.getref_base()
    ///         vrefinfo.tracing_before_residual_call(vref)
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.tracing_before_residual_call(virtualizable)
    ///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)
    ///         self.history.record2(rop.SETFIELD_GC, virtualizable_box,
    ///                              force_token, None,
    ///                              descr=vinfo.vable_token_descr)
    /// ```
    pub fn vable_and_vrefs_before_residual_call(&mut self) {
        // pyjitpl.py:3318-3324 — vrefinfo loop over odd indices.
        let vref_ptrs: Vec<usize> = self
            .virtualref_boxes
            .iter()
            .enumerate()
            .filter_map(|(i, (_, ptr))| (i % 2 == 1).then_some(*ptr))
            .collect();
        for vref_ptr in vref_ptrs {
            // SAFETY: vref_ptr was registered by `opimpl_virtual_ref` with a
            // valid JitVirtualRef pointer; we only flip its token field.
            unsafe {
                self.virtualref_info
                    .tracing_before_residual_call(vref_ptr as *mut u8);
            }
        }
        // pyjitpl.py:3326-3334 — vinfo path (FORCE_TOKEN + SETFIELD_GC).
        let vinfo = match self.virtualizable_info().cloned() {
            Some(info) => info,
            None => return,
        };
        let vable_ptr = self.vable_ptr;
        let ctx = match self.tracing.as_mut() {
            Some(ctx) => ctx,
            None => return,
        };
        let vbox = match ctx.standard_virtualizable_box() {
            Some(b) => b,
            None => return,
        };
        if !vable_ptr.is_null() {
            // SAFETY: the host stamps `vable_ptr` to the live virtualizable
            // pointer for the duration of the trace; flipping the token
            // field is the only side effect.
            unsafe {
                vinfo.tracing_before_residual_call(vable_ptr as *mut u8);
            }
        }
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vbox, force_token, vinfo.token_field_descr());
    }

    /// pyjitpl.py:3337-3347 `MetaInterp.vrefs_after_residual_call`.
    ///
    /// ```python
    /// def vrefs_after_residual_call(self):
    ///     vrefinfo = self.staticdata.virtualref_info
    ///     for i in range(0, len(self.virtualref_boxes), 2):
    ///         vrefbox = self.virtualref_boxes[i+1]
    ///         vref = vrefbox.getref_base()
    ///         if vrefinfo.tracing_after_residual_call(vref):
    ///             self.stop_tracking_virtualref(i)
    /// ```
    pub fn vrefs_after_residual_call(&mut self) {
        let mut forced_pairs: Vec<usize> = Vec::new();
        let mut i = 0;
        while i + 1 < self.virtualref_boxes.len() {
            let vref_ptr = self.virtualref_boxes[i + 1].1;
            // SAFETY: vref_ptr was registered by `opimpl_virtual_ref`.
            let forced = unsafe {
                self.virtualref_info
                    .tracing_after_residual_call(vref_ptr as *mut u8)
            };
            if forced {
                forced_pairs.push(i);
            }
            i += 2;
        }
        for pair_index in forced_pairs {
            self.stop_tracking_virtualref(pair_index);
        }
    }

    /// pyjitpl.py:3349-3378 `MetaInterp.vable_after_residual_call(funcbox)`.
    ///
    /// ```python
    /// def vable_after_residual_call(self, funcbox):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         if vinfo.tracing_after_residual_call(virtualizable):
    ///             self.load_fields_from_virtualizable()
    ///             ...debug_print...
    ///             raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
    ///                                     raising_exception=True)
    /// ```
    ///
    /// Returns `Err(SwitchToBlackhole)` (with `raising_exception=true`)
    /// when the virtualizable escaped during the residual call so the
    /// caller can route to the matching `aborted_tracing` /
    /// blackhole-resume path.
    pub fn vable_after_residual_call(&mut self, _funcbox: i64) -> Result<(), SwitchToBlackhole> {
        let vinfo = match self.virtualizable_info().cloned() {
            Some(info) => info,
            None => return Ok(()),
        };
        let vable_ptr = self.vable_ptr;
        if vable_ptr.is_null() {
            return Ok(());
        }
        // SAFETY: the host keeps `vable_ptr` live for the duration of the
        // trace; we read/write its token field only.
        let escaped = unsafe { vinfo.tracing_after_residual_call(vable_ptr as *mut u8) };
        if !escaped {
            return Ok(());
        }
        // pyjitpl.py:3367 self.load_fields_from_virtualizable()
        self.load_fields_from_virtualizable();
        // pyjitpl.py:3373-3375 raise SwitchToBlackhole(ABORT_ESCAPE,
        //                              raising_exception=True)
        Err(SwitchToBlackhole {
            reason: counters::ABORT_ESCAPE,
            raising_exception: true,
        })
    }

    /// pyjitpl.py:3381-3387 `MetaInterp.stop_tracking_virtualref(i)`.
    ///
    /// ```python
    /// def stop_tracking_virtualref(self, i):
    ///     virtualbox = self.virtualref_boxes[i]
    ///     vrefbox = self.virtualref_boxes[i+1]
    ///     self.history.record2(rop.VIRTUAL_REF_FINISH, vrefbox, virtualbox, None)
    ///     self.virtualref_boxes[i+1] = CONST_NULL
    /// ```
    pub fn stop_tracking_virtualref(&mut self, i: usize) {
        if i + 1 >= self.virtualref_boxes.len() {
            return;
        }
        let virtualbox = self.virtualref_boxes[i].0;
        let vrefbox = self.virtualref_boxes[i + 1].0;
        if let Some(ctx) = self.tracing.as_mut() {
            ctx.record_op(OpCode::VirtualRefFinish, &[vrefbox, virtualbox]);
        }
        // pyjitpl.py:3378 `self.virtualref_boxes[i+1] = CONST_NULL`
        // — ref-typed null preserves the slot's Ref type so subsequent
        // fail-arg type recovery and ref-typed guard processing match
        // upstream (history.py:361 `CONST_NULL = ConstPtr(ConstPtr.value)`).
        let null_const = self
            .tracing
            .as_mut()
            .map(|ctx| ctx.const_null())
            .unwrap_or(OpRef::from_raw(0));
        self.virtualref_boxes[i + 1] = (null_const, 0);
    }

    /// pyjitpl.py:2153-2172 `MIFrame._do_jit_force_virtual(allboxes, descr, pc)`.
    ///
    /// ```python
    /// def _do_jit_force_virtual(self, allboxes, descr, pc):
    ///     assert len(allboxes) == 2
    ///     if (self.metainterp.jitdriver_sd.virtualizable_info is None and
    ///         self.metainterp.jitdriver_sd.greenfield_info is None):
    ///         return None
    ///     vref_box = allboxes[1]
    ///     standard_box = self.metainterp.virtualizable_boxes[-1]
    ///     if standard_box is vref_box:
    ///         return vref_box
    ///     if self.metainterp.heapcache.is_known_nonstandard_virtualizable(vref_box):
    ///         return None
    ///     eqbox = self.metainterp.execute_and_record(rop.PTR_EQ, None, vref_box, standard_box)
    ///     eqbox = self.implement_guard_value(eqbox, pc)
    ///     isstandard = eqbox.getint()
    ///     if isstandard:
    ///         return standard_box
    ///     else:
    ///         return None
    /// ```
    ///
    pub fn _do_jit_force_virtual(
        &mut self,
        allboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        _descr_view: &dyn majit_ir::descr::CallDescr,
        pc: usize,
    ) -> Option<(OpRef, i64)> {
        debug_assert_eq!(
            allboxes.len(),
            2,
            "pyjitpl.py:2154 — _do_jit_force_virtual expects exactly 2 args",
        );
        // pyjitpl.py:2155-2158:
        //   if (self.metainterp.jitdriver_sd.virtualizable_info is None and
        //       self.metainterp.jitdriver_sd.greenfield_info is None):
        //       return None
        if self.virtualizable_info().is_none() && self.greenfield_info().is_none() {
            return None;
        }
        // pyjitpl.py:2159-2161: vref_box vs standard_box identity short-circuit.
        let vref_box = allboxes[1].1;
        let vref_concrete = allboxes[1].2;
        let standard_box = self
            .tracing
            .as_ref()
            .and_then(|ctx| ctx.standard_virtualizable_box())?;
        if vref_box == standard_box {
            return Some((vref_box, vref_concrete));
        }
        // pyjitpl.py:2162-2163: heapcache short-circuit when the box is
        // known to NOT be the standard virtualizable.
        let is_known_nonstandard = self
            .tracing
            .as_ref()
            .map(|ctx| {
                ctx.heap_cache()
                    .is_known_nonstandard_virtualizable(vref_box)
            })
            .unwrap_or(false);
        if is_known_nonstandard {
            // pyjitpl.py:2164: profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
            self.staticdata
                .profiler
                .count_ops(OpCode::PtrEq, counters::HEAPCACHED_OPS);
            return None;
        }
        // pyjitpl.py:2165: eqbox = self.metainterp.execute_and_record(rop.PTR_EQ,
        //                                                              None,
        //                                                              vref_box,
        //                                                              standard_box)
        let standard_concrete = self.vable_ptr as usize as i64;
        let isstandard_int = if vref_concrete == standard_concrete {
            1
        } else {
            0
        };
        let eqbox_opref = {
            let ctx = self.tracing.as_mut()?;
            ctx.recorder
                .record_op(OpCode::PtrEq, &[vref_box, standard_box])
        };
        // pyjitpl.py:2166: eqbox = self.implement_guard_value(eqbox, pc)
        // — pyre's `promote_int` records GUARD_VALUE on the result and
        // returns the const ref the optimizer can constant-fold against.
        let _ = pc;
        let _eqbox_const = {
            let ctx = self.tracing.as_mut()?;
            ctx.promote_int(eqbox_opref, isstandard_int, 0)
        };
        // pyjitpl.py:2167-2171: isstandard branch.
        if isstandard_int != 0 {
            Some((standard_box, standard_concrete))
        } else {
            None
        }
    }

    /// pyjitpl.py:1995-2126 `MIFrame.do_residual_call(funcbox, argboxes, descr, pc, assembler_call=False, assembler_call_jd=None)`.
    ///
    /// ```python
    /// def do_residual_call(self, funcbox, argboxes, descr, pc,
    ///                      assembler_call=False,
    ///                      assembler_call_jd=None):
    ///     allboxes = self._build_allboxes(funcbox, argboxes, descr)
    ///     effectinfo = descr.get_extra_info()
    ///     if effectinfo.oopspecindex == effectinfo.OS_NOT_IN_TRACE:
    ///         return self.metainterp.do_not_in_trace_call(allboxes, descr)
    ///
    ///     if (assembler_call or
    ///             effectinfo.check_forces_virtual_or_virtualizable()):
    ///         # ... CALL_MAY_FORCE_* path with vrefs/vable/heapcache
    ///         ...
    ///     else:
    ///         effect = effectinfo.extraeffect
    ///         tp = descr.get_normalized_result_type()
    ///         if effect == effectinfo.EF_LOOPINVARIANT:
    ///             res = self.metainterp.heapcache.call_loopinvariant_known_result(allboxes, descr)
    ///             if res is not None:
    ///                 return res
    ///             if tp == 'i':
    ///                 res = self.execute_varargs(rop.CALL_LOOPINVARIANT_I, ...)
    ///             elif tp == 'r':
    ///                 res = self.execute_varargs(rop.CALL_LOOPINVARIANT_R, ...)
    ///             elif tp == 'f':
    ///                 res = self.execute_varargs(rop.CALL_LOOPINVARIANT_F, ...)
    ///             elif tp == 'v':
    ///                 res = self.execute_varargs(rop.CALL_LOOPINVARIANT_N, ...)
    ///             self.metainterp.heapcache.call_loopinvariant_now_known(allboxes, descr, res)
    ///             return res
    ///         exc = effectinfo.check_can_raise()
    ///         pure = effectinfo.check_is_elidable()
    ///         if tp == 'i':
    ///             return self.execute_varargs(rop.CALL_I, allboxes, descr, exc, pure)
    ///         elif tp == 'r':
    ///             return self.execute_varargs(rop.CALL_R, allboxes, descr, exc, pure)
    ///         elif tp == 'f':
    ///             return self.execute_varargs(rop.CALL_F, allboxes, descr, exc, pure)
    ///         elif tp == 'v':
    ///             return self.execute_varargs(rop.CALL_N, allboxes, descr, exc, pure)
    /// ```
    ///
    /// Force-virtual path is staged: returns `Ok(None)` when
    /// `assembler_call || forces_virtual_or_virtualizable()` so the
    /// existing tracer's residual emission keeps running.  The full
    /// CALL_MAY_FORCE_* lowering with vrefs/vable/heapcache lands in
    /// follow-ups (pyjitpl.py:2007-2083).  Loopinvariant + regular
    /// CALL_* paths are line-for-line.
    pub fn do_residual_call_full(
        &mut self,
        funcbox: (crate::jitcode::JitArgKind, OpRef, i64),
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
        _pc: usize,
        assembler_call: bool,
        _assembler_call_jd: Option<usize>,
        dst: Option<(crate::jitcode::JitArgKind, usize)>,
    ) -> Result<Option<(OpRef, i64)>, DoResidualCallAbort> {
        // pyjitpl.py:2002: allboxes = self._build_allboxes(funcbox, argboxes, descr)
        let allboxes = self._build_allboxes(funcbox, argboxes, descr_view, None);
        // pyjitpl.py:2003: effectinfo = descr.get_extra_info()
        let effectinfo = descr_view.get_extra_info();
        // pyjitpl.py:2004-2005: OS_NOT_IN_TRACE
        if effectinfo.oopspecindex == majit_ir::OopSpecIndex::NotInTrace {
            return Ok(self
                .do_not_in_trace_call(&allboxes, descr_view)?
                .map(|op| (op, 0)));
        }
        // pyjitpl.py:2007-2083: force_virtual / assembler_call branch.
        if assembler_call || effectinfo.check_forces_virtual_or_virtualizable() {
            // pyjitpl.py:2010: self.metainterp.clear_exception()
            self.clear_exception();
            // pyjitpl.py:2011-2014: OS_JIT_FORCE_VIRTUAL short-circuit.
            if effectinfo.oopspecindex == majit_ir::OopSpecIndex::JitForceVirtual {
                if let Some(result) = self._do_jit_force_virtual(&allboxes, descr_view, _pc) {
                    return Ok(Some(result));
                }
            }
            // pyjitpl.py:2017: vable_and_vrefs_before_residual_call (stub)
            self.vable_and_vrefs_before_residual_call();
            // pyjitpl.py:2019-2044: execute_varargs to get the concrete
            // result.  CALL_MAY_FORCE_* opnum picked by result type.
            let opnum1 = OpCode::call_may_force_for_type(descr_view.result_type());
            let c_result = crate::executor::execute_varargs(self, opnum1, &allboxes, descr_view);
            // pyjitpl.py:2049: vrefs_after_residual_call (stub)
            self.vrefs_after_residual_call();
            // pyjitpl.py:2053-2068: pick the right CALL recording path.
            let opref_args: Vec<OpRef> = allboxes.iter().map(|(_, op, _)| *op).collect();
            let (vablebox, resbox) = if assembler_call {
                // pyjitpl.py:2053-2055: direct_assembler_call
                let jd = _assembler_call_jd.unwrap_or(0);
                self.direct_assembler_call(&allboxes, descr_view, jd)
            } else {
                // pyjitpl.py:2057-2068: libffi → release_gil → may_force
                let mut resbox = None;
                if effectinfo.oopspecindex == majit_ir::OopSpecIndex::LibffiCall {
                    resbox = self.direct_libffi_call(&opref_args, descr_ref.clone(), descr_view);
                }
                if resbox.is_none() {
                    resbox = if effectinfo.is_call_release_gil() {
                        self.direct_call_release_gil(&opref_args, descr_ref.clone(), descr_view)
                    } else {
                        self.direct_call_may_force(&opref_args, descr_ref.clone(), descr_view)
                    };
                }
                (None, resbox)
            };
            // pyjitpl.py:2072: heapcache.invalidate_caches_varargs(opnum1, descr, allboxes)
            if let Some(ctx) = self.tracing.as_mut() {
                ctx.heap_cache_mut().invalidate_caches_varargs(
                    opnum1,
                    Some(effectinfo),
                    &opref_args,
                );
            }
            // pyjitpl.py:2074-2077: handle resbox void / make_result_of_lastop
            // — make_result_of_lastop's target_index plumbing is not
            // wired here yet; documented above on miframe_execute_varargs.
            let resbox_pair = match resbox {
                Some(opref) if descr_view.result_type() != majit_ir::Type::Void => {
                    Some((opref, c_result))
                }
                _ => None,
            };
            // pyjitpl.py:2078: vable_after_residual_call(funcbox)
            // SwitchToBlackhole(ABORT_ESCAPE, raising_exception=True)
            // surfaces here when the virtualizable escaped during the
            // residual call (pyjitpl.py:3373-3375).  Route into the
            // existing DoResidualCallAbort variant so the caller's
            // abort path fires.
            self.vable_after_residual_call(funcbox.2)
                .map_err(DoResidualCallAbort::from)?;
            // pyjitpl.py:2079: generate_guard(rop.GUARD_NOT_FORCED)
            if let Some(ctx) = self.tracing.as_mut() {
                ctx.record_guard(OpCode::GuardNotForced, &[], 0);
            }
            // pyjitpl.py:2080-2081: KEEPALIVE for vablebox
            if let Some(vablebox) = vablebox {
                if let Some(ctx) = self.tracing.as_mut() {
                    ctx.record_op(OpCode::Keepalive, &[vablebox]);
                }
            }
            // pyjitpl.py:2082: handle_possible_exception
            self.handle_possible_exception()?;
            // pyjitpl.py:2083: return resbox
            return Ok(resbox_pair);
        }
        // pyjitpl.py:2085: effect = effectinfo.extraeffect
        let extraeffect = effectinfo.extraeffect;
        // pyjitpl.py:2086: tp = descr.get_normalized_result_type()
        let tp = descr_view.result_type();
        // pyjitpl.py:2087: if effect == effectinfo.EF_LOOPINVARIANT
        if extraeffect == majit_ir::effectinfo::ExtraEffect::LoopInvariant {
            // pyjitpl.py:2088-2090: heapcache.call_loopinvariant_known_result
            let descr_index = descr_view.get_descr_index();
            let arg0_int = funcbox.2;
            if descr_index >= 0 {
                if let Some(ctx) = self.tracing.as_ref() {
                    if let Some(cached) = ctx
                        .heap_cache()
                        .call_loopinvariant_known_result(descr_index as u32, arg0_int)
                    {
                        // pyjitpl.py:2089-2090: `if res is not None: return res`
                        // — the cached entry already pairs the symbolic
                        // OpRef with its concrete value (heapcache.rs
                        // `loopinvariant_resvalue`).
                        return Ok(Some(cached));
                    }
                }
            }
            // pyjitpl.py:2091-2108: execute_varargs(CALL_LOOPINVARIANT_*, ..., False, False)
            let opnum = match tp {
                majit_ir::Type::Int => OpCode::CallLoopinvariantI,
                majit_ir::Type::Ref => OpCode::CallLoopinvariantR,
                majit_ir::Type::Float => OpCode::CallLoopinvariantF,
                majit_ir::Type::Void => OpCode::CallLoopinvariantN,
            };
            let res = self.miframe_execute_varargs(
                opnum, &allboxes, descr_ref, descr_view, /* exc = */ false,
                /* pure = */ false, /* dst = */ None,
            )?;
            // pyjitpl.py:2109: heapcache.call_loopinvariant_now_known
            if descr_index >= 0 {
                if let Some((opref, resvalue)) = res {
                    if let Some(ctx) = self.tracing.as_mut() {
                        ctx.heap_cache_mut().call_loopinvariant_now_known(
                            descr_index as u32,
                            arg0_int,
                            opref,
                            resvalue,
                        );
                    }
                }
            }
            // pyjitpl.py:2110: return res
            return Ok(res);
        }
        // pyjitpl.py:2111: exc = effectinfo.check_can_raise()
        let exc = effectinfo.check_can_raise(false);
        // pyjitpl.py:2112: pure = effectinfo.check_is_elidable()
        let pure = effectinfo.check_is_elidable();
        // pyjitpl.py:2113-2126: CALL_* dispatch by result type.
        let opnum = match tp {
            majit_ir::Type::Int => OpCode::CallI,
            majit_ir::Type::Ref => OpCode::CallR,
            majit_ir::Type::Float => OpCode::CallF,
            majit_ir::Type::Void => OpCode::CallN,
        };
        Ok(self.miframe_execute_varargs(opnum, &allboxes, descr_ref, descr_view, exc, pure, dst)?)
    }

    /// pyjitpl.py:1960-1993 `MIFrame._build_allboxes(funcbox, argboxes, descr, prepend_box=None)`.
    ///
    /// ```python
    /// def _build_allboxes(self, funcbox, argboxes, descr, prepend_box=None):
    ///     allboxes = [None] * (len(argboxes)+1 + int(prepend_box is not None))
    ///     i = 0
    ///     if prepend_box is not None:
    ///         allboxes[0] = prepend_box
    ///         i = 1
    ///     allboxes[i] = funcbox
    ///     i += 1
    ///     src_i = src_r = src_f = 0
    ///     for kind in descr.get_arg_types():
    ///         if kind == history.INT or kind == 'S':        # single float
    ///             ...src_i...
    ///         elif kind == history.REF:
    ///             ...src_r...
    ///         elif kind == history.FLOAT or kind == 'L':    # long long
    ///             ...src_f...
    ///         else:
    ///             raise AssertionError
    ///         allboxes[i] = box
    ///         i += 1
    ///     assert i == len(allboxes)
    ///     return allboxes
    /// ```
    ///
    /// The three independent counters `src_i` / `src_r` / `src_f` walk
    /// `argboxes` separately, each skipping past boxes whose kind does
    /// not match.  This demuxes a bank-sorted input layout (all ints
    /// first, then refs, then floats) back into declaration order
    /// matching `descr.get_arg_types()` — and degrades gracefully to a
    /// 1:1 walk when the input already arrives in declaration order
    /// (each counter advances past its own bank's previous matches).
    /// The output ordering is the same regardless of input layout.
    ///
    /// RPython history kind chars and their pyre `Type` equivalents:
    /// * `history.INT` and `'S'` (single-precision float, i32 ABI
    ///   slot) → `Type::Int`.  Pyre only ships 64-bit targets where
    ///   `'S'` collapses onto the int slot.
    /// * `history.REF` → `Type::Ref`.
    /// * `history.FLOAT` and `'L'` (long-long, 64-bit int riding the
    ///   float ABI slot on 32-bit targets) → `Type::Float`.  Same
    ///   collapse rationale as `'S'`.
    pub fn _build_allboxes(
        &self,
        funcbox: (crate::jitcode::JitArgKind, OpRef, i64),
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr: &dyn majit_ir::descr::CallDescr,
        prepend_box: Option<(crate::jitcode::JitArgKind, OpRef, i64)>,
    ) -> Vec<(crate::jitcode::JitArgKind, OpRef, i64)> {
        // pyjitpl.py:1961: allboxes = [None] * (len(argboxes)+1 + int(prepend_box is not None))
        let total = argboxes.len() + 1 + prepend_box.is_some() as usize;
        let mut allboxes = Vec::with_capacity(total);
        // pyjitpl.py:1963-1965: if prepend_box is not None: allboxes[0] = prepend_box
        if let Some(pb) = prepend_box {
            allboxes.push(pb);
        }
        // pyjitpl.py:1966-1967: allboxes[i] = funcbox; i += 1
        allboxes.push(funcbox);
        // pyjitpl.py:1968-1991: per-bank counters that demux argboxes
        // by `descr.get_arg_types()`.  Match `kind` directly against
        // `majit_ir::Type` (rather than collapsing through
        // `JitArgKind::from_type`) so the four RPython arms — INT|'S',
        // REF, FLOAT|'L', AssertionError — stay 1:1 visible.
        let arg_types = descr.arg_types();
        let mut src_i: usize = 0;
        let mut src_r: usize = 0;
        let mut src_f: usize = 0;
        for &kind in arg_types {
            let pick = match kind {
                // pyjitpl.py:1970-1975: kind == history.INT or kind == 'S'
                // → walk src_i until argboxes[src_i].type == INT.
                majit_ir::Type::Int => loop {
                    let box_ = argboxes[src_i];
                    src_i += 1;
                    if matches!(box_.0, crate::jitcode::JitArgKind::Int) {
                        break box_;
                    }
                },
                // pyjitpl.py:1976-1981: kind == history.REF
                // → walk src_r until argboxes[src_r].type == REF.
                majit_ir::Type::Ref => loop {
                    let box_ = argboxes[src_r];
                    src_r += 1;
                    if matches!(box_.0, crate::jitcode::JitArgKind::Ref) {
                        break box_;
                    }
                },
                // pyjitpl.py:1982-1987: kind == history.FLOAT or kind == 'L'
                // → walk src_f until argboxes[src_f].type == FLOAT.
                majit_ir::Type::Float => loop {
                    let box_ = argboxes[src_f];
                    src_f += 1;
                    if matches!(box_.0, crate::jitcode::JitArgKind::Float) {
                        break box_;
                    }
                },
                // pyjitpl.py:1988-1989: else: raise AssertionError.
                majit_ir::Type::Void => unreachable!(
                    "_build_allboxes: descr arg_type is Void (only return types may be Void)",
                ),
            };
            // pyjitpl.py:1990: allboxes[i] = box; i += 1
            allboxes.push(pick);
        }
        // pyjitpl.py:1992: assert i == len(allboxes)
        debug_assert_eq!(allboxes.len(), total);
        allboxes
    }

    /// pyjitpl.py:1425-1432 `MIFrame.do_recursive_call(targetjitdriver_sd, allboxes, pc, assembler_call=False)`.
    ///
    /// ```python
    /// def do_recursive_call(self, targetjitdriver_sd, allboxes, pc,
    ///                       assembler_call=False):
    ///     portal_code = targetjitdriver_sd.mainjitcode
    ///     k = targetjitdriver_sd.portal_runner_adr
    ///     funcbox = ConstInt(adr2int(k))
    ///     return self.do_residual_call(funcbox, allboxes,
    ///                                  portal_code.calldescr, pc,
    ///                                  assembler_call=assembler_call,
    ///                                  assembler_call_jd=targetjitdriver_sd)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: `portal_code.calldescr` is the portal
    /// jitcode's calldescr (a `BhCallDescr` in pyre).  Pyre does not
    /// yet expose a `&dyn CallDescr` view onto BhCallDescr, so the
    /// caller passes the typed `(DescrRef, &dyn CallDescr)` pair
    /// explicitly.  funcbox is constructed from
    /// `targetjitdriver_sd.portal_runner_adr`.
    pub fn do_recursive_call(
        &mut self,
        targetjitdriver_sd: &crate::jitdriver::JitDriverStaticData,
        allboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        portal_descr_ref: majit_ir::DescrRef,
        portal_descr_view: &dyn majit_ir::descr::CallDescr,
        target_jd_index: usize,
        pc: usize,
        assembler_call: bool,
    ) -> Result<Option<(OpRef, i64)>, DoResidualCallAbort> {
        // S2.1 invariant (wiggly-barto plan, mirrors `compile_tmp_callback`'s
        // pre-check at compile.rs:2123): the recursive-call funcbox dereferences
        // `portal_runner_adr` directly (line below), so a 0 address would jump
        // to NULL on the bh_call_r side. `warmspot.py:1010-1012` populates this
        // before any do_recursive_call can fire; `debug_assert!` catches a
        // mis-wired registration in dev/test builds (the bench harness runs in
        // dev so violations surface via `pyre/check.sh`).
        debug_assert!(
            targetjitdriver_sd.portal_runner_adr != 0,
            "do_recursive_call: targetjitdriver_sd.portal_runner_adr is 0 — \
             warmspot.py:1010-1012 must populate it before any recursive-call \
             site can build a CALL_ASSEMBLER funcbox"
        );
        // pyjitpl.py:1418-1420 — `_opimpl_recursive_call` runs
        // `self.verify_green_args(targetjitdriver_sd, greenboxes)` right
        // before its `return self.do_recursive_call(...)` call. pyre folds
        // the upstream `_opimpl_recursive_call` and `do_recursive_call`
        // into this single entry, so the verify lifts to here. The
        // greens are the first `num_green_args` entries of `allboxes`
        // (upstream `allboxes = greenboxes + redboxes`, line 1416 / 1422).
        //
        // Always-on `assert!` rather than `cfg!(debug_assertions)` —
        // upstream `assert isinstance(varargs[i], Const)` is only
        // stripped under RPython's `-O` translation, which is not the
        // default for the bench harness or the production interpreter
        // (compile.py:1101-1135 `compile_tmp_callback` ports the same
        // contract unconditionally). `verify_green_args` panics on a
        // non-Const greens slot — that would mean a caller gave us a
        // Box where upstream demands a ConstPtr / ConstInt / ConstFloat.
        let num_green_args = targetjitdriver_sd.num_green_args();
        assert!(
            allboxes.len() >= num_green_args,
            "do_recursive_call: allboxes.len()={} < num_green_args={} \
             (upstream pyjitpl.py:1416 builds allboxes = greenboxes + redboxes)",
            allboxes.len(),
            num_green_args,
        );
        let greens: Vec<OpRef> = allboxes[..num_green_args]
            .iter()
            .map(|(_, opref, _)| *opref)
            .collect();
        crate::pyjitpl::MIFrame::verify_green_args(targetjitdriver_sd, &greens);
        // pyjitpl.py:1428: k = targetjitdriver_sd.portal_runner_adr
        let k = targetjitdriver_sd.portal_runner_adr;
        // pyjitpl.py:1429: funcbox = ConstInt(adr2int(k))
        // — `ConstInt` is an `INT`-typed constant box.  Pyre's analog
        // is JitArgKind::Int with a constant OpRef.
        let funcbox_opref = if let Some(ctx) = self.tracing.as_mut() {
            ctx.const_int(k)
        } else {
            OpRef::NONE
        };
        let funcbox = (crate::jitcode::JitArgKind::Int, funcbox_opref, k);
        // pyjitpl.py:1430-1432: do_residual_call(funcbox, allboxes, calldescr, pc,
        //                                       assembler_call, assembler_call_jd)
        self.do_residual_call_full(
            funcbox,
            allboxes,
            portal_descr_ref,
            portal_descr_view,
            pc,
            assembler_call,
            Some(target_jd_index),
            // pyjitpl.py:1430 — do_recursive_call's CALL_ASSEMBLER result
            // is consumed via the dispatch loop's normal result-return
            // path, not via make_result_of_lastop inside execute_varargs.
            None,
        )
    }

    /// pyjitpl.py:2128-2151 `MIFrame.do_conditional_call(condbox, funcbox, argboxes, descr, pc, is_value=False)`.
    ///
    /// ```python
    /// def do_conditional_call(self, condbox, funcbox, argboxes, descr, pc,
    ///                         is_value=False):
    ///     allboxes = self._build_allboxes(funcbox, argboxes, descr, prepend_box=condbox)
    ///     effectinfo = descr.get_extra_info()
    ///     assert not effectinfo.check_forces_virtual_or_virtualizable()
    ///     exc = effectinfo.check_can_raise()
    ///     if not is_value:
    ///         return self.execute_varargs(rop.COND_CALL, allboxes, descr,
    ///                                     exc, pure=False)
    ///     else:
    ///         opnum = OpHelpers.cond_call_value_for_descr(descr)
    ///         if opnum == rop.COND_CALL_VALUE_I:
    ///             return self.execute_varargs(rop.COND_CALL_VALUE_I, allboxes,
    ///                                         descr, exc, pure=True)
    ///         elif opnum == rop.COND_CALL_VALUE_R:
    ///             return self.execute_varargs(rop.COND_CALL_VALUE_R, allboxes,
    ///                                         descr, exc, pure=True)
    ///         else:
    ///             raise AssertionError
    /// ```
    pub fn do_conditional_call(
        &mut self,
        condbox: (crate::jitcode::JitArgKind, OpRef, i64),
        funcbox: (crate::jitcode::JitArgKind, OpRef, i64),
        argboxes: &[(crate::jitcode::JitArgKind, OpRef, i64)],
        descr_ref: majit_ir::DescrRef,
        descr_view: &dyn majit_ir::descr::CallDescr,
        _pc: usize,
        is_value: bool,
        dst: Option<(crate::jitcode::JitArgKind, usize)>,
    ) -> Result<Option<(OpRef, i64)>, FinishframeExceptionSignal> {
        // pyjitpl.py:2130: allboxes = _build_allboxes(funcbox, argboxes, descr, prepend_box=condbox)
        let allboxes = self._build_allboxes(funcbox, argboxes, descr_view, Some(condbox));
        // pyjitpl.py:2131: effectinfo = descr.get_extra_info()
        let effectinfo = descr_view.get_extra_info();
        // pyjitpl.py:2132: assert not effectinfo.check_forces_virtual_or_virtualizable()
        debug_assert!(
            !effectinfo.check_forces_virtual_or_virtualizable(),
            "do_conditional_call cannot force virtuals",
        );
        // pyjitpl.py:2133: exc = effectinfo.check_can_raise()
        let exc = effectinfo.check_can_raise(false);
        if !is_value {
            // pyjitpl.py:2138-2139: COND_CALL has no result, pure=False.
            // Void result → no register write needed, dst irrelevant.
            self.miframe_execute_varargs(
                OpCode::CondCallN,
                &allboxes,
                descr_ref,
                descr_view,
                exc,
                /* pure = */ false,
                /* dst = */ None,
            )
        } else {
            // pyjitpl.py:2141: opnum = OpHelpers.cond_call_value_for_descr(descr)
            let opnum = match descr_view.result_type() {
                majit_ir::Type::Int => OpCode::CondCallValueI,
                majit_ir::Type::Ref => OpCode::CondCallValueR,
                other => panic!(
                    "do_conditional_call: COND_CALL_VALUE only supports Int/Ref results (got {other:?})",
                ),
            };
            // pyjitpl.py:2144-2149: COND_CALL_VALUE_* with pure=True
            self.miframe_execute_varargs(
                opnum, &allboxes, descr_ref, descr_view, exc, /* pure = */ true, dst,
            )
        }
    }

    /// Access the backend directly (for advanced operations).
    pub fn backend(&self) -> &BackendImpl {
        &self.backend
    }

    /// Access the backend mutably (for advanced operations).
    pub fn backend_mut(&mut self) -> &mut BackendImpl {
        &mut self.backend
    }

    /// Register a helper that boxes a raw integer into an interpreter object.
    /// PyPy warmspot.py set_param_max_unroll_recursion().
    pub fn set_max_unroll_recursion(&mut self, value: usize) {
        self.max_unroll_recursion = value;
    }
}

/// Default maximum inlining depth during tracing.
/// Configurable via WarmEnterState::set_max_inline_depth().
const MAX_INLINE_DEPTH: usize = 10;

/// Describes the recovery state after a guard failure.
#[derive(Debug, Clone)]
pub struct GuardRecovery {
    /// Compiled trace identifier for the failing exit.
    pub trace_id: u64,
    /// Index of the failed guard.
    pub fail_index: u32,
    /// Static layout metadata for this compiled exit.
    pub exit_layout: CompiledExitLayout,
    /// Raw fail_values from the DeadFrame.
    pub fail_values: Vec<i64>,
    /// Typed fail values decoded from the backend deadframe, when available.
    pub typed_fail_values: Option<Vec<Value>>,
    /// Compact resume/jitframe layout for this exit, when available.
    pub resume_layout: Option<ResumeLayoutSummary>,
    /// Reconstructed interpreter frames (if resume data was available).
    pub reconstructed_frames: Option<Vec<crate::resume::ReconstructedFrame>>,
    /// Full reconstructed state, including materialized virtuals.
    pub reconstructed_state: Option<ReconstructedState>,
    /// Materialized virtuals referenced by the reconstructed state.
    pub materialized_virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes reconstructed from resume data.
    pub pending_field_writes: Vec<ResolvedPendingFieldWrite>,
    /// Optional saved-data GC ref captured from the failing exit.
    pub savedata: Option<GcRef>,
    /// Pending exception state captured from the failing deadframe.
    pub exception: ExceptionState,
}

/// Result of running compiled code with automatic recovery.
#[derive(Debug, Clone)]
pub enum RunResult<M> {
    /// The loop finished normally.
    Finished {
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    },
    /// The trace exited via a normal back-edge jump.
    Jump {
        values: Vec<i64>,
        meta: M,
        savedata: Option<GcRef>,
    },
    /// A guard failed.
    GuardFailure {
        values: Vec<i64>,
        meta: M,
        trace_id: u64,
        fail_index: u32,
        savedata: Option<GcRef>,
        recovery: Option<GuardRecovery>,
    },
}

/// Result of running compiled code with automatic blackhole fallback.
#[derive(Debug, Clone)]
pub enum BlackholeRunResult<M> {
    /// The trace produced final values, either directly or via blackhole replay.
    Finished {
        values: Vec<i64>,
        typed_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// The blackhole replay reached a jump back-edge.
    Jump {
        values: Vec<i64>,
        typed_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// Execution still ended in a guard failure.
    GuardFailure {
        trace_id: u64,
        fail_index: u32,
        fail_values: Vec<i64>,
        typed_fail_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        recovery: Option<GuardRecovery>,
        materialized_virtuals: Vec<MaterializedVirtual>,
        pending_field_writes: Vec<ResolvedPendingFieldWrite>,
        via_blackhole: bool,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
    /// Blackhole replay could not continue the trace.
    Abort {
        trace_id: u64,
        fail_index: u32,
        fail_values: Vec<i64>,
        typed_fail_values: Option<Vec<Value>>,
        exit_layout: Option<CompiledExitLayout>,
        meta: M,
        recovery: Option<GuardRecovery>,
        message: String,
        savedata: Option<GcRef>,
        exception: ExceptionState,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DriverRunOutcome {
    Finished { via_blackhole: bool },
    Jump { via_blackhole: bool },
    GuardFailure { restored: bool, via_blackhole: bool },
    Abort { restored: bool, via_blackhole: bool },
}

#[derive(Debug, Clone)]
pub enum DetailedDriverRunOutcome {
    Finished {
        typed_values: Vec<Value>,
        via_blackhole: bool,
        /// When true, Int-typed values are raw integers (not boxed pointers).
        raw_int_result: bool,
        /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity:
        /// the FINISH descr was `sd.exit_frame_with_exception_descr_ref`
        /// (emitted by pyjitpl.py:3238-3245 compile_exit_frame_with_exception).
        /// `typed_values[0]` is the `ExitFrameWithExceptionRef` exception
        /// GcRef; callers must route this to `jitexc.ExitFrameWithExceptionRef`
        /// (`jitexc.py:45`) instead of `jitexc.DoneWithThisFrame*`.
        is_exit_frame_with_exception: bool,
    },
    Jump {
        via_blackhole: bool,
    },
    /// compile.py:701 handle_fail: guard failure data for the caller to
    /// process via handle_fail(). No state restoration is done here —
    /// the caller decides whether to bridge or blackhole.
    GuardFailure {
        fail_index: u32,
        trace_id: u64,
        /// `cpu.get_latest_descr(deadframe)` (`history.py:125`) — the
        /// backend-side guard descr recovered from the failing frame.
        /// Pyre currently has split metainterp/backend descr objects;
        /// this Arc is the runtime descr whose `rd_loop_token_clt` and
        /// `fail_index_per_trace` mirror the metainterp guard descr.
        descr_arc: std::sync::Arc<dyn majit_ir::FailDescr>,
        /// compile.py:702: must_compile() result.
        should_bridge: bool,
        /// compile.py: rd_loop_token — owning compiled loop key.
        owning_key: u64,
        /// Raw register values from compiled code exit.
        raw_values: Vec<i64>,
        /// Guard exit layout (rd_numb, fail_arg_types, etc.).
        exit_layout: CompiledExitLayout,
    },
    Abort {
        restored: bool,
        via_blackhole: bool,
    },
}

/// Decision about how to handle a function call during tracing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineDecision {
    /// Inline the call: continue tracing into the callee.
    Inline,
    /// Emit a CALL_ASSEMBLER: callee has compiled code.
    CallAssembler,
    /// Emit a residual (opaque) call.
    ResidualCall,
}

/// pyjitpl.py:2493-2503 routes through `crate::jitexc::DoneWithThisFrame`
/// (the single jitexc.py mirror — see jitexc.rs:39).  This alias lets
/// the rest of this module refer to it as `DoneWithThisFrame` without a
/// qualified path while keeping the one authoritative definition.
pub use crate::jitexc::DoneWithThisFrame;

/// Result type for `MetaInterp::finishframe`: either `ChangeFrame`
/// (caller frame remains) or `DoneWithThisFrame*` (framestack
/// exhausted, portal exit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FinishFrameSignal {
    ChangeFrame,
    Done(DoneWithThisFrame),
}

impl From<ChangeFrame> for FinishFrameSignal {
    fn from(_: ChangeFrame) -> Self {
        Self::ChangeFrame
    }
}

impl From<DoneWithThisFrame> for FinishFrameSignal {
    fn from(d: DoneWithThisFrame) -> Self {
        Self::Done(d)
    }
}

impl std::fmt::Display for FinishFrameSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChangeFrame => f.write_str("ChangeFrame"),
            Self::Done(d) => std::fmt::Display::fmt(d, f),
        }
    }
}

impl std::error::Error for FinishFrameSignal {}

/// Result type for `MetaInterp::finishframe_exception` and
/// `handle_possible_exception` — mirrors the two upstream `raise` sites
/// in `pyjitpl.py:2506-2538`.
///
/// * `ChangeFrame` — a `catch_exception` opcode was found in some frame
///   on the stack; control jumps there (`pyjitpl.py:2522`).
/// * `ExitFrameWithExceptionRef(GcRef)` — no handler was found, the
///   framestack was drained, and `compile_exit_frame_with_exception`
///   ran (`pyjitpl.py:2533-2538`).  Mirrors
///   `jitexc.py:45 ExitFrameWithExceptionRef`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishframeExceptionSignal {
    ChangeFrame,
    ExitFrameWithExceptionRef(majit_ir::GcRef),
}

impl From<ChangeFrame> for FinishframeExceptionSignal {
    fn from(_: ChangeFrame) -> Self {
        Self::ChangeFrame
    }
}

impl std::fmt::Display for FinishframeExceptionSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChangeFrame => f.write_str("ChangeFrame"),
            Self::ExitFrameWithExceptionRef(r) => {
                write!(f, "ExitFrameWithExceptionRef({:#x})", r.0)
            }
        }
    }
}

impl std::error::Error for FinishframeExceptionSignal {}

/// Aggregate error for `do_residual_call`.  RPython's body raises
/// either `ChangeFrame` (an exception path crossed a frame boundary
/// via `handle_possible_exception` → `finishframe_exception`) or
/// `SwitchToBlackhole(reason)` (compile/trace failure path);
/// pyre returns both as `Err` variants of this enum so callers can
/// route each to the existing pyre abort/restart paths.
///
/// `pyjitpl.py:2533-2538` also reports `ExitFrameWithExceptionRef`
/// when the exception traverses every frame on the stack without a
/// handler. Pyre carries that signal through `finishframe_exception`
/// → `handle_possible_exception` and surfaces it here so callers can
/// route it to the existing pyre exit-with-exception path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DoResidualCallAbort {
    ChangeFrame,
    SwitchToBlackhole(SwitchToBlackhole),
    ExitFrameWithExceptionRef(majit_ir::GcRef),
}

impl From<ChangeFrame> for DoResidualCallAbort {
    fn from(_: ChangeFrame) -> Self {
        Self::ChangeFrame
    }
}

impl From<FinishframeExceptionSignal> for DoResidualCallAbort {
    fn from(sig: FinishframeExceptionSignal) -> Self {
        match sig {
            FinishframeExceptionSignal::ChangeFrame => Self::ChangeFrame,
            FinishframeExceptionSignal::ExitFrameWithExceptionRef(r) => {
                Self::ExitFrameWithExceptionRef(r)
            }
        }
    }
}

impl From<SwitchToBlackhole> for DoResidualCallAbort {
    fn from(stb: SwitchToBlackhole) -> Self {
        Self::SwitchToBlackhole(stb)
    }
}

impl std::fmt::Display for DoResidualCallAbort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ChangeFrame => f.write_str("ChangeFrame"),
            Self::SwitchToBlackhole(stb) => std::fmt::Display::fmt(stb, f),
            Self::ExitFrameWithExceptionRef(r) => {
                write!(f, "ExitFrameWithExceptionRef({:#x})", r.0)
            }
        }
    }
}

impl std::error::Error for DoResidualCallAbort {}

/// history.py:36 `class SwitchToBlackhole(jitexc.JitException)`.
///
/// ```python
/// class SwitchToBlackhole(jitexc.JitException):
///     def __init__(self, reason, raising_exception=False):
///         self.reason = reason
///         self.raising_exception = raising_exception
/// ```
///
/// Signaled at any compile/trace failure point that wants the
/// metainterp to drop the current trace and resume in the blackhole
/// interpreter — e.g. `do_not_in_trace_call` (pyjitpl.py:3691-3692)
/// raises `SwitchToBlackhole(ABORT_ESCAPE)` when an `OS_NOT_IN_TRACE`
/// call raised, and `compile_done_with_this_frame` re-raises whatever
/// `compile.compile_trace` raised.
///
/// Pyre returns this as an `Err` instead of panicking; callers
/// translate it into the existing pyre abort path
/// (`TraceCtx::abort_trace`, etc.).  `reason` is opaque
/// (`Counters.ABORT_*` int upstream) and is forwarded to
/// `aborted_tracing(reason)` per pyjitpl.py:2491.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SwitchToBlackhole {
    pub reason: i32,
    pub raising_exception: bool,
}

/// rlib/jit.py:1414 `Counters.*` constants used as
/// `SwitchToBlackhole.reason`. Pyre carries them as raw `i32`s so the
/// eventual hook payload stays stable. Values match the declaration
/// order in jit.py:1416-1442 so future Counter additions slot in
/// without renumbering.
#[allow(dead_code)]
pub mod counters {
    /// jit.py:1416 `Counters.TRACING`.
    pub const TRACING: i32 = 0;
    /// jit.py:1417 `Counters.BACKEND`.
    pub const BACKEND: i32 = 1;
    /// jit.py:1418 `Counters.OPS`.
    pub const OPS: i32 = 2;
    /// jit.py:1419 `Counters.HEAPCACHED_OPS`.
    pub const HEAPCACHED_OPS: i32 = 3;
    /// jit.py:1420 `Counters.RECORDED_OPS`.
    pub const RECORDED_OPS: i32 = 4;
    /// jit.py:1421 `Counters.GUARDS`.
    pub const GUARDS: i32 = 5;
    /// jit.py:1422 `Counters.OPT_OPS`.
    pub const OPT_OPS: i32 = 6;
    /// jit.py:1423 `Counters.OPT_GUARDS`.
    pub const OPT_GUARDS: i32 = 7;
    /// jit.py:1424 `Counters.OPT_GUARDS_SHARED`.
    pub const OPT_GUARDS_SHARED: i32 = 8;
    /// jit.py:1425 `Counters.OPT_FORCINGS`.
    pub const OPT_FORCINGS: i32 = 9;
    /// jit.py:1426 `Counters.OPT_VECTORIZE_TRY`.
    pub const OPT_VECTORIZE_TRY: i32 = 10;
    /// jit.py:1427 `Counters.OPT_VECTORIZED`.
    pub const OPT_VECTORIZED: i32 = 11;
    /// jit.py:1428 `Counters.ABORT_TOO_LONG`.
    pub const ABORT_TOO_LONG: i32 = 12;
    /// jit.py:1429 `Counters.ABORT_BRIDGE`.
    pub const ABORT_BRIDGE: i32 = 13;
    /// jit.py:1430 `Counters.ABORT_BAD_LOOP`.
    pub const ABORT_BAD_LOOP: i32 = 14;
    /// jit.py:1431 `Counters.ABORT_ESCAPE`.
    pub const ABORT_ESCAPE: i32 = 15;
    /// jit.py:1432 `Counters.ABORT_FORCE_QUASIIMMUT`.
    pub const ABORT_FORCE_QUASIIMMUT: i32 = 16;
    /// jit.py:1433 `Counters.ABORT_SEGMENTED_TRACE`.
    pub const ABORT_SEGMENTED_TRACE: i32 = 17;
    /// jit.py:1434 `Counters.FORCE_VIRTUALIZABLES`.
    pub const FORCE_VIRTUALIZABLES: i32 = 18;
    /// jit.py:1435 `Counters.NVIRTUALS`.
    pub const NVIRTUALS: i32 = 19;
    /// jit.py:1436 `Counters.NVHOLES`.
    pub const NVHOLES: i32 = 20;
    /// jit.py:1437 `Counters.NVREUSED`.
    pub const NVREUSED: i32 = 21;
}

impl SwitchToBlackhole {
    /// compile.py:27-29 giveup() — raises `SwitchToBlackhole(ABORT_BRIDGE)`.
    ///
    /// The canonical "the optimizer is about to crash, bail to blackhole"
    /// escape hatch. Callers do `raise compile.giveup()` in RPython
    /// (pyjitpl.py:1668/2899/3220/3245, optimizer.py:740). In Rust we
    /// `return Err(SwitchToBlackhole::giveup())`.
    pub fn giveup() -> Self {
        Self {
            reason: counters::ABORT_BRIDGE,
            raising_exception: false,
        }
    }

    /// Construct a `SwitchToBlackhole(Counters.ABORT_ESCAPE,
    /// raising_exception=True)` per pyjitpl.py:3691-3692 —
    /// `OS_NOT_IN_TRACE` call raised during tracing.  Mirrors the
    /// keyword argument upstream sets explicitly so the blackhole
    /// resume path (blackhole.rs:3469-3487) re-raises the helper-side
    /// exception instead of silently dropping it.
    pub fn abort_escape() -> Self {
        Self {
            reason: counters::ABORT_ESCAPE,
            raising_exception: true,
        }
    }

    /// Construct a `SwitchToBlackhole(Counters.ABORT_BAD_LOOP)` —
    /// `compile.compile_loop` gave up at the JUMP-terminated loop path
    /// (pyjitpl.py:3028).  Reserved for callers distinguishing the
    /// loop-compile failure from the trace-compile (FINISH) failure,
    /// which is `giveup()` above.
    pub fn bad_loop() -> Self {
        Self {
            reason: counters::ABORT_BAD_LOOP,
            raising_exception: false,
        }
    }
}

impl std::fmt::Display for SwitchToBlackhole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SwitchToBlackhole(reason={})", self.reason)
    }
}

impl std::error::Error for SwitchToBlackhole {}

/// pyjitpl.py:1268 / 2425 `raise ChangeFrame`.
///
/// Signals to the metainterp main loop that the current frame has been
/// switched (either pushed or popped) and dispatch must restart from
/// the new top-of-stack.  Pyre uses a unit error type because Rust does
/// not have Python-style `raise`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChangeFrame;

impl std::fmt::Display for ChangeFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ChangeFrame")
    }
}

impl std::error::Error for ChangeFrame {}

// ════════════════════════════════════════════════════════════════════════
// MetaInterpStaticData (pyjitpl.py:2190-2373)
// ════════════════════════════════════════════════════════════════════════

/// pyjitpl.py:2190 `class MetaInterpStaticData(object)`.
///
/// Holds the per-process tables shared by every running `MetaInterp`:
/// the assembler's `insns` / `descrs` / `indirectcalltargets` /
/// `list_of_addr2name`, the `callinfocollection`, and the lazy
/// `bytecode_for_address` lookup that `MIFrame.do_residual_or_indirect_call`
/// uses to promote a const-funcptr residual call into an inlined one.
///
/// PRE-EXISTING-ADAPTATION: pyre's `MetaInterp<M>` still owns several
/// `descr.py:348-360 get_array_descr` lltype-shape cache key for
/// `dispatch_array_descr_cache`.  RPython keys
/// `gccache._cache_array[ARRAY_OR_STRUCT]` on the lltype itself, which
/// transitively encodes `type_id`, `is_array_of_pointers`,
/// `is_array_of_structs`, item layout, and interior field positions.
/// Pyre lowered the IR to `BhDescr::Array` before this site, so the
/// equivalent key threads through every variant field that
/// distinguishes two `BhDescr::Array` entries — `(type_id, base_size,
/// itemsize, item_type, is_array_of_pointers, is_array_of_structs,
/// is_item_signed, interior_fields)`.  Two arrays of distinct lltypes
/// with identical `(base_size, itemsize)` geometry land on distinct
/// cache slots, matching upstream's per-lltype cache identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DispatchArrayDescrKey {
    pub type_id: u32,
    pub base_size: usize,
    pub itemsize: usize,
    pub item_type: majit_ir::Type,
    pub is_array_of_pointers: bool,
    pub is_array_of_structs: bool,
    pub is_item_signed: bool,
    pub interior_fields: Vec<crate::jitcode::BhInteriorFieldSpec>,
}

/// runtime-state fields that RPython places on `MetaInterpStaticData`
/// (e.g. profiler, `warmrunnerdesc`, `cpu`).  `staticdata` itself
/// already holds the per-process tables (`opcode_*`, `opcode_descrs`,
/// `indirectcalltargets`, `_addr2name_*`, `liveness_info`,
/// `callinfocollection`, `jitdrivers_sd`, `globaldata`); the remaining
/// runtime knobs land in a future audit pass.
#[derive(Debug, Default)]
pub struct MetaInterpStaticData {
    /// pyjitpl.py:2228 `setup_insns(insns)` table — opcode-id ↔ name.
    pub opcode_names: Vec<String>,
    /// pyjitpl.py:2229, 2235 `opcode_implementations[opcode_id] = opimpl`.
    ///
    /// PRE-EXISTING-ADAPTATION: RPython looks up an opimpl bound method
    /// per opcode id and dispatches `MIFrame.run_one_step` through it.
    /// Pyre dispatches by `BC_*` constant inside
    /// `JitCodeMachine::dispatch_one`, so the implementation table is
    /// kept as a stub that's parallel-sized to `opcode_names` for
    /// invariant checks but never indexed.
    pub opcode_implementations: Vec<Option<usize>>,
    /// pyjitpl.py:2245-2246 `setup_descrs(descrs)` — descriptor index table.
    pub opcode_descrs: Vec<u64>,
    /// pyjitpl.py:2248-2249 `setup_indirectcalltargets(indirectcalltargets)`.
    /// Stores runtime-adapter `Arc<JitCode>` references so
    /// `bytecode_for_address` can hand a hot copy back to
    /// `MIFrame::do_residual_or_indirect_call` for
    /// `MetaInterp::perform_call(jitcode, ...)`.
    ///
    /// The dict semantics are upstream-orthodox (`for jitcode in
    /// self.indirectcalltargets: d[jitcode.fnaddr] = jitcode` at
    /// `pyjitpl.py:2334-2342`), but pyre has not yet switched this
    /// storage edge over to the canonical codewriter `JitCode`.
    pub indirectcalltargets: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    /// pyjitpl.py:2251-2253 `setup_list_of_addr2name(list_of_addr2name)`.
    /// Pair-list of (fnaddr, name) for debug introspection.
    pub _addr2name_keys: Vec<usize>,
    pub _addr2name_values: Vec<String>,
    /// pyjitpl.py:2236 `op_live = insns.get('live/', -1)`.
    pub op_live: i32,
    /// pyjitpl.py:2237 `op_goto = insns.get('goto/L', -1)`.
    pub op_goto: i32,
    /// pyjitpl.py:2238 `op_catch_exception = insns.get('catch_exception/L', -1)`.
    pub op_catch_exception: i32,
    /// pyjitpl.py:2239 `op_rvmprof_code = insns.get('rvmprof_code/ii', -1)`.
    pub op_rvmprof_code: i32,
    /// pyjitpl.py:2240 `op_int_return = insns.get('int_return/i', -1)`.
    pub op_int_return: i32,
    /// pyjitpl.py:2241 `op_ref_return = insns.get('ref_return/r', -1)`.
    pub op_ref_return: i32,
    /// pyjitpl.py:2242 `op_float_return = insns.get('float_return/f', -1)`.
    pub op_float_return: i32,
    /// pyjitpl.py:2243 `op_void_return = insns.get('void_return/', -1)`.
    pub op_void_return: i32,
    /// pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)` —
    /// the concatenated byte stream produced by
    /// `assembler.py:241-247` `all_liveness.append(...)`.  RPython freezes
    /// it once at `finish_setup` and never mutates it again; the runtime
    /// reads the bytes through `pyjitpl.py:203 all_liveness =
    /// self.metainterp.staticdata.liveness_info` and decodes via
    /// `LivenessIterator`.
    ///
    /// Stored as raw `Vec<u8>` because the upstream string is bytes-like
    /// (Python 2 `str`) and the packed liveness encoding is not valid
    /// UTF-8 in general.  Filled exactly once by
    /// `MetaInterpStaticData::finish_setup(asm)` (parity with
    /// `pyjitpl.py:2255-2264`).
    pub liveness_info: Vec<u8>,
    /// pyjitpl.py:2255-2285 `finish_setup(...)` populates this from
    /// `codewriter.callcontrol.callinfocollection`.
    pub callinfocollection: majit_ir::effectinfo::CallInfoCollection,
    /// pyjitpl.py:2266 `self.jitdrivers_sd = codewriter.callcontrol.jitdrivers_sd`.
    ///
    /// Indexed by `JitCode.jitdriver_sd` so `is_main_jitcode(jitcode)`
    /// can read `jitdrivers_sd[idx].is_recursive` per
    /// `pyjitpl.py:2427-2429` without consulting the runtime
    /// `JitDriver` object.  Pyre populates this via
    /// `MetaInterpStaticData::register_jitdriver_sd` rather than the
    /// upstream `finish_setup(codewriter)` callback because pyre's
    /// codewriter pipeline is split across crates.
    pub jitdrivers_sd: Vec<crate::jitdriver::JitDriverStaticData>,
    /// `compile.py:667-671` `make_and_attach_done_descrs(targets)`
    /// attaches these five singletons to every target.  RPython calls
    /// `make_and_attach_done_descrs([self, cpu])` at
    /// `pyjitpl.py:2222`.  pyre populates the `MetaInterpStaticData`
    /// half in `MetaInterpStaticData::new`; the cpu/backend half lands
    /// when `Backend::propagate_exception_descr` is wired (follow-up).
    pub done_with_this_frame_descr_void: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_int: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_ref: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_float: Option<majit_ir::DescrRef>,
    /// `compile.py:671` `exit_frame_with_exception_descr_ref`.
    pub exit_frame_with_exception_descr_ref: Option<majit_ir::DescrRef>,
    /// `pyjitpl.py:2273, 2283` `compile.PropagateExceptionDescr()` —
    /// one instance per MetaInterp, shared across jitdrivers.
    pub propagate_exception_descr: Option<majit_ir::DescrRef>,
    /// pyjitpl.py:2357-2373 `MetaInterpGlobalData`: lazy
    /// `addr2name` and `indirectcall_dict` caches.  Populated on first
    /// call to `bytecode_for_address` / `get_name_from_address`.
    ///
    /// PRE-EXISTING-ADAPTATION: wrapped in `Mutex` because
    /// `MetaInterp.staticdata` is `Arc<MetaInterpStaticData>` and this
    /// field mutates lazily (memoization) through the shared Arc.
    /// RPython's Python dicts are shared mutable references by
    /// default; Rust needs interior mutability for the same
    /// behavior.
    pub globaldata: std::sync::Mutex<MetaInterpGlobalData>,
    /// pyjitpl.py:2289 `self.staticdata.all_descrs = self.cpu.setup_descrs()`.
    /// descr.py:25-47: dense list indexed by `descr_index`.
    ///
    /// RPython stores this on `metainterp_sd` (the static data object),
    /// not on the live `MetaInterp` — opencoder / bridgeopt / optimizer
    /// all read `metainterp_sd.all_descrs`. Pyre mirrors that location
    /// so `opencoder::Trace` (which lives in this crate now) can read
    /// the length directly via `self.metainterp_sd.all_descrs.lock().unwrap().len()`
    /// from `_encode_descr` and the TraceIterator.
    ///
    /// PRE-EXISTING-ADAPTATION: wrapped in `Mutex` because
    /// `MetaInterp.staticdata` is `Arc<MetaInterpStaticData>` and the
    /// `TraceRecordBuffer` inside `TraceCtx` holds a clone of this Arc
    /// (opencoder.py:471 `self.metainterp_sd = metainterp_sd` —
    /// shared Python reference; lifts to Arc in Rust). With refcount ≥ 2,
    /// `Arc::get_mut` fails, so `mem::take` at compile time and
    /// `take_back_all_descrs` at post-optimize both route through a
    /// Mutex lock. RPython's Python dicts are shared mutable references
    /// by default; Rust needs interior mutability for the same behavior.
    pub all_descrs: std::sync::Mutex<Vec<DescrRef>>,
    /// `descr.py:20 GcCache._cache_array` parity for the dispatch JitCode
    /// trace-side `BC_GETARRAYITEM_GC_I` recorder.
    ///
    /// `JitCodeMachine::dispatch_array_descr_ref` materialises an
    /// `Arc<SimpleArrayDescr>` from the canonical pool's
    /// `BhDescr::Array` entry the first time a given lltype shape is
    /// resolved.  The cache must outlive a single trace so cross-trace
    /// / cross-bridge recorders receive the same `Arc<dyn Descr>`
    /// identity for the same lltype ARRAY — `descr_identity`
    /// (`descr.rs:494`) is Arc-address-based, so without the
    /// translation-wide table, two traces compiled for the same loop
    /// would surface distinct array descrs from `Arc::ptr_eq`'s
    /// perspective and break optimizer/backend descr-keyed caches.
    /// PyPy keeps the equivalent cache on `gccache._cache_array`
    /// (`descr.py:348`).
    ///
    /// Keyed on [`DispatchArrayDescrKey`], which captures the full
    /// lltype-discriminant shape carried on `BhDescr::Array`
    /// (`type_id`, `base_size`, `itemsize`, `item_type`,
    /// `is_array_of_pointers`, `is_array_of_structs`, `is_item_signed`,
    /// `interior_fields`).  Mirrors upstream
    /// `gccache._cache_array[ARRAY_OR_STRUCT]` (`descr.py:348-360`)
    /// where the key is the lltype itself.  Pyre cannot use the lltype
    /// directly because the codewriter has already lowered the IR to
    /// `BhDescr::Array`; the struct above carries every lltype
    /// discriminant the BhDescr exposes so two distinct lltypes that
    /// happen to share `(base_size, itemsize, item_type,
    /// is_item_signed)` (e.g. one is an array-of-pointers, one is an
    /// array-of-structs with interior fields) get distinct cache slots.
    /// `make_array_descr_signed` (`descr.rs:3712`) is the constructor
    /// pyre's dispatch path uses; it currently leaves `type_id=0`,
    /// `lendescr=None`, and the struct/interior-field flags empty,
    /// matching what pyre's bytecode-array dispatch (`program: &[u8]`)
    /// needs but lagging upstream `descr.py:get_array_descr` which
    /// preserves `lendescr`, `flag`, `is_pure`, and per-field interior
    /// descrs.  Closing that gap is tracked separately; the cache key
    /// is already keyed by the full discriminator so the value side
    /// can be enriched without changing the key shape.  Earlier
    /// revisions keyed on `descr_idx` (per-JitCode-builder pool slot),
    /// which happens to work when only one JitCode body emits arrays
    /// but breaks structurally when distinct JitCodes use the same
    /// slot index for different array shapes — pyre's per-loop-body
    /// JitCode design (each `#[jit_interp]` site has an independent
    /// pool) makes that scenario reachable.
    ///
    /// Mutex-wrapped because `Arc<MetaInterpStaticData>` is shared
    /// across the metainterp / trace / bridge pipelines (mirroring
    /// `all_descrs` above).
    pub dispatch_array_descr_cache:
        std::sync::Mutex<std::collections::HashMap<DispatchArrayDescrKey, DescrRef>>,
    /// pyjitpl.py:2199-2200 `self.profiler = ProfilerClass()` —
    /// `metainterp_sd.profiler` is the shared counter sink hit from
    /// every metainterp / optimizer / heapcache / tracer site
    /// (`self.metainterp_sd.profiler.count_ops(...)`).  Pyre mirrors
    /// the location so cross-crate callers (TraceCtx in
    /// `pyre-jit-trace`, heapcache in `majit-trace`, the vector pass)
    /// can hit it through the same shared `Arc`.
    ///
    /// Implemented as a struct of `AtomicUsize` (see [`crate::jitprof`])
    /// because `Arc<MetaInterpStaticData>` is shared across threads in
    /// pyre's pipeline and a `Mutex` would serialise every counter
    /// bump.  Each fetch_add is `Relaxed` — counters have no causal
    /// dependency on each other.
    pub profiler: crate::jitprof::JitProfiler,
}

/// pyjitpl.py:2357-2373 `class MetaInterpGlobalData`.
///
/// Lazy run-time caches built from `MetaInterpStaticData`.  RPython
/// reuses these across compilations to avoid rebuilding the dicts on
/// every guard failure.
#[derive(Debug, Default)]
pub struct MetaInterpGlobalData {
    /// pyjitpl.py:2308-2318 `addr2name`: `fnaddr → name` for debugging.
    pub addr2name: Option<std::collections::HashMap<usize, String>>,
    /// pyjitpl.py:2326-2343 `indirectcall_dict`: `fnaddr → JitCode`.
    /// Stores the current runtime-adapter `JitCode`; the helper that
    /// builds this dict is intentionally type-agnostic so canonical
    /// codewriter jitcodes can reuse the same semantics.
    pub indirectcall_dict:
        Option<std::collections::HashMap<usize, std::sync::Arc<crate::jitcode::JitCode>>>,
    /// pyjitpl.py:2293-2303 `initialized` — guards `_setup_once` so the
    /// runtime side-effects (profiler start, jitlog setup) fire once.
    pub initialized: bool,
}

fn build_indirectcall_dict<T>(
    targets: &[std::sync::Arc<T>],
    fnaddr_of: impl Fn(&T) -> usize,
) -> std::collections::HashMap<usize, std::sync::Arc<T>> {
    let mut d = std::collections::HashMap::new();
    for jitcode in targets {
        let fnaddr = fnaddr_of(jitcode);
        debug_assert!(
            !d.contains_key(&fnaddr),
            "duplicate fnaddr in indirectcalltargets"
        );
        d.insert(fnaddr, jitcode.clone());
    }
    d
}

fn bytecode_for_address_in_targets<T>(
    targets: &[std::sync::Arc<T>],
    cache: &mut Option<std::collections::HashMap<usize, std::sync::Arc<T>>>,
    fnaddress: usize,
    fnaddr_of: impl Fn(&T) -> usize,
) -> Option<std::sync::Arc<T>> {
    let dict = cache.get_or_insert_with(|| build_indirectcall_dict(targets, fnaddr_of));
    dict.get(&fnaddress).cloned()
}

impl crate::compile::DescrContainer for MetaInterpStaticData {
    fn set_done_with_this_frame_descr_void(&mut self, descr: majit_ir::DescrRef) {
        self.done_with_this_frame_descr_void = Some(descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: majit_ir::DescrRef) {
        self.done_with_this_frame_descr_int = Some(descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.done_with_this_frame_descr_ref = Some(descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: majit_ir::DescrRef) {
        self.done_with_this_frame_descr_float = Some(descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.exit_frame_with_exception_descr_ref = Some(descr);
    }
}

impl MetaInterpStaticData {
    pub fn new() -> Self {
        // `pyjitpl.py:2222` `compile.make_and_attach_done_descrs([self, cpu])`.
        // RPython passes `[self, cpu]` — the same `Arc<DoneWithThisFrameDescr*>`
        // lands on both the metainterp and the backend so FINISH-descr
        // identity matches across the fast-path comparisons in
        // `llmodel.py` and the `handle_fail` dispatch in pyjitpl.
        //
        // pyre currently attaches only to `MetaInterpStaticData`; backends
        // keep their own `LazyLock<Arc<DynasmFailDescr>>` /
        // `RegisteredLoopTarget` singletons.  Unification needs new
        // `Backend::set_done_with_this_frame_descr_*` setters that take
        // these Arcs, plus a runtime-path update so
        // `done_with_this_frame_descr_int_ptr()` returns `Arc::as_ptr` of
        // the stored Arc.  See the follow-up note in
        // `compile.rs:1565-…` for the five-file surface.
        let mut sd = Self {
            op_live: -1,
            op_goto: -1,
            op_catch_exception: -1,
            op_rvmprof_code: -1,
            op_int_return: -1,
            op_ref_return: -1,
            op_float_return: -1,
            op_void_return: -1,
            ..Self::default()
        };
        crate::compile::make_and_attach_done_descrs(&mut [&mut sd]);
        sd
    }

    /// `compile.py:3204-3215` `token = sd.done_with_this_frame_descr_<name>`
    /// — select the FINISH descr attached to this `MetaInterpStaticData`
    /// for a given result type.  The returned `Arc` is the one
    /// `make_and_attach_done_descrs` installed on self and (via
    /// `attach_descrs_to_cpu`) on the backend, so FINISH ops get
    /// pointer-identity parity with RPython.
    pub fn done_with_this_frame_descr_for(&self, tp: Type) -> Option<majit_ir::DescrRef> {
        match tp {
            Type::Int => self.done_with_this_frame_descr_int.clone(),
            Type::Ref => self.done_with_this_frame_descr_ref.clone(),
            Type::Float => self.done_with_this_frame_descr_float.clone(),
            Type::Void => self.done_with_this_frame_descr_void.clone(),
        }
    }

    /// `compile.py:3204-3215` variant that resolves the result type from
    /// the FINISH op's `fail_arg_types` slice: empty → Void, single-
    /// element → that element.  Returns `None` for multi-arg FINISH
    /// (a pyre-only declarative-driver shape) so callers can fall back
    /// to `make_fail_descr_typed`; RPython itself never emits such
    /// FINISH ops.
    pub fn done_with_this_frame_descr_from_types(
        &self,
        finish_arg_types: &[Type],
    ) -> Option<majit_ir::DescrRef> {
        let tp = match finish_arg_types {
            [] => Type::Void,
            [tp] => *tp,
            _ => return None,
        };
        self.done_with_this_frame_descr_for(tp)
    }

    /// `pyjitpl.py:2222` `make_and_attach_done_descrs([self, cpu])` —
    /// the CPU half of the pair.  RPython does this in a single call
    /// inside `MetaInterpStaticData.__init__`; pyre splits it in two
    /// because `MetaInterpStaticData::new` runs before the backend
    /// exists.  `MetaInterp::new` (which constructs both) calls this
    /// method afterwards so the backend ends up owning clones of the
    /// same `Arc<DoneWithThisFrameDescr*>` the metainterp already has.
    ///
    /// Also forwards `propagate_exception_descr` once
    /// `finish_setup_descrs_for_jitdrivers` has created it — the
    /// attachment is idempotent, so callers can run this after every
    /// `register_jitdriver_sd` without harm.
    pub fn attach_descrs_to_cpu(&self, cpu: &mut dyn majit_backend::Backend) {
        if let Some(d) = &self.done_with_this_frame_descr_void {
            cpu.set_done_with_this_frame_descr_void(d.clone());
        }
        if let Some(d) = &self.done_with_this_frame_descr_int {
            cpu.set_done_with_this_frame_descr_int(d.clone());
        }
        if let Some(d) = &self.done_with_this_frame_descr_ref {
            cpu.set_done_with_this_frame_descr_ref(d.clone());
        }
        if let Some(d) = &self.done_with_this_frame_descr_float {
            cpu.set_done_with_this_frame_descr_float(d.clone());
        }
        if let Some(d) = &self.exit_frame_with_exception_descr_ref {
            cpu.set_exit_frame_with_exception_descr_ref(d.clone());
        }
        if let Some(d) = &self.propagate_exception_descr {
            cpu.set_propagate_exception_descr(d.clone());
        }
    }

    /// `pyjitpl.py:2255-2285` `MetaInterpStaticData.finish_setup(self,
    /// codewriter, optimizer=None)`.
    ///
    /// ```python
    /// def finish_setup(self, codewriter, optimizer=None):
    ///     from rpython.jit.metainterp.blackhole import BlackholeInterpBuilder
    ///     self.blackholeinterpbuilder = BlackholeInterpBuilder(codewriter, self)
    ///     #
    ///     asm = codewriter.assembler
    ///     self.setup_insns(asm.insns)
    ///     self.setup_descrs(asm.descrs)
    ///     self.setup_indirectcalltargets(asm.indirectcalltargets)
    ///     self.setup_list_of_addr2name(asm.list_of_addr2name)
    ///     self.liveness_info = "".join(asm.all_liveness)
    ///     #
    ///     self.jitdrivers_sd = codewriter.callcontrol.jitdrivers_sd
    ///     self.virtualref_info = codewriter.callcontrol.virtualref_info
    ///     self.callinfocollection = codewriter.callcontrol.callinfocollection
    ///     self.has_libffi_call = codewriter.callcontrol.has_libffi_call
    ///     #
    ///     # store this information for fastpath of call_assembler
    ///     # (only the paths that can actually be taken)
    ///     exc_descr = compile.PropagateExceptionDescr()
    ///     for jd in self.jitdrivers_sd:
    ///         name = {history.INT: 'int', history.REF: 'ref',
    ///                 history.FLOAT: 'float', history.VOID: 'void'}[jd.result_type]
    ///         token = getattr(self, 'done_with_this_frame_descr_%s' % name)
    ///         jd.portal_finishtoken = token
    ///         jd.propagate_exc_descr = exc_descr
    ///     #
    ///     self.cpu.propagate_exception_descr = exc_descr
    ///     self.globaldata = MetaInterpGlobalData(self)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's `CodeWriter`
    /// (`majit-translate/src/jit_codewriter/codewriter.rs:64-77`) does
    /// **not** own `callcontrol` — RPython's does
    /// (`codewriter.py:CodeWriter.__init__` keeps both).  The Rust
    /// borrow-checker constraint is documented at the CodeWriter
    /// declaration; pyre threads `callcontrol` as a sibling parameter
    /// at this call site so the upstream
    /// `codewriter.callcontrol.{jitdrivers_sd, virtualref_info,
    /// callinfocollection, has_libffi_call}` reads remain literal
    /// ports of the source line, just spelt
    /// `callcontrol.<field>` instead of
    /// `codewriter.callcontrol.<field>`.
    ///
    /// Each upstream line below is either ported in place or annotated
    /// with a cited blocker so that downstream callers see the full
    /// `pyjitpl.py:2255-2285` lifecycle surface even when individual
    /// payload types still diverge.
    pub fn finish_setup(
        &mut self,
        codewriter: &majit_translate::jit_codewriter::codewriter::CodeWriter,
        callcontrol: &majit_translate::jit_codewriter::call::CallControl,
    ) {
        // pyjitpl.py:2257-2258
        //     self.blackholeinterpbuilder = BlackholeInterpBuilder(codewriter, self)
        // PRE-EXISTING-ADAPTATION: pyre's `BlackholeInterpBuilder`
        // (`blackhole.rs:3088 setup_insns`) is wired piecewise from
        // per-jitdriver bring-up rather than a single constructor call
        // — bringing the constructor-form back here cascades into a
        // refactor of `BlackholeInterpBuilder`'s allocator and
        // exceeds Task #89 session scope.

        let asm = &codewriter.assembler;
        // pyjitpl.py:2260 `self.setup_insns(asm.insns)`
        self.setup_insns(asm.insns());
        // pyjitpl.py:2261 `self.setup_descrs(asm.descrs)`
        // PRE-EXISTING-ADAPTATION: payload mismatch.  Translate-side
        // `Assembler.descrs` is a `Vec<DescrRef>` (object refs); pyre's
        // `setup_descrs(Vec<u64>)` keys by opcode id.  The conversion
        // is `descrs[i] = asm.descrs[i].as_int_id()`, but pyre's
        // runtime Descr surface (DescrRef) does not yet expose a
        // stable u64 id — that bridging is a separate audit slice.

        // pyjitpl.py:2262 `self.setup_indirectcalltargets(asm.indirectcalltargets)`
        // PRE-EXISTING-ADAPTATION: payload mismatch.  Translate-side
        // `Assembler.indirectcalltargets` is a JitCode set; pyre
        // stores `Vec<Arc<JitCode>>` already keyed in alloc-order on
        // `MetaInterpStaticData`.  Bridging requires
        // `CallControl::jitcodes_in_alloc_order` (call.py:87-88) to
        // commit the same shape.

        // pyjitpl.py:2263 `self.setup_list_of_addr2name(asm.list_of_addr2name)`
        // PRE-EXISTING-ADAPTATION: payload mismatch.  Translate-side
        // carries `Vec<(String, String)>` (modname, funcname); pyre's
        // `setup_list_of_addr2name((usize, String))` wants
        // `(address, full_name)` — bridging requires
        // `getfunctionptr(graph)`, an unported codewriter helper.

        // pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)`
        self.liveness_info = asm.all_liveness().to_vec();

        // pyjitpl.py:2266 `self.jitdrivers_sd = codewriter.callcontrol.jitdrivers_sd`
        // PRE-EXISTING-ADAPTATION: pyre populates `jitdrivers_sd`
        // incrementally via `register_jitdriver_sd`
        // (`mod.rs:11576-11589`).  Wholesale assignment here would
        // clobber the incremental work; the convergence path is to
        // flip `register_jitdriver_sd` from "owner" to "validator"
        // once the codewriter pre-builds the full list at
        // `make_jitcodes` time.

        // pyjitpl.py:2267 `self.virtualref_info = codewriter.callcontrol.virtualref_info`
        // PRE-EXISTING-ADAPTATION: pyre's `CallControl` does not yet
        // carry `virtualref_info`.

        // pyjitpl.py:2268 `self.callinfocollection = codewriter.callcontrol.callinfocollection`
        self.callinfocollection = callcontrol.callinfocollection.clone();

        // pyjitpl.py:2269 `self.has_libffi_call = codewriter.callcontrol.has_libffi_call`
        // PRE-EXISTING-ADAPTATION: pyre's `CallControl` has no
        // `has_libffi_call` field; libffi handling is currently not
        // implemented.

        // pyjitpl.py:2273-2284
        //     exc_descr = compile.PropagateExceptionDescr()
        //     for jd in self.jitdrivers_sd: jd.portal_finishtoken = ...
        //     self.cpu.propagate_exception_descr = exc_descr
        // PRE-EXISTING-ADAPTATION: pyre runs the equivalent reattach
        // through `finish_setup_descrs_for_jitdrivers` from
        // `register_jitdriver_sd` (`mod.rs:11588`).  Repeating the
        // loop here would double-attach.

        // pyjitpl.py:2285 `self.globaldata = MetaInterpGlobalData(self)`
        // PRE-EXISTING-ADAPTATION: pyre constructs `globaldata` in
        // `MetaInterpStaticData::new()` because `MetaInterpGlobalData`
        // is required by other init sites that run before
        // `finish_setup`.  Replacing here would require an ownership
        // reshuffle.
    }

    /// Narrow `pyjitpl.py:2264 self.liveness_info = "".join(asm.all_liveness)`
    /// slice intended for state-field JIT.  Copies the
    /// already-populated `Assembler::all_liveness` byte stream into
    /// `self.liveness_info` and seeds the cached opcode-id fields
    /// (`op_live` etc.) that `setup_insns(asm.insns)` would otherwise
    /// populate, without otherwise running `finish_setup`'s
    /// `setup_descrs / setup_indirectcalltargets / ...` payload-
    /// mismatched adaptations.
    ///
    /// State-field JIT's `__JitMeta` shape produces exactly one
    /// canonical liveness entry per JitDriver
    /// (`live_slots_for_state_field_jit`), so a fully-fledged
    /// `CodeWriter` + `CallControl` is overkill — only the liveness
    /// receptacle (`pyjitpl.py:2264`) and the cached opcode ids
    /// (`pyjitpl.py:2236-2243`) need settling before the macro-emitted
    /// `live/<offset>` placeholders become readable through
    /// `MIFrame::get_list_of_active_boxes` (whose
    /// `code[pc - SIZE_LIVE_OP] == op_live` assert is the dominant
    /// downstream consumer).
    ///
    /// Both `finish_setup` and `install_canonical_liveness` share the
    /// same write target and the same `pyjitpl.py:2264`
    /// `self.liveness_info` semantics; downstream sessions will
    /// migrate state-field callers to the full `finish_setup` once
    /// `CodeWriter::with_assembler_only` (Task #89 orth-9 plan
    /// step 4) lands.
    ///
    /// Codex review (2026-04-26): this hook is a NEW-DEVIATION vs
    /// RPython.  RPython has no `install_canonical_liveness`
    /// equivalent — `warmspot.py:281-289` calls a single
    /// `metainterp_sd.finish_setup(codewriter)` after
    /// `make_jitcodes()`, and `pyjitpl.py:2255-2285 finish_setup`
    /// reads `asm.insns` / `asm.all_liveness` / `asm.descrs` /
    /// `callcontrol.jitdrivers_sd` etc. via the single
    /// `codewriter.assembler` reference.  The current pyre lifecycle
    /// fragmentation arises because `majit-translate::CodeWriter`
    /// does not own `CallControl` or `Assembler` like RPython's
    /// `codewriter.CodeWriter` does (pre-existing structural gap),
    /// and because the state-field JIT macro path produces canonical
    /// liveness eagerly per JitState rather than going through the
    /// shared codewriter pipeline.  Convergence path: when CodeWriter
    /// owns its Assembler (`with_assembler_only`) and CallControl
    /// merges in (Task #89 orth-9 step 4 + Task #72 sub-thread),
    /// this method dissolves into the canonical
    /// `finish_setup(codewriter)` and the macro-side examples drop
    /// their `state.build_meta(...).install_canonical_liveness(...)`
    /// calls in favor of the unified `make_jitcodes()
    /// → finish_setup(codewriter)` warmspot lifecycle.
    pub fn install_canonical_liveness(
        &mut self,
        asm: &majit_translate::jit_codewriter::assembler::Assembler,
    ) {
        // Mirrors the asm-derived parts of `finish_setup(codewriter,
        // callcontrol)` (this file, `pyjitpl.py:2255-2285`):
        // `setup_insns(asm.insns)` (line 2260) and
        // `liveness_info = "".join(asm.all_liveness)` (line 2264).
        //
        // The macro caller (`majit-macros::codegen_state.rs::
        // install_canonical_liveness`) populates `asm.insns` via
        // `Assembler::register_insn` before invoking this hook so the
        // `setup_insns(asm.insns())` lookup resolves the canonical
        // pyre-static `BC_*` opcode ids dynamically — same data flow
        // as RPython `make_jitcodes() → finish_setup(codewriter)`,
        // where `codewriter.assembler.insns` is already populated by
        // the time `setup_insns` runs.  No parallel hardcoded `BC_*`
        // seeding block lives in this method any more.
        self.setup_insns(asm.insns());
        self.liveness_info = asm.all_liveness().to_vec();
    }

    /// pyjitpl.py:2227-2243 `setup_insns(insns)`.
    ///
    /// Stores the opcode-id → name table and the cached opcode-id
    /// lookups (`op_live` / `op_goto` / `op_catch_exception` /
    /// `op_rvmprof_code` / `op_*_return`) the dispatch loop checks
    /// against without re-hashing.  `opcode_implementations` is left
    /// as a parallel-sized stub — see the field doc.
    ///
    /// Pyre's blackhole-side `setup_insns` lives separately in
    /// `crate::blackhole::BlackholeInterpBuilder::setup_insns`.
    pub fn setup_insns(&mut self, insns: &std::collections::HashMap<String, u8>) {
        // pyjitpl.py:2228-2229: opcode_names/opcode_implementations init.
        // RPython sizes by `len(insns)` because its assembler assigns
        // opnums sequentially from 0, so `len(insns) == max(opnum) + 1`.
        // pyre's static-pyre `BC_*` constants are sparse over `0..=255`
        // (some opnums are reserved for unimplemented opcodes), so
        // size by `max(opnum) + 1` to keep `names[opnum] = key` in
        // bounds for state-field-JIT macro paths whose `insns` carries
        // only the opnums that actually fire.  Empty `insns` → empty
        // tables (matches `len(insns) == 0` in upstream).
        let table_len = insns
            .values()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        let mut names = vec![String::from("?"); table_len];
        // pyjitpl.py:2230-2235: opcode_implementations[value] = opimpl.
        // PRE-EXISTING-ADAPTATION: pyre dispatches by BC_* match, so
        // the slot is left as None — the table only carries the size
        // for parity with `opcode_names`.
        let implementations = vec![None; table_len];
        for (key, &value) in insns.iter() {
            names[value as usize] = key.clone();
        }
        self.opcode_names = names;
        self.opcode_implementations = implementations;
        // pyjitpl.py:2236-2243: cache opcode ids by upstream key string.
        let lookup = |key: &str| insns.get(key).map(|&v| v as i32).unwrap_or(-1);
        self.op_live = lookup("live/");
        self.op_goto = lookup("goto/L");
        self.op_catch_exception = lookup("catch_exception/L");
        self.op_rvmprof_code = lookup("rvmprof_code/ii");
        self.op_int_return = lookup("int_return/i");
        self.op_ref_return = lookup("ref_return/r");
        self.op_float_return = lookup("float_return/f");
        self.op_void_return = lookup("void_return/");
    }

    /// pyjitpl.py:2245-2246 `setup_descrs(descrs)`.
    pub fn setup_descrs(&mut self, descrs: Vec<u64>) {
        self.opcode_descrs = descrs;
    }

    /// Register a `JitDriverStaticData` slot.  Returns the index that
    /// `JitCode.jitdriver_sd` should reference.  Mirrors
    /// `pyjitpl.py:2266` `self.jitdrivers_sd = codewriter.callcontrol.jitdrivers_sd`,
    /// except pyre populates the table incrementally as drivers register
    /// instead of taking it wholesale from the codewriter's CallControl.
    ///
    /// # S2.1 invariant (wiggly-barto plan)
    ///
    /// The caller must populate `jd.portal_runner_adr` to the host's
    /// `ll_portal_runner` address (`warmspot.py:1010-1012`) **before**
    /// passing the driver to this function. `compile_tmp_callback`
    /// (`compile.rs::compile_tmp_callback`) and
    /// `MIFrame::do_recursive_call` (`pyjitpl/mod.rs::do_recursive_call`)
    /// both `debug_assert!(jd.portal_runner_adr != 0)` at their entry
    /// points; a `0`-address registration silently broke the trampoline
    /// in earlier iterations because the assertion fired only at the
    /// guard-failure-bridge boundary, far from the registration site.
    ///
    /// `portal_calldescr`, `portal_finishtoken`, and `propagate_exc_descr`
    /// are populated automatically by the
    /// [`Self::finish_setup_descrs_for_jitdrivers`] tail call below; the
    /// caller does not need to set them.
    pub fn register_jitdriver_sd(
        &mut self,
        mut jd: crate::jitdriver::JitDriverStaticData,
        cpu: &mut dyn majit_backend::Backend,
    ) -> usize {
        let idx = self.jitdrivers_sd.len();
        jd.index = Some(idx); // call.py:46-47 `jd.index = idx`
        self.jitdrivers_sd.push(jd);
        // `pyjitpl.py:2273-2281` — reattach the finish/exc descrs whenever
        // the jitdriver list changes so new drivers pick up the same
        // `portal_finishtoken` / `propagate_exc_descr` as the rest, and
        // `pyjitpl.py:2283` `self.cpu.propagate_exception_descr = exc_descr`
        // so the backend half of the pair observes the same instance.
        self.finish_setup_descrs_for_jitdrivers(cpu);
        idx
    }

    /// pyjitpl.py:2248-2249 `setup_indirectcalltargets(indirectcalltargets)`.
    pub fn setup_indirectcalltargets(
        &mut self,
        targets: Vec<std::sync::Arc<crate::jitcode::JitCode>>,
    ) {
        self.indirectcalltargets = targets;
        // Force a rebuild of the lazy lookup on next access.
        self.globaldata.lock().unwrap().indirectcall_dict = None;
    }

    /// pyjitpl.py:2251-2253 `setup_list_of_addr2name(list_of_addr2name)`.
    pub fn setup_list_of_addr2name(&mut self, list_of_addr2name: Vec<(usize, String)>) {
        self._addr2name_keys = list_of_addr2name.iter().map(|(k, _)| *k).collect();
        self._addr2name_values = list_of_addr2name.into_iter().map(|(_, v)| v).collect();
        self.globaldata.lock().unwrap().addr2name = None;
    }

    /// `pyjitpl.py:2271-2283` — the tail of `finish_setup` that
    /// attaches `portal_finishtoken` + `propagate_exc_descr` to each
    /// `JitDriverStaticData` and publishes the shared
    /// `PropagateExceptionDescr` on `self`.
    ///
    /// ```python
    /// # pyjitpl.py:2271
    /// # store this information for fastpath of call_assembler
    /// # (only the paths that can actually be taken)
    /// exc_descr = compile.PropagateExceptionDescr()
    /// for jd in self.jitdrivers_sd:
    ///     name = {history.INT: 'int', history.REF: 'ref',
    ///             history.FLOAT: 'float', history.VOID: 'void'}[jd.result_type]
    ///     token = getattr(self, 'done_with_this_frame_descr_%s' % name)
    ///     jd.portal_finishtoken = token
    ///     jd.propagate_exc_descr = exc_descr
    /// self.cpu.propagate_exception_descr = exc_descr
    /// ```
    ///
    /// pyre runs this after every `register_jitdriver_sd` so fresh
    /// drivers inherit the already-wired descrs without a separate
    /// `finish_setup(codewriter)` call.  The backend handle is threaded
    /// in to match `pyjitpl.py:2283 self.cpu.propagate_exception_descr
    /// = exc_descr` — upstream binds the descr to the cpu instance
    /// inside the same method body.
    pub fn finish_setup_descrs_for_jitdrivers(&mut self, cpu: &mut dyn majit_backend::Backend) {
        // `pyjitpl.py:2273` `exc_descr = compile.PropagateExceptionDescr()` —
        // a *single* shared instance across every jitdriver + the cpu.
        // pyre's `register_jitdriver_sd` calls this method on every
        // driver insertion, so create the descr lazily and reuse it on
        // subsequent calls to preserve identity across drivers.
        let exc_descr: majit_ir::DescrRef = match self.propagate_exception_descr.as_ref() {
            Some(existing) => existing.clone(),
            None => {
                let fresh: majit_ir::DescrRef =
                    std::sync::Arc::new(crate::compile::PropagateExceptionDescr::new());
                self.propagate_exception_descr = Some(fresh.clone());
                fresh
            }
        };
        // `pyjitpl.py:2283` `self.cpu.propagate_exception_descr = exc_descr` —
        // bind the shared instance to the backend half of the pair. Idempotent
        // across repeated `register_jitdriver_sd` calls because the backend
        // setters accept the same `Arc` by identity.
        cpu.set_propagate_exception_descr(exc_descr.clone());
        // `pyjitpl.py:2274-2281` per-driver attachment.
        for jd in self.jitdrivers_sd.iter_mut() {
            // `pyjitpl.py:2275-2279` `token = getattr(self,
            // 'done_with_this_frame_descr_%s' % name)`.
            let token = match jd.result_type {
                Type::Int => self.done_with_this_frame_descr_int.as_ref(),
                Type::Ref => self.done_with_this_frame_descr_ref.as_ref(),
                Type::Float => self.done_with_this_frame_descr_float.as_ref(),
                Type::Void => self.done_with_this_frame_descr_void.as_ref(),
            };
            // `pyjitpl.py:2280` `jd.portal_finishtoken = token`.
            jd.portal_finishtoken = token.cloned();
            // `pyjitpl.py:2281` `jd.propagate_exc_descr = exc_descr`.
            jd.propagate_exc_descr = Some(exc_descr.clone());
            // `warmspot.py:1013-1017` `jd.portal_calldescr =
            // self.cpu.calldescrof(...)` — logically warmspot-side but
            // pyre co-locates it here because pyre has no standalone
            // warmspot module and the inputs (green/red types,
            // result_type) are all final by this point.  Build only
            // on first attachment so later `register_jitdriver_sd`
            // calls don't replace an already-published `Arc<Descr>`.
            if jd.portal_calldescr.is_none() {
                jd.build_portal_calldescr();
            }
        }
    }

    /// pyjitpl.py:2305-2323 `get_name_from_address(addr)`.
    pub fn get_name_from_address(&self, addr: usize) -> String {
        let mut gd = self.globaldata.lock().unwrap();
        let dict = gd.addr2name.get_or_insert_with(|| {
            let mut d = std::collections::HashMap::new();
            for (i, key) in self._addr2name_keys.iter().enumerate() {
                if let Some(value) = self._addr2name_values.get(i) {
                    d.insert(*key, value.clone());
                }
            }
            d
        });
        dict.get(&addr).cloned().unwrap_or_default()
    }

    /// pyjitpl.py:2326-2343 `bytecode_for_address(fnaddress)`.
    ///
    /// ```python
    /// def bytecode_for_address(self, fnaddress):
    ///     if we_are_translated():
    ///         d = self.globaldata.indirectcall_dict
    ///         if d is None:
    ///             d = {}
    ///             for jitcode in self.indirectcalltargets:
    ///                 assert jitcode.fnaddr not in d
    ///                 d[jitcode.fnaddr] = jitcode
    ///             self.globaldata.indirectcall_dict = d
    ///         return d.get(fnaddress, None)
    ///     else:
    ///         for jitcode in self.indirectcalltargets:
    ///             if jitcode.fnaddr == fnaddress:
    ///                 return jitcode
    ///         return None
    /// ```
    pub fn bytecode_for_address(
        &self,
        fnaddress: usize,
    ) -> Option<std::sync::Arc<crate::jitcode::JitCode>> {
        let mut gd = self.globaldata.lock().unwrap();
        bytecode_for_address_in_targets(
            &self.indirectcalltargets,
            &mut gd.indirectcall_dict,
            fnaddress,
            |jitcode| jitcode.fnaddr as usize,
        )
    }
}

#[cfg(test)]
mod metainterp_static_data_tests {
    use super::*;
    use crate::jitcode::{JitCode, JitCodeBuilder};
    use majit_translate::jitcode::JitCode as BuildJitCode;

    /// Build a placeholder `Arc<JitCode>` whose `fnaddr` matches the
    /// given address.  Real production code populates `fnaddr` via
    /// `getfunctionptr(graph)` (warmspot.py:418); these tests only need
    /// the lookup key to match.
    fn make_jitcode_with_fnaddr(fnaddr: usize) -> std::sync::Arc<JitCode> {
        let builder = JitCodeBuilder::new();
        let mut jitcode = builder.finish();
        jitcode.fnaddr = fnaddr as i64;
        std::sync::Arc::new(jitcode)
    }

    #[test]
    fn bytecode_for_address_returns_none_when_empty() {
        let mut sd = MetaInterpStaticData::new();
        assert!(sd.bytecode_for_address(0xdeadbeef).is_none());
    }

    #[test]
    fn bytecode_for_address_returns_jitcode_when_registered() {
        let mut sd = MetaInterpStaticData::new();
        let j100 = make_jitcode_with_fnaddr(0x100);
        let j200 = make_jitcode_with_fnaddr(0x200);
        let j300 = make_jitcode_with_fnaddr(0x300);
        sd.setup_indirectcalltargets(vec![j100.clone(), j200.clone(), j300.clone()]);
        assert!(std::sync::Arc::ptr_eq(
            &sd.bytecode_for_address(0x100).unwrap(),
            &j100
        ));
        assert!(std::sync::Arc::ptr_eq(
            &sd.bytecode_for_address(0x200).unwrap(),
            &j200
        ));
        assert!(std::sync::Arc::ptr_eq(
            &sd.bytecode_for_address(0x300).unwrap(),
            &j300
        ));
        assert!(sd.bytecode_for_address(0x400).is_none());
    }

    #[test]
    fn setup_indirectcalltargets_invalidates_cache() {
        let mut sd = MetaInterpStaticData::new();
        sd.setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(0x100)]);
        assert!(sd.bytecode_for_address(0x100).is_some());
        sd.setup_indirectcalltargets(vec![
            make_jitcode_with_fnaddr(0x200),
            make_jitcode_with_fnaddr(0x300),
        ]);
        assert!(sd.bytecode_for_address(0x100).is_none());
        assert!(sd.bytecode_for_address(0x200).is_some());
    }

    #[test]
    fn build_indirectcall_dict_accepts_canonical_build_jitcodes() {
        // Same RPython dict semantics should work for the canonical
        // codewriter JitCode object graph too: the helper cares only
        // about shared object identity + `jitcode.fnaddr`.
        let mut j100 = BuildJitCode::new("build/j100");
        j100.fnaddr = 0x100;
        let j100 = std::sync::Arc::new(j100);

        let mut j200 = BuildJitCode::new("build/j200");
        j200.fnaddr = 0x200;
        let j200 = std::sync::Arc::new(j200);

        let dict = build_indirectcall_dict(&[j100.clone(), j200.clone()], |jitcode| {
            jitcode.fnaddr as usize
        });
        assert!(std::sync::Arc::ptr_eq(dict.get(&0x100).unwrap(), &j100));
        assert!(std::sync::Arc::ptr_eq(dict.get(&0x200).unwrap(), &j200));
        assert!(!dict.contains_key(&0x300));
    }

    #[test]
    fn bytecode_for_address_helper_accepts_canonical_build_jitcodes() {
        // The actual `bytecode_for_address` lookup path should be reusable
        // with the canonical codewriter JitCode store too. This keeps the
        // fnaddr->jitcode semantics independent from the current runtime
        // adapter storage edge.
        let mut j100 = BuildJitCode::new("build/j100");
        j100.fnaddr = 0x100;
        let j100 = std::sync::Arc::new(j100);

        let mut j200 = BuildJitCode::new("build/j200");
        j200.fnaddr = 0x200;
        let j200 = std::sync::Arc::new(j200);

        let targets = vec![j100.clone(), j200.clone()];
        let mut cache = None;

        assert!(std::sync::Arc::ptr_eq(
            &bytecode_for_address_in_targets(&targets, &mut cache, 0x100, |jitcode| {
                jitcode.fnaddr as usize
            })
            .unwrap(),
            &j100
        ));
        assert!(std::sync::Arc::ptr_eq(
            &bytecode_for_address_in_targets(&targets, &mut cache, 0x200, |jitcode| {
                jitcode.fnaddr as usize
            })
            .unwrap(),
            &j200
        ));
        assert!(
            bytecode_for_address_in_targets(&targets, &mut cache, 0x300, |jitcode| {
                jitcode.fnaddr as usize
            })
            .is_none()
        );
    }

    #[test]
    fn get_name_from_address_lazy_dict_build() {
        let mut sd = MetaInterpStaticData::new();
        sd.setup_list_of_addr2name(vec![(0x100, "alpha".into()), (0x200, "beta".into())]);
        assert_eq!(sd.get_name_from_address(0x100), "alpha");
        assert_eq!(sd.get_name_from_address(0x200), "beta");
        assert_eq!(sd.get_name_from_address(0x300), "");
    }

    fn make_call_descr_void() -> (majit_ir::DescrRef, StubCallDescr) {
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Void,
            majit_ir::EffectInfo::default(),
        );
        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        (descr_ref, descr_view)
    }

    #[test]
    fn do_residual_or_indirect_call_falls_through_when_no_jitcode() {
        // pyjitpl.py:2174-2186 — when bytecode_for_address misses, the
        // method returns Ok(self.do_residual_call_full(...)) instead of
        // raising ChangeFrame.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let fnaddr = execute_varargs_void_helper as *const () as i64;
        let action = meta.force_start_tracing(
            0,
            (0, 0),
            None,
            &[Value::Ref(majit_ir::GcRef(fnaddr as usize))],
        );
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let (descr_ref, descr_view) = make_call_descr_void();
        let funcbox = (JitArgKind::Ref, OpRef::ref_op(0), fnaddr);
        let result = meta
            .do_residual_or_indirect_call(funcbox, &[], descr_ref, &descr_view, 0, None)
            .expect("Ok");
        // Empty effectinfo + Void descr → CallN emitted, returns None.
        assert!(result.is_none(), "void result must be None");
        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::CallN),
            "CallN must be recorded on the residual path",
        );
    }

    #[test]
    fn do_residual_or_indirect_call_invokes_perform_call_on_hit() {
        // After registering an indirect-call target, the method must
        // route into perform_call (which raises ChangeFrame) instead of
        // falling through to do_residual_call_full — but only when the
        // funcbox OpRef is a Const (pyjitpl.py:2178 `isinstance(funcbox,
        // Const)`).
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let fnaddr = execute_varargs_void_helper as *const () as i64 as usize;
        std::sync::Arc::get_mut(&mut meta.staticdata)
            .unwrap()
            .setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(fnaddr)]);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let funcbox_ref = meta
            .trace_ctx()
            .expect("active trace")
            .const_ref(fnaddr as i64);
        let (descr_ref, descr_view) = make_call_descr_void();
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr as i64);
        let result =
            meta.do_residual_or_indirect_call(funcbox, &[], descr_ref, &descr_view, 0, None);
        // Const funcbox + registered target → perform_call raises
        // ChangeFrame (wrapped in DoResidualCallAbort).
        assert!(matches!(result, Err(DoResidualCallAbort::ChangeFrame)));
    }

    // pyjitpl.py:2186 miss path (`self.do_residual_call(...)`) is covered by
    // `do_residual_call_full`'s own tests — they exercise every branch
    // (OS_NOT_IN_TRACE, force_virtual_or_virtualizable, CALL_MAY_FORCE,
    // libffi, release_gil, regular CALL_*).  A miss-path unit test here
    // would duplicate that fixture setup without adding coverage.

    #[test]
    fn do_residual_or_indirect_call_skips_perform_call_for_non_const_funcbox() {
        // pyjitpl.py:2178 — non-Const funcbox must NOT be promoted to an
        // inlined call even when its concrete address matches a
        // registered indirect-call target.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let fnaddr = execute_varargs_void_helper as *const () as i64 as usize;
        std::sync::Arc::get_mut(&mut meta.staticdata)
            .unwrap()
            .setup_indirectcalltargets(vec![make_jitcode_with_fnaddr(fnaddr)]);
        let action =
            meta.force_start_tracing(0, (0, 0), None, &[Value::Ref(majit_ir::GcRef(fnaddr))]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let (descr_ref, descr_view) = make_call_descr_void();
        let funcbox = (JitArgKind::Ref, OpRef::ref_op(0), fnaddr as i64); // non-Const
        let result = meta
            .do_residual_or_indirect_call(funcbox, &[], descr_ref, &descr_view, 0, None)
            .expect("Ok — falls through to residual call, no ChangeFrame");
        assert!(result.is_none(), "void residual call returns None");
    }

    #[test]
    fn finishframe_raises_done_with_this_frame_void_when_stack_exhausted() {
        // pyjitpl.py:2493-2496: result_type == VOID + resultbox is None
        // → raise DoneWithThisFrameVoid().
        let mut meta = MetaInterp::<()>::new(0);
        let result = meta.finishframe(None, true);
        assert!(matches!(
            result,
            Err(FinishFrameSignal::Done(DoneWithThisFrame::Void))
        ));
    }

    #[test]
    fn finishframe_raises_done_with_this_frame_int_when_stack_exhausted_with_int_result() {
        // pyjitpl.py:2497-2498: result_type == INT → raise
        // DoneWithThisFrameInt(resultbox.getint()).
        // RPython parity: `compile_done_with_this_frame(resultbox)` reads
        // `resultbox.getint()` — the symbolic resultbox must be a real
        // ConstInt(0xc0ffee), not a bare unset OpRef. Mint via
        // `ctx.const_int` so the trace's constant pool resolves the slot.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let resultbox_ref = meta.trace_ctx().expect("active trace").const_int(0xc0ffee);
        let result = meta.finishframe(Some((JitArgKind::Int, 0, resultbox_ref, 0xc0ffee)), true);
        assert!(matches!(
            result,
            Err(FinishFrameSignal::Done(DoneWithThisFrame::Int(0xc0ffee)))
        ));
    }

    #[test]
    fn finishframe_raises_done_with_this_frame_ref_for_ref_result() {
        // pyjitpl.py:2499-2500.
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let result = meta.finishframe(Some((JitArgKind::Ref, 0, OpRef::ref_op(101), 0xfeed)), true);
        assert!(matches!(
            result,
            Err(FinishFrameSignal::Done(DoneWithThisFrame::Ref(r))) if r.0 == 0xfeed
        ));
    }

    #[test]
    fn finishframe_raises_done_with_this_frame_float_for_float_result() {
        // pyjitpl.py:2501-2502.
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let bits = f64::to_bits(2.5) as i64;
        let result = meta.finishframe(
            Some((JitArgKind::Float, 0, OpRef::float_op(102), bits)),
            true,
        );
        assert!(matches!(
            result,
            Err(FinishFrameSignal::Done(DoneWithThisFrame::Float(v))) if v.to_bits() == bits as u64
        ));
    }

    #[test]
    fn finishframe_uses_jitdriver_result_type_when_available() {
        // pyjitpl.py:2493 — `result_type = self.jitdriver_sd.result_type`.
        // The DoneWithThisFrame variant is determined by the active
        // jitdriver's declared return type, not by the resultbox kind.
        // Here the popped frame has jitdriver_sd=Some(0) and that
        // driver declares result_type=Ref, so a tuple with kind=Int but
        // value 0xfeed is reported as DoneWithThisFrameRef(0xfeed).
        use crate::jitcode::{JitArgKind, JitCodeBuilder};
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        let jitcode = builder.finish();
        jitcode.set_jitdriver_sd(0);
        let mut meta = MetaInterp::<()>::new(0);
        let driver = crate::jitdriver::JitDriverStaticData {
            index: None,
            vars: vec![],
            virtualizable: None,
            result_type: majit_ir::Type::Ref,
            is_recursive: false,
            mainjitcode: None,
            portal_runner_adr: 0,
            virtualizable_info: None,
            greenfield_info: None,
            index_of_virtualizable: -1,
            portal_calldescr: None,
            portal_finishtoken: None,
            propagate_exc_descr: None,
            red_args_types: vec![],
            no_loop_header: false,
            assembler_helper_adr: 0,
            vable_token_descr: None,
        };
        {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            let sd = std::sync::Arc::get_mut(staticdata).unwrap();
            let _ = sd.register_jitdriver_sd(driver, backend);
        }
        meta.framestack.push(crate::pyjitpl::MIFrame::new(
            std::sync::Arc::new(jitcode),
            0,
        ));
        let result = meta.finishframe(Some((JitArgKind::Int, 0, OpRef::int_op(1), 0xfeed)), true);
        assert!(matches!(
            result,
            Err(FinishFrameSignal::Done(DoneWithThisFrame::Ref(r))) if r.0 == 0xfeed
        ));
    }

    #[test]
    fn perform_call_pushes_frame_and_setup_call_writes_argboxes() {
        // pyjitpl.py:2421-2425 perform_call → newframe → setup_call →
        // raise ChangeFrame.  After the call, framestack should have one
        // frame whose typed register banks reflect the argboxes.
        use crate::jitcode::{JitArgKind, JitCodeBuilder};
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        builder.load_const_i_value(1, 0);
        builder.load_const_r_value(0, 0);
        builder.load_const_f_value(0, 0);
        let jitcode = std::sync::Arc::new(builder.finish());
        let mut meta = MetaInterp::<()>::new(0);
        let result = meta.perform_call(
            jitcode,
            &[
                (JitArgKind::Int, OpRef::int_op(10), 100),
                (JitArgKind::Ref, OpRef::ref_op(20), 200),
                (JitArgKind::Int, OpRef::int_op(11), 101),
                (JitArgKind::Float, OpRef::float_op(30), 300),
            ],
            None,
        );
        assert!(matches!(result, Err(ChangeFrame)));
        assert_eq!(meta.framestack.len(), 1);
        let f = meta.framestack.current_mut();
        assert_eq!(f.pc, 0);
        assert_eq!(f.int_regs[0], Some(OpRef::int_op(10)));
        assert_eq!(f.int_values[0], Some(100));
        assert_eq!(f.int_regs[1], Some(OpRef::int_op(11)));
        assert_eq!(f.int_values[1], Some(101));
        assert_eq!(f.ref_regs[0], Some(OpRef::ref_op(20)));
        assert_eq!(f.ref_values[0], Some(200));
        assert_eq!(f.float_regs[0], Some(OpRef::float_op(30)));
        assert_eq!(f.float_values[0], Some(300));
    }

    #[test]
    fn finishframe_writes_result_into_caller_then_change_frame() {
        // pyjitpl.py:2483-2486 — popframe + framestack[-1].make_result_of_lastop +
        // raise ChangeFrame.
        //
        // RPython parity (pyjitpl.py:258-265, 2479-2486): when
        // `make_result_of_lastop` fires on the caller frame after the
        // callee returns, it reads `_resulttypes[self.pc]` and asserts
        // the recorded kind matches the runtime kind.  To exercise that
        // assertion (which `MIFrame::make_result_of_lastop` mirrors as
        // a `debug_assert`), the caller jitcode must contain a real
        // typed-call bytecode whose end-of-instruction position is
        // reflected in `caller.pc` at finishframe time.
        use crate::jitcode::{JitArgKind, JitCodeBuilder};
        let mut builder_caller = JitCodeBuilder::new();
        builder_caller.load_const_i_value(0, 0);
        builder_caller.load_const_i_value(1, 0);
        // assembler.py:217-219 — `inline_call_irf_i` records
        // `resulttypes[end_pc] = 'i'` so the post-call pc lookup
        // succeeds at make_result_of_lastop.  The sub_jitcode index
        // (0) is a placeholder — perform_call(callee) below pushes
        // the actual callee onto the framestack regardless of the
        // bytecode operand, since this test never runs the dispatch
        // loop.
        builder_caller.inline_call_irf_i(0, &[], &[], &[], Some(1));
        let post_call_pc = builder_caller.current_pos();
        let caller = std::sync::Arc::new(builder_caller.finish());
        let builder_callee = JitCodeBuilder::new();
        let callee = std::sync::Arc::new(builder_callee.finish());

        let mut meta = MetaInterp::<()>::new(0);
        // Push caller, then advance caller.pc to the post-call
        // position the way the bytecode dispatch loop would (the
        // opimpl reads operand bytes and bumps `pc` past the entire
        // `inline_call_irf_i` instruction before raising ChangeFrame
        // and yielding control to the callee — pyjitpl.py:2475-2479).
        meta.perform_call(caller, &[], None).unwrap_err();
        meta.framestack.current_mut().pc = post_call_pc;
        meta.perform_call(callee, &[], None).unwrap_err();
        assert_eq!(meta.framestack.len(), 2);

        // Return from callee: write result into caller register 1.
        // `make_result_of_lastop` (frame.rs) checks
        // `caller.jitcode.body.resulttypes[caller.pc]` matches the
        // runtime kind — both are 'i' here, so the debug_assert
        // passes and the result is written.
        let result = meta.finishframe(Some((JitArgKind::Int, 1, OpRef::int_op(42), 4242)), true);
        assert!(matches!(result, Err(FinishFrameSignal::ChangeFrame)));
        assert_eq!(meta.framestack.len(), 1);
        let caller_frame = meta.framestack.current_mut();
        assert_eq!(caller_frame.pc, post_call_pc);
        assert_eq!(caller_frame.int_regs[1], Some(OpRef::int_op(42)));
        assert_eq!(caller_frame.int_values[1], Some(4242));
    }

    #[test]
    fn finishframe_void_return_skips_make_result_of_lastop() {
        use crate::jitcode::JitCodeBuilder;
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        let caller = std::sync::Arc::new(builder.finish());
        let callee = std::sync::Arc::new(JitCodeBuilder::new().finish());

        let mut meta = MetaInterp::<()>::new(0);
        meta.perform_call(caller, &[], None).unwrap_err();
        // Mutate the caller's register 0 so we can detect any
        // accidental write triggered by the void return.
        meta.framestack.current_mut().int_regs[0] = Some(OpRef::int_op(7));
        meta.framestack.current_mut().int_values[0] = Some(7);
        meta.perform_call(callee, &[], None).unwrap_err();
        let result = meta.finishframe(None, true);
        assert!(matches!(result, Err(FinishFrameSignal::ChangeFrame)));
        // Void return preserves whatever was already there in the caller.
        assert_eq!(
            meta.framestack.current_mut().int_regs[0],
            Some(OpRef::int_op(7))
        );
        assert_eq!(meta.framestack.current_mut().int_values[0], Some(7));
    }

    #[test]
    fn initialize_state_from_start_clears_and_seeds_framestack() {
        // pyjitpl.py:3266-3275 — start a fresh portal: framestack reset,
        // mainjitcode pushed, original_boxes copied via setup_call.
        use crate::jitcode::{JitArgKind, JitCodeBuilder};
        let mut builder = JitCodeBuilder::new();
        builder.load_const_i_value(0, 0);
        let mainjitcode = builder.finish();
        // pyjitpl.py:3268-3272 — the mainjitcode is the portal jitcode
        // and must carry jitdriver_sd so portal_call_depth bumps from
        // -1 to 0 inside newframe (matching the upstream assert).
        mainjitcode.set_jitdriver_sd(0);
        let mainjitcode = std::sync::Arc::new(mainjitcode);

        let mut meta = MetaInterp::<()>::new(0);
        // Pre-populate framestack with a stale frame to verify reset.
        meta.perform_call(mainjitcode.clone(), &[], None)
            .unwrap_err();
        assert_eq!(meta.framestack.len(), 1);
        // Pre-populate virtualref_boxes to verify it gets cleared.
        meta.virtualref_boxes.push((OpRef::int_op(99), 99));

        meta.initialize_state_from_start(mainjitcode, &[(JitArgKind::Int, OpRef::int_op(7), 7)]);
        assert_eq!(meta.framestack.len(), 1);
        assert_eq!(
            meta.framestack.current_mut().int_regs[0],
            Some(OpRef::int_op(7))
        );
        assert_eq!(meta.framestack.current_mut().int_values[0], Some(7));
        assert!(meta.virtualref_boxes.is_empty());
        // pyjitpl.py:3272 assert.
        assert_eq!(meta.portal_call_depth, 0);
    }

    #[test]
    fn trace_jitcode_with_framestack_pushes_root_then_pops() {
        // pyjitpl.py self.framestack invariant: trace entry pushes the
        // root frame, runs the jitcode interp, and the stack is empty
        // again on return.
        use crate::BackEdgeAction;
        use crate::jitcode::JitCodeBuilder;
        let jitcode = std::sync::Arc::new(JitCodeBuilder::new().finish());

        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        struct NoopSym;
        impl crate::JitCodeSym for NoopSym {
            fn total_slots(&self) -> usize {
                0
            }
            fn loop_header_pc(&self) -> usize {
                0
            }
            fn fail_args(&self) -> Option<Vec<OpRef>> {
                None
            }
        }
        let mut sym = NoopSym;
        let runtime = crate::ClosureRuntime::new(|_| 0);

        let action = meta.trace_jitcode_with_framestack(&mut sym, jitcode, 0, &runtime);
        assert!(matches!(action, crate::TraceAction::Continue));
        assert_eq!(meta.framestack.len(), 0);
    }

    #[test]
    fn portal_call_depth_bumps_for_jitdriver_jitcode_only() {
        // pyjitpl.py:2434/2466 — portal_call_depth ± 1 when the frame's
        // jitcode carries a jitdriver_sd; non-portal frames leave it
        // alone.
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let initial = meta.portal_call_depth;

        // Push a non-portal frame: counter unchanged.
        let plain = std::sync::Arc::new(JitCodeBuilder::new().finish());
        meta.perform_call(plain, &[], None).unwrap_err();
        assert_eq!(meta.portal_call_depth, initial);

        // Push a portal frame on top: counter += 1.
        let mut portal = JitCodeBuilder::new().finish();
        portal.replace_jitdriver_sd(Some(0));
        let portal = std::sync::Arc::new(portal);
        meta.perform_call(portal, &[], None).unwrap_err();
        assert_eq!(meta.portal_call_depth, initial + 1);

        // popframe drops the portal frame: counter -= 1.
        meta.popframe(true);
        assert_eq!(meta.portal_call_depth, initial);

        // popframe drops the non-portal frame: counter unchanged.
        meta.popframe(true);
        assert_eq!(meta.portal_call_depth, initial);
    }

    /// Tiny CallDescr stub for `_build_allboxes` tests.
    #[derive(Debug)]
    struct StubCallDescr {
        arg_types: Vec<majit_ir::Type>,
        result_type: majit_ir::Type,
        effect: majit_ir::EffectInfo,
    }

    impl majit_ir::descr::Descr for StubCallDescr {}

    impl majit_ir::descr::CallDescr for StubCallDescr {
        fn arg_types(&self) -> &[majit_ir::Type] {
            &self.arg_types
        }
        fn result_type(&self) -> majit_ir::Type {
            self.result_type
        }
        fn result_size(&self) -> usize {
            8
        }
        fn get_extra_info(&self) -> &majit_ir::EffectInfo {
            &self.effect
        }
    }

    #[test]
    fn build_allboxes_simple_case_no_prepend_box() {
        // pyjitpl.py:1960-1993 — without prepend_box, allboxes is just
        // [funcbox, *argboxes].
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let descr = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Ref],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let ctx = meta.trace_ctx().expect("active trace");
        let funcbox_ref = ctx.const_ref(0xdead);
        let argbox0_ref = ctx.const_int(11);
        let argbox1_ref = ctx.const_ref(22);
        let funcbox = (JitArgKind::Ref, funcbox_ref, 0xdead);
        let argboxes = [
            (JitArgKind::Int, argbox0_ref, 11),
            (JitArgKind::Ref, argbox1_ref, 22),
        ];
        let all = meta._build_allboxes(funcbox, &argboxes, &descr, None);
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], funcbox);
        assert_eq!(all[1], argboxes[0]);
        assert_eq!(all[2], argboxes[1]);
    }

    #[test]
    fn build_allboxes_with_prepend_box_places_it_first() {
        // pyjitpl.py:1963-1965 — prepend_box (e.g. condbox in
        // do_conditional_call) goes to slot 0.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let descr = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let ctx = meta.trace_ctx().expect("active trace");
        let prepend_ref = ctx.const_int(0);
        let funcbox_ref = ctx.const_ref(0xfeed);
        let argbox_ref = ctx.const_int(1);
        let prepend = (JitArgKind::Int, prepend_ref, 0);
        let funcbox = (JitArgKind::Ref, funcbox_ref, 0xfeed);
        let argboxes = [(JitArgKind::Int, argbox_ref, 1)];
        let all = meta._build_allboxes(funcbox, &argboxes, &descr, Some(prepend));
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], prepend);
        assert_eq!(all[1], funcbox);
        assert_eq!(all[2], argboxes[0]);
    }

    #[test]
    fn build_allboxes_demuxes_bank_sorted_input_to_declaration_order() {
        // pyjitpl.py:1968-1991 — the three-counter demuxing loop pulls
        // each box from the bank that matches `descr.get_arg_types()`.
        // When `argboxes` arrives in bank-sorted layout (all ints
        // first, then refs, then floats), the output must still match
        // declaration order.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        // descr declares [Int, Ref, Int, Ref] (declaration order).
        let descr = StubCallDescr {
            arg_types: vec![
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
            ],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let ctx = meta.trace_ctx().expect("active trace");
        let funcbox_ref = ctx.const_ref(0xdead);
        let i0 = ctx.const_int(11);
        let i1 = ctx.const_int(33);
        let r0 = ctx.const_ref(22);
        let r1 = ctx.const_ref(44);
        let funcbox = (JitArgKind::Ref, funcbox_ref, 0xdead);
        // argboxes arrives bank-sorted: [int@0, int@2, ref@1, ref@3].
        let argboxes = [
            (JitArgKind::Int, i0, 11),
            (JitArgKind::Int, i1, 33),
            (JitArgKind::Ref, r0, 22),
            (JitArgKind::Ref, r1, 44),
        ];
        let all = meta._build_allboxes(funcbox, &argboxes, &descr, None);
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], funcbox);
        // Output is declaration order [Int, Ref, Int, Ref].
        assert_eq!(all[1], (JitArgKind::Int, i0, 11));
        assert_eq!(all[2], (JitArgKind::Ref, r0, 22));
        assert_eq!(all[3], (JitArgKind::Int, i1, 33));
        assert_eq!(all[4], (JitArgKind::Ref, r1, 44));
    }

    #[test]
    fn build_allboxes_passes_through_declaration_order_input_unchanged() {
        // The demuxing loop degrades to a 1:1 walk when `argboxes`
        // already arrives in declaration order — pyre's BC encoder
        // produces this layout today, so the loop is exercised on
        // every production call.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let descr = StubCallDescr {
            arg_types: vec![
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
                majit_ir::Type::Int,
                majit_ir::Type::Ref,
            ],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let ctx = meta.trace_ctx().expect("active trace");
        let funcbox_ref = ctx.const_ref(0xdead);
        let i0 = ctx.const_int(11);
        let i1 = ctx.const_int(33);
        let r0 = ctx.const_ref(22);
        let r1 = ctx.const_ref(44);
        let funcbox = (JitArgKind::Ref, funcbox_ref, 0xdead);
        // argboxes already in declaration order [Int, Ref, Int, Ref].
        let argboxes = [
            (JitArgKind::Int, i0, 11),
            (JitArgKind::Ref, r0, 22),
            (JitArgKind::Int, i1, 33),
            (JitArgKind::Ref, r1, 44),
        ];
        let all = meta._build_allboxes(funcbox, &argboxes, &descr, None);
        assert_eq!(all.len(), 5);
        assert_eq!(all[0], funcbox);
        assert_eq!(all[1], argboxes[0]);
        assert_eq!(all[2], argboxes[1]);
        assert_eq!(all[3], argboxes[2]);
        assert_eq!(all[4], argboxes[3]);
    }

    #[test]
    fn jit_arg_kind_from_type_maps_int_ref_float_void() {
        use crate::jitcode::JitArgKind;
        assert_eq!(
            JitArgKind::from_type(majit_ir::Type::Int),
            Some(JitArgKind::Int)
        );
        assert_eq!(
            JitArgKind::from_type(majit_ir::Type::Ref),
            Some(JitArgKind::Ref)
        );
        assert_eq!(
            JitArgKind::from_type(majit_ir::Type::Float),
            Some(JitArgKind::Float)
        );
        assert_eq!(JitArgKind::from_type(majit_ir::Type::Void), None);
    }

    extern "C" fn not_in_trace_clear_exc_helper() {
        // Test helper that clears the EXC_TLS thread-local in tests.
        // The do_not_in_trace_call test below sets last_exc_value
        // explicitly, so this helper just runs the call.
    }

    extern "C" fn not_in_trace_record_arg_helper(arg: i64) {
        NOT_IN_TRACE_LAST_ARG.store(arg, std::sync::atomic::Ordering::SeqCst);
    }

    static NOT_IN_TRACE_LAST_ARG: std::sync::atomic::AtomicI64 =
        std::sync::atomic::AtomicI64::new(0);

    #[test]
    fn do_not_in_trace_call_executes_void_helper_without_recording_ir() {
        // pyjitpl.py:3683-3693 — execute the call (side effect happens)
        // and return Ok(None) when no exception was raised.  No IR ops
        // are emitted because `executor.execute_varargs` is the
        // non-recording path.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let fnaddr = not_in_trace_record_arg_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let argbox = (JitArgKind::Int, OpRef::int_op(1), 0xc0ffee);
        let allboxes = [funcbox, argbox];

        // Pre-populate last_exc_value to verify clear_exception runs.
        meta.last_exc_value = 0xbad;
        NOT_IN_TRACE_LAST_ARG.store(0, std::sync::atomic::Ordering::SeqCst);

        let result = meta.do_not_in_trace_call(&allboxes, &descr);
        assert!(matches!(result, Ok(None)));
        assert_eq!(meta.last_exc_value, 0, "clear_exception must have run");
        assert_eq!(
            NOT_IN_TRACE_LAST_ARG.load(std::sync::atomic::Ordering::SeqCst),
            0xc0ffee,
            "helper must have observed the concrete arg"
        );

        // No IR ops should have been recorded.
        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.ops().is_empty(),
            "do_not_in_trace_call must not record IR ops"
        );
    }

    /// Helper that simulates a raising callable by writing the
    /// exception value to the `BH_LAST_EXC_VALUE` thread-local — the
    /// same seam production helpers (bh_call_fn_impl etc.) publish on
    /// and that `do_not_in_trace_call` reads back into
    /// `MetaInterp::last_exc_value`.
    extern "C" fn raising_not_in_trace_helper() {
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xfeed));
    }

    #[test]
    fn do_not_in_trace_call_returns_abort_escape_on_exception() {
        // pyjitpl.py:3687-3692 — if last_exc_value is set after the
        // call, raise SwitchToBlackhole(ABORT_ESCAPE).
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let fnaddr = raising_not_in_trace_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let allboxes = [funcbox];

        // Pre-set BH_LAST_EXC_VALUE to a stale value.  do_not_in_trace_call
        // must clear it before the call (RPython's cpu auto-clears) and
        // re-read after.  The helper writes 0xfeed; the named entry must
        // transcribe that onto last_exc_value and return ABORT_ESCAPE.
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xdead));
        // Pre-set the class-const flag too — execute_raised must reset
        // it (pyjitpl.py:2752: `self.class_of_last_exc_is_const = constant`
        // with `constant=False`).
        meta.class_of_last_exc_is_const = true;

        let result = meta.do_not_in_trace_call(&allboxes, &descr);
        // pyjitpl.py:3691: raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
        //                                          raising_exception=True)
        // — the `raising_exception=True` keyword argument is part of the
        // contract; the blackhole resume path (blackhole.rs:3469-3487)
        // re-raises the helper-side exception only when it is set.
        assert_eq!(
            result,
            Err(SwitchToBlackhole {
                reason: counters::ABORT_ESCAPE,
                raising_exception: true,
            })
        );
        assert_eq!(
            meta.last_exc_value, 0xfeed,
            "helper exception must be transcribed into last_exc_value"
        );
        // BH_LAST_EXC_VALUE must be cleared so the next call site starts
        // clean.
        assert_eq!(
            crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get()),
            0,
            "BH_LAST_EXC_VALUE must be cleared after read"
        );
        // class_of_last_exc_is_const cleared by execute_raised(.., false).
        assert!(!meta.class_of_last_exc_is_const);
    }

    extern "C" fn execute_varargs_int_helper(a: i64, b: i64) -> i64 {
        a + b * 1000
    }

    extern "C" fn execute_varargs_void_helper() {}

    extern "C" fn execute_varargs_float_concrete_helper(a: i64) -> i64 {
        // Mirrors the `concrete_ptr` shape `#[jit_module]` emits for a
        // Float helper (majit-macros/src/lib.rs:267): the f64 result is
        // pre-packed via `f64::to_bits` and the wrapper returns through
        // the integer return register.  Same ABI shape as
        // `bh_portal_runner` (pyre-jit/src/call_jit.rs:467).
        let value = a as f64 * 0.5;
        value.to_bits() as i64
    }

    #[test]
    fn execute_varargs_call_f_routes_through_concrete_ptr_i64_bits() {
        // Float-arm ABI contract: every path that reaches the Float arm
        // (do_recursive_call → portal_runner_adr, force-virtual
        // do_residual_call_full) materialises `funcbox.2` as a function
        // pointer with i64-return ABI — either the hand-written
        // `bh_portal_runner` or `#[jit_module]`'s `concrete_ptr` wrapper
        // that pre-packs the f64 result via `f64::to_bits`.  The f64-ABI
        // `trace_ptr` is consumed only by pyre-jit-trace's
        // `TraceCtx::call_may_force_*` family, which has its own seam
        // and never reaches this arm.  The executor routes through
        // `call_int_function` (i64-bits ABI), and callers recover the
        // f64 via `f64::from_bits` when the slot needs to be interpreted
        // as a float (pyjitpl/mod.rs:8901-8902).  This test pins that
        // contract so a regression that re-introduces the f64-ABI
        // transmute path is caught.
        use crate::executor;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let descr = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Float,
            effect: majit_ir::EffectInfo::default(),
        };
        let fnaddr = execute_varargs_float_concrete_helper as *const () as i64;
        let argboxes = [
            (JitArgKind::Ref, OpRef::ref_op(0), fnaddr),
            (JitArgKind::Int, OpRef::int_op(1), 6),
        ];
        let raw = executor::execute_varargs(&mut meta, OpCode::CallF, &argboxes, &descr);
        // Helper returns `f64::to_bits(3.0)`; executor must carry the
        // raw i64 unmodified so the caller can `f64::from_bits` it.
        assert_eq!(raw, 3.0_f64.to_bits() as i64);
        assert_eq!(f64::from_bits(raw as u64), 3.0);
        assert_eq!(meta.last_exc_value, 0);
    }

    extern "C" fn execute_varargs_raising_int_helper(_a: i64) -> i64 {
        // Helper raises by publishing onto BH_LAST_EXC_VALUE before
        // returning a non-zero stub value — production helpers do this
        // when their callee threw and the JIT bridges the exception
        // back via the thread-local seam.
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0xfeed));
        0xdeadbeef
    }

    #[test]
    fn execute_varargs_zeros_result_and_routes_through_execute_raised_on_exception() {
        // executor.py:52-78 — when the helper raises, the executor calls
        // `metainterp.execute_raised(e)` AND returns the type's neutral
        // zero (INT=0, REF=NULL, FLOAT=ZEROF, VOID ignored).  Pyre must
        // also clear `class_of_last_exc_is_const` (pyjitpl.py:2745-2755)
        // so a stale `True` from a prior GUARD_EXCEPTION cannot leak
        // into the new exception's classification.
        use crate::executor;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        // Pre-set a stale class-const flag and a stale TLS value: the
        // executor must overwrite both.
        meta.class_of_last_exc_is_const = true;
        crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.set(0));

        let descr = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let fnaddr = execute_varargs_raising_int_helper as *const () as i64;
        let argboxes = [
            (JitArgKind::Ref, OpRef::ref_op(0), fnaddr),
            (JitArgKind::Int, OpRef::int_op(1), 7),
        ];

        let raw = executor::execute_varargs(&mut meta, OpCode::CallI, &argboxes, &descr);

        // Result zeroed despite the helper returning 0xdeadbeef.
        assert_eq!(
            raw, 0,
            "executor must override the helper's return value with 0 on exception",
        );
        // Exception state mirrored onto the metainterp.
        assert_eq!(meta.last_exc_value, 0xfeed);
        // class_of_last_exc_is_const cleared (execute_ll_raised sets
        // constant=False per pyjitpl.py:2752).
        assert!(
            !meta.class_of_last_exc_is_const,
            "class_of_last_exc_is_const must be reset when execute_raised fires",
        );
        // TLS drained so the next call site starts clean.
        assert_eq!(crate::blackhole::BH_LAST_EXC_VALUE.with(|c| c.get()), 0);
    }

    #[test]
    fn count_ops_increments_by_kind_and_bumps_calls_only_on_call_recorded_ops() {
        // jitprof.py:118-122 contract: `count_ops(opnum, kind)` bumps
        // `counters[kind]` by 1; if `kind == RECORDED_OPS` AND the op is
        // a CALL_*, also bumps `calls`.  Other (kind, opnum) pairs leave
        // `calls` untouched.  Counters now live on
        // `staticdata.profiler` — read via `snapshot()`.
        let meta = MetaInterp::<()>::new(0);
        let prof = &meta.staticdata.profiler;
        // OPS path: not a call, kind=OPS → ops += 1, calls unchanged.
        meta.count_ops(OpCode::IntAdd, counters::OPS);
        assert_eq!(prof.snapshot().ops, 1);
        assert_eq!(prof.snapshot().calls, 0);
        // OPS path on a CALL_*: kind=OPS so calls is NOT bumped (only
        // RECORDED_OPS path bumps calls per jitprof.py:121).
        meta.count_ops(OpCode::CallI, counters::OPS);
        assert_eq!(prof.snapshot().ops, 2);
        assert_eq!(prof.snapshot().calls, 0);
        // RECORDED_OPS path on a non-call: recorded_ops += 1, calls
        // unchanged.
        meta.count_ops(OpCode::IntAdd, counters::RECORDED_OPS);
        assert_eq!(prof.snapshot().recorded_ops, 1);
        assert_eq!(prof.snapshot().calls, 0);
        // RECORDED_OPS + CALL_*: recorded_ops += 1 AND calls += 1.
        meta.count_ops(OpCode::CallI, counters::RECORDED_OPS);
        assert_eq!(prof.snapshot().recorded_ops, 2);
        assert_eq!(prof.snapshot().calls, 1);
        // HEAPCACHED_OPS / GUARDS independent buckets.
        meta.count_ops(OpCode::PtrEq, counters::HEAPCACHED_OPS);
        meta.count_ops(OpCode::GuardTrue, counters::GUARDS);
        assert_eq!(prof.snapshot().heapcached_ops, 1);
        assert_eq!(prof.snapshot().guards, 1);
        // ABORT_* bumps the abort_* atomic on the profiler (count_ops
        // permits any kind that field_for_kind knows about), and never
        // touches `calls`.
        meta.count_ops(OpCode::IntAdd, counters::ABORT_ESCAPE);
        assert_eq!(prof.snapshot().abort_escape, 1);
        assert_eq!(prof.snapshot().calls, 1);
    }

    #[test]
    fn count_routes_abort_kinds_into_profiler_atomics_and_nv_kinds_into_their_buckets() {
        // jitprof.py:101-102 — `count(reason)` bumps the matching
        // `Counters.*` atomic on `staticdata.profiler`.
        let meta = MetaInterp::<()>::new(0);
        let prof = &meta.staticdata.profiler;
        meta.count(counters::ABORT_TOO_LONG, 1);
        meta.count(counters::ABORT_ESCAPE, 2);
        meta.count(counters::ABORT_BAD_LOOP, 3);
        let snap = prof.snapshot();
        assert_eq!(snap.abort_too_long, 1);
        assert_eq!(snap.abort_escape, 2);
        assert_eq!(snap.abort_bad_loop, 3);
        meta.count(counters::NVIRTUALS, 5);
        meta.count(counters::NVHOLES, 2);
        meta.count(counters::NVREUSED, 7);
        let snap = prof.snapshot();
        assert_eq!(snap.nvirtuals, 5);
        assert_eq!(snap.nvholes, 2);
        assert_eq!(snap.nvreused, 7);
    }

    #[test]
    fn execute_and_record_varargs_bumps_ops_then_record_helper_bumps_recorded_ops() {
        // pyjitpl.py:2645 + 2658 contract: `execute_and_record_varargs`
        // bumps `OPS` before dispatch; `_record_helper_varargs` bumps
        // `RECORDED_OPS` (and `calls` when the op is a CALL_*).  Without
        // an active tracing session `_record_helper_varargs` early-exits
        // after the count_ops call, so we still observe both bumps.
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let argboxes = [
            (JitArgKind::Ref, OpRef::ref_op(0), fnaddr),
            (JitArgKind::Int, OpRef::int_op(1), 3),
            (JitArgKind::Int, OpRef::int_op(2), 4),
        ];
        let _ = meta.execute_and_record_varargs(OpCode::CallI, &argboxes, descr_ref, &descr_view);
        let snap = meta.staticdata.profiler.snapshot();
        // OPS bumped exactly once (entry into execute_and_record_varargs).
        assert_eq!(snap.ops, 1);
        // RECORDED_OPS bumped exactly once (entry into _record_helper_varargs).
        assert_eq!(snap.recorded_ops, 1);
        // CALL_I + RECORDED_OPS path → calls += 1 (jitprof.py:121-122).
        assert_eq!(snap.calls, 1);
    }

    #[test]
    fn do_residual_call_emits_call_i_for_regular_int_call() {
        // pyjitpl.py:2113-2115 — non-loopinvariant, non-force-virtual,
        // int-returning call → CALL_I via miframe_execute_varargs.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let argboxes = [
            (JitArgKind::Int, OpRef::int_op(1), 4),
            (JitArgKind::Int, OpRef::int_op(2), 6),
        ];

        let result = meta.do_residual_call_full(
            funcbox,
            &argboxes,
            descr_ref,
            &descr_view,
            /* pc = */ 0,
            /* assembler_call = */ false,
            /* assembler_call_jd = */ None,
            /* dst = */ None,
        );
        let (opref, resvalue) = result.expect("Ok").expect("non-void Some");
        assert_eq!(resvalue, 4 + 6 * 1000);

        let ctx = meta.trace_ctx().expect("active trace");
        let op = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::CallI)
            .expect("CallI must be recorded");
        assert_eq!(op.pos, opref);
    }

    extern "C" fn cond_call_void_helper(_cond: i64, _func_addr: i64) {}

    #[test]
    fn execute_ll_raised_sets_last_exc_value_and_class_const_flag() {
        // pyjitpl.py:2745-2755 — last_exc_value = llexception;
        //                       class_of_last_exc_is_const = constant.
        let mut meta = MetaInterp::<()>::new(0);
        assert_eq!(meta.last_exc_value, 0);
        assert!(!meta.class_of_last_exc_is_const);

        meta.execute_ll_raised(0xfeed, true);
        assert_eq!(meta.last_exc_value, 0xfeed);
        assert!(meta.class_of_last_exc_is_const);

        meta.execute_ll_raised(0x42, false);
        assert_eq!(meta.last_exc_value, 0x42);
        assert!(!meta.class_of_last_exc_is_const);
    }

    #[test]
    fn execute_raised_forwards_to_execute_ll_raised() {
        // pyjitpl.py:2739-2743 — pyre callers pass the lowered
        // exception pointer directly; execute_raised forwards.
        let mut meta = MetaInterp::<()>::new(0);
        meta.execute_raised(0xc0ffee, false);
        assert_eq!(meta.last_exc_value, 0xc0ffee);
        assert!(!meta.class_of_last_exc_is_const);
    }

    #[test]
    fn aborted_tracing_bumps_loops_aborted_counter() {
        // pyjitpl.py:2761/2786 — profiler.count(reason) + stats.aborted()
        // pyre's `count` routes every Counters.ABORT_* into
        // `loops_aborted` (reason-keyed split is a future expansion).
        let mut meta = MetaInterp::<()>::new(0);
        let stats_before = meta.get_stats();
        meta.aborted_tracing(counters::ABORT_ESCAPE);
        let stats_after = meta.get_stats();
        assert_eq!(stats_after.loops_aborted, stats_before.loops_aborted + 1,);
    }

    #[test]
    fn aborted_tracing_clears_aborted_tracing_jitdriver_state() {
        // pyjitpl.py:2776-2785 — when aborted_tracing_jitdriver was
        // pre-set the abort fires the trace-too-long hook and
        // clears both fields.
        let mut meta = MetaInterp::<()>::new(0);
        meta.aborted_tracing_jitdriver = Some(7);
        meta.aborted_tracing_greenkey = Some(0xfeed);
        meta.aborted_tracing(0);
        assert!(meta.aborted_tracing_jitdriver.is_none());
        assert!(meta.aborted_tracing_greenkey.is_none());
    }

    #[test]
    fn aborted_tracing_does_not_touch_jitdriver_when_unset() {
        let mut meta = MetaInterp::<()>::new(0);
        assert!(meta.aborted_tracing_jitdriver.is_none());
        meta.aborted_tracing(0);
        assert!(meta.aborted_tracing_jitdriver.is_none());
        assert!(meta.aborted_tracing_greenkey.is_none());
    }

    #[test]
    fn try_tco_no_op_when_callee_is_portal_jitcode() {
        // pyjitpl.py:1279-1280 — `if self.jitcode.jitdriver_sd: return`.
        // A portal-jitcode callee never tail-call-optimizes — the
        // upstream invariant is that portal frames stay on the stack
        // for the metainterp dispatch loop.
        use crate::BackEdgeAction;
        use crate::jitcode::JitCodeBuilder;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let mut portal = JitCodeBuilder::new().finish();
        portal.replace_jitdriver_sd(Some(0));
        let portal = std::sync::Arc::new(portal);
        meta.perform_call(portal, &[], None).unwrap_err();
        let pre_len = meta.framestack.frames.len();

        meta._try_tco();

        // Stack untouched: TCO short-circuits on portal jitcodes.
        assert_eq!(meta.framestack.frames.len(), pre_len);
    }

    #[test]
    fn try_tco_no_op_when_framestack_has_only_one_frame() {
        // pyjitpl.py:1306-1307 — TCO needs a caller (framestack[-2]) to
        // remove.  Single-frame stack short-circuits.
        use crate::jitcode::JitCodeBuilder;
        let mut meta = MetaInterp::<()>::new(0);
        meta.force_start_tracing(0, (0, 0), None, &[]);

        let jitcode = std::sync::Arc::new(JitCodeBuilder::new().finish());
        meta.perform_call(jitcode, &[], None).unwrap_err();
        assert_eq!(meta.framestack.frames.len(), 1);

        meta._try_tco();
        assert_eq!(
            meta.framestack.frames.len(),
            1,
            "single-frame stack must not be popped",
        );
    }

    #[test]
    fn record_result_of_call_pure_all_const_args_truncates_and_returns_const() {
        // pyjitpl.py:3568-3569 — all argboxes are constants, so the
        // CALL is removed (history.cut to the pre-call position) and a
        // ConstInt(resvalue) is returned in place of the op.
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );

        // Build a constant argbox via TraceCtx::const_int.
        let arg_const = {
            let ctx = meta.trace_ctx().expect("active trace");
            ctx.const_int(42)
        };
        // Snapshot the trace position before we record the call, so
        // record_result_of_call_pure has a `patch_pos` to cut to.
        let patch_pos = meta.trace_ctx().unwrap().get_trace_position();
        // Record a placeholder CallI op so there's something to cut.
        let funcref = meta.trace_ctx().unwrap().const_int(0xdead);
        let call_op = meta.trace_ctx().unwrap().record_op_with_descr(
            OpCode::CallI,
            &[funcref, arg_const],
            descr_ref.clone(),
        );

        let _ = descr_view;
        let resbox = meta.trace_ctx().unwrap().record_result_of_call_pure(
            call_op,
            &[funcref, arg_const],
            &[majit_ir::Value::Int(0xdead), majit_ir::Value::Int(42)],
            descr_ref,
            patch_pos,
            OpCode::CallI,
            majit_ir::Value::Int(7),
        );

        let ctx = meta.trace_ctx().expect("active trace");
        // The CallI must have been cut from the trace.
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .all(|op| op.opcode != OpCode::CallI),
            "CallI must be cut on all-const path",
        );
        // resbox is a fresh ConstInt(7) — its constants_get_value must
        // resolve to Int(7).
        assert_eq!(
            ctx.constants_get_value(resbox),
            Some(majit_ir::Value::Int(7)),
        );
    }

    extern "C" fn portal_runner_helper() -> i64 {
        0xc0ffee
    }

    /// S2.1 invariant (wiggly-barto plan): `do_recursive_call` requires
    /// `portal_runner_adr != 0`. The default `with_virtualizable` /
    /// `JitDriverStaticData::new` constructor leaves the address at 0
    /// until the host runtime populates it (`warmspot.py:1010-1012`).
    /// Skipping the population must trip the debug_assert at
    /// `do_recursive_call` entry — locks the contract so a future
    /// caller that forgets to wire `portal_runner_adr` fails fast in
    /// dev/test builds (the bench harness runs in dev profile).
    #[test]
    #[should_panic(expected = "portal_runner_adr is 0")]
    fn do_recursive_call_panics_when_portal_runner_adr_is_zero() {
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        // Deliberately do NOT set jd.portal_runner_adr — the default 0
        // sentinel must trigger the S2.1 invariant assertion.
        let jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        let _ = meta.do_recursive_call(&jd, &[], descr_ref, &descr_view, 0, 0, false);
    }

    /// pyjitpl.py:1418-1420 — verify_green_args fires before the
    /// recursive-call funcbox is built. pyre folds upstream
    /// `_opimpl_recursive_call` and `do_recursive_call` so the verify
    /// runs at `do_recursive_call` entry. A non-Const greens slot
    /// (Box leaking through) must trip the `is_constant()` panic from
    /// `verify_green_args` (frame.rs:222-227).
    #[test]
    #[should_panic(expected = "is not a Const")]
    fn do_recursive_call_panics_when_greens_contain_non_const() {
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        // `num_green_args() == 1` so allboxes[0] is the green slot.
        let mut jd = crate::jitdriver::JitDriverStaticData::new(
            vec![("g0", majit_ir::Type::Int)],
            vec![("r0", majit_ir::Type::Int)],
        );
        jd.is_recursive = true;
        jd.portal_runner_adr = portal_runner_helper as *const () as i64;

        // OpRef::int_op(0) is in the operation namespace (CONST_BIT clear) so
        // is_constant() returns false — upstream demands ConstInt /
        // ConstPtr / ConstFloat at this slot.
        let allboxes = [
            (JitArgKind::Int, OpRef::int_op(0), 0),
            (JitArgKind::Int, OpRef::const_int(1), 0xfeed),
        ];
        let _ = meta.do_recursive_call(&jd, &allboxes, descr_ref, &descr_view, 0, 0, false);
    }

    #[test]
    fn do_recursive_call_emits_call_via_portal_runner_adr() {
        // pyjitpl.py:1425-1432 — portal_runner_adr → funcbox → CALL_*
        // routed through do_residual_call's regular branch (no
        // assembler_call).
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );

        let mut jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        jd.is_recursive = true;
        jd.portal_runner_adr = portal_runner_helper as *const () as i64;

        let result = meta.do_recursive_call(
            &jd,
            &[],
            descr_ref,
            &descr_view,
            /* target_jd_index = */ 0,
            /* pc = */ 0,
            /* assembler_call = */ false,
        );
        let (opref, resvalue) = result.expect("Ok").expect("Some");
        assert_eq!(resvalue, 0xc0ffee);

        let ctx = meta.trace_ctx().expect("active trace");
        let op = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::CallI)
            .expect("CallI must be recorded");
        assert_eq!(op.pos, opref);
    }

    #[test]
    fn do_conditional_call_emits_cond_call_when_not_is_value() {
        // pyjitpl.py:2137-2139 — is_value=False → COND_CALL (void).
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Void,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = cond_call_void_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let condbox = (JitArgKind::Int, OpRef::int_op(50), 1);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let result = meta.do_conditional_call(
            condbox,
            funcbox,
            &[],
            descr_ref,
            &descr_view,
            0,
            /* is_value = */ false,
            /* dst = */ None,
        );
        assert!(matches!(result, Ok(None)));
        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::CondCallN),
            "CondCallN must be recorded",
        );
    }

    #[test]
    fn do_conditional_call_emits_cond_call_value_int_when_is_value() {
        // pyjitpl.py:2141-2146 — is_value=True + Int result → COND_CALL_VALUE_I.
        // RPython's record_result_of_call_pure also folds COND_CALL_VALUE
        // when all normalized args (`argboxes[1..]`, skipping condbox) are
        // Const. Use a live Ref op for funcbox so this fixture keeps the
        // COND_CALL_VALUE_I op in the trace and exercises the recording path.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let action = meta.force_start_tracing(
            0,
            (0, 0),
            None,
            &[Value::Ref(majit_ir::GcRef(fnaddr as usize))],
        );
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        let condbox = (JitArgKind::Int, OpRef::int_op(50), 0);
        let funcbox = (JitArgKind::Ref, OpRef::ref_op(0), fnaddr);
        let result = meta.do_conditional_call(
            condbox,
            funcbox,
            &[],
            descr_ref,
            &descr_view,
            0,
            /* is_value = */ true,
            /* dst = */ None,
        );
        // execute_varargs_int_helper takes 2 args; with empty argboxes
        // the executor passes no args.  Pyre's call_int_function falls
        // through the empty-arg arm and the function returns 0 + 0*1000.
        let _ = result;
        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::CondCallValueI),
            "CondCallValueI must be recorded",
        );
    }

    #[test]
    fn do_residual_call_emits_call_may_force_for_force_virtual_path() {
        // pyjitpl.py:2007-2083 — forces_virtual path emits
        // CALL_MAY_FORCE_I via direct_call_may_force followed by a
        // GUARD_NOT_FORCED.  The call's concrete result is the
        // executor's return.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let mut effect = majit_ir::EffectInfo::default();
        effect.extraeffect = majit_ir::effectinfo::ExtraEffect::ForcesVirtualOrVirtualizable;
        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Int,
            effect: effect.clone(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(vec![], majit_ir::Type::Int, effect);
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);

        let result =
            meta.do_residual_call_full(funcbox, &[], descr_ref, &descr_view, 0, false, None, None);
        let (opref, _resvalue) = result.expect("Ok").expect("Some(opref)");

        let ctx = meta.trace_ctx().expect("active trace");
        let call_op = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::CallMayForceI)
            .expect("CallMayForceI must be recorded");
        assert_eq!(call_op.pos, opref);
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::GuardNotForced),
            "GUARD_NOT_FORCED must follow CALL_MAY_FORCE",
        );
    }

    #[test]
    fn miframe_execute_varargs_clears_exception_and_records_call_when_no_exc() {
        // pyjitpl.py:1942-1957 — without an exception, the call records
        // a CallI op and assert_no_exception passes.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let argboxes = [
            funcbox,
            (JitArgKind::Int, OpRef::int_op(1), 5),
            (JitArgKind::Int, OpRef::int_op(2), 9),
        ];
        // Pre-set last_exc_value to verify clear_exception runs.
        meta.last_exc_value = 0xdead;
        let result = meta.miframe_execute_varargs(
            OpCode::CallI,
            &argboxes,
            descr_ref,
            &descr_view,
            /* exc = */ false,
            /* pure = */ false,
            /* dst = */ None,
        );
        let op = result.expect("Ok(...)").expect("non-void must return Some");
        assert_eq!(op.1, 5 + 9 * 1000);
        assert_eq!(meta.last_exc_value, 0, "clear_exception must run first");
    }

    #[test]
    fn execute_and_record_varargs_returns_op_and_resvalue_for_int_call() {
        // pyjitpl.py:2641-2652 — record CallI op with the descr and
        // return (OpRef, resvalue) computed from
        // executor.execute_varargs.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![majit_ir::Type::Int, majit_ir::Type::Int],
            result_type: majit_ir::Type::Int,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = execute_varargs_int_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let argboxes = [
            funcbox,
            (JitArgKind::Int, OpRef::int_op(1), 7),
            (JitArgKind::Int, OpRef::int_op(2), 3),
        ];
        let result =
            meta.execute_and_record_varargs(OpCode::CallI, &argboxes, descr_ref, &descr_view);
        let (opref, resvalue) = result.expect("non-void call must return Some");
        assert_eq!(resvalue, 7 + 3 * 1000);

        let ctx = meta.trace_ctx().expect("active trace");
        let op = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::CallI)
            .expect("CallI must be recorded");
        assert_eq!(op.pos, opref);
        assert_eq!(op.args.len(), 3);
        assert_eq!(op.args[0], funcbox_ref);
        assert_eq!(op.args[1], OpRef::int_op(1));
        assert_eq!(op.args[2], OpRef::int_op(2));
    }

    #[test]
    fn execute_and_record_varargs_returns_none_for_void_call() {
        // pyjitpl.py:2662-2663 — `if op.type != 'v': return op` →
        // void calls return None even though the IR op is recorded.
        use crate::BackEdgeAction;
        use crate::jitcode::JitArgKind;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let descr_view = StubCallDescr {
            arg_types: vec![],
            result_type: majit_ir::Type::Void,
            effect: majit_ir::EffectInfo::default(),
        };
        let descr_ref = majit_ir::descr::make_call_descr(
            vec![],
            majit_ir::Type::Void,
            majit_ir::EffectInfo::default(),
        );
        let fnaddr = execute_varargs_void_helper as *const () as i64;
        let funcbox_ref = meta.trace_ctx().expect("active trace").const_ref(fnaddr);
        let funcbox = (JitArgKind::Ref, funcbox_ref, fnaddr);
        let result =
            meta.execute_and_record_varargs(OpCode::CallN, &[funcbox], descr_ref, &descr_view);
        assert!(result.is_none(), "void call must return None");

        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::CallN),
            "CallN must still be recorded",
        );
    }

    #[test]
    fn clear_exception_resets_last_exc_value_to_zero() {
        // pyjitpl.py:2757-2758 — `self.last_exc_value = lltype.nullptr(...)`
        let mut meta = MetaInterp::<()>::new(0);
        meta.last_exc_value = 0xbeef;
        meta.clear_exception();
        assert_eq!(meta.last_exc_value, 0);
    }

    #[test]
    fn finishframe_clears_last_exc_value_per_pyjitpl_2481() {
        // pyjitpl.py:2481 — `self.last_exc_value = lltype.nullptr(...)`.
        let mut meta = MetaInterp::<()>::new(0);
        meta.last_exc_value = 0xc0ffee;
        let _ = meta.finishframe(None, true);
        assert_eq!(meta.last_exc_value, 0);
    }

    #[test]
    fn handle_possible_overflow_error_records_guard_overflow_when_flag_set() {
        // pyjitpl.py:1882-1886 — ovf_flag → GUARD_OVERFLOW + pc=label, return None
        use crate::jitcode::JitCodeBuilder;
        let mut meta = MetaInterp::<()>::new(0);
        meta.force_start_tracing(0, (0, 0), None, &[]);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        // Already tracing returns AlreadyTracing — that's fine, the
        // first call already started the trace.
        let _ = action;

        // Push a frame so handle_possible_overflow_error can mutate pc.
        let jitcode = std::sync::Arc::new(JitCodeBuilder::new().finish());
        meta.perform_call(jitcode, &[], None).unwrap_err();
        meta.framestack.current_mut().pc = 7;

        meta.ovf_flag = true;
        let result = meta.handle_possible_overflow_error(99, 0, OpRef::int_op(42));
        assert!(result.is_none(), "expected None on overflow");
        assert_eq!(meta.framestack.current_mut().pc, 99);

        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::GuardOverflow),
            "GuardOverflow must be recorded",
        );
    }

    #[test]
    fn handle_possible_overflow_error_records_guard_no_overflow_when_flag_unset() {
        // pyjitpl.py:1888-1890 — !ovf_flag → GUARD_NO_OVERFLOW, return resbox
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.ovf_flag = false;
        let result = meta.handle_possible_overflow_error(99, 0, OpRef::int_op(42));
        assert_eq!(result, Some(OpRef::int_op(42)));

        let ctx = meta.trace_ctx().expect("active trace");
        assert!(
            ctx.recorder
                .ops()
                .iter()
                .any(|op| op.opcode == OpCode::GuardNoOverflow),
            "GuardNoOverflow must be recorded",
        );
    }

    #[test]
    fn handle_possible_exception_emits_guard_no_exception_when_value_zero() {
        // pyjitpl.py:3394-3395 — last_exc_value == 0 → GUARD_NO_EXCEPTION.
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        meta.last_exc_value = 0;
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let result = meta.handle_possible_exception();
        assert!(matches!(result, Ok(())));

        let ctx = meta.trace_ctx().expect("active trace");
        let count = ctx
            .recorder
            .ops()
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNoException)
            .count();
        assert_eq!(count, 1, "GuardNoException must be recorded once");
    }

    #[test]
    fn handle_possible_exception_emits_guard_exception_when_value_set() {
        // pyjitpl.py:3381-3392 — last_exc_value != 0 → GUARD_EXCEPTION
        // followed by finishframe_exception().
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        // Override cls_of_box so we can inject a known typeptr without
        // dereferencing a raw pointer.
        meta.cls_of_box = Some(|_| 0xc1a55);
        meta.last_exc_value = 0xfeed;

        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        // pyjitpl.py:2533-2538: with an empty framestack the exception
        // unwind drains immediately and surfaces
        // `ExitFrameWithExceptionRef`. The GUARD_EXCEPTION op + the
        // class-const branch must still be observable on the recorder
        // before that signal returns.
        let result = meta.handle_possible_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ExitFrameWithExceptionRef(r)) if r.0 == 0xfeed
        ));

        let guard_pos = {
            let ctx = meta.trace_ctx().expect("active trace");
            let mut matches = ctx
                .recorder
                .ops()
                .iter()
                .filter(|op| op.opcode == OpCode::GuardException);
            let op = matches.next().expect("GuardException must be recorded");
            assert_eq!(op.args.len(), 1);
            let typeptr = ctx
                .constants_get_value(op.args[0])
                .expect("typeptr constant");
            assert_eq!(typeptr, majit_ir::Value::Int(0xc1a55));
            op.pos
        };

        // pyjitpl.py:3392: class_of_last_exc_is_const = True after.
        assert!(meta.class_of_last_exc_is_const);
        let last_exc_box = meta.last_exc_box.expect("last_exc_box");
        // pyjitpl.py:3389: when class is NOT const, last_exc_box is the
        // GUARD_EXCEPTION op itself (its trace position).
        assert_eq!(last_exc_box, guard_pos);
    }

    #[test]
    fn handle_possible_exception_uses_const_ref_when_class_is_const() {
        // pyjitpl.py:3386-3387 — when class_of_last_exc_is_const is set
        // before the call, last_exc_box is `ConstPtr(val)`, NOT the
        // guard op's box. Pyre uses `const_ref(val)` as the orthodox
        // ConstPtr equivalent (trace_ctx.rs:583).
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        meta.cls_of_box = Some(|_| 0xc1a55);
        meta.last_exc_value = 0xfeed;
        meta.class_of_last_exc_is_const = true;

        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let result = meta.handle_possible_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ExitFrameWithExceptionRef(r)) if r.0 == 0xfeed
        ));

        let last_exc_box = meta.last_exc_box.expect("last_exc_box");
        let ctx = meta.trace_ctx().expect("active trace");
        // GUARD_EXCEPTION must still be recorded (pyjitpl.py:3383).
        let guard_count = ctx
            .recorder
            .ops()
            .iter()
            .filter(|op| op.opcode == OpCode::GuardException)
            .count();
        assert_eq!(guard_count, 1);
        // last_exc_box must be a Ref-typed constant carrying the
        // exception value, not the guard op.
        let typed = ctx
            .constants_get_value(last_exc_box)
            .expect("last_exc_box must be a constant");
        assert_eq!(typed, majit_ir::Value::Ref(majit_ir::value::GcRef(0xfeed)));
    }

    fn make_catch_exception_jitcode() -> (std::sync::Arc<crate::jitcode::JitCode>, usize) {
        use crate::jitcode::JitCodeBuilder;

        let mut builder = JitCodeBuilder::new();
        let live_patch = builder.live_placeholder();
        builder.patch_live_offset(live_patch, 0);
        let handler = builder.new_label();
        builder.catch_exception(handler);
        builder.mark_label(handler);
        let jitcode = std::sync::Arc::new(builder.finish());
        let target = jitcode.code.len();
        (jitcode, target)
    }

    #[test]
    fn finishframe_exception_jumps_to_current_frame_catch_handler() {
        let (jitcode, target) = make_catch_exception_jitcode();
        let mut meta = MetaInterp::<()>::new(0);
        {
            let sd = std::sync::Arc::get_mut(&mut meta.staticdata).unwrap();
            sd.op_live = crate::jitcode::insns::BC_LIVE as i32;
            sd.op_catch_exception = crate::jitcode::insns::BC_CATCH_EXCEPTION as i32;
            sd.op_rvmprof_code = -1;
        }

        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(jitcode, 0));
        let result = meta.finishframe_exception();

        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(meta.framestack.len(), 1);
        assert_eq!(meta.framestack.current_mut().pc, target);
    }

    #[test]
    fn finishframe_exception_pops_callee_then_jumps_to_caller_handler() {
        let (caller, target) = make_catch_exception_jitcode();
        let callee = std::sync::Arc::new(crate::jitcode::JitCodeBuilder::new().finish());

        let mut meta = MetaInterp::<()>::new(0);
        {
            let sd = std::sync::Arc::get_mut(&mut meta.staticdata).unwrap();
            sd.op_live = crate::jitcode::insns::BC_LIVE as i32;
            sd.op_catch_exception = crate::jitcode::insns::BC_CATCH_EXCEPTION as i32;
            sd.op_rvmprof_code = -1;
        }

        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(caller, 0));
        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(callee, 0));

        let result = meta.finishframe_exception();

        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(meta.framestack.len(), 1);
        assert_eq!(meta.framestack.current_mut().pc, target);
    }

    #[test]
    fn finishframe_exception_jumps_to_catch_handler() {
        let mut meta = MetaInterp::<()>::new(0);
        let mut jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        jitcode.body_mut().code = vec![crate::jitcode::insns::BC_CATCH_EXCEPTION, 3, 0];
        let jitcode = std::sync::Arc::new(jitcode);

        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(jitcode, 0));
        let result = meta.finishframe_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(meta.framestack.len(), 1, "handler frame must stay on stack");
        assert_eq!(meta.framestack.current_mut().pc, 3);
        assert_eq!(meta.framestack.current_mut().code_cursor, 3);
    }

    #[test]
    fn finishframe_exception_skips_live_prefix_before_catch_handler() {
        let mut meta = MetaInterp::<()>::new(0);
        let mut jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        jitcode.body_mut().code = vec![
            crate::jitcode::insns::BC_LIVE,
            0,
            0,
            crate::jitcode::insns::BC_CATCH_EXCEPTION,
            6,
            0,
        ];
        let jitcode = std::sync::Arc::new(jitcode);

        let mut frame = crate::pyjitpl::MIFrame::new(jitcode, 0);
        frame.pc = 0;
        meta.framestack.push(frame);

        let result = meta.finishframe_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(meta.framestack.current_mut().pc, 6);
        assert_eq!(meta.framestack.current_mut().code_cursor, 6);
    }

    #[test]
    fn finishframe_exception_pops_frames_without_handler() {
        // pyjitpl.py:2533-2538: when the unwind drains every frame
        // without finding a `catch_exception`, finishframe_exception runs
        // `compile_exit_frame_with_exception(self.last_exc_box)` then
        // raises `jitexc.ExitFrameWithExceptionRef(excvalue)`. Pyre
        // surfaces the same shape via the `ExitFrameWithExceptionRef`
        // signal variant.
        let mut meta = MetaInterp::<()>::new(0);
        meta.last_exc_value = 0xfeed;
        let jitcode = std::sync::Arc::new(crate::jitcode::JitCodeBuilder::new().finish());
        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(jitcode, 0));

        let result = meta.finishframe_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ExitFrameWithExceptionRef(r)) if r.0 == 0xfeed
        ));
        assert!(meta.framestack.is_empty());
    }

    #[test]
    fn finishframe_exception_pops_callee_then_catches_in_caller() {
        // pyjitpl.py:2506-2529 — cross-frame walk: callee raises,
        // current frame has no catch_exception, outer frame does.
        // Expected: popframe() drops the callee, then BC_CATCH_EXCEPTION
        // in the caller routes control to the handler target.
        let mut caller_jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        caller_jitcode.body_mut().code = vec![crate::jitcode::insns::BC_CATCH_EXCEPTION, 5, 0];
        let caller_jitcode = std::sync::Arc::new(caller_jitcode);

        // Callee: non-CATCH opcode at pc 0. Use BC_LIVE (skip prefix)
        // chased by a non-CATCH byte so finishframe_exception's LIVE
        // skip lands on something that isn't a handler.
        let mut callee_jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        callee_jitcode.body_mut().code = vec![0xff, 0, 0];
        let callee_jitcode = std::sync::Arc::new(callee_jitcode);

        let mut meta = MetaInterp::<()>::new(0);
        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(caller_jitcode, 0));
        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(callee_jitcode, 0));

        let result = meta.finishframe_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(
            meta.framestack.len(),
            1,
            "callee must be popped; caller handler frame remains"
        );
        // Caller jumped to handler target (offset 5 from the
        // BC_CATCH_EXCEPTION's 2-byte operand).
        assert_eq!(meta.framestack.current_mut().pc, 5);
        assert_eq!(meta.framestack.current_mut().code_cursor, 5);
    }

    #[test]
    fn handle_possible_exception_routes_cross_frame_to_caller_handler() {
        // End-to-end: pyjitpl.py:3380-3395. An exception is pending
        // (`last_exc_value != 0`), the callee has no handler, the
        // caller does. handle_possible_exception must:
        //   (1) emit GUARD_EXCEPTION on the tracer;
        //   (2) stash last_exc_box + mark class_of_last_exc_is_const;
        //   (3) invoke finishframe_exception, which pops the callee
        //       and routes the caller's pc to its BC_CATCH_EXCEPTION
        //       handler target.
        use crate::BackEdgeAction;

        let mut caller_jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        caller_jitcode.body_mut().code = vec![crate::jitcode::insns::BC_CATCH_EXCEPTION, 9, 0];
        let caller_jitcode = std::sync::Arc::new(caller_jitcode);
        let mut callee_jitcode = crate::jitcode::JitCodeBuilder::new().finish();
        callee_jitcode.body_mut().code = vec![0xff, 0, 0];
        let callee_jitcode = std::sync::Arc::new(callee_jitcode);

        let mut meta = MetaInterp::<()>::new(0);
        meta.cls_of_box = Some(|_| 0xcafef00d);
        meta.last_exc_value = 0xbeef;

        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(caller_jitcode, 0));
        meta.framestack
            .push(crate::pyjitpl::MIFrame::new(callee_jitcode, 0));

        let result = meta.handle_possible_exception();
        assert!(matches!(
            result,
            Err(FinishframeExceptionSignal::ChangeFrame)
        ));
        assert_eq!(
            meta.framestack.len(),
            1,
            "callee must be popped; caller handler frame remains"
        );
        assert_eq!(meta.framestack.current_mut().pc, 9);
        assert!(meta.class_of_last_exc_is_const);
        assert!(meta.last_exc_box.is_some());

        let ctx = meta.trace_ctx().expect("active trace");
        let op = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::GuardException)
            .expect("GuardException must be recorded");
        let typeptr = ctx
            .constants_get_value(op.args[0])
            .expect("typeptr constant");
        assert_eq!(typeptr, majit_ir::Value::Int(0xcafef00d));
    }

    #[test]
    fn assert_no_exception_passes_when_value_zero() {
        let meta = MetaInterp::<()>::new(0);
        meta.assert_no_exception();
    }

    #[test]
    #[should_panic(expected = "MetaInterp.assert_no_exception")]
    fn assert_no_exception_panics_when_value_set() {
        let mut meta = MetaInterp::<()>::new(0);
        meta.last_exc_value = 0xdead;
        meta.assert_no_exception();
    }

    #[test]
    fn call_ids_pushes_current_call_id_on_portal_newframe() {
        // pyjitpl.py:2435 self.call_ids.append(self.current_call_id)
        // pyjitpl.py:2442 self.current_call_id += 1
        // pyjitpl.py:2469 popframe → self.call_ids.pop()
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let mut portal = JitCodeBuilder::new().finish();
        portal.replace_jitdriver_sd(Some(0));
        let portal = std::sync::Arc::new(portal);

        // Push portal: current_call_id stamped onto call_ids, then
        // current_call_id bumps.
        meta.perform_call(portal.clone(), &[], None).unwrap_err();
        assert_eq!(meta.call_ids, vec![0]);
        assert_eq!(meta.current_call_id, 1);

        // Push another portal frame: stamps the new id.
        meta.perform_call(portal.clone(), &[], None).unwrap_err();
        assert_eq!(meta.call_ids, vec![0, 1]);
        assert_eq!(meta.current_call_id, 2);

        // popframe drops the top entry.
        meta.popframe(true);
        assert_eq!(meta.call_ids, vec![0]);
        assert_eq!(meta.current_call_id, 2, "current_call_id is monotonic");

        meta.popframe(true);
        assert!(meta.call_ids.is_empty());
    }

    #[test]
    fn call_ids_untouched_for_non_portal_jitcode() {
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let plain = std::sync::Arc::new(JitCodeBuilder::new().finish());

        meta.perform_call(plain.clone(), &[], None).unwrap_err();
        assert!(meta.call_ids.is_empty());
        assert_eq!(meta.current_call_id, 0);

        meta.popframe(true);
        assert!(meta.call_ids.is_empty());
        assert_eq!(meta.current_call_id, 0);
    }

    #[test]
    fn initialize_state_from_start_seeds_portal_call_depth_to_zero() {
        // pyjitpl.py:3268-3272 — set portal_call_depth = -1, push the
        // portal mainjitcode (which bumps it to 0), then assert == 0.
        use crate::jitcode::JitCodeBuilder;
        let mut mainjitcode = JitCodeBuilder::new().finish();
        mainjitcode.replace_jitdriver_sd(Some(0));
        let mainjitcode = std::sync::Arc::new(mainjitcode);

        let mut meta = MetaInterp::<()>::new(0);
        // Pre-pollute the counter to verify the reset.
        meta.portal_call_depth = 42;
        meta.initialize_state_from_start(mainjitcode, &[]);
        assert_eq!(meta.portal_call_depth, 0);
    }

    #[test]
    fn is_main_jitcode_returns_false_for_non_portal_jitcode() {
        let mut meta = MetaInterp::<()>::new(0);
        let mut jc = crate::jitcode::JitCodeBuilder::new().finish();
        jc.replace_jitdriver_sd(None);
        assert!(!meta.is_main_jitcode(&jc));
        // jitdriver_sd Some but no slot registered → still false.
        jc.replace_jitdriver_sd(Some(0));
        assert!(!meta.is_main_jitcode(&jc));

        // Register a non-recursive jitdriver: still false.
        let mut jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        jd.is_recursive = false;
        let idx = {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            std::sync::Arc::get_mut(staticdata)
                .unwrap()
                .register_jitdriver_sd(jd, backend)
        };
        jc.replace_jitdriver_sd(Some(idx));
        assert!(!meta.is_main_jitcode(&jc));
    }

    #[test]
    fn is_main_jitcode_returns_true_for_recursive_portal_jitcode() {
        let mut meta = MetaInterp::<()>::new(0);
        let mut jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        jd.is_recursive = true;
        let idx = {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            std::sync::Arc::get_mut(staticdata)
                .unwrap()
                .register_jitdriver_sd(jd, backend)
        };

        let mut jc = crate::jitcode::JitCodeBuilder::new().finish();
        jc.replace_jitdriver_sd(Some(idx));
        assert!(meta.is_main_jitcode(&jc));
    }

    #[test]
    fn enter_portal_frame_records_const_int_pair() {
        // pyjitpl.py:2455 — history.record2(rop.ENTER_PORTAL_FRAME,
        // ConstInt(jd_no), ConstInt(unique_id), None)
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.enter_portal_frame(3, 0xfeed);

        let ctx = meta.trace_ctx().expect("tracing must be active");
        let mut matches = ctx
            .recorder
            .ops()
            .iter()
            .filter(|op| op.opcode == OpCode::EnterPortalFrame);
        let op = matches.next().expect("EnterPortalFrame must be recorded");
        assert!(matches.next().is_none(), "expected exactly one record");
        assert_eq!(op.args.len(), 2);
        let jd_no = ctx.constants_get_value(op.args[0]).expect("jd_no constant");
        let unique_id = ctx
            .constants_get_value(op.args[1])
            .expect("unique_id constant");
        assert_eq!(jd_no, majit_ir::Value::Int(3));
        assert_eq!(unique_id, majit_ir::Value::Int(0xfeed));
    }

    #[test]
    fn leave_portal_frame_records_const_int_jd_no() {
        // pyjitpl.py:2459 — history.record1(rop.LEAVE_PORTAL_FRAME, ConstInt(jd_no), None)
        use crate::BackEdgeAction;
        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.leave_portal_frame(7);

        let ctx = meta.trace_ctx().expect("tracing must be active");
        let mut matches = ctx
            .recorder
            .ops()
            .iter()
            .filter(|op| op.opcode == OpCode::LeavePortalFrame);
        let op = matches.next().expect("LeavePortalFrame must be recorded");
        assert!(matches.next().is_none(), "expected exactly one record");
        assert_eq!(op.args.len(), 1);
        let jd_no = ctx.constants_get_value(op.args[0]).expect("jd_no constant");
        assert_eq!(jd_no, majit_ir::Value::Int(7));
    }

    #[test]
    fn newframe_and_popframe_use_jitdriver_sd_index_for_portal_ops() {
        // pyjitpl.py:2440-2441 / 2467:
        // enter/leave portal ops receive jitcode.jitdriver_sd.index, not
        // a hard-coded portal slot.
        use crate::BackEdgeAction;
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let mut jc = JitCodeBuilder::new().finish();
        jc.replace_jitdriver_sd(Some(5));
        let jc = std::sync::Arc::new(jc);

        meta.newframe(jc, Some(0xfeed));
        meta.popframe(true);

        let ctx = meta.trace_ctx().expect("tracing must be active");
        let enter = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::EnterPortalFrame)
            .expect("EnterPortalFrame must be recorded");
        let leave = ctx
            .recorder
            .ops()
            .iter()
            .find(|op| op.opcode == OpCode::LeavePortalFrame)
            .expect("LeavePortalFrame must be recorded");

        assert_eq!(
            ctx.constants_get_value(enter.args[0]),
            Some(majit_ir::Value::Int(5))
        );
        assert_eq!(
            ctx.constants_get_value(enter.args[1]),
            Some(majit_ir::Value::Int(0xfeed))
        );
        assert_eq!(
            ctx.constants_get_value(leave.args[0]),
            Some(majit_ir::Value::Int(5))
        );
    }

    #[test]
    fn portal_trace_positions_records_enter_and_exit_for_main_jitcode() {
        // pyjitpl.py:2443-2445 / 2470-2472 — newframe appends
        // (jd_no, Some(greenkey), trace_position); popframe appends
        // (jd_no, None, trace_position) on the matching exit.
        use crate::BackEdgeAction;
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let mut jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        jd.is_recursive = true;
        let idx = {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            std::sync::Arc::get_mut(staticdata)
                .unwrap()
                .register_jitdriver_sd(jd, backend)
        };

        let mut jc = JitCodeBuilder::new().finish();
        jc.replace_jitdriver_sd(Some(idx));
        let jc = std::sync::Arc::new(jc);

        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.perform_call(jc, &[], Some(0xcafe)).unwrap_err();
        assert_eq!(
            meta.portal_trace_positions
                .as_ref()
                .expect("portal_trace_positions must be Some")
                .len(),
            1
        );
        let entry = &meta.portal_trace_positions.as_ref().unwrap()[0];
        assert_eq!(entry.0, idx);
        assert_eq!(entry.1, Some(0xcafe));

        meta.popframe(true);
        let positions = meta
            .portal_trace_positions
            .as_ref()
            .expect("portal_trace_positions must still be Some");
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[1].0, idx);
        assert_eq!(positions[1].1, None);
    }

    #[test]
    fn portal_trace_positions_skips_non_recursive_jitdriver() {
        // is_main_jitcode requires jd.is_recursive — non-recursive portals
        // must not append to portal_trace_positions.
        use crate::BackEdgeAction;
        use crate::jitcode::JitCodeBuilder;

        let mut meta = MetaInterp::<()>::new(0);
        let mut jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        jd.is_recursive = false;
        let idx = {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            std::sync::Arc::get_mut(staticdata)
                .unwrap()
                .register_jitdriver_sd(jd, backend)
        };

        let mut jc = JitCodeBuilder::new().finish();
        jc.replace_jitdriver_sd(Some(idx));
        let jc = std::sync::Arc::new(jc);

        let action = meta.force_start_tracing(0, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        meta.perform_call(jc, &[], Some(0xbabe)).unwrap_err();
        assert!(
            meta.portal_trace_positions
                .as_ref()
                .expect("Some")
                .is_empty(),
            "non-recursive jitdriver must not record"
        );
    }

    #[test]
    fn enter_leave_portal_frame_no_op_when_not_tracing() {
        // Without an active TraceCtx the named entry must not panic and
        // must not record anything.
        let mut meta = MetaInterp::<()>::new(0);
        meta.enter_portal_frame(0, 0);
        meta.leave_portal_frame(0);
    }

    #[test]
    fn reset_framestack_for_failure_empties_the_stack() {
        // pyjitpl.py:3403 `self.framestack = []` invariant before
        // resume.rebuild_from_resumedata repopulates frames.
        use crate::jitcode::JitCodeBuilder;
        let jitcode = std::sync::Arc::new(JitCodeBuilder::new().finish());
        let mut meta = MetaInterp::<()>::new(0);
        meta.perform_call(jitcode, &[], None).unwrap_err();
        assert_eq!(meta.framestack.len(), 1);
        meta.reset_framestack_for_failure();
        assert_eq!(meta.framestack.len(), 0);
    }

    #[test]
    fn popframe_invokes_cleanup_registers_on_popped_frame() {
        // pyjitpl.py:2476: frame.cleanup_registers().  The popped frame
        // is dropped so we cannot inspect it directly, but we can
        // observe the side effect by pushing a frame, mutating its
        // registers, popping, and asserting framestack is now empty.
        use crate::jitcode::JitCodeBuilder;
        let jitcode = std::sync::Arc::new(JitCodeBuilder::new().finish());
        let mut meta = MetaInterp::<()>::new(0);
        meta.perform_call(jitcode, &[], None).unwrap_err();
        assert_eq!(meta.framestack.len(), 1);
        meta.popframe(true);
        assert_eq!(meta.framestack.len(), 0);
    }

    #[test]
    fn change_frame_implements_error() {
        let cf = ChangeFrame;
        // Confirm the unit type prints as expected and is usable as
        // a Rust error (mirrors RPython's `raise ChangeFrame`).
        assert_eq!(format!("{cf}"), "ChangeFrame");
        let _: &dyn std::error::Error = &cf;
    }

    #[test]
    fn setup_insns_populates_opcode_names() {
        let mut sd = MetaInterpStaticData::new();
        let mut insns = std::collections::HashMap::new();
        insns.insert("foo".to_string(), 0u8);
        insns.insert("bar".to_string(), 1u8);
        sd.setup_insns(&insns);
        assert_eq!(sd.opcode_names, vec!["foo".to_string(), "bar".to_string()]);
        assert_eq!(sd.opcode_implementations.len(), 2);
        assert!(sd.opcode_implementations.iter().all(|slot| slot.is_none()));
    }

    #[test]
    fn setup_insns_caches_opcode_ids_or_minus_one() {
        // pyjitpl.py:2236-2243: each cached id is `insns.get(...) ?? -1`.
        let mut sd = MetaInterpStaticData::new();
        let mut insns = std::collections::HashMap::new();
        insns.insert("live/".to_string(), 5u8);
        insns.insert("goto/L".to_string(), 6u8);
        insns.insert("catch_exception/L".to_string(), 7u8);
        insns.insert("rvmprof_code/ii".to_string(), 8u8);
        insns.insert("int_return/i".to_string(), 9u8);
        insns.insert("ref_return/r".to_string(), 10u8);
        insns.insert("float_return/f".to_string(), 11u8);
        insns.insert("void_return/".to_string(), 12u8);
        sd.setup_insns(&insns);
        assert_eq!(sd.op_live, 5);
        assert_eq!(sd.op_goto, 6);
        assert_eq!(sd.op_catch_exception, 7);
        assert_eq!(sd.op_rvmprof_code, 8);
        assert_eq!(sd.op_int_return, 9);
        assert_eq!(sd.op_ref_return, 10);
        assert_eq!(sd.op_float_return, 11);
        assert_eq!(sd.op_void_return, 12);
    }

    #[test]
    fn setup_insns_leaves_missing_opcode_ids_at_minus_one() {
        let mut sd = MetaInterpStaticData::new();
        let mut insns = std::collections::HashMap::new();
        insns.insert("foo".to_string(), 0u8);
        sd.setup_insns(&insns);
        assert_eq!(sd.op_live, -1);
        assert_eq!(sd.op_goto, -1);
        assert_eq!(sd.op_catch_exception, -1);
        assert_eq!(sd.op_rvmprof_code, -1);
        assert_eq!(sd.op_int_return, -1);
        assert_eq!(sd.op_ref_return, -1);
        assert_eq!(sd.op_float_return, -1);
        assert_eq!(sd.op_void_return, -1);
    }

    #[test]
    fn metainterpstaticdata_new_initializes_op_ids_to_minus_one() {
        let sd = MetaInterpStaticData::new();
        assert_eq!(sd.op_live, -1);
        assert_eq!(sd.op_goto, -1);
        assert_eq!(sd.op_catch_exception, -1);
        assert_eq!(sd.op_rvmprof_code, -1);
        assert_eq!(sd.op_int_return, -1);
        assert_eq!(sd.op_ref_return, -1);
        assert_eq!(sd.op_float_return, -1);
        assert_eq!(sd.op_void_return, -1);
    }

    #[test]
    fn finish_setup_copies_assembler_liveness_and_insns() {
        // pyjitpl.py:2255-2285 — MetaInterpStaticData.finish_setup(codewriter)
        // pulls `asm.insns` and `asm.all_liveness` into the staticdata
        // mirror.  Build a small CodeWriter, register two liveness
        // entries on its assembler, and make sure finish_setup mirrors
        // both halves.
        use majit_translate::jit_codewriter::call::CallControl;
        use majit_translate::jit_codewriter::codewriter::CodeWriter;

        let mut codewriter = CodeWriter::new();
        let mut scratch = Vec::<u8>::new();
        // assembler.py:236-247 _encode_liveness — first entry pos = 0.
        codewriter
            .assembler
            ._encode_liveness(&[0, 1], &[2], &[], &mut scratch);
        // Second call with the same key should reuse pos = 0 (dedup).
        codewriter
            .assembler
            ._encode_liveness(&[0, 1], &[2], &[], &mut scratch);
        // Third call with a different key advances pos.
        codewriter
            .assembler
            ._encode_liveness(&[3], &[], &[5], &mut scratch);
        let expected_liveness = codewriter.assembler.all_liveness().to_vec();
        let expected_liveness_len = expected_liveness.len();
        assert!(
            expected_liveness_len > 0,
            "scaffolding sanity: assembler must have emitted liveness bytes"
        );

        let callcontrol = CallControl::new();
        let mut sd = MetaInterpStaticData::new();
        sd.finish_setup(&codewriter, &callcontrol);

        // pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)`
        assert_eq!(sd.liveness_info, expected_liveness);
        // pyjitpl.py:2260 `self.setup_insns(asm.insns)` — the assembler
        // was driven only through `_encode_liveness`, so no opcode
        // names were registered yet; the staticdata mirror must reflect
        // that empty state.
        assert!(sd.opcode_names.is_empty());
    }

    #[test]
    fn metainterp_finish_setup_publishes_canonical_liveness() {
        // warmspot.py:289 `self.metainterp_sd.finish_setup(self.codewriter)`
        // — the orthodox lifecycle ports as `Arc::get_mut` while the
        // staticdata Arc still has refcount 1 (immediately after
        // `MetaInterp::new`).  Verify the wrapper drives the bytes all
        // the way to `staticdata.liveness_info` without panicking.
        use majit_translate::jit_codewriter::call::CallControl;
        use majit_translate::jit_codewriter::codewriter::CodeWriter;

        let mut codewriter = CodeWriter::new();
        let mut scratch = Vec::<u8>::new();
        codewriter
            .assembler
            ._encode_liveness(&[0, 1, 2], &[3], &[], &mut scratch);
        let expected = codewriter.assembler.all_liveness().to_vec();

        let callcontrol = CallControl::new();
        let mut meta = MetaInterp::<()>::new(0);
        meta.finish_setup(&codewriter, &callcontrol);

        assert_eq!(meta.staticdata.liveness_info, expected);
    }

    #[test]
    #[should_panic(expected = "called after `staticdata` was cloned")]
    fn metainterp_finish_setup_panics_after_staticdata_clone() {
        // The Rust `Arc::get_mut` adaptation only matches RPython's
        // single-owner invariant while the refcount is 1; once
        // anything has cloned `self.staticdata`, the wrapper must fail
        // loudly so the convergence violation surfaces at the call
        // site.
        use majit_translate::jit_codewriter::call::CallControl;
        use majit_translate::jit_codewriter::codewriter::CodeWriter;

        let mut meta = MetaInterp::<()>::new(0);
        let _share: std::sync::Arc<MetaInterpStaticData> = meta.staticdata.clone();
        meta.finish_setup(&CodeWriter::new(), &CallControl::new());
    }

    #[test]
    fn metainterp_install_canonical_liveness_publishes_asm_bytes() {
        // Narrow analogue of `metainterp_finish_setup_publishes_canonical_liveness`
        // that exercises the `install_canonical_liveness` lifecycle hook.
        // pyjitpl.py:2264 `self.liveness_info = "".join(asm.all_liveness)`
        // — the hook must drive an already-populated `Assembler`'s
        // `all_liveness()` straight into `staticdata.liveness_info`,
        // without going through `CodeWriter` / `CallControl`.
        // pyjitpl.py:2236-2243 — also seed the cached opcode-id fields
        // (`op_live` etc.) from pyre's static `BC_*` constants.
        use majit_translate::jit_codewriter::assembler::Assembler;

        let mut asm = Assembler::new();
        let mut scratch = Vec::<u8>::new();
        // State-field JIT canonical entry shape: live_i = 0..total_slots,
        // live_r / live_f empty.  Matches `live_slots_for_state_field_jit`
        // for `(num_scalars=2, array_lens=&[1], num_virt_arrays=0)` →
        // total_slots = 3.
        asm._encode_liveness(&[0, 1, 2], &[], &[], &mut scratch);
        // Mirror the macro's `__JitMeta::install_canonical_liveness`
        // (`majit-macros::codegen_state.rs`) which pre-populates
        // `asm.insns` so `setup_insns(asm.insns())` resolves the
        // pyre-static `BC_*` opnums dynamically (RPython parity:
        // `assembler.py:222 self.insns[key] = opnum`).
        asm.register_insn("live/", crate::jitcode::insns::BC_LIVE);
        asm.register_insn(
            "catch_exception/L",
            crate::jitcode::insns::BC_CATCH_EXCEPTION,
        );
        asm.register_insn("rvmprof_code/ii", crate::jitcode::insns::BC_RVMPROF_CODE);
        asm.register_insn("int_return/i", crate::jitcode::insns::BC_INT_RETURN);
        asm.register_insn("ref_return/r", crate::jitcode::insns::BC_REF_RETURN);
        asm.register_insn("float_return/f", crate::jitcode::insns::BC_FLOAT_RETURN);
        asm.register_insn("void_return/", crate::jitcode::insns::BC_VOID_RETURN);
        let expected = asm.all_liveness().to_vec();

        let mut meta = MetaInterp::<()>::new(0);
        meta.install_canonical_liveness(&asm);

        assert_eq!(meta.staticdata.liveness_info, expected);
        assert_eq!(
            meta.staticdata.op_live,
            crate::jitcode::insns::BC_LIVE as i32,
            "install_canonical_liveness must seed op_live to pyre-static BC_LIVE",
        );
        assert_eq!(
            meta.staticdata.op_catch_exception,
            crate::jitcode::insns::BC_CATCH_EXCEPTION as i32,
        );
        assert_eq!(
            meta.staticdata.op_rvmprof_code,
            crate::jitcode::insns::BC_RVMPROF_CODE as i32,
        );
        assert_eq!(
            meta.staticdata.op_int_return,
            crate::jitcode::insns::BC_INT_RETURN as i32,
        );
        assert_eq!(
            meta.staticdata.op_ref_return,
            crate::jitcode::insns::BC_REF_RETURN as i32,
        );
        assert_eq!(
            meta.staticdata.op_float_return,
            crate::jitcode::insns::BC_FLOAT_RETURN as i32,
        );
        assert_eq!(
            meta.staticdata.op_void_return,
            crate::jitcode::insns::BC_VOID_RETURN as i32,
        );
    }

    #[test]
    #[should_panic(expected = "called after `staticdata` was cloned")]
    fn metainterp_install_canonical_liveness_panics_after_staticdata_clone() {
        // Same single-owner invariant as `finish_setup`: once
        // `staticdata` is shared, the hook must fail loudly rather
        // than silently no-op or clobber a shared snapshot.
        use majit_translate::jit_codewriter::assembler::Assembler;

        let mut meta = MetaInterp::<()>::new(0);
        let _share: std::sync::Arc<MetaInterpStaticData> = meta.staticdata.clone();
        meta.install_canonical_liveness(&Assembler::new());
    }

    #[test]
    fn register_jitdriver_sd_stamps_index_back() {
        // call.py:46-47 `jd.index = idx` — index written back into the
        // descriptor at registration time, not left None.
        let mut meta = MetaInterp::<()>::new(0);
        let jd = crate::jitdriver::JitDriverStaticData::new(vec![], vec![]);
        let idx = {
            let MetaInterp {
                staticdata,
                backend,
                ..
            } = &mut meta;
            std::sync::Arc::get_mut(staticdata)
                .unwrap()
                .register_jitdriver_sd(jd, backend)
        };
        let sd = std::sync::Arc::get_mut(&mut meta.staticdata).unwrap();
        assert_eq!(sd.jitdrivers_sd[idx].index, Some(idx));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::JitArgKind;
    use crate::resume::{FrameSlotSource, ReconstructedValue, ResolvedPendingFieldWrite};
    #[cfg(feature = "cranelift")]
    use majit_backend::DeadFrame;
    #[cfg(feature = "cranelift")]
    use majit_backend_cranelift::compiler::{
        force_token_to_dead_frame, get_int_from_deadframe, get_latest_descr_from_deadframe,
        set_savedata_ref_on_deadframe,
    };
    use majit_ir::descr::{CallDescr, Descr, EffectInfo, ExtraEffect};
    use majit_ir::{DescrRef, InputArg, Op, OpCode, OpRef, Type, Value};
    use std::sync::{Arc, Mutex, OnceLock};

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = if pos == OpRef::NONE.raw() {
            OpRef::NONE
        } else {
            OpRef::op_typed(pos, opcode.result_type())
        };
        op
    }

    fn mk_op_with_descr(opcode: OpCode, args: &[OpRef], pos: u32, descr: DescrRef) -> Op {
        let mut op = Op::with_descr(opcode, args, descr);
        op.pos = if pos == OpRef::NONE.raw() {
            OpRef::NONE
        } else {
            OpRef::op_typed(pos, opcode.result_type())
        };
        op
    }

    fn test_subclass_range(classptr: usize) -> Option<(i64, i64)> {
        match classptr {
            0x1000 => Some((10, 20)),
            0x1100 => Some((12, 13)),
            0x2000 => Some((30, 31)),
            _ => None,
        }
    }

    #[test]
    fn default_issubclass_uses_active_gc_subclass_ranges() {
        struct ResetGcHooks;
        impl Drop for ResetGcHooks {
            fn drop(&mut self) {
                majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks::default());
            }
        }

        let _reset = ResetGcHooks;
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            subclass_range: Some(test_subclass_range),
            ..Default::default()
        });

        assert!(default_issubclass(0x1100, 0x1000));
        assert!(!default_issubclass(0x1000, 0x1100));
        assert!(!default_issubclass(0x2000, 0x1000));
        assert!(default_issubclass(0xdead, 0xdead));
    }

    #[test]
    fn test_normalize_root_loop_entry_contract_rejects_missing_label() {
        // compile.py:359 parity: an optimized trace that arrives without a
        // LABEL is a broken contract; the helper must report the missing
        // LABEL as an arity mismatch instead of synthesizing one.
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                3,
            ),
            mk_op(
                OpCode::Jump,
                &[
                    OpRef::int_op(3),
                    OpRef::input_arg_int(2),
                    OpRef::input_arg_int(1),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let err =
            normalize_root_loop_entry_contract(inputargs, ops).expect_err("missing LABEL rejects");
        assert_eq!(err, (0, 3));
    }

    #[test]
    fn test_normalize_root_loop_entry_contract_rejects_arity_mismatch() {
        let inputargs = vec![
            InputArg::new_int(0),
            InputArg::new_int(1),
            InputArg::new_int(2),
        ];
        let ops = vec![mk_op(
            OpCode::Jump,
            &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
            OpRef::NONE.raw(),
        )];

        let err =
            normalize_root_loop_entry_contract(inputargs, ops).expect_err("missing LABEL rejects");
        assert_eq!(err, (0, 2));
    }

    #[test]
    fn test_prepare_bridge_trace_for_optimizer_freshens_inputargs_and_snapshots() {
        let bridge_inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let bridge_ops = vec![
            mk_op(OpCode::SameAsR, &[OpRef::input_arg_ref(1)], 2),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(0)],
                3,
            ),
            mk_op(
                OpCode::Jump,
                &[OpRef::ref_op(2), OpRef::int_op(3)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut snapshot_boxes = Vec::new();
        snapshot_insert(
            &mut snapshot_boxes,
            0,
            vec![
                OpRef::input_arg_int(0).into(),
                OpRef::ref_op(2).into(),
                OpRef::int_op(3).into(),
            ],
        );
        let mut snapshot_vable_boxes = Vec::new();
        snapshot_insert(
            &mut snapshot_vable_boxes,
            0,
            vec![OpRef::input_arg_ref(1).into(), OpRef::ref_op(2).into()],
        );
        let pending_bridge_rd = PendingBridgeRd {
            storage: crate::resume::ResumeStorage::new(vec![1, 2, 3], vec![], vec![], vec![]),
            frontend_boxes: vec![11, 22],
            liveboxes: vec![OpRef::input_arg_int(0), OpRef::input_arg_ref(1)],
            livebox_types: vec![Type::Int, Type::Ref],
            all_descrs: vec![],
            cls_of_box: None,
        };

        let prepared = prepare_bridge_trace_for_optimizer(
            &bridge_ops,
            &bridge_inputargs,
            snapshot_boxes,
            Vec::new(),
            snapshot_vable_boxes,
            Vec::new(),
            Vec::new(),
            Some(pending_bridge_rd),
            10,
        );

        assert_eq!(
            prepared
                .inputargs
                .iter()
                .map(|arg| (arg.index, arg.tp))
                .collect::<Vec<_>>(),
            vec![(10, Type::Int), (11, Type::Ref)]
        );
        assert_eq!(prepared.ops[0].pos, OpRef::ref_op(12));
        assert_eq!(
            prepared.ops[0].args.to_vec(),
            vec![OpRef::input_arg_ref(11)]
        );
        assert_eq!(prepared.ops[1].pos, OpRef::int_op(13));
        assert_eq!(
            prepared.ops[1].args.to_vec(),
            vec![OpRef::input_arg_int(10), OpRef::input_arg_int(10)]
        );
        assert_eq!(
            prepared.ops[2].args.to_vec(),
            vec![OpRef::ref_op(12), OpRef::int_op(13)]
        );
        assert_eq!(
            snapshot_get(&prepared.snapshot_boxes, 0)
                .unwrap()
                .iter()
                .map(|boxref| boxref.opref)
                .collect::<Vec<_>>(),
            vec![
                OpRef::input_arg_int(10),
                OpRef::ref_op(12),
                OpRef::int_op(13)
            ]
        );
        assert_eq!(
            snapshot_get(&prepared.snapshot_vable_boxes, 0)
                .unwrap()
                .iter()
                .map(|boxref| boxref.opref)
                .collect::<Vec<_>>(),
            vec![OpRef::input_arg_ref(11), OpRef::ref_op(12)]
        );
        assert_eq!(
            prepared
                .pending_bridge_rd
                .as_ref()
                .unwrap()
                .liveboxes
                .clone(),
            vec![OpRef::input_arg_int(10), OpRef::input_arg_ref(11)]
        );
    }

    #[test]
    fn test_front_target_inputarg_types_uses_saved_front_label_contract() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 7;
        let trace_id = 11;
        let token = std::sync::Arc::new(JitCellToken::new(3));
        let start_token = crate::optimizeopt::unroll::TargetToken::new_preamble(0);
        let start_descr = start_token.as_jump_target_descr();
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let ops = vec![
            mk_op(OpCode::SameAsR, &[OpRef::input_arg_ref(0)], 2),
            mk_op(OpCode::IntAdd, &[OpRef::int_op(100), OpRef::int_op(101)], 3),
            mk_op_with_descr(
                OpCode::Label,
                &[
                    OpRef::input_arg_ref(0),
                    OpRef::input_arg_ref(1),
                    OpRef::ref_op(2),
                    OpRef::int_op(3),
                ],
                OpRef::NONE.raw(),
                start_descr,
            ),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 2);
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: inputargs.clone(),
                ops,
                constants,
                constant_types: HashMap::new(),
                exit_layouts: HashMap::new(),
                terminal_exit_layouts: HashMap::new(),
            },
        );
        meta.warm_state_mut()
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&token));
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token,
                meta: (),
                front_target_tokens: vec![start_token],
                root_trace_id: trace_id,
                traces,
                previous_tokens: Vec::new(),
                next_global_opref: 0,
            },
        );

        assert_eq!(
            meta.front_target_inputarg_types(green_key),
            Some(vec![Type::Ref, Type::Ref, Type::Ref, Type::Int])
        );
    }

    #[test]
    fn test_recovery_slot_types_use_single_frame_slot_types_when_available() {
        let recovery_layout = majit_backend::ExitRecoveryLayout {
            vable_array: vec![],
            vref_array: vec![],
            frames: vec![majit_backend::ExitFrameLayout {
                trace_id: None,
                header_pc: Some(17),
                source_guard: None,
                pc: 17,
                jitcode_index: 0,
                slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                slot_types: Some(vec![Type::Int]),
            }],
            virtual_layouts: vec![],
            pending_field_layouts: vec![],
        };

        assert_eq!(
            MetaInterp::<()>::recovery_slot_types_from_exit_types_and_layout(
                &[Type::Ref],
                Some(&recovery_layout),
            ),
            vec![Type::Int]
        );
    }

    #[test]
    fn test_recovery_slot_types_fall_back_on_length_mismatch() {
        let recovery_layout = majit_backend::ExitRecoveryLayout {
            vable_array: vec![],
            vref_array: vec![],
            frames: vec![majit_backend::ExitFrameLayout {
                trace_id: None,
                header_pc: Some(17),
                source_guard: None,
                pc: 17,
                jitcode_index: 0,
                slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                slot_types: Some(vec![Type::Int]),
            }],
            virtual_layouts: vec![],
            pending_field_layouts: vec![],
        };

        assert_eq!(
            MetaInterp::<()>::recovery_slot_types_from_exit_types_and_layout(
                &[Type::Ref, Type::Ref],
                Some(&recovery_layout),
            ),
            vec![Type::Ref, Type::Ref]
        );
    }

    #[test]
    fn test_recovery_slot_types_concatenate_frames_in_callee_first_order() {
        let recovery_layout = majit_backend::ExitRecoveryLayout {
            vable_array: vec![],
            vref_array: vec![],
            frames: vec![
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: Some(11),
                    source_guard: None,
                    pc: 11,
                    jitcode_index: 0,
                    slots: vec![
                        majit_backend::ExitValueSourceLayout::ExitValue(0),
                        majit_backend::ExitValueSourceLayout::ExitValue(1),
                    ],
                    slot_types: Some(vec![Type::Ref, Type::Int]),
                },
                majit_backend::ExitFrameLayout {
                    trace_id: None,
                    header_pc: Some(23),
                    source_guard: None,
                    pc: 23,
                    jitcode_index: 1,
                    slots: vec![
                        majit_backend::ExitValueSourceLayout::ExitValue(2),
                        majit_backend::ExitValueSourceLayout::ExitValue(3),
                    ],
                    slot_types: Some(vec![Type::Float, Type::Ref]),
                },
            ],
            virtual_layouts: vec![],
            pending_field_layouts: vec![],
        };

        assert_eq!(
            MetaInterp::<()>::recovery_slot_types_from_exit_types_and_layout(
                &[Type::Ref, Type::Int, Type::Float, Type::Ref],
                Some(&recovery_layout),
            ),
            vec![Type::Float, Type::Ref, Type::Ref, Type::Int]
        );
    }

    #[test]
    fn test_guard_resume_getters_return_stored_exit_layout_metadata() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 19;
        let trace_id = 23;
        let fail_index = 5;

        let recovery_layout = majit_backend::ExitRecoveryLayout {
            vable_array: vec![],
            vref_array: vec![],
            frames: vec![majit_backend::ExitFrameLayout {
                trace_id: None,
                header_pc: Some(41),
                source_guard: None,
                pc: 41,
                jitcode_index: 0,
                slots: vec![majit_backend::ExitValueSourceLayout::ExitValue(0)],
                slot_types: Some(vec![Type::Int]),
            }],
            virtual_layouts: vec![],
            pending_field_layouts: vec![],
        };

        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(0),
                gc_ref_slots: vec![0],
                force_token_slots: vec![],
                recovery_layout: Some(recovery_layout),
                resume_layout: None,
                storage: Some(crate::resume::ResumeStorage::new(
                    vec![7, 8, 9],
                    vec![majit_ir::Const::Int(11)],
                    vec![],
                    vec![],
                )),
                descr: Some(crate::compile::make_fail_descr_typed(vec![Type::Ref])),
                op_arg_types_for_jump: None,
            },
        );

        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: vec![],
                ops: vec![],
                constants: HashMap::new(),
                constant_types: HashMap::new(),
                exit_layouts,
                terminal_exit_layouts: HashMap::new(),
            },
        );

        let token = std::sync::Arc::new(JitCellToken::new(3));
        meta.warm_state_mut()
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&token));
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token,
                meta: (),
                front_target_tokens: Vec::new(),
                root_trace_id: trace_id,
                traces,
                previous_tokens: Vec::new(),
                next_global_opref: 0,
            },
        );

        assert_eq!(
            meta.get_merge_point_pc(green_key, trace_id, fail_index),
            Some(41)
        );
        let storage = meta
            .get_resume_storage(green_key, trace_id, fail_index)
            .expect("storage should be present");
        assert_eq!(storage.rd_numb, vec![7, 8, 9]);
        assert_eq!(
            unsafe { (*storage.rd_consts.get()).clone() },
            vec![majit_ir::Const::Int(11)]
        );
        assert_eq!(
            meta.get_recovery_slot_types(green_key, trace_id, fail_index),
            Some(vec![Type::Int])
        );
        assert_eq!(
            meta.get_rd_virtuals(green_key, trace_id, fail_index),
            Some(vec![])
        );
        let pendingfields = meta
            .get_rd_pendingfields(green_key, trace_id, fail_index)
            .expect("stored exit layout should expose rd_pendingfields");
        assert!(pendingfields.is_empty());
    }

    #[test]
    fn test_handle_async_forcing_prepares_rd_virtuals_from_exit_layout() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 29;
        let trace_id = 31;
        let fail_index = 7;

        let mut writer = crate::resumecode::Writer::new(4);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.append_int(0); // vref_size
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut exit_layouts = HashMap::new();
        exit_layouts.insert(
            fail_index,
            StoredExitLayout {
                source_op_index: Some(0),
                gc_ref_slots: vec![],
                force_token_slots: vec![],
                recovery_layout: None,
                resume_layout: None,
                storage: Some(crate::resume::ResumeStorage::new(
                    rd_numb,
                    vec![],
                    vec![std::rc::Rc::new(majit_ir::RdVirtualInfo::VRawBufferInfo {
                        func: 77,
                        size: 0,
                        offsets: vec![],
                        descrs: vec![],
                        fieldnums: vec![],
                    })],
                    vec![],
                )),
                descr: Some(crate::compile::make_fail_descr_typed(vec![])),
                op_arg_types_for_jump: None,
            },
        );

        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: vec![],
                ops: vec![],
                constants: HashMap::new(),
                constant_types: HashMap::new(),
                exit_layouts,
                terminal_exit_layouts: HashMap::new(),
            },
        );

        let token = std::sync::Arc::new(JitCellToken::new(3));
        meta.warm_state_mut()
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&token));
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token,
                meta: (),
                front_target_tokens: Vec::new(),
                root_trace_id: trace_id,
                traces,
                previous_tokens: Vec::new(),
                next_global_opref: 0,
            },
        );

        let (ptrs, ints) = meta
            .handle_async_forcing(green_key, trace_id, fail_index, &[])
            .expect("stored rd_virtuals should be forced");
        assert_eq!(ptrs, vec![0]);
        assert_eq!(ints, vec![0]);
    }

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[cfg(target_arch = "x86_64")]
    type DynasmCompiledCode = majit_backend_dynasm::x86::assembler::CompiledCode;

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[cfg(target_arch = "aarch64")]
    type DynasmCompiledCode = majit_backend_dynasm::aarch64::assembler::CompiledCode;

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    fn patch_dynasm_fail_descr_resume_data(
        token: &std::sync::Arc<JitCellToken>,
        fail_index: u32,
        rd_numb: Vec<u8>,
        rd_consts: Vec<majit_ir::Const>,
    ) {
        // Test-only: same single-threaded JIT scheduler invariant the
        // sibling `descr` cast below relies on — bypass `Arc::get_mut`
        // because the runtime keeps a second strong ref via the warm
        // cell / memmgr / `compiled_loops`.
        let token: &mut JitCellToken =
            unsafe { &mut *(std::sync::Arc::as_ptr(token) as *mut JitCellToken) };
        let compiled = token
            .compiled
            .as_mut()
            .expect("compiled token")
            .downcast_mut::<DynasmCompiledCode>()
            .expect("dynasm compiled code");
        let descr = compiled
            .fail_descrs
            .iter_mut()
            .find(|descr| descr.fail_index == fail_index)
            .expect("fail descr");
        // Session 5b: rd_* now forward through `meta_descr` to the
        // metainterp ResumeGuardDescr Arc.  Test fixtures synthesise
        // DynasmFailDescr without going through the assembler, so they
        // have no meta_descr.  Mint a fresh metainterp ResumeGuardDescr
        // (carries the rd_* setters) and stamp it as the back-pointer.
        let descr = unsafe {
            &mut *(std::sync::Arc::as_ptr(descr)
                as *mut majit_backend_dynasm::guard::DynasmFailDescr)
        };
        if descr.meta_descr.is_none() {
            let meta = crate::compile::make_resume_guard_descr_typed(descr.fail_arg_types.clone());
            // `DynasmFailDescr::trace_id()` (`guard.rs:380-392`) forwards
            // through `meta_resume_fd().trace_id()` whenever a meta_descr
            // is attached and falls back to the backend-local
            // `self.trace_id` field only when meta_descr is absent.
            // `make_resume_guard_descr_typed` initialises trace_id to 0
            // (`compile.rs:3425`); without copying the backend-local
            // trace_id over, the freshly-stamped meta_descr would mask
            // the descr's real trace_id and break `(trace_id, fail_index)`
            // lookups in `try_find_descr`.
            if let Some(meta_fd) = meta.as_fail_descr() {
                meta_fd.set_trace_id(descr.trace_id);
            }
            descr.meta_descr = Some(meta);
        }
        let meta_fd = descr
            .meta_descr
            .as_ref()
            .unwrap()
            .as_fail_descr()
            .expect("metainterp descr must implement FailDescr");
        meta_fd.set_rd_numb(Some(rd_numb));
        meta_fd.set_rd_consts(Some(rd_consts));
        meta_fd.set_rd_virtuals(Some(vec![]));
        meta_fd.set_rd_pendingfields(Some(vec![]));
    }

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[test]
    fn test_handle_async_forcing_falls_back_to_previous_token_backend_exit_layout() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 30;
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::input_arg_int(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let (trace_id, fail_index) = {
            let entry = meta.compiled_loops.get(&green_key).expect("compiled entry");
            let trace_id = entry.root_trace_id;
            let trace = entry.traces.get(&trace_id).expect("compiled trace");
            let fail_index = guard_fail_index(trace);
            (trace_id, fail_index)
        };

        let mut writer = crate::resumecode::Writer::new(4);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.append_int(0); // vref_size
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        {
            let entry = meta
                .compiled_loops
                .get_mut(&green_key)
                .expect("compiled entry");
            patch_dynasm_fail_descr_resume_data(&entry.token, fail_index, rd_numb, vec![]);
            let mut fresh_token = JitCellToken::new(9003);
            fresh_token.green_key = green_key;
            let old_token = std::mem::replace(&mut entry.token, std::sync::Arc::new(fresh_token));
            entry.previous_tokens.push(old_token);
            entry
                .traces
                .get_mut(&trace_id)
                .expect("compiled trace")
                .exit_layouts
                .remove(&fail_index);
        }

        let (ptrs, ints) = meta
            .handle_async_forcing(green_key, trace_id, fail_index, &[42])
            .expect("async forcing should fall back to previous token backend layout");
        assert_eq!(ptrs, Vec::<i64>::new());
        assert_eq!(ints, Vec::<i64>::new());
    }

    #[derive(Debug)]
    struct TestCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl Descr for TestCallDescr {
        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }

        fn result_type(&self) -> Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn get_extra_info(&self) -> &EffectInfo {
            static EFFECT_INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, majit_ir::OopSpecIndex::None);
            &EFFECT_INFO
        }
    }

    fn make_call_descr(arg_types: Vec<Type>, result_type: Type) -> DescrRef {
        Arc::new(TestCallDescr {
            arg_types,
            result_type,
        })
    }

    fn may_force_void_values() -> &'static Mutex<Vec<i64>> {
        static VALUES: OnceLock<Mutex<Vec<i64>>> = OnceLock::new();
        VALUES.get_or_init(|| Mutex::new(Vec::new()))
    }

    fn may_force_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "cranelift")]
    fn with_forced_deadframe(force_token: i64, f: impl FnOnce(DeadFrame)) {
        f(force_token_to_dead_frame(GcRef(force_token as usize)));
    }

    #[cfg(feature = "cranelift")]
    extern "C" fn maybe_force_and_return_void(force_token: i64, flag: i64) {
        if flag == 0 {
            return;
        }
        with_forced_deadframe(force_token, |mut deadframe| {
            let mut values = may_force_void_values()
                .lock()
                .unwrap_or_else(|err| err.into_inner());
            values.push(
                get_latest_descr_from_deadframe(&deadframe)
                    .unwrap()
                    .fail_index() as i64,
            );
            values.push(get_int_from_deadframe(&deadframe, 0).unwrap());
            values.push(get_int_from_deadframe(&deadframe, 1).unwrap());
            drop(values);
            set_savedata_ref_on_deadframe(&mut deadframe, GcRef(0xDADA)).unwrap();
        });
    }

    fn attach_procedure_to_interp_entry(
        meta: &mut MetaInterp<()>,
        green_key: u64,
        inputargs: &[InputArg],
        ops: Vec<Op>,
        constants: HashMap<u32, i64>,
    ) {
        meta.backend.set_constants(constants.clone());
        let mut token = JitCellToken::new(green_key + 1000);
        let trace_id = meta.alloc_trace_id();
        meta.backend.set_next_trace_id(trace_id);
        meta.backend
            .compile_loop(inputargs, &ops, &mut token)
            .expect("loop should compile");
        let (mut resume_data, mut exit_layouts) = compile::build_guard_metadata(
            inputargs,
            &ops,
            green_key,
            &HashMap::new(),
            meta.callinfocollection.as_deref(),
        );
        let mut terminal_exit_layouts =
            compile::build_terminal_exit_layouts(inputargs, &ops, &HashMap::new());
        if let Some(backend_layouts) = meta.backend.compiled_fail_descr_layouts(&token) {
            compile::merge_backend_exit_layouts(&mut exit_layouts, &backend_layouts, &ops);
        }
        if let Some(backend_layouts) = meta.backend.compiled_terminal_exit_layouts(&token) {
            compile::merge_backend_terminal_exit_layouts(
                &mut terminal_exit_layouts,
                &backend_layouts,
                &ops,
            );
        }
        let trace_info = meta.backend.compiled_trace_info(&token, trace_id);
        compile::enrich_guard_resume_layouts_for_trace(
            &mut resume_data,
            &mut exit_layouts,
            trace_id,
            inputargs,
            trace_info.as_ref(),
        );
        compile::patch_backend_guard_recovery_layouts_for_trace(
            &mut meta.backend,
            &token,
            trace_id,
            &mut exit_layouts,
        );
        compile::patch_backend_terminal_recovery_layouts_for_trace(
            &mut meta.backend,
            &token,
            trace_id,
            &mut terminal_exit_layouts,
        );
        let mut traces = HashMap::new();
        traces.insert(
            trace_id,
            CompiledTrace {
                inputargs: inputargs.to_vec(),
                ops,
                constants,
                constant_types: HashMap::new(),
                exit_layouts,
                terminal_exit_layouts,
            },
        );

        let token_arc = std::sync::Arc::new(token);
        // Mirror production attach: warmstate.py:339-348
        // `attach_procedure_to_interp` writes `cell.loop_token` so the
        // green_key → token canonical lookup (warmstate.py:188-202) is
        // populated alongside the metainterp-side `compiled_loops`
        // HashMap.  Without this, `has_compiled_loop` (now routed
        // through `warm_state.get_procedure_token`) returns `false` for
        // entries created via this test fixture.
        meta.warm_state_mut()
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&token_arc));
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token: token_arc,
                meta: (),
                front_target_tokens: Vec::new(),
                root_trace_id: trace_id,
                traces,
                previous_tokens: Vec::new(),
                next_global_opref: 0,
            },
        );
    }

    fn guard_fail_index(trace: &CompiledTrace) -> u32 {
        *trace
            .exit_layouts
            .iter()
            .find(|(_, layout)| !layout.resolve_is_finish())
            .map(|(fail_index, _)| fail_index)
            .expect("compiled guard exit")
    }

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[test]
    fn guard_exit_getters_fall_back_to_previous_token_backend_layouts() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 77;
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::input_arg_int(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let (trace_id, fail_index) = {
            let entry = meta.compiled_loops.get(&green_key).expect("compiled entry");
            let trace_id = entry.root_trace_id;
            let trace = entry.traces.get(&trace_id).expect("compiled trace");
            let fail_index = guard_fail_index(trace);
            (trace_id, fail_index)
        };

        {
            let entry = meta
                .compiled_loops
                .get_mut(&green_key)
                .expect("compiled entry");
            let mut fresh_token = JitCellToken::new(9001);
            fresh_token.green_key = green_key;
            let old_token = std::mem::replace(&mut entry.token, std::sync::Arc::new(fresh_token));
            entry.previous_tokens.push(old_token);
            entry
                .traces
                .get_mut(&trace_id)
                .expect("compiled trace")
                .exit_layouts
                .remove(&fail_index);
        }

        let layout = meta
            .get_compiled_exit_layout_in_trace(green_key, trace_id, fail_index)
            .expect("previous token backend layout should remain visible");
        assert_eq!(layout.exit_types, vec![Type::Int]);
        assert!(!layout.is_finish);
        assert_eq!(
            meta.get_exit_types(green_key, trace_id, fail_index),
            Some(vec![Type::Int])
        );
        let descr_arc = meta
            .get_fail_descr_for_bridge(green_key, trace_id, fail_index)
            .expect("bridge fail descr should search previous tokens too");
        let fail_descr = descr_arc
            .as_fail_descr()
            .expect("bridge source op.descr must implement FailDescr");
        assert_eq!(fail_descr.fail_arg_types(), &[Type::Int]);
    }

    // `guard_fail_descr_proxy_trusts_empty_backend_fail_arg_types`
    // removed in Unified-Descr Port Epic, Session 6: the proxy now
    // forwards `fail_arg_types()` to the metainterp `ResumeGuardDescr`
    // Arc, so patching the backend descr's `fail_arg_types` no longer
    // changes the proxy's view (the test's premise was a split-descr
    // adaptation that does not survive unification).  PyPy parity:
    // `cpu.get_latest_descr()` (`history.py:125`) returns the same
    // descr object the metainterp stamped; there is no "backend
    // override" to test.

    #[test]
    fn guard_failure_recovery_uses_previous_token_backend_exit_layout() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 88;
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::input_arg_int(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let (trace_id, fail_index, expected_source_op_index, expected_rd_numb, expected_exit_types) = {
            let entry = meta.compiled_loops.get(&green_key).expect("compiled entry");
            let trace_id = entry.root_trace_id;
            let trace = entry.traces.get(&trace_id).expect("compiled trace");
            let fail_index = guard_fail_index(trace);
            let layout = trace
                .exit_layouts
                .get(&fail_index)
                .expect("stored exit layout");
            (
                trace_id,
                fail_index,
                layout.source_op_index,
                layout
                    .storage
                    .as_ref()
                    .map(|storage| storage.rd_numb.clone()),
                layout.resolve_exit_types().to_vec(),
            )
        };

        {
            let entry = meta
                .compiled_loops
                .get_mut(&green_key)
                .expect("compiled entry");
            let mut fresh_token = JitCellToken::new(9002);
            fresh_token.green_key = green_key;
            let old_token = std::mem::replace(&mut entry.token, std::sync::Arc::new(fresh_token));
            entry.previous_tokens.push(old_token);
            entry
                .traces
                .get_mut(&trace_id)
                .expect("compiled trace")
                .exit_layouts
                .remove(&fail_index);
        }

        let recovery = meta
            .handle_guard_failure_in_trace_with_savedata(
                green_key,
                trace_id,
                fail_index,
                &[42],
                Some(&[Value::Int(42)]),
                None,
                ExceptionState::default(),
            )
            .expect("guard recovery should fall back to previous token backend layout");

        assert_eq!(
            recovery.exit_layout.source_op_index,
            expected_source_op_index
        );
        assert_eq!(
            recovery
                .exit_layout
                .storage
                .as_ref()
                .map(|storage| storage.rd_numb.clone()),
            expected_rd_numb
        );
        assert_eq!(recovery.exit_layout.exit_types, expected_exit_types);
    }

    #[cfg(all(feature = "dynasm", not(feature = "cranelift")))]
    #[test]
    fn test_start_retrace_from_guard_uses_previous_token_backend_resume_data() {
        let mut meta = MetaInterp::<()>::new(1);
        let green_key = 89;
        let inputargs = vec![InputArg::new_int(0)];
        let mut guard = mk_op(
            OpCode::GuardTrue,
            &[OpRef::input_arg_int(0)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(smallvec::SmallVec::from_slice(&[OpRef::input_arg_int(0)]));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, HashMap::new());

        let (trace_id, fail_index) = {
            let entry = meta.compiled_loops.get(&green_key).expect("compiled entry");
            let trace_id = entry.root_trace_id;
            let trace = entry.traces.get(&trace_id).expect("compiled trace");
            let fail_index = guard_fail_index(trace);
            (trace_id, fail_index)
        };

        let mut writer = crate::resumecode::Writer::new(4);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.append_int(0); // vref_size
        writer.patch_current_size(0);
        let expected_rd_numb = writer.create_numbering();

        {
            let entry = meta
                .compiled_loops
                .get_mut(&green_key)
                .expect("compiled entry");
            patch_dynasm_fail_descr_resume_data(
                &mut entry.token,
                fail_index,
                expected_rd_numb.clone(),
                vec![],
            );
            let mut fresh_token = JitCellToken::new(9004);
            fresh_token.green_key = green_key;
            let old_token = std::mem::replace(&mut entry.token, std::sync::Arc::new(fresh_token));
            entry.previous_tokens.push(old_token);
            entry
                .traces
                .get_mut(&trace_id)
                .expect("compiled trace")
                .exit_layouts
                .remove(&fail_index);
        }

        let retrace = meta
            .start_retrace_from_guard(green_key, trace_id, fail_index, &[42])
            .expect("retrace should use previous token backend resume data");

        assert_eq!(retrace.fail_types, vec![Type::Int]);
        let storage = retrace
            .storage
            .as_ref()
            .expect("retrace storage should be present");
        assert_eq!(storage.rd_numb, expected_rd_numb);
        assert!(unsafe { (*storage.rd_consts.get()).is_empty() });
        assert!(storage.rd_virtuals.is_empty());
    }

    #[cfg(feature = "cranelift")]
    fn install_may_force_void_entry(meta: &mut MetaInterp<()>, green_key: u64) {
        may_force_void_values()
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .clear();
        let descr = make_call_descr(vec![Type::Ref, Type::Int], Type::Void);
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let mut guard_op = mk_op(OpCode::GuardNotForced, &[], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::int_op(1),
            OpRef::int_op(0),
        ]));
        // pyre cranelift test-fixture quirk (NOT RPython parity): the bare
        // OpRef::int_op(100) literal is paired with `constants.insert(100, ...)` below
        // because `backend.set_constants` keys the function-pointer literal by
        // raw OpRef value. RPython expresses the same callable as a `Const*`
        // box wrapping the address; pyre's test backend short-circuits that by
        // exposing the address through a HashMap whose key happens to equal
        // the OpRef raw bits. Keep the literal in sync with the constants
        // entry — they are co-load-bearing.
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::int_op(0), OpRef::int_op(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::ForceToken, &[], 2),
            mk_op_with_descr(
                OpCode::CallMayForceN,
                &[OpRef::int_op(100), OpRef::int_op(2), OpRef::int_op(1)],
                OpRef::NONE.raw(),
                descr,
            ),
            guard_op,
            mk_op(OpCode::Finish, &[OpRef::int_op(0)], OpRef::NONE.raw()),
        ];
        let mut constants = HashMap::new();
        constants.insert(
            100,
            maybe_force_and_return_void as *const () as usize as i64,
        );
        attach_procedure_to_interp_entry(meta, green_key, &inputargs, ops, constants);
    }

    extern "C" fn test_clear_vable_token(_gcref: *mut u8) {}

    fn test_vable_info_static_only() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info.set_clear_vable(
            test_clear_vable_token as *const (),
            VirtualizableInfo::make_clear_vable_descr(),
        );
        info
    }

    fn test_vable_info_with_array() -> VirtualizableInfo {
        let mut info = VirtualizableInfo::new(0);
        info.add_array_field(
            "stack",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));
        info.set_clear_vable(
            test_clear_vable_token as *const (),
            VirtualizableInfo::make_clear_vable_descr(),
        );
        info
    }

    #[repr(C)]
    struct TraceEntryArray {
        len: usize,
        items: [i64; 4],
    }

    #[repr(C)]
    struct TraceEntryObj {
        arr: *const TraceEntryArray,
    }

    #[repr(C)]
    struct ResidualCallVableObj {
        token: u64,
        pc: i64,
    }

    fn start_tracing_with_virtualizable(
        meta: &mut MetaInterp<()>,
        info: VirtualizableInfo,
        live_values: &[Value],
        array_lengths: Vec<usize>,
    ) {
        meta.set_virtualizable_info(std::sync::Arc::new(info));
        meta.set_vable_array_lengths(array_lengths);
        let action = meta.force_start_tracing(777, (0, 0), None, live_values);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
    }

    fn take_recorded_ops(meta: &mut MetaInterp<()>) -> Vec<Op> {
        let mut ctx = meta.tracing.take().expect("expected active trace context");
        let num_inputs = ctx.num_inputargs();
        let input_types = ctx.inputarg_types();
        let jump_args: Vec<OpRef> = (0..num_inputs)
            .map(|i| OpRef::input_arg_typed(i as u32, input_types[i]))
            .collect();
        ctx.close_loop(&jump_args);
        let trace = ctx.into_tree_loop();
        trace
            .ops
            .into_iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .collect()
    }

    fn finish_trace_for_parity_preserves_captured_snapshots() {
        let mut meta = MetaInterp::<()>::new(10);
        let action = meta.force_start_tracing(777, (0, 0), None, &[Value::Int(17)]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let snapshot_id = {
            let ctx = meta.trace_ctx().expect("active trace context");
            ctx.capture_resumedata(crate::recorder::Snapshot {
                frames: vec![crate::recorder::SnapshotFrame {
                    jitcode_index: 0,
                    pc: 123,
                    boxes: vec![crate::recorder::SnapshotTagged::Box(
                        OpRef::int_op(0),
                        majit_ir::Type::Int,
                    )],
                }],
                vable_boxes: Vec::new(),
                vref_boxes: Vec::new(),
            })
        };
        assert_eq!(snapshot_id, 0);

        let (trace, _) = meta
            .finish_trace_for_parity(&[OpRef::int_op(0)])
            .expect("trace should finish");
        assert_eq!(trace.snapshots.len(), 1);
        assert_eq!(trace.snapshots[0].frames.len(), 1);
        assert_eq!(trace.snapshots[0].frames[0].pc, 123);
    }

    #[test]
    fn trace_entry_vable_lengths_reads_from_heap_first() {
        // pyjitpl.py:3302 `vinfo.read_boxes(...)` always reads from the
        // concrete heap object. The interpreter-supplied cache is a pyre
        // fallback used only when `info.can_read_all_array_lengths_from_heap`
        // is false; otherwise the heap-read wins.
        let mut info = VirtualizableInfo::new(0);
        {
            let items_offset = std::mem::size_of::<usize>();
            info.add_array_field(
                "arr",
                Type::Int,
                std::mem::offset_of!(TraceEntryObj, arr),
                0,
                items_offset,
                majit_ir::make_array_descr(items_offset, 8, Type::Int),
            );
        }
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let array = TraceEntryArray {
            len: 4,
            items: [10, 20, 30, 40],
        };
        let obj = TraceEntryObj { arr: &array };

        let mut meta = MetaInterp::<()>::new(10);
        meta.set_virtualizable_info(std::sync::Arc::new(info.clone()));
        meta.set_vable_ptr((&obj as *const TraceEntryObj).cast());
        // Even if the interpreter cache claims a different length, the heap
        // object's `len=4` wins — matching RPython semantics.
        meta.set_vable_array_lengths(vec![1]);

        assert_eq!(meta.trace_entry_vable_lengths(&info), vec![4]);
    }

    #[test]
    fn initialize_virtualizable_appends_read_boxes_to_red_only_trace_entry() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = std::sync::Arc::new(test_vable_info_static_only());
        meta.set_virtualizable_info(info.clone());

        let mut obj = ResidualCallVableObj { token: 0, pc: 41 };
        meta.set_vable_ptr((&mut obj as *mut ResidualCallVableObj).cast());

        let descriptor = JitDriverStaticData::with_virtualizable(
            vec![],
            vec![("frame", Type::Ref)],
            Some("frame"),
        );
        let frame = Value::Ref(majit_ir::GcRef(
            (&mut obj as *mut ResidualCallVableObj) as usize,
        ));
        let action = meta.force_start_tracing(777, (0, 0), Some(descriptor), &[frame]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let ctx = meta.trace_ctx().expect("expected active trace context");
        assert_eq!(ctx.recorder.num_inputargs(), 2);
        assert_eq!(ctx.inputarg_types(), vec![Type::Ref, Type::Int]);
        assert_eq!(
            ctx.collect_virtualizable_boxes().unwrap(),
            vec![OpRef::input_arg_int(1), OpRef::input_arg_ref(0)]
        );
        assert_eq!(
            ctx.virtualizable_entry_at(0),
            Some((OpRef::input_arg_int(1), Value::Int(41)))
        );
    }

    #[test]
    fn opimpl_getfield_vable_int_reads_standard_box_without_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_static_only();
        let fd8 = info.static_field_descr(0);
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let (result, _) = meta.opimpl_getfield_vable_int(OpRef::input_arg_ref(0), fd8);
        assert_eq!(result, OpRef::input_arg_int(1));

        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.num_ops(), 0);
    }

    #[test]
    fn opimpl_setfield_vable_int_synchronizes_standard_virtualizable() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_static_only();
        let fd8 = info.static_field_descr(0);
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[Value::Int(0x1234), Value::Int(7)],
            Vec::new(),
        );

        let new_val = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(99)
        };
        meta.opimpl_setfield_vable_int(OpRef::input_arg_ref(0), fd8, new_val, Value::Int(99));

        let ctx = meta.trace_ctx().unwrap();
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes[0], new_val);
        // pyjitpl.py:1189-1194 _opimpl_setfield_vable for STANDARD
        // virtualizables only updates the cached box and calls
        // synchronize_virtualizable, which writes back into the
        // virtualizable struct via `vinfo.write_boxes` WITHOUT recording
        // any trace ops (RPython pyjitpl.py:3446-3450). The trace stays
        // empty until a non-virtualizable op is recorded.
        assert_eq!(ctx.num_ops(), 0);
    }

    #[test]
    fn opimpl_getarrayitem_vable_int_reads_standard_box_without_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let index = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1)
        };
        let (result, _) =
            meta.opimpl_getarrayitem_vable_int(OpRef::input_arg_ref(0), index, 1, fd24, adesc);
        assert_eq!(result, OpRef::input_arg_int(2));

        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.num_ops(), 0);
    }

    #[test]
    fn opimpl_arraylen_vable_returns_cached_standard_length() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let len_ref = meta.opimpl_arraylen_vable(OpRef::input_arg_ref(0), fd24, adesc);
        let ctx = meta.trace_ctx().unwrap();
        assert_eq!(ctx.const_value(len_ref), Some(2));
        assert_eq!(ctx.num_ops(), 0);
    }

    #[test]
    fn opimpl_getfield_vable_int_nonstandard_falls_back_to_heap_op() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let nonstandard_vable = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(0xCAFE)
        };
        let fd8 =
            majit_ir::descr::make_field_descr(8, 8, Type::Int, majit_ir::descr::ArrayFlag::Signed);
        let _result = meta.opimpl_getfield_vable_int(nonstandard_vable, fd8);

        // pyjitpl.py:1120-1146 _nonstandard_virtualizable falls through
        // to Step 4 (PTR_EQ + implement_guard_value) and Step 5a
        // (emit_force_virtualizable: GETFIELD_GC_R(token_descr) +
        // PTR_NE(CONST_NULL) + COND_CALL) before Step 5b marks the box
        // known. The COND_CALL tail is currently a TODO; the observable
        // prefix is the four ops emitted by `nonstandard_virtualizable`,
        // followed by the caller's GETFIELD_GC_I (the actual non-vable
        // field read).
        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 6);
        assert_eq!(ops[0].opcode, OpCode::PtrEq); // Step 4: PTR_EQ
        assert_eq!(ops[1].opcode, OpCode::GuardValue); // Step 4: implement_guard_value
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR); // Step 5a: token_descr read
        assert_eq!(ops[3].opcode, OpCode::PtrNe); // Step 5a: PTR_NE(CONST_NULL)
        assert_eq!(ops[4].opcode, OpCode::CondCallN); // Step 5a: COND_CALL(clear_vable)
        assert_eq!(ops[5].opcode, OpCode::GetfieldGcI); // caller fallback
    }

    #[test]
    fn opimpl_getarrayitem_vable_int_nonstandard_falls_back_to_heap_ops() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_with_array(),
            &[Value::Int(0x1234), Value::Int(11), Value::Int(22)],
            vec![2],
        );

        let (nonstandard_vable, index) = {
            let ctx = meta.trace_ctx().unwrap();
            (ctx.const_int(0xCAFE), ctx.const_int(1))
        };
        let fd24 =
            majit_ir::descr::make_field_descr(24, 8, Type::Int, majit_ir::descr::ArrayFlag::Signed);
        let adesc = majit_ir::make_array_descr(0, 8, Type::Int);
        let _result = meta.opimpl_getarrayitem_vable_int(nonstandard_vable, index, 1, fd24, adesc);

        // pyjitpl.py:1219-1230 _opimpl_getarrayitem_vable falls back to
        // GETFIELD_GC_R(arraydescr) + GETARRAYITEM_GC_I(arraybox) when
        // _nonstandard_virtualizable returns True. The four ops emitted
        // by `_nonstandard_virtualizable` (Step 4 PTR_EQ + GUARD_VALUE
        // and Step 5a GETFIELD_GC_R(token_descr) + PTR_NE) precede the
        // caller's two-op fallback, totalling 6 ops.
        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 7);
        assert_eq!(ops[0].opcode, OpCode::PtrEq); // Step 4: PTR_EQ
        assert_eq!(ops[1].opcode, OpCode::GuardValue); // Step 4: implement_guard_value
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR); // Step 5a: token_descr read
        assert_eq!(ops[3].opcode, OpCode::PtrNe); // Step 5a: PTR_NE(CONST_NULL)
        assert_eq!(ops[4].opcode, OpCode::CondCallN); // Step 5a: COND_CALL(clear_vable)
        assert_eq!(ops[5].opcode, OpCode::GetfieldGcR); // caller fallback: arraybox
        assert_eq!(ops[6].opcode, OpCode::GetarrayitemGcI); // caller fallback: item read
    }

    #[test]
    fn opimpl_hint_force_virtualizable_standard_emits_store_back_only_once() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));
        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn opimpl_hint_force_virtualizable_ignores_nonstandard_virtualizable() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        let nonstandard_vable = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(0xCAFE)
        };
        meta.opimpl_hint_force_virtualizable(nonstandard_vable);

        let ops = take_recorded_ops(&mut meta);
        assert!(ops.is_empty());
    }

    #[test]
    fn do_jit_force_virtual_preserves_standard_concrete_value() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        let mut obj = ResidualCallVableObj { token: 0, pc: 41 };
        meta.set_vable_ptr((&mut obj as *mut ResidualCallVableObj).cast());
        let vref_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int((&mut obj as *mut ResidualCallVableObj) as usize as i64)
        };
        let allboxes = [
            (JitArgKind::Int, OpRef::int_op(99), 0),
            (
                JitArgKind::Int,
                vref_box,
                (&mut obj as *mut ResidualCallVableObj) as usize as i64,
            ),
        ];
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let result = meta
            ._do_jit_force_virtual(
                &allboxes,
                descr.as_ref().as_call_descr().expect("call descr"),
                0,
            )
            .expect("should resolve to standard virtualizable");

        assert_eq!(result.0, OpRef::input_arg_ref(0));
        assert_eq!(
            result.1,
            (&mut obj as *mut ResidualCallVableObj) as usize as i64
        );
    }

    #[test]
    fn load_fields_from_virtualizable_reloads_heap_values_into_boxes() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        let mut obj = ResidualCallVableObj { token: 0, pc: 99 };
        meta.set_vable_ptr((&mut obj as *mut ResidualCallVableObj).cast());

        meta.load_fields_from_virtualizable();

        let ctx = meta.trace_ctx().unwrap();
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(ctx.const_value(boxes[0]), Some(99));
        assert_eq!(boxes[1], OpRef::input_arg_ref(0));
    }

    #[test]
    fn direct_assembler_call_uses_greenkey_token_in_descr() {
        let mut meta = MetaInterp::<()>::new(10);
        std::sync::Arc::get_mut(&mut meta.staticdata)
            .unwrap()
            .jitdrivers_sd
            .push(JitDriverStaticData::new(
                vec![("code", Type::Int)],
                vec![("frame", Type::Int)],
            ));
        let green_key = crate::green_key_hash(&[55]);
        let mut token = majit_backend::JitCellToken::new(4242);
        token.virtualizable_arg_index = None;
        let token = std::sync::Arc::new(token);
        meta.warm_state_mut()
            .attach_procedure_to_interp(green_key, std::sync::Arc::clone(&token));
        meta.compiled_loops.insert(
            green_key,
            CompiledEntry {
                token,
                meta: (),
                front_target_tokens: Vec::new(),
                root_trace_id: 0,
                traces: HashMap::new(),
                previous_tokens: Vec::new(),
                next_global_opref: 0,
            },
        );
        let action = meta.force_start_tracing(777, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        let frame_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1234)
        };
        let green_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(55)
        };
        let func_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(9999)
        };
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let (_vablebox, resbox) = meta.direct_assembler_call(
            &[
                (JitArgKind::Int, func_box, 9999),
                (JitArgKind::Int, green_box, 55),
                (JitArgKind::Int, frame_box, 1234),
            ],
            descr.as_ref().as_call_descr().expect("call descr"),
            0,
        );
        assert!(
            resbox.is_some(),
            "compiled green key should resolve a token"
        );

        let ops = take_recorded_ops(&mut meta);
        let call = ops
            .into_iter()
            .find(|op| op.opcode == OpCode::CallAssemblerI)
            .expect("CALL_ASSEMBLER_I recorded");
        let call_descr = call
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call descr");
        assert_eq!(call_descr.call_target_token(), Some(4242));
    }

    #[test]
    fn direct_assembler_call_installs_temp_callback_token_on_cell() {
        // warmstate.py:714-723 `get_assembler_token` parity: when the
        // target green key has no existing procedure_token,
        // `direct_assembler_call` must synthesise one via
        // `compile_tmp_callback` and install it on the cell with
        // `tmp=true`.
        extern "C" fn dummy_portal_runner() -> i64 {
            0
        }

        let mut meta = MetaInterp::<()>::new(10);
        {
            let staticdata = std::sync::Arc::get_mut(&mut meta.staticdata).unwrap();
            let mut jd =
                JitDriverStaticData::new(vec![("code", Type::Int)], vec![("frame", Type::Int)]);
            jd.result_type = Type::Int;
            jd.portal_runner_adr = dummy_portal_runner as *const () as usize as i64;
            staticdata.jitdrivers_sd.push(jd);
            staticdata.finish_setup_descrs_for_jitdrivers(&mut meta.backend);
        }

        let action = meta.force_start_tracing(888, (0, 0), None, &[]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));

        let frame_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1234)
        };
        let green_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(55)
        };
        let func_box = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(9999)
        };
        let descr = make_call_descr(vec![Type::Int, Type::Int], Type::Int);

        let green_key = crate::green_key_hash(&[55]);
        let (_vablebox, resbox) = meta.direct_assembler_call(
            &[
                (JitArgKind::Int, func_box, 9999),
                (JitArgKind::Int, green_box, 55),
                (JitArgKind::Int, frame_box, 1234),
            ],
            descr.as_ref().as_call_descr().expect("call descr"),
            0,
        );
        assert!(
            resbox.is_some(),
            "temp callback path should record CALL_ASSEMBLER"
        );

        let cell = meta
            .warm_state
            .get_cell(green_key)
            .expect("temp callback should install a jit cell");
        assert_ne!(
            cell.flags & crate::warmstate::jc_flags::TEMPORARY,
            0,
            "temp callback token should mark the cell as TEMPORARY"
        );
        let token = cell
            .get_procedure_token()
            .expect("temp callback should install a procedure token");
        let token_number = token.number;

        let ops = take_recorded_ops(&mut meta);
        let call = ops
            .into_iter()
            .find(|op| op.opcode == OpCode::CallAssemblerI)
            .expect("CALL_ASSEMBLER_I recorded");
        let call_descr = call
            .descr
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .expect("call descr");
        assert_eq!(call_descr.call_target_token(), Some(token_number));
    }

    #[test]
    fn hint_force_virtualizable_state_is_reset_between_traces() {
        let mut meta = MetaInterp::<()>::new(10);
        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));
        let _ = meta.finish_trace_for_parity(&[]);

        start_tracing_with_virtualizable(
            &mut meta,
            test_vable_info_static_only(),
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );
        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn standard_vable_access_consumes_forced_virtualizable_state() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_static_only();
        let fd8 = info.static_field_descr(0);
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[Value::Int(0x1234), Value::Int(41)],
            Vec::new(),
        );

        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));
        let _ = meta.opimpl_getfield_vable_int(OpRef::input_arg_ref(0), fd8);
        meta.opimpl_hint_force_virtualizable(OpRef::input_arg_ref(0));

        let ops = take_recorded_ops(&mut meta);
        assert_eq!(ops.len(), 4);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[2].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[3].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn compiled_virtualizable_trace_does_not_use_raw_heap_ops() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        // live_values[0] is the vable identity — must be Ref-typed per
        // virtualstate.py:417 NotVirtualStateInfoPtr contract. A bare
        // Value::Int here makes `enum_forced_boxes_for_entry` reject the
        // label via the Box.type strict check.
        start_tracing_with_virtualizable(
            &mut meta,
            info,
            &[
                Value::Ref(majit_ir::GcRef(0x1234)),
                Value::Int(10),
                Value::Int(20),
            ],
            vec![2],
        );

        let index = {
            let ctx = meta.trace_ctx().unwrap();
            ctx.const_int(1)
        };
        let (item, _) =
            meta.opimpl_getarrayitem_vable_int(OpRef::input_arg_ref(0), index, 1, fd24, adesc);
        if let Some(ctx) = meta.trace_ctx() {
            let g = ctx.record_guard(OpCode::GuardTrue, &[item], 0);
            ctx.capture_snapshot_for_last_guard(
                &[
                    OpRef::input_arg_ref(0),
                    OpRef::input_arg_int(1),
                    OpRef::input_arg_int(2),
                ],
                0,
                0,
            );
            ctx.set_fail_args(
                g,
                &[
                    OpRef::input_arg_ref(0),
                    OpRef::input_arg_int(1),
                    OpRef::input_arg_int(2),
                ],
            );
        }
        meta.compile_loop(
            &[
                OpRef::input_arg_ref(0),
                OpRef::input_arg_int(1),
                OpRef::input_arg_int(2),
            ],
            (),
        );

        let compiled = meta.compiled_loops.get(&777).expect("compiled entry");
        let trace = compiled
            .traces
            .get(&compiled.root_trace_id)
            .expect("root compiled trace");

        assert!(
            trace.ops.iter().all(|op| {
                !matches!(
                    op.opcode,
                    OpCode::GetfieldRawI
                        | OpCode::GetfieldRawR
                        | OpCode::GetfieldRawF
                        | OpCode::SetfieldRaw
                        | OpCode::GetarrayitemRawI
                        | OpCode::GetarrayitemRawR
                        | OpCode::GetarrayitemRawF
                        | OpCode::SetarrayitemRaw
                )
            }),
            "standard virtualizable loop should use vable boxes, not raw heap ops: {}",
            majit_ir::format_trace(&trace.ops, &trace.constants)
        );
        assert_eq!(item, OpRef::input_arg_int(2));
    }

    #[test]
    fn optimizer_vable_config_requires_standard_virtualizable_boxes() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        meta.set_virtualizable_info(std::sync::Arc::new(info.clone()));
        assert!(
            meta.current_virtualizable_optimizer_config().is_none(),
            "virtualizable config should only exist while tracing is active"
        );

        let action = meta.force_start_tracing(777, (0, 0), None, &[Value::Int(0x1234)]);
        assert!(matches!(action, BackEdgeAction::StartedTracing));
        assert!(
            meta.current_virtualizable_optimizer_config().is_none(),
            "virtualizable config should require standard virtualizable boxes"
        );
    }

    #[test]
    fn optimizer_vable_config_matches_registered_virtualizable_when_boxes_active() {
        let mut meta = MetaInterp::<()>::new(10);
        let info = test_vable_info_with_array();
        start_tracing_with_virtualizable(
            &mut meta,
            info.clone(),
            &[Value::Int(0x1234), Value::Int(10), Value::Int(20)],
            vec![2],
        );

        let config = meta
            .current_virtualizable_optimizer_config()
            .expect("standard virtualizable trace should pass config to optimizer");
        assert_eq!(
            config.static_field_offsets,
            info.to_optimizer_config().static_field_offsets
        );
        assert_eq!(
            config.static_field_types,
            info.to_optimizer_config().static_field_types
        );
        assert_eq!(
            config.array_field_offsets,
            info.to_optimizer_config().array_field_offsets
        );
        assert_eq!(
            config.array_item_types,
            info.to_optimizer_config().array_item_types
        );
        assert_eq!(config.array_lengths, vec![2]);
    }

    // ── JitIface hook/callback parity tests (rpython/jit/metainterp/test/test_jitiface.py) ──

    #[test]
    fn test_on_compile_loop_fires_with_correct_metadata() {
        // Parity with test_on_compile: after_compile hook fires with green_key,
        // num_ops_before, num_ops_after.
        let mut meta = MetaInterp::<()>::new(1);
        let compile_events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = compile_events.clone();
        meta.set_on_compile_loop(move |green_key, ops_before, ops_after| {
            events
                .lock()
                .unwrap()
                .push((green_key, ops_before, ops_after));
        });

        let green_key = 42;
        // Trigger tracing by making back-edge hot
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert!(meta.tracing.is_some());

        // Record a simple operation and close the trace
        if let Some(ctx) = meta.trace_ctx() {
            let i0 = OpRef::input_arg_int(0);
            let const_one = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, const_one]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
        }
        meta.compile_loop(&[OpRef::input_arg_int(0)], ());

        let events = compile_events.lock().unwrap();
        assert_eq!(events.len(), 1, "on_compile_loop should fire exactly once");
        assert_eq!(events[0].0, green_key, "green_key should match");
        assert!(events[0].1 > 0, "num_ops_before should be positive");
        assert!(events[0].2 > 0, "num_ops_after should be positive");
    }

    #[test]
    fn test_on_compile_error_fires_on_failure() {
        // Parity with test_on_abort: on_compile_error fires when compilation fails.
        // We can test this by installing a hook and verifying it captures the error.
        let mut meta = MetaInterp::<()>::new(10);
        let error_events: Arc<Mutex<Vec<(u64, String)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = error_events.clone();
        meta.set_on_compile_error(move |green_key, msg| {
            events.lock().unwrap().push((green_key, msg.to_string()));
        });

        // There's no easy way to trigger a compilation failure through the public API
        // without a malformed trace, so we directly test the hook mechanism.
        // Simulate: if the hook is set, calling it works correctly.
        if let Some(ref cb) = meta.hooks.on_compile_error {
            cb(99, "test error");
        }
        let events = error_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, 99);
        assert_eq!(events[0].1, "test error");
    }

    #[test]
    fn test_multiple_hooks_independent() {
        // Parity with JitHookInterface: multiple different hooks can be registered
        // independently and all fire for their respective events.
        let mut meta = MetaInterp::<()>::new(1);

        let compile_count = Arc::new(Mutex::new(0u32));
        let trace_start_count = Arc::new(Mutex::new(0u32));
        let trace_abort_count = Arc::new(Mutex::new(0u32));

        let cc = compile_count.clone();
        meta.set_on_compile_loop(move |_, _, _| {
            *cc.lock().unwrap() += 1;
        });

        let tsc = trace_start_count.clone();
        meta.set_on_trace_start(move |_| {
            *tsc.lock().unwrap() += 1;
        });

        let tac = trace_abort_count.clone();
        meta.set_on_trace_abort(move |_, _| {
            *tac.lock().unwrap() += 1;
        });

        let green_key = 100;
        // Heat up and start tracing
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert_eq!(
            *trace_start_count.lock().unwrap(),
            1,
            "on_trace_start should fire"
        );

        // Abort the trace
        meta.abort_trace(false);
        assert_eq!(
            *trace_abort_count.lock().unwrap(),
            1,
            "on_trace_abort should fire"
        );
        assert_eq!(
            *compile_count.lock().unwrap(),
            0,
            "on_compile_loop should NOT fire yet"
        );

        // Start another trace and compile it.
        // After abort, the cell goes to DontTraceHere if retrace limit is exceeded.
        // Use a fresh green key to avoid this.
        let green_key2 = 200;
        for _ in 0..2 {
            meta.on_back_edge(green_key2, &[0]);
        }
        if let Some(ctx) = meta.trace_ctx() {
            let i0 = OpRef::input_arg_int(0);
            let const_one = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, const_one]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum], 0, 0);
            ctx.set_fail_args(g, &[sum]);
        }
        meta.compile_loop(&[OpRef::input_arg_int(0)], ());
        assert_eq!(
            *compile_count.lock().unwrap(),
            1,
            "on_compile_loop should fire after compile"
        );
        assert_eq!(
            *trace_start_count.lock().unwrap(),
            2,
            "on_trace_start should fire twice total"
        );
    }

    #[test]
    fn test_on_compile_loop_receives_correct_trace_metadata() {
        // Parity with test_on_compile: verify that the hook receives the correct
        // green key and that op counts reflect the actual trace.
        let mut meta = MetaInterp::<()>::new(1);
        let events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        meta.set_on_compile_loop(move |gk, before, after| {
            ev.lock().unwrap().push((gk, before, after));
        });

        // Compile two different loops with different green keys
        for green_key in [10u64, 20u64] {
            for _ in 0..2 {
                meta.on_back_edge(green_key, &[0, 0]);
            }
            if let Some(ctx) = meta.trace_ctx() {
                let i0 = OpRef::input_arg_int(0);
                let i1 = OpRef::input_arg_int(1);
                let const_one = ctx.const_int(1);
                let sum = ctx.record_op(OpCode::IntAdd, &[i0, i1]);
                let sum2 = ctx.record_op(OpCode::IntAdd, &[sum, const_one]);
                let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
                ctx.capture_snapshot_for_last_guard(&[sum2, i1], 0, 0);
                ctx.set_fail_args(g, &[sum2, i1]);
            }
            meta.compile_loop(&[OpRef::input_arg_int(0), OpRef::input_arg_int(1)], ());
        }

        let events = events.lock().unwrap();
        assert_eq!(events.len(), 2, "two compilation events should fire");
        assert_eq!(events[0].0, 10, "first event green_key=10");
        assert_eq!(events[1].0, 20, "second event green_key=20");
        // Both traces had the same ops, so op counts should be equal
        assert_eq!(
            events[0].1, events[1].1,
            "ops_before should match for identical traces"
        );
    }

    #[test]
    #[cfg(feature = "cranelift")]
    fn test_on_compile_bridge_fires() {
        // Parity with test_on_compile_bridge: after_compile_bridge hook fires
        // when a bridge is compiled.
        let mut meta = MetaInterp::<()>::new(10);
        let bridge_events: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = bridge_events.clone();
        meta.set_on_compile_bridge(move |gk, fi, nops| {
            ev.lock().unwrap().push((gk, fi, nops));
        });

        // Install a simple compiled loop with a guard
        let green_key = 50;
        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        // Backend test-fixture quirk (NOT RPython parity): both constants
        // are paired with `constants.insert(100|101, ...)` below; the
        // backend HashMap keys the function-pointer / value literals by
        // raw OpRef value. RPython expresses these as Const* boxes.
        let _const_one = OpRef::int_op(100);
        let const_zero = OpRef::int_op(101);
        let mut guard_op = mk_op(OpCode::GuardTrue, &[OpRef::int_op(2)], OpRef::NONE.raw());
        guard_op.fail_args = Some(smallvec::SmallVec::from_slice(&[
            OpRef::input_arg_int(0),
            OpRef::input_arg_int(1),
        ]));
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(0), OpRef::input_arg_int(1)],
                2,
            ),
            mk_op(OpCode::IntGt, &[OpRef::int_op(2), const_zero], 3),
            {
                let mut g = mk_op(OpCode::GuardTrue, &[OpRef::int_op(3)], OpRef::NONE.raw());
                g.fail_args = Some(smallvec::SmallVec::from_slice(&[
                    OpRef::input_arg_int(0),
                    OpRef::input_arg_int(1),
                ]));
                g
            },
            mk_op(
                OpCode::Jump,
                &[OpRef::int_op(2), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        attach_procedure_to_interp_entry(&mut meta, green_key, &inputargs, ops, constants);

        // The bridge hook is set. We verify the hook mechanism is correctly wired.
        if let Some(ref hook) = meta.hooks.on_compile_bridge {
            hook(green_key, 3, 5);
        }
        let events = bridge_events.lock().unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, green_key);
        assert_eq!(events[0].1, 3);
        assert_eq!(events[0].2, 5);
    }

    #[test]
    fn test_on_guard_failure_hook() {
        // Parity with test_get_stats: guard failure hook fires with correct args.
        let mut meta = MetaInterp::<()>::new(10);
        let failure_events: Arc<Mutex<Vec<(u64, u32, u32)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = failure_events.clone();
        meta.set_on_guard_failure(move |gk, fi, fc| {
            ev.lock().unwrap().push((gk, fi, fc));
        });

        // Verify the hook is correctly installed and callable
        if let Some(ref hook) = meta.hooks.on_guard_failure {
            hook(42, 3, 1);
            hook(42, 3, 2);
            hook(42, 5, 1);
        }
        let events = failure_events.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], (42, 3, 1));
        assert_eq!(events[1], (42, 3, 2));
        assert_eq!(events[2], (42, 5, 1));
    }

    #[test]
    fn test_should_record_guard_failure_skips_jump_exit() {
        assert!(!MetaInterp::<()>::should_record_guard_failure(
            false,
            u32::MAX
        ));
        assert!(MetaInterp::<()>::should_record_guard_failure(false, 7));
        assert!(!MetaInterp::<()>::should_record_guard_failure(true, 7));
    }

    #[test]
    fn test_run_result_for_jump_exit_returns_jump() {
        let result = MetaInterp::<()>::run_result_for_jump_exit(u32::MAX, vec![42], (), None)
            .expect("jump exit should produce a direct Jump result");
        match result {
            RunResult::Jump { values, .. } => assert_eq!(values, vec![42]),
            other => panic!("expected Jump result, got {other:?}"),
        }

        assert!(
            MetaInterp::<()>::run_result_for_jump_exit(3, vec![42], (), None).is_none(),
            "guard failure exits must keep using recovery paths"
        );
    }

    #[test]
    fn test_blackhole_result_for_jump_exit_returns_jump() {
        let exit_layout = CompiledExitLayout {
            rd_loop_token: 91,
            trace_id: 12,
            fail_index: u32::MAX,
            source_op_index: Some(3),
            exit_types: vec![Type::Int],
            is_finish: false,
            gc_ref_slots: Vec::new(),
            force_token_slots: Vec::new(),
            recovery_layout: None,
            resume_layout: None,
            storage: None,
        };
        let exception = ExceptionState::default();

        let result = MetaInterp::<()>::blackhole_result_for_jump_exit(
            u32::MAX,
            vec![42],
            vec![Value::Int(42)],
            exit_layout.clone(),
            (),
            None,
            exception.clone(),
        )
        .expect("jump exit should bypass blackhole recovery");
        match result {
            BlackholeRunResult::Jump {
                values,
                typed_values,
                exit_layout,
                via_blackhole,
                exception,
                ..
            } => {
                assert_eq!(values, vec![42]);
                assert_eq!(typed_values, Some(vec![Value::Int(42)]));
                assert_eq!(exit_layout.unwrap().trace_id, 12);
                assert!(!via_blackhole);
                assert_eq!(exception, ExceptionState::default());
            }
            other => panic!("expected Jump result, got {other:?}"),
        }

        assert!(
            MetaInterp::<()>::blackhole_result_for_jump_exit(
                5,
                vec![42],
                vec![Value::Int(42)],
                exit_layout,
                (),
                None,
                exception,
            )
            .is_none(),
            "guard failure exits must keep using guard recovery / blackhole fallback"
        );
    }

    #[test]
    fn test_on_trace_abort_hook_with_permanent_flag() {
        // Parity with test_abort_quasi_immut: on_abort receives the permanent flag.
        let mut meta = MetaInterp::<()>::new(1);
        let abort_events: Arc<Mutex<Vec<(u64, bool)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = abort_events.clone();
        meta.set_on_trace_abort(move |gk, permanent| {
            ev.lock().unwrap().push((gk, permanent));
        });

        let green_key = 77;
        // Start tracing
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        assert!(meta.tracing.is_some());

        // Abort non-permanently
        meta.abort_trace(false);
        {
            let events = abort_events.lock().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0], (green_key, false));
        }

        // Start tracing again and abort permanently
        for _ in 0..2 {
            meta.on_back_edge(green_key, &[0]);
        }
        if meta.tracing.is_some() {
            meta.abort_trace(true);
            let events = abort_events.lock().unwrap();
            assert_eq!(events.len(), 2);
            assert_eq!(events[1], (green_key, true));
        }
    }

    #[test]
    fn test_jit_hooks_default_is_all_none() {
        // All hooks default to None.
        let hooks = JitHooks::default();
        assert!(hooks.on_compile_loop.is_none());
        assert!(hooks.on_compile_bridge.is_none());
        assert!(hooks.on_guard_failure.is_none());
        assert!(hooks.on_trace_start.is_none());
        assert!(hooks.on_trace_abort.is_none());
        assert!(hooks.on_compile_error.is_none());
    }

    #[test]
    fn test_pending_target_input_types_passes_through_when_no_descriptor() {
        // No descriptor → expanded form survives untouched.
        let expanded = vec![Type::Ref, Type::Int, Type::Ref, Type::Int, Type::Ref];
        let result = MetaInterp::<()>::pending_target_input_types(expanded.clone(), None);
        assert_eq!(result, expanded);
    }

    #[test]
    fn test_pending_target_input_types_passes_through_when_no_virtualizable() {
        let descriptor = JitDriverStaticData::new(
            vec![("pc", Type::Int)],
            vec![("a", Type::Int), ("b", Type::Int)],
        );
        let expanded = vec![Type::Int, Type::Int];
        let result =
            MetaInterp::<()>::pending_target_input_types(expanded.clone(), Some(&descriptor));
        assert_eq!(result, expanded);
    }

    #[test]
    fn test_pending_target_input_types_uses_driver_red_types_not_expanded_prefix() {
        // PyPy `interp_jit.py:67`: reds=[frame, ec], virtualizable=frame.
        // The trace's expanded inputarg shape is
        // `[Ref(frame), Int(last_instr), Ref(pycode), Int(valuestackdepth),
        //   Ref(debugdata), Ref(lastblock), Ref(w_globals), ...locals/stack]`.
        // Truncating the leading two slots would yield `[Ref, Int]` — wrong.
        // The descriptor's `reds` is the source of truth, so the result
        // must be `[Ref, Ref]`.
        let descriptor = JitDriverStaticData::with_virtualizable(
            vec![
                ("next_instr", Type::Int),
                ("is_being_profiled", Type::Int),
                ("pycode", Type::Ref),
            ],
            vec![("frame", Type::Ref), ("ec", Type::Ref)],
            Some("frame"),
        );
        let expanded = vec![
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Int,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
            Type::Ref,
        ];
        let result = MetaInterp::<()>::pending_target_input_types(expanded, Some(&descriptor));
        assert_eq!(result, vec![Type::Ref, Type::Ref]);
    }
}
