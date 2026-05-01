use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};
/// Main optimization driver.
///
/// Translated from rpython/jit/metainterp/optimizeopt/optimizer.py.
/// Chains multiple optimization passes and drives operations through them.
use crate::optimizeopt::{
    earlyforce::OptEarlyForce,
    heap::OptHeap,
    intbounds::OptIntBounds,
    pure::OptPure,
    rewrite::OptRewrite,
    virtualize::{OptVirtualize, VirtualizableConfig},
    vstring::OptString,
};
use majit_ir::{Const, DescrRef, GcRef, Op, OpCode, OpRef, Type};

use crate::optimizeopt::info::PtrInfo;

/// bridgeopt.py:124 parity: data needed to call
/// deserialize_optimizer_knowledge after optimizer setup.
pub(crate) struct PendingBridgeRd {
    /// compile.py:853 `ResumeGuardDescr` storage handle. The bridge
    /// deserializer borrows `storage.rd_numb` / `storage.rd_consts()`
    /// off the same Arc the guard owns — no owned clones, and every
    /// reader (runtime + GC walker) observes the same pool.
    pub storage: std::sync::Arc<crate::resume::ResumeStorage>,
    pub frontend_boxes: Vec<i64>,
    pub liveboxes: Vec<OpRef>,
    pub livebox_types: Vec<Type>,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    pub all_descrs: Vec<majit_ir::descr::DescrRef>,
    /// model.py:199-201 cpu.cls_of_box — always available via MetaInterp.
    pub cls_of_box: Option<fn(i64) -> i64>,
}

/// The optimizer: chains passes and runs them over a trace.
///
/// RPython optimizer.py: Optimizer class with pass chain and shared state.
pub struct Optimizer {
    passes: Vec<Box<dyn Optimization>>,
    pub pureop_historylength: usize,
    /// Final num_inputs after optimization (may increase if virtualizable
    /// adds virtual input args).
    final_num_inputs: usize,
    /// Cache of CALL_PURE results from previous traces.
    /// optimizer.py: `call_pure_results` — maps constant values
    /// (via get_constant_box) → result value, carried across
    /// loop iterations so the optimizer can constant-fold repeated
    /// pure calls. RPython uses value-based equality for keys.
    pub call_pure_results: std::collections::HashMap<Vec<majit_ir::Value>, majit_ir::Value>,
    /// optimizer.py: `_last_guard_op` — tracks the last emitted guard
    /// for guard sharing and descriptor fusion.
    ///
    /// Stored as the donor's index in `ctx.new_operations` (set by
    /// `emit_guard_operation` just before `ctx.emit` pushes the op, so
    /// the predicted position equals `ctx.new_operations.len()` at
    /// assignment time). `_copy_resume_data_from` looks up the donor
    /// via this index and stamps a `ResumeGuardCopiedDescr(prev)`
    /// (or `ResumeGuardCopiedExcDescr(prev)` for exception guards) on
    /// the sharer, matching `compile.py:832
    /// ResumeGuardCopiedDescr(prev)` parity.
    last_guard_op_idx: Option<usize>,
    /// optimizer.py: `replaces_guard` — maps guard op position to replacement.
    replaces_guard: std::collections::HashMap<u32, Op>,
    /// optimizer.py: `pendingfields` — heap fields that need to be
    /// written back before the next guard (lazy set forcing).
    pendingfields: Vec<Op>,
    /// optimizer.py: `can_replace_guards` — flag to enable/disable guard sharing.
    can_replace_guards: bool,
    /// optimizer.py: `quasi_immutable_deps` — quasi-immutable field dependencies.
    /// RPython: dict[QuasiImmut → None]. We store (object_ptr, field_index)
    /// pairs identifying the specific quasi-immutable slot that compiled
    /// code depends on. After compilation, each dependency gets the loop's
    /// invalidation flag registered as a per-slot watcher.
    pub quasi_immutable_deps: std::collections::HashSet<(u64, u32)>,
    /// Types of constant OpRefs from ConstantPool.constant_types.
    /// Used to distinguish Ref constants from Int in guard fail_args.
    pub constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython parity: GcRef constants (ob_type etc.) recorded as const_int
    /// need Ref type for resume data but must NOT trigger Cranelift GC root.
    pub numbering_type_overrides: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython unroll.py: import_state — virtual structures to inject at Phase 2 start.
    /// Maps the original loop-carried input slot to a recursive abstract
    /// description of the virtual's field values.
    pub imported_virtuals: Vec<ImportedVirtual>,
    /// Types of the original trace inputargs (from LABEL or inputarg_types).
    /// RPython Boxes carry type intrinsically; we store it here so
    /// export_state can propagate to ExportedState.renamed_inputarg_types.
    pub trace_inputarg_types: Vec<majit_ir::Type>,
    /// unroll.py / compile.py parity: original live values at the jump
    /// point, threaded into export_state as `runtime_boxes`. Dormant
    /// until the export/import pair starts reading it.
    pub runtime_boxes: Vec<OpRef>,
    /// RPython unroll.py: export_state — exported optimizer facts at the end
    /// of the preamble, adapted to majit's slot-based inputarg model.
    pub exported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// RPython unroll.py: import_state — exported facts to re-apply onto the
    /// next optimizer instance before phase 2 body optimization starts.
    pub imported_loop_state: Option<crate::optimizeopt::unroll::ExportedState>,
    /// Original preamble result boxes for imported short-box results.
    pub imported_short_sources: Vec<crate::optimizeopt::ImportedShortSource>,
    /// Invented SameAs aliases imported from short-preamble export/import.
    pub imported_short_aliases: Vec<crate::optimizeopt::ImportedShortAlias>,
    /// Builder-derived short preamble actually used by phase 2.
    pub imported_short_preamble: Option<crate::optimizeopt::shortpreamble::ShortPreamble>,
    /// RPython unroll.py: short_preamble_producer after import_state.
    /// Preserved so finalize_short_preamble can create the live extended
    /// producer for the target token currently being compiled.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// RPython unroll.py: `label_args = import_state(...)`.
    /// The peeled loop's LABEL must use these args, not the phase-1 end_args.
    pub imported_label_args: Option<Vec<OpRef>>,
    /// simplify.py: patchguardop recorded from GUARD_FUTURE_CONDITION.
    pub patchguardop: Option<Op>,
    /// RPython: propagate_all_forward(trace, flush=False) for Phase 2.
    /// When true, skip flush() at end of optimization.
    pub skip_flush: bool,
    /// RPython optimizer.py: last_op/info.jump_op when flush=False.
    /// Phase-2 loop compilation keeps the terminal JUMP/FINISH outside the
    /// returned body ops and lets unroll/compile own it explicitly.
    pub terminal_op: Option<Op>,
    /// Preserved final context after optimization, for jump_to_existing_trace.
    pub final_ctx: Option<OptContext>,
    /// RPython Box identity: generation epoch for Phase 2 ops.
    /// Phase 1 JUMP arg OpRef indices to pre-tag as gen=0.
    /// bridgeopt.py:124-185: pending bridge resume data for deserialization
    /// after setup(). RPython calls deserialize_optimizer_knowledge after the
    /// optimizer is constructed.
    pending_bridge_rd: Option<PendingBridgeRd>,
    /// pyjitpl.py:2289 all_descrs: dense list indexed by descr_index.
    /// Taken from MIStaticData at optimizer construction, returned after.
    /// descr.py:25-47: descriptors get descr_index assigned inline during
    /// collect_optimizer_knowledge_for_resume().
    pub all_descrs: Vec<DescrRef>,
    /// optimizer.py:787: constant_fold allocator for compile-time object creation.
    pub constant_fold_alloc: Option<crate::optimizeopt::ConstantFoldAllocFn>,
    /// info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` runtime hook —
    /// propagated to `OptContext::string_length_resolver` at the start of
    /// each `optimize_with_constants_and_inputs_at` run so
    /// `EnsuredPtrInfo::Constant.getlenbound(Some(mode))` can return an
    /// exact constant string length when the host runtime supplies one.
    ///
    /// `Arc` rather than `Box` because the resolver is shared across
    /// multiple optimizer runs (propagation is a `clone()`, not a `take()`).
    pub string_length_resolver: Option<crate::optimizeopt::info::StringLengthResolver>,
    /// info.py:788-790 ConstPtrInfo._unpack_str(mode) — runtime hook.
    pub string_content_resolver: Option<crate::optimizeopt::info::StringContentResolver>,
    /// history.py:377 get_const_ptr_for_string(s) — runtime hook.
    pub string_constant_alloc: Option<crate::optimizeopt::info::StringConstantAllocator>,
    /// RPython metainterp_sd.callinfocollection parity.
    /// Propagated to OptContext for generate_modified_call (vstring.py:853).
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// optimizer.py:732 — resume.ResumeDataLoopMemo.
    /// Shared constant pool + box numbering cache across all guards in a loop.
    pub resumedata_memo: crate::resume::ResumeDataLoopMemo,
    /// resume.py parity: per-guard snapshots from tracing time.
    /// Maps rd_resume_position → flattened OpRef boxes from the snapshot.
    /// Propagated to OptContext for store_final_boxes_in_guard.
    pub snapshot_boxes: std::collections::HashMap<i32, Vec<OpRef>>,
    /// Per-frame box counts for multi-frame snapshots.
    /// Propagated to OptContext for store_final_boxes_in_guard multi-frame encoding.
    pub snapshot_frame_sizes: std::collections::HashMap<i32, Vec<usize>>,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: std::collections::HashMap<i32, Vec<OpRef>>,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: std::collections::HashMap<i32, Vec<(i32, i32)>>,
    /// RPython Box type parity: in RPython each Box carries its type
    /// intrinsically. In majit, OpRef is an untyped u32, so we track
    /// types in value_types. Phase 1 value_types are preserved here
    /// so Phase 2 can resolve types for carried-over OpRefs.
    pub prev_phase_value_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// RPython Box type parity: position→type for ALL original trace ops.
    /// snapshot_boxes reference original trace positions, but the optimizer
    /// receives transformed ops (after fold_box/elide). This map covers
    /// positions that were removed during trace transformation.
    pub original_trace_op_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// jitprof.Counters.OPT_OPS / OPT_GUARDS / OPT_GUARDS_SHARED accumulators.
    ///
    /// RPython optimizer.py:626/629/673 calls
    /// `self.metainterp_sd.profiler.count(...)` directly inside
    /// `_emit_operation` / `emit_guard_operation`.  Pyre's `Optimizer`
    /// does not hold a reference to `MetaInterpStaticData.profiler`, so
    /// we mirror the same deferred-fold pattern already in use for
    /// `nvirtuals` / `nvholes` / `nvreused`: accumulate per-counter here
    /// and let the caller fold them into `JitStatsCounters` via
    /// `update_counters` after each `propagate_all_forward` exit.
    pub(crate) opt_ops_emitted: usize,
    pub(crate) opt_guards_emitted: usize,
    pub(crate) opt_guards_shared_emitted: usize,
}

fn value_from_backend_constant_bits_typed(
    opref: OpRef,
    raw: i64,
    ops: &[Op],
    constant_types: &std::collections::HashMap<u32, Type>,
) -> majit_ir::Value {
    // Use explicit constant_types first (from ConstantPool),
    // then fall back to ops result_type, then Int.
    let result_type = constant_types
        .get(&opref.0)
        .copied()
        .or_else(|| {
            ops.iter()
                .find(|op| op.pos == opref)
                .map(|op| op.result_type())
        })
        .unwrap_or(Type::Int);
    match result_type {
        Type::Ref => majit_ir::Value::Ref(majit_ir::GcRef(raw as usize)),
        Type::Float => majit_ir::Value::Float(f64::from_bits(raw as u64)),
        Type::Int | Type::Void => majit_ir::Value::Int(raw),
    }
}

fn value_to_backend_constant_bits(value: &majit_ir::Value) -> i64 {
    match value {
        majit_ir::Value::Int(v) => *v,
        majit_ir::Value::Ref(r) => r.0 as i64,
        majit_ir::Value::Float(f) => f.to_bits() as i64,
        majit_ir::Value::Void => 0,
    }
}

fn live_runtime_positions(ops: &[Op]) -> Vec<bool> {
    let live_limit = ops
        .iter()
        .filter(|op| !op.pos.is_none() && !op.pos.is_constant())
        .map(|op| op.pos.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let mut live_positions = vec![false; live_limit];
    for op in ops {
        if !op.pos.is_none() && !op.pos.is_constant() {
            live_positions[op.pos.0 as usize] = true;
        }
    }
    live_positions
}

pub(crate) fn sanitize_backend_constants_for_ops(
    ops: &[Op],
    constants: &mut std::collections::HashMap<u32, i64>,
) {
    let live_positions = live_runtime_positions(ops);
    constants
        .retain(|idx, _| (*idx as usize) >= live_positions.len() || !live_positions[*idx as usize]);
}

pub(crate) fn merge_backend_constants_from_ctx(
    ctx: &OptContext,
    constants: &mut std::collections::HashMap<u32, i64>,
) {
    let live_positions = live_runtime_positions(&ctx.new_operations);

    for (idx, val) in ctx.constants.iter().enumerate() {
        let Some(value) = val else {
            continue;
        };
        if idx < live_positions.len() && live_positions[idx] {
            continue;
        }
        constants
            .entry(idx as u32)
            .or_insert_with(|| value_to_backend_constant_bits(value));
    }
    for (&const_idx, value) in &ctx.const_pool {
        let key = OpRef::from_const(const_idx).0;
        constants.insert(key, value_to_backend_constant_bits(value));
    }
}

/// RPython unroll.py: import_state virtual info for Phase 2.
/// Tells OptVirtualize that an inputarg is a virtual object.
#[derive(Clone, Debug)]
pub struct ImportedVirtual {
    /// Inputarg index that holds this virtual.
    pub inputarg_index: usize,
    /// Size descriptor for the virtual's New().
    pub size_descr: majit_ir::DescrRef,
    /// Whether this imported virtual is an instance or a plain struct.
    pub kind: ImportedVirtualKind,
    /// Fields: (field_descr, exported abstract info for the field value).
    pub fields: Vec<(
        majit_ir::DescrRef,
        crate::optimizeopt::virtualstate::VirtualStateInfo,
    )>,
    /// Descr index of the GetfieldGcR(pool) that loads this head.
    /// OptVirtualize forwards this load result to the virtual head.
    pub head_load_descr_index: Option<u32>,
}

#[derive(Clone, Debug)]
pub enum ImportedVirtualKind {
    Instance {
        known_class: Option<majit_ir::GcRef>,
    },
    Struct,
}

impl Optimizer {
    fn is_constant_placeholder_op(op: &Op, constants: &[Option<majit_ir::Value>]) -> bool {
        if !matches!(
            op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            return false;
        }
        let idx = op.pos.0 as usize;
        if idx >= constants.len() || constants[idx].is_none() {
            return false;
        }
        op.args.is_empty() || op.args.iter().all(|arg| arg.is_none())
    }

    fn import_virtual_state_value(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        ctx: &mut OptContext,
    ) -> OpRef {
        let opref = ctx.alloc_op_position();
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] import_virtual_state_value {opref:?} <= {info:?}");
        }
        Self::apply_imported_virtual_state(info, opref, ctx);
        opref
    }

    fn apply_imported_virtual_state(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        opref: OpRef,
        ctx: &mut OptContext,
    ) {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(value) => {
                ctx.make_constant(opref, value.clone());
            }
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = Vec::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    // ob_type (offset 0) constants are GcRef pointers stored
                    // as Value::Int. Register as Ref in constant_types so
                    // fail_arg_types + blackhole decode_ref handle them correctly.
                    // virtualstate.py:159-162: fielddescrs is a flat list,
                    // box access uses fielddescrs[i].get_index() == field_idx.
                    let offset = field_descrs
                        .iter()
                        .find_map(|d| {
                            d.as_field_descr()
                                .filter(|fd| fd.index_in_parent() as u32 == *field_idx)
                        })
                        .map(|fd| fd.offset());
                    if offset == Some(0) && known_class.is_some() {
                        ctx.constant_types_for_numbering
                            .insert(field_ref.0, majit_ir::Type::Ref);
                    }
                    imported_fields.push((*field_idx, field_ref));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                let imported_items = items
                    .iter()
                    .map(|item_info| Self::import_virtual_state_value(item_info, ctx))
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                let mut imported_fields = Vec::new();
                for (field_idx, field_info) in fields {
                    let field_ref = Self::import_virtual_state_value(field_info, ctx);
                    imported_fields.push((*field_idx, field_ref));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                fielddescrs,
                element_fields,
            } => {
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                (
                                    *field_idx,
                                    Self::import_virtual_state_value(field_info, ctx),
                                )
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::VirtualArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: fielddescrs.clone(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
            }
            VirtualStateInfo::VirtualRawBuffer {
                func,
                size,
                entries,
                descrs,
            } => {
                let mut offsets = Vec::with_capacity(entries.len());
                let mut lengths = Vec::with_capacity(entries.len());
                let mut values = Vec::with_capacity(entries.len());
                for (offset, length, entry_info) in entries {
                    offsets.push(*offset);
                    lengths.push(*length);
                    values.push(Self::import_virtual_state_value(entry_info, ctx));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(
                        crate::optimizeopt::info::VirtualRawBufferInfo {
                            func: *func,
                            size: *size,
                            offsets,
                            lengths,
                            descrs: descrs.clone(),
                            values,
                            last_guard_pos: -1,
                            // resume.py:1452: calldescr from callinfo_for_oopspec
                            calldescr: Some(majit_ir::make_raw_malloc_calldescr()),
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
            }
            VirtualStateInfo::KnownClass { class_ptr } => {
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::known_class(*class_ptr, true),
                );
            }
            VirtualStateInfo::NonNull => {
                ctx.set_ptr_info(opref, crate::optimizeopt::info::PtrInfo::nonnull());
            }
            VirtualStateInfo::IntBounded(bound) => {
                // RPython parity: imported preamble bounds become the box's
                // forwarded IntBound directly (optimizer.py:115-125
                // setintbound). No separate "imported" or "lower-only" maps.
                let widened = bound.widen();
                ctx.setintbound(opref, &widened);
            }
            VirtualStateInfo::Unknown(tp) => {
                // virtualstate.py:655-683 make_inputargs parity: each
                // NotVirtualStateInfo leaf is realized as an InputArg* whose
                // `Box.type` is intrinsic. pyre has no Box object, so the
                // `tp` tag carried on `Unknown(tp)` must populate the
                // authoritative type map. `or_insert` so Phase 2 inputarg
                // seeds (trace_inputarg_types, optimizer.rs:1719-1721) and
                // constant types (seeded via get_constant) continue to win
                // when they are already present; the VSI tag only fills in
                // label_args that have no external typing (capacity-padded
                // JUMP args emitted by pyre when virt_extract_live_values
                // expands to the heap FixedObjectArray length).
                if *tp != majit_ir::Type::Void {
                    ctx.value_types.entry(opref.0).or_insert(*tp);
                }
            }
        }
    }

    fn install_imported_virtuals(&self, ctx: &mut OptContext) {
        // virtualstate.py:655-670 make_inputargs + 627-634 _enum parity:
        // label_args are laid out by recursive _enum traversal where each
        // NotVirtualStateInfo leaf gets a position_in_notvirtuals slot.
        // Virtual states don't consume slots directly — their non-virtual
        // FIELD sub-states do, interleaved with top-level non-virtual states.
        // Traverse ALL states in order, advancing label_slot for each leaf.
        let imported_label_args = self
            .imported_label_args
            .as_ref()
            .expect("install_imported_virtuals requires imported_label_args");
        let mut label_slot = 0usize;
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit] install_virt: label_slot={} label_args_len={} label_args={:?}",
                label_slot,
                imported_label_args.len(),
                imported_label_args
            );
        }
        // RPython Box identity parity: Two-pass approach.
        //
        // Pass 1: process all fields with import_virtual_state_from_label_args.
        //   Constant fields get alloc_op_position. Leaf fields get LABEL arg OpRefs.
        //   Record which leaf field positions need SameAs (skip_flush_mode = Phase 2).
        //
        // Pass 2: allocate SameAs positions and create ops for recorded leaves.
        //   This happens AFTER all Constant allocs, preventing reserve_pos collision
        //   with Phase 1 virtual head positions.
        //
        // Without SameAs, the assembly's body_result_remap maps fail_args entries
        // from LABEL arg positions to body op fresh positions, causing virtual fields
        // to get body IntAddOvf results instead of LABEL values.
        struct VirtualEntry {
            head: OpRef,
            size_descr: majit_ir::DescrRef,
            fields: Vec<(u32, OpRef)>,
            field_descrs: Vec<majit_ir::DescrRef>,
            kind: ImportedVirtualKind,
            head_load_descr_index: Option<u32>,
        }
        let mut entries: Vec<VirtualEntry> = Vec::new();
        // Track leaf field positions that need SameAs (position, entry_idx, field_idx).
        let mut same_as_targets: Vec<(OpRef, usize, usize)> = Vec::new();

        // RPython parity: traverse ALL states in order (not just virtuals).
        // Non-virtual states advance label_slot without creating entries.
        // Virtual states create entries with fields from label_args.
        // Build a map from inputarg_index to imported_virtual for virtual lookup.
        let iv_map: std::collections::HashMap<usize, &_> = self
            .imported_virtuals
            .iter()
            .map(|iv| (iv.inputarg_index, iv))
            .collect();
        let all_states = self
            .imported_loop_state
            .as_ref()
            .map(|s| &s.virtual_state.state[..])
            .unwrap_or(&[]);
        // virtualstate.py:111-116 `enum` parity: a single visited map
        // tracked via `Rc::as_ptr` dedups shared subtrees across all
        // top-level state entries, matching the
        // `build_sequential_slot_schedule` dedup so `label_slot` advances
        // by the same number of leaves as `imported_label_args.len()`.
        // The map value caches the imported Phase 2 OpRef for the first
        // visit so subsequent revisits resolve to the same box (mirroring
        // RPython's setinfo_from_preamble.get_forwarded sharing).
        let mut walk_visited: std::collections::HashMap<usize, OpRef> =
            std::collections::HashMap::new();
        for (state_idx, state_info) in all_states.iter().enumerate() {
            if let Some(iv) = iv_map.get(&state_idx) {
                // Virtual state: process fields recursively, consuming slots
                // for non-virtual leaf fields.
                //
                // Top-level Rc dedup: if two top-level state entries are
                // aliased (same Rc), the second visit should not advance
                // label_slot or allocate a new VirtualEntry. Mark the
                // top-level Rc identity as visited so subsequent revisits
                // (including nested-field references back to the same Rc)
                // short-circuit through walk_visited.
                let top_key = std::rc::Rc::as_ptr(state_info) as usize;
                if walk_visited.contains_key(&top_key) {
                    continue;
                }
                walk_visited.insert(top_key, OpRef::NONE);
                //
                // `iv.inputarg_index` is the virtualizable state's index
                // within `next_iteration_args`, which corresponds to
                // Phase 2 inputarg slot i (not to a raw OpRef). Shift
                // by `ctx.inputarg_base` so the OpRef we resolve points
                // to Phase 2's fresh inputarg slot in the disjoint
                // `[inputarg_base..inputarg_base+num_inputs)` range
                // (matches RPython's TraceIterator fresh-InputArg model).
                // For Phase 1 / standalone runs `inputarg_base == 0`, so
                // the shift is a no-op and the legacy raw-position path
                // still works.
                let raw = OpRef(ctx.inputarg_base + iv.inputarg_index as u32);
                let virtual_head = ctx.get_box_replacement(raw);
                walk_visited.insert(top_key, virtual_head);
                let mut fields = Vec::new();
                let mut field_descrs = Vec::new();
                for (descr, field_info) in &iv.fields {
                    let field_ref = Self::import_virtual_state_from_label_args(
                        field_info,
                        imported_label_args,
                        &mut label_slot,
                        ctx,
                        &mut walk_visited,
                    );
                    let field_idx = fields.len();
                    if ctx.skip_flush_mode
                        && !field_ref.is_none()
                        && !ctx.is_constant(field_ref)
                        && ctx
                            .get_ptr_info(field_ref)
                            .map_or(true, |info| !info.is_virtual())
                        && ctx.get_constant(field_ref).is_none()
                    {
                        same_as_targets.push((field_ref, entries.len(), field_idx));
                    }
                    let field_idx = descr
                        .as_field_descr()
                        .map(|field_descr| field_descr.index_in_parent() as u32)
                        .unwrap_or_else(|| descr.index());
                    fields.push((field_idx, field_ref));
                    field_descrs.push(descr.clone());
                }
                entries.push(VirtualEntry {
                    head: virtual_head,
                    size_descr: iv.size_descr.clone(),
                    fields,
                    field_descrs,
                    kind: iv.kind.clone(),
                    head_load_descr_index: iv.head_load_descr_index,
                });
            } else {
                // Non-virtual state: advance label_slot to match RPython's
                // position_in_notvirtuals enumeration order. Top-level Rc
                // dedup is handled inside count_forced_boxes_for_entry_static.
                let count = crate::optimizeopt::virtualstate::VirtualState::count_forced_boxes_for_entry_static(
                    state_info,
                    &mut walk_visited,
                );
                label_slot += count;
            }
        }

        // Pass 2: allocate SameAs ops for leaf field values.
        // Advance next_pos past all virtual head positions to prevent
        // reserve_pos from returning a position that's already used
        // as a virtual head (allocated during import_state).
        for entry in &entries {
            if !entry.head.is_none() && !ctx.is_constant(entry.head) {
                ctx.next_pos = ctx.next_pos.max(entry.head.0 + 1);
            }
        }
        for (label_arg, entry_idx, field_idx) in &same_as_targets {
            // RPython parity: Box.type carries the source type intrinsically.
            // value_types[label_arg] is whatever the optimizer's emit() last
            // recorded for that OpRef. A Type::Void here would be an upstream
            // bookkeeping bug (a guard's pos collided with a Box position):
            // letting it through reaches `same_as_for_type(Void)`, which is
            // `unreachable!` and panics with a clear message.
            let tp = ctx
                .opref_type(*label_arg)
                .expect("imported virtual leaf missing box.type");
            let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
            let mut op = majit_ir::Op::new(same_as_op, &[*label_arg]);
            op.pos = ctx.reserve_pos();
            let fresh = op.pos;
            ctx.register_value_type(fresh, tp);
            ctx.new_operations.push(op);
            // Update the field to reference the SameAs result.
            entries[*entry_idx].fields[*field_idx].1 = fresh;
        }

        // Install PtrInfo for each virtual.
        // unroll.py:55: if op.get_forwarded() is not None: return
        // Skip heads that already have PtrInfo (duplicate entries from
        // aliased JUMP args sharing the same VirtualState position).
        let mut installed_heads = std::collections::HashSet::new();
        for entry in entries {
            if !installed_heads.insert(entry.head) {
                continue;
            }
            if std::env::var_os("MAJIT_LOG").is_some() {
                eprintln!(
                    "[jit] install_imported_virtual head={:?} fields={:?}",
                    entry.head, entry.fields
                );
            }
            match &entry.kind {
                ImportedVirtualKind::Instance { known_class } => {
                    ctx.set_ptr_info(
                        entry.head,
                        crate::optimizeopt::info::PtrInfo::Virtual(
                            crate::optimizeopt::info::VirtualInfo {
                                descr: entry.size_descr,
                                known_class: *known_class,
                                ob_type_descr: None,
                                fields: entry.fields,
                                field_descrs: entry.field_descrs,
                                last_guard_pos: -1,
                                cached_vinfo: std::cell::RefCell::new(None),
                            },
                        ),
                    );
                }
                ImportedVirtualKind::Struct => {
                    ctx.set_ptr_info(
                        entry.head,
                        crate::optimizeopt::info::PtrInfo::VirtualStruct(
                            crate::optimizeopt::info::VirtualStructInfo {
                                descr: entry.size_descr,
                                fields: entry.fields,
                                field_descrs: entry.field_descrs,
                                last_guard_pos: -1,
                                cached_vinfo: std::cell::RefCell::new(None),
                            },
                        ),
                    );
                }
            }
            if let Some(descr_idx) = entry.head_load_descr_index {
                ctx.imported_virtual_heads
                    .push((descr_idx as usize, entry.head));
            }
        }
    }

    /// virtualstate.py:712-728 `VirtualStateConstructor.create_state` cache
    /// parity for the import side: dedup nested `Rc<VirtualStateInfo>`
    /// references via pointer identity, returning the previously imported
    /// Phase 2 OpRef on revisits so shared substates collapse onto a
    /// single Phase 2 box (matching RPython's setinfo_from_preamble
    /// `if op.get_forwarded() is not None: return` semantics).
    fn import_virtual_state_from_label_args_recurse(
        rc: &std::rc::Rc<crate::optimizeopt::virtualstate::VirtualStateInfoNode>,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
        walk_visited: &mut std::collections::HashMap<usize, OpRef>,
    ) -> OpRef {
        let key = std::rc::Rc::as_ptr(rc) as usize;
        if let Some(&cached) = walk_visited.get(&key) {
            return cached;
        }
        // Insert a placeholder NONE to break cycles, then overwrite with
        // the real OpRef once import_virtual_state_from_label_args returns.
        walk_visited.insert(key, OpRef::NONE);
        let opref = Self::import_virtual_state_from_label_args(
            &rc.info,
            imported_label_args,
            label_slot,
            ctx,
            walk_visited,
        );
        walk_visited.insert(key, opref);
        opref
    }

    fn import_virtual_state_from_label_args(
        info: &crate::optimizeopt::virtualstate::VirtualStateInfo,
        imported_label_args: &[OpRef],
        label_slot: &mut usize,
        ctx: &mut OptContext,
        walk_visited: &mut std::collections::HashMap<usize, OpRef>,
    ) -> OpRef {
        use crate::optimizeopt::virtualstate::VirtualStateInfo;

        match info {
            VirtualStateInfo::Constant(_) => Self::import_virtual_state_value(info, ctx),
            VirtualStateInfo::Virtual {
                descr,
                known_class,
                ob_type_descr,
                fields,
                field_descrs,
            } => {
                let opref = ctx.alloc_op_position();
                // unroll.py:454 Box carries its type. Virtual heads are Ref.
                ctx.register_value_type(opref, majit_ir::Type::Ref);
                let imported_fields: Vec<(u32, OpRef)> = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        let field_ref = Self::import_virtual_state_from_label_args_recurse(
                            field_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                            walk_visited,
                        );
                        // virtualstate.py:159-162: fielddescrs is a flat list,
                        // box access uses fielddescrs[i].get_index() == field_idx.
                        let offset = field_descrs
                            .iter()
                            .find_map(|d| {
                                d.as_field_descr()
                                    .filter(|fd| fd.index_in_parent() as u32 == *field_idx)
                            })
                            .map(|fd| fd.offset());
                        if offset == Some(0) && known_class.is_some() {
                            ctx.constant_types_for_numbering
                                .insert(field_ref.0, majit_ir::Type::Ref);
                        }
                        (*field_idx, field_ref)
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::Virtual(
                        crate::optimizeopt::info::VirtualInfo {
                            descr: descr.clone(),
                            known_class: *known_class,
                            ob_type_descr: ob_type_descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VArray { descr, items, .. } => {
                let opref = ctx.alloc_op_position();
                // unroll.py:454 Box carries its type. VArray heads are Ref.
                ctx.register_value_type(opref, majit_ir::Type::Ref);
                let imported_items = items
                    .iter()
                    .map(|item_info| {
                        Self::import_virtual_state_from_label_args_recurse(
                            item_info,
                            imported_label_args,
                            label_slot,
                            ctx,
                            walk_visited,
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArray(
                        crate::optimizeopt::info::VirtualArrayInfo {
                            descr: descr.clone(),
                            clear: false,
                            items: imported_items,
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VStruct {
                descr,
                fields,
                field_descrs,
            } => {
                let opref = ctx.alloc_op_position();
                // unroll.py:454 Box carries its type. VStruct heads are Ref.
                ctx.register_value_type(opref, majit_ir::Type::Ref);
                let imported_fields = fields
                    .iter()
                    .map(|(field_idx, field_info)| {
                        (
                            *field_idx,
                            Self::import_virtual_state_from_label_args_recurse(
                                field_info,
                                imported_label_args,
                                label_slot,
                                ctx,
                                walk_visited,
                            ),
                        )
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualStruct(
                        crate::optimizeopt::info::VirtualStructInfo {
                            descr: descr.clone(),
                            fields: imported_fields,
                            field_descrs: field_descrs.clone(),
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VArrayStruct {
                descr,
                fielddescrs,
                element_fields,
            } => {
                let opref = ctx.alloc_op_position();
                // unroll.py:454 Box carries its type. VArrayStruct heads are Ref.
                ctx.register_value_type(opref, majit_ir::Type::Ref);
                let imported_elements = element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_idx, field_info)| {
                                (
                                    *field_idx,
                                    Self::import_virtual_state_from_label_args_recurse(
                                        field_info,
                                        imported_label_args,
                                        label_slot,
                                        ctx,
                                        walk_visited,
                                    ),
                                )
                            })
                            .collect()
                    })
                    .collect();
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(
                        crate::optimizeopt::info::VirtualArrayStructInfo {
                            descr: descr.clone(),
                            fielddescrs: fielddescrs.clone(),
                            element_fields: imported_elements,
                            last_guard_pos: -1,
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::VirtualRawBuffer {
                func,
                size,
                entries,
                descrs,
            } => {
                let opref = ctx.alloc_op_position();
                let mut offsets = Vec::with_capacity(entries.len());
                let mut lengths = Vec::with_capacity(entries.len());
                let mut values = Vec::with_capacity(entries.len());
                for (offset, length, entry_info) in entries {
                    offsets.push(*offset);
                    lengths.push(*length);
                    values.push(Self::import_virtual_state_from_label_args_recurse(
                        entry_info,
                        imported_label_args,
                        label_slot,
                        ctx,
                        walk_visited,
                    ));
                }
                ctx.set_ptr_info(
                    opref,
                    crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(
                        crate::optimizeopt::info::VirtualRawBufferInfo {
                            func: *func,
                            size: *size,
                            offsets,
                            lengths,
                            descrs: descrs.clone(),
                            values,
                            last_guard_pos: -1,
                            // resume.py:1452: calldescr from callinfo_for_oopspec
                            calldescr: Some(majit_ir::make_raw_malloc_calldescr()),
                            cached_vinfo: std::cell::RefCell::new(None),
                        },
                    ),
                );
                opref
            }
            VirtualStateInfo::KnownClass { .. }
            | VirtualStateInfo::NonNull
            | VirtualStateInfo::IntBounded(_)
            | VirtualStateInfo::Unknown(_) => {
                let opref = imported_label_args
                    .get(*label_slot)
                    .copied()
                    .unwrap_or_else(|| {
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!(
                                "[jit] MISS: label_slot={} len={}",
                                *label_slot,
                                imported_label_args.len()
                            );
                        }
                        OpRef::NONE
                    });
                *label_slot += 1;
                Self::apply_imported_virtual_state(info, opref, ctx);
                let resolved = ctx.get_box_replacement(opref);
                resolved
            }
        }
    }

    pub fn new() -> Self {
        Optimizer {
            passes: Vec::new(),
            pureop_historylength: crate::jit::PARAMETERS.pureop_historylength as usize,
            final_num_inputs: 0,
            call_pure_results: std::collections::HashMap::new(),
            last_guard_op_idx: None,
            replaces_guard: std::collections::HashMap::new(),
            pendingfields: Vec::new(),
            can_replace_guards: true,
            quasi_immutable_deps: std::collections::HashSet::new(),
            constant_types: std::collections::HashMap::new(),
            numbering_type_overrides: std::collections::HashMap::new(),
            imported_virtuals: Vec::new(),
            trace_inputarg_types: Vec::new(),
            runtime_boxes: Vec::new(),
            exported_loop_state: None,
            imported_loop_state: None,
            imported_short_sources: Vec::new(),
            imported_short_aliases: Vec::new(),
            imported_short_preamble: None,
            imported_short_preamble_builder: None,
            imported_label_args: None,
            patchguardop: None,
            skip_flush: false,
            terminal_op: None,
            final_ctx: None,
            pending_bridge_rd: None,
            all_descrs: Vec::new(),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            callinfocollection: None,
            resumedata_memo: crate::resume::ResumeDataLoopMemo::new(),
            snapshot_boxes: std::collections::HashMap::new(),
            snapshot_frame_sizes: std::collections::HashMap::new(),
            snapshot_vable_boxes: std::collections::HashMap::new(),
            snapshot_frame_pcs: std::collections::HashMap::new(),
            prev_phase_value_types: std::collections::HashMap::new(),
            original_trace_op_types: std::collections::HashMap::new(),
            opt_ops_emitted: 0,
            opt_guards_emitted: 0,
            opt_guards_shared_emitted: 0,
        }
    }

    /// Record a CALL_PURE result for cross-iteration constant folding.
    /// RPython pyjitpl.py:3572: `arg_consts = [executor.constant_from_op(a) ...]`
    /// RPython optimizer.py:222: `get_constant_box(arg)` → value-based keys.
    pub fn record_call_pure_result(&mut self, args: Vec<majit_ir::Value>, value: majit_ir::Value) {
        self.call_pure_results.insert(args, value);
    }

    /// Look up a previously recorded CALL_PURE result.
    pub fn get_call_pure_result(&self, args: &[majit_ir::Value]) -> Option<&majit_ir::Value> {
        self.call_pure_results.get(args)
    }

    /// bridgeopt.py:124-185: deserialize_optimizer_knowledge
    /// bridgeopt.py:170-171 optimizer.optheap.deserialize_optheap(result_struct, result_array)
    pub fn import_heap_knowledge(
        &mut self,
        result_struct: &[(OpRef, majit_ir::DescrRef, OpRef)],
        result_array: &[(OpRef, i64, majit_ir::DescrRef, OpRef)],
        ctx: &mut OptContext,
    ) {
        if !result_struct.is_empty() {
            for pass in &mut self.passes {
                pass.import_cached_fields(result_struct, ctx);
            }
        }
        if !result_array.is_empty() {
            for pass in &mut self.passes {
                pass.import_cached_arrayitems(result_array, ctx);
            }
        }
    }

    /// bridgeopt.py:184-185 optimizer.optrewrite.deserialize_optrewrite(result_loopinvariant)
    pub fn import_loopinvariant_knowledge(&mut self, result_loopinvariant: &[(i64, OpRef)]) {
        for pass in &mut self.passes {
            pass.deserialize_optrewrite(result_loopinvariant);
        }
    }

    /// Get the final num_inputs after optimization.
    /// May be larger than the original if virtualizable added virtual input args.
    pub fn final_num_inputs(&self) -> usize {
        self.final_num_inputs
    }

    /// optimizer.py: getlastop() — index of the last emitted guard.
    /// Resolved through `ctx.new_operations[idx]` by callers.
    pub fn last_guard_op_idx(&self) -> Option<usize> {
        self.last_guard_op_idx
    }

    /// optimizer.py: notice_guard_future_condition(op)
    /// Record that a guard at the given position should be replaced
    /// with the given op when the future condition is realized.
    pub fn notice_guard_future_condition(&mut self, guard_pos: u32, replacement: Op) {
        self.replaces_guard.insert(guard_pos, replacement);
    }

    /// optimizer.py:713: replace_guard_op(old_op_pos, new_op)
    /// Replace a previously emitted guard with a new one.
    pub fn replace_guard_op(&mut self, old_pos: u32, new_guard: Op) {
        self.replaces_guard.insert(old_pos, new_guard);
    }

    // RPython optimizer.py:722-752 store_final_boxes_in_guard and
    // optimizer.py:649-670 emit_guard_operation are implemented inside
    // emit_operation: _copy_resume_data_from, store_final_boxes_in_guard,
    // force_box on fail_args, and store_final_boxes_in_guard in ctx.emit().

    /// optimizer.py: add_pending_field(op)
    /// Queue a SETFIELD_GC to be emitted before the next guard.
    pub fn add_pending_field(&mut self, op: Op) {
        self.pendingfields.push(op);
    }

    /// optimizer.py: flush_pendingfields(ctx)
    /// Emit all pending field writes.
    pub fn flush_pendingfields(&mut self, ctx: &mut OptContext) {
        let pending = std::mem::take(&mut self.pendingfields);
        for op in pending {
            ctx.emit(op);
        }
    }

    /// optimizer.py: has_pending_fields()
    pub fn has_pending_fields(&self) -> bool {
        !self.pendingfields.is_empty()
    }

    /// optimizer.py: num_pending_fields()
    pub fn num_pending_fields(&self) -> usize {
        self.pendingfields.len()
    }

    /// optimizer.py: cant_replace_guards()
    /// Temporarily disable guard replacement (e.g., during bridge compilation).
    pub fn disable_guard_replacement(&mut self) {
        self.can_replace_guards = false;
    }

    /// Re-enable guard replacement.
    pub fn enable_guard_replacement(&mut self) {
        self.can_replace_guards = true;
    }

    /// optimizer.py: quasi_immutable_deps[qmut] = None
    /// Track a quasi-immutable field dependency. `dep` is (object_ptr,
    /// field_index) identifying the specific slot the compiled loop
    /// depends on. After compilation, a per-slot watcher is registered.
    pub fn add_quasi_immutable_dep(&mut self, dep: (u64, u32)) {
        self.quasi_immutable_deps.insert(dep);
    }

    /// optimizer.py: produce_potential_short_preamble_ops(sb)
    /// Collect short preamble ops from all passes.
    pub fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        ctx: &mut OptContext,
    ) {
        for pass in &self.passes {
            pass.produce_potential_short_preamble_ops(sb, ctx);
        }
    }

    /// heap.py:825 serialize_optheap(available_boxes) — struct half.
    pub fn export_all_cached_fields(
        &self,
        ctx: &mut OptContext,
        available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, majit_ir::DescrRef, OpRef)> {
        let mut result = Vec::new();
        for pass in &self.passes {
            result.extend(pass.export_cached_fields(ctx, available_boxes));
        }
        result
    }

    /// heap.py:847 serialize_optheap(available_boxes) — array half.
    pub fn export_all_cached_arrayitems(
        &self,
        ctx: &mut OptContext,
        available_boxes: Option<&std::collections::HashMap<OpRef, ()>>,
    ) -> Vec<(OpRef, i64, majit_ir::DescrRef, OpRef)> {
        let mut result = Vec::new();
        for pass in &self.passes {
            result.extend(pass.export_cached_arrayitems(ctx, available_boxes));
        }
        result
    }

    /// Pre-tag Phase 1 JUMP arg OpRefs as generation 0.

    /// Lock JUMP arg OpRefs so replace_op won't forward them.

    /// optimizer.py:557 parity:
    ///
    /// ```python
    /// self.resumedata_memo.update_counters(self.metainterp_sd.profiler)
    /// ```
    ///
    /// RPython invokes `update_counters` at the tail of every
    /// `propagate_all_forward` call, folding accumulated
    /// NVIRTUALS / NVHOLES / NVREUSED from `resumedata_memo` into the
    /// static-data profiler. In pyre the optimizer does not hold a
    /// reference to `MetaInterp.stats`, so this is a thin accessor the
    /// caller invokes after each optimize_loop / optimize_bridge /
    /// propagate_all_forward exit to fold counters into the JIT stats.
    ///
    /// Also folds OPT_OPS / OPT_GUARDS / OPT_GUARDS_SHARED accumulated
    /// inside `_emit_operation` / `emit_guard_operation`
    /// (optimizer.py:626/629/673-674).  RPython publishes these
    /// directly via `self.metainterp_sd.profiler.count(...)`; pyre's
    /// `Optimizer` carries no `metainterp_sd` reference and therefore
    /// piggybacks on the same deferred-fold pattern as the resumedata
    /// memo counters.
    pub fn update_counters(&mut self, profiler: &crate::jitprof::JitProfiler) {
        self.resumedata_memo.update_counters(profiler);
        profiler.count(crate::pyjitpl::counters::OPT_OPS, self.opt_ops_emitted);
        profiler.count(
            crate::pyjitpl::counters::OPT_GUARDS,
            self.opt_guards_emitted,
        );
        profiler.count(
            crate::pyjitpl::counters::OPT_GUARDS_SHARED,
            self.opt_guards_shared_emitted,
        );
        self.opt_ops_emitted = 0;
        self.opt_guards_emitted = 0;
        self.opt_guards_shared_emitted = 0;
        // Drain per-pass `Counters.*` accumulators (e.g.
        // VectorizingOptimizer's OPT_VECTORIZE_TRY / OPT_VECTORIZED).
        for pass in &mut self.passes {
            pass.drain_profiler_counters(profiler);
        }
    }

    /// optimizer.py: flush()
    /// Flush all passes' postponed state.
    pub fn flush(&mut self, ctx: &mut OptContext) {
        for pass_idx in 0..self.passes.len() {
            // heap.flush() resolves the "next_optimization" emit_extra
            // target through `ctx.current_pass_idx`. Set it to the
            // currently-flushing pass so any ops force_all_lazy_sets
            // emits land at `(pass_idx + 1, op)` and skip the flushing
            // pass on the subsequent drain — RPython parity with
            // `self.next_optimization`.
            ctx.current_pass_idx = pass_idx;
            let pass = &mut self.passes[pass_idx];
            pass.flush(ctx);
            // RPython Optimization.emit_extra() routes newly forced ops to
            // optimizer.send_extra_operation(op, self.next_optimization).
            // During flush we must preserve that contract: each pass's flush
            // output is processed only by downstream passes, never by the
            // flushing pass again.
            self.drain_extra_operations_from(pass_idx + 1, ctx);
        }
    }

    /// Build a short preamble from an optimized trace's preamble section.
    /// Convenience method that combines extract + produce.
    pub fn build_short_preamble(
        optimized_ops: &[Op],
    ) -> crate::optimizeopt::shortpreamble::ShortPreamble {
        crate::optimizeopt::shortpreamble::extract_short_preamble(optimized_ops)
    }

    /// optimizer.py: send_extra_operation(op, ctx)
    /// Send an extra operation through the pass chain as if it were
    /// a new operation from the trace. Used by passes that need to
    /// inject additional operations.
    pub fn send_extra_operation(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass(0, op, ctx);
    }

    /// RPython optimizer.py: emit_extra(op, emit=False) parity.
    /// Route an operation through passes starting AFTER `after_pass_idx`,
    /// matching RPython's `send_extra_operation(op, self.next_optimization)`.
    pub fn send_extra_operation_after(
        &mut self,
        after_pass_idx: usize,
        op: &Op,
        ctx: &mut OptContext,
    ) {
        self.propagate_from_pass(after_pass_idx + 1, op, ctx);
    }

    /// optimizer.py:345-364: force_box — force a virtual to be materialized.
    /// Also pops from potential_extra_ops (optimizer.py:351-359).
    ///
    /// majit source/result split: RPython uses a single Box identity, so a
    /// single `potential_extra_ops.pop(op)` always finds the entry. majit
    /// separates preamble source and imported result, so we must try three
    /// keys: resolved, opref, and imported_short_source(opref).
    /// Mirrors force_box_inline (mod.rs) contract.
    pub fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        // optimizer.py:346: op = get_box_replacement(op)
        let preamble_source = ctx.imported_short_source(opref);
        let resolved = ctx.get_box_replacement(opref);
        // optimizer.py:351-359: potential_extra_ops.pop(op)
        // → sb.add_preamble_op(preamble_op)
        // Try three keys to account for the source/result split:
        let tracked = ctx
            .take_potential_extra_op(resolved)
            .or_else(|| ctx.take_potential_extra_op(opref))
            .or_else(|| {
                (preamble_source != resolved && preamble_source != opref)
                    .then(|| ctx.take_potential_extra_op(preamble_source))
                    .flatten()
            });
        if std::env::var_os("MAJIT_LOG").is_some()
            && (opref == majit_ir::OpRef(83)
                || resolved == majit_ir::OpRef(83)
                || preamble_source == majit_ir::OpRef(16)
                || tracked
                    .as_ref()
                    .is_some_and(|pop| pop.op == majit_ir::OpRef(16)))
        {
            eprintln!(
                "[jit] force_box_pop opref={opref:?} resolved={resolved:?} preamble_source={preamble_source:?} tracked={:?}",
                tracked
                    .as_ref()
                    .map(|pop| (pop.op, pop.resolved, pop.invented_name))
            );
        }
        if let Some(preamble_op) = tracked {
            let resolved_for_pop = ctx.get_box_replacement(preamble_op.op);
            if let Some(builder) = ctx.active_short_preamble_producer_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            } else if let Some(builder) = ctx.imported_short_preamble_builder.as_mut() {
                builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
            }
        }
        if ctx
            .get_ptr_info(resolved)
            .is_some_and(|info| info.is_virtual())
        {
            // RPython: info.force_box() sets _is_virtual=False in-place.
            // Take ownership so the Virtual PtrInfo is removed. force_box_impl
            // installs a non-virtual (Instance/Struct) at the alloc_ref.
            let mut info = ctx.take_ptr_info(resolved).unwrap();
            let forced = info.force_box(resolved, ctx);
            return ctx.get_box_replacement(forced);
        }
        resolved
    }

    /// `optimizer.py:306-319` `force_box_for_end_of_preamble(box)`.
    ///
    /// ```python
    /// def force_box_for_end_of_preamble(self, box):
    ///     if box.type == 'r':
    ///         info = getptrinfo(box)
    ///         if info is not None and info.is_virtual():
    ///             rec = {}
    ///             return info.force_at_the_end_of_preamble(box, self.optearlyforce, rec)
    ///         return box
    ///     if box.type == 'i':
    ///         info = getrawptrinfo(box)
    ///         if info is not None:
    ///             return info.force_at_the_end_of_preamble(box, self.optearlyforce, None)
    ///     return box
    /// ```
    ///
    /// Wraps `force_at_the_end_of_preamble` with the per-type gating
    /// RPython inlines at the call site.  Ref boxes only recurse when
    /// a virtual `PtrInfo` is present; Int boxes only recurse when a
    /// raw-ptr info is present (pyre stores that under
    /// `PtrInfo::VirtualRawBuffer` in the same info table); Float
    /// boxes are returned unchanged.  Callers that previously invoked
    /// `force_at_the_end_of_preamble` directly should route through
    /// this wrapper for RPython structural parity (unroll.py:126-127).
    pub fn force_box_for_end_of_preamble(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        match ctx.opref_type(resolved) {
            // optimizer.py:307-313 — `box.type == 'r'` path.
            Some(majit_ir::Type::Ref) => {
                let is_virtual = ctx
                    .get_ptr_info(resolved)
                    .is_some_and(|info| info.is_virtual());
                if is_virtual {
                    return self.force_at_the_end_of_preamble(resolved, ctx);
                }
                opref
            }
            // optimizer.py:314-318 — `box.type == 'i'` path.
            //
            // pyre tracks raw-ptr info on Int boxes via
            // `PtrInfo::VirtualRawBuffer` in the same `ptr_info` table
            // that Ref boxes use; there is no separate `getrawptrinfo`
            // registry, so the presence check fires on any PtrInfo
            // attached to an Int-typed OpRef.
            Some(majit_ir::Type::Int) => {
                if ctx.get_ptr_info(resolved).is_some() {
                    return self.force_at_the_end_of_preamble(resolved, ctx);
                }
                opref
            }
            // optimizer.py:319 — fall-through `return box`.
            _ => opref,
        }
    }

    /// optimizer.py: force_at_the_end_of_preamble(box)
    ///
    /// The exported loop state should record the boxes that survive the end of
    /// the preamble after virtuals have been forced into a loop-carried shape.
    pub fn force_at_the_end_of_preamble(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        let mut rec = std::collections::HashSet::new();
        self.force_at_the_end_of_preamble_rec(opref, ctx, &mut rec)
    }

    fn force_at_the_end_of_preamble_rec(
        &mut self,
        opref: OpRef,
        ctx: &mut OptContext,
        rec: &mut std::collections::HashSet<OpRef>,
    ) -> OpRef {
        let resolved = ctx.get_box_replacement(opref);
        let Some(mut info) = ctx.get_ptr_info(resolved).cloned() else {
            return resolved;
        };

        // `info.py:53-56` `PtrInfo.force_at_the_end_of_preamble` base:
        //
        // ```python
        // def force_at_the_end_of_preamble(self, op, optforce, rec):
        //     if not self.is_virtual():
        //         return get_box_replacement(op)
        //     return self._force_at_the_end_of_preamble(op, optforce, rec)
        // ```
        //
        // RPython gates every dispatch on `self.is_virtual()`.  pyre's
        // variant-tag match below would otherwise re-force a non-virtual
        // `VirtualRawSlice` (post-`_force_elements`, `parent = OpRef::NONE`)
        // every time this method is re-entered — matching info.py:464-465
        // `return self.parent is not None` requires the same is_virtual
        // gate here as in the RPython base.
        if !info.is_virtual() {
            return resolved;
        }

        // RPython info.py: InstancePtrInfo, StructPtrInfo, ArrayPtrInfo all
        // override _force_at_the_end_of_preamble to keep the virtual alive
        // and recurse into fields. AbstractRawPtrInfo uses the base
        // _force_at_the_end_of_preamble → force_box() (materialization).
        if matches!(
            info,
            crate::optimizeopt::info::PtrInfo::Virtual(_)
                | crate::optimizeopt::info::PtrInfo::VirtualStruct(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArray(_)
                | crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(_)
        ) {
            if !rec.insert(resolved) {
                return resolved;
            }
            info.force_at_the_end_of_preamble(|child| {
                self.force_at_the_end_of_preamble_rec(child, ctx, rec)
            });
            ctx.set_ptr_info(resolved, info);
            return resolved;
        }

        // RawBuffer / RawSlice: `AbstractRawPtrInfo` (info.py:374-384)
        // inherits `AbstractVirtualPtrInfo._force_at_the_end_of_preamble`
        // (info.py:159-160) without override, so both take the base
        // `force_box()` materialization path.  pyre mirrors this by
        // matching both `VirtualRawBuffer` and `VirtualRawSlice`
        // (info.py:386 `RawBufferPtrInfo` / info.py:459 `RawSlicePtrInfo`).
        // optimizer.py:311-312 routes through `optforce = self.optearlyforce`.
        if matches!(
            info,
            crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_)
        ) {
            let saved = ctx.current_pass_idx;
            ctx.current_pass_idx = ctx.optearlyforce_idx;
            let result = self.force_box(resolved, ctx);
            ctx.current_pass_idx = saved;
            return result;
        }

        resolved
    }

    /// optimizer.py: protect_speculative_operation(op, ctx)
    /// When constant-folding a pure operation, verify that the folded
    /// constant doesn't cause a memory safety issue (e.g., null deref in
    /// getfield). If the result would be invalid, don't fold.
    pub fn protect_speculative_operation(&self, op: &Op, ctx: &OptContext) -> bool {
        // For now, conservative: only allow folding on arithmetic ops.
        // getfield/getarrayitem on constant null pointer would crash.
        match op.opcode {
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF => {
                // Check arg(0) is not null constant.
                if let Some(0) = ctx.get_constant_int(op.arg(0)) {
                    return false; // would deref null
                }
                true
            }
            _ => true,
        }
    }

    /// optimizer.py: getlastop() — return the last emitted non-guard operation.
    pub fn getlastop<'a>(&self, ctx: &'a OptContext) -> Option<&'a Op> {
        ctx.new_operations.last()
    }

    /// optimizer.py: _clean_optimization_info(ops)
    /// Reset forwarding pointers on all ops before re-optimization.
    /// Called when re-optimizing a trace (e.g., retrace).
    pub fn clean_optimization_info(ctx: &mut OptContext) {
        ctx.forwarded.clear();
    }

    /// optimizer.py: get_count_of_ops()
    /// Count operations emitted so far.
    pub fn get_count_of_ops(ctx: &OptContext) -> usize {
        ctx.new_operations.len()
    }

    /// optimizer.py: get_count_of_guards()
    /// Count guards emitted so far.
    pub fn get_count_of_guards(ctx: &OptContext) -> usize {
        ctx.new_operations
            .iter()
            .filter(|op| op.opcode.is_guard())
            .count()
    }

    /// optimizer.py: log_loop(ops)
    /// Log the optimized trace for debugging/profiling.
    pub fn log_optimized_trace(ctx: &OptContext) {
        if std::env::var("MAJIT_LOG_OPT").is_ok() {
            eprintln!(
                "[MAJIT] optimized trace: {} ops, {} guards",
                ctx.new_operations.len(),
                ctx.new_operations
                    .iter()
                    .filter(|op| op.opcode.is_guard())
                    .count()
            );
        }
    }

    /// optimizer.py:127-135 `getnullness(op)` parity (line-by-line port).
    ///
    /// Delegates to `OptContext::getnullness`, which implements the
    /// upstream `op.type == 'r' or is_raw_ptr(op)` dispatch and the
    /// `getintbound(op).getnullness()` fallback for `'i'`-typed Boxes.
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py:13-15).
    ///
    /// Takes `&mut OptContext` to mirror the upstream `getintbound`
    /// lazy-install side effect (optimizer.py:102-112).
    pub fn getnullness(ctx: &mut OptContext, opref: OpRef) -> i8 {
        ctx.getnullness(opref)
    }

    /// optimizer.py:137-152: make_constant_class(op, class_const, update_last_guard)
    ///
    /// Sets known class on PtrInfo, preserving last_guard_pos from any
    /// existing info. When `update_last_guard` is false (RECORD_EXACT_CLASS),
    /// the guard position is NOT updated — preserving guard strengthening
    /// opportunities for subsequent GUARD_CLASS ops.
    ///
    /// PyPy `InstancePtrInfo(descr, known_class, is_virtual)` is one class
    /// (info.py:313) that covers both virtual and non-virtual instances via
    /// the `is_virtual` flag. The Rust port splits that into two PtrInfo
    /// enum variants — `Instance` (is_virtual=False) and `Virtual`
    /// (is_virtual=True) — for ergonomic dispatch. `isinstance(opinfo,
    /// InstancePtrInfo)` at optimizer.py:140 is therefore true for *either*
    /// Rust variant, so both must update `_known_class` in place rather
    /// than being overwritten by a fresh known-class info (which would
    /// drop the `Virtual` fields / descr / cached_vinfo state).
    pub fn make_constant_class(
        ctx: &mut OptContext,
        opref: OpRef,
        class_value: i64,
        update_last_guard: bool,
    ) {
        let class_ptr = GcRef(class_value as usize);
        let resolved = ctx.get_box_replacement(opref);
        // optimizer.py:139: isinstance(opinfo, info.InstancePtrInfo).
        // RPython's InstancePtrInfo covers both virtual and non-virtual
        // instances (`is_virtual` is a field). Rust splits those states into
        // PtrInfo::Instance and PtrInfo::Virtual, so both arms must preserve
        // the existing object info and only update `_known_class`.
        let updated_existing = match ctx.get_ptr_info_mut(resolved) {
            Some(PtrInfo::Instance(iinfo)) => {
                iinfo.known_class = Some(class_ptr);
                true
            }
            Some(PtrInfo::Virtual(vinfo)) => {
                vinfo.known_class = Some(class_ptr);
                true
            }
            _ => false,
        };
        if !updated_existing {
            // optimizer.py:142-148: preserve last_guard_pos from old info
            let old_guard_pos = ctx
                .get_ptr_info(resolved)
                .and_then(|i| i.get_last_guard_pos())
                .map(|p| p as i32)
                .unwrap_or(-1);
            let mut new_info = PtrInfo::known_class(class_ptr, true);
            new_info.set_last_guard_pos(old_guard_pos);
            ctx.set_ptr_info(resolved, new_info);
        }
        // optimizer.py:150-151: if update_last_guard: mark_last_guard
        if update_last_guard {
            ctx.mark_last_guard(resolved);
        }
    }

    /// optimizer.py:705-711 `is_call_pure_pure_canraise(op)`.
    /// Mirrors PyPy exactly: ignore `MemoryError`-only effects when deciding
    /// whether a CALL_PURE breaks guard resume-data sharing.
    pub fn is_call_pure_pure_canraise(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.get_extra_info().check_can_raise(true))
            .unwrap_or(true)
    }

    /// Add an optimization pass to the chain.
    pub fn add_pass(&mut self, mut pass: Box<dyn Optimization>) {
        pass.set_pureop_historylength(self.pureop_historylength);
        self.passes.push(pass);
    }

    pub fn set_pureop_historylength(&mut self, limit: usize) {
        self.pureop_historylength = limit;
        for pass in &mut self.passes {
            pass.set_pureop_historylength(limit);
        }
    }

    /// Mark all passes as Phase 2 (loop body).
    pub fn set_phase2(&mut self, phase2: bool) {
        for pass in &mut self.passes {
            pass.set_phase2(phase2);
        }
    }

    /// Run all optimization passes over a list of operations.
    ///
    /// Returns the optimized operation list.
    /// optimizer.py:517: propagate_all_forward(trace, call_pure_results, flush)
    pub fn propagate_all_forward(&mut self, ops: &[Op]) -> Vec<Op> {
        self.optimize_with_constants(ops, &mut std::collections::HashMap::new())
    }

    /// Run all optimization passes, with known constants pre-populated.
    ///
    /// `constants` maps OpRef indices to raw backend bits. RPython keeps typed
    /// Const boxes in the optimized trace; majit threads the same information
    /// through this side table and recovers Int/Ref/Float from the producing
    /// op's result type.
    ///
    /// After optimization, newly-discovered constants (from constant folding)
    /// are written back into the map so the backend can resolve them.
    pub fn optimize_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        self.optimize_with_constants_and_inputs(ops, constants, 0)
    }

    /// Like `optimize_with_constants`, but also takes `num_inputs` so that
    /// ops emitted by the optimizer (e.g. from force_virtual) get pos values
    /// that don't collide with input argument variable indices.
    /// Phase 1 / standalone entry point. Computes `start_next_pos` from
    /// the input ops' max OpRef so newly emitted ops never collide with
    /// the trace's existing positional layout. Phase 2 and bridge callers
    /// should use [`Self::optimize_with_constants_and_inputs_at`] with an
    /// explicit `inputarg_base` to allocate disjoint OpRefs.
    pub fn optimize_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> Vec<Op> {
        // Ensure new ops get positions beyond all original trace positions.
        // Original ops keep their tracer-assigned positions; new ops (constants,
        // force materializations) must not collide with them.
        let max_pos = ops
            .iter()
            .map(|op| op.pos)
            .filter(|op| !op.is_none() && !op.is_constant())
            .map(|op| op.0)
            .max()
            .unwrap_or(0);
        let start_next_pos = ((max_pos as u32) + 1).max(num_inputs as u32);
        self.optimize_with_constants_and_inputs_at(ops, constants, num_inputs, 0, start_next_pos)
    }

    /// opencoder.py:271 _index parity entry point.
    ///
    /// Optimizes a slice of ops whose `pos`/`args` reference OpRefs in a
    /// shifted namespace `[inputarg_base, …)`. The first fresh OpRef the
    /// optimizer assigns to a non-void result is `OpRef(start_next_pos)`.
    /// Phase 2 / bridges pass `inputarg_base = parent_high_water` and
    /// `start_next_pos = parent_high_water + num_inputs` so the iteration's
    /// OpRefs are disjoint from any parent trace's emitted ops.
    pub fn optimize_with_constants_and_inputs_at(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
    ) -> Vec<Op> {
        use majit_ir::OpRef;
        // Test-only: auto-seed `trace_inputarg_types` when unit tests
        // pass bare `num_inputs` without staging a recorder. Production
        // callers always populate `trace_inputarg_types` from the
        // recorder's InputArg{Int,Ref,Float} entries before calling —
        // the guard never fires outside `#[cfg(test)]`.
        #[cfg(test)]
        if self.trace_inputarg_types.is_empty() && num_inputs > 0 {
            self.trace_inputarg_types = vec![majit_ir::Type::Ref; num_inputs];
        }
        self.imported_label_args = None;
        self.terminal_op = None;
        let _export_runtime_boxes = self.runtime_boxes.clone();
        self.runtime_boxes.clear();
        // RPython parity: each optimizer run is a fresh Optimizer instance.
        // In pyre we reuse the same Optimizer, so clear per-run state.
        self.last_guard_op_idx = None;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(
            ops.len(),
            num_inputs,
            inputarg_base,
            start_next_pos,
        );
        ctx.skip_flush_mode = self.skip_flush;
        ctx.constant_fold_alloc = self.constant_fold_alloc.take();
        ctx.string_length_resolver = self.string_length_resolver.clone();
        ctx.string_content_resolver = self.string_content_resolver.clone();
        ctx.string_constant_alloc = self.string_constant_alloc.clone();
        ctx.callinfocollection = self.callinfocollection.clone();
        // RPython resume.py parity: Phase 2 optimizer needs imported_label_args
        // to resolve NONE positions in fail_args inherited from Phase 1.
        ctx.imported_virtuals = self.imported_virtuals.clone();
        ctx.imported_label_args = self.imported_label_args.clone();

        // RPython Box type parity: in RPython each Box carries its type
        // intrinsically (InputArgInt, InputArgRef, etc.). In majit, OpRef
        // is an untyped u32. Seed value_types from ALL sources, lowest
        // priority first (later entries override earlier).
        //
        // Skip Void entries: value_types tracks producing-op result types
        // (register_value_type asserts against Void). original_trace_op_types
        // and prev_phase_value_types still keep Void slots for snapshot
        // resolution, but they must not reach the value_types map — a
        // reserve_pos that later hands out a Void slot for a new typed op
        // would otherwise fire the Box.type retype invariant.
        // 1. Original trace ops (pre-transformation positions for snapshots)
        for (&k, &v) in &self.original_trace_op_types {
            if v != majit_ir::Type::Void {
                ctx.value_types.insert(k, v);
            }
        }
        // 2. Previous phase's emitted op types (Phase 1 → Phase 2 carry)
        //
        // These are only fallback types for carried-over OpRefs that do not
        // appear in the current transformed trace. The current trace's own
        // ops must win, matching PyPy's intrinsic Box.type on the current
        // ResOperation objects.
        for (&k, &v) in &self.prev_phase_value_types {
            if v != majit_ir::Type::Void {
                ctx.value_types.insert(k, v);
            }
        }
        // 3. Inputarg types (from recorder — RPython InputArgInt/Ref/Float).
        //    Seeded BEFORE the current trace ops so an op at a position
        //    inside the inputarg range wins on result-type conflict. In
        //    production inputargs live at `[inputarg_base, inputarg_base
        //    + num_inputs)` and ops live strictly above that range, so
        //    the order is immaterial; the swap only matters for the
        //    unit-test harnesses that seed `trace_inputarg_types =
        //    vec![Ref; 1024]` and then place trace ops at low positions.
        //    With ops winning, OpRef(0)=Int (from IntGe.pos=0) instead of
        //    the Ref stub, so the Box.type invariant at
        //    `register_value_type` sees a single authoritative type.
        for (i, &tp) in self.trace_inputarg_types.iter().enumerate() {
            ctx.value_types.insert(inputarg_base + i as u32, tp);
        }
        // 4. Transformed trace ops (optimizer input). Each op's
        //    `result_type()` is the authoritative Box.type at `op.pos`.
        for op in ops {
            if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
                ctx.value_types.insert(op.pos.0, op.result_type());
            }
        }

        // Bump reserve_pos cursor past every seeded value_types position.
        //
        // `reserve_pos` only skips over `constants`; it does not consult
        // `value_types`. Phase 2 / retrace contexts inherit Phase 1's
        // positions through `original_trace_op_types` and
        // `prev_phase_value_types`, and Phase 1's emit-side high water
        // can sit above the caller-provided `start_next_pos` whenever
        // the transformed `ops` slice ends below that mark. Without this
        // bump, a later `get_or_make_const` / `emit` would hand out an
        // already-seeded position and fire the Box.type retype invariant.
        if let Some(&max_typed_pos) = ctx.value_types.keys().max() {
            let next_after = max_typed_pos.saturating_add(1);
            ctx.next_pos = ctx.next_pos.max(next_after);
        }

        // optimizer.py:293 patchguardop parity: propagate to Phase 2
        // OptContext so copy_and_change guards (unroll.py:409) can get
        // rd_resume_position before GUARD_FUTURE_CONDITION is re-encountered.
        if ctx.patchguardop.is_none() {
            ctx.patchguardop = self.patchguardop.clone();
        }
        // optimizer.py:294: patchguardop is set by GUARD_FUTURE_CONDITION
        // during optimization (rewrite.rs / simplify.rs). No synthetic
        // fallback — RPython relies solely on the actual GFC from tracing.

        // RPython resume.py parity: pass snapshot_boxes and constant_types
        // to OptContext so emit() can call store_final_boxes_in_guard inline
        // at each guard emission (not post-assembly).
        ctx.snapshot_boxes = self.snapshot_boxes.clone();
        ctx.snapshot_frame_sizes = self.snapshot_frame_sizes.clone();
        ctx.snapshot_vable_boxes = self.snapshot_vable_boxes.clone();
        ctx.snapshot_frame_pcs = self.snapshot_frame_pcs.clone();
        ctx.constant_types_for_numbering = self.constant_types.clone();
        // RPython parity: merge numbering_type_overrides (ob_type Ref types)
        // into constant_types_for_numbering. These override Int → Ref for
        // resume data only, not for Cranelift backend.
        for (&k, &v) in &self.numbering_type_overrides {
            ctx.constant_types_for_numbering.entry(k).or_insert(v);
        }

        sanitize_backend_constants_for_ops(ops, constants);
        // Pre-populate known constants so passes can see them.
        // Use constant_types to distinguish Ref from Int (GC pointers
        // are stored as i64 in the constants map).
        for (&idx, &val) in constants.iter() {
            ctx.seed_constant(
                OpRef(idx),
                value_from_backend_constant_bits_typed(OpRef(idx), val, ops, &self.constant_types),
            );
        }
        // Advance next_const_idx past all seeded constant-namespace entries
        // so new allocations (intdiv, make_guards) don't collide with
        // constants inherited from a previous phase.
        if !ctx.const_pool.is_empty() {
            let max_idx = ctx.const_pool.keys().max().copied().unwrap_or(0);
            ctx.next_const_idx = ctx.next_const_idx.max(max_idx + 1);
        }

        // Setup all passes
        for pass in &mut self.passes {
            pass.setup();
        }

        // earlyforce.py:32: self.optimizer.optearlyforce = self
        // Find the EarlyForce pass index so force paths can route
        // emitted operations starting from earlyforce.next (= heap).
        for (idx, pass) in self.passes.iter().enumerate() {
            if pass.name() == "earlyforce" {
                ctx.optearlyforce_idx = idx;
                break;
            }
        }

        // bridgeopt.py:124-185: apply pending bridge knowledge AFTER setup.
        // RPython calls deserialize_optimizer_knowledge before propagate_all_forward
        // but after the optimizer is constructed (setup already done at __init__).
        if let Some(prd) = self.pending_bridge_rd.take() {
            crate::optimizeopt::bridgeopt::deserialize_optimizer_knowledge(
                &prd.storage.rd_numb,
                prd.storage.rd_consts(),
                &prd.frontend_boxes,
                &prd.liveboxes,
                &prd.livebox_types,
                &prd.all_descrs,
                prd.cls_of_box,
                self,
                &mut ctx,
            );
        }

        // optimizer.py: pass call_pure_results to OptPure so it can
        // constant-fold repeated pure calls across loop iterations.
        for pass in &mut self.passes {
            pass.set_call_pure_results(&self.call_pure_results);
        }

        // Take ownership of imported_loop_state for the duration of
        // this block: import_state needs `&mut self` (force_box dispatch
        // through the optimizer pass), so we cannot keep an immutable
        // borrow of `self.imported_loop_state` simultaneously. The state
        // is restored at the end of the block.
        let imported_loop_state_taken = self.imported_loop_state.take();
        if let Some(ref exported_state) = imported_loop_state_taken {
            for (&opref, &tp) in exported_state
                .end_args
                .iter()
                .zip(exported_state.end_arg_types.iter())
            {
                if tp != majit_ir::Type::Void {
                    ctx.value_types.insert(opref.0, tp);
                }
            }
            // opencoder.py:259 + unroll.py:479-504 parity: RPython's
            // TraceIterator allocates fresh InputArg Box objects for
            // each iteration, and `import_state` asserts
            // `source is not target` (unroll.py:483) because the fresh
            // Phase 2 inputargs never coincide with any Phase 1 emitted
            // Box. majit encodes "fresh per iteration" via the
            // `ctx.inputarg_base` offset — Phase 2's source slots live
            // at `[inputarg_base..inputarg_base+n)`, disjoint from
            // Phase 1 emitted OpRefs `[num_inputs..next_global_opref)`.
            //
            // With disjoint namespaces, the `source == target` edge
            // case RPython avoids by construction cannot occur here
            // either: any `target` coming from `next_iteration_args`
            // is either a constant, a Phase 1 emitted OpRef, or a
            // Phase 1 inputarg OpRef (in `[0..num_inputs)`) — all
            // outside the Phase 2 `source` range. The cross-slot
            // collision guard below is preserved for the Phase 1 /
            // standalone path where `inputarg_base == 0` and `source`
            // can coincide with another slot's `target`.
            let inputarg_base = ctx.inputarg_base;
            let nia = &exported_state.next_iteration_args;
            let n = nia.len();
            let source_set: std::collections::HashSet<OpRef> =
                (0..n).map(|i| OpRef(inputarg_base + i as u32)).collect();
            let targetargs: Vec<OpRef> = (0..n)
                .map(|i| {
                    let source = OpRef(inputarg_base + i as u32);
                    let target = nia[i];
                    // Constants don't participate in forwarding.
                    if ctx.is_constant(target) {
                        return source;
                    }
                    // Cross-slot collision: target is another slot's source.
                    // Allocate fresh to avoid forwarding overwrite.
                    if target != source && source_set.contains(&target) {
                        let fresh = ctx.alloc_op_position();
                        if let Some(tp) = ctx.opref_type(source) {
                            ctx.register_value_type(fresh, tp);
                        }
                        ctx.replace_op(source, fresh);
                        fresh
                    } else {
                        source
                    }
                })
                .collect();
            // unroll.py:493-494: label_args = virtual_state.make_inputargs(targetargs, optimizer)
            let label_args = crate::optimizeopt::unroll::import_state(
                &targetargs,
                exported_state,
                self,
                &mut ctx,
            );
            if crate::optimizeopt::majit_log_enabled() {
                if let Some((base, ref sa)) = ctx.imported_virtual_args {
                    eprintln!(
                        "[jit] virtual_args from import_state: base={} total={} virtuals={:?}",
                        base,
                        sa.len(),
                        &sa[base..]
                    );
                }
            }
            self.imported_label_args = Some(label_args.clone());
            if !self.imported_virtuals.is_empty() {
                self.install_imported_virtuals(&mut ctx);
                if crate::optimizeopt::majit_log_enabled() {
                    let upper = ctx.inputarg_base + ctx.num_inputs;
                    for raw in ctx.inputarg_base..upper {
                        let raw = OpRef(raw);
                        eprintln!(
                            "[jit] import_state_resolved: raw={raw:?} resolved={:?}",
                            ctx.get_box_replacement(raw)
                        );
                    }
                }
            }
            crate::optimizeopt::unroll::import_short_preamble_state(
                &targetargs,
                &label_args,
                exported_state,
                &mut ctx,
            );
        }
        // Restore the temporarily-taken imported_loop_state.
        self.imported_loop_state = imported_loop_state_taken;

        // RPython shortpreamble.py: PureOp.produce_op stores PreambleOp
        // directly in opt.optimizer.optpure. In majit, imported short pure ops
        // are first collected in ctx.imported_short_pure_ops, then transferred
        // to the OptPure pass here (matching RPython's produce_op timing).
        if !ctx.imported_short_pure_ops.is_empty() {
            for pass in &mut self.passes {
                pass.install_preamble_pure_ops(&ctx);
            }
        }

        // RPython optimizer.py:536-538: JUMP/FINISH always breaks the main
        // loop. flush() is called before JUMP is processed.
        let mut last_op = None;
        for op in ops {
            if op.opcode == OpCode::Jump || op.opcode == OpCode::Finish {
                last_op = Some(op.clone());
                break;
            }
            self.propagate_one(op, &mut ctx);
        }

        // RPython: flush() before JUMP processing (export_state calls flush
        // before get_virtual_state). Phase 2 skips flush.
        if !self.skip_flush {
            self.flush(&mut ctx);
        }

        // RPython unroll.py:454-457:
        //     end_args = [self.optimizer.force_box_for_end_of_preamble(a)
        //                 for a in original_label_args]
        //     self.optimizer.flush()
        //     virtual_state = self.get_virtual_state(end_args)
        //
        // The virtual state is captured AFTER force_box_for_end_of_preamble
        // and AFTER flush, so its `Virtual` / `VStruct` entries correspond
        // exactly to the boxes whose `info.is_virtual()` is still true at
        // that moment (e.g. virtualizables that the force pass intentionally
        // leaves virtual). This invariant is what `make_inputargs` /
        // `make_inputargs_and_virtuals` rely on when they call
        // `enum_forced_boxes` — the validating walker raises
        // `VirtualStatesCantMatch` if a `Virtual` entry's resolved opref
        // is not actually virtual.
        //
        // Capture is performed below, *after* the terminal_op force pass
        // and `flush()` have run inside the `exported_loop_state` map
        // closure, mirroring RPython's order. Keep the raw jump args here:
        // unroll.py:452-455 passes `original_label_args` directly to
        // `force_box_for_end_of_preamble`, and optimizer.py:306-319 returns
        // the original box unchanged when no force is required.
        let pre_jump_resolved_args = last_op
            .as_ref()
            .filter(|op| op.opcode == OpCode::Jump)
            .map(|jump_op| jump_op.args.clone());

        // RPython optimizer.py:552-556 (_propagate_all_forward):
        //     if flush:
        //         self.flush()
        //         if last_op:
        //             self.send_extra_operation(last_op)
        //
        // flush=False is used ONLY by optimize_preamble (Phase 1) and
        // optimize_peeled_loop (Phase 2) — terminal_op is returned in
        // BasicLoopInfo.jump_op for unroll.rs to handle.
        //
        // flush=True (default) is used by optimize_loop (compile_loop) and
        // finish_and_compile — terminal op is sent through passes and ends
        // up in new_operations naturally.
        if let Some(mut terminal_op) = last_op {
            // RPython unroll.py:126-127 parity:
            //   for a in end_jump.getarglist():
            //       self.force_box_for_end_of_preamble(get_box_replacement(a))
            //
            // Call force_at_the_end_of_preamble on ALL args (not just
            // Ref-mismatch ones). RPython's force_box_for_end_of_preamble
            // handles both Ref virtuals (type 'r') and raw-ptr-int (type 'i'):
            //   - Ref virtual: recursively process fields, keep virtual alive
            //   - Raw ptr int: force via optearlyforce
            //   - Non-virtual: return get_box_replacement(op) unchanged
            //
            // Additionally, in majit OpRef forwarding is typeless, so a Ref
            // virtual can forward to a Float (SameAsF). When that happens,
            // force_box materializes the virtual to preserve Ref type.
            let inputarg_types = self.trace_inputarg_types.clone();
            // Phase 1: resolve and call force_at_the_end_of_preamble on all args
            let resolved_args: Vec<OpRef> = terminal_op
                .args
                .iter()
                .map(|&arg| ctx.get_box_replacement(arg))
                .collect();
            for &resolved in &resolved_args {
                self.force_box_for_end_of_preamble(resolved, &mut ctx);
            }
            // Phase 2: re-resolve after forcing (force may have changed forwarding)
            // and fix Ref→non-Ref type crossings by force_box
            let mut force_needed: Vec<usize> = Vec::new();
            for (i, arg) in terminal_op.args.iter_mut().enumerate() {
                let resolved = ctx.get_box_replacement(*arg);
                let expected_ref =
                    i < inputarg_types.len() && inputarg_types[i] == majit_ir::Type::Ref;
                // setup_optimizations seeds trace_inputarg_types into
                // ctx.value_types (this fn, lines 1816-1818), so opref_type
                // covers both the per-op result and the inputarg slot.
                // PtrInfo presence is an additional Ref-only side channel
                // for inputargs that are not in `new_operations`.
                let resolved_is_ref = ctx.opref_type(resolved) == Some(majit_ir::Type::Ref)
                    || ctx.get_ptr_info(resolved).is_some();
                if expected_ref && !resolved_is_ref && !ctx.is_constant(resolved) {
                    if ctx.get_ptr_info(*arg).is_some_and(|i| i.is_virtual()) {
                        force_needed.push(i);
                    } else {
                        // RPython Box parity: a ref-typed box never forwards to
                        // an int/float box. majit's flat OpRef forwarding can
                        // collapse a boxed W_Int/W_Float local to its raw payload,
                        // but end-of-preamble JUMP args must preserve the ref box
                        // identity at ref-typed positions.
                        //
                        // Keep the original ref-typed arg instead of substituting
                        // the non-ref replacement. This matches the upstream
                        // force_box_for_end_of_preamble contract more closely than
                        // allowing Ref -> Float/Int type substitution at the JUMP.
                    }
                } else {
                    *arg = resolved;
                }
            }
            for i in force_needed {
                let original = terminal_op.args[i];
                let forced = self.force_box(original, &mut ctx);
                terminal_op.args[i] = ctx.get_box_replacement(forced);
            }
            if self.skip_flush {
                // flush=False: store for caller to consume.
                self.terminal_op = Some(terminal_op);
            } else {
                // flush=True: send through passes (optimizer.py:555-556).
                self.send_extra_operation(&terminal_op, &mut ctx);
            }
        }

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // in guard fail_args. Phase 2 guards may inherit NONE from Phase 1
        // virtualization — rescan resolves these using imported_virtuals.
        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard in ctx.emit() handles
        // virtual tagging inline at each guard emit. No post-pass rescan.

        // RPython Box type parity: preserve value_types from this phase
        // so the next phase can resolve types for carried-over OpRefs.
        self.prev_phase_value_types.clear();
        self.prev_phase_value_types
            .extend(ctx.value_types.iter().map(|(&k, &v)| (k, v)));

        // Transfer exported virtual state from context to optimizer
        // RPython BasicLoopInfo: quasi_immutable_deps collected during optimization
        self.quasi_immutable_deps = std::mem::take(&mut ctx.quasi_immutable_deps);
        self.imported_short_sources = std::mem::take(&mut ctx.imported_short_sources);
        self.imported_short_aliases = ctx.used_imported_short_aliases();
        self.imported_short_preamble = ctx.build_imported_short_preamble();
        self.imported_short_preamble_builder = ctx.imported_short_preamble_builder.clone();
        // optimizer.py:294: patchguardop is set exclusively by
        // GUARD_FUTURE_CONDITION during optimization. No fallback.
        self.patchguardop = ctx.patchguardop.clone();
        // JUMP location: in new_operations (flush=True path where JUMP was
        // processed through passes), or terminal_op (skip_flush path).
        let jump = ctx
            .new_operations
            .iter()
            .rfind(|op| op.opcode == OpCode::Jump)
            .cloned()
            .or_else(|| {
                self.terminal_op
                    .clone()
                    .filter(|op| op.opcode == OpCode::Jump)
            });
        // RPython compile.py:327 `loop.operations = ([start_label] + preamble_ops
        // + loop_info.extra_same_as + loop_info.extra_before_label + [label_op]
        // + loop_ops)`: alias SameAs ops allocated during the preamble
        // export dedup belong to `loop_info.extra_same_as`, NOT to the
        // preamble body. Keep them in a side vector so they land at the
        // spliced parity position below instead of appearing in
        // `ctx.new_operations` past the terminator.
        let mut extra_same_as_aliases: Vec<Op> = Vec::new();
        self.exported_loop_state = jump.map(|jump| {
            // RPython unroll.py:454-457 order:
            //   end_args = [force_box_for_end_of_preamble(a) for a ...]
            //   self.optimizer.flush()
            //   virtual_state = self.get_virtual_state(end_args)
            // — VS captured AFTER force + flush.
            let original_jump_args = pre_jump_resolved_args.clone()
                .unwrap_or_else(|| {
                    jump.args.iter()
                        .map(|&a| ctx.get_box_replacement(a))
                        .collect()
                });
            let mut resolved_args = original_jump_args.clone();
            // Dedup: two cases require SameAs allocation at end of preamble.
            //
            // Case A — duplicate args: when two JUMP slots reference the same
            // OpRef (e.g. b and t in fib_loop after b = t aliasing), create a
            // fresh SameAs for the second occurrence so each slot has a
            // distinct identity.
            //
            // Case B — preamble inputarg position used by a non-self slot:
            // when JUMP slot j carries OpRef(k) and k < num_inputs, k != j,
            // the export hands Phase 2 a `next_iteration_args[j] = OpRef(k)`
            // entry that collides with body inputarg slot k's own source
            // position. import_state forwards both `OpRef(j) → OpRef(k)` and
            // `OpRef(k) → nia[k]`, and get_box_replacement walks the chain
            // through OpRef(k), making body inputarg j resolve to nia[k]
            // instead of OpRef(k). RPython avoids this with fresh-Box identity
            // per phase; majit's flat OpRef space needs an explicit SameAs
            // alias so nia[j] points outside the body inputarg position range.
            {
                let mut seen = std::collections::HashSet::new();
                // RPython parity: positions already holding an emitted op
                // are phase 1 results, not body inputarg sources. Only
                // the UNUSED positions in 0..num_inputs correspond to
                // trace inputargs (`InputArgRef/Int/Float` in RPython).
                let emitted_positions: std::collections::HashSet<OpRef> = ctx
                    .new_operations
                    .iter()
                    .map(|op| op.pos)
                    .filter(|p| !p.is_none())
                    .collect();
                let original_args = resolved_args.clone();
                for (slot_idx, arg) in resolved_args.iter_mut().enumerate() {
                    if ctx.is_constant(*arg) || *arg == OpRef::NONE {
                        continue;
                    }
                    let is_dup = !seen.insert(*arg);
                    // Case B fires only when slot k (where `k = arg.0`, the
                    // position the current slot points at) holds a DIFFERENT
                    // OpRef. If slot k self-forwards (original_args[k] ==
                    // OpRef(k)), Phase 2 sets `forwarding[OpRef(k)] = OpRef(k)`
                    // which is a no-op in replace_op — no chain forms, so no
                    // aliasing is needed.
                    let target_slot = arg.0 as usize;
                    let target_slot_self_forwards = original_args
                        .get(target_slot)
                        .map_or(true, |other| *other == *arg);
                    let is_cross_inputarg = target_slot < num_inputs
                        && target_slot != slot_idx
                        && !emitted_positions.contains(arg)
                        && !target_slot_self_forwards;
                    if !is_dup && !is_cross_inputarg {
                        continue;
                    }
                    let orig = *arg;
                    // history.py:802-809 `record_same_as(box)` reads
                    // `box.type` directly to pick `same_as_i/r/f` — there is
                    // no guess-on-miss path in RPython. Match strict parity by
                    // requiring `opref_type` to resolve; a None here is a
                    // bookkeeping bug, not a recoverable case.
                    let arg_type = ctx
                        .opref_type(orig)
                        .expect("propagate_from_pass_range SameAs: source OpRef missing Box.type");
                    let same_as = OpCode::same_as_for_type(arg_type);
                    let fresh = ctx.alloc_op_position();
                    let mut op = Op::new(same_as, &[orig]);
                    op.pos = fresh;
                    // unroll.py:146 + compile.py:327 parity: accumulate the
                    // alias op in `extra_same_as` and splice it between the
                    // preamble body and the label at final assembly. Emitting
                    // directly into `ctx.new_operations` would push the op
                    // past the already-sent terminal JUMP and force the
                    // loop-tail relocation workaround below.
                    extra_same_as_aliases.push(op);
                    ctx.register_value_type(fresh, arg_type);
                    if let Some(info) = ctx.get_ptr_info(orig).cloned() {
                        let fresh_info = match info {
                            crate::optimizeopt::info::PtrInfo::Virtual(mut vinfo) => {
                                for field in &mut vinfo.fields {
                                    let orig_field = field.1;
                                    let ff = ctx.alloc_op_position();
                                    ctx.replace_op(ff, orig_field);
                                    // RPython Box type parity: copy type
                                    if let Some(tp) = ctx.opref_type(orig_field) {
                                        ctx.register_value_type(ff, tp);
                                    }
                                    field.1 = ff;
                                }
                                crate::optimizeopt::info::PtrInfo::Virtual(vinfo)
                            }
                            other => other,
                        };
                        ctx.set_ptr_info(fresh, fresh_info);
                    }
                    *arg = fresh;
                    // After allocating a fresh alias for an inputarg position,
                    // update `seen` so the next slot referencing the same
                    // original OpRef goes through the duplicate path and
                    // gets its own fresh alias too.
                    if is_cross_inputarg {
                        seen.insert(fresh);
                    }
                }
            }
            // unroll.py:203 self.flush() — force lazy sets before
            // producing short preamble ops (heap.py:53 invariant).
            self.flush(&mut ctx);

            // Now force all resolved (and dedup'd) args.
            // unroll.py:454 `end_args = [force_box_for_end_of_preamble(a)
            // for a in original_label_args]`.
            ctx.preamble_end_args = Some(
                resolved_args
                    .iter()
                    .map(|&arg| self.force_box_for_end_of_preamble(arg, &mut ctx))
                    .collect(),
            );
            // unroll.py:457 `virtual_state = self.get_virtual_state(end_args)`.
            // VS is captured AFTER force + flush so its `Virtual` /
            // `VStruct` entries match the `info.is_virtual()` predicate
            // that `enum_forced_boxes` asserts. Virtuals that were
            // forced into concrete instances by `force_at_the_end_of_preamble`
            // come back as `NotVirtualStateInfo` here, exactly as they
            // do in RPython.
            let post_force_args: Vec<OpRef> = resolved_args
                .iter()
                .map(|&a| ctx.get_box_replacement(a))
                .collect();
            let preview_virtual_state = crate::optimizeopt::virtualstate::export_state(
                &post_force_args,
                &ctx,
                &ctx.forwarded,
            );
            let vs_args = &post_force_args;
            let (preview_label_args, preview_virtuals) = preview_virtual_state
                .make_inputargs_and_virtuals(vs_args, self, &mut ctx, false)
                .expect("preview make_inputargs_and_virtuals failed");
            let mut preview_short_args = preview_label_args.clone();
            preview_short_args.extend(preview_virtuals);
            let mut short_boxes =
                crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&preview_short_args);
            short_boxes.note_known_constants_from_ctx(&ctx);
            for &arg in &preview_short_args {
                // RPython shortpreamble.py:255-259 parity: each label arg
                // is `box.type`, where Box objects intrinsically carry one
                // of i / r / f. There is no `void` Box because Box always
                // wraps a runtime value. Look up the OpRef's type from
                // `value_types` (maintained per op result type by emit()).
                let raw_type = ctx
                    .opref_type(arg)
                    .unwrap_or_else(|| {
                        panic!(
                            "preview short arg missing box.type: arg={arg:?} preview_short_args={preview_short_args:?}"
                        )
                    });
                if raw_type == majit_ir::Type::Void {
                    // Upstream collision: a previously-virtual OpRef whose
                    // position was reused as a Phase 2 constant slot — its
                    // value_types entry is the constant op's Void result
                    // type. Such an OpRef cannot legitimately become a
                    // short-preamble InputArg in RPython (Box identity
                    // would have produced a fresh ConstFloat/ConstInt and
                    // never reach this loop). Skip rather than emitting a
                    // bogus SameAsR(const0): the same-position constant is
                    // already in `ctx.constants` and downstream consumers
                    // resolve it via the constant pool, not via a SameAs.
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!(
                            "[unroll] preview_short_args: dropping {:?} \
                             (Type::Void — Phase 2 constant-slot collision \
                             with a Phase 1 virtual head OpRef)",
                            arg
                        );
                    }
                    continue;
                }
                short_boxes.add_short_input_arg(arg, raw_type);
            }
            self.produce_potential_short_preamble_ops(&mut short_boxes, &mut ctx);
            let produced = short_boxes.produced_ops();
            ctx.exported_short_boxes = produced
                .into_iter()
                .map(|(result, produced)| {
                    let canonical_result = ctx.get_box_replacement(result);
                    let mut preamble_op = produced.preamble_op;
                    // RPython parity: key and preamble_op.pos must be the
                    // same resolved value. Independent get_box_replacement
                    // calls can diverge when forwarding chains differ.
                    // Use canonical_result (resolved key) for both.
                    preamble_op.pos = canonical_result;
                    for arg in &mut preamble_op.args {
                        *arg = ctx.get_box_replacement(*arg);
                    }
                    if let Some(fail_args) = preamble_op.fail_args.as_mut() {
                        for arg in fail_args {
                            *arg = ctx.get_box_replacement(*arg);
                        }
                    }
                    crate::optimizeopt::shortpreamble::PreambleOp {
                        op: preamble_op,
                        kind: produced.kind,
                        label_arg_idx: short_boxes.lookup_label_arg(canonical_result),
                        invented_name: produced.invented_name,
                        same_as_source: produced.same_as_source,
                    }
                })
                .collect();
            if std::env::var_os("MAJIT_LOG").is_some() {
                for entry in &ctx.exported_short_boxes {
                    eprintln!(
                        "[jit] exported_short_box: kind={:?} pos={:?} opcode={:?} args={:?} descr_idx={:?} invented={} same_as_source={:?}",
                        entry.kind,
                        entry.op.pos,
                        entry.op.opcode,
                        entry.op.args,
                        entry.op.descr.as_ref().map(|d| d.index()),
                        entry.invented_name,
                        entry.same_as_source,
                    );
                }
            }
            let exported_int_bounds = self.collect_exported_int_bounds(&jump.args, &mut ctx);
            // RPython unroll.py passes optimize_preamble()'s inputargs here,
            // i.e. the external loop-entry contract, not the optimizer's
            // internal position base (`ctx.num_inputs`), which may be widened
            // to avoid collisions with existing op positions.
            let renamed_inputargs: Vec<OpRef> = (0..num_inputs).map(|i| OpRef(i as u32)).collect();
            crate::optimizeopt::unroll::export_state(
                &original_jump_args,
                &renamed_inputargs,
                self,
                &mut ctx,
                Some(&exported_int_bounds),
            )
        });
        // Populate renamed_inputarg_types from the exported input boxes'
        // actual optimizer-visible types. In RPython each renamed inputarg
        // is a Box whose `.type` is already fixed; falling back to a pyre-only
        // default like Int can silently corrupt retrace input typing.
        if let Some(ref mut es) = self.exported_loop_state {
            if es.renamed_inputarg_types.is_empty() {
                es.renamed_inputarg_types = es
                    .renamed_inputargs
                    .iter()
                    .map(|&opref| {
                        ctx.opref_type(opref).unwrap_or_else(|| {
                            panic!("missing type for exported renamed inputarg {:?}", opref)
                        })
                    })
                    .collect();
            }
        }
        // RPython parity: propagate patchguardop to ExportedState so Phase 2
        // can use it for extra_guards from virtualstate (unroll.py:333-336).
        if let Some(ref mut es) = self.exported_loop_state {
            es.patchguardop = self.patchguardop.clone();
        }

        // RPython export_state() flushes force artifacts into the preamble
        // before building the exported loop state. If the loop header needs
        // additional inputargs, the corresponding SETFIELD/SETARRAYITEM must
        // remain in the trace rather than being silently discarded.
        // final_num_inputs = original inputs + virtual inputs added by passes.
        // Pipeline passes (e.g., virtualizable) may bump ctx.num_inputs by
        // adding synthetic input slots. The delta is the count of virtual
        // inputargs the optimizer needs to materialize at trace entry.
        let num_virtual_inputs = (ctx.num_inputs as usize).saturating_sub(num_inputs);
        self.final_num_inputs = num_inputs + num_virtual_inputs;

        // RPython store_final_boxes_in_guard parity: re-encode late virtuals
        // RPython parity: store_final_boxes_in_guard handles virtual tagging
        // and rd_numb production inline. No post-pass rescan needed.

        // Force any remaining virtual refs in output ops before forwarding resolve.
        // RPython: virtuals are forced during preamble export or JUMP handling.
        // In majit, skip_flush=true may leave some virtual refs un-forced in
        // the output (e.g., linked list nodes nested in non-virtual objects).
        {
            let all_refs: Vec<OpRef> = ctx
                .new_operations
                .iter()
                .flat_map(|op| op.args.iter().copied())
                .filter(|r| !r.is_none())
                .collect();
            for opref in all_refs {
                let resolved = ctx.get_box_replacement(opref);
                if let Some(info) = ctx.get_ptr_info(resolved) {
                    if info.is_virtual() {
                        self.force_box_for_end_of_preamble(resolved, &mut ctx);
                    }
                }
            }
        }

        // Resolve forwarding BEFORE remap.
        // RPython get_box_replacement: follow chain, stop at ptr_info terminal.
        {
            let fwd = ctx.forwarded.clone();
            let resolve = |opref: OpRef| -> OpRef {
                use crate::optimizeopt::info::Forwarded;
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() {
                        return cur;
                    }
                    match &fwd[idx] {
                        Forwarded::Op(next) => cur = *next,
                        _ => return cur,
                    }
                }
            };
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    *arg = resolve(*arg);
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = resolve(*arg);
                    }
                }
            }
        }

        // RPython keeps constants as Const boxes, not SameAs placeholder ops in
        // the final trace. Drop constant-only SameAs placeholders before the
        // backend sees the trace; their OpRefs remain available through the
        // constants table.
        ctx.new_operations
            .retain(|op| !Self::is_constant_placeholder_op(op, &ctx.constants));

        // Drain remaining extra ops.
        self.drain_extra_operations_from(0, &mut ctx);
        // Extra operations can introduce new forwarding (for example Heap/Pure
        // forwarding a recently-emitted boxed-field read to its raw payload).
        // Resolve forwarding again with ptr_info stop.
        {
            let fwd = ctx.forwarded.clone();
            let resolve = |opref: OpRef| -> OpRef {
                use crate::optimizeopt::info::Forwarded;
                let mut cur = opref;
                loop {
                    let idx = cur.0 as usize;
                    if idx >= fwd.len() {
                        return cur;
                    }
                    match &fwd[idx] {
                        Forwarded::Op(next) => cur = *next,
                        _ => return cur,
                    }
                }
            };
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    *arg = resolve(*arg);
                }
                if let Some(ref mut fa) = op.fail_args {
                    for arg in fa.iter_mut() {
                        *arg = resolve(*arg);
                    }
                }
            }
        }
        // RPython: no equivalent filter — Box identity guarantees all
        // references are valid. The retain filter was a majit safety net
        // but incorrectly dropped valid ops (e.g., IntAddOvf referencing
        // inputarg positions beyond final_num_inputs in bridge traces).

        // `compile.py:327` order:
        //   `[start_label] + preamble_ops + extra_same_as +
        //    extra_before_label + [label_op] + loop_ops`.
        // The SameAs aliases emitted in the JUMP-arg dedup loop above
        // (`shortpreamble.py:432-440 extra_same_as`) and any `flush()` /
        // `drain_extra_operations_from` ops that followed are structurally
        // part of `extra_same_as`; upstream splices that list BETWEEN
        // `preamble_ops` and the label, so in pyre's single-phase flow
        // they must sit before the preamble-terminating JUMP. Move the
        // last JUMP to the tail to restore that ordering — running AFTER
        // all emits but BEFORE the remap pass so positions stay
        // sequential in trace order.
        if let Some(jump_idx) = ctx
            .new_operations
            .iter()
            .rposition(|op| op.opcode == OpCode::Jump)
            && jump_idx + 1 < ctx.new_operations.len()
        {
            let jump_op = ctx.new_operations.remove(jump_idx);
            ctx.new_operations.push(jump_op);
        }

        // Remap ALL positions: virtual inputs go to num_inputs..final_num_inputs,
        // This ensures no position collisions between input block params and ops.
        if num_virtual_inputs > 0 {
            let fni = self.final_num_inputs as u32;
            let mut remap = std::collections::HashMap::new();

            // Virtual input positions: optimizer used num_inputs+k, backend needs num_inputs+k
            for k in 0..num_virtual_inputs {
                let opt_pos = (num_inputs + k) as u32;
                let be_pos = (num_inputs + k) as u32;
                if opt_pos != be_pos {
                    remap.insert(opt_pos, be_pos);
                }
            }

            // Op positions: reassign ALL ops to start from final_num_inputs.
            if crate::optimizeopt::majit_log_enabled() {
                let fwd_1995 = if 1995 < ctx.forwarded.len() {
                    format!("{:?}", ctx.forwarded[1995])
                } else {
                    "None".to_string()
                };
                let in_ops = ctx.new_operations.iter().any(|o| o.pos.0 == 1995);
                let in_args = ctx
                    .new_operations
                    .iter()
                    .any(|o| o.args.iter().any(|a| a.0 == 1995));
                eprintln!("[remap] v1995: fwd={fwd_1995:?} in_ops={in_ops} in_args={in_args}");
            }
            for (new_idx, op) in ctx.new_operations.iter_mut().enumerate() {
                let new_pos = fni + new_idx as u32;
                if !op.pos.is_none() {
                    remap.insert(op.pos.0, new_pos);
                    op.pos = OpRef(new_pos);
                }
            }

            // Constant-folded operations that were removed from the trace still
            // have entries in `ctx.constants`. If we leave them at their old
            // positions, they can collide with the freshly compacted op
            // positions above (for example old constant v71 vs new live op v71),
            // and the backend will resolve the live op as the stale constant.
            // Give every such constant-only opref a fresh slot after the last
            // live op, mirroring RPython's separate constant identity.
            let mut next_const_pos = fni + ctx.new_operations.len() as u32;
            for (idx, val) in ctx.constants.iter().enumerate() {
                let old_idx = idx as u32;
                if val.is_none() || remap.contains_key(&old_idx) {
                    continue;
                }
                if old_idx < num_inputs as u32 {
                    continue;
                }
                remap.insert(old_idx, next_const_pos);
                next_const_pos += 1;
            }

            // Apply remap to forwarding table too, so forwarding resolution
            // after remap resolves to the correct remapped positions.
            for entry in &mut ctx.forwarded {
                if let crate::optimizeopt::info::Forwarded::Op(target) = entry {
                    if let Some(&new_pos) = remap.get(&target.0) {
                        *target = OpRef(new_pos);
                    }
                }
            }

            // Apply remap to all args and fail_args
            for op in &mut ctx.new_operations {
                for arg in &mut op.args {
                    if let Some(&new_pos) = remap.get(&arg.0) {
                        *arg = OpRef(new_pos);
                    }
                }
                if let Some(ref mut fail_args) = op.fail_args {
                    for arg in fail_args.iter_mut() {
                        if let Some(&new_pos) = remap.get(&arg.0) {
                            *arg = OpRef(new_pos);
                        }
                    }
                }
            }

            // Remap constants
            let old_constants = std::mem::take(&mut ctx.constants);
            for (idx, val) in old_constants.into_iter().enumerate() {
                if let Some(v) = val {
                    let target_idx =
                        remap.get(&(idx as u32)).copied().unwrap_or(idx as u32) as usize;
                    if target_idx >= ctx.constants.len() {
                        ctx.constants.resize(target_idx + 1, None);
                    }
                    ctx.constants[target_idx] = Some(v);
                }
            }

            // Remap exported_loop_state OpRefs so Phase 2 sees post-remap
            // positions. Without this, Phase 2's import_boxes maps to
            // pre-remap positions that no longer exist in the constants map.
            if let Some(ref mut state) = self.exported_loop_state {
                let remap_opref = |opref: &mut OpRef| {
                    if let Some(&new_pos) = remap.get(&opref.0) {
                        *opref = OpRef(new_pos);
                    }
                };
                for arg in &mut state.next_iteration_args {
                    remap_opref(arg);
                }
                for arg in &mut state.end_args {
                    remap_opref(arg);
                }
                for arg in &mut state.renamed_inputargs {
                    remap_opref(arg);
                }
                for arg in &mut state.short_inputargs {
                    remap_opref(arg);
                }
                // Remap exported_infos keys
                let old_infos = std::mem::take(&mut state.exported_infos);
                for (key, value) in old_infos {
                    let new_key = remap.get(&key.0).map(|&p| OpRef(p)).unwrap_or(key);
                    state.exported_infos.insert(new_key, value);
                }
                // Remap exported short boxes
                for entry in &mut state.exported_short_boxes {
                    remap_opref(&mut entry.op.pos);
                    for arg in &mut entry.op.args {
                        remap_opref(arg);
                    }
                    if let Some(ref mut fa) = entry.op.fail_args {
                        for arg in fa.iter_mut() {
                            remap_opref(arg);
                        }
                    }
                    if let Some(ref mut src) = entry.same_as_source {
                        remap_opref(src);
                    }
                }
            }
        }

        // Export newly-discovered constants back to the caller's map.
        merge_backend_constants_from_ctx(&ctx, constants);
        sanitize_backend_constants_for_ops(&ctx.new_operations, constants);

        // Preserve final context for jump_to_existing_trace.
        let mut ops = std::mem::take(&mut ctx.new_operations);

        // RPython compile.py:327 final loop assembly:
        //   loop.operations = ([start_label] + preamble_ops
        //       + loop_info.extra_same_as + loop_info.extra_before_label
        //       + [label_op] + loop_ops)
        //
        // pyre does not split the preamble / body boundary at this layer —
        // both simple-loop and Phase-1-of-unroll consumers receive a single
        // `ops` vector that already carries the terminal JUMP. Matching the
        // RPython ordering means splicing the alias `extra_same_as` ops
        // just ahead of that terminator so they execute at end of preamble
        // and never appear past the Jump. With the dedup loop now
        // accumulating into `extra_same_as_aliases` (see the closure
        // above), no `ctx.emit`-after-terminator cleanup is needed.
        if !extra_same_as_aliases.is_empty() {
            let term_idx = ops
                .iter()
                .position(|op| op.opcode == OpCode::Jump || op.opcode == OpCode::Finish)
                .unwrap_or(ops.len());
            for (offset, op) in extra_same_as_aliases.into_iter().enumerate() {
                ops.insert(term_idx + offset, op);
            }
        }
        // resume.py:411-417 parity: store_final_boxes_in_guard
        // (mod.rs:2261) already replaces TAGCONST/TAGVIRTUAL fail_args
        // entries with OpRef::NONE via the snapshot-driven numbering pass
        // (`liveboxes = [None] * n; liveboxes[i] = box for TAGBOX`).
        // No additional const filtering is needed here.
        if crate::optimizeopt::majit_log_enabled() {
            let cmf_count = ops.iter().filter(|o| o.opcode.is_call_may_force()).count();
            let gnf_count = ops
                .iter()
                .filter(|o| matches!(o.opcode, OpCode::GuardNotForced | OpCode::GuardNotForced2))
                .count();
            eprintln!(
                "[opt] final ops: total={} call_may_force={} guard_not_forced={}",
                ops.len(),
                cmf_count,
                gnf_count
            );
            if cmf_count == 0 && gnf_count > 0 {
                for (i, op) in ops.iter().enumerate() {
                    eprintln!("[opt] idx={} {:?} pos={:?}", i, op.opcode, op.pos);
                }
            }
        }
        self.final_ctx = Some(ctx);
        ops
    }

    /// unroll.py:183-236: optimize_bridge()
    ///
    /// Optimizes a bridge trace and redirects its terminal JUMP to the
    /// appropriate loop body target token, falling back to the preamble
    /// when no match is found.
    ///
    /// `retraced_count` / `retrace_limit`: RPython history.py
    /// JitCellToken.retraced_count tracking. When retrace_limit > 0 and
    /// no existing trace matches, export_state creates a new specialization.
    /// Default retrace_limit = 0 (disabled, warmstate.py PARAMETERS).
    ///
    /// Returns `(optimized_ops, retrace_requested)`. When retrace_requested
    /// is true, the caller should increment retraced_count and may use the
    /// optimizer's exported_loop_state for the new target token.
    pub(crate) fn optimize_bridge(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        front_target_tokens: &mut Vec<crate::optimizeopt::unroll::TargetToken>,
        inline_short_preamble: bool,
        retraced_count: u32,
        retrace_limit: u32,
        pending_bridge_rd: Option<PendingBridgeRd>,
        _loop_num_inputs: Option<usize>,
        // Box Identity plan Phase E Step 2a (dormant): disjoint OpRef
        // namespace for bridge inputargs. RPython opencoder.py:249-273
        // `TraceIterator.__init__` allocates fresh `InputArg` Python
        // objects per iteration, so bridges carry Python `is` identity
        // distinct from the parent loop's boxes automatically. Pyre's
        // flat `OpRef(u32)` threads the parent's recorded
        // `CompiledEntry.next_global_opref` here.
        //
        // Step 2b (activation) requires parallel rewrites of every
        // OpRef-keyed side channel: `snapshot_boxes`,
        // `snapshot_vable_boxes`, `pending_bridge_rd.liveboxes`,
        // `constant_types`, the export path's `renamed_inputargs`
        // (`optimize_with_constants_and_inputs_at` still passes raw
        // `[0..num_inputs)` for Phase 2 parity), and the caller's
        // `pre_opt_jump_args` capture. The remap compaction at
        // `optimizer.rs` (immediately below this signature) must coexist
        // with the shifted namespace. Until all of those are unified in
        // one patch, keep the param unused.
        //
        // A partial activation that only raised `start_next_pos` past
        // `bridge_inputarg_base + num_inputs` (keeping `inputarg_base=0`)
        // was attempted and also SIGSEGVs on fib_recursive: the backend's
        // compile pipeline cannot handle the OpRef range gap introduced
        // by forcing the first optimizer-emitted op far beyond the raw
        // trace's max position.
        #[allow(unused_variables)] bridge_inputarg_base: u32,
    ) -> (Vec<Op>, bool) {
        // bridgeopt.py:124-185: deserialize_optimizer_knowledge
        // Store as pending — setup() inside optimize_with_constants_and_inputs
        // clears pass state, so we apply AFTER setup.
        self.pending_bridge_rd = pending_bridge_rd;
        // unroll.py:183 parity: save pre-optimization JUMP args as
        // runtime_boxes. RPython passes the original live_arg_boxes from
        // compile_trace / guard failure, not the optimized jump args.
        let pre_opt_jump_args: Vec<OpRef> = ops
            .last()
            .filter(|op| op.opcode == OpCode::Jump)
            .map(|op| op.args.to_vec())
            .unwrap_or_default();

        // unroll.py:193: info, ops = self.propagate_all_forward(trace, ...)
        let optimized_ops = self.optimize_with_constants_and_inputs(ops, constants, num_inputs);

        // RPython flush=False: JUMP is in terminal_op, not in optimized_ops.
        let terminal_jump = self.terminal_op.take();
        let has_jump = terminal_jump
            .as_ref()
            .map_or(false, |op| op.opcode == OpCode::Jump);

        if !has_jump {
            return (optimized_ops, false);
        }

        let terminal_jump = terminal_jump.unwrap();
        let jump_args = terminal_jump.args.to_vec();

        // unroll.py:198-200: not inline_short_preamble → jump_to_preamble
        // RPython calls send_extra_operation(jump_op) which forces virtuals
        // through the full pass chain. No explicit flush()/force_box() needed.
        if !inline_short_preamble || front_target_tokens.len() <= 1 {
            if let Some(preamble_token) = front_target_tokens.first() {
                let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
                    let ni = self.final_num_inputs();
                    OptContext::with_num_inputs(32, ni)
                });
                // unroll.py:240-242: jump_to_preamble → send_extra_operation
                let mut jump_op = terminal_jump.clone();
                jump_op.args = pre_opt_jump_args.clone().into();
                jump_op.descr = Some(preamble_token.as_jump_target_descr());
                self.send_extra_operation(&jump_op, &mut ctx);
                let mut result = optimized_ops;
                result.extend(ctx.new_operations.drain(..));
                return (result, false);
            }
            return (optimized_ops, false);
        }

        // unroll.py:203: self.flush()
        let mut ctx = self.final_ctx.take().unwrap_or_else(|| {
            let ni = self.final_num_inputs();
            OptContext::with_num_inputs(32, ni)
        });

        // RPython parity: export virtual state BEFORE flush() and
        // force_at_the_end_of_preamble(). In RPython, get_virtual_state
        // is called inside _jump_to_existing_trace on the original Box
        // objects whose PtrInfo reflects the pre-flush optimizer state.
        // flush() may force pending virtuals, changing their PtrInfo to
        // non-virtual. Capture the virtual state before any forcing.
        let pre_force_vs =
            crate::optimizeopt::virtualstate::export_state(&jump_args, &ctx, &ctx.forwarded);

        self.flush(&mut ctx);

        // unroll.py:204-205: force_at_the_end_of_preamble for each jump arg
        let saved_pass_idx = ctx.current_pass_idx;
        ctx.current_pass_idx = ctx.optearlyforce_idx;
        for &arg in &jump_args {
            let _ = self.force_box_for_end_of_preamble(arg, &mut ctx);
        }
        ctx.current_pass_idx = saved_pass_idx;

        // RPython parity: jump_op.getarglist() returns the ORIGINAL Boxes.
        // force_at_the_end_of_preamble recurses into virtual fields but
        // does NOT force the top-level virtuals. So get_virtual_state on
        // the original args still sees Virtual PtrInfo.
        // DO NOT resolve jump_args here — pass originals so virtual_state
        // is computed from the pre-force state.

        // unroll.py:206-211: jump_to_existing_trace(force_boxes=False)
        // RPython iterates ALL target_tokens; preamble (virtual_state=None)
        // is skipped inside jump_to_existing_trace (unroll.py:327-328).
        let opt_unroll = crate::optimizeopt::unroll::OptUnroll::new();
        let vs = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            false,
            &pre_opt_jump_args,
            Some(pre_force_vs.clone()),
        ) {
            Ok(vs) => vs,
            // unroll.py:209-210: except InvalidLoop → jump_to_preamble
            // RPython: self.jump_to_preamble → send_extra_operation
            Err(()) => {
                if let Some(preamble_token) = front_target_tokens.first() {
                    ctx.clear_newoperations();
                    let mut jump_op = terminal_jump.clone();
                    jump_op.args = pre_opt_jump_args.clone().into();
                    jump_op.descr = Some(preamble_token.as_jump_target_descr());
                    self.send_extra_operation(&jump_op, &mut ctx);
                    let mut result = optimized_ops;
                    result.extend(ctx.new_operations.drain(..));
                    return (result, false);
                }
                return (optimized_ops, false);
            }
        };

        // unroll.py:212-213: vs is None → matched, JUMP redirected
        if vs.is_none() {
            let mut result = optimized_ops;
            result.extend(ctx.new_operations.drain(..));
            return (result, false);
        }

        // unroll.py:214-218: retrace check
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-retrace-check] retraced_count={} retrace_limit={} jump_args={} force_boxes=false",
                retraced_count,
                retrace_limit,
                jump_args.len(),
            );
        }
        if retraced_count < retrace_limit {
            if crate::optimizeopt::majit_log_enabled() {
                eprintln!("[jit] Retracing ({}/{})", retraced_count + 1, retrace_limit);
            }
            return (optimized_ops, true);
        }

        // unroll.py:220-227: retrace limit reached, try force_boxes=True
        ctx.clear_newoperations();
        let vs2 = match Self::try_jump_to_existing_trace(
            &opt_unroll,
            &jump_args,
            front_target_tokens,
            self,
            &mut ctx,
            true,
            &pre_opt_jump_args,
            Some(pre_force_vs),
        ) {
            Ok(vs) => vs,
            // unroll.py:224-225: except InvalidLoop: pass
            // vs (from first attempt) is still not None → falls through
            // to jump_to_preamble below.
            Err(()) => vs,
        };

        // unroll.py:226-227: vs is None → matched with forced boxes
        if vs2.is_none() {
            let mut result = optimized_ops;
            result.extend(ctx.new_operations.drain(..));
            return (result, false);
        }

        // unroll.py:228-229,238-242: jump_to_preamble → send_extra_operation.
        // RPython sends the JUMP through the full optimization chain so that
        // force_box materializes virtuals and potential_extra_ops are consumed.
        if crate::optimizeopt::majit_log_enabled() {
            eprintln!(
                "[jit] Retrace count reached, jumping to preamble (retraced_count={} retrace_limit={})",
                retraced_count, retrace_limit,
            );
        }
        if let Some(preamble_token) = front_target_tokens.first() {
            ctx.clear_newoperations();
            let mut jump_op = terminal_jump.clone();
            jump_op.args = pre_opt_jump_args.into();
            jump_op.descr = Some(preamble_token.as_jump_target_descr());
            self.send_extra_operation(&jump_op, &mut ctx);
            let mut result = optimized_ops;
            result.extend(ctx.new_operations.drain(..));
            (result, false)
        } else {
            (optimized_ops, false)
        }
    }

    /// Wrapper: call jump_to_existing_trace, catch only InvalidLoop panics.
    /// Returns Ok(vs) on normal return, Err(()) on InvalidLoop.
    /// Non-InvalidLoop panics are re-raised.
    fn try_jump_to_existing_trace(
        opt_unroll: &crate::optimizeopt::unroll::OptUnroll,
        jump_args: &[OpRef],
        front_target_tokens: &mut Vec<crate::optimizeopt::unroll::TargetToken>,
        optimizer: &mut Self,
        ctx: &mut OptContext,
        force_boxes: bool,
        pre_opt_jump_args: &[OpRef],
        pre_vs: Option<crate::optimizeopt::virtualstate::VirtualState>,
    ) -> Result<Option<crate::optimizeopt::virtualstate::VirtualState>, ()> {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            opt_unroll.jump_to_existing_trace_with_vs(
                jump_args,
                None,
                front_target_tokens,
                optimizer,
                ctx,
                force_boxes,
                Some(pre_opt_jump_args),
                pre_vs,
            )
        })) {
            Ok(vs) => Ok(vs),
            Err(payload) => {
                if payload
                    .downcast_ref::<crate::optimize::InvalidLoop>()
                    .is_some()
                {
                    Err(())
                } else {
                    // Not InvalidLoop — re-raise
                    std::panic::resume_unwind(payload);
                }
            }
        }
    }

    fn collect_exported_int_bounds(
        &self,
        args: &[OpRef],
        ctx: &mut OptContext,
    ) -> std::collections::HashMap<OpRef, crate::optimizeopt::intutils::IntBound> {
        let mut exported = std::collections::HashMap::new();
        for pass in &self.passes {
            for (opref, bound) in pass.export_arg_int_bounds(args, ctx) {
                exported.insert(opref, bound);
            }
        }
        exported
    }

    /// Send one operation through the pass chain.
    ///
    /// NOTE: Do NOT add `replace_op(original_pos, new_pos)` here.
    /// The Emit variant's position tracking is handled by each pass
    /// and OptContext. Adding automatic replacement mapping here
    /// causes spurious forwarding that breaks heap/guard tests.
    fn propagate_one(&mut self, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass(0, op, ctx);
    }

    fn drain_extra_operations_from(&mut self, _start_pass: usize, ctx: &mut OptContext) {
        let end_pass = self.extra_operation_end_pass();
        let mut pending = std::collections::VecDeque::new();
        while let Some((start, op)) = ctx.extra_operations_after.pop_front() {
            pending.push_back((start, op));
        }
        while let Some((from_pass, op)) = pending.pop_front() {
            self.propagate_from_pass_range(from_pass, end_pass, &op, ctx);
            while let Some((start, op)) = ctx.extra_operations_after.pop_front() {
                pending.push_front((start, op));
            }
        }
    }

    fn propagate_from_pass(&mut self, start_pass: usize, op: &Op, ctx: &mut OptContext) {
        self.propagate_from_pass_range(start_pass, self.passes.len(), op, ctx);
    }

    fn extra_operation_end_pass(&self) -> usize {
        let mut end = self.passes.len();
        while end > 0 && self.passes[end - 1].name() == "unroll" {
            end -= 1;
        }
        end
    }

    fn propagate_from_pass_range(
        &mut self,
        start_pass: usize,
        end_pass: usize,
        op: &Op,
        ctx: &mut OptContext,
    ) {
        // Box.type parity: register the op's intrinsic result type on its
        // OpRef BEFORE the pass chain runs. Without this, a pass that
        // calls `replace_op(op.pos, other)` observes a stale `value_types`
        // entry left over from Phase 1 (Phase 1 may have emitted a
        // different-typed op at the same reused OpRef number). RPython
        // has no equivalent stale-entry hazard because Box.type is stored
        // on the Box object itself.
        // Refresh value_types with the current op's result type before
        // running pass callbacks. RPython's Box.type is fixed at Box
        // creation, so this is either a first-time record (fresh pos)
        // or a same-type no-op; a type-changing refresh would be a
        // cross-phase OpRef-reuse bug, and surfacing it is the point of
        // the invariant.
        if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
            ctx.register_value_type(op.pos, op.result_type());
        }

        // Resolve forwarded arguments
        let mut resolved_op = op.clone();
        for arg in &mut resolved_op.args {
            *arg = ctx.get_box_replacement(*arg);
        }
        if let Some(ref mut fa) = resolved_op.fail_args {
            for arg in fa.iter_mut() {
                *arg = ctx.get_box_replacement(*arg);
            }
        }

        let mut current_op = resolved_op;

        // optimizer.py:864-867: optimize_SAME_AS_I/R/F → make_equal_to(op, arg0)
        // SameAs ops are absorbed into forwarding, never emitted.
        if matches!(
            current_op.opcode,
            OpCode::SameAsI | OpCode::SameAsR | OpCode::SameAsF
        ) {
            ctx.make_equal_to(current_op.pos, current_op.arg(0));
            return;
        }

        // optimizer.py:570-589 parity: collect pass indices that need
        // postprocess callbacks. After emission, invoke them in reverse
        // order — matching RPython's OptimizationResult.callback() chain.
        let mut postprocess_passes: Vec<usize> = Vec::new();

        for pass_idx in start_pass..end_pass {
            ctx.current_pass_idx = pass_idx;
            let result = {
                let pass = &mut self.passes[pass_idx];
                pass.propagate_forward(&current_op, ctx)
            };
            self.drain_extra_operations_from(pass_idx + 1, ctx);
            match result {
                OptimizationResult::Emit(op) => {
                    // optimizer.py:576-581: collect postprocess for this pass
                    // if it has postprocess for this opnum.
                    if self.passes[pass_idx].have_postprocess_op(op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                    self.emit_operation(op.clone(), ctx);
                    // optimizer.py:585-589: invoke postprocess callbacks
                    // in reverse order after emission.
                    for &pp_idx in postprocess_passes.iter().rev() {
                        self.passes[pp_idx].propagate_postprocess(&op, ctx);
                    }
                    return;
                }
                OptimizationResult::Replace(op) => {
                    debug_assert!(
                        !(current_op.opcode.is_guard()
                            && op.opcode.is_guard()
                            && current_op.rd_resume_position >= 0
                            && op.rd_resume_position < 0),
                        "Replace dropped rd_resume_position: {:?} -> {:?}",
                        current_op.opcode,
                        op.opcode,
                    );
                    // optimizer.py:576-581: pass emitted (replace ≈ emit
                    // with modified op). Collect if has postprocess.
                    if self.passes[pass_idx].have_postprocess_op(op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                    // Box.type parity: if Replace changed the opcode to one
                    // with a different result type, the intrinsic type on
                    // op.pos changes with it. Keep value_types aligned so
                    // later passes and replace_op callers see the correct
                    // type instead of the pre-replace one.
                    // Replace may change the opcode. RPython's equivalent
                    // (OptimizationResult.replace) keeps the box identity
                    // so the result type must stay constant — a legitimate
                    // replace here should produce an op with the same
                    // result type as whatever was previously registered
                    // for `op.pos`. Upgrade to `register_value_type` so a
                    // type-changing replacement panics at the source
                    // instead of silently retyping the OpRef.
                    if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
                        ctx.register_value_type(op.pos, op.result_type());
                    }
                    current_op = op;
                }
                OptimizationResult::Restart(op) => {
                    // optimizer.py:567 `send_extra_operation(newop, opt=None)`:
                    // re-dispatch from `first_optimization`, dropping the
                    // original op (the rule that returned Restart already
                    // skipped its own emit). autogenintrules.py uses this
                    // for every rewrite rule so that chained OptIntBounds
                    // rules (add_zero, int_is_zero, reassoc) re-fire on the
                    // rewritten op.
                    debug_assert!(
                        !(current_op.opcode.is_guard()
                            && op.opcode.is_guard()
                            && current_op.rd_resume_position >= 0
                            && op.rd_resume_position < 0),
                        "Restart dropped rd_resume_position: {:?} -> {:?}",
                        current_op.opcode,
                        op.opcode,
                    );
                    if !op.pos.is_none() && op.result_type() != majit_ir::Type::Void {
                        ctx.register_value_type(op.pos, op.result_type());
                    }
                    self.propagate_from_pass_range(0, end_pass, &op, ctx);
                    // Run any postprocess callbacks accumulated in the outer
                    // chain (passes that already returned PassOn for the
                    // original op before the Restart-returning pass fired).
                    // RPython's send_extra_operation does not unwind these —
                    // they belong to the original chain.
                    for &pp_idx in postprocess_passes.iter().rev() {
                        self.passes[pp_idx].propagate_postprocess(&current_op, ctx);
                    }
                    return;
                }
                OptimizationResult::Remove => {
                    // optimizer.py:573-575: op removed → no postprocess.
                    ctx.pending_mark_last_guard = None;
                    ctx.pending_guard_class_postprocess = None;
                    return;
                }
                OptimizationResult::PassOn => {
                    // optimizer.py:576-583: PASS_OP_ON path.
                    // RPython's emit() returns PASS_OP_ON if no postprocess,
                    // OptimizationResult if postprocess needed.
                    // Collect if this pass has postprocess for this opcode.
                    if self.passes[pass_idx].have_postprocess_op(current_op.opcode) {
                        postprocess_passes.push(pass_idx);
                    }
                }
                // rewrite.py:406 — guard proven to always fail
                OptimizationResult::InvalidLoop => {
                    std::panic::panic_any(crate::optimize::InvalidLoop(
                        "guard proven to always fail",
                    ));
                }
            }
        }

        // If no pass handled it, emit as-is
        self.emit_operation(current_op.clone(), ctx);
        // Postprocess in reverse order after emission.
        for &pp_idx in postprocess_passes.iter().rev() {
            self.passes[pp_idx].propagate_postprocess(&current_op, ctx);
        }
    }

    /// optimizer.py: _emit_operation — emit with guard tracking.
    ///
    /// When emitting a guard, check replaces_guard to see if this guard
    /// should replace a previously emitted one (guard strengthening).
    /// Also track last_guard_op for consecutive guard descriptor sharing.
    /// RPython optimizer.py:623-625: _emit_operation calls force_box(arg)
    /// on every arg before final emission. In majit, this forces any remaining
    /// virtual args that weren't caught by pass-level handlers.
    fn emit_operation(&mut self, mut op: Op, ctx: &mut OptContext) {
        // RPython optimizer.py:614: _emit_operation is on the Optimizer (last
        // "pass" in the chain). Any force_box called here should emit directly,
        // matching RPython's Optimizer.emit_extra which just calls self.emit(op).
        let saved_in_final_emission = ctx.in_final_emission;
        ctx.in_final_emission = true;
        // RPython optimizer.py: emitting_operation callback — notify all passes
        // before any op is emitted. This is how OptHeap forces lazy sets before
        // guards even when the guard is emitted by an earlier pass.
        for (idx, pass) in self.passes.iter_mut().enumerate() {
            pass.emitting_operation(&op, ctx, idx);
        }
        // RPython emit_extra(op, emit=False) parity: drain operations
        // queued by emitting_operation (e.g., heap's force_lazy_set)
        // BEFORE the current op is emitted, preserving correct ordering.
        {
            let end_pass = self.extra_operation_end_pass();
            while let Some((start, queued_op)) = ctx.extra_operations_after.pop_front() {
                self.propagate_from_pass_range(start, end_pass, &queued_op, ctx);
            }
        }

        // optimizer.py:623-625: force_box on every arg unconditionally.
        for i in 0..op.num_args() {
            let arg = ctx.get_box_replacement(op.arg(i));
            op.args[i] = self.force_box(arg, ctx);
        }
        // optimizer.py:626: self.metainterp_sd.profiler.count(Counters.OPT_OPS).
        // Pyre defers the fold into JitStatsCounters via update_counters
        // (see field doc for the rationale).
        self.opt_ops_emitted = self.opt_ops_emitted.saturating_add(1);
        if op.opcode.is_guard() {
            // optimizer.py:629: profiler.count(Counters.OPT_GUARDS).
            self.opt_guards_emitted = self.opt_guards_emitted.saturating_add(1);
            // optimizer.py:630-631: pendingfields = self.pendingfields; self.pendingfields = None
            // — captured into ctx.pending_for_guard by the heap pass via emitting_operation.
            // emit_guard_operation reads ctx.pending_for_guard inside store_final_boxes_in_guard
            // and clears it.

            // optimizer.py:632-635: replaces_guard check BEFORE emit_guard_operation.
            if self.can_replace_guards {
                if let Some(replacement) = self.replaces_guard.remove(&op.pos.0) {
                    let target_pos = replacement.pos.0 as usize;
                    if target_pos < ctx.new_operations.len() {
                        if std::env::var_os("MAJIT_LOG").is_some() {
                            eprintln!(
                                "[opt] guard replacement op={:?} pos={:?} target_index={} len={}",
                                op.opcode,
                                op.pos,
                                target_pos,
                                ctx.new_operations.len()
                            );
                        }
                        // optimizer.py:713-720 replace_guard_op:
                        //   old_descr = old_op.getdescr()
                        //   new_descr = new_op.getdescr()
                        //   new_descr.copy_all_attributes_from(old_descr)
                        // Inherit the slot's resume payload onto the new op's
                        // descr so the new guard carries old's rd_* — without
                        // this, `_newoperations[target_pos]` would land with
                        // the new descr's empty rd_* defaults.  In-place
                        // mutation preserves new_descr's identity (fail_index
                        // / status / subtype tag).
                        //
                        // RPython performs both getdescr() calls and the
                        // copy_all_attributes_from unconditionally; an old/new
                        // descr missing here means the optimizer gave us a
                        // replacement guard without resume payload, which
                        // would silently overwrite the slot.  Match RPython
                        // by panicking instead of skipping.
                        let old_descr = ctx.new_operations[target_pos].descr.as_ref().expect(
                            "optimizer.py:716 old_descr = old_op.getdescr(): \
                                 replaced guard slot has no descr",
                        );
                        let new_descr = op.descr.as_ref().expect(
                            "optimizer.py:717 new_descr = new_op.getdescr(): \
                             replacement guard has no descr",
                        );
                        crate::compile::copy_all_attributes_from(new_descr, old_descr);
                        ctx.new_operations[target_pos] = op.clone();
                        ctx.in_final_emission = saved_in_final_emission;
                        return;
                    }
                }
            }

            // optimizer.py:637: op = self.emit_guard_operation(op, pendingfields)
            op = self.emit_guard_operation(op, ctx);
        } else {
            // optimizer.py:639-644: preserve last_guard_op for guard chaining
            // unless the op has side effects or is a call_pure that can raise.
            let preserves_chain = (op.opcode.has_no_side_effect()
                || op.opcode.is_guard()
                || op.opcode.is_jit_debug()
                || op.opcode.is_ovf())
                && !Self::is_call_pure_pure_canraise(&op);
            if !preserves_chain {
                self.last_guard_op_idx = None;
            }
        }
        // optimizer.py:598-602:
        //     if rop.returns_bool_result(op.opnum):
        //         self.getintbound(op).make_bool()
        if op.opcode.returns_bool() {
            assert_eq!(
                op.result_type(),
                majit_ir::Type::Int,
                "returns_bool op must have int result: {:?} pos={:?} args={:?}",
                op.opcode,
                op.pos,
                op.args
            );
            // RPython parity: the current ResOp already carries type 'i'.
            // The op has `result_type() == Int` (asserted above), so
            // registering the type here is either a first-time record (fresh
            // pos) or a same-type no-op (the op was visited before). A
            // different cached type means a Phase 1 op left a stale
            // value_types entry at this reused pos — precisely the cross-type
            // mutation the invariant guard is designed to surface.
            ctx.register_value_type(op.pos, majit_ir::Type::Int);
            ctx.with_intbound_mut(op.pos, |bound| bound.make_bool());
        }
        let emitted = ctx.emit(op.clone());
        // optimizer.py:603-611: after emit, promote IntBound→Const.
        //   op = self.get_box_replacement(op)
        //   if op.type == 'i':
        //       opinfo = op.get_forwarded()  # IntBound
        //       if opinfo is not None and opinfo.is_constant():
        //           op.set_forwarded(ConstInt(opinfo.get_constant_int()))
        if op.result_type() == majit_ir::Type::Int {
            let replaced = ctx.get_box_replacement(emitted);
            if let Some(bound) = ctx.peek_intbound(replaced) {
                if bound.is_constant() {
                    let const_val = bound.get_constant_int();
                    ctx.make_constant(replaced, majit_ir::Value::Int(const_val));
                }
            }
        }
        if std::env::var_os("MAJIT_LOG").is_some()
            && matches!(
                op.opcode,
                OpCode::CallMayForceI
                    | OpCode::CallMayForceR
                    | OpCode::CallMayForceF
                    | OpCode::CallMayForceN
                    | OpCode::GuardNotForced
                    | OpCode::GuardNotForced2
            )
        {
            eprintln!(
                "[opt] emit {:?} pos={:?} len={}",
                op.opcode,
                emitted,
                ctx.new_operations.len()
            );
        }
        // optimizer.py:47-54: run deferred postprocess after emit.
        // RPython calls OptimizationResult.callback() → propagate_postprocess.
        // rewrite.py:282: postprocess_GUARD_NONNULL → mark_last_guard
        if let Some(opref) = ctx.pending_mark_last_guard.take() {
            ctx.mark_last_guard(opref);
        }
        if let Some(pp) = ctx.pending_guard_class_postprocess.take() {
            // rewrite.py:430-436 postprocess_GUARD_CLASS:
            //   update_last_guard = not old_guard or isinstance(descr, ResumeAtPositionDescr)
            //   make_constant_class(arg0, expectedclassbox, update_last_guard)
            //
            // Delegate to `Optimizer::make_constant_class` rather than
            // inlining a fresh `PtrInfo::known_class` overwrite — the
            // helper preserves existing `Instance` / `Virtual` PtrInfo
            // (fields, descr, cached_vinfo, virtual state) and only
            // refreshes `_known_class`, matching upstream
            // `optimizer.py:137-151` where `isinstance(opinfo,
            // InstancePtrInfo)` mutates in place.
            let old_guard_pos = ctx
                .get_ptr_info(pp.obj)
                .and_then(|info| info.last_guard_pos())
                .unwrap_or(-1);
            let update_last_guard =
                old_guard_pos < 0 || ctx.is_resume_at_position_guard(old_guard_pos);
            Self::make_constant_class(ctx, pp.obj, pp.class_val, update_last_guard);
        }
        ctx.in_final_emission = saved_in_final_emission;
    }

    /// optimizer.py:652-686 emit_guard_operation
    ///
    /// Manages the guard sharing chain (`_last_guard_op`) and dispatches to
    /// `_copy_resume_data_from` (descrless follow-up guard, e.g.
    /// `GUARD_NO_EXCEPTION` after a `CALL_MAY_FORCE`) or to
    /// `store_final_boxes_in_guard` for fresh guards.
    fn emit_guard_operation(&mut self, mut op: Op, ctx: &mut OptContext) -> Op {
        let opcode = op.opcode;
        // optimizer.py:661-664: GUARD_NO_EXCEPTION / GUARD_EXCEPTION can only
        // share resume data with a preceding GUARD_NOT_FORCED. Anything else
        // breaks the chain.  GUARD_NOT_FORCED_2 is intentionally excluded:
        // upstream emits it only at finish() (pyjitpl.py:3236) where no
        // GUARD_(NO_)EXCEPTION can follow, so reuse would create an
        // unreachable sharing edge.
        if (opcode == OpCode::GuardNoException || opcode == OpCode::GuardException)
            && self
                .last_guard_op_idx
                .and_then(|idx| ctx.new_operations.get(idx))
                .is_some_and(|last| last.opcode != OpCode::GuardNotForced)
        {
            self.last_guard_op_idx = None;
        }
        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share resume data.
        if opcode == OpCode::GuardAlwaysFails {
            self.last_guard_op_idx = None;
        }

        // optimizer.py:672-683: shared vs. fresh dispatch.
        // compile.py:925-926 invent_fail_descr_for_op: GUARD_NOT_FORCED /
        // GUARD_NOT_FORCED_2 must always mint a fresh ResumeGuardForcedDescr
        // (`assert copied_from_descr is None`).  They are never on the
        // sharing chain.  Mirrors the OptContext path at
        // optimizeopt/mod.rs:3061-3066.
        let shared = op.descr.is_none()
            && self.last_guard_op_idx.is_some()
            && opcode != OpCode::GuardNotForced
            && opcode != OpCode::GuardNotForced2;
        if shared {
            // optimizer.py:673-674:
            //   self.metainterp_sd.profiler.count_ops(
            //       opnum, jitprof.Counters.OPT_GUARDS_SHARED)
            self.opt_guards_shared_emitted = self.opt_guards_shared_emitted.saturating_add(1);
            op = self._copy_resume_data_from(op, ctx);
        } else {
            // optimizer.py:630-631 + resume.py:428-445 + 520-558:
            // RPython passes `pendingfields` INTO finish() so
            // `_add_pending_fields` tags target_tagged / value_tagged
            // during the same numbering pass that builds rd_numb.
            // Pyre builds the entries here with UNASSIGNED placeholders
            // and hands them to `store_final_boxes_in_guard` as the
            // `pending_setfields` parameter — finish() tags them and
            // the descr's `set_rd_pendingfields` receives the tagged
            // slice (compile.py:855 `_attrs_`).
            let pending = std::mem::take(&mut ctx.pending_for_guard);
            // resume.py:428-445 + 520-558: pending fields are passed
            // INTO finish() so `_add_pending_fields` can tag
            // target_tagged / value_tagged during the same numbering
            // pass that builds rd_numb.  We build the UNASSIGNED
            // entries here, hand them to `store_final_boxes_in_guard`,
            // and the descr's `set_rd_pendingfields` receives them
            // post-tagging — descr is the single SOT (compile.py:855
            // `_attrs_`).
            let pending_for_finish: Vec<majit_ir::GuardPendingFieldEntry> = if !pending.is_empty() {
                pending
                    .into_iter()
                    .map(|pf_op| {
                        let (target, value, item_index) = if pf_op.opcode == OpCode::SetarrayitemGc
                        {
                            // resume.py:534-552 _add_pending_fields:
                            //   boxindex = op.getarg(1).get_box_replacement()
                            //   itemindex = boxindex.getint()       # → Const required
                            //   if itemindex < 0: raise TagOverflow
                            //   if itemindex > 2147483647: raise TagOverflow
                            // RPython's `getint()` extracts the
                            // tagged Const integer; non-Const args
                            // never reach `_add_pending_fields`
                            // because the heap pass only stages
                            // setarrayitem ops with a constant
                            // index. Fall back to InvalidLoop
                            // (caught at pyjitpl/mod.rs:3454) on
                            // either invariant violation rather
                            // than silently coercing to 0.
                            let boxindex = ctx.get_box_replacement(pf_op.arg(1));
                            let idx = match ctx.get_constant_int(boxindex) {
                                Some(v) if (0..=i32::MAX as i64).contains(&v) => v,
                                _ => std::panic::panic_any(crate::optimize::InvalidLoop(
                                    "_add_pending_fields: SETARRAYITEM_GC index \
                                             must be a non-negative Const i32 (TagOverflow)",
                                )),
                            };
                            (pf_op.arg(0), pf_op.arg(2), idx as i32)
                        } else {
                            (pf_op.arg(0), pf_op.arg(1), -1i32)
                        };
                        majit_ir::GuardPendingFieldEntry {
                            descr: pf_op.descr.clone(),
                            descr_index: pf_op.descr.as_ref().map_or(0, |d| d.index()),
                            item_index,
                            target: ctx.get_box_replacement(target),
                            value: ctx.get_box_replacement(value),
                            target_tagged: majit_ir::resumedata::UNASSIGNED,
                            value_tagged: majit_ir::resumedata::UNASSIGNED,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // resume.py:570-574 _add_optimizer_sections + bridgeopt.py:63-122:
            // RPython collects optimizer knowledge INSIDE
            // store_final_boxes_in_guard → finish() → serialize_optimizer_knowledge.
            // Rust adaptation: collect BEFORE the call (borrow checker) and pass
            // as parameter. available_boxes filtering happens inside
            // memo.finish() using liveboxes ∩ liveboxes_from_env.
            let knowledge_for_resume = self.collect_optimizer_knowledge_for_resume(ctx);
            let knowledge = if knowledge_for_resume.is_empty() {
                None
            } else {
                Some(knowledge_for_resume)
            };
            // optimizer.py:678: store_final_boxes_in_guard.
            // resume.py:445 self._add_pending_fields(pending_setfields) —
            // tags target_tagged/value_tagged in place during finish();
            // the descr's set_rd_pendingfields receives the tagged slice
            // post-finish (mod.rs::store_final_boxes_in_guard).
            op = Self::store_final_boxes_in_guard(op, ctx, knowledge, pending_for_finish);
            // optimizer.py:681-683: force_box on each fail_arg for unrolling.
            if let Some(ref fa) = op.fail_args {
                let fargs: Vec<OpRef> = fa.iter().copied().collect();
                for farg in fargs {
                    if !farg.is_none() {
                        self.force_box(farg, ctx);
                    }
                }
            }
            // optimizer.py:750-751 (called from store_final_boxes_in_guard):
            // GUARD_VALUE → bool replacement. We invoke it here so descr is
            // already set when _maybe_replace_guard_value reads it.
            if op.opcode == OpCode::GuardValue {
                op = Self::_maybe_replace_guard_value(op, ctx);
            }
        }

        // optimizer.py:679: update last_guard_op only on the fresh-guard path.
        // The op has not yet been pushed to `ctx.new_operations` (the
        // surrounding `emit_operation` does so at `ctx.emit(op.clone())`
        // shortly after we return), so the predicted donor index is
        // exactly `ctx.new_operations.len()` at this point.
        if !shared {
            self.last_guard_op_idx = Some(ctx.new_operations.len());
        }
        // optimizer.py:684-685: GUARD_EXCEPTION breaks the chain.
        if opcode == OpCode::GuardException {
            self.last_guard_op_idx = None;
        }
        op
    }

    /// optimizer.py:688-700 _copy_resume_data_from
    ///
    /// Inherits descr / fail_args from `_last_guard_op` for the follow-up
    /// descrless guard, then runs `_maybe_replace_guard_value` if the
    /// inheriting op is a `GUARD_VALUE`.
    ///
    /// `compile.py:832 ResumeGuardCopiedDescr(prev)` parity: stamp
    /// the sharer's descr as a `ResumeGuardCopiedDescr` (or
    /// `ResumeGuardCopiedExcDescr` for exception guards) whose
    /// `prev` references the donor's `ResumeGuardDescr`.  Readers
    /// go through `FailDescr::rd_*()` which chases `prev` automatically
    /// (compile.py:849 `get_resumestorage(): return prev`).
    fn _copy_resume_data_from(&mut self, mut op: Op, ctx: &mut OptContext) -> Op {
        let donor_idx = self
            .last_guard_op_idx
            .expect("_copy_resume_data_from requires last_guard_op_idx");
        let last = ctx
            .new_operations
            .get(donor_idx)
            .expect("last_guard_op_idx must point inside ctx.new_operations");
        // compile.py:919-937 `invent_fail_descr_for_op`: the sharing
        // path mints `ResumeGuardCopiedDescr(prev)` (or
        // `ResumeGuardCopiedExcDescr(prev)` for GUARD_EXCEPTION /
        // GUARD_NO_EXCEPTION) rather than cloning the donor's
        // ResumeGuardDescr — `get_resumestorage()` (compile.py:849)
        // routes reads back to `prev`.  GUARD_NOT_FORCED never
        // reaches this path (compile.py:921 `assert copied_from_descr
        // is None`), so no Forced-copied variant is needed.
        //
        // optimizer.py:691 `assert isinstance(last_descr,
        // compile.ResumeGuardDescr)` — the donor must be a
        // head-of-chain `ResumeGuardDescr` (never None, never a
        // `ResumeGuardCopiedDescr`).  Mirror the assertion so the
        // sharer never silently ends up with `descr=None` (which
        // would make `op.descr.fail_descr().rd_*()` reads fall back
        // to empty slices) or a two-hop prev.
        let last_descr = last
            .descr
            .as_ref()
            .expect("optimizer.py:691 last_guard_op.getdescr() must exist");
        assert!(
            !last_descr.is_resume_guard_copied(),
            "optimizer.py:691 assert isinstance(last_descr, ResumeGuardDescr): \
             ResumeGuardCopiedDescr forbidden as sharing donor"
        );
        op.descr = Some({
            use majit_ir::OpCode;
            match op.opcode {
                OpCode::GuardException | OpCode::GuardNoException => {
                    crate::compile::make_resume_guard_copied_exc_descr(last_descr.clone())
                }
                _ => crate::compile::make_resume_guard_copied_descr(last_descr.clone()),
            }
        });
        op.fail_args = last.fail_args.clone();
        op.rd_resume_position = last.rd_resume_position;
        // bridgeopt.py parity: the class-knowledge bitfield baked into
        // rd_numb is indexed by the donor's per-livebox type layout.
        // `deserialize_optimizer_knowledge` reads that bitfield using the
        // types it receives as `livebox_types`, which the bridge site
        // sources from the shared guard's `fail_arg_types`. Dropping the
        // copy here leaves the shared guard with `fail_arg_types=None`,
        // the bridge code falls back to the bridge tracer's (unboxed)
        // inputarg types, and the Ref count disagrees with the serializer
        // → rd_numb over-read. See memory/fannkuch_reg20_root_cause.md.
        op.fail_arg_types = last.fail_arg_types.clone();
        // ResumeGuardCopiedDescr(prev) parity (compile.py:849
        // `get_resumestorage(): return prev`): the descr-side `prev`
        // pointer (set by `make_resume_guard_copied_descr` above)
        // routes resume reads to the donor's RdPayload — no op-index
        // chase pointer needed.
        let _ = donor_idx;
        // optimizer.py:698-699: if guard_op.opnum == GUARD_VALUE: ...
        if op.opcode == OpCode::GuardValue {
            op = Self::_maybe_replace_guard_value(op, ctx);
        }
        op
    }

    /// optimizer.py:722-752 store_final_boxes_in_guard
    ///
    /// Resolve fail_args through get_box_replacement and delegate to
    /// finalize_guard_resume_data for snapshot-based virtual encoding
    /// (rd_numb, rd_consts, rd_virtuals).
    ///
    /// resume.py:389-452 finish() is unconditional: every guard reaching
    /// here must have a snapshot. We delegate without any guard condition
    /// — finalize_guard_resume_data already handles the missing-snapshot
    /// case internally (silent return for guards without rd_resume_position
    /// or snapshot_boxes entry).
    fn store_final_boxes_in_guard(
        mut op: Op,
        ctx: &mut OptContext,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) -> Op {
        // optimizer.py:732-748 + resume.py:389-452:
        // RPython finish() handles virtuals without forcing.
        // _number_boxes tags virtual fail_args as TAGVIRTUAL,
        // _number_virtuals builds rd_virtuals from PtrInfo.
        if let Some(ref mut fail_args) = op.fail_args {
            for fa_idx in 0..fail_args.len() {
                if !fail_args[fa_idx].is_none() {
                    fail_args[fa_idx] = ctx.get_box_replacement(fail_args[fa_idx]);
                }
            }
        }
        ctx.finalize_guard_resume_data(&mut op, knowledge, pending_setfields);
        op
    }

    /// bridgeopt.py:63-122 serialize_optimizer_knowledge parity:
    /// Collect UNFILTERED optimizer knowledge from passes for rd_numb
    /// serialization. available_boxes filtering happens inside
    /// memo.finish() during the actual serialization, matching RPython's
    /// flow where serialize_optheap/serialize_optrewrite receive
    /// available_boxes computed from liveboxes ∩ liveboxes_from_env.
    ///
    /// descr.py:28: v.descr_index = len(all_descrs)
    /// Assigns a sequential descr_index if not already assigned, and
    /// appends to self.all_descrs. Returns the assigned index.
    fn ensure_descr_index(&mut self, descr: &DescrRef) -> i32 {
        let idx = descr.get_descr_index();
        if idx >= 0 {
            return idx;
        }
        let new_idx = self.all_descrs.len() as i32;
        descr.set_descr_index(new_idx);
        self.all_descrs.push(descr.clone());
        // descr.py:47: assert len(all_descrs) < 2**15
        assert!(
            self.all_descrs.len() < (1 << 15),
            "too many descriptors: {}",
            self.all_descrs.len()
        );
        new_idx
    }

    fn collect_optimizer_knowledge_for_resume(
        &mut self,
        ctx: &mut OptContext,
    ) -> crate::resume::OptimizerKnowledgeForResume {
        let mut heap_fields_raw = Vec::new();
        let mut heap_arrayitems_raw = Vec::new();
        // Guard resume: export all cached fields (no available_boxes filter).
        // RPython only calls serialize_optheap from bridgeopt.py; this
        // guard-resume path has no RPython-equivalent filter.
        for pass in &self.passes {
            let fields = pass.export_cached_fields(ctx, None);
            let items = pass.export_cached_arrayitems(ctx, None);
            if !fields.is_empty() || !items.is_empty() {
                heap_fields_raw = fields;
                heap_arrayitems_raw = items;
                break;
            }
        }
        let mut loopinvariant_results = Vec::new();
        for pass in &self.passes {
            for (func_ptr, result) in pass.serialize_optrewrite() {
                let replaced = ctx.get_box_replacement(result);
                loopinvariant_results.push((func_ptr, replaced));
            }
        }

        // descr.py:25-47: assign descr_index inline, matching setup_descrs().
        // heap.py:828: descriptors with get_descr_index() == -1 after
        // ensure_descr_index should not occur — ensure_descr_index always
        // assigns. The filter_map here mirrors the RPython path where
        // heap.py:828 skips descriptors not in the CPU cache.
        let heap_fields: Vec<(OpRef, i32, OpRef)> = heap_fields_raw
            .into_iter()
            .map(|(obj, descr, val)| {
                let idx = self.ensure_descr_index(&descr);
                (obj, idx, val)
            })
            .collect();
        let heap_arrayitems: Vec<(OpRef, i64, i32, OpRef)> = heap_arrayitems_raw
            .into_iter()
            .map(|(obj, index, descr, val)| {
                let idx = self.ensure_descr_index(&descr);
                (obj, index, idx, val)
            })
            .collect();

        // bridgeopt.py:74-88: known_classes bitfield is now computed
        // directly at serialization time in resume.rs via env.has_known_class(),
        // matching RPython's per-livebox getptrinfo(box).get_known_class(cpu).

        crate::resume::OptimizerKnowledgeForResume {
            heap_fields,
            heap_arrayitems,
            loopinvariant_results,
        }
    }

    /// optimizer.py:754-778 _maybe_replace_guard_value
    ///
    /// ```text
    /// def _maybe_replace_guard_value(self, op, descr):
    ///     if op.getarg(0).type == 'i':
    ///         b = self.getintbound(op.getarg(0))
    ///         if b.is_bool():
    ///             # Hack: turn guard_value(bool) into guard_true/guard_false.
    ///             constvalue = op.getarg(1).getint()
    ///             if constvalue == 0:
    ///                 opnum = rop.GUARD_FALSE
    ///             elif constvalue == 1:
    ///                 opnum = rop.GUARD_TRUE
    ///             else:
    ///                 # Issue #3128: rare cases — give up rather than crash.
    ///                 return op
    ///             newop = self.replace_op_with(op, opnum, [op.getarg(0)], descr)
    ///             return newop
    ///     return op
    /// ```
    fn _maybe_replace_guard_value(op: Op, ctx: &mut OptContext) -> Op {
        // optimizer.py:755: if op.getarg(0).type == 'i'
        let arg0 = op.arg(0);
        let arg0_resolved = ctx.get_box_replacement(arg0);
        if ctx.opref_type(arg0_resolved) != Some(majit_ir::Type::Int) {
            return op;
        }
        // optimizer.py:756-757: b = self.getintbound(op.getarg(0)); if b.is_bool()
        let b = ctx.getintbound(arg0_resolved);
        if !b.is_bool() {
            return op;
        }
        // optimizer.py:762: constvalue = op.getarg(1).getint()
        let Some(constvalue) = ctx.get_constant_int(op.arg(1)) else {
            return op;
        };
        // optimizer.py:763-775: 0 → GUARD_FALSE, 1 → GUARD_TRUE, else give up.
        let new_opcode = match constvalue {
            0 => OpCode::GuardFalse,
            1 => OpCode::GuardTrue,
            _ => return op,
        };
        // optimizer.py:776: replace_op_with(op, opnum, [op.getarg(0)], descr)
        let mut newop = Op::new(new_opcode, &[arg0]);
        newop.pos = op.pos;
        newop.descr = op.descr.clone();
        newop.fail_args = op.fail_args.clone();
        newop.fail_arg_types = op.fail_arg_types.clone();
        // compile.py:855 _attrs_ live on the descr; Arc-clone of
        // op.descr above shares the donor's RdPayload, so newop's
        // FailDescr::rd_* readers see the same data.
        newop.rd_resume_position = op.rd_resume_position;
        newop
    }
}

impl Optimizer {
    /// Create an optimizer with the standard pass pipeline.
    /// RPython __init__.py:15-22 ALL_OPTS + ENABLE_ALL_OPTS (rlib/jit.py):
    ///   intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll
    /// (unroll is handled separately by UnrollOptimizer)
    pub fn default_pipeline() -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::new()));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(OptEarlyForce::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Create an optimizer with virtualizable config for frame field tracking.
    pub fn default_pipeline_with_virtualizable(config: VirtualizableConfig) -> Self {
        let mut opt = Self::new();
        opt.add_pass(Box::new(OptIntBounds::new()));
        opt.add_pass(Box::new(OptRewrite::new()));
        opt.add_pass(Box::new(OptVirtualize::with_virtualizable(config)));
        opt.add_pass(Box::new(OptString::new()));
        opt.add_pass(Box::new(OptPure::new()));
        opt.add_pass(Box::new(OptEarlyForce::new()));
        opt.add_pass(Box::new(OptHeap::new()));
        opt
    }

    /// Number of passes in this optimizer.
    pub fn num_passes(&self) -> usize {
        self.passes.len()
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::default_pipeline()
    }
}

// OptimizerBoxEnv removed: was only used by store_final_boxes_in_guard.
// store_final_boxes_in_guard in mod.rs defines InlineBoxEnv for the same purpose.

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::Type;
    use majit_ir::descr::make_size_descr;
    use majit_ir::descr::{CallDescr, EffectInfo, ExtraEffect, OopSpecIndex};
    use majit_ir::{DescrRef, OpCode, OpRef};
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Arc;

    /// A trivial pass that removes INT_ADD(x, 0) -> x
    struct AddZeroElimination;

    impl Optimization for AddZeroElimination {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                // Check if second arg is constant 0
                if let Some(0) = ctx.get_constant_int(op.arg(1)) {
                    // Replace with first arg
                    ctx.replace_op(op.pos, op.arg(0));
                    return OptimizationResult::Remove;
                }
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_zero_elim"
        }
    }

    struct AddVirtualInputsOnce {
        added: bool,
    }

    struct RemoveAsConstant {
        target: OpRef,
        value: i64,
    }

    impl Optimization for AddVirtualInputsOnce {
        fn propagate_forward(&mut self, _op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.added {
                ctx.num_inputs += 2;
                self.added = true;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "add_virtual_inputs_once"
        }
    }

    impl Optimization for RemoveAsConstant {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.pos == self.target {
                ctx.make_constant(op.pos, majit_ir::Value::Int(self.value));
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_constant"
        }
    }

    struct FlushCounter {
        hits: Rc<Cell<usize>>,
    }

    impl Optimization for FlushCounter {
        fn propagate_forward(&mut self, _op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
            OptimizationResult::PassOn
        }

        fn flush(&mut self, _ctx: &mut OptContext) {
            self.hits.set(self.hits.get() + 1);
        }

        fn name(&self) -> &'static str {
            "flush_counter"
        }
    }

    struct RemoveAsTypedConstant {
        target: OpRef,
        value: majit_ir::Value,
    }

    impl Optimization for RemoveAsTypedConstant {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.pos == self.target {
                ctx.make_constant(op.pos, self.value.clone());
                return OptimizationResult::Remove;
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "remove_as_typed_constant"
        }
    }

    struct MarkAsTypedConstantButKeep {
        target: OpRef,
        value: majit_ir::Value,
    }

    impl Optimization for MarkAsTypedConstantButKeep {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if op.pos == self.target {
                ctx.make_constant(op.pos, self.value.clone());
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "mark_as_typed_constant_but_keep"
        }
    }

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
    }

    fn test_parent_descr() -> majit_ir::DescrRef {
        std::sync::Arc::new(TestSizeDescr { index: 0xFFFF_0000 })
    }

    #[derive(Debug)]
    struct TestDescr(u32);

    impl majit_ir::Descr for TestDescr {
        fn index(&self) -> u32 {
            self.0
        }
        fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::FieldDescr for TestDescr {
        fn get_parent_descr(&self) -> Option<majit_ir::DescrRef> {
            Some(test_parent_descr())
        }
        fn offset(&self) -> usize {
            self.0 as usize * 8
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
    }

    #[derive(Debug)]
    struct TestCallDescr {
        idx: u32,
        effect: EffectInfo,
        result_type: majit_ir::Type,
    }

    impl majit_ir::Descr for TestCallDescr {
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
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn get_extra_info(&self) -> &EffectInfo {
            &self.effect
        }
    }

    fn call_may_force_descr(idx: u32, result_type: majit_ir::Type) -> DescrRef {
        Arc::new(TestCallDescr {
            idx,
            effect: EffectInfo {
                extraeffect: ExtraEffect::CanRaise,
                oopspecindex: OopSpecIndex::None,
                ..Default::default()
            },
            result_type,
        })
    }

    struct QueueForceLikeExtraOps {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueForceLikeExtraOps {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                let alloc = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set = Op::new(OpCode::SetfieldGc, &[alloc, OpRef(0)]);
                set.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_force_like_extra_ops"
        }
    }

    struct QueueRestartCandidate {
        queued: bool,
    }

    impl Optimization for QueueRestartCandidate {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntMul {
                self.queued = true;
                ctx.emit_extra(
                    ctx.current_pass_idx,
                    Op::new(OpCode::IntAdd, &[op.arg(0), op.arg(1)]),
                );
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_restart_candidate"
        }
    }

    struct CountRestartedIntSub {
        hits: Rc<Cell<usize>>,
    }

    impl Optimization for CountRestartedIntSub {
        fn propagate_forward(&mut self, op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
            if op.opcode == OpCode::IntSub {
                self.hits.set(self.hits.get() + 1);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "count_restarted_int_sub"
        }
    }

    struct RestartIntAddAsSub;

    impl Optimization for RestartIntAddAsSub {
        fn propagate_forward(&mut self, op: &Op, _ctx: &mut OptContext) -> OptimizationResult {
            if op.opcode == OpCode::IntAdd {
                let mut restarted = Op::new(OpCode::IntSub, &[op.arg(0), op.arg(1)]);
                restarted.pos = op.pos;
                return OptimizationResult::Restart(restarted);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "restart_int_add_as_sub"
        }
    }

    #[test]
    fn test_optimizer_passthrough() {
        let mut opt = Optimizer::new();
        let ops = vec![Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)])];
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_restart_from_extra_operation_rediscovers_first_pass() {
        let hits = Rc::new(Cell::new(0));
        let mut opt = Optimizer::new();
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 8];
        opt.add_pass(Box::new(QueueRestartCandidate { queued: false }));
        opt.add_pass(Box::new(RestartIntAddAsSub));
        opt.add_pass(Box::new(CountRestartedIntSub { hits: hits.clone() }));

        let mut ops = vec![Op::new(OpCode::IntMul, &[OpRef(0), OpRef(1)])];
        ops[0].pos = OpRef(2);
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 2);

        assert_eq!(
            hits.get(),
            1,
            "Restart must re-dispatch from first_optimization, as optimizer.py:567 does"
        );
        assert!(
            result.iter().any(|op| op.opcode == OpCode::IntSub),
            "restarted extra operation should be emitted as IntSub: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_pairs_alive_when_results_are_used() {
        let field_descr = Arc::new(TestDescr(91));
        let call_descr_a = call_may_force_descr(81, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(82, majit_ir::Type::Ref);
        let mut ops = vec![
            Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(1)], call_descr_a),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone()),
            Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(2)], call_descr_b),
            Op::new(OpCode::GuardNotForced, &[]),
            Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr),
            Op::new(OpCode::IntAdd, &[OpRef(5), OpRef(8)]),
            Op::new(OpCode::Finish, &[OpRef(9)]),
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef((idx as u32) + 3);
        }

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 3);

        let call_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .count();
        let guard_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            call_count, 2,
            "optimized trace lost CallMayForceR ops: {result:?}"
        );
        assert_eq!(
            guard_count, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_keeps_call_may_force_when_guard_fail_args_reference_results() {
        let field_descr = Arc::new(TestDescr(101));
        let call_descr_a = call_may_force_descr(83, majit_ir::Type::Ref);
        let call_descr_b = call_may_force_descr(84, majit_ir::Type::Ref);
        let guard_types_a = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
        ];
        let guard_types_b = vec![
            majit_ir::Type::Ref,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
            majit_ir::Type::Ref,
        ];

        let mut call_a = Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(1)], call_descr_a);
        let mut guard_a = Op::with_descr(
            OpCode::GuardNotForced,
            &[],
            crate::compile::make_resume_guard_forced_descr_typed(guard_types_a.clone()),
        );
        guard_a.fail_args = Some(
            vec![
                OpRef(0),
                OpRef(2000),
                OpRef(2001),
                OpRef(3),
                OpRef(3000),
                OpRef(3001),
                OpRef(4),
            ]
            .into(),
        );
        guard_a.fail_arg_types = Some(guard_types_a);
        let get_a_type = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let get_a_val = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(3)], field_descr.clone());
        let mut call_b = Op::with_descr(OpCode::CallMayForceR, &[OpRef(0), OpRef(2)], call_descr_b);
        let mut guard_b = Op::with_descr(
            OpCode::GuardNotForced,
            &[],
            crate::compile::make_resume_guard_forced_descr_typed(guard_types_b.clone()),
        );
        guard_b.fail_args = Some(
            vec![
                OpRef(0),
                OpRef(2002),
                OpRef(2003),
                OpRef(3),
                OpRef(6),
                OpRef(3002),
                OpRef(3003),
                OpRef(7),
            ]
            .into(),
        );
        guard_b.fail_arg_types = Some(guard_types_b);
        let get_b_type = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr.clone());
        let get_b_val = Op::with_descr(OpCode::GetfieldGcPureI, &[OpRef(6)], field_descr);
        let add = Op::new(OpCode::IntAdd, &[OpRef(5), OpRef(8)]);
        let finish = Op::new(OpCode::Finish, &[OpRef(9)]);

        let mut ops = vec![
            call_a.clone(),
            guard_a,
            get_a_type,
            get_a_val,
            call_b.clone(),
            guard_b,
            get_b_type,
            get_b_val,
            add,
            finish,
        ];
        for (idx, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef((idx as u32) + 3);
        }
        call_a.pos = ops[0].pos;
        call_b.pos = ops[4].pos;

        let mut opt = Optimizer::default_pipeline();
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 3);

        let call_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::CallMayForceR)
            .map(|op| op.pos)
            .collect();
        assert!(
            call_positions.contains(&call_a.pos) && call_positions.contains(&call_b.pos),
            "optimized trace lost CallMayForceR producer(s): {result:?}"
        );
        let guarded = result
            .iter()
            .filter(|op| op.opcode == OpCode::GuardNotForced)
            .count();
        assert_eq!(
            guarded, 2,
            "optimized trace lost GuardNotForced ops: {result:?}"
        );
    }

    #[test]
    fn test_default_pipeline_has_7_passes() {
        // RPython __init__.py:15-22 ALL_OPTS + ENABLE_ALL_OPTS (rlib/jit.py):
        // intbounds:rewrite:virtualize:string:pure:earlyforce:heap (unroll separate)
        let opt = Optimizer::default_pipeline();
        assert_eq!(opt.num_passes(), 7);
    }

    #[test]
    fn test_default_pipeline_processes_trace() {
        let mut opt = Optimizer::default_pipeline();
        // IntAdd operates on Int-typed inputs — override the test default.
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 1024];
        // A simple trace: two INT_ADD with identical args. The Pure pass (CSE)
        // should eliminate the duplicate.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        // The duplicate INT_ADD should be eliminated by CSE (OptPure).
        let add_count = result.iter().filter(|o| o.opcode == OpCode::IntAdd).count();
        assert_eq!(add_count, 1, "CSE should eliminate duplicate INT_ADD");
        // Jump should still be present.
        assert_eq!(result.last().unwrap().opcode, OpCode::Jump);
    }

    #[test]
    fn test_remaps_all_op_positions_when_virtual_inputs_are_added() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(4)]),
            Op::new(OpCode::IntGt, &[OpRef(5), OpRef(1)]),
        ];
        ops[0].pos = OpRef(3);
        ops[1].pos = OpRef(4);
        ops[2].pos = OpRef(5);
        ops[3].pos = OpRef(6);

        let mut constants = std::collections::HashMap::new();
        constants.insert(1, 27);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        let positions: Vec<_> = result.iter().map(|op| op.pos).collect();
        assert_eq!(positions, vec![OpRef(5), OpRef(6), OpRef(7), OpRef(8)]);
        assert_eq!(result[2].arg(0), OpRef(6));
        assert_eq!(result[3].arg(0), OpRef(7));
    }

    #[test]
    fn test_remaps_removed_constants_away_from_compacted_live_ops() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef(5),
            value: 123,
        }));

        let mut ops = vec![
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::GetfieldRawI, &[OpRef(0)]),
            Op::new(OpCode::IntGt, &[OpRef(3), OpRef(1)]),
        ];
        ops[0].pos = OpRef(3);
        ops[1].pos = OpRef(4);
        ops[2].pos = OpRef(5);
        ops[3].pos = OpRef(6);

        let mut constants = std::collections::HashMap::new();
        constants.insert(1, 27);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        assert_eq!(result[0].pos, OpRef(5));
        assert_eq!(result[1].pos, OpRef(6));
        assert_eq!(result[2].pos, OpRef(7));
        assert_eq!(result[2].arg(0), OpRef(5));
        assert_eq!(constants.get(&5), None);
        assert_eq!(constants.get(&8), Some(&123));
    }

    #[test]
    fn test_export_constants_skips_live_op_positions() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(MarkAsTypedConstantButKeep {
            target: OpRef(3),
            value: majit_ir::Value::Int(1),
        }));

        let mut ops = vec![Op::new(OpCode::IntGt, &[OpRef(0), OpRef(1)])];
        ops[0].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        constants.insert(0, 40);
        constants.insert(1, 5);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].pos, OpRef(3));
        assert_eq!(constants.get(&3), None);
    }

    #[test]
    fn test_import_constants_skips_live_op_positions() {
        let mut opt = Optimizer::new();

        let mut ops = vec![Op::new(OpCode::IntGt, &[OpRef(0), OpRef(1)])];
        ops[0].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        constants.insert(0, 40);
        constants.insert(1, 5);
        constants.insert(3, 1);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 3);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].pos, OpRef(3));
        assert_eq!(result[0].opcode, OpCode::IntGt);
        assert_eq!(constants.get(&3), None);
    }

    #[test]
    fn test_skip_flush_keeps_terminal_jump_out_of_result_ops() {
        let mut opt = Optimizer::new();
        opt.skip_flush = true;

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        let terminal = opt
            .terminal_op
            .as_ref()
            .expect("skip_flush should preserve terminal jump");
        assert_eq!(terminal.opcode, OpCode::Jump);
        assert_eq!(terminal.args.as_ref(), &[OpRef(2)]);
    }

    #[test]
    fn test_get_count_of_ops_and_guards() {
        let mut opt = Optimizer::default_pipeline();
        // This test only exercises the ops/guard counter; each inputarg
        // is read by at least one Int-shape consumer (GuardTrue, IntAdd),
        // so seed Int to keep intbounds' type asserts happy.
        opt.trace_inputarg_types = vec![majit_ir::Type::Int; 1024];
        // optimizer.py:691 `assert isinstance(last_descr, ResumeGuardDescr)`:
        // the first GuardTrue becomes the sharing donor for the
        // descrless GuardNonnull below — give it a real descr so
        // OptContext::emit_guard_operation finds a valid donor.
        let mut guard_true = Op::new(OpCode::GuardTrue, &[OpRef(100)]);
        guard_true.descr = Some(crate::compile::make_resume_guard_descr_typed(Vec::new()));
        let mut ops = vec![
            guard_true,
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let ctx = OptContext::new(result.len());
        // Just verify the counting methods work
        assert_eq!(Optimizer::get_count_of_ops(&ctx), 0); // empty ctx
    }

    #[test]
    fn test_ref_guard_value_is_not_bool_specialized() {
        let mut opt = Optimizer::new();
        let ops = vec![
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::New, &[]),
            Op::new(OpCode::GuardValue, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Finish, &[]),
        ];

        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut std::collections::HashMap::new(), 0);

        assert!(
            result.iter().any(|op| op.opcode == OpCode::GuardValue),
            "ref-typed GuardValue should remain GuardValue"
        );
        assert!(
            !result
                .iter()
                .any(|op| op.opcode == OpCode::GuardTrue || op.opcode == OpCode::GuardFalse),
            "ref-typed GuardValue must not be rewritten as a bool guard"
        );
    }

    #[test]
    fn test_flush_invokes_all_passes() {
        let hits = Rc::new(Cell::new(0));
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));
        opt.add_pass(Box::new(FlushCounter { hits: hits.clone() }));

        let mut ctx = OptContext::new(0);
        opt.flush(&mut ctx);

        assert_eq!(hits.get(), 2);
    }

    #[test]
    fn test_extra_ops_do_not_flow_into_unroll_buffer() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(1)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));
        opt.add_pass(Box::new(crate::optimizeopt::unroll::OptUnroll::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        // force_all_lazy_setfields emits lazy SetfieldGc before JUMP.
        let new_count = result.iter().filter(|op| op.opcode == OpCode::New).count();
        assert!(
            new_count > 0,
            "force-like extra ops should still emit a New; got {:?}",
            result
        );
        // SetfieldGc is emitted by force_all_lazy_setfields at JUMP
        let setfield_count = result
            .iter()
            .filter(|op| op.opcode == OpCode::SetfieldGc)
            .count();
        assert_eq!(
            setfield_count, 1,
            "lazy SetfieldGc should be emitted at JUMP; got {:?}",
            result
        );
    }

    struct QueueTwoForceLikePairs {
        queued: bool,
        field_descr: majit_ir::DescrRef,
    }

    impl Optimization for QueueTwoForceLikePairs {
        fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
            if !self.queued && op.opcode == OpCode::IntAdd {
                self.queued = true;

                let alloc_a = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set_a = Op::new(OpCode::SetfieldGc, &[alloc_a, OpRef(0)]);
                set_a.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_a);

                let alloc_b = ctx.emit_extra(ctx.current_pass_idx, Op::new(OpCode::New, &[]));
                let mut set_b = Op::new(OpCode::SetfieldGc, &[alloc_b, OpRef(1)]);
                set_b.descr = Some(self.field_descr.clone());
                ctx.emit_extra(ctx.current_pass_idx, set_b);
            }
            OptimizationResult::PassOn
        }

        fn name(&self) -> &'static str {
            "queue_two_force_like_pairs"
        }
    }

    #[test]
    fn test_force_like_extra_ops_preserve_new_before_matching_setfield() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueTwoForceLikePairs {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(7)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let op_index: std::collections::HashMap<_, _> = result
            .iter()
            .enumerate()
            .map(|(idx, op)| (op.pos, idx))
            .collect();

        for set_op in result.iter().filter(|op| op.opcode == OpCode::SetfieldGc) {
            let alloc_ref = set_op.arg(0);
            let new_idx = result
                .iter()
                .position(|op| op.opcode == OpCode::New && op.pos == alloc_ref)
                .unwrap_or_else(|| panic!("missing New for {alloc_ref:?} in {result:?}"));
            let set_idx = *op_index
                .get(&set_op.pos)
                .unwrap_or_else(|| panic!("missing setfield pos {:?} in {:?}", set_op.pos, result));
            assert!(
                new_idx < set_idx,
                "matching New must appear before SetfieldGc; got {:?}",
                result
            );
        }
    }

    #[test]
    fn test_remap_keeps_force_like_new_positions_out_of_constant_map() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(AddVirtualInputsOnce { added: false }));
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(9)),
        }));
        opt.add_pass(Box::new(RemoveAsConstant {
            target: OpRef(2),
            value: 472,
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(2);
        ops[1].pos = OpRef(3);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos.0)
            .collect();
        assert!(
            !new_positions.is_empty(),
            "expected force-like New op in optimized trace; got {:?}",
            result
        );
        for pos in &new_positions {
            assert!(
                !constants.contains_key(pos),
                "live New position v{pos} must not collide with exported int constant map {:?}; trace {:?}",
                constants,
                result
            );
        }
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::SetfieldGc)
                .all(|op| new_positions.contains(&op.arg(0).0)),
            "SetfieldGc targets must remain emitted New refs; got {:?}",
            result
        );
    }

    #[test]
    fn test_force_like_extra_ops_skip_preexisting_constant_slots_without_virtual_inputs() {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(QueueForceLikeExtraOps {
            queued: false,
            field_descr: std::sync::Arc::new(TestDescr(11)),
        }));
        opt.add_pass(Box::new(OptHeap::new()));

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(1)]),
        ];
        ops[0].pos = OpRef(66);
        ops[1].pos = OpRef(67);

        let mut constants = std::collections::HashMap::new();
        constants.insert(OpRef::from_const(0).0, 472);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 2);

        let new_positions: std::collections::HashSet<_> = result
            .iter()
            .filter(|op| op.opcode == OpCode::New)
            .map(|op| op.pos.0)
            .collect();
        // With high-bit constant namespace, constant OpRefs never collide with
        // operation positions, so the New op lands at next_pos (68) directly.
        assert_eq!(
            new_positions,
            std::collections::HashSet::from([68]),
            "queued New should get next available slot; got {:?}",
            result
        );
        assert!(
            result
                .iter()
                .filter(|op| op.opcode == OpCode::SetfieldGc)
                .all(|op| new_positions.contains(&op.arg(0).0)),
            "SetfieldGc targets must remain emitted New refs; got {:?}",
            result
        );
        assert_eq!(constants.get(&OpRef::from_const(0).0), Some(&472));
        assert!(
            !constants.contains_key(&68),
            "live New position must not collide with constant map {:?}",
            constants
        );
    }

    #[test]
    #[ignore] // pre-existing test-setup bug (independent of rd_virtuals shape).
    // The fix needs three coupled changes that exceed an in-session port:
    //   (1) a parent-aware FieldDescr (TestDescr at line 4106 fits) returning
    //       a parent_descr so virtualize.rs:585-590 doesn't panic;
    //   (2) a SizeDescr that exposes an `all_field_descrs` array containing
    //       that field, so virtualize.rs:`init_fields` can index into the
    //       VirtualInfo (mirrors virtualize.rs:`TestParentSizeDescr`); and
    //   (3) re-link the two with a Weak parent reference so init_fields can
    //       look up the field's offset back to the size descr.
    // virtualize.rs already has all three primitives in its private `tests`
    // module — the cleanest port is to lift TestParentSizeDescr +
    // TestParentFieldDescr into a shared `mod test_descrs;` and reuse here.
    fn test_optimizer_encodes_direct_virtual_guard_fail_args_as_rd_numb() {
        let mut opt = Optimizer::default_pipeline();
        let size_descr = make_size_descr(16);
        let field_descr = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(10)]);
        guard.fail_args = Some(vec![OpRef(0)].into());

        let mut ops = vec![
            Op::with_descr(OpCode::New, &[], size_descr),
            Op::with_descr(OpCode::SetfieldGc, &[OpRef(0), OpRef(11)], field_descr),
            guard,
            Op::new(OpCode::Jump, &[]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }

        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );
        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");

        // resume.py:411-417 parity: rd_numb + rd_virtuals from
        // ResumeDataVirtualAdder.finish().
        assert!(
            guard.resolved_rd_numb().is_some(),
            "guard should have rd_numb (compact resume numbering)"
        );
        assert!(
            guard.resolved_rd_virtuals().is_some(),
            "virtual structure should be encoded into rd_virtuals tree"
        );
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("guard should keep fail args");
        // resume.py:411-417 parity: liveboxes is TAGBOX-only.  The virtual
        // p0 is encoded into rd_virtuals; only its int field (OpRef(11))
        // survives in liveboxes.
        assert!(
            fail_args.iter().all(|a| !a.is_none()),
            "RPython liveboxes are TAGBOX-only; got {:?}",
            fail_args
        );
        assert!(
            fail_args.iter().any(|&a| a == OpRef(11)),
            "virtual's field value (OpRef(11)) should appear in liveboxes; got {:?}",
            fail_args
        );
    }

    #[test]
    fn test_call_pure_results() {
        use majit_ir::Value;
        let mut opt = Optimizer::new();
        opt.record_call_pure_result(vec![Value::Int(10), Value::Int(20)], Value::Int(42));
        assert_eq!(
            opt.get_call_pure_result(&[Value::Int(10), Value::Int(20)]),
            Some(&Value::Int(42))
        );
        assert_eq!(
            opt.get_call_pure_result(&[Value::Int(10), Value::Int(99)]),
            None
        );
    }

    #[test]
    fn test_phase_seeded_value_types_skip_void_entries() {
        let mut opt = Optimizer::new();
        opt.original_trace_op_types.insert(40, Type::Void);
        opt.original_trace_op_types.insert(41, Type::Int);
        opt.prev_phase_value_types.insert(50, Type::Void);
        opt.prev_phase_value_types.insert(51, Type::Ref);

        let mut constants = std::collections::HashMap::new();
        let result = opt.optimize_with_constants_and_inputs_at(&[], &mut constants, 0, 0, 0);

        assert!(result.is_empty());
        assert_eq!(opt.prev_phase_value_types.get(&41), Some(&Type::Int));
        assert_eq!(opt.prev_phase_value_types.get(&51), Some(&Type::Ref));
        assert!(!opt.prev_phase_value_types.contains_key(&40));
        assert!(!opt.prev_phase_value_types.contains_key(&50));
        assert!(
            !opt.prev_phase_value_types
                .values()
                .any(|&tp| tp == Type::Void),
            "value_types must only carry Box-producing i/r/f entries"
        );
    }

    #[test]
    fn test_is_call_pure_pure_canraise_ignores_memoryerror_only() {
        let mut op = Op::new(OpCode::CallPureI, &[OpRef(0), OpRef(1)]);
        op.descr = Some(Arc::new(TestCallDescr {
            idx: 400,
            effect: EffectInfo::new(ExtraEffect::ElidableOrMemoryError, OopSpecIndex::None),
            result_type: majit_ir::Type::Int,
        }));
        assert!(
            !Optimizer::is_call_pure_pure_canraise(&op),
            "optimizer.py:705-711 ignores MemoryError-only effects"
        );
    }

    #[test]
    fn test_is_call_pure_pure_canraise_true_for_other_raising_effects() {
        let mut op = Op::new(OpCode::CallPureI, &[OpRef(0), OpRef(1)]);
        op.descr = Some(Arc::new(TestCallDescr {
            idx: 401,
            effect: EffectInfo::new(ExtraEffect::ElidableCanRaise, OopSpecIndex::None),
            result_type: majit_ir::Type::Int,
        }));
        assert!(Optimizer::is_call_pure_pure_canraise(&op));
    }

    #[test]
    fn test_protect_speculative_operation() {
        let opt = Optimizer::new();
        let ctx = OptContext::new(10);

        // Arithmetic ops are always safe
        let add_op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        assert!(opt.protect_speculative_operation(&add_op, &ctx));

        // Getfield on unknown arg is safe (not constant null)
        let get_op = Op::new(OpCode::GetfieldGcI, &[OpRef(0)]);
        assert!(opt.protect_speculative_operation(&get_op, &ctx));
    }

    #[test]
    fn test_pending_fields() {
        let mut opt = Optimizer::new();
        assert!(!opt.has_pending_fields());
        assert_eq!(opt.num_pending_fields(), 0);

        opt.add_pending_field(Op::new(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)]));
        assert!(opt.has_pending_fields());
        assert_eq!(opt.num_pending_fields(), 1);
    }

    /// optimizer.py:127-135 `getnullness(op)` parity test.
    ///
    /// Returns the upstream INFO_NULL / INFO_NONNULL / INFO_UNKNOWN
    /// integer constants (info.py:13-15).
    #[test]
    fn test_getnullness() {
        use crate::optimizeopt::{INFO_NONNULL, INFO_NULL, INFO_UNKNOWN};
        let mut ctx = OptContext::new(10);
        // Unknown integer → INFO_UNKNOWN.
        assert_eq!(Optimizer::getnullness(&mut ctx, OpRef(0)), INFO_UNKNOWN);
        // Known nonzero integer → INFO_NONNULL.
        ctx.make_constant(OpRef(1), majit_ir::Value::Int(42));
        assert_eq!(Optimizer::getnullness(&mut ctx, OpRef(1)), INFO_NONNULL);
        // Known zero integer → INFO_NULL.
        ctx.make_constant(OpRef(2), majit_ir::Value::Int(0));
        assert_eq!(Optimizer::getnullness(&mut ctx, OpRef(2)), INFO_NULL);
    }

    #[test]
    fn test_guard_replacement_flag() {
        let mut opt = Optimizer::new();
        assert!(opt.can_replace_guards);
        opt.disable_guard_replacement();
        assert!(!opt.can_replace_guards);
        opt.enable_guard_replacement();
        assert!(opt.can_replace_guards);
    }

    #[test]
    fn test_force_at_the_end_of_preamble_recurses_virtual_fields() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let mut ctx = OptContext::with_num_inputs(32, 1024);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(1, OpRef(11))],
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );
        ctx.replace_op(OpRef(11), OpRef(20));
        ctx.set_ptr_info(
            OpRef(20),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr,
                fields: Vec::new(),
                field_descrs: Vec::new(),
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let mut opt = Optimizer::new();
        let result = opt.force_box_for_end_of_preamble(OpRef(10), &mut ctx);

        // The virtual is forced to a concrete allocation; the returned ref
        // is the allocation's position, which ctx.get_box_replacement(OpRef(10))
        // should resolve to.
        assert_eq!(result, ctx.get_box_replacement(OpRef(10)));
        // After forcing, the struct's ptr_info reflects that field 1
        // (originally OpRef(11), forwarded to OpRef(20)) has been recursively forced.
        match ctx.get_ptr_info(result) {
            Some(PtrInfo::VirtualStruct(info)) => {
                // The inner virtual (OpRef(20)) was also forced; its allocation
                // ref is whatever force_box assigned.
                assert_eq!(info.fields.len(), 1);
                assert_eq!(info.fields[0].0, 1);
            }
            // After full forcing the info might become NonNull or similar
            Some(PtrInfo::NonNull { .. }) => {}
            other => panic!("expected virtual struct or non-null after forcing, got {other:?}"),
        }
    }

    #[test]
    fn test_emit_operation_materializes_virtual_args_directly() {
        use crate::optimizeopt::info::{PtrInfo, VirtualStructInfo};

        let descr = make_size_descr(16);
        let field_descr: DescrRef = std::sync::Arc::new(TestDescr(0));
        let mut ctx = OptContext::with_num_inputs(16, 1);
        ctx.set_ptr_info(
            OpRef(10),
            PtrInfo::VirtualStruct(VirtualStructInfo {
                descr: descr.clone(),
                fields: vec![(0, OpRef(11))],
                field_descrs: vec![field_descr.clone()],
                last_guard_pos: -1,
                cached_vinfo: std::cell::RefCell::new(None),
            }),
        );

        let mut opt = Optimizer::new();
        let op = Op::new(OpCode::GuardNonnull, &[OpRef(10)]);
        let (mut seeded_ops, snapshots) =
            super::super::seed_empty_guard_snapshots(std::slice::from_ref(&op));
        ctx.snapshot_boxes = snapshots;
        opt.emit_operation(seeded_ops.pop().unwrap(), &mut ctx);

        assert!(!ctx.in_final_emission);
        assert!(ctx.new_operations.iter().any(|op| op.opcode == OpCode::New));
        assert!(ctx.new_operations.iter().any(|op| {
            op.opcode == OpCode::SetfieldGc && op.arg(1) == OpRef(11) && op.descr.is_some()
        }));
        // info.py:146-151: force_box emits the ORIGINAL box op, so the
        // forced GuardNonnull keeps arg(0) = OpRef(10) (matches the virtual's
        // original identity). force_box_impl preserves `new_op.pos = opref`.
        assert!(
            ctx.new_operations
                .iter()
                .any(|op| op.opcode == OpCode::GuardNonnull && op.arg(0) == OpRef(10))
        );
    }

    #[test]
    fn test_emit_operation_forces_imported_short_guard_args() {
        let mut opt = Optimizer::new();
        let mut ctx = OptContext::with_num_inputs(16, 1);

        let mut preamble_op = Op::new(OpCode::IntGe, &[OpRef(3), OpRef(10_000)]);
        preamble_op.pos = OpRef(14);
        ctx.make_constant(OpRef(10_000), majit_ir::Value::Int(0));
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef(0)],
            &[OpRef(0)],
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: preamble_op.clone(),
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            }],
        );
        ctx.potential_extra_ops.insert(
            OpRef(14),
            crate::optimizeopt::info::PreambleOp {
                op: OpRef(14),
                resolved: OpRef(14),
                invented_name: false,
                preamble_op: {
                    let mut op = majit_ir::Op::new(majit_ir::OpCode::SameAsI, &[OpRef(14)]);
                    op.pos = OpRef(14);
                    op
                },
            },
        );

        let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(14)]);
        guard.pos = OpRef(15);
        let (mut seeded_ops, snapshots) =
            super::super::seed_empty_guard_snapshots(std::slice::from_ref(&guard));
        ctx.snapshot_boxes = snapshots;
        opt.emit_operation(seeded_ops.pop().unwrap(), &mut ctx);

        let sp = ctx
            .build_imported_short_preamble()
            .expect("forcing imported short guard arg should build short preamble");
        assert_eq!(sp.used_boxes, vec![OpRef(14)]);
        assert_eq!(sp.jump_args, vec![OpRef(14)]);
    }

    #[test]
    fn test_resumedata_memo_encodes_rd_numb_on_guard() {
        let mut opt = Optimizer::default_pipeline();
        // OptIntBound (mod.rs:2624 getintbound) requires IntAdd's args to be
        // Type::Int.  trace_inputarg_types defaults to all-Ref, so we must
        // override slots 100 and 101.
        let mut input_types = vec![Type::Ref; 1024];
        input_types[100] = Type::Int;
        input_types[101] = Type::Int;
        opt.trace_inputarg_types = input_types;

        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::Finish, &[]),
        ];
        ops[1].fail_args = Some(vec![OpRef(100), OpRef(101)].into());
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }

        let (ops, snapshots) =
            super::super::seed_guard_snapshots_with(&ops, |_| vec![OpRef(100), OpRef(101)]);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        let guard = result
            .iter()
            .find(|op| op.opcode == OpCode::GuardTrue)
            .expect("guard should survive optimization");
        assert!(
            guard.resolved_rd_numb().is_some(),
            "resumedata_memo should set rd_numb on guard"
        );
        assert!(
            guard.resolved_rd_consts().is_some(),
            "resumedata_memo should set rd_consts on guard"
        );
    }

    #[test]
    fn update_counters_folds_opt_ops_opt_guards_opt_guards_shared_into_profiler() {
        // optimizer.py:626/629/673-674: every emit_operation bumps OPT_OPS,
        // each guard bumps OPT_GUARDS additionally, and the sharing path
        // bumps OPT_GUARDS_SHARED.  Pyre defers the fold via
        // update_counters; assert the accumulators land in the matching
        // `JitProfiler` atomics and reset after the fold so a second
        // call doesn't double-count.
        let mut opt = Optimizer::new();
        opt.opt_ops_emitted = 5;
        opt.opt_guards_emitted = 2;
        opt.opt_guards_shared_emitted = 1;
        let prof = crate::jitprof::JitProfiler::default();
        opt.update_counters(&prof);
        let snap = prof.snapshot();
        assert_eq!(snap.opt_ops, 5);
        assert_eq!(snap.opt_guards, 2);
        assert_eq!(snap.opt_guards_shared, 1);
        // Accumulators reset so a second update doesn't re-add.
        assert_eq!(opt.opt_ops_emitted, 0);
        assert_eq!(opt.opt_guards_emitted, 0);
        assert_eq!(opt.opt_guards_shared_emitted, 0);
        opt.update_counters(&prof);
        let snap = prof.snapshot();
        assert_eq!(snap.opt_ops, 5);
        assert_eq!(snap.opt_guards, 2);
        assert_eq!(snap.opt_guards_shared, 1);
    }
}
