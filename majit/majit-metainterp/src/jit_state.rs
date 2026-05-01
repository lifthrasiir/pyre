//! No direct RPython equivalent — trait abstraction for interpreter state
//! (RPython uses concrete MetaInterp class in pyjitpl.py).

use majit_ir::{GcRef, OpRef, Type, Value};

use crate::blackhole::ExceptionState;
use crate::jitdriver::JitDriverStaticData;
use crate::resume::{
    MaterializedValue, MaterializedVirtual, ReconstructedState, ResolvedPendingFieldWrite,
    ResumeFrameLayoutSummary,
};
use crate::virtualizable::VirtualizableInfo;

/// Layout description for replaying a pending field or array write during deopt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PendingFieldWriteLayout {
    Field {
        offset: usize,
        value_type: Type,
    },
    ArrayItem {
        base_offset: usize,
        item_size: usize,
        item_type: Type,
    },
}

/// Session-scoped materialization cache that persists across guard failure
/// resumption within the same deopt session.
///
/// When GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE, the same
/// virtuals may need to be materialized again. This cache ensures each virtual
/// is materialized at most once per deopt session and reused throughout.
#[derive(Debug, Clone, Default)]
pub struct DeoptMaterializationCache {
    /// Maps virtual indices to materialized GcRefs. `None` means not yet
    /// materialized; `Some(gcref)` means already materialized in this session.
    pub refs: Vec<Option<GcRef>>,
}

impl DeoptMaterializationCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self { refs: Vec::new() }
    }

    /// Create a cache pre-sized for the given number of virtuals.
    pub fn with_capacity(num_virtuals: usize) -> Self {
        Self {
            refs: vec![None; num_virtuals],
        }
    }

    /// Return the number of cached entries (including None slots).
    pub fn len(&self) -> usize {
        self.refs.len()
    }

    /// Return true if no slots have been allocated.
    pub fn is_empty(&self) -> bool {
        self.refs.is_empty()
    }

    /// Return the number of successfully materialized entries.
    pub fn materialized_count(&self) -> usize {
        self.refs.iter().filter(|r| r.is_some()).count()
    }

    /// Clear all cached entries, ending the deopt session.
    pub fn clear(&mut self) {
        self.refs.clear();
    }

    /// Look up a cached materialization by virtual index.
    pub fn get(&self, virtual_index: usize) -> Option<GcRef> {
        self.refs.get(virtual_index).copied().flatten()
    }

    /// Ensure the cache has room for at least `count` virtual indices.
    pub fn ensure_capacity(&mut self, count: usize) {
        if self.refs.len() < count {
            self.refs.resize(count, None);
        }
    }

    /// Insert a materialized ref at the given virtual index.
    pub fn insert(&mut self, virtual_index: usize, gc_ref: GcRef) {
        self.ensure_capacity(virtual_index + 1);
        self.refs[virtual_index] = Some(gc_ref);
    }
}

/// resume.py:1042 rebuild_from_resumedata return value.
///
/// RPython returns `(liveboxes, virtualizable_boxes, virtualref_boxes)`.
/// `liveboxes[i]` = InputArgBox for TAGBOX(i), None otherwise.
/// Frame registers are filled with InputArgBox/ConstBox/VirtualBox mix.
///
/// Rust equivalent: `RebuiltFrame.values` carries per-slot RebuiltValue
/// (Box/Const/Int/Virtual/Unassigned). `constants` holds constant OpRef
/// entries for the optimizer constant pool (no RPython equivalent —
/// RPython uses ConstBox objects natively in the trace recording).
///
/// resume.py:1245 `decode_box(num, kind)` parity: each
/// `RebuiltValue::Box(idx, kind)` carries its own kind, sourced from the
/// parent guard's `fail_arg_types` at `decode_tagged` time. Consumers
/// read the kind directly from the variant — no parallel
/// `fail_arg_types` field on this struct is needed.
#[derive(Debug, Clone)]
pub struct ResumeDataResult {
    /// resume.py:1057: per-frame decoded values from rd_numb.
    /// Each RebuiltValue::Box(i, kind) → liveboxes[i] in RPython.
    pub frames: Vec<majit_ir::resumedata::RebuiltFrame>,
    /// resume.py:1045: virtualizable boxes (decoded from vable section).
    pub virtualizable_values: Vec<majit_ir::resumedata::RebuiltValue>,
    /// resume.py:1045: virtualref box pairs (decoded from vref section).
    pub virtualref_values: Vec<majit_ir::resumedata::RebuiltValue>,
    /// compile.py:853 `ResumeGuardDescr` storage handle — the shared
    /// `Arc<ResumeStorage>` owned by the guard whose rd_numb we just
    /// decoded. Virtual materialization (resume.py:1251
    /// `decode_box` → `self.consts[num-TAG_CONST_OFFSET]`) reads
    /// `storage.rd_consts()` directly; no owned copy lives on this
    /// struct.
    pub storage: Option<std::sync::Arc<crate::resume::ResumeStorage>>,
    /// resume.py:1042 num_failargs read from rd_numb header.
    /// Needed by `materialize_bridge_virtual` to translate negative TAGBOX
    /// numbers in `rd_virtuals` fieldnums (resume.py:1556-1564 decode_box:
    /// Python-style negative indexing into the parent guard's liveboxes
    /// array; the decoder normalizes via `val + num_failargs`).
    pub num_failargs: i32,
}

/// Interpreter-specific JIT state contract.
pub trait JitState: Sized {
    type Meta: Clone;
    type Sym;
    type Env: ?Sized;

    fn can_trace(&self) -> bool {
        true
    }

    fn build_meta(&self, header_pc: usize, env: &Self::Env) -> Self::Meta;

    fn extract_live(&self, meta: &Self::Meta) -> Vec<i64>;

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        let raw = self.extract_live(meta);
        let types = self.live_value_types(meta);
        raw.into_iter()
            .zip(types)
            .map(|(v, t)| match t {
                Type::Float => Value::Float(f64::from_bits(v as u64)),
                Type::Ref => Value::Ref(GcRef(v as usize)),
                _ => Value::Int(v),
            })
            .collect()
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        self.extract_live(meta).iter().map(|_| Type::Int).collect()
    }

    fn create_sym(meta: &Self::Meta, header_pc: usize) -> Self::Sym;

    /// Seed tracing-time concrete mirrors in the symbolic state.
    ///
    /// Interpreters that lower state reads through jitcode need this so
    /// branch decisions during tracing use the real runtime values.
    fn initialize_sym(&self, _sym: &mut Self::Sym, _meta: &Self::Meta) {}

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool;

    /// pyjitpl.py:2982 get_procedure_token(greenboxes): compute the
    /// green key for a given loop header PC. Used by bridge trace close
    /// to find compiled targets at the destination loop header.
    fn green_key_for_pc(&self, _pc: usize) -> Option<u64> {
        None
    }

    /// The code object pointer for green key computation.
    /// RPython: jitdriver_sd.jitcodes[jitcode_pos]
    fn code_ptr(&self) -> usize {
        0
    }

    /// compile.py:269-270 parity: update meta when the trace is cut at
    /// a cross-loop merge point. `header_pc` is the bytecode PC of the
    /// new loop header; `original_box_types` are the types of the live
    /// boxes at the cut point (the new inputargs).
    fn update_meta_for_cut(
        _meta: &mut Self::Meta,
        _header_pc: usize,
        _original_box_types: &[Type],
    ) {
    }

    /// resume.py:1042 parity: set up bridge-specific symbolic local mapping.
    /// Called after rebuild_from_resumedata to map frame locals to bridge
    /// InputArg OpRefs. In RPython, MIFrame.registers are populated with
    /// InputArg/Const boxes from rebuild; this is the Rust equivalent.
    ///
    /// `ctx` is the bridge's active trace context. When `resume_data` carries
    /// `RebuiltValue::Virtual` entries, the implementation must emit
    /// `NEW_WITH_VTABLE`/`SETFIELD_GC` ops via `ctx` to materialize each
    /// virtual at trace start, mirroring RPython's
    /// `ResumeDataBoxReader.consume_boxes` → `rd_virtuals[i].allocate(...)`
    /// → `metainterp.execute_new_with_vtable` (resume.py:945-956 getvirtual_ptr).
    /// `rd_virtuals` is the parent guard's virtual descriptor table from
    /// the source trace's exit layout (None when the parent had no virtuals).
    fn setup_bridge_sym(
        _sym: &mut Self::Sym,
        _ctx: &mut crate::trace_ctx::TraceCtx,
        _resume_data: &ResumeDataResult,
        _rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
        _fail_values: &[i64],
        _fail_types: &[Type],
    ) {
    }

    /// resume.py:1042-1057 rebuild_from_resumedata: decode rd_numb to
    /// reconstruct the complete frame state for bridge tracing.
    ///
    /// RPython creates ResumeDataBoxReader, consumes vable/vref/frame
    /// sections, fills MIFrame registers with InputArgBox/ConstBox mix,
    /// returns (liveboxes, virtualizable_boxes, virtualref_boxes).
    ///
    /// Returns None when storage is not available (legacy path).
    /// `storage` is the guard-owned shared `Arc<ResumeStorage>` handle
    /// (compile.py:853 `ResumeGuardDescr`); readers borrow
    /// `storage.rd_numb` / `storage.rd_consts()` instead of owned
    /// copies.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        _fail_arg_types: &[Type],
        _storage: Option<&std::sync::Arc<crate::resume::ResumeStorage>>,
    ) -> Option<ResumeDataResult> {
        None
    }

    /// pyjitpl.py:3158-3175 compile_loop parity: build final meta from
    /// the MergePoint that matched at close time, not from the trace start.
    ///
    /// RPython's compile_loop extracts greenkey and inputargs from
    /// `original_boxes` (which comes from `current_merge_points[j]`).
    /// This method rebuilds the meta entirely from MergePoint data,
    /// replacing the post-patch `update_meta_for_cut` band-aid.
    fn build_meta_from_merge_point(
        provisional: &Self::Meta,
        header_pc: usize,
        original_box_types: &[Type],
    ) -> Self::Meta {
        // Default: clone + patch (backward compat for non-pyre consumers)
        let mut meta = provisional.clone();
        Self::update_meta_for_cut(&mut meta, header_pc, original_box_types);
        meta
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]);

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let ints: Vec<i64> = values
            .iter()
            .map(|value| match value {
                Value::Int(v) => *v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect();
        self.restore(meta, &ints);
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        self.restore_values(meta, values);
        true
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return true;
        };
        let lengths = if info.can_read_all_array_lengths_from_heap() {
            unsafe { info.read_array_lengths_from_heap(obj_ptr.cast_const()) }
        } else {
            let Some(lengths) = self.virtualizable_array_lengths(meta, virtualizable, info) else {
                return true;
            };
            lengths
        };
        let (static_boxes, array_boxes) =
            unsafe { info.read_all_boxes(obj_ptr.cast_const(), &lengths) };
        self.import_virtualizable_boxes(meta, virtualizable, info, &static_boxes, &array_boxes)
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(obj_ptr) = self.virtualizable_heap_ptr(meta, virtualizable, info) else {
            return;
        };
        let Some((static_boxes, array_boxes)) =
            self.export_virtualizable_boxes(meta, virtualizable, info)
        else {
            return;
        };
        unsafe {
            info.write_all_boxes(obj_ptr, &static_boxes, &array_boxes);
            info.reset_vable_token(obj_ptr);
        }
    }

    fn sync_named_virtualizable_before_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) -> bool {
        match info {
            Some(info) => self.sync_virtualizable_before_jit(meta, virtualizable, info),
            None => true,
        }
    }

    fn sync_named_virtualizable_after_jit(
        &mut self,
        meta: &Self::Meta,
        virtualizable: &str,
        info: Option<&VirtualizableInfo>,
    ) {
        if let Some(info) = info {
            self.sync_virtualizable_after_jit(meta, virtualizable, info);
        }
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        None
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        None
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        _static_boxes: &[i64],
        _array_boxes: &[Vec<i64>],
    ) -> bool {
        true
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        None
    }

    /// Called before a residual call (CALL_MAY_FORCE) during tracing.
    ///
    /// Writes virtualizable state to heap via SETFIELD_GC ops so the callee
    /// can observe up-to-date virtualizable fields. Interpreters that use
    /// virtualizable objects should override this to emit the necessary
    /// SETFIELD_GC operations for each tracked field.
    ///
    /// Default: no-op (interpreters without virtualizable fields).
    fn sync_virtualizable_before_residual_call(&self, _ctx: &mut crate::trace_ctx::TraceCtx) {}

    /// Called after a residual call (CALL_MAY_FORCE) during tracing.
    ///
    /// Re-reads virtualizable state from heap via GETFIELD_GC ops because
    /// the callee may have modified virtualizable fields. Also reports whether
    /// the standard virtualizable token protocol observed a force, in which
    /// case the caller should abort tracing instead of emitting GUARD_NOT_FORCED.
    ///
    /// Default: no-op (interpreters without virtualizable fields).
    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut crate::trace_ctx::TraceCtx,
    ) -> ResidualVirtualizableSync {
        ResidualVirtualizableSync::default()
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef>;

    /// Bridge `Sym`'s state slots onto the current `MIFrame.int_regs`
    /// and return a single-frame `recorder::Snapshot` for the guard
    /// about to be (or just-) recorded.  Default no-op returns `None`,
    /// preserving the legacy side-vec-only path; state-field JIT (the
    /// macro emit at `majit-macros::jit_interp::codegen_state`) and
    /// any other consumer that wants its `record_*_with_fail_args`
    /// guards to also carry an `rd_resume_position` overrides this
    /// hook.
    ///
    /// PRE-EXISTING-ADAPTATION (Task #89 framestack-lift Session 4):
    /// this is the JitState-level entry point for the framestack-lift
    /// epic.  It mirrors `JitCodeSym::populate_frame_int_regs`
    /// (`pyjitpl/dispatch.rs:113-146`) but is reachable from
    /// `merge_point` / `force_finish_trace` where the generic `S::Sym`
    /// is not constrained to implement `JitCodeSym`.  Once
    /// `JitState::Sym: JitCodeSym` becomes the universal contract
    /// (Task #89 orth-9 step 4 reshape), this hook collapses into the
    /// `JitCodeSym` method directly.
    fn populate_frame_for_guard(
        _sym: &Self::Sym,
        _frames: &mut crate::pyjitpl::MIFrameStack,
    ) -> Option<crate::recorder::Snapshot> {
        None
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        Self::collect_jump_args(sym)
            .into_iter()
            .map(|opref| (opref, Type::Int))
            .collect()
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool;

    fn validate_close_with_jump_args(
        sym: &Self::Sym,
        meta: &Self::Meta,
        _jump_args: &[OpRef],
    ) -> bool {
        Self::validate_close(sym, meta)
    }

    /// resume.py:1049 parity: push a reconstructed caller frame.
    /// Called during multi-frame deopt: innermost frame is current,
    /// outer frames are pushed in reverse order (outermost first).
    ///
    /// `frame_index`: 0 = outermost, N-1 = innermost (current)
    /// `values`: typed values for this frame's slots
    /// `pc`: the program counter / bytecode position for this frame
    /// `jitcode_index`: JitCode index for CodeObject lookup
    ///
    /// Returns true if the frame was successfully pushed.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        false // Default: no multi-frame support
    }

    /// Pop the current frame and resume the caller frame.
    /// Called when blackhole execution of the innermost frame completes.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Default: no multi-frame support
    }

    /// RPython virtualizable.py: build VirtualizableInfo.
    /// Override when the interpreter has a virtualizable object.
    ///
    /// Returns `Arc<VirtualizableInfo>` so the field descriptors built
    /// by `finalize_arc` can carry the vinfo backref that
    /// `FieldDescr::get_vinfo()` returns at `emit_force_virtualizable`
    /// time (pyjitpl.py:1148-1149).
    #[allow(non_snake_case)]
    fn __build_virtualizable_info()
    -> Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>> {
        None
    }

    /// Check if multi-frame caller-stack restore is supported.
    fn supports_multi_frame_restore(&self) -> bool {
        false
    }

    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        _materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        None
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        meta: &Self::Meta,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
        _materialized_refs: &[Option<GcRef>],
    ) -> Option<GcRef> {
        self.materialize_virtual_ref(meta, virtual_index, materialized)
    }

    fn pending_field_write_layout(
        &self,
        _meta: &Self::Meta,
        _descr_index: u32,
        _is_array_item: bool,
    ) -> Option<PendingFieldWriteLayout> {
        None
    }

    fn restore_reconstructed_frames(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
        self.try_restore_reconstructed_frames_with_cache(
            meta,
            reconstructed_state,
            exception,
            &cache,
        )
    }

    fn restore_reconstructed_frames_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        exception: &ExceptionState,
    ) -> bool {
        let total_frames = reconstructed_state.frames.len();
        let can_use_generic_layout = total_frames == 1
            || reconstructed_state
                .frames
                .iter()
                .enumerate()
                .any(|(frame_index, frame)| {
                    frame.slot_types.as_ref().is_some()
                        || self
                            .reconstructed_frame_value_types_with_metadata(
                                meta,
                                frame_index,
                                total_frames,
                                frame,
                            )
                            .is_some()
                        || frame_layouts
                            .and_then(|layouts| layouts.get(frame_index))
                            .filter(|layout| {
                                layout.pc == frame.pc
                                    && layout.slot_layouts.len() == frame.values.len()
                            })
                            .and_then(|layout| layout.slot_types.as_ref())
                            .is_some()
                });

        if can_use_generic_layout {
            let cache = self.materialize_virtual_refs(meta, materialized_virtuals);
            if self.try_restore_reconstructed_frames_with_cache_and_layout(
                meta,
                reconstructed_state,
                exception,
                &cache,
                frame_layouts,
            ) {
                return true;
            }
        }

        self.restore_reconstructed_frames(
            meta,
            reconstructed_state,
            materialized_virtuals,
            exception,
        )
    }

    fn reconstructed_frame_value_types(
        &self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        None
    }

    fn reconstructed_frame_value_types_with_metadata(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
    ) -> Option<Vec<Type>> {
        self.reconstructed_frame_value_types(meta, frame_index, total_frames, frame.pc)
    }

    fn restore_reconstructed_frame_values(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        _values: &[Value],
        _exception: &ExceptionState,
    ) -> bool {
        false
    }

    fn restore_reconstructed_frame_values_with_metadata(
        &mut self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
        values: &[Value],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_reconstructed_frame_values(
            meta,
            frame_index,
            total_frames,
            frame.pc,
            values,
            exception,
        )
    }

    fn try_restore_reconstructed_frames_with_cache(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
    ) -> bool {
        self.try_restore_reconstructed_frames_with_cache_and_layout(
            meta,
            reconstructed_state,
            exception,
            materialized_refs,
            None,
        )
    }

    fn try_restore_reconstructed_frames_with_cache_and_layout(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
    ) -> bool {
        if reconstructed_state.frames.is_empty() {
            return false;
        }

        let total_frames = reconstructed_state.frames.len();
        let mut used_generic = false;
        for (frame_index, frame) in reconstructed_state.frames.iter().enumerate() {
            let layout_types = frame.slot_types.clone().or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.get(frame_index))
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            });
            if let Some(types) = self
                .reconstructed_frame_value_types_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                )
                .or(layout_types)
            {
                let Some(values) =
                    frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
                else {
                    return false;
                };
                if self.restore_reconstructed_frame_values_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                    &values,
                    exception,
                ) {
                    used_generic = true;
                    continue;
                }
                if total_frames == 1 && frame_index + 1 == total_frames {
                    self.restore_values(meta, &values);
                    return true;
                }
                return false;
            }
        }

        if used_generic {
            return true;
        }

        let frame = reconstructed_state.frames.last().expect("non-empty");
        let types = frame
            .slot_types
            .clone()
            .or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.last())
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            })
            .unwrap_or_else(|| self.live_value_types(meta));
        if types.len() != frame.values.len() {
            return false;
        }
        let Some(values) =
            frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
        else {
            return false;
        };
        self.restore_values(meta, &values);
        true
    }

    fn restore_guard_failure(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_guard_failure_with_resume_layout(
            meta,
            fail_values,
            reconstructed_state,
            None,
            materialized_virtuals,
            pending_field_writes,
            exception,
        )
    }

    fn restore_guard_failure_with_resume_layout(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
    ) -> bool {
        self.restore_guard_failure_with_session_cache(
            meta,
            fail_values,
            reconstructed_state,
            frame_layouts,
            materialized_virtuals,
            pending_field_writes,
            exception,
            None,
        )
    }

    fn materialize_virtual_refs(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
    ) -> Vec<Option<GcRef>> {
        let mut refs = vec![None; materialized_virtuals.len()];
        let mut progress = true;
        while progress {
            progress = false;
            for (virtual_index, virtual_value) in materialized_virtuals.iter().enumerate() {
                if refs[virtual_index].is_some() {
                    continue;
                }
                if let Some(materialized) = self.materialize_virtual_ref_with_refs(
                    meta,
                    virtual_index,
                    virtual_value,
                    &refs,
                ) {
                    refs[virtual_index] = Some(materialized);
                    progress = true;
                }
            }
        }
        refs
    }

    /// Materialize virtual refs into an existing session cache.
    ///
    /// Virtuals that are already present in the cache (from a previous deopt
    /// in the same session) are skipped. Only unmaterialized entries trigger
    /// a call to `materialize_virtual_ref_with_refs`.
    fn materialize_virtual_refs_into_cache(
        &mut self,
        meta: &Self::Meta,
        materialized_virtuals: &[MaterializedVirtual],
        cache: &mut DeoptMaterializationCache,
    ) {
        cache.ensure_capacity(materialized_virtuals.len());
        let mut progress = true;
        while progress {
            progress = false;
            for (idx, virt) in materialized_virtuals.iter().enumerate() {
                if cache.refs[idx].is_some() {
                    continue; // Already materialized (possibly from previous session)
                }
                if let Some(materialized) =
                    self.materialize_virtual_ref_with_refs(meta, idx, virt, &cache.refs)
                {
                    cache.refs[idx] = Some(materialized);
                    progress = true;
                }
            }
        }
    }

    /// Force a single virtual using the session cache. Returns the cached
    /// result if already materialized, otherwise materializes and inserts
    /// into the cache.
    fn force_virtual_with_cache(
        &mut self,
        meta: &Self::Meta,
        cache: &mut DeoptMaterializationCache,
        virtual_index: usize,
        materialized: &MaterializedVirtual,
    ) -> Option<GcRef> {
        // Check cache first
        if let Some(cached) = cache.get(virtual_index) {
            return Some(cached);
        }
        // Materialize and cache
        cache.ensure_capacity(virtual_index + 1);
        let result =
            self.materialize_virtual_ref_with_refs(meta, virtual_index, materialized, &cache.refs)?;
        cache.refs[virtual_index] = Some(result);
        Some(result)
    }

    /// Restore guard failure state using a persistent session cache.
    ///
    /// This is the session-aware variant of `restore_guard_failure_with_resume_layout`.
    /// When `session_cache` is provided, materialized virtuals are stored in
    /// it and reused across subsequent deopts within the same session (e.g.,
    /// when GUARD_NOT_FORCED triggers resumption after CALL_MAY_FORCE).
    fn restore_guard_failure_with_session_cache(
        &mut self,
        meta: &Self::Meta,
        fail_values: &[i64],
        reconstructed_state: Option<&ReconstructedState>,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        materialized_virtuals: &[MaterializedVirtual],
        pending_field_writes: &[ResolvedPendingFieldWrite],
        exception: &ExceptionState,
        session_cache: Option<&mut DeoptMaterializationCache>,
    ) -> bool {
        if let Some(reconstructed_state) = reconstructed_state {
            let materialized_refs = if let Some(cache) = session_cache {
                self.materialize_virtual_refs_into_cache(meta, materialized_virtuals, cache);
                cache.refs.clone()
            } else {
                self.materialize_virtual_refs(meta, materialized_virtuals)
            };
            let total_frames = reconstructed_state.frames.len();

            // Multi-frame caller-stack reconstruction: push outer frames onto
            // the interpreter's call stack and restore only the innermost as
            // current state. This mirrors RPython's resume.py which rebuilds
            // the full framestack so blackhole can unwind through callers.
            if total_frames > 1 && self.supports_multi_frame_restore() {
                let restored = self.restore_multi_frame_with_caller_stack(
                    meta,
                    reconstructed_state,
                    frame_layouts,
                    exception,
                    &materialized_refs,
                );
                if restored {
                    self.replay_pending_field_writes(
                        meta,
                        pending_field_writes,
                        &materialized_refs,
                    );
                    return true;
                }
                // Fall through to existing paths if multi-frame push failed
            }

            let can_use_generic_cache = total_frames == 1
                || reconstructed_state
                    .frames
                    .iter()
                    .enumerate()
                    .any(|(frame_index, frame)| {
                        frame.slot_types.as_ref().is_some()
                            || self
                                .reconstructed_frame_value_types_with_metadata(
                                    meta,
                                    frame_index,
                                    total_frames,
                                    frame,
                                )
                                .is_some()
                            || frame_layouts
                                .and_then(|layouts| layouts.get(frame_index))
                                .filter(|layout| {
                                    layout.pc == frame.pc
                                        && layout.slot_layouts.len() == frame.values.len()
                                })
                                .and_then(|layout| layout.slot_types.as_ref())
                                .is_some()
                    });

            let restored = if can_use_generic_cache {
                let restored = self.try_restore_reconstructed_frames_with_cache_and_layout(
                    meta,
                    reconstructed_state,
                    exception,
                    &materialized_refs,
                    frame_layouts,
                );
                if restored {
                    true
                } else {
                    self.restore_reconstructed_frames_with_resume_layout(
                        meta,
                        reconstructed_state,
                        frame_layouts,
                        materialized_virtuals,
                        exception,
                    )
                }
            } else {
                self.restore_reconstructed_frames_with_resume_layout(
                    meta,
                    reconstructed_state,
                    frame_layouts,
                    materialized_virtuals,
                    exception,
                )
            };

            if !restored {
                return false;
            }
            self.replay_pending_field_writes(meta, pending_field_writes, &materialized_refs);
            return true;
        }

        self.restore(meta, fail_values);
        true
    }

    /// Multi-frame caller-stack reconstruction.
    ///
    /// Pushes outer frames (outermost first) onto the interpreter's call
    /// stack via `push_caller_frame`, then restores the innermost frame as
    /// the current interpreter state. This allows blackhole execution to
    /// unwind through the reconstructed caller frames.
    fn restore_multi_frame_with_caller_stack(
        &mut self,
        meta: &Self::Meta,
        reconstructed_state: &ReconstructedState,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
        exception: &ExceptionState,
        materialized_refs: &[Option<GcRef>],
    ) -> bool {
        let total_frames = reconstructed_state.frames.len();
        if total_frames < 2 {
            return false;
        }

        // Push outer frames (indices 0..total_frames-1) onto the call stack.
        // Frame 0 is the outermost caller, frame total_frames-2 is the
        // direct caller of the innermost frame.
        for frame_index in 0..total_frames - 1 {
            let frame = &reconstructed_state.frames[frame_index];
            let types =
                self.resolve_frame_types(meta, frame_index, total_frames, frame, frame_layouts);
            let Some(types) = types else {
                return false;
            };
            let Some(values) =
                frame_values_from_reconstructed(&frame.values, &types, materialized_refs)
            else {
                return false;
            };
            if !self.push_caller_frame(
                meta,
                frame_index,
                total_frames,
                &values,
                frame.pc,
                frame.jitcode_index,
            ) {
                return false;
            }
        }

        // Restore the innermost frame as current state.
        let innermost_index = total_frames - 1;
        let innermost = &reconstructed_state.frames[innermost_index];
        let types = self.resolve_frame_types(
            meta,
            innermost_index,
            total_frames,
            innermost,
            frame_layouts,
        );
        let Some(types) = types else {
            return false;
        };
        let Some(values) =
            frame_values_from_reconstructed(&innermost.values, &types, materialized_refs)
        else {
            return false;
        };
        self.restore_reconstructed_frame_values_with_metadata(
            meta,
            innermost_index,
            total_frames,
            innermost,
            &values,
            exception,
        )
    }

    /// Resolve the type layout for a single reconstructed frame, trying
    /// (in order): per-frame slot_types, interpreter-provided types,
    /// and resume layout types.
    fn resolve_frame_types(
        &self,
        meta: &Self::Meta,
        frame_index: usize,
        total_frames: usize,
        frame: &crate::resume::ReconstructedFrame,
        frame_layouts: Option<&[ResumeFrameLayoutSummary]>,
    ) -> Option<Vec<Type>> {
        frame
            .slot_types
            .clone()
            .or_else(|| {
                self.reconstructed_frame_value_types_with_metadata(
                    meta,
                    frame_index,
                    total_frames,
                    frame,
                )
            })
            .or_else(|| {
                frame_layouts
                    .and_then(|layouts| layouts.get(frame_index))
                    .filter(|layout| {
                        layout.pc == frame.pc && layout.slot_layouts.len() == frame.values.len()
                    })
                    .and_then(|layout| layout.slot_types.clone())
            })
    }

    fn replay_pending_field_writes(
        &mut self,
        meta: &Self::Meta,
        pending_field_writes: &[ResolvedPendingFieldWrite],
        materialized_refs: &[Option<GcRef>],
    ) {
        for pending in pending_field_writes {
            let Some(layout) = self.pending_field_write_layout(
                meta,
                pending.descr_index,
                pending.item_index.is_some(),
            ) else {
                continue;
            };
            let Some(target) = materialized_value_to_i64(&pending.target, materialized_refs) else {
                continue;
            };
            let Some(value) = materialized_value_to_i64(&pending.value, materialized_refs) else {
                continue;
            };
            unsafe {
                apply_pending_field_write(layout, target as usize, pending.item_index, value)
            };
        }
    }
}

/// Outcome of virtualizable synchronization around a residual call.
///
/// `updated_fields` covers nonstandard/heap-based re-reads.
/// `forced` mirrors the standard virtualizable token protocol: when true,
/// tracing should abort instead of recording `GUARD_NOT_FORCED`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidualVirtualizableSync {
    pub updated_fields: Vec<(u32, OpRef)>,
    pub forced: bool,
}

fn frame_values_from_reconstructed(
    frame_values: &[crate::resume::ReconstructedValue],
    types: &[Type],
    materialized_refs: &[Option<GcRef>],
) -> Option<Vec<Value>> {
    if frame_values.len() != types.len() {
        return None;
    }
    let mut result = Vec::with_capacity(frame_values.len());
    for (value, expected) in frame_values.iter().zip(types.iter().copied()) {
        let decoded = match (value, expected) {
            (crate::resume::ReconstructedValue::Value(raw), Type::Int) => Value::Int(*raw),
            (crate::resume::ReconstructedValue::Value(raw), Type::Ref) => {
                Value::Ref(GcRef(*raw as usize))
            }
            (crate::resume::ReconstructedValue::Value(raw), Type::Float) => {
                Value::Float(f64::from_bits(*raw as u64))
            }
            (crate::resume::ReconstructedValue::Value(_), Type::Void) => Value::Void,
            (crate::resume::ReconstructedValue::Virtual(index), _) => {
                // Virtual objects always materialize as Ref, regardless of the
                // expected type. The interpreter's restore_values() knows to
                // treat these as object references.
                Value::Ref(materialized_refs.get(*index).copied().flatten()?)
            }
            (crate::resume::ReconstructedValue::Uninitialized, _) => Value::Void,
            (crate::resume::ReconstructedValue::Unavailable, _) => Value::Void,
        };
        result.push(decoded);
    }
    Some(result)
}

fn materialized_value_to_i64(
    value: &MaterializedValue,
    materialized_refs: &[Option<GcRef>],
) -> Option<i64> {
    match value {
        MaterializedValue::Value(value) => Some(*value),
        MaterializedValue::VirtualRef(index) => materialized_refs
            .get(*index)
            .copied()
            .flatten()
            .map(|gc_ref| gc_ref.as_usize() as i64),
    }
}

unsafe fn apply_pending_field_write(
    layout: PendingFieldWriteLayout,
    target: usize,
    item_index: Option<usize>,
    value: i64,
) {
    unsafe {
        match layout {
            PendingFieldWriteLayout::Field { offset, value_type } => {
                let ptr = (target as *mut u8).add(offset);
                write_typed_value(ptr, value, value_type);
            }
            PendingFieldWriteLayout::ArrayItem {
                base_offset,
                item_size,
                item_type,
            } => {
                let index = item_index.unwrap_or(0);
                let ptr = (target as *mut u8).add(base_offset + index * item_size);
                write_typed_value(ptr, value, item_type);
            }
        }
    }
}

unsafe fn write_typed_value(ptr: *mut u8, value: i64, value_type: Type) {
    unsafe {
        match value_type {
            Type::Int => (ptr as *mut i64).write(value),
            Type::Ref => (ptr as *mut usize).write(value as usize),
            Type::Float => (ptr as *mut f64).write(f64::from_bits(value as u64)),
            Type::Void => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::{MaterializedValue, MaterializedVirtual};
    use std::cell::Cell;

    /// Test state that tracks how many times `materialize_virtual_ref` was
    /// called, so we can verify the session cache prevents re-materialization.
    #[derive(Default)]
    struct CachingTestState {
        materialize_call_count: Cell<usize>,
        restored_values: Vec<Value>,
        restored_frames: Vec<(usize, u64, Vec<Value>)>,
    }

    impl JitState for CachingTestState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _: usize, _: &()) -> () {}
        fn extract_live(&self, _: &()) -> Vec<i64> {
            Vec::new()
        }
        fn create_sym(_: &(), _: usize) -> () {}
        fn is_compatible(&self, _: &()) -> bool {
            true
        }
        fn restore(&mut self, _: &(), _: &[i64]) {}

        fn restore_values(&mut self, _: &(), values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &(),
            frame_index: usize,
            _total_frames: usize,
            frame: &crate::resume::ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.restored_frames
                .push((frame_index, frame.pc, values.to_vec()));
            true
        }

        fn materialize_virtual_ref(
            &mut self,
            _meta: &(),
            virtual_index: usize,
            _materialized: &MaterializedVirtual,
        ) -> Option<GcRef> {
            self.materialize_call_count
                .set(self.materialize_call_count.get() + 1);
            Some(GcRef(0xB000 + virtual_index))
        }

        fn collect_jump_args(_: &()) -> Vec<OpRef> {
            Vec::new()
        }
        fn validate_close(_: &(), _: &()) -> bool {
            true
        }
    }

    #[test]
    fn test_deopt_materialization_cache_basic() {
        let mut cache = DeoptMaterializationCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);

        cache.insert(0, GcRef(0x100));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(0), Some(GcRef(0x100)));
        assert_eq!(cache.get(1), None);
        assert_eq!(cache.materialized_count(), 1);

        cache.insert(2, GcRef(0x200));
        assert_eq!(cache.len(), 3); // 0, 1(None), 2
        assert_eq!(cache.get(2), Some(GcRef(0x200)));
        assert_eq!(cache.materialized_count(), 2);

        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.materialized_count(), 0);
    }

    #[test]
    fn test_materialization_cache_persists_across_guard_not_forced() {
        let mut state = CachingTestState::default();

        let virtuals_session1 = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B into session cache
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_first = state.materialize_call_count.get();
        assert_eq!(calls_after_first, 2);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));

        // Second deopt (GUARD_NOT_FORCED resumption) in same session:
        // same virtuals should NOT be re-materialized
        state.materialize_virtual_refs_into_cache(&(), &virtuals_session1, &mut session_cache);

        let calls_after_second = state.materialize_call_count.get();
        assert_eq!(
            calls_after_second, calls_after_first,
            "virtuals should not be re-materialized from cache"
        );
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
    }

    #[test]
    fn test_materialization_cache_extends_for_new_virtuals() {
        let mut state = CachingTestState::default();

        let virtuals_first = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
        ];

        // First deopt: materialize A and B
        let mut session_cache = DeoptMaterializationCache::new();
        state.materialize_virtual_refs_into_cache(&(), &virtuals_first, &mut session_cache);
        assert_eq!(state.materialize_call_count.get(), 2);

        // Second deopt adds virtual C
        let virtuals_second = vec![
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 1,
                descr_index: 0,
                fields: vec![(0, MaterializedValue::Value(100))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 2,
                descr_index: 1,
                fields: vec![(0, MaterializedValue::Value(200))],
            },
            MaterializedVirtual::Obj {
                descr: None,
                type_id: 3,
                descr_index: 2,
                fields: vec![(0, MaterializedValue::Value(300))],
            },
        ];

        state.materialize_virtual_refs_into_cache(&(), &virtuals_second, &mut session_cache);

        // Only 1 new call for virtual C (A and B were cached)
        assert_eq!(state.materialize_call_count.get(), 3);
        assert_eq!(session_cache.get(0), Some(GcRef(0xB000)));
        assert_eq!(session_cache.get(1), Some(GcRef(0xB001)));
        assert_eq!(session_cache.get(2), Some(GcRef(0xB002)));
    }

    #[test]
    fn test_virtual_ref_forcing_uses_session_cache() {
        let mut state = CachingTestState::default();
        let mut cache = DeoptMaterializationCache::new();

        let virt = MaterializedVirtual::Obj {
            descr: None,
            type_id: 1,
            descr_index: 0,
            fields: vec![(0, MaterializedValue::Value(42))],
        };

        // First call: materializes and caches
        let result1 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result1, Some(GcRef(0xB000)));
        assert_eq!(state.materialize_call_count.get(), 1);

        // Second call: returns cached value without materializing
        let result2 = state.force_virtual_with_cache(&(), &mut cache, 0, &virt);
        assert_eq!(result2, Some(GcRef(0xB000)));
        assert_eq!(
            state.materialize_call_count.get(),
            1,
            "should not re-materialize cached virtual"
        );
    }
}
