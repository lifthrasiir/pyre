//! Tracing context: wraps Trace + ConstantPool with convenience API.
//!
//! `TraceCtx` owns the struct definition, constructors, accessors,
//! constant management, and virtualizable machinery.  The recording
//! and compile-bookkeeping roles are split across sibling modules:
//!
//! * **History role** → `history.rs` `impl TraceCtx`:
//!   `record_*`, `get_trace_position`, `cut_trace`, `replace_box`,
//!   `into_tree_loop`, all `call_*` / `guard_*` recording wrappers.
//!
//! * **Compile role** → `compile.rs` `impl TraceCtx`:
//!   `add/clear/get/has_merge_point*`,
//!   inline-trace tracking (`push/pop_inline_*`, `recursive_depth`).
//!
//! `MergePoint` is defined here alongside `current_merge_points`,
//! matching RPython where `MetaInterp` (pyjitpl.py) owns both.
//!
//! Remaining convergence: reshape `MetaInterp` fields from
//! `meta.trace_ctx` to `meta.history` + `meta.trace` (upstream
//! parity); that cascades into every call site of `TraceCtx::*`.

use crate::opencoder::Box as OcBox;
use crate::recorder::Trace;
use majit_ir::{DescrRef, GreenKey, OpCode, OpRef, Type, Value};
use majit_trace::heapcache::HeapCache;

use majit_backend::JitCellToken;

use crate::constant_pool::ConstantPool;
use crate::jitcode::JitArgKind;
// `make_resume_guard_descr*` is no longer needed at the tracer side —
// guards record `descr=None` and the optimizer's
// `store_final_boxes_in_guard` mints the descr (codex #3 / pyjitpl.py:2548
// generate_guard parity).
use crate::jitdriver::JitDriverStaticData;
use crate::virtualizable::VirtualizableInfo;

/// Inverse of `heap_value_for`: encode a typed `Value` into the raw i64
/// bit-pattern that `VirtualizableInfo::write_field`/`write_array_item`
/// interpret per field/item type.
pub(crate) fn value_to_raw_bits(value: Value) -> i64 {
    match value {
        Value::Int(v) => v,
        Value::Float(f) => f.to_bits() as i64,
        Value::Ref(r) => r.as_usize() as i64,
        Value::Void => 0,
    }
}

/// pyjitpl.py:1135-1138 `rop.PTR_EQ` runtime outcome.  Compare the
/// concrete ptrs carried by two Refs (virtualizable identity).  Non-Ref
/// values are never compared as virtualizable boxes, so falling into the
/// catch-all `false` branch preserves the Step 4 "not standard" path in
/// `is_nonstandard_virtualizable` — this is how an `opref` backed by a
/// non-Ref constant (e.g. `ConstInt(0xCAFE)` in a test) still resolves to
/// `isstandard = 0` and proceeds to Step 5 / `emit_force_virtualizable`,
/// matching upstream's runtime behavior for the same bogus input.
fn concrete_ptrs_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Ref(ra), Value::Ref(rb)) => ra == rb,
        _ => false,
    }
}

/// pyjitpl.py:2989 — a visited loop header with its trace position.
///
/// RPython stores `(original_boxes, start)` where `original_boxes` is the
/// full list of green+red args at the first visit, and `start` is a 5-tuple
/// trace position. majit stores the equivalent as OpRef vectors + TracePosition.
#[derive(Clone, Debug)]
pub struct MergePoint {
    /// Green key of the loop header.
    pub green_key: u64,
    /// Trace position when this loop header was first visited.
    pub position: crate::recorder::TracePosition,
    /// pyjitpl.py:2989: original_boxes — live variable OpRefs at the first
    /// visit to this loop header. Used by compile_loop/compile_retrace as
    /// the inputargs for trace cutting.
    pub original_boxes: Vec<OpRef>,
    /// Types of original_boxes. RPython Box carries type implicitly;
    /// majit stores types alongside OpRefs for compile_retrace parity.
    pub original_box_types: Vec<Type>,
    /// Bytecode PC of this loop header. Used by cut_trace_from to update
    /// meta when the trace closes at a different loop.
    pub header_pc: usize,
}

/// Tracing context: wraps Trace + ConstantPool with convenience API.
///
/// The interpreter uses this during trace recording to:
/// - Record IR operations
/// - Manage constants (with deduplication)
/// - Record guards (with auto-generated FailDescr)
/// - Record function calls (with auto-generated CallDescr)
pub struct TraceCtx {
    pub(crate) recorder: Trace,
    /// opencoder.py:472 `self.metainterp_sd = metainterp_sd` — the trace
    /// recorder holds a shared reference to the JIT's static data so
    /// `_encode_descr` can route global descriptors through
    /// `metainterp_sd.all_descrs`. Pyre tracks it on TraceCtx instead
    /// of `recorder::Trace` because the swap to `TraceRecordBuffer`
    /// (Step 2e.2b) needs the Arc available at constructor time; wiring
    /// it at the TraceCtx layer lets the eventual swap reuse this
    /// plumbing without threading more parameters through
    /// `MetaInterp::setup_tracing` etc.
    pub(crate) metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    pub(crate) green_key: u64,
    root_green_key: u64,
    /// Structured `(code_ptr, pc)` counterpart to `green_key`. Keeps
    /// pyjitpl.py:1396-1401's element-wise `same_constant` parity when
    /// comparing against inline-frame green keys; the u64 `green_key`
    /// above is the hash derived from this pair and stays the identity
    /// key for HashMap lookups (warmstate / compiled_loops / pending
    /// token) while comparisons route through the raw pair.
    pub(crate) green_key_raw: (usize, usize),
    pub(crate) root_green_key_raw: (usize, usize),
    pub(crate) constants: ConstantPool,
    /// Stack of inlined function frames (callee green keys as raw
    /// `(code_ptr, pc)` pairs). rpython/jit/metainterp/pyjitpl.py:1390
    /// walks `self.metainterp.framestack` element-wise; pyre mirrors
    /// that by storing the structured greenkey per inline frame and
    /// doing tuple-equality comparisons in [`recursive_depth`] and
    /// [`is_tracing_key`].
    pub(crate) inline_frames: Vec<(usize, usize)>,
    /// Start positions for currently active inlined trace-through frames.
    ///
    /// This mirrors the subset of PyPy's `portal_trace_positions` that we
    /// need for `find_biggest_function()`: active inlined callees and the
    /// trace length at which each one started tracing.
    pub(crate) inline_trace_positions: Vec<(u64, usize)>,
    /// Structured green key values (if provided by the interpreter).
    green_key_values: Option<GreenKey>,
    /// Declarative driver layout metadata, if provided by the interpreter.
    pub(crate) driver_descriptor: Option<JitDriverStaticData>,
    /// Standard virtualizable boxes -- OpRefs for each static field + array element.
    /// When set, vable_getfield/setfield access these instead of emitting heap ops.
    /// Layout: [field_0, ..., field_N, arr_0[0], ..., arr_0[M], ..., vable_ref]
    ///
    /// The last element (`boxes[-1]`) is the standard virtualizable identity
    /// (RPython parity: `virtualizable_boxes[-1]`). Used by gen_store_back_in_vable
    /// to distinguish standard vs nonstandard virtualizable.
    pub(crate) virtualizable_boxes: Option<Vec<OpRef>>,
    /// Concrete shadow of `virtualizable_boxes`. Same layout, each slot carries
    /// the current runtime `Value` (RPython Box ≡ OpRef + concrete value).
    /// Seeded from `original_boxes` in `initialize_virtualizable` and kept in
    /// sync on every standard vable write (`vable_setfield`,
    /// `vable_setarrayitem_indexed`, `store_local_value` mirror).
    virtualizable_values: Option<Vec<Value>>,
    /// VirtualizableInfo for the standard virtualizable (if any).
    virtualizable_info: Option<std::sync::Arc<VirtualizableInfo>>,
    /// Lengths of each virtualizable array field, needed for flat index computation.
    virtualizable_array_lengths: Option<Vec<usize>>,
    /// Live virtualizable heap pointer (pyjitpl.py:3446 write_boxes target).
    /// Mirrored from `MetaInterp::vable_ptr` at trace/bridge-entry.  Used by
    /// `synchronize_virtualizable` to write `virtualizable_values` back to
    /// the live PyFrame after every standard vable setfield / setarrayitem
    /// (virtualizable.py:101 write_boxes parity). `None` disables the
    /// write — unit-test or init-before-run path.
    virtualizable_heap_ptr: Option<*const u8>,
    /// Header PC at which this trace started (0 = function entry).
    pub header_pc: usize,
    /// When a cross-loop cut occurs (trace closes at inner loop header),
    /// the green key for the inner loop. Used to register an alias
    /// so can_enter_jit at the inner back-edge finds the outer key's entry.
    pub cut_inner_green_key: Option<u64>,
    /// pyjitpl.py:3030 current_merge_points — loop headers visited during
    /// tracing with their trace positions. First visit records the key +
    /// position; second visit closes the loop.
    pub(crate) current_merge_points: Vec<MergePoint>,
    /// pyjitpl.py:2979 reached_loop_header parity: callback to check
    /// has_compiled_targets(ptoken) for a given green key. Bridge traces
    /// skip loop headers without compiled targets. Live lookup (not snapshot)
    /// matches RPython's get_procedure_token(greenboxes) + has_compiled_targets.
    pub has_compiled_targets_fn: Option<Box<dyn Fn(u64) -> bool>>,
    /// pyjitpl.py:2978 `if not self.partial_trace:` parity at
    /// `reached_loop_header` — explicit "this trace started from a
    /// guard failure" flag.  RPython distinguishes via
    /// `self.resumekey` typing (`ResumeGuardDescr` vs
    /// `ResumeFromInterpDescr`); pyre sets this to `true` at
    /// `start_bridge_tracing` and leaves the default `false` for
    /// primary entries.  Consumers that need bridge-only behavior
    /// (e.g. `pyre-jit-trace::metainterp::run_to_end`'s close-loop
    /// skip when no compiled targets exist for the current
    /// greenkey) gate on this flag instead of fn presence.
    pub is_bridge_trace: bool,
    /// pyjitpl.py:1551 `if self.metainterp.portal_call_depth: return` parity
    /// — live read of `MetaInterp.portal_call_depth` at the
    /// `BC_JIT_MERGE_POINT` first-iteration auto loop-header gate.  When
    /// nested portal calls are active (`portal_call_depth != 0`), RPython
    /// skips the auto-stamp and waits for an explicit `loop_header` op.
    /// Pyre exposes this as a Fn pointer so the trace ctx (which owns
    /// the cross-component flow at dispatch time) can sample the
    /// metainterp's depth counter without holding a back-reference.
    pub portal_call_depth_fn: Option<Box<dyn Fn() -> i32>>,
    /// pyjitpl.py: `metainterp.staticdata.callinfocollection`. Needed by
    /// `ResumeDataBoxReader.concat_strings` / `slice_string` / `concat_unicodes`
    /// / `slice_unicode` (resume.py:1143-1188) which look up the
    /// `OS_STR_CONCAT` / `OS_STR_SLICE` / `OS_UNI_CONCAT` / `OS_UNI_SLICE`
    /// calldescr + func pointers while rematerializing virtual strings
    /// during bridge-virtual reconstruction.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// pyjitpl.py:2398: tracing-time heap cache.
    /// Tracks field/array values, allocations, escape status, and class/nullity
    /// knowledge during tracing to avoid recording redundant operations.
    pub(crate) heap_cache: HeapCache,
    /// pyjitpl.py:2411 force_finish_trace: when True, trace is segmented
    /// at 80% of limit via _create_segmented_trace_and_blackhole.
    pub(crate) force_finish: bool,
    /// pyjitpl.py:2594 frame.pc: last bytecode pc passed to trace_fn.
    /// Used by force_finish_trace segmenting to record the guard-point pc.
    pub last_traced_pc: usize,
    /// GC-safe constant OpRefs for each initial inputarg at trace start.
    /// Each entry is a ConstantPool-allocated OpRef whose value is kept
    /// up-to-date by the shadow stack (Ref constants survive GC moves).
    /// Used by cut_trace_from to remap escaped original inputargs to
    /// existing pool constants, avoiding both stale pointers and
    /// entry-contract mismatches.
    pub initial_inputarg_consts: Vec<majit_ir::OpRef>,
    /// pyjitpl.py:1087 parity: quasi-immutable field read needs a
    /// GUARD_NOT_INVALIDATED with full snapshot at the field read's orgpc.
    /// Stores Some(orgpc) when pending.
    pending_guard_not_invalidated_pc: Option<usize>,
    /// pyjitpl.py:2394 `MetaInterp.forced_virtualizable` parity. Tracks the
    /// vbox handed to `gen_store_back_in_vable` so the second
    /// `opimpl_hint_force_virtualizable` of the same trace can be skipped.
    /// RPython resets this in `MetaInterp.__init__`; pyre keeps it on
    /// TraceCtx because TraceCtx is freshly created per trace and the
    /// MetaInterp is reused across traces.
    forced_virtualizable: Option<OpRef>,
    /// pyjitpl.py:2397: call_pure_results — maps constant argument tuples
    /// to their concrete result values, recorded during tracing.
    /// Passed to the optimizer for cross-iteration CALL_PURE folding.
    pub(crate) call_pure_results: std::collections::HashMap<Vec<Value>, Value>,
    /// Cached `warmstate.trace_limit` snapshot for this tracing session.
    /// pyjitpl.py:2789 reads `self.jitdriver_sd.warmstate.trace_limit` each
    /// call; pyre snapshots it at `setup_tracing` time (warmstate owns the
    /// live value). Default mirrors rlib/jit.py:592 (trace_limit = 6000).
    pub(crate) trace_limit: usize,
    /// Pyre-only snapshot side table (opencoder.py stores snapshots inline
    /// in `_snapshot_data` / `_snapshot_array_data` byte streams).
    /// `capture_resumedata` pushes one entry per guard; the returned id
    /// is stored on the guard op's `rd_resume_position`.  Grows
    /// monotonically across `cut_trace` (matches the pre-Task #70
    /// behavior — see `cut_trace` for rationale).  Will migrate to the
    /// byte-stream form carried by `TraceRecordBuffer` alongside the
    /// eventual field swap (Task #59 / #70).
    pub(crate) snapshots: Vec<crate::recorder::Snapshot>,
    /// pyjitpl.py:2898 `self.resumekey_original_loop_token = ...`.
    /// The source loop token of the bridge trace, populated at
    /// `start_retrace_from_guard` from the failed guard descr's
    /// `rd_loop_token`.  `None` for loop-entry traces (RPython
    /// `isinstance(self.resumekey, compile.ResumeFromInterpDescr)` is
    /// True).  Used by `prepare_trace_segmenting` (pyjitpl.py:2825-
    /// 2834) to set the `FORCE_BRIDGE_SEGMENTING` bit on the loop
    /// token when bridge tracing aborts without an inlinable function.
    pub(crate) resumekey_original_loop_token: Option<std::sync::Arc<JitCellToken>>,
}

/// rlib/jit.py:592 default `trace_limit` — mirrored here so standalone
/// TraceCtx construction (unit tests, `setup_tracing` before a warmstate
/// override) matches the RPython baseline.
pub const DEFAULT_TRACE_LIMIT: usize = 6000;

impl TraceCtx {
    /// opencoder.py:472 `self.metainterp_sd` — shared static data the
    /// recorder was constructed with. Read-only handle for callers that
    /// need to reach the per-process descr pools and terminal descrs
    /// (`done_with_this_frame_descr_*`,
    /// `exit_frame_with_exception_descr_ref`) without owning a separate
    /// reference.
    pub fn metainterp_sd(&self) -> &std::sync::Arc<crate::MetaInterpStaticData> {
        &self.metainterp_sd
    }

    /// pyjitpl.py:2398: access the tracing-time heap cache.
    pub fn heap_cache(&self) -> &HeapCache {
        &self.heap_cache
    }

    /// Mutable access to the tracing-time heap cache.
    pub fn heap_cache_mut(&mut self) -> &mut HeapCache {
        &mut self.heap_cache
    }

    /// pyjitpl.py:1087 parity: check if a quasi-immut guard is pending.
    pub fn pending_guard_not_invalidated_pc(&self) -> Option<usize> {
        self.pending_guard_not_invalidated_pc
    }

    /// Set pending quasi-immut guard with the field read's orgpc.
    pub fn set_pending_guard_not_invalidated(&mut self, pc: Option<usize>) {
        self.pending_guard_not_invalidated_pc = pc;
    }

    /// pyjitpl.py:1776-1780: jit.isvirtual(obj) — check if an object
    /// is likely virtual (allocated during this trace and not escaped).
    pub fn is_likely_virtual(&self, obj: OpRef) -> bool {
        self.heap_cache.is_likely_virtual(obj)
    }

    /// pyjitpl.py:1805-1806: record VIRTUAL_REF(box, cindex).
    /// `cindex` = ConstInt(len(virtualref_boxes) // 2) — pair index.
    /// The optimizer can later eliminate the vref if the object stays virtual.
    pub fn virtual_ref(&mut self, obj: OpRef, cindex: OpRef) -> OpRef {
        let result = Self::do_record_op(
            &mut self.recorder,
            &self.constants,
            OpCode::VirtualRefR,
            &[obj, cindex],
        );
        // pyjitpl.py:1807: heapcache.new(resbox)
        self.heap_cache.new_object(result);
        result
    }

    /// Create a standalone TraceCtx for testing or external use.
    ///
    /// Internally synthesizes a fresh `Arc<MetaInterpStaticData>` —
    /// test-only parity with `RPython test_opencoder.py:24` `class
    /// metainterp_sd: all_descrs = []` which similarly stubs a
    /// MetaInterpStaticData fixture for unit tests. Production callers
    /// (`MetaInterp::force_start_tracing` / `setup_tracing` /
    /// `start_bridge_trace`) go through `TraceCtx::new` directly with
    /// `self.staticdata.clone()`.
    pub fn for_test(num_inputs: usize) -> Self {
        let mut recorder = Trace::new();
        for _ in 0..num_inputs {
            recorder.record_input_arg(majit_ir::Type::Int);
        }
        Self::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// Create a TraceCtx for tests whose input args have mixed types.
    /// Analog of RPython `MetaInterp.create_empty_loop()` +
    /// `inputargs = [Box(tp) for tp in types]`.
    pub fn for_test_types(types: &[majit_ir::Type]) -> Self {
        let mut recorder = Trace::new();
        for &tp in types {
            recorder.record_input_arg(tp);
        }
        Self::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        )
    }

    /// Take the recorder out of this context (consumes self).
    pub fn into_recorder(self) -> Trace {
        self.recorder
    }

    pub(crate) fn new(
        recorder: Trace,
        green_key: u64,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        let initial_position = recorder.get_position();
        let initial_types: Vec<Type> = recorder.inputarg_types().to_vec();
        let initial_boxes: Vec<OpRef> = initial_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| OpRef::input_arg_typed(i as u32, tp))
            .collect();
        TraceCtx {
            recorder,
            metainterp_sd,
            green_key,
            root_green_key: green_key,
            green_key_raw: (0, 0),
            root_green_key_raw: (0, 0),
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            inline_trace_positions: Vec::new(),
            green_key_values: None,
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_values: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            header_pc: 0,
            cut_inner_green_key: None,
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_types,
                original_boxes: initial_boxes.clone(),
                header_pc: 0,
            }],
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            is_bridge_trace: false,
            portal_call_depth_fn: None,
            callinfocollection: None,
            call_pure_results: std::collections::HashMap::new(),
            trace_limit: DEFAULT_TRACE_LIMIT,
            snapshots: Vec::new(),
            resumekey_original_loop_token: None,
        }
    }

    /// Create a TraceCtx with a structured green key.
    pub(crate) fn with_green_key(
        recorder: Trace,
        green_key: u64,
        green_key_values: GreenKey,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) -> Self {
        let initial_position = recorder.get_position();
        // RPython pyjitpl.py:2878: initial merge point types come from
        // live_arg_boxes which carry actual types (INT/REF/FLOAT).
        let initial_input_types = recorder.inputarg_types();
        let initial_boxes: Vec<OpRef> = initial_input_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| OpRef::input_arg_typed(i as u32, tp))
            .collect();
        TraceCtx {
            recorder,
            metainterp_sd,
            green_key,
            root_green_key: green_key,
            green_key_raw: (0, 0),
            root_green_key_raw: (0, 0),
            constants: ConstantPool::new(),
            inline_frames: Vec::new(),
            inline_trace_positions: Vec::new(),
            green_key_values: Some(green_key_values),
            driver_descriptor: None,
            virtualizable_boxes: None,
            virtualizable_values: None,
            virtualizable_info: None,
            virtualizable_array_lengths: None,
            virtualizable_heap_ptr: None,
            header_pc: 0,
            cut_inner_green_key: None,
            current_merge_points: vec![MergePoint {
                green_key,
                position: initial_position,
                original_box_types: initial_input_types,
                original_boxes: initial_boxes.clone(),
                header_pc: 0,
            }],
            heap_cache: HeapCache::new(),
            force_finish: false,
            last_traced_pc: 0,
            initial_inputarg_consts: vec![],
            pending_guard_not_invalidated_pc: None,
            forced_virtualizable: None,
            has_compiled_targets_fn: None,
            is_bridge_trace: false,
            portal_call_depth_fn: None,
            callinfocollection: None,
            call_pure_results: std::collections::HashMap::new(),
            trace_limit: DEFAULT_TRACE_LIMIT,
            snapshots: Vec::new(),
            resumekey_original_loop_token: None,
        }
    }

    /// Get or create a constant OpRef for a given i64 value.
    pub fn const_int(&mut self, value: i64) -> OpRef {
        self.constants.get_or_insert(value)
    }

    /// Get or create a Ref-typed constant OpRef.
    /// executor.py:544 constant_from_op(op) parity: get typed Value for OpRef.
    pub fn constants_get_value(&self, opref: OpRef) -> Option<Value> {
        self.constants.get_value(opref)
    }

    /// RPython parity: Ref constants preserve their type so guard
    /// fail_args are correctly typed during guard failure recovery.
    pub fn const_ref(&mut self, value: i64) -> OpRef {
        self.constants
            .get_or_insert_typed(value, majit_ir::Type::Ref)
    }

    /// history.py:361 CONST_NULL = ConstPtr(ConstPtr.value).
    /// Ref-typed null pointer constant.
    pub fn const_null(&mut self) -> OpRef {
        self.const_ref(0)
    }

    /// Get or create a Float-typed constant OpRef.
    pub fn const_float(&mut self, value: i64) -> OpRef {
        self.constants
            .get_or_insert_typed(value, majit_ir::Type::Float)
    }

    /// Mark an existing constant OpRef with a specific type.
    /// Used when const_int() was called but the value is actually a Ref pointer
    /// (e.g., ob_type field). This preserves Cranelift's Int treatment while
    /// recording the true type for resume data.
    pub fn mark_const_type(&mut self, opref: OpRef, tp: majit_ir::Type) {
        self.constants.mark_type(opref, tp);
    }

    /// Return the type of a constant OpRef, if recorded.
    ///
    /// `numbering_type_overrides` takes priority over `constant_type`:
    /// `mark_type(opref, Ref)` exists specifically to retype a raw-pointer
    /// `ConstInt` as `Ref` for resume-data encoding, mirroring RPython's
    /// `getrawptrinfo` ConstInt→Ref retag at numbering time. The intrinsic
    /// `Box.type = 'i'` stays in `constant_type` but the snapshot consumer
    /// needs the override to produce the correct `TAGCONSTPTR`. Without
    /// checking the override first, `get_or_insert`'s `Type::Int` entry
    /// masks every `mark_type(_, Ref)` and Ref pointers get serialized as
    /// integer bits.
    pub fn const_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        self.constants
            .numbering_type_overrides()
            .get(&opref.raw())
            .copied()
            .or_else(|| self.constants.constant_type(opref))
    }

    /// Return the concrete value for a constant OpRef, if it is a pooled constant.
    pub fn const_value(&self, opref: OpRef) -> Option<i64> {
        self.constants.as_ref().get(&opref.raw()).copied()
    }

    /// Constant-fold a pure field read on a constant object pointer.
    /// If `obj` is a constant and `descr` is immutable, reads the field
    /// at runtime and returns the value as a constant OpRef.
    pub fn try_const_fold_pure_field(
        &mut self,
        obj: OpRef,
        descr: &dyn majit_ir::Descr,
    ) -> Option<OpRef> {
        if !descr.is_always_pure() {
            return None;
        }
        let obj_ptr = self.const_value(obj)? as usize;
        if obj_ptr == 0 {
            return None;
        }
        let fd = descr.as_field_descr()?;
        let offset = fd.offset();
        let field_size = fd.field_size();
        let value = unsafe {
            let base = obj_ptr as *const u8;
            match field_size {
                8 => *(base.add(offset) as *const i64),
                4 if fd.is_field_signed() => *(base.add(offset) as *const i32) as i64,
                4 => *(base.add(offset) as *const u32) as i64,
                _ => return None,
            }
        };
        Some(self.const_int(value))
    }

    /// M1 bridge: translate a pyre `OpRef` into the `opencoder::Box` that
    /// `TraceRecordBuffer::record_op(&[Box], descr)` expects.
    ///
    /// Pyre's OpRef uses the high bit (`>= 10_000` via `OpRef::from_const`)
    /// to mark constants and keeps their values in a side `ConstantPool`;
    /// RPython's opencoder takes concrete `Const{Int,Float,Ptr}` /
    /// `AbstractResOp` boxes and encodes them inline through the
    /// `_bigints` / `_floats` / `_refs` pools in `_encode`.
    ///
    /// This helper is the conversion point between the two worlds.  It
    /// unblocks M2 (routing `TraceCtx::record_*` through TraceRecordBuffer's
    /// Box-taking API) without touching any call site yet.
    ///
    /// Panics when a constant OpRef has no pool entry — that is a genuine
    /// invariant break (every `is_constant()` OpRef must have been minted
    /// via `ConstantPool::get_or_insert_typed` or friends).
    pub fn opref_to_box(&self, opref: OpRef) -> OcBox {
        if opref.is_constant() {
            let value = self
                .constants
                .get_value(opref)
                .unwrap_or_else(|| panic!("opref_to_box: constant {:?} not in pool", opref));
            match value {
                Value::Int(v) => OcBox::ConstInt(v),
                Value::Float(f) => OcBox::ConstFloat(f.to_bits()),
                Value::Ref(r) => OcBox::ConstPtr(r.as_usize() as u64),
                Value::Void => {
                    panic!("opref_to_box: constant {:?} has Void type", opref)
                }
            }
        } else {
            OcBox::of_op(opref)
        }
    }

    /// RPython `original_boxes[index]` lookup for the currently active trace.
    ///
    /// `MetaInterp.setup_tracing` snapshots each trace-entry concrete value in
    /// `initial_inputarg_consts`; the inputarg Box identity itself is still the
    /// ordinary `OpRef(index)`, matching RPython's `original_boxes` list.
    pub fn initial_inputarg_argbox(&self, index: usize) -> Option<(JitArgKind, OpRef, i64)> {
        let tp = self.recorder.inputarg_types().get(index).copied()?;
        let const_ref = self.initial_inputarg_consts.get(index).copied()?;
        let value = self.constants.get_value(const_ref)?;
        let bits = match value {
            Value::Int(v) => v,
            Value::Ref(r) => r.as_usize() as i64,
            Value::Float(v) => v.to_bits() as i64,
            Value::Void => 0,
        };
        let kind = match tp {
            Type::Int => JitArgKind::Int,
            Type::Ref => JitArgKind::Ref,
            Type::Float => JitArgKind::Float,
            Type::Void => return None,
        };
        // resoperation.py:719/727/739 InputArg{Int,Float,Ref}: the
        // inputarg Box carries `box.type` directly. Mint the typed
        // variant here so callers see the same {Int,Float,Ref} discrimination
        // RPython's original_boxes[index] would produce.
        Some((kind, OpRef::input_arg_typed(index as u32, tp), bits))
    }

    /// JitCode setup argbox for the standard virtualizable.
    ///
    /// This is the observer-mode counterpart of
    /// `pyjitpl.py:3271 f.setup_call(original_boxes)`: prefer the exact
    /// trace-entry red inputarg named by `jd.index_of_virtualizable`, and
    /// fall back to `virtualizable_boxes[-1]` only for legacy pyre traces that
    /// initialized the standard virtualizable before descriptor metadata was
    /// threaded through.
    pub fn standard_virtualizable_jitcode_argbox(&self) -> Option<(JitArgKind, OpRef, i64)> {
        if let Some(argbox) = self
            .driver_descriptor()
            .and_then(|driver| driver.virtualizable_arg_index())
            .and_then(|index| self.initial_inputarg_argbox(index))
        {
            return Some(argbox);
        }

        let opref = self.standard_virtualizable_box()?;
        let concrete = match self.standard_virtualizable_concrete()? {
            Value::Ref(r) => r.as_usize() as i64,
            Value::Int(v) => v,
            Value::Float(v) => v.to_bits() as i64,
            Value::Void => return None,
        };
        Some((JitArgKind::Ref, opref, concrete))
    }

    /// to a reached loop header during tracing.
    pub fn root_green_key(&self) -> u64 {
        self.root_green_key
    }

    /// Number of input arguments to the current trace.
    pub fn num_inputs(&self) -> usize {
        self.recorder.num_inputargs()
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.recorder.inputarg_types()
    }

    /// Number of traced operations recorded so far.
    pub fn num_ops(&self) -> usize {
        self.recorder.num_ops()
    }

    /// The structured green key values, if provided.
    pub fn green_key_values(&self) -> Option<&GreenKey> {
        self.green_key_values.as_ref()
    }

    /// Set the structured green key values.
    pub fn set_green_key_values(&mut self, values: GreenKey) {
        self.green_key_values = Some(values);
    }

    /// The declarative JitDriver descriptor, if provided.
    pub fn driver_descriptor(&self) -> Option<&JitDriverStaticData> {
        self.driver_descriptor.as_ref()
    }

    /// Attach declarative JitDriver metadata to the active trace.
    pub fn set_driver_descriptor(&mut self, descriptor: JitDriverStaticData) {
        self.driver_descriptor = Some(descriptor);
    }

    /// Initialize standard virtualizable boxes from input args.
    /// Called at trace start when a virtualizable is registered.
    ///
    /// `input_oprefs` / `input_values` contain one (OpRef, Value) pair per
    /// static field + array element in the same flat layout as
    /// `VirtualizableInfo::get_index_in_array`. `vable_ref` / `vable_ref_value`
    /// are the OpRef and concrete of the virtualizable object (frame pointer).
    /// Boxes layout: [field0, ..., fieldN, arr[0], ..., arr[M], vable_ref]
    /// where `boxes[-1]` is the standard virtualizable identity (RPython parity).
    pub fn init_virtualizable_boxes(
        &mut self,
        info: &VirtualizableInfo,
        vable_ref: OpRef,
        vable_ref_value: Value,
        input_oprefs: &[OpRef],
        input_values: &[Value],
        array_lengths: &[usize],
    ) {
        let mut boxes = input_oprefs.to_vec();
        boxes.push(vable_ref); // RPython: virtualizable_boxes[-1] = vable identity
        self.virtualizable_boxes = Some(boxes);
        if input_values.is_empty() {
            // Caller has no live concrete values (e.g. bridge-entry rebuild
            // helper in pyre-jit-trace::state::seed_virtualizable_boxes).
            // Disable the concrete shadow; `virtualizable_entry_at` will
            // return None and readers fall back to the zero placeholder,
            // same as the pre-concrete-shadow state.
            self.virtualizable_values = None;
        } else {
            assert_eq!(
                input_oprefs.len(),
                input_values.len(),
                "init_virtualizable_boxes: OpRef and Value slices must match",
            );
            let mut values = input_values.to_vec();
            values.push(vable_ref_value);
            self.virtualizable_values = Some(values);
        }
        self.virtualizable_info = Some(std::sync::Arc::new(info.clone()));
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Collect the current virtualizable boxes (for close_loop / finish).
    /// Returns `None` if no standard virtualizable is active.
    pub fn collect_virtualizable_boxes(&self) -> Option<Vec<OpRef>> {
        self.virtualizable_boxes.clone()
    }

    // (synchronize_virtualizable helper follows)

    /// Mirror of `MetaInterp::vable_ptr` used by `synchronize_virtualizable`.
    /// Callers set this at trace/bridge-entry so writes to
    /// `virtualizable_values` can propagate to the live PyFrame without
    /// routing back through MetaInterp (pyjitpl.py:3446 write_boxes target).
    pub fn set_virtualizable_heap_ptr(&mut self, ptr: *const u8) {
        self.virtualizable_heap_ptr = if ptr.is_null() { None } else { Some(ptr) };
    }

    /// pyjitpl.py:3446-3450 `synchronize_virtualizable()`.
    ///
    /// Writes the concrete half of `virtualizable_boxes` (the
    /// `virtualizable_values` shadow) back to the live virtualizable via
    /// `VirtualizableInfo::write_all_boxes`. The trailing identity slot
    /// (`virtualizable_boxes[-1]`) is excluded — RPython's `write_boxes`
    /// stops at `self.num_arrays + self.static_fields.len()` and leaves the
    /// identity untouched. No-op when the heap pointer, `virtualizable_info`,
    /// or `virtualizable_values` is unavailable.
    pub(crate) fn synchronize_virtualizable(&self) {
        let Some(heap_ptr) = self.virtualizable_heap_ptr else {
            return;
        };
        let Some(info) = self.virtualizable_info.as_ref() else {
            return;
        };
        let Some(values) = self.virtualizable_values.as_ref() else {
            return;
        };
        let Some(lengths) = self.virtualizable_array_lengths.as_ref() else {
            return;
        };
        let static_count = info.num_static_extra_boxes;
        if values.len() < static_count {
            return;
        }
        let mut static_bits: Vec<i64> = Vec::with_capacity(static_count);
        for v in &values[..static_count] {
            static_bits.push(value_to_raw_bits(*v));
        }
        let mut array_bits: Vec<Vec<i64>> = Vec::with_capacity(lengths.len());
        let mut cursor = static_count;
        for &len in lengths {
            if cursor + len > values.len() {
                return;
            }
            let mut items: Vec<i64> = Vec::with_capacity(len);
            for v in &values[cursor..cursor + len] {
                items.push(value_to_raw_bits(*v));
            }
            array_bits.push(items);
            cursor += len;
        }
        // Safety: `heap_ptr` is cached at trace/bridge entry from
        // `MetaInterp::vable_ptr`, which the JitState pins for the trace
        // session's lifetime. `write_all_boxes` uses typed offsets derived
        // from the same VirtualizableInfo used at the matching heap read.
        unsafe {
            info.write_all_boxes(heap_ptr as *mut u8, &static_bits, &array_bits);
        }
    }

    /// Read a standard virtualizable box by flat index.
    ///
    /// The last slot is the standard virtualizable identity itself
    /// (`virtualizable_boxes[-1]` in RPython terms).
    pub fn virtualizable_box_at(&self, index: usize) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.get(index).copied())
    }

    /// Read a standard virtualizable slot as (OpRef, concrete Value) — RPython
    /// `virtualizable_boxes[index]` parity: a Box carries both the traced
    /// reference and its concrete value. Callers that need to seed a register
    /// with both halves of the Box (e.g. `BC_GETARRAYITEM_VABLE_R` →
    /// `set_ref_reg`) MUST use this instead of `virtualizable_box_at`.
    pub fn virtualizable_entry_at(&self, index: usize) -> Option<(OpRef, Value)> {
        let boxes = self.virtualizable_boxes.as_ref()?;
        let values = self.virtualizable_values.as_ref()?;
        let opref = *boxes.get(index)?;
        let value = *values.get(index)?;
        Some((opref, value))
    }

    /// Declared majit_ir::Type for a flat virtualizable slot.
    ///
    /// Mirrors the layout used by `initialize_virtualizable`: the first
    /// `num_static_extra_boxes` slots take their types from
    /// `VirtualizableInfo.static_fields[i].field_type`, subsequent array
    /// slots take `array_fields[a].item_type`, and the trailing identity
    /// slot (`virtualizable_boxes[-1]`) is always `Ref`.  Returns `None`
    /// when no VirtualizableInfo is registered or the index falls outside
    /// the active layout.
    pub fn virtualizable_slot_type(&self, flat_idx: usize) -> Option<Type> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_deref().unwrap_or(&[]);
        let total_array: usize = lengths.iter().sum();
        let static_count = info.num_static_extra_boxes;
        if flat_idx < static_count {
            return Some(info.static_fields[flat_idx].field_type);
        }
        let array_local_idx = flat_idx - static_count;
        if array_local_idx < total_array {
            let mut remaining = array_local_idx;
            for (a, &len) in lengths.iter().enumerate() {
                if remaining < len {
                    return Some(info.array_fields[a].item_type);
                }
                remaining -= len;
            }
        }
        if flat_idx == static_count + total_array {
            // virtualizable_boxes[-1] — the identity slot.
            return Some(Type::Ref);
        }
        None
    }

    /// Update a standard virtualizable box (OpRef) by flat index.
    ///
    /// Used by SameAs dedup / `replace_box` walks — SSA-rename operations that
    /// do NOT change the concrete value carried by the slot. For updates that
    /// also change concrete (vable set{field,arrayitem}), use
    /// `set_virtualizable_entry_at`.
    pub fn set_virtualizable_box_at(&mut self, index: usize, value: OpRef) -> bool {
        if let Some(boxes) = &mut self.virtualizable_boxes {
            if let Some(slot) = boxes.get_mut(index) {
                *slot = value;
                return true;
            }
        }
        false
    }

    /// Update both halves of a standard virtualizable slot (OpRef + concrete).
    ///
    /// pyjitpl.py:1237 parity:
    ///
    /// ```text
    ///     self.metainterp.virtualizable_boxes[index] = valuebox
    ///     self.metainterp.synchronize_virtualizable()
    /// ```
    ///
    /// Writes the entire Box (SSA identity + concrete value) atomically so
    /// the (OpRef, concrete) pair never diverges.  Callers must ensure
    /// `value.get_type()` matches the slot's declared type
    /// (`virtualizable_slot_type(index)`); RPython guarantees this at the
    /// source level by emitting `NEW_W_INT` / `NEW_W_FLOAT` before any
    /// STORE into a Ref-typed `locals_cells_stack_w` slot
    /// (pypy/interpreter/pyframe.py:84 `list[W_Object]`).  Pyre's codewriter
    /// does not yet mirror that boxing at STORE_FAST → vable (Phase 4-5 of
    /// the portal-locals lowering plan); until it does, non-Phase-D paths
    /// like `pyre::trace_opcode::store_local_value` may write a pyre-unboxed
    /// `Value::Int`/`Value::Float` into a Ref slot and a later
    /// `BC_GETARRAYITEM_VABLE_R` read will decode 0 via `value_as_ref_bits`.
    /// That null is a pyre-upstream parity gap, not a shadow bug — the
    /// shadow faithfully reflects the caller's Box.
    pub fn set_virtualizable_entry_at(&mut self, index: usize, opref: OpRef, value: Value) {
        let (boxes_opt, values_opt) = (
            &mut self.virtualizable_boxes,
            &mut self.virtualizable_values,
        );
        let boxes = boxes_opt
            .as_mut()
            .expect("set_virtualizable_entry_at: virtualizable_boxes missing");
        let values = values_opt
            .as_mut()
            .expect("set_virtualizable_entry_at: virtualizable_values missing");
        assert_eq!(
            boxes.len(),
            values.len(),
            "set_virtualizable_entry_at: boxes/values length mismatch",
        );
        assert!(
            index < boxes.len(),
            "set_virtualizable_entry_at: index {index} out of range for {} slots",
            boxes.len(),
        );
        boxes[index] = opref;
        values[index] = value;
    }

    /// Return the standard virtualizable identity (`virtualizable_boxes[-1]`).
    pub fn standard_virtualizable_box(&self) -> Option<OpRef> {
        self.virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied())
    }

    /// Length of the symbolic virtualizable shadow, or `None` when no
    /// virtualizable is bound.  Probe-only accessor used by the
    /// `MAJIT_PROBE_BRIDGE`-gated logging in pyre's bridge setup +
    /// `push_typed_value` to surface bound-check off-by-ones before
    /// `set_virtualizable_entry_at` panics.
    pub fn virtualizable_boxes_len(&self) -> Option<usize> {
        self.virtualizable_boxes.as_ref().map(|boxes| boxes.len())
    }

    /// Concrete shadow of the standard virtualizable — the raw heap pointer
    /// `standard_virtualizable_box` refers to. Parallels
    /// `MetaInterp.virtualizable_boxes[-1].getref_base()` at runtime; pyre
    /// keeps the shadow in `virtualizable_values[-1]` because `OpRef` alone
    /// cannot carry the concrete ptr through the tracer.  Used by
    /// `is_nonstandard_virtualizable` Step 4 to realize the runtime
    /// `isstandard = concrete_eq(box, standard_box)` compare that upstream
    /// pyjitpl.py:1135-1138 performs via `rop.PTR_EQ` +
    /// `implement_guard_value`.
    pub fn standard_virtualizable_concrete(&self) -> Option<Value> {
        self.virtualizable_values
            .as_ref()
            .and_then(|values| values.last().copied())
    }

    /// Best-effort concrete (runtime) value associated with an OpRef, from
    /// TraceCtx-local state.  Parallels upstream `box.getref_base()` /
    /// `box.getint()` / `box.getfloatstorage()` — in RPython each Box
    /// carries its own runtime concrete via the Box subclass; pyre's
    /// `OpRef` is opaque, so concrete must be reconstructed.
    ///
    /// Resolution order, mirroring the subclass dispatch upstream performs
    /// implicitly:
    ///   1. Constant OpRefs — read from the `ConstantPool` (value + type).
    ///   2. `standard_virtualizable_box()` — use the runtime shadow held in
    ///      `virtualizable_values[-1]`.
    ///   3. Fallback — a sentinel `Value::Ref(GcRef(usize::MAX))` that
    ///      never matches a real heap pointer, so PTR_EQ comparisons with
    ///      the standard vable resolve to "different" at trace time.  The
    ///      fallback is the conservative answer for OpRefs whose concrete
    ///      lives only on the interpreter stack and must be threaded in
    ///      by the external caller (Task #68): alias canonicalization for
    ///      those paths remains off until dispatch.rs supplies the real
    ///      concrete.
    pub fn concrete_of_opref(&self, opref: OpRef) -> Value {
        if opref.is_constant() {
            if let Some(raw) = self.constants.as_ref().get(&opref.raw()).copied() {
                // Constants always have their type pinned by the
                // pool's `get_or_insert*` paths (constant_pool.rs:101 /
                // 134); a constant present in the pool with no recorded
                // type is a structural bug.
                let tp = self.constants.constant_type(opref).unwrap_or_else(|| {
                    panic!(
                        "constant_pool.constant_type missing for constant OpRef({}) (raw={}): \
                         constant_pool.get_or_insert / get_or_insert_typed must populate \
                         constant_types in lockstep",
                        opref.raw(),
                        raw
                    )
                });
                return match tp {
                    Type::Int => Value::Int(raw),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Void => Value::Void,
                };
            }
        }
        if Some(opref) == self.standard_virtualizable_box() {
            if let Some(v) = self.standard_virtualizable_concrete() {
                return v;
            }
        }
        Value::Ref(majit_ir::GcRef(usize::MAX))
    }

    /// Whether standard virtualizable boxes are active.
    pub fn has_virtualizable_boxes(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Drop the tracing-time virtualizable_boxes mirror.
    ///
    /// Used at bridge entry: `init_symbolic` seeds the cache with OpRefs
    /// derived from the *parent* loop's `vable_array_base`, but the
    /// bridge owns a fresh inputarg stream (its own `OpRef::from_raw(0..N)` bound
    /// to parent-guard fail_args). Keeping the parent seed makes
    /// subsequent `vable_getarrayitem_*` / `vable_setarrayitem_*` reads
    /// return stale parent-loop OpRefs; clearing forces the vable path
    /// to fall through to the raw `GetarrayitemGc` / `SetarrayitemGc`
    /// (`ctx.has_virtualizable_boxes() == false` branch) until the
    /// bridge itself reseeds via resume data — matching
    /// rpython/jit/metainterp/pyjitpl.py:3400-3430 where the
    /// `virtualizable_boxes` are rebuilt from the guard's resume data
    /// before the bridge replays any vable op.
    pub fn clear_virtualizable_boxes(&mut self) {
        self.virtualizable_boxes = None;
    }

    /// Set virtualizable_boxes with VirtualizableInfo and array lengths.
    /// Used by bridge tracing where the boxes are reconstructed from
    /// resume data (pyjitpl.py:3400 rebuild_state_after_failure parity).
    ///
    /// `values` carries the concrete shadow that parallels `boxes`. Callers
    /// must pass the matching live values recovered from the guard's fail
    /// args; an empty `values` slice disables the concrete shadow for the
    /// duration of the bridge (only safe when the bridge does not execute
    /// any `BC_GET*_VABLE_*` opcodes that feed `set_*_reg`).
    pub fn set_virtualizable_boxes_with_info(
        &mut self,
        boxes: Vec<OpRef>,
        values: Vec<Value>,
        info: &VirtualizableInfo,
        array_lengths: &[usize],
    ) {
        if !values.is_empty() {
            assert_eq!(
                boxes.len(),
                values.len(),
                "set_virtualizable_boxes_with_info: boxes/values length mismatch",
            );
            self.virtualizable_values = Some(values);
        } else {
            self.virtualizable_values = None;
        }
        self.virtualizable_boxes = Some(boxes);
        self.virtualizable_info = Some(std::sync::Arc::new(info.clone()));
        self.virtualizable_array_lengths = Some(array_lengths.to_vec());
    }

    /// Canonical virtualizable metadata for the active standard virtualizable.
    pub fn virtualizable_info(&self) -> Option<&std::sync::Arc<VirtualizableInfo>> {
        self.virtualizable_info.as_ref()
    }

    /// Cached array lengths for the active standard virtualizable.
    pub fn virtualizable_array_lengths(&self) -> Option<&[usize]> {
        self.virtualizable_array_lengths.as_deref()
    }

    /// pyjitpl.py:2394 `forced_virtualizable` accessor.
    pub fn forced_virtualizable(&self) -> Option<OpRef> {
        self.forced_virtualizable
    }

    /// pyjitpl.py:1126-1127 / 3478 `forced_virtualizable` mutator.
    pub fn set_forced_virtualizable(&mut self, value: Option<OpRef>) {
        self.forced_virtualizable = value;
    }

    // ── hint API consumption (RPython annotator/codewriter equivalent) ──

    /// Consume `hint(frame, access_directly=True)` during tracing.
    ///
    /// RPython's annotator generates JitCode that bypasses heap ops for
    /// virtualizable fields. In majit, this initializes the standard
    /// virtualizable boxes model so that subsequent vable_getfield/setfield
    /// calls access boxes directly instead of emitting heap ops.
    ///
    /// Must be called after `init_virtualizable_boxes`.
    /// Returns `true` if standard access is now active.
    pub fn hint_access_directly(&self) -> bool {
        self.virtualizable_boxes.is_some()
    }

    /// Consume `hint(frame, fresh_virtualizable=True)` during tracing.
    ///
    /// Marks that the virtualizable was freshly allocated, so its token is
    /// guaranteed to be TOKEN_NONE. The tracer skips token-check preamble.
    /// No IR is emitted; this is a tracing-time optimization.
    pub fn hint_fresh_virtualizable(&mut self, _vable_opref: OpRef) {
        // No IR needed — the token is already NONE for fresh objects.
        // This hint prevents the tracer from emitting unnecessary
        // GuardValue(token, 0) at loop entry for freshly created frames.
    }

    /// pyjitpl.py:3222-3236 `MetaInterp.store_token_in_vable()`.
    ///
    /// ```text
    /// def store_token_in_vable(self):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is None:
    ///         return
    ///     vbox = self.virtualizable_boxes[-1]
    ///     if vbox is self.forced_virtualizable:
    ///         return # we already forced it by hand
    ///     # in case the force_token has not been recorded, record it here
    ///     # to make sure we know the virtualizable can be broken. However,
    ///     # the contents of the virtualizable should be generally correct
    ///     force_token = self.history.record0(rop.FORCE_TOKEN,
    ///                                        lltype.nullptr(llmemory.GCREF.TO))
    ///     self.history.record2(rop.SETFIELD_GC, vbox, force_token,
    ///                          None, descr=vinfo.vable_token_descr)
    ///     self.generate_guard(rop.GUARD_NOT_FORCED_2)
    /// ```
    pub fn store_token_in_vable_setfield(&mut self) -> bool {
        let info = match self.virtualizable_info.clone() {
            Some(info) => info,
            None => return false,
        };
        let vbox = match self.standard_virtualizable_box() {
            Some(b) => b,
            None => return false,
        };
        if self.forced_virtualizable == Some(vbox) {
            return false;
        }
        let force_token =
            Self::do_record_op(&mut self.recorder, &self.constants, OpCode::ForceToken, &[]);
        let token_descr = info.token_field_descr();
        self.vable_setfield_descr(vbox, force_token, token_descr);
        // pyjitpl.py:3236 self.generate_guard(rop.GUARD_NOT_FORCED_2)
        // is recorded by the caller via the proper guard generation
        // path (`MIFrame::generate_guard` in the pyre frontend) so the
        // guard captures fresh resumedata at the current framestack
        // position, matching RPython's gen_store_back_in_vable.
        true
    }

    /// pyjitpl.py:3465-3497 `MetaInterp.gen_store_back_in_vable(box)`.
    ///
    /// ```text
    /// def gen_store_back_in_vable(self, box):
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         # xxx only write back the fields really modified
    ///         vbox = self.virtualizable_boxes[-1]
    ///         if vbox is not box:
    ///             # ignore the hint on non-standard virtualizable
    ///             # specifically, ignore it on a virtual
    ///             return
    ///         if self.forced_virtualizable is not None:
    ///             # this can happen only in strange cases, but we don't care
    ///             # it was already forced
    ///             return
    ///         self.forced_virtualizable = vbox
    ///         ...emit SETFIELD_GC for each static field...
    ///         ...emit SETARRAYITEM_GC for each array item...
    ///         ...emit final SETFIELD_GC(vbox, NULL, vable_token_descr)...
    /// ```
    pub fn gen_store_back_in_vable(&mut self, vable_opref: OpRef) {
        let (info, boxes, lengths) = match (
            self.virtualizable_info.clone(),
            self.virtualizable_boxes.clone(),
            self.virtualizable_array_lengths.clone(),
        ) {
            (Some(info), Some(boxes), Some(lengths)) => (info, boxes, lengths),
            _ => return,
        };

        // pyjitpl.py:3469 vbox = self.virtualizable_boxes[-1]
        // pyjitpl.py:3470-3473 if vbox is not box: return  (ignore nonstandard)
        if boxes.last().copied() != Some(vable_opref) {
            return;
        }

        // pyjitpl.py:3474-3477 if forced_virtualizable is not None: return
        if self.forced_virtualizable.is_some() {
            return;
        }
        // pyjitpl.py:3478 self.forced_virtualizable = vbox
        self.forced_virtualizable = Some(vable_opref);

        for field_index in 0..info.static_fields.len() {
            if let Some(&value) = boxes.get(field_index) {
                let descr = info.static_field_descr(field_index);
                self.vable_setfield_descr(vable_opref, value, descr);
            }
        }

        let mut flat_box_index = info.static_fields.len();
        for array_index in 0..info.array_fields.len() {
            let len = lengths.get(array_index).copied().unwrap_or(0);
            let field_descr = info.array_pointer_field_descr(array_index);
            let array_descr = info.array_item_descr(array_index);
            let array_ref = self.vable_getfield_ref_descr(vable_opref, field_descr);
            for item_index in 0..len {
                if let Some(&value) = boxes.get(flat_box_index) {
                    let index = self.const_int(item_index as i64);
                    self.vable_setarrayitem_descr(array_ref, index, value, array_descr.clone());
                }
                flat_box_index += 1;
            }
        }

        let null = self.const_int(0);
        self.vable_setfield_descr(vable_opref, null, info.token_field_descr());
    }

    /// Emit `SetfieldGc` for every static field and
    /// `GetfieldGcR` + `SetarrayitemGc` for every array item of the
    /// virtualizable referenced by `vable_opref`, mirroring
    /// `pyjitpl.py:3479-3494` field/array writes but without the
    /// `gen_store_back_in_vable` one-shot bookkeeping (no
    /// `forced_virtualizable` guard, no terminating
    /// `vable_token = NULL` write).
    ///
    /// Companion to `compile.py:425-461 patch_new_loop_to_load_
    /// virtualizable_fields`: when a JUMP target's label is reduced to
    /// reds-only, the patched loop preamble re-loads vable static and
    /// array fields from the heap on every iteration, so the heap MUST
    /// be source-of-truth before the JUMP. Tracing-time symbolic
    /// updates (`sym.vable_*` / `sym.registers_r[..nlocals]`) do not
    /// touch the heap; this helper closes the gap.
    ///
    /// Reads `self.virtualizable_boxes` for the values to write; that
    /// vector mirrors `[static_field_0, ..., array_0_item_0, ...,
    /// vable_box_identity]`. Caller is responsible for ensuring the
    /// boxes vector is up-to-date with the trace's current state at the
    /// point of the call (Slice 3 of the Task #21 fix plan).
    ///
    /// Skips emission when the active vable is non-standard or when
    /// virtualizable infrastructure is absent. Safe to call multiple
    /// times — unlike `gen_store_back_in_vable`, repeat calls re-emit
    /// the writebacks rather than no-op'ing.
    ///
    /// Dormant — wired up by Slice 2 (`close_loop_args_at` invocation
    /// behind the `inputarg_types.len() <= 1 + NUM_EXTRA_REDS` gate so
    /// descriptor=None paths see no new emission).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn gen_writeback_vable_to_heap(&mut self, vable_opref: OpRef) {
        let (info, boxes, lengths) = match (
            self.virtualizable_info.clone(),
            self.virtualizable_boxes.clone(),
            self.virtualizable_array_lengths.clone(),
        ) {
            (Some(info), Some(boxes), Some(lengths)) => (info, boxes, lengths),
            _ => return,
        };

        // pyjitpl.py:3469 `vbox = self.virtualizable_boxes[-1]` — same
        // standard-vable filter as `gen_store_back_in_vable` so this
        // helper is safe to call from a generic JUMP/exit emitter
        // without the caller having to re-check the vbox identity.
        if boxes.last().copied() != Some(vable_opref) {
            return;
        }

        for field_index in 0..info.static_fields.len() {
            if let Some(&value) = boxes.get(field_index) {
                let descr = info.static_field_descr(field_index);
                self.vable_setfield_descr(vable_opref, value, descr);
            }
        }

        let mut flat_box_index = info.static_fields.len();
        for array_index in 0..info.array_fields.len() {
            let len = lengths.get(array_index).copied().unwrap_or(0);
            let field_descr = info.array_pointer_field_descr(array_index);
            let array_descr = info.array_item_descr(array_index);
            let array_ref = self.vable_getfield_ref_descr(vable_opref, field_descr);
            for item_index in 0..len {
                if let Some(&value) = boxes.get(flat_box_index) {
                    let index = self.const_int(item_index as i64);
                    self.vable_setarrayitem_descr(array_ref, index, value, array_descr.clone());
                }
                flat_box_index += 1;
            }
        }
    }

    /// `compile.py:425-461 patch_new_loop_to_load_virtualizable_fields`
    /// mirrored at the call site instead of the callee preamble.
    ///
    /// Emits `GETFIELD_GC` for every static field and `GETFIELD_GC_R`
    /// + `GETARRAYITEM_GC` for every array item of the virtualizable
    /// referenced by `vable`. Returns the freshly recorded OpRefs in
    /// `[scalar_0, ..., scalar_{N-1}, array_0_item_0, ...,
    /// array_K_item_M]` order — matching `VableExpansion`'s slot
    /// layout (excluding the leading frame ref at slot 0).
    ///
    /// `array_lengths[i]` is the live element count of the i-th array
    /// field, mirroring `vinfo.get_array_length(vable, arrayindex)`
    /// at compile.py:443. The caller is expected to have read these
    /// off the concrete virtualizable before tracing the call.
    ///
    /// Dormant — call-site migration from
    /// `call_assembler_with_vable_expansion_args` to
    /// `call_assembler_red_only_*` will plug this in once the callee
    /// JUMP-terminated paths run `patch_new_loop_to_load_virtualizable
    /// _fields` (pyjitpl/mod.rs:3090-3098 deferred epic). Covered by
    /// `emit_vable_field_reads_emits_compile_py_shape` so the helper
    /// stays honest until the call-site flip lands.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn emit_vable_field_reads(
        &mut self,
        vable: OpRef,
        vinfo: &VirtualizableInfo,
        array_lengths: &[usize],
    ) -> Vec<OpRef> {
        let mut expanded = Vec::with_capacity(
            vinfo.static_fields.len() + array_lengths.iter().copied().sum::<usize>(),
        );

        // compile.py:434-440 — GETFIELD_GC per static field.
        let static_descrs = vinfo.static_field_descrs();
        for (fi, field) in vinfo.static_fields.iter().enumerate() {
            let opcode = match field.field_type {
                Type::Int => OpCode::GetfieldGcI,
                Type::Ref => OpCode::GetfieldGcR,
                Type::Float => OpCode::GetfieldGcF,
                Type::Void => panic!("emit_vable_field_reads: static field {fi} has Void type"),
            };
            let descr = static_descrs[fi].clone();
            let opref = self.record_op_with_descr(opcode, &[vable], descr);
            expanded.push(opref);
        }

        // compile.py:441-457 — GETFIELD_GC_R(array ptr) + GETARRAYITEM_GC.
        let array_field_descrs = vinfo.array_field_descrs();
        for (ai, array_field_descr) in array_field_descrs.iter().enumerate() {
            let array_len = array_lengths.get(ai).copied().unwrap_or(0);
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable], array_field_descr.clone());
            let array_descr = vinfo.array_descrs[ai].clone();
            let item_opcode = match vinfo.array_fields[ai].item_type {
                Type::Int => OpCode::GetarrayitemGcI,
                Type::Ref => OpCode::GetarrayitemGcR,
                Type::Float => OpCode::GetarrayitemGcF,
                Type::Void => panic!("emit_vable_field_reads: array {ai} has Void item_type"),
            };
            for index in 0..array_len {
                let const_idx = self.const_int(index as i64);
                let opref = self.record_op_with_descr(
                    item_opcode,
                    &[array_opref, const_idx],
                    array_descr.clone(),
                );
                expanded.push(opref);
            }
        }
        expanded
    }

    /// pyjitpl.py:1148-1158 `MIFrame.emit_force_virtualizable(fielddescr, box)`.
    ///
    /// ```text
    /// def emit_force_virtualizable(self, fielddescr, box):
    ///     vinfo = fielddescr.get_vinfo()
    ///     assert vinfo is not None
    ///     token_descr = vinfo.vable_token_descr
    ///     mi = self.metainterp
    ///     tokenbox = mi.execute_and_record(rop.GETFIELD_GC_R, token_descr, box)
    ///     condbox = mi.execute_and_record(rop.PTR_NE, None, tokenbox, CONST_NULL)
    ///     funcbox = ConstInt(rffi.cast(lltype.Signed, vinfo.clear_vable_ptr))
    ///     calldescr = vinfo.clear_vable_descr
    ///     self.execute_varargs(rop.COND_CALL, [condbox, funcbox, box],
    ///                          calldescr, False, False)
    /// ```
    fn emit_force_virtualizable(&mut self, fielddescr: &DescrRef, vable_opref: OpRef) {
        //     vinfo = fielddescr.get_vinfo()
        //     assert vinfo is not None
        //
        // `finalize_arc` stamps every field descriptor with a
        // `Weak<dyn VinfoMarker>` backref; `get_vinfo()` upgrades it
        // and returns the owning `VirtualizableInfo`.  When the
        // descriptor was built via the legacy by-value
        // `set_parent_descr` path (no Arc available), `get_vinfo()`
        // returns `None` and pyre falls back to the active
        // `self.virtualizable_info` slot so the existing by-value
        // test harness keeps working.
        let marker = fielddescr.as_field_descr().and_then(|fd| fd.get_vinfo());
        let (token_descr, clear_ptr, clear_descr) = {
            let info_ref: &VirtualizableInfo = if let Some(ref m) = marker {
                m.as_any()
                    .downcast_ref::<VirtualizableInfo>()
                    .expect("emit_force_virtualizable: VinfoMarker is not a VirtualizableInfo")
            } else {
                self.virtualizable_info
                    .as_deref()
                    .expect("emit_force_virtualizable: vinfo is None")
            };
            //     token_descr = vinfo.vable_token_descr
            let token_descr = info_ref.token_field_descr();
            //     funcbox = ConstInt(rffi.cast(lltype.Signed, vinfo.clear_vable_ptr))
            let clear_ptr = info_ref
                .clear_vable_ptr
                .expect("emit_force_virtualizable: clear_vable_ptr not set");
            //     calldescr = vinfo.clear_vable_descr
            let clear_descr = info_ref
                .clear_vable_descr
                .clone()
                .expect("emit_force_virtualizable: clear_vable_descr not set");
            (token_descr, clear_ptr, clear_descr)
        };
        //     tokenbox = mi.execute_and_record(rop.GETFIELD_GC_R, token_descr, box)
        let tokenbox = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], token_descr);
        //     condbox = mi.execute_and_record(rop.PTR_NE, None, tokenbox, CONST_NULL)
        let null_ref = self.const_null();
        let condbox = self.record_op(OpCode::PtrNe, &[tokenbox, null_ref]);
        let funcbox = self.const_int(clear_ptr as i64);
        //     self.execute_varargs(rop.COND_CALL, [condbox, funcbox, box],
        //                          calldescr, False, False)
        Self::do_record_op_with_descr(
            &mut self.recorder,
            &self.constants,
            OpCode::CondCallN,
            &[condbox, funcbox, vable_opref],
            clear_descr,
        );
    }

    /// pyjitpl.py:1120-1146 `_nonstandard_virtualizable(pc, box, fielddescr)`.
    ///
    /// ```text
    ///  def _nonstandard_virtualizable(self, pc, box, fielddescr):
    ///      # returns True if 'box' is actually not the "standard" virtualizable
    ///      # that is stored in metainterp.virtualizable_boxes[-1]
    ///      if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
    ///          self.metainterp.staticdata.profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
    ///          return True
    ///      if box is self.metainterp.forced_virtualizable:
    ///          self.metainterp.forced_virtualizable = None
    ///      if (self.metainterp.jitdriver_sd.virtualizable_info is not None or
    ///          self.metainterp.jitdriver_sd.greenfield_info is not None):
    ///          standard_box = self.metainterp.virtualizable_boxes[-1]
    ///          if standard_box is box:
    ///              return False
    ///          vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///          if vinfo is fielddescr.get_vinfo():
    ///              eqbox = self.metainterp.execute_and_record(rop.PTR_EQ, None,
    ///                                                         box, standard_box)
    ///              eqbox = self.implement_guard_value(eqbox, pc)
    ///              isstandard = eqbox.getint()
    ///              if isstandard:
    ///                  if box.type == 'r':
    ///                      self.metainterp.replace_box(box, standard_box)
    ///                  return False
    ///      if not self.metainterp.heapcache.is_unescaped(box):
    ///          self.emit_force_virtualizable(fielddescr, box)
    ///      self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
    ///      return True
    /// ```
    ///
    /// In pyre this is the LIVE entry path used by the jitcode machine
    /// (`vable_*_indexed`) at trace time. The pyjitpl::nonstandard_virtualizable
    /// duplicate is reachable only from the legacy `opimpl_*_vable` test
    /// surface. The two implementations carry the same line-by-line shape so
    /// the structural divergence is duplication-only — fixing the type-tag
    /// refactor will let us collapse them into a single entry point.
    fn is_nonstandard_virtualizable(
        &mut self,
        pc: usize,
        vable_opref: OpRef,
        fielddescr: &DescrRef,
        concrete: Value,
    ) -> bool {
        // Step 1: heapcache short-circuit.
        //     if self.metainterp.heapcache.is_known_nonstandard_virtualizable(box):
        //         self.metainterp.staticdata.profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS)
        //         return True
        if self
            .heap_cache
            .is_known_nonstandard_virtualizable(vable_opref)
        {
            // pyjitpl.py:1124 profiler.count_ops(rop.PTR_EQ, Counters.HEAPCACHED_OPS).
            self.profiler()
                .count_ops(OpCode::PtrEq, crate::pyjitpl::counters::HEAPCACHED_OPS);
            return true;
        }
        // Step 2: forced_virtualizable reset on identity.
        //     if box is self.metainterp.forced_virtualizable:
        //         self.metainterp.forced_virtualizable = None
        if self.forced_virtualizable == Some(vable_opref) {
            self.forced_virtualizable = None;
        }
        // Step 3: standard_box identity check.
        //     standard_box = self.metainterp.virtualizable_boxes[-1]
        //     if standard_box is box:
        //         return False
        let standard_box = self
            .virtualizable_boxes
            .as_ref()
            .and_then(|boxes| boxes.last().copied());
        let Some(standard_box) = standard_box else {
            // No boxes → treat as nonstandard.
            return true;
        };
        if standard_box == vable_opref {
            return false;
        }
        // Step 4: PTR_EQ + implement_guard_value + replace_box.
        //     vinfo = self.metainterp.jitdriver_sd.virtualizable_info
        //     if vinfo is fielddescr.get_vinfo():
        //         eqbox = self.metainterp.execute_and_record(
        //             rop.PTR_EQ, None, box, standard_box)
        //         eqbox = self.implement_guard_value(eqbox, pc)
        //         isstandard = eqbox.getint()
        //         if isstandard:
        //             if box.type == 'r':
        //                 self.metainterp.replace_box(box, standard_box)
        //             return False
        //
        // `fielddescr.get_vinfo()` upgrades the backref stamped by
        // `finalize_arc`.  When the descriptor carries a vinfo backref,
        // the upstream `vinfo is fielddescr.get_vinfo()` check holds iff
        // the active `virtualizable_info` is the same concrete type
        // (pyre single-driver: trivially true).  When the descriptor
        // lacks a backref (by-value legacy path), pyre skips the
        // PTR_EQ/replace_box short-circuit and falls through to Step 5 —
        // same behaviour as upstream when the fielddescr came from a
        // different jitdriver's vinfo.
        let descriptor_vinfo = fielddescr.as_field_descr().and_then(|fd| fd.get_vinfo());
        let descriptor_has_matching_vinfo = match descriptor_vinfo {
            // Backref stamped by `finalize_arc` → concrete type must be
            // our `VirtualizableInfo`.  Pyre's single-driver model means
            // every marker that downcasts successfully is the active
            // vinfo; this is the structural mirror of upstream's Python
            // `vinfo is fielddescr.get_vinfo()` identity check.
            Some(ref m) => m.as_any().is::<VirtualizableInfo>(),
            // Legacy by-value descriptor → no backref to compare
            // against.  Treat as "matching" so the PTR_EQ/replace_box
            // block still runs for test harnesses that pre-date
            // `finalize_arc`.  Production pyre always stamps backrefs.
            None => true,
        };
        if descriptor_has_matching_vinfo {
            let standard_concrete = self
                .standard_virtualizable_concrete()
                .unwrap_or(Value::Ref(majit_ir::GcRef(usize::MAX)));
            // pyjitpl.py:1135-1138 `eqbox = self.metainterp.execute_and_record(
            //     rop.PTR_EQ, None, box, standard_box);
            //     eqbox = self.implement_guard_value(eqbox, pc);
            //     isstandard = eqbox.getint()`.
            //
            // pyre resolves `isstandard` by comparing the traced concrete
            // ptrs directly (see `concrete_of_opref` for how `concrete` is
            // reconstructed from tracer-local state).  The subsequent
            // `promote_int` records the GUARD_VALUE that commits the
            // runtime outcome to the trace.  `pc` threads through for
            // RPython signature parity; pyre's `record_guard` seeds the
            // guard descr via `num_live` (live-var count), not pc, so the
            // parameter is documented here but not consumed at this layer.
            let _ = pc;
            let eqbox = self.record_op(OpCode::PtrEq, &[vable_opref, standard_box]);
            let isstandard: i64 = if concrete_ptrs_eq(&concrete, &standard_concrete) {
                1
            } else {
                0
            };
            self.promote_int(eqbox, isstandard, 0);
            if isstandard != 0 {
                // pyjitpl.py:1140-1142 `if box.type == 'r':
                //     self.metainterp.replace_box(box, standard_box)`.
                // Virtualizables are always Refs in pyre, so the
                // `box.type == 'r'` check is unconditional here.
                // Upstream's `MetaInterp.replace_box` includes a
                // framestack walk (see `MetaInterp::replace_box` in
                // pyjitpl/mod.rs); reaching that walk from TraceCtx
                // requires a MetaInterp backref which does not exist
                // today — Task #68 C tracks the architectural move.
                self.replace_box(vable_opref, standard_box);
                return false;
            }
        }
        // Step 5a: emit_force_virtualizable.
        //     if not self.metainterp.heapcache.is_unescaped(box):
        //         self.emit_force_virtualizable(fielddescr, box)
        //
        //     def emit_force_virtualizable(self, fielddescr, box):
        //         vinfo = fielddescr.get_vinfo()
        //         token_descr = vinfo.vable_token_descr
        //         tokenbox = mi.execute_and_record(
        //             rop.GETFIELD_GC_R, token_descr, box)
        //         condbox = mi.execute_and_record(
        //             rop.PTR_NE, None, tokenbox, CONST_NULL)
        //         funcbox = ConstInt(rffi.cast(Signed, vinfo.clear_vable_ptr))
        //         self.execute_varargs(
        //             rop.COND_CALL, [condbox, funcbox, box],
        //             vinfo.clear_vable_descr, False, False)
        if !self.heap_cache.is_unescaped(vable_opref) {
            self.emit_force_virtualizable(fielddescr, vable_opref);
        }
        // Step 5b: mark this box as a known nonstandard virtualizable so
        // future accesses short-circuit at Step 1.
        //     self.metainterp.heapcache.nonstandard_virtualizables_now_known(box)
        self.heap_cache
            .nonstandard_virtualizables_now_known(vable_opref);
        true
    }

    /// pyjitpl.py:1167-1172 `opimpl_getfield_vable_i(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_i(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_i(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fielddescr, concrete) {
            // self.opimpl_getfield_gc_i(box, fielddescr)
            let op = self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        // self.metainterp.check_synchronized_virtualizable() — no-op in pyre.
        // index = self._get_virtualizable_field_index(fielddescr)
        // return self.metainterp.virtualizable_boxes[index]
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        // Fallback for tests/missing layout
        let op = self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Record a virtualizable field read with an explicit field descriptor.
    pub fn vable_getfield_int_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcI, &[vable_opref], descr)
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable(box, valuebox, fielddescr, pc)`.
    ///
    /// ```text
    ///  def _opimpl_setfield_vable(self, box, valuebox, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      self.metainterp.virtualizable_boxes[index] = valuebox
    ///      self.metainterp.synchronize_virtualizable()
    ///      # XXX only the index'th field needs to be synchronized, really
    /// ```
    pub fn vable_setfield(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        let vable_concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fielddescr, vable_concrete) {
            // self._opimpl_setfield_gc_any(box, valuebox, fielddescr)
            self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], fielddescr);
            return;
        }
        // index = self._get_virtualizable_field_index(fielddescr)
        // self.metainterp.virtualizable_boxes[index] = valuebox
        let index = self
            .virtualizable_info
            .as_ref()
            .expect("vable_setfield: virtualizable_info missing")
            .static_field_by_descr(&fielddescr)
            .expect("vable_setfield: standard virtualizable field descr missing");
        self.set_virtualizable_entry_at(index, value, concrete);
        // pyjitpl.py:3446 write_boxes parity: mirror the updated
        // shadow slot back into the live virtualizable.
        self.synchronize_virtualizable();
    }

    /// Record a virtualizable field write with an explicit field descriptor.
    pub fn vable_setfield_descr(&mut self, vable_opref: OpRef, value: OpRef, descr: DescrRef) {
        self.record_op_with_descr(OpCode::SetfieldGc, &[vable_opref, value], descr);
    }

    /// pyjitpl.py:1173-1179 `opimpl_getfield_vable_r(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_r(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_r(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fielddescr, concrete) {
            let op = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        let op = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Record a virtualizable ref field read with an explicit field descriptor.
    pub fn vable_getfield_ref_descr(&mut self, vable_opref: OpRef, descr: DescrRef) -> OpRef {
        self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], descr)
    }

    /// pyjitpl.py:1180-1186 `opimpl_getfield_vable_f(box, fielddescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_getfield_vable_f(self, box, fielddescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fielddescr):
    ///          return self.opimpl_getfield_gc_f(box, fielddescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_virtualizable_field_index(fielddescr)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getfield_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fielddescr, concrete) {
            let op = self.record_op_with_descr(OpCode::GetfieldGcF, &[vable_opref], fielddescr);
            return (op, Value::Void);
        }
        let index = self
            .virtualizable_info
            .as_ref()
            .and_then(|info| info.static_field_by_descr(&fielddescr));
        if let Some(idx) = index {
            if let Some(entry) = self.virtualizable_entry_at(idx) {
                return entry;
            }
        }
        let op = self.record_op_with_descr(OpCode::GetfieldGcF, &[vable_opref], fielddescr);
        (op, Value::Void)
    }

    /// Standard virtualizable array item read (int).
    /// `array_field_offset` identifies which array field, `item_index` is the element index.
    /// If standard boxes are active, reads from the flat box array directly.
    pub fn vable_getarrayitem_int_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let op = self.record_op_with_descr(OpCode::GetarrayitemGcI, &[array_opref, index], adescr);
        (op, Value::Void)
    }

    /// pyjitpl.py:1201-1216 `_get_arrayitem_vable_index(pc, arrayfielddescr, indexbox)`.
    ///
    /// ```text
    ///  def _get_arrayitem_vable_index(self, pc, arrayfielddescr, indexbox):
    ///      indexbox = self.implement_guard_value(indexbox, pc)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
    ///      index = indexbox.getint()
    ///      assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
    ///      return vinfo.get_index_in_array(virtualizable, arrayindex, index)
    /// ```
    fn get_arrayitem_vable_index(
        &mut self,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: &DescrRef,
    ) -> Option<usize> {
        // indexbox = self.implement_guard_value(indexbox, pc)
        let promoted_index = if index.is_constant() {
            index
        } else {
            self.promote_int(index, index_runtime_value, 0)
        };
        let _ = promoted_index;
        let item_index = usize::try_from(index_runtime_value).ok()?;
        // arrayindex = vinfo.array_field_by_descrs[arrayfielddescr]
        // assert 0 <= index < vinfo.get_array_length(virtualizable, arrayindex)
        // return vinfo.get_index_in_array(virtualizable, arrayindex, index)
        self.vable_array_flat_index(fdescr, item_index)
    }

    /// pyjitpl.py:1218-1230 `_opimpl_getarrayitem_vable(box, indexbox, fdescr, adescr, pc)`
    /// (int variant via `opimpl_getarrayitem_vable_i = _opimpl_getarrayitem_vable`).
    ///
    /// ```text
    ///  def _opimpl_getarrayitem_vable(self, box, indexbox, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          ...
    ///          return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
    ///      self.metainterp.check_synchronized_virtualizable()
    ///      index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
    ///      return self.metainterp.virtualizable_boxes[index]
    /// ```
    pub fn vable_getarrayitem_int_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fdescr, concrete) {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_getarrayitem_gc_i(arraybox, indexbox, adescr)
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (
                self.vable_getarrayitem_int_descr(array_opref, index, adescr),
                Value::Void,
            );
        }
        // index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
        // return self.metainterp.virtualizable_boxes[index]
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        // Fallback: vable layout missing — go through getfield + arrayitem.
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_int_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_int_descr(array_opref, index, adescr),
                Value::Void,
            )
        }
    }

    /// Standard virtualizable array item read (ref).
    pub fn vable_getarrayitem_ref_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let op = self.record_op_with_descr(OpCode::GetarrayitemGcR, &[array_opref, index], adescr);
        (op, Value::Void)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — ref variant.
    pub fn vable_getarrayitem_ref_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fdescr, concrete) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (
                self.vable_getarrayitem_ref_descr(array_opref, index, adescr),
                Value::Void,
            );
        }
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_ref_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_ref_descr(array_opref, index, adescr),
                Value::Void,
            )
        }
    }

    /// Standard virtualizable array item read (float).
    pub fn vable_getarrayitem_float_vable(
        &mut self,
        array_opref: OpRef,
        fdescr: &DescrRef,
        item_index: usize,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        if let Some(flat_idx) = self.vable_array_flat_index(fdescr, item_index) {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let index = self.const_int(item_index as i64);
        let op = self.record_op_with_descr(OpCode::GetarrayitemGcF, &[array_opref, index], adescr);
        (op, Value::Void)
    }

    /// pyjitpl.py:1218-1234 `_opimpl_getarrayitem_vable` — float variant.
    pub fn vable_getarrayitem_float_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> (OpRef, Value) {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fdescr, concrete) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return (
                self.vable_getarrayitem_float_descr(array_opref, index, adescr),
                Value::Void,
            );
        }
        if let Some(flat_idx) = self.get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
        {
            if let Some(entry) = self.virtualizable_entry_at(flat_idx) {
                return entry;
            }
        }
        let array_opref =
            self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr.clone());
        if let Ok(item_index) = usize::try_from(index_runtime_value) {
            self.vable_getarrayitem_float_vable(array_opref, &fdescr, item_index, adescr)
        } else {
            (
                self.vable_getarrayitem_float_descr(array_opref, index, adescr),
                Value::Void,
            )
        }
    }

    /// Standard virtualizable array item write at a known flat slot index.
    /// `item_index` is the element index within the array described by `fdescr`.
    pub fn vable_setarrayitem_vable(
        &mut self,
        fdescr: &DescrRef,
        item_index: usize,
        value: OpRef,
        concrete: Value,
    ) {
        let flat_idx = self
            .vable_array_flat_index(fdescr, item_index)
            .expect("vable_setarrayitem_vable: standard virtualizable array slot missing");
        self.set_virtualizable_entry_at(flat_idx, value, concrete);
        self.synchronize_virtualizable();
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable(box, indexbox, valuebox, fdescr, adescr, pc)`.
    pub fn vable_setarrayitem_indexed(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
        adescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        let vable_concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fdescr, vable_concrete) {
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            self.vable_setarrayitem_descr(array_opref, index, value, adescr);
            return;
        }
        // index = self._get_arrayitem_vable_index(pc, fdescr, indexbox)
        // self.metainterp.virtualizable_boxes[index] = valuebox
        // self.metainterp.synchronize_virtualizable()
        let flat_idx = self
            .get_arrayitem_vable_index(index, index_runtime_value, &fdescr)
            .expect("vable_setarrayitem_indexed: virtualizable array slot missing");
        self.set_virtualizable_entry_at(flat_idx, value, concrete);
        self.synchronize_virtualizable();
    }

    /// pyjitpl.py:1253-1263 `opimpl_arraylen_vable(box, fdescr, adescr, pc)`.
    ///
    /// ```text
    ///  def opimpl_arraylen_vable(self, box, fdescr, adescr, pc):
    ///      if self._nonstandard_virtualizable(pc, box, fdescr):
    ///          arraybox = self.opimpl_getfield_gc_r(box, fdescr)
    ///          return self.opimpl_arraylen_gc(arraybox, adescr)
    ///      vinfo = self.metainterp.jitdriver_sd.virtualizable_info
    ///      virtualizable_box = self.metainterp.virtualizable_boxes[-1]
    ///      virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///      arrayindex = vinfo.array_field_by_descrs[fdescr]
    ///      result = vinfo.get_array_length(virtualizable, arrayindex)
    ///      return ConstInt(result)
    /// ```
    pub fn vable_arraylen_vable(
        &mut self,
        vable_opref: OpRef,
        fdescr: DescrRef,
        adescr: DescrRef,
    ) -> OpRef {
        let concrete = self.concrete_of_opref(vable_opref);
        if self.is_nonstandard_virtualizable(0, vable_opref, &fdescr, concrete) {
            // arraybox = self.opimpl_getfield_gc_r(box, fdescr)
            // return self.opimpl_arraylen_gc(arraybox, adescr)
            let array_opref =
                self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
            return self.record_op_with_descr(OpCode::ArraylenGc, &[array_opref], adescr);
        }
        // arrayindex = vinfo.array_field_by_descrs[fdescr]
        // result = vinfo.get_array_length(virtualizable, arrayindex)
        // return ConstInt(result)
        if let (Some(info), Some(lengths)) =
            (&self.virtualizable_info, &self.virtualizable_array_lengths)
        {
            if let Some(array_idx) = info.array_field_by_descr(&fdescr) {
                if let Some(&length) = lengths.get(array_idx) {
                    return self.const_int(length as i64);
                }
            }
        }
        // Fallback when the layout is unavailable.
        let array_opref = self.record_op_with_descr(OpCode::GetfieldGcR, &[vable_opref], fdescr);
        self.record_op_with_descr(OpCode::ArraylenGc, &[array_opref], adescr)
    }

    /// Compute the flat index into virtualizable_boxes for an array element.
    /// Returns `None` if standard virtualizable is not active or the array field is unknown.
    fn vable_array_flat_index(&self, fdescr: &DescrRef, item_index: usize) -> Option<usize> {
        let info = self.virtualizable_info.as_ref()?;
        let lengths = self.virtualizable_array_lengths.as_ref()?;
        let array_idx = info.array_field_by_descr(fdescr)?;
        Some(info.get_index_in_array(array_idx, item_index, lengths))
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_int_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcI, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_ref_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcR, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item read with an explicit array descriptor.
    pub fn vable_getarrayitem_float_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        descr: DescrRef,
    ) -> OpRef {
        self.record_op_with_descr(OpCode::GetarrayitemGcF, &[array_opref, index], descr)
    }

    /// Record a virtualizable array item write with an explicit array descriptor.
    pub fn vable_setarrayitem_descr(
        &mut self,
        array_opref: OpRef,
        index: OpRef,
        value: OpRef,
        descr: DescrRef,
    ) {
        self.record_op_with_descr(OpCode::SetarrayitemGc, &[array_opref, index, value], descr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit_state::JitState;
    use majit_backend::JitCellToken;
    use majit_ir::Type;

    extern "C" fn dummy_call_target() {}

    #[test]
    fn const_type_honors_resume_data_override() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx.const_int(0);
        // Box.type immutability: a ConstInt's intrinsic type is always Int,
        // recorded in the constant pool at allocation time.
        assert_eq!(ctx.const_type(c), Some(Type::Int));
        ctx.mark_const_type(c, Type::Ref);
        // mark_type → numbering_type_overrides takes priority over the
        // intrinsic Int; this is the raw-pointer-ConstInt retag path.
        assert_eq!(ctx.const_type(c), Some(Type::Ref));
    }

    // ── M1 · opref_to_box bridge tests ─────────────────────────────────

    /// M1: non-constant OpRefs (inputargs + recorded op results) map
    /// straight to Box::ResOp(opref.raw()).  No constant-pool lookup.
    #[test]
    fn test_opref_to_box_non_constant_m1() {
        let mut ctx = TraceCtx::for_test(2);
        let i0 = OpRef::input_arg_int(0); // first inputarg
        let i1 = OpRef::input_arg_int(1); // second inputarg
        let add = ctx.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(ctx.opref_to_box(i0), OcBox::ResOp(0));
        assert_eq!(ctx.opref_to_box(i1), OcBox::ResOp(1));
        assert_eq!(ctx.opref_to_box(add), OcBox::ResOp(add.raw()));
    }

    /// M1: constant OpRefs route through ConstantPool::get_value for
    /// type-preserving Box::Const* construction.
    #[test]
    fn test_opref_to_box_constant_int_m1() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx.const_int(42);
        assert!(c.is_constant());
        assert_eq!(ctx.opref_to_box(c), OcBox::ConstInt(42));
    }

    #[test]
    fn test_opref_to_box_constant_float_m1() {
        let mut ctx = TraceCtx::for_test(0);
        let c = ctx
            .constants
            .get_or_insert_typed((3.14_f64).to_bits() as i64, Type::Float);
        assert!(c.is_constant());
        match ctx.opref_to_box(c) {
            OcBox::ConstFloat(bits) => {
                assert_eq!(f64::from_bits(bits), 3.14);
            }
            other => panic!("expected ConstFloat, got {:?}", other),
        }
    }

    #[test]
    fn test_opref_to_box_constant_ref_m1() {
        let mut ctx = TraceCtx::for_test(0);
        // Non-zero Ref address exercises the shadow-stack rooting path.
        let addr = 0xdead_beef_u64;
        let c = ctx.constants.get_or_insert_typed(addr as i64, Type::Ref);
        assert!(c.is_constant());
        assert_eq!(ctx.opref_to_box(c), OcBox::ConstPtr(addr));
    }

    /// M1: constant OpRef that was not registered in the pool is a
    /// genuine invariant break — the helper must panic rather than
    /// silently substitute a Box::ResOp.
    #[test]
    #[should_panic(expected = "not in pool")]
    fn test_opref_to_box_orphan_constant_panics_m1() {
        let ctx = TraceCtx::for_test(0);
        // Forge a constant-namespace OpRef without touching the pool.
        let orphan = OpRef::const_int(7);
        ctx.opref_to_box(orphan);
    }

    #[derive(Clone, Copy)]
    struct TestSyncField {
        field_descr_idx: u32,
        value: OpRef,
        field_type: Type,
    }

    struct TestSyncState {
        vable_ref: OpRef,
        fields: Vec<TestSyncField>,
        forced: bool,
    }

    impl JitState for TestSyncState {
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
        fn collect_jump_args(_: &()) -> Vec<OpRef> {
            Vec::new()
        }
        fn validate_close(_: &(), _: &()) -> bool {
            true
        }

        fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
            for field in &self.fields {
                let descr = crate::compile::make_fail_descr(field.field_descr_idx as usize);
                ctx.record_op_with_descr(OpCode::SetfieldGc, &[self.vable_ref, field.value], descr);
            }
        }

        fn sync_virtualizable_after_residual_call(
            &self,
            ctx: &mut TraceCtx,
        ) -> crate::jit_state::ResidualVirtualizableSync {
            if self.forced {
                return crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                };
            }
            let updated_fields = self
                .fields
                .iter()
                .map(|field| {
                    let opcode = OpCode::getfield_for_type(field.field_type);
                    let descr = crate::compile::make_fail_descr(field.field_descr_idx as usize);
                    let new_ref = ctx.record_op_with_descr(opcode, &[self.vable_ref], descr);
                    (field.field_descr_idx, new_ref)
                })
                .collect();
            crate::jit_state::ResidualVirtualizableSync {
                updated_fields,
                forced: false,
            }
        }
    }

    fn take_all_ops(ctx: TraceCtx) -> Vec<majit_ir::Op> {
        let mut recorder = ctx.recorder;
        let inputarg_types = recorder.inputarg_types();
        let jump_args: Vec<OpRef> = inputarg_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| OpRef::input_arg_typed(i as u32, tp))
            .collect();
        recorder.close_loop(&jump_args);
        let trace = recorder.get_trace();
        // Return only non-JUMP ops
        trace
            .ops
            .iter()
            .filter(|op| op.opcode != OpCode::Jump)
            .cloned()
            .collect()
    }

    #[test]
    fn call_may_force_with_jitstate_sync_emits_setfield_before_and_getfield_after() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 42,
                value: field_val,
                field_type: Type::Int,
            }],
            forced: false,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert!(result.raw() > 0);
        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 42);
        assert_ne!(sync.updated_fields[0].1, field_val);

        let ops = take_all_ops(ctx);
        assert!(ops.len() >= 4);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_void_emits_correct_sequence() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 10,
                value: field_val,
                field_type: Type::Int,
            }],
            forced: false,
        };

        let sync = ctx.call_may_force_with_jitstate_sync_void(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 10);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceN);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_multiple_fields() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let int_val = recorder.record_input_arg(Type::Int);
        let ref_val = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![
                TestSyncField {
                    field_descr_idx: 0,
                    value: int_val,
                    field_type: Type::Int,
                },
                TestSyncField {
                    field_descr_idx: 1,
                    value: ref_val,
                    field_type: Type::Ref,
                },
            ],
            forced: false,
        };

        let (_, sync) = ctx.call_may_force_with_jitstate_sync_ref(
            dummy_call_target as *const (),
            &[int_val],
            &[Type::Int],
            &state,
            3,
        );

        assert_eq!(sync.updated_fields.len(), 2);
        assert_eq!(sync.updated_fields[0].0, 0);
        assert_eq!(sync.updated_fields[1].0, 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[2].opcode, OpCode::CallMayForceR);
        assert_eq!(ops[3].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[4].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[5].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_empty_jitstate_sync_behaves_like_plain_call() {
        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let state = TestSyncState {
            vable_ref: vable,
            fields: Vec::new(),
            forced: false,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(result.raw() > 0);
        assert!(sync.updated_fields.is_empty());

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_float_field() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let float_val = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let state = TestSyncState {
            vable_ref: vable,
            fields: vec![TestSyncField {
                field_descr_idx: 5,
                value: float_val,
                field_type: Type::Float,
            }],
            forced: false,
        };

        let (_, sync) = ctx.call_may_force_with_jitstate_sync_float(
            dummy_call_target as *const (),
            &[float_val],
            &[Type::Float],
            &state,
            2,
        );

        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 5);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceF);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcF);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_default_noop() {
        use crate::jit_state::JitState;

        #[derive(Default)]
        struct NoVableState;

        impl JitState for NoVableState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = NoVableState;

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        // Default JitState does no sync => no extra ops
        assert!(result.raw() > 0);
        assert!(!sync.forced);
        assert!(sync.updated_fields.is_empty());

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[1].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_custom_impl() {
        use crate::jit_state::JitState;

        struct VableState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for VableState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                // Write field 0 to heap
                let fd = majit_ir::make_field_descr(0, 8, Type::Int, majit_ir::ArrayFlag::Signed);
                ctx.vable_setfield(self.vable_ref, fd, self.field_val, Value::Int(0));
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                // Re-read field 0 from heap
                let fd = majit_ir::make_field_descr(0, 8, Type::Int, majit_ir::ArrayFlag::Signed);
                let (new_ref, _) = ctx.vable_getfield_int(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = VableState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Int],
            &state,
            2,
        );

        assert!(result.raw() > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);
        assert_eq!(sync.updated_fields[0].0, 0);

        let ops = take_all_ops(ctx);
        // SetfieldGc(before) + CallMayForceI + GetfieldGcI(after) + GuardNotForced
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceI);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcI);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_skips_guard_when_forced() {
        use crate::jit_state::JitState;

        struct ForcedState;

        impl JitState for ForcedState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                _ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                }
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = ForcedState;

        let (_result, sync) = ctx.call_may_force_with_jitstate_sync_int(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(sync.forced);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceI);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_ref_custom_impl() {
        use crate::jit_state::JitState;

        struct RefState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for RefState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                let fd = majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::ArrayFlag::Pointer);
                ctx.vable_setfield(
                    self.vable_ref,
                    fd,
                    self.field_val,
                    Value::Ref(majit_ir::GcRef::NULL),
                );
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let fd = majit_ir::make_field_descr(0, 8, Type::Ref, majit_ir::ArrayFlag::Pointer);
                let (new_ref, _) = ctx.vable_getfield_ref(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = RefState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_ref(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Ref],
            &state,
            2,
        );

        assert!(result.raw() > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceR);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcR);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_float_custom_impl() {
        use crate::jit_state::JitState;

        struct FloatState {
            vable_ref: OpRef,
            field_val: OpRef,
        }

        impl JitState for FloatState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
                let fd = majit_ir::make_field_descr(0, 8, Type::Float, majit_ir::ArrayFlag::Float);
                ctx.vable_setfield(self.vable_ref, fd, self.field_val, Value::Float(0.0));
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                let fd = majit_ir::make_field_descr(0, 8, Type::Float, majit_ir::ArrayFlag::Float);
                let (new_ref, _) = ctx.vable_getfield_float(self.vable_ref, fd);
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: vec![(0, new_ref)],
                    forced: false,
                }
            }
        }

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let field_val = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = FloatState {
            vable_ref: vable,
            field_val,
        };

        let (result, sync) = ctx.call_may_force_with_jitstate_sync_float(
            dummy_call_target as *const (),
            &[field_val],
            &[Type::Float],
            &state,
            2,
        );

        assert!(result.raw() > 0);
        assert!(!sync.forced);
        assert_eq!(sync.updated_fields.len(), 1);

        let ops = take_all_ops(ctx);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(ops[1].opcode, OpCode::CallMayForceF);
        assert_eq!(ops[2].opcode, OpCode::GetfieldGcF);
        assert_eq!(ops[3].opcode, OpCode::GuardNotForced);
    }

    #[test]
    fn call_may_force_with_jitstate_sync_void_skips_guard_when_forced() {
        use crate::jit_state::JitState;

        struct ForcedVoidState;

        impl JitState for ForcedVoidState {
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
            fn collect_jump_args(_: &()) -> Vec<OpRef> {
                Vec::new()
            }
            fn validate_close(_: &(), _: &()) -> bool {
                true
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                _ctx: &mut TraceCtx,
            ) -> crate::jit_state::ResidualVirtualizableSync {
                crate::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced: true,
                }
            }
        }

        let mut recorder = Trace::new();
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        let state = ForcedVoidState;

        let sync = ctx.call_may_force_with_jitstate_sync_void(
            dummy_call_target as *const (),
            &[val],
            &[Type::Int],
            &state,
            1,
        );

        assert!(sync.forced);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::CallMayForceN);
    }

    // --- virtualizable_boxes tests ---

    fn make_test_vable_info() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_field("sp", Type::Int, 16);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    fn make_test_vable_info_with_array() -> crate::virtualizable::VirtualizableInfo {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        info
    }

    // Test helper: typed placeholder matching each slot's declared type so
    // the Box's (OpRef, concrete) pair stays internally consistent — the
    // RPython `virtualizable_boxes[index] = valuebox` invariant.  Tests
    // only inspect OpRef plumbing; the concrete half is never read.
    fn ph(ty: Type) -> Value {
        match ty {
            Type::Int => Value::Int(0),
            Type::Float => Value::Float(0.0),
            Type::Ref => Value::Ref(majit_ir::GcRef::NULL),
            Type::Void => Value::Void,
        }
    }

    #[test]
    fn standard_vable_getfield_reads_from_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int); // pc
        let box1 = recorder.record_input_arg(Type::Int); // sp
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // getfield with offset=8 → static field 0 → box0
        let (result, _) = ctx.vable_getfield_int(vable, fd8);
        assert_eq!(result, box0);
        // getfield with offset=16 → static field 1 → box1
        let (result, _) = ctx.vable_getfield_int(vable, fd16);
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getfield should not emit ops"
        );
    }

    #[test]
    fn standard_vable_setfield_writes_to_boxes() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let fd16 = info.static_field_descr(1);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // setfield offset=8 → updates box0
        ctx.vable_setfield(vable, fd8.clone(), new_val, ph(Type::Int));

        // Box 0 should now be new_val
        let (result, _) = ctx.vable_getfield_int(vable, fd8);
        assert_eq!(result, new_val);
        // Box 1 unchanged
        let (result, _) = ctx.vable_getfield_int(vable, fd16);
        assert_eq!(result, box1);

        // No heap ops should have been emitted
        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable setfield should not emit ops"
        );
    }

    #[test]
    fn nonstandard_vable_getfield_emits_heap_op() {
        // Without init_virtualizable_boxes, falls back to GETFIELD_GC_I
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let _result = ctx.vable_getfield_int(vable, fd8);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn nonstandard_vable_setfield_emits_heap_op() {
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let fd8 = majit_ir::make_field_descr(8, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        ctx.vable_setfield(vable, fd8, val, ph(Type::Int));

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn standard_vable_getfield_unknown_offset_emits_heap_op() {
        let info = make_test_vable_info();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // Unknown offset (999) → fallback to heap op
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let _result = ctx.vable_getfield_int(vable, fd999);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
    }

    #[test]
    fn standard_vable_getfield_ref_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("obj", Type::Ref, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(&info, vable, ph(Type::Ref), &[box0], &[ph(Type::Ref)], &[]);

        let (result, _) = ctx.vable_getfield_ref(vable, fd8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn standard_vable_getfield_float_reads_from_boxes() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("val", Type::Float, 8);
        let parent = majit_ir::descr::make_size_descr(0);
        info.set_parent_descr(parent);
        let fd8 = info.static_field_descr(0);

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Float);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0],
            &[ph(Type::Float)],
            &[],
        );

        let (result, _) = ctx.vable_getfield_float(vable, fd8);
        assert_eq!(result, box0);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_reads_from_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        // 1 static field (pc) + 3 array elements
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let box_arr2 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1, box_arr2],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[3], // array has 3 elements
        );

        // Array field offset=24, item_index=0 → box_arr0
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0, adesc.clone());
        assert_eq!(r0, box_arr0);
        // item_index=1 → box_arr1
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1, adesc.clone());
        assert_eq!(r1, box_arr1);
        // item_index=2 → box_arr2
        let (r2, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 2, adesc);
        assert_eq!(r2, box_arr2);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "standard vable getarrayitem should not emit ops"
        );
    }

    #[test]
    fn vable_setarrayitem_writes_to_boxes() {
        let info = make_test_vable_info_with_array();
        let fd24 = info.array_pointer_field_descr(0);
        let adesc = info.array_item_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let box_arr1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Int), ph(Type::Int)],
            &[2], // array has 2 elements
        );

        // Write to array[1]
        ctx.vable_setarrayitem_vable(&fd24, 1, new_val, ph(Type::Int));

        // Read back: array[0] unchanged, array[1] updated
        let (r0, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 0, adesc.clone());
        assert_eq!(r0, box_arr0);
        let (r1, _) = ctx.vable_getarrayitem_int_vable(vable, &fd24, 1, adesc);
        assert_eq!(r1, new_val);

        let ops = take_all_ops(ctx);
        assert!(ops.is_empty());
    }

    #[test]
    fn vable_getarrayitem_unknown_array_emits_heap_op() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        // Unknown array field offset → fallback
        let fd999 = majit_ir::make_field_descr(999, 8, Type::Int, majit_ir::ArrayFlag::Signed);
        let adesc = info.array_item_descr(0);
        let _r = ctx.vable_getarrayitem_int_vable(vable, &fd999, 0, adesc);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opcode, OpCode::GetarrayitemGcI);
    }

    #[test]
    fn collect_virtualizable_boxes_returns_current_state() {
        let info = make_test_vable_info();
        let fd8 = info.static_field_descr(0);
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box0 = recorder.record_input_arg(Type::Int);
        let box1 = recorder.record_input_arg(Type::Int);
        let new_val = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        // Before init: None
        assert!(ctx.collect_virtualizable_boxes().is_none());

        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box0, box1],
            &[ph(Type::Int), ph(Type::Int)],
            &[],
        );

        // After init: has boxes (field0, field1, vable_ref sentinel)
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![box0, box1, vable]);

        // After mutation
        ctx.vable_setfield(vable, fd8, new_val, ph(Type::Int));
        let boxes = ctx.collect_virtualizable_boxes().unwrap();
        assert_eq!(boxes, vec![new_val, box1, vable]);
    }

    #[test]
    fn gen_store_back_in_vable_uses_field_and_array_descrs() {
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Ref,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Ref),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let box_arr1 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );

        ctx.gen_store_back_in_vable(vable);

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[0].descr.as_ref().map(|d| d.index()),
            Some(info.static_field_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].descr.as_ref().map(|d| d.index()),
            Some(info.array_pointer_field_descr(0).index())
        );
        assert_eq!(ops[2].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[2].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[3].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[3].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[4].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[4].descr.as_ref().map(|d| d.index()),
            Some(info.token_field_descr().index())
        );
    }

    #[test]
    fn gen_store_back_in_vable_ignores_nonstandard_virtualizable() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let other_vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_store_back_in_vable(other_vable);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "nonstandard virtualizable must not use standard store-back path"
        );
    }

    #[test]
    fn gen_writeback_vable_to_heap_emits_setfield_setarrayitem_no_token_null() {
        // Same shape as gen_store_back_in_vable but stripped of:
        //   - the forced_virtualizable one-shot guard (no `forced` set)
        //   - the trailing `vable_token = NULL` SetfieldGc
        // Ops emitted (4 total): SetfieldGc(pc), GetfieldGcR(locals_array),
        // SetarrayitemGc(arr,0,...), SetarrayitemGc(arr,1,...).
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Ref,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Ref),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Ref);
        let box_arr1 = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0, box_arr1],
            &[ph(Type::Int), ph(Type::Ref), ph(Type::Ref)],
            &[2],
        );

        ctx.gen_writeback_vable_to_heap(vable);

        // forced_virtualizable must NOT be set — caller may invoke again.
        assert!(ctx.forced_virtualizable.is_none());

        let ops = take_all_ops(ctx);
        assert_eq!(ops.len(), 4, "no trailing token-NULL setfield");
        assert_eq!(ops[0].opcode, OpCode::SetfieldGc);
        assert_eq!(
            ops[0].descr.as_ref().map(|d| d.index()),
            Some(info.static_field_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].descr.as_ref().map(|d| d.index()),
            Some(info.array_pointer_field_descr(0).index())
        );
        assert_eq!(ops[2].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[2].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
        assert_eq!(ops[3].opcode, OpCode::SetarrayitemGc);
        assert_eq!(
            ops[3].descr.as_ref().map(|d| d.index()),
            Some(info.array_item_descr(0).index())
        );
    }

    #[test]
    fn gen_writeback_vable_to_heap_re_emits_on_repeat_calls() {
        // gen_store_back_in_vable's forced_virtualizable guard would
        // suppress the second call; gen_writeback_vable_to_heap must NOT.
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_writeback_vable_to_heap(vable);
        let after_first = ctx.num_ops();
        ctx.gen_writeback_vable_to_heap(vable);
        let after_second = ctx.num_ops();

        assert!(
            after_second > after_first,
            "second writeback must emit (no one-shot guard); after_first={after_first} \
             after_second={after_second}"
        );
    }

    #[test]
    fn gen_writeback_vable_to_heap_skips_nonstandard_vable() {
        let info = make_test_vable_info_with_array();
        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let other_vable = recorder.record_input_arg(Type::Ref);
        let box_pc = recorder.record_input_arg(Type::Int);
        let box_arr0 = recorder.record_input_arg(Type::Int);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );
        ctx.init_virtualizable_boxes(
            &info,
            vable,
            ph(Type::Ref),
            &[box_pc, box_arr0],
            &[ph(Type::Int), ph(Type::Int)],
            &[1],
        );

        ctx.gen_writeback_vable_to_heap(other_vable);

        let ops = take_all_ops(ctx);
        assert!(
            ops.is_empty(),
            "nonstandard virtualizable must not be written back"
        );
    }

    #[test]
    fn emit_vable_field_reads_emits_compile_py_shape() {
        // compile.py:425-461 patch_new_loop_to_load_virtualizable_fields shape:
        //   [GETFIELD_GC_I(vable, pc_descr),
        //    GETFIELD_GC_R(vable, locals_array_descr),
        //    GETARRAYITEM_GC_I(arr, 0, item_descr),
        //    GETARRAYITEM_GC_I(arr, 1, item_descr),
        //    GETARRAYITEM_GC_I(arr, 2, item_descr)]
        let mut info = crate::virtualizable::VirtualizableInfo::new(0);
        info.add_field("pc", Type::Int, 8);
        info.add_array_field(
            "locals",
            Type::Int,
            24,
            0,
            0,
            majit_ir::make_array_descr(0, 8, Type::Int),
        );
        info.set_parent_descr(majit_ir::descr::make_size_descr(64));

        let mut recorder = Trace::new();
        let vable = recorder.record_input_arg(Type::Ref);
        let mut ctx = TraceCtx::new(
            recorder,
            0,
            std::sync::Arc::new(crate::MetaInterpStaticData::new()),
        );

        let expanded = ctx.emit_vable_field_reads(vable, &info, &[3]);
        assert_eq!(
            expanded.len(),
            4,
            "1 scalar + 3 array items = 4 expanded slots"
        );

        let ops = take_all_ops(ctx);
        // 1 GETFIELD_GC (pc) + 1 GETFIELD_GC_R (locals ptr) + 3 GETARRAYITEM_GC = 5 ops.
        assert_eq!(ops.len(), 5);
        assert_eq!(ops[0].opcode, OpCode::GetfieldGcI);
        assert_eq!(
            ops[0].descr.as_ref().map(|d| d.index()),
            Some(info.static_field_descr(0).index())
        );
        assert_eq!(ops[1].opcode, OpCode::GetfieldGcR);
        assert_eq!(
            ops[1].descr.as_ref().map(|d| d.index()),
            Some(info.array_pointer_field_descr(0).index())
        );
        for k in 0..3 {
            assert_eq!(ops[2 + k].opcode, OpCode::GetarrayitemGcI);
            assert_eq!(
                ops[2 + k].descr.as_ref().map(|d| d.index()),
                Some(info.array_item_descr(0).index())
            );
        }
    }
}
