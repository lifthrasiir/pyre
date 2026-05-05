/// Guard failure handling for the Cranelift backend.
///
/// When a guard fails at runtime, execution exits the JIT-compiled loop
/// and values stay in the JitFrame. The JitFrame GcRef is returned as
/// the deadframe (RPython llmodel.py parity).
///
/// Bridge support: when a guard fails frequently, a bridge trace can be
/// compiled and attached to the fail descriptor. On subsequent guard
/// failures, execution transfers to the bridge instead of returning to
/// the interpreter.
use crate::compiler::{register_gc_roots, unregister_gc_roots};
use majit_backend::{CompiledTraceInfo, ExitRecoveryLayout, FailDescrLayout, TerminalExitLayout};
use majit_gc::GcMap;
use majit_ir::{AccumInfo, Const, DescrRef, FailDescr, GcRef, Type};
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Compiled bridge data attached to a guard's fail descriptor.
///
/// When a bridge is compiled, its code pointer and metadata are stored
/// here so `execute_token` can dispatch to the bridge on guard failure.
pub struct BridgeData {
    /// Compiled trace identifier for this bridge.
    pub trace_id: u64,
    /// Input types expected at the bridge header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this bridge trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to.
    pub source_guard: (u64, u32),
    /// Recovery-layout caller prefix inherited from the source guard.
    pub caller_prefix_layout: Option<ExitRecoveryLayout>,
    /// Function pointer to the bridge's compiled code.
    /// Same calling convention as a compiled loop:
    ///   fn(inputs_ptr: *const i64, outputs_ptr: *mut i64, roots_ptr: *mut i64) -> i64
    pub code_ptr: *const u8,
    /// Fail descriptors within the bridge (guards + finish).
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at construction.
    pub fail_descrs: Box<[Arc<CraneliftFailDescr>]>,
    /// Number of input arguments the bridge expects.
    /// Set to parent guard's fail_arg count (not optimizer-reduced count)
    /// so execute_bridge passes all parent outputs and indices align.
    pub num_inputs: usize,
    /// Number of shadow-root slots the bridge expects.
    pub num_ref_roots: usize,
    /// Maximum output slots for guard exits within the bridge.
    pub max_output_slots: usize,
    /// Static terminal-exit layouts within the bridge trace.
    /// Write-once during bridge compilation, read-only after.
    /// No lock needed — RPython ResumeGuardDescr has no lock (GIL).
    pub terminal_exit_layouts: UnsafeCell<Vec<TerminalExitLayout>>,
    /// When true, a bridge Finish with matching arity should re-enter
    /// the parent loop instead of returning to the interpreter.
    /// Set for bridges that reach the loop's merge_point.
    pub loop_reentry: bool,
    /// compile.py:186: record_loop_or_bridge sets descr.rd_loop_token = clt
    /// on ALL guards (loop and bridge). The bridge shares the parent loop's
    /// invalidation flag (AtomicBool). Holding an Arc clone keeps the flag
    /// alive as long as the bridge exists.
    pub invalidated_arc: Option<Arc<std::sync::atomic::AtomicBool>>,
}

unsafe impl Send for BridgeData {}
unsafe impl Sync for BridgeData {}

impl BridgeData {
    #[inline]
    pub fn terminal_exit_layouts_ref(&self) -> &Vec<TerminalExitLayout> {
        unsafe { &*self.terminal_exit_layouts.get() }
    }

    #[inline]
    pub fn terminal_exit_layouts_mut(&self) -> &mut Vec<TerminalExitLayout> {
        unsafe { &mut *self.terminal_exit_layouts.get() }
    }
}

impl std::fmt::Debug for BridgeData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BridgeData")
            .field("trace_id", &self.trace_id)
            .field("input_types", &self.input_types)
            .field("header_pc", &self.header_pc)
            .field("source_guard", &self.source_guard)
            .field("caller_prefix_layout", &self.caller_prefix_layout)
            .field("code_ptr", &self.code_ptr)
            .field("num_inputs", &self.num_inputs)
            .field("num_ref_roots", &self.num_ref_roots)
            .field("terminal_exit_layouts", unsafe {
                &*self.terminal_exit_layouts.get()
            })
            .finish()
    }
}

/// Concrete fail descriptor used by the Cranelift backend.
///
/// Carries the fail_index and the types of values that will be
/// saved in the DeadFrame on guard failure.
///
/// Also tracks guard failure count and an optional bridge that
/// should be executed instead of returning to the interpreter.
pub struct CraneliftFailDescr {
    pub fail_index: u32,
    pub source_op_index: Option<usize>,
    pub trace_id: u64,
    /// RPython resumedescr.original_greenkey parity: the green_key of
    /// the compiled loop this guard belongs to.
    pub green_key: u64,
    pub fail_arg_types: Vec<Type>,
    pub gc_map: GcMap,
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,
    /// history.py:470-499 TargetToken parity for cross-loop JUMP.
    /// True for external JUMP exits (JUMP whose target TargetToken lives in
    /// a different compiled function). assembler.py:2456-2462 closing_jump
    /// emits a raw JMP to `target_token._ll_loop_code`. Cranelift can't
    /// emit raw inter-function JMPs, so the exit returns to the dispatcher
    /// which reads `target_descr` and re-enters the target loop via the
    /// registered `JitCellToken.number -> RegisteredLoopTarget` metadata.
    /// Mutually exclusive with is_finish.
    pub is_external_jump: bool,
    /// The TargetToken descriptor this JUMP targets, used by the dispatcher
    /// to look up the target loop. Present only when is_external_jump=true.
    /// (history.py:470 TargetToken — identity is the Arc's allocation address,
    /// matching PyPy's `target_tokens_currently_compiling[descr] = None` dict
    /// keyed by descriptor identity.)
    pub target_descr: Option<DescrRef>,
    pub force_token_slots: Vec<usize>,
    /// Write-once during compilation, read-only after.
    /// No lock — RPython ResumeGuardDescr has no lock (GIL).
    pub trace_info: UnsafeCell<Option<CompiledTraceInfo>>,
    /// Write-once during bridge compilation, read-only after.
    pub recovery_layout: UnsafeCell<Option<ExitRecoveryLayout>>,
    /// compile.py:688-692 ResumeGuardDescr.status:
    /// Stores jitcounter hash (from store_hash / fetch_next_hash).
    /// Used by must_compile() to tick the guard's counter slot.
    /// Assigned at compile time, read at guard failure time.
    pub status: std::sync::atomic::AtomicU64,
    /// Number of times this guard has failed (for bridge compilation heuristics).
    pub fail_count: AtomicU32,
    /// schedule.py:654-655 / history.py:143-147 — vector guard metadata
    /// copied from the frontend fail descriptor during lowering.
    pub vector_info: Vec<AccumInfo>,
    /// Compiled bridge attached to this guard, if any.
    /// Write-once when bridge is compiled, read-only after.
    /// No lock — RPython compile.py attach_bridge has no lock (GIL).
    pub bridge: UnsafeCell<Option<BridgeData>>,
    /// Atomic cache of bridge code_ptr for lock-free dispatch.
    pub bridge_code_ptr_cache: std::sync::atomic::AtomicUsize,
    /// Frame slot count required by the attached bridge's prologue. Set
    /// at `attach_bridge` time from `BridgeData::{max_output_slots,
    /// num_inputs, num_ref_roots}`.
    ///
    /// When a guard fires, the parent's already-allocated JITFrame may
    /// have fewer slots than the bridge needs (the parent was sized at
    /// allocation time using the CompiledLoopToken's then-current
    /// `frame_info.jfi_frame_depth`, before this bridge bumped it via
    /// `compiler.rs:13144 update_frame_depth`).  RPython recovers by
    /// running `_frame_realloc_slowpath` (`aarch64/assembler.py:434-493`
    /// → `llmodel.py:127-154 realloc_frame`) which allocates a deeper
    /// JITFrame, copies the old slots, sets `jf_forward`, and continues
    /// into the bridge with the new pointer.  pyre's
    /// `majit-backend/src/jitframe.rs:realloc_frame` ports the helper,
    /// but two pieces are missing for cranelift to call it inline:
    ///   1. cranelift's `run_compiled_code_inner` allocates each JITFrame
    ///      without setting `jf_frame_info`, so `realloc_frame`'s
    ///      `(*old_jf).jf_frame_info` would null-deref;
    ///   2. there is no `cranelift_realloc_jitframe_slowpath` shim
    ///      analogous to dynasm's runtime helper, and
    ///      `emit_attached_bridge_dispatch` cannot call one without
    ///      adding a CFG join.
    /// Until both are wired, `emit_attached_bridge_dispatch` gates
    /// dispatch on `frame_len >= required_frame_len` and falls through
    /// to the deadframe exit when the parent frame is too small.  The
    /// fallback is functionally correct: the deadframe returns to the
    /// interpreter, which re-enters via the green key, allocating a
    /// fresh JITFrame sized from the CLT's now-updated `frame_info`,
    /// and dispatching the bridge cleanly.  Same end-state as RPython,
    /// one extra interpreter round-trip on the guard fire that first
    /// triggers it.
    pub bridge_frame_depth_cache: std::sync::atomic::AtomicUsize,
    /// `compile.py:186` `descr.rd_loop_token = clt` line-by-line port:
    /// the owning `Arc<CompiledLoopToken>`. Late-set by the post-compile
    /// walker that ports `compile.py:183-203 record_loop_or_bridge`.
    /// Together with `CompiledLoopToken.loop_token_wref`
    /// (`compile.py:180-181`) this gives readers a direct chain
    /// `descr.rd_loop_token_clt() -> clt.upgrade -> Arc<JitCellToken>`
    /// matching RPython's `descr.rd_loop_token.loop_token_wref()`
    /// access (`pyjitpl.py:2897`).
    pub rd_loop_token_clt: UnsafeCell<Option<std::sync::Arc<majit_backend::CompiledLoopToken>>>,
    /// Unified-Descr Port Epic Session 5a: back-pointer to the metainterp
    /// `ResumeGuardDescr` Arc the optimizer stamped onto the originating
    /// guard op (`op.descr`).  PyPy keeps a single descr object per
    /// guard (`history.py:121`); pyre's split-descr architecture stores
    /// this Arc as a back-pointer so subsequent Session 5b/c/d can
    /// migrate readers of duplicated fields (`rd_numb`/`rd_consts`/
    /// `rd_virtuals`/`rd_pendingfields`/`fail_arg_types` etc.) to read
    /// through the metainterp Arc instead of the local copy.
    ///
    /// `None` for synthetic backend descrs minted by the runtime
    /// classifier (`compiler.rs::find_descr_by_ptr` for FINISH /
    /// PropagateExceptionDescr / ExitFrameWithExceptionDescr exits) —
    /// those exits route through dedicated metainterp Done* descrs
    /// owned by `MetaInterpStaticData`, not via `op.descr`.
    pub meta_descr: Option<DescrRef>,
}

impl std::fmt::Debug for CraneliftFailDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CraneliftFailDescr")
            .field("fail_index", &self.fail_index)
            .field("source_op_index", &self.source_op_index)
            .field("trace_id", &self.trace_id)
            .field("fail_arg_types", &self.fail_arg_types)
            .field("gc_map", &self.gc_map)
            .field("is_finish", &self.is_finish)
            .field("is_external_jump", &self.is_external_jump)
            .field(
                "target_descr",
                &self.target_descr.as_ref().map(|d| d.repr()),
            )
            .field("force_token_slots", &self.force_token_slots)
            .field("trace_info", unsafe { &*self.trace_info.get() })
            .field("recovery_layout", unsafe { &*self.recovery_layout.get() })
            .field("fail_count", &self.fail_count.load(Ordering::Relaxed))
            .field("vector_info", &self.vector_info)
            .field("has_bridge", &unsafe { &*self.bridge.get() }.is_some())
            .finish()
    }
}

// Safety: CraneliftFailDescr is accessed from a single thread (the JIT thread).
// UnsafeCell fields (bridge, trace_info, recovery_layout) are write-once during
// compilation and read-only thereafter. RPython's ResumeGuardDescr has no locks
// (GIL-protected). pyre is single-threaded (no-GIL, single thread).
unsafe impl Send for CraneliftFailDescr {}
unsafe impl Sync for CraneliftFailDescr {}

impl CraneliftFailDescr {
    fn gc_map_for_types(fail_arg_types: &[Type], force_token_slots: &[usize]) -> GcMap {
        let mut gc_map = GcMap::new();
        for (slot, tp) in fail_arg_types.iter().enumerate() {
            if *tp == Type::Ref && !force_token_slots.contains(&slot) {
                gc_map.set_ref(slot);
            }
        }
        gc_map
    }

    /// Create a new fail descriptor.
    pub fn new(fail_index: u32, fail_arg_types: Vec<Type>) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            false,
            Vec::new(),
            None,
        )
    }

    pub fn new_with_kind(fail_index: u32, fail_arg_types: Vec<Type>, is_finish: bool) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            Vec::new(),
            None,
        )
    }

    pub fn new_with_kind_and_force_tokens(
        fail_index: u32,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        force_token_slots: Vec<usize>,
    ) -> Self {
        Self::new_with_trace_and_kind_and_force_tokens(
            fail_index,
            0,
            fail_arg_types,
            is_finish,
            force_token_slots,
            None,
        )
    }

    pub fn new_with_trace_and_kind_and_force_tokens(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        is_finish: bool,
        mut force_token_slots: Vec<usize>,
        recovery_layout: Option<ExitRecoveryLayout>,
    ) -> Self {
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            source_op_index: None,
            trace_id,
            green_key: 0,
            gc_map: Self::gc_map_for_types(&fail_arg_types, &force_token_slots),
            fail_arg_types,
            is_finish,
            is_exit_frame_with_exception: false,
            is_external_jump: false,
            target_descr: None,
            force_token_slots,
            trace_info: UnsafeCell::new(None),
            recovery_layout: UnsafeCell::new(recovery_layout),
            status: std::sync::atomic::AtomicU64::new(0),
            fail_count: AtomicU32::new(0),
            vector_info: Vec::new(),
            bridge: UnsafeCell::new(None),
            bridge_code_ptr_cache: std::sync::atomic::AtomicUsize::new(0),
            bridge_frame_depth_cache: std::sync::atomic::AtomicUsize::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            meta_descr: None,
        }
    }

    /// Construct a fail descriptor for an external JUMP exit.
    /// assembler.py:2456-2462 closing_jump parity: JUMP whose target
    /// TargetToken lives in a different compiled function. Cranelift can't
    /// emit raw inter-function JMPs, so the dispatcher receives this descr
    /// and re-enters the target loop via the registered target token.
    pub fn new_external_jump(
        fail_index: u32,
        trace_id: u64,
        fail_arg_types: Vec<Type>,
        mut force_token_slots: Vec<usize>,
        recovery_layout: Option<ExitRecoveryLayout>,
        target_descr: DescrRef,
    ) -> Self {
        force_token_slots.sort_unstable();
        force_token_slots.dedup();
        CraneliftFailDescr {
            fail_index,
            source_op_index: None,
            trace_id,
            green_key: 0,
            gc_map: Self::gc_map_for_types(&fail_arg_types, &force_token_slots),
            fail_arg_types,
            is_finish: false,
            is_exit_frame_with_exception: false,
            is_external_jump: true,
            target_descr: Some(target_descr),
            force_token_slots,
            trace_info: UnsafeCell::new(None),
            recovery_layout: UnsafeCell::new(recovery_layout),
            status: std::sync::atomic::AtomicU64::new(0),
            fail_count: AtomicU32::new(0),
            vector_info: Vec::new(),
            bridge: UnsafeCell::new(None),
            bridge_code_ptr_cache: std::sync::atomic::AtomicUsize::new(0),
            bridge_frame_depth_cache: std::sync::atomic::AtomicUsize::new(0),
            rd_loop_token_clt: UnsafeCell::new(None),
            meta_descr: None,
        }
    }

    /// `compile.py:186` write side: invoked by the post-compile walker
    /// once per ResumeDescr in the newly-compiled trace.  Stamps the
    /// owning `Arc<CompiledLoopToken>`.
    pub fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<majit_backend::CompiledLoopToken>) {
        unsafe { *self.rd_loop_token_clt.get() = Some(clt) };
    }

    /// `compile.py:186` reader for the clt-typed slot.
    pub fn rd_loop_token_clt(&self) -> Option<&std::sync::Arc<majit_backend::CompiledLoopToken>> {
        unsafe { (*self.rd_loop_token_clt.get()).as_ref() }
    }

    // UnsafeCell accessor helpers — single-threaded, no lock needed.
    // RPython ResumeGuardDescr fields are plain attributes (GIL-protected).

    #[inline]
    pub fn bridge_ref(&self) -> &Option<BridgeData> {
        unsafe { &*self.bridge.get() }
    }

    #[inline]
    pub fn trace_info_ref(&self) -> &Option<CompiledTraceInfo> {
        unsafe { &*self.trace_info.get() }
    }

    #[inline]
    pub fn recovery_layout_ref(&self) -> &Option<ExitRecoveryLayout> {
        unsafe { &*self.recovery_layout.get() }
    }

    /// Increment the failure counter and return the new value.
    pub fn increment_fail_count(&self) -> u32 {
        self.fail_count.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Get the current failure count.
    pub fn get_fail_count(&self) -> u32 {
        self.fail_count.load(Ordering::Relaxed)
    }

    /// Whether a bridge has been attached to this guard.
    pub fn has_bridge(&self) -> bool {
        self.bridge_code_ptr_cache
            .load(std::sync::atomic::Ordering::Relaxed)
            != 0
    }

    /// Get bridge code_ptr without Mutex lock (atomic read).
    pub fn bridge_code_ptr(&self) -> *const u8 {
        self.bridge_code_ptr_cache
            .load(std::sync::atomic::Ordering::Relaxed) as *const u8
    }

    /// Attach a compiled bridge to this guard.
    pub fn attach_bridge(&self, bridge: BridgeData) {
        let code_ptr = bridge.code_ptr as usize;
        let frame_depth = bridge
            .max_output_slots
            .max(bridge.num_inputs)
            .max(1)
            .saturating_add(bridge.num_ref_roots);
        unsafe { *self.bridge.get() = Some(bridge) };
        self.bridge_frame_depth_cache
            .store(frame_depth, std::sync::atomic::Ordering::Release);
        self.bridge_code_ptr_cache
            .store(code_ptr, std::sync::atomic::Ordering::Release);
    }

    // compile.py:687-696 status encoding constants.
    pub const ST_BUSY_FLAG: u64 = 0x01;
    pub const ST_TYPE_MASK: u64 = 0x06;
    pub const ST_SHIFT: u32 = 3;
    pub const ST_SHIFT_MASK: u64 = !((1u64 << Self::ST_SHIFT) - 1); // -(1 << ST_SHIFT)
    pub const TY_NONE: u64 = 0x00;
    pub const TY_INT: u64 = 0x02;
    pub const TY_REF: u64 = 0x04;
    pub const TY_FLOAT: u64 = 0x06;

    /// compile.py:826-830 store_hash: assign a unique jitcounter hash.
    /// `self.status = hash & self.ST_SHIFT_MASK`
    pub fn store_hash(&self, hash: u64) {
        self.status.store(
            hash & Self::ST_SHIFT_MASK,
            std::sync::atomic::Ordering::Release,
        );
    }

    /// compile.py:741-745: read status for must_compile.
    pub fn get_status(&self) -> u64 {
        self.status.load(std::sync::atomic::Ordering::Acquire)
    }

    /// compile.py:786-788: start_compiling — set ST_BUSY_FLAG.
    pub fn start_compiling(&self) {
        self.status
            .fetch_or(Self::ST_BUSY_FLAG, std::sync::atomic::Ordering::AcqRel);
    }

    /// compile.py:790-795: done_compiling — clear ST_BUSY_FLAG.
    pub fn done_compiling(&self) {
        self.status
            .fetch_and(!Self::ST_BUSY_FLAG, std::sync::atomic::Ordering::AcqRel);
    }

    /// compile.py:750: check ST_BUSY_FLAG.
    pub fn is_compiling(&self) -> bool {
        self.status.load(std::sync::atomic::Ordering::Acquire) & Self::ST_BUSY_FLAG != 0
    }

    /// compile.py:813-824: make_a_counter_per_value — for GUARD_VALUE,
    /// encode the fail_arg index and type tag in status.
    /// `self.status = ty | (index << ST_SHIFT)`
    pub fn make_a_counter_per_value(&self, index: u32, type_tag: u64) {
        let status = type_tag | ((index as u64) << Self::ST_SHIFT);
        self.status
            .store(status, std::sync::atomic::Ordering::Release);
    }

    /// Take the bridge data out of this fail descriptor, leaving None.
    pub fn take_bridge(&self) -> Option<BridgeData> {
        let bridge = unsafe { &mut *self.bridge.get() }.take();
        if bridge.is_some() {
            self.bridge_code_ptr_cache
                .store(0, std::sync::atomic::Ordering::Release);
            self.bridge_frame_depth_cache
                .store(0, std::sync::atomic::Ordering::Release);
        }
        bridge
    }

    pub fn set_recovery_layout(&self, recovery_layout: ExitRecoveryLayout) {
        unsafe { *self.recovery_layout.get() = Some(recovery_layout) };
    }

    pub fn set_source_op_index(&mut self, source_op_index: usize) {
        self.source_op_index = Some(source_op_index);
    }

    pub fn set_trace_info(&self, trace_info: CompiledTraceInfo) {
        unsafe { *self.trace_info.get() = Some(trace_info) };
    }

    pub fn gc_map(&self) -> &GcMap {
        &self.gc_map
    }

    pub fn is_finish(&self) -> bool {
        self.is_finish
    }

    pub fn is_force_token_slot(&self, slot: usize) -> bool {
        self.force_token_slots.binary_search(&slot).is_ok()
    }

    pub fn layout(&self) -> FailDescrLayout {
        let gc_ref_slots = self
            .fail_arg_types
            .iter()
            .enumerate()
            .filter_map(|(slot, _)| self.gc_map.is_ref(slot).then_some(slot))
            .collect();
        let recovery = unsafe { &*self.recovery_layout.get() }.clone();
        let frame_stack = recovery.as_ref().map(|r| r.frames.clone());
        // resume.py:450-488 propagate rd_* for post-eviction reconstruction.
        // Read through the metainterp ResumeGuardDescr Arc (Session 5b
        // back-pointer) — single source of truth.
        let meta_fd = self.meta_descr.as_ref().and_then(|m| m.as_fail_descr());
        FailDescrLayout {
            fail_index: self.fail_index,
            source_op_index: self.source_op_index,
            // Session 5c: read trace_id through meta_descr too.
            trace_id: meta_fd.map_or(self.trace_id, |fd| fd.trace_id()),
            trace_info: unsafe { &*self.trace_info.get() }.clone(),
            fail_arg_types: self.fail_arg_types.clone(),
            is_finish: self.is_finish,
            gc_ref_slots,
            force_token_slots: self.force_token_slots.clone(),
            recovery_layout: recovery,
            frame_stack,
            rd_numb: meta_fd.and_then(|fd| fd.rd_numb()).map(|s| s.to_vec()),
            rd_consts: meta_fd.and_then(|fd| fd.rd_consts()).map(|s| s.to_vec()),
            rd_virtuals: meta_fd.and_then(|fd| fd.rd_virtuals()).map(|s| s.to_vec()),
            rd_pendingfields: meta_fd
                .and_then(|fd| fd.rd_pendingfields())
                .map(|s| s.to_vec()),
        }
    }
}

impl majit_ir::Descr for CraneliftFailDescr {
    fn index(&self) -> u32 {
        self.fail_index
    }

    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }

    /// `compile.py:185` `isinstance(descr, ResumeDescr)` parity. Backend
    /// `CraneliftFailDescr` plays the role of upstream's
    /// `ResumeGuardDescr` for guard-failure exits, of the
    /// `DoneWithThisFrame*` / `ExitFrameWithExceptionDescr` family for
    /// finish exits, and of `TargetToken` for external JUMP exits (the
    /// dispatcher-routed cross-loop JUMP path).  Only the first is a
    /// `ResumeDescr` in upstream; finish descrs and `TargetToken`s are
    /// distinct class hierarchies and `compile.py:185` skips them.
    ///
    /// Session 5d: forward through metainterp ResumeGuardDescr Arc when
    /// available.  `is_external_jump` short-circuits to false because
    /// cranelift's external-JUMP descrs are backend-only synthetic
    /// objects with no metainterp counterpart (meta_descr is None for
    /// them anyway, but the early-return makes the intent explicit).
    fn is_resume_guard(&self) -> bool {
        if self.is_external_jump {
            return false;
        }
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .map_or(!self.is_finish, |fd| fd.is_resume_guard())
    }
}

impl FailDescr for CraneliftFailDescr {
    fn fail_index(&self) -> u32 {
        self.fail_index
    }

    fn fail_index_per_trace(&self) -> u32 {
        // The backend descr's structural `fail_index` IS the per-trace
        // key — `assembler.py:227 self.faildescr.index = i` is allocated
        // per-trace at backend compile time.  Only the metainterp side
        // distinguishes a global `fail_index` (alloc_fail_index counter)
        // from the per-trace key; the backend has only the per-trace
        // value.  Override the trait default (0) so that callers that
        // receive the backend descr through `bridge_source_descr`'s
        // fallback chain (mod.rs:7713) can still locate the source guard.
        self.fail_index
    }

    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }

    fn is_finish(&self) -> bool {
        // Session 5d: forward through metainterp ResumeGuardDescr Arc.
        // DoneWithThisFrame{Void,Int,Ref,Float} and ExitFrameWithExceptionDescrRef
        // override `is_finish() -> true` on the metainterp side; the rest
        // — including `PropagateExceptionDescr` (`compile.py:1092`
        // `class PropagateExceptionDescr(AbstractFailDescr)` inherits
        // `final_descr = False`, see `compile.rs:2314`) — take the trait
        // default `false`.
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .map_or(self.is_finish, |fd| fd.is_finish())
    }

    fn is_exit_frame_with_exception(&self) -> bool {
        // Session 5d: forward through metainterp ResumeGuardDescr Arc.
        // ExitFrameWithExceptionDescrRef overrides on the metainterp side.
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .map_or(self.is_exit_frame_with_exception, |fd| {
                fd.is_exit_frame_with_exception()
            })
    }

    fn is_external_jump(&self) -> bool {
        // Session 5d: backend-only flag, no metainterp counterpart —
        // external-JUMP descrs are synthesized at the cranelift backend
        // for cross-loop JUMP targets and have meta_descr == None.
        self.is_external_jump
    }

    fn target_descr(&self) -> Option<&DescrRef> {
        self.target_descr.as_ref()
    }

    fn trace_id(&self) -> u64 {
        // Session 5c: forward through metainterp ResumeGuardDescr Arc
        // (Session 5a back-pointer).  Single source of truth — the
        // metainterp side stamps trace_id in record_loop_or_bridge
        // (compile.py:185 line-by-line counterpart).  Fallback to
        // local field for synthetic test descrs that bypass the
        // assembler-time meta_descr stamp.
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .map_or(self.trace_id, |fd| fd.trace_id())
    }

    fn rd_loop_token_clt(&self) -> Option<&dyn std::any::Any> {
        CraneliftFailDescr::rd_loop_token_clt(self).map(|arc| arc as &dyn std::any::Any)
    }

    fn set_rd_loop_token_clt(&self, clt: std::sync::Arc<dyn std::any::Any + Send + Sync>) {
        let typed: std::sync::Arc<majit_backend::CompiledLoopToken> = clt
            .downcast::<majit_backend::CompiledLoopToken>()
            .expect("set_rd_loop_token_clt expected Arc<CompiledLoopToken>");
        CraneliftFailDescr::set_rd_loop_token_clt(self, typed);
    }

    fn is_gc_ref_slot(&self, slot: usize) -> bool {
        self.gc_map.is_ref(slot)
    }

    fn force_token_slots(&self) -> &[usize] {
        &self.force_token_slots
    }

    fn vector_info(&self) -> Vec<AccumInfo> {
        self.vector_info.clone()
    }

    fn get_status(&self) -> u64 {
        self.get_status()
    }

    fn start_compiling(&self) {
        self.start_compiling()
    }

    fn done_compiling(&self) {
        self.done_compiling()
    }

    fn is_compiling(&self) -> bool {
        self.is_compiling()
    }

    // resume.py:450-488 readers: forward to the metainterp ResumeGuardDescr
    // Arc captured at assembly time (Session 5a back-pointer).  Single
    // source of truth — drops the duplicated local rd_* copies.
    fn rd_numb(&self) -> Option<&[u8]> {
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .and_then(|fd| fd.rd_numb())
    }
    fn rd_consts(&self) -> Option<&[majit_ir::Const]> {
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .and_then(|fd| fd.rd_consts())
    }
    fn rd_virtuals(&self) -> Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]> {
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .and_then(|fd| fd.rd_virtuals())
    }
    fn rd_pendingfields(&self) -> Option<&[majit_ir::GuardPendingFieldEntry]> {
        self.meta_descr
            .as_ref()
            .and_then(|m| m.as_fail_descr())
            .and_then(|fd| fd.rd_pendingfields())
    }
}

// ── JitFrameDeadFrame (llmodel.py deadframe-as-jitframe parity) ─────

/// RPython llmodel.py parity: the deadframe IS the JitFrame.
///
/// In RPython, `execute_token` returns the JitFrame GCREF directly as
/// the deadframe. Values stay in `jf_frame[]` — no copying to `Vec<i64>`.
/// `get_int_value(deadframe, index)` reads directly from `jf_frame[index]`.
pub struct JitFrameDeadFrame {
    /// GcRef pointing to the heap-allocated JitFrame.
    pub jf_gcref: GcRef,
    /// The fail descriptor for this exit.
    pub fail_descr: Arc<CraneliftFailDescr>,
    /// Original attached `jf_descr` identity for finish exits emitted by
    /// the metainterp (`DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`).
    pub latest_descr: Option<DescrRef>,
    /// True when `register_roots` has registered `jf_gcref` with the
    /// active cranelift GC, so `Drop` knows to remove it. Replaces the
    /// pre-removal `gc_runtime_id` field that paired registration with
    /// a per-trace runtime id; the active GC is now a single thread-local
    /// (`compiler.rs CRANELIFT_ACTIVE_GC`, mirroring `llmodel.py:58`).
    pub roots_registered: bool,
    /// Keeps the frame memory alive for non-GC allocations.
    pub _heap_owner: Option<Vec<i64>>,
}

/// Byte offset from JitFrame start to jf_frame[0].
const JF_FRAME_ITEM0_BYTES: usize = 64;
/// Byte offset to jf_savedata field.
const JF_SAVEDATA_BYTES: usize = 32;
/// Byte offset to jf_guard_exc field.
const JF_GUARD_EXC_BYTES: usize = 40;

impl JitFrameDeadFrame {
    pub fn new(
        jf_gcref: GcRef,
        fail_descr: Arc<CraneliftFailDescr>,
        latest_descr: Option<DescrRef>,
        heap_owner: Option<Vec<i64>>,
    ) -> Self {
        JitFrameDeadFrame {
            jf_gcref,
            fail_descr,
            latest_descr,
            roots_registered: false,
            _heap_owner: heap_owner,
        }
    }

    pub fn register_roots(&mut self) {
        self.roots_registered = register_gc_roots(std::slice::from_mut(&mut self.jf_gcref));
    }

    #[inline]
    pub fn get_int(&self, index: usize) -> i64 {
        unsafe { *((self.jf_gcref.0 + JF_FRAME_ITEM0_BYTES + index * 8) as *const i64) }
    }

    #[inline]
    pub fn get_float(&self, index: usize) -> f64 {
        f64::from_bits(self.get_int(index) as u64)
    }

    #[inline]
    pub fn get_ref(&self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    pub fn take_ref_for_call_result(&mut self, index: usize) -> GcRef {
        GcRef(self.get_int(index) as usize)
    }

    #[inline]
    pub fn get_savedata_ref(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref.0 + JF_SAVEDATA_BYTES) as *const usize) })
    }

    #[inline]
    pub fn try_get_savedata_ref(&self) -> Option<GcRef> {
        let r = self.get_savedata_ref();
        if r.is_null() { None } else { Some(r) }
    }

    #[inline]
    pub fn set_savedata_ref(&mut self, data: GcRef) {
        unsafe { *((self.jf_gcref.0 + JF_SAVEDATA_BYTES) as *mut usize) = data.0 };
    }

    #[inline]
    pub fn grab_exc_value(&self) -> GcRef {
        GcRef(unsafe { *((self.jf_gcref.0 + JF_GUARD_EXC_BYTES) as *const usize) })
    }
}

impl Drop for JitFrameDeadFrame {
    fn drop(&mut self) {
        if self.roots_registered {
            unregister_gc_roots(std::slice::from_mut(&mut self.jf_gcref));
        }
    }
}
