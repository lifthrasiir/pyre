use majit_backend::ExitValueSourceLayout;

/// RPython resume.py:993-1007: materialize deferred virtualizable SetfieldGc.
fn materialize_pending_fields(exit_layout: &CompiledExitLayout, raw_values: &[i64]) {
    let Some(ref recovery) = exit_layout.recovery_layout else {
        return;
    };
    for pf in &recovery.pending_field_layouts {
        let struct_ptr = match &pf.target {
            ExitValueSourceLayout::ExitValue(slot) if *slot < raw_values.len() => {
                raw_values[*slot] as *mut u8
            }
            ExitValueSourceLayout::Constant(c) => *c as *mut u8,
            _ => std::ptr::null_mut(),
        };
        let value = match &pf.value {
            ExitValueSourceLayout::ExitValue(slot) if *slot < raw_values.len() => raw_values[*slot],
            ExitValueSourceLayout::Constant(c) => *c,
            _ => 0,
        };
        if struct_ptr.is_null() {
            continue;
        }
        // Virtualizable field offsets from descriptor index encoding:
        // 0x8000_0003 = head (offset 0), 0x8000_0004 = size (offset 8)
        let (offset, size) = match pf.descr_index {
            idx if idx & 0x8000_0000 != 0 => match idx {
                0x8000_0003 => (0, 8),
                0x8000_0004 => (8, 8),
                _ => continue,
            },
            idx => ((idx >> 2) as usize, 8),
        };
        unsafe {
            let p = struct_ptr.add(offset);
            match size {
                8 => *(p as *mut i64) = value,
                4 => *(p as *mut i32) = value as i32,
                _ => {}
            }
        }
    }
}

use crate::TraceAction;
use crate::blackhole::ExceptionState;
use crate::jit_state::JitState;
use crate::pyjitpl::{
    AbortReason, BackEdgeAction, CompiledExitLayout, DetailedDriverRunOutcome, InlineDecision,
    JitStats, MetaInterp,
};
use crate::resume::ResumeLayoutSummary;
use crate::trace_ctx::TraceCtx;
use crate::virtualizable::VirtualizableInfo;
use majit_gc::GcAllocator;
use majit_ir::OpRef;
use majit_ir::descr::DescrRef;
use majit_ir::{Const, GreenKey, GreenType, JitDriverVar, Type, Value, VarKind};

/// Descriptor for a JitDriver's variable layout.
///
/// Mirrors RPython's `JitDriver(greens=[...], reds=[...])`:
/// - `greens` are compile-time constants identifying the loop header
/// - `reds` are runtime values carried as InputArgs
///
/// The interpreter declares this once per JitDriver and passes it to
/// MetaInterp for structured green/red handling.
#[derive(Clone, Debug)]
pub struct JitDriverStaticData {
    /// warmspot.py / pyjitpl.py parity: registered driver slot index.
    ///
    /// RPython stores this on `jitdriver_sd.index` and threads it
    /// through portal-frame bookkeeping plus `JitCellToken` ownership.
    /// It stays `None` until `register_jitdriver_sd` assigns the slot.
    pub index: Option<usize>,
    /// All variables in declaration order.
    pub vars: Vec<JitDriverVar>,
    /// Optional name of the virtualizable red variable.
    pub virtualizable: Option<String>,
    /// warmspot.py:449 jd.result_type — portal function return type.
    /// Determined once at driver setup from the portal's return signature.
    pub result_type: Type,
    /// rpython/rlib/jit.py JitDriver(is_recursive=True) parity.
    /// Mirrors RPython's `jitcode.jitdriver_sd.jitdriver.is_recursive`
    /// access — see pyjitpl.py:2427-2429 `is_main_jitcode`.  Pyre stores
    /// the flag directly on the static data so MetaInterp can decide
    /// without touching the runtime `JitDriver` object.
    pub is_recursive: bool,
    /// warmspot.py:921 `jd.mainjitcode = pyjitpl_class(jd, mainjitcode)`.
    ///
    /// The portal jitcode for this driver — the entry point of the
    /// recursive function being JITed.  Read by
    /// `MetaInterp.do_recursive_call` (pyjitpl.py:1427) to set
    /// `portal_code.calldescr` for the residual CALL_ASSEMBLER op.
    pub mainjitcode: Option<std::sync::Arc<crate::jitcode::JitCode>>,
    /// warmspot.py:946 `jd.portal_runner_adr = adr_of(portal_runner)`.
    ///
    /// Address of the portal_runner C function — the funcbox for
    /// `do_recursive_call`'s residual CALL_ASSEMBLER op
    /// (pyjitpl.py:1428-1429).
    pub portal_runner_adr: i64,
    /// jitdriver.py:16 + warmspot.py:520-545 `jd.virtualizable_info`.
    ///
    /// Per-driver `VirtualizableInfo` populated during warmspot setup
    /// (`make_virtualizable_infos`).  Read by
    /// `MIFrame._do_jit_force_virtual` (pyjitpl.py:2155),
    /// `vable_and_vrefs_before_residual_call` (pyjitpl.py:3320), and
    /// peers that consult `self.metainterp.jitdriver_sd.virtualizable_info`.
    /// Pyre keeps a duplicate single-slot on `MetaInterp` for the
    /// production path that hasn't yet rerouted through this slot —
    /// `MetaInterp::register_jitdriver_sd` propagates from MetaInterp
    /// to here when both are present so per-driver readers stay in
    /// sync without forcing the production registration refactor.
    pub virtualizable_info: Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>>,
    /// jitdriver.py:17 + warmspot.py:520-525 `jd.greenfield_info`.
    ///
    /// Per-driver `GreenFieldInfo` populated when `jd.jitdriver.greens`
    /// contains a dotted name.  Read by `_do_jit_force_virtual`
    /// (pyjitpl.py:2156) — the bail check fires when both this slot
    /// and `virtualizable_info` are None.
    pub greenfield_info: Option<crate::greenfield::GreenFieldInfo>,
    /// jitdriver.py:28 + warmspot.py:529/538 `jd.index_of_virtualizable`.
    ///
    /// Index inside the red-arg list pointing to the virtualizable
    /// argument; -1 when the driver has no virtualizable.  Mirrored
    /// by `virtualizable_arg_index()` so legacy callers keep
    /// returning `Option<usize>`.
    pub index_of_virtualizable: i32,
    /// jitdriver.py:11 `self.portal_calldescr ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:1013` `jd.portal_calldescr = self.cpu.calldescrof(...)`.
    /// The `CallDescr` describing the portal_runner signature; read by
    /// `compile_tmp_callback` (`compile.py:1131-1132`) to pick the
    /// `CALL_*` opcode and attach as the residual call's descr.
    pub portal_calldescr: Option<majit_ir::DescrRef>,
    /// jitdriver.py:21 `self.portal_finishtoken ... rpython.jit.metainterp.pyjitpl`.
    ///
    /// `pyjitpl.py:2279-2280` sets this to the type-appropriate
    /// `MetaInterpStaticData.done_with_this_frame_descr_{int,ref,float,void}`
    /// singleton.  Used as the `FINISH` descr emitted by
    /// `compile_tmp_callback` (`compile.py:1142`).
    pub portal_finishtoken: Option<majit_ir::DescrRef>,
    /// jitdriver.py:22 `self.propagate_exc_descr .. rpython.jit.metainterp.pyjitpl`.
    ///
    /// `pyjitpl.py:2281` shares a single `PropagateExceptionDescr`
    /// instance across all jitdrivers.  Used as the `GUARD_NO_EXCEPTION`
    /// descr emitted by `compile_tmp_callback` (`compile.py:1141`).
    pub propagate_exc_descr: Option<majit_ir::DescrRef>,
    /// jitdriver.py:14 `self.red_args_types ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:664` `red_args_types = [history.getkind(TYPE)[0] for
    /// TYPE in jd._JIT_ENTER_FUNCTYPE.ARGS[jd.num_green_args:]]`.
    /// One char per red arg: 'i' / 'r' / 'f'. Consumed by
    /// `compile_tmp_callback` (`compile.py:1107-1118`) to decide the
    /// `CALL_*` opcode for each red and to size `rd_consts` /
    /// `rd_unicodes` arrays.
    ///
    /// Populated at construction from `vars[red].tp.to_char()`.
    pub red_args_types: Vec<char>,
    /// jitdriver.py:20 `self.no_loop_header ... rpython.jit.metainterp.warmspot`.
    ///
    /// `warmspot.py:558` sets True when the codewriter has not emitted a
    /// `loop_header` op for this driver — read by `MIFrame.opimpl_jit_
    /// merge_point` (`pyjitpl.py:1571-1574`) to skip the implicit
    /// `can_enter_jit` check at the merge point.
    ///
    /// Default `false` matches upstream's "attribute absent until
    /// warmspot decides otherwise" state.
    pub no_loop_header: bool,
    /// jitdriver.py:27 `self.assembler_helper_adr` (CALL_ASSEMBLER).
    ///
    /// `warmspot.py:1031-1035` `jd.assembler_helper_adr =
    /// llmemory.cast_ptr_to_adr(self.cpu.cast_ptr_to_int(
    /// assembler_helper_ptr))`. Address of the per-driver
    /// `assembler_helper` callable invoked by the backend's
    /// `CALL_ASSEMBLER` op to handle the `done_with_this_frame_*` /
    /// `propagate_exception` post-loop branch.
    ///
    /// `0` mirrors upstream's "attribute absent" / "not yet wired" state
    /// — populated by S2.1 alongside `portal_runner_adr`.
    pub assembler_helper_adr: i64,
    /// jitdriver.py:29 `self.vable_token_descr` (CALL_ASSEMBLER).
    ///
    /// `warmspot.py:546` `jd.vable_token_descr = self.cpu.fielddescrof(
    /// VABLERTI, 'vable_token')`. Used by the backend's `CALL_ASSEMBLER`
    /// op to load / clear the virtualizable's `vable_token` slot when
    /// crossing the assembler-helper boundary.
    ///
    /// `None` until the per-driver virtualizable layout is known —
    /// populated alongside `virtualizable_info` once the host runtime
    /// supplies the field descriptor.
    pub vable_token_descr: Option<majit_ir::DescrRef>,
}

impl JitDriverStaticData {
    /// Create a descriptor from green and red variable lists.
    pub fn new(greens: Vec<(&str, Type)>, reds: Vec<(&str, Type)>) -> Self {
        Self::with_virtualizable(greens, reds, None)
    }

    /// Create a descriptor with optional virtualizable metadata.
    pub fn with_virtualizable(
        greens: Vec<(&str, Type)>,
        reds: Vec<(&str, Type)>,
        virtualizable: Option<&str>,
    ) -> Self {
        let mut vars = Vec::new();
        for (name, tp) in greens {
            vars.push(JitDriverVar::green(name, tp));
        }
        for (name, tp) in reds {
            vars.push(JitDriverVar::red(name, tp));
        }
        // warmspot.py:664 — `red_args_types` carries one history-kind
        // char per red arg in declaration order. Built from the red-only
        // slice of `vars` so adding a green/red split helper later (S2)
        // does not need a parallel update path here.
        let red_args_types: Vec<char> = vars
            .iter()
            .filter(|v| v.kind == VarKind::Red)
            .map(|v| v.tp.to_char())
            .collect();
        let mut sd = JitDriverStaticData {
            index: None,
            vars,
            virtualizable: virtualizable.map(str::to_string),
            result_type: Type::Ref,
            is_recursive: false,
            mainjitcode: None,
            portal_runner_adr: 0,
            virtualizable_info: None,
            greenfield_info: None,
            index_of_virtualizable: -1,
            portal_calldescr: None,
            portal_finishtoken: None,
            propagate_exc_descr: None,
            red_args_types,
            no_loop_header: false,
            assembler_helper_adr: 0,
            vable_token_descr: None,
        };
        // warmspot.py:529/538 — keep `index_of_virtualizable` in sync
        // with the `virtualizable_arg_index()` derived from `reds`.
        // Stays `-1` when the driver has no virtualizable.
        if let Some(idx) = sd.virtualizable_arg_index() {
            sd.index_of_virtualizable = idx as i32;
        }
        sd
    }

    /// `warmspot.py:1013-1017` `jd.portal_calldescr = self.cpu.calldescrof(
    ///     jd._PTR_PORTAL_FUNCTYPE.TO, ...ARGS, ...RESULT, EffectInfo.MOST_GENERAL)`.
    ///
    /// Populates `self.portal_calldescr` from `self.vars` (green + red
    /// types, in declaration order) and `self.result_type`.  Idempotent:
    /// a fresh descriptor is built every call so subsequent mutations of
    /// `result_type` propagate.
    pub fn build_portal_calldescr(&mut self) {
        use majit_ir::{EffectInfo, SimpleCallDescr};
        use std::sync::atomic::{AtomicU32, Ordering};
        // `warmspot.py:1013` passes `EffectInfo.MOST_GENERAL` — the portal
        // runner can do anything.
        static NEXT_IDX: AtomicU32 = AtomicU32::new(0x4100_0000);
        let idx = NEXT_IDX.fetch_add(1, Ordering::Relaxed);
        // `warmspot.py:1015` `jd._PTR_PORTAL_FUNCTYPE.TO.ARGS`: the full
        // argument list of the portal runner, i.e. green + red (same
        // order as `vars`).  pyre keeps the same layout via `VarKind`.
        let arg_types: Vec<Type> = self.vars.iter().map(|v| v.tp).collect();
        // `descr.py:478` `result_signed = True` for signed ints; pyre
        // defaults to signed since `Type::Int` is the 64-bit signed slot.
        let result_size = match self.result_type {
            Type::Void => 0,
            _ => 8,
        };
        let descr: majit_ir::DescrRef = std::sync::Arc::new(SimpleCallDescr::new(
            idx,
            arg_types,
            self.result_type,
            true,
            result_size,
            EffectInfo::MOST_GENERAL,
        ));
        self.portal_calldescr = Some(descr);
    }

    /// Get only the green variables.
    pub fn greens(&self) -> Vec<&JitDriverVar> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .collect()
    }

    /// Get only the red variables.
    pub fn reds(&self) -> Vec<&JitDriverVar> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Red)
            .collect()
    }

    /// Number of green variables.
    pub fn num_greens(&self) -> usize {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .count()
    }

    /// Number of red variables.
    pub fn num_reds(&self) -> usize {
        self.vars.iter().filter(|v| v.kind == VarKind::Red).count()
    }

    /// jitdriver.py:12 `self.num_green_args ... rpython.jit.metainterp.warmspot`.
    ///
    /// Upstream-canonical name; alias of [`Self::num_greens`]. Matches
    /// `warmspot.py:665` `jd.num_green_args = len(jd._green_args_spec)`
    /// callers (`compile.py:1107-1118` `compile_tmp_callback`,
    /// `pyjitpl.py:1530-1535` green-constant assertion).
    pub fn num_green_args(&self) -> usize {
        self.num_greens()
    }

    /// jitdriver.py:13 `self.num_red_args ... rpython.jit.metainterp.warmspot`.
    ///
    /// Upstream-canonical name; alias of [`Self::num_reds`]. Matches
    /// `warmspot.py:666` `jd.num_red_args = len(red_args_types)` callers
    /// (`compile.py:1107-1118` red-arg loop bounds).
    pub fn num_red_args(&self) -> usize {
        self.num_reds()
    }

    /// `warmspot.py:663` `jd._green_args_spec = [TYPE for ARG, TYPE in
    /// jd._JIT_ENTER_FUNCTYPE.ARGS[:jd.num_green_args]]`.
    ///
    /// Returns the per-green-arg `GreenType` spec used by
    /// `make_unwrap_greenkey` (`warmstate.py:535-553`) and
    /// `make_jitcell_subclass` (`warmstate.py:557-654`) to dispatch
    /// `equal_whatever` / `hash_whatever`.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's `JitDriverVar.tp` carries only
    /// the IR-level [`Type`] (i/r/f/v) without the upstream STR/UNICODE
    /// distinction (`GreenType::Str`, `GreenType::Unicode`). The
    /// resulting spec collapses string / unicode greens to plain
    /// `GreenType::Ref` and falls back to pointer identity equality.
    /// Drivers that need typed string greens must override via the
    /// `EntryPoint::schema` registration path (`register_entry_point_typed`
    /// at jitdriver.rs:2047) until the IR carries the subtype natively.
    pub fn green_args_spec(&self) -> Vec<GreenType> {
        self.vars
            .iter()
            .filter(|v| v.kind == VarKind::Green)
            .map(|v| GreenType::from(v.tp))
            .collect()
    }

    /// Convert `self.red_args_types` (history-kind char list per
    /// `warmspot.py:664`) into the IR-level `Type` slice consumed by
    /// `compile_tmp_callback` (`compile.py:1114-1124` red-arg loop).
    ///
    /// Pyre stores `red_args_types: Vec<char>` to mirror upstream's
    /// `'i' / 'r' / 'f'` representation; this helper bridges to the IR
    /// `Type` enum without the caller having to re-implement the
    /// mapping. The S2.4 cutover will drop the explicit `red_arg_types`
    /// argument from `compile_tmp_callback` and read from the jd
    /// directly via this helper.
    ///
    /// Panics if any entry is not one of 'i', 'r', 'f', 'v'.
    pub fn red_arg_types_as_ir_types(&self) -> Vec<Type> {
        self.red_args_types
            .iter()
            .map(|c| match c {
                'i' => Type::Int,
                'r' => Type::Ref,
                'f' => Type::Float,
                'v' => Type::Void,
                other => panic!(
                    "JitDriverStaticData::red_arg_types_as_ir_types: \
                     unknown history kind char {:?} in red_args_types — \
                     warmspot.py:664 emits only 'i' / 'r' / 'f' / 'v'",
                    other
                ),
            })
            .collect()
    }

    /// `warmstate.py:535-553` `make_unwrap_greenkey().unwrap_greenkey(greenkey)`.
    ///
    /// Upstream's `unwrap_greenkey` walks `_green_args_spec` and calls
    /// `unwrap(TYPE, greenbox)` on each `Const` greenbox, returning a
    /// tuple of unwrapped values. Pyre's caller has already extracted
    /// the `i64` payload from the `Const` opref before calling this
    /// helper (the upstream `Const`-isinstance check lives in
    /// `MIFrame::verify_green_args` at `frame.rs` per `pyjitpl.py:1530-1535`),
    /// so the input here is the already-unwrapped value list.
    ///
    /// Panics if `values.len() != self.num_green_args()` — upstream's
    /// spec-driven loop has an implicit assertion via the
    /// `unrolling_iterable` length match.
    pub fn unwrap_greenkey(&self, values: &[i64]) -> GreenKey {
        let spec = self.green_args_spec();
        assert_eq!(
            values.len(),
            spec.len(),
            "unwrap_greenkey: expected {} green values per _green_args_spec, got {}",
            spec.len(),
            values.len(),
        );
        GreenKey::with_types(values.to_vec(), spec)
    }

    /// Get the virtualizable variable, if any.
    pub fn virtualizable(&self) -> Option<&JitDriverVar> {
        let name = self.virtualizable.as_deref()?;
        self.vars.iter().find(|var| var.name == name)
    }

    /// rewrite.py:684 `jd.index_of_virtualizable` parity.
    ///
    /// Returns the index inside the red-arg list / original
    /// CALL_ASSEMBLER arglist, not the absolute index in `vars`.
    pub fn virtualizable_arg_index(&self) -> Option<usize> {
        let name = self.virtualizable.as_deref()?;
        self.reds().iter().position(|var| var.name == name)
    }
}

/// Trait implemented by declarative `#[jit_driver]` marker types.
///
/// This provides a stable seam between proc-macro-generated driver metadata
/// and the runtime `JitDriver` orchestration layer.
pub trait DeclarativeJitDriver {
    const GREENS: &'static [&'static str];
    const REDS: &'static [&'static str];
    const NUM_VARS: usize;
    const NUM_GREENS: usize;
    const NUM_REDS: usize;
    const VIRTUALIZABLE: Option<&'static str>;

    fn descriptor(
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<JitDriverStaticData, &'static str>;

    fn green_key(values: &[i64]) -> Result<GreenKey, &'static str>;
}

/// A named entry point registered with a [`JitDriver`].
///
/// Multiple functions can share the same JitDriver and compiled loops
/// by registering additional entry points with distinct green key schemas.
/// This models the warmspot multi-entry pattern where several
/// `jit_merge_point` calls in different functions share one driver.
#[derive(Debug, Clone)]
pub struct EntryPoint {
    /// Human-readable name of this entry point (e.g., function name).
    pub name: String,
    /// warmstate.py:564 `_green_args_spec` — per-entry TYPE used by
    /// `JitCell.comparekey` / `get_uhash`. `GreenType` retains the
    /// STR / UNICODE distinction that IR `Type` collapses into `Ref`.
    pub schema: Vec<GreenType>,
}

/// High-level JIT driver that automates the tracing lifecycle.
///
/// Wraps [`MetaInterp`] and manages symbolic state, replacing ~80 lines
/// of boilerplate in each interpreter's mainloop with two method calls:
///
/// - [`merge_point`](JitDriver::merge_point) — replaces the tracing hook
///   (`jit_merge_point` in RPython)
/// - [`back_edge`](JitDriver::back_edge) — replaces back-edge handling
///   (`can_enter_jit` in RPython)
///
/// # Type Parameter
///
/// `S` implements [`JitState`] and defines how the interpreter's live
/// state maps to/from the JIT's representation.
/// PyPy warmspot.py JitDriver equivalent.
///
/// Manages the tracing/compilation lifecycle for a single jitdriver.
/// PyPy names: JitDriver(greens=[...], reds=[...], is_recursive=True).
pub struct JitDriver<S: JitState> {
    meta: MetaInterp<S::Meta>,
    sym: Option<S::Sym>,
    /// pyjitpl.py:2978-2983 reached_loop_header direct compile_trace parity.
    /// The frontend decides whether to call compile_trace(); the driver only
    /// consumes the successful result and switches back to compiled code.
    compile_trace_success: bool,
    descriptor: Option<JitDriverStaticData>,
    /// resume.py:1042: result of rebuild_from_resumedata for bridge tracing.
    /// RPython stores this in MIFrame registers; pyre stores it here
    /// for the caller to initialize PyreSym slot-to-OpRef mapping.
    pub resume_data_result: Option<crate::jit_state::ResumeDataResult>,
    /// RPython pyjitpl.py:3101 parity: true when the current bridge
    /// traces from an exception guard (GUARD_EXCEPTION / GUARD_NO_EXCEPTION).
    /// The caller should emit SAVE_EXC_CLASS + SAVE_EXCEPTION at trace start.
    pub last_bridge_is_exception_guard: bool,
    /// Additional entry points sharing this driver's compiled loops.
    entry_points: Vec<EntryPoint>,
    /// PyPy JitDriver(is_recursive=True): enables max_unroll_recursion
    /// for recursive portal calls (pyjitpl.py _opimpl_recursive_call).
    is_recursive: bool,
    /// Shared quasi-immutable notifier for periodic loop invalidation.
    /// RPython compile.py:205: loop.quasi_immutable_deps registration.
    /// All compiled loops register their invalidation flag here.
    /// A background thread periodically calls invalidate() to force
    /// GUARD_NOT_INVALIDATED exits in compiled code.
    epoch_qmut: std::sync::Arc<std::sync::Mutex<crate::quasiimmut::QuasiImmut>>,
    /// Handle for the background invalidation thread.
    _invalidation_thread: Option<std::thread::JoinHandle<()>>,
    /// RPython metainterp_sd.jitcodes parity: factory callback that
    /// produces JitCode for a given (program, pc, op) triple.
    /// Used by BlackholeInterpreter to resume from guard failure points.
    /// Registered by #[jit_interp] macro at startup.
    jitcode_factory:
        Option<Box<dyn Fn(&S::Env, usize, u8) -> Option<crate::jitcode::JitCode> + Send>>,
    /// resume.py:1367 — CPU allocation backend for virtual materialization
    /// during blackhole resume. Registered by pyre/aheui at startup.
    blackhole_allocator: Option<Box<dyn crate::resume::BlackholeAllocator + Send>>,
    /// warmspot.py:961 handle_jitexception parity: portal runner callback.
    /// Called when ContinueRunningNormally is raised at a recursive portal
    /// level during blackhole execution. Re-enters the portal function
    /// with green/red args and returns the result.
    portal_runner: Option<
        Box<
            dyn Fn(
                    &crate::jitexc::JitException,
                )
                    -> Result<(crate::blackhole::BhReturnType, i64), crate::jitexc::JitException>
                + Send,
        >,
    >,
}

impl<S: JitState> JitDriver<S> {
    /// Create a new JitDriver with the given hot-counting threshold.
    pub fn new(threshold: u32) -> Self {
        let mut meta = MetaInterp::new(threshold);
        if let Some(info) = S::__build_virtualizable_info() {
            meta.set_virtualizable_info(info);
        }
        let epoch_qmut =
            std::sync::Arc::new(std::sync::Mutex::new(crate::quasiimmut::QuasiImmut::new()));
        // Background thread: periodically invalidate all registered loops.
        // RPython uses GC/signal-triggered invalidation; we use a timer as
        // a portable equivalent. Period matches PyPy's checkinterval (~10ms).
        #[cfg(not(target_arch = "wasm32"))]
        let invalidation_thread = {
            let qmut = epoch_qmut.clone();
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    if let Ok(mut qmut) = qmut.lock() {
                        if qmut.has_watchers() {
                            qmut.invalidate();
                        }
                    }
                }
            })
        };
        JitDriver {
            meta,
            sym: None,
            compile_trace_success: false,
            descriptor: None,
            resume_data_result: None,
            last_bridge_is_exception_guard: false,
            entry_points: Vec::new(),
            is_recursive: false,
            epoch_qmut,
            #[cfg(not(target_arch = "wasm32"))]
            _invalidation_thread: Some(invalidation_thread),
            #[cfg(target_arch = "wasm32")]
            _invalidation_thread: None,
            jitcode_factory: None,
            blackhole_allocator: None,
            portal_runner: None,
        }
    }

    /// Install the state-field JIT canonical liveness payload before
    /// any tracing path runs.  Mirrors RPython `warmspot.py:281-289`'s
    /// `make_jitcodes() → finish_setup(codewriter)` for the narrow
    /// `pyjitpl.py:2264 self.liveness_info = "".join(asm.all_liveness)`
    /// slice — full `finish_setup` requires `CodeWriter` /
    /// `CallControl` construction (Task #89 orth-9 plan step 4) which
    /// is not yet wired.
    ///
    /// Caller responsibility: build a fresh
    /// `majit_translate::jit_codewriter::assembler::Assembler`,
    /// register the canonical
    /// `__JitMeta::canonical_liveness_slots()` triple via
    /// `Assembler::_encode_liveness`, then pass `&asm` here.  The
    /// single-owner `Arc::get_mut` invariant (`MetaInterp::finish_setup`)
    /// applies — call before the first trace.
    pub fn install_canonical_liveness(
        &mut self,
        asm: &majit_translate::jit_codewriter::assembler::Assembler,
    ) {
        self.meta.install_canonical_liveness(asm);
    }

    /// Register a portal runner callback for blackhole ContinueRunningNormally.
    ///
    /// warmspot.py:1039 handle_jitexception_from_blackhole parity:
    /// called when ContinueRunningNormally is raised at a recursive portal
    /// level. The callback re-enters the portal function and returns the result.
    pub fn register_portal_runner(
        &mut self,
        runner: impl Fn(
            &crate::jitexc::JitException,
        )
            -> Result<(crate::blackhole::BhReturnType, i64), crate::jitexc::JitException>
        + Send
        + 'static,
    ) {
        self.portal_runner = Some(Box::new(runner));
    }

    /// Register a JitCode factory callback for blackhole resume.
    ///
    /// RPython: `metainterp_sd.jitcodes` stores pre-compiled JitCodes.
    /// In majit, the `#[jit_interp]` macro generates JitCode on-demand
    /// from the interpreter bytecode. This callback provides that.
    pub fn register_jitcode_factory(
        &mut self,
        factory: impl Fn(&S::Env, usize, u8) -> Option<crate::jitcode::JitCode> + Send + 'static,
    ) {
        self.jitcode_factory = Some(Box::new(factory));
    }

    /// Register a BlackholeAllocator for virtual materialization during
    /// guard failure blackhole resume. Without this, virtual objects
    /// and raw buffers are allocated as null/zero.
    pub fn register_blackhole_allocator(
        &mut self,
        allocator: impl crate::resume::BlackholeAllocator + Send + 'static,
    ) {
        self.blackhole_allocator = Some(Box::new(allocator));
    }

    pub fn with_descriptor(threshold: u32, descriptor: JitDriverStaticData) -> Self {
        let mut driver = Self::new(threshold);
        // warmstate.py:564 `_green_args_spec` carries lltype TYPE per
        // green arg. `JitDriverVar.tp` is the IR `Type`; convert through
        // `From<Type>` which maps `Ref` to `GreenType::Ref`. Callers that
        // need the stricter `STR` / `UNICODE` distinction use
        // `register_entry_point` with an explicit schema.
        let greens: Vec<GreenType> = descriptor.greens().iter().map(|v| v.tp.into()).collect();
        driver.descriptor = Some(descriptor);
        driver.entry_points.push(EntryPoint {
            name: "primary".to_string(),
            schema: greens,
        });
        driver
    }

    /// Get compiled loop metadata for the given green key.
    pub fn get_compiled_meta(&self, green_key: u64) -> Option<&S::Meta> {
        self.meta.get_compiled_meta(green_key)
    }

    /// PyPy warmstate.py get_assembler_token(greenkey).
    /// Returns the JitCellToken for the compiled loop at this green key.
    pub fn get_loop_token(&self, green_key: u64) -> Option<&majit_backend::JitCellToken> {
        self.meta.get_loop_token(green_key)
    }

    /// Get the compiled loop's num_inputs (after preamble patching).
    pub fn get_compiled_num_inputs(&self, green_key: u64) -> Option<usize> {
        self.meta.get_compiled_num_inputs(green_key)
    }

    /// RPython resume_in_blackhole parity: resume execution from the guard
    /// failure point using the blackhole interpreter.
    pub fn blackhole_guard_failure(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: crate::blackhole::ExceptionState,
    ) -> Option<(
        crate::blackhole::BlackholeResult,
        crate::blackhole::ExceptionState,
    )> {
        self.meta
            .blackhole_guard_failure(green_key, trace_id, fail_index, fail_values, exception)
    }

    /// Like `blackhole_guard_failure` but with a CallAssembler callback.
    pub fn blackhole_guard_failure_ca(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        fail_values: &[i64],
        exception: crate::blackhole::ExceptionState,
        call_assembler_fn: Option<&crate::blackhole::CallAssemblerFn>,
    ) -> Option<(
        crate::blackhole::BlackholeResult,
        crate::blackhole::ExceptionState,
    )> {
        self.meta.blackhole_guard_failure_ca(
            green_key,
            trace_id,
            fail_index,
            fail_values,
            exception,
            call_assembler_fn,
        )
    }

    /// RPython resume_in_blackhole parity: resume from guard failure using
    /// the jitcode-based BlackholeInterpreter.
    ///
    /// Uses the registered jitcode_factory to produce a JitCode for the
    /// guard failure's resume_pc, then runs the BlackholeInterpreter from
    /// that point to complete the iteration.
    pub fn blackhole_resume_jitcode(
        &self,
        env: &S::Env,
        resume_pc: usize,
        resume_op: u8,
        fail_values: &[i64],
        inputarg_count: usize,
    ) -> Option<crate::blackhole::BlackholeInterpreter> {
        let factory = self.jitcode_factory.as_ref()?;
        let jitcode = factory(env, resume_pc, resume_op)?;

        let mut bh = crate::blackhole::BlackholeInterpreter::new();
        bh.setposition(std::sync::Arc::new(jitcode), 0);

        // Set inputarg register values from fail_values
        for (i, &val) in fail_values.iter().take(inputarg_count).enumerate() {
            bh.setarg_i(i, val);
        }

        if crate::majit_log_enabled() {
            eprintln!(
                "[bh-jitcode] resume at pc={} op={} inputargs={}",
                resume_pc, resume_op, inputarg_count
            );
        }

        // Run the blackhole interpreter
        bh.run();

        Some(bh)
    }

    /// resume.py:1312 blackhole_from_resumedata parity: get the
    /// recovery slot types for building typed Value array from raw fail_values.
    pub fn get_recovery_slot_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        self.meta
            .get_recovery_slot_types(green_key, trace_id, fail_index)
    }

    /// compile.py:853 `ResumeGuardDescr` storage handle forwarder.
    pub fn get_resume_storage(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<std::sync::Arc<crate::resume::ResumeStorage>> {
        self.meta
            .get_resume_storage(green_key, trace_id, fail_index)
    }

    pub fn get_exit_types(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Vec<Type>> {
        self.meta.get_exit_types(green_key, trace_id, fail_index)
    }

    /// compile.py:710 recovery_layout header_pc parity: get the merge point
    /// PC for blackhole resume from a guard exit.
    pub fn get_merge_point_pc(
        &self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<u64> {
        self.meta
            .get_merge_point_pc(green_key, trace_id, fail_index)
    }

    /// Attach a GC allocator to the active backend.
    pub fn set_gc_allocator(&mut self, gc: Box<dyn GcAllocator>) {
        self.meta.backend_mut().set_gc_allocator(gc);
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    /// Frontend supplies the byte offset of the type pointer field, mirroring
    /// RPython's `symbolic.get_field_token(rclass.OBJECT, 'typeptr', ...)`.
    /// `None` matches RPython's gcremovetypeptr behavior (no vtable write).
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.meta.backend_mut().set_vtable_offset(offset);
    }

    /// PyPy JitDriver(is_recursive=True).
    /// Enables max_unroll_recursion for recursive portal calls.
    pub fn set_is_recursive(&mut self, value: bool) {
        self.is_recursive = value;
    }

    /// PyPy warmspot.py set_param_max_unroll_recursion().
    pub fn set_max_unroll_recursion(&mut self, value: usize) {
        self.meta.set_max_unroll_recursion(value);
    }

    /// RPython-compatible alias used by warmstate/compile wiring.
    pub fn set_param_max_unroll_recursion(&mut self, value: usize) {
        self.set_max_unroll_recursion(value);
    }

    /// Whether this driver was declared recursive.
    pub fn is_recursive(&self) -> bool {
        self.is_recursive
    }

    pub fn with_declarative<D: DeclarativeJitDriver>(
        threshold: u32,
        green_types: &[Type],
        red_types: &[Type],
    ) -> Result<Self, &'static str> {
        let descriptor = D::descriptor(green_types, red_types)?;
        Ok(Self::with_descriptor(threshold, descriptor))
    }

    /// Whether the driver is currently tracing.
    #[inline]
    pub fn is_tracing(&self) -> bool {
        self.meta.is_tracing()
    }

    /// RPython JC_TRACING parity: true only when tracing this specific
    /// key. `target_raw` is the structured `(code_ptr, pc)` greenkey
    /// matching pyjitpl.py:1396-1401's element-wise comparison.
    #[inline]
    pub fn is_tracing_key(&self, target_raw: (usize, usize)) -> bool {
        self.meta.is_tracing_key(target_raw)
    }

    /// Whether the driver is currently tracing a bridge.
    #[inline]
    pub fn is_bridge_tracing(&self) -> bool {
        self.meta.bridge_info().is_some()
    }

    /// The green key of the active trace, if any.
    pub fn current_trace_green_key(&mut self) -> Option<u64> {
        self.meta.trace_ctx().map(|ctx| ctx.green_key())
    }

    /// ResumeFromInterpDescr parity: data needed to compile an entry bridge
    /// from the currently active trace.
    pub fn compile_trace_entry_data(&mut self) -> Option<(u64, S::Meta)> {
        let original_green_key = self.meta.trace_ctx().map(|ctx| ctx.root_green_key())?;
        let trace_meta = self.meta.trace_meta()?.clone();
        Some((original_green_key, trace_meta))
    }

    /// RPython rlib.jit.current_trace_length().
    /// Returns the number of ops in the active trace, or -1 if not tracing.
    pub fn current_trace_length(&mut self) -> i64 {
        match self.meta.trace_ctx() {
            Some(ctx) => ctx.num_ops() as i64,
            None => -1,
        }
    }

    /// Abort the active trace and clear driver-owned symbolic state.
    pub fn abort_current_trace(&mut self, permanent: bool) {
        if self.meta.is_tracing() {
            self.meta.abort_trace(permanent);
            self.sym = None;
            self.meta.clear_trace_session();
            self.compile_trace_success = false;
        }
    }

    /// reached_loop_header() called compile_trace() successfully and the
    /// driver should switch to compiled code after the merge-point closure
    /// returns.
    ///
    /// PRE-EXISTING-ADAPTATION: substitutes for `ContinueRunningNormally`
    /// stack unwind (`pyjitpl.py:3095` `raise_if_successful`). RPython
    /// raises across the interpreter to break out of trace recording; the
    /// Rust dispatch loop cannot raise across the FFI boundary into
    /// `merge_point_keyed`, so a one-shot `bool` consumed by
    /// `take_compile_trace_success` substitutes. Same semantics: stop
    /// tracing immediately, switch into compiled code on next iteration.
    pub fn note_compile_trace_success(&mut self) {
        if crate::majit_log_enabled() {
            eprintln!("[jit][compile-trace] note success");
        }
        self.compile_trace_success = true;
    }

    pub fn compile_trace_success_pending(&self) -> bool {
        self.compile_trace_success
    }

    fn take_compile_trace_success(&mut self) -> bool {
        std::mem::take(&mut self.compile_trace_success)
    }

    /// Tracing hook — call at the top of the dispatch loop.
    ///
    /// If tracing is active, calls `trace_fn` with the active [`TraceCtx`]
    /// and symbolic state, then handles the result automatically:
    ///
    /// - `CloseLoop` → validates depths, collects jump args, compiles
    /// - `Finish` → compiles a terminal trace ending in `FINISH`
    /// - `Abort` → aborts trace (may retry later)
    /// - `AbortPermanent` → aborts trace permanently
    /// - `Continue` → no action
    ///
    /// # Example
    ///
    /// ```ignore
    /// driver.merge_point(|ctx, sym| {
    ///     trace_instruction(ctx, sym, program, pc, &state)
    /// });
    /// ```
    #[inline]
    pub fn merge_point<F>(&mut self, trace_fn: F)
    where
        F: FnOnce(&mut TraceCtx, &mut S::Sym) -> TraceAction,
    {
        if !self.meta.is_tracing() {
            return;
        }
        if self.sym.is_none() || self.meta.trace_meta().is_none() {
            if crate::majit_log_enabled() {
                eprintln!("[mp] abort:sym_none");
            }
            self.meta.abort_trace(false);
            self.sym = None;
            self.meta.clear_trace_session();
            return;
        }

        // Phase 1: split-borrow self into meta (for ctx) and sym, run closure
        let mut action = {
            let Some(ctx) = self.meta.trace_ctx() else {
                if crate::majit_log_enabled() {
                    eprintln!("[mp] abort:ctx_none");
                }
                self.sym = None;
                self.meta.clear_trace_session();
                return;
            };
            let Some(sym) = self.sym.as_mut() else {
                if crate::majit_log_enabled() {
                    eprintln!("[mp] abort:sym_none2");
                }
                self.meta.abort_trace(false);
                self.meta.clear_trace_session();
                return;
            };
            trace_fn(ctx, sym)
        }; // ctx and sym references dropped here

        // pyjitpl.py:1618 force_finish_trace segmenting check.
        //
        // pyjitpl.py:1622 _create_segmented_trace_and_blackhole:
        //   1. generate_guard(GUARD_ALWAYS_FAILS)
        //   2. unreachable FINISH(exception_descr)
        //   3. compile_simple_loop(patch_jumpop_at_end=False)  ← inserts Label
        //   4. SwitchToBlackhole(ABORT_SEGMENTED_TRACE)
        // pyjitpl.py:1618 force_finish_trace segmenting fallback.
        // The proc-macro path handles this inside the closure (where __pc
        // is captured). For non-macro consumers whose trace_fn doesn't
        // emit SegmentedLoop, this fallback uses ctx.last_traced_pc
        // (recorded by proc-macro or pyre trace_fn) as the guard-point pc.
        if matches!(action, TraceAction::Continue) && self.meta.force_finish_trace {
            let should_segment = self
                .meta
                .trace_ctx()
                .map(|ctx| ctx.num_ops() > ctx.trace_limit() * 4 / 5)
                .unwrap_or(false);
            if should_segment {
                // RPython `pyjitpl.py:2558-2602 MetaInterp.generate_guard`
                // pairs every guard recording with `capture_resumedata`,
                // and segmenting itself is independent of whether the
                // sym-side framestack-lift override is wired:
                // `_create_segmented_trace_and_blackhole`
                // (`pyjitpl.py:1622`) always emits
                // `generate_guard(GUARD_ALWAYS_FAILS) + Finish +
                // SwitchToBlackhole`.  When
                // `JitState::populate_frame_for_guard` is overridden
                // (state-field-JIT macro consumers), pair the guard with
                // the bridge snapshot.
                //
                // PRE-EXISTING-ADAPTATION: when the default no-op
                // returns `None`, this low-level driver has no MIFrame
                // to walk — it only knows the active jump boxes from
                // `S::collect_jump_args`.  We feed those into
                // `capture_snapshot_for_last_guard`, which records a
                // single placeholder frame in lieu of the RPython
                // `framestack` walk (`pyjitpl.py:2586
                // capture_resumedata`).  Bounded scope: this branch is
                // only reached on the segmented-loop force path of the
                // low-level driver and dissolves once Task #89 lifts
                // `S::Sym` into `MIFrame::populate_for_guard` so every
                // path captures uniformly via the framestack walk.
                let pre_snapshot = self
                    .sym
                    .as_ref()
                    .and_then(|sym| S::populate_frame_for_guard(sym, &mut self.meta.framestack));
                let mut current_live = self
                    .sym
                    .as_ref()
                    .map(|sym| S::collect_jump_args(sym))
                    .unwrap_or_default();
                if let Some(ctx) = self.meta.trace_ctx() {
                    // pyjitpl.py:2594: use last_traced_pc
                    // (= frame.pc at the guard point), not header_pc.
                    let pc_opref = ctx.const_int(ctx.last_traced_pc as i64);
                    current_live.push(pc_opref);
                    // history.py:802 strict-types parity: every active
                    // OpRef must resolve to a known Box.type; a miss is
                    // a tracer bookkeeping bug, not a recoverable case.
                    let live_types: Vec<majit_ir::Type> = current_live
                        .iter()
                        .map(|opref| {
                            ctx.get_opref_type(*opref)
                                .expect("force_finish_trace: active OpRef missing Box.type")
                        })
                        .collect();
                    let guard_opref =
                        ctx.record_guard_typed(majit_ir::OpCode::GuardAlwaysFails, &[], live_types);
                    if let Some(snapshot) = pre_snapshot {
                        let snapshot_id = ctx.capture_resumedata(snapshot);
                        ctx.set_last_guard_resume_position(snapshot_id);
                    } else {
                        // PRE-EXISTING-ADAPTATION: see header comment.
                        // No framestack-lift override available, so
                        // capture the active low-level boxes via the
                        // segmented-driver placeholder helper instead
                        // of inventing a fail_args-only fallback.
                        ctx.capture_snapshot_for_last_guard(&current_live);
                        ctx.set_fail_args(guard_opref, &current_live);
                    }
                    let dummy = ctx.const_int(0);
                    ctx.record_finish(dummy, majit_ir::Type::Int);
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[jit] force_finish_trace: segmenting at {} ops (limit {}) pc={}",
                            ctx.num_ops(),
                            ctx.trace_limit(),
                            ctx.last_traced_pc
                        );
                    }
                }
                action = TraceAction::SegmentedLoop;
            }
        }

        // Phase 2: handle trace result with full access to self
        match action {
            TraceAction::Continue => {}
            TraceAction::CompileTrace => {
                self.meta.abort_trace(false);
                self.sym = None;
                self.meta.clear_trace_session();
                self.compile_trace_success = true;
                return;
            }
            TraceAction::CloseLoop => {
                // pyjitpl.py:2979-3036 reached_loop_header parity.
                // Path 1: bridge — only if has_compiled_targets (line 2982).
                if let Some(bridge) = self.meta.bridge_info() {
                    let bridge_key = bridge.green_key;
                    let bridge_trace_id = bridge.trace_id;
                    let bridge_fail_index = bridge.fail_index;
                    let has_targets = self.meta.has_compiled_targets(bridge_key);
                    if has_targets {
                        if crate::majit_log_enabled() {
                            eprintln!(
                                "[bridge] CloseLoop -> close_bridge key={} trace={} fail={}",
                                bridge_key, bridge_trace_id, bridge_fail_index
                            );
                        }
                        if let Some(sym) = self.sym.as_ref() {
                            let finish_args = S::collect_jump_args(sym);
                            let result = self.meta.close_bridge(
                                bridge_key,
                                bridge_trace_id,
                                bridge_fail_index,
                                &finish_args,
                            );
                            match result {
                                crate::pyjitpl::BridgeCompileResult::Compiled => {
                                    self.sym = None;
                                    self.meta.clear_trace_session();
                                    // pyjitpl.py:3095-3099 raise_if_successful():
                                    // successful bridge closure terminates tracing.
                                    self.meta.abort_trace(false);
                                    return;
                                }
                                // pyjitpl.py:2993-3007: after retrace_needed(),
                                // partial_trace is set on MetaInterp. Fall through
                                // to compile_loop → compile_retrace in same call.
                                crate::pyjitpl::BridgeCompileResult::RetraceNeeded => {
                                    self.meta.take_bridge_info();
                                    // Fall through — do NOT return.
                                }
                                crate::pyjitpl::BridgeCompileResult::Failed => {
                                    self.meta.abort_trace(false);
                                    self.sym = None;
                                    self.meta.clear_trace_session();
                                    return;
                                }
                            }
                        } else {
                            self.meta.abort_trace(false);
                            self.sym = None;
                            self.meta.clear_trace_session();
                            return;
                        }
                    } else {
                        // No targets: consume bridge_info, fall through to
                        // compile_loop (pyjitpl.py:3014-3017).
                        self.meta.take_bridge_info();
                    }
                }
                let Some(trace_meta) = self.meta.trace_meta().cloned() else {
                    self.meta.abort_trace(false);
                    self.sym = None;
                    self.meta.clear_trace_session();
                    return;
                };
                let Some(sym) = self.sym.as_ref() else {
                    self.meta.abort_trace(false);
                    self.meta.clear_trace_session();
                    return;
                };
                if S::validate_close(sym, &trace_meta) {
                    let jump_args = S::collect_jump_args(sym);
                    // pyjitpl.py:2993-3036 parity: compile_loop borrows the
                    // frontend meta the same way `self.history` is a shared
                    // mutable object on the RPython MetaInterp — the caller
                    // does not drain it. Read via `trace_meta().cloned()` so
                    // the `active_trace_session` envelope stays live for the
                    // duration of the compile. `compile_loop` itself owns
                    // the post-exit invariant that session lifetime tracks
                    // `self.tracing` (see `compile_loop` wrapper).
                    let provisional_meta = self.meta.trace_meta().cloned().unwrap();
                    // When cross-loop cut redirects to a different header_pc,
                    // rebuild meta from merge point to get the inner header's
                    // frame layout (vsd, slot_types).
                    let meta = {
                        if let Some((cut_pc, ref cut_types)) = self.meta.cross_loop_cut_info() {
                            S::build_meta_from_merge_point(&provisional_meta, cut_pc, cut_types)
                        } else {
                            provisional_meta
                        }
                    };
                    // pyjitpl.py:3015-3030 compile_loop + raise_if_successful.
                    let outcome = self.meta.compile_loop(&jump_args, meta);
                    match outcome {
                        crate::CompileOutcome::Compiled { .. } => {
                            // pyjitpl.py:3095 raise_if_successful → done.
                            // compile_loop already drained tracing+session.
                        }
                        crate::CompileOutcome::Cancelled => {
                            // pyjitpl.py:3018-3032 parity: cancel_count is
                            // incremented inside compile_loop and tracing
                            // continues unless `cancelled_too_many_times`
                            // escalates to Aborted. The wrapper restored
                            // session↔tracing alignment so no extra cleanup
                            // is required here — either the early-Cancelled
                            // path kept both live (tracing continues) or a
                            // late-Cancelled path drained both (next bound
                            // reaches begins a fresh session).
                        }
                        crate::CompileOutcome::Aborted => {
                            // pyjitpl.py:3028 SwitchToBlackhole(ABORT_BAD_LOOP)
                            self.meta.abort_trace(false);
                        }
                    }
                } else {
                    if crate::majit_log_enabled() {
                        eprintln!("[mp] abort:validate_close");
                    }
                    self.meta.abort_trace(false);
                    self.meta.clear_trace_session();
                }
                self.sym = None;
            }
            TraceAction::CloseLoopWithArgs {
                jump_args,
                loop_header_pc,
            } => {
                // pyjitpl.py:2979-2990 reached_loop_header parity.
                if let Some(bridge) = self.meta.bridge_info() {
                    let bridge_key = bridge.green_key;
                    let bridge_trace_id = bridge.trace_id;
                    let bridge_fail_index = bridge.fail_index;
                    let bridge_code_ptr = bridge.code_ptr;
                    // pyjitpl.py:2982: ptoken = self.get_procedure_token(greenboxes)
                    // Use the TARGET loop header's green key, not the bridge origin.
                    let target_key = loop_header_pc
                        .map(|pc| crate::green_key_from_code_ptr(bridge_code_ptr, pc))
                        .unwrap_or(bridge_key);
                    let has_targets = self.meta.has_compiled_targets(target_key);
                    if has_targets {
                        let result = self.meta.close_bridge(
                            target_key,
                            bridge_trace_id,
                            bridge_fail_index,
                            &jump_args,
                        );
                        match result {
                            crate::pyjitpl::BridgeCompileResult::Compiled => {
                                self.sym = None;
                                self.meta.clear_trace_session();
                                // pyjitpl.py:3095-3099 raise_if_successful():
                                // successful bridge closure terminates tracing.
                                self.meta.abort_trace(false);
                                return;
                            }
                            // pyjitpl.py:2993-3007: after retrace_needed(),
                            // partial_trace is set on MetaInterp. Fall through
                            // to compile_loop → compile_retrace in same call.
                            crate::pyjitpl::BridgeCompileResult::RetraceNeeded => {
                                self.meta.take_bridge_info();
                                // Fall through — do NOT return.
                            }
                            crate::pyjitpl::BridgeCompileResult::Failed => {
                                self.meta.abort_trace(false);
                                self.sym = None;
                                self.meta.clear_trace_session();
                                return;
                            }
                        }
                    } else {
                        // No targets: consume bridge_info, fall through.
                        self.meta.take_bridge_info();
                    }
                }
                // pyjitpl.py:2983: compile_trace already compiled a bridge
                // to the existing loop. Don't call compile_loop again.
                if self.compile_trace_success_pending() {
                    self.sym = None;
                    self.meta.clear_trace_session();
                    return;
                }
                let Some(trace_meta) = self.meta.trace_meta().cloned() else {
                    self.meta.abort_trace(false);
                    self.sym = None;
                    self.meta.clear_trace_session();
                    return;
                };
                let Some(sym) = self.sym.as_ref() else {
                    self.meta.abort_trace(false);
                    self.meta.clear_trace_session();
                    return;
                };
                if S::validate_close_with_jump_args(sym, &trace_meta, &jump_args) {
                    // pyjitpl.py:2993-3036 parity: keep `active_trace_session`
                    // alive across compile_loop — the session envelope mirrors
                    // RPython's `self.history` which compile_loop borrows
                    // without consuming.
                    let provisional_meta = self.meta.trace_meta().cloned().unwrap();
                    let meta = {
                        if let Some((cut_pc, ref cut_types)) = self.meta.cross_loop_cut_info() {
                            S::build_meta_from_merge_point(&provisional_meta, cut_pc, cut_types)
                        } else {
                            provisional_meta
                        }
                    };
                    // pyjitpl.py:3015-3030 compile_loop + raise_if_successful.
                    let outcome = self.meta.compile_loop(&jump_args, meta);
                    match outcome {
                        crate::CompileOutcome::Compiled { .. } => {
                            // pyjitpl.py:3095 raise_if_successful → done.
                        }
                        crate::CompileOutcome::Cancelled => {
                            // pyjitpl.py:3018-3032 parity: no explicit abort,
                            // tracing continues (or is already drained inside
                            // compile_loop for late-cancel cases). The wrapper
                            // maintains the session↔tracing invariant.
                        }
                        crate::CompileOutcome::Aborted => {
                            // pyjitpl.py:3028 SwitchToBlackhole
                            self.meta.abort_trace(false);
                        }
                    }
                } else {
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[mp] abort:validate_close_with_jump_args actual_len={}",
                            jump_args.len()
                        );
                    }
                    self.meta.abort_trace(false);
                    self.meta.clear_trace_session();
                }
                self.sym = None;
            }
            TraceAction::Finish {
                finish_args,
                finish_arg_types,
                exit_with_exception,
            } => {
                // pyjitpl.py:3198-3220 + 3238-3245 parity — upstream's
                // `compile_done_with_this_frame` and
                // `compile_exit_frame_with_exception` both call
                // `compile.compile_trace(self, self.resumekey, exits)` →
                // `compile.giveup()` on failure.  Both live on
                // `MetaInterp` in upstream; the shared helper
                // `compile_finish_from_active_session` dispatches bridge
                // vs root based on the session's `bridge_info` and picks
                // the FINISH descr per `exit_with_exception`.  The
                // TraceAction::Finish arm stays as a thin wrapper that
                // routes the pyre-dispatch-layer finish emission
                // (dispatch.rs:298) into the same helper, matching the
                // `except SwitchToBlackhole as stb: self.aborted_tracing`
                // shape at pyjitpl.py:2491.
                if let Err(stb) = self.meta.compile_finish_from_active_session(
                    &finish_args,
                    finish_arg_types,
                    exit_with_exception,
                ) {
                    self.meta.aborted_tracing(stb.reason);
                }
                self.sym = None;
            }
            TraceAction::SegmentedLoop => {
                // pyjitpl.py:1658-1663 _create_segmented_trace_and_blackhole:
                //   target_token = compile.compile_simple_loop(...)
                //   warmstate.attach_procedure_to_interp(greenkey, token)
                // + pyjitpl.py:1673 SwitchToBlackhole(ABORT_SEGMENTED_TRACE)
                let meta = self.meta.take_trace_meta().unwrap();
                if let Some(green_key) = self.meta.compile_simple_loop(meta) {
                    // PRE-EXISTING-ADAPTATION: see pyjitpl/mod.rs compile_loop
                    // site for the RPython citation and convergence path.
                    // Segmented loop mirrors `pyjitpl.py:1662-1663`.
                    let install_num = self.meta.warm_state.alloc_token_number();
                    let install_token = majit_backend::JitCellToken::new(install_num);
                    self.meta
                        .warm_state
                        .attach_procedure_to_interp(green_key, install_token);
                }
                // Blackhole transition: clear all driver tracing state.
                self.sym = None;
                self.meta.clear_trace_session();
            }
            TraceAction::Abort => {
                if crate::majit_log_enabled() && self.meta.bridge_info().is_some() {
                    eprintln!("[bridge] Abort during bridge tracing");
                }
                // pyjitpl.py:2788-2807 blackhole_if_trace_too_long runs the
                // segmenting / disable bookkeeping and returns the abort
                // reason.  Unwind once with abort_trace_live + aborted_tracing
                // matching RPython's `raise SwitchToBlackhole(ABORT_TOO_LONG)`
                // → `_interpret` handler → `aborted_tracing(reason)` shape.
                let reason = self
                    .meta
                    .blackhole_if_trace_too_long()
                    .unwrap_or(AbortReason::Generic);
                self.meta.abort_trace_live(false);
                self.meta.aborted_tracing(reason.as_int());
                self.sym = None;
                self.meta.clear_trace_session();
            }
            TraceAction::AbortPermanent => {
                if crate::majit_log_enabled() && self.meta.bridge_info().is_some() {
                    eprintln!("[bridge] AbortPermanent during bridge tracing");
                }
                self.meta.abort_trace(true);
                self.sym = None;
                self.meta.clear_trace_session();
            }
        }
    }

    /// pyjitpl.py:2788-2807 `blackhole_if_trace_too_long` + pyjitpl.py:2760
    /// `aborted_tracing(reason)` parity shape.
    ///
    /// Runs the upstream bookkeeping, unwinds the live tracing state via
    /// `abort_trace_live`, then fires `aborted_tracing(reason)` for the
    /// accounting event — all exactly once — and clears driver-owned
    /// per-session state (sym / trace_meta / bridge_info).  Returns
    /// `true` when an abort fired so callers can act on it.
    #[inline]
    pub fn blackhole_if_trace_too_long(&mut self) -> bool {
        let Some(reason) = self.meta.blackhole_if_trace_too_long() else {
            return false;
        };
        self.meta.abort_trace_live(false);
        self.meta.aborted_tracing(reason.as_int());
        self.clear_tracing_session_state();
        true
    }

    #[cold]
    #[inline(never)]
    fn clear_tracing_session_state(&mut self) {
        self.sym = None;
        self.meta.clear_trace_session();
    }

    /// Back-edge handler — call when a backward jump is detected.
    ///
    /// Handles hot counting, trace start, and compiled code execution.
    /// Returns `true` if compiled code ran successfully and the interpreter
    /// should jump to `target_pc` with restored state.
    ///
    /// `pre_run` is called before executing compiled code (e.g., to flush
    /// buffered I/O).
    ///
    /// # Example
    ///
    /// ```ignore
    /// if target <= pc {
    ///     if driver.back_edge(target, &mut state, program, || writer.flush()) {
    ///         pc = target;
    ///         continue;
    ///     }
    /// }
    /// ```
    /// Returns `Some(resume_pc)` when compiled code ran (FINISH or guard
    /// failure with blackhole resume). Returns `None` otherwise.
    pub fn back_edge(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<usize> {
        self.back_edge_internal(target_pc as u64, None, target_pc, state, env, pre_run)
    }

    pub fn back_edge_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<usize> {
        self.back_edge_internal(green_key, None, target_pc, state, env, pre_run)
    }

    #[cold]
    #[inline(never)]
    pub fn back_edge_structured(
        &mut self,
        green_key: GreenKey,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<usize> {
        let key = green_key.hash_u64();
        self.back_edge_internal(key, Some(green_key), target_pc, state, env, pre_run)
    }

    pub fn back_edge_or_run_compiled(
        &mut self,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(
            target_pc as u64,
            None,
            target_pc,
            state,
            env,
            pre_run,
        )
    }

    pub fn back_edge_or_run_compiled_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        self.back_edge_or_run_compiled_internal(green_key, None, target_pc, state, env, pre_run)
    }

    /// RPython-style jit_merge_point dispatch hook.
    ///
    /// Called from the outer interpreter dispatch loop on every iteration.
    /// This hook only feeds the active trace. Warmup counting, tracing start,
    /// and compiled-code entry belong to `can_enter_jit`/function-entry paths.
    pub fn jit_merge_point_keyed<F>(
        &mut self,
        _green_key: u64,
        target_pc: usize,
        _state: &mut S,
        _env: &S::Env,
        _pre_run: impl FnOnce(),
        trace_fn: F,
    ) -> Option<DetailedDriverRunOutcome>
    where
        F: FnOnce(&mut TraceCtx, &mut S::Sym) -> TraceAction,
    {
        // RPython JC_TRACING parity: in RPython, MetaInterp._interpret()
        // runs all bytecodes within the same portal call. In pyre,
        // eval_loop_jit calls jit_merge_point at each bytecode, so the
        // green_key changes per bytecode (includes pc). Use is_tracing()
        // (not is_tracing_key) to accept all bytecodes from the tracing
        // portal. Nested function calls are prevented by JIT_TRACING
        // flag in eval_loop_jit.
        if self.meta.is_tracing() {
            // pyjitpl.py:2594: record frame.pc for capture_resumedata.
            if let Some(ctx) = self.meta.trace_ctx() {
                ctx.last_traced_pc = target_pc;
            }
            self.merge_point(trace_fn);
            if self.take_compile_trace_success() {
                self.meta.abort_trace(false);
                self.sym = None;
                self.meta.clear_trace_session();
                return Some(DetailedDriverRunOutcome::Jump {
                    via_blackhole: false,
                });
            }
        }
        None
    }

    /// RPython-style can_enter_jit path.
    ///
    /// Called from a back-edge. Unlike `jit_merge_point_keyed`, this path
    /// never starts tracing and only runs already-compiled code.
    pub fn can_enter_jit_keyed(
        &mut self,
        green_key: u64,
        _target_pc: usize,
        state: &mut S,
        _env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        if self.meta.is_tracing() || !state.can_trace() {
            return None;
        }
        if self.meta.has_compiled_loop(green_key) {
            return Some(self.run_compiled_detailed_keyed(green_key, state, pre_run));
        }
        None
    }

    pub fn back_edge_declarative<D: DeclarativeJitDriver>(
        &mut self,
        green_values: &[i64],
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Result<Option<usize>, &'static str> {
        let green_key = D::green_key(green_values)?;
        Ok(self.back_edge_structured(green_key, target_pc, state, env, pre_run))
    }

    /// RPython warmstate.py:482-501 / compile.py:711 parity.
    ///
    /// Returns `Some(pc)` when compiled code ran:
    /// - FINISH → `Some(target_pc)` (loop completed, re-enter at header)
    /// - Guard failure → `Some(resume_pc)` (resume from failure point)
    /// Returns `None` when no compiled code exists or tracing started.
    fn back_edge_internal(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<usize> {
        if self.meta.is_tracing() || !state.can_trace() {
            return None;
        }

        if self.meta.has_compiled_loop(green_key) {
            let compiled_meta = self.meta.get_compiled_meta(green_key).unwrap().clone();
            let descriptor = self.driver_descriptor_for(state, &compiled_meta);
            if !state.is_compatible(&compiled_meta) {
                self.meta.invalidate_loop(green_key);
                return None;
            }
            if !self.sync_before(state, &compiled_meta, descriptor.as_ref()) {
                return None;
            }
            let live_values = state.extract_live_values(&compiled_meta);
            if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
                return None;
            }
            let Some(live_values) = self.extend_compiled_live_values(
                green_key,
                state,
                &compiled_meta,
                descriptor.as_ref(),
                live_values,
            ) else {
                return None;
            };

            pre_run();

            let Some(result) = self
                .meta
                .run_compiled_detailed_with_values(green_key, &live_values)
            else {
                return None;
            };

            if result.is_finish {
                let run_meta = result.meta.clone();
                state.restore_values(&run_meta, &result.typed_values);
                let run_descriptor = self.driver_descriptor_for(state, &run_meta);
                self.sync_after(state, &run_meta, run_descriptor.as_ref());
                return Some(target_pc);
            }

            // Normal loop back-edge JUMP, not a guard failure.
            if result.fail_index == u32::MAX {
                let run_meta = result.meta.clone();
                state.restore_values(&run_meta, &result.typed_values);
                let run_descriptor = self.driver_descriptor_for(state, &run_meta);
                self.sync_after(state, &run_meta, run_descriptor.as_ref());
                return Some(target_pc);
            }

            // compile.py:701-716 handle_fail
            let fail_index = result.fail_index;
            let trace_id = result.trace_id;
            let exit_layout = result.exit_layout.clone();
            let raw_values = result.values.clone();
            let descr_addr = result.descr_addr;
            drop(result);

            // must_compile tick for bridge threshold counting.
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit] handle_fail: fail_index={} trace_id={} descr_addr={:#x} values={}",
                    fail_index,
                    trace_id,
                    descr_addr,
                    raw_values.len()
                );
            }
            let guard_loop_key = if exit_layout.rd_loop_token != 0 {
                exit_layout.rd_loop_token
            } else {
                green_key
            };
            let (must_compile, owning_key) = self.meta.must_compile_with_values(
                guard_loop_key,
                trace_id,
                fail_index,
                &raw_values,
                descr_addr,
            );
            // compile.py:702-703: must_compile() and not stack_almost_full().
            let should_bridge =
                must_compile && !majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full();

            // Extract guard_resume_pc from fail_args (last Int value).
            // compile.py:800-809: use owning_key (rd_loop_token) for bridge attachment.
            let num_inputs = self.meta.compiled_num_inputs(owning_key);
            let guard_resume_pc = if raw_values.len() > num_inputs {
                raw_values[raw_values.len() - 1] as usize
            } else {
                target_pc
            };

            if should_bridge {
                // compile.py:704-709: _trace_and_compile_from_bridge
                let bridge_ok = self.start_bridge_tracing(
                    owning_key,
                    trace_id,
                    fail_index,
                    state,
                    env,
                    guard_resume_pc,
                    target_pc,
                );
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bridge] start_bridge_tracing key={} trace={} fail={} resume_pc={} ok={}",
                        green_key, trace_id, fail_index, guard_resume_pc, bridge_ok
                    );
                }
            }

            // compile.py:711 resume_in_blackhole
            // compile.py:853 `ResumeGuardDescr` storage — borrow
            // `rd_numb` / `rd_consts()` / `rd_virtuals` off the shared
            // Arc so blackhole resume observes the guard-owned pool
            // the GC walker updates (no per-call Vec clone).
            if let Some(storage) = exit_layout.storage.as_deref() {
                let rd_numb = storage.rd_numb.as_slice();
                let rd_consts_slice: &[Const] = storage.rd_consts();

                // Convert RdVirtualInfo → VirtualInfo for blackhole resume.
                let rd_virtuals_converted: Option<Vec<crate::resume::VirtualInfo>> = {
                    let count = raw_values.len() as i32;
                    Some(
                        storage
                            .rd_virtuals
                            .iter()
                            .map(|rd| {
                                crate::resume::rd_virtual_to_virtual_info(
                                    rd,
                                    rd_consts_slice,
                                    count,
                                )
                            })
                            .collect(),
                    )
                };
                let rd_virtuals_slice = rd_virtuals_converted.as_deref();

                // resume.py:1339: resolve jitcode from (jitcode_pos, pc).
                //
                // Only state-field-JIT users (e.g., aheui via
                // `register_jitcode_factory`) reach this path; pyre does not
                // register a factory so `as_ref()?` short-circuits to None.
                //
                // Multi-frame state-field-JIT chain reconstruction
                // (RPython `resume.py:1334-1340` parity): the root frame's
                // `jitcode_index` holds the outer program pc (from
                // `build_state_field_snapshot`'s `program_pc` argument);
                // subsequent frames' `jitcode_index` holds
                // `parent_descr_idx` (set by `BC_INLINE_CALL` at the
                // sub-frame setup).  We thread the most-recently-resolved
                // jitcode through `last_resolved` so
                // `resolve_jitcode(parent_descr_idx, pc)` can walk
                // `parent.descrs[idx].as_jitcode()` without re-invoking
                // the factory.
                let last_resolved: std::cell::RefCell<
                    Option<std::sync::Arc<majit_metainterp::JitCode>>,
                > = std::cell::RefCell::new(None);
                let resolve_jitcode =
                    |jitcode_index: i32, pc: i32| -> Option<crate::resume::ResolvedJitCode> {
                        let resolved_jitcode = if last_resolved.borrow().is_none() {
                            // Root frame: build via factory(program_pc).
                            let factory = self.jitcode_factory.as_ref()?;
                            let jitcode = factory(env, jitcode_index as usize, 0)?;
                            std::sync::Arc::new(jitcode)
                        } else {
                            // Sub-frame: index into the parent's
                            // `descrs` array.  Mirrors RPython
                            // `BC_INLINE_CALL` operand decoding
                            // (`blackhole.py:150-157`) where the `j`
                            // argcode resolves through `descrs[idx]`.
                            let parent = last_resolved
                                .borrow()
                                .as_ref()
                                .expect("parent exists")
                                .clone();
                            parent
                                .exec
                                .descrs
                                .get(jitcode_index as usize)
                                .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode)?
                                .clone()
                        };
                        *last_resolved.borrow_mut() = Some(resolved_jitcode.clone());
                        Some(crate::resume::ResolvedJitCode::new(
                            resolved_jitcode,
                            pc as usize,
                        ))
                    };

                let fallback_alloc = crate::resume::NullAllocator;
                let allocator: &dyn crate::resume::BlackholeAllocator = self
                    .blackhole_allocator
                    .as_deref()
                    .unwrap_or(&fallback_alloc);

                let mut bh_builder = crate::blackhole::BlackholeInterpBuilder::new();
                bh_builder.setup_cached_control_opcodes(
                    self.meta_interp().staticdata.op_live,
                    self.meta_interp().staticdata.op_catch_exception,
                    self.meta_interp().staticdata.op_rvmprof_code,
                );
                let all_liveness = self.meta_interp().staticdata.liveness_info.as_slice();
                let bh = crate::resume::blackhole_from_resumedata(
                    &mut bh_builder,
                    &resolve_jitcode,
                    rd_numb,
                    rd_consts_slice,
                    all_liveness,
                    &raw_values,
                    Some(&exit_layout.exit_types),
                    rd_virtuals_slice,
                    Some(&storage.rd_pendingfields), // rd_guard_pendingfields
                    Some(&self.meta_interp().virtualref_info as &dyn crate::resume::VRefInfo),
                    None, // vinfo
                    None, // ginfo
                    allocator,
                );
                if let Some((bh, _vable_ptr)) = bh {
                    let exc =
                        crate::blackhole::BlackholeInterpreter::prepare_resume_from_failure(0);
                    let jit_exc = crate::blackhole::run_forever_with_portal(
                        &mut bh_builder,
                        bh,
                        exc,
                        self.portal_runner.as_ref().map(|r| {
                            r.as_ref()
                                as &dyn Fn(
                                    &crate::jitexc::JitException,
                                ) -> Result<
                                    (crate::blackhole::BhReturnType, i64),
                                    crate::jitexc::JitException,
                                >
                        }),
                    );
                    // compile.py:716 assert 0, "unreachable"
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[bh] back_edge_internal: run_forever completed with {:?}",
                            jit_exc
                        );
                    }
                    return Some(target_pc);
                }
            }

            return Some(target_pc);
        }

        self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env);
        None
    }

    fn back_edge_or_run_compiled_internal(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> Option<DetailedDriverRunOutcome> {
        if self.meta.is_tracing() {
            return None;
        }
        if !state.can_trace() {
            if crate::majit_log_enabled() {
                eprintln!("[jit] back_edge blocked: can_trace=false key={}", green_key);
            }
            return None;
        }

        if self.meta.has_compiled_loop(green_key) {
            return Some(self.run_compiled_detailed_keyed(green_key, state, pre_run));
        }

        self.maybe_start_tracing(green_key, structured_green_key, target_pc, state, env);
        None
    }

    /// Force-start tracing for a function entry.
    ///
    /// Bypasses the WarmEnterState hot counter (the caller already did its own
    /// counting). Used for function-entry JIT where the threshold is
    /// controlled externally.
    pub fn force_start_tracing(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) {
        // Note: no is_hot_or_tracing check here — the caller (try_function_entry_jit)
        // already verified the threshold. force_start_tracing must unconditionally start.
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return;
        }

        match self.meta.force_start_tracing(
            green_key,
            (state.code_ptr(), target_pc),
            descriptor,
            &live_values,
        ) {
            BackEdgeAction::StartedTracing => {
                if let Some(ctx) = self.meta.trace_ctx() {
                    ctx.header_pc = target_pc;
                }
                let mut sym = S::create_sym(&meta, target_pc);
                state.initialize_sym(&mut sym, &meta);
                self.sym = Some(sym);
                self.meta.begin_trace_session(meta);
            }
            _ => {}
        }
    }

    #[cold]
    #[inline(never)]
    /// RPython warmstate.py:425 bound_reached parity.
    ///
    /// Called when the counter threshold has ALREADY fired (from eval_loop_jit).
    /// Bypasses counter.tick() inside maybe_compile, allowing decay_counters()
    /// to be called before tracing starts.
    pub fn bound_reached(&mut self, green_key: u64, target_pc: usize, state: &mut S, env: &S::Env) {
        if self.meta.is_tracing() {
            return;
        }
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return;
        }

        match self.meta.bound_reached(
            green_key,
            (state.code_ptr(), target_pc),
            None,
            descriptor,
            &live_values,
        ) {
            BackEdgeAction::StartedTracing => {
                if let Some(ctx) = self.meta.trace_ctx() {
                    ctx.header_pc = target_pc;
                }
                let mut sym = S::create_sym(&meta, target_pc);
                state.initialize_sym(&mut sym, &meta);
                self.sym = Some(sym);
                self.meta.begin_trace_session(meta);
            }
            _ => {}
        }
    }

    fn maybe_start_tracing(
        &mut self,
        green_key: u64,
        structured_green_key: Option<GreenKey>,
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
    ) {
        let meta = state.build_meta(target_pc, env);
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return;
        }

        match self.meta.on_back_edge_typed(
            green_key,
            (state.code_ptr(), target_pc),
            structured_green_key,
            descriptor,
            &live_values,
        ) {
            BackEdgeAction::Interpret => {}
            BackEdgeAction::StartedTracing => {
                if let Some(ctx) = self.meta.trace_ctx() {
                    ctx.header_pc = target_pc;
                }
                let mut sym = S::create_sym(&meta, target_pc);
                state.initialize_sym(&mut sym, &meta);
                self.sym = Some(sym);
                self.meta.begin_trace_session(meta);
            }
            BackEdgeAction::AlreadyTracing | BackEdgeAction::RunCompiled => {}
        }
    }

    fn driver_descriptor_for(&self, state: &S, meta: &S::Meta) -> Option<JitDriverStaticData> {
        self.descriptor
            .clone()
            .or_else(|| state.driver_descriptor(meta))
    }

    fn live_values_match_descriptor(
        descriptor: Option<&JitDriverStaticData>,
        live_values: &[Value],
    ) -> bool {
        let Some(descriptor) = descriptor else {
            return true;
        };
        let reds = descriptor.reds();
        reds.len() == live_values.len()
            && reds
                .iter()
                .zip(live_values.iter())
                .all(|(var, value)| var.tp == value.get_type())
    }

    fn flatten_virtualizable_values(
        info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> Vec<Value> {
        let mut values = Vec::with_capacity(
            static_boxes.len() + array_boxes.iter().map(Vec::len).sum::<usize>(),
        );
        for (field, &raw) in info.static_fields.iter().zip(static_boxes.iter()) {
            values.push(match field.field_type {
                Type::Int => Value::Int(raw),
                Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                Type::Void => Value::Void,
            });
        }
        for (array, items) in info.array_fields.iter().zip(array_boxes.iter()) {
            for &raw in items {
                values.push(match array.item_type {
                    Type::Int => Value::Int(raw),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Void => Value::Void,
                });
            }
        }
        values
    }

    /// warmstate.py:482-511: extend live values with virtualizable fields
    /// when the compiled loop expects more inputs than currently available.
    /// RPython always has jitdriver_sd available; pyre may have descriptor=None
    /// when re-entering from guard failure, so fall back to virtualizable_info
    /// directly.
    fn extend_compiled_live_values(
        &self,
        green_key: u64,
        state: &S,
        meta: &S::Meta,
        descriptor: Option<&JitDriverStaticData>,
        mut live_values: Vec<Value>,
    ) -> Option<Vec<Value>> {
        let compiled_inputs = self.meta.get_compiled_num_inputs(green_key)?;
        if compiled_inputs <= live_values.len() {
            return Some(live_values);
        }
        // Try descriptor path first (RPython jitdriver_sd.virtualizable).
        let vable_name = descriptor
            .and_then(|d| d.virtualizable())
            .map(|v| v.name.clone());
        // Fall back to virtualizable_info's default name if no descriptor.
        let info = self.meta.virtualizable_info()?;
        // jitdriver_sd.virtualizable_info.name (interp_jit.py:25)
        let name = vable_name.unwrap_or_else(|| info.name.clone());
        let (static_boxes, array_boxes) = state.export_virtualizable_boxes(meta, &name, info)?;
        let extra_values = Self::flatten_virtualizable_values(info, &static_boxes, &array_boxes);
        if live_values.len() + extra_values.len() != compiled_inputs {
            return None;
        }
        live_values.extend(extra_values);
        Some(live_values)
    }

    fn sync_before(
        &mut self,
        state: &mut S,
        meta: &S::Meta,
        descriptor: Option<&JitDriverStaticData>,
    ) -> bool {
        // Descriptor-gated token-reset path — RPython
        // compile.py::sync_before_jit calls `vinfo.reset_vable_token(virt)`
        // when the JitDriver has a virtualizable declared.  Pyre routes
        // this through the interpreter's `sync_named_virtualizable_before_jit`.
        if let Some(descriptor) = descriptor {
            if let Some(virtualizable) = descriptor.virtualizable() {
                let ok = state.sync_named_virtualizable_before_jit(
                    meta,
                    &virtualizable.name,
                    self.meta.virtualizable_info().map(|a| a.as_ref()),
                );
                if !ok {
                    return false;
                }
            }
        }

        // Refresh the trace-entry vable heap pointer so `initialize_virtualizable`
        // reads array lengths directly from the concrete virtualizable object
        // (pyjitpl.py:3302 `vinfo.read_boxes(cpu, virtualizable, startindex)`).
        // `VirtualizableInfo.name` is the canonical identifier declared via
        // `virtualizable!(name = "...")`, so we can feed it straight to the
        // interpreter hooks without going through the descriptor.
        self.meta.set_vable_ptr(std::ptr::null());
        self.meta.set_vable_array_lengths(Vec::new());
        let info_clone = self.meta.virtualizable_info().cloned();
        if let Some(ref info) = info_clone {
            let vable_name = info.name.clone();
            if let Some(ptr) = state.virtualizable_heap_ptr(meta, &vable_name, info) {
                self.meta.set_vable_ptr(ptr.cast_const());
            }
            // Fallback cache for layouts that cannot expose array length on
            // the heap object alone (header-less embedded arrays). Unused in
            // production since PyFrame's `locals_cells_stack_w` is embedded
            // with a length_offset; retained so tests that don't stage a
            // fake heap object still reach `initialize_virtualizable`.
            let fallback_lengths = state
                .virtualizable_array_lengths(meta, &vable_name, info)
                .unwrap_or_default();
            self.meta.set_vable_array_lengths(fallback_lengths);
        }
        true
    }

    fn sync_after(&self, state: &mut S, meta: &S::Meta, descriptor: Option<&JitDriverStaticData>) {
        let Some(descriptor) = descriptor else {
            return;
        };
        let Some(virtualizable) = descriptor.virtualizable() else {
            return;
        };
        state.sync_named_virtualizable_after_jit(
            meta,
            &virtualizable.name,
            self.meta.virtualizable_info().map(|a| a.as_ref()),
        );
    }

    /// Register an additional entry point for this driver.
    ///
    /// Multiple functions can share the same JitDriver and compiled loops.
    /// Each entry point has a name and a green key schema describing the
    /// types of its green (loop-invariant) variables.
    pub fn register_entry_point(&mut self, name: &str, green_key_schema: &[Type]) {
        // Accept IR `Type` for source compatibility. Callers that need to
        // declare STR/UNICODE greens can call `register_entry_point_typed`.
        let schema: Vec<GreenType> = green_key_schema.iter().copied().map(Into::into).collect();
        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            schema,
        });
    }

    /// warmstate.py:564-565 — register an entry point with an explicit
    /// `GreenType` schema. Frontends that declare STR / UNICODE green
    /// args call this path so `comparekey` / `get_uhash` dispatch through
    /// `equal_whatever` / `hash_whatever` correctly.
    pub fn register_entry_point_typed(&mut self, name: &str, green_key_schema: &[GreenType]) {
        self.entry_points.push(EntryPoint {
            name: name.to_string(),
            schema: green_key_schema.to_vec(),
        });
    }

    /// Return all registered entry points.
    pub fn entry_points(&self) -> &[EntryPoint] {
        &self.entry_points
    }

    /// Look up an entry point by name.
    pub fn find_entry_point(&self, name: &str) -> Option<&EntryPoint> {
        self.entry_points.iter().find(|ep| ep.name == name)
    }

    /// Set virtualizable info for frame virtualization.
    pub fn set_virtualizable_info(&mut self, info: std::sync::Arc<VirtualizableInfo>) {
        self.meta.set_virtualizable_info(info);
    }

    /// Set the compilation threshold.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.meta.set_threshold(threshold);
    }

    /// RPython-compatible wrapper used by compile.py.
    pub fn send_loop_to_backend(
        &mut self,
        jump_args: &[OpRef],
        meta: S::Meta,
    ) -> crate::pyjitpl::CompileOutcome {
        self.meta.compile_loop(jump_args, meta)
    }

    /// RPython-compatible wrapper used by compile.py for bridges.
    pub fn send_bridge_to_backend(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        finish_args: &[OpRef],
    ) -> crate::pyjitpl::BridgeCompileResult {
        self.meta
            .close_bridge(green_key, trace_id, fail_index, finish_args)
    }

    /// Set the bridge compilation threshold.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.meta.set_bridge_threshold(threshold);
    }

    /// Set the trace eagerness (guard failure threshold for bridge tracing).
    pub fn set_trace_eagerness(&mut self, eagerness: u32) {
        self.meta.set_trace_eagerness(eagerness);
    }

    /// Set a callback for loop compilation events.
    pub fn set_on_compile_loop(&mut self, f: impl Fn(u64, usize, usize) + Send + 'static) {
        self.meta.set_on_compile_loop(f);
    }

    /// Set a callback for guard failure events.
    pub fn set_on_guard_failure(&mut self, f: impl Fn(u64, u32, u32) + Send + 'static) {
        self.meta.set_on_guard_failure(f);
    }

    /// Set a JIT parameter by name at runtime.
    ///
    /// Supported parameters:
    /// - `"threshold"` — compilation hot-count threshold
    /// - `"trace_eagerness"` — guard failure count before bridge tracing
    /// - `"bridge_threshold"` — guard failure count before bridge compilation
    /// - `"function_threshold"` — function call count before inlining
    ///
    /// Unknown parameter names are silently ignored.
    pub fn set_param(&mut self, name: &str, value: i64) {
        match name {
            "threshold" => self.meta.set_threshold(value as u32),
            "trace_eagerness" => self.meta.set_trace_eagerness(value as u32),
            "bridge_threshold" => self.meta.set_bridge_threshold(value as u32),
            "function_threshold" => self.meta.set_function_threshold(value as u32),
            "max_unroll_recursion" => self.set_param_max_unroll_recursion(value as usize),
            _ => self.meta.warm_state_mut().set_param(name, value),
        }
    }

    /// Return a snapshot of the cumulative JIT compilation statistics.
    pub fn get_stats(&self) -> JitStats {
        self.meta.get_stats()
    }

    /// Get direct access to the underlying MetaInterp.
    pub fn meta_interp(&self) -> &MetaInterp<S::Meta> {
        &self.meta
    }

    /// Get mutable access to the underlying MetaInterp.
    pub fn meta_interp_mut(&mut self) -> &mut MetaInterp<S::Meta> {
        &mut self.meta
    }

    /// framework.py `root_walker.walk_roots` parity: visit every Ref-typed
    /// entry in every live compiled trace's `rd_consts` pool so the minor
    /// collector can forward them. See `MetaInterp::walk_rd_consts_refs`
    /// for the full rationale.
    pub fn walk_rd_consts_refs(&mut self, visitor: impl FnMut(&mut majit_ir::GcRef)) {
        self.meta.walk_rd_consts_refs(visitor);
    }

    pub fn run_compiled_with_blackhole_fallback_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> crate::pyjitpl::DriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, descriptor.as_ref()) {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let Some(live_values) = self.extend_compiled_live_values(
            green_key,
            state,
            &meta,
            descriptor.as_ref(),
            live_values,
        ) else {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        pre_run();
        let Some(result) = self
            .meta
            .run_with_blackhole_fallback_with_values(green_key, &live_values)
        else {
            return crate::pyjitpl::DriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        match result {
            crate::pyjitpl::BlackholeRunResult::Finished {
                values,
                typed_values,
                meta,
                via_blackhole,
                ..
            } => {
                if let Some(values) = typed_values.as_ref() {
                    state.restore_values(&meta, values);
                } else {
                    state.restore(&meta, &values);
                }
                self.sync_after(state, &meta, descriptor.as_ref());
                crate::pyjitpl::DriverRunOutcome::Finished { via_blackhole }
            }
            crate::pyjitpl::BlackholeRunResult::Jump {
                values,
                typed_values,
                exit_layout,
                meta,
                via_blackhole,
                ..
            } => {
                if let Some(values) = typed_values.as_ref() {
                    state.restore_values(&meta, values);
                } else if let Some(layout) = exit_layout.as_ref() {
                    state.restore_values(&meta, &Self::decode_exit_layout_values(&values, layout));
                } else if let Some(values) =
                    Self::decode_descriptor_values(descriptor.as_ref(), &values)
                {
                    state.restore_values(&meta, &values);
                } else {
                    state.restore(&meta, &values);
                }
                self.sync_after(state, &meta, descriptor.as_ref());
                crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole }
            }
            crate::pyjitpl::BlackholeRunResult::GuardFailure {
                fail_values,
                typed_fail_values,
                exit_layout,
                meta,
                recovery,
                via_blackhole,
                ..
            } => {
                let patched_resume_layout = recovery.as_ref().and_then(|recovery| {
                    recovery.resume_layout.as_ref().and_then(|layout| {
                        Self::resume_layout_with_descriptor_slot_types(descriptor.as_ref(), layout)
                    })
                });
                let restored = if let Some(recovery) = recovery.as_ref() {
                    state.restore_guard_failure_with_resume_layout(
                        &meta,
                        &fail_values,
                        recovery.reconstructed_state.as_ref(),
                        patched_resume_layout
                            .as_ref()
                            .or(recovery.resume_layout.as_ref())
                            .map(|layout| layout.frame_layouts.as_slice()),
                        &recovery.materialized_virtuals,
                        &recovery.pending_field_writes,
                        &recovery.exception,
                    )
                } else if let Some(values) = typed_fail_values.as_ref() {
                    state.restore_guard_failure_values(&meta, values, &ExceptionState::default())
                } else if let Some(layout) = exit_layout.as_ref() {
                    state.restore_guard_failure_values(
                        &meta,
                        &Self::decode_exit_layout_values(&fail_values, layout),
                        &ExceptionState::default(),
                    )
                } else if let Some(values) =
                    Self::decode_descriptor_values(descriptor.as_ref(), &fail_values)
                {
                    state.restore_guard_failure_values(&meta, &values, &ExceptionState::default())
                } else {
                    state.restore_guard_failure_with_resume_layout(
                        &meta,
                        &fail_values,
                        None,
                        None,
                        &[],
                        &[],
                        &ExceptionState::default(),
                    )
                };
                if restored {
                    self.sync_after(state, &meta, descriptor.as_ref());
                }
                crate::pyjitpl::DriverRunOutcome::GuardFailure {
                    restored,
                    via_blackhole,
                }
            }
            crate::pyjitpl::BlackholeRunResult::Abort { .. } => {
                crate::pyjitpl::DriverRunOutcome::Abort {
                    restored: false,
                    via_blackhole: true,
                }
            }
        }
    }

    pub fn run_compiled_with_blackhole_fallback_declarative<D: DeclarativeJitDriver>(
        &mut self,
        green_values: &[i64],
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> Result<crate::pyjitpl::DriverRunOutcome, &'static str> {
        let green_key = D::green_key(green_values)?;
        Ok(self.run_compiled_with_blackhole_fallback_keyed(green_key.hash_u64(), state, pre_run))
    }

    pub fn run_compiled_detailed_keyed(
        &mut self,
        green_key: u64,
        state: &mut S,
        pre_run: impl FnOnce(),
    ) -> DetailedDriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !state.is_compatible(&meta) || !self.sync_before(state, &meta, descriptor.as_ref()) {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if crate::majit_log_enabled() {
            let vals: Vec<String> = live_values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("[{}]={:?}", i, v))
                .collect();
            eprintln!("[jit] run_bridge live_values: {}", vals.join(", "));
        }
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let Some(live_values) = self.extend_compiled_live_values(
            green_key,
            state,
            &meta,
            descriptor.as_ref(),
            live_values,
        ) else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        pre_run();
        let Some(result) = self
            .meta
            .run_compiled_detailed_with_values(green_key, &live_values)
        else {
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        if result.is_finish {
            let typed_values = result.typed_values.clone();
            let is_exit_frame_with_exception = result.is_exit_frame_with_exception;
            drop(result);
            return DetailedDriverRunOutcome::Finished {
                typed_values,
                via_blackhole: false,
                raw_int_result: self.meta.has_raw_int_finish(),
                is_exit_frame_with_exception,
            };
        }

        let exit_meta = result.meta.clone();
        state.restore_values(&exit_meta, &result.typed_values);
        self.sync_after(state, &exit_meta, descriptor.as_ref());
        DetailedDriverRunOutcome::Jump {
            via_blackhole: false,
        }
    }

    /// RPython compile.py:714 + pyjitpl.py rebuild_state_after_failure():
    /// run compiled code, distinguish normal JUMP from guard failure, and
    /// allow the caller to rebuild interpreter state and start bridge tracing
    /// from the recovered resume pc.
    /// warmstate.py:398-422 execute_assembler: run compiled code and return
    /// the outcome. For guard failures, returns raw exit data without any
    /// state restoration — the caller processes it via handle_fail().
    pub fn run_compiled_detailed_with_bridge_keyed(
        &mut self,
        green_key: u64,
        target_pc: usize,
        state: &mut S,
        _env: &S::Env,
        pre_run: impl FnOnce(),
    ) -> DetailedDriverRunOutcome {
        let Some(meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=missing-compiled-meta target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        let descriptor = self.driver_descriptor_for(state, &meta);
        // warmstate.py:482-511: RPython enters execute_token directly
        // when the green key matches — no is_compatible check.
        // pyre still needs this gate because compiled code reads
        // virtualizable slots by positional index (not by field
        // descriptor), so a valuestackdepth mismatch causes wrong
        // slot reads. Remove when virtualizable field access uses
        // descriptor-based addressing (RPython virtualizable.py parity).
        if !state.is_compatible(&meta) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-skip] key={} reason=incompatible-state target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=sync-before target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let live_values = state.extract_live_values(&meta);
        if crate::majit_log_enabled() {
            let vals: Vec<String> = live_values
                .iter()
                .enumerate()
                .map(|(i, v)| format!("[{}]={:?}", i, v))
                .collect();
            eprintln!("[jit] BRIDGE live_values: {}", vals.join(", "));
        }
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=live-values-descriptor-mismatch target_pc={} nvals={}",
                    green_key,
                    target_pc,
                    live_values.len()
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        }
        let Some(live_values) = self.extend_compiled_live_values(
            green_key,
            state,
            &meta,
            descriptor.as_ref(),
            live_values,
        ) else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=extend-live-values target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };
        pre_run();
        let Some(result) = self
            .meta
            .run_compiled_detailed_with_values(green_key, &live_values)
        else {
            if crate::majit_log_enabled() {
                eprintln!(
                    "[jit][run-compiled-abort] key={} reason=backend-run target_pc={}",
                    green_key, target_pc
                );
            }
            return DetailedDriverRunOutcome::Abort {
                restored: false,
                via_blackhole: false,
            };
        };

        let is_finish = result.is_finish;
        let is_exit_frame_with_exception = result.is_exit_frame_with_exception;
        let exit_meta = result.meta.clone();
        let fail_index = result.fail_index;
        let trace_id = result.trace_id;
        let exit_layout = result.exit_layout.clone();
        let typed_values = result.typed_values.clone();
        let raw_values = result.values.clone();
        let descr_addr = result.descr_addr;
        drop(result);

        // memmgr.py:58-61: keep_loop_alive(loop_token)
        self.meta.keep_loop_alive(green_key);

        if is_finish {
            return DetailedDriverRunOutcome::Finished {
                typed_values,
                via_blackhole: false,
                raw_int_result: self.meta.has_raw_int_finish(),
                is_exit_frame_with_exception,
            };
        }

        // Normal loop back-edge JUMP, not a guard failure.
        if fail_index == u32::MAX {
            state.restore_values(&exit_meta, &typed_values);
            self.sync_after(state, &exit_meta, descriptor.as_ref());
            return DetailedDriverRunOutcome::Jump {
                via_blackhole: false,
            };
        }

        // compile.py:701-717 handle_fail / must_compile: single tick+check.
        let guard_loop_key = if exit_layout.rd_loop_token != 0 {
            exit_layout.rd_loop_token
        } else {
            green_key
        };
        let (must_compile, owning_key) = self.meta.must_compile_with_values(
            guard_loop_key,
            trace_id,
            fail_index,
            &raw_values,
            descr_addr,
        );
        // compile.py:702-703: must_compile() and not stack_almost_full().
        let should_bridge =
            must_compile && !majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full();

        // Return raw guard failure data. State restoration and bridge/
        // blackhole decision happen in the caller's handle_fail().
        DetailedDriverRunOutcome::GuardFailure {
            fail_index,
            trace_id,
            should_bridge,
            owning_key,
            descr_addr,
            raw_values,
            exit_layout,
        }
    }

    fn decode_descriptor_values(
        descriptor: Option<&JitDriverStaticData>,
        raw_values: &[i64],
    ) -> Option<Vec<Value>> {
        let descriptor = descriptor?;
        let reds = descriptor.reds();
        if reds.len() != raw_values.len() {
            return None;
        }
        Some(
            reds.iter()
                .zip(raw_values.iter().copied())
                .map(|(var, raw)| match var.tp {
                    Type::Int => Value::Int(raw),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Void => Value::Void,
                })
                .collect(),
        )
    }

    fn decode_exit_layout_values(raw_values: &[i64], layout: &CompiledExitLayout) -> Vec<Value> {
        layout
            .exit_types
            .iter()
            .enumerate()
            .map(|(index, tp)| {
                let raw = raw_values.get(index).copied().unwrap_or(0);
                match tp {
                    Type::Int => Value::Int(raw),
                    Type::Ref => Value::Ref(majit_ir::GcRef(raw as usize)),
                    Type::Float => Value::Float(f64::from_bits(raw as u64)),
                    Type::Void => Value::Void,
                }
            })
            .collect()
    }

    fn resume_layout_with_descriptor_slot_types(
        descriptor: Option<&JitDriverStaticData>,
        resume_layout: &ResumeLayoutSummary,
    ) -> Option<ResumeLayoutSummary> {
        let descriptor = descriptor?;
        let red_types: Vec<Type> = descriptor.reds().iter().map(|var| var.tp).collect();
        let last = resume_layout.frame_layouts.last()?;
        if last.slot_types.is_some() || last.slot_layouts.len() != red_types.len() {
            return None;
        }
        let mut patched = resume_layout.clone();
        if let Some(last) = patched.frame_layouts.last_mut() {
            last.slot_types = Some(red_types);
        }
        Some(patched)
    }

    /// Invalidate a compiled loop, forcing fallback to interpretation.
    pub fn invalidate_loop(&mut self, green_key: u64) {
        self.meta.invalidate_loop(green_key);
    }

    /// Check whether a compiled loop exists for a given green key.
    #[inline]
    pub fn has_compiled_loop(&self, green_key: u64) -> bool {
        self.meta.has_compiled_loop(green_key)
    }

    /// Actual key the last compile_loop stored under.
    pub fn last_compiled_key(&self) -> Option<u64> {
        self.meta.last_compiled_key()
    }

    /// warmstate.py:437-444 starting cell's green_key (the cell on which
    /// TRACING must be cleared in the finally block). Returns None when
    /// no trace is in progress.
    pub fn starting_green_key(&self) -> Option<u64> {
        self.meta.starting_green_key()
    }

    /// Check if any guard in the compiled trace has Float-typed fail_args.
    pub fn compiled_trace_has_float_guards(&self, green_key: u64) -> bool {
        self.meta.compiled_trace_has_float_guards(green_key)
    }

    /// Check if the compiled trace is safe for bridge compilation.
    pub fn compiled_trace_safe_for_bridge(
        &self,
        green_key: u64,
        slot_types: &[majit_ir::Type],
    ) -> bool {
        self.meta
            .compiled_trace_safe_for_bridge(green_key, slot_types)
    }

    /// Remove all compiled code. Used when guard-fail recovery detects
    /// unrecoverable state (e.g., null Ref slots from incomplete
    /// resume data), preventing repeated entry→guard-fail loops.
    pub fn invalidate_all_compiled(&mut self) {
        self.meta.clear_compiled_loops();
    }

    /// interp_jit.py:259 releaseall → memmgr.py:85 release_all_loops.
    ///
    /// memmgr.py:85 release_all_loops → alive_loops.clear().
    ///
    /// RPython: alive_loops holds the only strong references to
    /// LoopTokens. Clearing it makes tokens eligible for GC collection.
    /// JitCells hold weakrefs, so tokens referenced by an active thread
    /// stack survive until the stack unwinds. Tokens are NOT invalidated.
    ///
    /// majit has no GC or weakrefs. We clear compiled_loops (the
    /// equivalent of alive_loops) which drops majit's strong ownership.
    /// Warm-state cells are untouched — they can re-trigger compilation.
    /// Machine code is not invalidated: if a guard failure path still
    /// references a removed entry it will simply miss the lookup and
    /// fall back to the interpreter, matching the RPython "dangling
    /// weakref → re-compile" path.
    pub fn mark_all_loops_for_release(&mut self) {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][releaseall] releasing {} compiled loops",
                self.meta.compiled_loops.len()
            );
        }
        self.meta.compiled_loops.clear();
    }

    /// Invalidate compiled code for a specific trace_id, removing the
    /// compiled_loops entry whose root_trace_id matches.
    pub fn invalidate_compiled_trace(&mut self, trace_id: u64) {
        self.meta
            .compiled_loops
            .retain(|_, entry| entry.root_trace_id != trace_id);
    }

    /// Remove the compiled entry for a specific green key.
    /// Unlike invalidate_loop (which flags the token), this removes the
    /// entry entirely so both has_compiled_loop and has_compiled_targets
    /// return false, allowing recompilation.
    pub fn remove_compiled_loop(&mut self, green_key: u64) {
        self.meta.compiled_loops.remove(&green_key);
    }

    /// warmspot.py:449 — set the per-driver result_type.
    pub fn set_result_type(&mut self, tp: Type) {
        self.meta.set_result_type(tp);
    }

    /// warmstate.py:385 — whether this driver's portal returns a raw int.
    pub fn has_raw_int_finish(&self) -> bool {
        self.meta.has_raw_int_finish()
    }

    /// Get the loop token number for a compiled loop.
    pub fn get_loop_token_number(&self, green_key: u64) -> Option<u64> {
        self.meta.get_loop_token(green_key).map(|t| t.number)
    }

    /// Get the pre-allocated token number for the trace being recorded.
    ///
    /// Returns `Some(number)` if `green_key` matches the current trace's
    /// target, enabling self-recursive call_assembler emission.
    pub fn get_pending_token_number(&self, green_key: u64) -> Option<u64> {
        self.meta.get_pending_token_number(green_key)
    }

    /// Decide how to handle a function call during tracing.
    ///
    /// Returns `Inline` if the callee should be traced through,
    /// `CallAssembler` if it already has compiled code, or
    /// `ResidualCall` if it should be left as an opaque call.
    pub fn should_inline(&mut self, callee_key: u64, callee_raw: (usize, usize)) -> InlineDecision {
        self.meta.should_inline(callee_key, callee_raw)
    }

    /// Inline decision with externally-held ctx (merge_point callback).
    pub fn should_inline_with_ctx(
        &mut self,
        callee_key: u64,
        callee_raw: (usize, usize),
        ctx: &crate::trace_ctx::TraceCtx,
    ) -> InlineDecision {
        self.meta
            .should_inline_with_ctx(callee_key, callee_raw, ctx)
    }

    /// Begin inlining a function call during tracing.
    ///
    /// Records EnterPortalFrame and pushes an inline frame.
    /// Returns `true` if inlining started successfully.
    pub fn enter_inline_frame(&mut self, callee_raw: (usize, usize)) -> bool {
        self.meta.enter_inline_frame(callee_raw)
    }

    /// End an inlined function call during tracing.
    ///
    /// Records LeavePortalFrame and pops the inline frame.
    pub fn leave_inline_frame(&mut self) {
        self.meta.leave_inline_frame()
    }

    /// Get the current inlining depth.
    pub fn inline_depth(&self) -> usize {
        self.meta.inline_depth()
    }

    /// RPython equivalent: `opimpl_hint_force_virtualizable(box)`
    ///
    /// Call during tracing when the interpreter encounters
    /// `hint_force_virtualizable`. Emits IR to flush virtualizable
    /// boxes back to the heap.
    pub fn opimpl_hint_force_virtualizable(&mut self, vable_opref: OpRef) {
        self.meta.opimpl_hint_force_virtualizable(vable_opref);
    }

    pub fn opimpl_getfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta.opimpl_getfield_vable_int(vable_opref, fielddescr)
    }

    pub fn opimpl_getfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta.opimpl_getfield_vable_ref(vable_opref, fielddescr)
    }

    pub fn opimpl_getfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta
            .opimpl_getfield_vable_float(vable_opref, fielddescr)
    }

    pub fn opimpl_setfield_vable_int(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_int(vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_setfield_vable_ref(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_ref(vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_setfield_vable_float(
        &mut self,
        vable_opref: OpRef,
        fielddescr: DescrRef,
        value: OpRef,
        concrete: Value,
    ) {
        self.meta
            .opimpl_setfield_vable_float(vable_opref, fielddescr, value, concrete);
    }

    pub fn opimpl_getarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta
            .opimpl_getarrayitem_vable_int(vable_opref, index, index_runtime_value, fdescr)
    }

    pub fn opimpl_getarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta
            .opimpl_getarrayitem_vable_ref(vable_opref, index, index_runtime_value, fdescr)
    }

    pub fn opimpl_getarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        fdescr: DescrRef,
    ) -> (OpRef, Value) {
        self.meta
            .opimpl_getarrayitem_vable_float(vable_opref, index, index_runtime_value, fdescr)
    }

    pub fn opimpl_setarrayitem_vable_int(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_int(
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
        );
    }

    pub fn opimpl_setarrayitem_vable_ref(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_ref(
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
        );
    }

    pub fn opimpl_setarrayitem_vable_float(
        &mut self,
        vable_opref: OpRef,
        index: OpRef,
        index_runtime_value: i64,
        value: OpRef,
        concrete: Value,
        fdescr: DescrRef,
    ) {
        self.meta.opimpl_setarrayitem_vable_float(
            vable_opref,
            index,
            index_runtime_value,
            value,
            concrete,
            fdescr,
        );
    }

    pub fn opimpl_arraylen_vable(&mut self, vable_opref: OpRef, fdescr: DescrRef) -> OpRef {
        self.meta.opimpl_arraylen_vable(vable_opref, fdescr)
    }

    /// Start bridge tracing from a guard failure point.
    ///
    /// Uses the compiled loop's stored meta so that the sym's
    /// storage_layout matches the parent loop's inputargs format.
    /// `resume_pc` is where interpretation resumes after the guard failure.
    /// `loop_header_pc` is the parent loop's back-edge target.
    pub fn start_bridge_tracing(
        &mut self,
        green_key: u64,
        trace_id: u64,
        fail_index: u32,
        state: &mut S,
        env: &S::Env,
        resume_pc: usize,
        _loop_header_pc: usize,
    ) -> bool {
        let Some(_loop_meta) = self.meta.get_compiled_meta(green_key).cloned() else {
            return false;
        };

        if !state.can_trace() {
            return false;
        }

        // RPython parity: bridge tracing starts from the state rebuilt after
        // guard failure, not from the original loop-header metadata.
        // compile.py handle_fail -> pyjitpl.handle_guard_failure ->
        // rebuild_state_after_failure() resumes the current interpreter
        // position and traces forward from there.
        let mut trace_meta = state.build_meta(resume_pc, env);

        let live_values = state.extract_live(&trace_meta);
        let live_types = state.live_value_types(&trace_meta);

        let retrace = match self.meta.start_retrace_from_guard(
            green_key,
            trace_id,
            fail_index,
            &live_values,
            &live_types,
        ) {
            Some(r) => r,
            None => return false,
        };

        // resume.py:1042 rebuild_from_resumedata parity: decode rd_numb
        // to reconstruct the complete frame for bridge tracing.
        // Falls back to update_meta_for_bridge (legacy truncation) when
        // rd_numb is not available.
        //
        // compile.py:853 `ResumeGuardDescr` storage — pass the shared
        // Arc to the decoder so it borrows the guard-owned pool (no
        // clone) and can thread the same handle into
        // `ResumeDataResult`.
        let resume_data_result = S::rebuild_from_resumedata(
            &mut trace_meta,
            &retrace.fail_types,
            retrace.storage.as_ref(),
        );
        if resume_data_result.is_none() {
            S::update_meta_for_bridge(&mut trace_meta, &retrace.fail_types);
        }

        let mut sym = S::create_sym(&trace_meta, resume_pc);
        state.initialize_sym(&mut sym, &trace_meta);
        self.sym = Some(sym);
        // pyjitpl.py:2890 parity: bridge traces start at resume_pc, not at
        // function entry (pc=0). Set header_pc so init_symbolic correctly
        // detects this is NOT a function-entry trace.
        // pyjitpl.py:2979 reached_loop_header parity:
        // Set live lookup callback for has_compiled_targets(greenkey).
        // RPython: ptoken = self.get_procedure_token(greenboxes)
        //          if has_compiled_targets(ptoken): ...
        // Use a raw pointer to self.meta for the callback — safe because
        // bridge tracing runs synchronously within start_bridge_tracing's
        // caller scope, so self.meta outlives the callback.
        let meta_ptr = &self.meta as *const _ as *const ();
        if let Some(ref mut ctx) = self.meta.tracing {
            ctx.header_pc = resume_pc;
            ctx.has_compiled_targets_fn = Some(Box::new(move |gk: u64| -> bool {
                let meta = unsafe { &*(meta_ptr as *const crate::pyjitpl::MetaInterp<S::Meta>) };
                meta.has_compiled_targets(gk)
            }));
        }
        // resume.py:1042 parity: map frame locals to bridge InputArg OpRefs
        // so bridge tracing sees locals as symbolic variables, not concrete values.
        // resume.py:945-956 getvirtual_ptr parity: when frame.values contains
        // RebuiltValue::Virtual entries, setup_bridge_sym must emit
        // NEW_WITH_VTABLE/SETFIELD_GC ops via the active trace ctx so the
        // bridge has fresh local instances instead of stale vable-array reads.
        if let Some(ref bfm) = resume_data_result {
            // bridgeopt.py:124 / resume.py:1260 parity: `Box(n, tp)` in
            // `rd_numb` indexes the guard's encoder-time liveboxes, whose
            // runtime values are the backend's raw deadframe (the guard's
            // `fail_args` image). `extract_live` returns the virtualizable
            // array in vable layout order — a different stream with a
            // different length. Pass the raw deadframe values that the
            // caller stashed via `set_pending_frontend_boxes` so
            // `decode_concrete(Box(n, tp))` reads `raw_values[n]` and
            // `fail_types[n]` (length-aligned with `fail_descr.fail_arg_types`).
            let raw_values: Vec<i64> = self
                .meta
                .pending_frontend_boxes_ref()
                .map(|s| s.to_vec())
                .unwrap_or_default();
            if let (Some(ref mut sym), Some(ref mut ctx)) =
                (self.sym.as_mut(), self.meta.tracing.as_mut())
            {
                S::setup_bridge_sym(
                    sym,
                    ctx,
                    bfm,
                    retrace.storage.as_deref().map(|s| s.rd_virtuals.as_slice()),
                    &raw_values,
                    &retrace.fail_types,
                );
            }
        }
        self.meta.begin_trace_session(trace_meta);
        // resume.py:1047-1055 parity:
        //   ResumeDataBoxReader.consume_boxes() rebuilds the frame state,
        //   and bridge tracing continues from that restored interpreter
        //   frame.  The authoritative slot metadata is therefore the
        //   state rebuilt by `state.build_meta(resume_pc, env)`, not the
        //   raw `RebuiltFrame.values` section.
        //
        // `RebuiltFrame.values` contains only the frame's register slots;
        // it does NOT have the synthetic [frame, ni, vsd] header used by
        // pyre's fallback `update_meta_for_bridge()` path.  Reinterpreting
        // those slot values as header-prefixed fail_arg types shifts the
        // frame layout and can mis-bind bridge return values.
        //
        // Keep `trace_meta` as built from the restored state when rd_numb
        // is available; only the no-rd_numb fallback should call
        // `update_meta_for_bridge()`.
        //
        // Constants are NOT injected into the pool up-front: resume.py:1245
        // `decode_box` returns live ConstInt/ConstPtr boxes at the moment
        // each tagged value is read, so majit's parity path registers each
        // constant via `ctx.const_int/const_ref/const_float` inside
        // `setup_bridge_sym`'s resolve closure and inside
        // `materialize_bridge_virtual`'s `decode_fieldnum`.
        self.resume_data_result = resume_data_result;
        let code_ptr = state.code_ptr();
        self.meta
            .set_bridge_trace_info(crate::pyjitpl::BridgeTraceInfo {
                green_key,
                trace_id,
                fail_index,
                code_ptr,
            });
        // RPython pyjitpl.py:2908 — bridge traces start with empty
        // current_merge_points (no loop header to match against).
        if let Some(ref mut ctx) = self.meta.tracing {
            ctx.clear_merge_points();
        }

        // RPython pyjitpl.py:3101 _prepare_exception_resumption parity:
        // For exception guard bridges, the caller should emit
        // SAVE_EXC_CLASS + SAVE_EXCEPTION at the trace start.
        // Store the flag so it's accessible to the pyre-jit caller.
        self.last_bridge_is_exception_guard = retrace.is_exception_guard;

        true
    }

    /// Generic back-edge runner for storage-pool interpreters.
    ///
    /// Encapsulates the common pattern: check tracing state, hash green key,
    /// try compiled execution, and handle guard failure / bridge compilation.
    ///
    /// The caller provides:
    /// - `green_values`: green key as i64 slice (e.g., `[target_pc, selected]`)
    /// - `target_pc`: the back-edge target PC
    /// - `pre_run`: callback to execute before compiled code (e.g., flush I/O)
    /// - `on_guard_failure`: callback to restore interpreter state from guard
    ///   failure values; returns `Some(resume_pc)` on success
    ///
    /// Returns `Some(resume_pc)` if compiled code ran or guard state was restored.
    #[inline]
    /// RPython parity: pyjitpl.py handle_guard_failure → rebuild_state_after_failure.
    /// `on_guard_failure` receives (state, meta, raw_values, exit_layout) where
    /// exit_layout contains recovery_layout with rd_virtuals equivalent for
    /// materializing virtual objects after guard failure.
    pub fn run_back_edge_generic(
        &mut self,
        green_values: &[i64],
        target_pc: usize,
        state: &mut S,
        env: &S::Env,
        pre_run: impl FnOnce(),
        on_guard_failure: impl Fn(
            &mut S,
            &S::Meta,
            &[i64],
            &crate::compile::CompiledExitLayout,
        ) -> Option<usize>,
    ) -> Option<usize> {
        if self.is_tracing() || !state.can_trace() {
            return None;
        }

        let key_hash = crate::green_key_hash(green_values);

        if !self.has_compiled_loop(key_hash) {
            // warmstate.py:446,465: cold fast path — check would_fire without
            // ticking to skip GreenKey allocation and build_meta for cold keys.
            // Advance counter via counter_tick which respects cell state
            // (DONT_TRACE_HERE, DontTraceHere — warmstate.py:484).
            if !self.meta.warm_state_ref().counter_would_fire(key_hash) {
                self.meta.warm_state_mut().counter_tick(key_hash);
                return None;
            }
            let green_key = GreenKey::new(green_values.to_vec());
            // Preserve resume_pc from back_edge_structured (guard
            // failure returns the guard's pc, not the loop header).
            return self.back_edge_structured(green_key, target_pc, state, env, pre_run);
        }

        let meta = self.meta.get_compiled_meta(key_hash)?;
        if !state.is_compatible(meta) {
            return None;
        }
        let meta = meta.clone();
        let descriptor = self.driver_descriptor_for(state, &meta);
        if !self.sync_before(state, &meta, descriptor.as_ref()) {
            return None;
        }
        let live_values = state.extract_live_values(&meta);
        if !Self::live_values_match_descriptor(descriptor.as_ref(), &live_values) {
            return None;
        }
        let live_values = self.extend_compiled_live_values(
            key_hash,
            state,
            &meta,
            descriptor.as_ref(),
            live_values,
        )?;
        pre_run();

        // RPython compile.py:205-207: register loop token with
        // quasi-immutable deps so the background invalidation thread
        // can force GUARD_NOT_INVALIDATED exits periodically.
        if let Some(token) = self.meta.get_loop_token(key_hash) {
            if let Ok(mut qmut) = self.epoch_qmut.lock() {
                qmut.register(&token.invalidation_flag());
            }
        }

        let mut live_values = live_values;
        loop {
            let result = self
                .meta
                .run_compiled_raw_detailed_with_values(key_hash, &live_values)?;
            let is_finish = result.is_finish;
            let fail_index = result.fail_index;
            let trace_id = result.trace_id;
            let result_meta = result.meta.clone();
            let typed_values = result.typed_values;
            let raw_values = result.values;
            let exit_layout = result.exit_layout;
            let descr_addr = result.descr_addr;

            if is_finish || fail_index == u32::MAX {
                state.restore_values(&result_meta, &typed_values);
                self.sync_after(state, &result_meta, descriptor.as_ref());
                // Re-enter compiled code if state is still compatible
                if let Some(meta) = self.meta.get_compiled_meta(key_hash) {
                    if state.is_compatible(meta) {
                        let meta = meta.clone();
                        let nd = self.driver_descriptor_for(state, &meta);
                        if self.sync_before(state, &meta, nd.as_ref()) {
                            let nl = state.extract_live_values(&meta);
                            if Self::live_values_match_descriptor(nd.as_ref(), &nl) {
                                if let Some(v) = self.extend_compiled_live_values(
                                    key_hash,
                                    state,
                                    &meta,
                                    nd.as_ref(),
                                    nl,
                                ) {
                                    live_values = v;
                                    continue;
                                }
                            }
                        }
                    }
                }
                return Some(target_pc);
            }

            // compile.py:701-716 handle_fail
            //
            // RPython's guard failure handling:
            //   if must_compile and not stack_almost_full:
            //       _trace_and_compile_from_bridge(...)
            //   else:
            //       resume_in_blackhole(...)
            //   assert 0, "unreachable"

            let (must_compile, owning_key) = self.meta.must_compile_with_values(
                key_hash,
                trace_id,
                fail_index,
                &raw_values,
                descr_addr,
            );
            // compile.py:702-703: must_compile() and not stack_almost_full().
            let should_bridge =
                must_compile && !majit_metainterp::MetaInterp::<S::Meta>::stack_almost_full();

            // Extract guard_resume_pc from fail_args.
            // compile.py:800-809: use owning_key (rd_loop_token).
            let num_inputs = self.meta.compiled_num_inputs(owning_key);
            let guard_resume_pc = if raw_values.len() > num_inputs {
                raw_values[raw_values.len() - 1] as usize
            } else {
                target_pc
            };

            if should_bridge {
                // compile.py:704-709: _trace_and_compile_from_bridge
                // Restore state for bridge tracing start point.
                let resume_pc = on_guard_failure(state, &result_meta, &raw_values, &exit_layout);
                let resume_pc = resume_pc.unwrap_or(guard_resume_pc);
                materialize_pending_fields(&exit_layout, &raw_values);
                self.sync_after(state, &result_meta, descriptor.as_ref());

                let bridge_ok = self.start_bridge_tracing(
                    owning_key, trace_id, fail_index, state, env, resume_pc, target_pc,
                );
                if crate::majit_log_enabled() {
                    eprintln!(
                        "[bridge] start_bridge_tracing key={} trace={} fail={} resume_pc={} ok={}",
                        key_hash, trace_id, fail_index, resume_pc, bridge_ok
                    );
                }
                return Some(resume_pc);
            }

            // compile.py:711 resume_in_blackhole
            // compile.py:853 `ResumeGuardDescr` storage — see the
            // companion site above.
            if let Some(storage) = exit_layout.storage.as_deref() {
                let rd_numb = storage.rd_numb.as_slice();
                let rd_consts_slice: &[Const] = storage.rd_consts();

                // Convert RdVirtualInfo → VirtualInfo for blackhole resume.
                let rd_virtuals_converted: Option<Vec<crate::resume::VirtualInfo>> = {
                    let count = raw_values.len() as i32;
                    Some(
                        storage
                            .rd_virtuals
                            .iter()
                            .map(|rd| {
                                crate::resume::rd_virtual_to_virtual_info(
                                    rd,
                                    rd_consts_slice,
                                    count,
                                )
                            })
                            .collect(),
                    )
                };
                let rd_virtuals_slice = rd_virtuals_converted.as_deref();

                // resume.py:1339 multi-frame state-field-JIT chain
                // reconstruction (mirrors the loop_jit guard-fail path).
                let jitcode_factory_ref = self.jitcode_factory.as_ref();
                let last_resolved: std::cell::RefCell<
                    Option<std::sync::Arc<majit_metainterp::JitCode>>,
                > = std::cell::RefCell::new(None);
                let resolve_jitcode =
                    |jitcode_index: i32, pc: i32| -> Option<crate::resume::ResolvedJitCode> {
                        let resolved_jitcode = if last_resolved.borrow().is_none() {
                            let factory = jitcode_factory_ref?;
                            let jitcode = factory(env, jitcode_index as usize, 0)?;
                            std::sync::Arc::new(jitcode)
                        } else {
                            let parent = last_resolved
                                .borrow()
                                .as_ref()
                                .expect("parent exists")
                                .clone();
                            parent
                                .exec
                                .descrs
                                .get(jitcode_index as usize)
                                .and_then(crate::jitcode::RuntimeBhDescr::as_jitcode)?
                                .clone()
                        };
                        *last_resolved.borrow_mut() = Some(resolved_jitcode.clone());
                        Some(crate::resume::ResolvedJitCode::new(
                            resolved_jitcode,
                            pc as usize,
                        ))
                    };

                let fallback_alloc = crate::resume::NullAllocator;
                let allocator: &dyn crate::resume::BlackholeAllocator = self
                    .blackhole_allocator
                    .as_deref()
                    .unwrap_or(&fallback_alloc);

                let mut bh_builder = crate::blackhole::BlackholeInterpBuilder::new();
                bh_builder.setup_cached_control_opcodes(
                    self.meta_interp().staticdata.op_live,
                    self.meta_interp().staticdata.op_catch_exception,
                    self.meta_interp().staticdata.op_rvmprof_code,
                );
                let all_liveness = self.meta_interp().staticdata.liveness_info.as_slice();
                let bh = crate::resume::blackhole_from_resumedata(
                    &mut bh_builder,
                    &resolve_jitcode,
                    rd_numb,
                    rd_consts_slice,
                    all_liveness,
                    &raw_values,
                    Some(exit_layout.exit_types.as_slice()),
                    rd_virtuals_slice,
                    Some(&storage.rd_pendingfields), // rd_guard_pendingfields
                    Some(&self.meta_interp().virtualref_info as &dyn crate::resume::VRefInfo),
                    None, // vinfo
                    None, // ginfo
                    allocator,
                );
                if let Some((bh, _vable_ptr)) = bh {
                    let exc =
                        crate::blackhole::BlackholeInterpreter::prepare_resume_from_failure(0);
                    let jit_exc = crate::blackhole::run_forever_with_portal(
                        &mut bh_builder,
                        bh,
                        exc,
                        self.portal_runner.as_ref().map(|r| {
                            r.as_ref()
                                as &dyn Fn(
                                    &crate::jitexc::JitException,
                                ) -> Result<
                                    (crate::blackhole::BhReturnType, i64),
                                    crate::jitexc::JitException,
                                >
                        }),
                    );
                    // compile.py:716 assert 0, "unreachable"
                    if crate::majit_log_enabled() {
                        eprintln!(
                            "[bh] run_back_edge_generic: run_forever completed with {:?}",
                            jit_exc
                        );
                    }
                    return Some(target_pc);
                }
            }

            // Legacy fallback: no rd_numb or jitcode resolution failed.
            let resume_pc = on_guard_failure(state, &result_meta, &raw_values, &exit_layout);
            let resume_pc = resume_pc.unwrap_or(target_pc);
            materialize_pending_fields(&exit_layout, &raw_values);
            self.sync_after(state, &result_meta, descriptor.as_ref());
            return Some(resume_pc);
        } // end loop { run_compiled ... }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resume::ReconstructedFrame;
    use majit_ir::{GcRef, OpCode, OpRef, Type, Value};

    #[derive(Default)]
    struct TypedRestoreState {
        live_values: Vec<i64>,
        restored_values: Vec<Value>,
        restore_called: bool,
    }

    #[derive(Default)]
    struct TypedInputState {
        raw_live_values: Vec<i64>,
        typed_live_values: Vec<Value>,
        restored_values: Vec<Value>,
        raw_restore_calls: usize,
        typed_restore_calls: usize,
    }

    #[derive(Default)]
    struct FrameMetadataState {
        seen_trace_id: Option<u64>,
        seen_header_pc: Option<u64>,
        seen_source_guard: Option<(u64, u32)>,
        restored_values: Vec<Value>,
    }

    #[derive(Default)]
    struct NonTraceableState;

    impl JitState for TypedRestoreState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.restore_called = true;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for TypedInputState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            self.raw_live_values.clone()
        }

        fn extract_live_values(&self, _meta: &Self::Meta) -> Vec<Value> {
            self.typed_live_values.clone()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {
            self.raw_restore_calls += 1;
        }

        fn restore_values(&mut self, _meta: &Self::Meta, values: &[Value]) {
            self.typed_restore_calls += 1;
            self.restored_values = values.to_vec();
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for FrameMetadataState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn restore_reconstructed_frame_values_with_metadata(
            &mut self,
            _meta: &Self::Meta,
            _frame_index: usize,
            _total_frames: usize,
            frame: &ReconstructedFrame,
            values: &[Value],
            _exception: &ExceptionState,
        ) -> bool {
            self.seen_trace_id = frame.trace_id;
            self.seen_header_pc = frame.header_pc;
            self.seen_source_guard = frame.source_guard;
            self.restored_values = values.to_vec();
            true
        }

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    impl JitState for NonTraceableState {
        type Meta = ();
        type Sym = ();
        type Env = ();

        fn can_trace(&self) -> bool {
            false
        }

        fn build_meta(&self, _header_pc: usize, _env: &Self::Env) -> Self::Meta {}

        fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
            Vec::new()
        }

        fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {}

        fn is_compatible(&self, _meta: &Self::Meta) -> bool {
            true
        }

        fn restore(&mut self, _meta: &Self::Meta, _values: &[i64]) {}

        fn collect_jump_args(_sym: &Self::Sym) -> Vec<OpRef> {
            Vec::new()
        }

        fn validate_close(_sym: &Self::Sym, _meta: &Self::Meta) -> bool {
            true
        }
    }

    #[test]
    fn jit_merge_point_only_drives_active_trace() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        let key = 123u64;
        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };

        assert!(
            driver
                .jit_merge_point_keyed(
                    key,
                    7,
                    &mut state,
                    &(),
                    || {},
                    |_ctx, _sym| { TraceAction::Continue }
                )
                .is_none()
        );
        assert!(
            !driver.is_tracing(),
            "jit_merge_point must not warm up or start tracing by itself"
        );

        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(!driver.is_tracing(), "first back-edge should only warm up");

        assert!(
            driver
                .back_edge_or_run_compiled_keyed(key, 7, &mut state, &(), || {})
                .is_none()
        );
        assert!(
            driver.is_tracing(),
            "back-edge should start tracing once hot"
        );
    }

    #[test]
    fn blackhole_jump_reports_via_blackhole_even_with_typed_restore_values() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        let key = 9u64;

        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));

        // Use input arg as guard condition so optimizer cannot constant-fold.
        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let i0 = OpRef(0); // input arg
            let g = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0]);
            ctx.set_fail_args(g, &[i0]);
        };
        driver.meta.compile_loop(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedRestoreState {
            live_values: vec![1],
            ..Default::default()
        };
        match driver.run_compiled_with_blackhole_fallback_keyed(key, &mut state, || {}) {
            crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole } => {
                assert!(via_blackhole);
            }
            crate::pyjitpl::DriverRunOutcome::GuardFailure {
                restored,
                via_blackhole,
            } => {
                assert!(restored);
                assert!(via_blackhole);
            }
            other => panic!("expected Jump or GuardFailure outcome, got {other:?}"),
        }
    }

    #[test]
    fn run_compiled_detailed_keyed_uses_typed_live_inputs() {
        let mut driver = JitDriver::<TypedInputState>::new(2);
        let key = 11u64;
        // Input 0 is Int (used as GuardFalse condition); inputs 1/2 are
        // Ref/Float so the typed restore path still exercises mixed types.
        // GUARD_TRUE/FALSE in RPython are typed 'i' (resoperation.py
        // arglist_types) — the guard condition must be Int.
        let typed_live_values = vec![Value::Int(1), Value::Ref(GcRef(0x1234)), Value::Float(3.5)];

        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));

        // Input 0 is Int(1) — nonzero so GuardFalse fails at runtime.
        // Optimizer cannot constant-fold an input arg.
        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let i0 = OpRef(0);
            let g = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[OpRef(0), OpRef(1), OpRef(2)]);
            ctx.set_fail_args(g, &[OpRef(0), OpRef(1), OpRef(2)]);
        };
        driver
            .meta
            .compile_loop(&[OpRef(0), OpRef(1), OpRef(2)], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedInputState {
            raw_live_values: vec![1, 0, 0],
            typed_live_values: typed_live_values.clone(),
            ..Default::default()
        };
        match driver.run_compiled_detailed_keyed(key, &mut state, || {}) {
            DetailedDriverRunOutcome::Jump { via_blackhole } => {
                assert!(!via_blackhole);
            }
            other => panic!("expected Jump outcome, got {other:?}"),
        }
        assert_eq!(state.restored_values, typed_live_values);
        assert_eq!(state.raw_restore_calls, 0);
        assert_eq!(state.typed_restore_calls, 1);
    }

    #[test]
    fn blackhole_fallback_uses_typed_live_inputs() {
        let mut driver = JitDriver::<TypedInputState>::new(2);
        let key = 12u64;
        // Input 0 is Int (used as GuardFalse condition); inputs 1/2 are
        // Ref/Float so the typed restore path still exercises mixed types.
        // GUARD_TRUE/FALSE in RPython are typed 'i' (resoperation.py
        // arglist_types) — the guard condition must be Int.
        let typed_live_values = vec![Value::Int(1), Value::Ref(GcRef(0x5678)), Value::Float(6.25)];

        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver
                .meta
                .on_back_edge_typed(key, (0, 0), None, None, &typed_live_values),
            BackEdgeAction::StartedTracing
        ));

        // Input 0 is Int(1) — nonzero so GuardFalse fails at runtime.
        {
            let ctx = driver.meta.trace_ctx().expect("trace ctx should exist");
            let i0 = OpRef(0);
            let g = ctx.record_guard(OpCode::GuardFalse, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[OpRef(0), OpRef(1), OpRef(2)]);
            ctx.set_fail_args(g, &[OpRef(0), OpRef(1), OpRef(2)]);
        };
        driver
            .meta
            .compile_loop(&[OpRef(0), OpRef(1), OpRef(2)], ());
        assert!(driver.has_compiled_loop(key));

        let mut state = TypedInputState {
            raw_live_values: vec![1, 0, 0],
            typed_live_values: typed_live_values.clone(),
            ..Default::default()
        };
        match driver.run_compiled_with_blackhole_fallback_keyed(key, &mut state, || {}) {
            crate::pyjitpl::DriverRunOutcome::Jump { via_blackhole } => {
                assert!(via_blackhole);
            }
            crate::pyjitpl::DriverRunOutcome::GuardFailure {
                restored,
                via_blackhole,
            } => {
                // Guard failure is restored via blackhole — acceptable.
                assert!(restored);
                assert!(via_blackhole);
            }
            other => panic!("expected Jump or GuardFailure outcome, got {other:?}"),
        }
        assert_eq!(state.restored_values, typed_live_values);
    }

    #[test]
    fn test_threshold_1_fires_on_first_tick() {
        // counter.py: compute_threshold(1) = 1.0/(1-0.001) ≈ 1.001
        // First tick adds ≈1.001 to 0.0, reaching ≥1.0 immediately.
        let mut driver = JitDriver::<TypedRestoreState>::new(1);
        let key = 500u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::StartedTracing
        ));
    }

    #[test]
    fn test_set_param_threshold() {
        let mut driver = JitDriver::<TypedRestoreState>::new(10);
        // Initially threshold is 10 — not hot after 2 ticks.
        let key = 100u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[]),
            BackEdgeAction::Interpret
        ));

        // Lower threshold to 2 via set_param — next tick should start tracing.
        // (threshold=1 fires on first tick: increment=1/0.999≈1.001 ≥ 1.0)
        driver.set_param("threshold", 2);
        let key2 = 200u64;
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key2, &[]),
            BackEdgeAction::StartedTracing
        ));
    }

    #[test]
    fn test_get_stats_after_compile() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        let key = 300u64;

        // Stats should be zero initially.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.loops_aborted, 0);

        // Warm up and start tracing. Pass a single live value so that
        // setup_tracing (pyjitpl.rs:1186) registers one inputarg via
        // recorder.record_input_arg — this gives the trace a valid
        // `OpRef(0)` inputarg that the GuardTrue below references.
        // Without the inputarg, TraceIterator (opencoder.py:257-266)
        // correctly rejects a TAGBOX reference to an unseeded cache
        // slot (`_get(i)` asserts `res is not None`).
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0i64]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0i64]),
            BackEdgeAction::StartedTracing
        ));

        // Record minimal trace and compile.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0);
            ctx.const_int(42);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[i0]);
            ctx.set_fail_args(g, &[i0]);
        }
        driver.meta.compile_loop(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(key));

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.loops_aborted, 0);
        assert_eq!(stats.bridges_compiled, 0);
    }

    #[test]
    fn test_set_param_unknown_ignored() {
        let mut driver = JitDriver::<TypedRestoreState>::new(5);
        // Unknown params should not panic or cause any side effects.
        driver.set_param("nonexistent_param", 999);
        driver.set_param("", 0);
        driver.set_param("enable_opts", 1);

        // Driver should still work normally after unknown params.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
    }

    #[test]
    fn test_new_auto_registers_virtualizable_info_from_state() {
        #[derive(Default)]
        struct AutoVableState;

        impl JitState for AutoVableState {
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
            fn __build_virtualizable_info()
            -> Option<std::sync::Arc<crate::virtualizable::VirtualizableInfo>> {
                let mut info = crate::virtualizable::VirtualizableInfo::new(24);
                info.add_field("pc", Type::Int, 8);
                info.add_array_field(
                    "locals_w",
                    Type::Ref,
                    16,
                    0,
                    0,
                    majit_ir::make_array_descr(0, 8, Type::Ref),
                );
                Some(info.finalize_arc(majit_ir::descr::make_size_descr(64)))
            }
        }

        let driver = JitDriver::<AutoVableState>::new(5);
        let info = driver
            .meta_interp()
            .virtualizable_info()
            .expect("driver should auto-register __build_virtualizable_info");
        assert_eq!(info.token_offset, 24);
        assert_eq!(info.static_fields.len(), 1);
        assert_eq!(info.array_fields.len(), 1);
        assert_eq!(info.static_fields[0].name, "pc");
        assert_eq!(info.array_fields[0].name, "locals_w");
    }

    // ── Event-driven JitHookInterface parity tests ──
    // These exercise the full compilation pipeline (tracing → optimize → Cranelift)
    // and verify hooks fire with correct metadata, unlike wiring-only tests that
    // call hook closures directly.

    #[test]
    fn test_hook_on_compile_fires_through_real_pipeline() {
        use std::sync::{Arc, Mutex};

        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        let compile_events: Arc<Mutex<Vec<(u64, usize, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let events = compile_events.clone();
        driver.set_on_compile_loop(move |green_key, ops_before, ops_after| {
            events
                .lock()
                .unwrap()
                .push((green_key, ops_before, ops_after));
        });

        let key = 42u64;

        // Warm up: first back_edge increments counter, second starts tracing.
        // Pass one live value so OpRef(0) = i0 is available.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        // Record a real trace: IntAdd + GuardTrue(input) + JUMP(sum)
        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0); // input arg from on_back_edge
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum]);
            ctx.set_fail_args(g, &[sum]);
            sum
        };

        // Close: JUMP args must match body inputargs after preamble peeling.
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        let events = compile_events.lock().unwrap();
        assert_eq!(events.len(), 1, "on_compile_loop should fire exactly once");
        assert_eq!(events[0].0, key, "green_key should match");
        assert!(events[0].1 > 0, "num_ops_before should be positive");
        assert!(events[0].2 > 0, "num_ops_after should be positive");
    }

    #[test]
    fn test_hook_get_stats_matches_real_compile_count() {
        let mut driver = JitDriver::<TypedRestoreState>::new(2);

        // Initially zero.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 0);
        assert_eq!(stats.bridges_compiled, 0);

        // Compile two distinct loops.
        for key in [100u64, 200u64] {
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::Interpret
            ));
            assert!(matches!(
                driver.meta.on_back_edge(key, &[0]),
                BackEdgeAction::StartedTracing
            ));
            {
                let ctx = driver.meta.trace_ctx().expect("should be tracing");
                let i0 = OpRef(0);
                let c1 = ctx.const_int(1);
                let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
                let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
                ctx.capture_snapshot_for_last_guard(&[sum]);
                ctx.set_fail_args(g, &[sum]);
            }
            driver.meta.compile_loop(&[OpRef(0)], ());
            assert!(driver.has_compiled_loop(key));
        }

        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 2);
        assert_eq!(stats.loops_aborted, 0);
    }

    #[test]
    fn test_hook_on_compile_bridge_fires_through_real_pipeline() {
        use std::sync::{Arc, Mutex};

        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        let bridge_events: Arc<Mutex<Vec<(u64, u32, usize)>>> = Arc::new(Mutex::new(Vec::new()));
        let ev = bridge_events.clone();
        driver.meta.set_on_compile_bridge(move |gk, fi, nops| {
            ev.lock().unwrap().push((gk, fi, nops));
        });

        let key = 50u64;

        // Step 1: Compile a loop with a guard on a non-constant condition.
        // The guard must survive optimization, so use an input arg as condition.
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[1]),
            BackEdgeAction::StartedTracing
        ));
        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0); // input arg (non-constant = won't be folded)
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum]);
            ctx.set_fail_args(g, &[sum]);
            sum
        };
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        // Identify the fail_index assigned to the first guard in the optimized trace.
        // build_guard_metadata assigns fail_index sequentially starting from 0.
        let fail_index = 0u32;

        // Step 2: Start bridge tracing via start_retrace (simulates guard failure path).
        assert!(driver.meta.start_retrace(key, fail_index, &[0]));

        // Step 3: Record a bridge trace and compile it via close_bridge.
        {
            let ctx = driver.meta.trace_ctx().expect("should be tracing bridge");
            let i0 = OpRef(0); // bridge input from start_retrace
            let c2 = ctx.const_int(2);
            ctx.record_op(OpCode::IntAdd, &[i0, c2]);
        }
        let trace_id = 0u64; // will be normalized to root_trace_id
        let result = driver
            .meta
            .close_bridge(key, trace_id, fail_index, &[OpRef(0)]);
        assert_eq!(
            result,
            crate::pyjitpl::BridgeCompileResult::Compiled,
            "bridge should compile successfully"
        );

        let events = bridge_events.lock().unwrap();
        assert_eq!(
            events.len(),
            1,
            "on_compile_bridge should fire exactly once"
        );
        assert_eq!(events[0].0, key, "bridge green_key should match");
        assert_eq!(events[0].1, fail_index, "bridge fail_index should match");
        assert!(events[0].2 > 0, "bridge num_ops should be positive");

        // Stats should reflect 1 loop + 1 bridge.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
        assert_eq!(stats.bridges_compiled, 1);
    }

    #[test]
    fn test_start_bridge_tracing_skips_non_traceable_state() {
        let mut driver = JitDriver::<NonTraceableState>::new(2);
        let green_key = 404u64;
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(green_key, &[0]),
            BackEdgeAction::StartedTracing
        ));
        {
            // Use input arg so optimizer cannot fold. Input is 0 (falsy)
            // at runtime → GuardTrue fails.
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[]);
            ctx.set_fail_args(g, &[]);
        }
        driver.meta.compile_loop(&[OpRef(0)], ());
        assert!(driver.has_compiled_loop(green_key));
        let failure = driver
            .meta
            .run_compiled_detailed(green_key, &[0])
            .expect("guard should fail");
        let trace_id = failure.trace_id;
        let fail_index = failure.fail_index;

        let started = driver.start_bridge_tracing(
            green_key,
            trace_id,
            fail_index,
            &mut NonTraceableState,
            &(),
            0,
            0,
        );
        assert!(!started);
        assert!(!driver.meta.is_tracing());
    }

    // ── Multi-entry point lifecycle tests ──
    // Parity with warmspot.py multi-driver entry semantics: multiple functions
    // can share the same JitDriver and compiled loops via register_entry_point.

    #[test]
    fn test_multi_entry_point_registration() {
        let mut driver = JitDriver::<TypedRestoreState>::new(1);

        // Initially no entry points.
        assert!(driver.entry_points().is_empty());

        // Register two entry points with different green key schemas.
        driver.register_entry_point("loop_main", &[Type::Int]);
        driver.register_entry_point("loop_helper", &[Type::Int, Type::Ref]);

        assert_eq!(driver.entry_points().len(), 2);

        // Verify first entry point.
        let ep0 = driver
            .find_entry_point("loop_main")
            .expect("should find loop_main");
        assert_eq!(ep0.name, "loop_main");
        assert_eq!(ep0.schema, vec![GreenType::Int]);

        // Verify second entry point.
        let ep1 = driver
            .find_entry_point("loop_helper")
            .expect("should find loop_helper");
        assert_eq!(ep1.name, "loop_helper");
        assert_eq!(ep1.schema, vec![GreenType::Int, GreenType::Ref]);

        // Non-existent entry point returns None.
        assert!(driver.find_entry_point("nonexistent").is_none());
    }

    #[test]
    fn test_multi_entry_points_share_compiled_loops() {
        // Two entry points register different functions on the same driver.
        // A loop compiled from one entry point is visible to back_edge calls
        // from any entry point, since they share the same MetaInterp and
        // compiled loop table (keyed by green_key, not by entry point name).
        let mut driver = JitDriver::<TypedRestoreState>::new(2);
        driver.register_entry_point("func_a", &[Type::Int]);
        driver.register_entry_point("func_b", &[Type::Int]);

        // Both entry points are registered.
        assert_eq!(driver.entry_points().len(), 2);

        // Compile a loop from "func_a"'s perspective using green_key=10.
        let key = 10u64;
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::Interpret
        ));
        assert!(matches!(
            driver.meta.on_back_edge(key, &[0]),
            BackEdgeAction::StartedTracing
        ));

        let sum = {
            let ctx = driver.meta.trace_ctx().expect("should be tracing");
            let i0 = OpRef(0);
            let c1 = ctx.const_int(1);
            let sum = ctx.record_op(OpCode::IntAdd, &[i0, c1]);
            let g = ctx.record_guard(OpCode::GuardTrue, &[i0], 0);
            ctx.capture_snapshot_for_last_guard(&[sum]);
            ctx.set_fail_args(g, &[sum]);
            sum
        };
        driver.meta.compile_loop(&[sum], ());
        assert!(driver.has_compiled_loop(key));

        // "func_b" can see the compiled loop via the same green_key.
        // The compiled loop is shared — it doesn't matter which entry point
        // triggered the compilation.
        assert!(
            driver.has_compiled_loop(key),
            "compiled loop should be visible from any entry point"
        );

        // A different green key from "func_b" does not see the loop.
        let other_key = 20u64;
        assert!(
            !driver.has_compiled_loop(other_key),
            "different green_key should not match"
        );

        // The driver stats reflect exactly one compiled loop.
        let stats = driver.get_stats();
        assert_eq!(stats.loops_compiled, 1);
    }

    #[test]
    fn jitdriver_install_canonical_liveness_publishes_macro_emit_composition() {
        // Mirrors the body of the macro-emitted
        // `__JitMeta::install_canonical_liveness`
        // (`majit-macros/src/jit_interp/codegen_state.rs`):
        //   1. Build a fresh `Assembler`.
        //   2. Encode the canonical
        //      `live_slots_for_state_field_jit(num_scalars, &lens, num_virt)`
        //      triple via `_encode_liveness`.
        //   3. Forward to `JitDriver::install_canonical_liveness`.
        //
        // RPython parity: `warmspot.py:281-289` →
        // `pyjitpl.py:2264 self.liveness_info = "".join(asm.all_liveness)`
        // — the metainterp-side test
        // (`pyjitpl/mod.rs::metainterp_install_canonical_liveness_publishes_asm_bytes`)
        // exercises the inner layer; this driver-level test guards the
        // pass-through wrapper the macro actually targets.
        use majit_translate::jit_codewriter::assembler::Assembler;

        // Build the canonical liveness exactly the way the macro expansion
        // does (orth-6 helper + orth-2 _encode_liveness + insns
        // registration mirroring `assembler.py:222 self.insns[key] = opnum`).
        let (live_i, live_r, live_f) = crate::live_slots_for_state_field_jit(2, &[1], 0);
        let mut asm = Assembler::new();
        let mut scratch = Vec::<u8>::new();
        asm._encode_liveness(&live_i, &live_r, &live_f, &mut scratch);
        asm.register_insn("live/", crate::jitcode::BC_LIVE);
        asm.register_insn("catch_exception/L", crate::jitcode::BC_CATCH_EXCEPTION);
        asm.register_insn("rvmprof_code/ii", crate::jitcode::BC_RVMPROF_CODE);
        asm.register_insn("int_return/i", crate::jitcode::BC_INT_RETURN);
        asm.register_insn("ref_return/r", crate::jitcode::BC_REF_RETURN);
        asm.register_insn("float_return/f", crate::jitcode::BC_FLOAT_RETURN);
        asm.register_insn("void_return/", crate::jitcode::BC_VOID_RETURN);
        let expected = asm.all_liveness().to_vec();

        let mut driver = JitDriver::<TypedRestoreState>::new(0);
        driver.install_canonical_liveness(&asm);

        assert_eq!(driver.meta.staticdata.liveness_info, expected);
        // `setup_insns(asm.insns())` (`pyjitpl.py:2236-2243`) resolves
        // `op_live` from the `live/` entry the macro registers above
        // — same data flow as RPython where `make_jitcodes()`
        // populates `asm.insns` before `finish_setup(codewriter)`.
        assert_eq!(
            driver.meta.staticdata.op_live,
            crate::jitcode::BC_LIVE as i32,
        );
    }

    /// `warmspot.py:663-665` `jd._green_args_spec = [...]` /
    /// `jd.num_green_args = len(jd._green_args_spec)` — spec length
    /// must match the number of green vars on the jd.
    #[test]
    fn green_args_spec_length_matches_num_green_args() {
        let sd = JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Ref)],
            vec![("r0", Type::Int)],
        );
        let spec = sd.green_args_spec();
        assert_eq!(spec.len(), sd.num_green_args());
        assert_eq!(spec, vec![GreenType::Int, GreenType::Ref]);
    }

    /// `warmstate.py:535-553` `make_unwrap_greenkey().unwrap_greenkey(greenkey)`
    /// — the resulting `GreenKey` carries values + types in declaration
    /// order so `comparekey` / `get_uhash` dispatch is type-correct.
    #[test]
    fn unwrap_greenkey_uses_green_args_spec_in_declaration_order() {
        let sd = JitDriverStaticData::new(
            vec![("g_int", Type::Int), ("g_ref", Type::Ref)],
            vec![("r_int", Type::Int)],
        );
        let key = sd.unwrap_greenkey(&[42, 0xdeadbeef]);
        assert_eq!(key.values, vec![42, 0xdeadbeef]);
        assert_eq!(key.types, vec![GreenType::Int, GreenType::Ref]);
    }

    /// `warmstate.py:535-553` — upstream's `unrolling_iterable` loop
    /// implicitly asserts `len(greenkey) == len(_green_args_spec)`. The
    /// pyre helper makes that assertion explicit.
    #[test]
    #[should_panic(expected = "expected 2 green values per _green_args_spec, got 1")]
    fn unwrap_greenkey_panics_when_value_count_mismatches_spec() {
        let sd = JitDriverStaticData::new(vec![("g0", Type::Int), ("g1", Type::Int)], vec![]);
        let _ = sd.unwrap_greenkey(&[42]);
    }

    /// `warmspot.py:664` `red_args_types = [history.getkind(TYPE)[0] ...]`.
    /// The IR-type bridge must preserve declaration order and each char
    /// must round-trip to the matching `Type` variant.
    #[test]
    fn red_arg_types_as_ir_types_maps_chars_in_declaration_order() {
        let sd = JitDriverStaticData::new(
            vec![],
            vec![("r0", Type::Int), ("r1", Type::Ref), ("r2", Type::Float)],
        );
        let mapped = sd.red_arg_types_as_ir_types();
        assert_eq!(mapped, vec![Type::Int, Type::Ref, Type::Float]);
        assert_eq!(mapped.len(), sd.num_red_args());
    }

    /// `warmspot.py:664` only emits 'i' / 'r' / 'f' / 'v'; an unknown
    /// char indicates corrupted state and must surface immediately.
    #[test]
    #[should_panic(expected = "unknown history kind char")]
    fn red_arg_types_as_ir_types_panics_on_unknown_char() {
        let mut sd = JitDriverStaticData::new(vec![], vec![("r0", Type::Int)]);
        sd.red_args_types = vec!['x'];
        let _ = sd.red_arg_types_as_ir_types();
    }

    /// `warmstate.py:574-582` `JitCell.comparekey(*greenargs2)` — two
    /// `GreenKey`s built from the same spec + values must compare equal,
    /// and a value mismatch in any position must compare unequal.
    #[test]
    fn unwrap_greenkey_round_trips_through_comparekey_matches() {
        use crate::warmstate::BaseJitCell;

        let sd = JitDriverStaticData::new(vec![("g0", Type::Int), ("g1", Type::Ref)], vec![]);
        let key_a = sd.unwrap_greenkey(&[7, 0x1000]);
        let key_b = sd.unwrap_greenkey(&[7, 0x1000]);
        let key_c = sd.unwrap_greenkey(&[7, 0x2000]);

        let mut cell = BaseJitCell::new();
        cell.comparekey = Some(key_a);
        assert!(cell.comparekey_matches(&key_b));
        assert!(!cell.comparekey_matches(&key_c));
    }
}
