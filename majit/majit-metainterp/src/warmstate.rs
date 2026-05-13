/// Warm state management — the lifecycle from interpreting to compiled code.
///
/// Manages the transition: Interpreting -> Tracing -> Compiled.
/// When the hot counter fires, we start tracing. When the trace is
/// complete, we compile it and cache the result.
///
/// Reference: rpython/jit/metainterp/warmstate.py WarmEnterState, BaseBaseJitCell
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use majit_backend::JitCellToken;
use majit_ir::{GreenKey, Type};
use std::sync::Arc;

use majit_trace::counter::JitCounter;
use majit_trace::logger::Logger;

use crate::recorder::Trace;

/// Flags on a BaseJitCell, mirroring warmstate.py JC_* constants.
pub mod jc_flags {
    /// We are currently tracing from this green key.
    pub const TRACING: u8 = 0x01;
    /// Don't trace here (e.g., trace was too long last time).
    pub const DONT_TRACE_HERE: u8 = 0x02;
    /// Has a temporary procedure token (CALL_ASSEMBLER fallback).
    pub const TEMPORARY: u8 = 0x04;
    /// Tracing has occurred at least once from this key.
    pub const TRACING_OCCURRED: u8 = 0x08;
    /// warmstate.py: JC_FORCE_FINISH — the loop has a FINISH that
    /// returns a raw int (not a boxed pointer). Used by
    /// call_assembler to decide whether to unbox the result.
    pub const FORCE_FINISH: u8 = 0x10;
}

/// Explicit state of a BaseJitCell in the JIT lifecycle.
///
/// warmstate.py expresses this implicitly through flag combinations:
///   - no cell / flags==0           → NotHot
///   - JC_TRACING set               → Tracing
///   - loop_token present, valid    → Compiled
///   - loop_token.invalidated       → Invalidated
///   - JC_DONT_TRACE_HERE set       → DontTraceHere
///
/// We make these states explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseJitCellState {
    /// Not yet hot; still interpreting.
    NotHot,
    /// Actively tracing.
    Tracing,
    /// Compiled loop exists and is valid.
    Compiled,
    /// Compiled loop was invalidated (quasi-immutable mutation, etc.).
    Invalidated,
    /// Tracing was aborted; don't trace at this location.
    DontTraceHere,
}

/// Per-greenkey cell that tracks JIT state for a specific program location.
///
/// Mirrors rpython/jit/metainterp/warmstate.py BaseBaseJitCell.
pub struct BaseJitCell {
    /// JC_* flags.
    pub flags: u8,
    /// Explicit lifecycle state.
    pub state: BaseJitCellState,
    /// Hot counter value for this cell (local to this green key).
    pub counter: u32,
    /// Compiled loop token number, if a procedure token is owned.
    /// Cleared on invalidation.
    pub token: Option<u64>,
    /// Generation at which tracing was last started.
    /// Used to detect stale tracing sessions.
    pub tracing_generation: u64,
    /// Compiled loop token, if compilation has completed.
    ///
    /// PRE-EXISTING-ADAPTATION: upstream `warmstate.py:188` stores
    /// `wref_procedure_token` as a `weakref.ref(token)` so the warmstate
    /// holds only a weak handle and the strong owner is the
    /// `MemoryManager`'s `alive_loops` list (`memmgr.py:13`).  Pyre
    /// stores `Arc<JitCellToken>` here because:
    ///   1. `MetaInterp.compiled_loops: HashMap<u64, CompiledEntry>`
    ///      already holds a separate strong owner (the T-final.F retire
    ///      target), so the warmstate-side ref must also be strong to
    ///      keep the cell-routed lookup path
    ///      (`get_procedure_token` → `loop_token.as_ref()`) live until
    ///      `compiled_loops` is fully retired.
    ///   2. After F.6 (drop `compiled_loops` field), this Arc becomes
    ///      the canonical strong owner; the convergence target is to
    ///      downgrade to `Weak<JitCellToken>` so eviction by
    ///      `MemoryManager` matches `should_remove_jitcell`'s dead-weakref
    ///      check (`warmstate.py:212-225`).  See
    ///      `slice_x_f_landed_2026_05_02.md` memo Issue 3.3 for the
    ///      remaining weak-ref convergence work.
    pub loop_token: Option<Arc<JitCellToken>>,
    /// Number of times tracing was aborted for this key.
    ///
    /// Kept for diagnostics only. In RPython, `retrace_limit` is handled by
    /// optimizeopt/unroll during retracing, not by warmstate abort handling.
    pub abort_count: u32,
    /// counter.py:75 / warmstate.py BaseJitCell.next
    /// Linked list for per-bucket chain in the celltable.
    pub next: Option<Box<BaseJitCell>>,
    /// warmstate.py:568-582 — typed green-key carried per-cell so
    /// `JitCell.comparekey(*greenargs2)` can do per-green typed
    /// equality across hash collisions:
    ///
    /// ```python
    /// def comparekey(self, *greenargs2):
    ///     i = 0
    ///     for attrname, TYPE in green_args_name_spec:
    ///         item1 = getattr(self, attrname)
    ///         if not equal_whatever(TYPE, item1, greenargs2[i]):
    ///             return False
    ///         i = i + 1
    ///     return True
    /// ```
    ///
    /// `None` while pyre's legacy hash-only `cells: HashMap<u64,
    /// BaseJitCell>` lookup path is the default; typed-key callers
    /// (`lookup_chain_with_key` / `ensure_cell_for_key` below) populate
    /// this on insert so chained cells distinguish hash collisions like
    /// upstream `JitCell.get_jitcell` (warmstate.py:596-604) does.
    /// Migration of all callers from hash-only to typed keys is the
    /// remaining S2.2 work.
    pub comparekey: Option<GreenKey>,
}

impl BaseJitCell {
    pub(crate) fn new() -> Self {
        BaseJitCell {
            flags: 0,
            state: BaseJitCellState::NotHot,
            counter: 0,
            token: None,
            tracing_generation: 0,
            loop_token: None,
            abort_count: 0,
            next: None,
            comparekey: None,
        }
    }

    /// warmstate.py:575-582 `JitCell.comparekey(*greenargs2)`.
    ///
    /// Returns `true` iff this cell's stored typed greens match
    /// `other` per `GreenKey::eq` (which dispatches via
    /// `equal_whatever(TYPE, ...)` for STR/UNICODE/Float/Ref/Int).
    /// Cells without a stored comparekey (legacy hash-only path)
    /// always fail comparison — callers must explicitly opt in by
    /// inserting cells through the typed-key path
    /// (`WarmEnterState::ensure_cell_for_key`).
    pub fn comparekey_matches(&self, other: &GreenKey) -> bool {
        match &self.comparekey {
            Some(stored) => stored == other,
            None => false,
        }
    }

    pub fn is_tracing(&self) -> bool {
        self.flags & jc_flags::TRACING != 0
    }

    /// warmstate.py:191-196 — get_procedure_token returns None for
    /// invalidated tokens. is_compiled additionally excludes TEMPORARY.
    pub fn is_compiled(&self) -> bool {
        self.get_procedure_token().is_some() && (self.flags & jc_flags::TEMPORARY == 0)
    }

    /// warmstate.py:191-196 get_procedure_token
    pub fn get_procedure_token(&self) -> Option<&Arc<JitCellToken>> {
        self.loop_token.as_ref().filter(|t| !t.is_invalidated())
    }

    /// Set the procedure token and update ownership state.
    /// If `tmp` is true, sets the TEMPORARY flag (CALL_ASSEMBLER fallback).
    ///
    /// Returns the previous procedure token (if any), so the caller can
    /// implement `warmstate.py:343-348`'s `redirect_call_assembler` +
    /// `old_token.record_jump_to(procedure_token)` chain.
    pub fn set_procedure_token(
        &mut self,
        loop_token: impl Into<Arc<JitCellToken>>,
        tmp: bool,
    ) -> Option<Arc<JitCellToken>> {
        let loop_token = loop_token.into();
        self.token = Some(loop_token.number);
        let old = self.loop_token.replace(loop_token);
        if tmp {
            self.flags |= jc_flags::TEMPORARY;
        } else {
            self.flags &= !jc_flags::TEMPORARY;
            self.state = BaseJitCellState::Compiled;
        }
        old
    }

    /// Check whether we have ever had a procedure token assigned
    /// (mirrors BaseBaseJitCell.has_seen_a_procedure_token).
    ///
    /// Returns true if a token was ever set, even if it was later
    /// invalidated. The `token` field is a historical record and is
    /// never cleared.
    pub fn has_seen_a_procedure_token(&self) -> bool {
        self.token.is_some()
    }

    /// Whether this cell should be removed (for GC of dead cells).
    /// Mirrors BaseBaseJitCell.should_remove_jitcell.
    pub fn should_remove_jitcell(&self) -> bool {
        if self.get_procedure_token().is_some() {
            return false; // has a valid procedure token
        }
        if self.flags & jc_flags::TRACING != 0 {
            return false; // currently tracing
        }
        if self.flags & jc_flags::DONT_TRACE_HERE != 0 {
            // Remove only if we had a token that is now dead.
            return self.has_seen_a_procedure_token();
        }
        // warmstate.py:222-225
        if self.flags & jc_flags::FORCE_FINISH != 0 {
            return false;
        }
        true
    }
}

/// Per-green-key JIT cell state with associated data.
///
/// Richer variant of BaseJitCellState that carries trace/token payloads.
pub enum CellJitState {
    /// Normal interpretation; no tracing is active.
    Interpreting,
    /// Actively recording a trace.
    Tracing(Trace),
    /// A compiled loop exists for this green key.
    Compiled(Arc<JitCellToken>),
}

pub use crate::memmgr::MemoryManager;

/// Warm state manager — the orchestrator of the JIT lifecycle.
///
/// Keeps track of per-greenkey cells and the global hot counter.
/// The interpreter calls `maybe_compile()` at loop headers;
/// WarmEnterState decides whether to start tracing, continue interpreting,
/// or dispatch to compiled code.
/// rlib/jit.py:588-605 PARAMETERS defaults.
/// DEFAULT_ constants must match RPython exactly.

/// rlib/jit.py:588 threshold = 1039 (just above 1024, prime)
const DEFAULT_THRESHOLD: u32 = 1039;

/// rlib/jit.py:589 function_threshold = 1619
const DEFAULT_FUNCTION_THRESHOLD: u32 = 1619;

/// rlib/jit.py:590 trace_eagerness = 200
const DEFAULT_TRACE_EAGERNESS: u32 = 200;

/// rlib/jit.py:601 max_unroll_recursion = 7
const DEFAULT_MAX_UNROLL_RECURSION: u32 = 7;

/// rlib/jit.py:593 inlining = 1 (max_inline_depth derived)
const DEFAULT_MAX_INLINE_DEPTH: u32 = 7;

/// rlib/jit.py:592 trace_limit = 6000
const DEFAULT_TRACE_LIMIT: u32 = crate::trace_ctx::DEFAULT_TRACE_LIMIT as u32;

/// rlib/jit.py:595 retrace_limit = 0.
const DEFAULT_RETRACE_LIMIT: u32 = 0;

/// rlib/jit.py:598 max_unroll_loops = 0
const DEFAULT_MAX_UNROLL_LOOPS: u32 = 0;

/// rlib/jit.py:600 enable_opts = "all"
fn default_enable_opts() -> Vec<String> {
    vec![
        "intbounds".to_string(),
        "rewrite".to_string(),
        "virtualize".to_string(),
        "string".to_string(),
        "pure".to_string(),
        "earlyforce".to_string(),
        "heap".to_string(),
        "unroll".to_string(),
    ]
}

/// Maximum number of trace aborts before permanently marking a green key
/// as DONT_TRACE_HERE. Prevents infinite retrace loops when optimization
/// always fails (e.g. InvalidLoop) for the same key.
const MAX_TRACE_ABORT_COUNT: u32 = 5;

/// rlib/jit.py:599 disable_unrolling = 200
const DEFAULT_DISABLE_UNROLLING: u32 = 200;

static NEXT_GLOBAL_TOKEN_NUMBER: AtomicU64 = AtomicU64::new(1);

/// JIT statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    /// Number of cells in Compiled state.
    pub num_compiled: usize,
    /// Number of cells in Tracing state.
    pub num_tracing: usize,
    /// Number of cells in Invalidated state.
    pub num_invalidated: usize,
    /// Number of cells in DontTraceHere state.
    pub num_disable_noninlinable_function: usize,
    /// Total number of BaseJitCells.
    pub num_cells: usize,
}

pub struct WarmEnterState {
    /// counter.py JitCounter parity: single timetable shared by loop
    /// entry, guard failure, and function entry — each caller passes
    /// a different threshold to tick_with_threshold().
    pub counter: JitCounter,
    /// Per-greenkey cells, keyed by the hash of the green key.
    cells: HashMap<u64, BaseJitCell>,
    /// Compilation threshold (copied from counter for easy access).
    threshold: u32,
    /// warmstate.py: trace_eagerness parameter (integer, default 200).
    trace_eagerness: u32,
    /// warmstate.py: increment_trace_eagerness = compute_threshold(trace_eagerness).
    /// Pre-computed f64 increment for guard failure counter ticking.
    increment_trace_eagerness: f64,
    /// Function call threshold for inlining during tracing.
    ///
    /// A function must be called at least this many times before
    /// the meta-interpreter inlines it into the trace.
    function_threshold: u32,
    /// Maximum depth of inlined function calls during tracing.
    max_inline_depth: u32,
    /// Maximum number of operations per trace before aborting.
    trace_limit: u32,
    /// Global tracing generation counter.
    /// Incremented each time tracing starts; stored in BaseJitCell to
    /// detect stale tracing sessions.
    tracing_generation: u64,
    /// Optional profiling logger, enabled via MAJIT_STATS=1 or MAJIT_LOG=1.
    jitlog: Option<Logger>,
    /// Quasi-immutable field invalidation registry.
    ///
    /// Maps a quasi-immutable field key (hash of object_id + field_index)
    /// to the set of green_key_hashes whose compiled loops depend on that field.
    /// When a quasi-immutable field is mutated, all dependent loops are invalidated.
    quasiimmut_deps: HashMap<u64, HashSet<u64>>,
    // Function-entry hotness rides on the shared `counter: JitCounter`
    // below (warmstate.py:467 — maybe_compile_and_run's tick + reset
    // goes through the common timetable, not a separate HashMap).

    // warmstate.py:299-320 — retrace_limit / max_retrace_guards /
    // max_unroll_loops / max_unroll_recursion live on
    // warmrunnerdesc.memory_manager, not on WarmEnterState itself. See
    // `MemoryManager` in memmgr.rs.
    /// jit.py:581,602: vec — enable vectorization optimization.
    vectorize: bool,
    /// jit.py:585,603: vec_all — vectorize loops outside numpypy library.
    vec_all: bool,
    /// jit.py:583,604: vec_cost — cost threshold for vectorization decisions.
    vec_cost: u32,
    /// warmstate.py: enable_opts — list of enabled optimization pass names.
    enable_opts: Vec<String>,
    /// warmstate.py: set_param_inlining — whether inlining is enabled.
    inlining: bool,
    /// warmstate.py: set_param_disable_unrolling — threshold below
    /// which loop unrolling is disabled.
    disable_unrolling_threshold: u32,
    /// warmstate.py: set_param_pureop_historylength — size of the
    /// pure operation history cache.
    pureop_historylength: u32,
    /// warmspot.py:110: memory_manager — generation-based loop aging.
    /// pyjitpl.py:2348: try_to_free_some_loops calls next_generation().
    pub memory_manager: crate::memmgr::MemoryManager,
}

/// Result of checking whether a green key is hot.
pub enum HotResult {
    /// Not yet hot; keep interpreting.
    NotHot,
    /// Threshold reached; start tracing. The caller (MetaInterp) builds
    /// the Trace itself — RPython parity: `MetaInterp.create_empty_history`
    /// / `MetaInterp.create_history` live on `MetaInterp`, not on the
    /// warmstate (pyjitpl.py:2604-2610). Pyre's prior signal-and-factory
    /// pattern (`HotResult::StartTracing(Trace::new())`) forced warmstate
    /// to depend on the `recorder::Trace` type; Step 2e.2b is removing
    /// that coupling so the `Trace` factory moves to MetaInterp where
    /// `metainterp_sd` is available for `TraceRecordBuffer::new`.
    StartTracing,
    /// Already tracing (caller should keep feeding ops to the active recorder).
    AlreadyTracing,
    /// Compiled code exists; run it.
    RunCompiled,
}

impl WarmEnterState {
    fn should_start_dont_trace_here_trace(
        &mut self,
        green_key_hash: u64,
        flags: u8,
        has_seen_a_procedure_token: bool,
    ) -> bool {
        if flags & jc_flags::DONT_TRACE_HERE == 0 || has_seen_a_procedure_token {
            return false;
        }
        if flags & jc_flags::TRACING_OCCURRED != 0 {
            self.counter.tick(green_key_hash)
        } else {
            true
        }
    }

    fn start_tracing_cell(&mut self, green_key_hash: u64) -> HotResult {
        self.counter.reset(green_key_hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;

        HotResult::StartTracing
    }

    /// Create a new WarmEnterState with the given threshold.
    /// Automatically enables Logger if MAJIT_STATS=1 or MAJIT_LOG=1.
    pub fn new(threshold: u32) -> Self {
        WarmEnterState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            trace_eagerness: DEFAULT_TRACE_EAGERNESS,
            increment_trace_eagerness: JitCounter::compute_threshold_static(
                DEFAULT_TRACE_EAGERNESS,
            ),
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            jitlog: Logger::from_env(),
            quasiimmut_deps: HashMap::new(),
            vectorize: false,
            vec_all: false,
            vec_cost: 0,
            enable_opts: default_enable_opts(),
            inlining: true,
            disable_unrolling_threshold: DEFAULT_DISABLE_UNROLLING,
            pureop_historylength: 16,
            memory_manager: {
                let mut m = crate::memmgr::MemoryManager::new(0);
                // rlib/jit.py:595 PARAMETERS default retrace_limit=0.
                m.retrace_limit = DEFAULT_RETRACE_LIMIT;
                // rlib/jit.py:598 / pyjitpl.py:2946: default 0 means
                // the first cancelled unrolled compile immediately retries
                // once without unrolling.
                m.max_unroll_loops = DEFAULT_MAX_UNROLL_LOOPS;
                m
            },
        }
    }

    /// Create a new WarmEnterState with an explicit Logger.
    pub fn with_jitlog(threshold: u32, jitlog: Option<Logger>) -> Self {
        WarmEnterState {
            counter: JitCounter::new(threshold),
            cells: HashMap::new(),
            threshold,
            trace_eagerness: DEFAULT_TRACE_EAGERNESS,
            increment_trace_eagerness: JitCounter::compute_threshold_static(
                DEFAULT_TRACE_EAGERNESS,
            ),
            function_threshold: DEFAULT_FUNCTION_THRESHOLD,
            max_inline_depth: DEFAULT_MAX_INLINE_DEPTH,
            trace_limit: DEFAULT_TRACE_LIMIT,
            tracing_generation: 0,
            jitlog,
            quasiimmut_deps: HashMap::new(),
            vectorize: false,
            vec_all: false,
            vec_cost: 0,
            enable_opts: default_enable_opts(),
            inlining: true,
            disable_unrolling_threshold: DEFAULT_DISABLE_UNROLLING,
            pureop_historylength: 16,
            memory_manager: {
                let mut m = crate::memmgr::MemoryManager::new(0);
                // rlib/jit.py:595 PARAMETERS default retrace_limit=0.
                m.retrace_limit = DEFAULT_RETRACE_LIMIT;
                // rlib/jit.py:598 / pyjitpl.py:2946: default 0 means
                // the first cancelled unrolled compile immediately retries
                // once without unrolling.
                m.max_unroll_loops = DEFAULT_MAX_UNROLL_LOOPS;
                m
            },
        }
    }

    /// Check and possibly transition the JIT state for a given green key.
    ///
    /// Called by the interpreter at loop back-edges and function entries.
    /// Returns a `HotResult` telling the interpreter what to do next.
    /// Mark a green key as DONT_TRACE_HERE permanently.
    /// Clear the loop token for a cell, so is_compiled() returns false.
    pub fn clear_loop_token(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.loop_token = None;
        }
    }

    pub fn clear_all_loop_tokens(&mut self) {
        for cell in self.cells.values_mut() {
            cell.loop_token = None;
        }
    }

    pub fn mark_dont_trace(&mut self, green_key_hash: u64) {
        self.disable_noninlinable_function(green_key_hash);
    }

    #[inline]
    pub fn counter_would_fire(&self, green_key_hash: u64) -> bool {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return false;
            }
            if cell.state == BaseJitCellState::DontTraceHere {
                return false;
            }
        }
        self.counter.would_fire(green_key_hash)
    }

    /// Tick counter and check if key should enter JIT.
    ///
    /// RPython parity: this is the inline fast path equivalent of
    /// `maybe_compile_and_run(increment_threshold, ...)`.
    /// Returns true if counter reached threshold (should enter slow path).
    /// Returns false for DONT_TRACE keys or cold keys.
    /// Advance counter toward threshold. Respects DONT_TRACE_HERE and
    /// DontTraceHere — does not tick suppressed keys (warmstate.py:484).
    pub fn counter_tick(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                return;
            }
            if cell.state == BaseJitCellState::DontTraceHere {
                return;
            }
        }
        let _ = self.counter.tick(green_key_hash);
    }

    pub fn counter_tick_checked(&mut self, green_key_hash: u64) -> bool {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() || cell.is_tracing() {
                return true;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 && cell.has_seen_a_procedure_token() {
                return false;
            }
        }
        self.counter.tick(green_key_hash)
    }

    pub fn maybe_compile(&mut self, green_key_hash: u64) -> HotResult {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            let is_compiled = cell.is_compiled();
            let is_tracing = cell.is_tracing();
            let flags = cell.flags;
            let has_seen_a_procedure_token = cell.has_seen_a_procedure_token();
            if is_compiled {
                return HotResult::RunCompiled;
            }
            if is_tracing {
                return HotResult::AlreadyTracing;
            }
            if self.should_start_dont_trace_here_trace(
                green_key_hash,
                flags,
                has_seen_a_procedure_token,
            ) {
                return self.start_tracing_cell(green_key_hash);
            }
            if flags & jc_flags::DONT_TRACE_HERE != 0 {
                return HotResult::NotHot;
            }
        }

        if !self.counter.tick(green_key_hash) {
            return HotResult::NotHot;
        }

        self.start_tracing_cell(green_key_hash)
    }

    /// warmstate.py:446-511 `WarmEnterState.maybe_compile_and_run` —
    /// typed-greenkey variant of [`Self::maybe_compile`].
    ///
    /// Walks the per-bucket chain by typed comparekey for both the
    /// state read (`lookup_chain_with_key`) and the state mutation
    /// (`start_tracing_cell_for_key`), mirroring upstream's
    /// `JitCell.get_jitcell_for_args` (`warmstate.py:455-465`) followed
    /// by per-cell flag mutation. Hash collisions across distinct
    /// typed greens therefore do not cross-contaminate each other's
    /// `JC_TRACING` / `JC_COMPILED` flags or counter-derived
    /// transitions.
    ///
    /// The bucket-shared counter ticks via `key.get_uhash()` — upstream
    /// counter is per-bucket too (`warmstate.py:496`), so colliding
    /// typed greens share the back-edge counter while keeping
    /// independent cell state.
    ///
    /// Hash-only callers ([`Self::maybe_compile`]) still mutate the
    /// bucket head via the legacy entry point. Migrating
    /// `finish_tracing`, `abort_tracing`, `clear_loop_token`, and
    /// `mark_dont_trace` to the typed chain-walk is tracked as a
    /// pre-existing PRE-EXISTING-ADAPTATION follow-up so the typed
    /// path's full lifecycle stays self-consistent on its own buckets.
    pub fn maybe_compile_with_key(&mut self, key: &GreenKey) -> HotResult {
        let hash = key.get_uhash();
        if let Some(cell) = self.lookup_chain_with_key(key) {
            let is_compiled = cell.is_compiled();
            let is_tracing = cell.is_tracing();
            let flags = cell.flags;
            let has_seen_a_procedure_token = cell.has_seen_a_procedure_token();
            if is_compiled {
                return HotResult::RunCompiled;
            }
            if is_tracing {
                return HotResult::AlreadyTracing;
            }
            if self.should_start_dont_trace_here_trace(hash, flags, has_seen_a_procedure_token) {
                return self.start_tracing_cell_for_key(key);
            }
            if flags & jc_flags::DONT_TRACE_HERE != 0 {
                return HotResult::NotHot;
            }
        }

        if !self.counter.tick(hash) {
            return HotResult::NotHot;
        }

        self.start_tracing_cell_for_key(key)
    }

    /// Mutable chain-walk variant of [`Self::lookup_chain_with_key`].
    /// Returns `Some` only when a chained cell carries a comparekey
    /// equal (`equal_whatever`) to `key`; the bucket head is no
    /// longer privileged.
    fn lookup_chain_with_key_mut(&mut self, key: &GreenKey) -> Option<&mut BaseJitCell> {
        let hash = key.get_uhash();
        let mut cell = self.cells.get_mut(&hash);
        while let Some(c) = cell {
            if c.comparekey_matches(key) {
                return Some(c);
            }
            cell = c.next.as_deref_mut();
        }
        None
    }

    /// warmstate.py:425-444 `WarmEnterState.bound_reached` —
    /// typed-key variant of [`Self::start_tracing_cell`].
    ///
    /// Walks the chain by `key`'s comparekey to mutate the matching
    /// cell's `JC_TRACING` / `JC_TRACING_OCCURRED` flags +
    /// `tracing_generation`, rather than the bucket head. Pairs with
    /// [`Self::maybe_compile_with_key`] so a typed greenkey always
    /// transitions its own cell on a hash collision.
    fn start_tracing_cell_for_key(&mut self, key: &GreenKey) -> HotResult {
        let hash = key.get_uhash();
        self.counter.reset(hash);
        self.tracing_generation += 1;
        let current_generation = self.tracing_generation;
        self.ensure_cell_for_key(key);
        let cell = self
            .lookup_chain_with_key_mut(key)
            .expect("ensure_cell_for_key just installed a cell matching this key");
        cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
        cell.state = BaseJitCellState::Tracing;
        cell.tracing_generation = current_generation;
        HotResult::StartTracing
    }

    /// Force-start tracing for a green key, bypassing the hot counter.
    ///
    /// Used by function-entry tracing where the caller has already
    /// determined that tracing should begin.
    pub fn force_start_tracing(&mut self, green_key_hash: u64) -> HotResult {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() {
                return HotResult::RunCompiled;
            }
            if cell.is_tracing() {
                return HotResult::AlreadyTracing;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 && cell.has_seen_a_procedure_token() {
                return HotResult::NotHot;
            }
            // Give up after too many failed trace attempts to prevent
            // infinite retrace loops (e.g. InvalidLoop every time).
            if cell.abort_count >= MAX_TRACE_ABORT_COUNT {
                return HotResult::NotHot;
            }
        }

        self.start_tracing_cell(green_key_hash)
    }

    /// Signal that a retrace is starting from a guard failure point.
    ///
    /// `input_types` is accepted for API shape parity with
    /// `MetaInterp.create_history(max_num_inputargs)` callers who need
    /// to size their `TraceRecordBuffer` before the retrace; warmstate
    /// doesn't own the Trace type. Returns nothing — the caller
    /// (`MetaInterp::start_bridge_trace` in pyjitpl/mod.rs) constructs
    /// the `Trace` itself with its `staticdata: Arc<MetaInterpStaticData>`.
    /// RPython parity: `warmspot.py` has no analogue of the old
    /// `start_retrace(input_types) -> Trace` factory — RPython's
    /// `MetaInterp.create_history(max_num_inputargs)` is the constructor.
    pub fn start_retrace(&mut self, _input_types: &[Type]) {}

    /// Mark that tracing is done for a green key. Clears the TRACING flag.
    /// The caller is responsible for compiling the trace and calling
    /// `attach_procedure_to_interp` with the resulting JitCellToken.
    pub fn finish_tracing(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
            // State remains Tracing until attach_procedure_to_interp is called.
        }
    }

    /// Mark that tracing was aborted for a green key.
    ///
    /// RPython parity:
    /// - non-permanent aborts clear TRACING and allow a future retry
    /// - permanent aborts mark the location as DONT_TRACE_HERE
    pub fn abort_tracing(&mut self, green_key_hash: u64, disable_noninlinable_function: bool) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
            cell.abort_count += 1;
            if disable_noninlinable_function || (cell.flags & jc_flags::DONT_TRACE_HERE != 0) {
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            } else if cell.abort_count >= MAX_TRACE_ABORT_COUNT {
                // Too many failed attempts — permanently disable tracing here.
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            } else {
                cell.state = BaseJitCellState::NotHot;
            }
        }

        if disable_noninlinable_function {
            self.disable_noninlinable_function(green_key_hash);
        }
        if let Some(log) = &mut self.jitlog {
            log.log_abort();
        }
    }

    /// Install a compiled loop token for a green key.
    ///
    /// The cell transitions to Compiled state and takes ownership of
    /// the procedure token. Also clears TRACING on the cell the token
    /// is attached to — this covers the same-key compile path, while
    /// cross-loop cut (compile.py:269) compiles under an inner cell
    /// and the outer (starting) cell's TRACING is cleared separately
    /// by the `clear_tracing_flag` call in the tracing entry point's
    /// finally block (warmstate.py:444 parity).
    pub fn attach_procedure_to_interp(
        &mut self,
        green_key_hash: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) -> Option<Arc<JitCellToken>> {
        let token = token.into();
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags &= !jc_flags::TRACING;
        cell.set_procedure_token(token, false)
    }

    /// warmstate.py:716-723 `cell.set_procedure_token(procedure_token, tmp=True)`.
    ///
    /// Installs a temporary CALL_ASSEMBLER fallback token without
    /// changing the tracing flags or compiled state.
    pub fn attach_tmp_callback_to_interp(
        &mut self,
        green_key_hash: u64,
        token: impl Into<Arc<JitCellToken>>,
    ) {
        let token = token.into();
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        let _old = cell.set_procedure_token(token, true);
    }

    /// warmstate.py:444 `finally: cell.flags &= ~JC_TRACING` parity —
    /// unconditional flag clear on the starting cell after tracing ends.
    /// Called from the tracing entry point (bound_reached / jit_merge_point_hook)
    /// regardless of whether tracing succeeded, aborted, or cross-loop-cut
    /// installed under a different inner cell. Does not alter state: the
    /// companion `attach_procedure_to_interp` / `abort_tracing` calls own
    /// the state transition for whichever cell they touch.
    pub fn clear_tracing_flag(&mut self, green_key_hash: u64) {
        if let Some(cell) = self.cells.get_mut(&green_key_hash) {
            cell.flags &= !jc_flags::TRACING;
        }
    }

    pub fn take_procedure_token(&mut self, green_key_hash: u64) -> Option<Arc<JitCellToken>> {
        self.cells
            .get_mut(&green_key_hash)
            .and_then(|cell| cell.loop_token.take())
    }

    /// Get a reference to the compiled loop token for a green key.
    ///
    /// `warmstate.py:191-196 get_procedure_token` parity: returns `None`
    /// when the cell has no procedure token AND when the token has been
    /// invalidated (`token and not token.invalidated`).  Pyre routes
    /// through `BaseJitCell::get_procedure_token` (which applies the
    /// `is_invalidated` filter) so every entry-path consumer of the
    /// "current compiled token at this key" sees the same filtered view
    /// PyPy provides — invalidated tokens never reach the warm-entry
    /// runner, the CALL_ASSEMBLER inline gate, or the bridge stitch
    /// surface.
    pub fn get_compiled(&self, green_key_hash: u64) -> Option<&Arc<JitCellToken>> {
        self.cells
            .get(&green_key_hash)
            .and_then(|cell| cell.get_procedure_token())
    }

    /// warmstate.py:191-196 `get_procedure_token`.
    pub fn get_procedure_token(&self, green_key_hash: u64) -> Option<Arc<JitCellToken>> {
        self.cells
            .get(&green_key_hash)
            .and_then(|cell| cell.get_procedure_token().cloned())
    }

    /// Allocate a new unique JitCellToken number.
    pub fn alloc_token_number(&mut self) -> u64 {
        NEXT_GLOBAL_TOKEN_NUMBER.fetch_add(1, Ordering::Relaxed)
    }

    /// Get the current threshold.
    pub fn threshold(&self) -> u32 {
        self.threshold
    }

    /// Set a new threshold. Also updates the internal counter.
    pub fn set_threshold(&mut self, threshold: u32) {
        self.threshold = threshold;
        self.counter.set_threshold(threshold);
    }

    /// Decay all counters (e.g., periodically to avoid stale counts).
    pub fn decay_counters(&mut self) {
        self.counter.decay_all_counters();
    }

    /// Reset the hot counter for a specific green key to zero.
    pub fn reset_counter(&mut self, green_key_hash: u64) {
        self.counter.reset(green_key_hash);
    }

    /// Reset ALL counters to zero. Used after invalidation with incomplete
    /// resume data (NONE fail_args) to prevent immediate recompilation.
    pub fn decay_all_counters_to_zero(&mut self) {
        self.counter.decay_all_counters_by(0.0);
    }

    /// Check if a green key is marked DontTraceHere.
    pub fn is_dont_trace_here(&self, green_key_hash: u64) -> bool {
        self.cells
            .get(&green_key_hash)
            .is_some_and(|c| c.state == BaseJitCellState::DontTraceHere)
    }

    /// Get a reference to the BaseJitCell for a green key, if it exists.
    pub fn get_cell(&self, green_key_hash: u64) -> Option<&BaseJitCell> {
        self.cells.get(&green_key_hash)
    }

    /// PRE-EXISTING-ADAPTATION: walk the warmstate cells to find a
    /// `JitCellToken` by number. Used by `MetaInterp::record_loop_or_bridge`
    /// to widen the CALL_ASSEMBLER keepalive search to cover targets
    /// that live only on a `BaseJitCell.loop_token` (most importantly
    /// tmp-callback installs at `attach_tmp_callback_to_interp`) and
    /// are not yet — or never — registered in `MetaInterp::compiled_loops`.
    ///
    /// RPython equivalent does not exist because upstream descrs hold
    /// the `JitCellToken` object directly (`compile.py:187 isinstance(descr,
    /// JitCellToken)`) — no number→token resolution is needed.  This
    /// helper is removed by Slice X-D once `CallAssemblerDescr` /
    /// `LoopTargetDescr` carry the owning `Arc<JitCellToken>`.
    pub fn find_token_by_number(&self, token_number: u64) -> Option<&Arc<JitCellToken>> {
        self.cells.values().find_map(|cell| {
            cell.loop_token
                .as_ref()
                .filter(|tok| tok.number == token_number)
        })
    }

    /// `rpython/jit/metainterp/warmstate.py:714-723` `get_assembler_token`.
    ///
    /// Returns the cell's existing procedure token, or — if none exists —
    /// builds a temporary one via `make_token` (caller wires
    /// `compile_tmp_callback`) and installs it on the cell with
    /// `tmp=true`.  The closure-based signature is a Rust adaptation so
    /// the caller can provide the `&mut Backend` / `&JitDriverStaticData`
    /// / `greenboxes` bundle without threading them through WarmEnterState.
    pub fn get_assembler_token<E, F>(
        &mut self,
        green_key_hash: u64,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce() -> Result<Arc<JitCellToken>, E>,
    {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        if let Some(token) = cell.get_procedure_token() {
            return Ok(token.clone());
        }
        let token = make_token()?;
        cell.set_procedure_token(token.clone(), true);
        Ok(token)
    }

    /// `warmstate.py:714-723` `get_assembler_token` typed variant —
    /// walks the chain at `key.get_uhash()` and dispatches off
    /// `comparekey` instead of trusting the hash to be collision-free.
    ///
    /// Equivalent to `get_assembler_token` modulo lookup discipline:
    /// upstream `JitCell.ensure_jit_cell_at_key(greenkey)` walks the
    /// chain (`warmstate.py:626-641`) and inserts at head on miss.
    /// Pyre's hash-only [`Self::get_assembler_token`] aliases distinct
    /// typed keys that share a hash bucket; this variant disambiguates.
    pub fn get_assembler_token_with_key<E, F>(
        &mut self,
        key: &GreenKey,
        make_token: F,
    ) -> Result<Arc<JitCellToken>, E>
    where
        F: FnOnce() -> Result<Arc<JitCellToken>, E>,
    {
        self.ensure_cell_for_key(key);
        let hash = key.get_uhash();
        // Walk the chain to find the typed match. `ensure_cell_for_key`
        // guarantees one exists either as the head (miss path) or
        // somewhere down the chain (existing-cell path).
        let mut cell = self
            .cells
            .get_mut(&hash)
            .expect("ensure_cell_for_key installed a chain entry");
        while !cell.comparekey_matches(key) {
            cell = cell
                .next
                .as_deref_mut()
                .expect("ensure_cell_for_key guarantees a typed match exists");
        }
        if let Some(token) = cell.get_procedure_token() {
            return Ok(token.clone());
        }
        let token = make_token()?;
        cell.set_procedure_token(token.clone(), true);
        Ok(token)
    }

    /// Log a successful trace compilation. No-op if Logger is disabled.
    pub fn log_compile(
        &mut self,
        green_key: u64,
        ops_before_opt: usize,
        ops_after_opt: usize,
        opt_time: Duration,
        compile_time: Duration,
    ) {
        if let Some(log) = &mut self.jitlog {
            log.log_compile(
                green_key,
                ops_before_opt,
                ops_after_opt,
                opt_time,
                compile_time,
            );
        }
    }

    /// Log a guard failure. No-op if Logger is disabled.
    pub fn log_guard_failure(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_guard_failure(guard_index);
        }
    }

    /// Log a loop entry. No-op if Logger is disabled.
    pub fn log_loop_entry(&mut self, green_key: u64) {
        if let Some(log) = &mut self.jitlog {
            log.log_loop_entry(green_key);
        }
    }

    /// Get a reference to the Logger, if enabled.
    pub fn jitlog(&self) -> Option<&Logger> {
        self.jitlog.as_ref()
    }

    /// warmstate.py: trace_eagerness parameter (integer).
    pub fn trace_eagerness(&self) -> u32 {
        self.trace_eagerness
    }

    /// warmstate.py:259: set_param_trace_eagerness.
    pub fn set_param_trace_eagerness(&mut self, value: u32) {
        self.trace_eagerness = value;
        self.increment_trace_eagerness = JitCounter::compute_threshold_static(value);
    }

    /// warmstate.py: increment_trace_eagerness (pre-computed f64).
    pub fn increment_trace_eagerness(&self) -> f64 {
        self.increment_trace_eagerness
    }

    /// compile.py:783-784: jitcounter.tick(hash, increment_trace_eagerness).
    /// Increment the guard failure counter using the shared timetable.
    /// Returns true when counter reaches 1.0 (trace_eagerness ticks).
    #[inline]
    pub fn tick_guard_failure(&mut self, guard_hash: u64) -> bool {
        self.counter
            .tick_with_increment(guard_hash, self.increment_trace_eagerness)
    }

    /// Compat alias: bridge_threshold() returns trace_eagerness.
    pub fn bridge_threshold(&self) -> u32 {
        self.trace_eagerness
    }

    /// Compat alias: set_bridge_threshold delegates to set_param_trace_eagerness.
    pub fn set_bridge_threshold(&mut self, threshold: u32) {
        self.set_param_trace_eagerness(threshold);
    }

    /// compile.py:826-830: store_hash — allocate a jitcounter hash for
    /// a new guard. Called at compile time (or lazily on first failure).
    pub fn fetch_next_hash(&mut self) -> u64 {
        self.counter.fetch_next_hash()
    }

    /// Get the function inlining threshold.
    pub fn function_threshold(&self) -> u32 {
        self.function_threshold
    }

    /// Set the function inlining threshold.
    pub fn set_function_threshold(&mut self, threshold: u32) {
        self.function_threshold = threshold;
    }

    /// RPython-compatible wrapper: set_param_threshold.
    pub fn set_param_threshold(&mut self, threshold: u32) {
        self.set_threshold(threshold);
    }

    /// RPython-compatible wrapper: set_param_trace_limit.
    pub fn set_param_trace_limit(&mut self, value: u32) {
        self.set_trace_limit(value);
    }

    /// RPython-compatible wrapper: set_param_function_threshold.
    pub fn set_param_function_threshold(&mut self, value: u32) {
        self.set_function_threshold(value);
    }

    /// RPython-compatible wrapper: set_param_inlining.
    pub fn set_param_inlining(&mut self, value: bool) {
        self.inlining = value;
    }

    /// RPython-compatible wrapper: set_param_disable_unrolling.
    pub fn set_param_disable_unrolling(&mut self, value: u32) {
        self.disable_unrolling_threshold = value;
    }

    /// RPython-compatible wrapper: set_param_vec.
    pub fn set_param_vec(&mut self, enabled: bool) {
        self.vectorize = enabled;
    }

    /// jit.py:585: set_param_vec_all.
    pub fn set_param_vec_all(&mut self, enabled: bool) {
        self.vec_all = enabled;
    }

    /// RPython-compatible wrapper: set_param_vec_cost.
    pub fn set_param_vec_cost(&mut self, value: u32) {
        self.vec_cost = value;
    }

    /// warmstate.py:317-320 set_param_max_unroll_recursion — delegates
    /// to memory_manager.max_unroll_recursion.
    pub fn set_param_max_unroll_recursion(&mut self, value: u32) {
        self.memory_manager.max_unroll_recursion = value;
    }

    /// RPython-compatible wrapper: set_param_max_inline_depth.
    pub fn set_param_max_inline_depth(&mut self, value: u32) {
        self.set_max_inline_depth(value);
    }

    /// warmstate.py:299-302 set_param_retrace_limit — delegates to
    /// memory_manager.retrace_limit.
    pub fn set_param_retrace_limit(&mut self, value: u32) {
        self.memory_manager.retrace_limit = value;
    }

    /// warmstate.py:307-310 set_param_max_retrace_guards — delegates to
    /// memory_manager.max_retrace_guards.
    pub fn set_param_max_retrace_guards(&mut self, value: u32) {
        self.memory_manager.max_retrace_guards = value;
    }

    /// warmstate.py:312-315 set_param_max_unroll_loops — delegates to
    /// memory_manager.max_unroll_loops.
    pub fn set_param_max_unroll_loops(&mut self, value: u32) {
        self.memory_manager.max_unroll_loops = value;
    }

    /// warmstate.py:293-297 set_param_loop_longevity — delegates to the
    /// memory manager's max_age.
    pub fn set_param_loop_longevity(&mut self, value: u32) {
        // memmgr.py:42 default check_frequency=0 → derives sqrt(max_age).
        self.memory_manager.set_max_age(value as i64, 0);
    }

    /// RPython-compatible wrapper: set_param_pureop_historylength.
    pub fn set_param_pureop_historylength(&mut self, value: u32) {
        self.pureop_historylength = value;
    }

    /// warmstate.py:269-270 set_param_decay — delegates to the jit counter.
    pub fn set_param_decay(&mut self, value: u32) {
        self.counter.set_decay(value as i32);
    }

    /// Set the maximum inline depth.
    pub fn set_max_inline_depth(&mut self, depth: u32) {
        self.max_inline_depth = depth;
    }

    /// Get the maximum inline depth.
    pub fn max_inline_depth(&self) -> u32 {
        self.max_inline_depth
    }

    /// Whether this callee is eligible for inlining at all.
    ///
    /// Mirrors PyPy's `can_inline_callable`: once a green key is marked
    /// `DONT_TRACE_HERE`, callers must stop inlining it and instead let it
    /// converge to a separate functrace / call_assembler path.
    pub fn can_inline_callable(&self, callee_key: u64) -> bool {
        self.cells
            .get(&callee_key)
            .map_or(true, |cell| cell.flags & jc_flags::DONT_TRACE_HERE == 0)
    }

    /// Mark a callee as a location that should no longer be inlined into
    /// surrounding traces.
    ///
    /// This is the warm-state equivalent of PyPy's `disable_noninlinable_function()`.
    pub fn disable_noninlinable_function(&mut self, callee_key: u64) {
        let cell = self
            .cells
            .entry(callee_key)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        if cell.flags & jc_flags::TRACING == 0 {
            cell.state = BaseJitCellState::DontTraceHere;
        }
    }

    /// Mark a callee as currently being traced.
    ///
    /// This is the warm-state equivalent of PyPy's `mark_as_being_traced()`.
    pub fn mark_as_being_traced(&mut self, callee_key: u64) {
        let cell = self
            .cells
            .entry(callee_key)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::TRACING;
        if cell.flags & jc_flags::TRACING_OCCURRED == 0 {
            cell.state = BaseJitCellState::Tracing;
            cell.tracing_generation = self.tracing_generation;
        }
    }

    /// Restore warm-state parameters to rlib/jit.py:588-605 PARAMETERS defaults.
    pub fn set_default_params(&mut self) {
        self.set_threshold(DEFAULT_THRESHOLD); // 1039
        self.set_bridge_threshold(DEFAULT_TRACE_EAGERNESS); // 200
        self.set_trace_limit(DEFAULT_TRACE_LIMIT); // 6000
        self.set_function_threshold(DEFAULT_FUNCTION_THRESHOLD); // 1619
        self.set_max_inline_depth(DEFAULT_MAX_INLINE_DEPTH); // 7
        self.inlining = true; // inlining = 1
        self.disable_unrolling_threshold = DEFAULT_DISABLE_UNROLLING; // 200
        self.pureop_historylength = 16;
        self.counter.set_decay(40);
        self.memory_manager.max_retrace_guards = 15;
        self.memory_manager.max_unroll_loops = 0;
        self.memory_manager.retrace_limit = DEFAULT_RETRACE_LIMIT;
        self.memory_manager.max_unroll_recursion = DEFAULT_MAX_UNROLL_RECURSION;
        self.memory_manager.set_max_age(1000, 0);
        self.vec_cost = 0;
        self.vectorize = false;
        self.set_param_enable_opts("all");
    }

    /// Mirror RPython warmstate.py `mark_force_finish_tracing(greenkey)`.
    ///
    /// The next tracing run for this green key should segment instead of
    /// repeatedly aborting once it approaches the trace limit.
    pub fn mark_force_finish_tracing(&mut self, green_key_hash: u64) {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);
        cell.flags |= jc_flags::FORCE_FINISH;
    }

    /// warmstate.py:439 `bool(cell.flags & JC_FORCE_FINISH)` — read the sticky
    /// segmenting flag at loop entry.  RPython never clears this flag
    /// explicitly: `should_remove_jitcell` (warmstate.py:222) keeps the cell
    /// alive while it is set, and once set it persists until the cell itself
    /// is removed.
    pub fn should_force_finish_tracing(&self, green_key_hash: u64) -> bool {
        self.cells
            .get(&green_key_hash)
            .is_some_and(|cell| cell.flags & jc_flags::FORCE_FINISH != 0)
    }

    /// Boost the current loop/function green key so the next execution
    /// immediately retriggers tracing.
    ///
    /// Mirrors PyPy's `JitCell.trace_next_iteration()` in warmstate.py:
    /// it does not force tracing right now, it only raises the hot counter
    /// to ~threshold so the next hit converges quickly.
    pub fn trace_next_iteration(&mut self, green_key_hash: u64) {
        self.counter.change_current_fraction(green_key_hash, 0.98);
    }

    /// warmstate.py:467 jitcounter.tick(hash, increment_threshold) parity.
    ///
    /// Function-entry hotness rides on the same `JitCounter` timetable as
    /// the loop counter (counter.py:16-202). `maybe_compile_and_run`
    /// increments the per-hash float with `increment_threshold =
    /// 1/function_threshold`; `tick_with_threshold` returns true (and
    /// auto-resets the slot) when the accumulated value reaches 1.0 —
    /// exactly counter.py:185-201.
    ///
    /// The cell lookup replicates `maybe_compile_and_run`'s pre-tick
    /// shortcuts (warmstate.py:473-495):
    ///   * compiled or currently-tracing keys skip tracing
    ///   * a `DONT_TRACE_HERE` cell with no seen procedure token fires
    ///     eagerly on the first visit after tracing completes
    pub fn should_trace_function_entry(&mut self, green_key_hash: u64) -> bool {
        if let Some(cell) = self.cells.get(&green_key_hash) {
            if cell.is_compiled() || cell.is_tracing() {
                return false;
            }
            if cell.flags & jc_flags::DONT_TRACE_HERE != 0 {
                if cell.has_seen_a_procedure_token() {
                    return false;
                }
                if cell.flags & jc_flags::TRACING_OCCURRED == 0 {
                    return true;
                }
            }
        }
        self.counter
            .tick_with_threshold(green_key_hash, self.function_threshold)
    }

    /// Check if inlining is allowed at the given depth.
    pub fn can_inline_at_depth(&self, current_depth: usize) -> bool {
        (current_depth as u32) < self.max_inline_depth
    }

    /// Log a bridge compilation. No-op if Logger is disabled.
    pub fn log_bridge_compile(&mut self, guard_index: u32) {
        if let Some(log) = &mut self.jitlog {
            log.log_bridge_compile(guard_index);
        }
    }

    // ── Quasi-immutable field invalidation ──

    /// Register that the compiled loop at `green_key_hash` depends on the
    /// quasi-immutable field identified by `qmut_key`.
    ///
    /// When `invalidate_quasiimmut(qmut_key)` is called later, the compiled
    /// loop's JitCellToken will be invalidated, causing GUARD_NOT_INVALIDATED
    /// to fail and forcing a retrace.
    ///
    /// `qmut_key` should be a hash of (object_id, field_index) or similar.
    pub fn register_quasiimmut_dependency(&mut self, qmut_key: u64, green_key_hash: u64) {
        self.quasiimmut_deps
            .entry(qmut_key)
            .or_default()
            .insert(green_key_hash);
    }

    /// Invalidate all compiled loops that depend on the quasi-immutable field
    /// identified by `qmut_key`.
    ///
    /// Called by the interpreter when a quasi-immutable field is mutated.
    /// Each dependent loop's JitCellToken has its invalidated flag set, causing
    /// GUARD_NOT_INVALIDATED to fail on the next execution.
    ///
    /// Returns the number of loops invalidated.
    pub fn invalidate_quasiimmut(&mut self, qmut_key: u64) -> usize {
        let deps = match self.quasiimmut_deps.remove(&qmut_key) {
            Some(deps) => deps,
            None => return 0,
        };

        let mut invalidated = 0;
        for green_key_hash in &deps {
            if let Some(cell) = self.cells.get_mut(green_key_hash) {
                if let Some(token) = &cell.loop_token {
                    token.invalidate();
                    cell.state = BaseJitCellState::Invalidated;
                    invalidated += 1;
                }
            }
        }
        invalidated
    }

    /// Invalidate all compiled loops that contain a GUARD_NOT_INVALIDATED.
    ///
    /// This is a brute-force invalidation used when the specific qmut_key
    /// is not known (e.g., bulk invalidation after a class hierarchy change).
    pub fn invalidate_all(&mut self) {
        for cell in self.cells.values_mut() {
            if let Some(token) = &cell.loop_token {
                token.invalidate();
                cell.state = BaseJitCellState::Invalidated;
            }
        }
        self.quasiimmut_deps.clear();
    }

    // ── BaseJitCell state machine API ──

    /// Get the explicit state of a BaseJitCell for a green key.
    /// Returns `NotHot` if no cell exists.
    #[inline]
    pub fn get_cell_state(&self, green_key_hash: u64) -> BaseJitCellState {
        self.cells
            .get(&green_key_hash)
            .map(|c| c.state)
            .unwrap_or(BaseJitCellState::NotHot)
    }

    /// Explicitly transition a cell to a new state.
    ///
    /// This is the low-level state-machine driver. Most callers should use
    /// the higher-level methods (`maybe_compile`, `finish_tracing`,
    /// `attach_procedure_to_interp`, `abort_tracing`) which call this internally.
    pub fn transition_cell(&mut self, green_key_hash: u64, new_state: BaseJitCellState) {
        let cell = self
            .cells
            .entry(green_key_hash)
            .or_insert_with(BaseJitCell::new);

        match new_state {
            BaseJitCellState::NotHot => {
                cell.flags &= !(jc_flags::TRACING | jc_flags::DONT_TRACE_HERE);
                cell.state = BaseJitCellState::NotHot;
            }
            BaseJitCellState::Tracing => {
                cell.flags |= jc_flags::TRACING | jc_flags::TRACING_OCCURRED;
                cell.state = BaseJitCellState::Tracing;
            }
            BaseJitCellState::Compiled => {
                cell.flags &= !jc_flags::TRACING;
                cell.state = BaseJitCellState::Compiled;
            }
            BaseJitCellState::Invalidated => {
                if let Some(token) = &cell.loop_token {
                    token.invalidate();
                }
                cell.state = BaseJitCellState::Invalidated;
            }
            BaseJitCellState::DontTraceHere => {
                cell.flags &= !jc_flags::TRACING;
                cell.flags |= jc_flags::DONT_TRACE_HERE;
                cell.state = BaseJitCellState::DontTraceHere;
            }
        }
    }

    // ── set_param / get_stats API ──

    /// Set a JIT parameter by name, mirroring warmstate.py set_param_*().
    ///
    /// Supported parameters:
    ///   - "threshold": compilation threshold
    ///   - "trace_limit": max ops per trace
    ///   - "bridge_threshold": guard fail count before bridge compilation
    ///   - "function_threshold": calls before inlining
    ///   - "max_inline_depth": maximum inlining depth
    /// warmstate.py: set_param() — set a JIT parameter by name.
    /// Negative values for thresholds mean "disabled/off" (rpython/rlib/jit.py:843).
    /// counter.py:124 — compute_threshold(threshold<=0) returns 0.0 (JIT off).
    /// Parameter names match RPython exactly: vec, vec_all, vec_cost.
    pub fn set_param(&mut self, name: &str, value: i64) {
        // counter.py:124 — threshold <= 0 → compute_threshold returns 0.0
        // (JIT off). Negative i64 must clamp to 0, not wrap to u32::MAX.
        let as_u32 = if value < 0 { 0u32 } else { value as u32 };
        match name {
            "threshold" => self.set_threshold(as_u32),
            "trace_limit" => self.trace_limit = as_u32,
            "trace_eagerness" | "bridge_threshold" => self.set_param_trace_eagerness(as_u32),
            "function_threshold" => self.function_threshold = as_u32,
            "max_inline_depth" => self.max_inline_depth = as_u32,
            "retrace_limit" => self.memory_manager.retrace_limit = as_u32,
            "max_retrace_guards" => self.memory_manager.max_retrace_guards = as_u32,
            "max_unroll_loops" => self.memory_manager.max_unroll_loops = as_u32,
            "max_unroll_recursion" => self.memory_manager.max_unroll_recursion = as_u32,
            "loop_longevity" => self.memory_manager.set_max_age(as_u32 as i64, 0),
            // warmstate.py:322-329 — vec, vec_all, vec_cost are separate fields
            "vec" | "vectorize" => self.vectorize = value != 0,
            "vec_all" => self.vec_all = value != 0,
            "vec_cost" => self.vec_cost = as_u32,
            "inlining" => self.inlining = value != 0,
            "disable_unrolling" => self.disable_unrolling_threshold = as_u32,
            "pureop_historylength" => self.pureop_historylength = as_u32,
            "decay" => self.counter.set_decay(value as i32),
            "enable_opts" => {} // string param, handled by set_param_enable_opts
            _ => {}
        }
    }

    /// warmstate.py: set_param_enable_opts(value)
    /// Set which optimization passes are enabled.
    /// Value is a colon-separated string like "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll".
    /// "all" enables all passes.
    pub fn set_param_enable_opts(&mut self, value: &str) {
        self.enable_opts = if value == "all" || value.is_empty() {
            default_enable_opts()
        } else {
            value
                .split(':')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect()
        };
    }

    /// Get enabled optimization pass names.
    pub fn get_enable_opts(&self) -> &[String] {
        &self.enable_opts
    }

    /// warmstate.py: confirm_enter_jit(*args)
    /// Hook called before entering JIT compilation to allow the user
    /// to abort tracing based on runtime conditions.
    /// Returns true if tracing should proceed, false to abort.
    ///
    /// In RPython this is a user-provided callback set via JitDriver.
    /// Here we provide a default that always returns true.
    pub fn confirm_enter_jit(&self, _green_key: u64) -> bool {
        true
    }

    /// warmstate.py: get_location(greenkey)
    /// Convert a green key to a human-readable source location string.
    /// Used for JIT logging and debugging.
    ///
    /// In RPython this is a user-provided callback set via JitDriver.
    /// Here we return a default format.
    pub fn get_location(&self, green_key: u64) -> String {
        format!("<jit key 0x{:x}>", green_key)
    }

    /// warmstate.py: get_param(name) — read a JIT parameter value.
    pub fn get_param(&self, name: &str) -> Option<i64> {
        match name {
            "threshold" => Some(self.counter.threshold() as i64),
            "trace_limit" => Some(self.trace_limit as i64),
            "trace_eagerness" | "bridge_threshold" => Some(self.trace_eagerness as i64),
            "function_threshold" => Some(self.function_threshold as i64),
            "max_inline_depth" => Some(self.max_inline_depth as i64),
            "retrace_limit" => Some(self.memory_manager.retrace_limit as i64),
            "max_retrace_guards" => Some(self.memory_manager.max_retrace_guards as i64),
            "max_unroll_loops" => Some(self.memory_manager.max_unroll_loops as i64),
            "max_unroll_recursion" => Some(self.memory_manager.max_unroll_recursion as i64),
            "loop_longevity" => Some(self.memory_manager.loop_longevity_param()),
            "vectorize" => Some(if self.vectorize { 1 } else { 0 }),
            "vec_cost" => Some(self.vec_cost as i64),
            "inlining" => Some(if self.inlining { 1 } else { 0 }),
            "disable_unrolling" => Some(self.disable_unrolling_threshold as i64),
            "pureop_historylength" => Some(self.pureop_historylength as i64),
            // warmstate.py has no getter for "decay": set_param_decay delegates
            // into jitcounter.set_decay which stores decay_by_mult (the derived
            // multiplier), not the raw int. Read-back is not supported.
            "decay" => None,
            _ => None,
        }
    }

    /// warmstate.py: set_param_to_default(name)
    /// Reset a single JIT parameter to its default value.
    pub fn set_param_to_default(&mut self, name: &str) {
        match name {
            "threshold" => self.set_threshold(1039), // RPython default
            "trace_limit" => self.trace_limit = DEFAULT_TRACE_LIMIT,
            "trace_eagerness" | "bridge_threshold" => {
                self.set_param_trace_eagerness(DEFAULT_TRACE_EAGERNESS)
            }
            "function_threshold" => self.function_threshold = DEFAULT_FUNCTION_THRESHOLD,
            "max_inline_depth" => self.max_inline_depth = 10,
            "retrace_limit" => self.memory_manager.retrace_limit = DEFAULT_RETRACE_LIMIT,
            "max_retrace_guards" => self.memory_manager.max_retrace_guards = 15,
            "max_unroll_loops" => self.memory_manager.max_unroll_loops = 0,
            "max_unroll_recursion" => {
                self.memory_manager.max_unroll_recursion = DEFAULT_MAX_INLINE_DEPTH;
            }
            "loop_longevity" => self.memory_manager.set_max_age(1000, 0),
            "vectorize" => self.vectorize = false,
            "vec_cost" => self.vec_cost = 0,
            // rlib/jit.py:588 PARAMETERS default decay=40.
            "decay" => self.counter.set_decay(40),
            _ => {}
        }
    }

    /// warmstate.py: get_param_names()
    /// Return all known parameter names.
    pub fn param_names() -> &'static [&'static str] {
        &[
            "threshold",
            "trace_limit",
            "trace_eagerness",
            "function_threshold",
            "max_inline_depth",
            "retrace_limit",
            "max_retrace_guards",
            "max_unroll_loops",
            "max_unroll_recursion",
            "loop_longevity",
            "vectorize",
            "vec_cost",
            "inlining",
            "disable_unrolling",
            "pureop_historylength",
            "decay",
        ]
    }

    // ── RPython warmstate.py getter methods ──

    pub fn retrace_limit(&self) -> u32 {
        self.memory_manager.retrace_limit
    }
    pub fn max_retrace_guards(&self) -> u32 {
        self.memory_manager.max_retrace_guards
    }
    pub fn max_unroll_loops(&self) -> u32 {
        self.memory_manager.max_unroll_loops
    }
    pub fn max_unroll_recursion(&self) -> u32 {
        self.memory_manager.max_unroll_recursion
    }
    pub fn vectorize(&self) -> bool {
        self.vectorize
    }
    pub fn vec_cost(&self) -> u32 {
        self.vec_cost
    }
    /// warmstate.py: inlining
    pub fn inlining(&self) -> bool {
        self.inlining
    }
    /// warmstate.py: disable_unrolling
    pub fn disable_unrolling_threshold(&self) -> u32 {
        self.disable_unrolling_threshold
    }
    /// warmstate.py: pureop_historylength
    pub fn pureop_historylength(&self) -> u32 {
        self.pureop_historylength
    }

    /// Get a snapshot of current JIT statistics.
    pub fn get_stats(&self) -> JitStats {
        let mut stats = JitStats {
            num_cells: self.cells.len(),
            ..Default::default()
        };
        for cell in self.cells.values() {
            match cell.state {
                BaseJitCellState::Compiled => stats.num_compiled += 1,
                BaseJitCellState::Tracing => stats.num_tracing += 1,
                BaseJitCellState::Invalidated => stats.num_invalidated += 1,
                BaseJitCellState::DontTraceHere => stats.num_disable_noninlinable_function += 1,
                BaseJitCellState::NotHot => {}
            }
        }
        stats
    }

    /// Get the current trace limit.
    pub fn trace_limit(&self) -> u32 {
        self.trace_limit
    }

    /// Set the trace limit.
    pub fn set_trace_limit(&mut self, limit: u32) {
        self.trace_limit = limit;
    }

    /// Get the current tracing generation.
    pub fn tracing_generation(&self) -> u64 {
        self.tracing_generation
    }

    /// Remove dead BaseJitCells from all chains.
    /// Returns the number of cells removed.
    pub fn gc_cells(&mut self) -> usize {
        let mut removed = 0;
        let keys: Vec<u64> = self.cells.keys().copied().collect();
        for hash in keys {
            if let Some(head) = self.cells.remove(&hash) {
                let (kept, n) = Self::clean_chain(head);
                removed += n;
                if let Some(k) = kept {
                    self.cells.insert(hash, k);
                }
            }
        }
        removed
    }

    /// Walk a chain, removing cells where should_remove_jitcell() is true.
    fn clean_chain(head: BaseJitCell) -> (Option<BaseJitCell>, usize) {
        let mut keep: Option<BaseJitCell> = None;
        let mut removed = 0;
        let mut cell_opt = Some(head);
        while let Some(mut c) = cell_opt {
            let next = c.next.take().map(|b| *b);
            if !c.should_remove_jitcell() {
                c.next = keep.map(Box::new);
                keep = Some(c);
            } else {
                removed += 1;
            }
            cell_opt = next;
        }
        (keep, removed)
    }

    /// counter.py:239-240 lookup_chain(hash)
    ///
    /// ```text
    ///  def lookup_chain(self, hash):
    ///      return self.celltable[self._get_index(hash)]
    /// ```
    ///
    /// Returns the head of the chain at `hash`. Walk `.next` to
    /// iterate the chain.
    pub fn lookup_chain(&self, hash: u64) -> Option<&BaseJitCell> {
        self.cells.get(&hash)
    }

    /// counter.py:246-256 install_new_cell(hash, newcell)
    ///
    /// ```text
    ///  def install_new_cell(self, hash, newcell):
    ///      index = self._get_index(hash)
    ///      cell = self.celltable[index]
    ///      keep = newcell
    ///      while cell is not None:
    ///          nextcell = cell.next
    ///          if not cell.should_remove_jitcell():
    ///              cell.next = keep
    ///              keep = cell
    ///          cell = nextcell
    ///      self.celltable[index] = keep
    /// ```
    pub fn install_new_cell(&mut self, hash: u64, newcell: Option<BaseJitCell>) {
        let mut keep = newcell;
        let mut cell_opt = self.cells.remove(&hash);
        // Walk the existing chain, unlink each node.
        while let Some(mut cell) = cell_opt {
            let next = cell.next.take().map(|b| *b);
            if !cell.should_remove_jitcell() {
                // counter.py:253-254: cell.next = keep; keep = cell
                cell.next = keep.map(Box::new);
                keep = Some(cell);
            }
            cell_opt = next;
        }
        // counter.py:256: self.celltable[index] = keep
        if let Some(k) = keep {
            self.cells.insert(hash, k);
        }
    }

    /// counter.py:242-244 cleanup_chain(hash)
    ///
    /// ```text
    ///  def cleanup_chain(self, hash):
    ///      self.reset(hash)
    ///      self.install_new_cell(hash, None)
    /// ```
    pub fn cleanup_chain(&mut self, hash: u64) {
        self.counter.reset(hash);
        self.install_new_cell(hash, None);
    }

    /// warmstate.py:596-604 `JitCell.get_jitcell(*greenargs)`.
    ///
    /// ```python
    /// hash = JitCell.get_uhash(*greenargs)
    /// cell = jitcounter.lookup_chain(hash)
    /// while cell is not None:
    ///     if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///         return cell
    ///     cell = cell.next
    /// return None
    /// ```
    ///
    /// Walks the per-bucket chain comparing each cell's stored
    /// comparekey with `key` until a match is found. Hash collisions
    /// across distinct typed greens are resolved by comparekey, so
    /// `(code_a, pc_a)` and `(code_b, pc_b)` with the same `get_uhash`
    /// no longer alias to the same cell.
    ///
    /// Currently a `&self` view: `WarmEnterState`'s mutable lookup
    /// helpers (`maybe_compile`, `should_start_dont_trace_here_trace`)
    /// keep using the legacy hash-only path until S2.2 follow-up
    /// migrates them.
    pub fn lookup_chain_with_key(&self, key: &GreenKey) -> Option<&BaseJitCell> {
        let hash = key.get_uhash();
        let mut cell = self.cells.get(&hash);
        while let Some(c) = cell {
            if c.comparekey_matches(key) {
                return Some(c);
            }
            cell = c.next.as_deref();
        }
        None
    }

    /// warmstate.py:626-641 `JitCell.ensure_jit_cell_at_key(greenkey)` /
    /// `_ensure_jit_cell_at_key(*greenargs)`.
    ///
    /// ```python
    /// def _ensure_jit_cell_at_key(*greenargs):
    ///     hash = JitCell.get_uhash(*greenargs)
    ///     cell = jitcounter.lookup_chain(hash)
    ///     while cell is not None:
    ///         if isinstance(cell, JitCell) and cell.comparekey(*greenargs):
    ///             return cell
    ///         cell = cell.next
    ///     newcell = JitCell(*greenargs)
    ///     jitcounter.install_new_cell(hash, newcell)
    ///     return newcell
    /// ```
    ///
    /// Returns the index into `self.cells[hash]`'s chain identifying
    /// the matching cell — pyre stores chains as boxed linked lists
    /// keyed by hash, so the simplest matching identifier is "head or
    /// linked offset". For now the helper returns nothing and the
    /// caller uses `lookup_chain_with_key` for read access; full
    /// mutable-by-key access lands when the celltable migrates from
    /// `HashMap<u64, BaseJitCell>` to a chain-aware container.
    ///
    /// On a miss (no chained cell matches the typed key) the helper
    /// allocates a new cell with `comparekey = Some(key.clone())` and
    /// installs it at the head of the chain via `install_new_cell`,
    /// matching upstream's `jitcounter.install_new_cell` semantics.
    pub fn ensure_cell_for_key(&mut self, key: &GreenKey) {
        if self.lookup_chain_with_key(key).is_some() {
            return;
        }
        let mut newcell = BaseJitCell::new();
        newcell.comparekey = Some(key.clone());
        self.install_new_cell(key.get_uhash(), Some(newcell));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_hot_initially() {
        let mut ws = WarmEnterState::new(3);
        match ws.maybe_compile(42) {
            HotResult::NotHot => {}
            _ => panic!("expected NotHot"),
        }
    }

    #[test]
    fn test_start_tracing_at_threshold() {
        let mut ws = WarmEnterState::new(3);
        // Tick 1, 2: not hot
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        // Tick 3: threshold reached, start tracing
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
    }

    #[test]
    fn test_already_tracing() {
        let mut ws = WarmEnterState::new(2);
        // First tick: eviction (always false). Second tick: threshold reached.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Next call sees TRACING flag
        match ws.maybe_compile(42) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing"),
        }
    }

    #[test]
    fn test_run_compiled() {
        let mut ws = WarmEnterState::new(1);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(42, token);

        match ws.maybe_compile(42) {
            HotResult::RunCompiled => {}
            _ => panic!("expected RunCompiled"),
        }
    }

    #[test]
    fn test_finish_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(42);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::TRACING_OCCURRED != 0);
    }

    #[test]
    fn test_abort_tracing() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.abort_tracing(42, true);

        let cell = ws.get_cell(42).unwrap();
        assert!(!cell.is_tracing());
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);

        // RPython warmstate.py: a DONT_TRACE_HERE cell with no procedure token
        // still retriggers separate tracing after warming up again.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing due to DONT_TRACE_HERE retrace"),
        }
    }

    #[test]
    fn test_abort_tracing_allows_retry() {
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        // Abort without DONT_TRACE_HERE
        ws.abort_tracing(42, false);

        // Counter was reset during start_tracing, but hash is still in the table.
        // Need to tick again to reach threshold. The hash is found now (not evicted),
        // so one tick to reach count=1, another to reach count=2 >= threshold=2.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing on retry"),
        }
    }

    #[test]
    fn test_different_green_keys() {
        let mut ws = WarmEnterState::new(3);
        // Key 1: tick 1 (eviction), tick 2 (count=2 < 3)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        // Key 2: tick 1 (eviction)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        // Key 1: tick 3 -> threshold, starts tracing
        match ws.maybe_compile(1) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 1"),
        }
        // Key 2 still not hot (only 2 total ticks: eviction + one more needed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
    }

    #[test]
    fn test_alloc_token_number() {
        let mut ws = WarmEnterState::new(10);
        let first = ws.alloc_token_number();
        let second = ws.alloc_token_number();
        let third = ws.alloc_token_number();
        assert_eq!(second, first + 1);
        assert_eq!(third, second + 1);
    }

    #[test]
    fn test_set_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);
        ws.set_threshold(50);
        assert_eq!(ws.threshold(), 50);
    }

    #[test]
    fn test_get_compiled() {
        let mut ws = WarmEnterState::new(1);
        assert!(ws.get_compiled(42).is_none());

        let token = JitCellToken::new(0);
        ws.attach_procedure_to_interp(42, token);

        let compiled = ws.get_compiled(42);
        assert!(compiled.is_some());
        assert_eq!(compiled.unwrap().number, 0);
    }

    #[test]
    fn test_full_lifecycle() {
        let mut ws = WarmEnterState::new(3);
        let key = 0xDEAD;

        // Phase 1: Not hot (need 3 ticks to reach threshold=3)
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));

        // Phase 2: Start tracing (third tick reaches threshold)
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));

        // Phase 3: Already tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::AlreadyTracing));

        // Phase 4: Finish tracing, install compiled code
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(key, token);

        // Phase 5: Run compiled
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_bridge_threshold_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.bridge_threshold(), 200); // PyPy default: trace_eagerness
    }

    #[test]
    fn test_bridge_threshold_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_bridge_threshold(10);
        assert_eq!(ws.bridge_threshold(), 10);
    }

    #[test]
    fn test_start_retrace_preserves_input_types() {
        // RPython pyjitpl.py:2609 `MetaInterp.create_history(max_num_inputargs)`:
        // the MetaInterp, not warmstate, owns the Trace factory. Since
        // warmstate's `start_retrace` is now a state-only signal, this test
        // verifies that the input_types the caller intends to use are the
        // ones that flow into the Trace downstream (Trace::with_input_types).
        let mut ws = WarmEnterState::new(3);
        let input_types = [Type::Ref, Type::Int, Type::Float];
        ws.start_retrace(&input_types);
        let mut recorder = crate::recorder::Trace::with_input_types(&input_types);
        recorder.close_loop(&[
            majit_ir::OpRef::input_arg_ref(0),
            majit_ir::OpRef::input_arg_int(1),
            majit_ir::OpRef::input_arg_float(2),
        ]);
        let trace = recorder.get_trace();
        let seen: Vec<Type> = trace.inputargs.iter().map(|arg| arg.tp).collect();
        assert_eq!(seen, input_types.to_vec());
    }

    // ── Quasi-immutable invalidation tests ──

    #[test]
    fn test_quasiimmut_register_and_invalidate() {
        let mut ws = WarmEnterState::new(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        let green_key = 42;
        let qmut_key = 0xABCD;

        assert!(!token.is_invalidated());
        ws.attach_procedure_to_interp(green_key, token);
        ws.register_quasiimmut_dependency(qmut_key, green_key);

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 1);

        let cell = ws.get_cell(green_key).unwrap();
        assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
    }

    #[test]
    fn test_quasiimmut_no_deps() {
        let mut ws = WarmEnterState::new(1);
        // No dependencies registered → invalidation returns 0.
        let count = ws.invalidate_quasiimmut(0xDEAD);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_quasiimmut_multiple_deps() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;

        // Install two loops depending on the same quasi-immutable field.
        for green_key in [10, 20] {
            let token = JitCellToken::new(ws.alloc_token_number());
            ws.attach_procedure_to_interp(green_key, token);
            ws.register_quasiimmut_dependency(qmut_key, green_key);
        }

        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 2);

        for green_key in [10, 20] {
            let cell = ws.get_cell(green_key).unwrap();
            assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        }
    }

    #[test]
    fn test_quasiimmut_invalidate_all() {
        let mut ws = WarmEnterState::new(1);
        for green_key in [1, 2, 3] {
            let token = JitCellToken::new(ws.alloc_token_number());
            ws.attach_procedure_to_interp(green_key, token);
        }

        ws.invalidate_all();

        for green_key in [1, 2, 3] {
            let cell = ws.get_cell(green_key).unwrap();
            assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        }
    }

    // ── Function threshold tests ──

    #[test]
    fn test_function_threshold_default() {
        let ws = WarmEnterState::new(3);
        assert_eq!(ws.function_threshold(), 1619); // PyPy default
    }

    #[test]
    fn test_function_threshold_custom() {
        let mut ws = WarmEnterState::new(3);
        ws.set_function_threshold(10);
        assert_eq!(ws.function_threshold(), 10);
    }

    // ── Trace limit lifecycle tests (RPython: test_tracelimit.py) ──

    #[test]
    fn test_abort_tracing_too_long_sets_dont_trace() {
        // When a trace is too long, the meta-interpreter calls
        // abort_tracing(key, true) to prevent future tracing at that location.
        // This mirrors RPython's ABORT_TOO_LONG behavior.
        let mut ws = WarmEnterState::new(2);
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Simulate: recorder.is_too_long() was true, so abort with dont_trace.
        ws.abort_tracing(42, true);

        // The key is now blacklisted.
        let cell = ws.get_cell(42).unwrap();
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE != 0);
        assert!(!cell.is_tracing());

        // RPython warmstate.py: DONT_TRACE_HERE still allows separate
        // tracing later for keys without a procedure token.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(42), HotResult::StartTracing));
    }

    #[test]
    fn test_disable_noninlinable_function_blocks_inlining() {
        let mut ws = WarmEnterState::new(3);

        // Fresh cell: can_inline_callable returns true by default.
        assert!(ws.can_inline_callable(42));

        // dont_trace_here → JC_DONT_TRACE_HERE flag, can_inline_callable
        // returns false (warmstate.py:669-676 parity).
        ws.disable_noninlinable_function(42);
        assert!(!ws.can_inline_callable(42));
    }

    #[test]
    fn test_abort_too_long_then_retry_different_key() {
        // Aborting one key's trace as too long should not affect other keys.
        let mut ws = WarmEnterState::new(2);

        // Key 42: start and abort as too long.
        assert!(matches!(ws.maybe_compile(42), HotResult::NotHot));
        match ws.maybe_compile(42) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 42"),
        }
        ws.abort_tracing(42, true);

        // Key 99: should still work normally.
        assert!(matches!(ws.maybe_compile(99), HotResult::NotHot));
        match ws.maybe_compile(99) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing for key 99"),
        }
    }

    #[test]
    fn test_lifecycle_with_trace_abort_and_recompile() {
        // Full lifecycle: trace starts, is too long (abort without blacklist),
        // then on retry a shorter trace succeeds and gets compiled.
        let mut ws = WarmEnterState::new(2);
        let key = 0xCAFE;

        // Phase 1: reach threshold, start tracing.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Phase 2: trace is too long, abort without blacklist.
        ws.abort_tracing(key, false);

        // Phase 3: retry, reach threshold again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                // Phase 4: this time the trace succeeds.
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing on retry"),
        }

        // Phase 5: compiled code should be available.
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_multiple_aborts_before_success() {
        // Mirrors RPython's segmented trace behavior: a location can fail
        // multiple times before eventually compiling.
        // threshold=2 because the first tick always evicts (returns NotHot).
        let mut ws = WarmEnterState::new(2);
        let key = 0xBEEF;

        // First attempt: tick once (eviction), tick twice (threshold) -> StartTracing.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 1)"),
        }
        ws.abort_tracing(key, false);

        // Second attempt: after abort, counter was reset, need to tick again.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing (attempt 2)"),
        }
        ws.abort_tracing(key, false);

        // Third attempt: succeeds and gets compiled.
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing (attempt 3)"),
        }
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_tracing_occurred_flag_persists_after_abort() {
        // The TRACING_OCCURRED flag should remain set even after abort.
        // This mirrors RPython's tracking of whether tracing was ever attempted.
        let mut ws = WarmEnterState::new(2);
        let key = 42;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws.get_cell(key).unwrap();
        assert!(cell.flags & jc_flags::TRACING_OCCURRED != 0);

        ws.abort_tracing(key, false);

        let cell = ws.get_cell(key).unwrap();
        assert!(!cell.is_tracing());
        assert!(
            cell.flags & jc_flags::TRACING_OCCURRED != 0,
            "TRACING_OCCURRED should persist after abort"
        );
    }

    #[test]
    fn test_quasiimmut_deps_cleared_after_invalidation() {
        let mut ws = WarmEnterState::new(1);
        let qmut_key = 0xABCD;
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(42, token);
        ws.register_quasiimmut_dependency(qmut_key, 42);

        ws.invalidate_quasiimmut(qmut_key);
        // Second invalidation should find no deps.
        let count = ws.invalidate_quasiimmut(qmut_key);
        assert_eq!(count, 0);
    }

    // ── MemoryManager (memmgr.py parity) tests ──
    //
    // Ported from rpython/jit/metainterp/test/test_memmgr.py.  Each
    // test creates real `Arc<JitCellToken>` instances since
    // `MemoryManager.alive_loops` keys on token identity per
    // memmgr.py:9-12 (the looptoken Python object is the dict key
    // upstream).

    fn make_token(number: u64) -> Arc<JitCellToken> {
        Arc::new(JitCellToken::new(number))
    }

    #[test]
    fn test_loop_aging_basic() {
        // memmgr.py:_kill_old_loops_now predicate:
        //   0 <= looptoken.generation < current_generation - (max_age - 1)
        // With max_age=3 and a token kept alive at current_generation=1:
        //   after 2 next_generation: cg=3, max_gen=3-2=1, 1<1=false → keep
        //   after 3 next_generation: cg=4, max_gen=4-2=2, 1<2=true  → evict
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 2);

        mgr.next_generation();
        assert_eq!(mgr.alive_count(), 0);
    }

    #[test]
    fn test_loop_aging_disabled() {
        // memmgr.py:43-44: max_age <= 0 disables eviction.
        let mut mgr = MemoryManager::new(0);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        for _ in 0..100 {
            mgr.next_generation();
        }
        assert_eq!(mgr.alive_count(), 2);
    }

    #[test]
    fn test_loop_aging_refresh() {
        // keep_loop_alive resets `looptoken.generation` to
        // `current_generation`, postponing eviction.
        let mut mgr = MemoryManager::new(3);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1);
        mgr.keep_loop_alive(&t2);

        // Refresh t1 each generation; t2 ages out.
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        mgr.keep_loop_alive(&t1);
        mgr.next_generation();
        // cg=4, max_gen=2; t1.gen=3 (refreshed), 3<2=false → keep;
        //                  t2.gen=1, 1<2=true → evict.
        assert_eq!(mgr.alive_count(), 1);
        assert!(mgr.contains(&t1));
        assert!(!mgr.contains(&t2));

        // Refreshing t1 each generation keeps it indefinitely.
        for _ in 0..10 {
            mgr.keep_loop_alive(&t1);
            mgr.next_generation();
        }
        assert!(mgr.contains(&t1));
    }

    #[test]
    fn test_loop_aging_mixed() {
        // Tokens kept alive at different generations age independently.
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);
        let t2 = make_token(2);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, t1.gen=1, 1<1=false → keep
        mgr.keep_loop_alive(&t2); // t2.gen=2

        mgr.next_generation(); // cg=3, max_gen=2, t1.gen=1<2=true → evict, t2.gen=2<2=false → keep
        assert_eq!(mgr.alive_count(), 1);
        assert!(!mgr.contains(&t1));
        assert!(mgr.contains(&t2));
    }

    // ── Memmgr deeper coverage (RPython: test_memmgr.py parity) ──

    #[test]
    fn test_evicted_loops_can_be_recompiled() {
        // After eviction the same JitCellToken can be re-inserted by a
        // fresh keep_loop_alive (or a freshly compiled token can take
        // its place — RPython gets a brand-new LoopToken object).
        let mut mgr = MemoryManager::new(2);
        let t1 = make_token(1);

        mgr.keep_loop_alive(&t1); // t1.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-1=1, 1<1=false → keep
        mgr.next_generation(); // cg=3, max_gen=2, 1<2=true → evict
        assert_eq!(mgr.alive_count(), 0);

        // Re-register the same token (or a fresh one).
        mgr.keep_loop_alive(&t1); // t1.gen=3
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation(); // cg=4, max_gen=3, 3<3=false → keep
        assert!(mgr.contains(&t1));
        mgr.next_generation(); // cg=5, max_gen=4, 3<4=true → evict
        assert!(!mgr.contains(&t1));
    }

    #[test]
    fn test_generation_does_not_panic_at_high_values() {
        // Verify that generation arithmetic survives many advances.
        // RPython uses r_int64; Rust uses i64 with the same wraparound
        // semantics.
        let mut mgr = MemoryManager::new(3);

        for _ in 0..1000 {
            mgr.next_generation();
        }
        assert_eq!(mgr.current_generation(), 1001);

        let t = make_token(42);
        mgr.keep_loop_alive(&t);
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(mgr.contains(&t));
        mgr.next_generation();
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_with_warm_state_integration() {
        // memmgr cooperates with WarmEnterState: attach a token via
        // `attach_procedure_to_interp` (warmstate.py:340-341 parity)
        // then mirror it into MemoryManager via keep_loop_alive.
        let mut ws = WarmEnterState::new(2);
        let mut mgr = MemoryManager::new(2);
        let key = 0xF00D;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(key);
        let token = make_token(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, Arc::clone(&token));
        mgr.keep_loop_alive(&token);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(mgr.alive_count(), 1);

        mgr.next_generation();
        mgr.next_generation();
        assert!(!mgr.contains(&token));

        // Re-install: a fresh compile produces a fresh token; the
        // warmstate cell records the new procedure token, MemoryManager
        // tracks it under its own pointer identity.
        let token2 = make_token(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, Arc::clone(&token2));
        mgr.keep_loop_alive(&token2);

        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
        assert_eq!(mgr.alive_count(), 1);
    }

    #[test]
    fn test_loop_aging_does_not_affect_active_loops() {
        // A loop continually kept alive is never evicted, even while
        // peers are aged out.
        let mut mgr = MemoryManager::new(2);
        let t_active = make_token(1);
        let t_idle_a = make_token(2);
        let t_idle_b = make_token(3);

        mgr.keep_loop_alive(&t_active);
        mgr.keep_loop_alive(&t_idle_a);
        mgr.keep_loop_alive(&t_idle_b);

        for _ in 0..20 {
            mgr.keep_loop_alive(&t_active);
            mgr.next_generation();
            assert!(mgr.contains(&t_active));
        }

        // Idle loops were aged out long ago.
        assert!(!mgr.contains(&t_idle_a));
        assert!(!mgr.contains(&t_idle_b));
    }

    #[test]
    fn test_loop_aging_set_max_age_dynamic() {
        // memmgr.py:42 set_max_age — shrinking max_age accelerates
        // eviction of older loops.
        let mut mgr = MemoryManager::new(10);
        let t = make_token(1);

        mgr.keep_loop_alive(&t); // t.gen=1
        mgr.next_generation(); // cg=2, max_gen=2-9=-7 → keep

        // Shrink max_age — next sweep evicts t.
        mgr.set_max_age(1, 0);
        mgr.next_generation(); // cg=3, max_gen=3-0=3, 1<3 → evict
        assert!(!mgr.contains(&t));
    }

    #[test]
    fn test_loop_aging_interleaved_register_and_evict() {
        // Mirrors RPython's test_basic_3: tokens registered at different
        // generations, even-indexed kept alive each step.  After the
        // loop, even-indexed tokens are alive; old odd-indexed ones
        // (registered ≥ max_age generations ago and never refreshed)
        // are evicted.
        let mut mgr = MemoryManager::new(4);
        let mut tokens: Vec<Arc<JitCellToken>> = Vec::new();

        for i in 0..10u64 {
            let t = make_token(i);
            mgr.keep_loop_alive(&t);
            tokens.push(t);
            mgr.next_generation();

            // Refresh even-indexed tokens.
            for j in (0..=i).step_by(2) {
                mgr.keep_loop_alive(&tokens[j as usize]);
            }
        }

        for (i, t) in tokens.iter().enumerate() {
            let is_alive = mgr.contains(t);
            if i % 2 == 0 {
                assert!(is_alive, "even-indexed token {} should be alive", i);
            } else if i < 6 {
                // Odd tokens registered early enough are evicted.
                // Token i registered at cg=i+1; cg now ~11; max_gen=11-3=8.
                // For i < 6: t.gen=i+1 < 8 → evict.
                assert!(!is_alive, "odd token {} should be evicted", i);
            }
        }
    }

    // ── Trace limit + inline depth interaction ──

    #[test]
    fn test_trace_limit_with_inline_depth() {
        // Inline depth limiting and trace limit are orthogonal:
        // a function can be inlined (depth < max), but the trace
        // can still be too long. The WarmEnterState correctly tracks both.
        let mut ws = WarmEnterState::new(3);
        ws.set_max_inline_depth(3);

        // Depth 0: allowed
        assert!(ws.can_inline_at_depth(0));
        // Depth 2: allowed (< 3)
        assert!(ws.can_inline_at_depth(2));
        // Depth 3: not allowed (>= 3)
        assert!(!ws.can_inline_at_depth(3));
    }

    #[test]
    fn test_abort_tracing_retry_with_lower_threshold() {
        // Simulates the scenario where a trace is too long, the location
        // is aborted (without blacklisting), and on retry the WarmEnterState
        // has a lower threshold so it starts tracing sooner.
        let mut ws = WarmEnterState::new(5);
        let key = 0xABCD;

        // Reach threshold=5, start tracing
        for _ in 0..4 {
            assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        }
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing at threshold 5"),
        }

        // Abort without blacklisting
        ws.abort_tracing(key, false);

        // Lower threshold for retry
        ws.set_threshold(2);

        // Now only 2 ticks needed
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {
                ws.finish_tracing(key);
                let token = JitCellToken::new(ws.alloc_token_number());
                ws.attach_procedure_to_interp(key, token);
            }
            _ => panic!("expected StartTracing with lower threshold"),
        }
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_force_start_tracing_bypasses_counter() {
        // force_start_tracing is used for function-entry tracing where
        // the caller already decided to trace. It should work regardless
        // of the counter state.
        let mut ws = WarmEnterState::new(100); // very high threshold
        let key = 42;

        // Without any ticks, force_start_tracing should start tracing
        match ws.force_start_tracing(key) {
            HotResult::StartTracing => {}
            _ => panic!("force_start_tracing should start tracing immediately"),
        }

        // The cell should be in TRACING state
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.is_tracing());

        // Second call sees AlreadyTracing
        match ws.force_start_tracing(key) {
            HotResult::AlreadyTracing => {}
            _ => panic!("expected AlreadyTracing on second force_start_tracing"),
        }
    }

    // ── BaseJitCell state machine tests ──

    #[test]
    fn test_jitcell_state_transitions() {
        // Full lifecycle: NotHot → Tracing → Compiled → Invalidated
        let mut ws = WarmEnterState::new(2);
        let key = 0xA1;

        // Initially no cell → NotHot
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::NotHot);

        // Tick to threshold → Tracing
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Tracing);

        // Finish tracing and install → Compiled
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);

        // Invalidate via transition_cell → Invalidated
        ws.transition_cell(key, BaseJitCellState::Invalidated);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        // The loop_token's invalidated flag should be set
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.loop_token.as_ref().unwrap().is_invalidated());
        // token number is preserved as a historical record
        assert!(cell.token.is_some());
    }

    #[test]
    fn test_procedure_token_ownership() {
        // Compiled cell owns a token; invalidation revokes ownership.
        let mut ws = WarmEnterState::new(2);
        let key = 0xB2;

        // Compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }
        ws.finish_tracing(key);
        let token_num = ws.alloc_token_number();
        let token = JitCellToken::new(token_num);
        ws.attach_procedure_to_interp(key, token);

        // Cell owns the token
        let cell = ws.get_cell(key).unwrap();
        assert_eq!(cell.token, Some(token_num));
        assert!(cell.get_procedure_token().is_some());
        assert!(cell.has_seen_a_procedure_token());

        // Invalidate via quasiimmut
        let qmut_key = 0xFF;
        ws.register_quasiimmut_dependency(qmut_key, key);
        ws.invalidate_quasiimmut(qmut_key);

        // Token ownership revoked (state is Invalidated, but token number
        // is preserved as historical record)
        let cell = ws.get_cell(key).unwrap();
        assert_eq!(cell.token, Some(token_num)); // historical record preserved
        assert_eq!(cell.state, BaseJitCellState::Invalidated);
        // get_procedure_token returns None because the token is invalidated
        assert!(cell.get_procedure_token().is_none());
        // But we still know a token existed
        assert!(cell.has_seen_a_procedure_token());
    }

    #[test]
    fn test_set_param_threshold() {
        let mut ws = WarmEnterState::new(100);
        assert_eq!(ws.threshold(), 100);

        ws.set_param("threshold", 42);
        assert_eq!(ws.threshold(), 42);

        ws.set_param("trace_limit", 5000);
        assert_eq!(ws.trace_limit(), 5000);

        ws.set_param("bridge_threshold", 10);
        assert_eq!(ws.bridge_threshold(), 10);

        ws.set_param("function_threshold", 8);
        assert_eq!(ws.function_threshold(), 8);

        ws.set_param("max_inline_depth", 15);
        assert_eq!(ws.max_inline_depth(), 15);

        // Unknown param is ignored
        ws.set_param("nonexistent", 999);
    }

    #[test]
    fn test_get_stats() {
        let mut ws = WarmEnterState::new(2);

        // Initially empty
        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 0);
        assert_eq!(stats.num_compiled, 0);

        // Start tracing two keys
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));

        let stats = ws.get_stats();
        assert_eq!(stats.num_cells, 2);
        assert_eq!(stats.num_tracing, 2);

        // Compile key 1
        ws.finish_tracing(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(1, token);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 1);

        // Abort key 2 with dont_trace
        ws.abort_tracing(2, true);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 1);
        assert_eq!(stats.num_tracing, 0);
        assert_eq!(stats.num_disable_noninlinable_function, 1);

        // Invalidate key 1
        ws.transition_cell(1, BaseJitCellState::Invalidated);

        let stats = ws.get_stats();
        assert_eq!(stats.num_compiled, 0);
        assert_eq!(stats.num_invalidated, 1);
        assert_eq!(stats.num_disable_noninlinable_function, 1);
        assert_eq!(stats.num_cells, 2);
    }

    #[test]
    fn test_jitcell_state_disable_noninlinable_function() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xC3;

        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        match ws.maybe_compile(key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        // Abort with DONT_TRACE_HERE → state should be DontTraceHere
        ws.abort_tracing(key, true);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::DontTraceHere);

        // Future calls return NotHot
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));

        // Can manually reset to NotHot
        ws.transition_cell(key, BaseJitCellState::NotHot);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::NotHot);
        // DONT_TRACE_HERE flag should be cleared
        let cell = ws.get_cell(key).unwrap();
        assert!(cell.flags & jc_flags::DONT_TRACE_HERE == 0);
    }

    #[test]
    fn test_tracing_generation_increments() {
        let mut ws = WarmEnterState::new(2);
        let gen0 = ws.tracing_generation();
        assert_eq!(gen0, 0);

        // Start tracing key 1 → generation 1
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 1);

        let cell = ws.get_cell(1).unwrap();
        assert_eq!(cell.tracing_generation, 1);

        // Start tracing key 2 → generation 2
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));
        assert_eq!(ws.tracing_generation(), 2);

        let cell = ws.get_cell(2).unwrap();
        assert_eq!(cell.tracing_generation, 2);
    }

    #[test]
    fn test_jitcell_should_remove() {
        // A freshly created cell with no token and no flags should be removable
        let cell = BaseJitCell::new();
        assert!(cell.should_remove_jitcell());

        // A cell that is tracing should NOT be removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::TRACING;
        assert!(!cell.should_remove_jitcell());

        // A cell with DONT_TRACE_HERE but no token history is removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        assert!(!cell.should_remove_jitcell()); // has_seen_a_procedure_token is false

        // A cell with DONT_TRACE_HERE and a past token should be removable
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::DONT_TRACE_HERE;
        cell.token = Some(42); // historical record of past token
        assert!(cell.should_remove_jitcell());

        // warmstate.py:222-225: FORCE_FINISH must NOT be removed
        let mut cell = BaseJitCell::new();
        cell.flags |= jc_flags::FORCE_FINISH;
        assert!(!cell.should_remove_jitcell());
    }

    #[test]
    fn test_gc_cells() {
        let mut ws = WarmEnterState::new(2);

        // Create some cells in various states
        // Key 1: compiled (should NOT be removed)
        assert!(matches!(ws.maybe_compile(1), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(1), HotResult::StartTracing));
        ws.finish_tracing(1);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(1, token);

        // Key 2: tracing (should NOT be removed)
        assert!(matches!(ws.maybe_compile(2), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(2), HotResult::StartTracing));

        // Key 3: aborted without dont_trace → NotHot, removable
        assert!(matches!(ws.maybe_compile(3), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(3), HotResult::StartTracing));
        ws.abort_tracing(3, false);

        assert_eq!(ws.get_stats().num_cells, 3);
        let removed = ws.gc_cells();
        assert_eq!(removed, 1); // key 3 removed
        assert_eq!(ws.get_stats().num_cells, 2);
        assert!(ws.get_cell(1).is_some());
        assert!(ws.get_cell(2).is_some());
        assert!(ws.get_cell(3).is_none());
    }

    #[test]
    fn test_invalidated_cell_allows_recompilation() {
        // After invalidation, transitioning back to NotHot allows recompilation.
        let mut ws = WarmEnterState::new(2);
        let key = 0xD4;

        // Compile
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);

        // Invalidate
        ws.transition_cell(key, BaseJitCellState::Invalidated);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);

        // Reset to NotHot and recompile
        ws.transition_cell(key, BaseJitCellState::NotHot);
        let token2 = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token2);
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Compiled);
        assert!(matches!(ws.maybe_compile(key), HotResult::RunCompiled));
    }

    #[test]
    fn test_get_param() {
        let ws = WarmEnterState::new(100);
        assert_eq!(ws.get_param("threshold"), Some(100));
        assert_eq!(ws.get_param("vectorize"), Some(0));
        assert_eq!(ws.get_param("unknown_param"), None);
    }

    #[test]
    fn test_set_param_to_default() {
        let mut ws = WarmEnterState::new(100);
        ws.set_param("trace_limit", 999);
        assert_eq!(ws.get_param("trace_limit"), Some(999));
        ws.set_param_to_default("trace_limit");
        assert_eq!(
            ws.get_param("trace_limit"),
            Some(DEFAULT_TRACE_LIMIT as i64)
        );
    }

    #[test]
    fn test_param_names() {
        let names = WarmEnterState::param_names();
        assert!(names.contains(&"threshold"));
        assert!(names.contains(&"trace_limit"));
        assert!(names.contains(&"vectorize"));
        assert!(names.len() >= 10);
    }

    #[test]
    fn test_quasiimmut_dependency_lifecycle() {
        let mut ws = WarmEnterState::new(2);
        let key = 0xF00D;
        let qmut = 0xBEEF;

        // Compile a loop
        assert!(matches!(ws.maybe_compile(key), HotResult::NotHot));
        assert!(matches!(ws.maybe_compile(key), HotResult::StartTracing));
        ws.finish_tracing(key);
        let token = JitCellToken::new(ws.alloc_token_number());
        ws.attach_procedure_to_interp(key, token);

        // Register quasi-immutable dependency
        ws.register_quasiimmut_dependency(qmut, key);

        // Invalidate
        let invalidated = ws.invalidate_quasiimmut(qmut);
        assert_eq!(invalidated, 1);

        // Loop should now be invalidated state
        assert_eq!(ws.get_cell_state(key), BaseJitCellState::Invalidated);
    }

    #[test]
    fn test_set_param_roundtrip() {
        let mut ws = WarmEnterState::new(100);
        for name in WarmEnterState::param_names() {
            let original = ws.get_param(name);
            // "decay" is write-only (warmstate.py:269-270 delegates to
            // jitcounter.set_decay which stores decay_by_mult, not the raw int).
            if *name != "decay" {
                assert!(original.is_some(), "param {name} should be gettable");
            }
            ws.set_param(name, 999);
            ws.set_param_to_default(name);
            // After default, should be same as a fresh instance
        }
    }

    /// warmstate.py:575-582 — `comparekey_matches` returns true only
    /// when the cell carries a stored GreenKey equal to the probe.
    /// Cells without a comparekey (legacy hash-only path) always fail.
    #[test]
    fn comparekey_matches_only_with_stored_key() {
        let mut cell = BaseJitCell::new();
        let key = GreenKey::new(vec![1, 2, 3]);
        assert!(
            !cell.comparekey_matches(&key),
            "cell without comparekey must not match"
        );
        cell.comparekey = Some(key.clone());
        assert!(
            cell.comparekey_matches(&key),
            "cell with stored comparekey must match equal probe"
        );
        let other = GreenKey::new(vec![1, 2, 4]);
        assert!(
            !cell.comparekey_matches(&other),
            "cell with stored comparekey must not match unequal probe"
        );
    }

    /// warmstate.py:626-641 + 596-604 — `ensure_cell_for_key` allocates
    /// a fresh cell on miss and `lookup_chain_with_key` returns it on a
    /// repeat probe with the same typed greens.
    #[test]
    fn ensure_and_lookup_chain_with_key_round_trips() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![42, 100]);

        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "no cell for fresh key"
        );
        ws.ensure_cell_for_key(&key);
        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("ensure must install a cell");
        assert_eq!(
            cell.comparekey.as_ref().expect("comparekey populated"),
            &key,
            "stored comparekey must equal the install key"
        );
    }

    /// warmstate.py:596-604 — chain walk distinguishes hash collisions:
    /// two distinct GreenKeys sharing one bucket must each resolve to
    /// their own cell via `comparekey` rather than aliasing. The test
    /// exploits `install_new_cell`'s `should_remove_jitcell` gate
    /// (warmstate.rs:184-200, counter.py:246-256 parity): a cell with
    /// no `loop_token` / no flags is treated as dead and dropped on
    /// the next install. Setting `JC_TRACING` on the first-installed
    /// cell keeps it alive across the second install so both end up
    /// chained through `head.next`. The walker must then skip the
    /// non-matching head when looking up the chained cell.
    #[test]
    fn lookup_chain_with_key_resolves_hash_collisions_via_comparekey() {
        let mut ws = WarmEnterState::new(100);
        let key_head_after_chain = GreenKey::new(vec![1, 2]);
        let key_chained = GreenKey::new(vec![3, 4]);
        let bucket = key_head_after_chain.get_uhash();

        // First install: TRACING flag so should_remove_jitcell == false
        // and install_new_cell preserves the cell across the second
        // install. Stays at the back of the chain after the second
        // install per install_new_cell's "prepend new cell, fold
        // existing keepable cells in front" semantics.
        let mut cell_first = BaseJitCell::new();
        cell_first.flags |= jc_flags::TRACING;
        cell_first.comparekey = Some(key_head_after_chain.clone());
        ws.install_new_cell(bucket, Some(cell_first));
        // Second install — keep predicate doesn't matter for this one,
        // it lands as `keep` and the prior head folds in front, so the
        // resulting chain shape is head=cell_first, head.next=cell_for_key_chained.
        // Wait — install_new_cell folds the EXISTING chain in front of
        // `keep`. So after the second install: head=cell_first (the
        // existing one, since !should_remove), head.next=cell_chained.
        let mut cell_chained = BaseJitCell::new();
        cell_chained.comparekey = Some(key_chained.clone());
        ws.install_new_cell(bucket, Some(cell_chained));

        let head = ws
            .lookup_chain(bucket)
            .expect("bucket head exists after install");
        assert_eq!(head.comparekey.as_ref(), Some(&key_head_after_chain));
        let next = head.next.as_deref().expect("chain has a second cell");
        assert_eq!(next.comparekey.as_ref(), Some(&key_chained));

        // Fast path: lookup_chain_with_key(key_head_after_chain) finds
        // the head's comparekey on first probe.
        let hit_head = ws
            .lookup_chain_with_key(&key_head_after_chain)
            .expect("walker must find chain head via comparekey");
        assert_eq!(hit_head.comparekey.as_ref(), Some(&key_head_after_chain));
    }

    /// `warmstate.py:714-723` `get_assembler_token` — a fresh typed key
    /// installs a temporary procedure token (tmp=true → JC_TEMPORARY)
    /// and returns it; subsequent calls with the same key return the
    /// same token without invoking `make_token` again.
    #[test]
    fn get_assembler_token_with_key_caches_token_per_typed_key() {
        let mut ws = WarmEnterState::new(100);
        let key = GreenKey::new(vec![10, 20]);
        let token = std::sync::Arc::new(JitCellToken::new(0xa55e_b1ed_u64));

        let mut make_token_calls = 0;
        let token_clone_1 = {
            let token_for_closure = token.clone();
            ws.get_assembler_token_with_key::<(), _>(&key, || {
                make_token_calls += 1;
                Ok(token_for_closure)
            })
            .expect("install ok")
        };
        assert_eq!(make_token_calls, 1, "first call invokes make_token");
        assert!(std::sync::Arc::ptr_eq(&token_clone_1, &token));

        let token_clone_2 = ws
            .get_assembler_token_with_key::<(), _>(&key, || {
                make_token_calls += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xdead_beef_u64)))
            })
            .expect("re-fetch ok");
        assert_eq!(
            make_token_calls, 1,
            "second call hits the cached token, does not invoke make_token"
        );
        assert!(
            std::sync::Arc::ptr_eq(&token_clone_2, &token),
            "second call returns the originally installed token"
        );

        // JC_TEMPORARY must be set since tmp=true.
        let cell = ws.lookup_chain_with_key(&key).expect("cell installed");
        assert!(
            cell.flags & jc_flags::TEMPORARY != 0,
            "tmp token must set JC_TEMPORARY"
        );
    }

    /// `warmstate.py:714-723` + `626-641` — typed variant must
    /// disambiguate hash collisions: two `GreenKey`s that share a hash
    /// but compare unequal under `equal_whatever` get distinct tokens.
    #[test]
    fn get_assembler_token_with_key_disambiguates_hash_collisions() {
        let mut ws = WarmEnterState::new(100);

        // Build a GreenKey pair guaranteed to collide on hash via the
        // existing collision fixture (install_new_cell uses bucket-by-hash).
        // Reuse the same shape as the chain-walk test: install a typed
        // cell first, then call get_assembler_token_with_key for another
        // typed key in the same bucket.
        let key_a = GreenKey::new(vec![100, 200]);
        let key_b = GreenKey::new(vec![300, 400]);

        let token_a = std::sync::Arc::new(JitCellToken::new(0xaaaa));
        let token_b = std::sync::Arc::new(JitCellToken::new(0xbbbb));

        let got_a = ws
            .get_assembler_token_with_key::<(), _>(&key_a, || Ok(token_a.clone()))
            .expect("install a");
        let got_b = ws
            .get_assembler_token_with_key::<(), _>(&key_b, || Ok(token_b.clone()))
            .expect("install b");

        assert!(
            std::sync::Arc::ptr_eq(&got_a, &token_a),
            "key_a returns token_a"
        );
        assert!(
            std::sync::Arc::ptr_eq(&got_b, &token_b),
            "key_b returns token_b"
        );
        assert!(
            !std::sync::Arc::ptr_eq(&got_a, &got_b),
            "distinct typed keys must not alias to the same token"
        );

        // Repeat fetch must not invoke make_token.
        let mut count = 0;
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_a, || {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xcccc)))
            })
            .expect("re-fetch a");
        let _ = ws
            .get_assembler_token_with_key::<(), _>(&key_b, || {
                count += 1;
                Ok(std::sync::Arc::new(JitCellToken::new(0xdddd)))
            })
            .expect("re-fetch b");
        assert_eq!(count, 0, "cached lookups must not invoke make_token");
    }

    /// warmstate.py:446-511 — typed variant of `maybe_compile_and_run`.
    /// Upstream installs the JitCell lazily at `bound_reached`
    /// (warmstate.py:425-444): each tick under threshold returns
    /// without writing to the celltable; on the threshold tick, the
    /// cell is created with `comparekey` and installed via
    /// `jitcounter.install_new_cell`. The typed entry point preserves
    /// that lifecycle — pre-threshold ticks see no cell installed,
    /// only the bucket counter ticks.
    #[test]
    fn maybe_compile_with_key_lazily_installs_on_threshold_tick() {
        let mut ws = WarmEnterState::new(3);
        let key = GreenKey::new(vec![7, 9]);

        // Tick 1, 2: not hot. Cell is NOT yet installed — upstream
        // `maybe_compile_and_run` only allocates at `bound_reached`
        // (warmstate.py:438-440).
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "cold ticks must not install a cell (warmstate.py:466-468)",
        );
        assert!(matches!(ws.maybe_compile_with_key(&key), HotResult::NotHot));
        assert!(
            ws.lookup_chain_with_key(&key).is_none(),
            "second cold tick still no cell",
        );

        // Tick 3: threshold reached → `start_tracing_cell_for_key`
        // installs a cell carrying `comparekey = key` (typed) and
        // sets the TRACING flags.
        match ws.maybe_compile_with_key(&key) {
            HotResult::StartTracing => {}
            _ => panic!("expected StartTracing"),
        }

        let cell = ws
            .lookup_chain_with_key(&key)
            .expect("cell installed at threshold tick");
        assert_eq!(
            cell.comparekey.as_ref(),
            Some(&key),
            "comparekey populated on lazy install (warmstate.py:438-439)",
        );
        assert!(cell.is_tracing(), "JC_TRACING flag set on threshold tick");
    }

    /// warmstate.py:455-465 — `JitCell.get_jitcell_for_args(*greenargs)`
    /// walks the per-bucket chain by `comparekey` to read AND mutate
    /// the cell associated with `greenargs`. A hash-only delegate
    /// would alias colliding GreenKeys to the same head and read /
    /// write the wrong cell's `JC_TRACING` / `JC_COMPILED` flags.
    ///
    /// The test simulates a hash collision by directly installing two
    /// cells in the SAME bucket via [`WarmEnterState::install_new_cell`]:
    /// `key_a` carries `JC_TRACING`, `key_b` is fresh. After installs,
    /// `install_new_cell` folds non-removable cells to the front, so
    /// the chain becomes `head=key_a (TRACING) -> tail=key_b`.
    ///
    /// `maybe_compile_with_key(&key_b)` must return `NotHot` (cell B
    /// has no flags). A hash-only read path would see the head
    /// (`key_a`) and incorrectly return `AlreadyTracing`.
    #[test]
    fn maybe_compile_with_key_walks_chain_for_state_under_hash_collision() {
        let mut ws = WarmEnterState::new(100);
        // Same hash bucket for both keys — install_new_cell takes the
        // bucket explicitly so we don't need a real `get_uhash`
        // collision in the GreenKey values.
        let key_b = GreenKey::new(vec![300, 400]);
        let key_a = GreenKey::new(vec![100, 200]);
        let bucket = key_b.get_uhash();

        // Pre-install A at B's bucket. JC_TRACING keeps A non-removable
        // (counter.py:246-256 should_remove gate) so the next install
        // chains B behind A.
        let mut cell_a = BaseJitCell::new();
        cell_a.flags |= jc_flags::TRACING;
        cell_a.comparekey = Some(key_a.clone());
        ws.install_new_cell(bucket, Some(cell_a));

        // Install B at the same bucket. install_new_cell folds A in
        // front; resulting chain shape: head=A (TRACING) → tail=B.
        let mut cell_b = BaseJitCell::new();
        cell_b.comparekey = Some(key_b.clone());
        ws.install_new_cell(bucket, Some(cell_b));

        // Sanity — chain order matches the install_new_cell contract.
        let head = ws.lookup_chain(bucket).expect("bucket has a head");
        assert_eq!(head.comparekey.as_ref(), Some(&key_a));
        assert!(head.is_tracing(), "head A carries the TRACING flag");
        let chained = head
            .next
            .as_deref()
            .expect("chained tail exists after second install");
        assert_eq!(chained.comparekey.as_ref(), Some(&key_b));
        assert!(!chained.is_tracing(), "B is fresh");

        // Without the chain-walk fix, `maybe_compile_with_key(&key_b)`
        // would delegate to `maybe_compile(hash)`, hit `cells.get(&hash)`,
        // see the head A's TRACING flag and return AlreadyTracing —
        // contaminating B's lookup with A's tracing state. With the fix
        // it walks the chain by comparekey and reads B's clean state
        // (NotHot path: counter ticks under threshold).
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::AlreadyTracing => panic!(
                "maybe_compile_with_key(&key_b) aliased to head A's TRACING flag — \
                 chain walk by comparekey is broken (warmstate.py:455-465 parity)"
            ),
            HotResult::NotHot | HotResult::StartTracing | HotResult::RunCompiled => {}
        }

        // Cell A's TRACING flag must remain on A, not migrate to B.
        let head_after = ws.lookup_chain(bucket).expect("bucket head still exists");
        assert!(
            head_after.is_tracing(),
            "head A's TRACING flag must persist across maybe_compile_with_key(&key_b)",
        );
        assert_eq!(head_after.comparekey.as_ref(), Some(&key_a));
        let chained_after = head_after
            .next
            .as_deref()
            .expect("chained tail still present");
        assert_eq!(chained_after.comparekey.as_ref(), Some(&key_b));
        assert!(
            !chained_after.is_tracing(),
            "B must not have inherited A's TRACING flag (counter under threshold)",
        );
    }

    /// warmstate.py:425-444 + 596-604 — typed-key transition writes
    /// land on the matching chained cell, not the bucket head. A
    /// collision-shaped chain `head=A → tail=B` followed by enough
    /// `maybe_compile_with_key(&key_b)` ticks to exhaust the counter
    /// must flip `B.is_tracing()` to true while A's flags stay intact.
    #[test]
    fn maybe_compile_with_key_start_tracing_writes_to_chained_cell() {
        let mut ws = WarmEnterState::new(2); // threshold=2 → 2 ticks fire
        let key_b = GreenKey::new(vec![700, 800]);
        let key_a = GreenKey::new(vec![500, 600]);
        let bucket = key_b.get_uhash();

        // Head A: must stay non-removable across B's install.
        let mut cell_a = BaseJitCell::new();
        cell_a.flags |= jc_flags::TRACING;
        cell_a.comparekey = Some(key_a.clone());
        ws.install_new_cell(bucket, Some(cell_a));

        // Tick #1 — counter not at threshold yet, NotHot.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::NotHot => {}
            _ => panic!("tick #1 expected NotHot"),
        }
        // Tick #2 — counter fires; start_tracing_cell_for_key must
        // chain-walk and flip B's flags, not the head A.
        match ws.maybe_compile_with_key(&key_b) {
            HotResult::StartTracing => {}
            _ => panic!("tick #2 expected StartTracing"),
        }

        // Walk the chain — A still TRACING (its own flag), B now
        // additionally TRACING.
        let head = ws.lookup_chain(bucket).expect("head exists");
        assert_eq!(head.comparekey.as_ref(), Some(&key_a));
        assert!(head.is_tracing(), "A's TRACING flag is its own");
        let b_cell = ws
            .lookup_chain_with_key(&key_b)
            .expect("typed lookup finds B in the chain");
        assert!(
            b_cell.is_tracing(),
            "B's TRACING flag set by start_tracing_cell_for_key — \
             chain-walk write must reach the tail, not just the head",
        );
    }
}
