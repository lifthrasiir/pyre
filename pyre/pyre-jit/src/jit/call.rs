//! RPython: `rpython/jit/codewriter/call.py` `CallControl`.
//!
//! pyre-specific analog of RPython's `CallControl`. RPython's CallControl
//! owns the JIT's per-warmspot state: the `jitcodes` dict mapping graphs
//! to compiled JitCodes and the `unfinished_graphs` queue that
//! `CodeWriter.make_jitcodes()` drains at warmspot init (codewriter.py:74-89).
//!
//! pyre's "graph" is a `CodeObject` pointer; otherwise the field-by-field
//! port matches upstream:
//!
//! | RPython `call.py`           | pyre `call.rs`              |
//! |-----------------------------|-----------------------------|
//! | `self.virtualref_info`      | `self.virtualref_info`      |
//! | `self.has_libffi_call`      | `self.has_libffi_call`      |
//! | `self.cpu`                  | `self.cpu`                  |
//! | `self.jitdrivers_sd`        | `self.jitdrivers_sd`        |
//! | `self.jitcodes`             | `self.jitcodes`             |
//! | `self.unfinished_graphs`    | `self.unfinished_graphs`    |
//! | `self.callinfocollection`   | `self.callinfocollection`   |
//! | `__init__(cpu, jitdrivers_sd)` | `new(cpu, jitdrivers_sd)`|
//! | `get_jitcode()`             | `get_jitcode()`             |
//! | `grab_initial_jitcodes()`   | `grab_initial_jitcodes()`   |
//! | `enum_pending_graphs()`     | `enum_pending_graphs()`     |
//! | (n/a — pure-lookup helper)  | `find_jitcode()`            |
//!
//! PRE-EXISTING-ADAPTATION: RPython's CallControl is owned by a single
//! `CodeWriter` instance (`warmspot.py:245`) for the lifetime of the JIT.
//! pyre mirrors that with a per-thread singleton accessed via
//! `CodeWriter::instance()`; interior mutability of the queue/cache is
//! expressed with `UnsafeCell` on the owning `CodeWriter`.

use std::collections::HashMap;

use majit_translate::jitcode::BhCallDescr;
use pyre_interpreter::CodeObject;
use pyre_jit_trace::PyJitCode;

use super::cpu::Cpu;

/// RPython: `rpython/jit/codewriter/effectinfo.py` `class CallInfoCollection`.
///
/// Owns the `_callinfo_for_oopspec` table that maps `oopspec_name → (calldescr,
/// func_addr)` so `assembler.finished()` can register the raw helper addresses
/// for symbolic debugging.
///
/// PRE-EXISTING-ADAPTATION: pyre has no `oopspec` system — every blackhole
/// helper goes through `bh_*` C ABI thunks declared on `CodeWriter` directly,
/// not through an indirected `_callinfo_for_oopspec` lookup. The struct is an
/// empty shell so `CallControl` and `Assembler::finished` keep their RPython
/// signatures intact; populating it would require an oopspec layer that
/// pyre's Python-bytecode-direct lowering does not need.
#[derive(Debug, Default)]
pub struct CallInfoCollection;

impl CallInfoCollection {
    pub fn new() -> Self {
        Self
    }
}

/// RPython: `rpython/jit/metainterp/warmspot.py` `JitDriverStaticData` —
/// the per-portal record `CallControl.jitdrivers_sd` holds and
/// `grab_initial_jitcodes` (call.py:145-148) iterates.  RPython's
/// `JitDriverStaticData` carries far more (`name`, `index`,
/// `mainjitcode`, `_green_args_spec`, …); pyre adds only the fields
/// the lazy portal-discovery path actually consumes today.
///
/// PRE-EXISTING-ADAPTATION: pyre carries `w_code` and `merge_point_pc`
/// alongside `portal_graph` because pyre's "graph" is a Python
/// CodeObject with no flow-graph identity. `w_code` is the wrapping
/// `PyObjectRef`, threaded through so the runtime side keys its
/// per-CodeObject lookups by the same identity; `merge_point_pc` is
/// the Python PC of the `MERGE_POINT` opcode the first trace through
/// the portal reveals (RPython has no analog because portal PCs are
/// statically known at flow-graph time).
///
/// `Copy` was previously derived because every field was `Copy` /
/// `Option<Copy>`; restoring `mainjitcode` (`Option<Arc<PyJitCode>>`
/// per `call.py:147`) takes that away. Only `Clone` remains —
/// per-field reads inside `grab_initial_jitcodes` snapshot the
/// `Copy` fields explicitly to keep the iteration cheap.
#[derive(Clone)]
pub struct JitDriverStaticData {
    /// RPython: `JitDriverStaticData.portal_graph`. The CodeObject
    /// the portal_runner enters when `get_jitcode(jd.portal_graph)`
    /// fires inside `grab_initial_jitcodes`.
    pub portal_graph: *const CodeObject,
    /// pyre-only: the `PyObjectRef` wrapper around `portal_graph`,
    /// kept on the jd so the trace-side jitcode table indexes by the
    /// same identity the runtime uses.
    ///
    /// **Deletion criterion (S3.2)**: removable once a canonical
    /// `CodeObject ↔ PyObjectRef` registry exists, OR every callback
    /// that consumes `w_code` (green-key build, get_jitcode, the
    /// portal-discovery hook in `eval.rs::jit_merge_point_hook`) has
    /// migrated to keying by the raw `*const CodeObject` stored in
    /// `portal_graph`. Until then this slot is the single source of
    /// truth that bridges the two identities — see the wiggly-barto
    /// epic plan, "Risk Assessment" row 4.
    pub w_code: *const (),
    /// pyre-only refinement hint forwarded to `get_jitcode`.
    /// `Some(pc)` rebuilds the cached `PyJitCode` if the recorded
    /// merge-point PC changes from a previous registration. RPython
    /// has no analog because its portal PCs are statically known.
    ///
    /// **Deletion criterion (S3.2)**: removable once the per-Python-bytecode
    /// `orgpc` is carried explicitly by `MIFrame`'s
    /// `BC_JIT_MERGE_POINT` payload (mirroring upstream `pyjitpl.py:
    /// 1536-1538` `orgpc` operand) and the JitCode PC ↔ Python PC map
    /// is the single source of truth. Until then this slot keeps the
    /// existing trace-side `make_green_key(code, pc)` lookup keyed on
    /// the same merge-point PC the runtime registered — see plan
    /// "Risk Assessment" row 3.
    pub merge_point_pc: Option<usize>,
    /// RPython: `JitDriverStaticData.mainjitcode`
    /// (`call.py:147` `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`
    /// left-hand side, plus `call.py:148`
    /// `jd.mainjitcode.jitdriver_sd = jd`).  Populated by
    /// [`CallControl::grab_initial_jitcodes`] with the same Arc that
    /// `CallControl.jitcodes[graph]` holds, mirroring RPython's
    /// "same `JitCode` instance is shared between the cc.jitcodes dict
    /// and jd.mainjitcode" identity.  Stays `None` until
    /// `grab_initial_jitcodes` fires, matching RPython's
    /// `jd.mainjitcode = None` before call.py:147.
    ///
    /// PRE-EXISTING-ADAPTATION (type-only): RPython types this as
    /// `JitCode`; pyre stores `Arc<PyJitCode>`, where `PyJitCode` is a
    /// thin wrapper around `Arc<RuntimeJitCode>` plus pyre-only
    /// metadata (`PyJitCodeMetadata`, `code_ptr`, `w_code`, `has_abort`,
    /// `merge_point_pc`).  The wrapper exists because pyre's codewriter
    /// publication path needs interior mutability to fill an empty
    /// skeleton in place (so that the same Arc identity sits in both
    /// `cc.jitcodes[graph]` and `jd.mainjitcode` while the assembler
    /// drains pending graphs into populated payloads).  The runtime
    /// `JitCode` (`RuntimeJitCode`) is reachable via
    /// `mainjitcode.jitcode` and is the object `jitdriver_sd` is
    /// stamped onto, matching upstream's
    /// `jd.mainjitcode.jitdriver_sd = jd` line-for-line.  Production
    /// access drills explicitly through `.jitcode` rather than relying
    /// on `Deref`, so the wrapper boundary stays visible at every site.
    ///
    /// **Deletion criterion**: removable once the codewriter publication
    /// path is reworked so the runtime `JitCode` itself can be
    /// late-stamped without an outer skeleton (the per-jitcode pyre-only
    /// metadata moves to a side table keyed by `RuntimeJitCode` Arc
    /// identity).  See `pyjitcode_wrapper_dissolution_slice_a_audit_*`
    /// memos for the multi-session plan.
    pub mainjitcode: Option<std::sync::Arc<PyJitCode>>,
}

impl JitDriverStaticData {
    /// Normalize `portal_graph` to the runtime raw-code identity when a
    /// wrapper object is available. This keeps the stored field closer
    /// to RPython's `jd.portal_graph is graph` contract: after
    /// registration, every portal lookup/grab path sees the same raw
    /// `CodeObject*` that execution paths derive from `w_code`.
    pub(crate) fn canonicalized(mut self) -> Self {
        if !self.w_code.is_null() {
            self.portal_graph = unsafe {
                pyre_interpreter::w_code_get_ptr(self.w_code as pyre_object::PyObjectRef)
                    as *const CodeObject
            };
        }
        self
    }
}

/// RPython: `rpython/jit/codewriter/call.py:21` `class CallControl(object)`.
pub struct CallControl {
    /// call.py:22 `virtualref_info = None` — class-level default,
    /// populated by `CodeWriter.setup_vrefinfo` (codewriter.py:91-94).
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `virtualref` machinery yet
    /// (no `@jit.virtual_ref` annotations, no `vref_info` lookup); the
    /// slot is `None` and the setter is a no-op shell.
    pub virtualref_info: Option<()>,

    /// call.py:23 `has_libffi_call = False` — class-level default.
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `_call_aroundstate_target_`
    /// rewriting in `getcalldescr`, so this stays `false`.
    pub has_libffi_call: bool,

    /// call.py:27 `self.cpu = cpu`.
    ///
    /// pyre's `Cpu` (see [`super::cpu::Cpu`]) bundles the blackhole
    /// helper function pointers used by `transform_graph_to_jitcode`
    /// to populate the `JitCode` fn-ptr table. The same `cpu` is also
    /// reachable via `CodeWriter::cpu(&self)` (codewriter.py:21
    /// `self.cpu = cpu`).
    pub cpu: Cpu,

    /// call.py:28 `self.jitdrivers_sd = jitdrivers_sd`.
    ///
    /// `grab_initial_jitcodes()` (call.py:145-148) iterates this list
    /// and calls `get_jitcode(jd.portal_graph)` on each entry. pyre
    /// populates the list lazily via `CodeWriter::setup_jitdriver`
    /// (codewriter.py:96-99), once per unique portal CodeObject the
    /// runtime decides to JIT — see [`JitDriverStaticData`] for the
    /// pyre-only adapter fields.
    pub jitdrivers_sd: Vec<JitDriverStaticData>,

    /// call.py:29 `self.jitcodes = {}` — map `{graph: jitcode}`.
    ///
    /// pyre keys on the canonical raw `CodeObject*` graph identity.
    /// Callers normalize wrapper-backed runtime inputs before they
    /// reach `CallControl`, so the cache itself can stay a plain
    /// `{graph: jitcode}` map like RPython's.
    ///
    /// Values are `Arc<PyJitCode>` so the same allocation can sit in
    /// `MetaInterpStaticData.jitcodes` too. RPython's two stores hold
    /// references to identical Python `JitCode` objects through
    /// refcount semantics; pyre mirrors that by sharing the `Arc` —
    /// the `find_jitcode_arc` helper hands the trace-side
    /// `state::jitcode_for` callback the same `Arc` it just installed
    /// here. `get_jitcode` returns a `&PyJitCode` elaborated into
    /// `'static` to match RPython's Python semantics (see the SAFETY
    /// comment there); the `Arc` keeps the address stable across
    /// `HashMap` rehashes.
    pub jitcodes: HashMap<usize, std::sync::Arc<PyJitCode>>,

    /// call.py:30 `self.unfinished_graphs = []` — LIFO queue of graphs
    /// pending compilation. Populated by `get_jitcode()` when it sees a
    /// new graph and drained by `enum_pending_graphs()` inside
    /// `make_jitcodes()` (codewriter.py:79).
    ///
    /// pyre now mirrors RPython's bare-graph queue directly. The
    /// pyre-only `w_code` / `merge_point_pc` adapter state lives on the
    /// cached `PyJitCode` shell instead of in the queue tuple so
    /// `enum_pending_graphs()` can match `call.py:150-153` again.
    pub unfinished_graphs: Vec<*const CodeObject>,

    /// call.py:31 `self.callinfocollection = CallInfoCollection()`.
    ///
    /// Fed to `Assembler::finished()` at the end of
    /// `CodeWriter::make_jitcodes` (codewriter.py:85). See
    /// [`CallInfoCollection`] for the pyre-side shell rationale.
    pub callinfocollection: CallInfoCollection,
}

impl CallControl {
    /// RPython: `CallControl.__init__(cpu=None, jitdrivers_sd=[])`
    /// (call.py:25-47).
    pub fn new(cpu: Cpu, jitdrivers_sd: Vec<JitDriverStaticData>) -> Self {
        // call.py:26 `assert isinstance(jitdrivers_sd, list)`.
        // Rust's type system enforces this at compile time.
        Self {
            virtualref_info: None,
            has_libffi_call: false,
            cpu,
            jitdrivers_sd,
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
            callinfocollection: CallInfoCollection::new(),
        }
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// for jd in self.jitdrivers_sd:
    ///     jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///     jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// PARITY: `JitDriverStaticData.mainjitcode` (call.py:147 left-hand
    /// side) is assigned immediately from `get_jitcode`. The
    /// back-reference at call.py:148 (`jd.mainjitcode.jitdriver_sd = jd`)
    /// is stamped onto the populated runtime `JitCode` in
    /// `CodeWriter::finalize_jitcode`, where the precise jdindex is known
    /// from `CallControl.jitdrivers_sd`.
    pub fn grab_initial_jitcodes(&mut self) {
        // Index loop because get_jitcode borrows `self.jitcodes`
        // mutably, which would conflict with an immutable borrow over
        // `self.jitdrivers_sd`. The fields snapshotted below
        // (`portal_graph`, `w_code`, `merge_point_pc`) are all `Copy`,
        // so the per-iteration field reads stay cheap.
        for i in 0..self.jitdrivers_sd.len() {
            let portal_graph = self.jitdrivers_sd[i].portal_graph;
            let w_code = self.jitdrivers_sd[i].w_code;
            let merge_point_pc = self.jitdrivers_sd[i].merge_point_pc;
            let code = unsafe { &*portal_graph };
            // call.py:147 `jd.mainjitcode = self.get_jitcode(jd.portal_graph)`.
            // Inserts an empty PyJitCode skeleton into `jitcodes`, pushes
            // the graph onto `unfinished_graphs`. Drop the returned
            // clone immediately so the cached slot is uniquely owned for
            // the call.py:148 stamp below.
            drop(self.get_jitcode(code, w_code, merge_point_pc));
            // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — stamp the
            // skeleton's `jitdriver_sd` while the outer `Arc<PyJitCode>`
            // and inner `Arc<JitCode>` still have refcount 1 (only the
            // jitcodes slot holds them). Then bind `jd.mainjitcode` to
            // that same Arc so consumers reading `cc.jitcodes[key]` and
            // `jd.mainjitcode` see byte-identical Arc identity per the
            // upstream invariant.
            let key = Self::jitcode_key(portal_graph);
            if let Some(slot) = self.jitcodes.get_mut(&key) {
                // RPython call.py:148 — `jd.mainjitcode.jitdriver_sd =
                // jd`.  Canonical's `jitdriver_sd` is an OnceLock with
                // interior mutability, so the write succeeds through
                // `&Arc<JitCode>` even when the surrounding Arc is
                // shared (e.g. `cc.jitcodes` and `jd.mainjitcode`).
                let pyjitcode = std::sync::Arc::clone(slot);
                if pyjitcode.jitcode.jitdriver_sd().is_none() {
                    pyjitcode.jitcode.set_jitdriver_sd(i);
                }
                self.jitdrivers_sd[i].mainjitcode = Some(std::sync::Arc::clone(slot));
            }
        }
    }

    /// Pure lookup of an already-compiled JitCode. Matches RPython's
    /// runtime pattern where callers read `self.jitcodes[graph]`
    /// directly after `make_jitcodes()` has populated the dict; no
    /// compilation side effect. Returns `None` if `code` has not been
    /// compiled yet.
    pub fn find_jitcode(&self, code: *const CodeObject) -> Option<&PyJitCode> {
        self.jitcodes.get(&(code as usize)).map(|arc| arc.as_ref())
    }

    /// `Arc`-cloning variant used by the trace-side `state::jitcode_for`
    /// callback: returns the same `Arc<PyJitCode>` the SD will store,
    /// so both stores reference one allocation. RPython's
    /// `MetaInterpStaticData.jitcodes`, `CallControl.jitcodes`, and
    /// `JitDriverStaticData.mainjitcode` hold the same Python `JitCode`
    /// objects via refcount semantics; this helper is the Rust analog.
    pub fn find_jitcode_arc(&self, code: *const CodeObject) -> Option<std::sync::Arc<PyJitCode>> {
        self.jitcodes
            .get(&(code as usize))
            .map(std::sync::Arc::clone)
    }

    /// Compiled-entry variant used by setup-time publishers and tests:
    /// return the shared `Arc<PyJitCode>` only when the entry is fully
    /// populated by the `make_jitcodes()` drain (`PyJitCode::is_populated`
    /// — `pc_map` non-empty). RPython never exposes skeleton entries to
    /// runtime readers; a `None` here means the caller is still in setup
    /// and must finish draining before publishing to trace-side SD.
    pub fn find_compiled_jitcode_arc(
        &self,
        code: *const CodeObject,
    ) -> Option<std::sync::Arc<PyJitCode>> {
        let arc = self.jitcodes.get(&(code as usize))?;
        if !arc.is_populated() {
            return None;
        }
        Some(std::sync::Arc::clone(arc))
    }

    /// Recover the pyre-only inputs the drain needs for a queued graph.
    /// The queue itself keeps RPython's bare `graph` shape; the extra
    /// runtime identity/refinement data lives on the cached skeleton.
    pub fn queued_graph_inputs(
        &self,
        code: *const CodeObject,
    ) -> Option<(*const (), Option<usize>)> {
        let arc = self.jitcodes.get(&(code as usize))?;
        Some((arc.w_code, arc.merge_point_pc))
    }

    /// Reverse lookup: find the `PyJitCode` whose inner `JitCode` matches
    /// the given raw pointer. Pyre-only adaptation: RPython's
    /// `JitCode` has its portal-register assignments embedded in the
    /// regular inputargs, so the blackhole interpreter can fill them
    /// via `_prepare_next_section` with no side channel. Pyre carries
    /// `portal_frame_reg`/`portal_ec_reg` on `PyJitCodeMetadata`
    /// instead, so the blackhole chain needs this reverse map to
    /// populate them after `blackhole_from_resumedata` builds the chain.
    pub fn find_pyjitcode_by_jitcode_ptr(
        &self,
        jitcode_ptr: *const majit_metainterp::jitcode::JitCode,
    ) -> Option<&PyJitCode> {
        self.jitcodes
            .values()
            .find(|pyjit| std::sync::Arc::as_ptr(&pyjit.jitcode) == jitcode_ptr)
            .map(|arc| arc.as_ref())
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// while self.unfinished_graphs:
    ///     graph = self.unfinished_graphs.pop()
    ///     yield graph, self.jitcodes[graph]
    /// ```
    ///
    pub fn enum_pending_graphs(&mut self) -> Option<*const CodeObject> {
        self.unfinished_graphs.pop()
    }

    /// RPython: `CallControl.jitdriver_sd_from_portal_graph(graph)` —
    /// returns the `JitDriverStaticData` whose `portal_graph` matches
    /// `code`, or `None` if `code` is not a registered portal. Used by
    /// `CodeWriter::transform_graph_to_jitcode` (codewriter.py:37) to
    /// learn whether a graph is a portal **before** running jtransform,
    /// matching the RPython `portal_jd = self.callcontrol.…` lookup.
    ///
    /// pyre returns the jdindex (`usize`) instead of a borrowed `&jd`
    /// so callers can keep mutating `CallControl` afterwards without
    /// fighting the borrow checker; the index is stable across the
    /// lifetime of `jitdrivers_sd` because `setup_jitdriver` only
    /// appends and dedups.
    pub fn jitdriver_sd_from_portal_graph(&self, code: *const CodeObject) -> Option<usize> {
        self.jitdrivers_sd
            .iter()
            .position(|jd| jd.portal_graph == code)
    }

    /// RPython: `CallControl.get_jitcode_calldescr(graph)` (call.py:174-187).
    ///
    /// ```python
    /// def get_jitcode_calldescr(self, graph):
    ///     fnptr = getfunctionptr(graph)
    ///     FUNC = lltype.typeOf(fnptr).TO
    ///     fnaddr = llmemory.cast_ptr_to_adr(fnptr)
    ///     NON_VOID_ARGS = [ARG for ARG in FUNC.ARGS if ARG is not lltype.Void]
    ///     calldescr = self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS),
    ///                                      FUNC.RESULT, EffectInfo.MOST_GENERAL)
    ///     return (fnaddr, calldescr)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre's blackhole calls every Python
    /// function through one `bh_portal_runner(frame: ref) -> ref`
    /// stub — the C-ABI is identical for every CodeObject because the
    /// portal runner unwraps the frame and dispatches dynamically. So
    /// `(fnaddr, calldescr)` is constant across all graphs in pyre,
    /// while RPython has one pair per `FUNC` type because lltype-typed
    /// `direct_call` ops can carry varying signatures. Keeping the
    /// method shape preserves the call.py:167
    /// `(fnaddr, calldescr) = self.get_jitcode_calldescr(graph)` flow
    /// even though both values are constants here.
    pub fn get_jitcode_calldescr(&self, _graph: *const CodeObject) -> (i64, BhCallDescr) {
        let fnaddr = crate::call_jit::bh_portal_runner as *const () as usize as i64;
        let calldescr = BhCallDescr::from_arg_classes(
            "r".to_string(),
            'r',
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        (fnaddr, calldescr)
    }

    /// RPython: `CallControl.get_jitcode(self, graph, called_from=None)`
    /// (call.py:155-172).
    ///
    /// ```python
    /// def get_jitcode(self, graph, called_from=None):
    ///     try:
    ///         return self.jitcodes[graph]
    ///     except KeyError:
    ///         ...
    ///         fnaddr, calldescr = self.get_jitcode_calldescr(graph)
    ///         jitcode = JitCode(graph.name, fnaddr, calldescr,
    ///                           called_from=called_from)
    ///         self.jitcodes[graph] = jitcode
    ///         self.unfinished_graphs.append(graph)
    ///         return jitcode
    /// ```
    ///
    /// pyre creates an empty `PyJitCode` skeleton (call.py:168 — RPython
    /// `JitCode(name, fnaddr, calldescr)` without bytecode) and queues
    /// the graph for the drain in `CodeWriter::make_jitcodes`, which
    /// re-runs `transform_graph_to_jitcode` and replaces the slot with
    /// the populated entry (codewriter.py:80
    /// `transform_graph_to_jitcode(graph, jitcode, ...)`).
    ///
    /// PRE-EXISTING-ADAPTATION: `merge_point_pc` is a pyre-only refinement
    /// — the first trace reveals the `MERGE_POINT` opcode's PC, which
    /// must be re-recorded on the cache entry. When it changes the
    /// entry is reset to a new skeleton and re-queued so the drain
    /// recompiles with the refined hint. RPython has no analog because
    /// portal PCs are statically known.
    pub fn get_jitcode(
        &mut self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> std::sync::Arc<PyJitCode> {
        // RPython's `get_jitcode(graph)` receives the exact graph object
        // the dict is keyed by. Match that by requiring callers to pass
        // the canonical raw graph here; portal setup canonicalizes
        // `jd.portal_graph` once, and the lazy trace-side callback
        // unwraps `w_code` before calling into `CallControl`.
        let code_ptr = code as *const CodeObject;
        let key = code_ptr as usize;
        let needs_rebuild = if let Some(existing) = self.jitcodes.get(&key) {
            merge_point_pc.is_some() && existing.merge_point_pc != merge_point_pc
        } else {
            true
        };
        if needs_rebuild {
            // call.py:168-171 — create JitCode skeleton, insert, append
            // to unfinished_graphs. The body fill (jtransform / regalloc
            // / flatten / assemble) is deferred to
            // `CodeWriter::make_jitcodes`'s drain loop at
            // codewriter.py:80 `transform_graph_to_jitcode(graph,
            // jitcode, verbose, len(all_jitcodes))`.
            self.reset_jitcode_skeleton(key, code_ptr, w_code, merge_point_pc);
            // call.py:171 `self.unfinished_graphs.append(graph)`. pyre
            // also re-pushes on a merge_point_pc refinement so the
            // drain picks the refined entry up again for the recompile
            // — see `make_jitcodes` at codewriter.rs.
            self.unfinished_graphs.push(code_ptr);
        }
        std::sync::Arc::clone(self.jitcodes.get(&key).unwrap())
    }

    /// Reset the cached slot back to an empty skeleton — the state that
    /// follows `JitCode(graph.name, fnaddr, calldescr, ...)` at
    /// `call.py:168` before the drain re-runs `assembler.assemble`
    /// (codewriter.py:67) on it.
    ///
    /// PRE-EXISTING-ADAPTATION: this entry point exists only because
    /// `get_jitcode` re-runs the drain when `merge_point_pc` is refined
    /// — RPython has no analog because portal PCs are statically known
    /// (call.py:155 `if graph in self.jitcodes: return self.jitcodes[graph]`
    /// has no reset branch).
    ///
    /// This method must NOT use the in-place payload mutation that
    /// `publish_jitcode` relies on: after the first drain, `jd.mainjitcode`
    /// and any trace-side `MetaInterpStaticData.jitcodes` clone are
    /// already pointing at the populated `Arc<PyJitCode>`. Replacing the
    /// payload in place here would clobber those holders back to a
    /// skeleton state, which RPython's "runtime reader never observes a
    /// reset shell" invariant rules out. Instead we always insert a new
    /// `Arc<PyJitCode>`; the previous Arc stays populated for any holder
    /// that already cloned it, matching the pre-Slice-2 behavior that
    /// fell back to `Arc::new(...)` on `Arc::get_mut` failure.
    pub fn reset_jitcode_skeleton(
        &mut self,
        key: usize,
        code_ptr: *const CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) {
        self.jitcodes.insert(
            key,
            std::sync::Arc::new(PyJitCode::skeleton(code_ptr, w_code, merge_point_pc)),
        );
    }

    /// Publish the populated jitcode into `self.jitcodes[graph]`.
    ///
    /// RPython mutates the same `JitCode` object in place — the drain
    /// at `codewriter.py:80` calls `transform_graph_to_jitcode(graph,
    /// jitcode, ...)` where `jitcode` is the skeleton that was already
    /// stored at `call.py:170` (`self.jitcodes[graph] = jitcode`), and
    /// inside that call `self.assembler.assemble(ssarepr, jitcode,
    /// num_regs)` (codewriter.py:67) fills the skeleton's fields.
    ///
    /// Pyre's `transform_graph_to_jitcode` returns a fresh `PyJitCode`
    /// instead of mutating the skeleton, so this helper bridges the
    /// split by filling the cached object in place. Both the outer
    /// `Arc<PyJitCode>` and the inner runtime `Arc<JitCode>` allocation
    /// stay stable; the latter is required before the orthodox
    /// `jtransform.handle_regular_call -> inline_call_*` port can store
    /// callee JitCode descriptors in callers.
    pub fn publish_jitcode(
        &mut self,
        key: usize,
        pyjitcode: PyJitCode,
    ) -> std::sync::Arc<PyJitCode> {
        if let Some(slot) = self.jitcodes.get_mut(&key) {
            // SAFETY: `publish_jitcode` runs on the JIT setup thread
            // during the codewriter drain (codewriter.py:79-85). The
            // cached `Arc<PyJitCode>` is only shared with `jd.mainjitcode`
            // and any pre-bound `MetaInterpStaticData.jitcodes` clone,
            // which were established by the same setup thread just
            // before this call. No runtime tracing or blackhole resume
            // can be observing the slot at this point — those paths
            // only run after `make_jitcodes` returns.
            unsafe {
                slot.replace_with(pyjitcode);
            }
            return std::sync::Arc::clone(slot);
        }
        let arc = std::sync::Arc::new(pyjitcode);
        self.jitcodes.insert(key, arc);
        std::sync::Arc::clone(self.jitcodes.get(&key).unwrap())
    }

    pub(crate) fn jitcode_key(code: *const CodeObject) -> usize {
        code as usize
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new(Cpu::default(), Vec::new())
    }
}
