//! Code generation pipeline — majit's equivalent of `rpython/jit/codewriter/`.
//!
//! ```text
//! rpython/jit/codewriter/          majit-translate/src/codewriter/
//! ├── codewriter.py          →     ├── mod.rs (CodeWriter struct)
//! ├── jtransform.py          →     ├── jtransform.rs
//! ├── flatten.py + assembler.py →  ├── codegen.rs
//! └── call.py                →     └── (call.rs)
//! ```

pub mod codegen;

pub use codegen::{
    CodegenValueKind, IoShim, JitDriverConfig, VirtualizableCodegenConfig, generate_jitcode,
};

use std::collections::HashMap;
use std::sync::Arc;

use crate::assembler::Assembler;
use crate::call::CallControl;
use crate::flatten::{RegKind, flatten_with_types};
use crate::jit_codewriter::type_state::TypeResolutionState;
use crate::jitcode::JitCode;
use crate::jtransform::GraphTransformConfig;
use crate::model::{FunctionGraph, ValueId};
use crate::parse::CallPath;

/// Output of `CodeWriter::make_jitcodes` — the populated JitCode registry
/// paired with its alloc-order index view.
///
/// Parity shape: RPython `rpython/jit/codewriter/call.py:87-88` holds two
/// fields on `CallControl` itself — `self.jitcodes = {}` (graph-keyed dict)
/// and `self.all_jitcodes = []` (alloc-order list). Pyre preserves both
/// fields on `CallControl` and additionally returns them as a single value
/// from `make_jitcodes`. The wrapper is a PRE-EXISTING-ADAPTATION: upstream
/// callers (`warmspot.py:281-282`) read `self.all_jitcodes` off the
/// CallControl directly; pyre pairs the two views so that downstream
/// consumers (e.g. `crate::generated::all_jitcodes`) receive one value and
/// do not need to pass `CallControl` around to perform later lookups.
pub struct AllJitCodes {
    /// RPython `call.py:87 self.jitcodes = {}` (graph-keyed dict). Pyre's
    /// graph identity at this boundary is `CallPath`.
    pub by_path: HashMap<CallPath, Arc<JitCode>>,
    /// RPython `call.py:88 self.all_jitcodes = []` (alloc-order list). The
    /// `jitcode.index == i` invariant from `codewriter.py:80` is enforced
    /// by `CallControl::collect_jitcodes_in_alloc_order`.
    pub in_order: Vec<Arc<JitCode>>,
}

impl AllJitCodes {
    pub fn len(&self) -> usize {
        self.in_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.in_order.is_empty()
    }

    pub fn get(&self, path: &CallPath) -> Option<&Arc<JitCode>> {
        self.by_path.get(path)
    }
}

/// RPython: `codewriter.py::CodeWriter`.
///
/// Orchestrates the full JitCode generation pipeline:
///   annotate → rtype → jtransform → regalloc → flatten → liveness → assemble
///
/// RPython's CodeWriter owns both the Assembler and CallControl.
/// In majit, CallControl is passed by `&mut` reference to avoid
/// lifetime entanglement with the Transformer's borrows.
pub struct CodeWriter {
    /// RPython: `self.assembler = Assembler()` (codewriter.py:22).
    pub assembler: Assembler,
    /// RPython: `self.debug = True` (codewriter.py:18).
    pub debug: bool,
    /// Slice 6 — lazy program-wide `PyreCallRegistry` for the
    /// PYRE_RTYPER dual-gate (cf. `transform_graph_to_jitcode`).
    /// Built on first use from `callcontrol.function_graphs()` and
    /// reused across every dual-gated graph in the same CodeWriter
    /// run, mirroring upstream `Bookkeeper.descs` lifecycle (one
    /// descs map per `Translator` driving an `RPythonAnnotator`).
    real_rtyper_registry: std::cell::RefCell<
        Option<std::rc::Rc<crate::translator::rtyper::pyre_call_registry::PyreCallRegistry>>,
    >,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// `debug` mirrors the class-level default `debug = True`
    /// (`codewriter.py:18`). Upstream always produces per-jitcode
    /// diagnostic output (`log.dot()` in `print_ssa_repr`), so pyre
    /// matches by defaulting `debug: true`. Tests that want silent
    /// operation may flip the field after construction.
    pub fn new() -> Self {
        Self {
            assembler: Assembler::new(),
            debug: true,
            real_rtyper_registry: std::cell::RefCell::new(None),
        }
    }

    /// Get-or-build the program-wide `PyreCallRegistry` for the
    /// PYRE_RTYPER dual-gate.  Builds once per CodeWriter run and
    /// caches; subsequent calls reuse the same `Rc`.  The registry
    /// is populated from `callcontrol.function_graphs()` — pyre's
    /// program-wide graph map, the production analog of
    /// `SemanticProgram.functions`.
    fn dual_gate_registry(
        &self,
        callcontrol: &CallControl,
    ) -> std::rc::Rc<crate::translator::rtyper::pyre_call_registry::PyreCallRegistry> {
        if let Some(existing) = self.real_rtyper_registry.borrow().as_ref() {
            return existing.clone();
        }
        let registry = std::rc::Rc::new(
            crate::translator::rtyper::pyre_call_registry::PyreCallRegistry::new(std::rc::Rc::new(
                crate::annotator::bookkeeper::Bookkeeper::new(),
            )),
        );
        // PyPy's `Bookkeeper.compute_at_fixpoint` raises through to
        // the caller (`bookkeeper.py:108-127`); pyre's dual-gate
        // mirrors that propagation by routing the populate `TyperError`
        // through `is_known_unported`.  Known-unported categories
        // leave a partial registry (the per-graph dual-gate will
        // Skip-classify the affected callsites via the cascade
        // `"not registered in PyreCallRegistry"`).  Unknown errors
        // panic immediately so parity bugs surface here rather than
        // silently shifting downstream behind a Skip mask.
        let populate_result =
            crate::translator::rtyper::cutover::populate_call_registry_from_call_graphs(
                callcontrol.function_graphs(),
                &registry,
            );
        if let Err(err) = populate_result {
            let msg = format!("{err}");
            if !crate::translator::rtyper::cutover::is_known_unported(&msg) {
                panic!("populate_call_registry_from_call_graphs failed: {msg}");
            }
        }
        // RPython parity: `Translator.buildannotator()` /
        // `Translator.buildrtyper()` (`translator.py:69-83`) construct
        // exactly one annotator and one rtyper per Translator and assert
        // on re-entry.  Pyre mirrors that contract through
        // [`PyreCallRegistry::ensure_session`]
        // (`pyre_call_registry.rs:210-234`), which lazily builds a single
        // `(RPythonAnnotator, RPythonTyper)` pair on first use and
        // returns the cached pair on every subsequent
        // `specialize_legacy_graph_with_registry_seed` call.  The
        // registry itself is cached on the `CodeWriter`
        // (`real_rtyper_registry`), so all per-graph subjects of one
        // CodeWriter share the same annotator + rtyper, matching
        // upstream's "one annotator + one rtyper per Translator"
        // semantics.
        *self.real_rtyper_registry.borrow_mut() = Some(registry.clone());
        registry
    }

    /// RPython: `CodeWriter.transform_graph_to_jitcode()` (codewriter.py:33-72).
    ///
    /// Transforms a FunctionGraph into a JitCode through the 4-step pipeline.
    /// Upstream signature `(self, graph, jitcode, verbose, index)`. Pyre adds
    /// `path` / `callcontrol` / `config` as PRE-EXISTING-ADAPTATION:
    ///   - `path`: graph identity surrogate (upstream uses `graph` object
    ///     identity; pyre uses `CallPath`).
    ///   - `callcontrol`: upstream has `self.callcontrol`; pyre passes
    ///     `&mut` due to Rust borrow-checker constraints (the Transformer
    ///     borrows callcontrol during `transform()`).
    ///   - `config`: pyre's `GraphTransformConfig` carries options that
    ///     upstream keeps in globals / command-line flags.
    ///
    /// Steps:
    ///   0. annotate + rtype (majit-specific; RPython does this before codewriter)
    ///   1. jtransform — `transform_graph()` (codewriter.py:42)
    ///   2. regalloc — `perform_register_allocation()` per kind (codewriter.py:45-47)
    ///   3. flatten — `flatten_graph()` (codewriter.py:53)
    ///   3b. liveness — `compute_liveness()` (codewriter.py:56, called inside assemble)
    ///   4. assemble — `assembler.assemble()` (codewriter.py:67)
    ///   5. `jitcode.index = index` (codewriter.py:68)
    ///   6. `if self.debug: self.print_ssa_repr(ssarepr, portal_jd, verbose)`
    ///      (codewriter.py:71-72)
    /// Slice 12.2 / 12.4 — shared dual-gate type-resolve entry.
    ///
    /// Runs [`dual_gate_check_with_registry`] against the
    /// program-wide `PyreCallRegistry`; on Match the real path's
    /// `TypeResolutionState` is returned directly, on Skip the
    /// legacy walker (`legacy_annotator::annotate` +
    /// `legacy_resolve::resolve_types`) computes the fallback so the
    /// non-portal jitcodes that didn't pass the real path still get a
    /// sound type state.
    ///
    /// `diag_label` is appended to the optional Skip log line and the
    /// real-path panic message; production callers pass
    /// `path.canonical_key()`-style identification, the lib.rs
    /// debug-snapshot path passes `graph.name`.
    pub fn dual_gate_type_state(
        &mut self,
        graph: &FunctionGraph,
        callcontrol: &mut CallControl,
        diag_label: &str,
    ) -> TypeResolutionState {
        let dual_gate_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let registry = self.dual_gate_registry(callcontrol);
            crate::translator::rtyper::cutover::dual_gate_check_with_registry(graph, &registry)
        }));
        let outcome = match dual_gate_outcome {
            Ok(result) => result,
            Err(payload) => {
                let msg = if let Some(s) = payload.downcast_ref::<&'static str>() {
                    (*s).to_string()
                } else if let Some(s) = payload.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "<unrecognised panic payload>".to_string()
                };
                if crate::translator::rtyper::cutover::is_known_unported(&msg) {
                    Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(
                        format!("registry build panicked: {msg}"),
                    ))
                } else {
                    Err(format!("registry build panicked: {msg}"))
                }
            }
        };
        match outcome {
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Match {
                real_state,
                real_annotations: _,
            }) => real_state,
            Ok(crate::translator::rtyper::cutover::DualGateOutcome::Skip(reason)) => {
                if std::env::var_os("PYRE_RTYPER_VERBOSE").is_some_and(|v| v == "1") {
                    eprintln!(
                        "[PYRE_RTYPER skip] graph {diag_label:?} ({:?}): {reason}",
                        graph.name,
                    );
                }
                let annotations = crate::translator::rtyper::legacy_annotator::annotate(graph);
                crate::translator::rtyper::legacy_resolve::resolve_types(graph, &annotations)
            }
            Err(diff) => panic!(
                "PYRE_RTYPER real-path failure on graph {diag_label:?} ({:?}): {diff}",
                graph.name,
            ),
        }
    }

    pub fn transform_graph_to_jitcode(
        &mut self,
        graph: &FunctionGraph,
        path: &crate::parse::CallPath,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
        jitcode: &std::sync::Arc<JitCode>,
        verbose: bool,
        index: usize,
    ) {
        // RPython: graph = copygraph(graph, shallowvars=True) (codewriter.py:38)
        // In Rust, Transformer.transform() already clones the graph.

        // Step 0: annotate + rtype (majit-specific)
        // RPython: types are already on Variable.concretetype from the rtyper.
        //
        // Slice 12.2 narrowed the legacy walker to the Skip arm; Slice
        // 12.4 lifted the dual-gate logic into the shared
        // [`Self::dual_gate_type_state`] helper so the parser-level
        // debug snapshot in `build_canonical_opcode_dispatch`
        // (lib.rs:898) can route through the same path.
        let canonical_diag = path.canonical_key().to_string();
        let mut type_state = self.dual_gate_type_state(graph, callcontrol, &canonical_diag);

        // Step 0b: rtyper-equivalent indirect_call lowering
        // (`translator/rtyper/rpbc.rs::lower_indirect_calls`).
        // RPython rpbc.py:199-217 emits `indirect_call(funcptr, *args,
        // c_graphs)` during rtype; pyre runs the same pass here so
        // jtransform sees `OpKind::IndirectCall` (with funcptr already
        // a regular ValueId), never `CallTarget::Indirect`.
        let mut graph_owned = graph.clone();
        crate::translator::rtyper::rpbc::lower_indirect_calls(
            &mut graph_owned,
            &mut type_state,
            callcontrol,
        );
        #[cfg(debug_assertions)]
        crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(&graph_owned);
        let graph = &graph_owned;

        // RPython codewriter.py:37 `portal_jd =
        // self.callcontrol.jitdriver_sd_from_portal_graph(graph)` — look
        // up the JitDriverStaticData keyed by portal graph identity. In
        // pyre `CallPath` is the surrogate, so resolve via `path`. Store
        // only the index to avoid a second borrow against
        // `callcontrol` during Transformer construction; the
        // Transformer can re-dereference `jitdrivers_sd[portal_jd_index]`
        // on demand if it needs the full record.
        let portal_jd_index = callcontrol
            .jitdriver_sd_from_portal_graph(path)
            .map(|jd| jd.index);

        // Step 1: jtransform (codewriter.py:42)
        // RPython: transform_graph(graph, cpu, callcontrol, portal_jd)
        let rewritten = {
            let mut transformer = crate::jtransform::Transformer::new(config)
                .with_callcontrol(callcontrol)
                .with_portal_jd(portal_jd_index)
                .with_type_state(&type_state);
            transformer.transform(graph)
        };
        // Transformer is dropped here, releasing the &mut CallControl borrow.

        // RPython stores `.concretetype` on each Variable. Pyre merges
        // the pre-jtransform rtyper table with explicit result kinds
        // introduced by jtransform (`CallResidual.result_kind`, etc.)
        // without letting stale pre-rewrite entries override them.
        //
        // The merge precedence is `stamped > post_result > original`:
        //   - `original` = pre-jtransform `type_state` from the real
        //     RPythonTyper (already typed every pre-rewrite ValueId).
        //   - `post_result` = `authoritative_result_types(rewritten)`
        //     reads the rewritten ops' declared `result_ty`/`result_
        //     kind` fields (per-op ground-truth).
        //   - `stamped` = `synth_kinds` jtransform produced for
        //     freshly-synthesised values.
        //
        // The legacy `resolve_types(rewritten_graph, annotations)`
        // walker that previously fed `merge_synth_kinds`'s `post_
        // resolve` lane is no longer called: every post-rewrite
        // ValueId either reuses a pre-rewrite identity (covered by
        // `original`), is an op.result with declared kind (covered by
        // `post_result`), or is a synth jtransform value (covered by
        // `stamped`).  Block inputargs introduced by jtransform pick
        // up types via `original` (jtransform reuses the pre-rewrite
        // inputarg ValueIds when splicing new control flow).
        let post_result_types =
            crate::jit_codewriter::type_state::authoritative_result_types(&rewritten.graph);
        let rewritten_type_state = crate::jit_codewriter::type_state::merge_synth_kinds(
            &type_state,
            crate::jit_codewriter::type_state::TypeResolutionState::new(),
            post_result_types,
            &rewritten.synth_kinds,
        );

        // Step 2: regalloc (codewriter.py:45-47)
        // RPython: for kind in KINDS: regallocs[kind] = perform_register_allocation(graph, kind)
        let value_kinds =
            crate::jit_codewriter::type_state::build_value_kinds(&rewritten_type_state);
        let regallocs =
            crate::regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);

        // Step 3: flatten (codewriter.py:53)
        // RPython: ssarepr = flatten_graph(graph, regallocs, cpu=cpu)
        let mut ssarepr = flatten_with_types(&rewritten.graph, &rewritten_type_state, &regallocs);

        // Step 3b + 4: liveness + assemble (codewriter.py:56,67)
        // RPython: compute_liveness(ssarepr) then assembler.assemble(ssarepr, jitcode, num_regs)
        // In majit, assemble() calls compute_liveness() internally and now
        // returns the body so the codewriter can fill calldescr before
        // committing the shell via `set_body`.
        let mut body =
            self.assembler
                .assemble_with_callcontrol(&mut ssarepr, &regallocs, Some(callcontrol));

        // call.py:174-187 get_jitcode_calldescr:
        //   FUNC = lltype.typeOf(fnptr).TO
        //   NON_VOID_ARGS = [ARG for ARG in FUNC.ARGS if ARG is not lltype.Void]
        //   calldescr = self.cpu.calldescrof(FUNC, tuple(NON_VOID_ARGS),
        //                                    FUNC.RESULT, EffectInfo.MOST_GENERAL)
        // Source of truth for `result_type` is the declared return type
        // registered on `CallControl` (mirrors RPython's `FUNC.RESULT`,
        // which comes from `getfunctionptr(graph)._obj`'s lltype). The
        // CFG terminator scan stays as a `debug_assert!` cross-check so
        // graphs that disagree with their declared signature surface
        // immediately.
        {
            let start_block = rewritten.graph.block(rewritten.graph.startblock);
            let mut arg_classes = String::new();
            for arg_id in &start_block.inputargs {
                match ssarepr.value_kinds.get(arg_id) {
                    Some(RegKind::Int) => arg_classes.push('i'),
                    Some(RegKind::Ref) => arg_classes.push('r'),
                    Some(RegKind::Float) => arg_classes.push('f'),
                    None => arg_classes.push('i'),
                }
            }
            let cfg_kind = graph_result_kind(&rewritten.graph, &ssarepr.value_kinds);
            let declared_kind = callcontrol.declared_return_kind(path);
            let result_type = declared_kind.unwrap_or(cfg_kind);
            // Cross-check: when both sources are present they must agree,
            // with one pre-existing exception. RPython `call.py:182-187
            // get_jitcode_calldescr` derives FUNC.RESULT from the declared
            // Rust-side return type (via `function_return_types`).
            // `graph_result_kind` independently walks the CFG and reports
            // the coloring produced by the rtyper. In pyre these two
            // sources can disagree specifically on `i ↔ r`: PyObjectRef
            // is a pointer (declared as `r`) but some helper graphs
            // (e.g. `unwrap_cell`, several `CellObject` accessors) return
            // it through an integer-tagged path that the rtyper colors
            // as `i`. Neither side is wrong — RPython's `lltype`
            // unification chooses `r` for the call descriptor while
            // pyre's coloring chooses `i` for the SSA value. `v` (void)
            // mismatches are also allowed for synthesized graphs. Any
            // OTHER mismatch (e.g. i ↔ f, r ↔ f) is still a bug.
            debug_assert!(
                declared_kind.is_none_or(|d| {
                    d == cfg_kind
                        || cfg_kind == 'v'
                        || (d == 'r' && cfg_kind == 'i')
                        || (d == 'i' && cfg_kind == 'r')
                }),
                "graph {} declared FUNC.RESULT={} but CFG return kind is {}",
                rewritten.graph.name,
                declared_kind.unwrap(),
                cfg_kind,
            );
            body.calldescr = crate::jitcode::BhCallDescr::from_arg_classes(
                arg_classes,
                result_type,
                majit_ir::descr::EffectInfo::MOST_GENERAL,
            );
        }

        // Commit the body to the pre-allocated `Arc<JitCode>` shell.
        // RPython mutates the JitCode in place; pyre uses `OnceLock`
        // so that shells handed out earlier (e.g. into
        // `JitDriverStaticData.mainjitcode` by `grab_initial_jitcodes`)
        // see the same body without locking.
        jitcode.set_body(body);

        // RPython `codewriter.py:68 jitcode.index = index` — assign the
        // dense position in `all_jitcodes[]` immediately after
        // `assembler.assemble(...)` (upstream line 67). The
        // `in_order[i].index == i` invariant asserted by
        // `CallControl::collect_jitcodes_in_alloc_order` (and mirrored in
        // `test_phase_f_all_jitcodes::all_jitcodes_indices_match_alloc_order`)
        // depends on the caller having passed `index = finished_jitcodes.len()`
        // BEFORE appending.
        jitcode.set_index(index);

        if self.debug {
            // RPython `codewriter.py:71-72` → `print_ssa_repr(ssarepr,
            // portal_jd, verbose)` → `log.dot()` (default, verbose=False)
            // or `print(format_assembler(ssarepr))` (verbose=True). Pyre
            // currently mirrors only the low-noise branch: one line per
            // jitcode with the name, analogous to upstream's udir
            // filename (`codewriter.py:122-125
            // dir.join(name+extra).write(format_assembler(ssarepr))`).
            // The `verbose` parameter is plumbed to match the upstream
            // signature; the high-verbosity branch lands when
            // `format_assembler` is ported.
            let _ = verbose;
            let _ = &ssarepr;
            eprintln!("[CodeWriter] {}", jitcode.name);
        }
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// Full pipeline: grab_initial_jitcodes → enum_pending_graphs loop → finished.
    pub fn make_jitcodes(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        // RPython: self.callcontrol.grab_initial_jitcodes() (codewriter.py:76)
        callcontrol.grab_initial_jitcodes();
        self.make_jitcodes_pending(callcontrol, config)
    }

    /// Drain pending graphs and fill each `Arc<JitCode>` shell's body.
    ///
    /// RPython codewriter.py:79-84: the enum_pending_graphs loop. Pyre
    /// stores the allocated `Arc<JitCode>` shells inside
    /// `CallControl::jitcodes`; this loop pops one path at a time and
    /// commits its body via `JitCode::set_body`. The
    /// `all_jitcodes[i].index == i` invariant (RPython codewriter.py:80)
    /// is guaranteed by `CallControl::collect_jitcodes_in_alloc_order`.
    ///
    /// `verbose` threads through to `transform_graph_to_jitcode`'s debug
    /// branch; upstream `codewriter.py:74 make_jitcodes(verbose=False)`
    /// treats `False` as the default, matching pyre's call site in
    /// `make_jitcodes` which does not expose the knob to callers yet.
    pub fn drain_pending_graphs(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) {
        // RPython: for graph, jitcode in self.callcontrol.enum_pending_graphs():
        //            self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
        //
        // RPython's enum_pending_graphs() pops from unfinished_graphs (LIFO).
        // During transform, new graphs may be discovered and added via
        // get_jitcode(). We pop one at a time to match RPython's yield semantics.
        loop {
            let Some((path, jitcode)) = callcontrol.enum_pending_graphs() else {
                break;
            };
            let Some(graph) = callcontrol.function_graphs().get(&path).cloned() else {
                // RPython `enum_pending_graphs` (codewriter.py:79-84)
                // never yields a jitcode whose graph is missing —
                // `get_jitcode()` only allocates shells for paths that
                // already live in `function_graphs`. Phase I3 restored
                // this invariant for pyre by routing
                // `handle_regular_call` through the qualified
                // `CallControl::target_to_path` (jtransform.rs:970).
                // If this branch fires, a new producer has been added
                // that bypasses `target_to_path` or inserts under an
                // alias key. Producer-side bug, not an expected
                // runtime condition.
                panic!(
                    "drain_pending_graphs: jitcode shell has no matching graph — \
                     path={:?} idx={:?} name={:?}. Producer allocated a jitcode \
                     under a path that `function_graphs` does not contain. \
                     Fix the producer (prefer `CallControl::target_to_path`) \
                     instead of silently skipping.",
                    path.segments,
                    jitcode.try_index(),
                    jitcode.name,
                );
            };
            // RPython `codewriter.py:80` passes `index = len(all_jitcodes)`,
            // i.e. the slot this jitcode will occupy AFTER the append on
            // line 81. `finished_jitcodes_len()` is pyre's equivalent read.
            let index = callcontrol.finished_jitcodes_len();
            self.transform_graph_to_jitcode(
                &graph,
                &path,
                callcontrol,
                config,
                &jitcode,
                /* verbose = */ false,
                index,
            );
            // RPython `codewriter.py:81 all_jitcodes.append(jitcode)`.
            // `transform_graph_to_jitcode` already set `jitcode.index`.
            callcontrol.finish_jitcode(jitcode.clone());

            // RPython call.py:148: jd.mainjitcode.jitdriver_sd = jd
            for jd in callcontrol.jitdrivers_sd() {
                if jd.portal_graph == path {
                    jitcode.set_jitdriver_sd(jd.index);
                }
            }
        }
    }

    /// Process all pending graphs and finalize.
    ///
    /// RPython codewriter.py:79-85: enum_pending_graphs loop + finished.
    pub fn make_jitcodes_pending(
        &mut self,
        callcontrol: &mut CallControl,
        config: &GraphTransformConfig,
    ) -> AllJitCodes {
        self.drain_pending_graphs(callcontrol, config);
        self.assembler.finished(&callcontrol.callinfocollection);
        let in_order = callcontrol.collect_jitcodes_in_alloc_order();
        let by_path = callcontrol.jitcodes().clone();
        AllJitCodes { by_path, in_order }
    }
}

impl Default for CodeWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Mirror of `FUNC.RESULT` in `rpython/jit/codewriter/call.py:181-187`.
///
/// Upstream reads `lltype.typeOf(fnptr).TO.RESULT` from the function
/// pointer type; the graph-level surface is
/// `flowspace/model.py:17-18` `graph.returnblock = Block([return_var])`,
/// where `return_var.concretetype` carries the same information.
/// pyre stores the kind in `value_kinds[returnblock.inputargs[0]]`
/// after the rtyper pass, so read it directly instead of scanning
/// block terminators.
fn graph_result_kind(
    graph: &FunctionGraph,
    value_kinds: &std::collections::HashMap<ValueId, RegKind>,
) -> char {
    let returnblock = graph.block(graph.returnblock);
    match returnblock.inputargs.first() {
        Some(vid) => match value_kinds.get(vid) {
            Some(RegKind::Int) => 'i',
            Some(RegKind::Ref) => 'r',
            Some(RegKind::Float) => 'f',
            None => 'v',
        },
        None => 'v',
    }
}
