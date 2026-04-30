//! Port of `rpython/translator/backendopt/all.py`.

use std::path::Path;
use std::rc::Rc;

use crate::config::config::{Config, ConfigValue, OptionValue};
use crate::config::translationoption::get_combined_translation_config;
use crate::flowspace::model::{
    BlockRefExt, ConstValue, Constant, FunctionGraph, GraphRef, HOST_ENV, Hlvalue, LinkRef,
};
use crate::translator::backendopt::{
    constfold, gilanalysis, inline, merge_if_blocks, removenoops, stat, storesink,
};
use crate::translator::rtyper::lltypesystem::lltype::LowLevelType;
use crate::translator::rtyper::rmodel::inputconst_from_lltype;
use crate::translator::rtyper::rtyper::{GenopResult, LowLevelOpList, RPythonTyper};
use crate::translator::simplify;
use crate::translator::tool::taskengine::TaskError;
use crate::translator::translator::TranslationContext;

/// Port of upstream `backend_optimizations(translator, graphs=None,
/// secondary=False, inline_graph_from_anywhere=False, **kwds)` at
/// `all.py:35-130`.
///
/// `kwds` is `Vec<(String, OptionValue)>` rather than `HashMap` so the
/// caller's `**kwds` order is preserved through the
/// `for key, value in kwds.iteritems():` walk at `config.py:131`.
/// Upstream RPython is Python 2; iteration order there is unspecified
/// for plain `dict`, so the Vec preserves the caller's literal argument
/// order — see [`crate::config::config::Config::set`] for the full
/// citation.
///
/// `live_config` is upstream's `translator.config` carried as
/// [`Rc<Config>`].  The local [`TranslationContext`] holds only a typed
/// snapshot, so the driver passes the live schema-driven `Rc<Config>`
/// it owns (`driver.py:194 TranslationContext(config=self.config)`).
/// When `None` is supplied we fall back to the schema defaults — that
/// path is exercised only by tests that build a translator from
/// scratch without going through the driver.
pub fn backend_optimizations(
    translator: Rc<TranslationContext>,
    graphs: Option<Vec<GraphRef>>,
    secondary: bool,
    inline_graph_from_anywhere: bool,
    kwds: Vec<(String, OptionValue)>,
    live_config: Option<&Rc<Config>>,
) -> Result<(), TaskError> {
    // Upstream `all.py:43-44`:
    // `config = translator.config.translation.backendopt.copy(as_default=True)`
    // then `config.set(**kwds)`.
    let config = backendopt_config(kwds, live_config)?;

    // Upstream `all.py:46-47`: `graphs is None` falls back to
    // `translator.graphs`.
    let graphs = graphs.unwrap_or_else(|| translator.graphs.borrow().clone());

    // Upstream `all.py:48-49`:
    //     for graph in graphs:
    //         assert not getattr(graph, '_seen_by_the_backend', False)
    for graph in &graphs {
        if graph.borrow()._seen_by_the_backend.get() {
            return Err(TaskError {
                message: format!(
                    "all.py:48 backend_optimizations: graph {:?} already \
                     seen by the C backend",
                    graph.borrow().name
                ),
            });
        }
    }

    // Upstream `all.py:51-130` runs each sub-pass in pipeline order;
    // the function returns implicitly after the last pass. The Rust
    // port runs every ported pass in upstream order and surfaces an
    // unported pass as a `TaskError` when (and only when) the live
    // config requests it — exactly the upstream "pass raises mid-
    // pipeline" semantic. Earlier "collect every missing leaf up
    // front" was a NEW-DEVIATION that skipped the ported passes
    // entirely.

    // Upstream `:51-53 print_statistics`. The first emission carries
    // the literal `"per-graph.txt"` save-details path (only the
    // pre-optimisation summary, never the post-pass calls below).
    if boolopt(&config, "print_statistics")? {
        print_statistics(
            &translator,
            "before optimizations",
            Some(Path::new("per-graph.txt")),
        );
    }

    // Upstream `:55-57 replace_we_are_jitted`.
    if boolopt(&config, "replace_we_are_jitted")? {
        for graph in &graphs {
            constfold::replace_we_are_jitted(&graph.borrow());
        }
    }

    // Upstream `:59-61 remove_asserts`.
    if boolopt(&config, "remove_asserts")? {
        constfold_pass(&config, &graphs)?;
        remove_asserts(&translator, &graphs)?;
    }

    // Upstream `:63-66 really_remove_asserts → removenoops.remove_debug_assert`.
    // Comment at upstream `:66`: "the dead operations will be killed
    // by the remove_obvious_noops below".
    if boolopt(&config, "really_remove_asserts")? {
        for graph in &graphs {
            removenoops::remove_debug_assert(&graph.borrow());
        }
    }

    // Upstream `:69-80 remove_obvious_noops()` (first invocation).
    remove_obvious_noops(&config, &translator, &graphs)?;

    // Upstream `:82-92 inline + mallocs phase`.
    let inline_on = boolopt(&config, "inline")?;
    let mallocs_on = boolopt(&config, "mallocs")?;
    if inline_on || mallocs_on {
        // Upstream `:83 heuristic = get_function(config.inline_heuristic)`.
        let heuristic_name = stropt(&config, "inline_heuristic")?.unwrap_or_else(|| {
            "rpython.translator.backendopt.inline.inlining_heuristic".to_string()
        });
        let heuristic = get_function(&heuristic_name)?;
        // Upstream `:84-87 if config.inline: threshold =
        // config.inline_threshold else: threshold = 0`.
        let threshold = if inline_on {
            floatopt(&config, "inline_threshold")?
        } else {
            0.0
        };
        // Upstream `:88-91 inline_malloc_removal_phase(...)`. The
        // `call_count_pred` slot is `None` here — upstream's
        // `:84-91` non-profile branch never supplies one. The
        // profile-based branch at `:99-113` would build a counter-
        // backed predicate and call this same helper; pyre's
        // profile branch is still gated as a TaskError below until
        // `translator.driver_instrument_result` lands.
        inline_malloc_removal_phase(
            &config,
            &translator,
            &graphs,
            threshold,
            heuristic,
            None,
            inline_graph_from_anywhere,
        )?;
        // Upstream `:92 constfold(config, graphs)`.
        constfold_pass(&config, &graphs)?;
    }

    // Upstream `:94-97 storesink phase`.
    if boolopt(&config, "storesink")? {
        remove_obvious_noops(&config, &translator, &graphs)?;
        for graph in &graphs {
            storesink::storesink_graph(&graph.borrow());
        }
    }

    // Upstream `:99-113 profile_based_inline`.
    //
    // The static pieces of this branch are ported:
    // `inline::instrument_inline_candidates` lives at
    // `inline.rs::instrument_inline_candidates`, the `call_count_pred`
    // carrier is `inline::CallCountPred` (`Rc<RefCell<dyn FnMut>>`),
    // and `inline_malloc_removal_phase` accepts the predicate
    // verbatim. `get_function(config.profile_based_inline_heuristic)`
    // is wired through the same closed-world dotted-name registry the
    // non-profile branch uses (`get_function` below). The remaining
    // blocker is the runtime piece:
    // `translator.driver_instrument_result(filename)` (upstream
    // `driver.py:driver_instrument_result`) compiles the instrumented
    // graph through the C backend, runs it, and reads back per-label
    // counters. Pyre's C-backend driver is not ported, so the path
    // stays gated as a `TaskError` — but the `get_function` lookup
    // runs first, matching upstream's `:101` line order. Convergence
    // path: port the C-backend instrument-and-run pipeline, then
    // replace this gate with
    //     let counters = translator.driver_instrument_result(...)?;
    //     let pred: CallCountPred = Rc::new(RefCell::new(
    //         move |label| (label as usize) < counters.len()
    //             && counters[label as usize] > 250,
    //     ));
    //     inline::instrument_inline_candidates(&graphs, threshold, &translator);
    //     inline_malloc_removal_phase(&config, &translator, &graphs,
    //                                  threshold, profile_heuristic,
    //                                  Some(pred), inline_graph_from_anywhere)?;
    if stropt(&config, "profile_based_inline")?.is_some() && !secondary {
        // Upstream `:101 heuristic = get_function(config.profile_based_inline_heuristic)`.
        // Surface registry misses ahead of the C-backend gate so
        // misconfigured dotted names fail fast with the same shape
        // the non-profile branch uses.
        let profile_heuristic_name = stropt(&config, "profile_based_inline_heuristic")?
            .unwrap_or_else(|| {
                "rpython.translator.backendopt.inline.inlining_heuristic".to_string()
            });
        let _profile_heuristic = get_function(&profile_heuristic_name)?;
        return Err(TaskError {
            message: "all.py:103-104 profile_based_inline: static pieces ported \
                      (get_function / instrument_inline_candidates / CallCountPred / \
                      inline_malloc_removal_phase signature); blocked on \
                      translator.driver_instrument_result (driver.py) — \
                      pyre's C-backend instrument-and-run pipeline is unported"
                .to_string(),
        });
    }

    // Upstream `:114 constfold(config, graphs)` — runs unconditionally
    // (gated only by config.constfold inside `constfold_pass`).
    constfold_pass(&config, &graphs)?;

    // Upstream `:116-119 merge_if_blocks`. The `verbose` flag
    // tracks `translator.config.translation.verbose`; when this
    // entry is invoked without a live root config (synthetic test
    // path), fall back to `False` — matching upstream's default
    // for `translation.verbose`.
    if boolopt(&config, "merge_if_blocks")? {
        let verbose = match live_config {
            Some(root) => match root.get("translation.verbose").map_err(task_error)? {
                ConfigValue::Value(OptionValue::Bool(b)) => b,
                ConfigValue::Value(OptionValue::None) => false,
                other => {
                    return Err(TaskError {
                        message: format!(
                            "all.py:119 translation.verbose: expected bool, got {other:?}"
                        ),
                    });
                }
            },
            None => false,
        };
        for graph in &graphs {
            merge_if_blocks::merge_if_blocks(&graph.borrow(), verbose);
        }
    }

    if boolopt(&config, "print_statistics")? {
        print_statistics(&translator, "after if-to-switch", None);
    }

    // Upstream `:125 remove_obvious_noops()` (second invocation).
    remove_obvious_noops(&config, &translator, &graphs)?;

    // Upstream `:127-128 for graph in graphs: checkgraph(graph)`.
    for graph in &graphs {
        crate::flowspace::model::checkgraph(&graph.borrow());
    }

    // Upstream `:130 gilanalysis.analyze(graphs, translator)`.
    //
    // `gilanalysis::analyze` constructs a `GilAnalyzer`
    // (`graphanalyze::GraphAnalyzer<bool, ()>`) and invokes
    // `analyze_direct_call` for every graph carrying
    // `_no_release_gil_`. Pyre is freethreaded, so this is not a
    // literal GIL-release check: the analyzer treats the upstream
    // flag as a no-thread-safepoint contract and rejects transitive
    // callees that close the stack, break transactions, or cross an
    // unresolved external-call boundary.
    gilanalysis::analyze(&graphs, &translator)
}

/// RPython `inline_malloc_removal_phase(config, translator, graphs,
/// inline_threshold, inline_heuristic, call_count_pred=None,
/// inline_graph_from_anywhere=False)` at `all.py:138-164`.
///
/// `call_count_pred` is the predicate `auto_inline_graphs` consults
/// when an `instrument_count`-tagged op selects the
/// profile-based-inline path (`inline.py:176-182`). Upstream's only
/// `call_count_pred=...` caller is `backend_optimizations`'s
/// `profile_based_inline` branch (`:106-113`); pyre's wrapper wires
/// the parameter through verbatim so callers can opt-in once
/// `translator.driver_instrument_result` (the runtime counter
/// supplier at `driver.py`) is ported.
pub(crate) fn inline_malloc_removal_phase(
    config: &Rc<Config>,
    translator: &Rc<TranslationContext>,
    graphs: &[GraphRef],
    inline_threshold: f64,
    inline_heuristic: fn(&GraphRef) -> (f64, bool),
    call_count_pred: Option<inline::CallCountPred>,
    inline_graph_from_anywhere: bool,
) -> Result<(), TaskError> {
    // Upstream `:143-151 if inline_threshold: log.inlining(...) ;
    // inline.auto_inline_graphs(...)`. `log.inlining` is a
    // verbose-only log call (`support.py:21-26`); skipping it is the
    // same convention as everywhere else in this module.
    if inline_threshold != 0.0 {
        inline::auto_inline_graphs(
            translator,
            graphs,
            inline_threshold,
            inline_heuristic,
            call_count_pred,
            inline_graph_from_anywhere,
        )
        .map_err(|e| TaskError {
            message: format!("all.py:148 auto_inline_graphs: {}", e.0),
        })?;

        // Upstream `:153-155 if config.print_statistics: print_statistics(...)`.
        if boolopt(config, "print_statistics")? {
            print_statistics(translator, "after inlining", None);
        }
    }

    // Upstream `:158-164 if config.mallocs: log.malloc(...) ;
    // remove_mallocs(translator, graphs); ...`.
    //
    // `malloc.py` is a 566-LOC escape-analysis pass. The driver
    // `remove_mallocs(translator, graphs)` (`malloc.py:553-566`) wraps
    // a `LLTypeMallocRemover` (`:333-547`, subclass of
    // `BaseMallocRemover`, `:26-332`) that:
    //
    //   1. Builds a per-graph `LifeTime` UnionFind (`:9-24` +
    //      `:121-?compute_lifetimes`) over malloc result vars, tracking
    //      creation- and use-points.
    //   2. Walks operations to identify `malloc` ops whose result
    //      never escapes (no aliasing into a non-removable use), then
    //      replaces field reads/writes with direct local-var access.
    //   3. Calls `removenoops.remove_same_as`,
    //      `simplify.eliminate_empty_blocks`, and
    //      `simplify.transform_dead_op_vars` to clean up.
    //
    // Local availability of deps:
    //
    //   * `tool::algo::unionfind::UnionFind` — already ported.
    //   * `simplify::transform_dead_op_vars` /
    //     `eliminate_empty_blocks` — already ported.
    //   * `removenoops::remove_same_as` — already ported.
    //   * `lltype::Struct` / array type introspection — partially
    //     ported (struct kinds + immutable_field landed; the
    //     `_arrayfld` / nested-struct interior layout the malloc
    //     remover relies on is gated on the same `_parentable` /
    //     `_parent_link` work `constfold.rs::fixup_solid` cites).
    //
    // The actual blocker is the 566 LOC of `malloc.py` itself, not a
    // missing dep — the port is a self-contained ~3-4 day chunk that
    // mostly mirrors line-by-line. It stays gated as a `TaskError`
    // because the upstream default has `mallocs=True`, so silently
    // returning `Ok(())` would mask a real configuration mismatch
    // (regression catcher: `inline_malloc_phase_surfaces_mallocs_taskerror_when_enabled`).
    //
    // Convergence path: lift `malloc.py` ~`:9-566` into
    // `backendopt/malloc.rs` as a new module; expose
    // `remove_mallocs(translator, graphs)` and call it from here.
    if boolopt(config, "mallocs")? {
        return Err(TaskError {
            message: "all.py:160 remove_mallocs: PRE-EXISTING-ADAPTATION — \
                      malloc.py (566 LOC LLTypeMallocRemover / \
                      BaseMallocRemover escape-analysis pass) is \
                      unported. Upstream default has mallocs=True so the \
                      default backendopt pipeline currently surfaces this \
                      gate. Convergence path: port malloc.py:9-566 verbatim \
                      into a new backendopt/malloc.rs module \
                      (UnionFind / simplify / removenoops / \
                      lltype.Struct deps already landed locally) and \
                      replace this gate with `malloc::remove_mallocs(\
                      translator, graphs)`."
                .to_string(),
        });
    }

    Ok(())
}

/// RPython `constfold(config, graphs)` at `all.py:133-136`.
pub(crate) fn constfold_pass(config: &Rc<Config>, graphs: &[GraphRef]) -> Result<(), TaskError> {
    if boolopt(config, "constfold")? {
        for graph in graphs {
            constfold::constant_fold_graph(&graph.borrow());
        }
    }
    Ok(())
}

/// RPython `removeassert.remove_asserts(translator, graphs)` at
/// `removeassert.py:8-53`.
pub(crate) fn remove_asserts(
    translator: &TranslationContext,
    graphs: &[GraphRef],
) -> Result<(), TaskError> {
    // Upstream `:9 rtyper = translator.rtyper`.
    let rtyper = translator.rtyper().ok_or_else(|| TaskError {
        message: "removeassert.py:9 remove_asserts: translator.rtyper is None".to_string(),
    })?;
    // Upstream `:10 excdata = rtyper.exceptiondata`.
    let excdata = rtyper.exceptiondata().map_err(|e| TaskError {
        message: format!("removeassert.py:10 exceptiondata: {e}"),
    })?;
    // Upstream `:11 clsdef =
    //     translator.annotator.bookkeeper.getuniqueclassdef(AssertionError)`.
    let annotator = translator.annotator().ok_or_else(|| TaskError {
        message: "removeassert.py:11 remove_asserts: translator.annotator is None".to_string(),
    })?;
    let assertion_error = HOST_ENV
        .lookup_builtin("AssertionError")
        .ok_or_else(|| TaskError {
            message: "removeassert.py:11 AssertionError missing from HOST_ENV".to_string(),
        })?;
    let clsdef = annotator
        .bookkeeper
        .getuniqueclassdef(&assertion_error)
        .map_err(|e| TaskError {
            message: format!("removeassert.py:11 getuniqueclassdef(AssertionError): {e}"),
        })?;
    // Upstream `:12 ll_AssertionError =
    //     excdata.get_standard_ll_exc_instance(rtyper, clsdef)`.
    let ll_assertion_error = excdata
        .get_standard_ll_exc_instance(&rtyper, Some(clsdef))
        .map_err(|e| TaskError {
            message: format!("removeassert.py:12 get_standard_ll_exc_instance: {e}"),
        })?;
    // Upstream `:13 total_count = [0, 0]`. The accumulator feeds the
    // final `log.removeassert(...)` summary at `:42-53`. Pyre's
    // `AnsiLogger("backendopt")` channel is a documented
    // PRE-EXISTING-ADAPTATION (`support.rs` module doc); the
    // accumulator structure is preserved here so the body mirrors
    // upstream line-for-line and lights up the moment the logger
    // lands.
    let mut total_count: [usize; 2] = [0, 0];

    for graph in graphs {
        // Upstream `:16-17 count = 0; morework = True`.
        let mut count = 0usize;
        loop {
            let mut morework = false;
            // Upstream `:20-21 eliminate_empty_blocks(graph);
            // join_blocks(graph)`.
            {
                let graph_b = graph.borrow();
                simplify::eliminate_empty_blocks(&graph_b);
                simplify::join_blocks(&graph_b);
            }
            let links = graph.borrow().iterlinks();
            for link in links {
                let is_assertion_link = {
                    let graph_b = graph.borrow();
                    assertion_link_matches(&graph_b, &link, &ll_assertion_error)
                };
                if !is_assertion_link {
                    continue;
                }
                // Upstream `:26-33 if kill_assertion_link(graph, link):
                //     count += 1; morework = True; break
                // else: total_count[0] += 1
                //     if verbose: log.removeassert("cannot remove ...")`.
                if kill_assertion_link(&graph.borrow(), &link, &rtyper)? {
                    count += 1;
                    morework = true;
                    break;
                } else {
                    total_count[0] += 1;
                    // log.removeassert("cannot remove ...") — no-op
                    // (logger unported).
                }
            }
            if !morework {
                break;
            }
        }
        // Upstream `:34-40 if count: total_count[1] += count; ...
        // checkgraph(graph)`.
        if count != 0 {
            total_count[1] += count;
            // log.removeassert("removed %d asserts in %s") — no-op.
            crate::flowspace::model::checkgraph(&graph.borrow());
        }
    }
    // Upstream `:41-53 if total_count[0] == 0: msg = ... else: ...
    // if msg is not None: log.removeassert(msg)`. Logger no-op
    // gates the entire emission; the accumulator computation is
    // still preserved above.
    let _ = total_count;
    Ok(())
}

fn assertion_link_matches(
    graph: &FunctionGraph,
    link: &LinkRef,
    ll_assertion_error: &Constant,
) -> bool {
    let link_b = link.borrow();
    let Some(target) = &link_b.target else {
        return false;
    };
    if !Rc::ptr_eq(target, &graph.exceptblock) {
        return false;
    }
    matches!(
        link_b.args.get(1).and_then(|arg| arg.as_ref()),
        Some(Hlvalue::Constant(c)) if c == ll_assertion_error
    )
}

/// RPython `removeassert.kill_assertion_link(graph, link)` at
/// `removeassert.py:38-62`.
fn kill_assertion_link(
    graph: &FunctionGraph,
    link: &LinkRef,
    rtyper: &Rc<RPythonTyper>,
) -> Result<bool, TaskError> {
    let block = link
        .borrow()
        .prevblock
        .as_ref()
        .and_then(|prev| prev.upgrade())
        .ok_or_else(|| TaskError {
            message: "removeassert.py:39 kill_assertion_link: link.prevblock missing".to_string(),
        })?;
    let mut exits: Vec<LinkRef> = block.borrow().exits.clone();
    if exits.len() <= 1 {
        return Ok(false);
    }
    let link_index = exits
        .iter()
        .position(|candidate| Rc::ptr_eq(candidate, link))
        .ok_or_else(|| TaskError {
            message: "removeassert.py:39 kill_assertion_link: link not in prevblock.exits"
                .to_string(),
        })?;
    let mut remove_condition = exits.len() == 2;
    if block.borrow().canraise() {
        if link_index == 0 {
            return Ok(false);
        }
    } else {
        let exitswitch = block.borrow().exitswitch.clone();
        if exitswitch.as_ref().and_then(hlvalue_concretetype).as_ref() != Some(&LowLevelType::Bool)
        {
            remove_condition = false;
        } else {
            if !remove_condition {
                return Err(TaskError {
                    message:
                        "removeassert.py:49 kill_assertion_link: bool exitswitch without two exits"
                            .to_string(),
                });
            }
            let exitswitch = exitswitch.expect("checked above");
            let mut newops = LowLevelOpList::new(Rc::clone(rtyper), None);
            let condition = if hlvalue_is_true(link.borrow().exitcase.as_ref()) {
                let inverted = newops
                    .genop(
                        "bool_not",
                        vec![exitswitch],
                        GenopResult::LLType(LowLevelType::Bool),
                    )
                    .expect("bool_not has Bool result");
                Hlvalue::Variable(inverted)
            } else {
                exitswitch
            };
            let msg = format!("assertion failed in {}", graph.name);
            let c_msg = inputconst_from_lltype(&LowLevelType::Void, &ConstValue::byte_str(msg))
                .map_err(|e| TaskError {
                    message: format!("removeassert.py:55 inputconst(Void, msg): {e}"),
                })?;
            newops.genop(
                "debug_assert",
                vec![condition, Hlvalue::Constant(c_msg)],
                GenopResult::Void,
            );
            block.borrow_mut().operations.extend(newops.ops);
        }
    }
    exits.remove(link_index);
    if remove_condition {
        block.borrow_mut().exitswitch = None;
        if let Some(first) = exits.first() {
            let mut first_b = first.borrow_mut();
            first_b.exitcase = None;
            first_b.llexitcase = None;
        }
    }
    block.recloseblock(exits);
    Ok(true)
}

fn hlvalue_concretetype(value: &Hlvalue) -> Option<LowLevelType> {
    match value {
        Hlvalue::Variable(v) => v.concretetype(),
        Hlvalue::Constant(c) => c.concretetype.clone(),
    }
}

fn hlvalue_is_true(value: Option<&Hlvalue>) -> bool {
    match value {
        Some(Hlvalue::Constant(c)) => match &c.value {
            ConstValue::Bool(value) => *value,
            ConstValue::Int(value) => *value != 0,
            ConstValue::None => false,
            _ => true,
        },
        Some(Hlvalue::Variable(_)) => true,
        None => false,
    }
}

/// RPython nested `remove_obvious_noops()` at `all.py:69-80`.
pub(crate) fn remove_obvious_noops(
    config: &Rc<Config>,
    translator: &TranslationContext,
    graphs: &[GraphRef],
) -> Result<(), TaskError> {
    for graph in graphs {
        let graph = graph.borrow();
        removenoops::remove_same_as(&graph);
        simplify::eliminate_empty_blocks(&graph);
        simplify::transform_dead_op_vars(&graph, Some(translator));
        removenoops::remove_duplicate_casts(&graph, translator);
    }
    if boolopt(config, "print_statistics")? {
        print_statistics(translator, "after no-op removal", None);
    }
    Ok(())
}

/// Upstream `print("after %s:" % phase); print_statistics(translator.graphs[0],
/// translator, ...)` at `all.py:51-53` / `:76-78` / `:121-123` /
/// `:153-155` / `:162-164`. Only the first call (pre-optimisation)
/// takes a non-default `save_per_graph_details = "per-graph.txt"` —
/// every later call defaults to `None`.
fn print_statistics(
    translator: &TranslationContext,
    phase: &str,
    save_per_graph_details: Option<&Path>,
) {
    println!("{phase}:");
    let graphs = translator.graphs.borrow();
    if let Some(entry) = graphs.first() {
        // Upstream call sites pass `ignore_stack_checks=False` (the
        // default) at every call site in `all.py`.
        stat::print_statistics(entry, translator, save_per_graph_details, false);
    }
}

fn backendopt_config(
    kwds: Vec<(String, OptionValue)>,
    live_config: Option<&Rc<Config>>,
) -> Result<Rc<Config>, TaskError> {
    // Upstream `all.py:43`:
    // `config = translator.config.translation.backendopt.copy(as_default=True)`.
    // Take the backendopt subgroup off whichever `Rc<Config>` the
    // caller is willing to share — the live one when available, the
    // fresh schema otherwise.
    let owned_root: Option<Rc<Config>> = if live_config.is_none() {
        Some(get_combined_translation_config(None, None, None, true).map_err(task_error)?)
    } else {
        None
    };
    let root: &Rc<Config> = live_config.unwrap_or_else(|| owned_root.as_ref().unwrap());
    let backendopt = match root.get("translation.backendopt").map_err(task_error)? {
        ConfigValue::SubConfig(config) => config.copy(true),
        other => {
            return Err(TaskError {
                message: format!("all.py:43 expected backendopt SubConfig, got {other:?}"),
            });
        }
    };
    backendopt.set(kwds).map_err(task_error)?;
    Ok(backendopt)
}

fn boolopt(config: &Rc<Config>, name: &str) -> Result<bool, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Bool(value)) => Ok(value),
        ConfigValue::Value(OptionValue::None) => Ok(false),
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected bool, got {other:?}"),
        }),
    }
}

fn floatopt(config: &Rc<Config>, name: &str) -> Result<f64, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Float(value)) => Ok(value),
        ConfigValue::Value(OptionValue::Int(value)) => Ok(value as f64),
        ConfigValue::Value(OptionValue::Bool(value)) => Ok(if value { 1.0 } else { 0.0 }),
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected float, got {other:?}"),
        }),
    }
}

fn stropt(config: &Rc<Config>, name: &str) -> Result<Option<String>, TaskError> {
    match config.get(name).map_err(task_error)? {
        ConfigValue::Value(OptionValue::Str(value)) if !value.is_empty() => Ok(Some(value)),
        ConfigValue::Value(OptionValue::Choice(value)) if !value.is_empty() => Ok(Some(value)),
        ConfigValue::Value(OptionValue::None) => Ok(None),
        ConfigValue::Value(OptionValue::Str(_)) | ConfigValue::Value(OptionValue::Choice(_)) => {
            Ok(None)
        }
        other => Err(TaskError {
            message: format!("all.py backendopt config {name}: expected string, got {other:?}"),
        }),
    }
}

fn task_error(error: impl std::fmt::Debug) -> TaskError {
    TaskError {
        message: format!("all.py backend_optimizations config error: {error:?}"),
    }
}

/// RPython `get_function(dottedname)` at `all.py:19-33`.
///
/// Upstream resolves an arbitrary dotted name through `__import__`
/// + `getattr`. Pyre has no Python-style import resolver, so this
/// helper carries the closed-world equivalent: a registry mapping
/// the dotted names that upstream config defaults ship into the
/// already-ported Rust callable. Misses surface as `TaskError`
/// (the same shape upstream's `Exception("Function %s not found")`
/// at `:31` produces); future heuristic ports register an entry
/// alongside their landing commit.
///
/// The two production callers live at `:83 inline_heuristic` and
/// `:101 profile_based_inline_heuristic`. Both default to
/// `"rpython.translator.backendopt.inline.inlining_heuristic"`
/// (`translationoption.py:216`, `:239`) which maps to
/// [`inline::inlining_heuristic`].
fn get_function(dottedname: &str) -> Result<fn(&GraphRef) -> (f64, bool), TaskError> {
    match dottedname {
        "rpython.translator.backendopt.inline.inlining_heuristic" => {
            Ok(inline::inlining_heuristic as fn(&GraphRef) -> (f64, bool))
        }
        other => Err(TaskError {
            message: format!(
                "all.py:31 get_function: dotted name {other:?} is not registered \
                 in pyre's closed-world heuristic registry. Pyre has no \
                 __import__ shim; register the upstream callable in \
                 backendopt::all::get_function alongside the port."
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::annrpython::RPythonAnnotator;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, LAST_EXCEPTION, Link,
        SpaceOperation, Variable,
    };
    use crate::translator::rtyper::rtyper::RPythonTyper;
    use std::cell::RefCell;

    fn fixture_translator() -> Rc<TranslationContext> {
        Rc::new(TranslationContext::new())
    }

    fn graph_ref(graph: FunctionGraph) -> GraphRef {
        Rc::new(RefCell::new(graph))
    }

    /// Default backendopt config has `inline`, `mallocs` True.
    /// `mallocs` remains unported locally — the malloc.py port has
    /// not landed, so `inline_malloc_removal_phase` surfaces it as a
    /// `TaskError` at upstream `:160`. Tests disable `mallocs` so the
    /// structural shell exercises every other pass — including the
    /// now-ported `inline.auto_inline_graphs`, `storesink`, and
    /// `merge_if_blocks`.
    fn ported_only_kwds() -> Vec<(String, OptionValue)> {
        vec![("mallocs".to_string(), OptionValue::Bool(false))]
    }

    fn make_int_constfold_graph() -> (Variable, GraphRef) {
        let r = Variable::named("r");
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(r.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(r.clone())],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        (r, graph_ref(graph))
    }

    #[test]
    fn backendopt_runs_to_terminal_gilanalysis() {
        // Upstream `all.py:35-130` runs the full pipeline. The
        // local port has `inline` / `mallocs` /
        // `profile_based_inline` gated off via config kwds in
        // `ported_only_kwds`; every other pass — including
        // `gilanalysis::analyze` at the tail (`:130`) — is ported.
        // This fixture carries no `_no_release_gil_` marker, so the
        // freethreaded safepoint analysis has no roots to reject.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        backend_optimizations(
            fixture_translator(),
            Some(vec![graph]),
            false,
            false,
            ported_only_kwds(),
            None,
        )
        .expect("backendopt should run cleanly through the gilanalysis tail");
    }

    #[test]
    fn remove_obvious_noops_helper_drops_same_as_op() {
        // The pipeline-helper used by `backend_optimizations` once
        // every leaf lands. Tested directly so the partial pipeline
        // can still be exercised without going through the
        // fail-fast public entry point.
        let x = Variable::named("x");
        let y = Variable::named("y");
        let start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "same_as",
            vec![Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(y.clone()),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(y)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);
        let translator = fixture_translator();
        let config = backendopt_config(ported_only_kwds(), None).expect("config");

        remove_obvious_noops(&config, &translator, &[graph.clone()]).expect("remove_obvious_noops");

        let borrowed = graph.borrow();
        assert!(borrowed.startblock.borrow().operations.is_empty());
        let link_arg = borrowed.startblock.borrow().exits[0].borrow().args[0]
            .clone()
            .expect("link arg");
        assert_eq!(link_arg, Hlvalue::Variable(x));
    }

    #[test]
    fn constfold_pass_helper_folds_int_add() {
        let (_r, graph) = make_int_constfold_graph();
        let config = backendopt_config(ported_only_kwds(), None).expect("config");

        constfold_pass(&config, &[graph.clone()]).expect("constfold_pass");

        let borrowed = graph.borrow();
        assert!(borrowed.startblock.borrow().operations.is_empty());
        let link_arg = borrowed.startblock.borrow().exits[0].borrow().args[0]
            .clone()
            .expect("link arg");
        assert!(matches!(
            link_arg,
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(3),
                ..
            })
        ));
    }

    #[test]
    fn kill_assertion_link_drops_canraise_assertion_exit() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let v = Variable::named("v");
        let result = Variable::named("result");
        let body = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", body.clone());
        body.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Variable(v)],
            Hlvalue::Variable(result.clone()),
        ));
        body.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::new(ConstValue::Atom(
            LAST_EXCEPTION.clone(),
        ))));
        let ok = Link::new(
            vec![Hlvalue::Variable(result)],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let ll_assertion = Constant::with_concretetype(ConstValue::Int(42), LowLevelType::Signed);
        let err = Link::new(
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::builtin("AssertionError"))),
                Hlvalue::Constant(ll_assertion.clone()),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::builtin(
                "AssertionError",
            )))),
        )
        .into_ref();
        body.closeblock(vec![ok.clone(), err.clone()]);

        assert!(assertion_link_matches(&graph, &err, &ll_assertion));
        assert!(kill_assertion_link(&graph, &err, &rtyper).expect("kill assertion link"));

        assert_eq!(body.borrow().exits.len(), 1);
        assert!(body.borrow().exitswitch.is_none());
        assert!(Rc::ptr_eq(
            body.borrow().exits[0].borrow().target.as_ref().unwrap(),
            &graph.returnblock
        ));
        assert!(body.borrow().exits[0].borrow().exitcase.is_none());
    }

    #[test]
    fn kill_assertion_link_rewrites_bool_switch_to_debug_assert() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let rtyper = Rc::new(RPythonTyper::new(&ann));
        let cond = Variable::named("cond");
        cond.set_concretetype(Some(LowLevelType::Bool));
        let body = Block::shared(vec![Hlvalue::Variable(cond.clone())]);
        let graph = FunctionGraph::new("f", body.clone());
        body.borrow_mut().exitswitch = Some(Hlvalue::Variable(cond));
        let ok = Link::new(
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Some(graph.returnblock.clone()),
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(false),
                LowLevelType::Bool,
            ))),
        )
        .into_ref();
        let ll_assertion = Constant::with_concretetype(ConstValue::Int(7), LowLevelType::Signed);
        let err = Link::new(
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::builtin("AssertionError"))),
                Hlvalue::Constant(ll_assertion.clone()),
            ],
            Some(graph.exceptblock.clone()),
            Some(Hlvalue::Constant(Constant::with_concretetype(
                ConstValue::Bool(true),
                LowLevelType::Bool,
            ))),
        )
        .into_ref();
        body.closeblock(vec![ok.clone(), err.clone()]);

        assert!(kill_assertion_link(&graph, &err, &rtyper).expect("kill assertion link"));

        let body_b = body.borrow();
        assert_eq!(body_b.operations.len(), 2);
        assert_eq!(body_b.operations[0].opname, "bool_not");
        assert_eq!(body_b.operations[1].opname, "debug_assert");
        assert_eq!(body_b.exits.len(), 1);
        assert!(body_b.exitswitch.is_none());
        assert!(body_b.exits[0].borrow().exitcase.is_none());
    }

    #[test]
    fn inline_malloc_phase_runs_auto_inline_graphs_then_constfold() {
        // `inline=true, mallocs=false` exercises the wired
        // `inline_malloc_removal_phase` (upstream `:88-91`) followed
        // by the `constfold(config, graphs)` cleanup at upstream
        // `:92`. The fixture has no inter-graph calls, so
        // `auto_inline_graphs`'s callgraph is empty and the pass is
        // a no-op — the `int_add(1, 2)` is folded by the trailing
        // `constfold_pass`.
        let (_r, graph) = make_int_constfold_graph();
        backend_optimizations(
            fixture_translator(),
            Some(vec![graph.clone()]),
            false,
            false,
            ported_only_kwds(),
            None,
        )
        .expect("backendopt with inline=true should run cleanly");

        let borrowed = graph.borrow();
        assert!(borrowed.startblock.borrow().operations.is_empty());
        let link_arg = borrowed.startblock.borrow().exits[0].borrow().args[0]
            .clone()
            .expect("link arg");
        assert!(matches!(
            link_arg,
            Hlvalue::Constant(Constant {
                value: ConstValue::Int(3),
                ..
            })
        ));
    }

    #[test]
    fn inline_malloc_phase_surfaces_mallocs_taskerror_when_enabled() {
        // `mallocs=true` (the upstream default) is unported because
        // `malloc.py::remove_mallocs` has not landed.
        // `inline_malloc_removal_phase` surfaces a `TaskError` when
        // the gate runs, matching the convention of every other
        // unported pass in this module.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        let result = backend_optimizations(
            fixture_translator(),
            Some(vec![graph]),
            false,
            false,
            // Default `mallocs=true`, default `inline=true`.
            Vec::new(),
            None,
        );

        match result {
            Err(e) => assert!(
                e.message.contains("remove_mallocs"),
                "expected remove_mallocs TaskError, got {:?}",
                e.message
            ),
            Ok(()) => panic!("expected TaskError when mallocs=true is enabled"),
        }
    }

    #[test]
    fn inline_heuristic_other_than_default_returns_taskerror() {
        // Upstream `get_function(dottedname)` at `all.py:19-33`
        // uses `__import__` + `getattr` to resolve any dotted name.
        // Pyre's closed-world registry only carries the names that
        // upstream config defaults ship; a misconfigured dotted
        // name surfaces the `:31` "Function %s not found"
        // equivalent rather than getting silently mapped.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        let kwds = vec![
            ("mallocs".to_string(), OptionValue::Bool(false)),
            (
                "inline_heuristic".to_string(),
                OptionValue::Str("custom.heuristic.path".to_string()),
            ),
        ];

        let result = backend_optimizations(
            fixture_translator(),
            Some(vec![graph]),
            false,
            false,
            kwds,
            None,
        );

        match result {
            Err(e) => assert!(
                e.message.contains("custom.heuristic.path") && e.message.contains("get_function"),
                "expected get_function TaskError carrying the unresolved dotted name, got {:?}",
                e.message
            ),
            Ok(()) => panic!("expected TaskError on non-default inline_heuristic"),
        }
    }

    #[test]
    fn get_function_resolves_default_inlining_heuristic() {
        let resolved = get_function("rpython.translator.backendopt.inline.inlining_heuristic")
            .expect("default name must resolve");
        // The function pointer comparison nails the registry binding
        // to the locally-ported `inline::inlining_heuristic`.
        assert_eq!(
            resolved as usize,
            inline::inlining_heuristic as fn(&GraphRef) -> (f64, bool) as usize
        );
    }

    #[test]
    fn get_function_unknown_dotted_name_yields_taskerror() {
        let err = get_function("not.a.real.heuristic").expect_err("unknown name must fail");
        assert!(err.message.contains("not.a.real.heuristic"));
        assert!(err.message.contains("registry"));
    }

    #[test]
    fn profile_based_inline_surfaces_get_function_miss_before_runtime() {
        // Upstream `:101 heuristic = get_function(...)` runs before
        // `:103 counters = translator.driver_instrument_result(...)`.
        // A misconfigured `profile_based_inline_heuristic` must
        // surface the registry miss without falling through to the
        // unported C-backend gate.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        let kwds = vec![
            ("mallocs".to_string(), OptionValue::Bool(false)),
            (
                "profile_based_inline".to_string(),
                OptionValue::Str("any-non-empty-arg".to_string()),
            ),
            (
                "profile_based_inline_heuristic".to_string(),
                OptionValue::Str("not.a.real.heuristic".to_string()),
            ),
        ];

        let err = backend_optimizations(
            fixture_translator(),
            Some(vec![graph]),
            false,
            false,
            kwds,
            None,
        )
        .expect_err("registry miss must surface as TaskError");
        assert!(
            err.message.contains("not.a.real.heuristic") && err.message.contains("get_function"),
            "expected get_function miss, got {:?}",
            err.message
        );
    }

    #[test]
    fn profile_based_inline_default_heuristic_surfaces_runtime_blocker() {
        // With `profile_based_inline=Some(_)` and the default
        // `profile_based_inline_heuristic`, the registry resolves
        // cleanly — the next blocker is the unported C-backend
        // `translator.driver_instrument_result`.
        let start = Block::shared(vec![]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let graph = graph_ref(graph);

        let kwds = vec![
            ("mallocs".to_string(), OptionValue::Bool(false)),
            (
                "profile_based_inline".to_string(),
                OptionValue::Str("any-non-empty-arg".to_string()),
            ),
        ];

        let err = backend_optimizations(
            fixture_translator(),
            Some(vec![graph]),
            false,
            false,
            kwds,
            None,
        )
        .expect_err("C-backend driver runtime is unported");
        assert!(
            err.message.contains("driver_instrument_result"),
            "expected driver_instrument_result TaskError, got {:?}",
            err.message
        );
    }
}
