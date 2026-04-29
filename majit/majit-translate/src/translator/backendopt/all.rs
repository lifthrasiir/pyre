//! Port of `rpython/translator/backendopt/all.py`.

use std::path::Path;
use std::rc::Rc;

use crate::config::config::{Config, ConfigValue, OptionValue};
use crate::config::translationoption::get_combined_translation_config;
use crate::flowspace::model::GraphRef;
use crate::translator::backendopt::{
    constfold, gilanalysis, inline, merge_if_blocks, removenoops, stat, storesink,
};
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
    //
    // `removeassert.py:8 remove_asserts(translator, graphs)` walks each
    // graph's `iterlinks()` looking for links that target
    // `graph.exceptblock` with a constant `ll_AssertionError` value,
    // then strips the assertion via `kill_assertion_link`. Both
    // `iterlinks` and `recloseblock` are already in
    // `flowspace::model.rs`, but the body needs four rtyper-side
    // primitives that are still skeletons:
    //
    //   * `RPythonTyper.exceptiondata.get_standard_ll_exc_instance(
    //     rtyper, clsdef)` (`exceptiondata.py:get_standard_ll_exc_instance`)
    //     — returns the canonical `ll_AssertionError` constant the
    //     scan compares against.
    //   * `bookkeeper.getuniqueclassdef(AssertionError)`
    //     (`bookkeeper.py:430`) — pyre's annotator already exposes
    //     `getuniqueclassdef`, but the `AssertionError` host class
    //     must be registered in the bookkeeper namespace first.
    //   * `LowLevelOpList()` + `newops.genop('bool_not', ...)`
    //     (`rtyper.py LowLevelOpList`) — `kill_assertion_link` rewrites
    //     `if cond: raise AssertionError` to `debug_assert(not cond,
    //     msg)`. Requires the LLOpList builder.
    //   * `inputconst(lltype.Void, msg)` (`rtyper.py inputconst`) —
    //     constructs the `Void` message constant.
    //
    // Convergence path: once `rtyper::exceptiondata` lands the standard
    // exception singletons + `LowLevelOpList`/`inputconst`, lift the
    // 56-LOC `removeassert.py` body verbatim. Comment-only TaskError
    // until then.
    if boolopt(&config, "remove_asserts")? {
        constfold_pass(&config, &graphs)?;
        return Err(TaskError {
            message: "all.py:61 remove_asserts: requires \
                      rtyper.exceptiondata.get_standard_ll_exc_instance + \
                      LowLevelOpList + inputconst (rtyper skeleton unported)"
                .to_string(),
        });
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
        // Pyre has no Python-style dotted-name `__import__` resolver
        // (`get_function` at upstream `:19-33`); the only heuristic
        // ever shipped by upstream is
        // `"rpython.translator.backendopt.inline.inlining_heuristic"`,
        // so the port matches that literal and falls through to the
        // ported `inlining_heuristic` callable.  Any other dotted
        // name would be a NEW-DEVIATION at the call site upstream
        // never exercises, so it surfaces as a `TaskError` rather
        // than getting silently mapped.
        let heuristic_name = stropt(&config, "inline_heuristic")?.unwrap_or_else(|| {
            "rpython.translator.backendopt.inline.inlining_heuristic".to_string()
        });
        if heuristic_name != "rpython.translator.backendopt.inline.inlining_heuristic" {
            return Err(TaskError {
                message: format!(
                    "all.py:83 inline_heuristic={heuristic_name:?} not ported \
                     (only rpython.translator.backendopt.inline.inlining_heuristic \
                     is wired)"
                ),
            });
        }
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
            inline::inlining_heuristic,
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
    // verbatim. The runtime piece is the blocker:
    // `translator.driver_instrument_result(filename)` (upstream
    // `driver.py`) compiles the instrumented graph through the C
    // backend, runs it, and reads back per-label counters. Pyre's
    // C-backend driver is not ported, so the path stays gated as a
    // `TaskError`. Convergence path: port the C-backend
    // instrument-and-run pipeline, then replace this gate with
    //     let counters = translator.driver_instrument_result(...)?;
    //     let pred: CallCountPred = Rc::new(RefCell::new(
    //         move |label| (label as usize) < counters.len()
    //             && counters[label as usize] > 250,
    //     ));
    //     inline::instrument_inline_candidates(&graphs, threshold, &translator);
    //     inline_malloc_removal_phase(&config, &translator, &graphs,
    //                                  threshold, inline_heuristic,
    //                                  Some(pred), inline_graph_from_anywhere)?;
    if stropt(&config, "profile_based_inline")?.is_some() && !secondary {
        return Err(TaskError {
            message: "all.py:99-113 profile_based_inline: static pieces ported \
                      (instrument_inline_candidates / CallCountPred / \
                      inline_malloc_removal_phase signature); blocked on \
                      C-backend driver_instrument_result runtime"
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
            message: "all.py:160 remove_mallocs: needs malloc.py port \
                      (566 LOC self-contained escape-analysis pass; \
                      UnionFind / simplify / removenoops deps already \
                      landed locally — body itself unported)"
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, SpaceOperation,
        Variable,
    };
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
        // `inline_heuristic` is structurally a dotted name resolved
        // by upstream's `get_function`. Pyre does not have a
        // Python-style import resolver, so any value other than the
        // single shipped default should surface a `TaskError`
        // instead of being silently mapped — matching the
        // closed-world parity contract documented at the call site.
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
                e.message.contains("inline_heuristic"),
                "expected inline_heuristic TaskError, got {:?}",
                e.message
            ),
            Ok(()) => panic!("expected TaskError on non-default inline_heuristic"),
        }
    }
}
