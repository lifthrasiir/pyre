//! Port of `rpython/translator/backendopt/inline.py`.
//!
//! Slice 1 lands the read-only foundation: the call-graph
//! collectors / matchers / RaiseAnalyzer-driven predicates that
//! upstream uses to drive `BaseInliner` (the heavy graph-rewriting
//! machinery is unported — see [`super::all`] for the gating
//! `inline` / `mallocs` `TaskError`).
//!
//! Ported in this file:
//!
//! * `CannotInline` (`:14-15`) — surfaced as [`CannotInline`].
//! * `CanRaise` (`:18-20`) — surfaced as [`CanRaiseHint`].
//! * `collect_called_graphs(graph, translator)` (`:23-40`).
//! * `iter_callsites(graph, calling_what)` (`:42-55`),
//!   `find_callsites` (`:57-58`), `iter_first_callsites`
//!   (`:60-65`), `contains_call` (`:67-73`).
//! * `does_raise_directly(graph, raise_analyzer)` (`:109-122`).
//! * `any_call_to_raising_graphs(from_graph, translator,
//!   raise_analyzer)` (`:124-142`).
//!
//! Slice 2 lands the auto-inlining cost helpers and the
//! call-graph collector consumed by `auto_inlining`:
//!
//! * `OP_WEIGHTS` (`:470-476`) → `op_weight()` helper.
//! * `block_weight(block)` (`:478-488`).
//! * `static_instruction_count(graph)` (`:532-536`).
//! * `always_inline(graph)` (`:604-606`) — PRE-EXISTING-ADAPTATION,
//!   pyre's `GraphFunc` lacks the `_always_inline_` slot.
//! * `inlinable_static_callers` (`:546-567`).
//!
//! Slice 3 lands the median-execution-cost solver and the actual
//! `inlining_heuristic` consumer:
//!
//! * `measure_median_execution_cost(graph)` (`:491-530`) — uses
//!   [`crate::tool::algo::sparsemat::SparseMatrix`] and
//!   [`super::support::find_loop_blocks`].
//! * `inlining_heuristic(graph)` (`:538-544`).
//!
//! Slice 4 ports the actual inliner machinery in three sub-slices:
//!
//! * Slice 4a: `BaseInliner` foundation — fields, ctor, simple
//!   helpers (`get_graph_from_op` / `get_new_name` / `passon_vars` /
//!   `copy_operation` / `copy_link` / `copy_block` /
//!   `search_for_calls` / `find_args_in_exceptional_case`).
//! * Slice 4b: orchestrator and rewire mutators — `inline_all` /
//!   `inline_once` / `do_inline` / `rewire_returnblock` /
//!   `rewire_exceptblock` / `rewire_exceptblock_no_guard` /
//!   `cleanup`. Exception-guarded calls return `CannotInline`
//!   pending Slice 4d.
//! * Slice 4c: public entry points — `BaseInliner::new_inliner`
//!   (`:439-459`), free-function [`inline_function`] (`:75-80`),
//!   free-function [`simple_inline_function`] (`:82-86`, wraps
//!   [`inline_function`] with the upstream defaults; threads `()`
//!   in place of `lltype_to_classdef_mapping()` since that mapping
//!   is consumed only by Slice 4d's
//!   `rewire_exceptblock_with_guard`).
//!
//! Deferred to subsequent slices (anchored in upstream order):
//!
//! * Slice 4d: `_find_exception_type` (`:89-107`),
//!   `rewire_exceptblock_with_guard` (`:326-353`),
//!   `generic_exception_matching` (`:355-390`) — blocked on
//!   `lltype.normalizeptr` + `RPythonTyper.lltype_to_classdef_mapping()`
//!   ports.
//! * Slice 5: `instrument_inline_candidates` (`:569-602`) /
//!   `auto_inlining` / `auto_inline_graphs` (`:608-731`) — depend on
//!   the heap-based driver loop.

use std::cell::RefCell;
use std::rc::Rc;

use crate::flowspace::model::{ConstValue, GraphKey, GraphRef, Hlvalue, SpaceOperation};
use crate::translator::backendopt::canraise::RaiseAnalyzer;
use crate::translator::simplify::get_graph_for_call;
use crate::translator::translator::TranslationContext;

/// Carrier for upstream's `call_count_pred` closure
/// (`inline.py:138 BaseInliner.__init__`,
/// `inline.py:608 auto_inlining`). Upstream Python passes the same
/// callable instance to every nested `inline_function` call;
/// `Rc<RefCell<dyn FnMut>>` is the minimal Rust adaptation that
/// preserves shared mutable identity (a `Box<dyn FnMut>` is not
/// `Clone`, so cloning into per-parent `BaseInliner`s would force
/// `None`).
pub type CallCountPred = Rc<RefCell<dyn FnMut(i64) -> bool>>;

/// `class CannotInline(Exception)` at `inline.py:14-15`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CannotInline(pub String);

impl CannotInline {
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

/// `class CanRaise(object): def __init__(self, can_raise)` at
/// `inline.py:18-20`. Upstream stores a single `bool` and uses the
/// instance as a sentinel returned by `collect_called_graphs` when
/// the callee is opaque (e.g. `op.args[0]` of a `direct_call` whose
/// `_obj` lookup failed).
///
/// pyre's `collect_called_graphs` does not synthesize `CanRaise`
/// instances yet (the only producers live in deeper backend passes
/// that are still unported); the type is surfaced here so
/// `any_call_to_raising_graphs` can match it line by line per
/// upstream `:129-130`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanRaiseHint {
    pub can_raise: bool,
}

impl CanRaiseHint {
    pub fn new(can_raise: bool) -> Self {
        Self { can_raise }
    }
}

/// One-of for `collect_called_graphs`'s set of "graphs or
/// somethings". Upstream's set holds either a `FunctionGraph`, a
/// raw `op.args[0]` Constant, or a `CanRaise` hint. The Rust port
/// distinguishes the three through this enum so
/// `any_call_to_raising_graphs` (`:124-133`) can route each variant
/// the same way upstream does.
#[derive(Clone, Debug)]
pub enum CalledThing {
    /// `op.args[0].value._obj.graph` — resolved graph.
    Graph(GraphRef),
    /// `op.args[0]` raw arg. Upstream pushes the LinkArg-shaped
    /// `Constant` into the set when `get_graph` returns `None`
    /// (`:30-32`); pyre carries the matching [`Hlvalue`].
    OpaqueArg(Hlvalue),
    /// `CanRaise` synthesised hint. Pyre never produces this today;
    /// keep the variant aligned with upstream `:129-130`.
    CanRaise(CanRaiseHint),
}

impl PartialEq for CalledThing {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (CalledThing::Graph(a), CalledThing::Graph(b)) => Rc::ptr_eq(a, b),
            (CalledThing::OpaqueArg(a), CalledThing::OpaqueArg(b)) => a == b,
            (CalledThing::CanRaise(a), CalledThing::CanRaise(b)) => a == b,
            _ => false,
        }
    }
}

/// Predicate matcher passed to `iter_callsites(graph, calling_what)`
/// at `inline.py:42-55`. Upstream accepts:
///
/// * `None` — match every direct_call.
/// * a `FunctionGraph` — match calls whose resolved callee
///   `funcobj.graph is calling_what`.
/// * a Python callable — match calls whose `funcobj._callable is
///   calling_what`.
///
/// `_callable` is `Option<String>` in pyre's `_func` carrier
/// (`rtyper/lltypesystem/lltype.rs:689`); the port matches against
/// the upstream Python callable identity by storing the qualname
/// string and comparing for equality. Production callers today only
/// pass `Any` or `Graph(_)`; the `CallableName` arm is provided so
/// the surface matches upstream verbatim for the day downstream
/// passes start consuming it.
pub enum CalleeMatcher<'a> {
    Any,
    Graph(&'a GraphRef),
    CallableName(&'a str),
}

/// Resolved callsite triple `(graph, block, op_index)` produced by
/// upstream `iter_callsites` at `:42-55`.
#[derive(Clone)]
pub struct CallSite {
    /// `funcobj.graph` resolved through `get_graph_for_call`. May be
    /// `None` per upstream `:50 graph = getattr(funcobj, 'graph',
    /// None)` — call sites whose callee is an external helper still
    /// yield, but with no graph reference attached.
    pub graph: Option<GraphRef>,
    pub block: crate::flowspace::model::BlockRef,
    pub op_index: usize,
}

/// `collect_called_graphs(graph, translator)` at `inline.py:23-40`.
///
/// Walks every block / op of `graph`, accumulating each callee
/// (graph, opaque arg, or CanRaise hint) into a set. The Rust port
/// uses `Vec<CalledThing>` with explicit dedup because pyre carries
/// no `__hash__` lattice on `Hlvalue::Constant`; the upstream `set`
/// behaviour is preserved through the `PartialEq` impl above.
pub fn collect_called_graphs(
    graph: &GraphRef,
    translator: &TranslationContext,
) -> Vec<CalledThing> {
    let mut out: Vec<CalledThing> = Vec::new();
    let push_unique = |out: &mut Vec<CalledThing>, item: CalledThing| {
        if !out.iter().any(|existing| *existing == item) {
            out.push(item);
        }
    };

    let blocks = graph.borrow().iterblocks();
    for block in &blocks {
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        for op in &ops {
            // Upstream `:27-32 if op.opname == "direct_call":`.
            if op.opname == "direct_call" {
                if let Some(arg0) = op.args.first() {
                    match get_graph_for_call(arg0, translator) {
                        Some(callee) => push_unique(&mut out, CalledThing::Graph(callee)),
                        None => push_unique(&mut out, CalledThing::OpaqueArg(arg0.clone())),
                    }
                }
            }
            // Upstream `:33-39 if op.opname == "indirect_call":`.
            if op.opname == "indirect_call" {
                let last = op.args.last();
                match last {
                    Some(Hlvalue::Constant(c)) => match &c.value {
                        ConstValue::Graphs(keys) => {
                            // Resolve every key to a graph through
                            // the translator — upstream walks the
                            // tuple of graphs verbatim.
                            let trans_graphs = translator.graphs.borrow();
                            for k in keys {
                                if let Some(g) = trans_graphs
                                    .iter()
                                    .find(|g| GraphKey::of(g).as_usize() == *k)
                                {
                                    push_unique(&mut out, CalledThing::Graph(g.clone()));
                                }
                            }
                        }
                        // Upstream `:35-36 if graphs is None:` — the
                        // candidate-graph list was elided. Push the
                        // raw `op.args[0]` opaque arg.
                        _ => {
                            if let Some(arg0) = op.args.first() {
                                push_unique(&mut out, CalledThing::OpaqueArg(arg0.clone()));
                            }
                        }
                    },
                    _ => {
                        if let Some(arg0) = op.args.first() {
                            push_unique(&mut out, CalledThing::OpaqueArg(arg0.clone()));
                        }
                    }
                }
            }
        }
    }
    out
}

/// `iter_callsites(graph, calling_what)` at `inline.py:42-55`.
///
/// Returns the list of `(callee_graph, block, op_index)` triples
/// for every `direct_call` op whose resolved `funcobj` matches
/// `calling_what` per [`CalleeMatcher`].
///
/// The Rust port returns a `Vec` rather than a generator — pyre's
/// upstream call sites either materialise via `find_callsites`
/// (`:57-58`) or stop after the first hit via
/// `iter_first_callsites` / `contains_call`. [`contains_call`]
/// short-circuits at the first match.
pub fn iter_callsites(
    graph: &GraphRef,
    calling_what: CalleeMatcher<'_>,
    translator: &TranslationContext,
) -> Vec<CallSite> {
    let mut out: Vec<CallSite> = Vec::new();
    let blocks = graph.borrow().iterblocks();
    for block in &blocks {
        let n_ops = block.borrow().operations.len();
        for i in 0..n_ops {
            let op = block.borrow().operations[i].clone();
            // Upstream `:45-48 if op.opname == "direct_call": funcobj
            // = op.args[0].value._obj else: continue`.
            if op.opname != "direct_call" {
                continue;
            }
            // Upstream `:50 graph = getattr(funcobj, 'graph', None)`.
            let callee_graph = op
                .args
                .first()
                .and_then(|arg| get_graph_for_call(arg, translator));
            // Upstream `:52-54`: match against the callable identity.
            let matched = match &calling_what {
                CalleeMatcher::Any => true,
                CalleeMatcher::Graph(target) => match &callee_graph {
                    Some(cg) => Rc::ptr_eq(cg, target),
                    None => false,
                },
                CalleeMatcher::CallableName(name) => callable_name_matches(&op, name),
            };
            if matched {
                out.push(CallSite {
                    graph: callee_graph,
                    block: block.clone(),
                    op_index: i,
                });
            }
        }
    }
    out
}

/// `find_callsites(graph, calling_what)` at `inline.py:57-58`.
///
/// Upstream is `list(iter_callsites(...))`; the Rust port already
/// materialises a `Vec` from [`iter_callsites`], so this is a thin
/// wrapper provided for surface parity.
pub fn find_callsites(
    graph: &GraphRef,
    calling_what: CalleeMatcher<'_>,
    translator: &TranslationContext,
) -> Vec<CallSite> {
    iter_callsites(graph, calling_what, translator)
}

/// `contains_call(graph, calling_what)` at `inline.py:67-73`.
///
/// Upstream wraps `iter_callsites` in a `try/except StopIteration`
/// to short-circuit on the first match. The Rust port walks until
/// the first hit and returns immediately so large graphs do not
/// materialise an unused Vec.
pub fn contains_call(
    graph: &GraphRef,
    calling_what: CalleeMatcher<'_>,
    translator: &TranslationContext,
) -> bool {
    let blocks = graph.borrow().iterblocks();
    for block in &blocks {
        let n_ops = block.borrow().operations.len();
        for i in 0..n_ops {
            let op = block.borrow().operations[i].clone();
            if op.opname != "direct_call" {
                continue;
            }
            let callee_graph = op
                .args
                .first()
                .and_then(|arg| get_graph_for_call(arg, translator));
            let matched = match &calling_what {
                CalleeMatcher::Any => true,
                CalleeMatcher::Graph(target) => match &callee_graph {
                    Some(cg) => Rc::ptr_eq(cg, target),
                    None => false,
                },
                CalleeMatcher::CallableName(name) => callable_name_matches(&op, name),
            };
            if matched {
                return true;
            }
        }
    }
    false
}

/// `getattr(funcobj, '_callable', None) is calling_what` at
/// `inline.py:54`. Pyre's `_callable` slot is `Option<String>`; the
/// upstream identity check on Python objects becomes string
/// equality on the qualname.
fn callable_name_matches(op: &SpaceOperation, name: &str) -> bool {
    let Some(Hlvalue::Constant(c)) = op.args.first() else {
        return false;
    };
    let ConstValue::LLPtr(p) = &c.value else {
        return false;
    };
    use crate::translator::rtyper::lltypesystem::lltype::_ptr_obj;
    let Ok(_ptr_obj::Func(funcobj)) = p._obj() else {
        return false;
    };
    funcobj._callable.as_deref() == Some(name)
}

/// `does_raise_directly(graph, raise_analyzer)` at
/// `inline.py:109-122`.
///
/// > this function checks, whether graph contains operations which
/// > can raise and which are not exception guarded
pub fn does_raise_directly(graph: &GraphRef, raise_analyzer: &mut RaiseAnalyzer<'_>) -> bool {
    let blocks = graph.borrow().iterblocks();
    let exceptblock = graph.borrow().exceptblock.clone();
    for block in &blocks {
        // Upstream `:113-114 if block is graph.exceptblock: return
        // True`.
        if Rc::ptr_eq(block, &exceptblock) {
            return true;
        }
        // Upstream `:115-118`:
        //     if block.canraise:
        //         consider_ops_to = -1
        //     else:
        //         consider_ops_to = len(block.operations)
        let canraise = block.borrow().canraise();
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        let consider_ops_to = if canraise {
            ops.len().saturating_sub(1)
        } else {
            ops.len()
        };
        for op in ops.iter().take(consider_ops_to) {
            if raise_analyzer.can_raise(op, None) {
                return true;
            }
        }
    }
    false
}

/// `any_call_to_raising_graphs(from_graph, translator,
/// raise_analyzer)` at `inline.py:124-142`.
pub fn any_call_to_raising_graphs(
    from_graph: &GraphRef,
    translator: &TranslationContext,
    raise_analyzer: &mut RaiseAnalyzer<'_>,
) -> bool {
    // Upstream `:125-133`: walk every callee.
    for thing in collect_called_graphs(from_graph, translator) {
        match thing {
            CalledThing::Graph(g) => {
                if does_raise_directly(&g, raise_analyzer) {
                    return true;
                }
            }
            CalledThing::CanRaise(hint) => {
                if hint.can_raise {
                    return true;
                }
            }
            // Upstream `:132-133 else: return True # conservatively`.
            CalledThing::OpaqueArg(_) => {
                return true;
            }
        }
    }
    // Upstream `:134-141`: also walk `from_graph`'s own ops.
    let blocks = from_graph.borrow().iterblocks();
    for block in &blocks {
        let canraise = block.borrow().canraise();
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        let consider_ops_to = if canraise {
            ops.len().saturating_sub(1)
        } else {
            ops.len()
        };
        for op in ops.iter().take(consider_ops_to) {
            if raise_analyzer.can_raise(op, None) {
                return true;
            }
        }
    }
    false
}

// ============================================================
// Auto-inlining heuristics — pure read-only graph cost model.
// Slice 2 of `inline.py` covers `OP_WEIGHTS` / `block_weight` /
// `static_instruction_count` / `inlinable_static_callers` /
// `always_inline`. The `_dont_inline_` / `_always_inline_` flag
// reads default to upstream's `getattr(..., default)` fallback —
// pyre's `GraphFunc` does not yet surface either flag.
// ============================================================

/// `OP_WEIGHTS` at `inline.py:470-476`. Per-opname weight table
/// consumed by `block_weight`. Opnames absent from the map default
/// to `1` per upstream `weights.get(op.opname, 1)` at `:485`.
///
/// The Rust port returns the upstream default through
/// [`op_weight`] rather than exposing the dictionary directly,
/// because Rust does not have a literal-keyed dict-with-default
/// shape; the helper's body mirrors upstream's six entries
/// verbatim.
fn op_weight(opname: &str) -> i64 {
    // Upstream `:470-476`:
    //     OP_WEIGHTS = {'same_as': 0,
    //                   'cast_pointer': 0,
    //                   'malloc': 2,
    //                   'instrument_count': 0,
    //                   'debug_assert': -1,
    //                   'jit_force_virtualizable': 0,
    //                   }
    match opname {
        "same_as" => 0,
        "cast_pointer" => 0,
        "malloc" => 2,
        "instrument_count" => 0,
        "debug_assert" => -1,
        "jit_force_virtualizable" => 0,
        // Upstream `:485 weights.get(op.opname, 1)`.
        _ => 1,
    }
}

/// `block_weight(block, weights=OP_WEIGHTS)` at
/// `inline.py:478-488`.
///
/// ```python
/// def block_weight(block, weights=OP_WEIGHTS):
///     total = 0
///     for op in block.operations:
///         if op.opname == "direct_call":
///             total += 1.5 + len(op.args) / 2
///         elif op.opname == "indirect_call":
///             total += 2 + len(op.args) / 2
///         total += weights.get(op.opname, 1)
///     if block.exitswitch is not None:
///         total += 1
///     return max(0, total)
/// ```
///
/// The Rust port returns `f64`. Upstream's `total` floats once a
/// `direct_call` adds the `1.5` constant; pyre tracks the result
/// as `f64` end-to-end so the integer-only path (no calls in the
/// block) remains representable exactly.
///
/// Upstream Python 2's `len(op.args) / 2` is integer division
/// because both operands are `int`. The Rust port preserves the
/// integer-divide semantics by computing `len() / 2` on `usize`
/// before promoting to `f64`.
pub fn block_weight(block: &crate::flowspace::model::BlockRef) -> f64 {
    let b = block.borrow();
    let mut total: f64 = 0.0;
    for op in &b.operations {
        // Upstream `:481-484`:
        //     if op.opname == "direct_call":
        //         total += 1.5 + len(op.args) / 2
        //     elif op.opname == "indirect_call":
        //         total += 2 + len(op.args) / 2
        let half_args = (op.args.len() / 2) as f64;
        if op.opname == "direct_call" {
            total += 1.5 + half_args;
        } else if op.opname == "indirect_call" {
            total += 2.0 + half_args;
        }
        // Upstream `:485 total += weights.get(op.opname, 1)`.
        total += op_weight(op.opname.as_str()) as f64;
    }
    // Upstream `:486-487 if block.exitswitch is not None: total += 1`.
    if b.exitswitch.is_some() {
        total += 1.0;
    }
    // Upstream `:488 return max(0, total)`.
    if total < 0.0 { 0.0 } else { total }
}

/// `static_instruction_count(graph)` at `inline.py:532-536`.
///
/// ```python
/// def static_instruction_count(graph):
///     count = 0
///     for block in graph.iterblocks():
///         count += block_weight(block)
///     return count
/// ```
pub fn static_instruction_count(graph: &GraphRef) -> f64 {
    let blocks = graph.borrow().iterblocks();
    let mut count: f64 = 0.0;
    for block in &blocks {
        count += block_weight(block);
    }
    count
}

/// `always_inline(graph)` at `inline.py:604-606`.
///
/// ```python
/// def always_inline(graph):
///     return (hasattr(graph, 'func') and
///             getattr(graph.func, '_always_inline_', None))
/// ```
///
/// Reads `graph.func._always_inline_` (added at
/// `model.rs:3185` alongside `_no_release_gil_` and the other
/// per-function backendopt flags). `false` mirrors upstream's
/// `getattr(..., None)` for a function that has no `func`
/// attribute or no `_always_inline_` attribute.
pub fn always_inline(graph: &GraphRef) -> bool {
    let g = graph.borrow();
    g.func.as_ref().is_some_and(|f| f._always_inline_)
}

/// Captures the `(parent, callee)` /
/// `(parent, block, op, callee)` tuple shapes returned by
/// `inlinable_static_callers` at `:546-567`. Upstream switches the
/// shape on the `store_calls` flag; the Rust port models the same
/// dichotomy through this enum so callers can pattern-match.
#[derive(Clone)]
pub enum InlinableCallerEntry {
    /// Upstream `store_calls=False` arm — emits `(parent, callee)`.
    GraphPair { parent: GraphRef, callee: GraphRef },
    /// Upstream `store_calls=True` arm — emits `(parent, block, op,
    /// callee)`.
    OpSite {
        parent: GraphRef,
        block: crate::flowspace::model::BlockRef,
        op: SpaceOperation,
        callee: GraphRef,
    },
}

/// `inlinable_static_callers(graphs, store_calls=False,
/// ok_to_call=None)` at `inline.py:546-567`.
///
/// ```python
/// def inlinable_static_callers(graphs, store_calls=False, ok_to_call=None):
///     if ok_to_call is None:
///         ok_to_call = set(graphs)
///     result = []
///     def add(parentgraph, block, op, graph):
///         if store_calls:
///             result.append((parentgraph, block, op, graph))
///         else:
///             result.append((parentgraph, graph))
///     for parentgraph in graphs:
///         for block in parentgraph.iterblocks():
///             for op in block.operations:
///                 if op.opname == "direct_call":
///                     funcobj = op.args[0].value._obj
///                     graph = getattr(funcobj, 'graph', None)
///                     if graph is not None and graph in ok_to_call:
///                         if getattr(getattr(funcobj, '_callable', None),
///                                    '_dont_inline_', False):
///                             continue
///                         add(parentgraph, block, op, graph)
///     return result
/// ```
///
/// `ok_to_call=None` upstream defaults to "every graph in
/// `graphs`". Pyre's `Option<&[GraphRef]>` mirrors this — `None`
/// uses the input list as the candidate set.
///
/// `_dont_inline_` is a Python attribute on the Python callable.
/// Pyre's `_callable` slot is `Option<String>`; the upstream
/// `getattr` with default `False` reduces to "always False" today.
/// PRE-EXISTING-ADAPTATION documented at [`callable_dont_inline`].
pub fn inlinable_static_callers(
    graphs: &[GraphRef],
    store_calls: bool,
    ok_to_call: Option<&[GraphRef]>,
    translator: &TranslationContext,
) -> Vec<InlinableCallerEntry> {
    // Upstream `:547-548 if ok_to_call is None: ok_to_call =
    // set(graphs)`. Reuse the input list when absent.
    let ok_to_call_owned: Vec<GraphRef>;
    let ok_set: &[GraphRef] = match ok_to_call {
        Some(s) => s,
        None => {
            ok_to_call_owned = graphs.to_vec();
            &ok_to_call_owned
        }
    };
    let mut out: Vec<InlinableCallerEntry> = Vec::new();
    for parent in graphs {
        let blocks = parent.borrow().iterblocks();
        for block in &blocks {
            let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
            for op in &ops {
                if op.opname != "direct_call" {
                    continue;
                }
                // Upstream `:560-561 funcobj = op.args[0].value._obj;
                // graph = getattr(funcobj, 'graph', None)`.
                let Some(arg0) = op.args.first() else {
                    continue;
                };
                let Some(callee) = get_graph_for_call(arg0, translator) else {
                    continue;
                };
                // Upstream `:562 if graph is not None and graph in
                // ok_to_call`.
                if !ok_set.iter().any(|g| Rc::ptr_eq(g, &callee)) {
                    continue;
                }
                // Upstream `:563-564`:
                //     if getattr(getattr(funcobj, '_callable', None),
                //                '_dont_inline_', False):
                //         continue
                if callable_dont_inline(&callee) {
                    continue;
                }
                // Upstream `:566 add(parentgraph, block, op, graph)`.
                if store_calls {
                    out.push(InlinableCallerEntry::OpSite {
                        parent: parent.clone(),
                        block: block.clone(),
                        op: op.clone(),
                        callee: callee.clone(),
                    });
                } else {
                    out.push(InlinableCallerEntry::GraphPair {
                        parent: parent.clone(),
                        callee: callee.clone(),
                    });
                }
            }
        }
    }
    out
}

/// `getattr(getattr(funcobj, '_callable', None), '_dont_inline_',
/// False)` at `inline.py:563-564`.
///
/// Pyre routes the upstream `funcobj._callable` indirection through
/// the resolved `graph.func` carrier (`model.rs:3196 _dont_inline_`).
/// The `_callable` Python attribute is conceptually the same object
/// as the `GraphFunc` here — both wrap the user-level Python
/// function — so reading the flag off the GraphFunc matches upstream
/// without an intermediate string lookup.
fn callable_dont_inline(callee: &GraphRef) -> bool {
    let g = callee.borrow();
    g.func.as_ref().is_some_and(|f| f._dont_inline_)
}

// ============================================================
// Slice 3: median-execution-cost solver + inlining_heuristic.
// Depends on `tool::algo::sparsemat::SparseMatrix` and
// `super::support::find_loop_blocks` — both ported as foundation
// in this slice's commits.
// ============================================================

/// `measure_median_execution_cost(graph)` at `inline.py:491-530`.
///
/// Solves the linear system that estimates how many times each
/// block runs on average per call. Loop back-edges get a 0.7
/// staying-in-the-loop probability; non-loop exits split the
/// remaining 0.3 (or split 1.0 evenly when the block has no loop
/// edges). Upstream returns `sys.maxint` on a singular matrix —
/// the Rust port returns `i64::MAX as f64` so the upstream
/// downstream comparison `0.9999 * ... + count` saturates the
/// same way.
pub fn measure_median_execution_cost(graph: &GraphRef) -> f64 {
    use crate::flowspace::model::BlockKey;
    use crate::tool::algo::sparsemat::SparseMatrix;
    use crate::translator::backendopt::support::find_loop_blocks;

    // Upstream `:492-496`:
    //     blocks = []
    //     blockmap = {}
    //     for block in graph.iterblocks():
    //         blockmap[block] = len(blocks)
    //         blocks.append(block)
    let blocks: Vec<crate::flowspace::model::BlockRef> = graph.borrow().iterblocks();
    let mut blockmap: std::collections::HashMap<BlockKey, usize> =
        std::collections::HashMap::with_capacity(blocks.len());
    for (i, b) in blocks.iter().enumerate() {
        blockmap.insert(BlockKey::of(b), i);
    }
    // Upstream `:497`:
    //     loops = find_loop_blocks(graph)
    let loops = find_loop_blocks(&graph.borrow());
    // Upstream `:498-499`:
    //     M = sparsemat.SparseMatrix(len(blocks))
    //     vector = []
    let mut matrix = SparseMatrix::new(blocks.len());
    let mut vector: Vec<f64> = Vec::with_capacity(blocks.len());

    for (i, block) in blocks.iter().enumerate() {
        // Upstream `:501-502`:
        //     vector.append(block_weight(block))
        //     M[i, i] = 1
        vector.push(block_weight(block));
        matrix.set(i, i, 1.0);
        // Upstream `:503 if block.exits:`.
        let exits = block.borrow().exits.clone();
        if exits.is_empty() {
            continue;
        }
        // Upstream `:504-507`:
        //     if block not in loops:
        //         current_loop_start = None
        //     else:
        //         current_loop_start = loops[block]
        let block_key = BlockKey::of(block);
        let current_loop_start: Option<crate::flowspace::model::BlockRef> =
            loops.get(&block_key).cloned();
        // Upstream `:508-512`:
        //     loop_exits = []
        //     for link in block.exits:
        //         if (link.target in loops and
        //             loops[link.target] is current_loop_start):
        //             loop_exits.append(link)
        // The `link.target in loops` short-circuit is load-bearing:
        // a target outside the loop dict can never be a loop_exit
        // even when the source block is also outside the loop dict.
        // Older code routed both-None through `same_loop_start`'s
        // `(None, None) => true` arm, which let non-loop blocks be
        // misclassified as same-loop and flipped the 0.3/0.7
        // weighting at `:513-517`.
        let mut loop_exits: Vec<crate::flowspace::model::LinkRef> = Vec::new();
        for link in &exits {
            if let Some(target) = link.borrow().target.clone() {
                let target_key = BlockKey::of(&target);
                if let Some(target_loop_start) = loops.get(&target_key) {
                    if same_loop_start(Some(target_loop_start), current_loop_start.as_ref()) {
                        loop_exits.push(link.clone());
                    }
                }
            }
        }
        // Upstream `:513-517`:
        //     if len(loop_exits) and len(loop_exits) < len(block.exits):
        //         f = 0.3 / (len(block.exits) - len(loop_exits))
        //         b = 0.7 / len(loop_exits)
        //     else:
        //         b = f = 1.0 / len(block.exits)
        let n_exits = exits.len();
        let n_loop_exits = loop_exits.len();
        let (b_weight, f_weight) = if n_loop_exits > 0 && n_loop_exits < n_exits {
            (
                0.7 / n_loop_exits as f64,
                0.3 / (n_exits - n_loop_exits) as f64,
            )
        } else {
            let v = 1.0 / n_exits as f64;
            (v, v)
        };
        // Upstream `:518-523`:
        //     for link in block.exits:
        //         if (link.target in loops and
        //             loops[link.target] is current_loop_start):
        //             M[i, blockmap[link.target]] -= b
        //         else:
        //             M[i, blockmap[link.target]] -= f
        // The `link.target in loops` short-circuit is the same one
        // exercised at `:510` above; without it, both-None at
        // `same_loop_start` flips a forward edge into a back edge
        // and the 0.7/0.3 weighting routes the wrong way.
        for link in &exits {
            let Some(target) = link.borrow().target.clone() else {
                continue;
            };
            let target_key = BlockKey::of(&target);
            let Some(&j) = blockmap.get(&target_key) else {
                continue;
            };
            let in_same_loop = match loops.get(&target_key) {
                Some(target_loop_start) => {
                    same_loop_start(Some(target_loop_start), current_loop_start.as_ref())
                }
                None => false,
            };
            let delta = if in_same_loop { -b_weight } else { -f_weight };
            matrix.add(i, j, delta);
        }
    }
    // Upstream `:524-530`:
    //     try:
    //         Solution = M.solve(vector)
    //     except ValueError:
    //         return sys.maxint
    //     else:
    //         res = Solution[blockmap[graph.startblock]]
    //         return max(res, 0.0)
    let solution = match matrix.solve(&vector) {
        Ok(s) => s,
        Err(_) => return i64::MAX as f64,
    };
    let start_idx = match blockmap.get(&BlockKey::of(&graph.borrow().startblock)) {
        Some(&i) => i,
        // Upstream's startblock is always in iterblocks; defensive
        // fallback returns sys.maxint to mirror an unsolvable shape.
        None => return i64::MAX as f64,
    };
    let res = solution[start_idx];
    if res > 0.0 { res } else { 0.0 }
}

/// Helper for upstream's `loops[link.target] is current_loop_start`
/// (`inline.py:511`, `:520`). Both sides are `Option<&BlockRef>`;
/// equality is identity-based per upstream Python `is` operator.
/// Both call sites first short-circuit on `link.target in loops`,
/// so this helper only runs when `target_loop_start` is `Some(...)`.
/// The `(None, None)` arm intentionally returns `false` — upstream
/// would never reach the `is` comparison without `link.target in
/// loops` being true.
fn same_loop_start(
    a: Option<&crate::flowspace::model::BlockRef>,
    b: Option<&crate::flowspace::model::BlockRef>,
) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => Rc::ptr_eq(a, b),
        _ => false,
    }
}

/// `inlining_heuristic(graph)` at `inline.py:538-544`.
///
/// ```python
/// def inlining_heuristic(graph):
///     # XXX ponderation factors?
///     count = static_instruction_count(graph)
///     if count >= 200:
///         return count, True
///     return (0.9999 * measure_median_execution_cost(graph) +
///             count), True       # may be NaN
/// ```
///
/// Returns `(weight, fixed)` matching upstream's tuple shape;
/// `fixed=true` in every arm.
pub fn inlining_heuristic(graph: &GraphRef) -> (f64, bool) {
    let count = static_instruction_count(graph);
    if count >= 200.0 {
        return (count, true);
    }
    let median = measure_median_execution_cost(graph);
    (0.9999 * median + count, true)
}

// ============================================================
// Slice 4a: BaseInliner foundation — fields, constructor, and
// the read-only / pure helpers (get_new_name, passon_vars,
// copy_operation, copy_link, copy_block, search_for_calls,
// find_args_in_exceptional_case). The `do_inline` orchestrator
// and `rewire_*` mutators land in Slice 4b; the public
// `Inliner` / `OneShotInliner` / `inline_function` entry points
// land in Slice 4c.
// ============================================================

/// Sub-inliner kind selector — controls
/// [`BaseInliner::search_for_calls`]'s behaviour.
///
/// Upstream models this through subclassing
/// (`Inliner`, `OneShotInliner` at `inline.py:439-463`); the Rust
/// port uses a kind enum so the structural state stays in one
/// place.
pub enum InlinerKind {
    /// `class Inliner(BaseInliner)` (`:439-459`). `inline_func` is
    /// the matched callee — every direct_call op whose resolved
    /// graph or callable name matches gets enqueued.
    Inliner(InlineFuncTarget),
    /// `class OneShotInliner(BaseInliner)` (`:461-463`).
    /// `search_for_calls` is overridden to no-op; the
    /// `block_to_index` map is populated externally before
    /// `inline_all` runs.
    OneShot,
}

/// Polymorphic identity for upstream's `inline_func` parameter.
/// Mirrors the `iter_callsites` matcher (`:42-55`) split between
/// "match by graph identity" and "match by Python callable".
#[derive(Clone)]
pub enum InlineFuncTarget {
    /// Upstream `iter_callsites`'s `graph is calling_what` arm.
    Graph(GraphRef),
    /// Upstream `iter_callsites`'s
    /// `getattr(funcobj, '_callable', None) is calling_what` arm.
    /// Pyre stores `_callable` as the qualname string.
    CallableName(String),
}

/// Cache key for upstream `BaseInliner._passon_vars` at
/// `inline.py:241-246`. The dict is keyed by either a `Block`
/// (the common case) or an `int` (fallback used inside
/// `generic_exception_matching` at `:362`).
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum PassonCacheKey {
    /// Upstream `_passon_vars[block]` keyed on block identity.
    Block(crate::flowspace::model::BlockKey),
    /// Upstream `_passon_vars[i]` keyed on the loop index.
    Index(usize),
}

/// `class BaseInliner` at `inline.py:144-436`.
///
/// Holds every cross-call piece of inliner state: the host
/// translator, the host graph being modified, the per-inline
/// varmap and copied-block cache, and the queue of (block,
/// op_index, callee) tuples to inline.
///
/// Slice 4a populates the struct surface and the read-only
/// helpers. Slice 4b adds `inline_once` / `do_inline` and the
/// rewire mutators; Slice 4c adds the public entry points.
pub struct BaseInliner<'t> {
    // ----- upstream `__init__` parameters -----
    /// Upstream `self.translator`.
    pub translator: &'t TranslationContext,
    /// Upstream `self.graph` — the host graph being mutated.
    pub graph: GraphRef,
    /// Upstream `self.do_cleanup` (`:153`). Toggles the trailing
    /// `cleanup_graph(self.graph)` call inside `inline_all`.
    pub do_cleanup: bool,
    /// Upstream `self.inline_guarded_calls`.
    pub inline_guarded_calls: bool,
    /// Upstream `self.inline_guarded_calls_no_matter_what`.
    pub inline_guarded_calls_no_matter_what: bool,
    /// Upstream `self.raise_analyzer`.
    pub raise_analyzer: crate::translator::backendopt::canraise::RaiseAnalyzer<'t>,
    /// Upstream `self.lltype_to_classdef`. Pyre's
    /// `RPythonTyper.lltype_to_classdef_mapping()` is unported;
    /// the slot is held as `()` so the public surface lines up
    /// for the day the dependency lands. Consumed only by
    /// `rewire_exceptblock_with_guard` (Slice 4d).
    pub lltype_to_classdef: (),
    /// Upstream `self.call_count_pred` — predicate consulted
    /// inside `inline_all` for the
    /// `instrument_count`-tagged path (`:176-182`).
    ///
    /// Carrier is `Rc<RefCell<dyn FnMut(i64) -> bool>>` so the
    /// predicate is shareable across the per-parent
    /// `inline_function` calls in [`auto_inlining`] — upstream
    /// passes the same Python closure to every `inline_function`
    /// invocation; pyre Arcshares the closure cell to mirror that.
    pub call_count_pred: Option<CallCountPred>,

    // ----- upstream `Inliner.__init__` extension -----
    /// Upstream `self.inline_func` and the kind discriminator.
    pub kind: InlinerKind,

    // ----- upstream "inline-all" queue -----
    /// Upstream `self.block_to_index`. Maps each pending block
    /// to a per-op-index callee dict.
    ///
    /// Pyre carries the `BlockRef` handle alongside the index
    /// dict because Python's dict iteration yields the live
    /// `Block` object directly (`for block, d in
    /// self.block_to_index.popitem()` at upstream `:168`); pyre's
    /// `BlockKey` is identity-only and can't recover the `Rc`
    /// without an external channel. Storing the handle here
    /// mirrors upstream's "key is the block itself" semantic with
    /// the smallest possible carrier change.
    pub block_to_index: std::collections::HashMap<
        crate::flowspace::model::BlockKey,
        (
            crate::flowspace::model::BlockRef,
            std::collections::HashMap<usize, GraphRef>,
        ),
    >,

    // ----- upstream "per-inline-call" state -----
    /// Upstream `self.varmap` (`:194`).
    pub varmap: std::collections::HashMap<
        crate::flowspace::model::Variable,
        crate::flowspace::model::Variable,
    >,
    /// Upstream `self._copied_blocks` (`:195`).
    pub copied_blocks: std::collections::HashMap<
        crate::flowspace::model::BlockKey,
        crate::flowspace::model::BlockRef,
    >,
    /// Upstream `self.op` — the call op being inlined (`:196`).
    pub op: Option<SpaceOperation>,
    /// Upstream `self.graph_to_inline` (`:197`).
    pub graph_to_inline: Option<GraphRef>,
    /// Upstream `self.exception_guarded` (`:198`).
    pub exception_guarded: bool,
    /// Upstream `self._passon_vars` (`:208`).
    pub passon_vars_cache:
        std::collections::HashMap<PassonCacheKey, Vec<crate::flowspace::model::Variable>>,
    /// Upstream `self.entrymap` (`:209`).
    pub entrymap: std::collections::HashMap<
        crate::flowspace::model::BlockKey,
        Vec<crate::flowspace::model::LinkRef>,
    >,
    /// Upstream `self.original_passon_vars` (`:400-401`).
    pub original_passon_vars: Vec<crate::flowspace::model::Variable>,
}

impl<'t> BaseInliner<'t> {
    /// `BaseInliner.__init__` at `inline.py:145-161`.
    ///
    /// Upstream's signature defaults: `inline_guarded_calls=False`,
    /// `inline_guarded_calls_no_matter_what=False`,
    /// `raise_analyzer=None` (asserted non-None at `:158`),
    /// `call_count_pred=None`, `cleanup=True`.
    pub fn new(
        translator: &'t TranslationContext,
        graph: GraphRef,
        kind: InlinerKind,
        raise_analyzer: crate::translator::backendopt::canraise::RaiseAnalyzer<'t>,
        inline_guarded_calls: bool,
        inline_guarded_calls_no_matter_what: bool,
        call_count_pred: Option<CallCountPred>,
        cleanup: bool,
    ) -> Self {
        Self {
            translator,
            graph,
            do_cleanup: cleanup,
            inline_guarded_calls,
            inline_guarded_calls_no_matter_what,
            raise_analyzer,
            lltype_to_classdef: (),
            call_count_pred,
            kind,
            block_to_index: std::collections::HashMap::new(),
            varmap: std::collections::HashMap::new(),
            copied_blocks: std::collections::HashMap::new(),
            op: None,
            graph_to_inline: None,
            exception_guarded: false,
            passon_vars_cache: std::collections::HashMap::new(),
            entrymap: std::collections::HashMap::new(),
            original_passon_vars: Vec::new(),
        }
    }

    /// `get_graph_from_op(self, op)` at `inline.py:189-191`.
    ///
    /// > assert op.opname == 'direct_call'
    /// > return self.op.args[0].value._obj.graph
    ///
    /// Upstream returns the graph attached to the function ptr in
    /// `op.args[0]`. The Rust port routes through
    /// `simplify::get_graph_for_call` for the same lookup.
    pub fn get_graph_from_op(&self, op: &SpaceOperation) -> Option<GraphRef> {
        assert_eq!(
            op.opname, "direct_call",
            "inline.py:190 get_graph_from_op: op.opname must be 'direct_call'",
        );
        op.args
            .first()
            .and_then(|arg| get_graph_for_call(arg, self.translator))
    }

    /// `get_new_name(self, var)` at `inline.py:232-239`.
    ///
    /// ```python
    /// def get_new_name(self, var):
    ///     if var is None:
    ///         return None
    ///     if isinstance(var, Constant):
    ///         return var
    ///     if var not in self.varmap:
    ///         self.varmap[var] = var.copy()
    ///     return self.varmap[var]
    /// ```
    ///
    /// Constants pass through unchanged. Variables get a fresh
    /// `Variable::copy()` cached in `self.varmap`.
    pub fn get_new_name(&mut self, arg: &Hlvalue) -> Hlvalue {
        match arg {
            Hlvalue::Constant(c) => Hlvalue::Constant(c.clone()),
            Hlvalue::Variable(v) => {
                if !self.varmap.contains_key(v) {
                    self.varmap.insert(v.clone(), v.copy());
                }
                Hlvalue::Variable(self.varmap.get(v).cloned().unwrap())
            }
        }
    }

    /// Optional-argument variant of `get_new_name` for the
    /// `link.last_exception` / `link.last_exc_value` slots that
    /// upstream feeds through the same helper at `:269-270`.
    pub fn get_new_name_optional(&mut self, arg: Option<&Hlvalue>) -> Option<Hlvalue> {
        arg.map(|a| self.get_new_name(a))
    }

    /// `passon_vars(self, cache_key)` at `inline.py:241-246`.
    ///
    /// ```python
    /// def passon_vars(self, cache_key):
    ///     if cache_key in self._passon_vars:
    ///         return self._passon_vars[cache_key]
    ///     result = [var.copy() for var in self.original_passon_vars]
    ///     self._passon_vars[cache_key] = result
    ///     return result
    /// ```
    pub fn passon_vars(
        &mut self,
        cache_key: PassonCacheKey,
    ) -> Vec<crate::flowspace::model::Variable> {
        if let Some(cached) = self.passon_vars_cache.get(&cache_key) {
            return cached.clone();
        }
        let fresh: Vec<crate::flowspace::model::Variable> =
            self.original_passon_vars.iter().map(|v| v.copy()).collect();
        self.passon_vars_cache.insert(cache_key, fresh.clone());
        fresh
    }

    /// `copy_operation(self, op)` at `inline.py:248-251`.
    ///
    /// ```python
    /// def copy_operation(self, op):
    ///     args = [self.get_new_name(arg) for arg in op.args]
    ///     result = SpaceOperation(op.opname, args, self.get_new_name(op.result))
    ///     return result
    /// ```
    pub fn copy_operation(&mut self, op: &SpaceOperation) -> SpaceOperation {
        let args: Vec<Hlvalue> = op.args.iter().map(|a| self.get_new_name(a)).collect();
        let result = self.get_new_name(&op.result);
        SpaceOperation::new(op.opname.clone(), args, result)
    }

    /// `copy_link(self, link, prevblock)` at `inline.py:266-273`.
    ///
    /// ```python
    /// def copy_link(self, link, prevblock):
    ///     newargs = [self.get_new_name(a) for a in link.args] + self.passon_vars(prevblock)
    ///     newlink = Link(newargs, self.copy_block(link.target), link.exitcase)
    ///     newlink.last_exception = self.get_new_name(link.last_exception)
    ///     newlink.last_exc_value = self.get_new_name(link.last_exc_value)
    ///     if hasattr(link, 'llexitcase'):
    ///         newlink.llexitcase = link.llexitcase
    ///     return newlink
    /// ```
    pub fn copy_link(
        &mut self,
        link: &crate::flowspace::model::LinkRef,
        prevblock: &crate::flowspace::model::BlockRef,
    ) -> crate::flowspace::model::LinkRef {
        use crate::flowspace::model::{BlockKey, Link};
        // Snapshot so we can drop the link's borrow before any
        // recursion below reaches back into the same Link.
        let (link_args, link_target, link_exitcase, link_llexitcase, link_last_exc, link_last_v) = {
            let l = link.borrow();
            (
                l.args.clone(),
                l.target.clone(),
                l.exitcase.clone(),
                l.llexitcase.clone(),
                l.last_exception.clone(),
                l.last_exc_value.clone(),
            )
        };
        // Upstream `:267 newargs = [self.get_new_name(a) for a in
        // link.args] + self.passon_vars(prevblock)`.
        let mut new_args: Vec<Hlvalue> = Vec::with_capacity(link_args.len());
        for arg in &link_args {
            // Upstream `link.args` is a list of args; pyre's
            // `LinkArg = Option<Hlvalue>` admits a `None` slot.
            // `get_new_name(None) -> None` per upstream `:233`,
            // and `Link.new_mergeable` accepts `None` link args.
            // Here we mirror upstream by routing every concrete
            // arg through `get_new_name`; the `None` path is
            // expressed by carrying the option through.
            match arg {
                Some(a) => new_args.push(self.get_new_name(a)),
                None => {
                    // Pyre's flowspace `Link.args` is Vec<LinkArg> =
                    // Vec<Option<Hlvalue>>; `get_new_name(None) ==
                    // None` upstream skips this slot. Drop into the
                    // `Hlvalue::Constant(None)` literal so the
                    // upstream "missing arg" position survives.
                    new_args.push(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                        crate::flowspace::model::ConstValue::None,
                    )));
                }
            }
        }
        let passon = self.passon_vars(PassonCacheKey::Block(BlockKey::of(prevblock)));
        for v in &passon {
            new_args.push(Hlvalue::Variable(v.clone()));
        }
        // Upstream `:268 newlink = Link(newargs,
        // self.copy_block(link.target), link.exitcase)`.
        let new_target = link_target
            .as_ref()
            .map(|t| self.copy_block(t))
            .expect("inline.py:268 copy_link expects link.target to be Some");
        let mut new_link = Link::new(new_args, Some(new_target), link_exitcase);
        // Upstream `:269 newlink.last_exception =
        // self.get_new_name(link.last_exception)`.
        new_link.last_exception = self.get_new_name_optional(link_last_exc.as_ref());
        // Upstream `:270 newlink.last_exc_value =
        // self.get_new_name(link.last_exc_value)`.
        new_link.last_exc_value = self.get_new_name_optional(link_last_v.as_ref());
        // Upstream `:271-272 if hasattr(link, 'llexitcase'):
        // newlink.llexitcase = link.llexitcase`.
        new_link.llexitcase = link_llexitcase;
        crate::flowspace::model::LinkRef::new(std::cell::RefCell::new(new_link))
    }

    /// `copy_block(self, block)` at `inline.py:253-264`.
    ///
    /// ```python
    /// def copy_block(self, block):
    ///     if block in self._copied_blocks:
    ///         return self._copied_blocks[block]
    ///     args = ([self.get_new_name(var) for var in block.inputargs] +
    ///             self.passon_vars(block))
    ///     newblock = Block(args)
    ///     self._copied_blocks[block] = newblock
    ///     newblock.operations = [self.copy_operation(op) for op in block.operations]
    ///     newblock.closeblock(*[self.copy_link(link, block) for link in block.exits])
    ///     newblock.exitswitch = self.get_new_name(block.exitswitch)
    ///     self.search_for_calls(newblock)
    ///     return newblock
    /// ```
    ///
    /// Two-phase construction mirrors upstream: insert the empty
    /// `newblock` in `_copied_blocks` BEFORE recursing, so cyclic
    /// refs (e.g. self-loops) terminate.
    pub fn copy_block(
        &mut self,
        block: &crate::flowspace::model::BlockRef,
    ) -> crate::flowspace::model::BlockRef {
        use crate::flowspace::model::{Block, BlockKey, BlockRefExt};
        let key = BlockKey::of(block);
        if let Some(existing) = self.copied_blocks.get(&key) {
            return existing.clone();
        }
        // Upstream `:256-258`:
        //     args = ([self.get_new_name(var) for var in block.inputargs]
        //             + self.passon_vars(block))
        let inputargs = block.borrow().inputargs.clone();
        let mut args: Vec<Hlvalue> = inputargs.iter().map(|v| self.get_new_name(v)).collect();
        let passon = self.passon_vars(PassonCacheKey::Block(key.clone()));
        for v in &passon {
            args.push(Hlvalue::Variable(v.clone()));
        }
        // Upstream `:258 newblock = Block(args)` then `:259
        // self._copied_blocks[block] = newblock`. Allocate first
        // so recursive copy_link calls can find the entry.
        let new_block = Block::shared(args);
        self.copied_blocks.insert(key.clone(), new_block.clone());
        // Upstream `:260 newblock.operations = [self.copy_operation(op)
        // for op in block.operations]`.
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        let new_ops: Vec<SpaceOperation> = ops.iter().map(|op| self.copy_operation(op)).collect();
        new_block.borrow_mut().operations = new_ops;
        // Upstream `:261 newblock.closeblock(...)`.
        let exits: Vec<crate::flowspace::model::LinkRef> =
            block.borrow().exits.iter().cloned().collect();
        let new_exits: Vec<crate::flowspace::model::LinkRef> = exits
            .iter()
            .map(|link| self.copy_link(link, block))
            .collect();
        new_block.closeblock(new_exits);
        // Upstream `:262 newblock.exitswitch =
        // self.get_new_name(block.exitswitch)`.
        let exitswitch = block.borrow().exitswitch.clone();
        new_block.borrow_mut().exitswitch = exitswitch.map(|sw| self.get_new_name(&sw));
        // Upstream `:263 self.search_for_calls(newblock)`.
        self.search_for_calls(&new_block);
        new_block
    }

    /// `search_for_calls(self, block)` at `inline.py:212-230` for
    /// the `Inliner` subclass and `:462-463` for the `OneShotInliner`
    /// no-op override.
    ///
    /// `Inliner.__init__` at `:439-459` calls
    /// `BaseInliner.search_for_calls` only via `copy_block` after the
    /// first round of callsites is enqueued. The Rust port routes
    /// the `OneShot` arm to early return; the `Inliner` arm walks
    /// the new block for direct_call ops whose callee matches
    /// `inline_func`.
    pub fn search_for_calls(&mut self, block: &crate::flowspace::model::BlockRef) {
        use crate::flowspace::model::BlockKey;
        let target = match &self.kind {
            InlinerKind::OneShot => return,
            InlinerKind::Inliner(t) => t.clone(),
        };
        // Upstream `:213 d = {}`.
        let mut d: std::collections::HashMap<usize, GraphRef> = std::collections::HashMap::new();
        let ops: Vec<SpaceOperation> = block.borrow().operations.clone();
        for (i, op) in ops.iter().enumerate() {
            // Upstream `:215-218 if op.opname == "direct_call":
            // funcobj = op.args[0].value._obj else: continue`.
            if op.opname != "direct_call" {
                continue;
            }
            // Upstream `:219 graph = getattr(funcobj, 'graph', None)`.
            let callee_graph = op
                .args
                .first()
                .and_then(|arg| get_graph_for_call(arg, self.translator));
            // Upstream `:221-222 if (graph is self.inline_func or
            // getattr(funcobj, '_callable', None) is self.inline_func):`.
            let matched = match &target {
                InlineFuncTarget::Graph(t) => match &callee_graph {
                    Some(cg) => Rc::ptr_eq(cg, t),
                    None => false,
                },
                InlineFuncTarget::CallableName(name) => callable_name_matches(op, name),
            };
            if matched {
                // Upstream `:223 d[i] = graph`.
                if let Some(g) = callee_graph {
                    d.insert(i, g);
                }
            }
        }
        // Upstream `:224-230`:
        //     if d:
        //         self.block_to_index[block] = d
        //     else:
        //         try:
        //             del self.block_to_index[block]
        //         except KeyError:
        //             pass
        if !d.is_empty() {
            self.block_to_index
                .insert(BlockKey::of(block), (block.clone(), d));
        } else {
            self.block_to_index.remove(&BlockKey::of(block));
        }
    }

    /// `find_args_in_exceptional_case(self, link, block, etype,
    /// evalue, afterblock, passon_vars)` at `inline.py:275-287`.
    ///
    /// ```python
    /// def find_args_in_exceptional_case(self, link, block, etype, evalue, afterblock, passon_vars):
    ///     linkargs = []
    ///     for arg in link.args:
    ///         if arg == link.last_exception:
    ///             linkargs.append(etype)
    ///         elif arg == link.last_exc_value:
    ///             linkargs.append(evalue)
    ///         elif isinstance(arg, Constant):
    ///             linkargs.append(arg)
    ///         else:
    ///             index = afterblock.inputargs.index(arg)
    ///             linkargs.append(passon_vars[index - 1])
    ///     return linkargs
    /// ```
    ///
    /// Note: pyre's `link.args` is `Vec<LinkArg> = Vec<Option<Hlvalue>>`.
    /// Upstream Python iterates the list with no `None` slots; pyre
    /// surfaces those as upstream's `Hlvalue::Constant(None)`
    /// shape per the comment in [`Self::copy_link`].
    pub fn find_args_in_exceptional_case(
        &self,
        link: &crate::flowspace::model::LinkRef,
        _block: &crate::flowspace::model::BlockRef,
        etype: &Hlvalue,
        evalue: &Hlvalue,
        afterblock: &crate::flowspace::model::BlockRef,
        passon_vars: &[crate::flowspace::model::Variable],
    ) -> Vec<Hlvalue> {
        let l = link.borrow();
        let last_exc = l.last_exception.clone();
        let last_v = l.last_exc_value.clone();
        let mut linkargs: Vec<Hlvalue> = Vec::with_capacity(l.args.len());
        for arg in &l.args {
            let Some(a) = arg.as_ref() else {
                // Defensive fallback: emit None constant for the
                // missing-slot path. Upstream never feeds a None
                // through this helper because the call site walks
                // `afterblock.exits[1:]` (exception edges, where
                // every arg is concrete).
                linkargs.push(Hlvalue::Constant(crate::flowspace::model::Constant::new(
                    crate::flowspace::model::ConstValue::None,
                )));
                continue;
            };
            if Some(a) == last_exc.as_ref() {
                linkargs.push(etype.clone());
            } else if Some(a) == last_v.as_ref() {
                linkargs.push(evalue.clone());
            } else if matches!(a, Hlvalue::Constant(_)) {
                linkargs.push(a.clone());
            } else {
                // Upstream `:285-286`:
                //     index = afterblock.inputargs.index(arg)
                //     linkargs.append(passon_vars[index - 1])
                let idx = afterblock
                    .borrow()
                    .inputargs
                    .iter()
                    .position(|ia| ia == a)
                    .expect("inline.py:285 afterblock.inputargs.index(arg)");
                linkargs.push(Hlvalue::Variable(
                    passon_vars
                        .get(idx.wrapping_sub(1))
                        .cloned()
                        .expect("inline.py:286 passon_vars[index-1]"),
                ));
            }
        }
        linkargs
    }

    // ----- Slice 4b: orchestrator + rewire mutators + drivers -----

    /// `inline_all(self)` at `inline.py:163-187`.
    ///
    /// Drains `block_to_index` one (block, op_index, callee) at a
    /// time and runs `inline_once`. Recursive callees are
    /// rejected up-front via `contains_call(subgraph, subgraph)`
    /// per upstream `:172-173`. Returns the number of successful
    /// inlines.
    pub fn inline_all(&mut self) -> Result<usize, CannotInline> {
        
        let mut count = 0usize;
        let mut non_recursive: std::collections::HashMap<usize, ()> =
            std::collections::HashMap::new();
        // Upstream `:167 while self.block_to_index:`.
        loop {
            // Upstream `:168 block, d = self.block_to_index.popitem()`.
            // Pyre's HashMap iteration order is unspecified; pop any
            // entry — order does not affect correctness because each
            // call site is processed exactly once.
            let block_key = match self.block_to_index.keys().next().cloned() {
                Some(k) => k,
                None => break,
            };
            let (block, mut d) = self
                .block_to_index
                .remove(&block_key)
                .expect("block_to_index.popitem(): key just observed");
            // Upstream `:169 index_operation, subgraph = d.popitem()`.
            let index_operation = match d.keys().next().copied() {
                Some(i) => i,
                None => continue,
            };
            let subgraph = d
                .remove(&index_operation)
                .expect("d.popitem(): key just observed");
            // Upstream `:170-171 if d: self.block_to_index[block] = d`.
            if !d.is_empty() {
                self.block_to_index
                    .insert(block_key.clone(), (block.clone(), d));
            }
            // Upstream `:172-175`:
            //     if subgraph not in non_recursive and contains_call(subgraph, subgraph):
            //         raise CannotInline("inlining a recursive function")
            //     else:
            //         non_recursive[subgraph] = True
            let subgraph_id = std::rc::Rc::as_ptr(&subgraph) as usize;
            if !non_recursive.contains_key(&subgraph_id)
                && contains_call(&subgraph, CalleeMatcher::Graph(&subgraph), self.translator)
            {
                return Err(CannotInline::new(
                    "inline.py:172 inlining a recursive function",
                ));
            }
            non_recursive.insert(subgraph_id, ());
            // Upstream `:176-182 call_count_pred` filter — only fires
            // when the inliner was constructed with one. Read the
            // instrument_count tag at index_operation-1.
            if let Some(pred) = self.call_count_pred.clone() {
                let count_op = block
                    .borrow()
                    .operations
                    .get(index_operation.wrapping_sub(1))
                    .cloned();
                if let Some(count_op) = count_op {
                    assert_eq!(
                        count_op.opname, "instrument_count",
                        "inline.py:178 callcount op must be instrument_count"
                    );
                    if let Some(Hlvalue::Constant(c0)) = count_op.args.first() {
                        assert!(
                            matches!(&c0.value, crate::flowspace::model::ConstValue::ByteStr(s)
                                if s == b"inline")
                                || matches!(&c0.value, crate::flowspace::model::ConstValue::UniStr(s)
                                    if s == "inline"),
                            "inline.py:179 instrument_count tag must be 'inline'"
                        );
                    }
                    if let Some(Hlvalue::Constant(c1)) = count_op.args.get(1) {
                        if let crate::flowspace::model::ConstValue::Int(label) = c1.value {
                            if !pred.borrow_mut()(label) {
                                continue;
                            }
                        }
                    }
                }
            }
            // Upstream `:183 self.inline_once(block, index_operation)`.
            self.inline_once(&block, index_operation)?;
            count += 1;
        }
        // Upstream `:185-186 if self.do_cleanup: self.cleanup()`.
        if self.do_cleanup {
            self.cleanup();
        }
        Ok(count)
    }

    /// `inline_once(self, block, index_operation)` at
    /// `inline.py:193-210`.
    ///
    /// Resets per-call state, captures the call op + callee graph,
    /// performs the exception-guarded check, and dispatches to
    /// `do_inline`. Returns CannotInline on:
    ///
    /// * upstream `:204 inline_guarded_calls AND
    ///   does_raise_directly` arm.
    /// * upstream `:205-207 not inline_guarded_calls AND
    ///   any_call_to_raising_graphs` arm.
    /// * Slice 4d-not-yet-ported guarded-inline path.
    pub fn inline_once(
        &mut self,
        block: &crate::flowspace::model::BlockRef,
        index_operation: usize,
    ) -> Result<(), CannotInline> {
        // Upstream `:194-197`:
        //     self.varmap = {}
        //     self._copied_blocks = {}
        //     self.op = block.operations[index_operation]
        //     self.graph_to_inline = self.get_graph_from_op(self.op)
        self.varmap.clear();
        self.copied_blocks.clear();
        let op = block
            .borrow()
            .operations
            .get(index_operation)
            .cloned()
            .expect("inline.py:196 block.operations[index_operation]");
        let graph_to_inline = self
            .get_graph_from_op(&op)
            .expect("inline.py:197 get_graph_from_op: callee graph required");
        self.op = Some(op.clone());
        self.graph_to_inline = Some(graph_to_inline.clone());
        self.exception_guarded = false;
        // Upstream `:199-207 if self.op is block.raising_op:` —
        // identity-compares the call op against the block's raising
        // op. Pyre's `Block::raising_op()` returns the trailing op
        // when `block.canraise()`. Compare by SpaceOperation
        // result-Variable id since SpaceOperation does not carry
        // a position-based identity.
        let is_raising = {
            let b = block.borrow();
            match (b.canraise(), b.operations.last()) {
                (true, Some(last)) => {
                    last as *const SpaceOperation
                        == &b.operations[index_operation] as *const SpaceOperation
                }
                _ => false,
            }
        };
        if is_raising {
            self.exception_guarded = true;
            // Upstream `:201-207`:
            //     if self.inline_guarded_calls:
            //         if (not self.inline_guarded_calls_no_matter_what and
            //             does_raise_directly(self.graph_to_inline, self.raise_analyzer)):
            //             raise CannotInline("can't inline because the call is exception guarded")
            //     elif any_call_to_raising_graphs(...):
            //         raise CannotInline("can't handle exceptions")
            //
            // Pass-through cases ultimately route through
            // `rewire_exceptblock` → `rewire_exceptblock_with_guard`,
            // which depends on:
            //   * `lltype.normalizeptr` (`lltype.py:1139-1161`) —
            //     pointer normalization for exception-class
            //     constants in `_find_exception_type`.
            //   * `RPythonTyper.lltype_to_classdef_mapping()` —
            //     dict from lltype Ptr to ClassDef, used to look up
            //     the matching exception handler.
            //   * `RPythonTyper.exceptiondata.fn_exception_match` —
            //     runtime helper that checks vtable identity.
            //   * `RPythonTyper.exceptiondata.lltype_of_exception_type`
            //     — type alias used to extract the vtable from the
            //     class repr.
            //   * `RPythonTyper.exceptiondata.generate_exception_match`
            //     — code-generator that emits ll-ops for the
            //     `generic_exception_matching` fallback path.
            //   * `rclass.getclassrepr(rtyper, classdef)` — class
            //     repr resolver from the rtyper rclass module.
            //
            // None of these have local equivalents (rtyper is a
            // skeleton; the exceptiondata machinery is genuinely
            // unported). The unconditional rejection at this gate
            // is therefore unavoidable today, but the upstream
            // conditional shape is preserved so the CannotInline
            // message identifies which upstream branch would have
            // fired and a reviewer can verify the
            // would-have-passed-through cases are the only ones the
            // Slice 4d gap blocks.
            if self.inline_guarded_calls {
                if !self.inline_guarded_calls_no_matter_what
                    && does_raise_directly(&graph_to_inline, &mut self.raise_analyzer)
                {
                    return Err(CannotInline::new(
                        "can't inline because the call is exception guarded",
                    ));
                }
                return Err(CannotInline::new(
                    "inline.py:201-204 inline_guarded_calls pass-through requires \
                     rewire_exceptblock_with_guard (Slice 4d gap: \
                     lltype.normalizeptr + lltype_to_classdef_mapping unported).",
                ));
            } else if any_call_to_raising_graphs(
                &graph_to_inline,
                self.translator,
                &mut self.raise_analyzer,
            ) {
                return Err(CannotInline::new("can't handle exceptions"));
            } else {
                return Err(CannotInline::new(
                    "inline.py:205-207 non-raising callee under \
                     inline_guarded_calls=False would pass through to \
                     rewire_exceptblock_with_guard (Slice 4d gap: \
                     lltype.normalizeptr + lltype_to_classdef_mapping unported).",
                ));
            }
        }
        // Upstream `:208-209`:
        //     self._passon_vars = {}
        //     self.entrymap = mkentrymap(self.graph_to_inline)
        self.passon_vars_cache.clear();
        self.entrymap = crate::flowspace::model::mkentrymap(&graph_to_inline.borrow());
        // Upstream `:210 self.do_inline(block, index_operation)`.
        self.do_inline(block, index_operation);
        Ok(())
    }

    /// `do_inline(self, block, index_operation)` at
    /// `inline.py:392-430`.
    ///
    /// Splits the host block at `index_operation`, copies the
    /// callee's start block, threads call args through the
    /// synthetic link, prepends the call result to the
    /// post-call block's inputargs, and rewires the callee's
    /// returnblock / exceptblock back into the host graph.
    fn do_inline(&mut self, block: &crate::flowspace::model::BlockRef, index_operation: usize) {
        use crate::flowspace::model::BlockKey;
        // Upstream `:393 splitlink = split_block(block, index_operation)`.
        let splitlink = crate::translator::unsimplify::split_block(block, index_operation, None);
        let afterblock = splitlink
            .borrow()
            .target
            .clone()
            .expect("inline.py:394 split_block returns a link with target");
        // Upstream `:400-401`:
        //     self.original_passon_vars = [arg for arg in block.exits[0].args
        //                                      if isinstance(arg, Variable)]
        // After split_block, block has exactly one exit (splitlink).
        // The synthetic link's args are exactly the upstream
        // `block.exits[0].args` in the post-split state. Filter to
        // the Variable members.
        self.original_passon_vars = splitlink
            .borrow()
            .args
            .iter()
            .filter_map(|a| match a {
                Some(Hlvalue::Variable(v)) => Some(v.clone()),
                _ => None,
            })
            .collect();
        // Upstream `:402 assert afterblock.operations[0].opname ==
        // self.op.opname`.
        let op = self
            .op
            .as_ref()
            .expect("inline.py:196 self.op set in inline_once")
            .clone();
        let after_first = afterblock
            .borrow()
            .operations
            .first()
            .cloned()
            .expect("inline.py:402 afterblock has at least one op");
        assert_eq!(
            after_first.opname, op.opname,
            "inline.py:402 afterblock.operations[0].opname vs self.op.opname mismatch",
        );
        // Upstream `:403 self.op = afterblock.operations.pop(0)`.
        // The first op of afterblock is the call op; remove and
        // re-bind self.op to that instance (same opname/args/result
        // because split_block did not rewrite the op contents).
        let popped = afterblock.borrow_mut().operations.remove(0);
        self.op = Some(popped.clone());
        let op = popped;
        // Upstream `:405 linktoinlined = splitlink`.
        let linktoinlined = splitlink.clone();
        // Upstream `:406 copiedstartblock =
        // self.copy_block(self.graph_to_inline.startblock)`.
        let graph_to_inline = self
            .graph_to_inline
            .as_ref()
            .expect("inline.py:197 graph_to_inline set in inline_once")
            .clone();
        let inline_startblock = graph_to_inline.borrow().startblock.clone();
        let copiedstartblock = self.copy_block(&inline_startblock);
        // Upstream `:408-414`:
        //     passon_args = []
        //     for arg in self.op.args[1:]:
        //         if isinstance(arg, Constant):
        //             passon_args.append(arg)
        //         else:
        //             index = afterblock.inputargs.index(arg)
        //             passon_args.append(linktoinlined.args[index])
        //     passon_args += self.original_passon_vars
        let mut passon_args: Vec<Hlvalue> = Vec::with_capacity(op.args.len());
        for arg in op.args.iter().skip(1) {
            match arg {
                Hlvalue::Constant(_) => passon_args.push(arg.clone()),
                Hlvalue::Variable(_) => {
                    let idx = afterblock
                        .borrow()
                        .inputargs
                        .iter()
                        .position(|ia| ia == arg)
                        .expect("inline.py:413 afterblock.inputargs.index(arg)");
                    let linktoinlined_arg = linktoinlined
                        .borrow()
                        .args
                        .get(idx)
                        .cloned()
                        .expect("inline.py:414 linktoinlined.args[index]");
                    // The synthetic link arg is `Option<Hlvalue>`;
                    // upstream Python's link.args[index] never has
                    // a None hole at this position because the
                    // varmap-driven split_block only populates
                    // concrete entries.
                    passon_args.push(
                        linktoinlined_arg
                            .expect("inline.py:414 linktoinlined.args[index] not None"),
                    );
                }
            }
        }
        for v in &self.original_passon_vars {
            passon_args.push(Hlvalue::Variable(v.clone()));
        }
        // Upstream `:417-419`:
        //     linktoinlined.target = copiedstartblock
        //     linktoinlined.args = passon_args
        //     afterblock.inputargs = [self.op.result] + afterblock.inputargs
        {
            let mut l = linktoinlined.borrow_mut();
            l.target = Some(copiedstartblock.clone());
            l.args = passon_args.into_iter().map(Some).collect();
        }
        {
            let mut ab = afterblock.borrow_mut();
            let mut new_inputs: Vec<Hlvalue> = Vec::with_capacity(ab.inputargs.len() + 1);
            new_inputs.push(op.result.clone());
            new_inputs.extend(ab.inputargs.drain(..));
            ab.inputargs = new_inputs;
        }
        // Upstream `:421 if self.graph_to_inline.returnblock in
        // self.entrymap`.
        let returnblock = graph_to_inline.borrow().returnblock.clone();
        if self.entrymap.contains_key(&BlockKey::of(&returnblock)) {
            self.rewire_returnblock(&afterblock);
        }
        // Upstream `:423 if self.graph_to_inline.exceptblock in
        // self.entrymap`.
        let exceptblock = graph_to_inline.borrow().exceptblock.clone();
        if self.entrymap.contains_key(&BlockKey::of(&exceptblock)) {
            self.rewire_exceptblock(&afterblock);
        }
        // Upstream `:425-428 if self.exception_guarded:` —
        // unreachable in Slice 4b because `inline_once` already
        // bails on exception_guarded. Kept as a sanity assert.
        assert!(
            !self.exception_guarded,
            "inline.py:425 Slice 4b should never reach do_inline with \
             exception_guarded=true",
        );
        // Upstream `:429-430 self.search_for_calls(afterblock);
        // self.search_for_calls(block)`.
        self.search_for_calls(&afterblock);
        self.search_for_calls(block);
    }

    /// `rewire_returnblock(self, afterblock)` at `inline.py:289-296`.
    ///
    /// Wires the inlined returnblock's outgoing edge to
    /// `afterblock`: `linkfrominlined = Link([returnvar +
    /// passon_vars(returnblock)], afterblock)`. Upstream uses
    /// `recloseblock(linkfrominlined)` to discard the inlined
    /// returnblock's existing exits (which would be `()` in the
    /// callee anyway).
    fn rewire_returnblock(&mut self, afterblock: &crate::flowspace::model::BlockRef) {
        use crate::flowspace::model::{BlockKey, BlockRefExt, Link};
        let graph_to_inline = self
            .graph_to_inline
            .as_ref()
            .expect("inline.py:290 graph_to_inline")
            .clone();
        let returnblock = graph_to_inline.borrow().returnblock.clone();
        let copiedreturnblock = self.copy_block(&returnblock);
        // Upstream `:291-292`:
        //     linkargs = ([copiedreturnblock.inputargs[0]] +
        //                 self.passon_vars(self.graph_to_inline.returnblock))
        let mut linkargs: Vec<Hlvalue> = Vec::new();
        linkargs.push(
            copiedreturnblock
                .borrow()
                .inputargs
                .first()
                .cloned()
                .expect("inline.py:291 copiedreturnblock.inputargs[0]"),
        );
        let passon = self.passon_vars(PassonCacheKey::Block(BlockKey::of(&returnblock)));
        for v in &passon {
            linkargs.push(Hlvalue::Variable(v.clone()));
        }
        // Upstream `:293 linkfrominlined = Link(linkargs, afterblock)`.
        let linkfrominlined = Link::new(linkargs, Some(afterblock.clone()), None).into_ref();
        // Upstream `:294-295`:
        //     copiedreturnblock.exitswitch = None
        //     copiedreturnblock.recloseblock(linkfrominlined)
        copiedreturnblock.borrow_mut().exitswitch = None;
        copiedreturnblock.recloseblock(vec![linkfrominlined]);
        // Upstream `:296 assert copiedreturnblock.exits[0].target ==
        // afterblock`.
        let exit_target = copiedreturnblock.borrow().exits[0]
            .borrow()
            .target
            .clone()
            .expect("inline.py:296 copied returnblock has target");
        assert!(
            std::rc::Rc::ptr_eq(&exit_target, afterblock),
            "inline.py:296 copiedreturnblock.exits[0].target == afterblock",
        );
    }

    /// `rewire_exceptblock(self, afterblock)` at
    /// `inline.py:298-308`. Dispatcher between guarded and
    /// non-guarded variants.
    fn rewire_exceptblock(&mut self, afterblock: &crate::flowspace::model::BlockRef) {
        let graph_to_inline = self
            .graph_to_inline
            .as_ref()
            .expect("inline.py:300 graph_to_inline")
            .clone();
        let exceptblock = graph_to_inline.borrow().exceptblock.clone();
        // Upstream `:300 copiedexceptblock =
        // self.copy_block(self.graph_to_inline.exceptblock)`.
        let copiedexceptblock = self.copy_block(&exceptblock);
        if !self.exception_guarded {
            // Upstream `:302
            // self.rewire_exceptblock_no_guard(...)`.
            self.rewire_exceptblock_no_guard(afterblock, &copiedexceptblock);
        } else {
            // Slice 4d gap; gated by `inline_once` returning
            // CannotInline, so this branch is unreachable today.
            unreachable!(
                "inline.py:303-308 exception-guarded path requires \
                 rewire_exceptblock_with_guard + generic_exception_matching \
                 (Slice 4d). inline_once gates this case with CannotInline."
            );
        }
    }

    /// `rewire_exceptblock_no_guard(self, afterblock,
    /// copiedexceptblock)` at `inline.py:310-324`.
    ///
    /// For each entry-link of the inlined exceptblock, rewire the
    /// corresponding link in the *copied* graph to bypass
    /// `copiedexceptblock` and go directly to the host graph's
    /// exceptblock. Drops any extra link args beyond the first
    /// two (etype, evalue).
    fn rewire_exceptblock_no_guard(
        &mut self,
        _afterblock: &crate::flowspace::model::BlockRef,
        copiedexceptblock: &crate::flowspace::model::BlockRef,
    ) {
        let graph_to_inline = self
            .graph_to_inline
            .as_ref()
            .expect("inline.py:312 graph_to_inline")
            .clone();
        let exceptblock = graph_to_inline.borrow().exceptblock.clone();
        let host_exceptblock = self.graph.borrow().exceptblock.clone();
        // Upstream `:312 for link in
        // self.entrymap[self.graph_to_inline.exceptblock]:`. The
        // entrymap is keyed on the inlined graph's blocks; the
        // value is a list of incoming Links.
        let entry_links: Vec<crate::flowspace::model::LinkRef> = self
            .entrymap
            .get(&crate::flowspace::model::BlockKey::of(&exceptblock))
            .cloned()
            .unwrap_or_default();
        for link in &entry_links {
            // Upstream `:313 copiedblock =
            // self.copy_block(link.prevblock)`.
            let prevblock = match link.borrow().prevblock.as_ref().and_then(|w| w.upgrade()) {
                Some(p) => p,
                None => continue,
            };
            let copiedblock = self.copy_block(&prevblock);
            // Upstream `:314-324 for copiedlink in
            // copiedblock.exits:`.
            let copied_exits: Vec<crate::flowspace::model::LinkRef> =
                copiedblock.borrow().exits.iter().cloned().collect();
            for copiedlink in &copied_exits {
                let target = match copiedlink.borrow().target.clone() {
                    Some(t) => t,
                    None => continue,
                };
                if !std::rc::Rc::ptr_eq(&target, copiedexceptblock) {
                    continue;
                }
                // Upstream `:316-317`:
                //     copiedlink.args = copiedlink.args[:2]
                //     copiedlink.target = self.graph.exceptblock
                {
                    let mut cl = copiedlink.borrow_mut();
                    cl.args.truncate(2);
                    cl.target = Some(host_exceptblock.clone());
                }
                // Upstream `:318-324 propagate concretetype`:
                //     for a1, a2 in zip(copiedlink.args,
                //                       self.graph.exceptblock.inputargs):
                //         if hasattr(a2, 'concretetype'):
                //             assert a1.concretetype == a2.concretetype
                //         else:
                //             a2.concretetype = a1.concretetype
                let cl_args = copiedlink.borrow().args.clone();
                let host_inputs = host_exceptblock.borrow().inputargs.clone();
                for (a1, a2) in cl_args.iter().zip(host_inputs.iter()) {
                    let a1_ct = match a1 {
                        Some(Hlvalue::Variable(v)) => v.concretetype(),
                        Some(Hlvalue::Constant(c)) => c.concretetype.clone(),
                        None => continue,
                    };
                    if let Hlvalue::Variable(a2_v) = a2 {
                        match a2_v.concretetype() {
                            Some(existing) => {
                                if let Some(a1_ct) = a1_ct {
                                    assert_eq!(
                                        existing, a1_ct,
                                        "inline.py:321 concretetype mismatch on \
                                         host_exceptblock.inputargs",
                                    );
                                }
                            }
                            None => {
                                a2_v.set_concretetype(a1_ct);
                            }
                        }
                    }
                }
            }
        }
    }

    /// `cleanup(self)` at `inline.py:432-436`.
    ///
    /// > cleaning up -- makes sense to be done after inlining,
    /// > because the inliner inserted quite some empty blocks
    /// > and blocks that can be joined.
    pub fn cleanup(&mut self) {
        crate::translator::simplify::cleanup_graph(&self.graph.borrow());
    }

    /// `Inliner.__init__(translator, graph, inline_func,
    /// lltype_to_classdef, ...)` at `inline.py:439-459`.
    ///
    /// ```python
    /// class Inliner(BaseInliner):
    ///     def __init__(self, translator, graph, inline_func, lltype_to_classdef,
    ///                  inline_guarded_calls=False,
    ///                  inline_guarded_calls_no_matter_what=False,
    ///                  raise_analyzer=None,
    ///                  call_count_pred=None,
    ///                  cleanup=True):
    ///         BaseInliner.__init__(self, translator, graph, lltype_to_classdef,
    ///                              inline_guarded_calls,
    ///                              inline_guarded_calls_no_matter_what,
    ///                              raise_analyzer,
    ///                              call_count_pred,
    ///                              cleanup)
    ///         self.inline_func = inline_func
    ///         # to simplify exception matching
    ///         join_blocks(graph)
    ///         # find callsites *after* joining blocks...
    ///         callsites = find_callsites(graph, inline_func)
    ///         self.block_to_index = {}
    ///         for g, block, i in callsites:
    ///             self.block_to_index.setdefault(block, {})[i] = g
    /// ```
    ///
    /// Pyre encodes upstream's `inline_func` into
    /// [`InlinerKind::Inliner`] rather than carrying a separate
    /// `Inliner` Rust type — the `OneShotInliner` / `Inliner`
    /// distinction is observed via the `kind` discriminator inside
    /// [`Self::search_for_calls`] (mirroring upstream's
    /// `OneShotInliner.search_for_calls = pass` override at
    /// `:461-463`).
    ///
    /// Sites whose `funcobj.graph` is `None` (callee has no
    /// FlowGraph attached, e.g. matched by `_callable` name on an
    /// external helper) are dropped at seed time. Upstream's
    /// `inline_all` would call `contains_call(None, None)` on those
    /// and AttributeError out, so the dropped behaviour matches
    /// upstream's reachable contract.
    pub fn new_inliner(
        translator: &'t TranslationContext,
        graph: GraphRef,
        inline_func: InlineFuncTarget,
        raise_analyzer: crate::translator::backendopt::canraise::RaiseAnalyzer<'t>,
        inline_guarded_calls: bool,
        inline_guarded_calls_no_matter_what: bool,
        call_count_pred: Option<CallCountPred>,
        cleanup: bool,
    ) -> Self {
        use crate::flowspace::model::BlockKey;
        // Upstream `:454 join_blocks(graph)`.
        crate::translator::simplify::join_blocks(&graph.borrow());
        // Upstream `:456 callsites = find_callsites(graph,
        // inline_func)`. Borrow `inline_func` to build the matcher,
        // then move it into the kind discriminator below.
        let matcher = match &inline_func {
            InlineFuncTarget::Graph(g) => CalleeMatcher::Graph(g),
            InlineFuncTarget::CallableName(name) => CalleeMatcher::CallableName(name.as_str()),
        };
        let callsites = find_callsites(&graph, matcher, translator);
        // Upstream `:446-451 BaseInliner.__init__(self, ...)`.
        let mut inliner = BaseInliner::new(
            translator,
            graph,
            InlinerKind::Inliner(inline_func),
            raise_analyzer,
            inline_guarded_calls,
            inline_guarded_calls_no_matter_what,
            call_count_pred,
            cleanup,
        );
        // Upstream `:457-459`:
        //     self.block_to_index = {}
        //     for g, block, i in callsites:
        //         self.block_to_index.setdefault(block, {})[i] = g
        for cs in callsites {
            let Some(callee) = cs.graph else {
                continue;
            };
            let key = BlockKey::of(&cs.block);
            inliner
                .block_to_index
                .entry(key)
                .or_insert_with(|| (cs.block.clone(), std::collections::HashMap::new()))
                .1
                .insert(cs.op_index, callee);
        }
        inliner
    }
}

/// `inline_function(translator, inline_func, graph,
/// lltype_to_classdef, raise_analyzer, call_count_pred=None,
/// cleanup=True)` at `inline.py:75-80`.
///
/// ```python
/// def inline_function(translator, inline_func, graph, lltype_to_classdef,
///                     raise_analyzer, call_count_pred=None, cleanup=True):
///     inliner = Inliner(translator, graph, inline_func, lltype_to_classdef,
///                       raise_analyzer=raise_analyzer,
///                       call_count_pred=call_count_pred, cleanup=cleanup)
///     return inliner.inline_all()
/// ```
///
/// `lltype_to_classdef` is unported (see
/// [`BaseInliner::lltype_to_classdef`] docstring); the parameter is
/// preserved as `()` for surface parity with the upstream signature
/// and threaded through unchanged. Upstream's defaults
/// `inline_guarded_calls=False` /
/// `inline_guarded_calls_no_matter_what=False` are not separately
/// configurable here — the wrapper covers only the public surface
/// upstream's free function exposes.
pub fn inline_function<'t>(
    translator: &'t TranslationContext,
    inline_func: InlineFuncTarget,
    graph: GraphRef,
    _lltype_to_classdef: (),
    raise_analyzer: RaiseAnalyzer<'t>,
    call_count_pred: Option<CallCountPred>,
    cleanup: bool,
) -> Result<usize, CannotInline> {
    let mut inliner = BaseInliner::new_inliner(
        translator,
        graph,
        inline_func,
        raise_analyzer,
        false,
        false,
        call_count_pred,
        cleanup,
    );
    inliner.inline_all()
}

/// `simple_inline_function(translator, inline_func, graph)` at
/// `inline.py:82-86`.
///
/// ```python
/// def simple_inline_function(translator, inline_func, graph):
///     inliner = Inliner(translator, graph, inline_func,
///                       translator.rtyper.lltype_to_classdef_mapping(),
///                       raise_analyzer=RaiseAnalyzer(translator))
///     return inliner.inline_all()
/// ```
///
/// `lltype_to_classdef_mapping()` is consumed only by Slice 4d's
/// `rewire_exceptblock_with_guard` / `generic_exception_matching`,
/// which `inline_once` already gates with `CannotInline`. Pyre's
/// thread-through is `()`; the wrapper is otherwise a faithful
/// port — non-exception-guarded callsites inline as upstream does,
/// and the guarded ones surface the same `CannotInline` shape until
/// Slice 4d lands.
pub fn simple_inline_function<'t>(
    translator: &'t TranslationContext,
    inline_func: InlineFuncTarget,
    graph: GraphRef,
) -> Result<usize, CannotInline> {
    let raise_analyzer = RaiseAnalyzer::new(translator);
    inline_function(
        translator,
        inline_func,
        graph,
        (),
        raise_analyzer,
        None,
        true,
    )
}

// ============================================================
// Slice 5: automatic inlining driver.
//
// `instrument_inline_candidates` (`:569-602`),
// `auto_inlining` (`:608-713`),
// `auto_inline_graphs` (`:715-731`).
// ============================================================

/// `instrument_inline_candidates(graphs, threshold)` at
/// `inline.py:569-602`.
///
/// ```python
/// def instrument_inline_candidates(graphs, threshold):
///     cache = {None: False}
///     def candidate(graph):
///         try:
///             return cache[graph]
///         except KeyError:
///             res = static_instruction_count(graph) <= threshold
///             cache[graph] = res
///             return res
///     n = 0
///     for parentgraph in graphs:
///         for block in parentgraph.iterblocks():
///             ops = block.operations
///             i = len(ops) - 1
///             while i >= 0:
///                 op = ops[i]
///                 i -= 1
///                 if op.opname == "direct_call":
///                     funcobj = op.args[0].value._obj
///                     graph = getattr(funcobj, 'graph', None)
///                     if graph is not None:
///                         if getattr(getattr(funcobj, '_callable', None),
///                                    '_dont_inline_', False):
///                             continue
///                     if candidate(graph):
///                         tag = Constant('inline', Void)
///                         label = Constant(n, Signed)
///                         dummy = Variable()
///                         dummy.concretetype = Void
///                         count = SpaceOperation('instrument_count',
///                                                [tag, label], dummy)
///                         ops.insert(i + 1, count)
///                         n += 1
///     log.inlining("%d call sites instrumented" % n)
/// ```
///
/// Walks every block of every graph in `graphs`, inserting an
/// `instrument_count` op directly **before** each `direct_call`
/// whose callee is small enough to be a viable inlining candidate
/// (`static_instruction_count(callee) <= threshold`). The two
/// args of the inserted op are a `'inline'` tag (Void) and a
/// monotonically increasing `Signed` label.
///
/// Calls whose callee has the `_dont_inline_` flag set on its
/// `_callable` are skipped — see [`callable_dont_inline`] for the
/// pyre PRE-EXISTING-ADAPTATION on that flag.
///
/// Returns the number of inserted ops. Upstream's trailing
/// `log.inlining` is replaced with a return value so tests can
/// assert against the count without binding a logger.
pub fn instrument_inline_candidates(
    graphs: &[GraphRef],
    threshold: f64,
    translator: &TranslationContext,
) -> usize {
    use crate::flowspace::model::{ConstValue, Constant, Variable};
    use crate::translator::rtyper::lltypesystem::lltype::{SIGNED, VOID};

    // Upstream `:570 cache = {None: False}`. Sentinel-keyed by
    // graph identity (None -> graphless callee, sets to False).
    // Real graphs index by `GraphKey::of(graph).as_usize()`.
    let mut cache: std::collections::HashMap<Option<usize>, bool> =
        std::collections::HashMap::new();
    cache.insert(None, false);

    let mut n: usize = 0;
    for parentgraph in graphs {
        let blocks = parentgraph.borrow().iterblocks();
        for block in &blocks {
            // Upstream `:581-583 ops = block.operations; i = len(ops)
            // - 1; while i >= 0:`. Iterate in reverse over a snapshot
            // length; inserts at i+1 (post-decrement) push later ops
            // out of the way without affecting earlier indices.
            let n_ops = block.borrow().operations.len();
            let mut i: isize = n_ops as isize - 1;
            while i >= 0 {
                // Upstream `:584-585 op = ops[i]; i -= 1`.
                let op = block.borrow().operations[i as usize].clone();
                i -= 1;
                // Upstream `:586-587 if op.opname == "direct_call":`.
                if op.opname != "direct_call" {
                    continue;
                }
                // Upstream `:587-588 funcobj = op.args[0].value._obj;
                // graph = getattr(funcobj, 'graph', None)`.
                let callee_graph = op
                    .args
                    .first()
                    .and_then(|arg| get_graph_for_call(arg, translator));
                // Upstream `:589-592 if graph is not None: if
                // getattr(getattr(funcobj, '_callable', None),
                // '_dont_inline_', False): continue`.
                if let Some(g) = callee_graph.as_ref() {
                    if callable_dont_inline(g) {
                        continue;
                    }
                }
                // Upstream `:593 if candidate(graph):`.
                let key = callee_graph.as_ref().map(|g| GraphKey::of(g).as_usize());
                let res = match cache.get(&key) {
                    Some(&v) => v,
                    None => {
                        // Upstream `candidate(graph)` body — only
                        // reached when `key` is `Some(...)` because
                        // the `None` key was pre-seeded with `False`.
                        let val = match &callee_graph {
                            Some(g) => static_instruction_count(g) <= threshold,
                            None => false,
                        };
                        cache.insert(key, val);
                        val
                    }
                };
                if !res {
                    continue;
                }
                // Upstream `:594-599 tag = Constant('inline', Void);
                // label = Constant(n, Signed); dummy = Variable();
                // dummy.concretetype = Void; count =
                // SpaceOperation('instrument_count', [tag, label],
                // dummy)`.
                let tag = Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::UniStr("inline".to_string()),
                    VOID,
                ));
                let label = Hlvalue::Constant(Constant::with_concretetype(
                    ConstValue::Int(n as i64),
                    SIGNED,
                ));
                let dummy = Variable::new();
                dummy.set_concretetype(Some(VOID));
                let count_op = SpaceOperation::new(
                    "instrument_count",
                    vec![tag, label],
                    Hlvalue::Variable(dummy),
                );
                // Upstream `:600 ops.insert(i + 1, count)`. `i` was
                // already decremented; `i + 1` is the original
                // op's index, so the count op is inserted *before*
                // the matched direct_call.
                block
                    .borrow_mut()
                    .operations
                    .insert((i + 1) as usize, count_op);
                n += 1;
            }
        }
    }
    n
}

/// Heap entry for [`auto_inlining`] mirroring upstream's
/// `(weight, -len(callers), graph)` tuple at `inline.py:624` /
/// `:647` / `:700`. Upstream's `heapq` is a min-heap; pyre uses
/// `BinaryHeap` (max-heap) wrapped in `Reverse` to recover min-first
/// ordering. The third tie-breaker is graph identity — pyre orders
/// by `Rc::as_ptr` for total ordering.
#[derive(Clone, Debug)]
struct AutoInliningHeapEntry {
    weight: f64,
    neg_callers: i64,
    graph: GraphRef,
}

impl PartialEq for AutoInliningHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.weight.total_cmp(&other.weight) == std::cmp::Ordering::Equal
            && self.neg_callers == other.neg_callers
            && Rc::ptr_eq(&self.graph, &other.graph)
    }
}

impl Eq for AutoInliningHeapEntry {}

impl PartialOrd for AutoInliningHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AutoInliningHeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight
            .total_cmp(&other.weight)
            .then(self.neg_callers.cmp(&other.neg_callers))
            .then((Rc::as_ptr(&self.graph) as usize).cmp(&(Rc::as_ptr(&other.graph) as usize)))
    }
}

/// `auto_inlining(translator, threshold, callgraph=None,
/// call_count_pred=None, heuristic=inlining_heuristic)` at
/// `inline.py:608-713`.
///
/// Heap-driven driver that repeatedly picks the lowest-weight graph
/// from the call graph and inlines it into every parent that still
/// calls it, accumulating a successor call graph as it goes.
/// Returns the total number of inlines performed.
///
/// `call_count_pred` is shared across every per-parent
/// `inline_function` invocation via [`CallCountPred`]
/// (`Rc<RefCell<dyn FnMut(i64) -> bool>>`) — upstream Python keeps
/// the same closure object alive across the heap-driven loop, and
/// the `Rc` carrier is the minimum Rust adaptation that preserves
/// that identity (a `Box<dyn FnMut>` would force every clone to
/// degrade to `None`).
///
/// Upstream's `RaiseAnalyzer` is constructed once and reused across
/// every `inline_function` invocation; pyre's
/// `BaseInliner::new_inliner` consumes it, so this slice creates a
/// fresh `RaiseAnalyzer::new(translator)` per inline call. The
/// observable cache miss is bounded — `RaiseAnalyzer.analyzed_calls`
/// memoizes within a single `inline_all`, where most lookups are
/// concentrated; cross-graph sharing is a perf-only deviation.
/// Convergence path: change `BaseInliner.raise_analyzer` to a
/// borrowed `&'a mut RaiseAnalyzer<'t>`, threading a second lifetime
/// through the struct.
pub fn auto_inlining(
    translator: &TranslationContext,
    threshold: f64,
    callgraph: Option<Vec<(GraphRef, GraphRef)>>,
    heuristic: fn(&GraphRef) -> (f64, bool),
    call_count_pred: Option<CallCountPred>,
) -> Result<usize, CannotInline> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};

    // Upstream `:613 assert threshold is not None and threshold != 1`.
    assert!(
        threshold != 1.0,
        "inline.py:613 auto_inlining: threshold must not be 1.0",
    );

    // Upstream `:614 to_cleanup = {}`.
    let mut to_cleanup: HashMap<usize, GraphRef> = HashMap::new();
    // Upstream `:616-617 callers = {}; callees = {}`.
    let mut callers: HashMap<usize, HashMap<usize, GraphRef>> = HashMap::new();
    let mut callees: HashMap<usize, HashMap<usize, GraphRef>> = HashMap::new();
    // Pyre needs a key→graph reverse lookup since `HashMap<usize, _>`
    // can't recover `GraphRef` from the key alone (upstream Python
    // stores the graph object directly as a dict key).
    let mut key_to_graph: HashMap<usize, GraphRef> = HashMap::new();

    // Upstream `:618-619 if callgraph is None: callgraph =
    // inlinable_static_callers(translator.graphs)`.
    let callgraph_pairs: Vec<(GraphRef, GraphRef)> = match callgraph {
        Some(c) => c,
        None => {
            let trans_graphs: Vec<GraphRef> = translator.graphs.borrow().clone();
            let entries = inlinable_static_callers(&trans_graphs, false, None, translator);
            entries
                .into_iter()
                .filter_map(|e| match e {
                    InlinableCallerEntry::GraphPair { parent, callee } => Some((parent, callee)),
                    InlinableCallerEntry::OpSite { .. } => None,
                })
                .collect()
        }
    };

    // Upstream `:620-622 for graph1, graph2 in callgraph: callers...;
    // callees...`.
    for (g1, g2) in &callgraph_pairs {
        let k1 = GraphKey::of(g1).as_usize();
        let k2 = GraphKey::of(g2).as_usize();
        key_to_graph.insert(k1, g1.clone());
        key_to_graph.insert(k2, g2.clone());
        callers.entry(k2).or_default().insert(k1, g1.clone());
        callees.entry(k1).or_default().insert(k2, g2.clone());
    }

    // Upstream `:624 heap = [(0.0, -len(callers[g]), g) for g in callers]`.
    let mut heap: BinaryHeap<Reverse<AutoInliningHeapEntry>> = callers
        .iter()
        .map(|(k, parents)| {
            Reverse(AutoInliningHeapEntry {
                weight: 0.0,
                neg_callers: -(parents.len() as i64),
                graph: key_to_graph[k].clone(),
            })
        })
        .collect();

    // Upstream `:625-626 valid_weight = {}; try_again = {}`.
    let mut valid_weight: HashMap<usize, bool> = HashMap::new();
    let mut try_again: HashMap<usize, String> = HashMap::new();
    // Upstream `:627 lltype_to_classdef =
    // translator.rtyper.lltype_to_classdef_mapping()`. Threaded as
    // `()` per `inline_function` signature; PRE-EXISTING-ADAPTATION
    // documented at `BaseInliner.lltype_to_classdef`.
    let lltype_to_classdef: () = ();
    // Upstream `:629 count = 0`.
    let mut count: usize = 0;

    // Upstream `:630 while heap:`.
    'main_loop: while let Some(Reverse(top)) = heap.peek().cloned() {
        let graph_key = GraphKey::of(&top.graph).as_usize();
        let weight = top.weight;

        // Upstream `:632-651 if not valid_weight.get(graph): ...`.
        if !valid_weight.get(&graph_key).copied().unwrap_or(false) {
            // Upstream `:633-634 if always_inline(graph): weight,
            // fixed = 0.0, True`.
            let (new_weight, fixed) = if always_inline(&top.graph) {
                (0.0, true)
            } else {
                // Upstream `:636 weight, fixed = heuristic(graph)`.
                let (w, f) = heuristic(&top.graph);
                // Upstream `:644-645 if not (weight < 1e9): weight = 1e9`.
                let w = if !(w < 1e9) { 1e9 } else { w };
                (w, f)
            };
            // Upstream `:647 heapreplace(heap, ...)`. Rust BinaryHeap
            // has no heapreplace; pop+push reaches the same end state.
            heap.pop();
            let parents_len = callers.get(&graph_key).map(|c| c.len() as i64).unwrap_or(0);
            heap.push(Reverse(AutoInliningHeapEntry {
                weight: new_weight,
                neg_callers: -parents_len,
                graph: top.graph.clone(),
            }));
            // Upstream `:648 valid_weight[graph] = True`.
            valid_weight.insert(graph_key, true);
            // Upstream `:649-650 if not fixed: try_again[graph] =
            // 'initial'`.
            if !fixed {
                try_again.insert(graph_key, "initial".to_string());
            }
            continue;
        }

        // Upstream `:653-667 if weight >= threshold: ...`.
        if weight >= threshold {
            // Upstream `:657-662 finished = True; for i in range(len(heap)):
            //   graph = heap[i][2]; if not valid_weight.get(graph):
            //     heap[i] = (0.0, heap[i][1], graph); finished = False`.
            // Rust: drain the heap into a Vec, mutate in place,
            // rebuild via `BinaryHeap::from`.
            let mut entries: Vec<AutoInliningHeapEntry> =
                heap.drain().map(|Reverse(e)| e).collect();
            let mut finished = true;
            for e in entries.iter_mut() {
                let k = GraphKey::of(&e.graph).as_usize();
                if !valid_weight.get(&k).copied().unwrap_or(false) {
                    e.weight = 0.0;
                    finished = false;
                }
            }
            // Upstream `:663-664 if finished: break`.
            if finished {
                break 'main_loop;
            }
            // Upstream `:665-667 else: heapify(heap); continue`.
            heap = entries.into_iter().map(Reverse).collect();
            continue;
        }

        // Upstream `:669 heappop(heap)`.
        heap.pop();
        // Upstream `:670-674 if callers[graph]: log`. The log is
        // skipped — pyre returns `count` from the function so callers
        // can report instead.

        // Upstream `:675 for parentgraph in callers[graph]:`. Snapshot
        // the iteration set; the inner body mutates `callers[graph2]`
        // for graph2 ≠ graph but never `callers[graph]` itself.
        let parents_of_graph: Vec<(usize, GraphRef)> = callers
            .get(&graph_key)
            .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
            .unwrap_or_default();

        for (parent_key, parent_graph) in parents_of_graph {
            // Upstream `:676-677 if parentgraph == graph: continue`.
            if parent_key == graph_key {
                continue;
            }
            // Upstream `:679-687`:
            //     try:
            //         subcount = inline_function(translator, graph,
            //                                    parentgraph,
            //                                    lltype_to_classdef,
            //                                    raise_analyzer,
            //                                    call_count_pred,
            //                                    cleanup=False)
            //         to_cleanup[parentgraph] = True
            //         res = bool(subcount)
            //     except CannotInline as e:
            //         try_again[graph] = str(e)
            //         res = CannotInline
            // Pyre creates a fresh RaiseAnalyzer per call; see the
            // function-level PRE-EXISTING-ADAPTATION note.
            let raise = RaiseAnalyzer::new(translator);
            let result = inline_function(
                translator,
                InlineFuncTarget::Graph(top.graph.clone()),
                parent_graph.clone(),
                lltype_to_classdef,
                raise,
                call_count_pred.clone(),
                false, // upstream `:682 cleanup=False`
            );
            let (subcount, res_is_true) = match result {
                Ok(c) => {
                    to_cleanup.insert(parent_key, parent_graph.clone());
                    (c, c > 0)
                }
                Err(e) => {
                    try_again.insert(graph_key, e.0);
                    (0, false)
                }
            };
            // Upstream `:688-701 if res is True: ...`.
            if res_is_true {
                count += subcount;
                // Upstream `:692-694 for graph2 in callees.get(graph,
                // {}): callees[parentgraph][graph2] = True;
                // callers[graph2][parentgraph] = True`.
                let g2_keys: Vec<(usize, GraphRef)> = callees
                    .get(&graph_key)
                    .map(|m| m.iter().map(|(k, v)| (*k, v.clone())).collect())
                    .unwrap_or_default();
                for (g2_key, g2_graph) in g2_keys {
                    callees
                        .entry(parent_key)
                        .or_default()
                        .insert(g2_key, g2_graph.clone());
                    callers
                        .entry(g2_key)
                        .or_default()
                        .insert(parent_key, parent_graph.clone());
                    key_to_graph.entry(g2_key).or_insert(g2_graph);
                }
                // Upstream `:695-700 if parentgraph in try_again: ...
                // heappush(heap, (0.0, -len(callers[parentgraph]),
                // parentgraph))`.
                if try_again.contains_key(&parent_key) {
                    try_again.remove(&parent_key);
                    let parents_of_parent_len = callers
                        .get(&parent_key)
                        .map(|c| c.len() as i64)
                        .unwrap_or(0);
                    heap.push(Reverse(AutoInliningHeapEntry {
                        weight: 0.0,
                        neg_callers: -parents_of_parent_len,
                        graph: parent_graph.clone(),
                    }));
                }
                // Upstream `:701 valid_weight[parentgraph] = False`.
                valid_weight.insert(parent_key, false);
            }
        }
    }

    // Upstream `:703-709`:
    //     invalid = [(graph, msg) for graph, msg in try_again.items()
    //                              if always_inline(graph) is True]
    //     if invalid:
    //         message = '\n'.join([
    //             "%s has _always_inline_=True but inlining failed:\n\t%s" %
    //             (graph, msg) for (graph, msg) in invalid])
    //         raise CannotInline(message)
    let invalid: Vec<(usize, String)> = try_again
        .iter()
        .filter(|(k, _)| {
            key_to_graph
                .get(k)
                .map(|g| always_inline(g))
                .unwrap_or(false)
        })
        .map(|(k, v)| (*k, v.clone()))
        .collect();
    if !invalid.is_empty() {
        let lines: Vec<String> = invalid
            .iter()
            .map(|(k, msg)| {
                let name = key_to_graph
                    .get(k)
                    .map(|g| g.borrow().name.clone())
                    .unwrap_or_else(|| format!("<graph#{k}>"));
                format!("{name} has _always_inline_=True but inlining failed:\n\t{msg}")
            })
            .collect();
        return Err(CannotInline::new(lines.join("\n")));
    }

    // Upstream `:711-712 for graph in to_cleanup: cleanup_graph(graph)`.
    for graph in to_cleanup.values() {
        crate::translator::simplify::cleanup_graph(&graph.borrow());
    }
    // Upstream `:713 return count`.
    Ok(count)
}

/// `auto_inline_graphs(translator, graphs, threshold,
/// call_count_pred=None, heuristic=inlining_heuristic,
/// inline_graph_from_anywhere=False)` at `inline.py:715-731`.
///
/// ```python
/// def auto_inline_graphs(translator, graphs, threshold, call_count_pred=None,
///                        heuristic=inlining_heuristic,
///                        inline_graph_from_anywhere=False):
///     if inline_graph_from_anywhere:
///         ok_to_call = set([graph for graph in translator.graphs
///                                 if not hasattr(graph, 'exceptiontransformed')])
///     else:
///         ok_to_call = None
///     callgraph = inlinable_static_callers(graphs, ok_to_call=ok_to_call)
///     count = auto_inlining(translator, threshold, callgraph=callgraph,
///                           heuristic=heuristic,
///                           call_count_pred=call_count_pred)
///     log.inlining('inlined %d callsites.' % (count,))
///     for graph in graphs:
///         removenoops.remove_duplicate_casts(graph, translator)
/// ```
///
/// `call_count_pred` is forwarded to [`auto_inlining`] verbatim;
/// the shareable `Rc<RefCell<dyn FnMut>>` carrier is documented at
/// [`CallCountPred`].
///
/// `inline_graph_from_anywhere=true` would upstream filter out
/// graphs that have already been exception-transformed
/// (`hasattr(graph, 'exceptiontransformed')`). Pyre has no
/// `exceptiontransformed` attribute carrier yet
/// (`exceptiontransform.py` is unported), so the filter is a no-op
/// — every graph in `translator.graphs` qualifies.
/// PRE-EXISTING-ADAPTATION: when `exceptiontransform` lands and
/// stamps the attribute, this arm needs the corresponding skip.
pub fn auto_inline_graphs(
    translator: &TranslationContext,
    graphs: &[GraphRef],
    threshold: f64,
    heuristic: fn(&GraphRef) -> (f64, bool),
    call_count_pred: Option<CallCountPred>,
    inline_graph_from_anywhere: bool,
) -> Result<usize, CannotInline> {
    // Upstream `:718-724 if inline_graph_from_anywhere: ok_to_call =
    // {...}; else: ok_to_call = None`.
    let ok_to_call_storage: Vec<GraphRef> = if inline_graph_from_anywhere {
        translator.graphs.borrow().clone()
    } else {
        Vec::new()
    };
    let ok_to_call: Option<&[GraphRef]> = if inline_graph_from_anywhere {
        Some(&ok_to_call_storage)
    } else {
        None
    };
    // Upstream `:725 callgraph = inlinable_static_callers(graphs,
    // ok_to_call=ok_to_call)`.
    let entries = inlinable_static_callers(graphs, false, ok_to_call, translator);
    let callgraph: Vec<(GraphRef, GraphRef)> = entries
        .into_iter()
        .filter_map(|e| match e {
            InlinableCallerEntry::GraphPair { parent, callee } => Some((parent, callee)),
            InlinableCallerEntry::OpSite { .. } => None,
        })
        .collect();
    // Upstream `:726-728 count = auto_inlining(...)`.
    let count = auto_inlining(
        translator,
        threshold,
        Some(callgraph),
        heuristic,
        call_count_pred,
    )?;
    // Upstream `:729 log.inlining('inlined %d callsites.' % (count,))`.
    // Skipped — the return value carries the count.
    // Upstream `:730-731 for graph in graphs:
    // removenoops.remove_duplicate_casts(graph, translator)`.
    for g in graphs {
        let _ = crate::translator::backendopt::removenoops::remove_duplicate_casts(
            &g.borrow(),
            translator,
        );
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Hlvalue, Link, SpaceOperation,
        Variable,
    };
    use std::cell::RefCell;

    fn fixture_translator() -> TranslationContext {
        TranslationContext::new()
    }

    fn empty_graph(name: &str) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new(name, start.clone());
        let return_target = graph.returnblock.clone();
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    /// Single-op graph carrying an int_add (pure, non-raising).
    fn int_add_graph(name: &str) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new(name, start.clone());
        let return_target = graph.returnblock.clone();
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    /// Single-op graph carrying an int_add_ovf (raises OverflowError).
    fn int_add_ovf_graph(name: &str) -> GraphRef {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new(name, start.clone());
        let return_target = graph.returnblock.clone();
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        Rc::new(RefCell::new(graph))
    }

    #[test]
    fn cannot_inline_carries_message() {
        let e = CannotInline::new("test");
        assert_eq!(e.0, "test");
    }

    #[test]
    fn can_raise_hint_round_trips_bool() {
        assert!(CanRaiseHint::new(true).can_raise);
        assert!(!CanRaiseHint::new(false).can_raise);
    }

    #[test]
    fn collect_called_graphs_on_no_call_graph_is_empty() {
        let translator = fixture_translator();
        let g = int_add_graph("f");
        assert!(collect_called_graphs(&g, &translator).is_empty());
    }

    #[test]
    fn iter_callsites_skips_non_direct_call_ops() {
        let translator = fixture_translator();
        let g = int_add_graph("f");
        assert!(iter_callsites(&g, CalleeMatcher::Any, &translator).is_empty());
    }

    #[test]
    fn contains_call_returns_false_when_no_call_present() {
        let translator = fixture_translator();
        let g = int_add_graph("f");
        assert!(!contains_call(&g, CalleeMatcher::Any, &translator));
    }

    #[test]
    fn does_raise_directly_int_add_graph_is_false() {
        let translator = fixture_translator();
        let g = int_add_graph("f");
        let mut r = RaiseAnalyzer::new(&translator);
        assert!(!does_raise_directly(&g, &mut r));
    }

    #[test]
    fn does_raise_directly_int_add_ovf_graph_is_true() {
        let translator = fixture_translator();
        let g = int_add_ovf_graph("f");
        let mut r = RaiseAnalyzer::new(&translator);
        assert!(does_raise_directly(&g, &mut r));
    }

    #[test]
    fn does_raise_directly_skips_last_raising_op_under_canraise_block() {
        // `consider_ops_to = -1` upstream: when `block.canraise`, the
        // last op is the raising op and is excluded — its raising
        // potential is already encoded by the exception edge. Build
        // a block whose final op is int_add_ovf with the
        // c_last_exception sentinel as exitswitch and verify the
        // walker does not double-count.
        let translator = fixture_translator();
        let mut r = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let return_target = graph.returnblock.clone();
        start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add_ovf",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                Hlvalue::Constant(Constant::new(ConstValue::Int(2))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        // Fake c_last_exception exitswitch: use the LAST_EXCEPTION
        // atom so `canraise()` returns True.
        use crate::flowspace::model::{Atom, LAST_EXCEPTION};
        let last_exc_atom = Atom {
            name: LAST_EXCEPTION.name.clone(),
        };
        start.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::new(ConstValue::Atom(
            last_exc_atom,
        ))));
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        // exceptblock is reachable via the canraise edge — upstream
        // `:113-114 if block is graph.exceptblock: return True` —
        // but our test graph never closes the exceptblock as a
        // proper exit, so iterblocks should not include it. Verify
        // walk excludes the trailing int_add_ovf because the block
        // is canraise.
        // (`iterblocks` only enumerates reachable blocks; exceptblock
        // is reachable only when something links to it.)
        assert!(!does_raise_directly(&g, &mut r));
    }

    #[test]
    fn does_raise_directly_marks_reachable_exceptblock_as_raising() {
        // Graph whose exceptblock is reachable through a link — the
        // walker should bail with `True` per upstream `:113-114`.
        let translator = fixture_translator();
        let mut r = RaiseAnalyzer::new(&translator);
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        let except_target = graph.exceptblock.clone();
        // Direct link straight into the exceptblock.
        start.closeblock(vec![
            Link::new(
                vec![
                    Hlvalue::Constant(Constant::new(ConstValue::None)),
                    Hlvalue::Constant(Constant::new(ConstValue::None)),
                ],
                Some(except_target),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        assert!(does_raise_directly(&g, &mut r));
    }

    #[test]
    fn any_call_to_raising_graphs_with_no_call_falls_through_to_self_walk() {
        // No `direct_call` callees ⇒ walk falls through to the
        // `:134-141` self-walk arm and inspects from_graph's own ops.
        let translator = fixture_translator();
        let mut r = RaiseAnalyzer::new(&translator);
        // Pure int_add ⇒ False.
        let pure = int_add_graph("f");
        assert!(!any_call_to_raising_graphs(&pure, &translator, &mut r));
        // int_add_ovf ⇒ True.
        let raises = int_add_ovf_graph("g");
        assert!(any_call_to_raising_graphs(&raises, &translator, &mut r));
    }

    #[test]
    fn calleematcher_any_matches_every_direct_call_arity_zero_graph() {
        // Sanity: empty graph has no callsites.
        let translator = fixture_translator();
        let g = empty_graph("h");
        assert!(iter_callsites(&g, CalleeMatcher::Any, &translator).is_empty());
        assert!(find_callsites(&g, CalleeMatcher::Any, &translator).is_empty());
        assert!(!contains_call(&g, CalleeMatcher::Any, &translator));
    }

    // ----- Slice 2: auto-inlining heuristics -----

    #[test]
    fn op_weight_table_matches_upstream_six_overrides() {
        // Upstream :470-476.
        assert_eq!(op_weight("same_as"), 0);
        assert_eq!(op_weight("cast_pointer"), 0);
        assert_eq!(op_weight("malloc"), 2);
        assert_eq!(op_weight("instrument_count"), 0);
        assert_eq!(op_weight("debug_assert"), -1);
        assert_eq!(op_weight("jit_force_virtualizable"), 0);
        // Upstream :485 default — `weights.get(opname, 1)`.
        assert_eq!(op_weight("int_add"), 1);
        assert_eq!(op_weight("any_other_op_name"), 1);
    }

    #[test]
    fn block_weight_int_add_block_with_no_exitswitch_returns_one() {
        // int_add ⇒ default weight 1; no exitswitch ⇒ no +1.
        let g = int_add_graph("f");
        let block = g.borrow().startblock.clone();
        assert_eq!(block_weight(&block), 1.0);
    }

    #[test]
    fn block_weight_empty_block_with_no_exitswitch_returns_zero() {
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        let block = g.borrow().startblock.clone();
        assert_eq!(block_weight(&block), 0.0);
    }

    #[test]
    fn block_weight_direct_call_two_args_returns_two_point_five_plus_default() {
        // direct_call: 1.5 + len(args)/2 = 1.5 + 1 = 2.5; plus
        // weights.get('direct_call', 1) = 1; total = 3.5.
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                Hlvalue::Constant(Constant::new(ConstValue::None)),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        let block = g.borrow().startblock.clone();
        assert_eq!(block_weight(&block), 3.5);
    }

    #[test]
    fn block_weight_clamps_negative_at_zero() {
        // debug_assert weight is -1; a single one and no exitswitch
        // would give -1, clamped to 0 by max(0, total).
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("f", start.clone());
        start.borrow_mut().operations.push(SpaceOperation::new(
            "debug_assert",
            vec![Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))],
            Hlvalue::Variable(Variable::named("r")),
        ));
        start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        let block = g.borrow().startblock.clone();
        assert_eq!(block_weight(&block), 0.0);
    }

    #[test]
    fn static_instruction_count_int_add_graph_returns_one() {
        let g = int_add_graph("f");
        assert_eq!(static_instruction_count(&g), 1.0);
    }

    #[test]
    fn always_inline_returns_false_until_flag_lands() {
        let g = int_add_graph("f");
        // PRE-EXISTING-ADAPTATION: pyre's GraphFunc has no
        // `_always_inline_` slot, so the helper is structurally
        // always-False today.
        assert!(!always_inline(&g));
    }

    #[test]
    fn inlinable_static_callers_no_calls_returns_empty() {
        let translator = fixture_translator();
        let g = int_add_graph("f");
        let result = inlinable_static_callers(&[g.clone()], false, None, &translator);
        assert_eq!(result.len(), 0);
        let result_with_ops = inlinable_static_callers(&[g], true, None, &translator);
        assert_eq!(result_with_ops.len(), 0);
    }

    // ----- Slice 3: median-execution cost + inlining_heuristic -----

    #[test]
    fn measure_median_execution_cost_int_add_graph_returns_block_weight() {
        // Graph has a single startblock (with int_add op weight 1)
        // and a returnblock (weight 0). The link from startblock to
        // returnblock yields full probability flow, so the median
        // execution cost equals startblock's weight = 1.
        let g = int_add_graph("f");
        let cost = measure_median_execution_cost(&g);
        // Linear graph: cost == startblock weight.
        assert!((cost - 1.0).abs() < 1e-6);
    }

    #[test]
    fn inlining_heuristic_small_graph_combines_count_and_median() {
        let g = int_add_graph("f");
        let (weight, fixed) = inlining_heuristic(&g);
        assert!(fixed);
        // count = 1, median ≈ 1 ⇒ 0.9999 + 1 ≈ 1.9999
        assert!((weight - 1.9999).abs() < 1e-3);
    }

    #[test]
    fn inlining_heuristic_returns_count_when_above_threshold() {
        // Build a graph whose static_instruction_count >= 200.
        // 200 ops of int_add ⇒ count = 200 ⇒ heuristic returns 200.
        let v = Variable::named("x");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let graph = FunctionGraph::new("big", start.clone());
        let return_target = graph.returnblock.clone();
        for i in 0..200 {
            start.borrow_mut().operations.push(SpaceOperation::new(
                "int_add",
                vec![
                    Hlvalue::Constant(Constant::new(ConstValue::Int(i as i64))),
                    Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
                ],
                Hlvalue::Variable(Variable::named(&format!("r{i}"))),
            ));
        }
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(return_target), None).into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(graph));
        let (weight, fixed) = inlining_heuristic(&g);
        assert!(fixed);
        assert_eq!(weight, 200.0);
    }

    // ----- Slice 4a: BaseInliner construction + simple helpers -----

    fn fixture_inliner<'t>(
        translator: &'t TranslationContext,
        graph: GraphRef,
        kind: InlinerKind,
    ) -> BaseInliner<'t> {
        let raise_analyzer = RaiseAnalyzer::new(translator);
        BaseInliner::new(
            translator,
            graph,
            kind,
            raise_analyzer,
            false,
            false,
            None,
            true,
        )
    }

    #[test]
    fn baseinliner_new_initialises_state_to_upstream_defaults() {
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let inliner = fixture_inliner(&translator, g.clone(), InlinerKind::OneShot);
        assert!(inliner.do_cleanup);
        assert!(!inliner.inline_guarded_calls);
        assert!(!inliner.inline_guarded_calls_no_matter_what);
        assert!(inliner.block_to_index.is_empty());
        assert!(inliner.varmap.is_empty());
        assert!(inliner.copied_blocks.is_empty());
        assert!(inliner.op.is_none());
        assert!(inliner.graph_to_inline.is_none());
        assert!(!inliner.exception_guarded);
        assert!(inliner.passon_vars_cache.is_empty());
        assert!(inliner.entrymap.is_empty());
        assert!(inliner.original_passon_vars.is_empty());
    }

    #[test]
    fn get_new_name_caches_variable_copy_and_passes_constants_through() {
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g, InlinerKind::OneShot);
        let v = Variable::named("v");
        let renamed = inliner.get_new_name(&Hlvalue::Variable(v.clone()));
        let Hlvalue::Variable(rv) = renamed else {
            panic!("expected Variable")
        };
        assert_ne!(rv.id(), v.id(), "fresh copy");
        // Cached: second call returns the same copy.
        let renamed2 = inliner.get_new_name(&Hlvalue::Variable(v.clone()));
        let Hlvalue::Variable(rv2) = renamed2 else {
            panic!()
        };
        assert_eq!(rv.id(), rv2.id(), "same cached copy");
        // Constants pass through unchanged.
        let c = Hlvalue::Constant(Constant::new(ConstValue::Int(42)));
        assert_eq!(inliner.get_new_name(&c), c);
    }

    #[test]
    fn passon_vars_caches_on_block_key() {
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g, InlinerKind::OneShot);
        inliner.original_passon_vars = vec![Variable::named("a"), Variable::named("b")];
        let block = Block::shared(vec![]);
        let key = crate::flowspace::model::BlockKey::of(&block);
        let first = inliner.passon_vars(PassonCacheKey::Block(key.clone()));
        let second = inliner.passon_vars(PassonCacheKey::Block(key));
        // Same identities returned on the second call.
        assert_eq!(first.len(), 2);
        assert_eq!(first[0].id(), second[0].id());
        assert_eq!(first[1].id(), second[1].id());
        // The copies have fresh ids relative to the originals.
        assert_ne!(first[0].id(), inliner.original_passon_vars[0].id());
    }

    #[test]
    fn copy_operation_renames_args_and_result_through_varmap() {
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g, InlinerKind::OneShot);
        let a = Variable::named("a");
        let r = Variable::named("r");
        let op = SpaceOperation::new(
            "int_add",
            vec![
                Hlvalue::Variable(a.clone()),
                Hlvalue::Constant(Constant::new(ConstValue::Int(1))),
            ],
            Hlvalue::Variable(r.clone()),
        );
        let new_op = inliner.copy_operation(&op);
        assert_eq!(new_op.opname, "int_add");
        let Hlvalue::Variable(arg0) = &new_op.args[0] else {
            panic!()
        };
        let Hlvalue::Constant(_) = &new_op.args[1] else {
            panic!("constant should pass through")
        };
        assert_ne!(arg0.id(), a.id());
        let Hlvalue::Variable(res) = &new_op.result else {
            panic!()
        };
        assert_ne!(res.id(), r.id());
    }

    #[test]
    fn search_for_calls_oneshot_kind_is_noop() {
        // OneShot: search_for_calls early-returns; block_to_index
        // stays empty even when the block carries a direct_call.
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g.clone(), InlinerKind::OneShot);
        let v = Variable::named("v");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![Hlvalue::Constant(Constant::new(ConstValue::None))],
            Hlvalue::Variable(Variable::named("r")),
        ));
        inliner.search_for_calls(&start);
        assert!(inliner.block_to_index.is_empty());
    }

    #[test]
    fn copy_block_terminates_on_self_loop_via_copied_blocks_cache() {
        // Self-loop block: start -> start. copy_block must not
        // recurse forever — the upstream `_copied_blocks[block] =
        // newblock` cache write at `:259` happens BEFORE copy_link
        // recurses into `link.target = block` (self-loop).
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g, InlinerKind::OneShot);
        let v = Variable::named("v");
        let start = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        start.closeblock(vec![
            Link::new(vec![Hlvalue::Variable(v)], Some(start.clone()), None).into_ref(),
        ]);
        let copied = inliner.copy_block(&start);
        // The copy's exit links back to itself (the COPIED block,
        // not the original).
        let exit0 = copied.borrow().exits[0].clone();
        let target = exit0.borrow().target.clone().unwrap();
        assert!(Rc::ptr_eq(&target, &copied));
    }

    // ----- Slice 4b: orchestrator + drivers -----

    #[test]
    fn inline_all_empty_queue_returns_zero() {
        let translator = fixture_translator();
        let g = int_add_graph("host");
        let mut inliner = fixture_inliner(&translator, g, InlinerKind::OneShot);
        // do_cleanup is True by default — make sure cleanup_graph
        // tolerates a clean queue.
        let count = inliner.inline_all().expect("empty queue");
        assert_eq!(count, 0);
    }

    #[test]
    fn inline_once_rejects_exception_guarded_call_with_slice_4d_gap() {
        // Build a host graph whose only block has a single
        // direct_call op AND `block.canraise()` is true (i.e. the
        // call is a try-except site). Pyre's `Block::canraise()`
        // requires `block.exitswitch == c_last_exception`. For Slice
        // 4b we synthesise that shape and verify inline_once
        // returns CannotInline.
        use crate::flowspace::model::{Atom, LAST_EXCEPTION};
        let translator = fixture_translator();
        let host = int_add_graph("host");

        // Trivial callee — only needs to be resolvable as a graph.
        let callee_x = Variable::named("x");
        let callee_start = Block::shared(vec![Hlvalue::Variable(callee_x.clone())]);
        let callee_graph = FunctionGraph::new("callee", callee_start.clone());
        callee_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(callee_x)],
                Some(callee_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let callee: GraphRef = Rc::new(RefCell::new(callee_graph));
        translator.graphs.borrow_mut().push(callee.clone());

        let mut inliner = fixture_inliner(&translator, host.clone(), InlinerKind::OneShot);

        // Build a synthetic block with a direct_call as the trailing
        // op, plus c_last_exception exitswitch.
        let v = Variable::named("v");
        let block = Block::shared(vec![Hlvalue::Variable(v.clone())]);
        let funcobj = make_func_constant(&callee, "callee");
        block.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![
                funcobj,
                Hlvalue::Constant(Constant::new(ConstValue::Int(0))),
            ],
            Hlvalue::Variable(Variable::named("r")),
        ));
        let last_exc_atom = Atom {
            name: LAST_EXCEPTION.name.clone(),
        };
        block.borrow_mut().exitswitch = Some(Hlvalue::Constant(Constant::new(ConstValue::Atom(
            last_exc_atom,
        ))));
        block.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(v)],
                Some(host.borrow().returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        // index_operation = 0 — only op in the block. The op is the
        // raising op because `block.canraise() && block.operations[-1] is op`.
        let result = inliner.inline_once(&block, 0);
        assert!(
            matches!(&result, Err(e) if e.0.contains("Slice 4d gap")),
            "expected Slice-4d-gap CannotInline, got {result:?}",
        );
    }

    #[test]
    fn inline_once_inlines_single_call_no_exceptions() {
        // Build a host graph that calls a small callee. Verify the
        // host's call op is replaced by the callee body and the
        // result is wired through the post-call block. This
        // exercises the full do_inline path without exception
        // handling.
        //
        // Callee `f(x): return x + x` (one int_add op, returnblock).
        // Host `g(): r = f(7); return r`.
        use crate::flowspace::model::{ConstValue, Constant};
        let translator = fixture_translator();

        // ---- Callee ----
        let x = Variable::named("x");
        let f_start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let f_graph = FunctionGraph::new("f", f_start.clone());
        let f_r = Variable::named("f_r");
        f_start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(f_r.clone()),
        ));
        f_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(f_r)],
                Some(f_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let f: GraphRef = Rc::new(RefCell::new(f_graph));

        // The callee graph needs to live somewhere reachable by
        // get_graph_for_call. Put it in the translator's graph
        // list so the funcobj-→graph lookup succeeds.
        translator.graphs.borrow_mut().push(f.clone());

        // Build the funcobj Constant pointing at f.
        let f_funcobj_const = make_func_constant(&f, "f");

        // ---- Host ----
        let g_start = Block::shared(vec![]);
        let g_graph = FunctionGraph::new("g", g_start.clone());
        let g_r = Variable::named("g_r");
        let seven = Hlvalue::Constant(Constant::new(ConstValue::Int(7)));
        g_start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            vec![f_funcobj_const, seven],
            Hlvalue::Variable(g_r.clone()),
        ));
        g_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(g_r)],
                Some(g_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let g: GraphRef = Rc::new(RefCell::new(g_graph));

        // Construct the inliner targeting f for inlining into g.
        let raise = RaiseAnalyzer::new(&translator);
        let mut inliner = BaseInliner::new(
            &translator,
            g.clone(),
            InlinerKind::Inliner(InlineFuncTarget::Graph(f.clone())),
            raise,
            false,
            false,
            None,
            true,
        );

        // Inline once. After this, the host's startblock should no
        // longer contain a direct_call to f.
        inliner
            .inline_once(&g_start, 0)
            .expect("non-guarded inline should succeed");

        // Walk all blocks reachable from the host start; assert
        // none of them retain a direct_call op.
        let blocks = g.borrow().iterblocks();
        let any_direct_call = blocks.iter().any(|b| {
            b.borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call")
        });
        assert!(!any_direct_call, "direct_call should have been inlined out");
    }

    // ----- Slice 4c: public entry points -----

    /// Helper: build the trivial host graph `g(): r = f(...); return r`
    /// where the call's funcobj points at `callee`. Returns the host
    /// `GraphRef` plus its startblock (which carries the `direct_call`
    /// op at index 0).
    fn host_calling(
        callee: &GraphRef,
        callee_name: &str,
        extra_arg: Option<Hlvalue>,
    ) -> (GraphRef, crate::flowspace::model::BlockRef) {
        let g_start = Block::shared(vec![]);
        let g_graph = FunctionGraph::new("g", g_start.clone());
        let g_r = Variable::named("g_r");
        let mut args = vec![make_func_constant(callee, callee_name)];
        if let Some(a) = extra_arg {
            args.push(a);
        }
        g_start.borrow_mut().operations.push(SpaceOperation::new(
            "direct_call",
            args,
            Hlvalue::Variable(g_r.clone()),
        ));
        g_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(g_r)],
                Some(g_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        (Rc::new(RefCell::new(g_graph)), g_start)
    }

    #[test]
    fn new_inliner_seeds_block_to_index_for_matching_callee() {
        let translator = fixture_translator();
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        let (_g, _g_start) = host_calling(&f, "f", None);
        let host = host_calling(&f, "f", None).0;
        let raise = RaiseAnalyzer::new(&translator);
        let inliner = BaseInliner::new_inliner(
            &translator,
            host,
            InlineFuncTarget::Graph(f.clone()),
            raise,
            false,
            false,
            None,
            true,
        );
        // One block carrying one direct_call → one block_to_index entry
        // with one (op_index → callee) tuple.
        assert_eq!(inliner.block_to_index.len(), 1);
        let (_block, dict) = inliner.block_to_index.values().next().unwrap();
        assert_eq!(dict.len(), 1);
        let callee = dict.values().next().unwrap();
        assert!(Rc::ptr_eq(callee, &f));
    }

    #[test]
    fn new_inliner_seeds_zero_when_callee_does_not_match() {
        let translator = fixture_translator();
        let f = int_add_graph("f");
        let h = int_add_graph("h");
        translator.graphs.borrow_mut().push(f.clone());
        translator.graphs.borrow_mut().push(h.clone());
        // Host calls f, but matcher targets the unrelated `h`.
        let (host, _) = host_calling(&f, "f", None);
        let raise = RaiseAnalyzer::new(&translator);
        let inliner = BaseInliner::new_inliner(
            &translator,
            host,
            InlineFuncTarget::Graph(h.clone()),
            raise,
            false,
            false,
            None,
            true,
        );
        assert_eq!(inliner.block_to_index.len(), 0);
    }

    #[test]
    fn inline_function_returns_count_and_clears_direct_call() {
        let translator = fixture_translator();
        // Callee f(x): return x + x
        let x = Variable::named("x");
        let f_start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let f_graph = FunctionGraph::new("f", f_start.clone());
        let f_r = Variable::named("f_r");
        f_start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(f_r.clone()),
        ));
        f_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(f_r)],
                Some(f_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let f: GraphRef = Rc::new(RefCell::new(f_graph));
        translator.graphs.borrow_mut().push(f.clone());

        let seven = Hlvalue::Constant(Constant::new(ConstValue::Int(7)));
        let (g, _) = host_calling(&f, "f", Some(seven));
        let raise = RaiseAnalyzer::new(&translator);
        let count = inline_function(
            &translator,
            InlineFuncTarget::Graph(f.clone()),
            g.clone(),
            (),
            raise,
            None,
            true,
        )
        .expect("inline_function returns Ok");
        assert_eq!(count, 1, "exactly one direct_call should be inlined");
        let any_dc = g.borrow().iterblocks().iter().any(|b| {
            b.borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call")
        });
        assert!(!any_dc, "post-inline graph must have no direct_call");
    }

    #[test]
    fn simple_inline_function_no_callees_returns_zero() {
        // `simple_inline_function` wraps `inline_function` with the
        // upstream defaults (`raise_analyzer=RaiseAnalyzer(translator)`,
        // `lltype_to_classdef={}`, `cleanup=True`). The host graph
        // contains no `direct_call`s into `f`, so the inliner runs
        // its scan-and-cleanup pipeline and returns 0.
        let translator = fixture_translator();
        let g = empty_graph("g");
        let f = empty_graph("f");
        let count = simple_inline_function(&translator, InlineFuncTarget::Graph(f), g)
            .expect("non-exception-guarded inline must succeed");
        assert_eq!(count, 0);
    }

    /// Helper: build a `Hlvalue::Constant` carrying an `_func`
    /// `_ptr` whose `graph` slot points to the supplied callee.
    /// Routes through upstream `lltype.functionptr` for parity.
    fn make_func_constant(graph: &GraphRef, name: &str) -> Hlvalue {
        use crate::flowspace::model::GraphKey;
        use crate::translator::rtyper::lltypesystem::lltype::{
            FuncType, LowLevelType, functionptr,
        };
        let func_type = FuncType {
            args: vec![LowLevelType::Signed],
            result: LowLevelType::Signed,
        };
        let key = GraphKey::of(graph).as_usize();
        let ptr = functionptr(func_type, name, Some(key), Some(name.to_string()));
        Hlvalue::Constant(Constant::new(ConstValue::LLPtr(Box::new(ptr))))
    }

    // ----- Slice 5a: instrument_inline_candidates -----

    #[test]
    fn instrument_inline_candidates_empty_graphs_returns_zero() {
        let translator = fixture_translator();
        let n = instrument_inline_candidates(&[], 100.0, &translator);
        assert_eq!(n, 0);
    }

    #[test]
    fn instrument_inline_candidates_no_direct_call_returns_zero() {
        let translator = fixture_translator();
        let g = int_add_graph("g");
        let n = instrument_inline_candidates(&[g], 100.0, &translator);
        assert_eq!(n, 0);
    }

    #[test]
    fn instrument_inline_candidates_inserts_count_op_before_direct_call() {
        let translator = fixture_translator();
        // Tiny callee — static_instruction_count == 1.0 ≤ threshold.
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        let (host, host_start) = host_calling(&f, "f", None);
        let n = instrument_inline_candidates(&[host.clone()], 100.0, &translator);
        assert_eq!(n, 1);
        // The startblock should now hold [instrument_count, direct_call].
        let ops = host_start.borrow().operations.clone();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0].opname, "instrument_count");
        assert_eq!(ops[1].opname, "direct_call");
        // Tag is the UniStr "inline" with concretetype Void.
        let Hlvalue::Constant(tag_c) = &ops[0].args[0] else {
            panic!("tag should be a Constant");
        };
        assert!(matches!(&tag_c.value, ConstValue::UniStr(s) if s == "inline"));
        let Hlvalue::Constant(label_c) = &ops[0].args[1] else {
            panic!("label should be a Constant");
        };
        assert!(matches!(label_c.value, ConstValue::Int(0)));
        let _ = host;
    }

    #[test]
    fn instrument_inline_candidates_skips_oversized_callee() {
        let translator = fixture_translator();
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        let (host, host_start) = host_calling(&f, "f", None);
        // Threshold 0.0 < static_instruction_count(f) — never matches.
        let n = instrument_inline_candidates(&[host], 0.0, &translator);
        assert_eq!(n, 0);
        let ops = host_start.borrow().operations.clone();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].opname, "direct_call");
    }

    #[test]
    fn instrument_inline_candidates_assigns_increasing_labels() {
        let translator = fixture_translator();
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        // Two host graphs, each calling f once.
        let (host_a, start_a) = host_calling(&f, "f", None);
        let (host_b, start_b) = host_calling(&f, "f", None);
        let n = instrument_inline_candidates(&[host_a.clone(), host_b.clone()], 100.0, &translator);
        assert_eq!(n, 2);
        let ops_a = start_a.borrow().operations.clone();
        let ops_b = start_b.borrow().operations.clone();
        let label_a = match &ops_a[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(v) => *v,
                _ => panic!(),
            },
            _ => panic!(),
        };
        let label_b = match &ops_b[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(v) => *v,
                _ => panic!(),
            },
            _ => panic!(),
        };
        // Labels should be 0 and 1 (monotonic, in iteration order).
        let mut labels = [label_a, label_b];
        labels.sort();
        assert_eq!(labels, [0, 1]);
    }

    // ----- Slice 5b: auto_inlining -----

    /// Test heuristic that always reports a low fixed weight so the
    /// `weight >= threshold` early-exit doesn't fire. Returns
    /// `(0.5, true)` for every graph.
    fn always_low_weight_heuristic(_g: &GraphRef) -> (f64, bool) {
        (0.5, true)
    }

    /// Test heuristic that reports a huge weight to force the
    /// `weight >= threshold` exit and exercise the rebuild path
    /// when no graph remains to revalidate.
    fn always_huge_weight_heuristic(_g: &GraphRef) -> (f64, bool) {
        (1.0e10, true)
    }

    #[test]
    fn auto_inlining_empty_callgraph_returns_zero() {
        let translator = fixture_translator();
        let count = auto_inlining(
            &translator,
            100.0,
            Some(vec![]),
            always_low_weight_heuristic,
            None,
        )
        .expect("empty callgraph");
        assert_eq!(count, 0);
    }

    #[test]
    #[should_panic(expected = "threshold must not be 1.0")]
    fn auto_inlining_threshold_one_panics() {
        let translator = fixture_translator();
        let _ = auto_inlining(
            &translator,
            1.0,
            Some(vec![]),
            always_low_weight_heuristic,
            None,
        );
    }

    #[test]
    fn auto_inlining_inlines_callee_into_single_parent() {
        let translator = fixture_translator();
        // Callee f(x): return x + x (one int_add op).
        let x = Variable::named("x");
        let f_start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let f_graph = FunctionGraph::new("f", f_start.clone());
        let f_r = Variable::named("f_r");
        f_start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(f_r.clone()),
        ));
        f_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(f_r)],
                Some(f_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let f: GraphRef = Rc::new(RefCell::new(f_graph));
        translator.graphs.borrow_mut().push(f.clone());

        let seven = Hlvalue::Constant(Constant::new(ConstValue::Int(7)));
        let (g, _) = host_calling(&f, "f", Some(seven));
        translator.graphs.borrow_mut().push(g.clone());

        let count = auto_inlining(
            &translator,
            100.0,
            Some(vec![(g.clone(), f.clone())]),
            always_low_weight_heuristic,
            None,
        )
        .expect("auto_inlining succeeds");
        assert_eq!(count, 1, "single (g, f) edge should produce one inline");
        let any_dc = g.borrow().iterblocks().iter().any(|b| {
            b.borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call")
        });
        assert!(!any_dc, "direct_call should be inlined out");
    }

    #[test]
    fn auto_inlining_threshold_skips_oversized_callee() {
        let translator = fixture_translator();
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        let (g, _) = host_calling(&f, "f", None);
        translator.graphs.borrow_mut().push(g.clone());
        // Threshold 100 < heuristic weight 1e10 → never fires.
        let count = auto_inlining(
            &translator,
            100.0,
            Some(vec![(g.clone(), f.clone())]),
            always_huge_weight_heuristic,
            None,
        )
        .expect("auto_inlining returns Ok even when no inlines fire");
        assert_eq!(count, 0);
        let any_dc = g.borrow().iterblocks().iter().any(|b| {
            b.borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call")
        });
        assert!(
            any_dc,
            "direct_call must be left intact when over threshold"
        );
    }

    // ----- Slice 5c: auto_inline_graphs -----

    #[test]
    fn auto_inline_graphs_empty_returns_zero() {
        let translator = fixture_translator();
        let count = auto_inline_graphs(
            &translator,
            &[],
            100.0,
            always_low_weight_heuristic,
            None,
            false,
        )
        .expect("empty graphs");
        assert_eq!(count, 0);
    }

    #[test]
    fn auto_inline_graphs_inlines_callee_into_parent_via_static_callers() {
        // Build callee f and host g; both registered in
        // translator.graphs. Pass [g, f] as the input slice and let
        // `inlinable_static_callers` discover the (g, f) edge.
        let translator = fixture_translator();
        let x = Variable::named("x");
        let f_start = Block::shared(vec![Hlvalue::Variable(x.clone())]);
        let f_graph = FunctionGraph::new("f", f_start.clone());
        let f_r = Variable::named("f_r");
        f_start.borrow_mut().operations.push(SpaceOperation::new(
            "int_add",
            vec![Hlvalue::Variable(x.clone()), Hlvalue::Variable(x.clone())],
            Hlvalue::Variable(f_r.clone()),
        ));
        f_start.closeblock(vec![
            Link::new(
                vec![Hlvalue::Variable(f_r)],
                Some(f_graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        let f: GraphRef = Rc::new(RefCell::new(f_graph));
        translator.graphs.borrow_mut().push(f.clone());

        let seven = Hlvalue::Constant(Constant::new(ConstValue::Int(7)));
        let (g, _) = host_calling(&f, "f", Some(seven));
        translator.graphs.borrow_mut().push(g.clone());

        let count = auto_inline_graphs(
            &translator,
            &[g.clone(), f.clone()],
            100.0,
            always_low_weight_heuristic,
            None,
            false,
        )
        .expect("auto_inline_graphs succeeds");
        assert_eq!(count, 1);
        let any_dc = g.borrow().iterblocks().iter().any(|b| {
            b.borrow()
                .operations
                .iter()
                .any(|op| op.opname == "direct_call")
        });
        assert!(!any_dc, "direct_call should be inlined out");
    }

    #[test]
    fn auto_inline_graphs_inline_from_anywhere_runs_without_panic() {
        // `inline_graph_from_anywhere=true` switches to the
        // translator.graphs ok_to_call set. Pyre has no
        // exceptiontransformed filter today; this test just ensures
        // the alternate path inlines the (g, f) edge cleanly.
        // Pass an actual call arg to f so the inlined body has a
        // value to bind to f's startblock inputargs.
        let translator = fixture_translator();
        let f = int_add_graph("f");
        translator.graphs.borrow_mut().push(f.clone());
        let arg = Hlvalue::Constant(Constant::new(ConstValue::Int(0)));
        let (g, _) = host_calling(&f, "f", Some(arg));
        translator.graphs.borrow_mut().push(g.clone());

        let _ = auto_inline_graphs(
            &translator,
            &[g.clone()],
            100.0,
            always_low_weight_heuristic,
            None,
            true,
        )
        .expect("inline_graph_from_anywhere path must not crash");
    }
}
