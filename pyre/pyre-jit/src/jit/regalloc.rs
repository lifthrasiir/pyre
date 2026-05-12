//! Post-pass register allocation for pyre's per-CodeObject SSARepr.
//!
//! Mirrors the two-file split in RPython:
//!
//!   * `rpython/jit/codewriter/regalloc.py:6-8`
//!     `perform_register_allocation(graph, kind)` — thin wrapper around
//!     `tool.algo.regalloc.perform_register_allocation`. Pyre's analog is
//!     `perform_register_allocation` below.
//!   * `rpython/tool/algo/regalloc.py:8-15`
//!     ```python
//!     regalloc = RegAllocator(graph, consider_var, ListOfKind)
//!     regalloc.make_dependencies()    # interference graph
//!     regalloc.coalesce_variables()   # union-find on jump edges
//!     regalloc.find_node_coloring()   # chordal coloring
//!     ```
//!     Pyre's analog is `RegAllocator` + the three private methods of
//!     the same name.
//!   * `rpython/jit/codewriter/flatten.py:88-100` `enforce_input_args` —
//!     after coloring, `swapcolors` rotates inputarg colors into
//!     `0..n-1`. Pyre's analog is `enforce_input_args` below.
//!   * `rpython/jit/codewriter/codewriter.py:62-67` —
//!     `num_regs[kind] = max(coloring)+1` per kind, packed into the
//!     `JitCode`. Pyre's analog is `RegAllocator::num_colors` plus the
//!     `AllocationResult.num_regs` field.
//!
//! Architecture difference (PRE-EXISTING-ADAPTATION): RPython's
//! `RegAllocator` consumes a `FunctionGraph` (block + link.args
//! structure). Pyre's input is a CPython `CodeObject` translated into
//! a flat `SSARepr` by the dispatch loop, so `make_dependencies` works
//! over the populated `SSARepr` via a backward live-set walk and
//! `coalesce_variables` operates on `move_X` instructions (the
//! SSARepr-level remnant of jump-edge `link.args ↔ inputargs`
//! pairings). The chordal coloring algorithm itself is shared with
//! `majit-translate`'s flow-graph regalloc through
//! `majit_translate::regalloc::DependencyGraph::find_node_coloring`
//! (line-by-line port of `rpython/tool/algo/color.py:31-85`).

use std::collections::{HashMap, HashSet};

use majit_translate::jit_codewriter::flatten::reorder_renaming_list;
use majit_translate::model::ValueId;
use majit_translate::regalloc::DependencyGraph;
use majit_translate::tool::algo::unionfind::UnionFind;

use super::flatten::{DescrOperand, Insn, Kind, Operand, Register, SSARepr, TLabel};
use super::flow::{ExitSwitch, ExitSwitchElement, FlowValue, FunctionGraph as FlowGraph, Variable};

#[inline]
fn variable_value_id(v: Variable) -> ValueId {
    ValueId(v.id.0 as usize)
}

#[derive(Debug, Clone)]
pub(super) struct GraphAllocationResult {
    pub coloring: HashMap<super::flow::VariableId, u16>,
    pub num_colors: u16,
}

/// Field names follow `rpython/tool/algo/regalloc.py:47-49` exactly —
/// `_depgraph`, `_unionfind`, `_coloring`.  The union-find is the real
/// `tool/algo/unionfind.py UnionFind` port (`()` info, matching upstream
/// `info_factory=None`).
struct FlowGraphRegAllocator {
    _depgraph: DependencyGraph,
    _unionfind: UnionFind<super::flow::VariableId, ()>,
    _coloring: HashMap<super::flow::VariableId, u16>,
}

impl FlowGraphRegAllocator {
    fn new() -> Self {
        Self {
            _depgraph: DependencyGraph::new(),
            _unionfind: UnionFind::new(|_| ()),
            _coloring: HashMap::new(),
        }
    }

    fn make_dependencies(&mut self, graph: &FlowGraph, kind: Kind) {
        for block in graph.iterblocks() {
            let block_borrow = block.borrow();
            let mut die_at: HashMap<super::flow::VariableId, usize> = HashMap::new();
            for arg in &block_borrow.inputargs {
                if let Some(v) = arg.as_variable() {
                    if v.kind == Some(kind) {
                        die_at.insert(v.id, 0);
                    }
                }
            }
            for (i, op) in block_borrow.operations.iter().enumerate() {
                for arg in &op.args {
                    for v in arg.variables() {
                        if v.kind == Some(kind) {
                            die_at.insert(v.id, i);
                        }
                    }
                }
                if let Some(v) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if v.kind == Some(kind) {
                        die_at.insert(v.id, i + 1);
                    }
                }
            }
            match &block_borrow.exitswitch {
                Some(ExitSwitch::Value(value)) => {
                    if let Some(v) = value.as_variable() {
                        die_at.remove(&v.id);
                    }
                }
                Some(ExitSwitch::Tuple(values)) => {
                    for value in values {
                        if let ExitSwitchElement::Value(value) = value {
                            if let Some(v) = value.as_variable() {
                                die_at.remove(&v.id);
                            }
                        }
                    }
                }
                None => {}
            }
            for link in &block_borrow.exits {
                for arg in &link.borrow().args {
                    if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                        die_at.remove(&v.id);
                    }
                }
            }
            let mut die_list: Vec<(usize, super::flow::VariableId)> =
                die_at.into_iter().map(|(var, time)| (time, var)).collect();
            die_list.sort_by_key(|(time, _)| *time);
            die_list.push((usize::MAX, super::flow::VariableId(u32::MAX)));

            let livevars: Vec<Variable> = block_borrow
                .inputargs
                .iter()
                .filter_map(FlowValue::as_variable)
                .filter(|v| v.kind == Some(kind))
                .collect();
            for (i, &v) in livevars.iter().enumerate() {
                self._depgraph.add_node(variable_value_id(v));
                for j in 0..i {
                    self._depgraph
                        .add_edge(variable_value_id(livevars[j]), variable_value_id(v));
                }
            }
            // upstream: `livevars = set(livevars)` — shadow the list
            // with the set rather than renaming to `alive`.
            let mut livevars: HashSet<super::flow::VariableId> =
                livevars.into_iter().map(|v| v.id).collect();
            let mut die_index = 0;
            for (i, op) in block_borrow.operations.iter().enumerate() {
                while die_list[die_index].0 == i {
                    livevars.remove(&die_list[die_index].1);
                    die_index += 1;
                }
                if let Some(result) = op.result.as_ref().and_then(FlowValue::as_variable) {
                    if result.kind == Some(kind) {
                        self._depgraph.add_node(variable_value_id(result));
                        // upstream (`regalloc.py:73`): add an edge from
                        // every live var to `result`.  `result` is added
                        // to `livevars` only *after* the loop, so no
                        // self-edge guard is needed.
                        for &v in &livevars {
                            self._depgraph
                                .add_edge(ValueId(v.0 as usize), variable_value_id(result));
                        }
                        livevars.insert(result.id);
                    }
                }
            }
        }
    }

    fn coalesce_variables(&mut self, graph: &FlowGraph, kind: Kind) {
        let mut pendingblocks = graph.iterblocks();
        while let Some(block) = pendingblocks.pop() {
            // Match `rpython/tool/algo/regalloc.py:82-86`: walk from the
            // end of the graph first because resume/blackhole execution
            // typically restarts in the middle rather than at the entry.
            let block_borrow = block.borrow();
            for link in &block_borrow.exits {
                let link_borrow = link.borrow();
                if let Some(v) = link_borrow.last_exception {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(variable_value_id(v));
                    }
                }
                if let Some(v) = link_borrow.last_exc_value {
                    if v.kind == Some(kind) {
                        self._depgraph.add_node(variable_value_id(v));
                    }
                }
                let Some(target) = link_borrow.target.clone() else {
                    continue;
                };
                let target_borrow = target.borrow();
                for (arg, target_input) in
                    link_borrow.args.iter().zip(target_borrow.inputargs.iter())
                {
                    let Some(src) = arg.as_ref().and_then(FlowValue::as_variable) else {
                        continue;
                    };
                    let Some(dst) = target_input.as_variable() else {
                        continue;
                    };
                    self.try_coalesce(src, dst, kind);
                }
            }
        }
    }

    fn try_coalesce(&mut self, v: Variable, w: Variable, kind: Kind) {
        if v.kind != Some(kind) || w.kind != Some(kind) {
            return;
        }
        let v0 = self._unionfind.find_rep(v.id);
        let w0 = self._unionfind.find_rep(w.id);
        if v0 == w0 {
            return;
        }
        if self
            ._depgraph
            .has_edge(ValueId(v0.0 as usize), ValueId(w0.0 as usize))
        {
            return;
        }
        let (_, rep) = self._unionfind.union(v0, w0);
        // upstream (`regalloc.py:108`): `assert uf.find_rep(v0) is
        // uf.find_rep(w0) is rep`.
        debug_assert_eq!(self._unionfind.find_rep(v0), rep);
        debug_assert_eq!(self._unionfind.find_rep(w0), rep);
        if rep == v0 {
            self._depgraph
                .coalesce(ValueId(w0.0 as usize), ValueId(v0.0 as usize));
        } else {
            // upstream (`regalloc.py:112`): `assert rep is w0`.
            debug_assert_eq!(rep, w0);
            self._depgraph
                .coalesce(ValueId(v0.0 as usize), ValueId(w0.0 as usize));
        }
    }

    fn find_node_coloring(&mut self) {
        self._coloring = self
            ._depgraph
            .find_node_coloring()
            .into_iter()
            .map(|(value, color)| (super::flow::VariableId(value.0 as u32), color as u16))
            .collect();
    }

    fn getcolor(&mut self, v: Variable) -> Option<u16> {
        let rep = self._unionfind.find_rep(v.id);
        self._coloring.get(&rep).copied()
    }

    fn find_num_colors(&self) -> u16 {
        self._coloring.values().copied().max().map_or(0, |m| m + 1)
    }
}

/// `rpython/tool/algo/regalloc.py:8-15 perform_register_allocation`.
///
/// Upstream signature is `(graph, consider_var, ListOfKind=())`.  Pyre
/// bakes `consider_var` into the single `kind` filter because `Kind` is a
/// closed enum (Int/Ref/Float) whereas upstream's `consider_var` is an
/// open predicate over lltype concreteness.  `ListOfKind` is not a
/// parameter because pyre has exactly one such class (`FlowListOfKind`).
///
/// The `_graph_` prefix disambiguates from the sibling
/// `perform_register_allocation` SSARepr scanner further down in this
/// module.  **Currently not invoked from the production regalloc
/// path** — it is a tested algorithm port kept to cover the CFG-level
/// coalesce rules of `regalloc.py:79-112 coalesce_variables`, but pyre
/// cannot drop the SSARepr scanner and activate this one the way
/// upstream does, for two orthogonal reasons:
///
///   1. Pyre's `getoutputargs()` builds `link.args` by positional walk
///      over `targetstate.mergeable()` (`codewriter.rs::FrameState::
///      getoutputargs`), so every pair this function would produce at
///      the CFG level is trivially `(slot, slot)` — a structural no-op
///      relative to what SSARepr-level coalescing already sees.
///   2. The SSARepr scanner handles **intra-block** `*_copy`
///      sequences (emitted by `emit_ref_copy!`/`emit_int_copy!` inline
///      for STORE_FAST-LOAD_FAST fusions) that have no Link-level
///      representation and therefore cannot be coalesced at the CFG
///      level.
///
/// Keeping the helper documents that the RPython algorithm is ported
/// correctly for the CFG shape pyre does maintain; the unit tests
/// exercise exactly that surface.  It is not dead code, but it is also
/// not a drop-in replacement for the scanner.
pub(super) fn perform_graph_register_allocation(
    graph: &FlowGraph,
    kind: Kind,
) -> GraphAllocationResult {
    let mut allocator = FlowGraphRegAllocator::new();
    allocator.make_dependencies(graph, kind);
    allocator.coalesce_variables(graph, kind);
    allocator.find_node_coloring();

    let mut coloring = HashMap::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for variable in block_borrow.getvariables() {
            if variable.kind == Some(kind) {
                if let Some(color) = allocator.getcolor(variable) {
                    coloring.insert(variable.id, color);
                }
            }
        }
        for link in &block_borrow.exits {
            let link_borrow = link.borrow();
            if let Some(v) = link_borrow.last_exception {
                if v.kind == Some(kind) {
                    if let Some(color) = allocator.getcolor(v) {
                        coloring.insert(v.id, color);
                    }
                }
            }
            if let Some(v) = link_borrow.last_exc_value {
                if v.kind == Some(kind) {
                    if let Some(color) = allocator.getcolor(v) {
                        coloring.insert(v.id, color);
                    }
                }
            }
            for arg in &link_borrow.args {
                if let Some(v) = arg.as_ref().and_then(FlowValue::as_variable) {
                    if v.kind == Some(kind) {
                        if let Some(color) = allocator.getcolor(v) {
                            coloring.insert(v.id, color);
                        }
                    }
                }
            }
        }
    }

    GraphAllocationResult {
        coloring,
        num_colors: allocator.find_num_colors(),
    }
}

/// Run `perform_graph_register_allocation` once per `Kind` and collect
/// the per-kind `GraphAllocationResult`s into a single map, mirroring
/// `rpython/jit/codewriter/codewriter.py:44-46`:
///
/// ```python
/// regallocs = {}
/// for kind in KINDS:
///     regallocs[kind] = perform_register_allocation(graph, kind)
/// ```
///
/// The resulting `HashMap<Kind, GraphAllocationResult>` is the direct
/// analog of upstream's `regallocs` dict and is the input shape that a
/// future `flatten_graph(graph, regallocs, ...)` driver will read.
/// (Phase 1 of the production-wiring plan tracked as Task #224.)
pub(super) fn perform_graph_register_allocation_all_kinds(
    graph: &FlowGraph,
) -> HashMap<Kind, GraphAllocationResult> {
    let mut regallocs = HashMap::new();
    for &kind in &Kind::ALL {
        regallocs.insert(kind, perform_graph_register_allocation(graph, kind));
    }
    regallocs
}

/// Mirrors `rpython/jit/codewriter/flatten.py:88-100 enforce_input_args`.
///
/// Walks the startblock's inputargs in source order; for each inputarg
/// of kind `K` whose current color in `regallocs[K]` does not equal
/// the next "real" color for that kind (`0, 1, 2, ...` per appearance),
/// invokes `swapcolors(realcol, curcol)` over the entire
/// `coloring` map (`rpython/tool/algo/regalloc.py:138-143`).
///
/// Upstream `flatten_graph` runs this immediately after
/// `regallocs[kind] = perform_register_allocation(graph, kind)` and
/// before `generate_ssa_form` walks links, so every downstream
/// observer sees the post-swap coloring. Pyre's pipeline currently
/// bypasses `flatten_graph`; callers that want
/// `count_link_renamings_per_kind` (or any future graph-driven
/// `*_copy` emitter) to match what `flatten_graph` would produce must
/// run this simulation explicitly first.
///
/// Tracks the production-wiring follow-up in Task #214.
pub(super) fn enforce_input_args_simulation(
    graph: &FlowGraph,
    regallocs: &mut HashMap<Kind, GraphAllocationResult>,
) {
    let inputargs = graph.startblock.borrow().inputargs.clone();
    let mut numkinds: HashMap<Kind, u16> = HashMap::new();
    for arg in &inputargs {
        let Some(v) = arg.as_variable() else { continue };
        let Some(kind) = v.kind else { continue };
        let realcol = *numkinds.get(&kind).unwrap_or(&0);
        numkinds.insert(kind, realcol + 1);
        let alloc = regallocs
            .get_mut(&kind)
            .expect("regallocs must contain an entry for every Kind seen in inputargs");
        // Inputarg never appeared in any instruction — coloring
        // skipped it. Swap is unnecessary because no register refers
        // to its color (mirrors the SSARepr-side `enforce_input_args`
        // shortcut).
        let Some(&curcol) = alloc.coloring.get(&v.id) else {
            continue;
        };
        if curcol == realcol {
            continue;
        }
        assert!(
            curcol > realcol,
            "enforce_input_args_simulation: inputarg color {} must be >= realcol {} \
             (regalloc.py invariant)",
            curcol,
            realcol,
        );
        for color in alloc.coloring.values_mut() {
            if *color == curcol {
                *color = realcol;
            } else if *color == realcol {
                *color = curcol;
            }
        }
    }
}

/// Count the `*_copy` / `*_push` / `*_pop` ops that `insert_renamings`
/// (`rpython/jit/codewriter/flatten.py:306-334`) would emit per `Kind`
/// across every Link in `graph`, using `regallocs[kind].coloring` to
/// project Variables to register colors.
///
/// Companion to `perform_graph_register_allocation_all_kinds`: with
/// `Block.inputargs` populated at every walker block-creation site
/// (Phase 4 Session 14), the `link.args ↔ target.inputargs`
/// correspondence is well-defined for every Link in the graph, so the
/// renaming count matches the post-coalesce shape `flatten_graph` will
/// produce when it replaces the inline `*_copy` walker emission
/// (Task #227 Phase 4 endgame).
///
/// Skips the same-color pairs that upstream `insert_renamings` filters
/// at `flatten.py:317` (`if src == RenameOperand::Register(dst):
/// continue`) and the `last_exception` / `last_exc_value` pairs that
/// `generate_last_exc` covers separately.
///
/// Observation-only — no SSARepr or graph mutation.
pub(super) fn count_link_renamings_per_kind(
    graph: &FlowGraph,
    regallocs: &HashMap<Kind, GraphAllocationResult>,
) -> HashMap<Kind, usize> {
    // PRE-EXISTING-ADAPTATION: the `regallocs` passed in come from
    // `perform_graph_register_allocation_all_kinds`, which runs the
    // chordal coloring directly. Upstream `flatten_graph` follows
    // that with `enforce_input_args` (`flatten.py:88-100`), swapping
    // inputarg colors so the startblock's inputargs land at colors
    // `0, 1, 2, …` per kind before `generate_ssa_form` walks links.
    // Callers that want this probe's output to match
    // `flatten_graph`'s post-swap reality should invoke
    // `enforce_input_args_simulation(graph, &mut regallocs)` first;
    // otherwise the per-link step count is a lower bound, since the
    // swap can shift inputarg colors and re-introduce previously
    // coalesced renamings. Production wiring is Task #214.
    let mut counts: HashMap<Kind, usize> = HashMap::new();
    for block in graph.iterblocks() {
        let block_borrow = block.borrow();
        for link in &block_borrow.exits {
            let link_borrow = link.borrow();
            let Some(target) = link_borrow.target.clone() else {
                continue;
            };
            let target_inputargs = target.borrow().inputargs.clone();
            // `model.py:114-116 Link.__init__` enforces
            // `len(args) == len(target.inputargs)` via plain `assert`
            // (always-on in Python, only stripped under `-O`).
            // Pyre's `Link::new_mergeable` (flow.rs:797-800) carries
            // a release-build `assert_eq!` at construction; mirror
            // that strength here so a future link-builder that
            // bypasses the constructor invariant fails identically
            // in release.  Silently truncating via `.zip()` would
            // under-report renaming steps.
            assert_eq!(
                link_borrow.args.len(),
                target_inputargs.len(),
                "count_link_renamings_per_kind: Link.__init__ arity invariant violated \
                 (link.args={} target.inputargs={})",
                link_borrow.args.len(),
                target_inputargs.len(),
            );
            let last_exception = link_borrow.last_exception;
            let last_exc_value = link_borrow.last_exc_value;
            let mut per_kind: HashMap<Kind, (Vec<Register>, Vec<Register>)> = HashMap::new();
            for (arg, inputarg) in link_borrow.args.iter().zip(target_inputargs.iter()) {
                let Some(src_value) = arg.as_ref() else {
                    continue;
                };
                let Some(dst_var) = inputarg.as_variable() else {
                    continue;
                };
                let Some(kind) = dst_var.kind else {
                    continue;
                };
                let Some(regalloc) = regallocs.get(&kind) else {
                    continue;
                };
                let Some(&dst_color) = regalloc.coloring.get(&dst_var.id) else {
                    continue;
                };

                // `flatten.py:306-334 insert_renamings` calls
                // `getcolor(v_out)` which (`flatten.py:382`) returns
                // the source variable's color when `v_out` is a
                // Variable, OR returns the constant itself when
                // `v_out` is a Constant.  Constant sources can never
                // equal a Register-typed `dst_color`, so they ALWAYS
                // generate a renaming step (`load const into dst`)
                // — count them as +1 step each, separate from the
                // Variable cycle-resolution path that
                // `reorder_renaming_list` handles.  The previous
                // version of this probe skipped Constant args
                // entirely, undercounting renamings whenever a Link
                // carried a Constant into a typed inputarg.
                if let Some(src_var) = src_value.as_variable() {
                    let src_var_opt = Some(src_var);
                    if src_var_opt == last_exception || src_var_opt == last_exc_value {
                        continue;
                    }
                    if src_var.kind != Some(kind) {
                        continue;
                    }
                    let Some(&src_color) = regalloc.coloring.get(&src_var.id) else {
                        continue;
                    };
                    if src_color == dst_color {
                        continue;
                    }
                    let (frm, to) = per_kind.entry(kind).or_default();
                    frm.push(Register::new(kind, src_color));
                    to.push(Register::new(kind, dst_color));
                } else if src_value.as_constant().is_some() {
                    // Constant src — always emits one renaming step.
                    *counts.entry(kind).or_insert(0) += 1;
                }
            }
            for (kind, (frm, to)) in per_kind {
                let steps = reorder_renaming_list(&frm, &to);
                *counts.entry(kind).or_insert(0) += steps.len();
            }
        }
    }
    counts
}

/// Count operations recorded in `graph.iterblocks().operations` per
/// opname. This is what `flatten_graph(graph, regallocs)` would walk
/// before emitting the SSARepr (`flatten.py:60-100 generate_ssa_form`
/// iterates `block.operations` per reachable block). Compared against
/// `count_ssa_ops_per_opname(ssarepr)` to quantify the gap between
/// pyre's parallel graph (sparse `record_graph_op` coverage today) and
/// the canonical inline-emit SSARepr the walker still produces.
///
/// As `record_graph_op` coverage expands one op family at a time
/// (Task #227 Phase 4 endgame), the graph-side per-opname count
/// converges toward the SSA-side count for each retired family. Once
/// they match for a family, that family's inline emit is safe to
/// drop in favour of `flatten_graph` emission.
pub(super) fn count_graph_ops_per_opname(graph: &FlowGraph) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for block in graph.iterblocks() {
        for op in &block.borrow().operations {
            *counts.entry(op.opname.clone()).or_insert(0) += 1;
        }
    }
    counts
}

/// Count `Insn::Op` opnames in `ssarepr.insns`. The inline-emit
/// counterpart to `count_graph_ops_per_opname`; together they bracket
/// the convergence target for `flatten_graph` adoption (Task #227).
/// Skips `Insn::Label`, `Insn::Unreachable`, and `Insn::PcAnchor`
/// (none of which are emitted by `GraphFlattener::emit_space_operation`).
pub(super) fn count_ssa_ops_per_opname(ssarepr: &SSARepr) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    for insn in &ssarepr.insns {
        if let Insn::Op { opname, .. } = insn {
            *counts.entry(opname.clone()).or_insert(0) += 1;
        }
    }
    counts
}

/// Count `*_copy` opnames in `ssarepr.insns` per `Kind`. Pyre's walker
/// emits these inline at `emit_ref_copy!` / `emit_load_const_i` /
/// `emit_pushvalue_ref!` / `emit_popvalue_ref!` etc. for stack/local
/// lifecycle (no upstream equivalent — RPython `flatten.py:306-334`
/// emits `*_copy` only via `insert_renamings` at link points). The two
/// counts are tracked separately so the walker→flatten transition
/// (Task #227) can be quantified before flipping the canonical SSARepr
/// source.
pub(super) fn count_ssa_copy_ops_per_kind(ssarepr: &SSARepr) -> HashMap<Kind, usize> {
    let mut counts: HashMap<Kind, usize> = HashMap::new();
    for insn in &ssarepr.insns {
        if let Insn::Op { opname, .. } = insn {
            for &kind in &Kind::ALL {
                let prefix = match kind {
                    Kind::Int => "int_copy",
                    Kind::Ref => "ref_copy",
                    Kind::Float => "float_copy",
                };
                if opname == prefix {
                    *counts.entry(kind).or_insert(0) += 1;
                    break;
                }
            }
        }
    }
    counts
}

/// External-input registers preserved across coloring.
///
/// RPython parity: `regalloc.py:54-60` adds pairwise interference
/// edges between every variable in `block.inputargs`, and
/// `flatten.py:88-100` enforces that those same inputargs land on
/// colors `0..n-1` via `swapcolors`. Pyre's analog is the set of
/// registers that arrive pre-populated:
///
/// - The `nlocals` low Ref registers (locals 0..nlocals) are the
///   trace-side analog of `block.inputargs`. Both `trace_opcode.rs`
///   (the LiveVars expansion) and the bytecode walker decode
///   `register_idx < nlocals` as "this register holds Python local
///   `register_idx`'s value", so the post-regalloc colors of these
///   registers must equal their pre-coloring indices. This is
///   guaranteed by `enforce_input_args` (not by interference
///   heuristics), in line with `flatten.py:88-100`.
/// - Portal red args (`frame_reg`, `ec_reg`) are pre-populated by
///   `BlackholeInterpreter::fill_portal_registers`
///   (blackhole.rs:1133-1140) at compile-time-fixed register slots
///   produced by `RegisterLayout::compute`; they get colors
///   `nlocals` and `nlocals+1` after `enforce_input_args` runs.
pub(super) struct ExternalInputs {
    pub portal_frame_reg: u16,
    pub portal_ec_reg: u16,
    pub portal_inputs: bool,
}

/// Result of `allocate_registers`.
///
/// `rename` carries the per-kind `(pre_index → post_index)` map
/// applied by `apply_rename`. `num_regs` carries the per-kind
/// `max(color)+1` value the assembler stores in `JitCode.num_regs_*`
/// (codewriter.py:62-67).
pub(super) struct AllocationResult {
    pub rename: HashMap<(Kind, u16), u16>,
    pub num_regs: HashMap<Kind, u16>,
}

/// Run register allocation on `ssarepr` and produce the rename map
/// plus per-kind `num_regs`.
///
/// `nlocals` is the number of CPython fast locals (`code.varnames.len()`).
///
/// `cfg_coalesce_pairs` is the output of
/// `codewriter::collect_link_slot_pairs` — `(source_slot,
/// target_slot)` pairs from CFG link boundaries, all of Ref kind
/// because every `FrameState.mergeable()` position in pyre holds a
/// Ref-kind Variable (locals, stack, last_exc pair).  See
/// `collect_link_slot_pairs` docstring + `regalloc.py:79-96`.
///
/// RPython parity: `codewriter.py:45-47, 62-67`.
pub(super) fn allocate_registers(
    ssarepr: &SSARepr,
    nlocals: usize,
    inputs: ExternalInputs,
    cfg_coalesce_pairs: &[(u16, u16)],
) -> AllocationResult {
    // codewriter.py:45-47 `for kind in KINDS:
    //   regallocs[kind] = perform_register_allocation(graph, kind)`
    let mut allocators: HashMap<Kind, RegAllocator> = HashMap::new();
    for &kind in &Kind::ALL {
        let mut external: Vec<u16> = Vec::new();
        if kind == Kind::Ref {
            for i in 0..nlocals as u16 {
                external.push(i);
            }
            // Phase 2.1c (plan staged-sauteeing-koala): stack slots are
            // no longer pinned to colors `nlocals..nlocals+max_stackdepth`.
            // `stack_slot_color_map` (PyJitCodeMetadata) records the
            // post-rename color so decoders / blackhole resume can
            // translate slot → color without assuming identity.
            if inputs.portal_inputs {
                if inputs.portal_frame_reg != u16::MAX {
                    external.push(inputs.portal_frame_reg);
                }
                if inputs.portal_ec_reg != u16::MAX {
                    external.push(inputs.portal_ec_reg);
                }
            }
        }
        // `cfg_coalesce_pairs` are CFG-level Variable pairs from Link
        // boundaries (regalloc.py:79-96).  All mergeable positions in
        // pyre's FrameState are Ref-kind, so pass them only to the
        // Ref allocator; Int / Float regalloc sees an empty slice.
        let cfg_pairs_for_kind: &[(u16, u16)] = if kind == Kind::Ref {
            cfg_coalesce_pairs
        } else {
            &[]
        };
        let alloc = perform_register_allocation(ssarepr, kind, &external, cfg_pairs_for_kind);
        allocators.insert(kind, alloc);
    }

    // flatten.py:88-100 `enforce_input_args` — rotate inputarg colors
    // into 0..n-1 via swapcolors so the trace-side `idx < nlocals`
    // decode is guaranteed by code rather than by an interference
    // heuristic.
    enforce_input_args(&mut allocators, nlocals, &inputs);

    // codewriter.py:62-67 `num_regs = {kind: max(coloring)+1 if coloring else 0}`.
    let mut rename: HashMap<(Kind, u16), u16> = HashMap::new();
    let mut num_regs: HashMap<Kind, u16> = HashMap::new();
    for (&kind, alloc) in allocators.iter() {
        for (&pre, &post) in alloc.coloring.iter() {
            if pre != post {
                rename.insert((kind, pre), post);
            }
        }
        num_regs.insert(kind, alloc.find_num_colors());
    }
    AllocationResult { rename, num_regs }
}

/// `flatten.py:88-100` `GraphFlattener.enforce_input_args`.
///
/// RPython:
/// ```python
/// def enforce_input_args(self):
///     inputargs = self.graph.startblock.inputargs
///     numkinds = {}
///     for v in inputargs:
///         kind = getkind(v.concretetype)
///         if kind == 'void':
///             continue
///         curcol = self.regallocs[kind].getcolor(v)
///         realcol = numkinds.get(kind, 0)
///         numkinds[kind] = realcol + 1
///         if curcol != realcol:
///             assert curcol > realcol
///             self.regallocs[kind].swapcolors(realcol, curcol)
/// ```
///
/// Pyre's `inputargs` are all of kind `Ref`: locals `0..nlocals` (the
/// trace-side Python-local mirror) followed by the portal red args
/// (frame, ec). Int and Float kinds have no inputargs — see
/// `ExternalInputs` docstring.
fn enforce_input_args(
    allocators: &mut HashMap<Kind, RegAllocator>,
    nlocals: usize,
    inputs: &ExternalInputs,
) {
    let alloc = allocators
        .get_mut(&Kind::Ref)
        .expect("Ref allocator must exist");
    let mut input_indices: Vec<u16> = (0..nlocals as u16).collect();
    // Phase 2.1c (plan staged-sauteeing-koala): stack slots no longer
    // rotated into fixed colors. The decoder consults
    // `stack_slot_color_map` to recover the post-rename color.
    if inputs.portal_inputs {
        if inputs.portal_frame_reg != u16::MAX {
            input_indices.push(inputs.portal_frame_reg);
        }
        if inputs.portal_ec_reg != u16::MAX {
            input_indices.push(inputs.portal_ec_reg);
        }
    }
    for (realcol, &v) in input_indices.iter().enumerate() {
        let realcol = realcol as u16;
        let curcol = match alloc.getcolor(v) {
            Some(c) => c,
            // Inputarg never appeared in any instruction — coloring
            // skipped it. swap is unnecessary because no register
            // refers to its color.
            None => continue,
        };
        if curcol != realcol {
            assert!(
                curcol > realcol,
                "enforce_input_args: inputarg color {} must be >= realcol {} (regalloc.py invariant)",
                curcol,
                realcol
            );
            alloc.swapcolors(realcol, curcol);
        }
    }
}

/// RPython `regalloc.py:6` `perform_register_allocation(graph, kind)`
/// + `tool/algo/regalloc.py:8-15`. Builds a `RegAllocator` and runs
/// the three-stage pipeline.
///
/// Dual coalesce sources:
///   1. `cfg_coalesce_pairs` — pre-computed `(source_slot,
///      target_slot)` pairs from `codewriter::collect_link_slot_pairs`.
///      These are the upstream `tool/algo/regalloc.py:79-96`
///      `link.args ↔ link.target.inputargs` pairs, projected from
///      Variables onto pyre's u16 register slots via the walker's
///      positional alignment (see `collect_link_slot_pairs`
///      docstring).  Trivially-equal pairs by `getoutputargs`
///      construction — their `try_coalesce` is a no-op when
///      `src_slot == dst_slot`, but the call preserves RPython's
///      exact iteration shape.
///   2. SSARepr `*_copy` scanner (PRE-EXISTING-ADAPTATION) — pyre's
///      walker emits intra-block `int_copy` / `ref_copy` /
///      `float_copy` ops for stack shuffling / STORE_FAST sequences
///      that have no CFG / Link representation.  The scanner unions
///      each copy's src and dst so the chordal coloring reuses one
///      color, turning the runtime copy into a no-op when
///      `src_color == dst_color`.  Upstream does not have this
///      SSARepr-level pass because `flatten.py:306-334`
///      `insert_renamings` places its copies post-coalesce, so
///      RPython's CFG-level `regalloc.py:79-96` is the sole coalesce
///      source.  In pyre both sources are orthogonal — the CFG
///      pairs cover cross-block link boundaries, the scanner covers
///      intra-block shuffles — so both run.
fn perform_register_allocation(
    ssarepr: &SSARepr,
    kind: Kind,
    external_inputs: &[u16],
    cfg_coalesce_pairs: &[(u16, u16)],
) -> RegAllocator {
    let mut alloc = RegAllocator::new();
    alloc.make_dependencies(ssarepr, kind, external_inputs);
    // regalloc.py:79-96 CFG-level coalesce: union link.args[i] with
    // link.target.inputargs[i] for every block exit.  Fed in from
    // `codewriter::collect_link_slot_pairs` (Ref kind only).
    for &(src_slot, dst_slot) in cfg_coalesce_pairs {
        alloc.try_coalesce(src_slot, dst_slot);
    }
    alloc.coalesce_variables(ssarepr, kind);
    alloc.find_node_coloring();
    alloc
}

/// `tool/algo/regalloc.py:18-143` `RegAllocator`.
///
/// RPython:
/// ```python
/// class RegAllocator(object):
///     def __init__(self, graph, consider_var, ListOfKind): ...
///     def make_dependencies(self): ...
///     def coalesce_variables(self): ...
///     def find_node_coloring(self): ...
///     def find_num_colors(self): ...
///     def getcolor(self, v): ...
///     def swapcolors(self, col1, col2): ...
/// ```
struct RegAllocator {
    depgraph: DependencyGraph,
    /// Union-find over register indices (RPython
    /// `tool.algo.unionfind.UnionFind.link_to_parent`). Created
    /// lazily; missing nodes self-rep.
    unionfind: HashMap<u16, u16>,
    /// RPython `UnionFind.weight` — only roots carry entries.
    unionfind_weight: HashMap<u16, usize>,
    coloring: HashMap<u16, u16>,
}

impl RegAllocator {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: HashMap::new(),
            unionfind_weight: HashMap::new(),
            coloring: HashMap::new(),
        }
    }

    /// `unionfind.find_rep` with path compression.
    fn find_rep(&mut self, v: u16) -> u16 {
        if !self.unionfind.contains_key(&v) {
            self.unionfind.insert(v, v);
            self.unionfind_weight.insert(v, 1);
            return v;
        }
        let mut root = v;
        while self.unionfind[&root] != root {
            root = self.unionfind[&root];
        }
        let mut current = v;
        while current != root {
            let next = self.unionfind[&current];
            self.unionfind.insert(current, root);
            current = next;
        }
        root
    }

    /// `unionfind.union` — weighted union, matching
    /// `rpython/tool/algo/unionfind.py:67-91`.
    fn union(&mut self, v0: u16, w0: u16) -> u16 {
        let r1 = self.find_rep(v0);
        let r2 = self.find_rep(w0);
        if r1 == r2 {
            return r1;
        }
        let w1 = self.unionfind_weight.get(&r1).copied().unwrap_or(1);
        let w2 = self.unionfind_weight.get(&r2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (r1, r2) } else { (r2, r1) };
        self.unionfind.insert(loser, winner);
        self.unionfind_weight.remove(&loser);
        *self.unionfind_weight.entry(winner).or_insert(0) = w1 + w2;
        winner
    }

    /// `regalloc.py:26-77` `RegAllocator.make_dependencies`.
    ///
    /// RPython walks each block forward computing per-variable
    /// `die_at` (last-use index), then forward again killing dead
    /// variables and adding interference edges between every result
    /// register and the currently-alive set. Pyre's input is a flat
    /// `SSARepr`, so the equivalent live-set computation is done in
    /// a backward sweep with a fixpoint over labels (analogous to
    /// `liveness.py`'s alive-set propagation).
    fn make_dependencies(&mut self, ssarepr: &SSARepr, kind: Kind, external_inputs: &[u16]) {
        // regalloc.py:54-60 `for i, v in enumerate(livevars):
        //   ... for j in range(i): dg.add_edge(livevars[j], v)`.
        for (i, &v) in external_inputs.iter().enumerate() {
            self.depgraph.add_node(ValueId(v as usize));
            for j in 0..i {
                if external_inputs[j] != v {
                    self.depgraph
                        .add_edge(ValueId(external_inputs[j] as usize), ValueId(v as usize));
                }
            }
        }

        // Backward live-set walk over the SSARepr with a label
        // fixpoint. Equivalent in effect to `regalloc.py:62-77`'s
        // forward `die_at` driven loop, just expressed in the
        // direction natural to pyre's flat instruction list (matches
        // `super::liveness::_compute_liveness_must_continue`).
        let mut label2alive: HashMap<String, HashSet<u16>> = HashMap::new();
        loop {
            let mut alive: HashSet<u16> = HashSet::new();
            let mut must_continue = false;

            for insn in ssarepr.insns.iter().rev() {
                match insn {
                    // PRE-EXISTING-ADAPTATION (see `Insn::PcAnchor` in
                    // `flatten.rs`). Anchors carry no operand or
                    // liveness — pure SSARepr-position markers for
                    // post-assemble pc_map build. Skip in the
                    // interference walk.
                    Insn::PcAnchor(_) => {}
                    Insn::Label(label) => {
                        let alive_at_point = label2alive.entry(label.name.clone()).or_default();
                        let prevlength = alive_at_point.len();
                        alive_at_point.extend(alive.iter().copied());
                        if prevlength != alive_at_point.len() {
                            must_continue = true;
                        }
                    }
                    Insn::Unreachable => {
                        alive.clear();
                    }
                    Insn::Op { args, result, .. } => {
                        // Defs: `'->' result` interferes with everything
                        // currently alive (regalloc.py:70-76).
                        if let Some(reg) = result {
                            if reg.kind == kind {
                                self.depgraph.add_node(ValueId(reg.index as usize));
                                for &a in &alive {
                                    if a != reg.index {
                                        self.depgraph.add_node(ValueId(a as usize));
                                        self.depgraph.add_edge(
                                            ValueId(reg.index as usize),
                                            ValueId(a as usize),
                                        );
                                    }
                                }
                                alive.remove(&reg.index);
                            }
                        }
                        // Uses: every Register / ListOfKind in args
                        // becomes alive for preceding instructions.
                        for x in args {
                            match x {
                                Operand::Register(reg) if reg.kind == kind => {
                                    alive.insert(reg.index);
                                    self.depgraph.add_node(ValueId(reg.index as usize));
                                }
                                Operand::ListOfKind(lst) if lst.kind == kind => {
                                    for y in &lst.content {
                                        if let Operand::Register(reg) = y {
                                            if reg.kind == kind {
                                                alive.insert(reg.index);
                                                self.depgraph.add_node(ValueId(reg.index as usize));
                                            }
                                        }
                                    }
                                }
                                Operand::TLabel(lbl) => follow_label(&mut alive, &label2alive, lbl),
                                Operand::Descr(rc) => {
                                    if let DescrOperand::SwitchDict(descr) = &**rc {
                                        for (_, label) in &descr.labels {
                                            follow_label(&mut alive, &label2alive, label);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }

            if !must_continue {
                break;
            }
        }
    }

    /// `regalloc.py:79-112` `RegAllocator.coalesce_variables`.
    ///
    /// RPython coalesces `link.args[i]` with `link.target.inputargs[i]`
    /// for every block-exit link. Pyre's `SSARepr` is post-flatten
    /// and has no `link.args ↔ inputargs` pairing — the flatten step
    /// dropped it in favor of pinned PyFrame slot indices. The
    /// SSARepr-level remnant of jump-edge unification is the `*_copy`
    /// instruction, which expresses `target_register := source`.
    /// Coalescing `*_copy`'s source and result lets the chordal
    /// coloring assign them the same color, turning the copy into a
    /// runtime no-op when src == dst.
    ///
    /// PRE-EXISTING-ADAPTATION: SSARepr-level copy coalescing instead
    /// of FunctionGraph-level link.args coalescing. The effect is a
    /// strict subset of RPython's because pyre still does not see the
    /// original cross-block link representation.
    fn coalesce_variables(&mut self, ssarepr: &SSARepr, kind: Kind) {
        let copy_op = match kind {
            Kind::Int => "int_copy",
            Kind::Ref => "ref_copy",
            Kind::Float => "float_copy",
        };
        for insn in &ssarepr.insns {
            if let Insn::Op {
                opname,
                args,
                result,
            } = insn
            {
                if opname != copy_op {
                    continue;
                }
                let dst = match result {
                    Some(r) if r.kind == kind => *r,
                    _ => continue,
                };
                let src = match args.first() {
                    Some(Operand::Register(r)) if r.kind == kind => *r,
                    _ => continue,
                };
                self.try_coalesce(src.index, dst.index);
            }
        }
    }

    /// `regalloc.py:98-112` `RegAllocator._try_coalesce`.
    fn try_coalesce(&mut self, v: u16, w: u16) {
        let v0 = self.find_rep(v);
        let w0 = self.find_rep(w);
        if v0 == w0 {
            return;
        }
        if self
            .depgraph
            .has_edge(ValueId(v0 as usize), ValueId(w0 as usize))
        {
            return;
        }
        let rep = self.union(v0, w0);
        if rep == v0 {
            self.depgraph
                .coalesce(ValueId(w0 as usize), ValueId(v0 as usize));
        } else {
            self.depgraph
                .coalesce(ValueId(v0 as usize), ValueId(w0 as usize));
        }
    }

    /// `regalloc.py:114-120` `RegAllocator.find_node_coloring`.
    fn find_node_coloring(&mut self) {
        let coloring = self.depgraph.find_node_coloring();
        // RPython stores coloring keyed by union-find rep; pyre
        // expands back to all per-register entries by composing with
        // the union-find map so getcolor / rename can dereference any
        // original register index.
        self.coloring = HashMap::new();
        // Collect all registers seen (both in unionfind and in
        // depgraph rep set).
        let all_regs: HashSet<u16> = self
            .unionfind
            .keys()
            .copied()
            .chain(coloring.keys().map(|vid| vid.0 as u16))
            .collect();
        for v in all_regs {
            let rep = self.find_rep(v);
            if let Some(&color) = coloring.get(&ValueId(rep as usize)) {
                self.coloring.insert(v, color as u16);
            }
        }
    }

    /// `regalloc.py:129-130` `RegAllocator.getcolor`.
    fn getcolor(&mut self, v: u16) -> Option<u16> {
        let rep = self.find_rep(v);
        // The expanded coloring always carries per-register entries
        // (see find_node_coloring), but defend against unused regs.
        self.coloring
            .get(&v)
            .copied()
            .or_else(|| self.coloring.get(&rep).copied())
    }

    /// `regalloc.py:138-143` `RegAllocator.swapcolors`.
    fn swapcolors(&mut self, col1: u16, col2: u16) {
        for color in self.coloring.values_mut() {
            if *color == col1 {
                *color = col2;
            } else if *color == col2 {
                *color = col1;
            }
        }
    }

    /// `rpython/tool/algo/regalloc.py:122-127` `RegAllocator.find_num_colors`:
    /// `max(self._coloring.values())+1 if self._coloring else 0`.
    fn find_num_colors(&self) -> u16 {
        self.coloring.values().copied().max().map_or(0, |m| m + 1)
    }
}

#[inline]
fn follow_label(
    alive: &mut HashSet<u16>,
    label2alive: &HashMap<String, HashSet<u16>>,
    lbl: &TLabel,
) {
    if let Some(alive_at_point) = label2alive.get(&lbl.name) {
        alive.extend(alive_at_point.iter().copied());
    }
}

/// Apply the rename table to the `SSARepr` in place.
///
/// Walks every `Insn::Op` (including `-live-` markers), rewriting
/// `Register` operands and `result` registers through the rename
/// table. Leaves constants, labels, descrs, and indirect-call-target
/// operands untouched.
///
/// `-live-` markers happen to be empty at the point this function
/// runs in the current pipeline (dispatch emits placeholders,
/// `filter_liveness_in_place` only populates them AFTER, per
/// `codewriter.py:44-56` parity), so the walk is a no-op on them —
/// but the handling is order-agnostic: if a `-live-` marker ever
/// arrives here with registers, they'd be remapped consistently with
/// the surrounding ops.
pub(super) fn apply_rename(ssarepr: &mut SSARepr, rename: &HashMap<(Kind, u16), u16>) {
    if rename.is_empty() {
        return;
    }
    for insn in ssarepr.insns.iter_mut() {
        match insn {
            Insn::Op { args, result, .. } => {
                if let Some(reg) = result {
                    rename_register(reg, rename);
                }
                for op in args.iter_mut() {
                    rename_operand(op, rename);
                }
            }
            Insn::Label(_) | Insn::Unreachable | Insn::PcAnchor(_) => {}
        }
    }
}

fn rename_operand(op: &mut Operand, rename: &HashMap<(Kind, u16), u16>) {
    match op {
        Operand::Register(reg) => rename_register(reg, rename),
        Operand::ListOfKind(lst) => {
            for inner in lst.content.iter_mut() {
                rename_operand(inner, rename);
            }
        }
        _ => {}
    }
}

#[inline]
fn rename_register(reg: &mut Register, rename: &HashMap<(Kind, u16), u16>) {
    if let Some(&new) = rename.get(&(reg.kind, reg.index)) {
        reg.index = new;
    }
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{Insn, Kind, ListOfKind, Operand, Register, SSARepr};
    use super::super::flow::{
        Block, Constant, FlowListOfKind, FunctionGraph, Link, SpaceOperation, Variable, VariableId,
        push_op,
    };
    use super::*;

    fn op_def(name: &str, args: Vec<Operand>, result: Register) -> Insn {
        Insn::op_with_result(name, args, result)
    }

    fn op_use(name: &str, args: Vec<Operand>) -> Insn {
        Insn::op(name, args)
    }

    fn reg(kind: Kind, idx: u16) -> Operand {
        Operand::reg(kind, idx)
    }

    fn r(kind: Kind, idx: u16) -> Register {
        Register::new(kind, idx)
    }

    fn flow_var(id: u32, kind: Kind) -> Variable {
        Variable::new(VariableId(id), kind)
    }

    #[test]
    fn all_kinds_driver_produces_regalloc_for_every_kind() {
        // Build a graph with one variable per kind as startblock inputs,
        // and a returnblock that takes a single Int so the link has
        // matching arity.
        let v0 = flow_var(0, Kind::Int);
        let vr = flow_var(1, Kind::Ref);
        let vf = flow_var(2, Kind::Float);
        let start = Block::shared(vec![v0.into(), vr.into(), vf.into()]);
        let graph = FunctionGraph::new("all_kinds", start.clone(), Some(v0));
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let regallocs = perform_graph_register_allocation_all_kinds(&graph);
        assert_eq!(regallocs.len(), 3);
        for &kind in &Kind::ALL {
            let result = regallocs
                .get(&kind)
                .unwrap_or_else(|| panic!("missing regalloc for {kind:?}"));
            // Each kind has at least one variable (Int: v0 twice via
            // return link; Ref: vr in startblock inputargs; Float: vf).
            // Colorings never exceed num_colors.
            assert!(
                result.num_colors >= 1,
                "kind {kind:?} expected at least one color, got {}",
                result.num_colors
            );
            for (_id, color) in &result.coloring {
                assert!(
                    *color < result.num_colors,
                    "kind {kind:?} color {color} exceeds num_colors {}",
                    result.num_colors
                );
            }
        }
    }

    #[test]
    fn graph_regalloc_reuses_color_for_non_overlapping_values() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_regalloc", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new("same_as", vec![v0.into()], Some(v1.into()), 0),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_graph_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_coalesces_goto_link_args_with_target_inputargs() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_goto", start.clone(), None);
        let next = graph.new_block(vec![v1.into()]);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_graph_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_seeds_exception_extravars_as_colorable_nodes() {
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("graph_exc", start.clone(), None);
        let target = graph.new_block(Vec::new());
        let exc_type = flow_var(10, Kind::Int);
        let mut link = Link::new(Vec::new(), Some(target), None);
        link.extravars(Some(exc_type), None);
        start.closeblock(vec![link.into_ref()]);

        let result = perform_graph_register_allocation(&graph, Kind::Int);
        assert_eq!(result.coloring.get(&exc_type.id), Some(&0));
        assert_eq!(result.num_colors, 1);
    }

    #[test]
    fn graph_regalloc_marks_listofkind_args_as_uses() {
        let v0 = flow_var(0, Kind::Int);
        let v1 = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_listofkind", start.clone(), Some(v1));
        push_op(
            &start,
            SpaceOperation::new(
                "same_as",
                vec![Constant::signed(1).into()],
                Some(v1.into()),
                0,
            ),
        );
        push_op(
            &start,
            SpaceOperation::new(
                "consume",
                vec![FlowListOfKind::new(Kind::Int, vec![v0.into()]).into()],
                None,
                0,
            ),
        );
        start.closeblock(vec![
            Link::new(vec![v1.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let result = perform_graph_register_allocation(&graph, Kind::Int);
        assert_ne!(result.coloring.get(&v0.id), result.coloring.get(&v1.id));
        assert_eq!(result.num_colors, 2);
    }

    /// `flatten.py:88-100 enforce_input_args` parity at the graph
    /// allocator level: after the simulation, every kind's startblock
    /// inputargs occupy colors `0, 1, 2, …` in source order.
    #[test]
    fn enforce_input_args_simulation_normalises_graph_inputarg_colors() {
        // 2 Ref inputargs + 1 Int inputarg, all live across an op
        // that defines fresh Variables of each kind so the chordal
        // coloring has to place every node on its own color.
        let a = flow_var(0, Kind::Ref);
        let b = flow_var(1, Kind::Ref);
        let i = flow_var(2, Kind::Int);
        let r0 = flow_var(10, Kind::Ref);
        let i0 = flow_var(11, Kind::Int);
        let start = Block::shared(vec![a.into(), b.into(), i.into()]);
        let mut graph = FunctionGraph::new("enforce_sim", start.clone(), None);
        push_op(
            &start,
            SpaceOperation::new("consume_ref", vec![a.into(), b.into()], Some(r0.into()), 0),
        );
        push_op(
            &start,
            SpaceOperation::new("consume_int", vec![i.into()], Some(i0.into()), 0),
        );
        let next = graph.new_block(vec![r0.into(), i0.into()]);
        start.closeblock(vec![
            Link::new(vec![r0.into(), i0.into()], Some(next), None).into_ref(),
        ]);

        let mut regallocs = perform_graph_register_allocation_all_kinds(&graph);
        enforce_input_args_simulation(&graph, &mut regallocs);

        let ref_colors = &regallocs[&Kind::Ref].coloring;
        let int_colors = &regallocs[&Kind::Int].coloring;
        assert_eq!(
            ref_colors.get(&a.id).copied(),
            Some(0),
            "first Ref inputarg must occupy color 0 post-simulation"
        );
        assert_eq!(
            ref_colors.get(&b.id).copied(),
            Some(1),
            "second Ref inputarg must occupy color 1 post-simulation"
        );
        assert_eq!(
            int_colors.get(&i.id).copied(),
            Some(0),
            "first Int inputarg must occupy color 0 post-simulation"
        );
    }

    /// (a) inputargs land on consecutive colors `0..n-1` regardless
    /// of how the chordal coloring picks initial colors.
    #[test]
    fn enforce_input_args_normalises_inputarg_colors() {
        // 2 locals + 2 portal regs; introduce an op that defines a
        // fresh Ref register and has all 4 inputargs alive. The
        // chordal coloring assigns 5 different colors; without
        // enforce_input_args nothing constrains *which* color goes
        // to which register.
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_def(
            "consume",
            vec![
                reg(Kind::Ref, 0),
                reg(Kind::Ref, 1),
                reg(Kind::Ref, 100),
                reg(Kind::Ref, 101),
            ],
            r(Kind::Ref, 200),
        ));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 200)]));

        let inputs = ExternalInputs {
            portal_frame_reg: 100,
            portal_ec_reg: 101,
            portal_inputs: true,
        };
        let result = allocate_registers(&ssarepr, 2, inputs, &[]);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        // locals 0,1 → colors 0,1; portal regs → 2,3.
        assert_eq!(new(0), 0, "local 0 must keep color 0 after enforce");
        assert_eq!(new(1), 1, "local 1 must keep color 1 after enforce");
        assert_eq!(new(100), 2, "portal_frame_reg must land on color 2");
        assert_eq!(new(101), 3, "portal_ec_reg must land on color 3");
    }

    /// (b) RPython `regalloc.py:54-60` parity: a dead local's color is
    /// reused by an SSA temp whose live range starts after the local
    /// dies.  Resume data is Box-by-identity, not register-slot indexed,
    /// so temp/local color sharing is sound.
    #[test]
    fn dead_local_color_reused() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr
            .insns
            .push(op_use("read_local", vec![reg(Kind::Ref, 0)]));
        ssarepr
            .insns
            .push(op_def("make_value", vec![], r(Kind::Ref, 100)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 100)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let mut new = |old: u16| result.rename.get(&(Kind::Ref, old)).copied().unwrap_or(old);
        assert_eq!(new(0), 0, "local 0 stays at color 0 (enforce_input_args)");
        assert_eq!(
            new(100),
            0,
            "temp 100 reuses local 0's color (local dies before temp's def)"
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(1),
            "single color for the disjoint live ranges"
        );
    }

    /// (c) `num_regs[kind]` equals the number of distinct colors
    /// used after coloring (codewriter.py:62-67).
    #[test]
    fn num_regs_matches_max_color_plus_one() {
        // Force 3 simultaneously-live Ref registers + 1 Int.
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_def("a", vec![], r(Kind::Ref, 50)));
        ssarepr.insns.push(op_def("b", vec![], r(Kind::Ref, 51)));
        ssarepr.insns.push(op_def("c", vec![], r(Kind::Ref, 52)));
        ssarepr.insns.push(op_use(
            "use_all",
            vec![reg(Kind::Ref, 50), reg(Kind::Ref, 51), reg(Kind::Ref, 52)],
        ));
        ssarepr.insns.push(op_def("i0", vec![], r(Kind::Int, 7)));
        ssarepr
            .insns
            .push(op_use("use_int", vec![reg(Kind::Int, 7)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        assert_eq!(result.num_regs.get(&Kind::Ref).copied(), Some(3));
        assert_eq!(result.num_regs.get(&Kind::Int).copied(), Some(1));
        assert_eq!(result.num_regs.get(&Kind::Float).copied(), Some(0));
    }

    #[test]
    fn union_keeps_heavier_partition_as_representative() {
        let mut alloc = RegAllocator::new();

        assert_eq!(alloc.union(10, 11), 10);
        assert_eq!(alloc.find_rep(10), 10);
        assert_eq!(alloc.find_rep(11), 10);

        assert_eq!(alloc.union(12, 13), 12);
        assert_eq!(alloc.find_rep(12), 12);
        assert_eq!(alloc.find_rep(13), 12);

        assert_eq!(alloc.union(10, 12), 10);
        assert_eq!(alloc.find_rep(10), 10);
        assert_eq!(alloc.find_rep(11), 10);
        assert_eq!(alloc.find_rep(12), 10);
        assert_eq!(alloc.find_rep(13), 10);
    }

    /// (d) `coalesce_variables` unifies a `*_copy dst <- src` source
    /// and target into the same color when they don't interfere.
    #[test]
    fn move_source_and_target_coalesce_to_same_color() {
        // r5 = produce; ref_copy r6 <- r5; use r6.
        // r5 dies at the copy; r6 takes over. With coalescing they
        // should share a color.
        let mut ssarepr = SSARepr::new("t");
        ssarepr
            .insns
            .push(op_def("produce", vec![], r(Kind::Ref, 5)));
        ssarepr
            .insns
            .push(op_def("ref_copy", vec![reg(Kind::Ref, 5)], r(Kind::Ref, 6)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 6)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 0, inputs, &[]);
        let new5 = result.rename.get(&(Kind::Ref, 5)).copied().unwrap_or(5);
        let new6 = result.rename.get(&(Kind::Ref, 6)).copied().unwrap_or(6);
        assert_eq!(
            new5, new6,
            "coalesce_variables should give ref_copy src and dst the same color (got {} vs {})",
            new5, new6
        );
        assert_eq!(
            result.num_regs.get(&Kind::Ref).copied(),
            Some(1),
            "after coalesce only one Ref color is needed"
        );
    }

    /// (e) RPython `flatten.py:88-100` parity: non-inputarg registers
    /// reuse a dead inputarg's slot when their live range starts after
    /// the inputarg's last use.  Resume data is Box-by-identity, so the
    /// reuse is sound — the same color identifying two non-overlapping
    /// values does not break the resume contract.
    #[test]
    fn non_inputarg_can_reuse_inputarg_color() {
        let mut ssarepr = SSARepr::new("t");
        ssarepr.insns.push(op_use(
            "read_local",
            vec![Operand::ListOfKind(ListOfKind::new(
                Kind::Ref,
                vec![reg(Kind::Ref, 0)],
            ))],
        ));
        ssarepr
            .insns
            .push(op_def("make_value", vec![], r(Kind::Ref, 50)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 50)]));

        let inputs = ExternalInputs {
            portal_frame_reg: u16::MAX,
            portal_ec_reg: u16::MAX,
            portal_inputs: false,
        };
        let result = allocate_registers(&ssarepr, 1, inputs, &[]);
        let new50 = result.rename.get(&(Kind::Ref, 50)).copied().unwrap_or(50);
        assert_eq!(
            new50, 0,
            "non-inputarg reg 50 reuses dead inputarg 0's color (RPython parity)"
        );
    }

    #[test]
    fn weighted_union_prefers_heavier_partition() {
        let mut alloc = RegAllocator::new();
        assert_eq!(alloc.union(1, 2), 1);
        assert_eq!(alloc.union(3, 4), 3);
        assert_eq!(alloc.union(3, 5), 3);
        assert_eq!(
            alloc.union(1, 3),
            3,
            "RPython UnionFind keeps the heavier partition's representative"
        );
        assert_eq!(alloc.find_rep(1), 3);
        assert_eq!(alloc.find_rep(2), 3);
        assert_eq!(alloc.find_rep(4), 3);
        assert_eq!(alloc.find_rep(5), 3);
    }

    #[test]
    fn count_link_renamings_skips_same_color_pair() {
        // startblock(v0 Int) → returnblock(v_ret Int) via Link args=[v0].
        // Graph regalloc coalesces v0 and v_ret to the same color (both
        // 0), so insert_renamings would emit zero `*_copy` ops.
        let v0 = flow_var(0, Kind::Int);
        let v_ret = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let graph = FunctionGraph::new("count_renamings_same", start.clone(), Some(v_ret));
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let regallocs = perform_graph_register_allocation_all_kinds(&graph);
        let counts = count_link_renamings_per_kind(&graph, &regallocs);
        assert_eq!(counts.get(&Kind::Int).copied().unwrap_or(0), 0);
        assert_eq!(counts.get(&Kind::Ref).copied().unwrap_or(0), 0);
        assert_eq!(counts.get(&Kind::Float).copied().unwrap_or(0), 0);
    }

    #[test]
    fn count_link_renamings_counts_one_step_when_colors_differ() {
        // Synthesize a one-step renaming using a hand-built coloring
        // map so the test pins the helper's pair-emission logic
        // directly (without depending on coalesce/interference
        // shapes that perform_graph_register_allocation chooses).
        // startblock(v_src Int) → next(v_dst Int) via Link args=[v_src].
        let v_src = flow_var(0, Kind::Int);
        let v_dst = flow_var(1, Kind::Int);
        let start = Block::shared(vec![v_src.into()]);
        let mut graph = FunctionGraph::new("count_renamings_step", start.clone(), None);
        let next = graph.new_block(vec![v_dst.into()]);
        start.closeblock(vec![
            Link::new(vec![v_src.into()], Some(next.clone()), None).into_ref(),
        ]);
        next.closeblock(vec![
            // graph.returnblock has 1 untyped Variable as inputarg
            // (FunctionGraph::new with `None` allocates one). Pass any
            // FlowValue; the helper skips untyped dst pairs at the
            // `dst_var.kind` short-circuit.
            Link::new(vec![v_dst.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let mut coloring = HashMap::new();
        coloring.insert(v_src.id, 0u16);
        coloring.insert(v_dst.id, 1u16);
        let mut regallocs: HashMap<Kind, GraphAllocationResult> = HashMap::new();
        regallocs.insert(
            Kind::Int,
            GraphAllocationResult {
                coloring,
                num_colors: 2,
            },
        );
        // Empty per-kind entries for Ref/Float so the helper's
        // `regallocs.get(&kind)` short-circuits cleanly.
        regallocs.insert(
            Kind::Ref,
            GraphAllocationResult {
                coloring: HashMap::new(),
                num_colors: 0,
            },
        );
        regallocs.insert(
            Kind::Float,
            GraphAllocationResult {
                coloring: HashMap::new(),
                num_colors: 0,
            },
        );

        let counts = count_link_renamings_per_kind(&graph, &regallocs);
        // Single (v_src@0 → v_dst@1) pair: no cycle, one `int_copy` step.
        assert_eq!(counts.get(&Kind::Int).copied().unwrap_or(0), 1);
        assert_eq!(counts.get(&Kind::Ref).copied().unwrap_or(0), 0);
        assert_eq!(counts.get(&Kind::Float).copied().unwrap_or(0), 0);
    }

    #[test]
    fn count_graph_ops_per_opname_aggregates_across_blocks() {
        // Graph with 2 reachable blocks; each block carries
        // SpaceOperations that the helper aggregates by opname.
        let v0 = flow_var(0, Kind::Int);
        let start = Block::shared(vec![v0.into()]);
        let mut graph = FunctionGraph::new("graph_ops_count", start.clone(), Some(v0));
        push_op(&start, SpaceOperation::new("foo", vec![v0.into()], None, 0));
        push_op(&start, SpaceOperation::new("bar", vec![v0.into()], None, 0));
        let mid = graph.new_block(vec![v0.into()]);
        start.closeblock(vec![
            Link::new(vec![v0.into()], Some(mid.clone()), None).into_ref(),
        ]);
        push_op(&mid, SpaceOperation::new("foo", vec![v0.into()], None, 0));
        mid.closeblock(vec![
            Link::new(vec![v0.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);
        let counts = count_graph_ops_per_opname(&graph);
        assert_eq!(counts.get("foo").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("bar").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("baz").copied().unwrap_or(0), 0);
    }

    #[test]
    fn count_ssa_ops_per_opname_aggregates_insns() {
        let mut ssarepr = SSARepr::new("ssa_ops_count");
        ssarepr.insns.push(op_use("foo", vec![]));
        ssarepr.insns.push(op_use("foo", vec![]));
        ssarepr.insns.push(op_use("bar", vec![]));
        let counts = count_ssa_ops_per_opname(&ssarepr);
        assert_eq!(counts.get("foo").copied().unwrap_or(0), 2);
        assert_eq!(counts.get("bar").copied().unwrap_or(0), 1);
        assert_eq!(counts.get("baz").copied().unwrap_or(0), 0);
    }

    #[test]
    fn count_ssa_copy_ops_aggregates_per_kind() {
        let mut ssarepr = SSARepr::new("count_copy_ops");
        ssarepr.insns.push(op_def(
            "int_copy",
            vec![Operand::ConstInt(7)],
            r(Kind::Int, 0),
        ));
        ssarepr
            .insns
            .push(op_def("int_copy", vec![reg(Kind::Int, 0)], r(Kind::Int, 1)));
        ssarepr
            .insns
            .push(op_def("ref_copy", vec![reg(Kind::Ref, 0)], r(Kind::Ref, 1)));
        ssarepr
            .insns
            .push(op_use("ref_return", vec![reg(Kind::Ref, 1)]));
        let counts = count_ssa_copy_ops_per_kind(&ssarepr);
        assert_eq!(counts.get(&Kind::Int).copied().unwrap_or(0), 2);
        assert_eq!(counts.get(&Kind::Ref).copied().unwrap_or(0), 1);
        assert_eq!(counts.get(&Kind::Float).copied().unwrap_or(0), 0);
    }
}
