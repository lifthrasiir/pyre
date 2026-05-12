//! Register allocation on the control flow graph.
//!
//! RPython equivalent: `rpython/jit/codewriter/regalloc.py` +
//! `rpython/tool/algo/regalloc.py` + `rpython/tool/algo/color.py`.
//!
//! Operates on `FunctionGraph` (Block structure), NOT on flattened ops.
//! RPython runs regalloc BEFORE flatten: codewriter.py:45-47.
//!
//! 1. Build interference graph per-block (die_at analysis)
//! 2. Coalesce variables connected by Goto link args
//! 3. Greedy graph coloring via lexicographic BFS

use std::collections::{HashMap, HashSet};

use crate::flatten::RegKind;
use crate::model::{Block, FunctionGraph, ValueId};

// ── DependencyGraph (RPython tool/algo/color.py) ──────────────────

/// Interference graph for register allocation.
///
/// RPython: `color.py::DependencyGraph`.
///
/// Public so non-FunctionGraph callers (pyre's CPython-bytecode
/// codewriter; see `pyre/pyre-jit/src/jit/regalloc.rs`) can reuse the
/// chordal coloring without round-tripping through `FunctionGraph`.
/// Node identity is `ValueId`; per-kind callers can encode their
/// kind-specific index as `ValueId(index)` because the coloring is
/// run independently per kind (see `regalloc.py:8`).
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    all_nodes: Vec<ValueId>,
    neighbours: HashMap<ValueId, HashSet<ValueId>>,
}

impl DependencyGraph {
    pub fn new() -> Self {
        Self {
            all_nodes: Vec::new(),
            neighbours: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, v: ValueId) {
        if !self.neighbours.contains_key(&v) {
            self.all_nodes.push(v);
            self.neighbours.insert(v, HashSet::new());
        }
    }

    pub fn add_edge(&mut self, v1: ValueId, v2: ValueId) {
        if v1 == v2 {
            return;
        }
        self.neighbours.entry(v1).or_default().insert(v2);
        self.neighbours.entry(v2).or_default().insert(v1);
    }

    /// RPython: `color.py::DependencyGraph.coalesce(vold, vnew)`.
    /// Folds `vold`'s adjacency list into `vnew` and removes `vold`.
    /// Used by `RegAllocator.coalesce_variables` after a successful
    /// union so the chordal coloring sees a single combined node.
    pub fn coalesce(&mut self, vold: ValueId, vnew: ValueId) {
        if let Some(old_neighbours) = self.neighbours.remove(&vold) {
            for n in old_neighbours {
                if let Some(ns) = self.neighbours.get_mut(&n) {
                    ns.remove(&vold);
                    if n != vnew {
                        ns.insert(vnew);
                        self.neighbours.entry(vnew).or_default().insert(n);
                    }
                }
            }
        }
    }

    /// RPython: `regalloc.py:105` `v0 not in dg.neighbours[w0]`.
    /// Returns true iff there is an interference edge between `v1` and `v2`.
    pub fn has_edge(&self, v1: ValueId, v2: ValueId) -> bool {
        self.neighbours
            .get(&v1)
            .map_or(false, |ns| ns.contains(&v2))
    }

    fn getnodes(&self) -> Vec<ValueId> {
        self.all_nodes
            .iter()
            .filter(|v| self.neighbours.contains_key(v))
            .copied()
            .collect()
    }

    /// RPython: `DependencyGraph.lexicographic_order()`
    fn lexicographic_order(&self) -> Vec<ValueId> {
        let nodes = self.getnodes();
        if nodes.is_empty() {
            return Vec::new();
        }
        let mut sigma: Vec<Vec<ValueId>> = vec![nodes.into_iter().rev().collect()];
        let mut result = Vec::new();
        while !sigma.is_empty() && !sigma[0].is_empty() {
            let v = sigma[0].pop().unwrap();
            result.push(v);
            let neighb = self.neighbours.get(&v).cloned().unwrap_or_default();
            let mut new_sigma = Vec::new();
            for s in sigma {
                let (s1, s2): (Vec<_>, Vec<_>) = s.into_iter().partition(|x| neighb.contains(x));
                if !s1.is_empty() {
                    new_sigma.push(s1);
                }
                if !s2.is_empty() {
                    new_sigma.push(s2);
                }
            }
            sigma = new_sigma;
        }
        result
    }

    /// RPython: `DependencyGraph.find_node_coloring()`
    /// Uses `HashSet<usize>` — no color limit (fixes u64 overflow).
    pub fn find_node_coloring(&self) -> HashMap<ValueId, usize> {
        let mut result = HashMap::new();
        for v in self.lexicographic_order() {
            let mut forbidden: HashSet<usize> = HashSet::new();
            if let Some(neighbours) = self.neighbours.get(&v) {
                for &n in neighbours {
                    if let Some(&color) = result.get(&n) {
                        forbidden.insert(color);
                    }
                }
            }
            let mut num = 0;
            while forbidden.contains(&num) {
                num += 1;
            }
            result.insert(v, num);
        }
        result
    }
}

// ── UnionFind (RPython tool/algo/unionfind.py) ────────────────────

#[derive(Debug, Clone)]
struct UnionFind {
    parent: HashMap<ValueId, ValueId>,
    weight: HashMap<ValueId, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            weight: HashMap::new(),
        }
    }

    fn find_rep(&mut self, v: ValueId) -> ValueId {
        if !self.parent.contains_key(&v) {
            self.parent.insert(v, v);
            self.weight.insert(v, 1);
            return v;
        }
        let mut root = v;
        while self.parent[&root] != root {
            root = self.parent[&root];
        }
        let mut current = v;
        while current != root {
            let next = self.parent[&current];
            self.parent.insert(current, root);
            current = next;
        }
        root
    }

    fn union(&mut self, v1: ValueId, v2: ValueId) -> ValueId {
        let rep1 = self.find_rep(v1);
        let rep2 = self.find_rep(v2);
        if rep1 == rep2 {
            return rep1;
        }
        let w1 = self.weight.get(&rep1).copied().unwrap_or(1);
        let w2 = self.weight.get(&rep2).copied().unwrap_or(1);
        let (winner, loser) = if w1 >= w2 { (rep1, rep2) } else { (rep2, rep1) };
        self.parent.insert(loser, winner);
        self.weight.remove(&loser);
        *self.weight.entry(winner).or_insert(0) = w1 + w2;
        winner
    }
}

// ── RegAllocator (RPython tool/algo/regalloc.py) ──────────────────

/// Register allocator on FunctionGraph (Block structure).
///
/// RPython: `regalloc.py::RegAllocator`.
/// Runs BEFORE flatten, on Block/SpaceOperation structure.
#[derive(Debug)]
struct RegAllocator {
    depgraph: DependencyGraph,
    unionfind: UnionFind,
    coloring: HashMap<ValueId, usize>,
}

impl RegAllocator {
    fn new() -> Self {
        Self {
            depgraph: DependencyGraph::new(),
            unionfind: UnionFind::new(),
            coloring: HashMap::new(),
        }
    }

    /// RPython: `RegAllocator.make_dependencies()` — regalloc.py:26-77.
    /// Per-block die_at analysis.
    fn make_dependencies(&mut self, graph: &FunctionGraph, consider: &dyn Fn(ValueId) -> bool) {
        for block in &graph.blocks {
            self.process_block(block, graph, consider);
        }
    }

    /// Process one block: compute die_at, build interference edges.
    fn process_block(
        &mut self,
        block: &Block,
        _graph: &FunctionGraph,
        consider: &dyn Fn(ValueId) -> bool,
    ) {
        // die_at: last usage index of each variable in this block.
        let mut die_at: HashMap<ValueId, usize> = HashMap::new();
        for &v in &block.inputargs {
            die_at.insert(v, 0);
        }
        for (i, op) in block.operations.iter().enumerate() {
            for v in crate::inline::op_value_refs(&op.kind) {
                die_at.insert(v, i);
            }
            if let Some(result) = op.result {
                die_at.insert(result, i + 1);
            }
        }
        // Variables used in exit links stay alive until block end.
        // RPython `rpython/jit/codewriter/regalloc.py:71-78 compute_liveness`:
        // iterate `block.exits` for `link.args` + `block.exitswitch` for the
        // branch condition.
        for link in &block.exits {
            for arg in &link.args {
                if let Some(v) = arg.as_value() {
                    die_at.remove(&v);
                }
            }
        }
        if let Some(crate::model::ExitSwitch::Value(cond)) = &block.exitswitch {
            die_at.remove(cond);
        }
        let mut die_list: Vec<(usize, ValueId)> = die_at.into_iter().map(|(v, t)| (t, v)).collect();
        die_list.sort();
        die_list.push((usize::MAX, ValueId(0)));

        // inputargs all interfere with each other
        let livevars: Vec<ValueId> = block
            .inputargs
            .iter()
            .filter(|v| consider(**v))
            .copied()
            .collect();
        for (i, &v) in livevars.iter().enumerate() {
            self.depgraph.add_node(v);
            for j in 0..i {
                self.depgraph.add_edge(livevars[j], v);
            }
        }
        let mut alive: HashSet<ValueId> = livevars.into_iter().collect();

        // Scan ops, kill at die_at, add interference edges
        let mut die_index = 0;
        for (i, op) in block.operations.iter().enumerate() {
            while die_list[die_index].0 == i {
                alive.remove(&die_list[die_index].1);
                die_index += 1;
            }
            if let Some(result) = op.result {
                if consider(result) {
                    self.depgraph.add_node(result);
                    for &v in &alive {
                        if consider(v) {
                            self.depgraph.add_edge(v, result);
                        }
                    }
                    alive.insert(result);
                }
            }
        }
    }

    /// RPython: `RegAllocator.coalesce_variables()` — regalloc.py:79-96.
    /// Coalesce link.args[i] with target.inputargs[i] for every exit
    /// link, matching upstream's `for link in block.exits: ...` loop.
    /// Upstream also pre-seeds the depgraph with nodes for
    /// `link.last_exception` / `link.last_exc_value` so any downstream
    /// `getcolor(v)` against those extravars finds a colored node.
    fn coalesce_variables(&mut self, graph: &FunctionGraph, consider: &dyn Fn(ValueId) -> bool) {
        for block in &graph.blocks {
            for link in &block.exits {
                // RPython `regalloc.py:92-95`: add `link.last_exception` and
                // `link.last_exc_value` to the dep graph so subsequent
                // `_try_coalesce` calls find a node for them in
                // `DependencyGraph.coalesce`'s `neighbours[vnew]`
                // lookup.  Without this, coalescing a link arg into an
                // exception-target inputarg leaves the rep outside
                // `neighbours`, and `find_node_coloring` silently skips
                // it (`color.py:25-27 getnodes()` filters
                // `all_nodes` by `neighbours.contains_key`).
                if let Some(arg) = &link.last_exception {
                    if let Some(v) = arg.as_value() {
                        if consider(v) {
                            self.depgraph.add_node(v);
                        }
                    }
                }
                if let Some(arg) = &link.last_exc_value {
                    if let Some(v) = arg.as_value() {
                        if consider(v) {
                            self.depgraph.add_node(v);
                        }
                    }
                }
                let target_block = graph.block(link.target);
                for (v, &w) in link.args.iter().zip(target_block.inputargs.iter()) {
                    if let Some(v) = v.as_value() {
                        // RPython `regalloc.py` adds every link arg to the
                        // dependency graph before attempting coalesce so
                        // `find_node_coloring` visits it.  pyre's
                        // `process_block` only enumerates block inputargs
                        // and op results; a value that is referenced only
                        // as a link arg (no local producer in the source
                        // block) would otherwise be silently absent from
                        // `neighbours`, and `getcolor` returns `None`,
                        // which then propagates into the assembler as a
                        // missing coloring panic.
                        if consider(v) {
                            self.depgraph.add_node(v);
                        }
                        self.try_coalesce(v, w, consider);
                    }
                }
            }
        }
    }

    fn try_coalesce(&mut self, v: ValueId, w: ValueId, consider: &dyn Fn(ValueId) -> bool) {
        if !consider(v) || !consider(w) {
            return;
        }
        let v0 = self.unionfind.find_rep(v);
        let w0 = self.unionfind.find_rep(w);
        if v0 == w0 {
            return;
        }
        if self
            .depgraph
            .neighbours
            .get(&w0)
            .map_or(false, |ns| ns.contains(&v0))
        {
            return;
        }
        let rep = self.unionfind.union(v0, w0);
        if rep == v0 {
            self.depgraph.coalesce(w0, v0);
        } else {
            self.depgraph.coalesce(v0, w0);
        }
    }

    fn find_node_coloring(&mut self) {
        self.coloring = self.depgraph.find_node_coloring();
    }

    fn getcolor(&mut self, v: ValueId) -> Option<usize> {
        let rep = self.unionfind.find_rep(v);
        self.coloring.get(&rep).copied()
    }
}

// ── Public API ────────────────────────────────────────────────────

/// Result of register allocation for one kind.
#[derive(Debug, Clone)]
pub struct RegAllocResult {
    pub coloring: HashMap<ValueId, usize>,
    pub num_regs: usize,
}

/// Perform register allocation for a single kind on a graph.
///
/// RPython: `regalloc.py::perform_register_allocation(graph, kind)`.
/// Runs on FunctionGraph (Block structure), BEFORE flatten.
pub fn perform_register_allocation(
    graph: &FunctionGraph,
    kind: RegKind,
    value_kinds: &HashMap<ValueId, RegKind>,
) -> RegAllocResult {
    let consider = |v: ValueId| -> bool { value_kinds.get(&v).copied() == Some(kind) };
    let mut allocator = RegAllocator::new();
    allocator.make_dependencies(graph, &consider);
    allocator.coalesce_variables(graph, &consider);
    allocator.find_node_coloring();

    let mut coloring = HashMap::new();
    let mut max_reg = 0usize;
    for (&vid, &vkind) in value_kinds {
        if vkind == kind {
            if let Some(color) = allocator.getcolor(vid) {
                coloring.insert(vid, color);
                if color + 1 > max_reg {
                    max_reg = color + 1;
                }
            }
        }
    }
    RegAllocResult {
        coloring,
        num_regs: max_reg,
    }
}

/// Augment a `value_kinds` table with the canonical `exceptblock.inputargs`
/// kinds — upstream rtyper assigns `(etype, evalue)` concretetypes
/// `Ptr(OBJECT_VTABLE)` / `Ptr(OBJECT)` in `rpython/rtyper/rclass.py`; the
/// codewriter reads them as `i`-typed class identity and `r`-typed instance
/// pointer respectively (`flatten.py:143` `raise %r`, `flatten.py:220-231`
/// `last_exception/>i` + `goto_if_exception_mismatch/i`). pyre's codewriter
/// creates the canonical exceptblock eagerly in `FunctionGraph::new`, so
/// the matching kinds must be available whether or not the upstream rtyper
/// pass populated them. Both `perform_all_register_allocations` and
/// `flatten_with_types` apply this contract; exposing it as a helper lets
/// test fixtures that drive `flatten()` directly seed the same augmented
/// table that the assembler's `lookup_coloring` consumes.
pub fn augment_value_kinds_with_canonical_exceptblock(
    graph: &FunctionGraph,
    value_kinds: &HashMap<ValueId, RegKind>,
) -> HashMap<ValueId, RegKind> {
    let mut augmented = value_kinds.clone();
    let except_args = &graph.block(graph.exceptblock).inputargs;
    if except_args.len() == 2 {
        augmented.entry(except_args[0]).or_insert(RegKind::Int);
        augmented.entry(except_args[1]).or_insert(RegKind::Ref);
    }
    augmented
}

/// Perform register allocation for all three kinds.
pub fn perform_all_register_allocations(
    graph: &FunctionGraph,
    value_kinds: &HashMap<ValueId, RegKind>,
) -> HashMap<RegKind, RegAllocResult> {
    let augmented = augment_value_kinds_with_canonical_exceptblock(graph, value_kinds);
    let mut result = HashMap::new();
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        result.insert(kind, perform_register_allocation(graph, kind, &augmented));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FunctionGraph, OpKind, ValueType};

    #[test]
    fn non_overlapping_lifetimes_share_register() {
        // v0 = Input; v1 = BinOp(v0, v0); Return v1
        // v0 dies when v1 is defined → no interference → can share register.
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v1 = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0,
                    rhs: v0,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v1));

        let mut vk = HashMap::new();
        vk.insert(v0, RegKind::Int);
        vk.insert(v1, RegKind::Int);
        let result = perform_register_allocation(&graph, RegKind::Int, &vk);
        // v0 and v1 don't overlap → can share
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn overlapping_lifetimes_need_different_registers() {
        // v0 = Input; v1 = Input; v2 = BinOp(v0, v1); Return v2
        // v0 and v1 are both alive when v2 is defined → v0 and v1 interfere
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v1 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "b".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let v2 = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".into(),
                    lhs: v0,
                    rhs: v1,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v2));

        let mut vk = HashMap::new();
        vk.insert(v0, RegKind::Int);
        vk.insert(v1, RegKind::Int);
        vk.insert(v2, RegKind::Int);
        let result = perform_register_allocation(&graph, RegKind::Int, &vk);
        assert_ne!(
            result.coloring.get(&v0),
            result.coloring.get(&v1),
            "v0 and v1 are simultaneously alive → different registers"
        );
        // v2 can share with v0 or v1 (they die before v2's definition)
        assert!(result.num_regs >= 2);
    }

    #[test]
    fn goto_link_coalescing() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v0 = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "a".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        let (block1, block1_args) = graph.create_block_with_args(1);
        let v1 = block1_args[0];
        graph.set_goto(entry, block1, vec![v0]);
        graph.set_return(block1, Some(v1));

        let mut vk = HashMap::new();
        vk.insert(v0, RegKind::Int);
        vk.insert(v1, RegKind::Int);
        let result = perform_register_allocation(&graph, RegKind::Int, &vk);
        assert_eq!(result.coloring.get(&v0), result.coloring.get(&v1));
        assert_eq!(result.num_regs, 1);
    }

    #[test]
    fn coloring_unbounded() {
        let mut dg = DependencyGraph::new();
        for i in 0..100 {
            dg.add_node(ValueId(i));
        }
        for i in 0..99 {
            dg.add_edge(ValueId(i), ValueId(i + 1));
        }
        let coloring = dg.find_node_coloring();
        assert_eq!(coloring.len(), 100);
        let max_color = coloring.values().max().copied().unwrap_or(0);
        assert!(
            max_color <= 1,
            "chain needs at most 2 colors, got {}",
            max_color + 1
        );
    }
}
