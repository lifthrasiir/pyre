//! RPython `rpython/translator/backendopt/ssa.py` — SSI ↔ SSA
//! conversion utilities built on top of `UnionFind`.
//!
//! Upstream module header (ssa.py:1-2) imports only `Variable`,
//! `mkentrymap`, and `UnionFind`; the Rust port mirrors that surface.
//!
//! ```python
//! from rpython.flowspace.model import Variable, mkentrymap
//! from rpython.tool.algo.unionfind import UnionFind
//! ```

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::flowspace::model::{BlockKey, BlockRef, FunctionGraph, Hlvalue, Variable, mkentrymap};
use crate::tool::algo::unionfind::UnionFind;

/// One unification "opportunity" in upstream's list-of-lists layout.
///
/// ```python
/// vars = [block, inputvar, linkvar, linkvar, linkvar...]
/// ```
///
/// Rust makes the fields explicit: `block` is the owning block of the
/// phi node, `inputvar` is the `Variable` that `block.inputargs[n]`
/// carries, and `linkvars` are the nth args from each incoming link.
/// `linkvars` can mix Variables and Constants — the `without_const`
/// flag at construction time mirrors upstream's split of the two lists.
#[derive(Clone, Debug)]
struct Opportunity {
    block: BlockRef,
    inputvar: Hlvalue,
    linkvars: Vec<Hlvalue>,
}

/// RPython `class DataFlowFamilyBuilder` (ssa.py:4-90).
///
/// > Follow the flow of the data in the graph. Builds a UnionFind
/// > grouping all the variables by families: each family contains
/// > exactly one variable where a value is stored into -- either by an
/// > operation or a merge -- and all following variables where the
/// > value is just passed unmerged into the next block.
pub struct DataFlowFamilyBuilder {
    /// RPython `self.opportunities` (ssa.py:17).
    opportunities: Vec<Opportunity>,
    /// RPython `self.opportunities_with_const` (ssa.py:18).
    opportunities_with_const: Vec<Opportunity>,
    /// RPython `self.variable_families = UnionFind()` (ssa.py:36).
    pub variable_families: UnionFind<Hlvalue, ()>,
}

impl DataFlowFamilyBuilder {
    /// RPython `DataFlowFamilyBuilder.__init__(self, graph)`
    /// (ssa.py:12-36).
    ///
    /// ```python
    /// def __init__(self, graph):
    ///     opportunities = []
    ///     opportunities_with_const = []
    ///     entrymap = mkentrymap(graph)
    ///     del entrymap[graph.startblock]
    ///     for block, links in entrymap.items():
    ///         assert links
    ///         for n, inputvar in enumerate(block.inputargs):
    ///             vars = [block, inputvar]
    ///             put_in = opportunities
    ///             for link in links:
    ///                 var = link.args[n]
    ///                 if not isinstance(var, Variable):
    ///                     put_in = opportunities_with_const
    ///                 vars.append(var)
    ///             put_in.append(vars)
    ///     self.opportunities = opportunities
    ///     self.opportunities_with_const = opportunities_with_const
    ///     self.variable_families = UnionFind()
    /// ```
    pub fn new(graph: &FunctionGraph) -> Self {
        let mut opportunities: Vec<Opportunity> = Vec::new();
        let mut opportunities_with_const: Vec<Opportunity> = Vec::new();

        let mut entrymap = mkentrymap(graph);
        // upstream: `del entrymap[graph.startblock]`. Our mkentrymap
        // seeds the startblock with a synthetic `Link(graph.getargs(),
        // startblock)`; drop it to match upstream's delete.
        entrymap.remove(&BlockKey::of(&graph.startblock));

        for (block_key, links) in entrymap.iter() {
            // upstream: `assert links`.
            assert!(!links.is_empty(), "entrymap entry has no links");
            // Recover the block ref from the first link's target
            // (identity equals block_key).
            let block = links
                .first()
                .and_then(|l| l.borrow().target.clone())
                .expect("link.target missing");
            debug_assert_eq!(BlockKey::of(&block), *block_key);

            let inputargs = block.borrow().inputargs.clone();
            for (n, inputvar) in inputargs.iter().enumerate() {
                // upstream: `vars = [block, inputvar]`.
                let mut linkvars: Vec<Hlvalue> = Vec::with_capacity(links.len());
                // upstream: `put_in = opportunities`.
                let mut put_in_const = false;
                for link in links {
                    // upstream: `var = link.args[n]`.
                    let link_args = link.borrow().args.clone();
                    let Some(Some(var)) = link_args.get(n).cloned() else {
                        panic!(
                            "DataFlowFamilyBuilder: link.args[{n}] missing at block {:?}",
                            block_key
                        );
                    };
                    // upstream: `if not isinstance(var, Variable):
                    // put_in = opportunities_with_const`.
                    if matches!(var, Hlvalue::Constant(_)) {
                        put_in_const = true;
                    }
                    linkvars.push(var);
                }
                let opp = Opportunity {
                    block: block.clone(),
                    inputvar: inputvar.clone(),
                    linkvars,
                };
                if put_in_const {
                    opportunities_with_const.push(opp);
                } else {
                    opportunities.push(opp);
                }
            }
        }

        DataFlowFamilyBuilder {
            opportunities,
            opportunities_with_const,
            variable_families: UnionFind::new(|_: &Hlvalue| ()),
        }
    }

    /// RPython `DataFlowFamilyBuilder.complete(self)` (ssa.py:38-63).
    ///
    /// ```python
    /// def complete(self):
    ///     variable_families = self.variable_families
    ///     any_progress_at_all = False
    ///     progress = True
    ///     while progress:
    ///         progress = False
    ///         pending_opportunities = []
    ///         for vars in self.opportunities:
    ///             repvars = [variable_families.find_rep(v1) for v1 in vars[1:]]
    ///             repvars_without_duplicates = dict.fromkeys(repvars)
    ///             count = len(repvars_without_duplicates)
    ///             if count > 2:
    ///                 pending_opportunities.append(vars[:1] + repvars)
    ///             elif count == 2:
    ///                 variable_families.union(*repvars_without_duplicates)
    ///                 progress = True
    ///         self.opportunities = pending_opportunities
    ///         any_progress_at_all |= progress
    ///     return any_progress_at_all
    /// ```
    pub fn complete(&mut self) -> bool {
        let mut any_progress_at_all = false;
        let mut progress = true;
        while progress {
            progress = false;
            let mut pending_opportunities: Vec<Opportunity> = Vec::new();
            // Move out so we can iterate while mutating the UF.
            let opps = std::mem::take(&mut self.opportunities);
            for opp in opps {
                // upstream: `repvars = [find_rep(v1) for v1 in
                // vars[1:]]` — `vars[1:]` = [inputvar, *linkvars].
                let mut repvars: Vec<Hlvalue> = Vec::with_capacity(1 + opp.linkvars.len());
                repvars.push(self.variable_families.find_rep(opp.inputvar.clone()));
                for v in &opp.linkvars {
                    repvars.push(self.variable_families.find_rep(v.clone()));
                }
                // upstream: `repvars_without_duplicates = dict.fromkeys(repvars)`
                // — insertion-ordered unique keys.
                let unique: Vec<Hlvalue> = {
                    let mut seen: HashSet<Hlvalue> = HashSet::new();
                    let mut out = Vec::new();
                    for v in &repvars {
                        if seen.insert(v.clone()) {
                            out.push(v.clone());
                        }
                    }
                    out
                };
                match unique.len() {
                    n if n > 2 => {
                        // upstream: `pending_opportunities.append(
                        //     vars[:1] + repvars)` — recycle with
                        // vars[1:] replaced by their representatives.
                        // vars[0] is `block`; inputvar goes in
                        // `repvars[0]` and linkvars fill the tail.
                        let inputvar = repvars[0].clone();
                        let linkvars = repvars[1..].to_vec();
                        pending_opportunities.push(Opportunity {
                            block: opp.block.clone(),
                            inputvar,
                            linkvars,
                        });
                    }
                    2 => {
                        // upstream: `variable_families.union(*repvars_without_duplicates)`.
                        self.variable_families
                            .union(unique[0].clone(), unique[1].clone());
                        progress = true;
                    }
                    _ => {
                        // count == 0 or 1 — all repvars identical or
                        // empty, nothing to do.
                    }
                }
            }
            self.opportunities = pending_opportunities;
            any_progress_at_all |= progress;
        }
        any_progress_at_all
    }

    /// RPython `DataFlowFamilyBuilder.merge_identical_phi_nodes(self)`
    /// (ssa.py:65-86).
    pub fn merge_identical_phi_nodes(&mut self) -> bool {
        let mut any_progress_at_all = false;
        let mut progress = true;
        while progress {
            progress = false;
            // upstream: `block_phi_nodes = {}`.
            let mut block_phi_nodes: HashMap<(BlockKey, Vec<Hlvalue>), Hlvalue> = HashMap::new();
            // upstream: `for vars in self.opportunities + self.opportunities_with_const`.
            let all_opps: Vec<&Opportunity> = self
                .opportunities
                .iter()
                .chain(self.opportunities_with_const.iter())
                .collect();
            for opp in all_opps {
                let blockvar = opp.inputvar.clone();
                // upstream: `linksvars = [find_rep(v) for v in linksvars]`.
                let linksvars_rep: Vec<Hlvalue> = opp
                    .linkvars
                    .iter()
                    .map(|v| self.variable_families.find_rep(v.clone()))
                    .collect();
                // upstream: `phi_node = (block,) + tuple(linksvars)`.
                let phi_node = (BlockKey::of(&opp.block), linksvars_rep);
                if let Some(blockvar1) = block_phi_nodes.get(&phi_node).cloned() {
                    // upstream: `if variable_families.union(blockvar1,
                    // blockvar)[0]: progress = True`.
                    let (not_noop, _) = self.variable_families.union(blockvar1, blockvar);
                    if not_noop {
                        progress = true;
                    }
                } else {
                    block_phi_nodes.insert(phi_node, blockvar);
                }
            }
            any_progress_at_all |= progress;
        }
        any_progress_at_all
    }

    /// RPython `DataFlowFamilyBuilder.get_variable_families(self)`
    /// (ssa.py:88-90).
    ///
    /// ```python
    /// def get_variable_families(self):
    ///     self.complete()
    ///     return self.variable_families
    /// ```
    ///
    /// Upstream returns the same attribute the builder retains (Python
    /// reference semantics), letting callers continue to call
    /// `merge_identical_phi_nodes` / `complete` on the builder while
    /// holding the family handle. The Rust port mirrors that shape by
    /// taking `&mut self` and returning `&mut UnionFind` — callers
    /// that just want the UF and can drop the builder should route
    /// through `into_variable_families` below.
    pub fn get_variable_families(&mut self) -> &mut UnionFind<Hlvalue, ()> {
        self.complete();
        &mut self.variable_families
    }

    /// Consume-variant of `get_variable_families` — useful when the
    /// caller only needs the final UF.
    pub fn into_variable_families(mut self) -> UnionFind<Hlvalue, ()> {
        self.complete();
        self.variable_families
    }
}

/// Apply a Variable-id-keyed name/nr rewrite to every Variable slot
/// inside a block. The Rust port needs this helper because cloning a
/// `Variable` copies `_name`/`_nr` by value (see note in
/// `ssi_to_ssa`), so upstream's identity-propagating
/// `v.set_name_from(v1)` does not reach the graph-owned storage on its
/// own.
fn rewrite_variables_in_block(block: &BlockRef, rename: &HashMap<u64, (String, i64)>) {
    if rename.is_empty() {
        return;
    }
    let mut b = block.borrow_mut();
    for arg in b.inputargs.iter_mut() {
        if let Hlvalue::Variable(v) = arg {
            if let Some((name, nr)) = rename.get(&v.id()) {
                v.set_name(name.clone(), *nr);
            }
        }
    }
    for op in b.operations.iter_mut() {
        for arg in op.args.iter_mut() {
            if let Hlvalue::Variable(v) = arg {
                if let Some((name, nr)) = rename.get(&v.id()) {
                    v.set_name(name.clone(), *nr);
                }
            }
        }
        if let Hlvalue::Variable(v) = &mut op.result {
            if let Some((name, nr)) = rename.get(&v.id()) {
                v.set_name(name.clone(), *nr);
            }
        }
    }
    if let Some(Hlvalue::Variable(v)) = b.exitswitch.as_mut() {
        if let Some((name, nr)) = rename.get(&v.id()) {
            v.set_name(name.clone(), *nr);
        }
    }
    for link in &b.exits {
        let mut l = link.borrow_mut();
        for arg in l.args.iter_mut() {
            if let Some(Hlvalue::Variable(v)) = arg {
                if let Some((name, nr)) = rename.get(&v.id()) {
                    v.set_name(name.clone(), *nr);
                }
            }
        }
        if let Some(Hlvalue::Variable(v)) = l.last_exception.as_mut() {
            if let Some((name, nr)) = rename.get(&v.id()) {
                v.set_name(name.clone(), *nr);
            }
        }
        if let Some(Hlvalue::Variable(v)) = l.last_exc_value.as_mut() {
            if let Some((name, nr)) = rename.get(&v.id()) {
                v.set_name(name.clone(), *nr);
            }
        }
    }
}

/// RPython `variables_created_in(block)` (ssa.py:128-132).
///
/// ```python
/// def variables_created_in(block):
///     result = set(block.inputargs)
///     for op in block.operations:
///         result.add(op.result)
///     return result
/// ```
pub fn variables_created_in(block: &BlockRef) -> HashSet<Hlvalue> {
    let mut result: HashSet<Hlvalue> = HashSet::new();
    let b = block.borrow();
    for v in &b.inputargs {
        result.insert(v.clone());
    }
    for op in &b.operations {
        result.insert(op.result.clone());
    }
    result
}

/// RPython `SSI_to_SSA(graph)` (ssa.py:93-124).
///
/// ```python
/// def SSI_to_SSA(graph):
///     variable_families = DataFlowFamilyBuilder(graph).get_variable_families()
///     for v in variable_families.keys():
///         v1 = variable_families.find_rep(v)
///         if v1 != v:
///             v.set_name_from(v1)
///     # sanity checks follow
/// ```
///
/// Upstream's sanity checks verify no duplicate `.name` and uniform
/// `.concretetype` per family. The Rust port preserves them because
/// they catch upstream bugs at port-time too — if they fire, the
/// graph shape diverged from what `DataFlowFamilyBuilder` expected.
pub fn ssi_to_ssa(graph: &FunctionGraph) {
    let mut variable_families = DataFlowFamilyBuilder::new(graph).into_variable_families();
    // upstream: `for v in variable_families.keys(): ... v.set_name_from(v1)`.
    // Upstream mutates the Variable in place and Python identity
    // propagates the rename through every storage slot that holds the
    // same Variable. The Rust port's Variable has owned `_name` /
    // `_nr` storage — cloning makes the field copies independent. So
    // after computing the target prefix/nr for each family via the
    // UF, do a second pass that walks every block's inputargs /
    // operations / exits and rewrites the name on matching Variables
    // by id. This keeps the call-site contract identical to upstream.
    let keys: Vec<Hlvalue> = variable_families.keys().cloned().collect();
    let mut rename_targets: HashMap<u64, (String, i64)> = HashMap::new();
    for v in keys {
        let Hlvalue::Variable(v_var) = v else {
            continue;
        };
        let rep = variable_families.find_rep(Hlvalue::Variable(v_var.clone()));
        if let Hlvalue::Variable(rep_var) = rep {
            if rep_var != v_var {
                // Materialise rep_var's lazy nr before reading it so
                // set_name_from's observable state matches upstream.
                let _ = rep_var.name();
                rename_targets.insert(
                    v_var.id(),
                    (rep_var.name_prefix().to_string(), rep_var.nr()),
                );
            }
        }
    }
    for block in graph.iterblocks() {
        rewrite_variables_in_block(&block, &rename_targets);
    }

    // upstream sanity-check block: unique names per block + uniform
    // concretetype per family.
    let mut variables_by_name: HashMap<String, Vec<Variable>> = HashMap::new();
    for block in graph.iterblocks() {
        let mut vars: Vec<Variable> = Vec::new();
        for op in &block.borrow().operations {
            if let Hlvalue::Variable(v) = &op.result {
                vars.push(v.clone());
            }
        }
        for link in &block.borrow().exits {
            let l = link.borrow();
            if let Some(Hlvalue::Variable(v)) = &l.last_exception {
                vars.push(v.clone());
            }
            if let Some(Hlvalue::Variable(v)) = &l.last_exc_value {
                vars.push(v.clone());
            }
        }
        let mut seen: HashSet<String> = HashSet::new();
        for v in &vars {
            let name = v.name();
            assert!(
                seen.insert(name.clone()),
                "duplicate variable name {name} in block"
            );
        }
        for v in vars {
            variables_by_name.entry(v.name()).or_default().push(v);
        }
    }
    for (vname, vlist) in &variables_by_name {
        let first_ct = vlist.first().and_then(|v| v.concretetype());
        for v in vlist {
            assert_eq!(
                v.concretetype(),
                first_ct,
                "variables called {vname} have mixed concretetypes"
            );
        }
    }
}

/// RPython `SSA_to_SSI(graph, annotator=None)` (ssa.py:135-196).
///
/// ```python
/// def SSA_to_SSI(graph, annotator=None):
///     entrymap = mkentrymap(graph)
///     del entrymap[graph.startblock]
///     builder = DataFlowFamilyBuilder(graph)
///     variable_families = builder.get_variable_families()
///     del builder
///     pending = []
///     for block in graph.iterblocks():
///         if block not in entrymap:
///             continue
///         variables_created = variables_created_in(block)
///         seen = set(variables_created)
///         variables_used = []
///         def record_used_var(v):
///             if v not in seen:
///                 variables_used.append(v)
///                 seen.add(v)
///         for op in block.operations:
///             for arg in op.args:
///                 record_used_var(arg)
///         record_used_var(block.exitswitch)
///         for link in block.exits:
///             for arg in link.args:
///                 record_used_var(arg)
///         for v in variables_used:
///             if (isinstance(v, Variable) and
///                     v._name not in ('last_exception_', 'last_exc_value_')):
///                 pending.append((block, v))
///     while pending:
///         block, v = pending.pop()
///         v_rep = variable_families.find_rep(v)
///         variables_created = variables_created_in(block)
///         if v in variables_created:
///             continue
///         for w in variables_created:
///             w_rep = variable_families.find_rep(w)
///             if v_rep is w_rep:
///                 block.renamevariables({v: w})
///                 break
///         else:
///             try:
///                 links = entrymap[block]
///             except KeyError:
///                 raise Exception("SSA_to_SSI failed: no way to give a value to"
///                                 " %r in %r" % (v, block))
///             w = v.copy()
///             variable_families.union(v, w)
///             block.renamevariables({v: w})
///             block.inputargs.append(w)
///             for link in links:
///                 link.args.append(v)
///                 pending.append((link.prevblock, v))
/// ```
///
/// The `annotator` parameter on upstream is unused in the function
/// body (ignored for API parity with older callers). The Rust port
/// preserves that shape: `_annotator: Option<&RPythonAnnotator>`
/// matches upstream's `annotator=None` default so callers don't have
/// to pick between two signatures.
pub fn ssa_to_ssi(
    graph: &FunctionGraph,
    _annotator: Option<&crate::annotator::annrpython::RPythonAnnotator>,
) {
    let mut entrymap = mkentrymap(graph);
    entrymap.remove(&BlockKey::of(&graph.startblock));

    let mut variable_families = DataFlowFamilyBuilder::new(graph).into_variable_families();

    // upstream: build initial pending list by walking each block and
    // recording used-but-undefined Variables.
    let mut pending: Vec<(BlockRef, Variable)> = Vec::new();
    for block in graph.iterblocks() {
        if !entrymap.contains_key(&BlockKey::of(&block)) {
            continue;
        }
        let variables_created = variables_created_in(&block);
        let mut seen: HashSet<Hlvalue> = variables_created.clone();
        let mut variables_used: Vec<Hlvalue> = Vec::new();
        let record_used_var =
            |v: Option<&Hlvalue>, seen: &mut HashSet<Hlvalue>, used: &mut Vec<Hlvalue>| {
                if let Some(vv) = v {
                    if !seen.contains(vv) {
                        used.push(vv.clone());
                        seen.insert(vv.clone());
                    }
                }
            };

        let b = block.borrow();
        for op in &b.operations {
            for arg in &op.args {
                record_used_var(Some(arg), &mut seen, &mut variables_used);
            }
        }
        // upstream: `record_used_var(block.exitswitch)` — exitswitch is
        // Optional; upstream's Python accepts `None` gracefully.
        record_used_var(b.exitswitch.as_ref(), &mut seen, &mut variables_used);
        for link in &b.exits {
            let l = link.borrow();
            for arg in &l.args {
                record_used_var(arg.as_ref(), &mut seen, &mut variables_used);
            }
        }
        drop(b);

        for v in variables_used {
            if let Hlvalue::Variable(var) = v {
                // upstream: `v._name not in ('last_exception_',
                // 'last_exc_value_')`.
                let prefix = var.name_prefix();
                if prefix == "last_exception_" || prefix == "last_exc_value_" {
                    continue;
                }
                pending.push((block.clone(), var));
            }
        }
    }

    while let Some((block, v)) = pending.pop() {
        let v_rep = variable_families.find_rep(Hlvalue::Variable(v.clone()));
        let variables_created = variables_created_in(&block);
        if variables_created.contains(&Hlvalue::Variable(v.clone())) {
            continue;
        }
        // upstream: linear search for a w in the same family.
        let mut matched: Option<Variable> = None;
        for w in &variables_created {
            let Hlvalue::Variable(w_var) = w else {
                continue;
            };
            let w_rep = variable_families.find_rep(w.clone());
            if w_rep == v_rep {
                matched = Some(w_var.clone());
                break;
            }
        }
        if let Some(w_var) = matched {
            // upstream: `block.renamevariables({v: w})`.
            let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
            renaming.insert(v.clone(), Hlvalue::Variable(w_var));
            block.borrow_mut().renamevariables(&renaming);
        } else {
            // upstream: else branch — add `v` to every incoming link.
            let links = entrymap
                .get(&BlockKey::of(&block))
                .cloned()
                .unwrap_or_else(|| {
                    panic!(
                        "SSA_to_SSI failed: no way to give a value to {} in block",
                        v.name()
                    )
                });
            let w = v.copy();
            variable_families.union(Hlvalue::Variable(v.clone()), Hlvalue::Variable(w.clone()));
            let mut renaming: HashMap<Variable, Hlvalue> = HashMap::new();
            renaming.insert(v.clone(), Hlvalue::Variable(w.clone()));
            block.borrow_mut().renamevariables(&renaming);
            block.borrow_mut().inputargs.push(Hlvalue::Variable(w));
            for link in &links {
                // upstream: `link.args.append(v); pending.append((link.prevblock, v))`.
                link.borrow_mut()
                    .args
                    .push(Some(Hlvalue::Variable(v.clone())));
                let prev = link
                    .borrow()
                    .prevblock
                    .as_ref()
                    .and_then(|w| w.upgrade())
                    .expect("Link.prevblock missing");
                pending.push((prev, v.clone()));
            }
        }
    }

    // Silence unused-type warnings on the Rc import during cfg paths
    // that exclude tests — `Rc` gets used via variables_created_in's
    // HashSet<Hlvalue> (Hlvalue::Variable(Variable) stores by value,
    // not Rc). Keep the import alive to stay close to upstream
    // `from ... import Variable` shape.
    let _ = Rc::<()>::new(());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::{
        Block, BlockRefExt, ConstValue, Constant, FunctionGraph, Link, SpaceOperation,
    };
    use std::cell::RefCell;

    fn mk_graph_start_to_block(name: &str, block: BlockRef) -> FunctionGraph {
        FunctionGraph::new(name, block)
    }

    #[test]
    fn data_flow_family_builder_unifies_linear_chain() {
        // start(v_in) -> middle(v_mid) -> return(v_ret)
        // Each link passes the predecessor's var straight through.
        // After complete(), v_in, v_mid, v_ret are in the same family.
        let v_in = Variable::new();
        let v_mid = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        let middle = Block::shared(vec![Hlvalue::Variable(v_mid.clone())]);

        let graph = mk_graph_start_to_block("f", start.clone());
        let returnblock = graph.returnblock.clone();
        let v_ret = match &returnblock.borrow().inputargs[0] {
            Hlvalue::Variable(v) => v.clone(),
            _ => unreachable!(),
        };

        // start -> middle with args [v_in].
        let l1 = Link::new(
            vec![Hlvalue::Variable(v_in.clone())],
            Some(middle.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![l1]);

        // middle -> return with args [v_mid].
        let l2 = Link::new(
            vec![Hlvalue::Variable(v_mid.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        middle.closeblock(vec![l2]);

        let mut builder = DataFlowFamilyBuilder::new(&graph);
        assert!(builder.complete());
        let mut uf = builder.variable_families;
        // All three are in the same family.
        let rep_in = uf.find_rep(Hlvalue::Variable(v_in.clone()));
        let rep_mid = uf.find_rep(Hlvalue::Variable(v_mid.clone()));
        let rep_ret = uf.find_rep(Hlvalue::Variable(v_ret.clone()));
        assert_eq!(rep_in, rep_mid);
        assert_eq!(rep_mid, rep_ret);
    }

    #[test]
    fn variables_created_in_collects_inputargs_and_op_results() {
        let v_in = Variable::new();
        let v_out = Variable::new();
        let b = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        b.borrow_mut().operations.push(SpaceOperation::new(
            "op",
            vec![Hlvalue::Variable(v_in.clone())],
            Hlvalue::Variable(v_out.clone()),
        ));
        let created = variables_created_in(&b);
        assert_eq!(created.len(), 2);
        assert!(created.contains(&Hlvalue::Variable(v_in)));
        assert!(created.contains(&Hlvalue::Variable(v_out)));
    }

    #[test]
    fn ssi_to_ssa_collapses_chain_to_single_name() {
        // start(v_in) -> middle(v_mid) -> return(v_ret). After
        // SSI_to_SSA, the three variables must share the same name
        // prefix (upstream: ssa.py:106-108 `v.set_name_from(v1)`). The
        // exact prefix is whichever Variable wins the UF weighted-union
        // race, so we don't assert the specific value — only that all
        // three agree.
        let v_in = Variable::new();
        let v_mid = Variable::new();
        let start = Block::shared(vec![Hlvalue::Variable(v_in.clone())]);
        let middle = Block::shared(vec![Hlvalue::Variable(v_mid.clone())]);

        let graph = mk_graph_start_to_block("f", start.clone());
        let returnblock = graph.returnblock.clone();

        let l1 = Link::new(
            vec![Hlvalue::Variable(v_in.clone())],
            Some(middle.clone()),
            None,
        )
        .into_ref();
        start.closeblock(vec![l1]);
        let l2 = Link::new(
            vec![Hlvalue::Variable(v_mid.clone())],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        middle.closeblock(vec![l2]);

        ssi_to_ssa(&graph);

        let mid_prefix = match &middle.borrow().inputargs[0] {
            Hlvalue::Variable(v) => v.name_prefix().to_string(),
            _ => unreachable!(),
        };
        let ret_prefix = match &returnblock.borrow().inputargs[0] {
            Hlvalue::Variable(v) => v.name_prefix().to_string(),
            _ => unreachable!(),
        };
        // Both non-startblock variables must share the same prefix
        // post-rename. startblock's v_in isn't in the UF keyset (its
        // block is removed from entrymap), so we don't assert on it.
        assert_eq!(mid_prefix, ret_prefix);
    }

    #[test]
    fn merge_identical_phi_nodes_unifies_two_equal_phis() {
        // A block receives (via two incoming links) the same
        // Constant(1) into two different input slots. The two input
        // vars should then end up in one family via
        // merge_identical_phi_nodes.
        let v_a = Variable::new();
        let v_b = Variable::new();
        let body = Block::shared(vec![
            Hlvalue::Variable(v_a.clone()),
            Hlvalue::Variable(v_b.clone()),
        ]);
        let start = Block::shared(vec![]);
        let graph = mk_graph_start_to_block("f", start.clone());
        let returnblock = graph.returnblock.clone();

        // start -> body with [c1, c1] (same constant twice).
        let c1 = || Hlvalue::Constant(Constant::new(ConstValue::Int(1)));
        let link_sb = Link::new(vec![c1(), c1()], Some(body.clone()), None).into_ref();
        start.closeblock(vec![link_sb]);
        let link_br = Link::new(
            vec![body.borrow().inputargs[0].clone()],
            Some(returnblock.clone()),
            None,
        )
        .into_ref();
        body.closeblock(vec![link_br]);

        let mut builder = DataFlowFamilyBuilder::new(&graph);
        builder.merge_identical_phi_nodes();
        let mut uf = builder.variable_families;
        let rep_a = uf.find_rep(Hlvalue::Variable(v_a.clone()));
        let rep_b = uf.find_rep(Hlvalue::Variable(v_b.clone()));
        assert_eq!(rep_a, rep_b, "identical phi nodes should merge");
    }
}
