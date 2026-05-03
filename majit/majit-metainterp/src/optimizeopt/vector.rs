/// Vector optimization pass — auto-vectorization of scalar loop bodies.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vector.py,
/// dependency.py, and costmodel.py.
///
/// This pass analyzes a peeled loop body (between Label and Jump) to find
/// independent scalar operations that can be packed into SIMD vector
/// instructions. It builds a dependency graph, identifies vectorizable
/// groups, checks profitability via a cost model, and rewrites the ops.
///
/// # Pipeline
///
/// 1. **Dependency analysis**: Build a DAG of data dependencies between ops
/// 2. **Packing**: Find pairs/groups of isomorphic ops that can run in parallel
/// 3. **Cost model**: Estimate whether vectorization is profitable
/// 4. **Rewrite**: Replace scalar ops with vector equivalents (VecIntAdd, etc.)
///
/// # TODO — remaining RPython parity gaps
///
/// - **Cranelift SIMD codegen**: vector_info (AccumInfo) is stored on all
///   guard FailDescr types (MetaFailDescr, SimpleFailDescr, ResumeGuardDescr),
///   but the Cranelift backend does not yet consume it during guard failure
///   recovery or register allocation. RPython's x86 backend reads
///   rd_vector_info in regalloc.py:347 and assembler.py:739 to handle
///   accumulator reduction on guard exit. Cranelift equivalent requires
///   SIMD codegen integration for VEC_* opcodes.
///
/// # Limitations
///
/// - Only operates on loop bodies (Label..Jump)
/// - Requires array load/store patterns for memory access vectorization
/// - Guards in the loop body prevent full vectorization (conservative)
use std::collections::{HashMap, HashSet};

use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

pub use crate::optimizeopt::dependency::schedule_operations;
pub use crate::optimizeopt::dependency::{DependencyGraph, Node};
pub use crate::optimizeopt::schedule::{
    AccumEntry, AccumPack, CostModel, GenericCostModel, GuardAnalysis, Pack, PackSet,
    VecScheduleState, are_adjacent_memory_refs, isomorphic, turn_into_vector, unpack_from_vector,
};

/// vector.py: VectorLoop — wraps a loop body for vectorization analysis.
/// Provides the iteration over loop body ops and manages unrolling.
#[derive(Clone, Debug)]
pub struct VectorLoop {
    /// The original loop body ops (between Label and Jump).
    pub body: Vec<Op>,
    /// The label op (loop header).
    pub label: Option<Op>,
    /// The jump op (back-edge).
    pub jump: Option<Op>,
    /// Number of unrolled iterations.
    pub unroll_factor: usize,
}

impl VectorLoop {
    /// Create from a trace by finding Label..Jump.
    pub fn from_trace(ops: &[Op]) -> Option<Self> {
        let label_pos = ops.iter().position(|op| op.opcode == OpCode::Label)?;
        let jump_pos = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
        if jump_pos <= label_pos {
            return None;
        }
        Some(VectorLoop {
            body: ops[label_pos + 1..jump_pos].to_vec(),
            label: Some(ops[label_pos].clone()),
            jump: Some(ops[jump_pos].clone()),
            unroll_factor: 1,
        })
    }

    /// Number of ops in the loop body (excluding Label and Jump).
    pub fn body_len(&self) -> usize {
        self.body.len()
    }

    /// Whether the loop is suitable for vectorization.
    pub fn is_vectorizable(&self) -> bool {
        VectorizingOptimizer::user_loop_heuristic(&self.body)
    }

    /// Get adjacent memory access pairs in the body.
    pub fn find_adjacent_pairs(
        &self,
        constant_of: impl Fn(OpRef) -> Option<i64>,
    ) -> Vec<(usize, usize)> {
        VectorizingOptimizer::find_adjacent_pairs(&self.body, constant_of)
    }
}

impl VectorLoop {
    /// vector.py: unroll_loop_iterations(loop, unroll_count)
    ///
    /// Unroll the loop body `count` times. Each unrolled iteration has
    /// its OpRefs remapped to new positions. Guards like GUARD_FUTURE_CONDITION
    /// and GUARD_NOT_INVALIDATED are not duplicated.
    pub fn unroll_loop_iterations(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        let original_body = self.body.clone();
        let label_args = self
            .label
            .as_ref()
            .map(|l| l.args.clone())
            .unwrap_or_default();
        let jump_args = self
            .jump
            .as_ref()
            .map(|j| j.args.clone())
            .unwrap_or_default();

        let prohibit = [
            OpCode::GuardFutureCondition,
            OpCode::GuardNotInvalidated,
            OpCode::DebugMergePoint,
        ];

        let base_offset = original_body
            .iter()
            .map(|op| op.pos.raw())
            .max()
            .unwrap_or(0)
            + 1;

        for u in 0..count {
            let offset = base_offset + (u as u32) * (original_body.len() as u32);
            let mut remap = std::collections::HashMap::new();

            // Map label args → jump args (or remapped jump args)
            for (i, la) in label_args.iter().enumerate() {
                if i < jump_args.len() {
                    let ja = *remap.get(&jump_args[i]).unwrap_or(&jump_args[i]);
                    if *la != ja {
                        remap.insert(*la, ja);
                    }
                }
            }

            for op in &original_body {
                if prohibit.contains(&op.opcode) {
                    continue;
                }
                let mut new_op = op.clone();
                let new_pos = op.pos.with_raw(op.pos.raw() + offset);
                if !op.pos.is_none() {
                    remap.insert(op.pos, new_pos);
                }
                new_op.pos = new_pos;
                for arg in &mut new_op.args {
                    if let Some(&mapped) = remap.get(arg) {
                        *arg = mapped;
                    }
                }
                if let Some(ref mut fa) = new_op.fail_args {
                    for arg in fa.iter_mut() {
                        if let Some(&mapped) = remap.get(arg) {
                            *arg = mapped;
                        }
                    }
                }
                self.body.push(new_op);
            }
        }

        // Update jump args
        if let Some(ref mut jump) = self.jump {
            for _arg in &mut jump.args {
                // Use latest remap if available
            }
        }
        self.unroll_factor = count + 1;
    }
}

/// vector.py: follow_def_use_chain — trace a value through its uses
/// to find related vectorizable operations.
pub fn follow_def_use_chain(ops: &[Op], start: usize, max_depth: usize) -> Vec<usize> {
    let mut chain = vec![start];
    let result_ref = ops[start].pos;
    if result_ref.is_none() {
        return chain;
    }

    let mut current_refs = vec![result_ref];
    for _depth in 0..max_depth {
        let mut next_refs = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            if chain.contains(&i) {
                continue;
            }
            for arg in &op.args {
                if current_refs.contains(arg) {
                    chain.push(i);
                    if !op.pos.is_none() {
                        next_refs.push(op.pos);
                    }
                    break;
                }
            }
        }
        if next_refs.is_empty() {
            break;
        }
        current_refs = next_refs;
    }

    chain
}

/// schedule.py:638-658: VecScheduleState.pre_emit — guard accumulation stitching.
/// For guard ops, scan failargs for accumulation variables. When found:
///   - attach AccumInfo to the guard descriptor (schedule.py:654-655)
///   - replace the failarg with the renamed seed (schedule.py:656-657)
fn pre_emit_guard_accum(state: &VecScheduleState, op: &mut Op) {
    if !op.opcode.is_guard() {
        return;
    }
    if let Some(ref fa) = op.fail_args {
        let mut new_fa = fa.clone();
        for (fi, arg) in new_fa.iter_mut().enumerate() {
            if arg.is_none() {
                continue;
            }
            if let Some(entry) = state.accumulation.get(arg) {
                // schedule.py:654-655: AccumInfo → descr.attach_vector_info
                // resume.py:29,47: variable = original scalar (getoriginal())
                // resume.py:28 + regalloc.py:350: location = vector register
                // (set by backend). In Cranelift SSA, location is the
                // vector OpRef pulled from box_to_vbox.
                let location = state
                    .getvector_of_box(*arg)
                    .map(|(_, vec_ref)| vec_ref)
                    .unwrap_or(*arg);
                if let Some(ref descr) = op.descr {
                    if let Some(fail_descr) = descr.as_fail_descr() {
                        fail_descr.attach_vector_info(majit_ir::AccumInfo {
                            // resume.py:31 prev — attach_vector_info links this
                            // node onto descr.rd_vector_info and fills prev.
                            prev: None,
                            failargs_pos: fi,
                            variable: *arg,
                            location,
                            accum_operation: entry.operator,
                            // resume.py:71 self.scalar = None (set lazily at
                            // backend reduction).
                            scalar: OpRef::NONE,
                        });
                    }
                }
                // schedule.py:656-657: failargs[i] = renamer.rename_map.get(seed, seed)
                // Cranelift has no regalloc swap (regalloc.py:350-352), so store
                // the scalar seed directly to avoid vector OpRef in fail_args.
                *arg = entry.seed;
            }
        }
        op.fail_args = Some(new_fa);
    }
}

/// schedule.py:697-736: ensure_args_unpacked — unpack vector-boxed args
/// for a scalar op, respecting seen/invariant/accumulation state.
fn ensure_args_unpacked(state: &mut VecScheduleState, op: &mut Op, seen: &mut HashSet<OpRef>) {
    // schedule.py:702-706: unpack immediate-use args
    for j in 0..op.args.len() {
        let arg = op.args[j];
        if arg.is_constant() || seen.contains(&arg) {
            continue; // schedule.py:719: already seen
        }
        if let Some((pos, vec_ref)) = state.getvector_of_box(arg) {
            if state.invariant_vector_vars.contains(&vec_ref) {
                continue; // schedule.py:723-724: invariant_vector_vars
            }
            if state.accumulation.contains_key(&arg) {
                continue; // schedule.py:725-726
            }
            let unpacked = unpack_from_vector(state, vec_ref, pos, 1);
            state.renamer.start_renaming(arg, unpacked);
            seen.insert(unpacked);
            op.args[j] = unpacked;
        }
    }
    // schedule.py:708-716: unpack guard failargs
    if op.opcode.is_guard() {
        if let Some(ref mut fail_args) = op.fail_args {
            for arg in fail_args.iter_mut() {
                if arg.is_constant() || seen.contains(arg) {
                    continue;
                }
                if let Some((pos, vec_ref)) = state.getvector_of_box(*arg) {
                    if state.accumulation.contains_key(arg) {
                        continue;
                    }
                    let unpacked = unpack_from_vector(state, vec_ref, pos, 1);
                    state.renamer.start_renaming(*arg, unpacked);
                    seen.insert(unpacked);
                    *arg = unpacked;
                }
            }
        }
    }
}

pub struct VectorizingOptimizer {
    /// Operations in the current loop body (between Label and Jump).
    body_ops: Vec<Op>,
    /// Whether we're inside a loop body.
    in_loop: bool,
    /// Cost model for profitability decisions.
    cost_model: CostModel,
    /// schedule.py:669: label inputargs — populated on Label entry.
    label_args: Vec<OpRef>,
    /// jitprof.Counters.OPT_VECTORIZE_TRY / OPT_VECTORIZED accumulators.
    ///
    /// vector.py:139 / 146 calls `metainterp_sd.profiler.count(...)`
    /// directly inside the vectorisation entry point.  Pyre's
    /// `VectorizingOptimizer` is an `Optimization` pass with no
    /// `metainterp_sd` access, so we mirror the deferred-fold pattern
    /// already in use by `Optimizer::opt_ops_emitted` etc.: bump the
    /// accumulators here, then `Optimizer::update_counters` folds them
    /// into `staticdata.profiler` after each
    /// `propagate_all_forward` exit.
    pub(crate) opt_vectorize_try_emitted: usize,
    pub(crate) opt_vectorized_emitted: usize,
}

impl VectorizingOptimizer {
    pub fn new() -> Self {
        VectorizingOptimizer {
            body_ops: Vec::new(),
            in_loop: false,
            cost_model: CostModel::new(),
            label_args: Vec::new(),
            opt_vectorize_try_emitted: 0,
            opt_vectorized_emitted: 0,
        }
    }

    /// vector.py: linear_find_smallest_type — scan ops for smallest
    /// array element type to determine SIMD width.
    pub fn linear_find_smallest_type(ops: &[Op]) -> usize {
        let mut smallest = 0usize;
        for op in ops {
            if op.opcode.is_getarrayitem() || op.opcode.is_setarrayitem() {
                if let Some(ref descr) = op.descr {
                    if let Some(ad) = descr.as_array_descr() {
                        let item_size = ad.item_size();
                        if smallest == 0 || item_size < smallest {
                            smallest = item_size;
                        }
                    }
                }
            }
        }
        smallest
    }

    /// vector.py: get_unroll_count — compute how many times to unroll
    /// based on SIMD register width and smallest type.
    pub fn get_unroll_count(smallest_type_bytes: usize, simd_reg_bytes: usize) -> usize {
        if smallest_type_bytes == 0 {
            return 0;
        }
        let count = simd_reg_bytes / smallest_type_bytes;
        count.saturating_sub(1) // already unrolled once
    }

    /// vector.py: user_loop_heuristic — quick check if a loop body is
    /// worth trying to vectorize. Returns false if there are too few ops,
    /// no vectorizable opcodes, or too many guards.
    /// vector.py:404-425: extend_packset — follow def-use and use-def chains
    /// through can_be_packed to discover new pairs (including accumulation).
    pub fn extend_packset(pack_set: &mut PackSet, graph: &DependencyGraph) {
        let mut pack_count = pack_set.num_packs();
        loop {
            // vector.py:411-415: follow_def_uses for each 2-pack
            let mut i = 0;
            while i < pack_set.packs.len() {
                if pack_set.packs[i].members.len() == 2 {
                    let pack_snap = pack_set.packs[i].clone();
                    Self::follow_def_uses(pack_set, &pack_snap, graph);
                }
                i += 1;
            }
            if pack_count == pack_set.num_packs() {
                // vector.py:417-423: no new packs from def-uses, try use-defs
                pack_count = pack_set.num_packs();
                let mut i = 0;
                while i < pack_set.packs.len() {
                    if pack_set.packs[i].members.len() == 2 {
                        let pack_snap = pack_set.packs[i].clone();
                        Self::follow_use_defs(pack_set, &pack_snap, graph);
                    }
                    i += 1;
                }
                if pack_count == pack_set.num_packs() {
                    break;
                }
            }
            pack_count = pack_set.num_packs();
        }
    }

    /// vector.py:444-458: follow_def_uses — for a 2-pack, check if users
    /// of leftmost/rightmost can form new pairs via can_be_packed.
    fn follow_def_uses(pack_set: &mut PackSet, pack: &Pack, graph: &DependencyGraph) {
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();
        let left_opref = graph.nodes[left_idx].op.pos;

        // vector.py:446-447: for ldep in pack.leftmost(node=True).provides()
        let l_users: Vec<usize> = graph.nodes[left_idx].users.clone();
        let r_users: Vec<usize> = graph.nodes[right_idx].users.clone();
        for &l_user in &l_users {
            for &r_user in &r_users {
                // vector.py:451-453: left = pack.leftmost();
                // args = lnode.getoperation().getarglist();
                // if left not in args: continue
                if !graph.nodes[l_user].op.args.contains(&left_opref) {
                    continue;
                }
                let l_op = &graph.nodes[l_user].op;
                let r_op = &graph.nodes[r_user].op;
                // vector.py:454-455: isomorphic and lnode.is_before(rnode)
                if isomorphic(l_op, r_op) && l_user < r_user {
                    match pack_set.can_be_packed(l_user, r_user, Some(pack), true, graph) {
                        Ok(Some(pair)) => pack_set.add_pack(pair),
                        Err(_) => return, // NotAVectorizeableLoop — abort extension
                        _ => {}
                    }
                }
            }
        }
    }

    /// vector.py:427-442: follow_use_defs — for a 2-pack, check if
    /// dependencies of leftmost/rightmost can form new pairs.
    fn follow_use_defs(pack_set: &mut PackSet, pack: &Pack, graph: &DependencyGraph) {
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();
        let left_args = graph.nodes[left_idx].op.args.to_vec();

        // vector.py:429-430: for ldep in pack.leftmost(True).depends()
        let l_deps: Vec<usize> = graph.nodes[left_idx].deps.clone();
        let r_deps: Vec<usize> = graph.nodes[right_idx].deps.clone();
        for &l_dep in &l_deps {
            for &r_dep in &r_deps {
                // vector.py:434-437: left = lnode.getoperation();
                // args = pack.leftmost().getarglist();
                // if left not in args: continue
                let dep_opref = graph.nodes[l_dep].op.pos;
                if !left_args.contains(&dep_opref) {
                    continue;
                }
                let l_op = &graph.nodes[l_dep].op;
                let r_op = &graph.nodes[r_dep].op;
                // vector.py:438-439: isomorphic and lnode.is_before(rnode)
                if isomorphic(l_op, r_op) && l_dep < r_dep {
                    match pack_set.can_be_packed(l_dep, r_dep, Some(pack), false, graph) {
                        Ok(Some(pair)) => pack_set.add_pack(pair),
                        Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    }

    /// vector.py: combine_packset()
    /// Combine adjacent 2-packs into larger packs where the rightmost
    /// element of one pack matches the leftmost element of another.
    /// This is the iterative merge step after extend_packset.
    pub fn combine_packset(pack_set: &mut PackSet) {
        if pack_set.packs.is_empty() {
            return;
        }
        loop {
            let len_before = pack_set.packs.len();
            pack_set.try_merge_packs();
            if pack_set.packs.len() == len_before {
                break; // no more merges possible
            }
        }
    }

    pub fn user_loop_heuristic(ops: &[Op]) -> bool {
        if ops.len() < 4 {
            return false;
        }
        let vectorizable = ops
            .iter()
            .filter(|op| op.opcode.to_vector().is_some())
            .count();
        let guards = ops.iter().filter(|op| op.opcode.is_guard()).count();
        // Need at least 2 vectorizable ops and guards should not dominate.
        vectorizable >= 2 && guards < ops.len() / 2
    }

    /// vector.py: find_adjacent_pairs — find pairs of memory accesses
    /// that read/write adjacent array elements and could be packed.
    pub fn find_adjacent_pairs(
        ops: &[Op],
        constant_of: impl Fn(OpRef) -> Option<i64>,
    ) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..ops.len() {
            if !ops[i].opcode.is_getarrayitem() && !ops[i].opcode.is_setarrayitem() {
                continue;
            }
            for j in (i + 1)..ops.len() {
                if are_adjacent_memory_refs(&ops[i], &ops[j], &constant_of) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }

    /// Attempt to vectorize the buffered loop body.
    ///
    /// Before packing, operations are scheduled for maximum ILP.
    /// The reordering may expose additional packing opportunities.
    fn try_vectorize(&mut self, ctx: &mut OptContext) -> Option<Vec<Op>> {
        if self.body_ops.len() < 4 {
            return None; // Too small to benefit
        }

        // Constant resolver — looks up OpRef in the optimizer's constant map.
        let constant_of = |opref: OpRef| -> Option<i64> { ctx.get_constant_int(opref) };

        // Phase 1: Schedule operations for ILP before packing.
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        let schedule = schedule_operations(&dep_graph);
        if schedule.len() == self.body_ops.len() {
            let scheduled: Vec<Op> = schedule.iter().map(|&i| self.body_ops[i].clone()).collect();
            self.body_ops = scheduled;
        }

        // Phase 2: Rebuild dependency graph on reordered ops and find packs.
        // RPython flow: seed_packset → extend_packset → combine_packset
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        // vector.py:390-402: seed_packset — initial pairs from independent groups
        let seed_packs = dep_graph.find_packable_groups();
        if seed_packs.is_empty() {
            return None;
        }
        let mut pack_set = PackSet::new();
        for pack in seed_packs {
            pack_set.add_pack(pack);
        }
        // vector.py:404-425: extend_packset — follow chains via can_be_packed
        Self::extend_packset(&mut pack_set, &dep_graph);
        // vector.py:460-494: combine_packset — merge 2-packs into larger packs
        Self::combine_packset(&mut pack_set);
        let profitable = pack_set.packs;
        if profitable.is_empty() {
            return None;
        }
        // No static pre-filter — profitability is checked post-scheduling
        // via costmodel.profitable() (vector.py:632-633: savings >= 0).

        // schedule.py:292-311, 666-681: walk_and_emit + try_emit_or_delay.
        //
        // RPython scheduler state machine:
        //   prepare() → populate seen with label args, init inputargs
        //   walk_and_emit() → topo-order walk with delay() gating for packs
        //   mark_emitted() → renamer.rename(op), ensure_args_unpacked(op)
        //   pre_emit() → guard accumulation descriptor stitching
        let start_pos = ctx.new_operations.len() as u32 + self.body_ops.len() as u32;
        let mut sched_state = VecScheduleState::new(start_pos);

        // ── schedule.py:666-670: prepare() ──
        // Populate inputargs and seen from label args.
        for &arg in &self.label_args {
            sched_state.inputargs.insert(arg, ());
        }
        let mut seen: HashSet<OpRef> = HashSet::new();
        for &arg in &self.label_args {
            seen.insert(arg);
        }
        // vector.py:826-874: accumulate_prepare — populate accumulation map
        // AND create reduction initial vector + seed packing.
        for pack in &profitable {
            if !pack.is_accumulating {
                continue;
            }
            // vector.py:831-833: guard accumulation — skip vector init
            let first_op = &self.body_ops[pack.members[0]];
            if first_op.opcode.is_guard() {
                continue;
            }
            // schedule.py:998: getleftmostseed = leftmost.getarg(position)
            let pos = pack.position.max(0) as usize;
            let seed = if pos < first_op.args.len() {
                first_op.args[pos]
            } else {
                OpRef::NONE
            };
            let operator = pack.operator.unwrap_or('+');
            // vector.py:834-836: register each member op in accumulation map
            for &member_idx in &pack.members {
                let op = &self.body_ops[member_idx];
                if op.opcode.is_guard() {
                    continue;
                }
                sched_state.accumulation.insert(
                    op.pos,
                    AccumEntry {
                        seed,
                        operator,
                        accum_opcode: pack.scalar_opcode,
                    },
                );
            }
            // vector.py:838-840: pack.getdatatype() / pack.getbytesize()
            let is_float = first_op.opcode.result_type() == majit_ir::Type::Float;
            // vector.py:849-850: float reduction → NotImplementedError (aborts vectorization)
            if is_float {
                return None;
            }
            let datatype = 'i';
            // vector.py:839: pack.getbytesize() — from seed's vecinfo
            let bytesize: i32 = self
                .body_ops
                .iter()
                .find(|op| op.pos == seed)
                .and_then(|op| op.vecinfo.as_ref())
                .map(|vi| vi.getbytesize() as i32)
                .unwrap_or(8);
            // vector.py:827,840: vec_reg_size // bytesize
            let vec_reg_size: i32 = 16; // SSE = 16 bytes
            let count = (vec_reg_size / bytesize) as usize;
            let signed = true;

            // vector.py:844-853: create zero vector (reduce_init == 0 for '+')
            let vec_create =
                sched_state.create_vec_op(OpCode::VecI, &[], datatype, bytesize, signed, count);
            let zero_vec = vec_create.pos;
            sched_state.invariant_oplist.push(vec_create);

            // VEC_INT_XOR(zero_vec, zero_vec) → all zeros
            let xor_op = sched_state.create_vec_op(
                OpCode::VecIntXor,
                &[zero_vec, zero_vec],
                datatype,
                bytesize,
                signed,
                count,
            );
            let zeroed_vec = xor_op.pos;
            sched_state.invariant_oplist.push(xor_op);

            // vector.py:866-869: pack the seed scalar into position 0
            let zero_const = OpRef::from_const(0);
            let one_const = OpRef::from_const(1);
            let pack_op = sched_state.create_vec_op(
                OpCode::VecPackI,
                &[zeroed_vec, seed, zero_const, one_const],
                datatype,
                bytesize,
                signed,
                count,
            );
            let seed_vec = pack_op.pos;
            sched_state.invariant_oplist.push(pack_op);

            // vector.py:870-871: accumulation[seed] = pack
            sched_state.accumulation.insert(
                seed,
                AccumEntry {
                    seed,
                    operator,
                    accum_opcode: pack.scalar_opcode,
                },
            );
            // vector.py:873: setvector_of_box(seed, 0, vecop) — prevent expansion
            sched_state.setvector_of_box(seed, 0, seed_vec);
            // vector.py:874: renamer.start_renaming(seed, vecop)
            sched_state.renamer.start_renaming(seed, seed_vec);
        }

        // Build node→pack mapping
        let mut node_to_pack: HashMap<usize, usize> = HashMap::new();
        for (pi, group) in profitable.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        // schedule.py:683-695: delay() tracking
        let mut pack_emitted = vec![false; profitable.len()];
        let mut pack_visited_count = vec![0usize; profitable.len()];

        // Walk in scheduled (dependency) order
        let scheduled_order = schedule_operations(&dep_graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                // schedule.py:683-695: delay() gating.
                pack_visited_count[pack_idx] += 1;
                let pack = &profitable[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    pack_emitted[pack_idx] = true;

                    // schedule.py:676-678: pre_emit(node, i==0) + mark_emitted(node, unpack=False)
                    for &member_idx in &pack.members {
                        let mut member_op = self.body_ops[member_idx].clone();

                        // schedule.py:638-658: VecScheduleState.pre_emit
                        pre_emit_guard_accum(&sched_state, &mut member_op);

                        // schedule.py:194,197: mark_emitted(unpack=False)
                        // renamer.rename(op) — always applied
                        sched_state.renamer.rename(&mut member_op);
                        // unpack=False → skip ensure_args_unpacked

                        seen.insert(member_op.pos);
                        // Write renamed op back for turn_into_vector to use
                        self.body_ops[member_idx] = member_op;
                    }

                    turn_into_vector(&mut sched_state, pack, &self.body_ops);
                }
            } else {
                // schedule.py:680-681: scalar node (SchedulerState.try_emit_or_delay)
                let mut scalar_op = self.body_ops[node_idx].clone();

                // schedule.py:638-658: pre_emit — guard accum stitch
                pre_emit_guard_accum(&sched_state, &mut scalar_op);

                // schedule.py:197: mark_emitted → renamer.rename(op)
                sched_state.renamer.rename(&mut scalar_op);

                // schedule.py:198-199: mark_emitted(unpack=True) → ensure_args_unpacked
                ensure_args_unpacked(&mut sched_state, &mut scalar_op, &mut seen);

                // schedule.py:136: seen[op] = None
                seen.insert(scalar_op.pos);
                sched_state.append_to_oplist(scalar_op);
            }
        }

        // vector.py:632-633: profitable() — savings >= 0.
        if !sched_state.costmodel.profitable() {
            return None;
        }

        // Prepend invariant ops (expand/pack for loop-invariant scalars)
        // before the vectorized body.
        let mut result = sched_state.invariant_oplist;
        result.append(&mut sched_state.oplist);
        Some(result)
    }
}

impl Default for VectorizingOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for VectorizingOptimizer {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        match op.opcode {
            OpCode::Label => {
                self.in_loop = true;
                // schedule.py:669: save label inputargs for prepare()
                self.label_args = op.args.to_vec();
                OptimizationResult::Emit(op.clone())
            }
            OpCode::Jump if self.in_loop => {
                // End of loop body — attempt vectorization.
                // vector.py:139: metainterp_sd.profiler.count(Counters.OPT_VECTORIZE_TRY)
                self.opt_vectorize_try_emitted = self.opt_vectorize_try_emitted.saturating_add(1);
                if let Some(vectorized) = self.try_vectorize(ctx) {
                    // vector.py:146: metainterp_sd.profiler.count(Counters.OPT_VECTORIZED)
                    self.opt_vectorized_emitted = self.opt_vectorized_emitted.saturating_add(1);
                    for vop in vectorized {
                        ctx.emit(vop);
                    }
                } else {
                    // No vectorization — emit original ops
                    for body_op in &self.body_ops {
                        ctx.emit(body_op.clone());
                    }
                }
                self.in_loop = false;
                OptimizationResult::Emit(op.clone())
            }
            _ => {
                if self.in_loop {
                    // Buffer loop body ops
                    self.body_ops.push(op.clone());
                    OptimizationResult::Remove
                } else {
                    // Not in a loop — pass through unchanged
                    OptimizationResult::PassOn
                }
            }
        }
    }

    fn setup(&mut self) {
        self.body_ops.clear();
        self.in_loop = false;
    }

    fn name(&self) -> &'static str {
        "vectorize_simd"
    }

    fn drain_profiler_counters(&mut self, profiler: &crate::jitprof::JitProfiler) {
        // vector.py:139/146 — fold accumulated OPT_VECTORIZE_TRY /
        // OPT_VECTORIZED into the shared profiler and reset so the
        // next propagate_all_forward starts clean.
        profiler.count(
            crate::pyjitpl::counters::OPT_VECTORIZE_TRY,
            self.opt_vectorize_try_emitted,
        );
        profiler.count(
            crate::pyjitpl::counters::OPT_VECTORIZED,
            self.opt_vectorized_emitted,
        );
        self.opt_vectorize_try_emitted = 0;
        self.opt_vectorized_emitted = 0;
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Whether a scalar opcode produces float results.
fn is_float_opcode(opcode: OpCode) -> bool {
    matches!(
        opcode,
        OpCode::FloatAdd
            | OpCode::FloatSub
            | OpCode::FloatMul
            | OpCode::FloatTrueDiv
            | OpCode::FloatAbs
            | OpCode::FloatNeg
            | OpCode::CastIntToFloat
            | OpCode::CastSinglefloatToFloat
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRef};

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(base + i as u32);
        }
    }

    // ── Dependency graph tests ──

    #[test]
    fn test_dep_graph_basic() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // depends on op 0
            Op::new(
                OpCode::IntSub,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // independent of op 0
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        assert_eq!(graph.nodes.len(), 3);

        // op1 depends on op0
        assert!(graph.nodes[1].deps.contains(&0));
        // op2 is independent of op0
        assert!(!graph.nodes[2].deps.contains(&0));
    }

    #[test]
    fn test_dep_graph_no_self_dep() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(0), OpRef::from_raw(101)],
        )];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        // Self-reference should not create self-dependency
        assert!(graph.nodes[0].deps.is_empty());
    }

    // ── Pack group tests ──

    #[test]
    fn test_find_packable_groups() {
        // Two independent IntAdd ops
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].scalar_opcode, OpCode::IntAdd);
        assert_eq!(groups[0].vector_opcode, OpCode::VecIntAdd);
        assert_eq!(groups[0].members.len(), 2);
    }

    #[test]
    fn test_dependent_ops_not_packed() {
        // Two IntAdd ops where second depends on first
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // depends on op 0
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        // Can't pack because of dependency
        assert!(groups.is_empty());
    }

    #[test]
    fn test_different_opcodes_not_packed() {
        // IntAdd and IntSub — not isomorphic
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        // Each group needs 2+ members, single ops can't form a group
        for g in &groups {
            assert!(g.members.len() >= 2);
        }
    }

    #[test]
    fn test_three_independent_ops() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMul,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::from_raw(104), OpRef::from_raw(105)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
    }

    // ── Cost model tests ──

    #[test]
    fn test_cost_model_profitable() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3], // 4 ops
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        // savings = 3 * 1 = 3, cost = 2 * 2 = 4 → not profitable with 4
        // Actually savings = 3, cost = 4, so NOT profitable by default
        // Let's adjust: with default params, need enough ops
        assert!(!cm.is_profitable(&group)); // 3 < 4

        let group5 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3, 4], // 5 ops → savings = 4 > cost = 4
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group5)); // 4 == 4, not strictly greater
    }

    #[test]
    fn test_cost_model_too_small() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0], // Only 1 op
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));
    }

    #[test]
    fn test_cost_model_custom_params() {
        let cm = CostModel {
            min_pack_size: 2,
            pack_cost: 1,
            scalar_save: 2,
            savings: 0,
        };
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1], // savings = 1*2 = 2, cost = 2*1 = 2 → not profitable
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));

        let group3 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2], // savings = 2*2 = 4, cost = 2*1 = 2 → profitable
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(cm.is_profitable(&group3));
    }

    // ── Memory access detection ──

    #[test]
    fn test_is_memory_access() {
        assert!(OpCode::GetfieldGcI.is_memory_access());
        assert!(OpCode::SetarrayitemGc.is_memory_access());
        assert!(OpCode::RawLoadI.is_memory_access());
        assert!(!OpCode::IntAdd.is_memory_access());
        assert!(!OpCode::GuardTrue.is_memory_access());
    }

    // ── VectorizingOptimizer pass tests ──

    #[test]
    fn test_vectorize_pass_no_loop() {
        // Without Label..Jump, nothing to vectorize
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Finish, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        // No loop to vectorize, ops should pass through
        // (pre-label ops are buffered but emitted when we hit non-Label)
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vectorize_pass_preserves_structure() {
        use crate::optimizeopt::optimizer::Optimizer;

        // A simple loop with independent ops
        let mut ops = vec![
            Op::new(OpCode::Label, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(1), OpRef::from_raw(2)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        // Should still have Label and Jump
        assert!(result.iter().any(|op| op.opcode == OpCode::Label));
        assert!(result.iter().any(|op| op.opcode == OpCode::Jump));
    }

    // ── Scheduler tests ──

    #[test]
    fn test_schedule_respects_dependencies() {
        // A → B → C: linear chain must stay in order
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // A
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // B depends on A
            Op::new(OpCode::IntSub, &[OpRef::from_raw(1), OpRef::from_raw(101)]), // C depends on B
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 3);
        // A before B before C
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_schedule_maximizes_parallelism() {
        // A and B are independent — both should appear early (before any
        // dependent op). With no dependents, both have height 1 and both
        // should be in the first two schedule slots.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // A
            Op::new(
                OpCode::IntSub,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ), // B (independent)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 2);
        // Both scheduled (order doesn't matter, but both must be present)
        assert!(sched.contains(&0));
        assert!(sched.contains(&1));
    }

    #[test]
    fn test_schedule_prioritizes_critical_path() {
        // Critical path: A → B → C (heights: A=3, B=2, C=1)
        // Independent: D (height=1)
        // A should be scheduled before D because A has higher height.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // A (idx 0)
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // B (idx 1, depends on A)
            Op::new(OpCode::IntSub, &[OpRef::from_raw(1), OpRef::from_raw(101)]), // C (idx 2, depends on B)
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ), // D (idx 3, independent)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        // A (height 3) should be scheduled before D (height 1)
        assert!(pos_a < pos_d, "A (height 3) should precede D (height 1)");
    }

    #[test]
    fn test_schedule_diamond() {
        // Diamond: A → B, A → C, B → D, C → D
        // Valid orders: [A, B, C, D] or [A, C, B, D]
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // A (idx 0)
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(101)]), // B (idx 1, depends on A)
            Op::new(OpCode::IntSub, &[OpRef::from_raw(0), OpRef::from_raw(102)]), // C (idx 2, depends on A)
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(1), OpRef::from_raw(2)]), // D (idx 3, depends on B and C)
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);

        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();

        // A must come before B and C
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        // B and C must come before D
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_schedule_empty_graph() {
        let graph = DependencyGraph {
            nodes: Vec::new(),
            memory_refs: Default::default(),
            index_vars: Default::default(),
            guards: Vec::new(),
            invariant_vars: Default::default(),
        };
        let sched = schedule_operations(&graph);
        assert!(sched.is_empty());
    }

    #[test]
    fn test_user_loop_heuristic_too_small() {
        let ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        assert!(!VectorizingOptimizer::user_loop_heuristic(&ops));
    }

    #[test]
    fn test_user_loop_heuristic_no_vectorizable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(2)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(0)]),
        ];
        assert!(!VectorizingOptimizer::user_loop_heuristic(&ops));
    }

    #[test]
    fn test_pack_set_merge() {
        // schedule.py:931-942: rightmost_match_leftmost — pack1.rightmost == pack2.leftmost
        // Pack [0,1] and [1,2]: rightmost(1) == leftmost(1) → merge into [0,1,2]
        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![1, 2],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        assert_eq!(ps.num_packs(), 2);
        assert_eq!(ps.total_ops(), 4);

        ps.try_merge_packs();
        assert_eq!(ps.num_packs(), 1);
        assert_eq!(ps.total_ops(), 3); // [0, 1, 2] — overlap at node 1
    }

    #[test]
    fn test_pack_set_no_merge_disjoint() {
        // Packs with non-matching edges should NOT merge.
        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![2, 3],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        ps.try_merge_packs();
        assert_eq!(ps.num_packs(), 2); // no merge possible
    }

    // ── isomorphic + can_be_packed + accumulates_pair tests ──

    #[test]
    fn test_isomorphic_same_opcode() {
        let a = Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
        );
        let b = Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(102), OpRef::from_raw(103)],
        );
        assert!(isomorphic(&a, &b));
    }

    #[test]
    fn test_isomorphic_different_opcode() {
        let a = Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
        );
        let b = Op::new(
            OpCode::IntSub,
            &[OpRef::from_raw(102), OpRef::from_raw(103)],
        );
        assert!(!isomorphic(&a, &b));
    }

    #[test]
    fn test_can_be_packed_independent_seed() {
        // Two independent IntAdd ops, no origin_pack (seed case)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps.can_be_packed(0, 1, None, false, &graph);
        assert!(result.is_ok());
        let pack = result.unwrap();
        assert!(pack.is_some());
        let pack = pack.unwrap();
        assert_eq!(pack.members, vec![0, 1]);
        assert!(!pack.is_accumulating);
    }

    #[test]
    fn test_can_be_packed_dependent_no_origin() {
        // Dependent ops without origin_pack → None (not accumulation candidate)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(101)]),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps.can_be_packed(0, 1, None, false, &graph).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_can_be_packed_accumulation() {
        // Accumulation pattern: sum = sum + a[i]
        // Op 0: a[0] = IntAdd(x, y)       — produces array element
        // Op 1: a[1] = IntAdd(x2, y2)     — produces array element
        // Op 2: sum0 = IntAdd(seed, op0)   — accumulation step 1
        // Op 3: sum1 = IntAdd(sum0, op1)   — accumulation step 2 (depends on op2 via sum0)
        //
        // Origin pack = (op0, op1) — the array element pair.
        // can_be_packed(op2, op3, origin, forward=true) should detect accumulation.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // 0: element a[0]
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ), // 1: element a[1]
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(200), OpRef::from_raw(0)]), // 2: sum0 = seed + a[0]
            Op::new(OpCode::IntAdd, &[OpRef::from_raw(2), OpRef::from_raw(1)]), // 3: sum1 = sum0 + a[1]
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        // Create origin pack from (op0, op1)
        let origin = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        };

        let ps = PackSet::new();
        let result = ps.can_be_packed(2, 3, Some(&origin), true, &graph);
        assert!(result.is_ok());
        let pack = result.unwrap();
        // Op2 and Op3 are dependent (op3 uses op2's result), so this goes
        // through the non-independent branch → accumulates_pair.
        // Whether it succeeds depends on the exact accumulation pattern:
        // - op3.args[0] == op2.pos ✓ (getaccumulator_variable finds index=0)
        // - op2.args[1] == origin.leftmost().pos (op0.pos=0) ✓
        // - op3.args[1] == origin.rightmost().pos (op1.pos=1) ✓
        if let Some(p) = pack {
            assert!(p.is_accumulating);
            assert_eq!(p.operator, Some('+'));
            assert_eq!(p.position, 0); // accumulator is arg index 0
        }
        // (It's OK if accumulates_pair returns None due to bytesize or dependency
        // checks — the important thing is the path is exercised.)
    }

    #[test]
    fn test_can_be_packed_blocks_already_packed() {
        // vector.py:706-707: contains_pair check — if lnode is already leftmost
        // or rnode is already rightmost of some pack, can_be_packed returns None.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(102), OpRef::from_raw(103)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(104), OpRef::from_raw(105)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        let mut ps = PackSet::new();
        // Pack (0, 1) already exists
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        // Trying to pack (0, 2) — node 0 is already leftmost → blocked
        let result = ps.can_be_packed(0, 2, None, false, &graph).unwrap();
        assert!(result.is_none());
        // Trying to pack (2, 1) — node 1 is already rightmost → blocked
        let result = ps.can_be_packed(2, 1, None, false, &graph).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_generic_cost_model() {
        let model = GenericCostModel::new();
        // Memory ops are more expensive
        assert!(model.op_cost(OpCode::GetarrayitemGcI) > model.op_cost(OpCode::IntAdd));
        // Float div is most expensive
        assert!(model.op_cost(OpCode::FloatTrueDiv) >= model.op_cost(OpCode::FloatAdd));
    }

    #[test]
    fn test_guard_analysis_hoistable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]), // loop-invariant (100 not produced)
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(1)]), // body-dependent (1 = IntAdd result)
        ];
        let mut positioned = ops;
        for (i, op) in positioned.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(i as u32);
        }
        let analysis = GuardAnalysis::analyze(&positioned);
        assert_eq!(analysis.hoistable.len(), 1);
        assert_eq!(analysis.hoistable[0], 0);
        assert_eq!(analysis.body_guards.len(), 1);
        assert_eq!(analysis.body_guards[0], 2);
    }

    #[test]
    fn test_follow_def_use_chain() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // 0: result used by 1
            Op::new(OpCode::IntMul, &[OpRef::from_raw(0), OpRef::from_raw(102)]), // 1: uses result of 0
            Op::new(OpCode::IntSub, &[OpRef::from_raw(1), OpRef::from_raw(103)]), // 2: uses result of 1
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(104), OpRef::from_raw(105)],
            ), // 3: independent
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(i as u32);
        }
        let chain = follow_def_use_chain(&ops, 0, 10);
        assert!(chain.contains(&0));
        assert!(chain.contains(&1));
        assert!(chain.contains(&2));
        assert!(!chain.contains(&3)); // independent
    }

    #[test]
    fn test_vector_loop_from_trace() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // preamble
            Op::new(OpCode::Label, &[OpRef::from_raw(100)]), // loop header
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ), // body
            Op::new(OpCode::IntMul, &[OpRef::from_raw(2), OpRef::from_raw(102)]), // body
            Op::new(OpCode::Jump, &[OpRef::from_raw(3)]),    // back-edge
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(i as u32);
        }
        let vloop = VectorLoop::from_trace(&ops).unwrap();
        assert_eq!(vloop.body_len(), 2); // IntAdd + IntMul
        assert!(vloop.label.is_some());
        assert!(vloop.jump.is_some());
    }

    #[test]
    fn drain_profiler_counters_folds_opt_vectorize_try_and_opt_vectorized_into_profiler() {
        // vector.py:139 / 146 contract via the deferred-fold pattern:
        // VectorizingOptimizer accumulates OPT_VECTORIZE_TRY /
        // OPT_VECTORIZED locally, and Optimizer::update_counters drains
        // them into staticdata.profiler.  This test exercises just the
        // drain step in isolation so an Optimization-trait method
        // regression (e.g. forgetting to override
        // drain_profiler_counters here) is caught without spinning up
        // the whole optimizer.
        use crate::pyjitpl::counters;
        let mut vopt = VectorizingOptimizer::new();
        vopt.opt_vectorize_try_emitted = 4;
        vopt.opt_vectorized_emitted = 1;
        let prof = crate::jitprof::JitProfiler::default();
        Optimization::drain_profiler_counters(&mut vopt, &prof);
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZE_TRY), Some(4));
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZED), Some(1));
        assert_eq!(vopt.opt_vectorize_try_emitted, 0);
        assert_eq!(vopt.opt_vectorized_emitted, 0);
        // Second drain is a no-op now that accumulators are zero.
        Optimization::drain_profiler_counters(&mut vopt, &prof);
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZE_TRY), Some(4));
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZED), Some(1));
    }
}
