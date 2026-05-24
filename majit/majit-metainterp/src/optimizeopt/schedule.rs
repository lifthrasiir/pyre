//! Instruction scheduling and pack management for vectorization.
//!
//! Mirrors RPython's `schedule.py` and `costmodel.py`: pack groups,
//! pack sets, accumulation tracking, guard analysis, and cost models.

use majit_ir::{Op, OpCode, OpRef, Type};

use crate::r#box::BoxRef;
use crate::optimizeopt::dependency::DependencyGraph;

// ── vector.py:670-678: isomorphic ─────────────────────────────────────

/// vector.py:670-678 helper — returns the bytesize from the op's vecinfo
/// slot if present, otherwise the default for the op's result type.
/// INT_WORD = 8, FLOAT_WORD = 8 on 64-bit; void → 0.
fn get_op_bytesize(op: &Op) -> i32 {
    if let Some(vi) = op.get_vecinfo() {
        return vi.getbytesize() as i32;
    }
    match op.opcode.result_type() {
        majit_ir::Type::Float => 8,
        majit_ir::Type::Void => 0,
        _ => 8, // INT_WORD on 64-bit
    }
}

/// vector.py:670-678: isomorphic — two ops can be packed if they have
/// the same opcode AND the same vecinfo bytesize.
pub fn isomorphic(l_op: &Op, r_op: &Op) -> bool {
    if l_op.opcode != r_op.opcode {
        return false;
    }
    get_op_bytesize(l_op) == get_op_bytesize(r_op)
}

/// vector.py:670-678 production path. Uses `forwarded_vecinfo`, which stores
/// `VectorizationInfo` in the same `_forwarded` slot that RPython uses.
pub fn isomorphic_with_state(state: &mut VecScheduleState, l_op: &Op, r_op: &Op) -> bool {
    if l_op.opcode != r_op.opcode {
        return false;
    }
    let l_vecinfo = state.forwarded_vecinfo(l_op);
    let r_vecinfo = state.forwarded_vecinfo(r_op);
    l_vecinfo.bytesize == r_vecinfo.bytesize
}

/// schedule.py:781+: A pack is a set of n isomorphic operations that can
/// execute as a single SIMD instruction.
#[derive(Clone, Debug)]
pub struct Pack {
    /// The scalar opcode of the group members.
    pub scalar_opcode: OpCode,
    /// The vector opcode to replace them with.
    pub vector_opcode: OpCode,
    /// Indices into the DepGraph nodes.
    pub members: Vec<usize>,
    /// schedule.py:811: whether this pack tracks an accumulation (reduction).
    pub is_accumulating: bool,
    /// schedule.py:989: accumulation argument position (-1 = none).
    pub position: i32,
    /// schedule.py:981: AccumPack.operator — '+' for INT_ADD, 'g' for guard, None for normal.
    pub operator: Option<char>,
}

/// vector.py: PackSet — manages packs and supports merging
/// 2-packs into 4-packs (or larger) when possible.
#[derive(Clone, Debug, Default)]
pub struct PackSet {
    /// All packs found so far.
    pub packs: Vec<Pack>,
}

impl PackSet {
    pub fn new() -> Self {
        PackSet { packs: Vec::new() }
    }

    /// Add a pack to the set.
    pub fn add_pack(&mut self, pack: Pack) {
        self.packs.push(pack);
    }

    /// vector.py:460-494: combine_packset — merge packs where
    /// pack1.rightmost == pack2.leftmost (schedule.py:931-942).
    /// Only merges packs with matching edge, NOT just same opcode.
    pub fn try_merge_packs(&mut self) {
        loop {
            let len_before = self.packs.len();
            let mut i = 0;
            while i < self.packs.len() {
                let mut j = 0;
                while j < self.packs.len() {
                    if i == j {
                        j += 1;
                        continue;
                    }
                    if i < self.packs.len() && j < self.packs.len() {
                        // schedule.py:931-942: rightmost_match_leftmost
                        let rightmost = *self.packs[i].members.last().unwrap_or(&usize::MAX);
                        let leftmost = *self.packs[j].members.first().unwrap_or(&usize::MAX);
                        // schedule.py:937-941: accumulating pack constraints
                        let accum_ok = if self.packs[i].is_accumulating {
                            self.packs[j].is_accumulating
                                && self.packs[i].position == self.packs[j].position
                        } else {
                            true
                        };
                        if rightmost == leftmost
                            && self.packs[i].scalar_opcode == self.packs[j].scalar_opcode
                            && accum_ok
                        {
                            // vector.py:753+: combine — merge j into i, skip overlap
                            let mut merged_members = self.packs[i].members.clone();
                            merged_members.extend_from_slice(&self.packs[j].members[1..]);
                            self.packs[i].members = merged_members;
                            self.packs.remove(j);
                            if j < i {
                                i -= 1;
                            }
                            continue; // re-check from j
                        }
                    }
                    j += 1;
                }
                i += 1;
            }
            if self.packs.len() == len_before {
                break;
            }
        }
    }

    /// vector.py: extend_packset()
    ///
    /// Follow dependency chains to find more candidates to put into pairs.
    /// For each existing pack, check if the users (def→use) or producers
    /// (use→def) of the packed ops can also form isomorphic pairs.
    pub fn extend_packset(&mut self, graph: &DependencyGraph) {
        loop {
            let count_before = self.packs.len();
            let num_packs = self.packs.len();
            for pi in 0..num_packs {
                if self.packs[pi].members.len() < 2 {
                    continue;
                }
                let left = self.packs[pi].members[0];
                let right = self.packs[pi].members[1];
                // follow_def_uses: users of left/right that are isomorphic
                for &uleft in &graph.nodes[left].users {
                    for &uright in &graph.nodes[right].users {
                        if uleft < uright
                            && graph.nodes[uleft].op.opcode == graph.nodes[uright].op.opcode
                            && !self.already_packed(uleft)
                            && !self.already_packed(uright)
                        {
                            let sc = graph.nodes[uleft].op.opcode;
                            self.packs.push(Pack {
                                scalar_opcode: sc,
                                vector_opcode: sc.to_vector().unwrap_or(sc),
                                members: vec![uleft, uright],
                                is_accumulating: false,
                                position: -1,
                                operator: None,
                            });
                        }
                    }
                }
                // follow_use_defs: deps of left/right that are isomorphic
                for &dleft in &graph.nodes[left].deps {
                    for &dright in &graph.nodes[right].deps {
                        if dleft < dright
                            && graph.nodes[dleft].op.opcode == graph.nodes[dright].op.opcode
                            && !self.already_packed(dleft)
                            && !self.already_packed(dright)
                        {
                            let sc = graph.nodes[dleft].op.opcode;
                            self.packs.push(Pack {
                                scalar_opcode: sc,
                                vector_opcode: sc.to_vector().unwrap_or(sc),
                                members: vec![dleft, dright],
                                is_accumulating: false,
                                position: -1,
                                operator: None,
                            });
                        }
                    }
                }
            }
            if self.packs.len() == count_before {
                break;
            }
        }
    }

    /// Check if an op index is already in some pack.
    fn already_packed(&self, idx: usize) -> bool {
        self.packs.iter().any(|p| p.members.contains(&idx))
    }

    /// vector.py: combine_packset()
    ///
    /// Combine packs that share edges: if pack1.rightmost == pack2.leftmost,
    /// merge them into a longer pack. Iterates until stable.
    pub fn combine_packset(&mut self) {
        loop {
            let len_before = self.packs.len();
            let mut i = 0;
            while i < self.packs.len() {
                let mut j = 0;
                while j < self.packs.len() {
                    if i == j {
                        j += 1;
                        continue;
                    }
                    if i < self.packs.len() && j < self.packs.len() {
                        let right_of_i = *self.packs[i].members.last().unwrap_or(&usize::MAX);
                        let left_of_j = *self.packs[j].members.first().unwrap_or(&usize::MAX);
                        if right_of_i == left_of_j
                            && self.packs[i].scalar_opcode == self.packs[j].scalar_opcode
                        {
                            // Merge j into i
                            let mut merged_members = self.packs[i].members.clone();
                            merged_members.extend_from_slice(&self.packs[j].members[1..]);
                            self.packs[i].members = merged_members;
                            self.packs.remove(j);
                            if j < i {
                                i -= 1;
                            }
                            continue; // re-check from j
                        }
                    }
                    j += 1;
                }
                i += 1;
            }
            if self.packs.len() == len_before {
                break;
            }
        }
    }

    /// Number of packs.
    pub fn num_packs(&self) -> usize {
        self.packs.len()
    }

    /// Total number of ops across all packs.
    pub fn total_ops(&self) -> usize {
        self.packs.iter().map(|p| p.members.len()).sum()
    }

    // ── vector.py:692-824: can_be_packed + accumulation detection ──

    /// vector.py:692-722: can_be_packed — central pack eligibility check.
    /// Returns Ok(Some(Pack)) if lnode and rnode can form a pair.
    /// Returns Err(NotAVectorizeableLoop) if vectorization must abort.
    pub fn can_be_packed(
        &self,
        state: &mut VecScheduleState,
        lnode: usize,
        rnode: usize,
        origin_pack: Option<&Pack>,
        forward: bool,
        graph: &DependencyGraph,
    ) -> Result<Option<Pack>, NotAVectorizeableLoop> {
        let l_op = &graph.nodes[lnode].op;
        let r_op = &graph.nodes[rnode].op;

        if !isomorphic_with_state(state, l_op, r_op) {
            return Ok(None);
        }

        // vector.py:698: lnode.independent(rnode)
        let independent = !graph.has_dependency(lnode, rnode);

        if independent {
            // vector.py:700-704: forward + accumulating origin → abort
            if let Some(op) = origin_pack {
                if forward && op.is_accumulating {
                    return Err(NotAVectorizeableLoop);
                }
            }
            // vector.py:706-707
            if self.contains_pair(lnode, rnode) {
                return Ok(None);
            }
            // vector.py:709-714: no origin → seed pair
            if origin_pack.is_none() {
                let vec_op = match l_op.opcode.to_vector() {
                    Some(v) => v,
                    None => return Ok(None),
                };
                return Ok(Some(Pack {
                    scalar_opcode: l_op.opcode,
                    vector_opcode: vec_op,
                    members: vec![lnode, rnode],
                    is_accumulating: false,
                    position: -1,
                    operator: None,
                }));
            }
            // vector.py:715-716: profitable_pack check
            let origin = origin_pack.unwrap();
            if self.profitable_pack(lnode, rnode, origin, forward, graph) {
                let vec_op = match l_op.opcode.to_vector() {
                    Some(v) => v,
                    None => return Ok(None),
                };
                return Ok(Some(Pack {
                    scalar_opcode: l_op.opcode,
                    vector_opcode: vec_op,
                    members: vec![lnode, rnode],
                    is_accumulating: false,
                    position: -1,
                    operator: None,
                }));
            }
        } else {
            // vector.py:717-721: non-independent branch (accumulation)
            if self.contains_pair(lnode, rnode) {
                return Ok(None);
            }
            if origin_pack.is_some() {
                return Ok(self.accumulates_pair(state, lnode, rnode, origin_pack.unwrap(), graph));
            }
        }

        Ok(None)
    }

    /// vector.py:724-729: contains_pair — check if lnode or rnode is already
    /// the leftmost or rightmost of some existing pack.
    fn contains_pair(&self, lnode: usize, rnode: usize) -> bool {
        for pack in &self.packs {
            if pack.members.first() == Some(&lnode) || pack.members.last() == Some(&rnode) {
                return true;
            }
        }
        false
    }

    /// vector.py:731-738: profitable_pack
    fn profitable_pack(
        &self,
        lnode: usize,
        rnode: usize,
        origin_pack: &Pack,
        forward: bool,
        graph: &DependencyGraph,
    ) -> bool {
        let l_op = &graph.nodes[lnode].op;
        let r_op = &graph.nodes[rnode].op;
        let origin_left_op = &graph.nodes[origin_pack.members[0]].op;
        let origin_right_op = &graph.nodes[*origin_pack.members.last().unwrap()].op;

        if Self::prohibit_packing(origin_left_op, l_op, forward) {
            return false;
        }
        if Self::prohibit_packing(origin_right_op, r_op, forward) {
            return false;
        }
        true
    }

    /// vector.py:740-751: prohibit_packing — block certain op combinations.
    fn prohibit_packing(packed: &Op, inquestion: &Op, forward: bool) -> bool {
        // vector.py:742: inquestion.vector == -1
        // In RPython, resoperation._vector == -1 marks non-vectorizable ops.
        // In majit, this is equivalent to to_vector() returning None.
        if inquestion.opcode.to_vector().is_none() {
            return true;
        }
        // vector.py:744-746: packed is array access and its index arg is inquestion
        if (packed.opcode.is_getarrayitem()
            || packed.opcode.is_setarrayitem()
            || matches!(
                packed.opcode,
                OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore
            ))
            && packed.num_args() >= 2
            && packed.arg(1) == inquestion.pos.get()
        {
            return true;
        }
        // vector.py:747-750: INT_SIGNEXT backward prohibition
        if !forward && packed.opcode == OpCode::IntSignext {
            return true;
        }
        false
    }

    /// vector.py:766-818: accumulates_pair — detect accumulation pattern
    /// between two isomorphic, dependent nodes.
    fn accumulates_pair(
        &self,
        state: &mut VecScheduleState,
        lnode: usize,
        rnode: usize,
        origin_pack: &Pack,
        graph: &DependencyGraph,
    ) -> Option<Pack> {
        let left = &graph.nodes[lnode].op;
        let opnum = left.opcode;

        // vector.py:772-774: AccumPack.SUPPORTED = { INT_ADD: '+' }
        let operator = match opnum {
            OpCode::IntAdd => '+',
            _ => {
                // vector.py:814-816: guard accumulation (GUARD_TRUE/GUARD_FALSE)
                if left.opcode.is_guard()
                    && matches!(left.opcode, OpCode::GuardTrue | OpCode::GuardFalse)
                {
                    let vec_op = left.opcode.to_vector().unwrap_or(left.opcode);
                    return Some(Pack {
                        scalar_opcode: left.opcode,
                        vector_opcode: vec_op,
                        members: vec![lnode, rnode],
                        is_accumulating: true,
                        position: 0,
                        operator: Some('g'),
                    });
                }
                return None;
            }
        };

        let right = &graph.nodes[rnode].op;

        // vector.py:778: assert left.numargs() == 2 and not left.returns_void()
        if left.num_args() != 2 || left.opcode.result_type() == majit_ir::Type::Void {
            return None;
        }

        // vector.py:779: scalar, index = self.getaccumulator_variable(left, right, origin_pack)
        let (scalar, index) = Self::getaccumulator_variable(left, right);
        if scalar.is_none() {
            return None;
        }
        let index = index as usize;

        // vector.py:782-787: dependency only because of the scalar?
        for dep in &graph.nodes[lnode].adjacent_list {
            if dep.to_idx == rnode && !dep.because_of(left.pos.get()) {
                return None;
            }
        }

        // vector.py:789: scalar = left.getarg(index)  (original accumulator variable)
        // vector.py:793-796: other args must align with origin pack
        let other_index = (index + 1) % 2;
        let origin_left_pos = graph.nodes[origin_pack.members[0]].op.pos.get();
        let origin_right_pos = graph.nodes[*origin_pack.members.last().unwrap()]
            .op
            .pos
            .get();

        if left.getarglist().get(other_index).copied() != Some(origin_left_pos) {
            return None;
        }
        if right.getarglist().get(other_index).copied() != Some(origin_right_pos) {
            return None;
        }

        // vector.py:799-812: bytesize must equal word size
        let size: i32 = if left.opcode.result_type() == majit_ir::Type::Float {
            8 // FLOAT_WORD
        } else {
            8 // INT_WORD on 64-bit
        };
        let l_bs = state.forwarded_vecinfo(left).getbytesize() as i32;
        let r_bs = state.forwarded_vecinfo(right).getbytesize() as i32;
        if !(l_bs == r_bs && l_bs == size) {
            return None;
        }

        // vector.py:813: return AccumPack([lnode, rnode], operator, index)
        let vec_op = left.opcode.to_vector().unwrap_or(left.opcode);
        Some(Pack {
            scalar_opcode: left.opcode,
            vector_opcode: vec_op,
            members: vec![lnode, rnode],
            is_accumulating: true,
            position: index as i32,
            operator: Some(operator),
        })
    }

    /// vector.py:820-824: getaccumulator_variable — find which arg of right
    /// is the result of left (the accumulator variable).
    fn getaccumulator_variable(left: &Op, right: &Op) -> (Option<OpRef>, i32) {
        for (i, arg) in right.getarglist().iter().enumerate() {
            if *arg == left.pos.get() {
                return (Some(*arg), i as i32);
            }
        }
        (None, -1)
    }
}

/// vector.py: Adjacent memory reference detection.
/// Checks if two memory operations access adjacent array elements.
pub fn are_adjacent_memory_refs(
    op_a: &majit_ir::Op,
    op_b: &majit_ir::Op,
    constant_of: impl Fn(OpRef) -> Option<i64>,
) -> bool {
    // Both must be the same opcode (e.g., GETARRAYITEM_GC_I)
    if op_a.opcode != op_b.opcode {
        return false;
    }
    // Both must access the same array (arg0)
    if op_a.num_args() < 2 || op_b.num_args() < 2 {
        return false;
    }
    if op_a.arg(0) != op_b.arg(0) {
        return false;
    }
    // Indices must differ by exactly 1
    if let (Some(idx_a), Some(idx_b)) = (constant_of(op_a.arg(1)), constant_of(op_b.arg(1))) {
        return (idx_b - idx_a).abs() == 1;
    }
    false
}

/// vector.py: Accumulation pack — tracks reduction operations
/// (e.g., sum += array[i]) that can be vectorized with horizontal
/// reduction instructions.
#[derive(Clone, Debug)]
/// schedule.py:980-1003: AccumPack — accumulation (reduction) pack.
pub struct AccumPack {
    /// The scalar opcode of the accumulation (e.g., IntAdd, FloatAdd).
    pub scalar_opcode: OpCode,
    /// schedule.py:981: operator ('+' for INT_ADD, etc.)
    pub operator: char,
    /// schedule.py:983: position — which arg of the op is the accumulator.
    pub position: usize,
    /// The initial accumulator value OpRef.
    pub init_value: OpRef,
    /// Indices of the accumulation operations in the loop body.
    pub members: Vec<usize>,
    /// Whether this is a float accumulation.
    pub is_float: bool,
}

/// Accumulation info stored in the accumulation map.
/// schedule.py:649: state.accumulation[arg] = pack
#[derive(Clone, Debug)]
pub struct AccumEntry {
    /// schedule.py:998: getleftmostseed() — first member's arg at `position`.
    pub seed: OpRef,
    /// schedule.py:981: operator character ('+' etc.)
    pub operator: char,
    /// The original scalar opcode (preserves int/float distinction).
    pub accum_opcode: OpCode,
}

/// vector.py: Guard analysis result — determines which guards can be
/// moved to the loop header (hoisted) to expose more vectorization.
#[derive(Clone, Debug)]
pub struct GuardAnalysis {
    /// Guards that can be hoisted to the loop header.
    pub hoistable: Vec<usize>,
    /// Guards that must remain in the loop body.
    pub body_guards: Vec<usize>,
}

impl GuardAnalysis {
    /// Analyze guards in a loop body for hoistability.
    /// vector.py: analyze_guards()
    /// A guard is hoistable if its arguments are loop-invariant
    /// (not produced by any op in the loop body).
    pub fn analyze(ops: &[Op]) -> Self {
        let mut body_results: majit_ir::vec_set::VecSet<OpRef> = majit_ir::vec_set::VecSet::new();
        for op in ops {
            if !op.pos.get().is_none() {
                body_results.insert(op.pos.get());
            }
        }

        let mut hoistable = Vec::new();
        let mut body_guards = Vec::new();

        for (i, op) in ops.iter().enumerate() {
            if !op.opcode.is_guard() {
                continue;
            }
            let all_invariant = op
                .getarglist()
                .iter()
                .all(|arg| !body_results.contains(arg));
            if all_invariant {
                hoistable.push(i);
            } else {
                body_guards.push(i);
            }
        }

        GuardAnalysis {
            hoistable,
            body_guards,
        }
    }
}

// ── Cost Model ──────────────────────────────────────────────────────────

/// Cost model for deciding whether vectorization is profitable.
///
/// From rpython/jit/metainterp/optimizeopt/costmodel.py.
///
/// Vectorization has overhead from:
/// - Pack/unpack operations to move scalars into/out of vector registers
/// - Potential register pressure increase
///
/// It saves:
/// - Instruction count reduction (N scalar ops → 1 vector op)
/// - Memory bandwidth (packed loads/stores)
/// costmodel.py: GenericCostModel — per-opcode cost estimation.
/// Maps opcodes to their estimated cost in abstract units.
pub struct GenericCostModel {
    /// Per-opcode cost overrides: opcode → cost.
    per_opcode_cost: crate::optimizeopt::vec_assoc::VecAssoc<OpCode, i32>,
    /// Default cost for opcodes not in the override map.
    default_cost: i32,
}

impl GenericCostModel {
    pub fn new() -> Self {
        let mut costs: crate::optimizeopt::vec_assoc::VecAssoc<OpCode, i32> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        // costmodel.py: memory ops are more expensive than ALU ops
        costs.insert(OpCode::GetarrayitemGcI, 3);
        costs.insert(OpCode::GetarrayitemGcR, 3);
        costs.insert(OpCode::GetarrayitemGcF, 3);
        costs.insert(OpCode::SetarrayitemGc, 3);
        costs.insert(OpCode::GetfieldGcI, 2);
        costs.insert(OpCode::GetfieldGcR, 2);
        costs.insert(OpCode::SetfieldGc, 2);
        // Float ops are more expensive
        costs.insert(OpCode::FloatAdd, 2);
        costs.insert(OpCode::FloatSub, 2);
        costs.insert(OpCode::FloatMul, 2);
        costs.insert(OpCode::FloatTrueDiv, 4);
        GenericCostModel {
            per_opcode_cost: costs,
            default_cost: 1,
        }
    }

    /// Get the cost of a single operation.
    pub fn op_cost(&self, opcode: OpCode) -> i32 {
        self.per_opcode_cost
            .get(&opcode)
            .copied()
            .unwrap_or(self.default_cost)
    }

    /// Estimate total savings from vectorizing a pack group.
    pub fn estimate_savings(&self, group: &Pack) -> i32 {
        let n = group.members.len() as i32;
        let per_op = self.op_cost(group.scalar_opcode);
        // Savings = (n-1) ops eliminated * per-op cost
        // Cost = pack + unpack overhead
        let savings = (n - 1) * per_op;
        let overhead = 2 * 2; // 2 pack/unpack ops at cost 2 each
        savings - overhead
    }
}

impl Default for GenericCostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// vector.py:601-668 (GenericCostModel): Cost model using a single `savings`
/// counter. Positive savings = profitable; negative = not worth it.
pub struct CostModel {
    /// Minimum group size to consider vectorization (default: 2).
    pub min_pack_size: usize,
    /// Overhead per pack/unpack operation (in abstract cost units).
    pub pack_cost: i32,
    /// Saving per eliminated scalar op (in abstract cost units).
    pub scalar_save: i32,
    /// vector.py:609: single savings counter (positive = profitable).
    pub savings: i32,
}

impl CostModel {
    pub fn new() -> Self {
        CostModel {
            min_pack_size: 2,
            pack_cost: 2,
            scalar_save: 1,
            savings: 0,
        }
    }

    /// vector.py:611: reset_savings
    pub fn reset_savings(&mut self) {
        self.savings = 0;
    }

    /// vector.py:636-643 (GenericCostModel.record_pack_savings):
    /// Record savings from vectorizing a pack. INT_SIGNEXT gets special cost.
    pub fn record_pack_savings(&mut self, pack: &Pack, times: usize) {
        let (cost, benefit_factor) = if pack.scalar_opcode == OpCode::IntSignext {
            // vector.py:645-650: cb_signext — no benefit for signext
            (1i32, 0i32)
        } else {
            (1, 1)
        };
        self.savings += benefit_factor * times as i32 - cost;
    }

    /// vector.py:659-665 (GenericCostModel.record_vector_pack):
    /// Record cost of packing a scalar into a vector.
    pub fn record_vector_pack(&mut self, is_float: bool, index: usize, count: usize) {
        if is_float && index == 1 && count == 1 {
            self.savings -= 2;
            return;
        }
        self.savings -= count as i32;
    }

    /// vector.py:667-668: record_vector_unpack delegates to record_vector_pack.
    pub fn record_vector_unpack(&mut self, is_float: bool, index: usize, count: usize) {
        self.record_vector_pack(is_float, index, count);
    }

    /// vector.py:652-657 (GenericCostModel.record_cast_int):
    /// Record cost of integer sign-extension / size conversion.
    pub fn record_cast_int(&mut self, fromsize: usize, tosize: usize, count: usize) {
        if fromsize == 8 && tosize == 4 && count == 2 {
            self.savings -= 1;
        } else {
            self.savings -= count as i32;
        }
    }

    /// vector.py:632-633: profitable — savings >= 0.
    pub fn profitable(&self) -> bool {
        self.savings >= 0
    }

    /// Estimate whether vectorizing a group is profitable (pre-filter).
    pub fn is_profitable(&self, group: &Pack) -> bool {
        let n = group.members.len() as i32;
        if n < self.min_pack_size as i32 {
            return false;
        }
        let savings = (n - 1) * self.scalar_save;
        let pack_ops = 2;
        let cost = pack_ops * self.pack_cost;
        savings > cost
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ── schedule.py:584-779: VecScheduleState ─────────────────────

/// schedule.py:584-779: State for vector-aware instruction scheduling.
/// Tracks which scalar ops have been mapped to vector ops, handles
/// pack/unpack/expand operations, and manages the output op list.
pub struct VecScheduleState {
    /// Map from scalar OpRef → (index_in_vector, vector OpRef).
    pub box_to_vbox: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, (usize, OpRef)>,
    /// Output operations (vector + remaining scalar).
    pub oplist: Vec<Op>,
    /// Renamer for SSA fixup during vectorization.
    pub renamer: super::renamer::Renamer,
    /// Cost model for profitability analysis.
    pub costmodel: CostModel,
    /// schedule.py:587-588: expanded_map — tracks expanded scalars.
    pub expanded_map: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, Vec<(OpRef, i32)>>,
    /// schedule.py:591: inputargs of the loop label.
    pub inputargs: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, ()>,
    /// schedule.py:38,723: invariant_vector_vars — vector ops created by expand()
    /// for loop-invariant scalars (constants and inputargs). Populated in
    /// expand() (schedule.py:554-555), called from prepare_arguments().
    pub invariant_vector_vars: majit_ir::vec_set::VecSet<OpRef>,
    /// schedule.py:532: invariant_oplist — ops to emit before the loop.
    pub invariant_oplist: Vec<Op>,
    /// schedule.py:595: accumulation info.
    pub accumulation: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, AccumEntry>,
    /// Next OpRef counter for newly created vector ops.
    next_pos: u32,
    /// RPython `box._forwarded` storage used by the vectorizer for
    /// `VectorizationInfo`. This is local to scheduling because upstream
    /// clears the vectorization `_forwarded` state after the vector pass
    /// (`vector.py:58-60`, `finaloplist`:88-90).
    ///
    /// Keyed by full `OpRef` (variant + payload) so the
    /// `InputArg*` and `*Op` namespaces stay disjoint — PyPy uses Box
    /// object identity (`AbstractValue` `is`), so e.g. an inputarg at
    /// slot 0 and an op at position 0 cannot collide upstream and must
    /// not collide here either.
    box_pool: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, BoxRef>,
}

impl VecScheduleState {
    pub fn new(start_pos: u32) -> Self {
        VecScheduleState {
            box_to_vbox: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            oplist: Vec::new(),
            renamer: super::renamer::Renamer::new(),
            costmodel: CostModel::new(),
            expanded_map: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            inputargs: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            invariant_vector_vars: majit_ir::vec_set::VecSet::new(),
            invariant_oplist: Vec::new(),
            accumulation: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            next_pos: start_pos,
            box_pool: crate::optimizeopt::vec_assoc::VecAssoc::new(),
        }
    }

    /// vector.py:54-60 setup/teardown_vectorization. Seed only the local
    /// scheduling BoxPool; RPython clears this `_forwarded` state after the
    /// vector pass, so it must not leak into the main optimizer BoxPool.
    ///
    /// `constant_of` resolves the optimizer's const-pool so the
    /// `resoperation.py:181-186 INT_SIGNEXT` branch can read `arg1.value`
    /// at stamp time — PyPy reads the value off the `ConstInt` instance
    /// directly because `_args[1]` IS the const; pyre's flat-OpRef
    /// encoding stores constants as a pool index, so the resolver fills
    /// the same role at the moment we populate the inline vecinfo slot.
    pub fn setup_vectorization(&mut self, ops: &[Op], constant_of: &dyn Fn(OpRef) -> Option<i64>) {
        for op in ops {
            let info = if op.opcode == OpCode::IntSignext {
                self.int_signext_vecinfo(op, constant_of)
            } else {
                self.vectorization_info_for_op(op)
            };
            if let Some(b) = self.ensure_box_for_opref(op.pos.get()) {
                b.set_forwarded_vector_info(info);
            }
        }
    }

    /// `resoperation.py:181-186 VectorizationInfo.__init__` INT_SIGNEXT
    /// branch.  PyPy reads `op.getarg(1).value` off the `ConstInt` object
    /// directly; pyre's flat-OpRef encoding stores the const as a pool
    /// index, so the caller-supplied resolver materialises the value at
    /// stamp time.  Falls back to the generic branch when arg1 isn't a
    /// resolvable constant (mirrors PyPy's `assert isinstance(arg1, ConstInt)`
    /// degrading to the typecast-fallthrough path in non-translated runs).
    fn int_signext_vecinfo(
        &self,
        op: &Op,
        constant_of: &dyn Fn(OpRef) -> Option<i64>,
    ) -> majit_ir::VectorizationInfo {
        if op.num_args() >= 2 {
            if let Some(bytesize) = constant_of(op.arg(1)) {
                if (i8::MIN as i64..=i8::MAX as i64).contains(&bytesize) {
                    let mut info = majit_ir::VectorizationInfo::new();
                    info.setinfo('i', bytesize as i8, true);
                    return info;
                }
            }
        }
        self.vectorization_info_for_op(op)
    }

    fn ensure_box_for_opref(&mut self, opref: OpRef) -> Option<BoxRef> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        if let Some(existing) = self.box_pool.get(&opref) {
            return Some(existing.clone());
        }
        let tp = opref.ty().unwrap_or(Type::Void);
        let raw = opref.raw();
        let b = if matches!(
            opref,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            BoxRef::new_inputarg(tp, raw)
        } else {
            BoxRef::new_resop(tp, raw)
        };
        self.box_pool.insert(opref, b.clone());
        Some(b)
    }

    fn get_forwarded_vecinfo(&self, opref: OpRef) -> Option<majit_ir::VectorizationInfo> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        self.box_pool.get(&opref).and_then(|b| b.vector_info())
    }

    fn set_forwarded_vecinfo(&mut self, opref: OpRef, info: majit_ir::VectorizationInfo) {
        if let Some(b) = self.ensure_box_for_opref(opref) {
            b.set_forwarded_vector_info(info);
        }
    }

    /// schedule.py:20-28 `forwarded_vecinfo(op)`.
    pub fn forwarded_vecinfo(&mut self, op: &Op) -> majit_ir::VectorizationInfo {
        let opref = op.pos.get();
        if let Some(info) = self.get_forwarded_vecinfo(opref) {
            return info;
        }
        let info = self.vectorization_info_for_op(op);
        if !opref.is_constant() {
            self.set_forwarded_vecinfo(opref, info.clone());
        }
        info
    }

    fn forwarded_vecinfo_for_ref(
        &mut self,
        opref: OpRef,
        ops: &[Op],
    ) -> majit_ir::VectorizationInfo {
        if let Some(info) = self.get_forwarded_vecinfo(opref) {
            return info;
        }
        if let Some(op) = ops
            .iter()
            .chain(self.oplist.iter())
            .chain(self.invariant_oplist.iter())
            .find(|op| op.pos.get() == opref)
            .cloned()
        {
            return self.forwarded_vecinfo(&op);
        }
        let tp = opref.ty().unwrap_or(Type::Int);
        let info = majit_ir::VectorizationInfo::from_type(tp);
        if !opref.is_constant() {
            self.set_forwarded_vecinfo(opref, info.clone());
        }
        info
    }

    /// resoperation.py:163-212 `VectorizationInfo(op)` for ResOps.
    /// Const/InputArg are handled by the cache miss path in
    /// `forwarded_vecinfo_for_ref` (`VectorizationInfo.from_type`).
    ///
    /// Mirrors PyPy `resoperation.py:163-212 VectorizationInfo.__init__`.
    /// The `INT_SIGNEXT` branch (`:181-186`) is handled in
    /// `int_signext_vecinfo` at setup time because it needs the
    /// caller-supplied const-pool resolver; once the inline vecinfo slot
    /// is stamped, every later lookup hits the cache and never re-enters
    /// this method for INT_SIGNEXT.  A bare miss-path call on
    /// INT_SIGNEXT (e.g. a synthesised op that bypassed setup) falls
    /// through to the typecast branch which returns `None` for it, and
    /// then to the source-operand pass-through — same degradation as
    /// PyPy's non-translated `assert` failure path.
    fn vectorization_info_for_op(&self, op: &Op) -> majit_ir::VectorizationInfo {
        // resoperation.py:170-180 primitive_array_access branch.
        if op.opcode.is_primitive_array_access_opcode() {
            if let Some(descr) = op.getdescr() {
                if let Some(arr) = majit_ir::descr::descr_arc_as_array_descr(descr) {
                    if arr.is_array_of_primitives() {
                        let datatype = match op.result_type() {
                            Type::Int => 'i',
                            Type::Float => 'f',
                            Type::Ref => 'r',
                            Type::Void => 'v',
                        };
                        let bytesize = arr.item_size() as i8;
                        let signed = arr.is_item_signed();
                        let mut info = majit_ir::VectorizationInfo::new();
                        info.setinfo(datatype, bytesize, signed);
                        return info;
                    }
                }
            }
        }

        // resoperation.py:187-190 is_typecast branch (INT_SIGNEXT static
        // gating returns None per `cast_to_bytesize_static`; the dynamic
        // INT_SIGNEXT bytesize is stamped in `int_signext_vecinfo` at
        // setup time and read back through the inline vecinfo slot).
        if op.opcode.is_typecast() {
            if let Some(bytesize) = op.opcode.cast_to_bytesize_static() {
                let (_ft, tt) = op.opcode.cast_types();
                let mut info = majit_ir::VectorizationInfo::new();
                info.setinfo(tt, bytesize as i8, true);
                return info;
            }
        }

        // resoperation.py:192-209 else branch:
        //   type = op.type
        //   signed = type == 'i'
        //   bytesize = -1
        //   if op.numargs() > 0:
        //       i = 0
        //       arg = op.getarg(i)
        //       while arg.is_constant() and i+1 < op.numargs():
        //           i += 1
        //           arg = op.getarg(i)
        //       if not arg.is_constant():
        //           vecinfo = arg.get_forwarded()
        //           if vecinfo is not None and isinstance(vecinfo, VectorizationInfo):
        //               if vecinfo.datatype != '\x00' and vecinfo.bytesize != -1:
        //                   type     = vecinfo.datatype
        //                   signed   = vecinfo.signed
        //                   bytesize = vecinfo.bytesize
        //   if rop.returns_bool_result(op.opnum):
        //       type = 'i'
        //   self.setinfo(type, bytesize, signed)
        let mut tp = op.result_type();
        let mut datatype = match tp {
            Type::Int => 'i',
            Type::Float => 'f',
            Type::Ref => 'r',
            Type::Void => 'v',
        };
        let mut signed = datatype == 'i';
        let mut bytesize: i8 = -1;

        let n = op.num_args();
        if n > 0 {
            let mut i = 0usize;
            let mut arg = op.arg(i);
            while arg.is_constant() && i + 1 < n {
                i += 1;
                arg = op.arg(i);
            }
            if !arg.is_constant() {
                if let Some(vinfo) = self.get_forwarded_vecinfo(arg) {
                    if vinfo.datatype != '\x00' && vinfo.bytesize != -1 {
                        datatype = vinfo.datatype;
                        tp = match datatype {
                            'i' => Type::Int,
                            'f' => Type::Float,
                            'r' => Type::Ref,
                            _ => tp,
                        };
                        signed = vinfo.signed;
                        bytesize = vinfo.bytesize;
                    }
                }
            }
        }

        if op.opcode.returns_bool() {
            datatype = 'i';
            tp = Type::Int;
        }
        let _ = tp;

        let mut info = majit_ir::VectorizationInfo::new();
        info.setinfo(datatype, bytesize, signed);
        info
    }

    /// Allocate a fresh typed OpRef for a newly created vector op. The
    /// caller supplies the op's result type (`opcode.result_type()`) so
    /// the returned OpRef carries the proper `Int/Float/Ref/Void`
    /// variant tag.
    pub fn alloc_op_pos(&mut self, tp: Type) -> OpRef {
        let pos = OpRef::op_typed(self.next_pos, tp);
        self.next_pos += 1;
        pos
    }

    /// resoperation.py:111-116 (VecOperationNew): Create a vector op with
    /// proper VectorizationInfo. All vector helper functions should use this
    /// instead of raw Op::new + register_vec_type.
    pub fn create_vec_op(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        datatype: char,
        bytesize: i32,
        signed: bool,
        count: usize,
    ) -> Op {
        let op = Op::new(opcode, args);
        op.pos.set(self.alloc_op_pos(opcode.result_type()));
        let mut vinfo = majit_ir::VectorizationInfo::new();
        vinfo.setinfo(datatype, bytesize as i8, signed);
        vinfo.count = count as i16;
        // resoperation.py:111-115 VecOperationNew.__init__ stores the
        // datatype/bytesize/signed/count on the op object itself; copy_and_change
        // (resoperation.py:511-518) propagates them. Cache the same payload
        // on `Op.vecinfo` so the vector shape survives schedule-state teardown.
        op.set_vecinfo(vinfo.clone());
        self.set_forwarded_vecinfo(op.pos.get(), vinfo);
        op
    }

    /// Check if an OpRef refers to a float-type vector op.  The OpRef
    /// variant itself carries the result-type tag (`opcode.result_type()`
    /// at `alloc_op_pos`), mirroring PyPy's `opclasses[opnum].type == 'f'`
    /// gating in `resoperation.py:1597` — no side-table needed.
    pub fn is_float_vector(&self, opref: OpRef) -> bool {
        opref.ty() == Some(Type::Float)
    }

    /// schedule.py:625-630: setvector_of_box — record that scalar_op
    /// is at index `idx` in the vector `vecop`.
    pub fn setvector_of_box(&mut self, scalar_op: OpRef, idx: usize, vecop: OpRef) {
        self.box_to_vbox.insert(scalar_op, (idx, vecop));
    }

    /// schedule.py:632-638: getvector_of_box — look up which vector
    /// op contains the scalar op.
    pub fn getvector_of_box(&self, scalar_op: OpRef) -> Option<(usize, OpRef)> {
        self.box_to_vbox.get(&scalar_op).copied()
    }

    /// schedule.py:640-650: append to output.
    pub fn append_to_oplist(&mut self, op: Op) {
        self.oplist.push(op);
    }

    /// schedule.py:754-760: remember_args_in_vector — after creating a new
    /// vector from assemble/position, update box_to_vbox for each scalar arg
    /// so future lookups find the correct vector box.
    pub fn remember_args_in_vector(
        &mut self,
        pack: &Pack,
        index: usize,
        vecbox: OpRef,
        ops: &[Op],
    ) {
        for (i, &member_idx) in pack.members.iter().enumerate() {
            let op = &ops[member_idx];
            if index >= op.num_args() {
                break;
            }
            let arg = op.arg(index);
            // schedule.py:757-760:
            //   vecinfo = forwarded_vecinfo(arg)
            //   if i >= vecinfo.count: break
            //   self.setvector_of_box(arg, i, box)
            let vecinfo = self.forwarded_vecinfo_for_ref(arg, ops);
            if (i as i16) >= vecinfo.count {
                break;
            }
            self.setvector_of_box(arg, i, vecbox);
        }
    }

    // ── schedule.py:524-633: expand / find_expanded ──

    /// schedule.py:597-604: record that `args` were expanded into `vecop`.
    pub fn record_expansion(&mut self, args: &[OpRef], vecop: OpRef) {
        let mut index: i32 = 0;
        if args.len() == 1 {
            index = -1; // schedule.py:600: broadcast marker
        }
        for arg in args {
            self.expanded_map
                .entry_or_insert_with(*arg, Vec::new)
                .push((vecop, index));
            index += 1;
        }
    }

    /// schedule.py:606-633: find_expanded — look up if `args` have already
    /// been expanded into a vector op.
    pub fn find_expanded(&self, args: &[OpRef]) -> Option<OpRef> {
        if args.len() == 1 {
            // schedule.py:607-612: single arg → look for broadcast (index == -1)
            let candidates = self.expanded_map.get(&args[0])?;
            for &(vecop, index) in candidates {
                if index == -1 {
                    return Some(vecop);
                }
            }
            return None;
        }
        // schedule.py:614-632: multi-arg → intersect candidates at correct positions.
        // For each arg position i, collect vecops that expanded arg at index i.
        // A vecop is valid only if it appears at every position — intersect.
        let mut possible: crate::optimizeopt::vec_assoc::VecAssoc<OpRef, bool> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        for (i, arg) in args.iter().enumerate() {
            let expansions = match self.expanded_map.get(arg) {
                Some(e) => e,
                None => return None,
            };
            // schedule.py:617-618: filter by index match AND possible.get(vecop, True)
            let candidates: Vec<OpRef> = expansions
                .iter()
                .filter(|&&(vecop, idx)| {
                    idx == i as i32 && possible.get(&vecop).copied().unwrap_or(true)
                })
                .map(|&(vecop, _)| vecop)
                .collect();
            // schedule.py:620-623: invalidate vecops NOT in this position's candidates
            for (k, v) in possible.iter_mut() {
                if !candidates.contains(k) {
                    *v = false;
                }
            }
            // schedule.py:625: mark surviving candidates as valid
            for vecop in candidates {
                if !possible.contains_key(&vecop) {
                    possible.insert(vecop, true);
                }
            }
            if possible.is_empty() {
                return None;
            }
        }
        possible.iter().find(|(_, v)| **v).map(|(k, _)| *k)
    }
}

// ── schedule.py:317-400: turn_into_vector and helpers ─────────────────────

/// schedule.py:317-320: failnbail_transformation
#[derive(Debug)]
pub struct NotAVectorizeableLoop;
#[derive(Debug)]
pub struct NotAProfitableLoop;

/// Combined failure mode for `optimize_vector` / `run_optimization`,
/// mirroring vector.py:154-166's two `except` arms. Callers convert this
/// back to a "no-vectorize-this-time" decision and replay the original
/// loop ops; the distinction is kept so future passes (e.g. logging or
/// GuardStrengthenOpt) can react differently.
#[derive(Debug)]
pub enum VectorizeError {
    NotVectorizeable,
    NotProfitable,
}

impl From<NotAVectorizeableLoop> for VectorizeError {
    fn from(_: NotAVectorizeableLoop) -> Self {
        VectorizeError::NotVectorizeable
    }
}

impl From<NotAProfitableLoop> for VectorizeError {
    fn from(_: NotAProfitableLoop) -> Self {
        VectorizeError::NotProfitable
    }
}

/// schedule.py:462-474: check_if_pack_supported — validate pack constraints.
pub fn check_if_pack_supported(
    state: &mut VecScheduleState,
    pack: &Pack,
    ops: &[Op],
) -> Result<(), NotAProfitableLoop> {
    let first_op = &ops[pack.members[0]];
    // schedule.py:471-474: INT_MUL with bytesize 8 or 1 is not profitable
    if first_op.opcode == OpCode::IntMul {
        let vi = state.forwarded_vecinfo(first_op);
        let insize = vi.getbytesize();
        if insize == 8 || insize == 1 {
            return Err(NotAProfitableLoop);
        }
    }
    Ok(())
}

/// schedule.py:476-486: unpack_from_vector — extract a scalar from a vector box.
/// Creates a VecUnpack op with the correct type (I or F) based on the
/// vector box's datatype. Mirrors OpHelpers.create_vec_unpack(var.type, ...).
pub fn unpack_from_vector(
    state: &mut VecScheduleState,
    vec_ref: OpRef,
    index: usize,
    count: usize,
) -> OpRef {
    assert!(count > 0);
    let index_const = OpRef::const_int(index as u32);
    let count_const = OpRef::const_int(count as u32);
    let is_float = state.is_float_vector(vec_ref);
    let unpack_opcode = if is_float {
        OpCode::VecUnpackF
    } else {
        OpCode::VecUnpackI
    };
    // schedule.py:479-483: forwarded_vecinfo(arg).bytesize/signed
    let (datatype, bytesize, signed) = get_vec_info(state, vec_ref, &[]);
    let unpack_op = state.create_vec_op(
        unpack_opcode,
        &[vec_ref, index_const, count_const],
        datatype,
        bytesize,
        signed,
        count,
    );
    let result = unpack_op.pos.get();
    // schedule.py:484: costmodel.record_vector_unpack
    state.costmodel.record_vector_unpack(is_float, index, count);
    state.append_to_oplist(unpack_op);
    result
}

/// schedule.py:388-400: prepare_fail_arguments — process guard failargs
/// for vectorized guard ops, unpacking vector boxes to scalar.
pub fn prepare_fail_arguments(
    state: &mut VecScheduleState,
    pack: &Pack,
    ops: &[Op],
    vecop: &mut Op,
) {
    let first_op = &ops[pack.members[0]];
    if !first_op.opcode.is_guard() {
        return;
    }
    if let Some(fail_args) = first_op.getfailargs() {
        let mut new_fail_args: smallvec::SmallVec<[OpRef; 3]> = fail_args.iter().copied().collect();
        for arg in new_fail_args.iter_mut() {
            // schedule.py:393-394: look up if arg is in a vector box
            let (_pos, newarg) = state.getvector_of_box(*arg).unwrap_or((0, *arg));
            if newarg != *arg {
                // schedule.py:396-397: vector box → unpack at position 0
                let unpacked = unpack_from_vector(state, newarg, 0, 1);
                *arg = unpacked;
            }
        }
        vecop.setfailargs(new_fail_args);
    }
}

/// schedule.py:352-386: prepare_arguments — transform scalar args to vector args.
///
/// RPython gates this on cpu.vector_ext.get_operation_restriction(left).
/// majit uses Cranelift which has no per-op restriction object, so we apply
/// the same logic for all args: reuse, assemble, position, crop, or expand.
pub fn prepare_arguments(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut Vec<OpRef>,
    ops: &[Op],
) {
    for i in 0..args.len() {
        let arg = args[i];
        // schedule.py:375-376: check if arg is in a vector box
        if let Some((pos, vecop)) = state.getvector_of_box(arg) {
            // schedule.py:382: case 1a — reuse existing vector
            args[i] = vecop;
            // schedule.py:383: case 1c — scattered values
            assemble_scattered_values(state, pack, args, i, ops);
            // schedule.py:384: case 1d — wrong position in vector
            position_values(state, pack, args, i, pos, ops);
            // schedule.py:385: case 1b — size mismatch (crop)
            crop_vector(state, pack, args, i, ops);
        } else {
            // schedule.py:377-378: case 2 — not in a vector, expand
            expand(state, pack, args, arg, i, ops);
        }
    }
}

/// schedule.py:420-428: assemble_scattered_values — when pack members'
/// args at `index` reside in different vector boxes, gather them into one.
fn assemble_scattered_values(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut Vec<OpRef>,
    index: usize,
    ops: &[Op],
) {
    // schedule.py:422: collect each member's arg at this index
    let mut args_at_index: Vec<OpRef> = pack
        .members
        .iter()
        .map(|&m| {
            let op = &ops[m];
            if index < op.num_args() {
                op.arg(index)
            } else {
                args[index]
            }
        })
        .collect();
    // schedule.py:423: first one is already assigned
    args_at_index[0] = args[index];

    // schedule.py:424: check which vector boxes these args reside in
    let mut vectors: Vec<(usize, OpRef)> = Vec::new();
    for &a in &args_at_index {
        if let Some((pos, vecop)) = state.getvector_of_box(a) {
            if vectors.is_empty() || vectors.last().map(|v| v.1) != Some(vecop) {
                vectors.push((pos, vecop));
            }
        }
    }

    // schedule.py:425-428: if scattered across >1 vector, gather
    if vectors.len() > 1 {
        args[index] = gather(state, &vectors, pack.members.len(), ops);
        // schedule.py:428: remember_args_in_vector
        state.remember_args_in_vector(pack, index, args[index], ops);
    }
}

/// schedule.py:430-441: gather — combine multiple vector fragments into one.
/// Uses each fragment's actual lane count (vecinfo.count / newvecinfo.count)
/// to determine insertion position and guard against overfill.
fn gather(
    state: &mut VecScheduleState,
    vectors: &[(usize, OpRef)],
    count: usize,
    ops: &[Op],
) -> OpRef {
    let (_, mut arg) = vectors[0];
    let mut i = 1;
    while i < vectors.len() {
        let (newarg_pos, newarg) = vectors[i];
        // schedule.py:436-437: get actual lane counts from vecinfo
        let arg_count = get_vec_count(state, arg, ops);
        let newarg_count = get_vec_count(state, newarg, ops);
        // schedule.py:438: guard: combined count must fit in target
        if arg_count + newarg_count <= count {
            // schedule.py:439: pack newarg into arg at arg's current count
            arg = pack_into_vector(state, arg, arg_count, newarg, newarg_pos, newarg_count, ops);
        }
        i += 1;
    }
    arg
}

/// Get the lane count of a vector OpRef. Falls back to 1.
fn get_vec_count(state: &mut VecScheduleState, opref: OpRef, ops: &[Op]) -> usize {
    state.forwarded_vecinfo_for_ref(opref, ops).count.max(1) as usize
}

/// Get (datatype, bytesize, signed) from a vector OpRef's vecinfo.
/// Falls back to is_float_vector registry + default 8-byte.
fn get_vec_info(state: &mut VecScheduleState, opref: OpRef, ops: &[Op]) -> (char, i32, bool) {
    let vi = state.forwarded_vecinfo_for_ref(opref, ops);
    if vi.datatype != '\0' {
        return (vi.datatype, vi.bytesize as i32, vi.signed);
    }
    let is_float = state.is_float_vector(opref);
    if is_float {
        ('f', 8, false)
    } else {
        ('i', 8, true)
    }
}

/// schedule.py:488-502: pack_into_vector — insert `src` at position `tidx`
/// in `tgt`, producing a wider vector.
///
/// tgt = [1,2,3,4,_,_,_,_], src = [5,6,_,_]
/// result = [1,2,3,4,5,6,_,_] (tidx=4, scount=2)
fn pack_into_vector(
    state: &mut VecScheduleState,
    tgt: OpRef,
    tidx: usize,
    src: OpRef,
    sidx: usize,
    scount: usize,
    ops: &[Op],
) -> OpRef {
    // schedule.py:493: assert sidx == 0
    debug_assert!(sidx == 0, "pack_into_vector: sidx must be 0, got {}", sidx);
    let is_float = state.is_float_vector(tgt);
    let pack_opcode = if is_float {
        OpCode::VecPackF
    } else {
        OpCode::VecPackI
    };
    let tidx_const = OpRef::const_int(tidx as u32);
    let scount_const = OpRef::const_int(scount as u32);
    // schedule.py:494-497: forwarded_vecinfo(tgt).bytesize/signed, newcount
    let tgt_count = get_vec_count(state, tgt, ops);
    let newcount = tgt_count + scount;
    let (datatype, bytesize, signed) = get_vec_info(state, tgt, ops);
    let vecop = state.create_vec_op(
        pack_opcode,
        &[tgt, src, tidx_const, scount_const],
        datatype,
        bytesize,
        signed,
        newcount,
    );
    let result = vecop.pos.get();
    state.append_to_oplist(vecop);
    // schedule.py:499: record cost
    state.costmodel.record_vector_pack(is_float, 0, scount);
    result
}

/// schedule.py:443-460: position_values — if an arg is at position != 0
/// in its vector box but needs to be at position 0, unpack it.
fn position_values(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut Vec<OpRef>,
    index: usize,
    position: usize,
    ops: &[Op],
) {
    // schedule.py:453-460: position != 0 → unpack to reposition
    if position != 0 {
        let arg = args[index];
        // schedule.py:458: count = restrict.max_input_count(vecinfo.count)
        // Without oprestrict, default to 1 (extract single element)
        let count = 1;
        args[index] = unpack_from_vector(state, arg, position, count);
        // schedule.py:460: remember_args_in_vector
        state.remember_args_in_vector(pack, index, args[index], ops);
    }
}

/// schedule.py:402-418: crop_vector — if the vector's element size doesn't
/// match what the operation requires, insert VEC_INT_SIGNEXT.
fn crop_vector(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut Vec<OpRef>,
    index: usize,
    ops: &[Op],
) {
    let arg = args[index];
    let first_op = &ops[pack.members[0]];
    // schedule.py:406-408: check if bytesize needs conversion
    // Determine the vector's current element size and the op's expected size
    let arg_bytesize = get_op_bytesize_for_ref(state, arg, ops);
    let op_bytesize = state.forwarded_vecinfo(&ops[pack.members[0]]).getbytesize() as i32;
    if arg_bytesize > 0 && op_bytesize > 0 && arg_bytesize != op_bytesize {
        // schedule.py:411-417: integer type → VEC_INT_SIGNEXT
        if first_op.opcode.result_type() != majit_ir::Type::Float {
            let newsize_const = OpRef::const_int(op_bytesize as u32);
            let vec_count = get_vec_count(state, arg, ops);
            // schedule.py:414-415: VecOperationNew with proper vecinfo
            let signext_op = state.create_vec_op(
                OpCode::VecIntSignext,
                &[arg, newsize_const],
                'i',
                op_bytesize,
                true, // signed
                vec_count,
            );
            let result = signext_op.pos.get();
            state.append_to_oplist(signext_op);
            // schedule.py:417: record cost
            state
                .costmodel
                .record_cast_int(arg_bytesize as usize, op_bytesize as usize, vec_count);
            args[index] = result;
        }
    }
}

/// Helper: get bytesize for an OpRef that may be a vector op created during scheduling.
fn get_op_bytesize_for_ref(state: &mut VecScheduleState, opref: OpRef, ops: &[Op]) -> i32 {
    state.forwarded_vecinfo_for_ref(opref, ops).getbytesize() as i32
}

/// schedule.py:524-582: expand — broadcast or gather a scalar into a vector box.
///
/// Two cases:
///   1. All pack members use the same arg at position `index` → VecExpand (broadcast)
///   2. Different args per member → Vec + VecPack (gather)
///
/// Loop-invariant expansions (constants and inputargs) go to invariant_oplist
/// and are tracked in invariant_vector_vars.
fn expand(
    state: &mut VecScheduleState,
    pack: &Pack,
    args: &mut Vec<OpRef>,
    arg: OpRef,
    index: usize,
    ops: &[Op],
) {
    // schedule.py:532-537: choose target list (invariant vs inline)
    let is_invariant = arg.is_constant() || state.inputargs.contains_key(&arg);

    // schedule.py:539-543: check if all pack members have the same arg at `index`
    let all_same = pack.members.iter().all(|&m| {
        let op = &ops[m];
        index < op.num_args() && op.arg(index) == arg
    });

    // datatype is `arg.type` per PyPy `OpHelpers.create_vec_expand`
    // (resoperation.py:1556-1562) — opcode dispatch + the resulting
    // vecinfo.datatype both come from the arg being expanded.
    let datatype = match arg.ty().unwrap_or(Type::Void) {
        Type::Int => 'i',
        Type::Float => 'f',
        Type::Ref => 'r',
        Type::Void => 'v',
    };
    let is_float = datatype == 'f';
    let numops = pack.members.len();

    if all_same {
        // schedule.py:546-558: VecExpand (broadcast)
        if let Some(existing) = state.find_expanded(&[arg]) {
            args[index] = existing;
            return;
        }
        // schedule.py:550-552: bytesize/signed come from the left-most
        // pack op's vecinfo, NOT from `arg` — the pack's element width is
        // the authoritative shape for the broadcast destination.
        //   left = pack.leftmost()
        //   vecinfo = forwarded_vecinfo(left)
        //   vecop = OpHelpers.create_vec_expand(arg, vecinfo.bytesize,
        //                                       vecinfo.signed, pack.numops())
        let left_op = &ops[pack.members[0]];
        let left_info = state.forwarded_vecinfo(left_op);
        let bytesize = left_info.bytesize as i32;
        let signed = left_info.signed;
        let expand_opcode = if is_float {
            OpCode::VecExpandF
        } else {
            OpCode::VecExpandI
        };
        let vecop = state.create_vec_op(expand_opcode, &[arg], datatype, bytesize, signed, numops);
        let vecop_pos = vecop.pos.get();
        if is_invariant {
            state.invariant_oplist.push(vecop);
            state.invariant_vector_vars.insert(vecop_pos);
        } else {
            state.append_to_oplist(vecop);
        }
        state.record_expansion(&[arg], vecop_pos);
        args[index] = vecop_pos;
        return;
    }

    // schedule.py:567: arg_vecinfo = forwarded_vecinfo(arg)
    //   vecop = OpHelpers.create_vec(arg.type, arg_vecinfo.bytesize,
    //                                arg_vecinfo.signed, pack.opnum())
    // Only the heterogeneous (VecPack/gather) branch uses arg's vecinfo —
    // each pack member contributes its own scalar to the vector, so the
    // arg's element width is the shape source.
    let arg_info = state.forwarded_vecinfo_for_ref(arg, ops);
    let bytesize = arg_info.bytesize as i32;
    let signed = arg_info.signed;

    // schedule.py:560-582: VecPack (gather) — heterogeneous args
    let expandargs: Vec<OpRef> = pack
        .members
        .iter()
        .map(|&m| {
            let op = &ops[m];
            if index < op.num_args() {
                op.arg(index)
            } else {
                arg
            }
        })
        .collect();

    if let Some(existing) = state.find_expanded(&expandargs) {
        args[index] = existing;
        return;
    }

    // schedule.py:568: create_vec(datatype, bytesize, signed, count)
    let vec_create_opcode = if is_float { OpCode::VecF } else { OpCode::VecI };
    let vec_create =
        state.create_vec_op(vec_create_opcode, &[], datatype, bytesize, signed, numops);
    let mut current_vec = vec_create.pos.get();
    if is_invariant {
        state.invariant_oplist.push(vec_create);
    } else {
        state.append_to_oplist(vec_create);
    }

    // schedule.py:570-577: pack each member's arg into the vector
    let pack_opcode = if is_float {
        OpCode::VecPackF
    } else {
        OpCode::VecPackI
    };
    for (i, &member_arg) in expandargs.iter().enumerate() {
        let i_const = OpRef::const_int(i as u32);
        let one_const = OpRef::const_int(1);
        // schedule.py:575: create_vec_pack(type, args, bytesize, signed, count+1)
        let pack_op = state.create_vec_op(
            pack_opcode,
            &[current_vec, member_arg, i_const, one_const],
            datatype,
            bytesize,
            signed,
            i + 2, // schedule.py:576: vecinfo.count+1 (grows by 1 each iteration)
        );
        current_vec = pack_op.pos.get();
        state.costmodel.record_vector_pack(is_float, 0, 1);
        if is_invariant {
            state.invariant_oplist.push(pack_op);
        } else {
            state.append_to_oplist(pack_op);
        }
    }

    state.record_expansion(&expandargs, current_vec);
    if is_invariant {
        state.invariant_vector_vars.insert(current_vec);
    }
    args[index] = current_vec;
}

/// schedule.py:322-350: Turn a pack of scalar ops into a single vector op.
pub fn turn_into_vector(state: &mut VecScheduleState, pack: &Pack, ops: &[Op]) {
    if pack.members.is_empty() {
        return;
    }
    // schedule.py:324: check_if_pack_supported
    if check_if_pack_supported(state, pack, ops).is_err() {
        return;
    }
    let count = pack.members.len();
    let first_op = &ops[pack.members[0]];

    // schedule.py:325: costmodel.record_pack_savings
    state.costmodel.record_pack_savings(pack, count);

    let Some(vec_opcode) = first_op.opcode.to_vector() else {
        return; // not vectorizable
    };

    // schedule.py:335-336: build args list + prepare_arguments
    let mut args = first_op.getarglist().to_vec();
    prepare_arguments(state, pack, &mut args, ops);

    // schedule.py:337-338: VecOperation(left.vector, args, left, pack.numops())
    // resoperation.py:100-104: copy datatype/bytesize/signed from baseop's vecinfo
    let vi = state.forwarded_vecinfo(first_op);
    let (mut datatype, mut bytesize, signed) = (vi.datatype, vi.bytesize, vi.signed);
    // resoperation.py:105-108 VecOperation typecast override.
    //   if baseop.is_typecast():
    //       ft, tt = baseop.cast_types()
    //       datatype = tt
    //       bytesize = baseop.cast_to_bytesize()
    // INT_SIGNEXT is excluded by the static-bytesize gate (see
    // PRE-EXISTING-ADAPTATION in `vectorization_info_for_op`): the dynamic
    // arg1.value path needs const-pool threading.
    if first_op.opcode.is_typecast() {
        if let Some(bs) = first_op.opcode.cast_to_bytesize_static() {
            let (_ft, tt) = first_op.opcode.cast_types();
            datatype = tt;
            bytesize = bs as i8;
        }
    }
    let mut vecop =
        state.create_vec_op(vec_opcode, &args, datatype, bytesize as i32, signed, count);
    if let Some(d) = first_op.getdescr() {
        vecop.setdescr(d);
    }

    let vecop_pos = vecop.pos.get();
    // schedule.py:340-346: map scalar ops to vector positions
    for (i, &member_idx) in pack.members.iter().enumerate() {
        let op = &ops[member_idx];
        if op.opcode.result_type() == majit_ir::Type::Void {
            continue; // schedule.py:342-343: skip void ops
        }
        let scalar_pos = op.pos.get();
        if !scalar_pos.is_none() {
            state.setvector_of_box(scalar_pos, i, vecop_pos);
            // schedule.py:345-346: only rename for accumulating packs
            if pack.is_accumulating && !op.opcode.is_guard() {
                state.renamer.start_renaming(scalar_pos, vecop_pos);
            }
        }
    }

    // schedule.py:347-348: handle guard failargs
    if first_op.opcode.is_guard() {
        prepare_fail_arguments(state, pack, ops, &mut vecop);
    }

    state.append_to_oplist(vecop);
    assert!(count >= 1); // schedule.py:350
}
