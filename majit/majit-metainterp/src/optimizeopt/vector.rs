/// Vectorization optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/vector.py.
/// This is the core of the vec optimization — it combines dependency.py
/// and schedule.py to rewrite a loop in vectorized form.
///
/// # RPython parity notes
///
/// - In RPython, `VectorizingOptimizer` extends `Optimizer` (the main
///   optimization dispatcher). In Rust, `VectorizingOptimizer` implements
///   the `Optimization` trait (a sub-pass in the `Optimizer` pipeline).
///   This is a PRE-EXISTING-ADAPTATION: the Rust optimization pipeline
///   uses a trait-object chain rather than inheritance. The standalone
///   `optimize_vector()` function provides the RPython-shaped entry point.
///
/// - `CostModel`, `GenericCostModel`, `PackSet`, `isomorphic` live in
///   `schedule.rs` in Rust but in `vector.py` in RPython. This is a
///   PRE-EXISTING-ADAPTATION driven by the Rust module split done before
///   parity enforcement. They are re-exported here.
use majit_ir::vec_set::VecSet;

use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::dependency::DependencyGraph;
use crate::optimizeopt::renamer::Renamer;
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

// Re-exports: these types live in schedule.rs but are defined in vector.py
// in RPython. Re-exporting preserves the public API surface.
pub use crate::optimizeopt::dependency::{Node, schedule_operations};
pub use crate::optimizeopt::schedule::{
    AccumEntry, AccumPack, CostModel, GenericCostModel, GuardAnalysis, NotAProfitableLoop,
    NotAVectorizeableLoop, Pack, PackSet, VecScheduleState, VectorizeError,
    are_adjacent_memory_refs, isomorphic, turn_into_vector, unpack_from_vector,
};

// ── vector.py:35-40: copy_resop ────────────────────────────────────────

/// vector.py:35-40: copy_resop — clone an op, preserving VectorizationInfo.
///
/// In RPython, `get_forwarded()` returns VectorizationInfo if set on the
/// op, and copy_resop re-attaches it to the clone. In Rust, `Op::clone()`
/// already preserves the `vecinfo` field, so this is a trivial clone.
fn copy_resop(op: &Op) -> Op {
    op.clone()
}

// ── vector.py:42-120: VectorLoop ───────────────────────────────────────

/// vector.py:42-120: VectorLoop — wraps a loop body (Label..operations..Jump)
/// for vectorization analysis and transformation.
#[derive(Clone, Debug)]
pub struct VectorLoop {
    /// vector.py:44: self.label = label
    pub label: Op,
    /// vector.py:45: self.inputargs = label.getarglist_copy()
    pub inputargs: Vec<OpRef>,
    /// vector.py:46: self.prefix = []
    pub prefix: Vec<Op>,
    /// vector.py:47: self.prefix_label = None
    pub prefix_label: Option<Op>,
    /// vector.py:49: self.operations = oplist
    pub operations: Vec<Op>,
    /// vector.py:50: self.jump = jump
    pub jump: Op,
    /// vector.py:52: self.align_operations = []
    pub align_operations: Vec<Op>,
}

impl VectorLoop {
    /// vector.py:43-52: __init__(self, label, oplist, jump)
    pub fn new(label: Op, operations: Vec<Op>, jump: Op) -> Self {
        debug_assert_eq!(label.opcode, OpCode::Label);
        debug_assert!(
            jump.opcode == OpCode::Jump,
            "expected Jump, got {:?}",
            jump.opcode
        );
        let inputargs = label.getarglist().to_vec();
        VectorLoop {
            label,
            inputargs,
            prefix: Vec::new(),
            prefix_label: None,
            operations,
            jump,
            align_operations: Vec::new(),
        }
    }

    /// Convenience constructor: extract VectorLoop from a trace by finding
    /// Label..Jump. Not in RPython — the caller splits the trace before
    /// constructing VectorLoop. Kept for backward compatibility with tests.
    pub fn from_trace(ops: &[Op]) -> Option<Self> {
        let label_pos = ops.iter().position(|op| op.opcode == OpCode::Label)?;
        let jump_pos = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
        if jump_pos <= label_pos {
            return None;
        }
        Some(VectorLoop::new(
            ops[label_pos].clone(),
            ops[label_pos + 1..jump_pos].to_vec(),
            ops[jump_pos].clone(),
        ))
    }

    /// vector.py:54-56: setup_vectorization — attach VectorizationInfo to
    /// each operation in the loop body.
    pub fn setup_vectorization(&mut self) {
        for op in &mut self.operations {
            if !op.has_vecinfo() {
                let mut vi = majit_ir::VectorizationInfo::new();
                let tp = op.opcode.result_type();
                let dt = match tp {
                    majit_ir::Type::Float => 'f',
                    majit_ir::Type::Void => '\0',
                    _ => 'i',
                };
                if dt != '\0' {
                    vi.setinfo(dt, -1, true);
                }
                op.set_vecinfo(vi);
            }
        }
    }

    /// vector.py:58-60: teardown_vectorization — remove VectorizationInfo.
    pub fn teardown_vectorization(&mut self) {
        for op in &mut self.operations {
            op.clear_vecinfo();
        }
    }

    /// vector.py:62-92: finaloplist — assemble the complete operation list
    /// for compilation.
    ///
    /// `jitcell_token`: when supplied, allocate fresh `TargetToken`s for
    ///   `self.label` (`reset_label_token` true) and `self.prefix_label` /
    ///   `self.jump`, mirroring `vector.py:62-79`. When `None`, the descr
    ///   block is skipped — equivalent to RPython's `if jitcell_token:` guard.
    /// `reset_label_token`: vector.py:64 — choose between minting a new
    ///   `TargetToken` for the label (true) or pulling the existing jump
    ///   descr (false).
    /// `label`: include `self.label` at the front. When false, follow
    ///   `vector.py:87-90` and clear the vectorize-time scratch
    ///   (`set_forwarded(None)` upstream) from every emitted prefix op plus
    ///   the jump. In majit the vectorize-time scratch lives on
    ///   `VectorizationInfo`, so the equivalent is `clear_vecinfo()`.
    pub fn finaloplist(
        &self,
        jitcell_token: Option<&std::sync::Arc<majit_backend::JitCellToken>>,
        reset_label_token: bool,
        label: bool,
    ) -> Vec<Op> {
        // vector.py:63-79: descr wiring against the owning JitCellToken.
        if let Some(jcell) = jitcell_token {
            // vector.py:64-72
            if reset_label_token {
                let token =
                    std::sync::Arc::new(crate::optimizeopt::unroll::TargetToken::new_loop(0));
                let descr = token.as_jump_target_descr();
                jcell.target_tokens.lock().push(descr.clone());
                self.label.setdescr(descr);
            }
            // else: vector.py:70-72 grabs token = self.jump.getdescr(); the
            // result is only used below to re-setdescr the jump when there's
            // no prefix_label, which is already what's there. Skip.

            // vector.py:73-77: prefix_label gets its own TargetToken, and
            // the jump is rebound to point at it.
            if let Some(ref prefix_label) = self.prefix_label {
                let pre_token =
                    std::sync::Arc::new(crate::optimizeopt::unroll::TargetToken::new_loop(0));
                let pre_descr = pre_token.as_jump_target_descr();
                jcell.target_tokens.lock().push(pre_descr.clone());
                prefix_label.setdescr(pre_descr.clone());
                self.jump.setdescr(pre_descr);
            } else if reset_label_token {
                // vector.py:78-79: with no prefix_label, re-bind jump to the
                // label's freshly-minted token.
                if let Some(label_descr) = self.label.getdescr() {
                    self.jump.setdescr(label_descr);
                }
            }
        }

        // vector.py:80-84
        let mut oplist: Vec<Op> = Vec::new();
        if let Some(ref prefix_label) = self.prefix_label {
            oplist.extend(self.prefix.iter().cloned());
            oplist.push(prefix_label.clone());
        } else if !self.prefix.is_empty() {
            oplist.extend(self.prefix.iter().cloned());
        }
        // vector.py:85-86
        if label {
            oplist.insert(0, self.label.clone());
        }
        // vector.py:87-90: when not emitting the label op (i.e. the prefix
        // is the *only* thing being compiled this round, e.g. a bridge),
        // strip vectorization scratch so nothing leaks into the next pass.
        if !label {
            for op in &oplist {
                op.clear_vecinfo();
            }
            self.jump.clear_vecinfo();
        }
        // vector.py:91
        oplist.extend(self.operations.iter().cloned());
        oplist.push(self.jump.clone());
        oplist
    }

    /// vector.py:94-120: clone — deep-clone the loop with renaming.
    pub fn clone_loop(&self) -> Self {
        let mut renamer = Renamer::new();
        let _label = copy_resop(&self.label);
        let mut prefix = Vec::new();
        for op in &self.prefix {
            let mut newop = copy_resop(op);
            renamer.rename(&mut newop);
            if newop.opcode.result_type() != majit_ir::Type::Void {
                renamer.start_renaming(op.pos.get(), newop.pos.get());
            }
            prefix.push(newop);
        }
        let prefix_label = self.prefix_label.as_ref().map(|pl| {
            let mut newpl = copy_resop(pl);
            renamer.rename(&mut newpl);
            newpl
        });
        let mut operations = Vec::new();
        for op in &self.operations {
            let mut newop = copy_resop(op);
            renamer.rename(&mut newop);
            if newop.opcode.result_type() != majit_ir::Type::Void {
                renamer.start_renaming(op.pos.get(), newop.pos.get());
            }
            operations.push(newop);
        }
        let mut jump = copy_resop(&self.jump);
        renamer.rename(&mut jump);
        let mut loop_ = VectorLoop::new(copy_resop(&self.label), operations, jump);
        loop_.prefix = prefix;
        loop_.prefix_label = prefix_label;
        loop_
    }

    /// Number of ops in the loop body (excluding Label and Jump).
    pub fn body_len(&self) -> usize {
        self.operations.len()
    }
}

// ── vector.py:122-173: optimize_vector ─────────────────────────────────

/// vector.py:122-173: optimize_vector — top-level entry point.
///
/// Creates a VectorizingOptimizer, runs vectorization on the loop, and
/// returns the rewritten op list. The loop is modified in place.
///
/// PRE-EXISTING-ADAPTATION: In RPython this receives `metainterp_sd`,
/// `jitdriver_sd`, `warmstate`, `loop_info`, `loop_ops`, `jitcell_token`.
/// In Rust we receive a `VectorLoop` directly and pass cost/size params.
/// The `metainterp_sd`-dependent parts (profiler counting, logger) are
/// handled via the `Optimization` trait impl when used in the pipeline.
pub fn optimize_vector(
    loop_: &mut VectorLoop,
    cost_threshold: i32,
    vec_size: usize,
) -> Result<Vec<Op>, VectorizeError> {
    // vector.py:126-128
    if loop_.operations.is_empty() {
        return Err(VectorizeError::NotVectorizeable);
    }

    // vector.py:134 `version = info.snapshot(loop)` — keep an untouched
    // clone so that *any* downstream failure (NotAVectorizeableLoop /
    // NotAProfitableLoop / panic-equivalent) restores the caller-visible
    // VectorLoop to its pre-vectorize shape. The clone is only used on the
    // error path; on success we hand back the vectorized ops directly.
    let version = loop_.clone_loop();

    // vector.py:135: loop.setup_vectorization()
    loop_.setup_vectorization();

    let result = (|| -> Result<Vec<Op>, VectorizeError> {
        // vector.py:142-143
        let mut opt = VectorizingOptimizer::new_with_params(cost_threshold, vec_size);
        opt.run_optimization(loop_)
    })();

    // vector.py:172: finally: loop.teardown_vectorization()
    loop_.teardown_vectorization();

    if result.is_err() {
        // vector.py:155 / :160: `return loop_info, version.loop.finaloplist()`.
        // Restore the pre-vectorize ops into loop_ so the caller can resume
        // from a clean state if it wants to inspect the loop further.
        *loop_ = version;
    }

    result
}

// ── vector.py:175-205: user_loop_bail_fast_path ────────────────────────

/// vector.py:175-205: user_loop_bail_fast_path — quick pre-check.
///
/// Returns `true` if the loop should be SKIPPED (bailed on) for
/// vectorization. In RPython, `user_code and user_loop_bail_fast_path()`
/// is checked before entering the optimizer.
pub fn user_loop_bail_fast_path(loop_: &VectorLoop) -> bool {
    let mut _resop_count = 0;
    let mut _vector_instr = 0;
    let mut _guard_count = 0;
    // vector.py:183: at_least_one_array_access = True  (RPython bug — always True,
    // because line 194 only ever re-assigns True.  Match upstream literal.)
    let mut at_least_one_array_access = true;

    for op in &loop_.operations {
        // vector.py:185-186: skip jit debug ops
        if op.opcode.is_jit_debug() {
            continue;
        }
        // vector.py:188-189: count vectorizable non-guard ops
        if op.opcode.to_vector().is_some() && !op.opcode.is_guard() {
            _vector_instr += 1;
        }
        _resop_count += 1;
        // vector.py:193-194: is_primitive_array_access
        if op.opcode.is_getarrayitem()
            || op.opcode.is_setarrayitem()
            || matches!(
                op.opcode,
                OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore
            )
        {
            at_least_one_array_access = true;
        }
        // vector.py:196-197: bail on calls
        if op.opcode.is_call() || op.opcode.is_call_assembler() {
            return true;
        }
        // vector.py:199-200
        if op.opcode.is_guard() {
            _guard_count += 1;
        }
    }
    // vector.py:202-203
    if !at_least_one_array_access {
        return true;
    }
    false
}

// ── vector.py:207-600: VectorizingOptimizer ────────────────────────────

/// vector.py:207-600: VectorizingOptimizer — the vectorization optimizer.
///
/// In RPython, this extends `Optimizer` and is the top-level optimizer for
/// the vector pass. In Rust, it implements `Optimization` and is used as
/// a sub-pass in the `Optimizer` pipeline (PRE-EXISTING-ADAPTATION).
///
/// The RPython-shaped methods (`run_optimization`, `unroll_loop_iterations`,
/// etc.) are provided alongside the `Optimization` trait impl.
pub struct VectorizingOptimizer {
    /// vector.py:215: self.packset = None
    packset: Option<PackSet>,
    /// vector.py:216: self.unroll_count = 0
    pub unroll_count: usize,
    /// vector.py:217: self.smallest_type_bytes = 0
    smallest_type_bytes: usize,
    /// vector.py:218: self.orig_label_args = None
    orig_label_args: Option<Vec<OpRef>>,
    /// vector.py:213: self.cost_threshold = cost_threshold
    cost_threshold: i32,
    /// vector.py:214: self.vector_ext.vec_size()
    vec_size: usize,
    /// vector.py:244: self.vector_ext.should_align_unroll
    /// True on x86 SSE (default); a future backend abstraction can set
    /// false for platforms where alignment unrolling is not beneficial.
    should_align_unroll: bool,

    // ── Rust Optimization trait fields (PRE-EXISTING-ADAPTATION) ──
    // These support the sub-pass integration path where VectorizingOptimizer
    // is used inside an Optimizer pipeline via the Optimization trait.
    /// Buffered loop body ops (populated by propagate_forward).
    body_ops: Vec<Op>,
    /// Whether we're inside a Label..Jump loop body.
    in_loop: bool,
    /// schedule.py:669: label inputargs — populated on Label entry.
    label_args: Vec<OpRef>,
    /// Deferred profiler counter: OPT_VECTORIZE_TRY.
    pub(crate) opt_vectorize_try_emitted: usize,
    /// Deferred profiler counter: OPT_VECTORIZED.
    pub(crate) opt_vectorized_emitted: usize,
}

impl VectorizingOptimizer {
    /// vector.py:210-218: __init__ — default constructor for sub-pass use.
    pub fn new() -> Self {
        VectorizingOptimizer {
            packset: None,
            unroll_count: 0,
            smallest_type_bytes: 0,
            orig_label_args: None,
            cost_threshold: 0,
            vec_size: 16, // SSE default
            should_align_unroll: true,
            body_ops: Vec::new(),
            in_loop: false,
            label_args: Vec::new(),
            opt_vectorize_try_emitted: 0,
            opt_vectorized_emitted: 0,
        }
    }

    /// Constructor with explicit parameters (for standalone optimize_vector).
    pub fn new_with_params(cost_threshold: i32, vec_size: usize) -> Self {
        let mut opt = Self::new();
        opt.cost_threshold = cost_threshold;
        opt.vec_size = vec_size;
        opt
    }

    // ── vector.py:220-271: run_optimization ────────────────────────────

    /// vector.py:220-271: run_optimization — the main vectorization pipeline.
    ///
    /// 1. Find smallest type → determine unroll count
    /// 2. Analyse index calculations → reorder for guard hoisting
    /// 3. Unroll the loop
    /// 4. Build dependency graph → find adjacent memory refs
    /// 5. Extend and combine packset
    /// 6. Schedule with cost model
    /// 7. Guard strengthening
    /// 8. Re-schedule for cleanup
    pub fn run_optimization(&mut self, loop_: &mut VectorLoop) -> Result<Vec<Op>, VectorizeError> {
        // vector.py:221
        self.orig_label_args = Some(loop_.label.getarglist().to_vec());

        // vector.py:222
        self.linear_find_smallest_type(loop_);
        let byte_count = self.smallest_type_bytes;

        // vector.py:224
        let vsize = self.vec_size;

        // vector.py:227-235: bail checks
        if vsize == 0 {
            return Err(VectorizeError::NotVectorizeable);
        }
        if byte_count == 0 {
            return Err(VectorizeError::NotVectorizeable);
        }
        if loop_.label.opcode != OpCode::Label {
            return Err(VectorizeError::NotVectorizeable);
        }

        // vector.py:237-240: analyse_index_calculations → reorder
        let constant_of = |_opref: OpRef| -> Option<i64> { None };
        if let Some(graph) = self.analyse_index_calculations(loop_, &constant_of) {
            let schedule = schedule_operations(&graph);
            if schedule.len() == loop_.operations.len() {
                let scheduled: Vec<Op> = schedule
                    .iter()
                    .map(|&i| loop_.operations[i].clone())
                    .collect();
                loop_.operations = scheduled;
            }
        }

        // vector.py:243-247: unroll.
        // RPython: `align_unroll = self.unroll_count == 1 and
        //           self.vector_ext.should_align_unroll`
        // should_align_unroll is a backend flag (True on x86 SSE, False on
        // some other backends). We default to true since only x86_64 is
        // supported; the flag is stored on self so a future backend
        // abstraction can override it.
        self.unroll_count = Self::get_unroll_count(byte_count, vsize);
        let align_unroll = self.unroll_count == 1 && self.should_align_unroll;
        loop_.unroll_loop_iterations(self.unroll_count, align_unroll);

        // vector.py:250-253: vectorize — build graph, find adjacent memory refs
        let graph = DependencyGraph::build(&loop_.operations, &constant_of);
        // VecScheduleState is created before find_adjacent_memory_refs/
        // extend_packset because PackSet::can_be_packed now consults it via
        // isomorphic_with_state (vector.py: packset.can_be_packed reaches
        // state for accumulation/invariant lookups; pre-state form was a
        // pre-rebase fork).
        let start_pos = loop_
            .operations
            .iter()
            .map(|op| op.pos.get().raw())
            .max()
            .unwrap_or(0)
            + 1;
        let mut sched_state = VecScheduleState::new(start_pos);
        // vector.py:135 loop.setup_vectorization()
        sched_state.setup_vectorization(&loop_.operations, &constant_of);
        // vector.py:606-609 CostModel.__init__: savings = 0, threshold stored
        // separately. Initializing savings = self.cost_threshold inverted the
        // gate — a positive threshold made profitable() trivially true.
        let costmodel = CostModel {
            min_pack_size: 2,
            pack_cost: 2,
            scalar_save: 1,
            savings: 0,
        };
        sched_state.costmodel = costmodel;

        self.find_adjacent_memory_refs(&graph, loop_, &mut sched_state);

        // vector.py:253-254: extend and combine — combine_packset raises
        // NotAVectorizeableLoop on an empty packset (vector.py:468-470).
        self.extend_packset(&graph, &mut sched_state);
        self.combine_packset()?;

        // vector.py:254-258: schedule with cost model
        let packset = self.packset.take().unwrap_or_default();

        // Populate inputargs/seen from label args
        for &arg in loop_.label.getarglist().iter() {
            sched_state.inputargs.insert(arg, ());
        }
        let mut seen: VecSet<OpRef> = loop_.label.getarglist().iter().copied().collect();

        // accumulate_prepare for accumulation packs
        for pack in &packset.packs {
            if !pack.is_accumulating {
                continue;
            }
            let first_op = &loop_.operations[pack.members[0]];
            if first_op.opcode.is_guard() {
                continue;
            }
            let pos = pack.position.max(0) as usize;
            let seed = if pos < first_op.num_args() {
                first_op.arg(pos)
            } else {
                OpRef::NONE
            };
            let operator = pack.operator.unwrap_or('+');
            for &member_idx in &pack.members {
                let op = &loop_.operations[member_idx];
                if op.opcode.is_guard() {
                    continue;
                }
                sched_state.accumulation.insert(
                    op.pos.get(),
                    AccumEntry {
                        seed,
                        operator,
                        accum_opcode: pack.scalar_opcode,
                    },
                );
            }
            let is_float = first_op.opcode.result_type() == majit_ir::Type::Float;
            if is_float {
                return Err(VectorizeError::NotVectorizeable);
            }
            let datatype = 'i';
            let bytesize: i32 = loop_
                .operations
                .iter()
                .find(|op| op.pos.get() == seed)
                .and_then(|op| op.get_vecinfo())
                .map(|vi| vi.getbytesize() as i32)
                .unwrap_or(8);
            let vec_reg_size: i32 = self.vec_size as i32;
            let count = (vec_reg_size / bytesize) as usize;
            let signed = true;

            let vec_create =
                sched_state.create_vec_op(OpCode::VecI, &[], datatype, bytesize, signed, count);
            let zero_vec = vec_create.pos.get();
            sched_state.invariant_oplist.push(vec_create);

            let xor_op = sched_state.create_vec_op(
                OpCode::VecIntXor,
                &[zero_vec, zero_vec],
                datatype,
                bytesize,
                signed,
                count,
            );
            let zeroed_vec = xor_op.pos.get();
            sched_state.invariant_oplist.push(xor_op);

            let zero_const = OpRef::const_int(0);
            let one_const = OpRef::const_int(1);
            let pack_op = sched_state.create_vec_op(
                OpCode::VecPackI,
                &[zeroed_vec, seed, zero_const, one_const],
                datatype,
                bytesize,
                signed,
                count,
            );
            let seed_vec = pack_op.pos.get();
            sched_state.invariant_oplist.push(pack_op);

            sched_state.accumulation.insert(
                seed,
                AccumEntry {
                    seed,
                    operator,
                    accum_opcode: pack.scalar_opcode,
                },
            );
            sched_state.setvector_of_box(seed, 0, seed_vec);
            sched_state.renamer.start_renaming(seed, seed_vec);
        }

        // Build node→pack mapping
        let mut node_to_pack: crate::optimizeopt::vec_assoc::VecAssoc<usize, usize> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        for (pi, group) in packset.packs.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        let mut pack_emitted = vec![false; packset.packs.len()];
        let mut pack_visited_count = vec![0usize; packset.packs.len()];

        let scheduled_order = schedule_operations(&graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                pack_visited_count[pack_idx] += 1;
                let pack = &packset.packs[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    pack_emitted[pack_idx] = true;
                    for &member_idx in &pack.members {
                        let mut member_op = loop_.operations[member_idx].clone();
                        pre_emit_guard_accum(&sched_state, &mut member_op);
                        sched_state.renamer.rename(&mut member_op);
                        seen.insert(member_op.pos.get());
                        loop_.operations[member_idx] = member_op;
                    }
                    turn_into_vector(&mut sched_state, pack, &loop_.operations);
                }
            } else {
                let mut scalar_op = loop_.operations[node_idx].clone();
                pre_emit_guard_accum(&sched_state, &mut scalar_op);
                sched_state.renamer.rename(&mut scalar_op);
                ensure_args_unpacked(&mut sched_state, &mut scalar_op, &mut seen);
                seen.insert(scalar_op.pos.get());
                sched_state.append_to_oplist(scalar_op);
            }
        }

        // schedule.py:762 VecScheduleState.post_schedule — moves
        // invariant_oplist into loop.prefix and routes invariant_vector_vars
        // through prefix_label/jump renaming. That post-pass is not yet
        // ported. If anything is sitting in either list (e.g. expand()'d
        // splat ops or accumulator seed packs), splicing it into the loop
        // body — which is what the original concatenation below did — is a
        // silent semantic divergence. Bail until post_schedule lands.
        if !sched_state.invariant_oplist.is_empty() || !sched_state.invariant_vector_vars.is_empty()
        {
            return Err(VectorizeError::NotVectorizeable);
        }

        // vector.py:257-258: profitability check — distinct error variant
        // so the caller (or future GuardStrengthenOpt logging) can react to
        // cost-model rejection separately from a structural bail.
        if !sched_state.costmodel.profitable() {
            return Err(VectorizeError::NotProfitable);
        }

        // vector.py:267-269: extra_before_label = loop.align_operations;
        // for op in loop.align_operations: op.set_forwarded(None).
        // We hand the align_operations back through `loop_.align_operations`
        // (already populated by `unroll_loop_iterations` on the align arm);
        // clearing vecinfo matches the upstream `set_forwarded(None)` reset
        // so post-vectorize passes don't see stale VectorizationInfo.
        for op in &loop_.align_operations {
            op.clear_vecinfo();
        }

        let result = sched_state.oplist;

        // Update loop operations for finaloplist
        loop_.operations = result.clone();

        Ok(result)
    }

    // ── vector.py:273-344: unroll_loop_iterations ──────────────────────

    /// vector.py:359-367: linear_find_smallest_type — scan ops for the
    /// smallest array element byte size to determine SIMD width.
    pub fn linear_find_smallest_type(&mut self, loop_: &VectorLoop) {
        for op in &loop_.operations {
            if op.opcode.is_getarrayitem()
                || op.opcode.is_setarrayitem()
                || matches!(
                    op.opcode,
                    OpCode::RawLoadI | OpCode::RawLoadF | OpCode::RawStore
                )
            {
                if let Some(descr) = op.getdescr() {
                    if let Some(ad) = descr.as_array_descr() {
                        let item_size = ad.item_size();
                        if self.smallest_type_bytes == 0 || item_size < self.smallest_type_bytes {
                            self.smallest_type_bytes = item_size;
                        }
                    }
                }
            }
        }
    }

    /// vector.py:369-376: get_unroll_count — compute how many times to
    /// unroll based on SIMD register width and smallest type.
    pub fn get_unroll_count(smallest_type_bytes: usize, simd_reg_bytes: usize) -> usize {
        if smallest_type_bytes == 0 {
            return 0;
        }
        let count = simd_reg_bytes / smallest_type_bytes;
        count.saturating_sub(1) // already unrolled once
    }

    // ── vector.py:346-357: copy_guard_descr ────────────────────────────

    /// vector.py:346-357: copy_guard_descr — clone guard descriptor and
    /// rename fail args during unrolling.
    fn copy_guard_descr(renamer: &Renamer, copied_op: &mut Op) {
        // vector.py:349-350: descr.clone() — already cloned by copy_resop
        // vector.py:351: failargs = renamer.rename_failargs(copied_op, clone=True)
        if let Some(fail_args) = copied_op.getfailargs() {
            let renamed = renamer.rename_failargs(&fail_args);
            copied_op.setfailargs(renamed.into());
        }
    }

    // ── vector.py:378-402: find_adjacent_memory_refs ───────────────────

    /// vector.py:378-402: find_adjacent_memory_refs — seed the packset
    /// with pairs of adjacent memory accesses.
    fn find_adjacent_memory_refs(
        &mut self,
        graph: &DependencyGraph,
        _loop: &VectorLoop,
        state: &mut VecScheduleState,
    ) {
        let _vec_size = self.vec_size;
        let mut packset = PackSet::new();

        // vector.py:391-402: for each pair of memory refs, check adjacency
        let memory_refs: Vec<(usize, &crate::optimizeopt::dependency::MemoryRef)> = graph
            .memory_refs
            .iter()
            .map(|(&idx, mref)| (idx, mref))
            .collect();

        for &(node_a, memref_a) in &memory_refs {
            for &(node_b, memref_b) in &memory_refs {
                if node_a == node_b {
                    continue;
                }
                // vector.py:399: memref_a.is_adjacent_after(memref_b)
                if memref_a.is_adjacent_after(memref_b) {
                    // vector.py:400-401: packset.can_be_packed(node_a, node_b, None, False)
                    match packset.can_be_packed(state, node_a, node_b, None, false, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        _ => {}
                    }
                }
            }
        }

        self.packset = Some(packset);
    }

    // ── vector.py:404-458: extend_packset / follow_def_uses / follow_use_defs

    /// vector.py:404-425: extend_packset — follow dependency chains to find
    /// more candidates to put into pairs.
    pub fn extend_packset(&mut self, graph: &DependencyGraph, state: &mut VecScheduleState) {
        let packset = match self.packset.as_mut() {
            Some(ps) => ps,
            None => return,
        };
        let mut pack_count = packset.num_packs();
        loop {
            // vector.py:411-415: follow_def_uses for each 2-pack
            let num_packs = packset.packs.len();
            for i in 0..num_packs {
                if packset.packs[i].members.len() == 2 {
                    let pack_snap = packset.packs[i].clone();
                    Self::follow_def_uses(packset, &pack_snap, graph, state);
                }
            }
            if pack_count == packset.num_packs() {
                // vector.py:417-423: no new packs from def-uses, try use-defs
                pack_count = packset.num_packs();
                let num_packs = packset.packs.len();
                for i in 0..num_packs {
                    if packset.packs[i].members.len() == 2 {
                        let pack_snap = packset.packs[i].clone();
                        Self::follow_use_defs(packset, &pack_snap, graph, state);
                    }
                }
                if pack_count == packset.num_packs() {
                    break;
                }
            }
            pack_count = packset.num_packs();
        }
    }

    /// vector.py:427-442: follow_use_defs — for a 2-pack, check if
    /// dependencies of leftmost/rightmost can form new pairs.
    fn follow_use_defs(
        packset: &mut PackSet,
        pack: &Pack,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        debug_assert!(pack.members.len() == 2);
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();

        // vector.py:429-430: for ldep in pack.leftmost(True).depends()
        let l_deps: Vec<usize> = graph.nodes[left_idx].deps.clone();
        let r_deps: Vec<usize> = graph.nodes[right_idx].deps.clone();

        for &l_dep in &l_deps {
            for &r_dep in &r_deps {
                // vector.py:434-437: left = lnode.getoperation();
                // args = pack.leftmost().getarglist(); if left not in args: continue
                let dep_opref = graph.nodes[l_dep].op.pos.get();
                let left_args = graph.nodes[left_idx].op.getarglist();
                if !left_args.contains(&dep_opref) {
                    continue;
                }
                let l_op = &graph.nodes[l_dep].op;
                let r_op = &graph.nodes[r_dep].op;
                // vector.py:438-439: isomorphic and lnode.is_before(rnode)
                if isomorphic(l_op, r_op) && l_dep < r_dep {
                    match packset.can_be_packed(state, l_dep, r_dep, Some(pack), false, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    }

    /// vector.py:444-458: follow_def_uses — for a 2-pack, check if users
    /// of leftmost/rightmost can form new pairs via can_be_packed.
    fn follow_def_uses(
        packset: &mut PackSet,
        pack: &Pack,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        debug_assert!(pack.members.len() == 2);
        let left_idx = pack.members[0];
        let right_idx = *pack.members.last().unwrap();
        let left_opref = graph.nodes[left_idx].op.pos.get();

        // vector.py:446-447: for ldep in pack.leftmost(node=True).provides()
        let l_users: Vec<usize> = graph.nodes[left_idx].users.clone();
        let r_users: Vec<usize> = graph.nodes[right_idx].users.clone();

        for &l_user in &l_users {
            for &r_user in &r_users {
                // vector.py:451-453: left = pack.leftmost()
                // args = lnode.getoperation().getarglist()
                // if left not in args: continue
                if !graph.nodes[l_user].op.getarglist().contains(&left_opref) {
                    continue;
                }
                let l_op = &graph.nodes[l_user].op;
                let r_op = &graph.nodes[r_user].op;
                // vector.py:454-455: isomorphic and lnode.is_before(rnode)
                if isomorphic(l_op, r_op) && l_user < r_user {
                    match packset.can_be_packed(state, l_user, r_user, Some(pack), true, graph) {
                        Ok(Some(pair)) => packset.add_pack(pair),
                        Err(_) => return,
                        _ => {}
                    }
                }
            }
        }
    }

    // ── vector.py:460-494: combine_packset ─────────────────────────────

    /// vector.py:460-496: combine_packset — merge adjacent 2-packs into
    /// larger packs, then split overloaded packs.
    pub fn combine_packset(&mut self) -> Result<(), NotAVectorizeableLoop> {
        let packset = match self.packset.as_mut() {
            Some(ps) => ps,
            None => return Err(NotAVectorizeableLoop),
        };

        // vector.py:468-470: empty packset → raise NotAVectorizeableLoop
        if packset.packs.is_empty() {
            return Err(NotAVectorizeableLoop);
        }

        // vector.py:474-494: iterative merge
        loop {
            let len_before = packset.packs.len();
            packset.try_merge_packs();
            if packset.packs.len() == len_before {
                break;
            }
        }

        // vector.py:496: self.packset.split_overloaded_packs(self.cpu.vector_ext)
        // PRE-EXISTING-ADAPTATION: split_overloaded_packs not yet ported.
        // RPython splits packs that exceed the vector register size and
        // removes packs that are too small (< FULL load). This requires
        // pack_load() and Pack.FULL which depend on vectorization info
        // infrastructure not yet available in Rust.
        Ok(())
    }

    // ── vector.py:515-521: schedule ────────────────────────────────────

    /// vector.py:515-521: schedule — run the scheduler on the given state.
    fn schedule_state(_state: &mut VecScheduleState, _graph: &DependencyGraph) {
        // vector.py:516: state.prepare() — handled by caller
        // vector.py:517-518: scheduler.walk_and_emit(state) — scheduling
        //   is done inline in run_optimization via schedule_operations
        // vector.py:520: state.post_schedule() — handled by caller
    }

    // ── vector.py:523-583: analyse_index_calculations ──────────────────

    /// vector.py:523-583: analyse_index_calculations — move guarding
    /// instructions (and their dependencies) to the loop header.
    ///
    /// This ensures guards fail "early" and relax dependencies, which is
    /// a prerequisite for vectorization.
    ///
    /// PRE-EXISTING-ADAPTATION: The full RPython implementation requires:
    /// - DependencyGraph.imaginary_node() — synthetic graph nodes
    /// - Node.iterate_paths() — path enumeration with blacklist
    /// - Path.is_always_pure() — purity analysis along paths
    /// - Node.remove_edge_to() / edge_to() — graph mutation
    /// These dependency.rs primitives are not yet ported. Until they are,
    /// return None unconditionally — the earlier "zero-dep guard" heuristic
    /// did not actually rewire the graph the way RPython does, and feeding
    /// the unmodified graph back to the caller as a reschedule basis was a
    /// silent divergence. mark_guard is similarly stubbed.
    fn analyse_index_calculations(
        &self,
        _loop_: &VectorLoop,
        _constant_of: &dyn Fn(OpRef) -> Option<i64>,
    ) -> Option<DependencyGraph> {
        None
    }

    // ── vector.py:585-599: mark_guard ──────────────────────────────────

    /// vector.py:585-599: mark_guard — marks a guard as an early exit
    /// by attaching a CompileLoopVersionDescr and setting failargs to
    /// the label's input args.
    ///
    /// PRE-EXISTING-ADAPTATION: CompileLoopVersionDescr is not yet ported
    /// to Rust. When it is, this should create the descr and attach it.
    fn mark_guard(&self, _guard_idx: usize, _loop_: &VectorLoop) {
        // vector.py:588-594: create CompileLoopVersionDescr, copy attrs
        // vector.py:595-599: set failargs to label.getarglist_copy()
        // Requires CompileLoopVersionDescr from compile.rs — not yet ported.
    }

    // ── Optimization trait helper: try_vectorize ───────────────────────

    /// Attempt to vectorize the buffered loop body (Optimization trait path).
    ///
    /// This is the sub-pass equivalent of run_optimization, used when
    /// VectorizingOptimizer is embedded in an Optimizer pipeline.
    fn try_vectorize(&mut self, ctx: &mut OptContext) -> Option<Vec<Op>> {
        if self.body_ops.len() < 4 {
            return None;
        }

        let constant_of = |opref: OpRef| -> Option<i64> { ctx.get_constant_int(opref) };

        let start_pos = ctx.new_operations.len() as u32 + self.body_ops.len() as u32;
        let mut sched_state = VecScheduleState::new(start_pos);
        // vector.py:135 loop.setup_vectorization() — stamps
        // VectorizationInfo on each op. INT_SIGNEXT reads arg1's const
        // value for bytesize (resoperation.py:181); the constant_of
        // resolver feeds that through so the inline vecinfo slot matches
        // PyPy's VectorizationInfo(op) constructor.
        sched_state.setup_vectorization(&self.body_ops, &constant_of);

        // Phase 1: Schedule operations for ILP before packing.
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        let schedule = schedule_operations(&dep_graph);
        if schedule.len() == self.body_ops.len() {
            let scheduled: Vec<Op> = schedule.iter().map(|&i| self.body_ops[i].clone()).collect();
            self.body_ops = scheduled;
            sched_state.setup_vectorization(&self.body_ops, &constant_of);
        }

        // Phase 2: Rebuild dependency graph and find packs.
        let dep_graph = DependencyGraph::build(&self.body_ops, &constant_of);
        let seed_packs = dep_graph.find_packable_groups();
        if seed_packs.is_empty() {
            return None;
        }
        let mut pack_set = PackSet::new();
        for pack in seed_packs {
            pack_set.add_pack(pack);
        }
        Self::extend_packset_static(&mut pack_set, &dep_graph, &mut sched_state);
        Self::combine_packset_static(&mut pack_set);
        let profitable = pack_set.packs;
        if profitable.is_empty() {
            return None;
        }

        // schedule.py:666-670: prepare() — populate inputargs and seen
        for &arg in &self.label_args {
            sched_state.inputargs.insert(arg, ());
        }
        let mut seen: VecSet<OpRef> = self.label_args.iter().copied().collect();

        // accumulate_prepare
        for pack in &profitable {
            if !pack.is_accumulating {
                continue;
            }
            let first_op = &self.body_ops[pack.members[0]];
            if first_op.opcode.is_guard() {
                continue;
            }
            let pos = pack.position.max(0) as usize;
            let seed = if pos < first_op.num_args() {
                first_op.arg(pos)
            } else {
                OpRef::NONE
            };
            let operator = pack.operator.unwrap_or('+');
            for &member_idx in &pack.members {
                let op = &self.body_ops[member_idx];
                if op.opcode.is_guard() {
                    continue;
                }
                sched_state.accumulation.insert(
                    op.pos.get(),
                    AccumEntry {
                        seed,
                        operator,
                        accum_opcode: pack.scalar_opcode,
                    },
                );
            }
            let is_float = first_op.opcode.result_type() == majit_ir::Type::Float;
            if is_float {
                return None;
            }
            let datatype = 'i';
            let bytesize: i32 = self
                .body_ops
                .iter()
                .find(|op| op.pos.get() == seed)
                .and_then(|op| op.get_vecinfo())
                .map(|vi| vi.getbytesize() as i32)
                .unwrap_or(8);
            let vec_reg_size: i32 = self.vec_size as i32;
            let count = (vec_reg_size / bytesize) as usize;
            let signed = true;

            let vec_create =
                sched_state.create_vec_op(OpCode::VecI, &[], datatype, bytesize, signed, count);
            let zero_vec = vec_create.pos.get();
            sched_state.invariant_oplist.push(vec_create);

            let xor_op = sched_state.create_vec_op(
                OpCode::VecIntXor,
                &[zero_vec, zero_vec],
                datatype,
                bytesize,
                signed,
                count,
            );
            let zeroed_vec = xor_op.pos.get();
            sched_state.invariant_oplist.push(xor_op);

            let zero_const = OpRef::const_int(0);
            let one_const = OpRef::const_int(1);
            let pack_op = sched_state.create_vec_op(
                OpCode::VecPackI,
                &[zeroed_vec, seed, zero_const, one_const],
                datatype,
                bytesize,
                signed,
                count,
            );
            let seed_vec = pack_op.pos.get();
            sched_state.invariant_oplist.push(pack_op);

            sched_state.accumulation.insert(
                seed,
                AccumEntry {
                    seed,
                    operator,
                    accum_opcode: pack.scalar_opcode,
                },
            );
            sched_state.setvector_of_box(seed, 0, seed_vec);
            sched_state.renamer.start_renaming(seed, seed_vec);
        }

        // Build node→pack mapping
        let mut node_to_pack: crate::optimizeopt::vec_assoc::VecAssoc<usize, usize> =
            crate::optimizeopt::vec_assoc::VecAssoc::new();
        for (pi, group) in profitable.iter().enumerate() {
            for &idx in &group.members {
                node_to_pack.insert(idx, pi);
            }
        }

        let mut pack_emitted = vec![false; profitable.len()];
        let mut pack_visited_count = vec![0usize; profitable.len()];

        let scheduled_order = schedule_operations(&dep_graph);
        for &node_idx in &scheduled_order {
            if let Some(&pack_idx) = node_to_pack.get(&node_idx) {
                pack_visited_count[pack_idx] += 1;
                let pack = &profitable[pack_idx];
                let all_ready = pack_visited_count[pack_idx] == pack.members.len();

                if all_ready && !pack_emitted[pack_idx] {
                    pack_emitted[pack_idx] = true;
                    for &member_idx in &pack.members {
                        let mut member_op = self.body_ops[member_idx].clone();
                        pre_emit_guard_accum(&sched_state, &mut member_op);
                        sched_state.renamer.rename(&mut member_op);
                        seen.insert(member_op.pos.get());
                        self.body_ops[member_idx] = member_op;
                    }
                    turn_into_vector(&mut sched_state, pack, &self.body_ops);
                }
            } else {
                let mut scalar_op = self.body_ops[node_idx].clone();
                pre_emit_guard_accum(&sched_state, &mut scalar_op);
                sched_state.renamer.rename(&mut scalar_op);
                ensure_args_unpacked(&mut sched_state, &mut scalar_op, &mut seen);
                seen.insert(scalar_op.pos.get());
                sched_state.append_to_oplist(scalar_op);
            }
        }

        // schedule.py:762 post_schedule not yet ported — same bail rationale
        // as run_optimization (do not splice invariant_oplist /
        // invariant_vector_vars into the loop body).
        if !sched_state.invariant_oplist.is_empty() || !sched_state.invariant_vector_vars.is_empty()
        {
            return None;
        }

        if !sched_state.costmodel.profitable() {
            return None;
        }

        Some(sched_state.oplist)
    }

    // ── Static variants for extend/combine (used by try_vectorize) ─────

    fn extend_packset_static(
        pack_set: &mut PackSet,
        graph: &DependencyGraph,
        state: &mut VecScheduleState,
    ) {
        let mut pack_count = pack_set.num_packs();
        loop {
            let num_packs = pack_set.packs.len();
            for i in 0..num_packs {
                if pack_set.packs[i].members.len() == 2 {
                    let pack_snap = pack_set.packs[i].clone();
                    Self::follow_def_uses(pack_set, &pack_snap, graph, state);
                }
            }
            if pack_count == pack_set.num_packs() {
                pack_count = pack_set.num_packs();
                let num_packs = pack_set.packs.len();
                for i in 0..num_packs {
                    if pack_set.packs[i].members.len() == 2 {
                        let pack_snap = pack_set.packs[i].clone();
                        Self::follow_use_defs(pack_set, &pack_snap, graph, state);
                    }
                }
                if pack_count == pack_set.num_packs() {
                    break;
                }
            }
            pack_count = pack_set.num_packs();
        }
    }

    fn combine_packset_static(pack_set: &mut PackSet) {
        if pack_set.packs.is_empty() {
            return;
        }
        loop {
            let len_before = pack_set.packs.len();
            pack_set.try_merge_packs();
            if pack_set.packs.len() == len_before {
                break;
            }
        }
    }
}

// ── VectorLoop: unroll_loop_iterations ─────────────────────────────────

impl VectorLoop {
    /// vector.py:273-344: unroll_loop_iterations — unroll the loop body
    /// `count` times with proper renaming.
    ///
    /// `align_unroll_once` (vector.py:273) requests one extra alignment
    /// unroll. When set: `count` is bumped by one before the unroll runs;
    /// after the first iteration a fresh `LABEL` is materialised and
    /// installed as `self.label`; the *original* body is moved to
    /// `self.align_operations` (consumed before the unrolled loop) while
    /// `self.operations` is replaced by the unrolled sequence. When unset,
    /// `self.operations` becomes `original + unrolled` (the default body
    /// shape).
    pub fn unroll_loop_iterations(&mut self, count: usize, align_unroll_once: bool) {
        if count == 0 {
            return;
        }
        // vector.py:284 — bump count once for the alignment pass.
        let unroll_count = if align_unroll_once { count + 1 } else { count };
        let original_body = self.operations.clone();
        let label_args = self.label.getarglist_copy();
        let jump_args = self.jump.getarglist_copy();

        // vector.py:281-283: prohibited opcodes — not duplicated during unroll
        let prohibit = [
            OpCode::GuardFutureCondition,
            OpCode::GuardNotInvalidated,
            OpCode::DebugMergePoint,
        ];

        let mut renamer = Renamer::new();
        let mut unrolled = Vec::new();
        // vector.py:292 `new_label = loop.label` — the label install-target
        // is the existing label by default; the align-unroll arm overwrites
        // it with a freshly minted LABEL after the first body copy.
        let mut new_label = self.label.clone();

        let base_offset = original_body
            .iter()
            .map(|op| op.pos.get().raw())
            .max()
            .unwrap_or(0)
            + 1;

        for u in 0..unroll_count {
            // vector.py:296-301: fill rename map: label args → jump args
            for i in 0..label_args.len().min(jump_args.len()) {
                let la = label_args[i];
                let ja = renamer.rename_box(jump_args[i]);
                if la != ja {
                    renamer.start_renaming(la, ja);
                }
            }

            let offset = base_offset + (u as u32) * (original_body.len() as u32);

            // vector.py:303-322: copy and rename each op
            for op in &original_body {
                if prohibit.contains(&op.opcode) {
                    continue;
                }
                let mut copied_op = copy_resop(op);

                // vector.py:307-310: new result box → rename mapping
                let new_pos = op.pos.get().with_raw(op.pos.get().raw() + offset);
                if !op.pos.get().is_none() {
                    renamer.start_renaming(op.pos.get(), new_pos);
                }
                copied_op.pos.set(new_pos);

                // vector.py:312-315: rename args
                for i in 0..copied_op.num_args() {
                    let renamed = renamer.rename_box(copied_op.arg(i));
                    copied_op.setarg(i, renamed);
                }

                // vector.py:319-320: rename guard fail args
                if copied_op.opcode.is_guard() {
                    VectorizingOptimizer::copy_guard_descr(&renamer, &mut copied_op);
                }

                unrolled.push(copied_op);
            }

            // vector.py:324-328 — after the first iteration of an align
            // unroll, mint a fresh LABEL using the same descr and arglist
            // as the original label, then run the renamer over it so its
            // args track the rename state at this point.
            if align_unroll_once && u == 0 {
                let mut minted = Op::new(OpCode::Label, &label_args);
                if let Some(descr) = self.label.getdescr() {
                    minted.setdescr(descr);
                }
                for i in 0..minted.num_args() {
                    let renamed = renamer.rename_box(minted.arg(i));
                    minted.setarg(i, renamed);
                }
                new_label = minted;
            }
        }

        // vector.py:334-337: update jump args with final renaming
        for i in 0..self.jump.num_args() {
            let renamed = renamer.rename_box(self.jump.arg(i));
            self.jump.setarg(i, renamed);
        }

        // vector.py:339-344
        self.label = new_label;
        if align_unroll_once {
            self.align_operations = original_body;
            self.operations = unrolled;
        } else {
            self.operations.extend(unrolled);
        }
    }
}

// ── schedule.py helpers used by the vectorizer ─────────────────────────
// These functions are from schedule.py in RPython, not vector.py.
// They are placed here because they are called from the vectorizer's
// scheduling logic in try_vectorize / run_optimization.

/// schedule.py:638-658: pre_emit_guard_accum — guard accumulation stitching.
/// For guard ops, scan failargs for accumulation variables. When found:
///   - attach AccumInfo to the guard descriptor (schedule.py:654-655)
///   - replace the failarg with the renamed seed (schedule.py:656-657)
fn pre_emit_guard_accum(state: &VecScheduleState, op: &mut Op) {
    if !op.opcode.is_guard() {
        return;
    }
    if let Some(fa) = op.getfailargs() {
        let mut new_fa = fa.clone();
        for (fi, arg) in new_fa.iter_mut().enumerate() {
            if arg.is_none() {
                continue;
            }
            if let Some(entry) = state.accumulation.get(arg) {
                let location = state
                    .getvector_of_box(*arg)
                    .map(|(_, vec_ref)| vec_ref)
                    .unwrap_or(*arg);
                if let Some(descr) = op.getdescr() {
                    if let Some(fail_descr) = descr.as_fail_descr() {
                        fail_descr.attach_vector_info(majit_ir::AccumInfo {
                            prev: None,
                            failargs_pos: fi,
                            variable: *arg,
                            location,
                            accum_operation: entry.operator,
                            scalar: OpRef::NONE,
                        });
                    }
                }
                *arg = entry.seed;
            }
        }
        op.setfailargs(new_fa);
    }
}

/// schedule.py:697-736: ensure_args_unpacked — unpack vector-boxed args
/// for a scalar op, respecting seen/invariant/accumulation state.
fn ensure_args_unpacked(state: &mut VecScheduleState, op: &mut Op, seen: &mut VecSet<OpRef>) {
    // schedule.py:702-706: unpack immediate-use args
    for j in 0..op.num_args() {
        let arg = op.arg(j);
        if arg.is_constant() || seen.contains(&arg) {
            continue;
        }
        if let Some((pos, vec_ref)) = state.getvector_of_box(arg) {
            if state.invariant_vector_vars.contains(&vec_ref) {
                continue;
            }
            if state.accumulation.contains_key(&arg) {
                continue;
            }
            let unpacked = unpack_from_vector(state, vec_ref, pos, 1);
            state.renamer.start_renaming(arg, unpacked);
            seen.insert(unpacked);
            op.setarg(j, unpacked);
        }
    }
    // schedule.py:708-716: unpack guard failargs
    if op.opcode.is_guard() {
        if let Some(mut fail_args) = op.getfailargs() {
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
            op.setfailargs(fail_args);
        }
    }
}

// ── Optimization trait impl (PRE-EXISTING-ADAPTATION) ──────────────────
// In RPython, VectorizingOptimizer extends Optimizer and is called via
// optimize_vector(). In Rust, it participates in the Optimizer pipeline
// as an Optimization sub-pass. This impl bridges the two worlds.

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
                self.label_args = op.getarglist().to_vec();
                OptimizationResult::Emit(op.clone())
            }
            OpCode::Jump if self.in_loop => {
                // vector.py:139
                self.opt_vectorize_try_emitted = self.opt_vectorize_try_emitted.saturating_add(1);
                if let Some(vectorized) = self.try_vectorize(ctx) {
                    // vector.py:146
                    self.opt_vectorized_emitted = self.opt_vectorized_emitted.saturating_add(1);
                    for vop in vectorized {
                        ctx.emit(vop);
                    }
                } else {
                    for body_op in &self.body_ops {
                        ctx.emit(body_op.clone());
                    }
                }
                self.in_loop = false;
                self.body_ops.clear();
                OptimizationResult::Emit(op.clone())
            }
            _ => {
                if self.in_loop {
                    self.body_ops.push(op.clone());
                    OptimizationResult::Remove
                } else {
                    OptimizationResult::PassOn
                }
            }
        }
    }

    fn setup(&mut self) {
        self.body_ops.clear();
        self.in_loop = false;
        self.packset = None;
        self.unroll_count = 0;
        self.smallest_type_bytes = 0;
        self.orig_label_args = None;
    }

    fn name(&self) -> &'static str {
        "vectorize_simd"
    }

    fn drain_profiler_counters(&mut self, profiler: &crate::jitprof::JitProfiler) {
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

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRef};

    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos
                .set(OpRef::op_typed(base + i as u32, op.result_type()));
        }
    }

    // ── VectorLoop tests ──

    #[test]
    fn test_vector_loop_from_trace() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(OpCode::Label, &[OpRef::input_arg_int(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::int_op(2), OpRef::input_arg_int(102)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(3)]),
        ];
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
        }
        let vloop = VectorLoop::from_trace(&ops).unwrap();
        assert_eq!(vloop.body_len(), 2); // IntAdd + IntMul
        assert_eq!(vloop.label.opcode, OpCode::Label);
        assert_eq!(vloop.jump.opcode, OpCode::Jump);
        assert_eq!(vloop.inputargs.len(), 1);
    }

    #[test]
    fn test_vector_loop_new() {
        let label = Op::new(OpCode::Label, &[OpRef::input_arg_int(100)]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        )];
        let jump = Op::new(OpCode::Jump, &[OpRef::int_op(0)]);
        let vloop = VectorLoop::new(label, ops, jump);
        assert_eq!(vloop.body_len(), 1);
        assert_eq!(vloop.inputargs, vec![OpRef::input_arg_int(100)]);
        assert!(vloop.prefix.is_empty());
        assert!(vloop.prefix_label.is_none());
        assert!(vloop.align_operations.is_empty());
    }

    #[test]
    fn test_vector_loop_finaloplist() {
        let label = Op::new(OpCode::Label, &[OpRef::input_arg_int(100)]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        )];
        let jump = Op::new(OpCode::Jump, &[OpRef::int_op(0)]);
        let vloop = VectorLoop::new(label, ops, jump);

        let with_label = vloop.finaloplist(None, true, true);
        assert_eq!(with_label.len(), 3); // Label + IntAdd + Jump
        assert_eq!(with_label[0].opcode, OpCode::Label);

        let without_label = vloop.finaloplist(None, true, false);
        assert_eq!(without_label.len(), 2); // IntAdd + Jump
    }

    #[test]
    fn test_user_loop_bail_fast_path_no_array() {
        let label = Op::new(OpCode::Label, &[OpRef::input_arg_int(100)]);
        let ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        )];
        let jump = Op::new(OpCode::Jump, &[OpRef::int_op(0)]);
        let vloop = VectorLoop::new(label, ops, jump);
        // vector.py:183 initializes at_least_one_array_access = True and only
        // re-assigns True, so the "no array access" branch is unreachable.
        // Match upstream literal: no array access does NOT bail.
        assert!(!user_loop_bail_fast_path(&vloop));
    }

    // ── Dependency graph tests ──

    #[test]
    fn test_dep_graph_basic() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.nodes[1].deps.contains(&0));
        assert!(!graph.nodes[2].deps.contains(&0));
    }

    #[test]
    fn test_dep_graph_no_self_dep() {
        let mut ops = vec![Op::new(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::input_arg_int(101)],
        )];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        assert!(graph.nodes[0].deps.is_empty());
    }

    // ── Pack group tests ──

    #[test]
    fn test_find_packable_groups() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
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
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();
        assert!(groups.is_empty());
    }

    #[test]
    fn test_different_opcodes_not_packed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let groups = graph.find_packable_groups();
        for g in &groups {
            assert!(g.members.len() >= 2);
        }
    }

    #[test]
    fn test_three_independent_ops() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMul,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::input_arg_int(104), OpRef::input_arg_int(105)],
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
            members: vec![0, 1, 2, 3],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));

        let group5 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2, 3, 4],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group5));
    }

    #[test]
    fn test_cost_model_too_small() {
        let cm = CostModel::new();
        let group = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0],
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
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        };
        assert!(!cm.is_profitable(&group));

        let group3 = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1, 2],
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
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(OpCode::Finish, &[OpRef::int_op(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut majit_ir::VecAssoc::new(),
            1024,
            crate::r#box::BoxPool::new(),
        );
        assert!(!result.is_empty());
    }

    #[test]
    fn test_vectorize_pass_preserves_structure() {
        use crate::optimizeopt::optimizer::Optimizer;

        let mut ops = vec![
            Op::new(
                OpCode::Label,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::int_op(1), OpRef::int_op(2)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(VectorizingOptimizer::new()));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut majit_ir::VecAssoc::new(),
            1024,
            crate::r#box::BoxPool::new(),
        );
        assert!(result.iter().any(|op| op.opcode == OpCode::Label));
        assert!(result.iter().any(|op| op.opcode == OpCode::Jump));
    }

    // ── Scheduler tests ──

    #[test]
    fn test_schedule_respects_dependencies() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::int_op(1), OpRef::input_arg_int(101)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 3);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_schedule_maximizes_parallelism() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 2);
        assert!(sched.contains(&0));
        assert!(sched.contains(&1));
    }

    #[test]
    fn test_schedule_prioritizes_critical_path() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::int_op(1), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        assert!(pos_a < pos_d, "A (height 3) should precede D (height 1)");
    }

    #[test]
    fn test_schedule_diamond() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntMul,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntSub,
                &[OpRef::int_op(0), OpRef::input_arg_int(102)],
            ),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(1), OpRef::int_op(2)]),
        ];
        assign_positions(&mut ops, 0);

        let graph = DependencyGraph::build(&ops, &|_| None);
        let sched = schedule_operations(&graph);

        assert_eq!(sched.len(), 4);
        let pos_a = sched.iter().position(|&x| x == 0).unwrap();
        let pos_b = sched.iter().position(|&x| x == 1).unwrap();
        let pos_c = sched.iter().position(|&x| x == 2).unwrap();
        let pos_d = sched.iter().position(|&x| x == 3).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
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
    fn test_pack_set_merge() {
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
        assert_eq!(ps.total_ops(), 3);
    }

    #[test]
    fn test_pack_set_no_merge_disjoint() {
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
        assert_eq!(ps.num_packs(), 2);
    }

    // ── isomorphic + can_be_packed + accumulates_pair tests ──

    #[test]
    fn test_isomorphic_same_opcode() {
        let a = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        );
        let b = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
        );
        assert!(isomorphic(&a, &b));
    }

    #[test]
    fn test_isomorphic_different_opcode() {
        let a = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        );
        let b = Op::new(
            OpCode::IntSub,
            &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
        );
        assert!(!isomorphic(&a, &b));
    }

    #[test]
    fn test_can_be_packed_independent_seed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps.can_be_packed(&mut VecScheduleState::new(0), 0, 1, None, false, &graph);
        assert!(result.is_ok());
        let pack = result.unwrap();
        assert!(pack.is_some());
        let pack = pack.unwrap();
        assert_eq!(pack.members, vec![0, 1]);
        assert!(!pack.is_accumulating);
    }

    #[test]
    fn test_can_be_packed_dependent_no_origin() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::int_op(0), OpRef::input_arg_int(101)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);
        let ps = PackSet::new();

        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 0, 1, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_can_be_packed_accumulation() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(200), OpRef::int_op(0)],
            ),
            Op::new(OpCode::IntAdd, &[OpRef::int_op(2), OpRef::int_op(1)]),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        let origin = Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        };

        let ps = PackSet::new();
        let result = ps.can_be_packed(
            &mut VecScheduleState::new(0),
            2,
            3,
            Some(&origin),
            true,
            &graph,
        );
        assert!(result.is_ok());
        let pack = result.unwrap();
        if let Some(p) = pack {
            assert!(p.is_accumulating);
            assert_eq!(p.operator, Some('+'));
            assert_eq!(p.position, 0);
        }
    }

    #[test]
    fn test_can_be_packed_blocks_already_packed() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(102), OpRef::input_arg_int(103)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(104), OpRef::input_arg_int(105)],
            ),
        ];
        assign_positions(&mut ops, 0);
        let graph = DependencyGraph::build(&ops, &|_| None);

        let mut ps = PackSet::new();
        ps.add_pack(Pack {
            scalar_opcode: OpCode::IntAdd,
            vector_opcode: OpCode::VecIntAdd,
            members: vec![0, 1],
            is_accumulating: false,
            position: -1,
            operator: None,
        });
        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 0, 2, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
        let result = ps
            .can_be_packed(&mut VecScheduleState::new(0), 2, 1, None, false, &graph)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_generic_cost_model() {
        let model = GenericCostModel::new();
        assert!(model.op_cost(OpCode::GetarrayitemGcI) > model.op_cost(OpCode::IntAdd));
        assert!(model.op_cost(OpCode::FloatTrueDiv) >= model.op_cost(OpCode::FloatAdd));
    }

    #[test]
    fn test_guard_analysis_hoistable() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::input_arg_int(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::int_op(1)]),
        ];
        let mut positioned = ops;
        for (i, op) in positioned.iter_mut().enumerate() {
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
        }
        let analysis = GuardAnalysis::analyze(&positioned);
        assert_eq!(analysis.hoistable.len(), 1);
        assert_eq!(analysis.hoistable[0], 0);
        assert_eq!(analysis.body_guards.len(), 1);
        assert_eq!(analysis.body_guards[0], 2);
    }

    #[test]
    fn drain_profiler_counters_folds_opt_vectorize_try_and_opt_vectorized_into_profiler() {
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
        Optimization::drain_profiler_counters(&mut vopt, &prof);
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZE_TRY), Some(4));
        assert_eq!(prof.get_counter(counters::OPT_VECTORIZED), Some(1));
    }

    // ── Unroll tests ──

    #[test]
    fn test_unroll_loop_iterations() {
        let mut label = Op::new(
            OpCode::Label,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        );
        label.pos.set(OpRef::op_typed(0, majit_ir::Type::Void));
        let mut body_op = Op::new(
            OpCode::IntAdd,
            &[OpRef::input_arg_int(100), OpRef::input_arg_int(101)],
        );
        body_op.pos.set(OpRef::int_op(1));
        let mut jump = Op::new(OpCode::Jump, &[OpRef::int_op(1), OpRef::input_arg_int(101)]);
        jump.pos.set(OpRef::op_typed(2, majit_ir::Type::Void));

        let mut vloop = VectorLoop::new(label, vec![body_op], jump);
        assert_eq!(vloop.body_len(), 1);

        vloop.unroll_loop_iterations(2, false);
        // Original 1 + 2 unrolled copies = 3
        assert_eq!(vloop.body_len(), 3);
    }

    #[test]
    fn test_get_unroll_count() {
        // 16 byte SIMD register, 4 byte elements → 3 additional unrolls
        assert_eq!(VectorizingOptimizer::get_unroll_count(4, 16), 3);
        // 16 byte SIMD register, 8 byte elements → 1 additional unroll
        assert_eq!(VectorizingOptimizer::get_unroll_count(8, 16), 1);
        // 0 byte smallest → 0
        assert_eq!(VectorizingOptimizer::get_unroll_count(0, 16), 0);
    }
}
