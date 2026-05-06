/// Short preamble: minimal operations to replay when entering a peeled loop
/// from a bridge rather than from the preamble.
///
/// Translated from rpython/jit/metainterp/optimizeopt/shortpreamble.py.
///
/// After loop peeling, the optimizer processes the peeled iteration (preamble)
/// and discovers facts about loop-carried values (constants, types, bounds).
/// The loop body is then optimized assuming those facts hold.
///
/// When a bridge later jumps to the loop header (Label), it doesn't have
/// the preamble's context. The "short preamble" is a minimal set of operations
/// — typically guards — that re-establish the facts the loop body depends on.
///
/// Without the short preamble, bridges would need to either:
/// 1. Re-execute the entire preamble (wasteful), or
/// 2. Be conservative and lose optimizations (slow)
///
/// # Structure
///
/// ```text
/// [preamble]               ← full first iteration, optimizer learns facts
///   Label(...)             ← loop header, short preamble stored here
///   [short preamble ops]   ← replayed when a bridge enters here
/// [optimized body]         ← relies on facts from preamble
///   Jump(...)              ← back-edge
/// ```
///
/// # Integration
///
/// The `ShortPreambleBuilder` is used during optimization. When the optimizer
/// processes the preamble and finds guards/operations that establish facts
/// the body depends on, it records them. At the Label, the builder finalizes
/// into a `ShortPreamble` that is stored alongside the compiled loop.
use std::collections::{HashMap, HashSet};

use majit_ir::{Op, OpCode, OpRef};

use crate::optimizeopt::virtualstate::VirtualState;

/// A recorded preamble operation that bridges must replay.
///
/// Each entry captures an operation from the preamble that was either:
/// - A guard that the body assumes always holds
/// - A pure operation whose result the body uses as a known value
/// - A type/class check that enables downstream specialization
#[derive(Clone, Debug)]
pub struct ShortPreambleOp {
    /// The operation to replay (with args referencing label arg indices).
    pub op: Op,
    /// Which label arg indices this op's arguments map to.
    /// Maps from op arg position to label arg index.
    pub arg_mapping: Vec<(usize, usize)>,
    /// Which label arg indices this op's fail args map to.
    /// Maps from fail_arg position to label arg index.
    pub fail_arg_mapping: Vec<(usize, usize)>,
}

/// The complete short preamble for a peeled loop.
///
/// Stored alongside the Label's target token. When a bridge targets
/// this label, the short preamble ops are prepended to establish
/// the optimization context the loop body expects.
#[derive(Clone, Debug)]
pub struct ShortPreamble {
    /// Operations to prepend when entering the loop from a bridge.
    /// These are guards and setup ops with args referencing label arg indices.
    pub ops: Vec<ShortPreambleOp>,
    /// Input args of the short preamble Label.
    /// RPython stores the full short preamble as [Label(short_inputargs), ...].
    pub inputargs: Vec<OpRef>,
    /// Extra loop-header values carried by the short preamble Jump.
    /// RPython appends `sb.used_boxes` to the loop label and jumps with
    /// `args + extra`, where `extra` is the remapped version of these boxes.
    pub used_boxes: Vec<OpRef>,
    /// Preamble producer results used by the short preamble's own trailing JUMP.
    /// RPython keeps this separate from `used_boxes`: the loop contract carries
    /// body boxes, while the short preamble JUMP reuses the corresponding
    /// preamble-produced values.
    pub jump_args: Vec<OpRef>,
    /// The exported virtual state at the loop header (from the preamble's exit).
    /// Used to check bridge compatibility and generate additional guards.
    pub exported_state: Option<VirtualState>,
    /// RPython parity: constant values referenced by short preamble ops.
    /// In RPython, short preamble ops embed Const objects (GC-tracked) that
    /// survive across compilations. In pyre, OpRef indices reference the
    /// loop's constant pool. This map captures (value, type) for each
    /// constant OpRef so bridges can re-register them in their own pool.
    pub constants: HashMap<u32, (i64, majit_ir::Type)>,
    /// RPython parity: PtrInfo for each inputarg, from Phase 1 export.
    /// shortpreamble.py:414-425: preamble_op.set_forwarded(info)
    /// Used by inline_short_preamble to propagate PtrInfo to jump_args
    /// so guards added by use_box are eliminated as redundant.
    pub inputarg_infos: Vec<Option<crate::optimizeopt::info::PtrInfo>>,
    /// RPython parity: Phase 1 inputargs preserved across Extended rebuilds.
    /// In RPython, short ops reference renamed inputargs (new Box objects)
    /// that are stable across compilations. In majit, ops keep original
    /// preamble OpRefs. When the Extended builder rebuilds with different
    /// inputargs (Phase 2 label_args), ops may reference Phase 1 OpRefs
    /// that aren't in the new inputargs. This field stores the original
    /// Phase 1 inputargs so inline_short_preamble can map them to jump_args.
    pub phase1_inputargs: Option<Vec<OpRef>>,
}

impl ShortPreamble {
    /// Create an empty short preamble (no extra operations needed).
    pub fn empty() -> Self {
        ShortPreamble {
            ops: Vec::new(),
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state: None,
            constants: HashMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
        }
    }

    /// Whether this short preamble has any operations to replay.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Number of operations in the short preamble.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Generate the operations to prepend when a bridge enters the loop.
    ///
    /// `bridge_args` are the OpRefs that the bridge provides as values
    /// for each label arg. The short preamble ops are instantiated with
    /// these concrete references.
    pub fn instantiate(&self, bridge_args: &[OpRef]) -> Vec<Op> {
        let mut result: Vec<Op> = Vec::with_capacity(self.ops.len());

        for entry in &self.ops {
            let mut op = entry.op.clone();

            // Remap arguments: replace label arg indices with bridge's concrete refs
            for (arg_pos, label_idx) in &entry.arg_mapping {
                if let Some(bridge_ref) = bridge_args.get(*label_idx) {
                    if *arg_pos < op.args.len() {
                        op.args[*arg_pos] = *bridge_ref;
                    }
                }
            }

            if let Some(ref mut fail_args) = op.fail_args {
                for (fail_arg_pos, label_idx) in &entry.fail_arg_mapping {
                    if let Some(bridge_ref) = bridge_args.get(*label_idx) {
                        if *fail_arg_pos < fail_args.len() {
                            fail_args[*fail_arg_pos] = *bridge_ref;
                        }
                    }
                }
            }

            if op.opcode.is_guard_overflow()
                && !matches!(result.last(), Some(prev) if prev.opcode.is_ovf())
            {
                continue;
            }

            result.push(op);
        }

        result
    }
}

impl ShortPreamble {
    /// shortpreamble.py: apply to bridge — prepend instantiated short preamble
    /// ops to a bridge trace, creating a complete trace that the optimizer
    /// can process with full preamble context.
    pub fn apply_to_bridge(&self, bridge_args: &[OpRef], bridge_ops: &[Op]) -> Vec<Op> {
        let mut result = self.instantiate(bridge_args);
        result.extend_from_slice(bridge_ops);
        result
    }

    /// Count guards in the short preamble.
    pub fn num_guards(&self) -> usize {
        self.ops.iter().filter(|e| e.op.opcode.is_guard()).count()
    }

    /// Count pure ops in the short preamble.
    pub fn num_pure_ops(&self) -> usize {
        self.ops
            .iter()
            .filter(|e| e.op.opcode.is_always_pure())
            .count()
    }
}

/// Collector that extracts short preamble operations from an already-built
/// preamble trace.
///
/// This is intentionally separate from RPython's `ShortPreambleBuilder`.
/// The RPython builder consumes exported short boxes while building phase 2;
/// this collector just turns a preamble section into a `ShortPreamble`.
pub struct CollectedShortPreambleBuilder {
    /// Raw ops collected during the preamble phase (before Label).
    raw_ops: Vec<Op>,
    /// Map from preamble OpRef to label arg index (set when Label is found).
    preamble_to_label_arg: HashMap<OpRef, usize>,
    /// Whether the builder is still collecting (before Label).
    active: bool,
}

impl CollectedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedShortPreambleBuilder {
            raw_ops: Vec::new(),
            preamble_to_label_arg: HashMap::new(),
            active: true,
        }
    }

    /// Set up the mapping from preamble OpRefs to label arg indices.
    ///
    /// Called when the Label is encountered. `label_args` are the OpRefs
    /// that the Label carries (= the loop-carried values from the preamble).
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.preamble_to_label_arg.clear();
        for (i, opref) in label_args.iter().enumerate() {
            self.preamble_to_label_arg.insert(*opref, i);
        }
        self.active = false; // Switch from preamble to body phase
    }

    /// Record a guard from the preamble that the body depends on.
    ///
    /// The guard's arguments should reference preamble OpRefs that
    /// are carried across the Label as label args.
    pub fn add_preamble_guard(&mut self, op: &Op) {
        if !self.active {
            return; // Only collect during preamble phase
        }

        // Only record guard operations
        if !op.opcode.is_guard() {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Record any preamble operation (guard or pure) that establishes
    /// a fact the body depends on.
    pub fn add_preamble_op(&mut self, op: &Op) {
        if !self.active {
            return;
        }

        self.raw_ops.push(op.clone());
    }

    /// Finalize the builder into a ShortPreamble.
    ///
    /// Called after the Label has been processed and the mapping is set.
    /// Computes arg mappings using the preamble-to-label-arg map.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let entries = self
            .raw_ops
            .into_iter()
            .map(|op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in op.args.iter().enumerate() {
                    if let Some(&label_idx) = self.preamble_to_label_arg.get(arg_ref) {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = &op.fail_args {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(&label_idx) = self.preamble_to_label_arg.get(fail_arg_ref) {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op,
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        // Reconstruct label_args order from preamble_to_label_arg mapping
        let mut inputargs_by_idx: Vec<(usize, OpRef)> = self
            .preamble_to_label_arg
            .iter()
            .map(|(&opref, &idx)| (idx, opref))
            .collect();
        inputargs_by_idx.sort_by_key(|(idx, _)| *idx);
        let inputargs: Vec<OpRef> = inputargs_by_idx.into_iter().map(|(_, r)| r).collect();

        ShortPreamble {
            ops: entries,
            inputargs,
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
            constants: HashMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
        }
    }
}

impl Default for CollectedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of preamble operations.
///
/// shortpreamble.py: PreambleOp, HeapOp, PureOp, LoopInvariantOp, GuardOp
/// Each type determines how the operation is replayed when a bridge enters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PreambleOpKind {
    /// shortpreamble.py: PreambleOp — base class for all preamble operations.
    /// A generic preamble operation (guard or other).
    Guard,
    /// shortpreamble.py: ShortInputArg — renamed inputarg for a label slot.
    InputArg,
    /// shortpreamble.py: HeapOp — a heap read (GETFIELD_GC, GETARRAYITEM_GC)
    /// that was cached during the preamble. On bridge entry, the field/array
    /// must be re-read to populate the cache.
    Heap,
    /// shortpreamble.py: PureOp — a pure operation whose result was used
    /// in the loop body. On bridge entry, the pure op is re-computed.
    Pure,
    /// shortpreamble.py: LoopInvariantOp — a CALL_LOOPINVARIANT that was
    /// cached for the loop iteration. On bridge entry, re-execute the call.
    LoopInvariant,
}

/// Extended preamble operation with classification.
///
/// shortpreamble.py: used by ShortBoxes and ExtendedShortPreambleBuilder
/// to track which operations need replay and how.
#[derive(Clone, Debug)]
pub struct PreambleOp {
    /// The operation to replay.
    pub op: Op,
    /// Classification of this operation.
    pub kind: PreambleOpKind,
    /// Index of the argument in the label (None if not a label arg).
    pub label_arg_idx: Option<usize>,
    /// RPython shortpreamble.py: whether this producer was assigned an
    /// invented SameAs name because another producer won the original slot.
    pub invented_name: bool,
    /// Original result box this invented name aliases, if any.
    pub same_as_source: Option<OpRef>,
}

impl PreambleOp {
    /// shortpreamble.py: add_op_to_short(sb) — per-kind logic.
    ///
    /// For HeapOp: reconstruct the getfield/getarrayitem with remapped args.
    /// For PureOp: reconstruct the pure op (promoting to CALL_PURE if call).
    /// For LoopInvariantOp: reconstruct as CALL_LOOPINVARIANT.
    pub fn add_op_to_short(&self, sb: &mut ShortBoxes) -> Option<ProducedShortOp> {
        let preamble_op = match &self.kind {
            PreambleOpKind::InputArg | PreambleOpKind::Guard => self.op.clone(),
            PreambleOpKind::Heap => {
                let mut op = self.op.clone();
                let preamble_arg = sb.produce_arg(self.op.arg(0))?;
                if self.op.opcode.is_getfield() {
                    op.args = vec![preamble_arg].into();
                } else {
                    // shortpreamble.py:99-102: GETARRAYITEM keeps the
                    // original constant index; only the array/base arg goes
                    // through produce_arg().
                    op.args = vec![preamble_arg, self.op.arg(1)].into();
                }
                op
            }
            PreambleOpKind::Pure => {
                let args = self
                    .op
                    .args
                    .iter()
                    .map(|&arg| sb.produce_arg(arg))
                    .collect::<Option<Vec<_>>>()?;
                let mut op = self.op.clone();
                op.args = args.into_iter().collect();
                if op.opcode.is_call() {
                    op.opcode = match op.opcode {
                        OpCode::CallI => OpCode::CallPureI,
                        OpCode::CallR => OpCode::CallPureR,
                        OpCode::CallF => OpCode::CallPureF,
                        OpCode::CallN => OpCode::CallPureN,
                        other => other,
                    };
                }
                op
            }
            PreambleOpKind::LoopInvariant => {
                let args = self
                    .op
                    .args
                    .iter()
                    .map(|&arg| sb.produce_arg(arg))
                    .collect::<Option<Vec<_>>>()?;
                let mut op = self.op.clone();
                op.args = args.into_iter().collect();
                op.opcode = match op.opcode {
                    OpCode::CallI => OpCode::CallLoopinvariantI,
                    OpCode::CallR => OpCode::CallLoopinvariantR,
                    OpCode::CallF => OpCode::CallLoopinvariantF,
                    OpCode::CallN => OpCode::CallLoopinvariantN,
                    other => other,
                };
                op
            }
        };
        Some(ProducedShortOp {
            kind: self.kind.clone(),
            preamble_op,
            invented_name: self.invented_name,
            same_as_source: self.same_as_source,
        })
    }
}

/// shortpreamble.py: ShortBoxes — tracks which values from the preamble
/// are "boxed" into the short preamble. Maps label arg indices to
/// the operations that produce them.
#[derive(Clone, Debug, Default)]
pub struct ShortBoxes {
    /// Mapping from exported label arg box to its position.
    pub label_arg_positions: HashMap<OpRef, usize>,
    /// shortpreamble.py: potential_ops
    potential_ops: HashMap<OpRef, PotentialShortOp>,
    /// Ordered insertion list for potential ops, matching shortpreamble.py's
    /// OrderedDict iteration contract.
    potential_order: Vec<OpRef>,
    /// shortpreamble.py: produced_short_boxes
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
    /// Production order for exported short ops.
    produced_order: Vec<OpRef>,
    /// shortpreamble.py: const_short_boxes
    const_short_boxes: Vec<PreambleOp>,
    /// RPython shortpreamble.py: Const boxes are directly admissible in
    /// produce_arg(). majit models constants as OpRef entries in OptContext,
    /// so we track which OpRefs correspond to constants here.
    known_constants: HashSet<OpRef>,
    /// shortpreamble.py: short_inputargs
    short_inputargs: Vec<OpRef>,
    /// shortpreamble.py: boxes_in_production
    boxes_in_production: HashSet<OpRef>,
    /// Fresh synthetic names for invented short-box aliases.
    next_synthetic_pos: u32,
    /// The number of label args.
    pub num_label_args: usize,
}

#[derive(Clone, Debug)]
enum PotentialShortOp {
    Preamble(PreambleOp),
    Compound(CompoundOp),
}

impl PotentialShortOp {
    fn add_op_to_short(&self, sb: &mut ShortBoxes) -> Option<ProducedShortOp> {
        match self {
            PotentialShortOp::Preamble(op) => op.add_op_to_short(sb),
            PotentialShortOp::Compound(compound) => {
                let produced = compound.flatten(sb, Vec::new());
                if produced.is_empty() {
                    None
                } else {
                    let index = ShortBoxes::_pick_op_index(&produced, true);
                    let chosen = produced[index].clone();
                    for (i, mut alt) in produced.into_iter().enumerate() {
                        if i == index {
                            continue;
                        }
                        let alias = OpRef::from_raw(sb.next_synthetic_pos);
                        sb.next_synthetic_pos += 1;
                        alt.preamble_op.pos = alias;
                        alt.invented_name = true;
                        alt.same_as_source = Some(compound.res);
                        sb.produced_short_boxes.insert(alias, alt.clone());
                        sb.produced_order.push(alias);
                    }
                    Some(chosen)
                }
            }
        }
    }
}

impl ShortBoxes {
    pub fn new(num_label_args: usize) -> Self {
        ShortBoxes {
            label_arg_positions: HashMap::new(),
            potential_ops: HashMap::new(),
            potential_order: Vec::new(),
            produced_short_boxes: HashMap::new(),
            produced_order: Vec::new(),
            const_short_boxes: Vec::new(),
            known_constants: HashSet::new(),
            short_inputargs: Vec::new(),
            boxes_in_production: HashSet::new(),
            next_synthetic_pos: 0,
            num_label_args,
        }
    }

    pub fn with_label_args(label_args: &[OpRef]) -> Self {
        let mut boxes = Self::new(label_args.len());
        for (idx, &arg) in label_args.iter().enumerate() {
            boxes.label_arg_positions.insert(arg, idx);
            boxes.short_inputargs.push(arg);
            // Skip OpRef::NONE sentinels: they are placeholders for slots with
            // no value (e.g. virtual fields the runtime PtrInfo did not yet
            // populate at JUMP time). Including u32::MAX would saturate
            // next_synthetic_pos and overflow on the first invented_name alloc.
            // RPython Box identity sidesteps this — make_inputargs there returns
            // only real Boxes.
            if !arg.is_none() {
                boxes.next_synthetic_pos =
                    boxes.next_synthetic_pos.max(arg.raw().saturating_add(1));
            }
        }
        boxes
    }

    pub fn lookup_label_arg(&self, opref: OpRef) -> Option<usize> {
        self.label_arg_positions.get(&opref).copied()
    }

    /// RPython parity: check if opref is reachable in the short preamble.
    pub fn is_reachable(&self, opref: OpRef) -> bool {
        self.label_arg_positions.contains_key(&opref)
            || self.known_constants.contains(&opref)
            || self.potential_ops.contains_key(&opref)
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        self.known_constants.insert(opref);
        self.next_synthetic_pos = self.next_synthetic_pos.max(opref.raw().saturating_add(1));
    }

    pub fn note_known_constants_from_ctx(&mut self, ctx: &crate::optimizeopt::OptContext) {
        // RPython shortpreamble.py only exposes real `Const` objects here.
        // In majit those live in the constant namespace as typed Const*
        // OpRefs, preserving history.py:220/261/307 `box.type`.
        // Ordinary OpRefs that happen to be known-constant for the current
        // iteration must stay trace-local boxes, or short-preamble import can
        // leak one iteration's guard knowledge into the next.
        for (&const_idx, value) in &ctx.const_pool {
            let tp = match value {
                majit_ir::Value::Int(_) => majit_ir::Type::Int,
                majit_ir::Value::Float(_) => majit_ir::Type::Float,
                majit_ir::Value::Ref(_) => majit_ir::Type::Ref,
                majit_ir::Value::Void => panic!("short preamble cannot import a ConstVoid"),
            };
            self.note_known_constant(OpRef::const_typed(const_idx, tp));
        }
    }

    fn add_op(&mut self, result: OpRef, pop: PotentialShortOp) {
        if !self.potential_ops.contains_key(&result) {
            self.potential_order.push(result);
        }
        self.next_synthetic_pos = self.next_synthetic_pos.max(result.raw().saturating_add(1));
        self.potential_ops.insert(result, pop);
    }

    /// Add a pure operation as a short-box candidate.
    /// shortpreamble.py: sb.add_pure_op(op)
    pub fn add_pure_op(&mut self, op: Op) {
        let result = op.pos;
        self.add_potential_op(self.lookup_label_arg(result), op, PreambleOpKind::Pure);
    }

    /// shortpreamble.py:369-374 add_heap_op(op, getfield_op)
    ///
    /// `op.pos` is the box the GETFIELD/GETARRAYITEM produces. If that box is
    /// a constant, route to `const_short_boxes` (RPython:
    /// `if isinstance(op, Const): self.const_short_boxes.append(HeapOp(op, getfield_op))`).
    /// Otherwise it joins `potential_ops` as a heap candidate.
    pub fn add_heap_op(&mut self, op: Op) {
        let result = op.pos;
        if result.is_constant() || self.known_constants.contains(&result) {
            // shortpreamble.py:371-373: const_short_boxes.append(HeapOp(...))
            let label_arg_idx = self.lookup_label_arg(result);
            self.const_short_boxes.push(PreambleOp {
                op,
                kind: PreambleOpKind::Heap,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            });
            return;
        }
        self.add_potential_op(self.lookup_label_arg(result), op, PreambleOpKind::Heap);
    }

    /// Add a loop-invariant call as a short-box candidate.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let result = op.pos;
        self.add_potential_op(
            self.lookup_label_arg(result),
            op,
            PreambleOpKind::LoopInvariant,
        );
    }

    pub(crate) fn add_short_input_arg(&mut self, arg: OpRef, arg_type: majit_ir::Type) {
        // shortpreamble.py:255-259 parity: ShortInputArg's BoxType is the
        // intrinsic `box.type` (BoxInt → same_as_i, BoxRef → same_as_r,
        // BoxFloat → same_as_f). The caller is responsible for filtering
        // out positions whose `value_types` is `Void` (e.g. constant
        // collisions); a `Void` reaching here panics in
        // `OpCode::same_as_for_type`'s `unreachable!` arm with a clear
        // RPython-parity violation message.
        let label_arg_idx = self.lookup_label_arg(arg);
        let mut same_as = Op::new(OpCode::same_as_for_type(arg_type), &[arg]);
        same_as.pos = arg;
        if !self.potential_order.contains(&arg) {
            self.potential_order.push(arg);
        }
        self.potential_ops.insert(
            arg,
            PotentialShortOp::Preamble(PreambleOp {
                op: same_as,
                kind: PreambleOpKind::InputArg,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            }),
        );
    }

    fn produce_arg(&mut self, opref: OpRef) -> Option<OpRef> {
        if let Some(existing) = self.produced_short_boxes.get(&opref) {
            return Some(existing.preamble_op.pos);
        }
        if self.boxes_in_production.contains(&opref) {
            return None;
        }
        if self.known_constants.contains(&opref) {
            return Some(opref);
        }
        if self.potential_ops.contains_key(&opref) {
            return self
                .materialize_one(opref)
                .map(|produced| produced.preamble_op.pos);
        }
        // Label args are always available as inputs (RPython: isinstance(op, InputArgIntOp))
        if self.label_arg_positions.contains_key(&opref) {
            return Some(opref);
        }
        None
    }

    /// shortpreamble.py:298-309 _pick_op_index
    ///
    /// Pick which compound-op alternative becomes the canonical short box.
    /// Heap ops are last-resort; the first non-Heap candidate wins on the
    /// initial `pick_other=True` pass. If two non-Heap candidates are seen,
    /// recurse with `pick_other=False`, which restricts the second pass to
    /// `ShortInputArg` candidates only (the highest-priority class).
    /// Falls back to index 0 when no candidate qualifies.
    fn _pick_op_index(lst: &[ProducedShortOp], pick_other: bool) -> usize {
        let mut index: Option<usize> = None;
        for (i, item) in lst.iter().enumerate() {
            let prefer = !matches!(item.kind, PreambleOpKind::Heap)
                && (pick_other || item.kind == PreambleOpKind::InputArg);
            if prefer {
                if index.is_some() {
                    debug_assert!(pick_other, "second-tier ambiguity");
                    return Self::_pick_op_index(lst, false);
                }
                index = Some(i);
            }
        }
        index.unwrap_or(0)
    }

    fn materialize_one(&mut self, result: OpRef) -> Option<ProducedShortOp> {
        if let Some(existing) = self.produced_short_boxes.get(&result) {
            return Some(existing.clone());
        }
        if self.boxes_in_production.contains(&result) {
            return None;
        }
        let candidate = self.potential_ops.get(&result)?.clone();
        self.boxes_in_production.insert(result);
        let produced = candidate.add_op_to_short(self)?;
        self.produced_short_boxes.insert(result, produced.clone());
        self.produced_order.push(result);
        self.boxes_in_production.remove(&result);
        Some(produced)
    }

    /// shortpreamble.py: produced_short_boxes after add_op_to_short().
    pub fn produced_ops(&mut self) -> Vec<(OpRef, ProducedShortOp)> {
        let keys = self.potential_order.clone();
        for key in keys {
            let _ = self.materialize_one(key);
        }
        self.produced_order
            .iter()
            .filter_map(|key| {
                self.produced_short_boxes
                    .get(key)
                    .cloned()
                    .map(|produced| (*key, produced))
            })
            .collect()
    }

    /// shortpreamble.py:246-281 ShortBoxes.create_short_boxes
    ///
    /// Materialize all `potential_ops` (already populated by
    /// `produce_potential_short_preamble_ops` calls on each pass) into
    /// `produced_short_boxes`, then walk `const_short_boxes`: for each
    /// constant heap read, try to produce its struct argument; if that
    /// succeeds, emit a `getfield` short op whose first arg is the
    /// produced preamble box. Constant heap reads do not allocate fresh
    /// names — they reuse the original constant box.
    ///
    /// `label_args` / `label_arg_types` populate `short_inputargs` via
    /// `add_short_input_arg` (RPython:
    /// `self.potential_ops[box] = ShortInputArg(box, renamed)`).
    pub fn create_short_boxes(
        &mut self,
        label_args: &[OpRef],
        label_arg_types: &[majit_ir::Type],
    ) -> Vec<ProducedShortOp> {
        // shortpreamble.py:255-259: register every label arg as a
        // ShortInputArg potential op.
        for (i, &arg) in label_args.iter().enumerate() {
            // shortpreamble.py:256 reads label_args[i].type intrinsically;
            // pyre's parallel array must match label_args length.
            let arg_type = label_arg_types.get(i).copied().unwrap_or_else(|| {
                panic!(
                    "missing label_arg_types[{}] (label_args.len()={}): \
                     create_short_boxes needs the parallel type list in \
                     lockstep with label_args",
                    i,
                    label_args.len()
                )
            });
            self.add_short_input_arg(arg, arg_type);
        }

        // shortpreamble.py:261: optimizer.produce_potential_short_preamble_ops(self)
        // — caller must have invoked this on each pass before calling
        // create_short_boxes (majit threads passes externally).

        // shortpreamble.py:263-267: short_boxes = []; for shortop in potential_ops.values(): add_op_to_short
        let mut short_boxes: Vec<ProducedShortOp> =
            self.produced_ops().into_iter().map(|(_, op)| op).collect();

        // shortpreamble.py:272-280: walk const_short_boxes and try to
        // produce a struct preamble arg, then emit the getfield op.
        let const_pending: Vec<PreambleOp> = std::mem::take(&mut self.const_short_boxes);
        for short_op in const_pending {
            let getfield_op = &short_op.op;
            if getfield_op.args.is_empty() {
                continue;
            }
            let struct_arg = getfield_op.arg(0);
            let Some(preamble_arg) = self.produce_arg(struct_arg) else {
                continue;
            };
            // shortpreamble.py:277-278: copy_and_change(opnum, [preamble_arg] + args[1:])
            let mut new_args = vec![preamble_arg];
            new_args.extend_from_slice(&getfield_op.args[1..]);
            let mut new_op = Op::with_descr(
                getfield_op.opcode,
                &new_args,
                getfield_op
                    .descr
                    .clone()
                    .unwrap_or_else(|| panic!("const_short_boxes heap op without descr")),
            );
            new_op.pos = getfield_op.pos;
            // shortpreamble.py:279: ProducedShortOp(short_op, preamble_op)
            short_boxes.push(ProducedShortOp {
                preamble_op: new_op,
                kind: PreambleOpKind::Heap,
                invented_name: false,
                same_as_source: None,
            });
        }
        short_boxes
    }

    /// shortpreamble.py: create_short_inputargs(label_args)
    /// Build the input args for the short preamble from label args.
    /// Returns OpRefs for each label arg that has a producer, or the
    /// original label arg if no producer exists.
    pub fn create_short_inputargs(&self, label_args: &[OpRef]) -> Vec<OpRef> {
        if self.short_inputargs.is_empty() {
            label_args.to_vec()
        } else {
            self.short_inputargs.clone()
        }
    }

    /// shortpreamble.py: add_potential_op(op, pop)
    /// Add a produced operation to the short boxes at the given position.
    pub fn add_potential_op(&mut self, label_arg_idx: Option<usize>, op: Op, kind: PreambleOpKind) {
        let result = op.pos;
        let pop = PotentialShortOp::Preamble(PreambleOp {
            op,
            kind,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
        let next = match self.potential_ops.remove(&result) {
            Some(prev) => PotentialShortOp::Compound(CompoundOp {
                res: result,
                one: Box::new(pop),
                two: Box::new(prev),
            }),
            None => pop,
        };
        self.add_op(result, next);
    }
}

/// shortpreamble.py: create_short_boxes(optimizer, inputargs, label_args)
///
/// Existing call sites pass an `optimizer_ops` slice that the new
/// method-form helper does not consume; the slice is preserved for
/// API stability and ignored. Prefer calling
/// `ShortBoxes::create_short_boxes` directly.
pub fn create_short_boxes(
    short_boxes: &mut ShortBoxes,
    label_args: &[OpRef],
    label_arg_types: &[majit_ir::Type],
    _optimizer_ops: &[Op],
) -> Vec<ProducedShortOp> {
    short_boxes.create_short_boxes(label_args, label_arg_types)
}

/// Collector-side extended builder for extracting categorized preamble ops from
/// a peeled trace.
///
/// This is intentionally separate from RPython's active
/// `ExtendedShortPreambleBuilder`, which operates while building the short
/// preamble for phase 2 / bridge entry.
pub struct CollectedExtendedShortPreambleBuilder {
    /// Guards from the preamble.
    guards: Vec<PreambleOp>,
    /// Heap reads from the preamble.
    heap_ops: Vec<PreambleOp>,
    /// Pure operations from the preamble.
    pure_ops: Vec<PreambleOp>,
    /// Loop-invariant calls from the preamble.
    loopinvariant_ops: Vec<PreambleOp>,
    /// Map from preamble OpRef to label arg index.
    preamble_to_label_arg: HashMap<OpRef, usize>,
}

impl CollectedExtendedShortPreambleBuilder {
    pub fn new() -> Self {
        CollectedExtendedShortPreambleBuilder {
            guards: Vec::new(),
            heap_ops: Vec::new(),
            pure_ops: Vec::new(),
            loopinvariant_ops: Vec::new(),
            preamble_to_label_arg: HashMap::new(),
        }
    }

    /// Set the label args mapping.
    pub fn set_label_args(&mut self, label_args: &[OpRef]) {
        self.preamble_to_label_arg.clear();
        for (i, opref) in label_args.iter().enumerate() {
            self.preamble_to_label_arg.insert(*opref, i);
        }
    }

    /// Add a guard operation.
    pub fn add_guard(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.guards.push(PreambleOp {
            op,
            kind: PreambleOpKind::Guard,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a pure operation.
    pub fn add_pure_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.pure_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Pure,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a heap read.
    pub fn add_heap_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.heap_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::Heap,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Add a loop-invariant call.
    pub fn add_loopinvariant_op(&mut self, op: Op) {
        let label_arg_idx = self.preamble_to_label_arg.get(&op.pos).copied();
        self.loopinvariant_ops.push(PreambleOp {
            op,
            kind: PreambleOpKind::LoopInvariant,
            label_arg_idx,
            invented_name: false,
            same_as_source: None,
        });
    }

    /// Total number of recorded preamble operations.
    pub fn num_ops(&self) -> usize {
        self.guards.len() + self.heap_ops.len() + self.pure_ops.len() + self.loopinvariant_ops.len()
    }

    /// Build into a ShortPreamble, emitting operations in order:
    /// guards first, then heap reads, then pure ops, then loop-invariant.
    pub fn build(self, exported_state: Option<VirtualState>) -> ShortPreamble {
        let all_ops: Vec<PreambleOp> = self
            .guards
            .into_iter()
            .chain(self.heap_ops)
            .chain(self.pure_ops)
            .chain(self.loopinvariant_ops)
            .collect();

        let entries = all_ops
            .into_iter()
            .map(|preamble_op| {
                let mut arg_mapping = Vec::new();
                for (arg_pos, arg_ref) in preamble_op.op.args.iter().enumerate() {
                    if let Some(&label_idx) = self.preamble_to_label_arg.get(arg_ref) {
                        arg_mapping.push((arg_pos, label_idx));
                    }
                }
                let mut fail_arg_mapping = Vec::new();
                if let Some(fail_args) = &preamble_op.op.fail_args {
                    for (fail_arg_pos, fail_arg_ref) in fail_args.iter().enumerate() {
                        if let Some(&label_idx) = self.preamble_to_label_arg.get(fail_arg_ref) {
                            fail_arg_mapping.push((fail_arg_pos, label_idx));
                        }
                    }
                }
                ShortPreambleOp {
                    op: preamble_op.op,
                    arg_mapping,
                    fail_arg_mapping,
                }
            })
            .collect();

        ShortPreamble {
            ops: entries,
            inputargs: Vec::new(),
            used_boxes: Vec::new(),
            jump_args: Vec::new(),
            exported_state,
            constants: HashMap::new(),
            phase1_inputargs: None,
            inputarg_infos: Vec::new(),
        }
    }
}

impl Default for CollectedExtendedShortPreambleBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// shortpreamble.py: CompoundOp — a short op that is composed of
/// two sub-operations (e.g., getfield followed by getarrayitem).
#[derive(Clone, Debug)]
pub struct CompoundOp {
    /// The result OpRef of the compound operation.
    pub res: OpRef,
    /// First sub-operation.
    one: Box<PotentialShortOp>,
    /// Second sub-operation (depends on the result of `one`).
    two: Box<PotentialShortOp>,
}

impl CompoundOp {
    /// shortpreamble.py: CompoundOp.flatten(sb, l)
    ///
    /// Recursively flatten a tree of CompoundOps into a list of
    /// ProducedShortOps in dependency order (children first).
    pub fn flatten(
        &self,
        sb: &mut ShortBoxes,
        mut produced: Vec<ProducedShortOp>,
    ) -> Vec<ProducedShortOp> {
        match self.one.as_ref() {
            PotentialShortOp::Compound(compound) => {
                produced = compound.flatten(sb, produced);
            }
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb) {
                    produced.push(pop);
                }
            }
        }
        match self.two.as_ref() {
            PotentialShortOp::Compound(compound) => compound.flatten(sb, produced),
            PotentialShortOp::Preamble(op) => {
                if let Some(pop) = op.add_op_to_short(sb) {
                    produced.push(pop);
                }
                produced
            }
        }
    }
}

/// shortpreamble.py: ShortInputArg — a short op that represents
/// a label input argument (no actual operation needed, just maps
/// a preamble value to a label arg position).
#[derive(Clone, Debug)]
pub struct ShortInputArg {
    /// The result OpRef.
    pub res: OpRef,
    /// The preamble operation that produces this value.
    pub preamble_op: Op,
}

impl ShortInputArg {
    /// shortpreamble.py: ShortInputArg.add_op_to_short(sb)
    ///
    /// Returns a ProducedShortOp wrapping the preamble_op.
    /// For input args, the preamble_op is just forwarded.
    pub fn add_op_to_short(&self) -> ProducedShortOp {
        ProducedShortOp {
            kind: PreambleOpKind::InputArg,
            preamble_op: self.preamble_op.clone(),
            invented_name: false,
            same_as_source: None,
        }
    }

    /// shortpreamble.py: ShortInputArg.produce_op(opt, ...)
    ///
    /// For input args, produce_op is a no-op — the value is
    /// already available as a label argument.
    pub fn produce_op(&self) {
        // No-op: the value is directly available from label args
    }
}

/// shortpreamble.py: ProducedShortOp — wraps a short op with its
/// preamble counterpart for emission during bridge compilation.
#[derive(Clone, Debug)]
pub struct ProducedShortOp {
    /// The short op classification.
    pub kind: PreambleOpKind,
    /// The preamble operation to replay.
    pub preamble_op: Op,
    /// Whether this short op uses an invented SameAs result.
    pub invented_name: bool,
    /// Original result this invented name aliases.
    pub same_as_source: Option<OpRef>,
}

/// Phase B B.1: helper used by `ProducedShortOp::produce_op` to seed a
/// fresh constant-pool slot in the importing trace for a Const arg seen
/// in the imported short op. Mirrors the inline `imported_const_opref`
/// closure inside the legacy `import_short_preamble_ops` (unroll.rs:3510).
fn imported_const_opref(
    ctx: &mut crate::optimizeopt::OptContext,
    imported_constants: &mut HashMap<OpRef, OpRef>,
    source: OpRef,
    value: &majit_ir::Value,
) -> OpRef {
    if let Some(&opref) = imported_constants.get(&source) {
        return opref;
    }
    let opref = ctx.reserve_const_ref(value.get_type());
    ctx.seed_constant(opref, value.clone());
    imported_constants.insert(source, opref);
    opref
}

/// shortpreamble.py:283 `ShortBoxes.produce_arg` — classify an arg in
/// `produced.preamble_op.args` as Slot/Const/Produced.  RPython has a
/// single classification path; majit shares this function between
/// `ProducedShortOp::produce_op` (shortpreamble.rs) and
/// `OptContext::initialize_imported_short_preamble_builder_from_short_boxes`
/// (mod.rs) so the two consume sites cannot drift.
///
/// - Slot: `arg ∈ short_inputargs` (positional) → Phase 2 OpRef from `short_args`
/// - Const: `arg ∈ short_box_const_values` (producer-snapshotted) or `arg`
///   has known consumer-side constant value → seed fresh consumer-side slot
/// - Produced: `arg ∈ produced_results` (a previously imported producer's source)
pub(crate) fn classify_short_arg(
    ctx: &mut crate::optimizeopt::OptContext,
    arg: OpRef,
    short_inputargs: &[OpRef],
    short_args: &[OpRef],
    produced_results: &HashMap<OpRef, OpRef>,
    imported_constants: &mut HashMap<OpRef, OpRef>,
    short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
) -> Option<crate::optimizeopt::ImportedShortPureArg> {
    if let Some(slot) = short_inputargs.iter().position(|&i| i == arg) {
        return short_args
            .get(slot)
            .copied()
            .map(crate::optimizeopt::ImportedShortPureArg::OpRef);
    }
    // Const lookup priority: producer snapshot first (handles bridges and
    // unit-test consumer ctxs without pre-seeded const pool), then consumer
    // ctx (production: pre-seeded at optimizer.rs:1927).
    if let Some(value) = short_box_const_values
        .get(&arg)
        .cloned()
        .or_else(|| ctx.get_constant(arg))
    {
        let const_opref = imported_const_opref(ctx, imported_constants, arg, &value);
        return Some(crate::optimizeopt::ImportedShortPureArg::Const(
            value,
            const_opref,
        ));
    }
    produced_results
        .get(&arg)
        .copied()
        .map(crate::optimizeopt::ImportedShortPureArg::OpRef)
}

impl ProducedShortOp {
    /// shortpreamble.py:212-214 ProducedShortOp.produce_op + per-kind dispatch:
    /// - PureOp.produce_op (shortpreamble.py:112-126)
    /// - HeapOp.produce_op (shortpreamble.py:62-85, getfield + getarrayitem)
    /// - LoopInvariantOp.produce_op (shortpreamble.py:152-159)
    /// - ShortInputArg.produce_op (shortpreamble.py:233-234, no-op)
    ///
    /// Mutates `ctx` to register the imported preamble op into the
    /// optimizer's per-kind side-table (`imported_short_pure_ops` for Pure,
    /// `set_preamble_field` / `set_preamble_item` on PtrInfo for Heap,
    /// `imported_loop_invariant_results` for LoopInvariant), mirroring the
    /// legacy `import_short_preamble_ops` per-arm body. Records the resolved
    /// result OpRef in `produced_results` keyed by `self.preamble_op.pos`
    /// so successor entries can reference it.
    ///
    /// Phase B B.1 (parallel implementation, dead code): the legacy
    /// enum-dispatch path in `import_short_preamble_ops` (unroll.rs:3504-3925)
    /// remains the active caller. B.2 wires this method as the sole produce-op
    /// driver. B.3+ retire the legacy enum.
    ///
    /// Returns `None` when:
    /// - args cannot be fully classified, mirroring legacy
    ///   `collect_exported_short_ops` skip;
    /// - the kind is non-emit (Heap with non-getfield/getarrayitem opcode).
    pub fn produce_op(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &HashMap<OpRef, OpRef>,
        produced_results: &mut HashMap<OpRef, OpRef>,
        imported_constants: &mut HashMap<OpRef, OpRef>,
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let result = match self.kind {
            PreambleOpKind::Pure => self.produce_pure(
                ctx,
                short_inputargs,
                short_args,
                result_map,
                produced_results,
                imported_constants,
                short_box_const_values,
            )?,
            PreambleOpKind::Heap => match self.preamble_op.opcode {
                OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => self
                    .produce_heap_field(
                        ctx,
                        exported_infos,
                        short_inputargs,
                        short_args,
                        result_map,
                        produced_results,
                        imported_constants,
                        short_box_const_values,
                    )?,
                OpCode::GetarrayitemGcI | OpCode::GetarrayitemGcR | OpCode::GetarrayitemGcF => self
                    .produce_heap_array_item(
                        ctx,
                        exported_infos,
                        short_inputargs,
                        short_args,
                        result_map,
                        produced_results,
                        imported_constants,
                        short_box_const_values,
                    )?,
                _ => return None,
            },
            PreambleOpKind::LoopInvariant => self.produce_loop_invariant(
                ctx,
                short_inputargs,
                short_args,
                result_map,
                produced_results,
                imported_constants,
                short_box_const_values,
            )?,
            // shortpreamble.py:233-234 ShortInputArg.produce_op asserts
            // `not invented_name` and otherwise does nothing; the source pos
            // is already in `short_inputargs`, no Phase 2 OpRef to record.
            PreambleOpKind::InputArg => {
                debug_assert!(
                    !self.invented_name,
                    "shortpreamble.py:234: ShortInputArg cannot have invented_name"
                );
                return None;
            }
            PreambleOpKind::Guard => return None,
        };
        produced_results.insert(self.preamble_op.pos, result);
        Some(result)
    }

    /// shortpreamble.py:112-126 PureOp.produce_op
    fn produce_pure(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &HashMap<OpRef, OpRef>,
        produced_results: &mut HashMap<OpRef, OpRef>,
        imported_constants: &mut HashMap<OpRef, OpRef>,
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos;
        // Result OpRef was fixed before ShortPreambleBuilder construction,
        // matching RPython's object identity being available before
        // `ProducedShortOp.produce_op` runs.
        let result_opref = *result_map.get(&source)?;
        let opcode = self.preamble_op.opcode;
        // shortpreamble.py:116-120 line-by-line:
        //
        //   if invented_name:
        //       op = self.orig_op.copy_and_change(self.orig_op.getopnum())
        //       op.set_forwarded(self.res)
        //   else:
        //       op = self.res
        //
        // PyPy's `self.res` for the invented arm is `new_name` — the alt
        // SAME_AS Box created by `add_op_to_short` at shortpreamble.py:327.
        // It is a fresh body-Box that gets emitted into the trace via
        // `add_preamble_op` and lives at its own (alt-specific) position.
        // The transient `op = orig_op.copy(...)` is the PreambleOp wrapper:
        // `op.set_forwarded(self.res)` routes body refs from the wrapper
        // straight to the alt's body slot.
        //
        // pyre's flat-OpRef analog allocates the wrapper at `source` (the
        // synthetic alias from the compound-dedup pass at
        // shortpreamble.rs:478-491) and the alt body slot at `result_opref`
        // (= `result_map[source]`, the body-visible OpRef this alt's
        // imported Pure / SAME_AS will live at). Forward `source ->
        // result_opref` so body refs after `force_op_from_preamble`
        // resolve to the alt's own body identity, NOT to the canonical
        // (`same_as_source`) — collapsing both alt and canonical onto the
        // canonical's body ref erases PyPy's invented-name Box identity
        // and corrupts the SAME_AS in `extra_same_as`.
        // Non-invented re-uses self.res directly without forwarding.
        if self.invented_name {
            ctx.replace_op(source, result_opref);
        }
        // RPython `Box.type` parity.
        ctx.register_value_type(result_opref, opcode.result_type());
        let args = self
            .preamble_op
            .args
            .iter()
            .map(|&arg| {
                classify_short_arg(
                    ctx,
                    arg,
                    short_inputargs,
                    short_args,
                    produced_results,
                    imported_constants,
                    short_box_const_values,
                )
            })
            .collect::<Option<Vec<_>>>()?;
        // shortpreamble.py:121-126: PureOp.produce_op routes through
        // `optpure.pure(...)` (or `extra_call_pure` for calls). majit's
        // single-table `imported_short_pure_ops` covers both because
        // `pure.rs` consults it for both arms during `optimize_pure_op` and
        // `optimize_call_pure_*`.
        ctx.imported_short_pure_ops
            .push(crate::optimizeopt::ImportedShortPureOp::new(
                opcode,
                self.preamble_op.descr.clone(),
                args,
                result_opref,
                source,
                self.invented_name,
            ));
        // shortpreamble.py:432-440 add_preamble_op + 437-438 extra_same_as:
        // RPython collects the SameAs op into `short_preamble_producer.extra_same_as`
        // lazily at use-box time (force_op_from_preamble path).  majit's
        // `used_imported_short_aliases()` derives the alias list directly
        // from `imported_short_preamble_builder.extra_same_as()` at the same
        // phase boundary, so an eager `imported_short_aliases.push` here
        // would be a NEW-DEVIATION dual write.
        //
        // Path B parity (B.6.7): Heap/Array/LoopInvariant produce_* return
        // `source` so successor short-op dependency args resolve through
        // `produced_results` to Phase 1 source-namespace (matching body
        // refs after `force_op_from_preamble_op` returns `preamble_source`).
        // produce_pure follows the same convention for consistency — the
        // body-visible `result_opref` is still registered (Box.type, the
        // ImportedShortPureOp `result` field) but is no longer the value
        // that successor entries see via produced_results.
        let _ = result_opref;
        Some(source)
    }

    /// shortpreamble.py:62-79 HeapOp.produce_op (getfield case)
    fn produce_heap_field(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &HashMap<OpRef, OpRef>,
        produced_results: &HashMap<OpRef, OpRef>,
        imported_constants: &mut HashMap<OpRef, OpRef>,
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos;
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.descr.clone()?;
        // Object arg classification — Slot or Const only (RPython
        // shortpreamble.py:91-95 add_op_to_short uses `produce_arg`,
        // which admits Slot/Const).  We accept Produced too for completeness.
        let object_arg = self.preamble_op.arg(0);
        let obj_class = classify_short_arg(
            ctx,
            object_arg,
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let obj = match obj_class {
            crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
            crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
        };
        // Cat-2.2 alignment: forward `source -> result_opref` so body refs
        // after `force_op_from_preamble_op` resolve to the body-visible
        // OpRef without relying on the use-before-def assembly adaptation.
        // PyPy `HeapOp.produce_op` stores `PreambleOp(op=self.res,
        // preamble_op, invented_name)` in field cache; self.res is the
        // canonical body Box (Box identity makes it body-visible). pyre's
        // flat-OpRef analog uses `result_opref = result_map[source]` as the
        // body-visible slot.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        let descr_idx = descr.index();
        let obj_resolved = ctx.get_box_replacement(obj);
        // shortpreamble.py:66-68: if g.getarg(0) in exported_infos:
        //     setinfo_from_preamble(g.getarg(0), exported_infos[...])
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(pinfo)) = exported_infos.get(&object_arg)
        {
            ctx.setinfo_from_preamble(obj_resolved, pinfo, Some(exported_infos));
        }
        let mut getfield_op = Op::new(OpCode::getfield_for_type(result_type), &[obj_resolved]);
        getfield_op.descr = Some(descr.clone());
        // Cat-2.2 dual-slot rule (mod.rs:1817 replay_pos): replay.pos =
        // result_opref because `replace_op(source, result_opref)` installed
        // below clobbers source's slot. Seed info at result_opref's slot
        // (set_preamble_forwarded_info via replay_pos) so
        // `take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos)`
        // reads it back. PyPy `preamble_op.set_forwarded(info)` lives on a
        // distinct Op object, so this slot juggling is the pyre adaptation
        // of that Box-identity invariant.
        getfield_op.pos = result_opref;
        let pop = crate::optimizeopt::info::PreambleOp {
            op: source,
            invented_name: self.invented_name,
            preamble_op: getfield_op.clone(),
        };
        let parent_descr = getfield_op
            .descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .and_then(|fd| fd.get_parent_descr());
        if let Some(info) = ctx.get_const_info_mut(obj_resolved, parent_descr) {
            info.set_preamble_field(descr_idx, pop.clone());
        }
        // shortpreamble.py:72-74: ensure_ptr_info_arg0 + setfield(pop)
        let pop_for_field = pop.clone();
        ctx.with_ensured_ptr_info_arg0(&getfield_op, |mut struct_info| {
            if let Some(info) = struct_info.as_mut() {
                debug_assert!(
                    !info.is_virtual(),
                    "shortpreamble.py:74: imported heap field on virtual"
                );
                info.set_preamble_field(descr_idx, pop_for_field);
            }
        });
        // Cat-2.2 alignment: forward `source -> result_opref` after the
        // PtrInfo / const-info side tables have been seeded (so the seeds
        // see the unforwarded source key consistent with `pop.op = source`).
        ctx.replace_op(source, result_opref);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py:80-85 HeapOp.produce_op (getarrayitem case)
    fn produce_heap_array_item(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        exported_infos: &HashMap<OpRef, crate::optimizeopt::info::OpInfo>,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &HashMap<OpRef, OpRef>,
        produced_results: &HashMap<OpRef, OpRef>,
        imported_constants: &mut HashMap<OpRef, OpRef>,
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos;
        let result_type = self.preamble_op.result_type();
        let descr = self.preamble_op.descr.clone()?;
        let object_arg = self.preamble_op.arg(0);
        let obj_class = classify_short_arg(
            ctx,
            object_arg,
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let obj = match obj_class {
            crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
            crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
        };
        // shortpreamble.py:81 `g.getarg(1).getint()`: read the integer
        // VALUE of the index Const, not the OpRef raw bits.
        // `OpRef::raw()` returns the trace-namespace tagged u32 — it is
        // NOT the constant integer.  Resolve via classify_short_arg
        // which checks the producer snapshot (`short_box_const_values`)
        // first, then the consumer ctx const pool.
        let index_arg = self.preamble_op.arg(1);
        let index = match classify_short_arg(
            ctx,
            index_arg,
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )? {
            crate::optimizeopt::ImportedShortPureArg::Const(majit_ir::Value::Int(v), _) => v,
            _ => return None,
        };
        // Cat-2.2 alignment: symmetric to produce_heap_field. Forward
        // `source -> result_opref` so body refs after
        // `force_op_from_preamble_op` resolve to the body-visible OpRef
        // without relying on the use-before-def assembly adaptation.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        let obj_resolved = ctx.get_box_replacement(obj);
        // shortpreamble.py:68-71 applies to both getfield and
        // getarrayitem: if the base object has exported info, import it
        // before ensuring heap/array PtrInfo.
        if let Some(crate::optimizeopt::info::OpInfo::Ptr(pinfo)) = exported_infos.get(&object_arg)
        {
            ctx.setinfo_from_preamble(obj_resolved, pinfo, Some(exported_infos));
        }
        let index_const = ctx.make_constant_int(index);
        let mut getarrayitem_op = Op::new(
            OpCode::getarrayitem_for_type(result_type),
            &[obj_resolved, index_const],
        );
        getarrayitem_op.descr = Some(descr.clone());
        // Cat-2.2 dual-slot rule (mod.rs:1817 replay_pos): replay.pos =
        // result_opref. See produce_heap_field for the Box-identity-vs-flat-OpRef
        // adaptation rationale.
        getarrayitem_op.pos = result_opref;
        let pop = crate::optimizeopt::info::PreambleOp {
            op: source,
            invented_name: self.invented_name,
            preamble_op: getarrayitem_op.clone(),
        };
        if obj_resolved.is_constant() || ctx.get_constant(obj_resolved).is_some() {
            if let Some(info) = ctx.get_const_info_array_mut(obj_resolved, descr.clone()) {
                info.set_preamble_item(index as usize, pop.clone());
            }
        } else {
            let pop_for_array = pop.clone();
            ctx.with_ensured_ptr_info_arg0(&getarrayitem_op, |mut array_info| {
                if let Some(info) = array_info.as_mut() {
                    if let crate::optimizeopt::info::PtrInfo::Array(array_info) = info {
                        let _ = array_info.lenbound.make_gt_const(index);
                        let idx = index as usize;
                        if idx >= array_info.items.len() {
                            array_info.items.resize(
                                idx + 1,
                                crate::optimizeopt::info::FieldEntry::Value(OpRef::NONE),
                            );
                        }
                        array_info.items[idx] =
                            crate::optimizeopt::info::FieldEntry::Preamble(pop_for_array);
                    }
                }
            });
        }
        // Cat-2.2 alignment: forward `source -> result_opref` after the
        // const-info / ArrayPtrInfo side tables have been seeded.
        ctx.replace_op(source, result_opref);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }

    /// shortpreamble.py:152-159 LoopInvariantOp.produce_op
    fn produce_loop_invariant(
        &self,
        ctx: &mut crate::optimizeopt::OptContext,
        short_inputargs: &[OpRef],
        short_args: &[OpRef],
        result_map: &HashMap<OpRef, OpRef>,
        produced_results: &HashMap<OpRef, OpRef>,
        imported_constants: &mut HashMap<OpRef, OpRef>,
        short_box_const_values: &HashMap<OpRef, majit_ir::Value>,
    ) -> Option<OpRef> {
        let source = self.preamble_op.pos;
        let result_type = self.preamble_op.result_type();
        // shortpreamble.py:156-158 reads `self.res.getarg(0).getint()`
        // from the original Const box. In majit the const may only be
        // available through the producer-side snapshot, so classify it
        // through the same path as Pure/Heap args.
        let func_arg = classify_short_arg(
            ctx,
            self.preamble_op.arg(0),
            short_inputargs,
            short_args,
            produced_results,
            imported_constants,
            short_box_const_values,
        )?;
        let func_ptr = match func_arg {
            crate::optimizeopt::ImportedShortPureArg::Const(majit_ir::Value::Int(v), _) => v,
            _ => return None,
        };
        // Cat-2.2 alignment probe: forward `source -> result_opref` so body
        // refs after `force_op_from_preamble_op` resolve to the body-visible
        // OpRef without relying on the use-before-def assembly adaptation.
        // PyPy `LoopInvariantOp.produce_op` stores `PreambleOp(op=self.res,
        // preamble_op, invented_name)` in `loop_invariant_results`; self.res
        // is the canonical body Box. pyre's flat-OpRef analog uses
        // `result_opref = result_map[source]` as the body-visible slot and
        // installs the `source -> result_opref` forwarding here so the
        // consumer at `rewrite.rs:2809` resolves source to result_opref via
        // get_box_replacement uniformly.
        let result_opref = *result_map.get(&source)?;
        let _ = result_type;
        ctx.replace_op(source, result_opref);
        ctx.imported_loop_invariant_results.insert(func_ptr, source);
        // see produce_pure: extra_same_as collected lazily by
        // imported_short_preamble_builder; eager push would be a dual-write.
        Some(source)
    }
}

#[derive(Clone, Debug, Default)]
struct AbstractShortPreambleBuilderState {
    short: Vec<Op>,
    short_results: HashSet<OpRef>,
    used_boxes: Vec<OpRef>,
    short_preamble_jump: Vec<Op>,
    extra_same_as: Vec<Op>,
    short_inputargs: Vec<OpRef>,
    /// Known constant OpRefs. In RPython, isinstance(box, Const) is a type
    /// check. In majit, constant OpRefs must be explicitly tracked.
    known_constants: HashSet<OpRef>,
    /// B.6.4 canonical dedup for `record_imported_preamble_use`.
    /// `produced_short_boxes` is a dual-key map (source key + result_opref
    /// key both pointing at the same `ProducedShortOp`), so the source vs.
    /// body-visible distinction is not enough — RPython's Box identity
    /// makes one Box equal one slot regardless of how it is reached. The
    /// canonical key is `replay_op.pos` (a stable proxy for `self.res`):
    /// dedup here prevents two different lookup keys from pushing the
    /// same RPython Box twice into `used_boxes` /
    /// `short_preamble_jump` / `extra_same_as`.
    recorded_canonical_results: HashSet<OpRef>,
}

impl AbstractShortPreambleBuilderState {
    /// shortpreamble.py:432-440: add_preamble_op
    ///
    /// RPython (4 lines):
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    ///
    /// `same_as_source`: the original OpRef this invented name aliases.
    fn record_imported_preamble_use(
        &mut self,
        op: OpRef,
        replay_op: &Op,
        invented_name: bool,
        same_as_source: Option<OpRef>,
    ) {
        if !self.recorded_canonical_results.insert(replay_op.pos) {
            return;
        }
        if invented_name {
            let source = same_as_source.unwrap_or(op);
            let mut same_as = Op::new(OpCode::same_as_for_type(replay_op.result_type()), &[source]);
            same_as.pos = op;
            self.extra_same_as.push(same_as);
        }
        self.used_boxes.push(op);
        self.short_preamble_jump.push(replay_op.clone());
    }

    fn record_preamble_use(&mut self, result: OpRef, produced: &ProducedShortOp) {
        self.record_imported_preamble_use(
            result,
            &produced.preamble_op,
            produced.invented_name,
            produced.same_as_source,
        );
    }

    /// Internal: append preamble_op to short (with ovf guard).
    /// Used by add_op_to_short (recursive export-time path).
    fn append_to_short(&mut self, _result: OpRef, produced: &ProducedShortOp) -> Op {
        let canonical_result = produced.preamble_op.pos;
        if self.short_results.contains(&canonical_result) {
            return produced.preamble_op.clone();
        }
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        preamble_op
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive: iterates preamble_op's args (adding non-input deps
    /// + guards to short), then appends preamble_op + result guards.
    /// Called by force_op_from_preamble (unroll.py:32).
    ///
    /// RPython passes `preamble_op` directly (there is no lookup miss).
    /// `all_produced` + `pos_to_key` are still needed to resolve dependency
    /// args whose Phase-2 OpRefs differ from `produced_short_boxes` keys.
    fn use_box(
        &mut self,
        preamble_op: &Op,
        already_in_short: &HashSet<OpRef>,
        all_produced: &HashMap<OpRef, ProducedShortOp>,
        pos_to_key: &HashMap<OpRef, OpRef>,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) -> Op {
        let canonical_result = preamble_op.pos;
        if self.short_results.contains(&canonical_result)
            || already_in_short.contains(&canonical_result)
        {
            return preamble_op.clone();
        }
        // shortpreamble.py:383-396: iterate preamble_op args
        for &arg in &preamble_op.args {
            if self.short_results.contains(&arg)
                || already_in_short.contains(&arg)
                || self.short_inputargs.contains(&arg)
                || self.known_constants.contains(&arg)
            {
                continue;
            }
            // shortpreamble.py:393: self.short.append(arg)
            // Look up dep by key first, then by preamble_op.pos reverse index.
            // In RPython, Box identity makes this lookup trivial. In majit,
            // produce_arg returns preamble_op.pos which may differ from the
            // produced_short_boxes key.
            let dep = all_produced
                .get(&arg)
                .or_else(|| pos_to_key.get(&arg).and_then(|key| all_produced.get(key)));
            if let Some(dep) = dep {
                let dep_canonical = dep.preamble_op.pos;
                if !self.short_results.contains(&dep_canonical)
                    && !already_in_short.contains(&dep_canonical)
                {
                    self.short_results.insert(dep_canonical);
                    self.short.push(dep.preamble_op.clone());
                    if dep.preamble_op.opcode.is_ovf() {
                        self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
                    }
                }
            }
        }
        // shortpreamble.py:389,396: info.make_guards(arg, self.short, optimizer)
        self.short.extend_from_slice(arg_guards);
        // shortpreamble.py:398: self.short.append(preamble_op)
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        // shortpreamble.py:405-406: info.make_guards(preamble_op, self.short, optimizer)
        self.short.extend_from_slice(result_guards);
        preamble_op.clone()
    }
}

fn build_short_preamble_struct_from_ops(
    short_inputargs: &[OpRef],
    ops: &[Op],
    used_boxes: &[OpRef],
    jump_args: &[OpRef],
    loop_constants: &HashMap<u32, i64>,
    loop_constant_types: &HashMap<u32, majit_ir::Type>,
) -> ShortPreamble {
    let short_inputarg_positions: HashMap<OpRef, usize> = short_inputargs
        .iter()
        .enumerate()
        .map(|(idx, &arg)| (arg, idx))
        .collect();
    // Collect all OpRefs defined by the short preamble ops (as results).
    let mut defined_by_ops: HashSet<OpRef> = HashSet::new();
    for ia in short_inputargs {
        defined_by_ops.insert(*ia);
    }
    for op in ops {
        if !op.pos.is_none() {
            defined_by_ops.insert(op.pos);
        }
    }
    let entries = ops
        .iter()
        .cloned()
        .map(|op| {
            let arg_mapping = op
                .args
                .iter()
                .enumerate()
                .filter_map(|(arg_pos, arg_ref)| {
                    short_inputarg_positions
                        .get(arg_ref)
                        .copied()
                        .map(|label_idx| (arg_pos, label_idx))
                })
                .collect();
            let fail_arg_mapping = op
                .fail_args
                .as_ref()
                .map(|fail_args| {
                    fail_args
                        .iter()
                        .enumerate()
                        .filter_map(|(fail_arg_pos, fail_arg_ref)| {
                            short_inputarg_positions
                                .get(fail_arg_ref)
                                .copied()
                                .map(|label_idx| (fail_arg_pos, label_idx))
                        })
                        .collect()
                })
                .unwrap_or_default();
            ShortPreambleOp {
                op,
                arg_mapping,
                fail_arg_mapping,
            }
        })
        .collect();
    // RPython parity: capture constant values referenced by short preamble ops.
    // In RPython, Const objects are stored in op args and are GC-tracked. Here,
    // we snapshot the loop's constant pool entries for any OpRef referenced by
    // short preamble ops that isn't defined by the ops themselves.
    let mut constants: HashMap<u32, (i64, majit_ir::Type)> = HashMap::new();
    // RPython parity: if a Const exists in loop_constants, its type must
    // also exist in loop_constant_types — both maps are populated in
    // lockstep by the optimizer (constant_pool.rs). Missing type entry
    // for a known constant is a structural bug.
    let const_type_for = |raw: u32| -> majit_ir::Type {
        loop_constant_types.get(&raw).copied().unwrap_or_else(|| {
            panic!(
                "loop_constant_types missing entry for raw={} though loop_constants has it: \
                 the two maps must be populated in lockstep",
                raw
            )
        })
    };
    for op in ops {
        for &arg in &op.args {
            if !defined_by_ops.contains(&arg) {
                if let Some(&val) = loop_constants.get(&arg.raw()) {
                    constants.insert(arg.raw(), (val, const_type_for(arg.raw())));
                }
            }
        }
        if let Some(ref fa) = op.fail_args {
            for &arg in fa {
                if !defined_by_ops.contains(&arg) {
                    if let Some(&val) = loop_constants.get(&arg.raw()) {
                        constants.insert(arg.raw(), (val, const_type_for(arg.raw())));
                    }
                }
            }
        }
    }
    // Also check jump_args for constants
    for &arg in jump_args {
        if !defined_by_ops.contains(&arg) {
            if let Some(&val) = loop_constants.get(&arg.raw()) {
                constants.insert(arg.raw(), (val, const_type_for(arg.raw())));
            }
        }
    }
    ShortPreamble {
        ops: entries,
        inputargs: short_inputargs.to_vec(),
        used_boxes: used_boxes.to_vec(),
        jump_args: jump_args.to_vec(),
        exported_state: None,
        constants,
        phase1_inputargs: None,
        inputarg_infos: Vec::new(),
    }
}

/// shortpreamble.py: ShortPreambleBuilder
///
/// Builds the replayable short preamble from exported short boxes, while also
/// collecting `used_boxes`, `short_preamble_jump`, and `extra_same_as`.
///
/// Build reverse index: preamble_op.pos → key for entries where they differ.
/// In RPython, Box identity makes this unnecessary. In majit, produce_arg
/// returns preamble_op.pos which may differ from the key in produced_short_boxes.
fn build_pos_to_key(produced: &HashMap<OpRef, ProducedShortOp>) -> HashMap<OpRef, OpRef> {
    produced
        .iter()
        .filter(|(key, prod)| **key != prod.preamble_op.pos)
        .map(|(key, prod)| (prod.preamble_op.pos, *key))
        .collect()
}

#[derive(Clone, Debug)]
pub struct ShortPreambleBuilder {
    state: AbstractShortPreambleBuilderState,
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
}

impl ShortPreambleBuilder {
    pub fn new(
        label_args: &[OpRef],
        short_boxes: &[(OpRef, ProducedShortOp)],
        short_inputargs: &[OpRef],
    ) -> Self {
        let produced_short_boxes = short_boxes.iter().cloned().collect();
        ShortPreambleBuilder {
            state: AbstractShortPreambleBuilderState {
                short_inputargs: if short_inputargs.is_empty() {
                    label_args.to_vec()
                } else {
                    short_inputargs.to_vec()
                },
                ..AbstractShortPreambleBuilderState::default()
            },
            produced_short_boxes,
        }
    }

    pub fn note_known_constant(&mut self, opref: OpRef) {
        self.state.known_constants.insert(opref);
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut HashSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos;
        if self.state.short_results.contains(&canonical_result) {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result) {
            return None;
        }
        for &arg in &produced.preamble_op.args {
            // RPython: isinstance(arg, Const) → skip
            if self.state.known_constants.contains(&arg) {
                continue;
            }
            if self.produced_short_boxes.contains_key(&arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.remove(&result);
        Some(self.state.append_to_short(result, &produced))
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, used during
    /// export-time create_short_boxes to resolve transitive dependencies.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut HashSet::new())
    }

    /// shortpreamble.py:382-407: use_box(box, preamble_op, optimizer)
    /// Non-recursive. Called by force_op_from_preamble (unroll.py:32).
    ///
    /// RPython passes `preamble_op.preamble_op` directly — no lookup miss
    /// possible. majit prefers the produced_short_boxes lookup (which may
    /// carry a Phase-2 remapped pos), and falls back to `fallback_op` from
    /// info::PreambleOp when absent.
    pub fn use_box(
        &mut self,
        source: OpRef,
        fallback_op: &Op,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) {
        let preamble_op = match self.produced_short_boxes.get(&source) {
            Some(produced) => &produced.preamble_op,
            None => {
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit][use_box] produced_short_boxes miss for {source:?}, using fallback"
                    );
                }
                fallback_op
            }
        };
        let pos_to_key = build_pos_to_key(&self.produced_short_boxes);
        self.state.use_box(
            preamble_op,
            &HashSet::new(),
            &self.produced_short_boxes,
            &pos_to_key,
            arg_guards,
            result_guards,
        );
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    /// shortpreamble.py:432-440: add_preamble_op(preamble_op)
    /// Called from optimizer.force_box when popping from potential_extra_ops.
    ///
    /// RPython unconditionally appends to used_boxes and short_preamble_jump
    /// without any produced_short_boxes lookup:
    ///   op = preamble_op.op.get_box_replacement()
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    /// shortpreamble.py:432-440: add_preamble_op(preamble_op)
    /// RPython unconditionally appends:
    ///   op = preamble_op.op.get_box_replacement()
    ///   self.used_boxes.append(op)
    ///   self.short_preamble_jump.append(preamble_op.preamble_op)
    /// shortpreamble.py:432-440: add_preamble_op(preamble_op)
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: OpRef,
    ) {
        if let Some(produced) = self.produced_short_boxes.get(&preamble_op.op) {
            self.state.record_preamble_use(resolved_op, produced);
        } else {
            // shortpreamble.py:432-440: same 4-line pattern via common helper.
            let replay_op = &preamble_op.preamble_op;
            self.state.record_imported_preamble_use(
                resolved_op,
                replay_op,
                preamble_op.invented_name,
                Some(preamble_op.op),
            );
        }
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        self.state.record_preamble_use(result, &produced);
        true
    }

    pub fn build_short_preamble(&self) -> Vec<Op> {
        let mut result = Vec::with_capacity(self.state.short.len() + 2);
        result.push(Op::new(OpCode::Label, &self.state.short_inputargs));
        result.extend(self.state.short.iter().cloned());
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos)
            .collect();
        result.push(Op::new(OpCode::Jump, &jump_args));
        result
    }

    pub fn build_short_preamble_struct(
        &self,
        loop_constants: &HashMap<u32, i64>,
        loop_constant_types: &HashMap<u32, majit_ir::Type>,
    ) -> ShortPreamble {
        let jump_args: Vec<OpRef> = self
            .state
            .short_preamble_jump
            .iter()
            .map(|op| op.pos)
            .collect();
        build_short_preamble_struct_from_ops(
            &self.state.short_inputargs,
            &self.state.short,
            &self.state.used_boxes,
            &jump_args,
            loop_constants,
            loop_constant_types,
        )
    }

    pub fn used_boxes(&self) -> &[OpRef] {
        &self.state.used_boxes
    }

    pub fn short_preamble_jump(&self) -> &[Op] {
        &self.state.short_preamble_jump
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.state.extra_same_as
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.state.short_inputargs
    }
}

/// shortpreamble.py:448-482: ExtendedShortPreambleBuilder
///
/// RPython parity: single `short` list with JUMP sentinel at end.
/// `use_box()` pops JUMP, appends deps/guards/op, re-appends JUMP.
#[derive(Clone, Debug)]
pub struct ExtendedShortPreambleBuilder {
    produced_short_boxes: HashMap<OpRef, ProducedShortOp>,
    short_inputargs: Vec<OpRef>,
    /// shortpreamble.py:460: self.short = short — single ops list (base + JUMP sentinel)
    short: Vec<Op>,
    /// Tracks which OpRefs are already in `short` (for dedup).
    short_results: HashSet<OpRef>,
    /// Constants tracked for RPython isinstance(arg, Const) checks.
    known_constants: HashSet<OpRef>,
    extra_same_as: Vec<Op>,
    short_preamble_jump: Vec<Op>,
    base_extra_same_as: Vec<Op>,
    label_args: Vec<OpRef>,
    used_boxes: Vec<OpRef>,
    short_jump_args: Vec<OpRef>,
    pub target_token: u64,
    /// RPython parity: remap Phase 1 preamble OpRefs → current inputarg OpRefs.
    phase1_to_inputarg: HashMap<OpRef, OpRef>,
    /// B.6.4 canonical dedup keyed by `produced.preamble_op.pos`. Mirrors
    /// `AbstractShortPreambleBuilderState.recorded_canonical_results` —
    /// `produced_short_boxes` carries dual entries (source-key plus
    /// result_opref-key) for the same RPython Box, so per-key dedup
    /// (`label_args` etc.) cannot catch a second add via the alternate
    /// key. RPython's Box identity collapses both paths to one entry.
    recorded_canonical_results: HashSet<OpRef>,
}

impl ExtendedShortPreambleBuilder {
    pub fn new(target_token: u64, sb: &ShortPreambleBuilder) -> Self {
        ExtendedShortPreambleBuilder {
            produced_short_boxes: sb.produced_short_boxes.clone(),
            short_inputargs: sb.short_inputargs().to_vec(),
            short: Vec::new(),
            short_results: HashSet::new(),
            known_constants: HashSet::new(),
            extra_same_as: sb.extra_same_as().to_vec(),
            short_preamble_jump: Vec::new(),
            base_extra_same_as: sb.extra_same_as().to_vec(),
            label_args: Vec::new(),
            used_boxes: Vec::new(),
            short_jump_args: Vec::new(),
            target_token,
            phase1_to_inputarg: HashMap::new(),
            recorded_canonical_results: HashSet::new(),
        }
    }

    /// shortpreamble.py:458-461: setup(jump_args, short, label_args)
    ///
    /// RPython parity: builds single `short` list from base ops + JUMP sentinel.
    /// For each base op, ensures missing deps from produced_short_boxes are
    /// inserted before the consumer (RPython use_box arg-handling parity).
    ///
    /// Returns `true` on success; `false` if any op references an
    /// unresolvable Phase 1 OpRef. RPython equivalent: `produce_arg`
    /// returning None propagates up to `add_op_to_short` returning None,
    /// and the entire short_op is dropped (shortpreamble.py:283-296,
    /// 311-341). Pyre's structural counterpart is to bail out of `setup`
    /// so the caller (`jump_to_existing_trace`) can fall back to
    /// `jump_to_preamble` instead of attempting to inline a broken short
    /// preamble.
    pub fn setup(&mut self, short_preamble: &ShortPreamble, label_args: &[OpRef]) -> bool {
        // Build Phase 1 → current inputarg remap from arg_mapping.
        self.phase1_to_inputarg.clear();
        for entry in &short_preamble.ops {
            for &(arg_pos, label_idx) in &entry.arg_mapping {
                if let Some(&phase1_ref) = entry.op.args.get(arg_pos) {
                    if let Some(&current_inputarg) = label_args.get(label_idx) {
                        if phase1_ref != current_inputarg {
                            self.phase1_to_inputarg.insert(phase1_ref, current_inputarg);
                        }
                    }
                }
            }
        }
        // RPython parity: DO NOT mutate produced_short_boxes in-place.
        // RPython's setup() only sets self.short/jump_args/label_args;
        // the preamble producer (self) may be reused across multiple
        // setup() calls with different label_args. In-place remap would
        // corrupt the original Phase 1 args for subsequent calls.
        // Instead, remap on-the-fly when reading from produced_short_boxes.

        // Build single short list with inline dep resolution.
        let inputargs_set: HashSet<OpRef> = label_args.iter().copied().collect();
        let constants_set: HashSet<u32> = short_preamble.constants.keys().copied().collect();
        let pos_to_key = build_pos_to_key(&self.produced_short_boxes);
        self.short.clear();
        self.short_results.clear();
        for entry in &short_preamble.ops {
            let mut op = entry.op.clone();
            for arg in &mut op.args {
                if let Some(&remapped) = self.phase1_to_inputarg.get(arg) {
                    *arg = remapped;
                }
            }
            // RPython use_box arg loop: insert missing deps before this op.
            // Recursive: deps of deps are also inserted (transitive closure).
            for &arg in &op.args {
                if !self.insert_dep_recursive(arg, &inputargs_set, &constants_set, &pos_to_key) {
                    if crate::optimizeopt::majit_log_enabled() {
                        eprintln!(
                            "[jit] short_preamble setup: dropping inline (unresolved arg {:?} in op pos={:?} opcode={:?})",
                            arg, op.pos, op.opcode
                        );
                    }
                    self.short.clear();
                    self.short_results.clear();
                    self.label_args = label_args.to_vec();
                    return false;
                }
            }
            self.short_results.insert(op.pos);
            self.short.push(op);
        }
        // JUMP sentinel at end (RPython: short[-1] is always JUMP)
        let jump_args: Vec<OpRef> = short_preamble
            .jump_args
            .iter()
            .map(|arg| self.phase1_to_inputarg.get(arg).copied().unwrap_or(*arg))
            .collect();
        self.short.push(Op::new(OpCode::Jump, &jump_args));
        // Reset state
        self.extra_same_as = self.base_extra_same_as.clone();
        self.short_preamble_jump.clear();
        self.label_args = label_args.to_vec();
        self.used_boxes = short_preamble.used_boxes.clone();
        self.short_jump_args = jump_args;
        true
    }

    /// Recursively insert a dep (and its transitive deps) into self.short.
    /// Used by setup() to ensure all args of base ops are satisfied.
    /// Applies phase1_to_inputarg remap when reading from produced_short_boxes.
    ///
    /// Returns `true` if `arg` was resolved (or was already known); `false`
    /// if it was an unresolvable Phase 1 reference. Mirrors RPython
    /// `produce_arg → None` semantics (shortpreamble.py:283-296): a missing
    /// dep means the entire short preamble cannot be inlined cleanly, so
    /// the caller should bail out instead of leaving dangling args that
    /// later trip `inline_short_preamble`'s `_map_args` (unroll.py:404).
    fn insert_dep_recursive(
        &mut self,
        arg: OpRef,
        inputargs_set: &HashSet<OpRef>,
        constants_set: &HashSet<u32>,
        pos_to_key: &HashMap<OpRef, OpRef>,
    ) -> bool {
        if self.short_results.contains(&arg)
            || inputargs_set.contains(&arg)
            || self.known_constants.contains(&arg)
            || constants_set.contains(&arg.raw())
            || arg.is_constant()
            || arg.is_none()
        {
            return true;
        }
        // shortpreamble.py:284-285 — `op in self.produced_short_boxes`.
        // RPython uses Box identity; pyre needs both direct lookup and the
        // reverse `preamble_op.pos → key` lookup because produce_arg may
        // return a `.pos` distinct from the original key.
        let dep = self.produced_short_boxes.get(&arg).cloned().or_else(|| {
            pos_to_key
                .get(&arg)
                .and_then(|key| self.produced_short_boxes.get(key).cloned())
        });
        let Some(dep) = dep else {
            return false;
        };
        let dep_pos = dep.preamble_op.pos;
        if self.short_results.contains(&dep_pos) {
            return true;
        }
        // Remap dep args on-the-fly (don't mutate produced_short_boxes)
        let mut dep_op = dep.preamble_op.clone();
        for a in &mut dep_op.args {
            if let Some(&remapped) = self.phase1_to_inputarg.get(a) {
                *a = remapped;
            }
        }
        // Recurse into dep's own args first (transitive). If any sub-dep
        // can't be resolved, bail out — the dep cannot be safely emitted.
        for &dep_arg in &dep_op.args {
            if !self.insert_dep_recursive(dep_arg, inputargs_set, constants_set, pos_to_key) {
                return false;
            }
        }
        self.short_results.insert(dep_pos);
        self.short.push(dep_op);
        if dep.preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        true
    }

    fn use_box_recursive(&mut self, result: OpRef, visiting: &mut HashSet<OpRef>) -> Option<Op> {
        let produced = self.produced_short_boxes.get(&result)?.clone();
        let canonical_result = produced.preamble_op.pos;
        if self.short_results.contains(&canonical_result) {
            return Some(produced.preamble_op);
        }
        if !visiting.insert(result) {
            return None;
        }
        for &arg in &produced.preamble_op.args {
            if self.known_constants.contains(&arg) {
                continue;
            }
            if self.produced_short_boxes.contains_key(&arg) {
                let _ = self.use_box_recursive(arg, visiting);
            }
        }
        visiting.remove(&result);
        // Append to self.short directly
        let preamble_op = produced.preamble_op.clone();
        self.short_results.insert(canonical_result);
        self.short.push(preamble_op.clone());
        if preamble_op.opcode.is_ovf() {
            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
        }
        Some(preamble_op)
    }

    /// shortpreamble.py:465-476: add_preamble_op(preamble_op)
    ///
    /// RPython unconditionally appends to label_args and jump_args:
    ///   op = preamble_op.op.get_box_replacement()
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    /// shortpreamble.py:465-476: add_preamble_op(preamble_op)
    ///
    /// Extended version: label_args/jump_args instead of used_boxes.
    ///   op = preamble_op.op.get_box_replacement()
    ///   if preamble_op.invented_name:
    ///       self.extra_same_as.append(op)
    ///   self.label_args.append(op)
    ///   self.jump_args.append(preamble_op.preamble_op)
    pub fn add_preamble_op_from_pop(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
        resolved_op: OpRef,
    ) {
        let lookup_key = if self.produced_short_boxes.contains_key(&resolved_op) {
            resolved_op
        } else {
            preamble_op.op
        };
        if let Some(produced) = self.produced_short_boxes.get(&lookup_key).cloned() {
            self.add_tracked_preamble_op(resolved_op, &produced);
        } else {
            // shortpreamble.py:465-476: same pattern via replay_op.
            let replay_op = &preamble_op.preamble_op;
            if !self.recorded_canonical_results.insert(replay_op.pos) {
                return;
            }
            let op = resolved_op;
            if preamble_op.invented_name {
                let source = preamble_op.op;
                let mut same_as =
                    Op::new(OpCode::same_as_for_type(replay_op.result_type()), &[source]);
                same_as.pos = op;
                self.extra_same_as.push(same_as);
            }
            self.label_args.push(op);
            self.short_jump_args.push(replay_op.pos);
            self.short_preamble_jump.push(replay_op.clone());
        }
    }

    /// shortpreamble.py:471-477: add_preamble_op (internal)
    pub fn add_tracked_preamble_op(&mut self, result: OpRef, produced: &ProducedShortOp) {
        let current_result = produced.preamble_op.pos;
        if !self.recorded_canonical_results.insert(current_result) {
            return;
        }
        if produced.invented_name {
            let source = produced.same_as_source.unwrap_or(result);
            let mut op = Op::new(
                OpCode::same_as_for_type(produced.preamble_op.result_type()),
                &[source],
            );
            op.pos = current_result;
            self.extra_same_as.push(op);
        }
        self.label_args.push(result);
        self.used_boxes.push(current_result);
        self.short_jump_args.push(produced.preamble_op.pos);
        self.short_preamble_jump.push(produced.preamble_op.clone());
    }

    pub fn add_preamble_op(&mut self, result: OpRef) -> bool {
        let Some(produced) = self.produced_short_boxes.get(&result).cloned() else {
            return false;
        };
        self.add_tracked_preamble_op(result, &produced);
        true
    }

    /// shortpreamble.py:310: add_op_to_short — recursive, export-time.
    pub fn add_op_to_short(&mut self, result: OpRef) -> Option<Op> {
        self.use_box_recursive(result, &mut HashSet::new())
    }

    /// Remap a preamble op's args using phase1_to_inputarg (on-the-fly, no mutation).
    fn remap_op(&self, op: &Op) -> Op {
        if self.phase1_to_inputarg.is_empty() {
            return op.clone();
        }
        let mut remapped = op.clone();
        for arg in &mut remapped.args {
            if let Some(&r) = self.phase1_to_inputarg.get(arg) {
                *arg = r;
            }
        }
        remapped
    }

    /// shortpreamble.py:478-481: use_box — pop JUMP, add deps, re-append JUMP.
    /// Called by force_op_from_preamble (unroll.py:32).
    ///
    /// RPython passes `preamble_op.preamble_op` directly. majit uses the
    /// produced_short_boxes lookup for the Phase-2 remapped op, with
    /// `fallback_op` from info::PreambleOp as the safety net.
    pub fn use_box(
        &mut self,
        source: OpRef,
        fallback_op: &Op,
        arg_guards: &[Op],
        result_guards: &[Op],
    ) {
        let raw_op = match self.produced_short_boxes.get(&source) {
            Some(produced) => produced.preamble_op.clone(),
            None => {
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit][use_box ext] produced_short_boxes miss for {source:?}, using fallback"
                    );
                }
                fallback_op.clone()
            }
        };
        let preamble_op = self.remap_op(&raw_op);
        let canonical = preamble_op.pos;
        // shortpreamble.py:479: jump_op = self.short.pop()
        let jump_op = self.short.pop();
        // shortpreamble.py:480: AbstractShortPreambleBuilder.use_box(...)
        if !self.short_results.contains(&canonical) {
            let pos_to_key = build_pos_to_key(&self.produced_short_boxes);
            // Add deps for each arg
            for &arg in &preamble_op.args {
                if self.short_results.contains(&arg)
                    || self.short_inputargs.contains(&arg)
                    || self.known_constants.contains(&arg)
                {
                    continue;
                }
                let dep = self.produced_short_boxes.get(&arg).or_else(|| {
                    pos_to_key
                        .get(&arg)
                        .and_then(|key| self.produced_short_boxes.get(key))
                });
                if let Some(dep) = dep {
                    let dep_pos = dep.preamble_op.pos;
                    if !self.short_results.contains(&dep_pos) {
                        self.short_results.insert(dep_pos);
                        self.short.push(self.remap_op(&dep.preamble_op));
                        if dep.preamble_op.opcode.is_ovf() {
                            self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
                        }
                    }
                }
            }
            self.short.extend_from_slice(arg_guards);
            self.short_results.insert(canonical);
            self.short.push(preamble_op.clone());
            if preamble_op.opcode.is_ovf() {
                self.short.push(Op::new(OpCode::GuardNoOverflow, &[]));
            }
            self.short.extend_from_slice(result_guards);
        }
        // shortpreamble.py:481: self.short.append(jump_op)
        if let Some(jump) = jump_op {
            self.short.push(jump);
        }
    }

    pub fn produced_short_op(&self, result: OpRef) -> Option<ProducedShortOp> {
        self.produced_short_boxes.get(&result).cloned()
    }

    pub fn short_inputargs(&self) -> &[OpRef] {
        &self.short_inputargs
    }

    pub fn build_short_preamble_struct(
        &self,
        loop_constants: &HashMap<u32, i64>,
        loop_constant_types: &HashMap<u32, majit_ir::Type>,
    ) -> ShortPreamble {
        // short[..len-1] excludes the JUMP sentinel
        let ops: Vec<Op> = self.short[..self.short_ops_len()].to_vec();
        let inputargs = if self.label_args.is_empty() {
            &self.short_inputargs
        } else {
            &self.label_args
        };
        let mut sp = build_short_preamble_struct_from_ops(
            inputargs,
            &ops,
            &self.used_boxes,
            &self.short_jump_args,
            loop_constants,
            loop_constant_types,
        );
        if inputargs != &self.short_inputargs {
            sp.phase1_inputargs = Some(self.short_inputargs.clone());
        }
        sp
    }

    pub fn extra_same_as(&self) -> &[Op] {
        &self.extra_same_as
    }

    pub fn label_args(&self) -> &[OpRef] {
        &self.label_args
    }

    pub fn jump_args(&self) -> &[OpRef] {
        &self.short_jump_args
    }

    /// short ops length excluding JUMP sentinel.
    pub fn short_ops_len(&self) -> usize {
        self.short.len().saturating_sub(1)
    }

    pub fn short_op(&self, index: usize) -> Option<&Op> {
        if index < self.short_ops_len() {
            self.short.get(index)
        } else {
            None
        }
    }
}

/// shortpreamble.py: build short preamble from optimizer state.
/// Called after preamble optimization is complete.
/// Collects guards + pure ops from the optimized preamble and
/// maps them to label arg indices.
pub fn build_from_preamble_and_label(
    preamble_ops: &[Op],
    label_args: &[OpRef],
    exported_state: Option<VirtualState>,
) -> ShortPreamble {
    let mut builder = CollectedShortPreambleBuilder::new();
    let mut included_ovf_positions = HashSet::new();
    // Record all preamble ops
    for (idx, op) in preamble_ops.iter().enumerate() {
        if op.opcode.is_guard() {
            if op.opcode.is_guard_overflow()
                && idx > 0
                && preamble_ops[idx - 1].opcode.is_ovf()
                && included_ovf_positions.insert(preamble_ops[idx - 1].pos)
            {
                builder.add_preamble_op(&preamble_ops[idx - 1]);
            }
            builder.add_preamble_guard(op);
        } else if op.opcode.is_always_pure() {
            builder.add_preamble_op(op);
        }
    }
    // Set label args to create the mapping
    builder.set_label_args(label_args);
    builder.build(exported_state)
}

/// Extract guards AND pure ops from a peeled trace's preamble section.
///
/// Given a peeled trace (output of OptUnroll), identifies the preamble
/// section (before the Label) and collects all guard + pure operations
/// as short preamble entries.
///
/// This is a simpler alternative to integrating the builder with the
/// optimizer — it works on already-peeled traces.
pub fn extract_short_preamble(peeled_ops: &[Op]) -> ShortPreamble {
    // Find the Label position
    let label_pos = peeled_ops.iter().position(|op| op.opcode == OpCode::Label);

    let label_pos = match label_pos {
        Some(pos) => pos,
        None => return ShortPreamble::empty(), // No label = no peeling happened
    };

    let label_args = &peeled_ops[label_pos].args;

    // Build preamble-to-label-arg mapping
    let mut preamble_to_label: HashMap<OpRef, usize> = HashMap::new();
    for (i, arg) in label_args.iter().enumerate() {
        preamble_to_label.insert(*arg, i);
    }

    // shortpreamble.py: Collect guards AND pure operations from the preamble.
    // Guards must be replayed so the body's assumptions hold.
    // Pure ops whose results are used as label args must also be replayed
    // (e.g., GETFIELD from preamble that feeds into loop body).
    let mut entries = Vec::new();
    let mut included_positions = HashSet::new();
    for (idx, op) in peeled_ops[..label_pos].iter().enumerate() {
        let mut included_overflow_producer = false;
        if op.opcode.is_guard_overflow() && idx > 0 {
            let ovf_op = &peeled_ops[idx - 1];
            if ovf_op.opcode.is_ovf() && included_positions.insert(ovf_op.pos) {
                let ovf_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .args
                    .iter()
                    .enumerate()
                    .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
                    .collect();
                let ovf_fail_arg_mapping: Vec<(usize, usize)> = ovf_op
                    .fail_args
                    .as_ref()
                    .into_iter()
                    .flat_map(|fail_args| fail_args.iter().enumerate())
                    .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
                    .collect();
                if !ovf_arg_mapping.is_empty() || !ovf_fail_arg_mapping.is_empty() {
                    entries.push(ShortPreambleOp {
                        op: ovf_op.clone(),
                        arg_mapping: ovf_arg_mapping,
                        fail_arg_mapping: ovf_fail_arg_mapping,
                    });
                    included_overflow_producer = true;
                } else {
                    included_positions.remove(&ovf_op.pos);
                }
            }
        }
        if op.opcode.is_guard_overflow() && !included_overflow_producer {
            continue;
        }
        let include = op.opcode.is_guard() || op.opcode.is_always_pure();
        if !include {
            continue;
        }

        let arg_mapping: Vec<(usize, usize)> = op
            .args
            .iter()
            .enumerate()
            .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
            .collect();
        let fail_arg_mapping: Vec<(usize, usize)> = op
            .fail_args
            .as_ref()
            .into_iter()
            .flat_map(|fail_args| fail_args.iter().enumerate())
            .filter_map(|(pos, arg)| preamble_to_label.get(arg).map(|&idx| (pos, idx)))
            .collect();

        // Only include ops that reference label args
        if (!arg_mapping.is_empty() || !fail_arg_mapping.is_empty())
            && included_positions.insert(op.pos)
        {
            entries.push(ShortPreambleOp {
                op: op.clone(),
                arg_mapping,
                fail_arg_mapping,
            });
        }
    }

    ShortPreamble {
        ops: entries,
        inputargs: Vec::new(),
        used_boxes: Vec::new(),
        jump_args: Vec::new(),
        exported_state: None,
        constants: HashMap::new(),
        phase1_inputargs: None,
        inputarg_infos: Vec::new(),
    }
}

/// `unroll.py:497 ExportedState.short_boxes` shape: per-OpRef
/// `ProducedShortOp` records derived from `ctx.exported_short_boxes`,
/// with label-arg references in each preamble op renamed to the
/// matching short-inputarg slot
/// (`shortpreamble.py:269-270 ShortBoxes.create_short_boxes`).
///
/// OVF guards are filtered out: the guard entry depends on the
/// preceding `Int*Ovf` op and is re-emitted by the builder through
/// `append_to_short`'s `is_ovf` branch, so the standalone guard must
/// not appear in the produced map.
///
/// Phase B prep: extracted from
/// `build_short_preamble_from_exported_boxes` so that future B1 wiring
/// can store the same shape on `ExportedState.produced_short_boxes`
/// (audit memo `box_identity_phase_b_surface_audit_2026_05_02.md`
/// option (b)) without duplicating the rename logic.
pub fn produced_short_boxes_from_exported_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    exported_short_boxes: &[PreambleOp],
) -> Vec<(OpRef, ProducedShortOp)> {
    let inputarg_map: HashMap<OpRef, OpRef> = label_args
        .iter()
        .copied()
        .zip(short_inputargs.iter().copied())
        .collect();
    exported_short_boxes
        .iter()
        .filter(|entry| !entry.op.opcode.is_guard_overflow())
        .map(|entry| {
            let mut preamble_op = entry.op.clone();
            for arg in &mut preamble_op.args {
                if let Some(&renamed) = inputarg_map.get(arg) {
                    *arg = renamed;
                }
            }
            if let Some(fail_args) = preamble_op.fail_args.as_mut() {
                for arg in fail_args {
                    if let Some(&renamed) = inputarg_map.get(arg) {
                        *arg = renamed;
                    }
                }
            }
            (
                preamble_op.pos,
                ProducedShortOp {
                    kind: entry.kind.clone(),
                    preamble_op,
                    invented_name: entry.invented_name,
                    same_as_source: entry.same_as_source,
                },
            )
        })
        .collect()
}

/// `unroll.py:497-504` build path: drive `ShortPreambleBuilder` from
/// pre-derived `produced_short_boxes` (RPython `ExportedState.short_boxes`).
///
/// Phase B B1 first slice: separated from
/// `build_short_preamble_from_exported_boxes` so callers that already
/// hold the `Vec<(OpRef, ProducedShortOp)>` (e.g. `ExportedState.produced_short_boxes`
/// at `unroll.rs:2349`) can invoke the builder directly without
/// re-running the rename + filter pass.
pub fn build_short_preamble_from_produced_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    produced: &[(OpRef, ProducedShortOp)],
    loop_constants: &HashMap<u32, i64>,
    loop_constant_types: &HashMap<u32, majit_ir::Type>,
) -> ShortPreamble {
    let mut builder = ShortPreambleBuilder::new(label_args, produced, short_inputargs);
    // RPython parity: populate known_constants so produce_arg can resolve
    // constant OpRefs (10000+ range) in short op args. Without this,
    // add_op_to_short fails for ops like GetfieldGcPure(constant_ptr)
    // because produce_arg(constant_ptr) returns None.
    for &idx in loop_constants.keys() {
        builder.note_known_constant(OpRef::from_raw(idx));
    }
    for (result, _) in produced {
        let _ = builder.add_op_to_short(*result);
        let _ = builder.add_preamble_op(*result);
    }
    builder.build_short_preamble_struct(loop_constants, loop_constant_types)
}

pub fn build_short_preamble_from_exported_boxes(
    label_args: &[OpRef],
    short_inputargs: &[OpRef],
    exported_short_boxes: &[PreambleOp],
    loop_constants: &HashMap<u32, i64>,
    loop_constant_types: &HashMap<u32, majit_ir::Type>,
) -> ShortPreamble {
    let produced =
        produced_short_boxes_from_exported_boxes(label_args, short_inputargs, exported_short_boxes);
    build_short_preamble_from_produced_boxes(
        label_args,
        short_inputargs,
        &produced,
        loop_constants,
        loop_constant_types,
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

    #[test]
    fn test_empty_short_preamble() {
        let sp = ShortPreamble::empty();
        assert!(sp.is_empty());
        assert_eq!(sp.len(), 0);
    }

    #[test]
    fn test_extract_from_peeled_trace() {
        // Simulate a peeled trace:
        // 0: guard_true(v100)        ← preamble guard on loop-carried value
        // 1: int_add(v100, v101)     ← preamble computation
        // 2: Label(v100, v101)       ← loop header
        // 3: guard_true(v100)        ← body guard (same)
        // 4: int_add(v100, v101)     ← body computation
        // 5: Jump(v4, v101)          ← back-edge
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Label, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(4), OpRef::from_raw(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // Should have captured the preamble guard AND the pure IntAdd
        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntAdd);

        // The guard's arg v100 maps to label arg index 0
        assert_eq!(sp.ops[0].arg_mapping.len(), 1);
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0)); // arg position 0 → label arg 0
    }

    #[test]
    fn test_extract_no_label() {
        // No label = no peeling happened
        let ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Finish, &[OpRef::from_raw(0)]),
        ];

        let sp = extract_short_preamble(&ops);
        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_overflow_guard_includes_preceding_ovf_op() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[OpRef::from_raw(100), OpRef::from_raw(100)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].fail_args = Some(vec![OpRef::from_raw(100)].into());

        let sp = extract_short_preamble(&ops);

        assert_eq!(sp.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntMulOvf);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNoOverflow);
    }

    #[test]
    fn test_extract_overflow_guard_without_replayable_ovf_is_skipped() {
        let mut ops = vec![
            Op::new(
                OpCode::IntMulOvf,
                &[OpRef::from_raw(200), OpRef::from_raw(200)],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Label, &[OpRef::from_raw(100)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(100)]),
        ];
        assign_positions(&mut ops, 0);
        ops[1].fail_args = Some(vec![OpRef::from_raw(100)].into());

        let sp = extract_short_preamble(&ops);

        assert!(sp.is_empty());
    }

    #[test]
    fn test_extract_skips_non_label_guards() {
        // Guards that don't reference label args should not be included
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]), // refs temporary, not label arg
            Op::new(OpCode::Label, &[OpRef::from_raw(100)]),   // only v100 is a label arg
            Op::new(OpCode::Jump, &[OpRef::from_raw(100)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // The guard refs v0 (the IntAdd result), which is NOT in the label args.
        // But IntAdd refs v100 which IS a label arg → IntAdd IS extracted.
        // The guard on v0 is NOT extracted (v0 is not a label arg).
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_builder_collects_guards() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Simulate preamble processing
        let guard1 = Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]);
        let guard2 = Op::new(
            OpCode::GuardClass,
            &[OpRef::from_raw(101), OpRef::from_raw(200)],
        );
        let non_guard = Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
        );

        builder.add_preamble_guard(&guard1);
        builder.add_preamble_guard(&guard2);
        builder.add_preamble_guard(&non_guard); // should be ignored (not a guard)

        // Set label args (preamble phase ends)
        builder.set_label_args(&[OpRef::from_raw(100), OpRef::from_raw(101)]);

        // After label, no more collection
        let guard3 = Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]);
        builder.add_preamble_guard(&guard3); // should be ignored

        let sp = builder.build(None);
        assert_eq!(sp.len(), 2); // Only the two guards from the preamble
    }

    #[test]
    fn test_builder_maps_args_to_label_indices() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // Preamble has guard on v100 and v101
        let guard = Op::new(
            OpCode::GuardValue,
            &[OpRef::from_raw(100), OpRef::from_raw(200)],
        );
        builder.add_preamble_guard(&guard);

        // Label carries v100 as arg 0 and v101 as arg 1
        builder.set_label_args(&[OpRef::from_raw(100), OpRef::from_raw(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.ops[0].arg_mapping.len(), 1); // v100 → label arg 0
        assert_eq!(sp.ops[0].arg_mapping[0], (0, 0));
        // v200 is not a label arg, so it's not in the mapping
    }

    #[test]
    fn test_builder_add_preamble_op_any_type() {
        let mut builder = CollectedShortPreambleBuilder::new();

        // add_preamble_op accepts any op type (not just guards)
        let pure_op = Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
        );
        builder.add_preamble_op(&pure_op);

        builder.set_label_args(&[OpRef::from_raw(100), OpRef::from_raw(101)]);

        let sp = builder.build(None);
        assert_eq!(sp.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_extract_multiple_guards() {
        // Multiple guards in the preamble
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(101)]),
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(100), OpRef::from_raw(200)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Label, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::Jump, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // All three guards + the pure IntAdd reference label args
        assert_eq!(sp.len(), 4);
        assert_eq!(sp.ops[0].op.opcode, OpCode::GuardTrue);
        assert_eq!(sp.ops[1].op.opcode, OpCode::GuardNonnull);
        assert_eq!(sp.ops[2].op.opcode, OpCode::GuardClass);
        assert_eq!(sp.ops[3].op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_roundtrip_extract_and_instantiate() {
        // Full round-trip: peel → extract short preamble → instantiate for bridge
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::GuardClass,
                &[OpRef::from_raw(101), OpRef::from_raw(200)],
            ),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Label, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
            Op::new(OpCode::Jump, &[OpRef::from_raw(4), OpRef::from_raw(101)]),
        ];
        assign_positions(&mut ops, 0);

        let sp = extract_short_preamble(&ops);

        // Instantiate for bridge with new values
        let bridge_args = &[OpRef::from_raw(500), OpRef::from_raw(501)];
        let instantiated = sp.instantiate(bridge_args);

        // 2 guards + 1 pure IntAdd
        assert_eq!(instantiated.len(), 3);

        // Guard_true now checks bridge's v500 (was v100 → label arg 0)
        assert_eq!(instantiated[0].opcode, OpCode::GuardTrue);
        assert_eq!(instantiated[0].args[0], OpRef::from_raw(500));

        // Guard_class now checks bridge's v501 against constant v200
        assert_eq!(instantiated[1].opcode, OpCode::GuardClass);
        assert_eq!(instantiated[1].args[0], OpRef::from_raw(501)); // remapped
        assert_eq!(instantiated[1].args[1], OpRef::from_raw(200)); // constant, unchanged

        // IntAdd with remapped args
        assert_eq!(instantiated[2].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_uses_exported_order() {
        let exported = vec![
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                    op.pos = OpRef::from_raw(7);
                    op
                },
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: {
                    let mut op = Op::new(OpCode::IntSub, &[OpRef::from_raw(7), OpRef::from_raw(1)]);
                    op.pos = OpRef::from_raw(8);
                    op
                },
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(
            &[OpRef::from_raw(0), OpRef::from_raw(1)],
            &[OpRef::from_raw(10), OpRef::from_raw(11)],
            &exported,
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(sp.ops.len(), 2);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(sp.ops[1].op.opcode, OpCode::IntSub);
        assert_eq!(sp.ops[1].arg_mapping, vec![(1, 1)]);
        assert_eq!(sp.inputargs, vec![OpRef::from_raw(10), OpRef::from_raw(11)]);
    }

    #[test]
    fn test_build_short_preamble_from_exported_boxes_skips_standalone_overflow_guards() {
        let label_args = vec![OpRef::from_raw(10), OpRef::from_raw(11)];
        let short_inputargs = vec![OpRef::from_raw(100), OpRef::from_raw(101)];

        let mut ovf = Op::new(
            OpCode::IntAddOvf,
            &[OpRef::from_raw(10), OpRef::from_raw(11)],
        );
        ovf.pos = OpRef::from_raw(20);
        let guard = Op::new(OpCode::GuardNoOverflow, &[]);

        let exported = vec![
            PreambleOp {
                op: ovf,
                kind: PreambleOpKind::Pure,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
            PreambleOp {
                op: guard,
                kind: PreambleOpKind::Guard,
                label_arg_idx: None,
                invented_name: false,
                same_as_source: None,
            },
        ];

        let sp = build_short_preamble_from_exported_boxes(
            &label_args,
            &short_inputargs,
            &exported,
            &HashMap::new(),
            &HashMap::new(),
        );
        let opcodes: Vec<OpCode> = sp.ops.iter().map(|entry| entry.op.opcode).collect();
        assert_eq!(opcodes, vec![OpCode::IntAddOvf, OpCode::GuardNoOverflow]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_recurses_dependencies() {
        let produced = vec![
            (
                OpRef::from_raw(7),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    preamble_op: {
                        let mut op =
                            Op::new(OpCode::IntAdd, &[OpRef::from_raw(0), OpRef::from_raw(1)]);
                        op.pos = OpRef::from_raw(7);
                        op
                    },
                    invented_name: false,
                    same_as_source: None,
                },
            ),
            (
                OpRef::from_raw(8),
                ProducedShortOp {
                    kind: PreambleOpKind::Pure,
                    preamble_op: {
                        let mut op =
                            Op::new(OpCode::IntMul, &[OpRef::from_raw(7), OpRef::from_raw(1)]);
                        op.pos = OpRef::from_raw(8);
                        op
                    },
                    invented_name: false,
                    same_as_source: None,
                },
            ),
        ];
        let mut builder = ShortPreambleBuilder::new(
            &[OpRef::from_raw(0), OpRef::from_raw(1)],
            &produced,
            &[OpRef::from_raw(0), OpRef::from_raw(1)],
        );

        let used = builder.add_op_to_short(OpRef::from_raw(8)).unwrap();
        assert!(builder.add_preamble_op(OpRef::from_raw(7)));
        assert!(builder.add_preamble_op(OpRef::from_raw(8)));
        assert_eq!(used.opcode, OpCode::IntMul);
        let short = builder.build_short_preamble();
        assert_eq!(short[1].opcode, OpCode::IntAdd);
        assert_eq!(short[2].opcode, OpCode::IntMul);
        assert_eq!(
            builder.used_boxes(),
            &[OpRef::from_raw(7), OpRef::from_raw(8)]
        );
    }

    #[test]
    fn test_build_from_preamble_and_label() {
        let mut preamble = vec![
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]),
            Op::new(
                OpCode::IntAdd,
                &[OpRef::from_raw(100), OpRef::from_raw(101)],
            ),
        ];
        assign_positions(&mut preamble, 0);

        let label_args = &[OpRef::from_raw(100), OpRef::from_raw(101)];
        let sp = build_from_preamble_and_label(&preamble, label_args, None);

        // Guard + pure IntAdd
        assert_eq!(sp.len(), 2);
    }

    #[test]
    fn test_extended_builder() {
        let mut builder = CollectedExtendedShortPreambleBuilder::new();
        builder.set_label_args(&[OpRef::from_raw(100), OpRef::from_raw(101)]);
        builder.add_guard(Op::new(OpCode::GuardTrue, &[OpRef::from_raw(100)]));
        builder.add_pure_op(Op::new(
            OpCode::IntAdd,
            &[OpRef::from_raw(100), OpRef::from_raw(101)],
        ));
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(100)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(102);
        builder.add_heap_op(heap);
        builder.add_loopinvariant_op(Op::new(OpCode::CallI, &[OpRef::from_raw(100)]));
        assert_eq!(builder.num_ops(), 4);
    }

    #[test]
    fn test_short_boxes() {
        let mut sb = ShortBoxes::with_label_args(&[
            OpRef::from_raw(10),
            OpRef::from_raw(11),
            OpRef::from_raw(12),
        ]);
        assert_eq!(sb.num_label_args, 3);
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_raw(11)]);
        pure.pos = OpRef::from_raw(20);
        sb.add_pure_op(pure);
        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(10)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(21);
        sb.add_heap_op(heap);
        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);
    }

    #[test]
    fn test_short_boxes_reject_unknown_nonconstant_dependency() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef::from_raw(10)]);
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_raw(999)]);
        pure.pos = OpRef::from_raw(20);
        sb.add_pure_op(pure);

        let produced = sb.produced_ops();
        // The label arg OpRef::from_raw(10) itself is produced (as ShortInputArg),
        // but the pure op depending on unknown OpRef::from_raw(999) is rejected.
        assert!(
            !produced.iter().any(|(r, _)| *r == OpRef::from_raw(20)),
            "pure op with unknown dependency should be rejected"
        );
    }

    #[test]
    fn test_short_boxes_accept_known_constant_dependency() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef::from_raw(10)]);
        sb.note_known_constant(OpRef::from_raw(999));
        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(10), OpRef::from_raw(999)]);
        pure.pos = OpRef::from_raw(20);
        sb.add_pure_op(pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 1);
        let pure = produced
            .iter()
            .find(|(result, _)| *result == OpRef::from_raw(20))
            .expect("missing produced pure op");
        assert_eq!(
            pure.1.preamble_op.args.as_slice(),
            &[OpRef::from_raw(10), OpRef::from_raw(999)]
        );
    }

    #[test]
    fn test_short_boxes_compound_prefers_non_heap_and_emits_invented_alias() {
        let mut sb = ShortBoxes::with_label_args(&[
            OpRef::from_raw(10),
            OpRef::from_raw(30),
            OpRef::from_raw(31),
        ]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(10);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(30), OpRef::from_raw(31)]);
        pure.pos = OpRef::from_raw(10);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::from_raw(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, _)| *result != OpRef::from_raw(10))
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        assert_eq!(alias.1.same_as_source, Some(OpRef::from_raw(10)));
    }

    #[test]
    fn test_short_boxes_nested_compound_emits_multiple_invented_aliases() {
        let mut sb = ShortBoxes::with_label_args(&[
            OpRef::from_raw(20),
            OpRef::from_raw(30),
            OpRef::from_raw(31),
        ]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(20);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut loopinv = Op::new(OpCode::CallI, &[OpRef::from_raw(30)]);
        loopinv.pos = OpRef::from_raw(20);
        sb.add_potential_op(Some(0), loopinv, PreambleOpKind::LoopInvariant);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(30), OpRef::from_raw(31)]);
        pure.pos = OpRef::from_raw(20);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 3);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::from_raw(20))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::Pure);
        assert!(!chosen.1.invented_name);

        let aliases: Vec<_> = produced
            .iter()
            .filter(|(result, _)| *result != OpRef::from_raw(20))
            .collect();
        assert_eq!(aliases.len(), 2);
        assert!(aliases.iter().all(|(_, produced)| produced.invented_name));
        assert!(
            aliases
                .iter()
                .all(|(_, produced)| produced.same_as_source == Some(OpRef::from_raw(20)))
        );
    }

    #[test]
    fn test_rpython_create_short_boxes_prefers_short_inputarg_over_heap_result() {
        let mut sb = ShortBoxes::with_label_args(&[OpRef::from_raw(10), OpRef::from_raw(30)]);
        sb.add_short_input_arg(OpRef::from_raw(10), majit_ir::Type::Int);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(10);
        sb.add_heap_op(heap);

        let produced = sb.produced_ops();
        assert_eq!(produced.len(), 2);

        let chosen = produced
            .iter()
            .find(|(result, _)| *result == OpRef::from_raw(10))
            .unwrap();
        assert_eq!(chosen.1.kind, PreambleOpKind::InputArg);
        assert!(!chosen.1.invented_name);

        let alias = produced
            .iter()
            .find(|(result, _)| *result != OpRef::from_raw(10))
            .unwrap();
        assert_eq!(alias.1.kind, PreambleOpKind::Heap);
        assert!(alias.1.invented_name);
        assert_eq!(alias.1.same_as_source, Some(OpRef::from_raw(10)));
    }

    #[test]
    fn test_rpython_short_preamble_builder_add_op_to_short_builds_label_short_and_jump() {
        let mut sb = ShortBoxes::with_label_args(&[
            OpRef::from_raw(10),
            OpRef::from_raw(30),
            OpRef::from_raw(31),
        ]);

        let mut ovf = Op::new(
            OpCode::IntAddOvf,
            &[OpRef::from_raw(30), OpRef::from_raw(31)],
        );
        ovf.pos = OpRef::from_raw(10);
        sb.add_potential_op(Some(0), ovf, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef::from_raw(10)], &produced, &[OpRef::from_raw(10)]);
        let used = builder.add_op_to_short(OpRef::from_raw(10)).unwrap();
        assert!(builder.add_preamble_op(OpRef::from_raw(10)));
        assert_eq!(used.opcode, OpCode::IntAddOvf);
        assert_eq!(builder.used_boxes(), &[OpRef::from_raw(10)]);

        let short = builder.build_short_preamble();
        assert_eq!(short.len(), 4);
        assert_eq!(short[0].opcode, OpCode::Label);
        assert_eq!(short[1].opcode, OpCode::IntAddOvf);
        assert_eq!(short[2].opcode, OpCode::GuardNoOverflow);
        assert_eq!(short[3].opcode, OpCode::Jump);
        assert_eq!(short[3].args.as_slice(), &[OpRef::from_raw(10)]);
    }

    #[test]
    fn test_rpython_short_preamble_builder_tracks_extra_same_as() {
        let mut sb = ShortBoxes::with_label_args(&[
            OpRef::from_raw(20),
            OpRef::from_raw(30),
            OpRef::from_raw(31),
        ]);

        let mut heap = Op::with_descr(
            OpCode::GetfieldGcI,
            &[OpRef::from_raw(30)],
            majit_ir::make_field_descr(0, 8, majit_ir::Type::Int, majit_ir::ArrayFlag::Signed),
        );
        heap.pos = OpRef::from_raw(20);
        sb.add_potential_op(Some(0), heap, PreambleOpKind::Heap);

        let mut pure = Op::new(OpCode::IntAdd, &[OpRef::from_raw(30), OpRef::from_raw(31)]);
        pure.pos = OpRef::from_raw(20);
        sb.add_potential_op(Some(0), pure, PreambleOpKind::Pure);

        let produced = sb.produced_ops();
        let alias_result = produced
            .iter()
            .find(|(result, pop)| *result != OpRef::from_raw(20) && pop.invented_name)
            .map(|(result, _)| *result)
            .unwrap();

        let mut builder =
            ShortPreambleBuilder::new(&[OpRef::from_raw(20)], &produced, &[OpRef::from_raw(20)]);
        assert!(builder.add_preamble_op(alias_result));
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos, alias_result);
        assert_eq!(extra[0].args.as_slice(), &[OpRef::from_raw(20)]);
    }

    #[test]
    fn test_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let mut builder =
            ShortPreambleBuilder::new(&[OpRef::from_raw(7)], &[], &[OpRef::from_raw(7)]);
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[OpRef::from_raw(30)]);
        replay_op.pos = OpRef::from_raw(14);
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef::from_raw(14),
            invented_name: true,
            preamble_op: replay_op,
        };

        builder.add_preamble_op_from_pop(&pop, OpRef::from_raw(41));

        assert_eq!(builder.used_boxes(), &[OpRef::from_raw(41)]);
        assert_eq!(builder.short_preamble_jump().len(), 1);
        assert_eq!(builder.short_preamble_jump()[0].pos, OpRef::from_raw(14));
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos, OpRef::from_raw(41));
        assert_eq!(extra[0].args.as_slice(), &[OpRef::from_raw(14)]);
    }

    #[test]
    fn test_extended_short_preamble_builder_fallback_keeps_invented_name_alias_identity() {
        let sb = ShortPreambleBuilder::new(&[OpRef::from_raw(7)], &[], &[OpRef::from_raw(7)]);
        let mut builder = ExtendedShortPreambleBuilder::new(0, &sb);
        let mut replay_op = Op::new(OpCode::GetfieldGcI, &[OpRef::from_raw(30)]);
        replay_op.pos = OpRef::from_raw(14);
        let pop = crate::optimizeopt::info::PreambleOp {
            op: OpRef::from_raw(14),
            invented_name: true,
            preamble_op: replay_op,
        };

        builder.add_preamble_op_from_pop(&pop, OpRef::from_raw(41));

        assert_eq!(builder.label_args(), &[OpRef::from_raw(41)]);
        assert_eq!(builder.jump_args(), &[OpRef::from_raw(14)]);
        let extra = builder.extra_same_as();
        assert_eq!(extra.len(), 1);
        assert_eq!(extra[0].opcode, OpCode::SameAsI);
        assert_eq!(extra[0].pos, OpRef::from_raw(41));
        assert_eq!(extra[0].args.as_slice(), &[OpRef::from_raw(14)]);
    }
}
