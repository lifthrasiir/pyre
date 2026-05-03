/// Bridge optimization pass.
///
/// Translated from rpython/jit/metainterp/optimizeopt/bridgeopt.py.
///
/// When compiling a bridge (alternative path from a failed guard), we know
/// certain facts about values at the guard failure point — e.g., class is
/// known from the guard that failed, values are bounded from loop analysis.
///
/// This pass imports that knowledge so guards in the bridge body can be
/// removed or simplified.
use std::collections::HashMap;

use majit_ir::{GcRef, Op, OpCode, OpRef, Value};

use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// Known facts about values at bridge entry.
#[derive(Clone, Debug)]
pub struct BridgeKnowledge {
    /// Values known to be specific constants.
    pub known_constants: HashMap<OpRef, i64>,
    /// Values known to be non-null.
    pub known_nonnull: Vec<OpRef>,
    /// Values with known class (from GuardClass).
    pub known_classes: HashMap<OpRef, GcRef>,
    /// Integer bounds: (opref, lower, upper).
    pub known_bounds: HashMap<OpRef, (i64, i64)>,
    /// bridgeopt.py: known heap field values.
    /// (object_ref, field_descr_index) → value OpRef.
    pub known_fields: HashMap<(OpRef, u32), OpRef>,
    /// bridgeopt.py: known array item values.
    /// (array_ref, index, descr_index) → value OpRef.
    pub known_arrayitems: HashMap<(OpRef, i64, u32), OpRef>,
}

impl BridgeKnowledge {
    pub fn new() -> Self {
        BridgeKnowledge {
            known_constants: HashMap::new(),
            known_nonnull: Vec::new(),
            known_classes: HashMap::new(),
            known_bounds: HashMap::new(),
            known_fields: HashMap::new(),
            known_arrayitems: HashMap::new(),
        }
    }

    /// Add a known field value.
    pub fn add_known_field(&mut self, obj: OpRef, field_idx: u32, value: OpRef) {
        self.known_fields.insert((obj, field_idx), value);
    }

    /// Add a known array item value.
    pub fn add_known_arrayitem(&mut self, array: OpRef, index: i64, descr_idx: u32, value: OpRef) {
        self.known_arrayitems
            .insert((array, index, descr_idx), value);
    }

    /// Number of total known facts.
    pub fn num_facts(&self) -> usize {
        self.known_constants.len()
            + self.known_nonnull.len()
            + self.known_classes.len()
            + self.known_bounds.len()
            + self.known_fields.len()
            + self.known_arrayitems.len()
    }
}

impl Default for BridgeKnowledge {
    fn default() -> Self {
        Self::new()
    }
}

impl BridgeKnowledge {
    /// bridgeopt.py: serialize_optimizer_knowledge(numb_state, liveboxes, ...)
    /// Serialize knowledge into a compact byte representation for embedding
    /// in resume data. This allows bridges to inherit optimization knowledge
    /// from the loop that spawned them.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Format: [num_constants, (opref, value)*, num_nonnull, opref*,
        //          num_classes, (opref, class_ptr)*, num_bounds, (opref, lo, hi)*]
        buf.extend_from_slice(&(self.known_constants.len() as u32).to_le_bytes());
        for (&opref, &value) in &self.known_constants {
            buf.extend_from_slice(&opref.raw().to_le_bytes());
            buf.extend_from_slice(&value.to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_nonnull.len() as u32).to_le_bytes());
        for opref in &self.known_nonnull {
            buf.extend_from_slice(&opref.raw().to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_classes.len() as u32).to_le_bytes());
        for (&opref, &class) in &self.known_classes {
            buf.extend_from_slice(&opref.raw().to_le_bytes());
            buf.extend_from_slice(&(class.0 as u64).to_le_bytes());
        }
        buf.extend_from_slice(&(self.known_bounds.len() as u32).to_le_bytes());
        for (&opref, &(lo, hi)) in &self.known_bounds {
            buf.extend_from_slice(&opref.raw().to_le_bytes());
            buf.extend_from_slice(&lo.to_le_bytes());
            buf.extend_from_slice(&hi.to_le_bytes());
        }
        // Serialize known_fields
        buf.extend_from_slice(&(self.known_fields.len() as u32).to_le_bytes());
        for (&(obj, field_idx), &value) in &self.known_fields {
            buf.extend_from_slice(&obj.raw().to_le_bytes());
            buf.extend_from_slice(&field_idx.to_le_bytes());
            buf.extend_from_slice(&value.raw().to_le_bytes());
        }
        // Serialize known_arrayitems
        buf.extend_from_slice(&(self.known_arrayitems.len() as u32).to_le_bytes());
        for (&(array, index, descr_idx), &value) in &self.known_arrayitems {
            buf.extend_from_slice(&array.raw().to_le_bytes());
            buf.extend_from_slice(&index.to_le_bytes());
            buf.extend_from_slice(&descr_idx.to_le_bytes());
            buf.extend_from_slice(&value.raw().to_le_bytes());
        }
        buf
    }

    /// bridgeopt.py: deserialize_optimizer_knowledge(numb_state, liveboxes, ...)
    /// Reconstruct knowledge from serialized bytes.
    pub fn deserialize(buf: &[u8]) -> Option<Self> {
        let mut pos = 0;
        let mut k = BridgeKnowledge::new();

        if buf.len() < 4 {
            return None;
        }
        let n_const = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_const {
            if pos + 12 > buf.len() {
                return None;
            }
            let opref = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
            pos += 4;
            let value = i64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);
            pos += 8;
            k.known_constants.insert(opref, value);
        }

        if pos + 4 > buf.len() {
            return None;
        }
        let n_nonnull = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_nonnull {
            if pos + 4 > buf.len() {
                return None;
            }
            let opref = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
            pos += 4;
            k.known_nonnull.push(opref);
        }

        if pos + 4 > buf.len() {
            return None;
        }
        let n_class = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_class {
            if pos + 12 > buf.len() {
                return None;
            }
            let opref = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
            pos += 4;
            let class_val = u64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);
            pos += 8;
            k.known_classes.insert(opref, GcRef(class_val as usize));
        }

        if pos + 4 > buf.len() {
            return None;
        }
        let n_bounds = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        for _ in 0..n_bounds {
            if pos + 20 > buf.len() {
                return None;
            }
            let opref = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
            pos += 4;
            let lo = i64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);
            pos += 8;
            let hi = i64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);
            pos += 8;
            k.known_bounds.insert(opref, (lo, hi));
        }

        // Deserialize known_fields (if present — backwards compatible)
        if pos + 4 <= buf.len() {
            let n_fields = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            for _ in 0..n_fields {
                if pos + 12 > buf.len() {
                    break;
                }
                let obj = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
                pos += 4;
                let field_idx = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
                pos += 4;
                let value = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
                pos += 4;
                k.known_fields.insert((obj, field_idx), value);
            }
        }

        // Deserialize known_arrayitems (if present)
        if pos + 4 <= buf.len() {
            let n_items = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?) as usize;
            pos += 4;
            for _ in 0..n_items {
                if pos + 20 > buf.len() {
                    break;
                }
                let array = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
                pos += 4;
                let index = i64::from_le_bytes(buf[pos..pos + 8].try_into().ok()?);
                pos += 8;
                let descr_idx = u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?);
                pos += 4;
                let value = OpRef::from_raw(u32::from_le_bytes(buf[pos..pos + 4].try_into().ok()?));
                pos += 4;
                k.known_arrayitems.insert((array, index, descr_idx), value);
            }
        }

        Some(k)
    }

    /// Whether this knowledge has any useful facts.
    pub fn is_empty(&self) -> bool {
        self.known_constants.is_empty()
            && self.known_nonnull.is_empty()
            && self.known_classes.is_empty()
            && self.known_bounds.is_empty()
    }
}

/// The bridge optimization pass.
pub struct OptBridgeOpt {
    knowledge: BridgeKnowledge,
    /// Maps OpRef to the Op that produced it, for looking up comparison ops
    /// when processing GuardTrue/GuardFalse.
    produced_by: HashMap<OpRef, Op>,
}

impl OptBridgeOpt {
    pub fn new() -> Self {
        OptBridgeOpt {
            knowledge: BridgeKnowledge::new(),
            produced_by: HashMap::new(),
        }
    }

    /// Create a bridge optimizer with pre-existing knowledge.
    pub fn with_knowledge(knowledge: BridgeKnowledge) -> Self {
        OptBridgeOpt {
            knowledge,
            produced_by: HashMap::new(),
        }
    }

    /// Get the effective bounds for an operand: either from known_bounds or
    /// from a known constant (where lower == upper == constant value).
    fn get_bounds(&self, opref: OpRef, ctx: &OptContext) -> Option<(i64, i64)> {
        if let Some(&bounds) = self.knowledge.known_bounds.get(&opref) {
            return Some(bounds);
        }
        if let Some(val) = ctx.get_constant_int(opref) {
            return Some((val, val));
        }
        None
    }

    /// Check whether a comparison guard can be eliminated based on known bounds.
    ///
    /// For GuardTrue, the condition must be provably always true.
    /// For GuardFalse, the negation of the condition must be provably always true.
    fn can_eliminate_comparison_guard(
        &self,
        cond_op: &Op,
        guard_true: bool,
        ctx: &OptContext,
    ) -> bool {
        let cmp_opcode = cond_op.opcode;
        let a = ctx.get_box_replacement(cond_op.arg(0));
        let b = ctx.get_box_replacement(cond_op.arg(1));

        let a_bounds = self.get_bounds(a, ctx);
        let b_bounds = self.get_bounds(b, ctx);

        let (a_lo, a_hi) = match a_bounds {
            Some(bounds) => bounds,
            None => return false,
        };
        let (b_lo, b_hi) = match b_bounds {
            Some(bounds) => bounds,
            None => return false,
        };

        if guard_true {
            // Guard asserts the comparison is true.
            match cmp_opcode {
                OpCode::IntLt => a_hi < b_lo,
                OpCode::IntLe => a_hi <= b_lo,
                OpCode::IntGt => a_lo > b_hi,
                OpCode::IntGe => a_lo >= b_hi,
                // Unsigned comparisons: treat as unsigned, bounds still work.
                OpCode::UintLt => (a_hi as u64) < (b_lo as u64),
                OpCode::UintLe => (a_hi as u64) <= (b_lo as u64),
                OpCode::UintGt => (a_lo as u64) > (b_hi as u64),
                OpCode::UintGe => (a_lo as u64) >= (b_hi as u64),
                _ => false,
            }
        } else {
            match cmp_opcode {
                OpCode::IntLt => a_lo >= b_hi,
                OpCode::IntLe => a_lo > b_hi,
                OpCode::IntGt => a_hi <= b_lo,
                OpCode::IntGe => a_hi < b_lo,
                OpCode::UintLt => (a_lo as u64) >= (b_hi as u64),
                OpCode::UintLe => (a_lo as u64) > (b_hi as u64),
                OpCode::UintGt => (a_hi as u64) <= (b_lo as u64),
                OpCode::UintGe => (a_hi as u64) < (b_lo as u64),
                _ => false,
            }
        }
    }

    /// bridgeopt.py: deserialize_optimizer_knowledge
    /// Pre-populate the optimization context with all known facts.
    pub fn apply_knowledge(&self, ctx: &mut OptContext) {
        // bridgeopt.py: apply known constants
        for (&opref, &value) in &self.knowledge.known_constants {
            ctx.make_constant(opref, Value::Int(value));
        }
        // bridgeopt.py: apply known bounds — record as constants if single-value
        for (&opref, &(lo, hi)) in &self.knowledge.known_bounds {
            if lo == hi {
                ctx.make_constant(opref, Value::Int(lo));
            }
        }
    }

    /// bridgeopt.py: number of known class entries (for bitfield size)
    pub fn num_known_classes(&self) -> usize {
        self.knowledge.known_classes.len()
    }

    /// bridgeopt.py: number of known heap fields (for serialize size)
    pub fn num_known_fields(&self) -> usize {
        self.knowledge.known_fields.len()
    }

    /// Get the knowledge for inspection.
    pub fn knowledge(&self) -> &BridgeKnowledge {
        &self.knowledge
    }

    /// Get mutable knowledge for building.
    pub fn knowledge_mut(&mut self) -> &mut BridgeKnowledge {
        &mut self.knowledge
    }
}

impl Default for OptBridgeOpt {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptBridgeOpt {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Track comparison operations so we can look them up from guards.
        match op.opcode {
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::IntEq
            | OpCode::IntNe => {
                self.produced_by.insert(op.pos, op.clone());
            }
            _ => {}
        }

        match op.opcode {
            // If the guard value is already known as a constant,
            // the guard is redundant.
            OpCode::GuardValue => {
                let obj = ctx.get_box_replacement(op.arg(0));
                let expected = ctx.get_box_replacement(op.arg(1));
                if let (Some(a), Some(b)) =
                    (ctx.get_constant_int(obj), ctx.get_constant_int(expected))
                {
                    if a == b {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            // If we know the value is non-null from bridge knowledge,
            // the guard is redundant.
            OpCode::GuardNonnull => {
                let obj = ctx.get_box_replacement(op.arg(0));
                if self.knowledge.known_nonnull.contains(&obj) {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // If we know the class from bridge knowledge, class guard is redundant.
            OpCode::GuardClass | OpCode::GuardNonnullClass => {
                let obj = ctx.get_box_replacement(op.arg(0));
                if self.knowledge.known_classes.contains_key(&obj) {
                    return OptimizationResult::Remove;
                }
                OptimizationResult::PassOn
            }

            // Bounds-based guard elimination: if the comparison operands
            // have known bounds that prove the guard always succeeds,
            // the guard is redundant.
            OpCode::GuardTrue | OpCode::GuardFalse => {
                let cond_ref = ctx.get_box_replacement(op.arg(0));
                if let Some(cond_op) = self.produced_by.get(&cond_ref) {
                    let guard_true = op.opcode == OpCode::GuardTrue;
                    if self.can_eliminate_comparison_guard(cond_op, guard_true, ctx) {
                        return OptimizationResult::Remove;
                    }
                }
                OptimizationResult::PassOn
            }

            _ => OptimizationResult::PassOn,
        }
    }

    fn setup(&mut self) {}

    fn name(&self) -> &'static str {
        "bridgeopt"
    }
}

// ── bridgeopt.py: tag/decode helpers ──

/// bridgeopt.py: TAGCONST / TAGINT / TAGBOX constants for resume data encoding.
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;

/// bridgeopt.py: tag_box(box, liveboxes_from_env, memo)
/// Tag a live box reference for serialization in resume data.
/// Constants get TAGCONST, live boxes get their index with TAGBOX.
pub fn tag_box(opref: OpRef, liveboxes: &[OpRef]) -> u16 {
    if let Some(pos) = liveboxes.iter().position(|r| *r == opref) {
        ((pos as u16) << 2) | (TAGBOX as u16)
    } else {
        // Assume constant — encode as TAGINT with the raw value
        ((opref.raw() as u16) << 2) | (TAGINT as u16)
    }
}

/// bridgeopt.py: decode_box(tagged, liveboxes)
/// Decode a tagged reference back to an OpRef.
pub fn decode_box(tagged: u16, liveboxes: &[OpRef]) -> OpRef {
    let tag = (tagged & 0x3) as u8;
    let num = (tagged >> 2) as usize;
    match tag {
        TAGBOX => liveboxes.get(num).copied().unwrap_or(OpRef::NONE),
        TAGINT => OpRef::from_raw(num as u32),
        TAGCONST => OpRef::NONE, // constant pool lookup needed
        _ => OpRef::NONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizeopt::optimizer::Optimizer;

    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef::from_raw(i as u32);
        }
    }

    #[test]
    fn test_bridgeopt_removes_guard_nonnull_with_knowledge() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_nonnull.push(OpRef::from_raw(100));

        let mut ops = vec![Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert!(
            result.is_empty(),
            "guard_nonnull should be removed when known nonnull"
        );
    }

    #[test]
    fn test_bridgeopt_keeps_guard_nonnull_without_knowledge() {
        let mut ops = vec![Op::new(OpCode::GuardNonnull, &[OpRef::from_raw(100)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(
            result.len(),
            1,
            "guard_nonnull should be kept without knowledge"
        );
    }

    #[test]
    fn test_bridgeopt_removes_guard_class_with_knowledge() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_classes
            .insert(OpRef::from_raw(100), GcRef::NULL);

        let mut ops = vec![Op::new(
            OpCode::GuardClass,
            &[OpRef::from_raw(100), OpRef::from_raw(200)],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert!(
            result.is_empty(),
            "guard_class should be removed when class known"
        );
    }

    // --- Bounds-based guard elimination tests ---

    #[test]
    fn test_bounds_guard_elimination_int_lt() {
        // known_bounds for OpRef::from_raw(100) = (0, 50), constant 100
        // v0 = IntLt(OpRef::from_raw(100), const_200), GuardTrue(v0)
        // Since 50 < 100, the guard is always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef::from_raw(100), (0, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let mut constants = HashMap::new();
        constants.insert(200, 100);
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // IntLt remains, GuardTrue removed
        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
    }

    #[test]
    fn test_bounds_guard_not_eliminated_int_lt() {
        // known_bounds for OpRef::from_raw(100) = (0, 200), constant 100
        // Since 200 >= 100, cannot prove IntLt always true → keep guard.
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(100), (0, 200));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let mut constants = HashMap::new();
        constants.insert(200, 100);
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        assert_eq!(result.len(), 2, "GuardTrue should NOT be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
        assert_eq!(result[1].opcode, OpCode::GuardTrue);
    }

    #[test]
    fn test_bounds_guard_elimination_int_ge() {
        // known_bounds for OpRef::from_raw(100) = (50, 200)
        // known_bounds for OpRef::from_raw(101) = (10, 50)
        // IntGe(100, 101): a.lower(50) >= b.upper(50) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(100), (50, 200));
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntGe, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntGe);
    }

    #[test]
    fn test_bounds_guard_elimination_int_le() {
        // known_bounds for OpRef::from_raw(100) = (0, 30)
        // known_bounds for OpRef::from_raw(101) = (30, 100)
        // IntLe(100, 101): a.upper(30) <= b.lower(30) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef::from_raw(100), (0, 30));
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(101), (30, 100));

        let mut ops = vec![
            Op::new(OpCode::IntLe, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLe);
    }

    #[test]
    fn test_bounds_guard_elimination_int_gt() {
        // known_bounds for OpRef::from_raw(100) = (80, 200)
        // known_bounds for OpRef::from_raw(101) = (10, 50)
        // IntGt(100, 101): a.lower(80) > b.upper(50) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(100), (80, 200));
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntGt, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1, "GuardTrue should be removed");
        assert_eq!(result[0].opcode, OpCode::IntGt);
    }

    #[test]
    fn test_bounds_guard_false_elimination() {
        // GuardFalse(IntLt(a, b)) means a >= b.
        // known_bounds for OpRef::from_raw(100) = (80, 200), OpRef::from_raw(101) = (10, 50)
        // IntLt false means a >= b: a.lower(80) >= b.upper(50) → always false → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(100), (80, 200));
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 1, "GuardFalse should be removed");
        assert_eq!(result[0].opcode, OpCode::IntLt);
    }

    #[test]
    fn test_bounds_guard_false_not_eliminated() {
        // GuardFalse(IntLt(a, b)) means a >= b.
        // known_bounds for OpRef::from_raw(100) = (0, 200), OpRef::from_raw(101) = (10, 50)
        // a.lower(0) < b.upper(50), so cannot prove a >= b → keep guard.
        let mut knowledge = BridgeKnowledge::new();
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(100), (0, 200));
        knowledge
            .known_bounds
            .insert(OpRef::from_raw(101), (10, 50));

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardFalse, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::with_knowledge(knowledge)));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(result.len(), 2, "GuardFalse should NOT be removed");
    }

    #[test]
    fn test_bounds_guard_with_one_constant_operand() {
        // OpRef::from_raw(100) has known bounds (0, 50), OpRef::from_raw(200) is a constant 100.
        // IntLt(100, 200): a.upper(50) < b_const(100) → always true → remove.
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_bounds.insert(OpRef::from_raw(100), (0, 50));
        knowledge.known_constants.insert(OpRef::from_raw(200), 100);

        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(200)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let bridge = OptBridgeOpt::with_knowledge(knowledge);
        let mut ctx = OptContext::new(ops.len());
        bridge.apply_knowledge(&mut ctx);

        let mut pass = OptBridgeOpt::with_knowledge(bridge.knowledge.clone());
        // Process IntLt first so produced_by is populated.
        let r1 = pass.propagate_forward(&ops[0], &mut ctx);
        assert!(matches!(r1, OptimizationResult::PassOn));
        // Now process GuardTrue.
        let r2 = pass.propagate_forward(&ops[1], &mut ctx);
        assert!(
            matches!(r2, OptimizationResult::Remove),
            "GuardTrue should be removed with one constant operand"
        );
    }

    #[test]
    fn test_bounds_guard_no_bounds_info() {
        // No bounds info for either operand → guard kept.
        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef::from_raw(100), OpRef::from_raw(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef::from_raw(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptBridgeOpt::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result = opt.optimize_with_constants_and_inputs(
            &ops,
            &mut std::collections::HashMap::new(),
            1024,
        );

        assert_eq!(
            result.len(),
            2,
            "Guard should be kept when no bounds info available"
        );
    }

    #[test]
    fn test_bridgeopt_guard_value_with_known_constant() {
        let mut knowledge = BridgeKnowledge::new();
        knowledge.known_constants.insert(OpRef::from_raw(100), 42);

        let mut ops = vec![Op::new(
            OpCode::GuardValue,
            &[OpRef::from_raw(100), OpRef::from_raw(200)],
        )];
        assign_positions(&mut ops);

        let bridge = OptBridgeOpt::with_knowledge(knowledge);
        // Pre-populate constants
        let mut ctx = OptContext::new(ops.len());
        bridge.apply_knowledge(&mut ctx);
        ctx.make_constant(OpRef::from_raw(200), Value::Int(42));

        // Manual optimization to test with pre-populated constants
        let mut pass = OptBridgeOpt::new();
        let result = pass.propagate_forward(&ops[0], &mut ctx);
        match result {
            OptimizationResult::Remove => {} // expected
            other => panic!(
                "expected Remove, got {:?}",
                match other {
                    OptimizationResult::PassOn => "PassOn",
                    OptimizationResult::Emit(_) => "Emit",
                    OptimizationResult::Replace(_) => "Replace",
                    _ => "other",
                }
            ),
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut k = BridgeKnowledge::new();
        k.known_constants.insert(OpRef::from_raw(10), 42);
        k.known_constants.insert(OpRef::from_raw(20), -1);
        k.known_nonnull.push(OpRef::from_raw(30));
        k.known_classes.insert(OpRef::from_raw(40), GcRef(0x1000));
        k.known_bounds.insert(OpRef::from_raw(50), (0, 100));
        k.add_known_field(OpRef::from_raw(60), 5, OpRef::from_raw(70));
        k.add_known_arrayitem(OpRef::from_raw(80), 3, 7, OpRef::from_raw(90));

        let serialized = k.serialize();
        let deserialized = BridgeKnowledge::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.known_constants.len(), 2);
        assert_eq!(deserialized.known_constants[&OpRef::from_raw(10)], 42);
        assert_eq!(deserialized.known_constants[&OpRef::from_raw(20)], -1);
        assert_eq!(deserialized.known_nonnull.len(), 1);
        assert_eq!(deserialized.known_classes.len(), 1);
        assert_eq!(deserialized.known_bounds.len(), 1);
        assert_eq!(deserialized.known_fields.len(), 1);
        assert_eq!(deserialized.known_arrayitems.len(), 1);
        assert_eq!(deserialized.num_facts(), k.num_facts());
    }

    #[test]
    fn test_serialize_empty() {
        let k = BridgeKnowledge::new();
        assert!(k.is_empty());
        let serialized = k.serialize();
        let deserialized = BridgeKnowledge::deserialize(&serialized).unwrap();
        assert!(deserialized.is_empty());
    }
}

/// bridgeopt.py:124-185 deserialize_optimizer_knowledge.
///
/// Read optimizer knowledge from the guard's rd_numb and apply it
/// directly to the optimizer passes. RPython parity: the function
/// takes the optimizer and applies knowledge inline, never returning
/// an intermediate struct.
/// bridgeopt.py:124 signature:
/// deserialize_optimizer_knowledge(optimizer, resumestorage, frontend_boxes, liveboxes)
///
/// bridgeopt.py:63-122 `serialize_optimizer_knowledge(optimizer,
/// numb_state, liveboxes, liveboxes_from_env, memo)`.
///
/// Emits three serialized sections on every guard (RPython emits zeros
/// when the optheap/optrewrite caches are empty; the deserializer relies
/// on the sections always being present):
///
/// 1. known-class bitfield per Ref livebox (bridgeopt.py:74-90)
/// 2. heap field + array item triples (bridgeopt.py:92-108)
/// 3. loopinvariant call results (bridgeopt.py:113-122)
///
/// RPython splits the memo-side wrapper (`_add_optimizer_sections`,
/// resume.py:570-574) from the serialize core (`serialize_optimizer_knowledge`,
/// bridgeopt.py:63-122). pyre keeps the same split: this free function
/// carries the core, and `ResumeDataLoopMemo::_add_optimizer_sections`
/// forwards.
pub fn serialize_optimizer_knowledge(
    memo: &mut crate::resume::ResumeDataLoopMemo,
    numb_state: &mut crate::resume::NumberingState,
    liveboxes: &[Option<OpRef>],
    new_liveboxes: &crate::resume::LiveboxMap,
    env: &dyn majit_ir::BoxEnv,
    optimizer_knowledge: Option<&crate::resume::OptimizerKnowledgeForResume>,
) {
    // bridgeopt.py:64-67 `available_boxes = {}` followed by
    // `available_boxes[box] = None` — RPython uses a dict as a
    // membership set (values are always None). Rust mirrors the dict
    // shape with `HashMap<OpRef, ()>` rather than `HashSet<OpRef>` to
    // keep structural parity with the RPython data-structure choice.
    let available_boxes: std::collections::HashMap<OpRef, ()> = liveboxes
        .iter()
        .filter_map(|opt| *opt)
        .filter(|opref| numb_state.liveboxes.contains_key(opref.raw()))
        .map(|opref| (opref, ()))
        .collect();

    // bridgeopt.py:74-88: known classes bitfield
    // RPython: for each livebox, call getptrinfo(box).get_known_class(cpu).
    // The actual class pointer is recovered at deserialization time
    // via cpu.cls_of_box(frontend_boxes[i]).
    //
    // RPython Box.type parity: bridgeopt.py:77 uses `box.type != "r"`,
    // where `box.type` is intrinsic/immutable. Pyre reads the same
    // type that `finish()` stores in `numb_state.livebox_types` (this
    // map feeds `fail_arg_types` / `livebox_types` on the deserialize
    // side — see bridgeopt.rs below). If we queried `env.get_type()`
    // here instead, a livebox whose OptContext-side type differs from
    // its numbering-time type would cause serialize/deserialize to
    // disagree on which Ref-typed slots get a bitfield bit, producing
    // an out-of-bounds rd_numb read in `deserialize_optimizer_knowledge`
    // when super-instruction GEN widens the live register set.
    let mut bitfield: i32 = 0;
    let mut shifts = 0;
    for livebox in liveboxes {
        if let Some(opref) = livebox {
            let livebox_tp = numb_state
                .livebox_types
                .get(&opref.raw())
                .copied()
                .unwrap_or_else(|| env.get_type(*opref));
            if livebox_tp != majit_ir::Type::Ref {
                continue;
            }
            bitfield <<= 1;
            // bridgeopt.py:79-80: info = getptrinfo(box)
            // known_class = info is not None and info.get_known_class(cpu) is not None
            if env.has_known_class(*opref) {
                bitfield |= 1;
            }
            shifts += 1;
            if shifts == 6 {
                numb_state.append_int(bitfield);
                bitfield = 0;
                shifts = 0;
            }
        }
    }
    if shifts > 0 {
        numb_state.append_int(bitfield << (6 - shifts));
    }

    // bridgeopt.py:92-122: heap knowledge
    let Some(knowledge) = optimizer_knowledge else {
        // bridgeopt.py:109-111,121-122: no optheap/optrewrite → zeros
        numb_state.append_int(0); // struct fields count
        numb_state.append_int(0); // array items count
        numb_state.append_int(0); // loopinvariant count
        return;
    };
    // bridgeopt.py:93: triples_struct = optimizer.optheap.serialize_optheap(available_boxes)
    let filtered_fields: Vec<(OpRef, i32, OpRef)> = knowledge
        .heap_fields
        .iter()
        .copied()
        .filter(|&(obj, _, val)| {
            let obj_ok = env.is_const(obj) || available_boxes.contains_key(&obj);
            let val_ok = env.is_const(val) || available_boxes.contains_key(&val);
            obj_ok && val_ok
        })
        .collect();
    numb_state.append_int(filtered_fields.len() as i32);
    for (obj, descr_idx, val) in &filtered_fields {
        let obj_tag = memo._gettagged(*obj, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(obj_tag as i32);
        numb_state.append_int(*descr_idx);
        let val_tag = memo._gettagged(*val, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(val_tag as i32);
    }
    // bridgeopt.py:102-108: array items
    let filtered_arrayitems: Vec<(OpRef, i64, i32, OpRef)> = knowledge
        .heap_arrayitems
        .iter()
        .copied()
        .filter(|&(obj, _, _, val)| {
            let obj_ok = env.is_const(obj) || available_boxes.contains_key(&obj);
            let val_ok = env.is_const(val) || available_boxes.contains_key(&val);
            obj_ok && val_ok
        })
        .collect();
    numb_state.append_int(filtered_arrayitems.len() as i32);
    for (obj, index, descr_idx, val) in &filtered_arrayitems {
        let obj_tag = memo._gettagged(*obj, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(obj_tag as i32);
        numb_state.append_int(*index as i32);
        numb_state.append_int(*descr_idx);
        let val_tag = memo._gettagged(*val, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(val_tag as i32);
    }

    // bridgeopt.py:113-122: loopinvariant results
    let filtered_loopinvariant: Vec<(i64, OpRef)> = knowledge
        .loopinvariant_results
        .iter()
        .copied()
        .filter(|&(_, result)| env.is_const(result) || available_boxes.contains_key(&result))
        .collect();
    numb_state.append_int(filtered_loopinvariant.len() as i32);
    for (const_ptr, result) in &filtered_loopinvariant {
        let const_tag = memo.getconst_int(*const_ptr);
        numb_state.writer.append_short(const_tag as i32);
        let result_tag = memo._gettagged(*result, env, &numb_state.liveboxes, new_liveboxes);
        numb_state.writer.append_short(result_tag as i32);
    }
}

/// `frontend_boxes`: runtime values from guard failure (RPython Box objects
///   with concrete references). Used by cls_of_box to read vtable.
/// `cls_of_box`: model.py:199-201 cpu.cls_of_box(box) — reads typeptr from
///   a runtime Ref object. Returns the class pointer as i64.
pub fn deserialize_optimizer_knowledge(
    rd_numb: &[u8],
    rd_consts: &[majit_ir::Const],
    frontend_boxes: &[i64],
    liveboxes: &[OpRef],
    livebox_types: &[majit_ir::Type],
    all_descrs: &[majit_ir::descr::DescrRef],
    cls_of_box: Option<fn(i64) -> i64>,
    optimizer: &mut super::optimizer::Optimizer,
    ctx: &mut OptContext,
) {
    use crate::resume::{DecodedBox, decode_box};
    use majit_ir::resumecode::Reader;

    let mut reader = Reader::new(rd_numb);
    // bridgeopt.py:126: assert len(frontend_boxes) == len(liveboxes)
    assert!(
        frontend_boxes.len() == liveboxes.len(),
        "frontend_boxes.len()={} != liveboxes.len()={}",
        frontend_boxes.len(),
        liveboxes.len(),
    );

    // bridgeopt.py:130-131: skip resume section
    let startcount = reader.next_item();
    reader.jump((startcount - 1) as usize);

    // bridgeopt.py:133-146: class knowledge
    let mut bitfield: i32 = 0;
    let mut mask: i32 = 0;
    for (i, &livebox) in liveboxes.iter().enumerate() {
        // bridgeopt.py:135 reads `box.type` (intrinsic on the Box).
        // pyre's parallel side table must cover `liveboxes`.
        let tp = livebox_types.get(i).copied().unwrap_or_else(|| {
            panic!(
                "missing livebox_types[{}] (liveboxes.len()={}): \
                 RPython bridgeopt.py:135 reads box.type intrinsically; \
                 pyre's parallel array must match liveboxes length",
                i,
                liveboxes.len()
            )
        });
        if tp != majit_ir::Type::Ref {
            continue;
        }
        if mask == 0 {
            bitfield = reader.next_item();
            mask = 0b100000;
        }
        let class_known = (bitfield & mask) != 0;
        mask >>= 1;
        if class_known {
            // bridgeopt.py:145-146:
            //   cls = optimizer.cpu.cls_of_box(frontend_boxes[i])
            //   optimizer.make_constant_class(box, cls)
            // RPython's type system guarantees frontend_boxes[i] is a valid
            // GcRef when box.type == "r" and class_known is set. Our raw i64
            // encoding requires a nonnull check (RPython's box.nonnull()
            // equivalent, info.py:763).
            if let Some(cls_fn) = cls_of_box {
                let raw_ref = frontend_boxes[i];
                if raw_ref != 0 {
                    let cls = cls_fn(raw_ref);
                    super::optimizer::Optimizer::make_constant_class(ctx, livebox, cls, true);
                }
            }
        }
    }

    // bridgeopt.py:148-158: heap knowledge (struct fields)
    let length = reader.next_item();
    let mut result_struct = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:155: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_struct.push((opref1, descr.clone(), opref2));
    }
    // bridgeopt.py:159-169: heap knowledge (array items)
    let length = reader.next_item();
    let mut result_array = Vec::new();
    for _ in 0..length {
        let tagged = reader.next_item() as i16;
        let box1 = decode_box(tagged, rd_consts, liveboxes);
        let index = reader.next_item() as i64;
        let descr_index = reader.next_item();
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        // bridgeopt.py:166: descr = metainterp_sd.all_descrs[descr_index]
        let descr = &all_descrs[descr_index as usize];
        let opref1 = decoded_box_to_opref(&box1, ctx);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        result_array.push((opref1, index, descr.clone(), opref2));
    }
    // bridgeopt.py:170-171: optimizer.optheap.deserialize_optheap(...)
    if !result_struct.is_empty() || !result_array.is_empty() {
        optimizer.import_heap_knowledge(&result_struct, &result_array, ctx);
    }

    // bridgeopt.py:173-185: call_loopinvariant knowledge
    let length = reader.next_item();
    let mut result_loopinvariant = Vec::new();
    for _ in 0..length {
        let tagged1 = reader.next_item() as i16;
        let const_box = decode_box(tagged1, rd_consts, liveboxes);
        // bridgeopt.py:179-180: assert isinstance(const, ConstInt); i = const.getint()
        let DecodedBox::Const(majit_ir::Const::Int(const_int)) = const_box else {
            panic!(
                "bridgeopt: loopinvariant entry must be ConstInt, got {:?}",
                const_box
            );
        };
        let tagged2 = reader.next_item() as i16;
        let box2 = decode_box(tagged2, rd_consts, liveboxes);
        let opref2 = decoded_box_to_opref(&box2, ctx);
        // bridgeopt.py:183: result_loopinvariant.append((i, box))
        // No sentinel check — ConstInt(0) is a valid func_ptr value.
        result_loopinvariant.push((const_int, opref2));
    }
    // bridgeopt.py:184-185: optimizer.optrewrite.deserialize_optrewrite(...)
    if !result_loopinvariant.is_empty() {
        optimizer.import_loopinvariant_knowledge(&result_loopinvariant);
    }
}

/// Convert a DecodedBox to an OpRef for the bridge optimizer context.
///
/// RPython's deserialize path passes Const/Box objects directly. In majit,
/// constants must be registered in the optimizer's context to get an OpRef.
fn decoded_box_to_opref(decoded: &crate::resume::DecodedBox, ctx: &mut OptContext) -> OpRef {
    use crate::resume::DecodedBox;
    use majit_ir::Const;
    match decoded {
        DecodedBox::LiveBox(opref) => *opref,
        DecodedBox::Const(Const::Int(v)) => ctx.make_constant_int(*v),
        DecodedBox::Const(Const::Ref(r)) => ctx.make_constant_ref(*r),
        DecodedBox::Const(Const::Float(f)) => ctx.make_constant_float(*f),
    }
}
