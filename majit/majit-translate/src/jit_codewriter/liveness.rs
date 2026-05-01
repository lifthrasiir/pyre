//! Liveness computation for flattened JitCode instructions.
//!
//! RPython equivalent: `rpython/jit/codewriter/liveness.py`.
//!
//! Expands `-live-` markers in the flattened instruction sequence to
//! include all values that are alive at that point (written before and
//! read afterwards). This information is used by guard operations in the
//! meta-interpreter to know which values to save on failure.
//!
//! The algorithm is a backward dataflow analysis that iterates to fixpoint.

use std::collections::{HashMap, HashSet};

use crate::flatten::{FlatOp, Label, SSARepr};
use crate::model::ValueId;

/// Compute liveness for a flattened function.
///
/// RPython: `liveness.py::compute_liveness(ssarepr)`.
///
/// Modifies the flattened ops in place: each `FlatOp::Live` marker
/// gets its `live_values` set populated with all values alive at that
/// point in the instruction sequence.
/// RPython liveness.py:19-23.
pub fn compute_liveness(flattened: &mut SSARepr) {
    let mut label2alive: HashMap<Label, HashSet<ValueId>> = HashMap::new();

    // Iterate to fixpoint (RPython: while _compute_liveness_must_continue)
    loop {
        if !compute_liveness_pass(&mut flattened.insns, &mut label2alive) {
            break;
        }
    }
    // RPython liveness.py:23: remove_repeated_live(ssarepr)
    remove_repeated_live(&mut flattened.insns);
}

/// RPython liveness.py:82-116: remove_repeated_live.
///
/// Merges consecutive `-live-` markers into a single one (union of
/// all live values). Labels between them are preserved.
fn remove_repeated_live(ops: &mut Vec<FlatOp>) {
    let mut result: Vec<FlatOp> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if !matches!(&ops[i], FlatOp::Live { .. }) {
            result.push(ops[i].clone());
            i += 1;
            continue;
        }
        // Collect consecutive Live + Label runs
        let mut labels = Vec::new();
        let mut merged_live: HashSet<ValueId> = HashSet::new();
        while i < ops.len() {
            match &ops[i] {
                FlatOp::Live { live_values } => {
                    merged_live.extend(live_values.iter());
                    i += 1;
                }
                FlatOp::Label(_) => {
                    labels.push(ops[i].clone());
                    i += 1;
                }
                _ => break,
            }
        }
        // Emit labels first, then the merged -live-
        result.extend(labels);
        let mut merged: Vec<ValueId> = merged_live.into_iter().collect();
        merged.sort_by_key(|v| v.0);
        result.push(FlatOp::Live {
            live_values: merged,
        });
    }
    *ops = result;
}

/// One backward pass of liveness analysis.
/// Returns true if any label's alive set grew (needs another iteration).
///
/// RPython: `_compute_liveness_must_continue(ssarepr, label2alive)`.
///
/// Walks backward through the instruction sequence. At each `-live-`
/// marker, expands it to include all values alive at that point.
fn compute_liveness_pass(
    ops: &mut [FlatOp],
    label2alive: &mut HashMap<Label, HashSet<ValueId>>,
) -> bool {
    let mut alive: HashSet<ValueId> = HashSet::new();
    let mut must_continue = false;

    // Walk backward through instructions
    for i in (0..ops.len()).rev() {
        match &ops[i] {
            FlatOp::Label(label) => {
                let label = *label;
                let alive_at_point = label2alive.entry(label).or_default();
                let prev_len = alive_at_point.len();
                alive_at_point.extend(alive.iter());
                if alive_at_point.len() != prev_len {
                    must_continue = true;
                }
            }
            FlatOp::Live { live_values } => {
                // RPython liveness.py:44-52: -live- markers are expanded
                // to include all values currently alive at this point.
                // Also union any explicitly-forced values from jtransform.
                for v in live_values {
                    alive.insert(*v);
                }
                // Expand: replace this Live marker with all alive values.
                ops[i] = FlatOp::Live {
                    live_values: alive.iter().copied().collect(),
                };
            }
            FlatOp::EndOfBlock => {
                // RPython `liveness.py:55-57`: `'---'` resets the alive set.
                alive.clear();
            }
            FlatOp::Unreachable => {
                // RPython `unreachable` is just an opcode (no operands)
                // and goes through `liveness.py:59+`'s args-loop with
                // an empty arg list, so the alive set is unchanged.
                // The trailing `---` (always emitted right after) is
                // what resets it.
            }
            FlatOp::Op(inner_op) => {
                // Result is defined here — remove from alive
                if let Some(result) = inner_op.result {
                    alive.remove(&result);
                }
                // Operands are used here — add to alive
                for vid in crate::inline::op_value_refs(&inner_op.kind) {
                    alive.insert(vid);
                }
            }
            FlatOp::Jump(label) => {
                let label = *label;
                if let Some(alive_at_target) = label2alive.get(&label) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::CatchException { target } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfExceptionMismatch { target, .. } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfNot { cond, target } => {
                let cond = *cond;
                let target = *target;
                alive.insert(cond);
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Switch { value, targets } => {
                alive.insert(*value);
                for (_, target) in targets {
                    if let Some(alive_at_target) = label2alive.get(target) {
                        alive.extend(alive_at_target.iter());
                    }
                }
            }
            FlatOp::IntBinOpJumpIfOvf {
                target,
                lhs,
                rhs,
                dst,
                ..
            } => {
                alive.remove(dst);
                alive.insert(*lhs);
                alive.insert(*rhs);
                if let Some(alive_at_target) = label2alive.get(target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Move { dst, src } => {
                // `src` may be a Variable or a Constant (upstream
                // `getcolor` at flatten.py:382-384 returns the Constant
                // as-is); Constants contribute no live range.
                alive.remove(dst);
                if let Some(v) = src.as_value() {
                    alive.insert(v);
                }
            }
            FlatOp::Push(src) => {
                // RPython `flatten.py:329` `%s_push` — reads `v` into
                // tmpreg. Backwards: treat as a pure use of `src`.
                //
                // Only Variables reach this arm: upstream
                // `reorder_renaming_list` at `flatten.py:395-414` only
                // triggers a push when no progress is possible, which
                // requires every `to[i]` Register to appear in `frm`;
                // Constants are structurally distinct from Registers
                // (Python `==`), so a Constant `frm[i]` is always
                // consumed in a non-cycle step and never lands in the
                // cycle-break save slot.
                alive.insert(*src);
            }
            FlatOp::Pop(dst) => {
                // RPython `flatten.py:331` `%s_pop` — writes tmpreg
                // into `w`. Backwards: treat as a pure def of `dst`.
                alive.remove(dst);
            }
            FlatOp::LastException { dst } | FlatOp::LastExcValue { dst } => {
                alive.remove(dst);
            }
            FlatOp::Reraise => {}
            FlatOp::IntReturn(v) | FlatOp::RefReturn(v) | FlatOp::FloatReturn(v) => {
                // RPython blackhole `bhimpl_*_return(a)` reads `a` and
                // leaves the frame.  Backward walk: the return value is
                // alive at this point; after it (forward) nothing is.
                alive.clear();
                if let Some(value) = v.as_value() {
                    alive.insert(value);
                }
            }
            FlatOp::VoidReturn => {
                // `bhimpl_void_return()` has no args.  Nothing alive
                // after the return.
                alive.clear();
            }
            FlatOp::Raise(v) => {
                // `bhimpl_raise(excvalue)` reads the evalue and raises.
                alive.clear();
                if let Some(value) = v.as_value() {
                    alive.insert(value);
                }
            }
        }
    }

    must_continue
}

// ____________________________________________________________
// helper functions for compactly encoding and decoding liveness info
//
// liveness is encoded as a 2 byte offset into the single string all_liveness
// (which is stored on the metainterp_sd)

/// RPython liveness.py:125 `OFFSET_SIZE`.
pub const OFFSET_SIZE: usize = 2;

/// RPython liveness.py:127-131 `encode_offset(pos, code)`.
pub fn encode_offset(pos: usize, code: &mut Vec<u8>) {
    assert_eq!(OFFSET_SIZE, 2);
    code.push((pos & 0xff) as u8);
    code.push(((pos >> 8) & 0xff) as u8);
    assert_eq!(pos >> 16, 0);
}

/// RPython liveness.py:133-136 `decode_offset(jitcode, pc)`.
pub fn decode_offset(jitcode: &[u8], pc: usize) -> usize {
    assert_eq!(OFFSET_SIZE, 2);
    (jitcode[pc] as usize) | ((jitcode[pc + 1] as usize) << 8)
}

// within the string of all_liveness, we encode the bitsets of which of the 256
// registers are live as follows: first three byte with the number of set bits
// for each of the categories ints, refs, floats followed by the necessary
// number of bytes to store them (this number of bytes is implicit), for each of
// the categories
// | len live_i | len live_r | len live_f
// | bytes for live_i | bytes for live_r | bytes for live_f

/// RPython liveness.py:147-166 `encode_liveness(live)`.
///
/// Encodes a single register-kind bitset: `live` is a list of register
/// indices (each `< 256`). Returns the packed bitset bytes (no length
/// header — the caller is responsible for emitting the three
/// `len_i/len_r/len_f` header bytes).
///
/// Mirrors RPython's `live = sorted(live)` (liveness.py:148): the input
/// is sorted internally so callers can pass arbitrary-order or
/// duplicated slices without normalization.
pub fn encode_liveness(live: &[u8]) -> Vec<u8> {
    // RPython liveness.py:148 `live = sorted(live)`.
    let mut sorted: Vec<u8> = live.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut liveness: Vec<u8> = Vec::new();
    let mut offset: u32 = 0;
    let mut char_: u32 = 0;
    let mut i = 0;
    while i < sorted.len() {
        let x = sorted[i] as u32;
        let x = x.wrapping_sub(offset);
        if x >= 8 {
            liveness.push(char_ as u8);
            char_ = 0;
            offset += 8;
            continue;
        }
        char_ |= 1 << x;
        assert!(char_ < 256);
        i += 1;
    }
    if char_ != 0 {
        liveness.push(char_ as u8);
    }
    liveness
}

/// RPython liveness.py:170-200 `LivenessIterator`.
///
/// Iterates set bit positions from a bitset stored in `all_liveness`
/// starting at `offset`, producing `length` indices total.
#[derive(Debug, Clone)]
pub struct LivenessIterator<'a> {
    pub all_liveness: &'a [u8],
    pub offset: usize,
    pub length: u32,
    pub curr_byte: u32,
    pub count: u32,
}

impl<'a> LivenessIterator<'a> {
    /// RPython liveness.py:172-178 `__init__(self, offset, length, all_liveness)`.
    pub fn new(offset: usize, length: u32, all_liveness: &'a [u8]) -> Self {
        assert!(length != 0);
        LivenessIterator {
            all_liveness,
            offset,
            length,
            curr_byte: 0,
            count: 0,
        }
    }
}

impl<'a> Iterator for LivenessIterator<'a> {
    type Item = u32;

    /// RPython liveness.py:184-200 `next(self)`.
    fn next(&mut self) -> Option<u32> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        let mut count = self.count;
        let all_liveness = self.all_liveness;
        let mut curr_byte = self.curr_byte;
        // find next bit set
        loop {
            if (count & 7) == 0 {
                curr_byte = all_liveness[self.offset] as u32;
                self.curr_byte = curr_byte;
                self.offset += 1;
            }
            if (curr_byte >> (count & 7)) & 1 != 0 {
                self.count = count + 1;
                return Some(count);
            }
            count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::FlatOp;
    use crate::model::{OpKind, SpaceOperation, ValueType};

    #[test]
    fn basic_liveness() {
        // v0 = Input
        // v1 = ConstInt(42)
        // v2 = BinOp(v0, v1)
        // Return v2
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![
                FlatOp::Label(Label(0)),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(0)),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                    },
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(1)),
                    kind: OpKind::ConstInt(42),
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(ValueId(2)),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: ValueId(0),
                        rhs: ValueId(1),
                        result_ty: ValueType::Int,
                    },
                }),
            ],
            num_values: 3,
            num_blocks: 1,
            value_kinds: std::collections::HashMap::new(),
            insns_pos: None,
        };

        // Should not panic
        compute_liveness(&mut flat);
    }

    #[test]
    fn encode_decode_offset_roundtrip() {
        let mut code: Vec<u8> = Vec::new();
        encode_offset(0x1234, &mut code);
        assert_eq!(code, vec![0x34, 0x12]);
        assert_eq!(decode_offset(&code, 0), 0x1234);
    }

    #[test]
    fn encode_liveness_empty() {
        assert_eq!(encode_liveness(&[]), Vec::<u8>::new());
    }

    #[test]
    fn encode_liveness_small() {
        // live = [0, 1, 7] -> first byte has bits 0, 1, 7 set = 0b1000_0011 = 0x83
        assert_eq!(encode_liveness(&[0u8, 1, 7]), vec![0x83]);
    }

    #[test]
    fn encode_liveness_multi_byte() {
        // live = [0, 8, 15] -> byte 0 = 0x01, byte 1 = 0b1000_0001 = 0x81
        assert_eq!(encode_liveness(&[0u8, 8, 15]), vec![0x01, 0x81]);
    }

    #[test]
    fn liveness_iterator_roundtrip() {
        let live = [0u8, 3, 5, 9, 12, 17];
        let encoded = encode_liveness(&live);
        let mut it = LivenessIterator::new(0, live.len() as u32, &encoded);
        let decoded: Vec<u32> = (&mut it).collect();
        assert_eq!(decoded, live.iter().map(|&i| i as u32).collect::<Vec<_>>());
    }
}
