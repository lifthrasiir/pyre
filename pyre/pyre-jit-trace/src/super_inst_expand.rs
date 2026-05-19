//! Decompose CPython 3.13 super-instructions at the JIT reader's
//! input boundary.
//!
//! RPython's register-machine jitcode never carries CPython super-
//! instructions — the JIT sees only plain ops. The pyre equivalent is
//! to expand every fast-family super-instruction into its plain
//! `LOAD_FAST` / `STORE_FAST` components at the point where the
//! codewriter and liveness analysis read bytecode, so that every
//! downstream pass observes a uniform "one local op per unit" stream.
//!
//! The original `py_pc` is preserved across expanded ops: this
//! decomposition is an internal reader convention, not a rewrite of
//! `CodeObject.instructions`. Jump targets, the exception table, and
//! `pc_map` continue to reference the original opcode PC.
//!
//! # Return shape
//!
//! `expand_fast_op` returns `Some(([FastOp; 2], len))` when the
//! instruction is a fast-family op (plain or super). The caller reads
//! only `ops[..len]` — either one entry for plain ops or two for
//! super-instructions. Non-fast instructions return `None` so the
//! caller falls through to its existing dispatch.

use pyre_interpreter::bytecode::{Instruction, OpArg};

/// One expanded fast-family operation. Matches the shape of the CPython
/// 3.13 primitives after super-inst decomposition.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FastOp {
    /// `LOAD_FAST` / `LOAD_FAST_BORROW`.
    Load { local: u16, borrow: bool },
    /// `STORE_FAST`.
    Store { local: u16 },
}

impl FastOp {
    #[inline]
    pub fn local(&self) -> u16 {
        match self {
            FastOp::Load { local, .. } | FastOp::Store { local } => *local,
        }
    }
}

/// Expand a fast-family opcode (plain or super) into its plain
/// components. Returns `None` for non-fast instructions.
///
/// # Coverage
///
/// Plain: `LOAD_FAST`, `LOAD_FAST_BORROW`, `STORE_FAST`.
///
/// Super: `LOAD_FAST_LOAD_FAST`, `LOAD_FAST_BORROW_LOAD_FAST_BORROW`,
/// `STORE_FAST_LOAD_FAST`, `STORE_FAST_STORE_FAST`.
///
/// # Non-coverage
///
/// `LOAD_FAST_CHECK`, `LOAD_FAST_AND_CLEAR`, `DELETE_FAST` stay in
/// their own dispatch arms — they do not appear in super-instructions
/// and carry extra semantics the helpers here do not express.
pub fn expand_fast_op(instr: &Instruction, op_arg: OpArg) -> Option<([FastOp; 2], usize)> {
    match instr {
        Instruction::LoadFast { var_num } => {
            let local = var_num.get(op_arg).as_usize() as u16;
            let load = FastOp::Load {
                local,
                borrow: false,
            };
            Some(([load, load], 1))
        }
        Instruction::LoadFastBorrow { var_num } => {
            let local = var_num.get(op_arg).as_usize() as u16;
            let load = FastOp::Load {
                local,
                borrow: true,
            };
            Some(([load, load], 1))
        }
        Instruction::StoreFast { var_num } => {
            let local = var_num.get(op_arg).as_usize() as u16;
            let store = FastOp::Store { local };
            Some(([store, store], 1))
        }
        Instruction::LoadFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            Some((
                [
                    FastOp::Load {
                        local: u32::from(pair.idx_1()) as u16,
                        borrow: false,
                    },
                    FastOp::Load {
                        local: u32::from(pair.idx_2()) as u16,
                        borrow: false,
                    },
                ],
                2,
            ))
        }
        Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
            let pair = var_nums.get(op_arg);
            Some((
                [
                    FastOp::Load {
                        local: u32::from(pair.idx_1()) as u16,
                        borrow: true,
                    },
                    FastOp::Load {
                        local: u32::from(pair.idx_2()) as u16,
                        borrow: true,
                    },
                ],
                2,
            ))
        }
        Instruction::StoreFastLoadFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            Some((
                [
                    FastOp::Store {
                        local: u32::from(pair.idx_1()) as u16,
                    },
                    FastOp::Load {
                        local: u32::from(pair.idx_2()) as u16,
                        borrow: false,
                    },
                ],
                2,
            ))
        }
        Instruction::StoreFastStoreFast { var_nums } => {
            let pair = var_nums.get(op_arg);
            Some((
                [
                    FastOp::Store {
                        local: u32::from(pair.idx_1()) as u16,
                    },
                    FastOp::Store {
                        local: u32::from(pair.idx_2()) as u16,
                    },
                ],
                2,
            ))
        }
        _ => None,
    }
}

// Unit tests for `expand_fast_op` would need to construct
// `Instruction::LoadFastLoadFast { var_nums }` values whose
// `var_nums` field has an opaque auto-generated type. The shape
// guarantee — that every super-instruction flattens to two plain
// primitives in the documented order — is validated end-to-end by
// the codewriter and liveness lockstep tests plus the `check.py`
// benchmark suite (`nbody.py`, `fannkuch.py`).
