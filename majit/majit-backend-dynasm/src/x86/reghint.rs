//! Port of `rpython/jit/backend/x86/reghint.py`.
//!
//! Register hints that tell the regalloc which registers to prefer
//! for specific opcodes.  Runs as a pre-pass before the main regalloc
//! walk (invoked from `RegAlloc::_prepare`).

use crate::regalloc::LifetimeManager;
use crate::regloc::{EAX, ECX, EDX, RegLoc};
use crate::x86::callbuilder::{ARGUMENTS_GPR, ARGUMENTS_XMM};
use majit_ir::{Op, OpCode, OpRef, Type};
use std::collections::HashMap;

/// regalloc.py:26-28.
pub const SAVE_DEFAULT_REGS: u8 = 0;
pub const SAVE_GCREF_REGS: u8 = 1;
pub const SAVE_ALL_REGS: u8 = 2;

/// reghint.py:29 X86RegisterHints.
pub struct RegisterHints {
    save_around_call_regs_gpr: Vec<RegLoc>,
    all_regs_gpr: Vec<RegLoc>,
    all_regs_xmm: Vec<RegLoc>,
    /// Rust adaptation: RPython's `ConstInt.value` is embedded on the
    /// Box object; pyre's `OpRef` is a flat `u32` so constant values
    /// are looked up through this map.  Structural equivalent of
    /// `isinstance(arg, ConstInt) and arg.value`.
    constants: HashMap<u32, i64>,
}

impl RegisterHints {
    pub fn new(
        save_around_call_regs_gpr: Vec<RegLoc>,
        _save_around_call_regs_xmm: Vec<RegLoc>,
        all_regs_gpr: Vec<RegLoc>,
        all_regs_xmm: Vec<RegLoc>,
        constants: HashMap<u32, i64>,
    ) -> Self {
        RegisterHints {
            save_around_call_regs_gpr,
            all_regs_gpr,
            all_regs_xmm,
            constants,
        }
    }

    /// Rust equivalent of `isinstance(arg, ConstInt) and arg.value`
    /// — returns the const int value if available.
    fn get_const_int(&self, arg: OpRef) -> Option<i64> {
        if !arg.is_constant() {
            return None;
        }
        self.constants.get(&arg.raw()).copied()
    }

    /// reghint.py:30 `add_hints` — main entry called from
    /// `RegAlloc::_prepare`.
    pub fn add_hints(
        &self,
        longevity: &mut LifetimeManager,
        _inputargs: &[majit_ir::InputArg],
        operations: &[Op],
    ) {
        for (i, op) in operations.iter().enumerate() {
            let position = i as i32;
            // reghint.py:34-35 skip dead no-side-effect ops.
            if op.opcode.has_no_side_effect() && !longevity.contains(op.pos) {
                continue;
            }
            self.dispatch(op.opcode, longevity, op, position);
        }
    }

    /// reghint.py:166-172 `oplist` dispatch.  Implemented as a match
    /// here because Rust cannot build a class-method oplist at
    /// import time the way Python can.
    fn dispatch(&self, opcode: OpCode, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        match opcode {
            // reghint.py:43-45
            OpCode::IntNeg | OpCode::IntInvert => {
                self.consider_int_neg(longevity, op, position);
            }
            // reghint.py:70-77 _consider_binop_symm
            OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::IntMulOvf
            | OpCode::IntAddOvf => {
                self._consider_binop_symm(longevity, op, position);
            }
            // reghint.py:76 _consider_binop
            OpCode::IntSubOvf => {
                self._consider_binop(longevity, op, position);
            }
            // reghint.py:79-86 consider_int_add + consider_nursery_ptr_increment
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                self.consider_int_add(longevity, op, position);
            }
            // reghint.py:88-93
            OpCode::IntSub => {
                self.consider_int_sub(longevity, op, position);
            }
            // reghint.py:100-105 _consider_float_op
            OpCode::FloatAdd
            | OpCode::FloatSub
            | OpCode::FloatMul
            | OpCode::FloatTrueDiv
            | OpCode::FloatNeg
            | OpCode::FloatAbs => {
                self._consider_float_op(longevity, op, position);
            }
            // reghint.py:107-115 consider_int_lshift / rshift / urshift
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                self.consider_int_lshift(longevity, op, position);
            }
            // reghint.py:117-121 consider_uint_mul_high
            OpCode::UintMulHigh => {
                self.consider_uint_mul_high(longevity, position);
            }
            // reghint.py:123-128 consider_call_malloc_nursery + varsize variants
            OpCode::CallMallocNursery
            | OpCode::CallMallocNurseryVarsize
            | OpCode::CallMallocNurseryVarsizeFrame => {
                self.consider_call_malloc_nursery(longevity, op, position);
            }
            // reghint.py:147-150 consider_call_{i,r,f,n} → _consider_real_call
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN => {
                self._consider_real_call(longevity, op, position);
            }
            // reghint.py:154-157 consider_call_may_force_*
            OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN => {
                self._consider_call(longevity, op, position, true, 1);
            }
            // reghint.py:162-164 consider_call_release_gil_*
            OpCode::CallReleaseGilI | OpCode::CallReleaseGilF | OpCode::CallReleaseGilN => {
                self._consider_call(longevity, op, position, true, 2);
            }
            _ => {}
        }
    }

    /// reghint.py:43-45 consider_int_neg / consider_int_invert.
    fn consider_int_neg(&self, longevity: &mut LifetimeManager, op: &Op, _position: i32) {
        longevity.try_use_same_register(op.args[0], op.pos);
    }

    /// reghint.py:47-62 `_consider_binop_part`.
    fn _consider_binop_part(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
        symm: bool,
    ) {
        let mut x = op.args[0];
        let mut y = op.args[1];

        // For symmetrical operations, if y won't be used after the
        // current operation finishes, but x will be, then swap the
        // role of 'x' and 'y'.
        if symm {
            if x.is_constant() {
                std::mem::swap(&mut x, &mut y);
            } else if !y.is_constant() {
                let x_last = longevity.get(x).map(|lt| lt.last_usage).unwrap_or(-1);
                let y_last = longevity.get(y).map(|lt| lt.last_usage).unwrap_or(-1);
                if x_last > position && y_last == position {
                    std::mem::swap(&mut x, &mut y);
                }
            }
        }

        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos);
        }
    }

    /// reghint.py:64-65 `_consider_binop`.
    fn _consider_binop(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        self._consider_binop_part(longevity, op, position, false);
    }

    /// reghint.py:67-68 `_consider_binop_symm`.
    fn _consider_binop_symm(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        self._consider_binop_part(longevity, op, position, true);
    }

    /// reghint.py:79-84 `consider_int_add`.
    fn consider_int_add(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let y = op.args[1];
        if let Some(val) = self.get_const_int(y) {
            if fits_in_32bits(val) {
                // nothing to be hinted
                return;
            }
        }
        self._consider_binop_symm(longevity, op, position);
    }

    /// reghint.py:88-93 `consider_int_sub`.
    fn consider_int_sub(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let y = op.args[1];
        if let Some(val) = self.get_const_int(y) {
            if fits_in_32bits(-val) {
                return;
            }
        }
        self._consider_binop(longevity, op, position);
    }

    /// reghint.py:95-98 `_consider_float_op`.
    fn _consider_float_op(&self, longevity: &mut LifetimeManager, op: &Op, _position: i32) {
        let x = op.args[0];
        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos);
        }
    }

    /// reghint.py:107-112 `consider_int_lshift` (shared with rshift /
    /// uint_rshift per aliases at reghint.py:114-115).
    ///
    /// Shift amount must go in ecx (x86 `shl`/`shr` only accept cl).
    /// Fix `ecx` for `y` at this position, and bias the result to
    /// share with `x` via try_use_same_register.
    fn consider_int_lshift(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let x = op.args[0];
        let y = op.args[1];
        if !y.is_constant() {
            longevity.fixed_register(position, ECX, Some(y));
        }
        if !x.is_constant() {
            longevity.try_use_same_register(x, op.pos);
        }
    }

    /// reghint.py:117-121 `consider_uint_mul_high`.
    ///
    /// 64x64→128 MUL on x86 returns low in rax, high in rdx.  Reserve
    /// both at this position so the result/temp aren't spilled over.
    fn consider_uint_mul_high(&self, longevity: &mut LifetimeManager, position: i32) {
        longevity.fixed_register(position, EAX, None);
        longevity.fixed_register(position, EDX, None);
    }

    /// reghint.py:123-128 `consider_call_malloc_nursery` and its
    /// varsize/varsize_frame aliases — pins result to ecx and temp
    /// to edx, matching the regalloc.py:1013-1028 contract.
    fn consider_call_malloc_nursery(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
    ) {
        longevity.fixed_register(position, ECX, Some(op.pos));
        longevity.fixed_register(position, EDX, None);
    }

    /// reghint.py:138 `_consider_real_call`.
    fn _consider_real_call(&self, longevity: &mut LifetimeManager, op: &Op, position: i32) {
        let Some(descr) = op.descr.as_ref() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        // reghint.py:142-144 skip oopspec helpers.
        if calldescr.get_extra_info().has_oopspec() {
            return;
        }
        self._consider_call(longevity, op, position, false, 1);
    }

    /// reghint.py:130 `_consider_call`.
    fn _consider_call(
        &self,
        longevity: &mut LifetimeManager,
        op: &Op,
        position: i32,
        guard_not_forced: bool,
        first_arg_index: usize,
    ) {
        let Some(descr) = op.descr.as_ref() else {
            return;
        };
        let Some(calldescr) = descr.as_call_descr() else {
            return;
        };
        let gc_level = compute_gc_level(calldescr, guard_not_forced);
        let args = &op.args[first_arg_index..];
        let argtypes = calldescr.arg_types();
        self.hint(longevity, position, args, argtypes, gc_level);
    }

    /// reghint.py:207 `CallHints64.hint`.
    fn hint(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        args: &[OpRef],
        argtypes: &[Type],
        save_all_regs: u8,
    ) {
        let mut hinted_xmm: Vec<RegLoc> = Vec::new();
        let mut hinted_gpr: Vec<RegLoc> = Vec::new();
        let mut hinted_args: Vec<OpRef> = Vec::new();
        let mut next_arg_gpr = 0usize;
        let mut next_arg_xmm = 0usize;

        for (i, &arg) in args.iter().enumerate() {
            let arg_type = argtypes.get(i).copied().unwrap_or(Type::Int);
            if arg_type == Type::Float {
                if next_arg_xmm < ARGUMENTS_XMM.len() {
                    let tgt = ARGUMENTS_XMM[next_arg_xmm];
                    if !arg.is_constant() && !hinted_args.contains(&arg) {
                        longevity.fixed_register(position, tgt, Some(arg));
                        hinted_xmm.push(tgt);
                        hinted_args.push(arg);
                    }
                    next_arg_xmm += 1;
                }
            } else if next_arg_gpr < ARGUMENTS_GPR.len() {
                let tgt = ARGUMENTS_GPR[next_arg_gpr];
                if !arg.is_constant() && !hinted_args.contains(&arg) {
                    longevity.fixed_register(position, tgt, Some(arg));
                    hinted_gpr.push(tgt);
                    hinted_args.push(arg);
                }
                next_arg_gpr += 1;
            }
        }
        self._block_non_caller_save(longevity, position, save_all_regs, &hinted_gpr, &hinted_xmm);
    }

    /// reghint.py:176 `_block_non_caller_save`.
    fn _block_non_caller_save(
        &self,
        longevity: &mut LifetimeManager,
        position: i32,
        save_all_regs: u8,
        hinted_gpr: &[RegLoc],
        hinted_xmm: &[RegLoc],
    ) {
        // reghint.py:184-188 GPR selection.
        let gpr_regs: &[RegLoc] = if save_all_regs == SAVE_ALL_REGS {
            &self.all_regs_gpr
        } else {
            &self.save_around_call_regs_gpr
        };
        for &reg in gpr_regs {
            if !hinted_gpr.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
        // reghint.py:192-194 XMM: every x86-64 XMM reg is caller-save.
        for &reg in &self.all_regs_xmm {
            if !hinted_xmm.contains(&reg) {
                longevity.fixed_register(position, reg, None);
            }
        }
    }
}

/// x86/regalloc.py:35 `compute_gc_level`.
fn compute_gc_level(calldescr: &dyn majit_ir::descr::CallDescr, guard_not_forced: bool) -> u8 {
    if guard_not_forced {
        return SAVE_ALL_REGS;
    }
    if !calldescr.get_extra_info().check_can_collect() {
        return SAVE_DEFAULT_REGS;
    }
    SAVE_GCREF_REGS
}

/// x86/rx86.py `fits_in_32bits`.  x86 ADD/SUB imm32 encoding accepts
/// signed 32-bit immediates.
fn fits_in_32bits(value: i64) -> bool {
    (i32::MIN as i64) <= value && value <= (i32::MAX as i64)
}
