/// JitCodeBuilder — bytecode assembler for JitCode construction.
///
/// RPython codewriter/assembler.py: assembler that emits bytecodes into a
/// JitCode object. This remains in metainterp only as transitional pyre ABI
/// glue until callers consume `majit_translate::assembler::Assembler`.
use std::cmp::max;

use majit_backend::JitCellToken;
use majit_ir::OpCode;

use crate::jitcode;

use super::{
    JitArgKind, JitCallArg, JitCallAssemblerTarget, JitCallTarget, JitCode, RuntimeBhDescr,
};

#[derive(Default)]
pub struct JitCodeBuilder {
    /// RPython `jitcode.py:15` `self.name = name`. Propagated to the
    /// finished `JitCode` by `finish()`; empty by default until a
    /// caller provides the source function name via `set_name`.
    name: String,
    code: Vec<u8>,
    /// RPython `jitcode.py:40` `self._startpoints = startpoints` —
    /// populated as a sibling of every opcode-byte push so
    /// `JitCode.get_live_vars_info` (RPython `jitcode.py:85-90`) can
    /// fire its non-translated `assert pc in self._startpoints` check
    /// against runtime-emitted bytecode.  Recording happens through
    /// `start_instr` / `write_insn`; every helper that pushes an opcode
    /// byte goes through one of those.
    startpoints: std::collections::HashSet<usize>,
    /// RPython `assembler.py:176` `self.alllabels.add(len(self.code))` —
    /// every TLabel emit records the bytecode offset of the 2-byte
    /// label slot so `JitCode.follow_jump` (RPython `jitcode.py:108-109`)
    /// can fire its non-translated `assert position in self._alllabels`
    /// debug check.  Populated by `push_label_ref`.
    alllabels: std::collections::HashSet<usize>,
    num_regs_i: u16,
    num_regs_r: u16,
    num_regs_f: u16,
    constants_i: Vec<i64>,
    constants_r: Vec<i64>,
    constants_f: Vec<i64>,
    labels: Vec<Option<usize>>,
    patches: Vec<(usize, usize)>,
    /// RPython `assembler.py:131-138` `emit_const` encodes a constant
    /// operand as `count_regs[kind] + len(constants) - 1` — i.e. a
    /// register-space index that points into the constants suffix of
    /// the register file. pyre's builder cannot resolve that final
    /// index at emit time (num_regs_X grows as more touch_reg calls
    /// happen); instead each const-source operand writes a placeholder
    /// 2-byte slot and records `(offset, kind, pool_idx)` here. The
    /// `finish()` pass rewrites each placeholder to
    /// `num_regs_X + pool_idx` once the per-kind register count is final.
    const_patches: Vec<(usize, ConstKind, u16)>,
    /// Same role as `const_patches`, but for 1-byte placeholder slots.
    /// Used by `loop_header` / `jit_merge_point` jdindex emission, where
    /// the operand byte must hold a single `num_regs_X + pool_idx`
    /// register-file index (upstream `@arguments("i")` —
    /// `blackhole.py:1062,1066`). `patch_const_u8_refs()` rewrites each
    /// placeholder once `num_regs_X` is final and asserts the slot fits
    /// in u8.
    const_patches_u8: Vec<(usize, ConstKind, u16)>,
    /// Runtime descriptor pool emitted into `JitCodeExecState.descrs`
    /// on `finish()`. Every `BC_INLINE_CALL` / `BC_CALL_*` /
    /// `BC_RESIDUAL_CALL_*` operand is a 2-byte index into this pool
    /// (RPython `j`/`d` argcode → `descrs[idx]` dispatch).
    descrs: Vec<RuntimeBhDescr>,
    has_abort: bool,
    /// RPython `jitcode.py:16` `self.fnaddr = fnaddr`. RPython hands
    /// `fnaddr` to `JitCode.__init__` *before* the assembler fills the
    /// body; pyre stages it here through `set_fnaddr` so `finish()` can
    /// commit it alongside the body in a single object construction step.
    fnaddr: i64,
    /// RPython `jitcode.py:17` `self.calldescr = calldescr`. RPython
    /// hands the calldescr to `JitCode.__init__` *before* the assembler
    /// fills the body, so the field is committed alongside the
    /// `setup()` body.  Callers stage the value here through
    /// `set_calldescr` so `finish()` can stamp it into the body atomically;
    /// the post-assemble `body_mut().calldescr = ...` write the previous
    /// implementation used violated the upstream order
    /// (`call.py:167-169` constructs `JitCode(name, fnaddr, calldescr)`
    /// before `assembler.assemble`).
    calldescr: majit_translate::jitcode::BhCallDescr,
    /// Whether `num_regs_{i,r,f}` have been frozen at the regalloc-
    /// final value. While set, `touch_reg`/`touch_ref_reg`/
    /// `touch_float_reg` skip the `max` update so subsequent operand
    /// emission cannot extend the register file past what regalloc
    /// computed.
    ///
    /// This is the precondition for routing residual_call `Const*`
    /// args through the constants pool: the encoded byte
    /// `num_regs_kind + pool_idx` must remain stable from the moment
    /// `add_const_*` returns through `finish()`'s constants-suffix
    /// placement (`init_register_files_from_runtime_jitcode`,
    /// `blackhole.rs:1056-1075`). RPython's assembler is naturally in
    /// this regime — `assembler.py` runs after regalloc has fixed
    /// `num_regs[kind]` — so freeze restores parity with the upstream
    /// invariant.
    num_regs_frozen: bool,
}

/// Register-file kind for const_patches entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConstKind {
    Int,
    Ref,
    Float,
}

impl JitCodeBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// RPython `jitcode.py:14-15` `__init__(name, ...)`: set the symbolic
    /// name used by `dump()` / `Display`. Callers that know the source
    /// function name should call this before `finish()`.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Stage `fnaddr` (RPython `jitcode.py:16`) so `finish()` can
    /// commit it onto the constructed `JitCode` alongside the body —
    /// matching `JitCode(name, fnaddr, calldescr)` constructor order
    /// (`call.py:167-169`).
    pub fn set_fnaddr(&mut self, fnaddr: i64) {
        self.fnaddr = fnaddr;
    }

    /// Stage the calldescr returned by `get_jitcode_calldescr`
    /// (`call.py:167`) so `finish()` can stamp it into the body alongside
    /// the bytecode/constants pool, matching the upstream
    /// `JitCode(name, fnaddr, calldescr)` constructor order.
    pub fn set_calldescr(&mut self, calldescr: majit_translate::jitcode::BhCallDescr) {
        self.calldescr = calldescr;
    }

    /// Whether `abort()` was called on this builder.
    ///
    /// pyre-specific: when true the caller should treat the resulting
    /// JitCode as non-executable and fall back to the interpreter.
    /// RPython has no equivalent because a translator that emits a bad
    /// bytecode crashes the build via AssertionError instead. Keeping
    /// this flag on the builder (not on `JitCode` itself) lets the
    /// metainterp-side `JitCode` stay aligned with RPython jitcode.py.
    pub fn has_abort_flag(&self) -> bool {
        self.has_abort
    }

    /// Pyre-specific: force-set the abort flag from an outer pipeline
    /// step (e.g. liveness overflow after `finish()` has already packed
    /// the JitCode). Updates the builder state but since finish() has
    /// consumed self, callers use this before finish() or on a new
    /// tracking variable.
    pub fn set_abort_flag(&mut self, v: bool) {
        self.has_abort = v;
    }

    /// Current bytecode emission position.
    pub fn current_pos(&self) -> usize {
        self.code.len()
    }

    pub fn add_const_i(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_i
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_i.len() as u16;
        self.constants_i.push(value);
        index
    }

    /// Add a ref constant to the constant pool. Returns pool index.
    pub fn add_const_r(&mut self, value: i64) -> u16 {
        let index = self.constants_r.len() as u16;
        self.constants_r.push(value);
        index
    }

    /// Add a float constant (bits as i64) to the constant pool. Returns pool index.
    pub fn add_const_f(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_f
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
        let index = self.constants_f.len() as u16;
        self.constants_f.push(value);
        index
    }

    /// Current num_regs_i (for computing constant register indices).
    pub fn num_regs_i(&self) -> u16 {
        self.num_regs_i
    }

    /// Current num_regs_r (for computing constant register indices).
    pub fn num_regs_r(&self) -> u16 {
        self.num_regs_r
    }

    /// Current num_regs_f. Callers that need the full per-kind register
    /// ceiling (e.g. `SSAReprEmitter::finish_with`) read all three.
    pub fn num_regs_f(&self) -> u16 {
        self.num_regs_f
    }

    /// `assembler.py` parity: size of the int constant pool (used by
    /// assemble-time bounds checks that allow constant virtual-register
    /// indices `num_regs_i .. num_regs_i + num_consts_i()`).
    pub fn num_consts_i(&self) -> u16 {
        self.constants_i.len() as u16
    }

    /// Same as `num_consts_i` for the ref constant pool.
    pub fn num_consts_r(&self) -> u16 {
        self.constants_r.len() as u16
    }

    /// Same as `num_consts_i` for the float constant pool.
    pub fn num_consts_f(&self) -> u16 {
        self.constants_f.len() as u16
    }

    pub fn load_const_i_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_i(value);
        self.load_const_i(dst, const_idx);
    }

    /// Lower to RPython `int_copy/i>i` reading from the constants
    /// window of the register file (`assembler.py:80-138` `emit_const`
    /// produces the same byte sequence — a register-space index that
    /// points into the post-regs constants suffix). The src operand
    /// is written as a placeholder; `finish()` patches it to
    /// `num_regs_i + const_idx` once the per-kind register count is
    /// final.
    pub fn load_const_i(&mut self, dst: u16, const_idx: u16) {
        self.touch_reg(dst);
        self.write_insn("int_copy/i>i");
        let src_offset = self.code.len();
        self.push_u16(0);
        self.const_patches
            .push((src_offset, ConstKind::Int, const_idx));
        self.push_u16(dst);
    }

    // ── State field access (register/tape machines) ──

    /// Load a scalar state field value into an int register.
    pub fn load_state_field(&mut self, field_idx: u16, dest: u16) {
        self.touch_reg(dest);
        self.write_insn("load_state_field/di");
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    /// Store an int register value into a scalar state field.
    pub fn store_state_field(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.write_insn("store_state_field/di");
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    /// Load an array state field element into an int register.
    /// The element index comes from another int register.
    pub fn load_state_array(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.write_insn("load_state_array/dii");
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store an int register value into an array state field element.
    /// The element index comes from another int register.
    pub fn store_state_array(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.write_insn("store_state_array/dii");
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    // ── First-class virtualizable access (getfield_vable_*) ──

    pub fn vable_getfield_int(&mut self, dest: u16, field_idx: u16) {
        self.touch_reg(dest);
        self.start_instr(jitcode::BC_GETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_ref(&mut self, dest: u16, field_idx: u16) {
        self.touch_ref_reg(dest);
        self.start_instr(jitcode::BC_GETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_getfield_float(&mut self, dest: u16, field_idx: u16) {
        self.touch_float_reg(dest);
        self.start_instr(jitcode::BC_GETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(dest);
    }

    pub fn vable_setfield_int(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.start_instr(jitcode::BC_SETFIELD_VABLE_I);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_ref(&mut self, field_idx: u16, src: u16) {
        self.touch_ref_reg(src);
        self.start_instr(jitcode::BC_SETFIELD_VABLE_R);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_setfield_float(&mut self, field_idx: u16, src: u16) {
        self.touch_float_reg(src);
        self.start_instr(jitcode::BC_SETFIELD_VABLE_F);
        self.push_u16(field_idx);
        self.push_u16(src);
    }

    pub fn vable_getarrayitem_int(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_reg(dest);
        self.touch_reg(index_reg);
        self.start_instr(jitcode::BC_GETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_ref(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_ref_reg(dest);
        self.touch_reg(index_reg);
        self.start_instr(jitcode::BC_GETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_getarrayitem_float(&mut self, dest: u16, array_idx: u16, index_reg: u16) {
        self.touch_float_reg(dest);
        self.touch_reg(index_reg);
        self.start_instr(jitcode::BC_GETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    pub fn vable_setarrayitem_int(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.start_instr(jitcode::BC_SETARRAYITEM_VABLE_I);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_setarrayitem_ref(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_ref_reg(src);
        self.start_instr(jitcode::BC_SETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    /// Const-source variant of `vable_setarrayitem_ref`. Lowers a
    /// `setarrayitem_vable_r(vable, idx, ConstPtr(value))` SSA op
    /// (jtransform.py:1898 produces this when the value operand is a
    /// `Const`) by reusing the same `BC_SETARRAYITEM_VABLE_R` opcode
    /// with the src u16 deferred to the const-patch table — at finish
    /// time it is patched to `num_regs_r + const_idx`, addressing into
    /// the constants suffix of the unified register space (see
    /// `Frame::copy_constants` and `init_register_file_from_i64s`,
    /// matching upstream `assembler.py:80-138 emit_const` register-
    /// space convention). No new bytecode is required: blackhole
    /// (`registers_r[src]`) and tracer (`read_ref_reg(src)`) both read
    /// the pre-populated constant slot.
    pub fn vable_setarrayitem_ref_const_value(
        &mut self,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_r(value);
        self.touch_reg(index_reg);
        self.start_instr(jitcode::BC_SETARRAYITEM_VABLE_R);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        let src_offset = self.code.len();
        self.push_u16(0);
        self.const_patches
            .push((src_offset, ConstKind::Ref, const_idx));
    }

    pub fn vable_setarrayitem_float(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_float_reg(src);
        self.start_instr(jitcode::BC_SETARRAYITEM_VABLE_F);
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    pub fn vable_arraylen(&mut self, dest: u16, array_idx: u16) {
        self.touch_reg(dest);
        self.start_instr(jitcode::BC_ARRAYLEN_VABLE);
        self.push_u16(array_idx);
        self.push_u16(dest);
    }

    pub fn vable_force(&mut self) {
        self.start_instr(jitcode::BC_HINT_FORCE_VIRTUALIZABLE);
    }

    /// Load from a virtualizable state array: emit GETARRAYITEM_RAW_I.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn load_state_varray(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.write_insn("load_state_varray/dii");
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(dest);
    }

    /// Store to a virtualizable state array: emit SETARRAYITEM_RAW.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn store_state_varray(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.write_insn("store_state_varray/dii");
        self.push_u16(array_idx);
        self.push_u16(index_reg);
        self.push_u16(src);
    }

    /// RPython `blackhole.py:459-521` `bhimpl_int_*` per-opname handlers:
    /// each primitive has its own insn_id in `BlackholeInterpBuilder.insns`
    /// (`blackhole.py:52-81 setup_insns`). Emits via `write_insn` with
    /// the canonical `opname/ii>i` key so the opcode byte comes from the
    /// shared insns table rather than a hand-assigned `BC_*` constant.
    pub fn record_binop_i(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::IntAdd => "int_add/ii>i",
            OpCode::IntSub => "int_sub/ii>i",
            OpCode::IntMul => "int_mul/ii>i",
            OpCode::IntFloorDiv => "int_floordiv/ii>i",
            OpCode::IntMod => "int_mod/ii>i",
            OpCode::IntAnd => "int_and/ii>i",
            OpCode::IntOr => "int_or/ii>i",
            OpCode::IntXor => "int_xor/ii>i",
            OpCode::IntLshift => "int_lshift/ii>i",
            OpCode::IntRshift => "int_rshift/ii>i",
            OpCode::IntEq => "int_eq/ii>i",
            OpCode::IntNe => "int_ne/ii>i",
            OpCode::IntLt => "int_lt/ii>i",
            OpCode::IntLe => "int_le/ii>i",
            OpCode::IntGt => "int_gt/ii>i",
            OpCode::IntGe => "int_ge/ii>i",
            // Unsigned integer primitives — RPython `blackhole.py:471,521,571-582`.
            OpCode::UintRshift => "uint_rshift/ii>i",
            OpCode::UintMulHigh => "uint_mul_high/ii>i",
            OpCode::UintLt => "uint_lt/ii>i",
            OpCode::UintLe => "uint_le/ii>i",
            OpCode::UintGt => "uint_gt/ii>i",
            OpCode::UintGe => "uint_ge/ii>i",
            other => panic!("record_binop_i: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_reg(lhs);
        self.touch_reg(rhs);
        self.write_insn(key);
        self.push_u16(dst);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    /// RPython `blackhole.py:527-533` `bhimpl_int_{neg,invert}` per-opname
    /// handlers. See `record_binop_i` for the keying rationale.
    pub fn record_unary_i(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let key = match opcode {
            OpCode::IntNeg => "int_neg/i>i",
            OpCode::IntInvert => "int_invert/i>i",
            other => panic!("record_unary_i: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_reg(src);
        self.write_insn(key);
        self.push_u16(dst);
        self.push_u16(src);
    }

    /// RPython `blackhole.py:584-610` ref comparisons returning int:
    /// `ptr_eq`, `ptr_ne`, `instance_ptr_eq`, `instance_ptr_ne`.
    pub fn record_binop_r(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::PtrEq => "ptr_eq/rr>i",
            OpCode::PtrNe => "ptr_ne/rr>i",
            OpCode::InstancePtrEq => "instance_ptr_eq/rr>i",
            OpCode::InstancePtrNe => "instance_ptr_ne/rr>i",
            other => panic!("record_binop_r: unsupported opcode {other:?}"),
        };
        self.touch_reg(dst);
        self.touch_ref_reg(lhs);
        self.touch_ref_reg(rhs);
        self.write_insn(key);
        self.push_u16(dst);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    /// RPython `blackhole.py:591-596` unary ptr nullity checks returning int.
    pub fn ptr_iszero(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ptr_iszero/r>i");
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn ptr_nonzero(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ptr_nonzero/r>i");
        self.push_u16(dst);
        self.push_u16(src);
    }

    pub fn new_label(&mut self) -> u16 {
        let label = self.labels.len() as u16;
        self.labels.push(None);
        label
    }

    pub fn mark_label(&mut self, label: u16) {
        let slot = self
            .labels
            .get_mut(label as usize)
            .expect("jitcode label out of bounds");
        *slot = Some(self.code.len());
    }

    /// RPython `blackhole.py:913`
    /// `bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`. Take
    /// `label` iff `reg == 0`.
    pub fn goto_if_not_int_is_true(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.write_insn("goto_if_not_int_is_true/iL");
        self.push_u16(reg);
        self.push_label_ref(label);
    }

    // jtransform.py:196 `optimize_goto_if_not` folds
    // `v = int_lt(a, b); exitswitch = v` into a single jitcode op
    // emitted by flatten.py:247-250 as `goto_if_not_int_lt/iiL`.
    // blackhole.py:864-911 semantics: take branch iff comparison is
    // false, i.e. `int_lt(a, b) == False` → `position = target`.
    pub fn goto_if_not_int_lt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_lt/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_le(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_le/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_eq/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_ne/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_gt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_gt/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_int_ge(&mut self, a: u16, b: u16, label: u16) {
        self.touch_reg(a);
        self.touch_reg(b);
        self.write_insn("goto_if_not_int_ge/iiL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    // blackhole.py:752-798 float variants — same semantics, float regs.
    pub fn goto_if_not_float_lt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_lt/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_le(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_le/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_eq/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_ne/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_gt(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_gt/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_float_ge(&mut self, a: u16, b: u16, label: u16) {
        self.touch_float_reg(a);
        self.touch_float_reg(b);
        self.write_insn("goto_if_not_float_ge/ffL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    // blackhole.py:922-936 ptr variants — ref regs.
    pub fn goto_if_not_ptr_eq(&mut self, a: u16, b: u16, label: u16) {
        self.touch_ref_reg(a);
        self.touch_ref_reg(b);
        self.write_insn("goto_if_not_ptr_eq/rrL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_ne(&mut self, a: u16, b: u16, label: u16) {
        self.touch_ref_reg(a);
        self.touch_ref_reg(b);
        self.write_insn("goto_if_not_ptr_ne/rrL");
        self.push_u16(a);
        self.push_u16(b);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_iszero(&mut self, a: u16, label: u16) {
        self.touch_ref_reg(a);
        self.write_insn("goto_if_not_ptr_iszero/rL");
        self.push_u16(a);
        self.push_label_ref(label);
    }

    pub fn goto_if_not_ptr_nonzero(&mut self, a: u16, label: u16) {
        self.touch_ref_reg(a);
        self.write_insn("goto_if_not_ptr_nonzero/rL");
        self.push_u16(a);
        self.push_label_ref(label);
    }

    // blackhole.py:916-920 `bhimpl_goto_if_not_int_is_zero(a, target, pc)`:
    // fall through iff `not a` (a == 0), else take the target. jtransform.py:1212
    // `_rewrite_equality` rewrites `int_eq(x, 0)` → `int_is_zero(x)` so
    // flatten.py:247 specialises the bool exitswitch into this unary form.
    pub fn goto_if_not_int_is_zero(&mut self, a: u16, label: u16) {
        self.touch_reg(a);
        self.write_insn("goto_if_not_int_is_zero/iL");
        self.push_u16(a);
        self.push_label_ref(label);
    }

    pub fn jump(&mut self, label: u16) {
        self.write_insn("goto/L");
        self.push_label_ref(label);
    }

    /// RPython jtransform.py:1714-1718 handle_jit_marker__loop_header emits
    /// SpaceOperation('loop_header', [c_index], None). blackhole.py:1063
    /// bhimpl_loop_header(jdindex) is a no-op; pyjitpl.py:1527
    /// opimpl_loop_header records the jitdriver index for the trace.
    ///
    /// `@arguments("i")` (blackhole.py:1062) parity: jdindex is encoded
    /// as a single register-index byte pointing into the post-regs
    /// constants suffix of `registers_i`. `add_const_i` registers the
    /// value; the placeholder is patched at `finish()` once `num_regs_i`
    /// is final.
    pub fn loop_header(&mut self, jdindex: u8) {
        self.write_insn("loop_header/i");
        let const_idx = self.add_const_i(jdindex as i64);
        let offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((offset, ConstKind::Int, const_idx));
    }

    /// RPython `assembler.py:146-158` `Register('-live-', ...)` arm in
    /// `write_insn` — emit the `live/` opcode followed by the 2-byte
    /// offset returned from `Assembler._encode_liveness`.
    ///
    /// ```python
    /// elif insn[i].is_live():
    ///     self.code.append(chr(self.insns['live/']))   # 148
    ///     live_i = self.get_liveness_info(insn[i:], 'int')
    ///     live_r = self.get_liveness_info(insn[i:], 'ref')
    ///     live_f = self.get_liveness_info(insn[i:], 'float')
    ///     self._encode_liveness(live_i, live_r, live_f)  # 158
    /// ```
    ///
    /// Mirrors `Assembler::_encode_liveness` (assembler.py:235): the
    /// cache key is built from the set-equivalent (sorted, deduplicated)
    /// view of each `live_*` slice, so callers may pass arbitrary order.
    ///
    /// Pyre keeps `live_placeholder` + `patch_live_offset` below for
    /// deferred-patch callers that must emit the LIVE op before the
    /// canonical liveness entry is known (e.g., the macro state-field
    /// jitcode build registers its canonical entry once per process,
    /// long after individual jitcode bytes are written).
    pub fn live(
        &mut self,
        asm: &mut majit_translate::jit_codewriter::assembler::Assembler,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) {
        // assembler.py:148 `self.code.append(chr(self.insns['live/']))`
        self.write_insn("live/");
        // assembler.py:158 `self._encode_liveness(live_i, live_r, live_f)`
        asm._encode_liveness(live_i, live_r, live_f, &mut self.code);
    }

    /// RPython assembler.py: emit `live/` followed by a 2-byte offset into
    /// the shared all_liveness byte string. Returns the operand offset so the
    /// caller can patch it after computing liveness.
    pub fn live_placeholder(&mut self) -> usize {
        self.write_insn("live/");
        let patch_offset = self.code.len();
        self.push_u16(0);
        patch_offset
    }

    pub fn patch_live_offset(&mut self, patch_offset: usize, offset: u16) {
        let bytes = offset.to_le_bytes();
        self.code[patch_offset] = bytes[0];
        self.code[patch_offset + 1] = bytes[1];
    }

    /// RPython blackhole.py:969 `catch_exception/L`.
    pub fn catch_exception(&mut self, label: u16) {
        self.write_insn("catch_exception/L");
        self.push_label_ref(label);
    }

    /// RPython blackhole.py:993 `last_exc_value/>r`.
    pub fn last_exc_value(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("last_exc_value/>r");
        self.push_u16(dst);
    }

    /// RPython blackhole.py:987 `last_exception/>i`.
    pub fn last_exception(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.write_insn("last_exception/>i");
        self.push_u16(dst);
    }

    /// RPython blackhole.py:976-985 `goto_if_exception_mismatch/iL`.
    ///
    /// `vtable` is an int operand index, which may refer either to a real
    /// Int register or to an assembler-routed Int constant pool slot.
    /// Unlike `last_exception`, this operand is not necessarily a live
    /// register, so we intentionally do not call `touch_reg()` here.
    pub fn goto_if_exception_mismatch(&mut self, vtable: u16, label: u16) {
        self.write_insn("goto_if_exception_mismatch/iL");
        self.push_u16(vtable);
        self.push_label_ref(label);
    }

    /// blackhole.py:1066 bhimpl_jit_merge_point: portal merge point.
    ///
    /// assembler.py:181-196 parity: encodes jdindex + 6 typed register
    /// lists (greens_i, greens_r, greens_f, reds_i, reds_r, reds_f).
    /// Each list is [length:u8][reg_indices:u8...].
    ///
    /// interp_jit.py:64 portal contract:
    ///   greens = ['next_instr', 'is_being_profiled', 'pycode']
    ///   reds = ['frame', 'ec']
    ///
    /// `greens_i` = [next_instr_reg, is_being_profiled_reg] (constant slots)
    /// `greens_r` = [pycode_reg] (constant slot)
    /// `reds_r`   = [frame_reg, ec_reg] (dedicated portal registers)
    ///
    /// jdindex is emitted as a `registers_i` register index via
    /// `add_const_i` (upstream `@arguments("i")` — blackhole.py:1066),
    /// matching `loop_header` above. The hard-coded `0` here reflects
    /// pyre's single-portal invariant — once multiple jitdrivers are
    /// supported the caller passes the index, but the encoding stays
    /// pool-routed.
    pub fn jit_merge_point(&mut self, greens_i: &[u8], greens_r: &[u8], reds_r: &[u8]) {
        self.write_insn("jit_merge_point/IRR");
        let jdindex_const = self.add_const_i(0); // single jitdriver
        let jdindex_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((jdindex_offset, ConstKind::Int, jdindex_const));
        // gi: green int registers (next_instr, is_being_profiled)
        self.push_u8(greens_i.len() as u8);
        for &idx in greens_i {
            self.push_u8(idx);
        }
        // gr: green ref registers (pycode)
        self.push_u8(greens_r.len() as u8);
        for &idx in greens_r {
            self.push_u8(idx);
        }
        // gf: empty
        self.push_u8(0);
        // ri: empty
        self.push_u8(0);
        // rr: red ref registers (frame, ec)
        self.push_u8(reds_r.len() as u8);
        for &idx in reds_r {
            self.push_u8(idx);
        }
        // rf: empty
        self.push_u8(0);
    }

    pub fn abort(&mut self) {
        self.write_insn("abort/");
        self.has_abort = true;
    }

    /// RPython `blackhole.py:841-862` typed return opcodes.
    /// The return value is in register `src`; `void_return` has no operand.
    pub fn int_return(&mut self, src: u16) {
        self.touch_reg(src);
        self.write_insn("int_return/i");
        self.push_u16(src);
    }

    pub fn ref_return(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("ref_return/r");
        self.push_u16(src);
    }

    pub fn float_return(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.write_insn("float_return/f");
        self.push_u16(src);
    }

    pub fn void_return(&mut self) {
        self.write_insn("void_return/");
    }

    pub fn abort_permanent(&mut self) {
        self.write_insn("abort_permanent/");
    }

    pub fn unreachable(&mut self) {
        self.write_insn("unreachable/");
    }

    /// blackhole.py bhimpl_raise(excvalue): raise exception from register.
    pub fn emit_raise(&mut self, src: u16) {
        self.write_insn("raise/r");
        self.push_u16(src);
    }

    /// blackhole.py bhimpl_reraise(): re-raise exception_last_value.
    pub fn emit_reraise(&mut self) {
        self.write_insn("reraise/");
    }

    /// pyjitpl.py opimpl_int_guard_value: promote int register to constant.
    ///
    /// Blackhole: no-op (value passes through).
    /// Tracing: emits GUARD_VALUE to specialize the trace on this value.
    pub fn int_guard_value(&mut self, src: u16) {
        self.write_insn("int_guard_value/i");
        self.push_u16(src);
    }

    /// pyjitpl.py opimpl_ref_guard_value: promote ref register to constant.
    pub fn ref_guard_value(&mut self, src: u16) {
        self.write_insn("ref_guard_value/r");
        self.push_u16(src);
    }

    /// pyjitpl.py opimpl_float_guard_value: promote float register to constant.
    pub fn float_guard_value(&mut self, src: u16) {
        self.write_insn("float_guard_value/f");
        self.push_u16(src);
    }

    pub fn inline_call(&mut self, sub_jitcode_idx: u16) {
        self.inline_call_r_v(sub_jitcode_idx, &[], None);
    }

    pub fn inline_call_r_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, &[], args_r, &[], return_i, None, None);
    }

    pub fn inline_call_r_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, &[], args_r, &[], None, return_r, None);
    }

    pub fn inline_call_r_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_r: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, &[], args_r, &[], None, None, None);
    }

    pub fn inline_call_ir_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, args_i, args_r, &[], return_i, None, None);
    }

    pub fn inline_call_ir_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, args_i, args_r, &[], None, return_r, None);
    }

    pub fn inline_call_ir_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, args_i, args_r, &[], None, None, None);
    }

    pub fn inline_call_irf_i(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_i: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            args_i,
            args_r,
            args_f,
            return_i,
            None,
            None,
        );
    }

    pub fn inline_call_irf_r(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_r: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            args_i,
            args_r,
            args_f,
            None,
            return_r,
            None,
        );
    }

    pub fn inline_call_irf_f(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_f: Option<u16>,
    ) {
        self.inline_call_grouped(
            sub_jitcode_idx,
            args_i,
            args_r,
            args_f,
            None,
            None,
            return_f,
        );
    }

    pub fn inline_call_irf_v(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        _return_v: Option<u16>,
    ) {
        self.inline_call_grouped(sub_jitcode_idx, args_i, args_r, args_f, None, None, None);
    }

    fn inline_call_grouped(
        &mut self,
        sub_jitcode_idx: u16,
        args_i: &[(u16, u16)],
        args_r: &[(u16, u16)],
        args_f: &[(u16, u16)],
        return_i: Option<u16>,
        return_r: Option<u16>,
        return_f: Option<u16>,
    ) {
        let mut typed_args = Vec::with_capacity(args_i.len() + args_r.len() + args_f.len());
        typed_args.extend(
            args_i
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Int, caller_src, callee_dst)),
        );
        typed_args.extend(
            args_r
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Ref, caller_src, callee_dst)),
        );
        typed_args.extend(
            args_f
                .iter()
                .map(|&(caller_src, callee_dst)| (JitArgKind::Float, caller_src, callee_dst)),
        );
        self.inline_call_typed(sub_jitcode_idx, &typed_args, return_i, return_r, return_f);
    }

    fn inline_call_typed(
        &mut self,
        sub_jitcode_idx: u16,
        args: &[(JitArgKind, u16, u16)],
        return_i: Option<u16>,
        return_r: Option<u16>,
        return_f: Option<u16>,
    ) {
        for &(kind, caller_src, _) in args {
            match kind {
                JitArgKind::Int => self.touch_reg(caller_src),
                JitArgKind::Ref => self.touch_ref_reg(caller_src),
                JitArgKind::Float => self.touch_float_reg(caller_src),
            }
        }
        if let Some(caller_dst) = return_i {
            self.touch_reg(caller_dst);
        }
        if let Some(caller_dst) = return_r {
            self.touch_ref_reg(caller_dst);
        }
        if let Some(caller_dst) = return_f {
            self.touch_float_reg(caller_dst);
        }
        self.start_instr(jitcode::BC_INLINE_CALL);
        self.push_u16(sub_jitcode_idx);
        self.push_u16(args.len() as u16);
        for &(kind, caller_src, callee_dst) in args {
            self.push_u8(kind.encode());
            self.push_u16(caller_src);
            self.push_u16(callee_dst);
        }
        self.push_return_slot(return_i);
        self.push_return_slot(return_r);
        self.push_return_slot(return_f);
    }

    fn push_return_slot(&mut self, ret: Option<u16>) {
        match ret {
            Some(caller_dst) => self.push_u16(caller_dst),
            None => self.push_u16(u16::MAX),
        }
    }

    pub fn residual_call_void(&mut self, fn_ptr_idx: u16, src_reg: u16) {
        self.residual_call_void_args(fn_ptr_idx, &[src_reg]);
    }

    pub fn residual_call_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.residual_call_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn residual_call_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(jitcode::BC_RESIDUAL_CALL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_may_force_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_may_force_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(jitcode::BC_CALL_MAY_FORCE_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_release_gil_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_release_gil_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(jitcode::BC_CALL_RELEASE_GIL_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_loopinvariant_void_args(&mut self, fn_ptr_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_void_typed_args(fn_ptr_idx, &args);
    }

    pub fn call_loopinvariant_void_typed_args(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_void_like(jitcode::BC_CALL_LOOPINVARIANT_VOID, fn_ptr_idx, arg_regs);
    }

    pub fn call_assembler_void_args(&mut self, target_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_void_typed_args(target_idx, &args);
    }

    pub fn call_assembler_void_typed_args(&mut self, target_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_assembler_void_like(jitcode::BC_CALL_ASSEMBLER_VOID, target_idx, arg_regs);
    }

    pub fn call_may_force_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(jitcode::BC_CALL_MAY_FORCE_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(jitcode::BC_CALL_RELEASE_GIL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_int_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_int_like(
            jitcode::BC_CALL_LOOPINVARIANT_INT,
            fn_ptr_idx,
            arg_regs,
            dst,
        );
    }

    pub fn call_assembler_int(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_int_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_int_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_int_like(jitcode::BC_CALL_ASSEMBLER_INT, target_idx, arg_regs, dst);
    }

    pub fn call_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_int(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_int_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(jitcode::BC_CALL_INT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_int_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_int_like(jitcode::BC_CALL_PURE_INT, fn_ptr_idx, arg_regs, dst);
    }

    // ── conditional_call / record_known_result (jtransform.py:1665-1688, 292-313) ──

    /// RPython: `conditional_call_ir_v(condition, funcptr, calldescr, [i], [r])`
    /// Condition in cond_reg; if nonzero, call func with args. Result void.
    /// `typed_args` carries per-argument kind (int/ref) — RPython make_three_lists parity.
    pub fn conditional_call_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        cond_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_reg(cond_reg);
        self.call_cond_like(jitcode::BC_COND_CALL_VOID, fn_ptr_idx, cond_reg, typed_args);
    }

    /// RPython: `conditional_call_value_ir_i(value, funcptr, calldescr, [i], [r])`
    pub fn conditional_call_value_ir_i_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(value_reg);
        self.touch_reg(dst);
        self.call_cond_value_like(
            jitcode::BC_COND_CALL_VALUE_INT,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
        );
    }

    /// RPython: `conditional_call_value_ir_r`
    pub fn conditional_call_value_ir_r_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        value_reg: u16,
        typed_args: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_ref_reg(value_reg);
        self.touch_ref_reg(dst);
        self.call_cond_value_like(
            jitcode::BC_COND_CALL_VALUE_REF,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
        );
    }

    /// RPython: `record_known_result_i_ir_v(result, funcptr, calldescr, [i], [r])`
    pub fn record_known_result_i_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_reg(result_reg);
        self.call_cond_like(
            jitcode::BC_RECORD_KNOWN_RESULT_INT,
            fn_ptr_idx,
            result_reg,
            typed_args,
        );
    }

    /// RPython: `record_known_result_r_ir_v`
    pub fn record_known_result_r_ir_v_typed_args(
        &mut self,
        fn_ptr_idx: u16,
        result_reg: u16,
        typed_args: &[JitCallArg],
    ) {
        self.touch_ref_reg(result_reg);
        self.call_cond_like(
            jitcode::BC_RECORD_KNOWN_RESULT_REF,
            fn_ptr_idx,
            result_reg,
            typed_args,
        );
    }

    fn call_cond_like(&mut self, bc: u8, fn_ptr_idx: u16, first_reg: u16, args: &[JitCallArg]) {
        self.push_u8(bc);
        self.push_u16(first_reg);
        self.push_u16(fn_ptr_idx);
        self.push_u8(args.len() as u8);
        for arg in args {
            self.push_u8(arg.kind as u8);
        }
        for arg in args {
            self.push_u16(arg.reg);
        }
    }

    fn call_cond_value_like(
        &mut self,
        bc: u8,
        fn_ptr_idx: u16,
        value_reg: u16,
        args: &[JitCallArg],
        dst: u16,
    ) {
        self.push_u8(bc);
        self.push_u16(value_reg);
        self.push_u16(fn_ptr_idx);
        self.push_u8(args.len() as u8);
        for arg in args {
            self.push_u8(arg.kind as u8);
        }
        for arg in args {
            self.push_u16(arg.reg);
        }
        self.push_u16(dst);
    }

    /// RPython `blackhole.py:638-640` `bhimpl_int_copy(a) returns=i`.
    /// Byte layout follows `assembler.py:165-174`: each `Register` is
    /// emitted in argcode order, so `int_copy/i>i` stores `[src][dst]`
    /// and the `>i` result byte is the last operand.
    pub fn move_i(&mut self, dst: u16, src: u16) {
        self.touch_reg(dst);
        self.touch_reg(src);
        self.write_insn("int_copy/i>i");
        self.push_u16(src);
        self.push_u16(dst);
    }

    /// `flatten.py:329` `self.emitline('int_push', v)` / `blackhole.py:662-663`
    /// `bhimpl_int_push(a)` — save `src` into the int-kind scratch slot.
    pub fn push_i(&mut self, src: u16) {
        self.touch_reg(src);
        self.write_insn("int_push/i");
        self.push_u16(src);
    }

    /// `flatten.py:331` `self.emitline('int_pop', "->", w)` / `blackhole.py:672-673`
    /// `bhimpl_int_pop()` — load `dst` from the int-kind scratch slot.
    pub fn pop_i(&mut self, dst: u16) {
        self.touch_reg(dst);
        self.write_insn("int_pop/>i");
        self.push_u16(dst);
    }

    pub fn ensure_i_regs(&mut self, count: u16) {
        self.num_regs_i = max(self.num_regs_i, count);
    }

    pub fn ensure_r_regs(&mut self, count: u16) {
        self.num_regs_r = max(self.num_regs_r, count);
    }

    pub fn ensure_f_regs(&mut self, count: u16) {
        self.num_regs_f = max(self.num_regs_f, count);
    }

    /// Lock `num_regs_{i,r,f}` at their current values so subsequent
    /// `touch_*_reg` calls cannot grow them. Callers that route
    /// `Const*` args through the constants pool depend on this — see
    /// the `num_regs_frozen` field comment for the invariant.
    pub fn freeze_num_regs(&mut self) {
        self.num_regs_frozen = true;
    }

    // ── Ref-typed builder methods ─────────────────────────────

    pub fn load_const_r_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_r(value);
        self.load_const_r(dst, const_idx);
    }

    /// Lower to RPython `ref_copy/r>r` reading from the constants
    /// window of the ref register file. See `load_const_i` for the
    /// const-patch mechanism.
    pub fn load_const_r(&mut self, dst: u16, const_idx: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("ref_copy/r>r");
        let src_offset = self.code.len();
        self.push_u16(0);
        self.const_patches
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u16(dst);
    }

    /// RPython `blackhole.py:641-643` `bhimpl_ref_copy(a) returns=r`.
    /// See `move_i` for the `[src][dst]` argcode-order layout.
    pub fn move_r(&mut self, dst: u16, src: u16) {
        self.touch_ref_reg(dst);
        self.touch_ref_reg(src);
        self.write_insn("ref_copy/r>r");
        self.push_u16(src);
        self.push_u16(dst);
    }

    /// `flatten.py:329` `self.emitline('ref_push', v)` / `blackhole.py:665-666`
    /// `bhimpl_ref_push(a)` — save `src` into the ref-kind scratch slot.
    pub fn push_r(&mut self, src: u16) {
        self.touch_ref_reg(src);
        self.write_insn("ref_push/r");
        self.push_u16(src);
    }

    /// `flatten.py:331` `self.emitline('ref_pop', "->", w)` / `blackhole.py:675-676`
    /// `bhimpl_ref_pop()` — load `dst` from the ref-kind scratch slot.
    pub fn pop_r(&mut self, dst: u16) {
        self.touch_ref_reg(dst);
        self.write_insn("ref_pop/>r");
        self.push_u16(dst);
    }

    pub fn call_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(jitcode::BC_CALL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(jitcode::BC_CALL_PURE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_ref_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_ref_like(jitcode::BC_CALL_MAY_FORCE_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(jitcode::BC_CALL_RELEASE_GIL_REF, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_loopinvariant_ref(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_ref_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_ref_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_ref_like(
            jitcode::BC_CALL_LOOPINVARIANT_REF,
            fn_ptr_idx,
            arg_regs,
            dst,
        );
    }

    pub fn call_assembler_ref(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_ref_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_ref_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_ref_like(jitcode::BC_CALL_ASSEMBLER_REF, target_idx, arg_regs, dst);
    }

    // ── Float-typed builder methods ───────────────────────────

    pub fn load_const_f_value(&mut self, dst: u16, value: i64) {
        let const_idx = self.add_const_f(value);
        self.load_const_f(dst, const_idx);
    }

    /// Lower to RPython `float_copy/f>f` reading from the constants
    /// window of the float register file. See `load_const_i` for the
    /// const-patch mechanism.
    pub fn load_const_f(&mut self, dst: u16, const_idx: u16) {
        self.touch_float_reg(dst);
        self.write_insn("float_copy/f>f");
        let src_offset = self.code.len();
        self.push_u16(0);
        self.const_patches
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u16(dst);
    }

    /// RPython `blackhole.py:644-646` `bhimpl_float_copy(a) returns=f`.
    /// See `move_i` for the `[src][dst]` argcode-order layout.
    pub fn move_f(&mut self, dst: u16, src: u16) {
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.write_insn("float_copy/f>f");
        self.push_u16(src);
        self.push_u16(dst);
    }

    /// `flatten.py:329` `self.emitline('float_push', v)` / `blackhole.py:668-669`
    /// `bhimpl_float_push(a)` — save `src` into the float-kind scratch slot.
    pub fn push_f(&mut self, src: u16) {
        self.touch_float_reg(src);
        self.write_insn("float_push/f");
        self.push_u16(src);
    }

    /// `flatten.py:331` `self.emitline('float_pop', "->", w)` / `blackhole.py:678-679`
    /// `bhimpl_float_pop()` — load `dst` from the float-kind scratch slot.
    pub fn pop_f(&mut self, dst: u16) {
        self.touch_float_reg(dst);
        self.write_insn("float_pop/>f");
        self.push_u16(dst);
    }

    pub fn call_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_pure_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_pure_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(jitcode::BC_CALL_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_pure_float_typed(&mut self, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_float_like(jitcode::BC_CALL_PURE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_may_force_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_may_force_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_may_force_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(jitcode::BC_CALL_MAY_FORCE_FLOAT, fn_ptr_idx, arg_regs, dst);
    }

    pub fn call_release_gil_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_release_gil_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_release_gil_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(
            jitcode::BC_CALL_RELEASE_GIL_FLOAT,
            fn_ptr_idx,
            arg_regs,
            dst,
        );
    }

    pub fn call_loopinvariant_float(&mut self, fn_ptr_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_loopinvariant_float_typed(fn_ptr_idx, &args, dst);
    }

    pub fn call_loopinvariant_float_typed(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_float_like(
            jitcode::BC_CALL_LOOPINVARIANT_FLOAT,
            fn_ptr_idx,
            arg_regs,
            dst,
        );
    }

    pub fn call_assembler_float(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_float_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_float_typed(
        &mut self,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.call_assembler_float_like(jitcode::BC_CALL_ASSEMBLER_FLOAT, target_idx, arg_regs, dst);
    }

    /// RPython `blackhole.py:696-719` `bhimpl_float_{add,sub,mul,truediv}`
    /// per-opname handlers. `float_floordiv` / `float_mod` have no direct
    /// RPython `bhimpl_*` — those lower to a residual call at the
    /// codewriter layer, never reaching a jitcode bytecode.
    pub fn record_binop_f(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::FloatAdd => "float_add/ff>f",
            OpCode::FloatSub => "float_sub/ff>f",
            OpCode::FloatMul => "float_mul/ff>f",
            OpCode::FloatTrueDiv => "float_truediv/ff>f",
            other => panic!("record_binop_f: unsupported opcode {other:?}"),
        };
        self.touch_float_reg(dst);
        self.touch_float_reg(lhs);
        self.touch_float_reg(rhs);
        self.write_insn(key);
        self.push_u16(dst);
        self.push_u16(lhs);
        self.push_u16(rhs);
    }

    /// RPython `blackhole.py:689-695` `bhimpl_float_{neg,abs}` per-opname handlers.
    pub fn record_unary_f(&mut self, dst: u16, opcode: OpCode, src: u16) {
        let key = match opcode {
            OpCode::FloatNeg => "float_neg/f>f",
            OpCode::FloatAbs => "float_abs/f>f",
            other => panic!("record_unary_f: unsupported opcode {other:?}"),
        };
        self.touch_float_reg(dst);
        self.touch_float_reg(src);
        self.write_insn(key);
        self.push_u16(dst);
        self.push_u16(src);
    }

    /// Append a sub-JitCode descriptor and return its runtime
    /// `descrs` index. Mirrors the RPython build-time flow where
    /// `Assembler._encode_descr(jitcode)` adds the callee `JitCode` to
    /// the shared descrs list and returns the 2-byte index that
    /// `bhimpl_inline_call_*` later resolves via `self.descrs[idx]`
    /// (`blackhole.py:150-157`).
    pub fn add_sub_jitcode(&mut self, jitcode: JitCode) -> u16 {
        self.add_sub_jitcode_arc(std::sync::Arc::new(jitcode))
    }

    /// Variant accepting an already-shared `Arc<JitCode>` for callers
    /// that already hold a shared handle (e.g. a re-export from
    /// `MetaInterpStaticData::indirectcalltargets`).
    pub fn add_sub_jitcode_arc(&mut self, jitcode: std::sync::Arc<JitCode>) -> u16 {
        let idx = self.descrs.len() as u16;
        self.descrs.push(RuntimeBhDescr::JitCode(jitcode));
        idx
    }

    pub fn add_fn_ptr(&mut self, ptr: *const ()) -> u16 {
        self.add_call_target(ptr, ptr)
    }

    /// Append a function target descriptor and return its runtime
    /// `descrs` index. Mirrors RPython `Assembler._encode_descr(calldescr)`
    /// (assembler.py:140) where the 2-byte operand downstream resolves
    /// to `self.descrs[idx]` at dispatch time. pyre dedups identical
    /// `(trace_ptr, concrete_ptr)` pairs to match
    /// `Assembler._encode_descr` memoisation.
    pub fn add_call_target(&mut self, trace_ptr: *const (), concrete_ptr: *const ()) -> u16 {
        let target = JitCallTarget::new(trace_ptr, concrete_ptr);
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::Call(existing) = entry {
                if *existing == target {
                    return idx as u16;
                }
            }
        }
        let idx = self.descrs.len() as u16;
        self.descrs.push(RuntimeBhDescr::Call(target));
        idx
    }

    pub fn add_call_assembler_target_number(
        &mut self,
        token_number: u64,
        concrete_ptr: *const (),
    ) -> u16 {
        let target = JitCallAssemblerTarget::new(token_number, concrete_ptr);
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::AssemblerToken(existing) = entry {
                if *existing == target {
                    return idx as u16;
                }
            }
        }
        let idx = self.descrs.len() as u16;
        self.descrs.push(RuntimeBhDescr::AssemblerToken(target));
        idx
    }

    pub fn add_call_assembler_target(
        &mut self,
        target: &JitCellToken,
        concrete_ptr: *const (),
    ) -> u16 {
        self.add_call_assembler_target_number(target.number, concrete_ptr)
    }

    pub fn finish(mut self) -> JitCode {
        self.patch_labels();
        self.patch_const_refs();
        self.patch_const_u8_refs();
        let body = majit_translate::jitcode::JitCodeBody {
            // RPython `jitcode.py:17 self.calldescr = calldescr` — the
            // value was stored on the builder via `set_calldescr` (the
            // analog of upstream's constructor argument).  Without an
            // explicit stage call the field remains at the
            // `BhCallDescr::default()` zero, matching the pre-set state
            // RPython's constructor shows when `calldescr=None`.
            calldescr: self.calldescr,
            code: self.code,
            constants_i: self.constants_i,
            constants_r: self.constants_r,
            constants_f: self.constants_f,
            c_num_regs_i: self.num_regs_i,
            c_num_regs_r: self.num_regs_r,
            c_num_regs_f: self.num_regs_f,
            startpoints: self.startpoints,
            alllabels: self.alllabels,
            resulttypes: Default::default(),
            _ssarepr: None,
        };
        let mut jc = JitCode::new(self.name);
        // RPython `JitCode(name, fnaddr, calldescr)` (`call.py:167-169`)
        // writes `fnaddr` at construction time before the assembler fills
        // the body. Stage it on the builder via `set_fnaddr` and stamp it
        // here so callers do not need a post-`set_body` mutation.
        jc.fnaddr = self.fnaddr;
        jc.set_body(body);
        jc.exec = super::JitCodeExecState {
            descrs: self.descrs,
        };
        // codewriter.py:68 `jitcode.index = index` — back-stamped by
        // `state::jitcode_for` at registration time. JitCode::new
        // leaves the OnceLock unset; runtime call sites that
        // previously expected `index = 0` from the flat-struct
        // Default consume `try_index()` instead, which correctly
        // returns `None` until the back-stamp lands.
        jc
    }

    fn push_u8(&mut self, value: u8) {
        self.code.push(value);
    }

    fn push_u16(&mut self, value: u16) {
        self.code.extend_from_slice(&value.to_le_bytes());
    }

    /// RPython `assembler.py:216-222` `write_insn` opcode-byte lane:
    /// looks up `opname/argcodes` in the shared insns table and emits
    /// the assigned byte. Operand emission is still done by the
    /// surrounding method because pyre's 2-byte register operands and
    /// per-BC operand layouts do not yet match the 1-byte `emit_reg` /
    /// `emit_const` encoding on the RPython side.
    fn write_insn(&mut self, key: &'static str) {
        self.start_instr(jitcode::insn_byte(key));
    }

    /// Record the current bytecode offset as an instruction start
    /// (RPython `assembler.py:200-208` writes `self.startpoints.add(pos)`
    /// just before each opcode-byte push) and emit the opcode byte.
    /// Every helper that pushes a `BC_*` opcode goes through this so
    /// `JitCode.get_live_vars_info` (RPython `jitcode.py:85-90`) can
    /// fire its non-translated `assert pc in self._startpoints` check.
    fn start_instr(&mut self, opcode: u8) {
        self.startpoints.insert(self.code.len());
        self.push_u8(opcode);
    }

    fn call_int_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_void_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_void_like(&mut self, opcode: u8, target_idx: u16, arg_regs: &[JitCallArg]) {
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(target_idx);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn push_label_ref(&mut self, label: u16) {
        // RPython `assembler.py:176` records the offset of the 2-byte
        // label slot in `alllabels` just before the temp bytes go in.
        let patch_offset = self.code.len();
        self.alllabels.insert(patch_offset);
        self.push_u16(0);
        self.patches.push((label as usize, patch_offset));
    }

    fn touch_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            return;
        }
        self.num_regs_i = max(self.num_regs_i, reg.saturating_add(1));
    }

    fn touch_ref_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            return;
        }
        self.num_regs_r = max(self.num_regs_r, reg.saturating_add(1));
    }

    fn touch_float_reg(&mut self, reg: u16) {
        if self.num_regs_frozen {
            return;
        }
        self.num_regs_f = max(self.num_regs_f, reg.saturating_add(1));
    }

    fn call_ref_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_int_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_ref_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_ref_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_float_like(&mut self, opcode: u8, fn_ptr_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(fn_ptr_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn call_assembler_float_like(
        &mut self,
        opcode: u8,
        target_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.touch_float_reg(dst);
        for &arg in arg_regs {
            self.touch_call_arg(arg);
        }
        self.start_instr(opcode);
        self.push_u16(target_idx);
        self.push_u16(dst);
        self.push_u16(arg_regs.len() as u16);
        for &arg in arg_regs {
            self.push_u8(arg.kind.encode());
            self.push_u16(arg.reg);
        }
    }

    fn touch_call_arg(&mut self, arg: JitCallArg) {
        match arg.kind {
            JitArgKind::Int => self.touch_reg(arg.reg),
            JitArgKind::Ref => self.touch_ref_reg(arg.reg),
            JitArgKind::Float => self.touch_float_reg(arg.reg),
        }
    }

    fn patch_labels(&mut self) {
        for &(label_idx, patch_offset) in &self.patches {
            let target = self.labels[label_idx].expect("jitcode label was never marked") as u16;
            let bytes = target.to_le_bytes();
            self.code[patch_offset] = bytes[0];
            self.code[patch_offset + 1] = bytes[1];
        }
    }

    /// RPython `assembler.py:131-138` resolves a const-source operand
    /// to `count_regs[kind] + len(constants) - 1`. pyre performs the
    /// same resolution as a post-emission pass once per-kind
    /// `num_regs_X` is final: `num_regs_X + pool_idx` is the slot the
    /// register file-backed `{int,ref,float}_copy` reads from.
    fn patch_const_refs(&mut self) {
        for &(offset, kind, pool_idx) in &self.const_patches {
            let base = match kind {
                ConstKind::Int => self.num_regs_i,
                ConstKind::Ref => self.num_regs_r,
                ConstKind::Float => self.num_regs_f,
            };
            let slot = base + pool_idx;
            let bytes = slot.to_le_bytes();
            self.code[offset] = bytes[0];
            self.code[offset + 1] = bytes[1];
        }
    }

    /// 1-byte counterpart of `patch_const_refs`. `loop_header/i` and
    /// `jit_merge_point/IRR` jdindex bytes (`@arguments("i")`,
    /// blackhole.py:1062,1066) carry a single register-index byte.
    /// Asserts the resolved slot fits in u8 — overflow means the portal
    /// has more than 255 int registers + constants combined, which
    /// would also break the broader 1-byte register encoding.
    fn patch_const_u8_refs(&mut self) {
        for &(offset, kind, pool_idx) in &self.const_patches_u8 {
            let base = match kind {
                ConstKind::Int => self.num_regs_i,
                ConstKind::Ref => self.num_regs_r,
                ConstKind::Float => self.num_regs_f,
            };
            let slot = base + pool_idx;
            assert!(
                slot <= u8::MAX as u16,
                "patch_const_u8_refs: slot {slot} (base={base} + pool_idx={pool_idx}) overflows u8"
            );
            self.code[offset] = slot as u8;
        }
    }
}

/// RPython `assembler.py:218-231 get_liveness_info(insn, kind)` adapted
/// for the flat-state JIT: every state_field slot is permanently live,
/// so the canonical `(live_i, live_r, live_f)` triple just enumerates
/// the int register file from `0..total_slots`.
///
/// state-field JIT enforces `Type::Int` on every slot at macro
/// expansion (`codegen_state.rs:30-43`), so `live_r` and `live_f` are
/// empty.  `total_slots` matches `JitCodeSym::total_slots`:
/// `num_scalars + sum(array_lens) + 2 * num_virt_arrays`.
///
/// The returned `Vec<u8>`s are caller-owned and can be passed directly
/// into `JitCodeBuilder::live` / `Assembler::_encode_liveness`.
///
/// # Panics
///
/// Panics if `total_slots > 255`.  RPython jitcode register indices are
/// `chr(...)`-encoded `u8`s (`assembler.py:241`,
/// `liveness.py:148-159`); a state-field JIT with more than 256 live
/// slots would overflow that encoding.
pub fn live_slots_for_state_field_jit(
    num_scalars: usize,
    array_lens: &[usize],
    num_virt_arrays: usize,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let total_slots: usize = num_scalars + array_lens.iter().sum::<usize>() + 2 * num_virt_arrays;
    assert!(
        total_slots <= u8::MAX as usize + 1 - 1, // i.e. < 256
        "live_slots_for_state_field_jit: total_slots={total_slots} exceeds RPython jitcode \
         u8 register-index limit (assembler.py:241, liveness.py:148-159)",
    );
    let live_i: Vec<u8> = (0..total_slots as u32).map(|i| i as u8).collect();
    (live_i, Vec::new(), Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_translate::jit_codewriter::assembler::Assembler;

    #[test]
    fn live_writes_opcode_byte_then_two_offset_bytes() {
        // RPython assembler.py:146-158 — `live/` opcode followed by the
        // 2-byte offset returned by `_encode_liveness`.  The first call
        // for a never-seen liveness key must produce offset 0; the next
        // call with the same key must reuse offset 0 (dedup).
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        builder.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(builder.code.len(), 1 + 2, "opcode + u16 offset");
        // u16-LE 0 means the canonical entry for this key sits at the
        // start of `Assembler::all_liveness`.
        assert_eq!(builder.code[1], 0);
        assert_eq!(builder.code[2], 0);

        builder.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(
            builder.code.len(),
            (1 + 2) * 2,
            "second call appends another opcode + offset, no all_liveness growth"
        );
        // assembler.py:236-238 dedup: identical key reuses the same
        // 0-offset entry, so the second pair of offset bytes is also 0.
        assert_eq!(builder.code[4], 0);
        assert_eq!(builder.code[5], 0);

        // assembler.py:30 `self.all_liveness = []` — only one canonical
        // payload exists despite two LIVE op emissions.
        let three_header_bytes = 3;
        let live_i_payload = 1; // [0, 1] fits in one bitset byte
        let live_r_payload = 1; // [2] fits in one bitset byte
        let live_f_payload = 0; // empty bitset → 0 bytes
        assert_eq!(
            asm.all_liveness().len(),
            three_header_bytes + live_i_payload + live_r_payload + live_f_payload,
        );
    }

    #[test]
    fn live_distinct_keys_advance_offset() {
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        builder.live(&mut asm, &[0], &[], &[]);
        let first_offset = u16::from_le_bytes([builder.code[1], builder.code[2]]);
        assert_eq!(first_offset, 0);

        builder.live(&mut asm, &[5], &[], &[]);
        let second_offset = u16::from_le_bytes([builder.code[4], builder.code[5]]);
        assert!(
            second_offset > 0,
            "distinct live_i key must advance into a fresh `all_liveness` entry"
        );
        assert_eq!(
            second_offset as usize,
            3 + 1,
            "first entry occupies 3 header + 1 payload byte"
        );
    }

    #[test]
    fn state_field_canonical_slots_empty_state() {
        // No scalars, no arrays, no virt arrays — empty triple.
        let (live_i, live_r, live_f) = super::live_slots_for_state_field_jit(0, &[], 0);
        assert!(live_i.is_empty());
        assert!(live_r.is_empty());
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_mixed_layout() {
        // Mirrors the `tlc` example shape: 1 scalar (`stackpos`) + 1
        // virt array (`stack`, ptr+len) plus a synthetic 3-element
        // flattened array.  total_slots = 1 + 3 + 2 = 6, so
        // live_i = [0, 1, 2, 3, 4, 5] and ref/float banks are empty.
        let (live_i, live_r, live_f) = super::live_slots_for_state_field_jit(1, &[3], 1);
        assert_eq!(live_i, vec![0u8, 1, 2, 3, 4, 5]);
        assert!(live_r.is_empty());
        assert!(live_f.is_empty());
    }

    #[test]
    fn state_field_canonical_slots_feeds_assembler_encode() {
        // The triple plumbs straight into `Assembler::_encode_liveness`
        // / `JitCodeBuilder::live` without further reshaping.  Two
        // distinct shapes must dedup or advance through `all_liveness`
        // exactly as `live_writes_opcode_byte_then_two_offset_bytes`
        // / `live_distinct_keys_advance_offset` cover for hand-built
        // triples.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();
        let (li, lr, lf) = super::live_slots_for_state_field_jit(2, &[1], 0);
        builder.live(&mut asm, &li, &lr, &lf);
        // Header bytes = 3 (len_i, len_r, len_f).  live_i has 3 indices
        // (0, 1, 2) all in the first bitset byte.  live_r/live_f empty.
        assert_eq!(asm.all_liveness().len(), 3 + 1);
        assert_eq!(asm.all_liveness()[0], 3, "len(live_i) header byte");
        assert_eq!(asm.all_liveness()[1], 0, "len(live_r) header byte");
        assert_eq!(asm.all_liveness()[2], 0, "len(live_f) header byte");
    }

    #[test]
    #[should_panic(expected = "exceeds RPython jitcode u8 register-index limit")]
    fn state_field_canonical_slots_panics_on_overflow() {
        // 256 slots overflows the u8 register-index encoding upstream
        // jitcode bytes are written through (assembler.py:241).
        let _ = super::live_slots_for_state_field_jit(256, &[], 0);
    }
}
