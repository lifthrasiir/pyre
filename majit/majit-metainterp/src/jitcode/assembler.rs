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
    CanonicalBhDescr, JitArgKind, JitCallArg, JitCallAssemblerTarget, JitCallTarget, JitCode,
    RuntimeBhDescr,
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
    /// Pyre-only bridge for canonical `residual_call_*_v`: the bytecode
    /// itself keeps the RPython shape (`i` funcptr operand + `d`
    /// calldescr operand), while the runtime trace path still needs the
    /// separate `{trace_ptr, concrete_ptr}` pair. Keying by the `d`
    /// operand keeps the bridge per callsite and avoids collapsing
    /// different trace wrappers that share a concrete function pointer.
    /// Drained into `JitCodeExecState.call_descr_to_call_target` at
    /// `finish()`.
    call_descr_to_call_target: std::collections::HashMap<u16, JitCallTarget>,
    /// RPython `jitcode.py:47 self._resulttypes = resulttypes` —
    /// per-instruction result-kind char keyed by end-of-instruction
    /// position (`assembler.py:217-219`).  Consumed by
    /// `pyjitpl.py:264 make_result_of_lastop` as a non-translated
    /// debug-only type check; pyre fires the same assertion in
    /// `MIFrame::make_result_of_lastop` (`frame.rs`).
    ///
    /// Populated as the LAST step of every typed-result emit helper
    /// — pyre's encoding writes operands AFTER the opcode byte, so
    /// `self.code.len()` after the last `push_u*` call equals the
    /// end-of-instruction position the reader sees as `frame.pc`.
    resulttypes: std::collections::HashMap<usize, char>,
    /// Pending result-kind for a generic `write_insn("...>X")` call.
    /// RPython records the kind after all operands have been emitted
    /// (`assembler.py:217-219`).  In this builder the opcode helper
    /// runs before operands are pushed, so we defer recording until the
    /// next instruction starts or `finish()` seals the code.
    pending_resulttype: Option<char>,
    has_abort: bool,
    /// Bytecode offset of the `BC_JIT_MERGE_POINT(_C)` opcode byte —
    /// captured by `jit_merge_point()` immediately before
    /// `write_insn` pushes the opcode.  Propagated to
    /// `JitCodeExecState::jit_merge_point_offset` at `finish()`.
    /// `None` until `jit_merge_point()` is called; second call asserts
    /// to mirror RPython's "exactly one jit_merge_point per portal
    /// jitcode" invariant (`jtransform.py:1690-1712`).
    jit_merge_point_offset: Option<usize>,
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
    /// Phase 4 / Epic B.3-B.4 deferred-patch table populated by
    /// `live_placeholder_with_triple`. Each entry pairs a `live/<offset>`
    /// patch site (returned by `live_placeholder`) with the per-marker
    /// `(live_i, live_r, live_f)` triple computed by the macro lowerer's
    /// liveness walker (`compute_per_marker_liveness`).
    ///
    /// `finalize_liveness(asm)` walks this table once after body
    /// emission, registers each triple via
    /// `Assembler::_register_liveness_offset` (which dedupes against
    /// `all_liveness_positions`), and rewrites the BC_LIVE 2-byte slot
    /// via `patch_live_offset` so each marker points at its specific
    /// entry instead of the canonical "everything-alive" entry at
    /// offset 0.
    ///
    /// Mirrors the in-line `live` path (`assembler.py:146-158`) for
    /// callers that cannot supply an `&mut Assembler` at body-emit time —
    /// e.g. the macro-emitted per-pc JitCode factory which builds bodies
    /// before the driver-shared `Assembler` is locked.
    pending_live_triples: Vec<(usize, Vec<u8>, Vec<u8>, Vec<u8>)>,
    /// Phase 4 / Epic B.3-B.4: positions of leading-dummy `BC_LIVE` slots
    /// (`live_placeholder` without an explicit triple) whose 2-byte offset
    /// must be back-patched to the assembler's canonical "all-live" entry
    /// during `finalize_liveness`.
    ///
    /// Routing the leading dummy through deferred patching keeps the
    /// canonical entry from being pre-seeded at `all_liveness` offset 0
    /// before any real `-live-` marker is registered.  Per-marker triples
    /// (registered via `liveness_prebuild_tokens`'s direct
    /// `_register_liveness_offset` calls) get the IR-walk-ordered offsets
    /// at the head of `all_liveness`, matching `assembler.assemble`'s
    /// shape; the canonical entry lands at the tail (or wherever
    /// `ensure_canonical_liveness_offset` first registers it).  The
    /// leading dummy then back-patches its 2-byte slot to that tail
    /// offset rather than assuming offset 0.
    pending_canonical_patches: Vec<usize>,
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
    ///
    /// RPython `assembler.py:127-134 emit_const` dedups every kind via the
    /// shared `constants_dict` keyed by `(kind, Constant(value_key))`, where
    /// the ref `value_key` is `None` for nullptr and `value._obj.container`
    /// otherwise. Pyre's pool stores raw `i64` pointers; identical raw values
    /// represent the same logical container, so a value-equality lookup
    /// matches upstream's container-identity dedup for all live cases
    /// (PY_NULL collapses against itself; pointer constants collapse when the
    /// caller emits the same address twice).
    pub fn add_const_r(&mut self, value: i64) -> u16 {
        if let Some(index) = self
            .constants_r
            .iter()
            .position(|&existing| existing == value)
        {
            return index as u16;
        }
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
    /// `assembler.py:165-167` `i` argcode emits 1-byte register index;
    /// `assembler.py:197-207` `d` argcode emits 2-byte descr index.
    pub fn load_state_field(&mut self, field_idx: u16, dest: u16) {
        self.touch_reg(dest);
        self.write_insn("load_state_field/di");
        self.push_u16(field_idx);
        self.push_u8(dest as u8);
    }

    /// Store an int register value into a scalar state field.
    pub fn store_state_field(&mut self, field_idx: u16, src: u16) {
        self.touch_reg(src);
        self.write_insn("store_state_field/di");
        self.push_u16(field_idx);
        self.push_u8(src as u8);
    }

    /// Load an array state field element into an int register.
    /// The element index comes from another int register.
    pub fn load_state_array(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.write_insn("load_state_array/dii");
        self.push_u16(array_idx);
        self.push_u8(index_reg as u8);
        self.push_u8(dest as u8);
    }

    /// Store an int register value into an array state field element.
    /// The element index comes from another int register.
    pub fn store_state_array(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.write_insn("store_state_array/dii");
        self.push_u16(array_idx);
        self.push_u8(index_reg as u8);
        self.push_u8(src as u8);
    }

    // ── First-class virtualizable access (getfield_vable_*) ──
    //
    // Canonical-only API: every emit threads a `vable_reg` operand for
    // the live struct pointer (`pyjitpl.py:1166 _opimpl_setfield_vable_*`
    // `r` argcode), and the descr operand resolves through the runtime
    // descr pool (`assembler.py:165-167` + `:197-207`). The pre-orthodox
    // helper-style legacy methods that omitted `vable_reg` and inlined
    // the field index were retired in Stage 3c-2 once `jit_interp` DSL
    // and pyre-jit both committed to the canonical encoding.

    pub fn vable_getfield_int_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_i/rd>i");
        self.push_reg_u8(vable_reg, "getfield_vable_i base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_i result");
    }

    pub fn vable_getfield_ref_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_r/rd>r");
        self.push_reg_u8(vable_reg, "getfield_vable_r base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_r result");
    }

    pub fn vable_getfield_float_with_base(&mut self, dest: u16, vable_reg: u16, field_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(dest);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("getfield_vable_f/rd>f");
        self.push_reg_u8(vable_reg, "getfield_vable_f base");
        self.push_u16(field_descr);
        self.push_reg_u8(dest, "getfield_vable_f result");
    }

    pub fn vable_setfield_int_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_i/rid");
        self.push_reg_u8(vable_reg, "setfield_vable_i base");
        self.push_reg_u8(src, "setfield_vable_i value");
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_ref_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_ref_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_r/rrd");
        self.push_reg_u8(vable_reg, "setfield_vable_r base");
        self.push_reg_u8(src, "setfield_vable_r value");
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_ref_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_r(value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_r/rrd");
        self.push_reg_u8(vable_reg, "setfield_vable_r base");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u16(field_descr);
    }

    /// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` accepts any box for
    /// `valuebox` (i/r/f variants share the same generic body). The
    /// optimizer may fold a register source down to `ConstInt`; the
    /// assembler must accept that as a constant-pool patch matching
    /// the `_r` variant's pattern.
    pub fn vable_setfield_int_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_i(value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_i/rid");
        self.push_reg_u8(vable_reg, "setfield_vable_i base");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
    }

    /// pyjitpl.py:1188-1199 — float counterpart to
    /// `vable_setfield_int_const_value_with_base`.
    pub fn vable_setfield_float_const_value_with_base(
        &mut self,
        vable_reg: u16,
        field_idx: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_f(value);
        self.touch_ref_reg(vable_reg);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_f/rfd");
        self.push_reg_u8(vable_reg, "setfield_vable_f base");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u16(field_descr);
    }

    pub fn vable_setfield_float_with_base(&mut self, vable_reg: u16, field_idx: u16, src: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_float_reg(src);
        let field_descr = self.add_vable_field_descr(field_idx);
        self.write_insn("setfield_vable_f/rfd");
        self.push_reg_u8(vable_reg, "setfield_vable_f base");
        self.push_reg_u8(src, "setfield_vable_f value");
        self.push_u16(field_descr);
    }

    pub fn vable_getarrayitem_int_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("getarrayitem_vable_i/ridd>i");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_i index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_i result");
    }

    pub fn vable_getarrayitem_ref_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("getarrayitem_vable_r/ridd>r");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_r index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_r result");
    }

    pub fn vable_getarrayitem_float_with_base(
        &mut self,
        dest: u16,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("getarrayitem_vable_f/ridd>f");
        self.push_reg_u8(vable_reg, "getarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "getarrayitem_vable_f index");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "getarrayitem_vable_f result");
    }

    pub fn vable_setarrayitem_int_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_i index");
        self.push_reg_u8(src, "setarrayitem_vable_i value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_ref_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_ref_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_r index");
        self.push_reg_u8(src, "setarrayitem_vable_r value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_ref_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_r(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("setarrayitem_vable_r/rirdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_r base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_r index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Ref, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// pyjitpl.py:1236-1247 `_opimpl_setarrayitem_vable` accepts any
    /// box for `valuebox`; the optimizer may fold the source down to
    /// `ConstInt`.  See `vable_setfield_int_const_value_with_base`
    /// for the parity rationale.
    pub fn vable_setarrayitem_int_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_i(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Int, true);
        self.write_insn("setarrayitem_vable_i/riidd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_i base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_i index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Int, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    /// pyjitpl.py:1236-1247 — float counterpart to
    /// `vable_setarrayitem_int_const_value_with_base`.
    pub fn vable_setarrayitem_float_const_value_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        value: i64,
    ) {
        let const_idx = self.add_const_f(value);
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_f index");
        let src_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((src_offset, ConstKind::Float, const_idx));
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_setarrayitem_float_with_base(
        &mut self,
        vable_reg: u16,
        array_idx: u16,
        index_reg: u16,
        src: u16,
    ) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(index_reg);
        self.touch_float_reg(src);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Float, false);
        self.write_insn("setarrayitem_vable_f/rifdd");
        self.push_reg_u8(vable_reg, "setarrayitem_vable_f base");
        self.push_reg_u8(index_reg, "setarrayitem_vable_f index");
        self.push_reg_u8(src, "setarrayitem_vable_f value");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
    }

    pub fn vable_arraylen_with_base(&mut self, dest: u16, vable_reg: u16, array_idx: u16) {
        self.touch_ref_reg(vable_reg);
        self.touch_reg(dest);
        let field_descr = self.add_vable_array_field_descr(array_idx);
        let array_descr = self.add_vable_array_descr(majit_ir::value::Type::Ref, false);
        self.write_insn("arraylen_vable/rdd>i");
        self.push_reg_u8(vable_reg, "arraylen_vable base");
        self.push_u16(field_descr);
        self.push_u16(array_descr);
        self.push_reg_u8(dest, "arraylen_vable result");
    }

    pub fn vable_force_with_base(&mut self, vable_reg: u16) {
        self.touch_ref_reg(vable_reg);
        self.write_insn("hint_force_virtualizable/r");
        self.push_reg_u8(vable_reg, "hint_force_virtualizable base");
    }

    /// Load from a virtualizable state array: emit GETARRAYITEM_RAW_I.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn load_state_varray(&mut self, array_idx: u16, index_reg: u16, dest: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(dest);
        self.write_insn("load_state_varray/dii");
        self.push_u16(array_idx);
        self.push_u8(index_reg as u8);
        self.push_u8(dest as u8);
    }

    /// Store to a virtualizable state array: emit SETARRAYITEM_RAW.
    /// The array stays on heap; only ptr+len are tracked as inputargs.
    pub fn store_state_varray(&mut self, array_idx: u16, index_reg: u16, src: u16) {
        self.touch_reg(index_reg);
        self.touch_reg(src);
        self.write_insn("store_state_varray/dii");
        self.push_u16(array_idx);
        self.push_u8(index_reg as u8);
        self.push_u8(src as u8);
    }

    /// Load an integer element from a GC-managed array.
    ///
    /// blackhole.py `bhimpl_getarrayitem_gc_i @arguments("r","i","d",returns="i")`:
    /// reads `registers_r[array_reg]` as the array pointer,
    /// `registers_i[index_reg]` as the element index, and `descrs[descr_idx]`
    /// as the array descriptor; writes the result into `registers_i[dst]`.
    ///
    /// Encoding: `[BC_GETARRAYITEM_GC_I][array_reg u8][index_reg u8]
    ///             [descr_idx lo u8][descr_idx hi u8][dst u8]`.
    ///
    /// Used by the dispatch JitCode body to encode `let opcode = program[pc]`
    /// (pyopcode.py:171 `ord(co_code[next_instr])`).
    pub fn getarrayitem_gc_i(&mut self, dst: u16, array_reg: u16, index_reg: u16, descr_idx: u16) {
        self.touch_ref_reg(array_reg);
        self.touch_reg(index_reg);
        self.touch_reg(dst);
        self.write_insn("getarrayitem_gc_i/rid>i");
        self.push_reg_u8(array_reg, "getarrayitem_gc_i array");
        self.push_reg_u8(index_reg, "getarrayitem_gc_i index");
        self.push_u16(descr_idx);
        self.push_reg_u8(dst, "getarrayitem_gc_i dst");
    }

    /// Add a GC-array descriptor for a byte-element array to the descrs pool.
    ///
    /// Returns the descr index to pass as `descr_idx` to `getarrayitem_gc_i`.
    /// The descriptor models a `u8`-element GC array — the `program: &[u8]`
    /// bytecode slice used by the dispatch JitCode body.
    ///
    /// Deduped: multiple calls with the same shape return the same index.
    ///
    /// `itemsize=1` mirrors RPython `pypy/interpreter/pyopcode.py:171`
    /// `ord(co_code[next_instr])` byte-load semantics. `is_item_signed=false`
    /// because `ord()` yields a non-negative `0..=255` integer (zero-extend
    /// to `i64`, never sign-extend); the `u8` SUB-INTERVAL property must
    /// not be widened to a signed `i64` at the descr boundary or the
    /// backend would emit `movsx` on byte loads ≥ `0x80` and corrupt
    /// opcode dispatch.  `base_size=0` because Rust `&[u8]` data pointers
    /// (codegen_trace.rs:193 `*const #env_type as *const ()`) point
    /// directly at the first element without any GC header.
    pub fn add_gc_byte_array_descr(&mut self) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Array {
            base_size: 0,
            itemsize: 1,
            // base_size=0 → no length header (raw `&[u8]` data pointer
            // points directly at items[0]); descr.py:359-362 nolength
            // shape carries `lendescr=None`.
            len_offset: None,
            type_id: 0,
            item_type: majit_ir::value::Type::Int,
            is_array_of_pointers: false,
            is_array_of_structs: false,
            is_item_signed: false,
            ei_index: u32::MAX,
            // `&[u8]` byte-array descrs are minted at assembler bootstrap
            // with no source-level array_type_id; the structural tuple
            // (base_size=0, itemsize=1, …) uniquely identifies them.
            array_type_id: None,
            interior_fields: Vec::new(),
        })
    }

    /// RPython `blackhole.py:459-521` `bhimpl_int_*` per-opname handlers:
    /// each primitive has its own insn_id in `BlackholeInterpBuilder.insns`
    /// (`blackhole.py:52-81 setup_insns`). Emits via `write_insn` with
    /// the canonical `opname/ii>i` key so the opcode byte comes from the
    /// shared insns table rather than a hand-assigned `BC_*` constant.
    ///
    /// `OpCode::IntFloorDiv` / `OpCode::IntMod` are intentionally absent:
    /// `jtransform.py:575-577` rewrites both to
    /// `direct_call(ll_int_py_*)` before jitcode emission, so neither
    /// `bhimpl_*` nor the corresponding `int_(floordiv|mod)/ii>i` insns
    /// key exists upstream.  Pyre's β' redirect at
    /// `majit-translate/src/codegen.rs::generated_binary_int_value`
    /// covers the runtime trace path.
    pub fn record_binop_i(&mut self, dst: u16, opcode: OpCode, lhs: u16, rhs: u16) {
        let key = match opcode {
            OpCode::IntAdd => "int_add/ii>i",
            OpCode::IntSub => "int_sub/ii>i",
            OpCode::IntMul => "int_mul/ii>i",
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

    /// RPython `flatten.py:247` emits the bool exitswitch as opname
    /// `goto_if_not` (not `goto_if_not_int_is_true`).  The `_int_is_true`
    /// suffix in upstream is a Python class-attribute alias on
    /// `BlackholeInterpreter` (`blackhole.py:913`
    /// `bhimpl_goto_if_not_int_is_true = bhimpl_goto_if_not`) that
    /// shares the handler function under two attribute names — it is
    /// NOT a second opname registered in `Assembler.insns`.  The Rust
    /// method name preserves the longer attribute spelling for
    /// readability; the bytecode key matches upstream's single opname.
    pub fn goto_if_not_int_is_true(&mut self, reg: u16, label: u16) {
        self.touch_reg(reg);
        self.write_insn("goto_if_not/iL");
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
    /// SpaceOperation('loop_header', [c_index], None) with
    /// `Constant(jd.index, lltype.Signed)`. blackhole.py:1063
    /// bhimpl_loop_header(jdindex) is a no-op; pyjitpl.py:1527
    /// opimpl_loop_header records the jitdriver index for the trace.
    ///
    /// `@arguments("i")` (blackhole.py:1062) parity: jdindex is encoded
    /// as a single register-index byte pointing into the post-regs
    /// constants suffix of `registers_i`. `loop_header` is not in
    /// `assembler.py:312-346 USE_C_FORM`, so the only valid argcode is
    /// `i` (constants-pool slot) — the short-form `c` is never emitted
    /// regardless of jdindex magnitude. `add_const_i` registers the
    /// value; the placeholder is patched at `finish()` once `num_regs_i`
    /// is final.
    pub fn loop_header(&mut self, jdindex: i64) {
        self.write_insn("loop_header/i");
        let const_idx = self.add_const_i(jdindex);
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

    /// Emit the raw `BC_LIVE` opcode + 2-byte zero offset, returning the
    /// operand offset so a caller can record the patch for one of the
    /// deferred-patch lists.
    fn write_live_placeholder_bytes(&mut self) -> usize {
        self.write_insn("live/");
        let patch_offset = self.code.len();
        self.push_u16(0);
        patch_offset
    }

    /// RPython assembler.py: emit `live/` followed by a 2-byte offset into
    /// the shared all_liveness byte string.  Used for the leading dummy
    /// `BC_LIVE` slot at the start of every per-pc JitCode (which has no
    /// per-marker triple of its own — it points at the canonical "all
    /// live" entry).  The 2-byte slot is written as `0x0000` here and
    /// back-patched during [`finalize_liveness`] via
    /// `Assembler::ensure_canonical_liveness_offset`.  Returns the operand
    /// offset so callers may chain into a custom patcher if needed.
    pub fn live_placeholder(&mut self) -> usize {
        let patch_offset = self.write_live_placeholder_bytes();
        self.pending_canonical_patches.push(patch_offset);
        patch_offset
    }

    pub fn patch_live_offset(&mut self, patch_offset: usize, offset: u16) {
        let bytes = offset.to_le_bytes();
        self.code[patch_offset] = bytes[0];
        self.code[patch_offset + 1] = bytes[1];
    }

    /// Phase 4 / Epic B.3-B.4 deferred-patch entry point: emit a `live/`
    /// opcode followed by a 2-byte placeholder offset (mirroring
    /// [`live_placeholder`]) and record the per-marker
    /// `(live_i, live_r, live_f)` triple in `pending_live_triples` so
    /// [`finalize_liveness`] can later resolve and patch the offset.
    ///
    /// Each `live_*` slice must be a sorted+dedup register-set view
    /// matching the macro lowerer's
    /// `compute_per_marker_liveness` output (`liveness.py:33-79`).  Pyre
    /// renormalises in `_register_liveness_offset`, so callers may pass
    /// arbitrary order; passing the lowerer's already-sorted output
    /// matches RPython's `liveness.py:148 live = sorted(live)` shape.
    pub fn live_placeholder_with_triple(
        &mut self,
        live_i: &[u8],
        live_r: &[u8],
        live_f: &[u8],
    ) -> usize {
        let patch_offset = self.write_live_placeholder_bytes();
        self.pending_live_triples.push((
            patch_offset,
            live_i.to_vec(),
            live_r.to_vec(),
            live_f.to_vec(),
        ));
        patch_offset
    }

    /// Phase 4 / Epic B.3-B.4 finalisation step: register every pending
    /// per-marker liveness triple into `asm` (deduplicating against the
    /// shared `all_liveness_positions`) and rewrite each corresponding
    /// `live/<offset>` BC_LIVE slot via [`patch_live_offset`].
    ///
    /// Mirrors `assembler.py:146-158`'s per-marker
    /// `_encode_liveness(live_i, live_r, live_f) → encode_offset(pos)`
    /// pair, deferred to a single post-emit pass for callers (the
    /// `#[jit_interp]` per-pc JitCode factory) that can only acquire the
    /// driver-shared `&mut Assembler` after the body is built.
    ///
    /// Idempotent: drains `pending_live_triples` and
    /// `pending_canonical_patches` so a second call is a no-op.
    ///
    /// Within a single call, canonical patches are processed before
    /// per-marker triples; this lets `ensure_canonical_liveness_offset`
    /// short-circuit to the cached offset on every subsequent invocation.
    /// At the macro level the per-pc liveness *prebuild*
    /// (`liveness_prebuild_tokens` in `jitcode_lower.rs`) directly calls
    /// `_register_liveness_offset` for every walker-emitted marker before
    /// any `finalize_liveness` runs, so by the time the first leading
    /// dummy fires (at trace time) per-marker triples already populate
    /// `all_liveness` and the safety-net
    /// `ensure_canonical_liveness_offset` call in
    /// `install_canonical_liveness` has appended canonical at the end.
    /// This matches RPython `assembler.assemble`'s shape: real `-live-`
    /// markers occupy the IR-walk-ordered positions; the canonical
    /// "all-live" entry exists only as a pyre-side affordance for the
    /// leading-dummy assertion.
    ///
    /// If `pending_canonical_patches` is non-empty,
    /// `Assembler::ensure_canonical_liveness_offset` must succeed (i.e.
    /// `set_canonical_liveness_triple` must have been called earlier);
    /// the test paths that emit a leading dummy without ever calling
    /// `finalize_liveness` keep the original `0x0000` placeholder bytes
    /// in `code` and never observe this assertion.
    pub fn finalize_liveness(
        &mut self,
        asm: &mut majit_translate::jit_codewriter::assembler::Assembler,
    ) {
        let canonical_patches = std::mem::take(&mut self.pending_canonical_patches);
        if !canonical_patches.is_empty() {
            let canonical_off = asm.ensure_canonical_liveness_offset();
            assert!(
                canonical_off < (1 << 16),
                "canonical all_liveness offset {} exceeds 2-byte encoding",
                canonical_off
            );
            let canonical_off_u16 = canonical_off as u16;
            for patch_offset in canonical_patches {
                self.patch_live_offset(patch_offset, canonical_off_u16);
            }
        }
        for (patch_offset, live_i, live_r, live_f) in std::mem::take(&mut self.pending_live_triples)
        {
            let pos = asm._register_liveness_offset(&live_i, &live_r, &live_f);
            // assembler.py:248 `encode_offset(pos, self.code)` — pyre
            // patches the already-emitted 2-byte slot in place rather
            // than appending; the bit pattern is identical.
            assert!(
                pos < (1 << 16),
                "all_liveness offset {} exceeds 2-byte encoding",
                pos
            );
            self.patch_live_offset(patch_offset, pos as u16);
        }
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
    /// jdindex is emitted per assembler.py:163,312 USE_C_FORM rules —
    /// `'c'` (raw signed byte) when fitting in `i8`, otherwise `'i'`
    /// (constants-pool slot). The blackhole `@arguments("i", ...)`
    /// decoder (blackhole.py:113-123) interprets the byte per the
    /// runtime argcode.
    ///
    /// jtransform.py:1704 emits `Constant(self.portal_jd.index,
    /// lltype.Signed)` so the assembler-side jdindex is signed; the
    /// USE_C_FORM short-form check is `-128 <= value <= 127`
    /// (assembler.py:99-107).
    ///
    /// interp_jit.py:64 portal contract for pyre's current
    /// `pypyjitdriver`:
    ///   greens = ['next_instr', 'is_being_profiled', 'pycode']
    ///   reds = ['frame', 'ec']
    /// — so `greens_f`, `reds_i`, `reds_f` arrive empty today. The
    /// helper does not assume that: a future jitdriver with float
    /// greens or red ints flows through the same code path with
    /// non-empty lists.
    pub fn jit_merge_point(
        &mut self,
        jdindex: i64,
        greens_i: &[u8],
        greens_r: &[u8],
        greens_f: &[u8],
        reds_i: &[u8],
        reds_r: &[u8],
        reds_f: &[u8],
    ) {
        // Capture the bytecode offset of the first OPCODE byte (before
        // `write_insn` pushes it).  PyPy's portal dispatch loop executes
        // `jit_merge_point()` at every bytecode dispatch, so a lowered
        // portal jitcode can contain more than one merge point when pyre
        // materializes several Python loop headers in one dispatch body.
        // The offset is only used by `register_dispatch_jitcode` for
        // schema validation, so keeping the first one preserves that
        // validation without rejecting later structurally identical merge
        // points.
        if self.jit_merge_point_offset.is_none() {
            self.jit_merge_point_offset = Some(self.code.len());
        }
        if (-128..=127).contains(&jdindex) {
            self.write_insn("jit_merge_point/cIRFIRF");
            self.push_u8((jdindex & 0xFF) as u8);
        } else {
            self.write_insn("jit_merge_point/iIRFIRF");
            let jdindex_const = self.add_const_i(jdindex);
            let jdindex_offset = self.code.len();
            self.push_u8(0);
            self.const_patches_u8
                .push((jdindex_offset, ConstKind::Int, jdindex_const));
        }
        for list in [greens_i, greens_r, greens_f, reds_i, reds_r, reds_f] {
            // RPython `assembler.py` encodes list lengths as a single byte
            // (`chr(len(lst))`); going past 255 silently wraps in Rust and
            // mis-encodes the per-list count.  Strict assert so the wrap
            // surfaces as a hard fail.
            let len = list.len();
            assert!(
                len < 256,
                "jit_merge_point list length {len} exceeds u8 byte encoding"
            );
            self.push_u8(len as u8);
            for &idx in list {
                self.push_u8(idx);
            }
        }
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
        self.start_instr(jitcode::insns::BC_INLINE_CALL);
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
        // RPython `assembler.py:217-219` — record the result kind at
        // end-of-instruction position.  Inline-call's typed result
        // (consumed by `MIFrame::make_result_of_lastop` at the caller
        // frame after the callee's `finishframe_*_return`,
        // `pyjitpl/mod.rs:9975`) is determined by which
        // `return_{i,r,f}` slot the helper received.  At most one is
        // `Some` for a typed variant; all `None` for the void
        // variant (no record).
        match (return_i, return_r, return_f) {
            (Some(_), None, None) => self.record_resulttype('i'),
            (None, Some(_), None) => self.record_resulttype('r'),
            (None, None, Some(_)) => self.record_resulttype('f'),
            (None, None, None) => {} // _v variant — no result
            slots => {
                panic!("inline_call_typed: at most one return slot may be Some, got {slots:?}")
            }
        }
    }

    fn push_return_slot(&mut self, ret: Option<u16>) {
        match ret {
            Some(caller_dst) => self.push_u16(caller_dst),
            None => self.push_u16(u16::MAX),
        }
    }

    /// Slice 1 adapter — bridges `fn_ptr_idx`-using emit sites to the
    /// canonical `residual_call_*_v` byte layout. Looks up the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materializes
    /// `concrete_ptr` in the int constants pool, derives a `BhCallDescr`
    /// from `arg_regs` kinds (result `Void`, default `EffectInfo`), and
    /// records the pyre-only `(calldescr_idx → JitCallTarget)` bridge for
    /// trace-time pointer selection.
    ///
    /// The 7 production emit sites referenced in
    /// `pyre-call-family-canonical-migration.md` Slice 1 (Slice 1b)
    /// route through this adapter so callers do not have to thread
    /// `concrete_ptr` and `BhCallDescr` separately.
    pub fn residual_call_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        // pyjitpl.py:2655 do_residual_call invalidates the heapcache from
        // descriptor effects before recording the call. `default_effect_info()`
        // returns the PyPy `call.py:300-301` `EF_CAN_RAISE` shape with empty
        // raw sets (`effectinfo.py:293-299` else-branch) + empty bitstrings +
        // `can_collect=true`, matching the analyzer-empty external-call
        // outcome (`graphanalyze.py:60 bottom_result()`). Using
        // `EffectInfo::default()` (also empty) would drop the `CanRaise`
        // extraeffect, suppressing the `GUARD_NO_EXCEPTION` the walker emits
        // per `effectinfo.py:236 check_can_raise()`.
        self.residual_call_void_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            crate::call_descr::default_effect_info(),
        );
    }

    pub fn residual_call_void_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            effect_info,
            "residual_call_void_canonical_via_target",
        );
    }

    /// Slice 0e of `pyre-call-family-canonical-migration.md`: emit canonical
    /// `residual_call_{r,ir,irf}_v` opname/argcodes byte layout.
    ///
    /// Policy-specific void calls use this same bytecode family; their
    /// behaviour is carried by `calldescr.extra_info`, matching RPython
    /// `pyjitpl.py` `do_residual_call`.
    pub fn residual_call_void_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
    ) {
        let calldescr_idx = self.emit_canonical_call_void(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            funcptr,
            args,
            calldescr,
        );
        let funcptr = funcptr as *const ();
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::new(funcptr, funcptr));
    }

    /// Generic canonical `*_v` emission body shared by `residual_call`,
    /// `call_may_force`, `call_release_gil`, `call_loopinvariant`.
    ///
    /// `funcptr` is the raw concrete C-ABI pointer that
    /// `cpu.bh_call_v` (cranelift `compiler.rs:13934`, dynasm
    /// `runner.rs` Slice 0d override) will eventually transmute into a
    /// typed `extern "C" fn(...)`. The pointer is stashed in the int
    /// constants pool — RPython `assembler.py:127-138 emit_const`
    /// projects const-pool slot N into the post-regs window of
    /// `bh.registers_i` so the canonical handler at
    /// `blackhole.rs:6534, 6580, 6621` can resolve `bh.registers_i[
    /// code[pos]]` uniformly. `finish()` patches the placeholder byte
    /// to `num_regs_i + funcptr_const_idx` (1-byte ceiling enforced by
    /// `patch_const_u8_refs`).
    ///
    /// `args` is bucket-sorted by kind into the int / ref / float lists
    /// the canonical handler reads in order. Sub-form auto-selection:
    ///   - any float arg → `opcode_irf_v` (`iIRFd`)
    ///   - else any int arg → `opcode_ir_v` (`iIRd`)
    ///   - else → `opcode_r_v` (`iRd`)
    /// (R variant cannot hold int / float bytes — the handler reads only
    /// the funcptr_reg byte + the R list + the descr.)
    fn emit_canonical_call_void(
        &mut self,
        opcodes: (u8, u8, u8),
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
    ) -> u16 {
        let (opcode_r_v, opcode_ir_v, opcode_irf_v) = opcodes;
        let mut int_regs: Vec<u16> = Vec::new();
        let mut ref_regs: Vec<u16> = Vec::new();
        let mut float_regs: Vec<u16> = Vec::new();
        for &arg in args {
            self.touch_call_arg(arg);
            match arg.kind {
                JitArgKind::Int => int_regs.push(arg.reg),
                JitArgKind::Ref => ref_regs.push(arg.reg),
                JitArgKind::Float => float_regs.push(arg.reg),
            }
        }

        let has_float = !float_regs.is_empty();
        let has_int = !int_regs.is_empty();
        let opcode = if has_float {
            opcode_irf_v
        } else if has_int {
            opcode_ir_v
        } else {
            opcode_r_v
        };

        let funcptr_const_idx = self.add_const_i(funcptr);
        let calldescr_idx = self.add_call_descr(calldescr);

        self.start_instr(opcode);
        // funcptr_reg: 1-byte placeholder, finish() patches to
        // `num_regs_i + funcptr_const_idx` once per-kind register counts
        // freeze (RPython `assembler.py:127-138` const-pool projection).
        let funcptr_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((funcptr_offset, ConstKind::Int, funcptr_const_idx));

        // IR / IRF carry the int list before the ref list.
        if has_float || has_int {
            assert!(
                int_regs.len() <= u8::MAX as usize,
                "canonical call_*_v int arg count {} overflows u8",
                int_regs.len()
            );
            self.push_u8(int_regs.len() as u8);
            for reg in int_regs {
                self.push_reg_u8(reg, "canonical call_*_v int arg");
            }
        }
        // R / IR / IRF all carry the ref list.
        assert!(
            ref_regs.len() <= u8::MAX as usize,
            "canonical call_*_v ref arg count {} overflows u8",
            ref_regs.len()
        );
        self.push_u8(ref_regs.len() as u8);
        for reg in ref_regs {
            self.push_reg_u8(reg, "canonical call_*_v ref arg");
        }
        // IRF carries the float list.
        if has_float {
            assert!(
                float_regs.len() <= u8::MAX as usize,
                "canonical call_*_v float arg count {} overflows u8",
                float_regs.len()
            );
            self.push_u8(float_regs.len() as u8);
            for reg in float_regs {
                self.push_reg_u8(reg, "canonical call_*_v float arg");
            }
        }
        // calldescr is encoded as a 2-byte runtime descrs index — read by
        // `blackhole.rs:5853 read_descr`.
        self.push_u16(calldescr_idx);
        calldescr_idx
    }

    /// Emit a canonical `residual_call_*_v` whose calldescr carries
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`.
    pub fn call_may_force_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            // PyPy `call.py:288-289 if virtualizable_analyzer.analyze(op):`
            // selects `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, fed through
            // `effectinfo_from_writeanalyze` with the analyzer-empty
            // (`graphanalyze.py:60 bottom_result()`) default. The
            // resulting EI is the dedicated FVOV slot — distinct from
            // `MOST_GENERAL`/RandomEffects (only the
            // `randomeffects_analyzer` branch at `call.py:282-283`).
            // Routing to `MOST_GENERAL` over-invalidates the heap cache
            // via `has_random_effects() → clean_caches` PyPy reserves
            // for genuinely-random callees.
            crate::call_descr::forces_virtual_or_virtualizable_effect_info(),
            "call_may_force_void_canonical_via_target",
        );
    }

    /// Emit a canonical `residual_call_*_v` whose calldescr carries the
    /// release-gil marker. RPython selects this policy from
    /// `calldescr.extra_info`, not from a separate bytecode family.
    pub fn call_release_gil_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            majit_ir::descr::EffectInfo {
                // RPython `effectinfo.py:271-273 MOST_GENERAL` parity:
                // release-gil callees default to RandomEffects with
                // `can_invalidate=true` so the heapcache `clear_caches`
                // path fires (heapcache.py:343-353) instead of only
                // the escape-based fallback. effectinfo.py:149-155
                // keeps every readonly/write descr set None for
                // `EF_RANDOM_EFFECTS`. `(1, 0)` is the unresolved
                // sentinel — the inner
                // `emit_canonical_call_*_via_target` helper looks up
                // the `JitCallTarget` from `descrs[fn_ptr_idx]` and
                // calls `resolve_call_release_gil_target` to
                // substitute both the real
                // `_call_aroundstate_target_[0]` (`rffi.py:228`)
                // address and the wrapper's `save_err` flag bits
                // (`rffi.py:62-71`).
                call_release_gil_target: (1, 0),
                ..majit_ir::descr::EffectInfo::MOST_GENERAL
            },
            "call_release_gil_void_canonical_via_target",
        );
    }

    /// Emit a canonical `residual_call_*_v` whose calldescr carries
    /// `EF_LOOPINVARIANT`.
    pub fn call_loopinvariant_void_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
    ) {
        // RPython `codewriter/call.py:249-251 getcalldescr`:
        //   if loopinvariant:
        //       assert not NON_VOID_ARGS, ("arguments not supported for "
        //                                  "loop-invariant function!")
        // Loop-invariant direct_call must take no non-void arguments.
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.emit_canonical_call_void_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_V,
                jitcode::insns::BC_RESIDUAL_CALL_IR_V,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_V,
            ),
            fn_ptr_idx,
            arg_regs,
            // RPython `effectinfo.py:169-181 effectinfo_from_writeanalyze`:
            // EF_LOOPINVARIANT clears `_write_descrs_*`. Empty bitsets
            // here are intentional, not the unknown-callee fallback used
            // by may_force / release_gil.
            crate::call_descr::LOOPINVARIANT_EFFECT_INFO,
            "call_loopinvariant_void_canonical_via_target",
        );
    }

    /// Generic via_target body shared by Slices 1 and 2: resolves the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materializes
    /// `concrete_ptr` in the int constants pool, derives a void
    /// `BhCallDescr` from arg kinds and effect policy, and records the
    /// pyre trace/concrete pointer bridge against the emitted `d`
    /// operand.
    fn emit_canonical_call_void_via_target(
        &mut self,
        opcodes: (u8, u8, u8),
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        effect_info: majit_ir::descr::EffectInfo,
        helper_name: &'static str,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target,
            other => panic!(
                "{helper_name}: descrs[{fn_ptr_idx}] expected \
                 RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        let concrete_ptr = target.concrete_ptr as i64;
        let effect_info =
            resolve_call_release_gil_target(effect_info, target.concrete_ptr, target.save_err);

        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            majit_ir::value::Type::Void,
            effect_info,
        );
        let calldescr_idx =
            self.emit_canonical_call_void(opcodes, concrete_ptr, arg_regs, calldescr);
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    /// Sibling of `emit_canonical_call_void` for the non-void result
    /// shapes. RPython `blackhole.py:1228-1252`
    /// `bhimpl_residual_call_{r,ir,irf}_{i,r,f}`. The encoding mirrors
    /// the void form with one trailing `dst:u8` byte appended after
    /// the calldescr operand (consumed by handlers
    /// `handler_residual_call_*_{i,r,f}` at `blackhole.rs:6611-6660` via
    /// `code[p]`).
    ///
    /// Slice 4 Slice 0 of `pyre-call-family-canonical-migration.md`:
    /// foundation. Slice 4 Slice 1a (`residual_call_*_canonical_*`
    /// wrappers below) is the first non-dormant caller.
    fn emit_canonical_call_typed(
        &mut self,
        opcodes: (u8, u8, u8),
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
        dst: u16,
        dst_kind: JitArgKind,
    ) -> u16 {
        let calldescr_idx = self.emit_canonical_call_void(opcodes, funcptr, args, calldescr);
        // Result-bank touch + trailing dst byte (`>i` / `>r` / `>f`
        // suffix in `wellknown_bh_insns` keys).
        match dst_kind {
            JitArgKind::Int => self.touch_reg(dst),
            JitArgKind::Ref => self.touch_ref_reg(dst),
            JitArgKind::Float => self.touch_float_reg(dst),
        }
        self.push_reg_u8(dst, "canonical residual_call_*_{i,r,f} dst");
        // RPython `assembler.py:217-219` records `_resulttypes[len(self.code)]
        // = op.result.kind` for any typed-result op.  Mirror here so the
        // canonical residual_call_*_{i,r,f} family populates the same map
        // the legacy `call_*_like` siblings already do — `MIFrame::
        // make_result_of_lastop` (`pyjitpl.py:260-265`) reads it back when
        // a caller materialises the typed result.
        self.record_resulttype(match dst_kind {
            JitArgKind::Int => 'i',
            JitArgKind::Ref => 'r',
            JitArgKind::Float => 'f',
        });
        calldescr_idx
    }

    /// Generic typed via_target body — sibling of
    /// [`Self::emit_canonical_call_void_via_target`] threading `dst` /
    /// `dst_kind` for the trailing result-bank byte.  Resolves the
    /// `JitCallTarget` at `descrs[fn_ptr_idx]`, materialises
    /// `concrete_ptr` in the int constants pool, derives a typed
    /// `BhCallDescr` from arg kinds + `result_type` + `effect_info`,
    /// and records the pyre trace/concrete pointer bridge against the
    /// emitted `d` operand.
    #[allow(dead_code)]
    fn emit_canonical_call_typed_via_target(
        &mut self,
        opcodes: (u8, u8, u8),
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        result_type: majit_ir::value::Type,
        effect_info: majit_ir::descr::EffectInfo,
        dst: u16,
        dst_kind: JitArgKind,
        helper_name: &'static str,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target,
            other => panic!(
                "{helper_name}: descrs[{fn_ptr_idx}] expected \
                 RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        let concrete_ptr = target.concrete_ptr as i64;
        let effect_info =
            resolve_call_release_gil_target(effect_info, target.concrete_ptr, target.save_err);
        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            result_type,
            effect_info,
        );
        let calldescr_idx = self.emit_canonical_call_typed(
            opcodes,
            concrete_ptr,
            arg_regs,
            calldescr,
            dst,
            dst_kind,
        );
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    /// Slice 4 of `pyre-call-family-canonical-migration.md`: int-result
    /// sibling of [`Self::residual_call_void_canonical_via_target`].
    /// `bhimpl_residual_call_{r,ir,irf}_i` (`blackhole.py:1225-1247`)
    /// dispatches via `cpu.bh_call_i` and writes the result into
    /// `bh.registers_i[dst]`.
    #[allow(dead_code)]
    pub fn residual_call_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::default_effect_info(),
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_int_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_typed_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_I,
                jitcode::insns::BC_RESIDUAL_CALL_IR_I,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
            ),
            fn_ptr_idx,
            arg_regs,
            majit_ir::value::Type::Int,
            effect_info,
            dst,
            JitArgKind::Int,
            "residual_call_int_canonical_via_target",
        );
    }

    /// Slice 4: ref-result sibling.  `bhimpl_residual_call_{r,ir,irf}_r`
    /// (`blackhole.py:1228-1250`) dispatches via `cpu.bh_call_r`.
    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::default_effect_info(),
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        self.emit_canonical_call_typed_via_target(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_R,
                jitcode::insns::BC_RESIDUAL_CALL_IR_R,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
            ),
            fn_ptr_idx,
            arg_regs,
            majit_ir::value::Type::Ref,
            effect_info,
            dst,
            JitArgKind::Ref,
            "residual_call_ref_canonical_via_target",
        );
    }

    /// Float-result emission body.  `bhimpl_residual_call_irf_f` is the
    /// only float-result variant per `resoperation.py:1238-1248`, so
    /// the opcode is fixed — but the handler at
    /// `handler_residual_call_irf_f` (`blackhole.rs:6644`) always reads
    /// `read_list_f` after `read_list_r`, so the layout MUST emit an
    /// (empty-or-not) float list even when the call has no float args.
    /// This differs from `emit_canonical_call_void`, which skips the
    /// float-list bytes whenever `has_float == false` (a valid
    /// optimisation for the `R_V` / `IR_V` variants but illegal for
    /// `IRF_F`).
    fn emit_canonical_call_typed_irf_f(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) -> u16 {
        let mut int_regs: Vec<u16> = Vec::new();
        let mut ref_regs: Vec<u16> = Vec::new();
        let mut float_regs: Vec<u16> = Vec::new();
        for &arg in args {
            self.touch_call_arg(arg);
            match arg.kind {
                JitArgKind::Int => int_regs.push(arg.reg),
                JitArgKind::Ref => ref_regs.push(arg.reg),
                JitArgKind::Float => float_regs.push(arg.reg),
            }
        }
        let funcptr_const_idx = self.add_const_i(funcptr);
        let calldescr_idx = self.add_call_descr(calldescr);

        self.start_instr(jitcode::insns::BC_RESIDUAL_CALL_IRF_F);
        let funcptr_offset = self.code.len();
        self.push_u8(0);
        self.const_patches_u8
            .push((funcptr_offset, ConstKind::Int, funcptr_const_idx));

        // IRF mandates all three (count, regs) pairs, even if a list
        // is empty.  Layout matches `read_list_i` / `_r` / `_f` in the
        // canonical handler.
        for (regs, label) in [
            (&int_regs, "canonical residual_call_irf_f int arg"),
            (&ref_regs, "canonical residual_call_irf_f ref arg"),
            (&float_regs, "canonical residual_call_irf_f float arg"),
        ] {
            assert!(
                regs.len() <= u8::MAX as usize,
                "canonical residual_call_irf_f arg count {} overflows u8",
                regs.len()
            );
            self.push_u8(regs.len() as u8);
            for &reg in regs {
                self.push_reg_u8(reg, label);
            }
        }
        self.push_u16(calldescr_idx);
        self.touch_float_reg(dst);
        self.push_reg_u8(dst, "canonical residual_call_irf_f dst");
        // RPython `assembler.py:217-219` records `_resulttypes[len(self.code)]
        // = op.result.kind` for any typed-result op.  Sibling
        // `emit_canonical_call_typed` (line 1951) already records this for
        // `_i` / `_r` / `_f` (non-IRF) result variants — float-on-IRF must
        // do the same so `MIFrame::make_result_of_lastop`
        // (`pyjitpl.py:260-265`) reads the correct kind back.
        self.record_resulttype('f');
        calldescr_idx
    }

    /// Slice 4: float-result sibling.  Always uses `IRF_F` per
    /// `resoperation.py:1238-1248` ("no such thing" `R_F` / `IR_F`)
    /// and goes through [`Self::emit_canonical_call_typed_irf_f`] so
    /// the F list count byte is always present.
    #[allow(dead_code)]
    pub fn residual_call_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::default_effect_info(),
        );
    }

    #[allow(dead_code)]
    pub fn residual_call_float_canonical_via_target_with_effect_info(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
        effect_info: majit_ir::descr::EffectInfo,
    ) {
        let target = match self.descrs.get(fn_ptr_idx as usize) {
            Some(RuntimeBhDescr::Call(target)) => *target,
            other => panic!(
                "residual_call_float_canonical_via_target: descrs[{fn_ptr_idx}] \
                 expected RuntimeBhDescr::Call, got {other:?}"
            ),
        };
        let concrete_ptr = target.concrete_ptr as i64;
        let effect_info =
            resolve_call_release_gil_target(effect_info, target.concrete_ptr, target.save_err);
        let arg_classes: String = arg_regs
            .iter()
            .map(|a| match a.kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            })
            .collect();
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            arg_classes,
            majit_ir::value::Type::Float,
            effect_info,
        );
        let calldescr_idx =
            self.emit_canonical_call_typed_irf_f(concrete_ptr, arg_regs, calldescr, dst);
        self.call_descr_to_call_target.insert(calldescr_idx, target);
    }

    // ── Slice 4 Slice 1b: typed policy variants (dormant) ──
    //
    // Mirrors the void-family policy wrappers at lines 1772-1878.  Each
    // call_<policy>_<result>_canonical_via_target seeds the appropriate
    // EffectInfo and routes through the typed `_with_effect_info` body
    // (which selects the right `R_X / IR_X / IRF_X` opcode triple per
    // result kind).  RPython has no separate bytecode for these
    // policies — the policy is carried by `calldescr.extra_info`,
    // matching `pyjitpl.py do_residual_call`.

    /// Emit a canonical `residual_call_*_i` whose calldescr carries
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`.
    #[allow(dead_code)]
    pub fn call_may_force_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            // PyPy `call.py:288-289 if virtualizable_analyzer.analyze(op):`
            // selects `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, fed through
            // `effectinfo_from_writeanalyze` with the analyzer-empty
            // (`graphanalyze.py:60 bottom_result()`) default. The
            // resulting EI is the dedicated FVOV slot — distinct from
            // `MOST_GENERAL`/RandomEffects which is only the
            // `randomeffects_analyzer` branch (`call.py:282-283`).
            // Routing to `MOST_GENERAL` over-invalidates the heap cache
            // via `has_random_effects() → clean_caches` PyPy reserves
            // for genuinely-random callees.
            crate::call_descr::forces_virtual_or_virtualizable_effect_info(),
        );
    }

    /// Emit a canonical `residual_call_*_i` whose calldescr carries the
    /// release-gil marker.  `resolve_call_release_gil_target` fills
    /// `realfuncaddr` from the resolved `target.concrete_ptr`; the
    /// `(1, 0)` seed flips `is_call_release_gil()` for the resolver to
    /// pick up.
    #[allow(dead_code)]
    pub fn call_release_gil_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                // effectinfo.py:149-155: `EF_RANDOM_EFFECTS` keeps every
                // readonly/write descr set as `None`; spread MOST_GENERAL
                // for the wildcard rather than `default_effect_info()`'s
                // saturated `Some(vec![0xff; 8])` bitstrings.
                // `(1, 0)` is the unresolved sentinel — the inner
                // `emit_canonical_call_*_via_target` helper looks up the
                // `JitCallTarget` from `descrs[fn_ptr_idx]` and calls
                // `resolve_call_release_gil_target` to substitute both
                // the real `_call_aroundstate_target_[0]` (`rffi.py:228`)
                // address and the wrapper's `save_err` flag bits.
                call_release_gil_target: (1, 0),
                ..majit_ir::descr::EffectInfo::MOST_GENERAL
            },
        );
    }

    /// Emit a canonical `residual_call_*_i` whose calldescr carries
    /// `EF_LOOPINVARIANT`.  RPython `codewriter/call.py:249-251
    /// getcalldescr` rejects loop-invariant calls with non-void args.
    #[allow(dead_code)]
    pub fn call_loopinvariant_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// Ref-result sibling of [`Self::call_may_force_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_may_force_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            // PyPy `call.py:288-289 if virtualizable_analyzer.analyze(op):`
            // selects `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, fed through
            // `effectinfo_from_writeanalyze` with the analyzer-empty
            // (`graphanalyze.py:60 bottom_result()`) default. The
            // resulting EI is the dedicated FVOV slot — distinct from
            // `MOST_GENERAL`/RandomEffects which is only the
            // `randomeffects_analyzer` branch (`call.py:282-283`).
            // Routing to `MOST_GENERAL` over-invalidates the heap cache
            // via `has_random_effects() → clean_caches` PyPy reserves
            // for genuinely-random callees.
            crate::call_descr::forces_virtual_or_virtualizable_effect_info(),
        );
    }

    /// Ref-result sibling of [`Self::call_loopinvariant_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_loopinvariant_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// Float-result sibling of [`Self::call_may_force_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_may_force_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            // PyPy `call.py:288-289 if virtualizable_analyzer.analyze(op):`
            // selects `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`, fed through
            // `effectinfo_from_writeanalyze` with the analyzer-empty
            // (`graphanalyze.py:60 bottom_result()`) default. The
            // resulting EI is the dedicated FVOV slot — distinct from
            // `MOST_GENERAL`/RandomEffects which is only the
            // `randomeffects_analyzer` branch (`call.py:282-283`).
            // Routing to `MOST_GENERAL` over-invalidates the heap cache
            // via `has_random_effects() → clean_caches` PyPy reserves
            // for genuinely-random callees.
            crate::call_descr::forces_virtual_or_virtualizable_effect_info(),
        );
    }

    /// Float-result sibling of [`Self::call_release_gil_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_release_gil_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                // effectinfo.py:149-155: `EF_RANDOM_EFFECTS` keeps every
                // readonly/write descr set as `None`; spread MOST_GENERAL
                // for the wildcard rather than `default_effect_info()`'s
                // saturated `Some(vec![0xff; 8])` bitstrings.
                // `(1, 0)` is the unresolved sentinel — the inner
                // `emit_canonical_call_*_via_target` helper looks up the
                // `JitCallTarget` from `descrs[fn_ptr_idx]` and calls
                // `resolve_call_release_gil_target` to substitute both
                // the real `_call_aroundstate_target_[0]` (`rffi.py:228`)
                // address and the wrapper's `save_err` flag bits.
                call_release_gil_target: (1, 0),
                ..majit_ir::descr::EffectInfo::MOST_GENERAL
            },
        );
    }

    /// Float-result sibling of [`Self::call_loopinvariant_int_canonical_via_target`].
    #[allow(dead_code)]
    pub fn call_loopinvariant_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        assert!(
            arg_regs.is_empty(),
            "arguments not supported for loop-invariant function!",
        );
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            majit_ir::descr::EffectInfo {
                extraeffect: majit_ir::descr::ExtraEffect::LoopInvariant,
                ..majit_ir::descr::EffectInfo::default()
            },
        );
    }

    /// Slice 4: low-level int-result direct entry — sibling of
    /// [`Self::residual_call_void_canonical_typed_args`].  Skips the
    /// `JitCallTarget` resolution and registers a self-bridging
    /// `(funcptr, funcptr)` pair (mirrors void at line 1645).
    #[allow(dead_code)]
    pub fn residual_call_int_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_I,
                jitcode::insns::BC_RESIDUAL_CALL_IR_I,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_I,
            ),
            funcptr,
            args,
            calldescr,
            dst,
            JitArgKind::Int,
        );
        let funcptr = funcptr as *const ();
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::new(funcptr, funcptr));
    }

    #[allow(dead_code)]
    pub fn residual_call_ref_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed(
            (
                jitcode::insns::BC_RESIDUAL_CALL_R_R,
                jitcode::insns::BC_RESIDUAL_CALL_IR_R,
                jitcode::insns::BC_RESIDUAL_CALL_IRF_R,
            ),
            funcptr,
            args,
            calldescr,
            dst,
            JitArgKind::Ref,
        );
        let funcptr = funcptr as *const ();
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::new(funcptr, funcptr));
    }

    #[allow(dead_code)]
    pub fn residual_call_float_canonical_typed_args(
        &mut self,
        funcptr: i64,
        args: &[JitCallArg],
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
        dst: u16,
    ) {
        let calldescr_idx = self.emit_canonical_call_typed_irf_f(funcptr, args, calldescr, dst);
        let funcptr = funcptr as *const ();
        self.call_descr_to_call_target
            .insert(calldescr_idx, JitCallTarget::new(funcptr, funcptr));
    }

    pub fn call_assembler_void_args(&mut self, target_idx: u16, arg_regs: &[u16]) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_void_typed_args(target_idx, &args);
    }

    pub fn call_assembler_void_typed_args(&mut self, target_idx: u16, arg_regs: &[JitCallArg]) {
        self.call_assembler_void_like(jitcode::insns::BC_CALL_ASSEMBLER_VOID, target_idx, arg_regs);
    }

    pub fn call_assembler_int(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_int_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_int_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_int_like(
            jitcode::insns::BC_CALL_ASSEMBLER_INT,
            target_idx,
            arg_regs,
            dst,
        );
    }

    /// Parity #14 Slice C.4: Pure sibling of
    /// [`Self::residual_call_int_canonical_via_target`].  Emits the
    /// canonical `BC_RESIDUAL_CALL_{R,IR,IRF}_I` opcode with the
    /// calldescr's `extra_info` set to `ELIDABLE_CAN_RAISE`
    /// (`call_descr::ELIDABLE_EFFECT_INFO`).  The canonical walker
    /// (`majit-metainterp/src/pyjitpl/dispatch.rs` Slice C.1) reads
    /// `effectinfo.check_is_elidable()` and routes the result through
    /// `record_result_of_call_pure` mirroring `pyjitpl.py:2111-2115`,
    /// retiring the legacy `BC_CALL_PURE_INT`-specific code path.
    ///
    /// `_can_raise` is the conservative default — emits a trailing
    /// `GUARD_NO_EXCEPTION` because `effectinfo.check_can_raise(False)`
    /// is true for `extraeffect ≥ 3`. Callers that have classified
    /// the callee per `call.py:292-299 _canraise(op)` should prefer
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`] (no
    /// guard) or
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`] (guard
    /// retained, distinguished metadata).
    pub fn call_pure_int_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — `call.py:299 getcalldescr`'s
    /// `else` branch (`_canraise(op) == False`). The calldescr's
    /// `extraeffect == 0` makes `effectinfo.check_can_raise(False)`
    /// false, so the canonical walker records `CALL_PURE_*` *without*
    /// the trailing `GUARD_NO_EXCEPTION` (`pyjitpl.py:2126`).
    pub fn call_pure_int_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — `call.py:295 getcalldescr`'s
    /// `cr == "mem"` branch. Same dispatch as `_can_raise` (`extraeffect
    /// == 3` clears `check_can_raise(False)`'s gate at the boundary)
    /// but distinguishes memory-only failure modes for the optimizer.
    pub fn call_pure_int_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_int_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        );
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
        self.call_cond_like(
            jitcode::insns::BC_COND_CALL_VOID,
            fn_ptr_idx,
            cond_reg,
            typed_args,
        );
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
            jitcode::insns::BC_COND_CALL_VALUE_INT,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
            'i',
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
            jitcode::insns::BC_COND_CALL_VALUE_REF,
            fn_ptr_idx,
            value_reg,
            typed_args,
            dst,
            'r',
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
            jitcode::insns::BC_RECORD_KNOWN_RESULT_INT,
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
            jitcode::insns::BC_RECORD_KNOWN_RESULT_REF,
            fn_ptr_idx,
            result_reg,
            typed_args,
        );
    }

    fn call_cond_like(&mut self, bc: u8, fn_ptr_idx: u16, first_reg: u16, args: &[JitCallArg]) {
        self.start_instr(bc);
        self.push_u16(first_reg);
        self.push_u16(fn_ptr_idx);
        let arg_count = args.len();
        assert!(
            arg_count < 256,
            "conditional_call arg list length {arg_count} exceeds u8 byte encoding"
        );
        self.push_u8(arg_count as u8);
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
        result_kind: char,
    ) {
        self.start_instr(bc);
        self.push_u16(value_reg);
        self.push_u16(fn_ptr_idx);
        let arg_count = args.len();
        assert!(
            arg_count < 256,
            "conditional_call_value arg list length {arg_count} exceeds u8 byte encoding"
        );
        self.push_u8(arg_count as u8);
        for arg in args {
            self.push_u8(arg.kind as u8);
        }
        for arg in args {
            self.push_u16(arg.reg);
        }
        self.push_u16(dst);
        self.record_resulttype(result_kind);
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

    /// Parity #14 Slice C.4: see
    /// [`Self::call_pure_int_canonical_via_target`] for the rationale.
    pub fn call_pure_ref_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`].
    pub fn call_pure_ref_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`].
    pub fn call_pure_ref_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_ref_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
        );
    }

    // call_release_gil_ref / _typed intentionally absent:
    // resoperation.py:1243-1244 (`# no such thing`) excludes
    // CALL_RELEASE_GIL_R, so emitting BC_CALL_RELEASE_GIL_REF would
    // record an IR op the optimizer/backend cannot consume.

    pub fn call_assembler_ref(&mut self, target_idx: u16, arg_regs: &[u16], dst: u16) {
        let args: Vec<JitCallArg> = arg_regs.iter().copied().map(JitCallArg::int).collect();
        self.call_assembler_ref_typed(target_idx, &args, dst);
    }

    pub fn call_assembler_ref_typed(&mut self, target_idx: u16, arg_regs: &[JitCallArg], dst: u16) {
        self.call_assembler_ref_like(
            jitcode::insns::BC_CALL_ASSEMBLER_REF,
            target_idx,
            arg_regs,
            dst,
        );
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

    /// Parity #14 Slice C.4: see
    /// [`Self::call_pure_int_canonical_via_target`] for the rationale.
    pub fn call_pure_float_canonical_via_target(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_CANNOT_RAISE` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_cannot_raise`].
    pub fn call_pure_float_canonical_via_target_cannot_raise(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_CANNOT_RAISE_EFFECT_INFO,
        );
    }

    /// `EF_ELIDABLE_OR_MEMORYERROR` sibling — see
    /// [`Self::call_pure_int_canonical_via_target_or_memerror`].
    pub fn call_pure_float_canonical_via_target_or_memerror(
        &mut self,
        fn_ptr_idx: u16,
        arg_regs: &[JitCallArg],
        dst: u16,
    ) {
        self.residual_call_float_canonical_via_target_with_effect_info(
            fn_ptr_idx,
            arg_regs,
            dst,
            crate::call_descr::ELIDABLE_OR_MEMERROR_EFFECT_INFO,
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
        self.call_assembler_float_like(
            jitcode::insns::BC_CALL_ASSEMBLER_FLOAT,
            target_idx,
            arg_regs,
            dst,
        );
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

    /// `add_fn_ptr` variant carrying a per-callee
    /// [`crate::call_descr::EffectInfoSlot`] classification
    /// (`call.py:282-303 getcalldescr`'s `extraeffect` selection).
    /// Producers that statically know the helper's
    /// `_canraise` / `_elidable_function_` / `_jit_loop_invariant_`
    /// flags pick the matching slot; the dispatcher then threads the
    /// slot into `make_call_descr_from_target_slot` so the recorded
    /// trace descr carries the right `EffectInfo`.
    pub fn add_fn_ptr_with_slot(
        &mut self,
        ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
    ) -> u16 {
        self.add_call_target_with_slot(ptr, ptr, slot)
    }

    /// Append a function target descriptor and return its runtime
    /// `descrs` index. Mirrors RPython `Assembler._encode_descr(calldescr)`
    /// (assembler.py:140) where the 2-byte operand downstream resolves
    /// to `self.descrs[idx]` at dispatch time. pyre dedups identical
    /// `(trace_ptr, concrete_ptr)` pairs to match
    /// `Assembler._encode_descr` memoisation.
    pub fn add_call_target(&mut self, trace_ptr: *const (), concrete_ptr: *const ()) -> u16 {
        self.add_call_target_with_slot(
            trace_ptr,
            concrete_ptr,
            crate::call_descr::EffectInfoSlot::CanRaise,
        )
    }

    /// `add_call_target` variant carrying a per-callee
    /// [`crate::call_descr::EffectInfoSlot`] classification.  Dedup
    /// matches the full `(trace_ptr, concrete_ptr, slot)` triple so the
    /// same fn pointer registered with two different slots produces two
    /// distinct entries; in practice the same helper is registered with
    /// a single classification and the dedup matches the `add_call_target`
    /// path verbatim.  `save_err` defaults to `0` (`RFFI_ERR_NONE`,
    /// `rffi.py:80`); release-gil callees use [`add_call_target_with_save_err`]
    /// to thread the wrapper's `_call_aroundstate_target_[1]`
    /// (`rffi.py:228`) into the dedup key.
    pub fn add_call_target_with_slot(
        &mut self,
        trace_ptr: *const (),
        concrete_ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
    ) -> u16 {
        self.add_call_target_with_save_err(trace_ptr, concrete_ptr, slot, 0)
    }

    /// `add_call_target_with_slot` variant for release-gil callees:
    /// records the wrapper callable's
    /// `_call_aroundstate_target_ = (funcptr, save_err)` decoration
    /// (`rffi.py:228`).  The `(trace_ptr, concrete_ptr, slot,
    /// save_err)` tuple is the dedup key — same callee registered with
    /// two different `save_err` values produces two distinct entries
    /// because the recorded `EffectInfo.call_release_gil_target`
    /// differs.
    pub fn add_call_target_with_save_err(
        &mut self,
        trace_ptr: *const (),
        concrete_ptr: *const (),
        slot: crate::call_descr::EffectInfoSlot,
        save_err: i32,
    ) -> u16 {
        let target = JitCallTarget::with_save_err(trace_ptr, concrete_ptr, slot, save_err);
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

    fn add_vable_field_descr(&mut self, field_idx: u16) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::VableField {
            index: field_idx as usize,
        })
    }

    fn add_vable_array_field_descr(&mut self, array_idx: u16) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::VableArray {
            index: array_idx as usize,
        })
    }

    fn add_vable_array_descr(
        &mut self,
        item_type: majit_ir::value::Type,
        is_item_signed: bool,
    ) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Array {
            base_size: std::mem::size_of::<usize>(),
            itemsize: 8,
            len_offset: Some(0),
            type_id: 0,
            item_type,
            is_array_of_pointers: matches!(item_type, majit_ir::value::Type::Ref),
            is_array_of_structs: false,
            is_item_signed,
            ei_index: u32::MAX,
            // vable array slots are per-vinfo with no source-level
            // ARRAY type spelling; distinct slots are disambiguated by
            // the parent `VableArray { index }` variant.
            array_type_id: None,
            interior_fields: Vec::new(),
        })
    }

    /// Append a `CanonicalBhDescr::Call { calldescr }` entry to the descrs
    /// pool and return its index. Used by canonical `residual_call_*` /
    /// `call_*` emit paths that need a `d` argcode descriptor (RPython
    /// `assembler.py:197-207` `_encode_descr(calldescr)`).
    ///
    /// Slice 0 of `pyre-call-family-canonical-migration.md`: helper only.
    /// Slice 1 emits via this from the migrated emit sites.
    ///
    /// PRE-EXISTING-ADAPTATION: dedup is intentionally NOT done for the
    /// `Call` variant. RPython's `descr.py:660-668 _key_for_caching`
    /// dedups calldescrs on `(arg_classes, RESULT_ERASED, ffi_flags,
    /// extrainfo)` because the funcptr lives in `op.args[0]` separately
    /// from the descr. Pyre's adapter records a sidetable
    /// `JitCodeExecState.call_descr_to_call_target` keyed by the descr
    /// slot, so dedup'ing two distinct callees that share a signature
    /// would cause the second emit's `(trace_ptr, concrete_ptr)` pair to
    /// silently overwrite the first's. The convergence path is to lift
    /// the sidetable onto the funcptr int-const slot once trace_ptr and
    /// concrete_ptr unify; until then each emit gets a fresh descr slot.
    pub fn add_call_descr(
        &mut self,
        calldescr: majit_translate::jit_codewriter::jitcode::BhCallDescr,
    ) -> u16 {
        self.add_bh_descr(CanonicalBhDescr::Call { calldescr })
    }

    fn add_bh_descr(&mut self, descr: CanonicalBhDescr) -> u16 {
        for (idx, entry) in self.descrs.iter().enumerate() {
            if let RuntimeBhDescr::Descr(existing) = entry {
                if canonical_bh_descr_eq(existing, &descr) {
                    return idx as u16;
                }
            }
        }
        let idx = self.descrs.len() as u16;
        self.descrs.push(RuntimeBhDescr::Descr(descr));
        idx
    }

    pub fn finish(mut self) -> JitCode {
        self.flush_pending_resulttype();
        self.patch_labels();
        self.patch_const_refs();
        self.patch_const_u8_refs();
        // RPython `jitcode.py:47 self._resulttypes = resulttypes`.
        // Upstream `assembler.py:217-219` records the result-kind
        // char at the end-of-instruction position (`len(self.code)`
        // after operands, before the next instruction's opcode) for
        // every instruction whose argcodes contain `>X`.  Consumed
        // by `pyjitpl.py:264 make_result_of_lastop` in non-translated
        // builds as a debug-only type check:
        //
        // ```python
        // assert typeof[self.jitcode._resulttypes[self.pc]] == got_type
        // ```
        //
        // pyre fires the same assertion in
        // `MIFrame::make_result_of_lastop` (`frame.rs`).  Each
        // typed-result emit helper (`call_*_like`,
        // `call_assembler_*_like`, `inline_call_typed`) records the
        // kind via `self.record_resulttype(...)` as its LAST step.
        // Generic `write_insn("...>X")` stores `X` in
        // `pending_resulttype` and `start_instr` / `finish` flush it
        // after the operand bytes have been pushed.  The map handed
        // off to `JitCodeBody::resulttypes` is therefore keyed by
        // every typed-result instruction's end-of-instruction PC.
        let resulttypes = Some(self.resulttypes);
        // Stage 1 audit (bytecode encoding unification —
        // .claude/plans/TODO-bytecode-encoding-unification.md):
        // RPython enforces two distinct ceilings:
        //   * `jitcode.py:36 assert num_regs_i < 256 and ...` — the
        //     stored `c_num_regs_*` is a single char.
        //   * `assembler.py:132-133 val = count_regs[kind] +
        //     len(constants) - 1; assert 0 <= val < 256` — the last
        //     constant-slot index in the unified register-plus-const
        //     namespace must fit in one byte, so the total count
        //     `num_regs + len(constants) <= 256` (last index 255).
        // pyre legacy assembler still emits u16 operands, so this hook
        // gates the migration: if any production trace exceeds the
        // canonical ceiling per kind, the migration plan must grow a
        // spill mechanism before continuing.
        let total_i = self.num_regs_i as usize + self.constants_i.len();
        let total_r = self.num_regs_r as usize + self.constants_r.len();
        let total_f = self.num_regs_f as usize + self.constants_f.len();
        if crate::majit_log_enabled() {
            eprintln!(
                "[bcenc-audit] {:?} regs i={} r={} f={} consts i={} r={} f={} total i={} r={} f={}",
                self.name,
                self.num_regs_i,
                self.num_regs_r,
                self.num_regs_f,
                self.constants_i.len(),
                self.constants_r.len(),
                self.constants_f.len(),
                total_i,
                total_r,
                total_f,
            );
        }
        // RPython `jitcode.py:36` ceiling: `num_regs_X < 256` per kind.
        assert!(
            (self.num_regs_i as usize) < 256
                && (self.num_regs_r as usize) < 256
                && (self.num_regs_f as usize) < 256,
            "jitcode {:?} exceeds RPython jitcode.py:36 num_regs ceiling \
             (num_regs i={} r={} f={})",
            self.name,
            self.num_regs_i,
            self.num_regs_r,
            self.num_regs_f,
        );
        // RPython `assembler.py:132-133` ceiling: last unified slot
        // index `num_regs + len(consts) - 1 < 256`, i.e. `total <= 256`.
        assert!(
            total_i <= 256 && total_r <= 256 && total_f <= 256,
            "jitcode {:?} exceeds canonical 1-byte register pool \
             (i_total={total_i} r_total={total_r} f_total={total_f}); \
             see TODO-bytecode-encoding-unification.md Stage 1.3",
            self.name,
        );
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
            // RPython `assembler.py:271-281 make_jitcode(startpoints=
            // self.startpoints, alllabels=self.alllabels, ...)` —
            // assembled jitcodes always carry the recorded set, never
            // `None`. Wrap in `Some(...)` so the upstream None sentinel
            // is reserved for hand-built helper jitcodes that bypass the
            // builder (matching `JitCode.setup(..., startpoints=None,
            // alllabels=None)` defaults at jitcode.py:24).
            startpoints: Some(self.startpoints),
            alllabels: Some(self.alllabels),
            resulttypes,
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
            call_descr_to_call_target: self.call_descr_to_call_target,
            // Propagate the captured `BC_JIT_MERGE_POINT(_C)` opcode
            // offset so `register_dispatch_jitcode` validates the payload
            // by direct seek instead of byte-stream scan
            // (blackhole.py:107-156 argcode-based decode parity).
            jit_merge_point_offset: self.jit_merge_point_offset,
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

    fn push_reg_u8(&mut self, reg: u16, context: &'static str) {
        assert!(
            reg <= u8::MAX as u16,
            "{context}: register {reg} does not fit canonical u8 operand"
        );
        self.push_u8(reg as u8);
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
        self.pending_resulttype = key
            .split_once('>')
            .and_then(|(_, suffix)| suffix.chars().next())
            .filter(|kind| matches!(kind, 'i' | 'r' | 'f'));
    }

    /// Record the current bytecode offset as an instruction start
    /// (RPython `assembler.py:200-208` writes `self.startpoints.add(pos)`
    /// just before each opcode-byte push) and emit the opcode byte.
    /// Every helper that pushes a `BC_*` opcode goes through this so
    /// `JitCode.get_live_vars_info` (RPython `jitcode.py:85-90`) can
    /// fire its non-translated `assert pc in self._startpoints` check.
    fn start_instr(&mut self, opcode: u8) {
        self.flush_pending_resulttype();
        self.startpoints.insert(self.code.len());
        self.push_u8(opcode);
    }

    /// RPython `assembler.py:217-219`:
    ///
    /// ```python
    /// if '>' in argcodes:
    ///     assert argcodes.index('>') == len(argcodes) - 2
    ///     self.resulttypes[len(self.code)] = argcodes[-1]
    /// ```
    ///
    /// Called by typed-result emit helpers as their LAST step (after
    /// every operand byte has been pushed) so `self.code.len()`
    /// matches the upstream `len(self.code)` after-operands-before-
    /// next-instruction value.  Consumed by
    /// `MIFrame::make_result_of_lastop` (RPython `pyjitpl.py:264`)
    /// where `frame.pc` has already advanced past the instruction.
    fn record_resulttype(&mut self, kind: char) {
        self.resulttypes.insert(self.code.len(), kind);
    }

    fn flush_pending_resulttype(&mut self) {
        if let Some(kind) = self.pending_resulttype.take() {
            self.record_resulttype(kind);
        }
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
        self.record_resulttype('i');
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
        self.record_resulttype('r');
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
        self.record_resulttype('i');
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
        self.record_resulttype('r');
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
        self.record_resulttype('f');
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
        self.record_resulttype('f');
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
    /// `jit_merge_point/iIRFIRF` jdindex bytes (`@arguments("i")`,
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

/// pyjitpl.py:3675 `effectinfo.call_release_gil_target` parity.
///
/// PyPy populates `(realfuncaddr, saveerr)` at descr creation time:
/// `codewriter/call.py:252-258` reads `_call_aroundstate_target_` off
/// the wrapper callable and writes `(tgt_func, tgt_saveerr)` into the
/// slot — the wrapper at `direct_call`'s `args[0]` and the real GIL-
/// release target are intentionally distinct values.
///
/// Pyre's `#[jit_interp]` macro `release_gil_*` policy declarations
/// (`majit-macros/src/jit_interp/mod.rs:226-251`) carry no `saveerr`
/// attribute and no separate wrapper-vs-real-address distinction —
/// the macro emits a wrapper where `func_ptr` IS the C address.  The
/// emit-side wrappers seed `call_release_gil_target: (1, 0)` purely
/// to flip `EffectInfo::is_call_release_gil()` (`effectinfo.rs:292-295`,
/// `effectinfo.py:255-257`) on while the real address is unknown until
/// `descrs[fn_ptr_idx]` resolves.  This helper substitutes the
/// resolved `target.concrete_ptr` into that sentinel slot so the
/// descr's IR carries `(real_addr, saveerr=0)`.
///
/// Sentinel-only override: any descr whose
/// `call_release_gil_target.0` is already a real address (≠ sentinel
/// `1`) is left untouched, mirroring PyPy's "descr already carries
/// `(tgt_func, saveerr)` from the analyzer" structure at
/// `call.py:252-258`.  Today the only producer of explicit targets is
/// `trace_ctx::call_release_gil_{int,float}_typed`
/// (`trace_ctx.rs:3795, 3824`) which bypasses this resolver, but
/// keeping the override sentinel-conditional preserves the upstream
/// invariant for any future analyzer-driven descr.
///
/// Resolve the `(realfuncaddr, save_err)` pair on a release-gil EI.
///
/// `effectinfo.py:114, 197 call_release_gil_target = (target_fn_addr,
/// save_err)` mirrors `rffi.py:228 _call_aroundstate_target_ =
/// (funcptr, save_err)` — both halves come from the
/// `@llexternal(... save_err=...)` registration on the wrapper.  The
/// outer `call_release_gil_*_canonical_via_target` sites lack a
/// resolved `JitCallTarget`, so they emit `(1, 0)` as the unresolved
/// sentinel; this helper, called from
/// `emit_canonical_call_*_via_target` with the descr-resolved target,
/// substitutes both halves verbatim.  The `save_err` argument carries
/// the `JitCallTarget::save_err` field set by the macro DSL's
/// `#[jit_release_gil(save_err = N)]` attribute (`rffi.py:62-71` flag
/// bits, default `RFFI_ERR_NONE = 0`).
fn resolve_call_release_gil_target(
    mut effect_info: majit_ir::descr::EffectInfo,
    realfuncaddr: *const (),
    save_err: i32,
) -> majit_ir::descr::EffectInfo {
    // effectinfo.rs:292 is_call_release_gil checks `tgt_func != 0`, so
    // skip the substitution for non-release-gil callers (the slot
    // carries `_NO_CALL_RELEASE_GIL_TARGET = (0, 0)` for them).
    // Match the sentinel `1` exclusively so descrs with an already-
    // resolved `(tgt_func, saveerr)` from `_call_aroundstate_target_`
    // (`call.py:252-258`) are preserved.
    if effect_info.call_release_gil_target.0 == 1 {
        effect_info.call_release_gil_target = (realfuncaddr as usize as u64, save_err);
    }
    effect_info
}

fn canonical_bh_descr_eq(lhs: &CanonicalBhDescr, rhs: &CanonicalBhDescr) -> bool {
    match (lhs, rhs) {
        (
            CanonicalBhDescr::VableField { index: lhs },
            CanonicalBhDescr::VableField { index: rhs },
        ) => lhs == rhs,
        (
            CanonicalBhDescr::VableArray { index: lhs },
            CanonicalBhDescr::VableArray { index: rhs },
        ) => lhs == rhs,
        (
            CanonicalBhDescr::Array {
                base_size: lhs_base_size,
                itemsize: lhs_itemsize,
                len_offset: lhs_len_offset,
                type_id: lhs_type_id,
                item_type: lhs_item_type,
                is_array_of_pointers: lhs_is_array_of_pointers,
                is_array_of_structs: lhs_is_array_of_structs,
                is_item_signed: lhs_is_item_signed,
                ei_index: _,
                array_type_id: lhs_array_type_id,
                interior_fields: lhs_interior_fields,
            },
            CanonicalBhDescr::Array {
                base_size: rhs_base_size,
                itemsize: rhs_itemsize,
                len_offset: rhs_len_offset,
                type_id: rhs_type_id,
                item_type: rhs_item_type,
                is_array_of_pointers: rhs_is_array_of_pointers,
                is_array_of_structs: rhs_is_array_of_structs,
                is_item_signed: rhs_is_item_signed,
                ei_index: _,
                array_type_id: rhs_array_type_id,
                interior_fields: rhs_interior_fields,
            },
        ) => {
            // `ei_index` is intentionally NOT part of the identity
            // tuple — upstream `gccache._cache_array[ARRAY_OR_STRUCT]`
            // (`descr.py:348-360`) keys on the lltype itself, and
            // `compute_bitstrings` (`effectinfo.py:465`) later assigns
            // the index slot as a derived attribute that multiple
            // descrs are free to share.
            //
            // `array_type_id` joins the identity tuple as the
            // codewriter lltype-identity proxy so two ARRAY entries
            // that disagree on the Rust type spelling stay on distinct
            // canonical slots even when their numeric `type_id`
            // collides (default `0`).
            lhs_base_size == rhs_base_size
                && lhs_itemsize == rhs_itemsize
                && lhs_len_offset == rhs_len_offset
                && lhs_type_id == rhs_type_id
                && lhs_item_type == rhs_item_type
                && lhs_is_array_of_pointers == rhs_is_array_of_pointers
                && lhs_is_array_of_structs == rhs_is_array_of_structs
                && lhs_is_item_signed == rhs_is_item_signed
                && lhs_array_type_id == rhs_array_type_id
                && lhs_interior_fields == rhs_interior_fields
        }
        // PRE-EXISTING-ADAPTATION: `Call` variant intentionally falls
        // through `_ => false`. See `add_call_descr`'s docstring — pyre's
        // per-callsite `JitCodeExecState.call_descr_to_call_target`
        // sidetable is keyed by descr slot, so dedup'ing two distinct
        // callees that share a signature would clobber the sidetable
        // entry. The convergence path requires lifting the sidetable
        // onto the funcptr int-const slot first.
        _ => false,
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

    fn assert_resulttype_after(emit: impl FnOnce(&mut JitCodeBuilder), kind: char) {
        let mut builder = JitCodeBuilder::new();
        emit(&mut builder);
        let pc = builder.current_pos();
        let jitcode = builder.finish();
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&pc).copied()),
            Some(kind)
        );
    }

    fn assert_no_resulttype_after(emit: impl FnOnce(&mut JitCodeBuilder)) {
        let mut builder = JitCodeBuilder::new();
        emit(&mut builder);
        let pc = builder.current_pos();
        let jitcode = builder.finish();
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&pc).copied()),
            None
        );
    }

    #[test]
    fn typed_vable_helpers_record_resulttypes_at_end_pc() {
        // RPython assembler.py:217-219 records `argcodes[-1]` at
        // `len(code)` after all operands are emitted. These helper-side
        // adapters are not canonical argcode layouts, but their result
        // byte is still the last operand consumed by make_result_of_lastop.
        let vr = 0;
        assert_resulttype_after(|b| b.vable_getfield_int_with_base(0, vr, 1), 'i');
        assert_resulttype_after(|b| b.vable_getfield_ref_with_base(0, vr, 1), 'r');
        assert_resulttype_after(|b| b.vable_getfield_float_with_base(0, vr, 1), 'f');
        assert_resulttype_after(|b| b.vable_getarrayitem_int_with_base(0, vr, 1, 2), 'i');
        assert_resulttype_after(|b| b.vable_getarrayitem_ref_with_base(0, vr, 1, 2), 'r');
        assert_resulttype_after(|b| b.vable_getarrayitem_float_with_base(0, vr, 1, 2), 'f');
        assert_resulttype_after(|b| b.vable_arraylen_with_base(0, vr, 1), 'i');
    }

    #[test]
    fn canonical_vable_field_emit_uses_base_reg_and_descr_pool() {
        let mut builder = JitCodeBuilder::new();
        builder.vable_getfield_ref_with_base(2, 1, 3);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("getfield_vable_r/rd>r");
        assert_eq!(jitcode.code, vec![opcode, 1, 0, 0, 2]);
        assert!(matches!(
            &jitcode.exec.descrs[0],
            RuntimeBhDescr::Descr(CanonicalBhDescr::VableField { index: 3 })
        ));
        assert_eq!(
            jitcode
                .body()
                .resulttypes
                .as_ref()
                .and_then(|resulttypes| resulttypes.get(&5).copied()),
            Some('r')
        );
    }

    #[test]
    fn canonical_vable_array_emit_uses_two_descrs() {
        let mut builder = JitCodeBuilder::new();
        builder.vable_getarrayitem_ref_with_base(4, 1, 7, 2);
        let jitcode = builder.finish();
        let opcode = jitcode::insn_byte("getarrayitem_vable_r/ridd>r");
        assert_eq!(jitcode.code, vec![opcode, 1, 2, 0, 0, 1, 0, 4]);
        assert!(matches!(
            &jitcode.exec.descrs[0],
            RuntimeBhDescr::Descr(CanonicalBhDescr::VableArray { index: 7 })
        ));
        assert!(matches!(
            &jitcode.exec.descrs[1],
            RuntimeBhDescr::Descr(CanonicalBhDescr::Array {
                item_type: majit_ir::value::Type::Ref,
                ..
            })
        ));
    }

    #[test]
    fn add_call_descr_pushes_canonical_call_entry() {
        // Slice 0 of `pyre-call-family-canonical-migration.md` — verify
        // that `add_call_descr` puts a `BhDescr::Call { calldescr }` at the
        // returned pool index so canonical `residual_call_*_v` handlers
        // (`blackhole.rs:6886-6892`) can reach it via `read_descr` →
        // `as_calldescr()` (`majit-translate/src/jit_codewriter/jitcode.rs:1265`).
        let mut builder = JitCodeBuilder::new();
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "i".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let idx = builder.add_call_descr(calldescr);
        assert_eq!(idx, 0);
        let jitcode = builder.finish();
        let entry = &jitcode.exec.descrs[idx as usize];
        let bh_descr = entry.as_bh_descr().expect("call descr must be BhDescr");
        let cd = bh_descr.as_calldescr();
        assert_eq!(cd.arg_classes, "i");
        assert_eq!(cd.result_type, 'v');
    }

    #[test]
    fn residual_call_void_canonical_emits_irf_for_mixed_kinds() {
        // Slice 0e — `residual_call_void_canonical_typed_args` writes the
        // canonical byte layout that `blackhole.rs:6534 handler_residual_call_irf_v`
        // reads: 1B funcptr_reg + (countI:1 + regI×N) + (countR:1 + regR×M)
        // + (countF:1 + regF×K) + descr:2.
        let mut builder = JitCodeBuilder::new();
        // Reserve some live registers per kind so num_regs_X > 0 at finish().
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        builder.touch_call_arg(JitCallArg::float(1));
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "irf".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let funcptr = 0xDEAD_BEEFi64;
        let args = [
            JitCallArg::int(2),
            JitCallArg::reference(3),
            JitCallArg::float(1),
        ];
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(funcptr, &args, calldescr);
        let jitcode = builder.finish();
        // Opcode byte (start_instr) + funcptr_reg(1) + countI(1)+regI(1)
        // + countR(1)+regR(1) + countF(1)+regF(1) + descr(2) = 10 bytes.
        let bytes = &jitcode.code[start..start + 10];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_V);
        // funcptr_reg patched to num_regs_i + funcptr_const_idx (=0).
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        assert_eq!(bytes[6], 1); // countF
        assert_eq!(bytes[7], 1); // regF
        // descr u16 LE
        let descr_idx = (bytes[8] as u16) | ((bytes[9] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "irf");
    }

    #[test]
    fn residual_call_void_canonical_uses_r_variant_for_ref_only() {
        // No int / float args → BC_RESIDUAL_CALL_R_V; layout is
        // funcptr_reg(1) + countR(1) + regR×M + descr(2).
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::reference(5));
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "r".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(
            0x1234_5678,
            &[JitCallArg::reference(5)],
            calldescr,
        );
        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 6];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_R_V);
        // funcptr_reg post-regs slot.
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countR
        assert_eq!(bytes[3], 5); // regR
        // No int / float lists — descr immediately follows.
        let descr_idx = (bytes[4] as u16) | ((bytes[5] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(entry.as_bh_descr().unwrap().as_calldescr().arg_classes, "r");
    }

    #[test]
    fn residual_call_void_canonical_uses_ir_variant_when_no_floats() {
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "ir".to_string(),
            majit_ir::value::Type::Void,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_void_canonical_typed_args(
            0xCAFEi64,
            &[JitCallArg::int(2), JitCallArg::reference(3)],
            calldescr,
        );
        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 8];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IR_V);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        // No float list. descr 2 bytes.
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(
            entry.as_bh_descr().unwrap().as_calldescr().arg_classes,
            "ir"
        );
    }

    #[test]
    fn residual_call_int_canonical_emits_irf_with_dst_for_mixed_kinds() {
        // Slice 4 Slice 1a — `residual_call_int_canonical_typed_args`
        // writes the same byte layout as the void IRF case, plus a
        // trailing `dst:u8` byte (`>i` suffix in `wellknown_bh_insns`).
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_call_arg(JitCallArg::reference(3));
        builder.touch_call_arg(JitCallArg::float(1));
        builder.touch_reg(7); // dst slot
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "irf".to_string(),
            majit_ir::value::Type::Int,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_int_canonical_typed_args(
            0xDEAD_BEEFi64,
            &[
                JitCallArg::int(2),
                JitCallArg::reference(3),
                JitCallArg::float(1),
            ],
            calldescr,
            7,
        );
        let jitcode = builder.finish();
        // Opcode + funcptr_reg + countI + regI + countR + regR + countF
        // + regF + descr×2 + dst = 11 bytes.
        let bytes = &jitcode.code[start..start + 11];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_I);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 3); // regR
        assert_eq!(bytes[6], 1); // countF
        assert_eq!(bytes[7], 1); // regF
        let descr_idx = (bytes[8] as u16) | ((bytes[9] as u16) << 8);
        assert_eq!(bytes[10], 7); // dst
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "irf");
        assert_eq!(cd.result_type, 'i');
    }

    #[test]
    fn residual_call_float_canonical_always_uses_irf_even_for_int_only_args() {
        // Per `resoperation.py:1238-1248`, float-result residual_calls
        // only have an IRF form — the R / IR shapes are "no such thing".
        // `emit_canonical_call_typed_irf_f` therefore always emits all
        // three (count, regs) pairs, even when a list is empty, so the
        // canonical handler at `blackhole.rs:6644
        // handler_residual_call_irf_f` can read `read_list_i` /
        // `read_list_r` / `read_list_f` in sequence.
        let mut builder = JitCodeBuilder::new();
        builder.touch_call_arg(JitCallArg::int(2));
        builder.touch_float_reg(4); // dst float slot
        let calldescr = majit_translate::jit_codewriter::jitcode::BhCallDescr::from_signature(
            "i".to_string(),
            majit_ir::value::Type::Float,
            majit_ir::descr::EffectInfo::MOST_GENERAL,
        );
        let start = builder.current_pos();
        builder.residual_call_float_canonical_typed_args(
            0xCAFE_FACEi64,
            &[JitCallArg::int(2)],
            calldescr,
            4,
        );
        let jitcode = builder.finish();
        // Opcode(1) + funcptr_reg(1) + countI(1) + regI(1) + countR(1)
        // + countF(1) + descr(2) + dst(1) = 9 bytes for an int-only
        // float-returning call.
        let bytes = &jitcode.code[start..start + 9];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IRF_F);
        assert_eq!(bytes[1], jitcode.num_regs_i() as u8);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI
        assert_eq!(bytes[4], 0); // countR — no ref args
        assert_eq!(bytes[5], 0); // countF — no float args
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        assert_eq!(bytes[8], 4); // dst float reg
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        let cd = entry.as_bh_descr().unwrap().as_calldescr();
        assert_eq!(cd.arg_classes, "i");
        assert_eq!(cd.result_type, 'f');
    }

    #[test]
    fn residual_call_void_via_target_keeps_source_arg_classes() {
        let mut builder = JitCodeBuilder::new();
        let trace_ptr = 0x1111usize as *const ();
        let concrete_ptr = 0x2222usize as *const ();
        let fn_idx = builder.add_call_target(trace_ptr, concrete_ptr);
        let start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(
            fn_idx,
            &[JitCallArg::reference(1), JitCallArg::int(2)],
        );

        let jitcode = builder.finish();
        let bytes = &jitcode.code[start..start + 8];
        assert_eq!(bytes[0], jitcode::insns::BC_RESIDUAL_CALL_IR_V);
        assert_eq!(bytes[2], 1); // countI
        assert_eq!(bytes[3], 2); // regI: grouped before refs
        assert_eq!(bytes[4], 1); // countR
        assert_eq!(bytes[5], 1); // regR
        let descr_idx = (bytes[6] as u16) | ((bytes[7] as u16) << 8);
        let entry = &jitcode.exec.descrs[descr_idx as usize];
        assert_eq!(
            entry.as_bh_descr().unwrap().as_calldescr().arg_classes,
            "ri"
        );
        assert_eq!(
            jitcode.exec.call_descr_to_call_target.get(&descr_idx),
            Some(&JitCallTarget::new(trace_ptr, concrete_ptr))
        );
    }

    #[test]
    fn residual_call_target_bridge_is_keyed_per_calldescr() {
        let mut builder = JitCodeBuilder::new();
        let concrete_ptr = 0x3333usize as *const ();
        let first_idx = builder.add_call_target(0x4444usize as *const (), concrete_ptr);
        let second_idx = builder.add_call_target(0x5555usize as *const (), concrete_ptr);
        let first_start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(first_idx, &[]);
        let second_start = builder.current_pos();
        builder.residual_call_void_canonical_via_target(second_idx, &[]);

        let jitcode = builder.finish();
        let first_descr =
            (jitcode.code[first_start + 3] as u16) | ((jitcode.code[first_start + 4] as u16) << 8);
        let second_descr = (jitcode.code[second_start + 3] as u16)
            | ((jitcode.code[second_start + 4] as u16) << 8);
        assert_ne!(first_descr, second_descr);
        assert_eq!(
            jitcode
                .exec
                .call_descr_to_call_target
                .get(&first_descr)
                .unwrap()
                .trace_ptr,
            0x4444usize as *const ()
        );
        assert_eq!(
            jitcode
                .exec
                .call_descr_to_call_target
                .get(&second_descr)
                .unwrap()
                .trace_ptr,
            0x5555usize as *const ()
        );
    }

    #[test]
    fn typed_call_adapters_record_resulttypes_at_end_pc() {
        assert_resulttype_after(
            |b| b.conditional_call_value_ir_i_typed_args(0, 1, &[], 2),
            'i',
        );
        assert_resulttype_after(
            |b| b.conditional_call_value_ir_r_typed_args(0, 1, &[], 2),
            'r',
        );
        assert_resulttype_after(|b| b.call_assembler_float_typed(0, &[], 1), 'f');
    }

    #[test]
    fn typed_void_call_adapters_do_not_record_resulttypes() {
        assert_no_resulttype_after(|b| {
            let idx = b.add_fn_ptr(std::ptr::null());
            b.residual_call_void_canonical_via_target(idx, &[]);
        });
        assert_no_resulttype_after(|b| b.call_assembler_void_typed_args(0, &[]));
    }

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

    #[test]
    fn live_placeholder_with_triple_records_then_finalize_patches() {
        // B.3-B.4 deferred-patch round-trip: `live_placeholder_with_triple`
        // emits the same `live/<00 00>` shape as `live_placeholder` but
        // additionally captures the triple in `pending_live_triples`.
        // `finalize_liveness` then registers each triple via
        // `_register_liveness_offset` and rewrites the BC_LIVE 2-byte slot
        // to point at the dedup'd entry — equivalent to the in-line
        // `live(asm, ...)` encoding shape, deferred.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        let _ = builder.live_placeholder_with_triple(&[0, 1], &[2], &[]);
        // Pre-finalize: BC_LIVE byte + zero placeholder offset.
        assert_eq!(builder.code.len(), 1 + 2);
        assert_eq!(builder.code[1], 0);
        assert_eq!(builder.code[2], 0);
        assert_eq!(builder.pending_live_triples.len(), 1);

        builder.finalize_liveness(&mut asm);
        // Post-finalize: drained pending list, BC_LIVE slot patched to
        // point at the freshly-registered entry (offset 0 because it's
        // the first entry registered).
        assert!(builder.pending_live_triples.is_empty());
        assert_eq!(u16::from_le_bytes([builder.code[1], builder.code[2]]), 0);

        // A `live(asm, ...)` call with the same triple must dedup to the
        // same offset — no additional `all_liveness` growth.
        let pre_len = asm.all_liveness().len();
        let mut builder2 = JitCodeBuilder::new();
        builder2.live(&mut asm, &[0, 1], &[2], &[]);
        assert_eq!(asm.all_liveness().len(), pre_len);
        assert_eq!(u16::from_le_bytes([builder2.code[1], builder2.code[2]]), 0);
    }

    #[test]
    fn finalize_liveness_dedups_distinct_then_repeated_triples() {
        // Multiple pending triples: first distinct entries grow
        // all_liveness; repeats reuse offsets (assembler.py:235-238 dedup).
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();

        let pi0 = builder.live_placeholder_with_triple(&[0], &[], &[]);
        let pi1 = builder.live_placeholder_with_triple(&[5], &[], &[]);
        let pi2 = builder.live_placeholder_with_triple(&[0], &[], &[]); // repeat of pi0

        assert_eq!(builder.pending_live_triples.len(), 3);
        builder.finalize_liveness(&mut asm);

        let off0 = u16::from_le_bytes([builder.code[pi0], builder.code[pi0 + 1]]);
        let off1 = u16::from_le_bytes([builder.code[pi1], builder.code[pi1 + 1]]);
        let off2 = u16::from_le_bytes([builder.code[pi2], builder.code[pi2 + 1]]);

        assert_eq!(off0, 0, "first distinct entry sits at offset 0");
        assert_eq!(off1, 4, "second entry follows 3-header + 1 payload");
        assert_eq!(off2, off0, "repeat triple dedups to first entry");
    }

    #[test]
    fn finalize_liveness_is_idempotent_after_drain() {
        // Calling `finalize_liveness` twice must not double-patch (the
        // pending list is drained on first call; the second call is a
        // no-op).  Guards against accidental re-entry from caller flows
        // that loop or chain finalisation.
        let mut asm = Assembler::new();
        let mut builder = JitCodeBuilder::new();
        let patch = builder.live_placeholder_with_triple(&[3], &[], &[]);
        builder.finalize_liveness(&mut asm);
        let snapshot = builder.code.clone();
        let liveness_snapshot_len = asm.all_liveness().len();

        builder.finalize_liveness(&mut asm);
        assert_eq!(builder.code, snapshot, "second finalize is a no-op");
        assert_eq!(
            asm.all_liveness().len(),
            liveness_snapshot_len,
            "second finalize does not re-register"
        );
        // Sanity: patched offset is preserved.
        let off = u16::from_le_bytes([builder.code[patch], builder.code[patch + 1]]);
        assert_eq!(off, 0);
    }
}
