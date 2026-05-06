/// MIFrame and MIFrameStack — execution frames for jitcode interpretation.
///
/// RPython pyjitpl.py: class MIFrame holds jitcode reference, PC,
/// and three register arrays (int/ref/float). MIFrameStack manages
/// the call stack of nested inline calls.
use std::sync::Arc;

use majit_ir::{OpRef, Type};

use crate::constant_pool::ConstantPool;
use crate::jitcode::{JitArgKind, JitCode, read_u8, read_u16};
use crate::opencoder::{Box as OpBox, TraceRecordBuffer};
use crate::recorder::SnapshotTagged;

/// Map an int register (OpRef, concrete value) to an `OpBox`.
/// Constant OpRefs materialize as `ConstInt(value)`; real trace slots
/// materialize as `ResOp(opref.raw())`. Mirrors RPython's implicit
/// isinstance dispatch in `_encode(box)` when the caller passes
/// `self.registers_i[index]` (which is already a `ConstInt` /
/// `InputArg` / `AbstractResOp`).
#[inline]
fn register_to_box_int(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstInt(value)
    } else {
        OpBox::ResOp(opref.raw())
    }
}

/// Map a ref register (OpRef, concrete address) to an `OpBox`.
/// `value` stores the raw address of the GC ref (cast to i64 at write
/// time).
#[inline]
fn register_to_box_ref(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstPtr(value as u64)
    } else {
        OpBox::ResOp(opref.raw())
    }
}

/// Map a float register (OpRef, raw bits) to an `OpBox`.
/// `value` stores the bit-casted `f64` payload.
#[inline]
fn register_to_box_float(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstFloat(value as u64)
    } else {
        OpBox::ResOp(opref.raw())
    }
}

/// A single execution frame for jitcode bytecode.
///
/// RPython pyjitpl.py: class MIFrame
///
/// PRE-EXISTING-ADAPTATION: RPython holds `self.metainterp` as a
/// back-pointer to the owning MetaInterp; pyre takes `&mut MetaInterp`
/// as a parameter on call sites instead so that the borrow checker
/// allows `MetaInterp::framestack` to own a `Vec<MIFrame>` without
/// self-referential aliasing.  `jitcode` is held as `Arc<JitCode>` so
/// the frame can outlive any single function-scope borrow — this
/// matches the upstream model where frames live as long as
/// `MetaInterp.framestack` keeps them alive.
pub struct MIFrame {
    pub jitcode: Arc<JitCode>,
    pub pc: usize,
    pub code_cursor: usize,
    pub int_regs: Vec<Option<OpRef>>,
    pub int_values: Vec<Option<i64>>,
    pub ref_regs: Vec<Option<OpRef>>,
    pub ref_values: Vec<Option<i64>>,
    pub float_regs: Vec<Option<OpRef>>,
    pub float_values: Vec<Option<i64>>,
    pub inline_frame: bool,
    pub return_i: Option<usize>,
    pub return_r: Option<usize>,
    pub return_f: Option<usize>,
    /// pyjitpl.py `MIFrame.greenkey` — set when this frame is a
    /// recursive portal call (pyjitpl.py:80).
    pub greenkey: Option<u64>,
    /// pyjitpl.py:91 `self._result_argcode = 'v'`.
    ///
    /// Single-byte argcode of the *previous* opimpl's result type
    /// (`b'i'` / `b'r'` / `b'f'` / `b'v'`).  Updated by call-recording
    /// opimpls before they advance the pc; consulted by `_try_tco`
    /// (pyjitpl.py:1281) to decide whether the call's result type
    /// matches a following `*_return` opcode.  Initialized to `b'v'`
    /// because a fresh frame's "previous opimpl" is the implicit
    /// frame setup (returns void).
    pub _result_argcode: u8,
    /// Rust bytecode adaptation for `pyjitpl.py:186 ord(self.bytecode[self.pc - 1])`.
    ///
    /// RPython encodes the result register as the last byte before the
    /// post-call LIVE marker, so `get_list_of_active_boxes(in_a_call=True)`
    /// can recover it from `self.pc - 1`.  pyre's grouped inline-call
    /// encoding stores three optional return slots (`i/r/f`) as u16 values,
    /// so the exact result index is carried here while preserving
    /// `_result_argcode` for the line-by-line clear logic.
    pub result_arg_index: Option<usize>,
    /// pyjitpl.py `MIFrame.pushed_box` — the box that the previous
    /// `_opimpl_any_push` instruction parked on the frame.  Reset to
    /// `None` by `cleanup_registers` so a recycled frame does not keep
    /// the parked box alive.
    pub pushed_box: Option<OpRef>,
    /// pyjitpl.py:93 `self.parent_snapshot = -1`.
    ///
    /// Set by `TraceRecordBuffer::_ensure_parent_resumedata` to the
    /// snapshot_index returned by the paired `create_snapshot(back,
    /// is_last)` call — so that a later `capture_resumedata` whose
    /// framestack shares this frame as an ancestor can short-circuit
    /// the parent chain walk by patching with `snapshot_add_prev(
    /// target.parent_snapshot)` instead of re-emitting the snapshot.
    pub parent_snapshot: i64,
    /// pyjitpl.py:95 `self.unroll_iterations = 1`.
    pub unroll_iterations: usize,
    /// State-field-JIT multi-frame snapshot wiring.
    ///
    /// For frames pushed by `BC_INLINE_CALL` this holds the index into
    /// the *parent* frame's `jitcode.descrs` array that the dispatcher
    /// resolved into the sub-jitcode for this frame
    /// (`pyjitpl/dispatch.rs` BC_INLINE_CALL `descrs.get(sub_idx)`). Root
    /// frames (`trace_jitcode` portal entry) carry `u32::MAX`. The
    /// snapshot-side `build_state_field_snapshot` packs this into
    /// `SnapshotFrame.jitcode_index` for non-root frames, and the
    /// resume-side `register_jitcode_factory` callback uses it to walk
    /// `parent.descrs[idx].as_jitcode()` instead of re-invoking the
    /// factory.  RPython's equivalent is `frame.jitcode.index` —
    /// resolvable through `metainterp_sd.jitcodes[idx]` because RPython
    /// pre-registers every jitcode globally; per-opcode majit has no
    /// such global registry, so the parent-relative index threads the
    /// chain together.
    pub parent_descr_idx: u32,
}

impl MIFrame {
    pub fn new(jitcode: Arc<JitCode>, pc: usize) -> Self {
        // RPython `pyjitpl.py:81-90` `MIFrame.setup` creates the typed
        // register arrays sized to `num_regs_and_consts_*()`.  pyre
        // keeps the raw allocation step separate so low-level tests and
        // resume paths can construct a frame without immediately
        // interning constants into a `TraceCtx`.
        //
        // `MetaInterp::newframe`, `trace_jitcode_with_framestack`,
        // `trace_jitcode`, and inline-call dispatch should call
        // `MIFrame::setup(...)` below, which mirrors the rest of
        // `pyjitpl.py:74-95`.
        //
        // RPython `pyjitpl.py:2247-2253` `MIFrame.setup`:
        //   self.registers_i = [None] * jitcode.num_regs_and_consts_i()
        //   self.registers_r = [None] * jitcode.num_regs_and_consts_r()
        //   self.registers_f = [None] * jitcode.num_regs_and_consts_f()
        // The register file spans working registers AND the constants area
        // (`[num_regs_X .. num_regs_and_consts_X)`).
        let regs_and_consts_i = jitcode.num_regs_and_consts_i();
        let regs_and_consts_r = jitcode.num_regs_and_consts_r();
        let regs_and_consts_f = jitcode.num_regs_and_consts_f();
        Self {
            jitcode,
            pc,
            code_cursor: 0,
            int_regs: vec![None; regs_and_consts_i],
            int_values: vec![None; regs_and_consts_i],
            ref_regs: vec![None; regs_and_consts_r],
            ref_values: vec![None; regs_and_consts_r],
            float_regs: vec![None; regs_and_consts_f],
            float_values: vec![None; regs_and_consts_f],
            inline_frame: false,
            return_i: None,
            return_r: None,
            return_f: None,
            greenkey: None,
            _result_argcode: b'v',
            result_arg_index: None,
            pushed_box: None,
            parent_snapshot: -1,
            unroll_iterations: 1,
            parent_descr_idx: u32::MAX,
        }
    }

    /// RPython `pyjitpl.py:74-95` `MIFrame.setup(jitcode, greenkey=None)`.
    ///
    /// This is the entry-point that should back normal frame creation:
    /// allocate the typed register files, remember the recursive-portal
    /// greenkey, copy constants into the register window, and reset the
    /// per-frame bookkeeping fields that upstream initializes in
    /// `setup()`.
    pub fn setup(
        jitcode: Arc<JitCode>,
        pc: usize,
        greenkey: Option<u64>,
        ctx: Option<&mut crate::trace_ctx::TraceCtx>,
    ) -> Self {
        let mut frame = Self::new(jitcode, pc);
        frame.greenkey = greenkey;
        if let Some(ctx) = ctx {
            frame.copy_constants(ctx);
        }
        frame._result_argcode = b'v';
        frame.result_arg_index = None;
        frame.parent_snapshot = -1;
        frame.unroll_iterations = 1;
        frame
    }

    pub fn next_u8(&mut self) -> u8 {
        read_u8(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn next_u16(&mut self) -> u16 {
        read_u16(&self.jitcode.code, &mut self.code_cursor)
    }

    /// Peek a u16 at absolute position `pos` without advancing the
    /// cursor.  Mirrors the canonical decode helper on
    /// `BlackholeInterpreter::peek_u16_at` so trace dispatch can do the
    /// same dual-encoding auto-detect for vable opcodes.
    pub fn peek_u16_at(&self, pos: usize) -> Option<u16> {
        let code = &self.jitcode.code;
        if pos + 1 >= code.len() {
            return None;
        }
        Some((code[pos] as u16) | ((code[pos + 1] as u16) << 8))
    }

    /// Resolve a `d`-argcode descr index against the per-jitcode
    /// runtime descr pool (`exec.descrs`), returning the canonical
    /// `BhDescr` if present.  Trace dispatch uses this to detect when a
    /// vable opcode was emitted in canonical form (descr lives in the
    /// pool) versus pyre's pre-orthodox legacy form (descr operand is
    /// the field index inline).
    pub fn runtime_bh_descr(&self, descr_idx: usize) -> Option<&crate::blackhole::BhDescr> {
        self.jitcode
            .exec
            .descrs
            .get(descr_idx)
            .and_then(crate::jitcode::RuntimeBhDescr::as_bh_descr)
    }

    /// Read the static-field index from a canonical `VableField` descr
    /// pool entry at `pos`. Stage 3c-3 dropped the dual-mode
    /// auto-detect, so the bytes must already be canonical
    /// (`assembler.py:165-167` + `:197-207`).
    pub fn vable_field_index_at(&self, pos: usize) -> usize {
        let idx = self
            .peek_u16_at(pos)
            .expect("vable_field_index_at: descr operand out of bounds");
        match self.runtime_bh_descr(idx as usize) {
            Some(crate::blackhole::BhDescr::VableField { index }) => *index,
            other => {
                panic!("vable_field_index_at: expected VableField at descr {idx}, got {other:?}")
            }
        }
    }

    /// Read the array-field index from a canonical
    /// (`VableArray`, `Array`) descr pool pair.
    pub fn vable_array_index_pair_at(&self, field_pos: usize, array_pos: usize) -> usize {
        let field_idx = self
            .peek_u16_at(field_pos)
            .expect("vable_array_index_pair_at: field descr out of bounds");
        let array_idx = self
            .peek_u16_at(array_pos)
            .expect("vable_array_index_pair_at: array descr out of bounds");
        let field_descr = self.runtime_bh_descr(field_idx as usize);
        let array_descr = self.runtime_bh_descr(array_idx as usize);
        match (field_descr, array_descr) {
            (
                Some(crate::blackhole::BhDescr::VableArray { index }),
                Some(crate::blackhole::BhDescr::Array { .. }),
            ) => *index,
            other => panic!(
                "vable_array_index_pair_at: expected (VableArray, Array) at descrs ({field_idx}, {array_idx}), got {other:?}"
            ),
        }
    }

    /// Decode a `getfield_vable_<kind>/rd>X` operand triple, returning
    /// `(vable_reg, field_idx, dest_reg)` per `assembler.py:165-167` +
    /// `:197-207`. Canonical layout: 1B vable_reg + 2B descr_pool_idx
    /// + 1B dest_reg. The leading `r` operand carries the live struct
    /// register consumed as the `struct` argument by RPython
    /// `pyjitpl.py:1166 _opimpl_setfield_vable_*`.
    pub fn read_vable_getfield(&mut self) -> (usize, usize, usize) {
        let field_idx = self.vable_field_index_at(self.code_cursor + 1);
        let base = self.next_u8() as usize;
        self.code_cursor += 2;
        let dst = self.next_u8() as usize;
        (base, field_idx, dst)
    }

    /// Decode a `setfield_vable_<kind>/rXd` operand triple, returning
    /// `(vable_reg, field_idx, value_reg)`. Canonical layout: 1B
    /// vable_reg + 1B value_reg + 2B descr_pool_idx.
    pub fn read_vable_setfield(&mut self) -> (usize, usize, usize) {
        let field_idx = self.vable_field_index_at(self.code_cursor + 2);
        let base = self.next_u8() as usize;
        let src = self.next_u8() as usize;
        self.code_cursor += 2;
        (base, field_idx, src)
    }

    /// Decode a `getarrayitem_vable_<kind>/ridd>X` operand quintuple,
    /// returning `(vable_reg, array_idx, index_reg, dest_reg)`.
    /// Canonical layout: 1B vable_reg + 1B index_reg + 2B fdescr + 2B
    /// adescr + 1B dest.
    pub fn read_vable_getarrayitem(&mut self) -> (usize, usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 2, self.code_cursor + 4);
        let base = self.next_u8() as usize;
        let index_reg = self.next_u8() as usize;
        self.code_cursor += 4;
        let dst = self.next_u8() as usize;
        (base, array_idx, index_reg, dst)
    }

    /// Decode a `setarrayitem_vable_<kind>/riXdd` operand quintuple,
    /// returning `(vable_reg, array_idx, index_reg, value_reg)`.
    /// Canonical layout: 1B vable_reg + 1B index_reg + 1B value_reg +
    /// 2B fdescr + 2B adescr.
    pub fn read_vable_setarrayitem(&mut self) -> (usize, usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 3, self.code_cursor + 5);
        let base = self.next_u8() as usize;
        let index_reg = self.next_u8() as usize;
        let src = self.next_u8() as usize;
        self.code_cursor += 4;
        (base, array_idx, index_reg, src)
    }

    /// Decode an `arraylen_vable/rdd>i` operand triple, returning
    /// `(vable_reg, array_idx, dest_reg)`. Canonical layout: 1B
    /// vable_reg + 2B fdescr + 2B adescr + 1B dest.
    pub fn read_vable_arraylen(&mut self) -> (usize, usize, usize) {
        let array_idx = self.vable_array_index_pair_at(self.code_cursor + 1, self.code_cursor + 3);
        let base = self.next_u8() as usize;
        self.code_cursor += 4;
        let dst = self.next_u8() as usize;
        (base, array_idx, dst)
    }

    /// pyjitpl.py:1530-1535 `MIFrame.verify_green_args(jitdriver_sd, varargs)`.
    ///
    /// ```python
    /// def verify_green_args(self, jitdriver_sd, varargs):
    ///     num_green_args = jitdriver_sd.num_green_args
    ///     assert len(varargs) == num_green_args
    ///     for i in range(num_green_args):
    ///         assert isinstance(varargs[i], Const)
    /// ```
    ///
    /// Called from `opimpl_jit_merge_point` (pyjitpl.py:1541) so the
    /// metainterp aborts immediately if a green arg arrives as a Box
    /// (non-constant) — that would mean the codewriter / annotator
    /// failed to mark it as compile-time-known. The pyre port lifts
    /// `varargs` from the upstream `boxes3` typed-list to a flat
    /// `&[OpRef]` (the trace-side representation of green args after
    /// `make_three_lists` flattens by kind in jtransform). Each element
    /// must satisfy [`OpRef::is_constant`] — the constant-namespace
    /// flag set by `OpRef::from_const`.
    ///
    /// Production callers land with S2.3 follow-up (the metainterp-side
    /// `opimpl_jit_merge_point` port). For now the helper is dead code
    /// outside tests; it locks the contract upstream relies on so a
    /// future caller cannot silently accept a non-Const green.
    pub fn verify_green_args(
        jitdriver_sd: &crate::jitdriver::JitDriverStaticData,
        varargs: &[majit_ir::OpRef],
    ) {
        let num_green_args = jitdriver_sd.num_green_args();
        assert_eq!(
            varargs.len(),
            num_green_args,
            "verify_green_args: expected {} greens, got {}",
            num_green_args,
            varargs.len()
        );
        for (i, opref) in varargs.iter().enumerate() {
            assert!(
                opref.is_constant(),
                "verify_green_args: greens[{i}] = {opref:?} is not a Const \
                 (upstream pyjitpl.py:1534 asserts isinstance(varargs[i], Const))",
            );
        }
    }

    /// pyjitpl.py:98-119 `MIFrame.copy_constants`.
    ///
    /// RPython copies each entry of `jitcode.constants_{i,r,f}` into the
    /// register file at `registers_X[num_regs_X + i]`, wrapping it in
    /// `Const{Int,Ptr,Float}`. `MIFrame.setup` (pyjitpl.py:82-90) invokes
    /// this inline right after sizing the register arrays to
    /// `num_regs_and_consts_X`.
    ///
    /// pyre stores an `OpRef` + concrete `i64` pair per slot; `ctx` is
    /// needed to intern the const into the trace context's constant pool
    /// before the opref is written into the register file.
    pub fn copy_constants(&mut self, ctx: &mut crate::trace_ctx::TraceCtx) {
        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        for (i, &value) in self.jitcode.constants_i.iter().enumerate() {
            let slot = num_regs_i + i;
            self.int_regs[slot] = Some(ctx.const_int(value));
            self.int_values[slot] = Some(value);
        }
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        for (i, &value) in self.jitcode.constants_r.iter().enumerate() {
            let slot = num_regs_r + i;
            self.ref_regs[slot] = Some(ctx.const_ref(value));
            self.ref_values[slot] = Some(value);
        }
        let num_regs_f = self.jitcode.c_num_regs_f as usize;
        for (i, &value) in self.jitcode.constants_f.iter().enumerate() {
            let slot = num_regs_f + i;
            self.float_regs[slot] = Some(ctx.const_float(value));
            self.float_values[slot] = Some(value);
        }
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.jitcode.code.len()
    }

    /// pyjitpl.py:121-127 `MIFrame.cleanup_registers()`.
    ///
    /// ```python
    /// def cleanup_registers(self):
    ///     for i in range(self.jitcode.num_regs_r()):
    ///         self.registers_r[i] = None
    ///     self.pushed_box = None
    /// ```
    ///
    /// Iterates `0..num_regs_r()` (RPython skips the constants area
    /// that lives past `num_regs_r`); pyre's `ref_regs` is sized
    /// exactly to `num_regs_r` so the loop scans the same slots.
    /// `ref_values` is cleared in lockstep — it is the pyre-only
    /// concrete-value mirror that lives next to each box.
    pub fn cleanup_registers(&mut self) {
        let num_regs_r = self.jitcode.num_regs_r() as usize;
        for i in 0..num_regs_r {
            self.ref_regs[i] = None;
            self.ref_values[i] = None;
        }
        self.pushed_box = None;
    }

    /// pyjitpl.py:1878-1879 `MIFrame.setup_resume_at_op(pc)`.
    pub fn setup_resume_at_op(&mut self, pc: usize) {
        self.pc = pc;
    }

    /// pyjitpl.py:258-275 `MIFrame.make_result_of_lastop(resultbox)`.
    ///
    /// Stores the result of the last opimpl into the typed register at
    /// `target_index`. RPython reads `target_index = ord(self.bytecode[self.pc-1])`
    /// from the bytecode; pyre's call BC encodes `dst` explicitly so
    /// callers pass it directly.
    pub fn make_result_of_lastop(
        &mut self,
        kind: JitArgKind,
        target_index: usize,
        opref: OpRef,
        concrete: i64,
    ) {
        #[cfg(debug_assertions)]
        {
            // pyjitpl.py:260-264 non-translated check:
            //
            //     if not we_are_translated():
            //         typeof = {'i': history.INT, 'r': history.REF,
            //                   'f': history.FLOAT}
            //         assert typeof[self.jitcode._resulttypes[self.pc]] == got_type
            //
            // Rust `debug_assertions` is the closest equivalent of
            // `not we_are_translated()`: fail loudly in non-translated
            // developer builds, but elide the whole check in release.
            let got = match kind {
                JitArgKind::Int => 'i',
                JitArgKind::Ref => 'r',
                JitArgKind::Float => 'f',
            };
            let resulttypes = self
                .jitcode
                .core()
                .body()
                .resulttypes
                .as_ref()
                .expect("make_result_of_lastop: _resulttypes is None");
            let recorded = *resulttypes
                .get(&self.pc)
                .expect("make_result_of_lastop: missing _resulttypes[pc]");
            assert_eq!(
                recorded, got,
                "make_result_of_lastop: jitcode._resulttypes[{}] = {recorded:?} but runtime kind = {got:?}",
                self.pc
            );
        }
        match kind {
            JitArgKind::Int => {
                self.int_regs[target_index] = Some(opref);
                self.int_values[target_index] = Some(concrete);
            }
            JitArgKind::Ref => {
                self.ref_regs[target_index] = Some(opref);
                self.ref_values[target_index] = Some(concrete);
            }
            JitArgKind::Float => {
                self.float_regs[target_index] = Some(opref);
                self.float_values[target_index] = Some(concrete);
            }
        }
    }

    /// pyjitpl.py:1862-1876 `MIFrame.setup_call(argboxes)`.
    ///
    /// Resets `pc` to 0 and copies each argbox into the first slot of
    /// its typed register bank in declaration order. RPython's
    /// `setup_call` consults `box.type`; pyre's `OpRef` does not carry
    /// type info, so the caller passes a typed `(kind, value, concrete)`
    /// tuple per arg.
    ///
    /// Also resets `parent_snapshot = -1`. In RPython this lives in
    /// `MIFrame.setup()` (pyjitpl.py:93) which always precedes
    /// `setup_call`; pyre's `MIFrame::new` already sets it, but any
    /// future frame-recycling path (upstream `free_frames_list`) would
    /// reuse an MIFrame and skip `new()`, so we reset here to match the
    /// RPython "every call-entry clears parent_snapshot" invariant.
    /// pyjitpl.py:177-234 `MIFrame.get_list_of_active_boxes`.
    ///
    /// Reads the LIVE-op liveness header preceding the current pc and
    /// pushes each live register onto the trace's snapshot-array data
    /// in int → ref → float order.  Returns the `_snapshot_array_data`
    /// offset produced by `new_array` (RPython `storage`).
    ///
    /// `op_live` and `all_liveness` are threaded through from
    /// `MetaInterpStaticData` (RPython
    /// `self.metainterp.staticdata.op_live` /
    /// `.liveness_info`) — pyre passes them explicitly so this method
    /// does not depend on `MetaInterpStaticData` structurally.
    ///
    /// `pool` carries the trace's `ConstantPool` so the in_a_call
    /// branch can mint `history.CONST_FALSE` / `CONST_NULL` /
    /// `history.CONST_FZERO` OpRefs and write them into the cleared
    /// register slot exactly as pyjitpl.py:188-192 does.  Tests that
    /// only care about the snapshot output may pass `None` — the
    /// fallback path then substitutes `Box::Const*(0)` directly into
    /// the snapshot array (byte-identical to the structural path) but
    /// leaves the register slot untouched.
    pub fn get_list_of_active_boxes(
        &mut self,
        in_a_call: bool,
        trace: &mut TraceRecordBuffer,
        pool: Option<&mut ConstantPool>,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> i64 {
        const SIZE_LIVE_OP: usize = majit_translate::liveness::OFFSET_SIZE + 1;
        use majit_translate::liveness::{LivenessIterator, decode_offset};

        // pyjitpl.py:180-193 — in_a_call branch.  The frame that holds
        // the in-flight CALL instruction has a "result" register slot
        // that is not yet defined (the call has not returned).  RPython
        // clears that slot to a zero constant (history.CONST_FALSE /
        // CONST_NULL / history.CONST_FZERO) so the snapshot captures a
        // well-defined placeholder instead of pre-call stale data.
        //
        // When `pool` is provided we mirror RPython exactly: mint the
        // zero constant, write it into the register slot (and the
        // parallel `*_values` mirror), and let the bank loop below
        // emit it through the normal register → OpBox path.  This makes
        // a subsequent `get_list_of_active_boxes` (e.g. a re-snapshot
        // of the same in-flight call before `make_result_of_lastop`)
        // see the cleared slot rather than the pre-call stale contents.
        //
        // When `pool` is `None` (test fixtures that do not wire a
        // ConstantPool) we fall back to substituting the cleared box
        // directly into the snapshot array via `clear_*_idx`.  The
        // snapshot bytes are identical, but the register slot retains
        // its pre-call contents — acceptable for tests that only check
        // snapshot bytes.
        let (clear_int_idx, clear_ref_idx, clear_float_idx) = if in_a_call {
            let argcode = self._result_argcode;
            let index = self
                .result_arg_index
                .take()
                .unwrap_or_else(|| self.jitcode.code[self.pc - 1] as usize);
            // pyjitpl.py:193 `self._result_argcode = '?'` — mark cleared.
            self._result_argcode = b'?';
            if let Some(pool) = pool {
                // pyjitpl.py:184-192 register clearing via ConstantPool.
                match argcode {
                    b'i' => {
                        let opref = pool.get_or_insert(0);
                        self.int_regs[index] = Some(opref);
                        self.int_values[index] = Some(0);
                        (None, None, None)
                    }
                    b'r' => {
                        let opref = pool.get_or_insert_typed(0, Type::Ref);
                        self.ref_regs[index] = Some(opref);
                        self.ref_values[index] = Some(0);
                        (None, None, None)
                    }
                    b'f' => {
                        let opref = pool.get_or_insert_typed(0, Type::Float);
                        self.float_regs[index] = Some(opref);
                        self.float_values[index] = Some(0);
                        (None, None, None)
                    }
                    _ => (None, None, None),
                }
            } else {
                // Test-fallback: substitute zero in the snapshot output
                // without mutating the register slot.
                match argcode {
                    b'i' => (Some(index), None, None),
                    b'r' => (None, Some(index), None),
                    b'f' => (None, None, Some(index)),
                    _ => (None, None, None),
                }
            }
        } else {
            (None, None, None)
        };

        // pyjitpl.py:194-198 — pick the pc of the preceding LIVE op.
        let pc = if in_a_call || after_residual_call {
            self.pc
        } else {
            self.pc - SIZE_LIVE_OP
        };

        // pyjitpl.py:199 `assert ord(self.jitcode.code[pc]) == op_live`.
        debug_assert_eq!(self.jitcode.code[pc], op_live);

        // pyjitpl.py:202-207 — decode offset + per-type lengths.
        let mut offset = decode_offset(&self.jitcode.code, pc + 1);
        let length_i = all_liveness[offset] as u32;
        let length_r = all_liveness[offset + 1] as u32;
        let length_f = all_liveness[offset + 2] as u32;
        offset += 3;

        // pyjitpl.py:209-214 — pre-allocate the storage array.
        let total = (length_i + length_r + length_f) as usize;
        let storage = trace.new_array(total);

        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        let num_regs_f = self.jitcode.c_num_regs_f as usize;

        // pyjitpl.py:216-221 — push live int registers.  Liveness
        // indices are interpreted against the `num_regs_and_consts_i`
        // layout (RPython pyjitpl.py:82-83): the first `num_regs_i`
        // slots are runtime registers, the remaining are jitcode
        // constants copied in by `setup()`.  pyre keeps registers
        // separate from the jitcode's `constants_i` Vec, so a liveness
        // index past `num_regs_i` reads directly from
        // `self.jitcode.constants_i[idx - num_regs_i]` and emits the
        // corresponding `ConstInt` box — matching what RPython's
        // `copy_constants` + `registers_i[index]` lookup would return.
        if length_i > 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let b = if Some(idx) == clear_int_idx {
                    // pyjitpl.py:184-185 history.CONST_FALSE clearing.
                    OpBox::ConstInt(0)
                } else if idx < num_regs_i {
                    let opref = self.int_regs[idx]
                        .expect("get_list_of_active_boxes: int register uninitialized");
                    let value = self.int_values[idx]
                        .expect("get_list_of_active_boxes: int value uninitialized");
                    register_to_box_int(opref, value)
                } else {
                    // pyjitpl.py:82-83 `copy_constants(..., constants_i, ...,
                    // ConstInt)` — constants live in the `[num_regs_i ..
                    // num_regs_and_consts_i)` back slots.
                    OpBox::ConstInt(self.jitcode.constants_i[idx - num_regs_i])
                };
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset;
        }

        // pyjitpl.py:222-227 — push live ref registers.  Constants
        // area mirror of the int path above;
        // `jitcode.constants_r[i]` stores raw GC addresses as i64 so
        // we cast through `u64` for `Box::ConstPtr`.
        if length_r > 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let b = if Some(idx) == clear_ref_idx {
                    // pyjitpl.py:186-187 CONST_NULL clearing.
                    OpBox::ConstPtr(0)
                } else if idx < num_regs_r {
                    let opref = self.ref_regs[idx]
                        .expect("get_list_of_active_boxes: ref register uninitialized");
                    let value = self.ref_values[idx]
                        .expect("get_list_of_active_boxes: ref value uninitialized");
                    register_to_box_ref(opref, value)
                } else {
                    // pyjitpl.py:84-85 `copy_constants(..., constants_r, ...,
                    // ConstPtrJitCode)` — constants_r store raw GC
                    // addresses; we cast through u64 for `Box::ConstPtr`.
                    OpBox::ConstPtr(self.jitcode.constants_r[idx - num_regs_r] as u64)
                };
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset;
        }

        // pyjitpl.py:228-233 — push live float registers.
        if length_f > 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let b = if Some(idx) == clear_float_idx {
                    // pyjitpl.py:188-189 history.CONST_FZERO clearing.
                    OpBox::ConstFloat(0)
                } else if idx < num_regs_f {
                    let opref = self.float_regs[idx]
                        .expect("get_list_of_active_boxes: float register uninitialized");
                    let value = self.float_values[idx]
                        .expect("get_list_of_active_boxes: float value uninitialized");
                    register_to_box_float(opref, value)
                } else {
                    // pyjitpl.py:86-87 `copy_constants(..., constants_f,
                    // ..., ConstFloat)` — `constants_f[i]` stores the
                    // raw bits of the f64.
                    OpBox::ConstFloat(self.jitcode.constants_f[idx - num_regs_f] as u64)
                };
                trace._add_box_to_storage_box(b);
            }
            let _ = it.offset; // offset no longer read after the last bank
        }

        storage
    }

    /// Side-table snapshot counterpart of
    /// [`Self::get_list_of_active_boxes`].
    ///
    /// This follows the same pyjitpl.py:177-234 control flow but returns
    /// `recorder::SnapshotTagged` entries for the legacy `TraceCtx`
    /// snapshot side table instead of writing opencoder byte arrays.
    pub fn get_list_of_active_snapshot_boxes(
        &mut self,
        in_a_call: bool,
        pool: Option<&mut ConstantPool>,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> Vec<SnapshotTagged> {
        const SIZE_LIVE_OP: usize = majit_translate::liveness::OFFSET_SIZE + 1;
        use majit_translate::liveness::{LivenessIterator, decode_offset};

        let (clear_int_idx, clear_ref_idx, clear_float_idx) = if in_a_call {
            let argcode = self._result_argcode;
            let index = self
                .result_arg_index
                .take()
                .unwrap_or_else(|| self.jitcode.code[self.pc - 1] as usize);
            self._result_argcode = b'?';
            if let Some(pool) = pool {
                match argcode {
                    b'i' => {
                        let opref = pool.get_or_insert(0);
                        self.int_regs[index] = Some(opref);
                        self.int_values[index] = Some(0);
                    }
                    b'r' => {
                        let opref = pool.get_or_insert_typed(0, Type::Ref);
                        self.ref_regs[index] = Some(opref);
                        self.ref_values[index] = Some(0);
                    }
                    b'f' => {
                        let opref = pool.get_or_insert_typed(0, Type::Float);
                        self.float_regs[index] = Some(opref);
                        self.float_values[index] = Some(0);
                    }
                    _ => {}
                }
                (None, None, None)
            } else {
                match argcode {
                    b'i' => (Some(index), None, None),
                    b'r' => (None, Some(index), None),
                    b'f' => (None, None, Some(index)),
                    _ => (None, None, None),
                }
            }
        } else {
            (None, None, None)
        };

        let pc = if in_a_call || after_residual_call {
            self.pc
        } else {
            self.pc - SIZE_LIVE_OP
        };
        debug_assert_eq!(self.jitcode.code[pc], op_live);

        let mut offset = decode_offset(&self.jitcode.code, pc + 1);
        let length_i = all_liveness[offset] as u32;
        let length_r = all_liveness[offset + 1] as u32;
        let length_f = all_liveness[offset + 2] as u32;
        offset += 3;

        let total = (length_i + length_r + length_f) as usize;
        let mut boxes = Vec::with_capacity(total);

        let num_regs_i = self.jitcode.c_num_regs_i as usize;
        let num_regs_r = self.jitcode.c_num_regs_r as usize;
        let num_regs_f = self.jitcode.c_num_regs_f as usize;

        if length_i > 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_int_idx {
                    SnapshotTagged::Const(0, Type::Int)
                } else if idx < num_regs_i {
                    let opref = self.int_regs[idx]
                        .expect("get_list_of_active_snapshot_boxes: int register uninitialized");
                    let value = self.int_values[idx]
                        .expect("get_list_of_active_snapshot_boxes: int value uninitialized");
                    if opref.is_constant() {
                        SnapshotTagged::Const(value, Type::Int)
                    } else {
                        SnapshotTagged::Box(opref, Type::Int)
                    }
                } else {
                    SnapshotTagged::Const(self.jitcode.constants_i[idx - num_regs_i], Type::Int)
                };
                boxes.push(tagged);
            }
            offset = it.offset;
        }

        if length_r > 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_ref_idx {
                    SnapshotTagged::Const(0, Type::Ref)
                } else if idx < num_regs_r {
                    let opref = self.ref_regs[idx]
                        .expect("get_list_of_active_snapshot_boxes: ref register uninitialized");
                    let value = self.ref_values[idx]
                        .expect("get_list_of_active_snapshot_boxes: ref value uninitialized");
                    if opref.is_constant() {
                        SnapshotTagged::Const(value, Type::Ref)
                    } else {
                        SnapshotTagged::Box(opref, Type::Ref)
                    }
                } else {
                    SnapshotTagged::Const(self.jitcode.constants_r[idx - num_regs_r], Type::Ref)
                };
                boxes.push(tagged);
            }
            offset = it.offset;
        }

        if length_f > 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let tagged = if Some(idx) == clear_float_idx {
                    SnapshotTagged::Const(0, Type::Float)
                } else if idx < num_regs_f {
                    let opref = self.float_regs[idx]
                        .expect("get_list_of_active_snapshot_boxes: float register uninitialized");
                    let value = self.float_values[idx]
                        .expect("get_list_of_active_snapshot_boxes: float value uninitialized");
                    if opref.is_constant() {
                        SnapshotTagged::Const(value, Type::Float)
                    } else {
                        SnapshotTagged::Box(opref, Type::Float)
                    }
                } else {
                    SnapshotTagged::Const(self.jitcode.constants_f[idx - num_regs_f], Type::Float)
                };
                boxes.push(tagged);
            }
        }

        boxes
    }

    /// pyjitpl.py:236-255 `MIFrame.replace_active_box_in_frame(oldbox, newbox)`.
    ///
    /// ```python
    /// def replace_active_box_in_frame(self, oldbox, newbox):
    ///     if oldbox.type == 'i':
    ///         count = self.jitcode.num_regs_i()
    ///         registers = self.registers_i
    ///     elif oldbox.type == 'r':
    ///         count = self.jitcode.num_regs_r()
    ///         registers = self.registers_r
    ///     elif oldbox.type == 'f':
    ///         count = self.jitcode.num_regs_f()
    ///         registers = self.registers_f
    ///     else:
    ///         assert 0, oldbox
    ///     if not count:
    ///         return
    ///     for i in range(count):
    ///         if registers[i] is oldbox:
    ///             registers[i] = newbox
    /// ```
    ///
    /// pyjitpl.py:240 dispatches on `oldbox.type` — `'i'` / `'r'` / `'f'`
    /// pick the matching register bank, the `else` arm is `assert 0,
    /// oldbox`. Pyre's OpRef does not carry a type tag at the call
    /// boundary, so the bank is selected by an explicit `oldbox_type`
    /// parameter that the caller (`MetaInterp::replace_box`) resolves
    /// once via `TraceCtx::get_opref_type`. `Type::Void` panics with the
    /// upstream assertion message.
    pub fn replace_active_box_in_frame(&mut self, oldbox: OpRef, newbox: OpRef, oldbox_type: Type) {
        let registers = match oldbox_type {
            Type::Int => &mut self.int_regs,
            Type::Ref => &mut self.ref_regs,
            Type::Float => &mut self.float_regs,
            Type::Void => panic!(
                "replace_active_box_in_frame: void oldbox {:?} (pyjitpl.py:240 `assert 0, oldbox`)",
                oldbox
            ),
        };
        if registers.is_empty() {
            return;
        }
        for slot in registers.iter_mut() {
            if *slot == Some(oldbox) {
                *slot = Some(newbox);
            }
        }
    }

    pub fn setup_call(&mut self, argboxes: &[(JitArgKind, OpRef, i64)]) {
        self.pc = 0;
        self.parent_snapshot = -1;
        let mut count_i = 0;
        let mut count_r = 0;
        let mut count_f = 0;
        for (kind, value, concrete) in argboxes {
            match kind {
                JitArgKind::Int => {
                    self.int_regs[count_i] = Some(*value);
                    self.int_values[count_i] = Some(*concrete);
                    count_i += 1;
                }
                JitArgKind::Ref => {
                    self.ref_regs[count_r] = Some(*value);
                    self.ref_values[count_r] = Some(*concrete);
                    count_r += 1;
                }
                JitArgKind::Float => {
                    self.float_regs[count_f] = Some(*value);
                    self.float_values[count_f] = Some(*concrete);
                    count_f += 1;
                }
            }
        }
    }
}

/// RPython pyjitpl.py: MetaInterp.framestack
#[derive(Default)]
pub struct MIFrameStack {
    pub frames: Vec<MIFrame>,
}

impl MIFrameStack {
    /// Empty framestack.  Mirrors `self.framestack = []` in
    /// `MetaInterp.initialize_state_from_start` (pyjitpl.py:3269) and
    /// `rebuild_state_after_failure` (pyjitpl.py:3403).
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    /// Build a stack pre-seeded with one root frame.  Pyre's
    /// `JitCodeMachine` always opens with a single root frame, so this
    /// constructor mirrors `MIFrameStack::new(root)` from before the
    /// `Arc<JitCode>` migration.
    pub fn new(root: MIFrame) -> Self {
        Self { frames: vec![root] }
    }

    pub fn current_mut(&mut self) -> &mut MIFrame {
        self.frames.last_mut().expect("empty JitCode frame stack")
    }

    pub fn push(&mut self, frame: MIFrame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<MIFrame> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode::JitCodeBuilder;
    use majit_ir::OpRef;

    fn make_jitcode_with_regs(num_i: u16, num_r: u16, num_f: u16) -> Arc<JitCode> {
        let mut builder = JitCodeBuilder::new();
        for i in 0..num_i {
            builder.load_const_i_value(i, 0);
        }
        for i in 0..num_r {
            builder.load_const_r_value(i, 0);
        }
        for i in 0..num_f {
            builder.load_const_f_value(i, 0);
        }
        let mut jitcode = builder.finish();
        jitcode.body_mut().resulttypes = Some([(0, 'i'), (1, 'r'), (2, 'f')].into_iter().collect());
        Arc::new(jitcode)
    }

    #[test]
    fn setup_call_distributes_argboxes_by_kind_in_declaration_order() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 99;
        frame.setup_call(&[
            (JitArgKind::Int, OpRef::int_op(10), 100),
            (JitArgKind::Ref, OpRef::ref_op(20), 200),
            (JitArgKind::Int, OpRef::int_op(11), 101),
            (JitArgKind::Float, OpRef::float_op(30), 300),
            (JitArgKind::Ref, OpRef::ref_op(21), 201),
        ]);

        assert_eq!(frame.pc, 0);
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(10)));
        assert_eq!(frame.int_values[0], Some(100));
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(11)));
        assert_eq!(frame.int_values[1], Some(101));
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(20)));
        assert_eq!(frame.ref_values[0], Some(200));
        assert_eq!(frame.ref_regs[1], Some(OpRef::ref_op(21)));
        assert_eq!(frame.ref_values[1], Some(201));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(30)));
        assert_eq!(frame.float_values[0], Some(300));
    }

    /// Step 1b: `MIFrame::get_list_of_active_boxes` (non-in_a_call
    /// path) pushes live int / ref / float registers onto the trace's
    /// `_snapshot_array_data` in declaration order, returning the
    /// `new_array` offset. Tests with `after_residual_call=true` so the
    /// LIVE op sits exactly at `self.pc` (pyjitpl.py:194-195).
    #[test]
    fn get_list_of_active_boxes_populates_trace_snapshot_array() {
        use crate::opencoder::{Box as OpBox, TraceRecordBuffer};
        use majit_ir::OpRef;
        use std::sync::Arc;

        // 1 int reg + 1 ref reg + 0 float regs live at pc=0.
        let mut builder = JitCodeBuilder::new();
        let mut jitcode = builder.finish();
        jitcode.body_mut().c_num_regs_i = 1;
        jitcode.body_mut().c_num_regs_r = 1;
        jitcode.body_mut().c_num_regs_f = 0;
        const LIVE_OP: u8 = 0x42;
        // bytecode: [LIVE_OP, offset_lo, offset_hi] at pc 0..3.
        // decode_offset reads 2 bytes → liveness info at offset 0.
        jitcode.body_mut().code = vec![LIVE_OP, 0x00, 0x00];
        let jitcode = Arc::new(jitcode);

        // all_liveness at offset 0:
        //   len_i=1 len_r=1 len_f=0  →  3 header bytes
        //   int bitmask byte: bit 0 = register 0 live
        //   ref bitmask byte: bit 0 = register 0 live
        let all_liveness: Vec<u8> = vec![1, 1, 0, 0b0000_0001, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode, 0);
        // int_regs[0] non-constant OpRef::int_op(5) → Box::ResOp(5).
        frame.int_regs[0] = Some(OpRef::int_op(5));
        frame.int_values[0] = Some(0);
        // ref_regs[0] constant pointer addr=0xdead_beef → Box::ConstPtr.
        frame.ref_regs[0] = Some(OpRef::from_const(7));
        frame.ref_values[0] = Some(0xdead_beef);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(16, sd);
        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* pool */ None,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        // Two boxes pushed → array length prefix + 2 varints.
        // new_array(2) returns the offset before the length prefix.
        assert!(
            storage > 0,
            "storage offset must be non-zero for non-empty array"
        );

        // Verify the encoded tagged values match what we expect:
        // int: ResOp(5) encodes via _encode_box_position(5).
        // ref: ConstPtr(0xdead_beef) encodes via _encode_ptr.
        // Decode directly from _snapshot_array_data at `storage`.
        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 2, "array length prefix");
        let p0 = storage as usize + consumed;
        let (tag0, c0) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p0..]);
        assert_eq!(
            tag0,
            TraceRecordBuffer::_encode_box_position(5),
            "int register → TAGBOX(5)"
        );
        let p1 = p0 + c0;
        let (tag1, _) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p1..]);
        assert!(tag1 != 0, "ref register should encode to non-zero tag");
    }

    /// pyjitpl.py:180-193 in_a_call branch — the parent frame in a
    /// nested `capture_resumedata` walk clears the result register of
    /// the in-flight CALL to a zero constant.  When the liveness list
    /// INCLUDES the cleared slot, the snapshot array must record
    /// `Box::ConstInt(0)` (RPython's `history.CONST_FALSE`) instead
    /// of the pre-call register contents.
    ///
    /// Bytecode layout around `self.pc`:
    /// `[... call_op][dst_idx]_at_pc-1 [LIVE_byte]_at_pc
    ///  [off_lo][off_hi] ...` — the LIVE op carries the 2-byte offset
    /// to the bank-size + index bytes inside `all_liveness`.
    #[test]
    fn get_list_of_active_boxes_in_a_call_emits_const_for_cleared_slot() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        // `[0x88_call, 0x00_dst_idx, OP_LIVE, 0x00, 0x00]`
        //            ^^ pc - 1          ^^ pc
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2; // index of OP_LIVE byte

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        // `decode_offset(code, live_pc + 1)` reads offset bytes at
        // indices 3 and 4 — both 0, so `offset` into `all_liveness`
        // is 0.  Layout: `[length_i, length_r, length_f,
        //                 int_bitset_byte]`.  `length_i=1` means "1
        // live int bit set"; bitset byte `0b0000_0001` lights up
        // register index 0 (the cleared slot).
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        // Pre-call stale contents — must NOT leak into the snapshot.
        frame.int_regs[0] = Some(OpRef::int_op(777));
        frame.int_values[0] = Some(666);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* pool */ None,
            OP_LIVE,
            &all_liveness,
            /* after_residual_call */ false,
        );

        // pyjitpl.py:193 — `_result_argcode` flips to `b'?'` after
        // the clear.
        assert_eq!(frame._result_argcode, b'?');

        // Snapshot array content: `[length=1][ConstInt(0) tag]`.
        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // `ConstInt(0)` encodes as `tag(TAGINT, 0) = 0`.
        assert_eq!(tag0, 0, "cleared slot must encode as ConstInt(0)");
    }

    /// pyjitpl.py:184-192 — when a `ConstantPool` is provided, the
    /// in_a_call branch mutates the register slot itself to the
    /// pool's zero constant.  A second snapshot of the same in-flight
    /// call (after `_result_argcode` flips to `b'?'`) therefore also
    /// sees the cleared register, instead of the pre-call stale box.
    #[test]
    fn get_list_of_active_boxes_in_a_call_with_pool_mutates_register() {
        use crate::constant_pool::ConstantPool;
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        frame.int_regs[0] = Some(OpRef::int_op(777));
        frame.int_values[0] = Some(666);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);
        let mut pool = ConstantPool::new();

        let _ = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            Some(&mut pool),
            OP_LIVE,
            &all_liveness,
            /* after_residual_call */ false,
        );

        // Register slot now holds the pool's zero ConstInt OpRef
        // (constants live in the namespace whose OpRef has the high
        // const bit set), and the parallel value mirror is 0.
        let cleared = frame.int_regs[0].expect("register cleared, not unset");
        assert!(
            cleared.is_constant(),
            "cleared register must be a constant OpRef"
        );
        assert_eq!(frame.int_values[0], Some(0));
        assert_eq!(frame._result_argcode, b'?');
    }

    /// Complement: when the liveness walk lists a register OTHER than
    /// the cleared slot, the snapshot records the register's pre-
    /// existing box (not ConstInt(0)), while `_result_argcode` still
    /// flips to `b'?'`.
    #[test]
    fn get_list_of_active_boxes_in_a_call_preserves_other_registers() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![0x88, 0x00, OP_LIVE, 0x00, 0x00];
        let live_pc = 2;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(2, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            jc
        };

        // Liveness: live int idx 1 (NOT the cleared slot 0).
        // bitset byte 0b0000_0010 lights up bit 1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0010];

        let mut frame = MIFrame::new(jitcode_arc, live_pc);
        frame._result_argcode = b'i';
        frame.int_regs[0] = Some(OpRef::int_op(10)); // cleared — not listed.
        frame.int_values[0] = Some(999);
        frame.int_regs[1] = Some(OpRef::int_op(11)); // live — recorded as ResOp(11).
        frame.int_values[1] = Some(42);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(2, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* pool */ None,
            OP_LIVE,
            &all_liveness,
            false,
        );
        assert_eq!(frame._result_argcode, b'?');

        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // OpRef::int_op(11) encodes as `tag(TAGBOX, 11) = 11 << 2 | 3 = 47`.
        assert_eq!(tag0, TraceRecordBuffer::_encode_box_position(11));
    }

    /// pyjitpl.py:82-83 `copy_constants` — liveness indices past
    /// `num_regs_i` read from `jitcode.constants_i`.  pyre does not
    /// copy the constants into the `int_regs` Vec; instead the
    /// liveness walk reads `jitcode.constants_i[idx - num_regs_i]`
    /// directly and emits `Box::ConstInt(v)`.  This test confirms
    /// parity-equivalent snapshot content for the constants area.
    #[test]
    fn get_list_of_active_boxes_reads_constants_area() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        const OP_LIVE: u8 = 0x42;
        let code = vec![OP_LIVE, 0x00, 0x00];
        let live_pc = 0;

        let jitcode_arc = {
            let mut jc = make_jitcode_with_regs(1, 0, 0);
            let jc_mut = Arc::get_mut(&mut jc).expect("fresh Arc");
            jc_mut.body_mut().code = code;
            // `num_regs_i == 1`; constants_i has a single entry at
            // "register" index 1 (= num_regs_i).
            jc_mut.body_mut().constants_i = vec![1234];
            jc
        };

        // `self.pc = live_pc + SIZE_LIVE_OP` (in_a_call=false path:
        // `pc = self.pc - SIZE_LIVE_OP`).
        let current_pc = live_pc + majit_translate::jit_codewriter::liveness::OFFSET_SIZE + 1;

        // Liveness lists register index `1` — falls into the
        // constants area.  bitset byte 0b0000_0010 lights up bit 1.
        let all_liveness: Vec<u8> = vec![1, 0, 0, 0b0000_0010];

        let mut frame = MIFrame::new(jitcode_arc, current_pc);
        frame.int_regs[0] = Some(OpRef::int_op(5)); // real register — not referenced.
        frame.int_values[0] = Some(99);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);

        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            /* pool */ None,
            OP_LIVE,
            &all_liveness,
            false,
        );

        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 1);
        let (tag0, _) = crate::opencoder::decode_varint_signed(
            &trace._snapshot_array_data[storage as usize + consumed..],
        );
        // `ConstInt(1234)` — fits in SMALL_INT range so TAGINT with
        // shifted value.  `tag(TAGINT, 1234) = 1234 << 2 = 4936`.
        assert_eq!(tag0, 1234 << 2);
    }

    #[test]
    fn cleanup_registers_clears_ref_slots_and_pushed_box() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.int_regs[0] = Some(OpRef::int_op(1));
        frame.int_values[0] = Some(11);
        frame.ref_regs[0] = Some(OpRef::ref_op(2));
        frame.ref_values[0] = Some(22);
        frame.float_regs[0] = Some(OpRef::float_op(3));
        frame.float_values[0] = Some(33);
        frame.pushed_box = Some(OpRef::int_op(99));

        frame.cleanup_registers();

        // pyjitpl.py:121-127: int and float slots are untouched.
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(1)));
        assert_eq!(frame.int_values[0], Some(11));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(3)));
        assert_eq!(frame.float_values[0], Some(33));
        // pyjitpl.py:124-126: ref slots [0, num_regs_r()) are cleared.
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_values.iter().all(|v| v.is_none()));
        // pyjitpl.py:127: pushed_box is reset to None.
        assert_eq!(frame.pushed_box, None);
    }

    /// pyjitpl.py:236-255 `MIFrame.replace_active_box_in_frame`.
    ///
    /// Bank dispatch: `oldbox.type` selects which register array to scan;
    /// the other banks must NOT be touched.  Walk replaces every slot
    /// whose `Some(opref)` matches `oldbox` with `Some(newbox)`; non-
    /// matching slots stay untouched.
    #[test]
    fn replace_active_box_in_frame_replaces_only_matching_oprefs() {
        let jitcode = make_jitcode_with_regs(3, 2, 1);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef::int_op(7));
        frame.int_regs[1] = Some(OpRef::int_op(8));
        frame.int_regs[2] = Some(OpRef::int_op(7)); // duplicate — must also flip.
        frame.ref_regs[0] = Some(OpRef::ref_op(7)); // same raw u32 but different bank/variant.
        frame.float_regs[0] = Some(OpRef::float_op(7));

        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Int);

        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(42)));
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(8)));
        assert_eq!(frame.int_regs[2], Some(OpRef::int_op(42)));
        // Ref / float banks untouched — bank dispatch is by oldbox.type.
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(7)));
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(7)));
    }

    /// Empty bank short-circuit: pyjitpl.py:248 `if not count: return`.
    #[test]
    fn replace_active_box_in_frame_returns_early_when_bank_empty() {
        let jitcode = make_jitcode_with_regs(0, 1, 0);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.ref_regs[0] = Some(OpRef::ref_op(7));
        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Int);
        // Int bank empty — ref bank stays untouched.
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(7)));
    }

    /// Type::Void is not a valid box type (pyjitpl.py:246 `assert 0,
    /// oldbox`).
    #[test]
    #[should_panic(expected = "replace_active_box_in_frame: void oldbox")]
    fn replace_active_box_in_frame_void_type_panics_like_rpython() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef::int_op(7));
        frame.replace_active_box_in_frame(OpRef::int_op(7), OpRef::int_op(42), Type::Void);
    }

    #[test]
    fn make_result_of_lastop_stores_into_typed_slot() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);

        frame.pc = 0;
        frame.make_result_of_lastop(JitArgKind::Int, 1, OpRef::int_op(7), 77);
        assert_eq!(frame.int_regs[1], Some(OpRef::int_op(7)));
        assert_eq!(frame.int_values[1], Some(77));

        frame.pc = 1;
        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef::ref_op(8), 88);
        assert_eq!(frame.ref_regs[0], Some(OpRef::ref_op(8)));
        assert_eq!(frame.ref_values[0], Some(88));

        frame.pc = 2;
        frame.make_result_of_lastop(JitArgKind::Float, 0, OpRef::float_op(9), 99);
        assert_eq!(frame.float_regs[0], Some(OpRef::float_op(9)));
        assert_eq!(frame.float_values[0], Some(99));
    }

    #[test]
    fn make_result_of_lastop_accepts_matching_recorded_resulttype() {
        // RPython `pyjitpl.py:260-265`: in non-translated builds the
        // assertion compares the resultbox kind against
        // `jitcode._resulttypes[frame.pc]`.  pyre's writer-side
        // `JitCodeBuilder::record_resulttype` populates the map at
        // end-of-instruction position; this test asserts the reader
        // accepts a matching kind without panicking.
        use crate::JitCallArg;
        let mut builder = JitCodeBuilder::new();
        // Emit a canonical Pure residual_call that records 'i' at
        // end-of-instr per `assembler.py:217-219` (Parity #14 Slice C.5).
        let fn_idx = builder.add_fn_ptr(0x1usize as *const ());
        builder.call_pure_int_canonical_via_target(fn_idx, &[JitCallArg::int(0)], 0);
        let jitcode = Arc::new(builder.finish());
        let post_call_pc = jitcode.body().code.len();
        let recorded = jitcode
            .body()
            .resulttypes
            .as_ref()
            .and_then(|resulttypes| resulttypes.get(&post_call_pc).copied());
        assert_eq!(
            recorded,
            Some('i'),
            "writer must record 'i' at end-of-instruction PC for residual_call_int"
        );

        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.pc = post_call_pc;
        // Matching kind — assertion passes silently.
        frame.make_result_of_lastop(JitArgKind::Int, 0, OpRef::int_op(7), 77);
        assert_eq!(frame.int_regs[0], Some(OpRef::int_op(7)));
    }

    #[test]
    #[should_panic(expected = "_resulttypes")]
    fn make_result_of_lastop_panics_on_recorded_resulttype_mismatch() {
        // Same writer setup as the matching variant — but the reader
        // passes `Ref` against a recorded `'i'`, which fires the
        // RPython `pyjitpl.py:264` assertion port.
        use crate::JitCallArg;
        let mut builder = JitCodeBuilder::new();
        let fn_idx = builder.add_fn_ptr(0x1usize as *const ());
        builder.call_pure_int_canonical_via_target(fn_idx, &[JitCallArg::int(0)], 0);
        let jitcode = Arc::new(builder.finish());
        let post_call_pc = jitcode.body().code.len();

        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.pc = post_call_pc;
        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef::ref_op(7), 77);
    }

    #[test]
    fn setup_resume_at_op_assigns_pc() {
        let jitcode = make_jitcode_with_regs(0, 0, 0);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.setup_resume_at_op(123);
        assert_eq!(frame.pc, 123);
    }

    #[test]
    fn setup_call_with_empty_argboxes_only_resets_pc() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 42;
        frame.setup_call(&[]);
        assert_eq!(frame.pc, 0);
        assert!(frame.int_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.float_regs.iter().all(|r| r.is_none()));
    }

    #[test]
    fn setup_populates_constants_and_greenkey_inline() {
        let mut jitcode = make_jitcode_with_regs(1, 1, 1);
        {
            let jc = Arc::get_mut(&mut jitcode).expect("fresh Arc");
            jc.body_mut().constants_i = vec![123];
            jc.body_mut().constants_r = vec![0x1234];
            jc.body_mut().constants_f = vec![1.5f64.to_bits() as i64];
        }
        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut recorder = crate::recorder::Trace::new();
        let _ = recorder.record_input_arg(Type::Int);
        let mut ctx = crate::trace_ctx::TraceCtx::new(recorder, 0, sd);

        let frame = MIFrame::setup(jitcode, 7, Some(0xfeed), Some(&mut ctx));

        assert_eq!(frame.pc, 7);
        assert_eq!(frame.greenkey, Some(0xfeed));
        assert_eq!(frame._result_argcode, b'v');
        assert_eq!(frame.parent_snapshot, -1);
        assert_eq!(frame.unroll_iterations, 1);
        assert_eq!(frame.int_values[1], Some(123));
        assert_eq!(frame.ref_values[1], Some(0x1234));
        assert_eq!(frame.float_values[1], Some(1.5f64.to_bits() as i64));
        assert!(frame.int_regs[1].is_some());
        assert!(frame.ref_regs[1].is_some());
        assert!(frame.float_regs[1].is_some());
    }

    /// pyjitpl.py:1530-1535 — `verify_green_args` accepts only Const
    /// OpRefs whose count matches `jitdriver_sd.num_green_args`.
    #[test]
    fn verify_green_args_accepts_constants_matching_num_greens() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Int)],
            vec![("r0", Type::Int)],
        );
        let greens = vec![OpRef::from_const(0), OpRef::from_const(1)];
        // Must not panic — both opref are Const-tagged and length matches.
        MIFrame::verify_green_args(&jd, &greens);
    }

    #[test]
    #[should_panic(expected = "expected 2 greens")]
    fn verify_green_args_rejects_wrong_count() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(
            vec![("g0", Type::Int), ("g1", Type::Int)],
            vec![],
        );
        // Only one green provided — must fail count check.
        MIFrame::verify_green_args(&jd, &[OpRef::from_const(0)]);
    }

    #[test]
    #[should_panic(expected = "is not a Const")]
    fn verify_green_args_rejects_non_const_opref() {
        use majit_ir::Type;
        let jd = crate::jitdriver::JitDriverStaticData::new(vec![("g0", Type::Int)], vec![]);
        // Plain OpRef::int_op(0) is in the operation namespace (no CONST_BIT) —
        // upstream pyjitpl.py:1534 asserts isinstance(varargs[i], Const).
        MIFrame::verify_green_args(&jd, &[OpRef::int_op(0)]);
    }
}
