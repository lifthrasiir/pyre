/// aarch64/assembler.py: AssemblerARM64 — aarch64 JIT code generation backend.
///
/// Generates machine code from IR operations via dynasm-rs.
/// RPython: AssemblerARM64(ResOpAssembler) in aarch64/assembler.py.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use std::collections::HashMap;
use std::sync::Arc;

// aarch64/assembler.py parity: aarch64-only backend.
use dynasmrt::aarch64::Assembler;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use majit_backend::{BackendError, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout};
use majit_ir::{FailDescr, InputArg, LoopTargetDescr, Op, OpCode, OpRef, TargetArgLoc, Type};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::guard::DynasmFailDescr;
use crate::jitframe::{
    FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FRAME_OFS, JF_GCMAP_OFS,
    JF_GUARD_EXC_OFS,
};
use crate::regalloc::{RegAlloc, RegAllocOp};
use crate::regloc::Loc;
use crate::runner::GuardGcTypeInfo;

const AARCH64_GEN_REGS: [crate::regloc::RegLoc; 16] = crate::aarch64::registers::ALL_REGS;

const AARCH64_FLOAT_REGS: [crate::regloc::RegLoc; 8] = crate::aarch64::registers::ALL_VFP_REGS;

/// Resolved argument: either a frame slot (frame-pointer-relative offset) or a constant.
enum ResolvedArg {
    /// Frame-pointer-relative byte offset: [rbp + offset] on x64, [x29, #offset] on aarch64.
    Slot(i32),
    /// Immediate constant value.
    Const(i64),
}

fn loop_target_descr(op: &Op) -> Option<&dyn LoopTargetDescr> {
    op.descr
        .as_deref()
        .and_then(majit_ir::Descr::as_loop_target_descr)
}

/// Pointer-identity key for `target_tokens_currently_compiling`. PyPy
/// x86/assembler.py:93 keys it by the descr Python object itself; we use
/// the underlying allocation address of the `Arc<dyn Descr>` so two
/// distinct TargetToken descriptors are never confused.
fn loop_target_id(op: &Op) -> Option<usize> {
    op.descr.as_ref().map(majit_ir::descr_identity)
}

fn target_argloc_from_loc(loc: Loc) -> TargetArgLoc {
    match loc {
        Loc::Reg(r) => TargetArgLoc::Reg {
            regnum: r.value,
            is_xmm: r.is_xmm,
        },
        Loc::Ebp(e) => TargetArgLoc::Ebp {
            ebp_offset: e.value,
            is_float: e.is_float,
        },
        Loc::Frame(f) => TargetArgLoc::Frame {
            position: f.position,
            ebp_offset: f.ebp_loc.value,
            is_float: f.ebp_loc.is_float,
        },
        Loc::Immed(i) => TargetArgLoc::Immed {
            value: i.value,
            is_float: i.is_float,
        },
        Loc::Addr(a) => TargetArgLoc::Addr {
            base: a.base,
            index: a.index,
            scale: a.scale,
            offset: a.offset,
        },
    }
}

fn loc_from_target_argloc(loc: &TargetArgLoc) -> Loc {
    match *loc {
        TargetArgLoc::Reg { regnum, is_xmm } => {
            Loc::Reg(crate::regloc::RegLoc::new(regnum, is_xmm))
        }
        TargetArgLoc::Ebp {
            ebp_offset,
            is_float,
        } => Loc::Ebp(crate::regloc::RawEbpLoc {
            value: ebp_offset,
            is_float,
        }),
        TargetArgLoc::Frame {
            position,
            ebp_offset,
            is_float,
        } => Loc::Frame(crate::regloc::FrameLoc::new(position, ebp_offset, is_float)),
        TargetArgLoc::Immed { value, is_float } => {
            Loc::Immed(crate::regloc::ImmedLoc { value, is_float })
        }
        TargetArgLoc::Addr {
            base,
            index,
            scale,
            offset,
        } => Loc::Addr(crate::regloc::AddressLoc {
            base,
            index,
            scale,
            offset,
        }),
    }
}

fn all_gen_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_GEN_REGS
}

fn all_float_regs() -> &'static [crate::regloc::RegLoc] {
    &AARCH64_FLOAT_REGS
}

fn core_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_gen_regs()
        .iter()
        .position(|candidate| *candidate == reg)
}

fn float_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    all_float_regs()
        .iter()
        .position(|candidate| *candidate == reg)
        .map(|idx| all_gen_regs().len() + idx)
}

fn reg_position_in_jitframe(reg: crate::regloc::RegLoc) -> Option<usize> {
    if reg.is_xmm {
        float_reg_position(reg)
    } else {
        core_reg_position(reg)
    }
}

// ── Abstract condition codes ──
// Architecture-independent CC values used throughout the assembler.
// Converted to arch-specific encoding at emission time.
const CC_O: u8 = 0;
const CC_NO: u8 = 1;
const CC_B: u8 = 2; // unsigned <
const CC_AE: u8 = 3; // unsigned >=
const CC_E: u8 = 4; // ==
const CC_NE: u8 = 5; // !=
const CC_BE: u8 = 6; // unsigned <=
const CC_A: u8 = 7; // unsigned >
const CC_S: u8 = 8;
const CC_NS: u8 = 9;
const CC_L: u8 = 10; // signed <
const CC_GE: u8 = 11; // signed >=
const CC_LE: u8 = 12; // signed <=
const CC_G: u8 = 13; // signed >

/// Invert a condition code.
fn invert_cc(cc: u8) -> u8 {
    match cc {
        CC_O => CC_NO,
        CC_NO => CC_O,
        CC_B => CC_AE,
        CC_AE => CC_B,
        CC_E => CC_NE,
        CC_NE => CC_E,
        CC_BE => CC_A,
        CC_A => CC_BE,
        CC_S => CC_NS,
        CC_NS => CC_S,
        CC_L => CC_GE,
        CC_GE => CC_L,
        CC_LE => CC_G,
        CC_G => CC_LE,
        _ => CC_E, // fallback
    }
}

/// assembler.py:47 AssemblerARM64.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
pub struct AssemblerARM64 {
    /// The dynasm assembler (rx86.py + codebuf.py combined).
    pub(crate) mc: Assembler,
    /// assembler.py:83 pending_guard_tokens — guards awaiting recovery stubs.
    pending_guard_tokens: Vec<GuardToken>,
    /// GC bitmap to push before the current collecting call (e.g.,
    /// CallMallocNursery slow path). Set by the `RegAllocOp::Perform`
    /// emit path when `gcmap: Some(..)` is carried, cleared after.
    pending_malloc_nursery_gcmap: Option<usize>,
    /// Frame depth (in WORD units) for the current trace.
    frame_depth: usize,
    /// Fail descriptors built during assembly.
    fail_descrs: Vec<Arc<DynasmFailDescr>>,
    /// trace_id for this compilation.
    trace_id: u64,
    /// header_pc (green_key) for this compilation.
    header_pc: u64,
    /// Input argument types.
    input_types: Vec<Type>,
    /// assembler.py:641 rebuild_faillocs_from_descr parity:
    /// bridge input locations recovered from the source guard descr.
    bridge_input_locs: Option<Vec<Loc>>,

    // ── State tracking for code generation ──
    /// Maps OpRef index → jitframe slot index.
    opref_to_slot: HashMap<u32, usize>,
    /// resoperation.py Box.type parity: OpRef → Type for fail_arg_types
    /// inference. Populated during code generation from input types and
    /// op result types.
    value_types: HashMap<u32, Type>,
    /// Constants: OpRef index (>= 10000) → i64 value.
    constants: HashMap<u32, i64>,
    /// Constant type annotations for float immediates and fail args.
    constant_types: HashMap<u32, Type>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// x86/assembler.py:93 target_tokens_currently_compiling parity.
    /// Keyed by descriptor pointer identity (PyPy uses Python `is`).
    target_tokens_currently_compiling: HashMap<usize, DynamicLabel>,
    compiled_target_tokens: Vec<majit_ir::DescrRef>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// llsupport/gc.py:563 vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: HashMap<i64, u32>,
    /// TYPE_INFO / CLASSTYPE constants for `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`, fetched by the runner from the active gc_ll_descr.
    guard_gc_type_info: Option<GuardGcTypeInfo>,
    /// Constant classptr → `(subclassrange_min, subclassrange_max)`, matching
    /// `loc_check_against_class.getint()` field reads in
    /// `aarch64/opassembler.py:695-698`.
    classptr_to_subclass_range: HashMap<i64, (i64, i64)>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// Leaked pointer holding the resolved entry address for self-recursive
    /// CALL_ASSEMBLER via the execute trampoline. Written after finalization.
    self_entry_addr_ptr: *mut usize,
    /// assembler.py:320 descr._ll_function_addr parity:
    /// Maps call_target_token → compiled code address for CALL_ASSEMBLER.
    /// Populated by the runner before compilation, from registered loop targets.
    call_assembler_targets: HashMap<u64, usize>,
    /// opassembler.py:1177 _finish_gcmap.
    finish_gcmap: Option<*mut usize>,
    /// opassembler.py:1215 gcmap_for_finish.
    gcmap_for_finish: *mut usize,
    /// assembler.py:2207 _store_force_index parity:
    /// Pre-allocated fail descr for the next GUARD_NOT_FORCED, created
    /// at CALL_ASSEMBLER emission time so we can store its pointer to
    /// jf_force_descr before the call. Consumed by the subsequent
    /// GUARD_NOT_FORCED guard emission.
    pending_force_descr: Option<Arc<DynasmFailDescr>>,
    /// `compile.py:665-674` + `pyjitpl.py:2283`: snapshot of the six
    /// descr pointers attached to the owning cpu instance.  Matches
    /// RPython backend code that reads `self.cpu.done_with_this_frame_descr_*`
    /// / `self.cpu.exit_frame_with_exception_descr_ref` /
    /// `self.cpu.propagate_exception_descr` at every FINISH /
    /// CALL_ASSEMBLER emission site.
    attached_descrs: crate::guard::AttachedDescrPtrs,
    /// `Arc` clone of the owning cpu's attachment handle.  Its heap
    /// address is baked into the CALL_ASSEMBLER helper call site as a
    /// compile-time immediate (`Arc::as_ptr`) and the `Arc` is moved
    /// into the resulting `CompiledCode` so the pointee outlives any
    /// subsequent `DynasmBackend` drop — same role as RPython's
    /// `self.cpu` attribute-access after whole-program translation,
    /// where the `cpu` object's identity is guaranteed by Python.
    cpu_handle: crate::guard::CpuDescrHandle,
}

/// assembler.py GuardToken — represents a pending guard needing
/// a recovery stub to be written after the main loop body.
struct GuardToken {
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// The fail descriptor for this guard.
    fail_descr: Arc<DynasmFailDescr>,
    /// Constants to store in frame during recovery.
    /// Each entry: (frame_slot_index, constant_value).
    const_stores: Vec<(usize, i64)>,
    /// opassembler.py:515 GuardToken.gcmap.
    gcmap: *mut usize,
}

/// Compiled output from assemble_loop/assemble_bridge.
pub struct CompiledCode {
    /// Executable memory buffer (keeps code alive).
    pub buffer: ExecutableBuffer,
    /// Entry point offset within the buffer.
    pub entry_offset: AssemblyOffset,
    /// Fail descriptors for guards + FINISH ops.
    /// Frozen after compile — `Box<[T]>` reflects RPython's no-mutation
    /// contract (compile.py:183-203 record_loop_or_bridge). Position
    /// equals `descr.fail_index` by an invariant asserted at conversion
    /// from the in-progress `AssemblerARM64.fail_descrs` Vec.
    pub fail_descrs: Box<[Arc<DynasmFailDescr>]>,
    /// Input argument types.
    pub input_types: Vec<Type>,
    /// `compile.py:665-674` parity: `Arc` clone of the owning cpu's
    /// attachment handle.  Keeps the heap pointee alive for the whole
    /// lifetime of this compiled trace so the `cpu_handle` immediate
    /// baked into the CALL_ASSEMBLER helper call site never dangles,
    /// even if the emitting `DynasmBackend` is dropped first.
    pub cpu_attachments: crate::guard::CpuDescrHandle,
    /// trace_id.
    pub trace_id: u64,
    /// header_pc (green_key).
    pub header_pc: u64,
    /// Frame depth (number of jitframe slots used).
    /// AtomicUsize for redirect_call_assembler's update_frame_info
    /// parity: may be updated through &CompiledCode (shared ref).
    pub frame_depth: std::sync::atomic::AtomicUsize,
}

#[allow(dead_code)]
impl AssemblerARM64 {
    /// assembler.py:54 __init__
    pub(crate) fn new(
        trace_id: u64,
        header_pc: u64,
        constants: HashMap<u32, i64>,
        vtable_offset: Option<usize>,
        classptr_to_typeid: HashMap<i64, u32>,
        guard_gc_type_info: Option<GuardGcTypeInfo>,
        classptr_to_subclass_range: HashMap<i64, (i64, i64)>,
        attached_descrs: crate::guard::AttachedDescrPtrs,
        cpu_handle: crate::guard::CpuDescrHandle,
    ) -> Self {
        AssemblerARM64 {
            mc: Assembler::new().unwrap(),
            pending_guard_tokens: Vec::new(),
            pending_malloc_nursery_gcmap: None,
            frame_depth: JITFRAME_FIXED_SIZE,
            fail_descrs: Vec::new(),
            trace_id,
            header_pc,
            input_types: Vec::new(),
            bridge_input_locs: None,
            opref_to_slot: HashMap::new(),
            value_types: HashMap::new(),
            constants,
            constant_types: HashMap::new(),
            next_slot: 0,
            guard_success_cc: None,
            target_tokens_currently_compiling: HashMap::new(),
            compiled_target_tokens: Vec::new(),
            vtable_offset,
            classptr_to_typeid,
            guard_gc_type_info,
            classptr_to_subclass_range,
            self_entry_label: None,
            self_entry_addr_ptr: Box::into_raw(Box::new(0usize)),
            call_assembler_targets: HashMap::new(),
            finish_gcmap: None,
            gcmap_for_finish: {
                let gcmap = allocate_gcmap(1, JITFRAME_FIXED_SIZE);
                gcmap_set_bit(gcmap, 0);
                gcmap
            },
            pending_force_descr: None,
            attached_descrs,
            cpu_handle,
        }
    }

    /// `compile.py:665` parity: heap-pinned address of `self.cpu`'s
    /// attachment handle, derived from the Arc clone.  Baked into the
    /// CALL_ASSEMBLER helper call site.
    fn cpu_handle_ptr(&self) -> i64 {
        Arc::as_ptr(&self.cpu_handle) as *const () as i64
    }

    /// `compile.py:665-674` parity: attach the six metainterp descrs on
    /// the emission side.  Mirrors `self.cpu.done_with_this_frame_descr_*`
    /// reads in `rpython/jit/backend/aarch64/assembler.py`.
    fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> i64 {
        self.attached_descrs
            .done_with_this_frame_descr_ptr_for_type(tp) as i64
    }

    /// `compile.py:658` parity: `self.cpu.exit_frame_with_exception_descr_ref`.
    fn exit_frame_with_exception_descr_ref_ptr(&self) -> i64 {
        self.attached_descrs.exit_frame_with_exception_descr_ref as i64
    }

    /// `pyjitpl.py:2283` parity: `self.cpu.propagate_exception_descr`.
    /// Stamped into `jf_descr` by the inline propagate path emitted at
    /// `OpCode::CheckMemoryError` (opassembler.py:258
    /// `emit_op_check_memory_error` → `propagate_memoryerror_if_reg_is_null`).
    fn propagate_exception_descr_ptr(&self) -> i64 {
        self.attached_descrs.propagate_exception_descr as i64
    }

    // ----------------------------------------------------------------
    // Helper methods
    // ----------------------------------------------------------------

    /// Frame-pointer-relative byte offset for a given slot index.
    /// Slots are absolute jf_frame indices, including the fixed
    /// JITFRAME-managed prefix. FIRST_ITEM_OFFSET accounts for the object
    /// header and array-length word that precede jf_frame[0].
    fn slot_offset(slot: usize) -> i32 {
        FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32
    }

    /// Resolve an OpRef to either a frame slot offset or an immediate constant.
    /// Cranelift resolve_opref parity: check constants map FIRST
    /// (regardless of CONST_BIT), then fall back to slot mapping.
    fn resolve_opref(&self, opref: OpRef) -> ResolvedArg {
        // Op results take precedence over constants (Cranelift parity).
        if let Some(&slot) = self.opref_to_slot.get(&opref.0) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        if let Some(&val) = self.constants.get(&opref.0) {
            return ResolvedArg::Const(val);
        }
        // Unmapped OpRef — treat as 0.
        ResolvedArg::Const(0)
    }

    /// Allocate a frame slot for an OpRef and return the ABSOLUTE jitframe
    /// slot index (already offset by JITFRAME_FIXED_SIZE so slot_offset()
    /// yields the user-area byte offset directly).
    ///
    /// Reuses the existing slot if the OpRef has already been allocated.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref.0) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref.0, slot);
        slot
    }

    // ── Location-aware code emission (RPython regalloc parity) ──
    // assembler.py regalloc_mov: move value between any two locations.

    /// assembler.py:1145 regalloc_mov(from_loc, to_loc).
    /// Emit a move between any two locations: reg↔reg, reg↔frame, imm→reg, imm→frame.
    pub(crate) fn regalloc_mov(&mut self, src: &Loc, dst: &Loc) {
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm] remap-mov: {:?} -> {:?}", src, dst);
        }
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            (Loc::Reg(s), Loc::Reg(d)) => {
                if s.is_xmm && d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), D(s.value));
                } else if !s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(d.value), X(s.value));
                } else if s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; fmov X(d.value), D(s.value));
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(s.value));
                }
            }
            (Loc::Reg(s), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                if s.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; str D(s.value), [x29, ofs as u32]);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; str X(s.value), [x29, ofs as u32]);
                }
            }
            (Loc::Frame(f), Loc::Reg(d)) => {
                let ofs = f.ebp_loc.value;
                if d.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; ldr D(d.value), [x29, ofs as u32]);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(d.value), [x29, ofs as u32]);
                }
            }
            (Loc::Immed(i), Loc::Reg(d)) => {
                if d.is_xmm {
                    self.emit_mov_imm64(16, i.value); // x16 = scratch
                    dynasm!(self.mc ; .arch aarch64 ; fmov D(d.value), X(16));
                } else {
                    self.emit_mov_imm64(d.value as u32, i.value);
                }
            }
            (Loc::Immed(i), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                self.emit_mov_imm64(16, i.value);
                dynasm!(self.mc ; .arch aarch64 ; str x16, [x29, ofs as u32]);
            }
            (Loc::Frame(f1), Loc::Frame(f2)) if f1.position == f2.position => {}
            (Loc::Frame(f1), Loc::Frame(f2)) => {
                let o1 = f1.ebp_loc.value;
                let o2 = f2.ebp_loc.value;
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x16, [x29, o1 as u32]
                    ; str x16, [x29, o2 as u32]
                );
            }
            _ => {}
        }
    }

    fn loc_as_key(loc: &Loc) -> i32 {
        match loc {
            Loc::Reg(r) if r.is_xmm => 0x2000 + i32::from(r.value),
            Loc::Reg(r) => 0x1000 + i32::from(r.value),
            Loc::Frame(f) => f.ebp_loc.value,
            Loc::Ebp(e) => e.value,
            Loc::Immed(_) => i32::MIN,
            Loc::Addr(a) => a.offset,
        }
    }

    fn loc_width(loc: &Loc) -> usize {
        match loc {
            Loc::Reg(r) => r.get_width(),
            Loc::Frame(f) => f.ebp_loc.get_width(),
            Loc::Ebp(e) => e.get_width(),
            _ => WORD,
        }
    }

    fn regalloc_push(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; str D(r.value), [sp, #-16]!);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; str X(r.value), [sp, #-16]!);
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(15), [x29, f.ebp_loc.value as u32] ; str D(15), [sp, #-16]!);
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32] ; str x16, [sp, #-16]!);
            }
            _ => {}
        }
    }

    fn regalloc_pop(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(r.value), [sp], #16);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(r.value), [sp], #16);
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch aarch64 ; ldr D(15), [sp], #16 ; str D(15), [x29, f.ebp_loc.value as u32]);
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [sp], #16 ; str x16, [x29, f.ebp_loc.value as u32]);
            }
            _ => {}
        }
    }

    fn remap_frame_layout(&mut self, src_locations: &[Loc], dst_locations: &[Loc], tmpreg: Loc) {
        let mut pending_dests = dst_locations.len() as i32;
        let mut srccount: HashMap<i32, i32> = HashMap::new();
        for dst in dst_locations {
            srccount.insert(Self::loc_as_key(dst), 0);
        }
        for i in 0..dst_locations.len() {
            let src = src_locations[i];
            if src.is_immed() {
                continue;
            }
            let key = Self::loc_as_key(&src);
            if let Some(cnt) = srccount.get_mut(&key) {
                if key == Self::loc_as_key(&dst_locations[i]) {
                    *cnt = -(dst_locations.len() as i32) - 1;
                    pending_dests -= 1;
                } else {
                    *cnt += 1;
                }
            }
        }

        while pending_dests > 0 {
            let mut progress = false;
            for i in 0..dst_locations.len() {
                let dst = dst_locations[i];
                let key = Self::loc_as_key(&dst);
                if srccount.get(&key).copied().unwrap_or(-1) == 0 {
                    srccount.insert(key, -1);
                    pending_dests -= 1;
                    let src = src_locations[i];
                    if !src.is_immed() {
                        let src_key = Self::loc_as_key(&src);
                        if let Some(cnt) = srccount.get_mut(&src_key) {
                            *cnt -= 1;
                        }
                    }
                    if dst.is_stack() && src.is_stack() {
                        self.regalloc_mov(&src, &tmpreg);
                        self.regalloc_mov(&tmpreg, &dst);
                    } else {
                        self.regalloc_mov(&src, &dst);
                    }
                    progress = true;
                }
            }
            if !progress {
                let mut sources: HashMap<i32, Loc> = HashMap::new();
                for i in 0..dst_locations.len() {
                    sources.insert(Self::loc_as_key(&dst_locations[i]), src_locations[i]);
                }
                for dst in dst_locations {
                    let originalkey = Self::loc_as_key(dst);
                    if srccount.get(&originalkey).copied().unwrap_or(-1) >= 0 {
                        self.regalloc_push(dst);
                        let mut cur_dst = *dst;
                        loop {
                            let key = Self::loc_as_key(&cur_dst);
                            srccount.insert(key, -1);
                            pending_dests -= 1;
                            let src = sources[&key];
                            if Self::loc_as_key(&src) == originalkey {
                                break;
                            }
                            if cur_dst.is_stack() && src.is_stack() {
                                self.regalloc_mov(&src, &tmpreg);
                                self.regalloc_mov(&tmpreg, &cur_dst);
                            } else {
                                self.regalloc_mov(&src, &cur_dst);
                            }
                            cur_dst = src;
                        }
                        self.regalloc_pop(&cur_dst);
                    }
                }
            }
        }
    }

    fn remap_frame_layout_mixed(
        &mut self,
        src_locations1: &[Loc],
        dst_locations1: &[Loc],
        tmpreg1: Loc,
        src_locations2: &[Loc],
        dst_locations2: &[Loc],
        tmpreg2: Loc,
    ) {
        let mut extrapushes = Vec::new();
        let mut dst_keys = HashMap::new();
        for loc in dst_locations1 {
            dst_keys.insert(Self::loc_as_key(loc), ());
        }
        let mut src_locations2red = Vec::new();
        let mut dst_locations2red = Vec::new();
        for i in 0..src_locations2.len() {
            let loc = src_locations2[i];
            let dstloc = dst_locations2[i];
            if loc.is_stack() {
                let key = Self::loc_as_key(&loc);
                if dst_keys.contains_key(&key)
                    || (Self::loc_width(&loc) > WORD && dst_keys.contains_key(&(key + WORD as i32)))
                {
                    self.regalloc_push(&loc);
                    extrapushes.push(dstloc);
                    continue;
                }
            }
            src_locations2red.push(loc);
            dst_locations2red.push(dstloc);
        }
        self.remap_frame_layout(src_locations1, dst_locations1, tmpreg1);
        self.remap_frame_layout(&src_locations2red, &dst_locations2red, tmpreg2);
        while let Some(loc) = extrapushes.pop() {
            self.regalloc_pop(&loc);
        }
    }

    /// Load a (lhs, src) pair for a 3-operand binop into registers,
    /// returning the register numbers to use. Uses x17 for lhs scratch
    /// and x16 for src scratch when the loc is Frame/Immed.
    fn load_3op_into_scratch(&mut self, lhs: &Loc, src: &Loc) -> (u8, u8) {
        let lhs_reg = match lhs {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                17
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(17, i.value);
                17
            }
            _ => 17,
        };
        let src_reg = match src {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => 16,
        };
        (lhs_reg, src_reg)
    }

    /// Emit a 3-operand integer binop: `OP dst, lhs, src`.
    ///
    /// aarch64/opassembler.py parity: `ADD/SUB/AND/ORR/EOR Rd, Rn, Rm` —
    /// the hardware supports distinct Rd and Rn, unlike x86's 2-operand
    /// `SUB dst, dst, src`. Handles reg/frame/immediate forms for both
    /// operands, routing scratch via x16/x17 as needed.
    fn emit_binop_3op(&mut self, opcode: OpCode, dst_reg: u8, lhs: &Loc, src: &Loc) {
        let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
        let d = dst_reg;
        let l = lhs_reg as u8;
        let s = src_reg as u8;
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                dynasm!(self.mc ; .arch aarch64 ; add X(d), X(l), X(s));
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(l), X(s));
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(l), X(s));
            }
            OpCode::IntAnd => {
                dynasm!(self.mc ; .arch aarch64 ; and X(d), X(l), X(s));
            }
            OpCode::IntOr => {
                dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(l), X(s));
            }
            OpCode::IntXor => {
                dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(l), X(s));
            }
            _ => {}
        }
    }

    /// Emit: ADD/SUB/AND/OR/XOR reg, loc
    fn emit_binop_reg_loc(&mut self, opcode: OpCode, dst_reg: u8, src: &Loc) {
        // aarch64: load src to x16 scratch if not in register
        let src_reg = match src {
            Loc::Reg(s) => s.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        let d = dst_reg;
        let s = src_reg as u8;
        match opcode {
            OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                dynasm!(self.mc ; .arch aarch64 ; add X(d), X(d), X(s));
            }
            OpCode::IntSub | OpCode::IntSubOvf => {
                dynasm!(self.mc ; .arch aarch64 ; sub X(d), X(d), X(s));
            }
            OpCode::IntMul | OpCode::IntMulOvf => {
                dynasm!(self.mc ; .arch aarch64 ; mul X(d), X(d), X(s));
            }
            OpCode::IntAnd => {
                dynasm!(self.mc ; .arch aarch64 ; and X(d), X(d), X(s));
            }
            OpCode::IntOr => {
                dynasm!(self.mc ; .arch aarch64 ; orr X(d), X(d), X(s));
            }
            OpCode::IntXor => {
                dynasm!(self.mc ; .arch aarch64 ; eor X(d), X(d), X(s));
            }
            _ => {}
        }
        return;
    }

    /// Emit: CMP loc0, loc1
    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        // Load loc0 into x16 if needed, loc1 into x17 if needed
        let r0 = match loc0 {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        let r1 = match loc1 {
            Loc::Reg(s) => s.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                17
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(17, i.value);
                17
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; cmp X(r0 as u8), X(r1 as u8));
        return;
    }

    /// Emit: TEST loc, loc (for guard_true/guard_false)
    fn emit_test_loc(&mut self, loc: &Loc) {
        let r = match loc {
            Loc::Reg(r) => r.value,
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                16
            }
            Loc::Immed(i) => {
                self.emit_mov_imm64(16, i.value);
                16
            }
            _ => return,
        };
        dynasm!(self.mc ; .arch aarch64 ; tst X(r as u8), X(r as u8));
        return;
    }

    // ── AArch64 helper: load a 64-bit immediate into register Xn ──
    pub(crate) fn emit_mov_imm64(&mut self, reg: u32, val: i64) {
        let v = val as u64;
        let r = reg as u8;
        dynasm!(self.mc ; .arch aarch64
            ; movz X(r), (v & 0xFFFF) as u32
            ; movk X(r), ((v >> 16) & 0xFFFF) as u32, lsl 16
            ; movk X(r), ((v >> 32) & 0xFFFF) as u32, lsl 32
            ; movk X(r), ((v >> 48) & 0xFFFF) as u32, lsl 48
        );
    }

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(0, val);
            }
        }
    }

    /// Emit: load a regalloc Loc into RAX (x64) / X0 (aarch64).
    /// Unlike load_arg_to_rax, this uses the regalloc-determined location
    /// instead of resolve_opref(), so register-carried values are preserved.
    fn emit_load_to_rax(&mut self, loc: Loc) {
        let rax = Loc::Reg(crate::regloc::RegLoc {
            value: 0,
            is_xmm: false,
        });
        match loc {
            Loc::Reg(r) if r.value == 0 && !r.is_xmm => {
                // already in rax/x0
            }
            Loc::Immed(imm) => {
                self.emit_mov_imm64(0, imm.value);
            }
            _ => self.regalloc_mov(&loc, &rax),
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x1, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(1, val);
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, offset as u32]
        );
    }

    // ----------------------------------------------------------------
    // assembler.py:543 _call_header — function prologue
    // ----------------------------------------------------------------

    fn setup_input_state(&mut self, inputargs: &[InputArg]) {
        // opref_to_slot stores ABSOLUTE jitframe slot indices so that
        // slot_offset(slot) returns the correct byte offset directly.
        // User position `p` maps to absolute slot `p + JITFRAME_FIXED_SIZE`.
        if let Some(ref input_locs) = self.bridge_input_locs {
            let mut max_abs_slot = JITFRAME_FIXED_SIZE;
            for (ia, loc) in inputargs.iter().zip(input_locs.iter()) {
                if let Loc::Frame(floc) = loc {
                    let abs_slot = JITFRAME_FIXED_SIZE + floc.position;
                    self.opref_to_slot.insert(ia.index, abs_slot);
                    if abs_slot + 1 > max_abs_slot {
                        max_abs_slot = abs_slot + 1;
                    }
                }
                self.value_types.insert(ia.index, ia.tp);
            }
            self.next_slot = max_abs_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot.insert(ia.index, JITFRAME_FIXED_SIZE + i);
                self.value_types.insert(ia.index, ia.tp);
            }
            self.next_slot = JITFRAME_FIXED_SIZE + inputargs.len();
        }
    }

    /// Emit the function prologue.
    /// x64: System V AMD64 ABI — first arg (jf_ptr) in RDI.
    /// aarch64: AAPCS64 — first arg (jf_ptr) in X0.
    ///
    /// Save slots: 32 bytes total (fp/lr at [sp,#0], x19/x20 at
    /// [sp,#16]).  Mirrors aarch64/assembler.py:1117-1122 which
    /// iterates `r.callee_saved_registers = [x19, x20]`.
    ///
    /// aarch64/assembler.py:1092-1114 `_call_header_with_stack_check`:
    /// inline SP probe right after the frame pointer is captured,
    /// before `gen_shadowstack_header`. On overflow, return the
    /// incoming jf_ptr in x0 without pushing to the shadow stack so
    /// the JIT glue drains the overflow flag on the way back to the
    /// interpreter.
    ///
    /// Inline probe (aarch64/assembler.py:1099-1114 parity):
    /// ```text
    ///   gen_load_int x30, endaddr       ; load endaddr
    ///   LDR  x30, [x30]                  ; x30 = end
    ///   gen_load_int x16, lengthaddr
    ///   LDR  x16, [x16]                  ; x16 = length
    ///   MOV  x17, sp                     ; x17 = current sp (sp can't be
    ///                                    ;   the source of SUB_rr directly)
    ///   SUB  x30, x30, x17               ; x30 = ofs = end - sp
    ///   CMP  x30, x16
    ///   B.LS continue                    ; fast path: ofs <= length
    ///   MOV  x0, sp                      ; arg0 = current sp
    ///   gen_load_int x17, slowpath
    ///   BLR  x17                         ; call pyre_stack_too_big_slowpath
    ///   CBZ  w0, continue                ; slowpath: 0 = OK
    ///   ; fallthrough = real overflow → return x29 as jf_ptr
    /// ```
    fn _call_header(&mut self, inputargs: &[InputArg]) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x29, x30, [sp, #-32]!
            ; stp x19, x20, [sp, #16]   // save callee-saved regs
            ; mov x29, x0
        );
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr != 0 {
            if let Some(addrs) = crate::stack_check_addresses() {
                let continue_label = self.mc.new_dynamic_label();
                let exc_value_addr = crate::jit_exc_value_addr() as i64;
                let exc_type_addr = crate::jit_exc_type_addr() as i64;
                // Fast path: load end, subtract sp, compare with length.
                self.emit_mov_imm64(30, addrs.end_adr as i64); // lr holds end addr
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x30, [x30]                  // x30 = *end_adr
                );
                self.emit_mov_imm64(16, addrs.length_adr as i64); // ip0 holds length addr
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x16, [x16]                  // x16 = *length_adr
                    ; mov x17, sp                     // x17 = sp (can't SUB sp directly)
                    ; sub x30, x30, x17               // x30 = ofs = end - sp
                    ; cmp x30, x16
                    ; b.ls =>continue_label           // fast path OK: ofs <= length
                    // Slow path: call pyre_stack_too_big_slowpath(sp).
                    ; mov x0, sp
                );
                self.emit_mov_imm64(17, addrs.slowpath_addr as i64);
                dynasm!(self.mc ; .arch aarch64
                    ; blr x17
                    ; cbz w0, =>continue_label        // slowpath says OK
                );
                // aarch64/assembler.py `_build_stack_check_slowpath` jumps
                // into the same propagate-exception path as x86: move
                // pos_exc_value into jf_guard_exc, clear pos_exception, stamp
                // propagate_exception_descr, then return the incoming jf_ptr.
                self.emit_mov_imm64(16, exc_value_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x16]
                    ; str xzr, [x16]
                    ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
                );
                self.emit_mov_imm64(16, exc_type_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; str xzr, [x16]
                );
                self.emit_mov_imm64(0, propagate_descr);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, JF_DESCR_OFS as u32]
                    // Overflow fallthrough: return x29 as jf_ptr.
                    ; mov x0, x29
                    ; ldp x19, x20, [sp, #16]
                    ; ldp x29, x30, [sp], #32
                    ; ret
                    ; =>continue_label
                );
            }
        }
        // When addresses are not registered (tests / early startup), no
        // stack check is emitted — aarch64/assembler.py:1094-1095 parity.
        self.gen_shadowstack_header();
        self.setup_input_state(inputargs);
    }

    // ----------------------------------------------------------------
    // assembler.py:2153 _call_footer — function epilogue
    // ----------------------------------------------------------------

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    fn _call_footer(&mut self) {
        self.gen_footer_shadowstack();
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
            ; ldp x19, x20, [sp, #16]   // restore callee-saved regs
            ; ldp x29, x30, [sp], #32
            ; ret
        );
    }

    /// aarch64/assembler.py:1422 `gen_shadowstack_header` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip1.value, rst)
    ///   self.load_reg(mc, r.x8, r.ip1)   # x8 = *rst = root_stack_top
    ///   mc.gen_load_int(r.ip0.value, 1)
    ///   self.store_reg(mc, r.ip0, r.x8)  # x8[0] = 1 (is_minor marker)
    ///   self.store_reg(mc, r.fp, r.x8, WORD) # x8[WORD] = fp (jf_ptr)
    ///   mc.ADD_ri(r.x8.value, r.x8.value, 2 * WORD)
    ///   self.store_reg(mc, r.x8, r.ip1)  # *rst = x8
    /// ```
    ///
    /// Pushes two words onto the jf shadow stack: the `is_minor` marker
    /// (1) and the current jitframe pointer. Done inline on every JIT
    /// function entry so the collector can walk jf roots without ever
    /// calling into Rust.
    fn gen_shadowstack_header(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        // Register assignment mirrors assembler.py:1426-1436 literally:
        // r.ip1 = x17 holds the rst address; r.ip0 = x16 holds the
        // `1` is_minor marker; r.x8 = x8 holds the loaded top.
        self.emit_mov_imm64(17, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x8, [x17]             // x8 = *rst = root_stack_top
            ; mov x16, 1                 // is_minor marker
            ; str x16, [x8]             // [x8] = 1
            ; str x29, [x8, 8]          // [x8 + WORD] = fp (jf_ptr)
            ; add x8, x8, 16            // x8 += 2*WORD
            ; str x8, [x17]             // *rst = x8
        );
    }

    /// aarch64/assembler.py:1438 `gen_footer_shadowstack` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip1, r.ip0)  # ip1 = *rst = top
    ///   mc.SUB_ri(r.ip1.value, r.ip1.value, 2 * WORD)
    ///   self.store_reg(mc, r.ip1, r.ip0) # *rst = ip1
    /// ```
    fn gen_footer_shadowstack(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x17, [x16]            // x17 = *rst = top
            ; sub x17, x17, 16          // top -= 2*WORD
            ; str x17, [x16]            // *rst = top
        );
    }

    /// assembler.py:993 push_gcmap.
    fn push_gcmap(&mut self, gcmap: *mut usize) {
        let gcmap_ptr = gcmap as i64;
        self.emit_mov_imm64(0, gcmap_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// assembler.py:1000 pop_gcmap.
    fn pop_gcmap(&mut self) {
        dynasm!(self.mc ; .arch aarch64
            ; str xzr, [x29, JF_GCMAP_OFS as u32]
        );
    }

    /// aarch64/assembler.py:967 `_reload_frame_if_necessary` parity:
    ///
    /// ```python
    ///   rst = gcrootmap.get_root_stack_top_addr()
    ///   mc.gen_load_int(r.ip0.value, rst)
    ///   self.load_reg(mc, r.ip0, r.ip0)       # ip0 = *rst = root_stack_top
    ///   mc.SUB_ri(r.ip0.value, r.ip0.value, WORD)
    ///   mc.LDR_ri(r.fp.value, r.ip0.value, 0) # fp = *(top - WORD) = jf_ptr
    /// ```
    ///
    /// After a collecting helper call the GC may have moved the
    /// jitframe; the visitor rewrites the shadow-stack top entry in
    /// place during copy, so the current pointer lives at
    /// `*(root_stack_top - WORD)`. Reload fp/x29 from there.
    fn reload_frame_if_necessary(&mut self) {
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        self.emit_mov_imm64(16, rst_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]            // ip0 = *rst_addr = root_stack_top
            ; sub x16, x16, 8           // ip0 -= WORD
            ; ldr x29, [x16]            // fp = *(top - WORD) = jf_ptr
        );
    }

    fn guard_gcmap_from_faillocs(
        &self,
        fail_arg_types: &[Type],
        faillocs: &[Option<Loc>],
    ) -> *mut usize {
        let frame_depth = self.frame_depth.saturating_sub(JITFRAME_FIXED_SIZE);
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(faillocs.iter()) {
            if *tp != Type::Ref {
                continue;
            }
            match loc {
                Some(Loc::Reg(r)) => {
                    if let Some(position) = reg_position_in_jitframe(*r) {
                        gcmap_set_bit(gcmap, position);
                    }
                }
                Some(Loc::Frame(f)) => {
                    gcmap_set_bit(gcmap, f.position + JITFRAME_FIXED_SIZE);
                }
                _ => {}
            }
        }
        gcmap
    }

    fn gcmap_from_fail_arg_locs(
        &self,
        fail_arg_types: &[Type],
        fail_arg_locs: &[Option<usize>],
    ) -> *mut usize {
        let frame_depth = self.frame_depth.saturating_sub(JITFRAME_FIXED_SIZE);
        let gcmap = allocate_gcmap(frame_depth, JITFRAME_FIXED_SIZE);
        for (tp, loc) in fail_arg_types.iter().zip(fail_arg_locs.iter()) {
            if *tp == Type::Ref {
                if let Some(position) = loc {
                    gcmap_set_bit(gcmap, *position);
                }
            }
        }
        gcmap
    }

    // ----------------------------------------------------------------
    // assembler.py:501 assemble_loop
    // ----------------------------------------------------------------

    /// assembler.py:501 assemble_loop: compile a loop trace.
    ///
    /// Returns compiled code with fail descriptors and entry point.
    pub fn assemble_loop(
        mut self,
        inputargs: &[InputArg],
        ops: &[Op],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = inputargs.iter().map(|ia| ia.tp).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        self._assemble(inputargs, ops, true)?;

        // regalloc sets fail_arg_locs in append_guard_token_with_faillocs.
        // No allocate_unmapped_fail_arg_slots needed.

        // assembler.py:553 write_pending_failure_recoveries
        let stub_offsets = self.write_pending_failure_recoveries();

        // assembler.py:556 materialize_loop — finalize to executable memory
        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        // assembler.py:849 patch_pending_failure_recoveries
        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // Write resolved entry address for self-recursive CALL_ASSEMBLER
        // trampoline. The JIT code loads from this pointer at runtime.
        unsafe { *self.self_entry_addr_ptr = rawstart + entry.0 };

        for descr in &self.compiled_target_tokens {
            if let Some(loop_descr) = descr.as_loop_target_descr() {
                loop_descr.set_ll_loop_code(loop_descr.ll_loop_code() + rawstart);
            }
        }

        // Load-bearing identity invariant for runtime dispatch: pyre's
        // guard-fail trampoline reads `jitframe.jf_descr_index` and indexes
        // `compiled.fail_descrs[idx]` directly (runner.rs::find_descr_by_ptr
        // fast path + assembler-emitted store of `fail_index` into
        // `jf_descr` slot).  Promoting from `debug_assert!` so a release
        // build catches the position/fail_index mismatch eagerly rather
        // than dispatching to the wrong descr at runtime.
        assert!(
            self.fail_descrs
                .iter()
                .enumerate()
                .all(|(i, d)| d.fail_index as usize == i),
            "fail_descrs position must equal fail_index"
        );
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
        })
    }

    /// assembler.py:320 descr._ll_function_addr parity: store
    /// call_target_token → code_addr mappings for CALL_ASSEMBLER.
    pub fn set_call_assembler_targets(&mut self, targets: HashMap<u64, usize>) {
        self.call_assembler_targets = targets;
    }

    /// llsupport/assembler.py:201 rebuild_faillocs_from_descr — reconstruct
    /// the locations of bridge inputargs from the guard's recovery layout.
    ///
    /// patch_jump_for_descr overwrites the recovery stub with a direct
    /// jump to the bridge, so the register-save subroutine never runs.
    /// The bridge sees live registers exactly as they were at guard time.
    /// Return Reg locs for register positions, matching RPython.
    pub fn rebuild_faillocs_from_descr(
        descr: &DynasmFailDescr,
        inputargs: &[InputArg],
    ) -> Vec<Loc> {
        let mut locs = Vec::new();
        let gpr_regs = all_gen_regs();
        let float_regs = all_float_regs();
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        let mut input_i = 0usize;
        for &pos in &descr.rd_locs {
            if pos == 0xFFFF {
                continue;
            }
            let pos = pos as usize;
            if pos < gpr_regs.len() {
                // llsupport/assembler.py:211 — GPR: return register location
                locs.push(Loc::Reg(gpr_regs[pos]));
            } else if pos < gpr_regs.len() + float_regs.len() {
                // llsupport/assembler.py:213 — FPR: return float register
                locs.push(Loc::Reg(float_regs[pos - gpr_regs.len()]));
            } else {
                // llsupport/assembler.py:217 — frame slot
                let slot = pos - JITFRAME_FIXED_SIZE;
                let tp = inputargs.get(input_i).map(|ia| ia.tp).unwrap_or(Type::Int);
                locs.push(Loc::Frame(crate::regloc::FrameLoc::new(
                    slot,
                    crate::regalloc::get_ebp_ofs(base_ofs, slot),
                    tp == Type::Float,
                )));
            }
            input_i += 1;
        }
        locs
    }

    /// assembler.py:623 assemble_bridge: compile a bridge trace.
    pub fn assemble_bridge(
        mut self,
        _: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] assemble_bridge: input_types={:?} arglocs={:?}",
                inputargs.iter().map(|ia| ia.tp).collect::<Vec<_>>(),
                arglocs
            );
        }
        self.input_types = inputargs.iter().map(|ia| ia.tp).collect();
        self.bridge_input_locs = if arglocs.is_empty() {
            None
        } else {
            Some(arglocs.to_vec())
        };

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(inputargs, ops, false)?;
        let stub_offsets = self.write_pending_failure_recoveries();

        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // Load-bearing identity invariant for runtime dispatch: pyre's
        // guard-fail trampoline reads `jitframe.jf_descr_index` and indexes
        // `compiled.fail_descrs[idx]` directly (runner.rs::find_descr_by_ptr
        // fast path + assembler-emitted store of `fail_index` into
        // `jf_descr` slot).  Promoting from `debug_assert!` so a release
        // build catches the position/fail_index mismatch eagerly rather
        // than dispatching to the wrong descr at runtime.
        assert!(
            self.fail_descrs
                .iter()
                .enumerate()
                .all(|(i, d)| d.fail_index as usize == i),
            "fail_descrs position must equal fail_index"
        );
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        emit_prologue: bool,
    ) -> Result<(), BackendError> {
        if emit_prologue {
            self._call_header(inputargs);
        } else {
            self.setup_input_state(inputargs);
        }
        // input_slot_depth is an ABSOLUTE jitframe slot count (already
        // offset by JITFRAME_FIXED_SIZE via setup_input_state).
        let input_slot_depth = self.next_slot;

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        if std::env::var_os("MAJIT_J2PLAN_LOG").is_some() {
            let plan = crate::j2plan::TracePlan::build(inputargs, ops);
            eprintln!("[dynasm:j2plan] {}", plan.summary());
        }

        let mut ra = RegAlloc::new(self.constants.clone(), self.constant_types.clone());
        if let Some(ref arglocs) = self.bridge_input_locs {
            ra.prepare_bridge(inputargs, arglocs, ops);
        } else {
            ra.prepare_loop(inputargs, ops);
        }
        // assembler.py:374 walk_operations — get allocation decisions.
        let ra_ops = ra.walk_operations(inputargs, ops);
        // ra.get_final_frame_depth() returns a USER-position count; convert
        // to absolute by adding JITFRAME_FIXED_SIZE before comparing.
        let frame_slot_depth =
            input_slot_depth.max(JITFRAME_FIXED_SIZE + ra.get_final_frame_depth());
        self.frame_depth = self.frame_depth.max(frame_slot_depth);

        // Sync regalloc frame positions to opref_to_slot for backward
        // compatibility with genop_call/genop_call_assembler which still
        // use resolve_opref. When regalloc spills a value to a frame slot,
        // that slot's position must be visible to resolve_opref.
        // opref_to_slot stores ABSOLUTE jitframe slots (user position +
        // JITFRAME_FIXED_SIZE) so slot_offset(slot) gives the correct byte
        // offset without further adjustment.
        for iarg in inputargs {
            self.opref_to_slot
                .insert(iarg.index, JITFRAME_FIXED_SIZE + iarg.index as usize);
            self.value_types.insert(iarg.index, iarg.tp);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        for (&opref, lifetime) in ra.longevity.lifetimes_iter() {
            if let Some(floc) = lifetime.current_frame_loc {
                self.opref_to_slot
                    .insert(opref.0, JITFRAME_FIXED_SIZE + floc.position);
            }
        }
        // frame_slot_depth is already absolute (see calculation above).
        self.next_slot = frame_slot_depth;

        let mut fail_index = 0u32;

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] _assemble: {} ops → {} ra_ops, frame_depth={}",
                ops.len(),
                ra_ops.len(),
                self.frame_depth
            );
        }

        // ── Emit code from regalloc decisions ──
        for ra_op in &ra_ops {
            match ra_op {
                RegAllocOp::Skip => {
                    // Dead operation — skip.
                    continue;
                }
                RegAllocOp::Move { src, dst } => {
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!("[dynasm] move: {:?} → {:?}", src, dst);
                    }
                    self.regalloc_mov(src, dst);
                    continue;
                }
                RegAllocOp::Perform {
                    op_index,
                    arglocs,
                    result_loc,
                    gcmap,
                } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
                        eprintln!(
                            "[dynasm] emit[{}]: {:?} args=[{}] result={:?}",
                            op_index,
                            op.opcode,
                            al.join(", "),
                            result_loc
                        );
                    }
                    self.pending_malloc_nursery_gcmap = *gcmap;
                    self.regalloc_perform(
                        op,
                        *op_index,
                        arglocs,
                        result_loc.as_ref(),
                        fail_index,
                        ops,
                    );
                    self.pending_malloc_nursery_gcmap = None;
                    if !op.pos.is_none() {
                        self.value_types.insert(op.pos.0, op.result_type());
                    }
                }
                RegAllocOp::PerformGuard {
                    op_index,
                    arglocs,
                    result_loc,
                    faillocs,
                } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!(
                            "[dynasm] guard[{}]: {:?} args=[{}] faillocs={}",
                            op_index,
                            op.opcode,
                            arglocs
                                .iter()
                                .map(|l| format!("{:?}", l))
                                .collect::<Vec<_>>()
                                .join(", "),
                            faillocs.len()
                        );
                    }
                    self.regalloc_perform_guard(
                        op,
                        *op_index,
                        arglocs,
                        result_loc.as_ref(),
                        faillocs,
                        fail_index,
                    );
                    fail_index += 1;
                    if !op.pos.is_none() {
                        self.value_types.insert(op.pos.0, op.result_type());
                    }
                }
                RegAllocOp::PerformDiscard { op_index, arglocs } => {
                    let op = &ops[*op_index];
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        let al: Vec<String> = arglocs.iter().map(|l| format!("{:?}", l)).collect();
                        eprintln!(
                            "[dynasm] discard[{}]: {:?} args=[{}]",
                            op_index,
                            op.opcode,
                            al.join(", ")
                        );
                    }
                    self.regalloc_perform(op, *op_index, arglocs, None, fail_index, ops);
                    if op.opcode.is_guard() || op.opcode == OpCode::Finish {
                        fail_index += 1;
                    }
                }
            }
        }

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                fail_index
            );
        }

        Ok(())
    }

    /// assembler.py:326 regalloc_perform — emit code for a non-guard op.
    /// Called from the regalloc dispatch loop with pre-computed locations.
    fn regalloc_perform(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        fail_index: u32,
        ops: &[Op],
    ) {
        match op.opcode {
            OpCode::IntAddOvf => {
                // RPython aarch64/opassembler.py int_add_impl parity —
                // 3-operand `ADDS Rd, Rn, Rm` (s=1 sets flags for GUARD_NO_OVERFLOW).
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                    dynasm!(self.mc ; .arch aarch64
                        ; adds X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntSubOvf => {
                // aarch64/opassembler.py int_sub_impl: `SUBS Rd, Rn, Rm`.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                    dynasm!(self.mc ; .arch aarch64
                        ; subs X(dst.value), X(lhs_reg as u8), X(src_reg as u8));
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntMulOvf => {
                // aarch64/opassembler.py emit_comp_op_int_mul_ovf: smulh+mul+asr+cmp
                // against the 64-bit sign-extended high half.
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let (lhs_reg, src_reg) = self.load_3op_into_scratch(lhs, src);
                    dynasm!(self.mc ; .arch aarch64
                        ; smulh x15, X(lhs_reg as u8), X(src_reg as u8)
                        ; mul X(dst.value), X(lhs_reg as u8), X(src_reg as u8)
                        ; asr x14, X(dst.value), 63
                        ; cmp x15, x14
                    );
                    self.guard_success_cc = Some(CC_E);
                }
            }
            // ── Integer binary (aarch64 3-operand form) ──
            // RPython parity: aarch64/opassembler.py `int_sub_impl` emits
            // `SUB res, l0, l1` with three distinct locations — NOT the
            // x86 `SUB dst, dst, src` 2-operand pattern. The regalloc may
            // pick `res != arglocs[0]` (see `consider_int_sub` LEA path),
            // so we must read arglocs[0] as the true LHS.
            OpCode::IntAdd
            | OpCode::IntSub
            | OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    self.emit_binop_3op(op.opcode, dst.value, lhs, src);
                }
            }
            // ── Unary integer ──
            OpCode::IntNeg => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = match src {
                        Loc::Reg(r) => r.value,
                        Loc::Frame(f) => {
                            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                            17
                        }
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            17
                        }
                        _ => 17,
                    };
                    dynasm!(self.mc ; .arch aarch64 ; neg X(dst.value), X(src_reg as u8));
                }
            }
            OpCode::IntInvert => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = match src {
                        Loc::Reg(r) => r.value,
                        Loc::Frame(f) => {
                            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                            17
                        }
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            17
                        }
                        _ => 17,
                    };
                    dynasm!(self.mc ; .arch aarch64 ; mvn X(dst.value), X(src_reg as u8));
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(lhs), Some(shift_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64: shifts are true 3-operand forms (`lsl/asr/lsr
                    // Rd, Rn, Rm`). The regalloc may pick `result_loc !=
                    // arglocs[0]`, so we must read the real LHS from arg0,
                    // not from the destination register.
                    let lhs_reg = match lhs {
                        Loc::Reg(r) => r.value,
                        Loc::Frame(f) => {
                            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                            17
                        }
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            17
                        }
                        _ => 17,
                    };
                    let sr = match shift_loc {
                        Loc::Reg(s) => s.value,
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(16, i.value);
                            16
                        }
                        Loc::Frame(f) => {
                            dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x29, f.ebp_loc.value as u32]);
                            16
                        }
                        _ => 16,
                    };
                    match op.opcode {
                        OpCode::IntLshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsl X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::IntRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; asr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        OpCode::UintRshift => {
                            dynasm!(self.mc ; .arch aarch64 ; lsr X(dst.value), X(lhs_reg as u8), X(sr as u8));
                        }
                        _ => {}
                    }
                }
            }
            // ── Integer comparisons ──
            OpCode::IntLt
            | OpCode::IntLe
            | OpCode::IntGt
            | OpCode::IntGe
            | OpCode::IntEq
            | OpCode::IntNe
            | OpCode::UintLt
            | OpCode::UintLe
            | OpCode::UintGt
            | OpCode::UintGe
            | OpCode::PtrEq
            | OpCode::PtrNe
            | OpCode::InstancePtrEq
            | OpCode::InstancePtrNe => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                }
                if let Some(Loc::Reg(r)) = result_loc {
                    let cc = Self::opcode_to_cc(op.opcode);
                    self.emit_setcc(cc, r.value);
                }
            }
            OpCode::IntIsTrue => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_NE, r.value);
                }
            }
            OpCode::IntIsZero => {
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.emit_test_loc(src);
                    self.emit_setcc(CC_E, r.value);
                }
            }
            OpCode::UintMulHigh => {
                if let Some(Loc::Reg(dst)) = result_loc {
                    if arglocs.len() >= 2 {
                        let lhs = match arglocs[0] {
                            Loc::Reg(r) => r.value,
                            _ => {
                                self.regalloc_mov(
                                    &arglocs[0],
                                    &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                                );
                                16
                            }
                        };
                        let rhs = match arglocs[1] {
                            Loc::Reg(r) => r.value,
                            _ => {
                                self.regalloc_mov(
                                    &arglocs[1],
                                    &Loc::Reg(crate::regloc::RegLoc::new(17, false)),
                                );
                                17
                            }
                        };
                        dynasm!(self.mc ; .arch aarch64 ; umulh X(dst.value), X(lhs), X(rhs));
                    }
                }
            }
            OpCode::IntForceGeZero => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let src_reg = match src {
                        Loc::Reg(r) => r.value,
                        Loc::Frame(f) => {
                            dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                            17
                        }
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            17
                        }
                        _ => 17,
                    };
                    dynasm!(self.mc ; .arch aarch64
                        ; cmp X(src_reg as u8), xzr
                        ; csel X(dst.value), X(src_reg as u8), xzr, ge
                    );
                }
            }
            OpCode::IntSignext => {
                // arglocs = [argloc, numbytes_loc], result_loc = separate reg
                if let (Some(src), Some(Loc::Reg(r))) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, &Loc::Reg(*r));
                    // signext handled by assembler based on numbytes
                }
            }
            // ── Float binary ──
            OpCode::FloatAdd | OpCode::FloatSub | OpCode::FloatMul | OpCode::FloatTrueDiv => {
                if let (Some(Loc::Reg(dst)), Some(lhs_loc), Some(rhs_loc)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    // aarch64 float ops are true 3-operand forms
                    // (`fadd/fsub/fmul/fdiv Dd, Dn, Dm`). The regalloc may
                    // choose `result_loc != arglocs[0]`, so we must read the
                    // actual lhs from arg0 rather than assuming it already
                    // lives in the destination register.
                    let lhs_reg = match lhs_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(lhs_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    let src_reg = if let Loc::Reg(s) = rhs_loc {
                        *s
                    } else {
                        // Immed or Frame — load to scratch XMM (d14/xmm14)
                        let scratch = crate::regloc::RegLoc::new(14, true);
                        self.regalloc_mov(rhs_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch aarch64 ; fadd D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch aarch64 ; fsub D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch aarch64 ; fmul D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch aarch64 ; fdiv D(dst.value), D(lhs_reg.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg | OpCode::FloatAbs => {
                if let (Some(Loc::Reg(dst)), Some(src_loc)) = (result_loc, arglocs.first()) {
                    // aarch64 fneg/fabs are 2-operand `fneg/fabs Dd, Dn`.
                    // When regalloc chooses `dst != arg0`, reading `Dd` as
                    // the source consumes a stale destination register
                    // instead of the real input, so we must read from
                    // `arglocs[0]` like the float-binop path above.
                    let src_reg = match src_loc {
                        Loc::Reg(r) => *r,
                        _ => {
                            let scratch = crate::regloc::RegLoc::new(15, true);
                            self.regalloc_mov(src_loc, &Loc::Reg(scratch));
                            scratch
                        }
                    };
                    match op.opcode {
                        OpCode::FloatNeg => {
                            dynasm!(self.mc ; .arch aarch64 ; fneg D(dst.value), D(src_reg.value));
                        }
                        OpCode::FloatAbs => {
                            dynasm!(self.mc ; .arch aarch64 ; fabs D(dst.value), D(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            // ── Float comparisons ──
            OpCode::FloatLt
            | OpCode::FloatLe
            | OpCode::FloatEq
            | OpCode::FloatNe
            | OpCode::FloatGt
            | OpCode::FloatGe => {
                if let (Some(Loc::Reg(a)), Some(Loc::Reg(b))) = (arglocs.first(), arglocs.get(1)) {
                    dynasm!(self.mc ; .arch aarch64 ; fcmp D(a.value), D(b.value));
                    if let Some(Loc::Reg(r)) = result_loc {
                        let cc = Self::float_opcode_to_cc(op.opcode);
                        self.emit_setcc(cc, r.value);
                    }
                }
            }
            // ── Casts ──
            OpCode::CastIntToFloat => {
                if let (Some(src), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let sr = match src {
                        Loc::Reg(s) => s.value,
                        _ => {
                            self.regalloc_mov(
                                src,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    dynasm!(self.mc ; .arch aarch64 ; scvtf D(dst.value), X(sr));
                }
            }
            OpCode::CastFloatToInt => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    dynasm!(self.mc ; .arch aarch64 ; fcvtzs X(dst.value), D(src.value));
                }
            }
            // ── Same-as / identity ──
            OpCode::SameAsI
            | OpCode::SameAsR
            | OpCode::SameAsF
            | OpCode::CastPtrToInt
            | OpCode::CastIntToPtr
            | OpCode::CastOpaquePtr
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // ── Memory loads: getfield pattern ──
            OpCode::GetfieldGcI
            | OpCode::GetfieldGcR
            | OpCode::GetfieldGcF
            | OpCode::GetfieldGcPureI
            | OpCode::GetfieldGcPureR
            | OpCode::GetfieldGcPureF
            | OpCode::GetfieldRawI
            | OpCode::GetfieldRawR
            | OpCode::GetfieldRawF
            | OpCode::ArraylenGc
            | OpCode::Strlen
            | OpCode::Unicodelen => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    let ofs = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.offset() as i32)
                        .unwrap_or(0);
                    let field_size = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.field_size())
                        .unwrap_or(8);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), ofs as u32]);
                    } else {
                        match field_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [X(base.value), ofs as u32]);
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [X(base.value), ofs as u32]);
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [X(base.value), ofs as u32]);
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), ofs as u32]);
                            }
                        }
                    }
                }
            }
            // ── Memory loads: getarrayitem pattern ──
            OpCode::GetarrayitemGcI
            | OpCode::GetarrayitemGcR
            | OpCode::GetarrayitemGcF
            | OpCode::GetarrayitemGcPureI
            | OpCode::GetarrayitemGcPureR
            | OpCode::GetarrayitemGcPureF
            | OpCode::GetarrayitemRawI
            | OpCode::GetarrayitemRawR
            | OpCode::GetarrayitemRawF => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(Loc::Reg(dst))) =
                    (arglocs.first(), arglocs.get(1), result_loc)
                {
                    let (base_size, item_size, signed) = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_array_descr())
                        .map(|ad| {
                            (
                                ad.base_size() as i32,
                                ad.item_size() as i32,
                                op.opcode.result_type() == Type::Int && ad.is_item_signed(),
                            )
                        })
                        .unwrap_or((0, 8, false));
                    let index_reg = match index_loc {
                        Loc::Reg(r) => r.value,
                        _ => {
                            self.regalloc_mov(
                                index_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(16, false)),
                            );
                            16
                        }
                    };
                    if item_size != 1 {
                        self.emit_mov_imm64(17, item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul x16, X(index_reg), x17);
                    } else if index_reg != 16 {
                        dynasm!(self.mc ; .arch aarch64 ; mov x16, X(index_reg));
                    }
                    if base_size != 0 {
                        dynasm!(self.mc ; .arch aarch64 ; add x16, x16, base_size as u32);
                    }
                    dynasm!(self.mc ; .arch aarch64 ; add x16, X(base.value), x16);
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [x16]);
                    } else {
                        match item_size {
                            1 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsb X(dst.value), [x16])
                            }
                            1 => dynasm!(self.mc ; .arch aarch64 ; ldrb W(dst.value), [x16]),
                            2 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsh X(dst.value), [x16])
                            }
                            2 => dynasm!(self.mc ; .arch aarch64 ; ldrh W(dst.value), [x16]),
                            4 if signed => {
                                dynasm!(self.mc ; .arch aarch64 ; ldrsw X(dst.value), [x16])
                            }
                            4 => dynasm!(self.mc ; .arch aarch64 ; ldr W(dst.value), [x16]),
                            _ => dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]),
                        }
                    }
                }
            }
            // ── Memory stores: setarrayitem pattern ──
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let (base_size, item_size) = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_array_descr())
                        .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
                        .unwrap_or((0, 8));
                    let index_reg = 16u8;
                    let value_reg = 17u8;

                    self.regalloc_mov(
                        index_loc,
                        &Loc::Reg(crate::regloc::RegLoc::new(index_reg, false)),
                    );
                    if item_size != 1 {
                        self.emit_mov_imm64(value_reg.into(), item_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; mul X(index_reg), X(index_reg), X(value_reg));
                    }
                    if base_size != 0 {
                        self.emit_mov_imm64(value_reg.into(), base_size as i64);
                        dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(index_reg), X(value_reg));
                    }
                    dynasm!(self.mc ; .arch aarch64 ; add X(index_reg), X(base.value), X(index_reg));

                    match value_loc {
                        Loc::Reg(val) if val.is_xmm => {
                            dynasm!(self.mc ; .arch aarch64 ; str D(val.value), [X(index_reg)]);
                        }
                        Loc::Reg(val) => match item_size {
                            1 => {
                                dynasm!(self.mc ; .arch aarch64 ; strb W(val.value), [X(index_reg)])
                            }
                            2 => {
                                dynasm!(self.mc ; .arch aarch64 ; strh W(val.value), [X(index_reg)])
                            }
                            4 => {
                                dynasm!(self.mc ; .arch aarch64 ; str W(val.value), [X(index_reg)])
                            }
                            _ => {
                                dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(index_reg)])
                            }
                        },
                        _ => {
                            self.regalloc_mov(
                                value_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(value_reg, false)),
                            );
                            match item_size {
                                1 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strb W(value_reg), [X(index_reg)])
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch aarch64 ; strh W(value_reg), [X(index_reg)])
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch aarch64 ; str W(value_reg), [X(index_reg)])
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch aarch64 ; str X(value_reg), [X(index_reg)])
                                }
                            }
                        }
                    }
                }
            }
            // ── Memory stores: opassembler.rs emit_op_setfield_regalloc ──
            OpCode::SetfieldGc | OpCode::SetfieldRaw => {
                if let (Some(Loc::Reg(base)), Some(val_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let ofs = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.offset() as i32)
                        .unwrap_or(0);
                    let field_size = op
                        .descr
                        .as_ref()
                        .and_then(|d| d.as_field_descr())
                        .map(|fd| fd.field_size())
                        .unwrap_or(8);
                    self.emit_op_setfield_regalloc(base, val_loc, ofs, field_size);
                } else {
                    self.genop_discard_setfield(op);
                }
            }
            // ── aarch64/opassembler.py:370 _emit_op_gc_load ──
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)]
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let (Some(Loc::Reg(base)), Some(ofs_loc)) = (arglocs.first(), arglocs.get(1)) {
                    // aarch64/opassembler.py:371-374
                    let dst = match arglocs.get(2) {
                        Some(Loc::Reg(r)) => r,
                        _ => match result_loc {
                            Some(Loc::Reg(r)) => r,
                            _ => return,
                        },
                    };
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => op
                            .descr
                            .as_ref()
                            .and_then(|d| d.as_array_descr())
                            .map(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    self.emit_op_gcload_regalloc(base, ofs_loc, dst, nsize);
                }
            }
            // ── aarch64/opassembler.py:365 emit_op_gc_store ──
            // `value_loc, base_loc, ofs_loc, size_loc = arglocs`.
            // regalloc.rs:3042 `consider_gc_store` (mirroring RPython
            // aarch64/regalloc.py:520 prepare_op_gc_store) always emits a
            // 4-tuple `[Reg|Immed(value), Reg(base), Reg|Immed(ofs),
            // Immed(size)]`; value_loc may be Immed when the source is a
            // Const (llsupport/regalloc.py:625 `return_constant`), and
            // the emitter moves it to x16 before the store. Everything
            // else is tight and panics now so regalloc regressions can't
            // hide behind silent defaults.
            OpCode::GcStore | OpCode::RawStore => {
                let (value_loc, base_loc, ofs_loc, size_loc) = match arglocs {
                    [v, b, o, s] => (v, b, o, s),
                    _ => panic!(
                        "GcStore arglocs must be [value, base, ofs, size] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let val_reg = match value_loc {
                    Loc::Reg(r) => *r,
                    Loc::Immed(i) => {
                        self.emit_mov_imm64(16, i.value);
                        crate::regloc::RegLoc::new(16, false)
                    }
                    other => {
                        panic!("GcStore value_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                    }
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcStore base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                // aarch64/opassembler.py:367: scale = get_scale(size_loc.value)
                let size = match size_loc {
                    Loc::Immed(i) => i.value.unsigned_abs() as usize,
                    other => panic!(
                        "GcStore size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                // aarch64/opassembler.py:368: self._write_to_mem(value_loc, base_loc, ofs_loc, scale)
                self.emit_op_gcstore_regalloc(base, ofs_loc, &val_reg, size);
            }
            // ── aarch64/opassembler.py:396-412 _emit_op_gc_load_indexed ──
            // arglocs = [res_loc, base_loc, index_loc, imm(nsize), imm(ofs)]
            // per `consider_gc_load_indexed` in regalloc.rs:2981-3014.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                if let (Some(Loc::Reg(res)), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let nsize = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => op
                            .descr
                            .as_ref()
                            .and_then(|d| d.as_array_descr())
                            .map(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    // aarch64/opassembler.py:402 keeps `ofs_loc.value` at
                    // the Signed word width; `_try_use_older_box` (rewrite.
                    // py:383-387) may fold large displacements into the
                    // offset immediate so the emitter handles the full
                    // range via the check_imm_arg / materialize fallback.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => 0,
                    };
                    self.emit_op_gcload_indexed_regalloc(base, index, res, ofs, nsize);
                }
            }
            // ── aarch64/opassembler.py:381 emit_op_gc_store_indexed ──
            // arglocs = [value_loc, base_loc, index_loc, imm(size), imm(ofs)]
            OpCode::GcStoreIndexed => {
                if let (Some(value_loc), Some(Loc::Reg(base)), Some(Loc::Reg(index))) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let size = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value.unsigned_abs() as usize,
                        _ => 8,
                    };
                    // aarch64/opassembler.py:385-392 keeps `ofs_loc.value`
                    // at the Signed word width; large displacements folded
                    // in via `_try_use_older_box` must not be truncated.
                    let ofs = match arglocs.get(4) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => 0,
                    };
                    // aarch64/opassembler.py:385-392: combine ofs into ip0 = index + ofs.
                    // check_imm_arg() in regalloc.py:161 allows 0..4095; for any
                    // value outside that (including negatives) RPython falls back
                    // to load(ip0, ofs_loc) + ADD_rr. ip0 = x16 (reserved scratch).
                    let combined_index = if ofs != 0 {
                        if (0..4096).contains(&ofs) {
                            dynasm!(self.mc ; .arch aarch64
                                ; mov x16, X(index.value)
                                ; add x16, x16, ofs as u32);
                        } else {
                            self.emit_mov_imm64(16, ofs);
                            dynasm!(self.mc ; .arch aarch64
                                ; add x16, x16, X(index.value));
                        }
                        crate::regloc::RegLoc::new(16, false)
                    } else {
                        *index
                    };
                    let val_reg = match value_loc {
                        Loc::Reg(r) => *r,
                        Loc::Immed(i) => {
                            self.emit_mov_imm64(17, i.value);
                            crate::regloc::RegLoc::new(17, false)
                        }
                        _ => crate::regloc::RegLoc::new(17, false),
                    };
                    // aarch64/opassembler.py:393-394:
                    //   scale = get_scale(size_loc.value)
                    //   self._write_to_mem(value_loc, base_loc, index_loc, scale)
                    self.emit_op_gcstore_regalloc(base, &Loc::Reg(combined_index), &val_reg, size);
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                let jump_descr = loop_target_descr(op);
                let target_arglocs = jump_descr
                    .map(|descr| {
                        descr
                            .target_arglocs()
                            .into_iter()
                            .map(|loc| loc_from_target_argloc(&loc))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[dynasm] JUMP remap: src={:?} dst={:?}",
                        arglocs, target_arglocs
                    );
                }
                let mut src_locations1 = Vec::new();
                let mut dst_locations1 = Vec::new();
                let mut src_locations2 = Vec::new();
                let mut dst_locations2 = Vec::new();
                // x86/regalloc.py:1287: assert len(arglocs) == jump_op.numargs()
                let remap_count = if target_arglocs.is_empty() {
                    arglocs.len()
                } else {
                    assert_eq!(
                        arglocs.len(),
                        target_arglocs.len(),
                        "JUMP args ({}) must match target LABEL args ({})",
                        arglocs.len(),
                        target_arglocs.len(),
                    );
                    target_arglocs.len()
                };
                for (i, src_loc) in arglocs[..remap_count].iter().enumerate() {
                    let dst_loc = if i < target_arglocs.len() {
                        target_arglocs[i]
                    } else {
                        let dst_ofs = crate::regalloc::get_ebp_ofs(0, i);
                        Loc::Frame(crate::regloc::FrameLoc::new(i, dst_ofs, false))
                    };
                    let arg_tp = op
                        .args
                        .get(i)
                        .and_then(|arg| self.value_types.get(&arg.0))
                        .copied()
                        .unwrap_or(Type::Int);
                    if arg_tp == Type::Float {
                        src_locations2.push(*src_loc);
                        dst_locations2.push(dst_loc);
                    } else {
                        src_locations1.push(*src_loc);
                        dst_locations1.push(dst_loc);
                    }
                }
                let tmpreg1 = Loc::Reg(crate::regloc::RegLoc::new(16, false));
                let tmpreg2 = Loc::Reg(crate::regloc::RegLoc::new(15, true));
                self.remap_frame_layout_mixed(
                    &src_locations1,
                    &dst_locations1,
                    tmpreg1,
                    &src_locations2,
                    &dst_locations2,
                    tmpreg2,
                );
                // assembler.py:2456-2462 closing_jump
                if let Some(label) = loop_target_id(op)
                    .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
                {
                    dynasm!(self.mc ; .arch aarch64 ; b =>label);
                } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
                    // External JUMP: direct branch to target loop code.
                    // assembler.py:2461 mc.JMP(imm(target))
                    // Use x16 (IP0 scratch) to avoid clobbering remap'd regs.
                    self.emit_mov_imm64(16, target as i64);
                    dynasm!(self.mc ; .arch aarch64 ; br x16);
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op);
                let result_type = if fail_arg_types.is_empty() {
                    Type::Void
                } else {
                    fail_arg_types[0]
                };
                // `compile.py:658` ExitFrameWithExceptionDescrRef identity:
                // route to the metainterp `exit_frame_with_exception_descr_ref`
                // when the FINISH was emitted for
                // `pyjitpl.py:3238 compile_exit_frame_with_exception`.
                let is_exit_exc = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_fail_descr())
                    .map(|fd| fd.is_exit_frame_with_exception())
                    .unwrap_or(false);
                let global_descr_ptr = if is_exit_exc {
                    self.exit_frame_with_exception_descr_ref_ptr()
                } else {
                    self.done_with_this_frame_descr_ptr_for_type(result_type)
                };
                let descr = Arc::new(DynasmFailDescr::new(
                    fail_index,
                    self.trace_id,
                    fail_arg_types.clone(),
                    true,
                ));

                // Store result to jf_frame[0]
                if let Some(result) = arglocs.first() {
                    let slot0 = Loc::Frame(crate::regloc::FrameLoc::new(
                        0,
                        crate::jitframe::FIRST_ITEM_OFFSET as i32,
                        result_type == Type::Float,
                    ));
                    self.regalloc_mov(result, &slot0);
                }

                // Store descr ptr to jf_descr.
                self.emit_mov_imm64(0, global_descr_ptr);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);

                if result_type == Type::Ref {
                    if let Some(gcmap) = self.finish_gcmap {
                        gcmap_set_bit(gcmap, 0);
                        self.push_gcmap(gcmap);
                    } else {
                        self.push_gcmap(self.gcmap_for_finish);
                    }
                } else if let Some(gcmap) = self.finish_gcmap {
                    self.push_gcmap(gcmap);
                } else {
                    self.pop_gcmap();
                }

                self._call_footer();
                self.fail_descrs.push(descr);
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let label_descr = loop_target_descr(op);
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[dynasm] LABEL: new DynamicLabel({:?}) arglocs={:?}",
                        label, arglocs
                    );
                }
                dynasm!(self.mc ; =>label);
                if let Some(descr) = label_descr {
                    let stored_arglocs = arglocs
                        .iter()
                        .copied()
                        .map(target_argloc_from_loc)
                        .collect::<Vec<_>>();
                    if std::env::var_os("MAJIT_LOG").is_some() {
                        eprintln!("[dynasm] LABEL target_arglocs={:?}", stored_arglocs);
                    }
                    descr.set_target_arglocs(stored_arglocs);
                    descr.set_ll_loop_code(self.mc.offset().0);
                    if let Some(id) = loop_target_id(op) {
                        self.target_tokens_currently_compiling.insert(id, label);
                    }
                    if let Some(descr_ref) = op.descr.as_ref() {
                        self.compiled_target_tokens.push(descr_ref.clone());
                    }
                }
            }
            // ── Calls ──
            // RPython: regalloc consider_call does before_call (save caller-saved
            // regs), collects arglocs, calls after_call for result. The assembler
            // receives arglocs = [func_addr_or_descr_info..., arg_locs...] and
            // result_loc = register for return value.
            //
            // For now, flush all register-resident values to their frame slots
            // before the call, use the existing frame-slot genop_call, then
            // mark the result in the allocated register.
            OpCode::CallI
            | OpCode::CallR
            | OpCode::CallF
            | OpCode::CallN
            | OpCode::CallPureI
            | OpCode::CallPureR
            | OpCode::CallPureF
            | OpCode::CallPureN
            | OpCode::CallLoopinvariantI
            | OpCode::CallLoopinvariantR
            | OpCode::CallLoopinvariantF
            | OpCode::CallLoopinvariantN
            | OpCode::CallMayForceI
            | OpCode::CallMayForceR
            | OpCode::CallMayForceF
            | OpCode::CallMayForceN
            | OpCode::CallReleaseGilI
            | OpCode::CallReleaseGilR
            | OpCode::CallReleaseGilF
            | OpCode::CallReleaseGilN => {
                self.genop_call_with_arglocs(op, arglocs);
            }
            OpCode::CallAssemblerI
            | OpCode::CallAssemblerR
            | OpCode::CallAssemblerF
            | OpCode::CallAssemblerN => {
                // assembler.py:2207 _store_force_index parity:
                // store next GUARD_NOT_FORCED's descr ptr to jf_force_descr
                // BEFORE the call, so forcing code knows which guard to resume.
                self._store_force_index_if_next_guard(ops, op_index, fail_index);
                self.genop_call_assembler(op, arglocs);
            }
            OpCode::CondCallN => self.genop_discard_cond_call(op),
            OpCode::CondCallValueI | OpCode::CondCallValueR => {
                self.genop_cond_call_value(op);
            }
            // ── Allocation (raw, when GC rewriter is not active) ──
            OpCode::New => self.genop_new(op),
            OpCode::NewWithVtable => self.genop_new_with_vtable(op),
            OpCode::NewArray | OpCode::NewArrayClear => self.genop_new_array(op),
            OpCode::Newstr => self.genop_newstr(op),
            OpCode::Newunicode => self.genop_newunicode(op),
            // ── Allocation (rewritten by GC rewriter) ──
            // aarch64/regalloc.py:958 + assembler.py:682 malloc_cond parity
            OpCode::CallMallocNursery => {
                self.genop_call_malloc_nursery(op);
                if let Some(Loc::Reg(r)) = result_loc {
                    if r.value != 0 {
                        let rv = r.value;
                        dynasm!(self.mc ; .arch aarch64 ; mov X(rv), x0);
                    }
                }
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
            }
            // aarch64/assembler.py:715 malloc_cond_varsize_frame
            OpCode::CallMallocNurseryVarsizeFrame => {
                if let Some(Loc::Reg(sizeloc)) = arglocs.first() {
                    dynasm!(self.mc ; .arch aarch64 ; mov x0, X(sizeloc.value));
                }
                // push_gcmap
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                self.emit_mov_imm64(
                    2,
                    crate::runner::dynasm_nursery_slowpath as *const () as i64,
                );
                self.emit_malloc_slowpath_helper_call(2);
                // pop_gcmap
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
            }
            // aarch64/assembler.py:738 malloc_cond_varsize
            // arglocs = [lengthloc, imm(itemsize), imm(kind)]
            OpCode::CallMallocNurseryVarsize => {
                let base_size = op
                    .descr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map(|ad| ad.base_size())
                    .unwrap_or(16) as i64;
                let itemsize = match arglocs.get(1) {
                    Some(Loc::Immed(i)) => i.value,
                    _ => 8,
                };
                // _build_malloc_slowpath(kind='var') parity:
                // x0 = base_size, x1 = item_size, x2 = length
                self.emit_mov_imm64(0, base_size);
                self.emit_mov_imm64(1, itemsize);
                match arglocs.first() {
                    Some(Loc::Reg(len_r)) => {
                        dynasm!(self.mc ; .arch aarch64 ; mov x2, X(len_r.value));
                    }
                    Some(Loc::Immed(len_i)) => {
                        self.emit_mov_imm64(2, len_i.value);
                    }
                    _ => {
                        self.emit_mov_imm64(2, 0);
                    }
                }
                // push_gcmap
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                self.emit_mov_imm64(
                    3,
                    crate::runner::dynasm_nursery_slowpath_varsize as *const () as i64,
                );
                self.emit_malloc_slowpath_helper_call(3);
                // pop_gcmap
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
            }
            // aarch64/opassembler.py:258 `emit_op_check_memory_error` →
            // assembler.py:342-346 `propagate_memoryerror_if_reg_is_null`
            // — emit `CBNZ reg, skip` and inline the propagate path
            // (`_build_propagate_exception_path`, assembler.py:559-577)
            // so a NULL return from a malloc helper raises a MemoryError
            // via `self.cpu.propagate_exception_descr`.
            //
            // Upstream materializes `propagate_exception_path` once per
            // backend instance and per-CHECK_MEMORY_ERROR branches to
            // it.  Pyre's dynasm doesn't have that out-of-line trampoline
            // infrastructure yet, so the path is inlined per occurrence
            // — equivalent semantics, slightly more code per site.
            // CHECK_MEMORY_ERROR is rare (only after the four CALL_R
            // malloc helpers in `gen_call_malloc_gc`), so the size
            // overhead is negligible.
            //
            // Sequence per assembler.py:559-577:
            //   1. _store_and_reset_exception(mc, r.x0)
            //      — load pos_exc_value → x0, clear pos_exc_value and
            //      pos_exception (assembler.py:501-536).
            //   2. str x0, [fp, jf_guard_exc]
            //      — transfer the saved value into the deadframe so
            //      `cpu.grab_exc_value(deadframe)` can read it back in
            //      `PropagateExceptionDescr.handle_fail` (compile.py:1095).
            //   3. gen_load_int x0, propagate_exception_descr
            //   4. str x0, [fp, jf_descr]
            //   5. mov x0, fp + gen_func_epilog
            OpCode::CheckMemoryError => {
                let propagate_descr = self.propagate_exception_descr_ptr();
                if propagate_descr == 0 {
                    // Unattached `propagate_exception_descr` — typical
                    // for unit tests that bypass `MetaInterp::finish_setup`.
                    // Skip the null-check; in production
                    // (`pyjitpl.py:2283 self.cpu.propagate_exception_descr
                    // = exc_descr`) the descr is always set before
                    // `compile_loop` runs.
                } else {
                    let reg = match arglocs.first() {
                        Some(Loc::Reg(r)) if !r.is_xmm => r.value,
                        _ => panic!("CheckMemoryError arglocs[0] must be a non-xmm register"),
                    };
                    let skip = self.mc.new_dynamic_label();
                    let exc_value_addr = crate::jit_exc_value_addr() as i64;
                    let exc_type_addr = crate::jit_exc_type_addr() as i64;
                    // x0 is freely clobberable on the propagate path:
                    // _call_footer (assembler.py:574) overwrites x0 with fp
                    // before returning, so any live value in x0 was already
                    // dead the moment we branched into the path.
                    dynasm!(self.mc ; .arch aarch64
                        ; cbnz X(reg), =>skip
                    );
                    // assembler.py:509-512 — load pos_exc_value into x0.
                    self.emit_mov_imm64(16, exc_value_addr);
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x0, [x16]
                        // assembler.py:531-533 — clear pos_exc_value.
                        ; str xzr, [x16]
                    );
                    // assembler.py:535-536 — clear pos_exception.
                    self.emit_mov_imm64(16, exc_type_addr);
                    dynasm!(self.mc ; .arch aarch64
                        ; str xzr, [x16]
                        // assembler.py:565 — store x0 → jf_guard_exc.
                        ; str x0, [x29, JF_GUARD_EXC_OFS as u32]
                    );
                    // assembler.py:572-573 — store propagate_descr → jf_descr.
                    self.emit_mov_imm64(0, propagate_descr);
                    dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, JF_DESCR_OFS as u32]);
                    self._call_footer();
                    dynasm!(self.mc ; .arch aarch64 ; =>skip);
                }
            }
            // aarch64/opassembler.py:912 _write_barrier_fastpath parity.
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.emit_write_barrier_fastpath(op, &arglocs);
            }
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch aarch64 ; mov X(r.value), x29);
                }
            }
            OpCode::SaveException => self.genop_save_exception(op),
            OpCode::SaveExcClass => self.genop_save_exc_class(op),
            // ── Guards (via regalloc_perform_guard, shouldn't reach here) ──
            _ if op.opcode.is_guard() => {
                // Guards should go through regalloc_perform_guard, not here.
                // Fallback: use existing guard implementation.
                self.implement_guard_nojump(op, fail_index);
            }
            // ── No-ops ──
            _ => {}
        }
    }

    /// assembler.py:329 regalloc_perform_guard — emit guard with faillocs.
    fn regalloc_perform_guard(
        &mut self,
        op: &Op,
        op_index: usize,
        arglocs: &[Loc],
        result_loc: Option<&Loc>,
        faillocs: &[Option<Loc>],
        fail_index: u32,
    ) {
        match op.opcode {
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                // arglocs[0] = condition location
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.emit_test_loc(loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardValue => {
                if arglocs.len() >= 2 {
                    self.emit_cmp_loc_loc(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardClass => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardGcType => {
                if arglocs.len() >= 2 {
                    self._cmp_guard_gc_type(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardIsObject => {
                if let Some(obj_loc) = arglocs.first() {
                    self.emit_guard_is_object(obj_loc);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardSubclass => {
                if arglocs.len() >= 2 {
                    self.emit_guard_subclass(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_B);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardException => {
                if let Some(expected_loc) = arglocs.first() {
                    self.emit_guard_exception(expected_loc);
                    self.guard_success_cc = Some(CC_E);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
                self.emit_store_and_reset_exception(result_loc);
            }
            OpCode::GuardNonnullClass => {
                if arglocs.len() >= 2 {
                    self.emit_test_loc(&arglocs[0]);
                    let fail_label = self.emit_guard_jcc(CC_E);
                    self._cmp_guard_class(&arglocs[0], &arglocs[1]);
                    self.emit_jcc_to_label(CC_NE, fail_label);
                    self.append_guard_token_with_faillocs(
                        op, op_index, fail_index, fail_label, faillocs,
                    );
                }
            }
            OpCode::GuardNoException => {
                self.emit_guard_no_exception_check();
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNoOverflow | OpCode::GuardOverflow => {
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                dynasm!(self.mc ; .arch aarch64 ; ldr X(16), [x29, JF_DESCR_OFS as u32] ; cmp X(16), xzr);
                self.guard_success_cc = Some(CC_E);
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardNotInvalidated => {
                self.implement_guard_nojump_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardAlwaysFails => {
                self.implement_guard_always_fails_with_faillocs(op, op_index, fail_index, faillocs);
            }
            _ => {
                self.implement_guard_nojump_with_faillocs(op, op_index, fail_index, faillocs);
            }
        }
    }

    /// Helper: guard class comparison
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        if let Loc::Reg(obj) = obj_loc {
            if let Some(vtable_offset) = self.vtable_offset {
                let ofs = vtable_offset as u32;
                dynasm!(self.mc ; .arch aarch64 ; ldr x16, [X(obj.value), ofs]);
                self.regalloc_mov(class_loc, &Loc::Reg(crate::regloc::RegLoc::new(17, false)));
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            } else if let Loc::Immed(i) = class_loc {
                let expected_typeid = self
                    .lookup_typeid_from_classptr(i.value as usize)
                    .expect("GuardClass: missing typeid for classptr");
                self._cmp_guard_gc_type(
                    &Loc::Reg(*obj),
                    &Loc::Immed(crate::regloc::ImmedLoc::new(expected_typeid as i64)),
                );
            }
        }
    }

    fn require_guard_gc_type_info(&self, guard_name: &'static str) -> GuardGcTypeInfo {
        self.guard_gc_type_info.unwrap_or_else(|| {
            panic!(
                "{} requires cpu.supports_guard_gc_type and a TYPE_INFO layout",
                guard_name
            )
        })
    }

    fn lookup_subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        self.classptr_to_subclass_range
            .get(&(classptr as i64))
            .copied()
    }

    fn emit_load_gc_typeid_into_x16(&mut self, obj_reg: u8) {
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i64);
        self.emit_mov_imm64(16, tid_ofs);
        dynasm!(self.mc ; .arch aarch64
            ; add x16, X(obj_reg), x16
            ; ldr w16, [x16]
        );
    }

    /// aarch64 analogue of x86/assembler.py:1893-1901
    /// `_cmp_guard_gc_type`, adjusted for majit's `obj - GcHeader::SIZE`
    /// header layout.
    fn _cmp_guard_gc_type(&mut self, obj_loc: &Loc, expected_typeid_loc: &Loc) {
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        match expected_typeid_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                let ofs = frame.ebp_loc.value as u32;
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x17, [x29, ofs]
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            _ => {}
        }
    }

    /// aarch64/opassembler.py:650-663 `emit_op_guard_is_object`.
    fn emit_guard_is_object(&mut self, obj_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_IS_OBJECT");
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        self.emit_load_gc_typeid_into_x16(obj.value);
        if info.shift_by > 0 {
            dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
        }
        self.emit_mov_imm64(17, (info.base_type_info + info.infobits_offset) as i64);
        let flag = info.is_object_flag as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x16, x16, x17
            ; ldrb w16, [x16]
            ; mov w17, flag
            ; tst w16, w17
        );
    }

    /// aarch64/opassembler.py:665-706 `emit_op_guard_subclass`.
    fn emit_guard_subclass(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_SUBCLASS");
        let (Loc::Reg(obj), Loc::Immed(classptr)) = (obj_loc, class_loc) else {
            panic!(
                "GUARD_SUBCLASS expects [Reg object, Immed classptr] \
                 like aarch64/opassembler.py:667"
            );
        };
        let (check_min, check_max) = self
            .lookup_subclass_range(classptr.value as usize)
            .unwrap_or_else(|| {
                panic!(
                    "GUARD_SUBCLASS missing subclassrange_min/max for classptr {:#x}",
                    classptr.value
                )
            });
        if let Some(vtable_offset) = self.vtable_offset {
            let offset = vtable_offset as u32;
            let offset2 = info.subclassrange_min_offset as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldr x16, [X(obj.value), offset]
                ; ldr x16, [x16, offset2]
            );
        } else {
            self.emit_load_gc_typeid_into_x16(obj.value);
            if info.shift_by > 0 {
                dynasm!(self.mc ; .arch aarch64 ; lsl x16, x16, info.shift_by as u32);
            }
            self.emit_mov_imm64(
                17,
                (info.base_type_info + info.sizeof_ti + info.subclassrange_min_offset) as i64,
            );
            dynasm!(self.mc ; .arch aarch64
                ; add x16, x16, x17
                ; ldr x16, [x16]
            );
        }
        self.emit_mov_imm64(17, check_min);
        dynasm!(self.mc ; .arch aarch64 ; sub x16, x16, x17);
        self.emit_mov_imm64(17, check_max - check_min);
        dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
    }

    /// aarch64/opassembler.py:711-718 `emit_op_guard_exception`.
    fn emit_guard_exception(&mut self, expected_loc: &Loc) {
        self.emit_mov_imm64(16, crate::jit_exc_type_addr() as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x16, [x16]);
        match expected_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, X(expected.value));
            }
            Loc::Frame(frame) => {
                let ofs = frame.ebp_loc.value as u32;
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x17, [x29, ofs]
                    ; cmp x16, x17
                );
            }
            Loc::Immed(expected) => {
                self.emit_mov_imm64(17, expected.value);
                dynasm!(self.mc ; .arch aarch64 ; cmp x16, x17);
            }
            _ => {}
        }
    }

    /// `_store_and_reset_exception`: result = pos_exc_value; clear both
    /// pos_exception and pos_exc_value on the success fallthrough.
    fn emit_store_and_reset_exception(&mut self, result_loc: Option<&Loc>) {
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_value_addr);
        if let Some(loc) = result_loc {
            match loc {
                Loc::Reg(dst) => {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [x16]);
                }
                Loc::Frame(frame) => {
                    let ofs = frame.ebp_loc.value as u32;
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x17, [x16]
                        ; str x17, [x29, ofs]
                    );
                }
                _ => {}
            }
        }
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x16]);
    }

    /// aarch64/opassembler.py:720-724 `emit_op_guard_no_exception`.
    fn emit_cmp_no_exception(&mut self) {
        self.emit_mov_imm64(16, crate::jit_exc_value_addr() as i64);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]
            ; cmp x16, xzr
        );
    }

    /// Emit SETcc into a register (zero-extend to 64-bit).
    fn emit_setcc(&mut self, cc: u8, dst_reg: u8) {
        match cc {
            CC_E => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
            CC_NE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ne);
            }
            CC_L => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lt);
            }
            CC_GE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ge);
            }
            CC_LE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), le);
            }
            CC_G => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), gt);
            }
            CC_B => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), lo);
            }
            CC_AE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hs);
            }
            CC_BE => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), ls);
            }
            CC_A => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), hi);
            }
            CC_S => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), mi);
            }
            CC_NS => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), pl);
            }
            CC_O => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vs);
            }
            CC_NO => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), vc);
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; cset X(dst_reg), eq);
            }
        }
    }

    /// Map an integer comparison OpCode to a condition code.
    fn opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::IntLt => CC_L,
            OpCode::IntLe => CC_LE,
            OpCode::IntGt => CC_G,
            OpCode::IntGe => CC_GE,
            OpCode::IntEq | OpCode::PtrEq | OpCode::InstancePtrEq => CC_E,
            OpCode::IntNe | OpCode::PtrNe | OpCode::InstancePtrNe => CC_NE,
            OpCode::UintLt => CC_B,
            OpCode::UintLe => CC_BE,
            OpCode::UintGt => CC_A,
            OpCode::UintGe => CC_AE,
            _ => CC_E,
        }
    }

    /// Map a float comparison OpCode to a condition code (after ucomisd).
    fn float_opcode_to_cc(opcode: OpCode) -> u8 {
        match opcode {
            OpCode::FloatLt => CC_B,  // ucomisd: below = less than
            OpCode::FloatLe => CC_BE, // below or equal
            OpCode::FloatGt => CC_A,  // above
            OpCode::FloatGe => CC_AE, // above or equal
            OpCode::FloatEq => CC_E,  // equal
            OpCode::FloatNe => CC_NE, // not equal
            _ => CC_E,
        }
    }

    /// Guard with faillocs — emit conditional jump and store faillocs on descr.
    fn implement_guard_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard_with_faillocs: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    /// Guard no-jump with faillocs.
    fn implement_guard_nojump_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    fn implement_guard_always_fails_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        faillocs: &[Option<Loc>],
    ) {
        let fail_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
        self.append_guard_token_with_faillocs(op, op_index, fail_index, fail_label, faillocs);
    }

    /// Append guard token with regalloc faillocs instead of opref_to_slot snapshot.
    fn append_guard_token_with_faillocs(
        &mut self,
        op: &Op,
        op_index: usize,
        fail_index: u32,
        fail_label: DynamicLabel,
        faillocs: &[Option<Loc>],
    ) {
        let fail_arg_types = self.infer_fail_arg_types(op);
        // assembler.py:2207 _store_force_index parity:
        // If a CALL_ASSEMBLER already pre-allocated this guard's descr
        // (stored in pending_force_descr), reuse it — same Arc, same ptr
        // that was written to jf_force_descr.
        let descr = if let Some(pre) = self.pending_force_descr.take() {
            pre
        } else {
            Arc::new(DynasmFailDescr::new(
                fail_index,
                self.trace_id,
                fail_arg_types,
                false,
            ))
        };
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] guard-token: fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?} faillocs={:?}",
                fail_index,
                op_index,
                op.opcode,
                op.fail_args.as_ref(),
                &descr.fail_arg_types,
                faillocs
            );
        }

        // Convert regalloc faillocs to absolute jf_frame slots for the
        // helper/runner. This matches the fixed-slot-inclusive coordinate
        // system used by get_fp_offset() and compiled_loop_token._ll_initial_locs
        // in RPython.
        let mut const_stores: Vec<(usize, i64)> = Vec::new();
        let gpr_regs = all_gen_regs();
        let float_regs = all_float_regs();
        let fail_arg_locs: Vec<Option<usize>> = faillocs
            .iter()
            .map(|fl| match fl {
                Some(Loc::Reg(r)) => {
                    if r.is_xmm {
                        float_reg_position(*r)
                    } else {
                        core_reg_position(*r)
                    }
                }
                Some(Loc::Frame(f)) => Some(f.position + JITFRAME_FIXED_SIZE),
                Some(Loc::Immed(i)) => {
                    // Allocate a slot for this constant in the save area.
                    let slot = self.frame_depth;
                    self.frame_depth += 1;
                    const_stores.push((slot, i.value));
                    Some(slot)
                }
                _ => None,
            })
            .collect();
        let rd_locs: Vec<u16> = faillocs
            .iter()
            .map(|fl| match fl {
                None => 0xFFFF,
                Some(Loc::Frame(f)) => (f.position + JITFRAME_FIXED_SIZE) as u16,
                Some(Loc::Reg(r)) if r.is_xmm => {
                    (gpr_regs.len()
                        + float_regs
                            .iter()
                            .position(|reg| *reg == *r)
                            .expect("rd_locs: float register not in float_regs"))
                        as u16
                }
                Some(Loc::Reg(r)) => gpr_regs
                    .iter()
                    .position(|reg| *reg == *r)
                    .expect("rd_locs: register not in gen_regs")
                    as u16,
                Some(Loc::Immed(_)) => 0xFFFF,
                Some(Loc::Ebp(_)) | Some(Loc::Addr(_)) => 0xFFFF,
            })
            .collect();
        // Build identity recovery_layout (Cranelift identity_recovery_layout parity).
        // `jitcode_index: 0` is a placeholder: every guard is later patched
        // via `compile::patch_backend_guard_recovery_layouts_for_trace`
        // (compile.rs:1596) with a resume_layout-derived layout whose
        // jitcode_index originates from `Snapshot::single_frame`.
        let recovery_layout = {
            let slot_types = &descr.fail_arg_types;
            ExitRecoveryLayout {
                vable_array: vec![],
                vref_array: vec![],
                frames: vec![ExitFrameLayout {
                    trace_id: Some(self.trace_id),
                    header_pc: Some(self.header_pc),
                    source_guard: None,
                    pc: self.header_pc,
                    jitcode_index: 0,
                    slots: (0..slot_types.len())
                        .map(ExitValueSourceLayout::ExitValue)
                        .collect(),
                    slot_types: Some(slot_types.clone()),
                }],
                virtual_layouts: vec![],
                pending_field_layouts: vec![],
            }
        };
        unsafe {
            let descr_mut = &mut *(Arc::as_ptr(&descr) as *mut DynasmFailDescr);
            descr_mut.fail_arg_locs = fail_arg_locs;
            descr_mut.rd_locs = rd_locs;
            descr_mut.source_op_index = Some(op_index);
            // resume.py:450-488 parity: read rd_* through op.descr →
            // ResumeGuardDescr (or chase prev for copied descrs).
            let fd = op.descr.as_ref().and_then(|d| d.as_fail_descr());
            descr_mut.rd_numb = fd.and_then(|fd| fd.rd_numb()).map(|s| s.to_vec());
            descr_mut.rd_consts = fd.and_then(|fd| fd.rd_consts()).map(|s| s.to_vec());
            descr_mut.rd_virtuals = fd.and_then(|fd| fd.rd_virtuals()).map(|s| s.to_vec());
            descr_mut.rd_pendingfields =
                fd.and_then(|fd| fd.rd_pendingfields()).map(|s| s.to_vec());
            *descr_mut.recovery_layout.get_mut() = Some(recovery_layout);
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] guard-token-slots: fail_index={} fail_arg_locs={:?} rd_locs={:?}",
                fail_index, &descr.fail_arg_locs, &descr.rd_locs
            );
        }
        let gcmap = self.guard_gcmap_from_faillocs(&descr.fail_arg_types, faillocs);

        self.pending_guard_tokens.push(GuardToken {
            fail_label,
            fail_descr: descr.clone(),
            const_stores,
            gcmap,
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // assembler.py:652 write_pending_failure_recoveries
    // ----------------------------------------------------------------

    /// assembler.py:982 generate_quick_failure.
    ///
    /// RPython parity: the quick-failure stub saves managed registers into the
    /// fixed jitframe prefix before publishing jf_descr and returning.
    fn generate_quick_failure(
        &mut self,
        guard_token: GuardToken,
        save_regs_label: DynamicLabel,
    ) -> (Arc<DynasmFailDescr>, usize) {
        let stub_start = self.mc.offset();

        let fail_label = guard_token.fail_label;
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm] recovery stub: binding {:?}", fail_label);
        }
        dynasm!(self.mc ; .arch aarch64 ; =>fail_label);

        dynasm!(self.mc ; .arch aarch64 ; bl =>save_regs_label);

        let descr_ptr = Arc::as_ptr(&guard_token.fail_descr) as i64;
        self.emit_mov_imm64(0, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_DESCR_OFS as u32]
        );
        self.push_gcmap(guard_token.gcmap);

        for &(slot, val) in &guard_token.const_stores {
            let ofs = Self::slot_offset(slot);
            self.emit_mov_imm64(16, val);
            dynasm!(self.mc ; .arch aarch64 ; str x16, [x29, ofs as u32]);
        }

        self._call_footer();
        (guard_token.fail_descr, stub_start.0)
    }

    /// assembler.py:1005 write_pending_failure_recoveries.
    /// Returns recovery stub offsets for post-finalize address fixup.
    fn write_pending_failure_recoveries(&mut self) -> Vec<(Arc<DynasmFailDescr>, usize)> {
        // Emit a shared _push_all_regs_to_frame routine once, then let each
        // generate_quick_failure() stub call it.
        let save_regs_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; =>save_regs_label);
        let gprs = all_gen_regs();
        for &reg in gprs.iter() {
            let save_slot = core_reg_position(reg).expect("managed aarch64 GPR");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str X(reg.value), [x29, ofs]);
        }
        let fprs = all_float_regs();
        for &reg in fprs.iter() {
            let save_slot = float_reg_position(reg).expect("managed aarch64 VFP");
            let ofs = Self::slot_offset(save_slot) as u32;
            dynasm!(self.mc ; .arch aarch64 ; str D(reg.value), [x29, ofs]);
        }
        dynasm!(self.mc ; .arch aarch64 ; ret);

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            stub_offsets.push(self.generate_quick_failure(guard_token, save_regs_label));
        }
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[dynasm] write_pending done: {} stubs", stub_offsets.len());
        }
        stub_offsets
    }

    /// assembler.py:849 patch_pending_failure_recoveries — convert
    /// buffer-relative offsets to absolute addresses after finalize.
    fn patch_pending_failure_recoveries(
        rawstart: usize,
        stub_offsets: &[(Arc<DynasmFailDescr>, usize)],
    ) {
        for (descr, stub_offset) in stub_offsets {
            let abs_addr = rawstart + stub_offset;
            descr.set_adr_jump_offset(abs_addr);
        }
    }

    // ----------------------------------------------------------------
    // assembler.py:965-987 patch_jump_for_descr
    // ----------------------------------------------------------------

    /// assembler.py:965 patch_jump_for_descr: redirect a guard to a
    /// bridge by overwriting the recovery stub with a JMP to bridge.
    ///
    /// `adr_jump_offset` is the absolute address of the recovery stub
    /// (set by patch_pending_failure_recoveries). We overwrite the
    /// stub with "MOV r11, bridge_addr; JMP r11" (x64) or "BL imm26"
    /// (aarch64), matching rpython/jit/backend/aarch64/assembler.py
    /// patch_trace().
    pub fn patch_jump_for_descr(descr: &DynasmFailDescr, adr_new_target: usize) {
        let stub_addr = descr.adr_jump_offset();
        assert!(stub_addr != 0, "guard already patched");

        codebuf::with_writable(stub_addr as *mut u8, 16, || {
            // assembler.py:975 — unconditional B (not BL) to avoid
            // clobbering lr. RPython uses br ip0 (indirect), we use
            // B imm26 (direct) since the offset fits ±128 MB.
            let offset = adr_new_target as isize - stub_addr as isize;
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            let insn = 0x1400_0000 | imm26; // B imm26
            unsafe { (stub_addr as *mut u32).write(insn) };
        });

        flush_icache(stub_addr as *const u8, 16);

        // Verify patch was applied correctly
        if std::env::var_os("MAJIT_LOG").is_some() {
            let word = unsafe { (stub_addr as *const u32).read() };
            eprintln!(
                "[patch-verify] stub_addr={:#x} first_word={:#010x} target={:#x}",
                stub_addr, word, adr_new_target
            );
        }

        // assembler.py:987
        descr.set_adr_jump_offset(0); // "patched"
    }

    /// assembler.py:1138 redirect_call_assembler: patch old loop entry
    /// to JMP to new loop after retrace.
    pub fn redirect_call_assembler(old_addr: *const u8, new_addr: *const u8) {
        codebuf::with_writable(old_addr as *mut u8, 16, || {
            let offset = new_addr as isize - old_addr as isize;
            let imm26 = ((offset >> 2) & 0x03FF_FFFF) as u32;
            let insn = 0x1400_0000 | imm26;
            unsafe { (old_addr as *mut u32).write(insn) };
        });

        flush_icache(old_addr, 4);
    }

    // ----------------------------------------------------------------
    // genop_* — integer arithmetic
    // ----------------------------------------------------------------

    /// INT_ADD: result = arg0 + arg1
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_SUB: result = arg0 - arg1
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; sub x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MUL: result = arg0 * arg1
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; mul x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_AND: result = arg0 & arg1
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; and x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_OR: result = arg0 | arg1
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; orr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_XOR: result = arg0 ^ arg1
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; eor x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_NEG: result = -arg0
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; neg x0, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_INVERT: result = ~arg0
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; mvn x0, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_LSHIFT: result = arg0 << arg1
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; lsl x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; asr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; lsr x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — overflow arithmetic (assembler.py:1413-1425)
    // ----------------------------------------------------------------

    /// assembler.py:1856 genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; adds x0, x0, x1);
        self.store_rax_to_result(op.pos);
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1860 genop_int_sub_ovf.
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; subs x0, x0, x1);
        self.store_rax_to_result(op.pos);
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1864 genop_int_mul_ovf.
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        // aarch64/opassembler.py multiplies, computes SMULH, and then
        // compares the high word against the sign-extension of the low
        // word. regalloc.py's prepare_op_guard_no_overflow then uses
        // EQ for INT_MUL_OVF specifically.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; mul x2, x0, x1
            ; smulh x3, x0, x1
            ; asr x4, x2, 63
            ; cmp x3, x4
            ; mov x0, x2
        );
        self.store_rax_to_result(op.pos);
        self.guard_success_cc = Some(CC_E);
    }

    // ----------------------------------------------------------------
    // genop_* — comparisons
    // ----------------------------------------------------------------

    /// INT_LT/LE/GT/GE/EQ/NE/UINT_*: CMP arg0, arg1 then store CC.
    /// If the next op is a guard, guard_success_cc is set and consumed.
    /// Otherwise, materialize the boolean result via SETcc/CSET.
    fn genop_int_cmp(&mut self, op: &Op) {
        let cc = Self::opcode_to_cc(op.opcode);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, x1
        );

        // Store the CC for a following guard to consume.
        self.guard_success_cc = Some(cc);

        // Also materialize the boolean result for non-guard consumers.
        if !op.pos.is_none() {
            self.emit_setcc_to_result(cc, op.pos);
        }
    }

    /// Emit SETcc/CSET to materialize a boolean result.
    /// x64: SETcc AL; MOVZX EAX, AL
    /// aarch64: CSET X0, cc
    fn emit_setcc_to_result(&mut self, cc: u8, result_opref: OpRef) {
        // CSET Xd, cc — sets Xd to 1 if condition is true, 0 otherwise.
        // Note: CSET Xd, cc is an alias for CSINC Xd, XZR, XZR, invert(cc).
        match cc {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; cset x0, lt),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; cset x0, le),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; cset x0, gt),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ge),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ne),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; cset x0, lo),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; cset x0, ls),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; cset x0, hi),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; cset x0, hs),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; cset x0, vs),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; cset x0, vc),
            _ => dynasm!(self.mc ; .arch aarch64 ; cset x0, eq),
        }
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos);
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos.is_none() {
            self.emit_setcc_to_result(CC_E, op.pos);
        }
    }

    // ----------------------------------------------------------------
    // genop_* — guards
    // ----------------------------------------------------------------

    /// Emit a conditional jump to a recovery stub for a guard op.
    /// The condition is inverted: we jump to the stub when the guard FAILS.
    /// x86/assembler.py:1880-1891 _cmp_guard_class:
    ///   loc_ptr = locs[0]; loc_classptr = locs[1]
    ///   offset = self.cpu.vtable_offset
    ///   if offset is not None:
    ///       self.mc.CMP(mem(loc_ptr, offset), loc_classptr)
    ///   else:
    ///       assert isinstance(loc_classptr, ImmedLoc)
    ///       classptr = loc_classptr.value
    ///       expected_typeid = gc_ll_descr.
    ///           get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    ///       self._cmp_guard_gc_type(loc_ptr, ImmedLoc(expected_typeid))
    ///
    /// Inputs are pre-loaded: rax = loc_ptr (object), rcx = loc_classptr.
    /// Sets ZF=1 on equal, ZF=0 on not-equal — caller branches via Jcc.
    /// `expected_classptr_imm` is the immediate value of loc_classptr, used
    /// only by the gcremovetypeptr branch (RPython requires this to be an
    /// `ImmedLoc`).
    fn _cmp_guard_class_old(&mut self, expected_classptr_imm: Option<i64>) {
        if let Some(off_usize) = self.vtable_offset {
            // x86/assembler.py:1884-1885 vtable_offset path: full classptr CMP.
            let off_u32 = off_usize as u32;
            dynasm!(self.mc
                ; .arch aarch64
                ; ldr x0, [x0, #off_u32]
                ; cmp x0, x1
            );
        } else {
            // x86/assembler.py:1886-1891 gcremovetypeptr fallback +
            // x86/assembler.py:1893-1901 _cmp_guard_gc_type:
            //   on x86_64 the typeid is a 32-bit half-word at offset 0.
            let classptr = expected_classptr_imm.expect(
                "_cmp_guard_class: gcremovetypeptr requires loc_classptr \
                 to be an immediate (assert isinstance(loc_classptr, \
                 ImmedLoc) in x86/assembler.py:1887)",
            );
            let expected_typeid = self.lookup_typeid_from_classptr(classptr as usize).expect(
                "GuardClass: vtable_offset is None but the dynasm \
                     backend has no gc_ll_descr.get_typeid_from_classptr_if_\
                     gcremovetypeptr",
            );
            let typeid_w = expected_typeid as i32;
            dynasm!(self.mc
                ; .arch aarch64
                ; ldr w0, [x0]
                ; movz w1, #typeid_w as u32 & 0xffff
                ; movk w1, #(typeid_w as u32 >> 16) & 0xffff, lsl #16
                ; cmp w0, w1
            );
        }
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Looks up the materialized table populated by the runner from
    /// the active gc_ll_descr. RPython resolves the same value via
    /// `cpu.gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr`.
    fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        self.classptr_to_typeid.get(&(classptr as i64)).copied()
    }

    fn emit_guard_jcc(&mut self, fail_cc: u8) -> DynamicLabel {
        let fail_label = self.mc.new_dynamic_label();
        // aarch64: b.cond has 19-bit range (±1MB), which is too short
        // for forward references to recovery stubs. Use inverted condition
        // + unconditional branch (26-bit / ±128MB) pattern instead:
        //   b.NOT_cond >skip ; b =>fail_label ; skip:
        let skip = self.mc.new_dynamic_label();
        let inv = invert_cc(fail_cc);
        match inv {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; b.lt =>skip),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.le =>skip),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; b.gt =>skip),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.ge =>skip),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; b.lo =>skip),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.ls =>skip),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; b.hi =>skip),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.hs =>skip),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vs =>skip),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vc =>skip),
            CC_S => dynasm!(self.mc ; .arch aarch64 ; b.mi =>skip),
            CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.pl =>skip),
            _ => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
        }
        dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
        fail_label
    }

    fn emit_jcc_to_label(&mut self, fail_cc: u8, fail_label: DynamicLabel) {
        let skip = self.mc.new_dynamic_label();
        let inv = invert_cc(fail_cc);
        match inv {
            CC_L => dynasm!(self.mc ; .arch aarch64 ; b.lt =>skip),
            CC_LE => dynasm!(self.mc ; .arch aarch64 ; b.le =>skip),
            CC_G => dynasm!(self.mc ; .arch aarch64 ; b.gt =>skip),
            CC_GE => dynasm!(self.mc ; .arch aarch64 ; b.ge =>skip),
            CC_E => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
            CC_NE => dynasm!(self.mc ; .arch aarch64 ; b.ne =>skip),
            CC_B => dynasm!(self.mc ; .arch aarch64 ; b.lo =>skip),
            CC_BE => dynasm!(self.mc ; .arch aarch64 ; b.ls =>skip),
            CC_A => dynasm!(self.mc ; .arch aarch64 ; b.hi =>skip),
            CC_AE => dynasm!(self.mc ; .arch aarch64 ; b.hs =>skip),
            CC_O => dynasm!(self.mc ; .arch aarch64 ; b.vs =>skip),
            CC_NO => dynasm!(self.mc ; .arch aarch64 ; b.vc =>skip),
            CC_S => dynasm!(self.mc ; .arch aarch64 ; b.mi =>skip),
            CC_NS => dynasm!(self.mc ; .arch aarch64 ; b.pl =>skip),
            _ => dynasm!(self.mc ; .arch aarch64 ; b.eq =>skip),
        }
        dynasm!(self.mc ; .arch aarch64 ; b =>fail_label);
        dynasm!(self.mc ; .arch aarch64 ; =>skip);
    }

    /// assembler.py:2157 implement_guard — emit conditional jump to
    /// failure stub and append guard token to pending list.
    fn implement_guard(&mut self, op: &Op, fail_index: u32) {
        let cc = self
            .guard_success_cc
            .take()
            .expect("implement_guard: guard_success_cc not set");
        let fail_cc = invert_cc(cc);
        let fail_label = self.emit_guard_jcc(fail_cc);
        // guard_success_cc is already None after .take()

        self.append_guard_token(op, fail_index, fail_label);
    }

    /// Register a guard token without emitting a conditional jump.
    /// assembler.py:1799-1806 genop_guard_guard_not_invalidated pattern.
    fn implement_guard_nojump(&mut self, op: &Op, fail_index: u32) {
        let fail_label = self.mc.new_dynamic_label();
        self.append_guard_token(op, fail_index, fail_label);
    }

    /// Infer fail_arg_types from value_types or op.fail_arg_types.
    fn infer_fail_arg_types(&self, op: &Op) -> Vec<Type> {
        if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            if let Some(descr_types) = op
                .descr
                .as_ref()
                .and_then(|d| d.as_fail_descr())
                .map(|fd| fd.fail_arg_types().to_vec())
            {
                if !descr_types.is_empty() {
                    return descr_types;
                }
            }
        } else if let Some(fd) = op.descr.as_ref().and_then(|d| d.as_fail_descr()) {
            // Step A (43c64ee0bb) installs op.descr = ResumeGuardDescr
            // with post-numbering fail_arg_types via
            // store_final_boxes_in_guard (optimizeopt/mod.rs:3393-3404).
            // Prefer the descr for guards too; fall through to
            // op.fail_arg_types only for sharing-path guards
            // (optimizeopt/mod.rs:3068-3088) where op.descr=None.
            let dt = fd.fail_arg_types();
            let expected_len = op.fail_args.as_ref().map(|fa| fa.len()).unwrap_or(0);
            if dt.len() == expected_len && !dt.is_empty() {
                return dt.to_vec();
            }
        }
        if let Some(ref ts) = op.fail_arg_types {
            let expected_len = if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.args.len()
            } else {
                op.fail_args.as_ref().map(|fa| fa.len()).unwrap_or(0)
            };
            if ts.len() == expected_len {
                ts.clone()
            } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.args
                    .iter()
                    .map(|opref| self.value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                    .collect()
            } else if let Some(ref fa) = op.fail_args {
                fa.iter()
                    .map(|opref| {
                        if opref.is_none() {
                            Type::Ref
                        } else {
                            self.value_types.get(&opref.0).copied().unwrap_or(Type::Ref)
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else if let Some(ref fa) = op.fail_args {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        Type::Ref
                    } else {
                        self.value_types.get(&opref.0).copied().unwrap_or(Type::Ref)
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Common tail: build DynasmFailDescr, create GuardToken, push to
    /// pending_guard_tokens and fail_descrs.
    fn append_guard_token(&mut self, op: &Op, fail_index: u32, fail_label: DynamicLabel) {
        let fail_arg_types = self.infer_fail_arg_types(op);
        let fail_args = op
            .fail_args
            .as_ref()
            .map(|fa| fa.to_vec())
            .unwrap_or_default();
        // Dynasm helper parity: fail_arg_locs are absolute jf_frame slots
        // consumed via JitFrame::get_int_value(). None = virtual/unmapped.
        let fail_arg_locs: Vec<Option<usize>> = fail_args
            .iter()
            .map(|opref| {
                if opref.is_none() || opref.is_constant() {
                    None
                } else {
                    self.opref_to_slot
                        .get(&opref.0)
                        .copied()
                        .map(|slot| slot + JITFRAME_FIXED_SIZE)
                }
            })
            .collect();
        let mut descr = DynasmFailDescr::new(fail_index, self.trace_id, fail_arg_types, false);
        descr.fail_arg_locs = fail_arg_locs;
        let descr = Arc::new(descr);
        let gcmap = self.gcmap_from_fail_arg_locs(&descr.fail_arg_types, &descr.fail_arg_locs);
        self.pending_guard_tokens.push(GuardToken {
            fail_label,
            fail_descr: descr.clone(),
            const_stores: Vec::new(),
            gcmap,
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // assembler.py:1773 genop_guard_guard_true
    // ----------------------------------------------------------------

    /// regalloc.py:429 load_condition_into_cc — if guard_success_cc
    /// is not yet set, emit TEST arg0 and set cc = NZ.
    fn load_condition_into_cc(&mut self, op: &Op) {
        if self.guard_success_cc.is_none() {
            self.load_arg_to_rax(op.arg(0));
            dynasm!(self.mc ; .arch aarch64 ; cmp x0, 0);
            self.guard_success_cc = Some(CC_NE); // rx86.Conditions['NZ']
        }
    }

    /// assembler.py:1773 genop_guard_guard_true.
    /// genop_guard_guard_nonnull = genop_guard_guard_true (alias)
    /// genop_guard_guard_no_overflow = genop_guard_guard_true (alias)
    fn genop_guard_guard_true(&mut self, op: &Op, fail_index: u32) {
        self.load_condition_into_cc(op);
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1777 genop_guard_guard_false.
    /// genop_guard_guard_isnull = genop_guard_guard_false (alias)
    /// genop_guard_guard_overflow = genop_guard_guard_false (alias)
    fn genop_guard_guard_false(&mut self, op: &Op, fail_index: u32) {
        self.load_condition_into_cc(op);
        self.guard_success_cc = Some(invert_cc(self.guard_success_cc.take().unwrap()));
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1871 genop_guard_guard_value.
    fn genop_guard_guard_value(&mut self, op: &Op, fail_index: u32) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, x1);

        self.guard_success_cc = Some(CC_E);
        self.implement_guard(op, fail_index);

        // regalloc.py:496-501 make_a_counter_per_value
        if let Some(descr) = self.fail_descrs.last() {
            if let Some(fa) = op.fail_args.as_ref() {
                let arg0 = op.arg(0);
                if let Some(idx) = fa.iter().position(|&r| r == arg0) {
                    let type_tag = match descr.fail_arg_types.get(idx) {
                        Some(Type::Ref) => DynasmFailDescr::TY_REF,
                        Some(Type::Float) => DynasmFailDescr::TY_FLOAT,
                        _ => DynasmFailDescr::TY_INT,
                    };
                    descr.make_a_counter_per_value(idx as u32, type_tag);
                }
            }
        }
    }

    /// assembler.py:1903 genop_guard_guard_class.
    fn genop_guard_guard_class(&mut self, op: &Op, fail_index: u32) {
        let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => Some(v),
            _ => None,
        };
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        self._cmp_guard_class_old(expected_classptr_imm);
        self.guard_success_cc = Some(CC_E);
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1908 genop_guard_guard_nonnull_class.
    /// CMP(locs[0], imm1); JB → fail; _cmp_guard_class; JNE → fail
    fn genop_guard_guard_nonnull_class(&mut self, op: &Op, fail_index: u32) {
        self.load_arg_to_rax(op.arg(0));
        // assembler.py:1909 CMP(locs[0], imm1) — JB catches NULL (0)
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, 1);

        // assembler.py:1911 emit_forward_jump('B') — jump if below (NULL)
        let fail_label = self.emit_guard_jcc(CC_B);

        let expected_classptr_imm = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => Some(v),
            _ => None,
        };
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        self._cmp_guard_class_old(expected_classptr_imm);

        // assembler.py:1914 patch_forward_jump — both paths share the
        // same fail_label, so a second JNE to the same target suffices.
        dynasm!(self.mc ; .arch aarch64 ; b.ne =>fail_label);

        // assembler.py:1916-1917
        // guard_success_cc is not used here — we already emitted the jumps.
        // Manually append the guard token.
        self.append_guard_token(op, fail_index, fail_label);
    }

    /// assembler.py:1794 generate_guard_no_exception:
    /// `LDR loc, [loc]; CMP loc, #0` with success on zero.
    fn emit_guard_no_exception_check(&mut self) {
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        self.emit_mov_imm64(16, exc_type_addr);
        dynasm!(self.mc ; .arch aarch64
            ; ldr x16, [x16]
            ; cmp x16, xzr
        );
        self.guard_success_cc = Some(CC_E);
    }

    /// assembler.py:1782 genop_guard_guard_no_exception.
    fn genop_guard_guard_no_exception(&mut self, op: &Op, fail_index: u32) {
        self.emit_guard_no_exception_check();
        self.implement_guard(op, fail_index);
    }

    /// assembler.py:1799 genop_guard_guard_not_invalidated.
    /// Does NOT emit a conditional jump — records position for external
    /// invalidation patching.
    fn genop_guard_guard_not_invalidated(&mut self, op: &Op, fail_index: u32) {
        self.implement_guard_nojump(op, fail_index);
    }

    /// assembler.py:2207-2222 _store_force_index: before a call that may force,
    /// store the next GUARD_NOT_FORCED's fail descr ptr to jf_force_descr,
    /// and zero jf_descr so GUARD_NOT_FORCED's CMP [jf_descr], 0 starts clean.
    fn _store_force_index_if_next_guard(&mut self, ops: &[Op], op_idx: usize, fail_index: u32) {
        // assembler.py:2224-2226 _find_nearby_operation(+1)
        let next_idx = op_idx + 1;
        if next_idx >= ops.len() {
            return;
        }
        let next_op = &ops[next_idx];
        if next_op.opcode != OpCode::GuardNotForced && next_op.opcode != OpCode::GuardNotForced2 {
            return;
        }
        // Pre-allocate the fail descr for the next GUARD_NOT_FORCED.
        // The full metadata (faillocs, rd_numb, etc.) will be filled in
        // when the guard is actually emitted in append_guard_token_with_faillocs.
        let fail_arg_types = self.infer_fail_arg_types(next_op);
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types,
            false,
        ));
        let descr_ptr = Arc::as_ptr(&descr) as i64;
        self.pending_force_descr = Some(descr);

        // x86/assembler.py:2210-2222: store descr to jf_force_descr,
        // zero jf_descr.
        self.emit_mov_imm64(16, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str X(16), [x29, JF_FORCE_DESCR_OFS as u32]
            ; str xzr, [x29, JF_DESCR_OFS as u32]
        );
    }

    /// assembler.py:2228-2232 genop_guard_guard_not_forced.
    /// Check jf_descr == 0 (not forced); if non-zero, guard fails.
    fn genop_guard_guard_not_forced(&mut self, op: &Op, fail_index: u32) {
        // assembler.py:2229-2231: CMP [jf_descr], 0; guard_success_cc = 'E'
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x29, JF_DESCR_OFS as u32]
            ; cmp x0, 0
        );
        self.guard_success_cc = Some(CC_E);
        self.implement_guard(op, fail_index);
    }

    // ----------------------------------------------------------------
    // genop_* — control flow
    // ----------------------------------------------------------------

    /// LABEL: define the back-edge target for JUMP.
    ///
    /// RPython: LABEL does NOT emit code. The regalloc establishes
    /// the slot mapping. JUMP handles slot remapping.
    ///
    /// In our frame-slot model: preamble values may be in non-canonical
    /// slots. We emit copies from old→canonical slots BEFORE the
    /// LABEL binding, so they execute only on first entry from the
    /// preamble. JUMP writes directly to canonical slots and jumps
    /// to the LABEL, skipping the copies.
    fn genop_label(&mut self, op: &Op) {
        // Emit preamble→canonical copies BEFORE the label.
        // Two-pass push/pop: safely handles slot overlaps.
        let n_label = op.args.len();
        // Pass 1: push source values
        for i in 0..n_label {
            let arg_ref = op.args[i];
            if arg_ref.is_none() {
                let dst = Self::slot_offset(i);
                dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x29, dst as u32] ; str x0, [sp, #-16]!);
            } else if arg_ref.is_constant() {
                let val = self.constants.get(&arg_ref.0).copied().unwrap_or(0);
                self.emit_mov_imm64(0, val);
                dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!);
            } else if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref.0) {
                let src = Self::slot_offset(old_slot);
                dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x29, src as u32] ; str x0, [sp, #-16]!);
            } else {
                dynasm!(self.mc ; .arch aarch64 ; str xzr, [sp, #-16]!);
            }
        }
        // Pass 2: pop in reverse into canonical slots
        for i in (0..n_label).rev() {
            let dst = Self::slot_offset(i);
            dynasm!(self.mc ; .arch aarch64 ; ldr x0, [sp], #16 ; str x0, [x29, dst as u32]);
        }

        // Bind the LABEL — JUMP targets here (after the copies).
        let label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; =>label);
        if let Some(descr) = loop_target_descr(op) {
            descr.set_ll_loop_code(self.mc.offset().0);
            if let Some(id) = loop_target_id(op) {
                self.target_tokens_currently_compiling.insert(id, label);
            }
            if let Some(descr_ref) = op.descr.as_ref() {
                self.compiled_target_tokens.push(descr_ref.clone());
            }
        }

        // Remap: Label's arg[i] → canonical slot i
        for (i, &arg_ref) in op.args.iter().enumerate() {
            if !arg_ref.is_none() {
                self.opref_to_slot.insert(arg_ref.0, i);
            }
        }
        self.next_slot = self.next_slot.max(op.args.len());
    }

    /// jump.py:66 _move: emit a single slot-to-slot or const-to-slot move.
    fn emit_slot_move(&mut self, src: i32, dst: i32, is_const: bool, val: i64) {
        if is_const {
            self.emit_mov_imm64(0, val);
            dynasm!(self.mc ; .arch aarch64 ; str x0, [x29, dst as u32]);
        } else if src != dst {
            dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x29, src as u32]
                ; str x0, [x29, dst as u32]
            );
        }
    }

    /// JUMP: unconditional branch to the loop label.
    /// jump.py:1 remap_frame_layout parity: parallel move algorithm
    /// to handle cyclic slot dependencies.
    fn genop_jump(&mut self, op: &Op) {
        // Build src→dst move list.
        // Each entry: (src_offset_or_const, dst_offset, is_const, const_val)
        let n = op.args.len();
        let mut moves: Vec<(i32, i32, bool, i64)> = Vec::with_capacity(n);
        for (i, &arg_ref) in op.args.iter().enumerate() {
            let dst = Self::slot_offset(i);
            match self.resolve_opref(arg_ref) {
                ResolvedArg::Slot(src) => moves.push((src, dst, false, 0)),
                ResolvedArg::Const(val) => moves.push((0, dst, true, val)),
            }
        }

        // jump.py:1-64 remap_frame_layout: topological order with
        // cycle breaking via push/pop.
        // srccount[dst] = number of times dst appears as a src
        let mut srccount: HashMap<i32, i32> = HashMap::new();
        for m in &moves {
            srccount.entry(m.1).or_insert(0); // ensure dst exists
        }
        let mut pending = n as i32;
        for (i, m) in moves.iter().enumerate() {
            if m.2 {
                continue;
            } // constant → no src dependency
            let src = m.0;
            if let Some(cnt) = srccount.get_mut(&src) {
                if src == moves[i].1 {
                    // self-move: skip
                    *cnt = -(n as i32) - 1;
                    pending -= 1;
                } else {
                    *cnt += 1;
                }
            }
        }

        while pending > 0 {
            let mut progress = false;
            for i in 0..n {
                let dst = moves[i].1;
                if srccount.get(&dst).copied().unwrap_or(-1) == 0 {
                    *srccount.get_mut(&dst).unwrap() = -1; // done
                    pending -= 1;
                    if !moves[i].2 {
                        let src = moves[i].0;
                        if let Some(cnt) = srccount.get_mut(&src) {
                            *cnt -= 1;
                        }
                    }
                    self.emit_slot_move(moves[i].0, dst, moves[i].2, moves[i].3);
                    progress = true;
                }
            }
            if !progress {
                // Cycle: use push/pop to break it.
                for i in 0..n {
                    let dst = moves[i].1;
                    if srccount.get(&dst).copied().unwrap_or(-1) >= 0 {
                        // Push first dst in the cycle
                        dynasm!(self.mc ; .arch aarch64
                            ; ldr x0, [x29, dst as u32]
                            ; str x0, [sp, #-16]!
                        );
                        // Walk the cycle
                        let mut cur = i;
                        loop {
                            let cd = moves[cur].1;
                            *srccount.get_mut(&cd).unwrap() = -1;
                            pending -= 1;
                            // Find the move whose dst is this src
                            let src = moves[cur].0;
                            let next = moves.iter().position(|m| m.1 == src);
                            if let Some(ni) = next {
                                if srccount.get(&moves[ni].1).copied().unwrap_or(-1) < 0 {
                                    // End of cycle: pop into this slot
                                    dynasm!(self.mc ; .arch aarch64
                                        ; ldr x0, [sp], #16
                                        ; str x0, [x29, cd as u32]
                                    );
                                    break;
                                }
                                self.emit_slot_move(src, cd, false, 0);
                                cur = ni;
                            } else {
                                // No cycle found — emit move and break
                                self.emit_slot_move(moves[cur].0, cd, moves[cur].2, moves[cur].3);
                                break;
                            }
                        }
                    }
                }
            }
        }

        let jump_descr = loop_target_descr(op);
        if let Some(label) =
            loop_target_id(op).and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
        {
            // Same-buffer jump (loop body)
            dynasm!(self.mc ; .arch aarch64 ; b =>label);
        } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
            // assembler.py closing_jump parity: bridge jumps back to
            // the original loop's LABEL via absolute address.
            self.emit_mov_imm64(0, target as i64);
            dynasm!(self.mc ; .arch aarch64 ; br x0);
        }
    }

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    fn genop_finish(&mut self, op: &Op, fail_index: u32) {
        // compiler.rs:9667-9681 parity: trust explicit FINISH types only when
        // they match the actual result arity; otherwise infer from the op args.
        let finish_refs: Vec<OpRef> = op.args.iter().copied().collect();
        let fail_arg_types = if let Some(ref explicit) = op.fail_arg_types {
            if explicit.len() == finish_refs.len() {
                explicit.clone()
            } else {
                finish_refs
                    .iter()
                    .map(|opref| self.value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                    .collect()
            }
        } else {
            finish_refs
                .iter()
                .map(|opref| self.value_types.get(&opref.0).copied().unwrap_or(Type::Int))
                .collect()
        };
        // compile.py:618-669 parity: use type-specific global singleton.
        let descr = Arc::new(DynasmFailDescr::new(
            fail_index,
            self.trace_id,
            fail_arg_types.clone(),
            true,
        ));
        // Finish ops write the type-appropriate singleton pointer to jf_descr
        // so CALL_ASSEMBLER's fast path CMP matches the correct variant.
        let result_type = if fail_arg_types.is_empty() {
            Type::Void
        } else {
            fail_arg_types[0]
        };
        let global_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);

        // If there's a result argument, store it to jf_frame[0].
        // assembler.py:2291-2303 parity: float results use xmm0/MOVSD.
        if op.num_args() > 0 {
            let arg0 = op.arg(0);
            let slot0_offset = Self::slot_offset(0);
            if result_type == Type::Float {
                // Float: load to xmm0, store via MOVSD
                self.load_arg_to_rax(arg0); // loads raw bits
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, slot0_offset as u32]
                );
            } else {
                self.load_arg_to_rax(arg0);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x29, slot0_offset as u32]
                );
            }
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        // compile.py:665-674 parity: use global singleton pointer.
        let descr_ptr = global_descr_ptr;
        self.emit_mov_imm64(0, descr_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x29, JF_DESCR_OFS as u32]
        );

        if result_type == Type::Ref {
            if let Some(gcmap) = self.finish_gcmap {
                gcmap_set_bit(gcmap, 0);
                self.push_gcmap(gcmap);
            } else {
                self.push_gcmap(self.gcmap_for_finish);
            }
        } else if let Some(gcmap) = self.finish_gcmap {
            self.push_gcmap(gcmap);
        } else {
            self.pop_gcmap();
        }

        // Emit epilogue (return jf_ptr).
        self._call_footer();

        self.fail_descrs.push(descr);
    }

    // ----------------------------------------------------------------
    // genop_* — type conversions
    // ----------------------------------------------------------------

    /// SAME_AS: result = arg0 (copy value)
    /// SAME_AS: result = arg0 (identity).
    /// regalloc.py parity: no code emitted — just alias the slot.
    fn genop_same_as(&mut self, op: &Op) {
        let arg = op.arg(0);
        if let Some(&slot) = self.opref_to_slot.get(&arg.0) {
            self.opref_to_slot.insert(op.pos.0, slot);
        } else {
            self.load_arg_to_rax(arg);
            self.store_rax_to_result(op.pos);
        }
    }

    // ----------------------------------------------------------------
    // Float helpers
    // ----------------------------------------------------------------

    /// Load a float value from `opref` into XMM0 (x64) / D0 (aarch64).
    /// Float values are stored as bit-cast i64 in frame slots.
    fn load_float_arg_to_d0(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr d0, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                // Load constant via integer register, then move to float register.
                self.emit_mov_imm64(0, val);
                dynasm!(self.mc ; .arch aarch64
                    ; fmov d0, x0
                );
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr d1, [x29, offset as u32]
                );
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(1, val);
                dynasm!(self.mc ; .arch aarch64
                    ; fmov d1, x1
                );
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        dynasm!(self.mc ; .arch aarch64
            ; str d0, [x29, offset as u32]
        );
    }

    // ----------------------------------------------------------------
    // genop_* — float arithmetic
    // x86/assembler.py:1648 genop_float_add etc.
    // aarch64/assembler.py float equivalents
    // ----------------------------------------------------------------

    /// FLOAT_ADD: result = arg0 + arg1
    fn genop_float_add(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; fadd d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_SUB: result = arg0 - arg1
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; fsub d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_MUL: result = arg0 * arg1
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; fmul d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; fdiv d0, d0, d1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; fneg d0, d0
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; scvtf d0, x0
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; fcvtzs x0, d0
        );
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.
    // ----------------------------------------------------------------

    /// Extract the byte offset from an op's FieldDescr.
    /// Returns 0 if no field descriptor is present.
    fn field_offset_from_descr(op: &Op) -> i32 {
        op.descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.offset() as i32)
            .unwrap_or(0)
    }

    /// Extract the field size from an op's FieldDescr.
    /// Returns 8 (WORD) if no field descriptor is present.
    fn field_size_from_descr(op: &Op) -> usize {
        op.descr
            .as_ref()
            .and_then(|d| d.as_field_descr())
            .map(|fd| fd.field_size())
            .unwrap_or(8)
    }

    /// GETFIELD_GC_*: result = [arg0 + offset]
    /// The offset comes from the op's FieldDescr.
    fn genop_getfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load the object pointer from arg0.
        self.load_arg_to_rax(op.arg(0));

        // Load the field value at [rax + offset] into rax/x0.

        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0, offset as u32]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    /// SETFIELD_GC: [arg0 + offset] = arg1
    fn genop_discard_setfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load object pointer into rax/x0 and value into rcx/x1.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        match size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; strb w1, [x0, offset as u32]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; strh w1, [x0, offset as u32]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; str w1, [x0, offset as u32]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; str x1, [x0, offset as u32]
            ),
        }
    }

    /// GETARRAYITEM_GC_*: result = array[index]
    /// arg0 = array pointer, arg1 = index.
    /// The base_size and item_size come from the op's ArrayDescr.
    fn genop_getarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer and index.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        // Compute address: rax = rax + base_size + rcx * item_size

        // x1 = x1 * item_size; x0 = x0 + base_size + x1
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64); // x2 = item_size
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    /// SETARRAYITEM_GC: array[index] = value
    /// arg0 = array pointer, arg1 = index, arg2 = value.
    fn genop_discard_setarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0));
        // Load index.
        self.load_arg_to_rcx(op.arg(1));

        // Compute element address: rax = rax + base_size + rcx * item_size
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );

        // Now load value from arg2 and store it.
        // We need a third register: use rcx/x1 again for the value
        // (the address is in rax/x0).
        // Save rax/x0 (element address) before loading value.

        // Save x0 (element address) in x2, load value into x1.
        dynasm!(self.mc ; .arch aarch64
            ; mov x2, x0
        );
        self.load_arg_to_rcx(op.arg(2)); // loads into x1
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; strb w1, [x2]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; strh w1, [x2]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; str w1, [x2]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; str x1, [x2]
            ),
        }
    }

    /// ARRAYLEN_GC: result = array.length
    /// The length field location comes from the ArrayDescr's len_descr().
    fn genop_arraylen(&mut self, op: &Op) {
        let len_offset = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .and_then(|ad| ad.len_descr())
            .map(|ld| ld.offset() as i32)
            .unwrap_or(0); // Default: length at offset 0 in array header

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0));

        // Load length from [array + len_offset].
        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, len_offset as u32]
        );

        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — calls
    // x86/assembler.py:2230 _genop_call
    // ----------------------------------------------------------------

    fn argloc_imm(arglocs: &[Loc], index: usize) -> i64 {
        match arglocs.get(index) {
            Some(Loc::Immed(i)) => i.value,
            _ => 0,
        }
    }

    /// Emit a function call. `func_arg` is the index of the function
    /// pointer arg; call arguments start at `func_arg + 1`.
    fn emit_call(&mut self, op: &Op, func_arg: usize) {
        let arg_count = op.num_args();

        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        for i in (func_arg + 1)..arg_count.min(func_arg + 7) {
            let arg = op.arg(i);
            let abi_idx = i - func_arg - 1;
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => {
                    let reg = abi_idx as u8;
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr X(reg), [x29, offset as u32]
                    );
                }
                ResolvedArg::Const(val) => {
                    let reg = abi_idx as u32;
                    self.emit_mov_imm64(reg, val);
                }
            }
        }

        match self.resolve_opref(op.arg(func_arg)) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x8, [x29, offset as u32]
                    ; blr x8
                );
            }
            ResolvedArg::Const(val) => {
                self.emit_mov_imm64(8, val);
                dynasm!(self.mc ; .arch aarch64 ; blr x8);
            }
        }

        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    /// aarch64/opassembler.py:1036 _emit_call + aarch64/callbuilder.py:21-67
    /// prepare_arguments.
    ///
    /// Register-to-ABI-reg shuffles must go through remap_frame_layout to
    /// handle cycles: with naïve sequential `mov x0,x2; mov x1,x0` the
    /// second move reads the already-clobbered x0. RPython routes the
    /// non-float arg plan through remap_frame_layout(asm, src, dst, ip0)
    /// and preserves the fnloc across the remap via ip1 when it is a
    /// register.
    fn emit_call_from_arglocs(&mut self, arglocs: &[Loc], func_index: usize) {
        let arg_count = arglocs.len();

        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);

        // aarch64/callbuilder.py:54-60 — if fnloc is a core reg or stack,
        // save it to ip1 (x17) so later remap can clobber its source reg.
        let fnloc = arglocs.get(func_index).copied();
        let fnloc_in_ip1 = match fnloc {
            Some(Loc::Reg(r)) if !r.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; mov x17, X(r.value));
                true
            }
            Some(Loc::Frame(f)) if !f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch aarch64 ; ldr x17, [x29, f.ebp_loc.value as u32]);
                true
            }
            _ => false,
        };

        // aarch64/callbuilder.py:29-41 — build arg plan.
        // ABI regs x0..x7 for core, d0..d7 for float.
        let mut non_float_src: Vec<Loc> = Vec::new();
        let mut non_float_dst: Vec<Loc> = Vec::new();
        let mut float_src: Vec<Loc> = Vec::new();
        let mut float_dst: Vec<Loc> = Vec::new();
        let mut immed_args: Vec<(u8, i64, bool)> = Vec::new(); // (abi_idx, value, is_float)

        for i in (func_index + 1)..arg_count.min(func_index + 9) {
            let abi_idx = (i - func_index - 1) as u8;
            let arg = arglocs[i];
            match arg {
                Loc::Frame(f) if f.ebp_loc.is_float => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Frame(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Reg(r) if r.is_xmm => {
                    float_src.push(arg);
                    float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, true)));
                }
                Loc::Reg(_) => {
                    non_float_src.push(arg);
                    non_float_dst.push(Loc::Reg(crate::regloc::RegLoc::new(abi_idx, false)));
                }
                Loc::Immed(im) => {
                    immed_args.push((abi_idx, im.value, im.is_float));
                }
                _ => {}
            }
        }

        // aarch64/callbuilder.py:62-65 — remap non-float then float args.
        let tmp_nf = Loc::Reg(crate::regloc::RegLoc::new(16, false)); // x16 (ip0)
        let tmp_fp = Loc::Reg(crate::regloc::RegLoc::new(15, true)); // d15
        self.remap_frame_layout(&non_float_src, &non_float_dst, tmp_nf);
        self.remap_frame_layout(&float_src, &float_dst, tmp_fp);

        // Immediate args after remap (each targets a distinct ABI reg).
        for (abi_idx, val, is_float) in immed_args {
            if is_float {
                self.emit_mov_imm64(15, val);
                dynasm!(self.mc ; .arch aarch64 ; fmov D(abi_idx), X(15));
            } else {
                self.emit_mov_imm64(abi_idx as u32, val);
            }
        }

        // aarch64/callbuilder.py:67 — restore fnloc from ip1 / emit call.
        if fnloc_in_ip1 {
            dynasm!(self.mc ; .arch aarch64 ; blr x17);
        } else {
            match fnloc {
                Some(Loc::Immed(i)) => {
                    let val = i.value;
                    self.emit_mov_imm64(8, val);
                    dynasm!(self.mc ; .arch aarch64 ; blr x8);
                }
                _ => {}
            }
        }

        dynasm!(self.mc ; .arch aarch64 ; ldp x29, x30, [sp], #16);
    }

    fn ensure_call_result_bit_extension(&mut self, arglocs: &[Loc]) {
        let size = Self::argloc_imm(arglocs, 1) as usize;
        let signed = Self::argloc_imm(arglocs, 2) != 0;
        if size >= WORD {
            return;
        }

        match size {
            4 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; asr x0, x0, 32);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 32 ; lsr x0, x0, 32);
                }
            }
            2 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 48 ; asr x0, x0, 48);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFFFF);
                }
            }
            1 => {
                if signed {
                    dynasm!(self.mc ; .arch aarch64 ; lsl x0, x0, 56 ; asr x0, x0, 56);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; and x0, x0, 0xFF);
                }
            }
            _ => {}
        }
    }

    /// assembler.py:2176 _genop_call — internal call implementation.
    fn _genop_call(&mut self, op: &Op) {
        self.emit_call(op, 0);
    }

    fn _genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        let func_index = 3 + usize::from(op.opcode.is_call_release_gil());
        self.emit_call_from_arglocs(arglocs, func_index);
        if op.opcode.result_type() == Type::Int {
            self.ensure_call_result_bit_extension(arglocs);
        }
    }

    /// assembler.py:2169-2174 _genop_real_call.
    /// genop_call_i = genop_call_r = genop_call_f = genop_call_n
    fn genop_call(&mut self, op: &Op) {
        self._genop_call(op);
        if !op.pos.is_none() {
            if op.opcode.result_type() == Type::Float {
                self.store_d0_to_result(op.pos);
            } else {
                self.store_rax_to_result(op.pos);
            }
        }
    }

    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        self._genop_call_with_arglocs(op, arglocs);
        if !op.pos.is_none() {
            if op.opcode.result_type() == Type::Float {
                self.store_d0_to_result(op.pos);
            } else {
                self.store_rax_to_result(op.pos);
            }
        }
    }

    /// assembler.py:295-360 call_assembler: invoke a compiled JIT loop.
    ///
    /// RPython fast path (assembler.py:295-360):
    ///   1. _call_assembler_emit_call — call the target trace
    ///   2. _call_assembler_check_descr — CMP jf_descr == done_with_this_frame_descr
    ///   3. Path A (slow): call assembler_helper
    ///   4. Path B (fast): MOV result, [frame + ofs]
    ///   5. join paths
    ///
    /// RPython allocates the callee jitframe via malloc_jitframe (heap).
    /// We use malloc for parity: the callee's frame-slot model needs
    /// frame_depth slots (not just num_args), and heap allocation avoids
    /// stack overflow on deep recursion.
    /// assembler.py:295-360 call_assembler parity.
    /// Uses regalloc-provided arglocs to load callee arguments instead of
    /// resolve_opref(), which drops register-carried values to Const(0).
    fn genop_call_assembler(&mut self, op: &Op, arglocs: &[Loc]) {
        let call_descr = op.descr.as_ref().and_then(|d| d.as_call_descr());
        let expansion = call_descr.and_then(|d| d.vable_expansion());

        let num_args = op.args.len();
        let num_expanded_items = expansion
            .map(|exp| 1 + exp.scalar_fields.len() + exp.num_array_items)
            .unwrap_or(num_args);
        // llmodel.py:298 malloc_jitframe parity: callee needs frame_depth
        // slots (frame-slot model stores ALL intermediates in jitframe).
        let jf_slots = self.frame_depth.max(num_expanded_items);
        let jf_alloc_bytes = crate::jitframe::JitFrame::alloc_size(jf_slots) as i64;
        let jf_frame_len = jf_slots as i64;
        let calloc_ptr = libc::calloc as *const () as i64;
        let free_ptr = libc::free as *const () as i64;

        // Save callee-saved regs used as scratch by this sequence.
        // genop_call_assembler uses x19 (caller jf_ptr) and x20 (callee jf_ptr)
        // across calloc/call/free. These are callee-saved by ABI and saved
        // in _call_header, but the regalloc may have assigned them to
        // live variables. Push/pop to preserve the outer state.
        dynasm!(self.mc ; .arch aarch64
            ; stp x19, x20, [sp, #-16]!  // save x19/x20 on stack
            ; mov x19, x29               // x19 = caller's jf_ptr
        );

        // Allocate callee jitframe on heap via calloc.
        // Stack alignment: after prologue (push rbp + push r12) + return
        // addr, rsp ≡ -8 (mod 16). sub 8 aligns to 16 for ABI call.
        self.emit_mov_imm64(0, 1);
        self.emit_mov_imm64(1, jf_alloc_bytes);
        self.emit_mov_imm64(2, calloc_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; sub sp, sp, 16                // align
            ; blr x2                        // x0 = heap jf_ptr
            ; add sp, sp, 16                // unalign
        );

        // rdx/x20 = heap jf_ptr (held across arg stores).
        // load_arg_to_rax reads from [rbp+offset], rbp still = caller's jf.
        // Wait: rbp was saved to r12 but NOT changed yet. Actually we did
        // `mov r12, rbp` above, so rbp still points to caller's jf. ✓
        dynasm!(self.mc ; .arch aarch64
            ; mov x20, x0             // x20 = heap jf_ptr
            ; str xzr, [x20, JF_DESCR_OFS as u32]
        );
        self.emit_mov_imm64(0, jf_frame_len);
        dynasm!(self.mc ; .arch aarch64
            ; str x0, [x20, JF_FRAME_OFS as u32]
        );

        // rewrite.py:665-695 handle_call_assembler parity:
        // if VableExpansion is present, expand the caller frame reference
        // into the callee's full inputarg layout. Otherwise, copy the
        // raw CALL_ASSEMBLER arguments as ordinary loop inputs.
        //
        // All callee inputs live at absolute jitframe slots
        // [JITFRAME_FIXED_SIZE + relative_input_index].
        if let Some(expansion) = expansion {
            for slot in 0..num_expanded_items {
                let dest_offset = Self::slot_offset(JITFRAME_FIXED_SIZE + slot);

                if let Some(&(_, cval)) = expansion.const_overrides.iter().find(|(s, _)| *s == slot)
                {
                    self.emit_mov_imm64(0, cval);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x0, [x20, dest_offset as u32]
                    );
                    continue;
                }

                if let Some(&(_, arg_idx)) =
                    expansion.arg_overrides.iter().find(|(s, _)| *s == slot)
                {
                    let src = arglocs
                        .get(arg_idx)
                        .copied()
                        .expect("call_assembler arg override out of bounds");
                    self.emit_load_to_rax(src);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x0, [x20, dest_offset as u32]
                    );
                    continue;
                }

                if slot == 0 {
                    let frame_loc = arglocs
                        .first()
                        .copied()
                        .expect("call_assembler vable expansion missing frame arg");
                    self.emit_load_to_rax(frame_loc);
                    dynasm!(self.mc ; .arch aarch64
                        ; str x0, [x20, dest_offset as u32]
                    );
                    continue;
                }

                if slot <= expansion.scalar_fields.len() {
                    let (field_ofs, _) = expansion.scalar_fields[slot - 1];
                    let frame_loc = arglocs
                        .first()
                        .copied()
                        .expect("call_assembler vable expansion missing frame arg");
                    self.emit_load_to_rax(frame_loc);
                    dynasm!(self.mc ; .arch aarch64
                        ; ldr x0, [x0, field_ofs as u32]
                        ; str x0, [x20, dest_offset as u32]
                    );
                    continue;
                }

                let array_index = slot - 1 - expansion.scalar_fields.len();
                let data_ptr_ofs = expansion.array_struct_offset + expansion.array_ptr_offset;
                let item_ofs = (array_index * 8) as i32;
                let frame_loc = arglocs
                    .first()
                    .copied()
                    .expect("call_assembler vable expansion missing frame arg");
                self.emit_load_to_rax(frame_loc);
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x0, [x0, data_ptr_ofs as u32]
                    ; ldr x0, [x0, item_ofs as u32]
                    ; str x0, [x20, dest_offset as u32]
                );
            }
        } else {
            for (i, loc) in arglocs.iter().enumerate() {
                let dest_offset = Self::slot_offset(JITFRAME_FIXED_SIZE + i);
                self.emit_load_to_rax(*loc);
                dynasm!(self.mc ; .arch aarch64
                    ; str x0, [x20, dest_offset as u32]
                );
            }
        }

        // _call_assembler_emit_call (assembler.py:2267-2269):
        // rdi/x0 = callee jf_ptr.
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x20             // x0 = heap jf ptr
        );

        // assembler.py:320 _call_assembler_emit_call(descr._ll_function_addr, ...)
        // Resolve target address from descr.call_target_token() or self_entry_label.
        let target_addr: Option<usize> = op
            .descr
            .as_ref()
            .and_then(|d| d.as_loop_token_descr())
            .map(|td| td.loop_token_number())
            .and_then(|token| self.call_assembler_targets.get(&token).copied());

        // Exclude address 0 (pending token placeholder) to avoid calling null.
        let target_addr = target_addr.filter(|&a| a != 0);
        let is_resolved = target_addr.is_some() || self.self_entry_label.is_some();

        // assembler.py:324-336 call_assembler: select done_descr by op.type.
        let result_type = op.opcode.result_type();
        let done_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
        let helper_addr = crate::call_assembler_helper_addr() as i64;
        let green_key = self.header_pc as i64;

        if !is_resolved {
            // Pending/unresolved target: code not yet compiled.
            // RPython parity: call_assembler_fast_path (compiler.rs:2430)
            // detects null code_ptr and calls force_fn(inputs[0]) where
            // inputs[0] = the callee's frame pointer (a PyFrame).
            //
            // RPython uses the first argument slot of the callee jitframe.
            // Read it, free the heap jf, then call force_fn(frame_ptr).
            let force_addr = crate::call_assembler_force_fn_addr() as i64;
            if force_addr != 0 {
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x21, [x20, Self::slot_offset(JITFRAME_FIXED_SIZE) as u32]
                );
                self.emit_mov_imm64(2, free_ptr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x0, x20
                    ; blr x2                        // free(heap_jf)
                    ; mov x29, x19                  // restore caller jf_ptr
                    ; mov x0, x21                   // arg0 = PyFrame ptr
                );
                self.emit_mov_imm64(2, force_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; blr x2                        // x0 = force_fn(frame_ptr)
                );
                self.reload_frame_if_necessary();
            } else {
                self.emit_mov_imm64(2, free_ptr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x0, x20
                    ; blr x2
                    ; mov x29, x19
                    ; mov x0, 0
                );
            }
        } else {
            // Resolved target: call callee via execute trampoline
            // (stacker stack-growth protection for deep CALL_ASSEMBLER recursion).
            let trampoline_addr = crate::call_assembler_execute_addr() as i64;
            if let Some(addr) = target_addr {
                let addr = addr as i64;
                self.emit_mov_imm64(1, addr); // x1 = callee entry addr
                self.emit_mov_imm64(2, trampoline_addr);
                dynasm!(self.mc ; .arch aarch64 ; blr x2);
            } else if self.self_entry_label.is_some() {
                // Self-entry: load entry addr from self_entry_addr_ptr
                // (written after finalization).
                let addr_ptr = self.self_entry_addr_ptr as i64;
                self.emit_mov_imm64(1, addr_ptr); // x1 = &entry_addr
                dynasm!(self.mc ; .arch aarch64
                    ; ldr x1, [x1]                // x1 = entry_addr
                );
                self.emit_mov_imm64(2, trampoline_addr);
                dynasm!(self.mc ; .arch aarch64 ; blr x2);
            }

            // rax/x0 = callee's returned jf_ptr (= heap jf_ptr we passed).

            // Restore caller's jf_ptr.
            dynasm!(self.mc ; .arch aarch64
                ; mov x29, x19            // restore
            );
            self.reload_frame_if_necessary();

            // rax = callee's returned jf_ptr (heap-allocated).
            // Save it in rdx for descr check and free.
            dynasm!(self.mc ; .arch aarch64
                ; mov x20, x0             // x20 = callee jf_ptr
            );

            // _call_assembler_check_descr (assembler.py:2274-2278):
            //   CMP [jf_ptr + jf_descr_ofs], done_with_this_frame_descr_{type}
            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            self.emit_mov_imm64(2, done_descr_ptr);
            dynasm!(self.mc ; .arch aarch64
                ; ldr x1, [x20, JF_DESCR_OFS as u32]
                ; cmp x1, x2
                ; b.eq =>fast_path
            );
            // Path A (slow)
            // `compile.py:665` parity: pass `cpu_ptr` as arg0 so the
            // trampoline resolves the attached
            // `done_with_this_frame_descr_*` / `exit_frame_with_exception_descr_ref`
            // identities through the owning backend instance.
            {
                let cpu_ptr = self.cpu_handle_ptr();
                self.emit_mov_imm64(3, helper_addr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x1, x20                   // arg1 = callee_jf_ptr
                );
                self.emit_mov_imm64(2, green_key); // arg2 = green_key
                self.emit_mov_imm64(0, cpu_ptr); // arg0 = cpu_handle
                dynasm!(self.mc ; .arch aarch64
                    ; blr x3                        // x0 = helper result
                );
                self.reload_frame_if_necessary();
                dynasm!(self.mc ; .arch aarch64 ; b =>merge);
            }
            // Path B (fast)
            {
                dynasm!(self.mc ; .arch aarch64
                    ; =>fast_path
                    ; ldr x19, [x20, FIRST_ITEM_OFFSET as u32]
                );
                self.emit_mov_imm64(2, free_ptr);
                dynasm!(self.mc ; .arch aarch64
                    ; mov x0, x20
                    ; blr x2
                    ; mov x0, x19                   // x0 = result
                    ; =>merge
                );
            }
        } // end if is_resolved

        // Store result to the output slot (rax/x0 holds result).
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }

        // Restore callee-saved regs clobbered by this sequence.
        dynasm!(self.mc ; .arch aarch64
            ; ldp x19, x20, [sp], #16    // restore x19/x20
        );
    }

    // ----------------------------------------------------------------
    // genop_* — allocation
    // x86/assembler.py:2338 genop_new etc.
    // These require GC runtime support. Emit trap for now.
    // ----------------------------------------------------------------

    /// NEW: allocate a fixed-size object. Requires GC runtime.
    /// Emits a trap (UD2/BRK) until GC nursery allocation is wired.
    /// Zero-initialize `size` bytes at address in x0 (preserved).
    /// Avoids clobbering callee-saved registers (x19-x28).
    fn inline_memzero(&mut self, size: i64) {
        let mut ofs = 0i32;
        while ofs + 16 <= size as i32 {
            dynasm!(self.mc ; .arch aarch64 ; stp xzr, xzr, [x0, ofs]);
            ofs += 16;
        }
        if ofs + 8 <= size as i32 {
            let u = ofs as u32;
            dynasm!(self.mc ; .arch aarch64 ; str xzr, [x0, u]);
            ofs += 8;
        }
        if ofs + 4 <= size as i32 {
            let u = ofs as u32;
            dynasm!(self.mc ; .arch aarch64 ; str wzr, [x0, u]);
        }
    }

    /// aarch64/opassembler.py:912 _write_barrier_fastpath parity.
    fn emit_write_barrier_fastpath(&mut self, op: &Op, arglocs: &[Loc]) {
        let wb = match crate::runner::DYNASM_ACTIVE_GC.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|gc| gc.get_write_barrier_descr())
        }) {
            Some(Some(wb)) => wb,
            _ => return,
        };

        let loc_base = match arglocs.first() {
            Some(Loc::Reg(r)) => *r,
            _ => return,
        };
        let is_array = op.opcode == majit_ir::OpCode::CondCallGcWbArray;
        let card_marking = is_array && wb.jit_wb_cards_set != 0;

        // opassembler.py:922-929: build mask
        let mut mask = wb.jit_wb_if_flag_singlebyte as i64;
        if card_marking {
            mask |= wb.jit_wb_cards_set_singlebyte as i64;
        }
        mask &= 0xFF;

        // opassembler.py:934: LDRB ip0, [base, wb_byteofs]
        let byteofs = wb.jit_wb_if_flag_byteofs;
        self.emit_ldrb_signed_offset(&loc_base, byteofs);
        // opassembler.py:936-937: TST ip0, mask
        dynasm!(self.mc ; .arch aarch64
            ; mov w17, mask as u32
            ; tst w16, w17
        );

        // opassembler.py:938-939: BEQ done (flag not set → skip)
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.eq =>done);

        if card_marking {
            // opassembler.py:943-949: test GCFLAG_CARDS_SET
            let cards_mask = (wb.jit_wb_cards_set_singlebyte as u8) as u32;
            dynasm!(self.mc ; .arch aarch64
                ; mov w17, cards_mask
                ; tst w16, w17
            );
            let card_mark = self.mc.new_dynamic_label();
            dynasm!(self.mc ; .arch aarch64 ; b.ne =>card_mark);

            // opassembler.py:953-976: array-specific helper call
            self.emit_wb_helper_call(
                loc_base,
                crate::runner::dynasm_write_barrier_from_array as *const () as i64,
            );

            // opassembler.py:982-987: re-check CARDS_SET after helper
            self.emit_ldrb_signed_offset(&loc_base, byteofs);
            dynasm!(self.mc ; .arch aarch64
                ; mov w17, cards_mask
                ; tst w16, w17
                ; b.eq =>done
            );

            // opassembler.py:996-1015: card marking inline
            dynasm!(self.mc ; .arch aarch64 ; =>card_mark);
            if let Some(Loc::Reg(loc_index)) = arglocs.get(1) {
                let shift = 3 + wb.jit_wb_card_page_shift;
                dynasm!(self.mc ; .arch aarch64
                    ; lsr x16, X(loc_index.value), shift
                    ; mvn x30, x16
                    ; lsr x16, X(loc_index.value), wb.jit_wb_card_page_shift
                    ; and x17, x16, 7
                    ; mov x16, 1
                    ; lsl x17, x16, x17
                    ; ldrb w16, [X(loc_base.value), x30]
                    ; orr w16, w16, w17
                    ; strb w16, [X(loc_base.value), x30]
                );
            }
        } else {
            // opassembler.py:968-976: non-array slow path
            self.emit_wb_helper_call(
                loc_base,
                crate::runner::dynasm_write_barrier as *const () as i64,
            );
        }

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// Load byte at [base + signed_byteofs] into w16 (ip0).
    fn emit_ldrb_signed_offset(&mut self, base: &crate::regloc::RegLoc, byteofs: i32) {
        if byteofs >= 0 && byteofs < 4096 {
            let ofs = byteofs as u32;
            dynasm!(self.mc ; .arch aarch64
                ; ldrb W(16), [X(base.value), ofs]
            );
        } else if byteofs >= -256 && byteofs < 0 {
            dynasm!(self.mc ; .arch aarch64
                ; ldurb W(16), [X(base.value), byteofs]
            );
        } else {
            // Large negative offset: compute address in x16, then load
            self.emit_mov_imm64(16, byteofs as i64);
            dynasm!(self.mc ; .arch aarch64
                ; add x16, X(base.value), x16
                ; ldrb W(16), [x16]
            );
        }
    }

    /// _build_wb_slowpath parity: save all GPR + VFP regs, call helper, restore.
    /// RPython: _push_all_regs_to_jitframe(also_push_vpf=True) + BL + _pop_all
    /// opassembler.py:956-960: helper_num variant depends on live VFP bindings.
    fn emit_wb_helper_call(&mut self, loc_base: crate::regloc::RegLoc, helper: i64) {
        // _push_all_regs_to_jitframe: save x0-x15 (GPR) + d0-d15 (VFP)
        // Total: 128 bytes GPR + 128 bytes VFP = 256 bytes
        dynasm!(self.mc ; .arch aarch64
            ; stp x0, x1, [sp, -256]!
            ; stp x2, x3, [sp, 16]
            ; stp x4, x5, [sp, 32]
            ; stp x6, x7, [sp, 48]
            ; stp x8, x9, [sp, 64]
            ; stp x10, x11, [sp, 80]
            ; stp x12, x13, [sp, 96]
            ; stp x14, x15, [sp, 112]
            ; stp d0, d1, [sp, 128]
            ; stp d2, d3, [sp, 144]
            ; stp d4, d5, [sp, 160]
            ; stp d6, d7, [sp, 176]
            ; stp d8, d9, [sp, 192]
            ; stp d10, d11, [sp, 208]
            ; stp d12, d13, [sp, 224]
            ; stp d14, d15, [sp, 240]
        );
        if loc_base.value != 0 {
            dynasm!(self.mc ; .arch aarch64 ; mov x0, X(loc_base.value));
        }
        self.emit_mov_imm64(2, helper);
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        // _pop_all_regs_from_jitframe: restore VFP then GPR
        dynasm!(self.mc ; .arch aarch64
            ; ldp d14, d15, [sp, 240]
            ; ldp d12, d13, [sp, 224]
            ; ldp d10, d11, [sp, 208]
            ; ldp d8, d9, [sp, 192]
            ; ldp d6, d7, [sp, 176]
            ; ldp d4, d5, [sp, 160]
            ; ldp d2, d3, [sp, 144]
            ; ldp d0, d1, [sp, 128]
            ; ldp x14, x15, [sp, 112]
            ; ldp x12, x13, [sp, 96]
            ; ldp x10, x11, [sp, 80]
            ; ldp x8, x9, [sp, 64]
            ; ldp x6, x7, [sp, 48]
            ; ldp x4, x5, [sp, 32]
            ; ldp x2, x3, [sp, 16]
            ; ldp x0, x1, [sp], 256
        );
    }

    /// `_build_malloc_slowpath()` parity: preserve live state across the
    /// helper call. RPython's JIT-generated slowpath explicitly preserves all
    /// registers apart from the result convention registers; our Rust helper
    /// call must do the same at the call site.
    ///
    /// The nursery slowpath helpers communicate through `x0` (result) and may
    /// clobber `x1`, so we preserve `x2..x15` and `d0..d15` here.
    fn emit_malloc_slowpath_helper_call(&mut self, helper_reg: u8) {
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [sp, -240]!
            ; stp x4, x5, [sp, 16]
            ; stp x6, x7, [sp, 32]
            ; stp x8, x9, [sp, 48]
            ; stp x10, x11, [sp, 64]
            ; stp x12, x13, [sp, 80]
            ; stp x14, x15, [sp, 96]
            ; stp d0, d1, [sp, 112]
            ; stp d2, d3, [sp, 128]
            ; stp d4, d5, [sp, 144]
            ; stp d6, d7, [sp, 160]
            ; stp d8, d9, [sp, 176]
            ; stp d10, d11, [sp, 192]
            ; stp d12, d13, [sp, 208]
            ; stp d14, d15, [sp, 224]
            ; blr X(helper_reg)
            ; ldp d14, d15, [sp, 224]
            ; ldp d12, d13, [sp, 208]
            ; ldp d10, d11, [sp, 192]
            ; ldp d8, d9, [sp, 176]
            ; ldp d6, d7, [sp, 160]
            ; ldp d4, d5, [sp, 144]
            ; ldp d2, d3, [sp, 128]
            ; ldp d0, d1, [sp, 112]
            ; ldp x14, x15, [sp, 96]
            ; ldp x12, x13, [sp, 80]
            ; ldp x10, x11, [sp, 64]
            ; ldp x8, x9, [sp, 48]
            ; ldp x6, x7, [sp, 32]
            ; ldp x4, x5, [sp, 16]
            ; ldp x2, x3, [sp], 240
        );
    }

    /// aarch64/assembler.py:682 malloc_cond parity.
    ///
    /// Inline nursery bump allocation. total_size (from op.arg(0))
    /// includes GcHeader. Result in x0 = object pointer (after header).
    fn genop_call_malloc_nursery(&mut self, op: &Op) {
        let size_ref = op.arg(0);
        let total_size = self
            .constants
            .get(&size_ref.0)
            .copied()
            .unwrap_or(size_ref.0 as i64);
        let gc_hdr = majit_gc::header::GcHeader::SIZE as i64;
        // gc.py:525-531 — read nursery slot addresses from the active GC
        // descriptor (cpu.gc_ll_descr.get_nursery_free_addr() parity), not
        // from a process-global singleton.
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
        if nf_addr == 0 || nt_addr == 0 {
            self.emit_mov_imm64(0, total_size);
            self.emit_mov_imm64(
                2,
                crate::runner::dynasm_nursery_slowpath as *const () as i64,
            );
            self.emit_malloc_slowpath_helper_call(2);
            return;
        }

        // x0 = nursery_free, x1 = new_free, x2 = scratch, x3 = nursery_top
        self.emit_mov_imm64(2, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x2]);

        if total_size < 4096 {
            let ts = total_size as u32;
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, ts);
        } else {
            self.emit_mov_imm64(1, total_size);
            dynasm!(self.mc ; .arch aarch64 ; add x1, x0, x1);
        }

        self.emit_mov_imm64(3, nt_addr as i64);
        dynasm!(self.mc ; .arch aarch64 ; ldr x3, [x3]);
        dynasm!(self.mc ; .arch aarch64 ; cmp x1, x3);

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; b.hi =>slow_path);

        // Fast path: bump nursery_free, zero header, return payload ptr
        self.emit_mov_imm64(2, nf_addr as i64);
        dynasm!(self.mc ; .arch aarch64
            ; str x1, [x2]       // *nursery_free = new_free
            ; str xzr, [x0]      // zero GcHeader
        );
        let hs = gc_hdr as u32;
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, hs     // x0 = payload pointer (after header)
            ; b =>done
        );

        // Slow path: aarch64/assembler.py:605-676 `_build_malloc_slowpath` parity.
        //
        // Before calling `dynasm_nursery_slowpath` (which may trigger a
        // minor collection) we must spill every managed register holding a
        // live Ref into its canonical jitframe slot. `gcmap` identifies
        // those slots by the same register-index table that
        // `get_gcmap` writes bits from (`core_reg_index`:
        //   x0..x13 → slots 0..13, x19 → 14, x20 → 15).
        //
        // Without this spill the GC walks `jf_frame` slots 0..15 (per
        // gcmap bits), finds garbage, and returns with live Refs in CPU
        // registers left pointing into the pre-collection nursery. A
        // subsequent guard then captures those stale pointers into its
        // deadframe and the blackhole decoder crashes when reading the
        // freed object (observed in fib_recursive: `bh_binary_op_fn` gets
        // an int object with `ob_type=NULL` after minor GC).
        //
        // Matches RPython's `_push_all_regs_to_jitframe(mc, [r.x0, r.x1], True)`
        // before the `BL` and `_pop_all_regs_from_jitframe(mc, [r.x0, r.x1], ...)` after.
        // x0 is the total_size argument / return pointer; x1 is an internal
        // temp that regalloc already guarantees doesn't hold a live Ref at
        // the malloc site (see `MALLOC_NURSERY_CLOBBER`).
        dynasm!(self.mc ; .arch aarch64 ; =>slow_path);
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        // Save x2..x13 to slots 2..13 (x0/x1 excluded per ignored_regs).
        dynasm!(self.mc ; .arch aarch64
            ; stp x2, x3, [x29, base_ofs + 2 * 8]
            ; stp x4, x5, [x29, base_ofs + 4 * 8]
            ; stp x6, x7, [x29, base_ofs + 6 * 8]
            ; stp x8, x9, [x29, base_ofs + 8 * 8]
            ; stp x10, x11, [x29, base_ofs + 10 * 8]
            ; stp x12, x13, [x29, base_ofs + 12 * 8]
            ; stp x19, x20, [x29, base_ofs + 14 * 8]
        );
        // assembler.py:649-650: store gcmap to jf_gcmap so the collector
        // can trace live Refs pinned to frame slots during the slow-path
        // allocator call. The gcmap was captured at regalloc time in
        // `perform_with_gcmap` and threaded via RegAllocOp::Perform.gcmap.
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
            dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        }
        self.emit_mov_imm64(0, total_size);
        self.emit_mov_imm64(
            2,
            crate::runner::dynasm_nursery_slowpath as *const () as i64,
        );
        self.emit_malloc_slowpath_helper_call(2);
        // pop_gcmap: clear jf_gcmap after collecting call
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS as u32;
        dynasm!(self.mc ; .arch aarch64 ; str xzr, [x29, gcmap_ofs]);
        // Restore x2..x13, x19, x20 from jitframe slots (GC may have
        // updated the stored pointers). x0 keeps the allocated payload ptr.
        let base_ofs_r = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        dynasm!(self.mc ; .arch aarch64
            ; ldp x2, x3, [x29, base_ofs_r + 2 * 8]
            ; ldp x4, x5, [x29, base_ofs_r + 4 * 8]
            ; ldp x6, x7, [x29, base_ofs_r + 6 * 8]
            ; ldp x8, x9, [x29, base_ofs_r + 8 * 8]
            ; ldp x10, x11, [x29, base_ofs_r + 10 * 8]
            ; ldp x12, x13, [x29, base_ofs_r + 12 * 8]
            ; ldp x19, x20, [x29, base_ofs_r + 14 * 8]
        );

        dynasm!(self.mc ; .arch aarch64 ; =>done);
    }

    /// aarch64/assembler.py:682 malloc_cond parity.
    /// Inline nursery bump allocation for NEW.
    fn genop_new(&mut self, op: &Op) {
        let obj_size = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.size())
            .unwrap_or(16) as i64;
        self.emit_mov_imm64(0, obj_size);
        self.emit_mov_imm64(2, libc::malloc as *const () as i64);
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        self.inline_memzero(obj_size);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    fn genop_new_with_vtable(&mut self, op: &Op) {
        let obj_size = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.size())
            .unwrap_or(16) as i64;
        let vtable = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.vtable())
            .unwrap_or(0) as i64;
        self.emit_mov_imm64(0, obj_size);
        self.emit_mov_imm64(2, libc::malloc as *const () as i64);
        dynasm!(self.mc ; .arch aarch64 ; blr x2);
        self.inline_memzero(obj_size);
        if vtable != 0 {
            self.emit_mov_imm64(1, vtable);
            dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]);
        }
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate an array.
    fn genop_new_array(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    // ----------------------------------------------------------------
    // genop_* — misc
    // ----------------------------------------------------------------

    /// FORCE_TOKEN: return the jitframe pointer itself.
    /// x86/assembler.py genop_force_token: mov resloc, ebp
    fn genop_force_token(&mut self, op: &Op) {
        // The frame pointer is the force token.
        dynasm!(self.mc ; .arch aarch64
            ; mov x0, x29
        );
        self.store_rax_to_result(op.pos);
    }

    /// STRLEN / UNICODELEN: result = string.length
    /// Load the length field from the string/unicode object header.
    /// arg0 = string pointer. The length is at a fixed offset in the
    /// RPython string representation. For RPython strings, the length
    /// is typically at offset 8 (after the GC header / hash field).
    fn genop_strlen(&mut self, op: &Op) {
        // rewrite.py:273-282 parity — the length field offset comes from
        // `get_array_token(rstr.STR/UNICODE, ...)` → `ArrayDescr.lendescr`
        // in the JIT descr model.  Upstream emits `GC_LOAD_I` reading
        // `ArrayDescr.lendescr.offset`; the direct path mirrors that.
        let offset = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .and_then(|ad| ad.len_descr())
            .map(|ld| ld.offset() as i32)
            .unwrap_or_else(|| Self::field_offset_from_descr(op));
        self.load_arg_to_rax(op.arg(0));

        dynasm!(self.mc ; .arch aarch64
            ; ldr x0, [x0, offset as u32]
        );

        self.store_rax_to_result(op.pos);
    }

    /// STRGETITEM / UNICODEGETITEM: result = string[index]
    /// arg0 = string pointer, arg1 = index.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:295-306` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1.
    fn genop_strgetitem(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1)); // rstr.STR token defaults (basesize=17, itemsize=1)
        if op.opcode == OpCode::Strgetitem {
            debug_assert_eq!(item_size, 1, "STRGETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:299 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0)); // string pointer
        self.load_arg_to_rcx(op.arg(1)); // index

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64
                ; mul x1, x1, x2
            );
        }
        if base_size != 0 {
            dynasm!(self.mc ; .arch aarch64
                ; add x0, x0, base_size as u32
            );
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, x1
        );
        match item_size {
            1 => dynasm!(self.mc ; .arch aarch64
                ; ldrb w0, [x0]
            ),
            2 => dynasm!(self.mc ; .arch aarch64
                ; ldrh w0, [x0]
            ),
            4 => dynasm!(self.mc ; .arch aarch64
                ; ldr w0, [x0]
            ),
            _ => dynasm!(self.mc ; .arch aarch64
                ; ldr x0, [x0]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    // ================================================================
    // assembler.py:1817 genop_save_exc_class / genop_save_exception
    // ================================================================

    /// assembler.py:1817 genop_save_exc_class — stub: returns 0.
    fn genop_save_exc_class(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// assembler.py:1827 genop_save_exception — stub: returns 0.
    fn genop_save_exception(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch aarch64 ; mov x0, 0);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    // ================================================================
    // genop_* — extended integer arithmetic
    // ================================================================

    /// INT_FLOORDIV: result = arg0 / arg1 (signed)
    fn genop_int_floordiv(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MOD: result = arg0 % arg1 (signed)
    fn genop_int_mod(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; sdiv x2, x0, x1
            ; msub x0, x2, x1, x0
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64
            ; umulh x0, x0, x1
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_SIGNEXT: sign-extend from num_bytes width to 64 bits.
    fn genop_int_signext(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let num_bytes = match self.resolve_opref(op.arg(1)) {
            ResolvedArg::Const(v) => v,
            _ => 8,
        };
        let shift = 64 - num_bytes * 8;
        if shift > 0 && shift < 64 {
            let sh32 = shift as u32;
            dynasm!(self.mc ; .arch aarch64
                ; lsl x0, x0, sh32
                ; asr x0, x0, sh32
            );
        }
        self.store_rax_to_result(op.pos);
    }

    // ================================================================
    // genop_* — extended float operations
    // ================================================================

    /// FLOAT_ABS: result = |arg0|
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; fabs d0, d0
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_LT/LE/EQ/NE/GT/GE: float comparison.
    /// For lt/le, swap operands so JA/JAE handles NaN correctly.
    fn genop_float_cmp(&mut self, op: &Op) {
        let swap = matches!(op.opcode, OpCode::FloatLt | OpCode::FloatLe);
        if swap {
            self.load_float_arg_to_d0(op.arg(1));
            self.load_float_arg_to_d1(op.arg(0));
        } else {
            self.load_float_arg_to_d0(op.arg(0));
            self.load_float_arg_to_d1(op.arg(1));
        }

        dynasm!(self.mc ; .arch aarch64 ; fcmp d0, d1);
        match op.opcode {
            OpCode::FloatLt | OpCode::FloatGt => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, gt);
            }
            OpCode::FloatLe | OpCode::FloatGe => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, ge);
            }
            OpCode::FloatEq => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
            }
            OpCode::FloatNe => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, ne);
            }
            _ => {
                dynasm!(self.mc ; .arch aarch64 ; cset x0, eq);
            }
        }
        dynasm!(self.mc ; .arch aarch64 ; cmp x0, 0);
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// CAST_FLOAT_TO_SINGLEFLOAT: f64 → f32 (bits in lower 32 of i64)
    fn genop_cast_float_to_singlefloat(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; fcvt s0, d0
            ; fmov w0, s0
        );
        self.store_rax_to_result(op.pos);
    }

    /// CAST_SINGLEFLOAT_TO_FLOAT: f32 (bits in lower 32) → f64
    fn genop_cast_singlefloat_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch aarch64
            ; fmov s0, w0
            ; fcvt d0, s0
        );
        self.store_d0_to_result(op.pos);
    }

    // ================================================================
    // genop_* — GC memory operations
    // ================================================================

    /// Emit a sized load from [rax]/[x0]. Positive size = zero-extend,
    /// negative = sign-extend.
    fn emit_load_from_rax_sized(&mut self, itemsize: i32) {
        let abs_size = itemsize.unsigned_abs() as usize;
        let signed = itemsize < 0;
        match (abs_size, signed) {
            (1, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsb x0, [x0]),
            (2, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsh x0, [x0]),
            (4, true) => dynasm!(self.mc ; .arch aarch64 ; ldrsw x0, [x0]),
            (1, false) => dynasm!(self.mc ; .arch aarch64 ; ldrb w0, [x0]),
            (2, false) => dynasm!(self.mc ; .arch aarch64 ; ldrh w0, [x0]),
            (4, false) => dynasm!(self.mc ; .arch aarch64 ; ldr w0, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; ldr x0, [x0]),
        }
    }

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        match size {
            1 => dynasm!(self.mc ; .arch aarch64 ; strb w1, [x0]),
            2 => dynasm!(self.mc ; .arch aarch64 ; strh w1, [x0]),
            4 => dynasm!(self.mc ; .arch aarch64 ; str w1, [x0]),
            _ => dynasm!(self.mc ; .arch aarch64 ; str x1, [x0]),
        }
    }

    /// Resolve an OpRef that is expected to be a compile-time constant.
    fn resolve_const_or(&self, opref: OpRef, default: i64) -> i64 {
        match self.resolve_opref(opref) {
            ResolvedArg::Const(v) => v,
            _ => default,
        }
    }

    /// GC_LOAD_I/R/F: load from base + offset with given itemsize.
    /// arg(0) = base, arg(1) = offset, arg(2) = itemsize.
    fn genop_gc_load(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        let itemsize = self.resolve_const_or(op.arg(2), 8) as i32;
        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos);
    }

    /// GC_LOAD_INDEXED_I/R/F: load from base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=scale, arg(3)=base_offset, arg(4)=itemsize.
    fn genop_gc_load_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(2), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(3), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(4), 8) as i32;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        if scale != 1 {
            self.emit_mov_imm64(2, scale as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
        }

        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos);
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self.resolve_const_or(op.arg(3), 8).unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(5), 8).unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        if scale != 1 {
            self.emit_mov_imm64(2, scale as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, base_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// RAW_LOAD_I/F: load from base + offset using descriptor.
    fn genop_raw_load(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        self.emit_load_from_rax_sized(size as i32);
        let _ = offset; // offset is in the descriptor, not used for raw_load
        self.store_rax_to_result(op.pos);
    }

    /// RAW_STORE: store value to base + offset using descriptor.
    fn genop_discard_raw_store(&mut self, op: &Op) {
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(size);
    }

    // ================================================================
    // genop_* — interior field operations
    // ================================================================

    /// GETINTERIORFIELD_GC_I/R/F: load field from array-of-structs element.
    fn genop_getinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_interior_field_descr())
            .map(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        if total_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);

        self.emit_load_from_rax_sized(field_size as i32);
        self.store_rax_to_result(op.pos);
    }

    /// SETINTERIORFIELD_GC/RAW: write field in array-of-structs element.
    fn genop_discard_setinteriorfield(&mut self, op: &Op) {
        let (base_size, item_size, field_offset, field_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_interior_field_descr())
            .map(|id| {
                let ad = id.array_descr();
                let fd = id.field_descr();
                (
                    ad.base_size() as i32,
                    ad.item_size() as i32,
                    fd.offset() as i32,
                    fd.field_size(),
                )
            })
            .unwrap_or((8, 8, 0, 8));

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        if total_offset != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, total_offset as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);

        self.emit_store_to_rax_sized(field_size);
    }

    // ================================================================
    // genop_* — call variants
    // ================================================================

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    fn genop_discard_cond_call(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; cbz x0, =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    fn genop_cond_call_value(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch aarch64 ; cbnz x0, =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch aarch64 ; =>skip_label);

        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    // ================================================================
    // genop_* — string/array operations
    // ================================================================

    /// STRSETITEM / UNICODESETITEM: string[index] = value.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:307-318` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1; UNICODE does not.
    fn genop_discard_strsetitem(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1));
        if op.opcode == OpCode::Strsetitem {
            debug_assert_eq!(item_size, 1, "STRSETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:311 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0)); // string
        self.load_arg_to_rcx(op.arg(1)); // index
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size as i64);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
        );
        dynasm!(self.mc ; .arch aarch64 ; mov x2, x0);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch aarch64 ; mov x0, x2);
        self.emit_store_to_rax_sized(item_size as usize);
    }

    /// COPYSTRCONTENT / COPYUNICODECONTENT: copy substring.
    /// arg(0)=src, arg(1)=dst, arg(2)=src_start, arg(3)=dst_start, arg(4)=length.
    ///
    /// Fallback emitter for the no-rewriter path.  When the GC rewriter is
    /// present (production), `rewrite.py:1045-1080
    /// rewrite_copy_str_content` replaces this op with
    /// LOAD_EFFECTIVE_ADDRESS × 2 + CALL_N(memcpy, …) before assembly, so
    /// this function is never reached.  Kept for tests that run the
    /// backend directly on un-rewritten ops.  Mirrors upstream's basesize
    /// handling (`rewrite.py:1049-1053`).
    fn genop_discard_copystrcontent(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((16, 1));
        // rewrite.py:1049-1053 `rewrite_copy_str_content` — COPYSTRCONTENT
        // uses `str_descr.basesize - 1` to skip the `extra_item_after_alloc`
        // null terminator carried by `rstr.STR.chars` (`rstr.py:1226-1228`).
        // COPYUNICODECONTENT's unicode_descr has no extra item.  Mirrors the
        // same correction `strgetsetitem_token` applies for STR{GET,SET}ITEM.
        if op.opcode == OpCode::Copystrcontent {
            debug_assert_eq!(item_size, 1, "COPYSTRCONTENT itemsize must be 1");
            base_size -= 1;
        }

        // Compute byte_count = length * item_size
        self.load_arg_to_rax(op.arg(4));
        if item_size != 1 {
            self.emit_mov_imm64(1, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64 ; str x0, [sp, #-16]!); // byte_count

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(2));
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
            ; str x0, [sp, #-16]!  // src_addr
        );

        self.load_arg_to_rax(op.arg(1));
        self.load_arg_to_rcx(op.arg(3));
        if item_size != 1 {
            self.emit_mov_imm64(2, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
        );

        let memmove_ptr = libc::memmove as *const () as i64;
        dynasm!(self.mc ; .arch aarch64
            ; ldr x1, [sp], #16  // src
            ; ldr x2, [sp], #16  // count
        );
        // x0 = dst already
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
        self.emit_mov_imm64(3, memmove_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x3
            ; ldp x29, x30, [sp], #16
        );
    }

    /// NEWSTR: allocate a byte string of given length.
    /// `base_size` / `item_size` come from the injected ArrayDescr
    /// (`builtin_string_array_descr` in `runner.rs`), which encodes
    /// `get_array_token(rstr.STR, ...)` — basesize includes the +1
    /// extra_item_after_alloc null terminator.
    fn genop_newstr(&mut self, op: &Op) {
        let (base_size, item_size) = Self::array_token_from_descr(op, 16, 1);
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    /// NEWUNICODE: allocate a unicode string (4-byte chars).
    /// Basesize = 16 (no extra_item_after_alloc), itemsize = 4.
    fn genop_newunicode(&mut self, op: &Op) {
        let (base_size, item_size) = Self::array_token_from_descr(op, 16, 4);
        self.genop_alloc_varsize(op, base_size, item_size);
    }

    /// Read `(base_size, item_size)` from the injected ArrayDescr.
    /// Fallback used only when the descr is missing (should never happen
    /// for NEWSTR/NEWUNICODE after `inject_builtin_string_descrs`).
    fn array_token_from_descr(op: &Op, fallback_base: i64, fallback_item: i64) -> (i64, i64) {
        op.descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((fallback_base, fallback_item))
    }

    /// Shared implementation for NEWSTR / NEWUNICODE / NEW_ARRAY.
    /// Allocates base_size + length * item_size bytes, zero-fills,
    /// and writes length to the header.
    fn genop_alloc_varsize(&mut self, op: &Op, base_size: i64, item_size: i64) {
        // arg(0) = length
        self.load_arg_to_rax(op.arg(0));
        let malloc_ptr = libc::malloc as *const () as i64;
        let memset_ptr = libc::memset as *const () as i64;

        dynasm!(self.mc ; .arch aarch64
            ; str x0, [sp, #-16]!               // save length
        );
        if item_size != 1 {
            self.emit_mov_imm64(1, item_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; str x0, [sp, #-16]!               // save total_size
        );
        dynasm!(self.mc ; .arch aarch64 ; stp x29, x30, [sp, #-16]!);
        self.emit_mov_imm64(8, malloc_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
            ; ldr x2, [sp], #16                 // total_size
            ; str x0, [sp, #-16]!               // save ptr on stack
            ; mov x1, 0                          // val = 0
            ; stp x29, x30, [sp, #-16]!
        );
        self.emit_mov_imm64(8, memset_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
            ; ldr x0, [sp], #16                  // restore ptr from stack
            ; ldr x1, [sp], #16                  // length
            ; str x1, [x0, 8]                    // store length at offset 8
        );

        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// ZERO_ARRAY: zero a range in an array.
    /// arg(0)=base, arg(1)=start, arg(2)=size, arg(3)=scale_start, arg(4)=scale_size.
    fn genop_discard_zero_array(&mut self, op: &Op) {
        let (base_size, _) = op
            .descr
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .map(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));

        let scale_start = self.resolve_const_or(op.arg(3), 1);
        let scale_size = self.resolve_const_or(op.arg(4), 1);
        let memset_ptr = libc::memset as *const () as i64;

        // byte_offset = base_size + start * scale_start
        // byte_length = size * scale_size
        self.load_arg_to_rax(op.arg(0)); // base
        self.load_arg_to_rcx(op.arg(1)); // start

        if scale_start != 1 {
            self.emit_mov_imm64(2, scale_start);
            dynasm!(self.mc ; .arch aarch64 ; mul x1, x1, x2);
        }
        dynasm!(self.mc ; .arch aarch64
            ; add x0, x0, base_size as u32
            ; add x0, x0, x1
            ; str x0, [sp, #-16]!               // save dest
        );
        self.load_arg_to_rax(op.arg(2));
        if scale_size != 1 {
            self.emit_mov_imm64(1, scale_size);
            dynasm!(self.mc ; .arch aarch64 ; mul x0, x0, x1);
        }
        dynasm!(self.mc ; .arch aarch64
            ; mov x2, x0                         // byte_length
            ; ldr x0, [sp], #16                  // dest
            ; mov x1, 0
            ; stp x29, x30, [sp, #-16]!
        );
        self.emit_mov_imm64(8, memset_ptr);
        dynasm!(self.mc ; .arch aarch64
            ; blr x8
            ; ldp x29, x30, [sp], #16
        );
    }

    // ================================================================
    // genop_* — address computation
    // ================================================================

    /// LOAD_EFFECTIVE_ADDRESS: result = base + (index << shift) + baseofs.
    /// resoperation.py:1052-1054 — `[v_gcptr, v_index, c_baseofs, c_shift]`.
    /// arg(0)=base, arg(1)=index, arg(2)=baseofs, arg(3)=shift.
    fn genop_load_effective_address(&mut self, op: &Op) {
        let baseofs = self.resolve_const_or(op.arg(2), 0) as i32;
        let shift = self.resolve_const_or(op.arg(3), 0) as i32;

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        if shift != 0 {
            // add x0, x0, x1, lsl #shift fuses << and + in one insn;
            // opencode via a scratch shift to keep store_rax_to_result in rax.
            dynasm!(self.mc ; .arch aarch64 ; lsl x1, x1, shift as u32);
        }
        dynasm!(self.mc ; .arch aarch64 ; add x0, x0, x1);
        if baseofs != 0 {
            dynasm!(self.mc ; .arch aarch64 ; add x0, x0, baseofs as u32);
        }

        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // Public: set constants from external source
    // ----------------------------------------------------------------

    /// Populate the constants map. Called by the frontend before assembly
    /// if constant OpRefs are used (OpRef.0 >= 10000).
    pub fn set_constants(&mut self, constants: HashMap<u32, i64>) {
        self.constants = constants;
    }

    /// Set constant type annotations for the next compile call.
    pub fn set_constant_types(&mut self, constant_types: HashMap<u32, majit_ir::Type>) {
        self.constant_types = constant_types;
    }
}

/// Flush icache — aarch64 only.
fn flush_icache(addr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        unsafe { sys_icache_invalidate(addr as *mut u8, len) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(addr as *mut u8, (addr as *mut u8).add(len)) };
    }
}
