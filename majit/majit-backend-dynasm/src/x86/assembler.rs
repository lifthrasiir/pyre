/// x86/assembler.py: Assembler386 — x86_64 JIT code generation backend.
///
/// RPython: Assembler386(BaseAssembler, VectorAssemblerMixin)
/// in x86/assembler.py.
///
/// Key methods:
///   assemble_loop — assembler.py:501
///   assemble_bridge — assembler.py:623
///   _assemble — assembler.py:779 (walk ops + emit code)
///   patch_jump_for_descr — assembler.py:965
///   redirect_call_assembler — assembler.py:1138
use std::collections::HashMap;
use std::sync::Arc;

// x86/assembler.py parity: x86_64-only backend.
use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use majit_backend::{BackendError, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout};
use majit_ir::{FailDescr, InputArg, LoopTargetDescr, Op, OpCode, OpRef, TargetArgLoc, Type};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
use crate::guard::DynasmFailDescr;
use crate::jitframe::{
    FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_OFS,
    JF_GCMAP_OFS, JF_GUARD_EXC_OFS,
};
use crate::regalloc::{RegAlloc, RegAllocOp};
use crate::regloc::Loc;
use crate::runner::GuardGcTypeInfo;

/// x86/assembler.py: managed general-purpose registers.
const X86_GEN_REGS: [crate::regloc::RegLoc; 16] = [
    crate::regloc::RegLoc::new(0, false),
    crate::regloc::RegLoc::new(1, false),
    crate::regloc::RegLoc::new(2, false),
    crate::regloc::RegLoc::new(3, false),
    crate::regloc::RegLoc::new(4, false),
    crate::regloc::RegLoc::new(5, false),
    crate::regloc::RegLoc::new(6, false),
    crate::regloc::RegLoc::new(7, false),
    crate::regloc::RegLoc::new(8, false),
    crate::regloc::RegLoc::new(9, false),
    crate::regloc::RegLoc::new(10, false),
    crate::regloc::RegLoc::new(11, false),
    crate::regloc::RegLoc::new(12, false),
    crate::regloc::RegLoc::new(13, false),
    crate::regloc::RegLoc::new(19, false),
    crate::regloc::RegLoc::new(20, false),
];

/// x86/assembler.py: managed XMM/float registers.
const X86_FLOAT_REGS: [crate::regloc::RegLoc; 8] = [
    crate::regloc::RegLoc::new(0, true),
    crate::regloc::RegLoc::new(1, true),
    crate::regloc::RegLoc::new(2, true),
    crate::regloc::RegLoc::new(3, true),
    crate::regloc::RegLoc::new(4, true),
    crate::regloc::RegLoc::new(5, true),
    crate::regloc::RegLoc::new(6, true),
    crate::regloc::RegLoc::new(7, true),
];

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
    use crate::regloc::*;
    &[
        ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
    ]
}

fn all_float_regs() -> &'static [crate::regloc::RegLoc] {
    use crate::regloc::*;
    &[
        XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
        XMM14,
    ]
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

/// assembler.py:47 Assembler386.
/// In Rust, this is a transient builder — created per compilation,
/// not a long-lived object like RPython's.
pub struct Assembler386 {
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
    /// `x86/assembler.py:1971-1974`.
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
    /// Offset in machine code where the guard's conditional jump
    /// was emitted. We'll patch this to point to the recovery stub.
    jump_offset: AssemblyOffset,
    /// Dynamic label that the guard's Jcc jumps to — bound in
    /// write_pending_failure_recoveries to the recovery stub.
    fail_label: DynamicLabel,
    /// The fail descriptor for this guard.
    fail_descr: Arc<DynasmFailDescr>,
    /// Fail argument OpRefs for recovery (to save to sequential output slots).
    fail_args: Vec<OpRef>,
    /// regalloc parity: snapshot of opref_to_slot at guard emission time.
    /// Needed by recovery stubs to read fail_args from correct slots.
    opref_to_slot_snapshot: HashMap<u32, usize>,
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
    /// from the in-progress `Assembler386.fail_descrs` Vec.
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

impl Assembler386 {
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
        Assembler386 {
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
    /// reads in `rpython/jit/backend/x86/assembler.py`.
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
    /// `OpCode::CheckMemoryError` (assembler.py:1630-1641
    /// `genop_discard_check_memory_error`).
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

    /// Allocate a frame slot for an OpRef and return the slot index.
    /// Reuses existing slot if the OpRef already has one.
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
        match (src, dst) {
            (Loc::Reg(s), Loc::Reg(d)) if s == d => {}
            (Loc::Reg(s), Loc::Reg(d)) => {
                if s.is_xmm && d.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(d.value), Rx(s.value));
                } else if !s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), Rq(s.value));
                } else if s.is_xmm && !d.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movq Rq(d.value), Rx(s.value));
                } else {
                    dynasm!(self.mc ; .arch x64 ; movq Rx(d.value), Rq(s.value));
                }
            }
            (Loc::Reg(s), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                if s.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd [rbp + ofs], Rx(s.value));
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov [rbp + ofs], Rq(s.value));
                }
            }
            (Loc::Frame(f), Loc::Reg(d)) => {
                let ofs = f.ebp_loc.value;
                if d.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(d.value), [rbp + ofs]);
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), [rbp + ofs]);
                }
            }
            (Loc::Immed(i), Loc::Reg(d)) => {
                if d.is_xmm {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD i.value
                        ; movq Rx(d.value), Rq(scratch)
                    );
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(d.value), QWORD i.value);
                }
            }
            (Loc::Immed(i), Loc::Frame(f)) => {
                let ofs = f.ebp_loc.value;
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64
                    ; mov Rq(scratch), QWORD i.value
                    ; mov [rbp + ofs], Rq(scratch)
                );
            }
            (Loc::Frame(f1), Loc::Frame(f2)) if f1.position == f2.position => {}
            (Loc::Frame(f1), Loc::Frame(f2)) => {
                let o1 = f1.ebp_loc.value;
                let o2 = f2.ebp_loc.value;
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64
                    ; mov Rq(scratch), [rbp + o1]
                    ; mov [rbp + o2], Rq(scratch)
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
                dynasm!(self.mc ; .arch x64 ; sub rsp, 8 ; movsd [rsp], Rx(r.value));
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch x64 ; push Rq(r.value));
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch x64 ; sub rsp, 8 ; movsd xmm15, [rbp + f.ebp_loc.value] ; movsd [rsp], xmm15);
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + f.ebp_loc.value]);
            }
            _ => {}
        }
    }

    fn regalloc_pop(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) if r.is_xmm => {
                dynasm!(self.mc ; .arch x64 ; movsd Rx(r.value), [rsp] ; add rsp, 8);
            }
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch x64 ; pop Rq(r.value));
            }
            Loc::Frame(f) if f.ebp_loc.is_float => {
                dynasm!(self.mc ; .arch x64 ; movsd xmm15, [rsp] ; add rsp, 8 ; movsd [rbp + f.ebp_loc.value], xmm15);
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch x64 ; pop QWORD [rbp + f.ebp_loc.value]);
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

    /// Emit: ADD/SUB/AND/OR/XOR reg, loc
    fn emit_binop_reg_loc(&mut self, opcode: OpCode, dst_reg: u8, src: &Loc) {
        // aarch64: load src to x16 scratch if not in register
        match src {
            Loc::Reg(s) => match opcode {
                OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                    dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntSub | OpCode::IntSubOvf => {
                    dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntMul | OpCode::IntMulOvf => {
                    dynasm!(self.mc ; .arch x64 ; imul Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntAnd => {
                    dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntOr => {
                    dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), Rq(s.value));
                }
                OpCode::IntXor => {
                    dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), Rq(s.value));
                }
                _ => {}
            },
            Loc::Frame(f) => {
                let ofs = f.ebp_loc.value;
                match opcode {
                    OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                        dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntSub | OpCode::IntSubOvf => {
                        dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntMul | OpCode::IntMulOvf => {
                        dynasm!(self.mc ; .arch x64 ; imul Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntAnd => {
                        dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntOr => {
                        dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), [rbp + ofs]);
                    }
                    OpCode::IntXor => {
                        dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), [rbp + ofs]);
                    }
                    _ => {}
                }
            }
            Loc::Immed(i) => {
                let v = i.value as i32;
                match opcode {
                    OpCode::IntAdd | OpCode::IntAddOvf | OpCode::NurseryPtrIncrement => {
                        dynasm!(self.mc ; .arch x64 ; add Rq(dst_reg), v);
                    }
                    OpCode::IntSub | OpCode::IntSubOvf => {
                        dynasm!(self.mc ; .arch x64 ; sub Rq(dst_reg), v);
                    }
                    OpCode::IntAnd => {
                        dynasm!(self.mc ; .arch x64 ; and Rq(dst_reg), v);
                    }
                    OpCode::IntOr => {
                        dynasm!(self.mc ; .arch x64 ; or  Rq(dst_reg), v);
                    }
                    OpCode::IntXor => {
                        dynasm!(self.mc ; .arch x64 ; xor Rq(dst_reg), v);
                    }
                    _ => {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), QWORD i.value ; imul Rq(dst_reg), Rq(scratch));
                    }
                }
            }
            _ => {}
        }
    }

    /// Emit: CMP loc0, loc1
    fn emit_cmp_loc_loc(&mut self, loc0: &Loc, loc1: &Loc) {
        match (loc0, loc1) {
            (Loc::Reg(r), Loc::Reg(s)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), Rq(s.value));
            }
            (Loc::Reg(r), Loc::Frame(f)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), [rbp + f.ebp_loc.value]);
            }
            (Loc::Reg(r), Loc::Immed(i)) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(r.value), i.value as i32);
            }
            (Loc::Frame(f), Loc::Reg(s)) => {
                dynasm!(self.mc ; .arch x64 ; cmp [rbp + f.ebp_loc.value], Rq(s.value));
            }
            (Loc::Frame(f), Loc::Immed(i)) => {
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + f.ebp_loc.value], i.value as i32);
            }
            _ => {
                self.regalloc_mov(loc0, &Loc::Reg(crate::regloc::X86_64_SCRATCH_REG));
                self.emit_cmp_loc_loc(&Loc::Reg(crate::regloc::X86_64_SCRATCH_REG), loc1);
            }
        }
    }

    /// Emit: TEST loc, loc (for guard_true/guard_false)
    fn emit_test_loc(&mut self, loc: &Loc) {
        match loc {
            Loc::Reg(r) => {
                dynasm!(self.mc ; .arch x64 ; test Rq(r.value), Rq(r.value));
            }
            Loc::Frame(f) => {
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + f.ebp_loc.value], 0);
            }
            Loc::Immed(i) => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), QWORD i.value ; test Rq(scratch), Rq(scratch));
            }
            _ => {}
        }
    }

    // ── AArch64 helper: load a 64-bit immediate into register Xn ──

    /// Emit: load the value of `opref` into RAX (x64) / X0 (aarch64).
    fn load_arg_to_rax(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, [rbp + offset]
                );
            }
            ResolvedArg::Const(val) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, QWORD val as i64
                );
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
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD imm.value as i64
                );
            }
            _ => self.regalloc_mov(&loc, &rax),
        }
    }

    /// Emit: load the value of `opref` into RCX (x64) / X1 (aarch64).
    fn load_arg_to_rcx(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rcx, [rbp + offset]
                );
            }
            ResolvedArg::Const(val) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rcx, QWORD val as i64
                );
            }
        }
    }

    /// Emit: store RAX/X0 to the frame slot for `result_opref`.
    /// Allocates a new slot if needed.
    fn store_rax_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        dynasm!(self.mc
            ; .arch x64
            ; mov [rbp + offset], rax
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
    /// assembler.py:1080-1091 `_call_header_with_stack_check`: inline
    /// SP probe at the very top of every JIT loop so deep compiled-to-
    /// compiled CALL_ASSEMBLER recursion surfaces a controlled
    /// RecursionError. On overflow, the body is skipped and the
    /// caller-provided jf_ptr is returned in RAX so the JIT glue
    /// drains the overflow flag on the way back to the interpreter.
    ///
    /// Inline probe (assembler.py:1085-1091 parity):
    /// ```text
    ///   MOV  rax, [endaddr]        ; rpy_stacktoobig.stack_end
    ///   SUB  rax, rsp              ; ofs = end - current_sp
    ///   CMP  rax, [lengthaddr]     ; vs rpy_stacktoobig.stack_length
    ///   JBE  continue              ; fast path: ofs <= length
    ///   MOV  rdi, rsp              ; arg0 = current sp
    ///   CALL pyre_stack_too_big_slowpath
    ///   TEST al, al
    ///   JZ   continue              ; slowpath: 0 = OK
    ///   ; fallthrough = real overflow → return rbp as jf_ptr
    /// ```
    fn _call_header(&mut self, inputargs: &[InputArg]) {
        dynasm!(self.mc
            ; .arch x64
            ; push rbp
            ; push r12               // save r12 (used by genop_call_assembler)
            ; mov rbp, rdi
        );
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr != 0 {
            if let Some(addrs) = crate::stack_check_addresses() {
                let continue_label = self.mc.new_dynamic_label();
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                let exc_value_addr = crate::jit_exc_value_addr() as i64;
                let exc_type_addr = crate::jit_exc_type_addr() as i64;
                dynasm!(self.mc
                    ; .arch x64
                    // Fast path: load end, subtract SP, compare with length.
                    ; mov Rq(scratch), QWORD addrs.end_adr as i64
                    ; mov rax, [Rq(scratch)]
                    ; sub rax, rsp
                    ; mov Rq(scratch), QWORD addrs.length_adr as i64
                    ; cmp rax, [Rq(scratch)]
                    ; jbe =>continue_label
                    // Slow path: call pyre_stack_too_big_slowpath(rsp).
                    ; mov rdi, rsp
                    ; mov Rq(scratch), QWORD addrs.slowpath_addr as i64
                    ; call Rq(scratch)
                    ; test al, al
                    ; jz =>continue_label
                    // x86/assembler.py:347-390 `_build_stack_check_slowpath`:
                    // the slowpath raised into pos_exception(); merge with the
                    // propagate-exception path by moving that value into
                    // jf_guard_exc and publishing propagate_exception_descr.
                    ; mov Rq(scratch), QWORD exc_value_addr
                    ; mov rax, [Rq(scratch)]
                    ; mov QWORD [Rq(scratch)], 0
                    ; mov [rbp + JF_GUARD_EXC_OFS], rax
                    ; mov Rq(scratch), QWORD exc_type_addr
                    ; mov QWORD [Rq(scratch)], 0
                    ; mov Rq(scratch), QWORD propagate_descr
                    ; mov [rbp + JF_DESCR_OFS], Rq(scratch)
                    // Overflow fallthrough: return rbp as jf_ptr.
                    ; mov rax, rbp
                    ; pop r12
                    ; pop rbp
                    ; ret
                    ; =>continue_label
                );
            }
        }
        // When addresses are not registered (tests / early startup), no
        // stack check is emitted — assembler.py:1082-1083 parity.
        self.setup_input_state(inputargs);
    }

    // ----------------------------------------------------------------
    // assembler.py:2153 _call_footer — function epilogue
    // ----------------------------------------------------------------

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    fn _call_footer(&mut self) {
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, rbp
            ; pop r12                // restore r12
            ; pop rbp
            ; ret
        );
    }

    /// assembler.py:993 push_gcmap.
    fn push_gcmap(&mut self, gcmap: *mut usize) {
        let gcmap_ptr = gcmap as i64;
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD gcmap_ptr
            ; mov [rbp + JF_GCMAP_OFS], Rq(scratch)
        );
    }

    /// assembler.py:1000 pop_gcmap.
    fn pop_gcmap(&mut self) {
        dynasm!(self.mc ; .arch x64
            ; mov QWORD [rbp + JF_GCMAP_OFS], 0
        );
    }

    /// RPython `AbstractCallBuilder.emit`: CALL_ASSEMBLER is a collecting
    /// call, so the caller jitframe must publish the regalloc gcmap before
    /// entering the callee/helper and clear it only after reloading a possibly
    /// moved frame pointer.
    fn push_pending_call_gcmap(&mut self) -> bool {
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
            true
        } else {
            false
        }
    }

    fn pop_pending_call_gcmap_after_collect(&mut self, pushed: bool) {
        self.reload_frame_if_necessary();
        if pushed {
            self.pop_gcmap();
        }
    }

    /// assembler.py:405-412 _reload_frame_if_necessary parity:
    /// after a helper or collecting call, follow jf_forward so rbp/x29
    /// points at the current jitframe location.
    fn reload_frame_if_necessary(&mut self) {
        let loop_label = self.mc.new_dynamic_label();
        let done_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64
            ; =>loop_label
            ; mov rcx, [rbp + JF_FORWARD_OFS]
            ; test rcx, rcx
            ; je =>done_label
            ; mov rbp, rcx
            ; jmp =>loop_label
            ; =>done_label
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
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
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
                        eprintln!("[dynasm] discard[{}]: {:?}", op_index, op.opcode);
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
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntSubOvf => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            OpCode::IntMulOvf => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    self.guard_success_cc = Some(CC_NO);
                }
            }
            // ── Integer binary (result_loc == arglocs[0], guaranteed by regalloc) ──
            OpCode::IntAdd
            | OpCode::IntSub
            | OpCode::IntMul
            | OpCode::IntAnd
            | OpCode::IntOr
            | OpCode::IntXor
            | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(src)) = (result_loc, arglocs.get(1)) {
                    self.emit_binop_reg_loc(op.opcode, dst.value, src);
                }
            }
            // ── Unary integer (result in arglocs[0] register) ──
            OpCode::IntNeg => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch x64 ; neg Rq(r.value));
                }
            }
            OpCode::IntInvert => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch x64 ; not Rq(r.value));
                }
            }
            // ── Shifts ──
            OpCode::IntLshift | OpCode::IntRshift | OpCode::UintRshift => {
                if let (Some(Loc::Reg(dst)), Some(shift_loc)) = (result_loc, arglocs.get(1)) {
                    match shift_loc {
                        Loc::Immed(i) => {
                            let sh = i.value as i8;
                            match op.opcode {
                                OpCode::IntLshift => {
                                    dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), sh);
                                }
                                OpCode::IntRshift => {
                                    dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), sh);
                                }
                                OpCode::UintRshift => {
                                    dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), sh);
                                }
                                _ => {}
                            }
                        }
                        Loc::Reg(s) if s.value == 1 => match op.opcode {
                            OpCode::IntLshift => {
                                dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), cl);
                            }
                            OpCode::IntRshift => {
                                dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), cl);
                            }
                            OpCode::UintRshift => {
                                dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), cl);
                            }
                            _ => {}
                        },
                        _ => {
                            self.regalloc_mov(shift_loc, &Loc::Reg(crate::regloc::ECX));
                            match op.opcode {
                                OpCode::IntLshift => {
                                    dynasm!(self.mc ; .arch x64 ; shl Rq(dst.value), cl);
                                }
                                OpCode::IntRshift => {
                                    dynasm!(self.mc ; .arch x64 ; sar Rq(dst.value), cl);
                                }
                                OpCode::UintRshift => {
                                    dynasm!(self.mc ; .arch x64 ; shr Rq(dst.value), cl);
                                }
                                _ => {}
                            }
                        }
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
                    if let Some(src) = arglocs.first() {
                        match src {
                            Loc::Reg(s) => {
                                dynasm!(self.mc ; .arch x64 ; mul Rq(s.value));
                            }
                            Loc::Frame(f) => {
                                dynasm!(self.mc ; .arch x64 ; mul QWORD [rbp + f.ebp_loc.value]);
                            }
                            Loc::Immed(i) => {
                                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                                dynasm!(self.mc ; .arch x64
                                    ; mov Rq(scratch), QWORD i.value
                                    ; mul Rq(scratch)
                                );
                            }
                            _ => {}
                        }
                        if dst.value != crate::regloc::EDX.value {
                            dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), rdx);
                        }
                    }
                }
            }
            OpCode::IntForceGeZero => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch x64
                        ; test Rq(r.value), Rq(r.value)
                        ; jge >pos
                        ; xor Rq(r.value), Rq(r.value)
                        ; pos:
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
                if let Some(Loc::Reg(dst)) = result_loc {
                    // Ensure second arg is in an XMM register
                    let src_reg = if let Some(Loc::Reg(s)) = arglocs.get(1) {
                        *s
                    } else if let Some(src_loc) = arglocs.get(1) {
                        // Immed or Frame — load to scratch XMM (d14/xmm14)
                        let scratch = crate::regloc::RegLoc::new(14, true);
                        self.regalloc_mov(src_loc, &Loc::Reg(scratch));
                        scratch
                    } else {
                        return; // shouldn't happen
                    };
                    match op.opcode {
                        OpCode::FloatAdd => {
                            dynasm!(self.mc ; .arch x64 ; addsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatSub => {
                            dynasm!(self.mc ; .arch x64 ; subsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatMul => {
                            dynasm!(self.mc ; .arch x64 ; mulsd Rx(dst.value), Rx(src_reg.value));
                        }
                        OpCode::FloatTrueDiv => {
                            dynasm!(self.mc ; .arch x64 ; divsd Rx(dst.value), Rx(src_reg.value));
                        }
                        _ => {}
                    }
                }
            }
            OpCode::FloatNeg => {
                if let Some(Loc::Reg(r)) = result_loc {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; xorpd Rx(scratch as u8), Rx(scratch as u8)
                        ; subsd Rx(scratch as u8), Rx(r.value)
                        ; movsd Rx(r.value), Rx(scratch as u8)
                    );
                }
            }
            OpCode::FloatAbs => {
                if let Some(Loc::Reg(r)) = result_loc {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD 0x7FFFFFFFFFFFFFFF_u64 as i64
                        ; movq Rx(scratch as u8), Rq(scratch)
                        ; andpd Rx(r.value), Rx(scratch as u8)
                    );
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
                    dynasm!(self.mc ; .arch x64 ; ucomisd Rx(a.value), Rx(b.value));
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
                    dynasm!(self.mc ; .arch x64 ; cvtsi2sd Rx(dst.value), Rq(sr));
                }
            }
            OpCode::CastFloatToInt => {
                if let (Some(Loc::Reg(src)), Some(Loc::Reg(dst))) = (arglocs.first(), result_loc) {
                    dynasm!(self.mc ; .arch x64 ; cvttsd2si Rq(dst.value), Rx(src.value));
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
                        dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + ofs]);
                    } else {
                        match field_size {
                            1 => {
                                dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), BYTE [Rq(base.value) + ofs]);
                            }
                            2 => {
                                dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), WORD [Rq(base.value) + ofs]);
                            }
                            4 => {
                                dynasm!(self.mc ; .arch x64 ; movsxd Rq(dst.value), DWORD [Rq(base.value) + ofs]);
                            }
                            _ => {
                                dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + ofs]);
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
                                &Loc::Reg(crate::regloc::X86_64_SCRATCH_REG),
                            );
                            crate::regloc::X86_64_SCRATCH_REG.value
                        }
                    };
                    if item_size != 1 {
                        dynasm!(self.mc ; .arch x64 ; imul Rq(index_reg), Rq(index_reg), item_size);
                    }
                    if base_size != 0 {
                        dynasm!(self.mc ; .arch x64 ; lea rax, [Rq(base.value) + Rq(index_reg) + base_size]);
                    } else {
                        dynasm!(self.mc ; .arch x64 ; lea rax, [Rq(base.value) + Rq(index_reg)]);
                    }
                    if dst.is_xmm {
                        dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [rax]);
                    } else {
                        match item_size {
                            1 if signed => {
                                dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), BYTE [rax])
                            }
                            1 => dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), BYTE [rax]),
                            2 if signed => {
                                dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), WORD [rax])
                            }
                            2 => dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), WORD [rax]),
                            4 if signed => {
                                dynasm!(self.mc ; .arch x64 ; movsxd Rq(dst.value), DWORD [rax])
                            }
                            4 => dynasm!(self.mc ; .arch x64 ; mov Rd(dst.value), [rax]),
                            _ => dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [rax]),
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
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)]
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let (Some(Loc::Reg(base)), Some(ofs_loc)) = (arglocs.first(), arglocs.get(1)) {
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
            // ── GC store / raw store: opassembler.rs emit_op_gcstore_regalloc ──
            // arglocs = [value_loc, base_loc, ofs_loc, size_loc]
            OpCode::GcStore | OpCode::RawStore => {
                if arglocs.len() >= 3 {
                    if let (Some(Loc::Reg(val)), Some(Loc::Reg(base))) =
                        (arglocs.first(), arglocs.get(1))
                    {
                        let size = match arglocs.get(3) {
                            Some(Loc::Immed(i)) => i.value.unsigned_abs() as usize,
                            _ => 8,
                        };
                        self.emit_op_gcstore_regalloc(base, &arglocs[2], val, size);
                    }
                }
            }
            // ── x86/assembler.py:1753 genop_discard_gc_store_indexed ──
            // `base_loc, ofs_loc, value_loc, factor_loc, offset_loc, size_loc = arglocs`.
            // `dest_addr = AddressLoc(base_loc, ofs_loc, scale=get_scale(factor_loc.value), disp=offset_loc.value)`
            // emits `[base + ofs * 2**scale + disp]`. `load_supported_factors =
            // (1, 2, 4, 8)` (x86/runner.py:31), so the rewriter passes raw
            // byte strides in that set straight through here and the native
            // SIB scaled-index addressing does the multiply. Any other factor
            // must have been pre-scaled away in `cpu_simplify_scale`.
            OpCode::GcStoreIndexed => {
                if let (Some(Loc::Reg(base)), Some(Loc::Reg(ofs_reg)), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let factor = match arglocs.get(3) {
                        Some(Loc::Immed(i)) => i.value,
                        _ => 1,
                    };
                    let offset = match arglocs.get(4) {
                        Some(Loc::Immed(i)) => i.value as i32,
                        _ => 0,
                    };
                    let size = match arglocs.get(5) {
                        Some(Loc::Immed(i)) => i.value.unsigned_abs() as usize,
                        _ => 8,
                    };

                    // Dynasm's `*N` operand is a compile-time literal, so the
                    // runtime factor is dispatched to four parallel emitters.
                    // Each inner `emit` closure receives the ready-to-use
                    // (base, ofs, scale, disp) triple as explicit dynasm
                    // syntax. `factor == 1` drops the `*1` token because
                    // dynasm emits a tighter encoding without it.
                    // Immediate stores stage the value through
                    // `X86_64_SCRATCH_REG` (r11).  `r0`/rax is in the GPR
                    // allocation pool (x86/regalloc.rs:19), so using it as
                    // the scratch here would silently clobber `base.value`
                    // or `ofs_reg.value` whenever regalloc assigned them
                    // to EAX.  Upstream `save_into_mem` emits `MOV [mem],
                    // imm` directly (assembler.py:1671); dynasm-rs does
                    // not accept an immediate operand in the scaled-index
                    // `mov` template, so we stage through the dedicated
                    // non-allocatable scratch register instead.
                    macro_rules! emit_store_scaled {
                        ($scale:tt) => {{
                            match value_loc {
                                Loc::Reg(val) if val.is_xmm => {
                                    dynasm!(self.mc ; .arch x64
                                        ; movsd [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rx(val.value));
                                }
                                Loc::Reg(val) => match size {
                                    1 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rb(val.value)),
                                    2 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rw(val.value)),
                                    4 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rd(val.value)),
                                    _ => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rq(val.value)),
                                },
                                Loc::Immed(i) => {
                                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                                    dynasm!(self.mc ; .arch x64
                                        ; mov Rq(scratch), QWORD i.value);
                                    match size {
                                        1 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rb(scratch)),
                                        2 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rw(scratch)),
                                        4 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rd(scratch)),
                                        _ => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset], Rq(scratch)),
                                    }
                                }
                                _ => {}
                            }
                        }};
                    }
                    macro_rules! emit_store_unscaled {
                        () => {{
                            match value_loc {
                                Loc::Reg(val) if val.is_xmm => {
                                    dynasm!(self.mc ; .arch x64
                                        ; movsd [Rq(base.value) + Rq(ofs_reg.value) + offset], Rx(val.value));
                                }
                                Loc::Reg(val) => match size {
                                    1 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rb(val.value)),
                                    2 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rw(val.value)),
                                    4 => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rd(val.value)),
                                    _ => dynasm!(self.mc ; .arch x64
                                        ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rq(val.value)),
                                },
                                Loc::Immed(i) => {
                                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                                    dynasm!(self.mc ; .arch x64
                                        ; mov Rq(scratch), QWORD i.value);
                                    match size {
                                        1 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rb(scratch)),
                                        2 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rw(scratch)),
                                        4 => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rd(scratch)),
                                        _ => dynasm!(self.mc ; .arch x64
                                            ; mov [Rq(base.value) + Rq(ofs_reg.value) + offset], Rq(scratch)),
                                    }
                                }
                                _ => {}
                            }
                        }};
                    }
                    match factor {
                        1 => emit_store_unscaled!(),
                        2 => emit_store_scaled!(2),
                        4 => emit_store_scaled!(4),
                        8 => emit_store_scaled!(8),
                        other => panic!(
                            "x86 GcStoreIndexed: unsupported factor {other}; \
                             load_supported_factors = (1, 2, 4, 8)"
                        ),
                    }
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
                let mut src_locations1 = Vec::new();
                let mut dst_locations1 = Vec::new();
                let mut src_locations2 = Vec::new();
                let mut dst_locations2 = Vec::new();
                // x86/regalloc.py:1287: assert len(arglocs) == jump_op.numargs()
                // RPython enforces arity equality at regalloc time;
                // the assembler never sees surplus args.
                let remap_count = if target_arglocs.is_empty() {
                    arglocs.len()
                } else {
                    assert_eq!(
                        arglocs.len(),
                        target_arglocs.len(),
                        "JUMP args ({}) != target LABEL args ({})",
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
                    match src_loc {
                        Loc::Reg(r) if r.is_xmm => {
                            src_locations2.push(*src_loc);
                            dst_locations2.push(dst_loc);
                        }
                        Loc::Frame(f) if f.ebp_loc.is_float => {
                            src_locations2.push(*src_loc);
                            dst_locations2.push(dst_loc);
                        }
                        _ => {
                            src_locations1.push(*src_loc);
                            dst_locations1.push(dst_loc);
                        }
                    }
                }
                let tmpreg1 = Loc::Reg(crate::regloc::X86_64_SCRATCH_REG);
                let tmpreg2 = Loc::Reg(crate::regloc::XMM15);
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!(
                        "[dynasm] Jump remap: {} int src→dst, {} float src→dst",
                        src_locations1.len(),
                        src_locations2.len()
                    );
                    for (i, (s, d)) in src_locations1.iter().zip(dst_locations1.iter()).enumerate()
                    {
                        eprintln!("[dynasm]   int[{}]: {:?} → {:?}", i, s, d);
                    }
                }
                self.remap_frame_layout_mixed(
                    &src_locations1,
                    &dst_locations1,
                    tmpreg1,
                    &src_locations2,
                    &dst_locations2,
                    tmpreg2,
                );
                if let Some(label) = loop_target_id(op)
                    .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
                {
                    dynasm!(self.mc ; .arch x64 ; jmp =>label);
                } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
                    // External JUMP: direct JMP to target loop code.
                    // assembler.py:2461 mc.JMP(imm(target))
                    let addr = target as i64;
                    dynasm!(self.mc ; .arch x64
                        ; mov rax, QWORD addr
                        ; jmp rax
                    );
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
                // `pyjitpl.py:3238 compile_exit_frame_with_exception`.  The
                // runtime classifier (`runner.rs::find_descr_by_ptr`) then
                // dispatches into `jitexc.ExitFrameWithExceptionRef` rather
                // than `jitexc.DoneWithThisFrame*`.
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
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64
                    ; mov Rq(scratch), QWORD global_descr_ptr
                    ; mov [rbp + JF_DESCR_OFS], Rq(scratch)
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

                self._call_footer();
                self.fail_descrs.push(descr);
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let label_descr = loop_target_descr(op);
                if std::env::var_os("MAJIT_LOG").is_some() {
                    eprintln!("[dynasm] LABEL: new DynamicLabel({:?})", label);
                }
                dynasm!(self.mc ; =>label);
                if let Some(descr) = label_descr {
                    descr.set_target_arglocs(
                        arglocs
                            .iter()
                            .copied()
                            .map(target_argloc_from_loc)
                            .collect(),
                    );
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
            OpCode::CallMallocNursery => {
                self.genop_call_malloc_nursery(op, result_loc);
            }
            OpCode::CallMallocNurseryVarsizeFrame => {
                if let Some(Loc::Reg(sizeloc)) = arglocs.first() {
                    dynasm!(self.mc ; .arch x64 ; mov rdi, Rq(sizeloc.value as u8));
                }
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD crate::runner::dynasm_nursery_slowpath as *const () as i64
                    ; call rax
                );
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                self.reload_frame_if_necessary();
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
            }
            // x86/assembler.py:2567 malloc_cond_varsize parity
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
                // rdi = base_size, rsi = item_size, rdx = length
                dynasm!(self.mc ; .arch x64 ; mov rdi, QWORD base_size);
                dynasm!(self.mc ; .arch x64 ; mov rsi, QWORD itemsize);
                match arglocs.first() {
                    Some(Loc::Reg(len_r)) => {
                        dynasm!(self.mc ; .arch x64 ; mov rdx, Rq(len_r.value as u8));
                    }
                    Some(Loc::Immed(len_i)) => {
                        dynasm!(self.mc ; .arch x64 ; mov rdx, QWORD len_i.value);
                    }
                    _ => {
                        dynasm!(self.mc ; .arch x64 ; xor edx, edx);
                    }
                }
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD crate::runner::dynasm_nursery_slowpath_varsize as *const () as i64
                    ; call rax
                );
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
            }
            // x86/assembler.py:1630-1641 `genop_discard_check_memory_error`
            // — emit `TEST reg, reg` + `JNZ skip` and inline the
            // propagate path (`_build_propagate_exception_path`,
            // assembler.py:328-345) so a NULL return from a malloc
            // helper propagates as a MemoryError via
            // `self.cpu.propagate_exception_descr`.
            //
            // Upstream materializes `propagate_exception_path` once per
            // backend instance and per-CHECK_MEMORY_ERROR jumps to it.
            // Pyre's dynasm doesn't have that out-of-line trampoline
            // infrastructure yet, so the path is inlined per occurrence
            // — equivalent semantics, slightly more code per site.
            // CHECK_MEMORY_ERROR is rare (only after the four CALL_R
            // malloc helpers in `gen_call_malloc_gc`), so the size
            // overhead is negligible.
            //
            // Sequence per assembler.py:328-345:
            //   1. _store_and_reset_exception(self.mc, eax)
            //      — read pos_exc_value → eax, clear pos_exc_value and
            //      pos_exception (assembler.py:1826-1843).
            //   2. mov [jf_guard_exc], eax
            //      — transfer the saved value into the deadframe so
            //      `cpu.grab_exc_value(deadframe)` can read it back in
            //      `PropagateExceptionDescr.handle_fail` (compile.py:1095).
            //   3. mov [jf_descr], propagate_exception_descr
            //   4. _call_footer
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
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    let exc_value_addr = crate::jit_exc_value_addr() as i64;
                    let exc_type_addr = crate::jit_exc_type_addr() as i64;
                    // rax is freely clobberable on the propagate path:
                    // the next instruction emitted after the dynasm! block
                    // is `_call_footer` (assembler.py:1097: mov eax, ebp;
                    // ret), which overwrites rax with rbp before returning.
                    dynasm!(self.mc ; .arch x64
                        ; test Rq(reg), Rq(reg)
                        ; jnz =>skip
                        // assembler.py:1836 — MOV tmp, [pos_exc_value]
                        ; mov Rq(scratch), QWORD exc_value_addr
                        ; mov rax, [Rq(scratch)]
                        // assembler.py:1842-1843 — clear both globals.
                        ; mov QWORD [Rq(scratch)], 0
                        ; mov Rq(scratch), QWORD exc_type_addr
                        ; mov QWORD [Rq(scratch)], 0
                        // assembler.py:336-337 — MOV [jf_guard_exc], tmp
                        ; mov [rbp + JF_GUARD_EXC_OFS], rax
                        // assembler.py:339-340 — MOV [jf_descr], descr
                        ; mov Rq(scratch), QWORD propagate_descr
                        ; mov [rbp + JF_DESCR_OFS], Rq(scratch)
                    );
                    self._call_footer();
                    dynasm!(self.mc ; .arch x64 ; =>skip);
                }
            }
            // x86/assembler.py:2438 genop_discard_cond_call_gc_wb
            OpCode::CondCallGcWb | OpCode::CondCallGcWbArray => {
                self.emit_write_barrier_fastpath(op, &arglocs);
            }
            // ── Misc ──
            OpCode::ForceToken => {
                if let Some(Loc::Reg(r)) = result_loc {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(r.value), rbp);
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
                if arglocs.len() >= 2 {
                    self.emit_guard_is_object(&arglocs[0], &arglocs[1]);
                    self.guard_success_cc = Some(CC_NE);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardSubclass => {
                if arglocs.len() >= 3 {
                    self.emit_guard_subclass(&arglocs[0], &arglocs[1], &arglocs[2]);
                    self.guard_success_cc = Some(CC_B);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            OpCode::GuardException => {
                if arglocs.len() >= 2 {
                    self.emit_guard_exception(&arglocs[0], &arglocs[1]);
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
                dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + JF_DESCR_OFS], 0);
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
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                let ofs = vtable_offset as i32;
                dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), [Rq(obj.value) + ofs]);
                match class_loc {
                    Loc::Reg(c) => {
                        dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), Rq(c.value));
                    }
                    Loc::Immed(i) => {
                        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD i.value ; cmp Rq(scratch), rax);
                    }
                    _ => {}
                }
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

    fn emit_load_gc_typeid_into_reg(&mut self, obj_reg: u8, dst_reg: u8) {
        let tid_ofs = -(majit_gc::header::GcHeader::SIZE as i32);
        dynasm!(self.mc ; .arch x64 ; mov Rd(dst_reg), [Rq(obj_reg) + tid_ofs]);
    }

    /// x86/assembler.py:1893-1901 `_cmp_guard_gc_type`, adjusted for
    /// majit's object pointer: the GC header word lives at
    /// `obj - GcHeader::SIZE`, and a 32-bit load zero-extends the type id.
    fn _cmp_guard_gc_type(&mut self, obj_loc: &Loc, expected_typeid_loc: &Loc) {
        let Loc::Reg(obj) = obj_loc else {
            return;
        };
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        self.emit_load_gc_typeid_into_reg(obj.value, scratch);
        match expected_typeid_loc {
            Loc::Reg(expected) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), Rq(expected.value));
            }
            Loc::Frame(frame) => {
                let ofs = frame.ebp_loc.value;
                dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), [rbp + ofs]);
            }
            Loc::Immed(expected) => {
                let expected_i32 = expected.value as i32;
                dynasm!(self.mc ; .arch x64 ; cmp Rq(scratch), expected_i32);
            }
            _ => {}
        }
    }

    /// x86/assembler.py:1924-1943 `genop_guard_guard_is_object`.
    fn emit_guard_is_object(&mut self, obj_loc: &Loc, typeid_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_IS_OBJECT");
        let (Loc::Reg(obj), Loc::Reg(typeid)) = (obj_loc, typeid_loc) else {
            return;
        };
        self.emit_load_gc_typeid_into_reg(obj.value, typeid.value);
        if info.shift_by > 0 {
            let shift = info.shift_by as i8;
            dynasm!(self.mc ; .arch x64 ; shl Rq(typeid.value), shift);
        }
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let base_type_info = info.base_type_info as i64;
        let infobits_offset = info.infobits_offset as i32;
        let is_object_flag = info.is_object_flag as i8;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD base_type_info
            ; add Rq(typeid.value), Rq(scratch)
            ; test BYTE [Rq(typeid.value) + infobits_offset], is_object_flag
        );
    }

    /// x86/assembler.py:1945-1980 `genop_guard_guard_subclass`.
    fn emit_guard_subclass(&mut self, obj_loc: &Loc, class_loc: &Loc, tmp_loc: &Loc) {
        let info = self.require_guard_gc_type_info("GUARD_SUBCLASS");
        let (Loc::Reg(obj), Loc::Immed(classptr), Loc::Reg(tmp)) = (obj_loc, class_loc, tmp_loc)
        else {
            panic!(
                "GUARD_SUBCLASS expects [Reg object, Immed classptr, Reg tmp] \
                 like x86/assembler.py:1947"
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
            let offset = vtable_offset as i32;
            let offset2 = info.subclassrange_min_offset as i32;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(tmp.value), [Rq(obj.value) + offset]
                ; mov Rq(tmp.value), [Rq(tmp.value) + offset2]
            );
        } else {
            self.emit_load_gc_typeid_into_reg(obj.value, tmp.value);
            if info.shift_by > 0 {
                let shift = info.shift_by as i8;
                dynasm!(self.mc ; .arch x64 ; shl Rq(tmp.value), shift);
            }
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            let base =
                (info.base_type_info + info.sizeof_ti + info.subclassrange_min_offset) as i64;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD base
                ; add Rq(tmp.value), Rq(scratch)
                ; mov Rq(tmp.value), [Rq(tmp.value)]
            );
        }
        self.emit_sub_imm64(tmp.value, check_min);
        self.emit_cmp_imm64(tmp.value, check_max - check_min);
    }

    fn emit_sub_imm64(&mut self, reg: u8, value: i64) {
        if let Ok(v) = i32::try_from(value) {
            dynasm!(self.mc ; .arch x64 ; sub Rq(reg), v);
        } else {
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD value
                ; sub Rq(reg), Rq(scratch)
            );
        }
    }

    fn emit_cmp_imm64(&mut self, reg: u8, value: i64) {
        if let Ok(v) = i32::try_from(value) {
            dynasm!(self.mc ; .arch x64 ; cmp Rq(reg), v);
        } else {
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD value
                ; cmp Rq(reg), Rq(scratch)
            );
        }
    }

    fn emit_cmp_reg_loc_i64(&mut self, reg: u8, loc: &Loc) {
        match loc {
            Loc::Reg(other) => {
                dynasm!(self.mc ; .arch x64 ; cmp Rq(reg), Rq(other.value));
            }
            Loc::Frame(frame) => {
                let ofs = frame.ebp_loc.value;
                dynasm!(self.mc ; .arch x64 ; cmp Rq(reg), [rbp + ofs]);
            }
            Loc::Immed(value) => self.emit_cmp_imm64(reg, value.value),
            _ => {}
        }
    }

    /// x86/assembler.py:1808-1815 `genop_guard_guard_exception`.
    fn emit_guard_exception(&mut self, expected_loc: &Loc, tmp_loc: &Loc) {
        let Loc::Reg(tmp) = tmp_loc else {
            return;
        };
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(tmp.value), QWORD exc_type_addr
            ; mov Rq(tmp.value), [Rq(tmp.value)]
        );
        self.emit_cmp_reg_loc_i64(tmp.value, expected_loc);
    }

    /// `_store_and_reset_exception`: result = pos_exc_value; clear both
    /// pos_exception and pos_exc_value on the success fallthrough.
    fn emit_store_and_reset_exception(&mut self, result_loc: Option<&Loc>) {
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        if let Some(loc) = result_loc {
            dynasm!(self.mc ; .arch x64 ; mov Rq(scratch), QWORD exc_value_addr);
            match loc {
                Loc::Reg(dst) => {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(scratch)]);
                }
                Loc::Frame(frame) => {
                    let ofs = frame.ebp_loc.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), [Rq(scratch)]
                        ; mov [rbp + ofs], Rq(scratch)
                    );
                }
                _ => {}
            }
        }
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD exc_value_addr
            ; mov QWORD [Rq(scratch)], 0
            ; mov Rq(scratch), QWORD exc_type_addr
            ; mov QWORD [Rq(scratch)], 0
        );
    }

    /// x86/assembler.py:1797-1801 `generate_guard_no_exception`.
    fn emit_cmp_no_exception(&mut self) {
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD exc_type_addr
            ; cmp QWORD [Rq(scratch)], 0
        );
    }

    /// Emit SETcc into a register (zero-extend to 64-bit).
    fn emit_setcc(&mut self, cc: u8, dst_reg: u8) {
        dynasm!(self.mc ; .arch x64 ; xor Rd(dst_reg), Rd(dst_reg));
        match cc {
            CC_E => {
                dynasm!(self.mc ; .arch x64 ; sete  Rb(dst_reg));
            }
            CC_NE => {
                dynasm!(self.mc ; .arch x64 ; setne Rb(dst_reg));
            }
            CC_L => {
                dynasm!(self.mc ; .arch x64 ; setl  Rb(dst_reg));
            }
            CC_GE => {
                dynasm!(self.mc ; .arch x64 ; setge Rb(dst_reg));
            }
            CC_LE => {
                dynasm!(self.mc ; .arch x64 ; setle Rb(dst_reg));
            }
            CC_G => {
                dynasm!(self.mc ; .arch x64 ; setg  Rb(dst_reg));
            }
            CC_B => {
                dynasm!(self.mc ; .arch x64 ; setb  Rb(dst_reg));
            }
            CC_AE => {
                dynasm!(self.mc ; .arch x64 ; setae Rb(dst_reg));
            }
            CC_BE => {
                dynasm!(self.mc ; .arch x64 ; setbe Rb(dst_reg));
            }
            CC_A => {
                dynasm!(self.mc ; .arch x64 ; seta  Rb(dst_reg));
            }
            CC_S => {
                dynasm!(self.mc ; .arch x64 ; sets  Rb(dst_reg));
            }
            CC_NS => {
                dynasm!(self.mc ; .arch x64 ; setns Rb(dst_reg));
            }
            CC_O => {
                dynasm!(self.mc ; .arch x64 ; seto  Rb(dst_reg));
            }
            CC_NO => {
                dynasm!(self.mc ; .arch x64 ; setno Rb(dst_reg));
            }
            _ => {
                dynasm!(self.mc ; .arch x64 ; sete  Rb(dst_reg));
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
        dynasm!(self.mc ; .arch x64 ; jmp =>fail_label);
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
            jump_offset: self.mc.offset(),
            fail_label,
            fail_descr: descr.clone(),
            fail_args: op
                .fail_args
                .as_ref()
                .map(|fa| fa.to_vec())
                .unwrap_or_default(),
            opref_to_slot_snapshot: self.opref_to_slot.clone(),
            const_stores,
            gcmap,
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
        self.fail_descrs.push(descr);
    }

    /// Update fail_arg_locs on all pending guard descriptors.
    /// Unmapped (virtual/dead) OpRefs get None — the resume system
    /// handles them via rd_numb TAGVIRTUAL/TAGCONST encoding.
    fn allocate_unmapped_fail_arg_slots(&mut self) {
        for gt in &self.pending_guard_tokens {
            let locs: Vec<Option<usize>> = gt
                .fail_args
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
            unsafe {
                let descr_mut = &mut *(Arc::as_ptr(&gt.fail_descr) as *mut DynasmFailDescr);
                descr_mut.fail_arg_locs = locs;
            }
        }
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
        dynasm!(self.mc ; .arch x64 ; =>fail_label);

        dynasm!(self.mc ; .arch x64 ; call =>save_regs_label);

        let descr_ptr = Arc::as_ptr(&guard_token.fail_descr) as i64;
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, QWORD descr_ptr
            ; mov [rbp + JF_DESCR_OFS], rax
        );
        self.push_gcmap(guard_token.gcmap);

        for &(slot, val) in &guard_token.const_stores {
            let ofs = Self::slot_offset(slot);
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD val
                ; mov [rbp + ofs], Rq(scratch)
            );
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
        dynasm!(self.mc ; .arch x64 ; =>save_regs_label);
        use crate::regloc::*;
        let gprs = [
            ECX, EAX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
        ];
        for &reg in &gprs {
            let save_slot = core_reg_position(reg).expect("managed x86_64 GPR");
            let ofs = Self::slot_offset(save_slot);
            dynasm!(self.mc ; .arch x64 ; mov [rbp + ofs], Rq(reg.value));
        }
        let xmms = [
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
            XMM14,
        ];
        for &reg in &xmms {
            let save_slot = float_reg_position(reg).expect("managed x86_64 XMM");
            let ofs = Self::slot_offset(save_slot);
            dynasm!(self.mc ; .arch x64 ; movsd [rbp + ofs], Rx(reg.value));
        }
        dynasm!(self.mc ; .arch x64 ; ret);

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
            let stub_ptr = stub_addr as *mut u8;
            let offset = adr_new_target as isize - (stub_addr as isize + 5);
            if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                unsafe {
                    *stub_ptr = 0xE9;
                    (stub_ptr.add(1) as *mut i32).write(offset as i32);
                }
            } else {
                unsafe {
                    *stub_ptr = 0x49;
                    *stub_ptr.add(1) = 0xBB;
                    (stub_ptr.add(2) as *mut u64).write(adr_new_target as u64);
                    *stub_ptr.add(10) = 0x41;
                    *stub_ptr.add(11) = 0xFF;
                    *stub_ptr.add(12) = 0xE3;
                }
            }
        });

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
            let old_ptr = old_addr as *mut u8;
            let offset = new_addr as isize - (old_addr as isize + 5);
            if offset >= i32::MIN as isize && offset <= i32::MAX as isize {
                unsafe {
                    *old_ptr = 0xE9;
                    (old_ptr.add(1) as *mut i32).write(offset as i32);
                }
            } else {
                unsafe {
                    *old_ptr = 0x49;
                    *old_ptr.add(1) = 0xBB;
                    (old_ptr.add(2) as *mut u64).write(new_addr as u64);
                    *old_ptr.add(10) = 0x41;
                    *old_ptr.add(11) = 0xFF;
                    *old_ptr.add(12) = 0xE3;
                }
            }
        });
    }

    // ----------------------------------------------------------------
    // genop_* — integer arithmetic
    // ----------------------------------------------------------------

    /// INT_ADD: result = arg0 + arg1
    fn genop_int_add(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; add rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_SUB: result = arg0 - arg1
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; sub rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MUL: result = arg0 * arg1
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; imul rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_AND: result = arg0 & arg1
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; and rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_OR: result = arg0 | arg1
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; or rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_XOR: result = arg0 ^ arg1
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; xor rax, rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_NEG: result = -arg0
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; neg rax
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_INVERT: result = ~arg0
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; not rax
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_LSHIFT: result = arg0 << arg1
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; shl rax, cl
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; sar rax, cl
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; shr rax, cl
        );
        self.store_rax_to_result(op.pos);
    }

    // ----------------------------------------------------------------
    // genop_* — overflow arithmetic (assembler.py:1413-1425)
    // ----------------------------------------------------------------

    /// assembler.py:1856 genop_int_add_ovf — delegates to genop_int_add,
    /// then sets guard_success_cc = 'NO'. On x86, ADD always sets OF.
    fn genop_int_add_ovf(&mut self, op: &Op) {
        self.genop_int_add(op); // ADD sets OF on x86
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1860 genop_int_sub_ovf.
    fn genop_int_sub_ovf(&mut self, op: &Op) {
        self.genop_int_sub(op);
        self.guard_success_cc = Some(CC_NO);
    }

    /// assembler.py:1864 genop_int_mul_ovf.
    fn genop_int_mul_ovf(&mut self, op: &Op) {
        self.genop_int_mul(op); // IMUL sets OF on x86
        self.guard_success_cc = Some(CC_NO);
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
        dynasm!(self.mc
            ; .arch x64
            ; cmp rax, rcx
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
        match cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; setl al),
            CC_LE => dynasm!(self.mc ; .arch x64 ; setle al),
            CC_G => dynasm!(self.mc ; .arch x64 ; setg al),
            CC_GE => dynasm!(self.mc ; .arch x64 ; setge al),
            CC_E => dynasm!(self.mc ; .arch x64 ; sete al),
            CC_NE => dynasm!(self.mc ; .arch x64 ; setne al),
            CC_B => dynasm!(self.mc ; .arch x64 ; setb al),
            CC_BE => dynasm!(self.mc ; .arch x64 ; setbe al),
            CC_A => dynasm!(self.mc ; .arch x64 ; seta al),
            CC_AE => dynasm!(self.mc ; .arch x64 ; setae al),
            CC_O => dynasm!(self.mc ; .arch x64 ; seto al),
            CC_NO => dynasm!(self.mc ; .arch x64 ; setno al),
            _ => dynasm!(self.mc ; .arch x64 ; sete al),
        }
        dynasm!(self.mc
            ; .arch x64
            ; movzx eax, al
        );
        self.store_rax_to_result(result_opref);
    }

    /// INT_IS_TRUE: result = (arg0 != 0)
    fn genop_int_is_true(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos);
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
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
            let off_i32 = off_usize as i32;
            dynasm!(self.mc
                ; .arch x64
                ; mov rax, [rax + off_i32]
                ; cmp rax, rcx
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
            let typeid_i32 = expected_typeid as i32;
            dynasm!(self.mc
                ; .arch x64
                ; cmp DWORD [rax], typeid_i32
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
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; jl =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch x64 ; jle =>fail_label),
            CC_G => dynasm!(self.mc ; .arch x64 ; jg =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch x64 ; jge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch x64 ; jne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch x64 ; jb =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch x64 ; jbe =>fail_label),
            CC_A => dynasm!(self.mc ; .arch x64 ; ja =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch x64 ; jae =>fail_label),
            CC_O => dynasm!(self.mc ; .arch x64 ; jo =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch x64 ; jno =>fail_label),
            CC_S => dynasm!(self.mc ; .arch x64 ; js =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch x64 ; jns =>fail_label),
            _ => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
        }
        // aarch64: b.cond has 19-bit range (±1MB), which is too short
        // for forward references to recovery stubs. Use inverted condition
        // + unconditional branch (26-bit / ±128MB) pattern instead:
        //   b.NOT_cond >skip ; b =>fail_label ; skip:
        fail_label
    }

    fn emit_jcc_to_label(&mut self, fail_cc: u8, fail_label: DynamicLabel) {
        match fail_cc {
            CC_L => dynasm!(self.mc ; .arch x64 ; jl =>fail_label),
            CC_LE => dynasm!(self.mc ; .arch x64 ; jle =>fail_label),
            CC_G => dynasm!(self.mc ; .arch x64 ; jg =>fail_label),
            CC_GE => dynasm!(self.mc ; .arch x64 ; jge =>fail_label),
            CC_E => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
            CC_NE => dynasm!(self.mc ; .arch x64 ; jne =>fail_label),
            CC_B => dynasm!(self.mc ; .arch x64 ; jb =>fail_label),
            CC_BE => dynasm!(self.mc ; .arch x64 ; jbe =>fail_label),
            CC_A => dynasm!(self.mc ; .arch x64 ; ja =>fail_label),
            CC_AE => dynasm!(self.mc ; .arch x64 ; jae =>fail_label),
            CC_O => dynasm!(self.mc ; .arch x64 ; jo =>fail_label),
            CC_NO => dynasm!(self.mc ; .arch x64 ; jno =>fail_label),
            CC_S => dynasm!(self.mc ; .arch x64 ; js =>fail_label),
            CC_NS => dynasm!(self.mc ; .arch x64 ; jns =>fail_label),
            _ => dynasm!(self.mc ; .arch x64 ; je =>fail_label),
        }
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
        let jump_offset = self.mc.offset();
        self.pending_guard_tokens.push(GuardToken {
            jump_offset,
            fail_label,
            fail_descr: descr.clone(),
            fail_args,
            opref_to_slot_snapshot: self.opref_to_slot.clone(),
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
            dynasm!(self.mc ; .arch x64 ; test rax, rax);
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
        dynasm!(self.mc ; .arch x64 ; cmp rax, rcx);

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
        dynasm!(self.mc ; .arch x64 ; cmp rax, 1);

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
        dynasm!(self.mc ; .arch x64 ; jne =>fail_label);

        // assembler.py:1916-1917
        // guard_success_cc is not used here — we already emitted the jumps.
        // Manually append the guard token.
        self.append_guard_token(op, fail_index, fail_label);
    }

    /// assembler.py:1794 generate_guard_no_exception:
    /// `CMP heap(self.cpu.pos_exception()), imm0` with success on zero.
    fn emit_guard_no_exception_check(&mut self) {
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD exc_type_addr
            ; cmp QWORD [Rq(scratch)], 0
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
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD descr_ptr
            ; mov QWORD [rbp + JF_FORCE_DESCR_OFS], rax
            ; mov QWORD [rbp + JF_DESCR_OFS], 0
        );
    }

    /// assembler.py:2228-2232 genop_guard_guard_not_forced.
    /// Check jf_descr == 0 (not forced); if non-zero, guard fails.
    fn genop_guard_guard_not_forced(&mut self, op: &Op, fail_index: u32) {
        // assembler.py:2229-2231: CMP [jf_descr], 0; guard_success_cc = 'E'
        dynasm!(self.mc ; .arch x64
            ; cmp QWORD [rbp + JF_DESCR_OFS], 0           // jf_descr == 0?
        );
        self.guard_success_cc = Some(CC_E); // guard succeeds if equal (not forced)
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
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + dst]);
            } else if arg_ref.is_constant() {
                let val = self.constants.get(&arg_ref.0).copied().unwrap_or(0);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD val as i64 ; push rax);
            } else if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref.0) {
                let src = Self::slot_offset(old_slot);
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + src]);
            } else {
                dynasm!(self.mc ; .arch x64 ; push 0);
            }
        }
        // Pass 2: pop in reverse into canonical slots
        for i in (0..n_label).rev() {
            let dst = Self::slot_offset(i);
            dynasm!(self.mc ; .arch x64 ; pop QWORD [rbp + dst]);
        }

        // Bind the LABEL — JUMP targets here (after the copies).
        let label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; =>label);
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
            dynasm!(self.mc ; .arch x64
                ; mov rax, QWORD val
                ; mov [rbp + dst], rax
            );
        } else if src != dst {
            dynasm!(self.mc ; .arch x64
                ; mov rax, [rbp + src]
                ; mov [rbp + dst], rax
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
                        dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + dst]);
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
                                    dynasm!(self.mc ; .arch x64 ; pop QWORD [rbp + cd]);
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
            dynasm!(self.mc ; .arch x64 ; jmp =>label);
        } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
            // assembler.py closing_jump parity: bridge jumps back to
            // the original loop's LABEL via absolute address.
            let addr = target as i64;
            dynasm!(self.mc ; .arch x64
                ; mov rax, QWORD addr
                ; jmp rax
            );
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
                dynasm!(self.mc
                    ; .arch x64
                    ; mov [rbp + slot0_offset], rax  // store float bits via GPR
                );
            } else {
                self.load_arg_to_rax(arg0);
                dynasm!(self.mc
                    ; .arch x64
                    ; mov [rbp + slot0_offset], rax
                );
            }
        }

        // Store descr pointer at jf_ptr[0] (jf_descr slot).
        // compile.py:665-674 parity: use global singleton pointer.
        let descr_ptr = global_descr_ptr;
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, QWORD descr_ptr
            ; mov [rbp + JF_DESCR_OFS], rax
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
                dynasm!(self.mc
                    ; .arch x64
                    ; movsd xmm0, [rbp + offset]
                );
            }
            ResolvedArg::Const(val) => {
                // Load constant via integer register, then move to float register.
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, QWORD val as i64
                    ; movq xmm0, rax
                );
            }
        }
    }

    /// Load a float value from `opref` into XMM1 (x64) / D1 (aarch64).
    fn load_float_arg_to_d1(&mut self, opref: OpRef) {
        match self.resolve_opref(opref) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; movsd xmm1, [rbp + offset]
                );
            }
            ResolvedArg::Const(val) => {
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rax, QWORD val as i64
                    ; movq xmm1, rax
                );
            }
        }
    }

    /// Store XMM0 (x64) / D0 (aarch64) to the frame slot for `result_opref`.
    fn store_d0_to_result(&mut self, result_opref: OpRef) {
        let slot = self.allocate_slot(result_opref);
        let offset = Self::slot_offset(slot);
        dynasm!(self.mc
            ; .arch x64
            ; movsd [rbp + offset], xmm0
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
        dynasm!(self.mc
            ; .arch x64
            ; addsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_SUB: result = arg0 - arg1
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; subsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_MUL: result = arg0 * arg1
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; mulsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        self.load_float_arg_to_d1(op.arg(1));
        dynasm!(self.mc
            ; .arch x64
            ; divsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos);
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        // Load the sign-bit mask (0x8000_0000_0000_0000) into XMM1
        // via integer register, then XOR.
        let sign_mask: i64 = i64::MIN; // 0x8000000000000000
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, QWORD sign_mask
            ; movq xmm1, rax
            ; xorpd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; cvtsi2sd xmm0, rax
        );
        self.store_d0_to_result(op.pos);
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc
            ; .arch x64
            ; cvttsd2si rax, xmm0
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
            1 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, BYTE [rax + offset]
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, WORD [rax + offset]
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov eax, [rax + offset]
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov rax, [rax + offset]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    /// opassembler.py: emit_op_setfield — sized store via regalloc.
    fn emit_op_setfield_regalloc(
        &mut self,
        _base: &crate::regloc::RegLoc,
        _val_loc: &Loc,
        _ofs: i32,
        _field_size: usize,
    ) {
        todo!("x86 emit_op_setfield_regalloc")
    }

    /// opassembler.py: emit_op_gcload — sized load via regalloc.
    fn emit_op_gcload_regalloc(
        &mut self,
        _base: &crate::regloc::RegLoc,
        _ofs_loc: &Loc,
        _dst: &crate::regloc::RegLoc,
        _size: i64,
    ) {
        todo!("x86 emit_op_gcload_regalloc")
    }

    /// opassembler.py: emit_op_gcstore — sized store via regalloc.
    fn emit_op_gcstore_regalloc(
        &mut self,
        _base: &crate::regloc::RegLoc,
        _ofs_loc: &Loc,
        _val: &crate::regloc::RegLoc,
        _size: usize,
    ) {
        todo!("x86 emit_op_gcstore_regalloc")
    }

    /// SETFIELD_GC: [arg0 + offset] = arg1
    fn genop_discard_setfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load object pointer into rax/x0 and value into rcx/x1.
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));

        match size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], cl
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], cx
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], ecx
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov [rax + offset], rcx
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
        // rcx = rcx * item_size
        if item_size != 1 {
            dynasm!(self.mc
                ; .arch x64
                ; imul rcx, rcx, item_size
            );
        }
        // rax = rax + base_size + rcx
        dynasm!(self.mc
            ; .arch x64
            ; add rax, base_size
            ; add rax, rcx
        );
        match item_size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, BYTE [rax]
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, WORD [rax]
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov eax, [rax]
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov rax, [rax]
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
            dynasm!(self.mc
                ; .arch x64
                ; imul rcx, rcx, item_size
            );
        }
        dynasm!(self.mc
            ; .arch x64
            ; add rax, base_size
            ; add rax, rcx
        );

        // Now load value from arg2 and store it.
        // We need a third register: use rcx/x1 again for the value
        // (the address is in rax/x0).
        // Save rax/x0 (element address) before loading value.
        // Push address, load value into rcx, pop address into rax.
        dynasm!(self.mc
            ; .arch x64
            ; push rax
        );
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc
            ; .arch x64
            ; pop rax
        );

        match item_size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax], cl
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax], cx
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov [rax], ecx
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov [rax], rcx
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
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + len_offset]
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

        dynasm!(self.mc ; .arch x64 ; push rbp);

        for i in (func_arg + 1)..arg_count.min(func_arg + 7) {
            let arg = op.arg(i);
            let abi_idx = i - func_arg - 1;
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => match abi_idx {
                    0 => dynasm!(self.mc ; .arch x64 ; mov rdi, [rbp + offset]),
                    1 => dynasm!(self.mc ; .arch x64 ; mov rsi, [rbp + offset]),
                    2 => dynasm!(self.mc ; .arch x64 ; mov rdx, [rbp + offset]),
                    3 => dynasm!(self.mc ; .arch x64 ; mov rcx, [rbp + offset]),
                    4 => dynasm!(self.mc ; .arch x64 ; mov r8, [rbp + offset]),
                    5 => dynasm!(self.mc ; .arch x64 ; mov r9, [rbp + offset]),
                    _ => {}
                },
                ResolvedArg::Const(val) => match abi_idx {
                    0 => dynasm!(self.mc ; .arch x64 ; mov rdi, QWORD val as i64),
                    1 => dynasm!(self.mc ; .arch x64 ; mov rsi, QWORD val as i64),
                    2 => dynasm!(self.mc ; .arch x64 ; mov rdx, QWORD val as i64),
                    3 => dynasm!(self.mc ; .arch x64 ; mov rcx, QWORD val as i64),
                    4 => dynasm!(self.mc ; .arch x64 ; mov r8, QWORD val as i64),
                    5 => dynasm!(self.mc ; .arch x64 ; mov r9, QWORD val as i64),
                    _ => {}
                },
            }
        }

        match self.resolve_opref(op.arg(func_arg)) {
            ResolvedArg::Slot(offset) => {
                dynasm!(self.mc ; .arch x64
                    ; mov rax, [rbp + offset]
                    ; call rax
                );
            }
            ResolvedArg::Const(val) => {
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD val as i64
                    ; call rax
                );
            }
        }

        dynasm!(self.mc ; .arch x64 ; pop rbp);
    }

    /// aarch64/opassembler.py:1036 _emit_call.
    /// arglocs = [resloc, size, sign, func, args...] for normal CALLs and
    /// [resloc, size, sign, saveerr, func, args...] for CALL_RELEASE_GIL.
    fn emit_call_from_arglocs(&mut self, arglocs: &[Loc], func_index: usize) {
        let arg_count = arglocs.len();

        dynasm!(self.mc ; .arch x64 ; push rbp);

        for i in (func_index + 1)..arg_count.min(func_index + 7) {
            let abi_idx = i - func_index - 1;
            let arg = &arglocs[i];
            match arg {
                Loc::Frame(f) => {
                    let offset = f.ebp_loc.value;
                    match abi_idx {
                        0 => dynasm!(self.mc ; .arch x64 ; mov rdi, [rbp + offset]),
                        1 => dynasm!(self.mc ; .arch x64 ; mov rsi, [rbp + offset]),
                        2 => dynasm!(self.mc ; .arch x64 ; mov rdx, [rbp + offset]),
                        3 => dynasm!(self.mc ; .arch x64 ; mov rcx, [rbp + offset]),
                        4 => dynasm!(self.mc ; .arch x64 ; mov r8, [rbp + offset]),
                        5 => dynasm!(self.mc ; .arch x64 ; mov r9, [rbp + offset]),
                        _ => {}
                    }
                }
                Loc::Reg(r) => match abi_idx {
                    0 => dynasm!(self.mc ; .arch x64 ; mov rdi, Rq(r.value)),
                    1 => dynasm!(self.mc ; .arch x64 ; mov rsi, Rq(r.value)),
                    2 => dynasm!(self.mc ; .arch x64 ; mov rdx, Rq(r.value)),
                    3 => dynasm!(self.mc ; .arch x64 ; mov rcx, Rq(r.value)),
                    4 => dynasm!(self.mc ; .arch x64 ; mov r8, Rq(r.value)),
                    5 => dynasm!(self.mc ; .arch x64 ; mov r9, Rq(r.value)),
                    _ => {}
                },
                Loc::Immed(i) => {
                    let val = i.value;
                    match abi_idx {
                        0 => dynasm!(self.mc ; .arch x64 ; mov rdi, QWORD val),
                        1 => dynasm!(self.mc ; .arch x64 ; mov rsi, QWORD val),
                        2 => dynasm!(self.mc ; .arch x64 ; mov rdx, QWORD val),
                        3 => dynasm!(self.mc ; .arch x64 ; mov rcx, QWORD val),
                        4 => dynasm!(self.mc ; .arch x64 ; mov r8, QWORD val),
                        5 => dynasm!(self.mc ; .arch x64 ; mov r9, QWORD val),
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        match arglocs.get(func_index) {
            Some(Loc::Frame(f)) => {
                let offset = f.ebp_loc.value;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, [rbp + offset]
                    ; call rax
                );
            }
            Some(Loc::Reg(r)) => {
                dynasm!(self.mc ; .arch x64
                    ; mov rax, Rq(r.value)
                    ; call rax
                );
            }
            Some(Loc::Immed(i)) => {
                let val = i.value;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD val
                    ; call rax
                );
            }
            _ => {}
        }

        dynasm!(self.mc ; .arch x64 ; pop rbp);
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
                    dynasm!(self.mc ; .arch x64 ; shl rax, 32 ; sar rax, 32);
                } else {
                    dynasm!(self.mc ; .arch x64 ; shl rax, 32 ; shr rax, 32);
                }
            }
            2 => {
                if signed {
                    dynasm!(self.mc ; .arch x64 ; shl rax, 48 ; sar rax, 48);
                } else {
                    dynasm!(self.mc ; .arch x64 ; and rax, 0xFFFF);
                }
            }
            1 => {
                if signed {
                    dynasm!(self.mc ; .arch x64 ; shl rax, 56 ; sar rax, 56);
                } else {
                    dynasm!(self.mc ; .arch x64 ; and rax, 0xFF);
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
            self.store_rax_to_result(op.pos);
        }
    }

    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        self._genop_call_with_arglocs(op, arglocs);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
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
        if expansion.is_none() {
            let frame_loc = arglocs
                .first()
                .copied()
                .expect("call_assembler missing rewritten jitframe arg");
            let vable_loc = arglocs.get(1).copied();
            dynasm!(self.mc ; .arch x64
                ; mov r12, rbp
            );
            self.emit_load_to_rax(frame_loc);
            dynasm!(self.mc ; .arch x64
                ; mov rdx, rax
            );

            let target_addr: Option<usize> = op
                .descr
                .as_ref()
                .and_then(|d| d.as_call_descr())
                .and_then(|cd| cd.call_target_token())
                .and_then(|token| self.call_assembler_targets.get(&token).copied())
                .filter(|&addr| addr != 0);
            let is_resolved = target_addr.is_some() || self.self_entry_label.is_some();
            let result_type = op.opcode.result_type();
            let done_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
            let helper_addr = crate::call_assembler_helper_addr() as i64;
            let green_key = self.header_pc as i64;

            if !is_resolved {
                let force_addr = crate::call_assembler_force_fn_addr() as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rbp, r12
                );
                if force_addr != 0 {
                    if let Some(vloc) = vable_loc {
                        self.emit_load_to_rax(vloc);
                        dynasm!(self.mc ; .arch x64
                            ; mov rdi, rax
                        );
                    } else {
                        dynasm!(self.mc ; .arch x64
                            ; mov rdi, [rdx + FIRST_ITEM_OFFSET as i32]
                        );
                    }
                    let pushed_gcmap = self.push_pending_call_gcmap();
                    dynasm!(self.mc ; .arch x64
                        ; sub rsp, 8
                        ; mov rax, QWORD force_addr
                        ; call rax
                        ; add rsp, 8
                    );
                    self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                } else {
                    dynasm!(self.mc ; .arch x64
                        ; xor eax, eax
                    );
                }
                if !op.pos.is_none() {
                    self.store_rax_to_result(op.pos);
                }
                return;
            }

            let trampoline_addr = crate::call_assembler_execute_addr() as i64;
            let pushed_gcmap = self.push_pending_call_gcmap();
            dynasm!(self.mc ; .arch x64
                ; mov rdi, rdx
                ; sub rsp, 8
            );
            if let Some(addr) = target_addr {
                let addr = addr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rsi, QWORD addr
                    ; mov rax, QWORD trampoline_addr
                    ; call rax
                );
            } else if self.self_entry_label.is_some() {
                let addr_ptr = self.self_entry_addr_ptr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rsi, QWORD addr_ptr
                    ; mov rsi, [rsi]
                    ; mov rax, QWORD trampoline_addr
                    ; call rax
                );
            }
            dynasm!(self.mc ; .arch x64
                ; add rsp, 8
                ; mov rbp, r12
            );
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
            dynasm!(self.mc ; .arch x64
                ; mov rdx, rax
            );

            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            dynasm!(self.mc ; .arch x64
                ; mov rcx, [rdx + JF_DESCR_OFS]
                ; mov rax, QWORD done_descr_ptr
                ; cmp rcx, rax
                ; je =>fast_path
            );
            // `compile.py:665` parity: helper signature is
            // `(cpu_handle, callee_jf_ptr, green_key)`. Pass cpu_ptr as
            // arg0 (rdi) so the trampoline can resolve the attached
            // `done_with_this_frame_descr_*` /
            // `exit_frame_with_exception_descr_ref` identities.
            let cpu_ptr = self.cpu_handle_ptr();
            dynasm!(self.mc ; .arch x64
                ; mov rdi, QWORD cpu_ptr            // arg0 = cpu_handle
                ; mov rsi, rdx                      // arg1 = callee_jf_ptr
                ; mov rdx, QWORD green_key as i64   // arg2 = green_key
                ; sub rsp, 8
                ; mov rax, QWORD helper_addr
            );
            let pushed_gcmap = self.push_pending_call_gcmap();
            dynasm!(self.mc ; .arch x64
                ; call rax
                ; add rsp, 8
            );
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
            dynasm!(self.mc ; .arch x64
                ; jmp =>merge
                ; =>fast_path
            );
            if result_type == Type::Float {
                dynasm!(self.mc ; .arch x64
                    ; movsd xmm0, [rdx + FIRST_ITEM_OFFSET as i32]
                    ; movq rax, xmm0
                    ; =>merge
                );
            } else {
                dynasm!(self.mc ; .arch x64
                    ; mov rax, [rdx + FIRST_ITEM_OFFSET as i32]
                    ; =>merge
                );
            }
            if !op.pos.is_none() {
                self.store_rax_to_result(op.pos);
            }
            return;
        }

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
        dynasm!(self.mc ; .arch x64
            ; mov r12, rbp            // save caller's jf_ptr in r12
        );

        // Allocate callee jitframe on heap via calloc.
        // Stack alignment: after prologue (push rbp + push r12) + return
        // addr, rsp ≡ -8 (mod 16). sub 8 aligns to 16 for ABI call.
        dynasm!(self.mc ; .arch x64
            ; sub rsp, 8                        // align to 16
            ; mov rdi, 1                        // calloc nmemb
            ; mov rsi, QWORD jf_alloc_bytes     // calloc size
            ; mov rax, QWORD calloc_ptr
            ; call rax                          // rax = heap jf_ptr
            ; add rsp, 8                        // unalign
        );

        // rdx/x20 = heap jf_ptr (held across arg stores).
        // load_arg_to_rax reads from [rbp+offset], rbp still = caller's jf.
        // Wait: rbp was saved to r12 but NOT changed yet. Actually we did
        // `mov r12, rbp` above, so rbp still points to caller's jf. ✓
        dynasm!(self.mc ; .arch x64
            ; mov rdx, rax            // rdx = heap jf_ptr
            ; mov QWORD [rdx + JF_DESCR_OFS], 0
            ; mov QWORD [rdx + JF_FRAME_OFS as i32], jf_frame_len as i32
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
                    dynasm!(self.mc ; .arch x64
                        ; mov rax, QWORD cval
                        ; mov [rdx + dest_offset], rax
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
                    dynasm!(self.mc ; .arch x64
                        ; mov [rdx + dest_offset], rax
                    );
                    continue;
                }

                if slot == 0 {
                    let frame_loc = arglocs
                        .first()
                        .copied()
                        .expect("call_assembler vable expansion missing frame arg");
                    self.emit_load_to_rax(frame_loc);
                    dynasm!(self.mc ; .arch x64
                        ; mov [rdx + dest_offset], rax
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
                    dynasm!(self.mc ; .arch x64
                        ; mov rax, [rax + field_ofs as i32]
                        ; mov [rdx + dest_offset], rax
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
                dynasm!(self.mc ; .arch x64
                    ; mov rax, [rax + data_ptr_ofs as i32]
                    ; mov rax, [rax + item_ofs]
                    ; mov [rdx + dest_offset], rax
                );
            }
        } else {
            for (i, loc) in arglocs.iter().enumerate() {
                let dest_offset = Self::slot_offset(JITFRAME_FIXED_SIZE + i);
                self.emit_load_to_rax(*loc);
                dynasm!(self.mc ; .arch x64
                    ; mov [rdx + dest_offset], rax
                );
            }
        }

        // _call_assembler_emit_call (assembler.py:2267-2269):
        // rdi/x0 = callee jf_ptr.
        dynasm!(self.mc ; .arch x64
            ; mov rdi, rdx            // rdi = heap jf ptr
            ; sub rsp, 8              // align stack for call
        );

        // assembler.py:320 _call_assembler_emit_call(descr._ll_function_addr, ...)
        // Resolve target address from descr.call_target_token() or self_entry_label.
        let target_addr: Option<usize> = op
            .descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .and_then(|cd| cd.call_target_token())
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
                dynasm!(self.mc ; .arch x64
                    ; mov r13, [rdx + Self::slot_offset(JITFRAME_FIXED_SIZE) as i32]
                    ; mov rdi, rdx                   // free(heap_jf)
                    ; sub rsp, 8
                    ; mov rax, QWORD free_ptr
                    ; call rax
                    ; add rsp, 8
                    ; mov rbp, r12                   // restore caller jf_ptr
                    ; mov rdi, r13                   // arg0 = PyFrame ptr
                    ; sub rsp, 8
                    ; mov rax, QWORD force_addr
                    ; call rax                       // rax = force_fn(frame_ptr) = result
                    ; add rsp, 8
                );
                self.reload_frame_if_necessary();
            } else {
                // No force_fn registered — free and return 0.
                dynasm!(self.mc ; .arch x64
                    ; mov rdi, rdx                   // free(heap_jf)
                    ; sub rsp, 8
                    ; mov rax, QWORD free_ptr
                    ; call rax
                    ; add rsp, 8
                    ; mov rbp, r12                   // restore caller jf_ptr
                    ; xor eax, eax                   // result = 0
                );
            }
        } else {
            // Resolved target: call callee via execute trampoline
            // (stacker stack-growth protection for deep CALL_ASSEMBLER recursion).
            let trampoline_addr = crate::call_assembler_execute_addr() as i64;
            if let Some(addr) = target_addr {
                let addr = addr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rsi, QWORD addr             // arg1 = callee entry addr
                    ; mov rax, QWORD trampoline_addr
                    ; call rax
                );
            } else if self.self_entry_label.is_some() {
                // Self-entry: load entry addr from self_entry_addr_ptr
                // (written after finalization).
                let addr_ptr = self.self_entry_addr_ptr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rsi, QWORD addr_ptr         // rsi = &entry_addr
                    ; mov rsi, [rsi]                   // rsi = entry_addr
                    ; mov rax, QWORD trampoline_addr
                    ; call rax
                );
            }

            // rax/x0 = callee's returned jf_ptr (= heap jf_ptr we passed).
            dynasm!(self.mc ; .arch x64
                ; add rsp, 8              // undo alignment
            );

            // Restore caller's jf_ptr.
            dynasm!(self.mc ; .arch x64
                ; mov rbp, r12            // restore caller's jf_ptr
            );
            self.reload_frame_if_necessary();

            // rax = callee's returned jf_ptr (heap-allocated).
            // Save it in rdx for descr check and free.
            dynasm!(self.mc ; .arch x64
                ; mov rdx, rax            // rdx = callee jf_ptr
            );

            // _call_assembler_check_descr (assembler.py:2274-2278):
            //   CMP [jf_ptr + jf_descr_ofs], done_with_this_frame_descr_{type}
            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            dynasm!(self.mc ; .arch x64
                ; mov rcx, [rdx + JF_DESCR_OFS]     // rcx = jf_descr
                ; mov rax, QWORD done_descr_ptr
                ; cmp rcx, rax
                ; je =>fast_path
            );

            // Path A (slow): guard failure.
            // assembler.py:345-350 _call_assembler_emit_helper_call.
            // `compile.py:665` parity: pass `cpu_ptr` as arg0 so the
            // trampoline resolves `self.cpu.done_with_this_frame_descr_*` /
            // `exit_frame_with_exception_descr_ref` through the owning
            // backend instance rather than a per-thread fallback.
            let cpu_ptr = self.cpu_handle_ptr();
            dynasm!(self.mc ; .arch x64
                ; mov rdi, QWORD cpu_ptr            // arg0 = cpu_handle
                ; mov rsi, rdx                      // arg1 = callee_jf_ptr
                ; mov rdx, QWORD green_key as i64   // arg2 = green_key
                ; sub rsp, 8
                ; mov rax, QWORD helper_addr
                ; call rax                          // rax = helper result
                ; add rsp, 8
            );
            self.reload_frame_if_necessary();
            dynasm!(self.mc ; .arch x64 ; jmp =>merge);

            // Path B (fast): _call_assembler_load_result (assembler.py:2291-2303)
            dynasm!(self.mc ; .arch x64
                ; =>fast_path
            );
            if result_type == Type::Float {
                dynasm!(self.mc ; .arch x64
                    ; movsd xmm0, [rdx + FIRST_ITEM_OFFSET as i32]
                    ; movq r12, xmm0               // preserve bits across free
                    ; mov rdi, rdx                  // free(callee_jf)
                    ; sub rsp, 8
                    ; mov rax, QWORD free_ptr
                    ; call rax
                    ; add rsp, 8
                    ; mov rax, r12                  // float bits in rax
                    ; =>merge
                );
            } else {
                dynasm!(self.mc ; .arch x64
                    ; mov r12, [rdx + FIRST_ITEM_OFFSET as i32]
                    ; mov rdi, rdx                  // free(callee_jf)
                    ; sub rsp, 8
                    ; mov rax, QWORD free_ptr
                    ; call rax
                    ; add rsp, 8
                    ; mov rax, r12                  // rax = result
                    ; =>merge
                );
            }
        } // end if is_resolved

        // Store result to the output slot (rax/x0 holds result).
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }

        // Restore callee-saved regs clobbered by this sequence.
    }

    // ----------------------------------------------------------------
    // genop_* — allocation
    // x86/assembler.py:2338 genop_new etc.
    // These require GC runtime support. Emit trap for now.
    // ----------------------------------------------------------------

    /// x86/assembler.py:2438 _write_barrier_fastpath parity.
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
        let mut mask = wb.jit_wb_if_flag_singlebyte as i64;
        if card_marking {
            mask |= wb.jit_wb_cards_set_singlebyte as i64;
        }
        mask &= 0xFF;
        let byteofs = wb.jit_wb_if_flag_byteofs;
        // x86/assembler.py:2487: TEST byte [base+byteofs], mask
        dynasm!(self.mc ; .arch x64
            ; test BYTE [Rq(loc_base.value as u8) + byteofs], mask as i8
        );
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; jz =>done);

        if card_marking {
            // x86/assembler.py:2398-2408: test GCFLAG_CARDS_SET separately
            let cards_mask = (wb.jit_wb_cards_set_singlebyte as u8) as i8;
            dynasm!(self.mc ; .arch x64
                ; test BYTE [Rq(loc_base.value as u8) + byteofs], cards_mask
            );
            let card_mark = self.mc.new_dynamic_label();
            dynasm!(self.mc ; .arch x64 ; jnz =>card_mark);

            // No CARDS_SET yet: call array barrier helper
            self.emit_wb_helper_call_x86(
                loc_base,
                crate::runner::dynasm_write_barrier_from_array as *const () as i64,
            );

            // Re-check CARDS_SET after helper
            dynasm!(self.mc ; .arch x64
                ; test BYTE [Rq(loc_base.value as u8) + byteofs], cards_mask
                ; jz =>done
            );

            // Inline card bit set (x86/assembler.py:2398 WriteBarrierSlowPath parity)
            dynasm!(self.mc ; .arch x64 ; =>card_mark);
            if let Some(Loc::Reg(loc_index)) = arglocs.get(1) {
                let shift = (3 + wb.jit_wb_card_page_shift) as i8;
                // byte_index = ~(index >> (3+page_shift))
                dynasm!(self.mc ; .arch x64
                    ; mov rcx, Rq(loc_index.value as u8)
                    ; shr rcx, shift
                    ; not rcx
                );
                // bit_index = (index >> page_shift) & 7 → set bit
                let page_shift = wb.jit_wb_card_page_shift as i8;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, Rq(loc_index.value as u8)
                    ; shr rax, page_shift
                    ; and rax, 7
                    ; mov rdx, 1
                    ; shl rdx, cl  // cl = bit_index? no, need rax
                );
                // Actually simpler: use BTS instruction
                // card_byte_ofs = rcx, bit = (index >> page_shift) & 7
                // Just OR the byte directly
                dynasm!(self.mc ; .arch x64
                    ; mov rdx, 1
                    ; mov r8, Rq(loc_index.value as u8)
                    ; shr r8, page_shift
                    ; and r8, 7
                );
                // set bit: load byte, or, store
                dynasm!(self.mc ; .arch x64
                    ; mov cl, r8b
                    ; shl dl, cl
                    ; or BYTE [Rq(loc_base.value as u8) + rcx], dl
                );
            }
        } else {
            // Non-array: generic barrier
            self.emit_wb_helper_call_x86(
                loc_base,
                crate::runner::dynasm_write_barrier as *const () as i64,
            );
        }

        dynasm!(self.mc ; .arch x64 ; =>done);
    }

    /// _build_wb_slowpath parity: save all GPR + XMM regs, call helper, restore.
    /// x86/assembler.py:2331-2370 + 2417 (XMM variant).
    fn emit_wb_helper_call_x86(&mut self, loc_base: crate::regloc::RegLoc, helper: i64) {
        // Save all caller-saved GPRs
        dynasm!(self.mc ; .arch x64
            ; push rax ; push rcx ; push rdx ; push rsi ; push rdi
            ; push r8 ; push r9 ; push r10 ; push r11
        );
        // Save XMM caller-saved (xmm0-xmm15, 16 × 16 bytes = 256 bytes)
        dynasm!(self.mc ; .arch x64
            ; sub rsp, 256
            ; movaps [rsp], xmm0
            ; movaps [rsp + 16], xmm1
            ; movaps [rsp + 32], xmm2
            ; movaps [rsp + 48], xmm3
            ; movaps [rsp + 64], xmm4
            ; movaps [rsp + 80], xmm5
            ; movaps [rsp + 96], xmm6
            ; movaps [rsp + 112], xmm7
            ; movaps [rsp + 128], xmm8
            ; movaps [rsp + 144], xmm9
            ; movaps [rsp + 160], xmm10
            ; movaps [rsp + 176], xmm11
            ; movaps [rsp + 192], xmm12
            ; movaps [rsp + 208], xmm13
            ; movaps [rsp + 224], xmm14
            ; movaps [rsp + 240], xmm15
        );
        dynasm!(self.mc ; .arch x64
            ; mov rdi, Rq(loc_base.value as u8)
            ; mov rax, QWORD helper
            ; call rax
        );
        // Restore XMM
        dynasm!(self.mc ; .arch x64
            ; movaps xmm0, [rsp]
            ; movaps xmm1, [rsp + 16]
            ; movaps xmm2, [rsp + 32]
            ; movaps xmm3, [rsp + 48]
            ; movaps xmm4, [rsp + 64]
            ; movaps xmm5, [rsp + 80]
            ; movaps xmm6, [rsp + 96]
            ; movaps xmm7, [rsp + 112]
            ; movaps xmm8, [rsp + 128]
            ; movaps xmm9, [rsp + 144]
            ; movaps xmm10, [rsp + 160]
            ; movaps xmm11, [rsp + 176]
            ; movaps xmm12, [rsp + 192]
            ; movaps xmm13, [rsp + 208]
            ; movaps xmm14, [rsp + 224]
            ; movaps xmm15, [rsp + 240]
            ; add rsp, 256
        );
        // Restore GPRs
        dynasm!(self.mc ; .arch x64
            ; pop r11 ; pop r10 ; pop r9 ; pop r8
            ; pop rdi ; pop rsi ; pop rdx ; pop rcx ; pop rax
        );
    }

    /// x86/assembler.py:2556 malloc_cond parity.
    fn genop_call_malloc_nursery(&mut self, op: &Op, result_loc: Option<&Loc>) {
        let size_ref = op.arg(0);
        let total_size = self
            .constants
            .get(&size_ref.0)
            .copied()
            .unwrap_or(size_ref.0 as i64);
        let gc_header_size = majit_gc::header::GcHeader::SIZE as i64;
        // gc.py:525-531 — read nursery slot addresses from the active GC
        // descriptor (cpu.gc_ll_descr.get_nursery_free_addr() parity), not
        // from a process-global singleton.
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();

        let nf = nf_addr as i64;
        let nt = nt_addr as i64;

        // ecx = nursery_free, edx = new nursery_free
        dynasm!(self.mc ; .arch x64
            ; mov rcx, QWORD nf
            ; mov rcx, [rcx]
            ; lea rdx, [rcx + total_size as i32]
            ; mov rax, QWORD nt
            ; cmp rdx, [rax]
        );

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; ja =>slow_path);

        // Fast path: update nursery_free, zero header, compute obj ptr
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD nf
            ; mov [rax], rdx
            ; mov QWORD [rcx], 0       // zero GcHeader
            ; lea rax, [rcx + gc_header_size as i32]
        );
        dynasm!(self.mc ; .arch x64 ; jmp =>done);

        // Slow path: x86/assembler.py:2553 push_gcmap + call
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
        // Push gcmap captured at regalloc time (via RegAllocOp::Perform.gcmap)
        // so the collector can trace live frame-resident Refs through the
        // slow-path call. Fall back to NULL when not provided (degenerate).
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
        }
        let slowpath_fn = crate::runner::dynasm_nursery_slowpath as *const () as i64;
        dynasm!(self.mc ; .arch x64
            ; mov rdi, QWORD total_size
            ; mov rax, QWORD slowpath_fn
            ; call rax
        );
        // pop_gcmap
        dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);

        dynasm!(self.mc ; .arch x64 ; =>done);
        if let Some(Loc::Reg(r)) = result_loc {
            if r.value != 0 {
                let rv = r.value;
                dynasm!(self.mc ; .arch x64 ; mov Rq(rv), rax);
            }
        }
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// NEW: allocate a fixed-size object. Requires GC runtime.
    /// Emits a trap (UD2/BRK) until GC nursery allocation is wired.
    fn genop_new(&mut self, op: &Op) {
        // Simple allocation: call libc malloc(obj_size).
        // RPython uses GC nursery bump allocation; we use malloc as stub.
        let obj_size = op
            .descr
            .as_ref()
            .and_then(|d| d.as_size_descr())
            .map(|sd| sd.size())
            .unwrap_or(16) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        // Call malloc(obj_size)
        dynasm!(self.mc ; .arch x64
            ; mov rdi, QWORD obj_size
            ; mov rax, QWORD malloc_ptr
            ; call rax
        );
        // rax/x0 = pointer to allocated memory
        // Zero-initialize
        dynasm!(self.mc ; .arch x64
            ; push rax           // save ptr
            ; mov rdi, rax       // dest = ptr
            ; xor esi, esi       // val = 0
            ; mov rdx, QWORD obj_size // size
            ; mov rax, QWORD (libc::memset as *const () as i64)
            ; call rax
            ; pop rax            // restore ptr
        );
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// NEW_WITH_VTABLE: allocate and set vtable pointer.
    fn genop_new_with_vtable(&mut self, op: &Op) {
        // Same as New, but also write vtable at offset 0.
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
        let malloc_ptr = libc::malloc as *const () as i64;
        dynasm!(self.mc ; .arch x64
            ; mov rdi, QWORD obj_size
            ; mov rax, QWORD malloc_ptr
            ; call rax
            ; push rax
            ; mov rdi, rax
            ; xor esi, esi
            ; mov rdx, QWORD obj_size
            ; mov rax, QWORD (libc::memset as *const () as i64)
            ; call rax
            ; pop rax
        );
        // Write vtable at offset 0
        if vtable != 0 {
            dynasm!(self.mc ; .arch x64
                ; mov rcx, QWORD vtable
                ; mov [rax], rcx
            );
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
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, rbp
        );
        self.store_rax_to_result(op.pos);
    }

    /// STRLEN / UNICODELEN: result = string.length
    /// Load the length field from the string/unicode object header.
    /// arg0 = string pointer. The length is at a fixed offset in the
    /// RPython string representation. For RPython strings, the length
    /// is typically at offset 8 (after the GC header / hash field).
    fn genop_strlen(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        self.load_arg_to_rax(op.arg(0));

        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + offset]
        );

        self.store_rax_to_result(op.pos);
    }

    /// STRGETITEM / UNICODEGETITEM: result = string[index]
    /// arg0 = string pointer, arg1 = index.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:295-306` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1; UNICODE does not.
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

        // Address = rax + base_size + rcx * item_size
        if item_size != 1 {
            dynasm!(self.mc
                ; .arch x64
                ; imul rcx, rcx, item_size
            );
        }
        dynasm!(self.mc
            ; .arch x64
            ; add rax, base_size
            ; add rax, rcx
        );
        match item_size {
            1 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, BYTE [rax]
            ),
            2 => dynasm!(self.mc
                ; .arch x64
                ; movzx eax, WORD [rax]
            ),
            4 => dynasm!(self.mc
                ; .arch x64
                ; mov eax, [rax]
            ),
            _ => dynasm!(self.mc
                ; .arch x64
                ; mov rax, [rax]
            ),
        }

        self.store_rax_to_result(op.pos);
    }

    // ================================================================
    // assembler.py:1817 genop_save_exc_class / genop_save_exception
    // ================================================================

    /// assembler.py:1817 genop_save_exc_class — stub: returns 0.
    fn genop_save_exc_class(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// assembler.py:1827 genop_save_exception — stub: returns 0.
    fn genop_save_exception(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
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
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
        );
        self.store_rax_to_result(op.pos);
    }

    /// INT_MOD: result = arg0 % arg1 (signed)
    fn genop_int_mod(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
            ; mov rax, rdx
        );
        self.store_rax_to_result(op.pos);
    }

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch x64
            ; mul rcx
            ; mov rax, rdx
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
            let sh = shift as i8;
            dynasm!(self.mc ; .arch x64
                ; shl rax, sh
                ; sar rax, sh
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
        let mask: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD mask
            ; movq xmm1, rax
            ; andpd xmm0, xmm1
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

        dynasm!(self.mc ; .arch x64 ; ucomisd xmm0, xmm1);
        match op.opcode {
            OpCode::FloatLt | OpCode::FloatGt => {
                dynasm!(self.mc ; .arch x64 ; seta al ; movzx eax, al);
            }
            OpCode::FloatLe | OpCode::FloatGe => {
                dynasm!(self.mc ; .arch x64 ; setae al ; movzx eax, al);
            }
            OpCode::FloatEq => {
                dynasm!(self.mc ; .arch x64
                    ; sete al ; setnp cl ; and al, cl ; movzx eax, al
                );
            }
            OpCode::FloatNe => {
                dynasm!(self.mc ; .arch x64
                    ; setne al ; setp cl ; or al, cl ; movzx eax, al
                );
            }
            _ => {
                dynasm!(self.mc ; .arch x64 ; sete al ; movzx eax, al);
            }
        }
        dynasm!(self.mc ; .arch x64 ; test rax, rax);
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.is_none() {
            self.store_rax_to_result(op.pos);
        }
    }

    /// CAST_FLOAT_TO_SINGLEFLOAT: f64 → f32 (bits in lower 32 of i64)
    fn genop_cast_float_to_singlefloat(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0));
        dynasm!(self.mc ; .arch x64
            ; cvtsd2ss xmm0, xmm0
            ; movd eax, xmm0
        );
        self.store_rax_to_result(op.pos);
    }

    /// CAST_SINGLEFLOAT_TO_FLOAT: f32 (bits in lower 32) → f64
    fn genop_cast_singlefloat_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        dynasm!(self.mc ; .arch x64
            ; movd xmm0, eax
            ; cvtss2sd xmm0, xmm0
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
            (1, true) => dynasm!(self.mc ; .arch x64 ; movsx rax, BYTE [rax]),
            (2, true) => dynasm!(self.mc ; .arch x64 ; movsx rax, WORD [rax]),
            (4, true) => dynasm!(self.mc ; .arch x64
                ; mov eax, [rax]
                ; cdqe
            ),
            (1, false) => dynasm!(self.mc ; .arch x64 ; movzx eax, BYTE [rax]),
            (2, false) => dynasm!(self.mc ; .arch x64 ; movzx eax, WORD [rax]),
            (4, false) => dynasm!(self.mc ; .arch x64 ; mov eax, [rax]),
            _ => dynasm!(self.mc ; .arch x64 ; mov rax, [rax]),
        }
    }

    /// Emit a sized store of rcx/x1 to [rax]/[x0].
    fn emit_store_to_rax_sized(&mut self, size: usize) {
        match size {
            1 => dynasm!(self.mc ; .arch x64 ; mov [rax], cl),
            2 => dynasm!(self.mc ; .arch x64 ; mov [rax], cx),
            4 => dynasm!(self.mc ; .arch x64 ; mov [rax], ecx),
            _ => dynasm!(self.mc ; .arch x64 ; mov [rax], rcx),
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
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);

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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
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
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch x64 ; pop rax);
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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
        }
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// RAW_LOAD_I/F: load from base + offset using descriptor.
    fn genop_raw_load(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);

        self.emit_load_from_rax_sized(size as i32);
        let _ = offset; // offset is in the descriptor, not used for raw_load
        self.store_rax_to_result(op.pos);
    }

    /// RAW_STORE: store value to base + offset using descriptor.
    fn genop_discard_raw_store(&mut self, op: &Op) {
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(1));
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch x64 ; pop rax);
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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, total_offset
            ; add rax, rcx
        );

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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, total_offset
            ; add rax, rcx
        );
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch x64 ; pop rax);

        self.emit_store_to_rax_sized(field_size);
    }

    // ================================================================
    // genop_* — call variants
    // ================================================================

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    fn genop_discard_cond_call(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; test rax, rax ; jz =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch x64 ; =>skip_label);
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    fn genop_cond_call_value(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0));
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; test rax, rax ; jnz =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch x64 ; =>skip_label);

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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, base_size ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2));
        dynasm!(self.mc ; .arch x64 ; pop rax);
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
            dynasm!(self.mc ; .arch x64
                ; mov rcx, QWORD item_size
                ; imul rax, rcx
            );
        }
        dynasm!(self.mc ; .arch x64 ; push rax); // [rsp] = byte_count

        // Compute src_addr = src + base_size + src_start * item_size
        self.load_arg_to_rax(op.arg(0));
        self.load_arg_to_rcx(op.arg(2));
        if item_size != 1 {
            dynasm!(self.mc ; .arch x64
                ; mov rdx, QWORD item_size
                ; imul rcx, rdx
            );
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, DWORD base_size as i32
            ; add rax, rcx
            ; push rax  // [rsp] = src_addr
        );

        // Compute dst_addr = dst + base_size + dst_start * item_size
        self.load_arg_to_rax(op.arg(1));
        self.load_arg_to_rcx(op.arg(3));
        if item_size != 1 {
            dynasm!(self.mc ; .arch x64
                ; mov rdx, QWORD item_size
                ; imul rcx, rdx
            );
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, DWORD base_size as i32
            ; add rax, rcx
        );

        // memmove(dst_addr, src_addr, byte_count)
        let memmove_ptr = libc::memmove as *const () as i64;
        dynasm!(self.mc ; .arch x64
            ; mov rdi, rax   // dst
            ; pop rsi        // src
            ; pop rdx        // count
            ; push rbp
            ; mov rax, QWORD memmove_ptr
            ; call rax
            ; pop rbp
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

        // Save length, compute total_size = base_size + length * item_size
        dynasm!(self.mc ; .arch x64
            ; push rax                           // save length
            ; imul rax, rax, item_size as i32
            ; add rax, base_size as i32
            ; mov rdi, rax                       // malloc arg
            ; push rax                           // save total_size
            ; mov rax, QWORD malloc_ptr
            ; call rax
            ; pop rcx                            // rcx = total_size
            ; push rax                           // save ptr
            // memset(ptr, 0, total_size)
            ; mov rdi, rax
            ; xor esi, esi
            ; mov rdx, rcx
            ; mov rax, QWORD memset_ptr
            ; call rax
            ; pop rax                            // rax = ptr
            ; pop rcx                            // rcx = length
            // Store length at offset 8 (RPython string header)
            ; mov [rax + 8], rcx
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
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale_start as i32);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, DWORD base_size as i32
            ; add rax, rcx
            ; push rax                           // save dest
        );
        self.load_arg_to_rax(op.arg(2)); // size
        if scale_size != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rax, rax, scale_size as i32);
        }
        // memset(dest, 0, byte_length)
        dynasm!(self.mc ; .arch x64
            ; mov rdx, rax                       // byte_length
            ; pop rdi                            // dest
            ; xor esi, esi                       // val = 0
            ; push rbp
            ; mov rax, QWORD memset_ptr
            ; call rax
            ; pop rbp
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
            dynasm!(self.mc ; .arch x64 ; shl rcx, BYTE shift as i8);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if baseofs != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, baseofs);
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
