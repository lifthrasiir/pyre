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
use majit_ir::VecAssoc;
use std::sync::Arc;

// x86/assembler.py parity: x86_64-only backend.
use dynasmrt::x64::Assembler;
use dynasmrt::{AssemblyOffset, DynamicLabel, DynasmApi, DynasmLabelApi, ExecutableBuffer, dynasm};

use majit_backend::{BackendError, ExitFrameLayout, ExitRecoveryLayout, ExitValueSourceLayout};
use majit_ir::{FailDescr, InputArg, Op, OpCode, OpRc, OpRef, OpTypeIndex, TargetArgLoc, Type};

use crate::arch::*;
use crate::codebuf;
use crate::gcmap::{allocate_gcmap, gcmap_set_bit};
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

#[derive(Clone, Copy)]
enum AbiArgPlacement {
    Gpr(u8),
    Xmm(u8),
    Stack(i32),
}

/// `x86/assembler.py:254 _push_all_regs_to_frame` parity — free-fn
/// variant that emits into an arbitrary `dynasmrt::x64::Assembler`,
/// for use by helper-buffer builders that operate outside of an
/// `Assembler386` instance (e.g. `_build_malloc_slowpath`,
/// `_build_wb_slowpath`).  Logic is identical to
/// `Assembler386::push_all_regs_to_jitframe`; the per-arch slot table
/// and `FIRST_ITEM_OFFSET`-relative addressing match.
pub(crate) fn push_all_regs_to_jitframe_raw(
    asm: &mut Assembler,
    ignored_regs: &[crate::regloc::RegLoc],
    withfloats: bool,
) {
    for reg in crate::x86::regalloc::ALL_CORE_REGS.iter() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let slot = core_reg_position(*reg).expect("push_all_regs: managed x86_64 GPR");
        let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
        dynasm!(asm ; .arch x64 ; mov [rbp + ofs], Rq(reg.value));
    }
    if withfloats {
        for reg in crate::x86::regalloc::ALL_FLOAT_REGS.iter() {
            let slot = float_reg_position(*reg).expect("push_all_regs: managed x86_64 XMM");
            let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
            dynasm!(asm ; .arch x64 ; movsd [rbp + ofs], Rx(reg.value));
        }
    }
}

/// `x86/assembler.py:283 _pop_all_regs_from_frame` parity — free-fn
/// variant; see `push_all_regs_to_jitframe_raw` for usage notes.
pub(crate) fn pop_all_regs_from_jitframe_raw(
    asm: &mut Assembler,
    ignored_regs: &[crate::regloc::RegLoc],
    withfloats: bool,
) {
    for reg in crate::x86::regalloc::ALL_CORE_REGS.iter() {
        if ignored_regs.contains(reg) {
            continue;
        }
        let slot = core_reg_position(*reg).expect("pop_all_regs: managed x86_64 GPR");
        let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
        dynasm!(asm ; .arch x64 ; mov Rq(reg.value), [rbp + ofs]);
    }
    if withfloats {
        for reg in crate::x86::regalloc::ALL_FLOAT_REGS.iter() {
            let slot = float_reg_position(*reg).expect("pop_all_regs: managed x86_64 XMM");
            let ofs = FIRST_ITEM_OFFSET as i32 + (slot * WORD) as i32;
            dynasm!(asm ; .arch x64 ; movsd Rx(reg.value), [rbp + ofs]);
        }
    }
}

/// `x86/assembler.py:1130 _call_footer_shadowstack` parity (free-fn
/// variant for use outside of an `Assembler386` borrow — used by
/// `emit_call_footer_raw`, which the standalone propagate / malloc
/// slowpath trampolines reach for).  Subtracts `2 * WORD` from the
/// shadow-stack top, undoing the `gen_shadowstack_header` push that
/// the trace prologue emitted.
pub(crate) fn emit_footer_shadowstack_raw(asm: &mut Assembler) {
    let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
    dynasm!(asm ; .arch x64
        ; mov Rq(scratch), QWORD rst
        ; sub QWORD [Rq(scratch)], 16
    );
}

/// `x86/assembler.py:1093 _call_footer` parity (free-fn variant).
/// Restores the callee-save set established by `_call_header` and
/// returns the jitframe pointer in `rax`.  Must be entered with `rsp`
/// at trace-body alignment (i.e. the same value the trace's
/// `_call_header` left after its `SUB rsp, FRAME_FIXED_SIZE`).
pub(crate) fn emit_call_footer_raw(asm: &mut Assembler) {
    emit_footer_shadowstack_raw(asm);
    dynasm!(asm ; .arch x64 ; mov rax, rbp);
    // Win64: 8 callee-save GPRs × 8 = 64 bytes.  PyPy's
    // `arch.py:43` notes "never use r13 on Win64", but pyre's
    // `genop_call_assembler` uses r13 as a scratch that survives
    // a `free()` call (no other unused callee-save reg is live
    // enough at that point), so pyre adds r13 to the saved set —
    // filling the slot that was 8-byte padding under PyPy's
    // "5 regs + r14/r15 in shadow store + 12 pad" layout.
    #[cfg(target_os = "windows")]
    dynasm!(asm ; .arch x64
        ; mov rbx, [rsp + 0]
        ; mov rsi, [rsp + 8]
        ; mov rdi, [rsp + 16]
        ; mov r12, [rsp + 24]
        ; mov r14, [rsp + 32]
        ; mov r15, [rsp + 40]
        ; mov rbp, [rsp + 48]
        ; mov r13, [rsp + 56]
        ; add rsp, 72
    );
    #[cfg(not(target_os = "windows"))]
    dynasm!(asm ; .arch x64
        ; mov rbx, [rsp + 0]
        ; mov r12, [rsp + 8]
        ; mov r13, [rsp + 16]
        ; mov r14, [rsp + 24]
        ; mov r15, [rsp + 32]
        ; mov rbp, [rsp + 40]
        ; add rsp, 56
    );
    dynasm!(asm ; .arch x64 ; ret);
}

/// `assembler.py:328 _build_propagate_exception_path` parity —
/// pure builder.  Caching/ownership is the caller's responsibility:
/// `X86CpuExt::ensure_propagate_exception_path` (`x86/cpu_ext.rs`)
/// stores the resulting address in its `propagate_exception_path`
/// field, matching PyPy's `self.propagate_exception_path` attribute
/// on `Assembler386`.
///
/// **Calling convention (matches PyPy line 328-345):**
/// - Entry: reached only via `JMP` (not `CALL`) from a slowpath that
///   has already restored `rsp` to the trace body's alignment level
///   (i.e. the same value the trace's `_call_header` left after its
///   SUB).  `rbp` = jitframe (possibly reloaded after a GC move).
///   No other register conventions are assumed: every callee-save
///   was already restored by the slowpath's `_pop_all_regs_from_frame`,
///   and the live trace-body values were spilled to the jitframe.
/// - Exit: `_call_footer` semantics — restores callee-save GPRs
///   from `[rsp+...]`, sets `rax = rbp` (jitframe pointer return),
///   adds the prologue's SUB back, and `RET`s to the function that
///   originally entered the trace (PyPy: the C JIT shim;
///   pyre: the same role via `Asm::_call_header`/`_call_footer`).
///
/// `propagate_exception_descr` is read from `cpu_handle` and baked
/// as an i64 immediate into `[rbp + JF_DESCR_OFS]`.  Caller must
/// guarantee the descr is installed (`MetaInterp::finish_setup`,
/// `pyjitpl.py:2283`) before invoking this builder; the build asserts
/// otherwise.  PyPy itself would silently bake 0 in this case (and
/// fail at `handle_fail` dispatch time); pyre prefers a build-time
/// fail-fast for the same invariant.
///
/// Returns `(buffer, entry_addr)`.  The caller (`X86CpuExt`) owns the
/// `ExecutableBuffer` for the lifetime of the per-CPU stash so the RX
/// page is unmapped when the CPU drops — matches PyPy's `asmmemmgr`,
/// which roots helper buffers on the CPU.
pub(crate) fn build_propagate_exception_path(
    cpu_handle: &crate::guard::CpuDescrHandle,
) -> (ExecutableBuffer, usize) {
    let mut asm = Assembler::new().expect("propagate_exception_path: new Assembler");
    let propagate_descr = cpu_handle
        .read()
        .unwrap()
        .descr_ptrs()
        .propagate_exception_descr as i64;
    assert!(
        propagate_descr != 0,
        "build_propagate_exception_path: cpu_handle.propagate_exception_descr \
         must be installed (pyjitpl.py:2283) before the trampoline is built \
         on this CPU"
    );
    // assembler.py:1826-1843 `_store_and_reset_exception(self.mc, eax)`:
    // read pos_exc_value into RAX, clear both globals.  On real OOM
    // pos_exc_value is typically already NULL — the propagate descr's
    // `handle_fail` raises MemoryError directly — but mirror the
    // structure regardless.
    let exc_value_addr = crate::jit_exc_value_addr() as i64;
    let exc_type_addr = crate::jit_exc_type_addr() as i64;
    dynasm!(asm ; .arch x64
        ; mov rax, QWORD exc_value_addr
        // Use R11 (X86_64_SCRATCH_REG) so RCX is free for the descr.
        ; mov r11, [rax]
        ; mov QWORD [rax], 0
        ; mov rax, QWORD exc_type_addr
        ; mov QWORD [rax], 0
        // assembler.py:335-337 — MOV [jf_guard_exc], pos_exc_value
        ; mov [rbp + JF_GUARD_EXC_OFS], r11
        // assembler.py:338-340 — MOV [jf_descr], propagate_descr
        ; mov rcx, QWORD propagate_descr
        ; mov [rbp + JF_DESCR_OFS], rcx
    );
    // assembler.py:342 `self._call_footer()` — restore callee-save,
    // set rax = rbp, ADD rsp, prologue_size, RET.
    emit_call_footer_raw(&mut asm);
    let buffer = asm.finalize().expect("propagate_exception_path: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// `assembler.py:231 _build_malloc_slowpath(kind='fixed')` parity —
/// pure builder.  Caching/ownership is the caller's responsibility:
/// `X86CpuExt::ensure_malloc_slowpath_fixed` (`x86/cpu_ext.rs`)
/// stores the resulting address in its `malloc_slowpath_fixed`
/// field, matching PyPy's `self.malloc_slowpath` attribute on
/// `Assembler386`.
///
/// **Calling convention (matches PyPy line 233-243):**
/// - Entry: `rbp` = jitframe, `rcx` = old `nursery_head`,
///   `rdx` = `nursery_head + total_size` (set by the per-call-site
///   fast path's `lea rdx, [rcx + total_size]`).  `total_size`
///   recovered at runtime via `sub rdx, rcx`.  The caller has already
///   pushed the gcmap to `[rbp + JF_GCMAP_OFS]`.
/// - Exit: `rcx` = payload pointer (or 0 on OOM), matching PyPy line
///   304 `MOV_rr(ecx, eax)`; `rbp` reloaded from shadow stack top in
///   case a minor GC fired and moved the frame; every other GPR/XMM
///   restored from the jitframe save area, including `rax` which is
///   reset to its pre-call value.
///
/// `ecx` and `edx` are the only registers excluded from
/// push_all_regs / pop_all_regs: `ecx` carries the return value
/// (assembler.py:247 `_push_all_regs_to_frame(mc, [ecx, edx], floats)`),
/// `edx` is the runtime size carrier and the caller's regalloc has
/// already spilled any live value out of both before the call via
/// `MALLOC_NURSERY_CLOBBER` (regalloc.rs:101).
///
/// On OOM (`rax == 0` after the helper call) the trampoline tail-jumps
/// to `propagate_path` — the standalone `_build_propagate_exception_path`
/// trampoline returned by `build_propagate_exception_path`.  Caller
/// (`X86CpuExt::ensure_malloc_slowpath_fixed`) builds the propagate
/// path first and threads its address here, matching PyPy's
/// `setup_once` ordering.
///
/// Returns `(buffer, entry_addr)`.  Same ownership rule as
/// `build_propagate_exception_path` — the caller (`X86CpuExt`) holds
/// the `ExecutableBuffer` so the RX page lives as long as the CPU.
pub(crate) fn build_malloc_slowpath_fixed(
    cpu_handle: &crate::guard::CpuDescrHandle,
    propagate_path: usize,
) -> (ExecutableBuffer, usize) {
    // `cpu_handle` is still threaded through for symmetry with PyPy
    // (where `_build_malloc_slowpath` reads several `self.cpu`-rooted
    // attrs).  The propagate descr itself is now baked once into the
    // standalone propagate trampoline, not re-baked here.
    let _ = cpu_handle;
    let mut asm = Assembler::new().expect("malloc_slowpath: new Assembler");

    // assembler.py:264 `SUB_rr(edx, ecx)` — recover total_size at
    // runtime.  Both `malloc_cond` and `malloc_cond_varsize_frame`
    // hand off `ecx = old_nursery_free` / `edx = old_nursery_free +
    // total_bytes`, so the same `SUB` recovers the byte count for
    // either caller (PyPy line 2554 / 2578 both route through this
    // shared `malloc_slowpath`).
    dynasm!(asm ; .arch x64 ; sub rdx, rcx);

    let slowpath_fn = crate::runner::dynasm_nursery_slowpath as *const () as i64;
    build_malloc_slowpath_body(&mut asm, slowpath_fn, propagate_path);

    let buffer = asm.finalize().expect("malloc_slowpath: finalize");
    let ptr = crate::codebuf::buffer_ptr(&buffer) as usize;
    (buffer, ptr)
}

/// Body of the single PyPy `malloc_slowpath` trampoline used by both
/// `CallMallocNursery` (fixed-size object alloc) and
/// `CallMallocNurseryVarsizeFrame` (JITFRAME alloc).  Emits the
/// `_push_all_regs_to_frame([ecx, edx])` → CALL helper → reload_frame
/// → inline WB → OOM check → `MOV ecx, eax` → `_pop_all_regs_from_frame`
/// → RET sequence.  The caller emits the `SUB rdx, rcx` size recovery
/// just above this helper (the only line that differs between fixed
/// and varsize_frame in the trampoline header, and PyPy emits it for
/// both kinds via the same `_build_malloc_slowpath('fixed')` arm —
/// PyPy line 264).
///
/// `slowpath_fn` is the absolute address of the GC helper called by
/// `MOV rax, imm64; CALL rax`.  ABI: size in ARG0 (RCX on Win64, RDI
/// on SysV).  Result in RAX.
fn build_malloc_slowpath_body(asm: &mut Assembler, slowpath_fn: i64, propagate_path: usize) {
    let ignored = [crate::regloc::ECX, crate::regloc::EDX];

    // assembler.py:247 `_push_all_regs_to_frame(mc, [ecx, edx], floats)`.
    // Saves every managed GPR/XMM except ECX/EDX so the inner CALL's
    // caller-clobber set is contained.  EAX is included in the save
    // set (unlike pyre's prior `[EAX, EDX]` mask), preserving any
    // live caller value across the slowpath — the regalloc only
    // promises ECX/EDX clobber to the caller.
    push_all_regs_to_jitframe_raw(asm, &ignored, true);

    // assembler.py:258-261 `add_to_esp = 16 - WORD` plus Win64 shadow
    // space.  pyre's JIT body is 0-mod-16 (per `_call_header`'s
    // padding-slot SUB) and the outer `call rax` pushes the return
    // address, leaving rsp 8-mod-16 inside the trampoline.  The SUB
    // below brings rsp back to 0-mod-16 before the inner CALL — `8`
    // alignment alone on SysV, `8 + 32` shadow on Win64.
    #[cfg(target_os = "windows")]
    let align: i32 = 40;
    #[cfg(not(target_os = "windows"))]
    let align: i32 = 8;

    // assembler.py:270 `MOV_rr(ARG0, edx)` — size argument.
    #[cfg(target_os = "windows")]
    let arg0_reg: u8 = 1; // rcx
    #[cfg(not(target_os = "windows"))]
    let arg0_reg: u8 = 7; // rdi

    if align != 0 {
        dynasm!(asm ; .arch x64 ; sub rsp, align);
    }
    dynasm!(asm ; .arch x64
        ; mov Rq(arg0_reg), rdx
        ; mov rax, QWORD slowpath_fn
        ; call rax
    );
    if align != 0 {
        dynasm!(asm ; .arch x64 ; add rsp, align);
    }

    // assembler.py:296 `_reload_frame_if_necessary(mc)` — rebind rbp
    // from the shadow stack in case a minor GC moved the jitframe.
    // ECX is used as scratch here (matches PyPy `_reload_frame_if_necessary`
    // assembler.py:1375); the helper return value still lives in RAX at
    // this point and is moved into ECX only after the reload (and after
    // the WB inline below, which preserves RAX via push/pop).
    let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
    dynasm!(asm ; .arch x64
        ; mov rcx, QWORD rst_addr
        ; mov rcx, [rcx]
        ; mov rbp, [rcx - 8]
    );

    // assembler.py:_reload_frame_if_necessary line 1376 (Win64 + Linux
    // share): non-array write barrier on the reloaded jf so subsequent
    // Ref writes into frame slots are tracked by minor GC.  The barrier
    // body is conditional on the GC exposing a write-barrier descr.
    let wb_descr = crate::runner::DYNASM_ACTIVE_GC.with(|cell| {
        cell.borrow()
            .as_ref()
            .and_then(|gc| gc.get_write_barrier_descr())
    });
    if let Some(wb) = wb_descr {
        let byteofs = wb.jit_wb_if_flag_byteofs;
        let mask = wb.jit_wb_if_flag_singlebyte as i8;
        let skip_wb = asm.new_dynamic_label();
        dynasm!(asm ; .arch x64
            ; test BYTE [rbp + byteofs], mask
            ; jz =>skip_wb
        );
        // Inline WB helper call: rbp -> ARG0, save/restore rax across.
        // Stack accounting at this point: the trampoline's pre-call
        // alignment SUB has been fully reversed by the matching ADD,
        // so rsp is back at the trampoline-entry value of 8-mod-16
        // (pyre JIT body 0-mod-16 + outer-CALL push).  `push rax`
        // brings rsp to 0-mod-16 — already aligned for the inner CALL
        // on SysV, while Win64 still needs its 32-byte shadow space.
        let wb_fn = crate::runner::dynasm_write_barrier as *const () as i64;
        #[cfg(target_os = "windows")]
        dynasm!(asm ; .arch x64
            ; push rax
            ; sub rsp, 32           // 32 shadow (push rax already aligned)
            ; mov rcx, rbp
            ; mov rax, QWORD wb_fn
            ; call rax
            ; add rsp, 32
            ; pop rax
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(asm ; .arch x64
            ; push rax
            ; mov rdi, rbp
            ; mov rax, QWORD wb_fn
            ; call rax
            ; pop rax
        );
        dynasm!(asm ; .arch x64 ; =>skip_wb);
    }

    // assembler.py:298-322 — TEST/JZ-to-OOM-tail with the common (success)
    // case as fall-through, matching PyPy's branch layout exactly.  The
    // `JZ` lands on the OOM tail emitted after RET; the fall-through path
    // does `MOV ecx, eax` → `_pop_all_regs_from_frame` → `pop_gcmap` →
    // `RET`.
    let oom_tail = asm.new_dynamic_label();
    dynasm!(asm ; .arch x64
        ; test rax, rax
        ; jz =>oom_tail
    );

    // assembler.py:304 `MOV_rr(ecx, eax)` — deliver the helper return
    // value through ECX so it survives `pop_all` (which restores RAX
    // from the save area).  RAX is still valid at this point because
    // the WB inline above brackets its inner CALL with `push rax /
    // pop rax`.
    dynasm!(asm ; .arch x64 ; mov rcx, rax);

    // assembler.py:307 `_pop_all_regs_from_frame(mc, [ecx, edx], floats)`.
    pop_all_regs_from_jitframe_raw(asm, &ignored, true);
    // assembler.py:308 `self.pop_gcmap(mc)` — clear `JF_GCMAP_OFS`
    // before RET so the caller's regalloc layout (which never sees the
    // trampoline's saved-reg slots) is the only gcmap the next
    // collecting call walks.  Matches PyPy "trampoline owns the
    // gcmap-clear before RET" structure exactly.
    dynasm!(asm ; .arch x64
        ; mov QWORD [rbp + crate::jitframe::JF_GCMAP_OFS], 0
        ; ret
    );

    // assembler.py:309-322 OOM tail — patched JZ target above.  When the
    // slowpath helper returns NULL (`libc::calloc` / `gc.alloc_nursery_*`
    // OOM) PyPy tail-JMPs to the standalone `propagate_exception_path`
    // (line 322).  Stage `propagate_path` into a scratch register and
    // jump indirectly so the transfer is range-independent: dynasm-rs's
    // `jmp extern` would lower to `E9 + rel32`, but the malloc slowpath
    // and propagate trampoline live in separately-finalized
    // `ExecutableBuffer`s that aren't guaranteed to land within ±2GB
    // under ASLR, so `finalize()` would otherwise fail with
    // `ImpossibleRelocation` and trip the caller's `expect("malloc_slowpath:
    // finalize")` on affected layouts.
    let propagate_scratch = crate::regloc::X86_64_SCRATCH_REG.value;
    dynasm!(asm ; .arch x64
        ; =>oom_tail
        // assembler.py:321 `ADD esp, WORD` — pop the trampoline's own
        // CALL return address so `_call_footer` in the propagate
        // trampoline sees rsp at the trace's body alignment (the same
        // value the trace's `_call_header` left after its SUB).
        ; add rsp, 8
        ; mov Rq(propagate_scratch), QWORD propagate_path as i64
        ; jmp Rq(propagate_scratch)
    );
}

/// Pointer-identity key for `target_tokens_currently_compiling`. PyPy
/// x86/assembler.py:93 keys it by the descr Python object itself; we use
/// the underlying allocation address of the `Arc<dyn Descr>` so two
/// distinct TargetToken descriptors are never confused.
fn loop_target_id(op: &Op) -> Option<usize> {
    op.getdescr().as_ref().map(majit_ir::descr_identity)
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

fn core_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    crate::x86::regalloc::ALL_CORE_REGS
        .iter()
        .position(|candidate| *candidate == reg)
}

fn float_reg_position(reg: crate::regloc::RegLoc) -> Option<usize> {
    crate::x86::regalloc::ALL_FLOAT_REGS
        .iter()
        .position(|candidate| *candidate == reg)
        .map(|idx| crate::x86::regalloc::ALL_CORE_REGS.len() + idx)
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
///
/// Borrows the trace's `inputargs` and `operations` for its lifetime so
/// `OpRef → Type` resolves through `op.type_` / `inputarg.tp` directly
/// (RPython `box.type` parity); no `value_types: HashMap` side-table.
pub struct Assembler386<'a> {
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
    /// Fail descriptors built during assembly — wrapped in `FailDescrCell`
    /// so `Arc::as_ptr` is a thin pointer suitable for direct
    /// `Arc::from_raw` recovery (`history.py:113 AbstractDescr.show`).
    fail_descrs: Vec<std::sync::Arc<majit_ir::FailDescrCell>>,
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
    /// Maps OpRef → jitframe slot index.
    opref_to_slot: VecAssoc<OpRef, usize>,
    /// Trace inputargs — borrowed for `opref_type` lookups.
    inputargs: &'a [InputArg],
    /// Trace operations — borrowed for `opref_type` lookups (reads
    /// `op.type_` directly, RPython `box.type` parity).
    operations: &'a [Op],
    /// `inputarg_pos[arg.index] = idx in inputargs`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots. Mirrors
    /// `OpTypeIndex::inputarg_pos`.
    inputarg_pos: Vec<u32>,
    /// `op_pos[op.pos.raw()] = idx in operations`, sentinel
    /// [`OpTypeIndex::NO_POS`] for unset slots and Void/None ops.
    /// Mirrors `OpTypeIndex::op_pos`.
    op_pos: Vec<u32>,
    /// Constants: OpRef index (>= 10000) → i64 value.
    constants: majit_ir::VecAssoc<u32, i64>,
    /// Constant type annotations for float immediates and fail args.
    constant_types: majit_ir::VecAssoc<u32, Type>,
    /// Next available frame slot index.
    next_slot: usize,
    /// Condition code from the most recent CMP/TEST instruction,
    /// consumed by a following GUARD_TRUE/GUARD_FALSE.
    /// Stores an abstract condition code (CC_* constants).
    guard_success_cc: Option<u8>,
    /// x86/assembler.py:93 target_tokens_currently_compiling parity.
    /// Keyed by descriptor pointer identity (PyPy uses Python `is`).
    target_tokens_currently_compiling: VecAssoc<usize, DynamicLabel>,
    compiled_target_tokens: Vec<majit_ir::DescrRef>,
    /// llmodel.py:64-69 self.vtable_offset — typeptr field byte offset.
    /// `None` corresponds to RPython's gcremovetypeptr config.
    vtable_offset: Option<usize>,
    /// llsupport/gc.py:563 vtable→typeid table, materialized by the runner
    /// via gc_ll_descr.get_typeid_from_classptr_if_gcremovetypeptr. Used by
    /// the gcremovetypeptr branch of `_cmp_guard_class`.
    classptr_to_typeid: VecAssoc<i64, u32>,
    /// TYPE_INFO / CLASSTYPE constants for `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`, fetched by the runner from the active gc_ll_descr.
    guard_gc_type_info: Option<GuardGcTypeInfo>,
    /// Constant classptr → `(subclassrange_min, subclassrange_max)`, matching
    /// `loc_check_against_class.getint()` field reads in
    /// `x86/assembler.py:1971-1974`.
    classptr_to_subclass_range: VecAssoc<i64, (i64, i64)>,
    /// Dynamic label at the function entry for self-recursive CALL_ASSEMBLER.
    self_entry_label: Option<DynamicLabel>,
    /// Leaked pointer holding the resolved entry address for self-recursive
    /// CALL_ASSEMBLER via the execute trampoline. Written after finalization.
    self_entry_addr_ptr: *mut usize,
    /// assembler.py:320 descr._ll_function_addr parity:
    /// Maps call_target_token → compiled code address for CALL_ASSEMBLER.
    /// Populated by the runner before compilation, from registered loop targets.
    call_assembler_targets: VecAssoc<u64, usize>,
    /// opassembler.py:1177 _finish_gcmap.
    finish_gcmap: Option<*mut usize>,
    /// opassembler.py:1215 gcmap_for_finish.
    gcmap_for_finish: *mut usize,
    /// assembler.py:2207 _store_force_index parity:
    /// Pre-allocated fail descr for the next GUARD_NOT_FORCED, created
    /// at CALL_ASSEMBLER emission time so we can store its pointer to
    /// jf_force_descr before the call. Consumed by the subsequent
    /// GUARD_NOT_FORCED guard emission.
    pending_force_descr: Option<majit_ir::DescrRef>,
    /// Pre-wrapped `FailDescrCell` for the same pending guard.  Codegen
    /// bakes the cell's thin pointer into `JF_FORCE_DESCR_OFS` so that
    /// `force_token_to_dead_frame` (cranelift/compiler.rs:2660) can
    /// recover the descr via `recover_fail_descr_cell` without the
    /// fat-pointer mismatch a bare `Arc<dyn Descr>` ptr would cause.
    /// The same cell is consumed by `append_guard_token_with_faillocs`
    /// so jf_force_descr and jf_descr resolve to the same identity.
    pending_force_cell: Option<std::sync::Arc<majit_ir::FailDescrCell>>,
    /// `compile.py:665-674` + `pyjitpl.py:2283`: construction-time
    /// snapshot of the six descr pointers attached to the owning cpu
    /// instance.  Retained for constructor signature stability across
    /// runner / call_assembler callsites — emission helpers
    /// (`done_with_this_frame_descr_ptr_for_type`,
    /// `exit_frame_with_exception_descr_ref_ptr`,
    /// `propagate_exception_descr_ptr`) read live from `cpu_handle`
    /// instead so the raw ptr baked into `JF_DESCR_OFS` and the Arc
    /// stamped into `meta_descr` come from the same snapshot.
    #[allow(dead_code)]
    attached_descrs: crate::guard::AttachedDescrPtrs,
    /// `Arc` clone of the owning cpu's attachment handle.  Its heap
    /// address is baked into the CALL_ASSEMBLER helper call site as a
    /// compile-time immediate (`Arc::as_ptr`) and the `Arc` is moved
    /// into the resulting `CompiledCode` so the pointee outlives any
    /// subsequent `DynasmBackend` drop — same role as RPython's
    /// `self.cpu` attribute-access after whole-program translation,
    /// where the `cpu` object's identity is guaranteed by Python.
    cpu_handle: crate::guard::CpuDescrHandle,
    /// `assembler.py:94 setup()` `self.frame_depth_to_patch = []` —
    /// list of code-buffer byte offsets at which a placeholder 32-bit
    /// `0xffffff` was written for the stack-depth check / slowpath
    /// trampoline.  After materialisation, `patch_stack_checks`
    /// overwrites each entry with the final frame depth.
    ///
    /// Each entry stores an offset *relative to the start of the
    /// machine-code buffer* (i.e. pre-`rawstart`); `patch_stack_checks`
    /// adds `rawstart` to obtain the absolute address.
    frame_depth_to_patch: Vec<usize>,
    /// assembler.py:1003-1008 `_assemble`: the frame depth of a cross-loop
    /// JUMP target (its `target_frame_depth`), or 0 when the trace has no
    /// external JUMP.  The closing `JMP` enters the target loop's body,
    /// which may use deeper frame slots than this trace; `_assemble` grows
    /// `frame_depth` to fit so a bridge's prologue `_check_frame_depth`
    /// reallocs the in-flight JITFRAME large enough before the `JMP`.
    jump_target_frame_depth: usize,
    /// `x86/assembler.py:63` `self.malloc_slowpath` parity — entry
    /// pointer of the per-CPU malloc slowpath trampoline used by both
    /// `CallMallocNursery` and `CallMallocNurseryVarsizeFrame` (PyPy
    /// line 2554 / 2578 both route through the same `malloc_slowpath`).
    /// Resolved by `X86CpuExt::ensure_malloc_slowpath_fixed` and
    /// passed in at construction time so the emit path bakes it as a
    /// 64-bit immediate without re-touching the backend.
    malloc_slowpath_fixed: usize,
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
    /// The fail descriptor cell for this guard.  `Arc::as_ptr(&fail_descr)`
    /// is the thin pointer baked into `jf_descr`; the same cell instance is
    /// stored on `Assembler386::fail_descrs` so registration on the owning CLT
    /// keeps it alive while the recovery stub references its address.
    fail_descr: std::sync::Arc<majit_ir::FailDescrCell>,
    /// Fail argument OpRefs for recovery (to save to sequential output slots).
    fail_args: Vec<OpRef>,
    /// regalloc parity: snapshot of opref_to_slot at guard emission time.
    /// Needed by recovery stubs to read fail_args from correct slots.
    opref_to_slot_snapshot: VecAssoc<OpRef, usize>,
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
    pub fail_descrs: Box<[std::sync::Arc<majit_ir::FailDescrCell>]>,
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
    /// `None` for root loops; bridges set `(source_trace_id, source_fail_index_per_trace)`.
    pub source_guard: Option<(u64, u32)>,
}

impl<'a> Assembler386<'a> {
    /// rpython/jit/metainterp/history.py:220 `box.type` parity.
    /// Single source of truth: `op.type_` for ops, `inputarg.tp` for
    /// inputargs, `constant_types` for constants.
    #[inline]
    fn opref_type(&self, opref: OpRef) -> Option<Type> {
        self.opref_type_at(opref, None)
    }

    #[inline]
    fn opref_type_at(&self, opref: OpRef, at_op_index: Option<usize>) -> Option<Type> {
        let type_index = OpTypeIndex::from_parts(
            self.inputargs,
            self.operations,
            &self.inputarg_pos,
            &self.op_pos,
        );
        match at_op_index {
            Some(at) => type_index.opref_type_at(opref, at),
            None => type_index.opref_type(opref),
        }
    }
    /// assembler.py:54 __init__
    pub(crate) fn new(
        trace_id: u64,
        header_pc: u64,
        constants: majit_ir::VecAssoc<u32, i64>,
        vtable_offset: Option<usize>,
        classptr_to_typeid: VecAssoc<i64, u32>,
        guard_gc_type_info: Option<GuardGcTypeInfo>,
        classptr_to_subclass_range: VecAssoc<i64, (i64, i64)>,
        attached_descrs: crate::guard::AttachedDescrPtrs,
        cpu_handle: crate::guard::CpuDescrHandle,
        malloc_slowpath_fixed: usize,
        inputargs: &'a [InputArg],
        operations: &'a [Op],
    ) -> Self {
        let inputarg_pos = OpTypeIndex::build_inputarg_pos(inputargs);
        let op_pos = OpTypeIndex::build_op_pos(operations);
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
            opref_to_slot: VecAssoc::new(),
            inputargs,
            operations,
            inputarg_pos,
            op_pos,
            constants,
            constant_types: majit_ir::VecAssoc::new(),
            next_slot: 0,
            guard_success_cc: None,
            target_tokens_currently_compiling: VecAssoc::new(),
            compiled_target_tokens: Vec::new(),
            vtable_offset,
            classptr_to_typeid,
            guard_gc_type_info,
            classptr_to_subclass_range,
            self_entry_label: None,
            self_entry_addr_ptr: Box::into_raw(Box::new(0usize)),
            call_assembler_targets: VecAssoc::new(),
            finish_gcmap: None,
            gcmap_for_finish: {
                let gcmap = allocate_gcmap(1, JITFRAME_FIXED_SIZE);
                gcmap_set_bit(gcmap, 0);
                gcmap
            },
            pending_force_descr: None,
            pending_force_cell: None,
            attached_descrs,
            cpu_handle,
            frame_depth_to_patch: Vec::new(),
            jump_target_frame_depth: 0,
            malloc_slowpath_fixed,
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
    /// reads in `rpython/jit/backend/x86/assembler.py`.  Reads from the
    /// live `cpu_handle` snapshot so the raw pointer baked into
    /// `JF_DESCR_OFS` and the Arc returned by
    /// `done_with_this_frame_descr_arc_for_type` resolve to the same
    /// metainterp singleton.
    fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .done_with_this_frame_descr_ptr_for_type(tp) as i64
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` Arc lookup —
    /// returns the metainterp `DoneWithThisFrameDescr*` Arc the
    /// optimizer attached for the given result type.  Used to stamp
    /// `meta_descr` on backend FINISH descrs so trait forwarding routes
    /// `is_finish` / `fail_arg_types` through the metainterp class
    /// hierarchy (`compile.py:624 final_descr=True`).
    fn done_with_this_frame_descr_arc_for_type(&self, tp: Type) -> Option<majit_ir::DescrRef> {
        let attachments = self.cpu_handle.read().unwrap();
        match tp {
            Type::Void => attachments.done_with_this_frame_descr_void.clone(),
            Type::Int => attachments.done_with_this_frame_descr_int.clone(),
            Type::Ref => attachments.done_with_this_frame_descr_ref.clone(),
            Type::Float => attachments.done_with_this_frame_descr_float.clone(),
        }
    }

    /// `compile.py:658` parity: `self.cpu.exit_frame_with_exception_descr_ref`.
    fn exit_frame_with_exception_descr_ref_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .exit_frame_with_exception_descr_ref as i64
    }

    /// `pyjitpl.py:2283` parity: `self.cpu.propagate_exception_descr`.
    /// Stamped into `jf_descr` by the inline propagate path emitted at
    /// `OpCode::CheckMemoryError` (assembler.py:1630-1641
    /// `genop_discard_check_memory_error`).
    fn propagate_exception_descr_ptr(&self) -> i64 {
        self.cpu_handle
            .read()
            .unwrap()
            .descr_ptrs()
            .propagate_exception_descr as i64
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
        if let Some(&slot) = self.opref_to_slot.get(&opref) {
            return ResolvedArg::Slot(Self::slot_offset(slot));
        }
        // history.py:227/268/314 — inline-Const variants carry value inline.
        if let Some(val) = opref
            .inline_const_bits()
            .or_else(|| self.constants.get(&opref.raw()).copied())
        {
            return ResolvedArg::Const(val);
        }
        // history.py:227/268/314 — Const always carries a value, so a
        // constant OpRef with no resolvable value is an invariant break,
        // not a `#0`.
        if opref.is_constant() {
            panic!(
                "resolve_opref: legacy constant {opref:?} missing from constants pool — \
                 Const always carries a value (history.py:227/268/314)"
            );
        }
        // regalloc.py:102 `FrameManager.loc(must_exist=True)` raises KeyError
        // for a non-constant box that is neither register- nor frame-resident:
        // a used box is always slot-mapped or constant. Silently materializing
        // an unmapped box as `#0` would hide a wrong value — the same hazard
        // that moved genop_call_assembler onto regalloc arglocs. Fail loud.
        panic!(
            "resolve_opref: unmapped non-constant OpRef {opref:?} — every used \
             box must be slot-mapped or constant (regalloc.py:102 loc must_exist)"
        );
    }

    /// Allocate a frame slot for an OpRef and return the slot index.
    /// Reuses existing slot if the OpRef already has one.
    fn allocate_slot(&mut self, opref: OpRef) -> usize {
        if let Some(&existing) = self.opref_to_slot.get(&opref) {
            return existing;
        }
        let slot = self.next_slot;
        self.next_slot += 1;
        if self.next_slot + 1 > self.frame_depth {
            self.frame_depth = self.next_slot + 1;
        }
        self.opref_to_slot.insert(opref, slot);
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
        let mut srccount: VecAssoc<i32, i32> = VecAssoc::new();
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
                let mut sources: VecAssoc<i32, Loc> = VecAssoc::new();
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
        let mut dst_keys = VecAssoc::new();
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
                self.emit_cmp_imm64(r.value, i.value);
            }
            (Loc::Frame(f), Loc::Reg(s)) => {
                dynasm!(self.mc ; .arch x64 ; cmp [rbp + f.ebp_loc.value], Rq(s.value));
            }
            (Loc::Frame(f), Loc::Immed(i)) => {
                if let Ok(v) = i32::try_from(i.value) {
                    dynasm!(self.mc ; .arch x64 ; cmp QWORD [rbp + f.ebp_loc.value], v);
                } else {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD i.value
                        ; cmp QWORD [rbp + f.ebp_loc.value], Rq(scratch)
                    );
                }
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
                    self.opref_to_slot.insert(ia.opref(), abs_slot);
                    if abs_slot + 1 > max_abs_slot {
                        max_abs_slot = abs_slot + 1;
                    }
                }
            }
            self.next_slot = max_abs_slot;
        } else {
            for (i, ia) in inputargs.iter().enumerate() {
                self.opref_to_slot
                    .insert(ia.opref(), JITFRAME_FIXED_SIZE + i);
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
        // x86/assembler.py:1052 _call_header parity. PyPy reserves the
        // whole frame in a single `SUB esp, FRAME_FIXED_SIZE * WORD` and
        // stores `CALLEE_SAVE_REGISTERS` plus `ebp` at fixed offsets.
        // The Pyre variant uses the same shape (single SUB + offset
        // stores) without the PASS_ON_MY_FRAME scratch area or vmprof
        // slots that PyPy reserves but never populates here.
        //
        // Saved set per platform (matches PyPy's CALLEE_SAVE_REGISTERS):
        //   - x86_64 (System V): rbx, r12, r13, r14, r15 plus rbp
        //   - x86_64 (Win64):    rbx, rsi, rdi, r12, r14, r15 plus rbp
        //
        // Layout (lowest address first, all offsets relative to the new
        // rsp after the SUB):
        //   Win64:  [+0 rbx, +8 rsi, +16 rdi, +24 r12, +32 r14, +40 r15,
        //            +48 rbp, +56 r13, +64 pad]   → SUB 72 (8 slots +
        //            1 padding; body rsp at 0 mod 16 since function
        //            entry rsp was 8 mod 16)
        //   SysV:   [+0 rbx, +8 r12, +16 r13, +24 r14, +32 r15, +40 rbp,
        //            +48 pad]   → SUB 56 (6 slots + 1 padding; body rsp
        //            at 0 mod 16)
        //
        // The trailing padding slot (`+64` Win64, `+48` SysV) brings the
        // body rsp from the function-entry 8-mod-16 down to 0-mod-16,
        // matching PyPy's body alignment convention.  This lets every
        // inner CALL omit the per-call `SUB rsp, 8` alignment fixup
        // (PyPy `_build_malloc_slowpath` `add_to_esp = 16 - WORD = 8`
        // accounts for the dual: PyPy trampoline body is 8-mod-16
        // because PyPy JIT body is 0-mod-16 + CALL push).
        //
        // `r12` carries the caller's `rbp` (saved jf_ptr) across nested
        // `genop_call_assembler` reentries, so it must be preserved
        // alongside the rest of the callee-save set.  Win64 also saves
        // `r13` even though PyPy's `arch.py:43` skips it ("never use
        // r13"): pyre's `genop_call_assembler` does use r13 as a
        // scratch reg that must survive `free()`, so r13 occupies the
        // previously-padding slot at +56 and is restored by every
        // `_call_footer`/`emit_call_footer_raw` variant.
        #[cfg(target_os = "windows")]
        dynasm!(self.mc
            ; .arch x64
            ; sub rsp, 72
            ; mov [rsp + 0],  rbx
            ; mov [rsp + 8],  rsi
            ; mov [rsp + 16], rdi
            ; mov [rsp + 24], r12
            ; mov [rsp + 32], r14
            ; mov [rsp + 40], r15
            ; mov [rsp + 48], rbp
            ; mov [rsp + 56], r13
            ; mov rbp, rcx
        );
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.mc
            ; .arch x64
            ; sub rsp, 56
            ; mov [rsp + 0],  rbx
            ; mov [rsp + 8],  r12
            ; mov [rsp + 16], r13
            ; mov [rsp + 24], r14
            ; mov [rsp + 32], r15
            ; mov [rsp + 40], rbp
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
                );
                self.emit_abi_int_arg_from_reg(0, 4); // rsp
                dynasm!(self.mc
                    ; .arch x64
                    ; mov Rq(scratch), QWORD addrs.slowpath_addr as i64
                );
                self.emit_abi_call_reg(scratch);
                dynasm!(self.mc
                    ; .arch x64
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
                    // Overflow fallthrough: return rbp as jf_ptr.  Mirrors
                    // `_call_footer` without `gen_footer_shadowstack` —
                    // `gen_shadowstack_header` runs after this stack-check
                    // path, so no shadow-stack entry has been pushed yet.
                    ; mov rax, rbp
                );
                // Win64: r13 restored from the +56 slot — see
                // `_call_header` for why pyre saves r13 even though
                // PyPy's `arch.py:43` does not.
                #[cfg(target_os = "windows")]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rbx, [rsp + 0]
                    ; mov rsi, [rsp + 8]
                    ; mov rdi, [rsp + 16]
                    ; mov r12, [rsp + 24]
                    ; mov r14, [rsp + 32]
                    ; mov r15, [rsp + 40]
                    ; mov rbp, [rsp + 48]
                    ; mov r13, [rsp + 56]
                    ; add rsp, 72
                );
                #[cfg(not(target_os = "windows"))]
                dynasm!(self.mc
                    ; .arch x64
                    ; mov rbx, [rsp + 0]
                    ; mov r12, [rsp + 8]
                    ; mov r13, [rsp + 16]
                    ; mov r14, [rsp + 24]
                    ; mov r15, [rsp + 32]
                    ; mov rbp, [rsp + 40]
                    ; add rsp, 56
                );
                dynasm!(self.mc
                    ; .arch x64
                    ; ret
                    ; =>continue_label
                );
            }
        }
        // When addresses are not registered (tests / early startup), no
        // stack check is emitted — assembler.py:1082-1083 parity.
        self.gen_shadowstack_header();
        self.setup_input_state(inputargs);
    }

    fn abi_int_arg(idx: usize) -> AbiArgPlacement {
        #[cfg(target_os = "windows")]
        match idx {
            0 => AbiArgPlacement::Gpr(1), // rcx
            1 => AbiArgPlacement::Gpr(2), // rdx
            2 => AbiArgPlacement::Gpr(8),
            3 => AbiArgPlacement::Gpr(9),
            _ => AbiArgPlacement::Stack(32 + ((idx - 4) * WORD) as i32),
        }
        #[cfg(not(target_os = "windows"))]
        match idx {
            0 => AbiArgPlacement::Gpr(7), // rdi
            1 => AbiArgPlacement::Gpr(6), // rsi
            2 => AbiArgPlacement::Gpr(2), // rdx
            3 => AbiArgPlacement::Gpr(1), // rcx
            4 => AbiArgPlacement::Gpr(8),
            5 => AbiArgPlacement::Gpr(9),
            _ => AbiArgPlacement::Stack(((idx - 6) * WORD) as i32),
        }
    }

    fn build_abi_arg_placements(arg_types: &[Type]) -> (Vec<AbiArgPlacement>, usize) {
        let mut placements = Vec::with_capacity(arg_types.len());
        let mut stack_slots = 0usize;
        #[cfg(target_os = "windows")]
        {
            for (idx, tp) in arg_types.iter().copied().enumerate() {
                let placement = if idx < 4 {
                    if tp == Type::Float {
                        AbiArgPlacement::Xmm(idx as u8)
                    } else {
                        Self::abi_int_arg(idx)
                    }
                } else {
                    let ofs = 32 + ((idx - 4) * WORD) as i32;
                    stack_slots += 1;
                    AbiArgPlacement::Stack(ofs)
                };
                placements.push(placement);
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let mut gpr_idx = 0usize;
            let mut xmm_idx = 0usize;
            for tp in arg_types.iter().copied() {
                let placement = if tp == Type::Float {
                    if xmm_idx < 8 {
                        let p = AbiArgPlacement::Xmm(xmm_idx as u8);
                        xmm_idx += 1;
                        p
                    } else {
                        let p = AbiArgPlacement::Stack((stack_slots * WORD) as i32);
                        stack_slots += 1;
                        p
                    }
                } else if gpr_idx < 6 {
                    let p = Self::abi_int_arg(gpr_idx);
                    gpr_idx += 1;
                    p
                } else {
                    let p = AbiArgPlacement::Stack((stack_slots * WORD) as i32);
                    stack_slots += 1;
                    p
                };
                placements.push(placement);
            }
        }
        (placements, stack_slots)
    }

    fn emit_abi_arg_from_reg(
        &mut self,
        placement: AbiArgPlacement,
        src: crate::regloc::RegLoc,
        arg_type: Type,
    ) {
        match placement {
            AbiArgPlacement::Gpr(dst) => {
                if src.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movq Rq(dst), Rx(src.value));
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst), Rq(src.value));
                }
            }
            AbiArgPlacement::Xmm(dst) => {
                if src.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(dst), Rx(src.value));
                } else {
                    dynasm!(self.mc ; .arch x64 ; movq Rx(dst), Rq(src.value));
                }
            }
            AbiArgPlacement::Stack(offset) => {
                if src.is_xmm && arg_type == Type::Float {
                    dynasm!(self.mc ; .arch x64 ; movsd [rsp + offset], Rx(src.value));
                } else if src.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movq [rsp + offset], Rx(src.value));
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov [rsp + offset], Rq(src.value));
                }
            }
        }
    }

    fn emit_abi_arg_from_mem(&mut self, placement: AbiArgPlacement, offset: i32, arg_type: Type) {
        match placement {
            AbiArgPlacement::Gpr(dst) => dynasm!(self.mc ; .arch x64 ; mov Rq(dst), [rbp + offset]),
            AbiArgPlacement::Xmm(dst) => {
                if arg_type == Type::Float {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(dst), [rbp + offset]);
                } else {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), [rbp + offset]
                        ; movq Rx(dst), Rq(scratch)
                    );
                }
            }
            AbiArgPlacement::Stack(dst_offset) => {
                let scratch = crate::regloc::XMM15.value;
                if arg_type == Type::Float {
                    dynasm!(self.mc ; .arch x64
                        ; movsd Rx(scratch), [rbp + offset]
                        ; movsd [rsp + dst_offset], Rx(scratch)
                    );
                } else {
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), [rbp + offset]
                        ; mov [rsp + dst_offset], Rq(scratch)
                    );
                }
            }
        }
    }

    fn emit_abi_arg_from_imm(&mut self, placement: AbiArgPlacement, val: i64, arg_type: Type) {
        match placement {
            AbiArgPlacement::Gpr(dst) => dynasm!(self.mc ; .arch x64 ; mov Rq(dst), QWORD val),
            AbiArgPlacement::Xmm(dst) => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64
                    ; mov Rq(scratch), QWORD val
                    ; movq Rx(dst), Rq(scratch)
                );
            }
            AbiArgPlacement::Stack(offset) => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                if arg_type == Type::Float {
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD val
                        ; mov [rsp + offset], Rq(scratch)
                    );
                } else {
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD val
                        ; mov [rsp + offset], Rq(scratch)
                    );
                }
            }
        }
    }

    fn emit_abi_int_arg_from_reg(&mut self, idx: usize, src: u8) {
        self.emit_abi_arg_from_reg(
            Self::abi_int_arg(idx),
            crate::regloc::RegLoc::new(src, false),
            Type::Int,
        );
    }

    fn emit_abi_int_arg_from_imm(&mut self, idx: usize, val: i64) {
        self.emit_abi_arg_from_imm(Self::abi_int_arg(idx), val, Type::Int);
    }

    fn emit_win64_call_adjust(extra_pushes: usize) -> i32 {
        // Body rsp is 0-mod-16 (per `_call_header`'s padding-slot SUB).
        // With 0 extra pushes, only the 32-byte shadow space is needed
        // (rsp already aligned).  With 1 extra push, rsp is at 8-mod-16
        // so an extra 8 bytes of alignment pad is required before the
        // 32-byte shadow space → 40.
        if extra_pushes & 1 == 0 { 32 } else { 40 }
    }

    fn abi_reserved_call_area_size(extra_pushes: usize, stack_slots: usize) -> i32 {
        // Body rsp is 0-mod-16; the inversion in `needs_pad` accounts
        // for that vs the previous 8-mod-16 layout (see `_call_header`
        // comment for the alignment rationale).
        #[cfg(target_os = "windows")]
        {
            let base = 32 + (stack_slots * WORD) as i32;
            let needs_pad = if extra_pushes & 1 == 0 {
                base % 16 != 0
            } else {
                base % 16 == 0
            };
            base + if needs_pad { WORD as i32 } else { 0 }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let base = (stack_slots * WORD) as i32;
            let needs_pad = if extra_pushes & 1 == 0 {
                base % 16 != 0
            } else {
                base % 16 == 0
            };
            base + if needs_pad { WORD as i32 } else { 0 }
        }
    }

    fn emit_reserve_abi_call_area(&mut self, extra_pushes: usize, stack_slots: usize) -> i32 {
        let adjust = Self::abi_reserved_call_area_size(extra_pushes, stack_slots);
        if adjust != 0 {
            dynasm!(self.mc ; .arch x64 ; sub rsp, adjust);
        }
        adjust
    }

    fn emit_release_abi_call_area(&mut self, adjust: i32) {
        if adjust != 0 {
            dynasm!(self.mc ; .arch x64 ; add rsp, adjust);
        }
    }

    fn emit_abi_call_rax_with_extra_pushes(&mut self, extra_pushes: usize) {
        let _ = extra_pushes;
        #[cfg(target_os = "windows")]
        {
            let adjust = Self::emit_win64_call_adjust(extra_pushes);
            dynasm!(self.mc ; .arch x64
                ; sub rsp, adjust
                ; call rax
                ; add rsp, adjust
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            let adjust = Self::abi_reserved_call_area_size(extra_pushes, 0);
            if adjust != 0 {
                dynasm!(self.mc ; .arch x64 ; sub rsp, adjust);
            }
            dynasm!(self.mc ; .arch x64 ; call rax);
            if adjust != 0 {
                dynasm!(self.mc ; .arch x64 ; add rsp, adjust);
            }
        }
    }

    fn emit_abi_call_rax(&mut self) {
        self.emit_abi_call_rax_with_extra_pushes(0);
    }

    fn emit_abi_call_rax_aligned(&mut self) {
        // Body rsp is 0-mod-16 (see `_call_header`), so no alignment
        // SUB is needed before the inner CALL on either ABI; Win64
        // still needs the 32-byte shadow space.
        #[cfg(target_os = "windows")]
        self.emit_abi_call_rax_with_extra_pushes(0);
        #[cfg(not(target_os = "windows"))]
        dynasm!(self.mc ; .arch x64 ; call rax);
    }

    fn emit_abi_call_rax_after_one_push(&mut self) {
        self.emit_abi_call_rax_with_extra_pushes(1);
    }

    fn emit_abi_call_reg_with_extra_pushes(&mut self, reg: u8, extra_pushes: usize) {
        let _ = extra_pushes;
        #[cfg(target_os = "windows")]
        {
            let adjust = Self::emit_win64_call_adjust(extra_pushes);
            dynasm!(self.mc ; .arch x64
                ; sub rsp, adjust
                ; call Rq(reg)
                ; add rsp, adjust
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            let adjust = Self::abi_reserved_call_area_size(extra_pushes, 0);
            if adjust != 0 {
                dynasm!(self.mc ; .arch x64 ; sub rsp, adjust);
            }
            dynasm!(self.mc ; .arch x64 ; call Rq(reg));
            if adjust != 0 {
                dynasm!(self.mc ; .arch x64 ; add rsp, adjust);
            }
        }
    }

    fn emit_abi_call_reg(&mut self, reg: u8) {
        self.emit_abi_call_reg_with_extra_pushes(reg, 0);
    }

    // ----------------------------------------------------------------
    // assembler.py:2153 _call_footer — function epilogue
    // ----------------------------------------------------------------

    /// Emit the function epilogue: return jf_ptr in RAX/X0.
    /// Thin wrapper around the free-fn `emit_call_footer_raw` so the
    /// backend-owned malloc trampoline can emit byte-identical epilogue
    /// sequences when exiting through `propagate_exception_descr`.
    fn _call_footer(&mut self) {
        emit_call_footer_raw(&mut self.mc);
    }

    /// x86/assembler.py:254 `_push_all_regs_to_jitframe` parity. Writes
    /// every managed GPR (and optionally XMM) into its canonical
    /// jitframe save slot so a subsequent collecting helper call can
    /// trace live Refs via the gcmap. Skips registers in `ignored_regs`
    /// (typically the slow-path's argument / result register, which
    /// holds non-Ref data across the call).
    ///
    /// Iterates `crate::x86::regalloc::ALL_CORE_REGS` / `ALL_FLOAT_REGS`
    /// (the allocator pool — drops R13 and XMM5..XMM14 on Win64), and
    /// indexes the slot via `core_reg_position` / `float_reg_position`,
    /// which look up positions in the same Win64-aware lists.  This
    /// mirrors `regalloc.py all_reg_indexes`, which is built from the
    /// post-`remove(r13)` `all_regs` on Win64 (so R14→10, R15→11).
    /// `save_regs_label`, the `core_reg_index`-driven gcmap, and the
    /// post-call pop all consume positions through this same Win64-aware
    /// list, keeping the three in agreement.
    fn push_all_regs_to_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        push_all_regs_to_jitframe_raw(&mut self.mc, ignored_regs, withfloats);
    }

    /// x86/assembler.py:283 `_pop_all_regs_from_jitframe` parity.
    fn pop_all_regs_from_jitframe(
        &mut self,
        ignored_regs: &[crate::regloc::RegLoc],
        withfloats: bool,
    ) {
        pop_all_regs_from_jitframe_raw(&mut self.mc, ignored_regs, withfloats);
    }

    /// `assembler.py:910 _check_frame_depth` parity — emitted at every
    /// bridge entry to detect that the in-flight JITFRAME's variable
    /// section is wide enough for the bridge's spill requirements, and
    /// reallocate via `dynasm_realloc_frame` if not.
    ///
    /// Layout:
    /// ```text
    ///   CMP QWORD [rbp + JF_FRAME_OFS + LENGTHOFS], imm32_placeholder
    ///                              ; → frame_depth_to_patch[]
    ///   JGE  continue              ; fast path: frame large enough
    ///   ;; --- inlined slowpath body ---
    ///   push_all_regs_to_jitframe(&[], withfloats=true)
    ///   push_gcmap(gcmap)          ; publish live Refs for the collector
    ///   MOV  ARG0, rbp             ; old_jf
    ///   MOV  ARG1_r32, imm32_placeholder
    ///                              ; → frame_depth_to_patch[] (depth)
    ///   MOV  rax, &dynasm_realloc_frame
    ///   CALL rax                   ; rax = new_jf
    ///   MOV  rbp, rax              ; switch frame pointer
    ///   ;; update shadowstack top entry pushed by gen_shadowstack_header
    ///   MOV  scratch, [root_stack_top]
    ///   MOV  [scratch - WORD], rbp ; replace stale jf in shadow entry
    ///   pop_gcmap                  ; clear JF_GCMAP_OFS on new frame
    ///   pop_all_regs_from_jitframe(&[], withfloats=true)
    /// continue:
    /// ```
    ///
    /// The 32-bit `0xffffff` placeholder appears twice (CMP imm and the
    /// MOV ARG1 imm); both offsets land in `frame_depth_to_patch` so
    /// `patch_stack_checks` rewrites them in lockstep with the final
    /// `frame_depth`.
    ///
    /// `_call_header` already published `rbp` on the shadow stack via
    /// `gen_shadowstack_header`, so the entry at `[top - WORD]` holds
    /// the old jitframe; the slowpath rewrites that slot so a minor GC
    /// firing between this point and the next `gen_footer_shadowstack`
    /// observes the live new-frame pointer rather than the freed old.
    fn emit_check_frame_depth(&mut self, gcmap: *mut usize) {
        let frame_len_ofs = (JF_FRAME_OFS + crate::jitframe::LENGTHOFS) as i32;
        let placeholder: i32 = 0xffffff;

        // assembler.py:918 — CMP_bi(ofs, 0xffffff).  dynasm encodes
        // this as `48 81 7D disp8 imm32` (8 bytes) when the
        // displacement fits in i8 — which it does for `JF_FRAME_OFS`
        // (= 56).  The 4-byte immediate lives at the tail of the
        // instruction, so `offset - 4` is its buffer position.
        dynasm!(self.mc ; .arch x64
            ; cmp QWORD [rbp + frame_len_ofs], placeholder
        );
        let cmp_imm_ofs = self.mc.offset().0 - 4;
        self.frame_depth_to_patch.push(cmp_imm_ofs);

        // assembler.py:921 — sp = IncreaseStackSlowPath(mc, 'L').
        // PyPy uses condition 'L' (signed less than) for the slowpath
        // entry; the fast-path fall-through is the JGE-skip equivalent.
        let continue_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; jge =>continue_label);

        // ── inlined IncreaseStackSlowPath + _frame_realloc_slowpath ──
        // assembler.py:145 _push_all_regs_to_frame(mc, [], supports_floats)
        self.push_all_regs_to_jitframe(&[], true);
        // assembler.py:907 push_gcmap(store=True) — pyre writes to
        // [rbp + JF_GCMAP_OFS] rather than a stack slot, matching the
        // existing `push_gcmap` helper (the `store=True` arg flavor in
        // PyPy is a stack-slot variant that the shared trampoline
        // reads back; the inlined slowpath here uses the frame slot
        // directly since `dynasm_realloc_frame` does not need the
        // gcmap in a register).
        self.push_gcmap(gcmap);

        // assembler.py:173 — `_store_and_reset_exception(mc, None,
        // ebx, tmpreg)` parity.  pos_exc_value goes into
        // [rbp + JF_GUARD_EXC_OFS] (copied by `realloc_frame` to the
        // new frame at jitframe.rs:432); pos_exception goes into RBX
        // (callee-save across the C `realloc_frame` call).  Both globals
        // are cleared so the helper does not see leftover state.
        //
        // RBX's frame save slot already holds the caller's pre-slowpath
        // value (written by `push_all_regs_to_jitframe` above); we only
        // transiently use the RBX *register* during this slowpath, and
        // `pop_all_regs_from_jitframe` restores it before continuing.
        // The minor-GC root walk sees the saved value through the
        // RBX slot and the jf_guard_exc slot — both managed positions.
        let scratch_for_exc = crate::regloc::X86_64_SCRATCH_REG.value;
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch_for_exc), QWORD exc_value_addr
            ; mov Rq(scratch_for_exc), [Rq(scratch_for_exc)]
            ; mov [rbp + JF_GUARD_EXC_OFS], Rq(scratch_for_exc)
            ; mov Rq(scratch_for_exc), QWORD exc_type_addr
            ; mov rbx, [Rq(scratch_for_exc)]
            ; mov QWORD [Rq(scratch_for_exc)], 0
            ; mov Rq(scratch_for_exc), QWORD exc_value_addr
            ; mov QWORD [Rq(scratch_for_exc)], 0
        );

        // assembler.py:150-152 — MOV ARG0 = rbp, ARG1 = depth (from
        // stack in PyPy's shared trampoline; here from a patched imm).
        self.emit_abi_int_arg_from_reg(0, crate::regloc::EBP.value);
        let arg1_reg = match Self::abi_int_arg(1) {
            AbiArgPlacement::Gpr(r) => r,
            _ => panic!("emit_check_frame_depth: ARG1 must be a GPR on x86_64"),
        };
        // `MOV r32, imm32` is `B8+rd imm32` (5 bytes) for low regs
        // (rdx=2 on Win64, rsi=6 on Linux — both low).  The imm32
        // zero-extends to r64, which is correct for a positive depth.
        dynasm!(self.mc ; .arch x64 ; mov Rd(arg1_reg), placeholder);
        let arg1_imm_ofs = self.mc.offset().0 - 4;
        self.frame_depth_to_patch.push(arg1_imm_ofs);

        // assembler.py:175 — CALL imm(self.cpu.realloc_frame).  Pyre
        // bakes the C-ABI wrapper address as a 64-bit immediate
        // (PyPy's `realloc_frame` is exposed via the JIT-frontend's
        // cpu.realloc_frame pointer; the wrapper performs the same
        // libc::calloc + write_barrier sequence).
        let helper_addr = crate::runner::dynasm_realloc_frame as i64;
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD helper_addr);
        self.emit_abi_call_rax();

        // assembler.py:176 — MOV ebp, eax: rbp ← new jitframe.
        dynasm!(self.mc ; .arch x64 ; mov rbp, rax);

        // assembler.py:177 — `_restore_exception(mc, None, ebx, ecx)`
        // parity.  pos_exc_value comes back from the new frame's
        // JF_GUARD_EXC slot (copied by `realloc_frame`); pos_exception
        // comes back from RBX, which the C `realloc_frame` call has
        // preserved as a callee-save register.  Clear JF_GUARD_EXC
        // after the read so the slot does not retain a stale exc value
        // on the next guard exit.
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch_for_exc), [rbp + JF_GUARD_EXC_OFS]
            ; mov QWORD [rbp + JF_GUARD_EXC_OFS], 0
            ; mov rax, QWORD exc_value_addr
            ; mov [rax], Rq(scratch_for_exc)
            ; mov rax, QWORD exc_type_addr
            ; mov [rax], rbx
        );

        // assembler.py:181-184 — update shadow-stack top entry.  The
        // `gen_shadowstack_header` push writes the live jf at
        // `[top - WORD]` after incrementing top by 2*WORD; the
        // realloc must rewrite that slot so the GC visitor finds the
        // post-realloc frame on the next minor collection.
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD rst_addr
            ; mov Rq(scratch), [Rq(scratch)]
            ; mov [Rq(scratch) - 8], rbp
        );

        // assembler.py:186-187 — pop_gcmap + _pop_all_regs_from_frame.
        self.pop_gcmap();
        self.pop_all_regs_from_jitframe(&[], true);

        dynasm!(self.mc ; .arch x64 ; =>continue_label);
    }

    /// x86/assembler.py:1422 `gen_shadowstack_header` parity (mirrors
    /// aarch64). Pushes two words onto the jitframe shadow stack on
    /// every JIT function entry: an `is_minor` marker (`1`) and the
    /// current jitframe pointer (rbp). The GC walks this stack during
    /// minor-collect to update jf pointers — without it, a minor GC
    /// inside a recursive call (e.g. fib_recursive) leaves rbp dangling
    /// at the freed nursery slot.
    fn gen_shadowstack_header(&mut self) {
        let rst = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD rst
            ; mov rax, [Rq(scratch)]    // rax = *rst = top
            ; mov QWORD [rax], 1        // [top] = 1 (is_minor marker)
            ; mov [rax + 8], rbp        // [top + WORD] = rbp (jf_ptr)
            ; add rax, 16               // top += 2*WORD
            ; mov [Rq(scratch)], rax    // *rst = top
        );
    }

    /// x86/assembler.py:1130 `_call_footer_shadowstack` parity:
    ///
    /// ```python
    /// if rx86.fits_in_32bits(rst):
    ///     self.mc.SUB_ji8(rst, WORD * 2)       # SUB [rootstacktop], 16
    /// else:
    ///     self.mc.MOV_ri(ebx.value, rst)       # MOV ebx, rootstacktop
    ///     self.mc.SUB_mi8((ebx.value, 0), WORD * 2)  # SUB [ebx], 16
    /// ```
    ///
    /// One in-memory subtract — no need to load the current top into a
    /// register, decrement, and store back.
    fn gen_footer_shadowstack(&mut self) {
        emit_footer_shadowstack_raw(&mut self.mc);
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

    /// x86/assembler.py:1369-1383 `_reload_frame_if_necessary` parity:
    ///
    /// ```python
    ///   MOV ecx, [rootstacktop]   // shadow stack top pointer
    ///   MOV ebp, [ecx - WORD]     // jf_ptr at top - WORD
    ///   _write_barrier_fastpath(mc, wbdescr, [ebp], array=False,
    ///                           is_frame=True)
    /// ```
    ///
    /// After a collecting helper call the GC may have copied the
    /// jitframe from nursery to old gen. PyPy minor-GC does not write
    /// `jf_forward` on a move — that field is reserved for the
    /// `grow_jitframe` realloc path — so chasing `jf_forward` here
    /// reads the freed nursery slot. The shadow-stack entry IS
    /// rewritten by the GC visitor during copy, so the live jf_ptr
    /// lives at `*(root_stack_top - WORD)`. Reload `rbp` from there.
    ///
    /// Then re-apply the non-array write barrier on the new jitframe
    /// (`is_frame=True`): subsequent stores of nursery refs into
    /// jitframe slots must be tracked by minor GC, otherwise an
    /// old-gen jitframe holding a nursery pointer is missed during
    /// the next collection and the slot ends up dangling.
    fn reload_frame_if_necessary(&mut self) {
        let rst_addr = majit_gc::shadow_stack::get_root_stack_top_addr() as i64;
        dynasm!(self.mc ; .arch x64
            ; mov rcx, QWORD rst_addr
            ; mov rcx, [rcx]            // rcx = *rst_addr = root_stack_top
            ; mov rbp, [rcx - 8]        // rbp = *(top - WORD) = jf_ptr
        );
        // assembler.py:1378-1383 `_reload_frame_if_necessary` parity:
        //
        // ```python
        // wbdescr = self.cpu.gc_ll_descr.write_barrier_descr
        // if gcrootmap and wbdescr:
        //     # frame never uses card marking, so we enforce this is not
        //     # an array
        //     self._write_barrier_fastpath(mc, wbdescr, [ebp], array=False,
        //                                  is_frame=True)
        // ```
        //
        // After a collecting helper call the jitframe may have been
        // promoted from nursery to old-gen.  Subsequent stores of young
        // Refs into `[rbp + ofs]` would create old→young pointers
        // invisible to the GC — the WB fastpath re-arms the `TRACK_YOUNG_PTRS`
        // bit so the next collection scans the frame.  Reuses the shared
        // `emit_write_barrier_fastpath_kind` helper — `assembler.py:2388-2419`
        // expresses both `is_frame=True` and `is_frame=False` in a single
        // `_write_barrier_fastpath` whose addressing degenerates naturally
        // when `loc_base == ebp`.  `is_array=false` skips card marking
        // (assembler.py:2401 `if array and jit_wb_cards_set` gate); the
        // `helper_num=4` XMM-skip optimization is a perf adaptation not
        // a correctness gap and is not yet implemented.
        if crate::runner::DYNASM_ACTIVE_GC.with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|gc| gc.get_write_barrier_descr())
                .is_some()
        }) {
            let rbp_loc = Loc::Reg(crate::regloc::EBP);
            self.emit_write_barrier_fastpath_kind(&[rbp_loc], false);
        }
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
    pub fn assemble_loop(mut self) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp).collect();

        // assembler.py:537 prepare_loop — set up regalloc
        // For now, simplified: all args in frame slots

        // assembler.py:547 _assemble — generate code for all ops
        // Create a dynamic label at the entry point for self-recursive
        // CALL_ASSEMBLER (redirect_call_assembler parity).
        let entry_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; =>entry_label);
        self.self_entry_label = Some(entry_label);
        let entry = self.mc.offset();
        self._assemble(true)?;

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

        // assembler.py:556 patch_stack_checks — overwrite the 32-bit
        // `0xffffff` placeholders in any `_check_frame_depth` /
        // `_check_frame_depth_debug` emission with the loop's final
        // absolute frame depth (already includes `JITFRAME_FIXED_SIZE`).
        // No-op when the assembler did not emit a check (empty list).
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        // Write resolved entry address for self-recursive CALL_ASSEMBLER
        // trampoline. The JIT code loads from this pointer at runtime.
        unsafe { *self.self_entry_addr_ptr = rawstart + entry.0 };

        for descr in &self.compiled_target_tokens {
            if let Some(loop_descr) = descr.as_loop_target_descr() {
                // assembler.py:1003-1008 reads the target loop's
                // `frame_info.jfi_frame_depth` at a cross-loop JUMP; publish
                // this loop's full frame depth so a later trace's JUMP can
                // size its frame for this target.  Carries the depth grown by
                // `_assemble` for this loop's own onward JUMP.
                //
                // Store the companion `target_frame_depth` BEFORE the
                // `ll_loop_code` gate (both Release stores): a reader
                // Acquire-loads `ll_loop_code` and only reads
                // `target_frame_depth` once the gate is non-zero, so the depth
                // must become visible first — otherwise the reader pairs the
                // new code pointer with a stale 0 depth and bypasses the
                // frame-capacity check (descr.rs set_dispatch_target ordering
                // contract).  Dynasm ignores `label_block_id` (descr.rs:1236 —
                // it bakes the LABEL address straight into `ll_loop_code`), so
                // that companion is not published here.
                loop_descr.set_target_frame_depth(self.frame_depth);
                loop_descr.set_ll_loop_code(loop_descr.ll_loop_code() + rawstart);
            }
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: None,
        })
    }

    /// assembler.py:320 descr._ll_function_addr parity: store
    /// call_target_token → code_addr mappings for CALL_ASSEMBLER.
    pub fn set_call_assembler_targets(&mut self, targets: VecAssoc<u64, usize>) {
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
        descr: &dyn majit_ir::FailDescr,
        inputargs: &[InputArg],
    ) -> Vec<Loc> {
        let mut locs = Vec::new();
        let gpr_regs = crate::x86::regalloc::ALL_CORE_REGS;
        let float_regs = crate::x86::regalloc::ALL_FLOAT_REGS;
        let base_ofs = crate::jitframe::FIRST_ITEM_OFFSET as i32;
        let mut input_i = 0usize;
        for &pos in descr.rd_locs() {
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
        arglocs: &[Loc],
    ) -> Result<CompiledCode, BackendError> {
        self.input_types = self.inputargs.iter().map(|ia| ia.tp).collect();
        self.bridge_input_locs = if arglocs.is_empty() {
            None
        } else {
            Some(arglocs.to_vec())
        };

        // assembler.py:641 prepare_bridge
        let entry = self.mc.offset();
        self._assemble(false)?;
        let stub_offsets = self.write_pending_failure_recoveries();

        let buffer = self
            .mc
            .finalize()
            .map_err(|_| BackendError::CompilationFailed("dynasm finalize failed".to_string()))?;

        let rawstart = codebuf::buffer_ptr(&buffer) as usize;
        Self::patch_pending_failure_recoveries(rawstart, &stub_offsets);

        // assembler.py:658 patch_stack_checks — same as the loop path,
        // applied with the bridge's own absolute frame depth.  Bridges
        // routinely grow the depth past the original loop value; the
        // patch rewrites the placeholder `0xffffff` immediate(s) so the
        // CMP at bridge entry reflects the bridge's true requirement.
        Self::patch_stack_checks(self.frame_depth, rawstart, &self.frame_depth_to_patch);

        if crate::majit_dump_enabled() {
            let code = unsafe { std::slice::from_raw_parts(rawstart as *const u8, buffer.len()) };
            eprintln!(
                "[dynasm] BRIDGE CODE DUMP ({} bytes at {:#x}, entry +{:?}):",
                code.len(),
                rawstart,
                entry
            );
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.first().copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Position is the canonical fail_index identity (matching
        // `llsupport/assembler.py`'s `_allgcrefs` index — PyPy does not
        // carry per-emission `fail_index` on the descr itself).  Codegen
        // increments the `fail_index` counter in lockstep with
        // `fail_descrs.push`, so the contract is structural rather than
        // descr-internal.  The earlier per-descr assertion was a pyre
        // Deviation removed: singleton FINISH
        // descrs (`compile.py:623-662`) answer the trait-default `0`
        // for `fail_index_per_trace()` regardless of their Vec position.
        Ok(CompiledCode {
            buffer,
            entry_offset: entry,
            fail_descrs: self.fail_descrs.into_boxed_slice(),
            input_types: self.input_types,
            cpu_attachments: self.cpu_handle,
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            frame_depth: std::sync::atomic::AtomicUsize::new(self.frame_depth),
            source_guard: Some((fail_descr.trace_id(), fail_descr.fail_index_per_trace())),
        })
    }

    /// assembler.py:779 _assemble — walk operations and emit code.
    ///
    /// Uses the register allocator (regalloc.rs) to assign registers/frame
    /// locations, then emits code using those locations. This replaces the
    /// old frame-slot model where every value went through [rbp+offset].
    fn _assemble(&mut self, emit_prologue: bool) -> Result<(), BackendError> {
        let inputargs: &'a [InputArg] = self.inputargs;
        let ops: &'a [Op] = self.operations;
        if emit_prologue {
            self._call_header(inputargs);
        } else {
            self.setup_input_state(inputargs);
        }
        let input_slot_depth = self.next_slot;

        // ── Run register allocator ──
        // assembler.py:537 prepare_loop / assembler.py:638 prepare_bridge
        if crate::majit_j2plan_log_enabled() {
            let plan = crate::j2plan::TracePlan::build(inputargs, ops);
            // Independent debug toggle — not gated by MAJIT_LOG.
            eprintln!("[dynasm:j2plan] {}", plan.summary());
        }

        let mut ra = RegAlloc::new(
            self.constants.clone(),
            self.constant_types.clone(),
            inputargs,
            ops,
        );
        let is_bridge = self.bridge_input_locs.is_some();
        if let Some(ref arglocs) = self.bridge_input_locs {
            ra.prepare_bridge(arglocs);
        } else {
            ra.prepare_loop();
        }
        // assembler.py:647 — bridges emit `_check_frame_depth` between
        // `prepare_bridge` and `_update_at_exit` so the JIT can grow the
        // in-flight JITFRAME if the bridge's frame_depth exceeds the
        // loop's allocation.  Loops skip this (PyPy line 544 uses
        // `_check_frame_depth_debug`, a no-op outside DEBUG_FRAME_DEPTH).
        if is_bridge {
            let gcmap = ra.get_gcmap(&[], false);
            self.emit_check_frame_depth(gcmap);
        }
        // assembler.py:374 walk_operations — get allocation decisions.
        let ra_ops = ra.walk_operations();
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
                .insert(iarg.opref(), JITFRAME_FIXED_SIZE + iarg.index as usize);
        }
        // Also sync any frame allocations from regalloc's FrameManager.
        for (&opref, lifetime) in ra.longevity.lifetimes_iter() {
            if let Some(floc) = lifetime.current_frame_loc {
                self.opref_to_slot
                    .insert(opref, JITFRAME_FIXED_SIZE + floc.position);
            }
        }
        // frame_slot_depth is already absolute (see calculation above).
        self.next_slot = frame_slot_depth;

        let mut fail_index = 0u32;

        if crate::majit_log_enabled() {
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
                    if majit_ir::debug::have_debug_prints() {
                        majit_ir::debug::log_one(
                            "jit-backend",
                            &format!("move: {src:?} → {dst:?}"),
                        );
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
                    if crate::majit_log_enabled() {
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
                }
                RegAllocOp::PerformGuard {
                    op_index,
                    arglocs,
                    result_loc,
                    faillocs,
                } => {
                    let op = &ops[*op_index];
                    if crate::majit_log_enabled() {
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
                }
                RegAllocOp::PerformDiscard { op_index, arglocs } => {
                    let op = &ops[*op_index];
                    if crate::majit_log_enabled() {
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

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] _assemble done: pending_guard_tokens={} fail_index={}",
                self.pending_guard_tokens.len(),
                fail_index
            );
        }

        // assembler.py:1003-1008 `_assemble`: grow the frame to fit a
        // cross-loop JUMP target.  The closing `JMP` jumps into the target
        // loop's body, which can use deeper frame slots than this trace; for
        // a bridge the prologue `_check_frame_depth` then reallocs the live
        // JITFRAME to this grown depth on entry.  `target_frame_depth` is
        // full-width (includes JITFRAME_FIXED_SIZE), matching `frame_depth`,
        // so no adjustment is needed.  Zero (no external JUMP) is a no-op.
        self.frame_depth = self.frame_depth.max(self.jump_target_frame_depth);

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
            // x86/assembler.py:1881 genop_int_add uses LEA, not ADD, because
            // regalloc.py:566 consider_int_add routes 32-bit constants through
            // `_consider_lea`, which force-allocates a fresh result register
            // (independent of arg0). Emitting `add dst, src` here would
            // operate on whatever stale value Rq(dst) still held — for the
            // fib_loop trace this turned `t = i + 1` into `t = n_obj_ptr + 1`,
            // poisoning the new W_IntObject and tripping GuardTrue on the
            // next iteration. LEA also handles the consider_binop_symm path
            // where `dst == arg0`: `lea dst, [dst + src]` is identical to
            // `add dst, src`.
            OpCode::IntAdd | OpCode::NurseryPtrIncrement => {
                if let (Some(Loc::Reg(dst)), Some(a0), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    match (a0, src) {
                        (Loc::Reg(a), Loc::Reg(s)) => dynasm!(self.mc ; .arch x64
                            ; lea Rq(dst.value), [Rq(a.value) + Rq(s.value)]),
                        (Loc::Reg(a), Loc::Immed(i)) => {
                            let v = i.value as i32;
                            dynasm!(self.mc ; .arch x64
                                ; lea Rq(dst.value), [Rq(a.value) + v])
                        }
                        (Loc::Immed(i), Loc::Reg(s)) => {
                            let v = i.value as i32;
                            dynasm!(self.mc ; .arch x64
                                ; lea Rq(dst.value), [Rq(s.value) + v])
                        }
                        (Loc::Immed(i0), Loc::Immed(i1)) => {
                            let sum = i0.value.wrapping_add(i1.value);
                            dynasm!(self.mc ; .arch x64
                                ; mov Rq(dst.value), QWORD sum)
                        }
                        _ => self.emit_binop_reg_loc(op.opcode, dst.value, src),
                    }
                }
            }
            // x86/assembler.py:1268-1284 `_binaryop_or_lea(asmop='SUB',
            // is_add=False)`: when `result_loc is arglocs[0]` emit
            // `SUB dst, src` in place; otherwise the regalloc routed
            // through `_consider_lea` (consider_int_sub at
            // x86/regalloc.py:575) and produced a fresh result register,
            // and we must emit `LEA result_loc, [arglocs[0] - delta]`
            // — never `SUB dst, src`, which would corrupt `dst`'s stale
            // value (the bug seen in fannkuch as `q.int_items.ptr - 1`
            // landing in a fresh result register that previously held
            // the base pointer).  IntMul / IntAnd / IntOr / IntXor never
            // take the LEA path (regalloc.rs:2960 routes them through
            // `consider_binop_symm` which keeps result==arglocs[0]), so
            // a plain in-place op is correct for them.
            OpCode::IntSub => {
                if let (Some(Loc::Reg(dst)), Some(a0), Some(src)) =
                    (result_loc, arglocs.first(), arglocs.get(1))
                {
                    let same_as_lhs = matches!(a0, Loc::Reg(a) if a.value == dst.value);
                    if same_as_lhs {
                        self.emit_binop_reg_loc(op.opcode, dst.value, src);
                    } else {
                        match (a0, src) {
                            (Loc::Reg(a), Loc::Immed(i)) => {
                                // regalloc.py:577 — `_consider_lea` is guarded
                                // by `rx86.fits_in_32bits(-y.value)`. The
                                // wrapping negate matches PyPy `-y.value`;
                                // `y.value = 2147483648` is valid because
                                // disp32 = `-2147483648`. `i32::try_from`
                                // panics if the regalloc guard somehow let
                                // through a non-encodable value.
                                let v = i32::try_from(i.value.wrapping_neg()).expect(
                                    "IntSub LEA requires an immediate \
                                         encodable as signed disp32 after negation",
                                );
                                dynasm!(self.mc ; .arch x64
                                    ; lea Rq(dst.value), [Rq(a.value) + v])
                            }
                            _ => panic!(
                                "IntSub: result_loc != arglocs[0] requires LEA form \
                                 (arglocs[0]=Reg, arglocs[1]=Immed); got a0={a0:?} src={src:?}",
                            ),
                        }
                    }
                }
            }
            OpCode::IntMul | OpCode::IntAnd | OpCode::IntOr | OpCode::IntXor => {
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
            // x86/assembler.py:1301 `_cmpop` + 1286 `flush_cc` parity.
            // When the regalloc picks `frame_reg` (rbp) as the result
            // sentinel, the comparison's outcome lives in the condition
            // flags and the following guard consumes it directly —
            // saving the SETcc + MOVZX (+ later TEST) per CompOp. The
            // result-in-register path emits the boolean materialisation
            // as before for cases where the value is also read by a
            // non-guard consumer.
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
                let cc = Self::opcode_to_cc(op.opcode);
                self.flush_cc(cc, result_loc);
            }
            OpCode::IntIsTrue => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_NE, result_loc);
                }
            }
            OpCode::IntIsZero => {
                if let Some(src) = arglocs.first() {
                    self.emit_test_loc(src);
                    self.flush_cc(CC_E, result_loc);
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
                if let (Some(a_loc), Some(b_loc)) = (arglocs.first(), arglocs.get(1)) {
                    let b_reg = match b_loc {
                        Loc::Reg(b) => Some(*b),
                        _ => None,
                    };
                    let a = if let Loc::Reg(a) = a_loc {
                        *a
                    } else {
                        let scratch = if b_reg
                            .map_or(false, |b| b.value == crate::regloc::XMM15.value && b.is_xmm)
                        {
                            crate::regloc::XMM14
                        } else {
                            crate::regloc::XMM15
                        };
                        self.regalloc_mov(a_loc, &Loc::Reg(scratch));
                        scratch
                    };
                    let b = if let Loc::Reg(b) = b_loc {
                        *b
                    } else {
                        let scratch = if a.value == crate::regloc::XMM15.value && a.is_xmm {
                            crate::regloc::XMM14
                        } else {
                            crate::regloc::XMM15
                        };
                        self.regalloc_mov(b_loc, &Loc::Reg(scratch));
                        scratch
                    };
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
            | OpCode::CastOpaquePtr
            | OpCode::LoadFromGcTable
            | OpCode::VirtualRefR
            | OpCode::ConvertFloatBytesToLonglong
            | OpCode::ConvertLonglongBytesToFloat => {
                if let (Some(src), Some(dst)) = (arglocs.first(), result_loc) {
                    self.regalloc_mov(src, dst);
                }
            }
            // `assembler.py:1528 genop_cast_ptr_to_int = _genop_same_as`
            // / `:1529 genop_cast_int_to_ptr = _genop_same_as`.  PyPy's
            // x86 backend treats both casts as plain `mov` — the
            // AddressAsInt low-bit tag is a `blackhole.py:603-610`
            // interpreter-side software invariant, not a backend
            // codegen step.  Tagging at codegen would fold a fake odd
            // pointer back into the raw aligned-pointer space and
            // could collide with a real GC pointer.  `runner_test.py:
            // 1957 cast_int_to_ptr(-17) -> cast_ptr_to_int == -17`
            // expects strict identity through the compiled trace.
            OpCode::CastPtrToInt | OpCode::CastIntToPtr => {
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
                // regalloc.py:1154-1167 `_consider_gc_load` parity: both
                // `base_loc = self.rm.make_sure_var_in_reg(op.getarg(0), args)`
                // and `result_loc = self.force_allocate_reg(op)` force
                // register materialisation — pyre's
                // `consider_getfield_j2` (regalloc.rs:4309-4321) does the
                // same.  Silently no-op'ing on a non-Reg base or result
                // would mask a regalloc bug (e.g. a fresh `GETFIELD_GC`
                // arm that forgot the `make_sure_var_in_reg` call), so
                // surface the invariant violation explicitly.
                let base = match arglocs.first() {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "GetfieldGc/Strlen/Unicodelen/ArraylenGc base must be Loc::Reg \
                         (regalloc.py:1156 make_sure_var_in_reg invariant), got {other:?}",
                    ),
                };
                let dst = match result_loc {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "GetfieldGc/Strlen/Unicodelen/ArraylenGc result_loc must be Loc::Reg \
                         (regalloc.py:1158 force_allocate_reg invariant), got {other:?}",
                    ),
                };
                let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
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
                        .with_array_descr(|ad| {
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
                    let ofs = op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0);
                    let field_size = op.with_field_descr(|fd| fd.field_size()).unwrap_or(8);
                    self.emit_op_setfield_regalloc(base, val_loc, ofs, field_size);
                } else {
                    self.genop_discard_setfield(op);
                }
            }
            // arglocs = [base_loc, ofs_loc, res_loc, imm(nsize)].
            // `base_loc` may be Loc::Immed when the load is from a
            // constant pointer (e.g. `GcLoadI(jfi_descr_ptr, 8, 8)` for
            // the JITFRAME size in CallMallocNurseryVarsizeFrame).
            // llsupport/regalloc.py:625 return_constant returns the
            // bare Loc::Immed in that case and the assembler is
            // responsible for materializing it. Mirror aarch64 by
            // staging the constant through the scratch register (R11);
            // dropping the load left the destination register holding
            // stale heap pointers, which the varsize-frame slowpath
            // then dereferenced as an allocation size and tripped a
            // multi-terabyte OOM (fib_recursive on x86).
            OpCode::GcLoadI
            | OpCode::GcLoadR
            | OpCode::GcLoadF
            | OpCode::RawLoadI
            | OpCode::RawLoadF => {
                if let Some(ofs_loc) = arglocs.get(1) {
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
                            .with_array_descr(|ad| {
                                let s = ad.item_size() as i64;
                                if ad.is_item_signed() { -s } else { s }
                            })
                            .unwrap_or(8),
                    };
                    match arglocs.first() {
                        Some(Loc::Reg(base)) => {
                            self.emit_op_gcload_regalloc(base, ofs_loc, dst, nsize);
                        }
                        Some(Loc::Immed(base_i)) => {
                            let scratch = crate::regloc::X86_64_SCRATCH_REG;
                            // regalloc.rs materializes out-of-range
                            // offsets into LARGE_IMM_SCRATCH (R11). If
                            // we land in that case while base is also
                            // an immediate we cannot share R11.
                            if let Loc::Reg(r) = ofs_loc {
                                assert!(
                                    r.value != scratch.value,
                                    "GcLoad: base=Immed and ofs already occupies R11",
                                );
                            }
                            dynasm!(self.mc ; .arch x64
                                ; mov Rq(scratch.value), QWORD base_i.value);
                            self.emit_op_gcload_regalloc(&scratch, ofs_loc, dst, nsize);
                        }
                        other => {
                            panic!("GcLoad base_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                        }
                    }
                }
            }
            // ── GC store / raw store: opassembler.rs emit_op_gcstore_regalloc ──
            // arglocs = [value_loc, base_loc, ofs_loc, size_loc].
            // value_loc may be Loc::Immed when the source is a Const
            // (llsupport/regalloc.py:625 return_constant), so the emitter
            // must materialize it before the store — silently dropping
            // such writes left newly-allocated objects without vtables
            // and triggered downstream GuardClass failures.
            OpCode::GcStore | OpCode::RawStore => {
                let (value_loc, base_loc, ofs_loc, size_loc) = match arglocs {
                    [v, b, o, s] => (v, b, o, s),
                    _ => panic!(
                        "GcStore arglocs must be [value, base, ofs, size] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcStore base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let size = match size_loc {
                    Loc::Immed(i) => i.value.unsigned_abs() as usize,
                    other => panic!(
                        "GcStore size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                match value_loc {
                    Loc::Reg(val) => {
                        self.emit_op_gcstore_regalloc(base, ofs_loc, val, size);
                    }
                    Loc::Immed(val_imm) => {
                        self.emit_op_gcstore_imm_regalloc(base, ofs_loc, val_imm.value, size);
                    }
                    other => {
                        panic!("GcStore value_loc must be Loc::Reg or Loc::Immed, got {other:?}")
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
            // ── x86/assembler.py:1701 _genop_gc_load_indexed ──
            // Line-by-line port:
            //   base_loc, ofs_loc, scale_loc, offset_loc, size_loc, sign_loc = arglocs
            //   scale = get_scale(scale_loc.value)
            //   src_addr = addr_add(base_loc, ofs_loc, offset_loc.value, scale)
            //   self.load_from_mem(resloc, src_addr, size_loc, sign_loc)
            //
            // The regalloc passes the raw byte stride in `scale_loc`
            // (x86/regalloc.py:1184); `get_scale` converts 1/2/4/8 →
            // 0/1/2/3 SIB exponents. PyPy keeps the (1,2,4,8) check
            // implicit through `valid_addressing_size`; we surface the
            // unsupported factors as a panic to keep miscompiles loud.
            //
            // KNOWN ISSUE: this emission triggers a timing-dependent
            // segfault on `spectral_norm`-style traces (function call +
            // many iterations + later loop). Adding any eprintln in the
            // dispatch hides the bug, so it is likely a stale
            // base-array pointer surviving a minor GC during the inline
            // jitframe-alloc fast path.
            OpCode::GcLoadIndexedI | OpCode::GcLoadIndexedR | OpCode::GcLoadIndexedF => {
                let (base_loc, ofs_loc, scale_loc, offset_loc, size_loc, sign_loc) = match arglocs {
                    [b, o, sc, of, sz, sg] => (b, o, sc, of, sz, sg),
                    _ => panic!(
                        "GcLoadIndexed arglocs must be [base, ofs, scale, offset, size, sign] (got {} locs)",
                        arglocs.len(),
                    ),
                };
                let base = match base_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcLoadIndexed base_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let ofs_reg = match ofs_loc {
                    Loc::Reg(r) => r,
                    other => panic!(
                        "GcLoadIndexed ofs_loc must be Loc::Reg (regalloc contract), got {other:?}",
                    ),
                };
                let factor = match scale_loc {
                    Loc::Immed(i) => i.value,
                    other => panic!(
                        "GcLoadIndexed scale_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let offset = match offset_loc {
                    Loc::Immed(i) => i.value as i32,
                    other => panic!(
                        "GcLoadIndexed offset_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let size = match size_loc {
                    Loc::Immed(i) => i.value as usize,
                    other => panic!(
                        "GcLoadIndexed size_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let sign = match sign_loc {
                    Loc::Immed(i) => i.value != 0,
                    other => panic!(
                        "GcLoadIndexed sign_loc must be Loc::Immed (regalloc contract), got {other:?}",
                    ),
                };
                let dst = match result_loc {
                    Some(Loc::Reg(r)) => r,
                    other => panic!("GcLoadIndexed result_loc must be Loc::Reg, got {other:?}",),
                };

                // assembler.py:1645 `load_from_mem`: dispatch by (resloc.is_xmm,
                // size, sign). PyPy's `addr_add` returns `AddressLoc(base,
                // ofs, scale, disp)` which the encoder materializes as a
                // SIB scaled-index addressing mode straight on the MOV
                // template. dynasm-rs requires the SIB scale as a literal
                // at macro expansion time, so we dispatch (factor, size,
                // sign) through a `match` arm — functionally identical to
                // PyPy's single `mc.MOV*(resloc, src_addr)` once the
                // factor is bound. xmm targets always use MOVSD per
                // load_from_mem:1649; integer targets pick MOV /
                // MOVZX{8,16} / MOVSX{8,16,32} based on size+sign.
                // assembler.py:1645 load_from_mem allows WORD, 1, 2, and
                // (x86_64) 4; any other size is `not_implemented`.
                macro_rules! emit_load_scaled {
                    ($scale:tt) => {{
                        if dst.is_xmm {
                            dynasm!(self.mc ; .arch x64
                                ; movsd Rx(dst.value), [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                        } else {
                            match size {
                                1 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsx Rq(dst.value), BYTE [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; movzx Rq(dst.value), BYTE [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    }
                                }
                                2 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsx Rq(dst.value), WORD [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; movzx Rq(dst.value), WORD [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    }
                                }
                                4 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsxd Rq(dst.value), DWORD [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; mov Rd(dst.value), [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]);
                                    }
                                }
                                8 => dynasm!(self.mc ; .arch x64
                                    ; mov Rq(dst.value), [Rq(base.value) + Rq(ofs_reg.value) * $scale + offset]),
                                other => panic!(
                                    "load_from_mem: size {other} not in {{1, 2, 4, WORD}}"
                                ),
                            }
                        }
                    }};
                }
                macro_rules! emit_load_unscaled {
                    () => {{
                        if dst.is_xmm {
                            dynasm!(self.mc ; .arch x64
                                ; movsd Rx(dst.value), [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                        } else {
                            match size {
                                1 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsx Rq(dst.value), BYTE [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; movzx Rq(dst.value), BYTE [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    }
                                }
                                2 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsx Rq(dst.value), WORD [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; movzx Rq(dst.value), WORD [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    }
                                }
                                4 => {
                                    if sign {
                                        dynasm!(self.mc ; .arch x64
                                            ; movsxd Rq(dst.value), DWORD [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    } else {
                                        dynasm!(self.mc ; .arch x64
                                            ; mov Rd(dst.value), [Rq(base.value) + Rq(ofs_reg.value) + offset]);
                                    }
                                }
                                8 => dynasm!(self.mc ; .arch x64
                                    ; mov Rq(dst.value), [Rq(base.value) + Rq(ofs_reg.value) + offset]),
                                other => panic!(
                                    "load_from_mem: size {other} not in {{1, 2, 4, WORD}}"
                                ),
                            }
                        }
                    }};
                }
                match factor {
                    1 => emit_load_unscaled!(),
                    2 => emit_load_scaled!(2),
                    4 => emit_load_scaled!(4),
                    8 => emit_load_scaled!(8),
                    other => panic!(
                        "x86 GcLoadIndexed: unsupported factor {other}; \
                         load_supported_factors = (1, 2, 4, 8)"
                    ),
                }
            }
            // Structural adaptation: PyPy's llsupport/rewrite.py:132-154
            // normally lowers SETARRAYITEM_* to GC_STORE(_INDEXED), but
            // pyre's CI also exercises direct backend emission paths before
            // that rewrite has run.
            OpCode::SetarrayitemGc | OpCode::SetarrayitemRaw => {
                if let (Some(Loc::Reg(base)), Some(index_loc), Some(value_loc)) =
                    (arglocs.first(), arglocs.get(1), arglocs.get(2))
                {
                    let (base_size, item_size, is_ref_array) = op
                        .with_array_descr(|ad| {
                            (
                                ad.base_size() as i32,
                                ad.item_size() as i32,
                                ad.is_array_of_pointers(),
                            )
                        })
                        .unwrap_or((0, 8, false));
                    let index_reg = crate::regloc::X86_64_SCRATCH_REG.value;
                    // r11 is already the computed destination address here.
                    // Stage non-register values through a saved GPR: r12 is
                    // allocatable on x86-64, so using SCRATCH_REG_2 would
                    // clobber live regalloc state.
                    let value_reg = crate::regloc::EAX.value;

                    let store_may_need_wb = op.opcode == OpCode::SetarrayitemGc
                        && is_ref_array
                        && item_size as usize == WORD
                        && self.setarrayitem_value_needs_write_barrier(
                            op.arg(2).to_opref(),
                            value_loc,
                        );
                    if store_may_need_wb {
                        self.emit_setarrayitem_gc_write_barrier(&arglocs[..2]);
                    }

                    self.regalloc_mov(
                        index_loc,
                        &Loc::Reg(crate::regloc::RegLoc::new(index_reg, false)),
                    );
                    if item_size != 1 {
                        dynasm!(self.mc ; .arch x64 ; imul Rq(index_reg), Rq(index_reg), item_size);
                    }
                    if base_size != 0 {
                        dynasm!(self.mc ; .arch x64 ; add Rq(index_reg), base_size);
                    }
                    dynasm!(self.mc ; .arch x64 ; add Rq(index_reg), Rq(base.value));

                    match value_loc {
                        Loc::Reg(val) if val.is_xmm => {
                            dynasm!(self.mc ; .arch x64 ; movsd [Rq(index_reg)], Rx(val.value));
                        }
                        Loc::Reg(val) => match item_size {
                            1 => dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rb(val.value)),
                            2 => dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rw(val.value)),
                            4 => dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rd(val.value)),
                            _ => dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rq(val.value)),
                        },
                        _ => {
                            dynasm!(self.mc ; .arch x64 ; push rax);
                            self.regalloc_mov(
                                value_loc,
                                &Loc::Reg(crate::regloc::RegLoc::new(value_reg, false)),
                            );
                            match item_size {
                                1 => {
                                    dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rb(value_reg))
                                }
                                2 => {
                                    dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rw(value_reg))
                                }
                                4 => {
                                    dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rd(value_reg))
                                }
                                _ => {
                                    dynasm!(self.mc ; .arch x64 ; mov [Rq(index_reg)], Rq(value_reg))
                                }
                            }
                            dynasm!(self.mc ; .arch x64 ; pop rax);
                        }
                    }
                }
            }
            // ── Control flow ──
            OpCode::Jump => {
                let descr_arc = op.getdescr();
                let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
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
                if majit_ir::debug::have_debug_prints() {
                    let _s = majit_ir::debug::scope("jit-backend");
                    majit_ir::debug::debug_print(&format!(
                        "Jump remap: {} int src→dst, {} float src→dst",
                        src_locations1.len(),
                        src_locations2.len()
                    ));
                    for (i, (s, d)) in src_locations1.iter().zip(dst_locations1.iter()).enumerate()
                    {
                        majit_ir::debug::debug_print(&format!("  int[{i}]: {s:?} → {d:?}"));
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
                    // assembler.py:2461 mc.JMP(imm(target)) — PyPy's
                    // `LocationCodeBuilder._addr_as_reg_offset` (regloc.py:204)
                    // stages a 64-bit absolute target through
                    // `X86_64_SCRATCH_REG = r11`, never RAX.  Using RAX
                    // here clobbers the loop-carried Ref that the
                    // regalloc bound to it: a bridge whose body
                    // succeeds and rejoins the trace loop returns with
                    // RAX = `target` (a code address), and the next
                    // iteration's GuardClass(RAX) then misreads RAX as
                    // a Ref and SEGVs when it dereferences trace + 0x1B3
                    // expecting a class pointer.
                    // assembler.py:1003-1008 `_assemble`: record the target
                    // loop's frame depth so this trace's frame grows to fit it.
                    if let Some(descr) = jump_descr {
                        self.jump_target_frame_depth =
                            self.jump_target_frame_depth.max(descr.target_frame_depth());
                    }
                    let addr = target as i64;
                    let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD addr
                        ; jmp Rq(scratch)
                    );
                }
            }
            OpCode::Finish => {
                // RPython: genop_finish stores result at jf_frame[0] (base_ofs),
                // writes descr ptr to jf_descr, then calls _call_footer.
                // arglocs[0] = result location (if any)
                let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
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
                    .with_fail_descr(|fd| fd.is_exit_frame_with_exception())
                    .unwrap_or(false);
                let global_descr_ptr = if is_exit_exc {
                    self.exit_frame_with_exception_descr_ref_ptr()
                } else {
                    self.done_with_this_frame_descr_ptr_for_type(result_type)
                };
                // FINISH op exit (DoneWithThisFrame* / ExitFrameWithExceptionDescr).
                // `compile.py:185` skips these — not a `ResumeDescr`.
                // `genop_finish` (assembler.py:2114-2156) stamps the
                // metainterp singleton directly into `jf_descr` via the GC
                // table index; pyre's runtime classifier (`runner.rs::
                // find_descr_by_ptr` lines 1115-1151) short-circuits the
                // FINISH/Exit/Propagate ptrs to the cpu-attached singleton
                // before consulting the registry, so the per-emission
                // wrapper has no jf_descr role.  Push the singleton Arc
                // directly.  Test scaffolds must attach singletons (via
                // `attach_default_test_descrs` or `MetaInterp::new` per
                // `pyjitpl.py:2222 finish_setup`) before emitting FINISH.
                let descr: majit_ir::DescrRef = if is_exit_exc {
                    self.cpu_handle
                        .read()
                        .unwrap()
                        .exit_frame_with_exception_descr_ref
                        .clone()
                } else {
                    self.done_with_this_frame_descr_arc_for_type(result_type)
                }
                .expect(
                    "FINISH emission requires cpu-attached singleton — \
                     call `attach_default_test_descrs` or use `MetaInterp::new`",
                );

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
                // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
                // not the cell pointer.  `handle_fail_done_with_this_frame`
                // and `handle_fail_exit_frame_with_exception` match by
                // ptr-equality on the singleton, so the cell only carries
                // the keep-alive identity for `clt.asmmemmgr_gcreftracers`.
                self.fail_descrs
                    .push(majit_ir::FailDescrCell::wrap(descr.clone()));
            }
            OpCode::Label => {
                let label = self.mc.new_dynamic_label();
                let descr_arc = op.getdescr();
                let label_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
                if majit_ir::debug::have_debug_prints() {
                    majit_ir::debug::log_one(
                        "jit-backend",
                        &format!("LABEL: new DynamicLabel({label:?})"),
                    );
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
                    if let Some(id) = descr_arc.as_ref().map(majit_ir::descr_identity) {
                        self.target_tokens_currently_compiling.insert(id, label);
                    }
                    if let Some(descr_ref) = descr_arc.as_ref() {
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
            // assembler.py:2567 malloc_cond_varsize_frame parity.
            // The varsize_frame call site shares the entire slowpath
            // structure with the fixed-size variant: PyPy's
            // `MallocCondSlowPath` (line 2551) does `CALL malloc_slowpath`
            // regardless of which caller pushed it, and the slowpath
            // recovers `total_size = edx - ecx` either way.  Pyre
            // mirrors that — both arms emit the ECX/EDX bump-allocator
            // probe and `JA` into the same `malloc_slowpath_fixed`
            // trampoline.  The trampoline's gcmap-aware save/restore
            // preserves every regalloc-resident Ref across a minor
            // collection fired by the slowpath (fib_recursive on x86
            // exercised this path).
            OpCode::CallMallocNurseryVarsizeFrame => {
                let sizeloc = match arglocs.first() {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "CallMallocNurseryVarsizeFrame size arg must be Loc::Reg, got {other:?}",
                    ),
                };
                let result_reg = match result_loc {
                    Some(Loc::Reg(r)) => *r,
                    other => panic!(
                        "CallMallocNurseryVarsizeFrame result_loc must be Loc::Reg, got {other:?}",
                    ),
                };
                let sv = sizeloc.value;
                let rv = result_reg.value;
                // `MALLOC_NURSERY_CLOBBER` spills any live variable
                // out of RCX/RDX before this op, so `sizeloc` is
                // never in those registers and is disjoint from the
                // ECX/EDX probe pair.  R11 (X86_64_SCRATCH_REG) loads
                // the absolute nursery_free/nursery_top addresses.
                let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();
                let slow_path = self.mc.new_dynamic_label();
                let done = self.mc.new_dynamic_label();
                let gc_header_size = majit_gc::header::GcHeader::SIZE as i32;
                let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                if nf_addr == 0 || nt_addr == 0 {
                    dynasm!(self.mc ; .arch x64 ; jmp =>slow_path);
                } else {
                    // assembler.py:2572-2581 line-by-line — `MOV ecx,
                    // [nf]; LEA edx, [ecx + sizeloc + gc_hdr]; CMP edx,
                    // [nt]; JA slow; MOV [nf], edx`.  PyPy's LEA omits
                    // `gc_hdr` because its allocator accounts for the
                    // header inside the helper; pyre adds it here so
                    // the trampoline's `SUB rdx, rcx` recovers the
                    // exact byte count `dynasm_nursery_slowpath`
                    // expects (total bytes including header).
                    dynasm!(self.mc ; .arch x64
                        ; mov Rq(scratch), QWORD nf_addr as i64
                        ; mov rcx, [Rq(scratch)]
                        ; lea rdx, [rcx + Rq(sv) + gc_header_size]
                        ; mov Rq(scratch), QWORD nt_addr as i64
                        ; cmp rdx, [Rq(scratch)]
                        ; ja =>slow_path
                        ; mov Rq(scratch), QWORD nf_addr as i64
                        ; mov [Rq(scratch)], rdx
                        ; mov QWORD [rcx], 0
                        ; lea Rq(rv), [rcx + gc_header_size]
                        ; jmp =>done
                    );
                }
                dynasm!(self.mc ; .arch x64 ; =>slow_path);
                // Trampoline entry contract (assembler.py:264
                // `SUB_rr(edx, ecx)` → total size): caller must hand
                // off `rcx = old_nf` and `rdx = old_nf + total`, just
                // like `malloc_cond` does.  The `nf_addr == 0` guard
                // path above skipped the probe, so re-stage the
                // operands here.  `gc_header_size` is added so the
                // slowpath's `dynasm_nursery_slowpath(total)` sees the
                // same byte count the fast path proposed.  With no
                // active GC descriptor the previous code dereferenced
                // `[nf_addr]` where `nf_addr == 0` (crash before the
                // helper-only fallback ever ran), so synthesize
                // `(rcx=0, rdx=total)` directly — `sub edx, ecx`
                // recovers `total` for the trampoline either way.
                if nf_addr == 0 || nt_addr == 0 {
                    dynasm!(self.mc ; .arch x64
                        ; xor rcx, rcx
                        ; lea rdx, [Rq(sv) + gc_header_size]
                    );
                }
                if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
                    self.push_gcmap(gcmap as *mut usize);
                } else {
                    let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
                    dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                }
                // Stage the trampoline address through R11 so RAX
                // stays caller-live across the call (the trampoline
                // saves+restores it via the `[ECX, EDX]`-ignored
                // push_all_regs).
                let helper_addr = self.malloc_slowpath_fixed as i64;
                let call_scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                dynasm!(self.mc ; .arch x64
                    ; mov Rq(call_scratch), QWORD helper_addr
                    ; call Rq(call_scratch)
                );
                // assembler.py:304 — helper returns the payload in
                // ECX (`MOV_rr(ecx, eax)` inside the trampoline).
                // regalloc forces `result_reg = MALLOC_NURSERY_RESULT
                // = ECX`, so the MOV is elided in the common case.
                if result_reg.value != crate::regloc::ECX.value {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(rv), rcx);
                }
                dynasm!(self.mc ; .arch x64 ; =>done);
                if !op.pos.get().is_none() {
                    if result_reg.value != 0 {
                        dynasm!(self.mc ; .arch x64 ; mov rax, Rq(result_reg.value));
                    }
                    self.store_rax_to_result(op.pos.get());
                }
            }
            // x86/assembler.py:2567 malloc_cond_varsize parity
            // arglocs = [lengthloc, imm(itemsize), imm(kind)]
            OpCode::CallMallocNurseryVarsize => {
                let base_size = op.with_array_descr(|ad| ad.base_size()).unwrap_or(16) as i64;
                let itemsize = match arglocs.get(1) {
                    Some(Loc::Immed(i)) => i.value,
                    _ => 8,
                };
                self.emit_abi_int_arg_from_imm(0, base_size);
                self.emit_abi_int_arg_from_imm(1, itemsize);
                match arglocs.first() {
                    Some(Loc::Reg(len_r)) => {
                        self.emit_abi_int_arg_from_reg(2, len_r.value as u8);
                    }
                    Some(Loc::Immed(len_i)) => {
                        self.emit_abi_int_arg_from_imm(2, len_i.value);
                    }
                    _ => {
                        self.emit_abi_int_arg_from_imm(2, 0);
                    }
                }
                let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD crate::runner::dynasm_nursery_slowpath_varsize as *const () as i64
                );
                self.emit_abi_call_rax();
                // _build_malloc_slowpath parity (assembler.py:295-308):
                // reload the (possibly moved) jitframe before clearing the
                // gcmap, otherwise the clear would target the freed
                // nursery copy.
                self.reload_frame_if_necessary();
                // assembler.py:300-322 OOM propagate parity — the
                // varsize helper now returns NULL on `libc::calloc`
                // / `gc.alloc_varsize` failure; route that through
                // `propagate_exception_descr` rather than letting the
                // caller store a near-zero garbage pointer into the
                // result slot.
                self.emit_propagate_exception_if_zero(0);
                dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
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
                let reg = match arglocs.first() {
                    Some(Loc::Reg(r)) if !r.is_xmm => r.value,
                    _ => panic!("CheckMemoryError arglocs[0] must be a non-xmm register"),
                };
                self.emit_propagate_exception_if_zero(reg);
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
            // Guards never reach the non-guard regalloc dispatch — they
            // are emitted exclusively from `regalloc_perform_guard` via
            // the `RegAllocOp::PerformWithGuard` arm
            // (`assemble_loop` dispatch at line 1507).
            _ if op.opcode.is_guard() => unreachable!(
                "regalloc_perform reached with guard {:?}; guards must \
                 route through regalloc_perform_guard",
                op.opcode
            ),
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
            // x86/assembler.py:1773 `genop_guard_guard_true` is a bare
            // `implement_guard(guard_token)` — the regalloc routed the
            // condition through `load_condition_into_cc`, which either
            // reuses the cc from the prior CompOp (CC fusion) or emits
            // a TEST itself before this point. Mirror that here.
            OpCode::GuardTrue | OpCode::VecGuardTrue | OpCode::GuardNonnull => {
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.implement_guard_with_faillocs(op, op_index, fail_index, faillocs);
            }
            // x86/assembler.py:1777 `genop_guard_guard_false` inverts
            // the published cc, then implements. So a fused IntLt that
            // set CC_L turns into a CC_GE failure jump under GuardFalse.
            OpCode::GuardFalse | OpCode::VecGuardFalse | OpCode::GuardIsnull => {
                if let Some(loc) = arglocs.first() {
                    self.load_condition_into_cc(loc);
                }
                self.guard_success_cc = self.guard_success_cc.map(invert_cc);
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

    /// Helper: guard class comparison.
    /// x86/assembler.py:1880 `_cmp_guard_class` emits a single
    /// `CMP [obj + vtable_offset], classptr` so the object register is
    /// never touched. Mirror that: for register and 32-bit-fitting
    /// immediate classptrs we emit the memory-operand CMP directly; for
    /// 64-bit immediates we stage through the dedicated scratch (R11)
    /// rather than RAX, which may itself hold `obj_loc`. The earlier
    /// `mov rax, imm` clobbered `obj_loc` when the regalloc placed the
    /// object in RAX, leaving subsequent uses (e.g. the immediately
    /// following `move: Reg(0) → Frame(pos=N)`) writing the vtable
    /// constant into the deopt slot.
    fn _cmp_guard_class(&mut self, obj_loc: &Loc, class_loc: &Loc) {
        // Caller (genop_guard_guard_class) sets `guard_success_cc =
        // Some(CC_E)` immediately after this returns, so any path that
        // fails to emit a CMP would branch on stale flags from a
        // preceding instruction. Fail closed instead.
        let Loc::Reg(obj) = obj_loc else {
            panic!("GuardClass: obj_loc must be Loc::Reg, got {obj_loc:?}");
        };
        if let Some(vtable_offset) = self.vtable_offset {
            let ofs = vtable_offset as i32;
            match class_loc {
                Loc::Reg(c) => {
                    dynasm!(self.mc ; .arch x64
                        ; cmp QWORD [Rq(obj.value) + ofs], Rq(c.value));
                }
                Loc::Immed(i) => {
                    let fits_imm32 = (i.value as i32) as i64 == i.value;
                    if fits_imm32 {
                        dynasm!(self.mc ; .arch x64
                            ; cmp QWORD [Rq(obj.value) + ofs], i.value as i32);
                    } else {
                        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
                        dynasm!(self.mc ; .arch x64
                            ; mov Rq(scratch), QWORD i.value
                            ; cmp QWORD [Rq(obj.value) + ofs], Rq(scratch));
                    }
                }
                other => panic!(
                    "GuardClass (vtable form): class_loc must be Loc::Reg or Loc::Immed, got {other:?}",
                ),
            }
        } else {
            let Loc::Immed(i) = class_loc else {
                panic!("GuardClass (typeid form): class_loc must be Loc::Immed, got {class_loc:?}",);
            };
            let expected_typeid = self
                .lookup_typeid_from_classptr(i.value as usize)
                .expect("GuardClass: missing typeid for classptr");
            self._cmp_guard_gc_type(
                &Loc::Reg(*obj),
                &Loc::Immed(crate::regloc::ImmedLoc::new(expected_typeid as i64)),
            );
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
        // Callers (guard_class typeid form, guard_gc_type, ...) rely on
        // CC_E being set from this CMP. A silent no-op would leave the
        // guard branching on stale flags.
        let Loc::Reg(obj) = obj_loc else {
            panic!("guard_gc_type: obj_loc must be Loc::Reg, got {obj_loc:?}");
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
            other => {
                panic!("guard_gc_type: expected_typeid_loc must be Reg/Frame/Immed, got {other:?}",)
            }
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

    /// `assembler.py:328-345 _build_propagate_exception_path` inline.
    /// Emits `TEST reg, reg; JNZ skip; <propagate body>; skip:` —
    /// the per-site propagate sequence used by both `CheckMemoryError`
    /// (assembler.py:1630 `genop_discard_check_memory_error`) and the
    /// caller-side `CallMallocNursery` OOM check (`assembler.py:300-322`,
    /// which PyPy emits inside the shared `_build_malloc_slowpath`).
    ///
    /// `_call_footer` overwrites `rax` with `rbp` before returning, so
    /// `rax` is freely clobberable on the propagate path.  No-op when
    /// `propagate_exception_descr` is unattached (test harnesses that
    /// bypass `MetaInterp::finish_setup`); production
    /// (`pyjitpl.py:2283`) always sets it before `compile_loop` runs.
    fn emit_propagate_exception_if_zero(&mut self, reg: u8) {
        let propagate_descr = self.propagate_exception_descr_ptr();
        if propagate_descr == 0 {
            return;
        }
        let skip = self.mc.new_dynamic_label();
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        let exc_value_addr = crate::jit_exc_value_addr() as i64;
        let exc_type_addr = crate::jit_exc_type_addr() as i64;
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
        dynasm!(self.mc ; .arch x64 ; movzx Rd(dst_reg), Rb(dst_reg));
    }

    /// x86/assembler.py:1286 `flush_cc` parity.
    ///
    /// After emitting a CMP/TEST that leaves a boolean in the
    /// condition flags, call this. If the regalloc picked `frame_reg`
    /// (rbp) for `result_loc` the value is treated as living in the cc
    /// — `guard_success_cc` is published for the following guard to
    /// consume. Otherwise the boolean is materialised via a zeroed
    /// register + `SETcc` of its low byte. The MOV-zero + SETcc shape
    /// matches PyPy's emission and gives the regalloc a clean i64
    /// value for non-guard consumers (e.g. boolean stored into a
    /// frame slot).
    fn flush_cc(&mut self, cond: u8, result_loc: Option<&Loc>) {
        let frame_reg_value = crate::x86::regalloc::frame_reg().value;
        if let Some(Loc::Reg(r)) = result_loc {
            if r.value == frame_reg_value {
                // Sentinel: the next op accepts cc.
                debug_assert!(
                    self.guard_success_cc.is_none(),
                    "flush_cc: guard_success_cc already set",
                );
                self.guard_success_cc = Some(cond);
                return;
            }
            dynasm!(self.mc ; .arch x64 ; mov Rq(r.value), 0);
            self.emit_setcc(cond, r.value);
        }
    }

    /// x86/regalloc.py:429 `load_condition_into_cc` parity for the
    /// emit side. If the previous op already published a cond in
    /// `guard_success_cc`, the guard reads it directly. Otherwise the
    /// guard arg is a materialised boolean and we re-issue
    /// TEST + set CC_NE so `implement_guard` has a flag state to jump
    /// off of.
    fn load_condition_into_cc(&mut self, loc: &Loc) {
        if self.guard_success_cc.is_some() {
            return;
        }
        self.emit_test_loc(loc);
        self.guard_success_cc = Some(CC_NE);
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
        let fail_arg_types = self.infer_fail_arg_types(op, Some(op_index));
        // assembler.py:2207 _store_force_index parity:
        // If a CALL_ASSEMBLER already pre-allocated this guard's descr
        // (stored in pending_force_descr), reuse it — same Arc, same ptr
        // that was written to jf_force_descr.
        // Stamp the per-trace fail_index and trace_id onto the metainterp
        // ResumeGuardDescr (`op.descr`).  `compile.py:185` reserves these
        // slots for the `ResumeDescr` family; gate the writes accordingly
        // so non-resume meta descrs (Done* / Exit* / Propagate) take the
        // default panic path.  The metainterp's `build_guard_metadata`
        // (`compile.rs:232`) used to do this after backend codegen with
        // the same sequential counter; doing it here lets readers consume
        // the canonical metainterp identity before metadata builds.
        let descr_arc = op.getdescr();
        if let Some(d) = descr_arc.as_ref() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(pre) = self.pending_force_descr.take() {
            pre
        } else if let Some(d) = descr_arc {
            // Guard exit — `compile.py:185` ResumeGuardDescr family.
            // Use the metainterp `AbstractFailDescr` Arc from `op.descr`
            // directly; per-trace fail_index / trace_id were stamped above.
            let _unused = fail_arg_types; // already stored on op.descr's types slot
            d
        } else {
            // Test scaffold: tests synthesise guard ops without op.descr.
            // Mint a fresh metainterp ResumeGuardDescr to carry the
            // codegen-time identity (fail_index / trace_id / fail_arg_types).
            let fresh = majit_backend::make_resume_guard_descr_typed(fail_arg_types);
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        let descr_fd = descr.as_fail_descr().expect("guard descr is FailDescr");
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token: fail_index={} op_index={} opcode={:?} fail_args={:?} fail_arg_types={:?} faillocs={:?}",
                fail_index,
                op_index,
                op.opcode,
                op.getfailargs(),
                descr_fd.fail_arg_types(),
                faillocs
            );
        }

        // `llsupport/assembler.py:248-276 store_info_on_descr` parity:
        // encode each fail-arg location as a USHORT.  PyPy's encoding —
        //   None              → 0xFFFF
        //   GPR register      → position in `cpu.gen_regs`
        //   float register    → len(gen_regs) + position in `cpu.float_regs`
        //   stack             → (loc.value - base_ofs) // WORD
        //                         (here: `f.position + JITFRAME_FIXED_SIZE`)
        // PyPy regalloc never passes `Const` to `getfailargs()` — `loc()`
        // returns the immediate inline.  Pyre allocates a const-store
        // slot for `Loc::Immed` at codegen time and encodes the slot
        // into `rd_locs` so the deopt path treats it as a normal stack
        // position (`_decode_pos` in `llmodel.py:422-424`).
        let mut const_stores: Vec<(usize, i64)> = Vec::new();
        let gpr_regs = crate::x86::regalloc::ALL_CORE_REGS;
        let float_regs = crate::x86::regalloc::ALL_FLOAT_REGS;
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
                Some(Loc::Immed(i)) => {
                    // Allocate a const-store slot at codegen time;
                    // encode the slot into `rd_locs` (PyPy stack-position
                    // form) so deopt reads it like any other stack fail-arg.
                    let slot = self.frame_depth;
                    self.frame_depth += 1;
                    const_stores.push((slot, i.value));
                    slot as u16
                }
                Some(Loc::Ebp(_)) | Some(Loc::Addr(_)) => 0xFFFF,
            })
            .collect();
        // Stamp source_op_index directly on the meta descr (UnsafeCell slot
        // owned by ResumeGuardDescr / ResumeGuardCopiedDescr per
        // resume_guard_descr.rs:166); `layout_for_fail_descr` reads it back
        // via `fd.source_op_index()` so no side-table is needed.
        if descr_fd.is_resume_guard() || descr_fd.is_resume_guard_copied() {
            descr_fd.set_source_op_index(op_index);
        }
        // `llsupport/assembler.py:279 guardtok.faildescr.rd_locs = positions`
        // — write through the trait accessor so the metainterp
        // `AbstractFailDescr` (`history.py:132 _attrs_`) receives the
        // canonical copy.  Must follow the `meta_descr` stamp above.
        descr_fd.set_rd_locs(rd_locs);
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] guard-token-slots: fail_index={} rd_locs={:?}",
                fail_index,
                descr_fd.rd_locs()
            );
        }
        let gcmap = self.guard_gcmap_from_faillocs(descr_fd.fail_arg_types(), faillocs);

        // Reuse the cell pre-allocated by `_store_force_index_if_next_guard`
        // when this guard is paired with a CALL_ASSEMBLER's force-store —
        // jf_force_descr and jf_descr then resolve to the same cell, and
        // `fail_descrs[fail_index]` carries exactly one entry per guard.
        let cell = self
            .pending_force_cell
            .take()
            .unwrap_or_else(|| majit_ir::FailDescrCell::wrap(descr.clone()));
        self.pending_guard_tokens.push(GuardToken {
            jump_offset: self.mc.offset(),
            fail_label,
            fail_descr: cell.clone(),
            fail_args: op
                .getfailargs()
                .map(|fa| fa.iter().map(|a| a.to_opref()).collect())
                .unwrap_or_default(),
            opref_to_slot_snapshot: self.opref_to_slot.clone(),
            const_stores,
            gcmap,
        });
        if op.opcode == OpCode::GuardNotForced2 {
            self.finish_gcmap = Some(gcmap);
        }
        self.fail_descrs.push(cell);
    }

    /// Update `rd_locs` on all pending guard descriptors after the
    /// regalloc opref→slot map is finalised.  Unmapped (virtual/dead)
    /// OpRefs and constants get `0xFFFF` — the resume system handles
    /// them via `rd_numb` TAGVIRTUAL/TAGCONST encoding
    /// (`resume.py:450-488`).
    ///
    /// Parallels the pending_force path of PyPy
    /// `regalloc.py::store_force_descr → assembler.store_info_on_descr`
    /// (`llsupport/assembler.py:279 guardtok.faildescr.rd_locs = positions`).
    fn allocate_unmapped_fail_arg_slots(&mut self) {
        for gt in &self.pending_guard_tokens {
            let positions: Vec<u16> = gt
                .fail_args
                .iter()
                .map(|opref| {
                    if opref.is_none() || opref.is_constant() {
                        0xFFFFu16
                    } else {
                        self.opref_to_slot
                            .get(opref)
                            .copied()
                            .map(|slot| (slot + JITFRAME_FIXED_SIZE) as u16)
                            .unwrap_or(0xFFFFu16)
                    }
                })
                .collect();
            if let Some(meta_fd) = gt.fail_descr.as_fail_descr() {
                meta_fd.set_rd_locs(positions);
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
    ) -> (std::sync::Arc<majit_ir::FailDescrCell>, usize) {
        let stub_start = self.mc.offset();

        let fail_label = guard_token.fail_label;
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("recovery stub: binding {fail_label:?}"),
            );
        }
        dynasm!(self.mc ; .arch x64 ; =>fail_label);

        dynasm!(self.mc ; .arch x64 ; call =>save_regs_label);

        let descr_ptr = Arc::as_ptr(&guard_token.fail_descr) as *const () as i64;
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
    fn write_pending_failure_recoveries(
        &mut self,
    ) -> Vec<(std::sync::Arc<majit_ir::FailDescrCell>, usize)> {
        // Emit a shared _push_all_regs_to_frame routine once, then let each
        // generate_quick_failure() stub call it.  Iterate `ALL_CORE_REGS`
        // / `ALL_FLOAT_REGS` (Win64-aware: R13 dropped from GPRs, XMM5..14
        // dropped from FPRs) so save_regs_label, the gcmap built off
        // `core_reg_index`, and the post-call pop all agree on slot
        // assignments.
        let save_regs_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; =>save_regs_label);
        for &reg in crate::x86::regalloc::ALL_CORE_REGS.iter() {
            let save_slot = core_reg_position(reg).expect("managed x86_64 GPR");
            let ofs = Self::slot_offset(save_slot);
            dynasm!(self.mc ; .arch x64 ; mov [rbp + ofs], Rq(reg.value));
        }
        for &reg in crate::x86::regalloc::ALL_FLOAT_REGS.iter() {
            let save_slot = float_reg_position(reg).expect("managed x86_64 XMM");
            let ofs = Self::slot_offset(save_slot);
            dynasm!(self.mc ; .arch x64 ; movsd [rbp + ofs], Rx(reg.value));
        }
        dynasm!(self.mc ; .arch x64 ; ret);

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] write_pending_failure_recoveries: {} tokens",
                self.pending_guard_tokens.len()
            );
        }
        let mut stub_offsets = Vec::new();
        for guard_token in std::mem::take(&mut self.pending_guard_tokens) {
            stub_offsets.push(self.generate_quick_failure(guard_token, save_regs_label));
        }
        if majit_ir::debug::have_debug_prints() {
            majit_ir::debug::log_one(
                "jit-backend",
                &format!("write_pending done: {} stubs", stub_offsets.len()),
            );
        }
        stub_offsets
    }

    /// assembler.py:849 patch_pending_failure_recoveries — convert
    /// buffer-relative offsets to absolute addresses after finalize.
    fn patch_pending_failure_recoveries(
        rawstart: usize,
        stub_offsets: &[(std::sync::Arc<majit_ir::FailDescrCell>, usize)],
    ) {
        for (cell, stub_offset) in stub_offsets {
            let abs_addr = rawstart + stub_offset;
            if let Some(fd) = cell.as_fail_descr() {
                fd.set_adr_jump_offset(abs_addr);
            }
        }
    }

    /// `assembler.py:948 _patch_frame_depth` — overwrite the 32-bit
    /// `0xffffff` placeholder at `adr` with the finalised frame depth.
    ///
    /// PyPy uses `codebuf.MachineCodeBlockWrapper().writeimm32` +
    /// `copy_to_raw_memory(adr)`; here we write the four little-endian
    /// bytes directly inside a `with_writable` guard so the page-RW
    /// permissions match the platform's executable-memory policy.
    fn patch_frame_depth(adr: usize, allocated_depth: usize) {
        codebuf::with_writable(adr as *mut u8, 4, || unsafe {
            (adr as *mut i32).write_unaligned(allocated_depth as i32);
        });
    }

    /// `assembler.py:898 patch_stack_checks` — iterate
    /// `frame_depth_to_patch` and rewrite each placeholder immediate
    /// with the final `framedepth` (already absolute, including
    /// `JITFRAME_FIXED_SIZE`).
    ///
    /// Takes the patch list by slice rather than via `&self` so the
    /// caller (which has already consumed `self.mc` through
    /// `finalize()`) can still drive the patch step without keeping
    /// `Assembler386` partially moved.
    fn patch_stack_checks(framedepth: usize, rawstart: usize, offsets: &[usize]) {
        for &ofs in offsets {
            Self::patch_frame_depth(rawstart + ofs, framedepth);
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
    pub fn patch_jump_for_descr(descr: &dyn majit_ir::FailDescr, adr_new_target: usize) {
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
        if crate::majit_log_enabled() {
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
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; add rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_SUB: result = arg0 - arg1
    fn genop_int_sub(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; sub rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_MUL: result = arg0 * arg1
    fn genop_int_mul(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; imul rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_AND: result = arg0 & arg1
    fn genop_int_and(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; and rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_OR: result = arg0 | arg1
    fn genop_int_or(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; or rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_XOR: result = arg0 ^ arg1
    fn genop_int_xor(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; xor rax, rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_NEG: result = -arg0
    fn genop_int_neg(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; neg rax
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_INVERT: result = ~arg0
    fn genop_int_invert(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; not rax
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_LSHIFT: result = arg0 << arg1
    fn genop_int_lshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; shl rax, cl
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_RSHIFT: result = arg0 >> arg1 (arithmetic/signed)
    fn genop_int_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; sar rax, cl
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// UINT_RSHIFT: result = arg0 >> arg1 (logical/unsigned)
    fn genop_uint_rshift(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; shr rax, cl
        );
        self.store_rax_to_result(op.pos.get());
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

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; cmp rax, rcx
        );

        // Store the CC for a following guard to consume.
        self.guard_success_cc = Some(cc);

        // Also materialize the boolean result for non-guard consumers.
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(cc, op.pos.get());
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
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        self.guard_success_cc = Some(CC_NE);
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(CC_NE, op.pos.get());
        }
    }

    /// INT_IS_ZERO: result = (arg0 == 0)
    fn genop_int_is_zero(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; test rax, rax
        );
        self.guard_success_cc = Some(CC_E);
        if !op.pos.get().is_none() {
            self.emit_setcc_to_result(CC_E, op.pos.get());
        }
    }

    // ----------------------------------------------------------------
    // genop_* — guards
    // ----------------------------------------------------------------

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

    /// Infer fail_arg_types from `op.type_` (via `opref_type`) or
    /// `op.fail_arg_types`.
    fn infer_fail_arg_types(&self, op: &Op, op_index: Option<usize>) -> Vec<Type> {
        if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
            if let Some(descr_types) = op.with_fail_descr(|fd| fd.fail_arg_types().to_vec()) {
                if !descr_types.is_empty() {
                    return descr_types;
                }
            }
        }
        let descr_arc = op.getdescr();
        if let Some(fd) = descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            // Step A (43c64ee0bb) installs op.descr = ResumeGuardDescr
            // with post-numbering fail_arg_types via
            // store_final_boxes_in_guard (optimizeopt/mod.rs:3393-3404).
            // Prefer the descr for guards too; fall through to
            // op.fail_arg_types only for sharing-path guards
            // (optimizeopt/mod.rs:3068-3088) where op.descr=None.
            let dt = fd.fail_arg_types();
            let expected_len = op.getfailargs().map(|fa| fa.len()).unwrap_or(0);
            if dt.len() == expected_len && !dt.is_empty() {
                return dt.to_vec();
            }
        }
        if let Some(ts) = op.get_fail_arg_types() {
            let expected_len = if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.num_args()
            } else {
                op.getfailargs().map(|fa| fa.len()).unwrap_or(0)
            };
            if ts.len() == expected_len {
                ts.to_vec()
            } else if op.opcode == OpCode::Finish || op.opcode == OpCode::Jump {
                op.getarglist()
                    .iter()
                    .map(|opref| {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (Finish/Jump arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    })
                    .collect()
            } else if let Some(fa) = op.getfailargs() {
                fa.iter()
                    .map(|opref| {
                        if opref.is_none() {
                            // resume.py:411-417 parity: TAGCONST/TAGVIRTUAL
                            // slots are kept as OpRef::NONE in fail_args
                            // (PyPy filters them out; pyre keeps positional).
                            // `Type::Void` is the "hole" sentinel — value
                            // comes from the resume snapshot, not the
                            // deadframe, so downstream consumers
                            // (`gc_ref_slots`, `guard_gcmap_from_faillocs`,
                            // `typed_outputs` reconstruction) must skip
                            // these slots. Earlier code used `Type::Ref`
                            // which silently leaked a NULL `GcRef` into
                            // `gc_ref_slots` and the shadow stack.
                            Type::Void
                        } else {
                            self.opref_type_at(opref.to_opref(), op_index).unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                     op_index={:?} (fail_arg): RPython box.type is fixed at \
                                     construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            }
        } else if let Some(fa) = op.getfailargs() {
            fa.iter()
                .map(|opref| {
                    if opref.is_none() {
                        // resume.py:411-417 parity: see comment above —
                        // Type::Void is the "hole" sentinel.
                        Type::Void
                    } else {
                        self.opref_type_at(opref.to_opref(), op_index)
                            .unwrap_or_else(|| {
                                panic!(
                                    "infer_fail_arg_types: opref_type_at({:?}) returned None at \
                                 op_index={:?} (fail_arg): RPython box.type is fixed at \
                                 construction (resoperation.py:719/727/739)",
                                    opref, op_index
                                )
                            })
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
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
        let fail_arg_types = self.infer_fail_arg_types(next_op, Some(next_idx));
        // Pre-allocated GuardNotForced descr — ResumeGuardDescr family.
        // Stamp the metainterp `AbstractFailDescr` Arc from `next_op.descr`
        // here so `append_guard_token_with_faillocs` does not need a second
        // pass through `unsafe { Arc::as_ptr as *mut }`.
        let descr_arc = next_op.getdescr();
        if let Some(d) = descr_arc.as_ref() {
            if d.is_resume_guard() || d.is_resume_guard_copied() {
                if let Some(fd) = d.as_fail_descr() {
                    fd.set_fail_index_per_trace(fail_index);
                    fd.set_trace_id(self.trace_id);
                }
            }
        }
        let descr: majit_ir::DescrRef = if let Some(d) = descr_arc {
            let _unused = fail_arg_types;
            d
        } else {
            let fresh = majit_backend::make_resume_guard_descr_typed(fail_arg_types);
            if let Some(fd) = fresh.as_fail_descr() {
                fd.set_fail_index_per_trace(fail_index);
                fd.set_trace_id(self.trace_id);
            }
            fresh
        };
        // `force_token_to_dead_frame` (cranelift/compiler.rs:2660)
        // recovers `jf_force_descr` via `recover_fail_descr_cell`, which
        // requires a `FailDescrCell` thin pointer.  Bake the cell pointer
        // here (not the bare `Arc<dyn Descr>` fat-pointer data half) and
        // hand the cell off to `append_guard_token_with_faillocs` so the
        // inline guard-exit path bakes the same identity into jf_descr.
        let cell = majit_ir::FailDescrCell::wrap(descr.clone());
        let descr_ptr = Arc::as_ptr(&cell) as *const () as i64;
        self.pending_force_descr = Some(descr);
        self.pending_force_cell = Some(cell);

        // x86/assembler.py:2210-2222: store descr to jf_force_descr,
        // zero jf_descr.
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD descr_ptr
            ; mov QWORD [rbp + JF_FORCE_DESCR_OFS], rax
            ; mov QWORD [rbp + JF_DESCR_OFS], 0
        );
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
        let n_label = op.num_args();
        // Pass 1: push source values
        for i in 0..n_label {
            let arg_ref = op.arg(i).to_opref();
            if arg_ref.is_none() {
                let dst = Self::slot_offset(i);
                dynasm!(self.mc ; .arch x64 ; push QWORD [rbp + dst]);
            } else if arg_ref.is_constant() {
                let val = arg_ref
                    .inline_const_bits()
                    .unwrap_or_else(|| self.constants.get(&arg_ref.raw()).copied().unwrap_or(0));
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD val as i64 ; push rax);
            } else if let Some(&old_slot) = self.opref_to_slot.get(&arg_ref) {
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
        let descr_arc = op.getdescr();
        if let Some(descr) = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr()) {
            descr.set_ll_loop_code(self.mc.offset().0);
            if let Some(id) = descr_arc.as_ref().map(majit_ir::descr_identity) {
                self.target_tokens_currently_compiling.insert(id, label);
            }
            if let Some(descr_ref) = descr_arc.as_ref() {
                self.compiled_target_tokens.push(descr_ref.clone());
            }
        }

        // Remap: Label's arg[i] → canonical slot i
        for (i, arg_ref) in op.getarglist().iter().enumerate() {
            if !arg_ref.is_none() {
                self.opref_to_slot.insert(arg_ref.to_opref(), i);
            }
        }
        self.next_slot = self.next_slot.max(op.num_args());
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
        let n = op.num_args();
        let mut moves: Vec<(i32, i32, bool, i64)> = Vec::with_capacity(n);
        for (i, arg_ref) in op.getarglist().iter().enumerate() {
            let dst = Self::slot_offset(i);
            match self.resolve_opref(arg_ref.to_opref()) {
                ResolvedArg::Slot(src) => moves.push((src, dst, false, 0)),
                ResolvedArg::Const(val) => moves.push((0, dst, true, val)),
            }
        }

        // jump.py:1-64 remap_frame_layout: topological order with
        // cycle breaking via push/pop.
        // srccount[dst] = number of times dst appears as a src
        let mut srccount: VecAssoc<i32, i32> = VecAssoc::new();
        for m in &moves {
            if !srccount.contains_key(&m.1) {
                srccount.insert(m.1, 0); // ensure dst exists
            }
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

        let descr_arc = op.getdescr();
        let jump_descr = descr_arc.as_ref().and_then(|d| d.as_loop_target_descr());
        if let Some(label) = descr_arc
            .as_ref()
            .map(majit_ir::descr_identity)
            .and_then(|k| self.target_tokens_currently_compiling.get(&k).copied())
        {
            // Same-buffer jump (loop body)
            dynasm!(self.mc ; .arch x64 ; jmp =>label);
        } else if let Some(target) = jump_descr.map(|descr| descr.ll_loop_code()) {
            // assembler.py closing_jump parity: bridge jumps back to
            // the original loop's LABEL via absolute address. PyPy
            // stages the 64-bit target through `X86_64_SCRATCH_REG`
            // (r11), never RAX — using RAX here would clobber the
            // loop-carried Ref the regalloc bound to it.
            // assembler.py:1003-1008 `_assemble`: record the target loop's
            // frame depth so this trace's frame grows to fit it.
            if let Some(descr) = jump_descr {
                self.jump_target_frame_depth =
                    self.jump_target_frame_depth.max(descr.target_frame_depth());
            }
            let addr = target as i64;
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD addr
                ; jmp Rq(scratch)
            );
        }
    }

    /// FINISH: store result (if any), store descr ptr, return jf_ptr.
    fn genop_finish(&mut self, op: &Op, fail_index: u32) {
        // compiler.rs:9667-9681 parity: trust explicit FINISH types only when
        // they match the actual result arity; otherwise infer from the op args.
        let finish_refs: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
        let fail_arg_types = if let Some(explicit) = op.get_fail_arg_types() {
            if explicit.len() == finish_refs.len() {
                explicit.to_vec()
            } else {
                finish_refs
                    .iter()
                    .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                    .collect()
            }
        } else {
            finish_refs
                .iter()
                .map(|opref| self.opref_type_at(*opref, None).unwrap_or(Type::Int))
                .collect()
        };
        // compile.py:618-669 parity: use type-specific global singleton.
        // FINISH op exit (DoneWithThisFrame*) — `compile.py:185` skips these.
        // Finish ops write the type-appropriate singleton pointer to jf_descr
        // so CALL_ASSEMBLER's fast path CMP matches the correct variant.
        let result_type = if fail_arg_types.is_empty() {
            Type::Void
        } else {
            fail_arg_types[0]
        };
        let global_descr_ptr = self.done_with_this_frame_descr_ptr_for_type(result_type);
        // Singleton-direct push (see OpCode::Finish above for rationale).
        let descr: majit_ir::DescrRef = self
            .done_with_this_frame_descr_arc_for_type(result_type)
            .expect(
                "genop_finish requires cpu-attached singleton — \
                 call `attach_default_test_descrs` or use `MetaInterp::new`",
            );

        // If there's a result argument, store it to jf_frame[0].
        // assembler.py:2291-2303 parity: float results use xmm0/MOVSD.
        if op.num_args() > 0 {
            let arg0 = op.arg(0).to_opref();
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

        // Singleton: jf_descr bakes the cpu-attached `global_descr_ptr`,
        // not the cell pointer (see OpCode::Finish comment above).
        self.fail_descrs
            .push(majit_ir::FailDescrCell::wrap(descr.clone()));
    }

    // ----------------------------------------------------------------
    // genop_* — type conversions
    // ----------------------------------------------------------------

    /// SAME_AS: result = arg0 (copy value)
    /// SAME_AS: result = arg0 (identity).
    /// regalloc.py parity: no code emitted — just alias the slot.
    fn genop_same_as(&mut self, op: &Op) {
        let arg = op.arg(0).to_opref();
        if let Some(&slot) = self.opref_to_slot.get(&arg) {
            self.opref_to_slot.insert(op.pos.get(), slot);
        } else {
            self.load_arg_to_rax(arg);
            self.store_rax_to_result(op.pos.get());
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
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; addsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_SUB: result = arg0 - arg1
    fn genop_float_sub(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; subsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_MUL: result = arg0 * arg1
    fn genop_float_mul(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; mulsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_TRUEDIV: result = arg0 / arg1
    fn genop_float_truediv(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        self.load_float_arg_to_d1(op.arg(1).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; divsd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_NEG: result = -arg0
    /// x64: XOR with sign-bit mask (0x8000000000000000).
    /// aarch64: FNEG d0, d0.
    fn genop_float_neg(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        // Load the sign-bit mask (0x8000_0000_0000_0000) into XMM1
        // via integer register, then XOR.
        let sign_mask: i64 = i64::MIN; // 0x8000000000000000
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, QWORD sign_mask
            ; movq xmm1, rax
            ; xorpd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// CAST_INT_TO_FLOAT: result = (f64)arg0
    fn genop_cast_int_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; cvtsi2sd xmm0, rax
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// CAST_FLOAT_TO_INT: result = (i64)arg0 (truncation)
    fn genop_cast_float_to_int(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc
            ; .arch x64
            ; cvttsd2si rax, xmm0
        );
        self.store_rax_to_result(op.pos.get());
    }

    // ----------------------------------------------------------------
    // genop_* — memory operations
    // x86/assembler.py:1747 genop_getfield_gc etc.
    // ----------------------------------------------------------------

    /// Extract the byte offset from an op's FieldDescr.
    /// Returns 0 if no field descriptor is present.
    fn field_offset_from_descr(op: &Op) -> i32 {
        op.with_field_descr(|fd| fd.offset() as i32).unwrap_or(0)
    }

    /// Extract the field size from an op's FieldDescr.
    /// Returns 8 (WORD) if no field descriptor is present.
    fn field_size_from_descr(op: &Op) -> usize {
        op.with_field_descr(|fd| fd.field_size()).unwrap_or(8)
    }

    /// GETFIELD_GC_*: result = [arg0 + offset]
    /// The offset comes from the op's FieldDescr.
    fn genop_getfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load the object pointer from arg0.
        self.load_arg_to_rax(op.arg(0).to_opref());

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

        self.store_rax_to_result(op.pos.get());
    }

    /// x86/assembler.py:1746 genop_discard_setfield — sized store via regalloc.
    /// Stage non-register values through X86_64_SCRATCH_REG (r11), mirroring
    /// the aarch64 path that uses x16.
    fn emit_op_setfield_regalloc(
        &mut self,
        base: &crate::regloc::RegLoc,
        val_loc: &Loc,
        ofs: i32,
        field_size: usize,
    ) {
        if let Loc::Reg(v) = val_loc
            && v.is_xmm
        {
            dynasm!(self.mc ; .arch x64 ; movsd [Rq(base.value) + ofs], Rx(v.value));
            return;
        }
        let val_reg = match val_loc {
            Loc::Reg(v) => v.value,
            _ => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG;
                self.regalloc_mov(val_loc, &Loc::Reg(scratch));
                scratch.value
            }
        };
        match field_size {
            1 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rb(val_reg)),
            2 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rw(val_reg)),
            4 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rd(val_reg)),
            _ => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(val_reg)),
        }
    }

    /// x86/assembler.py:1691 _genop_gc_load — sized load via regalloc.
    /// `size`: byte size (1/2/4/8). Negative = signed load.
    fn emit_op_gcload_regalloc(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs_loc: &Loc,
        dst: &crate::regloc::RegLoc,
        size: i64,
    ) {
        let abs_size = size.unsigned_abs() as usize;
        let signed = size < 0;
        match ofs_loc {
            Loc::Immed(i) => {
                let o = i.value as i32;
                self.emit_gcload_sized(base, o, None, dst, abs_size, signed);
            }
            Loc::Reg(ofs_r) => {
                self.emit_gcload_sized(base, 0, Some(ofs_r), dst, abs_size, signed);
            }
            _ => {}
        }
    }

    /// Sized load: `[base + ofs]` or `[base + ofs_reg]` — assembler.py:1645 load_from_mem.
    fn emit_gcload_sized(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs: i32,
        ofs_reg: Option<&crate::regloc::RegLoc>,
        dst: &crate::regloc::RegLoc,
        size: usize,
        signed: bool,
    ) {
        if dst.is_xmm {
            if let Some(r) = ofs_reg {
                dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + Rq(r.value)]);
            } else {
                dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + ofs]);
            }
            return;
        }
        if let Some(r) = ofs_reg {
            match (size, signed) {
                (1, false) => {
                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), BYTE [Rq(base.value) + Rq(r.value)])
                }
                (1, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), BYTE [Rq(base.value) + Rq(r.value)])
                }
                (2, false) => {
                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), WORD [Rq(base.value) + Rq(r.value)])
                }
                (2, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), WORD [Rq(base.value) + Rq(r.value)])
                }
                (4, false) => {
                    dynasm!(self.mc ; .arch x64 ; mov Rd(dst.value), [Rq(base.value) + Rq(r.value)])
                }
                (4, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsxd Rq(dst.value), DWORD [Rq(base.value) + Rq(r.value)])
                }
                _ => {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + Rq(r.value)])
                }
            }
        } else {
            match (size, signed) {
                (1, false) => {
                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), BYTE [Rq(base.value) + ofs])
                }
                (1, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), BYTE [Rq(base.value) + ofs])
                }
                (2, false) => {
                    dynasm!(self.mc ; .arch x64 ; movzx Rq(dst.value), WORD [Rq(base.value) + ofs])
                }
                (2, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsx Rq(dst.value), WORD [Rq(base.value) + ofs])
                }
                (4, false) => {
                    dynasm!(self.mc ; .arch x64 ; mov Rd(dst.value), [Rq(base.value) + ofs])
                }
                (4, true) => {
                    dynasm!(self.mc ; .arch x64 ; movsxd Rq(dst.value), DWORD [Rq(base.value) + ofs])
                }
                _ => dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + ofs]),
            }
        }
    }

    /// x86/assembler.py:1746 genop_discard_gc_store — sized store via regalloc.
    fn emit_op_gcstore_regalloc(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs_loc: &Loc,
        val: &crate::regloc::RegLoc,
        size: usize,
    ) {
        match ofs_loc {
            Loc::Immed(i) => {
                let o = i.value as i32;
                self.emit_gcstore_sized(base, o, None, val, size);
            }
            Loc::Reg(ofs_r) => {
                self.emit_gcstore_sized(base, 0, Some(ofs_r), val, size);
            }
            other => panic!("GcStore: ofs_loc must be Loc::Reg or Loc::Immed, got {other:?}",),
        }
    }

    /// Immediate-value variant of `emit_op_gcstore_regalloc`.
    /// `llsupport/regalloc.py:625 return_constant` may return a bare
    /// `Loc::Immed` for a Const value, so GcStore reaches the emitter
    /// with the literal already in hand. x86 can write the immediate
    /// directly into memory when it fits in `imm32` (sign-extended for
    /// QWORD stores), avoiding the need for a staging register.
    fn emit_op_gcstore_imm_regalloc(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs_loc: &Loc,
        val: i64,
        size: usize,
    ) {
        let val_fits_imm32 = (val as i32) as i64 == val;
        let val_fits_at_size = match size {
            1 => (val & !0xFF) == 0 || (val | 0xFF) == -1,
            2 => (val & !0xFFFF) == 0 || (val | 0xFFFF) == -1,
            4 => (val & !0xFFFFFFFFi64) == 0 || (val | 0xFFFFFFFFi64) == -1,
            _ => val_fits_imm32,
        };
        if val_fits_at_size {
            match ofs_loc {
                Loc::Immed(i) => {
                    let o = i.value as i32;
                    self.emit_gcstore_imm_sized(base, o, None, val, size);
                }
                Loc::Reg(ofs_r) => {
                    self.emit_gcstore_imm_sized(base, 0, Some(ofs_r), val, size);
                }
                other => {
                    panic!("GcStore imm: ofs_loc must be Loc::Reg or Loc::Immed, got {other:?}",)
                }
            }
            return;
        }
        // 64-bit immediate that doesn't fit sign-extended imm32. The
        // regalloc materialises an out-of-range offset into
        // LARGE_IMM_SCRATCH (R11), so staging val through R11 too would
        // clobber the offset register. Split the QWORD write into two
        // DWORD writes instead — `mov DWORD [mem], imm32` accepts a
        // bare immediate and leaves the offset register intact.
        debug_assert_eq!(
            size, 8,
            "split-store path only reachable for QWORD stores; smaller sizes go through val_fits_at_size",
        );
        let lo = val as i32;
        let hi = (val >> 32) as i32;
        match ofs_loc {
            Loc::Immed(i) => {
                let o = i.value as i32;
                dynasm!(self.mc ; .arch x64
                    ; mov DWORD [Rq(base.value) + o], lo
                    ; mov DWORD [Rq(base.value) + o + 4], hi);
            }
            Loc::Reg(ofs_r) => {
                dynasm!(self.mc ; .arch x64
                    ; mov DWORD [Rq(base.value) + Rq(ofs_r.value)], lo
                    ; mov DWORD [Rq(base.value) + Rq(ofs_r.value) + 4], hi);
            }
            other => panic!("GcStore imm: ofs_loc must be Loc::Reg or Loc::Immed, got {other:?}",),
        }
    }

    /// Sized direct memory-immediate store: `mov SIZE [base + ofs(_reg)], imm`.
    fn emit_gcstore_imm_sized(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs: i32,
        ofs_reg: Option<&crate::regloc::RegLoc>,
        val: i64,
        size: usize,
    ) {
        if let Some(r) = ofs_reg {
            match size {
                1 => dynasm!(self.mc ; .arch x64
                    ; mov BYTE [Rq(base.value) + Rq(r.value)], val as i8),
                2 => dynasm!(self.mc ; .arch x64
                    ; mov WORD [Rq(base.value) + Rq(r.value)], val as i16),
                4 => dynasm!(self.mc ; .arch x64
                    ; mov DWORD [Rq(base.value) + Rq(r.value)], val as i32),
                8 => dynasm!(self.mc ; .arch x64
                    ; mov QWORD [Rq(base.value) + Rq(r.value)], val as i32),
                other => panic!("GcStore imm: unsupported store size {other}"),
            }
        } else {
            match size {
                1 => dynasm!(self.mc ; .arch x64
                    ; mov BYTE [Rq(base.value) + ofs], val as i8),
                2 => dynasm!(self.mc ; .arch x64
                    ; mov WORD [Rq(base.value) + ofs], val as i16),
                4 => dynasm!(self.mc ; .arch x64
                    ; mov DWORD [Rq(base.value) + ofs], val as i32),
                8 => dynasm!(self.mc ; .arch x64
                    ; mov QWORD [Rq(base.value) + ofs], val as i32),
                other => panic!("GcStore imm: unsupported store size {other}"),
            }
        }
    }

    /// Sized store: `[base + ofs]` or `[base + ofs_reg]` — assembler.py:1671 save_into_mem.
    fn emit_gcstore_sized(
        &mut self,
        base: &crate::regloc::RegLoc,
        ofs: i32,
        ofs_reg: Option<&crate::regloc::RegLoc>,
        val: &crate::regloc::RegLoc,
        size: usize,
    ) {
        if val.is_xmm {
            if let Some(r) = ofs_reg {
                dynasm!(self.mc ; .arch x64 ; movsd [Rq(base.value) + Rq(r.value)], Rx(val.value));
            } else {
                dynasm!(self.mc ; .arch x64 ; movsd [Rq(base.value) + ofs], Rx(val.value));
            }
            return;
        }
        if let Some(r) = ofs_reg {
            match size {
                1 => {
                    dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(r.value)], Rb(val.value))
                }
                2 => {
                    dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(r.value)], Rw(val.value))
                }
                4 => {
                    dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(r.value)], Rd(val.value))
                }
                8 => {
                    dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(r.value)], Rq(val.value))
                }
                other => panic!("GcStore: unsupported store size {other}"),
            }
        } else {
            match size {
                1 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rb(val.value)),
                2 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rw(val.value)),
                4 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rd(val.value)),
                8 => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(val.value)),
                other => panic!("GcStore: unsupported store size {other}"),
            }
        }
    }

    /// SETFIELD_GC: [arg0 + offset] = arg1
    fn genop_discard_setfield(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        // Load object pointer into rax/x0 and value into rcx/x1.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

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
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer and index.
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

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

        self.store_rax_to_result(op.pos.get());
    }

    /// SETARRAYITEM_GC: array[index] = value
    /// arg0 = array pointer, arg1 = index, arg2 = value.
    fn genop_discard_setarrayitem(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((8, 8));

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0).to_opref());
        // Load index.
        self.load_arg_to_rcx(op.arg(1).to_opref());

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
        self.load_arg_to_rcx(op.arg(2).to_opref());
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
        let descr_arc = op.getdescr();
        let len_offset = descr_arc
            .as_ref()
            .and_then(|d| d.as_array_descr())
            .and_then(|ad| ad.len_descr())
            .map(|ld| ld.offset() as i32)
            .unwrap_or(0); // Default: length at offset 0 in array header

        // Load array pointer.
        self.load_arg_to_rax(op.arg(0).to_opref());

        // Load length from [array + len_offset].
        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + len_offset]
        );

        self.store_rax_to_result(op.pos.get());
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
        let call_arg_count = arg_count.saturating_sub(func_arg + 1);
        let descr_arc = op.getdescr();
        let arg_types = descr_arc
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .map(|descr| descr.arg_types().to_vec())
            .filter(|types| types.len() == call_arg_count)
            .unwrap_or_else(|| {
                (func_arg + 1..arg_count)
                    .map(|i| self.opref_type(op.arg(i).to_opref()).unwrap_or(Type::Int))
                    .collect()
            });
        let (placements, stack_slots) = Self::build_abi_arg_placements(&arg_types);

        dynasm!(self.mc ; .arch x64 ; push rbp);
        let call_area_adjust = self.emit_reserve_abi_call_area(1, stack_slots);

        for i in (func_arg + 1)..arg_count {
            let arg = op.arg(i).to_opref();
            let abi_idx = i - func_arg - 1;
            let placement = placements[abi_idx];
            let arg_type = arg_types[abi_idx];
            match self.resolve_opref(arg) {
                ResolvedArg::Slot(offset) => {
                    self.emit_abi_arg_from_mem(placement, offset, arg_type)
                }
                ResolvedArg::Const(val) => {
                    self.emit_abi_arg_from_imm(placement, val as i64, arg_type)
                }
            }
        }

        match self.resolve_opref(op.arg(func_arg).to_opref()) {
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

        self.emit_release_abi_call_area(call_area_adjust);
        dynasm!(self.mc ; .arch x64 ; pop rbp);
    }

    /// aarch64/opassembler.py:1036 _emit_call.
    /// arglocs = [resloc, size, sign, func, args...] for normal CALLs and
    /// [resloc, size, sign, saveerr, func, args...] for CALL_RELEASE_GIL.
    ///
    /// Register-bound arg moves go through `remap_frame_layout_mixed`
    /// (a parallel-move algorithm) mirroring x86/callbuilder.py:584
    /// `prepare_arguments` → `remap_frame_layout`.  Emitting them naively
    /// in source order broke Win64 where two args could map to the same
    /// dst-then-src register (e.g. arg0 → rcx clobbering Reg(rcx) before
    /// arg1 reads it as Gpr(rdx)).  Linux SysV escaped the same code
    /// path because its rdi/rsi placement happened not to collide with
    /// regalloc-chosen rcx/rdx for these traces.
    fn emit_call_from_arglocs(&mut self, op: &Op, arglocs: &[Loc], func_index: usize) {
        let arg_count = arglocs.len();
        let call_arg_count = arg_count.saturating_sub(func_index + 1);
        let descr_arc = op.getdescr();
        let arg_types = descr_arc
            .as_ref()
            .and_then(|descr| descr.as_call_descr())
            .map(|descr| descr.arg_types().to_vec())
            .filter(|types| types.len() == call_arg_count)
            .unwrap_or_else(|| vec![Type::Int; call_arg_count]);
        let (placements, stack_slots) = Self::build_abi_arg_placements(&arg_types);

        dynasm!(self.mc ; .arch x64 ; push rbp);
        let call_area_adjust = self.emit_reserve_abi_call_area(1, stack_slots);

        // Pass 1: emit stack-dst args first.  Their sources may be
        // registers the parallel move below will overwrite, but stack
        // writes never disturb registers, so doing them up front keeps
        // every register source live for Pass 2.
        for i in (func_index + 1)..arg_count {
            let abi_idx = i - func_index - 1;
            let placement = placements[abi_idx];
            if !matches!(placement, AbiArgPlacement::Stack(_)) {
                continue;
            }
            let arg_type = arg_types[abi_idx];
            let arg = &arglocs[i];
            match arg {
                Loc::Frame(f) => self.emit_abi_arg_from_mem(placement, f.ebp_loc.value, arg_type),
                Loc::Reg(r) => self.emit_abi_arg_from_reg(placement, *r, arg_type),
                Loc::Immed(i) => self.emit_abi_arg_from_imm(placement, i.value, arg_type),
                _ => {}
            }
        }

        // Pass 2: parallel-move register-bound args (GPR and XMM groups
        // separately).  If the call target itself is a register, append
        // it to the int group with rax as the dst so the move algorithm
        // sees the dependency — otherwise loading the target after the
        // move could read a register whose old value has just been
        // overwritten by Gpr(reg)-placed args.
        let mut int_src: Vec<Loc> = Vec::new();
        let mut int_dst: Vec<Loc> = Vec::new();
        let mut xmm_src: Vec<Loc> = Vec::new();
        let mut xmm_dst: Vec<Loc> = Vec::new();
        for i in (func_index + 1)..arg_count {
            let abi_idx = i - func_index - 1;
            let placement = placements[abi_idx];
            let arg = arglocs[i];
            match placement {
                AbiArgPlacement::Gpr(dst_reg) => {
                    int_src.push(arg);
                    int_dst.push(Loc::Reg(crate::regloc::RegLoc::new(dst_reg, false)));
                }
                AbiArgPlacement::Xmm(dst_reg) => {
                    xmm_src.push(arg);
                    xmm_dst.push(Loc::Reg(crate::regloc::RegLoc::new(dst_reg, true)));
                }
                AbiArgPlacement::Stack(_) => {}
            }
        }
        let func_in_rax_after_move = matches!(arglocs.get(func_index), Some(Loc::Reg(_)));
        if let Some(Loc::Reg(r)) = arglocs.get(func_index) {
            int_src.push(Loc::Reg(*r));
            int_dst.push(Loc::Reg(crate::regloc::RegLoc::new(0, false))); // rax
        }
        let tmpreg1 = Loc::Reg(crate::regloc::X86_64_SCRATCH_REG);
        let tmpreg2 = Loc::Reg(crate::regloc::XMM15);
        self.remap_frame_layout_mixed(&int_src, &int_dst, tmpreg1, &xmm_src, &xmm_dst, tmpreg2);

        // Call.  For Immed/Frame targets, load rax now (parallel move
        // never touches rax or rbp, so this is safe).  For Reg targets,
        // the parallel move above already left the function pointer in
        // rax.
        if !func_in_rax_after_move {
            match arglocs.get(func_index) {
                Some(Loc::Frame(f)) => {
                    let offset = f.ebp_loc.value;
                    dynasm!(self.mc ; .arch x64 ; mov rax, [rbp + offset]);
                }
                Some(Loc::Immed(i)) => {
                    let val = i.value;
                    dynasm!(self.mc ; .arch x64 ; mov rax, QWORD val);
                }
                _ => {}
            }
        }
        dynasm!(self.mc ; .arch x64 ; call rax);

        self.emit_release_abi_call_area(call_area_adjust);
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
        self.emit_call_from_arglocs(op, arglocs, func_index);
        if op.opcode.result_type() == Type::Int {
            self.ensure_call_result_bit_extension(arglocs);
        }
    }

    /// assembler.py:2169-2174 _genop_real_call.
    /// genop_call_i = genop_call_r = genop_call_f = genop_call_n
    fn genop_call(&mut self, op: &Op) {
        self._genop_call(op);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    fn genop_call_with_arglocs(&mut self, op: &Op, arglocs: &[Loc]) {
        // llsupport/callbuilder.py:36 emit order — prepare_arguments() then
        // push_gcmap(); emit_raw_call(); ...; pop_gcmap() — collecting calls
        // must publish the regalloc gcmap before the raw call so live `Ref`
        // roots survive the slow path, and clear it after reloading a
        // possibly-moved frame pointer (assembler.py:296
        // `_reload_frame_if_necessary`).
        let can_collect = op
            .with_call_descr(|descr| descr.get_extra_info().check_can_collect())
            .unwrap_or(false);
        let pushed_gcmap = if can_collect {
            self.push_pending_call_gcmap()
        } else {
            false
        };
        self._genop_call_with_arglocs(op, arglocs);
        if can_collect {
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
        }
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
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
    /// llsupport/assembler.py:295 `call_assembler` + x86/assembler.py:2267
    /// `_call_assembler_emit_call` parity. Line-by-line port:
    /// 1. simple_call(target, [jf, threadlocal_loc])
    /// 2. CMP [eax + jf_descr_ofs], done_descr_imm
    /// 3. je fast_path
    /// 4. simple_call(asm_helper, [eax, vloc], result_loc)   ← slow path
    /// 5. jmp merge
    /// 6. fast_path: mov rax, [rax + first_item_ofs]
    /// 7. merge:
    ///
    /// Caller's rbp is preserved by the callee's _call_header/_call_footer
    /// (which push/pop it). After the call we still need
    /// `reload_frame_if_necessary` because a minor GC during the callee
    /// may have moved the caller jitframe; the popped rbp is the
    /// pre-GC address while the shadow stack carries the updated one.
    fn genop_call_assembler(&mut self, op: &Op, arglocs: &[Loc]) {
        let __descr_arc_call_descr = op.getdescr();
        let call_descr = __descr_arc_call_descr
            .as_ref()
            .and_then(|d| d.as_call_descr());
        let expansion = call_descr.and_then(|d| d.vable_expansion());
        if expansion.is_none() {
            let frame_loc = arglocs
                .first()
                .copied()
                .expect("call_assembler missing rewritten jitframe arg");
            let vable_loc = arglocs.get(1).copied();

            let target_addr: Option<usize> = __descr_arc_call_descr
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
                // Unresolved target: emit force-fn dispatch through
                // r12-saved rbp (kept as-is — this path is rare and not
                // on the recursive hot path).
                self.emit_load_to_rax(frame_loc);
                dynasm!(self.mc ; .arch x64 ; mov rdx, rax);
                let force_addr = crate::call_assembler_force_fn_addr() as i64;
                if force_addr != 0 {
                    if let Some(vloc) = vable_loc {
                        self.emit_load_to_rax(vloc);
                        self.emit_abi_int_arg_from_reg(0, 0);
                    } else {
                        dynasm!(self.mc ; .arch x64
                            ; mov rax, [rdx + FIRST_ITEM_OFFSET as i32]
                        );
                        self.emit_abi_int_arg_from_reg(0, 0);
                    }
                    let pushed_gcmap = self.push_pending_call_gcmap();
                    dynasm!(self.mc ; .arch x64 ; mov rax, QWORD force_addr);
                    self.emit_abi_call_rax_aligned();
                    self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
                } else {
                    dynasm!(self.mc ; .arch x64 ; xor eax, eax);
                }
                if !op.pos.get().is_none() {
                    self.store_rax_to_result(op.pos.get());
                }
                return;
            }

            // ── x86/assembler.py:2267 _call_assembler_emit_call ──
            // simple_call(target, [argloc]).  Branch directly to the
            // resolved callee entry — skip the Rust trampoline, which
            // would otherwise add an extra indirect call and (when
            // MAJIT_LOG was probed) a `std::env::var_os` per recursion.
            let pushed_gcmap = self.push_pending_call_gcmap();
            self.emit_load_to_rax(frame_loc); // rax = callee jf_ptr
            self.emit_abi_int_arg_from_reg(0, 0); // arg0 = jf (Windows: rcx = rax)
            if let Some(addr) = target_addr {
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD addr as i64);
                self.emit_abi_call_rax_aligned();
            } else {
                let addr_ptr = self.self_entry_addr_ptr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD addr_ptr
                    ; mov rax, [rax]
                );
                self.emit_abi_call_rax_aligned();
            }
            // Callee's _call_footer popped caller's rbp (= pre-GC
            // address). Reload from shadow stack so subsequent
            // frame-relative ops hit the moved jitframe.
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);

            // ── x86/assembler.py:2274 _call_assembler_check_descr ──
            // CMP [eax + jf_descr_ofs], imm(done_descr).
            // x86 has no 64-bit-immediate compare-with-memory, so
            // stage the pointer through R11 (LARGE_IMM_SCRATCH) — one
            // mov + one cmp instead of the previous load-into-reg +
            // load-imm + reg-reg compare. PyPy's `mc.CMP(mem, imm)`
            // does the same staging internally.
            let fast_path = self.mc.new_dynamic_label();
            let merge = self.mc.new_dynamic_label();
            let scratch = crate::regloc::X86_64_SCRATCH_REG.value;
            dynasm!(self.mc ; .arch x64
                ; mov Rq(scratch), QWORD done_descr_ptr
                ; cmp [rax + JF_DESCR_OFS], Rq(scratch)
                ; je =>fast_path
            );

            // ── Path A: x86/assembler.py:2271 _call_assembler_emit_helper_call ──
            // simple_call(asm_helper, [tmploc=rax, vloc], result_loc).
            // pyre's helper signature is (cpu_handle, callee_jf,
            // green_key) — see compile.py:665.
            let cpu_ptr = self.cpu_handle_ptr();
            self.emit_abi_int_arg_from_imm(0, cpu_ptr);
            self.emit_abi_int_arg_from_reg(1, 0); // arg1 = rax (callee jf)
            self.emit_abi_int_arg_from_imm(2, green_key);
            dynasm!(self.mc ; .arch x64 ; mov rax, QWORD helper_addr);
            let pushed_gcmap = self.push_pending_call_gcmap();
            self.emit_abi_call_rax_aligned();
            self.pop_pending_call_gcmap_after_collect(pushed_gcmap);
            dynasm!(self.mc ; .arch x64
                ; jmp =>merge
                ; =>fast_path
            );

            // ── Path B: x86/assembler.py:2291 _call_assembler_load_result ──
            // MOV result, [eax + first_item_ofs].
            if result_type == Type::Float {
                dynasm!(self.mc ; .arch x64
                    ; movsd xmm0, [rax + FIRST_ITEM_OFFSET as i32]
                    ; movq rax, xmm0
                    ; =>merge
                );
            } else {
                dynasm!(self.mc ; .arch x64
                    ; mov rax, [rax + FIRST_ITEM_OFFSET as i32]
                    ; =>merge
                );
            }
            if !op.pos.get().is_none() {
                self.store_rax_to_result(op.pos.get());
            }
            let _ = vable_loc;
            return;
        }

        let num_args = op.num_args();
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
        self.emit_abi_int_arg_from_imm(0, 1);
        self.emit_abi_int_arg_from_imm(1, jf_alloc_bytes);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD calloc_ptr);
        self.emit_abi_call_rax_aligned();

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
        self.emit_abi_int_arg_from_reg(0, 2);

        // assembler.py:320 _call_assembler_emit_call(descr._ll_function_addr, ...)
        // Resolve target address from descr.call_target_token() or self_entry_label.
        let descr_arc = op.getdescr();
        let target_addr: Option<usize> = descr_arc
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
                );
                self.emit_abi_int_arg_from_reg(0, 2);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD free_ptr);
                self.emit_abi_call_rax_aligned();
                dynasm!(self.mc ; .arch x64 ; mov rbp, r12); // restore caller jf_ptr
                self.emit_abi_int_arg_from_reg(0, 13);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD force_addr);
                self.emit_abi_call_rax_aligned();
                self.reload_frame_if_necessary();
            } else {
                // No force_fn registered — free and return 0.
                self.emit_abi_int_arg_from_reg(0, 2);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD free_ptr);
                self.emit_abi_call_rax_aligned();
                dynasm!(self.mc ; .arch x64
                    ; mov rbp, r12                   // restore caller jf_ptr
                    ; xor eax, eax                   // result = 0
                );
            }
        } else {
            // Resolved target: branch directly to the compiled callee
            // entry.  The previous trampoline indirection re-read
            // MAJIT_LOG on every recursive call (Win32 env lookup is
            // slow); a direct call matches cranelift's dispatch.
            if let Some(addr) = target_addr {
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD addr as i64);
                self.emit_abi_call_rax_aligned();
            } else if self.self_entry_label.is_some() {
                let addr_ptr = self.self_entry_addr_ptr as i64;
                dynasm!(self.mc ; .arch x64
                    ; mov rax, QWORD addr_ptr
                    ; mov rax, [rax]
                );
                self.emit_abi_call_rax_aligned();
            }

            // rax/x0 = callee's returned jf_ptr (= heap jf_ptr we passed).
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
            self.emit_abi_int_arg_from_imm(0, cpu_ptr);
            self.emit_abi_int_arg_from_reg(1, 2);
            self.emit_abi_int_arg_from_imm(2, green_key);
            dynasm!(self.mc ; .arch x64 ; mov rax, QWORD helper_addr);
            self.emit_abi_call_rax_aligned();
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
                );
                self.emit_abi_int_arg_from_reg(0, 2);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD free_ptr);
                self.emit_abi_call_rax_aligned();
                dynasm!(self.mc ; .arch x64
                    ; mov rax, r12                  // float bits in rax
                    ; =>merge
                );
            } else {
                dynasm!(self.mc ; .arch x64
                    ; mov r12, [rdx + FIRST_ITEM_OFFSET as i32]
                );
                self.emit_abi_int_arg_from_reg(0, 2);
                dynasm!(self.mc ; .arch x64 ; mov rax, QWORD free_ptr);
                self.emit_abi_call_rax_aligned();
                dynasm!(self.mc ; .arch x64
                    ; mov rax, r12                  // rax = result
                    ; =>merge
                );
            }
        } // end if is_resolved

        // Store result to the output slot (rax/x0 holds result).
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }

        // Restore callee-saved regs clobbered by this sequence.
    }

    // ----------------------------------------------------------------
    // genop_* — allocation
    // x86/assembler.py:2338 genop_new etc.
    // These require GC runtime support. Emit trap for now.
    // ----------------------------------------------------------------

    /// rewrite.py:936-942 `handle_write_barrier_setarrayitem` value gate.
    ///
    /// PyPy emits a barrier only for Ref-typed values, and `rgc.needs_write_barrier`
    /// returns false for NULL constants and true for non-NULL constants
    /// (rpython/rlib/rgc.py:285-297).  The normal rewriter path already performs
    /// this check; this is the direct-backend fallback for unre-written tests.
    fn setarrayitem_value_needs_write_barrier(&self, value: OpRef, value_loc: &Loc) -> bool {
        if matches!(value_loc, Loc::Reg(val) if val.is_xmm) {
            return false;
        }
        // history.py:227/268/314 — inline-Const variants carry value
        // inline; the variant tag IS the box type, subsuming the
        // legacy constant_types side table.
        if let Some(val) = value.inline_const_bits() {
            if !matches!(value, OpRef::ConstPtr(_)) {
                return false;
            }
            return val != 0;
        }
        if let Some(tp) = self.constant_types.get(&value.raw()) {
            if *tp != Type::Ref {
                return false;
            }
        }
        if let Some(&constant) = self.constants.get(&value.raw()) {
            return constant != 0;
        }
        !matches!(value_loc, Loc::Immed(i) if i.value == 0)
    }

    /// rewrite.py:955-973 `gen_write_barrier_array` for the direct
    /// SETARRAYITEM_GC fallback.  This assembler-only path has no
    /// `RewriteState.known_lengths`, so it mirrors PyPy's
    /// `known_length(v_base, LARGE)` default: unknown length selects the array
    /// barrier when card marking exists; otherwise it falls back to the generic
    /// write barrier.
    fn emit_setarrayitem_gc_write_barrier(&mut self, arglocs: &[Loc]) {
        let use_array_barrier = crate::runner::DYNASM_ACTIVE_GC.with(|cell| {
            cell.borrow()
                .as_ref()
                .and_then(|gc| gc.get_write_barrier_descr())
                .is_some_and(|wb| wb.jit_wb_cards_set != 0)
        });
        self.emit_write_barrier_fastpath_kind(arglocs, use_array_barrier);
    }

    /// x86/assembler.py:2438 _write_barrier_fastpath parity.
    fn emit_write_barrier_fastpath(&mut self, op: &Op, arglocs: &[Loc]) {
        let is_array = op.opcode == majit_ir::OpCode::CondCallGcWbArray;
        self.emit_write_barrier_fastpath_kind(arglocs, is_array);
    }

    fn emit_write_barrier_fastpath_kind(&mut self, arglocs: &[Loc], is_array: bool) {
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
            match arglocs.get(1) {
                Some(Loc::Reg(loc_index)) => {
                    let page_shift = wb.jit_wb_card_page_shift as i8;
                    let byte_shift = (3 + wb.jit_wb_card_page_shift) as i8;
                    dynasm!(self.mc ; .arch x64
                        ; push r10
                        ; push rcx
                        ; push rdx
                        ; mov r11, Rq(loc_index.value as u8)
                        ; mov r10, Rq(loc_base.value as u8)
                        ; shr r11, byte_shift
                        ; not r11
                        ; sub r11, majit_gc::header::GcHeader::SIZE as i32
                        ; mov rcx, Rq(loc_index.value as u8)
                        ; shr rcx, page_shift
                        ; and rcx, 7
                        ; mov dl, 1
                        ; shl dl, cl
                        ; or BYTE [r10 + r11], dl
                        ; pop rdx
                        ; pop rcx
                        ; pop r10
                    );
                }
                Some(Loc::Frame(loc_index)) => {
                    let page_shift = wb.jit_wb_card_page_shift as i8;
                    let byte_shift = (3 + wb.jit_wb_card_page_shift) as i8;
                    let index_offset = loc_index.ebp_loc.value;
                    dynasm!(self.mc ; .arch x64
                        ; push r10
                        ; push rcx
                        ; push rdx
                        ; mov r11, [rbp + index_offset]
                        ; mov r10, Rq(loc_base.value as u8)
                        ; shr r11, byte_shift
                        ; not r11
                        ; sub r11, majit_gc::header::GcHeader::SIZE as i32
                        ; mov rcx, [rbp + index_offset]
                        ; shr rcx, page_shift
                        ; and rcx, 7
                        ; mov dl, 1
                        ; shl dl, cl
                        ; or BYTE [r10 + r11], dl
                        ; pop rdx
                        ; pop rcx
                        ; pop r10
                    );
                }
                Some(Loc::Immed(loc_index)) => {
                    let byte_index = loc_index.value >> wb.jit_wb_card_page_shift;
                    let byte_ofs =
                        !((byte_index >> 3) as i64) - majit_gc::header::GcHeader::SIZE as i64;
                    let byte_val = 1_i64 << (byte_index & 7);
                    dynasm!(self.mc ; .arch x64
                        ; or BYTE [Rq(loc_base.value as u8) + byte_ofs as i32], byte_val as i8
                    );
                }
                _ => {}
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
        self.emit_abi_int_arg_from_reg(0, loc_base.value as u8);
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD helper
        );
        self.emit_abi_call_rax_after_one_push();
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
        let size_ref = op.arg(0).to_opref();
        // history.py:227 ConstInt.value carried inline — prefer the inline
        // payload before falling through to the legacy pool / raw u32.
        let total_size = size_ref.inline_const_bits().unwrap_or_else(|| {
            self.constants
                .get(&size_ref.raw())
                .copied()
                .unwrap_or(size_ref.raw() as i64)
        });
        let gc_header_size = majit_gc::header::GcHeader::SIZE as i64;
        // gc.py:525-531 — read nursery slot addresses from the active GC
        // descriptor (cpu.gc_ll_descr.get_nursery_free_addr() parity), not
        // from a process-global singleton.
        let (nf_addr, nt_addr) = crate::runner::dynasm_nursery_addrs();

        let nf = nf_addr as i64;
        let nt = nt_addr as i64;
        // assembler.py:2556 `malloc_cond` clobbers only ECX/EDX (the pair
        // regalloc spilled via MALLOC_NURSERY_CLOBBER) because PyPy's
        // encoder supports `MOV [imm64], reg` directly.  dynasm-rs has no
        // such encoding so we need a third register to stage the absolute
        // nursery slot addresses; use R11 (X86_64_SCRATCH_REG, outside
        // ALL_CORE_REGS) instead of RAX — RAX is in the regalloc pool and
        // clobbering it would silently destroy any live Box the regalloc
        // bound to it.  The slow path preserves RAX via push_all_regs.
        let scratch = crate::regloc::X86_64_SCRATCH_REG.value;

        // ecx = nursery_free, edx = new nursery_free
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD nf
            ; mov rcx, [Rq(scratch)]
            ; lea rdx, [rcx + total_size as i32]
            ; mov Rq(scratch), QWORD nt
            ; cmp rdx, [Rq(scratch)]
        );

        let slow_path = self.mc.new_dynamic_label();
        let done = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; ja =>slow_path);

        // Fast path: update nursery_free, zero header, compute obj ptr.
        // Stage the `*nf = new_free` store through R11; materialise the
        // payload pointer directly into `result_reg` (regalloc forces it
        // to ECX, MALLOC_NURSERY_RESULT) so both paths converge with the
        // payload in the same register.
        dynasm!(self.mc ; .arch x64
            ; mov Rq(scratch), QWORD nf
            ; mov [Rq(scratch)], rdx
            ; mov QWORD [rcx], 0       // zero GcHeader
        );
        let result_reg_for_payload = match result_loc {
            Some(Loc::Reg(r)) => r.value,
            _ => crate::regloc::ECX.value,
        };
        dynasm!(self.mc ; .arch x64
            ; lea Rq(result_reg_for_payload), [rcx + gc_header_size as i32]
        );
        dynasm!(self.mc ; .arch x64 ; jmp =>done);

        // Slow path: helper extraction (PyPy assembler.py:295 `mc.CALL`).
        dynasm!(self.mc ; .arch x64 ; =>slow_path);
        let gcmap_ofs = crate::jitframe::JF_GCMAP_OFS;
        if let Some(gcmap) = self.pending_malloc_nursery_gcmap {
            self.push_gcmap(gcmap as *mut usize);
        } else {
            dynasm!(self.mc ; .arch x64 ; mov QWORD [rbp + gcmap_ofs], 0);
        }
        // Stage the trampoline address through R11 so RAX still holds
        // the caller's pre-call value at the trampoline entry — its
        // `push_all_regs_to_frame([ECX, EDX])` then saves the real RAX,
        // and the matching pop restores it after the helper call.
        // Loading `helper_addr` into RAX here (the previous shape) would
        // clobber the caller's RAX, and the trampoline would save+restore
        // that already-clobbered value, silently dropping any live Box
        // the regalloc kept in RAX across this op.
        let helper_addr = self.malloc_slowpath_fixed as i64;
        let call_scratch = crate::regloc::X86_64_SCRATCH_REG.value;
        dynasm!(self.mc ; .arch x64
            ; mov Rq(call_scratch), QWORD helper_addr
            ; call Rq(call_scratch)
        );
        // assembler.py:304 — helper returns the payload in ECX
        // (`MOV_rr(ecx, eax)` inside the trampoline) so the value
        // survives the trampoline's `pop_all_regs([ECX, EDX])`.  The
        // regalloc forces `result_reg = MALLOC_NURSERY_RESULT = ECX`
        // (regalloc.rs:105), so the value already lives in the right
        // register and no caller-side copy is needed.  If a future
        // regalloc change picks a different `result_reg`, copy it
        // from RCX (not RAX, which is now the caller's preserved
        // pre-call value, not the helper return).
        //
        // OOM propagation: assembler.py:300-322 emits the `TEST/JZ
        // propagate_exception_path` *inside* the slowpath itself, and
        // pyre's `build_malloc_slowpath_fixed` now mirrors that —
        // when the underlying `dynasm_nursery_slowpath` returns NULL
        // the trampoline does the `_store_and_reset_exception`,
        // writes `jf_descr = propagate_exception_descr` and runs
        // `_call_footer` to exit the trace.  No call-site OOM check is
        // needed; if the trampoline ever returns here it succeeded.
        if let Some(Loc::Reg(r)) = result_loc {
            if r.value != crate::regloc::ECX.value {
                let rv = r.value;
                dynasm!(self.mc ; .arch x64 ; mov Rq(rv), rcx);
            }
        }
        // gcmap was cleared by `pop_gcmap(mc)` inside the trampoline
        // before RET (PyPy assembler.py:308); no caller-side clear is
        // needed here.  Matches the PyPy contract where the trampoline
        // owns the `JF_GCMAP_OFS` reset.
        let _ = gcmap_ofs;

        dynasm!(self.mc ; .arch x64 ; =>done);
        // Spill the result to the regalloc-assigned jitframe slot.  Stage
        // it directly from `result_reg`; routing through RAX (the previous
        // shape) silently clobbered any live Box the regalloc bound to
        // RAX across this op, since the malloc-nursery clobber set is
        // only ECX/EDX.
        let pos = op.pos.get();
        if !pos.is_none() {
            let slot = self.allocate_slot(pos);
            let offset = Self::slot_offset(slot);
            // malloc_cond / malloc_cond_varsize (assembler.py:2556,2604) keep
            // the allocated pointer in ecx and route it through the regalloc-
            // assigned `result_reg`.  Anything else here would spill a stale
            // RAX (now caller-live) instead of the object pointer.
            let Some(Loc::Reg(r)) = result_loc else {
                panic!("CallMallocNursery result_loc must be a register; got {result_loc:?}");
            };
            let rv = r.value;
            dynasm!(self.mc ; .arch x64 ; mov [rbp + offset], Rq(rv));
        }
    }

    /// NEW: allocate a fixed-size object. Requires GC runtime.
    /// Emits a trap (UD2/BRK) until GC nursery allocation is wired.
    fn genop_new(&mut self, op: &Op) {
        // Simple allocation: call libc malloc(obj_size).
        // RPython uses GC nursery bump allocation; we use malloc as stub.
        let obj_size = op.with_size_descr(|sd| sd.size()).unwrap_or(16) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        // Call malloc(obj_size)
        self.emit_abi_int_arg_from_imm(0, obj_size);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD malloc_ptr);
        self.emit_abi_call_rax();
        // rax/x0 = pointer to allocated memory
        // Zero-initialize
        self.emit_abi_int_arg_from_reg(0, 0);
        self.emit_abi_int_arg_from_imm(1, 0);
        self.emit_abi_int_arg_from_imm(2, obj_size);
        dynasm!(self.mc ; .arch x64
            ; push rax           // save ptr
            ; mov rax, QWORD (libc::memset as *const () as i64)
        );
        self.emit_abi_call_rax_after_one_push();
        dynasm!(self.mc ; .arch x64 ; pop rax); // restore ptr
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// NEW_WITH_VTABLE: allocate and set vtable pointer.
    fn genop_new_with_vtable(&mut self, op: &Op) {
        // Same as New, but also write vtable at offset 0.
        let obj_size = op.with_size_descr(|sd| sd.size()).unwrap_or(16) as i64;
        let vtable = op.with_size_descr(|sd| sd.vtable()).unwrap_or(0) as i64;
        let malloc_ptr = libc::malloc as *const () as i64;
        self.emit_abi_int_arg_from_imm(0, obj_size);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD malloc_ptr);
        self.emit_abi_call_rax();
        self.emit_abi_int_arg_from_reg(0, 0);
        self.emit_abi_int_arg_from_imm(1, 0);
        self.emit_abi_int_arg_from_imm(2, obj_size);
        dynasm!(self.mc ; .arch x64
            ; push rax
            ; mov rax, QWORD (libc::memset as *const () as i64)
        );
        self.emit_abi_call_rax_after_one_push();
        dynasm!(self.mc ; .arch x64 ; pop rax);
        // Write vtable at offset 0
        if vtable != 0 {
            dynasm!(self.mc ; .arch x64
                ; mov rcx, QWORD vtable
                ; mov [rax], rcx
            );
        }
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// NEW_ARRAY / NEW_ARRAY_CLEAR: allocate an array.
    fn genop_new_array(&mut self, op: &Op) {
        let (base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
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
        self.store_rax_to_result(op.pos.get());
    }

    /// STRLEN / UNICODELEN: result = string.length
    /// Load the length field from the string/unicode object header.
    /// arg0 = string pointer. The length is at a fixed offset in the
    /// RPython string representation. For RPython strings, the length
    /// is typically at offset 8 (after the GC header / hash field).
    fn genop_strlen(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        self.load_arg_to_rax(op.arg(0).to_opref());

        dynasm!(self.mc
            ; .arch x64
            ; mov rax, [rax + offset]
        );

        self.store_rax_to_result(op.pos.get());
    }

    /// STRGETITEM / UNICODEGETITEM: result = string[index]
    /// arg0 = string pointer, arg1 = index.
    /// Address = base + (basesize - extra_null) + index * itemsize, per
    /// `rewrite.py:295-306` — STR has `extra_item_after_alloc=1` so the
    /// token basesize overshoots the first char by 1; UNICODE does not.
    fn genop_strgetitem(&mut self, op: &Op) {
        let (mut base_size, item_size) = op
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1)); // rstr.STR token defaults (basesize=17, itemsize=1)
        if op.opcode == OpCode::Strgetitem {
            debug_assert_eq!(item_size, 1, "STRGETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:299 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0).to_opref()); // string pointer
        self.load_arg_to_rcx(op.arg(1).to_opref()); // index

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

        self.store_rax_to_result(op.pos.get());
    }

    // ================================================================
    // assembler.py:1817 genop_save_exc_class / genop_save_exception
    // ================================================================

    /// assembler.py:1817 genop_save_exc_class — stub: returns 0.
    fn genop_save_exc_class(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// assembler.py:1827 genop_save_exception — stub: returns 0.
    fn genop_save_exception(&mut self, op: &Op) {
        dynasm!(self.mc ; .arch x64 ; xor eax, eax);
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    // ================================================================
    // genop_* — extended integer arithmetic
    // ================================================================

    /// INT_FLOORDIV: result = arg0 / arg1 (signed)
    fn genop_int_floordiv(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_MOD: result = arg0 % arg1 (signed)
    fn genop_int_mod(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64
            ; cqo
            ; idiv rcx
            ; mov rax, rdx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// UINT_MUL_HIGH: upper 64 bits of unsigned multiply
    fn genop_uint_mul_high(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64
            ; mul rcx
            ; mov rax, rdx
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// INT_SIGNEXT: sign-extend from num_bytes width to 64 bits.
    fn genop_int_signext(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        let num_bytes = match self.resolve_opref(op.arg(1).to_opref()) {
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
        self.store_rax_to_result(op.pos.get());
    }

    // ================================================================
    // genop_* — extended float operations
    // ================================================================

    /// FLOAT_ABS: result = |arg0|
    fn genop_float_abs(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        let mask: i64 = i64::MAX; // 0x7FFF_FFFF_FFFF_FFFF
        dynasm!(self.mc ; .arch x64
            ; mov rax, QWORD mask
            ; movq xmm1, rax
            ; andpd xmm0, xmm1
        );
        self.store_d0_to_result(op.pos.get());
    }

    /// FLOAT_LT/LE/EQ/NE/GT/GE: float comparison.
    /// For lt/le, swap operands so JA/JAE handles NaN correctly.
    fn genop_float_cmp(&mut self, op: &Op) {
        let swap = matches!(op.opcode, OpCode::FloatLt | OpCode::FloatLe);
        if swap {
            self.load_float_arg_to_d0(op.arg(1).to_opref());
            self.load_float_arg_to_d1(op.arg(0).to_opref());
        } else {
            self.load_float_arg_to_d0(op.arg(0).to_opref());
            self.load_float_arg_to_d1(op.arg(1).to_opref());
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
        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// CAST_FLOAT_TO_SINGLEFLOAT: f64 → f32 (bits in lower 32 of i64)
    fn genop_cast_float_to_singlefloat(&mut self, op: &Op) {
        self.load_float_arg_to_d0(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch x64
            ; cvtsd2ss xmm0, xmm0
            ; movd eax, xmm0
        );
        self.store_rax_to_result(op.pos.get());
    }

    /// CAST_SINGLEFLOAT_TO_FLOAT: f32 (bits in lower 32) → f64
    fn genop_cast_singlefloat_to_float(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        dynasm!(self.mc ; .arch x64
            ; movd xmm0, eax
            ; cvtss2sd xmm0, xmm0
        );
        self.store_d0_to_result(op.pos.get());
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
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);

        let itemsize = self.resolve_const_or(op.arg(2).to_opref(), 8) as i32;
        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos.get());
    }

    /// GC_LOAD_INDEXED_I/R/F: load from base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=scale, arg(3)=base_offset, arg(4)=itemsize.
    fn genop_gc_load_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(2).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(3).to_opref(), 0) as i32;
        let itemsize = self.resolve_const_or(op.arg(4).to_opref(), 8) as i32;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        if scale != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
        }

        self.emit_load_from_rax_sized(itemsize);
        self.store_rax_to_result(op.pos.get());
    }

    /// GC_STORE: store value to base + offset.
    /// 4-arg form: arg(0)=base, arg(1)=offset, arg(2)=value, arg(3)=itemsize.
    fn genop_discard_gc_store(&mut self, op: &Op) {
        if op.num_args() < 4 {
            return; // 3-arg GC rewrite form — skip for now
        }
        let itemsize = self
            .resolve_const_or(op.arg(3).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// GC_STORE_INDEXED: store to base + base_offset + index * scale.
    /// arg(0)=base, arg(1)=index, arg(2)=value, arg(3)=scale,
    /// arg(4)=base_offset, arg(5)=itemsize.
    fn genop_discard_gc_store_indexed(&mut self, op: &Op) {
        let scale = self.resolve_const_or(op.arg(3).to_opref(), 1) as i32;
        let base_offset = self.resolve_const_or(op.arg(4).to_opref(), 0) as i32;
        let itemsize = self
            .resolve_const_or(op.arg(5).to_opref(), 8)
            .unsigned_abs() as usize;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        if scale != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if base_offset != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, base_offset);
        }
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(itemsize);
    }

    /// RAW_LOAD_I/F: load from base + offset using descriptor.
    fn genop_raw_load(&mut self, op: &Op) {
        let offset = Self::field_offset_from_descr(op);
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);

        self.emit_load_from_rax_sized(size as i32);
        let _ = offset; // offset is in the descriptor, not used for raw_load
        self.store_rax_to_result(op.pos.get());
    }

    /// RAW_STORE: store value to base + offset using descriptor.
    fn genop_discard_raw_store(&mut self, op: &Op) {
        let size = Self::field_size_from_descr(op);

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);
        self.emit_store_to_rax_sized(size);
    }

    // ================================================================
    // genop_* — interior field operations
    // ================================================================

    /// GETINTERIORFIELD_GC_I/R/F: load field from array-of-structs element.
    fn genop_getinteriorfield(&mut self, op: &Op) {
        let descr_arc = op.getdescr();
        let (base_size, item_size, field_offset, field_size) = descr_arc
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

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, total_offset
            ; add rax, rcx
        );

        self.emit_load_from_rax_sized(field_size as i32);
        self.store_rax_to_result(op.pos.get());
    }

    /// SETINTERIORFIELD_GC/RAW: write field in array-of-structs element.
    fn genop_discard_setinteriorfield(&mut self, op: &Op) {
        let descr_arc = op.getdescr();
        let (base_size, item_size, field_offset, field_size) = descr_arc
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

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());
        let total_offset = base_size + field_offset;

        if item_size != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, total_offset
            ; add rax, rcx
        );
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
        dynasm!(self.mc ; .arch x64 ; pop rax);

        self.emit_store_to_rax_sized(field_size);
    }

    // ================================================================
    // genop_* — call variants
    // ================================================================

    /// COND_CALL_N: if arg(0) != 0, call function at arg(1).
    ///
    /// `x86/assembler.py:2526 cond_call` parity: the regalloc may fuse
    /// a preceding CompOp's result into `guard_success_cc` rather than
    /// materialising the boolean (see `next_op_can_accept_cc`). When
    /// that's the case, `op.arg(0)` lives in the condition flags, not
    /// a register/slot — so we must branch off the CC directly instead
    /// of issuing `load_arg_to_rax; test rax, rax`, which would read
    /// `rbp` (the frame_reg sentinel) and miss the comparison result.
    fn genop_discard_cond_call(&mut self, op: &Op) {
        let skip_label = self.mc.new_dynamic_label();
        if let Some(cc) = self.guard_success_cc.take() {
            self.emit_jcc_to_label(invert_cc(cc), skip_label);
        } else {
            self.load_arg_to_rax(op.arg(0).to_opref());
            dynasm!(self.mc ; .arch x64 ; test rax, rax ; jz =>skip_label);
        }

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch x64 ; =>skip_label);
    }

    /// COND_CALL_VALUE_I/R: if arg(0) == 0, call function; else result = arg(0).
    fn genop_cond_call_value(&mut self, op: &Op) {
        self.load_arg_to_rax(op.arg(0).to_opref());
        let skip_label = self.mc.new_dynamic_label();
        dynasm!(self.mc ; .arch x64 ; test rax, rax ; jnz =>skip_label);

        self.emit_call(op, 1);

        dynasm!(self.mc ; .arch x64 ; =>skip_label);

        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
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
            .with_array_descr(|ad| (ad.base_size() as i32, ad.item_size() as i32))
            .unwrap_or((17, 1));
        if op.opcode == OpCode::Strsetitem {
            debug_assert_eq!(item_size, 1, "STRSETITEM itemsize must be 1");
            base_size -= 1; // rewrite.py:311 — skip the extra null character
        }

        self.load_arg_to_rax(op.arg(0).to_opref()); // string
        self.load_arg_to_rcx(op.arg(1).to_opref()); // index
        if item_size != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, item_size);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, base_size ; add rax, rcx);
        dynasm!(self.mc ; .arch x64 ; push rax);
        self.load_arg_to_rcx(op.arg(2).to_opref());
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
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
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
        self.load_arg_to_rax(op.arg(4).to_opref());
        if item_size != 1 {
            dynasm!(self.mc ; .arch x64
                ; mov rcx, QWORD item_size
                ; imul rax, rcx
            );
        }
        dynasm!(self.mc ; .arch x64 ; push rax); // [rsp] = byte_count

        // Compute src_addr = src + base_size + src_start * item_size
        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(2).to_opref());
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
        self.load_arg_to_rax(op.arg(1).to_opref());
        self.load_arg_to_rcx(op.arg(3).to_opref());
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
            ; pop rsi        // src
            ; pop rdx        // count
            ; push rbp
        );
        self.emit_abi_int_arg_from_reg(2, 2); // count
        self.emit_abi_int_arg_from_reg(1, 6); // src
        self.emit_abi_int_arg_from_reg(0, 0); // dst
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD memmove_ptr);
        self.emit_abi_call_rax_after_one_push();
        dynasm!(self.mc ; .arch x64 ; pop rbp);
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
        op.with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((fallback_base, fallback_item))
    }

    /// Shared implementation for NEWSTR / NEWUNICODE / NEW_ARRAY.
    /// Allocates base_size + length * item_size bytes, zero-fills,
    /// and writes length to the header.
    fn genop_alloc_varsize(&mut self, op: &Op, base_size: i64, item_size: i64) {
        // arg(0) = length
        self.load_arg_to_rax(op.arg(0).to_opref());
        let malloc_ptr = libc::malloc as *const () as i64;
        let memset_ptr = libc::memset as *const () as i64;

        // Save length, compute total_size = base_size + length * item_size
        dynasm!(self.mc ; .arch x64
            ; push rax                           // save length
            ; imul rax, rax, item_size as i32
            ; add rax, base_size as i32
            ; push rax                           // save total_size
        );
        self.emit_abi_int_arg_from_reg(0, 0);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD malloc_ptr);
        self.emit_abi_call_rax();
        dynasm!(self.mc ; .arch x64
            ; pop rcx                            // rcx = total_size
            ; push rax                           // save ptr
        );
        self.emit_abi_int_arg_from_reg(2, 1);
        self.emit_abi_int_arg_from_imm(1, 0);
        self.emit_abi_int_arg_from_reg(0, 0);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD memset_ptr);
        self.emit_abi_call_rax();
        dynasm!(self.mc ; .arch x64
            ; pop rax                            // rax = ptr
            ; pop rcx                            // rcx = length
            // Store length at offset 8 (RPython string header)
            ; mov [rax + 8], rcx
        );

        if !op.pos.get().is_none() {
            self.store_rax_to_result(op.pos.get());
        }
    }

    /// ZERO_ARRAY: zero a range in an array.
    /// arg(0)=base, arg(1)=start, arg(2)=size, arg(3)=scale_start, arg(4)=scale_size.
    fn genop_discard_zero_array(&mut self, op: &Op) {
        let (base_size, _) = op
            .with_array_descr(|ad| (ad.base_size() as i64, ad.item_size() as i64))
            .unwrap_or((8, 8));

        let scale_start = self.resolve_const_or(op.arg(3).to_opref(), 1);
        let scale_size = self.resolve_const_or(op.arg(4).to_opref(), 1);
        let memset_ptr = libc::memset as *const () as i64;

        // byte_offset = base_size + start * scale_start
        // byte_length = size * scale_size
        self.load_arg_to_rax(op.arg(0).to_opref()); // base
        self.load_arg_to_rcx(op.arg(1).to_opref()); // start

        if scale_start != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rcx, rcx, scale_start as i32);
        }
        dynasm!(self.mc ; .arch x64
            ; add rax, DWORD base_size as i32
            ; add rax, rcx
            ; push rax                           // save dest
        );
        self.load_arg_to_rax(op.arg(2).to_opref()); // size
        if scale_size != 1 {
            dynasm!(self.mc ; .arch x64 ; imul rax, rax, scale_size as i32);
        }
        // memset(dest, 0, byte_length)
        dynasm!(self.mc ; .arch x64
            ; mov rdx, rax                       // byte_length
            ; pop rax                            // dest
            ; push rbp
        );
        self.emit_abi_int_arg_from_reg(2, 2);
        self.emit_abi_int_arg_from_imm(1, 0);
        self.emit_abi_int_arg_from_reg(0, 0);
        dynasm!(self.mc ; .arch x64 ; mov rax, QWORD memset_ptr);
        self.emit_abi_call_rax_after_one_push();
        dynasm!(self.mc ; .arch x64 ; pop rbp);
    }

    // ================================================================
    // genop_* — address computation
    // ================================================================

    /// LOAD_EFFECTIVE_ADDRESS: result = base + (index << shift) + baseofs.
    /// resoperation.py:1052-1054 — `[v_gcptr, v_index, c_baseofs, c_shift]`.
    /// arg(0)=base, arg(1)=index, arg(2)=baseofs, arg(3)=shift.
    fn genop_load_effective_address(&mut self, op: &Op) {
        let baseofs = self.resolve_const_or(op.arg(2).to_opref(), 0) as i32;
        let shift = self.resolve_const_or(op.arg(3).to_opref(), 0) as i32;

        self.load_arg_to_rax(op.arg(0).to_opref());
        self.load_arg_to_rcx(op.arg(1).to_opref());

        if shift != 0 {
            dynasm!(self.mc ; .arch x64 ; shl rcx, BYTE shift as i8);
        }
        dynasm!(self.mc ; .arch x64 ; add rax, rcx);
        if baseofs != 0 {
            dynasm!(self.mc ; .arch x64 ; add rax, baseofs);
        }

        self.store_rax_to_result(op.pos.get());
    }

    // ----------------------------------------------------------------
    // Public: set constants from external source
    // ----------------------------------------------------------------

    /// Populate the constants map. Called by the frontend before assembly
    /// if constant OpRefs are used (OpRef.0 >= 10000).
    pub fn set_constants(&mut self, constants: majit_ir::VecAssoc<u32, i64>) {
        self.constants = constants;
    }

    /// Set constant type annotations for the next compile call.
    pub fn set_constant_types(&mut self, constant_types: majit_ir::VecAssoc<u32, majit_ir::Type>) {
        self.constant_types = constant_types;
    }
}
