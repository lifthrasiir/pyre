//! Phase 4 Epic B.5: regression guard for polymorphic per-pc liveness.
//!
//! Background: B.2 ports `rpython/jit/codewriter/liveness.py:33-79
//! _compute_liveness_must_continue` (backward dataflow over `op_metadata`),
//! B.3 registers each per-marker `(live_i, live_r, live_f)` triple via
//! `Assembler::_register_liveness_offset`, and B.4 patches the BC_LIVE
//! 2-byte slot per marker via `JitCodeBuilder::finalize_liveness`.
//!
//! Before B.3-B.4 every BC_LIVE marker in macro-lowered JitCodes pointed
//! at the canonical "everything-alive" entry at offset 0 — over-approximating
//! `frame.fail_args` during blackhole resume.  This test pins the post-B.3-B.4
//! invariants by driving a synthetic `#[jit_interp]` consumer whose arms have
//! intentionally-different op chains before each guard, so the lowerer's
//! per-marker analysis must produce distinct live sets per arm.
//!
//! Hard regression invariants enforced:
//! 1. `__prebuild_jitcode_liveness_*` registers more than just the canonical
//!    entry into the shared `Assembler` (proving the macro pipeline is
//!    actually running per-marker analysis, not collapsing every arm to the
//!    canonical set).
//! 2. `__jitcode_*` factory calls AFTER `install_canonical_liveness` must
//!    not grow `asm.all_liveness()` — every per-marker triple must already
//!    be registered by prebuild.  Mirrors the runtime assertion at
//!    `codegen_trace.rs:178-185`.
//! 3. At least two of the per-arm JitCodes must contain BC_LIVE markers
//!    pointing at *distinct* offsets (the polymorphism check).

use majit_metainterp::{Assembler, BC_LIVE, JitCode, JitDriver, JitState as _};

// ── Synthetic state ────────────────────────────────────────────────

struct Polymorphic4State {
    a: i64,
    b: i64,
    c: i64,
    d: i64,
}

const OP_GUARD_A: u8 = 1;
const OP_SUM_AB: u8 = 2;
const OP_SUM_ABC: u8 = 3;
const OP_SUM_ABCD: u8 = 4;
const OP_END: u8 = 0;

pub type Bytecode = [u8];

// `BytecodeExt::get_op` is consumed by the macro-emitted `__trace_*` fn
// (codegen_trace.rs:61 `program.get_op(pc)`), but the integration tests
// below drive only `__jitcode_*` / `__prebuild_*`, so the compiler flags
// the trait as dead code in this binary's reachability graph.
#[allow(dead_code)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}

impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

// Each arm drives a guard with a deliberately-different number of
// `load_state_field` + `record_binop_i` ops upstream, so the backward
// walker (`compute_per_marker_liveness`) sees distinct register
// allocations live at each arm's BC_LIVE marker.
//
// arm_GUARD_A:  R0 = a;                 marker; goto_if_not(R0)            → live={R0}
// arm_SUM_AB:   R0 = a; R1 = b; R2 = R0+R1; marker; goto_if_not(R2)        → live={R2}
// arm_SUM_ABC:  R0..R4 chain;            marker; goto_if_not(R4)            → live={R4}
// arm_SUM_ABCD: R0..R6 chain;            marker; goto_if_not(R6)            → live={R6}
//
// Encoded as `live_i` byte payloads, those four sets live at distinct
// offsets in `asm.all_liveness` (each register-index lands in a different
// bitmap byte position).
#[majit_macros::jit_interp(
    state = Polymorphic4State,
    env = Bytecode,
    state_fields = {
        a: int,
        b: int,
        c: int,
        d: int,
    },
)]
#[allow(unused_assignments, unused_variables)]
fn polymorphic_mainloop(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<Polymorphic4State> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = Polymorphic4State {
        a: 1,
        b: 1,
        c: 1,
        d: 1,
    };

    {
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }

    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_GUARD_A => {
                if state.a != 0 {
                    state.a = state.a + 1;
                }
            }
            OP_SUM_AB => {
                if state.a + state.b != 0 {
                    state.a = state.a + 1;
                }
            }
            OP_SUM_ABC => {
                if state.a + state.b + state.c != 0 {
                    state.a = state.a + 1;
                }
            }
            OP_SUM_ABCD => {
                if state.a + state.b + state.c + state.d != 0 {
                    state.a = state.a + 1;
                }
            }
            _ => break,
        }
    }
    state.a
}

// ── Helpers ────────────────────────────────────────────────────────

/// Walk a JitCode body and collect every BC_LIVE marker's 2-byte offset.
fn collect_bc_live_offsets(jitcode: &JitCode) -> Vec<u16> {
    let code = &jitcode.code;
    let mut offsets = Vec::new();
    let mut i = 0;
    while i < code.len() {
        if code[i] == BC_LIVE {
            // BC_LIVE is followed by a 2-byte little-endian offset into
            // `Assembler::all_liveness` (assembler.py:248 `encode_offset`).
            assert!(
                i + 2 < code.len(),
                "BC_LIVE at end of code without 2-byte offset payload"
            );
            offsets.push(u16::from_le_bytes([code[i + 1], code[i + 2]]));
            i += 3;
        } else {
            i += 1;
        }
    }
    offsets
}

/// Build the polymorphic factory's JitCode for a given (pc, op).
///
/// `__prebuild_jitcode_liveness_polymorphic_mainloop` must be called
/// once before any factory invocation so per-marker triples land in
/// `asm.all_liveness` ahead of the runtime `finalize_liveness` patch.
fn build_jitcode_for_op(asm: &mut Assembler, program: &[u8], pc: usize, op: u8) -> JitCode {
    __jitcode_polymorphic_mainloop(asm, program, pc, op)
        .unwrap_or_else(|| panic!("factory returned None for op={op}"))
}

// ── Tests ──────────────────────────────────────────────────────────

#[test]
fn prebuild_registers_more_than_canonical_entry() {
    // Drive the macro-emitted prebuild against a fresh Assembler. With four
    // arms whose register layouts at the BC_LIVE marker differ, the prebuild
    // must register at least one per-marker triple distinct from the
    // canonical `[0,1,2,3]` entry — otherwise the per-pc walker collapsed
    // every arm onto the canonical set (B.2 walker regression).
    let mut asm = Assembler::new();
    // Mirror `install_canonical_liveness`'s order: canonical entry first,
    // then per-arm triples via the macro-emitted prebuild.
    let canonical: Vec<u8> = (0..4u8).collect();
    let canonical_offset = asm._register_liveness_offset(&canonical, &[], &[]);
    assert_eq!(canonical_offset, 0, "canonical sits at offset 0");
    let canonical_len = asm.all_liveness().len();

    __prebuild_jitcode_liveness_polymorphic_mainloop(&mut asm);

    let post_prebuild_len = asm.all_liveness().len();
    assert!(
        post_prebuild_len > canonical_len,
        "prebuild must register at least one non-canonical triple \
         (canonical_len={canonical_len}, post_prebuild_len={post_prebuild_len})"
    );
}

#[test]
fn factory_does_not_grow_asm_after_prebuild() {
    // Mirror the runtime assertion at `codegen_trace.rs:178-185`: every
    // per-marker triple emitted by the per-arm builder must already be in
    // `asm.all_liveness` (prebuild hit), so `finalize_liveness` only
    // dedups against existing offsets.  A regression in the prebuild walker
    // (e.g., missing emit-site coverage) would surface as growth here.
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    // Stage the canonical triple so deferred-canonical patching in
    // `finalize_liveness` can resolve via
    // `ensure_canonical_liveness_offset` — `live_placeholder()` no longer
    // back-patches to a hard-coded offset 0.
    asm.set_canonical_liveness_triple(canonical.clone(), Vec::new(), Vec::new());
    let _ = asm._register_liveness_offset(&canonical, &[], &[]);
    __prebuild_jitcode_liveness_polymorphic_mainloop(&mut asm);

    let post_prebuild_len = asm.all_liveness().len();

    let program = [OP_GUARD_A, OP_SUM_AB, OP_SUM_ABC, OP_SUM_ABCD, OP_END];
    for &op in &[OP_GUARD_A, OP_SUM_AB, OP_SUM_ABC, OP_SUM_ABCD] {
        let _ = build_jitcode_for_op(&mut asm, &program, 1, op);
    }

    assert_eq!(
        asm.all_liveness().len(),
        post_prebuild_len,
        "factory finalize_liveness must not grow all_liveness post-prebuild"
    );
}

#[test]
fn distinct_arms_emit_distinct_bc_live_offsets() {
    // The polymorphism core: at least two of the four arms must emit a
    // BC_LIVE marker whose patched offset differs from the others.  If the
    // macro pipeline regressed to "every arm uses canonical offset 0", all
    // collected offsets would collapse to {0} and this assertion would
    // fire.
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..4u8).collect();
    // Stage the canonical triple before any `live_placeholder()` runs so
    // the deferred-canonical patcher (`finalize_liveness` →
    // `ensure_canonical_liveness_offset`) can resolve.
    asm.set_canonical_liveness_triple(canonical.clone(), Vec::new(), Vec::new());
    let _ = asm._register_liveness_offset(&canonical, &[], &[]);
    __prebuild_jitcode_liveness_polymorphic_mainloop(&mut asm);

    let program = [OP_GUARD_A, OP_SUM_AB, OP_SUM_ABC, OP_SUM_ABCD, OP_END];

    // Collect the per-arm BC_LIVE offset sets.  Skip the leading canonical
    // marker emitted by every arm at body offset 0 (codegen_trace.rs:175
    // `live_placeholder()` without per-pc triple → always patches to the
    // canonical entry at offset 0); the polymorphism signal lives in the
    // remaining markers.
    let mut per_arm_offsets: Vec<Vec<u16>> = Vec::new();
    for &op in &[OP_GUARD_A, OP_SUM_AB, OP_SUM_ABC, OP_SUM_ABCD] {
        let jitcode = build_jitcode_for_op(&mut asm, &program, 1, op);
        let mut offs = collect_bc_live_offsets(&jitcode);
        // drop the leading canonical marker (body offset 0)
        if !offs.is_empty() {
            offs.remove(0);
        }
        per_arm_offsets.push(offs);
    }

    // Sanity: each arm must emit at least one per-pc BC_LIVE marker
    // (the lowerer always emits a `live_placeholder_with_triple` ahead of
    // every conditional guard — the `if` condition in each arm).
    for (op, offs) in [OP_GUARD_A, OP_SUM_AB, OP_SUM_ABC, OP_SUM_ABCD]
        .iter()
        .zip(per_arm_offsets.iter())
    {
        assert!(
            !offs.is_empty(),
            "arm op={op} must emit at least one per-pc BC_LIVE marker; got none"
        );
    }

    // Polymorphism: the union of *all* per-arm marker offsets must contain
    // at least two distinct values.  A regression to "every arm uses the
    // same triple" would collapse this set to size 1.
    let mut all_offsets: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
    for offs in &per_arm_offsets {
        for &off in offs {
            all_offsets.insert(off);
        }
    }
    assert!(
        all_offsets.len() >= 2,
        "polymorphic per-pc liveness regressed: every per-pc BC_LIVE marker \
         points at the same offset — saw {all_offsets:?} across arms \
         OP_GUARD_A/OP_SUM_AB/OP_SUM_ABC/OP_SUM_ABCD"
    );
}
