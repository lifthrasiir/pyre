//! Slice 1 gate: dispatch JitCode IR shape unit tests.
//!
//! Verifies the singleton dispatch JitCode emitted by `__dispatch_jitcode_*`
//! contains the expected portal IR (BC_JIT_MERGE_POINT, BC_LOOP_HEADER) +
//! pre-dispatch ops in source order + opcode fetch + dispatch chain
//! (BC_GOTO_IF_NOT_INT_EQ_CONST chain) + arm BC_INLINE_CALL emissions +
//! loop close.

use majit_metainterp::jitcode::insns::{
    BC_ABORT, BC_GETARRAYITEM_GC_I, BC_GOTO_IF_NOT_INT_EQ, BC_INLINE_CALL, BC_INT_ADD,
    BC_INT_GUARD_VALUE, BC_INT_RETURN, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
    BC_STORE_STATE_FIELD,
};
use majit_metainterp::{Assembler, BC_GOTO, JitCode, JitDriver};

struct DispatchTestState {
    a: i64,
}

const OP_NOP: u8 = 0;
const OP_INC_A: u8 = 1;
const OP_END: u8 = 2;

pub type Bytecode = [u8];

#[allow(dead_code)]
trait BytecodeExt {
    fn get_op(&self, pc: usize) -> u8;
}
impl BytecodeExt for [u8] {
    fn get_op(&self, pc: usize) -> u8 {
        self[pc]
    }
}

#[majit_macros::jit_interp(
    state = DispatchTestState,
    env = Bytecode,
    state_fields = { a: int },
)]
#[allow(unused_assignments, unused_variables)]
fn dispatch_minimal(program: &Bytecode, threshold: u32) -> i64 {
    let mut driver: JitDriver<DispatchTestState> = JitDriver::new(threshold);
    let mut pc: usize = 0;
    let mut state = DispatchTestState { a: 0 };
    {
        use majit_metainterp::JitState as _;
        state
            .build_meta(0, program)
            .install_canonical_liveness(&mut driver);
    }
    while pc < program.len() {
        jit_merge_point!();
        let opcode = program[pc];
        pc += 1;
        match opcode {
            OP_NOP => {}
            OP_INC_A => state.a += 1,
            _ => break,
        }
    }
    state.a
}

#[test]
fn dispatch_jitcode_emits_portal_markers() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    assert!(!code.is_empty(), "dispatch JitCode body must not be empty");
    let mut i = 0;
    while i < code.len() && code[i] == BC_LIVE {
        i += 3;
    }
    assert!(
        i < code.len(),
        "dispatch JitCode body has only BC_LIVE markers; missing portal markers"
    );
    assert!(
        code[i] == BC_JIT_MERGE_POINT || code[i] == BC_JIT_MERGE_POINT_C,
        "dispatch JitCode portal must start with BC_JIT_MERGE_POINT or BC_JIT_MERGE_POINT_C, got byte {:#x} at offset {}",
        code[i],
        i
    );
}

#[test]
fn dispatch_jitcode_contains_inline_call_per_arm() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    let inline_call_count = code.iter().filter(|&&b| b == BC_INLINE_CALL).count();
    assert_eq!(
        inline_call_count, 2,
        "dispatch JitCode must emit one BC_INLINE_CALL per non-default arm; got {}",
        inline_call_count
    );
}

#[test]
fn dispatch_arm_subjitcode_lowers_state_field_write() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let sub_jitcodes = dispatch_jc
        .exec
        .descrs
        .iter()
        .filter_map(|descr| descr.as_jitcode())
        .collect::<Vec<_>>();

    assert_eq!(
        sub_jitcodes.len(),
        2,
        "dispatch JitCode must register one sub-JitCode per non-default arm"
    );
    assert!(
        sub_jitcodes
            .iter()
            .any(|sub| sub.code.iter().any(|&b| b == BC_STORE_STATE_FIELD)),
        "state.a += 1 arm must lower to store_state_field in the dispatch inline-call target; sub codes: {:?}",
        sub_jitcodes
            .iter()
            .map(|sub| sub.code.clone())
            .collect::<Vec<_>>()
    );
    assert!(
        sub_jitcodes
            .iter()
            .all(|sub| !sub.code.iter().any(|&b| b == BC_ABORT)),
        "lowerable dispatch arms must not degenerate to abort sub-JitCodes"
    );
}

mod or_pattern {
    use super::{Bytecode, BytecodeExt, OP_INC_A, OP_NOP};
    use majit_metainterp::jitcode::insns::BC_INLINE_CALL;
    use majit_metainterp::{Assembler, BC_GOTO, JitCode, JitDriver};

    struct OrDispatchState {
        a: i64,
    }

    #[majit_macros::jit_interp(
        state = OrDispatchState,
        env = Bytecode,
        state_fields = { a: int },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_or_pattern(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OrDispatchState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = OrDispatchState { a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP | OP_INC_A => state.a += 1,
                _ => break,
            }
        }
        state.a
    }

    #[test]
    fn dispatch_or_pattern_short_circuits_alternatives_before_inline_call() {
        let mut asm = Assembler::new();
        asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_or_pattern(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_or_pattern(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let first_inline = code
            .iter()
            .position(|&b| b == BC_INLINE_CALL)
            .expect("missing inline_call for OR-pattern arm");
        let first_goto = code
            .iter()
            .position(|&b| b == BC_GOTO)
            .expect("missing successful-alternative jump for OR pattern");

        assert!(
            first_goto < first_inline,
            "OR-pattern dispatch must jump from a successful early alternative \
             to the shared matched body before inline_call; otherwise A | B is \
             lowered as an accidental conjunction"
        );
    }
}

#[test]
fn dispatch_jitcode_contains_loop_back_goto() {
    let mut asm = Assembler::new();
    asm.set_canonical_liveness_triple(vec![0], vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    let goto_count = code.iter().filter(|&&b| b == BC_GOTO).count();
    assert!(
        goto_count >= 1,
        "dispatch JitCode must emit at least one BC_GOTO for loop closure; got {}",
        goto_count
    );
}

/// Slice 1.4: the dispatch JitCode body must contain the opcode-fetch IR ops
/// lowered from `let opcode = program[pc]; pc += 1;`:
///   1. BC_GETARRAYITEM_GC_I — loads program[pc] into an int register
///   2. BC_INT_ADD           — increments pc_reg by 1 (via load_const + int_add)
///
/// pyopcode.py:171 `opcode = ord(co_code[next_instr])` + `next_instr += 1`.
#[test]
fn dispatch_jitcode_lowers_opcode_fetch() {
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..1u8).collect();
    asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    let mp_idx = code
        .iter()
        .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
        .expect("missing JIT_MERGE_POINT");
    let post_mp = &code[mp_idx..];
    assert!(
        post_mp.iter().any(|&b| b == BC_GETARRAYITEM_GC_I),
        "dispatch JitCode body must emit BC_GETARRAYITEM_GC_I for program[pc]; \
         got bytes {:?}",
        post_mp
    );
    assert!(
        post_mp.iter().any(|&b| b == BC_INT_ADD),
        "dispatch JitCode body must emit BC_INT_ADD for pc += 1; \
         got bytes {:?}",
        post_mp
    );
}

/// Slice 1.5: the dispatch JitCode body must emit one BC_GOTO_IF_NOT_INT_EQ
/// per non-default arm (OP_NOP, OP_INC_A → ≥ 2 checks).
///
/// pyopcode.py:183+ if/elif chain over opcode constants.
/// jtransform.py:196-225 optimize_goto_if_not fuses int_eq + goto_if_not
/// into goto_if_not_int_eq/iiL (BC_GOTO_IF_NOT_INT_EQ).
#[test]
fn dispatch_jitcode_emits_chain_of_int_eq_dispatch() {
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..1u8).collect();
    asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    let chain_count = code.iter().filter(|&&b| b == BC_GOTO_IF_NOT_INT_EQ).count();
    assert!(
        chain_count >= 2,
        "dispatch chain must emit at least one BC_GOTO_IF_NOT_INT_EQ per \
         non-default arm; OP_NOP + OP_INC_A → expected ≥ 2, got {}",
        chain_count
    );
}

/// Slice 1.7: the dispatch JitCode default arm must emit a typed return.
/// dispatch_minimal returns `state.a` (i64 / Int binding) so the JitCode
/// must end with BC_INT_RETURN.
///
/// interp_jit.py:95-100 return boundary: when no arm matches the loop
/// exits; the dispatch JitCode signals this via an int_return insn.
#[test]
fn dispatch_jitcode_emits_typed_return_for_default_arm() {
    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..1u8).collect();
    asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");
    let code = &dispatch_jc.code;
    assert!(
        code.iter().any(|&b| b == BC_INT_RETURN),
        "dispatch JitCode default arm must emit BC_INT_RETURN for `state.a` return; \
         got bytes {:?}",
        code
    );
    // Position check: the typed return must come AFTER the dispatch chain's
    // last fall-through GOTO (which targets the typed-return site itself).
    let last_goto_idx = code
        .iter()
        .rposition(|&b| b == BC_GOTO)
        .expect("missing dispatch chain default GOTO");
    let return_idx = code
        .iter()
        .position(|&b| b == BC_INT_RETURN)
        .expect("missing BC_INT_RETURN");
    assert!(
        return_idx > last_goto_idx,
        "BC_INT_RETURN must come after the dispatch chain default GOTO; \
         got return@{} goto@{}",
        return_idx,
        last_goto_idx
    );
}

mod pre_promote {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct PromoteState {
        stackpos: i64,
    }

    type Bytecode = [u8];

    #[majit_macros::jit_interp(
        state = PromoteState,
        env = Bytecode,
        state_fields = { stackpos: int },
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_pre_promote(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<PromoteState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = PromoteState { stackpos: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // jtransform.py:608-615: hint(promote=True) → -live- + int_guard_value
            state.stackpos = majit_metainterp::jit::promote(state.stackpos);
            let opcode = program[pc];
            pc += 1;
            match opcode {
                0 => state.stackpos += 1,
                _ => break,
            }
        }
        state.stackpos
    }

    #[test]
    fn dispatch_jitcode_lowers_pre_dispatch_promote_in_position() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
        };
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_pre_promote(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_with_pre_promote(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        // Search for any int_guard_value opcode after the merge point.
        let guard_idx = code[mp_idx..].iter().position(|&b| b == BC_INT_GUARD_VALUE);
        assert!(
            guard_idx.is_some(),
            "pre-dispatch promote(state.stackpos) must lower to BC_INT_GUARD_VALUE \
             in dispatch JitCode body (RPython jtransform.py:608-615 -live- + guard_value pair)"
        );
    }
}

/// Slice 2.2: `__trace_dispatch_minimal` now accepts the dispatch JitCode
/// singleton as an extra parameter forwarded from `__merge_dispatch_minimal`.
/// When the dispatch JitCode is registered, the trace path uses it directly;
/// when None (pre-Slice-3), it falls back to the per-(pc, op) legacy factory.
///
/// This test exercises the full trace path end-to-end: warmup threshold=1
/// causes one pre-trace iteration, then the trace fires and records ops via
/// the legacy fallback (Slice 3 not yet run). Correctness is the same
/// regardless of which JitCode source is used.
#[test]
fn dispatch_minimal_traces_via_dispatch_jitcode_singleton() {
    // OP_INC_A increments state.a; threshold=1 ensures trace fires after 1 iter.
    // Result must equal 3 (three OP_INC_A ops), then OP_END breaks.
    let program = [OP_INC_A, OP_INC_A, OP_INC_A, OP_END];
    let result = dispatch_minimal(&program, 1);
    assert_eq!(
        result, 3,
        "dispatch_minimal must increment state.a per OP_INC_A; got {}",
        result
    );
}

/// Slice 2.1: JitDriver::register_dispatch_jitcode stores the dispatch JitCode
/// as an Arc singleton and dispatch_jitcode() returns None before registration
/// and Some after.
///
/// RPython parity: metainterp_sd.jitcodes[driver_idx] global registry slot,
/// scoped to the per-#[jit_interp] driver.
#[test]
fn register_dispatch_jitcode_stores_singleton() {
    use std::sync::Arc;

    let mut driver: JitDriver<DispatchTestState> = JitDriver::new(100);
    assert!(
        driver.dispatch_jitcode().is_none(),
        "register_dispatch_jitcode_stores_singleton: dispatch_jitcode must be None before registration"
    );
    // `register_dispatch_jitcode` validates the dispatch JitCode's
    // `BC_JIT_MERGE_POINT` payload against the driver descriptor
    // (`warmspot.py:660-666 make_args_specification` parity), so the
    // schema must be declared first — same lifecycle as the macro-
    // generated install path (`#declare_schema_fn_name(driver)` runs
    // before `ensure_descriptor_registered` in `codegen_state.rs:719`).
    __declare_jit_schema_dispatch_minimal(&mut driver);

    let mut asm = Assembler::new();
    let canonical: Vec<u8> = (0..1u8).collect();
    asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
    __prebuild_jitcode_liveness_dispatch_minimal(&mut asm);
    let _ = asm.ensure_canonical_liveness_offset();
    let dispatch_jc = __dispatch_jitcode_dispatch_minimal(&mut asm, 0i64)
        .expect("dispatch lower must succeed for fixture");

    driver.register_dispatch_jitcode(dispatch_jc);

    let stored = driver.dispatch_jitcode();
    assert!(
        stored.is_some(),
        "register_dispatch_jitcode_stores_singleton: dispatch_jitcode must be Some after register"
    );
    // Stored as Arc — ensure no panic on access
    let _arc: &Arc<JitCode> = stored.unwrap();
}

/// A.2.1 fixture: minimal `#[jit_interp]` dispatch loop carrying both the
/// `state.last_instr = pc as i64` store (`pyopcode.py:172`) AND the two-byte
/// `[opcode][oparg]` fetch with `pc += 2` (`pyopcode.py:179-181`).
///
/// A.2.1 only asserts the macro expands and the dispatch JitCode body builds
/// non-empty. A.2.2-A.2.4 add lower-side recognition for the new surfaces
/// (oparg fetch, last_instr store position) and pin shape via dedicated tests.
mod oparg_minimal {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct OpargState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;
    /// Backward-jump opcode whose arm contains `can_enter_jit!()` to mirror
    /// RPython's `interp_jit.py:118 pypyjitdriver.can_enter_jit(...)` inside
    /// `jump_absolute()`'s backward-branch path (`interp_jit.py:104 if jumpto
    /// >= next_instr: return jumpto` early-out).  The arm exists so the
    /// dispatch JitCode lowerer emits a `BC_LOOP_HEADER` for at least one
    /// arm — `lower_dispatch_chain`'s per-arm gating suppresses
    /// `loop_header` for forward-progress arms (OP_NOP, OP_ADD_I) that have
    /// no `can_enter_jit!` marker, matching `jtransform.py:1714-1723
    /// handle_jit_marker__loop_header` which only runs at user-placed
    /// `can_enter_jit` source sites.
    const OP_JUMP_BACK: u8 = 2;

    /// A.2.5 helper: stand-in for RPython
    /// `executioncontext.py:174 ec.bytecode_only_trace(self)` — the
    /// JIT-branch hook fired between `last_instr` store and opcode fetch.
    /// Pyre's `#[majit_macros::dont_look_inside]` attribute lowers a
    /// statement-form free-fn call to `BC_RESIDUAL_CALL_R_V` with
    /// `DEFAULT_EFFECT_INFO` (saturated read/write descrs +
    /// `EF_CAN_RAISE` per `assembler.rs:1652-1671 +
    /// call_descr.rs::DEFAULT_EFFECT_INFO`), matching plan v1 codex
    /// BLOCKER 2's required effect class for a side-effecting +
    /// can-raise hook.
    ///
    /// No args → `assembler.rs:1748-1769` auto-selection picks the R
    /// variant (no float / no int args → `BC_RESIDUAL_CALL_R_V`); the
    /// ref-list count byte is `0`. Pre-A.2.5 codex review BLOCKER 3
    /// (Ref-binding requirement) is satisfied by the no-args choice
    /// rather than threading a Ref binding through the dispatch loop.
    #[majit_macros::dont_look_inside]
    fn bytecode_only_trace_helper() {}

    #[majit_macros::jit_interp(
        state = OpargState,
        env = Bytecode,
        auto_calls = true,
        state_fields = { last_instr: int, acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_minimal(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OpargState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        // `can_enter_jit!()` rewrites to a runtime call that updates
        // `stacksize` after a back_edge close (mod.rs:1531
        // `#stacksize_expr = 0i32;`).  All examples that wire
        // `can_enter_jit!()` declare `let mut stacksize: i32 = 0;` in the
        // function scope; the OP_JUMP_BACK arm below requires the same.
        let mut stacksize: i32 = 0;
        let mut state = OpargState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:174  if jit.we_are_jitted():
            //                      ec.bytecode_only_trace(self)
            // (we_are_jitted branch only — the trace-fn version flows
            // through `#[dont_look_inside]` which pyre lowers as a
            // residual_call_void.)
            bytecode_only_trace_helper();
            // pyopcode.py:179  opcode = ord(co_code[next_instr])
            let opcode = program[pc];
            // pyopcode.py:180  oparg = ord(co_code[next_instr + 1])
            let oparg = program[pc + 1];
            // pyopcode.py:181  next_instr += 2
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                // Backward-jump arm — `can_enter_jit!()` at the call
                // site so `Lowerer::lower_stmt`'s `Stmt::Macro` arm
                // emits `BC_LOOP_HEADER` INSIDE this arm's sub-JitCode
                // (jtransform.py:1714-1723 handle_jit_marker__loop_header
                // at the source-level can_enter_jit position).  The
                // `if target < pc { ... }` conditional + `continue` shape
                // used by production consumers (interp_jit.py:104
                // `if jumpto >= next_instr: return jumpto` early-out
                // mirror) is kept on the runtime-side path via
                // `transform_function`'s `rewrite_body` substitution
                // (`mod.rs:1535 driver.back_edge_structured(...)`); the
                // dispatch JitCode lowerer cannot yet lower `if`+`continue`
                // inside arm bodies (multi-session lowerer epic), so the
                // fixture body is reduced to the macro call alone — that
                // is the smallest shape that exercises the call-site LH
                // emit per RPython parity.
                OP_JUMP_BACK => {
                    can_enter_jit!(driver, pc, &mut state, program, || {});
                }
                _ => break,
            }
        }
        state.acc
    }

    #[test]
    fn dispatch_oparg_minimal_builds_jitcode() {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        assert!(
            !dispatch_jc.code.is_empty(),
            "A.2.1: dispatch JitCode body must be non-empty for the oparg+last_instr fixture"
        );
    }

    /// A.2.2: the dispatch JitCode body must lower BOTH byte fetches in
    /// `let opcode = program[pc]; let oparg = program[pc + 1]; pc += 2;`
    /// per RPython `pyopcode.py:179-181`:
    ///   - opcode fetch → `BC_GETARRAYITEM_GC_I result, program, pc`
    ///   - oparg  fetch → `BC_INT_ADD offset, pc, +1` then
    ///                    `BC_GETARRAYITEM_GC_I result, program, offset`
    ///   - pc advance   → `BC_INT_ADD pc, pc, +2`
    ///
    /// Net post-merge-point op shape: BC_GETARRAYITEM_GC_I × 2 and at
    /// least two BC_INT_ADDs (one for `pc + 1` offset, one for `pc += 2`).
    #[test]
    fn dispatch_oparg_minimal_lowers_oparg_fetch_and_pc_advance() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I, BC_INT_ADD, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        let getarrayitem_count = post_mp
            .iter()
            .filter(|&&b| b == BC_GETARRAYITEM_GC_I)
            .count();
        assert_eq!(
            getarrayitem_count, 2,
            "A.2.2: dispatch JitCode body must emit exactly two BC_GETARRAYITEM_GC_I \
             ops (opcode + oparg per pyopcode.py:179-180); got {}",
            getarrayitem_count
        );

        let int_add_count = post_mp.iter().filter(|&&b| b == BC_INT_ADD).count();
        assert!(
            int_add_count >= 2,
            "A.2.2: dispatch JitCode body must emit ≥ 2 BC_INT_ADD ops \
             (one for pc+1 offset, one for pc += 2 advance); got {}",
            int_add_count
        );

        // Order check: the FIRST BC_GETARRAYITEM_GC_I (opcode fetch) must
        // precede the first BC_INT_ADD (which is the pc+1 offset compute
        // for the oparg fetch). The SECOND BC_GETARRAYITEM_GC_I (oparg
        // fetch) must come after the offset compute. RPython
        // `pyopcode.py:179-180` orders opcode → oparg.
        let first_getarr = post_mp
            .iter()
            .position(|&b| b == BC_GETARRAYITEM_GC_I)
            .expect("first BC_GETARRAYITEM_GC_I missing");
        let first_int_add = post_mp
            .iter()
            .position(|&b| b == BC_INT_ADD)
            .expect("first BC_INT_ADD missing");
        let second_getarr = post_mp
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == BC_GETARRAYITEM_GC_I)
            .nth(1)
            .map(|(i, _)| i)
            .expect("second BC_GETARRAYITEM_GC_I missing");
        assert!(
            first_getarr < first_int_add,
            "A.2.2: opcode fetch must precede pc+1 offset compute; \
             first_getarr={} first_int_add={}",
            first_getarr,
            first_int_add
        );
        assert!(
            first_int_add < second_getarr,
            "A.2.2: pc+1 offset compute must precede oparg fetch; \
             first_int_add={} second_getarr={}",
            first_int_add,
            second_getarr
        );
    }

    /// A.2.4: the `state.last_instr = pc as i64` write
    /// (`pyopcode.py:172  self.last_instr = intmask(next_instr)`) must
    /// lower to a single `BC_STORE_STATE_FIELD` op emitted in the
    /// prelude position — between the JIT merge-point markers and the
    /// opcode fetch. The existing `state_fields` macro lowering
    /// (`majit-macros/src/jit_interp/jitcode_lower.rs:1429`,
    /// `assembler.rs:345`, `dispatch.rs:1233`) already emits the op;
    /// A.2.4 reduces to a verification slice that pins position and
    /// count so a future refactor cannot silently drop the store or
    /// reorder it across the opcode fetch.
    #[test]
    fn dispatch_oparg_minimal_pins_last_instr_store_position() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_STORE_STATE_FIELD,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        let store_count = post_mp
            .iter()
            .filter(|&&b| b == BC_STORE_STATE_FIELD)
            .count();
        assert_eq!(
            store_count, 1,
            "A.2.4: dispatch JitCode body must emit exactly one \
             BC_STORE_STATE_FIELD for `state.last_instr = pc as i64` \
             (pyopcode.py:172); got {}",
            store_count
        );

        let store_pos = post_mp
            .iter()
            .position(|&b| b == BC_STORE_STATE_FIELD)
            .expect("BC_STORE_STATE_FIELD missing");
        let first_getarr = post_mp
            .iter()
            .position(|&b| b == BC_GETARRAYITEM_GC_I)
            .expect("BC_GETARRAYITEM_GC_I missing");
        assert!(
            store_pos < first_getarr,
            "A.2.4: last_instr store must precede opcode fetch \
             (pyopcode.py:172 before L179); store_pos={} first_getarr={}",
            store_pos,
            first_getarr
        );
    }

    /// A.2.5.b: the `bytecode_only_trace_helper()` call (RPython
    /// `pyopcode.py:174 ec.bytecode_only_trace(self)` JIT branch) must
    /// lower to `BC_RESIDUAL_CALL_R_V` per `assembler.rs:1748-1769`
    /// auto-selection (no float / no int args → R variant with
    /// `ref_count = 0`). Position: between the `BC_STORE_STATE_FIELD`
    /// for `state.last_instr` (L172) and the first `BC_GETARRAYITEM_GC_I`
    /// for the opcode fetch (L179) — RPython source order is
    /// L172 store → L174 trace → L179 fetch.
    ///
    /// Codex Pre-A.2.5 review BLOCKER 2 absorbed: canonical opcode is
    /// `BC_RESIDUAL_CALL_R_V` (159), not the legacy `BC_RESIDUAL_CALL_VOID_R`
    /// named in plan v1 (retired). BLOCKER 3 absorbed: the no-args
    /// helper picks the R variant directly without requiring a Ref
    /// binding to be threaded through the dispatch loop.
    ///
    /// Codex BLOCKER 4 (optimizer-no-hoist verification) is deferred
    /// to A.2.5.c — this test only pins the dispatch JitCode body
    /// shape, which is the byte stream the install gate sees, not the
    /// optimized loop IR. A future epic can add an optimized-IR
    /// observability harness and assert the call survives across two
    /// consecutive iterations without dedup or hoisting.
    #[test]
    fn dispatch_oparg_minimal_pins_bytecode_only_trace_residual_call() {
        use majit_metainterp::jitcode::insns::{
            BC_GETARRAYITEM_GC_I, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_RESIDUAL_CALL_R_V,
            BC_STORE_STATE_FIELD,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        // Counts use ≥ rather than == because the byte stream interleaves
        // opcode bytes with operand bytes; high-byte opcode values
        // (BC_RESIDUAL_CALL_R_V = 159) can collide with descr-index or
        // const-pool operand bytes in the surrounding ops. Strict
        // positional ordering via the first occurrence after each prior
        // landmark is the robust signal.
        let residual_count = post_mp
            .iter()
            .filter(|&&b| b == BC_RESIDUAL_CALL_R_V)
            .count();
        assert!(
            residual_count >= 1,
            "A.2.5.b: dispatch JitCode body must emit ≥ 1 BC_RESIDUAL_CALL_R_V \
             for the `#[dont_look_inside]`-annotated `bytecode_only_trace_helper()` \
             call (pyopcode.py:174); got {}",
            residual_count
        );

        let store_pos = post_mp
            .iter()
            .position(|&b| b == BC_STORE_STATE_FIELD)
            .expect("BC_STORE_STATE_FIELD missing");
        let residual_pos = post_mp
            .iter()
            .enumerate()
            .skip(store_pos + 1)
            .find(|&(_, &b)| b == BC_RESIDUAL_CALL_R_V)
            .map(|(i, _)| i)
            .expect("BC_RESIDUAL_CALL_R_V after BC_STORE_STATE_FIELD missing");
        let first_getarr = post_mp
            .iter()
            .enumerate()
            .skip(residual_pos + 1)
            .find(|&(_, &b)| b == BC_GETARRAYITEM_GC_I)
            .map(|(i, _)| i)
            .expect("BC_GETARRAYITEM_GC_I after BC_RESIDUAL_CALL_R_V missing");
        // Successful chained `find_after` proves the ordering:
        // BC_STORE_STATE_FIELD → BC_RESIDUAL_CALL_R_V → BC_GETARRAYITEM_GC_I,
        // matching pyopcode.py:172 → L174 → L179 source order.
        let _ = (store_pos, residual_pos, first_getarr);
    }

    /// A.3.3: the dispatch JitCode for `dispatch_oparg_minimal` (`greens = []`)
    /// must encode empty greens and real reds derived from portal inputs minus greens.
    ///
    /// Candidates: `["program", "pc"]`.  Greens empty → reds = both.
    /// `pc` is Int at i0 → reds_i = [0]; `program` is Ref at r0 → reds_r = [0].
    ///
    /// Full payload at offset +2 from the opcode byte (greens_base = mp_pos + 2):
    ///   greens_base + 0: greens_i_len = 0
    ///   greens_base + 1: greens_r_len = 0
    ///   greens_base + 2: greens_f_len = 0
    ///   greens_base + 3: reds_i_len   = 1  (pc → Int i0)
    ///   greens_base + 4: reds_i[0]    = 0
    ///   greens_base + 5: reds_r_len   = 1  (program → Ref r0)
    ///   greens_base + 6: reds_r[0]    = 0
    ///   greens_base + 7: reds_f_len   = 0
    #[test]
    fn dispatch_oparg_minimal_pins_real_reds_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.3: BC_JIT_MERGE_POINT(_C) must be present in dispatch JitCode");

        // Compact or non-compact: jdindex occupies 1 byte immediately after
        // the opcode byte, so the greens/reds payload begins at mp_pos + 2.
        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.3: payload too short to decode greens+reds; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        // Greens: all empty (greens = []).
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 0,
            "A.3.3: greens_i_len must be 0 (no greens declared); got {}",
            greens_i_len
        );
        let greens_r_len = code[greens_base + 1] as usize;
        assert_eq!(
            greens_r_len, 0,
            "A.3.3: greens_r_len must be 0 (no greens declared); got {}",
            greens_r_len
        );
        let greens_f_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.3: greens_f_len must be 0 (no greens declared); got {}",
            greens_f_len
        );

        // Reds: pc (Int/i0) then program (Ref/r0).
        let reds_i_len = code[greens_base + 3] as usize;
        assert_eq!(
            reds_i_len, 1,
            "A.3.3: reds_i_len must be 1 (pc → Int i0); got {}",
            reds_i_len
        );
        let reds_i_byte = code[greens_base + 4];
        assert_eq!(
            reds_i_byte, 0,
            "A.3.3: reds_i[0] must be 0 (pc at register i0); got {}",
            reds_i_byte
        );

        let reds_r_len = code[greens_base + 5] as usize;
        assert_eq!(
            reds_r_len, 1,
            "A.3.3: reds_r_len must be 1 (program → Ref r0); got {}",
            reds_r_len
        );
        let reds_r_byte = code[greens_base + 6];
        assert_eq!(
            reds_r_byte, 0,
            "A.3.3: reds_r[0] must be 0 (program at register r0); got {}",
            reds_r_byte
        );

        let reds_f_len = code[greens_base + 7] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3: reds_f_len must be 0 (no float reds); got {}",
            reds_f_len
        );
    }

    /// A.3.4: `BC_LOOP_HEADER` jdindex wire — call-site emission inside
    /// the OP_JUMP_BACK arm sub-JitCode.
    ///
    /// `jtransform.py:1714-1723` `handle_jit_marker__loop_header` emits
    /// `SpaceOperation('loop_header', [c_index], None)` with `c_index =
    /// Constant(jd.index, lltype.Signed)` AT THE SOURCE-LEVEL CALL SITE —
    /// in PyPy that is `interp_jit.py:118 pypyjitdriver.can_enter_jit(...)`
    /// inside `jump_absolute()`'s BACKWARD-JUMP BRANCH ONLY (the forward
    /// path early-returns at `interp_jit.py:104 if jumpto >= next_instr:
    /// return jumpto`).  Pyre's parity is achieved by recognising
    /// `can_enter_jit!(...)` as a `Stmt::Macro` inside `Lowerer::lower_stmt`
    /// (`jitcode_lower.rs:1970-2008`) and emitting the LH op INSIDE the
    /// arm body sub-JitCode at that exact stmt position.  No post-INLINE_CALL
    /// emission appears at the dispatch-JitCode level — that would over-emit
    /// on every arm execution including forward-progress arms.
    ///
    /// The test pins:
    ///   1. `BC_LOOP_HEADER` appears in the OP_JUMP_BACK arm sub-JitCode
    ///      (the one whose body contains `can_enter_jit!()`); OP_NOP and
    ///      OP_ADD_I arms must NOT emit it.
    ///   2. The byte immediately after the LH opcode is in valid const-pool
    ///      range and stores the jdindex constant (`0i64` for this invocation).
    ///   3. ZERO `BC_LOOP_HEADER` opcodes appear in the dispatch JitCode
    ///      body itself — strict parity with PyPy where the LH lives at
    ///      the user's source-level `can_enter_jit()` call site, NOT at
    ///      arm boundaries.
    #[test]
    fn dispatch_oparg_minimal_pins_loop_header_jdindex() {
        use majit_metainterp::BC_GOTO;
        use majit_metainterp::jitcode::insns::BC_LOOP_HEADER;

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let dispatch_code = &dispatch_jc.code;

        // Step 3 (negative parity at dispatch level): the dispatch JitCode
        // body must NOT contain any [BC_LOOP_HEADER, slot_in_pool, BC_GOTO]
        // 3-byte triple — the previous per-arm-back-edge emit always
        // produced this exact triple at every arm's loop-back location.
        // Anchoring on the trailing BC_GOTO disambiguates payload-byte
        // coincidences (a const-pool slot byte that happens to equal
        // BC_LOOP_HEADER is not a real op start).
        let dispatch_num_regs_i = dispatch_jc.c_num_regs_i as usize;
        let dispatch_const_start = dispatch_num_regs_i;
        let dispatch_const_end = dispatch_num_regs_i + dispatch_jc.constants_i.len();
        let dispatch_lh_at_back_edge_count = (0..dispatch_code.len().saturating_sub(2))
            .filter(|&i| {
                dispatch_code[i] == BC_LOOP_HEADER
                    && (dispatch_code[i + 1] as usize) >= dispatch_const_start
                    && (dispatch_code[i + 1] as usize) < dispatch_const_end
                    && dispatch_code[i + 2] == BC_GOTO
            })
            .count();
        assert_eq!(
            dispatch_lh_at_back_edge_count, 0,
            "A.3.4: dispatch JitCode body must contain ZERO \
             [BC_LOOP_HEADER, slot, BC_GOTO] triples — LH lives at the \
             source-level can_enter_jit!() call site INSIDE the arm body \
             sub-JitCode, mirroring `interp_jit.py:118` placement inside \
             `jump_absolute()`'s backward-branch only.  Got {} occurrences.",
            dispatch_lh_at_back_edge_count
        );

        // Step 1 (positive parity): EXACTLY ONE arm sub-JitCode contains
        // exactly one BC_LOOP_HEADER op (the OP_JUMP_BACK arm whose body
        // is `can_enter_jit!(...)`).  Forward-progress arms (OP_NOP /
        // OP_ADD_I) emit no LH.
        let arm_lh_emitting: Vec<_> = dispatch_jc
            .exec
            .descrs
            .iter()
            .filter_map(|descr| descr.as_jitcode())
            .filter(|sub| {
                let snr = sub.c_num_regs_i as usize;
                let cstart = snr;
                let cend = snr + sub.constants_i.len();
                (0..sub.code.len().saturating_sub(1)).any(|i| {
                    sub.code[i] == BC_LOOP_HEADER
                        && (sub.code[i + 1] as usize) >= cstart
                        && (sub.code[i + 1] as usize) < cend
                })
            })
            .collect();
        assert_eq!(
            arm_lh_emitting.len(),
            1,
            "A.3.4: exactly one arm sub-JitCode must emit BC_LOOP_HEADER \
             (the OP_JUMP_BACK arm carrying can_enter_jit!()); got {}",
            arm_lh_emitting.len()
        );
        let jump_back_jc = arm_lh_emitting[0];

        // Locate BC_LOOP_HEADER inside the OP_JUMP_BACK sub-JitCode and
        // verify the slot byte addresses the const-pool entry that
        // stores the jdindex value.  Anchor on slot-in-const-pool to
        // disambiguate payload coincidences inside the sub-JitCode.
        let sub_num_regs_i = jump_back_jc.c_num_regs_i as usize;
        let sub_const_start = sub_num_regs_i;
        let sub_const_end = sub_num_regs_i + jump_back_jc.constants_i.len();
        let sub_lh_pos = (0..jump_back_jc.code.len().saturating_sub(1))
            .find(|&i| {
                jump_back_jc.code[i] == BC_LOOP_HEADER
                    && (jump_back_jc.code[i + 1] as usize) >= sub_const_start
                    && (jump_back_jc.code[i + 1] as usize) < sub_const_end
            })
            .expect(
                "A.3.4: BC_LOOP_HEADER must appear in OP_JUMP_BACK arm sub-JitCode \
                 (lower_stmt Stmt::Macro recognition emit-site)",
            );

        // Step 2: the byte immediately after BC_LOOP_HEADER is the const-pool
        // slot byte; valid range already enforced by the anchor.
        let sub_slot_byte = jump_back_jc.code[sub_lh_pos + 1] as usize;

        // Step 2 (cont.): the constant value at that slot equals jdindex (0i64).
        let sub_const_idx = sub_slot_byte - sub_num_regs_i;
        let stored_jdindex = jump_back_jc.constants_i[sub_const_idx];
        assert_eq!(
            stored_jdindex, 0i64,
            "A.3.4: BC_LOOP_HEADER const-pool slot must store jdindex=0; got {}",
            stored_jdindex
        );

        let _ = (
            dispatch_lh_at_back_edge_count,
            sub_lh_pos,
            sub_slot_byte,
            sub_const_idx,
            stored_jdindex,
        );
    }

    /// A.3.5 (negative): with `greens = []`, NO `BC_*_GUARD_VALUE` op must
    /// appear in the prefix before `BC_JIT_MERGE_POINT`.
    ///
    /// Mirrors `jtransform.py:1693-1714 promote_greens`: the loop over
    /// `args[:num_green_args]` is empty when `num_green_args == 0`, so no
    /// guard ops are emitted.
    #[test]
    fn dispatch_oparg_minimal_no_promote_when_greens_empty() {
        use majit_metainterp::jitcode::insns::{
            BC_FLOAT_GUARD_VALUE, BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C,
            BC_REF_GUARD_VALUE,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_minimal(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_minimal(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("BC_JIT_MERGE_POINT(_C) must be present");

        let prefix = &code[..mp_pos];
        assert!(
            !prefix.contains(&BC_INT_GUARD_VALUE)
                && !prefix.contains(&BC_REF_GUARD_VALUE)
                && !prefix.contains(&BC_FLOAT_GUARD_VALUE),
            "A.3.5 (jtransform.py:1693-1714): empty greens must not emit any \
             *_guard_value before BC_JIT_MERGE_POINT; prefix={:?}",
            prefix
        );
    }
}

/// A.2.3a fixture: dispatch loop carrying the EXTENDED_ARG inner while
/// shape per RPython `pyopcode.py:187-193`. The fixture's structural
/// contents (inner `while opcode == EXTENDED_ARG`, multi-byte oparg
/// merge, HAVE_ARGUMENT corruption guard) are NOT lowered to dispatch
/// JitCode IR yet — A.2.3a is recognition-only. The fail-closed install
/// gate added in A.2.3a guarantees that lowering FAILS (returns `None`,
/// dispatch body empty, gate refuses install) when the inner while
/// shape cannot be recognized; the recognizer accepts this fixture's
/// `while opcode == EXTENDED_ARG` form, so the test asserts the body
/// is non-empty.
mod oparg_extended {
    use crate::BytecodeExt;
    use majit_metainterp::jitcode::insns::{
        BC_ABORT, BC_GETARRAYITEM_GC_I, BC_INT_MUL, BC_INT_OR, BC_JIT_MERGE_POINT,
        BC_JIT_MERGE_POINT_C,
    };
    use majit_metainterp::{Assembler, JitDriver};

    struct OpargExtState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;
    const EXTENDED_ARG: u8 = 254;
    const HAVE_ARGUMENT: u8 = 90;

    #[majit_macros::jit_interp(
        state = OpargExtState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [],
    )]
    #[allow(unused_assignments, unused_variables, unused_mut)]
    fn dispatch_oparg_extended(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<OpargExtState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = OpargExtState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:179-181  opcode/oparg fetch + pc += 2
            let mut opcode = program[pc];
            let mut oparg: i64 = program[pc + 1] as i64;
            pc += 2;
            // pyopcode.py:187-193  EXTENDED_ARG inner loop. Its body is
            // NOT lowered in A.2.3a — A.2.3b ports the multi-byte oparg
            // merge (`(oparg * 256) | arg2`) to BC_INT_MUL + BC_INT_OR
            // and the HAVE_ARGUMENT guard's `raise BytecodeCorruption`
            // (RPython L190-191) to a BC_ABORT bailout label per
            // Pre-A.2.3 codex review BLOCKERs (c) + (d).
            while opcode == EXTENDED_ARG {
                let opcode2 = program[pc];
                let arg2 = program[pc + 1];
                if opcode2 < HAVE_ARGUMENT {
                    panic!("BytecodeCorruption parity bailout (A.2.3b lowers to BC_ABORT)");
                }
                pc += 2;
                oparg = (oparg * 256) | arg2 as i64;
                opcode = opcode2;
            }
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg,
                _ => break,
            }
        }
        state.acc
    }

    /// A.2.3a positive test: the EXTENDED_ARG inner-while recognizer
    /// in `lower_pre_dispatch_stmts` accepts `while opcode ==
    /// EXTENDED_ARG`, so `lower_dispatch_body` returns `Some(body)`
    /// and the dispatch JitCode body is non-empty (and contains the
    /// A.2.2 opcode/oparg fetch + pc advance ops).
    #[test]
    fn dispatch_oparg_extended_builds_jitcode() {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_extended(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_extended(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        assert!(
            !dispatch_jc.code.is_empty(),
            "A.2.3a: dispatch JitCode body must be non-empty when the \
             inner-while shape `while opcode == EXTENDED_ARG` is \
             recognized — gate refuses install (empty body) only when \
             the shape FAILS to match"
        );
    }

    /// A.2.3b shape pin: lowered EXTENDED_ARG inner while emits the
    /// RPython L188-193 IR sequence:
    ///
    /// - 4 × BC_GETARRAYITEM_GC_I — outer opcode + outer oparg
    ///   (pyopcode.py:179-180) plus inner opcode2 + inner arg2
    ///   (pyopcode.py:188-189)
    /// - 1 × BC_INT_MUL — `oparg * 256` (Pre-A.2.3 codex BLOCKER (c):
    ///   pyre lacks `BC_INT_LSHIFT_IMM`, jtransform.py:363-366 keeps
    ///   int_mul symmetric, so the line-by-line port is INT_MUL)
    /// - 1 × BC_INT_OR — `(oparg * 256) | arg`
    /// - 1 × BC_ABORT — corruption-bailout for `opcode < HAVE_ARGUMENT`
    ///   (Pre-A.2.3 codex BLOCKER (d): RPython L190-191
    ///   `raise BytecodeCorruption` ports to the canonical local bailout
    ///   `BC_ABORT`, not `guard_value`)
    ///
    /// Ordering is also pinned: outer fetch precedes inner fetch
    /// precedes int_mul precedes int_or precedes abort. The inner
    /// `let opcode2 = program[pc]` aliases to the outer `opcode`
    /// register per RPython L188 (`opcode = ord(co_code[next_instr])`
    /// reuses the outer opcode slot), so the back-edge re-tests the
    /// freshly-fetched value without an extra `int_or` copy.
    #[test]
    fn dispatch_oparg_extended_pins_inner_while_ir_shape() {
        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_extended(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_extended(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;
        let mp_idx = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("missing JIT_MERGE_POINT");
        let post_mp = &code[mp_idx..];

        // Count assertions are intentionally positional: the JitCode
        // byte stream interleaves opcode bytes with operand bytes, so
        // a naive byte filter (.filter(|&&b| b == BC_X).count()) can
        // collide with operand values that happen to equal BC_X. The
        // existing `dispatch_oparg_minimal_lowers_oparg_fetch_and_pc_advance`
        // test gets away with `== 2` for BC_GETARRAYITEM_GC_I (0xa9)
        // because the fixture uses a small register window; this richer
        // EXTENDED_ARG fixture has more registers and high-byte descr
        // indices, so we assert minimums + ordering instead.
        let count = |target: u8| post_mp.iter().filter(|&&b| b == target).count();
        let getarr_count = count(BC_GETARRAYITEM_GC_I);
        assert!(
            getarr_count >= 4,
            "A.2.3b: dispatch JitCode body must emit ≥ 4 BC_GETARRAYITEM_GC_I \
             ops (outer opcode + outer oparg + inner opcode2 + inner arg2 \
             per pyopcode.py:179-180/188-189); got {}",
            getarr_count
        );
        assert!(
            count(BC_INT_MUL) >= 1,
            "A.2.3b: oparg merge `oparg * 256` must lower to ≥ 1 BC_INT_MUL \
             (pyre has no BC_INT_LSHIFT_IMM; jtransform.py:363-366 keeps \
             int_mul symmetric per Pre-A.2.3 codex BLOCKER (c))"
        );
        assert!(
            count(BC_INT_OR) >= 1,
            "A.2.3b: oparg merge `(oparg * 256) | arg` must lower to ≥ 1 \
             BC_INT_OR; opcode2 is aliased to the outer opcode register so \
             no second `int_or` copy is needed"
        );
        assert!(
            count(BC_ABORT) >= 1,
            "A.2.3b: HAVE_ARGUMENT corruption guard must lower `raise \
             BytecodeCorruption` to ≥ 1 BC_ABORT (Pre-A.2.3 codex BLOCKER \
             (d) — guard_value is wrong shape for a range check)"
        );

        // Ordering checks against FIRST occurrences in the byte stream.
        // The lowerer emits ops in source order
        // (pyopcode.py:179-180 outer fetches → L188-189 inner fetches →
        // L190 corruption abort → L191 next_instr += 2 → L193 oparg
        // merge). Operand-byte aliasing CAN spuriously pull a "first
        // occurrence" of BC_ABORT etc. before it is actually emitted —
        // but only if the operand happens to land before the real op,
        // which would also collapse the ordering chain into an
        // inconsistent prefix. Treating "first occurrence" as a strict
        // upper bound keeps the assertion conservative.
        let first_pos = |target: u8, after: usize| -> usize {
            post_mp
                .iter()
                .enumerate()
                .skip(after)
                .find(|&(_, &b)| b == target)
                .map(|(i, _)| i)
                .unwrap_or_else(|| panic!("BC_{:02x} after offset {} missing", target, after))
        };
        let outer_opcode_pos = first_pos(BC_GETARRAYITEM_GC_I, 0);
        let outer_oparg_pos = first_pos(BC_GETARRAYITEM_GC_I, outer_opcode_pos + 1);
        let inner_opcode_pos = first_pos(BC_GETARRAYITEM_GC_I, outer_oparg_pos + 1);
        let inner_arg_pos = first_pos(BC_GETARRAYITEM_GC_I, inner_opcode_pos + 1);
        let abort_pos = first_pos(BC_ABORT, inner_arg_pos + 1);
        let int_mul_pos = first_pos(BC_INT_MUL, abort_pos + 1);
        let int_or_pos = first_pos(BC_INT_OR, int_mul_pos + 1);
        // Pinning successive `find_after` succeeded → the ops appear in
        // exactly this order: outer opcode → outer oparg → inner
        // opcode2 → inner arg2 → corruption abort → oparg-merge int_mul
        // → oparg-merge int_or. RPython `pyopcode.py:179-193`.
        let _ = (
            outer_opcode_pos,
            outer_oparg_pos,
            inner_opcode_pos,
            inner_arg_pos,
            abort_pos,
            int_mul_pos,
            int_or_pos,
        );
    }
}

/// A.3.2 fixture: minimal dispatch loop with `greens = [pc]` declared.
///
/// Mirrors `dispatch_oparg_minimal` but adds `pc` to the green list so
/// `resolve_greens` must resolve the `pc` ident to register `i0` and emit
/// `greens_i = [0]` in the `jit_merge_point` payload.
mod oparg_with_pc_green {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct PcGreenState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = PcGreenState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [pc],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_with_pc_green(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<PcGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = PcGreenState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            // pyopcode.py:172  self.last_instr = intmask(next_instr)
            state.last_instr = pc as i64;
            // pyopcode.py:179  opcode = ord(co_code[next_instr])
            let opcode = program[pc];
            // pyopcode.py:180  oparg = ord(co_code[next_instr + 1])
            let oparg = program[pc + 1];
            // pyopcode.py:181  next_instr += 2
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                _ => break,
            }
        }
        state.acc
    }

    /// A.3.2/A.3.3: the dispatch JitCode for `dispatch_oparg_with_pc_green` must
    /// encode `greens_i = [0]` (pc → Int register i0), `greens_r = []`,
    /// `greens_f = []`, then `reds_i = []` (pc filtered as green),
    /// `reds_r = [0]` (program → Ref r0), `reds_f = []`.
    ///
    /// Payload layout after the opcode + jdindex byte (assembler.py encoding):
    ///   [greens_i_len][greens_i bytes...][greens_r_len][greens_r bytes...]
    ///   [greens_f_len][greens_f bytes...][reds_i_len][reds_i bytes...]
    ///   [reds_r_len][reds_r bytes...][reds_f_len]
    ///
    /// For `jdindex = 0i64` the compact form `BC_JIT_MERGE_POINT_C` is used
    /// (jdindex fits in -128..=127), followed by one signed-byte jdindex.
    /// Full payload at offset +2 from the opcode byte:
    ///   offset+2: greens_i_len = 1
    ///   offset+3: greens_i[0]  = 0  (register i0 = pc)
    ///   offset+4: greens_r_len = 0
    ///   offset+5: greens_f_len = 0
    ///   offset+6: reds_i_len   = 0  (pc filtered as green)
    ///   offset+7: reds_r_len   = 1  (program → Ref r0)
    ///   offset+8: reds_r[0]    = 0
    ///   offset+9: reds_f_len   = 0
    #[test]
    fn dispatch_oparg_with_pc_green_pins_real_greens_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pc_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pc_green(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.2: BC_JIT_MERGE_POINT(_C) must be present in dispatch JitCode");

        let is_compact = code[mp_pos] == BC_JIT_MERGE_POINT_C;
        // Compact form: opcode(1) + jdindex_byte(1) + payload.
        // Non-compact: opcode(1) + const_pool_offset_byte(1) + payload.
        // Either way the jdindex occupies exactly 1 byte immediately after
        // the opcode, so the greens lists begin at mp_pos + 2.
        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.2: payload too short to decode greens+reds; is_compact={}, mp_pos={}, len={}",
            is_compact,
            mp_pos,
            code.len()
        );

        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.2: greens_i_len must be 1 (pc is Int → i0); got {}",
            greens_i_len
        );

        let greens_i_byte = code[greens_base + 1];
        assert_eq!(
            greens_i_byte, 0,
            "A.3.2: greens_i[0] must be 0 (register i0 = pc); got {}",
            greens_i_byte
        );

        let greens_r_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_r_len, 0,
            "A.3.2: greens_r_len must be 0 (pc is Int, not Ref); got {}",
            greens_r_len
        );

        let greens_f_len = code[greens_base + 3] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.2: greens_f_len must be 0 (pc is Int, not Float); got {}",
            greens_f_len
        );

        // A.3.3 reds layout: greens = [pc], so pc is filtered out of reds.
        // Remaining candidates: ["program"] (Ref at r0).
        //
        // Byte layout continues directly after greens_f:
        //   greens_base + 4: reds_i_len  = 0  (pc filtered out — it's green)
        //   greens_base + 5: reds_r_len  = 1  (program → Ref r0)
        //   greens_base + 6: reds_r[0]   = 0
        //   greens_base + 7: reds_f_len  = 0

        let reds_i_len = code[greens_base + 4] as usize;
        assert_eq!(
            reds_i_len, 0,
            "A.3.3: reds_i_len must be 0 (pc filtered as green); got {}",
            reds_i_len
        );

        let reds_r_len = code[greens_base + 5] as usize;
        assert_eq!(
            reds_r_len, 1,
            "A.3.3: reds_r_len must be 1 (program → Ref r0); got {}",
            reds_r_len
        );
        let reds_r_byte = code[greens_base + 6];
        assert_eq!(
            reds_r_byte, 0,
            "A.3.3: reds_r[0] must be 0 (program at register r0); got {}",
            reds_r_byte
        );

        let reds_f_len = code[greens_base + 7] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3: reds_f_len must be 0 (no float reds); got {}",
            reds_f_len
        );
    }

    /// A.3.5: with `greens = [pc]`, a `-live-` + `BC_INT_GUARD_VALUE` pair
    /// must appear BEFORE `BC_JIT_MERGE_POINT(_C)` in the dispatch JitCode.
    ///
    /// Mirrors `jtransform.py:1693-1714 promote_greens`: for each green
    /// Variable, emit SpaceOperation('-live-', ...) then
    /// SpaceOperation('int_guard_value', [v], None).  `pc` is Int kind →
    /// int_guard_value.  The `-live-` byte is 1 opcode + 2-byte offset
    /// (from `live_placeholder`), so BC_LIVE is at `igv_pos - 3`.
    #[test]
    fn dispatch_oparg_with_pc_green_promotes_green_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pc_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pc_green(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("BC_JIT_MERGE_POINT(_C) must be present");

        // jtransform.py:1693: pc is Int → int_guard_value must precede merge point.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (jtransform.py:1693): pc green must be promoted via \
                 BC_INT_GUARD_VALUE before BC_JIT_MERGE_POINT",
            );

        assert!(
            igv_pos < mp_pos,
            "BC_INT_GUARD_VALUE must precede BC_JIT_MERGE_POINT"
        );

        // jtransform.py:1707: -live- marker immediately precedes each guard_value.
        // BC_LIVE is 1-byte opcode + 2-byte offset, so the BC_LIVE byte is at
        // igv_pos - 3.
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at position {} has no room for preceding -live- (need >= 3 bytes before)",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede BC_INT_GUARD_VALUE \
             for green promotion; code[{}]={:#04x}, expected BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );
    }
}

/// A.3.3 optional parity-anchor fixture: `greens = [pc, program]`.
///
/// Mirrors `interp_jit.py:67 reds = ['frame', 'ec']` adapted to pyre's
/// portal binding names.  Declaring both portal inputs as green leaves
/// reds = [] — this is the A.6 follow-up parity shape.
///
/// Byte layout (greens_base = mp_pos + 2):
///   greens_base + 0: greens_i_len = 1  (pc → Int i0)
///   greens_base + 1: greens_i[0]  = 0
///   greens_base + 2: greens_r_len = 1  (program → Ref r0)
///   greens_base + 3: greens_r[0]  = 0
///   greens_base + 4: greens_f_len = 0
///   greens_base + 5: reds_i_len   = 0  (pc is green)
///   greens_base + 6: reds_r_len   = 0  (program is green)
///   greens_base + 7: reds_f_len   = 0
mod oparg_with_pypy_parity_greens {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct ParityState {
        last_instr: i64,
        acc: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = ParityState,
        env = Bytecode,
        state_fields = { last_instr: int, acc: int },
        greens = [pc, program],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_oparg_with_pypy_parity_greens(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<ParityState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = ParityState {
            last_instr: 0,
            acc: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            jit_merge_point!();
            state.last_instr = pc as i64;
            let opcode = program[pc];
            let oparg = program[pc + 1];
            pc += 2;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.acc += oparg as i64,
                _ => break,
            }
        }
        state.acc
    }

    /// A.3.3: `greens = [pc, program]` yields empty reds and both portal
    /// inputs in greens buckets.  This is the A.6 parity anchor for
    /// `interp_jit.py:67 reds = ['frame', 'ec']` adapted to pyre.
    #[test]
    fn dispatch_oparg_with_pypy_parity_greens_pins_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_oparg_with_pypy_parity_greens(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_oparg_with_pypy_parity_greens(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.3 parity: BC_JIT_MERGE_POINT(_C) must be present");

        let greens_base = mp_pos + 2;

        assert!(
            greens_base + 7 < code.len(),
            "A.3.3 parity: payload too short; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        // Greens: pc (Int/i0) then program (Ref/r0).
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.3 parity: greens_i_len must be 1 (pc → Int i0); got {}",
            greens_i_len
        );
        let greens_i_byte = code[greens_base + 1];
        assert_eq!(
            greens_i_byte, 0,
            "A.3.3 parity: greens_i[0] must be 0 (register i0 = pc); got {}",
            greens_i_byte
        );
        let greens_r_len = code[greens_base + 2] as usize;
        assert_eq!(
            greens_r_len, 1,
            "A.3.3 parity: greens_r_len must be 1 (program → Ref r0); got {}",
            greens_r_len
        );
        let greens_r_byte = code[greens_base + 3];
        assert_eq!(
            greens_r_byte, 0,
            "A.3.3 parity: greens_r[0] must be 0 (register r0 = program); got {}",
            greens_r_byte
        );
        let greens_f_len = code[greens_base + 4] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.3 parity: greens_f_len must be 0; got {}",
            greens_f_len
        );

        // Reds: all empty (both portal inputs declared as green).
        let reds_i_len = code[greens_base + 5] as usize;
        assert_eq!(
            reds_i_len, 0,
            "A.3.3 parity: reds_i_len must be 0 (pc is green); got {}",
            reds_i_len
        );
        let reds_r_len = code[greens_base + 6] as usize;
        assert_eq!(
            reds_r_len, 0,
            "A.3.3 parity: reds_r_len must be 0 (program is green); got {}",
            reds_r_len
        );
        let reds_f_len = code[greens_base + 7] as usize;
        assert_eq!(
            reds_f_len, 0,
            "A.3.3 parity: reds_f_len must be 0; got {}",
            reds_f_len
        );
    }
}

/// A.3.6.1: body-local green from a state-field comparison (`state.f1 <= state.f2`).
///
/// Pre-merge-point body-local lowering exercises `bind_pre_merge_point_stmts`:
/// the `let g = state.f1 <= state.f2;` stmt sits BEFORE `jit_merge_point!()`
/// and must be bound (via `Lowerer::lower_local`) so that
/// `resolve_greens` / `emit_promote_greens` can resolve `g` via
/// `lowerer.bindings`. Without the pre-pass `emit_promote_greens` strict-
/// `expect`s on the missing binding (A.3.5).
///
/// `lower_value_expr` already handles `BinOp::Le` on int state fields, so
/// no extension to the expression vocabulary is required by this fixture.
mod oparg_with_body_local_state_le_green {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct LeGreenState {
        f1: i64,
        f2: i64,
        a: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = LeGreenState,
        env = Bytecode,
        state_fields = { f1: int, f2: int, a: int },
        greens = [g],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_body_local_state_le_green(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<LeGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = LeGreenState { f1: 0, f2: 0, a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            let g = state.f1 <= state.f2;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            let oparg = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.a += oparg as i64,
                _ => break,
            }
            let _ = g;
        }
        state.a
    }

    /// A.3.6.1: the dispatch JitCode for `dispatch_with_body_local_state_le_green`
    /// must lower the body-local `let g = state.f1 <= state.f2;` BEFORE the
    /// portal `BC_JIT_MERGE_POINT(_C)` and emit A.3.5's promote_greens pair
    /// (`-live-` + `BC_INT_GUARD_VALUE`) for the body-local int reg holding `g`.
    /// The same reg byte must appear in the `greens_i` payload of the merge
    /// point op (A.3.2 layout).
    #[test]
    fn dispatch_with_body_local_state_le_green_promotes_body_local_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_INT_LE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_body_local_state_le_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc =
            __dispatch_jitcode_dispatch_with_body_local_state_le_green(&mut asm, 0i64)
                .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.6.1: BC_JIT_MERGE_POINT(_C) must be present");

        // BC_INT_LE for `state.f1 <= state.f2` must land BEFORE the merge point.
        let int_le_pos = code[..mp_pos].iter().position(|&b| b == BC_INT_LE).expect(
            "A.3.6.1: BC_INT_LE must precede BC_JIT_MERGE_POINT \
                 (body-local `let g = state.f1 <= state.f2;` lowered via \
                 bind_pre_merge_point_stmts)",
        );
        assert!(
            int_le_pos < mp_pos,
            "BC_INT_LE must precede BC_JIT_MERGE_POINT"
        );

        // BC_INT_LE encoding: [opcode][result_reg][lhs_reg][rhs_reg]; the result
        // reg is byte int_le_pos + 1 and is `g`'s int register byte.
        assert!(
            int_le_pos + 3 < mp_pos,
            "BC_INT_LE payload truncated; int_le_pos={}, mp_pos={}",
            int_le_pos,
            mp_pos
        );
        let g_reg_byte = code[int_le_pos + 1];

        // A.3.5 promote_greens: BC_INT_GUARD_VALUE for `g` must precede the
        // merge point and reference `g`'s int reg.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (jtransform.py:1693): body-local green `g` must be \
                 promoted via BC_INT_GUARD_VALUE before BC_JIT_MERGE_POINT",
            );
        assert!(
            igv_pos > int_le_pos,
            "BC_INT_GUARD_VALUE must come AFTER the BC_INT_LE that defines `g`; \
             got int_le@{} igv@{}",
            int_le_pos,
            igv_pos
        );
        assert_eq!(
            code[igv_pos + 1],
            g_reg_byte,
            "A.3.6.1: BC_INT_GUARD_VALUE reg byte must equal the BC_INT_LE result \
             reg byte (`g`'s int register); got igv_reg={} le_result_reg={}",
            code[igv_pos + 1],
            g_reg_byte
        );

        // jtransform.py:1707: -live- marker immediately precedes guard_value
        // (1-byte opcode + 2-byte offset → BC_LIVE byte at igv_pos - 3).
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at {} has no room for preceding -live-",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede \
             BC_INT_GUARD_VALUE for green promotion; code[{}]={:#04x}, \
             expected BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );

        // A.3.2 merge-point payload: greens_i must include `g`'s reg byte.
        // Layout at greens_base = mp_pos + 2:
        //   greens_base + 0: greens_i_len
        //   greens_base + 1..: greens_i bytes
        let greens_base = mp_pos + 2;
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.6.1: greens_i_len must be 1 (only `g` is green); got {}",
            greens_i_len
        );
        assert_eq!(
            code[greens_base + 1],
            g_reg_byte,
            "A.3.6.1: greens_i[0] must equal `g`'s int register byte from \
             BC_INT_LE result; got greens_i[0]={} le_result_reg={}",
            code[greens_base + 1],
            g_reg_byte
        );
    }
}

/// A.3.6.3: pin the method-call body-local lowering shape against
/// aheui-jit's `let mut stackok = program.get_req_size(pc) as i32 <=
/// state.stacksize;` chain. The fixture exercises:
///   - body-local `let g = program.get_req_size(pc) as i32 <= state.f1
///     as i32;` placed before `jit_merge_point!()`,
///   - `MockProgram::get_req_size => elidable_int` policy registration
///     (the same canonical path used by aheui-jit at `aheui-jit/src/lib.
///     rs:302`),
///   - `greens = [g]` so the merge-point's `greens_i` payload + the
///     A.3.5 promote-greens pair both reference `g`'s int register.
///
/// The shape test pins the canonical `call_pure_int_canonical_via_target`
/// emission (`BC_RESIDUAL_CALL_IR_I` because receiver=Ref + pc=Int) to
/// land BEFORE `BC_JIT_MERGE_POINT(_C)` and threads its result reg
/// forward through `BC_INT_LE` and finally A.3.5's `BC_INT_GUARD_VALUE`
/// + greens_i payload byte. Locks aheui-jit's `stackok` lowering chain
/// so future refactors of `lower_method_call_value` (A.3.6.2) cannot
/// silently regress it.
///
/// RPython parity: `jtransform.py:456 handle_residual_call`
/// (graph-identity-keyed; pyre keys on canonical path so
/// `<MockProgram>::get_req_size` is the segment-form lookup that mirrors
/// the upstream identity match).
mod oparg_with_body_local_method_call_green {
    use majit_metainterp::{Assembler, JitDriver};
    use std::ops::Index;

    /// Mock env type registered against the `Program::get_req_size =>
    /// elidable_int` lookup. Implements `Index<usize, Output = u8>` +
    /// `len()` so the dispatch loop's `program[pc]` opcode fetch and
    /// `pc < program.len()` guard both compile in the macro's input
    /// function (the macro-emitted IR replaces them with
    /// `getarrayitem_gc_i` so the trait impl is only used by the
    /// non-JIT warm-up path).
    struct MockProgram {
        bytes: Vec<u8>,
    }

    impl MockProgram {
        /// Mirror of `ahsembler::Program::get_req_size` (`compiler.rs:28`)
        /// but without the `OP_REQSIZE` lookup table — the body is
        /// irrelevant for the shape test, only the registered
        /// `MockProgram::get_req_size` segment-form policy lookup matters.
        fn get_req_size(&self, _pc: usize) -> i64 {
            0
        }

        fn len(&self) -> usize {
            self.bytes.len()
        }

        /// Required by the `__trace_*` legacy-factory fallback path
        /// (`codegen_trace.rs:129 program.get_op(pc)`); the trace-fn
        /// signature is generic over `env_type`.
        #[allow(dead_code)]
        fn get_op(&self, pc: usize) -> u8 {
            self.bytes[pc]
        }
    }

    impl Index<usize> for MockProgram {
        type Output = u8;
        fn index(&self, pc: usize) -> &u8 {
            &self.bytes[pc]
        }
    }

    struct MethodCallGreenState {
        f1: i64,
        a: i64,
    }

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = MethodCallGreenState,
        env = MockProgram,
        state_fields = { f1: int, a: int },
        // `program.get_req_size(pc) as i32 <= state.f1 as i32` is the
        // body-local computation aheui-jit's main loop uses to derive
        // `stackok`; lower_method_call_value (A.3.6.2) routes the
        // method-call RHS through the call-policy table.
        calls = {
            MockProgram::get_req_size => elidable_int,
        },
        greens = [g],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_body_local_method_call_green(program: &MockProgram, threshold: u32) -> i64 {
        let mut driver: JitDriver<MethodCallGreenState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = MethodCallGreenState { f1: 0, a: 0 };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            // Mirrors `aheui-jit/src/lib.rs:456`:
            //   let mut stackok = program.get_req_size(pc) as i32
            //                       <= state.stacksize;
            let g = program.get_req_size(pc) as i32 <= state.f1 as i32;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.a += 1,
                _ => break,
            }
            let _ = g;
        }
        state.a
    }

    /// A.3.6.3: dispatch JitCode for `dispatch_with_body_local_method_
    /// call_green` must lower the body-local `let g = program.get_req_
    /// size(pc) as i32 <= state.f1 as i32;` BEFORE the portal
    /// `BC_JIT_MERGE_POINT(_C)`, threading the elidable canonical call
    /// op (`BC_RESIDUAL_CALL_IR_I` — receiver Ref + pc Int → IR family
    /// per `assembler.rs:1748-1769`) into `BC_INT_LE` and then A.3.5's
    /// `BC_INT_GUARD_VALUE`. The same reg byte must appear in the
    /// merge-point op's `greens_i` payload (A.3.2 layout).
    #[test]
    fn dispatch_with_body_local_method_call_green_pins_canonical_call_before_merge_point() {
        use majit_metainterp::jitcode::insns::{
            BC_INT_GUARD_VALUE, BC_INT_LE, BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C, BC_LIVE,
            BC_RESIDUAL_CALL_IR_I,
        };

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_body_local_method_call_green(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc =
            __dispatch_jitcode_dispatch_with_body_local_method_call_green(&mut asm, 0i64)
                .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.6.3: BC_JIT_MERGE_POINT(_C) must be present");

        // Canonical elidable call (`call_pure_int_canonical_via_target`)
        // for `program.get_req_size(pc)` must land BEFORE the merge
        // point. Receiver Ref + Int arg → IR-family opcode
        // `BC_RESIDUAL_CALL_IR_I` per `assembler.rs:1748-1769` family
        // selection.
        let call_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_RESIDUAL_CALL_IR_I)
            .expect(
                "A.3.6.3: BC_RESIDUAL_CALL_IR_I (elidable canonical \
                 method-call form) must precede BC_JIT_MERGE_POINT — \
                 lower_method_call_value should have routed `program.\
                 get_req_size(pc)` through call_pure_int_canonical_via_\
                 target with effect_info = ELIDABLE_EFFECT_INFO",
            );

        // BC_RESIDUAL_CALL_IR_I encoding (`assembler.rs:1741-1820 +
        // :1976-2006`) for arg classes "ri" + dst:
        //   [0] opcode
        //   [1] funcptr_reg
        //   [2] int_count = 1
        //   [3] int reg byte (pc → i0)
        //   [4] ref_count = 1
        //   [5] ref reg byte (program → r0)
        //   [6..8] calldescr_idx u16
        //   [8] dst reg byte (the int register holding the call result)
        // The call-result reg drives the LHS of the `<=` comparison.
        let call_op_len = 9;
        assert!(
            call_pos + call_op_len <= mp_pos,
            "A.3.6.3: BC_RESIDUAL_CALL_IR_I encoding truncated before \
             merge point; call_pos={} mp_pos={} need {} bytes",
            call_pos,
            mp_pos,
            call_op_len
        );
        let call_result_reg_byte = code[call_pos + 8];

        // BC_INT_LE for `<call_result> <= state.f1` must land AFTER the
        // canonical call op and BEFORE the merge point. The LHS of `<=`
        // is the call result, so BC_INT_LE's first arg byte (`int_le_pos
        // + 2`, i.e. lhs slot) equals `call_result_reg_byte`.
        let int_le_pos = code[..mp_pos].iter().position(|&b| b == BC_INT_LE).expect(
            "A.3.6.3: BC_INT_LE must precede BC_JIT_MERGE_POINT for \
                 the body-local `<=` comparison",
        );
        assert!(
            int_le_pos > call_pos,
            "BC_INT_LE must come AFTER the BC_RESIDUAL_CALL_IR_I that \
             produces its LHS; got call@{} int_le@{}",
            call_pos,
            int_le_pos
        );

        // BC_INT_LE encoding (`assembler.rs:799-833 record_binop_i`):
        //   [opcode][dst u16 LE][lhs u16 LE][rhs u16 LE]
        // Each reg is `push_u16` (2-byte LE), so the low byte at
        // `int_le_pos + 1 / +3 / +5` is the actual reg index for
        // reg < 256 (high byte at +2 / +4 / +6 is zero).
        assert!(
            int_le_pos + 7 <= mp_pos,
            "BC_INT_LE payload truncated; int_le_pos={}, mp_pos={}",
            int_le_pos,
            mp_pos
        );
        let g_reg_byte = code[int_le_pos + 1];
        assert_eq!(
            code[int_le_pos + 3],
            call_result_reg_byte,
            "A.3.6.3: BC_INT_LE lhs reg (at int_le_pos+3, low byte of \
             u16 LE) must equal the canonical call result reg; got \
             int_le_lhs={} call_result_reg={}",
            code[int_le_pos + 3],
            call_result_reg_byte,
        );

        // A.3.5 promote_greens: BC_INT_GUARD_VALUE for `g` must precede
        // the merge point and reference `g`'s int reg.
        let igv_pos = code[..mp_pos]
            .iter()
            .position(|&b| b == BC_INT_GUARD_VALUE)
            .expect(
                "A.3.5 (jtransform.py:1693): body-local green `g` must \
                 be promoted via BC_INT_GUARD_VALUE before \
                 BC_JIT_MERGE_POINT",
            );
        assert!(
            igv_pos > int_le_pos,
            "BC_INT_GUARD_VALUE must come AFTER the BC_INT_LE that \
             defines `g`; got int_le@{} igv@{}",
            int_le_pos,
            igv_pos
        );
        assert_eq!(
            code[igv_pos + 1],
            g_reg_byte,
            "A.3.6.3: BC_INT_GUARD_VALUE reg byte must equal the \
             BC_INT_LE result reg byte (`g`'s int register); got \
             igv_reg={} le_result_reg={}",
            code[igv_pos + 1],
            g_reg_byte
        );

        // jtransform.py:1707 -live- marker (1-byte opcode + 2-byte
        // offset → BC_LIVE byte at igv_pos - 3).
        assert!(
            igv_pos >= 3,
            "BC_INT_GUARD_VALUE at {} has no room for preceding -live-",
            igv_pos
        );
        assert_eq!(
            code[igv_pos - 3],
            BC_LIVE,
            "A.3.5 (jtransform.py:1707): -live- marker must precede \
             BC_INT_GUARD_VALUE; code[{}]={:#04x}, expected \
             BC_LIVE={:#04x}",
            igv_pos - 3,
            code[igv_pos - 3],
            BC_LIVE,
        );

        // A.3.2 merge-point payload: greens_i[0] must equal `g`'s int reg.
        // Layout at greens_base = mp_pos + 2:
        //   greens_base + 0: greens_i_len
        //   greens_base + 1..: greens_i bytes
        let greens_base = mp_pos + 2;
        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 1,
            "A.3.6.3: greens_i_len must be 1 (only `g` is green); got {}",
            greens_i_len
        );
        assert_eq!(
            code[greens_base + 1],
            g_reg_byte,
            "A.3.6.3: greens_i[0] must equal `g`'s int register byte \
             from BC_INT_LE result; got greens_i[0]={} le_result_reg={}",
            code[greens_base + 1],
            g_reg_byte
        );
    }
}

/// A.3.7: pin the full rpaheui 4-green parity merge-point shape.
/// rpaheui aheui.py:29 declares `greens=['pc', 'stackok', 'is_queue',
/// 'program']`; aheui-jit/src/lib.rs:354 mirrors it as
/// `greens=[pc, stackok, is_queue, program]` after A.3.6.1's body-local
/// walker landed and accepted the `let stackok = ...; let is_queue = ...;`
/// pre-merge-point chain.
///
/// Layout pinned here:
///   - greens_i_len = 3 (pc=Int/i0, stackok=Int from BinOp::Le,
///     is_queue=Int from BinOp::Eq).
///   - greens_r_len = 1 (program=Ref/r0).
///   - greens_f_len = 0.
///   - reds_i_len = reds_r_len = reds_f_len = 0 (both portal-input
///     reds are declared green).
///
/// Locks the byte order so a future macro change that re-buckets greens
/// or shifts the merge-point payload trips this fixture rather than
/// silently regressing rpaheui parity.
mod oparg_with_full_4_green_parity {
    use crate::BytecodeExt;
    use majit_metainterp::{Assembler, JitDriver};

    struct FullParityState {
        f1: i64,
        f2: i64,
        sel: i64,
    }

    type Bytecode = [u8];

    const OP_NOP: u8 = 0;
    const OP_ADD_I: u8 = 1;

    #[majit_macros::jit_interp(
        state = FullParityState,
        env = Bytecode,
        state_fields = { f1: int, f2: int, sel: int },
        greens = [pc, stackok, is_queue, program],
    )]
    #[allow(unused_assignments, unused_variables)]
    fn dispatch_with_full_4_green_parity(program: &Bytecode, threshold: u32) -> i64 {
        let mut driver: JitDriver<FullParityState> = JitDriver::new(threshold);
        let mut pc: usize = 0;
        let mut state = FullParityState {
            f1: 0,
            f2: 0,
            sel: 0,
        };
        {
            use majit_metainterp::JitState as _;
            state
                .build_meta(0, program)
                .install_canonical_liveness(&mut driver);
        }
        while pc < program.len() {
            // rpaheui aheui.py:252 stackok recompute.
            let stackok = state.f1 <= state.f2;
            // rpaheui aheui.py:284 is_queue (recomputed pre-merge-point
            // here from `state.sel == 21` per A.3.7's pyre adaptation).
            let is_queue = state.sel == 21i64;
            jit_merge_point!();
            let opcode = program[pc];
            pc += 1;
            match opcode {
                OP_NOP => {}
                OP_ADD_I => state.f1 += 1,
                _ => break,
            }
            let _ = stackok;
            let _ = is_queue;
        }
        state.f1
    }

    #[test]
    fn dispatch_with_full_4_green_parity_pins_layout() {
        use majit_metainterp::jitcode::insns::{BC_JIT_MERGE_POINT, BC_JIT_MERGE_POINT_C};

        let mut asm = Assembler::new();
        let canonical: Vec<u8> = (0..1u8).collect();
        asm.set_canonical_liveness_triple(canonical, vec![], vec![]);
        __prebuild_jitcode_liveness_dispatch_with_full_4_green_parity(&mut asm);
        let _ = asm.ensure_canonical_liveness_offset();
        let dispatch_jc = __dispatch_jitcode_dispatch_with_full_4_green_parity(&mut asm, 0i64)
            .expect("dispatch lower must succeed for fixture");
        let code = &dispatch_jc.code;

        let mp_pos = code
            .iter()
            .position(|&b| b == BC_JIT_MERGE_POINT || b == BC_JIT_MERGE_POINT_C)
            .expect("A.3.7: BC_JIT_MERGE_POINT(_C) must be present");

        // Layout (jtransform.py:1700 make_three_lists):
        //   greens_base + 0: greens_i_len   = 3
        //   greens_base + 1..3: greens_i bytes (pc, stackok, is_queue regs)
        //   greens_base + 4: greens_r_len   = 1
        //   greens_base + 5: greens_r[0]    = 0 (r0 = program)
        //   greens_base + 6: greens_f_len   = 0
        //   greens_base + 7: reds_i_len     = 0
        //   greens_base + 8: reds_r_len     = 0
        //   greens_base + 9: reds_f_len     = 0
        let greens_base = mp_pos + 2;
        assert!(
            greens_base + 9 < code.len(),
            "A.3.7: payload too short; greens_base={}, len={}",
            greens_base,
            code.len()
        );

        let greens_i_len = code[greens_base] as usize;
        assert_eq!(
            greens_i_len, 3,
            "A.3.7: greens_i_len must be 3 (pc + stackok + is_queue); got {}",
            greens_i_len
        );

        // greens_i[0] = pc → register i0 (= 0).
        assert_eq!(
            code[greens_base + 1],
            0,
            "A.3.7: greens_i[0] must be 0 (i0 = pc per portal-input \
             binding at jitcode_lower.rs:7145); got {}",
            code[greens_base + 1]
        );

        let greens_r_offset = greens_base + 1 + greens_i_len;
        let greens_r_len = code[greens_r_offset] as usize;
        assert_eq!(
            greens_r_len, 1,
            "A.3.7: greens_r_len must be 1 (program → Ref r0); got {}",
            greens_r_len
        );
        assert_eq!(
            code[greens_r_offset + 1],
            0,
            "A.3.7: greens_r[0] must be 0 (r0 = program per portal-input \
             binding at jitcode_lower.rs:7136); got {}",
            code[greens_r_offset + 1]
        );

        let greens_f_offset = greens_r_offset + 1 + greens_r_len;
        let greens_f_len = code[greens_f_offset] as usize;
        assert_eq!(
            greens_f_len, 0,
            "A.3.7: greens_f_len must be 0; got {}",
            greens_f_len
        );

        let reds_i_offset = greens_f_offset + 1 + greens_f_len;
        assert_eq!(
            code[reds_i_offset], 0,
            "A.3.7: reds_i_len must be 0 (pc declared as green); got {}",
            code[reds_i_offset]
        );
        let reds_r_offset = reds_i_offset + 1;
        assert_eq!(
            code[reds_r_offset], 0,
            "A.3.7: reds_r_len must be 0 (program declared as green); got {}",
            code[reds_r_offset]
        );
        let reds_f_offset = reds_r_offset + 1;
        assert_eq!(
            code[reds_f_offset], 0,
            "A.3.7: reds_f_len must be 0; got {}",
            code[reds_f_offset]
        );
    }
}
