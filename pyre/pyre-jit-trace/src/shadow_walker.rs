//! Shadow execution scaffolding.
//!
//! Master plan tingly-splashing-balloon.md Phase D-2: production tracing
//! runs the trait-driven `MIFrame::execute_opcode_step` (today's only
//! path) AND additionally re-runs the equivalent jitcode through the
//! symbolic walker (`jitcode_dispatch::dispatch_via_miframe`), then
//! asserts the two paths recorded the same trace ops. Mismatches panic
//! with a structured diff so divergences surface immediately.
//!
//! This module is the env-gated migration harness now wired from
//! `trace_opcode.rs`: for allow-listed Python opcodes it snapshots the
//! symbolic state, runs the codewriter-emitted jitcode through
//! `dispatch_via_miframe`, rolls the recorder/state back, lets the trait
//! path execute normally, and compares the two recorded op slices.
//!
//! Current production allow-list is deliberately tiny: `Nop` and the four
//! arm-id-zero siblings (`ExtendedArg`, `Resume`, `Cache`, `NotTaken`).
//! With `MAJIT_SHADOW_WALKER` unset this module is a runtime no-op. Future
//! slices grow [`opname_in_shadow_allow_list`] opcode-by-opcode until every
//! Python opcode is shadow-validated; slices E/F finally switch the
//! production path to the walker and remove the trait dispatch.
//!
//! RPython parity: there is no direct counterpart — RPython has only
//! the jitcode interpreter and never carried two parallel trace
//! recorders. Pyre's shadow_walker exists exclusively as a migration
//! safety net while the trait dispatch is being retired; once the
//! migration completes, this module can be removed (Phase E).

use std::sync::OnceLock;

use majit_ir::Op;
use majit_metainterp::recorder::TracePosition;
use pyre_interpreter::{Instruction, OpArg};

use crate::state::MIFrame;

/// Cache the env-var read so per-call sites don't pay the syscall on
/// every Python opcode. RPython has no equivalent — env-var feature
/// flags are a pyre-only adaptation, used during the trait→walker
/// migration only. See module-level docs for the migration plan.
fn shadow_walker_enabled_once() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MAJIT_SHADOW_WALKER")
            .ok()
            .as_deref()
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

/// Returns `true` iff the process started with `MAJIT_SHADOW_WALKER=1`.
/// Production hook sites (Slice 2) gate their shadow walker call on
/// this. Default off — production unaffected.
pub fn shadow_walker_enabled() -> bool {
    shadow_walker_enabled_once()
}

/// Declarative whitelist of Python instructions whose codewriter-emitted
/// jitcode arm is fully covered by
/// [`crate::jitcode_dispatch::dispatch_via_miframe`].
///
/// Caller contract: the production hook invokes the shadow validate path
/// only when this returns `true`. The list grows opcode-by-opcode as walker
/// coverage closes (`raise/r` GUARD_CLASS, virtualizable force-path ops,
/// `CALL_LOOPINVARIANT_R`, etc. — see `jitcode_dispatch.rs` "Production
/// fidelity gaps" header).
///
/// Adding an instruction here is a parity assertion: every opname inside
/// the codewriter-emitted arm must have a `dispatch_via_miframe` handler
/// that records byte-identical IR ops to the trait path. New entries
/// SHOULD ship with an integration test that runs the bench suite under
/// `MAJIT_SHADOW_WALKER=1` and confirms zero diff panics.
pub fn opname_in_shadow_allow_list(instruction: &Instruction) -> bool {
    // Phase D-3 status: Blocker #1 (descr_refs placeholder pool) closed
    // by [`crate::descr::make_descr_from_bh`] +
    // `crate::jitcode_runtime::ALL_DESCR_REFS` rewiring. Blocker #2
    // (synthetic Rust constructor wrapper) closed by jtransform's
    // `Ok` / `Err` / `Some` identity rewrite
    // (`majit/majit-translate/src/jit_codewriter/jtransform.rs
    //  ::rewrite_op_direct_call`): the trailing `int_copy +
    // residual_call_r_r/iRd>r` pair every opcode arm carried for the
    // `Ok(StepResult::Continue)` return wrapper is gone, so the walker
    // emits zero `CallR` ops for opcodes whose body has no real call.
    //
    // First production opcode allow-listed: `Instruction::Nop` (and the
    // four arm-id-zero siblings `ExtendedArg`, `Resume`, `Cache`,
    // `NotTaken`). The arm bytes after the elision land at:
    //
    //     ref_return/r ; ref_return/r ; live/ ; raise/r
    //
    // None of those record a trace op when the walker dispatches under
    // `is_top_level=false`: `ref_return/r` exits as
    // `DispatchOutcome::SubReturn` (no `Finish`) and the `raise/r`
    // / `live/` tails are unreachable for the success path. Trait
    // dispatch for the same opcodes runs `Ok(StepResult::Continue)`
    // natively and also emits zero ops, so walker == trait by
    // construction.
    //
    // To extend the allow-list beyond Nop family, decode the candidate
    // arm via `cargo test -p pyre-jit-trace ... dump_<name>_arm_bytes`
    // and confirm every opname routes to a recorder-side handler the
    // trait path also reaches. Anything still carrying a
    // `residual_call_*` wrapper without a matching trait-side record
    // panics under `MAJIT_SHADOW_WALKER=1`.
    matches!(
        instruction,
        Instruction::Nop
            | Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Cache
            | Instruction::NotTaken
    )
}

/// Carrier for the symbolic walker's record output — the trace ops it
/// emitted plus the recorder position taken before the walker mutated
/// MIFrame. [`shadow_validate_pre`] returns this on the walker leg;
/// the trait leg consumes it through [`shadow_validate_post`] which
/// recovers the trait-dispatch ops as
/// `ctx.ops()[pre_pos.._pos..post_pos._pos]` and compares them against
/// `walker_ops` via [`diff_recorded_ops`], panicking on mismatch.
///
/// The trace recorder rollback strategy (Phase D-2.3.b) is to call
/// `TraceCtx::cut_trace(pre_pos)` once the walker leg finishes
/// capturing its op slice. That returns the recorder to the pre-opcode
/// position so the trait path records into the same byte offset and
/// OpRefs stay stable across the two paths.
pub struct ShadowOutcome {
    /// `TraceCtx::get_trace_position()` snapshot taken before the walker
    /// ran. Used by [`shadow_validate_post`] to slice out the trait-leg
    /// ops `ctx.ops()[pre_pos.._pos..post_pos._pos]`.
    pre_pos: TracePosition,
    /// Owned copy of the ops the walker recorded. Cloned out of
    /// `ctx.ops()` before [`shadow_validate_pre`] rolled the recorder
    /// back via `cut_trace(pre_pos)`. Compared op-for-op against the
    /// trait leg's slice in [`shadow_validate_post`] via
    /// [`diff_recorded_ops`].
    walker_ops: Vec<Op>,
}

/// Walker leg of the shadow validator. When `MAJIT_SHADOW_WALKER=1` is
/// set AND `instruction` is in [`opname_in_shadow_allow_list`]:
///
/// 1. snapshot the recorder position + the symbolic register banks +
///    `last_exc_box` / `class_of_last_exc_is_const`,
/// 2. resolve the codewriter-emitted arm via
///    [`crate::jitcode_runtime::jitcode_for_instruction`],
/// 3. resolve the four `done_with_this_frame_descr_*` + the
///    `exit_frame_with_exception_descr_ref` from the shared
///    `MetaInterpStaticData` so descr identity matches the trait path,
/// 4. run [`crate::jitcode_dispatch::dispatch_via_miframe`] against the
///    arm with the production `descr_refs` pool
///    ([`crate::jitcode_runtime::all_descr_refs`]) and a sub-jitcode
///    lookup over [`crate::jitcode_runtime::all_jitcodes`],
/// 5. clone the walker-emitted op slice
///    `ctx.ops()[pre_pos.._pos..walker_post_pos._pos]` into the returned
///    [`ShadowOutcome`],
/// 6. roll the recorder back via `cut_trace(pre_pos)` and restore the
///    register banks + last-exc fields so the trait dispatch sees the
///    same state it would have without shadow on.
///
/// On `MAJIT_SHADOW_WALKER=0` or when the allow-list rejects
/// `instruction`, returns `None` and skips all of the above. Pass the
/// returned outcome (if any) to [`shadow_validate_post`] AFTER the trait
/// dispatch finishes.
pub fn shadow_validate_pre(
    miframe: &mut MIFrame,
    instruction: &Instruction,
    _op_arg: OpArg,
) -> Option<ShadowOutcome> {
    if !shadow_walker_enabled() {
        return None;
    }
    if !opname_in_shadow_allow_list(instruction) {
        return None;
    }

    // Resolve the codewriter-emitted arm. None means the parser emitted
    // `Wildcard`/`Unsupported` for this variant — production should
    // never have allow-listed it; bail loudly.
    let jitcode =
        crate::jitcode_runtime::jitcode_for_instruction(instruction).unwrap_or_else(|| {
            panic!(
                "shadow_walker: allow-listed instruction has no codewriter arm: {:?}",
                instruction
            )
        });

    // Snapshot pre-state. Borrow ctx and sym sequentially — both come
    // out of MIFrame's raw pointers so back-to-back reborrows are sound.
    let pre_pos = miframe.ctx().get_trace_position();
    let (
        saved_registers_r,
        saved_registers_i,
        saved_registers_f,
        saved_last_exc_box,
        saved_class_const,
    ) = {
        let s = miframe.sym();
        (
            s.registers_r.clone(),
            s.registers_i.clone(),
            s.registers_f.clone(),
            s.last_exc_box,
            s.class_of_last_exc_is_const,
        )
    };

    // Pull the five terminal descrs off MetaInterpStaticData so the
    // walker's `Finish` records carry the same DescrRef identities the
    // trait path emits via `compile_done_with_this_frame` /
    // `compile_exit_frame_with_exception` (`pyjitpl.py:3202-3242`).
    //
    // Shadow validation MODE: a missing terminal descr means
    // `MetaInterpStaticData::setup_descrs` never ran on this build,
    // which is a real configuration failure — feeding the walker
    // placeholder descrs would mask a setup bug behind a
    // false-negative parity report. Panic instead of silently
    // returning `None` so the env-gated diff actually shouts when
    // the build it's supposed to validate is misconfigured.
    let (done_void, done_int, done_ref, done_float, exit_exc_ref) = {
        let sd = miframe.ctx().metainterp_sd();
        let void = sd
            .done_with_this_frame_descr_void
            .clone()
            .expect("MAJIT_SHADOW_WALKER=1 requires MetaInterpStaticData::done_with_this_frame_descr_void to be wired");
        let int = sd
            .done_with_this_frame_descr_int
            .clone()
            .expect("MAJIT_SHADOW_WALKER=1 requires MetaInterpStaticData::done_with_this_frame_descr_int to be wired");
        let ref_ = sd
            .done_with_this_frame_descr_ref
            .clone()
            .expect("MAJIT_SHADOW_WALKER=1 requires MetaInterpStaticData::done_with_this_frame_descr_ref to be wired");
        let float = sd
            .done_with_this_frame_descr_float
            .clone()
            .expect("MAJIT_SHADOW_WALKER=1 requires MetaInterpStaticData::done_with_this_frame_descr_float to be wired");
        let exc = sd
            .exit_frame_with_exception_descr_ref
            .clone()
            .expect("MAJIT_SHADOW_WALKER=1 requires MetaInterpStaticData::exit_frame_with_exception_descr_ref to be wired");
        (void, int, ref_, float, exc)
    };

    // Production sub-jitcode lookup over the shared
    // `crate::jitcode_runtime::all_jitcodes()` table — same shape as
    // `jitcode_dispatch::tests::production_sub_jitcodes`. The Arc<JitCode>
    // entries live inside a `LazyLock<Vec<...>>`, so the `.code` slice
    // is `'static`-rooted as `SubJitCodeBody` requires.
    let sub_jitcode_lookup = |idx: usize| -> Option<crate::jitcode_dispatch::SubJitCodeBody> {
        let all = crate::jitcode_runtime::all_jitcodes();
        all.get(idx)
            .map(|jc| crate::jitcode_dispatch::SubJitCodeBody {
                code: jc.code.as_slice(),
                num_regs_r: jc.num_regs_r() as usize,
                num_regs_i: jc.num_regs_i() as usize,
                num_regs_f: jc.num_regs_f() as usize,
            })
    };

    // Walker emits ops directly into `miframe.ctx`. We only catch errors
    // here; the recorded ops are what `diff_recorded_ops` later
    // compares — walker-internal state (`SubReturn`/`Terminate`) is not
    // observable from the trait side.
    // is_top_level=false: the codewriter packages each Python opcode
    // into a self-contained sub-jitcode invoked from the outer
    // dispatcher via `inline_call_r_r/dR>r`. The arm always ends with a
    // `*_return/*` terminator. In the production trace, that terminator
    // surfaces as `SubReturn` to the dispatcher (no FINISH). The trait
    // dispatch path doesn't emit a per-opcode FINISH either, so shadow
    // mode must run the walker as a sub-frame to match.
    let walk_result = crate::jitcode_dispatch::dispatch_via_miframe(
        miframe,
        jitcode.code.as_slice(),
        0,
        crate::jitcode_runtime::all_descr_refs(),
        &sub_jitcode_lookup,
        done_ref,
        done_int,
        done_float,
        done_void,
        exit_exc_ref,
        false,
    );
    if let Err(e) = walk_result {
        panic!(
            "shadow_walker: dispatch_via_miframe failed for {:?}: {:?}",
            instruction, e
        );
    }

    // Clone walker-emitted ops out of the recorder before rolling back.
    let walker_post_pos = miframe.ctx().get_trace_position();
    let walker_ops: Vec<Op> = miframe
        .ctx()
        .ops()
        .get(pre_pos._pos..walker_post_pos._pos)
        .map(|s| s.to_vec())
        .unwrap_or_default();

    // Roll recorder + sym fields back so the trait dispatch sees its
    // pre-walker state. `cut_trace` truncates ops; sym.registers_* /
    // last_exc_* are restored from the snapshot so handlers like
    // `int_copy/i>i` or `raise/r` that mutate the banks don't leak into
    // the trait leg.
    miframe.ctx().cut_trace(pre_pos);
    {
        let s = miframe.sym_mut();
        s.registers_r = saved_registers_r;
        s.registers_i = saved_registers_i;
        s.registers_f = saved_registers_f;
        s.last_exc_box = saved_last_exc_box;
        s.class_of_last_exc_is_const = saved_class_const;
    }

    Some(ShadowOutcome {
        pre_pos,
        walker_ops,
    })
}

/// Trait leg of the shadow validator. Recovers the trait-dispatch op
/// slice as `ctx.ops()[outcome.pre_pos.._pos..post_pos._pos]`, runs
/// [`diff_recorded_ops`] against the walker's slice, and panics with
/// the structured diff on mismatch. Caller invokes this AFTER the trait
/// dispatch returns and only when [`shadow_validate_pre`] returned
/// `Some`.
pub fn shadow_validate_post(miframe: &mut MIFrame, outcome: ShadowOutcome) {
    let post_pos = miframe.ctx().get_trace_position();
    let trait_ops: Vec<Op> = miframe
        .ctx()
        .ops()
        .get(outcome.pre_pos._pos..post_pos._pos)
        .map(|s| s.to_vec())
        .unwrap_or_default();
    if let Some(diff) = diff_recorded_ops(&trait_ops, &outcome.walker_ops) {
        panic!("{}", diff);
    }
}

/// Compare the trace ops recorded by the trait-dispatch path against
/// the walker-shadow path for the same Python opcode. Returns:
/// * `None` — both paths produced the same op sequence (per
///   `OpCode + args` equality; descr identity via `Arc::ptr_eq`).
/// * `Some(diff)` — structured human-readable diff suitable for
///   panic message. Caller is expected to panic; this helper does not
///   abort process so unit tests can exercise both branches.
///
/// The comparison is intentionally narrow today: opcode + args + descr
/// pointer identity. `pos` differs by construction (the two paths
/// record into separate `TraceCtx` instances) and is excluded.
/// `fail_args`, `fail_arg_types`, `rd_resume_position`, `vecinfo` are
/// all set by later optimizer / unroll passes — they are absent at
/// recording time so excluding them adds no false-negatives. As later
/// slices land richer state in walker (e.g. capture_resumedata for
/// num_live), this helper can be tightened.
pub fn diff_recorded_ops(production: &[Op], shadow: &[Op]) -> Option<String> {
    if production.len() != shadow.len() {
        return Some(format!(
            "shadow_walker: op count mismatch — production={}, shadow={}\n\
             production opcodes: {:?}\nshadow opcodes:     {:?}",
            production.len(),
            shadow.len(),
            production.iter().map(|o| o.opcode).collect::<Vec<_>>(),
            shadow.iter().map(|o| o.opcode).collect::<Vec<_>>(),
        ));
    }
    for (idx, (p, s)) in production.iter().zip(shadow.iter()).enumerate() {
        if p.opcode != s.opcode {
            return Some(format!(
                "shadow_walker: op[{}] opcode mismatch — production={:?}, shadow={:?}",
                idx, p.opcode, s.opcode,
            ));
        }
        if p.args.as_slice() != s.args.as_slice() {
            return Some(format!(
                "shadow_walker: op[{}] ({:?}) args mismatch — \
                 production={:?}, shadow={:?}",
                idx,
                p.opcode,
                p.args.as_slice(),
                s.args.as_slice(),
            ));
        }
        match (&p.descr, &s.descr) {
            (None, None) => {}
            (Some(pd), Some(sd)) => {
                if !std::sync::Arc::ptr_eq(pd, sd) {
                    return Some(format!(
                        "shadow_walker: op[{}] ({:?}) descr identity mismatch — \
                         production={:p}, shadow={:p}",
                        idx,
                        p.opcode,
                        std::sync::Arc::as_ptr(pd),
                        std::sync::Arc::as_ptr(sd),
                    ));
                }
            }
            (production_some, shadow_some) => {
                return Some(format!(
                    "shadow_walker: op[{}] ({:?}) descr presence mismatch — \
                     production_has_descr={}, shadow_has_descr={}",
                    idx,
                    p.opcode,
                    production_some.is_some(),
                    shadow_some.is_some(),
                ));
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{DescrRef, OpCode, OpRef};

    fn op(opcode: OpCode, args: &[OpRef], descr: Option<DescrRef>) -> Op {
        Op {
            opcode,
            args: args.iter().copied().collect(),
            descr,
            pos: OpRef::op_typed(0, opcode.result_type()),
            type_: opcode.result_type(),
            fail_args: None,
            fail_arg_types: None,
            rd_resume_position: -1,
            vecinfo: None,
        }
    }

    #[test]
    fn shadow_walker_enabled_defaults_off() {
        // Without `MAJIT_SHADOW_WALKER=1` set in the test process env,
        // the gate must read `false`. The OnceLock cache means the
        // value is fixed for the whole test process — toggling the env
        // var inside one test would not flip the gate. This test only
        // verifies the default branch.
        if std::env::var("MAJIT_SHADOW_WALKER").ok().as_deref() == Some("1") {
            // Skip: the test process happens to have the gate on; the
            // default-off branch is unobservable.
            return;
        }
        assert!(!shadow_walker_enabled());
    }

    #[test]
    fn pop_top_stays_out_of_shadow_allow_list_until_recursive_jitcode_closes() {
        // PopTop's top-level arm recurses into a helper jitcode. The current
        // first unsupported callee opname is `getfield_vable_i/rd>i`; listing
        // PopTop here would claim parity before the recursive closure is wired.
        assert!(!opname_in_shadow_allow_list(&Instruction::PopTop));
    }

    #[test]
    fn diff_recorded_ops_returns_none_for_identical_traces() {
        let descr = make_fail_descr_ref();
        let trace_a = vec![
            op(OpCode::IntAdd, &[OpRef::int_op(0), OpRef::int_op(1)], None),
            op(OpCode::Finish, &[OpRef::int_op(2)], Some(descr.clone())),
        ];
        let trace_b = trace_a.clone();
        assert_eq!(diff_recorded_ops(&trace_a, &trace_b), None);
    }

    #[test]
    fn diff_recorded_ops_reports_count_mismatch() {
        let trace_a = vec![op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            None,
        )];
        let trace_b: Vec<Op> = vec![];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op count mismatch"), "diff was: {diff}");
        assert!(diff.contains("production=1"));
        assert!(diff.contains("shadow=0"));
    }

    #[test]
    fn diff_recorded_ops_reports_opcode_mismatch() {
        let trace_a = vec![op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            None,
        )];
        let trace_b = vec![op(
            OpCode::IntSub,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            None,
        )];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op[0] opcode mismatch"), "diff was: {diff}");
        assert!(diff.contains("IntAdd"));
        assert!(diff.contains("IntSub"));
    }

    #[test]
    fn diff_recorded_ops_reports_args_mismatch() {
        let trace_a = vec![op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(1)],
            None,
        )];
        let trace_b = vec![op(
            OpCode::IntAdd,
            &[OpRef::int_op(0), OpRef::int_op(2)],
            None,
        )];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("op[0]"));
        assert!(diff.contains("args mismatch"), "diff was: {diff}");
    }

    #[test]
    fn diff_recorded_ops_reports_descr_identity_mismatch() {
        let descr_a = make_fail_descr_ref();
        let descr_b = make_fail_descr_ref();
        let trace_a = vec![op(OpCode::Finish, &[OpRef::int_op(0)], Some(descr_a))];
        let trace_b = vec![op(OpCode::Finish, &[OpRef::int_op(0)], Some(descr_b))];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("descr identity mismatch"), "diff was: {diff}");
    }

    #[test]
    fn diff_recorded_ops_reports_descr_presence_mismatch() {
        let descr = make_fail_descr_ref();
        let trace_a = vec![op(OpCode::Finish, &[OpRef::int_op(0)], Some(descr))];
        let trace_b = vec![op(OpCode::Finish, &[OpRef::int_op(0)], None)];
        let diff = diff_recorded_ops(&trace_a, &trace_b).expect("must report mismatch");
        assert!(diff.contains("descr presence mismatch"), "diff was: {diff}");
        assert!(diff.contains("production_has_descr=true"));
        assert!(diff.contains("shadow_has_descr=false"));
    }

    fn make_fail_descr_ref() -> DescrRef {
        std::sync::Arc::new(majit_ir::SimpleFailDescr::new(0, 0, vec![]))
    }
}
