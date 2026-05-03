use majit_macros::{dont_look_inside, jit_inline};
use majit_metainterp::jitcode::JitCodeRuntimeExt;

fn assert_single_return_opcode(jitcode: &majit_metainterp::JitCode, key: &str) {
    let opcode = *majit_metainterp::jitcode::wellknown_bh_insns()
        .get(key)
        .unwrap_or_else(|| panic!("missing wellknown opcode for {key}"));
    assert_eq!(
        jitcode.code.len(),
        3,
        "helper should emit one return opcode"
    );
    assert_eq!(jitcode.code[0], opcode, "helper should end with {key}");
}

fn assert_trailing_return(
    jitcode: &majit_metainterp::JitCode,
    key: &str,
    kind: majit_metainterp::JitArgKind,
    reg: u16,
) {
    assert_single_return_opcode(jitcode, key);
    assert_eq!(
        jitcode.trailing_return_info(),
        Some((kind, reg)),
        "helper should expose trailing return info from jitcode bytecode"
    );
}

// ── Ref-returning inline helpers ────────────────────────────────────

#[jit_inline]
fn inline_ref_identity(ptr: usize) -> usize {
    ptr
}

// ── Float-returning inline helpers ──────────────────────────────────

#[jit_inline]
fn inline_float_identity(value: f64) -> f64 {
    value
}

#[jit_inline]
fn inline_mixed_int_identity(_ptr: usize, value: i64, _scale: f64) -> i64 {
    value
}

#[dont_look_inside]
fn wrapped_ref_identity(ptr: *const i64) -> *const i64 {
    ptr
}

#[dont_look_inside]
fn wrapped_float_identity(value: f64) -> f64 {
    value
}

#[dont_look_inside]
fn wrapped_int_identity(value: i64) -> i64 {
    value
}

// ── Tests ───────────────────────────────────────────────────────────

#[test]
fn jit_inline_ref_identity_generates_valid_jitcode() {
    // Phase 4 Epic B.3-B.4: inline helper jitcodes register their
    // per-marker liveness triples into an `Assembler` passed in by the
    // caller (the production path uses the driver-shared one).  Tests
    // that only inspect structural properties pass a freshly-allocated
    // `Assembler`; the resulting BC_LIVE 2-byte slots are scoped to that
    // local table — ref/float helpers below have no guards/markers, so
    // there's nothing for the test path to decode beyond the trailing
    // return.
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_ref_identity_with_asm(&mut asm);
    // RPython jitcode.py:37-39 c_num_regs_i/r/f
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert!(jitcode.c_num_regs_r >= 1, "at least 1 ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 0, "no float registers needed");
    assert_trailing_return(
        &jitcode,
        "ref_return/r",
        majit_metainterp::JitArgKind::Ref,
        0,
    );
}

#[test]
fn jit_inline_float_identity_generates_valid_jitcode() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_float_identity_with_asm(&mut asm);
    assert_eq!(jitcode.c_num_regs_i, 0, "no int registers needed");
    assert_eq!(jitcode.c_num_regs_r, 0, "no ref registers needed");
    assert!(
        jitcode.c_num_regs_f >= 1,
        "at least 1 float register needed"
    );
    assert_trailing_return(
        &jitcode,
        "float_return/f",
        majit_metainterp::JitArgKind::Float,
        0,
    );
}

#[test]
fn jit_inline_ref_identity_keeps_interpreter_behavior() {
    assert_eq!(inline_ref_identity(42), 42);
    assert_eq!(inline_ref_identity(0), 0);
}

#[test]
fn jit_inline_float_identity_keeps_interpreter_behavior() {
    assert_eq!(inline_float_identity(3.14), 3.14);
    assert_eq!(
        inline_float_identity(-0.0f64).to_bits(),
        (-0.0f64).to_bits()
    );
}

#[test]
fn jit_inline_mixed_identity_generates_dense_per_kind_jitcode() {
    let mut asm = majit_metainterp::Assembler::new();
    let jitcode = __majit_inline_jitcode_inline_mixed_int_identity_with_asm(&mut asm);
    assert_eq!(jitcode.c_num_regs_i, 1, "one int register needed");
    assert_eq!(jitcode.c_num_regs_r, 1, "one ref register needed");
    assert_eq!(jitcode.c_num_regs_f, 1, "one float register needed");
    assert_trailing_return(
        &jitcode,
        "int_return/i",
        majit_metainterp::JitArgKind::Int,
        0,
    );
}

#[test]
fn jit_inline_inferred_policy_only_advertises_int_return_helpers() {
    let (ref_policy, ref_builder, _, _) = __majit_call_policy_inline_ref_identity();
    assert_eq!(
        ref_policy, 0u8,
        "ref-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        ref_builder.is_null(),
        "ref-return inline helper should not expose inferred inline builder"
    );

    let (float_policy, float_builder, _, _) = __majit_call_policy_inline_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return inline helper should not claim inferred inline_int parity"
    );
    assert!(
        float_builder.is_null(),
        "float-return inline helper should not expose inferred inline builder"
    );

    let (int_policy, int_builder, _, _) = __majit_call_policy_inline_mixed_int_identity();
    assert_eq!(
        int_policy, 4u8,
        "int-return inline helper should keep inferred inline policy"
    );
    assert!(
        !int_builder.is_null(),
        "int-return inline helper should expose inferred inline builder"
    );
}

#[test]
fn wrapped_non_int_helpers_keep_targets_but_do_not_advertise_inferred_value_policy() {
    let (ref_policy, ref_inline_builder, ref_trace_target, ref_concrete_target) =
        __majit_call_policy_wrapped_ref_identity();
    assert_eq!(
        ref_policy, 0u8,
        "ref-return wrapped helper should not advertise inferred value-call policy"
    );
    assert!(
        ref_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !ref_trace_target.is_null() && !ref_concrete_target.is_null(),
        "explicit wrapped ref policy still needs trace/concrete targets"
    );

    let (float_policy, float_inline_builder, float_trace_target, float_concrete_target) =
        __majit_call_policy_wrapped_float_identity();
    assert_eq!(
        float_policy, 0u8,
        "float-return wrapped helper should not advertise inferred value-call policy"
    );
    assert!(
        float_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !float_trace_target.is_null() && !float_concrete_target.is_null(),
        "explicit wrapped float policy still needs trace/concrete targets"
    );

    let (int_policy, int_inline_builder, int_trace_target, int_concrete_target) =
        __majit_call_policy_wrapped_int_identity();
    assert_eq!(
        int_policy, 2u8,
        "int-return wrapped helper should keep inferred residual-int policy"
    );
    assert!(
        int_inline_builder.is_null(),
        "wrapped helper should not use inline builder slot"
    );
    assert!(
        !int_trace_target.is_null() && !int_concrete_target.is_null(),
        "int-return wrapped helper should still expose call targets"
    );
}

// ── JitCode runtime test: Ref inline call ───────────────────────────

#[test]
fn jit_inline_ref_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_ref_identity_with_asm(&mut asm);
    let (sub_return_kind, _sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("ref helper should end in ref_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Ref,
        "ref helper should report Ref kind"
    );

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a ref value in ref register 0
    builder.load_const_r_value(0, 0xDEAD);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_r_r(sub_idx, &[(0, 0)], Some(1));
    let jitcode = builder.finish();

    // Verify the JitCode was built without panics and has correct structure
    assert!(
        jitcode.c_num_regs_r >= 2,
        "caller needs at least 2 ref registers"
    );
    assert_eq!(
        jitcode.exec.descrs.len(),
        1,
        "one sub-jitcode for inline call"
    );
}

// ── JitCode runtime test: Float inline call ─────────────────────────

#[test]
fn jit_inline_float_identity_works_through_jitcode_builder() {
    use majit_metainterp::JitCodeBuilder;

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_float_identity_with_asm(&mut asm);
    let (sub_return_kind, _sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("float helper should end in float_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Float,
        "float helper should report Float kind"
    );

    let mut builder = JitCodeBuilder::new();
    // Simulate: caller has a float value in float register 0
    builder.load_const_f_value(0, f64::to_bits(3.14) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_irf_f(sub_idx, &[], &[], &[(0, 0)], Some(1));
    let jitcode = builder.finish();

    assert!(
        jitcode.c_num_regs_f >= 2,
        "caller needs at least 2 float registers"
    );
    assert_eq!(
        jitcode.exec.descrs.len(),
        1,
        "one sub-jitcode for inline call"
    );
}

#[test]
fn jit_inline_mixed_identity_uses_dense_kind_banks_at_runtime() {
    use majit_metainterp::JitCodeBuilder;
    use majit_metainterp::blackhole::BlackholeInterpreter;

    let mut asm = majit_metainterp::Assembler::new();
    let sub_jitcode = __majit_inline_jitcode_inline_mixed_int_identity_with_asm(&mut asm);
    let (sub_return_kind, sub_return_reg) = sub_jitcode
        .trailing_return_info()
        .expect("mixed helper should end in int_return");
    assert_eq!(
        sub_return_kind,
        majit_metainterp::JitArgKind::Int,
        "mixed helper should report Int kind"
    );
    assert_eq!(
        sub_return_reg, 0,
        "mixed helper int parameter should live in dense int reg 0"
    );

    let mut builder = JitCodeBuilder::new();
    builder.load_const_r_value(0, 0xDEAD);
    builder.load_const_i_value(0, 21);
    builder.load_const_f_value(0, f64::to_bits(3.5) as i64);
    let sub_idx = builder.add_sub_jitcode(sub_jitcode);
    builder.inline_call_irf_i(sub_idx, &[(0, 0)], &[(0, 0)], &[(0, 0)], Some(1));
    let jitcode = builder.finish();

    let mut bh = BlackholeInterpreter::new();
    bh.setposition(std::sync::Arc::new(jitcode), 0);
    let _ = bh.run();

    assert_eq!(bh.registers_i[1], 21);
}
