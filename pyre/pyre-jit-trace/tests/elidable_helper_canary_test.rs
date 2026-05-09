//! `#[elidable_cannot_raise]` production helper canary — proves that
//! the first helper attached on the pyre-jit-trace side
//! (`jit_int_in_small_cache_range`) fires through the elidable path
//! when traced.
//!
//! The metainterp-side fixture (`majit-metainterp/tests/elidable_canary_test.rs`)
//! showed the infrastructure live wire; this test goes further and
//! shows that **a macro-attached helper inside a production crate
//! (pyre-jit-trace) is patched to `CallPureI` on a real trace call**.
//!
//! Every trace function pointer routes through the macro-emitted
//! `extern "C" fn(_, _) -> i64` ABI trampoline (`__majit_call_target_*`).
//! PyPy `getfunctionptr` (`call.py:174-187`) parity — the function
//! pointer the trace records must match (down to ABI) the function the
//! JIT actually calls.
//!
//! `pyjitpl.py:1941-1958 MIFrame.execute_varargs(opnum, argboxes, descr,
//! exc=False, pure=True)` + `pyjitpl.py:3553-3579 record_result_of_call_pure`
//! still work when driven from a production helper.

use majit_ir::{OpCode, OpRef, Type, Value};
use majit_metainterp::{BackEdgeAction, MetaInterp};
use pyre_jit_trace::helpers::{
    __majit_call_policy_jit_int_in_small_cache_range,
    __majit_call_target_jit_int_in_small_cache_range,
    emit_trace_call_int_typed_elidable_cannot_raise,
};
use pyre_object::{__majit_call_policy_int_bit_count, __majit_call_target_int_bit_count};

/// Confirms that the macro emits `__majit_call_policy_*` /
/// `__majit_call_target_*` with the user fn's visibility and that the
/// 4-tuple's trace_target slot points at that wrapper.  PyPy
/// `getfunctionptr` parity, observed from outside the helper's crate.
#[test]
fn elidable_helper_macro_advertises_extern_c_trampoline() {
    let (policy, _, trace_target, concrete_target, _) =
        __majit_call_policy_jit_int_in_small_cache_range();
    assert_eq!(policy, 19u8);
    assert_eq!(
        trace_target, __majit_call_target_jit_int_in_small_cache_range as *const (),
        "policy 4-tuple trace_target must point to the macro-generated extern \"C\" wrapper",
    );
    assert!(!concrete_target.is_null());
}

#[test]
fn elidable_helper_traces_to_call_pure_i_when_args_not_all_const() {
    // One inputarg + one Const argument → all_const=false →
    // record_result_of_call_pure patches CallI to CallPureI.
    //
    // Strictly speaking `jit_int_in_small_cache_range` is a 1-arg
    // helper, so a single-inputarg scenario is enough to demonstrate
    // the live wire.

    let mut meta = MetaInterp::<()>::new(0);
    let live_x: i64 = 7;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::from_raw(0);
    // Use the macro ABI trampoline directly as the trace function
    // pointer — `extern "C" fn(i64) -> i64`.  The user fn's
    // `pub fn(i64) -> bool` signature may not share calling convention
    // with the C ABI, so feeding it straight into a `Type::Int` (i64)
    // target is unsafe.  Only the macro wrapper covers the bool→i64
    // conversion (1/0) at `majit-macros/src/lib.rs:192`.
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(live_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[live_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    let ops = ctx.ops();

    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "trace must contain CallPureI after record_result_of_call_pure patch; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const result must remain a live OpRef",
    );
}

/// Confirms that the `emit_trace_call_int_typed_elidable_cannot_raise`
/// production emitter is auto-patched to `CallPureI` in the trace.
/// Unlike `helpers.rs::emit_trace_call_int_typed`'s conservative
/// `default_effect_info` call, this wrapper threads an explicit
/// elidable EffectInfo through `record_result_of_call_pure`.
#[test]
fn emit_trace_call_int_typed_elidable_cannot_raise_routes_to_call_pure_i() {
    let mut meta = MetaInterp::<()>::new(0);
    let live_x: i64 = 0x_dead_beef;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::from_raw(0);
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(live_x);

    let resbox = {
        let ctx = meta.trace_ctx().expect("active trace");
        emit_trace_call_int_typed_elidable_cannot_raise(
            ctx,
            func_ptr,
            &[live_arg],
            &[Type::Int],
            &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
            Value::Int(concrete_result),
        )
    };

    let ctx = meta.trace_ctx().expect("active trace");
    let ops = ctx.ops();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "production emitter must record CallPureI; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut",
    );
    assert!(
        ctx.constants_get_value(resbox).is_none(),
        "non-all-const result must remain a live OpRef",
    );
}

/// Confirms that `pyre-object::int_bit_count` (port of RPython
/// `intobject.py:516 _bit_count`) is patched to `CallPureI` when traced.
/// Together with `jit_int_in_small_cache_range` (pyre-jit-trace side
/// attachment), this proves that the pyre-object side fires through the
/// same `record_result_of_call_pure` path.
#[test]
fn elidable_int_bit_count_macro_advertises_extern_c_trampoline_and_traces_call_pure_i() {
    let (policy, _, trace_target, _, _) = __majit_call_policy_int_bit_count();
    assert_eq!(policy, 19u8);
    assert_eq!(trace_target, __majit_call_target_int_bit_count as *const (),);

    let mut meta = MetaInterp::<()>::new(0);
    let live_x: i64 = 0x_0bad_cafe_dead_beef;
    let action = meta.force_start_tracing(0, (0, 0), None, &[Value::Int(live_x)]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let live_arg = OpRef::from_raw(0);
    let trace_fn = __majit_call_target_int_bit_count;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(live_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[live_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(live_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    let ops = ctx.ops();
    assert!(
        ops.iter().any(|op| op.opcode == OpCode::CallPureI),
        "int_bit_count must record CallPureI via record_result_of_call_pure",
    );
    assert!(
        ops.iter().all(|op| op.opcode != OpCode::CallI),
        "original CallI must be cut"
    );
    assert!(ctx.constants_get_value(resbox).is_none());
}

#[test]
fn elidable_helper_all_const_args_fold_to_const_and_cut_call() {
    let mut meta = MetaInterp::<()>::new(0);
    let action = meta.force_start_tracing(0, (0, 0), None, &[]);
    assert!(matches!(action, BackEdgeAction::StartedTracing));

    let const_x: i64 = 3;
    let trace_fn = __majit_call_target_jit_int_in_small_cache_range;
    let func_ptr = trace_fn as *const ();
    let concrete_result = trace_fn(const_x);
    let effect = majit_ir::EffectInfo::new(
        majit_ir::ExtraEffect::ElidableCannotRaise,
        majit_ir::OopSpecIndex::None,
    );

    let const_arg = meta.trace_ctx().expect("active trace").const_int(const_x);

    let resbox = meta.trace_ctx().unwrap().call_typed_with_effect_pure(
        OpCode::CallI,
        func_ptr,
        &[const_arg],
        &[Type::Int],
        Type::Int,
        effect,
        &[Value::Int(func_ptr as usize as i64), Value::Int(const_x)],
        Value::Int(concrete_result),
    );

    let ctx = meta.trace_ctx().expect("active trace");
    let ops = ctx.ops();
    assert!(
        ops.iter()
            .all(|op| op.opcode != OpCode::CallI && op.opcode != OpCode::CallPureI),
        "all-const elidable call must be fully cut; got opcodes {:?}",
        ops.iter().map(|op| op.opcode).collect::<Vec<_>>()
    );
    assert_eq!(
        ctx.constants_get_value(resbox),
        Some(Value::Int(concrete_result)),
        "all-const fold must return ConstInt(concrete_result)",
    );
}
