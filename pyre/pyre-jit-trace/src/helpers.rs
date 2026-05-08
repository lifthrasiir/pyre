//! JIT helper functions — `extern "C"` wrappers called from compiled traces.
//!
//! The JIT backend (Cranelift) emits C-ABI calls to these functions.
//! Each wraps a pyre-object or pyre-interpreter operation with the
//! correct calling convention and integer-based parameter passing.

use majit_ir::{EffectInfo, ExtraEffect, OopSpecIndex, OpCode, OpRef, Type, Value};
use majit_metainterp::{DEFAULT_EFFECT_INFO, TraceCtx};

use pyre_interpreter::{
    PyBigInt, PyError, binary_op_tag, compare_op_tag, jit_range_iter_next_or_null,
    jit_sequence_getitem,
};
use pyre_interpreter::{
    jit_binary_value_from_tag, jit_bool_value_from_truth, jit_compare_value_from_tag, jit_getattr,
    jit_setattr, jit_setitem, jit_truth_value, jit_unary_invert_value, jit_unary_negative_value,
};
use pyre_object::*;

pub use pyre_interpreter::{
    FlatBuildKind, callable_call_helper, flat_build_helper, jit_build_list_0, jit_build_list_1,
    jit_build_list_2, jit_build_list_3, jit_build_list_4, jit_build_list_5, jit_build_list_6,
    jit_build_list_7, jit_build_list_8, jit_build_map_0, jit_build_map_1, jit_build_map_2,
    jit_build_map_3, jit_build_map_4, jit_build_tuple_0, jit_build_tuple_1, jit_build_tuple_2,
    jit_build_tuple_3, jit_build_tuple_4, jit_build_tuple_5, jit_build_tuple_6, jit_build_tuple_7,
    jit_build_tuple_8, jit_call_callable_0, jit_call_callable_1, jit_call_callable_2,
    jit_call_callable_3, jit_call_callable_4, jit_call_callable_5, jit_call_callable_6,
    jit_call_callable_7, jit_call_callable_8, jit_load_name_from_namespace,
    jit_make_function_from_globals, jit_store_name_to_namespace, known_builtin_call_helper,
    known_function_call_helper, register_jit_function_caller,
};

fn trace_name_args(ctx: &mut TraceCtx, name: &str) -> [OpRef; 2] {
    [
        ctx.const_int(name.as_ptr() as usize as i64),
        ctx.const_int(name.len() as i64),
    ]
}

pub fn emit_trace_call_int(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_int(helper, args)
}

pub fn emit_trace_call_int_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // pyjitpl.py:1995-2068 do_residual_call parity: thread the
    // codewriter-analyzed `EffectInfo` through `record_nospec`. The
    // codewriter's `CallControl::getcalldescr`
    // (`majit-translate/src/jit_codewriter/call.rs`) ports call.py:210-335
    // including the raise / random-effects / write / collect /
    // virtualizable / quasi-immut analyzers; the gap is the trace-side
    // plumbing — pyre-jit-trace helpers live outside the codewriter
    // pipeline so the analyzer's per-callee EI never reaches this
    // emit site. Until per-helper EI registration lands (Task #64),
    // fall back to the conservative `DEFAULT_EFFECT_INFO`
    // (≡ `effectinfo.MOST_GENERAL` for unanalyzed callees: CanRaise +
    // all-writes-set bitmasks).
    ctx.call_int_typed_with_effect(helper, args, arg_types, DEFAULT_EFFECT_INFO)
}

/// `pyjitpl.py:1941-1958 MIFrame.execute_varargs(opnum, argboxes, descr,
/// exc=False, pure=True)` parity for direct trace emit paths.
///
/// `emit_trace_call_int_typed` calls into the tracer with
/// `DEFAULT_EFFECT_INFO` (`effectinfo.MOST_GENERAL`, CanRaise +
/// all-writes-set), so even an `#[elidable_cannot_raise]` callee is
/// recorded as a plain `CallI`.  This wrapper threads an explicit
/// `ElidableCannotRaise` `EffectInfo` through
/// `record_result_of_call_pure` (`pyjitpl.py:3553-3579`) and patches the
/// trace to `CallPureI`.
///
/// `concrete_arg_values` follows `_build_allboxes`
/// (`pyjitpl.py:1960-1993`): the funcbox's concrete value sits in slot 0
/// and the per-argument concrete values follow.  Direct trace-emit paths
/// have no jitcode-dispatch frame-operand-fetch, so the caller supplies
/// the values directly.
pub fn emit_trace_call_int_typed_elidable_cannot_raise(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
    concrete_arg_values: &[Value],
    concrete_result: Value,
) -> OpRef {
    let effect = EffectInfo::new(ExtraEffect::ElidableCannotRaise, OopSpecIndex::None);
    ctx.call_typed_with_effect_pure(
        OpCode::CallI,
        helper,
        args,
        arg_types,
        Type::Int,
        effect,
        concrete_arg_values,
        concrete_result,
    )
}

pub fn emit_trace_call_ref(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) -> OpRef {
    ctx.call_ref(helper, args)
}

pub fn emit_trace_call_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    // See emit_trace_call_int_typed for the plumbing-gap rationale.
    ctx.call_ref_typed_with_effect(helper, args, arg_types, DEFAULT_EFFECT_INFO)
}

pub fn emit_trace_call_void(ctx: &mut TraceCtx, helper: *const (), args: &[OpRef]) {
    ctx.call_void(helper, args);
}

pub fn emit_trace_call_void_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) {
    ctx.call_void_typed(helper, args, arg_types);
}

pub fn emit_trace_call_may_force_ref_typed(
    ctx: &mut TraceCtx,
    helper: *const (),
    args: &[OpRef],
    arg_types: &[Type],
) -> OpRef {
    ctx.call_may_force_ref_typed(helper, args, arg_types)
}

pub fn emit_trace_build_flat(
    ctx: &mut TraceCtx,
    kind: FlatBuildKind,
    items: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper_count = match kind {
        FlatBuildKind::Map => items.len() / 2,
        FlatBuildKind::List | FlatBuildKind::Tuple => items.len(),
    };
    let Some(helper) = flat_build_helper(kind, helper_count) else {
        let opname = match kind {
            FlatBuildKind::List => "list",
            FlatBuildKind::Tuple => "tuple",
            FlatBuildKind::Map => "map",
        };
        return Err(PyError::type_error(format!(
            "{opname} build arity not supported by JIT"
        )));
    };
    let arg_types = vec![Type::Ref; items.len()];
    Ok(ctx.call_ref_typed_with_effect(helper, items, &arg_types, DEFAULT_EFFECT_INFO))
}

pub fn emit_trace_call_callable(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = callable_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_call_known_builtin(
    ctx: &mut TraceCtx,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_builtin_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("builtin call arity not supported by JIT"))?;
    let mut call_args = vec![callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_ref_typed_with_effect(helper, &call_args, &arg_types, DEFAULT_EFFECT_INFO))
}

pub fn emit_trace_call_known_function(
    ctx: &mut TraceCtx,
    frame: OpRef,
    callable: OpRef,
    args: &[OpRef],
) -> Result<OpRef, PyError> {
    let helper = known_function_call_helper(args.len())
        .ok_or_else(|| PyError::type_error("function call arity not supported by JIT"))?;
    let mut call_args = vec![frame, callable];
    call_args.extend_from_slice(args);
    let mut arg_types = vec![Type::Ref, Type::Ref];
    arg_types.extend(std::iter::repeat_n(Type::Ref, args.len()));
    Ok(ctx.call_may_force_ref_typed(helper, &call_args, &arg_types))
}

pub fn emit_trace_unpack_sequence(
    ctx: &mut TraceCtx,
    seq: OpRef,
    count: usize,
) -> Result<Vec<OpRef>, PyError> {
    let mut items = Vec::with_capacity(count);
    for idx in 0..count {
        let idx_const = ctx.const_int(idx as i64);
        items.push(emit_trace_call_ref_typed(
            ctx,
            jit_sequence_getitem as *const (),
            &[seq, idx_const],
            &[Type::Ref, Type::Int],
        ));
    }
    Ok(items)
}

pub fn emit_trace_load_name_from_namespace(
    ctx: &mut TraceCtx,
    namespace: OpRef,
    name: &str,
) -> OpRef {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    emit_trace_call_ref_typed(
        ctx,
        jit_load_name_from_namespace as *const (),
        &[namespace, name_ptr, name_len],
        &[Type::Ref, Type::Int, Type::Int],
    )
}

pub fn emit_trace_store_name_to_namespace(
    ctx: &mut TraceCtx,
    namespace: OpRef,
    name: &str,
    value: OpRef,
) {
    let [name_ptr, name_len] = trace_name_args(ctx, name);
    emit_trace_call_void_typed(
        ctx,
        jit_store_name_to_namespace as *const (),
        &[namespace, name_ptr, name_len, value],
        &[Type::Ref, Type::Int, Type::Int, Type::Ref],
    );
}

pub fn emit_trace_truth_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_int_typed(ctx, jit_truth_value as *const (), &[value], &[Type::Ref])
}

pub fn emit_trace_bool_value_from_truth(ctx: &mut TraceCtx, truth: OpRef, negate: bool) -> OpRef {
    let truth = if negate {
        let one = ctx.const_int(1);
        ctx.record_op(OpCode::IntSub, &[one, truth])
    } else {
        truth
    };
    emit_trace_call_ref_typed(
        ctx,
        jit_bool_value_from_truth as *const (),
        &[truth],
        &[Type::Int],
    )
}

pub fn emit_trace_binary_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_interpreter::bytecode::BinaryOperator,
) -> Result<OpRef, PyError> {
    let Some(tag) = binary_op_tag(op) else {
        return Err(PyError::type_error(format!(
            "binary operation {op:?} not yet traceable"
        )));
    };
    let tag = ctx.const_int(tag);
    Ok(emit_trace_call_ref_typed(
        ctx,
        jit_binary_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    ))
}

pub fn emit_trace_compare_value(
    ctx: &mut TraceCtx,
    a: OpRef,
    b: OpRef,
    op: pyre_interpreter::bytecode::ComparisonOperator,
) -> OpRef {
    let tag = ctx.const_int(compare_op_tag(op));
    emit_trace_call_ref_typed(
        ctx,
        jit_compare_value_from_tag as *const (),
        &[a, b, tag],
        &[Type::Ref, Type::Ref, Type::Int],
    )
}

pub fn emit_trace_range_iter_next_or_null(ctx: &mut TraceCtx, iter: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_range_iter_next_or_null as *const (),
        &[iter],
        &[Type::Ref],
    )
}

/// RPython ConstPtr parity: boxed-object constants are Ref-typed.
/// The optimizer can constant-fold immutable field reads from Ref
/// constants (heap.py:640 constant_fold).
pub fn emit_trace_int_constant(ctx: &mut TraceCtx, value: i64) -> OpRef {
    ctx.const_ref(w_int_new(value) as i64)
}

pub fn emit_trace_float_constant(ctx: &mut TraceCtx, value: f64) -> OpRef {
    ctx.const_ref(box_float_constant(value) as i64)
}

pub fn emit_trace_unary_negative_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_unary_negative_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub fn emit_trace_unary_invert_value(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_trace_call_ref_typed(
        ctx,
        jit_unary_invert_value as *const (),
        &[value],
        &[Type::Ref],
    )
}

pub trait TraceHelperAccess {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R;
    fn trace_frame(&self) -> OpRef;
    fn trace_globals_ptr(&mut self) -> OpRef;
    fn trace_record_not_forced_guard(&mut self);

    fn trace_make_function(&mut self, code_obj: OpRef) -> Result<OpRef, PyError> {
        let globals = self.trace_globals_ptr();
        self.with_trace_ctx(|ctx| {
            Ok(emit_trace_call_ref_typed(
                ctx,
                jit_make_function_from_globals as *const (),
                &[globals, code_obj],
                &[Type::Ref, Type::Ref],
            ))
        })
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result =
            self.with_trace_ctx(|ctx| emit_trace_call_callable(ctx, frame, callable, args))?;
        self.trace_record_not_forced_guard();
        Ok(result)
    }

    fn trace_build_list(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::List, items))
    }

    fn trace_build_tuple(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::Tuple, items))
    }

    fn trace_build_map(&mut self, items: &[OpRef]) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_build_flat(ctx, FlatBuildKind::Map, items))
    }

    fn trace_store_subscr(&mut self, obj: OpRef, key: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            let _ = emit_trace_call_int_typed(
                ctx,
                jit_setitem as *const (),
                &[obj, key, value],
                &[Type::Ref, Type::Ref, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_load_attr(&mut self, obj: OpRef, name: &str) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            Ok(emit_trace_call_ref_typed(
                ctx,
                jit_getattr as *const (),
                &[obj, name_ptr, name_len],
                &[Type::Ref, Type::Int, Type::Int],
            ))
        })
    }

    fn trace_store_attr(&mut self, obj: OpRef, name: &str, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            let [name_ptr, name_len] = trace_name_args(ctx, name);
            let _ = emit_trace_call_int_typed(
                ctx,
                jit_setattr as *const (),
                &[obj, name_ptr, name_len, value],
                &[Type::Ref, Type::Int, Type::Int, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_list_append(&mut self, list: OpRef, value: OpRef) -> Result<(), PyError> {
        self.with_trace_ctx(|ctx| {
            emit_trace_call_void_typed(
                ctx,
                jit_list_append as *const (),
                &[list, value],
                &[Type::Ref, Type::Ref],
            );
            Ok(())
        })
    }

    fn trace_unpack_sequence(&mut self, seq: OpRef, count: usize) -> Result<Vec<OpRef>, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_unpack_sequence(ctx, seq, count))
    }

    fn trace_iter_next_value(&mut self, iter: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_range_iter_next_or_null(ctx, iter)))
    }

    fn trace_truth_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_truth_value(ctx, value)))
    }

    fn trace_bool_value_from_truth(
        &mut self,
        truth: OpRef,
        negate: bool,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_bool_value_from_truth(ctx, truth, negate)))
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| emit_trace_binary_value(ctx, a, b, op))
    }

    fn trace_compare_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::ComparisonOperator,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_compare_value(ctx, a, b, op)))
    }

    fn trace_unary_negative_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_unary_negative_value(ctx, value)))
    }

    fn trace_unary_invert_value(&mut self, value: OpRef) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_unary_invert_value(ctx, value)))
    }

    fn trace_int_constant(&mut self, value: i64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_int_constant(ctx, value)))
    }

    fn trace_bigint_constant(&mut self, value: &PyBigInt) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(box_bigint_constant(value) as i64)))
    }

    fn trace_float_constant(&mut self, value: f64) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(emit_trace_float_constant(ctx, value)))
    }

    fn trace_bool_constant(&mut self, value: bool) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(w_bool_from(value) as i64)))
    }

    fn trace_str_constant(&mut self, value: &str) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(box_str_constant(value) as i64)))
    }

    fn trace_bytes_constant(&mut self, value: &[u8]) -> Result<OpRef, PyError> {
        let r = pyre_object::bytesobject::w_bytes_from_bytes(value);
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(r as i64)))
    }

    fn trace_code_constant(
        &mut self,
        code: &pyre_interpreter::CodeObject,
    ) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| {
            Ok(ctx.const_ref(pyre_interpreter::box_code_constant(code) as i64))
        })
    }

    fn trace_none_constant(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(w_none() as i64)))
    }

    fn trace_ellipsis_constant(&mut self) -> Result<OpRef, PyError> {
        self.with_trace_ctx(|ctx| Ok(ctx.const_ref(pyre_object::noneobject::w_ellipsis() as i64)))
    }
}

/// Emit inline W_Int creation (NewWithVtable + SetfieldGc).
///
/// jtransform.py:908-911 rewrite_op_setfield: setfield on typeptr is dropped
/// — `new_with_vtable` writes the typeptr in the backend (llmodel.py:778-782).
pub fn emit_box_int_inline(
    ctx: &mut TraceCtx,
    raw_int: OpRef,
    size_descr: majit_ir::DescrRef,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py:908-911: rewrite_op_setfield skips typeptr setfield
    // entirely ("ignore the operation completely -- instead, it's done by
    // 'new'"). rewrite.py:479-484 handle_malloc_operation emits the vtable
    // setfield via fielddescr_vtable during GC rewrite of NEW_WITH_VTABLE.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    // Emit: SetfieldGc(v, intval, raw_int)
    let intval_idx = intval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_int], intval_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_op, intval_idx, raw_int);
    new_op
}

/// Emit inline W_Float creation (NewWithVtable + SetfieldGc).
pub fn emit_box_float_inline(
    ctx: &mut TraceCtx,
    raw_float: OpRef,
    size_descr: majit_ir::DescrRef,
    floatval_descr: majit_ir::DescrRef,
) -> OpRef {
    // jtransform.py:908-911 parity: typeptr setfield filtered in trace.
    let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr);
    ctx.heap_cache_mut().new_object(new_op);
    let floatval_idx = floatval_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_op, raw_float], floatval_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_op, floatval_idx, raw_float);
    new_op
}

/// Emit a fresh callee `PyFrame` directly into the trace IR for the
/// self-recursive single-int-argument fast path.
///
/// Replaces the opaque `jit_create_self_recursive_callee_frame_1_raw_int`
/// CallR that today (`call_jit.rs:2814`) wraps `arena.take()` + reuse
/// check + locals zero-fill + raw_int boxing in an opaque helper. The
/// helper is `#[dont_look_inside]` so the optimizer cannot virtualize
/// the new frame nor fold the boxing — every fib(35) iteration pays
/// the full helper trampoline (~336k calls/run, observed in
/// `phase2_3_self_recursive_call_perf_plan_2026_04_28.md`). PyPy emits
/// `direct_assembler_call` with `NewWithVtable(jitframe) + SetfieldGc(...)`
/// in trace IR (`backend/aarch64/opassembler.py:1080-1200`); this helper
/// is the closest pyre analogue given that pyre's `PyFrame` plays the
/// role both of CPython's `PyFrame` and of PyPy's separate `jitframe` /
/// virtualizable.
///
/// Restrictions held by the caller:
///   - self-recursive (callee `pycode` ≡ caller `pycode`), so the
///     caller passes `pycode` / `w_globals` / `execution_context` in
///     directly: `pycode` and `w_globals` arrive as trace-time
///     constants (the bound `W_CodeObject` and its `function.w_globals`,
///     both immutable for the trace's lifetime), and `execution_context`
///     arrives as the loop's already-materialised `sym.execution_context`
///     OpRef (per-thread; not safe to const-fold across thread entries).
///     This mirrors PyPy aarch64 `direct_assembler_call` (`backend/aarch64/
///     opassembler.py:1080-1200`) which writes the callee jitframe's vable
///     scalars from constants known at trace-compile time.
///   - 1 raw-int argument (no boxed-arg path; caller is responsible
///     for the `trace_guarded_int_payload` unbox).
///   - no cellvars/freevars on the callee — `init_cells` is skipped.
///     The caller verifies this against the concrete `W_CodeObject`
///     before invoking the helper.
///
/// The IR sequence mirrors `pyframe.rs::PyFrame::new_for_call_with_closure`:
///
/// 1. `emit_box_int_inline(raw_int_arg)` → `boxed` W_IntObject (one
///    nursery alloc, optimizer can fold into a virtual when the boxed
///    value never escapes through `GuardNotForced`).
/// 2. `NewArrayClear(array_size)` with `pyobject_gcarray_descr()` —
///    the locals_cells_stack_w `FixedObjectArray<PyObjectRef>`
///    backing storage. Clear so unset slots read as `PY_NULL`.
/// 3. `SetarrayitemGc(locals_array, 0, boxed)` — bind the lone
///    positional argument. Other slots stay `PY_NULL`.
/// 4. `NewWithVtable(pyframe_size_descr())` — `vtable=0` because
///    `PyFrame` is not an `rclass.OBJECT` instance (registered via
///    `TypeInfo::with_gc_ptrs`, see `pyre-jit/src/eval.rs::initialize_gc`),
///    so `handle_new` skips the vtable setfield (rewrite.py:925-933
///    `gen_new_with_vtable` early-out for `vtable == 0`).
/// 5. `SetfieldGc` ops for the constructor-visible fields. The non-zero
///    fields (`execution_context`, `pycode`, `w_globals`,
///    `locals_cells_stack_w`, `valuestackdepth`, `last_instr=-1`) mirror
///    `new_for_call_with_closure`; the nullable GC fields
///    (`f_generator_nowref`, `w_yielding_from`, `f_backref`) are written
///    explicitly to match the same constructor shape instead of relying on
///    an implicit backend zero-fill side effect.
pub fn emit_new_pyframe_inline_self_recursive(
    ctx: &mut TraceCtx,
    raw_int_arg: OpRef,
    array_size: usize,
    valuestackdepth: usize,
    pycode: OpRef,
    w_globals: OpRef,
    ec: OpRef,
) -> OpRef {
    use crate::descr::{
        int_intval_descr, pyframe_code_descr, pyframe_dict_storage_descr,
        pyframe_execution_context_descr, pyframe_f_backref_descr, pyframe_f_generator_nowref_descr,
        pyframe_locals_cells_stack_descr, pyframe_next_instr_descr, pyframe_size_descr,
        pyframe_stack_depth_descr, pyframe_w_yielding_from_descr, w_int_size_descr,
    };
    use crate::state::pyobject_gcarray_descr;

    // Step 1 — box the raw int into a fresh W_IntObject. Mirrors the
    // `w_int_new(raw_int_arg)` call inside the opaque helper.
    let boxed = emit_box_int_inline(ctx, raw_int_arg, w_int_size_descr(), int_intval_descr());

    // Step 2 — allocate the locals_cells_stack_w array. `NewArrayClear`
    // zeros every slot so any future LOAD_FAST on an unbound local
    // observes `PY_NULL` (UnboundLocalError parity).
    let len_ref = ctx.const_int(array_size as i64);
    let array_descr = pyobject_gcarray_descr();
    let locals_array =
        ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
    ctx.heap_cache_mut().new_object(locals_array);

    // Step 3 — locals[0] = boxed. The single positional argument of
    // the self-recursive call.
    let zero = ctx.const_int(0);
    ctx.record_op_with_descr(
        OpCode::SetarrayitemGc,
        &[locals_array, zero, boxed],
        array_descr,
    );

    // Step 4 — allocate the new PyFrame. NewWithVtable zero-fills the
    // payload; the GC tags it with `PYFRAME_GC_TYPE_ID` because the
    // size descr's parent type id is registered in `pyre-jit/src/eval.rs`.
    let new_frame = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], pyframe_size_descr());
    ctx.heap_cache_mut().new_object(new_frame);

    // Step 5 — SetfieldGc for the constructor-visible fields, mirroring
    // the explicit assignments inside `new_for_call_with_closure`.
    // Order matches the field declaration so the optimizer's lazy-set
    // replace logic groups them together.
    let ec_descr = pyframe_execution_context_descr();
    let ec_idx = ec_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, ec], ec_descr);
    ctx.heap_cache_mut().setfield_cached(new_frame, ec_idx, ec);

    let code_descr = pyframe_code_descr();
    let code_idx = code_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, pycode], code_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, code_idx, pycode);

    let globals_descr = pyframe_dict_storage_descr();
    let globals_idx = globals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, w_globals], globals_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, globals_idx, w_globals);

    let locals_descr = pyframe_locals_cells_stack_descr();
    let locals_idx = locals_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, locals_array], locals_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, locals_idx, locals_array);

    let vsd = ctx.const_int(valuestackdepth as i64);
    let vsd_descr = pyframe_stack_depth_descr();
    let vsd_idx = vsd_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, vsd], vsd_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, vsd_idx, vsd);

    let neg_one = ctx.const_int(-1);
    let last_instr_descr = pyframe_next_instr_descr();
    let last_instr_idx = last_instr_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, neg_one], last_instr_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, last_instr_idx, neg_one);

    let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);

    let generator_descr = pyframe_f_generator_nowref_descr();
    let generator_idx = generator_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, null_ref], generator_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, generator_idx, null_ref);

    let yielding_descr = pyframe_w_yielding_from_descr();
    let yielding_idx = yielding_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, null_ref], yielding_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, yielding_idx, null_ref);

    let backref_descr = pyframe_f_backref_descr();
    let backref_idx = backref_descr.index();
    ctx.record_op_with_descr(OpCode::SetfieldGc, &[new_frame, null_ref], backref_descr);
    ctx.heap_cache_mut()
        .setfield_cached(new_frame, backref_idx, null_ref);

    new_frame
}

// ── Elidable canary helper ──────────────────────────────────────────
//
// rlib/jit.py:13 `@jit.elidable` parity.  PyPy `intobject.py:891-895
// wrapint` in-range check parity: returns true iff `value` falls inside
// the prebuilt-int small cache range AND the cache is enabled.
//
// Deterministic for any `value`, no side effects, no raise →
// `EF_ELIDABLE_CANNOT_RAISE` (`call.py:299`).
//
// Pyre's first production-crate `#[elidable_cannot_raise]` callee.  The
// trace-side effect (`record_result_of_call_pure` patching `CallI` to
// `CallPureI`) is exercised by
// `pyre/pyre-jit-trace/tests/elidable_helper_canary_test.rs`, which
// invokes the helper through `TraceCtx::call_typed_with_effect_pure`
// with an explicit `ElidableCannotRaise` `EffectInfo`.  Production
// `emit_trace_call_int_typed` callsites still pass
// `DEFAULT_EFFECT_INFO` until per-helper EI registration (Task #64)
// lands; this canary closes the macro-side gap so the EI side can
// proceed in a separate slice.
#[majit_macros::elidable_cannot_raise]
pub fn jit_int_in_small_cache_range(value: i64) -> bool {
    w_int_small_cached(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_interpreter::PyExecutionContext;
    use pyre_interpreter::{ConstantData, compile_exec};

    /// Verifies that `#[elidable_cannot_raise]` emits the
    /// `INT_ELIDABLE_CANNOT_RAISE = 19` policy byte (`call.py:299`
    /// parity / `call_policy_byte.rs:96`) and produces non-null
    /// trace_target / concrete_target trampolines.  Lives in-crate
    /// because external integration tests cannot reach the macro's
    /// `pub(crate)` `__majit_call_policy_*` symbol from outside.
    #[test]
    fn elidable_helper_macro_emits_int_elidable_cannot_raise_byte() {
        let (policy, _, trace_target, concrete_target, _) =
            __majit_call_policy_jit_int_in_small_cache_range();
        assert_eq!(policy, 19u8);
        assert!(!trace_target.is_null());
        assert!(!concrete_target.is_null());
    }

    /// Confirms the helper still runs correctly under the elidable
    /// attribute and preserves the elidable invariant (same input →
    /// same output).  With the `WITHPREBUILTINT=false` default
    /// (`intobject.rs:44`) every input returns false.
    #[test]
    fn elidable_helper_is_deterministic_and_matches_pyre_object() {
        for &value in &[0i64, 7, -3, 42, i64::MIN, i64::MAX] {
            let r1 = jit_int_in_small_cache_range(value);
            let r2 = jit_int_in_small_cache_range(value);
            assert_eq!(r1, r2, "elidable helper must be deterministic");
            assert_eq!(
                r1,
                w_int_small_cached(value),
                "elidable wrapper must mirror pyre-object's underlying check",
            );
        }
    }

    #[test]
    fn test_callable_call_helper_dispatches_builtin_without_trace_side_branching() {
        // `fresh_dict_storage` seeds only `__builtins__`; individual
        // names like `abs` resolve via
        // `frame.get_builtin().getdictvalue(name)` (`pyopcode.py:921`).
        // Pull `abs` directly out of the EC's builtins storage to
        // mirror the same resolution path the interpreter follows.
        let ctx = PyExecutionContext::default();
        let storage_ptr = ctx.builtins_storage_ptr();
        let abs = unsafe { pyre_interpreter::dict_storage_get(&*storage_ptr, "abs") }
            .expect("abs builtin must exist");
        let result = jit_call_callable_1(0, abs as i64, w_int_new(-11) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 11);
        }
    }

    #[test]
    fn test_container_helpers_dispatch_expected_runtime_shapes() {
        let result = jit_build_tuple_2(w_int_new(3) as i64, w_int_new(5) as i64);
        let tuple = result as PyObjectRef;
        unsafe {
            assert!(is_tuple(tuple));
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 0).unwrap()), 3);
            assert_eq!(w_int_get_value(w_tuple_getitem(tuple, 1).unwrap()), 5);
        }

        let list = w_list_new(vec![w_int_new(2), w_int_new(4)]);
        let tuple = w_tuple_new(vec![w_int_new(7), w_int_new(9)]);
        unsafe {
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(list as i64, 1) as PyObjectRef),
                4
            );
            assert_eq!(
                w_int_get_value(jit_sequence_getitem(tuple as i64, 0) as PyObjectRef),
                7
            );
        }

        let result = jit_build_map_2(
            w_int_new(1) as i64,
            w_int_new(10) as i64,
            w_int_new(2) as i64,
            w_int_new(20) as i64,
        );
        let dict = result as PyObjectRef;
        unsafe {
            assert!(is_dict(dict));
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 1).unwrap()), 10);
            assert_eq!(w_int_get_value(w_dict_getitem(dict, 2).unwrap()), 20);
        }
    }

    #[test]
    fn test_numeric_helpers_reuse_objspace_semantics() {
        let result = jit_binary_value_from_tag(w_int_new(9) as i64, w_int_new(4) as i64, 1);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), 5);
        }

        let result = jit_compare_value_from_tag(w_int_new(2) as i64, w_int_new(7) as i64, 0);
        unsafe {
            assert!(w_bool_get_value(result as PyObjectRef));
        }

        let result = jit_unary_invert_value(w_int_new(5) as i64);
        unsafe {
            assert_eq!(w_int_get_value(result as PyObjectRef), !5);
        }
    }

    #[test]
    fn test_make_function_helper_wraps_code_object() {
        let module = compile_exec("def f(x):\n    return x").expect("compile failed");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } => Some(code.as_ref().clone()),
                _ => None,
            })
            .expect("expected nested function code");
        let code_ptr = Box::into_raw(Box::new(code)) as *const ();
        let code_obj = pyre_interpreter::w_code_new(code_ptr);
        let func = jit_make_function_from_globals(0, code_obj as i64) as PyObjectRef;

        unsafe {
            assert!(pyre_interpreter::is_function(func));
            // Function.code now stores the W_CodeObject, not the raw CodeObject.
            assert_eq!(
                pyre_interpreter::function_get_code(func),
                code_obj as *const ()
            );
        }
    }

    #[test]
    fn test_range_iter_next_helper_uses_runtime_iterator_step() {
        let iter = w_range_iter_new(0, 2, 1);
        let first = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let second = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let done = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(first), 0);
            assert_eq!(w_int_get_value(second), 1);
            assert!(done.is_null());
        }
    }
}
