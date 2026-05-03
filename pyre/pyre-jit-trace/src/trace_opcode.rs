//! MIFrame opcode handlers for trace-time JIT.
//!
//! Contains all `impl MIFrame` methods and trait implementations
//! (SharedOpcodeHandler, LocalOpcodeHandler, etc.).

use crate::state::*;

use majit_ir::{DescrRef, GcRef, OpCode, OpRef, Type, Value};
use majit_metainterp::{TraceAction, TraceCtx};

use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};

/// floatobject.py:561 `descr_pow` → `_pow(space, x, y)` parity.
///
/// `_pow` in floatobject.py:799-881 takes two raw floats and returns a
/// raw float (can raise OverflowError / ValueError / ZeroDivisionError).
/// The JIT trace records this as `CALL_F(float_pow_jit, lhs, rhs)`
/// (pyjitpl.py:2119-2121 CALL_F branch taken because
/// `check_forces_virtual_or_virtualizable()` is False for ll_math_pow,
/// and `exc=True` because EF_CAN_RAISE), followed by `GUARD_NO_EXCEPTION`
/// via `handle_possible_exception` (pyjitpl.py:1950-1955, 3395).
///
/// ll_math_pow (ll_math.py:260) is the can-raise helper (EF_CAN_RAISE),
/// NOT elidable and NOT force-virtual. Using Rust's native `x.powf(y)`
/// would drop the Python exception semantics (negative base fractional
/// exponent → ValueError, 0.0 raised to negative → ZeroDivisionError,
/// overflow → OverflowError). Using CALL_MAY_FORCE_F would be wrong
/// because the optimizer postpones that family until GUARD_NOT_FORCED
/// arrives (heap.py CALL_MAY_FORCE branch), which is the virtualizable
/// protocol — ll_math_pow does not touch virtualizables.
///
/// Extracted to module level for stable function pointer identity.
///
/// Must match `float_pow_impl` semantics in `baseobjspace.rs`: any
/// divergence would cause the JIT compiled code to produce a different
/// result from the interpreter for the same input (correctness bug).
pub(crate) extern "C" fn float_pow_jit(x: f64, y: f64) -> f64 {
    match pyre_interpreter::float_pow_raw(x, y) {
        Ok(z) => z,
        Err(err) => {
            // llmodel.py:194-199 _store_exception parity: set JIT exception
            // state so the following GuardNoException sees it and fails,
            // propagating the raise into the meta-interpreter.
            let exc_obj = err.to_exc_object();
            #[cfg(feature = "cranelift")]
            majit_backend_cranelift::jit_exc_raise(exc_obj as i64);
            #[cfg(feature = "dynasm")]
            majit_backend_dynasm::jit_exc_raise(exc_obj as i64);
            let _ = exc_obj; // suppress unused warning when no backend
            // Return value is discarded by GuardNoException path; use NaN
            // as a safe sentinel in case the guard is elided.
            f64::NAN
        }
    }
}
use pyre_interpreter::eval::{get_current_exception, set_current_exception};

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// The trace records the Python `CALL` that constructs an exception object,
/// then `RAISE_VARARGS` itself must materialize a real JIT exception before
/// `handle_possible_exception` emits `GuardException`. Without this explicit
/// helper the compiled bridge contains only the constructor call plus
/// `GuardException`, so the guard sees no pending exception and incorrectly
/// resumes down the normal path.
pub(crate) extern "C" fn raise_exception_jit(exc_obj: i64) {
    #[cfg(feature = "cranelift")]
    majit_backend_cranelift::jit_exc_raise(exc_obj);
    #[cfg(feature = "dynasm")]
    majit_backend_dynasm::jit_exc_raise(exc_obj);
    let _ = exc_obj;
}

/// Runtime helper for traced `RAISE_VARARGS`.
///
/// Mirrors `eval.rs:1035-1129` on the compiled path:
/// normalize the exception operand, normalize/attach the optional
/// cause, and publish the final exception via `jit_exc_raise` so the
/// following `GUARD_EXCEPTION` sees it.
pub(crate) extern "C" fn normalize_raise_varargs_jit(
    frame_ptr: i64,
    exc_obj: i64,
    cause_obj: i64,
) -> i64 {
    let frame_ptr = frame_ptr as *const pyre_interpreter::pyframe::PyFrame;
    let exc = exc_obj as pyre_object::PyObjectRef;
    let raw_cause = cause_obj as pyre_object::PyObjectRef;

    // pyopcode.py:704-722 — cause and exc normalization both run against
    // `self.space`/`frame.execution_context`. Pin the caller's frame
    // context for the whole body so the cause-class-call and the
    // exc-class-call observe the same namespace / thread state.
    let frame_ctx = if frame_ptr.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*frame_ptr).execution_context }
    };
    let saved_ctx = pyre_interpreter::call::take_last_exec_ctx();
    if !frame_ctx.is_null() {
        pyre_interpreter::call::set_last_exec_ctx(frame_ctx);
    }

    let cause = if raw_cause.is_null() {
        None
    } else {
        // pyopcode.py:706-707 — cause class-call must mirror the exc
        // class-call (pyopcode.py:711-713) on compiled traces. Force
        // both onto the plain interpreter path so the constructor
        // cannot re-enter the tracer.
        let result = {
            let _plain_guard = pyre_interpreter::call::force_plain_eval();
            normalize_raise_cause(raw_cause)
        };
        match result {
            Ok(cause) => Some(cause),
            Err(err) => {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let exc = err.to_exc_object();
                raise_exception_jit(exc as i64);
                return exc as i64;
            }
        }
    };

    let mut final_exc: pyre_object::PyObjectRef = unsafe {
        if pyre_object::is_exception(exc) {
            exc
        } else if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
            if frame_ctx.is_null() {
                pyre_interpreter::call::set_last_exec_ctx(saved_ctx);
                let err =
                    PyError::runtime_error("raise helper missing current frame").to_exc_object();
                raise_exception_jit(err as i64);
                return err as i64;
            }
            let result = {
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                pyre_interpreter::call::call_function_impl_result(exc, &[])
            };
            match result {
                Ok(obj) if pyre_object::is_exception(obj) => obj,
                Ok(_) => {
                    PyError::type_error("exceptions must derive from BaseException").to_exc_object()
                }
                Err(err) => err.to_exc_object(),
            }
        } else {
            PyError::type_error("exceptions must derive from BaseException").to_exc_object()
        }
    };

    pyre_interpreter::call::set_last_exec_ctx(saved_ctx);

    if let Err(err) = attach_raise_cause(final_exc, cause) {
        final_exc = err.to_exc_object();
    }
    raise_exception_jit(final_exc as i64);
    final_exc as i64
}

/// Runtime helper for traced `PUSH_EXC_INFO`: read the per-thread
/// `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py:786` / `eval.rs:1220-1229` semantics (save the
/// previous sys_exc_info before `CURRENT_EXCEPTION` is overwritten).
pub(crate) extern "C" fn trace_get_current_exception_jit() -> i64 {
    pyre_interpreter::eval::get_current_exception() as i64
}

/// Runtime helper for traced `PUSH_EXC_INFO` / `POP_EXCEPT`: write the
/// per-thread `CURRENT_EXCEPTION` slot so the compiled bridge preserves
/// `pyopcode.py:786/:778` / `eval.rs:1220-1229 / :1243-1249` semantics.
pub(crate) extern "C" fn trace_set_current_exception_jit(exc: i64) {
    pyre_interpreter::eval::set_current_exception(exc as pyre_object::PyObjectRef);
}
use pyre_interpreter::eval::{attach_raise_cause, normalize_raise_cause};
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_interpreter::{
    DictStorage, OpcodeStepExecutor, PyError, SharedOpcodeHandler, call_function,
    decode_instruction_at, execute_opcode_step, function_get_globals, is_builtin_code, is_function,
    range_iter_continues,
};

use pyre_object::PyObjectRef;
use pyre_object::listobject::w_list_getitem;
use pyre_object::pyobject::{
    FLOAT_TYPE, INT_TYPE, LIST_TYPE, PyType, TUPLE_TYPE, is_float, is_int, is_list, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::tupleobject::w_tuple_getitem;
use pyre_object::{
    PY_NULL, w_list_can_append_without_realloc, w_list_is_inline_storage, w_list_len,
    w_list_uses_float_storage, w_list_uses_int_storage, w_list_uses_object_storage, w_tuple_len,
};

use crate::descr::{
    dict_storage_values_len_descr, dict_storage_values_ptr_descr, float_floatval_descr,
    int_intval_descr, list_strategy_descr, ob_type_descr, w_float_size_descr, w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_DEBUGDATA_OFFSET, PYFRAME_LASTBLOCK_OFFSET, PYFRAME_PYCODE_OFFSET,
};
use crate::helpers::TraceHelperAccess;

/// Build the red `frame` argument for PyPy-style recursive CALL_ASSEMBLER.
///
/// RPython `pyjitpl.py:3589-3609 direct_assembler_call` records the portal
/// reds (`frame`, `ec`) directly. For the narrow fib-style case we can emit
/// the callee `PyFrame` as ordinary trace IR; cases that need closure or
/// debugdata setup keep the existing opaque helper fallback.
///
/// Returns `(frame, drop_needed)`. `drop_needed` is true only for the legacy
/// arena helper path; trace-visible frames are GC-owned.
fn emit_call_assembler_callee_frame(
    this: &mut MIFrame,
    ctx: &mut TraceCtx,
    callable: OpRef,
    args: &[OpRef],
    concrete_callable: PyObjectRef,
    w_callee_code: *const (),
    callee_code: &CodeObject,
    is_self_recursive: bool,
    self_recursive_raw_int_arg: Option<OpRef>,
) -> Result<(OpRef, bool), PyError> {
    if is_self_recursive && args.len() == 1 {
        if let Some(raw_arg) = self_recursive_raw_int_arg {
            let nlocals = callee_code.varnames.len();
            let ncells = pyre_interpreter::ncells(callee_code);
            let max_stack = callee_code.max_stackdepth as usize;
            let callee_globals = unsafe { function_get_globals(concrete_callable) };
            let stores_global = unsafe {
                pyre_interpreter::w_code_frame_stores_global(
                    w_callee_code as PyObjectRef,
                    callee_globals,
                )
            };
            if ncells == 0 && !stores_global {
                let pycode_const = ctx.const_ref(w_callee_code as i64);
                let w_globals_const = ctx.const_ref(callee_globals as i64);
                let ec = this.ensure_execution_context(ctx);
                let frame = crate::helpers::emit_new_pyframe_inline_self_recursive(
                    ctx,
                    raw_arg,
                    nlocals + ncells + max_stack,
                    nlocals + ncells,
                    pycode_const,
                    w_globals_const,
                    ec,
                );
                return Ok((frame, false));
            }
        }
    }

    if args.len() == 1 {
        let (helper, helper_arg_types, helper_args) = if is_self_recursive {
            if let Some(raw_arg) = self_recursive_raw_int_arg {
                let (helper, helper_arg_types) = one_arg_callee_frame_helper(Type::Int, true);
                (helper, helper_arg_types, vec![this.frame(), raw_arg])
            } else {
                let (helper, helper_arg_types) =
                    one_arg_callee_frame_helper(this.value_type(args[0]), true);
                (helper, helper_arg_types, vec![this.frame(), args[0]])
            }
        } else {
            let (helper, helper_arg_types) =
                one_arg_callee_frame_helper(this.value_type(args[0]), false);
            (
                helper,
                helper_arg_types,
                vec![this.frame(), callable, args[0]],
            )
        };
        let frame = ctx.call_ref_typed(helper, &helper_args, &helper_arg_types);
        return Ok((frame, true));
    }

    if let Some(frame_helper) = (crate::callbacks::get().callee_frame_helper)(args.len()) {
        let mut helper_args = if is_self_recursive {
            vec![this.frame()]
        } else {
            vec![this.frame(), callable]
        };
        helper_args.extend_from_slice(args);
        let helper_arg_types = frame_callable_arg_types(args.len());
        let frame = ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types);
        return Ok((frame, true));
    }

    Err(PyError::type_error(
        "call_assembler: no frame helper for nargs",
    ))
}

/// pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity helper.
///
/// `PyreSym.vable_*` is a pyre-only parallel symbolic cache that
/// RPython does not have — RPython's `metainterp.virtualizable_boxes`
/// is the single canonical source for vable static state.  Setfield-
/// vable opcodes (`_opimpl_setfield_vable`) and the JUMP-time
/// writeback path (`flush_to_frame`) publish into the boxes shadow so
/// future readers (`gen_writeback_vable_to_heap`, JUMP-arg dedup)
/// observe the same identity as `s.vable_*`.  Callers gate on
/// `s.owns_virtualizable_shadow()` before calling — upstream
/// `_opimpl_setfield_vable` short-circuits on
/// `_nonstandard_virtualizable` so callee inline frames never reach
/// the `metainterp.virtualizable_boxes[index] = valuebox` write, and
/// the call-site gate is pyre's analog (callee inline syms allocated
/// via `PyreSym::new_uninit` keep `vable_array_base` /
/// `bridge_local_oprefs` `None` so `owns_virtualizable_shadow()`
/// returns false for them).
///
/// Snapshot capture (`flush_to_frame_for_guard`) intentionally does
/// NOT mirror into the shared `ctx.virtualizable_boxes`.  Upstream
/// `rpython/jit/metainterp/pyjitpl.py:2586 capture_resumedata` reads
/// `metainterp.virtualizable_boxes` and hands it to
/// `rpython/jit/metainterp/opencoder.py:718
/// _list_of_boxes_virtualizable(boxes)` with no fallback heap source.
/// Pyre matches that single-source model in spirit for the two
/// per-opcode-advancing fields: `last_instr` / `valuestackdepth` are
/// rewritten in `s.vable_*` to their pre-opcode value at `orgpc - 1`
/// before the snapshot is built.  The other four scalars (`pycode`,
/// `debugdata`, `lastblock`, `w_globals`) keep whatever OpRef
/// `init_vable_indices` seeded at trace start because the tracer
/// never reaches their mutators under CPython 3.14 bytecode
/// (`pycode` / `w_globals`: only `pyframe.rs::frame_reinit`;
/// `debugdata`: only `getorcreate_debug_data` on debug paths;
/// `lastblock`: only `pyopcode.py:1268
/// SETUP_FINALLY/SETUP_EXCEPT/POP_BLOCK` which CPython 3.14 no
/// longer emits — try/except/finally goes through the zero-cost
/// `co_exceptiontable` consulted only on raise).  Convergence to
/// RPython's pure single-source model requires emitting
/// `_opimpl_setfield_vable` for those handlers if/when they are
/// re-introduced, after which the heap remains authoritative through
/// `metainterp.virtualizable_boxes` rather than through the snapshot
/// reader's own state.
///
/// The snapshot does not mirror its `s.vable_*` overrides into the
/// shared shadow because the shared shadow is the JUMP/JIT-time view
/// (live virtualizable values that `gen_writeback_vable_to_heap` and
/// `close_loop_args_at` consume), while `s.vable_last_instr/vsd`
/// carry the pre-opcode override that the snapshot reader needs.
/// The two stores stay distinct deliberately: `record_branch_guard`
/// saves `s.vable_last_instr/vsd` before flushing and restores them
/// after the snapshot is built, but the shared shadow has no
/// symmetric save/restore — mirroring the override there would leak
/// the pre-opcode value into the JUMP path.
///
/// `static_field_name` matches the canonical PyFrame virtualizable
/// spec at `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`
/// (`last_instr`, `pycode`, `valuestackdepth`, `debugdata`,
/// `lastblock`, `w_globals`).
///
/// No-op when `virtualizable_boxes` is not yet seeded
/// (non-virtualizable trace, or before `init_virtualizable_boxes`).
fn mirror_vable_static_to_boxes(
    ctx: &mut TraceCtx,
    static_field_name: &str,
    opref: OpRef,
    concrete: Value,
) {
    if !ctx.has_virtualizable_boxes() {
        return;
    }
    let idx = ctx
        .virtualizable_info()
        .and_then(|info| info.static_field_index_by_name(static_field_name));
    if let Some(idx) = idx {
        ctx.set_virtualizable_entry_at(idx, opref, concrete);
    }
}

impl MIFrame {
    fn active_execution_context(&self) -> *const pyre_interpreter::PyExecutionContext {
        let exec_ctx = self.sym().concrete_execution_context;
        if !exec_ctx.is_null() {
            return exec_ctx;
        }
        if self.concrete_frame_addr != 0 {
            let frame = unsafe {
                &*(self.concrete_frame_addr as *const pyre_interpreter::pyframe::PyFrame)
            };
            return frame.execution_context;
        }
        std::ptr::null()
    }

    /// Get the concrete return value from the frame's stack top.
    fn concrete_stack_value_at_return(&self) -> Option<PyObjectRef> {
        // MIFrame Box tracking: read from concrete_stack
        let s = self.sym();
        if s.valuestackdepth > 0 {
            let v = s.concrete_value_at(s.valuestackdepth - 1);
            if !v.is_null() {
                return Some(v.to_pyobj());
            }
        }
        None
    }

    pub(crate) fn next_instruction_consumes_comparison_truth(&self) -> bool {
        let code = unsafe { &*(*self.sym().jitcode).raw_code() };
        // RPython optimize_goto_if_not works on the semantic successor,
        // not on bytecode trivia like EXTENDED_ARG/NOT_TAKEN/CACHE.
        let mut pc = self.fallthrough_pc;
        loop {
            match decode_instruction_at(code, pc) {
                Some((instruction, _))
                    if instruction_is_trivia_between_compare_and_branch(instruction) =>
                {
                    pc += 1
                }
                Some((instruction, _)) => {
                    return instruction_consumes_comparison_truth(instruction);
                }
                None => return false,
            }
        }
    }

    pub fn from_sym(
        ctx: &mut TraceCtx,
        sym: &mut PyreSym,
        concrete_frame: usize,
        fallthrough_pc: usize,
        opcode_start_pc: usize,
    ) -> Self {
        // sym was initialized when its owning MetaInterpFrame was pushed
        // (trace.rs root push / metainterp.rs perform_call). MIFrame is a
        // borrowed per-instruction view; no re-initialization here.
        // RPython pyjitpl.py: orgpc = opcode start PC passed to each handler.
        let orgpc = opcode_start_pc;
        Self {
            ctx,
            sym,
            fallthrough_pc,
            concrete_frame_addr: concrete_frame,
            orgpc,
            pre_opcode_registers_r: None,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
        }
    }

    #[doc(hidden)]
    pub fn capture_current_fail_args(&mut self) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| this.current_fail_args(ctx))
    }

    #[doc(hidden)]
    pub fn capture_guard_class(
        &mut self,
        obj: OpRef,
        expected_type: *const pyre_object::pyobject::PyType,
    ) {
        self.with_ctx(|this, ctx| this.guard_class(ctx, obj, expected_type));
    }

    #[doc(hidden)]
    pub fn capture_trace_guarded_int_payload(&mut self, int_obj: OpRef) -> OpRef {
        self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj))
    }

    #[doc(hidden)]
    pub fn capture_record_branch_guard(
        &mut self,
        exitvalue: OpRef,
        condition: OpRef,
        branch_concrete: bool,
        next_instr: usize,
    ) {
        self.with_ctx(|this, ctx| {
            this.record_branch_guard(ctx, exitvalue, condition, branch_concrete, next_instr);
        });
    }

    #[doc(hidden)]
    pub fn capture_generate_guard(&mut self, opcode: OpCode, args: &[OpRef]) {
        self.with_ctx(|this, ctx| this.generate_guard(ctx, opcode, args));
    }

    #[doc(hidden)]
    pub fn capture_concrete_branch_truth_for_value(
        &mut self,
        value: OpRef,
        concrete_val: pyre_object::PyObjectRef,
    ) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(value, concrete_val)
    }

    #[doc(hidden)]
    pub fn capture_trace_dynamic_list_index(
        &mut self,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        self.with_ctx(|this, ctx| this.trace_dynamic_list_index(ctx, key, len, concrete_key))
    }

    #[doc(hidden)]
    pub fn capture_generated_list_getitem_by_strategy(
        &mut self,
        list: OpRef,
        key: OpRef,
        concrete_key: i64,
        strategy_id: i64,
    ) -> OpRef {
        self.with_ctx(|this, ctx| {
            crate::generated_list_getitem_by_strategy(
                this,
                ctx,
                list,
                key,
                concrete_key,
                strategy_id,
            )
        })
    }

    #[doc(hidden)]
    pub fn capture_list_append_value(
        &mut self,
        list: OpRef,
        value: OpRef,
        concrete_list: pyre_object::PyObjectRef,
        concrete_value: pyre_object::PyObjectRef,
    ) -> Result<(), PyError> {
        self.list_append_value(list, value, concrete_list, concrete_value)
    }

    #[doc(hidden)]
    pub fn capture_direct_len_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: pyre_object::PyObjectRef,
    ) -> Result<OpRef, PyError> {
        self.direct_len_value(callable, value, concrete_value)
    }

    #[doc(hidden)]
    pub fn capture_iter_next_value(
        &mut self,
        iter: OpRef,
        concrete_iter: pyre_object::PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        self.iter_next_value(iter, concrete_iter)
    }

    #[doc(hidden)]
    pub fn capture_close_loop_args_at(&mut self, target_pc: Option<usize>) -> Vec<OpRef> {
        self.with_ctx(|this, ctx| this.close_loop_args_at(ctx, target_pc))
    }

    #[doc(hidden)]
    pub fn symbolic_nlocals(&self) -> usize {
        self.sym().nlocals
    }

    #[doc(hidden)]
    pub fn symbolic_valuestackdepth(&self) -> usize {
        self.sym().valuestackdepth
    }

    #[doc(hidden)]
    pub fn symbolic_registers_r(&self) -> &[OpRef] {
        &self.sym().registers_r
    }

    #[doc(hidden)]
    pub fn capture_value_type(&self, opref: OpRef) -> Type {
        self.value_type(opref)
    }

    pub(crate) fn ctx(&mut self) -> &mut TraceCtx {
        unsafe { &mut *self.ctx }
    }

    pub(crate) fn with_ctx<R>(&mut self, f: impl FnOnce(&mut Self, &mut TraceCtx) -> R) -> R {
        let ctx = self.ctx;
        unsafe { f(self, &mut *ctx) }
    }

    #[inline]
    pub(crate) fn sym(&self) -> &PyreSym {
        unsafe { &*self.sym }
    }

    #[inline]
    pub(crate) fn sym_mut(&mut self) -> &mut PyreSym {
        unsafe { &mut *self.sym }
    }

    pub(crate) fn frame(&self) -> OpRef {
        self.sym().frame
    }

    /// `pypy/module/pypyjit/interp_jit.py:67 reds = ['frame', 'ec']` requires
    /// every CALL_ASSEMBLER red-args list and JUMP-args list to carry ec.
    /// Normal trace setup seeds `sym.execution_context`; this recovery keeps
    /// adapter paths from passing OpRef::NONE as the ec red.
    pub(crate) fn ensure_execution_context(&mut self, ctx: &mut TraceCtx) -> OpRef {
        let ec = self.sym().execution_context;
        if !ec.is_none() {
            return ec;
        }
        let recovered = ctx.record_op_with_descr(
            majit_ir::OpCode::GetfieldGcR,
            &[self.frame()],
            crate::descr::pyframe_execution_context_descr(),
        );
        self.sym_mut().execution_context = recovered;
        recovered
    }

    #[inline]
    fn capture_pre_opcode_state(&mut self) {
        let s = self.sym();
        let prefix_len = s.valuestackdepth.min(s.registers_r.len());
        self.pre_opcode_registers_r = Some(s.registers_r[..prefix_len].to_vec());
    }

    #[inline]
    fn clear_pre_opcode_state(&mut self) {
        self.pre_opcode_registers_r = None;
    }

    fn materialize_fail_arg_slot(
        &mut self,
        ctx: &mut TraceCtx,
        slot: OpRef,
        slot_type: Type,
        abs_idx: usize,
    ) -> OpRef {
        if !slot.is_none() {
            return slot;
        }
        let concrete_value = self.concrete_at(abs_idx).unwrap_or(PY_NULL);
        let typed_value = extract_concrete_typed_value(slot_type, concrete_value);
        fail_arg_opref_for_typed_value(ctx, typed_value)
    }

    /// `pyjitpl.py:177` `get_list_of_active_boxes` parity. Returns
    /// compact register boxes for live registers only.
    ///
    /// Both the tracer (here) and the blackhole bridge-resume decoder
    /// (`consume_one_section`, `resume.py:1381`) read the same
    /// `all_liveness` byte stream via `jitcode.get_live_vars_info(pc,
    /// op_live)` (`jitcode.py:82-93`) and iterate the per-bank register
    /// indices with `LivenessIterator` (`liveness.py:168-201`). One
    /// source, same order.
    fn get_list_of_active_boxes(
        &mut self,
        ctx: &mut TraceCtx,
        in_a_call: bool,
        after_residual_call: bool,
    ) -> Vec<OpRef> {
        // pyjitpl.py:194: in_a_call or after_residual_call → self.pc
        let live_pc = if in_a_call || after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        // resume.py:1045 consume_one_section invariant: every register
        // reported as live must be reachable via a valid OpRef. RPython
        // trivially satisfies this because every read populates
        // `registers_r[i]`. pyre's `registers_r` is the unified
        // abstract register file — locals occupy `[..nlocals]` and the
        // live stack tail occupies `[nlocals..nlocals+stack_only]`
        // (pyjitpl.py:70-78 MIFrame parity). A live register that the
        // trace has not yet produced (forward-live local across a
        // superinstruction edge, live stack slot resurrected after a
        // guard backtrack) keeps `OpRef::NONE`, poisoning the guard's
        // fail_args. Mirror RPython's invariant by forcing lazy init
        // for every live register via the same `_opimpl_getarrayitem
        // _vable` mirror read that LOAD_FAST uses (load_local_value
        // at trace_opcode.rs:660), BEFORE snapshotting registers_r
        // below. Source for the live indices is the same packed
        // `all_liveness` byte stream (`jitcode.get_live_vars_info(pc,
        // op_live)` at `jitcode.py:82-93`) that resume.py uses at
        // decode time — pyjitpl.py:218-225 `get_list_of_active_boxes`
        // analog, walking the full live register-file set.
        #[derive(Clone, Copy)]
        enum LiveBank {
            Int,
            Ref,
            Float,
        }
        impl LiveBank {
            fn ty(self) -> Type {
                match self {
                    LiveBank::Int => Type::Int,
                    LiveBank::Ref => Type::Ref,
                    LiveBank::Float => Type::Float,
                }
            }
        }

        let jitcode_ptr_pre = self.sym().jitcode;
        let live_regs_for_banks: Vec<(LiveBank, usize)> = unsafe {
            let jc = &*jitcode_ptr_pre;
            // Skeleton payload (no `pc_map` yet) → skip the lazy-load
            // preamble; the main path's skeleton-fallback branch
            // handles the same case.
            //
            // Portal-bridge (G.4.4-encoder.2/3): the portal jitcode
            // has no per-Python-PC `pc_map` because user opcodes are
            // dispatched by canonical portal arms at runtime.  The
            // RPython orthodox encoder (`pyjitpl.py:218-225`) reads
            // every live register unconditionally.  Pyre's portal-
            // bridge install seeds `metadata.stack_base` (G.3h) +
            // `depth_at_py_pc[pc]` (G.4.2) so the live ref slots at
            // this PC span `0..stack_base + depth` — locals + cells +
            // operand-stack tail.  Slice 3 covers the full range; the
            // lazy-load preamble (lines 555+) routes each through
            // `load_local_value` whose vable-shadow read (line 1218)
            // works for any flat slot in `vb[NUM_VABLE_SCALARS..
            // NUM_VABLE_SCALARS + nlocals + ncells + stackdepth]`.
            // The semantic_ref_slot_for_reg_color in-loop special-
            // case below provides identity mapping for portal-bridge
            // since no regalloc happens (color == slot index).
            if jc.payload.metadata.pc_map.is_empty() {
                if jc.payload.is_portal_bridge() {
                    let stack_base = jc.payload.metadata.stack_base;
                    let depth = jc
                        .payload
                        .metadata
                        .depth_at_py_pc
                        .get(live_pc)
                        .copied()
                        .unwrap_or(0) as usize;
                    (0..stack_base + depth)
                        .map(|idx| (LiveBank::Ref, idx))
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                // RPython `pyjitpl.py:218-225` reads each liveness bank
                // from its matching register file. Pyre's unified semantic
                // stack can hold an OpRef before the kind bank has been
                // populated, so collect every listed bank/index and complete
                // the matching bank immediately before the direct snapshot.
                let jit_pc = jc.payload.metadata.pc_map[live_pc];
                let op_live = crate::state::op_live();
                let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
                let all_liveness = crate::state::liveness_info_snapshot();
                if off + 2 >= all_liveness.len() {
                    Vec::new()
                } else {
                    let length_i = all_liveness[off] as u32;
                    let length_r = all_liveness[off + 1] as u32;
                    let length_f = all_liveness[off + 2] as u32;
                    let mut cursor = off + 3;
                    let mut out: Vec<(LiveBank, usize)> =
                        Vec::with_capacity((length_i + length_r + length_f) as usize);
                    use majit_translate::liveness::LivenessIterator;
                    if length_i != 0 {
                        let mut it = LivenessIterator::new(cursor, length_i, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Int, reg_idx as usize));
                        }
                        cursor = it.offset;
                    }
                    if length_r != 0 {
                        let mut it = LivenessIterator::new(cursor, length_r, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Ref, reg_idx as usize));
                        }
                        cursor = it.offset;
                    }
                    if length_f != 0 {
                        let mut it = LivenessIterator::new(cursor, length_f, &all_liveness);
                        while let Some(reg_idx) = it.next() {
                            out.push((LiveBank::Float, reg_idx as usize));
                        }
                    }
                    out
                }
            }
        };
        let (nlocals, valid_stack_only, jitcode_ptr, stack_slot_color_map, is_portal_bridge) = {
            let s = self.sym();
            let valid_stack_only = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r.len().saturating_sub(s.nlocals)
            } else {
                s.valuestackdepth.saturating_sub(s.nlocals)
            };
            let (stack_slot_color_map, is_portal_bridge) = if s.jitcode.is_null() {
                (Vec::new(), false)
            } else {
                unsafe {
                    let jc = &*s.jitcode;
                    (
                        jc.payload.metadata.stack_slot_color_map.clone(),
                        jc.payload.is_portal_bridge(),
                    )
                }
            };
            (
                s.nlocals,
                valid_stack_only,
                s.jitcode,
                stack_slot_color_map,
                is_portal_bridge,
            )
        };
        let mut bank_materializations: Vec<(LiveBank, usize, OpRef)> =
            Vec::with_capacity(live_regs_for_banks.len());
        for (bank, idx) in live_regs_for_banks {
            // G.4.4-encoder.3: portal-bridge has no regalloc, so
            // colors == slot indices (identity).  Bypass
            // `semantic_ref_slot_for_reg_color`'s color-map search
            // and stack_color_map fallback (which would only return
            // identity for `idx < nlocals`, dropping cells + stack).
            let slot_idx_opt = if is_portal_bridge {
                Some(idx)
            } else {
                crate::state::semantic_ref_slot_for_reg_color(
                    nlocals,
                    valid_stack_only,
                    &stack_slot_color_map,
                    idx,
                )
            };
            let Some(slot_idx) = slot_idx_opt else {
                continue;
            };
            // `load_local_value` errors when `slot_idx >= registers_r.len()`;
            // a live semantic slot beyond the current register-file
            // growth needs resizing before the mirror read can land the
            // value.
            {
                let s = self.sym_mut();
                if slot_idx >= s.registers_r.len() {
                    s.registers_r.resize(slot_idx + 1, OpRef::NONE);
                }
            }
            let cur = self
                .sym()
                .registers_r
                .get(slot_idx)
                .copied()
                .unwrap_or(OpRef::NONE);
            if cur == OpRef::NONE {
                let _ = MIFrame::load_local_value(self, ctx, slot_idx);
            }
            let semantic_value = self
                .pre_opcode_registers_r
                .as_ref()
                .and_then(|pre_r| pre_r.get(slot_idx).copied())
                .filter(|value| !value.is_none())
                .or_else(|| self.sym().registers_r.get(slot_idx).copied())
                .unwrap_or(OpRef::NONE);
            let bank_value = match bank {
                LiveBank::Ref => {
                    self.materialize_fail_arg_slot(ctx, semantic_value, Type::Ref, slot_idx)
                }
                LiveBank::Int | LiveBank::Float if self.value_type(semantic_value) == bank.ty() => {
                    semantic_value
                }
                LiveBank::Int | LiveBank::Float => {
                    self.materialize_fail_arg_slot(ctx, OpRef::NONE, bank.ty(), slot_idx)
                }
            };
            bank_materializations.push((bank, idx, bank_value));
        }
        let (registers_i, registers_r_bank, registers_r_semantic, registers_f) = {
            let s = self.sym();
            // Stage 3.4 Phase B: unified abstract register file view.
            // When a guard is being captured mid-opcode, read from
            // `pre_opcode_registers_r` (the full snapshot of
            // `registers_r` at opcode start). Otherwise read the live
            // `registers_r`. Both variants share a single indexing rule
            // so `live_{i,r,f}_regs` indices (which live in the
            // stack_base=nlocals register space) can be resolved with
            // one lookup instead of the legacy
            // `idx < nlocals ? locals : stack` split.
            //
            // Phase A dual-writes grow `registers_r` monotonically on
            // stack pushes; pop does not shrink it. Bound the view to
            // the valid locals + live stack_only range so stale slots
            // above the current (or pre-opcode) stack depth cannot
            // surface as active OpRefs. This matches the OLD
            // `stack_values.len()` bound on the
            // `stack_values[idx - nlocals]` read path.
            //
            // Portal-bridge (G.4.4-encoder.3): the trace's
            // `valuestackdepth` does not track user-bytecode stack
            // depth (the canonical portal jitcode owns the trace's
            // stack tracker, not user opcodes), so the per-CodeObject
            // bound `nlocals + valid_stack_only` undercounts.  The
            // metadata-derived `stack_base + depth_at_py_pc[pc]`
            // (G.3h + G.4.2) is the correct bound for portal-bridge
            // — symmetric with the encoder count side
            // (`state::frame_value_count_at` portal-bridge branch).
            let source_len = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r.len()
            } else {
                s.registers_r.len()
            };
            let valid_len = if is_portal_bridge {
                let payload = unsafe { &(&*jitcode_ptr).payload };
                let stack_base = payload.metadata.stack_base;
                let depth = payload
                    .metadata
                    .depth_at_py_pc
                    .get(live_pc)
                    .copied()
                    .unwrap_or(0) as usize;
                (stack_base + depth).min(source_len)
            } else {
                (s.nlocals + valid_stack_only).min(source_len)
            };
            let mut registers_i = s.registers_i.clone();
            let mut registers_r_bank = s.registers_r.clone();
            let mut registers_f = s.registers_f.clone();
            for &(bank, reg_idx, value) in &bank_materializations {
                match bank {
                    LiveBank::Int => {
                        if reg_idx >= registers_i.len() {
                            registers_i.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_i[reg_idx] = value;
                    }
                    LiveBank::Ref => {
                        if reg_idx >= registers_r_bank.len() {
                            registers_r_bank.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_r_bank[reg_idx] = value;
                    }
                    LiveBank::Float => {
                        if reg_idx >= registers_f.len() {
                            registers_f.resize(reg_idx + 1, OpRef::NONE);
                        }
                        registers_f[reg_idx] = value;
                    }
                }
            }
            let mut registers_r_semantic: Vec<OpRef> =
                if let Some(ref pre_r) = self.pre_opcode_registers_r {
                    pre_r[..valid_len.min(pre_r.len())].to_vec()
                } else {
                    s.registers_r[..valid_len.min(s.registers_r.len())].to_vec()
                };
            if in_a_call {
                if let Some(result_idx) = self.pending_result_stack_idx {
                    let abs_idx = s.nlocals + result_idx;
                    match self.pending_result_type.unwrap_or(Type::Ref) {
                        Type::Int => {
                            if result_idx >= registers_i.len() {
                                registers_i.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_i[result_idx] = ctx.const_int(0);
                        }
                        Type::Ref => {
                            let null_ref = ctx.const_ref(pyre_object::PY_NULL as i64);
                            if abs_idx < registers_r_semantic.len() {
                                registers_r_semantic[abs_idx] = null_ref;
                            }
                            // PyPy uses `_result_argcode` plus the bytecode
                            // dst register to clear the typed result bank.
                            // Pyre's Python-frame path receives a semantic
                            // stack depth, so Ref must be translated through
                            // the per-jitcode stack color map before touching
                            // the bank used by packed liveness.
                            let color_idx_opt = if is_portal_bridge {
                                Some(abs_idx)
                            } else {
                                stack_slot_color_map
                                    .get(result_idx)
                                    .copied()
                                    .map(|c| c as usize)
                            };
                            if let Some(color_idx) = color_idx_opt {
                                if color_idx >= registers_r_bank.len() {
                                    registers_r_bank.resize(color_idx + 1, OpRef::NONE);
                                }
                                registers_r_bank[color_idx] = null_ref;
                            }
                        }
                        Type::Float => {
                            if result_idx >= registers_f.len() {
                                registers_f.resize(result_idx + 1, OpRef::NONE);
                            }
                            registers_f[result_idx] = ctx.const_float(0);
                        }
                        Type::Void => {}
                    }
                }
            }
            (
                registers_i,
                registers_r_bank,
                registers_r_semantic,
                registers_f,
            )
        };
        // pyjitpl.py:202-203: read the 2-byte offset from JitCode.code
        // (upstream uses `decode_offset(self.jitcode.code, pc + 1)`) and
        // then read the `[len_i][len_r][len_f]` header from the shared
        // all_liveness byte string. Pyre stores the packed bytes on its
        // MetaInterpStaticData JitCode entry; upstream stores them on
        // metainterp_sd.
        //
        // Skeleton payload (no `pc_map` yet) → fall back to the
        // pyre-jit-trace LiveVars analysis. With the call.py-parity
        // jitcode_for callback wired up, this branch only fires for
        // sentinel/null jitcodes (PyreSym::new_uninit) that never
        // reach final code emission.
        let jc = unsafe { &*jitcode_ptr };
        // Discriminate via the explicit 3-state predicates
        // (`pyjitcode.rs` module doc): PortalBridge runs the G.4.3
        // positional fallback; Skeleton panics; PerCodeObject (which
        // has a non-empty `pc_map`) falls through to the canonical
        // `pyjitpl.py:199-233` decode path below.
        if jc.payload.is_portal_bridge() {
            // Portal-bridge encoder (G.4.3 + G.4.4-encoder.2/3):
            // emit a positional Ref-typed box list of length
            // `stack_base + depth_at_py_pc[live_pc]`, covering locals
            // + cells + the live operand stack tail.
            // `metadata.stack_base = code.varnames.len() + ncells(code)`
            // (G.3h) is the absolute boundary in
            // `PyFrame.locals_cells_stack_w` between the "always live"
            // slots and the depth-dependent operand stack — the same
            // boundary `set_stack_at` (state.rs:2762) uses on the
            // decoder writeback side.
            //
            // Encoder/decoder symmetry (G.4.3 + G.4.4-encoder.3): both
            // `state::frame_value_count_at` and
            // `restore_guard_failure_values` source their counts from
            // `metadata.stack_base + depth_at_py_pc[pc]` so the
            // upstream `pyjitpl.py:177` / `resume.py:1017-1026`
            // packed-liveness invariant is preserved.  The
            // `live_reg_idxs` derivation above (line 504-516) emits
            // the same range so the lazy-load preamble (line 584-633)
            // populates `registers_r` from the vable shadow before
            // the snapshot — RPython orthodox always-box semantics
            // (`pyjitpl.py:218-225`).  Slots whose vable shadow is
            // NONE fall through to the heap-read path inside
            // `load_local_value` (line 1240-1265), so the encoder
            // emits a real OpRef for every live slot.
            //
            // Pre-G.4.4-encoder.2/3 the encoder fell back to
            // `OpRef::NONE` for slots beyond `registers_r.len()`
            // (relying on `consume_one_section` to overwrite before
            // BH deref); slice 2/3 closed both KNOWN-INCOMPLETE
            // divergences from `pyjitpl.py:177-234` (single-bank
            // read + always-box) by routing the full live range
            // through the same lazy-load mechanism per-CodeObject
            // mode uses.
            let stack_base = jc.payload.metadata.stack_base;
            let depth = jc
                .payload
                .metadata
                .depth_at_py_pc
                .get(live_pc)
                .copied()
                .unwrap_or(0) as usize;
            let target_count = stack_base + depth;
            let mut boxes = Vec::with_capacity(target_count);
            for reg in 0..target_count {
                boxes.push(
                    registers_r_semantic
                        .get(reg)
                        .copied()
                        .unwrap_or(OpRef::NONE),
                );
            }
            return boxes;
        }
        if jc.payload.is_skeleton() {
            // `CallControl.get_jitcode` drain fills pc_map before any
            // guard capture (pyjitpl.py:199 parity). Phase X-0 eliminated
            // the out-of-range-pc source. Phase X-1(a) migrated the
            // remaining guard/resume tests to the real compile path in
            // `pyre-jit`. Unconditional panic — any hit is a bug.
            panic!(
                "get_list_of_active_boxes: skeleton jitcode (pc_map empty) \
                 at live_pc={} — Phase X-0/X-1 removed all known triggers; \
                 further hits are bugs.",
                live_pc
            );
        }
        // `pyjitpl.py:199-233` parity: decode the `-live-` offset from
        // the jitcode byte stream via `jitcode.get_live_vars_info(pc,
        // op_live)` (`jitcode.py:82-93`), read the `[len_i][len_r]
        // [len_f]` header in `all_liveness`, then iterate per-bank
        // register indices with `LivenessIterator` (`liveness.py:168-
        // 201`). Register indices snapshot into `registers_r` in
        // int → ref → float bank order to match the encoder/decoder
        // contract (`all_liveness` byte layout).
        let jit_pc = jc
            .payload
            .metadata
            .pc_map
            .get(live_pc)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "get_list_of_active_boxes: no pc_map entry for live_pc={} (pc_map.len={})",
                    live_pc,
                    jc.payload.metadata.pc_map.len()
                )
            });
        let op_live = crate::state::op_live();
        let off = jc.payload.jitcode.get_live_vars_info(jit_pc, op_live);
        let all_liveness = crate::state::liveness_info_snapshot();
        assert!(
            off + 2 < all_liveness.len(),
            "get_list_of_active_boxes: liveness offset {} + header 3 bytes exceeds all_liveness length {}",
            off,
            all_liveness.len()
        );
        let length_i = all_liveness[off] as u32;
        let length_r = all_liveness[off + 1] as u32;
        let length_f = all_liveness[off + 2] as u32;
        let mut cursor = off + 3;
        let mut boxes = Vec::with_capacity((length_i + length_r + length_f) as usize);
        // `pyjitpl.py:216-233` line-by-line parity: each live register
        // is read from its kind-specific bank via direct list indexing
        // (`self.registers_i[index]` etc).  Rust slice indexing
        // bounds-checks and panics on OOB, matching Python's
        // IndexError contract — a liveness-listed index out of bank
        // range is an encoder/codewriter invariant violation, not a
        // silent NONE.
        use majit_translate::liveness::LivenessIterator;
        if length_i != 0 {
            let mut it = LivenessIterator::new(cursor, length_i, &all_liveness);
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_i[reg_idx as usize]);
            }
            cursor = it.offset;
        }
        if length_r != 0 {
            let mut it = LivenessIterator::new(cursor, length_r, &all_liveness);
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_r_bank[reg_idx as usize]);
            }
            cursor = it.offset;
        }
        if length_f != 0 {
            let mut it = LivenessIterator::new(cursor, length_f, &all_liveness);
            while let Some(reg_idx) = it.next() {
                boxes.push(registers_f[reg_idx as usize]);
            }
        }
        boxes
    }

    /// RPython Box.type parity: build fail_arg_types matching compact
    /// active_boxes length. Each box carries its own immutable type.
    /// Header layout matches `virtualizable_gen.rs:33-35` (frame +
    /// `extra_reds` + `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS`):
    /// `[frame:Ref, ec:Ref, last_instr:Int, pycode:Ref,
    ///   valuestackdepth:Int, debugdata:Ref, lastblock:Ref,
    ///   w_globals:Ref]` — line-by-line PyPy parity with
    /// `interp_jit.py:25-31` plus `interp_jit.py:67 reds = ['frame', 'ec']`.
    fn build_fail_arg_types_for_active_boxes(&self, active_boxes: &[OpRef]) -> Vec<Type> {
        let mut types = crate::virtualizable_gen::virt_live_value_types(0);
        for &opref in active_boxes {
            types.push(self.value_type(opref));
        }
        types
    }

    pub(crate) fn value_type(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        // Match RPython Box.type semantics: the type is an intrinsic property
        // of the Box (history.py:220,262,308). Prefer the register-file slot
        // type because symbolic entries for function-entry traces / bridge
        // recovery carry the typed-local view. `registers_r` is the unified
        // abstract register file: `[..nlocals]` are locals, matched against
        // `symbolic_local_types`; `[nlocals..nlocals+stack_only]` are the
        // stack tail, matched against `symbolic_stack_types`. Falls back to
        // the recorder's inputarg/constant/op.result_type() when the OpRef
        // does not appear in the register file.
        let sym = self.sym();
        let nlocals = sym.nlocals;
        let stack_only = sym.valuestackdepth.saturating_sub(sym.nlocals);
        let reg_len = sym.registers_r.len();
        if let Some(idx) = sym.registers_r.iter().position(|&slot| slot == value) {
            if idx < nlocals {
                if let Some(&tp) = sym.symbolic_local_types.get(idx) {
                    return tp;
                }
            } else if idx < nlocals + stack_only && idx < reg_len {
                let stack_idx = idx - nlocals;
                if let Some(&tp) = sym.symbolic_stack_types.get(stack_idx) {
                    return tp;
                }
            }
        }
        let ctx_ref: &TraceCtx = unsafe { &*self.ctx };
        ctx_ref.get_opref_type(value).unwrap_or(Type::Ref)
    }

    /// RPython Box push: symbolic OpRef + concrete value together.
    ///
    /// Parity: the operand stack for virtualizable portal frames lives
    /// inside `locals_cells_stack_w` (virtualizable.py:86-98), a W_Root
    /// array, so every pushed slot is a Ref box. `value_type` is kept
    /// as a declared type hint but the stored OpRef is always boxed
    /// here. Callers that want to keep a raw Int/Float payload must
    /// unbox at the op site, not at push time.
    fn push_typed_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        value_type: Type,
        concrete: ConcreteValue,
    ) {
        let boxed = match value_type {
            Type::Int => wrapint(ctx, value),
            Type::Float => wrapfloat(ctx, value),
            _ => value,
        };
        let (has_vable, reg_idx, new_vsd) = {
            let s = self.sym_mut();
            let stack_idx = s.valuestackdepth.saturating_sub(s.nlocals);
            let reg_idx = s.nlocals + stack_idx;
            if reg_idx >= s.registers_r.len() {
                s.registers_r.resize(reg_idx + 1, OpRef::NONE);
            }
            s.registers_r[reg_idx] = boxed;
            if stack_idx >= s.symbolic_stack_types.len() {
                s.symbolic_stack_types.resize(stack_idx + 1, Type::Ref);
            }
            s.symbolic_stack_types[stack_idx] = Type::Ref;
            if stack_idx >= s.concrete_stack.len() {
                s.concrete_stack.resize(stack_idx + 1, ConcreteValue::Null);
            }
            s.concrete_stack[stack_idx] = concrete;
            s.valuestackdepth += 1;
            let new_vsd = s.valuestackdepth;
            (s.owns_virtualizable_shadow(), reg_idx, new_vsd)
        };
        // pyjitpl.py:1242-1247 `_opimpl_setarrayitem_vable` parity:
        //     self.metainterp.virtualizable_boxes[flat_idx] = valuebox
        // The symbolic stack slice of `locals_cells_stack_w` must stay in
        // lock-step with the virtualizable_boxes shadow — the shadow is
        // the single source of truth for `list_of_boxes_virtualizable`
        // (opencoder.py:718 `_list_of_boxes_virtualizable` parity). Stack
        // pushes mirror into the same flat layout that
        // `store_local_value` uses for locals: slot `reg_idx` →
        // `NUM_VABLE_SCALARS + reg_idx` in virtualizable_boxes.
        //
        // `concrete` is the caller's Box concrete (RPython valuebox). Ref /
        // Null concrete carries a real W_Root heap pointer, so update both
        // halves of the shadow. Int / Float concrete means pyre's lazy
        // `wrapint` / `wrapfloat` emitted a `NewWithVtable` OpRef without
        // allocating a W_IntObject / W_FloatObject yet — there is no heap
        // pointer to flush. Update only the OpRef half in that case so
        // `synchronize_virtualizable` keeps writing whatever valid W_Root
        // the PyFrame slot already held (virtualizable.py:101
        // `write_boxes` parity: every slot must be a real W_Root).
        if has_vable {
            let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + reg_idx;
            // Phase 0.5 probe-D — `MAJIT_PROBE_BRIDGE` gated. Captures
            // every push_typed_value that targets a virtualizable shadow
            // slot, so we can correlate the panic at
            // `set_virtualizable_entry_at: index N out of range for N
            // slots` with the symbolic state that produced it. Logging
            // every push instead of only the panicking one is intentional:
            // we need to see whether `valuestackdepth` was already at
            // `nlocals + max_stackdepth` (Phase 0.4 hypothesis) when the
            // failing push fired or whether it climbed there during the
            // bridge re-trace.
            if std::env::var_os("MAJIT_PROBE_BRIDGE").is_some() {
                let s = self.sym_mut();
                eprintln!(
                    "[probe-D][push_typed_value] valuestackdepth={} nlocals={} \
                     reg_idx={} flat_idx={} vable_boxes_len={:?}",
                    s.valuestackdepth,
                    s.nlocals,
                    reg_idx,
                    flat_idx,
                    ctx.virtualizable_boxes_len(),
                );
            }
            match concrete.to_ir_ref_value() {
                Some(v) => {
                    ctx.set_virtualizable_entry_at(flat_idx, boxed, v);
                }
                None => {
                    ctx.set_virtualizable_box_at(flat_idx, boxed);
                }
            }
            // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity (Slice 2,
            // Task #114). Each `setarrayitem_vable` advances the frame's
            // valuestackdepth by one (`pyframe.pushvalue`); upstream emits a
            // following `setfield_vable_i(virtualizable, vsd_descr, depth+1)`
            // through the metainterp's vable shadow.  Mirror the same advance
            // here so `s.vable_valuestackdepth` and the shadow's vsd scalar
            // slot stay current between flushes instead of lagging behind
            // until the next `flush_to_frame_for_guard` heap re-seed.
            let vsd_op = ctx.const_int(new_vsd as i64);
            self.sym_mut().vable_valuestackdepth = vsd_op;
            mirror_vable_static_to_boxes(
                ctx,
                "valuestackdepth",
                vsd_op,
                Value::Int(new_vsd as i64),
            );
        }
    }

    pub(crate) fn push_value(
        &mut self,
        _ctx: &mut TraceCtx,
        value: OpRef,
        concrete: ConcreteValue,
    ) {
        let value_type = self.value_type(value);
        self.push_typed_value(_ctx, value, value_type, concrete);
    }

    pub(crate) fn pop_value(&mut self, ctx: &mut TraceCtx) -> Result<OpRef, PyError> {
        let (value, has_vable, owns_vable, reg_idx, new_vsd) = {
            let s = self.sym_mut();
            let nlocals = s.nlocals;
            let stack_idx = s
                .valuestackdepth
                .checked_sub(nlocals + 1)
                .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace opcode"))?;
            let reg_idx = nlocals + stack_idx;
            if reg_idx >= s.registers_r.len() {
                s.registers_r.resize(reg_idx + 1, OpRef::NONE);
            }
            if s.registers_r[reg_idx] == OpRef::NONE {
                let abs_idx = reg_idx;
                let idx_const = ctx.const_int(abs_idx as i64);
                s.registers_r[reg_idx] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
            let value = s.registers_r[reg_idx];
            s.valuestackdepth -= 1;
            let new_vsd = s.valuestackdepth;
            // Task #136 partial: gating on `is_active_vable_owner`
            // intentionally does NOT cover bridges here. Unlike the other
            // four mirror sites (push_typed_value / swap_values /
            // store_local_value / finishframe_exception, all flipped to
            // `owns_virtualizable_shadow()`), clearing the bridge shadow
            // to NULL on every pop triggers a severe perf regression on
            // fib_recursive (~50x slowdown: 0.88s → 47s; answer remains
            // correct).
            //
            // Root cause (Task #138 probe): the pop-clear writes
            // `const_ref(PY_NULL)` into the vable shadow. When a bridge
            // subsequently inherits that shadow state and its IR
            // references the cleared slot (e.g. a later `SetfieldGc`
            // whose base operand resolves to the shadow entry), the
            // bridge optimizer's `ConstPtrInfo._get_info` rejects the
            // null constant with `InvalidLoop("null constant base
            // pointer")` (optimizeopt/mod.rs:4021, info.py:720-721
            // parity). Every bridge attempt aborts the same way, so
            // guard failures fall through to the blackhole and immediately
            // re-attempt compilation — trace-abort storm (~1.5k
            // InvalidLoops + 92k aborted trace attempts on
            // fib_recursive in 20s, from MAJIT_LOG + MAJIT_STATS
            // probe).
            //
            // Upstream `pyframe.py:411 popvalue_maybe_none` clears the
            // popped slot to `None` without this consequence, because
            // RPython's virtualizable layout does NOT include the
            // Python stack — the shadow has no stack-pop story. pyre's
            // stack-in-vable PRE-EXISTING-ADAPTATION puts the popped
            // slots in a structure the bridge optimizer reads as heap
            // bases, so the None-clear pattern does not translate.
            // Resolving this requires either removing the stack from
            // the vable shadow (multi-session) or teaching the bridge
            // optimizer to recognise cleared-stack-slot sentinels
            // separately from real null heap bases.
            (
                value,
                s.is_active_vable_owner,
                s.owns_virtualizable_shadow(),
                reg_idx,
                new_vsd,
            )
        };
        // pyframe.py:411-417 `popvalue_maybe_none` parity: the popped slot
        // is cleared to None in `locals_cells_stack_w`. Upstream lowers
        // this via `setarrayitem_vable_r(locals_cells_stack_w, depth,
        // None)`, which `_opimpl_setarrayitem_vable` mirrors into
        // `virtualizable_boxes`. Mirror the clear into the shadow so
        // subsequent snapshots do not pick up a stale pushed OpRef above
        // the current stack depth.
        if has_vable {
            let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + reg_idx;
            let null_opref = ctx.const_ref(pyre_object::PY_NULL as i64);
            let null_value = majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::PY_NULL as usize));
            ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
        }
        // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity (Slice 2,
        // Task #114). `pyframe.popvalue_maybe_none` decrements vsd as part
        // of the same sequence that clears the array slot; upstream emits a
        // `setfield_vable_i(virtualizable, vsd_descr, depth-1)` after the
        // setarrayitem_vable_r clear.  The bridge NULL-base issue
        // (Task #136 comment above) is specific to writing a const_ref(NULL)
        // into a Ref-typed array slot; vsd is an Int scalar, so the bridge
        // optimizer is unaffected and the gate stays at
        // `owns_virtualizable_shadow()` to keep parity with
        // `push_typed_value`.
        if owns_vable {
            let vsd_op = ctx.const_int(new_vsd as i64);
            self.sym_mut().vable_valuestackdepth = vsd_op;
            mirror_vable_static_to_boxes(
                ctx,
                "valuestackdepth",
                vsd_op,
                Value::Int(new_vsd as i64),
            );
        }
        Ok(value)
    }

    pub(crate) fn peek_value(
        &mut self,
        ctx: &mut TraceCtx,
        depth: usize,
    ) -> Result<OpRef, PyError> {
        let s = self.sym_mut();
        let nlocals = s.nlocals;
        let stack_idx = s
            .valuestackdepth
            .checked_sub(nlocals + depth + 1)
            .ok_or_else(|| pyre_interpreter::stack_underflow_error("trace peek"))?;
        let reg_idx = nlocals + stack_idx;
        if reg_idx >= s.registers_r.len() {
            s.registers_r.resize(reg_idx + 1, OpRef::NONE);
        }
        if s.registers_r[reg_idx] == OpRef::NONE {
            let abs_idx = reg_idx;
            let idx_const = ctx.const_int(abs_idx as i64);
            s.registers_r[reg_idx] =
                trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
        }
        let value = s.registers_r[reg_idx];
        Ok(value)
    }

    fn push_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        callable: OpRef,
        args: &[OpRef],
        call_pc: usize,
    ) {
        let null = ctx.const_ref(pyre_object::PY_NULL as i64);
        self.push_value(ctx, callable, ConcreteValue::Null);
        self.push_value(ctx, null, ConcreteValue::Ref(pyre_object::PY_NULL));
        for &arg in args {
            self.push_value(ctx, arg, ConcreteValue::Null);
        }
        self.sym_mut().pending_next_instr = Some(call_pc);
    }

    fn pop_call_replay_stack(
        &mut self,
        ctx: &mut TraceCtx,
        args_len: usize,
    ) -> Result<(), PyError> {
        for _ in 0..(2 + args_len) {
            let _ = self.pop_value(ctx)?;
        }
        self.sym_mut().pending_next_instr = None;
        Ok(())
    }

    pub(crate) fn swap_values(&mut self, ctx: &mut TraceCtx, depth: usize) -> Result<(), PyError> {
        let (has_vable, reg_top, reg_other) = {
            let s = self.sym_mut();
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            if depth == 0 || stack_only < depth {
                return Err(PyError::type_error("stack underflow during trace swap"));
            }
            let top_idx = stack_only - 1;
            let other_idx = stack_only - depth;
            let nlocals = s.nlocals;
            let reg_top = nlocals + top_idx;
            let reg_other = nlocals + other_idx;
            let needed = reg_top.max(reg_other) + 1;
            if s.registers_r.len() < needed {
                s.registers_r.resize(needed, OpRef::NONE);
            }
            if s.registers_r[reg_top] == OpRef::NONE {
                let idx_const = ctx.const_int(reg_top as i64);
                s.registers_r[reg_top] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
            if s.registers_r[reg_other] == OpRef::NONE {
                let idx_const = ctx.const_int(reg_other as i64);
                s.registers_r[reg_other] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
            s.registers_r.swap(reg_top, reg_other);
            if top_idx < s.symbolic_stack_types.len() && other_idx < s.symbolic_stack_types.len() {
                s.symbolic_stack_types.swap(top_idx, other_idx);
            }
            // MIFrame Box tracking: swap concrete values too
            if top_idx < s.concrete_stack.len() && other_idx < s.concrete_stack.len() {
                s.concrete_stack.swap(top_idx, other_idx);
            }
            (s.owns_virtualizable_shadow(), reg_top, reg_other)
        };
        // `virtualizable_boxes` is the single source of truth for the
        // frame's Ref array (opencoder.py:718). SWAP rewrites two slots
        // of `locals_cells_stack_w`, so swap the (OpRef, concrete) pairs
        // atomically via `virtualizable_entry_at`; reading each half
        // separately and reconstructing concrete via `concrete_of_opref`
        // would drop non-const Box identity into the sentinel fallback.
        if has_vable {
            let flat_top = crate::virtualizable_gen::NUM_VABLE_SCALARS + reg_top;
            let flat_other = crate::virtualizable_gen::NUM_VABLE_SCALARS + reg_other;
            if let (Some((op_top, val_top)), Some((op_other, val_other))) = (
                ctx.virtualizable_entry_at(flat_top),
                ctx.virtualizable_entry_at(flat_other),
            ) {
                ctx.set_virtualizable_entry_at(flat_top, op_other, val_other);
                ctx.set_virtualizable_entry_at(flat_other, op_top, val_top);
            }
        }
        Ok(())
    }

    pub(crate) fn load_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        // pyjitpl.py:1231 `_opimpl_getarrayitem_vable` (standard path):
        //     return self.metainterp.virtualizable_boxes[index]
        //
        // When the standard virtualizable is active, read the current OpRef
        // straight from the virtualizable_boxes cache (seeded by
        // initialize_virtualizable at setup_tracing, mirrored by
        // store_local_value on every STORE_FAST). This is the RPython
        // orthodox read path: no extra IR op, shadow-state only.
        //
        // Pyre's flat layout puts locals at flat indices
        // `num_static_extra_boxes .. num_static + nlocals`, i.e.
        // `NUM_VABLE_SCALARS + idx`. The scalar static fields live before
        // the array items, the standard-vable identity
        // (`virtualizable_boxes[-1]`) after.
        let vable_entry = {
            let s = self.sym();
            if s.is_active_vable_owner && s.bridge_local_oprefs.is_none() {
                let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + idx;
                ctx.virtualizable_box_at(flat_idx)
            } else {
                None
            }
        };
        if let Some(op) = vable_entry {
            let s = self.sym_mut();
            if idx >= s.registers_r.len() {
                return Err(PyError::type_error("local index out of range in trace"));
            }
            // Sync registers_r with the vable read so downstream
            // consumers that still consult the cache-field directly
            // (close_loop_args_at until Phase B migrates it) see the
            // same OpRef.
            s.registers_r[idx] = op;
            return Ok(op);
        }
        let s = self.sym_mut();
        if idx >= s.registers_r.len() {
            return Err(PyError::type_error("local index out of range in trace"));
        }
        if s.registers_r[idx] == OpRef::NONE {
            if s.bridge_local_oprefs.is_some() {
                // Bridge trace: OpRef::NONE means this local is a constant
                // or virtual from resume data, not a missing vable slot.
                // Read from the concrete frame via the locals_cells_stack_w
                // array.  RPython `virtualizable.py:85-99 read_boxes` does the
                // array-field access in two steps:
                //
                //   for _, fieldname in unroll_array_fields:
                //       lst = getattr(virtualizable, fieldname)   # :94
                //       for i in range(len(lst)):
                //           boxes.append(wrap(cpu, lst[i], ...))  # :96
                //
                // Step 1 (`getattr`) yields the array pointer; step 2
                // (`lst[i]`) is the indexed read.  pyre currently emits
                // step 1 as `OpCode::GetfieldRawI` via
                // `state.rs:frame_locals_cells_stack_array`.  The
                // upstream-orthodox emission is `GETFIELD_GC_R` because
                // `pyframe_locals_cells_stack_descr` is field 0 of
                // `PYFRAME_DESCR_GROUP` with `field_type = Type::Ref`
                // on a `PYFRAME_GC_TYPE_ID`-typed PyFrame.  The
                // cranelift backend's GC-barrier coverage for the
                // PYFRAME_DESCR_GROUP read path is incomplete — a
                // direct swap to `GetfieldGcR` SIGABRTs in
                // fib_recursive — so the swap is gated on bringing
                // that barrier support up first (separate parity
                // epic).  Step 2 (`lst[i]`) is `GETARRAYITEM_GC_R`
                // indexed off the array.  `trace_array_getitem_value`
                // uses `pyobject_gcarray_descr` (`base_size =
                // FIXED_ARRAY_ITEMS_OFFSET`) and so requires the array
                // base, not the virtualizable (PyFrame*) pointer.
                // Emit `frame_locals_cells_stack_array` to materialise
                // the array OpRef before indexing.
                let frame_ref = s.frame;
                let array_ref = crate::state::frame_locals_cells_stack_array(ctx, frame_ref);
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r[idx] = trace_array_getitem_value(ctx, array_ref, idx_const);
            } else {
                // Active vable owner whose registers_r[idx] is NONE cannot
                // exist: init_symbolic (state.rs:2618-2619) seeds
                // registers_r[idx] = OpRef::from_raw(base + idx) for every i in
                // 0..nlocals before any load_local_value runs. Reachability
                // audit (2026-04-28, MAJIT_PROBE_VABLE_FALLBACK) confirmed
                // this empirically: 0 firings across debug unit tests
                // (debug_assert!(false)) and 0 firings across 28 release
                // benchmark runs (env-gated eprintln). The remaining
                // fallthrough is the non-vable-owner path —
                // `s.locals_cells_stack_array_ref` is the callee's own
                // locals_cells_stack_w array (seeded by Stage 1 at
                // inline_function_call).
                let idx_const = ctx.const_int(idx as i64);
                s.registers_r[idx] =
                    trace_array_getitem_value(ctx, s.locals_cells_stack_array_ref, idx_const);
            }
        }
        Ok(s.registers_r[idx])
    }

    pub(crate) fn store_local_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
        value: OpRef,
        concrete: ConcreteValue,
    ) -> Result<(), PyError> {
        // RPython `_opimpl_setarrayitem_vable` (pyjitpl.py:1242-1247)
        // writes the value's Ref box directly into
        // `virtualizable_boxes[flat_idx]` AND the frame's symbolic
        // tracking. `locals_cells_stack_w` is declared as a W_Root array
        // (virtualizable.py:86-98), so the stored box must always be
        // Ref. pyre's producers satisfy this contract BEFORE reaching
        // this store: `push_typed_value` (trace_opcode.rs:471-482)
        // wraps raw Int/Float with `wrapint` / `wrapfloat` when a value
        // lands on the operand stack, so `pop_value` — and thus every
        // STORE_FAST / STORE_FAST_LOAD_FAST / STORE_FAST_STORE_FAST
        // handler (pyopcode.rs:394-430) — hands us a Ref. The earlier
        // "box on store" safety net (d2e530f3b9) is therefore redundant
        // in production; asserting the invariant here keeps any future
        // non-push producer from slipping a raw Int/Float through
        // (RPython Box.type parity — the fix belongs at the producer,
        // not the consumer).
        debug_assert_eq!(
            self.value_type(value),
            Type::Ref,
            "store_local_value: expected Ref-typed box (locals_cells_stack_w \
             is W_Root), got {:?} for {:?}. Producer must emit \
             wrapint/wrapfloat before the value reaches the symbolic \
             stack / local slot.",
            self.value_type(value),
            value,
        );
        let (has_vable, frame_ref, nlocals) = {
            let s = self.sym_mut();
            if idx >= s.registers_r.len() {
                return Err(PyError::type_error("local index out of range in trace"));
            }
            s.registers_r[idx] = value;
            if idx >= s.symbolic_local_types.len() {
                s.symbolic_local_types.resize(idx + 1, Type::Ref);
            }
            // virtualizable.py:86-98 read_boxes() parity: every item of
            // locals_cells_stack_w is Ref.
            s.symbolic_local_types[idx] = Type::Ref;
            (s.owns_virtualizable_shadow(), s.frame, s.nlocals)
        };
        // RPython pyjitpl.py:1242-1247 `_opimpl_setarrayitem_vable` parity:
        //     self.metainterp.virtualizable_boxes[flat_idx] = valuebox
        //     self.metainterp.synchronize_virtualizable()
        //
        // `valuebox` carries OpRef + concrete as a single Box. Ref / Null
        // concrete updates both halves; Int / Float concrete has no real
        // W_Root heap pointer (pyre's lazy `wrapint` / `wrapfloat`), so
        // update only the OpRef half and leave the shadow concrete at the
        // PyFrame's existing valid W_Root. virtualizable.py:101
        // `write_boxes` must see real boxed W_Root values in every slot.
        if has_vable && idx < nlocals {
            let _ = frame_ref;
            // NUM_VABLE_SCALARS = 6 (vable static field count, excluding
            // both the frame-identity slot and any non-vable extra reds).
            // virtualizable_boxes layout is [scalars.., array_items..,
            // vable_ref], so local idx maps to `NUM_VABLE_SCALARS + idx`.
            let flat_idx = crate::virtualizable_gen::NUM_VABLE_SCALARS + idx;
            match concrete.to_ir_ref_value() {
                Some(v) => {
                    ctx.set_virtualizable_entry_at(flat_idx, value, v);
                }
                None => {
                    ctx.set_virtualizable_box_at(flat_idx, value);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn load_namespace_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
    ) -> Result<OpRef, PyError> {
        // pyjitpl.py:1075-1089: quasi-immutable field pattern.
        // Module globals are effectively quasi-immutable — they rarely change
        // during hot loops. Emit GUARD_NOT_INVALIDATED on first global access
        // so compiled code is invalidated if globals mutate.
        if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
            self.generate_guard(ctx, OpCode::GuardNotInvalidated, &[]);
        }
        let frame = self.sym().frame;
        let namespace = frame_get_namespace(ctx, frame);
        let len = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            dict_storage_values_len_descr(),
        );
        self.guard_len_gt_index(ctx, len, idx);
        // Post-L1: `dict_storage_values_ptr_descr()` now points at
        // `*mut ItemsBlock` (not the legacy fat-pointer items base).
        // Add `ITEMS_BLOCK_ITEMS_OFFSET` to recover the raw items-base
        // pointer that `trace_raw_array_getitem_value` expects.
        let block = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            dict_storage_values_ptr_descr(),
        );
        let offset = ctx.const_int(pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET as i64);
        let values = ctx.record_op(OpCode::IntAdd, &[block, offset]);
        let idx_const = ctx.const_int(idx as i64);
        Ok(trace_raw_array_getitem_value(ctx, values, idx_const))
    }

    pub(crate) fn store_namespace_value(
        &mut self,
        ctx: &mut TraceCtx,
        idx: usize,
        value: OpRef,
    ) -> Result<(), PyError> {
        let frame = self.sym().frame;
        let namespace = frame_get_namespace(ctx, frame);
        let len = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            dict_storage_values_len_descr(),
        );
        self.guard_len_gt_index(ctx, len, idx);
        // Post-L1: `dict_storage_values_ptr_descr()` now returns a
        // `*mut ItemsBlock`. Recover the items-base pointer via
        // `+ ITEMS_BLOCK_ITEMS_OFFSET` before emitting the raw
        // setitem.
        let block = ctx.record_op_with_descr(
            OpCode::GetfieldRawI,
            &[namespace],
            dict_storage_values_ptr_descr(),
        );
        let offset = ctx.const_int(pyre_object::object_array::ITEMS_BLOCK_ITEMS_OFFSET as i64);
        let values = ctx.record_op(OpCode::IntAdd, &[block, offset]);
        let idx_const = ctx.const_int(idx as i64);
        trace_raw_array_setitem_value(ctx, values, idx_const, value);
        Ok(())
    }

    pub(crate) fn set_next_instr(&mut self, _ctx: &mut TraceCtx, target: usize) {
        self.sym_mut().pending_next_instr = Some(target);
    }

    pub(crate) fn fallthrough_pc(&self) -> usize {
        self.fallthrough_pc
    }

    /// Set pending_next_instr for trace advancement (step_root_frame).
    /// This is the NEXT bytecode PC — used by the MetaInterp to advance.
    pub(crate) fn prepare_fallthrough(&mut self) {
        self.sym_mut().pending_next_instr = Some(self.fallthrough_pc);
    }

    /// Set the original PC for the current opcode (RPython orgpc).
    /// All guards within this opcode will use orgpc as their resume PC.
    pub(crate) fn set_orgpc(&mut self, pc: usize) {
        self.orgpc = pc;
        self.publish_last_instr_to_vable(pc);
    }

    /// pyopcode.py:170-172 `dispatch_bytecode` parity (Slice 3a, Task #115):
    ///
    /// ```python
    /// while True:
    ///     self.last_instr = intmask(next_instr)   # explicit setattr at every
    ///                                             # dispatch top
    /// ```
    ///
    /// RPython's source-level setattr is picked up by the codewriter and
    /// emitted as a `setfield_vable_i(virtualizable, last_instr_descr, ...)`
    /// jitcode op; metainterp's `_opimpl_setfield_vable` (pyjitpl.py:1188-
    /// 1199) updates `metainterp.virtualizable_boxes[last_instr_index]` and
    /// `synchronize_virtualizable()`s the heap.  pyre's tracer is itself
    /// the dispatch loop, so the explicit publish happens here at the
    /// orgpc anchor — every `set_orgpc(pc)` call is the analogue of one
    /// `self.last_instr = next_instr` setattr.
    ///
    /// Semantic mismatch (PRE-EXISTING-ADAPTATION until Slice 3b lands):
    /// pyre's `PyFrame.last_instr` carries "PC of the last completed
    /// opcode" (= `orgpc - 1`), whereas RPython's carries "PC of the
    /// opcode now starting" (= `orgpc`).  Slice 3a publishes
    /// `orgpc - 1` to keep every existing reader (`flush_to_frame_for_guard`
    /// re-seed, `get_last_lineno`, `fget_f_lasti`, blackhole resume,
    /// guard fail_args) seeing the same value they always did.  Slice 3b
    /// flips the semantic to RPython's "current opcode PC" and removes
    /// the `-1` adjustment from this publish + the four other vable-side
    /// sites (`flush_to_frame`, `flush_to_frame_for_guard`,
    /// `close_loop_args_at` target override, `build_pending_inline_frame`
    /// after-call return-point).
    fn publish_last_instr_to_vable(&mut self, pc: usize) {
        if !self.sym().owns_virtualizable_shadow() {
            return;
        }
        let last_instr_value = pc as i64 - 1;
        self.with_ctx(|frame, ctx| {
            let last_instr_op = ctx.const_int(last_instr_value);
            frame.sym_mut().vable_last_instr = last_instr_op;
            mirror_vable_static_to_boxes(
                ctx,
                "last_instr",
                last_instr_op,
                Value::Int(last_instr_value),
            );
        });
    }

    /// Update virtualizable last_instr and valuestackdepth.
    /// RPython parity: always use orgpc (opcode start PC) as the semantic
    /// next instruction, so the heap frame stores `last_instr = orgpc - 1`.
    /// The trace loop advancement uses pending_next_instr separately
    /// (in metainterp.rs step_*_frame).
    pub(crate) fn flush_to_frame(&mut self, ctx: &mut TraceCtx) {
        let resume_pc = self.orgpc;
        let frame_addr = self.concrete_frame_addr;
        let (code_ptr, debugdata, lastblock) = if frame_addr != 0 {
            unsafe {
                (
                    *((frame_addr + PYFRAME_PYCODE_OFFSET) as *const usize),
                    *((frame_addr + PYFRAME_DEBUGDATA_OFFSET) as *const usize),
                    *((frame_addr + PYFRAME_LASTBLOCK_OFFSET) as *const usize),
                )
            }
        } else {
            (0, 0, 0)
        };
        let ns_ptr = self.sym().concrete_namespace as i64;
        // G.4.3a: portal-bridge frames trace eval_loop_jit, so each user
        // opcode is a residual call to execute_opcode_step whose
        // valuestackdepth side-effects do NOT advance `sym.valuestackdepth`.
        // The stale symbolic value would encode `vable_valuestackdepth =
        // stack_base`, leaving the resumed PyFrame with vsd ≤ stack_base
        // and crashing the next pop. Recompute from the per-PC user-side
        // depth metadata derived in `install_portal_for` (G.4.2).
        let vsd = self
            .portal_bridge_vable_vsd(resume_pc)
            .unwrap_or_else(|| self.sym().valuestackdepth as i64);
        // virtualizable.py:86-93 read_boxes: ALL static fields from the heap.
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let pycode_op = ctx.const_ref(code_ptr as i64);
        let vsd_op = ctx.const_int(vsd);
        let debugdata_op = ctx.const_ref(debugdata as i64);
        let lastblock_op = ctx.const_ref(lastblock as i64);
        let w_globals_op = ctx.const_ref(ns_ptr);
        let owns = {
            let s = self.sym_mut();
            s.vable_last_instr = last_instr_op;
            s.vable_pycode = pycode_op;
            s.vable_valuestackdepth = vsd_op;
            s.vable_debugdata = debugdata_op;
            s.vable_lastblock = lastblock_op;
            s.vable_w_globals = w_globals_op;
            s.owns_virtualizable_shadow()
        };
        // pyjitpl.py:1188-1199 `_opimpl_setfield_vable` parity (Slice 3.0b):
        // mirror the heap-read seed into the canonical
        // `metainterp.virtualizable_boxes` shadow so subsequent readers
        // (snapshot, gen_writeback_vable_to_heap, JUMP-arg dedup) see
        // the same identity that `s.vable_*` carries.
        if owns {
            mirror_vable_static_to_boxes(
                ctx,
                "last_instr",
                last_instr_op,
                Value::Int(last_instr_value),
            );
            mirror_vable_static_to_boxes(ctx, "pycode", pycode_op, Value::Ref(GcRef(code_ptr)));
            mirror_vable_static_to_boxes(ctx, "valuestackdepth", vsd_op, Value::Int(vsd));
            mirror_vable_static_to_boxes(
                ctx,
                "debugdata",
                debugdata_op,
                Value::Ref(GcRef(debugdata)),
            );
            mirror_vable_static_to_boxes(
                ctx,
                "lastblock",
                lastblock_op,
                Value::Ref(GcRef(lastblock)),
            );
            mirror_vable_static_to_boxes(
                ctx,
                "w_globals",
                w_globals_op,
                Value::Ref(GcRef(ns_ptr as usize)),
            );
        }
    }

    /// G.4.3a: For portal-bridge frames, derive the absolute
    /// valuestackdepth from `metadata.stack_base + depth_at_py_pc[pc]`
    /// (both populated by `install_portal_for` G.3h + G.4.2).  Returns
    /// `None` for non-portal-bridge or null-jitcode states so the caller
    /// can fall back to the stale `sym.valuestackdepth` /
    /// `pre_opcode_registers_r.len()` heuristic that the per-CodeObject
    /// path relies on.
    ///
    /// Why this is needed: portal-bridge tracing records each user opcode
    /// as a residual call to `execute_opcode_step`.  The symbolic
    /// `sym.valuestackdepth` only advances through `push_typed_value` /
    /// `pop_value` (`trace_opcode.rs:808/872`), neither of which fires
    /// for residual-call paths.  As a result the symbolic value stays at
    /// its initial seed (`= stack_base`) for the lifetime of the trace,
    /// and the encoded `vable_valuestackdepth` is too low.  Restoring
    /// that value into PyFrame.valuestackdepth crashes the next pop with
    /// `assertion failed: self.valuestackdepth > self.stack_base()`
    /// (`pyframe.rs:862`).  The metadata-driven computation matches the
    /// runtime PyFrame state for the same `(jitcode, py_pc)` pair, just
    /// like the per-CodeObject path's stack-effect walk does inside
    /// `pre_opcode_registers_r`.
    fn portal_bridge_vable_vsd(&self, pc: usize) -> Option<i64> {
        let s = self.sym();
        if s.jitcode.is_null() {
            return None;
        }
        let payload = unsafe { &(*s.jitcode).payload };
        if !payload.is_portal_bridge() {
            return None;
        }
        let depth = payload
            .metadata
            .depth_at_py_pc
            .get(pc)
            .copied()
            .unwrap_or(0) as usize;
        Some((payload.metadata.stack_base + depth) as i64)
    }

    /// capture_resumedata(resumepc=orgpc) parity: flush vable fields for guards.
    ///
    /// When a pre-opcode snapshot is present, sets vable_last_instr = orgpc - 1
    /// and vable_valuestackdepth = the snapshot depth. The guard's fail_args
    /// then carry the pre-opcode stack state so the blackhole interpreter
    /// can re-execute the opcode from orgpc.
    ///
    /// Note: `record_branch_guard` does NOT call this — branch guards
    /// build their own fail_args with post-pop state and other_target PC
    /// (see the comment there for why).
    fn flush_to_frame_for_guard(&mut self, ctx: &mut TraceCtx) {
        // RPython capture_resumedata(resumepc=orgpc) parity:
        // Always use orgpc (opcode start PC) as the resume PC.
        let resume_pc = self.orgpc;
        let vsd = self.portal_bridge_vable_vsd(resume_pc).unwrap_or_else(|| {
            let s = self.sym();
            self.pre_opcode_registers_r
                .as_ref()
                .map(|pre_r| pre_r.len())
                .unwrap_or(s.valuestackdepth) as i64
        });
        // pyjitpl.py:2586-2602 `capture_resumedata` parity: RPython reads
        // `metainterp.virtualizable_boxes` without mutating it. The two
        // fields that advance per-opcode (`last_instr`, `valuestackdepth`)
        // need a guard-time-correct override here because pyre's tracer
        // is itself the dispatch loop: at guard time the active opcode is
        // the one at `orgpc`, so the snapshot must encode the pre-opcode
        // state (`last_instr = orgpc - 1`, `valuestackdepth = pre-opcode
        // depth via pre_opcode_registers_r / portal_bridge_vable_vsd`).
        // The other four scalars (`pycode`, `debugdata`, `lastblock`,
        // `w_globals`) keep the inputarg OpRefs `init_vable_indices`
        // seeded at trace start because pyre-jit-trace never enters
        // their mutators under CPython 3.14 bytecode: `pycode` /
        // `w_globals` are set only by `pyframe.rs::frame_reinit`;
        // `debugdata` only by `getorcreate_debug_data` on debug paths;
        // `lastblock` only by `pyopcode.py:1268 SETUP_*/POP_BLOCK`,
        // none of which CPython 3.14 emits.  This matches RPython's
        // "boxes carry vable inputargs" model — see
        // `mirror_vable_static_to_boxes` doc for the convergence path
        // when those handlers are re-introduced.
        let last_instr_value = resume_pc as i64 - 1;
        let last_instr_op = ctx.const_int(last_instr_value);
        let vsd_op = ctx.const_int(vsd);
        let s = self.sym_mut();
        s.vable_last_instr = last_instr_op;
        s.vable_valuestackdepth = vsd_op;
        // The shared `ctx.virtualizable_boxes` shadow is intentionally
        // not mirrored here — see `mirror_vable_static_to_boxes` doc
        // for the convention.  The two stores have distinct roles:
        // `s.vable_*` is the snapshot reader's view (carries pre-opcode
        // overrides set here, save/restored by `record_branch_guard`),
        // `ctx.virtualizable_boxes` is the JUMP/JIT-time view (consumed
        // by `gen_writeback_vable_to_heap` and `close_loop_args_at`).
    }

    /// pyjitpl.py:3317-3335 vable_and_vrefs_before_residual_call.
    ///
    /// RPython structure:
    ///
    /// ```text
    /// def vable_and_vrefs_before_residual_call(self):
    ///     vrefinfo = self.staticdata.virtualref_info
    ///     for i in range(1, len(self.virtualref_boxes), 2):
    ///         vrefbox = self.virtualref_boxes[i]
    ///         vref = vrefbox.getref_base()
    ///         vrefinfo.tracing_before_residual_call(vref)
    ///     #
    ///     vinfo = self.jitdriver_sd.virtualizable_info
    ///     if vinfo is not None:
    ///         virtualizable_box = self.virtualizable_boxes[-1]
    ///         virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
    ///         vinfo.tracing_before_residual_call(virtualizable)
    ///         force_token = self.history.record0(rop.FORCE_TOKEN, ...)
    ///         self.history.record2(rop.SETFIELD_GC, virtualizable_box,
    ///                              force_token, None,
    ///                              descr=vinfo.vable_token_descr)
    /// ```
    ///
    /// Key points:
    ///   1. vref token marking is unconditional (no vinfo check).
    ///   2. virtualizable processing only runs when vinfo is not None.
    ///   3. No call to `gen_store_back_in_vable` — that helper is only
    ///      invoked from `opimpl_hint_force_virtualizable` (pyjitpl.py:1071).
    fn vable_and_vrefs_before_residual_call(&mut self, ctx: &mut TraceCtx) {
        // pyjitpl.py:3319-3322: virtualref token marking (ALWAYS runs,
        // even without virtualizable info).
        self.vrefs_before_residual_call();

        // pyjitpl.py:3326: vinfo = self.jitdriver_sd.virtualizable_info
        // pyjitpl.py:3327: if vinfo is not None:
        //
        // majit's pyre port uses `standard_virtualizable_box()` as the
        // vinfo proxy — it returns `Some(box)` iff the jitdriver has a
        // standard virtualizable registered for the current frame. RPython
        // checks the per-jitdriver `vinfo` first, then derefs the box; the
        // pyre-side null check on `concrete_vable_ptr` is the defensive
        // analogue of `unwrap_virtualizable_box(virtualizable_box)`.
        let Some(vable_ref) = ctx.standard_virtualizable_box() else {
            return;
        };
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return;
        }
        // pyjitpl.py:3329-3330:
        //   virtualizable = vinfo.unwrap_virtualizable_box(virtualizable_box)
        //   vinfo.tracing_before_residual_call(virtualizable)
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        unsafe {
            info.tracing_before_residual_call(obj_ptr);
        }
        // pyjitpl.py:3332-3335:
        //   force_token = self.history.record0(rop.FORCE_TOKEN,
        //                                      lltype.nullptr(llmemory.GCREF.TO))
        //   self.history.record2(rop.SETFIELD_GC, virtualizable_box,
        //                        force_token, None,
        //                        descr=vinfo.vable_token_descr)
        let force_token = ctx.force_token();
        ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
    }

    /// pyjitpl.py:3349-3366 vable_after_residual_call.
    ///
    /// Only checks virtualizable (not vrefs — those are checked
    /// separately by vrefs_after_residual_call at the call site).
    /// If virtualizable escaped, reloads fields and aborts tracing
    /// (SwitchToBlackhole parity).
    fn vable_after_residual_call(&mut self) -> Result<(), PyError> {
        // pyjitpl.py:3350-3351: if vinfo is not None:
        // RPython parity: use the same gate as before_residual_call
        // (ctx.standard_virtualizable_box). If before didn't set
        // TOKEN_TRACING_RESCALL (no vbox), after must not check it.
        let has_vbox = self.with_ctx(|_, ctx| ctx.standard_virtualizable_box().is_some());
        if !has_vbox {
            return Ok(());
        }
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return Ok(());
        }
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        let vable_forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
        if vable_forced {
            // pyjitpl.py:3356: self.load_fields_from_virtualizable()
            self.load_fields_from_virtualizable();
            // pyjitpl.py:3365-3366:
            //   raise SwitchToBlackhole(Counters.ABORT_ESCAPE,
            //                           raising_exception=True)
            return Err(PyError::runtime_error(
                "ABORT_ESCAPE: virtualizable escaped during residual call",
            ));
        }
        Ok(())
    }

    /// pyjitpl.py:3452-3463 load_fields_from_virtualizable.
    ///
    /// Force a reload of the virtualizable fields into the local
    /// boxes (called only in escaping cases, just before abort).
    fn load_fields_from_virtualizable(&mut self) {
        let obj_ptr = self.sym().concrete_vable_ptr;
        if obj_ptr.is_null() {
            return;
        }
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        // pyjitpl.py:3460-3462: self.virtualizable_boxes = vinfo.read_boxes(
        //     self.cpu, virtualizable, 0)
        // Re-read all virtualizable fields from the heap object.
        let lengths = unsafe { info.read_array_lengths_from_heap(obj_ptr as *const u8) };
        let (static_boxes, array_boxes) =
            unsafe { info.read_all_boxes(obj_ptr as *const u8, &lengths) };
        // Store back into PyreSym's concrete state so the blackhole
        // interpreter sees the up-to-date values.
        let sym = self.sym_mut();
        // Static fields: update concrete_locals for the virtualizable fields.
        for (i, &val) in static_boxes.iter().enumerate() {
            if i < sym.concrete_locals.len() {
                sym.concrete_locals[i] = ConcreteValue::Int(val);
            }
        }
        // Array fields: update concrete locals/stack from array boxes.
        if let Some(arr) = array_boxes.first() {
            let nlocals = sym.nlocals;
            for (i, &val) in arr.iter().enumerate() {
                if i < nlocals && i < sym.concrete_locals.len() {
                    sym.concrete_locals[i] = ConcreteValue::Ref(val as PyObjectRef);
                } else {
                    let stack_idx = i.saturating_sub(nlocals);
                    if stack_idx < sym.concrete_stack.len() {
                        sym.concrete_stack[stack_idx] = ConcreteValue::Ref(val as PyObjectRef);
                    }
                }
            }
        }
    }

    /// pyjitpl.py:3317-3337 parity: before residual call, set all
    /// active virtualref tokens to TOKEN_TRACING_RESCALL.
    fn vrefs_before_residual_call(&self) {
        let vref_boxes = &self.sym().virtualref_boxes;
        if vref_boxes.is_empty() {
            return;
        }
        // pyjitpl.py:3339: for each pair, call tracing_before on the
        // ODD slot (vrefbox = second element), not the virtual (first).
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        // virtualref_boxes = [(virt_sym, virt_ptr), (vref_sym, vref_ptr), ...]
        for pair in vref_boxes.chunks(2) {
            if let Some(&(_vref_sym, vref_ptr)) = pair.get(1) {
                if vref_ptr != 0 {
                    unsafe {
                        vref_info.tracing_before_residual_call(vref_ptr as *mut u8);
                    }
                }
            }
        }
    }

    /// pyjitpl.py:3337-3347 vrefs_after_residual_call parity:
    /// after residual call, check if any virtualref was forced.
    /// If forced, call stop_tracking_virtualref(i) to record
    /// VIRTUAL_REF_FINISH and replace odd slot with CONST_NULL.
    fn vrefs_after_residual_call(&mut self, ctx: &mut TraceCtx) {
        let sym = unsafe { &mut *self.sym };
        if sym.virtualref_boxes.is_empty() {
            return;
        }
        let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
        let len = sym.virtualref_boxes.len();
        let mut i = 0;
        while i < len {
            let (_, vref_ptr) = sym.virtualref_boxes[i + 1];
            if vref_ptr != 0 {
                let forced = unsafe { vref_info.tracing_after_residual_call(vref_ptr as *mut u8) };
                if forced {
                    Self::stop_tracking_virtualref(sym, ctx, i);
                }
            }
            i += 2;
        }
    }

    /// pyjitpl.py:3371-3378 stop_tracking_virtualref parity.
    ///
    /// Record VIRTUAL_REF_FINISH(vrefbox, virtualbox) and replace
    /// the odd slot with ConstPtr(NULL).
    fn stop_tracking_virtualref(sym: &mut PyreSym, ctx: &mut TraceCtx, i: usize) {
        let virt_opref = sym.virtualref_boxes[i].0;
        let (vref_opref, _) = sym.virtualref_boxes[i + 1];
        // pyjitpl.py:3376: record VIRTUAL_REF_FINISH(vrefbox, virtualbox)
        let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, virt_opref]);
        // pyjitpl.py:3378: self.virtualref_boxes[i+1] = CONST_NULL
        // history.py:361: CONST_NULL = ConstPtr(ConstPtr.value)
        let null_opref = ctx.const_null();
        sym.virtualref_boxes[i + 1] = (null_opref, 0);
    }

    /// Loop-carried values must follow the typed live-state contract used by
    /// PyreMeta::slot_types / restore_values().
    ///
    /// In pyre's typed INT/REF/FLOAT model, integer locals cross a loop JUMP
    /// as raw Int values, not freshly boxed W_Int objects.
    fn materialize_loop_carried_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        slot_type: Type,
    ) -> OpRef {
        match slot_type {
            Type::Int => match self.value_type(value) {
                Type::Int => value,
                Type::Ref => {
                    // Convert boxed W_Int back to its raw payload so the loop
                    // header sees the typed INT stream expected by restore_values().
                    self.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, value))
                }
                _ => value,
            },
            Type::Ref => match self.value_type(value) {
                Type::Int => {
                    // Virtualizable slots are Ref — re-box raw Int for the
                    // loop header which expects boxed W_IntObject.
                    let int_type_addr = &INT_TYPE as *const _ as i64;
                    crate::generated::trace_box_int(
                        ctx,
                        value,
                        w_int_size_descr(),
                        ob_type_descr(),
                        int_intval_descr(),
                        int_type_addr,
                    )
                }
                Type::Float => {
                    let float_type_addr = &FLOAT_TYPE as *const _ as i64;
                    crate::trace_box_float(
                        ctx,
                        value,
                        w_float_size_descr(),
                        ob_type_descr(),
                        float_floatval_descr(),
                        float_type_addr,
                    )
                }
                _ => value,
            },
            _ => value,
        }
    }

    pub(crate) fn close_loop_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.close_loop_args_at(ctx, None)
    }

    /// PRE-EXISTING-ADAPTATION: bundles `pyjitpl.py:2957-2965` `live_arg_boxes`
    /// construction (`greenboxes + redboxes + virtualizable_boxes`,
    /// `pop()` the trailing token) with the `vable_last_instr` pin
    /// (`pyjitpl.py:2973`). RPython performs both inline within
    /// `reached_loop_header`; pyre extracts them because pyre's "args"
    /// are `OpRef`s pulled from the unified `registers_r` register
    /// file, and the merge `target_pc` must be threaded through
    /// explicitly (RPython has it implicitly in `redboxes`).
    pub(crate) fn close_loop_args_at(
        &mut self,
        ctx: &mut TraceCtx,
        target_pc: Option<usize>,
    ) -> Vec<OpRef> {
        // pyjitpl.py:2954-2965 reached_loop_header: virtualizable_boxes
        // (read from locals_cells_stack_w[*] by virtualizable.py:86-98
        // read_boxes) are carried into the JUMP unchanged, including
        // stack slots. Do NOT truncate to nlocals here.
        let concrete_nlocals = self.sym().nlocals;
        let concrete_vsd = self.sym().valuestackdepth.max(concrete_nlocals);
        {
            let s = self.sym_mut();
            s.nlocals = concrete_nlocals;
            s.valuestackdepth = concrete_vsd;
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            // virtualizable.py:44 + interp_jit.py:25-31: locals_cells_stack_w[*]
            // is a W_Root array → every item is declared Ref. The loop-carried
            // types passed to the JUMP / merge point MUST be Ref for every
            // array slot; tracker-observed Int/Float types are internal to
            // unboxing lowering and must not leak into the inputarg contract.
            //
            // The type-map reset is deferred until AFTER the materialize loop
            // below. Resetting here would poison `self.value_type(value)` used
            // by `materialize_loop_carried_value` — an Int-typed OpRef on the
            // symbolic stack (e.g. a GetarrayitemRawI result LOAD_FASTed onto
            // the stack) would appear Ref to the materializer and skip the
            // Int→Ref boxing, handing the MIFrame merge-point a raw Int value
            // in a Ref-typed slot. Cut_trace_from then installs that raw Int
            // at a cut-inputarg position whose declared type is Ref, and the
            // downstream unroll pass produces JUMP args that pass an Int into
            // a Ref slot at runtime → SIGSEGV (str x24, [x12, #0x10] with x12
            // carrying the unboxed int payload).
            //
            // `registers_r` is the unified abstract register file;
            // reserve `[nlocals..nlocals+stack_only)` for the stack so
            // merge-point JUMP args have a stable slice.
            let min_regs_len = concrete_nlocals + stack_only;
            if s.registers_r.len() < min_regs_len {
                s.registers_r.resize(min_regs_len, OpRef::NONE);
            }
        }
        self.flush_to_frame(ctx);
        // pyjitpl.py:2973 reached_loop_header: a merge-point resume enters
        // the target loop at `pc`, so last_instr must be `pc - 1` so the
        // interpreter's `next_instr() = last_instr + 1` returns the merge
        // point. flush_to_frame already stored `orgpc - 1`; override with
        // the merge target.
        //
        // Task #21 Slice 3 propagation gap #1: propagate the override
        // into the virtualizable_boxes shadow so the writeback below
        // emits the merge-target PC, not the orgpc placed by
        // flush_to_frame. virtualizable_boxes[0] = vable_last_instr per
        // virtualizable_gen.rs:37-44 inputargs ordering.
        if let Some(pc) = target_pc {
            let last_instr_value = pc as i64 - 1;
            let opref = ctx.const_int(last_instr_value);
            let owns = {
                let s = self.sym_mut();
                s.vable_last_instr = opref;
                s.owns_virtualizable_shadow()
            };
            if owns {
                mirror_vable_static_to_boxes(
                    ctx,
                    "last_instr",
                    opref,
                    Value::Int(last_instr_value),
                );
            }
        }
        // Task #21 fix path 3 — vable heap-writeback before JUMP.
        //
        // When `descriptor=Some` + `patch_new_loop_to_load_virtualizable_
        // fields` (compile.py:425-461) reduces the target LABEL to
        // reds-only inputargs (`[frame, ec]`), the patched body's
        // GETFIELD_GC / GETARRAYITEM_GC preamble re-loads vable static
        // and array fields from the heap on every iteration. Pyre's
        // tracer updates symbolic `sym.vable_*` / `sym.registers_r`
        // during opcode tracing but does not touch the heap; without
        // this writeback, the bridge body re-reads stale state and
        // ouroboroses on the first guard fail.
        //
        // Gate selects the target LABEL's inputarg_types length:
        //   - root trace: ctx.inputarg_types() = own inputargs = LABEL types
        //   - bridge with resolvable loop_token: target loop's
        //     inputarg_types
        //   - bridge without (current_trace_green_key missing or lookup
        //     fails): None — gate stays off (target LABEL shape unknown)
        //
        // Threshold `< NUM_SCALAR_INPUTARGS` separates legacy (≥ 7
        // entries: `[frame, 6 scalars, locals..., stack...]`) from
        // reduced (1-2 entries: `[frame]` or `[frame, ec]`).
        //
        // Slice 3 closure (3.0a/b/c/d/e/e'): every direct `s.vable_*`
        // write that touches the active sym now mirrors into
        // `virtualizable_boxes` via `mirror_vable_static_to_boxes`,
        // and inline frame push/pop save+restore the boxes shadow so
        // it tracks the active sym across callee boundaries.  With
        // those propagation paths in place the writeback can fire
        // unconditionally on reduced LABELs.
        {
            let target_inputarg_len: Option<usize> = {
                let (driver, _) = crate::driver::driver_pair();
                if driver.is_bridge_tracing() {
                    driver
                        .current_trace_green_key()
                        .and_then(|gk| driver.get_loop_token(gk))
                        .map(|token| token.inputarg_types.len())
                } else {
                    Some(ctx.inputarg_types().len())
                }
            };
            if let Some(len) = target_inputarg_len {
                // `len > 0` excludes the bridge-target-loop-token-with-
                // empty-inputarg_types edge case (probe captured 73
                // such firings on dynasm nested_loop, all
                // target_inputarg_len=0 — likely a freshly-allocated
                // loop token whose types are not yet set). The
                // reds-only LABEL always carries `[frame]` (len=1) or
                // `[frame, ec]` (len=2).
                let is_reduced_label =
                    len > 0 && len < crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
                if is_reduced_label {
                    if let Some(vable_box) = ctx.standard_virtualizable_box() {
                        ctx.gen_writeback_vable_to_heap(vable_box);
                    }
                }
            }
        }
        debug_assert!(
            self.sym().nlocals > 0 || !self.sym().is_active_vable_owner,
            "nlocals must be set by init_symbolic before close_loop_args_at"
        );
        // RPython close_loop_args parity: JUMP args must match the target
        // label's types (inputarg_types). materialize_loop_carried_value
        // boxes values to match (e.g. Int → Ref for virtualizable locals).
        //
        // For bridge traces, ctx.inputarg_types() returns the bridge's
        // guard fail_arg types, NOT the root loop's label types. The JUMP
        // targets the root loop label, so resolve the root loop's LABEL/
        // inputargs types via `front_target_inputarg_types` (peeled-entry
        // LABEL when unrolled, root TreeLoop.inputargs otherwise — see
        // `MetaInterp::front_target_inputarg_types` doc).
        let inputarg_types = {
            let (driver, _) = crate::driver::driver_pair();
            if driver.is_bridge_tracing() {
                if let Some(gk) = driver.current_trace_green_key() {
                    driver
                        .front_target_inputarg_types(gk)
                        .unwrap_or_else(|| ctx.inputarg_types())
                } else {
                    ctx.inputarg_types()
                }
            } else {
                ctx.inputarg_types()
            }
        };
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // `extra_reds` reflects the canonical ec/red layout (NUM_EXTRA_REDS).
        // Drives the conditional ec push at args[1] and the dedup-side
        // OpRef ↔ virtualizable_box mapping below.
        let extra_reds = crate::virtualizable_gen::NUM_EXTRA_REDS;
        // pyjitpl.py:2954-2965 reached_loop_header parity: once the
        // descriptor-driven virtualizable path is active, JUMP args must carry
        // the full virtualizable array capacity. compile.rs later expands the
        // loop entry from the same heap lengths; emitting only the live stack
        // window here leaves too few source args for that expansion.
        let target_array_capacity = ctx
            .virtualizable_array_lengths()
            .map(|lengths| lengths.iter().copied().sum::<usize>())
            .filter(|&len| len >= self.sym().nlocals)
            .unwrap_or_else(|| {
                self.sym().nlocals
                    + self
                        .sym()
                        .valuestackdepth
                        .saturating_sub(self.sym().nlocals)
            });
        let (
            frame,
            execution_context,
            next_instr,
            code,
            stack_depth,
            debugdata,
            lastblock,
            namespace,
            nlocals,
            locals,
            stack,
            _local_types,
            _stack_types,
        ) = {
            let s = self.sym();
            let nlocals = s.nlocals;
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            // virtualizable.py:86-98 `read_boxes` + pyjitpl.py:2954-2965
            // `reached_loop_header`: `virtualizable_boxes` length is the
            // target vable array capacity (`nlocals + ncells + co_stacksize`),
            // not the live Python stack depth. JUMP args carry that full
            // capacity so every target LABEL slot has a matching source.
            // Slots beyond the live prefix are left as `OpRef::NONE` and
            // filled by `materialize_fail_arg_slot` below, which reads
            // `concrete_value_at` and falls back to `PY_NULL` for dead
            // capacity slots — mirroring RPython's null-padded
            // virtualizable_boxes tail.
            let reg_len = s.registers_r.len();
            let target_stack_capacity = target_array_capacity.saturating_sub(nlocals);
            let locals_len = nlocals.min(reg_len);
            let live_stack_len = stack_only.min(reg_len.saturating_sub(locals_len));
            let stack_slice_start = locals_len;
            let stack_slice_end = stack_slice_start + live_stack_len;
            let mut stack_vec = s.registers_r[stack_slice_start..stack_slice_end].to_vec();
            stack_vec.resize(target_stack_capacity, OpRef::NONE);
            let mut stack_types_vec =
                s.symbolic_stack_types[..stack_only.min(s.symbolic_stack_types.len())].to_vec();
            stack_types_vec.resize(target_stack_capacity, Type::Ref);
            (
                s.frame,
                s.execution_context,
                s.vable_last_instr,
                s.vable_pycode,
                s.vable_valuestackdepth,
                s.vable_debugdata,
                s.vable_lastblock,
                s.vable_w_globals,
                nlocals,
                s.registers_r[..locals_len].to_vec(),
                stack_vec,
                s.symbolic_local_types.clone(),
                stack_types_vec,
            )
        };
        let mut args = vec![frame];
        if extra_reds == 1 {
            args.push(execution_context);
        }
        args.extend_from_slice(&[
            next_instr,
            code,
            stack_depth,
            debugdata,
            lastblock,
            namespace,
        ]);
        for (idx, value) in locals.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(num_scalars + idx)
                .copied()
                .unwrap_or(Type::Ref);
            // Materialize NONE slots from concrete frame before boxing.
            // RPython's live_arg_boxes never contains holes at loop closure
            // because MIFrame.run_one_step always updates all live registers.
            let value = self.materialize_fail_arg_slot(ctx, value, target_type, idx);
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        for (stack_idx, value) in stack.into_iter().enumerate() {
            let target_type = inputarg_types
                .get(num_scalars + nlocals + stack_idx)
                .copied()
                .unwrap_or(Type::Ref);
            let value =
                self.materialize_fail_arg_slot(ctx, value, target_type, nlocals + stack_idx);
            args.push(self.materialize_loop_carried_value(ctx, value, target_type));
        }
        // virtualizable.py:44 parity (delayed): now that all materialize_loop_
        // carried_value calls have consulted each OpRef's actual type, flip the
        // symbolic type maps to the post-loop invariant where every array slot
        // is Ref. Downstream consumers (reached_loop_header's live_types, the
        // merge-point snapshot it stores) observe the Ref contract while the
        // box/unbox decisions above still see the pre-loop truth.
        {
            let s = self.sym_mut();
            let stack_only = s.valuestackdepth.saturating_sub(s.nlocals);
            s.symbolic_local_types = vec![Type::Ref; concrete_nlocals];
            s.symbolic_stack_types = vec![Type::Ref; stack_only];
        }
        // pyjitpl.py:2934-2965 remove_consts_and_duplicates:
        //     def remove_consts_and_duplicates(self, boxes, endindex, duplicates):
        //         for i in range(endindex):
        //             box = boxes[i]
        //             if isinstance(box, Const) or box in duplicates:
        //                 boxes[i] = self.history.record_same_as(box)
        //             else:
        //                 duplicates[box] = None
        //
        //     def reached_loop_header(self, greenboxes, redboxes):
        //         duplicates = {}
        //         self.remove_consts_and_duplicates(redboxes, len(redboxes),
        //                                           duplicates)
        //         live_arg_boxes = greenboxes + redboxes
        //         if self.jitdriver_sd.virtualizable_info is not None:
        //             self.remove_consts_and_duplicates(
        //                 self.virtualizable_boxes,
        //                 len(self.virtualizable_boxes)-1,
        //                 duplicates)
        //             live_arg_boxes += self.virtualizable_boxes
        //             live_arg_boxes.pop()
        //
        // RPython dedups ALL of redboxes (1 = vable_box = frame) AND
        // ALL of virtualizable_boxes[:-1] (static_fields + array_items),
        // sharing one `duplicates` dict across both calls. In pyre's flat
        // layout `args = [frame, ni, code, vsd, ns, locals..., stack...]`,
        // that corresponds to every index 0..args.len(). Previously pyre
        // skipped the 7 scalar header slots (frame + 6 static fields),
        // which is a line-by-line divergence from RPython.
        // Track slots that the dedup actually mutated so we can mirror the
        // `put_back_list_of_boxes3` mutation below (pyjitpl.py:1578 writes
        // the deduped redboxes back to the frame's registers; RPython's
        // `remove_consts_and_duplicates` additionally mutates
        // `self.virtualizable_boxes` in place so subsequent reads see the
        // SameAs-wrapped identities).
        let mut dedup_changed: Vec<(usize, OpRef)> = Vec::new();
        {
            use std::collections::HashSet;
            let mut duplicates: HashSet<OpRef> = HashSet::new();
            for i in 0..args.len() {
                let opref = args[i];
                if opref.is_constant() || !duplicates.insert(opref) {
                    // pyjitpl.py:2934-2965 `record_same_as(box)` uses the
                    // `box.type` intrinsic to pick `same_as_i/r/f` — the
                    // SameAs op's result type matches the input box, NEVER
                    // the slot's declared type. When `args[i]` is a constant
                    // whose Value type differs from the slot's declared
                    // `inputarg_types[i]` (e.g. an Int constant placeholder
                    // routed into a Ref-typed vable header slot), wrapping
                    // it as `same_as_for_type(slot_type)` produces a
                    // cross-type SameAs whose `make_equal_to` absorb in
                    // `optimizer.rs::propagate_from_pass_range` violates the
                    // Box.type invariant in `OptContext::replace_op`.
                    //
                    // Match RPython by deriving the SameAs op from the
                    // OpRef's actual type via `ctx.get_opref_type`, falling
                    // back to the slot type only when the OpRef has no
                    // recoverable type (which would be a separate bug).
                    let tp = ctx
                        .get_opref_type(opref)
                        .or_else(|| inputarg_types.get(i).copied())
                        .unwrap_or(majit_ir::Type::Ref);
                    let same_as_op = majit_ir::OpCode::same_as_for_type(tp);
                    let new_opref = ctx.record_op(same_as_op, &[opref]);
                    args[i] = new_opref;
                    dedup_changed.push((i, new_opref));
                }
            }
        }
        // pyjitpl.py:2961-2963 in-place mutation of self.virtualizable_boxes:
        //     self.remove_consts_and_duplicates(
        //         self.virtualizable_boxes,
        //         len(self.virtualizable_boxes)-1,
        //         duplicates)
        //
        // RPython's `remove_consts_and_duplicates` writes the SameAs results
        // back into `self.virtualizable_boxes[i]` IN PLACE for `i` in
        // `range(len-1)`. The trailing element (`virtualizable_boxes[-1]`,
        // the standard vable identity itself = pyre's frame OpRef) is
        // intentionally skipped. The mutated `self.virtualizable_boxes`
        // then feeds the GUARD_FUTURE_CONDITION snapshot below.
        //
        // pyre's `args` Vec layout is `[frame, ni, code, vsd, ns,
        // locals..., stack...]` where `args[0]` is the trailing
        // virtualizable identity (mapped to `vb[len-1]`) and `args[1..]`
        // is `vb[0..len-1]`. The line-by-line mirror here mutates
        // `ctx.virtualizable_boxes[i-1]` for every dedup'd `args[i]`
        // with `i >= 1`, leaving `vb[len-1]` (the trailing identity)
        // untouched.
        //
        // Note: pyjitpl.py:1578 `put_back_list_of_boxes3` writes the
        // dedup'd `redboxes` back to the FRAME's `registers_i/r/f`
        // arrays. RPython only runs `put_back_list_of_boxes3` from the
        // `opimpl_jit_merge_point` failed-to-close path (i.e. when
        // `reached_loop_header` returns normally instead of raising
        // SwitchToBlackhole). pyre's `close_loop_args_at` is the
        // SUCCESS path (the trace is closing), so the put_back has no
        // matching call site here — it would belong on the path where
        // pyre fails to close at a merge point and continues tracing,
        // which pyre's tracer does not currently expose.
        for &(idx, new_opref) in &dedup_changed {
            if idx <= extra_reds {
                // args[0] = frame = ctx.virtualizable_boxes[len-1].
                // Any extra reds that follow it are not part of
                // `virtualizable_boxes`, so only the virtualizable payload
                // starting after `[frame, extra_reds...]` is mirrored back.
                continue;
            }
            let vb_idx = idx - (1 + extra_reds);
            ctx.set_virtualizable_box_at(vb_idx, new_opref);
        }
        // pyjitpl.py:1578 put_back_list_of_boxes3: write dedup'd values back
        // to frame symbolic state so subsequent tracing sees the SameAs-wrapped
        // identities. RPython runs this on the "continue tracing" path after
        // reached_loop_header returns without closing. Harmless on the "close
        // loop" path since the frame won't be reused.
        {
            // `num_scalars` (NUM_SCALAR_INPUTARGS) already counts extra_reds
            // (frame + extra_reds + vable static fields) post-Phase 2.1.
            let total_scalar_prefix = num_scalars;
            let s = self.sym_mut();
            for &(idx, new_opref) in &dedup_changed {
                if idx < total_scalar_prefix {
                    match idx {
                        0 => s.frame = new_opref,
                        1 if extra_reds == 1 => s.execution_context = new_opref,
                        _ => match idx - extra_reds {
                            1 => s.vable_last_instr = new_opref,
                            2 => s.vable_pycode = new_opref,
                            3 => s.vable_valuestackdepth = new_opref,
                            4 => s.vable_debugdata = new_opref,
                            5 => s.vable_lastblock = new_opref,
                            6 => s.vable_w_globals = new_opref,
                            _ => {}
                        },
                    }
                } else {
                    let local_idx = idx - total_scalar_prefix;
                    // `registers_r` is the unified abstract register
                    // file; locals + stack tail share the same addr
                    // space, so the dedup'd rename writes to the single
                    // slot regardless of whether the dedup refers to a
                    // local or a stack entry.
                    if local_idx < s.registers_r.len() {
                        s.registers_r[local_idx] = new_opref;
                    }
                }
            }
        }
        // pyjitpl.py:2967-2969: generate a dummy GUARD_FUTURE_CONDITION
        // just before the JUMP so that unroll can use it when it's
        // creating artificial guards (patchguardop). record_guard calls
        // capture_resumedata which captures the full framestack +
        // virtualizable_boxes + virtualref_boxes.
        //
        // RPython only emits GUARD_FUTURE_CONDITION here. GUARD_NOT_INVALIDATED
        // is *not* unconditionally emitted before JUMP — pyjitpl.py:1086-1089
        // emits it only inside `opimpl_record_quasi_immutable_field`, after a
        // quasi-immut field read sets `heapcache.need_guard_not_invalidated`.
        // The pyre frontend does the same via `flush_guard_not_invalidated`,
        // so an unconditional emit here would (a) leak resume data for traces
        // that have no quasi-immut dep at all, and (b) leave a runtime guard
        // whose flag is decoupled from any watcher, which can spuriously
        // exit a hot inner loop with no chance of re-tracing.
        //
        // RPython parity: orgpc must be the loop header TARGET, not the
        // JUMP_BACKWARD's PC. The patchguardop from this GuardFutureCondition
        // provides the resume_position for all peeled body virtual state guards.
        // If orgpc is wrong, all those guards resume at the wrong PC.
        if let Some(pc) = target_pc {
            self.orgpc = pc;
        }
        self.generate_guard(ctx, majit_ir::OpCode::GuardFutureCondition, &[]);
        // pyjitpl.py:2971 assert len(self.virtualref_boxes) == 0,
        //     "missing virtual_ref_finish()?"
        // Reached loop header must not have dangling virtualrefs — they
        // should have been finished by prior vrefs_after_residual_call /
        // stop_tracking_virtualref. pyre's equivalent is sym.virtualref_boxes.
        debug_assert!(
            self.sym().virtualref_boxes.is_empty(),
            "missing virtual_ref_finish()? close_loop_args_at reached with \
             virtualref_boxes={:?}",
            self.sym().virtualref_boxes.len()
        );
        args
    }

    /// pyjitpl.py:2586 capture_resumedata: build fail_args for CURRENT
    /// top frame. Returns the scalar header plus active_boxes —
    /// `[frame, (ec)?, last_instr, pycode, valuestackdepth, debugdata,
    /// lastblock, w_globals, active_boxes...]` — matching
    /// `interp_jit.py:25-31 PyFrame._virtualizable_` /
    /// `virtualizable_spec.rs::PYFRAME_VABLE_FIELDS` line-by-line.
    /// `NUM_EXTRA_REDS` controls whether the ec slot
    /// (interp_jit.py:67 `reds = ['frame', 'ec']`) is present between
    /// frame and the vable static fields. Dormant under
    /// NUM_EXTRA_REDS=0 (skips ec push, preserves pre-ec 7-scalar
    /// layout). virtualizable.py:86 read_boxes: all static fields in
    /// order.
    pub(crate) fn current_fail_args(&mut self, ctx: &mut TraceCtx) -> Vec<OpRef> {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes = self.get_list_of_active_boxes(ctx, false, false);
        let s = self.sym();
        let mut fa =
            Vec::with_capacity(crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + active_boxes.len());
        fa.push(s.frame);
        if crate::virtualizable_gen::NUM_EXTRA_REDS > 0 {
            fa.push(s.execution_context);
        }
        fa.extend_from_slice(&[
            s.vable_last_instr,
            s.vable_pycode,
            s.vable_valuestackdepth,
            s.vable_debugdata,
            s.vable_lastblock,
            s.vable_w_globals,
        ]);
        fa.extend_from_slice(&active_boxes);
        fa
    }

    /// pyjitpl.py:1087 parity: after a field read that might have set the
    /// needs_guard_not_invalidated flag (quasi-immutable field), emit the
    /// guard with full snapshot via record_guard.
    pub(crate) fn flush_guard_not_invalidated(&mut self, ctx: &mut TraceCtx) {
        if let Some(saved_orgpc) = ctx.pending_guard_not_invalidated_pc() {
            ctx.set_pending_guard_not_invalidated(None);
            // pyjitpl.py:1087 parity: use the field read's orgpc so the
            // snapshot captures the correct liveness state.
            let current_orgpc = self.orgpc;
            self.orgpc = saved_orgpc;
            self.generate_guard(ctx, OpCode::GuardNotInvalidated, &[]);
            self.orgpc = current_orgpc;
        }
    }

    /// PyPy generate_guard + capture_resumedata: uses current_fail_args
    /// which encodes the full framestack for multi-frame resume.
    /// pyjitpl.py:3222 store_token_in_vable():
    ///   force_token = self.history.record0(rop.FORCE_TOKEN, ...)
    ///   self.history.record2(rop.SETFIELD_GC, vbox, force_token, ...)
    ///   self.generate_guard(rop.GUARD_NOT_FORCED_2)
    pub(crate) fn store_token_in_vable(&mut self, ctx: &mut TraceCtx) {
        if ctx.store_token_in_vable_setfield() {
            self.generate_guard(ctx, OpCode::GuardNotForced2, &[]);
        }
    }

    pub(crate) fn generate_guard(&mut self, ctx: &mut TraceCtx, opcode: OpCode, args: &[OpRef]) {
        // pyjitpl.py:2558-2560 generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The first arg of every data guard (GUARD_CLASS, GUARD_TRUE,
        // GUARD_NONNULL, GUARD_VALUE, ...) is the box being checked.
        // Control-flow guards (GUARD_NOT_FORCED, GUARD_NO_OVERFLOW,
        // GUARD_NOT_INVALIDATED, ...) call generate_guard with `args=&[]`,
        // so `args.first()` is None and the check is skipped — matching
        // RPython where `box=None` for those guards.
        if let Some(&first) = args.first() {
            if first.is_constant() {
                return;
            }
        }
        // pyjitpl.py:1087 parity: flush pending guard_not_invalidated
        // before recording any new guard (the quasi-immut guard should be
        // emitted with its own snapshot before the current guard).
        if opcode != OpCode::GuardNotInvalidated {
            self.flush_guard_not_invalidated(ctx);
        }
        // pyjitpl.py:2575-2578: determine after_residual_call from guard opcode.
        // opencoder.py:767: when true, all boxes in top frame are live
        // (liveness filter disabled for residual call guards).
        let after_residual_call = matches!(
            opcode,
            OpCode::GuardException
                | OpCode::GuardNoException
                | OpCode::GuardNotForced
                | OpCode::GuardAlwaysFails
        );
        if after_residual_call {
            // pyjitpl.py:2586-2602: residual-call guards snapshot the state
            // AFTER the call, using the auto-advanced pc and post-call
            // register file. The opcode-start snapshot is only for
            // re-executing the current opcode from orgpc.
            self.clear_pre_opcode_state();
        }
        // opencoder.py:819 capture_resumedata(framestack) parity:
        // Encode the full framestack [callee (top), caller (parent)] into
        // a multi-frame snapshot. The callee's pc is set to resumepc
        // (orgpc), while the caller keeps its original pc (return_point).
        // pyjitpl.py:2597 passes full framestack + vable/vref boxes.
        if !self.parent_frames.is_empty() {
            // pyjitpl.py:2593-2596: top frame pc = resumepc (orgpc)
            self.flush_to_frame_for_guard(ctx);
            // pyjitpl.py:177: active boxes = registers only (no header).
            let callee_active_boxes =
                self.get_list_of_active_boxes(ctx, false, after_residual_call);
            // RPython Box.type parity: snapshot types match the full
            // (un-filtered) active_boxes — constants are part of the
            // snapshot via TAGCONST.
            let callee_snapshot_types_full =
                self.build_fail_arg_types_for_active_boxes(&callee_active_boxes);

            // snapshot.pc must match the liveness PC used for active boxes
            // (get_list_of_active_boxes uses fallthrough_pc when after_residual_call).
            let callee_live_pc = if after_residual_call {
                self.fallthrough_pc
            } else {
                self.orgpc
            };
            // opencoder.py:819-834: snapshot uses active boxes (not fail_args).
            let snapshot = self.build_framestack_snapshot(
                ctx,
                callee_live_pc,
                &callee_active_boxes,
                &callee_snapshot_types_full,
            );
            // Task #91 slice 4a: snapshot is the source of truth — the
            // optimizer's `store_final_boxes_in_guard`
            // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from
            // the snapshot via `op.store_final_boxes(liveboxes)`
            // (mod.rs:3392), so the inline `fail_args` copy that the
            // legacy `record_guard_typed_with_fail_args` path used to
            // write was redundant.  Mirrors RPython
            // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
            // which records the guard with no inline fail_args and lets
            // `capture_resumedata` + `_number_boxes` populate them from
            // the snapshot chain.
            //
            // The state-fields header `[s.frame, s.vable_*]` that fed
            // the inline fail_args is now dead and removed; the
            // snapshot's `vable_boxes` already encodes the same
            // information via `build_virtualizable_boxes`. Parent-frame
            // types still flow into the recorded guard via
            // `extend_types_with_parents`.
            let types = self.extend_types_with_parents(ctx, callee_snapshot_types_full);
            let snapshot_id = ctx.capture_resumedata(snapshot);

            ctx.record_guard_typed(opcode, args, types);
            ctx.set_last_guard_resume_position(snapshot_id);
            return;
        }

        // pyjitpl.py:2586-2596 capture_resumedata(resumepc) parity:
        // Normal guards: resumepc = orgpc (re-execute the opcode from start).
        // after_residual_call guards (GUARD_NOT_FORCED, GUARD_NO_EXCEPTION):
        //   RPython generate_guard passes resumepc=-1, and capture_resumedata
        //   skips the "frame.pc = resumepc" assignment — frame.pc stays at
        //   the auto-advanced next instruction (pyre fallthrough_pc equivalent).
        //   This ensures the liveness PC, header ni, and blackhole resume PC
        //   all point to the instruction AFTER the call, not the call itself.
        let resume_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };
        self.generate_guard_core(ctx, opcode, args, resume_pc, after_residual_call);
    }

    /// Core guard recording with explicit resume PC.
    ///
    /// pyjitpl.py:2558-2584 generate_guard parity: record guard op,
    /// then call capture_resumedata.
    fn generate_guard_core(
        &mut self,
        ctx: &mut TraceCtx,
        opcode: OpCode,
        args: &[OpRef],
        resume_pc: usize,
        after_residual_call: bool,
    ) {
        self.flush_to_frame_for_guard(ctx);
        let active_boxes = self.get_list_of_active_boxes(ctx, false, after_residual_call);
        let snapshot_full_types = self.build_fail_arg_types_for_active_boxes(&active_boxes);
        let fail_arg_types = snapshot_full_types.clone();

        // Task #91 slice 4a: snapshot is the source of truth — the
        // optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from the
        // snapshot built below via `op.store_final_boxes(liveboxes)`
        // (mod.rs:3392), so the inline `fail_args` copy that the legacy
        // `record_guard_typed_with_fail_args` path used to write was
        // redundant.  Mirrors RPython
        // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
        // which records the guard with no inline fail_args and lets
        // `capture_resumedata` + `_number_boxes` populate them from the
        // snapshot chain.
        ctx.record_guard_typed(opcode, args, fail_arg_types);

        // pyjitpl.py:2579: self.capture_resumedata(resumepc, after_residual_call)
        self.capture_resumedata(
            ctx,
            resume_pc,
            after_residual_call,
            &active_boxes,
            &snapshot_full_types,
        );
        // pyjitpl.py:2581: self.staticdata.profiler.count_ops(opnum, Counters.GUARDS).
        // Atomic fetch_add through the shared `Arc<MetaInterpStaticData>`
        // — `&self` access is enough because `JitProfiler::count_ops`
        // bumps an `AtomicUsize` (Task #17 infrastructure).
        ctx.profiler()
            .count_ops(opcode, majit_metainterp::counters::GUARDS);
    }

    /// pyjitpl.py:2586-2602 capture_resumedata parity.
    ///
    /// Temporarily sets frame.pc = resumepc, captures the full framestack
    /// ([self as top/callee] + self.parent_frames as parents) plus
    /// virtualizable_boxes + virtualref_boxes into a snapshot, then
    /// restores frame.pc.  Matches opencoder.py:819-832
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, after_residual_call=False)` which walks the full
    /// `self.framestack` to build one SnapshotFrame per live frame.
    fn capture_resumedata(
        &mut self,
        ctx: &mut TraceCtx,
        resume_pc: usize,
        after_residual_call: bool,
        active_boxes: &[OpRef],
        snapshot_full_types: &[Type],
    ) {
        // pyjitpl.py:2594-2596: saved_pc = frame.pc; frame.pc = resumepc
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        // The snapshot's frame.pc must match the liveness PC used by
        // get_list_of_active_boxes.
        let snapshot_live_pc = if after_residual_call {
            self.fallthrough_pc
        } else {
            self.orgpc
        };

        // pyjitpl.py:2597-2600: history.trace.capture_resumedata(
        //     self.framestack, virtualizable_boxes, self.virtualref_boxes,
        //     after_residual_call)
        let snapshot = self.build_framestack_snapshot(
            ctx,
            snapshot_live_pc,
            active_boxes,
            snapshot_full_types,
        );
        let snapshot_id = ctx.capture_resumedata(snapshot);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// Extend a single-frame callee's `fail_arg_types` with the type
    /// contribution of every active parent frame.  Mirrors the
    /// `pyjitpl.py:2597` generate_guard collection: parent boxes ride
    /// the snapshot (built by `build_framestack_snapshot`), and the
    /// optimizer's `store_final_boxes_in_guard` (`optimizeopt/mod.rs:
    /// 3200`) overwrites `op.fail_args` from the snapshot via
    /// `op.store_final_boxes(liveboxes)` (mod.rs:3392), so an inline
    /// `fail_args` collection is redundant — only the per-frame
    /// `parent_types` need to flow into `record_guard_typed`.
    ///
    /// `build_framestack_snapshot` strips the `[s.frame, s.vable_*..]`
    /// scalar inputarg header from each parent frame's snapshot boxes
    /// (`opencoder.py:806` — parent frames contribute only active
    /// boxes; the virtualizable section is emitted once for the whole
    /// snapshot via `list_of_boxes_virtualizable`).  Mirror that here
    /// so the static type vector matches the static box shape per
    /// frame, otherwise the inline `fail_arg_types` stretched parent
    /// types over the snapshot's header-less box positions before the
    /// optimizer's snapshot-derived overwrite hid the divergence.
    fn extend_types_with_parents(&mut self, ctx: &mut TraceCtx, mut types: Vec<Type>) -> Vec<Type> {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        for parent in self.parent_frames.clone() {
            let (parent_types_full, _jitcode_index) =
                self.materialize_parent_frame_state(ctx, parent);
            if parent_types_full.len() > n {
                types.extend_from_slice(&parent_types_full[n..]);
            }
        }
        types
    }

    /// Build the full framestack `Snapshot` — top (callee) frame
    /// followed by every parent frame in `self.parent_frames` — plus
    /// virtualizable and virtualref boxes.  Mirrors opencoder.py:819-832
    /// `capture_resumedata(framestack, virtualizable_boxes,
    /// virtualref_boxes, ...)`.
    ///
    /// The caller is responsible for swapping `self.orgpc` if the
    /// snapshot pc differs from the current orgpc (RPython
    /// pyjitpl.py:2594-2602 MIFrame.capture_resumedata does the same
    /// before calling `history.trace.capture_resumedata`), and for
    /// computing `top_active_boxes` / `top_snapshot_types_full` under
    /// the liveness that applies to the swapped pc.
    fn build_framestack_snapshot(
        &mut self,
        ctx: &mut TraceCtx,
        top_pc: usize,
        top_active_boxes: &[OpRef],
        top_snapshot_types_full: &[Type],
    ) -> majit_metainterp::recorder::Snapshot {
        let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let top_snapshot_types = &top_snapshot_types_full[n..];
        let top_jitcode_index = unsafe { (*self.sym().jitcode).index } as u32;
        let mut frames = vec![majit_metainterp::recorder::SnapshotFrame {
            jitcode_index: top_jitcode_index,
            pc: top_pc as u32,
            boxes: Self::fail_args_to_snapshot_boxes_typed(
                top_active_boxes,
                top_snapshot_types,
                ctx,
            ),
        }];
        // opencoder.py:806: parent frames keep their original pc.
        // Snapshot boxes = active boxes only (skip scalar inputarg header).
        for parent in self.parent_frames.clone() {
            let (parent_types_full, parent_jitcode_index, parent_active) =
                self.materialize_parent_snapshot_state(ctx, parent);
            let parent_types: &[Type] = if parent_types_full.len() > n {
                &parent_types_full[n..]
            } else {
                &[]
            };
            frames.push(majit_metainterp::recorder::SnapshotFrame {
                jitcode_index: parent_jitcode_index,
                pc: parent.resume_pc as u32,
                boxes: Self::fail_args_to_snapshot_boxes_typed(&parent_active, parent_types, ctx),
            });
        }
        // opencoder.py:217 `SnapshotIterator.__init__` calls
        // `self.framestack.reverse()` so the numbering loop at
        // `resume.py:249-253` iterates outermost-first.  Pyre's
        // encoder builds `frames` innermost-first
        // (`[top, immediate_caller, ..., outermost]`); reverse here
        // so `Snapshot.frames[0]` is outermost-first, matching the
        // documented contract on `recorder.rs:54` "Frames in the
        // snapshot, outermost first" and `resume.rs:252`
        // "Input tuples for this factory follow the same caller-first
        // order".
        frames.reverse();
        let vable_boxes = self.list_of_boxes_virtualizable(ctx);
        let vref_boxes = Self::build_virtualref_boxes(self.sym(), ctx);
        // PHASE 1.4 candidate D probe: detect snapshot-time divergence
        // between vable_boxes (heap mirror) and registers_r (machine
        // register source). Both should be populated by store_local_value's
        // dual-write. Any divergence here means a code path updated one
        // shadow without the other — most likely load_local_value's lazy
        // fallback (trace_opcode.rs:1041-1063) which writes registers_r
        // but does NOT call set_virtualizable_box_at. See
        // memory/phase_1_4_cand_a_landed_raise_catch_diagnostic_2026_04_26.md.
        if std::env::var("PYRE_PROBE_SNAPSHOT").ok().as_deref() == Some("1") {
            let num_static = ctx
                .virtualizable_info()
                .map(|info| info.num_static_extra_boxes)
                .unwrap_or(0);
            let nlocals = self.sym().nlocals;
            let registers_r_src: Vec<OpRef> = if let Some(ref pre_r) = self.pre_opcode_registers_r {
                pre_r[..pre_r.len().min(nlocals)].to_vec()
            } else {
                let s = self.sym();
                s.registers_r[..s.registers_r.len().min(nlocals)].to_vec()
            };
            let mut diverge = 0usize;
            for i in 0..registers_r_src.len() {
                let reg_op = registers_r_src[i];
                let vable_op = ctx
                    .virtualizable_box_at(num_static + i)
                    .unwrap_or(OpRef::NONE);
                if !reg_op.is_none() && reg_op != vable_op {
                    eprintln!(
                        "[PROBE-D] vable/reg divergence top_pc={} local={} reg_opref={:?} vable_opref={:?}",
                        top_pc, i, reg_op, vable_op
                    );
                    diverge += 1;
                }
            }
            eprintln!(
                "[PROBE-D] ENTER top_pc={} nlocals={} reg_len={} num_static={} diverge_count={}",
                top_pc,
                nlocals,
                registers_r_src.len(),
                num_static,
                diverge
            );
        }
        majit_metainterp::recorder::Snapshot {
            frames,
            vable_boxes,
            vref_boxes,
        }
    }

    fn materialize_parent_frame_state(
        &mut self,
        ctx: &mut TraceCtx,
        parent: ResumeFrameState,
    ) -> (Vec<Type>, u32) {
        let (full_types, jitcode_index, _active_boxes) =
            self.materialize_parent_snapshot_state(ctx, parent);
        (full_types, jitcode_index)
    }

    fn materialize_parent_snapshot_state(
        &mut self,
        ctx: &mut TraceCtx,
        parent: ResumeFrameState,
    ) -> (Vec<Type>, u32, Vec<OpRef>) {
        // pyjitpl.py:2586 capture_resumedata parity: parent frames
        // contribute only their per-frame regular boxes (locals + stack)
        // to the snapshot.  The virtualizable scalars are emitted once
        // via `list_of_boxes_virtualizable` for the whole snapshot, not
        // per-frame.  Earlier pyre revisions called
        // `parent_frame.flush_to_frame_for_guard(ctx)` here and built a
        // `[s.frame, s.vable_*..., active_boxes]` fail_args vec for the
        // legacy `record_guard_typed_with_fail_args` path; both are dead
        // since the snapshot reader (`build_framestack_snapshot`) reads
        // active boxes directly and `store_final_boxes_in_guard`
        // overwrites `op.fail_args` from the snapshot
        // (`optimizeopt/mod.rs:3200`).  Keeping the parent flush around
        // mutates parent's `s.vable_*` mid-callee snapshot — the
        // structural blocker called out in
        // `vable_shadow_split_brain_load_bearing_2026_04_28` — so it is
        // dropped together with the dead fail_args build.
        let parent_sym = unsafe { &mut *parent.sym };
        let mut parent_frame = MIFrame::from_sym(
            ctx,
            parent_sym,
            parent.concrete_frame_addr,
            parent.resume_pc,
            parent.resume_pc,
        );
        parent_frame.pending_result_stack_idx = parent.pending_result_stack_idx;
        parent_frame.pending_result_type = parent.pending_result_type;
        let active_boxes = parent_frame.get_list_of_active_boxes(ctx, true, false);
        let full_types = parent_frame.build_fail_arg_types_for_active_boxes(&active_boxes);
        let jitcode_index = unsafe { (*parent_frame.sym().jitcode).index } as u32;
        (full_types, jitcode_index, active_boxes)
    }

    /// virtualizable.py:139 _get_virtualizable_field_boxes parity:
    /// [static_fields..., array_items..., virtualizable_ptr].
    /// pyjitpl.py:2586: self.virtualizable_boxes → vable_array.
    /// opencoder.py:603 _encode parity: encode OpRef as SnapshotTagged.
    /// Constant-pool OpRefs → Const(value, type) from pool.
    /// NONE → Const(0, Ref). Regular → Box.
    fn opref_to_snapshot_tagged(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        Self::opref_to_snapshot_tagged_for_slot(opref, ctx, None)
    }

    /// virtualizable.py:86-98 `read_boxes(cpu, virtualizable, startindex)` parity:
    /// each slot is wrapped via `wrap(cpu, value, startindex + i)` where the
    /// lltype (ARRAYITEMTYPE or static field `FIELDTYPE`) is declared and
    /// determines the resulting Const's INT/REF/FLOAT kind. pyre stores
    /// constants in a unified pool whose stored `const_type` may disagree
    /// with the slot's declared type (e.g. pointer constants opened via
    /// `const_int`), so snapshot encoding must prefer the slot's declared
    /// type when it is known — otherwise `_gettagged` → `getconst(val, tp)`
    /// picks TAGINT for a Ref-typed slot and the resume reader decodes a
    /// raw i64 where a PyObjectRef is expected.
    fn opref_to_snapshot_tagged_for_slot(
        opref: OpRef,
        ctx: &majit_metainterp::TraceCtx,
        declared_type: Option<majit_ir::Type>,
    ) -> majit_metainterp::recorder::SnapshotTagged {
        if opref.is_none() {
            majit_metainterp::recorder::SnapshotTagged::Const(
                0,
                declared_type.unwrap_or(majit_ir::Type::Ref),
            )
        } else if ctx.constant_value(opref).is_some() {
            let val = ctx.constant_value(opref).unwrap_or(0);
            // resume.py:157-183 `getconst(const)` dispatches on `const.type`.
            // Prefer the pool's actual const type over `declared_type`:
            // Box.type is immutable, so an Int-typed constant (e.g. an
            // intbounds-promoted local) must stay Int even when the slot
            // layout declares Ref. Retyping it here would seed the bridge
            // optimizer's const_pool with `Value::Ref(GcRef(small_int))`
            // and later trip the getintbound forwarding assertion. Fall
            // back to `declared_type` only when the pool has no type for
            // this OpRef (e.g. raw-pointer constants seeded without a
            // const_type entry).
            let tp = ctx
                .const_type(opref)
                .or(declared_type)
                .unwrap_or(majit_ir::Type::Int);
            majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
        } else {
            // resume.py:211,214: box.type for _number_boxes TAGVIRTUAL/TAGBOX.
            let tp = ctx
                .get_opref_type(opref)
                .unwrap_or_else(|| panic!("missing snapshot box type for {:?}", opref));
            majit_metainterp::recorder::SnapshotTagged::Box(opref.raw(), tp)
        }
    }

    /// RPython pyjitpl.py:2586 virtualizable_boxes parity.
    ///
    /// RPython creates SEPARATE Box objects for virtualizable_boxes via
    /// read_boxes()/wrap() — these are distinct from frame register boxes.
    /// _number_boxes dedup uses object identity, so vable and frame get
    /// independent TAGBOX indices → deadframe stores both.
    ///
    /// pyre uses the SAME OpRefs for both → _number_boxes dedup merges them
    /// → vable and frame sections share TAGBOX indices. Recovery uses frame
    /// sections with liveness-based mapping (restore_guard_failure_values),
    /// matching RPython's consume_boxes(position_info) architecture.
    ///
    /// Fresh identity approaches (VABLE_FRESH_BIT, VABLE_KEY_OFFSET)
    /// expand num_boxes → larger fail_args → deadframe/exit layout mismatch.
    /// Fix requires backend exit block recompilation after numbering,
    /// or trace-time SameAs emission for fresh vable OpRefs.
    fn list_of_boxes_virtualizable(
        &self,
        ctx: &mut majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        let sym = self.sym();
        // opencoder.py:718-726 _list_of_boxes_virtualizable parity:
        // RPython format: [virtualizable_ptr, static_fields..., array_items...]
        // (virtualizable_ptr moved from end to front).
        // virtualizable.py:86/139 read_boxes / load_list_of_boxes:
        // Memory order: [static_field_0, ..., array_items..., vable_ptr]
        // read_boxes creates fresh Box objects for each field via wrap().
        // opencoder.py:722 _list_of_boxes_virtualizable: reorders
        //   vable_ptr from end to front → snapshot = [vable_ptr, fields..., items...]
        let stack_only = sym.valuestackdepth.saturating_sub(sym.nlocals);
        let mut boxes = Vec::new();
        // opencoder.py:722: virtualizable_ptr FIRST.
        // The virtualizable frame pointer is always a GCREF.
        boxes.push(Self::opref_to_snapshot_tagged_for_slot(
            sym.frame,
            ctx,
            Some(majit_ir::Type::Ref),
        ));
        // Static fields in declared order (virtualizable.py:90-93).
        // virtualizable.py:131-133 wraps each value with its declared
        // `FIELDTYPE`; pyre mirrors that by consulting
        // `VirtualizableInfo::static_fields[i].field_type`.
        //
        // opencoder.py:718-726 `_list_of_boxes_virtualizable(boxes)`
        // parity: read from `ctx.virtualizable_boxes` (the canonical
        // analog of RPython's `metainterp.virtualizable_boxes`) for
        // the four invariant scalars (`pycode`, `debugdata`,
        // `lastblock`, `w_globals`), and recompute the two
        // per-opcode-advancing scalars (`last_instr`, `valuestackdepth`)
        // from `self.orgpc` / `pre_opcode_registers_r` /
        // `portal_bridge_vable_vsd(orgpc)` so the snapshot encodes
        // the pre-opcode state at `resume_pc` (the PROBE-VABLE-DIV
        // diagnostic on 2026-04-30 confirmed slot 0 / slot 2 are the
        // only divergence sources between the shared shadow and
        // `s.vable_*` — slots 1/3/4/5 always agree because their
        // mutators are unreachable under CPython 3.14 bytecode).
        //
        // The slot-0 inline override re-derives `resume_pc - 1`
        // because `flush_to_frame_for_guard` swaps `self.orgpc` to
        // `resume_pc` (capture_resumedata at line 2773), and writes
        // `s.vable_last_instr = const_int(resume_pc - 1)` without
        // mirroring to `ctx.virtualizable_boxes[0]`.  Reading
        // `ctx.virtualizable_boxes[0]` directly would pick up the
        // value `publish_last_instr_to_vable` wrote at the original
        // (pre-swap) orgpc, which is one off when `resume_pc !=
        // orgpc` (most branch guards).  The read-time recompute
        // matches `flush_to_frame_for_guard` so the snapshot stays
        // self-consistent.
        //
        // The slot-2 inline override re-derives the pre-opcode
        // valuestackdepth via `portal_bridge_vable_vsd(resume_pc)
        // .unwrap_or(pre_opcode_registers_r.len() / s.valuestackdepth)`
        // — same source `flush_to_frame_for_guard` uses to set
        // `s.vable_valuestackdepth`.
        //
        // Test-fixture fallback: `TraceCtx::for_test_types` callers
        // construct a ctx without registering `VirtualizableInfo` and
        // without seeding `ctx.virtualizable_boxes`.  In that mode
        // the `_with_compiled_trace_jitcode` fixtures expect
        // `sym.vable_field_oprefs()` to drive the snapshot — fall
        // back when the shared shadow is unseeded.
        let (vable_static_types, vsd_field_index, ni_field_index, num_static): (
            Vec<majit_ir::Type>,
            Option<usize>,
            Option<usize>,
            usize,
        ) = match ctx.virtualizable_info() {
            Some(info) => (
                info.static_fields.iter().map(|f| f.field_type).collect(),
                info.static_field_index_by_name("valuestackdepth"),
                info.static_field_index_by_name("last_instr"),
                info.num_static_extra_boxes,
            ),
            None => (Vec::new(), None, None, 0),
        };
        let pre_opcode_vsd: Option<i64> = if vsd_field_index.is_some() {
            let resume_pc = self.orgpc;
            Some(self.portal_bridge_vable_vsd(resume_pc).unwrap_or_else(|| {
                let s = self.sym();
                self.pre_opcode_registers_r
                    .as_ref()
                    .map(|pre_r| pre_r.len())
                    .unwrap_or(s.valuestackdepth) as i64
            }))
        } else {
            None
        };
        let pre_opcode_last_instr: Option<i64> = ni_field_index.map(|_| self.orgpc as i64 - 1);
        if num_static > 0 && ctx.has_virtualizable_boxes() {
            for idx in 0..num_static {
                let declared = vable_static_types.get(idx).copied();
                let opref = if Some(idx) == ni_field_index {
                    let li = pre_opcode_last_instr
                        .expect("pre_opcode_last_instr seeded when ni_field_index is Some");
                    ctx.const_int(li)
                } else if Some(idx) == vsd_field_index {
                    let vsd =
                        pre_opcode_vsd.expect("pre_opcode_vsd seeded when vsd_field_index is Some");
                    ctx.const_int(vsd)
                } else {
                    ctx.virtualizable_box_at(idx).unwrap_or(OpRef::NONE)
                };
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref, ctx, declared,
                ));
            }
        } else {
            // Test fallback: ctx has no vinfo and no seeded shadow.
            let fallback_types: Vec<majit_ir::Type> = if !vable_static_types.is_empty() {
                vable_static_types.clone()
            } else {
                vec![majit_ir::Type::Ref; sym.vable_field_oprefs().len()]
            };
            for (idx, opref) in sym.vable_field_oprefs().iter().enumerate() {
                let declared = fallback_types.get(idx).copied();
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    *opref, ctx, declared,
                ));
            }
        }
        // Array items: locals + stack (virtualizable.py:86 read_boxes).
        let _ = stack_only;
        let symbolic_stack_len = if let Some(ref pre_r) = self.pre_opcode_registers_r {
            pre_r.len().saturating_sub(sym.nlocals)
        } else {
            sym.registers_r.len().saturating_sub(sym.nlocals)
        };
        let concrete_frame = if !sym.concrete_vable_ptr.is_null() {
            Some(unsafe { &*(sym.concrete_vable_ptr as *const pyre_interpreter::pyframe::PyFrame) })
        } else {
            None
        };
        // virtualizable.py:86 read_boxes parity: encoder must emit one
        // box per slot in the heap-side `locals_cells_stack_w` array
        // because the decoder reads `vinfo.get_total_size(virtualizable)`
        // (= static_fields + heap array length) on the runtime PyFrame.
        // Using the symbolic current stack depth here was off by
        // (max_stackdepth - current_stack_depth) and produced
        // `vable_size - 1 != vinfo.get_total_size` panics whenever a
        // bridge tried to consume the snapshot at a state where the
        // physical frame had been allocated with stack room beyond the
        // current symbolic depth. Read the physical frame length and
        // pad missing slots with the live concrete value (or NULL).
        let physical_array_len = concrete_frame
            .map(|f| f.locals_w().len())
            .unwrap_or_else(|| {
                let current_vsd = self
                    .pre_opcode_registers_r
                    .as_ref()
                    .map(|pre_r| pre_r.len())
                    .unwrap_or(sym.valuestackdepth);
                let stack_depth = current_vsd
                    .saturating_sub(sym.nlocals)
                    .min(symbolic_stack_len);
                sym.nlocals + stack_depth
            });
        let full_array_len = physical_array_len;
        // virtualizable.py:135-137 `lst[j] = reader.load_next_value_of_type(
        // ARRAYITEMTYPE)` — every array slot is the array's declared item
        // type (GCREF for pyre's `locals_cells_stack_w`), regardless of
        // what the optimizer chose for the OpRef's own kind. Enforce this
        // at encoding time so a `LOAD_CONST 0` whose OpRef is Int-typed
        // still lands in the snapshot as a Ref constant.
        let array_item_type = ctx
            .virtualizable_info()
            .and_then(|info| info.array_fields.first().map(|a| a.item_type))
            .unwrap_or(majit_ir::Type::Ref);
        // virtualizable.py:86 read_boxes / opencoder.py:718 _list_of_boxes_virtualizable
        // parity: RPython snapshot reads directly from `self.virtualizable_boxes`,
        // which is the single source of truth mirrored by every
        // `_opimpl_setarrayitem_vable` via `synchronize_virtualizable`.
        // pyre's tracer mirrors EVERY write into `locals_cells_stack_w`
        // (locals via `store_local_value`, stack via `push_typed_value`
        // + `pop_value` + `swap_values` + `finishframe_exception`) into
        // `virtualizable_boxes`, so the shadow is the only source we
        // read here.
        let num_static = ctx
            .virtualizable_info()
            .map(|info| info.num_static_extra_boxes)
            .unwrap_or(0);
        for i in 0..full_array_len {
            // virtualizable_boxes layout:
            //   [field0, ..., fieldN, arr[0..M], vable_ref]
            // so array slot `i` lives at `num_static + i`. The trailing
            // `vable_ref` is at `virtualizable_boxes[-1]` and NEVER covers
            // an array slot — skip it via `.get()`.
            let opref = ctx
                .virtualizable_box_at(num_static + i)
                .unwrap_or(OpRef::NONE);
            if !opref.is_none() {
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    opref,
                    ctx,
                    Some(array_item_type),
                ));
            } else {
                // PRE-EXISTING-ADAPTATION: legacy `register == PyFrame
                // slot` conflation (plan Stage 3.4) lets STORE_FAST
                // write an unboxed int into `locals_cells_stack_w[i]`
                // when the trace IR optimizer promoted that local's
                // OpRef to Int. Reading those raw bits back here and
                // const-seeding them as `Const(val, Ref)` mistypes
                // the value in the bridge optimizer's const_pool —
                // verified root cause of the LoadFastLoadFast vable
                // conversion regression (memory:
                // vable_locals_lowering_analysis.md fix option b).
                //
                // Emit a NULL sentinel; the bridge resume will
                // re-fetch the actual slot value via
                // `vable_getarrayitem_r` against the live frame.
                let _ = concrete_frame;
                boxes.push(Self::opref_to_snapshot_tagged_for_slot(
                    OpRef::NONE,
                    ctx,
                    Some(array_item_type),
                ));
            }
        }
        boxes
    }

    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// pyjitpl.py:2597 virtualref_boxes parity.
    /// Returns pairs of (jit_virtual, real_vref) as SnapshotTagged.
    fn build_virtualref_boxes(
        sym: &PyreSym,
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        sym.virtualref_boxes
            .iter()
            .map(|&(opref, _concrete)| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// RPython pyjitpl.py:177 get_list_of_active_boxes parity:
    fn fail_args_to_snapshot_boxes(
        fail_args: &[OpRef],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        fail_args
            .iter()
            .map(|&opref| Self::opref_to_snapshot_tagged(opref, ctx))
            .collect()
    }

    /// snapshot boxes from active_boxes = [locals, stack].
    /// RPython: each Box carries type ('r'/'i'/'f') — pyre passes types
    /// explicitly so _number_boxes can detect virtual vs int correctly.
    fn fail_args_to_snapshot_boxes_typed(
        active_boxes: &[OpRef],
        types: &[majit_ir::Type],
        ctx: &majit_metainterp::TraceCtx,
    ) -> Vec<majit_metainterp::recorder::SnapshotTagged> {
        active_boxes
            .iter()
            .enumerate()
            .map(|(i, &opref)| {
                if opref.is_none() {
                    majit_metainterp::recorder::SnapshotTagged::Const(0, majit_ir::Type::Ref)
                } else if ctx.constant_value(opref).is_some() {
                    let val = ctx.constant_value(opref).unwrap_or(0);
                    // resume.py:157-183 `getconst(const)` dispatches on
                    // `const.type`; pyre's plain `const_int(v)` has an
                    // intrinsic INT type (see `opref_to_snapshot_tagged`).
                    let tp = ctx.const_type(opref).unwrap_or(majit_ir::Type::Int);
                    majit_metainterp::recorder::SnapshotTagged::Const(val, tp)
                } else {
                    let tp = types.get(i).copied().unwrap_or_else(|| {
                        panic!("missing fail-arg box type at index {} for {:?}", i, opref)
                    });
                    majit_metainterp::recorder::SnapshotTagged::Box(opref.raw(), tp)
                }
            })
            .collect()
    }

    /// pyjitpl.py:1916-1927 implement_guard_value parity.
    /// executor.py:544-551 constant_from_op(box): dispatches on box.type.
    pub(crate) fn implement_guard_value(
        &mut self,
        ctx: &mut TraceCtx,
        value: OpRef,
        expected: i64,
    ) {
        let expected_ref = match self.value_type(value) {
            majit_ir::Type::Ref => ctx.const_ref(expected),
            _ => ctx.const_int(expected),
        };
        self.generate_guard(ctx, OpCode::GuardValue, &[value, expected_ref]);
        // pyjitpl.py:3512: replace_box
        ctx.heap_cache_mut().replace_box(value, expected_ref);
    }

    pub(crate) fn guard_nonnull(&mut self, ctx: &mut TraceCtx, value: OpRef) {
        // pyjitpl.py:3517-3520 `_establish_nullity` cache hit:
        //     if heapcache.is_nullity_known(box):
        //         self.metainterp.staticdata.profiler.count_ops(rop.GUARD_NONNULL, Counters.HEAPCACHED_OPS)
        //         return value
        if ctx
            .heap_cache()
            .is_nullity_known(value, |op| ctx.const_value(op))
            == Some(true)
        {
            ctx.profiler().count_ops(
                OpCode::GuardNonnull,
                majit_metainterp::counters::HEAPCACHED_OPS,
            );
            return;
        }
        if ctx.heap_cache().is_class_known(value) {
            return; // class known implies nonnull (pyre-only short-circuit, no RPython count)
        }
        self.generate_guard(ctx, OpCode::GuardNonnull, &[value]);
        ctx.heap_cache_mut().nullity_now_known(value, true);
    }

    pub(crate) fn guard_range_iter(&mut self, ctx: &mut TraceCtx, obj: OpRef) {
        self.guard_class(ctx, obj, &RANGE_ITER_TYPE as *const PyType);
    }

    pub(crate) fn record_for_iter_guard(
        &mut self,
        ctx: &mut TraceCtx,
        next: OpRef,
        continues: bool,
    ) {
        // RPython range/xrange iter traces carry the next value as a raw int,
        // not an optional boxed object. Only ref-typed iterators need the
        // optional-value guard here.
        if self.value_type(next) != Type::Ref {
            return;
        }
        let opcode = if continues {
            OpCode::GuardNonnull
        } else {
            OpCode::GuardIsnull
        };
        self.generate_guard(ctx, opcode, &[next]);
        // heapcache: track nullity after for-iter guard
        ctx.heap_cache_mut().nullity_now_known(next, continues);
    }

    pub(crate) fn record_branch_guard(
        &mut self,
        ctx: &mut TraceCtx,
        _branch_value: OpRef,
        truth: OpRef,
        concrete_truth: bool,
        other_target: usize,
    ) {
        let opcode = if concrete_truth {
            OpCode::GuardTrue
        } else {
            OpCode::GuardFalse
        };

        // pyjitpl.py:510-520 goto_if_not(box, target, orgpc):
        //   self.metainterp.generate_guard(opnum, box, resumepc=orgpc)
        // blackhole.py:864 bhimpl_goto_if_not(a, target, pc) re-pops the
        // truth from a register and re-decides which arm to take.
        //
        // RPython's goto_if_not is a compound jitcode opcode produced by
        // jtransform.optimize_goto_if_not (jtransform.py:196): COMPARE_OP +
        // GOTO_IF_NOT are fused into a single op that takes the comparison
        // operands directly and branches. The trace's "register" for the
        // truth and the jitcode's "register" agree on type (int).
        //
        // Pyre emits the same fused op at tracer-dispatch level via
        // try_fused_compare_goto_if_not (pyjitpl.py:541-556 parity): the
        // fused dispatch pops COMPARE's operands, emits IntLt/FloatLt,
        // and calls into this function with the raw Int truth — the
        // symbolic stack carries no comparison value at the guard site.
        // Resuming at other_target (the runtime branch destination)
        // matches the fused jitcode's semantics: the blackhole skips the
        // separate POP_JUMP_IF_FALSE entirely and re-enters the
        // interpreter at the not-taken branch. Inline-frame branch
        // guards use the same resume_pc; parent frames keep their own
        // return_point pc.
        let resume_pc = other_target;

        // pyjitpl.py:2593-2602: saved_pc = frame.pc; frame.pc = resumepc;
        // capture_resumedata(); frame.pc = saved_pc
        // Save ALL state BEFORE flush (generate_guard_core parity).
        let saved_orgpc = self.orgpc;
        let saved_ni = self.sym().vable_last_instr;
        let saved_vsd = self.sym().vable_valuestackdepth;
        self.orgpc = resume_pc;

        // Branch guards resume at `other_target` (the runtime jump destination,
        // not the POP_JUMP_IF_* opcode's orgpc). At `other_target` the Python
        // interpreter has already popped the comparison truth, so the snapshot
        // must reflect the POST-POP register file. `pre_opcode_registers_r`
        // was captured at the START of POP_JUMP_IF
        // (PRE-POP) and still carry the Int truth — using them here would
        // emit a `Box(truth_opref, Int)` at a Ref-declared
        // `locals_cells_stack_w` slot and corrupt the PyFrame when the
        // guard resumes (the decoder writes a raw i64 where a PyObjectRef
        // is expected).
        //
        // pyjitpl.py:2586-2602 capture_resumedata mutates `frame.pc` to
        // `resumepc` for the duration of the capture and restores it on
        // exit; RPython has no pre-opcode snapshot because its jitcode is
        // register-machine and liveness is per-PC. Pyre's pre_opcode_*
        // fields are a NEW-DEVIATION that will be removed once per-PC
        // liveness is ported (codewriter/liveness.py). Until then, clear
        // them here so flush_to_frame_for_guard, get_list_of_active_boxes,
        // and list_of_boxes_virtualizable all read the current (post-pop)
        // `registers_r` / valuestackdepth. The next trace_code_step
        // re-captures the snapshot, so no restore is needed.
        self.clear_pre_opcode_state();

        self.flush_to_frame_for_guard(ctx);
        // pyjitpl.py:177: get_list_of_active_boxes uses frame.pc for liveness
        let callee_active_boxes = self.get_list_of_active_boxes(ctx, false, false);
        let callee_snapshot_types_full =
            self.build_fail_arg_types_for_active_boxes(&callee_active_boxes);

        // opencoder.py:819 capture_resumedata(framestack) parity:
        // Encode the full framestack [callee (top) with resume_pc=other_target,
        // caller(s) with return_point pc]. Branch guards in inline callees
        // use the same other_target adaptation — the callee frame's resume
        // pc points past POP_JUMP_IF_* (not at it), while parent frames
        // keep their original return_point pc.
        let snapshot = self.build_framestack_snapshot(
            ctx,
            resume_pc,
            &callee_active_boxes,
            &callee_snapshot_types_full,
        );
        // Task #91 slice 4a: snapshot is the source of truth — the
        // optimizer's `store_final_boxes_in_guard`
        // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args` from
        // the snapshot via `op.store_final_boxes(liveboxes)`
        // (mod.rs:3392), so the inline `fail_args` copy that the legacy
        // `record_guard_typed_with_fail_args` path used to write was
        // redundant.  Mirrors RPython
        // `pyjitpl.MetaInterp.generate_guard` (pyjitpl.py:2558-2602)
        // which records the guard with no inline fail_args and lets
        // `capture_resumedata` + `_number_boxes` populate them from the
        // snapshot chain.
        let types = self.extend_types_with_parents(ctx, callee_snapshot_types_full);
        let snapshot_id = ctx.capture_resumedata(snapshot);

        ctx.record_guard_typed(opcode, &[truth], types);
        ctx.set_last_guard_resume_position(snapshot_id);

        // pyjitpl.py:2602: frame.pc = saved_pc (restore all, generate_guard_core parity)
        self.orgpc = saved_orgpc;
        let s = self.sym_mut();
        s.vable_last_instr = saved_ni;
        s.vable_valuestackdepth = saved_vsd;
    }

    /// pyjitpl.py:541-556 opimpl_goto_if_not_<op>_<type> parity.
    ///
    /// Fused COMPARE_OP + POP_JUMP_IF_FALSE/TRUE dispatch. Mirrors the
    /// RPython jitcode-level `goto_if_not_int_<op>` fused opcode: emits a
    /// single IntLt/FloatLt followed by GUARD_TRUE/GUARD_FALSE without ever
    /// pushing the comparison truth as a stack value.
    ///
    /// Returns Ok(Some(action)) when the fused path consumed both the
    /// COMPARE_OP and the following PopJumpIf*. Returns Ok(None) when
    /// fuseability fails (next instruction is not PopJumpIf*, or the
    /// operands are not a compatible int/float pair) — caller falls back
    /// to non-fused per-opcode dispatch.
    pub(crate) fn try_fused_compare_goto_if_not(
        &mut self,
        code: &CodeObject,
        compare_pc: usize,
        op: ComparisonOperator,
    ) -> Result<Option<TraceAction>, PyError> {
        // Peek next non-trivia instruction for PopJumpIf*.
        let branch_pc = crate::metainterp::semantic_fallthrough_pc(code, compare_pc);
        let Some((branch_instr, branch_op_arg)) = decode_instruction_at(code, branch_pc) else {
            return Ok(None);
        };
        let (jump_if_true, delta) = match branch_instr {
            Instruction::PopJumpIfFalse { delta } => (false, delta.get(branch_op_arg).as_usize()),
            Instruction::PopJumpIfTrue { delta } => (true, delta.get(branch_op_arg).as_usize()),
            _ => return Ok(None),
        };

        // Fuseability gate: peek concrete values without popping. Only
        // commit to fusion when the codegen int/float fast path will
        // succeed (matches codegen.rs:1056 is_int×is_int and
        // codegen.rs:1103-1109 float-pair conditions). Inspect the
        // ConcreteValue variants directly so the gate does not allocate
        // a throwaway w_int box just to read its type.
        let (a_concrete, b_concrete) = {
            let s = self.sym();
            let top_idx = s
                .valuestackdepth
                .checked_sub(s.nlocals + 1)
                .ok_or_else(|| PyError::type_error("stack underflow in fused compare"))?;
            let a_idx = top_idx
                .checked_sub(1)
                .ok_or_else(|| PyError::type_error("stack underflow in fused compare"))?;
            (
                s.concrete_stack
                    .get(a_idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null),
                s.concrete_stack
                    .get(top_idx)
                    .copied()
                    .unwrap_or(ConcreteValue::Null),
            )
        };
        let (lhs_is_int, lhs_is_float) = classify_concrete(a_concrete);
        let (rhs_is_int, rhs_is_float) = classify_concrete(b_concrete);
        let ints_ok = lhs_is_int && rhs_is_int;
        let floats_ok = (lhs_is_float || lhs_is_int)
            && (rhs_is_float || rhs_is_int)
            && (lhs_is_float || rhs_is_float);
        if !ints_ok && !floats_ok {
            return Ok(None);
        }

        // Materialize the full concrete values only after the gate
        // succeeds; compare_value_direct + objspace_compare_{ints,floats}
        // both need them in PyObjectRef form to call into baseobjspace.
        let lhs_obj = a_concrete.to_pyobj();
        let rhs_obj = b_concrete.to_pyobj();
        if lhs_obj.is_null() || rhs_obj.is_null() {
            return Ok(None);
        }

        // Compute branch destinations (matches opcode_pop_jump_if at
        // pyopcode.rs:556). `fallthrough` is MIFrame's semantic_fallthrough_pc
        // past trivia; `target` is jump_target_forward (skip_caches + delta).
        let branch_fallthrough = crate::metainterp::semantic_fallthrough_pc(code, branch_pc);
        let branch_target =
            pyre_interpreter::jump_target_forward(&code.instructions, branch_pc + 1, delta);

        // Compute concrete branch direction locally. Matches baseobjspace
        // ::compare int/float pair semantics so the fused path does not
        // depend on any tracer-side cache written during trace emission.
        let concrete_truth = if ints_ok {
            unsafe { crate::state::objspace_compare_ints(lhs_obj, rhs_obj, op) }
        } else {
            unsafe { crate::state::objspace_compare_floats(lhs_obj, rhs_obj, op) }
        };

        // Commit: pop b, pop a (matches opcode_compare_op at pyopcode.rs:530).
        let b_ref = self.with_ctx(|this, ctx| this.pop_value(ctx))?;
        let a_ref = self.with_ctx(|this, ctx| this.pop_value(ctx))?;

        // Emit IntLt/FloatLt via compare_value_direct (generated int/float
        // fast path). `next_instruction_consumes_comparison_truth()` is
        // true here (the fuseability gate already saw PopJumpIf* as the
        // next non-trivia instruction), so the returned OpRef is the raw
        // Int truth, NOT a boxed bool.
        let truth = self.compare_value_direct(a_ref, b_ref, op, lhs_obj, rhs_obj)?;

        // Decide branch direction (opcode_pop_jump_if at pyopcode.rs:565-572).
        let should_jump = concrete_truth == jump_if_true;
        let other_target = if should_jump {
            branch_fallthrough
        } else {
            branch_target
        };
        let next_target = if should_jump {
            branch_target
        } else {
            branch_fallthrough
        };

        // Emit GUARD_TRUE/GUARD_FALSE. record_branch_guard swaps orgpc to
        // resume_pc (other_target), suppresses pre_opcode_*, and builds
        // the framestack snapshot before restoring state.
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, truth, truth, concrete_truth, other_target);
            Ok::<(), PyError>(())
        })?;

        // Advance past the branch; the outer dispatch loop picks this up.
        self.sym_mut().pending_next_instr = Some(next_target);
        Ok(Some(TraceAction::Continue))
    }

    /// RPython registers[idx] parity: read concrete value from Box arrays.
    fn concrete_at(&self, abs_idx: usize) -> Option<PyObjectRef> {
        let v = self.sym().concrete_value_at(abs_idx);
        if !v.is_null() {
            return Some(v.to_pyobj());
        }
        None
    }

    fn guard_int_object_value(&mut self, ctx: &mut TraceCtx, int_obj: OpRef, expected: i64) {
        self.guard_class(ctx, int_obj, &INT_TYPE as *const PyType);
        let actual_value = opimpl_getfield_gc_i(ctx, int_obj, int_intval_descr());
        self.implement_guard_value(ctx, actual_value, expected);
    }

    pub(crate) fn guard_int_like_value(&mut self, ctx: &mut TraceCtx, value: OpRef, expected: i64) {
        if self.value_type(value) == Type::Int {
            self.implement_guard_value(ctx, value, expected);
        } else {
            self.guard_int_object_value(ctx, value, expected);
        }
    }

    /// pyjitpl.py:1518 opimpl_guard_class
    pub(crate) fn guard_class(
        &mut self,
        ctx: &mut TraceCtx,
        obj: OpRef,
        expected_type: *const PyType,
    ) {
        // heapcache.py: skip guard if class already known for this object
        if ctx.heap_cache().is_class_known(obj) {
            return;
        }
        // pyjitpl.py:2558-2560 generate_guard parity:
        //     if isinstance(box, Const):    # no need for a guard
        //         return
        // The concrete value (and therefore its class) is known at trace
        // time, so the runtime type check is guaranteed to pass. RPython
        // also short-circuits before capture_resumedata, so no snapshot is
        // attached for the skipped guard. heapcache.class_now_known is
        // still called below — pyjitpl.py:1523 opimpl_guard_class invokes
        // it unconditionally after generate_guard.
        if obj.is_constant() {
            // pyjitpl.py:1087 parity: a pending GUARD_NOT_INVALIDATED from a
            // preceding quasi-immut field read must still be flushed even
            // when the type guard is skipped, otherwise the watcher and the
            // trace's quasi-immut dependency would be silently dropped.
            self.flush_guard_not_invalidated(ctx);
            ctx.heap_cache_mut()
                .class_now_known(obj, majit_ir::GcRef(expected_type as usize));
            return;
        }
        let expected_type_const = ctx.const_int(expected_type as usize as i64);
        self.generate_guard(ctx, OpCode::GuardNonnullClass, &[obj, expected_type_const]);
        // heapcache.py:470-473: class_now_known sets class + nullity.
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(expected_type as usize));
    }

    pub(crate) fn trace_guarded_int_payload(
        &mut self,
        ctx: &mut TraceCtx,
        int_obj: OpRef,
    ) -> OpRef {
        if self.value_type(int_obj) == Type::Int {
            return int_obj;
        }
        self.guard_class(ctx, int_obj, &INT_TYPE as *const PyType);
        opimpl_getfield_gc_i(ctx, int_obj, int_intval_descr())
    }

    pub(crate) fn guard_len_gt_index(&mut self, ctx: &mut TraceCtx, len: OpRef, index: usize) {
        let index = ctx.const_int(index as i64);
        let in_bounds = ctx.record_op(OpCode::IntGt, &[len, index]);
        self.generate_guard(ctx, OpCode::GuardTrue, &[in_bounds]);
    }

    pub(crate) fn guard_len_eq(&mut self, ctx: &mut TraceCtx, len: OpRef, expected: usize) {
        self.implement_guard_value(ctx, len, expected as i64);
    }

    pub(crate) fn guard_list_strategy(&mut self, ctx: &mut TraceCtx, obj: OpRef, expected: i64) {
        let strategy = opimpl_getfield_gc_i(ctx, obj, list_strategy_descr());
        self.implement_guard_value(ctx, strategy, expected);
    }

    /// pyjitpl.py:841-852 opimpl_check_resizable_neg_index delegate.
    pub(crate) fn trace_dynamic_list_index(
        &mut self,
        ctx: &mut TraceCtx,
        key: OpRef,
        len: OpRef,
        concrete_key: i64,
    ) -> OpRef {
        crate::generated_dynamic_list_index(self, ctx, key, len, concrete_key)
    }

    /// Unpack a known-length tuple. `tupleobject.py:376-390`
    /// `W_TupleObject` carries `wrappeditems` only; length is read
    /// from the GcArray header via `arraylen_gc(items_block,
    /// pyobject_gcarray_descr)`. Guard order matches pyjitpl.py:832-836:
    /// `guard_class(tuple) → getfield_gc_pure_r(wrappeditems) →
    /// arraylen_gc(items_block) → implement_guard_value(len, count) →
    /// per-index getarrayitem_gc_pure_r`.
    fn trace_unpack_known_tuple(
        &mut self,
        ctx: &mut TraceCtx,
        seq: OpRef,
        count: usize,
        items_descr: DescrRef,
    ) -> Vec<OpRef> {
        self.guard_class(ctx, seq, &TUPLE_TYPE as *const PyType);

        let items_block = crate::state::opimpl_getfield_gc_r(ctx, seq, items_descr);
        let len = crate::state::opimpl_arraylen_gc(
            ctx,
            items_block,
            crate::state::pyobject_gcarray_descr(),
        );
        self.implement_guard_value(ctx, len, count as i64);

        (0..count)
            .map(|idx| {
                let idx = ctx.const_int(idx as i64);
                crate::state::trace_items_block_getitem_value(ctx, items_block, idx)
            })
            .collect()
    }

    /// Unpack a known-length list under the Object strategy. List
    /// keeps the inline `length` field (rlist.py:116 `("length",
    /// Signed)`) so the length comes via `getfield_gc_i(length_descr)`,
    /// not `arraylen_gc`. Items live in the same `Ptr(GcArray(
    /// OBJECTPTR))` shape as tuples, read via `getarrayitem_gc_r`
    /// against `pyobject_gcarray_descr`.
    fn trace_unpack_known_list(
        &mut self,
        ctx: &mut TraceCtx,
        seq: OpRef,
        count: usize,
        length_descr: DescrRef,
        items_descr: DescrRef,
    ) -> Vec<OpRef> {
        self.guard_class(ctx, seq, &LIST_TYPE as *const PyType);

        let len = opimpl_getfield_gc_i(ctx, seq, length_descr);
        self.implement_guard_value(ctx, len, count as i64);

        let items_block = crate::state::opimpl_getfield_gc_r(ctx, seq, items_descr);
        (0..count)
            .map(|idx| {
                let idx = ctx.const_int(idx as i64);
                crate::state::trace_items_block_getitem_value(ctx, items_block, idx)
            })
            .collect()
    }

    pub(crate) fn unpack_sequence_value(
        &mut self,
        seq: OpRef,
        count: usize,
        concrete_seq: PyObjectRef,
    ) -> Result<Vec<FrontendOp>, PyError> {
        if concrete_seq.is_null() {
            let oprefs = TraceHelperAccess::trace_unpack_sequence(self, seq, count)?;
            return Ok(oprefs.into_iter().map(FrontendOp::void).collect());
        }

        // Extract concrete items from the sequence for RPython Box parity.
        let concrete_items: Vec<PyObjectRef> = unsafe {
            if is_tuple(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_tuple_getitem(concrete_seq, i as i64))
                    .collect()
            } else if is_list(concrete_seq) && w_list_uses_object_storage(concrete_seq) {
                (0..count)
                    .filter_map(|i| w_list_getitem(concrete_seq, i as i64))
                    .collect()
            } else {
                Vec::new()
            }
        };

        let oprefs = self.with_ctx(|this, ctx| unsafe {
            if is_tuple(concrete_seq) && w_tuple_len(concrete_seq) == count {
                return Ok(this.trace_unpack_known_tuple(
                    ctx,
                    seq,
                    count,
                    crate::descr::tuple_wrappeditems_descr(),
                ));
            }
            if is_list(concrete_seq)
                && w_list_uses_object_storage(concrete_seq)
                && w_list_len(concrete_seq) == count
            {
                return Ok(this.trace_unpack_known_list(
                    ctx,
                    seq,
                    count,
                    crate::descr::list_length_descr(),
                    crate::descr::list_items_descr(),
                ));
            }
            TraceHelperAccess::trace_unpack_sequence(this, seq, count)
        })?;

        Ok(oprefs
            .into_iter()
            .enumerate()
            .map(|(i, opref)| {
                let cv = concrete_items
                    .get(i)
                    .copied()
                    .map(ConcreteValue::from_pyobj)
                    .unwrap_or(ConcreteValue::Null);
                FrontendOp::new(opref, cv)
            })
            .collect())
    }

    pub(crate) fn binary_subscr_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        // Concrete subscr result for FrontendOp Box tracking.
        let subscr_concrete = if !concrete_obj.is_null() && !concrete_key.is_null() {
            if let Ok(result) = pyre_interpreter::baseobjspace::getitem(concrete_obj, concrete_key)
            {
                ConcreteValue::from_pyobj(result)
            } else {
                ConcreteValue::Null
            }
        } else {
            ConcreteValue::Null
        };

        // jtransform do_fixed_list_getitem / do_resizable_list_getitem dispatch.
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_subscr_value(
                this,
                ctx,
                a,
                b,
                concrete_obj,
                concrete_key,
            ))
        })?;
        if let Some(opref) = gen_result {
            return Ok(FrontendOp::new(opref, subscr_concrete));
        }
        let opref = self.trace_binary_value(a, b, BinaryOperator::Subscr)?;
        Ok(FrontendOp::new(opref, subscr_concrete))
    }

    pub(crate) fn binary_int_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i + int_OP_ovf + guard_no_overflow
        // + new_with_vtable + setfield_gc).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_int_value(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_binary_value(a, b, op)
    }

    pub(crate) fn binary_float_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: BinaryOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_f/cast_int_to_float + float_OP
        // + new_with_vtable + setfield_gc).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_binary_float_value(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_binary_value(a, b, op)
    }

    pub(crate) fn compare_value_direct(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: ComparisonOperator,
        concrete_lhs: PyObjectRef,
        concrete_rhs: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i/f + int_LT/float_LT, with
        // goto_if_not fusion truth caching).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_compare_value_direct(
                this,
                ctx,
                a,
                b,
                op,
                concrete_lhs,
                concrete_rhs,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_compare_value(a, b, op)
    }

    pub(crate) fn store_subscr_value(
        &mut self,
        obj: OpRef,
        key: OpRef,
        value: OpRef,
        concrete_obj: PyObjectRef,
        concrete_key: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        // jtransform do_resizable_list_setitem dispatch.
        let handled: bool = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_store_subscr_value(
                this,
                ctx,
                obj,
                key,
                value,
                concrete_obj,
                concrete_key,
                concrete_value,
            ))
        })?;
        if handled {
            return Ok(());
        }
        self.trace_store_subscr(obj, key, value)
    }

    pub(crate) fn list_append_value(
        &mut self,
        list: OpRef,
        value: OpRef,
        concrete_list: PyObjectRef,
        concrete_value: PyObjectRef,
    ) -> Result<(), PyError> {
        if concrete_list.is_null() {
            return self.trace_list_append(list, value);
        }

        unsafe {
            if is_list(concrete_list) && w_list_can_append_without_realloc(concrete_list) {
                // listobject.rs:234: typed strategies (int/float) de-specialize
                // to object strategy if the appended value doesn't match.
                // Only enter typed fast path when concrete value type matches.
                let strategy_id = if w_list_uses_object_storage(concrete_list) {
                    Some(0i64)
                } else if w_list_uses_int_storage(concrete_list)
                    && !concrete_value.is_null()
                    && is_int(concrete_value)
                    && crate::state::int_strategy_preserves_identity(concrete_value)
                {
                    Some(1i64)
                } else if w_list_uses_float_storage(concrete_list)
                    && !concrete_value.is_null()
                    && is_float(concrete_value)
                {
                    Some(2i64)
                } else {
                    None
                };
                if let Some(sid) = strategy_id {
                    let concrete_len = w_list_len(concrete_list);
                    let is_inline = w_list_is_inline_storage(concrete_list);
                    return self.with_ctx(|this, ctx| {
                        crate::generated_list_append_by_strategy(
                            this,
                            ctx,
                            list,
                            value,
                            concrete_len,
                            sid,
                            is_inline,
                        );
                        Ok(())
                    });
                }
            }
        }

        self.trace_list_append(list, value)
    }

    pub(crate) fn concrete_iter_continues(
        &self,
        concrete_iter: PyObjectRef,
    ) -> Result<bool, PyError> {
        range_iter_continues(concrete_iter)
    }

    pub(crate) fn trace_known_builtin_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
        })
    }

    pub(crate) fn direct_len_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield(length) for str/dict/list/tuple).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_len_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_abs_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_abs_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_type_value(
        &mut self,
        callable: OpRef,
        value: OpRef,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_type_value(
                this,
                ctx,
                value,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[value])
    }

    fn direct_isinstance_value(
        &mut self,
        callable: OpRef,
        obj: OpRef,
        type_name: OpRef,
        concrete_obj: PyObjectRef,
        concrete_type_name: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_isinstance_value(
                this,
                ctx,
                obj,
                type_name,
                concrete_obj,
                concrete_type_name,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[obj, type_name])
    }

    fn direct_minmax_value(
        &mut self,
        callable: OpRef,
        a: OpRef,
        b: OpRef,
        choose_max: bool,
        concrete_a: PyObjectRef,
        concrete_b: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_direct_minmax_value(
                this, ctx, a, b, choose_max, concrete_a, concrete_b,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_known_builtin_call(callable, &[a, b])
    }

    pub(crate) fn call_callable_value(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        concrete_args: &[PyObjectRef],
    ) -> Result<OpRef, PyError> {
        if concrete_callable.is_null() {
            debug_assert!(
                false,
                "concrete_callable should always be available during tracing"
            );
            return self.trace_call_callable(callable, args);
        }

        unsafe {
            let is_builtin = is_function(concrete_callable)
                && is_builtin_code(
                    pyre_interpreter::getcode(concrete_callable) as pyre_object::PyObjectRef
                );
            if is_builtin {
                let builtin_name = pyre_interpreter::function_get_name(concrete_callable);
                if args.len() == 1 {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    if builtin_name == "type" {
                        return self.direct_type_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "len" {
                        return self.direct_len_value(callable, args[0], c_arg0);
                    }
                    if builtin_name == "abs" {
                        return self.direct_abs_value(callable, args[0], c_arg0);
                    }
                } else if args.len() == 2 && builtin_name == "isinstance" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_isinstance_value(callable, args[0], args[1], c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "min" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], false, c_arg0, c_arg1);
                } else if args.len() == 2 && builtin_name == "max" {
                    let c_arg0 = concrete_args.first().copied().unwrap_or(PY_NULL);
                    let c_arg1 = concrete_args.get(1).copied().unwrap_or(PY_NULL);
                    self.with_ctx(|this, ctx| {
                        this.implement_guard_value(ctx, callable, concrete_callable as i64)
                    });
                    return self
                        .direct_minmax_value(callable, args[0], args[1], true, c_arg0, c_arg1);
                }
                return self.with_ctx(|this, ctx| {
                    this.implement_guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    crate::helpers::emit_trace_call_known_builtin(ctx, callable, &boxed_args)
                });
            }
            if is_function(concrete_callable) {
                let w_callee_code = pyre_interpreter::getcode(concrete_callable);
                let callee_key = crate::driver::make_green_key(w_callee_code, 0);
                // pyjitpl.py:1396-1401 element-wise greenkey: pyre's
                // tuple is `(code_ptr, pc)` so structural equality cannot
                // be fooled by hash collisions the way the derived u64
                // `callee_key` can (driver.rs:12 make_green_key).
                let callee_raw: (usize, usize) = (w_callee_code as usize, 0);
                let caller_raw: (usize, usize) = ((*self.sym().jitcode).code as usize, 0);
                let callee_code =
                    &*(pyre_interpreter::w_code_get_ptr(w_callee_code as pyre_object::PyObjectRef)
                        as *const CodeObject);
                let callee_has_loop = code_has_backward_jump(callee_code);
                let (driver, _) = crate::driver::driver_pair();
                let nargs = args.len();

                // RPython pyjitpl.py: do_residual_or_indirect_call() follows
                // direct jitcode calls via perform_call() before falling back
                // to residual helpers.  Mirror that for ordinary direct calls:
                // if we know the callee body and it is a small acyclic helper,
                // trace through it directly instead of waiting for
                // should_inline() to bless a helper-boundary inline.
                let is_self_recursive = callee_raw == caller_raw;
                let inline_decision = driver.should_inline(callee_key, callee_raw);
                let inline_framestack_active = !self.parent_frames.is_empty();
                let callee_inline_eligible = driver
                    .meta_interp()
                    .warm_state_ref()
                    .can_inline_callable(callee_key);
                let max_unroll_recursion =
                    driver.meta_interp().warm_state_ref().max_unroll_recursion() as usize;
                let recursive_depth = self.with_ctx(|_, ctx| ctx.recursive_depth(callee_raw));
                let concrete_arg0 = if nargs == 1 {
                    concrete_args.first().copied()
                } else {
                    None
                };
                // pyjitpl.py:1388-1402 element-wise greenkey walk:
                //   count = 0
                //   for f in self.metainterp.framestack:
                //       if f.jitcode is not portal_code: continue
                //       gk = f.greenkey
                //       if gk is None: continue
                //       for i in range(len(gk)):
                //           if not gk[i].same_constant(greenboxes[i]): break
                //       else: count += 1
                //
                // Pyre's greenkey is `(code_ptr, target_pc)`; matching the
                // tuple implies `f.jitcode is portal_code`. With
                // `is_recursive=True` on the single PyPyJitDriver every
                // framestack entry is a portal frame, so the upstream
                // filter is automatically satisfied and the count equals
                // `ctx.recursive_depth(callee_key)`: only already-inlined
                // portal frames count, matching PyPy's root frame whose
                // `greenkey` is None.
                let recursive_count = recursive_depth;
                let recursion_exceeded =
                    callee_inline_eligible && recursive_count >= max_unroll_recursion;

                // pyjitpl.py:1413 warmrunnerstate.dont_trace_here(greenboxes).
                // Upstream calls this unconditionally when the recursion
                // limit is hit on an inlinable callee; pyre's warm-state
                // equivalent sets the DONT_TRACE_HERE jitcell flag via
                // `disable_noninlinable_function` (majit-trace
                // warmstate.rs:931). Fires for self-recursion *and*
                // mutual recursion because upstream's count walk does
                // not distinguish them.
                if recursion_exceeded {
                    driver
                        .meta_interp_mut()
                        .warm_state_mut()
                        .disable_noninlinable_function(callee_key);
                }

                // pyjitpl.py:1376-1423 _opimpl_recursive_call: compute
                // assembler_call boolean.  While recursive depth stays under
                // `max_unroll_recursion`, InlineDecision::Inline takes the
                // perform_call path below; CALL_ASSEMBLER is reserved for
                // the same fall-through cases as PyPy.
                let assembler_call = if is_self_recursive
                    && inline_decision == majit_metainterp::InlineDecision::Inline
                    && !recursion_exceeded
                {
                    driver
                        .get_loop_token_number(callee_key)
                        .or_else(|| driver.get_pending_token_number(callee_key))
                        .is_some()
                } else {
                    // pyjitpl.py:1417 `assembler_call = True` after the
                    // inlining path falls through (either dont_trace_here
                    // fired or can_inline_callable was False). The
                    // `CallAssembler` variant covers both subcases.
                    recursion_exceeded
                        || inline_decision == majit_metainterp::InlineDecision::CallAssembler
                };

                // pyjitpl.py:1414-1416 / pyjitpl.py:2174-2186 parity:
                // perform_call when callee is inlinable and the recursion
                // count is below `max_unroll_recursion`.  PyPy does not
                // special-case self-recursive calls here; the framestack
                // depth check is the gate.
                let can_trace_through =
                    callee_inline_eligible && nargs <= 4 && !callee_has_loop && !recursion_exceeded;

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][direct-call] key={} nargs={} inline_eligible={} self_recursive={} recursive_depth={} max_unroll={} assembler_call={} can_trace_through={}",
                        callee_key,
                        nargs,
                        callee_inline_eligible,
                        is_self_recursive,
                        recursive_depth,
                        max_unroll_recursion,
                        assembler_call,
                        can_trace_through,
                    );
                }

                if can_trace_through {
                    // pyjitpl.py:1414-1416 perform_call parity.
                    match self.build_pending_inline_frame(
                        callable,
                        args,
                        concrete_callable,
                        callee_key,
                        concrete_args,
                    ) {
                        Ok(pending) => {
                            self.pending_inline_frame = Some(pending);
                            return self
                                .with_ctx(|_, ctx| Ok(ctx.const_ref(pyre_object::PY_NULL as i64)));
                        }
                        Err(err) => {
                            if majit_metainterp::majit_log_enabled() {
                                eprintln!(
                                    "[jit][perform-call] build_pending failed key={} err={}, residual path",
                                    callee_key, err
                                );
                            }
                            // Fall through to residual helper path
                        }
                    }
                }

                // pyjitpl.py:1422 do_recursive_call / do_residual_call:
                // assembler_call is already computed above. The Inline
                // path passes it to inline_function_call which gates
                // CALL_ASSEMBLER vs CALL_MAY_FORCE on the boolean.
                // Non-Inline paths (pending_token, CallAssembler match)
                // handle their own CALL_ASSEMBLER emission.
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-check] is_self={} cache_safe={} inline_active={} callee_key={}",
                        is_self_recursive,
                        (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable),
                        inline_framestack_active,
                        callee_key
                    );
                }

                if inline_decision == majit_metainterp::InlineDecision::Inline {
                    if let Some(frame_helper) = (crate::callbacks::get().callee_frame_helper)(nargs)
                    {
                        return self.inline_function_call(
                            callable,
                            args,
                            concrete_callable,
                            callee_key,
                            callee_raw,
                            frame_helper,
                            concrete_args,
                            assembler_call,
                        );
                    }
                }

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][call-dispatch] callee_key={} pending_token={:?} loop_token={:?} is_self={}",
                        callee_key,
                        driver.get_pending_token_number(callee_key),
                        driver.get_loop_token_number(callee_key),
                        is_self_recursive
                    );
                }
                if let Some(token_number) = driver.get_pending_token_number(callee_key) {
                    if nargs == 1 || (crate::callbacks::get().callee_frame_helper)(nargs).is_some()
                    {
                        let call_pc = self.fallthrough_pc.saturating_sub(1);
                        return self.with_ctx(|this, ctx| {
                            if !is_self_recursive {
                                this.implement_guard_value(ctx, callable, concrete_callable as i64);
                            }
                            let self_recursive_raw_arg = if is_self_recursive
                                && nargs == 1
                                && matches!(concrete_arg0, Some(arg) if is_int(arg))
                            {
                                Some(this.trace_guarded_int_payload(ctx, args[0]))
                            } else {
                                None
                            };
                            let (callee_frame, drop_callee_frame) =
                                emit_call_assembler_callee_frame(
                                    this,
                                    ctx,
                                    callable,
                                    args,
                                    concrete_callable,
                                    w_callee_code,
                                    callee_code,
                                    is_self_recursive,
                                    self_recursive_raw_arg,
                                )?;
                            // pyjitpl.py:2017: do_residual_call step 1
                            this.vable_and_vrefs_before_residual_call(ctx);
                            let ca_result = ctx.call_assembler_red_only_ref(
                                token_number,
                                &[callee_frame, this.sym().execution_context],
                                &[Type::Ref, Type::Ref],
                            );
                            // pyjitpl.py:2080-2081 direct_assembler_call:
                            // record KEEPALIVE on callee virtualizable so
                            // it survives until the result is consumed.
                            ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                            if drop_callee_frame {
                                // Only the opaque arena-helper path needs the
                                // explicit drop. Trace-visible PyFrames are
                                // GC-owned and must not go through arena.put.
                                ctx.call_void(
                                    crate::callbacks::get().jit_drop_callee_frame,
                                    &[callee_frame],
                                );
                            }
                            // pyjitpl.py:2049
                            this.vrefs_after_residual_call(ctx);
                            // pyjitpl.py:2078
                            this.vable_after_residual_call()?;
                            // pyjitpl.py:2079
                            this.push_call_replay_stack(ctx, callable, args, call_pc);
                            this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                            ctx.heap_cache_mut().invalidate_caches_for_escaped();
                            this.pop_call_replay_stack(ctx, args.len())?;
                            let result = if inline_framestack_active {
                                ca_result // already Ref — no unbox+rebox needed
                            } else {
                                // Caller unboxes: guard_class + getfield_gc_i
                                this.trace_guarded_int_payload(ctx, ca_result)
                            };
                            Ok(result)
                        });
                    }
                }

                match inline_decision {
                    majit_metainterp::InlineDecision::CallAssembler => {
                        // Trace-through: inline callee body instead of CallAssembler.
                        // Guards use parent_frames to avoid OpRef::NONE in fail_args.
                        if callee_inline_eligible
                            && !is_self_recursive
                            && nargs <= 4
                            && !callee_has_loop
                        {
                            match self.build_pending_inline_frame(
                                callable,
                                args,
                                concrete_callable,
                                callee_key,
                                concrete_args,
                            ) {
                                Ok(pending) => {
                                    self.pending_inline_frame = Some(pending);
                                    return self.with_ctx(|_, ctx| {
                                        Ok(ctx.const_ref(pyre_object::PY_NULL as i64))
                                    });
                                }
                                Err(err) => {
                                    if majit_metainterp::majit_log_enabled() {
                                        eprintln!(
                                            "[jit][perform-call] call-assembler inline failed key={} err={}",
                                            callee_key, err
                                        );
                                    }
                                }
                            }
                        }
                        // pyjitpl.py:1417 `assembler_call = True` route:
                        // `should_inline_core` already accepts both compiled
                        // and pending tokens (`callee_compiled` predicate),
                        // but the emit path here requires the callee to be
                        // fully compiled — `compiled_loops[callee_key]` must
                        // resolve, and the descr we build threads through
                        // `make_call_assembler_descr_by_number`'s number
                        // factory which then keys back to compiled_loops.
                        // Including the pending-token slot here regresses
                        // against main: when the callee is mid-compilation
                        // (pending only), `get_compiled_meta` returns None
                        // and the downstream consumers fail.  RPython's
                        // `get_assembler_token` (warmstate.py:714) handles
                        // the pending case by synthesising a
                        // `compile_tmp_callback` token; that path is not
                        // yet ported (Task #195).  Until it is, mirror
                        // main and gate strictly on compiled presence.
                        let Some(token_number) = driver.get_loop_token_number(callee_key) else {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                this.implement_guard_value(ctx, callable, concrete_callable as i64);
                                let result = crate::helpers::emit_trace_call_known_function(
                                    ctx,
                                    this.frame(),
                                    callable,
                                    args,
                                )?;
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                                this.pop_call_replay_stack(ctx, args.len())?;
                                Ok(result)
                            });
                        };

                        {
                            let call_pc = self.fallthrough_pc.saturating_sub(1);
                            return self.with_ctx(|this, ctx| {
                                // Self-recursive: no callable guard needed (same function).
                                // Non-self-recursive: guard on callable value.
                                if !is_self_recursive {
                                    this.implement_guard_value(
                                        ctx,
                                        callable,
                                        concrete_callable as i64,
                                    );
                                }
                                let self_recursive_raw_arg = if is_self_recursive
                                    && args.len() == 1
                                    && matches!(concrete_arg0, Some(arg) if is_int(arg))
                                {
                                    Some(this.trace_guarded_int_payload(ctx, args[0]))
                                } else {
                                    None
                                };
                                let (callee_frame, drop_callee_frame) =
                                    emit_call_assembler_callee_frame(
                                        this,
                                        ctx,
                                        callable,
                                        args,
                                        concrete_callable,
                                        w_callee_code,
                                        callee_code,
                                        is_self_recursive,
                                        self_recursive_raw_arg,
                                    )?;

                                // pyjitpl.py:2017: do_residual_call step 1
                                this.vable_and_vrefs_before_residual_call(ctx);
                                let ec = this.ensure_execution_context(ctx);
                                let ca_result = ctx.call_assembler_red_only_ref(
                                    token_number,
                                    &[callee_frame, ec],
                                    &[Type::Ref, Type::Ref],
                                );
                                ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                                if drop_callee_frame {
                                    ctx.call_void(
                                        crate::callbacks::get().jit_drop_callee_frame,
                                        &[callee_frame],
                                    );
                                }
                                // pyjitpl.py:2049
                                this.vrefs_after_residual_call(ctx);
                                // pyjitpl.py:2078
                                this.vable_after_residual_call()?;
                                // pyjitpl.py:2079
                                this.push_call_replay_stack(ctx, callable, args, call_pc);
                                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                                this.pop_call_replay_stack(ctx, args.len())?;
                                let result = if inline_framestack_active {
                                    ca_result // already Ref
                                } else {
                                    // Caller unboxes: guard_class + getfield_gc_i
                                    this.trace_guarded_int_payload(ctx, ca_result)
                                };
                                Ok(result)
                            });
                        }
                    }
                    // Inline is handled at the top of the dispatch (line 2763).
                    // If we reach here, either frame_helper was unavailable
                    // or inline_decision changed; fall to residual below.
                    majit_metainterp::InlineDecision::Inline
                    | majit_metainterp::InlineDecision::ResidualCall => {}
                }

                let call_pc = self.fallthrough_pc.saturating_sub(1);
                return self.with_ctx(|this, ctx| {
                    this.implement_guard_value(ctx, callable, concrete_callable as i64);
                    let boxed_args = box_args_for_python_helper(this, ctx, args);
                    let result = crate::helpers::emit_trace_call_known_function(
                        ctx,
                        this.frame(),
                        callable,
                        &boxed_args,
                    )?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    Ok(result)
                });
            }
        }

        self.trace_call_callable(callable, args)
    }

    fn build_pending_inline_frame(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
        passed_concrete_args: &[PyObjectRef],
    ) -> Result<PendingInlineFrame, PyError> {
        use pyre_interpreter::pyframe::PyFrame;

        self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, callable, concrete_callable as i64);
        });

        let concrete_args: Vec<PyObjectRef> = passed_concrete_args.to_vec();
        for (_idx, arg) in concrete_args.iter().copied().enumerate() {
            if arg.is_null() {
                return Err(PyError::type_error(
                    "pending inline frame lost concrete arg",
                ));
            }
        }

        let caller_code = unsafe { (*self.sym().jitcode).code };
        let caller_exec_ctx = self.sym().concrete_execution_context;
        let caller_namespace_ptr = self.sym().concrete_namespace;
        let w_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
        let _raw_code = unsafe {
            pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                as *const CodeObject
        };
        let globals = unsafe { function_get_globals(concrete_callable) };
        let closure = unsafe { pyre_interpreter::function_get_closure(concrete_callable) };
        // pyjitpl.py:1396-1401 element-wise greenkey — `(code_ptr, 0)`
        // tuple equality is lossless vs the derived u64 hash.
        let is_self_recursive = caller_code as usize == w_code as usize;
        let mut callee_frame = PyFrame::new_for_call_with_closure(
            w_code,
            &concrete_args,
            globals,
            caller_exec_ctx,
            closure,
        );
        callee_frame.fix_array_ptrs();

        let callee_code = unsafe { &*pyre_interpreter::pyframe_get_pycode(&callee_frame) };
        // pyframe.py:111: nlocals + ncellvars + nfreevars = stack base
        let callee_nlocals =
            callee_code.varnames.len() + pyre_interpreter::pyframe::ncells(callee_code);
        let caller_namespace = caller_namespace_ptr;
        let callee_globals = unsafe { function_get_globals(concrete_callable) };
        let can_skip_traced_callee_frame = !is_self_recursive
            && callee_globals == caller_namespace
            && callee_nlocals == args.len();

        let (callee_sym, drop_frame_opref) = if can_skip_traced_callee_frame {
            let frame = self.frame();
            let mut sym = PyreSym::new_uninit(frame);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = callee_nlocals;
            sym.registers_r = args.to_vec();
            sym.symbolic_local_types = args.iter().map(|&arg| self.value_type(arg)).collect();
            sym.symbolic_stack_types = Vec::new();
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            sym.concrete_namespace = callee_globals as *mut DictStorage;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            let (
                vable_last_instr,
                vable_pycode,
                vable_valuestackdepth,
                vable_debugdata,
                vable_lastblock,
                vable_w_globals,
            ) = self.with_ctx(|_, ctx| {
                // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
                // (including pyjitpl.py:97-119 copy_constants).
                sym.setup_kind_register_banks(ctx);
                let null = ctx.const_ref(pyre_object::PY_NULL as i64);
                (
                    ctx.const_int(0),
                    ctx.const_ref(w_code as i64),
                    ctx.const_int(callee_nlocals as i64),
                    null, // debugdata = None
                    null, // lastblock = None
                    ctx.const_ref(callee_globals as i64),
                )
            });
            sym.vable_last_instr = vable_last_instr;
            sym.vable_pycode = vable_pycode;
            sym.vable_valuestackdepth = vable_valuestackdepth;
            sym.vable_debugdata = vable_debugdata;
            sym.vable_lastblock = vable_lastblock;
            sym.vable_w_globals = vable_w_globals;
            (sym, None)
        } else {
            // Create symbolic OpRef for callee frame in trace
            let callee_frame_opref = self.with_ctx(|this, ctx| {
                if args.len() == 1 {
                    let (helper, helper_arg_types) =
                        one_arg_callee_frame_helper(this.value_type(args[0]), is_self_recursive);
                    if is_self_recursive {
                        ctx.call_ref_typed(helper, &[this.frame(), args[0]], &helper_arg_types)
                    } else {
                        ctx.call_ref_typed(
                            helper,
                            &[this.frame(), callable, args[0]],
                            &helper_arg_types,
                        )
                    }
                } else if let Some(frame_helper) =
                    (crate::callbacks::get().callee_frame_helper)(args.len())
                {
                    let mut helper_args = vec![this.frame(), callable];
                    helper_args.extend_from_slice(args);
                    let helper_arg_types = frame_callable_arg_types(args.len());
                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types)
                } else {
                    panic!("no frame helper for {} args", args.len());
                }
            });

            let mut sym = PyreSym::new_uninit(callee_frame_opref);
            sym.nlocals = callee_nlocals;
            sym.valuestackdepth = sym.nlocals;
            sym.registers_r = Vec::with_capacity(sym.nlocals);
            sym.symbolic_local_types = Vec::with_capacity(sym.nlocals);
            for i in 0..sym.nlocals {
                if i < args.len() {
                    sym.registers_r.push(args[i]);
                    sym.symbolic_local_types.push(self.value_type(args[i]));
                } else {
                    sym.registers_r.push(OpRef::NONE);
                    sym.symbolic_local_types.push(Type::Ref);
                }
            }
            sym.symbolic_stack_types = Vec::new();
            // MIFrame Box tracking: set concrete metadata for callee
            sym.concrete_locals = concrete_args
                .iter()
                .map(|&a| ConcreteValue::from_pyobj(a))
                .collect();
            sym.concrete_locals
                .resize(callee_nlocals, ConcreteValue::Null);
            sym.concrete_stack = Vec::new();
            sym.jitcode = jitcode_for(w_code);
            // pyjitpl.py:74-90 MIFrame.setup parity for the per-kind banks
            // (including pyjitpl.py:97-119 copy_constants).
            //
            // pyjitpl.py:1230 `_opimpl_getarrayitem_vable`: every MIFrame
            // accesses the metainterp-scope `virtualizable_boxes` cache
            // (already provided by `TraceCtx::virtualizable_boxes` /
            // `ctx.virtualizable_box_at`) only when its own bytecode
            // emits a vable opcode — i.e. when this MIFrame is the
            // active vable owner. For inlined callee frames, vable_*
            // opcodes are not emitted; locals are accessed via
            // `metainterp.framestack[i].registers_r` (in pyre this is
            // `sym.registers_r`) or, for slots that have not yet been
            // populated symbolically (cell / nested-scope slots that
            // arrive pre-populated on the heap), via the callee's own
            // `locals_cells_stack_w` array.  Seed
            // `locals_cells_stack_array_ref` to that array OpRef so the
            // heap fall-through in `load_local_value` /
            // `push_typed_value` / `pop_value` reads from the callee's
            // locals_cells_stack_w (parity with init_symbolic's non-vable
            // branch in state.rs:2540-2543), not from the unset OpRef::NONE
            // that emitted invalid `getarrayitem(NONE, idx)` IR before.
            self.with_ctx(|_, ctx| {
                sym.setup_kind_register_banks(ctx);
                sym.locals_cells_stack_array_ref =
                    frame_locals_cells_stack_array(ctx, callee_frame_opref);
            });
            sym.concrete_namespace = callee_globals as *mut DictStorage;
            sym.concrete_execution_context = self.sym().concrete_execution_context;
            (sym, Some(callee_frame_opref))
        };

        // pyjitpl.py:2601-2602: parent frame keeps its original pc
        // (return_point_pc = CALL fallthrough). The callee blackhole
        // handles the call; the caller continues AFTER the call returns.
        // Stack is post-dispatch (args consumed, no result yet).
        // last_instr = return_point_pc - 1 so the caller's next_instr()
        // returns the fallthrough PC when the blackhole restores the frame.
        //
        // Task #21 Slice 3 propagation gap #2: mirror the parent's
        // sym.vable_last_instr + sym.vable_valuestackdepth into the
        // virtualizable_boxes shadow when this sym owns it. Without
        // propagation, the heap still carries the caller's pre-call PC
        // and pre-call vsd, and once the writeback gate flips on, the
        // patched-parent loop's GETFIELD preamble re-loads the stale
        // values. virtualizable_boxes[0] = vable_last_instr,
        // [2] = vable_valuestackdepth per virtualizable_gen.rs:37-44.
        let return_point_pc = self.fallthrough_pc;
        self.with_ctx(|this, ctx| {
            let last_instr_value = return_point_pc as i64 - 1;
            let ni = ctx.const_int(last_instr_value);
            let vsd_value = this.sym().valuestackdepth as i64;
            let vsd = ctx.const_int(vsd_value);
            let owns = {
                let s = this.sym_mut();
                s.vable_last_instr = ni;
                s.vable_valuestackdepth = vsd;
                s.owns_virtualizable_shadow()
            };
            if owns {
                mirror_vable_static_to_boxes(ctx, "last_instr", ni, Value::Int(last_instr_value));
                mirror_vable_static_to_boxes(ctx, "valuestackdepth", vsd, Value::Int(vsd_value));
            }
        });
        // opencoder.py:819-834 parity: accumulate full parent chain.
        // Current frame becomes the newest parent; existing parents follow.
        let mut parent_frames = vec![ResumeFrameState {
            sym: self.sym,
            concrete_frame_addr: self.concrete_frame_addr,
            resume_pc: return_point_pc,
            // `metainterp::push_inline_frame` overwrites this with the
            // caller's just-computed result stack idx (`pyjitpl.py:181-193`
            // parity).
            pending_result_stack_idx: None,
            pending_result_type: None,
        }];
        parent_frames.extend(self.parent_frames.iter().cloned());
        Ok(PendingInlineFrame {
            sym: callee_sym,
            concrete_frame: callee_frame,
            drop_frame_opref,
            green_key: callee_key,
            // Raw greenkey pair for element-wise recursion comparison
            // (pyjitpl.py:1396-1401). target_pc is 0 for function
            // entries — matches `make_green_key(w_code, 0)`. `w_code`
            // and `callee_key` are the same greenkey inputs that built
            // `callee_key` near the top of this function.
            green_key_raw: (w_code as usize, 0),
            parent_frames,
            nargs: args.len(),
            caller_result_stack_idx: None,
            caller_result_type: Some(Type::Ref),
        })
    }

    /// pyjitpl.py:1425-1432 do_recursive_call +
    /// pyjitpl.py:1995-2083 do_residual_call parity.
    /// When assembler_call=true, emits CALL_ASSEMBLER via token lookup.
    /// Otherwise emits CALL_MAY_FORCE (normal residual call).
    fn inline_function_call(
        &mut self,
        callable: OpRef,
        args: &[OpRef],
        concrete_callable: PyObjectRef,
        callee_key: u64,
        callee_raw: (usize, usize),
        frame_helper: *const (),
        passed_concrete_args: &[PyObjectRef],
        assembler_call: bool,
    ) -> Result<OpRef, PyError> {
        let (driver, _) = crate::driver::driver_pair();
        let concrete_arg0 = if args.len() == 1 {
            passed_concrete_args.first().copied()
        } else {
            None
        };
        // Save CALL instruction PC so GuardNotForced can resume the CALL.
        let call_pc = self.fallthrough_pc.saturating_sub(1);

        let result = self.with_ctx(|this, ctx| {
            this.implement_guard_value(ctx, callable, concrete_callable as i64);

            if args.len() == 1 {
                let result = if matches!(concrete_arg0, Some(arg) if unsafe { is_int(arg) }) {
                    let raw_arg = this.trace_guarded_int_payload(ctx, args[0]);
                    // pyjitpl.py:1396-1401 element-wise greenkey.
                    let _ = callee_key;
                    let caller_raw: (usize, usize) =
                        (unsafe { (*this.sym().jitcode).code } as usize, 0);
                    let is_self_recursive = callee_raw == caller_raw;
                    // RPython parity: an opaque helper-boundary Python CALL
                    // still produces a boxed object result.  Even if the
                    // callee itself can finish with a raw int, the helper
                    // boxes at the boundary and the trace records a Ref.
                    let force_fn = if is_self_recursive
                        && (crate::callbacks::get().recursive_force_cache_safe)(concrete_callable)
                    {
                        crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
                    } else {
                        crate::callbacks::get().jit_force_recursive_call_argraw_boxed_1
                    };
                    // pyjitpl.py:2053-2055: direct_assembler_call only when
                    // assembler_call=True (computed in _opimpl_recursive_call).
                    // Token lookup happens AFTER the decision, not before.
                    let ca_token = if assembler_call {
                        driver
                            .get_loop_token_number(callee_key)
                            .or_else(|| driver.get_pending_token_number(callee_key))
                    } else {
                        None
                    };
                    let raw_result = if let Some(token_number) = ca_token {
                        let w_callee_code = unsafe { pyre_interpreter::getcode(concrete_callable) };
                        let callee_code = unsafe {
                            &*(pyre_interpreter::w_code_get_ptr(w_callee_code as PyObjectRef)
                                as *const CodeObject)
                        };
                        let self_recursive_raw_arg = if is_self_recursive {
                            Some(raw_arg)
                        } else {
                            None
                        };
                        let (callee_frame, drop_callee_frame) = emit_call_assembler_callee_frame(
                            this,
                            ctx,
                            callable,
                            args,
                            concrete_callable,
                            w_callee_code,
                            callee_code,
                            is_self_recursive,
                            self_recursive_raw_arg,
                        )?;
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        let ec = this.ensure_execution_context(ctx);
                        let ca_result = ctx.call_assembler_red_only_ref(
                            token_number,
                            &[callee_frame, ec],
                            &[Type::Ref, Type::Ref],
                        );
                        ctx.record_op(OpCode::Keepalive, &[callee_frame]);
                        if drop_callee_frame {
                            ctx.call_void(
                                crate::callbacks::get().jit_drop_callee_frame,
                                &[callee_frame],
                            );
                        }
                        ca_result
                    } else if force_fn
                        == crate::callbacks::get().jit_force_self_recursive_call_argraw_boxed_1
                    {
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), raw_arg],
                            &[Type::Ref, Type::Int],
                        )
                    } else {
                        // pyjitpl.py:2017: do_residual_call step 1
                        this.vable_and_vrefs_before_residual_call(ctx);
                        ctx.call_may_force_ref_typed(
                            force_fn,
                            &[this.frame(), callable, raw_arg],
                            &[Type::Ref, Type::Ref, Type::Int],
                        )
                    };
                    // pyjitpl.py:2049: vrefs_after_residual_call
                    this.vrefs_after_residual_call(ctx);
                    // pyjitpl.py:2078: vable_after_residual_call
                    this.vable_after_residual_call()?;
                    // pyjitpl.py:2079: GUARD_NOT_FORCED — emitted on every
                    // residual-call path (CA and CALL_MAY_FORCE alike).
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    // CA-path-only: unbox boxed result via guard_class + getfield_gc_i.
                    let result = if ca_token.is_some() {
                        this.trace_guarded_int_payload(ctx, raw_result)
                    } else {
                        raw_result
                    };
                    result
                } else {
                    let force_fn = crate::callbacks::get().jit_force_recursive_call_1;
                    // pyjitpl.py:2017: do_residual_call step 1
                    this.vable_and_vrefs_before_residual_call(ctx);
                    let result = ctx.call_may_force_ref_typed(
                        force_fn,
                        &[this.frame(), callable, args[0]],
                        &[Type::Ref, Type::Ref, Type::Ref],
                    );
                    // pyjitpl.py:2049: vrefs_after_residual_call
                    this.vrefs_after_residual_call(ctx);
                    // pyjitpl.py:2078: vable_after_residual_call
                    this.vable_after_residual_call()?;
                    this.push_call_replay_stack(ctx, callable, args, call_pc);
                    this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                    ctx.heap_cache_mut().invalidate_caches_for_escaped();
                    this.pop_call_replay_stack(ctx, args.len())?;
                    result
                };
                Ok(result)
            } else {
                let mut helper_args = vec![this.frame(), callable];
                helper_args.extend_from_slice(args);
                let helper_arg_types = frame_callable_arg_types(args.len());
                let callee_frame =
                    ctx.call_ref_typed(frame_helper, &helper_args, &helper_arg_types);
                let force_fn = crate::callbacks::get().jit_force_callee_frame;
                // pyjitpl.py:2017: do_residual_call step 1
                this.vable_and_vrefs_before_residual_call(ctx);
                let result = ctx.call_may_force_ref_typed(force_fn, &[callee_frame], &[Type::Ref]);
                // pyjitpl.py:2049: vrefs_after_residual_call
                this.vrefs_after_residual_call(ctx);
                // pyjitpl.py:2078: vable_after_residual_call
                this.vable_after_residual_call()?;
                this.push_call_replay_stack(ctx, callable, args, call_pc);
                this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
                ctx.heap_cache_mut().invalidate_caches_for_escaped();
                this.pop_call_replay_stack(ctx, args.len())?;
                ctx.call_void(
                    crate::callbacks::get().jit_drop_callee_frame,
                    &[callee_frame],
                );
                Ok(result)
            }
        });

        result
    }

    pub(crate) fn iter_next_value(
        &mut self,
        iter: OpRef,
        concrete_iter: PyObjectRef,
    ) -> Result<FrontendOp, PyError> {
        let concrete_continues = range_iter_continues(concrete_iter)?;
        let concrete_step =
            unsafe { (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).step };
        let concrete_current = unsafe {
            (*(concrete_iter as *const pyre_object::rangeobject::W_RangeIterator)).current
        };

        // Delegate to auto-generated function (RPython jitcode parity:
        // getfield(current/stop/step) → step sign guard → continues guard
        // → int_add_ovf → guard_no_overflow → setfield).
        let gen_result: Option<(OpRef, i64)> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_iter_next_value(
                this,
                ctx,
                iter,
                concrete_continues,
                concrete_step,
                concrete_current,
            ))
        })?;
        if let Some((opref, cv)) = gen_result {
            return Ok(FrontendOp::new(opref, ConcreteValue::Int(cv)));
        }
        let opref = self.trace_iter_next_value(iter)?;
        Ok(FrontendOp::void(opref))
    }

    /// PRE-EXISTING-ADAPTATION: pyre's tracer holds raw `PyObjectRef`
    /// (the runtime W_Root pointer) instead of typed `FrontendOp`
    /// banks. RPython's `pyjitpl.py:opimpl_goto_if_*` reads
    /// `box.getint()` / `RefFrontendOp._resref` directly, dispatching
    /// on `box.type == 'i'` / `'r'`. This wrapper collapses that
    /// type-dispatched read into a single `objspace_truth_value` probe
    /// gated by a null guard, since pyre's tracer keeps the W_Root
    /// pointer rather than a typed-Box hierarchy.
    pub(crate) fn concrete_branch_truth_for_value(
        &mut self,
        _value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<bool, PyError> {
        if concrete_val.is_null() {
            return Err(PyError::type_error(
                "missing concrete branch value during trace",
            ));
        }
        Ok(objspace_truth_value(concrete_val))
    }

    pub(crate) fn concrete_branch_truth(&mut self) -> Result<bool, PyError> {
        self.concrete_branch_truth_for_value(OpRef::NONE, PY_NULL)
    }

    pub(crate) fn truth_value_direct(
        &mut self,
        value: OpRef,
        concrete_val: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // type-specialized is_true via guard_class + getfield → int_ne).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_truth_value_direct(
                this,
                ctx,
                value,
                concrete_val,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        self.trace_truth_value(value)
    }

    pub(crate) fn unary_int_value(
        &mut self,
        value: OpRef,
        opcode: OpCode,
        concrete_value: PyObjectRef,
    ) -> Result<OpRef, PyError> {
        // Delegate to auto-generated function (RPython jitcode parity:
        // guard_class + getfield_gc_i + INT_NEG/INT_INVERT).
        let gen_result: Option<OpRef> = self.with_ctx(|this, ctx| {
            Ok::<_, PyError>(crate::generated_unary_int_value(
                this,
                ctx,
                value,
                opcode,
                concrete_value,
            ))
        })?;
        if let Some(result) = gen_result {
            return Ok(result);
        }
        match opcode {
            OpCode::IntNeg => self.trace_unary_negative_value(value),
            OpCode::IntInvert => self.trace_unary_invert_value(value),
            _ => unreachable!("unexpected unary opcode"),
        }
    }

    pub(crate) fn into_trace_action(
        &mut self,
        result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
    ) -> TraceAction {
        trace_step_result_to_action(self, result)
    }

    pub fn trace_code_step(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return TraceAction::Abort;
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step decode_failed pc={}",
                    pc
                );
            }
            return TraceAction::Abort;
        };

        // Phase D-1 wire-up: resolve each opcode to its per-arm jitcode
        // through the shared `MetaInterpStaticData.jitcodes` table
        // (RPython `CallControl.jitcodes ≡ metainterp_sd.jitcodes`,
        // warmspot.py:281-282). Env-gated log-only probe — no execution —
        // so we can confirm the build→runtime bridge stays in sync as
        // Phase D-2 (shadow execute) and Phase D-3 (opcode migration)
        // land. `unresolved` logs the single-variant miss so a dispatch
        // gap surfaces immediately.
        if std::env::var_os("PYRE_JIT_JITCODE_PROBE").is_some() {
            match crate::jitcode_runtime::jitcode_for_instruction(&instruction) {
                Some(jc) => {
                    // Walk the jitcode byte stream using the insns argcode
                    // protocol (blackhole.py:105-232 `_get_method.handler`).
                    // `complete` flags whether the decoder reached
                    // `code.len()` without hitting an unknown opcode — a
                    // structural check the probe can surface per-opcode.
                    let mut opnames = Vec::new();
                    let mut last_next = 0usize;
                    for op in crate::jitcode_runtime::decoded_ops(&jc.code) {
                        opnames.push(op.opname);
                        last_next = op.next_pc;
                    }
                    let complete = last_next == jc.code.len();
                    eprintln!(
                        "[jit][probe] pc={pc} instr={} arm_jitcode={} code_len={} ops={} complete={} trace={:?}",
                        crate::jitcode_runtime::instruction_variant_name(&instruction),
                        jc.name,
                        jc.code.len(),
                        opnames.len(),
                        complete,
                        opnames,
                    );
                }
                None => eprintln!(
                    "[jit][probe] pc={pc} instr={} unresolved",
                    crate::jitcode_runtime::instruction_variant_name(&instruction),
                ),
            }
        }

        self.set_orgpc(pc);
        self.prepare_fallthrough();
        // RPython pyjitpl.py captures resumedata at each guard site, not at
        // every opcode boundary. Pyre still needs an opcode-start snapshot
        // for stack-machine opcodes that can mutate stack/register state
        // before emitting a guard; keep it off for instructions that cannot
        // reach a guard at all.
        let needs_pre_opcode_snapshot =
            crate::state::instruction_needs_pre_opcode_snapshot(instruction);
        if needs_pre_opcode_snapshot {
            self.capture_pre_opcode_state();
        }
        // pyjitpl.py:541-556 opimpl_goto_if_not_<op>_<type> parity: when
        // the CompareOp is followed by PopJumpIf* and operands form a
        // compatible int/float pair, run the fused dispatch path and
        // skip the separate PopJumpIf* dispatch entirely.
        let fused_result = if let Instruction::CompareOp { opname } = instruction {
            let comp_op = opname.get(op_arg);
            match self.try_fused_compare_goto_if_not(code, pc, comp_op) {
                Ok(Some(action)) => Some(Ok(action)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        };
        let step_result = match fused_result {
            Some(Ok(action)) => {
                // Fused path consumed COMPARE + BRANCH. Clear snapshot
                // and return the action immediately (matches the
                // post-dispatch clear that normally follows
                // execute_opcode_step below).
                if needs_pre_opcode_snapshot {
                    self.clear_pre_opcode_state();
                }
                return action;
            }
            Some(Err(e)) => Err(e),
            None => {
                let shadow_outcome =
                    crate::shadow_walker::shadow_validate_pre(self, &instruction, op_arg);
                let result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
                if result.is_ok() {
                    if let Some(outcome) = shadow_outcome {
                        crate::shadow_walker::shadow_validate_post(self, outcome);
                    }
                }
                result
            }
        };
        // Clear pre-opcode snapshot immediately after opcode execution.
        // RPython: capture_resumedata is called at each guard; there is
        // no per-opcode snapshot lifecycle. Clearing here ensures the
        // snapshot doesn't leak into subsequent opcodes.
        if needs_pre_opcode_snapshot {
            self.clear_pre_opcode_state();
        }
        // RPython execute_ll_raised parity: generic exc=True ops store the
        // exception in last_exc_value before handle_possible_exception.
        // Valid raise/reraise paths seed `last_exc_box` directly; malformed
        // bytecode-level raises still surface as ordinary interpreter errors
        // and use the generic raised-exception path below.
        if let Err(ref err) = step_result {
            if !matches!(
                instruction,
                Instruction::Reraise { .. } | Instruction::RaiseVarargs { .. }
            ) || self.sym().last_exc_box == OpRef::NONE
            {
                let exc_obj = err.to_exc_object();
                let s = self.sym_mut();
                if s.last_exc_value.is_null() {
                    s.last_exc_value = exc_obj;
                }
            }
        }
        if matches!(instruction, Instruction::Reraise { .. }) {
            if self.sym().last_exc_box != OpRef::NONE {
                return self.handle_reraise(code, pc);
            }
            if step_result.is_err() {
                return self.handle_possible_exception(code, pc);
            }
        }
        if let Instruction::RaiseVarargs { argc } = instruction {
            if self.sym().last_exc_box != OpRef::NONE {
                return self.handle_raise_varargs(code, pc, argc.get(op_arg) as usize);
            }
            if step_result.is_err() {
                return self.handle_possible_exception(code, pc);
            }
        }
        // RPython pyjitpl.py:1956-1957 execute_varargs: exc=True ops
        // always call handle_possible_exception, which internally decides
        // GUARD_EXCEPTION vs GUARD_NO_EXCEPTION.
        if instruction_may_raise(instruction) {
            let action = self.handle_possible_exception(code, pc);
            if !matches!(action, TraceAction::Continue) {
                return action;
            }
            // RPython: handle_possible_exception consumes the exception.
            // If it returned Continue, the exception was handled (e.g.,
            // GUARD_EXCEPTION emitted) and tracing continues normally.
            // Treat handled exceptions as successful opcode completion.
            if step_result.is_err() {
                return TraceAction::Continue;
            }
        }
        match step_result {
            Err(ref e) => {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_code_step err at pc={} instr={:?} err={:?}",
                        pc, instruction, e
                    );
                }
                TraceAction::Abort
            }
            other => self.into_trace_action(other),
        }
    }

    /// RPython pyjitpl.py:3380 handle_possible_exception.
    ///
    /// Called after every may-raise opcode. Checks last_exc_value to decide:
    /// - exception raised → GUARD_EXCEPTION + finishframe_exception
    /// - no exception → GUARD_NO_EXCEPTION
    pub(crate) fn handle_possible_exception(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> TraceAction {
        if !self.sym().last_exc_value.is_null() {
            let exc_obj = self.sym().last_exc_value;

            // pyjitpl.py:3382-3384: ALWAYS emit GUARD_EXCEPTION first,
            // regardless of class_of_last_exc_is_const.
            let exc_type_ptr = unsafe {
                (*(exc_obj as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type as i64
            };

            let guard_op = self.with_ctx(|this, ctx| {
                // pyjitpl.py:2575-2578: after_residual_call=true for
                // GuardException — all boxes in top frame are live.
                let after_residual_call = true;
                let resume_pc = this.fallthrough_pc;
                let saved_orgpc = this.orgpc;
                this.orgpc = resume_pc;
                this.clear_pre_opcode_state();

                this.flush_to_frame_for_guard(ctx);
                let active_boxes = this.get_list_of_active_boxes(ctx, false, after_residual_call);
                let fail_arg_types = this.build_fail_arg_types_for_active_boxes(&active_boxes);

                // capture_resumedata parity: full framestack snapshot.
                // pyjitpl.py:2597: capture_resumedata(self.framestack, ...)
                let snapshot =
                    this.build_framestack_snapshot(ctx, resume_pc, &active_boxes, &fail_arg_types);
                // Task #91 slice 4a: snapshot is the source of truth —
                // the optimizer's `store_final_boxes_in_guard`
                // (`optimizeopt/mod.rs:3200`) overwrites `op.fail_args`
                // from the snapshot via
                // `op.store_final_boxes(liveboxes)` (mod.rs:3392), so
                // the inline `record_guard_typed_with_fail_args` copy
                // was redundant.  Mirrors RPython
                // `pyjitpl.MetaInterp.generate_guard`
                // (pyjitpl.py:2558-2602) which records the guard with
                // no inline fail_args and lets `capture_resumedata` +
                // `_number_boxes` populate them from the snapshot
                // chain.
                let all_types = this.extend_types_with_parents(ctx, fail_arg_types);
                let snapshot_id = ctx.capture_resumedata(snapshot);

                let exc_type_const = ctx.const_int(exc_type_ptr);
                let op = ctx.record_guard_typed(
                    majit_ir::OpCode::GuardException,
                    &[exc_type_const],
                    all_types,
                );
                ctx.set_last_guard_resume_position(snapshot_id);

                this.orgpc = saved_orgpc;
                op
            });

            // pyjitpl.py:3385-3392:
            //   val = cast_opaque_ptr(GCREF, self.last_exc_value)
            //   if self.class_of_last_exc_is_const:
            //       self.last_exc_box = ConstPtr(val)
            //   else:
            //       self.last_exc_box = op
            //       op.setref_base(val)
            //   self.class_of_last_exc_is_const = True
            if self.sym().class_of_last_exc_is_const {
                let exc_box = self.with_ctx(|_this, ctx| ctx.const_ref(exc_obj as i64));
                self.sym_mut().last_exc_box = exc_box;
            } else {
                self.sym_mut().last_exc_box = guard_op;
            }
            self.sym_mut().class_of_last_exc_is_const = true;

            self.finishframe_exception(code, pc)
        } else {
            // pyjitpl.py:3397: GUARD_NO_EXCEPTION
            self.with_ctx(|this, ctx| {
                this.generate_guard(ctx, majit_ir::OpCode::GuardNoException, &[]);
            });
            TraceAction::Continue
        }
    }

    /// RPython pyjitpl.py:1701 opimpl_reraise parity.
    ///
    /// Unlike generic exc=True ops, RERAISE does not go through
    /// GUARD_EXCEPTION. It resumes directly from the already-known
    /// `last_exc_value` and unwinds to the enclosing handler.
    fn handle_reraise(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        let s = self.sym();
        if s.last_exc_value.is_null() || s.last_exc_box == OpRef::NONE {
            return TraceAction::Abort;
        }
        self.finishframe_exception(code, pc)
    }

    /// RPython pyjitpl.py:1688 opimpl_raise parity.
    ///
    /// Explicit raise shares the unwind path with reraise, but seeds
    /// `last_exc_box` from the just-popped exception box instead of
    /// going through GUARD_EXCEPTION.
    fn handle_raise_varargs(&mut self, code: &CodeObject, pc: usize, argc: usize) -> TraceAction {
        if argc == 0 {
            return self.handle_reraise(code, pc);
        }
        {
            let s = self.sym();
            if s.last_exc_value.is_null() || s.last_exc_box == OpRef::NONE {
                return TraceAction::Abort;
            }
        }
        self.finishframe_exception(code, pc)
    }

    /// PyPy `RAISE_VARARGS` materialization paired with RPython
    /// pyjitpl.py:1690-1696 `opimpl_raise` bookkeeping.
    fn seed_raised_exception(&mut self, exc_box: OpRef, concrete_exc: PyObjectRef) {
        if !concrete_exc.is_null() {
            let exc_class_ptr = unsafe {
                (*(concrete_exc as *const pyre_object::excobject::W_ExceptionObject))
                    .ob_header
                    .ob_type
            };
            self.with_ctx(|this, ctx| {
                if !ctx.heap_cache().is_class_known(exc_box) {
                    let cls_const = ctx.const_int(exc_class_ptr as usize as i64);
                    this.generate_guard(ctx, OpCode::GuardClass, &[exc_box, cls_const]);
                    ctx.heap_cache_mut()
                        .class_now_known(exc_box, majit_ir::GcRef(exc_class_ptr as usize));
                }
            });
        }
        let s = self.sym_mut();
        s.last_exc_value = concrete_exc;
        s.last_exc_box = exc_box;
        s.class_of_last_exc_is_const = true;
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (single-frame).
    ///
    /// Checks current frame for an exception handler.
    /// If found: unwind stack to handler depth, push exception, continue.
    /// If not found: return Abort (metainterp handles multi-frame unwind).
    fn finishframe_exception(&mut self, code: &CodeObject, pc: usize) -> TraceAction {
        let exc_opref = self.sym().last_exc_box;
        let exc_obj = self.sym().last_exc_value;
        let concrete_frame_addr = self.concrete_frame_addr;

        // pyjitpl.py:2510-2520: scan for catch_exception handler
        // (Python 3.11+ exception table replaces RPython's op_catch_exception)
        if let Some(entry) =
            pyre_interpreter::bytecode::find_exception_handler(&code.exceptiontable, pc as u32)
        {
            let handler_pc = entry.target as usize;
            let handler_depth = entry.depth as usize;
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][finishframe_exception] pc={} handler={} depth={}",
                    pc, handler_pc, handler_depth
                );
            }

            // pyjitpl.py:2506 finishframe_exception: unwind stack to handler,
            // pyjitpl.py:2517: frame.pc = target; raise ChangeFrame
            let ncells = code.cellvars.len() + code.freevars.len();
            let nlocals = self.sym().nlocals;
            let target_stack_len = ncells + handler_depth;
            let (has_vable, old_vsd, new_vsd, lasti_reg_idx, exc_reg_idx) = {
                let s = self.sym_mut();
                let old_vsd = s.valuestackdepth;
                s.symbolic_stack_types.truncate(target_stack_len);
                s.concrete_stack.truncate(target_stack_len);
                s.valuestackdepth = nlocals + target_stack_len;
                // Unified register file: the stack tail lives in
                // `registers_r[nlocals..]`. Truncate and re-push in
                // lockstep with `symbolic_stack_types` / `concrete_stack`.
                if s.registers_r.len() > nlocals + target_stack_len {
                    s.registers_r.truncate(nlocals + target_stack_len);
                }
                let lasti_reg_idx = if entry.push_lasti {
                    let idx = s.registers_r.len();
                    s.symbolic_stack_types.push(Type::Ref);
                    s.concrete_stack
                        .push(ConcreteValue::Ref(pyre_object::w_int_new(pc as i64)));
                    s.valuestackdepth += 1;
                    s.registers_r.push(OpRef::NONE);
                    Some(idx)
                } else {
                    None
                };
                let exc_reg_idx = s.registers_r.len();
                s.symbolic_stack_types.push(Type::Ref);
                s.concrete_stack.push(ConcreteValue::Ref(exc_obj));
                s.valuestackdepth += 1;
                s.registers_r.push(exc_opref);
                (
                    s.owns_virtualizable_shadow(),
                    old_vsd,
                    s.valuestackdepth,
                    lasti_reg_idx,
                    exc_reg_idx,
                )
            };
            // opencoder.py:718 `_list_of_boxes_virtualizable` parity: the
            // snapshot reads only `virtualizable_boxes`, so the unwind
            // must also mirror onto the shadow. Clear truncated slots to
            // PY_NULL (pyframe.py:411 popvalue_maybe_none) and write the
            // push_lasti / exc_box slots to match registers_r.
            if has_vable {
                let null_opref =
                    self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::PY_NULL as i64));
                let null_value =
                    majit_ir::Value::Ref(majit_ir::GcRef(pyre_object::PY_NULL as usize));
                let static_offset = crate::virtualizable_gen::NUM_VABLE_SCALARS;
                // Clear shadow slots that were above new_vsd up to old_vsd.
                let lo = new_vsd.min(old_vsd);
                let hi = new_vsd.max(old_vsd);
                self.with_ctx(|_this, ctx| {
                    for reg_idx in lo..hi {
                        let flat_idx = static_offset + reg_idx;
                        ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
                    }
                    if let Some(idx) = lasti_reg_idx {
                        // registers_r stored OpRef::NONE for the lasti
                        // slot; mirror that as PY_NULL shadow. The
                        // concrete int is not tracked here because the
                        // caller pushed a ConcreteValue::Ref directly
                        // without an OpRef.
                        let flat_idx = static_offset + idx;
                        ctx.set_virtualizable_entry_at(flat_idx, null_opref, null_value);
                    }
                    let flat_idx = static_offset + exc_reg_idx;
                    let exc_concrete = majit_ir::Value::Ref(majit_ir::GcRef(exc_obj as usize));
                    ctx.set_virtualizable_entry_at(flat_idx, exc_opref, exc_concrete);
                });
            }
            // Sync concrete frame.
            let frame = unsafe { &mut *(concrete_frame_addr as *mut pyre_interpreter::PyFrame) };
            let target_depth = frame.nlocals() + frame.ncells() + handler_depth;
            while frame.valuestackdepth > target_depth {
                frame.pop();
            }
            if entry.push_lasti {
                frame.push(pyre_object::w_int_new(pc as i64));
            }
            frame.push(exc_obj);
            // pyjitpl.py:2518: frame.pc = target; raise ChangeFrame
            self.sym_mut().pending_next_instr = Some(handler_pc);
            TraceAction::Continue
        } else {
            // No handler in this frame — return Abort so metainterp's
            // multi-frame finishframe_exception can pop this frame and
            // try the parent (pyjitpl.py:2520 self.popframe() loop).
            // Root frame with no handler → metainterp emits FINISH
            // (pyjitpl.py:2532 compile_exit_frame_with_exception).
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] no handler pc={}", pc);
            }
            TraceAction::Abort
        }
    }

    pub fn trace_code_step_inline(
        &mut self,
        code: &CodeObject,
        pc: usize,
    ) -> InlineTraceStepAction {
        if pc >= code.instructions.len() {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline pc_oob pc={} code_len={}",
                    pc,
                    code.instructions.len()
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        }

        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] trace_code_step_inline decode_failed pc={}",
                    pc
                );
            }
            return InlineTraceStepAction::Trace(TraceAction::Abort);
        };

        self.set_orgpc(pc);
        self.prepare_fallthrough();
        // Keep inline-frame guard capture aligned with the root-frame path:
        // only opcodes that can actually reach a guard carry an opcode-start
        // snapshot, and specific guard paths may still suppress it.
        let needs_pre_opcode_snapshot =
            crate::state::instruction_needs_pre_opcode_snapshot(instruction);
        if needs_pre_opcode_snapshot {
            self.capture_pre_opcode_state();
        }
        // pyjitpl.py:541-556 fused COMPARE_OP + PopJumpIf* parity (see
        // trace_code_step's counterpart for the full rationale).
        let fused_result = if let Instruction::CompareOp { opname } = instruction {
            let comp_op = opname.get(op_arg);
            match self.try_fused_compare_goto_if_not(code, pc, comp_op) {
                Ok(Some(action)) => Some(Ok(action)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        };
        let step_result = match fused_result {
            Some(Ok(action)) => {
                // Symmetry with trace_code_step: if pre_opcode_* were ever
                // captured for the inline frame (none do today, but future
                // work may add it), clear them on the fused-return path the
                // same way the non-inline path does at the post-dispatch
                // clear below.
                if needs_pre_opcode_snapshot {
                    self.clear_pre_opcode_state();
                }
                return InlineTraceStepAction::Trace(action);
            }
            Some(Err(e)) => Err(e),
            None => {
                let shadow_outcome =
                    crate::shadow_walker::shadow_validate_pre(self, &instruction, op_arg);
                let result = execute_opcode_step(self, code, instruction, op_arg, pc + 1);
                if result.is_ok() {
                    if let Some(outcome) = shadow_outcome {
                        crate::shadow_walker::shadow_validate_post(self, outcome);
                    }
                }
                result
            }
        };
        if needs_pre_opcode_snapshot {
            self.clear_pre_opcode_state();
        }
        // RPython execute_ll_raised parity: generic exc=True ops store the
        // exception in last_exc_value. Valid raise/reraise paths seed
        // `last_exc_box` directly; malformed bytecode-level raises still
        // use the generic raised-exception path below.
        if let Err(ref err) = step_result {
            if !matches!(
                instruction,
                Instruction::Reraise { .. } | Instruction::RaiseVarargs { .. }
            ) || self.sym().last_exc_box == OpRef::NONE
            {
                let exc_obj = err.to_exc_object();
                let s = self.sym_mut();
                if s.last_exc_value.is_null() {
                    s.last_exc_value = exc_obj;
                }
            }
        }
        if matches!(instruction, Instruction::Reraise { .. }) {
            if self.sym().last_exc_box != OpRef::NONE {
                return InlineTraceStepAction::Trace(self.handle_reraise(code, pc));
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(self.handle_possible_exception(code, pc));
            }
        }
        if let Instruction::RaiseVarargs { argc } = instruction {
            if self.sym().last_exc_box != OpRef::NONE {
                return InlineTraceStepAction::Trace(self.handle_raise_varargs(
                    code,
                    pc,
                    argc.get(op_arg) as usize,
                ));
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(self.handle_possible_exception(code, pc));
            }
        }
        if instruction_may_raise(instruction) {
            let exc_action = self.handle_possible_exception(code, pc);
            if !matches!(exc_action, TraceAction::Continue) {
                return InlineTraceStepAction::Trace(exc_action);
            }
            if step_result.is_err() {
                return InlineTraceStepAction::Trace(TraceAction::Continue);
            }
        }
        let action = match step_result {
            Ok(pyre_interpreter::StepResult::Return(value)) => {
                // pyjitpl.py:2489-2502 compile_done_with_this_frame parity:
                // result_type is always REF. ensure_boxed_for_ca re-boxes
                // unboxed Int/Float values (NewWithVtable + SetfieldGc).
                let v = value.opref;
                let finish_value =
                    self.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, v));
                TraceAction::Finish {
                    finish_args: vec![finish_value],
                    finish_arg_types: vec![Type::Ref],
                    exit_with_exception: false,
                }
            }
            Err(_) => TraceAction::Abort,
            other => self.into_trace_action(other),
        };
        match action {
            TraceAction::Continue => {
                if let Some(mut pending) = self.pending_inline_frame.take() {
                    let sym = self.sym();
                    let result_idx = sym
                        .valuestackdepth
                        .saturating_sub(sym.nlocals)
                        .checked_sub(1)
                        .expect("pending inline frame missing caller result slot");
                    pending.caller_result_stack_idx = Some(result_idx);
                    InlineTraceStepAction::PushFrame(pending)
                } else {
                    InlineTraceStepAction::Trace(TraceAction::Continue)
                }
            }
            other => InlineTraceStepAction::Trace(other),
        }
    }
}

/// Returns (is_int, is_float) for the fused-dispatch fuseability gate.
/// Mirrors the concrete-type classification codegen.rs uses to pick
/// between the int and float fast paths in generated_compare_value_direct,
/// but reads the ConcreteValue variant directly so the check does not
/// have to allocate an intermediate w_int/w_float box.
/// Trace-side mirror of `pyre_interpreter::eval::check_exc_match_against`
/// (`pyre/pyre-interpreter/src/eval.rs:81-130`).  Kept structurally
/// identical so the recorded boolean matches the interpreter's
/// concrete computation — including tuple-of-types, builtin-function
/// alias, str-named exception kinds, and `is_type` + MRO fallback.
///
/// SAFETY: `exc_value` and `exc_type` must be non-null PyObjectRefs
/// owned by the running interpreter for the duration of this call.
unsafe fn trace_check_exc_match_against(
    exc_value: pyre_object::PyObjectRef,
    exc_type: pyre_object::PyObjectRef,
) -> bool {
    if !pyre_object::is_exception(exc_value) {
        return true;
    }
    if pyre_object::is_tuple(exc_type) {
        let n = pyre_object::w_tuple_len(exc_type) as i64;
        for i in 0..n {
            if let Some(elem) = pyre_object::w_tuple_getitem(exc_type, i) {
                if trace_check_exc_match_against(exc_value, elem) {
                    return true;
                }
            }
        }
        return false;
    }
    let kind = pyre_object::w_exception_get_kind(exc_value);
    if pyre_object::is_str(exc_type) {
        let type_name = pyre_object::w_str_get_value(exc_type);
        return pyre_object::exc_kind_matches(kind, type_name);
    }
    if pyre_interpreter::is_function(exc_type)
        && pyre_interpreter::is_builtin_code(
            pyre_interpreter::getcode(exc_type) as pyre_object::PyObjectRef
        )
    {
        let type_name = pyre_interpreter::function_get_name(exc_type);
        return pyre_object::exc_kind_matches(kind, type_name);
    }
    if pyre_object::is_type(exc_type) {
        let type_name = pyre_object::w_type_get_name(exc_type);
        if pyre_object::exc_kind_matches(kind, type_name) {
            return true;
        }
        // ExcKind may not reflect the actual Python class hierarchy
        // (e.g. FileNotFoundError created with ExcKind::Exception).
        // Fall back to checking the exception's w_class MRO.
        let w_class = (*exc_value).w_class;
        if !w_class.is_null() && pyre_object::is_type(w_class) {
            if std::ptr::eq(w_class, exc_type) {
                return true;
            }
            let mro = pyre_object::w_type_get_mro(w_class);
            if !mro.is_null() {
                for &t in &*mro {
                    if std::ptr::eq(t, exc_type) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    // Unrecognised type-spec format — eval.rs:130+ falls through to the
    // legacy isinstance helper.  At trace time we don't have that path
    // ported yet; report a structural mismatch loudly so the caller
    // surfaces a tracer regression rather than silently emitting True.
    false
}

fn classify_concrete(cv: ConcreteValue) -> (bool, bool) {
    match cv {
        ConcreteValue::Int(_) => (true, false),
        ConcreteValue::Float(_) => (false, true),
        ConcreteValue::Ref(obj) if !obj.is_null() => unsafe { (is_int(obj), is_float(obj)) },
        _ => (false, false),
    }
}

pub(crate) fn trace_step_result_to_action(
    state: &mut MIFrame,
    result: Result<pyre_interpreter::StepResult<FrontendOp>, PyError>,
) -> TraceAction {
    match result {
        Ok(pyre_interpreter::StepResult::Continue) => {
            let compile_trace_succeeded = {
                let (driver, _) = crate::driver::driver_pair();
                driver.compile_trace_success_pending()
            };
            if compile_trace_succeeded {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!("[jit][compile-trace] pending success seen in trace_step_result");
                }
                return TraceAction::CompileTrace;
            }
            if state.ctx().is_too_long() {
                let green_key = state.ctx().green_key();
                let root_green_key = state.with_ctx(|_, ctx| ctx.root_green_key());
                if let Some(biggest_key) = biggest_inline_trace_key(state) {
                    let (driver, _) = crate::driver::driver_pair();
                    let warm_state = driver.meta_interp_mut().warm_state_mut();
                    warm_state.disable_noninlinable_function(biggest_key);
                    warm_state.trace_next_iteration(root_green_key);
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!(
                            "[jit][trace-too-long] biggest_inline_key={} trace_next_iteration root_key={}",
                            biggest_key, root_green_key
                        );
                    }
                    return majit_metainterp::TraceAction::Abort;
                }
                let force_finish_trace = {
                    let (driver, _) = crate::driver::driver_pair();
                    driver.meta_interp().force_finish_trace_enabled()
                };
                if force_finish_trace {
                    let jump_args = state.with_ctx(|this, ctx| this.close_loop_args(ctx));
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!(
                            "[jit] force_finish_trace: closing loop early at key={}",
                            root_green_key
                        );
                    }
                    return TraceAction::CloseLoopWithArgs {
                        jump_args,
                        loop_header_pc: None,
                    };
                }
                note_root_trace_too_long(root_green_key);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][abort-reason] trace_too_long key={} root_key={}",
                        green_key, root_green_key
                    );
                }
                TraceAction::Abort
            } else {
                TraceAction::Continue
            }
        }
        Ok(pyre_interpreter::StepResult::CloseLoop {
            jump_args,
            loop_header_pc,
        }) => TraceAction::CloseLoopWithArgs {
            jump_args: jump_args.iter().map(|fop| fop.opref).collect(),
            loop_header_pc: Some(loop_header_pc),
        },
        Ok(pyre_interpreter::StepResult::Return(fop)) => {
            // pyjitpl.py:2489-2502 finishframe + compile_done_with_this_frame:
            //   result_type = self.jitdriver_sd.result_type  (STATIC property)
            //   elif result_type == history.REF:
            //       raise jitexc.DoneWithThisFrameRef(resultbox.getref_base())
            //
            // pyre eval_loop_jit sets result_type = Type::Ref. Every FINISH
            // at the portal exit must carry Type::Ref. If the optimizer
            // unboxed the return value to Int/Float, ensure_boxed_for_ca
            // re-boxes it (NewWithVtable + SetfieldGc).
            let value = fop.opref;
            let finish_value =
                state.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, value));
            // pyjitpl.py:3222 store_token_in_vable
            state.with_ctx(|this, ctx| {
                this.store_token_in_vable(ctx);
            });
            TraceAction::Finish {
                finish_args: vec![finish_value],
                finish_arg_types: vec![Type::Ref],
                exit_with_exception: false,
            }
        }
        Ok(pyre_interpreter::StepResult::Yield(fop)) => {
            // pyjitpl.py:3198 compile_done_with_this_frame parity:
            // Yield uses the same Ref result_type.
            let value = fop.opref;
            let finish_value =
                state.with_ctx(|this, ctx| crate::state::ensure_boxed_for_ca(ctx, this, value));
            // pyjitpl.py:3222 store_token_in_vable (same as return path)
            state.with_ctx(|this, ctx| {
                this.store_token_in_vable(ctx);
            });
            TraceAction::Finish {
                finish_args: vec![finish_value],
                finish_arg_types: vec![Type::Ref],
                exit_with_exception: false,
            }
        }
        Err(err) => {
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][abort-reason] step_error key={} err={}",
                    state.ctx().green_key(),
                    err
                );
            }
            TraceAction::Abort
        }
    }
}

impl TraceHelperAccess for MIFrame {
    fn with_trace_ctx<R>(&mut self, f: impl FnOnce(&mut TraceCtx) -> R) -> R {
        self.with_ctx(|_, ctx| f(ctx))
    }

    fn trace_frame(&self) -> OpRef {
        self.frame()
    }

    fn trace_globals_ptr(&mut self) -> OpRef {
        self.with_ctx(|this, ctx| frame_get_namespace(ctx, this.frame()))
    }

    fn trace_record_not_forced_guard(&mut self) {
        self.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardNotForced, &[]);
            // heapcache.py: invalidate_caches after non-pure calls.
            // The call may have mutated heap state, so cached field
            // values for escaped objects are no longer reliable.
            ctx.heap_cache_mut().invalidate_caches_for_escaped();
        });
    }

    fn trace_call_callable(&mut self, callable: OpRef, args: &[OpRef]) -> Result<OpRef, PyError> {
        let frame = self.trace_frame();
        let result = self.with_ctx(|this, ctx| {
            let boxed_args = box_args_for_python_helper(this, ctx, args);
            crate::helpers::emit_trace_call_callable(ctx, frame, callable, &boxed_args)
        })?;
        self.trace_record_not_forced_guard();
        Ok(result)
    }

    fn trace_binary_value(
        &mut self,
        a: OpRef,
        b: OpRef,
        op: pyre_interpreter::bytecode::BinaryOperator,
    ) -> Result<OpRef, PyError> {
        self.with_ctx(|this, ctx| {
            let lhs = box_value_for_python_helper(this, ctx, a);
            let rhs = box_value_for_python_helper(this, ctx, b);
            crate::helpers::emit_trace_binary_value(ctx, lhs, rhs, op)
        })
    }
}

// `impl SharedOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl LocalOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl NamespaceOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl StackOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl IterOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl TruthOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ControlFlowOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl BranchOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ArithmeticOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

// `impl ConstantOpcodeHandler for MIFrame` is auto-generated by majit-translate
// (see majit/majit-translate/src/codegen.rs::generate_trait_impls).

impl OpcodeStepExecutor for MIFrame {
    type Error = PyError;

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let value = SharedOpcodeHandler::pop_value(self)?;
        if self.value_type(value.opref) != Type::Ref {
            return Err(PyError::type_error("pop_jump_if_none expects a ref value"));
        }
        let concrete_obj = value.concrete.to_pyobj();
        if concrete_obj.is_null() {
            return Err(PyError::type_error(
                "missing concrete value for pop_jump_if_none during trace",
            ));
        }
        let truth = self.with_ctx(|_this, ctx| {
            let expected = ctx.const_ref(pyre_object::w_none() as i64);
            ctx.record_op(OpCode::PtrEq, &[value.opref, expected])
        });
        let should_jump = unsafe { pyre_object::is_none(concrete_obj) };
        let fallthrough = self.fallthrough_pc();
        if !should_jump {
            self.with_ctx(|this, ctx| {
                MIFrame::set_next_instr(this, ctx, target);
                Ok::<(), PyError>(())
            })?;
        }
        let other_target = if should_jump { fallthrough } else { target };
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, value.opref, truth, should_jump, other_target);
            Ok::<(), PyError>(())
        })?;
        let next_target = if should_jump { target } else { fallthrough };
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, next_target);
            Ok::<(), PyError>(())
        })
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let value = SharedOpcodeHandler::pop_value(self)?;
        if self.value_type(value.opref) != Type::Ref {
            return Err(PyError::type_error(
                "pop_jump_if_not_none expects a ref value",
            ));
        }
        let concrete_obj = value.concrete.to_pyobj();
        if concrete_obj.is_null() {
            return Err(PyError::type_error(
                "missing concrete value for pop_jump_if_not_none during trace",
            ));
        }
        let truth = self.with_ctx(|_this, ctx| {
            let expected = ctx.const_ref(pyre_object::w_none() as i64);
            ctx.record_op(OpCode::PtrNe, &[value.opref, expected])
        });
        let should_jump = !unsafe { pyre_object::is_none(concrete_obj) };
        let fallthrough = self.fallthrough_pc();
        if !should_jump {
            self.with_ctx(|this, ctx| {
                MIFrame::set_next_instr(this, ctx, target);
                Ok::<(), PyError>(())
            })?;
        }
        let other_target = if should_jump { fallthrough } else { target };
        self.with_ctx(|this, ctx| {
            MIFrame::record_branch_guard(this, ctx, value.opref, truth, should_jump, other_target);
            Ok::<(), PyError>(())
        })?;
        let next_target = if should_jump { target } else { fallthrough };
        self.with_ctx(|this, ctx| {
            MIFrame::set_next_instr(this, ctx, next_target);
            Ok::<(), PyError>(())
        })
    }

    /// Fix fusion opcode: load two locals with correct concrete tracking.
    /// FrontendOp carries concrete directly — no pending_concrete_push needed.
    fn load_fast_pair_checked(
        &mut self,
        idx1: usize,
        _name1: &str,
        idx2: usize,
        _name2: &str,
    ) -> Result<(), Self::Error> {
        let c1 = self
            .sym()
            .concrete_locals
            .get(idx1)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let c2 = self
            .sym()
            .concrete_locals
            .get(idx2)
            .copied()
            .unwrap_or(ConcreteValue::Null);
        let v1 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx1))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v1, c1))?;
        let v2 = self.with_ctx(|this, ctx| MIFrame::load_local_value(this, ctx, idx2))?;
        SharedOpcodeHandler::push_value(self, FrontendOp::new(v2, c2))?;
        Ok(())
    }

    fn to_bool(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }

    /// CPython/PyPy LOOKUP_METHOD parity.
    ///
    /// Mirror eval.rs:1543-1608: push `(attr, null_or_self)` where
    /// `null_or_self` is the bound receiver for instance/class methods and
    /// `PY_NULL` for already-bound methods / static methods / plain attrs.
    fn load_method(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = SharedOpcodeHandler::pop_value(self)?;
        let concrete_obj = obj.concrete.to_pyobj();
        let attr = <Self as SharedOpcodeHandler>::load_attr(self, obj, name)?;
        <Self as SharedOpcodeHandler>::push_value(self, attr)?;

        let null_value = self.with_ctx(|_this, ctx| {
            FrontendOp::new(
                ctx.const_ref(pyre_object::PY_NULL as i64),
                ConcreteValue::Ref(pyre_object::PY_NULL),
            )
        });
        if concrete_obj.is_null() {
            return <Self as SharedOpcodeHandler>::push_value(self, null_value);
        }

        let concrete_attr = attr.concrete.to_pyobj();
        if !concrete_attr.is_null() && unsafe { pyre_object::is_method(concrete_attr) } {
            return <Self as SharedOpcodeHandler>::push_value(self, null_value);
        }

        let bound = unsafe {
            if pyre_object::is_instance(concrete_obj) {
                let w_type = pyre_object::w_instance_get_type(concrete_obj);
                let raw = pyre_interpreter::lookup_in_type(w_type, name);
                match raw {
                    Some(d) if pyre_object::is_staticmethod(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_classmethod(d) => w_type,
                    Some(d) if pyre_object::is_type(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_property(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_object::is_member(d) => pyre_object::PY_NULL,
                    Some(d) if pyre_interpreter::is_function(d) => {
                        let ob_type = (*d).ob_type;
                        if std::ptr::eq(
                            ob_type,
                            &pyre_interpreter::BUILTIN_FUNCTION_TYPE as *const PyType,
                        ) {
                            pyre_object::PY_NULL
                        } else if std::ptr::eq(
                            ob_type,
                            &pyre_interpreter::FUNCTION_TYPE as *const PyType,
                        ) && pyre_interpreter::is_builtin_code(
                            pyre_interpreter::function_get_code(d) as pyre_object::PyObjectRef,
                        ) {
                            concrete_obj
                        } else {
                            concrete_obj
                        }
                    }
                    Some(_) => concrete_obj,
                    None => pyre_object::PY_NULL,
                }
            } else if pyre_object::is_type(concrete_obj) {
                let raw = pyre_interpreter::lookup_in_type(concrete_obj, name);
                match raw {
                    Some(d) if pyre_object::is_classmethod(d) => concrete_obj,
                    Some(_) => pyre_object::PY_NULL,
                    None => concrete_obj,
                }
            } else if pyre_interpreter::typedef::r#type(concrete_obj).is_some()
                && !pyre_object::is_module(concrete_obj)
            {
                concrete_obj
            } else {
                pyre_object::PY_NULL
            }
        };

        let receiver = if bound.is_null() {
            null_value
        } else if bound == concrete_obj {
            obj
        } else {
            let bound_opref = self.with_ctx(|_this, ctx| ctx.const_ref(bound as i64));
            FrontendOp::new(bound_opref, ConcreteValue::Ref(bound))
        };
        <Self as SharedOpcodeHandler>::push_value(self, receiver)
    }

    /// CPython/PyPy CALL_FUNCTION parity for the `load_method()` stack shape.
    ///
    /// shared_opcode.rs intentionally discards `null_or_self`; the concrete
    /// interpreter override in eval.rs:1640-1652 prepends it for instance
    /// method calls. Trace-time execution must do the same so `CALL` after
    /// `LOAD_METHOD` sees the same argument list as the interpreter.
    fn call(&mut self, nargs: usize) -> Result<(), Self::Error> {
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(<Self as SharedOpcodeHandler>::pop_value(self)?);
        }
        args.reverse();
        let null_or_self = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let callable = <Self as SharedOpcodeHandler>::pop_value(self)?;

        let result = if null_or_self.concrete.to_pyobj().is_null() {
            <Self as SharedOpcodeHandler>::call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(args.len() + 1);
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            <Self as SharedOpcodeHandler>::call_callable(self, callable, &full_args)?
        };
        <Self as SharedOpcodeHandler>::push_value(self, result)
    }

    // RPython exception handler tracing (pyjitpl.py:2506 finishframe_exception):
    // handle_possible_exception emits GUARD_EXCEPTION and continues at the
    // handler PC. These three overrides trace the handler-entry bytecodes.
    //
    //   PUSH_EXC_INFO   → pop exc, save prev CURRENT_EXCEPTION, set
    //                     CURRENT_EXCEPTION = exc, push prev, push exc
    //                     (pyopcode.py:786 / eval.rs:1220-1229)
    //   CHECK_EXC_MATCH → opimpl_goto_if_exception_mismatch (pyjitpl.py:1677)
    //   POP_EXCEPT      → pop prev, restore CURRENT_EXCEPTION = prev,
    //                     clear tracer last_exc_value (pyopcode.py:778 /
    //                     eval.rs:1243-1249)

    fn push_exc_info(&mut self) -> Result<(), Self::Error> {
        let exc = <Self as SharedOpcodeHandler>::pop_value(self)?;
        let exc_obj = exc.concrete.to_pyobj();
        // Emit the residual save/restore pair BEFORE mutating concrete
        // CURRENT_EXCEPTION so the compiled bridge performs the same
        // `prev = CURRENT_EXCEPTION; CURRENT_EXCEPTION = exc` sequence at
        // runtime (pyopcode.py:786 / eval.rs:1220-1229). Folding a
        // trace-time `ctx.const_ref(prev_exc as i64)` into the bridge
        // silently breaks nested exception state on re-entry.
        let prev_exc_opref = self.with_ctx(|_this, ctx| {
            ctx.call_ref_typed(trace_get_current_exception_jit as *const (), &[], &[])
        });
        self.with_ctx(|_this, ctx| {
            ctx.call_void_typed(
                trace_set_current_exception_jit as *const (),
                &[exc.opref],
                &[Type::Ref],
            );
        });
        let prev_exc = get_current_exception();
        set_current_exception(exc_obj);
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(prev_exc_opref, ConcreteValue::Ref(prev_exc)),
        )?;
        <Self as SharedOpcodeHandler>::push_value(self, exc)?;
        {
            let s = self.sym_mut();
            s.current_exc_value = exc_obj;
            s.current_exc_box = exc.opref;
        }
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        let _ = frame.pop();
        frame.push(prev_exc);
        frame.push(exc_obj);
        Ok(())
    }

    fn pop_except(&mut self) -> Result<(), Self::Error> {
        let prev_exc = <Self as SharedOpcodeHandler>::pop_value(self)?;
        // Emit the residual restore so the compiled bridge writes back the
        // saved sys_exc_info at runtime (pyopcode.py:778 / eval.rs:1243-1249).
        self.with_ctx(|_this, ctx| {
            ctx.call_void_typed(
                trace_set_current_exception_jit as *const (),
                &[prev_exc.opref],
                &[Type::Ref],
            );
        });
        set_current_exception(prev_exc.concrete.to_pyobj());
        {
            let s = self.sym_mut();
            s.current_exc_value = prev_exc.concrete.to_pyobj();
            s.current_exc_box = prev_exc.opref;
        }
        let frame =
            unsafe { &mut *(self.concrete_frame_addr as *mut pyre_interpreter::pyframe::PyFrame) };
        let _ = frame.pop();
        // RPython pyjitpl.py:2751 clear_exception: exception fully handled.
        let s = self.sym_mut();
        s.last_exc_value = std::ptr::null_mut();
        s.class_of_last_exc_is_const = false;
        Ok(())
    }

    /// RPython pyjitpl.py:1677 opimpl_goto_if_exception_mismatch +
    /// CPython `CHECK_EXC_MATCH` semantics.  Pops the expected exception
    /// type, checks against last_exc_value, and pushes the concrete
    /// match result.  GUARD_EXCEPTION already verified the class so this
    /// usually produces True, but multi-except blocks may produce False
    /// for non-matching clauses.
    ///
    /// Logic mirrors `pyre_interpreter::eval::check_exc_match_against`
    /// (`pyre/pyre-interpreter/src/eval.rs:81-130`) — including tuple-of-
    /// types and `is_type` + MRO walk — so the recorded boolean matches
    /// what the interpreter would compute.  RPython
    /// `pyjitpl.py:1680-1681` asserts last_exc_value + class_of_last_exc
    /// are both set; pyre aborts tracing instead of silently emitting a
    /// constant True if either operand is null at trace time.
    fn check_exc_match(&mut self) -> Result<(), Self::Error> {
        let exc_type_val = <Self as SharedOpcodeHandler>::pop_value(self).ok();
        let exc_type_obj = exc_type_val
            .as_ref()
            .map(|v| v.concrete.to_pyobj())
            .unwrap_or(std::ptr::null_mut());

        let last_exc = self.sym().last_exc_value;
        if last_exc.is_null() || exc_type_obj.is_null() {
            // RPython `pyjitpl.py:1680-1681` `assert last_exc_value;
            // assert class_of_last_exc_is_const`.  A null operand here
            // means a tracer invariant was broken upstream (typically
            // CHECK_EXC_MATCH ran without prior PUSH_EXC_INFO /
            // GUARD_EXCEPTION).  Bail out loudly rather than fold a
            // wrong constant into the trace.
            return Err(PyError::runtime_error(
                "CHECK_EXC_MATCH with null last_exc or exc_type during tracing",
            ));
        }

        let matched = unsafe { trace_check_exc_match_against(last_exc, exc_type_obj) };

        let result_obj = pyre_object::w_bool_from(matched);
        let result_opref = self.with_ctx(|_this, ctx| ctx.const_ref(result_obj as i64));
        <Self as SharedOpcodeHandler>::push_value(
            self,
            FrontendOp::new(result_opref, ConcreteValue::Ref(result_obj)),
        )?;
        Ok(())
    }

    /// `pyopcode.py:704-720 RAISE_VARARGS` + `eval.rs:1049-1128` parity.
    ///
    /// pyre's tracer plays the role RPython splits between
    /// `pypy/interpreter/pyopcode.py` (operand handling + normalization)
    /// and `rpython/jit/metainterp/pyjitpl.py:1688 opimpl_raise`
    /// (box-level bookkeeping). Both responsibilities live here:
    ///
    /// 1. Normalize the operand — `raise Type` must call the exception
    ///    class to obtain an instance, matching
    ///    `eval.rs:1049-1088` / `eval.rs:1091-1127`.
    /// 2. For `argc == 2`, pop cause (TOS) before exception (TOS1) and
    ///    attach it via `attach_raise_cause` so `raise X from Y` produces
    ///    the same observable `__cause__` state as the interpreter.
    /// 3. Emit a residual helper that performs the same normalization +
    ///    `jit_exc_raise` sequence at runtime, then seed metainterp
    ///    state (pyjitpl.py:1690-1696) from the trace-time concrete
    ///    value so `handle_raise_varargs` → `finishframe_exception`
    ///    keeps seeing a pending exception on both paths.
    fn raise_varargs(&mut self, argc: usize) -> Result<(), Self::Error> {
        if argc == 0 {
            // `eval.rs:1032-1048 RAISE_VARARGS 0` — reraise from the
            // active exception. Prefer the tracer-seeded
            // `last_exc_value` (RPython pyjitpl.py:1701 opimpl_reraise),
            // fall back to the `CURRENT_EXCEPTION` TLS slot the
            // interpreter reads.
            let exc = self.sym().last_exc_value;
            if !exc.is_null() {
                return Err(unsafe { PyError::from_exc_object(exc) });
            }
            let exc = get_current_exception();
            if unsafe { pyre_object::is_exception(exc) } {
                return Err(unsafe { PyError::from_exc_object(exc) });
            }
            return Err(PyError::runtime_error("No active exception to reraise"));
        }
        if argc > 2 {
            return Err(PyError::type_error("too many arguments for raise"));
        }
        // `eval.rs:1091-1094 RAISE_VARARGS 2` pops cause (TOS) first,
        // then exception (TOS1). `attach_raise_cause` runs only for
        // argc == 2.
        let (exc_val, cause, cause_opref) = if argc == 2 {
            let raw_cause = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let exc_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let raw_cause_obj = raw_cause.concrete.to_pyobj();
            let cause = if raw_cause_obj.is_null() {
                None
            } else {
                // pyopcode.py:706 `w_cause = space.call_function(w_cause)`
                // runs on the interpreter path; in pyre's tracer context
                // the call must be forced onto the plain eval loop so it
                // does not re-enter the tracer. Mirrors the exc-path
                // guard at `call_function(exc, &[])` below (commit
                // bef2ee2035 symmetrized JIT + blackhole helpers but
                // missed this tracer-time site).
                let _plain_guard = pyre_interpreter::call::force_plain_eval();
                Some(normalize_raise_cause(raw_cause_obj)?)
            };
            (exc_val, cause, raw_cause.opref)
        } else {
            let exc_val = <Self as SharedOpcodeHandler>::pop_value(self)?;
            let null_cause = self.with_ctx(|_this, ctx| ctx.const_ref(pyre_object::PY_NULL as i64));
            (exc_val, None, null_cause)
        };
        let exc = exc_val.concrete.to_pyobj();
        let emit_runtime_raise = |slf: &mut Self| {
            slf.with_ctx(|this, ctx| {
                ctx.call_ref_typed(
                    normalize_raise_varargs_jit as *const (),
                    &[this.frame(), exc_val.opref, cause_opref],
                    &[Type::Ref, Type::Ref, Type::Ref],
                )
            })
        };
        if exc.is_null() {
            let _ = emit_runtime_raise(self);
            return Err(PyError::value_error("raised during tracing"));
        }
        unsafe {
            if pyre_object::is_exception(exc) {
                // `eval.rs:1053-1055 / :1096-1098`: already an instance.
                let _ = emit_runtime_raise(self);
                attach_raise_cause(exc, cause)?;
                self.seed_raised_exception(exc_val.opref, exc);
                return Err(PyError::from_exc_object(exc));
            }
            // `eval.rs:1056-1083 / :1099-1121`: `raise SomeType`
            // normalizes by invoking the exception class with no args.
            // Run the call at trace time to get a concrete instance for
            // `GUARD_CLASS` + state seeding, and emit a residual
            // `call_ref` so the compiled trace re-invokes the constructor
            // each iteration instead of folding the trace-time heap
            // address into a `const_ref`.
            if pyre_interpreter::baseobjspace::exception_is_valid_obj_as_class_w(exc) {
                // Mirror `normalize_raise_value` (pyopcode.py:707/:713): the
                // trace-time constructor call must stay on the plain
                // interpreter path so it does not re-enter the tracer.
                let result = {
                    let _plain_guard = pyre_interpreter::call::force_plain_eval();
                    call_function(exc, &[])
                };
                if !pyre_object::is_exception(result) {
                    return Err(PyError::type_error(
                        "exceptions must derive from BaseException",
                    ));
                }
                let exc_box = emit_runtime_raise(self);
                attach_raise_cause(result, cause)?;
                self.seed_raised_exception(exc_box, result);
                return Err(PyError::from_exc_object(result));
            }
        }
        // Non-null, not an instance / exception class — matches
        // `eval.rs:1084-1087` / `eval.rs:1122-1126` fall-through.
        Err(PyError::type_error(
            "exceptions must derive from BaseException",
        ))
    }

    /// RPython opimpl_reraise (pyjitpl.py:1701).
    fn reraise(&mut self) -> Result<(), Self::Error> {
        let exc = self.sym().last_exc_value;
        unsafe {
            if pyre_object::is_exception(exc) {
                Err(PyError::from_exc_object(exc))
            } else {
                Err(PyError::runtime_error("reraise during tracing"))
            }
        }
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<pyre_interpreter::StepResult<FrontendOp>, Self::Error> {
        Err(PyError::type_error(format!(
            "unsupported instruction during trace: {instruction:?}"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyre_interpreter::eval::eval_frame_plain;
    use std::rc::Rc;

    #[cfg(feature = "cranelift")]
    fn clear_pending_jit_exception() {
        majit_backend_cranelift::jit_exc_raise(0);
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn clear_pending_jit_exception() {
        majit_backend_dynasm::jit_exc_raise(0);
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn clear_pending_jit_exception() {}

    #[cfg(feature = "cranelift")]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_cranelift::jit_exc_value_raw()
    }

    #[cfg(all(not(feature = "cranelift"), feature = "dynasm"))]
    fn pending_jit_exception_raw() -> i64 {
        majit_backend_dynasm::jit_exc_value_raw()
    }

    #[cfg(not(any(feature = "cranelift", feature = "dynasm")))]
    fn pending_jit_exception_raw() -> i64 {
        0
    }

    #[test]
    fn normalize_raise_varargs_jit_null_frame_still_publishes_pending_exception() {
        clear_pending_jit_exception();
        let code = pyre_interpreter::compile_exec("x = ValueError\n").expect("compile failed");
        let mut frame = pyre_interpreter::PyFrame::new_with_context(
            code,
            Rc::new(pyre_interpreter::PyExecutionContext::default()),
        );
        eval_frame_plain(&mut frame).expect("module body should execute");
        let exc_class = unsafe {
            (*frame.fget_w_globals())
                .get("x")
                .copied()
                .expect("namespace should contain ValueError")
        };

        let result = normalize_raise_varargs_jit(0, exc_class as i64, pyre_object::PY_NULL as i64);

        assert_eq!(result, pending_jit_exception_raw());
        let err = unsafe { pyre_interpreter::PyError::from_exc_object(result as PyObjectRef) };
        assert_eq!(err.kind, pyre_interpreter::PyErrorKind::RuntimeError);
        assert_eq!(err.message, "raise helper missing current frame");
        clear_pending_jit_exception();
    }

    #[test]
    fn get_list_of_active_boxes_reads_kind_specific_register_banks() {
        use majit_translate::liveness::encode_liveness;
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut all_liveness = vec![1, 1, 1];
        all_liveness.extend(encode_liveness(&[2]));
        all_liveness.extend(encode_liveness(&[1]));
        all_liveness.extend(encode_liveness(&[3]));
        let mut insns = HashMap::new();
        insns.insert("live/".to_string(), majit_metainterp::jitcode::BC_LIVE);
        crate::assembler::publish_state(&insns, &all_liveness, all_liveness.len(), 1);

        let runtime_jc = {
            let inner = majit_metainterp::jitcode::JitCode::new("get_list_of_active_boxes_test");
            inner.set_body(majit_translate::jitcode::JitCodeBody {
                code: vec![majit_metainterp::jitcode::BC_LIVE, 0, 0],
                c_num_regs_i: 4,
                c_num_regs_r: 4,
                c_num_regs_f: 4,
                // RPython `jitcode.py:85-90` `assert pc in self._startpoints`:
                // hand-crafted bodies must declare each opcode's offset.  The
                // single BC_LIVE here sits at byte 0.
                startpoints: Some([0_usize].into_iter().collect()),
                ..Default::default()
            });
            inner
        };
        let mut pyjit = crate::PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        pyjit.jitcode = Arc::new(runtime_jc);
        pyjit.metadata.pc_map.push(0);
        let inner_jc = crate::state::JitCode {
            code: std::ptr::null(),
            index: 0,
            payload: Arc::new(pyjit),
        };
        let inner_jc_ptr = Box::into_raw(Box::new(inner_jc));

        let int_box = OpRef::int_op(10);
        let ref_box = OpRef::ref_op(20);
        let float_box = OpRef::float_op(30);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.jitcode = inner_jc_ptr;
        // `registers_r` is the unified abstract register file
        // (locals[..nlocals] then stack[nlocals..nlocals+stack_only]); the
        // snapshot read translates the Ref liveness color through
        // `semantic_ref_slot_for_reg_color` to land at the same slot the
        // lazy-load preamble populates. Set nlocals=2 so the encoded ref
        // color 1 maps to semantic slot `locals[1]` via the in-range
        // fallback. Int and Float banks stay kind-specific (no
        // unification), so their bank-indexed setup is unchanged.
        sym.nlocals = 2;
        sym.valuestackdepth = 2;
        sym.registers_i = vec![OpRef::NONE, OpRef::NONE, int_box];
        sym.registers_r = vec![OpRef::NONE, ref_box];
        sym.registers_f = vec![OpRef::NONE, OpRef::NONE, OpRef::NONE, float_box];

        let mut ctx = TraceCtx::for_test(1);
        let mut frame = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_result_stack_idx: None,
            pending_result_type: None,
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
            pre_opcode_registers_r: None,
        };

        let active = frame.get_list_of_active_boxes(&mut ctx, false, false);
        assert_eq!(active, vec![int_box, ref_box, float_box]);

        unsafe {
            let _ = Box::from_raw(inner_jc_ptr as *mut crate::state::JitCode);
        }
    }
}
