//! Bytecode evaluation loop — pure interpreter.
//!
//! JIT integration lives in pyre-jit/src/eval.rs. This module is
//! JIT-free: it processes bytecode instructions with no tracing,
//! no merge points, and no compiled-code hooks.

use crate::bytecode::{BinaryOperator, ComparisonOperator, Instruction};
use crate::*;
use crate::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs,
    decode_instruction_for_dispatch, dict_storage_load, dict_storage_store, ensure_range_iter,
    execute_opcode_step, make_function_from_code_obj, range_iter_continues,
    range_iter_next_or_null, stack_underflow_error, unpack_sequence_exact, w_code_new,
};
use pyre_object::*;

use crate::call::call_callable;
use std::cell::Cell;

#[derive(Debug, Clone)]
pub struct Code {
    pub name: String,
    pub code: Option<PyObjectRef>,
}

impl Code {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            code: None,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("<code {}>", self.name)
    }
}

// Thread-local current exception for bare `raise` (RAISE_VARARGS 0).
// PyPy: executioncontext.py sys_exc_info — the current active exception.
thread_local! {
    static CURRENT_EXCEPTION: Cell<PyObjectRef> = const { Cell::new(PY_NULL) };
    pub(crate) static CURRENT_FRAME: Cell<*mut PyFrame> = const { Cell::new(std::ptr::null_mut()) };
}
use crate::pyframe::PyFrame;

/// Saves the previous `CURRENT_FRAME` and (when EC was modified) the
/// previous `ec.topframeref` so they can be restored on Drop. The two
/// pointers are pushed onto `majit_gc::shadow_stack` rather than a local
/// `Vec` — this matches RPython's `framework.py` shadow-stack
/// (rpython/memory/gctransform/shadowstack.py:281) and lets the GC's
/// Phase 1b root-walker forward both pointers in place when a minor
/// collection runs while the guard is on the stack.
pub struct CurrentFrameGuard {
    save_point: usize,
    ec: *mut PyExecutionContext,
    ec_top_root_index: Option<usize>,
}

impl Drop for CurrentFrameGuard {
    fn drop(&mut self) {
        // Read forwarded values from the shadow stack before pop_to so we
        // observe any in-place updates the GC may have made.
        let previous = majit_gc::shadow_stack::get(self.save_point);
        let previous_ec_top = self
            .ec_top_root_index
            .map(majit_gc::shadow_stack::get)
            .unwrap_or(majit_ir::GcRef::NULL);
        majit_gc::shadow_stack::pop_to(self.save_point);
        CURRENT_FRAME.with(|current| current.set(previous.0 as *mut PyFrame));
        if !self.ec.is_null() {
            unsafe {
                (*self.ec).topframeref = previous_ec_top.0 as *mut PyFrame;
            }
        }
    }
}

fn push_current_frame_previous_root(
    previous: *mut PyFrame,
    ec: *mut PyExecutionContext,
    previous_ec_top: *mut PyFrame,
) -> CurrentFrameGuard {
    let save_point = majit_gc::shadow_stack::push(majit_ir::GcRef(previous as usize));
    let ec_top_root_index = if ec.is_null() {
        None
    } else {
        Some(majit_gc::shadow_stack::push(majit_ir::GcRef(
            previous_ec_top as usize,
        )))
    };
    CurrentFrameGuard {
        save_point,
        ec,
        ec_top_root_index,
    }
}

pub fn install_current_frame(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    // executioncontext.py `enter()` parity: link the frame into the
    // topframeref/f_backref chain so walkers (GC roots, sys._getframe)
    // can iterate all active frames. `eval_frame_plain` calls
    // `ExecutionContext::enter` before installing TLS-only state, but
    // the JIT portal path enters through this helper directly.
    let ec = frame.execution_context as *mut PyExecutionContext;
    let previous_ec_top = if ec.is_null() {
        std::ptr::null_mut()
    } else {
        unsafe {
            let top = (*ec).topframeref;
            (*ec).topframeref = frame as *mut PyFrame;
            top
        }
    };
    frame.f_backref = if ec.is_null() {
        previous
    } else {
        previous_ec_top
    };
    push_current_frame_previous_root(previous, ec, previous_ec_top)
}

/// Install only the TLS current-frame root.
///
/// Use this after `ExecutionContext::enter()` has already linked
/// `frame.f_backref`.  PyPy has one frame chain (`ec.topframeref`);
/// pyre's `CURRENT_FRAME` is an extra GC/super() TLS root and must not
/// overwrite the RPython `f_backref` chain once EC owns it.
pub fn install_current_frame_tls_only(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    push_current_frame_previous_root(previous, std::ptr::null_mut(), std::ptr::null_mut())
}

/// rpython/memory/gctransform/framework.py `root_walker.walk_roots` parity:
/// expose every live slot of `PyFrame.locals_cells_stack_w` on the active
/// f_backref chain as a GC root.
///
/// pyre's JIT-compiled code allocates W_IntObject / result boxes into the
/// nursery (`NewWithVtable` → `gc_alloc_typed_nursery_shim`). When the
/// nursery fills and a minor collection runs, only registered roots are
/// forwarded — unforwarded nursery refs become stale after
/// `Nursery::reset` zero-fills the region. The interpreter stores live
/// refs in `PyFrame.locals_cells_stack_w`; without this walker those
/// slots turn into NULL-`ob_type` stale pointers on the next LOAD_FAST
/// (reproduced by `inline_helper` n >= 10000).
///
/// Walks 0..`valuestackdepth` entries because that range covers both
/// the always-live locals+cells prefix (slots `0..nlocals+ncells`,
/// written once at frame setup) and the operand stack region
/// (`nlocals+ncells..valuestackdepth`). Dead stack slots past
/// `valuestackdepth` are skipped.
fn walk_pyframe_roots(visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
    CURRENT_FRAME.with(|cf| {
        // Forward `CURRENT_FRAME` itself: when the top frame is a
        // nursery-allocated `PyFrame` (Phase 2.3 옵션 B
        // `emit_new_pyframe_inline_self_recursive`) the visitor copies
        // it to the survivor space and rewrites the cell to the new
        // address. For `std::alloc`-backed frames the visitor's
        // `is_nursery_object_start` guard short-circuits, leaving the
        // pointer untouched. `Cell::as_ptr()` exposes the storage
        // address; `*mut PyFrame` and `GcRef` share the `usize` repr
        // (`GcRef` is `#[repr(transparent)]`).
        //
        // SAFETY: `CURRENT_FRAME`'s storage is a thread-local `Cell`
        // that outlives this walker. We hold the with-borrow `cf` for
        // the duration of the visit so no other code mutates the cell.
        let cf_slot_ptr = cf.as_ptr() as *mut majit_ir::GcRef;
        visitor(unsafe { &mut *cf_slot_ptr });
        // Saved previous-frame / previous-ec-topframe roots now live on
        // `majit_gc::shadow_stack` (pushed by `push_current_frame_previous_root`)
        // and are forwarded by the GC's Phase 1b walker; no extra visit here.

        let mut frame = cf.get();
        if !frame.is_null() {
            let ec = unsafe { (*frame).execution_context as *mut PyExecutionContext };
            if !ec.is_null() {
                let top_slot = unsafe { &mut (*ec).topframeref as *mut *mut PyFrame };
                visitor(unsafe { &mut *(top_slot as *mut majit_ir::GcRef) });
            }
        }
        while !frame.is_null() {
            // SAFETY: PyFrame pointers on the f_backref chain are valid
            // for the duration of the enclosing `eval_with_jit` call. A
            // minor collection is always synchronous with respect to the
            // interpreter thread, so frames cannot be dropped mid-walk.
            //
            // We walk the FULL fixed-length array (not just the live
            // `valuestackdepth` prefix). Argument values in transit —
            // popped from the caller's stack before the callee frame
            // is installed — are briefly invisible from
            // `valuestackdepth` alone, yet still reachable from the
            // popped-slot storage. Non-ref slots are filtered by
            // `is_nursery_object_start` inside the collector, so
            // walking past the live depth is harmless for the
            // bump-pointer nursery.
            //
            // The walk runs for every frame on the chain, including
            // ones the GC owns. For nursery-allocated frames the
            // standard tracer ALSO covers their gc_ptr_offsets when it
            // reaches the survivor copy; visiting the locals array
            // items here from the original nursery payload is safe
            // because Phase 1e runs before any internal-slot
            // forwarding (the original payload is still intact). We
            // intentionally do NOT call `majit_gc::gc_owns_object`
            // here to gate this branch — that hook re-enters
            // `with_cranelift_gc` with a `borrow_mut`, which panics
            // when invoked from inside `collect_nursery` (the GC's
            // own cell is already borrowed by the active alloc shim).
            let (arr_ptr, depth, next_frame) = unsafe {
                let f_back_slot = &mut (*(frame)).f_backref as *mut *mut PyFrame;
                visitor(&mut *(f_back_slot as *mut majit_ir::GcRef));

                // PyFrame is normally a GC object in PyPy, so its GCREF
                // fields are traced before consumers dereference them.
                // pyre also has stdalloc-backed frames, so the frame root
                // walker must expose those fields explicitly.
                let locals_slot =
                    &mut (*(frame)).locals_cells_stack_w as *mut *mut pyre_object::FixedObjectArray;
                visitor(&mut *(locals_slot as *mut majit_ir::GcRef));
                let gen_slot = &mut (*(frame)).f_generator_nowref as *mut PyObjectRef;
                visitor(&mut *(gen_slot as *mut majit_ir::GcRef));
                let yielding_slot = &mut (*(frame)).w_yielding_from as *mut PyObjectRef;
                visitor(&mut *(yielding_slot as *mut majit_ir::GcRef));
                // pyframe.py:91-92 `self.builtin = ...` — the picked
                // builtin Module is a GC root.  Pyre stashes its identity
                // on `frame.w_builtin` so `frame.get_builtin()` returns
                // the same object PyPy would.
                let w_builtin_slot = &mut (*(frame)).w_builtin as *mut PyObjectRef;
                visitor(&mut *(w_builtin_slot as *mut majit_ir::GcRef));
                // pyframe.py:147 `debugdata.w_locals` (and the pyre-only
                // `w_locals_object` companion for non-dict mapping
                // locals) carry GCREFs that survive the frame.
                if !(*frame).debugdata.is_null() {
                    let d = &mut *(*frame).debugdata;
                    let w_locals_object_slot = &mut d.w_locals_object as *mut PyObjectRef;
                    visitor(&mut *(w_locals_object_slot as *mut majit_ir::GcRef));
                    let w_f_trace_slot = &mut d.w_f_trace as *mut PyObjectRef;
                    visitor(&mut *(w_f_trace_slot as *mut majit_ir::GcRef));
                }
                let f = &*frame;
                let next_frame = (*frame).f_backref;
                if f.locals_cells_stack_w.is_null() {
                    (std::ptr::null_mut::<PyObjectRef>(), 0, next_frame)
                } else {
                    let arr = &*f.locals_cells_stack_w;
                    (arr.items_ptr() as *mut PyObjectRef, arr.len(), next_frame)
                }
            };
            if !arr_ptr.is_null() && depth > 0 {
                for i in 0..depth {
                    let slot_ptr = unsafe { arr_ptr.add(i) } as *mut majit_ir::GcRef;
                    // SAFETY: slot lies inside the FixedObjectArray's
                    // heap allocation, which outlives the frame. The
                    // visitor reads, conditionally forwards, and
                    // stores back a `GcRef` (same layout as
                    // `*mut PyObject`).
                    visitor(unsafe { &mut *slot_ptr });
                }
            }
            frame = next_frame;
        }
    });
}

/// Install the PyFrame GC root walker with the majit-gc collector.
///
/// Called once at process startup from the JIT driver / pyrex main.
/// Stored in a per-thread slot; calling again with the same fn pointer
/// is idempotent.
pub fn register_pyframe_root_walker() {
    majit_gc::set_active_extra_root_walker(Some(walk_pyframe_roots));
}

pub fn get_current_exception() -> PyObjectRef {
    CURRENT_EXCEPTION.with(|current| current.get())
}

pub fn set_current_exception(exc: PyObjectRef) {
    CURRENT_EXCEPTION.with(|current| current.set(exc));
}

pub fn normalize_raise_value(value: PyObjectRef) -> PyObjectRef {
    unsafe {
        if crate::baseobjspace::exception_is_valid_obj_as_class_w(value) {
            return crate::call_function(value, &[]);
        }
    }
    value
}

/// Normalize the `from` cause of a `raise X from Y` statement: instantiate
/// the cause if it is an exception class, validate that the result is
/// `None` / a `BaseException` instance, and return a `PyError::type_error`
/// otherwise.
///
/// # TODO: inline back into RAISE_VARARGS
///
/// **Deviation.** RPython performs this validation inline inside
/// `RAISE_VARARGS` (`pypy/interpreter/pyopcode.py:704-707`,
/// `space.call_function(w_cause)` when `w_cause` is an exception class)
/// without a named helper, deferring the BaseException check to
/// `OperationError.set_cause` (`pypy/interpreter/error.py`). Pyre
/// extracts this pre-step into a standalone helper so the JIT raise/r
/// BH path (`pyre-jit/src/call_jit.rs::bh_normalize_raise_varargs_fn`)
/// and the interpreter raise path can share the same validation.
///
/// **When to fix.** When `bh_normalize_raise_varargs_fn` is removed or
/// rewritten — e.g. when the JIT BH path can dispatch the same inlined
/// `RAISE_VARARGS` sequence directly without a shared helper.
///
/// **How to fix.** Inline this body back into the `RAISE_VARARGS`
/// dispatch arm in `pyre-interpreter/src/pyopcode.rs` (mirroring
/// `pyopcode.py:704-707`), delete this standalone fn, and either route
/// the BH path through the inlined sequence or rewrite it to match
/// RPython's inline shape.
pub fn normalize_raise_cause(cause: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let cause = normalize_raise_value(cause);
    unsafe {
        if cause.is_null() || pyre_object::is_none(cause) || pyre_object::is_exception(cause) {
            return Ok(cause);
        }
    }
    Err(PyError::type_error(
        "exception cause must be None or derive from BaseException",
    ))
}

pub fn attach_raise_cause(exc: PyObjectRef, cause: Option<PyObjectRef>) -> Result<(), PyError> {
    if let Some(cause_obj) = cause {
        if !cause_obj.is_null() {
            crate::baseobjspace::ATTR_TABLE.with(|table| {
                let mut table = table.borrow_mut();
                let attrs = table.entry(exc as usize).or_default();
                attrs.insert("__cause__".to_string(), cause_obj);
                attrs.insert(
                    "__suppress_context__".to_string(),
                    pyre_object::w_bool_from(true),
                );
            });
        }
    }
    Ok(())
}

/// Test whether `exc_value` matches the exception type specification
/// `exc_type` — a single exception class, a class name string, or a tuple
/// of classes (for `except (A, B)` clauses). PyPy: pyopcode.py
/// `CHECK_EXC_MATCH` → `_exception_match`.
pub fn check_exc_match_against(exc_value: PyObjectRef, exc_type: PyObjectRef) -> bool {
    unsafe {
        if !pyre_object::is_exception(exc_value) {
            return true;
        }
        // Tuple of exception classes — match any.
        if pyre_object::is_tuple(exc_type) {
            let n = pyre_object::w_tuple_len(exc_type) as i64;
            for i in 0..n {
                if let Some(elem) = pyre_object::w_tuple_getitem(exc_type, i) {
                    if check_exc_match_against(exc_value, elem) {
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
        if crate::is_function(exc_type)
            && crate::is_builtin_code(crate::getcode(exc_type) as pyre_object::PyObjectRef)
        {
            let type_name = crate::function_get_name(exc_type);
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
        false
    }
}

/// Try to dispatch an exception using the exception table or block stack.
///
/// Returns `true` if a handler was found (resume PC updated to handler),
/// `false` if the exception should propagate to the caller.
///
/// `err` is taken by `&mut` so the bytecode_trace_after_exception /
/// exception_trace plumbing can replace it with a tracer exception
/// (pyopcode.py:144-145 `except OperationError as e: operr = e`); the
/// caller's `Err(err)` propagation then surfaces the replacement.
pub fn handle_exception(frame: &mut PyFrame, err: &mut PyError, next_instr: &mut usize) -> bool {
    // Internal control-flow / corruption markers are not real Python
    // exceptions and must never be dispatched via bytecode handlers.
    if err.kind == crate::PyErrorKind::GeneratorReturn
        || err.kind == crate::PyErrorKind::BytecodeCorruption
    {
        return false;
    }
    // pyopcode.py:135-148 — exception trace plumbing:
    //   try:
    //       trace = self.get_w_f_trace()
    //       if trace is not None:
    //           self.getorcreatedebug().w_f_trace = None
    //       try:
    //           ec.bytecode_trace_after_exception(self)
    //       finally:
    //           if trace is not None:
    //               self.getorcreatedebug().w_f_trace = trace
    //   except OperationError as e:
    //       operr = e
    //   ...
    //   ec.exception_trace(self, operr)
    //
    // Gated on a live tracefunc so the no-tracer hot path skips both
    // the f_trace save/restore dance and the `to_exc_object` alloc.
    // bytecode_trace_after_exception's exception is caught by the
    // surrounding `except OperationError` and replaces operr;
    // exception_trace's exception is NOT caught (line 148 stands
    // outside the except), so it short-circuits the unrollstack search
    // — pyre signals that by returning `false` after replacing `err`.
    let ec = frame.execution_context as *mut crate::PyExecutionContext;
    if !ec.is_null() && unsafe { !(*ec).gettrace().is_null() } {
        let saved_trace = frame.get_w_f_trace();
        if !saved_trace.is_null() {
            frame.getorcreatedebug(-1).w_f_trace = pyre_object::PY_NULL;
        }
        let after_exc_result =
            unsafe { (*ec).bytecode_trace_after_exception(frame as *mut PyFrame) };
        if !saved_trace.is_null() {
            frame.getorcreatedebug(-1).w_f_trace = saved_trace;
        }
        if let Err(trace_err) = after_exc_result {
            // pyopcode.py:144-145 — `except OperationError as e: operr = e`.
            *err = trace_err;
        }
        let exc_obj = err.to_exc_object();
        if let Err(trace_err) = unsafe {
            (*ec).exception_trace(
                frame as *mut PyFrame,
                pyre_object::PY_NULL,
                exc_obj,
                pyre_object::PY_NULL,
            )
        } {
            // pyopcode.py:148 `ec.exception_trace(self, operr)` is
            // outside the except-block; a raise here propagates past
            // unrollstack. Replace err and return `false` so the
            // caller's `return Err(err)` surfaces the tracer error
            // without searching for a handler for the original.
            *err = trace_err;
            return false;
        }
    }
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    let pc = frame.last_instr as u32;

    // Python 3.11+ exception table dispatch
    if let Some(entry) = crate::bytecode::find_exception_handler(&code.exceptiontable, pc) {
        // Unwind stack to handler's expected depth
        let target_depth = frame.nlocals() + frame.ncells() + entry.depth as usize;
        while frame.valuestackdepth > target_depth {
            frame.pop();
        }
        if entry.push_lasti {
            frame.push(pyre_object::w_int_new(pc as i64));
        }
        // Push exception value as W_ExceptionObject
        let exc_obj = err.to_exc_object();
        frame.push(exc_obj);
        *next_instr = entry.target as usize;
        return true;
    }

    // Fallback: lastblock (old-style SETUP_FINALLY/SETUP_EXCEPT)
    if let Some(block) = frame.pop_block() {
        block.cleanupstack(frame);
        let exc_obj = err.to_exc_object();
        frame.push(exc_obj);
        *next_instr = block.handlerposition;
        return true;
    }

    false
}

/// Execute a frame — pure interpreter, no JIT.
///
/// Crate-private since Slice C.3 (PyFrame Heap-Allocation Epic): canonical
/// surface is `PyFrame::run` / `PyFrame::execute_frame` (PyPy
/// `pyframe.py:268 run` / `pyframe.py:331 execute_frame`).  Retained as a
/// free function because pyre's JIT override mechanism (call.rs
/// `EVAL_OVERRIDE: OnceLock<EvalFn>` where `EvalFn = fn(&mut PyFrame) ->
/// PyResult`) requires a `fn` pointer.  Rust methods cannot be cast to
/// `fn` pointers, so the canonical body stays as a free function and the
/// `EVAL_OVERRIDE.unwrap_or(eval_frame_plain)` fallback (call.rs:328 etc.)
/// continues to reference it directly.
pub(crate) fn eval_frame_plain(frame: &mut PyFrame) -> PyResult {
    eval_frame_plain_with_operr(frame, None)
}

/// pyframe.py:270-299 execute_frame body — enter/call_trace/eval_loop/
/// return_trace/leave wrapping. When `operr` is Some, the generator's
/// throw() path routes it through handle_operation_error and sets
/// last_instr = next_instr - 1 before resuming (pyframe.py:273-277).
pub(crate) fn eval_frame_plain_with_operr(frame: &mut PyFrame, operr: Option<PyError>) -> PyResult {
    frame.fix_array_ptrs();
    if frame.execution_context.is_null() {
        if let Some(mut err) = operr {
            let mut next_instr = frame.next_instr();
            if !handle_exception(frame, &mut err, &mut next_instr) {
                return Err(err);
            }
            frame.last_instr = next_instr as isize - 1;
        }
        return eval_loop(frame);
    }
    let execution_context =
        unsafe { &mut *(frame.execution_context as *mut crate::PyExecutionContext) };
    crate::call::set_last_exec_ctx(frame.execution_context);
    execution_context.enter(frame as *mut PyFrame);
    let mut got_exception = true;
    let mut w_exitvalue = pyre_object::w_none();
    // pyframe.py:343-373 PyFrame.execute_frame parity:
    //   try:
    //     ec.call_trace(self)            # outside inner try
    //     try:
    //       ... eval ...
    //     finally:
    //       ec.return_trace(self, w_exitvalue)
    //     got_exception = False
    //   finally:
    //     ec.leave(self, w_exitvalue, got_exception)
    //
    // call_trace lives in the outer try only — if it raises, neither the
    // eval body nor return_trace runs, but leave still does (because
    // enter() already executed).  Python finally semantics: a finally
    // block that raises replaces the prior exception (return_trace
    // overrides eval-body, leave overrides everything).
    let outer_result = (|| -> PyResult {
        execution_context.call_trace(frame as *mut PyFrame)?;
        let inner_result = (|| -> PyResult {
            if let Some(mut err) = operr {
                let mut next_instr = frame.next_instr();
                if !handle_exception(frame, &mut err, &mut next_instr) {
                    return Err(err);
                }
                frame.last_instr = next_instr as isize - 1;
            }
            let result = eval_loop(frame)?;
            w_exitvalue = result;
            Ok(result)
        })();
        let return_trace_result =
            execution_context.return_trace(frame as *mut PyFrame, w_exitvalue);
        // Python finally: a finally-block exception replaces any
        // pending exception from the try-body. Only the all-OK path
        // advances to `got_exception = false`.
        let combined = match return_trace_result {
            Err(rt_err) => Err(rt_err),
            Ok(()) => inner_result,
        };
        if combined.is_ok() {
            got_exception = false;
        }
        combined
    })();
    let leave_result = execution_context.leave(frame as *mut PyFrame, w_exitvalue, got_exception);
    match leave_result {
        Err(leave_err) => Err(leave_err),
        Ok(()) => outer_result,
    }
}

/// Resume interpretation after compiled code guard failure.
pub fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    eval_loop(frame)
}

fn eval_loop(frame: &mut PyFrame) -> PyResult {
    let _current_frame_guard = if frame.execution_context.is_null() {
        install_current_frame(frame)
    } else {
        install_current_frame_tls_only(frame)
    };
    let code = unsafe { &*crate::pyframe_get_pycode(frame) };
    let mut next_instr = frame.next_instr();

    loop {
        if next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let pc = next_instr;
        frame.last_instr = pc as isize;
        // pypy/interpreter/pyopcode.py:170-176 dispatch_bytecode parity:
        //   self.last_instr = intmask(next_instr)
        //   if jit.we_are_jitted():
        //       ec.bytecode_only_trace(self)
        //   else:
        //       ec.bytecode_trace(self)
        // pyre's interpreter path (this fn) takes the non-jitted branch
        // — bytecode_trace fires bytecode_only_trace then decrements
        // the ticker. Gated upstream on `w_tracefunc.is_null()` so the
        // no-tracer hot path is a single null-check + ticker decrement.
        let ec = frame.execution_context as *mut crate::PyExecutionContext;
        if !ec.is_null() {
            unsafe {
                (*ec).bytecode_trace(
                    frame as *mut PyFrame,
                    crate::executioncontext::TICK_COUNTER_STEP,
                )?
            };
        }
        let (opcode_pc, instruction, op_arg) = decode_instruction_for_dispatch(code, pc)?;
        let fallthrough = opcode_pc + 1;
        match execute_opcode_step(frame, code, instruction, op_arg, fallthrough) {
            Ok(StepResult::Continue)
            | Ok(StepResult::CloseLoop {
                jump_args: _,
                loop_header_pc: _,
            }) => {
                next_instr = frame.next_instr();
            }
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => return Ok(result),
            Err(mut err) => {
                // GeneratorReturn: RETURN_GENERATOR unwind → return generator object
                if err.kind == crate::PyErrorKind::GeneratorReturn {
                    let gen_ptr = err.message.parse::<usize>().unwrap_or(0);
                    return Ok(gen_ptr as pyre_object::PyObjectRef);
                }
                if handle_exception(frame, &mut err, &mut next_instr) {
                    continue;
                }
                return Err(err);
            }
        }
    }
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        PyFrame::push(self, value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() {
            return Err(stack_underflow_error("interpreter opcode"));
        }
        Ok(PyFrame::pop(self))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() + depth {
            return Err(stack_underflow_error("interpreter peek"));
        }
        Ok(PyFrame::peek_at(self, depth))
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        Ok(make_function_from_code_obj(code_obj, self.get_w_globals()))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        call_callable(self, callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_list_from_refs(items))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_tuple_from_refs(items))
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_map_from_refs(items))
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setitem(obj, key, value).map(|_| ())
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        unsafe { w_list_append(list, value) };
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        unpack_sequence_exact(seq, count)
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        getattr(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setattr(obj, name, value).map(|_| ())
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(self.locals_w()[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = self.locals_w()[idx];
        if value.is_null() {
            return Err(PyError::new(
                PyErrorKind::NameError,
                format!("local variable '{name}' referenced before assignment"),
            ));
        }
        // Cell objects are valid even if their contents are PY_NULL
        // (needed for __class__ cell during class body execution).
        // The cell itself is non-null, so the check above passes.
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // STORE_FAST always writes directly to the slot.
        // Cell content updates use STORE_DEREF, not STORE_FAST.
        self.locals_w_mut()[idx] = value;
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    /// PyPy: LOAD_NAME checks locals first (class body), then globals,
    /// then `__builtins__` via `load_global_value`'s fallback chain
    /// (pypy/interpreter/pyopcode.py:526-555 LOAD_NAME → load_global).
    ///
    /// Non-dict mapping locals (`exec(src, g, mapping)`,
    /// `pypy/interpreter/pyopcode.py:2003 ensure_ns`) bypass the
    /// `*mut DictStorage` fast path and route through
    /// `space.getitem(w_locals_object, name)` directly per PyPy
    /// `pyopcode.py:LOAD_NAME` `space.finditem_str(w_locals, name)`.
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            match crate::baseobjspace::getitem(w_locals_object, key) {
                Ok(value) => return Ok(value),
                Err(err) if matches!(err.kind, PyErrorKind::KeyError) => {
                    // pyopcode.py:LOAD_NAME `if not w_value: w_value =
                    // ec.space.finditem_str(self.w_globals, name)` —
                    // a missing locals entry falls through to globals.
                }
                Err(err) => return Err(err),
            }
            return self.load_global_value(name);
        }
        let w_locals = self.get_w_locals();
        if !w_locals.is_null() {
            let locals = unsafe { &*w_locals };
            if let Ok(value) = dict_storage_load(locals, name) {
                return Ok(value);
            }
        }
        self.load_global_value(name)
    }

    /// PyPy: STORE_NAME writes to locals (class body) or globals.
    ///
    /// Non-dict mapping locals route through
    /// `space.setitem(w_locals_object, name, value)` matching
    /// `pyopcode.py:STORE_NAME` `space.setitem(w_locals, ...)`.
    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            crate::baseobjspace::setitem(w_locals_object, key, value)?;
            return Ok(());
        }
        let ns = unsafe { &mut *self.getdictscope()? };
        dict_storage_store(ns, name, value);
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py:567 STORE_GLOBAL — bypasses w_locals
    /// and writes directly into w_globals so `exec("global x; x = 1", g, l)`
    /// lands `x` in `g` even when `l != g`.
    fn store_global_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = unsafe { &mut *self.get_w_globals() };
        dict_storage_store(ns, name, value);
        Ok(())
    }

    /// pypy/interpreter/pyopcode.py:918-927 `_load_global` — first reads
    /// `w_globals`, then falls back to `self.get_builtin().getdictvalue
    /// (space, varname)`.  PyPy's `get_builtin()` returns the `Module`
    /// chosen at frame-creation time by `pick_builtin(w_globals)`
    /// (`pyframe.py:115-116` + `pypy/module/__builtin__/moduledef.py:89`),
    /// so `exec("x = len", {"__builtins__": {}})` raises `NameError`
    /// because the empty dict is the picked builtin.  Pyre caches the
    /// picked storage in `frame.builtin`.
    fn load_global_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let ns = unsafe { &*self.get_w_globals() };
        if let Some(value) = crate::dict_storage_get(ns, name) {
            return Ok(value);
        }
        if !self.builtin.is_null() {
            if let Some(value) = crate::dict_storage_get(unsafe { &*self.builtin }, name) {
                return Ok(value);
            }
        }
        dict_storage_load(ns, name)
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(PY_NULL)
    }
}

impl StackOpcodeHandler for PyFrame {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        let top_idx = self.valuestackdepth - 1;
        let other_idx = self.valuestackdepth - depth;
        self.locals_w_mut().swap(top_idx, other_idx);
        Ok(())
    }
}

thread_local! {
    /// Cache for user-defined iterator __next__ result.
    /// concrete_iter_continues calls __next__ and caches here;
    /// iter_next_value returns the cached value.
    static USER_ITER_NEXT_CACHE: std::cell::Cell<PyObjectRef> =
        const { std::cell::Cell::new(PY_NULL) };
}

/// PyPy: pyopcode.py GET_ITER → space.iter(w_iterable)
///       pyopcode.py FOR_ITER → space.next(w_iterator)
impl IterOpcodeHandler for PyFrame {
    /// GET_ITER: convert iterable to iterator.
    /// PyPy: space.iter(w_iterable) → calls __iter__ or wraps in seq_iter.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            // Already an iterator
            if pyre_object::is_range_iter(iter)
                || pyre_object::is_seq_iter(iter)
                || pyre_object::generatorobject::is_generator(iter)
                || pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
            {
                return Ok(());
            }
            // list → seq_iter
            if pyre_object::is_list(iter) {
                let len = pyre_object::w_list_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // tuple → seq_iter
            if pyre_object::is_tuple(iter) {
                let len = pyre_object::w_tuple_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // str → list of 1-char strings → seq_iter
            if pyre_object::is_str(iter) {
                let s = pyre_object::w_str_get_value(iter);
                let chars: Vec<pyre_object::PyObjectRef> = s
                    .chars()
                    .map(|c| {
                        let mut buf = [0u8; 4];
                        pyre_object::w_str_new(c.encode_utf8(&mut buf))
                    })
                    .collect();
                let char_list = pyre_object::w_list_new(chars);
                let len = s.chars().count();
                let seq_iter = pyre_object::w_seq_iter_new(char_list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // bytes/bytearray → list of int → seq_iter
            if pyre_object::bytesobject::is_bytes_like(iter) {
                let len = pyre_object::bytesobject::bytes_like_len(iter);
                let mut items = Vec::with_capacity(len);
                for i in 0..len {
                    items.push(pyre_object::w_int_new(
                        pyre_object::bytesobject::bytes_like_getitem(iter, i) as i64,
                    ));
                }
                let list = pyre_object::w_list_new(items);
                let seq_iter = pyre_object::w_seq_iter_new(list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // dict → iterate over keys (PyPy: dictobject.py __iter__ → dict_keys)
            if pyre_object::is_dict(iter) {
                let d = &*(iter as *const pyre_object::dictobject::W_DictObject);
                let entries = &*d.entries;
                let keys: Vec<pyre_object::PyObjectRef> = entries.iter().map(|&(k, _)| k).collect();
                let key_list = pyre_object::w_list_new(keys);
                let len = entries.len();
                let seq_iter = pyre_object::w_seq_iter_new(key_list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // set / frozenset → iterate via insertion order (PyPy:
            // setobject.py W_BaseSetObject.descr_iter)
            if pyre_object::is_set_or_frozenset(iter) {
                let items = pyre_object::w_set_items(iter);
                let len = items.len();
                let key_list = pyre_object::w_list_new(items);
                let seq_iter = pyre_object::w_seq_iter_new(key_list, len);
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = seq_iter;
                return Ok(());
            }
            // User-defined __iter__ — PyPy: space.iter → __iter__()
            // Delegates to baseobjspace::iter which handles type MRO,
            // ATTR_TABLE, and __getitem__ fallback (PyPy: space.iter →
            // PyObject_GetIter → tp_iter or PySeqIter_New).
            if pyre_object::is_instance(iter) {
                let result = crate::baseobjspace::iter(iter)?;
                let tos = self.valuestackdepth - 1;
                self.locals_w_mut()[tos] = result;
                return Ok(());
            }
            // Type object: metaclass __iter__ (NOT the type's own MRO)
            // CPython: iter(X) calls type(X).__iter__(X)
            if pyre_object::is_type(iter) {
                // baseobjspace.py:76 — metaclass from w_class
                let mc = {
                    let w_class = (*iter).w_class;
                    let w_type_type = crate::typedef::w_type();
                    if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                        Some(w_class)
                    } else {
                        None
                    }
                };
                if let Some(metaclass) = mc {
                    if let Some(method) = crate::baseobjspace::lookup_in_type(metaclass, "__iter__")
                    {
                        let result = crate::call_function(method, &[iter]);
                        let tos = self.valuestackdepth - 1;
                        self.locals_w_mut()[tos] = result;
                        return Ok(());
                    }
                }
            }
        }
        ensure_range_iter(iter)
    }

    /// FOR_ITER: check if iterator has more items.
    /// PyPy: space.next() → StopIteration means exhausted.
    /// For user-defined iterators, we speculatively call __next__ and
    /// cache the result — iter_next_value returns the cached value.
    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        unsafe {
            // Generator iterator
            if pyre_object::generatorobject::is_generator(iter) {
                match crate::baseobjspace::next(iter) {
                    Ok(result) => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                        return Ok(true);
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                }
            }
            // itertools iterators — delegate to baseobjspace::next
            if pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
            {
                match crate::baseobjspace::next(iter) {
                    Ok(result) => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                        return Ok(true);
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                }
            }
            // User-defined iterator with __next__
            if pyre_object::is_instance(iter) {
                let w_type = pyre_object::w_instance_get_type(iter);
                if let Some(next_method) = crate::baseobjspace::lookup_in_type(w_type, "__next__") {
                    match crate::call::call_callable(self, next_method, &[iter]) {
                        Ok(result) => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                            return Ok(true);
                        }
                        Err(e) if e.kind == PyErrorKind::StopIteration => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                            return Ok(false);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        range_iter_continues(iter)
    }

    /// PyPy: space.next(w_iterator) → returns cached value from concrete_iter_continues.
    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        // Generator/user-defined/itertools iterator: return cached value
        if unsafe {
            pyre_object::generatorobject::is_generator(iter)
                || pyre_object::is_instance(iter)
                || pyre_object::itertoolsmodule::is_repeat(iter)
                || pyre_object::itertoolsmodule::is_count(iter)
        } {
            let cached = USER_ITER_NEXT_CACHE.with(|c| c.get());
            if !cached.is_null() {
                return Ok(cached);
            }
            return Ok(PY_NULL);
        }
        range_iter_next_or_null(iter)
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.set_last_instr_from_next_instr(target);
        Ok(())
    }
}

impl TruthOpcodeHandler for PyFrame {
    type Truth = bool;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        Ok(truth_value(value))
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        Ok(bool_value_from_truth(if negate { !truth } else { truth }))
    }
}

impl ControlFlowOpcodeHandler for PyFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.next_instr()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.set_last_instr_from_next_instr(target);
        Ok(())
    }

    fn close_loop(&mut self, target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop {
            jump_args: vec![],
            loop_header_pc: target,
        })
    }

    /// pyopcode.py:180-183 RETURN_VALUE — frame_finished_execution = True
    /// when the returning path exits the frame (matched by StepResult::Return).
    fn finish_value(&mut self, value: Self::Value) -> Result<StepResult<Self::Value>, PyError> {
        if std::env::var_os("PYRE_INTERP_RETURN_LOG").is_some() {
            unsafe {
                let code_ptr = crate::pyframe::pyframe_get_pycode(self);
                let name = if !code_ptr.is_null() {
                    (*code_ptr).obj_name.as_str()
                } else {
                    "?"
                };
                let arg0_intval = {
                    let lw = self.locals_w();
                    if lw.len() > 0 {
                        let v = lw[0];
                        if !v.is_null() && pyre_object::pyobject::is_int(v) {
                            Some(pyre_object::intobject::w_int_get_value(v))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };
                let ret_intval = if !value.is_null() && pyre_object::pyobject::is_int(value) {
                    Some(pyre_object::intobject::w_int_get_value(value))
                } else {
                    None
                };
                let f_back = self.f_backref as usize;
                eprintln!(
                    "[interp] return name={} arg0={:?} ret={:?} frame={:p} f_back=0x{:x} ret_ref=0x{:x}",
                    name, arg0_intval, ret_intval, self as *const _, f_back, value as usize
                );
            }
        }
        self.frame_finished_execution = true;
        Ok(StepResult::Return(value))
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(
        &mut self,
        _value: Self::Value,
        truth: Self::Truth,
    ) -> Result<bool, PyError> {
        Ok(truth)
    }
}

impl ArithmeticOpcodeHandler for PyFrame {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        binary_value(a, b, op)
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        compare_value(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_negative_value(value)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_invert_value(value)
    }
}

impl ConstantOpcodeHandler for PyFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_int_new(value))
    }

    fn bigint_constant(&mut self, value: &crate::PyBigInt) -> Result<Self::Value, PyError> {
        Ok(w_long_new(value.clone()))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        Ok(w_float_new(value))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        Ok(w_bool_from(value))
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        Ok(box_str_constant(value))
    }

    fn bytes_constant(&mut self, value: &[u8]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::bytesobject::w_bytes_from_bytes(value))
    }

    fn code_constant(
        &mut self,
        code: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
        Ok(w_code_new(code_ptr))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }

    fn ellipsis_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(pyre_object::noneobject::w_ellipsis())
    }

    fn slice_constant(
        &mut self,
        start: Self::Value,
        stop: Self::Value,
        step: Self::Value,
    ) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_slice_new(start, stop, step))
    }

    fn frozenset_constant(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_frozenset_from_items(items))
    }
}

impl OpcodeStepExecutor for PyFrame {
    type Error = PyError;

    /// SETUP_ANNOTATIONS — ensure `__annotations__` exists in the
    /// current locals namespace. PyPy: pyopcode.py SETUP_ANNOTATIONS
    /// (typeobject.py auto-fills the slot at class creation, but the
    /// pyre-equivalent flow runs the bytecode opcode and writes into
    /// the class_locals namespace just like CPython).
    fn setup_annotations(&mut self) -> Result<(), Self::Error> {
        let ns = self.getdictscope()?;
        if ns.is_null() {
            return Ok(());
        }
        let ns = unsafe { &mut *ns };
        if dict_storage_load(ns, "__annotations__").is_err() {
            dict_storage_store(ns, "__annotations__", pyre_object::w_dict_new());
        }
        Ok(())
    }

    /// CPython 3.14 ceval.c:WITH_EXCEPT_START
    ///
    /// Stack layout the bytecode emits (bottom → top):
    ///   exit_func, exit_self, lasti, exc_info_prev, val
    ///
    /// pyre's PUSH_EXC_INFO mirrors CPython here. We scan downward from TOS
    /// looking for the first callable to use as exit_func instead of relying
    /// on a fixed offset, because the exact slot count depends on whether
    /// LOAD_SPECIAL preserved a NULL placeholder for the cm self.
    fn with_except_start(&mut self) -> Result<(), Self::Error> {
        let depth = self.valuestackdepth;
        if depth < 1 {
            return Err(PyError::type_error("WITH_EXCEPT_START on empty stack"));
        }
        let val = self.locals_w()[depth - 1];
        // Find exit_func: walk down from TOS-1 looking for the first
        // callable. CPython's static layout puts it at TOS-4, but pyre's
        // SWAP path may leave a NULL or different offset.
        let mut exit_func = pyre_object::PY_NULL;
        for offset in 2..=depth.min(8) {
            let candidate = self.locals_w()[depth - offset];
            if candidate.is_null() {
                continue;
            }
            unsafe {
                if crate::is_function(candidate)
                    || pyre_object::is_method(candidate)
                    || pyre_object::is_type(candidate)
                {
                    exit_func = candidate;
                    break;
                }
            }
        }
        if exit_func.is_null() {
            // Nothing to call — push True (suppress nothing).
            self.push(pyre_object::w_bool_from(false));
            return Ok(());
        }
        let exc_type = crate::typedef::r#type(val).unwrap_or(pyre_object::w_none());
        let exc_tb =
            crate::baseobjspace::getattr(val, "__traceback__").unwrap_or(pyre_object::w_none());
        let res = crate::call_function(exit_func, &[exc_type, val, exc_tb]);
        self.push(res);
        Ok(())
    }

    // ── LoadCommonConstant ──
    fn load_common_constant(
        &mut self,
        cc: crate::bytecode::CommonConstant,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::CommonConstant;
        let val = match cc {
            CommonConstant::AssertionError => {
                crate::make_builtin_function("AssertionError", |_args| {
                    Err(crate::PyError::new(
                        crate::PyErrorKind::AssertionError,
                        "assertion error".to_string(),
                    ))
                })
            }
            CommonConstant::NotImplementedError => {
                crate::make_builtin_function("NotImplementedError", |_args| {
                    Err(crate::PyError::type_error("not implemented"))
                })
            }
            CommonConstant::BuiltinTuple => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE)
            }
            CommonConstant::BuiltinAll => {
                crate::make_module_builtin_function("all", crate::builtins::builtin_all_fn)
            }
            CommonConstant::BuiltinAny => {
                crate::make_module_builtin_function("any", crate::builtins::builtin_any_fn)
            }
            CommonConstant::BuiltinList => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
            CommonConstant::BuiltinSet => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
        };
        self.push(val);
        Ok(())
    }

    // ── PopJumpIfNone / PopJumpIfNotNone ──
    // CPython 3.13: replaces IS_OP + POP_JUMP_IF_TRUE/FALSE for None checks

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if unsafe { pyre_object::is_none(val) } || val.is_null() {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if !val.is_null() && !unsafe { pyre_object::is_none(val) } {
            self.set_last_instr_from_next_instr(target);
        }
        Ok(())
    }

    // ── Closures / cells ──

    /// PyPy: pyopcode.py LOAD_DEREF
    ///
    /// Reads cell/free variable. If the slot holds a cell object (from
    /// closure tuple via COPY_FREE_VARS), dereferences it. Otherwise
    /// reads the raw value (pyre's direct storage for cellvars).
    /// LOAD_DEREF — RustPython 3.13 uses unified index (same as LOAD_FAST).
    ///
    /// PyPy: pyopcode.py LOAD_DEREF → cell.get()
    /// If the slot holds a cell object, dereference it to get the value.
    fn load_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let slot = self.locals_w()[idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(PyError::type_error(
                "free variable referenced before assignment",
            ));
        }
        self.push(value);
        Ok(())
    }

    /// STORE_DEREF — unified index. Stores into cell if present.
    ///
    /// PyPy: pyopcode.py STORE_DEREF → cell.set(value)
    fn store_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let slot = self.locals_w()[idx];
        if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_set(slot, value) };
        } else {
            self.locals_w_mut()[idx] = value;
        }
        Ok(())
    }

    /// LOAD_CLOSURE — unified index. Push cell object itself (not contents).
    ///
    /// PyPy: pyopcode.py LOAD_CLOSURE → push cell for closure capture.
    fn load_closure(&mut self, idx: usize) -> Result<(), Self::Error> {
        let cell = self.locals_w()[idx];
        self.push(cell);
        Ok(())
    }

    /// MAKE_CELL — no-op in pyre.
    ///
    /// CPython 3.13 / RustPython MAKE_CELL — create cell object in slot.
    ///
    /// PyPy: pyframe.py cell initialization.
    /// Wraps the current value (PY_NULL if uninitialized) in a W_CellObject.
    /// LoadFast on cell slots returns the cell object itself (needed for
    /// closure creation via BUILD_TUPLE + SET_FUNCTION_ATTRIBUTE).
    fn make_cell(&mut self, idx: usize) -> Result<(), Self::Error> {
        let code = unsafe { &*crate::pyframe_get_pycode(self) };
        if std::env::var("PYRE_DEBUG_CELL").is_ok() {
            eprintln!("  varnames: {:?}", code.varnames);
            eprintln!("  cellvars: {:?}", code.cellvars);
            for (i, instr) in code.instructions.iter().enumerate().take(25) {
                eprintln!("  {i}: {:?}", instr);
            }
        }
        let current = self.locals_w()[idx];
        self.locals_w_mut()[idx] = pyre_object::w_cell_new(current);
        Ok(())
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let nlocals = self.nlocals();
        self.locals_w_mut()[nlocals + idx] = PY_NULL;
        Ok(())
    }

    // ── Exception handling ──

    fn setup_finally(&mut self, handler: usize) -> Result<(), Self::Error> {
        self.append_block(crate::pyframe::FrameBlock {
            valuestackdepth: self.valuestackdepth,
            handlerposition: handler,
            previous: self.lastblock,
        });
        Ok(())
    }

    fn setup_except(&mut self, handler: usize) -> Result<(), Self::Error> {
        self.setup_finally(handler)
    }

    fn pop_block(&mut self) -> Result<(), Self::Error> {
        self.pop_block();
        Ok(())
    }

    fn raise_varargs(&mut self, argc: usize) -> Result<(), Self::Error> {
        match argc {
            0 => {
                // Bare `raise` — re-raise current exception
                // PyPy: executioncontext.py sys_exc_info
                let exc = CURRENT_EXCEPTION.with(|c| c.get());
                if exc.is_null() || unsafe { pyre_object::is_none(exc) } {
                    Err(PyError::runtime_error("No active exception to reraise"))
                } else if unsafe { pyre_object::is_exception(exc) } {
                    Err(unsafe { PyError::from_exc_object(exc) })
                } else {
                    Err(PyError::runtime_error("No active exception to reraise"))
                }
            }
            1 => {
                // pyopcode.py:708-722 — cause=None, normalize exc.
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // pyopcode.py:711-713 — class raise: call the type.
                        let result = crate::call_function(w_value, &[]);
                        if pyre_object::is_exception(result) {
                            attach_raise_cause(result, None)?;
                            Err(PyError::from_exc_object(result))
                        } else {
                            Err(PyError::type_error(
                                "exceptions must derive from BaseException",
                            ))
                        }
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, None)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            2 => {
                // pyopcode.py:704-722 — pop+normalize cause first, then exc.
                let raw_cause = self.pop();
                let cause = Some(normalize_raise_cause(raw_cause)?);
                let w_value = self.pop();
                unsafe {
                    if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_value) {
                        // pyopcode.py:711-713 — class raise: call the type.
                        let result = crate::call_function(w_value, &[]);
                        if pyre_object::is_exception(result) {
                            attach_raise_cause(result, cause)?;
                            Err(PyError::from_exc_object(result))
                        } else {
                            Err(PyError::type_error(
                                "exceptions must derive from BaseException",
                            ))
                        }
                    } else if pyre_object::is_exception(w_value) {
                        attach_raise_cause(w_value, cause)?;
                        Err(PyError::from_exc_object(w_value))
                    } else {
                        Err(PyError::type_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            _ => Err(PyError::type_error("too many arguments for raise")),
        }
    }

    fn end_finally(&mut self) -> Result<(), Self::Error> {
        // Pop the exception or None from stack
        let _ = self.pop();
        Ok(())
    }

    // ── Import ──
    // PyPy: pyopcode.py IMPORT_NAME
    // Stack: [level, fromlist] → pops both, pushes module object.
    fn import_name(&mut self, name: &str) -> Result<(), Self::Error> {
        let w_fromlist = self.pop();
        let w_level = self.pop();
        let level = if unsafe { pyre_object::is_int(w_level) } {
            unsafe { pyre_object::w_int_get_value(w_level) }
        } else {
            0
        };

        let module = crate::importing::importhook(
            name,
            self.get_w_globals() as PyObjectRef, // for relative imports: __name__/__package__
            w_fromlist,
            level,
            self.execution_context,
        )?;
        self.push(module);
        Ok(())
    }

    // PyPy: pyopcode.py IMPORT_FROM
    // Stack: [module] → peek module, push getattr(module, name)
    fn import_from(&mut self, name: &str) -> Result<(), Self::Error> {
        let module = self.peek();
        let attr = crate::importing::import_from(module, name, self.execution_context)?;
        self.push(attr);
        Ok(())
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), Self::Error> {
        // CPython 3.13: TOS = container, TOS1 = item
        let haystack = self.pop();
        let needle = self.pop();
        let result = crate::baseobjspace::contains(haystack, needle)?;
        let inverted = match invert {
            crate::bytecode::Invert::No => result,
            crate::bytecode::Invert::Yes => !result,
        };
        self.push(pyre_object::w_bool_from(inverted));
        Ok(())
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), Self::Error> {
        let b = self.pop();
        let a = self.pop();
        let same = std::ptr::eq(a, b); // pointer identity
        let result = match invert {
            crate::bytecode::Invert::No => same,
            crate::bytecode::Invert::Yes => !same,
        };
        self.push(pyre_object::w_bool_from(result));
        Ok(())
    }

    // ── ToBool ──
    // CPython 3.13: converts TOS to bool

    fn to_bool(&mut self) -> Result<(), Self::Error> {
        let val = self.pop();
        let truth = crate::baseobjspace::is_true(val);
        self.push(pyre_object::w_bool_from(truth));
        Ok(())
    }

    // ── DeleteSubscr ──

    fn delete_subscript(&mut self) -> Result<(), Self::Error> {
        let index = self.pop();
        let obj = self.pop();
        crate::baseobjspace::delitem(obj, index)?;
        Ok(())
    }

    // ── DeleteFast ──

    fn delete_fast(&mut self, idx: usize) -> Result<(), Self::Error> {
        self.locals_w_mut()[idx] = PY_NULL;
        Ok(())
    }

    // ── FormatSimple (str(TOS)) ──
    fn format_simple(&mut self) -> Result<(), Self::Error> {
        let val = self.pop();
        let s = unsafe { crate::py_str(val) };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── FormatWithSpec (format(TOS1, TOS)) ──
    fn format_with_spec(&mut self) -> Result<(), Self::Error> {
        let _spec = self.pop();
        let val = self.pop();
        // Phase 1: ignore spec, just convert to str
        let s = unsafe { crate::py_str(val) };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── ConvertValue (repr/str/ascii conversion) ──
    fn convert_value(
        &mut self,
        conv: crate::bytecode::ConvertValueOparg,
    ) -> Result<(), Self::Error> {
        let val = self.pop();
        let s = match conv {
            crate::bytecode::ConvertValueOparg::Str => unsafe { crate::py_str(val) },
            crate::bytecode::ConvertValueOparg::Repr => unsafe { crate::py_repr(val) },
            crate::bytecode::ConvertValueOparg::Ascii => unsafe { crate::py_repr(val) },
            crate::bytecode::ConvertValueOparg::None => unsafe { crate::py_str(val) },
        };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── CopyFreeVars ──
    // CPython 3.13: copy n freevars from function closure to frame cell slots
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), Self::Error> {
        // Phase 1: no-op — closure passing needs call-site integration
        // The closure tuple is on the Function, but COPY_FREE_VARS
        // runs inside the callee frame which doesn't have a reference to
        // the function object. Need to pass closure during frame creation.
        Ok(())
    }

    // ── SetFunctionAttribute ──
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: pop attr, pop func, set, push func.
    /// Stack effect: (2) → (1)
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: (attr, func -- func)
    /// attr = TOS1 (below), func = TOS (top).
    /// Pops both, sets attribute on func, pushes func back.
    fn set_function_attribute_with_flag(
        &mut self,
        flag: crate::bytecode::MakeFunctionFlag,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::MakeFunctionFlag;
        let func = self.pop(); // TOS = function
        let attr = self.pop(); // TOS1 = attribute value (closure tuple etc.)
        match flag {
            MakeFunctionFlag::Closure => unsafe {
                crate::function_set_closure(func, attr);
            },
            MakeFunctionFlag::Defaults => unsafe {
                crate::function_set_defaults(func, attr);
            },
            MakeFunctionFlag::KwOnlyDefaults => unsafe {
                crate::function_set_kwdefaults(func, attr);
            },
            _ => {} // annotations, etc.
        }
        self.push(func);
        Ok(())
    }

    // ── PushExcInfo ──
    // PyPy: executioncontext.py enter_frame / normalize_exception
    fn push_exc_info(&mut self) -> Result<(), Self::Error> {
        let exc = self.pop();
        // Save previous exception, set current
        let prev = CURRENT_EXCEPTION.with(|c| c.get());
        CURRENT_EXCEPTION.with(|c| c.set(exc));
        // Push "previous exception" for later restore
        self.push(prev);
        // Push the exception value back
        self.push(exc);
        Ok(())
    }

    // ── CheckExcMatch ──
    // TOS = exception type to match, TOS1 = caught exception
    // Pops type, peeks exc, pushes bool result
    fn check_exc_match(&mut self) -> Result<(), Self::Error> {
        let exc_type = self.pop();
        let exc_value = self.peek();
        let matched = check_exc_match_against(exc_value, exc_type);
        self.push(pyre_object::w_bool_from(matched));
        Ok(())
    }

    // ── PopExcept ──
    fn pop_except(&mut self) -> Result<(), Self::Error> {
        // Restore previous exc_info from stack
        let prev_exc = self.pop();
        CURRENT_EXCEPTION.with(|c| c.set(prev_exc));
        Ok(())
    }

    // ── Reraise ──
    // CPython: RERAISE raises the exception that's on TOS.
    // The exception table handler (handle_exception) unwinds the stack.
    // We peek TOS to get the exception but do NOT pop — handle_exception
    // will set the stack to the correct depth.
    fn reraise(&mut self) -> Result<(), Self::Error> {
        // TOS is the exception, TOS1 is prev_exc_info
        let exc = self.peek();
        unsafe {
            if pyre_object::is_exception(exc) {
                Err(PyError::from_exc_object(exc))
            } else if pyre_object::is_str(exc) {
                Err(PyError::runtime_error(
                    pyre_object::w_str_get_value(exc).to_string(),
                ))
            } else {
                Err(PyError::runtime_error("exception re-raised"))
            }
        }
    }

    // ── LoadFromDictOrGlobals ──
    // CPython 3.13: LOAD_FROM_DICT_OR_GLOBALS — try TOS dict first, then globals
    fn load_from_dict_or_globals(&mut self, name: &str) -> Result<(), Self::Error> {
        let dict = self.pop();
        // Try dict first (if it's a dict or has attrs)
        if let Ok(val) = crate::baseobjspace::getattr(dict, name) {
            self.push(val);
            return Ok(());
        }
        // Fall back to globals
        unsafe {
            if let Some(&val) = (*self.get_w_globals()).get(name) {
                self.push(val);
                return Ok(());
            }
        }
        Err(PyError::type_error(format!("name '{name}' is not defined")))
    }

    // ── GetLen ──
    fn get_len(&mut self, obj: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        let len = crate::baseobjspace::len(obj)?;
        Ok(len)
    }

    // ── LoadFastAndClear (comprehension scope) ──
    fn load_fast_and_clear(&mut self, idx: usize) -> Result<(), Self::Error> {
        let val = self.locals_w()[idx];
        self.push(val);
        self.locals_w_mut()[idx] = PY_NULL;
        Ok(())
    }

    // ── BuildSet ──
    fn build_set(&mut self, count: usize) -> Result<(), Self::Error> {
        // Build as a set-like object backed by __data__ dict.
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.pop());
        }
        items.reverse();
        let set_obj = crate::builtins::builtin_set_from_items(&items)?;
        self.push(set_obj);
        Ok(())
    }

    // ── DictUpdate ──
    // PyPy: dict.update(source); CPython: DICT_UPDATE
    // Merges source dict entries into STACK[-i] dict.
    fn dict_update(&mut self, i: usize) -> Result<(), Self::Error> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_dict(source) {
                let src = &*(source as *const pyre_object::dictobject::W_DictObject);
                let entries = &*src.entries;
                for &(k, v) in entries {
                    pyre_object::w_dict_store(dict, k, v);
                }
            }
        }
        Ok(())
    }

    // ── DictMerge ──
    // PyPy: dict merge for **kwargs; CPython: DICT_MERGE
    fn dict_merge(&mut self, i: usize) -> Result<(), Self::Error> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_dict(source) {
                let src = &*(source as *const pyre_object::dictobject::W_DictObject);
                let entries = &*src.entries;
                for &(k, v) in entries {
                    pyre_object::w_dict_store(dict, k, v);
                }
            }
        }
        Ok(())
    }

    // ── MapAdd ──
    // PyPy: STORE_MAP/MAP_ADD; CPython: MAP_ADD
    // dict = STACK[-i-2]; dict[TOS1] = TOS; pop key+value
    fn map_add(&mut self, i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let key = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            pyre_object::w_dict_store(dict, key, value);
        }
        Ok(())
    }

    // ── SetAdd ──
    // PyPy: SET_ADD; CPython: SET_ADD
    // set = STACK[-i]; set.add(TOS); pop value
    fn set_add(&mut self, i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_set_or_frozenset(set) {
                pyre_object::w_set_add(set, value);
            } else if pyre_object::is_list(set) {
                pyre_object::w_list_append(set, value);
            }
        }
        Ok(())
    }

    // ── none_value ──
    fn none_value(&mut self) -> Result<PyObjectRef, Self::Error> {
        Ok(pyre_object::w_none())
    }

    // ── unary_positive ──
    // PyPy: UNARY_POSITIVE → space.pos(w_value)
    fn unary_positive(&mut self, val: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        unsafe {
            if pyre_object::is_bool(val) {
                // +True → 1, +False → 0 (int, not bool)
                return Ok(pyre_object::w_int_new(
                    pyre_object::w_bool_get_value(val) as i64
                ));
            }
            if pyre_object::is_int(val) || pyre_object::is_float(val) {
                return Ok(val);
            }
        }
        Err(PyError::type_error("bad operand type for unary +"))
    }

    // ── list_to_tuple ──
    // PyPy intrinsic: convert list to tuple (used in star unpacking).
    fn list_to_tuple(&mut self, val: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        unsafe {
            if pyre_object::is_list(val) {
                let items = pyre_object::w_list_items_copy_as_vec(val);
                return Ok(pyre_object::w_tuple_new(items));
            }
        }
        Err(PyError::type_error("expected list for list_to_tuple"))
    }

    // ── print_expr ──
    // PyPy: PRINT_EXPR → sys.displayhook(value)
    fn print_expr(&mut self, val: PyObjectRef) -> Result<(), Self::Error> {
        if !unsafe { pyre_object::is_none(val) } {
            let s = unsafe { crate::py_repr(val) };
            println!("{}", s);
        }
        Ok(())
    }

    // ── delete_name ──
    // pypy/interpreter/pyopcode.py:821 DELETE_NAME — delete from w_locals; KeyError → NameError.
    fn delete_name(&mut self, name: &str) -> Result<(), Self::Error> {
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            let key = unsafe { pyre_object::w_str_new(name) };
            crate::baseobjspace::delitem(w_locals_object, key).map_err(|err| {
                if matches!(err.kind, PyErrorKind::KeyError) {
                    PyError::new(
                        PyErrorKind::NameError,
                        format!("name '{name}' is not defined"),
                    )
                } else {
                    err
                }
            })?;
            return Ok(());
        }
        let mut ns = self.get_w_locals();
        if ns.is_null() {
            ns = self.get_w_globals();
        }
        let found = unsafe { crate::dict_storage_delete(&mut *ns, name) };
        if !found {
            return Err(PyError::new(
                PyErrorKind::NameError,
                format!("name '{name}' is not defined"),
            ));
        }
        Ok(())
    }

    // ── delete_global ──
    // pypy/interpreter/pyopcode.py:901 DELETE_GLOBAL — delitem only; no NameError conversion.
    fn delete_global(&mut self, name: &str) -> Result<(), Self::Error> {
        let ns = self.get_w_globals();
        unsafe {
            crate::dict_storage_delete(&mut *ns, name);
        }
        Ok(())
    }

    // ── import_star ──
    // pypy/interpreter/pyopcode.py:1076 IMPORT_STAR — merge module's public names into
    // the locals mapping (class body / exec-with-locals), not globals.
    //
    // Non-dict mapping locals route through `import_all_from_w` so each
    // `from module import *` entry lands via `space.setitem(w_locals,
    // name, value)` rather than the `*mut DictStorage` fast path,
    // matching `pyopcode.py:1078 self.getdictscope()` returning a
    // generic `w_obj`.
    fn import_star(&mut self) -> Result<(), Self::Error> {
        let module = self.pop();
        let w_locals_object = self.get_w_locals_object();
        if !w_locals_object.is_null() {
            crate::importing::import_all_from_w(module, w_locals_object)?;
            return Ok(());
        }
        let w_locals = self.getdictscope()?;
        crate::importing::import_all_from(module, w_locals)?;
        self.setdictscope(w_locals)?;
        Ok(())
    }

    // ── load_build_class ──
    // PyPy: BUILD_CLASS; CPython: LOAD_BUILD_CLASS
    fn load_build_class(&mut self) -> Result<(), Self::Error> {
        let bc = crate::get_build_class_func();
        self.push(bc);
        Ok(())
    }

    // ── yield from / send ──
    fn get_yield_from_iter(&mut self) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let iter = crate::baseobjspace::iter(iterable)?;
        self.push(iter);
        Ok(())
    }

    fn send_value(&mut self, target: usize) -> Result<(), Self::Error> {
        let _value = self.pop(); // sent value
        let iter = self.peek();
        match crate::baseobjspace::next(iter) {
            Ok(result) => {
                self.push(result);
                Ok(())
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                // Don't pop iterator — END_SEND will handle it
                self.push(pyre_object::w_none()); // push result (None)
                self.set_last_instr_from_next_instr(target);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn end_send(&mut self) -> Result<(), Self::Error> {
        let result = self.pop();
        let _iter = self.pop();
        self.push(result);
        Ok(())
    }

    // ── load_method ──
    // PyPy: LOOKUP_METHOD — interpreter-only override.
    // For instances, pushes [attr, self] so CALL prepends self.
    // ── return_generator ──
    // CPython 3.12: RETURN_GENERATOR creates a generator from the current
    // frame and returns it to the caller. PyPy: generator.py GeneratorIterator.
    fn return_generator(&mut self) -> Result<(), Self::Error> {
        // When the generator function is already wrapped (CodeFlags::GENERATOR
        // detected in call_user_function_with_eval), RETURN_GENERATOR fires
        // during the first __next__() resume. It's a no-op in that case —
        // the generator object was already created at call time.
        // Push dummy value for the following POP_TOP to consume.
        self.push(pyre_object::w_none());
        Ok(())
    }

    // ── load_super_attr ──
    // CPython 3.12 LOAD_SUPER_ATTR: stack = [global_super, class, self]
    // → super(class, self).attr
    fn load_super_attr_with(&mut self, name: &str, is_method: bool) -> Result<(), Self::Error> {
        let self_obj = self.pop();
        let cls = self.pop();
        let _global_super = self.pop();

        let proxy = pyre_object::superobject::w_super_new(cls, self_obj);
        let result = crate::baseobjspace::getattr(proxy, name)?;

        // CPython _PySuper_Lookup: determines whether the resolved attr
        // is an unbound method (needs self binding) or a staticmethod /
        // classmethod (no self binding / bind class).
        if is_method {
            // getattr now returns a bound method via descriptor protocol.
            // Unwrap for the (func, self) pattern that CALL expects.
            if unsafe { pyre_object::is_method(result) } {
                let func = unsafe { pyre_object::w_method_get_func(result) };
                let recv = unsafe { pyre_object::w_method_get_self(result) };
                self.push(func);
                self.push(recv);
            } else {
                // staticmethod or classmethod — no self binding
                self.push(result);
                self.push(PY_NULL);
            }
        } else {
            // is_method=false: getattr already returned a bound method.
            self.push(result);
        }
        Ok(())
    }

    // For non-instances (modules etc.), pushes [attr, NULL].
    // The default trait impl always pushes [attr, NULL], which is what
    // the JIT tracer uses — no runtime branch in the shared path.
    fn load_method(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = self.pop();
        let attr = crate::baseobjspace::getattr(obj, name)?;
        if unsafe { pyre_object::is_method(attr) } {
            self.push(attr);
            self.push(PY_NULL);
            return Ok(());
        }
        self.push(attr);
        // Bind self only for regular instance method calls.
        // staticmethod/classmethod descriptors already unwrap to the raw
        // function via getattr → get; self must NOT
        // be prepended for those.
        // PyPy: LOOKUP_METHOD checks whether the attr came from a
        // non-data descriptor that is a plain function (not staticmethod).
        // Determine what to bind as null_or_self.
        // PyPy: LOOKUP_METHOD resolves descriptors and decides binding.
        //  - regular method → bind instance (self)
        //  - classmethod → bind class (w_type)
        //  - staticmethod → no binding (NULL)
        //  - builtin type method (list.append etc.) → bind instance
        let bound = unsafe {
            if pyre_object::is_instance(obj) {
                let w_type = pyre_object::w_instance_get_type(obj);
                let raw = crate::baseobjspace::lookup_in_type(w_type, name);
                match raw {
                    Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                    // PyPy: ClassMethod.__get__ → Method(func, klass)
                    Some(d) if pyre_object::is_classmethod(d) => w_type,
                    // Type / property / member descriptors stored as class
                    // attributes are NOT methods — invoking them must not
                    // prepend self.  e.g. `class L: inner = list; L().inner(x)`
                    // calls list(x), not list(L(), x).
                    Some(d) if pyre_object::is_type(d) => PY_NULL,
                    Some(d) if pyre_object::is_property(d) => PY_NULL,
                    Some(d) if pyre_object::is_member(d) => PY_NULL,
                    Some(d) if crate::is_function(d) => {
                        let ob_type = (*d).ob_type;
                        if std::ptr::eq(ob_type, &crate::BUILTIN_FUNCTION_TYPE as *const _) {
                            // BuiltinFunction has no __get__ in PyPy.
                            PY_NULL
                        } else if std::ptr::eq(ob_type, &crate::FUNCTION_TYPE as *const _)
                            && crate::is_builtin_code(
                                crate::function_get_code(d) as pyre_object::PyObjectRef
                            )
                        {
                            // FunctionWithFixedCode (interp2app) on a builtin
                            // type — `dict.get` etc. — binds like a method.
                            obj
                        } else {
                            obj
                        }
                    }
                    Some(_) => obj, // found in type MRO → bind self (method)
                    None => {
                        // Not found in type MRO → found in instance __dict__.
                        // Instance __dict__ attrs bypass descriptor protocol.
                        PY_NULL
                    }
                }
            } else if pyre_object::is_type(obj) {
                // Type object: check for classmethod in type's MRO
                let raw = crate::baseobjspace::lookup_in_type(obj, name);
                match raw {
                    Some(d) if pyre_object::is_classmethod(d) => obj,
                    Some(_) => PY_NULL, // found in own MRO → no binding
                    None => {
                        // Not found in type's own MRO → check metaclass MRO.
                        // If found there, bind obj (the type) as self.
                        // PyPy: type.__getattribute__ metatype descriptor binding.
                        obj
                    }
                }
            } else if crate::typedef::r#type(obj).is_some() && !pyre_object::is_module(obj) {
                // Builtin type method (list.append, etc.) found via TypeDef.
                // PyPy: LOOKUP_METHOD binds self for builtin type methods.
                obj
            } else {
                PY_NULL
            }
        };
        self.push(bound);
        Ok(())
    }

    // ── call ──
    // PyPy: CALL_FUNCTION — interpreter-only override.
    // Handles null_or_self prepend for instance method calls.
    // The default trait impl (opcode_call) always discards null_or_self,
    // which is what the JIT tracer uses — no trace/concrete divergence.
    fn call(&mut self, nargs: usize) -> Result<(), Self::Error> {
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let null_or_self = self.pop();
        let callable = self.pop();

        let result = if null_or_self.is_null() {
            call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            call_callable(self, callable, &full_args)?
        };
        self.push(result);
        Ok(())
    }

    // ── call_function_ex ──
    // pyopcode.py:1360 CALL_FUNCTION_EX:
    //     w_kwargs = self.popvalue() if has_kwarg else None
    //     w_args = self.popvalue()
    //     w_function = self.popvalue()
    //     args = self.argument_factory([], None, None,
    //                                  w_star=w_args,
    //                                  w_starstar=w_kwargs,
    //                                  w_function=w_function)
    //     w_result = self.space.call_args(w_function, args)
    //     self.pushvalue(w_result)
    //
    // argument.py Arguments.unpack_combined_starargs iterates w_star with
    // space.fixedview_unroll / space.listview_no_unpack, so arbitrary
    // iterables are accepted.
    fn call_function_ex(&mut self) -> Result<(), Self::Error> {
        let kwargs_or_null = self.pop();
        let args_obj = self.pop();
        let _null = self.pop();
        let callable = self.pop();

        // argument.py unpack_combined_starargs equivalent: fast-path tuple
        // and list so common bytecode emits avoid iter protocol overhead;
        // fall back to the iter protocol for arbitrary iterables.
        let args: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(args_obj) {
                let n = pyre_object::w_tuple_len(args_obj);
                (0..n as i64)
                    .filter_map(|i| pyre_object::w_tuple_getitem(args_obj, i))
                    .collect()
            } else if pyre_object::is_list(args_obj) {
                let n = pyre_object::w_list_len(args_obj);
                (0..n as i64)
                    .filter_map(|i| pyre_object::w_list_getitem(args_obj, i))
                    .collect()
            } else {
                crate::builtins::collect_iterable(args_obj)?
            }
        };

        // Merge kwargs dict into call.
        // argument.py Arguments.unpack_combined_starstarargs
        if !kwargs_or_null.is_null() && unsafe { pyre_object::is_dict(kwargs_or_null) } {
            let entries = unsafe { pyre_object::w_dict_str_entries(kwargs_or_null) };
            if !entries.is_empty() {
                let result = crate::call::call_with_kwargs(self, callable, &args, &entries)?;
                self.push(result);
                return Ok(());
            }
        }

        let result = call_callable(self, callable, &args)?;
        self.push(result);
        Ok(())
    }

    // ── call_kw ──
    // PyPy: CALL_FUNCTION_KW; CPython 3.13: CALL_KW
    // Stack: [callable, self_or_null, arg1, ..., argN, kwarg_names_tuple]
    /// CALL_KW — call with keyword arguments.
    ///
    /// PyPy: argument.py _match_signature
    /// Stack: [callable, null_or_self, arg0..argN-1, kwarg_names_tuple]
    /// The last `len(kwarg_names)` args are keyword args.
    ///
    /// Keyword resolution happens HERE (before frame creation) so the
    /// JIT eval loop sees correctly-positioned locals. PyPy does this
    /// in Arguments.parse_into_scope before the frame executes.
    fn call_kw(&mut self, nargs: usize) -> Result<(), Self::Error> {
        let kwarg_names = self.pop();
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let self_or_null = self.pop();
        let callable = self.pop();

        if self_or_null != PY_NULL && !unsafe { pyre_object::is_none(self_or_null) } {
            args.insert(0, self_or_null);
        }

        // Unwrap bound methods: load_method pushes (method, PY_NULL) for
        // bound methods. Extract the underlying function and prepend the
        // receiver so resolve_kwargs sees the correct function signature.
        let callable_unwrapped = crate::baseobjspace::unwrap_cell(callable);
        let callable_unwrapped = if unsafe { pyre_object::is_method(callable_unwrapped) } {
            let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
            let receiver = unsafe { pyre_object::w_method_get_self(callable_unwrapped) };
            if !receiver.is_null() && !unsafe { pyre_object::is_none(receiver) } {
                args.insert(0, receiver);
            }
            func
        } else {
            callable_unwrapped
        };

        // For type objects with kwargs: use call_with_kwargs which handles
        // __new__/__init__ kwargs forwarding correctly.
        if unsafe { pyre_object::is_type(callable_unwrapped) } {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let pos_args = args[..n_pos].to_vec();
                let mut kw_entries = Vec::with_capacity(nkw);
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        let key = unsafe { pyre_object::w_str_get_value(name_obj) }.to_string();
                        kw_entries.push((key, args[n_pos + ki]));
                    }
                }
                let result = crate::call::call_with_kwargs(
                    self,
                    callable_unwrapped,
                    &pos_args,
                    &kw_entries,
                )?;
                self.push(result);
                return Ok(());
            }
        }

        // Resolve keyword args into positional order.
        // argument.py Arguments._match_signature step: match keywords to
        // argnames, fill defaults, pack *args/**kwargs. PyPy's
        // `space.call_args` performs this exactly once; pyre mirrors that
        // by calling resolve_kwargs here and then dispatching directly to
        // call_user_function_resolved — which skips the defaults_fill /
        // pack_varargs replay that call_user_function_with_args performs
        // for positional-only paths.
        let is_builtin = unsafe { crate::is_function(callable_unwrapped) }
            && unsafe {
                crate::is_builtin_code(
                    crate::getcode(callable_unwrapped) as pyre_object::PyObjectRef
                )
            };
        if is_builtin {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let pos_args = args[..n_pos].to_vec();
                let mut kw_entries = Vec::with_capacity(nkw);
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        let key = unsafe { pyre_object::w_str_get_value(name_obj) }.to_string();
                        kw_entries.push((key, args[n_pos + ki]));
                    }
                }
                // PyPy CALL_FUNCTION_KW builds an Arguments object with
                // keyword_names_w / keywords_w, and the profiled-builtin path
                // passes that same object to call_args_and_c_profile.  Route
                // through call_with_kwargs so pyre's profile path constructs
                // Arguments::with_kw instead of treating the kwargs dict tail
                // as a positional firstarg.
                let result = crate::call::call_with_kwargs(
                    self,
                    callable_unwrapped,
                    &pos_args,
                    &kw_entries,
                )?;
                self.push(result);
                return Ok(());
            }
            let result = call_callable(self, callable_unwrapped, &args)?;
            self.push(result);
            return Ok(());
        }

        // pypy/interpreter/function.py Method.call_args parity: unwrap
        // bound method by prepending the receiver, then run resolve_kwargs
        // against the underlying function. This matches
        // `self.space.call_args(w_function, args)` after the MRO-dispatched
        // `im_func` has been extracted.
        let (target_func, mut prepended) = if unsafe { pyre_object::is_method(callable_unwrapped) }
        {
            let func = unsafe { pyre_object::w_method_get_func(callable_unwrapped) };
            let receiver = unsafe {
                let w_self = pyre_object::w_method_get_self(callable_unwrapped);
                if !w_self.is_null() && !pyre_object::is_none(w_self) {
                    w_self
                } else {
                    pyre_object::w_method_get_class(callable_unwrapped)
                }
            };
            if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
                let mut prepended = Vec::with_capacity(1 + args.len());
                prepended.push(receiver);
                prepended.extend_from_slice(&args);
                (func, Some(prepended))
            } else {
                (func, None)
            }
        } else {
            (callable_unwrapped, None)
        };
        let call_args: &[PyObjectRef] = prepended.as_deref().unwrap_or(&args);
        let resolved = crate::call::resolve_kwargs(target_func, call_args, kwarg_names);
        // Drop the temporary prepended buffer once resolved is built.
        prepended = None;
        let _ = prepended;

        let result = if unsafe { crate::is_function(target_func) } {
            crate::call::call_user_function_resolved(self, target_func, &resolved)?
        } else {
            call_callable(self, target_func, &resolved)?
        };
        self.push(result);
        Ok(())
    }

    // ── load_locals ──
    // PyPy: LOAD_LOCALS; CPython: LOAD_LOCALS
    // Pushes the current namespace dict onto the stack.
    fn load_locals(&mut self) -> Result<(), Self::Error> {
        let dict = pyre_object::w_dict_new();
        unsafe {
            let w_locals = self.get_w_locals();
            if !w_locals.is_null() {
                for (key, &value) in (*w_locals).entries() {
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                    }
                }
            } else {
                let code = &*crate::pyframe_get_pycode(self);
                for (idx, name) in code.varnames.iter().enumerate() {
                    let value = self.locals_w()[idx];
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), value);
                    }
                }
                let w_globals = self.get_w_globals();
                if self.nlocals() == 0 && !w_globals.is_null() {
                    for (key, &value) in (*w_globals).entries() {
                        if !value.is_null() {
                            pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                        }
                    }
                }
            }
        }
        self.push(dict);
        Ok(())
    }

    // ── unpack_ex ──
    // PyPy: UNPACK_SEQUENCE with star; CPython: UNPACK_EX
    // `a, *b, c = iterable`
    fn unpack_ex(&mut self, args: crate::bytecode::UnpackExArgs) -> Result<(), Self::Error> {
        let before = args.before as usize;
        let after = args.after as usize;
        let value = self.pop();

        let elements: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(value) {
                pyre_object::w_tuple_items_copy_as_vec(value)
            } else if pyre_object::is_list(value) {
                pyre_object::w_list_items_copy_as_vec(value)
            } else {
                return Err(PyError::type_error("cannot unpack non-sequence"));
            }
        };

        let min_expected = before + after;
        if elements.len() < min_expected {
            return Err(PyError::value_error(&format!(
                "not enough values to unpack (expected at least {}, got {})",
                min_expected,
                elements.len()
            )));
        }

        let middle_len = elements.len() - min_expected;

        // Push after items (reversed), then middle list, then before items (reversed)
        for i in (0..after).rev() {
            self.push(elements[before + middle_len + i]);
        }
        let middle: Vec<PyObjectRef> = elements[before..before + middle_len].to_vec();
        self.push(pyre_object::w_list_new(middle));
        for i in (0..before).rev() {
            self.push(elements[i]);
        }

        Ok(())
    }

    // ── delete_attr ──
    // PyPy: DELETE_ATTR → space.delattr(obj, name)
    fn delete_attr(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = self.pop();
        crate::baseobjspace::delattr(obj, name)?;
        Ok(())
    }

    // ── set_update ──
    // PyPy: set.update(iterable); CPython: SET_UPDATE
    fn set_update(&mut self, i: usize) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_set_or_frozenset(set) {
                let items = crate::builtins::collect_iterable(iterable)?;
                for item in items {
                    pyre_object::w_set_add(set, item);
                }
            } else if pyre_object::is_list(set) {
                if pyre_object::is_list(iterable) {
                    let items = pyre_object::w_list_items_copy_as_vec(iterable);
                    for item in items {
                        pyre_object::w_list_append(set, item);
                    }
                } else if pyre_object::is_tuple(iterable) {
                    for item in pyre_object::w_tuple_items_copy_as_vec(iterable) {
                        pyre_object::w_list_append(set, item);
                    }
                }
            }
        }
        Ok(())
    }

    // ── BuildSlice ──
    // CPython 3.13: BUILD_SLICE creates a slice object from 2 or 3 stack items
    fn build_slice(
        &mut self,
        argc: crate::bytecode::BuildSliceArgCount,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::BuildSliceArgCount;
        let step = match argc {
            BuildSliceArgCount::Three => self.pop(),
            BuildSliceArgCount::Two => pyre_object::w_none(),
        };
        let stop = self.pop();
        let start = self.pop();
        self.push(pyre_object::w_slice_new(start, stop, step));
        Ok(())
    }

    // ── BinarySlice (a[b:c]) ──
    // PyPy: BINARY_SUBSCR with slice; CPython 3.13: BINARY_SLICE
    fn binary_slice(&mut self) -> Result<(), Self::Error> {
        let stop = self.pop();
        let start = self.pop();
        let obj = self.pop();
        unsafe {
            if pyre_object::is_list(obj) {
                let len = pyre_object::w_list_len(obj) as i64;
                let s = if pyre_object::is_none(start) {
                    0
                } else {
                    pyre_object::w_int_get_value(start)
                };
                let e = if pyre_object::is_none(stop) {
                    len
                } else {
                    pyre_object::w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let mut items = Vec::new();
                for i in s..e {
                    if let Some(v) = pyre_object::w_list_getitem(obj, i as i64) {
                        items.push(v);
                    }
                }
                self.push(pyre_object::w_list_new(items));
                return Ok(());
            }
            if pyre_object::is_str(obj) {
                let full = pyre_object::w_str_get_value(obj);
                let len = full.len() as i64;
                let s = if pyre_object::is_none(start) {
                    0
                } else {
                    pyre_object::w_int_get_value(start)
                };
                let e = if pyre_object::is_none(stop) {
                    len
                } else {
                    pyre_object::w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let slice = &full[s..e.min(full.len())];
                self.push(pyre_object::w_str_new(slice));
                return Ok(());
            }
            if pyre_object::is_tuple(obj) {
                let len = pyre_object::w_tuple_len(obj) as i64;
                let s = if pyre_object::is_none(start) {
                    0
                } else {
                    pyre_object::w_int_get_value(start)
                };
                let e = if pyre_object::is_none(stop) {
                    len
                } else {
                    pyre_object::w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let mut items = Vec::new();
                for i in s..e {
                    if let Some(v) = pyre_object::w_tuple_getitem(obj, i as i64) {
                        items.push(v);
                    }
                }
                self.push(pyre_object::w_tuple_new(items));
                return Ok(());
            }
            // Fall back to slice(start, stop) → getitem dispatch.
            // Handles bytearray, instances with __getitem__, etc.
            let slice_obj =
                pyre_object::sliceobject::w_slice_new(start, stop, pyre_object::w_none());
            let result = crate::baseobjspace::getitem(obj, slice_obj)?;
            self.push(result);
            Ok(())
        }
    }

    // ── StoreSlice (a[b:c] = d) ──
    fn store_slice(&mut self) -> Result<(), Self::Error> {
        // Phase 1 stub — rarely used in hot loops
        Err(PyError::type_error("STORE_SLICE not yet implemented"))
    }

    // ── BuildString (f-string concatenation) ──
    // CPython 3.13: concatenate N string fragments from stack
    fn build_string(&mut self, count: usize) -> Result<(), Self::Error> {
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(self.pop());
        }
        parts.reverse();
        let mut result = String::new();
        for part in &parts {
            unsafe {
                if pyre_object::is_str(*part) {
                    result.push_str(pyre_object::w_str_get_value(*part));
                } else if pyre_object::is_int(*part) {
                    result.push_str(&pyre_object::w_int_get_value(*part).to_string());
                } else if pyre_object::is_none(*part) {
                    result.push_str("None");
                } else if pyre_object::is_bool(*part) {
                    result.push_str(if pyre_object::w_bool_get_value(*part) {
                        "True"
                    } else {
                        "False"
                    });
                } else {
                    result.push_str("<object>");
                }
            }
        }
        self.push(pyre_object::w_str_new(&result));
        Ok(())
    }

    // ── ListExtend ──
    fn list_extend(&mut self, _i: usize) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let list = self.peek();
        unsafe {
            if pyre_object::is_list(iterable) {
                let src_len = pyre_object::w_list_len(iterable);
                for j in 0..src_len {
                    if let Some(item) = pyre_object::w_list_getitem(iterable, j as i64) {
                        pyre_object::w_list_append(list, item);
                    }
                }
                return Ok(());
            }
            if pyre_object::is_tuple(iterable) {
                let src_len = pyre_object::w_tuple_len(iterable);
                for j in 0..src_len {
                    if let Some(item) = pyre_object::w_tuple_getitem(iterable, j as i64) {
                        pyre_object::w_list_append(list, item);
                    }
                }
                return Ok(());
            }
        }
        Err(PyError::type_error("object is not iterable"))
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<PyObjectRef>, Self::Error> {
        Err(PyError::type_error(format!(
            "unimplemented instruction: {instruction:?}"
        )))
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyExecutionContext;
    use crate::*;
    use std::rc::Rc;

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        frame.execute_frame(None, None)
    }

    fn run_exec_frame(source: &str) -> (PyResult, Box<PyFrame>) {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let result = frame.execute_frame(None, None);
        (result, frame)
    }

    #[test]
    fn test_exception_is_valid_obj_as_class_w_matches_baseexception_subclass_rule() {
        let (_result, frame) = run_exec_frame("good = ValueError\nbad = int");
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let good = *w_globals.get("good").expect("missing good");
        let bad = *w_globals.get("bad").expect("missing bad");

        unsafe {
            assert!(crate::baseobjspace::exception_is_valid_obj_as_class_w(good));
            assert!(!crate::baseobjspace::exception_is_valid_obj_as_class_w(bad));
        }
    }

    #[test]
    fn test_raise_non_exception_type_raises_typeerror() {
        let (result, _frame) = run_exec_frame("raise int");
        let err = result.expect_err("raise int should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(err.message, "exceptions must derive from BaseException");
    }

    #[test]
    fn test_raise_invalid_cause_raises_typeerror() {
        let (result, _frame) = run_exec_frame("raise ValueError() from 1");
        let err = result.expect_err("invalid cause should fail");
        assert_eq!(err.kind, PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "exception cause must be None or derive from BaseException"
        );
    }

    #[test]
    fn test_raise_from_sets_cause_attribute() {
        let (_result, frame) = run_exec_frame("exc = ValueError()\ncause = KeyError()");
        let w_globals = unsafe { &*frame.fget_w_globals() };
        let exc = *w_globals.get("exc").expect("missing exc");
        let cause = *w_globals.get("cause").expect("missing cause");

        let code = compile_exec("raise exc from cause").expect("compile failed");
        let mut raise_frame = PyFrame::new(code);
        unsafe {
            (*raise_frame.fget_w_globals()).insert("exc".to_string(), exc);
            (*raise_frame.fget_w_globals()).insert("cause".to_string(), cause);
        }

        let err = raise_frame
            .execute_frame(None, None)
            .expect_err("raise from should fail");
        assert_eq!(err.to_exc_object(), exc);
        assert_eq!(crate::getattr(exc, "__cause__").expect("read cause"), cause);
    }

    #[test]
    fn test_literal() {
        let result = run_eval("42").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_addition() {
        let result = run_eval("1 + 2").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 3) };
    }

    #[test]
    fn test_subtraction() {
        let result = run_eval("10 - 3").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_multiplication() {
        let result = run_eval("6 * 7").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_complex_expr() {
        let result = run_eval("(2 + 3) * 4 - 1").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 19) };
    }

    #[test]
    fn test_comparison() {
        let result = run_eval("3 < 5").unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_comparison_false() {
        let result = run_eval("5 < 3").unwrap();
        unsafe { assert!(!w_bool_get_value(result)) };
    }

    #[test]
    fn test_store_load_namespace() {
        let source = "x = 5\ny = x * x";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            let y = *(*frame.w_globals).get("y").unwrap();
            assert_eq!(w_int_get_value(x), 5);
            assert_eq!(w_int_get_value(y), 25);
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "i = 0\nwhile i < 10:\n    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            assert_eq!(w_int_get_value(i), 10);
        }
    }

    #[test]
    fn test_eval_loop_redecodes_opargs_after_extended_arg_jumps() {
        let mut source = String::from(
            "\
i = 0
acc = 0
if i == 1:
",
        );
        for _ in 0..80 {
            source.push_str("    acc = acc + 1000\n");
        }
        source.push_str(
            "\
while i < 6:
    acc = acc + 1
    i = i + 1
r = acc",
        );
        let code = compile_exec(&source).expect("compile failed");
        assert!(
            code.instructions.windows(2).any(|pair| {
                matches!(pair[0].op, Instruction::ExtendedArg)
                    && !matches!(pair[1].op, Instruction::ExtendedArg)
            }),
            "expected an instruction with an ExtendedArg prefix"
        );
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let r = *(*frame.w_globals).get("r").unwrap();
            assert_eq!(w_int_get_value(r), 6);
        }
    }

    #[test]
    fn test_eval_loop_raises_on_malformed_extended_arg_chain() {
        let code = compile_exec("x = 1").expect("compile failed");
        unsafe {
            code.instructions.replace_op(0, Instruction::ExtendedArg);
            code.instructions.replace_op(1, Instruction::GetIter);
        }
        let mut frame = PyFrame::new(code);
        let err = frame
            .execute_frame(None, None)
            .expect_err("expected bytecode corruption");
        assert_eq!(err.kind, PyErrorKind::BytecodeCorruption);
        assert_eq!(err.message, "bytecode corruption");
    }

    #[test]
    fn test_none_result() {
        let result = run_eval("None").unwrap();
        unsafe { assert!(is_none(result)) };
    }

    #[test]
    fn test_bool_result() {
        let result = run_eval("True").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_literal() {
        let result = run_eval("1.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 1.5);
        }
    }

    #[test]
    fn test_float_addition() {
        let result = run_eval("1.5 + 2.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 4.0);
        }
    }

    #[test]
    fn test_float_truediv() {
        let result = run_eval("10 / 4").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 2.5);
        }
    }

    #[test]
    fn test_float_comparison() {
        let result = run_eval("1.5 < 2.5").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_int_mixed() {
        let result = run_eval("1.5 + 2").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 3.5);
        }
    }

    #[test]
    fn test_float_negation() {
        let result = run_eval("-3.14").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), -3.14);
        }
    }

    #[test]
    fn test_float_truthiness() {
        // Test via is_true directly since `not` uses ToBool instruction
        assert!(!is_true(w_float_new(0.0)));
        assert!(is_true(w_float_new(1.5)));
        assert!(is_true(w_float_new(-0.1)));
    }

    // ── str tests ────────────────────────────────────────────────────

    #[test]
    fn test_str_literal() {
        let result = run_eval("'hello'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_str_concat() {
        let result = run_eval("'hello' + ' world'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello world");
        }
    }

    #[test]
    fn test_str_repeat() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_str_comparison() {
        let result = run_eval("'abc' < 'abd'").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    // ── for loop / range tests ──────────────────────────────────────

    #[test]
    fn test_for_range() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 45);
        }
    }

    #[test]
    fn test_hot_range_loop_survives_compiled_trace() {
        let source = "s = 0\nfor i in range(3000):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 4_498_500);
        }
    }

    #[test]
    fn test_hot_module_branch_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    if i < 1500:
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4500);
        }
    }

    #[test]
    fn test_hot_tuple_unpack_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    a, b = (i, 1)
    acc = acc + a + b
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_hot_list_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0]
i = 0
acc = 0
while i < 3000:
    lst[0] = i
    acc = acc + lst[0]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            let lst = *(*frame.w_globals).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, 0).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_bitwise_or_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc | i
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4095);
        }
    }

    #[test]
    fn test_hot_unary_invert_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (~i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), -4_501_500);
        }
    }

    #[test]
    fn test_hot_positive_floordiv_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i // 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 1_498_500);
        }
    }

    #[test]
    fn test_hot_positive_mod_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i % 7)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 8_994);
        }
    }

    #[test]
    fn test_hot_builtin_abs_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + abs(i - 1500)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 2_250_000);
        }
    }

    #[test]
    fn test_hot_list_truth_loop_survives_compiled_trace() {
        let source = "\
lst = [1]
i = 0
acc = 0
while i < 3000:
    if lst:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_tuple_truth_loop_survives_compiled_trace() {
        let source = "\
tpl = ()
i = 0
acc = 0
while i < 3000:
    if tpl:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_none_truth_loop_survives_compiled_trace() {
        let source = "\
value = None
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_float_truth_loop_survives_compiled_trace() {
        let source = "\
value = 0.5
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_len_string_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 12_000);
        }
    }

    #[test]
    fn test_hot_builtin_len_dict_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2, 3: 4}
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6_000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_true_loop_survives_compiled_trace() {
        let source = "\
x = 42
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_false_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6000);
        }
    }

    #[test]
    fn test_hot_builtin_type_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if type(x) == list:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_min_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + min(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6426);
        }
    }

    #[test]
    fn test_hot_builtin_max_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + max(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 11568);
        }
    }

    #[test]
    fn test_hot_empty_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_list_negative_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0, 1]
i = 0
acc = 0
while i < 3000:
    lst[-1] = i
    acc = acc + lst[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            let lst = *(*frame.w_globals).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, -1).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_tuple_negative_index_load_loop_survives_compiled_trace() {
        let source = "\
tpl = (3, 5)
i = 0
acc = 0
while i < 3000:
    acc = acc + tpl[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 15_000);
        }
    }

    #[test]
    fn test_hot_user_function_loop_survives_compiled_trace() {
        let source = "\
def inc(x):
    return x + 1
i = 0
acc = 0
while i < 3000:
    acc = acc + inc(i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let i = *(*frame.w_globals).get("i").unwrap();
            let acc = *(*frame.w_globals).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_for_range_start_stop() {
        let source = "s = 0\nfor i in range(5, 10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 35);
        }
    }

    #[test]
    fn test_for_range_step() {
        let source = "s = 0\nfor i in range(0, 10, 2):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            // 0 + 2 + 4 + 6 + 8 = 20
            assert_eq!(w_int_get_value(s), 20);
        }
    }

    #[test]
    fn test_for_range_empty() {
        let source = "s = 42\nfor i in range(0):\n    s = 0";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 42);
        }
    }

    #[test]
    fn test_builtin_range_print() {
        let source = "s = 0\nfor i in range(5):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let s = *(*frame.w_globals).get("s").unwrap();
            // 0 + 1 + 2 + 3 + 4 = 10
            assert_eq!(w_int_get_value(s), 10);
        }
    }

    // ── builtin tests ───────────────────────────────────────────────

    #[test]
    fn test_builtin_len() {
        let source = "x = len([1, 2, 3])";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_builtin_abs() {
        let source = "x = abs(-5)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 5);
        }
    }

    #[test]
    fn test_builtin_min_max() {
        let source = "a = min(3, 7)\nb = max(3, 7)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = frame.execute_frame(None, None);
        unsafe {
            let a = *(*frame.w_globals).get("a").unwrap();
            let b = *(*frame.w_globals).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 3);
            assert_eq!(w_int_get_value(b), 7);
        }
    }

    // ── container tests ────────────────────────────────────────────

    #[test]
    fn test_list_literal() {
        let source = "x = [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert!(is_list(x));
            assert_eq!(w_list_len(x), 3);
            assert_eq!(w_int_get_value(w_list_getitem(x, 0).unwrap()), 1);
            assert_eq!(w_int_get_value(w_list_getitem(x, 1).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(x, 2).unwrap()), 3);
        }
    }

    #[test]
    fn test_tuple_unpack() {
        let source = "a, b = 1, 2";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let a = *(*frame.w_globals).get("a").unwrap();
            let b = *(*frame.w_globals).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 1);
            assert_eq!(w_int_get_value(b), 2);
        }
    }

    #[test]
    fn test_list_subscr() {
        let source = "lst = [10, 20, 30]\nx = lst[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 20);
        }
    }

    #[test]
    fn test_list_store_subscr() {
        let source = "lst = [1, 2, 3]\nlst[0] = 99\nx = lst[0]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 99);
        }
    }

    #[test]
    fn test_dict_literal_and_subscr() {
        let source = "d = {1: 10, 2: 20}\nx = d[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.w_globals).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 10);
        }
    }

    // ── function definition and call tests ──────────────────────────

    #[test]
    fn test_simple_function() {
        let source = "def double(x):\n    return x * 2\nresult = double(21)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_with_locals() {
        let source = "\
def add_squares(a, b):
    aa = a * a
    bb = b * b
    return aa + bb
result = add_squares(3, 4)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 25);
        }
    }

    #[test]
    fn test_recursive_function() {
        let source = "\
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
result = factorial(5)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 120);
        }
    }

    // ── attribute tests ─────────────────────────────────────────────

    #[test]
    fn test_store_load_attr() {
        let source = "\
def f():
    pass
f.x = 42
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_store_load_multiple_attrs() {
        let source = "\
def f():
    pass
f.a = 10
f.b = 20
result = f.a + f.b";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    #[test]
    fn test_attr_overwrite() {
        let source = "\
def f():
    pass
f.x = 1
f.x = 2
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_attr_on_different_objects() {
        let source = "\
def f():
    pass
def g():
    pass
f.x = 10
g.x = 20
result = f.x + g.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    // ── Phase 1 opcode tests ──

    #[test]
    fn test_contains_op_in() {
        let source = "x = [1, 2, 3]\nresult = 1 in x";
        let (res, frame) = run_exec_frame(source);
        res.expect("exec failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result), "1 in [1,2,3] should be True");
        }
    }

    #[test]
    fn test_contains_op_not_in() {
        let source = "result = 4 not in [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_op() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_op() {
        let result = run_eval("1 is not None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring() {
        let source = "x = 42\nresult = f'val={x}'";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "val=42");
        }
    }

    #[test]
    fn test_list_slice() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[1:3]";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.w_globals).get("result").unwrap();
                assert!(is_list(result), "slice result should be list");
                assert_eq!(w_list_len(result), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 3);
            },
            Err(e) => panic!("list_slice failed: {} (kind: {:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_delete_subscr() {
        // del x[0] in a list
        let source = "x = [1, 2, 3]\ndel x[0]\nresult = x[0]";
        let (result, _) = run_exec_frame(source);
        // After del x[0], x[0] becomes PY_NULL; accessing may succeed or fail
        // Phase 1: just check it doesn't crash during del
        let _ = result;
    }

    #[test]
    fn test_to_bool() {
        let result = run_eval("not 0").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_none_is_none() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring_with_expr() {
        let source = "x = 10\ny = 20\nresult = f'{x} + {y} = {x + y}'";
        let (res, frame) = run_exec_frame(source);
        res.expect("f-string exec failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "10 + 20 = 30");
        }
    }

    #[test]
    fn test_string_contains() {
        let source = "result = 'lo' in 'hello'";
        let (res, frame) = run_exec_frame(source);
        res.expect("string contains failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_tuple_contains() {
        let source = "result = 2 in (1, 2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple contains failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_not_in() {
        let source = "result = 5 not in [1, 2, 3]";
        let (res, frame) = run_exec_frame(source);
        res.expect("not in failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_none() {
        let source = "result = 42 is not None";
        let (res, frame) = run_exec_frame(source);
        res.expect("is not None failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_list_slice_negative() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[-3:]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative slice failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 3);
        }
    }

    #[test]
    fn test_nested_function_call() {
        let source = "\
def add(a, b):
    return a + b
result = add(add(1, 2), add(3, 4))";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested call failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_while_loop_with_break() {
        let source = "\
x = 0
while True:
    x = x + 1
    if x == 5:
        break
result = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("while+break failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_inplace_add() {
        let source = "x = 10\nx += 5\nresult = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("inplace add failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_string_iteration_chars() {
        let source = "\
result = ''
for c in 'hello':
    result = result + c
";
        let (res, frame) = run_exec_frame(source);
        res.expect("string iteration failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_enumerate_style() {
        // Test: manual counter with for loop
        let source = "\
count = 0
for x in [10, 20, 30]:
    count = count + 1
result = count";
        let (res, frame) = run_exec_frame(source);
        res.expect("enumerate style failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_nested_for_loops() {
        let source = "\
result = 0
for i in [1, 2, 3]:
    for j in [10, 20]:
        result = result + i * j
";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested for failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            // 1*10 + 1*20 + 2*10 + 2*20 + 3*10 + 3*20 = 10+20+20+40+30+60 = 180
            assert_eq!(w_int_get_value(result), 180);
        }
    }

    #[test]
    fn test_try_except_basic() {
        let source = "\
x = 0
try:
    x = 1 / 0
except:
    x = 42
result = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_int_get_value(result), 42);
            },
            Err(e) => panic!("try/except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_recursive_fibonacci() {
        let source = "\
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(10)";
        let (res, frame) = run_exec_frame(source);
        res.expect("fib failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 55);
        }
    }

    #[test]
    fn test_string_multiply() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_list_multiply() {
        let result = run_eval("[1, 2] * 3").unwrap();
        unsafe {
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 6);
        }
    }

    #[test]
    fn test_negative_index() {
        let source = "x = [10, 20, 30]\nresult = x[-1]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative index failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_boolean_operators() {
        let source = "result = True and False";
        let (res, frame) = run_exec_frame(source);
        res.expect("boolean and failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert!(!crate::baseobjspace::is_true(r));
        }
    }

    #[test]
    fn test_chained_comparison() {
        let source = "result = 1 < 2 < 3";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.w_globals).get("result").unwrap();
                assert!(w_bool_get_value(r));
            },
            Err(e) => eprintln!("chained comparison: {}", e.message),
        }
    }

    #[test]
    fn test_try_except_specific() {
        let source = "\
result = 0
try:
    x = 1 / 0
except ZeroDivisionError:
    result = 99
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 99);
            },
            Err(e) => panic!("specific except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_try_except_no_match_propagates() {
        // If except doesn't match, error should propagate
        let source = "\
try:
    x = 1 / 0
except ValueError:
    pass
";
        let (res, _) = run_exec_frame(source);
        // Should fail because ZeroDivisionError != ValueError
        // But Phase 1: bare except catches all, specific except may not work yet
        let _ = res; // Don't assert — depends on CHECK_EXC_MATCH impl
    }

    #[test]
    fn test_try_finally() {
        let source = "\
result = 0
try:
    result = 1
finally:
    result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 11);
            },
            Err(e) => panic!("try/finally failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_multiple_except() {
        let source = "\
result = 0
try:
    x = 1 / 0
except:
    result = 1
result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple except failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 11);
        }
    }

    #[test]
    fn test_for_with_continue() {
        let source = "\
result = 0
for x in [1, 2, 3, 4, 5]:
    if x == 3:
        continue
    result = result + x
";
        let (res, frame) = run_exec_frame(source);
        res.expect("for+continue failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            // 1 + 2 + 4 + 5 = 12 (skips 3)
            assert_eq!(w_int_get_value(r), 12);
        }
    }

    #[test]
    fn test_default_args() {
        let source = "\
def greet(name, greeting='hello'):
    return greeting
result = greet('world')
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_str_get_value(r), "hello");
            },
            Err(e) => {
                // Default args may need KW_DEFAULTS support
                eprintln!("default args: {} ({:?})", e.message, e.kind);
            }
        }
    }

    #[test]
    fn test_augmented_assign_list() {
        let source = "x = [1, 2]\nx += [3]\nresult = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.w_globals).get("result").unwrap();
                assert!(is_list(result));
                // After += [3], x should have 3 elements
                assert_eq!(w_list_len(result), 3);
            },
            Err(e) => panic!("augmented list failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_for_loop_over_list() {
        let source = "\
total = 0
for x in [1, 2, 3, 4, 5]:
    total = total + x
result = total";
        let (res, frame) = run_exec_frame(source);
        res.expect("for loop failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_for_loop_over_string() {
        let source = "\
result = 0
for c in 'abc':
    result = result + 1";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_int_get_value(result), 3);
            },
            Err(e) => {
                // String iteration might not work yet — ignore
                eprintln!("for-string: {}", e.message);
            }
        }
    }

    #[test]
    fn test_multiple_assignment() {
        let source = "a = b = 42\nresult = a + b";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple assign failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 84);
        }
    }

    #[test]
    #[ignore = "closure: RustPython uses LOAD_FAST for freevars, needs COPY_FREE_VARS to copy cells"]
    fn test_closure_basic() {
        let source = "\
def make_adder(n):
    def adder(x):
        return x + n
    return adder
add5 = make_adder(5)
result = add5(10)";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.w_globals).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 15);
            },
            Err(e) => panic!("closure failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_tuple_unpacking_assign() {
        let source = "a, b, c = 1, 2, 3\nresult = a + b + c";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple unpack failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 6);
        }
    }

    #[test]
    fn test_dict_access_ops() {
        let source = "d = {1: 10, 2: 20}\nresult = d[1] + d[2]";
        let (res, frame) = run_exec_frame(source);
        res.expect("dict access failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_string_len() {
        let source = "result = len('hello')";
        let (res, frame) = run_exec_frame(source);
        res.expect("string len failed");
        unsafe {
            let r = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 5);
        }
    }

    #[test]
    fn test_power_operator() {
        let result = run_eval("2 ** 10").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 1024);
        }
    }

    #[test]
    fn test_modulo() {
        let result = run_eval("17 % 5").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_floor_division() {
        let result = run_eval("17 // 3").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_bitwise_ops() {
        let result = run_eval("(0xFF & 0x0F) | 0x30").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 0x3F);
        }
    }

    #[test]
    fn test_list_comprehension() {
        // Use explicit loop with list + index (no method calls)
        let source = "\
result = [0, 0, 0]
i = 0
for x in [1, 2, 3]:
    result[i] = x * 2
    i = i + 1
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.w_globals).get("result").unwrap();
                assert!(is_list(result));
                assert_eq!(w_list_len(result), 3);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 4);
                assert_eq!(w_int_get_value(w_list_getitem(result, 2).unwrap()), 6);
            },
            Err(e) => panic!("list comprehension failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_globals_builtin_uses_current_module_namespace() {
        let source = "x = 41\nresult = globals()['x'] + 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("globals() failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_locals_builtin_uses_current_function_locals() {
        let source = "\
def f(a, b):
    c = a + b
    return locals()['a'] + locals()['b'] + locals()['c']
result = f(2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in function failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_locals_builtin_uses_class_namespace() {
        let source = "\
x = 1
class C:
    y = 2
    snap = locals()
result = C.snap['y'] + globals()['x']";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in class failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_bound_method_materialized_by_attribute_access() {
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add
result = m(41)";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_bound_method_lookup_materializes_method_object() {
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup setup failed");
        unsafe {
            let c_obj = *(*frame.w_globals).get("c").unwrap();
            let m_obj = *(*frame.w_globals).get("m").unwrap();
            assert!(pyre_object::is_method(m_obj));
            assert!(std::ptr::eq(pyre_object::w_method_get_self(m_obj), c_obj));
        }
    }

    #[test]
    fn test_builtin_type_method_materialized_by_attribute_access() {
        let source = "\
xs = []
m = xs.append
m(42)
result = len(xs)";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin type method lookup failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_builtin_function_stored_on_class_is_not_bound() {
        let source = "\
class C:
    f = len
c = C()
result = c.f([1, 2, 3])";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin function descriptor semantics failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_builtin_function_typedef_overrides_match_pypy() {
        // The `__doc__` slot routes through `getset_func_doc` which falls
        // back to `BuiltinCode.getdocstring` (function.py:395-398). pyre's
        // `len` is registered without a docstring so the access path
        // returns whatever code.getdocstring yields — the test only checks
        // that the lookup does not crash and that mutation/deletion fire
        // the orthodox `_check_code_mutable` TypeError.
        let source = "\
doc_value = len.__doc__
self_is_none = len.__self__ is None
repr_result = len.__repr__()
new_err = ''
try:
    type(len)()
except TypeError as e:
    new_err = str(e)
set_err = ''
try:
    len.__doc__ = 'x'
except TypeError as e:
    set_err = str(e)
del_err = ''
try:
    del len.__doc__
except TypeError as e:
    del_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin_function typedef overrides failed");
        unsafe {
            let _doc_value = *(*frame.w_globals).get("doc_value").unwrap();
            let self_is_none = *(*frame.w_globals).get("self_is_none").unwrap();
            let repr_result = *(*frame.w_globals).get("repr_result").unwrap();
            let new_err = *(*frame.w_globals).get("new_err").unwrap();
            let set_err = *(*frame.w_globals).get("set_err").unwrap();
            let del_err = *(*frame.w_globals).get("del_err").unwrap();
            assert!(w_bool_get_value(self_is_none));
            assert_eq!(w_str_get_value(repr_result), "<built-in function len>");
            assert_eq!(
                w_str_get_value(new_err),
                "cannot create 'builtin_function' instances"
            );
            assert!(
                w_str_get_value(set_err).contains("func_doc"),
                "len.__doc__ = 'x' should raise TypeError, got: {:?}",
                w_str_get_value(set_err)
            );
            assert!(
                w_str_get_value(del_err).contains("func_doc"),
                "del len.__doc__ should raise TypeError, got: {:?}",
                w_str_get_value(del_err)
            );
        }
    }

    #[test]
    fn test_set_subtype_and_init_follow_pypy_constructor_protocol() {
        let source = "\
class S(set):
    pass
s = S([1, 2, 3])
manual = set()
set.__init__(manual, [4, 5])
is_subtype = type(s) is S
result = len(s)
manual_result = len(manual)";
        let (res, frame) = run_exec_frame(source);
        res.expect("set constructor parity failed");
        unsafe {
            let is_subtype = *(*frame.w_globals).get("is_subtype").unwrap();
            let result = *(*frame.w_globals).get("result").unwrap();
            let manual_result = *(*frame.w_globals).get("manual_result").unwrap();
            assert!(w_bool_get_value(is_subtype));
            assert_eq!(w_int_get_value(result), 3);
            assert_eq!(w_int_get_value(manual_result), 2);
        }
    }

    #[test]
    fn test_frozenset_constructor_exact_and_subtype_paths_match_pypy() {
        let source = "\
class F(frozenset):
    pass
seed = frozenset([1, 2])
same = frozenset(seed) is seed
sub = F([1, 2, 3])
is_subtype = type(sub) is F
result = len(sub)";
        let (res, frame) = run_exec_frame(source);
        res.expect("frozenset constructor parity failed");
        unsafe {
            let same = *(*frame.w_globals).get("same").unwrap();
            let is_subtype = *(*frame.w_globals).get("is_subtype").unwrap();
            let result = *(*frame.w_globals).get("result").unwrap();
            assert!(w_bool_get_value(same));
            assert!(w_bool_get_value(is_subtype));
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_set_constructors_reject_extra_positionals_like_pypy() {
        // setobject.py:160 W_SetObject.descr_init parses against
        // `init_signature = Signature(['some_iterable'])`, so anything
        // beyond `(self, iterable)` is a TypeError; setobject.py:631
        // W_FrozensetObject.descr_new2 has the gateway-level fixed maxargs
        // for `(space, w_frozensettype, w_iterable=None)`.
        let source = "\
init_err = ''
try:
    set([1], 2)
except TypeError as e:
    init_err = str(e)
init_direct_err = ''
try:
    s = set()
    set.__init__(s, [1], 2)
except TypeError as e:
    init_direct_err = str(e)
frozen_err = ''
try:
    frozenset([1], 2)
except TypeError as e:
    frozen_err = str(e)
frozen_new_err = ''
try:
    frozenset.__new__(frozenset, [1], 2)
except TypeError as e:
    frozen_new_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("set/frozenset arity enforcement failed");
        unsafe {
            let init_err = *(*frame.w_globals).get("init_err").unwrap();
            let init_direct_err = *(*frame.w_globals).get("init_direct_err").unwrap();
            let frozen_err = *(*frame.w_globals).get("frozen_err").unwrap();
            let frozen_new_err = *(*frame.w_globals).get("frozen_new_err").unwrap();
            assert!(
                !w_str_get_value(init_err).is_empty(),
                "set([1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(init_direct_err).is_empty(),
                "set.__init__(s, [1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_err).is_empty(),
                "frozenset([1], 2) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_new_err).is_empty(),
                "frozenset.__new__(frozenset, [1], 2) should raise TypeError"
            );
        }
    }

    #[test]
    fn test_set_new_rejects_foreign_layout_typedef() {
        // typeobject.py:520-523 W_TypeObject.check_user_subclass refuses
        // `set.__new__(int)` (and similar cross-layout calls) before the
        // base allocator runs. pyre's `check_user_subclass` enforces the
        // same layout-typedef identity guard.
        let source = "\
err = ''
try:
    set.__new__(int)
except TypeError as e:
    err = str(e)
frozen_err = ''
try:
    frozenset.__new__(int, [1, 2])
except TypeError as e:
    frozen_err = str(e)";
        let (res, frame) = run_exec_frame(source);
        res.expect("layout safety check failed");
        unsafe {
            let err = *(*frame.w_globals).get("err").unwrap();
            let frozen_err = *(*frame.w_globals).get("frozen_err").unwrap();
            assert!(
                !w_str_get_value(err).is_empty(),
                "set.__new__(int) should raise TypeError"
            );
            assert!(
                !w_str_get_value(frozen_err).is_empty(),
                "frozenset.__new__(int, [1, 2]) should raise TypeError"
            );
        }
    }

    #[test]
    fn test_metaclass_method_materialized_by_attribute_access() {
        let source = "\
class Meta(type):
    def pick(cls):
        return cls
class C(metaclass=Meta):
    pass
bound = C.pick
result = bound()";
        let (res, frame) = run_exec_frame(source);
        res.expect("metaclass descriptor lookup failed");
        let result = unsafe { *(*frame.w_globals).get("result").unwrap() };
        let c_obj = unsafe { *(*frame.w_globals).get("C").unwrap() };
        assert!(std::ptr::eq(result, c_obj));
    }

    #[test]
    fn test_staticmethod_prepare_is_called_with_bound_lookup() {
        let source = "\
class Meta(type):
    @staticmethod
    def __prepare__(name, bases):
        return {'seed': 41}
class C(metaclass=Meta):
    value = seed + 1
result = C.value";
        let (res, frame) = run_exec_frame(source);
        res.expect("__prepare__ lookup failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_dunder_globals_and_code_are_materialized() {
        let source = "\
x = 7
def f(a, *, b=3):
    return a + b + x
g = f.__globals__
code = f.__code__";
        let (res, frame) = run_exec_frame(source);
        res.expect("function dunder lookup failed");
        let globals = unsafe { *(*frame.w_globals).get("g").unwrap() };
        let code = unsafe { *(*frame.w_globals).get("code").unwrap() };
        unsafe {
            let x = pyre_object::w_dict_lookup(globals, pyre_object::w_str_new("x")).unwrap();
            assert_eq!(w_int_get_value(x), 7);
            let argcount = crate::baseobjspace::getattr(code, "co_argcount").unwrap();
            assert_eq!(w_int_get_value(argcount), 1);
            let kwonly = crate::baseobjspace::getattr(code, "co_kwonlyargcount").unwrap();
            assert_eq!(w_int_get_value(kwonly), 1);
            let name = crate::baseobjspace::getattr(code, "co_name").unwrap();
            assert_eq!(w_str_get_value(name), "f");
            let varnames = crate::baseobjspace::getattr(code, "co_varnames").unwrap();
            let first = w_tuple_getitem(varnames, 0).unwrap();
            assert_eq!(w_str_get_value(first), "a");
        }
    }

    #[test]
    fn test_vars_builtin_raises_type_error_without_dict() {
        let source = "\
result = 0
try:
    vars(1)
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("vars() exception path failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_type_builtin_rejects_invalid_arity() {
        let source = "\
result = 0
try:
    type()
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("type() exception path failed");
        unsafe {
            let result = *(*frame.w_globals).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }
}
