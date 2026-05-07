//! Function call dispatch — pure interpreter, no JIT dependencies.
//!
//! JIT-specific call infrastructure (force/bridge callbacks, callee frame
//! creation helpers, frame pool) lives in pyre-jit/src/call_jit.rs.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use crate::{
    DictStorage, PyError, PyResult, builtin_code_get, dispatch_callable, function_get_closure,
    function_get_globals,
};

thread_local! {
    /// Most recent error swallowed by `call_function_impl` /
    /// `call_user_function_with_args`. These functions return a bare
    /// `PyObjectRef` for legacy reasons; when the underlying call raises,
    /// they stash the error here and return PY_NULL / w_none() so that
    /// callers that need the real error can pull it back out via
    /// `take_call_error()`.
    ///
    /// Pattern is the inverse of CPython's `PyErr_Occurred()` — see
    /// `pyerrors.c`. Same idea: a thread-local error indicator paired with
    /// out-of-band NULL returns.
    static PENDING_CALL_ERROR: RefCell<Option<PyError>> = const { RefCell::new(None) };
}

/// Stash an error from the bare-PyObjectRef call path so a caller that
/// recognizes the NULL return can recover the original PyError.
pub fn set_call_error(e: PyError) {
    PENDING_CALL_ERROR.with(|slot| {
        *slot.borrow_mut() = Some(e);
    });
}

/// Take and clear the most recent stashed call error. Returns None if no
/// error is pending. Callers must pair this with the bare-return call
/// helpers (`call_function_impl`, `call_function_impl_raw`,
/// `call_user_function_with_args`) immediately after the call so the
/// error refers to the most recent failed dispatch.
pub fn take_call_error() -> Option<PyError> {
    PENDING_CALL_ERROR.with(|slot| slot.borrow_mut().take())
}

/// Clear any pending stashed error without consuming it.
pub fn clear_call_error() {
    PENDING_CALL_ERROR.with(|slot| {
        slot.borrow_mut().take();
    });
}
use pyre_object::{PY_NULL, PyObjectRef};

use crate::eval::eval_frame_plain;
use crate::pyframe::PyFrame;

// ── Eval function injection ──────────────────────────────────────
type EvalFn = fn(&mut PyFrame) -> PyResult;
static EVAL_OVERRIDE: OnceLock<EvalFn> = OnceLock::new();

type DepthBumpFn = fn() -> Option<Box<dyn std::any::Any>>;
static DEPTH_BUMP_OVERRIDE: OnceLock<DepthBumpFn> = OnceLock::new();

thread_local! {
    /// Call depth counter — incremented on every user function call,
    /// decremented on return. Replaces the Box<dyn Any> depth bump
    /// callback with a zero-allocation TLS increment.
    static CALL_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Get current call depth. Used by pyre-jit for JIT_CALL_DEPTH parity.
#[inline(always)]
pub fn call_depth() -> u32 {
    CALL_DEPTH.with(|d| d.get())
}

/// RAII guard that decrements CALL_DEPTH on drop.
struct CallDepthGuard;
impl Drop for CallDepthGuard {
    #[inline(always)]
    fn drop(&mut self) {
        CALL_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Register the JIT-aware eval function. Called by pyre-jit at startup.
pub fn register_eval_override(f: EvalFn) {
    let _ = EVAL_OVERRIDE.set(f);
}

// ── JIT parameter injection ──────────────────────────────────────
//
// `pypy/interpreter/executioncontext.py:296-298 settrace` invokes
// `jit.set_param(None, 'trace_limit', 10000)` on the global default
// jitdriver to widen the trace budget while a tracefunc is installed.
// pyre-interpreter cannot import pyre-jit (its lower-layer crate), so
// the JIT side registers a hook at boot that pyre-interpreter calls
// through.  Mirrors the `EVAL_OVERRIDE` pattern above.
type SetJitParamFn = fn(name: &str, value: i64);
static SET_JIT_PARAM_HOOK: OnceLock<SetJitParamFn> = OnceLock::new();

/// Register the hook that forwards `set_jit_param` calls into the JIT
/// runtime's `WarmState::set_param`. Called by pyre-jit at startup.
pub fn register_set_jit_param_hook(f: SetJitParamFn) {
    let _ = SET_JIT_PARAM_HOOK.set(f);
}

/// `rlib/jit.py:818 jit.set_param(driver=None, name, value)` analogue.
/// No-op when pyre-jit has not registered the hook (e.g. JIT-disabled
/// builds or boot-time callers that fire before the first `eval_with_jit`
/// invocation).
pub fn set_jit_param(name: &str, value: i64) {
    if let Some(hook) = SET_JIT_PARAM_HOOK.get() {
        hook(name, value);
    }
}

thread_local! {
    static FORCE_PLAIN_EVAL: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Last known valid execution context — for call_user_function_with_args.
    static LAST_EXEC_CTX: std::cell::Cell<*const crate::PyExecutionContext> =
        const { std::cell::Cell::new(std::ptr::null()) };
}

/// Set the last known execution context (called at eval loop entry).
pub fn set_last_exec_ctx(ctx: *const crate::PyExecutionContext) {
    LAST_EXEC_CTX.with(|c| c.set(ctx));
}

/// Snapshot the current thread-local execution context. Residual callers
/// that need to temporarily pin a different context (blackhole's
/// `bh_call_fn_impl` cold path, for example) pair this with
/// `set_last_exec_ctx` to restore the prior value on return.
pub fn take_last_exec_ctx() -> *const crate::PyExecutionContext {
    LAST_EXEC_CTX.with(|c| c.get())
}

/// `pypy/objspace/std/objspace.py space.getexecutioncontext()` analogue.
///
/// PyPy walks thread state and returns the live `ExecutionContext`,
/// creating one on demand.  Pyre stores the active context in a TLS
/// slot seeded at process boot by pyrex (`pyrex/src/lib.rs:185
/// set_last_exec_ctx(Rc::as_ptr(&execution_context))`) and
/// re-stamped on every `eval_frame_plain` entry.  The slot stays
/// pointing at the root EC for the lifetime of the process, so
/// `sys.gettrace`/`settrace`/`getprofile`/`setprofile` and other
/// `space.getexecutioncontext()` callers see the live EC even when
/// no eval frame is currently on the stack.
///
/// PRE-EXISTING-ADAPTATION: pyre is single-threaded today so the TLS
/// slot is effectively a global.  PyPy's per-thread `threadlocals`
/// dispatch lands when pyre adds its own thread state container.
pub fn getexecutioncontext() -> *const crate::PyExecutionContext {
    take_last_exec_ctx()
}

/// Guard that temporarily forces all nested calls to use the plain
/// interpreter, bypassing eval_with_jit. Used by force_fn to avoid
/// re-entering compiled code from blackhole execution.
pub struct ForcePlainEvalGuard;

impl Drop for ForcePlainEvalGuard {
    fn drop(&mut self) {
        let _ = FORCE_PLAIN_EVAL.try_with(|c| c.set(c.get().saturating_sub(1)));
    }
}

/// Check if force-plain-eval mode is active.
pub fn is_force_plain_eval() -> bool {
    FORCE_PLAIN_EVAL.with(|c| c.get() > 0)
}

/// Enter "force plain eval" mode. While active, `call_user_function` uses
/// `eval_frame_plain` instead of the JIT-aware eval override.
pub fn force_plain_eval() -> ForcePlainEvalGuard {
    FORCE_PLAIN_EVAL.with(|c| c.set(c.get() + 1));
    ForcePlainEvalGuard
}

/// Register the JIT call-depth bump function. Called by pyre-jit at startup.
pub fn register_depth_bump(f: DepthBumpFn) {
    let _ = DEPTH_BUMP_OVERRIDE.set(f);
}

fn call_user_function_with_eval(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
    eval_fn: EvalFn,
) -> PyResult {
    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let defaults = unsafe { crate::function_get_defaults(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // PyPy: pyframe.py handle_operation_error / init_cells
    // Fill missing positional args from defaults tuple.
    let code_ref = unsafe { &*func_code };
    let nparams = code_ref.arg_count as usize;
    let nargs = args.len();
    let filled_args = if nargs < nparams && !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        let mut full = Vec::with_capacity(nparams);
        full.extend_from_slice(args);
        // Defaults cover the LAST (nparams - first_default) parameters.
        // number of defaults = tuple length
        let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
            unsafe { pyre_object::w_tuple_len(defaults) }
        } else {
            0
        };
        let first_default = nparams - ndefaults;
        for i in nargs..nparams {
            if i >= first_default {
                let default_idx = i - first_default;
                if let Some(val) =
                    unsafe { pyre_object::w_tuple_getitem(defaults, default_idx as i64) }
                {
                    full.push(val);
                } else {
                    full.push(pyre_object::PY_NULL);
                }
            } else {
                full.push(pyre_object::PY_NULL);
            }
        }
        full
    } else {
        args.to_vec()
    };

    // Fill keyword-only defaults from kwdefaults dict
    let nkwonly = code_ref.kwonlyarg_count as usize;
    let mut filled_args = filled_args;
    if nkwonly > 0 {
        let kwdefaults = unsafe { crate::function_get_kwdefaults(callable) };
        // Ensure filled_args covers all positional + kwonly slots
        while filled_args.len() < nparams + nkwonly {
            filled_args.push(pyre_object::PY_NULL);
        }
        if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
            for ki in 0..nkwonly {
                let slot = nparams + ki;
                if filled_args[slot].is_null() {
                    let param_name = &code_ref.varnames[slot];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                        filled_args[slot] = val;
                    }
                }
            }
        }
    }

    let final_args = pack_varargs(code_ref, filled_args);

    // Generator function: create generator object instead of executing.
    // PyPy: generator.py GeneratorIterator.__init__ wraps PyFrame.
    // RustPython compiler uses CodeFlags::GENERATOR instead of RETURN_GENERATOR opcode.
    if code_ref
        .flags
        .intersects(crate::CodeFlags::GENERATOR | crate::CodeFlags::COROUTINE)
    {
        let mut gen_frame = PyFrame::new_for_call_with_closure(
            w_code,
            &final_args,
            globals,
            frame.execution_context,
            closure,
        );
        gen_frame.fix_array_ptrs();
        return gen_frame.run();
    }

    let mut func_frame = PyFrame::new_for_call_with_closure(
        w_code,
        &final_args,
        globals,
        frame.execution_context,
        closure,
    );
    func_frame.fix_array_ptrs();
    eval_fn(&mut func_frame)
}

/// Call a user function with pre-resolved args (scope already packed by
/// resolve_kwargs). Skips defaults-fill and pack_varargs — the caller
/// (call_kw) already produced the final scope via resolve_kwargs which
/// mirrors PyPy's Arguments.parse_into_scope.
pub fn call_user_function_resolved(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    CALL_DEPTH.with(|d| d.set(d.get() + 1));
    let _depth_guard = CallDepthGuard;

    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let code_ref = unsafe { &*func_code };

    // Generator function
    if code_ref
        .flags
        .intersects(crate::CodeFlags::GENERATOR | crate::CodeFlags::COROUTINE)
    {
        let mut gen_frame = PyFrame::new_for_call_with_closure(
            w_code,
            args,
            globals,
            frame.execution_context,
            closure,
        );
        gen_frame.fix_array_ptrs();
        return gen_frame.run();
    }

    let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
    let eval_fn = if plain_mode {
        eval_frame_plain
    } else {
        EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain)
    };

    let mut func_frame =
        PyFrame::new_for_call_with_closure(w_code, args, globals, frame.execution_context, closure);
    func_frame.fix_array_ptrs();
    eval_fn(&mut func_frame)
}

pub fn call_callable(frame: &mut PyFrame, callable: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    let callable = crate::baseobjspace::unwrap_cell(callable);
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable);
            if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            }
        };
        let mut call_args = Vec::with_capacity(1 + args.len());
        if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
            call_args.push(receiver);
        }
        call_args.extend_from_slice(args);
        return call_callable(frame, func, &call_args);
    }
    if unsafe { pyre_object::is_type(callable) } {
        return type_descr_call(frame, callable, args);
    }

    // staticmethod → unwrap
    // PyPy: function.py StaticMethod.descr_staticmethod__call__
    if unsafe { pyre_object::is_staticmethod(callable) } {
        let func = unsafe { pyre_object::w_staticmethod_get_func(callable) };
        return call_callable(frame, func, args);
    }
    // classmethod → unwrap
    if unsafe { pyre_object::is_classmethod(callable) } {
        let func = unsafe { pyre_object::w_classmethod_get_func(callable) };
        return call_callable(frame, func, args);
    }

    // Instance with __call__ — PyPy: descroperation.py descr_call
    if unsafe { pyre_object::is_instance(callable) } {
        let w_type = unsafe { pyre_object::w_instance_get_type(callable) };
        if let Some(call_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") } {
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(callable);
            call_args.extend_from_slice(args);
            return call_callable(frame, call_fn, &call_args);
        }
    }

    let frame_ptr = frame as *mut PyFrame;
    dispatch_callable(
        callable,
        |callable| {
            // baseobjspace.py:1243 — `if frame.get_is_being_profiled() and
            // is_builtin_code(w_func): ... return self.call_args_and_c_profile(...)`
            // The `is_builtin_code(w_func)` check is structurally implicit
            // here: dispatch_callable already routed via the builtin arm
            // (runtime_ops.rs:275 `if is_builtin_code(code) { on_builtin }`),
            // so reaching this closure means the callable is a builtin.
            // The remaining condition is the per-frame profile flag, set
            // by `ec.call_trace` (executioncontext.py:150) on frame entry
            // and cleared by `_c_call_return_trace` when profilefunc was
            // turned off (executioncontext.py:122-123).
            let profile_active = unsafe { (*frame_ptr).get_is_being_profiled() };
            if profile_active {
                let w_res = crate::baseobjspace::call_args_and_c_profile(
                    unsafe { &mut *frame_ptr },
                    callable,
                    args,
                );
                if w_res == pyre_object::PY_NULL {
                    return Err(take_call_error()
                        .unwrap_or_else(|| crate::PyError::value_error("call failed")));
                }
                return Ok(w_res);
            }
            let code = unsafe { crate::getcode(callable) };
            let func = unsafe { builtin_code_get(code as pyre_object::PyObjectRef) };
            func(args)
        },
        |callable| call_user_function(frame, callable, args),
    )
}

pub fn call_user_function(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    // Direct TLS increment — no Box allocation. Replaces DEPTH_BUMP_OVERRIDE callback.
    CALL_DEPTH.with(|d| d.set(d.get() + 1));
    let _depth_guard = CallDepthGuard;
    let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
    let eval_fn = if plain_mode {
        eval_frame_plain
    } else {
        EVAL_OVERRIDE.get().copied().unwrap_or(eval_frame_plain)
    };
    call_user_function_with_eval(frame, callable, args, eval_fn)
}

/// Plain interpreter-only user-function call.
///
/// JIT residual helpers should use this instead of the injected eval override.
/// PyPy residual calls are opaque slow paths; they should not accidentally
/// re-enter the caller's portal/tracing state.
pub fn call_user_function_plain(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    call_user_function_with_eval(frame, callable, args, eval_frame_plain)
}

/// Call a user function with an explicit execution context pointer.
/// Used by MIFrame Box tracking when concrete_frame is unavailable.
pub fn call_user_function_plain_with_ctx(
    execution_context: *const crate::PyExecutionContext,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    let w_code = unsafe { crate::getcode(callable) };
    let globals = unsafe { function_get_globals(callable) };
    let closure = unsafe { function_get_closure(callable) };
    let mut func_frame =
        PyFrame::new_for_call_with_closure(w_code, args, globals, execution_context, closure);
    func_frame.fix_array_ptrs();
    // pyframe.py:268-273 run — wrap as generator/coroutine when the code
    // carries CO_GENERATOR / CO_COROUTINE / CO_ASYNC_GENERATOR.  The
    // generator-flag dispatch site for the normal call path lives at
    // call.rs:1260; this MIFrame Box tracking entry must mirror it so
    // calls dispatched without a concrete_frame produce the same result
    // as calls dispatched with one.
    func_frame.run()
}

/// Explicit residual-call protocol used by JIT inline framestack concrete
/// execution.
///
/// PyPy treats residual calls reached from inline execution as opaque slow
/// paths. They must not accidentally reuse the generic JIT-aware
/// `call_user_function()` entry, because that can re-enter portal state that
/// belongs to the outer trace instead of the active inline framestack.
pub fn call_callable_inline_residual(
    frame: &PyFrame,
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> PyResult {
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe {
            let w_self = pyre_object::w_method_get_self(callable);
            if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            }
        };
        let mut call_args = Vec::with_capacity(1 + args.len());
        if !receiver.is_null() && unsafe { !pyre_object::is_none(receiver) } {
            call_args.push(receiver);
        }
        call_args.extend_from_slice(args);
        return call_callable_inline_residual(frame, func, &call_args);
    }
    dispatch_callable(
        callable,
        |callable| {
            let code = unsafe { crate::getcode(callable) };
            let func = unsafe { builtin_code_get(code as pyre_object::PyObjectRef) };
            func(args)
        },
        |callable| call_user_function_plain(frame, callable, args),
    )
}

// ── __build_class__ implementation ───────────────────────────────────
// PyPy equivalent: pyopcode.py BUILD_CLASS
//   1. Execute class body function with fresh namespace (class_locals)
//   2. Create W_TypeObject from the harvested namespace

/// Initialize interpreter callbacks and type registry.
///
/// PyPy: setup_builtin_modules / make_builtins — called once at startup.
/// Resolve keyword arguments into positional order.
///
/// PyPy: argument.py `_match_signature` + `_match_keywords`
///
/// Given:
///   - callable: function with code.varnames defining parameter names
///   - args: [positional_args..., kwarg_values...] (mixed)
///   - kwarg_names: tuple of str names for the last N args
///
/// Returns args rearranged so that keyword values are in the correct
/// parameter positions. This runs BEFORE frame creation so the JIT
/// eval loop sees correctly-positioned locals.
pub(crate) fn resolve_kwargs(
    callable: PyObjectRef,
    args: &[PyObjectRef],
    kwarg_names: PyObjectRef,
) -> Vec<PyObjectRef> {
    if kwarg_names.is_null() {
        return args.to_vec();
    }
    let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
        unsafe { pyre_object::w_tuple_len(kwarg_names) }
    } else {
        return args.to_vec();
    };
    if nkw == 0 {
        return args.to_vec();
    }

    // Resolve the target function's code object.
    // For user functions: direct code_ptr.
    // For type objects: look up __new__ in MRO (PyPy: Arguments used by descr_call).
    //
    // When callable is a type, type_descr_call will prepend `cls` as the first
    // arg to __new__, so the stack args correspond to __new__'s params[1:]
    // (skip_cls=1). For plain function calls skip_cls=0.
    let (target_func, skip_cls) = if unsafe { crate::is_function(callable) } {
        (callable, 0usize)
    } else if unsafe { pyre_object::is_type(callable) } {
        // For type objects, resolve kwargs against __init__ first (most
        // common case: user classes accept kwargs in __init__), falling
        // back to __new__ (e.g. immutable types, metaclasses).
        // PyPy: typeobject.py descr_call → Arguments._match_signature
        //   resolves against the winning __init__ or __new__.
        let init_fn = unsafe { crate::baseobjspace::lookup_in_type(callable, "__init__") };
        if let Some(init_fn) = init_fn {
            if unsafe { crate::is_function(init_fn) } {
                (init_fn, 1usize) // __init__(self, ...) → skip self
            } else {
                // __init__ is builtin → try __new__
                let bases_arg = if args.len() >= nkw + 2 {
                    args[1]
                } else {
                    pyre_object::PY_NULL
                };
                let w_winner = calculate_metaclass(callable, bases_arg).unwrap_or(callable);
                if let Some(new_fn) =
                    unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") }
                {
                    if unsafe { crate::is_function(new_fn) } {
                        (new_fn, 1usize)
                    } else {
                        return args.to_vec();
                    }
                } else {
                    return args.to_vec();
                }
            }
        } else {
            return args.to_vec();
        }
    } else {
        return args.to_vec();
    };

    let code_ptr = unsafe { crate::get_pycode(target_func) };
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    // Total named params = positional + keyword-only
    let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
    // Effective params = params visible to the caller (excludes implicit cls for types)
    let nparams = total_params - skip_cls;
    let n_pos = args.len() - nkw; // number of positional args

    // Start with PY_NULL for all effective params
    let mut result = vec![pyre_object::PY_NULL; nparams];

    // Fill positional args (PyPy: _match_signature step 1)
    for i in 0..n_pos.min(nparams) {
        result[i] = args[i];
    }

    // Match keywords to parameter names (PyPy: _match_keywords)
    // varnames[skip_cls..total_params] are the effective param names
    let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);
    let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
    let mut extra_kwargs: Vec<(PyObjectRef, PyObjectRef)> = Vec::new();
    for ki in 0..nkw {
        let kw_name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
        let Some(kw_name_obj) = kw_name else { continue };
        let kw_str = unsafe { pyre_object::w_str_get_value(kw_name_obj) };
        let kw_value = args[n_pos + ki];

        let mut matched = false;
        for pi in 0..nparams {
            if &*code.varnames[skip_cls + pi] == kw_str {
                result[pi] = kw_value;
                matched = true;
                break;
            }
        }
        if !matched && has_varkw {
            extra_kwargs.push((kw_name_obj, kw_value));
        }
    }

    // Fill positional defaults (PyPy: _match_signature defs_w)
    // Defaults cover the LAST N of the positional params (arg_count).
    let n_pos_params = code.arg_count as usize - skip_cls;
    let defaults = unsafe { crate::function_get_defaults(target_func) };
    if !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        if unsafe { pyre_object::is_tuple(defaults) } {
            let ndefaults = unsafe { pyre_object::w_tuple_len(defaults) };
            let first_default = n_pos_params.saturating_sub(ndefaults);
            for pi in first_default..n_pos_params {
                if result[pi].is_null() {
                    let di = pi - first_default;
                    if let Some(v) = unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) } {
                        result[pi] = v;
                    }
                }
            }
        }
    }

    // Fill keyword-only defaults from kwdefaults dict
    // PyPy: _match_signature fills from w_kw_defs
    let kwdefaults = unsafe { crate::function_get_kwdefaults(target_func) };
    if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
        let nkwonly = code.kwonlyarg_count as usize;
        for ki in 0..nkwonly {
            let pi = n_pos_params + ki; // position in result
            if result[pi].is_null() {
                let param_name = &code.varnames[skip_cls + pi];
                let key = pyre_object::w_str_new(param_name);
                if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                    result[pi] = val;
                }
            }
        }
    }

    // Pack *args and **kwargs into scope — PyPy _match_signature lines 207-259.
    // This produces the final scope_w that maps directly to frame locals.
    if has_varargs {
        let extra_pos: Vec<PyObjectRef> = if n_pos > nparams {
            args[nparams..n_pos].to_vec()
        } else {
            vec![]
        };
        result.push(pyre_object::w_tuple_new(extra_pos));
    }
    if has_varkw {
        let kw_dict = pyre_object::w_dict_new();
        for (key, value) in &extra_kwargs {
            unsafe {
                pyre_object::w_dict_store(kw_dict, *key, *value);
            }
        }
        result.push(kw_dict);
    }

    result
}

/// Call a user function with positional args + keyword args from a dict.
///
/// PyPy: argument.py Arguments._match_signature with keyword handling.
/// Used by CALL_FUNCTION_KW / CALL_KW and CALL_FUNCTION_EX when kwargs
/// are non-empty.
pub fn call_with_kwargs(
    frame: &mut crate::pyframe::PyFrame,
    callable: PyObjectRef,
    pos_args: &[PyObjectRef],
    kwargs: &[(String, PyObjectRef)],
) -> PyResult {
    let callable = crate::baseobjspace::unwrap_cell(callable);

    // Unwrap bound methods: prepend receiver to pos_args.
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let receiver = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !receiver.is_null() && !unsafe { pyre_object::is_none(receiver) } {
            full_args.push(receiver);
        }
        full_args.extend_from_slice(pos_args);
        return call_with_kwargs(frame, func, &full_args, kwargs);
    }

    if unsafe { crate::is_function(callable) } {
        let code = unsafe { crate::getcode(callable) };
        // For builtins: pack kwargs into a dict as last arg.
        //
        // PyPy keeps keyword_names_w / keywords_w on the live Arguments object
        // and gateway builtins parse that object.  Pyre's builtin ABI is still
        // the older flat slice, so the dict tail is a structural adaptation.
        // Keep the __pyre_kw__ marker here, in the one builtin kwargs packing
        // site, so CALL_KW and CALL_FUNCTION_EX have the same shape.
        if unsafe { crate::is_builtin_code(code as pyre_object::PyObjectRef) } {
            let mut full_args = pos_args.to_vec();
            if !kwargs.is_empty() {
                let kwargs_dict = pyre_object::w_dict_new();
                unsafe {
                    pyre_object::w_dict_store(
                        kwargs_dict,
                        pyre_object::w_str_new("__pyre_kw__"),
                        pyre_object::w_bool_from(true),
                    );
                }
                for (key, value) in kwargs {
                    unsafe {
                        pyre_object::w_dict_store(kwargs_dict, pyre_object::w_str_new(key), *value);
                    }
                }
                full_args.push(kwargs_dict);
                // Step 2 of the Arguments port: when this is a profiled
                // builtin call AND kwargs are present, route through
                // `call_args_and_c_profile_args` with a structured
                // `Arguments::with_kw(pos_args, keyword_names_w,
                // keywords_w)`.  Otherwise `call_args_and_c_profile`
                // (reached via `call_callable`'s on_builtin closure)
                // would build `Arguments::positional_only(full_args)`
                // and surface the trailing kwargs dict at index 0,
                // breaking the FunctionWithFixedCode rebinding's
                // firstarg() (`argument.py:164-168` returns `None`
                // when positional count is zero, not the kwargs dict).
                let frame_ptr = frame as *mut PyFrame;
                let profile_active = unsafe { (*frame_ptr).get_is_being_profiled() };
                if profile_active {
                    let keyword_names_w: Vec<pyre_object::PyObjectRef> = kwargs
                        .iter()
                        .map(|(k, _)| pyre_object::w_str_new(k))
                        .collect();
                    let keywords_w: Vec<pyre_object::PyObjectRef> =
                        kwargs.iter().map(|(_, v)| *v).collect();
                    let arguments = crate::argument::Arguments::with_kw(
                        pos_args,
                        &keyword_names_w,
                        &keywords_w,
                    );
                    let w_res = crate::baseobjspace::call_args_and_c_profile_args(
                        unsafe { &mut *frame_ptr },
                        callable,
                        &arguments,
                        &full_args,
                    );
                    if w_res == pyre_object::PY_NULL {
                        return Err(take_call_error()
                            .unwrap_or_else(|| crate::PyError::value_error("call failed")));
                    }
                    return Ok(w_res);
                }
            }
            return call_callable(frame, callable, &full_args);
        }

        // For user functions: resolve kwargs to parameter slots
        {
            let w_code = unsafe { crate::getcode(callable) };
            let code = unsafe {
                &*(crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                    as *const crate::CodeObject)
            };
            let total_params = (code.arg_count + code.kwonlyarg_count) as usize;
            let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);

            // Build parameter array
            let mut result = vec![pyre_object::PY_NULL; total_params];
            // Fill positional args
            for i in 0..pos_args.len().min(total_params) {
                result[i] = pos_args[i];
            }
            // Match keywords to parameter names
            let mut extra_kwargs: Vec<(String, PyObjectRef)> = Vec::new();
            for (key, value) in kwargs {
                let mut matched = false;
                for pi in 0..total_params {
                    if code.varnames[pi] == *key {
                        result[pi] = *value;
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    extra_kwargs.push((key.clone(), *value));
                }
            }

            // Fill positional defaults from __defaults__ tuple.
            let n_pos_params = code.arg_count as usize;
            let defaults = unsafe { crate::function_get_defaults(callable) };
            if !defaults.is_null() {
                let defaults = crate::baseobjspace::unwrap_cell(defaults);
                if unsafe { pyre_object::is_tuple(defaults) } {
                    let ndefaults = unsafe { pyre_object::w_tuple_len(defaults) };
                    let first_default = n_pos_params.saturating_sub(ndefaults);
                    for pi in first_default..n_pos_params {
                        if result[pi].is_null() {
                            let di = pi - first_default;
                            if let Some(v) =
                                unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) }
                            {
                                result[pi] = v;
                            }
                        }
                    }
                }
            }
            // Fill keyword-only defaults from __kwdefaults__ dict.
            // function.py Function._apply_defaults — kw-only args take their
            // defaults from the kwdefaults dict by name lookup.
            let nkwonly = code.kwonlyarg_count as usize;
            if nkwonly > 0 {
                let kwdefaults = unsafe { crate::function_get_kwdefaults(callable) };
                if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
                    for ki in 0..nkwonly {
                        let slot = n_pos_params + ki;
                        if slot < result.len() && result[slot].is_null() {
                            let param_name = &code.varnames[slot];
                            let key = pyre_object::w_str_new(param_name);
                            if let Some(v) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) }
                            {
                                result[slot] = v;
                            }
                        }
                    }
                }
            }

            // Pack *args and **kwargs
            let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
            let mut final_args = result;
            if has_varargs {
                let extra_pos: Vec<PyObjectRef> = if pos_args.len() > total_params {
                    pos_args[total_params..].to_vec()
                } else {
                    vec![]
                };
                final_args.push(pyre_object::w_tuple_new(extra_pos));
            }
            if has_varkw {
                let kw_dict = pyre_object::w_dict_new();
                for (key, value) in &extra_kwargs {
                    unsafe {
                        pyre_object::w_dict_store(kw_dict, pyre_object::w_str_new(key), *value);
                    }
                }
                final_args.push(kw_dict);
            }

            // Create frame and execute
            let globals = unsafe { function_get_globals(callable) };
            let closure = unsafe { function_get_closure(callable) };
            let mut func_frame = crate::pyframe::PyFrame::new_for_call_with_closure(
                w_code,
                &final_args,
                globals,
                frame.execution_context,
                closure,
            );
            func_frame.fix_array_ptrs();
            let plain_mode = FORCE_PLAIN_EVAL.with(|c| c.get() > 0);
            let eval_fn = if plain_mode {
                crate::eval::eval_frame_plain
            } else {
                EVAL_OVERRIDE
                    .get()
                    .copied()
                    .unwrap_or(crate::eval::eval_frame_plain)
            };
            return eval_fn(&mut func_frame);
        } // end user function branch
    } // end is_function

    // For type objects: allocate via __new__ then call __init__ with kwargs.
    // PyPy: typeobject.py descr_call → __new__ + __init__
    if unsafe { pyre_object::is_type(callable) } {
        // Types with acceptable_as_base_class=false (bool, NoneType) reject kwargs.
        // PyPy: boolobject.py descr_new uses @unwrap_spec (positional only).
        if !kwargs.is_empty()
            && !unsafe { pyre_object::w_type_get_acceptable_as_base_class(callable) }
        {
            let type_name = unsafe { pyre_object::w_type_get_name(callable) };
            return Err(crate::PyError::type_error(format!(
                "{}() takes no keyword arguments",
                type_name,
            )));
        }
        // Calculate the winning metaclass from bases.
        // type(name, bases, dict, **kw) needs to find the correct metaclass
        // and call its __new__ with the kwargs.
        let w_metaclass = if pos_args.len() >= 2 && unsafe { pyre_object::is_tuple(pos_args[1]) } {
            calculate_metaclass(callable, pos_args[1]).unwrap_or(callable)
        } else {
            callable
        };
        // Step 1: __new__(cls, *args, **kwargs)
        let instance = if let Some(new_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_metaclass, "__new__") }
        {
            let mut new_args = Vec::with_capacity(1 + pos_args.len());
            new_args.push(w_metaclass);
            new_args.extend_from_slice(pos_args);
            if unsafe { crate::is_function(new_fn) } && !kwargs.is_empty() {
                call_with_kwargs(frame, new_fn, &new_args, kwargs)?
            } else {
                call_callable(frame, new_fn, &new_args)?
            }
        } else {
            pyre_object::w_instance_new(callable)
        };
        // Step 2: __init__(self, *args, **kwargs) with full kwargs support.
        if let Some(w_insttype) = type_call_init_type(instance, callable) {
            if let Some(init_fn) =
                unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
            {
                let mut init_args = Vec::with_capacity(1 + pos_args.len());
                init_args.push(instance);
                init_args.extend_from_slice(pos_args);
                if unsafe { crate::is_function(init_fn) } && !kwargs.is_empty() {
                    call_with_kwargs(frame, init_fn, &init_args, kwargs)?;
                } else {
                    call_callable(frame, init_fn, &init_args)?;
                }
            }
        }
        return Ok(instance);
    }

    // For methods: unwrap and retry
    if unsafe { pyre_object::is_method(callable) } {
        let func = unsafe { pyre_object::w_method_get_func(callable) };
        let w_self = unsafe { pyre_object::w_method_get_self(callable) };
        let mut full_args = Vec::with_capacity(1 + pos_args.len());
        if !w_self.is_null() && unsafe { !pyre_object::is_none(w_self) } {
            full_args.push(w_self);
        }
        full_args.extend_from_slice(pos_args);
        return call_with_kwargs(frame, func, &full_args, kwargs);
    }

    // For instances with __call__: dispatch
    if unsafe { pyre_object::is_instance(callable) } {
        let w_type = unsafe { pyre_object::w_instance_get_type(callable) };
        if let Some(call_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__call__") } {
            let mut call_args = Vec::with_capacity(1 + pos_args.len());
            call_args.push(callable);
            call_args.extend_from_slice(pos_args);
            return call_with_kwargs(frame, call_fn, &call_args, kwargs);
        }
    }

    // Fallback: call_callable with positional args only
    call_callable(frame, callable, pos_args)
}

pub fn register_build_class() {
    crate::typedef::init_typeobjects();
    install_dict_storage_hooks();
}

/// Wire the storage ↔ W_DictObject sync hooks.  Idempotent — guarded
/// by an internal `Once` so repeated invocations
/// (`register_build_class` at runtime startup, `ExecutionContext::new`
/// defensive registration before module allocation) collapse to a
/// single registration without leaking additional function pointers
/// per call.  PyPy's single `W_DictMultiObject` owns both halves of
/// the dict view; pyre's split storage / W_DictObject layout would
/// otherwise let an early `w_dict_setitem_str` (e.g.
/// `Module.__init__`'s `__name__` write) silently miss the storage if
/// these hooks are not yet registered.
pub fn install_dict_storage_hooks() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        pyre_object::dictobject::register_dict_storage_store_hook(|ns_ptr, name, value| unsafe {
            let ns = &mut *(ns_ptr as *mut crate::DictStorage);
            crate::dict_storage_store(ns, name, value);
        });
        pyre_object::dictobject::register_dict_storage_delete_hook(|ns_ptr, name| unsafe {
            let ns = &mut *(ns_ptr as *mut crate::DictStorage);
            ns.remove(name);
        });
        pyre_object::dictobject::register_dict_storage_lookup_hook(|ns_ptr, name| unsafe {
            let ns = &*(ns_ptr as *const crate::DictStorage);
            crate::dict_storage_get(ns, name)
        });
        pyre_object::dictobject::register_dict_storage_items_hook(|ns_ptr| unsafe {
            let ns = &*(ns_ptr as *const crate::DictStorage);
            ns.entries().map(|(k, v)| (k.to_string(), *v)).collect()
        });
    });
}

/// `ObjSpace.call_function(callable, *args)` — direct implementation.
///
/// PyPy: baseobjspace.py `call_function`. Now a direct function call
/// (no callback — interpreter and runtime are in the same crate).
/// PyPy: baseobjspace.py `call_function`
///
/// Dispatches to builtins, user functions, and type objects.
/// Type call uses the same __new__ + __init__ protocol as type_descr_call.
/// Re-export for crate-external callers that need a frame-less call path.
///
/// This wrapper preserves the legacy `PyObjectRef`-returning shape used by
/// most call sites. Errors are stashed in `PENDING_CALL_ERROR`; callers
/// recover them via `take_call_error()` after a `PY_NULL` return.
pub fn call_function_impl_raw(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    match call_function_impl_result(callable, args) {
        Ok(result) => result,
        Err(e) => {
            if std::env::var("PYRE_DEBUG_CALL").is_ok() {
                eprintln!("[call_function_impl] error: {}", e.message);
            }
            set_call_error(e);
            PY_NULL
        }
    }
}

pub(crate) fn call_function_impl(callable: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_function_impl_raw(callable, args)
}

/// pypy/interpreter/baseobjspace.py call_function — Result-returning entry
/// point that mirrors PyPy's OperationError-raising space.call.
///
/// This is the canonical call path. `call_function_impl_raw` (legacy)
/// wraps it for callers that expect a bare `PyObjectRef` and stash the
/// error in `PENDING_CALL_ERROR` instead.
pub fn call_function_impl_result(
    callable: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    unsafe {
        if pyre_object::is_method(callable) {
            let func = pyre_object::w_method_get_func(callable);
            let w_self = pyre_object::w_method_get_self(callable);
            let receiver = if !w_self.is_null() && !pyre_object::is_none(w_self) {
                w_self
            } else {
                pyre_object::w_method_get_class(callable)
            };
            let mut call_args = Vec::with_capacity(1 + args.len());
            if !receiver.is_null() && !pyre_object::is_none(receiver) {
                call_args.push(receiver);
            }
            call_args.extend_from_slice(args);
            return call_function_impl_result(func, &call_args);
        }
        // All callables are Function objects.
        if crate::is_function(callable) {
            let code = crate::getcode(callable);
            if crate::is_builtin_code(code as pyre_object::PyObjectRef) {
                // Builtin function: direct Rust call. Errors propagate
                // naturally through the Result return type — this is the
                // PyPy/OperationError equivalent.
                let func = crate::builtin_code_get(code as pyre_object::PyObjectRef);
                return func(args);
            }
            // User function: create frame + eval. The bare-PyObjectRef
            // helper stashes any error in `PENDING_CALL_ERROR` and returns
            // PY_NULL; recover it here so it propagates as a real Result.
            clear_call_error();
            let result = call_user_function_with_args(callable, args);
            if result.is_null() {
                if let Some(err) = take_call_error() {
                    return Err(err);
                }
            }
            return Ok(result);
        }
        // Type object → descr_call: __new__ + __init__
        // PyPy: typeobject.py descr_call → lookup __new__, call, then __init__
        if pyre_object::is_type(callable) {
            clear_call_error();
            let result = type_descr_call_impl(callable, args);
            if result.is_null() {
                if let Some(err) = take_call_error() {
                    return Err(err);
                }
            }
            return Ok(result);
        }
        // staticmethod → unwrap and call the wrapped function
        // PyPy: function.py StaticMethod.descr_staticmethod__call__
        if pyre_object::is_staticmethod(callable) {
            let func = pyre_object::w_staticmethod_get_func(callable);
            return call_function_impl_result(func, args);
        }
        // classmethod → unwrap and call the wrapped function
        // PyPy: function.py ClassMethod.descr_classmethod__call__
        if pyre_object::is_classmethod(callable) {
            let func = pyre_object::w_classmethod_get_func(callable);
            return call_function_impl_result(func, args);
        }
        // Instance with __call__ — PyPy: descroperation.py
        if pyre_object::is_instance(callable) {
            let w_type = pyre_object::w_instance_get_type(callable);
            if let Some(call_fn) = crate::baseobjspace::lookup_in_type(w_type, "__call__") {
                let mut call_args = Vec::with_capacity(1 + args.len());
                call_args.push(callable);
                call_args.extend_from_slice(args);
                return call_function_impl_result(call_fn, &call_args);
            }
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not callable",
        unsafe { (*(*callable).ob_type).name }
    )))
}

/// CPython: typeobject.c calculate_metaclass
pub(crate) fn calculate_metaclass(
    mut w_winner: PyObjectRef,
    bases: PyObjectRef,
) -> Result<PyObjectRef, PyError> {
    if w_winner.is_null() {
        w_winner = crate::typedef::w_type();
    }
    if bases.is_null() || unsafe { !pyre_object::is_tuple(bases) } {
        return Ok(w_winner);
    }
    let n = unsafe { pyre_object::w_tuple_len(bases) };
    for i in 0..n {
        let Some(base) = (unsafe { pyre_object::w_tuple_getitem(bases, i as i64) }) else {
            continue;
        };
        let Some(w_base_type) = crate::typedef::r#type(base) else {
            continue;
        };
        if std::ptr::eq(w_winner, w_base_type) || issubtype_ptr(w_winner, w_base_type) {
            continue;
        }
        if issubtype_ptr(w_base_type, w_winner) {
            w_winner = w_base_type;
            continue;
        }
        return Err(PyError::type_error("metaclass conflict"));
    }
    Ok(w_winner)
}

/// Type call without a PyFrame.
/// PyPy: typeobject.py descr_call
fn type_descr_call_impl(w_type: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    // Step 1: __new__
    let instance =
        if let Some(new_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") } {
            let mut new_args = Vec::with_capacity(1 + args.len());
            new_args.push(w_type);
            new_args.extend_from_slice(args);
            call_function_impl(new_fn, &new_args)
        } else {
            pyre_object::w_instance_new(w_type)
        };

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy checks the Python-level type(instance), so builtin-layout subtypes
    // like set subclasses still run __init__.
    if let Some(w_insttype) = type_call_init_type(instance, w_type) {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let _ = call_function_impl(init_fn, &init_args);
        }
    }

    instance
}

fn type_call_init_type(instance: PyObjectRef, w_type: PyObjectRef) -> Option<PyObjectRef> {
    let w_insttype = crate::typedef::r#type(instance)?;
    if std::ptr::eq(w_insttype, w_type) || issubtype_ptr(w_insttype, w_type) {
        Some(w_insttype)
    } else {
        None
    }
}

/// Pointer-based subtype check for descr_call __init__ guard.
fn issubtype_ptr(w_type: PyObjectRef, cls: PyObjectRef) -> bool {
    let mro_ptr = unsafe { pyre_object::w_type_get_mro(w_type) };
    if mro_ptr.is_null() {
        return false;
    }
    unsafe { (*mro_ptr).iter().any(|&t| std::ptr::eq(t, cls)) }
}

/// Helper: call a user function with arbitrary args from descriptor context.
fn call_user_function_with_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    let w_code = unsafe { crate::getcode(func) };
    let globals = unsafe { function_get_globals(func) };
    let closure = unsafe { function_get_closure(func) };
    let defaults = unsafe { crate::function_get_defaults(func) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };
    let exec_ctx = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if exec_ctx.is_null() {
        LAST_EXEC_CTX.with(|c| c.get())
    } else {
        exec_ctx
    };

    // Fill defaults for missing args
    let code_ref = unsafe { &*func_code };
    let nparams = code_ref.arg_count as usize;
    let nargs = args.len();
    let filled_args = if nargs < nparams && !defaults.is_null() {
        let defaults = crate::baseobjspace::unwrap_cell(defaults);
        let mut full = Vec::with_capacity(nparams);
        full.extend_from_slice(args);
        let ndefaults = if unsafe { pyre_object::is_tuple(defaults) } {
            unsafe { pyre_object::w_tuple_len(defaults) }
        } else {
            0
        };
        let first_default = nparams - ndefaults;
        for i in nargs..nparams {
            if i >= first_default {
                let di = i - first_default;
                full.push(
                    unsafe { pyre_object::w_tuple_getitem(defaults, di as i64) }.unwrap_or(PY_NULL),
                );
            } else {
                full.push(PY_NULL);
            }
        }
        full
    } else {
        args.to_vec()
    };

    // Fill keyword-only defaults (same logic as call_user_function_with_eval)
    let nkwonly = code_ref.kwonlyarg_count as usize;
    let mut filled_args = filled_args;
    if nkwonly > 0 {
        let kwdefaults = unsafe { crate::function_get_kwdefaults(func) };
        while filled_args.len() < nparams + nkwonly {
            filled_args.push(PY_NULL);
        }
        if !kwdefaults.is_null() && unsafe { pyre_object::is_dict(kwdefaults) } {
            for ki in 0..nkwonly {
                let slot = nparams + ki;
                if filled_args[slot].is_null() {
                    let param_name = &code_ref.varnames[slot];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwdefaults, key) } {
                        filled_args[slot] = val;
                    }
                }
            }
        }
    }

    let final_args = pack_varargs(code_ref, filled_args);

    // Generator function: wrap frame in generator object
    if code_ref
        .flags
        .intersects(crate::CodeFlags::GENERATOR | crate::CodeFlags::COROUTINE)
    {
        let mut gen_frame =
            PyFrame::new_for_call_with_closure(w_code, &final_args, globals, exec_ctx, closure);
        gen_frame.fix_array_ptrs();
        return match gen_frame.run() {
            Ok(v) => v,
            Err(e) => {
                set_call_error(e);
                PY_NULL
            }
        };
    }

    let mut frame =
        PyFrame::new_for_call_with_closure(w_code, &final_args, globals, exec_ctx, closure);
    frame.fix_array_ptrs();
    match frame.execute_frame(None, None) {
        Ok(v) => v,
        Err(e) => {
            set_call_error(e);
            PY_NULL
        }
    }
}

/// Call a metaclass with extra keyword arguments.
///
/// PyPy: metaclass(name, bases, namespace, **kwds).
/// Resolves kwargs to the metaclass __new__'s kwonly / **kwds parameters.
fn call_metaclass_with_kwargs(
    w_metaclass: PyObjectRef,
    name: PyObjectRef,
    bases: PyObjectRef,
    w_namespace_dict: PyObjectRef,
    kwargs: PyObjectRef,
) -> PyObjectRef {
    // Find the metaclass __new__ method
    let new_fn = if unsafe { pyre_object::is_type(w_metaclass) } {
        unsafe { crate::baseobjspace::lookup_in_type(w_metaclass, "__new__") }
    } else {
        None
    };

    if let Some(new_fn) = new_fn {
        if unsafe { crate::is_function(new_fn) } {
            // User function: resolve kwargs to kwonly params
            let code_ptr = unsafe { crate::get_pycode(new_fn) };
            let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
            let nparams = code.arg_count as usize; // positional params
            let nkwonly = code.kwonlyarg_count as usize;

            // Build positional args: [mcs, name, bases, ns_dict]
            let mut args = vec![w_metaclass, name, bases, w_namespace_dict];

            // Fill kwonly params from kwargs dict
            for ki in 0..nkwonly {
                let param_idx = nparams + ki;
                if param_idx < code.varnames.len() {
                    let param_name = &code.varnames[param_idx];
                    let key = pyre_object::w_str_new(param_name);
                    if let Some(val) = unsafe { pyre_object::w_dict_lookup(kwargs, key) } {
                        args.push(val);
                    } else {
                        args.push(pyre_object::PY_NULL); // will be filled by defaults
                    }
                }
            }

            return call_user_function_with_args(new_fn, &args);
        }
    }

    // Fallback: call without kwargs
    crate::call_function(w_metaclass, &[name, bases, w_namespace_dict])
}

/// Pack excess positional args into *args tuple, add empty **kwargs dict.
/// PyPy: argument.py _match_signature varargs/varkeywords packing
fn pack_varargs(code: &crate::CodeObject, args: Vec<PyObjectRef>) -> Vec<PyObjectRef> {
    let nparams = (code.arg_count + code.kwonlyarg_count) as usize;
    let has_varargs = code.flags.contains(crate::CodeFlags::VARARGS);
    let has_varkw = code.flags.contains(crate::CodeFlags::VARKEYWORDS);

    if !has_varargs && !has_varkw {
        return args;
    }

    let mut packed = Vec::with_capacity(nparams + 2);
    // Regular positional args
    for i in 0..nparams.min(args.len()) {
        packed.push(args[i]);
    }
    // Fill missing params with PY_NULL
    while packed.len() < nparams {
        packed.push(pyre_object::PY_NULL);
    }
    if has_varargs {
        let extra: Vec<_> = if args.len() > nparams {
            args[nparams..].to_vec()
        } else {
            vec![]
        };
        packed.push(pyre_object::w_tuple_new(extra));
    }
    if has_varkw {
        packed.push(pyre_object::w_dict_new());
    }
    packed
}

/// The real __build_class__(body_fn, name, *bases) implementation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS →
///   w_methodsdict = call(body_fn)
///   w_newclass = call(metaclass, name, bases, methodsdict)
/// `__build_class__(func, name, *bases, metaclass=None, **kwds)`
///
/// PyPy: pyopcode.py BUILD_CLASS → build_class()
pub(crate) fn real_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "__build_class__ requires at least 2 arguments",
        ));
    }
    let body_fn = args[0];
    let name_obj = args[1];

    // Check if last arg is a kwargs dict (from CALL_KW)
    // PyPy: __build_class__(func, name, *bases, metaclass=None, **kwds)
    let (base_args, metaclass, extra_kwargs) = if args.len() > 2 {
        let last = args[args.len() - 1];
        if unsafe { pyre_object::is_dict(last) }
            && unsafe {
                pyre_object::w_dict_lookup(last, pyre_object::w_str_new("__pyre_kw__")).is_some()
            }
        {
            let w_metaclass =
                unsafe { pyre_object::w_dict_lookup(last, pyre_object::w_str_new("metaclass")) };
            // Collect extra kwargs (not metaclass, not __pyre_kw__)
            let extra = pyre_object::w_dict_new();
            unsafe {
                let d = &*(last as *const pyre_object::dictobject::W_DictObject);
                for &(k, v) in &*d.entries {
                    if pyre_object::is_str(k) {
                        let key = pyre_object::w_str_get_value(k);
                        if key != "metaclass" && key != "__pyre_kw__" {
                            pyre_object::w_dict_store(extra, k, v);
                        }
                    }
                }
            }
            (&args[2..args.len() - 1], w_metaclass, Some(extra))
        } else {
            (&args[2..], None, None)
        }
    } else {
        (&args[2..], None, None)
    };

    let name = unsafe { pyre_object::w_str_get_value(name_obj) };
    let bases_tuple = pyre_object::w_tuple_new(base_args.to_vec());

    // If no explicit metaclass, infer from bases (PyPy: calculate_metaclass)
    let w_metaclass = metaclass.or_else(|| {
        unsafe {
            if !pyre_object::is_tuple(bases_tuple) {
                return None;
            }
            let n = pyre_object::w_tuple_len(bases_tuple);
            for i in 0..n {
                if let Some(base) = pyre_object::w_tuple_getitem(bases_tuple, i as i64) {
                    if pyre_object::is_type(base) {
                        // baseobjspace.py:76 — metaclass from w_class
                        let w_class = (*base).w_class;
                        let w_type_type = crate::typedef::w_type();
                        if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                            return Some(w_class);
                        }
                    }
                }
            }
        }
        None
    });

    build_class_inner(body_fn, name, bases_tuple, w_metaclass, extra_kwargs)
}

fn build_class_inner(
    body_fn: PyObjectRef,
    name: &str,
    bases: PyObjectRef,
    w_metaclass: Option<PyObjectRef>,
    extra_kwargs: Option<PyObjectRef>,
) -> PyResult {
    let w_code = unsafe { crate::getcode(body_fn) };
    let globals = unsafe { function_get_globals(body_fn) };
    let closure = unsafe { function_get_closure(body_fn) };
    let func_code = unsafe {
        crate::w_code_get_ptr(w_code as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // Call metaclass.__prepare__(name, bases, **kwds) if it exists.
    // PyPy: build_class → metaclass.__prepare__(name, bases, **kwds)
    // Returns the namespace dict to use for the class body.
    let w_namespace = if let Some(w_metaclass) = w_metaclass {
        if unsafe { pyre_object::is_type(w_metaclass) } {
            match crate::baseobjspace::getattr(w_metaclass, "__prepare__") {
                Ok(prepare) => {
                    let ns_obj =
                        crate::call_function(prepare, &[pyre_object::w_str_new(name), bases]);
                    if !ns_obj.is_null() && unsafe { !pyre_object::is_none(ns_obj) } {
                        Some(ns_obj)
                    } else {
                        None
                    }
                }
                Err(_) => None,
            }
        } else {
            None
        }
    } else {
        None
    };

    // Create class namespace — use __prepare__ result or fresh namespace.
    // __prepare__ may return a dict subclass (e.g. EnumDict).
    // dict subclass instances created by w_instance_new store entries in
    // ATTR_TABLE, not in W_DictObject.entries. We handle both cases.
    let mut class_ns = Box::new(DictStorage::new());
    if let Some(w_prepared_dict) = w_namespace {
        if unsafe { pyre_object::is_dict(w_prepared_dict) } {
            let dict =
                unsafe { &*(w_prepared_dict as *const pyre_object::dictobject::W_DictObject) };
            for &(key, value) in unsafe { &*dict.entries } {
                if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                    crate::dict_storage_store(
                        &mut class_ns,
                        unsafe { pyre_object::w_str_get_value(key) },
                        value,
                    );
                }
            }
        }
        // dict subclass instance (e.g. EnumDict): backing dict via __dict_data__
        if unsafe { pyre_object::is_instance(w_prepared_dict) } {
            let backing = crate::type_methods::resolve_dict_backing(w_prepared_dict);
            if !backing.is_null() && unsafe { pyre_object::is_dict(backing) } {
                let dict = unsafe { &*(backing as *const pyre_object::dictobject::W_DictObject) };
                for &(key, value) in unsafe { &*dict.entries } {
                    if !value.is_null() && unsafe { pyre_object::is_str(key) } {
                        crate::dict_storage_store(
                            &mut class_ns,
                            unsafe { pyre_object::w_str_get_value(key) },
                            value,
                        );
                    }
                }
            }
        }
    }
    class_ns.fix_ptr();
    let class_ns_ptr = Box::into_raw(class_ns);

    // w_namespace: if __prepare__ returned a custom dict, we'll replay
    // class body stores into it after execution. This lets EnumDict etc.
    // track member definitions via __setitem__.

    let stored = BUILD_CLASS_EXEC_CTX.with(|c| c.get());
    let exec_ctx = if stored.is_null() {
        std::ptr::null::<crate::PyExecutionContext>()
    } else {
        stored
    };

    // Create frame with class_locals set AND closure from enclosing scope.
    // PyPy: executes class body with w_locals = fresh dict, w_globals = module globals,
    // and the closure tuple is passed through for LOAD_DEREF access.
    // Debug: dump code object for __class__ cell investigation
    let code_ref = unsafe { &*func_code };
    if std::env::var("PYRE_DEBUG_CLASS").is_ok() {
        eprintln!("[build_class] name={name}");
        eprintln!("  varnames: {:?}", code_ref.varnames);
        eprintln!("  cellvars: {:?}", code_ref.cellvars);
        eprintln!("  freevars: {:?}", code_ref.freevars);
        eprintln!(
            "  nlocals={} ncells={} nfree={}",
            code_ref.varnames.len(),
            code_ref.cellvars.len(),
            code_ref.freevars.len()
        );
        for (i, instr) in code_ref.instructions.iter().enumerate().take(20) {
            eprintln!("  {i}: {:?}", instr);
        }
    }

    let mut frame = PyFrame::new_for_call_with_closure(w_code, &[], globals, exec_ctx, closure);
    frame.setdictscope(class_ns_ptr)?;

    frame.execute_frame(None, None)?;

    // Create W_TypeObject from the class namespace
    // PyPy: type.__new__(type, name, bases, dict_w) + compute_mro + ready()
    // PyPy: typeobject.py — if not bases_w: bases_w = [space.w_object]
    let w_effective_bases = if bases.is_null()
        || !unsafe { pyre_object::is_tuple(bases) }
        || unsafe { pyre_object::w_tuple_len(bases) } == 0
    {
        let w_object = crate::typedef::w_object();
        if !w_object.is_null() {
            pyre_object::w_tuple_new(vec![w_object])
        } else {
            bases
        }
    } else {
        bases
    };
    // Create class via metaclass or default type()
    // PyPy: typeobject.py — metaclass(name, bases, dict_w) or type.__new__
    let w_type = if let Some(w_metaclass) = w_metaclass {
        // Convert class namespace to a dict for metaclass call.
        // If __prepare__ returned a custom dict, replay stores into it
        // so that __setitem__ side effects (e.g. EnumDict tracking) fire.
        let w_namespace_dict = if let Some(w_prepared_dict) = w_namespace {
            // Replay class body stores into prepared dict
            let ns = unsafe { &*class_ns_ptr };
            for (k, &v) in ns.entries() {
                if !v.is_null() {
                    let key = pyre_object::w_str_new(k);
                    // Use setitem to trigger __setitem__ on EnumDict etc.
                    let _ = crate::baseobjspace::setitem(w_prepared_dict, key, v);
                }
            }
            w_prepared_dict
        } else {
            let d = pyre_object::w_dict_new();
            let ns = unsafe { &*class_ns_ptr };
            for (k, &v) in ns.entries() {
                if !v.is_null() {
                    unsafe { pyre_object::w_dict_store(d, pyre_object::w_str_new(k), v) };
                }
            }
            d
        };
        // Call metaclass(name, bases, namespace, **kwds)
        // Pass the ORIGINAL bases (not w_effective_bases) — the metaclass
        // expects the user-declared bases. Default (object,) is added by
        // type.__new__ internally if needed.
        let name_obj = pyre_object::w_str_new(name);
        clear_call_error();
        let result = if let Some(kw) = extra_kwargs {
            // Only use kwargs path if there are actual extra kwargs
            let has_extra = unsafe { pyre_object::is_dict(kw) && pyre_object::w_dict_len(kw) > 0 };
            if has_extra {
                call_metaclass_with_kwargs(w_metaclass, name_obj, bases, w_namespace_dict, kw)
            } else {
                crate::call_function(w_metaclass, &[name_obj, bases, w_namespace_dict])
            }
        } else {
            crate::call_function(w_metaclass, &[name_obj, bases, w_namespace_dict])
        };
        // If the metaclass call raised, propagate the original error rather
        // than silently producing a NULL class object.
        if result.is_null() {
            if let Some(err) = take_call_error() {
                return Err(err);
            }
            return Err(PyError::type_error(format!(
                "metaclass call for {name} returned NULL"
            )));
        }
        // baseobjspace.py:76 getclass() — set w_class to the metaclass
        // so type(C) returns the correct metatype.
        if unsafe { pyre_object::is_type(result) } {
            let mro = unsafe { crate::baseobjspace::compute_default_mro(result) };
            unsafe { pyre_object::w_type_set_mro(result, mro) };
            unsafe {
                if (*result).w_class.is_null() {
                    (*result).w_class = w_metaclass;
                }
            }
        }
        result
    } else {
        let w = pyre_object::w_type_new(name, w_effective_bases, class_ns_ptr as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe {
            let ns = &*class_ns_ptr;
            create_all_slots(w, ns, w_effective_bases)?;
        }
        // baseobjspace.py:76 — set w_class to 'type' (default metaclass)
        unsafe {
            (*w).w_class = crate::typedef::w_type();
        }
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w) };
        unsafe { pyre_object::w_type_set_mro(w, mro) };
        // __set_name__ protocol — type_new_set_names
        // Only needed here because w_type_new is a raw Rust call that
        // bypasses the type() builtin (builtins.rs) which already calls
        // __set_name__. The metaclass path above goes through type.__new__()
        // which handles __set_name__ in builtins.rs, so we must NOT call it
        // again there to avoid double invocation.
        if unsafe { pyre_object::is_type(w) } {
            let ns = unsafe { &*class_ns_ptr };
            let entries: Vec<(String, PyObjectRef)> =
                ns.entries().map(|(k, &v)| (k.to_string(), v)).collect();
            for (attr_name, value) in entries {
                if !value.is_null() {
                    if let Ok(set_name) = crate::baseobjspace::getattr(value, "__set_name__") {
                        let _ = crate::call_function(
                            set_name,
                            &[w, pyre_object::w_str_new(&attr_name)],
                        );
                    }
                }
            }
        }
        w
    };

    // CPython: if __classcell__ is in the namespace, set the cell's content
    // to the newly created class. This enables `__class__` references in methods.
    let class_ns = unsafe { &*class_ns_ptr };
    if let Some(&classcell) = class_ns.get("__classcell__") {
        if !classcell.is_null() && unsafe { pyre_object::is_cell(classcell) } {
            unsafe { pyre_object::w_cell_set(classcell, w_type) };
        }
    }

    // Call __init_subclass__ on each base class
    // PyPy: typeobject.py type.__init__ → call __init_subclass__
    if !w_effective_bases.is_null() && unsafe { pyre_object::is_tuple(w_effective_bases) } {
        let n = unsafe { pyre_object::w_tuple_len(w_effective_bases) };
        for i in 0..n {
            if let Some(base) = unsafe { pyre_object::w_tuple_getitem(w_effective_bases, i as i64) }
            {
                if unsafe { pyre_object::is_type(base) } {
                    if let Some(init_sub) =
                        unsafe { crate::baseobjspace::lookup_in_type(base, "__init_subclass__") }
                    {
                        let _ = crate::call_function(init_sub, &[w_type]);
                    }
                }
            }
        }
    }

    Ok(w_type)
}

thread_local! {
    /// Execution context for __build_class__ calls.
    /// Set before eval_loop starts so build_class can access it.
    static BUILD_CLASS_EXEC_CTX: Cell<*const crate::PyExecutionContext> =
        const { Cell::new(std::ptr::null()) };
}

/// Set the execution context for __build_class__ to use.
pub fn set_build_class_exec_ctx(ctx: *const crate::PyExecutionContext) {
    BUILD_CLASS_EXEC_CTX.with(|c| c.set(ctx));
}

// ── Type calling (instance creation) ─────────────────────────────────
// PyPy equivalent: typeobject.py descr_call → __new__ + __init__

fn type_descr_call(frame: &mut PyFrame, w_type: PyObjectRef, args: &[PyObjectRef]) -> PyResult {
    // Step 1: Look up __new__ via type MRO → allocate instance
    // PyPy: typeobject.py descr_call → w_type.lookup_where('__new__'),
    // then bind/call the resulting descriptor with w_type as the first arg.
    let instance =
        if let Some(new_fn) = unsafe { crate::baseobjspace::lookup_in_type(w_type, "__new__") } {
            // Call __new__(cls, *args)
            let mut new_args = Vec::with_capacity(1 + args.len());
            new_args.push(w_type);
            new_args.extend_from_slice(args);
            call_callable(frame, new_fn, &new_args)?
        } else {
            // Default: allocate bare instance
            pyre_object::w_instance_new(w_type)
        };

    // Step 2: __init__ — only if __new__ returned an instance of w_type.
    // PyPy: descr_call — skips __init__ when __new__ returns a foreign type.
    if let Some(w_insttype) = type_call_init_type(instance, w_type) {
        if let Some(init_fn) =
            unsafe { crate::baseobjspace::lookup_in_type(w_insttype, "__init__") }
        {
            let mut init_args = Vec::with_capacity(1 + args.len());
            init_args.push(instance);
            init_args.extend_from_slice(args);
            let _ = call_callable(frame, init_fn, &init_args)?;
        }
    }

    Ok(instance)
}

/// typeobject.py:1157-1176 — unpack __slots__ to slot name strings.
///
/// PyPy:
///   if isinstance(w_slots, (bytes, unicode)):
///       slot_names_w = [w_slots]
///   else:
///       slot_names_w = space.unpackiterable(w_slots)
///   for w_slot_name in slot_names_w:
///       slot_name = space.text_w(w_slot_name)
fn collect_slot_names(w_slots: pyre_object::PyObjectRef) -> Result<Vec<String>, crate::PyError> {
    unsafe {
        // typeobject.py:1158-1162: str → single-element list, else unpackiterable
        let slot_names_w = if pyre_object::is_str(w_slots) {
            vec![w_slots]
        } else {
            crate::baseobjspace::unpackiterable(w_slots, -1)?
        };
        let mut names = Vec::new();
        for w_slot_name in slot_names_w {
            if !pyre_object::is_str(w_slot_name) {
                return Err(crate::PyError::type_error(
                    "__slots__ items must be strings, not type".to_string(),
                ));
            }
            let slot_name = pyre_object::w_str_get_value(w_slot_name).to_string();
            // typeobject.py:1208-1209 valid_slot_name
            if !valid_slot_name(&slot_name) {
                return Err(crate::PyError::type_error(
                    "__slots__ must be identifiers".to_string(),
                ));
            }
            names.push(slot_name);
        }
        Ok(names)
    }
}

/// typeobject.py:1234-1240 valid_slot_name:
///   if len(slot_name) == 0 or slot_name[0].isdigit(): return False
///   for c in slot_name: if not c.isalnum() and c != '_': return False
///   return True
fn valid_slot_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    let mut chars = name.chars();
    let first = chars.next().unwrap();
    if first.is_ascii_digit() {
        return false;
    }
    if !first.is_alphanumeric() && first != '_' {
        return false;
    }
    for c in chars {
        if !c.is_alphanumeric() && c != '_' {
            return false;
        }
    }
    true
}

/// astcompiler/misc.py:78-92 mangle(name, klass):
///   if not name.startswith('__'): return name
///   if name.endswith('__') or '.' in name: return name
///   strip leading underscores from klass
///   return "_%s%s" % (klass[i:], name)
fn mangle(name: &str, klass: &str) -> String {
    if !name.starts_with("__") {
        return name.to_string();
    }
    if name.ends_with("__") || name.contains('.') {
        return name.to_string();
    }
    let stripped = klass.trim_start_matches('_');
    if stripped.is_empty() {
        return name.to_string();
    }
    format!("_{stripped}{name}")
}

/// typeobject.py:1131-1140 copy_flags_from_bases:
///   w_self.hasdict |= w_base.hasdict
///   w_self.weakrefable |= w_base.weakrefable
unsafe fn copy_flags_from_bases(
    w_type: pyre_object::PyObjectRef,
    w_bases: pyre_object::PyObjectRef,
) {
    unsafe {
        if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            return;
        }
        let len = pyre_object::w_tuple_len(w_bases);
        for i in 0..len {
            if let Some(base) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                if pyre_object::is_type(base) {
                    if pyre_object::w_type_get_hasdict(base) {
                        pyre_object::w_type_set_hasdict(w_type, true);
                    }
                    if pyre_object::w_type_get_weakrefable(base) {
                        pyre_object::w_type_set_weakrefable(w_type, true);
                    }
                }
            }
        }
    }
}

/// typeobject.py:1143-1204 create_all_slots.
///
/// Returns `Err` for invalid __slots__ (TypeError), matching PyPy.
///
/// # Safety
/// `w_type` must be a valid W_TypeObject pointer.
pub unsafe fn create_all_slots(
    w_type: pyre_object::PyObjectRef,
    ns: &crate::DictStorage,
    w_bases: pyre_object::PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        use pyre_object::typeobject::{Layout, leak_layout};

        // typeobject.py:1245: w_bestbase = check_and_find_best_base(space, bases_w)
        let w_bestbase = check_and_find_best_base(w_bases)?;

        // typeobject.py:1254: copy_flags_from_bases — inherit hasdict/weakrefable
        copy_flags_from_bases(w_type, w_bases);

        // typeobject.py:1146: base_layout = w_bestbase.layout
        let base_layout = if w_bestbase.is_null() {
            std::ptr::null()
        } else {
            pyre_object::w_type_get_layout_ptr(w_bestbase)
        };
        let base_nslots = if base_layout.is_null() {
            0
        } else {
            (*base_layout).nslots
        };

        // typeobject.py:1150-1204 create_all_slots
        let mut newslotnames = Vec::new();
        let (mut wantdict, mut wantweakref);
        if let Some(&w_slots) = ns.get("__slots__") {
            // typeobject.py:1154-1176: has __slots__
            wantdict = false;
            wantweakref = false;
            let all_names = collect_slot_names(w_slots)?;
            for slot_name in &all_names {
                match slot_name.as_str() {
                    // typeobject.py:1165-1169: __dict__ slot
                    "__dict__" => {
                        if wantdict || pyre_object::w_type_get_hasdict(w_type) {
                            return Err(crate::PyError::type_error(
                                "__dict__ slot disallowed: we already got one".to_string(),
                            ));
                        }
                        wantdict = true;
                    }
                    // typeobject.py:1170-1174: __weakref__ slot
                    "__weakref__" => {
                        if wantweakref || pyre_object::w_type_get_weakrefable(w_type) {
                            return Err(crate::PyError::type_error(
                                "__weakref__ slot disallowed: we already got one".to_string(),
                            ));
                        }
                        wantweakref = true;
                    }
                    // typeobject.py:1175-1176: regular slot name
                    _ => newslotnames.push(slot_name.clone()),
                }
            }
            // typeobject.py:1178: string_sort(newslotnames)
            newslotnames.sort();

            // typeobject.py:1183-1189: create_slot loop
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            let type_name = pyre_object::w_type_get_name(w_type);
            let mut slot_index = base_nslots;
            let mut i = 0;
            while i < newslotnames.len() {
                // typeobject.py:1208-1209: valid_slot_name check
                if !valid_slot_name(&newslotnames[i]) {
                    return Err(crate::PyError::type_error(
                        "__slots__ must be identifiers".to_string(),
                    ));
                }
                // typeobject.py:1211: slot_name = mangle(slot_name, w_self.name)
                let mangled = mangle(&newslotnames[i], type_name);
                if !type_ns.is_null() && (*type_ns).get(mangled.as_str()).is_some() {
                    // typeobject.py:1219-1220: name conflict → skip this slot
                    newslotnames.remove(i);
                } else {
                    // typeobject.py:1216-1217: create_slot
                    newslotnames[i] = mangled.clone();
                    if !type_ns.is_null() {
                        let member = pyre_object::w_member_new(slot_index, mangled.clone(), w_type);
                        (*type_ns).insert(mangled, member);
                    }
                    slot_index += 1;
                    i += 1;
                }
            }
        } else {
            // typeobject.py:1151-1153: no __slots__
            wantdict = true;
            wantweakref = true;
        }

        // typeobject.py:1192-1195: create_dict_slot / create_weakref_slot
        if wantdict {
            create_dict_slot(w_type);
        }
        if wantweakref {
            create_weakref_slot(w_type);
        }

        // typeobject.py:1199-1204: layout computation
        let nslots = base_nslots + newslotnames.len() as u32;
        let typedef = if base_layout.is_null() {
            &pyre_object::pyobject::INSTANCE_TYPE as *const _
        } else {
            (*base_layout).typedef
        };
        let layout = if nslots == base_nslots && !base_layout.is_null() {
            base_layout
        } else {
            leak_layout(Layout {
                typedef,
                nslots,
                newslotnames,
                base_layout,
                acceptable_as_base_class: true,
            })
        };
        pyre_object::w_type_set_layout(w_type, layout);
        Ok(())
    }
}

/// objspace/std/typeobject.py:1222-1226 create_dict_slot.
///
/// ```python
/// def create_dict_slot(w_self):
///     if not w_self.hasdict:
///         w_self.dict_w.setdefault('__dict__',
///             dict_descr.copy_for_type(w_self))
///         w_self.hasdict = True
/// ```
unsafe fn create_dict_slot(w_type: pyre_object::PyObjectRef) {
    unsafe {
        if !pyre_object::w_type_get_hasdict(w_type) {
            let descr =
                crate::typedef::copy_descriptor_for_type(crate::typedef::dict_descr(), w_type);
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            if !type_ns.is_null() && (*type_ns).get("__dict__").is_none() {
                (*type_ns).insert("__dict__".to_string(), descr);
            }
            pyre_object::w_type_set_hasdict(w_type, true);
        }
    }
}

/// objspace/std/typeobject.py:1228-1232 create_weakref_slot.
///
/// ```python
/// def create_weakref_slot(w_self):
///     if not w_self.weakrefable:
///         w_self.dict_w.setdefault('__weakref__',
///             weakref_descr.copy_for_type(w_self))
///         w_self.weakrefable = True
/// ```
unsafe fn create_weakref_slot(w_type: pyre_object::PyObjectRef) {
    unsafe {
        if !pyre_object::w_type_get_weakrefable(w_type) {
            let descr =
                crate::typedef::copy_descriptor_for_type(crate::typedef::weakref_descr(), w_type);
            let type_ns = pyre_object::w_type_get_dict_ptr(w_type) as *mut crate::DictStorage;
            if !type_ns.is_null() && (*type_ns).get("__weakref__").is_none() {
                (*type_ns).insert("__weakref__".to_string(), descr);
            }
            pyre_object::w_type_set_weakrefable(w_type, true);
        }
    }
}

/// typeobject.py:1089-1105 find_best_base.
unsafe fn find_best_base(w_bases: pyre_object::PyObjectRef) -> pyre_object::PyObjectRef {
    unsafe {
        if w_bases.is_null() || !pyre_object::is_tuple(w_bases) {
            return std::ptr::null_mut();
        }
        let len = pyre_object::w_tuple_len(w_bases);
        let mut w_bestbase: pyre_object::PyObjectRef = std::ptr::null_mut();
        for i in 0..len {
            if let Some(w_candidate) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                if !pyre_object::is_type(w_candidate) {
                    continue;
                }
                if w_bestbase.is_null() {
                    w_bestbase = w_candidate;
                    continue;
                }
                let cand_layout = pyre_object::w_type_get_layout_ptr(w_candidate);
                let best_layout = pyre_object::w_type_get_layout_ptr(w_bestbase);
                if cand_layout != best_layout
                    && !cand_layout.is_null()
                    && (*cand_layout).issublayout(best_layout)
                {
                    w_bestbase = w_candidate;
                }
            }
        }
        w_bestbase
    }
}

/// typeobject.py:1107-1129 check_and_find_best_base:
///   w_bestbase = find_best_base(bases_w)
///   if w_bestbase is None: raise TypeError
///   if not w_bestbase.layout.typedef.acceptable_as_base_class: raise TypeError
///   for w_base in bases_w: check layout conflicts
unsafe fn check_and_find_best_base(
    w_bases: pyre_object::PyObjectRef,
) -> Result<pyre_object::PyObjectRef, crate::PyError> {
    unsafe {
        let w_bestbase = find_best_base(w_bases);
        // typeobject.py:1113-1115
        if w_bestbase.is_null() {
            return Err(crate::PyError::type_error(
                "a new-style class can't have only classic bases".to_string(),
            ));
        }
        // typeobject.py:1116-1118: acceptable_as_base_class check.
        // typedef.py:43: acceptable = '__new__' in rawdict.
        // bool and NoneType are not acceptable in Python 3.
        if !is_acceptable_base_class(w_bestbase) {
            return Err(crate::PyError::type_error(format!(
                "type '{}' is not an acceptable base class",
                pyre_object::w_type_get_name(w_bestbase),
            )));
        }
        // typeobject.py:1122-1128: check layout conflicts
        let best_layout = pyre_object::w_type_get_layout_ptr(w_bestbase);
        if !best_layout.is_null() && !w_bases.is_null() && pyre_object::is_tuple(w_bases) {
            let len = pyre_object::w_tuple_len(w_bases);
            for i in 0..len {
                if let Some(w_base) = pyre_object::w_tuple_getitem(w_bases, i as i64) {
                    if !pyre_object::is_type(w_base) {
                        continue;
                    }
                    let layout = pyre_object::w_type_get_layout_ptr(w_base);
                    if !layout.is_null() && !(*best_layout).issublayout(layout) {
                        return Err(crate::PyError::type_error(
                            "instance layout conflicts in multiple inheritance".to_string(),
                        ));
                    }
                }
            }
        }
        Ok(w_bestbase)
    }
}

/// typedef.py:43 `acceptable_as_base_class = '__new__' in rawdict`.
/// typeobject.py:1116 checks this flag on the bestbase.
unsafe fn is_acceptable_base_class(w_type: pyre_object::PyObjectRef) -> bool {
    unsafe { pyre_object::w_type_get_acceptable_as_base_class(w_type) }
}
