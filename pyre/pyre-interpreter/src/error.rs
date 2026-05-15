use pyre_object::PyObjectRef;
use pyre_object::excobject::{ExcKind, exc_kind_name, w_exception_new};
use std::io::Write;

#[derive(Debug, Clone)]
pub struct OperationError {
    pub w_type: PyObjectRef,
    pub w_value: PyObjectRef,
    pub _application_traceback: Option<PyObjectRef>,
}

impl OperationError {
    pub fn new(w_type: PyObjectRef, w_value: PyObjectRef) -> Self {
        Self {
            w_type,
            w_value,
            _application_traceback: None,
        }
    }

    pub fn get_w_value(&self, _space: PyObjectRef) -> PyObjectRef {
        let _ = _space;
        self.w_value
    }

    pub fn match_(&self, _space: PyObjectRef, _check: PyObjectRef) -> bool {
        false
    }

    /// pypy/interpreter/error.py:180-249 `normalize_exception`.
    ///
    /// ```python
    /// def normalize_exception(self, space):
    ///     w_type = self.w_type
    ///     w_value = self.get_w_value(space)
    ///     if space.exception_is_valid_obj_as_class_w(w_type):
    ///         if space.is_w(w_value, space.w_None):
    ///             w_value = space.call_function(w_type)
    ///             w_type = self._exception_getclass(space, w_value)
    ///         else:
    ///             w_valuetype = space.exception_getclass(w_value)
    ///             if space.exception_issubclass_w(w_valuetype, w_type):
    ///                 w_type = w_valuetype
    ///             else:
    ///                 if space.isinstance_w(w_value, space.w_tuple):
    ///                     w_value = space.call(w_type, w_value)
    ///                 else:
    ///                     w_value = space.call_function(w_type, w_value)
    ///                 w_type = self._exception_getclass(space, w_value)
    ///         if self._application_traceback:
    ///             from pypy.interpreter.pytraceback import PyTraceback
    ///             from pypy.module.exceptions.interp_exceptions import W_BaseException
    ///             tb = self._application_traceback
    ///             if (isinstance(w_value, W_BaseException) and
    ///                 isinstance(tb, PyTraceback)):
    ///                 # traceback hasn't escaped yet
    ///                 w_value.w_traceback = tb
    ///             else:
    ///                 # traceback has escaped
    ///                 space.setattr(w_value, space.newtext("__traceback__"),
    ///                               self.get_w_traceback(space))
    ///     else:
    ///         w_inst = w_type
    ///         w_instclass = self._exception_getclass(space, w_inst)
    ///         if not space.is_w(w_value, space.w_None):
    ///             raise oefmt(space.w_TypeError, ...)
    ///         w_value = w_inst
    ///         w_type = w_instclass
    ///     self.w_type = w_type
    ///     self._w_value = w_value
    ///     return w_value
    /// ```
    ///
    /// Mutates `self` to install the normalized `(w_type, w_value)` and
    /// returns the new `w_value`.
    ///
    /// `pypy/interpreter/error.py:225-236 normalize_exception` traceback
    /// attach: the W_BaseException-typed fast path writes
    /// `w_value.w_traceback = tb` directly; the generic fallback is
    /// `space.setattr(w_value, "__traceback__", tb)`.  Pyre's
    /// W_ExceptionObject grew the typed `w_traceback` slot, so the
    /// fast path is now reachable — `w_exception_set_traceback` is
    /// invoked when `w_value` is an exception instance, falling back
    /// to the generic setattr for anything else.
    pub fn normalize_exception(&mut self, space: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let mut w_type = self.w_type;
        let mut w_value = self.get_w_value(space);
        unsafe {
            if crate::baseobjspace::exception_is_valid_obj_as_class_w(w_type) {
                if w_value.is_null() || w_value == pyre_object::w_none() {
                    // error.py:208-210 (Class, None): instantiate Class()
                    w_value = crate::baseobjspace::call_function(w_type, &[]);
                    if w_value.is_null() {
                        return Err(crate::call::take_call_error()
                            .unwrap_or_else(|| PyError::type_error("constructor failed")));
                    }
                    w_type = self._exception_getclass(w_value)?;
                } else {
                    // error.py:212 w_valuetype = space.exception_getclass(w_value)
                    let w_valuetype = crate::baseobjspace::exception_getclass(w_value);
                    if !w_valuetype.is_null()
                        && crate::baseobjspace::exception_issubclass_w(w_valuetype, w_type)
                    {
                        // error.py:213-215 (Class, inst): use inst's exact class
                        w_type = w_valuetype;
                    } else {
                        // error.py:217 if space.isinstance_w(w_value, space.w_tuple):
                        let w_tuple_cls =
                            crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE);
                        if !w_tuple_cls.is_null()
                            && crate::baseobjspace::isinstance_w(w_value, w_tuple_cls)
                        {
                            // error.py:218-220 (Class, tuple): Class(*tuple)
                            let items = pyre_object::w_tuple_items_copy_as_vec(w_value);
                            w_value = crate::baseobjspace::call_function(w_type, &items);
                        } else {
                            // error.py:221-223 (Class, x): Class(x)
                            w_value = crate::baseobjspace::call_function(w_type, &[w_value]);
                        }
                        if w_value.is_null() {
                            return Err(crate::call::take_call_error()
                                .unwrap_or_else(|| PyError::type_error("constructor failed")));
                        }
                        w_type = self._exception_getclass(w_value)?;
                    }
                }
                // error.py:225-236 traceback attach — fast path writes
                // `w_value.w_traceback = tb` when `w_value` is a
                // `W_BaseException`, otherwise falls through to the
                // generic `space.setattr(w_value, "__traceback__", tb)`.
                if let Some(tb) = self._application_traceback {
                    if pyre_object::is_exception(w_value) {
                        pyre_object::excobject::w_exception_set_traceback(w_value, tb);
                    } else {
                        let _ = crate::baseobjspace::setattr(w_value, "__traceback__", tb);
                    }
                }
            } else {
                // error.py:238-245 (inst, None) — `raise inst`
                let w_inst = w_type;
                let w_instclass = self._exception_getclass(w_inst)?;
                if !w_value.is_null() && w_value != pyre_object::w_none() {
                    return Err(PyError::type_error(
                        "instance exception may not have a separate value",
                    ));
                }
                w_value = w_inst;
                w_type = w_instclass;
            }
        }
        self.w_type = w_type;
        self.w_value = w_value;
        Ok(w_value)
    }

    /// pypy/interpreter/error.py:251-257 `_exception_getclass(space, w_inst, what="exceptions")`.
    ///
    /// ```python
    /// def _exception_getclass(self, space, w_inst, what="exceptions"):
    ///     w_type = space.exception_getclass(w_inst)
    ///     if not space.exception_is_valid_class_w(w_type):
    ///         raise oefmt(space.w_TypeError, ...)
    ///     return w_type
    /// ```
    fn _exception_getclass(&self, w_inst: PyObjectRef) -> Result<PyObjectRef, PyError> {
        let w_type = crate::baseobjspace::exception_getclass(w_inst);
        if w_type.is_null() || !unsafe { crate::baseobjspace::exception_is_valid_class_w(w_type) } {
            return Err(PyError::type_error(
                "exceptions must derive from BaseException",
            ));
        }
        Ok(w_type)
    }

    /// `pypy/interpreter/error.py:422-434 chain_exceptions` parity:
    ///
    /// ```python
    /// def chain_exceptions(self, space, context):
    ///     w_value = self.normalize_exception(space)
    ///     w_context = context.normalize_exception(space)
    ///     if not space.is_w(w_value, w_context):
    ///         if not isinstance(w_value, W_BaseException):
    ///             raise oefmt(space.w_SystemError, "not an instance of Exception: %T", w_value)
    ///         if w_value.w_context is None:
    ///             _break_context_cycle(space, w_value, w_context)
    ///             w_value.descr_setcontext(space, w_context)
    /// ```
    ///
    /// Writes flow through the typed `w_context` slot on
    /// `W_ExceptionObject` (`pyre-object/src/excobject.rs:113-117
    /// W_BaseException class defaults`).
    pub fn chain_exceptions(
        &mut self,
        space: PyObjectRef,
        context: &mut OperationError,
    ) -> Result<(), PyError> {
        let w_value = self.normalize_exception(space)?;
        let w_context = context.normalize_exception(space)?;
        if std::ptr::eq(w_value, w_context) {
            return Ok(());
        }
        if !unsafe { pyre_object::is_exception(w_value) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        // `:432-434` — only set __context__ when it isn't already
        // stamped; mirrors CPython's `_PyErr_ChainExceptions` precedent.
        let existing = unsafe { pyre_object::excobject::w_exception_get_context(w_value) };
        if existing.is_null() {
            _break_context_cycle(w_value, w_context)?;
            unsafe { pyre_object::excobject::w_exception_set_context(w_value, w_context) };
        }
        Ok(())
    }
}

/// `pypy/interpreter/error.py:478-509 _break_context_cycle` parity —
/// Floyd cycle-detection over the `__context__` chain, breaking the
/// loop by writing `None` into the offending link before the new
/// `w_context` is attached.
fn _break_context_cycle(w_value: PyObjectRef, w_context: PyObjectRef) -> Result<(), PyError> {
    let mut w_rabbit = w_context;
    let mut w_tortoise = w_context;
    let mut update_tortoise_toggle = false;
    loop {
        if !unsafe { pyre_object::is_exception(w_rabbit) } {
            return Err(PyError::new(
                crate::PyErrorKind::SystemError,
                "not an instance of Exception".to_string(),
            ));
        }
        let w_next = unsafe { pyre_object::excobject::w_exception_get_context(w_rabbit) };
        if w_next.is_null() || unsafe { pyre_object::is_none(w_next) } {
            break;
        }
        if std::ptr::eq(w_next, w_value) {
            // `:497-498` — `w_rabbit.descr_setcontext(space, space.w_None)`
            // (a real None, mirroring PyPy's "internal None" → user-
            // visible None via the typed slot).
            unsafe {
                pyre_object::excobject::w_exception_set_context(w_rabbit, pyre_object::w_none())
            };
            break;
        }
        w_rabbit = w_next;
        if std::ptr::eq(w_rabbit, w_tortoise) {
            // `:502-503` — pre-existing cycle; don't set anything to None.
            break;
        }
        if update_tortoise_toggle {
            if !unsafe { pyre_object::is_exception(w_tortoise) } {
                return Err(PyError::new(
                    crate::PyErrorKind::RuntimeError,
                    "not an instance of Exception".to_string(),
                ));
            }
            w_tortoise = unsafe { pyre_object::excobject::w_exception_get_context(w_tortoise) };
        }
        update_tortoise_toggle = !update_tortoise_toggle;
    }
    Ok(())
}

impl From<OperationError> for PyError {
    fn from(value: OperationError) -> Self {
        let message = if value.w_value.is_null() {
            String::new()
        } else {
            "operation error".to_string()
        };
        PyError {
            kind: PyErrorKind::RuntimeError,
            message,
            exc_object: value.w_value,
            attach_tb: true,
            reraise_lasti: -1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClearedOpErr;

#[derive(Debug, Clone)]
pub struct OpErrFmtNoArgs;

/// Result type for Python operations.
pub type PyResult = Result<PyObjectRef, PyError>;

/// Python exception.
#[derive(Debug, Clone)]
pub struct PyError {
    pub kind: PyErrorKind,
    pub message: String,
    /// Cached W_ExceptionObject pointer — reused by to_exc_object()
    /// to avoid re-allocating an exception object that already exists.
    pub exc_object: PyObjectRef,
    /// `pypy/interpreter/pyopcode.py:122 handle_operation_error(..., attach_tb=True)`
    /// parity.  RERAISE opcode (`pyopcode.py:1348-1376 RERAISE`)
    /// surfaces the operror as `RaiseWithExplicitTraceback` which
    /// PyPy's bytecode dispatcher routes through
    /// `handle_operation_error(attach_tb=False)` to avoid stamping a
    /// spurious traceback frame on the cleanup path.  Pyre routes the
    /// same intent through this field: the `reraise` opcode
    /// (`eval.rs::reraise`) clears `attach_tb` so the
    /// re-fired `handle_exception` skips
    /// `record_application_traceback`.  Default `true` for any
    /// normally-constructed PyError; reraise flips it off.
    pub attach_tb: bool,
    /// `pypy/interpreter/pyopcode.py:122 handle_operation_error(..., reraise_lasti=-1)`
    /// parity.  RERAISE N reads the original raise-site lasti from the
    /// value stack and carries it through `RaiseWithExplicitTraceback`
    /// so the next exception-table dispatch can push the original
    /// raise-site offset (not the RERAISE instruction itself) as the
    /// `lasti` value, and so the no-handler propagation path can
    /// restore `last_instr` for correct `f_lineno`.  `-1` means "no
    /// reraise lasti carried" (default for primary raises).
    pub reraise_lasti: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyErrorKind {
    TypeError,
    ValueError,
    ZeroDivisionError,
    NameError,
    IndexError,
    KeyError,
    AttributeError,
    RuntimeError,
    StopIteration,
    OverflowError,
    ArithmeticError,
    ImportError,
    NotImplementedError,
    AssertionError,
    /// Raised by `_weakref` when a proxy is dereferenced after the
    /// referent has been collected.
    /// pypy/module/_weakref/interp__weakref.py:347
    /// `oefmt(space.w_ReferenceError, "weakly referenced object no longer exists")`.
    ReferenceError,
    /// BaseException subclass — raised inside generators by gen.close().
    /// Not a subclass of Exception, so `except Exception` does not catch it.
    GeneratorExit,
    RecursionError,
    /// Internal: RETURN_GENERATOR unwind signal (not a real exception).
    /// Carries the generator PyObjectRef as message.
    GeneratorReturn,
    /// Internal parity marker for pypy/interpreter/pycode.py:25
    /// `BytecodeCorruption`. This should not be raised as a user-level
    /// Python exception; it signals malformed bytecode in the interpreter.
    BytecodeCorruption,
    /// Base class for all operating-system errors.
    /// pypy/module/exceptions/interp_exceptions.py W_OSError.
    OSError,
    /// Subclass of OSError raised when a file or directory is not found.
    FileNotFoundError,
    /// pypy/interpreter/baseobjspace.py:419-420 DescrMismatch.
    ///
    /// ```python
    /// class DescrMismatch(Exception):
    ///     pass
    /// ```
    ///
    /// Internal control-flow exception raised by `space.descr_self_interp_w`
    /// when a descriptor's typecheck wrapper sees an instance of the wrong
    /// class. Caught by `GetSetProperty.descr_property_get/set/del` which
    /// then re-raises a user-visible TypeError via `descr_call_mismatch`.
    DescrMismatch,
    /// Raised by sys.exit(). Not a subclass of Exception.
    SystemExit,
    /// rpython/jit/metainterp/compile.py:1090 `memory_error = MemoryError()`
    /// — raised by `PropagateExceptionDescr.handle_fail` when a JIT
    /// malloc helper returns NULL (true OOM).  Not raised by user code
    /// in pyre; the runtime allocator never returns NULL except on
    /// genuine system OOM.
    MemoryError,
    /// `pypy/module/exceptions/interp_exceptions.py W_SystemError` —
    /// PyPy uses this for interpreter-internal invariant violations
    /// (e.g. `chain_exceptions` rejecting non-BaseException context
    /// at `error.py:429`).
    SystemError,
    /// Internal JIT trace-abort signal. This is not a Python exception
    /// class and must be intercepted before user-visible error handling.
    TraceAbort,
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        PyError {
            kind,
            message: message.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn type_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::TypeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn attribute_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::AttributeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn value_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ValueError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn zero_division(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ZeroDivisionError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn overflow_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::OverflowError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::RuntimeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn internal_trace_abort(reason: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::TraceAbort,
            message: reason.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: false,
            reraise_lasti: -1,
        }
    }

    pub fn key_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::KeyError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn os_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::OSError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    /// Raise an OSError (or FileNotFoundError when errno is ENOENT) with
    /// a platform-style error message.
    pub fn os_error_with_errno(errno: i32, msg: impl Into<String>) -> Self {
        let kind = if errno == 2 {
            PyErrorKind::FileNotFoundError
        } else {
            PyErrorKind::OSError
        };
        PyError {
            kind,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    /// pypy/module/_weakref/interp__weakref.py:347 — raised by `force()`
    /// when the referent of a proxy is no longer alive.
    pub fn reference_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ReferenceError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn recursion_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::RecursionError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    /// rpython/jit/metainterp/compile.py:1090 `memory_error = MemoryError()`
    /// — module-level singleton instance the JIT raises through
    /// `PropagateExceptionDescr.handle_fail` when a malloc helper
    /// returns NULL.
    pub fn memory_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::MemoryError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    pub fn stop_iteration() -> Self {
        PyError {
            kind: PyErrorKind::StopIteration,
            message: String::new(),
            exc_object: std::ptr::null_mut(),
            attach_tb: true,
            reraise_lasti: -1,
        }
    }

    /// Convert to a W_ExceptionObject for pushing onto the value stack.
    /// Reuses the cached object from from_exc_object() if available.
    ///
    /// Mirrors `pypy/interpreter/error.py:OperationError.get_w_value`'s
    /// upgrade-to-exception-instance path: when the OperationError
    /// carries a raw message (the `oefmt` shape), the materialised
    /// exception instance gets `args = (msg,)` per
    /// `pypy/module/exceptions/interp_exceptions.py:123-124
    /// W_BaseException.descr_init` — `self.args_w = args_w`.  Pyre
    /// stores `args_w` as a `W_ListObject`, so we stamp a one-element
    /// list `[msg_str]` here so `str(e)` and `repr(e)` and
    /// `e.args == (msg,)` all line up with PyPy.
    pub fn to_exc_object(&self) -> PyObjectRef {
        if !self.exc_object.is_null() {
            return self.exc_object;
        }
        let exc = w_exception_new(self.to_exc_kind(), &self.message);
        if !self.message.is_empty() {
            let msg = pyre_object::w_str_new(&self.message);
            let args_list = pyre_object::w_list_new(vec![msg]);
            unsafe { pyre_object::excobject::w_exception_set_args(exc, args_list) };
        }
        exc
    }

    fn to_exc_kind(&self) -> ExcKind {
        match self.kind {
            PyErrorKind::TypeError => ExcKind::TypeError,
            PyErrorKind::ValueError => ExcKind::ValueError,
            PyErrorKind::ZeroDivisionError => ExcKind::ZeroDivisionError,
            PyErrorKind::NameError => ExcKind::NameError,
            PyErrorKind::IndexError => ExcKind::IndexError,
            PyErrorKind::KeyError => ExcKind::KeyError,
            PyErrorKind::AttributeError => ExcKind::AttributeError,
            PyErrorKind::RuntimeError => ExcKind::RuntimeError,
            PyErrorKind::StopIteration => ExcKind::StopIteration,
            PyErrorKind::OverflowError => ExcKind::OverflowError,
            PyErrorKind::ArithmeticError => ExcKind::ArithmeticError,
            PyErrorKind::ImportError => ExcKind::ImportError,
            PyErrorKind::NotImplementedError => ExcKind::NotImplementedError,
            PyErrorKind::AssertionError => ExcKind::AssertionError,
            PyErrorKind::ReferenceError => ExcKind::ReferenceError,
            PyErrorKind::GeneratorExit => ExcKind::GeneratorExit,
            PyErrorKind::RecursionError => ExcKind::RecursionError,
            PyErrorKind::GeneratorReturn => ExcKind::RuntimeError,
            // Internal-only marker. If it escapes to object-space conversion,
            // degrade to RuntimeError rather than inventing a new exception type.
            PyErrorKind::BytecodeCorruption => ExcKind::RuntimeError,
            PyErrorKind::OSError => ExcKind::OSError,
            PyErrorKind::FileNotFoundError => ExcKind::FileNotFoundError,
            // DescrMismatch is a control-flow exception caught by
            // GetSetProperty.descr_property_get/set/del. If it ever escapes
            // to user code without being converted to TypeError it surfaces
            // as a TypeError, matching PyPy's eventual descr_call_mismatch.
            PyErrorKind::DescrMismatch => ExcKind::TypeError,
            PyErrorKind::SystemExit => ExcKind::SystemExit,
            PyErrorKind::MemoryError => ExcKind::MemoryError,
            PyErrorKind::SystemError => ExcKind::SystemError,
            PyErrorKind::TraceAbort => ExcKind::RuntimeError,
        }
    }

    /// Create a PyError from a W_ExceptionObject.
    ///
    /// # Safety
    /// `obj` must point to a valid `W_ExceptionObject`.
    pub unsafe fn from_exc_object(obj: PyObjectRef) -> Self {
        unsafe {
            let kind = pyre_object::excobject::w_exception_get_kind(obj);
            let message = pyre_object::excobject::w_exception_get_message(obj).to_string();
            PyError {
                kind: Self::kind_from_exc(kind),
                message,
                exc_object: obj,
                attach_tb: true,
                reraise_lasti: -1,
            }
        }
    }

    pub fn kind_from_exc(kind: ExcKind) -> PyErrorKind {
        match kind {
            ExcKind::TypeError => PyErrorKind::TypeError,
            ExcKind::ValueError => PyErrorKind::ValueError,
            ExcKind::ZeroDivisionError => PyErrorKind::ZeroDivisionError,
            ExcKind::NameError => PyErrorKind::NameError,
            ExcKind::IndexError => PyErrorKind::IndexError,
            ExcKind::KeyError => PyErrorKind::KeyError,
            ExcKind::AttributeError => PyErrorKind::AttributeError,
            ExcKind::RuntimeError => PyErrorKind::RuntimeError,
            ExcKind::StopIteration => PyErrorKind::StopIteration,
            ExcKind::OverflowError => PyErrorKind::OverflowError,
            ExcKind::ArithmeticError => PyErrorKind::ArithmeticError,
            ExcKind::ImportError => PyErrorKind::ImportError,
            ExcKind::NotImplementedError => PyErrorKind::NotImplementedError,
            ExcKind::AssertionError => PyErrorKind::AssertionError,
            ExcKind::ReferenceError => PyErrorKind::ReferenceError,
            ExcKind::GeneratorExit => PyErrorKind::GeneratorExit,
            ExcKind::RecursionError => PyErrorKind::RecursionError,
            ExcKind::BaseException | ExcKind::Exception => PyErrorKind::RuntimeError,
            ExcKind::OSError => PyErrorKind::OSError,
            ExcKind::FileNotFoundError => PyErrorKind::FileNotFoundError,
            // Unicode errors don't have a dedicated PyErrorKind; they
            // flow through the general ValueError handler.
            ExcKind::UnicodeDecodeError | ExcKind::UnicodeEncodeError => PyErrorKind::ValueError,
            ExcKind::SystemExit => PyErrorKind::SystemExit,
            ExcKind::MemoryError => PyErrorKind::MemoryError,
            ExcKind::SystemError => PyErrorKind::SystemError,
        }
    }

    pub fn render_exception(&self) -> String {
        let name = exc_kind_name(self.to_exc_kind());
        if self.message.is_empty() {
            name.to_string()
        } else {
            format!("{name}: {}", self.message)
        }
    }
}

impl std::fmt::Display for PyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.render_exception())
    }
}

pub fn write_exception<W: Write>(
    writer: &mut W,
    err: &PyError,
    include_traceback: bool,
) -> std::io::Result<()> {
    if include_traceback {
        // `traceback.py:171-194` __cause__ / __context__ chain
        // printing.  Recurse into the older exception first, emit
        // the bridging banner, then print the current exception.
        if !err.exc_object.is_null() {
            write_chained_context(writer, err.exc_object)?;
        }
        writeln!(writer, "Traceback (most recent call last):")?;
        write_traceback_chain(writer, err)?;
        writeln!(writer, "{}", err.render_exception())?;
        write_exception_notes(writer, err.exc_object)
    } else {
        writeln!(writer, "{}", err.render_exception())
    }
}

/// `Lib/traceback.py:171-200 TracebackException._format_*` parity —
/// walk `__cause__` (or `__context__` when `__suppress_context__` is
/// False) chains and emit each older exception with the appropriate
/// bridging banner before the current one.
fn write_chained_context<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let cause = unsafe { pyre_object::excobject::w_exception_get_cause(exc) };
    let context = unsafe { pyre_object::excobject::w_exception_get_context(exc) };
    let suppress = unsafe { pyre_object::excobject::w_exception_get_suppress_context(exc) };

    // `traceback.py:184-191` — `__cause__` wins over `__context__`;
    // `__suppress_context__` (set by `raise X from None`) hides
    // `__context__` only when no explicit cause was attached.
    let (older, banner) = if !cause.is_null() && !unsafe { pyre_object::is_none(cause) } {
        (
            cause,
            "\nThe above exception was the direct cause of the following exception:\n",
        )
    } else if !suppress && !context.is_null() && !unsafe { pyre_object::is_none(context) } {
        (
            context,
            "\nDuring handling of the above exception, another exception occurred:\n",
        )
    } else {
        return Ok(());
    };

    write_chained_context(writer, older)?;
    write_single_exception(writer, older)?;
    writeln!(writer, "{}", banner)?;
    Ok(())
}

/// Print one W_ExceptionObject as `Traceback ...\n<frames>\n<header>`.
/// Used by `write_chained_context` when recursing through chained
/// __cause__ / __context__ predecessors.
fn write_single_exception<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    writeln!(writer, "Traceback (most recent call last):")?;
    write_traceback_chain_from_exc(writer, exc)?;
    let render = render_exc_object(exc);
    writeln!(writer, "{}", render)?;
    write_exception_notes(writer, exc)
}

/// `Lib/traceback.py:_format_final_exc_line` notes block — each entry of
/// `e.__notes__` is printed on its own line after the exception
/// header.  Notes are stored in ATTR_TABLE as a list of strings by
/// `BaseException.add_note` (PEP 678).
fn write_exception_notes<W: Write>(writer: &mut W, exc: PyObjectRef) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let notes = match crate::baseobjspace::getattr(exc, "__notes__") {
        Ok(v) if !v.is_null() => v,
        _ => return Ok(()),
    };
    if !unsafe { pyre_object::is_list(notes) } {
        return Ok(());
    }
    let len = unsafe { pyre_object::w_list_len(notes) } as i64;
    for i in 0..len {
        let item = unsafe { pyre_object::w_list_getitem(notes, i) }.unwrap_or(pyre_object::PY_NULL);
        if item.is_null() {
            continue;
        }
        let s = unsafe { crate::display::py_str(item) };
        for line in s.lines() {
            writeln!(writer, "{}", line)?;
        }
    }
    Ok(())
}

/// Compose the `ExcName: msg` header for a W_ExceptionObject —
/// equivalent to `traceback.format_exception_only`'s last line.
fn render_exc_object(exc: PyObjectRef) -> String {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return String::from("<no exception>");
    }
    let kind = unsafe { pyre_object::excobject::w_exception_get_kind(exc) };
    let name = exc_kind_name(kind);
    // `str(e)` semantically — pyre stores args as a list; the
    // first arg's str repr is the message.  Empty args produces
    // bare `ExcName` per `traceback.format_exception_only`.
    let args = unsafe { pyre_object::excobject::w_exception_get_args(exc) };
    let msg = unsafe {
        if args.is_null() || !pyre_object::is_tuple(args) {
            String::new()
        } else {
            let len = pyre_object::w_tuple_len(args);
            if len == 0 {
                String::new()
            } else if len == 1 {
                let first = pyre_object::w_tuple_getitem(args, 0).unwrap_or(pyre_object::PY_NULL);
                if first.is_null() {
                    String::new()
                } else {
                    crate::display::py_str(first)
                }
            } else {
                // Multi-arg exceptions render as tuple repr — matches
                // BaseException.__str__ (`interp_exceptions.py:142`).
                let items: Vec<String> = (0..len as i64)
                    .filter_map(|i| pyre_object::w_tuple_getitem(args, i))
                    .map(|w| crate::display::py_repr(w))
                    .collect();
                format!("({})", items.join(", "))
            }
        }
    };
    if msg.is_empty() {
        name.to_string()
    } else {
        format!("{name}: {msg}")
    }
}

/// `pypy/interpreter/error.py:125-158 print_app_tb_only` parity —
/// walk the chained `W_PyTraceback` head→tail (outermost → innermost)
/// and print each frame as `File "<path>", line N, in <name>` plus the
/// source line.  Silently no-ops when `err.exc_object` is null, the
/// traceback slot is empty, or the source file can't be read.
fn write_traceback_chain<W: Write>(writer: &mut W, err: &PyError) -> std::io::Result<()> {
    if err.exc_object.is_null() {
        return Ok(());
    }
    write_traceback_chain_from_exc(writer, err.exc_object)
}

fn write_traceback_chain_from_exc<W: Write>(
    writer: &mut W,
    exc: PyObjectRef,
) -> std::io::Result<()> {
    if exc.is_null() || !unsafe { pyre_object::is_exception(exc) } {
        return Ok(());
    }
    let mut tb = unsafe { pyre_object::excobject::w_exception_get_traceback(exc) };
    while !tb.is_null() {
        if !unsafe { crate::pytraceback::is_pytraceback(tb) } {
            break;
        }
        let w_code = unsafe { crate::pytraceback::w_pytraceback_get_w_code(tb) };
        let lineno = unsafe { crate::pytraceback::w_pytraceback_get_lineno(tb) };
        let (filename, funcname) = if w_code.is_null() {
            (String::from("<unknown>"), String::from("<unknown>"))
        } else {
            // `w_code` is a GC-rooted `W_CodeObject` pointer captured
            // at `record_application_traceback` time; the inner
            // `CodeObject` lives as long as `w_code` is reachable.
            let code_obj = unsafe { crate::w_code_get_ptr(w_code) } as *const crate::CodeObject;
            if code_obj.is_null() {
                (String::from("<unknown>"), String::from("<unknown>"))
            } else {
                let code = unsafe { &*code_obj };
                (code.source_path.to_string(), code.obj_name.to_string())
            }
        };
        writeln!(
            writer,
            "  File \"{}\", line {}, in {}",
            filename, lineno, funcname
        )?;
        if let Some(line) = read_source_line(&filename, lineno) {
            let trimmed = line.trim_end_matches(['\n', '\r']);
            writeln!(writer, "    {}", trimmed.trim_start())?;
        }
        tb = unsafe { crate::pytraceback::w_pytraceback_get_w_next(tb) };
    }
    Ok(())
}

/// Open `filename` and return its `lineno`-th line (1-indexed).  Returns
/// `None` for synthetic / unreadable sources — matches PyPy's silent
/// `linecache.getline` fallback at `error.py:150`.
fn read_source_line(filename: &str, lineno: i64) -> Option<String> {
    if lineno <= 0 || filename.is_empty() || filename.starts_with('<') {
        return None;
    }
    let content = std::fs::read_to_string(filename).ok()?;
    content
        .lines()
        .nth((lineno - 1) as usize)
        .map(|s| s.to_string())
}

pub fn eprint_exception(err: &PyError, include_traceback: bool) {
    let mut stderr = std::io::stderr().lock();
    let _ = write_exception(&mut stderr, err, include_traceback);
}

pub fn get_cleared_operation_error(_space: PyObjectRef) -> OperationError {
    let _ = _space;
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn get_converted_unexpected_exception(
    _space: PyObjectRef,
    _error: &dyn std::error::Error,
) -> OperationError {
    let _ = (_space, _error);
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn decompose_valuefmt(valuefmt: &str) -> (Vec<String>, Vec<String>) {
    let mut strings = Vec::new();
    let mut formats = Vec::new();
    let mut current = String::new();

    let mut iter = valuefmt.chars().peekable();
    while let Some(ch) = iter.next() {
        if ch == '%' {
            if let Some('%') = iter.peek() {
                let _ = iter.next();
                current.push('%');
                continue;
            }
            strings.push(std::mem::take(&mut current));
            if let Some(spec) = iter.next() {
                formats.push(spec.to_string());
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        strings.push(current);
    }

    (strings, formats)
}

pub fn get_operrcls2(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    let (strings, _formats) = decompose_valuefmt(valuefmt);
    (std::ptr::null_mut(), strings)
}

#[cfg(test)]
mod tests {
    use super::{PyError, PyErrorKind, write_exception};

    #[test]
    fn render_exception_omits_empty_message_separator() {
        let err = PyError::new(PyErrorKind::StopIteration, "");
        assert_eq!(err.render_exception(), "StopIteration");
    }

    #[test]
    fn write_exception_includes_traceback_header() {
        let err = PyError::type_error("bad operand");
        let mut out = Vec::new();
        write_exception(&mut out, &err, true).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("Traceback (most recent call last):"));
        assert!(text.contains("TypeError: bad operand"));
    }
}

pub fn get_operr_class(valuefmt: &str) -> (PyObjectRef, Vec<String>) {
    get_operrcls2(valuefmt)
}

pub fn oefmt(w_type: PyObjectRef, valuefmt: &str, _args: impl std::fmt::Display) -> OperationError {
    let _ = valuefmt;
    let _ = format!("{}", _args);
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn debug_print(text: &str, file: Option<&mut dyn Write>, _newline: bool) {
    if let Some(file) = file {
        let _ = file.write_all(text.as_bytes());
    }
}

pub fn exception_from_errno(
    _space: PyObjectRef,
    w_type: PyObjectRef,
    _errno: i32,
) -> OperationError {
    let _ = _space;
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn exception_from_saved_errno(_space: PyObjectRef, w_type: PyObjectRef) -> OperationError {
    let _ = _space;
    OperationError::new(w_type, std::ptr::null_mut())
}

pub fn new_exception_class(
    _space: PyObjectRef,
    _name: &str,
    _bases: Option<PyObjectRef>,
    _dict: Option<PyObjectRef>,
) -> PyObjectRef {
    let _ = (_space, _name, _bases, _dict);
    std::ptr::null_mut()
}

pub fn wrap_oserror2(
    _space: PyObjectRef,
    _error: &dyn std::error::Error,
    _filename: Option<PyObjectRef>,
    _exception_class: Option<PyObjectRef>,
) -> OperationError {
    let _ = (_filename, _exception_class, _error);
    let _ = _space;
    OperationError::new(std::ptr::null_mut(), std::ptr::null_mut())
}

pub fn wrap_oserror(
    space: PyObjectRef,
    error: &dyn std::error::Error,
    _filename: Option<&str>,
    w_exception_class: Option<PyObjectRef>,
) -> OperationError {
    let _ = _filename;
    wrap_oserror2(space, error, None, w_exception_class)
}
