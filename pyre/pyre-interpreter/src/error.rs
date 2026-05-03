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
    /// PRE-EXISTING-ADAPTATION: the W_BaseException fast path
    /// (error.py:228-232 `w_value.w_traceback = tb`) requires the
    /// `w_traceback` slot to be present on `W_ExceptionObject`
    /// (`pyre-object/excobject.rs` carries only `kind` + `message`
    /// today).  The PyPy W_BaseException expansion epic is the
    /// prerequisite; until it lands, the fallback `space.setattr(w_value,
    /// "__traceback__", tb)` path covers both branches semantically —
    /// PyPy's two arms differ only in optimisation.
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
                // error.py:225-236 traceback attach.
                if let Some(tb) = self._application_traceback {
                    // PRE-EXISTING-ADAPTATION (see method docstring): the
                    // W_BaseException-typed fast path is unreachable until
                    // pyre grows the `w_traceback` slot on
                    // `W_ExceptionObject`.  Until then both arms fall through
                    // to the generic setattr escape (error.py:235-236), which
                    // PyPy uses unconditionally for any non-W_BaseException
                    // value.  No semantic loss — only the slot-write
                    // optimisation is deferred.
                    let _ = crate::baseobjspace::setattr(w_value, "__traceback__", tb);
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

    /// pypy/interpreter/error.py:418-427 `chain_exceptions`.
    ///
    /// PRE-EXISTING-ADAPTATION (Slice 3 of OperationError port,
    /// blocked): line-by-line port requires
    ///   - `W_BaseException.descr_setcontext(space, w_context)` to
    ///     write the `__context__` slot
    ///   - `_break_context_cycle` helper (error.py:443-)
    ///   - `W_BaseException` instance-of check
    ///
    /// Pyre's `W_ExceptionObject` (pyre-object/excobject.rs:59) carries
    /// only `kind` + `message` — no `w_context` / `w_cause` /
    /// `w_traceback` slots.  Adding those fields is the
    /// `W_BaseException` expansion epic (pypy/module/exceptions/
    /// interp_exceptions.py W_BaseException, ~200 LOC).
    ///
    /// Until that epic lands, `chain_exceptions` cannot be ported.
    /// `record_context` / `chain_exceptions_from_cause` (error.py:406,
    /// 429) inherit the same blocker.
    pub fn chain_exceptions(
        &mut self,
        _space: PyObjectRef,
        _context: &mut OperationError,
    ) -> Result<(), PyError> {
        // Stub: return Ok(()) until the W_BaseException expansion epic
        // wires up `descr_setcontext` and the per-exception
        // w_context / w_cause / w_traceback slots.
        Ok(())
    }
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
}

impl PyError {
    pub fn new(kind: PyErrorKind, message: impl Into<String>) -> Self {
        PyError {
            kind,
            message: message.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn type_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::TypeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn value_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ValueError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn zero_division(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ZeroDivisionError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn overflow_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::OverflowError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::RuntimeError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn key_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::KeyError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn os_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::OSError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
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
        }
    }

    /// pypy/module/_weakref/interp__weakref.py:347 — raised by `force()`
    /// when the referent of a proxy is no longer alive.
    pub fn reference_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::ReferenceError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
        }
    }

    pub fn recursion_error(msg: impl Into<String>) -> Self {
        PyError {
            kind: PyErrorKind::RecursionError,
            message: msg.into(),
            exc_object: std::ptr::null_mut(),
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
        }
    }

    pub fn stop_iteration() -> Self {
        PyError {
            kind: PyErrorKind::StopIteration,
            message: String::new(),
            exc_object: std::ptr::null_mut(),
        }
    }

    /// Convert to a W_ExceptionObject for pushing onto the value stack.
    /// Reuses the cached object from from_exc_object() if available.
    pub fn to_exc_object(&self) -> PyObjectRef {
        if !self.exc_object.is_null() {
            return self.exc_object;
        }
        w_exception_new(self.to_exc_kind(), &self.message)
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
        writeln!(writer, "Traceback (most recent call last):")?;
        writeln!(writer, "  {}", err.render_exception())
    } else {
        writeln!(writer, "{}", err.render_exception())
    }
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
