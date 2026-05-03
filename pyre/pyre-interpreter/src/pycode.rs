//! W_CodeObject — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;

/// Compatibility alias for PyPy's `PyCode` type.
pub type PyCode = W_CodeObject;

/// Compatibility marker for malformed bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeCorruption;

impl From<BytecodeCorruption> for crate::PyError {
    fn from(_: BytecodeCorruption) -> Self {
        crate::PyError::new(
            crate::PyErrorKind::BytecodeCorruption,
            "bytecode corruption",
        )
    }
}

/// Compatibility container for code-hook caching state.
#[derive(Debug, Default)]
pub struct CodeHookCache {
    _code_hook: Option<PyObjectRef>,
}

/// Type descriptor for code objects.
pub static CODE_TYPE: PyType = pyre_object::pyobject::new_pytype("code");

/// Python code object wrapper.
///
/// Stores an opaque pointer to the bytecode CodeObject. The pointer is
/// `Box::into_raw`'d from a cloned CodeObject, so we own the allocation.
#[repr(C)]
pub struct W_CodeObject {
    pub ob_header: PyObject,
    /// Opaque pointer to a `CodeObject` (owned via Box::into_raw).
    pub code_ptr: *const (),
    /// PyPy: `PyCode.w_globals`.
    pub w_globals: *mut crate::DictStorage,
    /// pycode.py:96 `self.hidden_applevel = hidden_applevel`.
    pub hidden_applevel: bool,
}

/// Field offset of `code_ptr` within `W_CodeObject`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(W_CodeObject, code_ptr);
/// Field offset of `w_globals` within `W_CodeObject`.
pub const CODE_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(W_CodeObject, w_globals);

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &W_CodeObject) -> PyObjectRef {
    let _ = _code;
    pyre_object::w_none()
}

/// pycode.py:637-659 _compute_args_as_cellvars
pub fn _compute_args_as_cellvars(
    varnames: &[String],
    cellvars: &[String],
    argcount: usize,
) -> Vec<isize> {
    let mut args_as_cellvars = Vec::new();
    for i in 0..cellvars.len() {
        let cellname = &cellvars[i];
        for j in 0..argcount {
            if *cellname == varnames[j] {
                while args_as_cellvars.len() < i {
                    args_as_cellvars.push(-1isize);
                }
                args_as_cellvars.push(j as isize);
            }
        }
    }
    args_as_cellvars
}

#[inline]
pub fn _code_const_eq(_space: PyObjectRef, w_a: PyObjectRef, w_b: PyObjectRef) -> bool {
    let _ = _space;
    std::ptr::eq(w_a, w_b)
}

#[inline]
pub fn _convert_const(_space: PyObjectRef, w_a: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    w_a
}

/// Allocate a new W_CodeObject wrapping an opaque code pointer.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` obtained
/// via `Box::into_raw`.
pub fn w_code_new(code_ptr: *const ()) -> PyObjectRef {
    let obj = Box::new(W_CodeObject {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr,
        w_globals: std::ptr::null_mut(),
        hidden_applevel: false,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// Box a cloned compiler code object into a heap Python code wrapper.
pub fn box_code_constant(code: &crate::CodeObject) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
    w_code_new(code_ptr)
}

/// Extract the opaque code pointer from a known W_CodeObject.
///
/// # Safety
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const W_CodeObject)).code_ptr }
}

/// PyPy: `PyCode.w_globals`.
#[inline]
pub unsafe fn w_code_get_w_globals(obj: PyObjectRef) -> *mut crate::DictStorage {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*(obj as *const W_CodeObject)).w_globals }
}

/// PyPy: `PyCode.w_globals = w_globals`.
#[inline]
pub unsafe fn w_code_set_w_globals(obj: PyObjectRef, w_globals: *mut crate::DictStorage) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut W_CodeObject)).w_globals = w_globals;
    }
}

/// PyPy: `PyCode.hidden_applevel`.
#[inline]
pub unsafe fn w_code_hidden_applevel(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { (*(obj as *const W_CodeObject)).hidden_applevel }
}

/// PyPy: `PyCode.frame_stores_global(w_globals)`.
#[inline]
pub unsafe fn w_code_frame_stores_global(
    obj: PyObjectRef,
    w_globals: *mut crate::DictStorage,
) -> bool {
    if obj.is_null() {
        return false;
    }
    let code = unsafe { &mut *(obj as *mut W_CodeObject) };
    if code.w_globals.is_null() {
        code.w_globals = w_globals;
        return false;
    }
    !std::ptr::eq(code.w_globals, w_globals)
}

/// Check if an object is a code object.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CODE_TYPE) }
}
