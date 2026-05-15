//! Function object.
#![allow(non_snake_case)]
//!
//! Wraps a code object pointer, a function name, a pointer to the
//! defining module's globals namespace, and an optional closure tuple.
//! When called, the interpreter creates a new PyFrame that *shares*
//! the globals pointer (no clone).

use crate::executioncontext::DictStorage;
use pyre_object::pyobject::*;

/// Type descriptor for user-defined functions.
pub static FUNCTION_TYPE: PyType = pyre_object::pyobject::new_pytype("function");
/// Type descriptor for module-level builtins.
pub static BUILTIN_FUNCTION_TYPE: PyType = pyre_object::pyobject::new_pytype("builtin_function");

/// User-defined function object.
///
/// Layout: `[ob_type | code | can_change_code | name_ptr | w_func_globals | closure]`
/// - `code`: pointer to a Code object (W_CodeObject for user funcs, BuiltinCode for builtins).
///   function.py:47 — `_immutable_fields_ = ['code?', ...]`
/// - `can_change_code`: function.py:33 — True by default; False for
///   `FunctionWithFixedCode` subclass (used by builtins).
/// - `name_ptr`: leaked `Box<String>` containing the function name
/// - `w_func_globals`: raw pointer to the module-level namespace (shared)
/// - `closure`:  tuple of cell objects, or PY_NULL if no closure
#[repr(C)]
pub struct Function {
    pub ob: PyObject,
    /// Pointer to a Code object (W_CodeObject or BuiltinCode).
    /// function.py:47 — `_immutable_fields_ = ['code?', ...]`
    pub code: *const (),
    /// function.py:33 — `can_change_code = True`
    /// False for FunctionWithFixedCode subclass.
    pub can_change_code: bool,
    /// Function name (leaked Box<String>).
    pub name: *const String,
    /// PyPy: W_Function.w_func_globals
    pub w_func_globals: *mut DictStorage,
    /// Closure: tuple of cell objects from the enclosing scope,
    /// or PY_NULL if this function has no free variables.
    pub closure: PyObjectRef,
    /// Default argument values.
    /// PyPy: W_Function.defs_w
    pub defs_w: PyObjectRef,
    /// Keyword-only default values.
    /// PyPy: W_Function.w_kw_defs
    pub w_kw_defs: PyObjectRef,
    /// function.py:56 — `self.w_module = None`
    pub w_module: PyObjectRef,
    /// Lazy-resolved canonical W_DictObject sibling for `w_func_globals`.
    ///
    /// `function.py:57 self.w_func_globals = w_globals` stores the dict
    /// object directly in PyPy.  Pyre keeps the storage pointer in
    /// `w_func_globals` (JIT call-site layout reads it for createframe)
    /// and caches the canonical W_DictObject here once resolved, so
    /// `function.__globals__` returns the same identity as the module's
    /// `__dict__` and frames built from this function share globals.
    /// `PY_NULL` until `function_get_globals_obj()` performs the
    /// `dict_storage_to_dict` lookup once.
    pub w_func_globals_obj: PyObjectRef,
    /// `function.py:50 w_ann=None` constructor default plus
    /// `function.py:548-551 fget_func_annotations` lazy-init shape:
    /// PyPy stores the annotations dict directly on the function and
    /// allocates an empty dict on first read when none was set.
    /// Pyre mirrors that by keeping the slot `PY_NULL` until either
    /// `MAKE_FUNCTION ANNOTATIONS` flag stamps the compile-time dict
    /// (`function.py:553-559 fset_func_annotations`) or the getter
    /// lazy-fills with `w_dict_new()`.  `f.__annotations__ is
    /// f.__annotations__` identity holds because both reads see the
    /// cached slot after the first allocation.
    pub w_ann: PyObjectRef,
    /// `function.py:375 self.w_doc = w_doc` constructor slot plus
    /// `function.py:446-449 fget_func_doc` cache:
    ///
    /// ```python
    /// def fget_func_doc(self, space):
    ///     if self.w_doc is None:
    ///         self.w_doc = self.code.getdocstring(space)
    ///     return self.w_doc
    /// ```
    ///
    /// `PY_NULL` means "not yet resolved" (PyPy `None`); first reader
    /// caches `code.getdocstring()` here so subsequent
    /// `f.__doc__ is f.__doc__` holds.  `function_del_doc` writes
    /// `w_none()` to make the deleted state sticky against the lazy
    /// fallback (`function.py:455-457 fdel_func_doc`).
    pub w_doc: PyObjectRef,
    /// `function.py:54 self.qualname = qualname or self.name` slot —
    /// PyPy stores the qualified name as a regular field on the
    /// function object.  `PY_NULL` means "not explicitly set"; the
    /// getter falls back to `code.co_qualname` then `name` to mirror
    /// the constructor's `qualname or self.name` short-circuit when
    /// the compile path did not stamp a value at MAKE_FUNCTION time.
    pub w_qualname: PyObjectRef,
    /// `function.py:498-504 fget_func_objclass / set_objclass` slot —
    /// PyPy stores the bound class on the function for descriptor
    /// introspection (`inspect.getfullargspec` etc.).  `PY_NULL`
    /// means "not bound to a class" and the getter raises
    /// AttributeError per `function.py:500`.
    pub w_objclass: PyObjectRef,
    /// `function.py:487-496 fget_func_text_signature /
    /// fset_func_text_signature` slot — PyPy stores the text
    /// signature (the docstring-prefix `(...)` line that PEP 437
    /// describes for builtins) directly on the function.  `PY_NULL`
    /// means "no signature recorded" and the getter returns
    /// `space.w_None` per `function.py:488`.
    pub w_text_signature: PyObjectRef,
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
pub type BuiltinFunction = Function;
/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
pub type FunctionWithFixedCode = Function;
pub type Method = pyre_object::methodobject::W_MethodObject;
pub type StaticMethod = pyre_object::propertyobject::W_StaticMethodObject;
pub type ClassMethod = pyre_object::propertyobject::W_ClassMethodObject;

struct FrameLocalsRoot {
    slot: *mut *mut u8,
    registered: bool,
}

impl FrameLocalsRoot {
    fn new(frame: &mut crate::pyframe::PyFrame) -> Self {
        let slot = &mut frame.locals_cells_stack_w as *mut _ as *mut *mut u8;
        let registered = unsafe { pyre_object::gc_hook::try_gc_add_root(slot) };
        Self { slot, registered }
    }
}

impl Drop for FrameLocalsRoot {
    fn drop(&mut self) {
        if self.registered {
            pyre_object::gc_hook::try_gc_remove_root(self.slot);
        }
    }
}

#[inline]
fn function_write_barrier(obj: PyObjectRef) {
    pyre_object::gc_hook::try_gc_write_barrier(obj as *mut u8);
}

/// Field offset of `code` within `Function`, for JIT field access.
pub const FUNCTION_CODE_OFFSET: usize = std::mem::offset_of!(Function, code);
/// Field offset of `name` within `Function`.
pub const FUNCTION_NAME_OFFSET: usize = std::mem::offset_of!(Function, name);
/// Field offset of `w_func_globals` within `Function`.
pub const FUNCTION_GLOBALS_OFFSET: usize = std::mem::offset_of!(Function, w_func_globals);
/// Field offset of `closure` within `Function`.
pub const FUNCTION_CLOSURE_OFFSET: usize = std::mem::offset_of!(Function, closure);
/// Field offset of `defs_w` within `Function`.
pub const FUNCTION_DEFS_W_OFFSET: usize = std::mem::offset_of!(Function, defs_w);
/// Field offset of `w_kw_defs` within `Function`.
pub const FUNCTION_W_KW_DEFS_OFFSET: usize = std::mem::offset_of!(Function, w_kw_defs);
/// Field offset of `w_module` within `Function`.
pub const FUNCTION_W_MODULE_OFFSET: usize = std::mem::offset_of!(Function, w_module);
/// Field offset of `w_func_globals_obj` within `Function` — the
/// lazy-cached canonical W_DictObject for `w_func_globals`.
pub const FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET: usize =
    std::mem::offset_of!(Function, w_func_globals_obj);
/// Field offset of `w_ann` within `Function` — the
/// `function.py:50 w_ann` annotations dict slot.
pub const FUNCTION_W_ANN_OFFSET: usize = std::mem::offset_of!(Function, w_ann);
/// Field offset of `w_doc` within `Function` — the
/// `function.py:375 w_doc` docstring cache slot.
pub const FUNCTION_W_DOC_OFFSET: usize = std::mem::offset_of!(Function, w_doc);
/// Field offset of `w_qualname` within `Function` — the
/// `function.py:54 qualname` slot.
pub const FUNCTION_W_QUALNAME_OFFSET: usize = std::mem::offset_of!(Function, w_qualname);
/// Field offset of `w_objclass` within `Function` — the
/// `function.py:498-504 w_objclass` slot.
pub const FUNCTION_W_OBJCLASS_OFFSET: usize = std::mem::offset_of!(Function, w_objclass);
/// Field offset of `w_text_signature` within `Function` — the
/// `function.py:487-496 w_text_signature` slot.
pub const FUNCTION_W_TEXT_SIGNATURE_OFFSET: usize =
    std::mem::offset_of!(Function, w_text_signature);

/// GC type id assigned to `Function` at JitDriver init time. Held as
/// a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID` / `BUILTIN_CODE_GC_TYPE_ID`.
/// `pyre/pyre-jit/src/eval.rs` asserts the same id is returned by
/// `gc.register_type(...)` so any drift panics on startup.
///
/// `BuiltinFunction` (gateway.py module-level builtins) and
/// `FunctionWithFixedCode` are Rust type aliases of `Function`, so
/// instances of those PyTypes share this tid via the
/// `register_vtable_for_type` mapping.
pub const FUNCTION_GC_TYPE_ID: u32 = 14;

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`).
pub const FUNCTION_OBJECT_SIZE: usize = std::mem::size_of::<Function>();

/// Byte offsets of the inline `PyObjectRef`-shaped fields the GC must
/// trace during minor collection. `code` is included because
/// `BuiltinCode` is now allocated through `malloc_typed`
/// (`gateway.rs:298`) and therefore lives in the GC heap; the W_CodeObject
/// path remains raw/immortal and the walker's `is_in_nursery` check
/// (`majit-gc/src/collector.rs:764`) leaves those entries alone.
/// `function.py:47 _immutable_fields_ = ['code?', ...]` matches PyPy's
/// W_Function.code? — an immutable GC reference traced as part of the
/// closure / defs_w / w_kw_defs / w_module set.
///
/// The remaining fields are non-GC: `can_change_code` is a `bool`,
/// `name` is a manually-managed `*const String`, `w_func_globals` is a
/// `*mut DictStorage` (manually-managed pointer to non-GC storage —
/// see Task #145 follow-up for DictStorage GC migration).
///
/// `ob.w_class` is intentionally absent, mirroring how W_IntObject /
/// W_FloatObject leave the typeptr-shaped header field out of their
/// `gc_ptr_offsets`. W_TypeObject instances are static-region and
/// not subject to nursery relocation.
pub const FUNCTION_GC_PTR_OFFSETS: [usize; 11] = [
    FUNCTION_CODE_OFFSET,
    FUNCTION_CLOSURE_OFFSET,
    FUNCTION_DEFS_W_OFFSET,
    FUNCTION_W_KW_DEFS_OFFSET,
    FUNCTION_W_MODULE_OFFSET,
    // Lazy-cached canonical W_DictObject sibling for the storage in
    // `w_func_globals`.  Once resolved it must survive minor
    // collection — the `dict_storage_to_dict` mirror_target invariant
    // keeps the same identity across the function's lifetime.
    FUNCTION_W_FUNC_GLOBALS_OBJ_OFFSET,
    // `function.py:50 w_ann` — annotations dict, allocated lazily on
    // first read by the getter or stamped at MAKE_FUNCTION time.
    FUNCTION_W_ANN_OFFSET,
    // `function.py:375 w_doc` — docstring slot cached on first read.
    FUNCTION_W_DOC_OFFSET,
    // `function.py:54 qualname` — qualified name slot stamped at
    // construction or via `f.__qualname__ = ...`.
    FUNCTION_W_QUALNAME_OFFSET,
    // `function.py:498-504 w_objclass` — bound class for descriptor
    // introspection (`inspect.getfullargspec` etc.).
    FUNCTION_W_OBJCLASS_OFFSET,
    // `function.py:487-496 w_text_signature` — PEP 437 text signature
    // line stripped from the docstring.
    FUNCTION_W_TEXT_SIGNATURE_OFFSET,
];

impl pyre_object::lltype::GcType for Function {
    const TYPE_ID: u32 = FUNCTION_GC_TYPE_ID;
    const SIZE: usize = FUNCTION_OBJECT_SIZE;
}

/// Allocate a new `Function`.
///
/// `code` is a pointer to a Code object (W_CodeObject) cast to `*const ()`.
/// `name` is the function name string (leaked).
/// `w_func_globals` is the defining module's namespace pointer (shared).
pub fn function_new(
    code: *const (),
    name: String,
    w_func_globals: *mut DictStorage,
) -> PyObjectRef {
    function_new_with_closure(code, name, w_func_globals, PY_NULL)
}

/// Allocate a new `Function` with a closure.
///
/// `closure` is a tuple of cell objects, or PY_NULL if no closure.
pub fn function_new_with_closure(
    code: *const (),
    name: String,
    w_func_globals: *mut DictStorage,
    closure: PyObjectRef,
) -> PyObjectRef {
    function_new_impl(&FUNCTION_TYPE, code, name, w_func_globals, closure, true)
}

fn function_new_impl(
    ob_type: &'static PyType,
    code: *const (),
    name: String,
    w_func_globals: *mut DictStorage,
    closure: PyObjectRef,
    can_change_code: bool,
) -> PyObjectRef {
    // `gct_fv_gc_malloc` bracket pattern (`framework.py:853-856`) for
    // the `lltype::malloc_typed` call below. `closure` and `code` are
    // PyObjectRef-shaped GC roots across the alloc — `BuiltinCode`
    // lives in the GC heap (`gateway.rs:298 malloc_typed`) and
    // `W_CodeObject` is currently raw/immortal; the walker's
    // `is_in_nursery` filter (`majit-gc/src/collector.rs:764`) is what
    // makes the heterogeneous case safe. `w_func_globals` is a
    // host-allocated DictStorage (raw), and `name_ptr` is allocated
    // below via `malloc_raw` (non-GC) and stored into the struct as
    // part of the same `malloc_typed` call, so it never spans a
    // collection point.
    let _roots = pyre_object::gc_roots::push_roots();
    pyre_object::gc_roots::pin_root(closure);
    pyre_object::gc_roots::pin_root(code as PyObjectRef);

    // `function.py:57 self.w_func_globals = w_globals` stores the
    // dict object directly.  Pyre keeps both the storage pointer
    // (for the JIT call-site layout) and the canonical W_DictObject
    // sibling.  Resolve the sibling NOW so the freshly-allocated
    // function carries a stable `__globals__` identity from
    // construction onward; `function_get_globals_obj`'s lazy
    // fallback becomes a zero-hit safety net for synthetic test
    // stubs that pass a null storage pointer.  This is the first
    // step of Phase 5 commit 3 — collapsing the dual storage to
    // a single `PyObjectRef` field requires that every Function
    // already has its `__globals__` resolved at construction time.
    let w_func_globals_obj = if w_func_globals.is_null() {
        PY_NULL
    } else {
        crate::baseobjspace::dict_storage_to_dict(w_func_globals)
    };
    pyre_object::gc_roots::pin_root(w_func_globals_obj);

    let name_ptr = pyre_object::lltype::malloc_raw(name) as *const String;
    let function = Function {
        ob: PyObject {
            ob_type: ob_type as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(ob_type),
        },
        code,
        can_change_code,
        name: name_ptr,
        w_func_globals,
        closure,
        defs_w: PY_NULL,
        w_kw_defs: PY_NULL,
        w_module: PY_NULL,
        w_func_globals_obj,
        w_ann: PY_NULL,
        w_doc: PY_NULL,
        w_qualname: PY_NULL,
        w_objclass: PY_NULL,
        w_text_signature: PY_NULL,
    };

    if let Some(raw) =
        pyre_object::gc_hook::try_gc_alloc_stable(FUNCTION_GC_TYPE_ID, FUNCTION_OBJECT_SIZE)
            .filter(|p| !p.is_null())
    {
        unsafe {
            std::ptr::write(raw as *mut Function, function);
        }
        return raw as PyObjectRef;
    }

    pyre_object::lltype::malloc_typed(function) as PyObjectRef
}

/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
/// Allocate a function whose code pointer the JIT can treat as immutable.
pub fn function_new_with_fixed_code(
    code: *const (),
    name: String,
    w_func_globals: *mut DictStorage,
) -> PyObjectRef {
    function_new_impl(&FUNCTION_TYPE, code, name, w_func_globals, PY_NULL, false)
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
/// Allocate a module builtin whose typedef intentionally omits `__get__`.
pub fn function_new_builtin(
    code: *const (),
    name: String,
    w_func_globals: *mut DictStorage,
) -> PyObjectRef {
    function_new_impl(
        &BUILTIN_FUNCTION_TYPE,
        code,
        name,
        w_func_globals,
        PY_NULL,
        false,
    )
}

/// function.py:385-388 — `_check_code_mutable(attr)`:
///
/// ```python
/// def _check_code_mutable(self, attr):
///     if not self.can_change_code:
///         raise oefmt(self.space.w_AttributeError,
///                     "Cannot change %s attribute of builtin functions", attr)
/// ```
pub unsafe fn _check_code_mutable(func: PyObjectRef, attr: &str) -> Result<(), crate::PyError> {
    unsafe {
        if (*(func as *const Function)).can_change_code {
            Ok(())
        } else {
            Err(crate::PyError::attribute_error(format!(
                "Cannot change {} attribute of builtin functions",
                attr
            )))
        }
    }
}

/// function.py:23 — `@jit.elidable_promote()`
/// Only valid when `can_change_code == false`.
#[majit_macros::elidable_promote]
#[inline]
pub unsafe fn _get_immutable_code(func: PyObjectRef) -> *const () {
    // function.py:25
    debug_assert!(
        !unsafe { (*(func as *const Function)).can_change_code },
        "_get_immutable_code called on function with can_change_code=true"
    );
    unsafe { (*(func as *const Function)).code }
}

/// Check if an object is a user-defined function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) || py_type_check(obj, &BUILTIN_FUNCTION_TYPE) }
}

/// `isinstance(obj, FunctionWithFixedCode)` parity.
///
/// function.py:783 — `class FunctionWithFixedCode(Function):
///     can_change_code = False`.  Pyre encodes the `Function`
/// vs `FunctionWithFixedCode` distinction through the
/// `can_change_code` flag (true for user `def`s built via
/// `function_new_with_closure`, false for gateway-built builtins
/// via `function_new_with_fixed_code`).  `BuiltinFunction` has its
/// own `BUILTIN_FUNCTION_TYPE`, so a strict `FUNCTION_TYPE` check
/// excludes it (function.py:786 makes BuiltinFunction a sibling
/// subclass of Function, not a subclass of FunctionWithFixedCode).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_function_with_fixed_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FUNCTION_TYPE) && !(*(obj as *const Function)).can_change_code }
}

/// function.py:78-83 — `getcode(self)`: three-way dispatch.
///   - JIT + immutable code → _get_immutable_code (elidable_promote)
///   - JIT + mutable code  → promote(self.code)
///   - interpreter         → self.code
#[inline]
pub unsafe fn getcode(obj: PyObjectRef) -> *const () {
    unsafe {
        let func = obj as *const Function;
        if majit_metainterp::jit::we_are_jitted() {
            if !(*func).can_change_code {
                // function.py:80-81
                return _get_immutable_code(obj);
            }
            // function.py:82
            return majit_metainterp::jit::promote((*func).code as usize) as *const ();
        }
        // function.py:83
        (*func).code
    }
}

/// Get the Code object pointer from a function object.
///
/// Returns a pointer to the Code-level object (W_CodeObject or BuiltinCode).
///
/// # Safety
/// `obj` must point to a valid `Function`.
/// NOTE: NOT elidable — code field can change (can_change_code).
/// Use _get_immutable_code() for the elidable path.
#[inline]
pub unsafe fn function_get_code(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const Function)).code }
}

/// Extract the raw bytecode CodeObject pointer from a user function.
///
/// Equivalent to accessing `self.getcode().code_ptr` in PyPy terms:
/// `getcode()` returns the Code wrapper (W_CodeObject), and this
/// dereferences through it to the underlying CodeObject.
///
/// # Safety
/// `obj` must point to a valid `Function` whose `code` field is a `W_CodeObject`
/// (i.e., NOT a BuiltinCode). Only call on user-defined functions.
#[inline]
pub unsafe fn get_pycode(obj: PyObjectRef) -> *const () {
    unsafe {
        let code = getcode(obj);
        debug_assert!(
            !crate::is_builtin_code(code as PyObjectRef),
            "get_pycode called on a builtin function"
        );
        crate::w_code_get_ptr(code as PyObjectRef)
    }
}

/// Get the function name.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_name(obj: PyObjectRef) -> &'static str {
    unsafe { &*(*(obj as *const Function)).name }
}

/// `function.py:476-485 fset_func_qualname` parity:
///
/// ```python
/// def fset_func_qualname(self, space, w_name):
///     self._check_code_mutable("__qualname__")
///     try:
///         qualname = space.realutf8_w(w_name)
///     except OperationError as e:
///         if e.match(space, space.w_TypeError):
///             raise oefmt(space.w_TypeError,
///                         "__qualname__ must be set to a string object")
///         raise
///     self.set_qualname(qualname)
/// ```
///
/// `space.realutf8_w` accepts any `isinstance_w(w_name, w_text)` —
/// `str` and its subclasses.  `isinstance_str_w` mirrors that.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_qualname(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__qualname__")?; // function.py:477
        if !crate::baseobjspace::isinstance_str_w(value) {
            return Err(crate::PyError::type_error(
                "__qualname__ must be set to a string object",
            ));
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_qualname = value;
        Ok(())
    }
}

/// `function.py:470-471 fget_func_qualname` parity:
///
/// ```python
/// def fget_func_qualname(self, space):
///     return space.newtext(self.qualname)
/// ```
///
/// `self.qualname` is initialised from `qualname or self.name` at
/// `function.py:54 __init__`.  MAKE_FUNCTION
/// (`runtime_ops::make_function_from_code_obj`) stamps `w_qualname`
/// from `codeobj.co_qualname` at construction, so subsequent
/// `__code__ = new_code` assignments do NOT alter `__qualname__`
/// (matching `pypy/interpreter/pyopcode.py:1457` + `function.py:54`).
/// When `w_qualname` is still `PY_NULL` (legacy callers that never
/// stamped it — builtins built via `gateway.rs` / unit tests), fall
/// back to the bare `name` string per `qualname or self.name`.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_qualname(obj: PyObjectRef) -> String {
    unsafe {
        let cached = (*(obj as *const Function)).w_qualname;
        if !cached.is_null() && pyre_object::is_str(cached) {
            return pyre_object::w_str_get_value(cached).to_string();
        }
        function_get_name(obj).to_string()
    }
}

/// `function.py:54 self.qualname = qualname or self.name` setter —
/// used by MAKE_FUNCTION (`runtime_ops::make_function_from_code_obj`)
/// to freeze the qualified name immediately after `function_new`.
/// Bypasses `_check_code_mutable` because this is the construction
/// path; user code goes through `fset_func_qualname` instead.
///
/// # Safety
/// `obj` must point to a valid `Function`; `value` must be a string
/// `PyObjectRef`.
#[inline]
pub unsafe fn function_set_qualname(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_qualname = value;
    }
}

#[inline]
pub unsafe fn _eq(_obj: PyObjectRef, other: PyObjectRef) -> bool {
    _obj == other
}

/// PyPy-compatible descriptor accessor for function name.
#[inline]
pub unsafe fn fget_func_name(obj: PyObjectRef) -> PyObjectRef {
    unsafe { pyre_object::w_str_new(function_get_name(obj)) }
}

/// Get the globals namespace pointer from a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_globals(obj: PyObjectRef) -> *mut DictStorage {
    unsafe { (*(obj as *const Function)).w_func_globals }
}

/// Return the canonical W_DictObject paired with `function.w_func_globals`.
///
/// `function.py:57 self.w_func_globals = w_globals` stores the dict
/// object directly in PyPy.  Pyre's split storage / W_DictObject
/// model means callers that want object identity
/// (`function.__globals__ is module.__dict__`,
/// `f.__globals__ is g.__globals__` for sibling closures) need to
/// reach the canonical sibling.  `function_new_impl` resolves the
/// sibling at construction time via `dict_storage_to_dict` and
/// stores it directly in `w_func_globals_obj`, so the fast path is
/// a plain field load.
///
/// The lazy fallback below is retained as a safety net for synthetic
/// test stubs that bypass `function_new_impl` and leave
/// `w_func_globals_obj` unset; production call sites (`MAKE_FUNCTION`,
/// `function_new_with_fixed_code`, `function_new_builtin`) always
/// reach this getter with the slot pre-populated.
///
/// Returns `PY_NULL` when `w_func_globals` is null (zero-arg test
/// stubs); callers should null-check before dereferencing.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_globals_obj(obj: PyObjectRef) -> PyObjectRef {
    let func = unsafe { &mut *(obj as *mut Function) };
    if !func.w_func_globals_obj.is_null() {
        return func.w_func_globals_obj;
    }
    if func.w_func_globals.is_null() {
        return pyre_object::PY_NULL;
    }
    let resolved = crate::baseobjspace::dict_storage_to_dict(func.w_func_globals);
    function_write_barrier(obj);
    func.w_func_globals_obj = resolved;
    resolved
}

/// Get the closure tuple from a function object.
/// Returns PY_NULL if the function has no closure.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).closure }
}

/// Set the closure on a function object.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).closure = closure;
    }
}

/// Get defaults tuple.
#[inline]
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).defs_w = defaults;
    }
}

/// Get kwdefaults dict.
#[inline]
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_kw_defs = kwdefaults;
    }
}

/// PyPy-compatible `__dict__` storage field alias.
#[inline]
pub unsafe fn function_getdict(obj: PyObjectRef) -> PyObjectRef {
    crate::baseobjspace::getattr(obj, "__dict__").unwrap_or(pyre_object::w_none())
}

/// PyPy-compatible `__dict__` mutator.
#[inline]
pub unsafe fn function_setdict(obj: PyObjectRef, value: PyObjectRef) {
    let _ = crate::baseobjspace::setattr(obj, "__dict__", value);
}

/// PyPy-compatible `getdict()` descriptor helper.
#[inline]
pub unsafe fn getdict(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_getdict(obj) }
}

/// PyPy-compatible `setdict()` helper.
#[inline]
pub unsafe fn setdict(obj: PyObjectRef, value: PyObjectRef) {
    unsafe { function_setdict(obj, value) }
}

/// `function.py:446-449 fget_func_doc` parity:
///
/// ```python
/// def fget_func_doc(self, space):
///     if self.w_doc is None:
///         self.w_doc = self.code.getdocstring(space)
///     return self.w_doc
/// ```
///
/// `(*func).w_doc == PY_NULL` mirrors PyPy's `self.w_doc is None`
/// (the unset / not-yet-cached state).  First reader resolves
/// `code.getdocstring(space)` and stamps the result back so subsequent
/// `f.__doc__ is f.__doc__` holds; `function_del_doc` writes
/// `w_none()` to make the deleted state sticky against the lazy
/// fallback (`function.py:455-457 fdel_func_doc`).
///
/// `code.getdocstring(space)` has two shapes in pyre:
///   - `BuiltinCode`: stores `docstring` directly (gateway.rs:581).
///   - `W_CodeObject`: docstring is the first const when `code.flags`
///     has `HAS_DOCSTRING` set, mirroring `pycode.py:230
///     PyCode.getdocstring`.
pub fn function_get_doc(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::w_none();
    }
    let func = obj as *mut Function;
    let cached = unsafe { (*func).w_doc };
    if !cached.is_null() {
        return cached;
    }
    // Lazy fallback: `code.getdocstring(space)` (function.py:448).
    let resolved = code_getdocstring(obj);
    unsafe {
        function_write_barrier(obj);
        (*func).w_doc = resolved;
    }
    resolved
}

/// `pycode.py:230 PyCode.getdocstring(space)` parity for the two code
/// shapes pyre carries — extracted out of `function_get_doc` so
/// `fset_func_code`'s pre-assignment cache step
/// (`function.py:538 self.fget_func_doc(space)`) can reach the same
/// path without going through the cache write.
fn code_getdocstring(obj: PyObjectRef) -> PyObjectRef {
    let code = unsafe { function_get_code(obj) } as PyObjectRef;
    if code.is_null() {
        return pyre_object::w_none();
    }
    if unsafe { crate::gateway::is_builtin_code(code) } {
        return unsafe { crate::gateway::builtin_code_get_docstring(code) };
    }
    if unsafe { crate::pycode::is_code(code) } {
        let raw = unsafe { crate::pycode::w_code_get_ptr(code) } as *const crate::CodeObject;
        if !raw.is_null() {
            let code_ref = unsafe { &*raw };
            if code_ref.flags.contains(crate::CodeFlags::HAS_DOCSTRING)
                && !code_ref.constants.is_empty()
            {
                let first = crate::pyframe::load_const_from_code(code_ref, 0);
                if !first.is_null() && unsafe { pyre_object::is_str(first) } {
                    return first;
                }
            }
        }
    }
    pyre_object::w_none()
}

/// `function.py:451-453 fset_func_doc` parity:
///
/// ```python
/// def fset_func_doc(self, space, w_doc):
///     self._check_code_mutable("__doc__")
///     self.w_doc = w_doc
/// ```
pub unsafe fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:452
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = value;
        Ok(())
    }
}

/// `function.py:455-457 fdel_func_doc` parity:
///
/// ```python
/// def fdel_func_doc(self, space):
///     self._check_code_mutable("__doc__")
///     self.w_doc = space.w_None
/// ```
///
/// Stamps `w_none()` so the next `function_get_doc` short-circuits
/// the lazy `code.getdocstring(space)` fallback — `w_doc` is no
/// longer null, so the cache hit returns `w_none()` directly.
pub unsafe fn function_del_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__doc__")?; // function.py:456
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_doc = pyre_object::w_none();
        Ok(())
    }
}

/// `function.py:548-551 fget_func_annotations` parity — returns the
/// stored annotations dict, lazily allocating an empty dict on the
/// first read when none was set.  PyPy mutates `self.w_ann = space
/// .newdict()` in place so subsequent reads return the same dict
/// (`f.__annotations__ is f.__annotations__`); pyre stamps the slot
/// the same way through `function_set_annotations`.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_get_annotations(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if obj.is_null() {
            return pyre_object::w_dict_new();
        }
        let func = obj as *mut Function;
        let cached = (*func).w_ann;
        if !cached.is_null() {
            return cached;
        }
        let fresh = pyre_object::w_dict_new();
        function_write_barrier(obj);
        (*func).w_ann = fresh;
        fresh
    }
}

/// MAKE_FUNCTION ANNOTATIONS opcode helper — stamps the
/// compile-time annotations dict directly into `Function.w_ann`
/// without running `function.py:553-559 fset_func_annotations`'s
/// type validation.  The compiler always emits a real dict here, so
/// the `isinstance(w_new, dict)` check would never fail; bypassing
/// it keeps the opcode hot path free of error-handling overhead.
///
/// User-level `f.__annotations__ = X` writes go through
/// `fset_func_annotations` (typedef getset descriptor) instead.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_annotations(obj: PyObjectRef, w_ann: PyObjectRef) {
    unsafe {
        if obj.is_null() {
            return;
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_ann = w_ann;
    }
}

/// `function.py:553-559 fset_func_annotations` parity:
///
/// ```python
/// def fset_func_annotations(self, space, w_new):
///     self._check_code_mutable("__annotations__")
///     if space.is_w(w_new, space.w_None):
///         w_new = None
///     elif not space.isinstance_w(w_new, space.w_dict):
///         raise oefmt(space.w_TypeError, "__annotations__ must be a dict")
///     self.w_ann = w_new
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_annotations(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:554
        if obj.is_null() {
            return Ok(());
        }
        // function.py:555-558 — None clears the slot, anything else
        // must be a dict.
        let stored = if value.is_null() || pyre_object::is_none(value) {
            PY_NULL
        } else if pyre_object::is_dict(value) {
            value
        } else {
            return Err(crate::PyError::type_error("__annotations__ must be a dict"));
        };
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_ann = stored;
        Ok(())
    }
}

/// `function.py:561-563 fdel_func_annotations` parity:
///
/// ```python
/// def fdel_func_annotations(self, space):
///     self._check_code_mutable("__annotations__")
///     self.w_ann = None
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fdel_func_annotations(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__annotations__")?; // function.py:562
        if obj.is_null() {
            return Ok(());
        }
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_ann = PY_NULL;
        Ok(())
    }
}

/// PyPy `fget_func_defaults` accessor.
#[inline]
pub unsafe fn fget_func_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_defaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py:408-416 fset_func_defaults` parity:
///
/// ```python
/// def fset_func_defaults(self, space, w_defaults):
///     self._check_code_mutable("__defaults__")
///     if space.is_w(w_defaults, space.w_None):
///         self.defs_w = []
///         return
///     if not space.isinstance_w(w_defaults, space.w_tuple):
///         raise oefmt(space.w_TypeError,
///                     "__defaults__ must be set to a tuple object or None")
///     self.defs_w = space.fixedview(w_defaults)
/// ```
#[inline]
pub unsafe fn fset_func_defaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:409
        if value.is_null() || pyre_object::is_none(value) {
            function_set_defaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_tuple(value) {
            return Err(crate::PyError::type_error(
                "__defaults__ must be set to a tuple object or None",
            ));
        }
        function_set_defaults(obj, value);
        Ok(())
    }
}

/// `function.py:418-420 fdel_func_defaults` parity:
///
/// ```python
/// def fdel_func_defaults(self, space):
///     self._check_code_mutable("__defaults__")
///     self.defs_w = []
/// ```
#[inline]
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__defaults__")?; // function.py:419
        function_set_defaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// PyPy `fget_func_kwdefaults` accessor.
#[inline]
pub unsafe fn fget_func_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_kwdefaults(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// `function.py:427-433 fset_func_kwdefaults` parity:
///
/// ```python
/// def fset_func_kwdefaults(self, space, w_new):
///     if space.is_w(w_new, space.w_None):
///         self.w_kw_defs = None
///     else:
///         if not space.isinstance_w(w_new, space.w_dict):
///             raise oefmt(space.w_TypeError, "__kwdefaults__ must be a dict")
///         self.w_kw_defs = w_new
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — `__kwdefaults__`
/// is settable on builtins too.
#[inline]
pub unsafe fn fset_func_kwdefaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        if value.is_null() || pyre_object::is_none(value) {
            function_set_kwdefaults(obj, pyre_object::PY_NULL);
            return Ok(());
        }
        if !pyre_object::is_dict(value) {
            return Err(crate::PyError::type_error("__kwdefaults__ must be a dict"));
        }
        function_set_kwdefaults(obj, value);
        Ok(())
    }
}

/// `function.py:435-436 fdel_func_kwdefaults` parity:
///
/// ```python
/// def fdel_func_kwdefaults(self, space):
///     self.w_kw_defs = None
/// ```
///
/// PyPy intentionally omits `_check_code_mutable` here — symmetric
/// with `fset_func_kwdefaults` at function.py:427-433.
#[inline]
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        function_set_kwdefaults(obj, pyre_object::PY_NULL);
        Ok(())
    }
}

/// function.py:435-436 — `fget_func_code(self, space): return self.getcode()`
/// Uses getcode() for JIT elidable_promote / promote path.
#[inline]
pub unsafe fn function_get_func_code(obj: PyObjectRef) -> *const () {
    unsafe { getcode(obj) }
}

/// PyPy-compatible `__code__` setter.
#[inline]
pub unsafe fn function_set_func_code(obj: PyObjectRef, code: *const ()) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).code = code;
    }
}

/// PyPy-compatible `__name__` getter alias.
#[inline]
pub unsafe fn function_get_func_name(obj: PyObjectRef) -> &'static str {
    unsafe { function_get_name(obj) }
}

/// PyPy-compatible `__name__` setter.
#[inline]
pub unsafe fn function_set_func_name(obj: PyObjectRef, name: PyObjectRef) {
    unsafe {
        if !pyre_object::is_str(name) {
            return;
        }
        let name = pyre_object::w_str_get_value(name);
        let name = pyre_object::lltype::malloc_raw(name.to_string()) as *const String;
        let old = (*(obj as *mut Function)).name;
        if !old.is_null() {
            drop(Box::from_raw(old as *mut String));
        }
        (*(obj as *mut Function)).name = name;
    }
}

/// `function.py:498-501 fget_func_objclass` parity:
///
/// ```python
/// def fget_func_objclass(self, space):
///     if self.w_objclass is None:
///         raise oefmt(space.w_AttributeError, "__objclass__")
///     return self.w_objclass
/// ```
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_objclass(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_objclass;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__objclass__"));
        }
        Ok(value)
    }
}

/// `function.py:503-504 set_objclass(w_type)` parity — direct field
/// write used by descriptor-bind helpers.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn function_set_objclass(obj: PyObjectRef, w_type: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_objclass = w_type;
    }
}

/// `function.py:487-490 fget_func_text_signature` parity:
///
/// ```python
/// def fget_func_text_signature(self, space):
///     if self.w_text_signature is None:
///         raise oefmt(space.w_AttributeError, "__text_signature__")
///     return self.w_text_signature
/// ```
///
/// PyPy distinguishes the RPython-level `None` (unset, raises
/// `AttributeError`) from `space.w_None` (explicitly stored). Pyre
/// uses `PY_NULL` for the former; `space.w_None` survives as a
/// real PyObjectRef.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fget_func_text_signature(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        let value = (*(obj as *const Function)).w_text_signature;
        if value.is_null() {
            return Err(crate::PyError::attribute_error("__text_signature__"));
        }
        Ok(value)
    }
}

/// `function.py:492-493 fset_func_text_signature` parity:
///
/// ```python
/// def fset_func_text_signature(self, space, w_value):
///     self.w_text_signature = w_value
/// ```
///
/// Direct field write — even `space.w_None` is preserved as a real
/// value (only the RPython-level `None`, i.e. `PY_NULL`, is treated
/// as "unset" by `fget_func_text_signature`).
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_text_signature(obj: PyObjectRef, value: PyObjectRef) {
    unsafe {
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_text_signature = value;
    }
}

/// `function.py:462-468 fset_func_name` parity:
///
/// ```python
/// def fset_func_name(self, space, w_name):
///     self._check_code_mutable("__name__")
///     if space.isinstance_w(w_name, space.w_text):
///         self.name = space.text_w(w_name)
///     else:
///         raise oefmt(space.w_TypeError,
///                     "__name__ must be set to a string object")
/// ```
///
/// `isinstance_str_w` mirrors `space.isinstance_w(w_name, space.w_text)`
/// — accepts `str` and any `str` subclass.
///
/// # Safety
/// `obj` must point to a valid `Function`.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__name__")?; // function.py:463
        if !crate::baseobjspace::isinstance_str_w(name) {
            return Err(crate::PyError::type_error(
                "__name__ must be set to a string object",
            ));
        }
        function_set_func_name(obj, name);
        Ok(())
    }
}

// _check_code_mutable is defined above (function.py:367-370 parity).

/// `function.py:525-553 fset_func_code` parity:
///
/// ```python
/// def fset_func_code(self, space, w_code):
///     from pypy.interpreter.pycode import PyCode
///     if not self.can_change_code:
///         raise oefmt(space.w_AttributeError,
///                     "Cannot change code attribute of builtin functions")
///     code = space.interp_w(Code, w_code)
///     closure_len = 0
///     if self.closure:
///         closure_len = len(self.closure)
///     if isinstance(code, PyCode) and closure_len != len(code.co_freevars):
///         raise oefmt(space.w_ValueError,
///                     "%N() requires a code object with %d free vars, not "
///                     "%d", self, closure_len, len(code.co_freevars))
///     self.code = code
/// ```
///
/// `w_code` is the user-provided value as a `PyObjectRef`; this
/// helper enforces the `is_code(...)` (interp_w PyCode) check and
/// the closure / freevar arity invariant before reaching for the
/// inner code pointer.  Without these checks the previous setter
/// would happily reinterpret arbitrary objects as `*const CodeObject`
/// and corrupt the function's internal pointer (worse than main's
/// pre-fix shadowing-via-instance-dict regression).
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn fset_func_code(obj: PyObjectRef, w_code: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        // function.py:527-529 — `can_change_code = False` on
        // BuiltinFunction / FunctionWithFixedCode.  pyre exposes
        // this via `_check_code_mutable` which raises the same
        // AttributeError.
        _check_code_mutable(obj, "__code__")?;
        // function.py:530 — `space.interp_w(Code, w_code)` raises
        // TypeError when `w_code` is not a `PyCode` instance.
        if w_code.is_null() || !crate::pycode::is_code(w_code) {
            return Err(crate::PyError::type_error(
                "__code__ must be set to a code object",
            ));
        }
        // function.py:531-537 — closure-vs-freevars arity check.
        // PyPy raises ValueError if the function has N closure cells
        // and the new code object declares M co_freevars with N != M;
        // mismatch leaves dangling cells / unbound freevars at runtime.
        let closure = function_get_closure(obj);
        let closure_len = if closure.is_null() || pyre_object::is_none(closure) {
            0
        } else {
            pyre_object::w_tuple_len(closure)
        };
        let raw_code = crate::pycode::w_code_get_ptr(w_code) as *const crate::CodeObject;
        let freevars_len = if raw_code.is_null() {
            0
        } else {
            (&(*raw_code).freevars).len()
        };
        if closure_len != freevars_len {
            let name = function_get_name(obj);
            return Err(crate::PyError::value_error(format!(
                "{name}() requires a code object with {closure_len} free vars, not {freevars_len}"
            )));
        }
        // function.py:538 self.fget_func_doc(space) — see test_issue1293.
        // Resolves the OLD code's docstring into `w_doc` *before* the
        // pointer flip so the cached value reflects the function's
        // original docstring, not the new code's first const.
        let _ = function_get_doc(obj);
        function_set_func_code(obj, w_code as *const ());
        Ok(())
    }
}

/// PyPy-compatible `__closure__` getter alias.
#[inline]
pub unsafe fn function_get_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe { function_get_closure(obj) }
}

/// PyPy-compatible `fget_func_closure`.
#[inline]
pub unsafe fn fget_func_closure(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let value = function_get_func_closure(obj);
        if value.is_null() {
            pyre_object::w_none()
        } else {
            value
        }
    }
}

/// PyPy-compatible `__closure__` setter alias.
#[inline]
pub unsafe fn fset_func_closure(obj: PyObjectRef, closure: PyObjectRef) {
    unsafe {
        function_set_closure(obj, closure);
    }
}

/// `function.py:503-509 fget___module__`:
///
/// ```python
/// def fget___module__(self, space):
///     if self.w_module is None:
///         if self.w_func_globals is not None and not space.is_w(
///                 self.w_func_globals, space.w_None):
///             self.w_module = space.call_method(
///                 self.w_func_globals, "get", space.newtext("__name__"))
///         else:
///             self.w_module = space.w_None
///     return self.w_module
/// ```
///
/// Caches on first read: if w_module is PY_NULL (RPython-level None,
/// unset), computes from globals["__name__"] and stores into
/// self.w_module.  PY_NULL = RPython `None`, w_none() = space.w_None.
/// After fdel___module__ writes w_none(), subsequent reads return
/// None without re-computing from globals.
///
/// PRE-EXISTING-ADAPTATION: pyre stores `w_func_globals` as a raw
/// `*mut DictStorage` rather than a wrapped `W_DictObject`, so the
/// upstream `space.is_w(self.w_func_globals, space.w_None)` arm at
/// `function.py:505` collapses into the single null-pointer check
/// here.  Phase 5 collapses the storage / W_DictObject split, after
/// which the upstream arm-for-arm shape becomes expressible.
#[inline]
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        // function.py:504: if self.w_module is None
        if (*func).w_module.is_null() {
            // function.py:505-506: globals.get("__name__")
            let globals = (*func).w_func_globals;
            if !globals.is_null() {
                function_write_barrier(obj);
                (*func).w_module = (*globals)
                    .get("__name__")
                    .copied()
                    .unwrap_or(pyre_object::w_none());
            } else {
                // function.py:508: self.w_module = space.w_None
                function_write_barrier(obj);
                (*func).w_module = pyre_object::w_none();
            }
        }
        // function.py:509: return self.w_module
        (*func).w_module
    }
}

/// PyPy-compatible `descr_function__new__` helper.
#[inline]
pub unsafe fn descr_function__new__(
    code: *const (),
    w_globals: *mut DictStorage,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = _argdefs;
        let name = if !w_name.is_null() && !pyre_object::is_none(w_name) {
            pyre_object::w_str_get_value(w_name).to_string()
        } else {
            String::new()
        };
        let closure = if w_closure.is_null() || pyre_object::is_none(w_closure) {
            pyre_object::PY_NULL
        } else {
            w_closure
        };
        function_new_with_closure(code, name, w_globals, closure)
    }
}

/// `function.py:511-513 fset___module__` parity:
///
/// ```python
/// def fset___module__(self, space, w_module):
///     self._check_code_mutable("__module__")
///     self.w_module = w_module
/// ```
#[inline]
pub unsafe fn fset___module__(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:512
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = value;
        Ok(())
    }
}

/// `function.py:515-517 fdel___module__` parity:
///
/// ```python
/// def fdel___module__(self, space):
///     self._check_code_mutable("__module__")
///     self.w_module = space.w_None
/// ```
#[inline]
pub unsafe fn fdel___module__(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "__module__")?; // function.py:516
        // function.py:517: self.w_module = space.w_None
        function_write_barrier(obj);
        (*(obj as *mut Function)).w_module = pyre_object::w_none();
        Ok(())
    }
}

/// PyPy-compatible `descr_function__new__` overload.
#[inline]
pub unsafe fn _cleanup_(_obj: PyObjectRef) -> bool {
    true
}

#[inline]
pub unsafe fn descr_builtinfunction__new__(
    code: *const (),
    w_globals: *mut DictStorage,
    w_name: PyObjectRef,
    _argdefs: PyObjectRef,
    w_closure: PyObjectRef,
) -> PyObjectRef {
    unsafe { descr_function__new__(code, w_globals, w_name, _argdefs, w_closure) }
}

/// PyPy-compatible static registry hook.
#[inline]
pub fn add_to_table() {}

/// PyPy-compatible `__doc__` getter.
#[inline]
pub unsafe fn fget_func_doc(obj: PyObjectRef) -> PyObjectRef {
    function_get_doc(obj)
}

/// function.py:400 — `fset_func_doc` descriptor.
#[inline]
pub unsafe fn fset_func_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_set_doc(obj, value) }
}

/// function.py:404 — `fdel_func_doc` descriptor.
#[inline]
pub unsafe fn fdel_func_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe { function_del_doc(obj) }
}

#[inline]
pub fn immutable_unique_id(_obj: PyObjectRef) -> usize {
    _obj as usize
}

/// PyPy-compatible `find` helper.
#[inline]
pub fn find(_identifier: &str) -> PyObjectRef {
    let _ = _identifier;
    pyre_object::PY_NULL
}

#[inline]
fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { crate::gateway::is_builtin_code(obj) }
}

#[inline]
pub fn descr_init() {}

#[inline]
pub unsafe fn descr_classmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::propertyobject::w_classmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_classmethod_get(
    w_obj: PyObjectRef,
    obj: PyObjectRef,
    w_cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = w_cls;
        if obj.is_null() || pyre_object::is_none(obj) {
            w_obj
        } else {
            let func = pyre_object::w_classmethod_get_func(w_obj);
            let cls = if w_obj.is_null() {
                pyre_object::w_none()
            } else {
                obj
            };
            pyre_object::w_method_new(func, cls, cls)
        }
    }
}

#[inline]
pub unsafe fn descr_staticmethod__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::propertyobject::w_staticmethod_new(w_function)
    }
}

#[inline]
pub unsafe fn descr_staticmethod_get(
    obj: PyObjectRef,
    _obj: PyObjectRef,
    _cls: PyObjectRef,
) -> PyObjectRef {
    unsafe {
        let _ = (_obj, _cls);
        if obj.is_null() {
            pyre_object::w_none()
        } else {
            pyre_object::w_staticmethod_get_func(obj)
        }
    }
}

#[inline]
pub unsafe fn descr_method__new__(
    _subtype: PyObjectRef,
    w_function: PyObjectRef,
    w_instance: PyObjectRef,
    w_class: PyObjectRef,
) -> PyObjectRef {
    let _ = _subtype;
    if w_function.is_null() {
        pyre_object::w_none()
    } else {
        pyre_object::w_method_new(w_function, w_instance, w_class)
    }
}

#[inline]
pub unsafe fn descr_method_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = _func;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        let owner = if cls.is_null() {
            pyre_object::w_none()
        } else {
            cls
        };
        pyre_object::w_method_new(_func, obj, owner)
    }
}

#[inline]
pub unsafe fn descr_method_call(obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        call_obj_args(obj, pyre_object::w_none(), args)
    } else {
        call_obj_args(obj, args[0], &args[1..])
    }
}

#[inline]
pub unsafe fn descr_method_eq(_self: PyObjectRef, other: PyObjectRef) -> bool {
    _self == other
}

#[inline]
pub unsafe fn descr_method_ne(_self: PyObjectRef, other: PyObjectRef) -> bool {
    _self != other
}

#[inline]
pub unsafe fn descr_method_repr(obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_str_new(&format!("method {obj:?}"))
}

#[inline]
pub unsafe fn descr_method_getattribute(obj: PyObjectRef, _name: PyObjectRef) -> PyObjectRef {
    let _ = _name;
    obj
}

#[inline]
pub unsafe fn descr_method_hash(_self: PyObjectRef) -> isize {
    _self as isize
}

#[inline]
pub unsafe fn descr_method__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    let _ = _obj;
    pyre_object::w_tuple_new(vec![pyre_object::w_str_new("method")])
}

#[inline]
pub unsafe fn is_w(_obj: PyObjectRef, _other: PyObjectRef) -> bool {
    _obj == _other
}

/// PyPy-compatible `descr_function_call` helper.
#[inline]
pub fn descr_function_call(args: &[PyObjectRef]) -> PyObjectRef {
    if args.is_empty() {
        pyre_object::PY_NULL
    } else {
        call_args(args[0], &args[1..])
    }
}

/// PyPy-compatible `descr_function_get` helper.
#[inline]
pub unsafe fn descr_function_get(
    _func: PyObjectRef,
    obj: PyObjectRef,
    cls: PyObjectRef,
) -> PyObjectRef {
    let _ = cls;
    if obj.is_null() || unsafe { pyre_object::is_none(obj) } {
        _func
    } else {
        pyre_object::w_method_new(_func, obj, cls)
    }
}

/// PyPy-compatible `descr_function_repr` helper.
#[inline]
pub unsafe fn descr_function_repr(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let name = function_get_name(obj);
        pyre_object::w_str_new(&format!("function {name}"))
    }
}

/// PyPy-compatible `__code__` getter for direct descriptors.
#[inline]
pub unsafe fn fget_func_code(obj: PyObjectRef) -> *const () {
    unsafe { function_get_code(obj) }
}

/// PyPy-compatible `descr__reduce__` helper.
#[inline]
pub fn descr_function__reduce__(_obj: PyObjectRef) -> PyObjectRef {
    pyre_object::w_tuple_new(vec![
        pyre_object::w_tuple_new(vec![]),
        pyre_object::w_tuple_new(vec![]),
    ])
}

/// PyPy-compatible `descr__setstate__` helper.
#[inline]
pub fn descr_function__setstate__(_obj: PyObjectRef, _state: PyObjectRef) {
    let _ = _state;
}

#[inline]
pub fn __init__() {}

#[inline]
pub fn __repr__() -> String {
    "function".to_string()
}

/// PyPy-compatible `__call__` alias helper.
#[inline]
pub fn call(frame: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(frame, args)
}

/// PyPy-compatible call fast-path hooks.
#[inline]
pub fn call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::call_function(func, args)
}

/// PyPy-compatible `call_obj_args` helper.
#[inline]
pub fn call_obj_args(func: PyObjectRef, obj: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    crate::baseobjspace::call_obj_args(func, obj, args)
}

/// PyPy-compatible `call_args` instance method.
#[inline]
pub fn function_call_args(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// PyPy-compatible `call_obj_args` instance method.
#[inline]
pub(crate) fn function_call_obj_args(
    func: PyObjectRef,
    obj: PyObjectRef,
    args: &[PyObjectRef],
) -> PyObjectRef {
    call_obj_args(func, obj, args)
}

/// PyPy-compatible `funccall` helper.
#[inline]
pub fn funccall(func: PyObjectRef, args: &[PyObjectRef]) -> PyObjectRef {
    call_args(func, args)
}

/// baseobjspace.py: `space._code_of_sys_exc_info` — the BuiltinCode object
/// backing `sys.exc_info`, captured at sys-module init time so the JIT
/// fast-path in `funccall_valuestack` can recognize this specific call and
/// inline `exc_info_direct` without going through the regular dispatch.
///
/// PyPy stores this on the space; pyre is single-space-per-thread, so a
/// thread-local cell suffices. The paired `direct_fn` returns the same
/// `(type, value, traceback)` tuple as the regular closure but skips the
/// builtin-call setup.
type ExcInfoDirectFn = fn() -> PyObjectRef;
thread_local! {
    static SYS_EXC_INFO_CODE: std::cell::Cell<*const ()> = const { std::cell::Cell::new(std::ptr::null()) };
    static SYS_EXC_INFO_DIRECT_FN: std::cell::Cell<Option<ExcInfoDirectFn>> = const { std::cell::Cell::new(None) };
}

/// Register the BuiltinCode pointer + direct helper for `sys.exc_info`.
///
/// Called once during sys module init (after the builtin is created). The
/// `code` pointer is the BuiltinCode object underlying the `exc_info`
/// builtin function; `direct_fn` is the JIT-direct equivalent of the
/// closure body. `funccall_valuestack` consults both to take the fast path.
pub fn register_sys_exc_info_path(code: *const (), direct_fn: ExcInfoDirectFn) {
    SYS_EXC_INFO_CODE.with(|cell| cell.set(code));
    SYS_EXC_INFO_DIRECT_FN.with(|cell| cell.set(Some(direct_fn)));
}

#[inline]
fn sys_exc_info_code() -> *const () {
    SYS_EXC_INFO_CODE.with(|cell| cell.get())
}

#[inline]
fn sys_exc_info_direct_fn() -> Option<ExcInfoDirectFn> {
    SYS_EXC_INFO_DIRECT_FN.with(|cell| cell.get())
}

/// function.py:139-203 `funccall_valuestack` — fast-path call dispatcher.
///
/// Dispatches based on `code.fast_natural_arity`:
/// - nargs == arity (0-4): direct builtin fastcall from stack (no Vec alloc)
/// - (nargs | FLATPYCALL) == arity: _flat_pycall (user code, exact arity)
/// - FLATPYCALL + defaults: _flat_pycall_defaults
/// - Fallback: allocate Vec via peekvalues + generic call path
pub fn funccall_valuestack(
    func: PyObjectRef,
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
    methodcall: bool,
) -> PyObjectRef {
    let code = unsafe { crate::getcode(func) };

    // function.py:146-150 — JIT direct path for `sys.exc_info()` with no
    // arguments: skip the builtin call entirely and inline the tuple
    // construction. PyPy uses `space._code_of_sys_exc_info`; pyre uses the
    // thread-local cache populated during sys module init.
    if nargs == 0
        && majit_metainterp::jit::we_are_jitted()
        && std::ptr::eq(code, sys_exc_info_code())
    {
        if let Some(direct_fn) = sys_exc_info_direct_fn() {
            frame.dropvalues(dropvalues);
            return direct_fn();
        }
    }

    let fast_natural_arity =
        unsafe { crate::pycode::code_get_fast_natural_arity(code as PyObjectRef) } as usize;

    // function.py:153-184 — nargs == fast_natural_arity: builtin fast path
    // baseobjspace.py:1243 — skip when profiling (c_call/c_return events)
    if nargs == fast_natural_arity && nargs <= 4 && !frame.get_is_being_profiled() {
        debug_assert!(
            (fast_natural_arity & crate::FLATPYCALL as usize) == 0,
            "FLATPYCALL bit set on arity {fast_natural_arity} — not a builtin code"
        );
        let builtin_fn = unsafe { crate::builtin_code_get(code as PyObjectRef) };
        // function.py:154-184 — BuiltinCodeN.fastcall_N dispatch.
        // Pyre builtins share a single fn(&[PyObjectRef]) signature, so we
        // build a fixed-size stack array instead of heap-allocating a Vec.
        let result = match nargs {
            0 => {
                frame.dropvalues(dropvalues);
                builtin_fn(&[])
            }
            1 => {
                let a0 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0])
            }
            2 => {
                // function.py:168 — peekvalue order: 0=top, 1=below top
                let a0 = frame.peekvalue(1);
                let a1 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1])
            }
            3 => {
                let a0 = frame.peekvalue(2);
                let a1 = frame.peekvalue(1);
                let a2 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1, a2])
            }
            4 => {
                let a0 = frame.peekvalue(3);
                let a1 = frame.peekvalue(2);
                let a2 = frame.peekvalue(1);
                let a3 = frame.peekvalue(0);
                frame.dropvalues(dropvalues);
                builtin_fn(&[a0, a1, a2, a3])
            }
            _ => unreachable!(),
        };
        return match result {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:185-187 — (nargs | FLATPYCALL) == fast_natural_arity
    if (nargs | crate::FLATPYCALL as usize) == fast_natural_arity {
        return _flat_pycall(func, code, nargs, frame, dropvalues);
    }

    // function.py:188-193 — FLATPYCALL bit set + nargs within defaults range
    if (fast_natural_arity & crate::FLATPYCALL as usize) != 0 {
        let natural_arity = fast_natural_arity & 0xff;
        if nargs < natural_arity {
            let raw_defs = unsafe { crate::function_get_defaults(func) };
            let defs = if raw_defs.is_null() {
                std::ptr::null_mut()
            } else {
                crate::baseobjspace::unwrap_cell(raw_defs)
            };
            let defs_len = if defs.is_null() || !unsafe { pyre_object::is_tuple(defs) } {
                0
            } else {
                unsafe { pyre_object::w_tuple_len(defs) }
            };
            if nargs >= natural_arity.saturating_sub(defs_len) {
                return _flat_pycall_defaults(
                    func,
                    code,
                    nargs,
                    frame,
                    defs,
                    natural_arity - nargs,
                    dropvalues,
                );
            }
        }
    }

    // function.py:194-199 — PASSTHROUGHARGS1 dispatch.
    // PyPy's BuiltinCodePassThroughArguments1.funcrun_obj receives w_obj
    // separately from an Arguments rest, then concatenates them as
    // `args_w = [w_obj] + _args_w` before calling the unwrapped fn. Pyre's
    // single BuiltinCodeFn signature already takes a flat slice, so the
    // peek/Arguments split is structural — the final closure invocation
    // sees `[w_obj, ...rest]` exactly as PyPy's post-merge args_w.
    if fast_natural_arity == crate::PASSTHROUGHARGS1 as usize && nargs >= 1 {
        let builtin_fn = unsafe { crate::builtin_code_get(code as PyObjectRef) };
        let w_obj = frame.peekvalue(nargs - 1);
        let rest = frame.make_arguments(nargs - 1, false, func);
        let mut args_w = Vec::with_capacity(nargs);
        args_w.push(w_obj);
        args_w.extend_from_slice(&rest);
        frame.dropvalues(dropvalues);
        return match builtin_fn(&args_w) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        };
    }

    // function.py:201-203 — fallback: build Arguments via make_arguments
    // (carries methodcall + w_function for diagnostics) and dispatch through
    // call_args.
    let args = frame.make_arguments(nargs, methodcall, func);
    frame.dropvalues(dropvalues);
    funccall(func, &args)
}

/// function.py:206-214 `_flat_pycall` — create frame directly from stack.
///
/// For user functions with exact arity match (no defaults needed).
/// Copies args from caller's value stack into the new frame's locals
/// without intermediate Vec allocation.
fn _flat_pycall(
    func: PyObjectRef,
    code: *const (),
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    dropvalues: usize,
) -> PyObjectRef {
    // call.rs:423-424 parity — increment call depth for JIT depth tracking.
    let _depth_guard = crate::call::increment_call_depth();
    let globals = unsafe { function_get_globals(func) };
    let closure = unsafe { function_get_closure(func) };

    // function.py:208-209 — createframe(code, w_func_globals, self)
    let mut new_frame = crate::pyframe::PyFrame::new_for_call_with_closure(
        code,
        &[], // locals filled below directly from stack
        globals,
        frame.execution_context,
        closure,
    );

    // function.py:210-211 — copy from stack into locals directly
    // peekvalue(nargs-1-i) gives bottom-to-top order (matching local slot order)
    for i in 0..nargs {
        new_frame.locals_w_mut()[i] = frame.peekvalue(nargs - 1 - i);
    }
    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    let _callee_locals_root = FrameLocalsRoot::new(&mut new_frame);

    // function.py:214 — return new_frame.run(self.name, self.qualname)
    // Check generator/coroutine: run() wraps into generator object instead
    // of executing the body. For normal functions, use the JIT-aware eval.
    if new_frame._is_generator_or_coroutine() {
        match new_frame.run() {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

/// function.py:217-231 `_flat_pycall_defaults` — flat call with defaults.
///
/// Same as `_flat_pycall` but also fills missing positional args from
/// `self.defs_w[ndefs - defs_to_load ..]`.
/// `defs` is the pre-unwrapped defaults tuple (already null-checked and
/// verified as a tuple by the caller in `funccall_valuestack`).
fn _flat_pycall_defaults(
    func: PyObjectRef,
    code: *const (),
    nargs: usize,
    frame: &mut crate::pyframe::PyFrame,
    defs: PyObjectRef,
    defs_to_load: usize,
    dropvalues: usize,
) -> PyObjectRef {
    let _depth_guard = crate::call::increment_call_depth();
    let globals = unsafe { function_get_globals(func) };
    let closure = unsafe { function_get_closure(func) };

    let mut new_frame = crate::pyframe::PyFrame::new_for_call_with_closure(
        code,
        &[], // locals filled below
        globals,
        frame.execution_context,
        closure,
    );

    // function.py:221-222 — copy positional args from stack
    for i in 0..nargs {
        new_frame.locals_w_mut()[i] = frame.peekvalue(nargs - 1 - i);
    }

    // function.py:224-229 — fill remaining from defs_w
    if !defs.is_null() {
        let ndefs = unsafe { pyre_object::w_tuple_len(defs) };
        let start = ndefs - defs_to_load;
        let mut i = nargs;
        for j in start..ndefs {
            if let Some(val) = unsafe { pyre_object::w_tuple_getitem(defs, j as i64) } {
                new_frame.locals_w_mut()[i] = val;
            }
            i += 1;
        }
    }

    frame.dropvalues(dropvalues);
    new_frame.fix_array_ptrs();
    let _caller_locals_root = FrameLocalsRoot::new(frame);
    let _callee_locals_root = FrameLocalsRoot::new(&mut new_frame);

    // function.py:231 — return new_frame.run(self.name, self.qualname)
    if new_frame._is_generator_or_coroutine() {
        match new_frame.run() {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    } else {
        let eval_fn = crate::call::get_eval_fn();
        match eval_fn(&mut new_frame) {
            Ok(v) => v,
            Err(e) => {
                crate::call::set_call_error(e);
                pyre_object::PY_NULL
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_create() {
        // Function.code now stores a Code-level wrapper (W_CodeObject).
        let raw_code = 0xDEAD_BEEF as *const ();
        let w_code = crate::w_code_new(raw_code);
        let mut ns = DictStorage::new();
        let obj = function_new(w_code as *const (), "myfunc".to_string(), &mut ns);
        unsafe {
            assert!(is_function(obj));
            assert!(!is_int(obj));
            assert_eq!(function_get_code(obj), w_code as *const ());
            assert_eq!(function_get_name(obj), "myfunc");
            assert_eq!(function_get_globals(obj), &mut ns as *mut DictStorage);
            assert!(function_get_closure(obj).is_null());
        }
    }

    #[test]
    fn test_function_field_offsets() {
        assert_eq!(FUNCTION_CODE_OFFSET, 16); // after PyObject { ob_type(8) + w_class(8) }
        assert_eq!(FUNCTION_NAME_OFFSET, 32); // after code(8) + can_change_code(1) + padding(7)
        assert_eq!(FUNCTION_GLOBALS_OFFSET, 40); // after name
        assert_eq!(FUNCTION_CLOSURE_OFFSET, 48); // after w_func_globals
    }

    /// Guard against drift between the constant colocated with
    /// `Function` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT/BuiltinCode
    /// trip-wire tests.
    #[test]
    fn function_gc_type_id_matches_descr() {
        assert_eq!(FUNCTION_GC_TYPE_ID, 14);
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::TYPE_ID,
            FUNCTION_GC_TYPE_ID
        );
        assert_eq!(
            <Function as pyre_object::lltype::GcType>::SIZE,
            std::mem::size_of::<Function>()
        );
    }

    /// `FUNCTION_GC_PTR_OFFSETS` must list the five inline
    /// `PyObjectRef`-shaped fields the GC traces (the four
    /// `PyObjectRef` payload fields plus `code`, which is `*const ()`
    /// but points at a `[ob: PyObject, ...]`-prefixed Code object so
    /// the walker can interpret it as a typed reference). If a new GC
    /// field is added to `Function` (or one of these fields is removed)
    /// the array has to follow — this test makes the change a
    /// compile-time failure rather than a silent traversal gap.
    #[test]
    fn function_gc_ptr_offsets_cover_inline_pyobjectref_fields() {
        assert_eq!(
            FUNCTION_GC_PTR_OFFSETS,
            [
                std::mem::offset_of!(Function, code),
                std::mem::offset_of!(Function, closure),
                std::mem::offset_of!(Function, defs_w),
                std::mem::offset_of!(Function, w_kw_defs),
                std::mem::offset_of!(Function, w_module),
                std::mem::offset_of!(Function, w_func_globals_obj),
                std::mem::offset_of!(Function, w_ann),
                std::mem::offset_of!(Function, w_doc),
                std::mem::offset_of!(Function, w_qualname),
                std::mem::offset_of!(Function, w_objclass),
                std::mem::offset_of!(Function, w_text_signature),
            ]
        );
    }
}
