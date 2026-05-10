//! Function object.
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
}

/// function.py:706 — `class BuiltinFunction(Function): can_change_code = False`
pub type BuiltinFunction = Function;
/// function.py:703 — `class FunctionWithFixedCode(Function): can_change_code = False`
pub type FunctionWithFixedCode = Function;
pub type Method = pyre_object::methodobject::W_MethodObject;
pub type StaticMethod = pyre_object::propertyobject::W_StaticMethodObject;
pub type ClassMethod = pyre_object::propertyobject::W_ClassMethodObject;

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
pub const FUNCTION_GC_PTR_OFFSETS: [usize; 6] = [
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

    let name_ptr = pyre_object::lltype::malloc_raw(name) as *const String;
    pyre_object::lltype::malloc_typed(Function {
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
        w_func_globals_obj: PY_NULL,
    }) as PyObjectRef
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

/// function.py:367-370 — `_check_code_mutable(attr)`:
/// Raises TypeError if function code is not mutable.
pub unsafe fn _check_code_mutable(func: PyObjectRef, attr: &str) -> Result<(), crate::PyError> {
    unsafe {
        if (*(func as *const Function)).can_change_code {
            Ok(())
        } else {
            Err(crate::PyError::type_error(format!(
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

/// pypy/interpreter/function.py:53 `self.qualname = qualname or self.name`.
///
/// PyPy's `Function` stores `qualname` as a separate field initialised
/// from the code object's `co_qualname` (see
/// `function.py:50-54 __init__`).  Pyre's `Function` does not yet
/// carry that field, so the lookup mirrors the same precedence that
/// `getattr(func, '__qualname__')` already implements (see
/// `baseobjspace.rs:3635-3653`):
///
/// 1. ATTR_TABLE override (corresponds to PyPy's `set_qualname`
///    `function.py:473-485`).
/// 2. The bytecode `code.qualname` (CPython's `co_qualname`).
/// 3. Fall back to the bare `function.name` — matches the
///    `qualname or self.name` short-circuit.
///
/// PRE-EXISTING-ADAPTATION: storing `qualname` directly on `Function`
/// requires growing the struct + GC offset table + JIT field offsets.
/// Until that lands, this accessor is the line-by-line stand-in for
/// `w_function.qualname`.
///
/// # Safety
/// `obj` must point to a valid `Function`.
pub unsafe fn function_get_qualname(obj: PyObjectRef) -> String {
    unsafe {
        // function.py:473-485 — explicit `set_qualname(qualname)`
        // mirrors `obj.__qualname__ = ...` which lands in ATTR_TABLE.
        if let Some(w_qn) = crate::baseobjspace::ATTR_TABLE.with(|table| {
            table
                .borrow()
                .get(&(obj as usize))
                .and_then(|d| d.get("__qualname__").copied())
        }) && pyre_object::is_str(w_qn)
        {
            return pyre_object::w_str_get_value(w_qn).to_string();
        }
        // function.py:50-54 — `qualname=None` defaults to `co_qualname`.
        let code = function_get_code(obj) as PyObjectRef;
        if !code.is_null() && crate::pycode::is_code(code) {
            let raw_code_ptr = crate::pycode::w_code_get_ptr(code) as *const crate::CodeObject;
            if !raw_code_ptr.is_null() {
                return (*raw_code_ptr).qualname.to_string();
            }
        }
        // `qualname or self.name`.
        function_get_name(obj).to_string()
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

/// Resolve the canonical W_DictObject paired with `function.w_func_globals`,
/// caching the result in the adjacent `w_func_globals_obj` slot.
///
/// `function.py:57 self.w_func_globals = w_globals` stores the dict
/// object directly in PyPy.  Pyre's split storage / W_DictObject
/// model means callers that want object identity
/// (`function.__globals__ is module.__dict__`,
/// `f.__globals__ is g.__globals__` for sibling closures) need to
/// reach the canonical sibling.  `dict_storage_to_dict` returns the
/// same instance per storage (`mirror_target` invariant), and the
/// cache field skips the lookup on subsequent calls.
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
    unsafe { (*(obj as *mut Function)).closure = closure }
}

/// Get defaults tuple.
#[inline]
pub unsafe fn function_get_defaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).defs_w }
}

/// Set defaults tuple.
#[inline]
pub unsafe fn function_set_defaults(obj: PyObjectRef, defaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).defs_w = defaults }
}

/// Get kwdefaults dict.
#[inline]
pub unsafe fn function_get_kwdefaults(obj: PyObjectRef) -> PyObjectRef {
    unsafe { (*(obj as *const Function)).w_kw_defs }
}

/// Set kwdefaults dict.
#[inline]
pub unsafe fn function_set_kwdefaults(obj: PyObjectRef, kwdefaults: PyObjectRef) {
    unsafe { (*(obj as *mut Function)).w_kw_defs = kwdefaults }
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

/// PyPy `W_Function.fget_func_doc` accessor (function.py:395-398).
///
/// ```text
/// def fget_func_doc(self, space):
///     if self.w_doc is None:
///         self.w_doc = self.code.getdocstring(space)
///     return self.w_doc
/// ```
///
/// pyre has no `w_doc` struct field, so the explicit override (set via
/// `function_set_doc`) lives in the instance dict. Both lookups bypass
/// `crate::getattr` because `getattr(func, "__doc__")` resolves to the
/// typedef-installed `getset_func_doc` descriptor, which would call back
/// into here and recurse forever.
pub fn function_get_doc(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::w_none();
    }
    // First-level: explicit override stored on the function instance
    // (function.py:400 fset_func_doc writes here).
    let w_dict = crate::objspace::std::mapdict::INSTANCE_DICT
        .with(|table| table.borrow().get(&(obj as usize)).copied())
        .unwrap_or(pyre_object::PY_NULL);
    if !w_dict.is_null() {
        if let Some(v) = unsafe { pyre_object::w_dict_getitem_str(w_dict, "__doc__") } {
            return v;
        }
    }
    // Second-level: lazy fallback to `code.getdocstring(space)`
    // (function.py:397). pyre's BuiltinCode stores `docstring` directly;
    // user-level CodeObjects do not yet, so they default to None.
    let code = unsafe { function_get_code(obj) } as PyObjectRef;
    if !code.is_null() && unsafe { crate::gateway::is_builtin_code(code) } {
        return unsafe { crate::gateway::builtin_code_get_docstring(code) };
    }
    pyre_object::w_none()
}

/// function.py:400 — `fset_func_doc` mutator.
///
/// Writes the doc directly into the instance dict; the typedef-installed
/// `getset_func_doc` descriptor would otherwise recurse if we routed
/// through `setattr`.
pub unsafe fn function_set_doc(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_doc")?; // function.py:401
        if obj.is_null() {
            return Ok(());
        }
        let w_dict = crate::baseobjspace::getdict(obj);
        if !w_dict.is_null() {
            pyre_object::w_dict_setitem_str(w_dict, "__doc__", value);
        }
        Ok(())
    }
}

/// function.py:404 — `fdel_func_doc` deleter.
///
/// Removes the doc directly from the instance dict (see `function_set_doc`
/// for the recursion rationale).
pub unsafe fn function_del_doc(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_doc")?; // function.py:405
        if obj.is_null() {
            return Ok(());
        }
        let w_dict = crate::baseobjspace::getdict(obj);
        if !w_dict.is_null() {
            pyre_object::w_dict_delitem_str(w_dict, "__doc__");
        }
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

/// function.py:381 — `fset_func_defaults` mutator.
#[inline]
pub unsafe fn fset_func_defaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_defaults")?; // function.py:382
        if value.is_null() || pyre_object::is_none(value) {
            function_set_defaults(obj, pyre_object::PY_NULL);
        } else {
            function_set_defaults(obj, value);
        }
        Ok(())
    }
}

/// function.py:391 — `fdel_func_defaults` deleter.
#[inline]
pub unsafe fn fdel_func_defaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_defaults")?; // function.py:392
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

/// function.py — `fset_func_kwdefaults` mutator.
#[inline]
pub unsafe fn fset_func_kwdefaults(
    obj: PyObjectRef,
    value: PyObjectRef,
) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_kwdefaults")?;
        if value.is_null() || pyre_object::is_none(value) {
            function_set_kwdefaults(obj, pyre_object::PY_NULL);
        } else {
            function_set_kwdefaults(obj, value);
        }
        Ok(())
    }
}

/// function.py — `fdel_func_kwdefaults` deleter.
#[inline]
pub unsafe fn fdel_func_kwdefaults(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_kwdefaults")?;
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

/// function.py:411 — `fset_func_name` setter.
#[inline]
pub unsafe fn fset_func_name(obj: PyObjectRef, name: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_name")?; // function.py:412
        function_set_func_name(obj, name);
        Ok(())
    }
}

// _check_code_mutable is defined above (function.py:367-370 parity).

/// PyPy-compatible `__globals__` getter alias.
#[inline]
pub unsafe fn function_get_w_globals(obj: PyObjectRef) -> *mut DictStorage {
    unsafe { function_get_globals(obj) }
}

/// PyPy-compatible `__globals__` setter alias.
#[inline]
pub unsafe fn function_set_w_globals(obj: PyObjectRef, globals: *mut DictStorage) {
    unsafe {
        (*(obj as *mut Function)).w_func_globals = globals;
    }
}

/// function.py:367 — fset_func_code checks mutability before setting.
#[inline]
pub unsafe fn fset_func_code(obj: PyObjectRef, code: *const ()) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_code")?;
        function_set_func_code(obj, code);
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

/// function.py:419-425 — `fget___module__`.
/// Caches on first read: if w_module is PY_NULL (unset), computes from
/// globals["__name__"] and stores into self.w_module. Always returns
/// self.w_module afterwards.
///
/// PY_NULL = RPython None (unset sentinel), w_none() = space.w_None.
/// After fdel___module__, w_module is w_none() (not PY_NULL), so
/// subsequent gets return None without re-computing from globals.
#[inline]
pub unsafe fn fget___module__(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        let func = obj as *mut Function;
        // function.py:420: if self.w_module is None
        if (*func).w_module.is_null() {
            // function.py:421-422: compute and cache
            let globals = (*func).w_func_globals;
            if !globals.is_null() {
                (*func).w_module = (*globals)
                    .get("__name__")
                    .copied()
                    .unwrap_or(pyre_object::w_none());
            } else {
                // function.py:424: self.w_module = space.w_None
                (*func).w_module = pyre_object::w_none();
            }
        }
        // function.py:425: return self.w_module
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

/// function.py:427 — `fset___module__` setter.
#[inline]
pub unsafe fn fset___module__(obj: PyObjectRef, value: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_name")?; // function.py:428
        // function.py:429: self.w_module = w_module
        (*(obj as *mut Function)).w_module = value;
        Ok(())
    }
}

/// function.py:431 — `fdel___module__` deleter.
#[inline]
pub unsafe fn fdel___module__(obj: PyObjectRef) -> Result<(), crate::PyError> {
    unsafe {
        _check_code_mutable(obj, "func_name")?; // function.py:432
        // function.py:433: self.w_module = space.w_None
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
    let _ = methodcall; // function.py:140 — only for better error messages
    let code = unsafe { crate::getcode(func) };
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

    // Fallback: Vec allocation path (function.py:201-203)
    if nargs == 0 {
        frame.dropvalues(dropvalues);
        funccall(func, &[])
    } else {
        let args = frame.peekvalues(nargs);
        frame.dropvalues(dropvalues);
        funccall(func, &args)
    }
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
            ]
        );
    }
}
