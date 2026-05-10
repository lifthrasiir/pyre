//! Built-in function objects.
//!
//! A `BuiltinCode` wraps a Rust function pointer that implements
//! a Python builtin like `print`, `len`, etc.

use pyre_object::pyobject::*;

/// pypy/interpreter/gateway.py:36-73 `class SignatureBuilder`.
///
/// ```python
/// class SignatureBuilder(object):
///     def __init__(self, ...):
///         ...
///         self.posonlyargcount = 0
///         self.kwonlystartindex = -1
///
///     def append(self, argname):
///         self.argnames.append(argname)
///
///     def marker_posonly(self):
///         assert self.posonlyargcount == 0
///         assert self.kwonlystartindex == -1
///         self.posonlyargcount = len(self.argnames)
///
///     def marker_kwonly(self):
///         assert self.kwonlystartindex == -1
///         self.kwonlystartindex = len(self.argnames)
///
///     def signature(self):
///         if self.kwonlystartindex == -1:
///             kwonlyargcount = 0
///         else:
///             kwonlyargcount = len(self.argnames) - self.kwonlystartindex
///         return Signature(self.argnames,
///                          self.varargname, self.kwargname,
///                          kwonlyargcount, self.posonlyargcount)
/// ```
///
/// Pyre carries `kwonlystartindex` rather than `kwonlyargcount`
/// directly so the marker-driven build (`marker_kwonly()` records
/// where the kw-only tail starts; `signature()` derives the count at
/// build time) matches PyPy 1:1.  The `-1` sentinel encodes "no
/// `marker_kwonly` call seen yet".
#[derive(Debug, Clone)]
pub struct SignatureBuilder {
    pub name: &'static str,
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
    pub posonlyargcount: usize,
    pub kwonlystartindex: isize,
}

impl Default for SignatureBuilder {
    fn default() -> Self {
        Self {
            name: "",
            argnames: Vec::new(),
            varargname: None,
            kwargname: None,
            posonlyargcount: 0,
            kwonlystartindex: -1,
        }
    }
}

impl SignatureBuilder {
    /// gateway.py:54-55 `append`.
    pub fn append(&mut self, argname: &'static str) {
        self.argnames.push(argname);
    }

    /// gateway.py:57-60 `marker_posonly`.  PyPy asserts the marker is
    /// emitted at most once and before `marker_kwonly`.
    pub fn marker_posonly(&mut self) {
        assert!(self.posonlyargcount == 0);
        assert!(self.kwonlystartindex == -1);
        self.posonlyargcount = self.argnames.len();
    }

    /// gateway.py:62-64 `marker_kwonly`.  PyPy asserts the marker is
    /// emitted at most once.
    pub fn marker_kwonly(&mut self) {
        assert!(self.kwonlystartindex == -1);
        self.kwonlystartindex = self.argnames.len() as isize;
    }

    /// gateway.py:66-73 `signature`.  Derives `kwonlyargcount` from
    /// the argname list length minus the `marker_kwonly` index, or 0
    /// if the marker never fired.
    pub fn signature(&self) -> Signature {
        let kwonlyargcount = if self.kwonlystartindex == -1 {
            0
        } else {
            self.argnames.len() - self.kwonlystartindex as usize
        };
        Signature {
            argnames: self.argnames.clone(),
            varargname: self.varargname,
            kwargname: self.kwargname,
            kwonlyargcount,
            posonlyargcount: self.posonlyargcount,
        }
    }
}

/// pypy/interpreter/signature.py:3-78 `class Signature`.
///
/// ```python
/// class Signature(object):
///     _immutable_ = True
///     _immutable_fields_ = ["argnames[*]"]
///     __slots__ = ("argnames", "posonlyargcount", "kwonlyargcount",
///                  "varargname", "kwargname")
/// ```
///
/// `argnames` contains both the positional-only and the positional
/// arguments; the count of positional-only arguments is
/// `posonlyargcount`.  Keyword-only argument names live at the tail
/// of `argnames` and are counted by `kwonlyargcount`.
#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub argnames: Vec<&'static str>,
    pub varargname: Option<&'static str>,
    pub kwargname: Option<&'static str>,
    pub posonlyargcount: usize,
    pub kwonlyargcount: usize,
}

impl Signature {
    /// pypy/interpreter/signature.py:8-16 `Signature.__init__`.
    pub fn new(
        argnames: Vec<&'static str>,
        varargname: Option<&'static str>,
        kwargname: Option<&'static str>,
        kwonlyargcount: usize,
        posonlyargcount: usize,
    ) -> Self {
        Self {
            argnames,
            varargname,
            kwargname,
            posonlyargcount,
            kwonlyargcount,
        }
    }

    /// pypy/interpreter/signature.py:18-24 `find_argname`:
    /// ```python
    /// @jit.elidable
    /// def find_argname(self, name):
    ///     try:
    ///         return self.argnames.index(name)
    ///     except ValueError:
    ///         pass
    ///     return -1
    /// ```
    pub fn find_argname(&self, name: &str) -> isize {
        for (i, arg) in self.argnames.iter().enumerate() {
            if *arg == name {
                return i as isize;
            }
        }
        -1
    }

    /// pypy/interpreter/signature.py:26-31 `find_w_argname`:
    /// ```python
    /// @jit.elidable
    /// def find_w_argname(self, w_name):
    ///     for i, name in enumerate(self.argnames):
    ///         if w_name.eq_unwrapped(name):
    ///             return i
    ///     return -1
    /// ```
    ///
    /// `w_name.eq_unwrapped(name)` compares the wrapped string with a
    /// raw `&str`; pyre delegates to `find_argname` after unwrapping the
    /// PyObject via `w_str_get_value`.  Non-string `w_name` returns `-1`
    /// (matches PyPy's RPython unwrap-or-fail semantics for strings).
    pub fn find_w_argname(&self, w_name: PyObjectRef) -> isize {
        if w_name.is_null() {
            return -1;
        }
        unsafe {
            if !pyre_object::is_str(w_name) {
                return -1;
            }
            let name = pyre_object::w_str_get_value(w_name);
            self.find_argname(name)
        }
    }

    /// pypy/interpreter/signature.py:33-34 `num_argnames`:
    /// ```python
    /// def num_argnames(self):
    ///     return len(self.argnames) - self.kwonlyargcount
    /// ```
    pub fn num_argnames(&self) -> usize {
        self.argnames.len() - self.kwonlyargcount
    }

    /// pypy/interpreter/signature.py:36-37 `num_posonlyargnames`:
    /// ```python
    /// def num_posonlyargnames(self):
    ///     return self.posonlyargcount
    /// ```
    pub fn num_posonlyargnames(&self) -> usize {
        self.posonlyargcount
    }

    /// pypy/interpreter/signature.py:39-40 `num_kwonlyargnames`:
    /// ```python
    /// def num_kwonlyargnames(self):
    ///     return self.kwonlyargcount
    /// ```
    pub fn num_kwonlyargnames(&self) -> usize {
        self.kwonlyargcount
    }

    /// pypy/interpreter/signature.py:42-43 `has_vararg`:
    /// ```python
    /// def has_vararg(self):
    ///     return self.varargname is not None
    /// ```
    pub fn has_vararg(&self) -> bool {
        self.varargname.is_some()
    }

    /// pypy/interpreter/signature.py:45-46 `has_kwarg`:
    /// ```python
    /// def has_kwarg(self):
    ///     return self.kwargname is not None
    /// ```
    pub fn has_kwarg(&self) -> bool {
        self.kwargname.is_some()
    }

    /// pypy/interpreter/signature.py:48-52 `scope_length`:
    /// ```python
    /// def scope_length(self):
    ///     scopelen = len(self.argnames)
    ///     scopelen += self.has_vararg()
    ///     scopelen += self.has_kwarg()
    ///     return scopelen
    /// ```
    pub fn scope_length(&self) -> usize {
        let mut scopelen = self.argnames.len();
        if self.has_vararg() {
            scopelen += 1;
        }
        if self.has_kwarg() {
            scopelen += 1;
        }
        scopelen
    }

    /// pypy/interpreter/signature.py:54-60 `getallvarnames`:
    /// ```python
    /// def getallvarnames(self):
    ///     argnames = self.argnames
    ///     if self.varargname is not None:
    ///         argnames = argnames + [self.varargname]
    ///     if self.kwargname is not None:
    ///         argnames = argnames + [self.kwargname]
    ///     return argnames
    /// ```
    pub fn getallvarnames(&self) -> Vec<&'static str> {
        let mut argnames = self.argnames.clone();
        if let Some(name) = self.varargname {
            argnames.push(name);
        }
        if let Some(name) = self.kwargname {
            argnames.push(name);
        }
        argnames
    }
}

#[derive(Debug, Clone)]
pub struct Unwrapper;

impl Unwrapper {
    pub fn unwrap(&self, _space: PyObjectRef, _value: PyObjectRef) -> PyObjectRef {
        let _ = (_space, _value);
        std::ptr::null_mut()
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpecRecipe {
    pub miniglobals: Vec<PyObjectRef>,
}

impl UnwrapSpecRecipe {
    pub fn dispatch<T>(&self, _el: T, _args: &mut Vec<PyObjectRef>) {
        let _ = (&self.miniglobals, _el, _args);
    }

    pub fn apply_over(&self, _unwrap_spec: &[&str], _space: PyObjectRef, _name: &str) {
        let _ = (_unwrap_spec, _space, _name);
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpecEmit;

impl UnwrapSpecEmit {
    pub fn new() -> Self {
        Self
    }

    pub fn succ(&mut self) -> usize {
        0
    }

    pub fn use_name(&mut self, obj: &'static str) -> &'static str {
        obj
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpec_Check {
    pub func: PyObjectRef,
}

impl UnwrapSpec_Check {
    pub fn new(func: PyObjectRef, _argnames: &[&'static str]) -> Self {
        Self { func }
    }
}

#[derive(Debug, Clone)]
pub struct UnwrapSpec_EmitRun;

#[derive(Debug, Clone)]
pub struct UnwrapSpec_EmitShortcut;

#[derive(Debug, Clone)]
pub struct UnwrapSpec_FastFunc_Unwrap;

#[derive(Debug, Clone)]
pub struct FastFuncNotSupported;

#[derive(Debug, Clone)]
pub struct BuiltinActivation;

#[derive(Debug, Clone)]
pub struct GatewayCache;

#[derive(Debug, Clone)]
pub struct BuiltinCodePassThroughArguments0 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCodePassThroughArguments1 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode0 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode1 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode2 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode3 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct BuiltinCode4 {
    pub code: PyObjectRef,
}

#[derive(Debug, Clone)]
pub struct WrappedDefault;

#[derive(Debug, Clone)]
pub struct ApplevelClass {
    pub source: Option<PyObjectRef>,
}

#[derive(Debug, Clone)]
pub struct ApplevelCache {
    pub base: GatewayCache,
}

#[allow(non_camel_case_types)]
pub type interp2app = BuiltinCode;

#[allow(non_camel_case_types)]
pub type interp2app_temp = interp2app;

#[allow(non_camel_case_types)]
pub type applevel_temp = ApplevelClass;

pub fn build_applevel_dict(_space: PyObjectRef) -> PyObjectRef {
    std::ptr::null_mut()
}

pub fn build_unwrap_spec(
    _func: PyObjectRef,
    _argnames: &[&str],
    _self_type: Option<&str>,
) -> UnwrapSpecRecipe {
    let _ = (_func, _argnames, _self_type);
    UnwrapSpecRecipe {
        miniglobals: Vec::new(),
    }
}

pub fn int_unwrapping_space_method<T>(_typ: T) -> &'static str {
    let _ = _typ;
    "int"
}

pub fn interp2app(func: PyObjectRef) -> PyObjectRef {
    let _ = func;
    make_builtin_function("interp2app", |_| Ok(std::ptr::null_mut()))
}

pub fn interp2app_temp(func: PyObjectRef) -> PyObjectRef {
    interp2app(func)
}

pub fn interpindirect2app(
    unbound_meth: PyObjectRef,
    _unwrap_spec: Option<&UnwrapSpecRecipe>,
) -> PyObjectRef {
    let _ = _unwrap_spec;
    interp2app(unbound_meth)
}

pub fn unwrap_spec(_spec: &[&'static str]) -> PyObjectRef {
    let _ = _spec;
    make_builtin_function("unwrap", |_| Ok(std::ptr::null_mut()))
}

pub fn appdef(
    source: &'static str,
    _applevel: ApplevelClass,
    _filename: Option<&str>,
) -> PyObjectRef {
    let _ = (source, _filename);
    std::ptr::null_mut()
}

pub fn app2interp_temp(func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    let _ = _filename;
    interp2app(func)
}

pub fn app2interp(func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    app2interp_temp(func, _filename)
}

pub fn applevel_temp(_func: PyObjectRef, _filename: Option<&str>) -> PyObjectRef {
    let _ = _filename;
    std::ptr::null_mut()
}

impl UnwrapSpec_FastFunc_Unwrap {
    pub fn visit_nonnegint(&mut self) {}
}

impl UnwrapSpec_EmitShortcut {
    pub fn handle(self) {}
}

/// Type descriptor for built-in code objects.
///
/// PyPy typedef.py: BuiltinCode.typedef = TypeDef('builtin-code', ...)
pub static BUILTIN_CODE_TYPE: PyType = pyre_object::pyobject::new_pytype("builtin-code");

/// GC type id assigned to `BuiltinCode` at JitDriver init time. Held
/// as a constant alongside the struct (rather than runtime-queried) so
/// the allocation hook can reach it without a back-channel, mirroring
/// `W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`. `pyre/pyre-jit/src/eval.rs`
/// asserts the same id is returned by `gc.register_type(...)` so any
/// drift panics on startup.
pub const BUILTIN_CODE_GC_TYPE_ID: u32 = 13;

/// Signature of a built-in function.
///
/// PyPy: all interp-level functions can raise OperationError.
/// pyre equivalent: returns Result so errors propagate through the call stack.
pub type BuiltinCodeFn = fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>;

/// A built-in function object.
///
/// `docstring` mirrors PyPy `BuiltinCode.docstring` (gateway.py:673
/// `self.docstring = doc or func.__doc__`). It is consulted by
/// `BuiltinCode::getdocstring`, which is the lazy fallback used by
/// `Function.fget_func_doc` (function.py:395-398).
#[repr(C)]
pub struct BuiltinCode {
    pub ob: PyObject,
    pub name: &'static str,
    pub func: BuiltinCodeFn,
    pub docstring: Option<&'static str>,
    /// eval.py:16-23 — `fast_natural_arity`. For builtins with fixed
    /// positional arity 0-4, this equals the arity directly. Builtins
    /// with optional/variadic args use HOPELESS (0x400).
    pub fast_natural_arity: u16,
}

/// Fixed payload size used by `gct_fv_gc_malloc`'s `c_size`
/// (`framework.py:811`). The payload has no inline GC pointers (`name`
/// / `docstring` are `'static` slices, `func` is a function pointer,
/// `ob.w_class` follows the existing W_IntObject / W_FloatObject
/// convention of leaving typeptr fixups out of `gc_ptr_offsets`).
pub const BUILTIN_CODE_OBJECT_SIZE: usize = std::mem::size_of::<BuiltinCode>();

impl pyre_object::lltype::GcType for BuiltinCode {
    const TYPE_ID: u32 = BUILTIN_CODE_GC_TYPE_ID;
    const SIZE: usize = BUILTIN_CODE_OBJECT_SIZE;
}

/// eval.py:16 — `FLATPYCALL = 0x100`.
pub const FLATPYCALL: u16 = 0x100;
/// eval.py:17 — `PASSTHROUGHARGS1 = 0x200`.
pub const PASSTHROUGHARGS1: u16 = 0x200;
/// eval.py:18 — `HOPELESS = 0x400`. Default for code that cannot fast-path.
pub const HOPELESS: u16 = 0x400;

/// Allocate a new `BuiltinCode` with no docstring.
/// `fast_natural_arity` defaults to HOPELESS (no fast path).
pub fn builtin_code_new(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    builtin_code_new_with_doc(name, func, None)
}

/// Allocate a new `BuiltinCode` with known fixed arity (0-4).
/// gateway.py:843 — `self.__class__ = globals()['BuiltinCode%d' % arity]`
pub fn builtin_code_new_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    debug_assert!(arity <= 4, "builtin arity {arity} for {name} exceeds fast-path max 4");
    builtin_code_new_full(name, func, None, arity)
}

/// Allocate a new `BuiltinCode` with an explicit docstring.
///
/// PyPy gateway.py:673 — `self.docstring = doc or func.__doc__`. Pyre has
/// no introspection of `func.__doc__`, so callers must pass the docstring
/// explicitly when registering a builtin.
pub fn builtin_code_new_with_doc(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: Option<&'static str>,
) -> PyObjectRef {
    builtin_code_new_full(name, func, docstring, HOPELESS)
}

/// Full constructor for `BuiltinCode`.
fn builtin_code_new_full(
    name: &'static str,
    func: BuiltinCodeFn,
    docstring: Option<&'static str>,
    fast_natural_arity: u16,
) -> PyObjectRef {
    pyre_object::lltype::malloc_typed(BuiltinCode {
        ob: PyObject {
            ob_type: &BUILTIN_CODE_TYPE,
            w_class: pyre_object::pyobject::get_instantiate(&BUILTIN_CODE_TYPE),
        },
        name,
        func,
        docstring,
        fast_natural_arity,
    }) as PyObjectRef
}

/// Check if an object is a built-in function.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_builtin_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BUILTIN_CODE_TYPE) }
}

/// Get the function pointer from a built-in function object.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get(obj: PyObjectRef) -> BuiltinCodeFn {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).func }
}

/// eval.py:16-23 — read `fast_natural_arity` from a BuiltinCode.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    unsafe { (*(obj as *const BuiltinCode)).fast_natural_arity }
}

/// Get the name of a built-in function.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_name(obj: PyObjectRef) -> &'static str {
    let func_obj = obj as *const BuiltinCode;
    unsafe { (*func_obj).name }
}

/// gateway.py:777 BuiltinCode.getdocstring — return the stored docstring
/// wrapped as a `str`, or `None` if no docstring was attached.
///
/// # Safety
/// `obj` must point to a valid `BuiltinCode`.
#[inline]
pub unsafe fn builtin_code_get_docstring(obj: PyObjectRef) -> PyObjectRef {
    let func_obj = obj as *const BuiltinCode;
    match unsafe { (*func_obj).docstring } {
        Some(s) => pyre_object::w_str_new(s),
        None => pyre_object::w_none(),
    }
}

/// gateway.py GatewayCache.build() parity — wrap a BuiltinCodeFn as FunctionWithFixedCode.
///
/// Creates a BuiltinCode (Code object) and wraps it in a Function with
/// `can_change_code = false`, matching PyPy's:
///   `fn = FunctionWithFixedCode(space, code, None, defs, forcename=gateway.name)`
pub fn make_builtin_function(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), std::ptr::null_mut())
}

/// `make_builtin_function` with known fixed arity for fast-path dispatch.
pub fn make_builtin_function_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_with_fixed_code(code as *const (), name.to_string(), std::ptr::null_mut())
}

/// mixedmodule.py:116 parity — wrap a BuiltinCodeFn as BuiltinFunction.
///
/// Module-level builtins are not descriptors: storing them on a user class
/// must not synthesize a bound method.
pub fn make_module_builtin_function(name: &'static str, func: BuiltinCodeFn) -> PyObjectRef {
    let code = builtin_code_new(name, func);
    crate::function_new_builtin(code as *const (), name.to_string(), std::ptr::null_mut())
}

/// `make_module_builtin_function` with known fixed arity for fast-path dispatch.
pub fn make_module_builtin_function_with_arity(
    name: &'static str,
    func: BuiltinCodeFn,
    arity: u16,
) -> PyObjectRef {
    let code = builtin_code_new_with_arity(name, func, arity);
    crate::function_new_builtin(code as *const (), name.to_string(), std::ptr::null_mut())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guard against drift between the constant colocated with
    /// `BuiltinCode` and the id that `pyre-jit/src/eval.rs` asserts at
    /// JitDriver init. Mirror of the W_INT/W_FLOAT trip-wire tests.
    #[test]
    fn builtin_code_gc_type_id_matches_descr() {
        assert_eq!(BUILTIN_CODE_GC_TYPE_ID, 13);
        assert_eq!(
            <BuiltinCode as pyre_object::lltype::GcType>::TYPE_ID,
            BUILTIN_CODE_GC_TYPE_ID
        );
        assert_eq!(
            <BuiltinCode as pyre_object::lltype::GcType>::SIZE,
            std::mem::size_of::<BuiltinCode>()
        );
    }

    /// pypy/interpreter/signature.py:33-46 accessor parity:
    /// `def f(a, b, /, c, d, *args, e, f, **kwargs): ...`
    /// → argnames=[a,b,c,d,e,f], varargname=args, kwargname=kwargs,
    /// posonlyargcount=2, kwonlyargcount=2.
    #[test]
    fn signature_accessor_parity() {
        let sig = Signature::new(
            vec!["a", "b", "c", "d", "e", "f"],
            Some("args"),
            Some("kwargs"),
            2,
            2,
        );
        // num_argnames = len(argnames) - kwonlyargcount = 6 - 2 = 4
        assert_eq!(sig.num_argnames(), 4);
        assert_eq!(sig.num_posonlyargnames(), 2);
        assert_eq!(sig.num_kwonlyargnames(), 2);
        assert!(sig.has_vararg());
        assert!(sig.has_kwarg());
        // scope_length = len(argnames) + has_vararg + has_kwarg = 6 + 1 + 1 = 8
        assert_eq!(sig.scope_length(), 8);
        // find_argname returns -1 for unknown
        assert_eq!(sig.find_argname("a"), 0);
        assert_eq!(sig.find_argname("e"), 4);
        assert_eq!(sig.find_argname("missing"), -1);
        // getallvarnames appends varargname + kwargname
        assert_eq!(
            sig.getallvarnames(),
            vec!["a", "b", "c", "d", "e", "f", "args", "kwargs"],
        );
    }

    /// `def f(a, b): ...` — no *args / **kwargs / kwonly.
    #[test]
    fn signature_minimal_no_extras() {
        let sig = Signature::new(vec!["a", "b"], None, None, 0, 0);
        assert_eq!(sig.num_argnames(), 2);
        assert_eq!(sig.num_kwonlyargnames(), 0);
        assert!(!sig.has_vararg());
        assert!(!sig.has_kwarg());
        assert_eq!(sig.scope_length(), 2);
        assert_eq!(sig.getallvarnames(), vec!["a", "b"]);
    }
}
