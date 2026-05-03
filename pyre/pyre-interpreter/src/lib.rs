//! pyre-interpreter — the Python interpreter.
//!
//! PyPy equivalent: pypy/interpreter/
//!
//! Contains the object space (baseobjspace.rs), bytecode evaluation (eval.rs),
//! frame management (pyframe.rs), function call dispatch (call.rs),
//! import machinery (importing.rs), builtin functions (builtins.rs),
//! type definitions (typedef.rs), and builtin modules (module/).

// ── Bytecode / compiler re-exports (was pyre-bytecode) ──
pub mod compile;
pub use compile::*;

// ── Core interpreter modules ──
pub mod argument;
pub mod baseobjspace;
pub mod builtins;
pub mod display;
pub mod error;
pub mod executioncontext;
pub mod frame_array;
pub mod function;
pub mod gateway;
pub mod jit_fnaddr;
pub mod listobject;
pub mod opcode_ops;
pub mod pycode;
pub mod pyopcode;
pub mod runtime_ops;
pub mod shared_opcode;
pub mod sliceobject;
pub mod stack_check;
pub mod type_methods;
pub mod typedef;
pub mod warn;

// ── Execution and import modules ──
pub mod call;
pub mod eval;
pub mod importing;
pub mod module;
pub mod objspace;
pub mod pyframe;

// ── Re-exports ──
pub use baseobjspace::*;
pub use builtins::*;
pub use display::*;
pub use error::*;
pub use executioncontext::*;
pub use frame_array::*;
pub use function::*;
pub use gateway::{
    BUILTIN_CODE_TYPE, BuiltinCode, BuiltinCodeFn, builtin_code_get, builtin_code_name,
    builtin_code_new, is_builtin_code, make_builtin_function, make_module_builtin_function,
};
pub use jit_fnaddr::*;
pub use malachite_bigint::BigInt as PyBigInt;
pub use opcode_ops::*;
pub use pycode::*;
pub use pyframe::*;
pub use pyopcode::*;
pub use runtime_ops::*;
pub use shared_opcode::*;

/// Every interpreter-level `PyType` static that represents a
/// `PyObject`-layout type (instances carry `ob_type` at offset 0,
/// matching `rclass.OBJECT`), paired with its parent class.
///
/// Same shape as `pyre_object::pyobject::all_foreign_pytypes`: each
/// entry is a `(type, parent)` tuple consumed by the JIT registration
/// loop in `pyre/pyre-jit/src/eval.rs`. The parent feeds
/// `TypeInfo::object_subclass` so `assign_inheritance_ids`
/// (normalizecalls.py:373-389) computes the right preorder bounds.
///
/// These live here rather than in `pyre_object::pyobject` because
/// `pyre-object` cannot depend on `pyre-interpreter`.
///
/// `BUILTIN_CODE_TYPE`, `FUNCTION_TYPE`, `BUILTIN_FUNCTION_TYPE` are
/// intentionally absent: they get their own ids
/// (`BUILTIN_CODE_GC_TYPE_ID`, `FUNCTION_GC_TYPE_ID`) because the GC
/// needs the actual payload size and inline `PyObjectRef` field
/// offsets, neither of which the foreign-pytype loop can derive from
/// `sizeof(PyObject)`. Future types whose instances reach the GC
/// nursery should follow the same pre-registration pattern (see
/// `eval.rs` BuiltinCode / Function blocks).
pub fn all_foreign_pytypes() -> &'static [(
    &'static pyre_object::pyobject::PyType,
    &'static pyre_object::pyobject::PyType,
)] {
    static PYTYPES: &[(
        &pyre_object::pyobject::PyType,
        &pyre_object::pyobject::PyType,
    )] = &[(&crate::pycode::CODE_TYPE, &pyre_object::INSTANCE_TYPE)];
    PYTYPES
}

// ── Print hook for wasm (stdout capture) ──
use std::cell::RefCell;
thread_local! {
    static PRINT_HOOK: RefCell<Option<fn(&str)>> = RefCell::new(None);
}

/// Set a hook that receives all `print()` output instead of stdout.
pub fn set_print_hook(hook: fn(&str)) {
    PRINT_HOOK.with(|h| *h.borrow_mut() = Some(hook));
}

/// Write a string through the print hook (if set) or stdout.
pub fn print_output(s: &str) {
    PRINT_HOOK.with(|h| {
        if let Some(hook) = *h.borrow() {
            hook(s);
        } else {
            print!("{s}");
        }
    });
}

// baseobjspace call helpers are re-exported from `baseobjspace`.
