use std::slice;
use std::sync::OnceLock;

use crate::bytecode::{BinaryOperator, ComparisonOperator};
use pyre_object::{
    PY_NULL, PyObjectRef, W_SeqIterator, is_instance, is_list, is_range_iter, is_seq_iter, is_str,
    is_tuple, w_dict_new, w_dict_store, w_int_get_value, w_int_new, w_list_getitem, w_list_len,
    w_list_new, w_range_iter_has_next, w_range_iter_next, w_str_get_value, w_str_new,
    w_tuple_getitem, w_tuple_len, w_tuple_new,
};

use crate::{
    DictStorage, PyError, PyErrorKind, builtin_code_get, function_get_code, function_new,
    is_builtin_code, is_function, w_code_get_ptr,
};

pub fn make_function_from_code_obj(
    code_obj: PyObjectRef,
    globals: *mut DictStorage,
) -> PyObjectRef {
    let code_ptr = unsafe { w_code_get_ptr(code_obj) };
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    // Store the W_CodeObject (Code-level wrapper) in Function.code,
    // not the raw CodeObject pointer.
    function_new(code_obj as *const (), code.qualname.to_string(), globals)
}

fn decode_name(name_ptr: i64, name_len: i64) -> Option<&'static str> {
    if name_ptr == 0 || name_len < 0 {
        return None;
    }
    let bytes = unsafe { slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    std::str::from_utf8(bytes).ok()
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_make_function_from_globals(globals: i64, code_obj: i64) -> i64 {
    make_function_from_code_obj(code_obj as PyObjectRef, globals as *mut DictStorage) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_load_name_from_namespace(
    namespace_ptr: i64,
    name_ptr: i64,
    name_len: i64,
) -> i64 {
    let namespace_ptr = namespace_ptr as *mut DictStorage;
    let Some(namespace) = (!namespace_ptr.is_null()).then_some(unsafe { &mut *namespace_ptr })
    else {
        return 0;
    };
    let Some(name) = decode_name(name_ptr, name_len) else {
        return 0;
    };
    dict_storage_get(namespace, name).unwrap_or(std::ptr::null_mut()) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_store_name_to_namespace(
    namespace_ptr: i64,
    name_ptr: i64,
    name_len: i64,
    value: i64,
) -> i64 {
    let namespace_ptr = namespace_ptr as *mut DictStorage;
    let Some(namespace) = (!namespace_ptr.is_null()).then_some(unsafe { &mut *namespace_ptr })
    else {
        return 0;
    };
    let Some(name) = decode_name(name_ptr, name_len) else {
        return 0;
    };
    dict_storage_store(namespace, name, value as PyObjectRef);
    0
}

type JitFunctionCaller =
    extern "C" fn(frame_ptr: i64, callable: i64, args: *const i64, nargs: i64) -> i64;

static JIT_FUNCTION_CALLER: OnceLock<JitFunctionCaller> = OnceLock::new();

pub fn register_jit_function_caller(caller: JitFunctionCaller) {
    let _ = JIT_FUNCTION_CALLER.set(caller);
}

fn call_builtin_with_args(callable: i64, args: &[i64]) -> i64 {
    let callable = callable as PyObjectRef;
    unsafe {
        let code = crate::getcode(callable);
        let func = builtin_code_get(code as PyObjectRef);
        let arg_slice = std::slice::from_raw_parts(args.as_ptr() as *const PyObjectRef, args.len());
        match func(arg_slice) {
            Ok(result) => result as i64,
            Err(e) => panic!("jit builtin call failed: {e}"),
        }
    }
}

fn call_user_function_with_args(frame_ptr: i64, callable: i64, args: &[i64]) -> i64 {
    let Some(caller) = JIT_FUNCTION_CALLER.get().copied() else {
        let callable = callable as PyObjectRef;
        let code_ptr = unsafe { function_get_code(callable) };
        panic!("jit function caller bridge is not installed for code_ptr={code_ptr:p}");
    };
    caller(frame_ptr, callable, args.as_ptr(), args.len() as i64)
}

fn call_callable_with_args(frame_ptr: i64, callable: i64, args: &[i64]) -> i64 {
    let _ = frame_ptr;
    let callable_ref = callable as PyObjectRef;
    let arg_slice =
        unsafe { std::slice::from_raw_parts(args.as_ptr() as *const PyObjectRef, args.len()) };
    match crate::call::call_function_impl_result(callable_ref, arg_slice) {
        Ok(result) => result as i64,
        Err(err) => panic!("jit callable dispatch failed: {err}"),
    }
}

macro_rules! define_callable_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(frame_ptr: i64, callable: i64 $(, $arg: i64)*) -> i64 {
            call_callable_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_builtin_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(callable: i64 $(, $arg: i64)*) -> i64 {
            call_builtin_with_args(callable, &[$($arg),*])
        }
    };
}

macro_rules! define_known_function_call_helper {
    ($name:ident $(, $arg:ident)*) => {
        #[majit_macros::jit_may_force]
        pub extern "C" fn $name(frame_ptr: i64, callable: i64 $(, $arg: i64)*) -> i64 {
            call_user_function_with_args(frame_ptr, callable, &[$($arg),*])
        }
    };
}

macro_rules! define_flat_ref_helper {
    ($inner:ident, $name:ident $(, $arg:ident)*) => {
        pub extern "C" fn $name($($arg: i64),*) -> i64 {
            $inner(&[$($arg),*])
        }
    };
}

define_callable_call_helper!(jit_call_callable_0);
define_callable_call_helper!(jit_call_callable_1, arg0);
define_callable_call_helper!(jit_call_callable_2, arg0, arg1);
define_callable_call_helper!(jit_call_callable_3, arg0, arg1, arg2);
define_callable_call_helper!(jit_call_callable_4, arg0, arg1, arg2, arg3);
define_callable_call_helper!(jit_call_callable_5, arg0, arg1, arg2, arg3, arg4);
define_callable_call_helper!(jit_call_callable_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_callable_call_helper!(
    jit_call_callable_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_callable_call_helper!(
    jit_call_callable_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_builtin_call_helper!(jit_call_known_builtin_0);
define_known_builtin_call_helper!(jit_call_known_builtin_1, arg0);
define_known_builtin_call_helper!(jit_call_known_builtin_2, arg0, arg1);
define_known_builtin_call_helper!(jit_call_known_builtin_3, arg0, arg1, arg2);
define_known_builtin_call_helper!(jit_call_known_builtin_4, arg0, arg1, arg2, arg3);
define_known_builtin_call_helper!(jit_call_known_builtin_5, arg0, arg1, arg2, arg3, arg4);
define_known_builtin_call_helper!(jit_call_known_builtin_6, arg0, arg1, arg2, arg3, arg4, arg5);
define_known_builtin_call_helper!(
    jit_call_known_builtin_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_builtin_call_helper!(
    jit_call_known_builtin_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_known_function_call_helper!(jit_call_known_function_0);
define_known_function_call_helper!(jit_call_known_function_1, arg0);
define_known_function_call_helper!(jit_call_known_function_2, arg0, arg1);
define_known_function_call_helper!(jit_call_known_function_3, arg0, arg1, arg2);
define_known_function_call_helper!(jit_call_known_function_4, arg0, arg1, arg2, arg3);
define_known_function_call_helper!(jit_call_known_function_5, arg0, arg1, arg2, arg3, arg4);
define_known_function_call_helper!(
    jit_call_known_function_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_known_function_call_helper!(
    jit_call_known_function_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_known_function_call_helper!(
    jit_call_known_function_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

pub fn dispatch_callable<R, FBuiltin, FUser>(
    callable: PyObjectRef,
    on_builtin: FBuiltin,
    on_user: FUser,
) -> Result<R, PyError>
where
    FBuiltin: FnOnce(PyObjectRef) -> Result<R, PyError>,
    FUser: FnOnce(PyObjectRef) -> Result<R, PyError>,
{
    // Drain any pending JIT-prologue overflow first so a backend
    // probe that already detected an overflow surfaces here as the
    // user-visible RecursionError. The pending-exception slot is
    // populated by the backend slowpath wrapper when the JIT
    // prologue probe trips — backend raises, glue propagates
    // (rpython/rlib/rstack.py:68-73 stack_check_slowpath parity).
    crate::stack_check::drain_jit_pending_exception()?;
    // rpython/rlib/rstack.py:42 stack_check(): every interpreter call
    // boundary also checks the native stack synchronously, so deep
    // interpreter recursion (no JIT involved) raises RecursionError
    // instead of letting the OS abort on a guard-page hit.
    crate::stack_check::stack_check()?;
    unsafe {
        if is_function(callable) {
            // All callables are Function objects. Check code type to distinguish
            // builtins (BuiltinCode) from user functions (W_CodeObject).
            let code = crate::getcode(callable);
            if is_builtin_code(code as PyObjectRef) {
                on_builtin(callable)
            } else {
                on_user(callable)
            }
        } else {
            Err(PyError::type_error(format!(
                "'{}' object is not callable",
                (*(*callable).ob_type).name
            )))
        }
    }
}

pub fn binary_op_tag(op: BinaryOperator) -> Option<i64> {
    Some(match op {
        BinaryOperator::Add | BinaryOperator::InplaceAdd => 0,
        BinaryOperator::Subtract | BinaryOperator::InplaceSubtract => 1,
        BinaryOperator::Multiply | BinaryOperator::InplaceMultiply => 2,
        BinaryOperator::FloorDivide | BinaryOperator::InplaceFloorDivide => 3,
        BinaryOperator::Remainder | BinaryOperator::InplaceRemainder => 4,
        BinaryOperator::TrueDivide | BinaryOperator::InplaceTrueDivide => 5,
        BinaryOperator::Subscr => 6,
        BinaryOperator::Power | BinaryOperator::InplacePower => 7,
        BinaryOperator::Lshift | BinaryOperator::InplaceLshift => 8,
        BinaryOperator::Rshift | BinaryOperator::InplaceRshift => 9,
        BinaryOperator::And | BinaryOperator::InplaceAnd => 10,
        BinaryOperator::Or | BinaryOperator::InplaceOr => 11,
        BinaryOperator::Xor | BinaryOperator::InplaceXor => 12,
        _ => return None,
    })
}

/// Reverse of binary_op_tag: tag (0-12) → BinaryOperator.
/// The blackhole interpreter receives the compact tag from the codewriter
/// and needs to recover the original operator for binary_value dispatch.
pub fn binary_op_from_tag(tag: i64) -> Option<BinaryOperator> {
    Some(match tag {
        0 => BinaryOperator::Add,
        1 => BinaryOperator::Subtract,
        2 => BinaryOperator::Multiply,
        3 => BinaryOperator::FloorDivide,
        4 => BinaryOperator::Remainder,
        5 => BinaryOperator::TrueDivide,
        6 => BinaryOperator::Subscr,
        7 => BinaryOperator::Power,
        8 => BinaryOperator::Lshift,
        9 => BinaryOperator::Rshift,
        10 => BinaryOperator::And,
        11 => BinaryOperator::Or,
        12 => BinaryOperator::Xor,
        _ => return None,
    })
}

pub fn compare_op_tag(op: ComparisonOperator) -> i64 {
    match op {
        ComparisonOperator::Less => 0,
        ComparisonOperator::LessOrEqual => 1,
        ComparisonOperator::Greater => 2,
        ComparisonOperator::GreaterOrEqual => 3,
        ComparisonOperator::Equal => 4,
        ComparisonOperator::NotEqual => 5,
    }
}

/// Reverse of compare_op_tag: tag (0-5) → ComparisonOperator.
pub fn compare_op_from_tag(tag: i64) -> Option<ComparisonOperator> {
    Some(match tag {
        0 => ComparisonOperator::Less,
        1 => ComparisonOperator::LessOrEqual,
        2 => ComparisonOperator::Greater,
        3 => ComparisonOperator::GreaterOrEqual,
        4 => ComparisonOperator::Equal,
        5 => ComparisonOperator::NotEqual,
        _ => return None,
    })
}

pub fn build_list_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    w_list_new(items.to_vec())
}

pub fn build_tuple_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    w_tuple_new(items.to_vec())
}

pub fn build_map_from_refs(items: &[PyObjectRef]) -> PyObjectRef {
    let dict = w_dict_new();
    for pair in items.chunks_exact(2) {
        let key = pair[0];
        let value = pair[1];
        unsafe {
            w_dict_store(dict, key, value);
        }
    }
    dict
}

fn build_list_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    build_list_from_refs(&items) as i64
}

fn build_tuple_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    build_tuple_from_refs(&items) as i64
}

fn build_map_from_args(args: &[i64]) -> i64 {
    let items: Vec<_> = args.iter().map(|&arg| arg as PyObjectRef).collect();
    build_map_from_refs(&items) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_list_0() -> i64 {
    w_list_new(vec![]) as i64
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_build_tuple_0() -> i64 {
    w_tuple_new(vec![]) as i64
}

define_flat_ref_helper!(build_list_from_args, jit_build_list_1, arg0);
define_flat_ref_helper!(build_list_from_args, jit_build_list_2, arg0, arg1);
define_flat_ref_helper!(build_list_from_args, jit_build_list_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_list_from_args,
    jit_build_list_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_1, arg0);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_2, arg0, arg1);
define_flat_ref_helper!(build_tuple_from_args, jit_build_tuple_3, arg0, arg1, arg2);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_4,
    arg0,
    arg1,
    arg2,
    arg3
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_5,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_6,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_7,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6
);
define_flat_ref_helper!(
    build_tuple_from_args,
    jit_build_tuple_8,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

define_flat_ref_helper!(build_map_from_args, jit_build_map_0);
define_flat_ref_helper!(build_map_from_args, jit_build_map_1, arg0, arg1);
define_flat_ref_helper!(build_map_from_args, jit_build_map_2, arg0, arg1, arg2, arg3);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_3,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5
);
define_flat_ref_helper!(
    build_map_from_args,
    jit_build_map_4,
    arg0,
    arg1,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7
);

pub fn callable_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_callable_0 as *const (),
        1 => jit_call_callable_1 as *const (),
        2 => jit_call_callable_2 as *const (),
        3 => jit_call_callable_3 as *const (),
        4 => jit_call_callable_4 as *const (),
        5 => jit_call_callable_5 as *const (),
        6 => jit_call_callable_6 as *const (),
        7 => jit_call_callable_7 as *const (),
        8 => jit_call_callable_8 as *const (),
        _ => return None,
    })
}

pub fn known_builtin_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_known_builtin_0 as *const (),
        1 => jit_call_known_builtin_1 as *const (),
        2 => jit_call_known_builtin_2 as *const (),
        3 => jit_call_known_builtin_3 as *const (),
        4 => jit_call_known_builtin_4 as *const (),
        5 => jit_call_known_builtin_5 as *const (),
        6 => jit_call_known_builtin_6 as *const (),
        7 => jit_call_known_builtin_7 as *const (),
        8 => jit_call_known_builtin_8 as *const (),
        _ => return None,
    })
}

pub fn known_function_call_helper(nargs: usize) -> Option<*const ()> {
    Some(match nargs {
        0 => jit_call_known_function_0 as *const (),
        1 => jit_call_known_function_1 as *const (),
        2 => jit_call_known_function_2 as *const (),
        3 => jit_call_known_function_3 as *const (),
        4 => jit_call_known_function_4 as *const (),
        5 => jit_call_known_function_5 as *const (),
        6 => jit_call_known_function_6 as *const (),
        7 => jit_call_known_function_7 as *const (),
        8 => jit_call_known_function_8 as *const (),
        _ => return None,
    })
}

#[derive(Clone, Copy)]
pub enum FlatBuildKind {
    List,
    Tuple,
    Map,
}

pub fn list_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => jit_build_list_0 as *const (),
        1 => jit_build_list_1 as *const (),
        2 => jit_build_list_2 as *const (),
        3 => jit_build_list_3 as *const (),
        4 => jit_build_list_4 as *const (),
        5 => jit_build_list_5 as *const (),
        6 => jit_build_list_6 as *const (),
        7 => jit_build_list_7 as *const (),
        8 => jit_build_list_8 as *const (),
        _ => return None,
    })
}

pub fn tuple_build_helper(count: usize) -> Option<*const ()> {
    Some(match count {
        0 => jit_build_tuple_0 as *const (),
        1 => jit_build_tuple_1 as *const (),
        2 => jit_build_tuple_2 as *const (),
        3 => jit_build_tuple_3 as *const (),
        4 => jit_build_tuple_4 as *const (),
        5 => jit_build_tuple_5 as *const (),
        6 => jit_build_tuple_6 as *const (),
        7 => jit_build_tuple_7 as *const (),
        8 => jit_build_tuple_8 as *const (),
        _ => return None,
    })
}

pub fn map_build_helper(pair_count: usize) -> Option<*const ()> {
    Some(match pair_count {
        0 => jit_build_map_0 as *const (),
        1 => jit_build_map_1 as *const (),
        2 => jit_build_map_2 as *const (),
        3 => jit_build_map_3 as *const (),
        4 => jit_build_map_4 as *const (),
        _ => return None,
    })
}

pub fn flat_build_helper(kind: FlatBuildKind, count: usize) -> Option<*const ()> {
    match kind {
        FlatBuildKind::List => list_build_helper(count),
        FlatBuildKind::Tuple => tuple_build_helper(count),
        FlatBuildKind::Map => map_build_helper(count),
    }
}

pub fn dict_storage_get(namespace: &DictStorage, name: &str) -> Option<PyObjectRef> {
    namespace.get(name).copied()
}

pub fn dict_storage_load(namespace: &DictStorage, name: &str) -> Result<PyObjectRef, PyError> {
    dict_storage_get(namespace, name).ok_or_else(|| {
        PyError::new(
            PyErrorKind::NameError,
            format!("name '{name}' is not defined"),
        )
    })
}

pub fn dict_storage_store(namespace: &mut DictStorage, name: &str, value: PyObjectRef) {
    namespace.insert(name.to_string(), value);
}

pub fn dict_storage_delete(namespace: &mut DictStorage, name: &str) -> bool {
    namespace.remove(name).is_some()
}

pub fn sequence_len(seq: PyObjectRef) -> Result<usize, PyError> {
    unsafe {
        if is_tuple(seq) {
            return Ok(w_tuple_len(seq));
        }
        if is_list(seq) {
            return Ok(w_list_len(seq));
        }
        if is_str(seq) {
            return Ok(w_str_get_value(seq).chars().count());
        }
        // Try __len__ on instances
        if is_instance(seq) {
            if let Ok(len_val) = crate::baseobjspace::len(seq) {
                return Ok(w_int_get_value(len_val) as usize);
            }
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            (*(*seq).ob_type).name
        )))
    }
}

pub fn sequence_getitem(seq: PyObjectRef, index: usize) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_tuple(seq) {
            return w_tuple_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("tuple index out of range"));
        }
        if is_list(seq) {
            return w_list_getitem(seq, index as i64)
                .ok_or_else(|| PyError::type_error("list index out of range"));
        }
        if is_str(seq) {
            let s = w_str_get_value(seq);
            return s
                .chars()
                .nth(index)
                .map(|c| {
                    let mut buf = [0u8; 4];
                    w_str_new(c.encode_utf8(&mut buf))
                })
                .ok_or_else(|| PyError::type_error("string index out of range"));
        }
        // Try getitem for instances
        if is_instance(seq) {
            return crate::baseobjspace::getitem(seq, w_int_new(index as i64));
        }
        Err(PyError::type_error(format!(
            "cannot unpack non-sequence {}",
            (*(*seq).ob_type).name
        )))
    }
}

#[majit_macros::jit_may_force]
pub extern "C" fn jit_sequence_getitem(seq: i64, index: i64) -> i64 {
    match sequence_getitem(seq as PyObjectRef, index as usize) {
        Ok(value) => value as i64,
        // Return PY_NULL on out-of-bounds — the guard after this call
        // will detect the null and side-exit to the interpreter.
        // RPython: residual calls that fail trigger guard failure, not crash.
        Err(_) => pyre_object::PY_NULL as i64,
    }
}

pub fn unpack_sequence_exact(seq: PyObjectRef, count: usize) -> Result<Vec<PyObjectRef>, PyError> {
    // Fast path for known sequence types
    if let Ok(len) = sequence_len(seq) {
        if len != count {
            return Err(PyError::type_error(format!(
                "not enough values to unpack (expected {count}, got {len})"
            )));
        }
        return (0..count).map(|idx| sequence_getitem(seq, idx)).collect();
    }
    // Fallback: iteration protocol (handles type objects with metaclass __iter__, etc.)
    // PyPy: ObjSpace.unpackiterable
    let iter = crate::baseobjspace::iter(seq)?;
    let mut items = Vec::with_capacity(count);
    for _ in 0..count {
        match crate::baseobjspace::next(iter) {
            Ok(val) => items.push(val),
            Err(e) if e.kind == PyErrorKind::StopIteration => {
                return Err(PyError::type_error(format!(
                    "not enough values to unpack (expected {count}, got {})",
                    items.len()
                )));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

pub fn ensure_range_iter(iter: PyObjectRef) -> Result<(), PyError> {
    unsafe {
        if is_range_iter(iter) || is_seq_iter(iter) {
            return Ok(());
        }
        // Convert list/tuple to seq iterator
        if is_list(iter) {
            // Replace TOS with a seq iterator wrapping the list
            // This is called on TOS after GET_ITER pops and pushes.
            // But ensure_iter is called ON the iter — it can't replace stack.
            // So we need to create iter before calling ensure.
            // Actually ensure_range_iter is called AFTER get_iter pushes.
            // The problem: GET_ITER calls ensure_iter_value on the pushed value.
            // For list, we need to push a seq_iter instead of the list itself.
        }
    }
    Err(PyError::type_error(format!(
        "'{}' object is not iterable",
        unsafe { (*(*iter).ob_type).name }
    )))
}

pub fn range_iter_continues(iter: PyObjectRef) -> Result<bool, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_has_next(iter));
        }
        if is_seq_iter(iter) {
            let si = &*(iter as *const W_SeqIterator);
            return Ok(si.index < si.length);
        }
    }
    Err(PyError::type_error("not an iterator"))
}

pub fn range_iter_next_or_null(iter: PyObjectRef) -> Result<PyObjectRef, PyError> {
    unsafe {
        if is_range_iter(iter) {
            return Ok(w_range_iter_next(iter).unwrap_or(PY_NULL));
        }
        if is_seq_iter(iter) {
            let si = &mut *(iter as *mut W_SeqIterator);
            if si.index < si.length {
                let idx = si.index;
                si.index += 1;
                if is_list(si.seq) {
                    return Ok(w_list_getitem(si.seq, idx).unwrap_or(PY_NULL));
                }
                if is_tuple(si.seq) {
                    return Ok(w_tuple_getitem(si.seq, idx).unwrap_or(PY_NULL));
                }
            }
            return Ok(PY_NULL);
        }
    }
    Err(PyError::type_error("not an iterator"))
}

#[majit_macros::dont_look_inside]
pub extern "C" fn jit_range_iter_next_or_null(iter: i64) -> i64 {
    match range_iter_next_or_null(iter as PyObjectRef) {
        Ok(value) => value as i64,
        Err(err) => panic!("range iter next failed in JIT: {err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PyExecutionContext;
    use pyre_object::{w_int_get_value, w_int_new};

    #[test]
    fn test_dispatch_callable_runs_builtin_branch() {
        let ctx = PyExecutionContext::default();
        let abs = ctx.lookup_builtin("abs").expect("abs builtin must exist");
        let result = dispatch_callable(
            abs,
            |callable| {
                // Builtins are now Function objects; extract code then func pointer.
                let code = unsafe { crate::getcode(callable) };
                let func = unsafe { crate::builtin_code_get(code as PyObjectRef) };
                func(&[w_int_new(-9)])
            },
            |_callable| panic!("builtin callable should not take user branch"),
        )
        .expect("builtin dispatch should succeed");

        unsafe {
            assert_eq!(w_int_get_value(result), 9);
        }
    }

    #[test]
    fn test_dispatch_callable_rejects_non_callable() {
        let err = dispatch_callable(w_int_new(3), |_callable| Ok(()), |_callable| Ok(()))
            .expect_err("non-callable dispatch should fail");

        assert!(matches!(err.kind, PyErrorKind::TypeError));
        assert!(err.message.contains("not callable"));
    }

    #[test]
    fn test_range_iter_helpers_share_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 3, 1);
        assert!(range_iter_continues(iter).unwrap());
        let first = range_iter_next_or_null(iter).unwrap();
        let second = range_iter_next_or_null(iter).unwrap();
        let done = range_iter_next_or_null(iter).unwrap();
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_range_iter_helper_shares_iterator_semantics() {
        let iter = pyre_object::w_range_iter_new(1, 3, 1);
        let first = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let second = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        let done = jit_range_iter_next_or_null(iter as i64) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(first), 1);
            assert_eq!(w_int_get_value(second), 2);
            assert!(done.is_null());
        }
    }

    #[test]
    fn test_jit_sequence_getitem_shares_runtime_sequence_semantics() {
        let tuple = pyre_object::w_tuple_new(vec![w_int_new(3), w_int_new(5)]);
        let item = jit_sequence_getitem(tuple as i64, 1) as PyObjectRef;
        unsafe {
            assert_eq!(w_int_get_value(item), 5);
        }
    }
}
