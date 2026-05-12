use malachite_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::executioncontext::DictStorage;
use crate::{
    PyDisplay, make_builtin_function, make_builtin_function_with_arity,
    make_module_builtin_function, make_module_builtin_function_with_arity,
};
use pyre_object::*;

/// Install the default builtins into a namespace.
pub fn install_default_builtins(namespace: &mut DictStorage) {
    namespace.get_or_insert_with("print", || {
        make_module_builtin_function("print", builtin_print)
    });
    namespace.get_or_insert_with("range", || {
        make_module_builtin_function("range", builtin_range)
    });
    namespace.get_or_insert_with("len", || {
        make_module_builtin_function_with_arity("len", builtin_len, 1)
    });
    namespace.get_or_insert_with("abs", || {
        make_module_builtin_function_with_arity("abs", builtin_abs, 1)
    });
    namespace.get_or_insert_with("min", || make_module_builtin_function("min", builtin_min));
    namespace.get_or_insert_with("max", || make_module_builtin_function("max", builtin_max));
    namespace.get_or_insert_with("type", || crate::typedef::w_type());
    namespace.get_or_insert_with("isinstance", || {
        make_module_builtin_function_with_arity("isinstance", builtin_isinstance, 2)
    });
    namespace.get_or_insert_with("str", || crate::typedef::gettypeobject(&STR_TYPE));
    namespace.get_or_insert_with("repr", || {
        make_module_builtin_function_with_arity("repr", builtin_repr, 1)
    });
    namespace.get_or_insert_with("int", || crate::typedef::gettypeobject(&INT_TYPE));
    namespace.get_or_insert_with("float", || crate::typedef::gettypeobject(&FLOAT_TYPE));
    namespace.get_or_insert_with("bool", || crate::typedef::gettypeobject(&BOOL_TYPE));
    namespace.get_or_insert_with("True", || w_bool_from(true));
    namespace.get_or_insert_with("False", || w_bool_from(false));
    namespace.get_or_insert_with("None", || w_none());
    namespace.get_or_insert_with("NotImplemented", || w_not_implemented());
    namespace.get_or_insert_with("hasattr", || {
        make_module_builtin_function_with_arity("hasattr", builtin_hasattr, 2)
    });
    namespace.get_or_insert_with("getattr", || {
        make_module_builtin_function("getattr", builtin_getattr)
    });
    namespace.get_or_insert_with("setattr", || {
        make_module_builtin_function_with_arity("setattr", builtin_setattr, 3)
    });
    namespace.get_or_insert_with("delattr", || {
        make_module_builtin_function_with_arity("delattr", builtin_delattr, 2)
    });
    namespace.get_or_insert_with("tuple", || crate::typedef::gettypeobject(&TUPLE_TYPE));
    namespace.get_or_insert_with("list", || crate::typedef::gettypeobject(&LIST_TYPE));
    namespace.get_or_insert_with("dict", || crate::typedef::gettypeobject(&DICT_TYPE));
    namespace.get_or_insert_with("object", || {
        // `object` is a W_TypeObject, not a builtin function.
        // PyPy: baseobjspace.py w_object = W_TypeObject("object", ...)
        crate::typedef::w_object()
    });
    namespace.get_or_insert_with("super", || {
        make_module_builtin_function("super", builtin_super)
    });
    namespace.get_or_insert_with("id", || {
        make_module_builtin_function_with_arity("id", builtin_id, 1)
    });
    namespace.get_or_insert_with("hash", || {
        make_module_builtin_function_with_arity("hash", builtin_hash, 1)
    });
    namespace.get_or_insert_with("ord", || {
        make_module_builtin_function_with_arity("ord", builtin_ord, 1)
    });
    namespace.get_or_insert_with("chr", || {
        make_module_builtin_function_with_arity("chr", builtin_chr, 1)
    });
    namespace.get_or_insert_with("map", || make_module_builtin_function("map", builtin_map));
    namespace.get_or_insert_with("zip", || make_module_builtin_function("zip", builtin_zip));
    namespace.get_or_insert_with("enumerate", || {
        make_module_builtin_function("enumerate", builtin_enumerate)
    });
    namespace.get_or_insert_with("reversed", || {
        make_module_builtin_function_with_arity("reversed", builtin_reversed, 1)
    });
    namespace.get_or_insert_with("sorted", || {
        make_module_builtin_function("sorted", builtin_sorted)
    });
    namespace.get_or_insert_with("iter", || {
        make_module_builtin_function("iter", builtin_iter)
    });
    namespace.get_or_insert_with("next", || {
        make_module_builtin_function("next", builtin_next)
    });
    namespace.get_or_insert_with("callable", || {
        make_module_builtin_function_with_arity("callable", builtin_callable, 1)
    });
    namespace.get_or_insert_with("vars", || {
        make_module_builtin_function("vars", builtin_vars)
    });
    namespace.get_or_insert_with("dir", || make_module_builtin_function("dir", builtin_dir));
    namespace.get_or_insert_with("__build_class__", || {
        make_module_builtin_function("__build_class__", |args| {
            crate::call::real_build_class(args)
        })
    });
    // bytearrayobject.py W_BytearrayObject — register the real type
    // (callable as a constructor and usable in isinstance(x, bytearray)).
    namespace.get_or_insert_with("bytearray", || {
        crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE)
    });
    // bytesobject.py W_BytesObject — immutable bytes type.
    namespace.get_or_insert_with("bytes", || {
        crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE)
    });
    namespace.get_or_insert_with("slice", || {
        // The slice type object, for isinstance(x, slice) checks.
        crate::typedef::gettypefor(&pyre_object::sliceobject::SLICE_TYPE)
            .unwrap_or(pyre_object::PY_NULL)
    });
    namespace.get_or_insert_with("frozenset", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE)
    });
    namespace.get_or_insert_with("set", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE)
    });
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::PROPERTY_TYPE)
    });
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::CLASSMETHOD_TYPE)
    });
    namespace.get_or_insert_with("Ellipsis", || pyre_object::noneobject::w_ellipsis());
    namespace.get_or_insert_with("__debug__", || w_bool_from(true));
    // memoryview stub: pyre doesn't model real buffer protocol, but
    // re._compiler._bytes_to_codes wants `memoryview(b).cast('I').tolist()`.
    // We register `memoryview` as a real type whose __new__ stores the
    // backing bytearray on the instance; .cast('I') and .tolist() do the
    // little-endian unpack inline.
    namespace.get_or_insert_with("memoryview", || {
        let tp = crate::typedef::make_builtin_type("memoryview", |ns| {
            crate::dict_storage_store(
                ns,
                "__new__",
                make_builtin_function_with_arity(
                    "__new__",
                    |args| {
                        // args[0] = cls (memoryview), args[1] = buffer-like
                        let cls = args.get(0).copied().unwrap_or(w_none());
                        let buf = args.get(1).copied().unwrap_or(w_none());
                        let inst = pyre_object::w_instance_new(cls);
                        crate::baseobjspace::setattr(inst, "__pyre_buf__", buf)?;
                        crate::baseobjspace::setattr(inst, "__pyre_fmt__", w_str_new("B"))?;
                        crate::baseobjspace::setattr(inst, "__pyre_itemsize__", w_int_new(1))?;
                        Ok(inst)
                    },
                    2,
                ),
            );
            crate::dict_storage_store(
                ns,
                "cast",
                make_builtin_function_with_arity(
                    "cast",
                    |args| {
                        let mv = args.get(0).copied().unwrap_or(w_none());
                        let fmt_obj = args.get(1).copied().unwrap_or(w_none());
                        let fmt = if unsafe { pyre_object::is_str(fmt_obj) } {
                            unsafe { pyre_object::w_str_get_value(fmt_obj) }
                        } else {
                            "B"
                        };
                        let itemsize: i64 = match fmt {
                            "I" | "i" | "L" | "l" | "f" => 4,
                            "Q" | "q" | "d" => 8,
                            "H" | "h" => 2,
                            _ => 1,
                        };
                        let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                        let cls = crate::typedef::r#type(mv).unwrap_or(pyre_object::PY_NULL);
                        let inst = pyre_object::w_instance_new(cls);
                        crate::baseobjspace::setattr(inst, "__pyre_buf__", buf)?;
                        crate::baseobjspace::setattr(inst, "__pyre_fmt__", w_str_new(fmt))?;
                        crate::baseobjspace::setattr(
                            inst,
                            "__pyre_itemsize__",
                            w_int_new(itemsize),
                        )?;
                        Ok(inst)
                    },
                    2,
                ),
            );
            crate::dict_storage_store(
                ns,
                "tolist",
                make_builtin_function_with_arity(
                    "tolist",
                    |args| {
                        let mv = args.get(0).copied().unwrap_or(w_none());
                        let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                        let itemsize_obj = crate::baseobjspace::getattr(mv, "__pyre_itemsize__")?;
                        let itemsize =
                            unsafe { pyre_object::w_int_get_value(itemsize_obj) } as usize;
                        let data = if unsafe { pyre_object::bytesobject::is_bytes_like(buf) } {
                            unsafe { pyre_object::bytesobject::bytes_like_data(buf) }
                        } else {
                            return Ok(w_list_new(vec![]));
                        };
                        let mut items = Vec::with_capacity(data.len() / itemsize.max(1));
                        let mut i = 0;
                        while i + itemsize <= data.len() {
                            let mut val: i64 = 0;
                            for j in 0..itemsize {
                                val |= (data[i + j] as i64) << (8 * j);
                            }
                            items.push(w_int_new(val));
                            i += itemsize;
                        }
                        Ok(w_list_new(items))
                    },
                    1,
                ),
            );
            crate::dict_storage_store(
                ns,
                "__len__",
                make_builtin_function_with_arity(
                    "__len__",
                    |args| {
                        let mv = args.get(0).copied().unwrap_or(w_none());
                        let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                        let itemsize_obj = crate::baseobjspace::getattr(mv, "__pyre_itemsize__")?;
                        let itemsize =
                            unsafe { pyre_object::w_int_get_value(itemsize_obj) } as usize;
                        let n = if unsafe { pyre_object::bytesobject::is_bytes_like(buf) } {
                            unsafe { pyre_object::bytesobject::bytes_like_len(buf) }
                        } else {
                            0
                        };
                        Ok(w_int_new((n / itemsize.max(1)) as i64))
                    },
                    1,
                ),
            );
            // memoryview.itemsize attribute — read from the per-instance
            // __pyre_itemsize__ slot via property descriptor.
            crate::dict_storage_store(
                ns,
                "itemsize",
                pyre_object::w_property_new(
                    make_builtin_function_with_arity(
                        "itemsize",
                        |args| {
                            let mv = args.get(0).copied().unwrap_or(w_none());
                            crate::baseobjspace::getattr(mv, "__pyre_itemsize__")
                        },
                        1,
                    ),
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ),
            );
        });
        // Store per-instance __pyre_buf__/__pyre_fmt__/__pyre_itemsize__ slots.
        unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
        tp
    });
    namespace.get_or_insert_with("globals", || {
        make_module_builtin_function_with_arity("globals", builtin_globals, 0)
    });
    namespace.get_or_insert_with("locals", || {
        make_module_builtin_function_with_arity("locals", builtin_locals, 0)
    });
    namespace.get_or_insert_with("exec", || {
        make_module_builtin_function("exec", builtin_exec)
    });
    namespace.get_or_insert_with("eval", || {
        make_module_builtin_function("eval", builtin_eval)
    });
    namespace.get_or_insert_with("compile", || {
        make_module_builtin_function("compile", builtin_compile)
    });
    namespace.get_or_insert_with("complex", || {
        make_module_builtin_function("complex", builtin_complex)
    });
    namespace.get_or_insert_with("filter", || {
        make_module_builtin_function("filter", |args| {
            if args.len() < 2 {
                return Ok(w_list_new(vec![]));
            }
            let func = args[0];
            let items = collect_iterable(args[1])?;
            let mut out = Vec::new();
            let func_is_none = unsafe { pyre_object::is_none(func) };
            for item in items {
                let keep = if func_is_none {
                    crate::baseobjspace::is_true(item)
                } else {
                    let result = crate::call_function(func, &[item]);
                    if result.is_null() {
                        false
                    } else {
                        crate::baseobjspace::is_true(result)
                    }
                };
                if keep {
                    out.push(item);
                }
            }
            Ok(w_list_new(out))
        })
    });
    namespace.get_or_insert_with("input", || {
        make_module_builtin_function("input", |_| Ok(pyre_object::w_str_new("")))
    });
    namespace.get_or_insert_with("open", || {
        make_module_builtin_function("open", builtin_open)
    });
    // Exception hierarchy — exceptions are real types so they can be
    // subclassed (`class FrozenInstanceError(AttributeError): pass`).
    // Built in dependency order: each subclass refers to its already-built
    // parent. PyPy: each typedef.py W_<Exception>.typedef registers a real
    // W_TypeObject in space.builtin.
    let base_exc = make_exc_type(
        "BaseException",
        exc_base_exception_new,
        crate::typedef::w_object(),
    );
    crate::dict_storage_store(namespace, "BaseException", base_exc);

    let exception = make_exc_type("Exception", exc_exception_new, base_exc);
    crate::dict_storage_store(namespace, "Exception", exception);

    let arithmetic = make_exc_type("ArithmeticError", exc_arithmetic_error_new, exception);
    crate::dict_storage_store(namespace, "ArithmeticError", arithmetic);
    crate::dict_storage_store(
        namespace,
        "ZeroDivisionError",
        make_exc_type("ZeroDivisionError", exc_zero_division_new, arithmetic),
    );
    crate::dict_storage_store(
        namespace,
        "OverflowError",
        make_exc_type("OverflowError", exc_overflow_error_new, arithmetic),
    );
    crate::dict_storage_store(
        namespace,
        "FloatingPointError",
        make_exc_type("FloatingPointError", exc_arithmetic_error_new, arithmetic),
    );

    let lookup_error = make_exc_type("LookupError", exc_exception_new, exception);
    crate::dict_storage_store(namespace, "LookupError", lookup_error);
    crate::dict_storage_store(
        namespace,
        "IndexError",
        make_exc_type("IndexError", exc_index_error_new, lookup_error),
    );
    crate::dict_storage_store(
        namespace,
        "KeyError",
        make_exc_type("KeyError", exc_key_error_new, lookup_error),
    );

    crate::dict_storage_store(
        namespace,
        "AttributeError",
        make_exc_type("AttributeError", exc_attribute_error_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "TypeError",
        make_exc_type("TypeError", exc_type_error_new, exception),
    );
    let value_error = make_exc_type("ValueError", exc_value_error_new, exception);
    crate::dict_storage_store(namespace, "ValueError", value_error);
    crate::dict_storage_store(
        namespace,
        "NameError",
        make_exc_type("NameError", exc_name_error_new, exception),
    );

    let runtime_error = make_exc_type("RuntimeError", exc_runtime_error_new, exception);
    crate::dict_storage_store(namespace, "RuntimeError", runtime_error);
    crate::dict_storage_store(
        namespace,
        "NotImplementedError",
        make_exc_type(
            "NotImplementedError",
            exc_not_implemented_error_new,
            runtime_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "RecursionError",
        make_exc_type("RecursionError", exc_runtime_error_new, runtime_error),
    );

    crate::dict_storage_store(
        namespace,
        "StopIteration",
        make_exc_type("StopIteration", exc_stop_iteration_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "StopAsyncIteration",
        make_exc_type("StopAsyncIteration", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "GeneratorExit",
        make_exc_type("GeneratorExit", exc_base_exception_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "SystemExit",
        make_exc_type("SystemExit", exc_base_exception_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "KeyboardInterrupt",
        make_exc_type("KeyboardInterrupt", exc_base_exception_new, base_exc),
    );

    let import_error = make_exc_type("ImportError", exc_import_error_new, exception);
    crate::dict_storage_store(namespace, "ImportError", import_error);
    crate::dict_storage_store(
        namespace,
        "ModuleNotFoundError",
        make_exc_type("ModuleNotFoundError", exc_import_error_new, import_error),
    );
    crate::dict_storage_store(
        namespace,
        "AssertionError",
        make_exc_type("AssertionError", exc_assertion_error_new, exception),
    );

    let os_error = make_exc_type("OSError", exc_exception_new, exception);
    crate::dict_storage_store(namespace, "OSError", os_error);
    crate::dict_storage_store(namespace, "IOError", os_error);
    crate::dict_storage_store(
        namespace,
        "FileNotFoundError",
        make_exc_type("FileNotFoundError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "FileExistsError",
        make_exc_type("FileExistsError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "PermissionError",
        make_exc_type("PermissionError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "NotADirectoryError",
        make_exc_type("NotADirectoryError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "IsADirectoryError",
        make_exc_type("IsADirectoryError", exc_exception_new, os_error),
    );

    let warning = make_exc_type("Warning", exc_exception_new, exception);
    crate::dict_storage_store(namespace, "Warning", warning);
    for warn_name in [
        "UserWarning",
        "DeprecationWarning",
        "PendingDeprecationWarning",
        "RuntimeWarning",
        "FutureWarning",
        "ImportWarning",
        "UnicodeWarning",
        "BytesWarning",
        "ResourceWarning",
        "SyntaxWarning",
        "EncodingWarning",
    ] {
        crate::dict_storage_store(
            namespace,
            warn_name,
            make_exc_type(warn_name, exc_exception_new, warning),
        );
    }

    let unicode_error = make_exc_type("UnicodeError", exc_value_error_new, value_error);
    crate::dict_storage_store(namespace, "UnicodeError", unicode_error);
    crate::dict_storage_store(
        namespace,
        "UnicodeDecodeError",
        make_exc_type("UnicodeDecodeError", exc_value_error_new, unicode_error),
    );
    crate::dict_storage_store(
        namespace,
        "UnicodeEncodeError",
        make_exc_type("UnicodeEncodeError", exc_value_error_new, unicode_error),
    );
    crate::dict_storage_store(
        namespace,
        "UnicodeTranslateError",
        make_exc_type("UnicodeTranslateError", exc_value_error_new, unicode_error),
    );

    crate::dict_storage_store(
        namespace,
        "BufferError",
        make_exc_type("BufferError", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "MemoryError",
        make_exc_type("MemoryError", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "ReferenceError",
        make_exc_type("ReferenceError", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "SystemError",
        make_exc_type("SystemError", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "EOFError",
        make_exc_type("EOFError", exc_exception_new, exception),
    );
    let syntax_error = make_exc_type("SyntaxError", exc_exception_new, exception);
    crate::dict_storage_store(namespace, "SyntaxError", syntax_error);
    let indentation_error = make_exc_type("IndentationError", exc_exception_new, syntax_error);
    crate::dict_storage_store(namespace, "IndentationError", indentation_error);
    crate::dict_storage_store(
        namespace,
        "TabError",
        make_exc_type("TabError", exc_exception_new, indentation_error),
    );
    crate::dict_storage_store(
        namespace,
        "BlockingIOError",
        make_exc_type("BlockingIOError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "ChildProcessError",
        make_exc_type("ChildProcessError", exc_exception_new, os_error),
    );
    let connection_error = make_exc_type("ConnectionError", exc_exception_new, os_error);
    crate::dict_storage_store(namespace, "ConnectionError", connection_error);
    crate::dict_storage_store(
        namespace,
        "BrokenPipeError",
        make_exc_type("BrokenPipeError", exc_exception_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionAbortedError",
        make_exc_type(
            "ConnectionAbortedError",
            exc_exception_new,
            connection_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionRefusedError",
        make_exc_type(
            "ConnectionRefusedError",
            exc_exception_new,
            connection_error,
        ),
    );
    crate::dict_storage_store(
        namespace,
        "ConnectionResetError",
        make_exc_type("ConnectionResetError", exc_exception_new, connection_error),
    );
    crate::dict_storage_store(
        namespace,
        "InterruptedError",
        make_exc_type("InterruptedError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "ProcessLookupError",
        make_exc_type("ProcessLookupError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "TimeoutError",
        make_exc_type("TimeoutError", exc_exception_new, os_error),
    );
    crate::dict_storage_store(
        namespace,
        "BaseExceptionGroup",
        make_exc_type("BaseExceptionGroup", exc_base_exception_new, base_exc),
    );
    crate::dict_storage_store(
        namespace,
        "ExceptionGroup",
        make_exc_type("ExceptionGroup", exc_exception_new, exception),
    );
    crate::dict_storage_store(
        namespace,
        "PythonFinalizationError",
        make_exc_type(
            "PythonFinalizationError",
            exc_runtime_error_new,
            runtime_error,
        ),
    );
    namespace.get_or_insert_with("any", || {
        make_module_builtin_function_with_arity("any", builtin_any, 1)
    });
    namespace.get_or_insert_with("all", || {
        make_module_builtin_function_with_arity("all", builtin_all, 1)
    });
    namespace.get_or_insert_with("sum", || make_module_builtin_function("sum", builtin_sum));
    namespace.get_or_insert_with("round", || {
        make_module_builtin_function("round", builtin_round)
    });
    namespace.get_or_insert_with("divmod", || {
        make_module_builtin_function("divmod", builtin_divmod)
    });
    namespace.get_or_insert_with("pow", || make_module_builtin_function("pow", builtin_pow));
    namespace.get_or_insert_with("hex", || make_module_builtin_function("hex", builtin_hex));
    namespace.get_or_insert_with("oct", || make_module_builtin_function("oct", builtin_oct));
    namespace.get_or_insert_with("bin", || make_module_builtin_function("bin", builtin_bin));
    namespace.get_or_insert_with("format", || {
        make_module_builtin_function("format", builtin_format)
    });
    namespace.get_or_insert_with("issubclass", || {
        make_module_builtin_function_with_arity("issubclass", builtin_issubclass, 2)
    });
    namespace.get_or_insert_with("__import__", || {
        make_module_builtin_function("__import__", builtin_import_stub)
    });

    // Descriptor types
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::PROPERTY_TYPE)
    });
    // staticmethod/classmethod registered as types for isinstance() support.
    // The type's __new__ creates the descriptor wrapper.
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::CLASSMETHOD_TYPE)
    });
}

/// Create a fresh namespace seeded with the default builtins.
pub fn new_builtin_dict_storage() -> DictStorage {
    crate::typedef::init_typeobjects();
    let mut namespace = DictStorage::new();
    install_default_builtins(&mut namespace);
    namespace
}

/// `print(*args)` — write space-separated str representations to stdout.
fn builtin_print(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Check if last arg is a kwargs dict (from CALL_KW builtin dispatch).
    // Distinguished from regular dict args by __pyre_kw__ marker key.
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            is_dict(last) && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__")).is_some()
        };
    let (positional, end, sep) = if is_kwargs {
        let kwargs = *args.last().unwrap();
        let end_key = w_str_new("end");
        let sep_key = w_str_new("sep");
        let end_val = unsafe { pyre_object::w_dict_lookup(kwargs, end_key) };
        let sep_val = unsafe { pyre_object::w_dict_lookup(kwargs, sep_key) };
        let end_str = end_val
            .map(|v| unsafe { crate::py_str(v) })
            .unwrap_or_else(|| "\n".to_string());
        let sep_str = sep_val
            .map(|v| unsafe { crate::py_str(v) })
            .unwrap_or_else(|| " ".to_string());
        (&args[..args.len() - 1], end_str, sep_str)
    } else {
        (args, "\n".to_string(), " ".to_string())
    };

    let parts: Vec<String> = positional
        .iter()
        .map(|&obj| format!("{}", PyDisplay(obj)))
        .collect();
    crate::print_output(&format!("{}{}", parts.join(&sep), end));
    Ok(w_none())
}

/// Extract an i64 from an int or long object. Panics if the long value
/// exceeds i64 range (range() cannot iterate over such large spans).
unsafe fn range_arg_to_i64(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_int(obj) {
            return w_int_get_value(obj);
        }
        if is_long(obj) {
            let val = w_long_get_value(obj);
            return val.to_i64().unwrap_or(i64::MAX);
        }
        return 0; // non-integer argument fallback
    }
}

/// `range(stop)` or `range(start, stop)` or `range(start, stop, step)`.
///
/// Returns a range iterator directly (simplified: no separate range object).
fn builtin_range(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    match args.len() {
        1 => {
            let stop = unsafe { range_arg_to_i64(args[0]) };
            Ok(w_range_iter_new(0, stop, 1))
        }
        2 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            Ok(w_range_iter_new(start, stop, 1))
        }
        3 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            let step = unsafe { range_arg_to_i64(args[2]) };
            Ok(w_range_iter_new(start, stop, step))
        }
        _ => panic!("range() takes 1 to 3 arguments"),
    }
}

/// `len(obj)` — return the length of an object.
/// `len(obj)` — PyPy: operation.py len → space.len_w
fn builtin_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "len() takes exactly one argument");
    crate::baseobjspace::len(args[0])
}

/// `abs(x)` — return the absolute value of a number.
pub fn builtin_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "abs() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_bool(obj) {
            return Ok(w_int_new(w_bool_get_value(obj) as i64));
        }
        if is_int(obj) {
            let v = w_int_get_value(obj);
            // i64::MIN.abs() overflows; promote to long
            return Ok(match v.checked_abs() {
                Some(r) => w_int_new(r),
                None => w_long_new(-BigInt::from(v)),
            });
        }
        if is_long(obj) {
            let val = w_long_get_value(obj).clone();
            return Ok(w_long_new(if val < BigInt::from(0) { -val } else { val }));
        }
        if is_float(obj) {
            return Ok(w_float_new(w_float_get_value(obj).abs()));
        }
    }
    // Instance __abs__ — PyPy: baseobjspace.py abs
    unsafe {
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__abs__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    Err(crate::PyError::type_error("bad operand type for abs()"))
}

/// Strip the trailing `__pyre_kw__` dict that `call_with_kwargs`
/// (`call.rs:727-744`) appends for builtin callees and return the
/// positional slice paired with a keyword lookup helper.
///
/// PyPy's gateway parses `Arguments.keyword_names_w` directly through
/// the `unwrap_spec` machinery; pyre's flat `BuiltinCodeFn` ABI lacks
/// that surface, so each kwarg-aware builtin reaches into the trailing
/// dict via this shared helper.
pub(crate) fn split_builtin_kwargs(args: &[PyObjectRef]) -> (&[PyObjectRef], Option<PyObjectRef>) {
    if let Some(&last) = args.last() {
        if unsafe {
            is_dict(last) && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__")).is_some()
        } {
            return (&args[..args.len() - 1], Some(last));
        }
    }
    (args, None)
}

/// Look up a single keyword argument from the kwargs dict produced by
/// `split_builtin_kwargs`. Returns `None` when no kwargs dict is present
/// or the requested key is absent.
pub(crate) fn kwarg_get(kwargs: Option<PyObjectRef>, name: &str) -> Option<PyObjectRef> {
    let dict = kwargs?;
    unsafe { pyre_object::w_dict_lookup(dict, w_str_new(name)) }
}

/// Reject any keyword argument whose name is not in `allowed`.  Mirrors
/// PyPy's `unwrap_spec` strict-keyword behaviour — for example
/// `pypy/module/__builtin__/functional.py:198-201 min_max` raises
/// `TypeError("min() got unexpected keyword argument")` whenever an
/// unknown kwarg slips in (only `key` and `default` are accepted).
/// pyre's flat builtin ABI has to police this manually because
/// `split_builtin_kwargs` does not enforce a signature.
///
/// `fn_name` is the bare function name used in the error message
/// ("min", "zip_longest", ...).  The `__pyre_kw__` marker entry the
/// gateway appends is filtered out; it is an implementation detail of
/// the kwargs encoding, not a user-visible argument.
pub(crate) fn kwarg_reject_unknown(
    kwargs: Option<PyObjectRef>,
    allowed: &[&str],
    fn_name: &str,
) -> Result<(), crate::PyError> {
    let dict = match kwargs {
        Some(d) => d,
        None => return Ok(()),
    };
    let entries = unsafe { pyre_object::w_dict_str_entries(dict) };
    for (key, _) in entries.iter() {
        if key == "__pyre_kw__" {
            continue;
        }
        if !allowed.iter().any(|name| *name == key.as_str()) {
            return Err(crate::PyError::type_error(format!(
                "{fn_name}() got an unexpected keyword argument '{key}'"
            )));
        }
    }
    Ok(())
}

/// `space.index_w(obj)` parity — `pypy/interpreter/baseobjspace.py
/// space.index_w` returns the int value of an object exposing
/// `__index__`.  Pyre handles the int / long / bool fast paths
/// directly and falls through to looking up `__index__` on the
/// object's type, mirroring PyPy's `lookup_in_type` pass before
/// raising `TypeError`.
pub(crate) fn space_index_w(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        if pyre_object::is_int(obj) {
            return Ok(pyre_object::w_int_get_value(obj));
        }
        if pyre_object::is_bool(obj) {
            return Ok(if pyre_object::w_bool_get_value(obj) {
                1
            } else {
                0
            });
        }
        if let Some(w_type) = crate::typedef::r#type(obj) {
            if let Some(index_fn) = crate::baseobjspace::lookup_in_type(w_type, "__index__") {
                let result = crate::call::call_function_impl_result(index_fn, &[obj])?;
                if pyre_object::is_int(result) {
                    return Ok(pyre_object::w_int_get_value(result));
                }
                if pyre_object::is_bool(result) {
                    return Ok(if pyre_object::w_bool_get_value(result) {
                        1
                    } else {
                        0
                    });
                }
            }
        }
    }
    let tp_name = unsafe {
        match crate::typedef::r#type(obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => "object".to_string(),
        }
    };
    Err(crate::PyError::type_error(format!(
        "'{tp_name}' object cannot be interpreted as an integer"
    )))
}

/// Convert an int or long object to BigInt for comparison.
pub(crate) unsafe fn obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }
}

/// `min(*args)` / `min(iterable)` — return the smallest value.
///
/// `pypy/module/__builtin__/functional.py:188-218 min_max`:
///   - reject any kwargs other than `key` / `default`
///   - reject `default=` paired with multiple positional args
///   - require ≥1 positional arg
fn builtin_min(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    min_max_dispatch(args, /* want_max= */ false, "min")
}

/// `max(a, b)` / `max(iterable)` — return the largest of two values or an iterable.
fn builtin_max(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    min_max_dispatch(args, /* want_max= */ true, "max")
}

fn min_max_dispatch(
    args: &[PyObjectRef],
    want_max: bool,
    fn_name: &str,
) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    // functional.py:198-201 — only `key` and `default` are accepted.
    kwarg_reject_unknown(kwargs, &["key", "default"], fn_name)?;
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let default = kwarg_get(kwargs, "default");
    // functional.py:216-218 — empty positional → TypeError, not panic.
    if positional.is_empty() {
        return Err(crate::PyError::type_error(format!(
            "{fn_name}() expected at least one argument, got 0"
        )));
    }
    // functional.py:206-210 — `default=` is only meaningful for the
    // single-iterable form; combining it with multiple positional args
    // is a user error.
    if positional.len() > 1 && default.is_some() {
        return Err(crate::PyError::type_error(format!(
            "Cannot specify a default for {fn_name}() with multiple positional arguments"
        )));
    }
    let items: Vec<PyObjectRef> = if positional.len() == 1 {
        collect_iterable(positional[0])?
    } else {
        positional.to_vec()
    };
    if items.is_empty() {
        if let Some(d) = default {
            return Ok(d);
        }
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("{fn_name}() iterable argument is empty"),
        ));
    }
    select_extremum(&items, key_fn, want_max)
}

/// Shared min/max body — `pypy/module/__builtin__/functional.py:115-148
/// min_max`.  Builds (key, item) pairs (identity when no `key=`),
/// keeps a running best by comparing keys via `space.gt`/`space.lt`
/// (the PyPy compare paths invoke `__gt__` / `__lt__` and propagate
/// errors), returns the corresponding item.  PyPy's stable-tie rule:
/// keep the first-seen extremum (`<` for min, `>` for max), matching
/// CPython 3.x semantics.
fn select_extremum(
    items: &[PyObjectRef],
    key_fn: Option<PyObjectRef>,
    want_max: bool,
) -> Result<PyObjectRef, crate::PyError> {
    let key_of = |item: PyObjectRef| -> PyObjectRef {
        match key_fn {
            Some(kf) => crate::call_function(kf, &[item]),
            None => item,
        }
    };
    let cmp_op = if want_max {
        crate::baseobjspace::CompareOp::Gt
    } else {
        crate::baseobjspace::CompareOp::Lt
    };
    let mut best_item = items[0];
    let mut best_key = key_of(best_item);
    for &item in &items[1..] {
        let key = key_of(item);
        // `functional.py:139 if space.is_true(space.gt(key, best_key))`
        // — route through the generic comparison dispatch which
        // handles int/long/str/float/tuple natively and falls
        // through to user-defined `__gt__`/`__lt__` for other
        // types.  Errors (TypeError from incomparable types) are
        // propagated to the caller as PyPy does.
        let result = crate::baseobjspace::compare(key, best_key, cmp_op)?;
        if crate::baseobjspace::is_true(result) {
            best_item = item;
            best_key = key;
        }
    }
    Ok(best_item)
}

/// `type(obj)` — return the type name as a string (simplified).
/// `type(obj)` — return the type of an object as a W_TypeObject.
///
/// PyPy: `space.type(w_obj)` → W_TypeObject
pub(crate) fn type_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // type.__new__(metatype, name, bases, dict)
    // May be called with extra self-binding from super():
    //   [self, metatype, name, bases, dict] — 5 args
    //   [metatype, name, bases, dict] — 4 args
    //   [metatype, obj] — 2 args (type(obj))
    // Find the (name, bases, dict) triple by scanning for the first str arg.
    // Also extract the metatype (first type arg before the name str).
    let mut w_metaclass = pyre_object::PY_NULL;
    for i in 0..args.len() {
        if unsafe { pyre_object::is_str(args[i]) } && i + 2 < args.len() {
            // Extract metatype from preceding args
            for j in 0..i {
                if unsafe { pyre_object::is_type(args[j]) } {
                    w_metaclass = args[j];
                }
            }
            return type_descr_new_with_metaclass(&args[i..], w_metaclass);
        }
    }
    if args.len() == 1 && unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    if args.len() == 1 {
        return type_descr_new_without_metaclass(args);
    }
    if args.len() == 2 {
        return type_descr_new_without_metaclass(&args[1..]);
    }
    Err(crate::PyError::type_error("type() takes 1 or 3 arguments"))
}
fn type_descr_new_without_metaclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    type_descr_new_with_metaclass(args, pyre_object::PY_NULL)
}

fn type_descr_new_with_metaclass(
    args: &[PyObjectRef],
    w_metaclass: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 && args.len() != 3 {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    // type(name, bases, dict) — 3-arg form creates a new type
    // PyPy: typeobject.py type.__new__(metatype, name, bases, dict)
    if args.len() == 3 {
        let name_obj = args[0];
        let bases = args[1];
        let w_namespace_dict = args[2];
        let name = unsafe { pyre_object::w_str_get_value(name_obj) };

        // CPython: calculate_metaclass — if bases have a custom metaclass,
        // delegate to that metaclass instead of using type.__new__ directly.
        if w_metaclass.is_null() && !bases.is_null() && unsafe { is_tuple(bases) } {
            let n = unsafe { w_tuple_len(bases) };
            for i in 0..n {
                if let Some(base) = unsafe { pyre_object::w_tuple_getitem(bases, i as i64) } {
                    if unsafe { pyre_object::is_type(base) } {
                        // baseobjspace.py:76 — metaclass from w_class
                        let w_metaclass = unsafe {
                            let w_class = (*base).w_class;
                            let w_type_type = crate::typedef::w_type();
                            if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                                Some(w_class)
                            } else {
                                None
                            }
                        };
                        if let Some(w_metaclass) = w_metaclass {
                            // Delegate: call metaclass(name, bases, dict, **kwds)
                            // Pass extra args from the original call
                            let mut metaclass_args = vec![name_obj, bases, w_namespace_dict];
                            if args.len() > 3 {
                                metaclass_args.extend_from_slice(&args[3..]);
                            }
                            return Ok(crate::call_function(w_metaclass, &metaclass_args));
                        }
                    }
                }
            }
        }

        // Convert dict to DictStorage
        let mut class_ns = Box::new(crate::DictStorage::new());
        class_ns.fix_ptr();
        if unsafe { is_dict(w_namespace_dict) } {
            let d = unsafe { &*(w_namespace_dict as *const pyre_object::dictobject::W_DictObject) };
            for &(k, v) in unsafe { &*d.entries } {
                if unsafe { is_str(k) } {
                    let key = unsafe { pyre_object::w_str_get_value(k) };
                    crate::dict_storage_store(&mut class_ns, key, v);
                }
            }
        }
        let ns_ptr = Box::into_raw(class_ns);

        // Default bases to (object,) if empty
        let w_effective_bases =
            if bases.is_null() || !unsafe { is_tuple(bases) } || unsafe { w_tuple_len(bases) } == 0
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

        // CPython: calculate_metaclass — delegate to winner if different
        let default_meta = if w_metaclass.is_null() {
            crate::typedef::w_type()
        } else {
            w_metaclass
        };
        let w_winner = crate::call::calculate_metaclass(default_meta, w_effective_bases)
            .unwrap_or(default_meta);
        if !std::ptr::eq(w_winner, default_meta) {
            // Winner is a different metaclass — delegate to its __new__
            if let Some(w_metaclass_new) =
                unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") }
            {
                let mut new_args = vec![w_winner, name_obj, bases, w_namespace_dict];
                if args.len() > 3 {
                    new_args.extend_from_slice(&args[3..]);
                }
                drop(unsafe { Box::from_raw(ns_ptr) });
                return Ok(crate::call_function(w_metaclass_new, &new_args));
            }
        }
        let w_metaclass = w_winner;

        let w_type = pyre_object::w_type_new(name, w_effective_bases, ns_ptr as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe {
            let ns = &*ns_ptr;
            crate::call::create_all_slots(w_type, ns, w_effective_bases)?;
        }
        // rclass.py:739-743 — set w_class (typeptr) at allocation time.
        // For type objects, w_class is the metaclass (type(C) → Meta).
        // baseobjspace.py:76 getclass() returns the metatype.
        unsafe {
            (*w_type).w_class = w_metaclass;
        }
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w_type) };
        unsafe { pyre_object::w_type_set_mro(w_type, mro) };

        // __set_name__ protocol — CPython: type_new_set_names
        // PyPy: typeobject.py type_new → call __set_name__(owner, name) on each descriptor
        if unsafe { is_dict(w_namespace_dict) } {
            let d = unsafe { &*(w_namespace_dict as *const pyre_object::dictobject::W_DictObject) };
            let entries: Vec<(PyObjectRef, PyObjectRef)> = unsafe { (*d.entries).clone() };
            for (k, v) in entries {
                if unsafe { is_str(k) } {
                    if let Ok(set_name) = crate::baseobjspace::getattr(v, "__set_name__") {
                        // getattr returns a bound method, so self is already bound.
                        // Call: bound_set_name(owner, name)
                        let _ = crate::call_function(set_name, &[w_type, k]);
                    }
                }
            }
        }

        return Ok(w_type);
    }

    // type(obj) — 1-arg form returns the type
    // PyPy objspace.py:400: space.type(w_obj) → w_obj.getclass(space)
    // typedef::type() respects __class__ override for all object kinds.
    let obj = args[0];
    if let Some(tp) = crate::typedef::r#type(obj) {
        return Ok(tp);
    }
    if obj.is_null() {
        return Ok(crate::typedef::gettypeobject(
            &pyre_object::pyobject::NONE_TYPE,
        ));
    }
    let name = unsafe { (*(*obj).ob_type).name };
    Ok(box_str_constant(name))
}

/// `isinstance(obj, cls)` — pypy/module/__builtin__/abstractinst.py
/// `app_isinstance` → `abstract_isinstance_w(allow_override=True)`.
fn builtin_isinstance(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "isinstance() takes exactly two arguments");
    Ok(w_bool_from(crate::baseobjspace::isinstance(
        args[0], args[1],
    )?))
}

/// isinstance(obj, cls) for JIT fast path.
///
/// Returns Some(bool) if the check can be resolved, None if cls format
/// is not supported for the fast path (e.g. tuple of types).
/// Uses the same MRO-based `issubtype_w` as the full dispatch.
pub fn call_isinstance(obj: PyObjectRef, cls: PyObjectRef) -> Option<bool> {
    unsafe {
        if is_type(cls) {
            return Some(crate::baseobjspace::isinstance_w(obj, cls));
        }
    }
    None
}

/// `issubclass(cls, classinfo)` — pypy/module/__builtin__/abstractinst.py
/// `app_issubclass` → `abstract_issubclass_w(allow_override=True)`.
fn builtin_issubclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "issubclass() takes exactly two arguments");
    Ok(w_bool_from(crate::baseobjspace::issubclass(
        args[0], args[1],
    )?))
}

// Descroperation helpers (lookup_type_special, should_try_reverse_first,
// try_dispatch_binary_special, try_dispatch_ternary_special,
// try_int_long_pow_with_modulo, binary_builtin_type_error,
// box_bigint_result, issubtype_w) live in `crate::baseobjspace` because
// they are space-level semantics shared between the builtin module,
// weakproxy wrappers, and any future opcode dispatch.

/// Exception type constructor — called as e.g. `ValueError("msg")`.
///
/// `pypy/module/exceptions/interp_exceptions.py:121-124
/// W_BaseException.descr_init` stores the constructor positional
/// arguments on `self.args_w` (an RPython list), then
/// `descr_str/descr_repr` (line 126-147) format from the same field.
/// Pyre wraps the args into a `W_ListObject` and stamps it into the
/// typed slot via `w_exception_set_args`, matching PyPy's
/// `self.args_w = args_w` shape; `w_exception_get_args` rebuilds a
/// fresh tuple per read so `e.args` mirrors
/// `space.newtuple(self.args_w)` semantics.  The message string keeps
/// driving `w_exception_get_message` for the lower-level error path.
macro_rules! exc_constructor {
    ($fn_name:ident, $kind:expr) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let msg: String = if args.is_empty() {
                String::new()
            } else if args.len() == 1 {
                unsafe { crate::display::py_str(args[0]) }
            } else {
                let parts: Vec<String> = args
                    .iter()
                    .map(|&a| unsafe { crate::display::py_repr(a) })
                    .collect();
                format!("({})", parts.join(", "))
            };
            let exc = pyre_object::excobject::w_exception_new($kind, &msg);
            let args_list = pyre_object::w_list_new(args.to_vec());
            unsafe {
                pyre_object::excobject::w_exception_set_args(exc, args_list);
            }
            Ok(exc)
        }
    };
}

exc_constructor!(
    exc_base_exception,
    pyre_object::excobject::ExcKind::BaseException
);
exc_constructor!(exc_exception, pyre_object::excobject::ExcKind::Exception);
exc_constructor!(
    exc_arithmetic_error,
    pyre_object::excobject::ExcKind::ArithmeticError
);
exc_constructor!(
    exc_zero_division,
    pyre_object::excobject::ExcKind::ZeroDivisionError
);
exc_constructor!(exc_type_error, pyre_object::excobject::ExcKind::TypeError);
exc_constructor!(exc_value_error, pyre_object::excobject::ExcKind::ValueError);
exc_constructor!(exc_key_error, pyre_object::excobject::ExcKind::KeyError);
exc_constructor!(exc_index_error, pyre_object::excobject::ExcKind::IndexError);
exc_constructor!(
    exc_attribute_error,
    pyre_object::excobject::ExcKind::AttributeError
);
exc_constructor!(exc_name_error, pyre_object::excobject::ExcKind::NameError);
exc_constructor!(
    exc_runtime_error,
    pyre_object::excobject::ExcKind::RuntimeError
);
exc_constructor!(
    exc_stop_iteration,
    pyre_object::excobject::ExcKind::StopIteration
);
exc_constructor!(
    exc_overflow_error,
    pyre_object::excobject::ExcKind::OverflowError
);
exc_constructor!(
    exc_import_error,
    pyre_object::excobject::ExcKind::ImportError
);
exc_constructor!(
    exc_not_implemented_error,
    pyre_object::excobject::ExcKind::NotImplementedError
);
exc_constructor!(
    exc_assertion_error,
    pyre_object::excobject::ExcKind::AssertionError
);

/// `cls.__new__` wrapper that strips `cls` and calls an exception constructor.
/// PyPy: each exception type's descr__new__ creates a W_<Kind>Object.
macro_rules! exc_new_wrapper {
    ($wrapper:ident, $ctor:ident) => {
        fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let cls = args.first().copied();
            let rest: &[PyObjectRef] = if args.is_empty() { args } else { &args[1..] };
            let exc = $ctor(rest)?;
            // Set the exception's w_class to the actual exception type (e.g. AssertionError)
            // so that `type(e) is AssertionError` holds and `except ExcType` via isinstance works.
            if let Some(cls) = cls {
                unsafe {
                    (*(exc as *mut pyre_object::PyObject)).w_class = cls;
                }
            }
            Ok(exc)
        }
    };
}

exc_new_wrapper!(exc_base_exception_new, exc_base_exception);
exc_new_wrapper!(exc_exception_new, exc_exception);
exc_new_wrapper!(exc_arithmetic_error_new, exc_arithmetic_error);
exc_new_wrapper!(exc_zero_division_new, exc_zero_division);
exc_new_wrapper!(exc_type_error_new, exc_type_error);
exc_new_wrapper!(exc_value_error_new, exc_value_error);
exc_new_wrapper!(exc_key_error_new, exc_key_error);
exc_new_wrapper!(exc_index_error_new, exc_index_error);
exc_new_wrapper!(exc_attribute_error_new, exc_attribute_error);
exc_new_wrapper!(exc_name_error_new, exc_name_error);
exc_new_wrapper!(exc_runtime_error_new, exc_runtime_error);
exc_new_wrapper!(exc_stop_iteration_new, exc_stop_iteration);
exc_new_wrapper!(exc_overflow_error_new, exc_overflow_error);
exc_new_wrapper!(exc_import_error_new, exc_import_error);
exc_new_wrapper!(exc_not_implemented_error_new, exc_not_implemented_error);
exc_new_wrapper!(exc_assertion_error_new, exc_assertion_error);

/// Build a builtin exception type with the given name, base, and __new__ wrapper.
fn make_exc_type(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    base: PyObjectRef,
) -> PyObjectRef {
    let cls = crate::typedef::make_builtin_type_with_base(
        name,
        move |ns| {
            crate::dict_storage_store(ns, "__new__", make_builtin_function("__new__", new_fn));
            // `pypy/module/exceptions/interp_exceptions.py:225-235`
            // `BaseException.with_traceback` — installed on every
            // builtin exception class so MRO lookup from a subclass
            // (`MyError.with_traceback`) hits the canonical method
            // even before user-level `class MyError(BaseException):`
            // metaclass walks BaseException's namespace.  PyPy adds
            // this to BaseException only; pyre's `make_exc_type`
            // wires it into every class because Pyre doesn't run
            // `BaseException.__init_subclass__` at builtin-bootstrap
            // time, so without per-class install `subclass.with_traceback`
            // raises AttributeError.
            if name == "BaseException" {
                crate::dict_storage_store(
                    ns,
                    "with_traceback",
                    make_builtin_function_with_arity(
                        "with_traceback",
                        |args| {
                            let w_self = *args.first().ok_or_else(|| {
                                crate::PyError::type_error(
                                    "with_traceback() missing 1 required positional argument: 'self'",
                                )
                            })?;
                            let w_tb = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
                            if !w_self.is_null() && unsafe { pyre_object::is_exception(w_self) } {
                                // `interp_exceptions.py:213-219
                                // descr_settraceback` — only None or
                                // PyTraceback is accepted.
                                let value =
                                    if w_tb.is_null() || unsafe { pyre_object::is_none(w_tb) } {
                                        pyre_object::PY_NULL
                                    } else if unsafe { crate::pytraceback::is_pytraceback(w_tb) } {
                                        w_tb
                                    } else {
                                        return Err(crate::PyError::type_error(
                                            "__traceback__ must be a traceback or None",
                                        ));
                                    };
                                unsafe {
                                    pyre_object::excobject::w_exception_set_traceback(
                                        w_self, value,
                                    );
                                }
                            }
                            Ok(w_self)
                        },
                        2,
                    ),
                );
                // `interp_exceptions.py:236-247 BaseException.add_note`
                // (Python 3.11+ PEP 678).  Appends a string to
                // `self.__notes__`, allocating the list on first call.
                // The list lives in ATTR_TABLE rather than a typed
                // W_ExceptionObject slot — notes are a rare attribute,
                // and the per-instance side store already handles
                // `e.__notes__` reads via baseobjspace::getattr.
                crate::dict_storage_store(
                    ns,
                    "add_note",
                    make_builtin_function_with_arity(
                        "add_note",
                        |args| {
                            let w_self = *args.first().ok_or_else(|| {
                                crate::PyError::type_error(
                                    "add_note() missing 1 required positional argument: 'self'",
                                )
                            })?;
                            let w_note = *args.get(1).ok_or_else(|| {
                                crate::PyError::type_error(
                                    "add_note() missing 1 required positional argument: 'note'",
                                )
                            })?;
                            // `interp_exceptions.py:238-239` — only
                            // `str` is accepted.
                            if w_note.is_null() || !unsafe { pyre_object::is_str(w_note) } {
                                return Err(crate::PyError::type_error("note must be a string"));
                            }
                            // `interp_exceptions.py:240-254` — lazy
                            // list allocation on first call; if the
                            // attribute is already set but NOT a list,
                            // PyPy raises TypeError("Cannot add note:
                            // __notes__ is not a list") per `:254`.
                            let existing = crate::baseobjspace::getattr(w_self, "__notes__")
                                .ok()
                                .filter(|w| !w.is_null());
                            let notes = match existing {
                                Some(v) if unsafe { pyre_object::is_list(v) } => v,
                                Some(_) => {
                                    return Err(crate::PyError::type_error(
                                        "Cannot add note: __notes__ is not a list",
                                    ));
                                }
                                None => {
                                    let fresh = pyre_object::w_list_new(Vec::new());
                                    let _ =
                                        crate::baseobjspace::setattr(w_self, "__notes__", fresh);
                                    fresh
                                }
                            };
                            unsafe { pyre_object::w_list_append(notes, w_note) };
                            Ok(pyre_object::w_none())
                        },
                        2,
                    ),
                );
            }
        },
        base,
    );
    // Record the class so typedef::r#type can map a raised exception
    // back to its specific builtin class (TypeError, ValueError, ...).
    register_exc_class(name, cls);
    cls
}

/// Thread-local registry from exception class name (as used by
/// `ExcKind → exc_kind_name`) to the W_TypeObject exposed in the builtins
/// namespace. Populated at init-builtins time via `make_exc_type`.
fn register_exc_class(name: &'static str, cls: PyObjectRef) {
    EXC_CLASS_REGISTRY.with(|r| {
        r.borrow_mut().insert(name, cls);
    });
}

/// Look up a builtin exception class by its `ExcKind` name. Returns
/// `None` if the registry hasn't been populated yet (e.g. before
/// install_default_builtins).
pub fn lookup_exc_class(name: &str) -> Option<PyObjectRef> {
    EXC_CLASS_REGISTRY.with(|r| r.borrow().get(name).copied())
}

thread_local! {
    static EXC_CLASS_REGISTRY: std::cell::RefCell<std::collections::HashMap<&'static str, PyObjectRef>>
        = std::cell::RefCell::new(std::collections::HashMap::new());
}

/// `__build_class__(body, name, *bases)` — class creation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS
/// Direct call to call::real_build_class (no callback needed —
/// interpreter and runtime are in the same crate).
fn builtin_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::call::real_build_class(args)
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    make_builtin_function("__build_class__", builtin_build_class)
}

/// `str(obj)` → convert to string
pub(crate) fn builtin_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_str_new(""));
    }
    let obj = args[0];
    unsafe {
        if is_str(obj) {
            return Ok(obj);
        }
    }
    let s = unsafe { crate::py_str(obj) };
    Ok(w_str_new(&s))
}

/// `repr(obj)` → string representation
fn builtin_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "repr() takes exactly one argument");
    let s = unsafe { crate::py_repr(args[0]) };
    Ok(w_str_new(&s))
}

/// `int(obj)` → convert to int
/// call_function with exception propagation.
/// PyPy's space.get_and_call_function returns normally or raises;
/// pyre's call_function stashes errors as PY_NULL. This helper
/// recovers stashed errors as Result.
pub(crate) fn call_and_check(
    method: PyObjectRef,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let result = crate::call_function(method, args);
    if result == pyre_object::PY_NULL {
        if let Some(err) = crate::call::take_call_error() {
            return Err(err);
        }
        return Err(crate::PyError::type_error("call returned NULL"));
    }
    Ok(result)
}

/// intobject.py:989-1050 _new_baseint
pub(crate) fn builtin_int(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_int_new(0));
    }
    let obj = args[0];
    let w_base = args.get(1).copied();

    if w_base.is_none() {
        // intobject.py:991: space.is_w(space.type(w_value), space.w_int)
        let w_type = crate::typedef::r#type(obj);
        let w_int = crate::typedef::gettypefor(&INT_TYPE);
        if w_type.is_some() && w_type == w_int {
            return Ok(obj);
        }
        // intobject.py:994: space.lookup(w_value, '__int__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__int__") } {
            // intobject.py:995: w_intvalue = space.int(w_value)
            let w_intvalue = call_and_check(method, &[obj])?;
            return ensure_baseint_result(w_intvalue, obj);
        }
        // intobject.py:997: space.lookup(w_value, '__trunc__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__trunc__") } {
            // intobject.py:998-999: DeprecationWarning
            crate::warn::warn_deprecation("The delegation of int() to __trunc__ is deprecated.");
            // intobject.py:1001: w_obj = space.trunc(w_value)
            let w_obj = call_and_check(method, &[obj])?;
            // intobject.py:1002: if not space.isinstance_w(w_obj, space.w_int)
            if !unsafe { pyre_object::pyobject::is_int_or_long(w_obj) } {
                // intobject.py:1003-1004: try: w_obj = space.index(w_obj)
                if let Some(idx_method) = unsafe { crate::baseobjspace::lookup(w_obj, "__index__") }
                {
                    let w_indexed = call_and_check(idx_method, &[w_obj])?;
                    return ensure_baseint_result(w_indexed, obj);
                }
                // intobject.py:1008-1011
                return Err(crate::PyError::type_error(
                    "__trunc__ returned non-Integral (type '%T')",
                ));
            }
            return ensure_baseint_result(w_obj, obj);
        }
        // intobject.py:1015: space.lookup(w_value, '__index__')
        if let Some(method) = unsafe { crate::baseobjspace::lookup(obj, "__index__") } {
            // intobject.py:1016: w_obj = space.index(w_value)
            let w_obj = call_and_check(method, &[obj])?;
            // intobject.py:1017: if not space.is_w(space.type(w_obj), space.w_int)
            let w_obj_type = crate::typedef::r#type(w_obj);
            if w_obj_type != w_int {
                // intobject.py:1018: if space.isinstance_w(w_obj, space.w_int)
                if unsafe { pyre_object::pyobject::is_int_or_long(w_obj) } {
                    // intobject.py:1019: w_obj = space.int(w_obj)
                    return ensure_baseint_result(w_obj, obj);
                }
                // intobject.py:1020-1023
                return Err(crate::PyError::type_error(
                    "int() argument must be a string, a bytes-like object or a number, not '%T'",
                ));
            }
            return ensure_baseint_result(w_obj, obj);
        }
        // intobject.py:1026-1034: str
        unsafe {
            if is_str(obj) {
                return parse_int_from_str(w_str_get_value(obj), 10);
            }
            // intobject.py:1035-1038: bytes / bytearray
            if pyre_object::bytesobject::is_bytes_like(obj) {
                let data = pyre_object::bytesobject::bytes_like_data(obj);
                let s = String::from_utf8_lossy(data);
                return parse_int_from_str(&s, 10);
            }
        }
        // intobject.py:1040-1050: buffer interface fallback → TypeError
        return Err(crate::PyError::type_error(
            "int() argument must be a string, a bytes-like object or a number, not '%T'",
        ));
    }

    // intobject.py:1051-1072: w_base is not None — parse with base
    let base = getindex_w_for_base(w_base.unwrap())?;
    unsafe {
        if is_str(obj) {
            return parse_int_from_str(w_str_get_value(obj), base);
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            let data = pyre_object::bytesobject::bytes_like_data(obj);
            let s = String::from_utf8_lossy(data);
            return parse_int_from_str(&s, base);
        }
    }
    Err(crate::PyError::type_error(
        "int() can't convert non-string with explicit base",
    ))
}

/// intobject.py:1093-1107 _ensure_baseint
fn ensure_baseint_result(
    obj: PyObjectRef,
    _original: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    unsafe {
        if is_int(obj) {
            // intobject.py:1096-1098: W_IntObject (or subclass) → wrapint
            return Ok(w_int_new(w_int_get_value(obj)));
        }
        if pyre_object::pyobject::is_long(obj) {
            // intobject.py:1100-1102: W_AbstractLongObject → newlong
            return Ok(pyre_object::longobject::w_long_new(
                pyre_object::longobject::w_long_get_value(obj).clone(),
            ));
        }
    }
    // intobject.py:1104-1107: shouldn't happen
    Err(crate::PyError::new(
        crate::PyErrorKind::RuntimeError,
        "internal error in int.__new__()".to_string(),
    ))
}

/// baseobjspace.py:1564-1596 space.getindex_w(w_base, None)
///
/// Calls __index__() on w_base and converts to i64.
/// On OverflowError (long that doesn't fit i64), returns 37 sentinel
/// (intobject.py:1057: causes ValueError in string_to_bigint).
fn getindex_w_for_base(w_base: PyObjectRef) -> Result<u32, crate::PyError> {
    let value = getindex_w(w_base)?;
    if value < 0 || value == 1 || value > 36 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            format!("int() base must be >= 2 and <= 36, or 0"),
        ));
    }
    Ok(value as u32)
}

/// baseobjspace.py:1564-1596 space.getindex_w(w_obj, None)
///
/// Return w_obj.__index__() as i64. On overflow, clamp to i64::MAX
/// (w_exception=None path).
pub(crate) fn getindex_w(w_obj: PyObjectRef) -> Result<i64, crate::PyError> {
    unsafe {
        if is_int(w_obj) {
            return Ok(w_int_get_value(w_obj));
        }
        if pyre_object::pyobject::is_long(w_obj) {
            // baseobjspace.py:1586-1591: try int_w, on overflow clamp
            use num_traits::ToPrimitive;
            let big = pyre_object::longobject::w_long_get_value(w_obj);
            return Ok(big.to_i64().unwrap_or(i64::MAX));
        }
        // baseobjspace.py:1568: w_index = self.index(w_obj)
        if let Some(method) = crate::baseobjspace::lookup(w_obj, "__index__") {
            let w_index = call_and_check(method, &[w_obj])?;
            if is_int(w_index) {
                return Ok(w_int_get_value(w_index));
            }
            if pyre_object::pyobject::is_long(w_index) {
                use num_traits::ToPrimitive;
                let big = pyre_object::longobject::w_long_get_value(w_index);
                return Ok(big.to_i64().unwrap_or(i64::MAX));
            }
        }
    }
    Err(crate::PyError::type_error(
        "int() second argument must be an integer, not '%T'",
    ))
}

/// Parse an integer from a string with the given base.
fn parse_int_from_str(s: &str, base: u32) -> Result<PyObjectRef, crate::PyError> {
    let s = s.trim();
    let (sign, rest) = if let Some(r) = s.strip_prefix('-') {
        (-1i64, r)
    } else if let Some(r) = s.strip_prefix('+') {
        (1i64, r)
    } else {
        (1i64, s)
    };
    let (radix, digits) = if base == 0 {
        if let Some(r) = rest.strip_prefix("0x").or(rest.strip_prefix("0X")) {
            (16u32, r)
        } else if let Some(r) = rest.strip_prefix("0b").or(rest.strip_prefix("0B")) {
            (2u32, r)
        } else if let Some(r) = rest.strip_prefix("0o").or(rest.strip_prefix("0O")) {
            (8u32, r)
        } else {
            (10u32, rest)
        }
    } else {
        let stripped = match base {
            16 => rest
                .strip_prefix("0x")
                .or(rest.strip_prefix("0X"))
                .unwrap_or(rest),
            2 => rest
                .strip_prefix("0b")
                .or(rest.strip_prefix("0B"))
                .unwrap_or(rest),
            8 => rest
                .strip_prefix("0o")
                .or(rest.strip_prefix("0O"))
                .unwrap_or(rest),
            _ => rest,
        };
        (base, stripped)
    };
    let cleaned: String = digits.chars().filter(|&c| c != '_').collect();
    if let Ok(v) = i64::from_str_radix(&cleaned, radix) {
        return Ok(w_int_new(sign * v));
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::ValueError,
        format!("invalid literal for int() with base {base}: '{s}'"),
    ))
}

/// `float(obj)` → convert to float
pub(crate) fn builtin_float(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(floatobject::w_float_new(0.0));
    }
    // Skip `cls` if called via `float.__new__(cls, value)`.
    let value_idx = if args.len() >= 2 && unsafe { pyre_object::is_type(args[0]) } {
        1
    } else {
        0
    };
    if value_idx >= args.len() {
        return Ok(floatobject::w_float_new(0.0));
    }
    let obj = args[value_idx];
    unsafe {
        if is_float(obj) {
            return Ok(obj);
        }
        if is_int(obj) {
            return Ok(floatobject::w_float_new(w_int_get_value(obj) as f64));
        }
        if is_bool(obj) {
            return Ok(floatobject::w_float_new(if w_bool_get_value(obj) {
                1.0
            } else {
                0.0
            }));
        }
        if pyre_object::is_long(obj) {
            use num_traits::ToPrimitive;
            return Ok(floatobject::w_float_new(
                pyre_object::w_long_get_value(obj)
                    .to_f64()
                    .unwrap_or(f64::NAN),
            ));
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<f64>() {
                return Ok(floatobject::w_float_new(v));
            }
            return Err(crate::PyError::value_error(format!(
                "could not convert string to float: {s:?}"
            )));
        }
    }
    // Fall back to __float__ dunder — PyPy: descroperation.py float.
    if let Ok(method) = crate::baseobjspace::getattr(obj, "__float__") {
        let result = crate::call_function(method, &[obj]);
        if !result.is_null() {
            unsafe {
                if is_float(result) {
                    return Ok(result);
                }
                if is_int(result) {
                    return Ok(floatobject::w_float_new(w_int_get_value(result) as f64));
                }
            }
        }
    }
    // __index__ dunder — integer-like objects.
    if let Ok(method) = crate::baseobjspace::getattr(obj, "__index__") {
        let result = crate::call_function(method, &[obj]);
        if !result.is_null() {
            unsafe {
                if is_int(result) {
                    return Ok(floatobject::w_float_new(w_int_get_value(result) as f64));
                }
            }
        }
    }
    Err(crate::PyError::type_error(
        "float() argument must be a string or a real number",
    ))
}

/// `hasattr(obj, name)` → bool — direct call (no callback needed after merge)
fn builtin_hasattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "hasattr() takes exactly two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    Ok(w_bool_from(crate::baseobjspace::getattr(obj, name).is_ok()))
}

/// `getattr(obj, name[, default])` → value — direct call
fn builtin_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "getattr() takes at least two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    match crate::baseobjspace::getattr(obj, name) {
        Ok(val) => Ok(val),
        Err(e) => {
            if args.len() > 2 {
                Ok(args[2]) // default value
            } else {
                Err(e) // propagate AttributeError
            }
        }
    }
}

/// `setattr(obj, name, value)` — direct call
fn builtin_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 3, "setattr() takes exactly three arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    let _ = crate::baseobjspace::setattr(obj, name, args[2]);
    Ok(w_none())
}

/// `delattr(obj, name)` — PyPy: baseobjspace.py delattr
fn builtin_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "delattr() takes exactly 2 arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    crate::baseobjspace::delattr(obj, name)?;
    Ok(w_none())
}

pub(crate) fn builtin_tuple(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_tuple_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_tuple(obj) {
            return Ok(obj);
        }
        if is_list(obj) {
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_tuple_new(items));
        }
    }
    Ok(w_tuple_new(collect_iterable(obj)?))
}

pub(crate) fn builtin_list_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_list(obj) {
            // Copy the list
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_tuple_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
    }
    // Consume iterator — PyPy: listobject.py W_ListObject(iterable)
    Ok(w_list_new(collect_iterable(obj)?))
}

pub fn collect_iterable(obj: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let it = crate::baseobjspace::iter(obj)?;
    let mut items = Vec::new();
    loop {
        match crate::baseobjspace::next(it) {
            Ok(v) => items.push(v),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

/// Create a `set` from a slice of elements.
///
/// PyPy: `setobject.py` W_SetObject.descr_init → `_initialize_set`.
pub fn builtin_set_from_items(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_set_from_items(items))
}

/// `dict()` — PyPy: dictobject.py W_DictMultiObject.descr_init
pub(crate) fn builtin_dict_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_dict_new());
    }
    let src = args[0];
    unsafe {
        if is_dict(src) {
            // PyPy: descr_init → shallow copy when first arg is a dict
            let dict = w_dict_new();
            for (k, v) in pyre_object::w_dict_items(src) {
                w_dict_store(dict, k, v);
            }
            return Ok(dict);
        }
    }
    // PyPy: dictobject.py update1 → update1_dict_dict (mapping) or update1_pairs (seq)
    //
    // Mapping protocol: if arg has `keys()`, iterate keys and use __getitem__.
    // This handles dict subclasses (e.g. enum.EnumDict) where is_dict() is false.
    if let Ok(keys_method) = crate::baseobjspace::getattr(src, "keys") {
        let dict = w_dict_new();
        let keys_obj = crate::call_function(keys_method, &[]);
        let keys = collect_iterable(keys_obj)?;
        for key in keys {
            let val = crate::baseobjspace::getitem(src, key)?;
            unsafe { w_dict_store(dict, key, val) };
        }
        return Ok(dict);
    }
    // Construct from iterable of (key, value) pairs.
    let dict = w_dict_new();
    let items = collect_iterable(src)?;
    for pair in items {
        let (k, v) = unsafe {
            if is_tuple(pair) && w_tuple_len(pair) == 2 {
                (
                    w_tuple_getitem(pair, 0).unwrap(),
                    w_tuple_getitem(pair, 1).unwrap(),
                )
            } else if is_list(pair) && w_list_len(pair) == 2 {
                (
                    w_list_getitem(pair, 0).unwrap(),
                    w_list_getitem(pair, 1).unwrap(),
                )
            } else {
                return Err(crate::PyError::type_error(
                    "dict update sequence element is not a 2-element sequence",
                ));
            }
        };
        unsafe { w_dict_store(dict, k, v) };
    }
    Ok(dict)
}

/// `super()` — PyPy: descriptor.py W_Super
/// `super(cls, obj)` — PyPy: superobject.py W_Super
///
/// Returns a proxy that looks up methods in cls's MRO starting after cls.
/// `getattr` handles the super proxy via `is_super` check.
///
/// Zero-arg super() finds __class__ and self from the calling frame.
/// CPython: Objects/typeobject.c super_init
fn builtin_super(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        let cls = args[0];
        let obj = args[1];
        return Ok(pyre_object::superobject::w_super_new(cls, obj));
    }
    // Zero-arg super(): find __class__ cell and first arg from calling frame
    //
    // IMPORTANT: CURRENT_FRAME points to the frame that is currently
    // executing the `super()` CALL.  For zero-arg super the __class__
    // cell lives in the *caller* of super(), which IS the current frame
    // (super is a builtin, not a user function that gets its own frame).
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame_ptr = current.get();
        if frame_ptr.is_null() {
            return Err(crate::PyError::runtime_error("super(): no current frame"));
        }
        let frame = unsafe { &*frame_ptr };
        let code = frame.code();

        // Find __class__ in freevars (it's a cell variable from the enclosing class scope)
        let num_locals = code.varnames.len();
        let ncellvars = code.cellvars.len();
        let locals = frame.locals_w().as_slice();

        let mut w_class = pyre_object::PY_NULL;

        // Check freevars for __class__
        for (slot, name) in code.freevars.iter().enumerate() {
            if name == "__class__" {
                let idx = num_locals + ncellvars + slot;
                if idx < locals.len() {
                    let cell = locals[idx];
                    if !cell.is_null() {
                        if unsafe { pyre_object::is_cell(cell) } {
                            w_class = unsafe { pyre_object::w_cell_get(cell) };
                        } else {
                            w_class = cell;
                        }
                    }
                }
                break;
            }
        }

        // Also check cellvars for __class__
        if w_class.is_null() {
            for (slot, name) in code.cellvars.iter().enumerate() {
                if name == "__class__" {
                    let idx = if code.varnames.iter().any(|v| v == name) {
                        code.varnames.iter().position(|v| v == name).unwrap()
                    } else {
                        num_locals + slot
                    };
                    if idx < locals.len() {
                        let cell = locals[idx];
                        if !cell.is_null() {
                            if unsafe { pyre_object::is_cell(cell) } {
                                w_class = unsafe { pyre_object::w_cell_get(cell) };
                            } else {
                                w_class = cell;
                            }
                        }
                    }
                    break;
                }
            }
        }

        if w_class.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): __class__ cell not found",
            ));
        }

        // First argument is self/cls/mcs (locals[0])
        let w_self = if locals.is_empty() {
            pyre_object::PY_NULL
        } else {
            locals[0]
        };

        if w_self.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): no first argument found",
            ));
        }

        Ok(pyre_object::superobject::w_super_new(w_class, w_self))
    })
}

/// `iter(obj)` — PyPy: baseobjspace.py iter
fn builtin_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "iter() requires at least one argument",
        ));
    }
    crate::baseobjspace::iter(args[0])
}

/// `next(iterator[, default])` — PyPy: baseobjspace.py next
fn builtin_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "next() requires at least one argument",
        ));
    }
    match crate::baseobjspace::next(args[0]) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::StopIteration && args.len() > 1 => {
            Ok(args[1]) // default value
        }
        Err(e) => Err(e),
    }
}

/// `callable(obj)` — PyPy: baseobjspace.py callable
fn builtin_callable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    let is_callable = unsafe {
        crate::is_function(obj)
            || pyre_object::is_type(obj)
            || (pyre_object::is_instance(obj)
                && crate::baseobjspace::lookup_in_type(
                    pyre_object::w_instance_get_type(obj),
                    "__call__",
                )
                .is_some())
    };
    Ok(w_bool_from(is_callable))
}

/// `compile(source, filename, mode, ...)` — PyPy: pyopcode.py builtin_compile
///
/// Compiles a Python string to a code object. Only `source`, `filename` and
/// `mode` are honoured; flags / dont_inherit / optimize are accepted but
/// ignored, matching the minimal stub PyPy uses for shim modules.
fn builtin_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "compile() requires source, filename, mode",
        ));
    }
    let source = args[0];
    let filename_obj = args[1];
    let mode_obj = args[2];
    let source_str = unsafe {
        if pyre_object::is_str(source) {
            pyre_object::w_str_get_value(source).to_string()
        } else if pyre_object::bytesobject::is_bytes_like(source) {
            String::from_utf8_lossy(pyre_object::bytesobject::bytes_like_data(source)).into_owned()
        } else {
            return Err(crate::PyError::type_error(
                "compile() arg 1 must be a string or bytes",
            ));
        }
    };
    let filename = unsafe {
        if pyre_object::is_str(filename_obj) {
            pyre_object::w_str_get_value(filename_obj).to_string()
        } else {
            "<string>".to_string()
        }
    };
    let mode = unsafe {
        if pyre_object::is_str(mode_obj) {
            pyre_object::w_str_get_value(mode_obj).to_string()
        } else {
            "exec".to_string()
        }
    };
    let mode = match mode.as_str() {
        "exec" => crate::compile::Mode::Exec,
        "eval" => crate::compile::Mode::Eval,
        "single" => crate::compile::Mode::Single,
        other => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!("compile() mode must be 'exec', 'eval' or 'single', not {other:?}"),
            ));
        }
    };
    let code = crate::compile::compile_source_with_filename(&source_str, mode, &filename)
        .map_err(|e| crate::PyError::new(crate::PyErrorKind::ValueError, e))?;
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    Ok(crate::w_code_new(code_ptr))
}

/// `exec(source_or_code, globals=None, locals=None)` — PyPy:
/// pyopcode.py builtin_exec.
///
/// Compiles `source` if necessary, then runs the resulting code object in
/// the supplied namespaces.  When the namespaces are dicts, pyre converts
/// them into `DictStorage`s before invocation and copies the post-run
/// namespace contents back so that callers see the new bindings.
fn builtin_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("exec() requires source"));
    }
    let source = args[0];
    let globals_arg = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let locals_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    exec_or_eval(source, globals_arg, locals_arg, false)
}

/// `eval(source_or_code, globals=None, locals=None)` — same plumbing as
/// exec but returns the value of the expression.
fn builtin_eval(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("eval() requires source"));
    }
    let source = args[0];
    let globals_arg = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let locals_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    exec_or_eval(source, globals_arg, locals_arg, true)
}

fn exec_or_eval(
    source: PyObjectRef,
    globals_arg: PyObjectRef,
    locals_arg: PyObjectRef,
    is_eval: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // Resolve a runnable code object: accept a precompiled W_Code or
    // compile a str on the fly.
    let code_obj_ref = unsafe {
        if pyre_object::is_str(source) {
            let s = pyre_object::w_str_get_value(source).to_string();
            let mode = if is_eval {
                crate::compile::Mode::Eval
            } else {
                crate::compile::Mode::Exec
            };
            let code = crate::compile::compile_source(&s, mode)
                .map_err(|e| crate::PyError::new(crate::PyErrorKind::ValueError, e))?;
            let code_ptr = Box::into_raw(Box::new(code)) as *const ();
            crate::w_code_new(code_ptr)
        } else if !source.is_null() && crate::is_code(source) {
            source
        } else {
            return Err(crate::PyError::type_error(
                "exec() / eval() expects str or code",
            ));
        }
    };
    let raw_code = unsafe {
        crate::w_code_get_ptr(code_obj_ref as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // pyopcode.py:738-759 exec_ — refuse a code object that needs a
    // closure with the exec-side TypeError.  PyPy exposes a keyword-only
    // `w_closure` parameter and builds an `outer_func` from it after
    // validating the tuple of cells.  Pyre's exec() builtin currently
    // accepts only source/globals/locals, so no closure cells can be
    // supplied yet.
    //
    // eval() takes no closure parameter even on CPython; the eval-side
    // error message comes from initialize_frame_scopes (pyframe.py:242-246
    // "directly executed code object may not contain free variables") via
    // createframe → ?, which is what compiling.py:99 eval_function reaches
    // through code.exec_code.
    if !is_eval {
        let needed_freevars = unsafe { (&*raw_code).freevars.len() };
        if needed_freevars > 0 {
            return Err(crate::PyError::type_error(format!(
                "code object requires a closure of exactly length {needed_freevars}"
            )));
        }
    }

    // pypy/interpreter/eval.py:28-33 Code.exec_code keeps w_globals and
    // w_locals as separate dict references — STORE_GLOBAL writes to
    // w_globals and STORE_NAME writes to w_locals.  Pyre mirrors this by
    // building a fresh DictStorage per role and syncing each back to the
    // caller's dict on exit.  When `locals is globals` (module-level exec
    // / dataclasses), both sides reuse the same storage so semantics
    // collapse to PyPy's `space.createframe(self, w_globals)` followed by
    // a same-dict setdictscope.
    fn is_none_or_null(w_obj: PyObjectRef) -> bool {
        w_obj.is_null() || unsafe { pyre_object::is_none(w_obj) }
    }

    fn type_name_of(w_obj: PyObjectRef) -> String {
        unsafe {
            match crate::typedef::r#type(w_obj) {
                Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
                None => (*(*w_obj).ob_type).name.to_string(),
            }
        }
    }

    fn is_dict_w(w_obj: PyObjectRef) -> bool {
        unsafe {
            let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
            crate::baseobjspace::isinstance_w(w_obj, w_dict_type)
        }
    }

    /// Build a `DictStorage` mirror for the user dict and bind the
    /// user dict's `W_DictObject` as the storage's `mirror_target`,
    /// so every storage-side write (STORE_GLOBAL) and delete
    /// (DELETE_GLOBAL) propagates straight back into the user dict's
    /// entries Vec while exec runs.  PyPy `pyopcode.py:771-776`
    /// achieves the same shape by running the frame on the user dict
    /// directly; pyre's frame layout still uses `*mut DictStorage`
    /// for the bytecode-handler fastpath, so we get parity by paring
    /// a forward storage proxy (W_DictObject → storage, already
    /// wired) with this back-mirror (storage → W_DictObject) — the
    /// two halves stand in for PyPy's single `W_DictMultiObject`.
    ///
    /// The initial population copy is mandatory because the bytecode
    /// handlers read from `*mut DictStorage`; LOAD_GLOBAL would
    /// otherwise miss every binding the user dict already held.  The
    /// mirror runs from this point on so no post-exec drain is
    /// needed: the user dict is always in sync.
    ///
    /// RAII binding for the frame's globals storage.
    ///
    /// Lifecycle (PyPy `pyopcode.py:771-776` parity — the user dict
    /// and the frame's `w_globals` are one and the same throughout
    /// the run):
    ///
    /// * **User dict has an existing `dict_storage_proxy`** (typical
    ///   for `module.__dict__` whose proxy is the module's own
    ///   storage with a back-mirror to the entries Vec): the binding
    ///   reuses that proxy directly as the frame's globals.  No temp
    ///   allocation, no proxy swap, no post-exec replay — STORE_GLOBAL
    ///   / DELETE_GLOBAL inside `exec("del x", module.__dict__)` land
    ///   on the very dict the module sees afterwards.
    ///
    /// * **User dict has no existing proxy** (typical fresh
    ///   `exec(src, {})`): the binding owns a temp `DictStorage`
    ///   pre-populated from the user dict's str-keyed entries with
    ///   `mirror_target = backing`; `attach_forward_proxy` is called
    ///   exactly once after all fallible setup so a `?` early-return
    ///   never leaves a dangling proxy hand-off.
    ///
    /// `Drop` (and `detach`) clears both halves idempotently: forward
    /// proxy restored to its pre-exec value, and `mirror_target`
    /// reset on the temp storage so the box can drop without an
    /// observer holding a freed pointer.  This is the cleanup hole
    /// guard for `createframe` and `setdictscope_object` early
    /// returns.
    struct GlobalsBinding {
        /// Owned temp storage when allocated; `None` when reusing an
        /// existing proxy storage.
        owned: Option<Box<crate::DictStorage>>,
        /// The storage the frame uses for `w_globals`.  Either the
        /// owned temp storage's interior pointer, or the user dict's
        /// existing proxy.
        storage_ptr: *mut crate::DictStorage,
        /// User dict's W_DictObject backing (`PY_NULL` when no user
        /// dict, e.g. `exec(src)` with no globals arg).
        backing: pyre_object::PyObjectRef,
        /// `true` once `attach_forward_proxy` swapped the user dict's
        /// proxy to point at our temp storage; cleared by `detach`.
        proxy_attached: bool,
        /// User dict's pre-attach proxy, restored by `detach` when
        /// `proxy_attached` is true.  Always null otherwise.
        saved_proxy: *mut u8,
    }

    impl GlobalsBinding {
        fn empty() -> Self {
            Self {
                owned: None,
                storage_ptr: std::ptr::null_mut(),
                backing: std::ptr::null_mut(),
                proxy_attached: false,
                saved_proxy: std::ptr::null_mut(),
            }
        }

        fn from_user_dict(d: pyre_object::PyObjectRef) -> Self {
            let backing = unsafe { crate::type_methods::resolve_dict_backing(d) };
            if backing.is_null() {
                return Self::empty();
            }
            let existing_proxy = unsafe { pyre_object::w_dict_get_dict_storage_proxy(backing) };
            if !existing_proxy.is_null() {
                // Reuse the existing proxy storage directly.  PyPy
                // `pyopcode.py:771` — the frame runs on the very
                // same dict the caller passed in, so `exec("del x",
                // module.__dict__)` removes from the module's
                // storage rather than a temp copy that gets thrown
                // away.  No proxy swap, no replay, no extra
                // bookkeeping.
                return Self {
                    owned: None,
                    storage_ptr: existing_proxy as *mut crate::DictStorage,
                    backing,
                    proxy_attached: false,
                    saved_proxy: std::ptr::null_mut(),
                };
            }
            // Fresh user dict — allocate a temp storage, pre-populate
            // from the user dict's str-keyed entries, and bind the
            // back-mirror so storage-side writes propagate to the
            // user dict's entries Vec while exec runs.  Forward proxy
            // attach is deferred to `attach_forward_proxy` so any
            // fallible setup between here and `frame.run()`
            // (createframe, setdictscope_object) cannot leave the
            // user dict pointing at a Box we're about to drop.
            let mut ns = Box::new(crate::DictStorage::new());
            unsafe {
                for (key, value) in pyre_object::w_dict_items(backing) {
                    if !value.is_null() && pyre_object::is_str(key) {
                        let name = pyre_object::w_str_get_value(key).to_string();
                        crate::dict_storage_store(&mut ns, &name, value);
                    }
                }
                ns.set_mirror_target(backing);
            }
            ns.fix_ptr();
            let storage_ptr: *mut crate::DictStorage = ns.as_mut() as *mut _;
            Self {
                owned: Some(ns),
                storage_ptr,
                backing,
                proxy_attached: false,
                saved_proxy: std::ptr::null_mut(),
            }
        }

        /// Attach the temp storage as the user dict's
        /// `dict_storage_proxy` so alias mutations
        /// (`exec("g['x']=1; y=x", g)`) propagate from the user
        /// dict's W_DictObject into the frame's storage.  No-op when
        /// reusing an existing proxy.  Call exactly once, AFTER all
        /// fallible setup, so a `?` early-return cannot leave the
        /// user dict pointing at a Box we're about to drop.
        unsafe fn attach_forward_proxy(&mut self) {
            unsafe {
                if self.proxy_attached
                    || self.owned.is_none()
                    || self.backing.is_null()
                    || self.storage_ptr.is_null()
                {
                    return;
                }
                self.saved_proxy = pyre_object::w_dict_get_dict_storage_proxy(self.backing);
                pyre_object::w_dict_set_dict_storage_proxy(
                    self.backing,
                    self.storage_ptr as *mut u8,
                );
                self.proxy_attached = true;
            }
        }

        /// Restore the user dict's pre-exec proxy and clear the temp
        /// storage's `mirror_target`.  Idempotent.  Called via `Drop`
        /// on every exit path including `?` early-returns and
        /// panic-unwinds.
        unsafe fn detach(&mut self) {
            unsafe {
                if self.proxy_attached {
                    pyre_object::w_dict_set_dict_storage_proxy(self.backing, self.saved_proxy);
                    self.proxy_attached = false;
                    self.saved_proxy = std::ptr::null_mut();
                }
                if let Some(storage) = self.owned.as_deref_mut() {
                    storage.set_mirror_target(pyre_object::PY_NULL);
                }
            }
        }

        fn storage_ptr(&self) -> *mut crate::DictStorage {
            self.storage_ptr
        }
    }

    impl Drop for GlobalsBinding {
        fn drop(&mut self) {
            unsafe { self.detach() }
        }
    }

    fn ensure_eval_builtins(
        ns: &mut crate::DictStorage,
        w_globals_obj: pyre_object::PyObjectRef,
        exec_ctx: *const crate::PyExecutionContext,
    ) -> Result<(), crate::PyError> {
        // pypy/module/__builtin__/compiling.py:109-110 eval:
        //
        //   if not space.contains_w(w_globals, space.newtext("__builtins__")):
        //       space.setitem_str(w_globals, "__builtins__", space.builtin)
        //
        // This is intentionally NOT pyopcode.py:773's `setdefault`
        // call-method path; dict-subclass `setdefault` overrides do
        // not fire for eval() in PyPy.
        let w_builtin = if !exec_ctx.is_null() {
            unsafe { (*exec_ctx).get_builtin() }
        } else {
            pyre_object::PY_NULL
        };
        if w_builtin.is_null() {
            return Ok(());
        }
        if !w_globals_obj.is_null() {
            let key = pyre_object::w_str_new("__builtins__");
            if !crate::baseobjspace::contains(w_globals_obj, key)? {
                crate::baseobjspace::setitem(w_globals_obj, key, w_builtin)?;
            }
            return Ok(());
        }
        if crate::dict_storage_get(ns, "__builtins__").is_none() {
            crate::dict_storage_store(ns, "__builtins__", w_builtin);
        }
        Ok(())
    }

    fn ensure_exec_builtins(
        ns: &mut crate::DictStorage,
        w_globals_obj: pyre_object::PyObjectRef,
        caller_frame: *const crate::PyFrame,
        exec_ctx: *const crate::PyExecutionContext,
    ) -> Result<(), crate::PyError> {
        // pypy/interpreter/pyopcode.py:773-774
        //   space.call_method(w_globals, 'setdefault',
        //                     '__builtins__', self.get_builtin())
        // — `self` is the caller frame, so `get_builtin()` returns the
        // builtin picked at caller-frame creation (`pyframe.py:115-116`),
        // not the EC's default.  When the caller frame's picked builtin
        // is unavailable (e.g. exec called from outside any frame), fall
        // through to the EC default.
        let w_builtin = if !caller_frame.is_null() {
            unsafe { (*caller_frame).get_builtin() }
        } else if !exec_ctx.is_null() {
            unsafe { (*exec_ctx).get_builtin() }
        } else {
            pyre_object::PY_NULL
        };
        if w_builtin.is_null() {
            return Ok(());
        }
        // PyPy `space.call_method(w_globals, 'setdefault', ...)` —
        // the receiver is the ORIGINAL `w_globals` object, not the
        // backing W_DictObject that GlobalsBinding strips off via
        // `resolve_dict_backing`.  Dispatching on the user object lets
        // dict-subclass `setdefault` overrides fire.
        if !w_globals_obj.is_null() {
            let key = pyre_object::w_str_new("__builtins__");
            let setdefault = crate::baseobjspace::getattr(w_globals_obj, "setdefault")?;
            crate::call_and_check(setdefault, &[key, w_builtin])?;
            return Ok(());
        }
        if crate::dict_storage_get(ns, "__builtins__").is_none() {
            crate::dict_storage_store(ns, "__builtins__", w_builtin);
        }
        Ok(())
    }

    // pypy/interpreter/pyopcode.py:2003-2013 ensure_ns —
    //   globals: not None ⇒ isinstance_w(w_dict) else TypeError
    //   locals : not None ⇒ space.lookup(__getitem__) is not None
    //                       else TypeError "must be a mapping or None"
    let funcname = if is_eval { "eval" } else { "exec" };
    if !is_none_or_null(globals_arg) && !is_dict_w(globals_arg) {
        return Err(crate::PyError::type_error(format!(
            "{funcname}() arg 2 must be a dict, not {}",
            type_name_of(globals_arg)
        )));
    }
    if !is_none_or_null(locals_arg)
        && unsafe { crate::baseobjspace::lookup(locals_arg, "__getitem__").is_none() }
    {
        return Err(crate::PyError::type_error(format!(
            "{funcname}() arg 3 must be a mapping or None, not {}",
            type_name_of(locals_arg)
        )));
    }

    let caller_frame = crate::eval::CURRENT_FRAME.with(|current| current.get());
    let exec_ctx = if caller_frame.is_null() {
        std::ptr::null::<crate::PyExecutionContext>()
    } else {
        unsafe { (*caller_frame).execution_context }
    };

    let mut globals_binding = if !is_none_or_null(globals_arg) {
        GlobalsBinding::from_user_dict(globals_arg)
    } else {
        GlobalsBinding::empty()
    };
    // Anonymous fallback storage for the no-globals + no-caller-frame
    // path (`exec(src)` outside any frame).  Owned outside the
    // GlobalsBinding because it has no W_DictObject backing — there
    // is nothing to mirror to and no proxy to swap.
    let mut anon_globals: Option<Box<crate::DictStorage>> = None;
    let mut globals_ptr = if !globals_binding.storage_ptr().is_null() {
        globals_binding.storage_ptr()
    } else if !caller_frame.is_null() {
        unsafe { (*caller_frame).get_w_globals() }
    } else {
        std::ptr::null_mut()
    };
    if globals_ptr.is_null() {
        let mut storage = Box::new(crate::DictStorage::new());
        storage.fix_ptr();
        let raw: *mut crate::DictStorage = storage.as_mut() as *mut _;
        anon_globals = Some(storage);
        globals_ptr = raw;
    }
    // Attach the forward proxy BEFORE seeding `__builtins__` so PyPy's
    // eval `setitem_str` / exec `call_method(..., 'setdefault', ...)`
    // lands in BOTH the user dict's W_DictObject AND the frame storage
    // that `pick_builtin` / LOAD_GLOBAL will consult.  PyPy's single
    // `W_DictMultiObject` makes this a non-issue; pyre's split storage
    // requires the proxy bridge live before the write.  Drop detaches
    // even on `?` early-return from builtins seeding / createframe.
    unsafe {
        globals_binding.attach_forward_proxy();
    }
    if !globals_ptr.is_null() {
        // Pass the ORIGINAL `globals_arg` (the user's PyObjectRef —
        // possibly a dict subclass instance) so eval/exec dispatch
        // through the same receiver PyPy uses.  When the caller omits
        // globals, Pyre only has the caller frame's `*mut DictStorage`,
        // so this falls back to storage-direct seeding.
        let w_globals_obj = if !is_none_or_null(globals_arg) {
            globals_arg
        } else if !caller_frame.is_null() {
            // Caller-frame globals are kept as `*mut DictStorage`;
            // there is no PyObjectRef to dispatch through.  Anonymous
            // path — direct storage write.
            pyre_object::PY_NULL
        } else {
            pyre_object::PY_NULL
        };
        if is_eval {
            ensure_eval_builtins(unsafe { &mut *globals_ptr }, w_globals_obj, exec_ctx)?;
        } else {
            ensure_exec_builtins(
                unsafe { &mut *globals_ptr },
                w_globals_obj,
                caller_frame,
                exec_ctx,
            )?;
        }
    }

    // pypy/interpreter/pyopcode.py:771-776 — `code.exec_code(space,
    // w_globals, w_locals, outer_func)` runs the frame on the
    // user-supplied dict directly.  Pyre routes locals through
    // `frame.setdictscope_object(w_locals)` so STORE_NAME / LOAD_NAME /
    // DELETE_NAME dispatch via `space.setitem` / `space.getitem` /
    // `space.delitem` on the live mapping (dict subclass `__getitem__`
    // overrides win, alias mutations are visible immediately, and
    // there is no entry/exit storage copy + drain pair).  Both exact
    // dicts and arbitrary `__getitem__`-bearing mappings now share
    // this path.
    //
    // pypy/interpreter/pyopcode.py:2015 ensure_ns — when the caller
    // omits both globals and locals, exec falls back to caller globals
    // (already wired above) AND caller `getdictscope()`.  When the
    // caller omits ONLY locals, locals collapse to globals (PyPy
    // `pyopcode.py:2010-2013`), which the existing same-storage shape
    // below covers via the `is_none_or_null(locals_arg)` skip.
    //
    // Resolve the implicit caller-locals only when globals_arg is also
    // None: that's the `exec(src)` shape where PyPy hands the caller's
    // live local mapping in via `frame.getdictscope()`.  When
    // globals_arg is supplied but locals_arg is None, PyPy collapses
    // locals=globals and pyre's existing same-dict path handles it.
    let mut implicit_caller_locals: pyre_object::PyObjectRef = std::ptr::null_mut();
    if is_none_or_null(globals_arg) && is_none_or_null(locals_arg) && !caller_frame.is_null() {
        // pyframe.py:540 getdictscope returns the caller's
        // w_locals (PyObjectRef) — same dict-or-mapping the
        // interpreter sees inside the calling function body.
        implicit_caller_locals = unsafe { (*caller_frame).getdictscope_w()? };
    }
    let mut locals_object_arg: pyre_object::PyObjectRef = std::ptr::null_mut();
    if !is_none_or_null(locals_arg) {
        let same_as_globals =
            !is_none_or_null(globals_arg) && std::ptr::eq(locals_arg, globals_arg);
        if !same_as_globals {
            // Dict and non-dict mapping arms share the
            // setdictscope_object path — for exact dict locals this
            // matches PyPy's `code.exec_code(space, w_globals,
            // w_locals)` chain (pyopcode.py:776) which feeds
            // `space.setitem(w_locals, name, value)` to STORE_NAME.
            // Pyre's earlier `is_dict_w` arm built a storage copy and
            // drained it back through a `Vec<String>` snapshot to
            // mirror DELETE_GLOBAL while preserving alias mutations;
            // routing through `setdictscope_object` retires the copy +
            // snapshot entirely.
            locals_object_arg = locals_arg;
        }
    }
    // eval.py:31-33 Code.exec_code → space.createframe(...) + frame.run().
    // For eval() with a code object that carries freevars, createframe
    // surfaces pyframe.py:242-246's TypeError "directly executed code
    // object may not contain free variables" directly — exec()'s
    // closure-mismatch TypeError was already raised above.
    let mut frame = match crate::createframe(code_obj_ref as *const (), globals_ptr, exec_ctx, None)
    {
        Ok(frame) => frame,
        Err(err) => {
            let _ = raw_code;
            return Err(err);
        }
    };
    frame.fix_array_ptrs();
    // eval.py:32 frame.setdictscope(w_locals, ...) — only when locals
    // were separately supplied.  Without this call, initialize_frame_scopes'
    // module-code arm has already bound w_locals = w_globals, matching
    // PyPy's `exec(src, g)` (and `exec(src, g, l)` where `l is g`).
    if !locals_object_arg.is_null() {
        frame.setdictscope_object(locals_object_arg)?;
    } else if !implicit_caller_locals.is_null() {
        // pyopcode.py:2015 — `exec(src)` with no globals/locals uses
        // the caller's `getdictscope()` as locals.  Skip when the
        // resolved object is the caller's globals (module-level exec
        // collapses to locals=globals — same-dict shape kept by the
        // module-frame's initialize_frame_scopes binding).
        let caller_globals = unsafe { (*caller_frame).get_w_globals() };
        let same_as_globals =
            !caller_globals.is_null() && std::ptr::eq(globals_ptr, caller_globals);
        if !same_as_globals {
            frame.setdictscope_object(implicit_caller_locals)?;
        }
    }
    // run() rather than execute_frame so that
    // `eval(compile("(x for x in [])", ..., 'eval'))` of generator-flagged
    // code returns the wrapped generator object instead of executing the
    // body inline.
    let result = frame.run();

    // Explicit detach so the proxy/mirror state is cleared BEFORE
    // `globals_binding` drops; `Drop` then becomes idempotent and
    // covers the rare panic-unwind path.  STORE_GLOBAL /
    // DELETE_GLOBAL writes during exec already landed on the user
    // dict via the back-mirror, and alias mutations propagated
    // through the forward proxy, so no post-exec drain is needed
    // (PyPy `pyopcode.py:771-776` parity — the user dict and the
    // frame's globals are one and the same throughout the run).
    unsafe {
        globals_binding.detach();
    }
    let _ = anon_globals; // keep anon storage alive across exec

    let _ = raw_code; // keep raw_code alive until after exec for safety.
    match result {
        Ok(v) if is_eval => Ok(v),
        Ok(_) => Ok(pyre_object::w_none()),
        Err(e) => Err(e),
    }
}

fn builtin_globals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("globals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        // `pypy/module/__builtin__/interp_inspect.py:5 globals_w` →
        // `caller.get_w_globals()` returns the dict directly without
        // wrapping.  PyPy keeps a single dict per module so subsequent
        // `globals()` / `frame.f_globals` / `f.__globals__` /
        // `module.__dict__` accesses on the same module share one
        // identity.  Pyre routes through the lazy cached
        // `get_w_globals_obj` (Step 1 of the w_globals type
        // migration) which returns the canonical W_DictObject paired
        // with the frame's storage — same identity invariant via
        // `dict_storage_to_dict`'s mirror_target.  Returning a fresh
        // wrapper per call (as the previous shape did) silently
        // diverged on `globals() is module.__dict__`.
        let dict = unsafe { (*frame).get_w_globals_obj() };
        if dict.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        Ok(dict)
    })
}

fn builtin_locals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("locals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "locals() requires an active frame",
            ));
        }
        // pyframe.py:540 getdictscope: non-dict mapping locals are
        // returned to caller as the live `w_locals_object` so
        // `locals()['x'] = ...` goes through the mapping's
        // `__setitem__` and reads observe the same object the
        // exec/eval caller passed in.
        let frame_mut = unsafe { &mut *frame };
        let w_locals_object = frame_mut.get_w_locals_object();
        if !w_locals_object.is_null() {
            return Ok(w_locals_object);
        }
        // `pypy/interpreter/pyframe.py:540 getdictscope` runs
        // `fast2locals()` then returns `self.debugdata.w_locals`.
        // `fast2locals` (`pyframe.py:557-562`) lazily allocates the
        // backing dict on first call and stamps it into
        // `debugdata.w_locals`, so subsequent calls reuse the same
        // storage — `locals() is locals()` (function scope) and
        // `locals() is globals()` (module scope, where
        // `debugdata.w_locals is w_globals`) both hold.  Pyre's
        // `frame.fast2locals()` follows the same shape; the canonical
        // W_DictObject for that storage is then resolved via
        // `dict_storage_to_dict` (mirror_target invariant).
        frame_mut.fast2locals()?;
        let w_locals = frame_mut.get_w_locals();
        if !w_locals.is_null() {
            return Ok(crate::baseobjspace::dict_storage_to_dict(
                w_locals as *const _,
            ));
        }
        // Fallback only fires when `fast2locals` neither found a
        // mapping nor materialised a storage (no fast locals, no
        // module-level w_globals shadow).  Build a plain dict so
        // `locals()` still returns *something* in that degenerate
        // case rather than `None`.
        Ok(pyre_object::w_dict_new())
    })
}

fn builtin_vars(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return builtin_locals(args);
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "vars() takes at most 1 argument.",
        ));
    }
    let obj = args[0];
    let has_dict = unsafe {
        pyre_object::is_instance(obj) || crate::is_function(obj) || pyre_object::is_module(obj)
    } || crate::baseobjspace::ATTR_TABLE
        .with(|table| table.borrow().contains_key(&(obj as usize)));
    if !has_dict {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    let dict = crate::baseobjspace::getattr(obj, "__dict__")
        .map_err(|_| crate::PyError::type_error("vars() argument must have __dict__ attribute"))?;
    if dict.is_null() || unsafe { pyre_object::is_none(dict) } {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    Ok(dict)
}

/// `dir([obj])` — PyPy: pypy/module/__builtin__/interp_classobj.py descr_dir
///
/// Without argument: names in the current local scope (not supported).
/// With argument: sorted list of attribute names from obj.__dict__ plus
/// type MRO. Modules expose their namespace via w_module_get_namespace.
fn builtin_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        // Return empty list — pyre doesn't currently expose locals here.
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    let mut names: Vec<String> = Vec::new();
    unsafe {
        if pyre_object::is_module(obj) {
            // Route through `w_module.w_dict` so dict-subclass-backed
            // Modules (`pypy/module/__builtin__/moduledef.py:102-103
            // Module(space, None, w_builtin)`) surface their entries
            // alongside storage-backed modules.  PyPy
            // `pypy/interpreter/module.py:77 Module.getdict()` returns
            // the dict directly regardless of subclass; pyre branches
            // on the underlying shape:
            //   - exact `W_DictObject` → `w_dict_str_entries` returns
            //     the storage-proxy union view in one call.
            //   - dict subclass instance → iterate keys via the
            //     standard `iter()` protocol so the subclass's
            //     `__iter__` override participates (PyPy's
            //     `space.iter(w_dict)` would do the same).
            let w_dict = pyre_object::w_module_get_w_dict(obj);
            if !w_dict.is_null() {
                if pyre_object::is_dict(w_dict) {
                    for (name, _) in pyre_object::dictobject::w_dict_str_entries(w_dict) {
                        names.push(name);
                    }
                } else if let Ok(keys_iter) = crate::baseobjspace::iter(w_dict) {
                    if let Ok(keys) = crate::builtins::collect_iterable(keys_iter) {
                        for k in keys {
                            if pyre_object::is_str(k) {
                                names.push(pyre_object::w_str_get_value(k).to_string());
                            }
                        }
                    }
                }
            }
        } else if pyre_object::is_type(obj) {
            let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(obj);
            if !ns_ptr.is_null() {
                let ns = &*(ns_ptr as *const DictStorage);
                for (name, _) in ns.entries() {
                    names.push(name.to_string());
                }
            }
        } else if pyre_object::is_instance(obj) {
            // typeobject.py:1247 type_dir collects names from the instance
            // dict and the type's MRO. The instance dict for hasdict objects
            // is the live W_DictObject returned by w_obj.getdict(space).
            let w_dict = crate::baseobjspace::getdict(obj);
            if !w_dict.is_null() {
                for (k, _) in pyre_object::w_dict_items(w_dict) {
                    if pyre_object::is_str(k) {
                        names.push(pyre_object::w_str_get_value(k).to_string());
                    }
                }
            }
            // Plus any legacy ATTR_TABLE entries (slot values stored via Member
            // descriptors before the live-dict path existed).
            crate::baseobjspace::ATTR_TABLE.with(|table| {
                if let Some(attrs) = table.borrow().get(&(obj as usize)) {
                    for (name, _) in attrs {
                        names.push(name.clone());
                    }
                }
            });
            // Plus the type's own namespace.
            let w_type = pyre_object::w_instance_get_type(obj);
            if !w_type.is_null() && pyre_object::is_type(w_type) {
                let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(w_type);
                if !ns_ptr.is_null() {
                    let ns = &*(ns_ptr as *const DictStorage);
                    for (name, _) in ns.entries() {
                        names.push(name.to_string());
                    }
                }
            }
        } else if pyre_object::is_dict(obj) {
            for (k, _) in pyre_object::w_dict_items(obj) {
                if pyre_object::is_str(k) {
                    names.push(pyre_object::w_str_get_value(k).to_string());
                }
            }
        } else {
            // Fallback: walk the object's type namespace so user-
            // visible attributes on builtin W_Root types (PyTraceback,
            // dict view, etc.) surface in dir().  Mirrors PyPy's
            // `typeobject.py:1247 type_dir` MRO walk.  Excluded for
            // module/instance/type/dict above because those have
            // richer paths that combine instance+class entries.
            if let Some(w_type) = crate::typedef::r#type(obj) {
                if pyre_object::is_type(w_type) {
                    let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(w_type);
                    if !ns_ptr.is_null() {
                        let ns = &*(ns_ptr as *const DictStorage);
                        for (name, _) in ns.entries() {
                            names.push(name.to_string());
                        }
                    }
                }
            }
        }
    }
    names.sort();
    names.dedup();
    let items: Vec<_> = names.into_iter().map(|s| w_str_new(&s)).collect();
    Ok(w_list_new(items))
}

/// `id(obj)` — PyPy: baseobjspace.py id → object identity as int
fn builtin_id(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "id() takes exactly one argument");
    Ok(w_int_new(args[0] as i64))
}

/// `hash(obj)` — PyPy: `descroperation.py:1006 hash`.
///
/// CPython / PyPy raise `TypeError: unhashable type: 'X'` when the
/// object's class lacks a non-None `__hash__` slot.  Built-in
/// mutable containers (dict, list, set, bytearray) explicitly set
/// `__hash__ = None` (`dictmultiobject.py:1431`, `listobject.py`,
/// `setobject.py`).  Pyre's `hash_value` falls through to identity
/// for un-typed objects, so this wrapper screens the known
/// unhashables before calling.
fn builtin_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "hash() takes exactly one argument");
    let obj = args[0];
    if obj.is_null() {
        return Err(crate::PyError::type_error("hash() argument is null"));
    }
    unsafe {
        let kind = if pyre_object::is_dict(obj) {
            Some("dict")
        } else if pyre_object::is_list(obj) {
            Some("list")
        } else if pyre_object::is_set(obj) {
            // `frozenset` is hashable per setobject.py:623-642 _hash_frozenset.
            Some("set")
        } else if pyre_object::is_bytearray(obj) {
            Some("bytearray")
        } else if pyre_object::dictviewobject::is_dict_view(obj) {
            // `dictmultiobject.py:1619 _is_set_like` views inherit set's
            // unhashable semantics; values view also isn't hashable.
            Some("dict view")
        } else {
            None
        };
        if let Some(name) = kind {
            return Err(crate::PyError::type_error(&format!(
                "unhashable type: '{}'",
                name
            )));
        }
    }
    Ok(w_int_new(hash_value(obj)))
}

/// `pypy/objspace/std/intobject.py:36-37` — `HASH_BITS = 61` (64-bit
/// host); `HASH_MODULUS = 2**HASH_BITS - 1`.  The Mersenne-prime
/// modulus is what makes pyre's `hash(42) == hash(42.0) ==
/// hash(2**100 + 42)`-class invariants hold: every per-type hash
/// reduces its input modulo the same `HASH_MODULUS`, so equal
/// numeric values land on the same residue.
const HASH_BITS: u32 = 61;
const HASH_MODULUS: u64 = (1u64 << HASH_BITS) - 1;
/// `floatobject.py:29-30` HASH_INF / HASH_NAN sentinels.
const HASH_INF: i64 = 314159;
const HASH_NAN: i64 = 0;

/// `pypy/objspace/std/intobject.py:1231-1249 _hash_int` line-by-line
/// port —
///
/// ```python
/// def _hash_int(a):
///     sign = 1 - ((a < 0) << 1)
///     x = r_uint(a)
///     x *= r_uint(sign)
///     x = (x & HASH_MODULUS) + (x >> HASH_BITS)
///     x -= HASH_MODULUS * (x >= HASH_MODULUS)
///     h = intmask(intmask(x) * sign)
///     return h - (h == -1)
/// ```
///
/// `intmask` is "wrap-around to i64" on 64-bit RPython; Rust uses
/// `as i64` for the same effect.  `-1` is the CPython-reserved "no
/// hash" sentinel, so any natural hash producing `-1` is bumped to
/// `-2`.
#[inline]
pub(crate) fn _hash_int(a: i64) -> i64 {
    let sign: i64 = 1 - (((a < 0) as i64) << 1);
    // r_uint(a) * r_uint(sign) — multiply as u64 to compute |a|
    // without UB on `a == i64::MIN`.  When a < 0, sign == -1 and
    // (-1 as u64) == u64::MAX; the wrapping product yields the
    // two's-complement negation of a, i.e. its absolute value.
    let mut x = (a as u64).wrapping_mul(sign as u64);
    x = (x & HASH_MODULUS) + (x >> HASH_BITS);
    if x >= HASH_MODULUS {
        x -= HASH_MODULUS;
    }
    let h = (x as i64).wrapping_mul(sign);
    h - (h == -1) as i64
}

/// `pypy/objspace/std/longobject.py:468-494 _hash_long` line-by-line
/// port:
///
/// ```python
/// def _hash_long(v):
///     i = v.numdigits() - 1
///     if i == -1: return 0
///     x = _load_unsigned_digit(0)
///     while i >= 0:
///         x = ((x << _HASH_SHIFT) & HASH_MODULUS) | (x >> (HASH_BITS - _HASH_SHIFT))
///         x += v.udigit(i)
///         if SHIFT > HASH_BITS:
///             x = (x & HASH_MODULUS) + (x >> HASH_BITS)
///         if x >= HASH_MODULUS:
///             x -= HASH_MODULUS
///         i -= 1
///     h = intmask(intmask(x) * v.get_sign())
///     return h - (h == -1)
/// ```
///
/// PyPy's `rbigint` uses 31-bit digits (`SHIFT = 31`); Rust's
/// `num_bigint::BigInt` exposes `iter_u32_digits()` so we use
/// `SHIFT = 32`.  Since `HASH_MODULUS = 2^61 - 1` is a Mersenne
/// prime, `value mod HASH_MODULUS` is independent of the digit
/// base — `_hash_int(v) == _hash_long(BigInt::from(v))` for any
/// `v` that fits in `i64`.
#[inline]
pub(crate) fn _hash_long(v: &BigInt) -> i64 {
    let sign = match v.sign() {
        malachite_bigint::Sign::Plus => 1i64,
        malachite_bigint::Sign::Minus => -1i64,
        malachite_bigint::Sign::NoSign => return 0, // numdigits == 0
    };
    // Walk digits from MSB to LSB.  `iter_u32_digits()` yields
    // little-endian; collect + reverse so we mirror PyPy's loop.
    let digits: Vec<u32> = v.iter_u32_digits().collect();
    let mut x: u64 = 0;
    const SHIFT: u32 = 32;
    const HASH_SHIFT: u32 = SHIFT % HASH_BITS; // 32 — `SHIFT > HASH_BITS` arm reached later
    for &d in digits.iter().rev() {
        // x = ((x << HASH_SHIFT) & HASH_MODULUS) | (x >> (HASH_BITS - HASH_SHIFT))
        let left = (x.wrapping_shl(HASH_SHIFT)) & HASH_MODULUS;
        let right = if HASH_BITS > HASH_SHIFT {
            x >> (HASH_BITS - HASH_SHIFT)
        } else {
            0
        };
        x = left | right;
        x = x.wrapping_add(d as u64);
        if SHIFT > HASH_BITS {
            x = (x & HASH_MODULUS) + (x >> HASH_BITS);
        }
        if x >= HASH_MODULUS {
            x -= HASH_MODULUS;
        }
    }
    let h = (x as i64).wrapping_mul(sign);
    h - (h == -1) as i64
}

/// `pypy/objspace/std/floatobject.py:790-822 _hash_float` line-by-line
/// port:
///
/// ```python
/// def _hash_float(v):
///     if math.isinf(v): return HASH_INF if v > 0 else -HASH_INF
///     # nan hash handled elsewhere (W_FloatObject.descr_hash routes to HASH_NAN)
///     m, e = math.frexp(v)
///     sign = 1
///     if m < 0: sign = -1; m = -m
///     x = r_uint(0)
///     while m:
///         x = ((x << 28) & HASH_MODULUS) | x >> (HASH_BITS - 28)
///         m *= 268435456.0          # 2**28
///         e -= 28
///         y = r_uint(m)
///         m -= y
///         x += y
///         if x >= HASH_MODULUS: x -= HASH_MODULUS
///     e = e % HASH_BITS if e >= 0 else HASH_BITS - 1 - ((-1 - e) % HASH_BITS)
///     x = ((x << e) & HASH_MODULUS) | x >> (HASH_BITS - e)
///     x = intmask(intmask(x) * sign)
///     x -= (x == -1)
///     return x
/// ```
///
/// For finite floats whose value is an integer that fits in `i64`,
/// this returns the same value as `_hash_int(v as i64)` — that's the
/// `hash(42) == hash(42.0)` invariant.  NaN is dispatched to
/// `HASH_NAN` by the caller (PyPy's `W_FloatObject.descr_hash` does
/// the NaN check before reaching `_hash_float`).
#[inline]
pub(crate) fn _hash_float(v: f64) -> i64 {
    if v.is_nan() {
        return HASH_NAN;
    }
    if v.is_infinite() {
        return if v > 0.0 { HASH_INF } else { -HASH_INF };
    }
    // For integral values that fit in i64, short-circuit to
    // `_hash_int(v as i64)` so `hash(2.0) == hash(2)`.  The frexp
    // walk below produces the same result, but the integer fast path
    // avoids floating-point noise on already-integer inputs.
    if v.fract() == 0.0 && (i64::MIN as f64) <= v && v <= (i64::MAX as f64) {
        return _hash_int(v as i64);
    }
    let (mut m, mut e) = libm_frexp(v);
    let mut sign: i64 = 1;
    if m < 0.0 {
        sign = -1;
        m = -m;
    }
    let mut x: u64 = 0;
    while m != 0.0 {
        x = ((x.wrapping_shl(28)) & HASH_MODULUS) | (x >> (HASH_BITS - 28));
        m *= 268435456.0; // 2**28
        e -= 28;
        let y = m as u64;
        m -= y as f64;
        x = x.wrapping_add(y);
        if x >= HASH_MODULUS {
            x -= HASH_MODULUS;
        }
    }
    // `e = e % HASH_BITS if e >= 0 else HASH_BITS - 1 - ((-1 - e) % HASH_BITS)`
    let e_mod: u32 = if e >= 0 {
        (e as u32) % HASH_BITS
    } else {
        HASH_BITS - 1 - (((-1 - e) as u32) % HASH_BITS)
    };
    x = ((x.wrapping_shl(e_mod)) & HASH_MODULUS) | (x >> (HASH_BITS - e_mod));
    let h = (x as i64).wrapping_mul(sign);
    h - (h == -1) as i64
}

/// Rust port of Python's `math.frexp(x) -> (mantissa, exponent)` where
/// `x == mantissa * 2**exponent` and `0.5 <= |mantissa| < 1` (or both
/// are 0).  `libm` isn't a workspace dep, so we use `f64::to_bits`
/// to peek at the IEEE-754 exponent directly.
#[inline]
fn libm_frexp(v: f64) -> (f64, i32) {
    if v == 0.0 || !v.is_finite() {
        return (v, 0);
    }
    let bits = v.to_bits();
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    if raw_exp == 0 {
        // Subnormal — normalise by multiplying with 2**54.
        let (m, e) = libm_frexp(v * (1u64 << 54) as f64);
        return (m, e - 54);
    }
    let exponent = raw_exp - 1022;
    let mantissa_bits = (bits & !(0x7ffu64 << 52)) | (1022u64 << 52);
    (f64::from_bits(mantissa_bits), exponent)
}

/// `pypy/objspace/std/tupleobject.py:358-401 descr_hash` line-by-line
/// port — xxHash sequence hash (CPython 3.8+ tuple hash):
///
/// ```python
/// XXPRIME_1 = 0x9E3779B185EBCA87
/// XXPRIME_2 = 0xC2B2AE3D27D4EB4F
/// XXPRIME_5 = 0x27D4EB2F165667C5
/// xxrotate = lambda x: (x << 31) | (x >> 33)
///
/// acc = XXPRIME_5
/// for w_item in items:
///     lane = space.hash_w(w_item)
///     acc += lane * XXPRIME_2
///     acc = xxrotate(acc)
///     acc *= XXPRIME_1
/// acc += len(items) ^ (XXPRIME_5 ^ 3527539)
/// if acc == -1: acc = 1546275796 + 1
/// return acc
/// ```
const XXPRIME_1: u64 = 11400714785074694791;
const XXPRIME_2: u64 = 14029467366897019727;
const XXPRIME_5: u64 = 2870177450012600261;

#[inline]
fn _hash_tuple_xx(items: &[i64]) -> i64 {
    let mut acc: u64 = XXPRIME_5;
    for &lane in items {
        acc = acc.wrapping_add((lane as u64).wrapping_mul(XXPRIME_2));
        // xxrotate: rotate-left 31 bits.
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(XXPRIME_1);
    }
    // Mangle in the length per `tupleobject.py:399`.
    let n = items.len() as u64;
    acc = acc.wrapping_add(n ^ (XXPRIME_5 ^ 3527539u64));
    let mut h = acc as i64;
    if h == -1 {
        h = 1546275796 + 1;
    }
    h
}

/// `pypy/objspace/std/unicodeobject.py:341-345 W_UnicodeObject.hash_w`
/// parity:
///
/// ```python
/// def hash_w(self):
///     x = compute_hash(self._utf8)
///     x -= (x == -1)
///     return x
/// ```
///
/// `compute_hash` is `rpython.rlib.objectmodel.compute_hash` —
/// on 64-bit hosts it delegates to `rpython.rlib.rsiphash.siphash24`
/// with a 16-byte secret key set via `rsiphash.choose_initial_seed`
/// (rpython/rlib/rsiphash.py:48).  The seed is read from
/// `PYTHONHASHSEED`, defaulting to a randomised value at process
/// start (CPython parity: `Random_Hash_Function_Seed_String`).
///
/// Pyre uses a fixed 16-byte key here so test runs are deterministic
/// (matching `PYTHONHASHSEED=0`).  Switching to a randomised seed
/// is straight-forward (`OnceLock<[u8; 16]>` seeded from
/// `getrandom` or the env var) once tests are robust to it.
fn _hash_str(s: &str) -> i64 {
    use core::hash::Hasher;
    // `rpython/rlib/rsiphash.py:60-62 _build_key_from_seed` — when
    // `PYTHONHASHSEED=0` the key is the 16-byte all-zero buffer.
    // Pyre runs with the deterministic seed for reproducibility,
    // matching PyPy's `PYTHONHASHSEED=0` byte-for-byte.  Wiring a
    // user-overridable seed is straight-forward (`OnceLock<[u8; 16]>`
    // sampled from `getrandom` or the env var) once tests are
    // robust to it.
    static SECRET: [u8; 16] = [0u8; 16];
    let mut hasher = siphasher::sip::SipHasher24::new_with_key(&SECRET);
    hasher.write(s.as_bytes());
    let raw = hasher.finish() as i64;
    raw - ((raw == -1) as i64)
}

/// `pypy/objspace/std/setobject.py:623-642 W_FrozensetObject.descr_hash`
/// line-by-line port:
///
/// ```python
/// multi = r_uint(1822399083) + r_uint(1822399083) + 1
/// hash = r_uint(1927868237)
/// hash *= r_uint(self.length() + 1)
/// for item in items:
///     h = space.hash_w(item)
///     value = (r_uint(h ^ (h << 16) ^ 89869747) * multi)
///     hash = hash ^ value
/// hash ^= (hash >> 11) ^ (hash >> 25)
/// hash = hash * 69069 + 907133923
/// hash = intmask(hash)
/// if hash == -1: hash = 590923713
/// return hash
/// ```
#[inline]
fn _hash_frozenset(items: &[i64]) -> i64 {
    let multi: u64 = 1822399083u64.wrapping_add(1822399083).wrapping_add(1);
    let mut h: u64 = 1927868237;
    h = h.wrapping_mul((items.len() as u64).wrapping_add(1));
    for &item_hash in items {
        let item_u = item_hash as u64;
        let v = (item_u ^ item_u.wrapping_shl(16) ^ 89869747u64).wrapping_mul(multi);
        h ^= v;
    }
    h ^= (h >> 11) ^ (h >> 25);
    h = h.wrapping_mul(69069).wrapping_add(907133923);
    let mut hi = h as i64;
    if hi == -1 {
        hi = 590923713;
    }
    hi
}

/// `pypy/objspace/std/objspace.py StdObjSpace.hash` parity — share one
/// implementation across builtin `hash()`, dict / set lookup, and
/// tuple/frozenset content hashing.  Dispatches to PyPy's per-type
/// hash helpers (`_hash_int`/`_hash_long`/`_hash_float`/
/// `_hash_tuple_xx`/`_hash_frozenset`), so:
///
/// - `hash(42) == hash(42.0) == hash(2**100 + 42)` (Mersenne mod)
/// - `hash((1, 2)) == hash((1, 2))` regardless of allocation identity
/// - `hash(frozenset(...))` is deterministic and order-independent
///
/// `unicodeobject.py W_UnicodeObject.descr_hash` routes through
/// RPython's `compute_hash(self._utf8)` which is siphash on 64-bit;
/// pyre keeps an FNV-style multiplicative mix here (functional but
/// not bit-identical to CPython/PyPy).  Convergence target: import
/// siphash24 from a workspace dep.
pub(crate) fn hash_value(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_int(obj) {
            return _hash_int(w_int_get_value(obj));
        }
        if is_bool(obj) {
            return if pyre_object::w_bool_get_value(obj) {
                1
            } else {
                0
            };
        }
        if is_long(obj) {
            return _hash_long(pyre_object::w_long_get_value(obj));
        }
        if is_float(obj) {
            return _hash_float(pyre_object::w_float_get_value(obj));
        }
        if is_str(obj) {
            return _hash_str(w_str_get_value(obj));
        }
        if pyre_object::is_none(obj) {
            return 0;
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let mut hashes = Vec::with_capacity(n);
            for i in 0..(n as i64) {
                if let Some(item) = w_tuple_getitem(obj, i) {
                    hashes.push(hash_value(item));
                }
            }
            return _hash_tuple_xx(&hashes);
        }
        if pyre_object::is_frozenset(obj) {
            let hashes: Vec<i64> = pyre_object::w_set_items(obj)
                .into_iter()
                .map(hash_value)
                .collect();
            return _hash_frozenset(&hashes);
        }
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__hash__") {
                let r = crate::call_function(method, &[obj]);
                if !r.is_null() && is_int(r) {
                    return w_int_get_value(r);
                }
            }
        }
        obj as i64
    }
}

/// `ord(c)` — PyPy: operation.py ord
fn builtin_ord(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "ord() takes exactly one argument");
    let s = unsafe { w_str_get_value(args[0]) };
    assert!(s.len() == 1, "ord() expected a character");
    Ok(w_int_new(s.chars().next().unwrap() as i64))
}

/// `chr(i)` — PyPy: operation.py chr
fn builtin_chr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "chr() takes exactly one argument");
    let obj = args[0];
    let val = if unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else if unsafe { is_str(obj) } {
        // pyre treats `b'...'` literals as str; iterating yields 1-char str.
        // Accept single-char str and use its codepoint as the value.
        let s = unsafe { w_str_get_value(obj) };
        match s.chars().next() {
            Some(c) => c as i64,
            None => {
                return Err(crate::PyError::type_error(
                    "chr() arg must be a non-empty string",
                ));
            }
        }
    } else {
        // int subclass instance — check __int_value__ via builtin_int
        match builtin_int(args) {
            Ok(v) if unsafe { is_int(v) } => unsafe { w_int_get_value(v) },
            _ => {
                return Err(crate::PyError::type_error(
                    "an integer is required (got type non-int)",
                ));
            }
        }
    };
    if val < 0 || val > 0x10ffff {
        return Err(crate::PyError::type_error(&format!(
            "chr() arg not in range(0x110000): {val}"
        )));
    }
    match char::from_u32(val as u32) {
        Some(c) => Ok(w_str_new(&c.to_string())),
        None => Err(crate::PyError::type_error(&format!(
            "chr() arg not in range(0x110000): {val}"
        ))),
    }
}

/// `map()` — PyPy: functional.py W_Map (returns iterator)
fn builtin_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "map() requires at least 2 arguments",
        ));
    }
    let func = args[0];
    // `pypy/module/__builtin__/functional.py:336-355 W_Map.descr_new`
    // accepts any number of iterables; each iteration calls
    // `func(*tuple_of_one_item_per_iterable)` and stops at the
    // shortest iterable.  Single-iterable map is the trivial case.
    let iters: Vec<Vec<PyObjectRef>> = args[1..]
        .iter()
        .map(|&it| collect_iterable(it))
        .collect::<Result<_, _>>()?;
    let min_len = iters.iter().map(|v| v.len()).min().unwrap_or(0);
    let mut results = Vec::with_capacity(min_len);
    for i in 0..min_len {
        let call_args: Vec<PyObjectRef> = iters.iter().map(|v| v[i]).collect();
        let result = crate::call_function(func, &call_args);
        results.push(result);
    }
    let n = results.len();
    let list = pyre_object::w_list_new(results);
    Ok(pyre_object::w_seq_iter_new(list, n))
}

/// `zip(*iterables)` — PyPy: functional.py W_Zip
fn builtin_zip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // `pypy/module/__builtin__/functional.py:411-414 W_Zip.descr_new`
    // accepts `strict` as a keyword.  Pyre's flat builtin ABI surfaces
    // kwargs as a trailing `__pyre_kw__` dict; strip it before the
    // positional walk and look up `strict` from it.
    let (args, kwargs) = split_builtin_kwargs(args);
    kwarg_reject_unknown(kwargs, &["strict"], "zip")?;
    let strict = kwarg_get(kwargs, "strict")
        .map(|v| crate::baseobjspace::is_true(v))
        .unwrap_or(false);
    if args.is_empty() {
        return Ok(pyre_object::w_seq_iter_new(
            pyre_object::w_list_new(vec![]),
            0,
        ));
    }
    // Collect all iterables into lists, zip them
    let mut iters: Vec<Vec<PyObjectRef>> = Vec::new();
    for &arg in args {
        let mut items = Vec::new();
        unsafe {
            if pyre_object::is_list(arg) {
                let n = pyre_object::w_list_len(arg);
                for i in 0..n {
                    if let Some(v) = pyre_object::w_list_getitem(arg, i as i64) {
                        items.push(v);
                    }
                }
            } else if pyre_object::is_tuple(arg) {
                let n = pyre_object::w_tuple_len(arg);
                for i in 0..n {
                    if let Some(v) = pyre_object::w_tuple_getitem(arg, i as i64) {
                        items.push(v);
                    }
                }
            } else {
                // Use iter/next protocol
                let it = crate::baseobjspace::iter(arg)?;
                loop {
                    match crate::baseobjspace::next(it) {
                        Ok(v) => items.push(v),
                        Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        iters.push(items);
    }
    let min_len = iters.iter().map(|v| v.len()).min().unwrap_or(0);
    // `functional.py:411-435 W_Zip.descr_new` — strict mode raises
    // ValueError when iterables have different lengths.  Detect by
    // checking max == min; report which argument was longer/shorter
    // per CPython's `zip()` message format.
    if strict {
        let max_len = iters.iter().map(|v| v.len()).max().unwrap_or(0);
        if max_len != min_len {
            // CPython's `zip()` reports the first SHORT argument (the
            // one with `len < max_len`) as "shorter than argument N"
            // where N is some earlier (longer) argument.  Find the
            // first short index — that's the one being reported.
            let short = iters.iter().position(|v| v.len() < max_len).unwrap_or(0);
            // Pick any longer argument as the reference point; CPython
            // names the first one (typically argument 1).
            let long = iters.iter().position(|v| v.len() == max_len).unwrap_or(0);
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!(
                    "zip() argument {} is shorter than argument {}",
                    short + 1,
                    long + 1
                ),
            ));
        }
    }
    let mut result = Vec::with_capacity(min_len);
    for i in 0..min_len {
        let tuple_items: Vec<_> = iters.iter().map(|v| v[i]).collect();
        result.push(pyre_object::w_tuple_new(tuple_items));
    }
    let list = pyre_object::w_list_new(result);
    Ok(pyre_object::w_seq_iter_new(list, min_len))
}

/// `pypy/module/__builtin__/functional.py:253-272 W_Enumerate.descr_new`
/// parity:
///
/// ```python
/// def descr_new(space, w_subtype, w_iterable, w_start=None):
///     ...
///     if w_start is None:
///         start = 0
///     else:
///         start = space.index_w(w_start)
///     ...
/// ```
///
/// `space.index_w` accepts ANY object exposing `__index__`
/// (subclasses of int, NumPy ints, etc.) — not just exact int.  The
/// kwarg surface is also strict: anything other than `start=` is a
/// TypeError per the gateway's parsed signature.
// `pypy/module/__builtin__/functional.py:253-275 W_Enumerate.descr___new__`
// line-by-line port — constructs the lazy `W_Enumerate` iterator,
// resolving `start` via `space.index_w` (with overflow promotion to a
// bigint slot) and capturing either the source iterator or the
// source list directly when `start == 0 + isinstance(it, list)`.
fn builtin_enumerate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "enumerate() requires at least one argument",
        ));
    }
    if positional.len() > 2 {
        return Err(crate::PyError::type_error(format!(
            "enumerate() takes at most 2 arguments ({} given)",
            positional.len()
        )));
    }
    kwarg_reject_unknown(kwargs, &["start"], "enumerate")?;
    let start_obj = if positional.len() > 1 {
        Some(positional[1])
    } else {
        kwarg_get(kwargs, "start")
    };
    // `functional.py:255-264 descr___new__` — `space.index(w_start)`
    // then `space.int_w(w_start)`; on OverflowError, drop into bigint
    // slot.  Pyre uses i64 directly and would overflow on bigint
    // start; surfaced as a PRE-EXISTING-ADAPTATION below (W_Enumerate
    // can still promote during iteration once start fits in i64).
    let start = match start_obj {
        Some(o) if !unsafe { pyre_object::is_none(o) } => space_index_w(o)?,
        _ => 0,
    };
    let source = positional[0];
    // `functional.py:268-271` — `if start == 0 and type(w_iterable) is
    // W_ListObject: w_iter = w_iterable` (skip space.iter for the
    // common list-source case so __next__ can `getitem(index)`
    // directly).  Otherwise call `space.iter(w_iterable)`.
    let w_iter_or_list = if start == 0 && unsafe { pyre_object::is_list(source) } {
        source
    } else {
        crate::baseobjspace::iter(source)?
    };
    Ok(pyre_object::enumerateobject::w_enumerate_new(
        w_iter_or_list,
        start,
        pyre_object::PY_NULL, // i64 fast-path active per :225-227
    ))
}

/// `reversed()` — PyPy: functional.py W_ReversedIterator
fn builtin_reversed(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "reversed() requires one argument",
        ));
    }
    let obj = args[0];
    unsafe {
        // List: reverse a copy
        if pyre_object::is_list(obj) {
            let n = pyre_object::w_list_len(obj);
            let mut items = Vec::with_capacity(n);
            for i in (0..n as i64).rev() {
                if let Some(v) = pyre_object::w_list_getitem(obj, i) {
                    items.push(v);
                }
            }
            return Ok(pyre_object::w_seq_iter_new(
                pyre_object::w_list_new(items),
                n,
            ));
        }
        // Tuple: reverse
        if pyre_object::is_tuple(obj) {
            let n = pyre_object::w_tuple_len(obj);
            let mut items = Vec::with_capacity(n);
            for i in (0..n as i64).rev() {
                if let Some(v) = pyre_object::w_tuple_getitem(obj, i) {
                    items.push(v);
                }
            }
            let t = pyre_object::w_tuple_new(items);
            return Ok(pyre_object::w_seq_iter_new(t, n));
        }
        // Instance __reversed__
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__reversed__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    // Fallback: collect any iterable (range, dict views, etc.) and reverse.
    // PyPy: operation.py reversed falls back to __reversed__ then __getitem__/
    // __len__. Our iterator protocol covers sequences via collect_iterable.
    let items = collect_iterable(obj)?;
    let mut rev = items;
    rev.reverse();
    let n = rev.len();
    Ok(pyre_object::w_seq_iter_new(pyre_object::w_list_new(rev), n))
}

/// `pypy/module/__builtin__/functional.py:328-340 builtin_sorted`
/// parity:
///
/// ```python
/// @unwrap_spec(reverse=bool)
/// def sorted(space, w_iterable, w_key=None, reverse=False):
///     w_lst = space.call_function(space.w_list, w_iterable)
///     space.call_method(w_lst, "sort", w_key, space.newbool(reverse))
///     return w_lst
/// ```
///
/// PyPy's `sort` then calls into `listobject.py W_ListObject.descr_sort`
/// which dispatches keys through `space.lt`.  Pyre mirrors:
///   - exactly one positional iterable (extras → TypeError),
///   - kwargs limited to `{key, reverse}` (others → TypeError),
///   - per-comparison errors (e.g. user `__lt__` raises) propagate
///     instead of silently falling back to "treat as not less".
fn builtin_sorted(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (positional, kwargs) = split_builtin_kwargs(args);
    if positional.is_empty() {
        return Err(crate::PyError::type_error(
            "sorted() requires at least one argument",
        ));
    }
    if positional.len() > 1 {
        return Err(crate::PyError::type_error(format!(
            "sorted() takes at most 1 positional argument ({} given)",
            positional.len()
        )));
    }
    kwarg_reject_unknown(kwargs, &["key", "reverse"], "sorted")?;
    let iterable = positional[0];
    let key_fn = kwarg_get(kwargs, "key").filter(|k| unsafe { !pyre_object::is_none(*k) });
    let reverse = kwarg_get(kwargs, "reverse")
        .map(|v| crate::baseobjspace::is_true(v))
        .unwrap_or(false);
    let mut items = collect_iterable(iterable)?;
    // `pypy/objspace/std/listobject.py W_ListObject.descr_sort` →
    // build (key, item) pairs, sort by key, optionally reverse.
    let keyed: Vec<(PyObjectRef, PyObjectRef)> = if let Some(kf) = key_fn {
        items
            .iter()
            .map(|&item| {
                let k = crate::call_function(kf, &[item]);
                (k, item)
            })
            .collect()
    } else {
        items.iter().map(|&item| (item, item)).collect()
    };
    let mut keyed = keyed;
    // `pypy/objspace/std/listsort.py listsort_lt` defers to
    // `space.lt(a, b)` and propagates exceptions; if the user's
    // `__lt__` raises, sort halts with that error.  Rust's
    // `sort_by` closure cannot return Result, so capture the first
    // error via a Cell and surface it after the sort completes.
    let sort_error: std::cell::Cell<Option<crate::PyError>> = std::cell::Cell::new(None);
    let sort_lt = |ka: PyObjectRef, kb: PyObjectRef| -> bool {
        if sort_error
            .take()
            .map(|e| {
                sort_error.set(Some(e));
                true
            })
            .unwrap_or(false)
        {
            return false;
        }
        match crate::baseobjspace::compare(ka, kb, crate::baseobjspace::CompareOp::Lt) {
            Ok(r) => crate::baseobjspace::is_true(r),
            Err(e) => {
                sort_error.set(Some(e));
                false
            }
        }
    };
    keyed.sort_by(|(ka, _), (kb, _)| {
        let ab = sort_lt(*ka, *kb);
        if ab {
            return std::cmp::Ordering::Less;
        }
        let ba = sort_lt(*kb, *ka);
        if ba {
            return std::cmp::Ordering::Greater;
        }
        // Fast-path tail kept for the cases where `compare` returns
        // `False` for both directions (legacy unhashable / unorderable
        // pairs that pyre still has) — preserves prior behaviour.
        unsafe {
            if is_int(*ka) && is_int(*kb) {
                return w_int_get_value(*ka).cmp(&w_int_get_value(*kb));
            }
            if is_str(*ka) && is_str(*kb) {
                return w_str_get_value(*ka).cmp(w_str_get_value(*kb));
            }
            if is_float(*ka) && is_float(*kb) {
                return pyre_object::w_float_get_value(*ka)
                    .partial_cmp(&pyre_object::w_float_get_value(*kb))
                    .unwrap_or(std::cmp::Ordering::Equal);
            }
            std::cmp::Ordering::Equal
        }
    });
    if let Some(err) = sort_error.take() {
        return Err(err);
    }
    if reverse {
        keyed.reverse();
    }
    items = keyed.into_iter().map(|(_, v)| v).collect();
    Ok(w_list_new(items))
}

/// `any(iterable)` — PyPy: operation.py any
/// `any(iterable)` — PyPy: baseobjspace.py any_w
pub fn builtin_any_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_any(args)
}
fn builtin_any(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "any() takes exactly one argument");
    let items = collect_iterable(args[0])?;
    for item in items {
        if crate::baseobjspace::is_true(item) {
            return Ok(w_bool_from(true));
        }
    }
    Ok(w_bool_from(false))
}

/// `all(iterable)` — PyPy: operation.py all
/// `all(iterable)` — PyPy: baseobjspace.py all_w
/// Shared file wrapper type — plain instance with hasdict so that
/// open() can attach `__file_data__` / `__file_pos__` / `__file_mode__`
/// as instance attributes, matching the PyPy FileIO/TextIOWrapper
/// duck-typing surface without a dedicated W_FileObject.
pub fn file_wrapper_type() -> PyObjectRef {
    thread_local! {
        static FILE_WRAPPER_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
    }
    FILE_WRAPPER_TYPE.with(|c| {
        *c.get_or_init(|| {
            let tp = crate::typedef::make_builtin_type("_io.TextIOWrapper", init_file_wrapper_type);
            unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
            tp
        })
    })
}

/// PyPy: pypy/module/_io/interp_iobase.py W_IOBase.
fn init_file_wrapper_type(ns: &mut DictStorage) {
    crate::dict_storage_store(ns, "read", make_builtin_function("read", file_method_read));
    crate::dict_storage_store(
        ns,
        "readline",
        make_builtin_function_with_arity("readline", file_method_readline, 1),
    );
    crate::dict_storage_store(
        ns,
        "readlines",
        make_builtin_function_with_arity("readlines", file_method_readlines, 1),
    );
    crate::dict_storage_store(
        ns,
        "write",
        make_builtin_function_with_arity("write", file_method_write, 2),
    );
    crate::dict_storage_store(
        ns,
        "close",
        make_builtin_function_with_arity("close", file_method_close, 1),
    );
    crate::dict_storage_store(
        ns,
        "flush",
        make_builtin_function_with_arity("flush", file_method_close, 1),
    );
    crate::dict_storage_store(
        ns,
        "__enter__",
        make_builtin_function_with_arity("__enter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__exit__",
        make_builtin_function("__exit__", |args| {
            // Call close on exit.
            let _ = file_method_close(&args[..1]);
            Ok(w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        make_builtin_function_with_arity("__iter__", |args| Ok(args[0]), 1),
    );
    crate::dict_storage_store(
        ns,
        "__next__",
        make_builtin_function_with_arity(
            "__next__",
            |args| {
                let line = file_method_readline(args)?;
                unsafe {
                    let s = pyre_object::w_str_get_value(line);
                    if s.is_empty() {
                        return Err(crate::PyError::stop_iteration());
                    }
                }
                Ok(line)
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "seek",
        make_builtin_function("seek", |args| {
            if args.len() >= 2 {
                let _ = crate::baseobjspace::setattr(args[0], "__file_pos__", args[1]);
            }
            Ok(w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "tell",
        make_builtin_function_with_arity(
            "tell",
            |args| {
                if let Ok(pos) = crate::baseobjspace::getattr(args[0], "__file_pos__") {
                    Ok(pos)
                } else {
                    Ok(w_int_new(0))
                }
            },
            1,
        ),
    );
}

fn file_get_data(self_obj: PyObjectRef) -> String {
    crate::baseobjspace::getattr(self_obj, "__file_data__")
        .ok()
        .and_then(|d| unsafe {
            if pyre_object::is_str(d) {
                Some(pyre_object::w_str_get_value(d).to_string())
            } else {
                None
            }
        })
        .unwrap_or_default()
}

fn file_get_pos(self_obj: PyObjectRef) -> usize {
    crate::baseobjspace::getattr(self_obj, "__file_pos__")
        .ok()
        .and_then(|p| unsafe {
            if pyre_object::is_int(p) {
                Some(pyre_object::w_int_get_value(p) as usize)
            } else {
                None
            }
        })
        .unwrap_or(0)
}

fn file_set_pos(self_obj: PyObjectRef, pos: usize) {
    let _ = crate::baseobjspace::setattr(self_obj, "__file_pos__", w_int_new(pos as i64));
}

fn file_method_read(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("read() requires self"));
    }
    let data = file_get_data(args[0]);
    let pos = file_get_pos(args[0]);
    let remaining = &data[pos.min(data.len())..];
    let n = if args.len() >= 2 {
        let n_val = unsafe { pyre_object::w_int_get_value(args[1]) };
        if n_val < 0 {
            remaining.len()
        } else {
            (n_val as usize).min(remaining.len())
        }
    } else {
        remaining.len()
    };
    // Slice at char boundaries — count by bytes.
    let end = n.min(remaining.len());
    let chunk = &remaining[..end];
    file_set_pos(args[0], pos + end);
    Ok(w_str_new(chunk))
}

fn file_method_readline(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("readline() requires self"));
    }
    let data = file_get_data(args[0]);
    let pos = file_get_pos(args[0]);
    if pos >= data.len() {
        return Ok(w_str_new(""));
    }
    let rest = &data[pos..];
    let end = rest.find('\n').map(|i| i + 1).unwrap_or(rest.len());
    let line = &rest[..end];
    file_set_pos(args[0], pos + end);
    Ok(w_str_new(line))
}

fn file_method_readlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("readlines() requires self"));
    }
    let mut lines = Vec::new();
    loop {
        let line = file_method_readline(args)?;
        let s = unsafe { pyre_object::w_str_get_value(line) };
        if s.is_empty() {
            break;
        }
        lines.push(line);
    }
    Ok(w_list_new(lines))
}

fn file_method_write(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error("write() requires (self, data)"));
    }
    // Append to __file_data__ and update on close.
    unsafe {
        let prev = file_get_data(args[0]);
        let s = if pyre_object::is_str(args[1]) {
            pyre_object::w_str_get_value(args[1]).to_string()
        } else if pyre_object::bytesobject::is_bytes_like(args[1]) {
            let data = pyre_object::bytesobject::bytes_like_data(args[1]);
            String::from_utf8_lossy(data).into_owned()
        } else {
            return Err(crate::PyError::type_error("write() expects str or bytes"));
        };
        let new_data = format!("{prev}{s}");
        let len = s.len();
        let _ = crate::baseobjspace::setattr(args[0], "__file_data__", w_str_new(&new_data));
        let _ = crate::baseobjspace::setattr(args[0], "__file_dirty__", w_bool_from(true));
        Ok(w_int_new(len as i64))
    }
}

fn file_method_close(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_none());
    }
    // If the file was opened in a writable mode, flush the in-memory
    // buffer to disk.
    let dirty = crate::baseobjspace::getattr(args[0], "__file_dirty__")
        .ok()
        .map(|v| unsafe { pyre_object::is_bool(v) && pyre_object::w_bool_get_value(v) })
        .unwrap_or(false);
    if dirty {
        if let (Ok(name), Ok(mode)) = (
            crate::baseobjspace::getattr(args[0], "__file_name__"),
            crate::baseobjspace::getattr(args[0], "__file_mode__"),
        ) {
            let name_s = unsafe { pyre_object::w_str_get_value(name).to_string() };
            let mode_s = unsafe { pyre_object::w_str_get_value(mode).to_string() };
            let data = file_get_data(args[0]);
            let append = mode_s.contains('a');
            let write_res = if append {
                std::fs::OpenOptions::new()
                    .append(true)
                    .create(true)
                    .open(&name_s)
                    .and_then(|mut f| std::io::Write::write_all(&mut f, data.as_bytes()))
            } else {
                std::fs::write(&name_s, data.as_bytes())
            };
            if let Err(e) = write_res {
                return Err(crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(5),
                    format!("{e}: '{name_s}'"),
                ));
            }
            let _ = crate::baseobjspace::setattr(args[0], "__file_dirty__", w_bool_from(false));
        }
    }
    Ok(w_none())
}

/// builtins.open(file, mode='r', ...) — PyPy: io.open → FileIO + TextIOWrapper.
/// Minimal implementation that loads the entire file into memory and
/// returns a file wrapper instance.
pub fn builtin_open(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("open() missing 'file' argument"));
    }
    let path_obj = args[0];
    let path = unsafe {
        if pyre_object::is_str(path_obj) {
            pyre_object::w_str_get_value(path_obj).to_string()
        } else if pyre_object::bytesobject::is_bytes_like(path_obj) {
            let data = pyre_object::bytesobject::bytes_like_data(path_obj);
            String::from_utf8_lossy(data).into_owned()
        } else if let Ok(fspath) = crate::baseobjspace::getattr(path_obj, "__fspath__") {
            let result = crate::call_function(fspath, &[path_obj]);
            if !result.is_null() && pyre_object::is_str(result) {
                pyre_object::w_str_get_value(result).to_string()
            } else {
                return Err(crate::PyError::type_error(
                    "open(): path should be str, bytes, os.PathLike",
                ));
            }
        } else {
            return Err(crate::PyError::type_error(
                "open(): path should be str, bytes, os.PathLike",
            ));
        }
    };
    let mode: String = if args.len() >= 2 {
        unsafe {
            if pyre_object::is_str(args[1]) {
                pyre_object::w_str_get_value(args[1]).to_string()
            } else {
                "r".to_string()
            }
        }
    } else {
        "r".to_string()
    };

    let binary = mode.contains('b');
    let writing = mode.contains('w') || mode.contains('a') || mode.contains('x');
    let reading = mode.contains('r') || !writing;

    let data: String = if reading && !mode.contains('w') && !mode.contains('x') {
        match std::fs::read(&path) {
            Ok(bytes) => {
                if binary {
                    // Store bytes-as-string for now; we only support ASCII binary.
                    String::from_utf8_lossy(&bytes).into_owned()
                } else {
                    match String::from_utf8(bytes) {
                        Ok(s) => s,
                        Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
                    }
                }
            }
            Err(_e) if writing => String::new(),
            Err(e) => {
                return Err(crate::PyError::os_error_with_errno(
                    e.raw_os_error().unwrap_or(2),
                    format!("{e}: '{path}'"),
                ));
            }
        }
    } else {
        String::new()
    };

    let wrapper = pyre_object::w_instance_new(file_wrapper_type());
    let _ = crate::baseobjspace::setattr(wrapper, "__file_data__", w_str_new(&data));
    let _ = crate::baseobjspace::setattr(wrapper, "__file_pos__", w_int_new(0));
    let _ = crate::baseobjspace::setattr(wrapper, "__file_name__", w_str_new(&path));
    let _ = crate::baseobjspace::setattr(wrapper, "__file_mode__", w_str_new(&mode));
    let _ = crate::baseobjspace::setattr(wrapper, "name", w_str_new(&path));
    let _ = crate::baseobjspace::setattr(wrapper, "mode", w_str_new(&mode));
    let _ = crate::baseobjspace::setattr(wrapper, "closed", w_bool_from(false));
    Ok(wrapper)
}

pub fn builtin_all_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_all(args)
}
fn builtin_all(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "all() takes exactly one argument");
    let items = collect_iterable(args[0])?;
    for item in items {
        if !crate::baseobjspace::is_true(item) {
            return Ok(w_bool_from(false));
        }
    }
    Ok(w_bool_from(true))
}

/// `sum(iterable, start=0)` — PyPy: operation.py sum
fn builtin_sum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "sum() takes at least one argument",
        ));
    }
    let iterable = args[0];
    let start = args.get(1).copied().unwrap_or_else(|| w_int_new(0));
    let mut acc = start;
    // Use the generic iterator protocol so sum() works with any iterable
    // (generators, ranges, sets, dict views, ...), matching PyPy.
    for item in crate::builtins::collect_iterable(iterable)? {
        acc = crate::baseobjspace::add(acc, item)?;
    }
    Ok(acc)
}

/// `round(number, ndigits=None)` — PyPy: operation.py round
/// Round half to even (banker's rounding), matching Python 3 semantics.
fn round_half_even(v: f64) -> f64 {
    let rounded = v.round();
    // When exactly halfway, round to even.
    if (v - rounded).abs() == 0.5 {
        let truncated = v.trunc();
        if truncated % 2.0 == 0.0 {
            truncated
        } else {
            rounded
        }
    } else {
        rounded
    }
}

fn builtin_round(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "round() takes at least one argument");
    let obj = args[0];
    let ndigits = args.get(1);
    unsafe {
        if is_float(obj) {
            let v = floatobject::w_float_get_value(obj);
            return Ok(match ndigits {
                Some(nd) if is_int(*nd) => {
                    let n = w_int_get_value(*nd);
                    let factor = 10f64.powi(n as i32);
                    floatobject::w_float_new(round_half_even(v * factor) / factor)
                }
                _ => w_int_new(round_half_even(v) as i64),
            });
        }
        if is_int(obj) {
            return Ok(obj);
        }
    }
    panic!("round() not supported for this type");
}

/// `divmod(a, b)` — pypy/interpreter/baseobjspace.py:2159 divmod row.
fn builtin_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "divmod() takes exactly two arguments");
    crate::baseobjspace::divmod(args[0], args[1])
}

/// `pow(base, exp[, mod])` — pypy/interpreter/baseobjspace.py:2160 pow row.
fn builtin_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "pow() takes at least two arguments");
    if args.len() >= 3 && !unsafe { is_none(args[2]) } {
        crate::baseobjspace::pow3(args[0], args[1], args[2])
    } else {
        crate::baseobjspace::pow(args[0], args[1])
    }
}

/// `hex(x)` — PyPy: operation.py hex
fn builtin_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "hex() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0x{:x}", -v)
    } else {
        format!("0x{v:x}")
    };
    Ok(w_str_new(&s))
}

/// `oct(x)` — PyPy: operation.py oct
fn builtin_oct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "oct() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0o{:o}", -v)
    } else {
        format!("0o{v:o}")
    };
    Ok(w_str_new(&s))
}

/// `bin(x)` — PyPy: operation.py bin
fn builtin_bin(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "bin() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0b{:b}", -v)
    } else {
        format!("0b{v:b}")
    };
    Ok(w_str_new(&s))
}

/// `complex(real=0, imag=0)` — PyPy: complexobject.py descr__new__
fn builtin_complex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    use pyre_object::*;
    let real = if args.is_empty() {
        0.0
    } else {
        unsafe {
            let a = args[0];
            if is_bool(a) {
                w_bool_get_value(a) as i64 as f64
            } else if is_int(a) {
                w_int_get_value(a) as f64
            } else if is_float(a) {
                w_float_get_value(a)
            } else if is_str(a) {
                let s = crate::py_str(a);
                s.trim().parse::<f64>().map_err(|_| {
                    crate::PyError::new(
                        crate::PyErrorKind::ValueError,
                        format!("could not convert string to complex: '{s}'"),
                    )
                })?
            } else {
                0.0
            }
        }
    };
    let imag = if args.len() > 1 {
        unsafe {
            let a = args[1];
            if is_bool(a) {
                w_bool_get_value(a) as i64 as f64
            } else if is_int(a) {
                w_int_get_value(a) as f64
            } else if is_float(a) {
                w_float_get_value(a)
            } else {
                0.0
            }
        }
    } else {
        0.0
    };
    // No complex type yet — return float as approximation
    if imag == 0.0 {
        Ok(pyre_object::w_float_new(real))
    } else {
        Err(crate::PyError::type_error(
            "complex numbers not yet supported",
        ))
    }
}

/// `format(value, format_spec='')` — PyPy: operation.py format
fn builtin_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "format() takes at least one argument");
    let value = args[0];
    let spec = if args.len() > 1 {
        unsafe { crate::py_str(args[1]) }
    } else {
        String::new()
    };
    // With format_spec: delegate to __format__ or handle basic int specs
    if !spec.is_empty() {
        unsafe {
            let int_val = if pyre_object::is_bool(value) {
                Some(pyre_object::w_bool_get_value(value) as i64)
            } else if pyre_object::is_int(value) {
                Some(pyre_object::w_int_get_value(value))
            } else {
                None
            };
            if let Some(v) = int_val {
                let s = match spec.as_str() {
                    "d" => format!("{v}"),
                    "b" => format!("{v:b}"),
                    "o" => format!("{v:o}"),
                    "x" => format!("{v:x}"),
                    "X" => format!("{v:X}"),
                    "n" => format!("{v}"),
                    _ => format!("{v}"),
                };
                return Ok(w_str_new(&s));
            }
            if pyre_object::is_float(value) {
                let fv = pyre_object::w_float_get_value(value);
                let s = match spec.as_str() {
                    "f" => format!("{fv:.6}"),
                    "e" => format!("{fv:e}"),
                    "g" => format!("{fv}"),
                    _ => format!("{fv}"),
                };
                return Ok(w_str_new(&s));
            }
        }
    }
    let s = unsafe { crate::py_str(args[0]) };
    Ok(w_str_new(&s))
}

/// `__import__(name, globals=None, locals=None, fromlist=(), level=0)`
/// — PyPy: pypy/module/__builtin__/interp_import.importhook.
fn builtin_import_stub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_value(name_obj) }
    } else {
        ""
    };
    let globals = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fromlist = args.get(3).copied().unwrap_or(pyre_object::PY_NULL);
    let level = args
        .get(4)
        .copied()
        .filter(|&a| unsafe { pyre_object::is_int(a) })
        .map(|a| unsafe { pyre_object::w_int_get_value(a) })
        .unwrap_or(0);
    let exec_ctx = crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            std::ptr::null::<crate::PyExecutionContext>()
        } else {
            unsafe { (*frame).execution_context }
        }
    });
    crate::importing::importhook(name, globals, fromlist, level, exec_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_divmod_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_divmod(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            1
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            2
        );
    }

    #[test]
    fn test_builtin_divmod_allows_lhs_dunder_before_dead_proxy_rhs() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("DivmodLhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__divmod__",
                make_builtin_function("__divmod__", |_| {
                    Ok(w_tuple_new(vec![w_int_new(41), w_int_new(1)]))
                }),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_divmod(&[lhs, dead_proxy]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            41
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            1
        );
    }

    #[test]
    fn test_builtin_pow_three_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3), w_int_new(13)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 8);
    }

    #[test]
    fn test_builtin_pow_two_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 125);
    }

    #[test]
    fn test_builtin_pow_three_arg_allows_lhs_dunder_before_dead_proxy_exp() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            crate::dict_storage_store(
                ns,
                "__pow__",
                make_builtin_function("__pow__", |_| Ok(w_int_new(99))),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_pow(&[lhs, dead_proxy, w_int_new(7)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 99);
    }

    #[test]
    fn test_builtin_pow_three_arg_rejects_negative_exponent() {
        crate::typedef::init_typeobjects();
        let err = builtin_pow(&[w_int_new(5), w_int_new(-1), w_int_new(13)]).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "pow() 2nd argument cannot be negative when 3rd argument specified"
        );
    }
}
