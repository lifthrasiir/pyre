//! operator module definition.
//!
//! PyPy equivalent: pypy/module/operator/

use crate::{DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity};
use pyre_object::*;

fn op_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "index() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            return Ok(obj);
        }
        if is_bool(obj) {
            return Ok(w_int_new(if w_bool_get_value(obj) { 1 } else { 0 }));
        }
    }
    // Try __index__ dunder
    Ok(crate::call_function_or_identity(obj, "__index__"))
}

fn op_add(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::add(args[0], args[1]).unwrap_or(w_none()))
}

fn op_sub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::sub(args[0], args[1]).unwrap_or(w_none()))
}

fn op_mul(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(crate::baseobjspace::mul(args[0], args[1]).unwrap_or(w_none()))
}

fn op_eq(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Eq)
            .unwrap_or(w_none()),
    )
}

fn op_lt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Lt)
            .unwrap_or(w_none()),
    )
}

fn op_gt(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    Ok(
        crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Gt)
            .unwrap_or(w_none()),
    )
}

pub fn init(ns: &mut DictStorage) {
    dict_storage_store(ns, "index", make_builtin_function_with_arity("index", op_index, 1));
    dict_storage_store(ns, "add", make_builtin_function_with_arity("add", op_add, 2));
    dict_storage_store(ns, "sub", make_builtin_function_with_arity("sub", op_sub, 2));
    dict_storage_store(ns, "mul", make_builtin_function_with_arity("mul", op_mul, 2));
    dict_storage_store(
        ns,
        "truediv",
        make_builtin_function_with_arity("truediv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::truediv(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "floordiv",
        make_builtin_function_with_arity("floordiv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::floordiv(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "mod",
        make_builtin_function_with_arity("mod", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::mod_(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "pow",
        make_builtin_function_with_arity("pow", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::pow(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "neg",
        make_builtin_function_with_arity("neg", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::neg(args[0])
        }, 1),
    );
    dict_storage_store(
        ns,
        "pos",
        make_builtin_function_with_arity("pos", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::pos(args[0])
        }, 1),
    );
    dict_storage_store(
        ns,
        "abs",
        make_builtin_function_with_arity("abs", |args| {
            assert!(args.len() == 1);
            crate::builtins::builtin_abs(args)
        }, 1),
    );
    dict_storage_store(
        ns,
        "invert",
        make_builtin_function_with_arity("invert", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::invert(args[0])
        }, 1),
    );
    dict_storage_store(
        ns,
        "lshift",
        make_builtin_function_with_arity("lshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::lshift(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "rshift",
        make_builtin_function_with_arity("rshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::rshift(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "and_",
        make_builtin_function_with_arity("and_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::and_(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "or_",
        make_builtin_function_with_arity("or_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::or_(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "xor",
        make_builtin_function_with_arity("xor", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::xor(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "not_",
        make_builtin_function_with_arity("not_", |args| {
            assert!(args.len() == 1);
            Ok(w_bool_from(!crate::baseobjspace::is_true(args[0])))
        }, 1),
    );
    // interp_operator.py:138
    dict_storage_store(
        ns,
        "truth",
        make_builtin_function_with_arity("truth", |args| {
            assert!(args.len() == 1);
            Ok(w_bool_from(crate::baseobjspace::is_true(args[0])))
        }, 1),
    );
    dict_storage_store(
        ns,
        "is_",
        make_builtin_function_with_arity("is_", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(std::ptr::eq(args[0], args[1])))
        }, 2),
    );
    dict_storage_store(
        ns,
        "is_not",
        make_builtin_function_with_arity("is_not", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(!std::ptr::eq(args[0], args[1])))
        }, 2),
    );
    dict_storage_store(
        ns,
        "contains",
        make_builtin_function_with_arity("contains", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(crate::baseobjspace::contains(
                args[0], args[1],
            )?))
        }, 2),
    );
    dict_storage_store(
        ns,
        "getitem",
        make_builtin_function_with_arity("getitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::getitem(args[0], args[1])
        }, 2),
    );
    dict_storage_store(
        ns,
        "setitem",
        make_builtin_function_with_arity("setitem", |args| {
            assert!(args.len() == 3);
            crate::baseobjspace::setitem(args[0], args[1], args[2])?;
            Ok(w_none())
        }, 3),
    );
    dict_storage_store(
        ns,
        "delitem",
        make_builtin_function_with_arity("delitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::delitem(args[0], args[1])?;
            Ok(w_none())
        }, 2),
    );
    // Underscore aliases (CPython: __add__/__sub__/... via operator module).
    dict_storage_store(ns, "__add__", make_builtin_function_with_arity("__add__", op_add, 2));
    dict_storage_store(ns, "__sub__", make_builtin_function_with_arity("__sub__", op_sub, 2));
    dict_storage_store(ns, "__mul__", make_builtin_function_with_arity("__mul__", op_mul, 2));
    dict_storage_store(ns, "eq", make_builtin_function_with_arity("eq", op_eq, 2));
    dict_storage_store(ns, "lt", make_builtin_function_with_arity("lt", op_lt, 2));
    dict_storage_store(ns, "gt", make_builtin_function_with_arity("gt", op_gt, 2));
    dict_storage_store(
        ns,
        "le",
        make_builtin_function_with_arity("le", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Le)
        }, 2),
    );
    dict_storage_store(
        ns,
        "ge",
        make_builtin_function_with_arity("ge", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ge)
        }, 2),
    );
    dict_storage_store(
        ns,
        "ne",
        make_builtin_function_with_arity("ne", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ne)
        }, 2),
    );
    // itemgetter/attrgetter stubs — return callable objects
    dict_storage_store(
        ns,
        "itemgetter",
        make_builtin_function("itemgetter", |args| {
            // itemgetter(key) → lambda obj: obj[key]
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    dict_storage_store(
        ns,
        "attrgetter",
        make_builtin_function("attrgetter", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    dict_storage_store(
        ns,
        "methodcaller",
        make_builtin_function("methodcaller", |args| {
            Ok(if args.is_empty() { w_none() } else { args[0] })
        }),
    );
    dict_storage_store(
        ns,
        "length_hint",
        // pypy/module/operator/interp_operator.py:213-219
        //   @unwrap_spec(default='index')
        //   def length_hint(space, w_iterable, default=0):
        //       return space.newint(space.length_hint(w_iterable, default))
        // — `default` defaults to 0, must be unwrapped via `__index__`.
        // Pyre routes through `crate::baseobjspace::length_hint` (the
        // `space.length_hint` port), so `__length_hint__` priority +
        // negative-result ValueError + default fallback all match PyPy.
        make_builtin_function("length_hint", |args| {
            // PyPy gateway signature `length_hint(space, w_iterable,
            // default=0)` (interp_operator.py:213) accepts exactly 1 or
            // 2 application-level arguments.  Pyre's varargs entrypoint
            // must reject overflow itself — the gateway layer that
            // would normally enforce this has not been ported yet.
            if args.is_empty() || args.len() > 2 {
                return Err(crate::PyError::type_error(format!(
                    "length_hint expected 1 or 2 arguments, got {}",
                    args.len()
                )));
            }
            let w_iterable = args[0];
            let default = if let Some(&w_default) = args.get(1) {
                // unwrap_spec='index' — apply __index__ chain via space_index.
                let w_index = crate::baseobjspace::space_index(w_default)?;
                crate::baseobjspace::int_w(w_index)?
            } else {
                0
            };
            let n = crate::baseobjspace::length_hint(w_iterable, default)?;
            Ok(w_int_new(n))
        }),
    );
}
