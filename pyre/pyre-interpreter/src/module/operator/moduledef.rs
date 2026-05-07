//! operator module definition.
//!
//! PyPy equivalent: pypy/module/operator/

use crate::{DictStorage, dict_storage_store, make_builtin_function};
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
    dict_storage_store(ns, "index", make_builtin_function("index", op_index));
    dict_storage_store(ns, "add", make_builtin_function("add", op_add));
    dict_storage_store(ns, "sub", make_builtin_function("sub", op_sub));
    dict_storage_store(ns, "mul", make_builtin_function("mul", op_mul));
    dict_storage_store(
        ns,
        "truediv",
        make_builtin_function("truediv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::truediv(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "floordiv",
        make_builtin_function("floordiv", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::floordiv(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "mod",
        make_builtin_function("mod", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::mod_(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "pow",
        make_builtin_function("pow", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::pow(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "neg",
        make_builtin_function("neg", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::neg(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "pos",
        make_builtin_function("pos", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::pos(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "abs",
        make_builtin_function("abs", |args| {
            assert!(args.len() == 1);
            crate::builtins::builtin_abs(args)
        }),
    );
    dict_storage_store(
        ns,
        "invert",
        make_builtin_function("invert", |args| {
            assert!(args.len() == 1);
            crate::baseobjspace::invert(args[0])
        }),
    );
    dict_storage_store(
        ns,
        "lshift",
        make_builtin_function("lshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::lshift(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "rshift",
        make_builtin_function("rshift", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::rshift(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "and_",
        make_builtin_function("and_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::and_(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "or_",
        make_builtin_function("or_", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::or_(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "xor",
        make_builtin_function("xor", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::xor(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "not_",
        make_builtin_function("not_", |args| {
            assert!(args.len() == 1);
            Ok(w_bool_from(!crate::baseobjspace::is_true(args[0])))
        }),
    );
    // interp_operator.py:138
    dict_storage_store(
        ns,
        "truth",
        make_builtin_function("truth", |args| {
            assert!(args.len() == 1);
            Ok(w_bool_from(crate::baseobjspace::is_true(args[0])))
        }),
    );
    dict_storage_store(
        ns,
        "is_",
        make_builtin_function("is_", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(std::ptr::eq(args[0], args[1])))
        }),
    );
    dict_storage_store(
        ns,
        "is_not",
        make_builtin_function("is_not", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(!std::ptr::eq(args[0], args[1])))
        }),
    );
    dict_storage_store(
        ns,
        "contains",
        make_builtin_function("contains", |args| {
            assert!(args.len() == 2);
            Ok(w_bool_from(crate::baseobjspace::contains(
                args[0], args[1],
            )?))
        }),
    );
    dict_storage_store(
        ns,
        "getitem",
        make_builtin_function("getitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::getitem(args[0], args[1])
        }),
    );
    dict_storage_store(
        ns,
        "setitem",
        make_builtin_function("setitem", |args| {
            assert!(args.len() == 3);
            crate::baseobjspace::setitem(args[0], args[1], args[2])?;
            Ok(w_none())
        }),
    );
    dict_storage_store(
        ns,
        "delitem",
        make_builtin_function("delitem", |args| {
            assert!(args.len() == 2);
            crate::baseobjspace::delitem(args[0], args[1])?;
            Ok(w_none())
        }),
    );
    // Underscore aliases (CPython: __add__/__sub__/... via operator module).
    dict_storage_store(ns, "__add__", make_builtin_function("__add__", op_add));
    dict_storage_store(ns, "__sub__", make_builtin_function("__sub__", op_sub));
    dict_storage_store(ns, "__mul__", make_builtin_function("__mul__", op_mul));
    dict_storage_store(ns, "eq", make_builtin_function("eq", op_eq));
    dict_storage_store(ns, "lt", make_builtin_function("lt", op_lt));
    dict_storage_store(ns, "gt", make_builtin_function("gt", op_gt));
    dict_storage_store(
        ns,
        "le",
        make_builtin_function("le", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Le)
        }),
    );
    dict_storage_store(
        ns,
        "ge",
        make_builtin_function("ge", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ge)
        }),
    );
    dict_storage_store(
        ns,
        "ne",
        make_builtin_function("ne", |args| {
            crate::baseobjspace::compare(args[0], args[1], crate::baseobjspace::CompareOp::Ne)
        }),
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
