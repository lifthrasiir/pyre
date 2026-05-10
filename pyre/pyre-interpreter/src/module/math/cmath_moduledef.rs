//! cmath module — complex math functions via pymath::cmath.
//!
//! PyPy equivalent: pypy/module/cmath/
//!
//! Stub: complex number type not yet implemented in pyre.
//! Functions are registered so `import cmath` succeeds, but complex
//! arithmetic requires W_ComplexObject (future work).

use crate::{DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity};
use pyre_object::*;

pub fn init(ns: &mut DictStorage) {
    // Constants
    dict_storage_store(ns, "pi", floatobject::w_float_new(pymath::math::PI));
    dict_storage_store(ns, "e", floatobject::w_float_new(pymath::math::E));
    dict_storage_store(ns, "tau", floatobject::w_float_new(pymath::math::TAU));
    dict_storage_store(ns, "inf", floatobject::w_float_new(pymath::math::INF));
    dict_storage_store(ns, "nan", floatobject::w_float_new(pymath::math::NAN));
    // infj, nanj would need complex type

    // Real-valued functions (work on float, stub for complex)
    dict_storage_store(
        ns,
        "phase",
        make_builtin_function_with_arity("phase", |args| {
            Ok(floatobject::w_float_new(
                super::interp_math::get_double(args[0]).atan2(0.0),
            ))
        }, 1),
    );
    dict_storage_store(
        ns,
        "polar",
        make_builtin_function_with_arity("polar", |args| {
            let x = super::interp_math::get_double(args[0]);
            Ok(w_tuple_new(vec![
                floatobject::w_float_new(x.abs()),
                floatobject::w_float_new(0.0),
            ]))
        }, 1),
    );
    dict_storage_store(
        ns,
        "rect",
        make_builtin_function_with_arity("rect", |args| {
            let r = super::interp_math::get_double(args[0]);
            let phi = super::interp_math::get_double(args[1]);
            Ok(floatobject::w_float_new(r * phi.cos()))
        }, 2),
    );
    dict_storage_store(
        ns,
        "isfinite",
        make_builtin_function_with_arity("isfinite", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_finite(),
            ))
        }, 1),
    );
    dict_storage_store(
        ns,
        "isinf",
        make_builtin_function_with_arity("isinf", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_infinite(),
            ))
        }, 1),
    );
    dict_storage_store(
        ns,
        "isnan",
        make_builtin_function_with_arity("isnan", |args| {
            Ok(w_bool_from(
                super::interp_math::get_double(args[0]).is_nan(),
            ))
        }, 1),
    );

    // Forward trig/exp functions to math equivalents for real input
    for (name, func) in [
        (
            "sqrt",
            super::interp_math::sqrt as fn(&[PyObjectRef]) -> Result<PyObjectRef, crate::PyError>,
        ),
        ("exp", super::interp_math::exp),
        ("log10", super::interp_math::log10),
        ("sin", super::interp_math::sin),
        ("cos", super::interp_math::cos),
        ("tan", super::interp_math::tan),
        ("asin", super::interp_math::asin),
        ("acos", super::interp_math::acos),
        ("atan", super::interp_math::atan),
        ("sinh", super::interp_math::sinh),
        ("cosh", super::interp_math::cosh),
        ("tanh", super::interp_math::tanh),
        ("asinh", super::interp_math::asinh),
        ("acosh", super::interp_math::acosh),
        ("atanh", super::interp_math::atanh),
    ] {
        dict_storage_store(ns, name, make_builtin_function_with_arity(name, func, 1));
    }
    // log accepts optional base argument — variable arity
    dict_storage_store(ns, "log", make_builtin_function("log", super::interp_math::log));
}
