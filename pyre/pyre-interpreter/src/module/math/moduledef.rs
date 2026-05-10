//! Math module definition — complete CPython parity via pymath crate.
//!
//! PyPy equivalent: pypy/module/math/moduledef.py

use super::interp_math;
use crate::{DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity};

pub fn init(ns: &mut DictStorage) {
    // ── Constants (PyPy: interp_math.State.__init__) ─────────────────
    dict_storage_store(
        ns,
        "e",
        pyre_object::floatobject::w_float_new(pymath::math::E),
    );
    dict_storage_store(
        ns,
        "pi",
        pyre_object::floatobject::w_float_new(pymath::math::PI),
    );
    dict_storage_store(
        ns,
        "tau",
        pyre_object::floatobject::w_float_new(pymath::math::TAU),
    );
    dict_storage_store(
        ns,
        "inf",
        pyre_object::floatobject::w_float_new(pymath::math::INF),
    );
    dict_storage_store(
        ns,
        "nan",
        pyre_object::floatobject::w_float_new(pymath::math::NAN),
    );

    // ── Trigonometric ────────────────────────────────────────────────
    dict_storage_store(ns, "sin", make_builtin_function_with_arity("sin", interp_math::sin, 1));
    dict_storage_store(ns, "cos", make_builtin_function_with_arity("cos", interp_math::cos, 1));
    dict_storage_store(ns, "tan", make_builtin_function_with_arity("tan", interp_math::tan, 1));
    dict_storage_store(ns, "asin", make_builtin_function_with_arity("asin", interp_math::asin, 1));
    dict_storage_store(ns, "acos", make_builtin_function_with_arity("acos", interp_math::acos, 1));
    dict_storage_store(ns, "atan", make_builtin_function_with_arity("atan", interp_math::atan, 1));
    dict_storage_store(
        ns,
        "atan2",
        make_builtin_function_with_arity("atan2", interp_math::atan2, 2),
    );
    dict_storage_store(ns, "sinh", make_builtin_function_with_arity("sinh", interp_math::sinh, 1));
    dict_storage_store(ns, "cosh", make_builtin_function_with_arity("cosh", interp_math::cosh, 1));
    dict_storage_store(ns, "tanh", make_builtin_function_with_arity("tanh", interp_math::tanh, 1));
    dict_storage_store(
        ns,
        "asinh",
        make_builtin_function_with_arity("asinh", interp_math::asinh, 1),
    );
    dict_storage_store(
        ns,
        "acosh",
        make_builtin_function_with_arity("acosh", interp_math::acosh, 1),
    );
    dict_storage_store(
        ns,
        "atanh",
        make_builtin_function_with_arity("atanh", interp_math::atanh, 1),
    );

    // ── Exponential / logarithmic ───────────────────────────────────
    dict_storage_store(ns, "sqrt", make_builtin_function_with_arity("sqrt", interp_math::sqrt, 1));
    dict_storage_store(ns, "cbrt", make_builtin_function_with_arity("cbrt", interp_math::cbrt, 1));
    dict_storage_store(ns, "exp", make_builtin_function_with_arity("exp", interp_math::exp, 1));
    dict_storage_store(ns, "exp2", make_builtin_function_with_arity("exp2", interp_math::exp2, 1));
    dict_storage_store(
        ns,
        "expm1",
        make_builtin_function_with_arity("expm1", interp_math::expm1, 1),
    );
    dict_storage_store(ns, "log", make_builtin_function("log", interp_math::log));
    dict_storage_store(ns, "log2", make_builtin_function_with_arity("log2", interp_math::log2, 1));
    dict_storage_store(
        ns,
        "log10",
        make_builtin_function_with_arity("log10", interp_math::log10, 1),
    );
    dict_storage_store(
        ns,
        "log1p",
        make_builtin_function_with_arity("log1p", interp_math::log1p, 1),
    );
    dict_storage_store(ns, "pow", make_builtin_function_with_arity("pow", interp_math::pow, 2));

    // ── Gamma / error ───────────────────────────────────────────────
    dict_storage_store(ns, "erf", make_builtin_function_with_arity("erf", interp_math::erf, 1));
    dict_storage_store(ns, "erfc", make_builtin_function_with_arity("erfc", interp_math::erfc, 1));
    dict_storage_store(
        ns,
        "gamma",
        make_builtin_function_with_arity("gamma", interp_math::gamma, 1),
    );
    dict_storage_store(
        ns,
        "lgamma",
        make_builtin_function_with_arity("lgamma", interp_math::lgamma, 1),
    );

    // ── Rounding / truncation ───────────────────────────────────────
    dict_storage_store(
        ns,
        "floor",
        make_builtin_function_with_arity("floor", interp_math::floor, 1),
    );
    dict_storage_store(ns, "ceil", make_builtin_function_with_arity("ceil", interp_math::ceil, 1));
    dict_storage_store(
        ns,
        "trunc",
        make_builtin_function_with_arity("trunc", interp_math::trunc, 1),
    );

    // ── Floating-point manipulation ─────────────────────────────────
    dict_storage_store(ns, "fabs", make_builtin_function_with_arity("fabs", interp_math::fabs, 1));
    dict_storage_store(ns, "fmod", make_builtin_function_with_arity("fmod", interp_math::fmod, 2));
    dict_storage_store(
        ns,
        "copysign",
        make_builtin_function_with_arity("copysign", interp_math::copysign, 2),
    );
    dict_storage_store(
        ns,
        "remainder",
        make_builtin_function_with_arity("remainder", interp_math::remainder, 2),
    );
    dict_storage_store(
        ns,
        "frexp",
        make_builtin_function_with_arity("frexp", interp_math::frexp, 1),
    );
    dict_storage_store(
        ns,
        "ldexp",
        make_builtin_function_with_arity("ldexp", interp_math::ldexp, 2),
    );
    dict_storage_store(ns, "modf", make_builtin_function_with_arity("modf", interp_math::modf, 1));
    dict_storage_store(
        ns,
        "nextafter",
        make_builtin_function("nextafter", interp_math::nextafter),
    );
    dict_storage_store(ns, "ulp", make_builtin_function_with_arity("ulp", interp_math::ulp, 1));
    dict_storage_store(ns, "fma", make_builtin_function_with_arity("fma", interp_math::fma, 3));

    // ── Classification ──────────────────────────────────────────────
    dict_storage_store(
        ns,
        "isinf",
        make_builtin_function_with_arity("isinf", interp_math::isinf, 1),
    );
    dict_storage_store(
        ns,
        "isnan",
        make_builtin_function_with_arity("isnan", interp_math::isnan, 1),
    );
    dict_storage_store(
        ns,
        "isfinite",
        make_builtin_function_with_arity("isfinite", interp_math::isfinite, 1),
    );
    dict_storage_store(
        ns,
        "isclose",
        make_builtin_function("isclose", interp_math::isclose),
    );

    // ── Conversion ──────────────────────────────────────────────────
    dict_storage_store(
        ns,
        "degrees",
        make_builtin_function_with_arity("degrees", interp_math::degrees, 1),
    );
    dict_storage_store(
        ns,
        "radians",
        make_builtin_function_with_arity("radians", interp_math::radians, 1),
    );

    // ── Multi-dimensional ───────────────────────────────────────────
    dict_storage_store(
        ns,
        "hypot",
        make_builtin_function("hypot", interp_math::hypot),
    );
    dict_storage_store(ns, "dist", make_builtin_function_with_arity("dist", interp_math::dist, 2));

    // ── Aggregation ─────────────────────────────────────────────────
    dict_storage_store(ns, "fsum", make_builtin_function_with_arity("fsum", interp_math::fsum, 1));
    dict_storage_store(ns, "prod", make_builtin_function("prod", interp_math::prod));
    dict_storage_store(
        ns,
        "sumprod",
        make_builtin_function_with_arity("sumprod", interp_math::sumprod, 2),
    );

    // ── Integer math ────────────────────────────────────────────────
    dict_storage_store(
        ns,
        "factorial",
        make_builtin_function_with_arity("factorial", interp_math::factorial, 1),
    );
    dict_storage_store(ns, "gcd", make_builtin_function("gcd", interp_math::gcd));
    dict_storage_store(ns, "lcm", make_builtin_function("lcm", interp_math::lcm));
    dict_storage_store(ns, "comb", make_builtin_function_with_arity("comb", interp_math::comb, 2));
    dict_storage_store(ns, "perm", make_builtin_function("perm", interp_math::perm));
    dict_storage_store(
        ns,
        "isqrt",
        make_builtin_function_with_arity("isqrt", interp_math::isqrt, 1),
    );
}
