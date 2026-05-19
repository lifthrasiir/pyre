//! pypy/objspace/std/formatting.py — printf-style string formatting.
#![allow(non_camel_case_types, non_snake_case)]

use crate::objspace::descroperation::{int_value, is_int_like};
use crate::{PyError, PyErrorKind, PyResult};
use pyre_object::*;

/// `str % args` — printf-style string formatting.
/// PyPy: unicodeobject.py mod__String_ANY → formatting.py
pub(crate) unsafe fn str_format_percent(fmt: PyObjectRef, args: PyObjectRef) -> PyResult {
    let fmt_str = w_str_get_value(fmt);
    // `formatting.py:39-46 StringFormatter.__init__` — when `args`
    // is a tuple, `values_w` is the unpacked positional list; for
    // any other shape (mapping, single value), `values_w` stays
    // None and `checkconsumed` is skipped.  Pyre tracks the same
    // distinction via `args_is_tuple` so the trailing surplus-args
    // check at the end fires only for tuple input — without it,
    // `"%s"`-only formats against a mapping that exposes
    // `__getitem__` would always trip "not all arguments converted"
    // even when the mapping was the (single) intended value.
    let args_is_tuple = is_tuple(args);
    let arg_list: Vec<PyObjectRef> = if args_is_tuple {
        let n = w_tuple_len(args);
        (0..n)
            .filter_map(|i| w_tuple_getitem(args, i as i64))
            .collect()
    } else {
        vec![args]
    };

    let mut result = String::new();
    let mut arg_idx = 0;
    let bytes = fmt_str.as_bytes();
    let mut i = 0;
    let take_next_arg =
        |arg_idx: &mut usize, arg_list: &[PyObjectRef]| -> Result<PyObjectRef, PyError> {
            // `formatting.py:574-582 nextinputvalue` — surfaces the
            // canonical "not enough arguments" message when the
            // positional pool runs dry.
            if *arg_idx >= arg_list.len() {
                return Err(PyError::type_error(
                    "not enough arguments for format string",
                ));
            }
            let a = arg_list[*arg_idx];
            *arg_idx += 1;
            Ok(a)
        };
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 1 < bytes.len() {
            i += 1;
            // Named format: %(name)s — `formatting.py:174-195
            // getmappingkey` scans up to the *balanced* closing
            // `)` (paren-depth counting so `%(a(b)c)s` parses key
            // `a(b)c`), then `space.getitem(args, w_key)` looks up
            // the mapping value (mapping-like object, not just
            // exact dict).
            let named_arg = if i < bytes.len() && bytes[i] == b'(' {
                i += 1; // skip the opening '('
                let key_start = i;
                let mut pcount: usize = 1;
                while i < bytes.len() {
                    let c = bytes[i];
                    if c == b')' {
                        pcount -= 1;
                        if pcount == 0 {
                            break;
                        }
                    } else if c == b'(' {
                        pcount += 1;
                    }
                    i += 1;
                }
                if i >= bytes.len() {
                    // `formatting.py:184-186 incomplete format key` —
                    // ran off the end of the format string without
                    // the matching `)`.
                    return Err(PyError::new(
                        PyErrorKind::ValueError,
                        "incomplete format key".to_string(),
                    ));
                }
                let key = String::from_utf8_lossy(&bytes[key_start..i]).into_owned();
                i += 1; // skip the closing ')'
                // Fast path for exact dict: avoid building a W_StrObject
                // when we can probe the dict storage directly.
                if is_dict(args) {
                    w_dict_getitem_str(args, &key)
                } else {
                    let w_key = pyre_object::w_str_new(&key);
                    Some(crate::baseobjspace::getitem(args, w_key)?)
                }
            } else {
                None
            };
            // `pypy/objspace/std/formatting.py StringFormatter._parse_spec`
            // — flags (`-`, `+`, ` `, `0`, `#`), width digits, optional
            // `.precision` digits, then conversion type.  pyre handles
            // the common subset; the asterisk (`*`) form is left to a
            // future round.
            let mut left_align = false;
            let mut zero_pad = false;
            let mut explicit_sign = false;
            let mut blank_sign = false;
            let mut alt_form = false;
            while i < bytes.len() {
                match bytes[i] {
                    b'-' => left_align = true,
                    b'0' => zero_pad = true,
                    b'+' => explicit_sign = true,
                    b' ' => blank_sign = true,
                    // `formatting.py:240-242` — `#` selects the
                    // alternate form (0x/0o/0b prefix on integers).
                    b'#' => alt_form = true,
                    _ => break,
                }
                i += 1;
            }
            // `formatting.py:266-274 peel_num` — `*` reads width /
            // precision from the next positional input.
            let mut width = 0usize;
            if i < bytes.len() && bytes[i] == b'*' {
                i += 1;
                let star_arg = take_next_arg(&mut arg_idx, &arg_list)?;
                if !is_int_like(star_arg) {
                    return Err(PyError::type_error("* wants int"));
                }
                let val = int_value(star_arg);
                if val < 0 {
                    // Negative star width acts like `-` flag with abs(width).
                    left_align = true;
                    width = (-val) as usize;
                } else {
                    width = val as usize;
                }
            } else {
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    width = width * 10 + (bytes[i] - b'0') as usize;
                    i += 1;
                }
            }
            let mut precision: Option<usize> = None;
            if i < bytes.len() && bytes[i] == b'.' {
                i += 1;
                if i < bytes.len() && bytes[i] == b'*' {
                    i += 1;
                    let star_arg = take_next_arg(&mut arg_idx, &arg_list)?;
                    if !is_int_like(star_arg) {
                        return Err(PyError::type_error("* wants int"));
                    }
                    let v = int_value(star_arg);
                    precision = Some(if v < 0 { 0 } else { v as usize });
                } else {
                    let mut p = 0usize;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        p = p * 10 + (bytes[i] - b'0') as usize;
                        i += 1;
                    }
                    precision = Some(p);
                }
            }
            if i >= bytes.len() {
                return Err(PyError::value_error("incomplete format"));
            }
            let spec = bytes[i] as char;
            i += 1;
            if spec == '%' {
                result.push('%');
                continue;
            }
            let arg = if let Some(na) = named_arg {
                na
            } else {
                take_next_arg(&mut arg_idx, &arg_list)?
            };
            // Helper closure: pad-and-emit body string respecting
            // width/left-align (zero-pad only matters for numeric paths
            // that build their own body with sign awareness).
            let pad = |body: String| -> String {
                if body.chars().count() >= width {
                    return body;
                }
                let need = width - body.chars().count();
                let mut s = String::with_capacity(width);
                if left_align {
                    s.push_str(&body);
                    for _ in 0..need {
                        s.push(' ');
                    }
                } else {
                    for _ in 0..need {
                        s.push(' ');
                    }
                    s.push_str(&body);
                }
                s
            };
            let sign_prefix = |val: i64| -> &'static str {
                if val < 0 {
                    "-"
                } else if explicit_sign {
                    "+"
                } else if blank_sign {
                    " "
                } else {
                    ""
                }
            };
            match spec {
                's' => {
                    let mut body = crate::py_str(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                'r' => {
                    let mut body = crate::py_repr(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                // `formatting.py fmt_a` (CPython `%a`) — repr() force-
                // ASCII-encoded.  Pyre's `py_repr` already produces
                // ASCII-clean output for the types it covers, so the
                // result matches PyPy `fmt_a` for the supported subset.
                'a' => {
                    let mut body = crate::py_repr(arg);
                    if let Some(p) = precision {
                        body = body.chars().take(p).collect();
                    }
                    result.push_str(&pad(body));
                }
                // `formatting.py:549-552 fmt_b` — `%b` is a *bytes-only*
                // conversion (formats a `bytes`/`bytearray`/`__bytes__`
                // object); for the unicode formatter PyPy calls
                // `self.unknown_fmtchar()` which raises ValueError
                // "unsupported format character 'b' (0x62) at index N".
                'b' => {
                    return Err(PyError::new(
                        PyErrorKind::ValueError,
                        format!("unsupported format character 'b' (0x62) at index {}", i - 1),
                    ));
                }
                // `formatting.py fmt_u` is documented as a deprecated
                // alias for `%d` and dispatches to the same body.
                'd' | 'i' | 'u' | 'x' | 'X' | 'o' => {
                    // `formatting.py:296-302 fmt_d / fmt_x ...` raises
                    // TypeError when the argument is not int-like
                    // (`%X format: an integer is required, not <type>`),
                    // not silently coerce via str().
                    if !is_int_like(arg) {
                        return Err(PyError::type_error(format!(
                            "%{spec} format: an integer is required, not {}",
                            crate::baseobjspace::object_functionstr_type_name(arg),
                        )));
                    }
                    let val = int_value(arg);
                    let abs = val.unsigned_abs();
                    let digits = match spec {
                        'x' => format!("{abs:x}"),
                        'X' => format!("{abs:X}"),
                        'o' => format!("{abs:o}"),
                        _ => format!("{abs}"),
                    };
                    // `formatting.py:240-242` — alt-form adds the
                    // base prefix (`0x`, `0X`, `0o`) for the matching
                    // radix; `%d/%i/%u` ignore it.  `%b` is not a
                    // unicode-formatter conversion (rejected above
                    // per `:549-552 fmt_b`).
                    let prefix = if alt_form {
                        match spec {
                            'x' => "0x",
                            'X' => "0X",
                            'o' => "0o",
                            _ => "",
                        }
                    } else {
                        ""
                    };
                    let sign = sign_prefix(val);
                    // Sign-aware zero pad (`%05d`, `%-5d` parity):
                    // `formatting.py:_pad_string` reserves width for
                    // sign + alt-prefix before padding zeros.
                    // Left-align disables zero pad.
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + prefix.len() + digits.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        s.push_str(prefix);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&digits);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{prefix}{digits}")));
                    }
                }
                // `formatting.py fmt_e / fmt_E / fmt_g / fmt_G` —
                // scientific / general float formatting.  Default
                // precision is 6 (CPython %e/%g), `%g` strips
                // trailing zeros and chooses scientific vs fixed
                // based on exponent magnitude (PyPy delegates to
                // `rfloat.formatd` with the matching format char).
                'e' | 'E' | 'g' | 'G' => {
                    // `formatting.py:303-308 fmt_e / fmt_g ...` raises
                    // TypeError when the argument is not numeric;
                    // silently substituting 0.0 for non-numeric input
                    // hid type bugs at the format site.
                    let val = if is_float(arg) {
                        pyre_object::floatobject::w_float_get_value(arg)
                    } else if is_int_like(arg) {
                        int_value(arg) as f64
                    } else {
                        return Err(PyError::type_error(format!(
                            "must be real number, not {}",
                            crate::baseobjspace::object_functionstr_type_name(arg),
                        )));
                    };
                    let prec = precision.unwrap_or(6);
                    let abs = val.abs();
                    let body = match spec {
                        'e' => normalise_exponent(&format!("{:.*e}", prec, abs), false),
                        'E' => normalise_exponent(&format!("{:.*E}", prec, abs), true),
                        // `%g` precision counts significant digits;
                        // Rust has no built-in formatter that exactly
                        // matches CPython's `%g` rules, so emit
                        // scientific when |v| >= 10^prec or < 1e-4
                        // (matching the CPython threshold), else
                        // fixed with (prec - 1 - exponent) decimals
                        // and trailing zero stripping.
                        'g' | 'G' => format_g_like(abs, prec, spec == 'G', alt_form),
                        _ => unreachable!(),
                    };
                    let sign = if val.is_sign_negative() && !val.is_nan() {
                        "-"
                    } else if explicit_sign {
                        "+"
                    } else if blank_sign {
                        " "
                    } else {
                        ""
                    };
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + body.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&body);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{body}")));
                    }
                }
                'f' | 'F' => {
                    // `formatting.py:303-308 fmt_f` — same TypeError as
                    // `%e/%g` for non-numeric arguments.
                    let val = if is_float(arg) {
                        pyre_object::floatobject::w_float_get_value(arg)
                    } else if is_int_like(arg) {
                        int_value(arg) as f64
                    } else {
                        return Err(PyError::type_error(format!(
                            "must be real number, not {}",
                            crate::baseobjspace::object_functionstr_type_name(arg),
                        )));
                    };
                    let prec = precision.unwrap_or(6);
                    let abs_body = format!("{:.*}", prec, val.abs());
                    let sign = if val.is_sign_negative() && !val.is_nan() {
                        "-"
                    } else if explicit_sign {
                        "+"
                    } else if blank_sign {
                        " "
                    } else {
                        ""
                    };
                    if zero_pad && !left_align {
                        let need = width.saturating_sub(sign.len() + abs_body.len());
                        let mut s = String::with_capacity(width);
                        s.push_str(sign);
                        for _ in 0..need {
                            s.push('0');
                        }
                        s.push_str(&abs_body);
                        result.push_str(&s);
                    } else {
                        result.push_str(&pad(format!("{sign}{abs_body}")));
                    }
                }
                'c' => {
                    // `formatting.py:283-294 fmt_c` parity:
                    //   - int branch: `if value < 0 or value > 0x10FFFF:
                    //     OverflowError("%c arg not in range(0x110000)")`
                    //   - str branch: `if len(s) != 1: TypeError(
                    //     "%c requires int or single character")`
                    //   - other types: same TypeError.
                    if is_int_like(arg) {
                        let v = int_value(arg);
                        if v < 0 || v > 0x10FFFF {
                            return Err(PyError::new(
                                PyErrorKind::OverflowError,
                                "%c arg not in range(0x110000)".to_string(),
                            ));
                        }
                        let c = char::from_u32(v as u32).ok_or_else(|| {
                            PyError::new(
                                PyErrorKind::OverflowError,
                                "%c arg not in range(0x110000)".to_string(),
                            )
                        })?;
                        result.push_str(&pad(c.to_string()));
                    } else if is_str(arg) {
                        let s = w_str_get_value(arg);
                        if s.chars().count() != 1 {
                            return Err(PyError::type_error("%c requires int or single character"));
                        }
                        result.push_str(&pad(s.to_string()));
                    } else {
                        return Err(PyError::type_error(format!(
                            "%c requires int or char, not {}",
                            crate::baseobjspace::object_functionstr_type_name(arg),
                        )));
                    }
                }
                // `formatting.py:328-329 unknown_fmtchar` raises
                // ValueError on any spec char outside FORMATTER_CHARS.
                _ => {
                    return Err(PyError::value_error(format!(
                        "unsupported format character '{spec}' (0x{:x}) at index {}",
                        spec as u32,
                        i.saturating_sub(1)
                    )));
                }
            }
        } else {
            let rest = std::str::from_utf8(&bytes[i..]).unwrap_or("\u{FFFD}");
            let ch = rest.chars().next().unwrap_or('\u{FFFD}');
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    // `formatting.py:572-580 checkconsumed` — surplus positional
    // arguments are an error only when the input came as a tuple
    // (PyPy's `values_w` invariant).  Mapping inputs (dict and any
    // other shape) skip this check because every replacement field
    // pulls from `args` by name, not by sequential index.
    if args_is_tuple && arg_idx < arg_list.len() {
        return Err(PyError::type_error(
            "not all arguments converted during string formatting",
        ));
    }
    Ok(w_str_new(&result))
}

/// `formatting.py fmt_g / fmt_G` parity helper — emits the `%g` body
/// for an *absolute* float value.  CPython's `%g` rules:
///   * precision ≤ 0 is treated as 1 (PyPy: `prec = max(prec, 1)`)
///   * choose scientific when `exp < -4` or `exp >= prec`
///     (`pypy/objspace/std/formatting.py:120 format_e_g_complex`)
///   * scientific body uses (`prec - 1`) decimals
///   * fixed body uses `(prec - 1 - exp)` decimals
///   * trailing zeros (and the trailing '.') are stripped unless
///     `#` (alt-form) was set; with alt-form the trailing zeros AND
///     the dangling '.' survive so the output advertises the full
///     requested precision.
pub(crate) fn format_g_like(abs: f64, prec: usize, upper: bool, alt_form: bool) -> String {
    if abs == 0.0 {
        return if alt_form {
            // `formatting.py:120-128` — alt-form `%#g` keeps `prec`
            // significant digits even for 0.0, so `"%#.4g" % 0`
            // renders as `"0.000"` (one digit before the dot, then
            // `prec - 1` zeros after).
            let after = prec.saturating_sub(1);
            if after == 0 {
                "0".to_string()
            } else {
                format!("0.{}", "0".repeat(after))
            }
        } else {
            "0".to_string()
        };
    }
    if !abs.is_finite() {
        return if upper {
            format!("{abs}").to_uppercase()
        } else {
            format!("{abs}")
        };
    }
    let prec = prec.max(1);
    let exp = abs.log10().floor() as i32;
    let use_sci = exp < -4 || exp >= prec as i32;
    let raw = if use_sci {
        let rust_exp = format!("{:.*e}", prec - 1, abs);
        normalise_exponent(&rust_exp, upper)
    } else {
        let dec = (prec as i32 - 1 - exp).max(0) as usize;
        format!("{abs:.dec$}")
    };
    if alt_form {
        // Alt-form preserves trailing zeros and the dangling '.';
        // ensure a '.' is present even when the natural body has
        // none (e.g. `"%#g" % 1` → `"1.00000"`).
        if use_sci {
            return raw;
        }
        return if raw.contains('.') {
            raw
        } else {
            // `prec - 1 - exp` was 0, so the body has no decimals;
            // re-render with the alt-form '.' suffix + trailing
            // zeros to advertise the full precision.
            let after = (prec as i32 - 1 - exp).max(0) as usize;
            if after == 0 {
                format!("{raw}.")
            } else {
                format!("{raw}.{}", "0".repeat(after))
            }
        };
    }
    // Strip trailing zeros from the mantissa (and a dangling '.').
    if use_sci {
        if let Some(epos) = raw.find(|c: char| c == 'e' || c == 'E') {
            let (mantissa, exp_part) = raw.split_at(epos);
            let trimmed = trim_trailing_zeros(mantissa);
            format!("{trimmed}{exp_part}")
        } else {
            raw
        }
    } else if raw.contains('.') {
        trim_trailing_zeros(&raw)
    } else {
        raw
    }
}

/// Convert Rust's `{:e}` / `{:E}` exponent encoding (no sign, minimal
/// width — `1.5e3` / `1.5e-3`) to PyPy / CPython `%e`-style
/// (`1.5e+03` / `1.5e-03` — explicit sign, exponent zero-padded to at
/// least two digits).  Mirrors `pypy/objspace/std/formatting.py
/// fmt_e` which delegates to `rfloat.formatd` with the C printf-style
/// padding rules.
pub(crate) fn normalise_exponent(raw: &str, upper: bool) -> String {
    let marker = if upper { 'E' } else { 'e' };
    let lower_marker = marker.to_ascii_lowercase();
    let upper_marker = marker.to_ascii_uppercase();
    let pos = match raw.find([lower_marker, upper_marker].as_ref()) {
        Some(p) => p,
        None => return raw.to_string(),
    };
    let (mantissa, exp_part) = raw.split_at(pos);
    let exp_str = &exp_part[1..]; // skip the marker
    let (sign_char, digits) = if let Some(rest) = exp_str.strip_prefix('-') {
        ('-', rest)
    } else if let Some(rest) = exp_str.strip_prefix('+') {
        ('+', rest)
    } else {
        ('+', exp_str)
    };
    let padded = if digits.len() < 2 {
        format!("0{digits}")
    } else {
        digits.to_string()
    };
    format!("{mantissa}{marker}{sign_char}{padded}")
}

fn trim_trailing_zeros(body: &str) -> String {
    if !body.contains('.') {
        return body.to_string();
    }
    let trimmed = body.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}
