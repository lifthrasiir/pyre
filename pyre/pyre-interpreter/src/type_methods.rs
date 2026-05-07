//! Builtin type method implementations.
//!
//! PyPy equivalents:
//!   pypy/objspace/std/listobject.py  (list methods)
//!   pypy/objspace/std/unicodeobject.py  (str methods)
//!   pypy/objspace/std/dictobject.py  (dict methods)
//!   pypy/objspace/std/tupleobject.py  (tuple methods)
//!
//! Separated from space.rs to avoid bloating the hot-path compilation
//! unit. Method functions are registered into TypeDef at startup.

use pyre_object::*;

// ── List methods ─────────────────────────────────────────────────────
// All take self (list) as first arg.

pub fn list_method_append(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "append() takes exactly one argument");
    unsafe { w_list_append(args[0], args[1]) };
    Ok(w_none())
}

pub fn list_method_extend(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    let list = args[0];
    let other = args[1];
    unsafe {
        if is_list(other) {
            let n = w_list_len(other);
            for i in 0..n {
                if let Some(item) = w_list_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        } else if is_tuple(other) {
            let n = w_tuple_len(other);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(other, i as i64) {
                    w_list_append(list, item);
                }
            }
        }
    }
    Ok(w_none())
}

/// PyPy: listobject.py descr_insert — list.insert(index, item)
pub fn list_method_insert(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 3, "insert() takes exactly 2 arguments");
    let index = unsafe { w_int_get_value(args[1]) };
    unsafe { pyre_object::listobject::w_list_insert(args[0], index, args[2]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_pop — list.pop([index])
pub fn list_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "pop() requires self");
    let index = if args.len() > 1 {
        unsafe { w_int_get_value(args[1]) }
    } else {
        -1 // default: pop last
    };
    unsafe {
        Ok(pyre_object::listobject::w_list_pop(args[0], index)
            .unwrap_or_else(|| panic!("pop from empty list")))
    }
}

/// PyPy: listobject.py descr_clear — list.clear()
pub fn list_method_clear(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    unsafe { pyre_object::listobject::w_list_clear(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_copy — list.copy()
pub fn list_method_copy(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let list = args[0];
    unsafe {
        let n = w_list_len(list);
        let mut items = Vec::with_capacity(n);
        for i in 0..n {
            if let Some(item) = w_list_getitem(list, i as i64) {
                items.push(item);
            }
        }
        Ok(w_list_new(items))
    }
}

/// PyPy: listobject.py descr_reverse — list.reverse()
pub fn list_method_reverse(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    unsafe { pyre_object::listobject::w_list_reverse(args[0]) };
    Ok(w_none())
}

/// PyPy: listobject.py descr_sort — list.sort()
///
/// Simplified: only sorts int lists. Full sort requires comparison protocol.
pub fn list_method_sort(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let list = args[0];
    unsafe {
        let n = w_list_len(list);
        let mut items: Vec<PyObjectRef> = (0..n)
            .filter_map(|i| w_list_getitem(list, i as i64))
            .collect();
        // Sort by int value (PyPy uses timsort with key/cmp)
        items.sort_by(|a, b| {
            if is_int(*a) && is_int(*b) {
                w_int_get_value(*a).cmp(&w_int_get_value(*b))
            } else {
                std::cmp::Ordering::Equal
            }
        });
        pyre_object::listobject::w_list_clear(list);
        for item in items {
            w_list_append(list, item);
        }
    }
    Ok(w_none())
}

/// listobject.py:795 `descr_index` — list.index(value[, start[, stop]]).
pub fn list_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "index() takes at least 1 argument");
    let list = args[0];
    let value = args[1];
    // listobject.py:799 unwrap_spec defaults: w_start=0 / w_stop=sys.maxint.
    // listobject.py:803 unwrap_start_stop handles negative normalization,
    // __index__ coercion and TypeError for non-index arguments.
    let size = unsafe { pyre_object::w_list_len(list) } as i64;
    let w_start = if args.len() >= 3 {
        args[2]
    } else {
        w_int_new(0)
    };
    let w_stop = if args.len() >= 4 {
        args[3]
    } else {
        w_int_new(i64::MAX)
    };
    let (start, stop) = crate::sliceobject::unwrap_start_stop(size, w_start, w_stop)?;
    match crate::listobject::w_list_find_or_count(list, value, start, stop, false)? {
        crate::listobject::FindOrCountResult::Index(i) => Ok(w_int_new(i)),
        crate::listobject::FindOrCountResult::NotFound => Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            // listobject.py:805 `oefmt(space.w_ValueError, "%R is not in list", w_value)`
            format!("{} is not in list", unsafe {
                crate::display::py_repr(value)
            }),
        )),
        crate::listobject::FindOrCountResult::Count(_) => {
            unreachable!("find_or_count with count=false never returns Count")
        }
    }
}

/// listobject.py:744 `descr_count` — list.count(value)
pub fn list_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "count() takes exactly 1 argument");
    let list = args[0];
    let value = args[1];
    match crate::listobject::w_list_find_or_count(list, value, 0, i64::MAX, true)? {
        crate::listobject::FindOrCountResult::Count(n) => Ok(w_int_new(n)),
        crate::listobject::FindOrCountResult::NotFound => Ok(w_int_new(0)),
        crate::listobject::FindOrCountResult::Index(_) => {
            unreachable!("find_or_count with count=true never returns Index")
        }
    }
}

/// listobject.py:782 `descr_remove` — list.remove(value).
pub fn list_method_remove(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "remove() takes exactly 1 argument");
    crate::listobject::w_list_remove(args[0], args[1])?;
    Ok(w_none())
}

// ── String methods ───────────────────────────────────────────────────

pub fn str_method_join(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2);
    let sep = unsafe { w_str_get_value(args[0]) };
    let iterable = args[1];
    let mut parts = Vec::new();
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    if is_str(item) {
                        parts.push(w_str_get_value(item).to_string());
                    }
                }
            }
        } else if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    if is_str(item) {
                        parts.push(w_str_get_value(item).to_string());
                    }
                }
            }
        }
    }
    Ok(w_str_new(&parts.join(sep)))
}

pub fn str_method_split(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let sep = if args.len() > 1 && !args[1].is_null() && unsafe { !is_none(args[1]) } {
        Some(unsafe { w_str_get_value(args[1]) })
    } else {
        None
    };
    let parts: Vec<PyObjectRef> = match sep {
        Some(sep) => s.split(sep).map(|p| w_str_new(p)).collect(),
        None => s.split_whitespace().map(|p| w_str_new(p)).collect(),
    };
    Ok(w_list_new(parts))
}

pub fn str_method_strip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    Ok(w_str_new(unsafe { w_str_get_value(args[0]) }.trim()))
}

pub fn str_method_lstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    Ok(w_str_new(unsafe { w_str_get_value(args[0]) }.trim_start()))
}

pub fn str_method_rstrip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    Ok(w_str_new(unsafe { w_str_get_value(args[0]) }.trim_end()))
}

pub fn str_method_startswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let prefix = unsafe { w_str_get_value(args[1]) };
    Ok(w_bool_from(s.starts_with(prefix)))
}

pub fn str_method_endswith(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let suffix = unsafe { w_str_get_value(args[1]) };
    Ok(w_bool_from(s.ends_with(suffix)))
}

pub fn str_method_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 3);
    let s = unsafe { w_str_get_value(args[0]) };
    let old = unsafe { w_str_get_value(args[1]) };
    let new = unsafe { w_str_get_value(args[2]) };
    Ok(w_str_new(&s.replace(old, new)))
}

pub fn str_method_find(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    Ok(w_int_new(s.find(sub).map(|i| i as i64).unwrap_or(-1)))
}

pub fn str_method_rfind(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    Ok(w_int_new(s.rfind(sub).map(|i| i as i64).unwrap_or(-1)))
}

pub fn str_method_upper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    Ok(w_str_new(
        &unsafe { w_str_get_value(args[0]) }.to_uppercase(),
    ))
}

pub fn str_method_lower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    Ok(w_str_new(
        &unsafe { w_str_get_value(args[0]) }.to_lowercase(),
    ))
}

/// PyPy: unicodeobject.py descr_format
/// Requires format spec parser — correct for no-arg case only.
/// `str.format(*args)` — PyPy: unicodeobject.py descr_format → newformat.py
pub fn str_method_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let fmt = unsafe { pyre_object::w_str_get_value(args[0]) };
    let fargs = &args[1..];

    let mut result = String::new();
    let mut auto_idx = 0usize;
    let bytes = fmt.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'{' {
                result.push('{');
                i += 2;
                continue;
            }
            i += 1;
            // Parse field: {}, {0}, {name}, {0:spec}
            let mut field = String::new();
            let mut spec = String::new();
            let mut in_spec = false;
            while i < bytes.len() && bytes[i] != b'}' {
                if bytes[i] == b':' && !in_spec {
                    in_spec = true;
                    i += 1;
                    continue;
                }
                if in_spec {
                    spec.push(bytes[i] as char);
                } else {
                    field.push(bytes[i] as char);
                }
                i += 1;
            }
            if i < bytes.len() {
                i += 1;
            } // skip '}'

            let idx = if field.is_empty() {
                let idx = auto_idx;
                auto_idx += 1;
                idx
            } else {
                field.parse::<usize>().unwrap_or(auto_idx)
            };
            let val = fargs.get(idx).copied().unwrap_or(pyre_object::w_none());
            let formatted = if spec.is_empty() {
                unsafe { crate::py_str(val) }
            } else {
                format_with_spec(val, &spec)
            };
            result.push_str(&formatted);
        } else if bytes[i] == b'}' && i + 1 < bytes.len() && bytes[i + 1] == b'}' {
            result.push('}');
            i += 2;
        } else {
            result.push(bytes[i] as char);
            i += 1;
        }
    }
    Ok(pyre_object::w_str_new(&result))
}

fn format_with_spec(val: PyObjectRef, spec: &str) -> String {
    unsafe {
        if pyre_object::is_int(val) {
            let v = pyre_object::w_int_get_value(val);
            if spec.ends_with('d') || spec.is_empty() {
                return format!("{v}");
            }
            if spec.ends_with('x') {
                return format!("{v:x}");
            }
            if spec.ends_with('o') {
                return format!("{v:o}");
            }
            if spec.ends_with('b') {
                return format!("{v:b}");
            }
            // Width/fill: e.g. "02" → zero-padded width 2
            if let Some(width) = spec
                .strip_suffix('d')
                .or(Some(spec))
                .and_then(|s| s.parse::<usize>().ok())
            {
                if spec.starts_with('0') {
                    return format!("{v:0>width$}");
                }
                return format!("{v:>width$}");
            }
            return format!("{v}");
        }
        if pyre_object::is_float(val) {
            let v = pyre_object::floatobject::w_float_get_value(val);
            if let Some(rest) = spec.strip_suffix('f') {
                let prec: usize = rest.trim_start_matches('.').parse().unwrap_or(6);
                return format!("{v:.prec$}");
            }
            return format!("{v}");
        }
        crate::py_str(val)
    }
}

/// PyPy: unicodeobject.py descr_encode → encode_object.
/// For the common 'utf-8' / 'ascii' fast paths, returns the UTF-8 bytes
/// of the string. Other codecs fall through to a best-effort UTF-8 encoding.
pub fn str_method_encode(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    // encoding arg (optional, default utf-8)
    let encoding: String = if args.len() >= 2 {
        unsafe {
            if pyre_object::is_str(args[1]) {
                w_str_get_value(args[1]).to_string()
            } else {
                "utf-8".to_string()
            }
        }
    } else {
        "utf-8".to_string()
    };
    let enc_lower = encoding.to_ascii_lowercase().replace('_', "-");
    match enc_lower.as_str() {
        "utf-8" | "utf8" | "u8" => Ok(pyre_object::w_bytes_from_bytes(s.as_bytes())),
        "ascii" | "us-ascii" | "646" => {
            if s.is_ascii() {
                Ok(pyre_object::w_bytes_from_bytes(s.as_bytes()))
            } else {
                Err(crate::PyError::value_error(
                    "'ascii' codec can't encode character: ordinal not in range(128)",
                ))
            }
        }
        "latin-1" | "latin1" | "iso-8859-1" | "8859" => {
            let mut out = Vec::with_capacity(s.len());
            for ch in s.chars() {
                if (ch as u32) > 0xFF {
                    return Err(crate::PyError::value_error(
                        "'latin-1' codec can't encode character: ordinal not in range(256)",
                    ));
                }
                out.push(ch as u8);
            }
            Ok(pyre_object::w_bytes_from_bytes(&out))
        }
        _ => Ok(pyre_object::w_bytes_from_bytes(s.as_bytes())),
    }
}

pub fn str_method_isdigit(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()),
    ))
}

pub fn str_method_isdecimal(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_ascii_digit() || c.is_numeric()),
    ))
}

pub fn str_method_isnumeric(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_numeric()),
    ))
}

pub fn str_method_istitle(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let mut cased = false;
    let mut prev_cased = false;
    for c in s.chars() {
        if c.is_uppercase() {
            if prev_cased {
                return Ok(w_bool_from(false));
            }
            prev_cased = true;
            cased = true;
        } else if c.is_lowercase() {
            if !prev_cased {
                return Ok(w_bool_from(false));
            }
            prev_cased = true;
            cased = true;
        } else {
            prev_cased = false;
        }
    }
    Ok(w_bool_from(cased))
}

pub fn str_method_isalpha(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_alphabetic()),
    ))
}

/// PyPy: unicodeobject.py descr_isidentifier
pub fn str_method_isidentifier(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(is_identifier(s)))
}

/// Check if a string is a valid Python identifier.
/// Python 3 identifiers: ID_Start ID_Continue*
/// Simplified: accepts ASCII identifiers + Unicode letters/digits.
fn is_identifier(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !first.is_alphabetic() && first != '_' {
        return false;
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

pub fn str_method_zfill(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) } as usize;
    if s.len() >= width {
        return Ok(args[0]);
    }
    let padding = "0".repeat(width - s.len());
    Ok(w_str_new(&format!("{padding}{s}")))
}

/// PyPy: unicodeobject.py descr_count
pub fn str_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    Ok(w_int_new(s.matches(sub).count() as i64))
}

/// PyPy: unicodeobject.py descr_index
pub fn str_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sub = unsafe { w_str_get_value(args[1]) };
    match s.find(sub) {
        Some(i) => Ok(w_int_new(i as i64)),
        None => panic!("ValueError: substring not found"),
    }
}

/// PyPy: unicodeobject.py descr_title
pub fn str_method_title(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let mut result = String::with_capacity(s.len());
    let mut prev_is_sep = true;
    for c in s.chars() {
        if prev_is_sep {
            for u in c.to_uppercase() {
                result.push(u);
            }
        } else {
            for l in c.to_lowercase() {
                result.push(l);
            }
        }
        prev_is_sep = !c.is_alphanumeric();
    }
    Ok(w_str_new(&result))
}

/// PyPy: unicodeobject.py descr_capitalize
pub fn str_method_capitalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let mut chars = s.chars();
    let result = match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            let lower: String = chars.flat_map(|c| c.to_lowercase()).collect();
            format!("{upper}{lower}")
        }
    };
    Ok(w_str_new(&result))
}

/// PyPy: unicodeobject.py descr_swapcase
pub fn str_method_swapcase(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let result: String = s
        .chars()
        .flat_map(|c| {
            if c.is_uppercase() {
                c.to_lowercase().collect::<Vec<_>>()
            } else {
                c.to_uppercase().collect::<Vec<_>>()
            }
        })
        .collect();
    Ok(w_str_new(&result))
}

/// PyPy: unicodeobject.py descr_center
pub fn str_method_center(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) } as usize;
    let fillchar = if args.len() > 2 {
        unsafe { w_str_get_value(args[2]) }
            .chars()
            .next()
            .unwrap_or(' ')
    } else {
        ' '
    };
    if s.len() >= width {
        return Ok(args[0]);
    }
    let total_pad = width - s.len();
    let left = total_pad / 2;
    let right = total_pad - left;
    let result = format!(
        "{}{}{}",
        fillchar.to_string().repeat(left),
        s,
        fillchar.to_string().repeat(right)
    );
    Ok(w_str_new(&result))
}

/// PyPy: unicodeobject.py descr_ljust
pub fn str_method_ljust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) } as usize;
    let fillchar = if args.len() > 2 {
        unsafe { w_str_get_value(args[2]) }
            .chars()
            .next()
            .unwrap_or(' ')
    } else {
        ' '
    };
    if s.len() >= width {
        return Ok(args[0]);
    }
    let pad = fillchar.to_string().repeat(width - s.len());
    Ok(w_str_new(&format!("{s}{pad}")))
}

/// PyPy: unicodeobject.py descr_rjust
pub fn str_method_rjust(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let width = unsafe { w_int_get_value(args[1]) } as usize;
    let fillchar = if args.len() > 2 {
        unsafe { w_str_get_value(args[2]) }
            .chars()
            .next()
            .unwrap_or(' ')
    } else {
        ' '
    };
    if s.len() >= width {
        return Ok(args[0]);
    }
    let pad = fillchar.to_string().repeat(width - s.len());
    Ok(w_str_new(&format!("{pad}{s}")))
}

/// PyPy: unicodeobject.py descr_isspace
pub fn str_method_isspace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_whitespace()),
    ))
}

/// PyPy: unicodeobject.py descr_isupper
pub fn str_method_isupper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let has_cased = s.chars().any(|c| c.is_alphabetic());
    Ok(w_bool_from(
        has_cased
            && s.chars()
                .filter(|c| c.is_alphabetic())
                .all(|c| c.is_uppercase()),
    ))
}

/// PyPy: unicodeobject.py descr_islower
pub fn str_method_islower(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let has_cased = s.chars().any(|c| c.is_alphabetic());
    Ok(w_bool_from(
        has_cased
            && s.chars()
                .filter(|c| c.is_alphabetic())
                .all(|c| c.is_lowercase()),
    ))
}

/// PyPy: unicodeobject.py descr_isalnum
pub fn str_method_isalnum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(
        !s.is_empty() && s.chars().all(|c| c.is_alphanumeric()),
    ))
}

/// PyPy: unicodeobject.py descr_isascii
pub fn str_method_isascii(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    Ok(w_bool_from(s.is_ascii()))
}

/// PyPy: unicodeobject.py descr_partition
pub fn str_method_partition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sep = unsafe { w_str_get_value(args[1]) };
    match s.find(sep) {
        Some(i) => Ok(w_tuple_new(vec![
            w_str_new(&s[..i]),
            w_str_new(sep),
            w_str_new(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![args[0], w_str_new(""), w_str_new("")])),
    }
}

/// PyPy: unicodeobject.py descr_rpartition
pub fn str_method_rpartition(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let sep = unsafe { w_str_get_value(args[1]) };
    match s.rfind(sep) {
        Some(i) => Ok(w_tuple_new(vec![
            w_str_new(&s[..i]),
            w_str_new(sep),
            w_str_new(&s[i + sep.len()..]),
        ])),
        None => Ok(w_tuple_new(vec![w_str_new(""), w_str_new(""), args[0]])),
    }
}

/// PyPy: unicodeobject.py descr_splitlines
pub fn str_method_splitlines(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let parts: Vec<PyObjectRef> = s.lines().map(|line| w_str_new(line)).collect();
    Ok(w_list_new(parts))
}

/// PyPy: unicodeobject.py descr_removeprefix (Python 3.9+)
pub fn str_method_removeprefix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let prefix = unsafe { w_str_get_value(args[1]) };
    if s.starts_with(prefix) {
        Ok(w_str_new(&s[prefix.len()..]))
    } else {
        Ok(args[0])
    }
}

/// PyPy: unicodeobject.py descr_removesuffix (Python 3.9+)
pub fn str_method_removesuffix(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let s = unsafe { w_str_get_value(args[0]) };
    let suffix = unsafe { w_str_get_value(args[1]) };
    if s.ends_with(suffix) {
        Ok(w_str_new(&s[..s.len() - suffix.len()]))
    } else {
        Ok(args[0])
    }
}

/// PyPy: unicodeobject.py descr_expandtabs
pub fn str_method_expandtabs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let s = unsafe { w_str_get_value(args[0]) };
    let tabsize = if args.len() > 1 {
        (unsafe { w_int_get_value(args[1]) }) as usize
    } else {
        8
    };
    let result = s.replace('\t', &" ".repeat(tabsize));
    Ok(w_str_new(&result))
}

/// PyPy: unicodeobject.py descr_translate
///
/// str.translate(table) — table is a dict mapping ordinals (int) to
/// ordinals (int), strings (str), or None (delete).
pub fn str_method_translate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "translate() takes exactly one argument");
    let s = unsafe { w_str_get_value(args[0]) };
    let table = args[1];
    let mut result = String::with_capacity(s.len());
    unsafe {
        for ch in s.chars() {
            let key = w_int_new(ch as i64);
            if let Some(val) = w_dict_lookup(table, key) {
                if is_none(val) {
                    // None → delete character
                } else if is_int(val) {
                    if let Some(c) = char::from_u32(w_int_get_value(val) as u32) {
                        result.push(c);
                    }
                } else if is_str(val) {
                    result.push_str(w_str_get_value(val));
                } else {
                    result.push(ch);
                }
            } else {
                result.push(ch);
            }
        }
    }
    Ok(w_str_new(&result))
}

// ── Dict methods ─────────────────────────────────────────────────────

/// Resolve the actual backing W_DictObject for either a plain dict or
/// a dict subclass instance (which stores data in `__dict_data__`).
///
/// PyPy: W_DictMultiObject subclass instances ARE dicts, so no indirection
/// is needed. In pyre, dict subclass instances are W_InstanceObject with a
/// backing dict stored as an attribute.
pub fn resolve_dict_backing(obj: PyObjectRef) -> PyObjectRef {
    unsafe {
        if is_dict(obj) {
            return obj;
        }
        if is_instance(obj) {
            if let Ok(backing) = crate::baseobjspace::getattr(obj, "__dict_data__") {
                if is_dict(backing) {
                    return backing;
                }
            }
        }
    }
    pyre_object::PY_NULL
}

pub fn dict_method_get(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if dict.is_null() {
        return Ok(default);
    }
    unsafe { Ok(w_dict_lookup(dict, key).unwrap_or(default)) }
}

/// PyPy: dictobject.py descr_keys — returns dict_keys view.
///
/// Routes through `w_dict_items` so the storage-proxy union view
/// (`Module.__dict__.keys()` reaching every binding the storage
/// owns, not just the entries Vec) matches PyPy's single
/// W_DictMultiObject.  The back-mirror wired in
/// `bind_module_back_mirror` keeps entries Vec in sync with
/// storage on writes/deletes; the union here is the safety net for
/// dicts whose proxy was set without a back-mirror (rare but
/// possible for transient `globals()` views).
pub fn dict_method_keys(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(w_list_new(vec![]));
    }
    unsafe {
        let keys: Vec<PyObjectRef> = pyre_object::w_dict_items(dict)
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        Ok(w_list_new(keys))
    }
}

/// PyPy: dictobject.py descr_values
pub fn dict_method_values(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(w_list_new(vec![]));
    }
    unsafe {
        let values: Vec<PyObjectRef> = pyre_object::w_dict_items(dict)
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        Ok(w_list_new(values))
    }
}

/// PyPy: dictobject.py descr_items
pub fn dict_method_items(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty());
    let dict = resolve_dict_backing(args[0]);
    if dict.is_null() {
        return Ok(w_list_new(vec![]));
    }
    unsafe {
        let items: Vec<PyObjectRef> = pyre_object::w_dict_items(dict)
            .into_iter()
            .map(|(k, v)| w_tuple_new(vec![k, v]))
            .collect();
        Ok(w_list_new(items))
    }
}

/// PyPy: dictobject.py descr_update — dict.update(other)
pub fn dict_method_update(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "update() takes at least 1 argument");
    let dict = resolve_dict_backing(args[0]);
    let other_raw = resolve_dict_backing(args[1]);
    if dict.is_null() {
        return Ok(w_none());
    }
    if !other_raw.is_null() {
        unsafe {
            // Walk the union view so updates from a storage-proxied
            // source dict (e.g. `module.__dict__`) include entries
            // that live only in the proxy storage.
            for (k, v) in pyre_object::w_dict_items(other_raw) {
                w_dict_store(dict, k, v);
            }
        }
    }
    // globals() proxy: sync dict entries to the backing DictStorage.
    dict_sync_dict_storage_proxy(dict);
    Ok(w_none())
}

/// If dict has a dict_storage_proxy (i.e. it was returned by `globals()`),
/// sync all str-keyed entries back to the DictStorage.
fn dict_sync_dict_storage_proxy(dict: PyObjectRef) {
    unsafe {
        let ns_ptr = pyre_object::w_dict_get_dict_storage_proxy(dict);
        if ns_ptr.is_null() {
            return;
        }
        let ns = &mut *(ns_ptr as *mut crate::DictStorage);
        let dict_obj = &*(dict as *const pyre_object::dictobject::W_DictObject);
        let entries = &*dict_obj.entries;
        for &(k, v) in entries {
            if pyre_object::is_str(k) {
                let name = pyre_object::w_str_get_value(k);
                crate::dict_storage_store(ns, name, v);
            }
        }
    }
}

/// PyPy: dictobject.py descr_pop
pub fn dict_method_pop(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "dict.pop() takes at least 1 argument");
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied();
    if !dict.is_null() {
        unsafe {
            if let Some(val) = w_dict_lookup(dict, key) {
                // Delete the key after lookup — PyPy: W_DictMultiObject.descr_pop
                if pyre_object::is_str(key) {
                    pyre_object::dictobject::w_dict_delitem_str(
                        dict,
                        pyre_object::w_str_get_value(key),
                    );
                }
                return Ok(val);
            }
        }
    }
    if let Some(d) = default {
        Ok(d)
    } else {
        Err(crate::PyError::key_error("dict.pop(): key not found"))
    }
}

pub fn dict_method_setdefault(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2);
    let dict = resolve_dict_backing(args[0]);
    let key = args[1];
    let default = args.get(2).copied().unwrap_or_else(w_none);
    if !dict.is_null() {
        unsafe {
            if let Some(existing) = w_dict_lookup(dict, key) {
                return Ok(existing);
            }
            w_dict_store(dict, key, default);
        }
    }
    Ok(default)
}

// ── Tuple methods ────────────────────────────────────────────────────

/// PyPy: tupleobject.py descr_index — tuple.index(value)
pub fn tuple_method_index(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "index() takes at least 1 argument");
    let tup = args[0];
    let value = args[1];
    unsafe {
        let n = w_tuple_len(tup);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(tup, i as i64) {
                if std::ptr::eq(item, value) {
                    return Ok(w_int_new(i as i64));
                }
                if is_int(item) && is_int(value) && w_int_get_value(item) == w_int_get_value(value)
                {
                    return Ok(w_int_new(i as i64));
                }
            }
        }
    }
    panic!("ValueError: tuple.index(x): x not in tuple")
}

/// PyPy: tupleobject.py descr_count — tuple.count(value)
pub fn tuple_method_count(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "count() takes exactly 1 argument");
    let tup = args[0];
    let value = args[1];
    let mut count: i64 = 0;
    unsafe {
        let n = w_tuple_len(tup);
        for i in 0..n {
            if let Some(item) = w_tuple_getitem(tup, i as i64) {
                if std::ptr::eq(item, value)
                    || (is_int(item)
                        && is_int(value)
                        && w_int_get_value(item) == w_int_get_value(value))
                {
                    count += 1;
                }
            }
        }
    }
    Ok(w_int_new(count))
}
