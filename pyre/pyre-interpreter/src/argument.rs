//! Arguments objects.
//!
//! Line-by-line port of `pypy/interpreter/argument.py:Arguments`.
//! Surface ported: struct + `new` (PyPy `__init__`) + `firstarg` +
//! `_combine_wrapped` + `_jit_few_keywords` + `fixedunpack` +
//! `unpack` + `replace_arguments` + `prepend` + `_match_signature` +
//! `_match_keywords` + `_collect_keyword_args` + `_parse` +
//! `parse_into_scope` + `parse_obj` + `frompacked` + `topacked`.
//!
//! Pyre's legacy `call::call_callable` surface still takes a flat
//! `&[PyObjectRef]`.  Callers that know the keyword layout route
//! through `Arguments::with_kw` before entering the profiled-builtin
//! path; callers with only a flat positional slice use
//! `Arguments::positional_only`.  Both shortcuts now delegate to `new`
//! so the PyPy `__init__` invariant chain runs once.

use pyre_object::PyObjectRef;

/// PyPy `oefmt(..., "%T", w_obj)` — resolves the wrapped object's
/// type name for diagnostic messages, mirroring
/// `space.type(w_obj).getname(space)`.  Used by the `*args` /
/// `**kwargs` parity messages below ("argument after * must be an
/// iterable, not %T", "argument after ** must be a mapping, not %T",
/// "keywords must be strings, not '%T'").
fn type_name_of(w_obj: PyObjectRef) -> String {
    unsafe {
        match crate::typedef::r#type(w_obj) {
            Some(tp) => pyre_object::w_type_get_name(tp).to_string(),
            None => (*(*w_obj).ob_type).name.to_string(),
        }
    }
}

/// pypy/interpreter/argument.py:13-17 `raise_type_error`.
///
/// ```python
/// @specialize.arg(2)
/// def raise_type_error(space, w_function, msg, *args):
///     if w_function is None:
///         raise oefmt(space.w_TypeError, msg, *args)
///     msg = "%s " + msg
///     raise oefmt(space.w_TypeError, msg, space.object_functionstr(w_function), *args)
/// ```
///
/// Returns a `PyError` of kind `TypeError` rather than raising via
/// `oefmt(space.w_TypeError, ...)`.  When `w_function` is `PY_NULL`
/// (PyPy's `None` sentinel) the message lands without prefix; otherwise
/// `object_functionstr` resolves a leading `"<qualname>() "` prefix.
///
/// Caller sites in `argument.py` (`_combine_starargs_wrapped`:99,
/// `_check_not_duplicate_kwargs`:415, `_do_combine_starstarargs_wrapped`:434
/// + 443) all use this helper to format diagnostic context.
///
/// PyPy's `oefmt(space.w_TypeError, "%s ..." , space.object_functionstr(...))`
/// instantiates the `%s` substitution at raise time; if
/// `object_functionstr` itself raises an async OperationError
/// (SystemExit / KeyboardInterrupt via `findattr`'s
/// `baseobjspace.py:881-884` re-raise), the async error preempts the
/// TypeError and propagates as the actual exception PyPy raises.
/// Pyre mirrors this by returning the async `PyError` verbatim
/// instead of building the TypeError prefix.
pub fn raise_type_error(w_function: PyObjectRef, msg: String) -> crate::PyError {
    if w_function.is_null() {
        return crate::PyError::type_error(msg);
    }
    match crate::baseobjspace::object_functionstr(w_function) {
        Ok(prefix) => crate::PyError::type_error(format!("{prefix} {msg}")),
        // Async findattr propagates as the actual raise (matches
        // PyPy `oefmt`'s argument-evaluation order: side effects of
        // `%s` formatting fire before the TypeError is constructed).
        Err(async_err) => async_err,
    }
}

/// pypy/interpreter/argument.py:419-423 `contains_w_names`.
///
/// ```python
/// def contains_w_names(w_key, keys_w):
///     for w_other in keys_w:
///         if w_other.eq_w(w_key):
///             return True
///     return False
/// ```
///
/// PyPy: `w_other.eq_w(w_key)` is the typed unicode method
/// `pypy/objspace/std/unicodeobject.py:348-351`:
///
/// ```python
/// def eq_w(self, w_other):
///     # shortcut for UnicodeDictStrategy
///     assert isinstance(w_other, W_UnicodeObject)
///     return self._utf8 == w_other._utf8
/// ```
///
/// Both arms in `argument.py` only feed string keys into this helper
/// (`_do_combine_starstarargs_wrapped` stamps `assert isinstance(w_key,
/// space.UnicodeObjectCls)` after `text_w`, and `keyword_names_w` is a
/// parallel string-typed list), so the typed dispatch reduces to byte
/// equality of the underlying utf8.
///
/// PRE-EXISTING-ADAPTATION: pyre's `W_UnicodeObject` does not yet
/// expose an `eq_w` method, so the byte-equality is open-coded via
/// `pyre_object::w_str_get_value` and gated on `is_str`.  The identity
/// fast path `is_w` is preserved as the cheap shortcut PyPy gets from
/// CPython-style interned-string identity.  Routing through
/// `space.eq_w` (the generic 2-arg fallback) would invoke Python-level
/// `__eq__` on hand-built Arguments and silently drift.
pub fn contains_w_names(w_key: PyObjectRef, keys_w: &[PyObjectRef]) -> bool {
    for &w_other in keys_w {
        if crate::baseobjspace::is_w(w_other, w_key) {
            return true;
        }
        unsafe {
            if pyre_object::is_str(w_other)
                && pyre_object::is_str(w_key)
                && pyre_object::w_str_get_value(w_other) == pyre_object::w_str_get_value(w_key)
            {
                return true;
            }
        }
    }
    false
}

/// pypy/interpreter/argument.py:410-417 `_check_not_duplicate_kwargs`.
///
/// ```python
/// def _check_not_duplicate_kwargs(space, existingkeywords_w, keyword_names_w, keywords_w, w_function):
///     # looks quadratic, but the JIT should remove all of it nicely.
///     # Also, all the lists should be small
///     for w_key in keyword_names_w:
///         if contains_w_names(w_key, existingkeywords_w):
///             raise_type_error(space, w_function,
///                         "got multiple values for keyword argument '%S'",
///                         w_key)
/// ```
///
/// `keywords_w` is the parallel value list, kept in the signature for
/// upstream parity but unused in the duplicate check (only the names
/// matter).
pub fn check_not_duplicate_kwargs(
    existingkeywords_w: &[PyObjectRef],
    keyword_names_w: &[PyObjectRef],
    _keywords_w: &[PyObjectRef],
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    for &w_key in keyword_names_w {
        if contains_w_names(w_key, existingkeywords_w) {
            let key_repr = unsafe {
                if pyre_object::is_str(w_key) {
                    pyre_object::w_str_get_value(w_key).to_string()
                } else {
                    crate::display::py_str(w_key)
                }
            };
            return Err(raise_type_error(
                w_function,
                format!("got multiple values for keyword argument '{key_repr}'"),
            ));
        }
    }
    Ok(())
}

/// pypy/interpreter/argument.py:425-459 `_do_combine_starstarargs_wrapped`.
///
/// ```python
/// def _do_combine_starstarargs_wrapped(space, keys_w, w_starstararg, keyword_names_w,
///         keywords_w, existingkeywords_w, is_dict, w_function):
///     i = 0
///     seen = {}
///     for w_key in keys_w:
///         try:
///             key = space.text_w(w_key)
///         except OperationError as e:
///             if e.match(space, space.w_TypeError):
///                 raise_type_error(space, w_function,
///                             "keywords must be strings, not '%T'",
///                             w_key)
///             raise
///         else:
///             if ((existingkeywords_w and
///                  contains_w_names(w_key, existingkeywords_w)) or
///                 key in seen
///             ):
///                 raise_type_error(space, w_function,
///                             "got multiple values for keyword argument '%S'",
///                             w_key)
///         seen[key] = None
///         assert isinstance(w_key, space.UnicodeObjectCls)
///         keyword_names_w[i] = w_key
///         if is_dict:
///             # issue 2435: bug-to-bug compatibility with cpython.
///             from pypy.objspace.descroperation import dict_getitem
///             w_descr = dict_getitem(space)
///             w_value = space.get_and_call_function(w_descr, w_starstararg, w_key)
///         else:
///             w_value = space.getitem(w_starstararg, w_key)
///         keywords_w[i] = w_value
///         i += 1
/// ```
///
/// argument.py:449-457 — when `is_dict` is True (an exact `dict`
/// instance, or a `dict` subclass with no `__iter__` override), PyPy
/// bypasses the type's `__getitem__` slot via `dict_getitem` to avoid
/// CPython issue 2435 where `dict` subclasses' `__getitem__` is silently
/// ignored.  Pyre routes the True arm through
/// `pyre_object::dictobject::w_dict_getitem_str` for the same direct
/// dict-storage access; the False arm goes through the generic
/// `space.getitem` so subclass `__getitem__` overrides win when they
/// should.  The string key was already extracted at line 197 so no
/// extra `text_w` call is needed.
pub fn do_combine_starstarargs_wrapped(
    keys_w: &[PyObjectRef],
    w_starstararg: PyObjectRef,
    keyword_names_w: &mut [PyObjectRef],
    keywords_w: &mut [PyObjectRef],
    existingkeywords_w: Option<&[PyObjectRef]>,
    is_dict: bool,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // argument.py:428 `seen = {}` — Python dict with None values.  Map
    // to `HashMap<String, ()>` rather than `HashSet` so the shape
    // matches PyPy's `dict` (key membership tested via `in`, value slot
    // present-but-unread per `seen[key] = None` at line 446).
    let mut seen: std::collections::HashMap<String, ()> = std::collections::HashMap::new();
    for (i, &w_key) in keys_w.iter().enumerate() {
        // argument.py:431 — `key = space.text_w(w_key)`; raise TypeError
        // if w_key is not a string.  argument.py:434-436 message:
        // `"keywords must be strings, not '%T'"`.
        let key = unsafe {
            if !pyre_object::is_str(w_key) {
                let tp = type_name_of(w_key);
                return Err(raise_type_error(
                    w_function,
                    format!("keywords must be strings, not '{tp}'"),
                ));
            }
            pyre_object::w_str_get_value(w_key).to_string()
        };
        // argument.py:439-445 — duplicate check against existing kwargs +
        // already-seen names in this iteration.
        let already_in_existing = existingkeywords_w
            .map(|existing| contains_w_names(w_key, existing))
            .unwrap_or(false);
        if already_in_existing || seen.contains_key(&key) {
            return Err(raise_type_error(
                w_function,
                format!("got multiple values for keyword argument '{key}'"),
            ));
        }
        // argument.py:448 — `keyword_names_w[i] = w_key`.
        keyword_names_w[i] = w_key;
        // argument.py:449-457 — value lookup.
        let w_value = if is_dict {
            // argument.py:449-455 — `dict_getitem` direct-storage
            // access, bypassing any subclass `__getitem__` override.
            // PyPy: `dict_getitem(space)` is the unbound dict method
            // descriptor; `space.get_and_call_function(w_descr,
            // w_starstararg, w_key)` invokes it on the subclass
            // instance.  PyPy's dict subclasses ARE
            // `W_DictMultiObject` so the descriptor reads the dict's
            // own storage directly.
            //
            // Pyre adaptation: dict subclasses are
            // `W_InstanceObject` with a `__dict_data__` backing dict
            // (`typedef.rs:820 dict_descr_new`).  Route through
            // `type_methods::resolve_dict_backing` to recover the
            // backing `W_DictObject`, then perform the same direct
            // storage read.  For exact `dict` instances this is a
            // no-op identity, so the fast path stays optimal.
            //
            // Key was extracted from `w_starstararg.keys()` immediately
            // upstream so it must be present in the dict; the only way
            // `w_dict_getitem_str` returns None is mid-iteration
            // mutation, which `dict.__getitem__` (the descriptor PyPy
            // calls via `dict_getitem`) surfaces as KeyError.
            let backing = crate::type_methods::resolve_dict_backing(w_starstararg);
            let direct = if backing.is_null() {
                None
            } else {
                unsafe { pyre_object::dictobject::w_dict_getitem_str(backing, &key) }
            };
            match direct {
                Some(v) => v,
                None => return Err(crate::PyError::key_error(format!("'{key}'"))),
            }
        } else {
            // argument.py:457 — `w_value = space.getitem(...)`.
            crate::baseobjspace::getitem(w_starstararg, w_key)?
        };
        keywords_w[i] = w_value;
        // argument.py:446 `seen[key] = None` — record key, value slot
        // is unread.
        seen.insert(key, ());
    }
    Ok(())
}

/// pypy/interpreter/argument.py:92-104 `_combine_starargs_wrapped`.
///
/// ```python
/// def _combine_starargs_wrapped(self, w_stararg, w_function=None):
///     # unpack the * arguments
///     space = self.space
///     try:
///         args_w = space.fixedview(w_stararg)
///     except OperationError as e:
///         if (e.match(space, space.w_TypeError) and
///                 not space.is_iterable(w_stararg)):
///             raise_type_error(space, w_function,
///                         "argument after * must be an iterable, not %T",
///                         w_stararg)
///         raise
///     self.arguments_w = self.arguments_w + args_w
/// ```
///
/// Operates on a borrowed `&mut Vec<PyObjectRef>` so the
/// `Arguments::_combine_wrapped` constructor flow can extend the
/// owned `self.arguments_w` Vec in place.
pub fn combine_starargs_wrapped(
    arguments_w: &mut Vec<PyObjectRef>,
    w_stararg: PyObjectRef,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    match crate::baseobjspace::fixedview(w_stararg, -1) {
        Ok(args_w) => {
            // argument.py:104 — `self.arguments_w = self.arguments_w + args_w`.
            arguments_w.extend(args_w);
            Ok(())
        }
        Err(e) => {
            // argument.py:97-103 — TypeError + non-iterable surfaces
            // `"argument after * must be an iterable, not %T"`; other
            // errors propagate.
            if e.kind == crate::PyErrorKind::TypeError
                && !crate::baseobjspace::is_iterable(w_stararg)
            {
                let tp = type_name_of(w_stararg);
                Err(raise_type_error(
                    w_function,
                    format!("argument after * must be an iterable, not {tp}"),
                ))
            } else {
                Err(e)
            }
        }
    }
}

/// pypy/interpreter/argument.py:106-150 `_combine_starstarargs_wrapped`.
///
/// ```python
/// def _combine_starstarargs_wrapped(self, w_starstararg, w_function=None):
///     # unpack the ** arguments
///     space = self.space
///     keyword_names_w, values_w = space.view_as_kwargs(w_starstararg)
///     if keyword_names_w is not None: # this path also taken for empty dicts
///         if self.keyword_names_w is None:
///             self.keyword_names_w = keyword_names_w
///             self.keywords_w = values_w
///         else:
///             _check_not_duplicate_kwargs(
///                 self.space, self.keyword_names_w, keyword_names_w, values_w,
///                 w_function)
///             self.keyword_names_w = self.keyword_names_w + keyword_names_w
///             self.keywords_w = self.keywords_w + values_w
///         return
///     is_dict = False
///     if space.isinstance_w(w_starstararg, space.w_dict):
///         w_st_iter = space.newtext("__iter__")
///         is_dict = space.is_w(
///                 space.findattr(space.type(w_starstararg), w_st_iter),
///                 space.findattr(space.w_dict, w_st_iter))
///         keys_w = space.unpackiterable(w_starstararg)
///     else:
///         try:
///             w_keys = space.call_method(w_starstararg, "keys")
///         except OperationError as e:
///             if e.match(space, space.w_AttributeError):
///                 raise_type_error(space, w_function,
///                             "argument after ** must be a mapping, not %T",
///                             w_starstararg)
///             raise
///         keys_w = space.unpackiterable(w_keys)
///     keywords_w = [None] * len(keys_w)
///     keyword_names_w = [None] * len(keys_w)
///     _do_combine_starstarargs_wrapped(...)
///     ...merge into self.keyword_names_w / self.keywords_w...
/// ```
///
/// PRE-EXISTING-ADAPTATION: pyre's `view_as_kwargs` always returns
/// `(None, [])` (kwargsdict variant unported), so the fast-path arm
/// (lines 110-120) is unreachable until the dict-strategy port lands.
/// The slow `keys()` iteration arm runs unconditionally for now.
pub fn combine_starstarargs_wrapped(
    keyword_names_out: &mut Vec<PyObjectRef>,
    keywords_out: &mut Vec<PyObjectRef>,
    w_starstararg: PyObjectRef,
    w_function: PyObjectRef,
) -> Result<(), crate::PyError> {
    // argument.py:109 — fast path via view_as_kwargs.  Both halves of
    // the tuple are `Option`; PyPy's base default is `(None, None)`
    // while the kwargsdict-aware override returns `(Some, Some)`.
    let (fast_names, fast_values) = crate::baseobjspace::view_as_kwargs(w_starstararg);
    if let Some(names) = fast_names {
        let values = fast_values
            .expect("baseobjspace.py:1159 view_as_kwargs returns matching Some/Some or None/None");
        // argument.py:111-119 — merge with optional duplicate check.
        if !keyword_names_out.is_empty() {
            check_not_duplicate_kwargs(keyword_names_out, &names, &values, w_function)?;
        }
        keyword_names_out.extend(names);
        keywords_out.extend(values);
        return Ok(());
    }
    // argument.py:121-128 — slow path: derive keys list, dispatch on
    // dict-fast-path vs generic mapping.
    //
    // ```python
    // is_dict = False
    // if space.isinstance_w(w_starstararg, space.w_dict):
    //     w_st_iter = space.newtext("__iter__")
    //     # bug-to-bug compatibility: CPython ignores __getitem__
    //     # overrides in dict subclasses if there is no __iter__
    //     is_dict = space.is_w(
    //             space.findattr(space.type(w_starstararg), w_st_iter),
    //             space.findattr(space.w_dict, w_st_iter))
    // ```
    //
    // The double check (isinstance + `__iter__` identity) ensures a
    // dict subclass that overrides `__iter__` falls through to the
    // generic `__getitem__` path, while subclasses that don't override
    // it use the direct dict-storage bypass.
    let w_dict_type = crate::typedef::gettypeobject(&pyre_object::pyobject::DICT_TYPE);
    // `gettypeobject` returns null before `init_typeobjects()` runs
    // (early bootstrap or unit tests).  Fall back to the exact-type
    // `pyre_object::is_dict` check in that case — pyre has no dict
    // subclass yet, so the answer matches PyPy's
    // `isinstance_w(w_starstararg, w_dict)` for every reachable
    // input.  Once `init_typeobjects` always runs before this code
    // path, the fallback collapses to dead code.
    let is_dict = unsafe {
        let isinst = if w_dict_type.is_null() {
            pyre_object::is_dict(w_starstararg)
        } else {
            crate::baseobjspace::isinstance_w(w_starstararg, w_dict_type)
        };
        if !isinst {
            false
        } else {
            // argument.py:127-128 — `space.findattr(space.type(...),
            // w_st_iter)`: PyPy issues `getattr` on the type object
            // itself, going through the metaclass / descriptor lookup
            // chain.  `space.findattr` returns `None` for ordinary
            // OperationError and re-raises async (baseobjspace.py:878).
            // Pyre's `findattr` matches the same shape (Option<W>),
            // modulo async-propagation (still a known gap covered by
            // the `findattr` PRE-EXISTING-ADAPTATION).
            let w_obj_type = crate::typedef::r#type(w_starstararg).unwrap_or(pyre_object::PY_NULL);
            let lhs = if w_obj_type.is_null() {
                None
            } else {
                crate::baseobjspace::findattr(w_obj_type, "__iter__")
            };
            let rhs = if w_dict_type.is_null() {
                None
            } else {
                crate::baseobjspace::findattr(w_dict_type, "__iter__")
            };
            // `space.is_w` is pointer identity.
            match (lhs, rhs) {
                (Some(a), Some(b)) => crate::baseobjspace::is_w(a, b),
                (None, None) => true,
                _ => false,
            }
        }
    };
    let keys_w: Vec<PyObjectRef> = if is_dict {
        crate::baseobjspace::unpackiterable(w_starstararg, -1)?
    } else {
        // argument.py:131-138 — the `try` covers the whole
        // `space.call_method(w_starstararg, "keys")`, so an
        // AttributeError raised either while finding `keys` OR inside
        // the keys() call is converted to the mapping TypeError below.
        let w_keys_meth = match crate::baseobjspace::getattr(w_starstararg, "keys") {
            Ok(m) => m,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                let tp = type_name_of(w_starstararg);
                return Err(raise_type_error(
                    w_function,
                    format!("argument after ** must be a mapping, not {tp}"),
                ));
            }
            Err(e) => return Err(e),
        };
        let w_keys = match crate::call::call_function_impl_result(w_keys_meth, &[]) {
            Ok(w) => w,
            Err(e) if e.kind == crate::PyErrorKind::AttributeError => {
                let tp = type_name_of(w_starstararg);
                return Err(raise_type_error(
                    w_function,
                    format!("argument after ** must be a mapping, not {tp}"),
                ));
            }
            Err(e) => return Err(e),
        };
        crate::baseobjspace::unpackiterable(w_keys, -1)?
    };
    // argument.py:140-141 — pre-allocate parallel name/value buffers.
    let n = keys_w.len();
    let mut keyword_names_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; n];
    let mut keywords_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; n];
    // argument.py:142-144 — slow-path body fills name/value buffers,
    // checking for duplicates against keyword_names_out (existing).
    let existing: Option<&[PyObjectRef]> = if keyword_names_out.is_empty() {
        None
    } else {
        Some(keyword_names_out.as_slice())
    };
    do_combine_starstarargs_wrapped(
        &keys_w,
        w_starstararg,
        &mut keyword_names_w,
        &mut keywords_w,
        existing,
        is_dict,
        w_function,
    )?;
    // argument.py:145-150 — merge into the running output buffers.
    keyword_names_out.extend(keyword_names_w);
    keywords_out.extend(keywords_w);
    Ok(())
}

/// `pypy/interpreter/argument.py:20 class Arguments`.
///
/// PyPy fields (argument.py:34-53):
/// ```text
/// self.space            -- always available; pyre passes context implicitly
/// self.arguments_w      -- list[w_obj]
/// self.keyword_names_w  -- list[w_text] or None
/// self.keywords_w       -- list[w_obj]   or None
/// self._jit_few_keywords -- bool, JIT unroll hint (argument.py:50)
/// self.methodcall       -- bool flag (argument.py:53)
/// ```
///
/// `w_stararg`, `w_starstararg`, `w_function` are constructor inputs
/// that PyPy's `_combine_wrapped` (argument.py:85-90) expands into
/// `arguments_w` / `keyword_names_w` / `keywords_w` at construction
/// time.  They are NOT stored as instance state in PyPy — only their
/// expanded form is.
///
/// Owns its `arguments_w` / `keyword_names_w` / `keywords_w` Vec
/// storage line-by-line with PyPy: `__init__` assigns `self.arguments_w =
/// args_w` (a list), and `_combine_starargs_wrapped` /
/// `_combine_starstarargs_wrapped` mutate the list via `+= args_w`.
/// Pyre's earlier borrowed-slice shape forced star expansion to
/// happen at the call site (because the slice could not grow);
/// owning the Vec lets `_combine_wrapped` invoke
/// `combine_starargs_wrapped` / `combine_starstarargs_wrapped`
/// in-place, matching the upstream constructor flow.
///
/// PRE-EXISTING-ADAPTATION (justified): PyPy stores RPython list
/// references — the same mutable list object can be aliased between
/// the caller's `args` and `Arguments.arguments_w`.  Pyre owns
/// `Vec<PyObjectRef>` and clones at construction.  Rust's borrow
/// checker forbids `replace_arguments` (argument.py:77-79) and
/// `_combine_starargs_wrapped` (argument.py:90-101) from building a
/// new `Arguments` that aliases `&self.arguments_w` while the
/// existing instance still holds a `&mut` to it.  Wrapping in
/// `Rc<RefCell<Vec<PyObjectRef>>>` would restore the alias shape
/// but adds runtime refcount + per-access borrow check overhead and
/// would diverge from pyre's owned-Vec convention used elsewhere
/// (`PyFrame.locals_w`, `Signature.argnames`, etc.).  Functional
/// consequence is identical — PyPy's call paths consume `Arguments`
/// before mutating the source, so post-construction mutations to
/// either side never propagate either way.
pub struct Arguments {
    /// argument.py:36 `self.arguments_w = args_w`.
    pub arguments_w: Vec<PyObjectRef>,
    /// argument.py:38 `self.keyword_names_w = keyword_names_w` (`None` allowed).
    pub keyword_names_w: Option<Vec<PyObjectRef>>,
    /// argument.py:39 `self.keywords_w = keywords_w` (`None` allowed,
    /// must be parallel to `keyword_names_w` when present —
    /// argument.py:42 `assert len(keywords_w) == len(keyword_names_w)`).
    pub keywords_w: Option<Vec<PyObjectRef>>,
    /// argument.py:50 `self._jit_few_keywords = self.keyword_names_w
    /// is None or jit.isconstant(len(self.keyword_names_w))`.
    /// Pyre's tracing JIT does not yet read this hint, but the field
    /// is set so the unroll predicate is observable when the JIT
    /// catches up.
    pub jit_few_keywords: bool,
    /// argument.py:53 `self.methodcall = methodcall`.  Default `false`
    /// for the `positional_only` / `with_kw` shortcuts; the future
    /// CALL_METHOD opcode port should set it `true`.
    pub methodcall: bool,
}

impl Arguments {
    /// pypy/interpreter/argument.py:31-53 `__init__` (full port).
    ///
    /// ```python
    /// def __init__(self, space, args_w, keyword_names_w=None,
    ///              keywords_w=None, w_stararg=None, w_starstararg=None,
    ///              methodcall=False, w_function=None):
    ///     self.space = space
    ///     assert isinstance(args_w, list)
    ///     self.arguments_w = args_w
    ///     self.keyword_names_w = keyword_names_w
    ///     self.keywords_w = keywords_w
    ///     if keyword_names_w is not None:
    ///         assert keywords_w is not None
    ///         assert len(keywords_w) == len(keyword_names_w)
    ///         make_sure_not_resized(self.keyword_names_w)
    ///         make_sure_not_resized(self.keywords_w)
    ///     make_sure_not_resized(self.arguments_w)
    ///     self._combine_wrapped(w_stararg, w_starstararg, w_function)
    ///     self._jit_few_keywords = self.keyword_names_w is None or jit.isconstant(len(self.keyword_names_w))
    ///     self.methodcall = methodcall
    /// ```
    ///
    /// `_jit_few_keywords` is a JIT-time elidability hint —
    /// `jit.isconstant(...)` is true when the trace recorder sees a
    /// fixed length.  Pyre approximates with `keyword_names_w.is_none()
    /// || keyword_names_w.len() <= JIT_FEW_KW_THRESHOLD`; the
    /// threshold mirrors PyPy's elidable unroll in
    /// `argument.py:73 unpack` (the JIT only unrolls a few iterations).
    ///
    /// `space` is implicit in pyre (carried by call sites).
    /// `_combine_wrapped(w_stararg, w_starstararg, w_function)` runs
    /// the star expansion in-place against the freshly-cloned Vec
    /// storage so that `Arguments::new(.., w_stararg=Some(w), ..)`
    /// behaves like PyPy's `Arguments(space, args_w, ..., w_stararg=w)`
    /// constructor (the helpers it depends on — `space.fixedview`,
    /// `space.view_as_kwargs`, `space.unpackiterable`,
    /// `space.is_iterable`, `space.call_method` — landed in
    /// Pre-B.1/Pre-B.2/Pre-B.3).  Returns a `PyError` propagated from
    /// the helpers when the star expansion fails (e.g. non-iterable
    /// `*` arg).
    pub fn new(
        args_w: &[PyObjectRef],
        keyword_names_w: Option<&[PyObjectRef]>,
        keywords_w: Option<&[PyObjectRef]>,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
        methodcall: bool,
        w_function: Option<PyObjectRef>,
    ) -> Result<Self, crate::PyError> {
        // argument.py:40-44 — keyword_names_w / keywords_w invariant.
        if let (Some(names), Some(values)) = (keyword_names_w, keywords_w) {
            debug_assert_eq!(names.len(), values.len());
        } else {
            debug_assert!(
                keyword_names_w.is_none() && keywords_w.is_none(),
                "keyword_names_w and keywords_w must agree on Some/None"
            );
        }
        let mut arguments = Self {
            arguments_w: args_w.to_vec(),
            keyword_names_w: keyword_names_w.map(|s| s.to_vec()),
            keywords_w: keywords_w.map(|s| s.to_vec()),
            // argument.py:50 — jit_few_keywords initial guess; the
            // post-`_combine_wrapped` value below is the canonical one.
            jit_few_keywords: keyword_names_w.is_none(),
            methodcall,
        };
        let w_function = w_function.unwrap_or(pyre_object::PY_NULL);
        arguments._combine_wrapped(w_stararg, w_starstararg, w_function)?;
        // argument.py:50 — recompute after `_combine_wrapped`, since
        // that helper may have grown `keyword_names_w`.
        arguments.jit_few_keywords = match arguments.keyword_names_w.as_ref() {
            None => true,
            Some(names) => names.len() <= 8, // pyre approximation of jit.isconstant
        };
        Ok(arguments)
    }

    /// pypy/interpreter/argument.py:85-90 `_combine_wrapped`.
    ///
    /// ```python
    /// def _combine_wrapped(self, w_stararg, w_starstararg, w_function=None):
    ///     "unpack the *arg and **kwd into arguments_w and keywords_w"
    ///     if w_stararg is not None:
    ///         self._combine_starargs_wrapped(w_stararg, w_function)
    ///     if w_starstararg is not None:
    ///         self._combine_starstarargs_wrapped(w_starstararg, w_function)
    /// ```
    ///
    /// Routes through the free-fn helpers `combine_starargs_wrapped` /
    /// `combine_starstarargs_wrapped` defined above (Phase A.1) which
    /// operate on `&mut Vec<PyObjectRef>` storage in-place.
    fn _combine_wrapped(
        &mut self,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
        w_function: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // argument.py:87-88 — `if w_stararg is not None: self._combine_starargs_wrapped(...)`.
        if let Some(w_star) = w_stararg {
            combine_starargs_wrapped(&mut self.arguments_w, w_star, w_function)?;
        }
        // argument.py:89-90 — `if w_starstararg is not None: self._combine_starstarargs_wrapped(...)`.
        // Materialize keyword_names_w / keywords_w as empty Vecs if the
        // field is currently None so the helper has a buffer to extend
        // (mirrors argument.py:111-112 `if self.keyword_names_w is None:
        // self.keyword_names_w = keyword_names_w`).  Reassign Some after
        // the helper runs whether or not it produced any keys, mirroring
        // PyPy's unconditional `self.keyword_names_w = ...` at lines
        // 145-150.
        if let Some(w_starstar) = w_starstararg {
            let mut names = self.keyword_names_w.take().unwrap_or_default();
            let mut values = self.keywords_w.take().unwrap_or_default();
            combine_starstarargs_wrapped(&mut names, &mut values, w_starstar, w_function)?;
            self.keyword_names_w = Some(names);
            self.keywords_w = Some(values);
        }
        Ok(())
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` (positional-only shortcut).
    ///
    /// Used for call surfaces that have only positional args (no
    /// kwargs, no star expansion).  Cannot fail because there are no
    /// star args to unpack.
    #[inline]
    pub fn positional_only(args_w: &[PyObjectRef]) -> Self {
        Self::new(args_w, None, None, None, None, false, None)
            .expect("positional_only cannot fail without star args")
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` (positional + kwargs shortcut).
    ///
    /// `keyword_names_w` and `keywords_w` are parallel slices
    /// (argument.py:42 `assert len(keywords_w) == len(keyword_names_w)`).
    /// Callers with both positional and kwargs (e.g. the
    /// `call.rs:call_with_kwargs` builtin path) use this to keep
    /// the kwargs separated from `arguments_w`, so `firstarg()`
    /// returns `arguments_w[0]` (or `None`) rather than surfacing
    /// the trailing kwargs dict that pyre's flat call surface
    /// otherwise appends to the merged slice.  Cannot fail because
    /// there are no star args to unpack.
    #[inline]
    pub fn with_kw(
        args_w: &[PyObjectRef],
        keyword_names_w: &[PyObjectRef],
        keywords_w: &[PyObjectRef],
    ) -> Self {
        Self::new(
            args_w,
            Some(keyword_names_w),
            Some(keywords_w),
            None,
            None,
            false,
            None,
        )
        .expect("with_kw cannot fail without star args")
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` shortcut with the
    /// `methodcall` flag explicit.  Use when the caller has both kwargs
    /// and the `methodcall` flag (e.g. CALL_METHOD lowering); when
    /// methodcall is false, `with_kw` is the lighter alternative.
    /// Cannot fail because there are no star args to unpack.
    #[inline]
    pub fn full(
        args_w: &[PyObjectRef],
        keyword_names_w: Option<&[PyObjectRef]>,
        keywords_w: Option<&[PyObjectRef]>,
        methodcall: bool,
    ) -> Self {
        Self::new(
            args_w,
            keyword_names_w,
            keywords_w,
            None,
            None,
            methodcall,
            None,
        )
        .expect("full cannot fail without star args")
    }

    /// pypy/interpreter/argument.py:153-162 `fixedunpack`.
    ///
    /// ```python
    /// def fixedunpack(self, argcount):
    ///     """The simplest argument parsing: get the 'argcount' arguments,
    ///     or raise a real ValueError if the length is wrong."""
    ///     if self.keyword_names_w:
    ///         raise ValueError("no keyword arguments expected")
    ///     if len(self.arguments_w) > argcount:
    ///         raise ValueError("too many arguments (%d expected)" % argcount)
    ///     elif len(self.arguments_w) < argcount:
    ///         raise ValueError("not enough arguments (%d expected)" % argcount)
    ///     return self.arguments_w
    /// ```
    ///
    /// Returns a borrowed slice instead of cloning (PyPy returns the
    /// list directly too — same borrow semantics as Python's list
    /// reference return).  Error variant carries the formatted PyPy
    /// message including `argcount` per argument.py:159-161.
    pub fn fixedunpack(&self, argcount: usize) -> Result<&[PyObjectRef], String> {
        if self
            .keyword_names_w
            .as_ref()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
        {
            return Err("no keyword arguments expected".to_string());
        }
        if self.arguments_w.len() > argcount {
            return Err(format!("too many arguments ({argcount} expected)"));
        }
        if self.arguments_w.len() < argcount {
            return Err(format!("not enough arguments ({argcount} expected)"));
        }
        Ok(&self.arguments_w)
    }

    /// argument.py:164-168 — line-by-line port:
    /// ```python
    /// def firstarg(self):
    ///     "Return the first argument for inspection."
    ///     if self.arguments_w:
    ///         return self.arguments_w[0]
    ///     return None
    /// ```
    #[inline]
    pub fn firstarg(&self) -> Option<PyObjectRef> {
        if !self.arguments_w.is_empty() {
            Some(self.arguments_w[0])
        } else {
            None
        }
    }

    /// pypy/interpreter/argument.py:68-75 `unpack`.
    ///
    /// ```python
    /// @jit.look_inside_iff(lambda self: self._jit_few_keywords)
    /// def unpack(self): # slowish
    ///     "Return a ([w1,w2...], {'kw':w3...}) pair."
    ///     kwds_w = {}
    ///     if self.keyword_names_w:
    ///         for i in range(len(self.keyword_names_w)):
    ///             kwds_w[self.space.text_w(self.keyword_names_w[i])] = self.keywords_w[i]
    ///     return self.arguments_w, kwds_w
    /// ```
    ///
    /// PyPy returns a Python interpreter-level dict (RPython `{}`),
    /// whose semantics are: keys overwrite on duplicate (last-write-
    /// wins), no insertion-order guarantee.  Pyre returns a
    /// `HashMap<String, PyObjectRef>` to preserve the overwrite
    /// behaviour exactly — abnormal/hand-built duplicate keyword
    /// names collapse to the last value as PyPy does.  `arguments_w`
    /// is cloned because the storage is owned (PyPy returns the list
    /// reference; Rust's borrow checker forbids returning `&Vec`
    /// alongside a freshly-built second value without lifetime
    /// gymnastics, and `unpack` is documented as "slowish").
    ///
    /// Returns `Err(PyError(TypeError))` if any keyword name is not a
    /// string — PyPy's `space.text_w(w_name)` raises TypeError in that
    /// case, and `unpack`'s upstream consumers propagate the error via
    /// `OperationError`.
    pub fn unpack(
        &self,
    ) -> Result<
        (
            Vec<PyObjectRef>,
            std::collections::HashMap<String, PyObjectRef>,
        ),
        crate::PyError,
    > {
        let mut kwds_w: std::collections::HashMap<String, PyObjectRef> =
            std::collections::HashMap::new();
        if let (Some(names), Some(values)) =
            (self.keyword_names_w.as_ref(), self.keywords_w.as_ref())
        {
            for (w_name, w_value) in names.iter().zip(values.iter()) {
                let key = unsafe {
                    if pyre_object::is_str(*w_name) {
                        pyre_object::w_str_get_value(*w_name).to_string()
                    } else {
                        // argument.py:72 — `space.text_w(...)`.  PyPy's
                        // `_typed_unwrap_error` (baseobjspace.py:313-315)
                        // raises `TypeError("expected str, got %T object")`.
                        let tp = type_name_of(*w_name);
                        return Err(crate::PyError::type_error(format!(
                            "expected str, got {tp} object",
                        )));
                    }
                };
                // argument.py:74 — `kwds_w[key] = ...` overwrites on duplicate.
                kwds_w.insert(key, *w_value);
            }
        }
        Ok((self.arguments_w.clone(), kwds_w))
    }

    /// pypy/interpreter/argument.py:77-79 `replace_arguments`.
    ///
    /// ```python
    /// def replace_arguments(self, args_w):
    ///     "Return a new Arguments with a args_w as positional arguments."
    ///     return Arguments(self.space, args_w, self.keyword_names_w, self.keywords_w)
    /// ```
    ///
    /// PyPy invokes the bare 4-arg `Arguments(space, args_w,
    /// keyword_names_w, keywords_w)` constructor — `methodcall`
    /// defaults to `False` (argument.py:33).  Carrying
    /// `self.methodcall` forward would surface a stale "Did you
    /// forget 'self'" hint at the next signature mismatch.
    /// PyPy passes the existing keyword lists straight through (they
    /// are immutable list references in RPython); pyre clones the
    /// owned Vecs because the new instance needs its own storage.
    /// Cannot fail because no star args are passed.
    pub fn replace_arguments(&self, args_w: Vec<PyObjectRef>) -> Self {
        let keyword_names_w = self.keyword_names_w.clone();
        let keywords_w = self.keywords_w.clone();
        Self {
            arguments_w: args_w,
            jit_few_keywords: keyword_names_w
                .as_ref()
                .map(|n| n.len() <= 8)
                .unwrap_or(true),
            methodcall: false,
            keyword_names_w,
            keywords_w,
        }
    }

    /// pypy/interpreter/argument.py:81-83 `prepend`.
    ///
    /// ```python
    /// def prepend(self, w_firstarg):
    ///     "Return a new Arguments with a new argument inserted first."
    ///     return self.replace_arguments([w_firstarg] + self.arguments_w)
    /// ```
    pub fn prepend(&self, w_firstarg: PyObjectRef) -> Self {
        let mut args_w = Vec::with_capacity(self.arguments_w.len() + 1);
        args_w.push(w_firstarg);
        args_w.extend_from_slice(&self.arguments_w);
        self.replace_arguments(args_w)
    }

    /// pypy/interpreter/argument.py:172-338 `_match_signature`.
    ///
    /// Parse positional + kwargs arguments against `signature`, filling
    /// `scope_w` (caller-provided buffer of length `signature.scope_length()`).
    /// Returns `Err(ArgErr::*)` on shape mismatch.  PyPy raises ArgErr
    /// from the call site and converts to `oefmt(space.w_TypeError, ...)`
    /// at `parse_obj` / `parse_into_scope`.
    ///
    /// Inputs (matching the upstream parameter names):
    /// - `w_firstarg`: optional implicit first arg (e.g. `self` for
    ///   bound methods); `PY_NULL` means absent.
    /// - `scope_w`: output buffer, must be `signature.scope_length()`
    ///   long, pre-filled with `PY_NULL`.
    /// - `signature`: callee signature.
    /// - `defaults_w`: positional default values, `None` if absent.
    /// - `w_kw_defs`: keyword-only defaults dict, `PY_NULL` if absent.
    /// - `blindargs`: number of "blind" positional args from `prepend()`
    ///   (mostly 0; non-zero only for the bound-method dispatcher).
    pub fn match_signature(
        &self,
        w_firstarg: PyObjectRef,
        scope_w: &mut [PyObjectRef],
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<(), MatchSignatureError> {
        let co_posonlyargcount = signature.posonlyargcount;
        let co_argcount = signature.num_argnames();
        let co_kwonlyargcount = signature.num_kwonlyargnames();
        let mut too_many_args = false;

        // argument.py:191-201 — put `w_firstarg` into the scope.
        let mut upfront: usize = 0;
        // PyPy can prepend `[w_firstarg] + args_w` (line 201) when the
        // callee shape is `def meth(*args)` and a method-call adds
        // `self`; pyre clones into a Vec for that rare path.
        let mut args_w_owned: Option<Vec<PyObjectRef>> = None;
        let args_w: &[PyObjectRef] = if !w_firstarg.is_null() {
            if co_argcount > 0 {
                scope_w[0] = w_firstarg;
                upfront = 1;
                &self.arguments_w
            } else {
                // argument.py:198-201 fall-back.
                let mut v = Vec::with_capacity(self.arguments_w.len() + 1);
                v.push(w_firstarg);
                v.extend_from_slice(&self.arguments_w);
                args_w_owned = Some(v);
                args_w_owned.as_deref().unwrap()
            }
        } else {
            &self.arguments_w
        };
        let num_args = args_w.len();
        let avail = num_args + upfront;

        // argument.py:206-209 — kwds count.
        let keyword_names_w: Option<&[PyObjectRef]> = self.keyword_names_w.as_deref();
        let num_kwds = keyword_names_w.map(|k| k.len()).unwrap_or(0);

        // argument.py:211-220 — positional copy.
        let mut input_argcount = upfront;
        if input_argcount < co_argcount {
            let take = num_args.min(co_argcount - upfront);
            for i in 0..take {
                scope_w[i + input_argcount] = args_w[i];
            }
            input_argcount += take;
        }

        // argument.py:222-236 — *vararg collection.
        if signature.has_vararg() {
            let args_left = co_argcount - upfront;
            // argument.py:225 — `assert args_left >= 0` always holds in
            // pyre (usize subtraction would have panicked above).
            let starargs_w: Vec<PyObjectRef> = if num_args > args_left {
                if args_left == 0 {
                    args_w.to_vec()
                } else {
                    args_w[args_left..].to_vec()
                }
            } else {
                Vec::new()
            };
            let loc = co_argcount + co_kwonlyargcount;
            scope_w[loc] = pyre_object::w_tuple_new(starargs_w);
        } else if avail > co_argcount {
            too_many_args = true;
        }

        // argument.py:238-242 — **kwargs dict allocation.
        let mut w_kwds: PyObjectRef = pyre_object::PY_NULL;
        if signature.has_kwarg() {
            // PyPy: `space.newdict(kwargs=True)` produces a kwargs-strategy
            // dict; pyre's W_DictObject lacks the strategy variant so a
            // plain dict is used (PRE-EXISTING-ADAPTATION).
            w_kwds = pyre_object::dictobject::w_dict_new();
            let kwarg_loc = co_argcount + co_kwonlyargcount + (signature.has_vararg() as usize);
            scope_w[kwarg_loc] = w_kwds;
        }

        // argument.py:244-271 — keyword arg matching.
        let mut num_remainingkwds: usize = 0;
        let keywords_w: Option<&[PyObjectRef]> = self.keywords_w.as_deref();
        let mut kwds_mapping: Option<Vec<isize>> = None;
        if num_kwds > 0 {
            // argument.py:251-254 — pre-init `kwds_mapping[i] = -1`.
            let mapping_len = co_argcount + co_kwonlyargcount - input_argcount;
            let mut mapping = vec![-1isize; mapping_len];
            // argument.py:259-262 — match keyword names to argnames.
            num_remainingkwds = match_keywords(
                signature,
                blindargs,
                co_posonlyargcount,
                input_argcount,
                keyword_names_w.unwrap(),
                &mut mapping,
            )?;
            if num_remainingkwds > 0 {
                if !w_kwds.is_null() {
                    // argument.py:266-268 — collect overflow into **kwarg.
                    collect_keyword_args(
                        keyword_names_w.unwrap(),
                        keywords_w.unwrap(),
                        w_kwds,
                        &mapping,
                    )?;
                } else {
                    // argument.py:270-271 — ArgErrUnknownKwds.  PyPy's
                    // `__init__` (argument.py:609-617) calls
                    // `space.text_w(keyword_names_w[i])` on the single
                    // offending name when `num_remainingkwds == 1`; if
                    // that text-conversion fails (non-str key), the
                    // OperationError propagates back to the caller as a
                    // TypeError.  Pyre mirrors that propagation rather
                    // than silently substituting an empty string.
                    let mut name = String::new();
                    if num_remainingkwds == 1 {
                        let names = keyword_names_w.unwrap();
                        for (i, &w_n) in names.iter().enumerate() {
                            if !mapping_contains(&mapping, i as isize) {
                                if unsafe { pyre_object::is_str(w_n) } {
                                    name = unsafe { pyre_object::w_str_get_value(w_n).to_string() };
                                } else {
                                    // baseobjspace.py:313-315 `_typed_unwrap_error`
                                    // → `TypeError("expected str, got %T object")`.
                                    let tp = type_name_of(w_n);
                                    return Err(MatchSignatureError::Py(
                                        crate::PyError::type_error(format!(
                                            "expected str, got {tp} object",
                                        )),
                                    ));
                                }
                                break;
                            }
                        }
                    }
                    return Err(MatchSignatureError::Shape(ArgErr::UnknownKwds {
                        num_kwds: num_remainingkwds,
                        kwd_name: name,
                    }));
                }
            }
            kwds_mapping = Some(mapping);
        }

        // argument.py:273-287 — fill missing args from kwds.
        let more_filling = input_argcount < co_argcount + co_kwonlyargcount;
        let mut def_first: usize = 0;
        if more_filling {
            let defaults_len = defaults_w.map(|d| d.len()).unwrap_or(0);
            def_first = co_argcount.saturating_sub(defaults_len);
            let mut j = 0usize;
            for i in input_argcount..(co_argcount + co_kwonlyargcount) {
                if let Some(ref mapping) = kwds_mapping {
                    let kwds_index = mapping[j];
                    j += 1;
                    if kwds_index >= 0 {
                        scope_w[i] = keywords_w.unwrap()[kwds_index as usize];
                    }
                }
            }
        }

        // argument.py:289-300 — too-many-args ArgErr.
        if too_many_args {
            let mut kwonly_given: usize = 0;
            for i in co_argcount..(co_argcount + co_kwonlyargcount) {
                if !scope_w[i].is_null() {
                    kwonly_given += 1;
                }
            }
            let num_defaults = defaults_w.map(|d| d.len()).unwrap_or(0);
            if self.methodcall {
                return Err(MatchSignatureError::Shape(ArgErr::TooManyMethod {
                    signature: signature.clone(),
                    num_defaults,
                    given: avail,
                    kwonly_given,
                }));
            } else {
                return Err(MatchSignatureError::Shape(ArgErr::TooMany {
                    signature: signature.clone(),
                    num_defaults,
                    given: avail,
                    kwonly_given,
                }));
            }
        }

        // argument.py:302-338 — fill defaults + collect missing names.
        if more_filling {
            let mut missing_positional: Option<Vec<String>> = None;
            let mut missing_kwonly: Option<Vec<String>> = None;
            // argument.py:306-315 — posonly defaults.
            for i in input_argcount..co_argcount {
                if !scope_w[i].is_null() {
                    continue;
                }
                let defnum = (i as isize) - (def_first as isize);
                if defnum >= 0 {
                    scope_w[i] = defaults_w.unwrap()[defnum as usize];
                } else if let Some(list) = missing_positional.as_mut() {
                    list.push(signature.argnames[i].to_string());
                } else {
                    missing_positional = Some(vec![signature.argnames[i].to_string()]);
                }
            }
            // argument.py:317-333 — kwonly defaults via w_kw_defs dict.
            for i in co_argcount..(co_argcount + co_kwonlyargcount) {
                if !scope_w[i].is_null() {
                    continue;
                }
                let name = signature.argnames[i];
                if w_kw_defs.is_null() {
                    if let Some(list) = missing_kwonly.as_mut() {
                        list.push(name.to_string());
                    } else {
                        missing_kwonly = Some(vec![name.to_string()]);
                    }
                    continue;
                }
                // PyPy `baseobjspace.py:870 finditem` re-raises any
                // `OperationError` other than KeyError, so a kwonly-defaults
                // dict with a subclass `__getitem__` raising e.g.
                // `RuntimeError` surfaces here instead of being
                // mis-classified as "default missing".
                match crate::baseobjspace::finditem_str(w_kw_defs, name)? {
                    Some(w_def) => scope_w[i] = w_def,
                    None => {
                        if let Some(list) = missing_kwonly.as_mut() {
                            list.push(name.to_string());
                        } else {
                            missing_kwonly = Some(vec![name.to_string()]);
                        }
                    }
                }
            }
            // argument.py:335-338 — surface ArgErrMissing.
            if let Some(missing) = missing_positional {
                return Err(MatchSignatureError::Shape(ArgErr::Missing {
                    missing,
                    positional: true,
                }));
            }
            if let Some(missing) = missing_kwonly {
                return Err(MatchSignatureError::Shape(ArgErr::Missing {
                    missing,
                    positional: false,
                }));
            }
        }

        Ok(())
    }

    /// pypy/interpreter/argument.py:357-365 `_parse`.
    ///
    /// ```python
    /// def _parse(self, w_firstarg, signature, defaults_w, w_kw_defs, blindargs=0):
    ///     scopelen = signature.scope_length()
    ///     scope_w = [None] * scopelen
    ///     self._match_signature(w_firstarg, scope_w, signature, defaults_w,
    ///                           w_kw_defs, blindargs)
    ///     return scope_w
    /// ```
    fn _parse(
        &self,
        w_firstarg: PyObjectRef,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<Vec<PyObjectRef>, MatchSignatureError> {
        let scopelen = signature.scope_length();
        let mut scope_w: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; scopelen];
        self.match_signature(
            w_firstarg,
            &mut scope_w,
            signature,
            defaults_w,
            w_kw_defs,
            blindargs,
        )?;
        Ok(scope_w)
    }

    /// pypy/interpreter/argument.py:341-355 `parse_into_scope`.
    ///
    /// ```python
    /// def parse_into_scope(self, w_firstarg,
    ///                      scope_w, fnname, signature, defaults_w=None,
    ///                      w_kw_defs=None):
    ///     try:
    ///         self._match_signature(w_firstarg,
    ///                               scope_w, signature, defaults_w,
    ///                               w_kw_defs, 0)
    ///     except ArgErr as e:
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    ///     return signature.scope_length()
    /// ```
    pub fn parse_into_scope(
        &self,
        w_firstarg: PyObjectRef,
        scope_w: &mut [PyObjectRef],
        fnname: &str,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
    ) -> Result<usize, crate::PyError> {
        match self.match_signature(w_firstarg, scope_w, signature, defaults_w, w_kw_defs, 0) {
            Ok(()) => Ok(signature.scope_length()),
            Err(MatchSignatureError::Shape(e)) => Err(crate::PyError::type_error(format!(
                "{}() {}",
                fnname,
                e.getmsg()
            ))),
            Err(MatchSignatureError::Py(e)) => Err(e),
        }
    }

    /// pypy/interpreter/argument.py:368-383 `parse_obj`.
    ///
    /// ```python
    /// def parse_obj(self, w_firstarg,
    ///               fnname, signature, defaults_w=None, w_kw_defs=None,
    ///               blindargs=0):
    ///     try:
    ///         return self._parse(w_firstarg, signature, defaults_w, w_kw_defs,
    ///                            blindargs)
    ///     except ArgErrUnknownKwds as e:
    ///         if not signature.has_kwarg() and signature.kwonlyargcount == 0:
    ///             raise oefmt(self.space.w_TypeError,
    ///                         "%s() takes no keyword arguments", fnname)
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    ///     except ArgErr as e:
    ///         raise oefmt(self.space.w_TypeError, "%s() %8", fnname, e.getmsg())
    /// ```
    pub fn parse_obj(
        &self,
        w_firstarg: PyObjectRef,
        fnname: &str,
        signature: &crate::gateway::Signature,
        defaults_w: Option<&[PyObjectRef]>,
        w_kw_defs: PyObjectRef,
        blindargs: usize,
    ) -> Result<Vec<PyObjectRef>, crate::PyError> {
        match self._parse(w_firstarg, signature, defaults_w, w_kw_defs, blindargs) {
            Ok(scope_w) => Ok(scope_w),
            Err(MatchSignatureError::Shape(ArgErr::UnknownKwds { num_kwds, kwd_name })) => {
                // argument.py:378-380 — special-case "takes no keyword
                // arguments" when the signature has no **kwarg AND no
                // keyword-only args.
                if !signature.has_kwarg() && signature.num_kwonlyargnames() == 0 {
                    return Err(crate::PyError::type_error(format!(
                        "{}() takes no keyword arguments",
                        fnname,
                    )));
                }
                let msg = ArgErr::UnknownKwds { num_kwds, kwd_name }.getmsg();
                Err(crate::PyError::type_error(format!("{}() {}", fnname, msg)))
            }
            Err(MatchSignatureError::Shape(e)) => Err(crate::PyError::type_error(format!(
                "{}() {}",
                fnname,
                e.getmsg()
            ))),
            Err(MatchSignatureError::Py(e)) => Err(e),
        }
    }

    /// pypy/interpreter/argument.py:385-389 `frompacked`.
    ///
    /// ```python
    /// @staticmethod
    /// def frompacked(space, w_args=None, w_kwds=None):
    ///     """Convenience static method to build an Arguments
    ///        from a wrapped sequence and a wrapped dictionary."""
    ///     return Arguments(space, [], w_stararg=w_args, w_starstararg=w_kwds)
    /// ```
    pub fn frompacked(
        w_args: Option<PyObjectRef>,
        w_kwds: Option<PyObjectRef>,
    ) -> Result<Self, crate::PyError> {
        Self::new(&[], None, None, w_args, w_kwds, false, None)
    }

    /// pypy/interpreter/argument.py:391-400 `topacked`.
    ///
    /// ```python
    /// def topacked(self):
    ///     """Express the Argument object as a pair of wrapped w_args, w_kwds."""
    ///     space = self.space
    ///     w_args = space.newtuple(self.arguments_w)
    ///     w_kwds = space.newdict()
    ///     if self.keyword_names_w is not None:
    ///         for i in range(len(self.keyword_names_w)):
    ///             w_key = self.keyword_names_w[i]
    ///             space.setitem(w_kwds, w_key, self.keywords_w[i])
    ///     return w_args, w_kwds
    /// ```
    pub fn topacked(&self) -> Result<(PyObjectRef, PyObjectRef), crate::PyError> {
        let w_args = pyre_object::w_tuple_new(self.arguments_w.clone());
        let w_kwds = pyre_object::dictobject::w_dict_new();
        if let (Some(names), Some(values)) =
            (self.keyword_names_w.as_ref(), self.keywords_w.as_ref())
        {
            for (w_key, w_value) in names.iter().zip(values.iter()) {
                crate::baseobjspace::setitem(w_kwds, *w_key, *w_value)?;
            }
        }
        Ok((w_args, w_kwds))
    }
}

/// pypy/interpreter/argument.py:172-338 `_match_signature` raises
/// either `ArgErr` (shape mismatch) or `OperationError` (PyError, from
/// `space.setitem` in `_collect_keyword_args`).  Pyre keeps both
/// exception types separate via this enum, so callers
/// (`parse_obj` / `parse_into_scope` in Phase B) can match on the
/// shape arm to reformat as `oefmt(space.w_TypeError, "%s() %8",
/// fnname, e.getmsg())` (argument.py:354) and propagate the Py arm
/// unchanged.
#[derive(Debug, Clone)]
pub enum MatchSignatureError {
    /// `ArgErr*` raised from upstream `_match_signature` /
    /// `_match_keywords`.
    Shape(ArgErr),
    /// `OperationError` raised from `space.setitem` /
    /// `space.finditem_str` etc. (collect_keyword_args fallout).
    Py(crate::PyError),
}

impl From<ArgErr> for MatchSignatureError {
    fn from(e: ArgErr) -> Self {
        MatchSignatureError::Shape(e)
    }
}

impl From<crate::PyError> for MatchSignatureError {
    fn from(e: crate::PyError) -> Self {
        MatchSignatureError::Py(e)
    }
}

/// pypy/interpreter/argument.py:464-501 `_match_keywords`.
///
/// ```python
/// def _match_keywords(space, signature, blindargs, co_posonlyargcount,
///                     input_argcount, keyword_names_w, kwds_mapping, _):
///     num_kwds = num_remainingkwds = len(keyword_names_w)
///     wrong_posonly = None
///     for i in range(num_kwds):
///         w_name = keyword_names_w[i]
///         assert w_name is not None
///         j = signature.find_w_argname(w_name)
///         if 0 <= j < co_posonlyargcount:
///             if signature.has_kwarg():
///                 j = -1
///             else:
///                 if wrong_posonly is None:
///                     wrong_posonly = []
///                 wrong_posonly.append(signature.argnames[j])
///                 continue
///         if j < input_argcount:
///             if blindargs <= j:
///                 raise ArgErrMultipleValues(space.text_w(w_name))
///         else:
///             kwds_mapping[j - input_argcount] = i
///             num_remainingkwds -= 1
///     if wrong_posonly:
///         raise ArgErrPosonlyAsKwds(wrong_posonly)
///     return num_remainingkwds
/// ```
pub fn match_keywords(
    signature: &crate::gateway::Signature,
    blindargs: usize,
    co_posonlyargcount: usize,
    input_argcount: usize,
    keyword_names_w: &[PyObjectRef],
    kwds_mapping: &mut [isize],
) -> Result<usize, MatchSignatureError> {
    let num_kwds = keyword_names_w.len();
    let mut num_remainingkwds = num_kwds;
    let mut wrong_posonly: Option<Vec<String>> = None;
    for i in 0..num_kwds {
        let w_name = keyword_names_w[i];
        debug_assert!(!w_name.is_null(), "keyword_names_w[i] is None");
        let mut j = signature.find_w_argname(w_name);
        // argument.py:474-484 — posonly conflict.
        if j >= 0 && (j as usize) < co_posonlyargcount {
            if signature.has_kwarg() {
                j = -1;
            } else {
                let argname = signature.argnames[j as usize].to_string();
                if let Some(list) = wrong_posonly.as_mut() {
                    list.push(argname);
                } else {
                    wrong_posonly = Some(vec![argname]);
                }
                continue;
            }
        }
        if j < input_argcount as isize {
            // argument.py:485-495 — multiple-values check.  PyPy raises
            // `ArgErrMultipleValues(space.text_w(w_name))`; if `w_name`
            // is non-string (hand-built `Arguments::with_kw`), `text_w`
            // raises a TypeError that propagates through the
            // `_match_signature` caller as `OperationError`.  Pyre
            // mirrors that propagation via `MatchSignatureError::Py`
            // rather than substituting an empty string (baseobjspace.py:
            // 313-315 `_typed_unwrap_error` message).
            if (blindargs as isize) <= j {
                let name = if !w_name.is_null() && unsafe { pyre_object::is_str(w_name) } {
                    unsafe { pyre_object::w_str_get_value(w_name).to_string() }
                } else {
                    let tp = type_name_of(w_name);
                    return Err(MatchSignatureError::Py(crate::PyError::type_error(
                        format!("expected str, got {tp} object"),
                    )));
                };
                return Err(MatchSignatureError::Shape(ArgErr::MultipleValues {
                    argname: name,
                }));
            }
        } else {
            kwds_mapping[(j - input_argcount as isize) as usize] = i as isize;
            num_remainingkwds -= 1;
        }
    }
    if let Some(wrong) = wrong_posonly {
        return Err(MatchSignatureError::Shape(ArgErr::PosonlyAsKwds {
            posonly_kwds: wrong,
        }));
    }
    Ok(num_remainingkwds)
}

/// pypy/interpreter/argument.py:506-516 `_collect_keyword_args`.
///
/// ```python
/// def _collect_keyword_args(space, keyword_names_w, keywords_w, w_kwds,
///                           kwds_mapping, _):
///     for i in range(len(keyword_names_w)):
///         for j in kwds_mapping:
///             if i == j:
///                 break
///         else:
///             w_key = keyword_names_w[i]
///             space.setitem(w_kwds, w_key, keywords_w[i])
/// ```
///
/// `kwds_mapping[k] == i` means slot `k` matched a real argname; this
/// helper walks the kwarg names and forwards every name that did NOT
/// match (i.e. did not appear in `kwds_mapping`) into the `**kwargs`
/// dict via `setitem`.
pub fn collect_keyword_args(
    keyword_names_w: &[PyObjectRef],
    keywords_w: &[PyObjectRef],
    w_kwds: PyObjectRef,
    kwds_mapping: &[isize],
) -> Result<(), crate::PyError> {
    for i in 0..keyword_names_w.len() {
        if mapping_contains(kwds_mapping, i as isize) {
            continue;
        }
        let w_key = keyword_names_w[i];
        crate::baseobjspace::setitem(w_kwds, w_key, keywords_w[i])?;
    }
    Ok(())
}

#[inline]
fn mapping_contains(mapping: &[isize], target: isize) -> bool {
    mapping.iter().any(|&j| j == target)
}

/// pypy/interpreter/argument.py:523-641 — `ArgErr` exception hierarchy.
///
/// PyPy declares `class ArgErr(Exception)` with abstract `getmsg()`
/// and 6 concrete subclasses (`ArgErrMissing`, `ArgErrTooMany`,
/// `ArgErrTooManyMethod`, `ArgErrMultipleValues`, `ArgErrUnknownKwds`,
/// `ArgErrPosonlyAsKwds`).  `_match_signature` / `_parse` / `parse_obj`
/// raise these and the surrounding `try/except ArgErr` arms reformat
/// the message via `oefmt(space.w_TypeError, "%s() %8", fnname,
/// e.getmsg())`.
///
/// Pyre folds the class hierarchy into a Rust enum with payload-only
/// variants — the six error shapes carry exactly the upstream
/// constructor parameters and `getmsg()` runs the per-variant
/// formatting line-by-line.
#[derive(Debug, Clone)]
pub enum ArgErr {
    /// argument.py:529-552 `ArgErrMissing(missing, positional)`.
    Missing {
        missing: Vec<String>,
        positional: bool,
    },
    /// argument.py:555-580 `ArgErrTooMany(signature, num_defaults, given, kwonly_given)`.
    TooMany {
        signature: crate::gateway::Signature,
        num_defaults: usize,
        given: usize,
        kwonly_given: usize,
    },
    /// argument.py:582-595 `ArgErrTooManyMethod` — same fields as
    /// `TooMany`, with appended "did you forget self?" hint when the
    /// signature shape suggests a missing self parameter.
    TooManyMethod {
        signature: crate::gateway::Signature,
        num_defaults: usize,
        given: usize,
        kwonly_given: usize,
    },
    /// argument.py:598-605 `ArgErrMultipleValues(argname)`.
    MultipleValues { argname: String },
    /// argument.py:607-627 `ArgErrUnknownKwds(space, num_remainingkwds,
    /// keyword_names_w, kwds_mapping)`.  PyPy's ctor walks the
    /// keyword_names_w list to extract a single offending name when
    /// `num_remainingkwds == 1`; pyre stores the resolved name string
    /// directly so the formatter can stay free of space refs.
    UnknownKwds { num_kwds: usize, kwd_name: String },
    /// argument.py:630-640 `ArgErrPosonlyAsKwds(posonly_kwds)`.
    PosonlyAsKwds { posonly_kwds: Vec<String> },
}

impl ArgErr {
    /// pypy/interpreter/argument.py:534-552 `ArgErrMissing.getmsg`,
    /// :562-580 `ArgErrTooMany.getmsg`, :589-595
    /// `ArgErrTooManyMethod.getmsg`, :603-605
    /// `ArgErrMultipleValues.getmsg`, :620-627
    /// `ArgErrUnknownKwds.getmsg`, :635-640
    /// `ArgErrPosonlyAsKwds.getmsg` — line-by-line ports.
    pub fn getmsg(&self) -> String {
        match self {
            ArgErr::Missing {
                missing,
                positional,
            } => {
                // argument.py:535-546 — comma-and-or-and join.
                let mut arguments_str = String::new();
                let n = missing.len();
                for (i, arg) in missing.iter().enumerate() {
                    if i == 0 {
                        // empty separator
                    } else if i == n - 1 {
                        if n == 2 {
                            arguments_str.push_str(" and ");
                        } else {
                            arguments_str.push_str(", and ");
                        }
                    } else {
                        arguments_str.push_str(", ");
                    }
                    arguments_str.push('\'');
                    arguments_str.push_str(arg);
                    arguments_str.push('\'');
                }
                let kind_str = if *positional {
                    "positional"
                } else {
                    "keyword-only"
                };
                let plural = if n != 1 { "s" } else { "" };
                format!("missing {n} required {kind_str} argument{plural}: {arguments_str}",)
            }
            ArgErr::TooMany {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => format_too_many(signature, *num_defaults, *given, *kwonly_given),
            ArgErr::TooManyMethod {
                signature,
                num_defaults,
                given,
                kwonly_given,
            } => {
                let mut msg = format_too_many(signature, *num_defaults, *given, *kwonly_given);
                let n = signature.num_argnames();
                // argument.py:592-594 — "did you forget self?" hint.
                if *given == n + 1 && (n == 0 || signature.argnames[0] != "self") {
                    msg.push_str(". Did you forget 'self' in the function definition?");
                }
                msg
            }
            ArgErr::MultipleValues { argname } => {
                format!("got multiple values for argument '{argname}'")
            }
            ArgErr::UnknownKwds { num_kwds, kwd_name } => {
                if *num_kwds == 1 {
                    format!("got an unexpected keyword argument '{kwd_name}'")
                } else {
                    format!("got {num_kwds} unexpected keyword arguments")
                }
            }
            ArgErr::PosonlyAsKwds { posonly_kwds } => {
                if posonly_kwds.len() == 1 {
                    format!(
                        "got a positional-only argument passed as keyword argument: '{}'",
                        posonly_kwds[0],
                    )
                } else {
                    format!(
                        "got some positional-only arguments passed as keyword arguments: '{}'",
                        posonly_kwds.join(", "),
                    )
                }
            }
        }
    }
}

/// pypy/interpreter/argument.py:562-580 `ArgErrTooMany.getmsg` body.
/// Extracted because both `TooMany` and `TooManyMethod` arms share it.
fn format_too_many(
    signature: &crate::gateway::Signature,
    num_defaults: usize,
    given: usize,
    kwonly_given: usize,
) -> String {
    let num_args = signature.num_argnames();
    let takes_str = if num_defaults > 0 {
        format!(
            "from {} to {} positional arguments",
            num_args - num_defaults,
            num_args,
        )
    } else {
        let plural = if num_args != 1 { "s" } else { "" };
        format!("{num_args} positional argument{plural}")
    };
    let given_str = if kwonly_given > 0 {
        let pos_plural = if given != 1 { "s" } else { "" };
        let kw_plural = if kwonly_given != 1 { "s" } else { "" };
        format!(
            "{given} positional argument{pos_plural} (and {kwonly_given} keyword-only argument{kw_plural}) were",
        )
    } else {
        let verb = if given != 1 { "were" } else { "was" };
        format!("{given} {verb}")
    };
    format!("takes {takes_str} but {given_str} given")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// pypy/interpreter/argument.py:14-15 — `w_function is None` arm.
    #[test]
    fn raise_type_error_no_function() {
        let err = raise_type_error(pyre_object::PY_NULL, "boom".to_string());
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(err.message, "boom");
    }

    /// pypy/interpreter/argument.py:16-17 — function-prefixed arm.
    /// With an int as `w_function` the prefix falls back to `str(int) ()`
    /// per `object_functionstr`'s scalar fallback path.  The fallback
    /// invokes `space.str(w_int)` which dispatches to `int.__str__`
    /// via type-MRO `lookup`; that lookup needs `init_typeobjects()`
    /// to have populated the W_TypeObject MRO.
    #[test]
    fn raise_type_error_with_function() {
        crate::typedef::init_typeobjects();
        let func = pyre_object::w_int_new(7);
        let err = raise_type_error(func, "needs an iterable".to_string());
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        // object_functionstr scalar fallback returns the str() of the
        // value, here "7"; raise_type_error joins it with the message.
        assert_eq!(err.message, "7 needs an iterable");
    }

    /// pypy/interpreter/argument.py:534-552 single-missing positional case.
    #[test]
    fn arg_err_missing_one_positional() {
        let err = ArgErr::Missing {
            missing: vec!["x".to_string()],
            positional: true,
        };
        assert_eq!(err.getmsg(), "missing 1 required positional argument: 'x'",);
    }

    /// pypy/interpreter/argument.py:540-541 — 2-missing uses ` and `.
    #[test]
    fn arg_err_missing_two_keyword_only() {
        let err = ArgErr::Missing {
            missing: vec!["a".to_string(), "b".to_string()],
            positional: false,
        };
        assert_eq!(
            err.getmsg(),
            "missing 2 required keyword-only arguments: 'a' and 'b'",
        );
    }

    /// pypy/interpreter/argument.py:543 — 3-or-more uses `, and `.
    #[test]
    fn arg_err_missing_three_with_oxford_comma() {
        let err = ArgErr::Missing {
            missing: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            positional: true,
        };
        assert_eq!(
            err.getmsg(),
            "missing 3 required positional arguments: 'a', 'b', and 'c'",
        );
    }

    /// pypy/interpreter/argument.py:603-605 `ArgErrMultipleValues`.
    #[test]
    fn arg_err_multiple_values() {
        let err = ArgErr::MultipleValues {
            argname: "kwarg".to_string(),
        };
        assert_eq!(err.getmsg(), "got multiple values for argument 'kwarg'");
    }

    /// pypy/interpreter/argument.py:621-626 `ArgErrUnknownKwds`
    /// — single-name vs multi-count branches.
    #[test]
    fn arg_err_unknown_kwds_branches() {
        let one = ArgErr::UnknownKwds {
            num_kwds: 1,
            kwd_name: "wibble".to_string(),
        };
        assert_eq!(one.getmsg(), "got an unexpected keyword argument 'wibble'");
        let many = ArgErr::UnknownKwds {
            num_kwds: 3,
            kwd_name: String::new(),
        };
        assert_eq!(many.getmsg(), "got 3 unexpected keyword arguments");
    }

    /// pypy/interpreter/argument.py:635-640 `ArgErrPosonlyAsKwds`
    /// — single vs multi-name plural branches.
    #[test]
    fn arg_err_posonly_as_kwds_branches() {
        let one = ArgErr::PosonlyAsKwds {
            posonly_kwds: vec!["a".to_string()],
        };
        assert_eq!(
            one.getmsg(),
            "got a positional-only argument passed as keyword argument: 'a'",
        );
        let many = ArgErr::PosonlyAsKwds {
            posonly_kwds: vec!["x".to_string(), "y".to_string()],
        };
        assert_eq!(
            many.getmsg(),
            "got some positional-only arguments passed as keyword arguments: 'x, y'",
        );
    }

    /// pypy/interpreter/argument.py:562-580 `ArgErrTooMany.getmsg` —
    /// no defaults, single-arg singular path.
    #[test]
    fn arg_err_too_many_no_defaults_singular() {
        let sig = crate::gateway::Signature::new(vec!["x"], None, None, 0, 0);
        let err = ArgErr::TooMany {
            signature: sig,
            num_defaults: 0,
            given: 2,
            kwonly_given: 0,
        };
        assert_eq!(err.getmsg(), "takes 1 positional argument but 2 were given");
    }

    /// pypy/interpreter/argument.py:565-567 — defaults branch produces
    /// "from X to Y positional arguments".
    #[test]
    fn arg_err_too_many_with_defaults() {
        let sig = crate::gateway::Signature::new(vec!["a", "b", "c"], None, None, 0, 0);
        let err = ArgErr::TooMany {
            signature: sig,
            num_defaults: 1,
            given: 4,
            kwonly_given: 0,
        };
        assert_eq!(
            err.getmsg(),
            "takes from 2 to 3 positional arguments but 4 were given",
        );
    }

    /// pypy/interpreter/argument.py:589-595 `ArgErrTooManyMethod`
    /// appends self-hint when shape matches and argnames[0] != "self".
    #[test]
    fn arg_err_too_many_method_self_hint_appended() {
        let sig = crate::gateway::Signature::new(vec!["x"], None, None, 0, 0);
        let err = ArgErr::TooManyMethod {
            signature: sig,
            num_defaults: 0,
            given: 2,
            kwonly_given: 0,
        };
        let msg = err.getmsg();
        assert!(msg.starts_with("takes 1 positional argument but 2 were given"));
        assert!(msg.contains("Did you forget 'self'"));
    }

    /// pypy/interpreter/argument.py:592-593 — when argnames[0] == "self"
    /// the hint is suppressed.
    #[test]
    fn arg_err_too_many_method_self_hint_suppressed() {
        let sig = crate::gateway::Signature::new(vec!["self", "x"], None, None, 0, 0);
        let err = ArgErr::TooManyMethod {
            signature: sig,
            num_defaults: 0,
            given: 3,
            kwonly_given: 0,
        };
        let msg = err.getmsg();
        assert!(!msg.contains("Did you forget"));
    }

    /// pypy/interpreter/argument.py:419-423 `contains_w_names` —
    /// string equality match.
    #[test]
    fn contains_w_names_string_match() {
        let names = vec![pyre_object::w_str_new("a"), pyre_object::w_str_new("b")];
        assert!(contains_w_names(pyre_object::w_str_new("a"), &names));
        assert!(contains_w_names(pyre_object::w_str_new("b"), &names));
        assert!(!contains_w_names(pyre_object::w_str_new("c"), &names));
    }

    /// pypy/interpreter/argument.py:410-417 `_check_not_duplicate_kwargs` —
    /// raises TypeError on overlap.
    #[test]
    fn check_not_duplicate_kwargs_raises_on_overlap() {
        let existing = vec![pyre_object::w_str_new("a")];
        let new = vec![pyre_object::w_str_new("a")];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        let err = check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect_err("should raise TypeError on duplicate");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message.contains("got multiple values"));
        assert!(err.message.contains("'a'"));
    }

    /// `check_not_duplicate_kwargs` accepts disjoint name sets.
    #[test]
    fn check_not_duplicate_kwargs_disjoint_ok() {
        let existing = vec![pyre_object::w_str_new("a")];
        let new = vec![pyre_object::w_str_new("b")];
        let values: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        check_not_duplicate_kwargs(&existing, &new, &values, pyre_object::PY_NULL)
            .expect("disjoint names should pass");
    }

    /// pypy/interpreter/argument.py:92-104 `_combine_starargs_wrapped`:
    /// list extension via fixedview.
    #[test]
    fn combine_starargs_wrapped_extends_from_list() {
        let mut args: Vec<PyObjectRef> = vec![pyre_object::w_int_new(1)];
        let stararg =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)]);
        combine_starargs_wrapped(&mut args, stararg, pyre_object::PY_NULL)
            .expect("list star arg should expand");
        assert_eq!(args.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(args[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(args[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(args[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py:97-103 — non-iterable star arg
    /// surfaces "argument after * must be an iterable".
    #[test]
    fn combine_starargs_wrapped_non_iterable_raises() {
        let mut args: Vec<PyObjectRef> = vec![];
        let stararg = pyre_object::w_int_new(42); // ints aren't iterable
        let err = combine_starargs_wrapped(&mut args, stararg, pyre_object::PY_NULL)
            .expect_err("int star arg should raise TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message.contains("argument after * must be an iterable"));
    }

    /// pypy/interpreter/argument.py:172-338 `match_signature` happy
    /// path: 2 positional args fill `def f(a, b): pass`.
    #[test]
    fn match_signature_positional_pass_through() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let args = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("happy path should pass");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(scope[1]), 2);
        }
    }

    /// pypy/interpreter/argument.py:289-300 `match_signature` raises
    /// `ArgErrTooMany` when avail > co_argcount.
    #[test]
    fn match_signature_too_many_args_raises() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let args = [
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("too many args should ArgErr");
        match err {
            MatchSignatureError::Shape(ArgErr::TooMany { given, .. }) => {
                assert_eq!(given, 3);
            }
            other => panic!("expected TooMany, got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py:335-336 `match_signature` raises
    /// `ArgErrMissing` for missing positional arguments.
    #[test]
    fn match_signature_missing_positional_raises() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let args = [pyre_object::w_int_new(1)];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("missing positional should ArgErr");
        match err {
            MatchSignatureError::Shape(ArgErr::Missing {
                missing,
                positional,
            }) => {
                assert_eq!(missing, vec!["b"]);
                assert!(positional);
            }
            other => panic!("expected Missing, got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py:223-234 `match_signature` collects
    /// extra positional args into the *vararg slot.
    #[test]
    fn match_signature_collects_vararg() {
        let sig = crate::gateway::Signature::new(vec!["a"], Some("rest"), None, 0, 0);
        let args = [
            pyre_object::w_int_new(1),
            pyre_object::w_int_new(2),
            pyre_object::w_int_new(3),
        ];
        let arguments = Arguments::positional_only(&args);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        arguments
            .match_signature(
                pyre_object::PY_NULL,
                &mut scope,
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect("vararg path should pass");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            // vararg lives at index 1 (after the named args).
            assert_eq!(pyre_object::w_tuple_len(scope[1]), 2);
        }
    }

    /// pypy/interpreter/argument.py:464-501 `match_keywords` matches
    /// keyword names against argnames + writes mapping.
    #[test]
    fn match_keywords_assigns_index() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let names = [pyre_object::w_str_new("b")];
        let mut mapping = vec![-1isize; 2];
        let remaining = match_keywords(&sig, 0, 0, 0, &names, &mut mapping)
            .expect("kwarg name match should succeed");
        assert_eq!(remaining, 0);
        assert_eq!(mapping, vec![-1, 0]); // 'b' is signature index 1, slot 1 (= 1 - input_argcount=0)
    }

    /// pypy/interpreter/argument.py:474-484 `match_keywords` flags
    /// posonly conflict via `ArgErrPosonlyAsKwds`.
    #[test]
    fn match_keywords_posonly_conflict() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 1);
        let names = [pyre_object::w_str_new("a")];
        let mut mapping = vec![-1isize; 2];
        let err = match_keywords(&sig, 0, 1, 0, &names, &mut mapping)
            .expect_err("posonly kwarg should raise");
        match err {
            MatchSignatureError::Shape(ArgErr::PosonlyAsKwds { posonly_kwds }) => {
                assert_eq!(posonly_kwds, vec!["a"]);
            }
            other => panic!("expected Shape(PosonlyAsKwds), got {other:?}"),
        }
    }

    /// pypy/interpreter/argument.py:106-150 `_combine_starstarargs_wrapped`:
    /// dict expansion fills parallel name/value buffers.
    #[test]
    fn combine_starstarargs_wrapped_dict_expansion() {
        let mut names: Vec<PyObjectRef> = vec![];
        let mut values: Vec<PyObjectRef> = vec![];
        let dict = pyre_object::dictobject::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(dict, "x", pyre_object::w_int_new(10));
            pyre_object::w_dict_setitem_str(dict, "y", pyre_object::w_int_new(20));
        }
        combine_starstarargs_wrapped(&mut names, &mut values, dict, pyre_object::PY_NULL)
            .expect("dict expansion should succeed");
        assert_eq!(names.len(), 2);
        assert_eq!(values.len(), 2);
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` with `w_stararg`:
    /// star-arg list extends `arguments_w` in-place (Phase A.3 routes
    /// through `combine_starargs_wrapped`).
    #[test]
    fn new_with_w_stararg_extends_arguments_w() {
        let pos = [pyre_object::w_int_new(1)];
        let stararg =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)]);
        let arguments = Arguments::new(&pos, None, None, Some(stararg), None, false, None)
            .expect("star expansion should succeed");
        assert_eq!(arguments.arguments_w.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(arguments.arguments_w[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` with `w_starstararg`:
    /// star-star dict expands into the kwargs Vecs (Phase A.3 routes
    /// through `combine_starstarargs_wrapped`).
    #[test]
    fn new_with_w_starstararg_fills_kwargs() {
        let pos = [pyre_object::w_int_new(1)];
        let dict = pyre_object::dictobject::w_dict_new();
        unsafe {
            pyre_object::w_dict_setitem_str(dict, "k", pyre_object::w_int_new(99));
        }
        let arguments = Arguments::new(&pos, None, None, None, Some(dict), false, None)
            .expect("starstar expansion should succeed");
        assert_eq!(arguments.arguments_w.len(), 1);
        let names = arguments.keyword_names_w.expect("kwargs should be Some");
        let values = arguments.keywords_w.expect("kwargs should be Some");
        assert_eq!(names.len(), 1);
        assert_eq!(values.len(), 1);
        unsafe {
            assert_eq!(pyre_object::w_str_get_value(names[0]), "k");
            assert_eq!(pyre_object::w_int_get_value(values[0]), 99);
        }
    }

    /// pypy/interpreter/argument.py:97-103 — non-iterable `*` arg
    /// surfaces as `Err(PyError)` from `Arguments::new` rather than the
    /// previous `unimplemented!()` panic.
    #[test]
    fn new_with_w_stararg_non_iterable_returns_err() {
        let pos: [PyObjectRef; 0] = [];
        let stararg = pyre_object::w_int_new(42); // not iterable
        match Arguments::new(&pos, None, None, Some(stararg), None, false, None) {
            Ok(_) => panic!("non-iterable star arg should fail"),
            Err(err) => {
                assert_eq!(err.kind, crate::PyErrorKind::TypeError);
                assert!(err.message.contains("argument after * must be an iterable"));
            }
        }
    }

    /// pypy/interpreter/argument.py:77-79 `replace_arguments` —
    /// returns a new Arguments with replaced positional list,
    /// keyword names/values shared (cloned in pyre).
    #[test]
    fn replace_arguments_preserves_kwargs() {
        let pos = [pyre_object::w_int_new(1)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let new_pos = vec![pyre_object::w_int_new(2), pyre_object::w_int_new(3)];
        let replaced = arguments.replace_arguments(new_pos);
        assert_eq!(replaced.arguments_w.len(), 2);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(replaced.arguments_w[0]), 2);
            assert_eq!(pyre_object::w_int_get_value(replaced.arguments_w[1]), 3);
        }
        let names = replaced.keyword_names_w.expect("kwargs preserved");
        assert_eq!(names.len(), 1);
    }

    /// pypy/interpreter/argument.py:81-83 `prepend` — `[w_firstarg] + args_w`.
    #[test]
    fn prepend_inserts_at_front() {
        let pos = [pyre_object::w_int_new(2), pyre_object::w_int_new(3)];
        let arguments = Arguments::positional_only(&pos);
        let prepended = arguments.prepend(pyre_object::w_int_new(1));
        assert_eq!(prepended.arguments_w.len(), 3);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[1]), 2);
            assert_eq!(pyre_object::w_int_get_value(prepended.arguments_w[2]), 3);
        }
    }

    /// pypy/interpreter/argument.py:68-75 `unpack` — returns
    /// (positional list, kwds dict).
    #[test]
    fn unpack_returns_args_and_kwds() {
        let pos = [pyre_object::w_int_new(10)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (args_w, kwds_w) = arguments.unpack().expect("string keys are valid");
        assert_eq!(args_w.len(), 1);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(args_w[0]), 10);
        }
        assert_eq!(kwds_w.len(), 1);
        let v = kwds_w.get("k").expect("key 'k' present");
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(*v), 99);
        }
    }

    /// pypy/interpreter/argument.py:71-74 — `kwds_w[key] = value`
    /// overwrites on duplicate (RPython interp-level dict semantics).
    /// Hand-built duplicate keyword names collapse to the last value,
    /// matching PyPy's behaviour on the same hand-built input.
    #[test]
    fn unpack_overwrites_duplicate_keys() {
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_str_new("dup"), pyre_object::w_str_new("dup")];
        let values = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (_, kwds_w) = arguments.unpack().expect("string keys are valid");
        assert_eq!(kwds_w.len(), 1, "duplicate keys must collapse");
        let v = kwds_w.get("dup").expect("key 'dup' present");
        unsafe {
            // last-write-wins → expected value 2 (not 1).
            assert_eq!(pyre_object::w_int_get_value(*v), 2);
        }
    }

    /// pypy/interpreter/argument.py:74 `space.text_w(w_name)` raises
    /// TypeError when the keyword name is not a string; `unpack`
    /// propagates.
    #[test]
    fn unpack_non_string_key_returns_err() {
        let pos: [PyObjectRef; 0] = [];
        let names = [pyre_object::w_int_new(7)];
        let values = [pyre_object::w_int_new(42)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        match arguments.unpack() {
            Ok(_) => panic!("non-string keyword name should TypeError"),
            Err(err) => {
                assert_eq!(err.kind, crate::PyErrorKind::TypeError);
                // baseobjspace.py:313-315 `_typed_unwrap_error`:
                // `expected str, got <T> object`.
                assert!(err.message.contains("expected str"));
            }
        }
    }

    /// pypy/interpreter/argument.py:341-355 `parse_into_scope` happy
    /// path: matches signature, returns scope_length.
    #[test]
    fn parse_into_scope_happy_path() {
        let sig = crate::gateway::Signature::new(vec!["a", "b"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&pos);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let scopelen = arguments
            .parse_into_scope(
                pyre_object::PY_NULL,
                &mut scope,
                "f",
                &sig,
                None,
                pyre_object::PY_NULL,
            )
            .expect("happy path should succeed");
        assert_eq!(scopelen, 2);
        unsafe {
            assert_eq!(pyre_object::w_int_get_value(scope[0]), 1);
            assert_eq!(pyre_object::w_int_get_value(scope[1]), 2);
        }
    }

    /// pypy/interpreter/argument.py:341-355 `parse_into_scope` shape
    /// mismatch surfaces as TypeError with `<fnname>() ` prefix.
    #[test]
    fn parse_into_scope_shape_error_formats_fnname_prefix() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let arguments = Arguments::positional_only(&pos);
        let mut scope: Vec<PyObjectRef> = vec![pyre_object::PY_NULL; sig.scope_length()];
        let err = arguments
            .parse_into_scope(
                pyre_object::PY_NULL,
                &mut scope,
                "myfn",
                &sig,
                None,
                pyre_object::PY_NULL,
            )
            .expect_err("too many args should TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert!(err.message.starts_with("myfn() "));
    }

    /// pypy/interpreter/argument.py:378-380 `parse_obj` special-cases
    /// "takes no keyword arguments" when no **kwarg + no kwonly args.
    #[test]
    fn parse_obj_unknown_kwds_no_kwarg_message() {
        let sig = crate::gateway::Signature::new(vec!["a"], None, None, 0, 0);
        let pos = [pyre_object::w_int_new(1)];
        let names = [pyre_object::w_str_new("wibble")];
        let values = [pyre_object::w_int_new(2)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let err = arguments
            .parse_obj(
                pyre_object::PY_NULL,
                "myfn",
                &sig,
                None,
                pyre_object::PY_NULL,
                0,
            )
            .expect_err("unknown kwarg should TypeError");
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(err.message, "myfn() takes no keyword arguments");
    }

    /// pypy/interpreter/argument.py:385-389 `frompacked` builds
    /// Arguments from packed star args.
    #[test]
    fn frompacked_expands_w_args() {
        let w_args =
            pyre_object::w_list_new(vec![pyre_object::w_int_new(1), pyre_object::w_int_new(2)]);
        let arguments = Arguments::frompacked(Some(w_args), None).expect("frompacked");
        assert_eq!(arguments.arguments_w.len(), 2);
    }

    /// pypy/interpreter/argument.py:391-400 `topacked` round-trips
    /// arguments_w through newtuple + newdict.
    #[test]
    fn topacked_round_trips() {
        let pos = [pyre_object::w_int_new(1), pyre_object::w_int_new(2)];
        let names = [pyre_object::w_str_new("k")];
        let values = [pyre_object::w_int_new(99)];
        let arguments = Arguments::with_kw(&pos, &names, &values);
        let (w_args, w_kwds) = arguments.topacked().expect("topacked");
        unsafe {
            assert_eq!(pyre_object::w_tuple_len(w_args), 2);
            assert!(pyre_object::is_dict(w_kwds));
        }
    }
}
