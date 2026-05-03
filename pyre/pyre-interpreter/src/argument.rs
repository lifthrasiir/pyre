//! Arguments objects.
//!
//! Line-by-line port of `pypy/interpreter/argument.py:Arguments`.
//! The struct + `new` (PyPy `__init__`) + `firstarg` + `_combine_wrapped`
//! + `_jit_few_keywords` + `fixedunpack` are ported in this slice.
//! `_match_signature`, `parse_into_scope`, `unpack`, `replace_arguments`,
//! `prepend` remain PRE-EXISTING-ADAPTATIONs pending downstream
//! consumers (and a Vec-owning rework — pyre's borrowed-slice shape
//! cannot return new `Arguments` instances by value, see the lifetime
//! note on the struct).
//!
//! Pyre's legacy `call::call_callable` surface still takes a flat
//! `&[PyObjectRef]`.  Callers that know the keyword layout route
//! through `Arguments::with_kw` before entering the profiled-builtin
//! path; callers with only a flat positional slice use
//! `Arguments::positional_only`.  Both shortcuts now delegate to `new`
//! so the PyPy `__init__` invariant chain runs once.

use pyre_object::PyObjectRef;

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
/// Borrows from the caller's slice; Arguments is short-lived (passes
/// through a single trace event call) and does not own its data.
/// PRE-EXISTING-ADAPTATION vs PyPy: `Arguments(self.space, args_w,
/// ...)` constructs a fresh Vec-owning instance; pyre returns
/// `Self<'a>` borrowing the caller's buffers, so methods that
/// allocate a new `Arguments` (`replace_arguments`, `prepend`) cannot
/// be ported until the call sites are reworked to own their slices.
pub struct Arguments<'a> {
    /// argument.py:36 `self.arguments_w = args_w`.
    pub arguments_w: &'a [PyObjectRef],
    /// argument.py:38 `self.keyword_names_w = keyword_names_w` (`None` allowed).
    pub keyword_names_w: Option<&'a [PyObjectRef]>,
    /// argument.py:39 `self.keywords_w = keywords_w` (`None` allowed,
    /// must be parallel to `keyword_names_w` when present —
    /// argument.py:42 `assert len(keywords_w) == len(keyword_names_w)`).
    pub keywords_w: Option<&'a [PyObjectRef]>,
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

impl<'a> Arguments<'a> {
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
    /// `_combine_wrapped(w_stararg, w_starstararg, w_function)` is
    /// invoked when star args are passed; today both must be `None`
    /// because the helpers it depends on (`space.fixedview`,
    /// `space.view_as_kwargs`, `space.unpackiterable`,
    /// `space.is_iterable`, `space.call_method`) form a port slice
    /// that has not landed.  Until then callers must run the
    /// equivalent expansion themselves and pass already-resolved
    /// `args_w` / `keyword_names_w` / `keywords_w`.
    #[inline]
    pub fn new(
        args_w: &'a [PyObjectRef],
        keyword_names_w: Option<&'a [PyObjectRef]>,
        keywords_w: Option<&'a [PyObjectRef]>,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
        methodcall: bool,
        _w_function: Option<PyObjectRef>,
    ) -> Self {
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
            arguments_w: args_w,
            keyword_names_w,
            keywords_w,
            // argument.py:50 — jit_few_keywords initial guess; the
            // post-`_combine_wrapped` value below is the canonical one.
            jit_few_keywords: keyword_names_w.is_none(),
            methodcall,
        };
        arguments._combine_wrapped(w_stararg, w_starstararg);
        // argument.py:50 — recompute after `_combine_wrapped`, since
        // that helper may have grown `keyword_names_w`.
        arguments.jit_few_keywords = match arguments.keyword_names_w {
            None => true,
            Some(names) => names.len() <= 8, // pyre approximation of jit.isconstant
        };
        arguments
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
    /// PRE-EXISTING-ADAPTATION: pyre's borrowed-slice shape cannot grow
    /// `arguments_w` in place (the slice is the caller's buffer).
    /// `_combine_starargs_wrapped` (argument.py:92-104) and
    /// `_combine_starstarargs_wrapped` (argument.py:106-150) require
    /// ownership of an extending Vec plus space-level helpers
    /// (`fixedview`, `view_as_kwargs`, `unpackiterable`, `is_iterable`,
    /// `call_method`, `findattr`).  Until pyre grows the
    /// owned-Vec variant of `Arguments`, callers must hand-roll the
    /// star expansion before calling `Arguments::new` and pass
    /// `w_stararg=None` / `w_starstararg=None`.
    fn _combine_wrapped(
        &mut self,
        w_stararg: Option<PyObjectRef>,
        w_starstararg: Option<PyObjectRef>,
    ) {
        if w_stararg.is_some() {
            unimplemented!(
                "_combine_starargs_wrapped pending owned-Vec Arguments + space.fixedview port"
            );
        }
        if w_starstararg.is_some() {
            unimplemented!(
                "_combine_starstarargs_wrapped pending owned-Vec Arguments + space.view_as_kwargs port"
            );
        }
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` (positional-only shortcut).
    ///
    /// `_combine_wrapped` is folded into the caller; pyre call sites
    /// supply `arguments_w` already resolved.  Used for call surfaces
    /// that have only positional args (no kwargs).
    #[inline]
    pub fn positional_only(args_w: &'a [PyObjectRef]) -> Self {
        Self::new(args_w, None, None, None, None, false, None)
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
    /// otherwise appends to the merged slice.
    #[inline]
    pub fn with_kw(
        args_w: &'a [PyObjectRef],
        keyword_names_w: &'a [PyObjectRef],
        keywords_w: &'a [PyObjectRef],
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
    }

    /// pypy/interpreter/argument.py:31-53 `__init__` shortcut with the
    /// `methodcall` flag explicit.  Use when the caller has both kwargs
    /// and the `methodcall` flag (e.g. CALL_METHOD lowering); when
    /// methodcall is false, `with_kw` is the lighter alternative.
    #[inline]
    pub fn full(
        args_w: &'a [PyObjectRef],
        keyword_names_w: Option<&'a [PyObjectRef]>,
        keywords_w: Option<&'a [PyObjectRef]>,
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
    /// reference return).
    pub fn fixedunpack(&self, argcount: usize) -> Result<&'a [PyObjectRef], &'static str> {
        if self.keyword_names_w.map(|s| !s.is_empty()).unwrap_or(false) {
            return Err("no keyword arguments expected");
        }
        if self.arguments_w.len() > argcount {
            return Err("too many arguments");
        }
        if self.arguments_w.len() < argcount {
            return Err("not enough arguments");
        }
        Ok(self.arguments_w)
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
}
