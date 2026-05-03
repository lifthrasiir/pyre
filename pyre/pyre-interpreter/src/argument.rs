//! Arguments objects.
//!
//! Line-by-line port of `pypy/interpreter/argument.py:Arguments`.
//! Step 1 of the Arguments port: defines the struct + `firstarg`
//! reader, which is the only consumer reached from the trace path
//! (`executioncontext._c_call_return_trace`'s FunctionWithFixedCode
//! rebinding at executioncontext.py:130).
//!
//! Pyre's legacy `call::call_callable` surface still takes a flat
//! `&[PyObjectRef]`.  Callers that know the keyword layout route
//! through `Arguments::with_kw` before entering the profiled-builtin
//! path; callers with only a flat positional slice use
//! `Arguments::positional_only`.

use pyre_object::PyObjectRef;

/// `pypy/interpreter/argument.py:20 class Arguments`.
///
/// PyPy fields (argument.py:34-46):
/// ```text
/// self.space            -- always available; pyre passes context implicitly
/// self.arguments_w      -- list[w_obj]
/// self.keyword_names_w  -- list[w_text] or None
/// self.keywords_w       -- list[w_obj]   or None  (parallel to keyword_names_w)
/// ```
///
/// `w_stararg`, `w_starstararg`, `methodcall`, `w_function` are constructor
/// inputs that PyPy's `_combine_wrapped` expands into `arguments_w` /
/// `keyword_names_w` / `keywords_w`.  Pyre's call surface already
/// pre-resolves them at `call.rs:711-756`, so the struct stores only
/// the resolved fields.
///
/// Borrows from the caller's slice; Arguments is short-lived (passes
/// through a single trace event call) and does not own its data.
pub struct Arguments<'a> {
    /// argument.py:36 `self.arguments_w = args_w`.
    pub arguments_w: &'a [PyObjectRef],
    /// argument.py:38 `self.keyword_names_w = keyword_names_w` (`None` allowed).
    pub keyword_names_w: Option<&'a [PyObjectRef]>,
    /// argument.py:39 `self.keywords_w = keywords_w` (`None` allowed,
    /// must be parallel to `keyword_names_w` when present —
    /// argument.py:42 `assert len(keywords_w) == len(keyword_names_w)`).
    pub keywords_w: Option<&'a [PyObjectRef]>,
}

impl<'a> Arguments<'a> {
    /// argument.py:31-53 `__init__` (positional-only construction).
    ///
    /// `_combine_wrapped` is folded into the caller; pyre call sites
    /// supply `arguments_w` already resolved.  Used for call surfaces
    /// that have only positional args (no kwargs).
    #[inline]
    pub fn positional_only(args_w: &'a [PyObjectRef]) -> Self {
        Self {
            arguments_w: args_w,
            keyword_names_w: None,
            keywords_w: None,
        }
    }

    /// argument.py:31-53 `__init__` (positional + kwargs construction).
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
        // argument.py:42 invariant.
        debug_assert_eq!(keyword_names_w.len(), keywords_w.len());
        Self {
            arguments_w: args_w,
            keyword_names_w: Some(keyword_names_w),
            keywords_w: Some(keywords_w),
        }
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
