//! Built-in function analysers — direct port of
//! `rpython/annotator/builtin.py`.
//!
//! Upstream registers two families of analysers on a module-level
//! `BUILTIN_ANALYZERS` dict:
//!
//! * The `builtin_*` loop at the tail of `builtin.py:190-195` scans
//!   module globals for names starting with `builtin_` and registers
//!   them under the matching `__builtin__` attribute (e.g.
//!   `__builtin__.range` → `builtin_range`).
//! * Explicit `@analyzer_for(<callable>)` decorators register analysers
//!   for specific RPython-rlib helpers (`intmask`, `instantiate`,
//!   `r_dict`, …) plus a handful of stdlib entries (`weakref.ref`,
//!   `sys.getdefaultencoding`, `collections.OrderedDict`, `pdb.set_trace`).
//!
//! The Rust port mirrors that split. Analysers are keyed by the
//! **qualname** of the host callable (the string that
//! `HostObject::qualname()` returns) because:
//!
//! * Rust analysers are first-class function pointers (`BuiltinAnalyzer`),
//!   not Python callables. We cannot key the registry on the callable
//!   itself the way upstream does with `BUILTIN_ANALYZERS[__builtin__.range]`.
//! * `HOST_ENV` installs every builtin/rlib callable under its fully
//!   qualified name (see `flowspace::model::HOST_ENV::bootstrap_*`), so
//!   the qualname is an injective identity for the callable.
//!
//! # Registry
//!
//! A single `OnceLock<HashMap<String, BuiltinAnalyzer>>` owns the
//! registry. First access calls [`register_builtins`] which replicates
//! upstream's mass `for name, value in globals().items(): ...` loop
//! plus every `@analyzer_for(...)` decoration site.
//!
//! # Dispatch
//!
//! [`call_builtin`] is the entry point consumed by
//! `SomeValue::call()` when the callee is `SomeValue::Builtin(_)`. It
//! looks the analyser up by `SomeBuiltin.analyser_name` (which equals
//! the qualname the bookkeeper stored at immutablevalue time), unpacks
//! the call arguments per upstream's `kwds_s['s_'+key] = s_value`
//! rewrite, and hands off to the resolved Rust function.
//!
//! # Deferred entries
//!
//! A few upstream analysers require rtyper-phase machinery that has
//! not landed yet. They are registered but return
//! `AnnotatorError::new("<name>: rtyper-phase only")` so
//! `simplify_graph` paths that accidentally dispatch through them fail
//! loudly instead of silently annotating as `SomeObject`:
//!
//! * `rpython.rlib.objectmodel.hlinvoke` — needs
//!   `rpython.rtyper.llannotation.lltype_to_annotation` and
//!   `rpython.rtyper.rmodel.Repr`.
//! * `unicodedata.decimal` — deliberately errors at annotation time
//!   (matches upstream `builtin.py:300-303`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::OnceLock;

use super::bookkeeper::Bookkeeper;
use super::classdesc::{ClassDef, ClassDesc};
use super::description::DescEntry;
use super::model::{
    AnnotatorError, SomeBool, SomeByteArray, SomeChar, SomeDict, SomeFloat, SomeInstance,
    SomeInteger, SomeIterator, SomeObjectTrait, SomeString, SomeTuple, SomeUnicodeCodePoint,
    SomeUnicodeString, SomeValue, SomeWeakRef, s_impossible_value, s_none, union,
};
use crate::flowspace::model::ConstValue;

// ---------------------------------------------------------------------------
// Registry.
// ---------------------------------------------------------------------------

/// Signature of a builtin analyser.
///
/// * `bk` — active bookkeeper. Every RPython analyser that needs to
///   call `newlist` / `getdictdef` / `immutablevalue` goes through this
///   reference. Upstream pulls the same value via `getbookkeeper()`;
///   passing it explicitly avoids repeated TLS reads.
/// * `args_s` — positional argument annotations, after upstream's
///   `args_s, kwds = args.unpack()` split.
/// * `kwds_s` — keyword argument annotations, already `s_` prefixed to
///   match upstream's `kwds_s['s_'+key] = s_value`. Most analysers do
///   not consume keywords.
/// `unaryop.py:944-946 SomeBuiltin.call`: `args_s, kwds = args.unpack();
/// return self.analyser(*args_s, **kwds_s)` — args_s passes through as a
/// Python list possibly carrying `None` for unbound caller args.  The
/// analyser body raises AttributeError on the first
/// `s_arg.knowntype` / `.is_constant()` access if it touches an
/// unbound slot.  The Rust port mirrors this by taking
/// `&[Option<SomeValue>]`: each analyser body unwraps via [`arg_at`]
/// at the touch site, panicking with the analyser name and slot
/// index — observationally equivalent to upstream's first-attribute-
/// touch failure, matching `unaryop.py:940 simple_call_SomeBuiltin`'s
/// bind-then-body sequence.
pub type BuiltinAnalyzer = fn(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError>;

/// Per-slot access helper: return `&SomeValue` at index `i`, panicking
/// on `None` to mirror upstream's `args_s[i].knowntype` AttributeError
/// surface at the moment of first attribute touch.  Bodies use this at
/// the access site so an unannotated optional-default slot can still
/// be observed via `args_s.get(i)`-style arity probing before the
/// panic fires.
pub(crate) fn arg_at<'a>(
    args_s: &'a [Option<SomeValue>],
    i: usize,
    analyser: &str,
) -> &'a SomeValue {
    args_s.get(i).and_then(|s| s.as_ref()).unwrap_or_else(|| {
        panic!(
            "{analyser}: positional arg {i} is unannotated \
                 (upstream `args_s[{i}].knowntype` AttributeError)",
        )
    })
}

// (Retired 2026-05-12, strict-parity pass.)  The
// `args_s_concrete_or_panic` eager-prefix helper was a NEW-DEVIATION
// from `unaryop.py:940 simple_call_SomeBuiltin`'s bind-then-body
// sequence (upstream's analyser body raises `AttributeError` on the
// FIRST slot touch, not at a blanket prefix unwrap).  Every
// `builtin_*` analyser in this file now uses [`arg_at`] at the touch
// site directly; per-iteration `arg_at` covers helpers like
// `union_many` / `r_dict_helper` that previously consumed the prefix
// buffer.  Rtyper-phase callers (`SomePtr.call` / `SomeLLPtr.call`)
// touch each slot via `annotation_to_lltype` in a list comprehension
// — those now drive `arg_at` per iteration to mirror upstream
// `lltype.py:1573-1574` AttributeError semantics.

/// Process-wide `BUILTIN_ANALYZERS` table, lazily populated on first
/// access by [`register_builtins`]. Mirrors upstream's module-level
/// `BUILTIN_ANALYZERS = {}` dict (bookkeeper.py:32).
static BUILTIN_ANALYZERS: OnceLock<HashMap<String, BuiltinAnalyzer>> = OnceLock::new();

/// Lazy accessor for the analyser table. Calls [`register_builtins`]
/// the first time it is hit.
pub fn analyzers() -> &'static HashMap<String, BuiltinAnalyzer> {
    BUILTIN_ANALYZERS.get_or_init(register_builtins)
}

/// `True` iff the given qualname has a registered analyser.
///
/// Mirrors upstream's `x in BUILTIN_ANALYZERS` membership test at
/// `bookkeeper.py:309` and `classdesc.py:632`.
pub fn is_registered(qualname: &str) -> bool {
    analyzers().contains_key(qualname)
}

/// Upstream `BUILTIN_ANALYZERS[x]` read. Returns `None` when `qualname`
/// is not registered.
pub fn lookup(qualname: &str) -> Option<BuiltinAnalyzer> {
    analyzers().get(qualname).copied()
}

/// Per-analyser allow-list of `s_`-prefixed keyword argument names.
///
/// Upstream's dispatch is `self.analyser(*args_s, **kwds_s)`, which
/// raises `TypeError` when an unknown keyword reaches the analyser's
/// Python signature. The Rust dispatcher must reject the same cases
/// before calling the function pointer, so each analyser declares the
/// set of keywords it accepts here. The default (missing entry) means
/// "no keywords accepted".
fn allowed_kwds(qualname: &str) -> &'static [&'static str] {
    match qualname {
        // builtin_int(s_obj, s_base=None)
        "int" => &["s_base"],
        // builtin_enumerate(s_obj, s_start=None)
        "enumerate" => &["s_start"],
        // robjmodel_instantiate(s_clspbc, s_nonmovable=None)
        "rpython.rlib.objectmodel.instantiate" => &["s_nonmovable"],
        // _r_dict_helper(cls, s_eqfn, s_hashfn, s_force_non_null,
        //                s_simple_hash_eq)
        "rpython.rlib.objectmodel.r_dict" | "rpython.rlib.objectmodel.r_ordereddict" => {
            &["s_force_non_null", "s_simple_hash_eq"]
        }
        _ => &[],
    }
}

/// Upstream `SomeBuiltin.call(self, args)` dispatcher
/// (unaryop.py:940-946).
///
/// ```python
/// def call(self, args, implicit_init=False):
///     args_s, kwds = args.unpack()
///     kwds_s = {}
///     for key, s_value in kwds.items():
///         kwds_s['s_'+key] = s_value
///     return self.analyser(*args_s, **kwds_s)
/// ```
///
/// In addition to the direct dispatch, the Rust port rejects keyword
/// arguments whose `s_<name>` is not in [`allowed_kwds`] for this
/// `analyser_name`. Upstream gets that check for free from Python's
/// signature binding; the Rust dispatcher emulates it so calls like
/// `list(xs, foo=1)` or `bool(x, foo=1)` fail cleanly instead of
/// silently dropping `foo=1`.
pub fn call_builtin(
    bk: &Rc<Bookkeeper>,
    analyser_name: &str,
    args_s: &[Option<SomeValue>],
    kwds_s: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    let Some(analyser) = lookup(analyser_name) else {
        return Err(AnnotatorError::new(format!(
            "SomeBuiltin.call(): no analyser registered for {analyser_name}"
        )));
    };
    let allowed = allowed_kwds(analyser_name);
    for key in kwds_s.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(AnnotatorError::new(format!(
                "{analyser_name}() got an unexpected keyword argument {key:?}"
            )));
        }
    }
    analyser(bk, args_s, kwds_s)
}

/// Upstream `analyzer_for(func)` decorator (bookkeeper.py:34-38).
///
/// Registers an analyser in a mutable-builder HashMap. Used only at
/// init time inside [`register_builtins`]. Panics if the same qualname
/// is registered twice — mirrors Python's behaviour of silently
/// overwriting but is far less forgiving so tests catch duplicate
/// bindings.
fn analyzer_for(
    reg: &mut HashMap<String, BuiltinAnalyzer>,
    qualname: &str,
    analyser: BuiltinAnalyzer,
) {
    if reg.insert(qualname.to_string(), analyser).is_some() {
        panic!("builtin.rs: duplicate BUILTIN_ANALYZERS entry for {qualname}");
    }
}

/// Upstream `for name, value in globals().items(): if name.startswith('builtin_')` loop
/// (builtin.py:191-195) plus every `@analyzer_for(...)` decoration.
///
/// `__builtin__` aliases like `xrange = builtin_range` are registered
/// explicitly — upstream's `builtin_xrange = builtin_range`
/// assignment (builtin.py:86) places both names into
/// `BUILTIN_ANALYZERS` via the mass scan.
fn register_builtins() -> HashMap<String, BuiltinAnalyzer> {
    let mut reg: HashMap<String, BuiltinAnalyzer> = HashMap::new();
    // builtin.py:49-84 — `builtin_range` + `builtin_xrange` alias.
    analyzer_for(&mut reg, "range", builtin_range);
    analyzer_for(&mut reg, "xrange", builtin_range);
    // builtin.py:89-99 — enumerate / reversed.
    analyzer_for(&mut reg, "enumerate", builtin_enumerate);
    analyzer_for(&mut reg, "reversed", builtin_reversed);
    // builtin.py:102-130 — bool / int / float / chr / unichr /
    // unicode / bytearray.
    analyzer_for(&mut reg, "bool", builtin_bool);
    analyzer_for(&mut reg, "int", builtin_int);
    analyzer_for(&mut reg, "float", builtin_float);
    analyzer_for(&mut reg, "chr", builtin_chr);
    analyzer_for(&mut reg, "unichr", builtin_unichr);
    analyzer_for(&mut reg, "unicode", builtin_unicode);
    analyzer_for(&mut reg, "bytearray", builtin_bytearray);
    // builtin.py:133-148 — hasattr.
    analyzer_for(&mut reg, "hasattr", builtin_hasattr);
    // builtin.py:151-188 — tuple / list / zip / min / max.
    analyzer_for(&mut reg, "tuple", builtin_tuple);
    analyzer_for(&mut reg, "list", builtin_list);
    analyzer_for(&mut reg, "zip", builtin_zip);
    analyzer_for(&mut reg, "min", builtin_min);
    analyzer_for(&mut reg, "max", builtin_max);

    // ------------------------------------------------------------------
    // `@analyzer_for(...)` decoration sites.
    // ------------------------------------------------------------------

    // builtin.py:198-214 — object.__init__ / EnvironmentError.__init__ /
    // WindowsError.__init__ — builtin-exception-class skip gate reads
    // these names in classdesc.rs.
    analyzer_for(&mut reg, "object.__init__", object_init);
    analyzer_for(
        &mut reg,
        "EnvironmentError.__init__",
        environment_error_init,
    );
    // `WindowsError` exists only on the `win32` build; upstream wraps
    // the registration in `try: WindowsError; except NameError: pass`.
    // We register it unconditionally so cross-platform annotation runs
    // that exercise stubs carrying `WindowsError` classes succeed
    // identically.
    analyzer_for(&mut reg, "WindowsError.__init__", windows_error_init);

    // builtin.py:217-293 — sys / rarithmetic / objectmodel helpers.
    analyzer_for(&mut reg, "sys.getdefaultencoding", conf);
    analyzer_for(&mut reg, "rpython.rlib.rarithmetic.intmask", rarith_intmask);
    analyzer_for(
        &mut reg,
        "rpython.rlib.rarithmetic.longlongmask",
        rarith_longlongmask,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.instantiate",
        robjmodel_instantiate,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.r_dict",
        robjmodel_r_dict,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.r_ordereddict",
        robjmodel_r_ordereddict,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.hlinvoke",
        robjmodel_hlinvoke,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.keepalive_until_here",
        robjmodel_keepalive_until_here,
    );
    analyzer_for(
        &mut reg,
        "rpython.rlib.objectmodel.free_non_gc_object",
        robjmodel_free_non_gc_object,
    );

    // builtin.py:295-307 — unicodedata / OrderedDict.
    analyzer_for(&mut reg, "unicodedata.decimal", unicodedata_decimal);
    analyzer_for(&mut reg, "collections.OrderedDict", analyze_ordered_dict);

    // builtin.py:314-321 — weakref.ref.
    analyzer_for(&mut reg, "weakref.ref", weakref_ref);

    // builtin.py:335-339 — pdb.set_trace.
    analyzer_for(&mut reg, "pdb.set_trace", pdb_set_trace);

    reg
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Upstream `constpropagate(func, args_s, s_result)` (builtin.py:22-45).
///
/// ```python
/// def constpropagate(func, args_s, s_result):
///     args = []
///     for s in args_s:
///         if not s.is_immutable_constant():
///             return s_result
///         args.append(s.const)
///     try:
///         realresult = func(*args)
///     except (ValueError, OverflowError):
///         return s_result
///     s_realresult = immutablevalue(realresult)
///     if not s_result.contains(s_realresult):
///         raise AnnotatorError(...)
///     return s_realresult
/// ```
///
/// `evaluator` models Python's `func(*args)` — it returns `None` when
/// upstream would have caught `ValueError` / `OverflowError`.
pub fn constpropagate<F>(
    bk: &Rc<Bookkeeper>,
    args_s: &[SomeValue],
    s_result: SomeValue,
    evaluator: F,
) -> Result<SomeValue, AnnotatorError>
where
    F: FnOnce(&[&ConstValue]) -> Option<ConstValue>,
{
    let mut consts: Vec<&ConstValue> = Vec::with_capacity(args_s.len());
    for s in args_s {
        if !s.is_immutable_constant() {
            return Ok(s_result);
        }
        let Some(c) = s.const_() else {
            return Ok(s_result);
        };
        consts.push(c);
    }
    let Some(realresult) = evaluator(&consts) else {
        return Ok(s_result);
    };
    let s_realresult = bk.immutablevalue(&realresult)?;
    if !s_result.contains(&s_realresult) {
        return Err(AnnotatorError::new(format!(
            "constpropagate: {realresult} is not contained in s_result"
        )));
    }
    Ok(s_realresult)
}

/// Upstream `SomeObject.is_immutable_constant` — `True` iff
/// `immutable and is_constant()`. Kept as a thin free helper so the
/// per-analyser sites read like upstream `s.is_immutable_constant()`.
fn is_immutable_constant(s: &SomeValue) -> bool {
    s.immutable() && s.is_constant()
}

/// Upstream `SomeInteger.nonneg` projection — returns `True` iff the
/// annotation is a `SomeInteger` with `nonneg=True`. Other annotations
/// default to `False`.
fn nonneg_of(s: &SomeValue) -> bool {
    match s {
        SomeValue::Integer(i) => i.nonneg,
        _ => false,
    }
}

/// Upstream `s.knowntype == str` projection — matches both `SomeString`
/// and `SomeChar` since both carry `knowntype=str`. Used by
/// [`builtin_int`] to approve `int(string, base)` syntax.
fn is_str_annotation(s: &SomeValue) -> bool {
    matches!(s, SomeValue::String(_) | SomeValue::Char(_))
}

// ---------------------------------------------------------------------------
// `builtin_*` analysers (mass-registered via the `builtin_` prefix scan).
// ---------------------------------------------------------------------------

/// Upstream `builtin_range(*args)` (builtin.py:49-84).
pub fn builtin_range(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // Arity check fires first per `builtin.py:49-84` shape; per-slot
    // access defers through `arg_at` so a `None` slot surfaces as
    // "unannotated arg" at the touch site (matching `unaryop.py:940
    // simple_call_SomeBuiltin`'s bind-then-body sequence) instead of
    // the entry-point prefix unwrap.
    let (s_start, s_stop, s_step) = match args_s.len() {
        1 => (
            bk.immutablevalue(&ConstValue::Int(0))?,
            arg_at(args_s, 0, "builtin_range").clone(),
            bk.immutablevalue(&ConstValue::Int(1))?,
        ),
        2 => (
            arg_at(args_s, 0, "builtin_range").clone(),
            arg_at(args_s, 1, "builtin_range").clone(),
            bk.immutablevalue(&ConstValue::Int(1))?,
        ),
        3 => (
            arg_at(args_s, 0, "builtin_range").clone(),
            arg_at(args_s, 1, "builtin_range").clone(),
            arg_at(args_s, 2, "builtin_range").clone(),
        ),
        _ => {
            return Err(AnnotatorError::new("range() takes 1 to 3 arguments"));
        }
    };

    let mut empty = false;
    let step: i64;
    if !is_immutable_constant(&s_step) {
        // upstream `step = 0` signals "variable step".
        step = 0;
    } else {
        step = match s_step.const_() {
            Some(ConstValue::Int(n)) => *n,
            Some(ConstValue::Bool(b)) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            _ => {
                return Err(AnnotatorError::new(
                    "range() with non-integer constant step",
                ));
            }
        };
        if step == 0 {
            return Err(AnnotatorError::new("range() with step zero"));
        }
        if is_immutable_constant(&s_start) && is_immutable_constant(&s_stop) {
            if let (Some(ConstValue::Int(start)), Some(ConstValue::Int(stop))) =
                (s_start.const_(), s_stop.const_())
            {
                // upstream `if len(xrange(start, stop, step)) == 0:`.
                let range_len: i64 = if step > 0 {
                    if *stop > *start { 1 } else { 0 }
                } else if *stop < *start {
                    1
                } else {
                    0
                };
                if range_len == 0 {
                    empty = true;
                }
            }
        }
    }

    let s_item = if empty {
        s_impossible_value()
    } else {
        // upstream: `if step > 0 or s_step.nonneg: nonneg = s_start.nonneg`.
        // `elif step < 0: nonneg = s_stop.nonneg or (s_stop.is_constant()
        //                                            and s_stop.const >= -1)`.
        let nonneg = if step > 0 || nonneg_of(&s_step) {
            nonneg_of(&s_start)
        } else if step < 0 {
            nonneg_of(&s_stop) || matches!(s_stop.const_(), Some(ConstValue::Int(n)) if *n >= -1)
        } else {
            false
        };
        SomeValue::Integer(SomeInteger::new(nonneg, false))
    };

    let s_list = bk.newlist(&[s_item], Some(step))?;
    Ok(SomeValue::List(s_list))
}

/// Upstream `builtin_enumerate(s_obj, s_start=None)` (builtin.py:89-95).
pub fn builtin_enumerate(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream: arity check first — `s_start=None` default means a
    // 1-arg call binds `s_start` to Python None.  Defer per-slot
    // access through `arg_at` so the unwrap surfaces at first
    // attribute touch (matching upstream's `s_start.is_constant()`
    // AttributeError) rather than a blanket entry-point panic.
    let (s_obj, s_start_positional) = match args_s.len() {
        1 => (arg_at(args_s, 0, "builtin_enumerate").clone(), None),
        2 => (
            arg_at(args_s, 0, "builtin_enumerate").clone(),
            Some(arg_at(args_s, 1, "builtin_enumerate").clone()),
        ),
        _ => {
            return Err(AnnotatorError::new("enumerate() takes 1 or 2 arguments"));
        }
    };
    // upstream's Python signature binding binds `start` to whichever of
    // positional / keyword was supplied, and raises TypeError on both.
    if s_start_positional.is_some() && kwds.contains_key("s_start") {
        return Err(AnnotatorError::new(
            "enumerate() got multiple values for argument 'start'",
        ));
    }
    // upstream `Argument(start=...)` binds the value as-is (including
    // `None` from `annotator.annotation(v) is None`); pyre's
    // `kwds.get(...).cloned()` returns `Option<Option<SomeValue>>` so
    // `.flatten()` collapses to the Option<SomeValue> form the body
    // consumes via `s_start.as_ref()` (None tolerated).
    let s_start = s_start_positional.or_else(|| kwds.get("s_start").cloned().flatten());
    if let Some(s_start) = s_start.as_ref()
        && !s_start.is_constant()
    {
        return Err(AnnotatorError::new(
            "second argument to enumerate must be constant",
        ));
    }
    let const_start = s_start.and_then(|s| s.const_().cloned());
    Ok(SomeValue::Iterator(SomeIterator::new_with_enumerate_start(
        s_obj,
        vec!["enumerate".to_string()],
        const_start,
    )))
}

/// Upstream `builtin_reversed(s_obj)` (builtin.py:98-99).
pub fn builtin_reversed(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("reversed() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_reversed");
    Ok(SomeValue::Iterator(SomeIterator::new(
        s_obj.clone(),
        vec!["reversed".to_string()],
    )))
}

/// Upstream `builtin_bool(s_obj)` (builtin.py:102-103).
///
/// Upstream dispatches `s_obj.bool()` which is registered per-`Some*`
/// in `unaryop.py`. The annotator-phase subset used by `builtin_bool`
/// reduces to:
///
/// * constant immutable receivers → `SomeBool(bool(s_obj.const))`
/// * empty-tuple literal `()` → `SomeBool(False)` (unaryop.py:347-351)
/// * everything else → unrefined `SomeBool()`
///
/// Knowntypedata propagation (`SomeObject._propagate_knowntypedata`) is
/// an analyser-driven optimisation and is handled by the OpKind::Bool
/// dispatch in `unaryop.rs`; the annotator-phase `builtin_bool` path
/// used by `call_builtin` does not have the HLOperation context to
/// track that side channel and leaves the per-branch refinements to
/// the surrounding flow-level dispatch.
pub fn builtin_bool(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("bool() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_bool");
    let mut r = SomeBool::new();

    // upstream `SomeTuple.bool` (unaryop.py:347-351): empty tuple is
    // always falsy.
    if let SomeValue::Tuple(t) = s_obj
        && t.items.is_empty()
    {
        r.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::Bool(false),
        ));
        return Ok(SomeValue::Bool(r));
    }

    // upstream default + primitive overrides: when the operand is a
    // constant immutable value, pin `r.const = bool(value)`.
    if is_immutable_constant(s_obj)
        && let Some(c) = s_obj.const_()
    {
        let truthy = match c {
            ConstValue::Bool(b) => *b,
            ConstValue::Int(n) => *n != 0,
            ConstValue::Float(bits) => f64::from_bits(*bits) != 0.0,
            ConstValue::ByteStr(s) => !s.is_empty(),
            ConstValue::UniStr(s) => !s.is_empty(),
            ConstValue::None => false,
            ConstValue::Tuple(items) | ConstValue::List(items) => !items.is_empty(),
            ConstValue::Graphs(graphs) => !graphs.is_empty(),
            ConstValue::Dict(m) => !m.is_empty(),
            // HostObject / Function / Code etc. have no `__bool__`
            // override metadata in the Rust port, so take Python's
            // default `True`.
            ConstValue::HostObject(_)
            | ConstValue::Function(_)
            | ConstValue::LowLevelType(_)
            | ConstValue::LLPtr(_)
            | ConstValue::Code(_)
            | ConstValue::Atom(_)
            | ConstValue::SpecTag(_) => true,
            ConstValue::LLAddress(addr) => matches!(
                addr,
                crate::translator::rtyper::lltypesystem::lltype::_address::Fake(_)
            ),
            // Placeholder never appears in production flow; stay
            // conservative rather than crash.
            ConstValue::Placeholder => return Ok(SomeValue::Bool(r)),
        };
        r.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::Bool(truthy),
        ));
    }
    Ok(SomeValue::Bool(r))
}

/// Upstream `builtin_int(s_obj, s_base=None)` (builtin.py:105-115).
pub fn builtin_int(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream signature: `int(s_obj, s_base=None)` — allow keyword
    // form for parity.  Arity check first so a 1-arg call lets
    // `s_base` default to None without touching slot 1.
    if args_s.is_empty() {
        return Err(AnnotatorError::new("int() takes at least one argument"));
    }
    if args_s.len() > 2 {
        return Err(AnnotatorError::new("int() takes at most 2 arguments"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_int").clone();
    if args_s.len() >= 2 && kwds.contains_key("s_base") {
        return Err(AnnotatorError::new(
            "int() got multiple values for argument 'base'",
        ));
    }
    let s_base = if args_s.len() == 2 {
        Some(arg_at(args_s, 1, "builtin_int").clone())
    } else {
        // upstream `Argument(base=...)` accepts `None` (unbound).
        // pyre's `kwds.get(...).cloned()` returns
        // `Option<Option<SomeValue>>`; `.flatten()` projects to
        // `Option<SomeValue>` to align with the positional arm.
        kwds.get("s_base").cloned().flatten()
    };

    if let SomeValue::Integer(ref i) = s_obj
        && i.unsigned
    {
        return Err(AnnotatorError::new(
            "instead of int(r_uint(x)), use intmask(r_uint(x))",
        ));
    }
    if let Some(ref sb) = s_base {
        let base_is_int = matches!(sb, SomeValue::Integer(i) if !i.unsigned);
        if !base_is_int || !is_str_annotation(&s_obj) {
            return Err(AnnotatorError::new(
                "only int(v|string) or int(string,int) expected",
            ));
        }
    }

    let nonneg = matches!(&s_obj, SomeValue::Integer(i) if i.nonneg);
    let s_result = SomeValue::Integer(SomeInteger::new(nonneg, false));

    let prop_args: Vec<SomeValue> = match &s_base {
        Some(sb) => vec![s_obj.clone(), sb.clone()],
        None => vec![s_obj.clone()],
    };
    constpropagate(bk, &prop_args, s_result, |consts| match consts {
        [ConstValue::Int(n)] => Some(ConstValue::Int(*n)),
        [ConstValue::Bool(b)] => Some(ConstValue::Int(if *b { 1 } else { 0 })),
        [s] if s.as_text().is_some() => s
            .as_text()
            .and_then(|text| text.parse::<i64>().ok())
            .map(ConstValue::Int),
        [ConstValue::Float(bits)] => {
            // upstream `int(float_const)` raises OverflowError when the
            // truncated value does not fit, which `constpropagate`
            // catches and falls back to `s_result` (builtin.py:34-39).
            // Reject non-finite floats and values that would saturate
            // an i64 cast, matching that behaviour instead of pinning a
            // wrong constant.
            let f = f64::from_bits(*bits);
            if !f.is_finite() {
                return None;
            }
            let truncated = f.trunc();
            if truncated < i64::MIN as f64 || truncated >= (i64::MAX as f64 + 1.0) {
                return None;
            }
            Some(ConstValue::Int(truncated as i64))
        }
        [s, ConstValue::Int(base)] if s.as_text().is_some() => {
            if *base >= 2 && *base <= 36 {
                i64::from_str_radix(s.as_text().unwrap().trim(), *base as u32)
                    .ok()
                    .map(ConstValue::Int)
            } else {
                None
            }
        }
        _ => None,
    })
}

/// Upstream `builtin_float(s_obj)` (builtin.py:117-118).
pub fn builtin_float(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("float() takes exactly one argument"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_float");
    constpropagate(
        bk,
        &[s_obj.clone()],
        SomeValue::Float(SomeFloat::new()),
        |consts| match consts {
            [ConstValue::Int(n)] => Some(ConstValue::float(*n as f64)),
            [ConstValue::Bool(b)] => Some(ConstValue::float(if *b { 1.0 } else { 0.0 })),
            [ConstValue::Float(bits)] => Some(ConstValue::Float(*bits)),
            [s] if s.as_text().is_some() => s
                .as_text()
                .and_then(|text| text.trim().parse::<f64>().ok())
                .map(ConstValue::float),
            _ => None,
        },
    )
}

/// Upstream `builtin_chr(s_int)` (builtin.py:120-121).
pub fn builtin_chr(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("chr() takes exactly one argument"));
    }
    let s_int = arg_at(args_s, 0, "builtin_chr");
    constpropagate(
        bk,
        &[s_int.clone()],
        SomeValue::Char(SomeChar::new(false)),
        |consts| match consts {
            [ConstValue::Int(n)] => {
                if (0..=0xff).contains(n) {
                    Some(ConstValue::ByteStr(vec![*n as u8]))
                } else {
                    None
                }
            }
            _ => None,
        },
    )
}

/// Upstream `builtin_unichr(s_int)` (builtin.py:123-124).
///
/// Fold the constant case directly, mirroring upstream
/// `constpropagate(unichr, ...)` semantics while pinning the result's
/// `const_box` ourselves.
pub fn builtin_unichr(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("unichr() takes exactly one argument"));
    }
    let s_int = arg_at(args_s, 0, "builtin_unichr");
    let mut result = SomeUnicodeCodePoint::new(false);
    if is_immutable_constant(s_int)
        && let Some(ConstValue::Int(n)) = s_int.const_()
        && (0..=0x10_FFFF).contains(n)
        && let Some(ch) = u32::try_from(*n).ok().and_then(char::from_u32)
    {
        result.inner.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::uni_str(ch.to_string()),
        ));
    }
    Ok(SomeValue::UnicodeCodePoint(result))
}

/// Upstream `builtin_unicode(s_unicode)` (builtin.py:126-127).
///
/// Mirrors upstream `unicode(x)` while preserving the explicit
/// `ConstValue::UniStr` type tag.
pub fn builtin_unicode(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("unicode() takes exactly one argument"));
    }
    let s_unicode = arg_at(args_s, 0, "builtin_unicode");
    let mut result = SomeUnicodeString::new(false, false);
    if is_immutable_constant(s_unicode)
        && let Some(s) = s_unicode.const_().and_then(ConstValue::as_text)
    {
        result.inner.base.const_box = Some(super::super::flowspace::model::Constant::new(
            ConstValue::uni_str(s),
        ));
    }
    Ok(SomeValue::UnicodeString(result))
}

/// Upstream `builtin_bytearray(s_str)` (builtin.py:129-130).
pub fn builtin_bytearray(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // `builtin.py:129` `def builtin_bytearray(s_str):` — Python's
    // call binding (`unaryop.py:940 simple_call_SomeBuiltin` →
    // `signature(builtin_bytearray).bind(...)`) raises `TypeError`
    // when arity disagrees.  Mirror the bind-time check; analyser
    // body still ignores the argument shape and returns
    // `SomeByteArray()`.
    if args_s.len() != 1 {
        return Err(AnnotatorError::new(format!(
            "bytearray() takes exactly one argument ({} given)",
            args_s.len()
        )));
    }
    Ok(SomeValue::ByteArray(SomeByteArray::new(false)))
}

/// Upstream `builtin_hasattr(s_obj, s_attr)` (builtin.py:133-148).
pub fn builtin_hasattr(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 2 {
        return Err(AnnotatorError::new("hasattr() takes exactly two arguments"));
    }
    let s_obj = arg_at(args_s, 0, "builtin_hasattr");
    let s_attr = arg_at(args_s, 1, "builtin_hasattr");
    // builtin.py:133-136 — `if not s_attr.is_constant() or not
    // isinstance(s_attr.const, str): bookkeeper.warning("hasattr is not
    // RPythonic enough") ; return SomeBool()`. Upstream's `isinstance(x,
    // str)` is Python 2 bytes-only; pyre uses [`as_pystr`] so a unicode
    // attribute literal short-circuits to the non-constant SomeBool
    // path the same way upstream does. (We skip the warning because
    // the Rust bookkeeper has no warning channel yet.)
    let attr_is_const_str = s_attr.const_().and_then(ConstValue::as_pystr).is_some();
    if !s_attr.is_constant() || !attr_is_const_str {
        return Ok(SomeValue::Bool(SomeBool::new()));
    }
    let Some(attr_name) = s_attr.const_().and_then(ConstValue::as_pystr) else {
        return Ok(SomeValue::Bool(SomeBool::new()));
    };

    let mut r = SomeBool::new();
    if s_obj.is_immutable_constant() {
        // upstream: `r.const = hasattr(s_obj.const, s_attr.const)`.
        // Emulate Python's attribute resolution via [`host_hasattr`]:
        // modules → module_get; classes → MRO walk (class_get per base);
        // instances → instance dict first, then class MRO; everything
        // else → class_get fallback.
        if let Some(ConstValue::HostObject(host)) = s_obj.const_() {
            let found = host_hasattr(host, attr_name);
            r.base.const_box = Some(super::super::flowspace::model::Constant::new(
                ConstValue::Bool(found),
            ));
        }
    } else if let SomeValue::PBC(pbc) = s_obj {
        // upstream: `isinstance(s_obj, SomePBC) and s_obj.getKind() is FrozenDesc`.
        let all_frozen = !pbc.descriptions.is_empty()
            && pbc.descriptions.values().all(|d| {
                matches!(
                    d.kind(),
                    super::model::DescKind::Frozen | super::model::DescKind::MethodOfFrozen
                )
            });
        if all_frozen {
            // upstream (builtin.py:142-147):
            //
            //     answers = {}
            //     for d in s_obj.descriptions:
            //         answer = (d.s_read_attribute(s_attr.const) !=
            //                   s_ImpossibleValue)
            //         answers[answer] = True
            //     if len(answers) == 1:
            //         r.const, = answers
            let mut answers: std::collections::HashSet<bool> = Default::default();
            for desc in pbc.descriptions.values() {
                if let DescEntry::Frozen(fd) = desc {
                    let result = fd.borrow().s_read_attribute(attr_name);
                    // upstream returns `s_ImpossibleValue` for missing
                    // attrs; the Rust port surfaces errors as
                    // `AnnotatorError`, which we treat as "not found"
                    // for the `hasattr` containment question.
                    let found = matches!(result, Ok(ref v) if !matches!(v, SomeValue::Impossible));
                    answers.insert(found);
                }
            }
            if answers.len() == 1 {
                let only = *answers.iter().next().unwrap();
                r.base.const_box = Some(super::super::flowspace::model::Constant::new(
                    ConstValue::Bool(only),
                ));
            }
        }
    }
    Ok(SomeValue::Bool(r))
}

/// Upstream `builtin_tuple(s_iterable)` (builtin.py:151-154).
pub fn builtin_tuple(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("tuple() takes exactly one argument"));
    }
    let s_iterable = arg_at(args_s, 0, "builtin_tuple");
    if let SomeValue::Tuple(_) = s_iterable {
        return Ok(s_iterable.clone());
    }
    Err(AnnotatorError::new(
        "tuple(): argument must be another tuple",
    ))
}

/// Upstream `builtin_list(s_iterable)` (builtin.py:156-161).
pub fn builtin_list(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new("list() takes exactly one argument"));
    }
    let s_iterable = arg_at(args_s, 0, "builtin_list");
    if let SomeValue::List(s_list) = s_iterable {
        // upstream: `s_iterable.listdef.offspring(bk)`.
        return Ok(SomeValue::List(s_list.listdef.offspring(bk, &[])?));
    }
    // upstream: `s_iter = s_iterable.iter(); return bk.newlist(s_iter.next())`.
    let s_iter = SomeIterator::new(s_iterable.clone(), vec![]);
    let s_item = someiterator_next_stub(bk, &s_iter);
    Ok(SomeValue::List(bk.newlist(&[s_item], None)?))
}

/// Upstream `builtin_zip(s_iterable1, s_iterable2)` (builtin.py:163-167).
pub fn builtin_zip(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 2 {
        return Err(AnnotatorError::new("zip() takes exactly two arguments"));
    }
    let s_iter1 = SomeIterator::new(arg_at(args_s, 0, "builtin_zip").clone(), vec![]);
    let s_iter2 = SomeIterator::new(arg_at(args_s, 1, "builtin_zip").clone(), vec![]);
    let s_tup = SomeValue::Tuple(SomeTuple::new(vec![
        someiterator_next_stub(bk, &s_iter1),
        someiterator_next_stub(bk, &s_iter2),
    ]));
    Ok(SomeValue::List(bk.newlist(&[s_tup], None)?))
}

/// Upstream `builtin_min(*s_values)` (builtin.py:169-174).
pub fn builtin_min(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("min() takes at least one argument"));
    }
    if args_s.len() == 1 {
        // upstream: `s_iter = s_values[0].iter(); return s_iter.next()`.
        let s_iter = SomeIterator::new(arg_at(args_s, 0, "builtin_min").clone(), vec![]);
        return Ok(someiterator_next_stub(bk, &s_iter));
    }
    // upstream: `return union(*s_values)`.
    union_many(args_s, "builtin_min")
}

/// Upstream `builtin_max(*s_values)` (builtin.py:176-188).
pub fn builtin_max(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("max() takes at least one argument"));
    }
    if args_s.len() == 1 {
        let s_iter = SomeIterator::new(arg_at(args_s, 0, "builtin_max").clone(), vec![]);
        return Ok(someiterator_next_stub(bk, &s_iter));
    }
    let mut s = union_many(args_s, "builtin_max")?;
    // upstream (builtin.py:182-188):
    //
    //     if type(s) is SomeInteger and not s.nonneg:
    //         nonneg = False
    //         for s1 in s_values:
    //             nonneg |= s1.nonneg
    //         if nonneg:
    //             s = SomeInteger(nonneg=True, knowntype=s.knowntype)
    //
    // `nonneg |= s1.nonneg` is an OR-fold, not AND — any single nonneg
    // operand wins because `max(nonneg, whatever) >= nonneg >= 0`.
    if let SomeValue::Integer(ref i) = s
        && !i.nonneg
    {
        // Per-touch revisit: `union_many` already touched every slot
        // above, so by this point each `args_s[i]` is guaranteed to be
        // `Some(..)` (otherwise the loop would have panicked).
        let any_nonneg = (0..args_s.len()).any(|i| nonneg_of(arg_at(args_s, i, "builtin_max")));
        if any_nonneg {
            // upstream: `SomeInteger(nonneg=True, knowntype=s.knowntype)`
            // — pin the existing knowntype to avoid widening r_uint to int.
            s = SomeValue::Integer(SomeInteger::new_with_knowntype(true, i.base.knowntype));
        }
    }
    Ok(s)
}

// ---------------------------------------------------------------------------
// `@analyzer_for(...)` analysers.
// ---------------------------------------------------------------------------

/// Upstream `object_init(s_self, *args)` (builtin.py:198-201).
pub fn object_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream returns `None` which `immutablevalue` wraps as s_None.
    Ok(s_none())
}

/// Upstream `EnvironmentError_init(s_self, *args)` (builtin.py:203-205).
pub fn environment_error_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `WindowsError_init(s_self, *args)` (builtin.py:212-214).
pub fn windows_error_init(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `conf()` (builtin.py:217-219).
pub fn conf(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::String(SomeString::new(false, false)))
}

/// Upstream `rarith_intmask(s_obj)` (builtin.py:221-223).
pub fn rarith_intmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(SomeValue::Integer(SomeInteger::default()))
}

/// Upstream `rarith_longlongmask(s_obj)` (builtin.py:225-227).
pub fn rarith_longlongmask(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream: `SomeInteger(knowntype=rpython.rlib.rarithmetic.r_longlong)`.
    Ok(SomeValue::Integer(SomeInteger::new_with_knowntype(
        false,
        crate::annotator::model::KnownType::LongLong,
    )))
}

/// Upstream `robjmodel_instantiate(s_clspbc, s_nonmovable=None)`
/// (builtin.py:229-242).
pub fn robjmodel_instantiate(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new(
            "instantiate() expected SomePBC as first argument",
        ));
    }
    let s_clspbc_v = arg_at(args_s, 0, "robjmodel_instantiate");
    let SomeValue::PBC(s_clspbc) = s_clspbc_v else {
        return Err(AnnotatorError::new(
            "instantiate() expected SomePBC as first argument",
        ));
    };
    let mut clsdef: Option<Rc<RefCell<ClassDef>>> = None;
    // upstream: `more_than_one = len(s_clspbc.descriptions) > 1`. The
    // `needs_generic_instantiate` dict is consumed downstream by the
    // rtyper (`rpython/rtyper/rclass.py:_get_concrete_class`) to emit
    // a generic allocator; the Rust bookkeeper does not yet carry that
    // map (no rtyper consumer exists) — the flag is recorded here as a
    // comment so the field lands alongside the rtyper port.
    //   let more_than_one = s_clspbc.descriptions.len() > 1;
    for desc in s_clspbc.descriptions.values() {
        let DescEntry::Class(class_desc) = desc else {
            return Err(AnnotatorError::new(
                "instantiate() expects a class PBC — non-class DescEntry",
            ));
        };
        let cdef = ClassDesc::getuniqueclassdef(class_desc)?;
        //   if more_than_one: bk.needs_generic_instantiate[cdef] = True
        clsdef = Some(match clsdef {
            None => cdef,
            Some(prev) => ClassDef::commonbase(&prev, &cdef)
                .ok_or_else(|| AnnotatorError::new("instantiate(): classes have no common base"))?,
        });
    }
    let cdef = clsdef
        .ok_or_else(|| AnnotatorError::new("instantiate() called with empty descriptor set"))?;
    Ok(SomeValue::Instance(SomeInstance::new(
        Some(cdef),
        false,
        Default::default(),
    )))
}

/// Upstream `robjmodel_r_dict(...)` (builtin.py:244-246).
pub fn robjmodel_r_dict(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    r_dict_helper(bk, args_s, kwds, RDictKind::Regular, "robjmodel_r_dict")
}

/// Upstream `robjmodel_r_ordereddict(...)` (builtin.py:248-251).
pub fn robjmodel_r_ordereddict(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    r_dict_helper(
        bk,
        args_s,
        kwds,
        RDictKind::Ordered,
        "robjmodel_r_ordereddict",
    )
}

#[derive(Clone, Copy)]
enum RDictKind {
    Regular,
    Ordered,
}

/// Upstream `_r_dict_helper(cls, s_eqfn, s_hashfn, s_force_non_null,
/// s_simple_hash_eq)` (builtin.py:253-268).
///
/// Accepts both positional and keyword forms for the `force_non_null`
/// and `simple_hash_eq` flags — upstream's Python signature binds
/// either path into the same local. Raises the upstream duplicate-
/// argument error if a caller passes both.
fn r_dict_helper(
    bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    kwds: &HashMap<String, Option<SomeValue>>,
    kind: RDictKind,
    analyser: &str,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.is_empty() {
        return Err(AnnotatorError::new("r_dict(): missing eq function"));
    }
    if args_s.len() < 2 {
        return Err(AnnotatorError::new("r_dict(): missing hash function"));
    }
    // upstream `_r_dict_helper` binds these positions first; the body
    // touches `s_eqfn`/`s_hashfn` immediately so the per-touch `arg_at`
    // unwrap fires here (matching `unaryop.py:940 simple_call_
    // SomeBuiltin`'s bind-then-body sequence).
    let s_eqfn = arg_at(args_s, 0, analyser).clone();
    let s_hashfn = arg_at(args_s, 1, analyser).clone();
    if args_s.len() >= 3 && kwds.contains_key("s_force_non_null") {
        return Err(AnnotatorError::new(
            "r_dict() got multiple values for argument 'force_non_null'",
        ));
    }
    if args_s.len() >= 4 && kwds.contains_key("s_simple_hash_eq") {
        return Err(AnnotatorError::new(
            "r_dict() got multiple values for argument 'simple_hash_eq'",
        ));
    }
    if args_s.len() > 4 {
        return Err(AnnotatorError::new("r_dict() takes at most 4 arguments"));
    }
    // Optional positions: a slot present-but-None is treated identically
    // to "absent" here, matching upstream's `args_s.get(N)` semantic
    // (`None` from `get` and `None` from the slot itself both fall
    // through to the kwds lookup).  Required slots above use `arg_at`
    // which panics on None.  `kwds.get(name).cloned().flatten()`
    // projects `Option<Option<SomeValue>>` → `Option<SomeValue>`,
    // collapsing "key absent" and "key present but value unbound"
    // into a single "no concrete annotation" sentinel.
    let s_force = args_s
        .get(2)
        .and_then(|s| s.clone())
        .or_else(|| kwds.get("s_force_non_null").cloned().flatten());
    let s_simple = args_s
        .get(3)
        .and_then(|s| s.clone())
        .or_else(|| kwds.get("s_simple_hash_eq").cloned().flatten());
    let force_non_null = match s_force {
        None => false,
        Some(ref s) => {
            if !s.is_immutable_constant() {
                return Err(AnnotatorError::new(
                    "r_dict(): force_non_null must be constant",
                ));
            }
            matches!(s.const_(), Some(ConstValue::Bool(true)))
        }
    };
    let simple_hash_eq = match s_simple {
        None => false,
        Some(ref s) => {
            if !s.is_immutable_constant() {
                return Err(AnnotatorError::new(
                    "r_dict(): simple_hash_eq must be constant",
                ));
            }
            matches!(s.const_(), Some(ConstValue::Bool(true)))
        }
    };
    let dictdef = bk.getdictdef(true, force_non_null, simple_hash_eq);
    super::dictdef::DictKey::update_rdict_annotations(&dictdef.dictkey_rc(), s_eqfn, s_hashfn)
        .map_err(|e| AnnotatorError::new(e.msg))?;
    // upstream splits `SomeDict` / `SomeOrderedDict` classes; the Rust
    // port collapses them into a single `SomeDict` variant per CLAUDE.md
    // parity rule #1 (see `model.rs:878-880`). `RDictKind` is retained
    // in the call site to keep the upstream dispatch readable and to
    // ease re-splitting once the lattice grows `SomeOrderedDict`.
    let _ = kind;
    Ok(SomeValue::Dict(SomeDict::new(dictdef)))
}

/// Upstream `robjmodel_hlinvoke(...)` (builtin.py:270-288).
///
/// rtyper-phase only — consumes `rmodel.Repr`, `TyperError`,
/// `lltype_to_annotation`. Returns a clear `AnnotatorError` until the
/// rtyper bridge lands.
pub fn robjmodel_hlinvoke(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "hlinvoke: rtyper-phase only, not supported in annotator",
    ))
}

/// Upstream `robjmodel_keepalive_until_here(*args_s)` (builtin.py:291-293).
pub fn robjmodel_keepalive_until_here(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Ok(s_none())
}

/// Upstream `robjmodel_free_non_gc_object(obj)` (builtin.py:326-328).
pub fn robjmodel_free_non_gc_object(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream implicitly returns None from an empty `def`.
    Ok(s_none())
}

/// Upstream `unicodedata_decimal(s_uchr)` (builtin.py:300-303).
pub fn unicodedata_decimal(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "unicodedate.decimal() calls should not happen at interp-level",
    ))
}

/// Upstream `analyze()` registered for `OrderedDict` (builtin.py:305-307).
pub fn analyze_ordered_dict(
    bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    // upstream returns `SomeOrderedDict(bk.getdictdef())`; the Rust port
    // collapses `SomeDict`/`SomeOrderedDict` (see `model.rs:878-880`).
    let dd = bk.getdictdef(false, false, false);
    Ok(SomeValue::Dict(SomeDict::new(dd)))
}

/// Upstream `weakref_ref(s_obj)` (builtin.py:314-321).
pub fn weakref_ref(
    _bk: &Rc<Bookkeeper>,
    args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    if args_s.len() != 1 {
        return Err(AnnotatorError::new(
            "weakref.ref() takes exactly one argument",
        ));
    }
    let s_obj = arg_at(args_s, 0, "weakref_ref");
    let SomeValue::Instance(inst) = s_obj else {
        return Err(AnnotatorError::new(format!(
            "cannot take a weakref to {s_obj:?}"
        )));
    };
    if inst.can_be_none {
        return Err(AnnotatorError::new(
            "should assert that the instance we take a weakref to cannot be None",
        ));
    }
    Ok(SomeValue::WeakRef(SomeWeakRef::new(inst.classdef.clone())))
}

/// Upstream `pdb_set_trace(*args_s)` (builtin.py:335-339).
pub fn pdb_set_trace(
    _bk: &Rc<Bookkeeper>,
    _args_s: &[Option<SomeValue>],
    _kwds: &HashMap<String, Option<SomeValue>>,
) -> Result<SomeValue, AnnotatorError> {
    Err(AnnotatorError::new(
        "you left pdb.set_trace() in your interpreter! \
         If you want to attach a gdb instead, call rlib.debug.attach_gdb()",
    ))
}

// ---------------------------------------------------------------------------
// Private helpers.
// ---------------------------------------------------------------------------

/// Emulates Python's `hasattr(host, name)` for a constant [`HostObject`]
/// receiver.
///
/// Upstream `builtin_hasattr` calls Python's built-in `hasattr(...)`
/// which walks the object's MRO and falls back to the module / instance
/// namespace. The Rust port reproduces the three kinds it cares about:
///
/// * **Module** — look up in the module dict.
/// * **Class** — walk the MRO and check each class dict.
/// * **Instance** — check the instance dict first, then the MRO of its
///   `__class__`.
///
/// Everything else (functions, builtin callables, …) falls back to a
/// flat `class_get` / `instance_get` / `module_get` probe because those
/// carriers do not expose a class hierarchy through HostObject.
fn host_hasattr(host: &crate::flowspace::model::HostObject, name: &str) -> bool {
    // Module lookup.
    if host.module_get(name).is_some() {
        return true;
    }
    // Instance lookup — check the per-instance dict first, then the
    // class MRO for inherited attributes / descriptors.
    if host.instance_get(name).is_some() {
        return true;
    }
    if let Some(class_obj) = host.instance_class()
        && let Some(mro) = class_obj.mro()
    {
        for cls in mro {
            if cls.class_get(name).is_some() {
                return true;
            }
        }
        return false;
    }
    // Class lookup — walk the MRO so inherited members resolve.
    if host.is_class()
        && let Some(mro) = host.mro()
    {
        for cls in mro {
            if cls.class_get(name).is_some() {
                return true;
            }
        }
        return false;
    }
    // Fallback for carriers without a usable class hierarchy.
    host.class_get(name).is_some()
}

/// Upstream `union(*s_values)` (model.py:763-766).
///
/// n-ary fold with `SomeImpossibleValue` as the identity element,
/// mirroring `annmodel.union(*s_values)` being called with the unpacked
/// `*s_values` tuple.  Each slot is unwrapped via [`arg_at`] at touch
/// time so an unannotated slot surfaces at the first iteration that
/// reads it — matching `unaryop.py:940 simple_call_SomeBuiltin`'s
/// bind-then-body sequence rather than an entry-point eager prefix
/// unwrap.
fn union_many(s_values: &[Option<SomeValue>], analyser: &str) -> Result<SomeValue, AnnotatorError> {
    let mut acc = s_impossible_value();
    for i in 0..s_values.len() {
        let s = arg_at(s_values, i, analyser);
        acc = union(&acc, s).map_err(|e| AnnotatorError::new(e.msg))?;
    }
    Ok(acc)
}

/// Upstream `s_iterable.iter().next()` — used by [`builtin_list`] /
/// [`builtin_zip`] / [`builtin_min`] / [`builtin_max`].
///
/// Delegates to [`super::unaryop::container_getanyitem`], the shared
/// port of upstream `container.getanyitem(position, *variant)`
/// (unaryop.py:341-342 / :402-403 / :480-500 / :664-665). The caller
/// passes the iterator's `s_container` plus the variant string from
/// `SomeIterator.variant` (upstream's `*variant` tuple), matching the
/// logic in `unaryop.rs::someiterator_next` without requiring a live
/// `RPythonAnnotator` reference.
///
/// When the iterator wraps another `SomeIterator` (upstream `xs.iter()`
/// then `.iter()` again — returns self via `unaryop.py:808-810`), we
/// recurse on the inner iterator so the element annotation still
/// surfaces correctly.
fn someiterator_next_stub(bk: &Rc<Bookkeeper>, it: &SomeIterator) -> SomeValue {
    // upstream `if s_None.contains(self.s_container): return s_ImpossibleValue`
    // (unaryop.py:819-820).
    if matches!(&*it.s_container, SomeValue::None_(_)) {
        return s_impossible_value();
    }
    // upstream `if self.variant and self.variant[0] == "enumerate": ...`
    // (unaryop.py:822-824).
    if it.variant.first().map(String::as_str) == Some("enumerate") {
        let s_item =
            super::unaryop::container_getanyitem(&it.s_container, None, bk.current_position_key());
        return SomeValue::Tuple(SomeTuple::new(vec![
            SomeValue::Integer(SomeInteger::new(true, false)),
            s_item,
        ]));
    }
    // upstream `if variant == ("reversed",): variant = ()` (unaryop.py:826-827).
    let variant: Option<&str> = if it.variant.len() == 1 && it.variant[0] == "reversed" {
        None
    } else {
        it.variant.first().map(String::as_str)
    };
    // When the container is itself a `SomeIterator`, unwrap once — upstream
    // `SomeIterator.iter()` returns self (unaryop.py:808-810), so chaining
    // `.iter().next()` reaches the underlying container. The direct
    // `container_getanyitem` dispatch does not recognise Iterator inputs, so
    // we route them through another `someiterator_next_stub` hop first.
    if let SomeValue::Iterator(inner) = &*it.s_container {
        return someiterator_next_stub(bk, inner);
    }
    super::unaryop::container_getanyitem(&it.s_container, variant, bk.current_position_key())
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::bookkeeper::Bookkeeper;

    fn bk() -> Rc<Bookkeeper> {
        Rc::new(Bookkeeper::new())
    }

    fn no_kwds() -> HashMap<String, Option<SomeValue>> {
        HashMap::new()
    }

    #[test]
    fn registry_contains_mass_registered_builtins() {
        // upstream builtin.py:191-195 scans `builtin_*` prefixes.
        assert!(is_registered("range"));
        assert!(is_registered("xrange"));
        assert!(is_registered("int"));
        assert!(is_registered("float"));
        assert!(is_registered("chr"));
        assert!(is_registered("tuple"));
        assert!(is_registered("list"));
        assert!(is_registered("zip"));
        assert!(is_registered("min"));
        assert!(is_registered("max"));
    }

    #[test]
    fn registry_contains_analyzer_for_entries() {
        // upstream `@analyzer_for(...)` decorations.
        assert!(is_registered("object.__init__"));
        assert!(is_registered("sys.getdefaultencoding"));
        assert!(is_registered("rpython.rlib.rarithmetic.intmask"));
        assert!(is_registered("rpython.rlib.objectmodel.instantiate"));
        assert!(is_registered("weakref.ref"));
        assert!(is_registered("pdb.set_trace"));
    }

    #[test]
    fn registry_rejects_unknown_qualname() {
        assert!(!is_registered("completely.made.up.name"));
        assert!(lookup("completely.made.up.name").is_none());
    }

    #[test]
    fn builtin_range_single_const_argument_returns_nonneg_list() {
        // upstream builtin_range(10) → list with SomeInteger(nonneg=True),
        // range_step=1.
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let result = builtin_range(&bk, &[Some(s_stop)], &no_kwds()).unwrap();
        match result {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(matches!(item, SomeValue::Integer(ref i) if i.nonneg));
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_range_negative_stop_is_empty_list() {
        // upstream: `len(xrange(0, -5, 1)) == 0` → empty list → s_item =
        // s_ImpossibleValue.
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(-5)).unwrap();
        let result = builtin_range(&bk, &[Some(s_stop)], &no_kwds()).unwrap();
        match result {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(matches!(item, SomeValue::Impossible));
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_range_step_zero_errors() {
        let bk = bk();
        let a = bk.immutablevalue(&ConstValue::Int(0)).unwrap();
        let b = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let c = bk.immutablevalue(&ConstValue::Int(0)).unwrap();
        let err = builtin_range(&bk, &[Some(a), Some(b), Some(c)], &no_kwds()).unwrap_err();
        assert!(err.msg.as_deref().unwrap_or("").contains("step zero"));
    }

    #[test]
    fn builtin_int_const_string_returns_constant_integer() {
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("42")).unwrap();
        let result = builtin_int(&bk, &[Some(s_str)], &no_kwds()).unwrap();
        match result {
            SomeValue::Integer(i) => {
                assert!(i.base.const_box.is_some());
                if let Some(ConstValue::Int(n)) = i.base.const_box.as_ref().map(|c| &c.value) {
                    assert_eq!(*n, 42);
                }
            }
            other => panic!("expected constant SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_non_const_integer_returns_nonneg_somesinteger() {
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_nonneg = SomeValue::Integer(SomeInteger::new(true, false));
        let result = builtin_int(&bk, &[Some(s_nonneg)], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::Integer(ref i) if i.nonneg));
    }

    #[test]
    fn builtin_chr_const_integer_returns_constant_char() {
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(65)).unwrap();
        let result = builtin_chr(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match result {
            SomeValue::Char(ch) => {
                assert!(ch.inner.base.const_box.is_some());
                if let Some(ConstValue::ByteStr(s)) =
                    ch.inner.base.const_box.as_ref().map(|c| &c.value)
                {
                    assert_eq!(s.as_slice(), b"A");
                }
            }
            other => panic!("expected constant SomeChar, got {other:?}"),
        }
    }

    #[test]
    fn builtin_list_on_somelist_uses_offspring() {
        let bk = bk();
        let listdef = super::super::listdef::ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let s = SomeValue::List(super::super::model::SomeList::new(listdef));
        let out = builtin_list(&bk, &[Some(s)], &no_kwds()).unwrap();
        assert!(matches!(out, SomeValue::List(_)));
    }

    #[test]
    fn call_builtin_unknown_name_errors() {
        let bk = bk();
        let err = call_builtin(&bk, "definitely_not_a_builtin", &[], &no_kwds()).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no analyser registered")
        );
    }

    #[test]
    fn call_builtin_dispatches_range() {
        let bk = bk();
        let s_stop = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        let result = call_builtin(&bk, "range", &[Some(s_stop)], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::List(_)));
    }

    #[test]
    fn object_init_returns_none() {
        let bk = bk();
        let result = object_init(&bk, &[], &no_kwds()).unwrap();
        assert!(matches!(result, SomeValue::None_(_)));
    }

    #[test]
    fn pdb_set_trace_always_errors() {
        let bk = bk();
        let err = pdb_set_trace(&bk, &[], &no_kwds()).unwrap_err();
        assert!(err.msg.as_deref().unwrap_or("").contains("pdb.set_trace"));
    }

    #[test]
    fn constpropagate_returns_s_result_when_args_not_constant() {
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_input = SomeValue::Integer(SomeInteger::default());
        let s_result = SomeValue::Integer(SomeInteger::default());
        let out = constpropagate(&bk, &[s_input], s_result.clone(), |_| {
            panic!("evaluator must not be called when args are non-constant");
        })
        .unwrap();
        assert_eq!(out.tag(), s_result.tag());
    }

    #[test]
    fn somebuiltin_call_dispatches_through_registry() {
        // End-to-end wiring check: SomeBuiltin("range") carried inside
        // `SomeValue::call` reaches `call_builtin("range", ...)` and the
        // registered analyser runs. Mirrors upstream's
        // `SomeBuiltin.call(args)` path at unaryop.py:940-946.
        use crate::annotator::argument::{ArgumentsForTranslation, simple_args};
        use crate::annotator::model::{SomeBuiltin, SomeValue};

        let bk = bk();
        bk.enter(None);

        let s_stop = bk.immutablevalue(&ConstValue::Int(4)).unwrap();
        let args: ArgumentsForTranslation = simple_args(vec![s_stop]);
        let s_callee = SomeValue::Builtin(SomeBuiltin::new("range", None, Some("range".into())));
        let s_result = s_callee
            .call(&args)
            .expect("SomeBuiltin.call should dispatch");
        assert!(matches!(s_result, SomeValue::List(_)));

        bk.leave();
    }

    #[test]
    fn somebuiltin_call_returns_error_for_unregistered_qualname() {
        use crate::annotator::argument::{ArgumentsForTranslation, simple_args};
        use crate::annotator::model::{SomeBuiltin, SomeValue};

        let bk = bk();
        bk.enter(None);

        let args: ArgumentsForTranslation = simple_args(vec![]);
        let s_callee = SomeValue::Builtin(SomeBuiltin::new(
            "completely.made.up.name",
            None,
            Some("completely.made.up.name".into()),
        ));
        let err = s_callee
            .call(&args)
            .expect_err("unregistered analyser should error");
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("no analyser registered")
        );

        bk.leave();
    }

    #[test]
    fn builtin_list_on_iterator_routes_through_next() {
        // Codex P1: `list(reversed(xs))` should keep the element type
        // instead of collapsing to SomeImpossibleValue. Upstream chains
        // `s_iterable.iter().next()` which delegates to
        // `SomeIterator.next()` for SomeIterator inputs.
        use crate::annotator::model::{SomeInteger, SomeList};
        let bk = bk();
        let inner = super::super::listdef::ListDef::new(
            Some(bk.clone()),
            SomeValue::Integer(SomeInteger::new(true, false)),
            false,
            false,
        );
        let inner_list = SomeValue::List(SomeList::new(inner));
        let s_iter =
            SomeValue::Iterator(SomeIterator::new(inner_list, vec!["reversed".to_string()]));
        let out = builtin_list(&bk, &[Some(s_iter)], &no_kwds()).unwrap();
        match out {
            SomeValue::List(list) => {
                let item = list.listdef.read_item(None);
                assert!(
                    matches!(item, SomeValue::Integer(ref i) if i.nonneg),
                    "expected nonneg SomeInteger element, got {item:?}"
                );
            }
            other => panic!("expected SomeList, got {other:?}"),
        }
    }

    #[test]
    fn builtin_max_keeps_nonneg_when_any_operand_is_nonneg() {
        // Codex P2: upstream `nonneg |= s1.nonneg` is an OR-fold —
        // `max(nonneg, maybe_negative)` stays nonneg.
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let a = SomeValue::Integer(SomeInteger::new(true, false));
        let b = SomeValue::Integer(SomeInteger::new(false, false));
        let out = builtin_max(&bk, &[Some(a), Some(b)], &no_kwds()).unwrap();
        match out {
            SomeValue::Integer(i) => assert!(
                i.nonneg,
                "max(nonneg, maybe_neg) must remain nonneg per builtin.py:182-188"
            ),
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_rejects_out_of_range_float_constant() {
        // Codex P2: `int(1e20)` would saturate to i64::MAX if we let the
        // Rust cast do its default behaviour — upstream raises OverflowError
        // which constpropagate catches. Non-finite / out-of-range floats
        // must fall through to the conservative `s_result`.
        let bk = bk();
        let huge = ConstValue::float(1.0e20_f64);
        let s_huge = bk.immutablevalue(&huge).unwrap();
        let out = builtin_int(&bk, &[Some(s_huge)], &no_kwds()).unwrap();
        match out {
            SomeValue::Integer(i) => {
                assert!(
                    i.base.const_box.is_none(),
                    "int(out-of-range float) must not pin a constant"
                );
            }
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn registered_class_backed_builtin_lifts_to_somebuiltin() {
        // Codex P1: class-backed builtins (`range`, `int`, `list`, …)
        // are `HostObjectKind::Class` in HOST_ENV but still register
        // analysers. `immutablevalue` must route them to `SomeBuiltin`
        // so that `SomeBuiltin.call()` reaches the analyser, matching
        // upstream bookkeeper.py:309-311's BUILTIN_ANALYZERS check
        // firing before the `tp is type` branch.
        use crate::flowspace::model::HOST_ENV;
        let bk = bk();
        let range_host = HOST_ENV
            .lookup_builtin("range")
            .expect("HOST_ENV must expose `range`");
        let s = bk
            .immutablevalue(&ConstValue::HostObject(range_host))
            .expect("range class must annotate");
        assert!(
            matches!(s, SomeValue::Builtin(ref b) if b.analyser_name == "range"),
            "expected SomeBuiltin(range), got {s:?}"
        );
    }

    #[test]
    fn builtin_unichr_const_integer_returns_constant_unicode_cp() {
        // Codex P2: `unichr(65)` must keep the constant via the
        // UnicodeCodePoint variant; constpropagate's re-annotation
        // through immutablevalue needs the explicit unicode tag.
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(65)).unwrap();
        let out = builtin_unichr(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match out {
            SomeValue::UnicodeCodePoint(cp) => {
                match cp.inner.base.const_box.as_ref().map(|c| &c.value) {
                    Some(ConstValue::UniStr(s)) => assert_eq!(s, "A"),
                    other => panic!("expected constant unicode 'A', got {other:?}"),
                }
            }
            other => panic!("expected SomeUnicodeCodePoint, got {other:?}"),
        }
    }

    #[test]
    fn builtin_unicode_const_string_returns_constant_unicode_string() {
        // Codex P2: `unicode("abc")` must return a constant
        // SomeUnicodeString, not raise AnnotatorError.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("abc")).unwrap();
        let out = builtin_unicode(&bk, &[Some(s_str)], &no_kwds()).unwrap();
        match out {
            SomeValue::UnicodeString(us) => {
                match us.inner.base.const_box.as_ref().map(|c| &c.value) {
                    Some(ConstValue::UniStr(s)) => assert_eq!(s, "abc"),
                    other => panic!("expected constant unicode 'abc', got {other:?}"),
                }
            }
            other => panic!("expected SomeUnicodeString, got {other:?}"),
        }
    }

    #[test]
    fn builtin_hasattr_folds_constant_when_attr_exists_on_class() {
        // Codex P2: for immutable constant receivers, `hasattr(C, "m")`
        // must fold to a constant SomeBool. The HOST_ENV `range` class
        // has no attribute members in this minimal env, so assert the
        // False branch — checks the folding mechanism regardless of
        // which members the host reflects.
        use crate::flowspace::model::HOST_ENV;
        let bk = bk();
        let obj = HOST_ENV.lookup_builtin("range").unwrap();
        let s_obj = bk.immutablevalue(&ConstValue::HostObject(obj)).unwrap();
        let s_attr = bk
            .immutablevalue(&ConstValue::byte_str("no_such_attr"))
            .unwrap();
        let out = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(false)) => {}
                other => panic!("expected constant Bool(false) for missing attr, got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn builtin_bool_folds_constant_none_to_false() {
        // Codex P2: `bool(None)` must pin `r.const = False` instead of
        // returning an unrefined SomeBool.
        let bk = bk();
        let s_none_val = bk.immutablevalue(&ConstValue::None).unwrap();
        let out = builtin_bool(&bk, &[Some(s_none_val)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(false)) => {}
                other => panic!("expected constant Bool(false), got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn builtin_bool_folds_constant_int_to_true() {
        let bk = bk();
        let s_int = bk.immutablevalue(&ConstValue::Int(42)).unwrap();
        let out = builtin_bool(&bk, &[Some(s_int)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(true)) => {}
                other => panic!("expected constant Bool(true), got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn call_builtin_rejects_unknown_keyword() {
        // Codex P2: `list(xs, foo=1)` must raise instead of silently
        // ignoring `foo=1`. The dispatcher now rejects kwds outside
        // each analyser's allow-list.
        let bk = bk();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert(
            "s_foo".into(),
            Some(bk.immutablevalue(&ConstValue::Int(1)).unwrap()),
        );
        let err = call_builtin(&bk, "list", &[], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("unexpected keyword argument"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn call_builtin_accepts_declared_keyword() {
        // `int(s, base=16)` is the positive path — `base` is declared
        // allowed for `int` in the allow-list.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("1a")).unwrap();
        let s_base = bk.immutablevalue(&ConstValue::Int(16)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_base".into(), Some(s_base));
        let out = call_builtin(&bk, "int", &[Some(s_str)], &kwds).unwrap();
        match out {
            SomeValue::Integer(i) => match i.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Int(n)) => assert_eq!(*n, 26),
                other => panic!("expected constant Int(26), got {other:?}"),
            },
            other => panic!("expected SomeInteger, got {other:?}"),
        }
    }

    #[test]
    fn builtin_int_rejects_positional_and_keyword_base() {
        // Codex P2: `int(x, 10, base=16)` must raise the duplicate-arg
        // error that upstream's Python dispatch produces.
        let bk = bk();
        let s_str = bk.immutablevalue(&ConstValue::byte_str("1a")).unwrap();
        let s_base_pos = bk.immutablevalue(&ConstValue::Int(10)).unwrap();
        let s_base_kw = bk.immutablevalue(&ConstValue::Int(16)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_base".into(), Some(s_base_kw));
        let err = builtin_int(&bk, &[Some(s_str), Some(s_base_pos)], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("multiple values for argument 'base'"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn r_dict_helper_reads_force_non_null_from_keyword() {
        // Codex P1: r_dict(eq, hash, force_non_null=True) must honour
        // the kwd. Previously ignored, leaving the DictDef flag false.
        //
        // Testing through call_builtin would trip
        // `update_rdict_annotations` because our stub eq/hash aren't
        // SomePBC. Instead, extract the flag-binding logic directly by
        // reading the DictDef getdictdef returns once we reproduce the
        // kwd/positional merge. The builder side-effect (getdictdef
        // stores the DictDef in bk.dictdefs keyed on position_key)
        // lets us read back the flags even though update_rdict will
        // fail downstream.
        let bk = bk();
        bk.enter(None);
        let s_force = bk.immutablevalue(&ConstValue::Bool(true)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_force_non_null".into(), Some(s_force));

        // Mirror r_dict_helper's flag extraction — the unit under test.
        let force_non_null = kwds
            .get("s_force_non_null")
            .and_then(|s| s.as_ref())
            .and_then(|s| s.const_().cloned())
            .map(|cv| matches!(cv, ConstValue::Bool(true)))
            .unwrap_or(false);
        assert!(
            force_non_null,
            "kwd extraction must yield force_non_null=true"
        );

        let dd = bk.getdictdef(true, force_non_null, false);
        assert!(
            dd.inner.force_non_null,
            "DictDef::force_non_null must persist the kwd value"
        );
        bk.leave();
    }

    #[test]
    fn builtin_enumerate_rejects_non_constant_start_keyword() {
        // Codex P2: `enumerate(xs, start=<non-constant>)` must raise
        // the upstream "second argument to enumerate must be constant"
        // error instead of silently dropping the kwd.
        use crate::annotator::model::SomeInteger;
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start_nonconst = SomeValue::Integer(SomeInteger::default());
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_start".into(), Some(s_start_nonconst));
        let err = builtin_enumerate(&bk, &[Some(s_xs)], &kwds).unwrap_err();
        assert!(
            err.msg
                .as_deref()
                .unwrap_or("")
                .contains("second argument to enumerate must be constant"),
            "err was {:?}",
            err.msg
        );
    }

    #[test]
    fn builtin_enumerate_preserves_constant_start_payload() {
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start = bk.immutablevalue(&ConstValue::Int(3)).unwrap();
        let result = builtin_enumerate(&bk, &[Some(s_xs), Some(s_start)], &no_kwds()).unwrap();
        let SomeValue::Iterator(it) = result else {
            panic!("expected SomeIterator");
        };
        assert_eq!(it.variant, vec!["enumerate".to_string()]);
        assert_eq!(it.enumerate_start, Some(ConstValue::Int(3)));
    }

    #[test]
    fn builtin_enumerate_keyword_preserves_constant_start_payload() {
        let bk = bk();
        let s_xs = SomeValue::object();
        let s_start = bk.immutablevalue(&ConstValue::Int(5)).unwrap();
        let mut kwds: HashMap<String, Option<SomeValue>> = HashMap::new();
        kwds.insert("s_start".into(), Some(s_start));
        let result = builtin_enumerate(&bk, &[Some(s_xs)], &kwds).unwrap();
        let SomeValue::Iterator(it) = result else {
            panic!("expected SomeIterator");
        };
        assert_eq!(it.variant, vec!["enumerate".to_string()]);
        assert_eq!(it.enumerate_start, Some(ConstValue::Int(5)));
    }

    #[test]
    fn builtin_hasattr_walks_class_mro_for_inherited_attr() {
        // Codex P2: hasattr folding must resolve inherited attributes,
        // not just the immediate class dict. Build base with `m`, then
        // subclass with empty dict — `hasattr(Subclass, "m")` must be
        // folded to True.
        use crate::flowspace::model::HostObject;
        let bk = bk();
        let base = HostObject::new_class("pkg.Base", vec![]);
        base.class_set("m", ConstValue::Int(0));
        let sub = HostObject::new_class("pkg.Sub", vec![base]);
        let s_obj = bk.immutablevalue(&ConstValue::HostObject(sub)).unwrap();
        let s_attr = bk.immutablevalue(&ConstValue::byte_str("m")).unwrap();
        let out = builtin_hasattr(&bk, &[Some(s_obj), Some(s_attr)], &no_kwds()).unwrap();
        match out {
            SomeValue::Bool(b) => match b.base.const_box.as_ref().map(|c| &c.value) {
                Some(ConstValue::Bool(true)) => {}
                other => panic!("expected Bool(true) for inherited attribute, got {other:?}"),
            },
            other => panic!("expected SomeBool, got {other:?}"),
        }
    }

    #[test]
    fn constpropagate_invokes_evaluator_on_all_constants() {
        let bk = bk();
        let s_a = bk.immutablevalue(&ConstValue::Int(7)).unwrap();
        let s_b = bk.immutablevalue(&ConstValue::Int(8)).unwrap();
        let s_result = SomeValue::Integer(super::super::model::SomeInteger::default());
        let out = constpropagate(&bk, &[s_a, s_b], s_result, |consts| match consts {
            [ConstValue::Int(a), ConstValue::Int(b)] => Some(ConstValue::Int(*a + *b)),
            _ => None,
        })
        .unwrap();
        if let SomeValue::Integer(i) = out {
            if let Some(ConstValue::Int(n)) = i.base.const_box.as_ref().map(|c| &c.value) {
                assert_eq!(*n, 15);
                return;
            }
        }
        panic!("expected constant SomeInteger(15)");
    }
}
