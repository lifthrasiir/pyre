//! rpython/rlib/jit.py — User-facing JIT hint API.
//!
//! Line-by-line structural port of RPython's `rpython/rlib/jit.py`.
//!
//! Item ordering in this file follows jit.py exactly. Each section is
//! annotated with the corresponding jit.py line number.
//!
//! Functions in this module are no-ops at interpreter runtime. When the
//! `#[jit_interp]` proc macro processes user code, calls to these functions
//! are recognized and lowered to JitCode operations:
//!
//! - `promote(x)` → `int_guard_value` / `ref_guard_value` / `float_guard_value`
//! - `record_exact_class(v, cls)` → RECORD_EXACT_CLASS IR op
//! - `record_exact_value(v, c)` → RECORD_EXACT_VALUE IR op

// ── DEBUG_ELIDABLE_FUNCTIONS ──
// rlib/jit.py:11

/// rlib/jit.py:11 — enables runtime consistency checks on elidable functions.
pub const DEBUG_ELIDABLE_FUNCTIONS: bool = false;

// ── elidable, purefunction ──
// rlib/jit.py:13 — `@elidable` decorator → proc macro `#[elidable]`
// rlib/jit.py:75 — `purefunction` is a deprecated alias for `elidable`.
//                   Rust users should write `#[elidable]` directly.

// ── hint ──
// rlib/jit.py:80-97

/// Hint for the JIT.
///
/// Possible hint kinds (RPython keyword arguments):
/// - promote — promote the argument from a variable into a constant
/// - promote_string — same, but promote string by value
/// - promote_unicode — same, but promote unicode string by value
/// - access_directly — directly access a virtualizable as a structure
/// - fresh_virtualizable — virtualizable was just allocated
/// - force_virtualizable — force the virtualizable early
///
/// In Rust, these are separate functions: `promote()`, `promote_string()`, etc.
/// Virtualizable hints are handled by the `virtualizable!` macro.
///
/// rlib/jit.py:80 — `hint(x, **kwds)` returns x unchanged.
#[inline(always)]
pub fn hint<T>(x: T) -> T {
    x
}

// ── promote ──
// rlib/jit.py:100-124

/// Promotes a variable in a trace to a constant.
///
/// When a variable is promoted, a guard is inserted that assumes the value
/// of the variable is constant. In other words, the value of the variable
/// is checked to be the same as it was at trace collection time.  Once the
/// variable is assumed constant, more aggressive constant folding may be
/// possible.
///
/// If however, the guard fails frequently, a bridge will be generated
/// this time assuming the constancy of the variable under its new value.
/// This optimisation should be used carefully, as in extreme cases, where
/// the promoted variable is not very constant at all, code explosion can
/// occur. In turn this leads to poor performance.
///
/// Overpromotion is characterised by a cascade of bridges branching from
/// very similar guard_value opcodes, each guarding the same variable under
/// a different value.
///
/// Note that promoting a string with `promote` will promote by pointer.
/// To promote a string by value, see `promote_string`.
///
/// rlib/jit.py:101 — `promote(x) = hint(x, promote=True)`
#[inline(always)]
pub fn promote<T: Copy>(x: T) -> T {
    hint(x)
}

// ── promote_string ──
// rlib/jit.py:127

/// Promote a string by value (not by pointer).
///
/// rlib/jit.py:127 — `promote_string(x) = hint(x, promote_string=True)`
#[inline(always)]
pub fn promote_string<T: Copy>(x: T) -> T {
    hint(x)
}

// ── promote_unicode ──
// rlib/jit.py:130

/// Promote a unicode string by value (not by pointer).
///
/// rlib/jit.py:130 — `promote_unicode(x) = hint(x, promote_unicode=True)`
#[inline(always)]
pub fn promote_unicode<T: Copy>(x: T) -> T {
    hint(x)
}

// ── dont_look_inside, look_inside, unroll_safe, loop_invariant ──
// rlib/jit.py:133, 142, 151, 162
//
// These are decorators in RPython, implemented as proc macro attributes
// in majit-macros: #[dont_look_inside], #[unroll_safe], #[loop_invariant].
// `look_inside` is deprecated in RPython; no majit equivalent needed.

// ── _get_args ──
// rlib/jit.py:172
//
// In RPython `_get_args(func)` extracts parameter names for elidable_promote
// and look_inside_iff code generation. In majit, both proc macros collect
// parameters from `ItemFn::sig.inputs` inline; no separate helper needed.

// ── elidable_promote, purefunction_promote, look_inside_iff, oopspec, not_in_trace ──
// rlib/jit.py:180, 203, 208, 250, 260 — all proc macros in majit-macros.

// ── isconstant ──
// rlib/jit.py:271-279

/// While tracing, returns whether or not the value is currently known to be
/// constant. This is not perfect, values can become constant later. Mostly for
/// use with look_inside_iff.
///
/// This is for advanced usage only.
///
/// rlib/jit.py:271
#[inline(always)]
pub fn isconstant<T: ?Sized>(_value: &T) -> bool {
    false
}

// ── isvirtual ──
// rlib/jit.py:283-292

/// Returns if this value is virtual, while tracing. Can be wrong in both
/// directions. It tries to be conservative by default, but can also sometimes
/// return True for something that does not end up completely virtual.
///
/// This is for advanced usage only.
///
/// rlib/jit.py:283
#[inline(always)]
pub fn isvirtual<T: ?Sized>(_value: &T) -> bool {
    false
}

// ── loop_unrolling_heuristic ──
// rlib/jit.py:295-301

/// In which cases iterating over items of lst can be unrolled.
///
/// rlib/jit.py:295
#[inline(always)]
pub fn loop_unrolling_heuristic<T>(lst: &[T], size: usize, cutoff: usize) -> bool {
    // rlib/jit.py:301 — `size == 0 or (isconstant(size) and (isvirtual(lst) or size <= cutoff))`
    // `isvirtual(lst)` is often lying; also require size to be constant.
    size == 0 || (isconstant(&size) && (isvirtual(lst) || size <= cutoff))
}

// ── we_are_jitted ──
// rlib/jit.py:355-358

/// Considered as true during tracing and blackholing,
/// so its consequences are reflected into jitted code.
///
/// rlib/jit.py:355
#[inline(always)]
pub fn we_are_jitted() -> bool {
    majit_backend::we_are_jitted()
}

// ── _we_are_jitted ──
// rlib/jit.py:360-361

/// rlib/jit.py:360 — `_we_are_jitted = CDefinedIntSymbolic('0 /* we are not jitted here */', default=0)`
///
/// A C-defined symbolic integer representing the JIT runtime flag at translation
/// time. In majit this is a runtime thread-local flag exposed via `we_are_jitted()`.
#[allow(non_upper_case_globals)]
pub const _we_are_jitted: i32 = 0;

// ── _get_virtualizable_token ──
// rlib/jit.py:363-369

/// An obscure API to get vable token. Used by _vmprof.
///
/// rlib/jit.py:363 — returns null GCREF pointer for untranslated/non-virtualizable frames.
#[inline(always)]
pub fn _get_virtualizable_token<T>(_frame: &T) -> *mut () {
    std::ptr::null_mut()
}

// ── current_trace_length ──
// rlib/jit.py:408-414

/// During JIT tracing, returns the current trace length (as a constant).
/// If not tracing, returns -1.
///
/// rlib/jit.py:409
#[inline(always)]
pub fn current_trace_length() -> i64 {
    -1
}

// ── jit_debug ──
// rlib/jit.py:416-421

/// When JITted, cause an extra operation JIT_DEBUG to appear in
/// the graphs.  Should not be left after debugging.
///
/// rlib/jit.py:417
#[inline(always)]
pub fn jit_debug(_string: &str, _arg1: i64, _arg2: i64, _arg3: i64, _arg4: i64) {}

// ── assert_green ──
// rlib/jit.py:423-428

/// Very strong assert: checks that 'value' is a green
/// (a JIT compile-time constant).
///
/// rlib/jit.py:425
#[inline(always)]
pub fn assert_green<T>(_value: &T) {}

// ── AssertGreenFailed ──
// rlib/jit.py:430-431

/// rlib/jit.py:430
#[derive(Debug)]
pub struct AssertGreenFailed;

impl std::fmt::Display for AssertGreenFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AssertGreenFailed")
    }
}

impl std::error::Error for AssertGreenFailed {}

// ── jit_callback ──
// rlib/jit.py:434-457

/// Use as a decorator for C callback functions, to insert a
/// jitdriver.jit_merge_point() at the start.  Only for callbacks
/// that typically invoke more app-level Python code.
///
/// rlib/jit.py:434
///
/// In majit, this concept maps to `#[jit_interp]` on the callback wrapper.
/// No standalone runtime helper is required; left as documentation for parity.
pub const fn jit_callback(_name: &'static str) {}

// ── virtual_ref ──
// rlib/jit.py:463-473

/// Creates a 'vref' object that contains a reference to 'x'.  Calls
/// to virtual_ref/virtual_ref_finish must be properly nested.  The idea
/// is that the object 'x' is supposed to be JITted as a virtual between
/// the calls to virtual_ref and virtual_ref_finish, but the 'vref'
/// object can escape at any point in time.  If at runtime it is
/// dereferenced, it returns 'x', which is then forced.
///
/// rlib/jit.py:465
#[inline(always)]
pub fn virtual_ref<T>(x: &T) -> DirectJitVRef<T> {
    // rlib/jit.py:519 — DirectJitVRef.__init__: assert x is not None
    // In Rust a `&T` is never null, so this assertion is statically satisfied.
    DirectJitVRef {
        inner: DirectVRef {
            _x: x as *const T,
            _state: VRefState::NonForced,
        },
    }
}

// ── virtual_ref_finish ──
// rlib/jit.py:475-480

/// See docstring in virtual_ref(x).
///
/// rlib/jit.py:477
#[inline(always)]
pub fn virtual_ref_finish<T>(vref: &mut DirectJitVRef<T>, x: &T) {
    // rlib/jit.py:479 — keepalive_until_here(x); otherwise the whole function
    // call is removed. In Rust this is implicit via lifetime analysis.
    _virtual_ref_finish(vref, x);
}

// ── non_virtual_ref ──
// rlib/jit.py:482-485

/// Creates a 'vref' that just returns x when called; nothing more special.
/// Used for None or for frames outside JIT scope.
///
/// rlib/jit.py:482
#[inline(always)]
pub fn non_virtual_ref<T>(x: &T) -> DirectVRef<T> {
    DirectVRef {
        _x: x as *const T,
        _state: VRefState::NonForced,
    }
}

// ── InvalidVirtualRef ──
// rlib/jit.py:487-491

/// Raised if we try to call a non-forced virtualref after the call to
/// virtual_ref_finish.
///
/// rlib/jit.py:487
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidVirtualRef;

impl std::fmt::Display for InvalidVirtualRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "InvalidVirtualRef")
    }
}

impl std::error::Error for InvalidVirtualRef {}

// ---------- implementation-specific ----------
// rlib/jit.py:493

// ── DirectVRef ──
// rlib/jit.py:495-515

/// rlib/jit.py:498 — _state field
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VRefState {
    /// rlib/jit.py:498 — 'non-forced'
    NonForced,
    /// rlib/jit.py:502 — 'forced'
    Forced,
    /// rlib/jit.py:515 — 'invalid'
    Invalid,
}

/// rlib/jit.py:495 — `class DirectVRef(object)`
#[derive(Debug)]
pub struct DirectVRef<T> {
    pub(crate) _x: *const T,
    pub(crate) _state: VRefState,
}

impl<T> DirectVRef<T> {
    /// rlib/jit.py:500 — `def __call__(self):`
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        match self._state {
            // rlib/jit.py:501-502
            VRefState::NonForced => {
                self._state = VRefState::Forced;
                Ok(unsafe { &*self._x })
            }
            // rlib/jit.py:503-504
            VRefState::Invalid => Err(InvalidVirtualRef),
            VRefState::Forced => Ok(unsafe { &*self._x }),
        }
    }

    /// rlib/jit.py:507 — `@property def virtual(self):`
    pub fn virtual_(&self) -> bool {
        self._state == VRefState::NonForced
    }

    /// rlib/jit.py:513 — `def _finish(self):`
    pub(crate) fn _finish(&mut self) {
        if self._state == VRefState::NonForced {
            self._state = VRefState::Invalid;
        }
    }
}

// ── DirectJitVRef ──
// rlib/jit.py:517-520

/// rlib/jit.py:517 — `class DirectJitVRef(DirectVRef)`
///
/// Rust has no inheritance; modelled as a newtype wrapping `DirectVRef<T>`.
/// Access to the base class fields goes through `self.inner`.
#[derive(Debug)]
pub struct DirectJitVRef<T> {
    pub(crate) inner: DirectVRef<T>,
}

impl<T> DirectJitVRef<T> {
    /// rlib/jit.py:500 — delegate to DirectVRef.__call__
    pub fn force(&mut self) -> Result<&T, InvalidVirtualRef> {
        self.inner.force()
    }

    /// rlib/jit.py:507 — delegate to DirectVRef.virtual
    pub fn virtual_(&self) -> bool {
        self.inner.virtual_()
    }
}

// ── _virtual_ref_finish ──
// rlib/jit.py:522-524

/// rlib/jit.py:522 — `def _virtual_ref_finish(vref, x):`
///
/// Private helper that asserts identity and calls `_finish()`.
#[inline(always)]
fn _virtual_ref_finish<T>(vref: &mut DirectJitVRef<T>, x: &T) {
    // rlib/jit.py:523 — assert vref._x is x
    debug_assert!(
        std::ptr::eq(vref.inner._x, x),
        "Invalid call to virtual_ref_finish"
    );
    // rlib/jit.py:524 — vref._finish()
    vref.inner._finish();
}

// ── vref_None ──
// rlib/jit.py:554

/// Pre-made vref for null/absent frames.
///
/// rlib/jit.py:554 — `vref_None = non_virtual_ref(None)`
pub const VREF_NONE: DirectVRef<()> = DirectVRef {
    _x: std::ptr::null(),
    _state: VRefState::NonForced,
};

// ── JitHintError ──
// rlib/jit.py:559-560

/// Inconsistency in the JIT hints.
///
/// rlib/jit.py:559
#[derive(Debug)]
pub struct JitHintError(pub String);

impl std::fmt::Display for JitHintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitHintError: {}", self.0)
    }
}

impl std::error::Error for JitHintError {}

// ── ENABLE_ALL_OPTS ──
// rlib/jit.py:562-563

/// rlib/jit.py:562
pub const ENABLE_ALL_OPTS: &str = "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll";

// ── PARAMETER_DOCS ──
// rlib/jit.py:565-586

/// Documentation strings for each JIT parameter.
///
/// rlib/jit.py:565 — `PARAMETER_DOCS` dict
pub fn parameter_doc(name: &str) -> &'static str {
    match name {
        "threshold" => "number of times a loop has to run for it to become hot",
        "function_threshold" => {
            "number of times a function must run for it to become traced from start"
        }
        "trace_eagerness" => {
            "number of times a guard has to fail before we start compiling a bridge"
        }
        "decay" => "amount to regularly decay counters by (0=none, 1000=max)",
        "trace_limit" => {
            "number of recorded operations before we abort tracing with ABORT_TOO_LONG"
        }
        "inlining" => "inline functions or not (1/0)",
        "loop_longevity" => {
            "a parameter controlling how long loops will be kept before being freed, an estimate"
        }
        "retrace_limit" => "how many times we can try retracing before giving up",
        "pureop_historylength" => {
            "how many pure operations the optimizer should remember for CSE (internal)"
        }
        "max_retrace_guards" => "number of extra guards a retrace can cause",
        "max_unroll_loops" => "number of extra unrollings a loop can cause",
        "disable_unrolling" => "after how many operations we should not unroll",
        "enable_opts" => "optimizations to enable, or all",
        "max_unroll_recursion" => "how many levels deep to unroll a recursive function",
        "vec" => "turn on the vectorization optimization (vecopt)",
        "vec_cost" => {
            "threshold for which traces to bail. Unpacking increases the counter, \
             vector operation decrease the cost"
        }
        "vec_all" => "try to vectorize trace loops that occur outside of the numpypy library",
        _ => "(unknown parameter)",
    }
}

// ── PARAMETERS ──
// rlib/jit.py:588-605

/// Default JIT parameters.
///
/// rlib/jit.py:588
pub const PARAMETERS: JitParameters = JitParameters {
    threshold: 1039,          // just above 1024, prime
    function_threshold: 1619, // slightly more than one above, also prime
    trace_eagerness: 200,
    decay: 40,
    trace_limit: 6000,
    inlining: true,
    loop_longevity: 1000,
    retrace_limit: 0,
    pureop_historylength: 16,
    max_retrace_guards: 15,
    max_unroll_loops: 0,
    disable_unrolling: 200,
    enable_opts: EnableOpts::All,
    max_unroll_recursion: 7,
    vec: false,
    vec_all: false,
    vec_cost: 0,
};

/// rlib/jit.py:588 — PARAMETERS dict
#[derive(Debug, Clone)]
pub struct JitParameters {
    pub threshold: u32,
    pub function_threshold: u32,
    pub trace_eagerness: u32,
    pub decay: u32,
    pub trace_limit: u32,
    pub inlining: bool,
    pub loop_longevity: u32,
    pub retrace_limit: u32,
    pub pureop_historylength: u32,
    pub max_retrace_guards: u32,
    pub max_unroll_loops: u32,
    pub disable_unrolling: u32,
    pub enable_opts: EnableOpts,
    pub max_unroll_recursion: u32,
    pub vec: bool,
    pub vec_all: bool,
    pub vec_cost: u32,
}

/// rlib/jit.py:562,600
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnableOpts {
    All,
    Custom(String),
}

// ── unroll_parameters ──
// rlib/jit.py:606

/// rlib/jit.py:606 — `unroll_parameters = unrolling_iterable(PARAMETERS.items())`
///
/// The canonical list of (name, default_value_as_i64) pairs used by
/// `set_user_param` to iterate over all parameters in a fixed order.
pub const UNROLL_PARAMETERS: &[(&str, i64)] = &[
    ("threshold", 1039),
    ("function_threshold", 1619),
    ("trace_eagerness", 200),
    ("decay", 40),
    ("trace_limit", 6000),
    ("inlining", 1),
    ("loop_longevity", 1000),
    ("retrace_limit", 0),
    ("pureop_historylength", 16),
    ("max_retrace_guards", 15),
    ("max_unroll_loops", 0),
    ("disable_unrolling", 200),
    ("max_unroll_recursion", 7),
    ("vec", 0),
    ("vec_all", 0),
    ("vec_cost", 0),
];

// ── JitDriver ──
// rlib/jit.py:610+
//
// Implemented in `majit-metainterp/src/jitdriver.rs` as `JitDriver<State>`.

// ── _set_param ──
// rlib/jit.py:812-816

/// rlib/jit.py:812 — `def _set_param(driver, name, value)`
///
/// Private dispatch to the driver-level parameter setter.
fn _set_param(params: &mut JitParameters, name: &str, value: i64) -> Result<(), JitHintError> {
    match name {
        "threshold" => params.threshold = value as u32,
        "function_threshold" => params.function_threshold = value as u32,
        "trace_eagerness" => params.trace_eagerness = value as u32,
        "decay" => params.decay = value as u32,
        "trace_limit" => params.trace_limit = value as u32,
        "inlining" => params.inlining = value != 0,
        "loop_longevity" => params.loop_longevity = value as u32,
        "retrace_limit" => params.retrace_limit = value as u32,
        "pureop_historylength" => params.pureop_historylength = value as u32,
        "max_retrace_guards" => params.max_retrace_guards = value as u32,
        "max_unroll_loops" => params.max_unroll_loops = value as u32,
        "disable_unrolling" => params.disable_unrolling = value as u32,
        "max_unroll_recursion" => params.max_unroll_recursion = value as u32,
        "vec" => params.vec = value != 0,
        "vec_all" => params.vec_all = value != 0,
        "vec_cost" => params.vec_cost = value as u32,
        _ => return Err(JitHintError(format!("unknown JIT parameter: {name}"))),
    }
    Ok(())
}

// ── set_param ──
// rlib/jit.py:818-822

/// Set a JIT parameter by name.
///
/// rlib/jit.py:819 — raises ValueError on unknown name.
pub fn set_param(params: &mut JitParameters, name: &str, value: i64) -> Result<(), JitHintError> {
    _set_param(params, name, value)
}

// ── set_param_to_default ──
// rlib/jit.py:824-827

/// Set a JIT parameter to its default value.
///
/// rlib/jit.py:825
pub fn set_param_to_default(params: &mut JitParameters, name: &str) -> Result<(), JitHintError> {
    for (n, default) in UNROLL_PARAMETERS {
        if *n == name {
            return _set_param(params, name, *default);
        }
    }
    Err(JitHintError(format!("unknown JIT parameter: {name}")))
}

// ── TraceLimitTooHigh ──
// rlib/jit.py:829-833

/// This is raised when the trace limit is too high for the chosen
/// opencoder model, recompile your interpreter with 'big' as
/// jit_opencoder_model.
///
/// rlib/jit.py:829
#[derive(Debug)]
pub struct TraceLimitTooHigh;

impl std::fmt::Display for TraceLimitTooHigh {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TraceLimitTooHigh")
    }
}

impl std::error::Error for TraceLimitTooHigh {}

// ── set_user_param ──
// rlib/jit.py:836-874

/// Set the tunable JIT parameters from a user-supplied string
/// following the format 'param=value,param=value', or 'off' to
/// disable the JIT.  For programmatic setting of parameters, use
/// directly `set_param`.
///
/// rlib/jit.py:836
pub fn set_user_param(params: &mut JitParameters, text: &str) -> Result<(), JitHintError> {
    // rlib/jit.py:842-845
    if text == "off" {
        set_param(params, "threshold", -1)?;
        set_param(params, "function_threshold", -1)?;
        return Ok(());
    }
    // rlib/jit.py:846-849
    if text == "default" {
        for (name1, _) in UNROLL_PARAMETERS {
            set_param_to_default(params, name1)?;
        }
        return Ok(());
    }
    // rlib/jit.py:850-874
    for s in text.split(',') {
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        let parts: Vec<&str> = s.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(JitHintError(format!("malformed token: {s}")));
        }
        let name = parts[0].trim();
        let value = parts[1].trim();
        if name == "enable_opts" {
            params.enable_opts = if value == "all" {
                EnableOpts::All
            } else {
                EnableOpts::Custom(value.to_string())
            };
            continue;
        }
        // rlib/jit.py:860 — for name1, _ in unroll_parameters:
        let mut found = false;
        for (name1, _) in UNROLL_PARAMETERS {
            if *name1 == name && *name1 != "enable_opts" {
                let ivalue: i64 = value
                    .parse()
                    .map_err(|_| JitHintError(format!("invalid value for {name}: {value}")))?;
                set_param(params, name1, ivalue)?;
                found = true;
                break;
            }
        }
        if !found {
            return Err(JitHintError(format!("unknown JIT parameter: {name}")));
        }
    }
    Ok(())
}

// ── AsmInfo, JitDebugInfo, JitHookInterface ──
// rlib/jit.py:1063, 1077, 1113
//
// `AsmInfo` is implemented in `majit-backend/src/lib.rs`. `JitDebugInfo` and
// `JitHookInterface` are debug-introspection interfaces that are not yet
// wired to user code in majit (hooks fire through compile/bridge paths).

// ── record_exact_class ──
// rlib/jit.py:1181-1189

/// Assure the JIT that value is an instance of cls. This is a precise
/// class check, like a guard_class.
///
/// See also `debug_assert_not_none(x)`, which asserts that x is not None
/// and also assures the JIT that it is the case.
///
/// rlib/jit.py:1181 — `assert type(value) is cls`
///
/// In Rust, exact class checks use vtable pointers.  The `cls` parameter
/// is a type discriminant (e.g., vtable address).  In interpreted mode,
/// this is a debug assertion; in JIT, it becomes RECORD_EXACT_CLASS.
#[inline(always)]
pub fn record_exact_class(value_vtable: usize, cls: usize) {
    debug_assert_eq!(
        value_vtable, cls,
        "record_exact_class: value vtable does not match expected class"
    );
}

// ── _jit_record_known_result ──
// rlib/jit.py:1220-1221

/// Marker, special-cased by jtransform during JIT compilation.
///
/// rlib/jit.py:1220 — `def _jit_record_known_result(result, function, *args): pass`
///
/// At runtime this is a no-op. During JIT tracing, the codewriter
/// converts calls to this function into a `RecordKnownResult` operation
/// that pairs an elidable call with its known result for constant folding.
///
/// Called from the `record_known_result!` macro.
#[inline(always)]
pub fn _jit_record_known_result<T>(_known_result: T, _func_result: T) {
    // No-op at runtime; the codewriter emits RecordKnownResult.
}

/// Implemented as a variadic macro in `majit-metainterp/src/lib.rs` since
/// Rust lacks `*args`; see `record_known_result!`.

// ── record_known_result ──
// rlib/jit.py:1224-1239
//
// RPython: `record_known_result(result, func, *args)`
// Rust doesn't have *args; use the `record_known_result!` macro in lib.rs
// which accepts any arity via variadic macro pattern.

// ── record_exact_value ──
// rlib/jit.py:1260-1265

/// Assure the JIT that value is the same as const_value.
///
/// rlib/jit.py:1260 — `assert value == const_value; return const_value`
#[inline(always)]
pub fn record_exact_value<T: Copy + PartialEq + std::fmt::Debug>(value: T, const_value: T) -> T {
    debug_assert_eq!(
        value, const_value,
        "record_exact_value called with two different arguments"
    );
    const_value
}

// ── conditional_call ──
// rlib/jit.py:1297-1316
//
// RPython: `conditional_call(condition, function, *args)`
//   → rtyper `jit_conditional_call(cond, funcptr, arg1, arg2, ...)` llop
//   → jtransform `conditional_call_ir_v` JitCode bytecode
//
// Rust: `conditional_call!(condition, func_path, arg1, arg2, ...)`
//   → jitcode_lower intercepts macro → `__builder.conditional_call_ir_v_typed_args`
//   → BC_COND_CALL_VOID JitCode bytecode
//
// Use the `conditional_call!` macro in `#[jit_interp]` functions for the
// JIT path. The plain function below is the non-JIT runtime helper —
// rlib/jit.py:1301-1316 `conditional_call` falls back to `if condition:
// function(*args)` outside JIT.

/// Does the same as:
///
/// ```text
/// if condition:
///     function(*args)
/// ```
///
/// rlib/jit.py:1301-1316 — non-JIT runtime helper.
///
/// For JIT-compiled `#[jit_interp]` functions, use `conditional_call!` macro
/// which takes `(condition, func_path, args...)` matching RPython's
/// `(condition, function, *args)`.
#[inline(always)]
pub fn conditional_call<F: FnOnce()>(condition: bool, function: F) {
    if condition {
        function()
    }
}

// ── conditional_call_elidable ──
// rlib/jit.py:1318-1359
//
// Same as conditional_call: use `conditional_call_elidable!` macro for JIT.
// The closure-based function below is a non-JIT runtime helper only.

// ── JitCondFalsy ──
// rlib/jit.py:1350-1357 — isinstance(value, int) and value == 0, else not value.

/// Trait expressing "falsy for `conditional_call_elidable`".
///
/// rlib/jit.py:1350-1357:
/// - If `value` is int: falsy iff `value == 0`
/// - Else (pointer/list/object): falsy iff `not value`
pub trait JitCondFalsy {
    fn jit_cond_falsy(&self) -> bool;
}

macro_rules! impl_jit_cond_falsy_int {
    ($($t:ty),*) => {
        $(
            impl JitCondFalsy for $t {
                #[inline(always)]
                fn jit_cond_falsy(&self) -> bool { *self == 0 }
            }
        )*
    };
}

impl_jit_cond_falsy_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl<T: ?Sized> JitCondFalsy for *const T {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_null()
    }
}

impl<T: ?Sized> JitCondFalsy for *mut T {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_null()
    }
}

impl<T> JitCondFalsy for Option<T> {
    #[inline(always)]
    fn jit_cond_falsy(&self) -> bool {
        self.is_none()
    }
}

// ── conditional_call_elidable ──
// rlib/jit.py:1321-1359

/// Does the same as:
///
/// ```text
/// if value == <0 or None or NULL>:
///     value = function(*args)
/// return value
/// ```
///
/// For the JIT.  Allows one branch which doesn't create a bridge,
/// typically used for caching.  The value and the function's return
/// type must match and cannot be a float: they must be either regular
/// 'int', or something that turns into a pointer.
///
/// Even if the function is not marked @elidable, it is still treated
/// mostly like one.  The only difference is that (in heapcache.py)
/// we don't assume this function won't change anything observable.
/// This is useful for caches.
///
/// rlib/jit.py:1322 — non-JIT runtime helper.
///
/// For JIT-compiled `#[jit_interp]` functions, use `conditional_call_elidable!` macro
/// which takes `(value, func_path, args...)` matching RPython's `(value, function, *args)`.
#[inline(always)]
pub fn conditional_call_elidable<T: JitCondFalsy, F: FnOnce() -> T>(value: T, function: F) -> T {
    if value.jit_cond_falsy() {
        function()
    } else {
        value
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promote_int() {
        assert_eq!(promote(42), 42);
    }

    #[test]
    fn test_promote_ref_copy() {
        let x: usize = 0xdead;
        assert_eq!(promote(x), 0xdead);
    }

    #[test]
    fn test_we_are_jitted_default() {
        assert!(!we_are_jitted());
    }

    #[test]
    fn test_isconstant_default() {
        assert!(!isconstant(&42));
    }

    #[test]
    fn test_isvirtual_default() {
        let v = vec![1, 2, 3];
        assert!(!isvirtual(&v));
    }

    #[test]
    fn test_current_trace_length_default() {
        assert_eq!(current_trace_length(), -1);
    }

    #[test]
    fn test_virtual_ref_lifecycle() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        assert!(vref.virtual_());
        assert_eq!(*vref.force().unwrap(), 42);
        assert!(!vref.virtual_());
    }

    #[test]
    fn test_virtual_ref_finish_invalidates() {
        let val = 42i64;
        let mut vref = virtual_ref(&val);
        virtual_ref_finish(&mut vref, &val);
        assert!(vref.force().is_err());
    }

    #[test]
    fn test_parameters_default() {
        assert_eq!(PARAMETERS.threshold, 1039);
        assert_eq!(PARAMETERS.function_threshold, 1619);
        assert_eq!(PARAMETERS.trace_eagerness, 200);
        assert_eq!(PARAMETERS.trace_limit, 6000);
    }

    #[test]
    fn test_set_user_param() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "threshold=100,trace_limit=3000").unwrap();
        assert_eq!(p.threshold, 100);
        assert_eq!(p.trace_limit, 3000);
        assert_eq!(p.function_threshold, 1619);
    }

    #[test]
    fn test_set_user_param_off() {
        let mut p = PARAMETERS;
        set_user_param(&mut p, "off").unwrap();
        assert_eq!(p.threshold, u32::MAX);
        assert_eq!(p.function_threshold, u32::MAX);
    }

    #[test]
    fn test_set_user_param_default() {
        let mut p = PARAMETERS;
        p.threshold = 999;
        set_user_param(&mut p, "default").unwrap();
        assert_eq!(p.threshold, 1039);
    }

    #[test]
    fn test_set_param_unknown_errors() {
        let mut p = PARAMETERS;
        assert!(set_param(&mut p, "nonexistent", 42).is_err());
    }

    #[test]
    fn test_set_user_param_unknown_errors() {
        let mut p = PARAMETERS;
        assert!(set_user_param(&mut p, "nonexistent=42").is_err());
    }

    #[test]
    fn test_conditional_call_elidable_int() {
        let result = conditional_call_elidable(42_i64, || 99_i64);
        assert_eq!(result, 42);
        let result = conditional_call_elidable(0_i64, || 99_i64);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_conditional_call_elidable_ptr() {
        let x: i64 = 42;
        let non_null: *const i64 = &x;
        let result = conditional_call_elidable(non_null, || std::ptr::null::<i64>());
        assert_eq!(result, non_null);

        let null: *const i64 = std::ptr::null();
        let fallback: i64 = 99;
        let result = conditional_call_elidable(null, || (&fallback) as *const i64);
        assert_eq!(result, (&fallback) as *const i64);
    }

    #[test]
    fn test_conditional_call_elidable_option() {
        let some_val: Option<i64> = Some(42);
        let result = conditional_call_elidable(some_val, || Some(99_i64));
        assert_eq!(result, Some(42));
        let none_val: Option<i64> = None;
        let result = conditional_call_elidable(none_val, || Some(99_i64));
        assert_eq!(result, Some(99));
    }

    #[test]
    fn test_record_exact_value_passthrough() {
        assert_eq!(record_exact_value(42, 42), 42);
    }

    #[test]
    fn test_loop_unrolling_heuristic() {
        let lst: &[i32] = &[];
        assert!(loop_unrolling_heuristic(lst, 0, 2));
        let lst = &[1, 2];
        // rlib/jit.py:301 — requires isconstant(size); runtime false → returns false.
        assert!(!loop_unrolling_heuristic(lst, 2, 2));
    }

    #[test]
    fn test_enable_opts_all() {
        assert_eq!(PARAMETERS.enable_opts, EnableOpts::All);
        assert_eq!(
            ENABLE_ALL_OPTS,
            "intbounds:rewrite:virtualize:string:pure:earlyforce:heap:unroll"
        );
    }

    #[test]
    fn test_vref_none() {
        assert_eq!(VREF_NONE._x, std::ptr::null());
        assert_eq!(VREF_NONE._state, VRefState::NonForced);
    }

    #[test]
    fn test_trace_limit_too_high() {
        let e = TraceLimitTooHigh;
        assert_eq!(format!("{e}"), "TraceLimitTooHigh");
    }

    #[test]
    fn test_assert_green_failed() {
        let e = AssertGreenFailed;
        assert_eq!(format!("{e}"), "AssertGreenFailed");
    }
}
