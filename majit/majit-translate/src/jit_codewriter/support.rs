//! Codewriter helpers shared by `call.py`, `jtransform.py`, etc.
//!
//! Translated from `rpython/jit/codewriter/support.py`. The pieces that
//! pyre actually consumes live here as line-by-line ports; the bulk of
//! `support.py` (`LLtypeHelpers`, `OOtypeHelpers`, RPython annotator
//! glue, `get_send_oopspec`) is RPython-internal and has no Rust
//! counterpart.  `parse_oopspec` / `normalize_opargs` are ported as
//! pure helper functions ([`parse_oopspec`] / [`normalize_opargs`])
//! and consumed by [`decode_builtin_call`] when argnames are
//! registered via `CallControl::mark_oopspec_argnames`.
//!
//! [`INLINE_CALLS_TO`] mirrors `support.py:444-449 inline_calls_to`
//! verbatim — the four entries upstream's `find_all_graphs` BFS seeds
//! so the inliner can always reach `int_abs` / `int_floordiv` /
//! `int_mod` / `ll_math.ll_math_sqrt`.
//!
//! [`builtin_func_for_spec`] is a line-by-line port of
//! `support.py:767-808`. The RPython algorithm — cache lookup, list-or-dict
//! discriminator, helper resolution through [`setup_extra_builtin`],
//! `need_result_type` prepend, c_func construction, cache write — is
//! reproduced in the same step order. The annotator-only edges
//! (`MixLevelHelperAnnotator.constfunc`, `lltype_to_annotation`,
//! `annmodel.SomePBC`) collapse to structural no-ops: pyre's helpers are
//! concrete `extern "C"` function pointers registered through
//! [`crate::jit_codewriter::call::CallControl::register_function_fnaddr`]
//! and resolved by canonical `_ll_<n>_<name>` lookup, so there is no
//! per-call graph materialisation. The returned [`BuiltinFuncSpec`]
//! bundles `(fnaddr, LIST_OR_DICT, need_result_type)` — pyre's analogue
//! of upstream's `(c_func, LIST_OR_DICT)` plus the side-channel
//! attribute. The cache lives behind [`CallControl::lookup_builtin_func_for_spec_cache`] /
//! [`CallControl::cache_builtin_func_for_spec`] (RPython
//! `rtyper._builtin_func_for_spec_cache`).
//!
//! [`setup_extra_builtin`] mirrors `support.py:683-693` verbatim — the
//! same `_ll_<n>_<name>` template, the same `build` prefix when `extra`
//! is supplied, the same KeyError fallback (pyre collapses
//! `globals()[name]` / `getattr(LLtypeHelpers, name).im_func` to a single
//! canonical fnaddr lookup, because pyre's helper namespace is flat).
//!
//! [`decode_builtin_call`] mirrors `support.py:755-765` line by line.
//! Upstream walks `op.args[0].value._obj` to read the helper's
//! `.oopspec` attribute (the spec string set by `@rlib.jit.oopspec`);
//! pyre has no `_obj` analogue on a Rust function pointer, so the
//! lookup goes through `CallControl::get_oopspec` — the registry
//! populated at translation time from `#[oopspec(...)]` attributes
//! and `lib.rs:707-741` static bindings.  The lookup is the structural
//! mirror of upstream's `_obj.<callable>.oopspec` access: same
//! information, different host vehicle.

use crate::jit_codewriter::call::CallControl;
use crate::model::{OpKind, SpaceOperation, ValueId};
use crate::parse::CallPath;

use majit_ir::value::Type;

/// `support.py:444-449 inline_calls_to`.
///
/// `(oopspec_name, ll_args, ll_res)` triples whose graphs the BFS
/// seeds so the optimizer can always look inside.  Mirrors upstream
/// line by line.
pub static INLINE_CALLS_TO: &[(&str, &[Type], Type)] = &[
    ("int_abs", &[Type::Int], Type::Int),
    ("int_floordiv", &[Type::Int, Type::Int], Type::Int),
    ("int_mod", &[Type::Int, Type::Int], Type::Int),
    ("ll_math.ll_math_sqrt", &[Type::Float], Type::Float),
];

/// `support.py:755-765 decode_builtin_call(op)`.
///
/// Line-by-line port.  Upstream:
///
/// ```python
/// def decode_builtin_call(op):
///     if op.opname == 'direct_call':
///         fnobj = op.args[0].value._obj
///         opargs = op.args[1:]
///         return get_call_oopspec_opargs(fnobj, opargs)
///     elif op.opname == 'gc_identityhash':
///         return get_identityhash_oopspec(op)
///     elif op.opname == 'gc_id':
///         return get_gcid_oopspec(op)
///     else:
///         raise ValueError(op.opname)
/// ```
///
/// Returns `(oopspec_name, opargs)`, mirroring the upstream 2-tuple.
///
/// `fnobj = op.args[0].value._obj` → `get_call_oopspec_opargs`:
/// upstream pulls the spec string off the callee function pointer's
/// metadata (`ll_func.oopspec`).  Pyre's `OpKind::Call` does not carry
/// the funcptr as args[0]; the equivalent metadata lives on
/// [`CallControl`] under [`CallControl::mark_oopspec`] (the registry
/// `lib.rs:599` / `:707-741` populates from `#[oopspec(...)]`
/// attributes and static jit.* bindings).  The lookup goes through
/// [`CallControl::get_oopspec`] — same information as
/// `op.args[0].value._obj.oopspec`, different host vehicle.  The spec
/// string is split at `(` to extract the bare name, matching
/// upstream's `parse_oopspec` first step (`support.py:707
/// operation_name, args = ll_func.oopspec.split('(', 1)`).
///
/// `opargs = op.args[1:]`: upstream drops the funcptr at args[0] and
/// returns the trailing positional args, then `normalize_opargs`
/// permutes / injects constants according to the placeholder pattern
/// in the spec string.  Pyre's `OpKind::Call::args` already excludes
/// the funcptr (the target is carried in `target: CallTarget`).
///
/// Strict-parity port: when `CallControl::get_oopspec_argnames(target)`
/// returns `Some(argnames)`, invoke the full [`parse_oopspec`] +
/// [`normalize_opargs`] line-by-line port.  When `None`, fall back to
/// bare-name extraction + positional forwarding — pyre's `lib.rs:707-741`
/// bindings have no `(...)` pattern, so no argname lookup is needed.
///
/// Return type is `Vec<NormalizedArg>` (not `Vec<ValueId>`): a slot
/// may be a passthrough `Pass(ValueId)` or a `ConstInt(i64)` literal
/// that the caller must materialise as an `OpKind::ConstInt` op.
/// `jtransform.rs::handle_builtin_call` allocates the ValueIds for
/// constant slots at the residual-call site.
///
/// `gc_identityhash` / `gc_id`: pyre has no corresponding OpKind
/// variant.  These ops would land alongside a port of
/// `rpython/jit/codewriter/jtransform.py rewrite_op_gc_identityhash`
/// / `rewrite_op_gc_id`; until the variants exist, the catch-all
/// `_` arm panics with the upstream `ValueError(op.opname)` message
/// shape so a future emitter that adds the variant trips here and
/// the corresponding arm gets ported in lockstep.
///
/// **Panics** when the call has no registered oopspec — upstream's
/// `parse_oopspec` raises `AttributeError` at `ll_func.oopspec` when
/// the wrapper carries no `oopspec` attribute.  Callers must only
/// invoke `decode_builtin_call` after the call has been classified
/// as a builtin (mirroring `jtransform.py:484` gating through
/// `handle_builtin_call`), at which point the oopspec attribute is
/// guaranteed by the `@oopspec` decorator that promoted the helper
/// to a builtin in the first place.
pub fn decode_builtin_call(
    op: &SpaceOperation,
    call_control: &CallControl,
) -> (String, Vec<NormalizedArg>) {
    match &op.kind {
        // `support.py:756-759`: op.opname == 'direct_call' → resolve via fnobj
        OpKind::Call { target, args, .. } => {
            // `support.py:757 fnobj = op.args[0].value._obj` →
            // `:759 get_call_oopspec_opargs(fnobj, opargs)` →
            // `:707 operation_name, args = ll_func.oopspec.split('(', 1)`.
            // Pyre's analogue: the spec string is registered on the
            // target via `mark_oopspec` (rlib/jit.py:250 `@oopspec(spec)`
            // semantics); the bare name is the part before `(`.
            let spec = call_control.get_oopspec(target).unwrap_or_else(|| {
                panic!(
                    "decode_builtin_call: target {target:?} carries no \
                     oopspec registration.  Upstream `support.py:707 \
                     operation_name, args = ll_func.oopspec.split('(', 1)` \
                     raises `AttributeError` on the same condition.  \
                     Callers must classify the call as a builtin via \
                     `CallControl::call_kind` before invoking \
                     `decode_builtin_call`, mirroring \
                     `jtransform.py:484 handle_builtin_call`'s upstream gating.",
                )
            });
            // `support.py:707 operation_name, args = ll_func.oopspec.split('(', 1)`
            // → `:726 get_call_oopspec_opargs` → `:727
            // oopspec, argtuple = parse_oopspec(fnobj)`.
            //
            // Strict-parity port: when the target's argnames are
            // registered (`CallControl::mark_oopspec_argnames`),
            // invoke the full [`parse_oopspec`] + [`normalize_opargs`]
            // pair.  When no argnames are registered (the dominant
            // case today — `lib.rs:707-741` bindings are bare dotted
            // names with no `(...)` pattern, so argname info is not
            // needed), fall back to the bare-name split and forward
            // `args` positionally (wrapped in `NormalizedArg::Pass`).
            match call_control.get_oopspec_argnames(target) {
                Some(argnames) => {
                    let argname_refs: Vec<&str> = argnames.iter().map(String::as_str).collect();
                    let (oopspec_name, argtuple) = parse_oopspec(spec, &argname_refs);
                    let normalized = normalize_opargs(&argtuple, args);
                    (oopspec_name, normalized)
                }
                None => {
                    // `support.py:758 opargs = op.args[1:]`: pyre's
                    // `OpKind::Call::args` already excludes the funcptr
                    // so the positional args flow through directly,
                    // wrapped as `Pass(vid)` for the uniform return shape.
                    let oopspec_name = spec.split('(').next().unwrap_or(spec).trim().to_string();
                    let opargs: Vec<NormalizedArg> =
                        args.iter().map(|vid| NormalizedArg::Pass(*vid)).collect();
                    (oopspec_name, opargs)
                }
            }
        }
        // `support.py:760-763`: gc_identityhash / gc_id branches.
        //
        // Pyre has no OpKind::GcIdentityHash / OpKind::GcId yet —
        // those would land alongside a port of
        // `rpython/jit/codewriter/jtransform.py rewrite_op_gc_identityhash`
        // / `rewrite_op_gc_id`.  When the variants land, replace the
        // catch-all panic with explicit arms that delegate to ported
        // `get_identityhash_oopspec` / `get_gcid_oopspec` helpers.
        //
        // `support.py:765`: raise ValueError(op.opname)
        _ => panic!(
            "decode_builtin_call: ValueError(op.opname) — upstream \
             `support.py:765 raise ValueError(op.opname)`.  Pyre OpKind \
             variant {:?} has no decode arm; if this is a newly emitted \
             op the corresponding `rewrite_op_<name>` must be ported and \
             an arm added here.",
            op.kind,
        ),
    }
}

/// `support.py:697-699 class Index` — placeholder for a positional
/// arg.  `normalize_opargs` walks an `argtuple` and replaces each
/// `Index(n)` with `opargs[n]` (passthrough), while non-Index entries
/// become constant injections.  Pyre's pure-function port emits this
/// enum from [`parse_oopspec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizeSlot {
    /// `support.py:713 argname2index[argname] = Index(n)` — positional
    /// passthrough.  `n` is the 0-based slot in the callee's
    /// argname list.
    Index(usize),
    /// `support.py:714 argtuple = eval(args, argname2index)` — integer
    /// literal injection.  Upstream's `eval` accepts any Python literal
    /// (strings, floats, nested tuples) but observed usage in
    /// `rint.py` / `rstr.py` confines slots to identifiers + integer
    /// literals; pyre's narrow port supports only the int case.
    /// Extending the parser for additional literal types is a follow-up
    /// (PRE-EXISTING-ADAPTATION).
    ConstInt(i64),
}

/// `support.py:717-724 normalize_opargs` return-shape.
///
/// Upstream returns a list where each entry is either an existing
/// `Variable` reference (`opargs[obj.n]`) or a fresh
/// `Constant(obj, lltype.typeOf(obj))`.  Pyre splits the two cases
/// into a discriminated enum so callers (`decode_builtin_call` and
/// downstream) can decide how to materialise constants — pyre's
/// constants are `OpKind::ConstInt` SSA ops whose creation needs
/// graph-builder access at the callsite.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NormalizedArg {
    /// `support.py:721 result.append(opargs[obj.n])` — pass an
    /// existing `ValueId` through verbatim.
    Pass(ValueId),
    /// `support.py:723 result.append(Constant(obj, lltype.typeOf(obj)))`
    /// — caller materialises a `ConstInt` op carrying this value.
    ConstInt(i64),
}

/// `support.py:701-715 parse_oopspec(fnobj)` — pure-function port.
///
/// ```python
/// def parse_oopspec(fnobj):
///     FUNCTYPE = lltype.typeOf(fnobj)
///     ll_func = fnobj._callable
///     nb_args = len(FUNCTYPE.ARGS)
///     argnames = ll_func.__code__.co_varnames[:nb_args]
///     # parse the oopspec and fill in the arguments
///     operation_name, args = ll_func.oopspec.split('(', 1)
///     assert args.endswith(')')
///     args = args[:-1] + ','     # trailing comma to force tuple syntax
///     if args.strip() == ',':
///         args = '()'
///     nb_args = len(argnames)
///     argname2index = dict(zip(argnames, [Index(n) for n in range(nb_args)]))
///     argtuple = eval(args, argname2index)
///     return operation_name, argtuple
/// ```
///
/// Pyre's port takes `spec` (the `ll_func.oopspec` string registered
/// via `mark_oopspec`) and `argnames` (the callee's positional parameter
/// names, registered via `mark_oopspec_argnames`) and returns
/// `(operation_name, argtuple)` matching the upstream 2-tuple.
///
/// Divergences vs upstream `eval(args, argname2index)`:
///   - String / float / nested-expression literals are not parsed.
///     Upstream's `eval` accepts them in principle but the observed
///     usage in `rint.py` / `rstr.py` is identifier + integer-literal
///     only.  An unknown slot panics with a citation pointing back to
///     the upstream `eval` and the convergence path.
///   - The spec-has-no-`(` case (a structural divergence from
///     upstream's `.split('(', 1)` which would raise ValueError)
///     returns an empty `argtuple` with the bare spec as
///     `operation_name`.  This handles pyre's `lib.rs:707-741`
///     bare-name registrations gracefully.
pub fn parse_oopspec(spec: &str, argnames: &[&str]) -> (String, Vec<NormalizeSlot>) {
    // `support.py:707 operation_name, args = ll_func.oopspec.split('(', 1)`
    let (operation_name, args_part) = match spec.split_once('(') {
        Some((name, rest)) => (name.trim().to_string(), rest),
        None => return (spec.trim().to_string(), Vec::new()),
    };
    // `support.py:708 assert args.endswith(')')`
    let inner = match args_part.strip_suffix(')') {
        Some(stripped) => stripped,
        None => panic!(
            "parse_oopspec: spec {spec:?} is missing the closing `)`.  \
             Upstream `support.py:708` asserts `args.endswith(')')`."
        ),
    };
    // `support.py:709 args = args[:-1] + ','` — trailing-comma tuple syntax.
    // `support.py:710-711 if args.strip() == ',': args = '()'`
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return (operation_name, Vec::new());
    }
    // `support.py:712 nb_args = len(argnames)`
    // `support.py:713 argname2index = dict(zip(argnames, [Index(n) for n in range(nb_args)]))`
    let argname2index: std::collections::HashMap<&str, usize> = argnames
        .iter()
        .enumerate()
        .map(|(n, name)| (*name, n))
        .collect();
    // `support.py:714 argtuple = eval(args, argname2index)`
    //
    // Pyre's narrow expression parser: comma-split, identifier or
    // integer literal per slot.  Empty slots (e.g. `"foo(x,)"` after
    // the trailing-comma append from upstream :709) are ignored — they
    // mirror the Python tuple syntax that lets `(x,)` produce a
    // 1-tuple.
    let argtuple: Vec<NormalizeSlot> = trimmed
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| match argname2index.get(s) {
            Some(n) => NormalizeSlot::Index(*n),
            None => match s.parse::<i64>() {
                Ok(n) => NormalizeSlot::ConstInt(n),
                Err(_) => panic!(
                    "parse_oopspec: slot {s:?} in spec {spec:?} is neither a known \
                     argname (one of {argnames:?}) nor an integer literal.  \
                     Upstream `support.py:714 eval(args, argname2index)` would \
                     also accept string / float / nested tuple expressions; \
                     pyre's narrow port handles only the two observed-upstream \
                     cases (Index + int).  Convergence path: extend this parser \
                     for the additional literal types, then add matching \
                     `NormalizeSlot` variants and `NormalizedArg` materialisation."
                ),
            },
        })
        .collect();
    (operation_name, argtuple)
}

/// `support.py:717-724 normalize_opargs(argtuple, opargs)` — pure-function port.
///
/// ```python
/// def normalize_opargs(argtuple, opargs):
///     result = []
///     for obj in argtuple:
///         if isinstance(obj, Index):
///             result.append(opargs[obj.n])
///         else:
///             result.append(Constant(obj, lltype.typeOf(obj)))
///     return result
/// ```
///
/// Pyre's [`NormalizedArg`] enum splits the two cases so the caller can
/// decide how to materialise the constant injection (pyre's constants
/// are `OpKind::ConstInt` SSA ops whose creation needs graph-builder
/// access at the residual-call callsite, not here).
pub fn normalize_opargs(argtuple: &[NormalizeSlot], opargs: &[ValueId]) -> Vec<NormalizedArg> {
    argtuple
        .iter()
        .map(|slot| match slot {
            // `support.py:720-721 if isinstance(obj, Index):
            //                        result.append(opargs[obj.n])`
            NormalizeSlot::Index(n) => NormalizedArg::Pass(opargs[*n]),
            // `support.py:722-723 else: result.append(Constant(obj, ...))`
            NormalizeSlot::ConstInt(v) => NormalizedArg::ConstInt(*v),
        })
        .collect()
}

/// `support.py:782-794 need_result_type` flag.
///
/// In upstream this is an attribute set on the wrapper function
/// (e.g. `LLtypeHelpers._ll_1_dict_keys.need_result_type = True` or
/// `= 'exact'`).  Pyre cannot read attributes off a function pointer,
/// so the flag is co-registered alongside the fnaddr when the host
/// has a helper that needs it.  The discriminator is:
///
/// * [`NeedResultType::No`] — `getattr(impl, 'need_result_type', False)` is False.
/// * [`NeedResultType::Approx`] — flag is truthy and not exactly `'exact'`
///   (upstream `:786 if impl.need_result_type != 'exact': ll_restype = ll_restype.TO`).
/// * [`NeedResultType::Exact`] — flag is exactly `'exact'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NeedResultType {
    No,
    Approx,
    Exact,
}

/// `support.py:770` cache key tuple
/// (`(oopspec_name, tuple(ll_args), ll_res, extrakey)`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BuiltinFuncSpecCacheKey {
    pub oopspec_name: String,
    pub ll_args: Vec<Type>,
    pub ll_res: Type,
    pub extrakey: Option<String>,
}

/// `support.py:767-808 builtin_func_for_spec` return value.
///
/// Upstream returns the 2-tuple `(c_func, LIST_OR_DICT)` where `c_func`
/// is a `Constant` whose `.value._obj` is a `_ptr` carrying both the
/// translated graph and the function pointer.  Pyre's analog surfaces:
///
/// * [`Self::impl_name`] — canonical `_ll_<n>_<name>` identifier produced
///   by [`setup_extra_builtin`].  Mirrors upstream's `impl.func_name`
///   (the wrapper looked up from `globals()` / `LLtypeHelpers`).
/// * [`Self::fnaddr`] — host-bound `extern "C"` function pointer
///   resolved through `setup_extra_builtin` (which panics if no
///   registration exists, mirroring `support.py:687-690` raise).
/// * [`Self::list_or_dict`] — the LIST_OR_DICT discriminator from
///   `support.py:776-779`.
/// * [`Self::need_result_type`] — `need_result_type` side-channel
///   attribute (`support.py:782-794`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinFuncSpec {
    pub impl_name: String,
    pub oopspec_name: String,
    pub ll_args: Vec<Type>,
    pub ll_res: Type,
    pub list_or_dict: Type,
    pub fnaddr: i64,
    pub need_result_type: NeedResultType,
}

/// `support.py:683-693 setup_extra_builtin(rtyper, oopspec_name, nb_args, extra=None)`.
///
/// Line-by-line port.  The upstream Python:
///
/// ```python
/// def setup_extra_builtin(rtyper, oopspec_name, nb_args, extra=None):
///     name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))
///     if extra is not None:
///         name = 'build' + name
///     try:
///         wrapper = globals()[name]
///     except KeyError:
///         wrapper = getattr(LLtypeHelpers, name).im_func
///     if extra is not None:
///         wrapper = wrapper(*extra)
///     return wrapper
/// ```
///
/// Pyre returns the `(canonical_name, fnaddr, need_result_type)` triple
/// because the `wrapper` upstream concept (a Python function object
/// carrying both an address and a `.need_result_type` attribute) does
/// not exist in Rust — those two pieces of data have to be surfaced
/// explicitly.
///
/// Resolution: pyre helpers are registered through
/// `register_function_fnaddr(CallPath::from_segments([canonical_name]))`
/// before the codewriter pipeline starts; this function reads that
/// registry.  The KeyError / `getattr(LLtypeHelpers, name).im_func`
/// fallback upstream walks two namespaces (module globals + class
/// dict); pyre's namespace is flat so a single lookup suffices.
///
/// **Panics** when the host has not registered a fnaddr under the
/// canonical name.  This mirrors `support.py:687-690` which raises
/// `AttributeError` from `getattr(LLtypeHelpers, name).im_func` when
/// both `globals()[name]` and `LLtypeHelpers.<name>` lookups fail.
/// Upstream never observes this raise in practice because every
/// helper named in `support.py` exists; pyre's host equivalents
/// (`pyre-interpreter::jit_trace_fnaddrs`) MUST register the
/// canonical name for any helper the codewriter pipeline asks for.
/// The strict panic is what lets `setup_extra_builtin` act as a
/// helper-existence invariant; weaker forms (returning `None`)
/// would let unbound helpers flow silently into placeholder fnaddrs
/// downstream and mask real wiring gaps.
pub fn setup_extra_builtin(
    call_control: Option<&CallControl>,
    oopspec_name: &str,
    nb_args: usize,
    extra: Option<&[String]>,
    extrakey: Option<&str>,
) -> (String, i64, NeedResultType) {
    // `support.py:684`: name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))
    let mut name = format!("_ll_{}_{}", nb_args, oopspec_name.replace('.', "_"));
    // `support.py:685-686`: if extra is not None: name = 'build' + name
    if extra.is_some() {
        name = format!("build{}", name);
    }
    // `support.py:687-690`:
    //   try:
    //       wrapper = globals()[name]
    //   except KeyError:
    //       wrapper = getattr(LLtypeHelpers, name).im_func
    //
    // Pyre's flat namespace lookup against the host-registered fnaddr
    // table.  An unregistered canonical name is a wiring bug —
    // upstream would raise `AttributeError` at the inner `getattr`
    // (the `im_func` attribute access on a missing class attribute);
    // pyre panics so the same failure surfaces loudly at codewriter
    // setup time instead of corrupting silently.
    let cc = call_control.expect(
        "setup_extra_builtin: call_control is required for a strict lookup; \
         upstream `support.py:687-690` resolves the wrapper through the rtyper-bound \
         `LLtypeHelpers` namespace, which pyre exposes via `CallControl::lookup_function_fnaddr`.",
    );
    // `support.py:691-692`: if extra is not None: wrapper = wrapper(*extra)
    //
    // Upstream calls the resolved wrapper as a Python factory to
    // produce the specialized helper.  Pyre cannot run a Rust
    // function pointer as a factory (no runtime codegen, no
    // annotator to drive specialisation); hosts pre-register the
    // specialised fnaddr per `(canonical_name, extrakey)` pair via
    // [`CallControl::register_builtin_factory`], and the lookup
    // happens BEFORE the generic `lookup_function_fnaddr` fallback
    // so the specialised binding wins.  When `extra.is_some()` and
    // no factory specialisation is registered, the fallback still
    // tries the build-prefixed canonical name (matching upstream's
    // KeyError-then-getattr two-step structure), letting a host that
    // pre-builds a single specialisation register under the same
    // canonical key without paying for the factory-registry layer.
    let factory_hit = match (extra, extrakey) {
        (Some(_), Some(k)) => cc.lookup_builtin_factory(&name, k),
        // `support.py:769 assert (extra is None) == (extrakey is None)`
        // is checked at the `builtin_func_for_spec` boundary; both
        // None here means the upstream `wrapper(*extra)` step is
        // skipped, and both Some without a registered factory falls
        // through to the canonical-name lookup below.
        _ => None,
    };
    let fnaddr = factory_hit.unwrap_or_else(|| {
        let canonical = CallPath::from_segments([name.as_str()]);
        cc.lookup_function_fnaddr(&canonical).unwrap_or_else(|| {
            panic!(
                "setup_extra_builtin: canonical helper `{name}` is not registered. \
                 Upstream `support.py:687-690` raises `AttributeError` at \
                 `getattr(LLtypeHelpers, name).im_func` when both `globals()[name]` \
                 and `LLtypeHelpers.<name>` lookups fail.  Pyre's host \
                 (`pyre-interpreter::jit_trace_fnaddrs` or equivalent) must register \
                 this helper under that exact canonical name before the codewriter \
                 pipeline invokes `builtin_func_for_spec({oopspec_name:?}, ...)`.",
            )
        })
    });
    // `need_result_type`: upstream reads `getattr(impl, 'need_result_type', False)`.
    // Pyre cannot reach into a function pointer; the flag is co-registered
    // through `CallControl::lookup_need_result_type` against the same
    // canonical name.  Unregistered canonical names default to
    // `NeedResultType::No`, matching `getattr(..., 'need_result_type', False)`'s
    // missing-attribute fallback.
    let need_result_type = cc
        .lookup_need_result_type(&name)
        .unwrap_or(NeedResultType::No);
    // `support.py:693`: return wrapper
    (name, fnaddr, need_result_type)
}

/// `support.py:767-808 builtin_func_for_spec(rtyper, oopspec_name, ll_args, ll_res, extra=None, extrakey=None)`.
///
/// Line-by-line port.  Upstream Python:
///
/// ```python
/// def builtin_func_for_spec(rtyper, oopspec_name, ll_args, ll_res,
///                           extra=None, extrakey=None):
///     assert (extra is None) == (extrakey is None)
///     key = (oopspec_name, tuple(ll_args), ll_res, extrakey)
///     try:
///         return rtyper._builtin_func_for_spec_cache[key]
///     except (KeyError, AttributeError):
///         pass
///     args_s = [lltype_to_annotation(v) for v in ll_args]
///     if '.' not in oopspec_name:    # 'newxxx' operations
///         LIST_OR_DICT = ll_res
///     else:
///         LIST_OR_DICT = ll_args[0]
///     s_result = lltype_to_annotation(ll_res)
///     impl = setup_extra_builtin(rtyper, oopspec_name, len(args_s), extra)
///     if getattr(impl, 'need_result_type', False):
///         ...                       # prepend SomePBC([desc]) to args_s
///     if hasattr(rtyper, 'annotator'):  # regular case
///         mixlevelann = MixLevelHelperAnnotator(rtyper)
///         c_func = mixlevelann.constfunc(impl, args_s, s_result)
///         mixlevelann.finish()
///     else:
///         # for testing only
///         c_func = Constant(oopspec_name,
///                           lltype.Ptr(lltype.FuncType(ll_args, ll_res)))
///     if not hasattr(rtyper, '_builtin_func_for_spec_cache'):
///         rtyper._builtin_func_for_spec_cache = {}
///     rtyper._builtin_func_for_spec_cache[key] = (c_func, LIST_OR_DICT)
///     return c_func, LIST_OR_DICT
/// ```
///
/// Pyre's `call_control` is the `rtyper` analog: it owns the
/// persistent fnaddr registry and the
/// `_builtin_func_for_spec_cache`.
///
/// **Caveat re: `call_control=None`**: upstream's testing-only
/// branch (`support.py:800-803`) does not need an annotator — it
/// constructs a placeholder `Constant(oopspec_name,
/// Ptr(FuncType(...)))` whose only consumer is the test that
/// inspects the spec name + arg types.  Pyre's structural divergence:
/// `setup_extra_builtin` always needs `call_control` for the strict
/// fnaddr lookup (`support.py:687-690 globals()[name]` /
/// `LLtypeHelpers.<name>.im_func` maps to
/// `CallControl::lookup_function_fnaddr` in pyre, and no Rust analog
/// of the symbolic Python identifier exists without a registry).
/// `setup_extra_builtin(None, ...)` therefore **panics** at the
/// `expect()` site, NOT returning a stub — current pyre callers
/// always supply a real `&CallControl`.  When a testing-only path
/// appears, port the upstream stub by returning `(canonical_name,
/// symbolic_fnaddr_for_path(...), NeedResultType::No)` from
/// `setup_extra_builtin` and having `builtin_func_for_spec` skip the
/// cache write (already handled).
pub fn builtin_func_for_spec(
    call_control: Option<&CallControl>,
    oopspec_name: &str,
    ll_args: &[Type],
    ll_res: Type,
    extra: Option<&[String]>,
    extrakey: Option<&str>,
) -> BuiltinFuncSpec {
    // `support.py:769`: assert (extra is None) == (extrakey is None)
    debug_assert_eq!(
        extra.is_none(),
        extrakey.is_none(),
        "support.py:769 — extra and extrakey must be supplied together",
    );

    // `support.py:770`: key = (oopspec_name, tuple(ll_args), ll_res, extrakey)
    let key = BuiltinFuncSpecCacheKey {
        oopspec_name: oopspec_name.to_string(),
        ll_args: ll_args.to_vec(),
        ll_res,
        extrakey: extrakey.map(String::from),
    };

    // `support.py:771-774`: try: return rtyper._builtin_func_for_spec_cache[key]
    //                       except (KeyError, AttributeError): pass
    if let Some(cc) = call_control {
        if let Some(cached) = cc.lookup_builtin_func_for_spec_cache(&key) {
            return cached;
        }
    }

    // `support.py:775`: args_s = [lltype_to_annotation(v) for v in ll_args]
    //
    // Pyre has no annotator; `ll_args` are carried verbatim through
    // the returned spec.  Upstream's `args_s` is consumed by
    // `mixlevelann.constfunc(impl, args_s, s_result)`, which has no
    // Rust counterpart.

    // `support.py:776-779`: LIST_OR_DICT discriminator.
    let list_or_dict = if !oopspec_name.contains('.') {
        // 'newxxx' operations
        ll_res
    } else {
        // dotted name (list / dict family, ll_math.*)
        debug_assert!(
            !ll_args.is_empty(),
            "support.py:779 — dotted oopspec name `{oopspec_name}` requires ll_args[0]",
        );
        ll_args[0]
    };

    // `support.py:780`: s_result = lltype_to_annotation(ll_res)
    // (no annotator surface in pyre; `ll_res` flows verbatim into the
    // returned spec)

    // `support.py:781`: impl = setup_extra_builtin(rtyper, oopspec_name, len(args_s), extra)
    //
    // `extrakey` is threaded as well so the factory-registry lookup
    // (`wrapper(*extra)` analog) can pick the right specialised
    // fnaddr — upstream's wrapper is invoked with `extra` itself, and
    // pyre's analog uses the host-supplied `extrakey` as the
    // matching cache discriminator.
    let (impl_name, fnaddr, need_result_type) =
        setup_extra_builtin(call_control, oopspec_name, ll_args.len(), extra, extrakey);

    // `support.py:782-794`: if getattr(impl, 'need_result_type', False):
    //                           if hasattr(rtyper, 'annotator'):
    //                               bk = rtyper.annotator.bookkeeper
    //                               ll_restype = ll_res
    //                               if impl.need_result_type != 'exact':
    //                                   ll_restype = ll_restype.TO
    //                               desc = bk.getdesc(ll_restype)
    //                           else:
    //                               class TestingDesc(object): ...
    //                               desc = TestingDesc()
    //                           args_s.insert(0, annmodel.SomePBC([desc]))
    //
    // The SomePBC prepended on `args_s` is annotator-only state: it
    // tells the annotator that the wrapper's first argument is a
    // pointer-to-type constant.  Pyre carries the same information on
    // the returned spec via `need_result_type`; downstream residual-call
    // emitters that handle `Approx` / `Exact` are responsible for
    // prepending the type-tag arg when materialising the call.  No
    // INLINE_CALLS_TO entry sets the flag, so this arm is structural
    // until the dict-iter helpers land.
    let _ = need_result_type;

    // `support.py:796-803`: c_func construction.
    //   if hasattr(rtyper, 'annotator'):   # regular case
    //       mixlevelann = MixLevelHelperAnnotator(rtyper)
    //       c_func = mixlevelann.constfunc(impl, args_s, s_result)
    //       mixlevelann.finish()
    //   else:                              # for testing only
    //       c_func = Constant(oopspec_name,
    //                         lltype.Ptr(lltype.FuncType(ll_args, ll_res)))
    //
    // Pyre always lands on the regular case: `setup_extra_builtin`
    // above resolved the host-bound `fnaddr`, which mirrors what
    // upstream's `mixlevelann.constfunc` materialises (the function
    // pointer half of the resulting Constant).  The testing-only
    // upstream branch (`Constant(oopspec_name, lltype.Ptr(...))` with
    // no real graph behind it) corresponds to a deliberate
    // host-config decision pyre does not exercise — the strict
    // panic in `setup_extra_builtin` prevents reaching this point
    // with an unbound helper.  No separate graph materialisation
    // happens because pyre cannot run the RPython annotator over a
    // Rust impl; hosts that want the BFS to look inside the helper
    // additionally call `register_function_graph` against the same
    // canonical `CallPath`.
    let spec = BuiltinFuncSpec {
        impl_name,
        oopspec_name: oopspec_name.to_string(),
        ll_args: ll_args.to_vec(),
        ll_res,
        list_or_dict,
        fnaddr,
        need_result_type,
    };

    // `support.py:805-807`: if not hasattr(rtyper, '_builtin_func_for_spec_cache'):
    //                           rtyper._builtin_func_for_spec_cache = {}
    //                       rtyper._builtin_func_for_spec_cache[key] = (c_func, LIST_OR_DICT)
    //
    // Pyre's cache lives on `CallControl` and is unconditionally
    // initialised at construction time (no `hasattr` guard needed —
    // the field is always present).  Cache writes happen behind a
    // `RefCell` so the read-only `call_control: Option<&CallControl>`
    // signature stays parity with upstream's `rtyper` parameter shape.
    if let Some(cc) = call_control {
        cc.cache_builtin_func_for_spec(key, spec.clone());
    }

    // `support.py:809`: return c_func, LIST_OR_DICT
    spec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_calls_to_matches_upstream_four_entries() {
        // `support.py:444-449` parity — exactly the four upstream
        // entries, in upstream order.  A new entry here means a new
        // upstream line landed; sync it instead of letting the
        // tables drift.
        assert_eq!(INLINE_CALLS_TO.len(), 4);
        assert_eq!(INLINE_CALLS_TO[0].0, "int_abs");
        assert_eq!(INLINE_CALLS_TO[0].1, &[Type::Int]);
        assert_eq!(INLINE_CALLS_TO[0].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[1].0, "int_floordiv");
        assert_eq!(INLINE_CALLS_TO[1].1, &[Type::Int, Type::Int]);
        assert_eq!(INLINE_CALLS_TO[1].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[2].0, "int_mod");
        assert_eq!(INLINE_CALLS_TO[2].1, &[Type::Int, Type::Int]);
        assert_eq!(INLINE_CALLS_TO[2].2, Type::Int);
        assert_eq!(INLINE_CALLS_TO[3].0, "ll_math.ll_math_sqrt");
        assert_eq!(INLINE_CALLS_TO[3].1, &[Type::Float]);
        assert_eq!(INLINE_CALLS_TO[3].2, Type::Float);
    }

    #[test]
    fn setup_extra_builtin_renders_canonical_name() {
        // `support.py:684 name = '_ll_%d_%s' % (nb_args, oopspec_name.replace('.', '_'))`.
        // Register stub fnaddrs so the strict-panic resolution succeeds —
        // the test is purely about the name template, not the lookup.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 1);
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 2);
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_ll_math_ll_math_sqrt"]), 3);
        let (name, _, _) = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
        assert_eq!(name, "_ll_2_int_mod");
        let (name, _, _) = setup_extra_builtin(Some(&cc), "int_abs", 1, None, None);
        assert_eq!(name, "_ll_1_int_abs");
        let (name, _, _) = setup_extra_builtin(Some(&cc), "ll_math.ll_math_sqrt", 1, None, None);
        // `.` → `_` per `support.py:684 .replace('.', '_')`
        assert_eq!(name, "_ll_1_ll_math_ll_math_sqrt");
    }

    #[test]
    fn setup_extra_builtin_prepends_build_when_extra_supplied() {
        // `support.py:685-686 if extra is not None: name = 'build' + name`
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_0_dict_make"]), 1);
        let extra: Vec<String> = vec!["foo".to_string()];
        let (name, _, _) =
            setup_extra_builtin(Some(&cc), "dict_make", 0, Some(&extra), Some("foo"));
        assert_eq!(name, "build_ll_0_dict_make");
    }

    #[test]
    fn setup_extra_builtin_resolves_host_registered_fnaddr() {
        let mut cc = CallControl::new();
        // Host registers under the canonical `_ll_<n>_<name>` form per
        // `support.py:684` template.
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0xdead_beef);
        let (_, fnaddr, _) = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
        assert_eq!(fnaddr, 0xdead_beef);
    }

    #[test]
    #[should_panic(expected = "canonical helper `_ll_2_int_mod` is not registered")]
    fn setup_extra_builtin_panics_when_canonical_unregistered() {
        // `support.py:687-690`: when both `globals()[name]` and
        // `getattr(LLtypeHelpers, name).im_func` lookups miss, RPython
        // raises `AttributeError`.  Pyre mirrors the raise via panic so
        // missing wiring surfaces loudly at setup time.
        let cc = CallControl::new();
        let _ = setup_extra_builtin(Some(&cc), "int_mod", 2, None, None);
    }

    #[test]
    fn setup_extra_builtin_reads_need_result_type_registry() {
        // `support.py:782-794`: `getattr(impl, 'need_result_type', False)`.
        // Pyre's host registers the flag against the canonical name; the
        // setup_extra_builtin lookup must observe it.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_dict_keys"]), 0x44);
        cc.register_need_result_type("_ll_1_dict_keys", NeedResultType::Approx);
        let (_, _, ty) = setup_extra_builtin(Some(&cc), "dict_keys", 1, None, None);
        assert_eq!(ty, NeedResultType::Approx);
    }

    #[test]
    fn setup_extra_builtin_uses_factory_registry_when_extra_supplied() {
        // `support.py:691-692 wrapper = wrapper(*extra)`: pyre's
        // factory registry returns the specialised fnaddr that the
        // host pre-built for `(canonical_name, extrakey)`.  The
        // generic `build_ll_2_dict_iter` canonical-name fallback is
        // intentionally bound to a different address so the test
        // distinguishes which path won — the factory match must
        // shadow the fallback.
        const FALLBACK: i64 = 0xfa11_bac1;
        const SPECIAL: i64 = 0xcafe_5e11;
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_dict_iter"]), FALLBACK);
        cc.register_builtin_factory("build_ll_2_dict_iter", "Ptr(STR)", SPECIAL);
        let extra: Vec<String> = vec!["Ptr(STR)".to_string()];
        let (_, fnaddr, _) =
            setup_extra_builtin(Some(&cc), "dict_iter", 2, Some(&extra), Some("Ptr(STR)"));
        assert_eq!(fnaddr, SPECIAL);
    }

    #[test]
    fn setup_extra_builtin_factory_falls_back_to_canonical_when_unregistered() {
        // `support.py:691-692` after the wrapper itself is found via
        // `globals()[name]`: pyre falls back to the generic
        // canonical-name fnaddr when no `(canonical, extrakey)`
        // specialisation is registered, matching the pre-factory
        // workaround pattern.
        const FALLBACK: i64 = 0xfa11_bac1;
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_dict_iter"]), FALLBACK);
        let extra: Vec<String> = vec!["Ptr(STR)".to_string()];
        let (_, fnaddr, _) =
            setup_extra_builtin(Some(&cc), "dict_iter", 2, Some(&extra), Some("Ptr(STR)"));
        assert_eq!(fnaddr, FALLBACK);
    }

    #[test]
    fn builtin_func_for_spec_no_dot_uses_ll_res_as_list_or_dict() {
        // `support.py:776-777`: if '.' not in oopspec_name: LIST_OR_DICT = ll_res
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 0x11);
        let spec = builtin_func_for_spec(Some(&cc), "int_abs", &[Type::Int], Type::Int, None, None);
        assert_eq!(spec.oopspec_name, "int_abs");
        assert_eq!(spec.impl_name, "_ll_1_int_abs");
        assert_eq!(spec.ll_args, &[Type::Int]);
        assert_eq!(spec.ll_res, Type::Int);
        assert_eq!(spec.list_or_dict, Type::Int);
        assert_eq!(spec.fnaddr, 0x11);
        assert_eq!(spec.need_result_type, NeedResultType::No);
    }

    #[test]
    fn builtin_func_for_spec_dotted_name_uses_first_arg_as_list_or_dict() {
        // `support.py:778-779`: dotted name → LIST_OR_DICT = ll_args[0]
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(
            CallPath::from_segments(["_ll_1_ll_math_ll_math_sqrt"]),
            0x22,
        );
        let spec = builtin_func_for_spec(
            Some(&cc),
            "ll_math.ll_math_sqrt",
            &[Type::Float],
            Type::Float,
            None,
            None,
        );
        assert_eq!(spec.list_or_dict, Type::Float);
        assert_eq!(spec.impl_name, "_ll_1_ll_math_ll_math_sqrt");
    }

    #[test]
    fn builtin_func_for_spec_resolves_host_registered_canonical_fnaddr() {
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0xdead_beef);
        let spec = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(spec.fnaddr, 0xdead_beef);
    }

    #[test]
    fn builtin_func_for_spec_caches_repeated_calls() {
        // `support.py:771-774` + `:805-807`: the second call returns
        // the cached spec.  Use a controlled fnaddr that we can flip
        // after the first call to confirm the second call reads from
        // the cache rather than re-resolving.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0x1111);
        let first = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(first.fnaddr, 0x1111);
        // Mutate the underlying registry; cache should shield the
        // second call from observing the update.
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_2_int_mod"]), 0x2222);
        let second = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            None,
            None,
        );
        assert_eq!(second.fnaddr, 0x1111);
    }

    #[test]
    fn builtin_func_for_spec_cache_keyed_on_extrakey() {
        // `support.py:770` key tuple includes extrakey — distinct
        // extrakeys must produce distinct cache entries even when
        // every other field matches.
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["build_ll_2_int_mod"]), 0xaaaa);
        let extra_a: Vec<String> = vec!["a".to_string()];
        let extra_b: Vec<String> = vec!["b".to_string()];
        let spec_a = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            Some(&extra_a),
            Some("a"),
        );
        let spec_b = builtin_func_for_spec(
            Some(&cc),
            "int_mod",
            &[Type::Int, Type::Int],
            Type::Int,
            Some(&extra_b),
            Some("b"),
        );
        // Both produce the `build_ll_2_int_mod` canonical name (extra
        // is_some → 'build' prefix) and the same fnaddr, but the cache
        // tracks them under distinct keys for upstream-parity dedup.
        assert_eq!(spec_a.impl_name, "build_ll_2_int_mod");
        assert_eq!(spec_b.impl_name, "build_ll_2_int_mod");
        assert_eq!(spec_a.fnaddr, 0xaaaa);
        assert_eq!(spec_b.fnaddr, 0xaaaa);
    }

    #[test]
    #[should_panic(expected = "extra and extrakey must be supplied together")]
    fn builtin_func_for_spec_rejects_mismatched_extra_extrakey() {
        // `support.py:769 assert (extra is None) == (extrakey is None)`
        let mut cc = CallControl::new();
        cc.register_function_fnaddr(CallPath::from_segments(["_ll_1_int_abs"]), 0xa);
        let extra: Vec<String> = vec!["a".to_string()];
        let _ = builtin_func_for_spec(
            Some(&cc),
            "int_abs",
            &[Type::Int],
            Type::Int,
            Some(&extra),
            None,
        );
    }

    // ───────── `decode_builtin_call` tests ─────────

    fn make_call_op(target: crate::model::CallTarget, args: Vec<ValueId>) -> SpaceOperation {
        SpaceOperation {
            result: None,
            kind: OpKind::Call {
                target,
                args,
                result_ty: crate::model::ValueType::Int,
            },
        }
    }

    #[test]
    fn decode_builtin_call_returns_registered_oopspec_and_positional_args() {
        // `support.py:707 operation_name, args = ll_func.oopspec.split('(', 1)`:
        // bare-name spec entries (e.g. lib.rs:707-741 jit.* bindings)
        // resolve to the spec value itself, no `(` stripping needed.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["jit", "isconstant"]);
        cc.mark_oopspec(path.clone(), "jit.isconstant".to_string());
        let op = make_call_op(
            crate::model::CallTarget::function_path(["jit", "isconstant"]),
            vec![ValueId(7), ValueId(11)],
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "jit.isconstant");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(ValueId(7)),
                NormalizedArg::Pass(ValueId(11))
            ]
        );
    }

    #[test]
    fn decode_builtin_call_strips_arg_pattern_from_spec_name() {
        // `support.py:707 ll_func.oopspec.split('(', 1)`: when the spec
        // carries the placeholder pattern (e.g. `"int.py_mod(x, y)"`),
        // the operation name is the prefix before `(`.  Without
        // argname registration the positional flow forwards args as
        // `NormalizedArg::Pass`.
        let mut cc = CallControl::new();
        cc.mark_oopspec(
            CallPath::from_segments(["int", "py_mod"]),
            "int.py_mod(x, y)".to_string(),
        );
        let op = make_call_op(
            crate::model::CallTarget::function_path(["int", "py_mod"]),
            vec![ValueId(2), ValueId(3)],
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "int.py_mod");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(ValueId(2)),
                NormalizedArg::Pass(ValueId(3))
            ]
        );
    }

    #[test]
    #[should_panic(expected = "carries no oopspec registration")]
    fn decode_builtin_call_panics_when_target_has_no_oopspec_registration() {
        // `support.py:707 ll_func.oopspec.split(...)` raises
        // `AttributeError` when the wrapper has no `oopspec`.  Pyre
        // mirrors via panic so callers gating on `CallKind::Builtin`
        // (per `jtransform.py:484 handle_builtin_call`) catch wiring
        // gaps loudly.
        let cc = CallControl::new();
        let op = make_call_op(
            crate::model::CallTarget::function_path(["some", "unregistered"]),
            vec![],
        );
        let _ = decode_builtin_call(&op, &cc);
    }

    #[test]
    #[should_panic(expected = "ValueError(op.opname)")]
    fn decode_builtin_call_panics_on_non_call_opkind() {
        // `support.py:765 raise ValueError(op.opname)`: any opname
        // outside {direct_call, gc_identityhash, gc_id} raises.  Pyre
        // mirrors the raise via panic.  IndirectCall is included
        // because upstream does NOT have an `indirect_call` arm
        // (the previous pyre `_ => None` for IndirectCall was a
        // NEW-DEVIATION; this test pins the corrected behaviour).
        let cc = CallControl::new();
        let op = SpaceOperation {
            result: None,
            kind: OpKind::IndirectCall {
                funcptr: ValueId(0),
                args: vec![],
                graphs: None,
                result_ty: crate::model::ValueType::Int,
            },
        };
        let _ = decode_builtin_call(&op, &cc);
    }

    // ── `parse_oopspec` + `normalize_opargs` pure-function ports ────

    #[test]
    fn parse_oopspec_returns_bare_name_when_spec_has_no_paren() {
        // `support.py:707 ll_func.oopspec.split('(', 1)` raises
        // ValueError upstream when there's no `(`.  Pyre's port
        // gracefully handles the bare-name registrations at
        // `lib.rs:707-741` (`"jit.isconstant"` etc.) by returning an
        // empty argtuple.
        let (name, argtuple) = super::parse_oopspec("jit.isconstant", &["value"]);
        assert_eq!(name, "jit.isconstant");
        assert!(argtuple.is_empty());
    }

    #[test]
    fn parse_oopspec_returns_empty_argtuple_for_empty_paren_body() {
        // `support.py:710-711 if args.strip() == ',': args = '()'` —
        // empty parenthesised body yields an empty argtuple.
        let (name, argtuple) = super::parse_oopspec("foo()", &[]);
        assert_eq!(name, "foo");
        assert!(argtuple.is_empty());
    }

    #[test]
    fn parse_oopspec_resolves_positional_identifiers_to_index_slots() {
        // `support.py:713-714 argname2index = dict(zip(argnames, [Index(n) for n in
        // range(nb_args)])); argtuple = eval(args, argname2index)` — the
        // dominant case where the spec's identifiers match the callee's
        // parameter names in order.
        let (name, argtuple) = super::parse_oopspec("int.py_mod(x, y)", &["x", "y"]);
        assert_eq!(name, "int.py_mod");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::Index(0),
                super::NormalizeSlot::Index(1)
            ]
        );
    }

    #[test]
    fn parse_oopspec_permutes_args_when_identifier_order_differs() {
        // `support.py:713-714` — if the spec lists identifiers in a
        // different order than the callee declares them, the eval
        // result is a permuted Index sequence.
        let (name, argtuple) = super::parse_oopspec("foo(y, x)", &["x", "y"]);
        assert_eq!(name, "foo");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::Index(1),
                super::NormalizeSlot::Index(0)
            ]
        );
    }

    #[test]
    fn parse_oopspec_injects_integer_literal_as_const_int_slot() {
        // `support.py:714 argtuple = eval(args, argname2index)` —
        // literals that don't match an argname pass through as
        // `Constant(obj, lltype.typeOf(obj))`.  Pyre's narrow port
        // accepts integer literals.
        let (name, argtuple) = super::parse_oopspec("foobar(2, c, i)", &["c", "i"]);
        assert_eq!(name, "foobar");
        assert_eq!(
            argtuple,
            vec![
                super::NormalizeSlot::ConstInt(2),
                super::NormalizeSlot::Index(0),
                super::NormalizeSlot::Index(1),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "missing the closing `)`")]
    fn parse_oopspec_panics_on_unclosed_paren() {
        // `support.py:708 assert args.endswith(')')` — mirror the
        // upstream AssertionError with a Rust panic.
        super::parse_oopspec("foo(x", &["x"]);
    }

    #[test]
    #[should_panic(expected = "is neither a known argname")]
    fn parse_oopspec_panics_on_unknown_non_integer_slot() {
        // `support.py:714 eval(args, argname2index)` would accept
        // `"z"` only if it's bound in argname2index; otherwise eval
        // raises NameError.  Pyre's port panics with a more
        // informative message citing the eval gap.
        super::parse_oopspec("foo(z)", &["x", "y"]);
    }

    #[test]
    fn normalize_opargs_passes_through_index_slots() {
        // `support.py:720-721 if isinstance(obj, Index):
        //                        result.append(opargs[obj.n])` — identity
        // pass for positional placeholders.
        let argtuple = vec![
            super::NormalizeSlot::Index(0),
            super::NormalizeSlot::Index(1),
        ];
        let opargs = vec![ValueId(11), ValueId(22)];
        let normalized = super::normalize_opargs(&argtuple, &opargs);
        assert_eq!(
            normalized,
            vec![
                super::NormalizedArg::Pass(ValueId(11)),
                super::NormalizedArg::Pass(ValueId(22)),
            ]
        );
    }

    #[test]
    fn normalize_opargs_permutes_and_injects_constants() {
        // Combined: permutation (Index(1) before Index(0)) + literal
        // injection (`Constant(7, ...)` upstream → `ConstInt(7)` pyre).
        let argtuple = vec![
            super::NormalizeSlot::ConstInt(7),
            super::NormalizeSlot::Index(1),
            super::NormalizeSlot::Index(0),
        ];
        let opargs = vec![ValueId(100), ValueId(200)];
        let normalized = super::normalize_opargs(&argtuple, &opargs);
        assert_eq!(
            normalized,
            vec![
                super::NormalizedArg::ConstInt(7),
                super::NormalizedArg::Pass(ValueId(200)),
                super::NormalizedArg::Pass(ValueId(100)),
            ]
        );
    }

    #[test]
    fn decode_builtin_call_uses_normalize_opargs_when_argnames_registered() {
        // End-to-end: with `mark_oopspec_argnames`, the full-port
        // branch in `decode_builtin_call` fires, parsing `(y, x)`
        // and permuting the positional opargs accordingly.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["my_helper"]);
        cc.mark_oopspec(path.clone(), "myop(y, x)".to_string());
        cc.mark_oopspec_argnames(path, vec!["x".to_string(), "y".to_string()]);
        let op = make_call_op(
            crate::model::CallTarget::function_path(["my_helper"]),
            vec![ValueId(10), ValueId(20)],
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "myop");
        // Argname `y` → Index(1) → opargs[1] = ValueId(20)
        // Argname `x` → Index(0) → opargs[0] = ValueId(10)
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::Pass(ValueId(20)),
                NormalizedArg::Pass(ValueId(10))
            ]
        );
    }

    #[test]
    fn decode_builtin_call_emits_const_int_slot_for_integer_literal() {
        // Mirrors `test_support.py:15-38 test_decode_builtin_call_nomethod`:
        //   `myfoobar(i, marker, c)` with `oopspec = 'foobar(2, c, i)'`.
        //   `decode_builtin_call(op)` returns `("foobar", [Constant(2),
        //   vc, vi])`.  Pyre's `NormalizedArg::ConstInt(2)` is the
        //   analogue of upstream's `Constant(2, Signed)` — the caller
        //   (`jtransform.rs::handle_builtin_call`) materialises a
        //   `OpKind::ConstInt(2)` op at the residual-call site.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["foobar"]);
        cc.mark_oopspec(path.clone(), "foobar(2, c, i)".to_string());
        cc.mark_oopspec_argnames(
            path,
            vec!["i".to_string(), "marker".to_string(), "c".to_string()],
        );
        let op = make_call_op(
            crate::model::CallTarget::function_path(["foobar"]),
            vec![ValueId(11), ValueId(22), ValueId(33)],
        );
        let (name, opargs) = decode_builtin_call(&op, &cc);
        assert_eq!(name, "foobar");
        assert_eq!(
            opargs,
            vec![
                NormalizedArg::ConstInt(2),
                NormalizedArg::Pass(ValueId(33)),
                NormalizedArg::Pass(ValueId(11)),
            ]
        );
    }
}
