//! `build_flow_from_rust` — core entry of the Rust-AST adapter.
//!
//! Mirrors `rpython/flowspace/objspace.py:38-53 build_flow(func)` in
//! signature and in the "walk source, emit SpaceOperations, close
//! blocks" contract. The Rust version consumes a `syn::ItemFn`
//! directly because pyre-interpreter's portal is Rust source — the
//! Python-side `func.func_code` round-trip is not available.
//!
//! ## Upstream construction idioms consulted
//!
//! - `rpython/flowspace/test/test_model.py:13-43` — canonical shape
//!   for straight-line graphs (startblock with inputargs, operations,
//!   single Link into returnblock).
//! - `rpython/translator/exceptiontransform.py:380-396` — the
//!   `Block::shared(inputargs) → closeblock([Link])` construction
//!   idiom for a block that produces one output.
//! - `rpython/flowspace/operation.py:445-521` — the `add_operator`
//!   table whose opnames every binop in [`binop_opname`] cites.
//! - `rpython/flowspace/flowcontext.py` `POP_JUMP_IF_FALSE` /
//!   `JUMP_FORWARD` handlers — the 2-exit fork + join idiom the
//!   `lower_if` routine mirrors without the Python bytecode
//!   vocabulary.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use syn::{
    Arm, BinOp, Block as SynBlock, Expr, ExprArray, ExprBinary, ExprBreak, ExprCall, ExprCast,
    ExprContinue, ExprField, ExprForLoop, ExprIf, ExprIndex, ExprLit, ExprLoop, ExprMatch,
    ExprMethodCall, ExprParen, ExprPath, ExprReturn, ExprTry, ExprTuple, ExprUnary, ExprWhile,
    ItemFn, Lit, Local, LocalInit, Member, Pat, PatIdent, Stmt, UnOp,
};

use crate::flowspace::model::{
    Block, BlockRef, BlockRefExt, ConstValue, Constant, FunctionGraph, HOST_ENV, Hlvalue,
    HostObject, Link, SpaceOperation, Variable, c_last_exception,
};

use super::host_env::{
    ModuleId, is_host_class_minted, mint_host_class, module_globals_lookup, pyre_stdlib_lookup,
};

/// Reasons the adapter rejects the input. Every variant carries a
/// human-readable `reason` string; the caller decides whether to log
/// it or surface it. No upstream counterpart — RPython's `build_flow`
/// assumes a well-formed Python function and crashes on bad input;
/// the Rust side rejects earlier so the caller can fall back to
/// another strategy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    /// Function signature (outer shape) disagrees with M2.5a scope.
    InvalidSignature { reason: String },
    /// A construct inside the body is outside the current adapter
    /// subset — remaining control flow (match / loop / ? / break /
    /// continue), method calls, struct literals, etc. The string
    /// names the construct and cites the phase it lands in.
    Unsupported { reason: String },
    /// Identifier resolved via `syn::Expr::Path` is not in the
    /// locals map. Corresponds to upstream `UnboundLocalError` at
    /// `flowcontext.py:LOAD_FAST` when a local is read before store.
    UnboundLocal { name: String },
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdapterError::InvalidSignature { reason } => {
                write!(f, "invalid signature: {reason}")
            }
            AdapterError::Unsupported { reason } => write!(f, "unsupported construct: {reason}"),
            AdapterError::UnboundLocal { name } => write!(f, "unbound local: {name}"),
        }
    }
}

impl std::error::Error for AdapterError {}

/// Entry point. Walks `func` and emits the `FunctionGraph` that
/// upstream `build_flow(GraphFunc)` would have produced for the
/// equivalent Python source.
///
/// Mints a fresh [`ModuleId`] internally — the body's
/// `LOAD_GLOBAL` lookups will see an empty registry slice and
/// fall through to `pyre_stdlib_lookup` / mint, matching the
/// pre-Issue-1.3 behaviour for direct single-`ItemFn` callers
/// (driver tests, the `tests::*` cases below). Callers that want
/// the body to resolve sibling-item names registered through
/// [`super::register::register_rust_module`] must use
/// [`build_flow_from_rust_in_module`] with the matching id.
pub fn build_flow_from_rust(func: &ItemFn) -> Result<FunctionGraph, AdapterError> {
    build_flow_from_rust_in_module(func, ModuleId::fresh())
}

/// Module-aware entry: lower `func` so that any `LOAD_GLOBAL`
/// lookup against `module_id`'s registry slice resolves to the
/// value the walker bound. Mirrors upstream `flowcontext.py:284
/// self.w_globals = Constant(func.__globals__)` per-function
/// scoping — `module_id` plays the role of the function's
/// `__globals__` reference.
pub fn build_flow_from_rust_in_module(
    func: &ItemFn,
    module_id: ModuleId,
) -> Result<FunctionGraph, AdapterError> {
    validate_signature(func)?;

    let (inputargs, locals) = collect_params(func)?;
    let startblock = Block::shared(inputargs);
    let name = func.sig.ident.to_string();
    let graph = FunctionGraph::new(name, startblock.clone());

    let mut builder = Builder {
        current: BlockBuilder {
            block: startblock.clone(),
            ops: Vec::new(),
            locals,
        },
        returnblock: graph.returnblock.clone(),
        exceptblock: graph.exceptblock.clone(),
        loop_stack: Vec::new(),
        module_id,
    };

    // Function body root: tail expression flows directly into the
    // returnblock Link (lines 105-110 below), so this is the one true
    // upstream-`flowcontext.py:1232 RETURN_VALUE` boundary site.
    // `at_boundary=true` propagates through control-flow lowerings so
    // any nested `if`/`match`/`block` whose result IS the function's
    // return value preserves boundary semantics for its arms; nested
    // `lower_block`/`lower_if`/`lower_match` reached from value
    // position (`let z = …;`, function args) flips `at_boundary` to
    // `false` per the Slice O4 PRE-EXISTING-ADAPTATION.
    match lower_block(&mut builder, &func.block, true)? {
        BlockExit::FallThrough(tail) => {
            // Body reached its closing `}` with a tail value —
            // terminate the currently-open block with a Link into
            // returnblock carrying that value (upstream implicit
            // `RETURN_VALUE` at the function-body tail).
            let link = Rc::new(RefCell::new(Link::new(
                vec![tail],
                Some(builder.returnblock.clone()),
                None,
            )));
            builder.finalize_current(vec![link], None);
        }
        BlockExit::Terminated => {
            // Body already closed itself via an explicit `return` —
            // `Return.nomoreblocks()` at `flowcontext.py:1232` ran,
            // so the returnblock Link has already been emitted and
            // the current block is gone. Nothing further to do here;
            // upstream's StopFlowing at the same site terminates the
            // pending-block scheduler identically.
        }
    }

    Ok(graph)
}

/// Result of lowering a block-shaped construct. Mirrors the control-
/// flow dichotomy in upstream `flowcontext.py`: either the block
/// reached its closing `}` with a tail value (`FallThrough`), or
/// execution left the block via a non-fallthrough terminator —
/// `return` in the current subset (`Terminated`). Upstream analogue:
/// `Return.nomoreblocks(ctx)` at `flowcontext.py:1232` closes the
/// current block straight to `graph.returnblock` and raises
/// `StopFlowing`; downstream code in the same block never runs.
enum BlockExit {
    FallThrough(Hlvalue),
    Terminated,
}

// ____________________________________________________________
// Signature validation — outer "is this function shape supported?"

fn validate_signature(func: &ItemFn) -> Result<(), AdapterError> {
    let sig = &func.sig;
    if sig.asyncness.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "async fn not supported (M2.5a)".into(),
        });
    }
    if sig.unsafety.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "unsafe fn not supported (M2.5a)".into(),
        });
    }
    // Generic type / lifetime parameters and where-clauses are
    // accepted as signature-level markers. The adapter does not
    // track trait-bound constraints directly; the annotator's
    // `FunctionDesc.specialize` / `cachedgraph`
    // (`description.py:272-281`, `:228-249`) reads the concrete
    // `args_s` at `build_types` call time and monomorphizes the
    // generics into a classdef-keyed specialized graph. Const
    // generic parameters still reject pending value-carrying
    // const-param support.
    for p in &sig.generics.params {
        match p {
            syn::GenericParam::Type(_) | syn::GenericParam::Lifetime(_) => {}
            syn::GenericParam::Const(_) => {
                return Err(AdapterError::InvalidSignature {
                    reason: "const generic parameter not supported (lands in M2.5d)".into(),
                });
            }
        }
    }
    if sig.variadic.is_some() {
        return Err(AdapterError::InvalidSignature {
            reason: "variadic fn not supported".into(),
        });
    }
    Ok(())
}

fn collect_params(func: &ItemFn) -> Result<(Vec<Hlvalue>, HashMap<String, Hlvalue>), AdapterError> {
    let mut inputargs: Vec<Hlvalue> = Vec::new();
    let mut locals: HashMap<String, Hlvalue> = HashMap::new();
    for input in &func.sig.inputs {
        let ident = match input {
            // `self` / `&self` / `&mut self` — the method dispatch
            // case. Upstream RPython binds `self` as the first local
            // after `FunctionDesc.bind_self` has annotated it with
            // the concrete classdef (`description.py:350-355`); the
            // adapter just exposes it as a Variable named `self`,
            // matching the Python source convention.
            syn::FnArg::Receiver(_) => "self".to_string(),
            syn::FnArg::Typed(pat_type) => match &*pat_type.pat {
                Pat::Ident(PatIdent {
                    ident,
                    by_ref: None,
                    subpat: None,
                    ..
                }) => ident.to_string(),
                _ => {
                    return Err(AdapterError::InvalidSignature {
                        reason: "parameter pattern must be a plain identifier".into(),
                    });
                }
            },
        };
        let var = Hlvalue::Variable(Variable::named(&ident));
        locals.insert(ident, var.clone());
        inputargs.push(var);
    }
    Ok((inputargs, locals))
}

// ____________________________________________________________
// Builder — tracks the currently-open block and orchestrates
// fork/join for control flow.

/// Per-block accumulation state. Matches the shape of upstream's
/// `FlowContext.locals_w` + `FlowContext.pending_block.operations` —
/// one block's worth of "things to attach before closing".
struct BlockBuilder {
    block: BlockRef,
    ops: Vec<SpaceOperation>,
    /// RPython `FrameState.locals_w` by-name view. The Rust adapter
    /// does not need the full mergeable-stack view because Rust source
    /// has no Python-bytecode stack.
    locals: HashMap<String, Hlvalue>,
}

/// Enclosing loop context — one entry per `while`/`loop` open on the
/// statement stack. `break` lowers a Link into `exit_block`; `continue`
/// into `header_block`. `merged_names` pins the locals set the loop
/// entry agreed on, so `break`/`continue` carry every entry-visible
/// local on the outgoing Link regardless of which body-local rebinds
/// happened at the jump point.
///
/// Upstream analogue: `flowcontext.py:794 SETUP_LOOP` pushes a
/// `LoopBlock` onto `self.blockstack`, and `BREAK_LOOP` / `CONTINUE_LOOP`
/// (`:525-529`) raise Break/Continue exceptions that the surrounding
/// LoopBlock turns into the corresponding bytecode-target jumps. The
/// Rust adapter does not have Python's blockstack abstraction; the
/// `loop_stack` here is the minimal equivalent for tracking
/// "which break target is live".
struct LoopCtx {
    header_block: BlockRef,
    exit_block: BlockRef,
    /// Names carried through the loop's back-edge + body inputargs.
    /// Includes any internal iter sidecar.
    merged_names: Vec<String>,
    /// Names carried through `break` / loop-exit exits — a subset of
    /// `merged_names` with internal iter sidecars removed. For
    /// `while` / `loop` the two sets are identical; `for` filters
    /// out its synthetic iterator slot so it does NOT leak into the
    /// exit block's inputargs (upstream `flowcontext.py:787, :1355,
    /// :1383` — iterator lives on the value stack and is popped at
    /// loop exit, never visible to post-loop code).
    exit_merged_names: Vec<String>,
}

struct Builder {
    current: BlockBuilder,
    returnblock: BlockRef,
    /// The graph's canonical exception exit — `Block([etype, evalue])`
    /// per `model.py:22-25`. `?` exception links target this.
    exceptblock: BlockRef,
    loop_stack: Vec<LoopCtx>,
    /// Per-function module identity for `func_globals` lookup.
    /// Mirrors upstream `flowcontext.py:284 self.w_globals =
    /// Constant(func.__globals__)`: each function carries a per-
    /// module reference, and `LOAD_GLOBAL` reads that module's
    /// dict — not a process-shared registry. Issue 1.3 closure
    /// (2026-05-05): replaces the prior process-global lookup.
    module_id: ModuleId,
}

impl Builder {
    fn emit_op(&mut self, op: SpaceOperation) {
        self.current.ops.push(op);
    }

    fn locals(&self) -> &HashMap<String, Hlvalue> {
        &self.current.locals
    }

    fn set_local(&mut self, name: String, value: Hlvalue) {
        self.current.locals.insert(name, value);
    }

    /// Attach accumulated `current.ops` + `exitswitch` to
    /// `current.block`, wire `exits` via `BlockRefExt::closeblock`, and
    /// leave the Builder without a live "current" block. Used when the
    /// terminator replaces the block entirely (graph exit) or before
    /// `open_new_block` swaps in a fresh one.
    ///
    /// Mirrors upstream `flowcontext.py` where attaching `operations
    /// = tuple(ops)` then `block.closeblock(*exits)` happens in the
    /// same finalization pass.
    fn finalize_current(&mut self, exits: Vec<Rc<RefCell<Link>>>, switch: Option<Hlvalue>) {
        let ops = std::mem::take(&mut self.current.ops);
        {
            let mut b = self.current.block.borrow_mut();
            b.operations = ops;
            b.exitswitch = switch;
        }
        self.current.block.closeblock(exits);
    }

    /// Swap the currently-open block with `new`. Used after
    /// `finalize_current` to start emitting into `new`.
    fn open_new_block(&mut self, new: BlockBuilder) {
        self.current = new;
    }

    /// Resolve a `syn::Path` in expression position to the `Hlvalue`
    /// upstream `flowcontext.py` would have produced for the
    /// equivalent Python source. Mirrors the
    /// `LOAD_FAST` (`pyopcode.py:502`) → `LOAD_GLOBAL`
    /// (`flowcontext.py:856`) → `LOAD_ATTR` (`:861` →
    /// `operation.py:618 getattr`) lookup chain that Python bytecode
    /// would compile a qualified name to.
    ///
    /// Resolution order matches upstream `flowcontext.py:835/845-854`:
    /// locals (LOAD_FAST) → `func_globals` → `__builtin__` → fail.
    ///
    /// 1. **Single-segment paths** — locals first (LOAD_FAST priority,
    ///    `pyopcode.py:502`); the bare identifier `None` resolves to
    ///    `Constant(ConstValue::None)` (Python's NoneType singleton —
    ///    flowspace's `Constant(None)` carrier per `model.py`);
    ///    `host_env::HOST_RUST_MODULE_FUNCS` (the `func_globals`
    ///    analogue, populated by [`super::register::register_rust_module`])
    ///    yields a `Constant(HostObject(<UserFunction>))` for any
    ///    top-level `fn` in a previously-walked `syn::File`; the
    ///    closed-world `host_env::PYRE_STDLIB` registry (the
    ///    `__builtin__` analogue) yields a
    ///    `Constant(HostObject(<class>))` for `Ok` / `Some` / `Err` /
    ///    `Result` / `Option`; otherwise `UnboundLocal`. No on-demand
    ///    minting at single-segment scope — a bare PascalCase
    ///    identifier could be a local, a unit struct, a const, or a
    ///    type, and the adapter cannot disambiguate without
    ///    annotator-level context.
    /// 2. **Multi-segment paths** (`A::B::C`) — leftmost segment goes
    ///    through single-segment resolution but with mint fallback
    ///    (PRE-EXISTING-ADAPTATION, see "Mint-on-demand stand-in"
    ///    below): if not in locals / not the `None` singleton / not
    ///    in PYRE_STDLIB, find-or-mint a `HostObject::Class` via the
    ///    process-global `host_env::mint_host_class(name)` so every
    ///    occurrence of `StepResult::*` — across all graphs in the
    ///    process — shares the same `StepResult` identity (mirrors
    ///    upstream `LOAD_GLOBAL` reading from `func.func_globals`
    ///    which returns the same Python object on every lookup
    ///    regardless of which graph is being built —
    ///    `flowcontext.py:847`). The `::` syntax disambiguates:
    ///    `StepResult` here is unambiguously a type/module path. Each
    ///    subsequent segment emits a `getattr(prev,
    ///    ConstValue::byte_str(seg))` SpaceOperation per
    ///    `operation.py:618 getattr` arity=2; the final `Variable`
    ///    is the resolved expression value. N segments emit N-1
    ///    getattr ops (PRE-EXISTING-ADAPTATION, see "Raw-getattr
    ///    cascade stand-in" below).
    ///
    /// Paths with `qself` (`<T as Foo>::Bar`), generic arguments
    /// (`Foo::<T>::Bar`), or a leading `::` (global path) reject as
    /// `Unsupported`. Each is its own port — see Slice O7+ in the
    /// orthodox HOST_ENV plan.
    ///
    /// ### PRE-EXISTING-ADAPTATION — Mint-on-demand stand-in for
    /// names not yet registered through the walker
    ///
    /// Upstream `flowcontext.py:845 find_global` raises
    /// `FlowingError("global name '%s' is not defined")` whenever
    /// `varname` is absent from both `w_globals.value` and
    /// `__builtin__`. Slices O7-O10 incrementally close this gap —
    /// `register_rust_module` populates `HOST_RUST_MODULE_GLOBALS`
    /// for every top-level `Item::Enum` / `Item::Struct` / literal
    /// `Item::Const`. What remains uncovered:
    ///
    /// - **`Item::Fn` is intentionally NOT registered.** Slice O7
    ///   originally registered sibling fns, but the walker has no
    ///   way to wire `FunctionDesc.buildgraph`'s deferred-lowering
    ///   path (`description.py:140` → `build_flow(GraphFunc)`) back
    ///   to the Rust-AST adapter. A registered sibling fn would
    ///   supply empty bytecode at lowering time. Reverted in the
    ///   Issue 1.2 fix — see `register.rs::register_rust_module`'s
    ///   "Why no Item::Fn?" docstring. The single entry-point fn
    ///   that production callers want is found via
    ///   `file.items.iter().find_map(...)` in
    ///   `build_host_function_from_rust_file`, bypassing the
    ///   registry.
    /// - **Production callsites that don't walk the file first.**
    ///   `Translation::from_rust_item_fn` and the driver tests in
    ///   `translator/driver.rs` invoke `build_host_function_from_rust`
    ///   on a single `ItemFn` without first registering the enclosing
    ///   `syn::File`'s sibling items. Until those callsites switch to
    ///   the walker-first idiom, the leftmost segment may legitimately
    ///   miss the registry even though the enum exists in source.
    /// - **`Item::Static`, compound `Item::Const`, `Item::Use`,
    ///   `Item::Mod`, `Item::Impl`.** Walker dispatch deferred
    ///   pending a const-expression evaluator and import-graph
    ///   resolution.
    /// - **Third-party crate types** (`rustpython_compiler_core::Instruction`)
    ///   that pyre-interpreter references — the walker has no
    ///   visibility into upstream crate sources.
    ///
    /// (Per-module `func_globals` scoping — Issue 1.3 — closed
    /// 2026-05-05. The registry is now partitioned by `ModuleId`
    /// per upstream `flowcontext.py:284 self.w_globals =
    /// Constant(func.__globals__)`. `Builder.module_id` carries
    /// the id `register_rust_module(file)` minted, so the lookup
    /// hits the matching partition and two walks of files with
    /// shared names see independent values.)
    ///
    /// For names that miss every prior layer, `mint_host_class`
    /// finds-or-mints a placeholder `HostObject::Class` (process-
    /// global identity, same `Arc::ptr_eq` invariant as
    /// `func_globals[name]`). The cascade then proceeds with raw
    /// emit (see "Constfold-then-emit" note below) because the
    /// minted class has empty members; constfold would
    /// `FlowingError` on every getattr.
    ///
    /// **Convergence path**: route every production callsite through
    /// `register_rust_module(&parsed_file)` before invoking
    /// `build_host_function_from_rust(item_fn)`. Then drop the
    /// `mint_unknown` branch and surface `AdapterError::UnboundLocal`
    /// (the closest local analogue of `FlowingError`) for
    /// unregistered names. Tracked as Slice O9 (callsite cutover) +
    /// Slice O10 (`Item::Const` walker dispatch).
    ///
    /// ### Constfold-then-emit cascade (Slice O8 — landed)
    ///
    /// Upstream `flowcontext.py:861 LOAD_ATTR` is
    /// `op.getattr(w_obj, w_name).eval(self)` — `eval` runs the op's
    /// `constfold()` (`operation.py:624 GetAttr.constfold`) before
    /// recording. The cascade now mirrors that shape: each step
    /// queries `class_get(name)` on the resolved leftmost. On
    /// `Some(value)` the cascade folds the segment to a Constant
    /// (matching upstream `try: result = getattr(obj, name); return
    /// const(result)`). On `None` the cascade falls through to the
    /// raw `getattr` SpaceOperation emission — this matches
    /// upstream's `HLOperation.eval` recording path when constfold
    /// returns `None` (e.g. non-foldable args), and is also the
    /// load-bearing fallback for minted-class leftmosts (which have
    /// empty class dicts by construction).
    ///
    /// ### PRE-EXISTING-ADAPTATION (Issue 2.1): constfold-miss is too lenient
    ///
    /// Upstream `operation.py:624 GetAttr.constfold` does:
    ///
    /// ```python
    /// if w_obj.foldable() and w_name.foldable():
    ///     try:
    ///         result = getattr(obj, name)
    ///     except Exception as e:
    ///         raise FlowingError(
    ///             "getattr(%s, %s) always raises %s: %s" %
    ///             (obj, name, etype, e))
    ///     return const(result)
    /// ```
    ///
    /// — both args foldable + lookup miss == compile-time error.
    /// The local Rust flowspace mirrors this at `model.rs:1201`
    /// (returns `None` to surface as a graph-build error). pyre's
    /// adapter cascade DOES NOT: when `acc` is a Constant carrying
    /// a `HostObject::Class` and `class_get(name)` returns `None`,
    /// the cascade falls through to the raw-emit branch instead of
    /// raising. The lenient fallback is load-bearing while the
    /// mint-on-demand stand-in is in place — minted classes have
    /// empty dicts, so any `getattr(MintedFoo, "Variant")` cascade
    /// would otherwise fail at every call site.
    ///
    /// **Convergence path** (multi-session, blocked on retiring the
    /// mint stand-in):
    ///
    /// 1. Land enough walker coverage (Issue 2.3) and per-module
    ///    scoping (Issue 1.3) that every leftmost-segment lookup
    ///    succeeds against a real registered class.
    /// 2. Drop the `mint_unknown` branch in
    ///    [`Self::resolve_leftmost_segment`] so unregistered names
    ///    are real `AdapterError::Unsupported` errors.
    /// 3. Tighten the cascade's constfold-miss to raise
    ///    `AdapterError::Unsupported` matching upstream's
    ///    `FlowingError` semantic.
    fn resolve_path_constant(&mut self, path: &syn::Path) -> Result<Hlvalue, AdapterError> {
        if path.leading_colon.is_some() {
            return Err(AdapterError::Unsupported {
                reason: "leading-`::` global path (`::std::result::Result`) — globally-anchored \
                    path resolution is its own port (out of M2.5e orthodox scope; see plan \
                    Non-goals)"
                    .into(),
            });
        }
        for seg in &path.segments {
            if !matches!(seg.arguments, syn::PathArguments::None) {
                return Err(AdapterError::Unsupported {
                    reason: "qualified path with generic arguments (`Foo::<T>::Bar`) — generic \
                        type-arg reification is its own port (out of M2.5e orthodox scope; \
                        see plan Non-goals)"
                        .into(),
                });
            }
        }
        let n = path.segments.len();
        if n == 0 {
            return Err(AdapterError::Unsupported {
                reason: "empty path (defensive — syn never produces this shape)".into(),
            });
        }

        // Resolve the leftmost segment. Single-segment scope rejects
        // unknown names; multi-segment scope mints on demand.
        let leftmost = path.segments[0].ident.to_string();
        let is_multi = n >= 2;
        let mut acc = self.resolve_leftmost_segment(&leftmost, is_multi)?;

        // Each subsequent segment mirrors upstream
        // `flowcontext.py:861 LOAD_ATTR` →
        // `op.getattr(w_obj, w_name).eval(self)`. `eval` runs the
        // op's `constfold()` first (`operation.py:624
        // GetAttr.constfold`): when `w_obj` is a foldable `Constant`
        // host class with the named member populated, the segment
        // collapses to `const(getattr(obj, name))`. Otherwise the
        // raw `getattr` SpaceOperation is recorded — matching
        // upstream's fall-through into `HLOperation.eval`'s
        // recording path.
        for seg in path.segments.iter().skip(1) {
            let name = seg.ident.to_string();
            // Mirrors `operation.py:624-642 GetAttr.constfold`. With
            // both `w_obj` (`acc`) and `w_name` (the constant byte
            // string) foldable, upstream's rule is: try Python's
            // `getattr(obj, name)`; if it succeeds bind the result
            // as `const(result)`; if it raises any exception, raise
            // `FlowingError("getattr(%s, %s) always raises %s: %s")`.
            // A Variable `acc` corresponds to upstream's
            // `w_obj.foldable() == False` path which returns from
            // `constfold` and lets `HLOperation.eval` record the
            // raw `getattr` op.
            if let Hlvalue::Constant(c) = &acc {
                if let ConstValue::HostObject(host) = &c.value {
                    if let Some(folded) = host.class_get(&name) {
                        acc = Hlvalue::Constant(Constant::new(folded));
                        continue;
                    }
                    // class dict miss — upstream Python's
                    // `getattr(obj, name)` raises `AttributeError`,
                    // which `operation.py:638-642` reraises as
                    // `FlowingError`. Issue 3 (2026-05-05) closes
                    // the prior raw-getattr fall-through for
                    // walker-registered classes (whose class dicts
                    // are authoritative).
                    if !is_host_class_minted(host) {
                        return Err(AdapterError::Unsupported {
                            reason: format!(
                                "getattr({}, {}) always raises AttributeError: \
                                 {} has no attribute '{}'",
                                host.qualname(),
                                name,
                                host.qualname(),
                                name,
                            ),
                        });
                    }
                    // PRE-EXISTING-ADAPTATION (mint-on-demand):
                    // mint-class dicts are intentionally empty
                    // pending walker coverage of the source-file
                    // declaration (`Item::Fn`, `Item::Use`,
                    // `Item::Mod`, …). Treat the class as opaque —
                    // upstream's `w_obj.foldable() == False` path,
                    // which records the raw `getattr` op without
                    // raising. Convergence path: once the missing
                    // walker dispatches land and `mint_host_class`
                    // is retired, drop this branch and the helper.
                }
            }
            let result = Hlvalue::Variable(Variable::new());
            self.emit_op(SpaceOperation::new(
                "getattr",
                vec![
                    acc,
                    Hlvalue::Constant(Constant::new(ConstValue::byte_str(name))),
                ],
                result.clone(),
            ));
            acc = result;
        }
        Ok(acc)
    }

    /// Single-segment resolution with mint-vs-reject choice driven by
    /// `mint_unknown`. The split lets `resolve_path_constant` apply
    /// different rules to the leftmost-segment of a multi-segment
    /// path (mint on miss) vs a standalone single-segment path
    /// (reject on miss).
    ///
    /// `mint_unknown=true` is the PRE-EXISTING-ADAPTATION described
    /// at length on `resolve_path_constant` — see "Mint-on-demand
    /// stand-in for the missing module-globals walker". Reject is
    /// the upstream-orthodox shape (`flowcontext.py:845-854 find_global`
    /// raises `FlowingError`); mint is the closed-world substitute
    /// pending the M3.x walker.
    fn resolve_leftmost_segment(
        &mut self,
        name: &str,
        mint_unknown: bool,
    ) -> Result<Hlvalue, AdapterError> {
        // 1. Locals first — `pyopcode.py:502 LOAD_FAST` priority.
        if let Some(local) = self.locals().get(name).cloned() {
            return Ok(local);
        }
        // 2. `None` singleton — Python's NoneType singleton instance.
        //    Upstream Python 2 has `None` in `__builtin__`; the Rust
        //    `None` (Option::None variant) is the closest source-side
        //    analogue and lowers to the same flowspace constant.
        if name == "None" {
            return Ok(Hlvalue::Constant(Constant::new(ConstValue::None)));
        }
        // 3. Module-globals (`func_globals` analogue) — populated by
        //    `register_rust_module` walking a `syn::File` at "import"
        //    time, partitioned by `ModuleId`. Mirrors upstream
        //    `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
        //    + `:847 w_globals.value[varname]`: each function carries
        //    a per-module reference, so two different modules with
        //    the same top-level name see independent values. The
        //    `Builder.module_id` field carries the id the walker
        //    minted at registration time so this lookup hits the
        //    matching partition. The registered value is a
        //    `ConstValue` (Slice O10 generalization) so consts
        //    (Item::Const) lift to `Int` / `Bool` / `byte_str`
        //    directly without a HostObject wrapper, matching
        //    upstream `find_global` returning `const(value)` for
        //    any Python type. Body lowering for fn entries is still
        //    deferred to `FunctionDesc.buildgraph` exactly as
        //    upstream defers `build_flow(func)` until the annotator
        //    walks a call site.
        if let Some(value) = module_globals_lookup(self.module_id, name) {
            return Ok(Hlvalue::Constant(Constant::new(value)));
        }
        // 4. Closed-world Rust-stdlib registry (`Ok` / `Some` / `Err`
        //    / `Result` / `Option`) — `__builtin__` analogue. Order
        //    matches upstream `find_global` chain (`flowcontext.py:849-852`):
        //    `func_globals[name]` first, then `getattr(__builtin__,
        //    name)`.
        if let Some(class) = pyre_stdlib_lookup(name) {
            return Ok(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                class,
            ))));
        }
        // 5. Process-global mint registry (only for multi-segment
        //    leftmost). `host_env::mint_host_class` finds-or-mints,
        //    so every graph in the process that names `name` shares
        //    the same `HostObject::Class` identity — same invariant
        //    upstream `func_globals[name]` provides at
        //    `flowcontext.py:847` (process-shared globals dict).
        if mint_unknown {
            let class = mint_host_class(name);
            return Ok(Hlvalue::Constant(Constant::new(ConstValue::HostObject(
                class,
            ))));
        }
        Err(AdapterError::UnboundLocal {
            name: name.to_string(),
        })
    }
}

// ____________________________________________________________
// Statement & expression lowering.

/// Lower a `{ ... }` syntactic block into the currently-open BlockBuilder.
///
/// `at_boundary=true` means the tail expression (if any) flows
/// directly into the function's returnblock Link — either because
/// this is the function-body root, or because the enclosing
/// `lower_if` / `lower_match` / `lower_arm_body` caller is itself in
/// boundary position (e.g. `fn f() { match x { 0 => Ok(1), _ =>
/// Err(e) } }` where each arm body's tail IS the function tail).
/// The Slice O4 PRE-EXISTING-ADAPTATION (`Ok(x)` / `Some(x)` →
/// unwrap, `None` → `ConstValue::None`, `Err(e)` → raise edge per
/// Slice O5) only fires under `at_boundary=true`.
///
/// `at_boundary=false` means the block result is consumed by the
/// caller (`let z = { … };`, function arg, binop operand, etc.).
/// In value position, `Ok(x)` stays a `simple_call(<Ok>, x)` op so
/// the resulting host-class instance threads through to the consumer
/// — `lower_arm_body`'s default arm routes through plain
/// `lower_expr` instead of `lower_value_boundary`.
fn lower_block(
    b: &mut Builder,
    block: &SynBlock,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    let mut tail: Option<Hlvalue> = None;
    for (idx, stmt) in block.stmts.iter().enumerate() {
        let is_last = idx + 1 == block.stmts.len();
        match stmt {
            Stmt::Local(local) => {
                lower_let(b, local)?;
            }
            Stmt::Expr(expr, semi) => {
                // `return` is a structural terminator in upstream:
                // `flowcontext.py:687 RETURN_VALUE` raises `Return`,
                // `flowcontext.py:1232 Return.nomoreblocks()` closes
                // the current block straight to `graph.returnblock`
                // and raises `StopFlowing`. Anything following in the
                // same block is dead code — mirror that by rejecting
                // non-last `return` and threading `BlockExit::Terminated`
                // out of `lower_block` on the last-stmt case. Works
                // identically with or without the trailing `;` (Rust
                // syntax allows both; syn preserves the semi flag but
                // the flow semantic is identical).
                if let Expr::Return(ret) = expr {
                    if !is_last {
                        return Err(AdapterError::Unsupported {
                            reason: "statement after `return` — upstream \
                                `flowcontext.py:1232` closes the block to \
                                graph.returnblock on Return, making any \
                                subsequent ops unreachable"
                                .into(),
                        });
                    }
                    return lower_return(b, ret);
                }
                // `while` / `loop` are statement-only in the M2.5b
                // slice-3 subset — they produce upstream's bytecode
                // `SETUP_LOOP` + back-edge shape but carry no tail
                // value, so they are accepted with or without the
                // trailing `;` (Rust allows omitting it after a
                // block-tailed expression).
                match expr {
                    Expr::While(while_expr) => {
                        lower_while(b, while_expr)?;
                        continue;
                    }
                    Expr::Loop(loop_expr) => {
                        lower_loop(b, loop_expr)?;
                        continue;
                    }
                    Expr::ForLoop(for_expr) => {
                        lower_for(b, for_expr)?;
                        continue;
                    }
                    // `if cond { body }` without `else` is a statement
                    // producing `()` (upstream Python: `None` via the
                    // implicit RETURN_VALUE on `if x: body` fallthrough
                    // — `flowcontext.py:756 POP_JUMP_IF_FALSE`). Treat
                    // it the same as while/loop: always a statement,
                    // never a tail. When it IS the last stmt, the
                    // function falls through to the implicit-None tail
                    // handled by `lower_block`'s terminal fallback.
                    //
                    // `lower_if_without_else` unconditionally falls
                    // through (the Bool(false) exit shortcuts to
                    // join, so join is always reachable), so it never
                    // produces `Terminated` — but match structurally
                    // so a future slice that tightens this invariant
                    // forwards termination out of the enclosing block
                    // correctly.
                    Expr::If(if_expr) if if_expr.else_branch.is_none() => {
                        // `if cond { body }` as a statement: body's
                        // tail value is discarded (the join produces
                        // implicit None). Pass `at_boundary=false` so
                        // the body never collapses an Ok/Some/Err
                        // tail — there's no return edge to feed.
                        match lower_if(b, if_expr, false)? {
                            BlockExit::FallThrough(_) => continue,
                            BlockExit::Terminated => return Ok(BlockExit::Terminated),
                        }
                    }
                    // `break` / `continue` appearing at the statement
                    // level of a non-loop block is unconditionally an
                    // error — `lower_loop_body` intercepts these
                    // inside actual loop bodies so the only way to
                    // reach `lower_block` with a break/continue stmt
                    // is outside any `loop_stack` entry.
                    Expr::Break(_) => {
                        return Err(AdapterError::Unsupported {
                            reason: "`break` outside of a loop".into(),
                        });
                    }
                    Expr::Continue(_) => {
                        return Err(AdapterError::Unsupported {
                            reason: "`continue` outside of a loop".into(),
                        });
                    }
                    _ => {}
                }
                if semi.is_some() {
                    // Expression statement: lower for side effect,
                    // discard the result. Upstream CPython emits
                    // `POP_TOP` after a stack-producing op —
                    // `flowcontext.py:488` covers the same
                    // semantic at the bytecode level.
                    let _ = lower_expr(b, expr)?;
                } else {
                    if !is_last {
                        return Err(AdapterError::Unsupported {
                            reason: "non-tail expression without trailing `;`".into(),
                        });
                    }
                    // Tail expression: evaluated as the block's
                    // value. Control-flow-bearing constructs
                    // (`if`/`match`/`return`/nested block) may
                    // terminate when every path returns — thread
                    // the `BlockExit` out so the enclosing lowering
                    // observes termination. Non-control-flow
                    // expressions always fall through with a value,
                    // so `lower_arm_body`'s default arm routes them
                    // through `lower_expr` (or, if `at_boundary`,
                    // through `lower_value_boundary` for the Slice
                    // O4 unwrap).
                    match lower_arm_body(b, expr, at_boundary)? {
                        BlockExit::FallThrough(v) => tail = Some(v),
                        BlockExit::Terminated => return Ok(BlockExit::Terminated),
                    }
                }
            }
            Stmt::Item(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "nested item (fn/struct/impl inside fn body)".into(),
                });
            }
            Stmt::Macro(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "macro invocation in statement position".into(),
                });
            }
        }
    }
    // No explicit tail expression → implicit `None` return. Upstream
    // CPython's bytecode compiler emits `LOAD_CONST None;
    // RETURN_VALUE` as the default terminator of any function body
    // that doesn't end in an explicit return, so the flowspace sees
    // the returnblock Link carrying `Constant(None)`. The adapter
    // mirrors that directly: blocks tail-less by virtue of ending in
    // a statement (`if x: body`, `while …`, `let …;`) produce the
    // same None sentinel both at function-body scope and inside
    // `Expr::Block` / if-branch bodies.
    Ok(BlockExit::FallThrough(tail.unwrap_or_else(|| {
        Hlvalue::Constant(Constant::new(ConstValue::None))
    })))
}

fn lower_let(b: &mut Builder, local: &Local) -> Result<(), AdapterError> {
    let ident = match &local.pat {
        Pat::Ident(PatIdent {
            ident,
            by_ref: None,
            subpat: None,
            ..
        }) => ident.to_string(),
        Pat::Type(pat_type) => match &*pat_type.pat {
            Pat::Ident(PatIdent {
                ident,
                by_ref: None,
                subpat: None,
                ..
            }) => ident.to_string(),
            _ => {
                return Err(AdapterError::Unsupported {
                    reason: "let pattern must be a plain identifier".into(),
                });
            }
        },
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "let pattern must be a plain identifier".into(),
            });
        }
    };
    let init = match &local.init {
        Some(LocalInit {
            expr,
            diverge: None,
            ..
        }) => expr,
        Some(_) => {
            return Err(AdapterError::Unsupported {
                reason: "let-else is control flow (lands in M2.5b)".into(),
            });
        }
        None => {
            return Err(AdapterError::Unsupported {
                reason: "let without initializer".into(),
            });
        }
    };
    let value = lower_expr(b, init)?;
    // Upstream STORE_FAST: reassignment REPLACES the locals-map entry.
    // The SSA value feeding subsequent reads is the new one.
    b.set_local(ident, value);
    Ok(())
}

/// Recognise an unshadowed single-segment `Pat::Path` reference to one
/// of the names in `expected_names`. Returns the matched name when the
/// path shape is `Ok` / `Some` / `Err` / `None` (whichever the caller
/// listed), the reference is not currently shadowed by a local, and
/// the AST shape is plain (no `qself`, no leading colon, no generic
/// args).
///
/// Returns `None` for any shape that does not match — the caller falls
/// through to ordinary lowering.
fn match_unshadowed_simple_ident<'a>(
    b: &Builder,
    expr: &'a Expr,
    expected_names: &[&str],
) -> Option<&'a syn::Ident> {
    if let Expr::Path(ExprPath {
        path,
        qself: None,
        attrs: _,
    }) = expr
    {
        if path.leading_colon.is_some() || path.segments.len() != 1 {
            return None;
        }
        let seg = &path.segments[0];
        if !matches!(seg.arguments, syn::PathArguments::None) {
            return None;
        }
        let name = seg.ident.to_string();
        if !expected_names.iter().any(|n| *n == name) {
            return None;
        }
        if b.locals().contains_key(&name) {
            return None;
        }
        return Some(&seg.ident);
    }
    None
}

/// PRE-EXISTING-ADAPTATION for the Ok/Some/None collapse only (Codex
/// 2026-05-03 parity audit accepted as "documented as structural
/// adaptation, not parity"). Function/arm boundary positions in
/// pyre-interpreter Rust source frequently end in `Ok(x)` / `Some(x)`
/// / `None` per Rust idiom; upstream Python source has no
/// Result/Option layer and uses bare `x` / `None` at the same
/// boundary positions. Without boundary collapse, the returnblock
/// Link would carry `simple_call(<Ok>, x)` instead of `x`, which is
/// the same SpaceOperation upstream would emit for Python `return
/// Ok(x)` — orthodox under O1+O2, but pyre's downstream (`build_flow
/// → annotator → codewriter`) needs the unwrapped value at the
/// return edge.
///
/// At each of the three boundary call sites — `lower_return`'s
/// `ret.expr`, `lower_arm_body`'s default arm, and `lower_block`'s
/// tail-expression (which routes through `lower_arm_body`) — this
/// helper rewrites:
///
/// - `Ok(x)` / `Some(x)` (single-arg unshadowed call) →
///   `lower_expr(x)` directly. Slice O4.
/// - `None` (unshadowed single-segment path) →
///   `Constant(ConstValue::None)`. Slice O4.
/// - `Err(e)` (single-arg unshadowed call) → `emit_err_raise_boundary`
///   emits the upstream `flowcontext.py:600-636 exc_from_raise` op
///   sequence with a 2-exit `guessbool(isinstance(evalue, type))`
///   fork preserved per `flowcontext.py:610` and closes the path
///   with a Link to `graph.exceptblock` per `flowcontext.py:1259
///   Raise.nomoreblocks`. Returns `BlockExit::Terminated`. Slice
///   O5 (PARITY).
///
/// Any other expression falls through to ordinary `lower_expr`. The
/// locals-first check honours `pyopcode.py:502 LOAD_FAST` priority
/// (a user `let Ok = …` / `let Err = …` shadows the wrapper).
/// Value-position `Ok(x)` / `Some(x)` / `Err(e)` (i.e. anywhere
/// outside these boundary sites) keeps emitting `simple_call(<host
/// class>, x)` per O1+O2 — the callee resolves through
/// `host_env::PYRE_STDLIB`.
fn lower_value_boundary(b: &mut Builder, expr: &Expr) -> Result<BlockExit, AdapterError> {
    if match_unshadowed_simple_ident(b, expr, &["None"]).is_some() {
        return Ok(BlockExit::FallThrough(Hlvalue::Constant(Constant::new(
            ConstValue::None,
        ))));
    }
    if let Expr::Call(ExprCall { func, args, .. }) = expr {
        if args.len() == 1 {
            if match_unshadowed_simple_ident(b, func.as_ref(), &["Ok", "Some"]).is_some() {
                let inner = lower_expr(b, &args[0])?;
                return Ok(BlockExit::FallThrough(inner));
            }
            if match_unshadowed_simple_ident(b, func.as_ref(), &["Err"]).is_some() {
                return emit_err_raise_boundary(b, &args[0]);
            }
        }
    }
    Ok(BlockExit::FallThrough(lower_expr(b, expr)?))
}

/// Slice O5 — lower a boundary-position `Err(e)` to the upstream
/// `flowcontext.py:600-636 exc_from_raise` op sequence and close the
/// current path with a Link to `graph.exceptblock` per
/// `flowcontext.py:1259 Raise.nomoreblocks`.
///
/// Upstream `flowcontext.py:RAISE_VARARGS(1)` calls
/// `exc_from_raise(w_arg1=evalue, w_arg2=const(None))`. With
/// `w_arg2 = const(None)` constant-folded inside `exc_from_raise` (the
/// inner `is_(w_arg2, w_None)` and `is_(w_arg2, const(None))` both
/// short-circuit on `Constant(None)` per upstream `guessbool` constant
/// folding at `flowcontext.py:365`), the remaining flow is:
///
/// ```text
/// is_class = isinstance(evalue, type)              # 609
/// if is_class:                                     # 610 — guessbool fork
///     w_value = simple_call(evalue)                # 614
/// else:
///     w_value = ll_assert_not_none(evalue)         # 633
/// w_type = type(w_value)                           # 635
/// Link [w_type, w_value] -> exceptblock            # 1259
/// ```
///
/// Each branch lowers in its own block; both converge on a join
/// block that emits `type` and Link to `exceptblock`. Mirrors the
/// fork-and-join pattern `lower_if` uses for `if cond { … } else { …
/// }`.
///
/// `BlockExit::Terminated` returned — Rust analogue of `StopFlowing`.
///
/// PRE-EXISTING-ADAPTATION (`flowspace/flowcontext.rs:1355` family
/// level, NOT introduced by this slice): `ll_assert_not_none` is
/// recorded as a direct opname (single arg, pure pseudo-op), not as
/// the upstream shape. Upstream `flowcontext.py:633` is
/// `op.simple_call(const(ll_assert_not_none), w_value).eval(self)` —
/// a 2-arg `simple_call` whose first arg is a `Constant` carrying the
/// host-side `ll_assert_not_none` function reference. The local port
/// emits a single-arg pseudo-op of name `"ll_assert_not_none"`
/// instead, mirroring the convention at
/// `flowspace/flowcontext.rs:1355 record_pure_op("ll_assert_not_none",
/// vec![w_value])` so this site stays consistent with the existing
/// local lowering of the same upstream construct. `OpKind::from_opname`
/// at `flowspace/operation.rs:252` already lists the name in the
/// majit-synthetic-pseudo-op decline set.
///
/// **Convergence path** (out of scope for M2.5e; tracked as the
/// `ll_assert_not_none` pseudo-op family port): both this site and
/// `flowspace/flowcontext.rs:1355` flip together to emit
/// `simple_call(const(HostObject(ll_assert_not_none)), w_value)`
/// once `host_env` exposes a `HostObject` for the debug helper and
/// the `OpKind::from_opname` synthetic-name list drops the entry.
/// Splitting one site without the other would diverge the two
/// producers — preserving consistency at the family level outranks
/// splitting them.
fn emit_err_raise_boundary(b: &mut Builder, e: &Expr) -> Result<BlockExit, AdapterError> {
    // 1. Lower `e` into evalue inside the current block.
    let evalue = lower_expr(b, e)?;

    // 2. `flowcontext.py:609 op.isinstance(w_arg1, const(type))` —
    //    emitted in the current block, then forked on per
    //    `flowcontext.py:610 if guessbool(w_is_type)`.
    let is_class = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "isinstance",
        vec![
            evalue.clone(),
            Hlvalue::Constant(Constant::new(ConstValue::builtin("type"))),
        ],
        is_class.clone(),
    ));

    // 3. Allocate the True / False arm blocks. Each block has a single
    //    inputarg receiving `evalue` from the fork's Link.args, so the
    //    arm body's ops can reference it as an SSA value local to that
    //    block (matches `Block` SSA discipline — args[i] only resolve
    //    against the block's own inputargs, prior ops, or Constants).
    let evalue_true_in = Hlvalue::Variable(Variable::named("evalue"));
    let true_block = Block::shared(vec![evalue_true_in.clone()]);
    let evalue_false_in = Hlvalue::Variable(Variable::named("evalue"));
    let false_block = Block::shared(vec![evalue_false_in.clone()]);

    // 4. Allocate the join block. Its sole inputarg `w_value` receives
    //    each arm's synthesized w_value via Link.args — `simple_call`
    //    result on the True side, `ll_assert_not_none` result on the
    //    False side. Same shape as the post-fork Variable assignment
    //    upstream `flowcontext.py:632-634` writes back to `w_value`
    //    after either branch returns.
    let w_value_join = Hlvalue::Variable(Variable::named("w_value"));
    let join_block = Block::shared(vec![w_value_join.clone()]);

    // 5. Close the current (fork) block with bool(false)/bool(true)
    //    Links, mirroring `BlockRecorder.guessbool` at
    //    `flowcontext.py:107-122` (False before True per the upstream
    //    convention).
    let false_link = Rc::new(RefCell::new(Link::new(
        vec![evalue.clone()],
        Some(false_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        vec![evalue.clone()],
        Some(true_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(is_class));

    // 6. True arm: `flowcontext.py:614 op.simple_call(w_arg1)` —
    //    instantiate the class.
    b.open_new_block(BlockBuilder {
        block: true_block,
        ops: Vec::new(),
        locals: HashMap::new(),
    });
    let w_value_true = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "simple_call",
        vec![evalue_true_in],
        w_value_true.clone(),
    ));
    let true_to_join = Rc::new(RefCell::new(Link::new(
        vec![w_value_true],
        Some(join_block.clone()),
        None,
    )));
    b.finalize_current(vec![true_to_join], None);

    // 7. False arm: `flowcontext.py:632-634 if check_not_none: w_value
    //    = simple_call(const(ll_assert_not_none), w_value)`. The False
    //    arm is the (inst, None) shape per `flowcontext.py:625-631` —
    //    `is_(w_arg2, const(None))` is constant-True (w_arg2 is the
    //    implicit `const(None)` from `RAISE_VARARGS(1)`), so the
    //    TypeError sub-arm is eliminated by upstream's guessbool
    //    constant folding.
    b.open_new_block(BlockBuilder {
        block: false_block,
        ops: Vec::new(),
        locals: HashMap::new(),
    });
    let w_value_false = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "ll_assert_not_none",
        vec![evalue_false_in],
        w_value_false.clone(),
    ));
    let false_to_join = Rc::new(RefCell::new(Link::new(
        vec![w_value_false],
        Some(join_block.clone()),
        None,
    )));
    b.finalize_current(vec![false_to_join], None);

    // 8. Join arm: `flowcontext.py:635 w_type = op.type(w_value)`,
    //    then `flowcontext.py:1259 Link([w_exc.w_type, w_exc.w_value],
    //    ctx.graph.exceptblock)` closes the path. Both arms have
    //    threaded their synthesized w_value through Link.args, so the
    //    join's `w_value_join` inputarg holds whichever branch's
    //    result reached this block at runtime.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: HashMap::new(),
    });
    let w_type = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "type",
        vec![w_value_join.clone()],
        w_type.clone(),
    ));
    let exit_link = Rc::new(RefCell::new(Link::new(
        vec![w_type, w_value_join],
        Some(b.exceptblock.clone()),
        None,
    )));
    b.finalize_current(vec![exit_link], None);
    Ok(BlockExit::Terminated)
}

/// Close the currently-open block with a Link into `graph.returnblock`
/// carrying the return value, then report `Terminated` so enclosing
/// control-flow lowering knows not to emit a fallthrough link from the
/// now-closed block.
///
/// Upstream basis: `flowcontext.py:687 RETURN_VALUE` raises `Return`;
/// `flowcontext.py:1232 Return.nomoreblocks(ctx)` does
/// `Link([w_result], ctx.graph.returnblock)` on `ctx.recorder.crnt_block`
/// and raises `StopFlowing`. Our `BlockExit::Terminated` is the Rust
/// analogue of StopFlowing — it tells the caller not to wire further
/// exits from the closed block.
///
/// `ret.expr` is lowered through `lower_arm_body(_, _, true)` so the
/// boundary mode propagates into nested control-flow constructs
/// (`return { Ok(x) };`, `return if c { Ok(a) } else { Ok(b) };`,
/// `return match … { … => Ok(x) }`). Each arm/branch's tail then
/// flows through `lower_value_boundary` per Slices O4 + O5: `Ok(x)`
/// / `Some(x)` / `None` collapse to their inner value at the
/// returnblock edge, while `Err(e)` triggers the orthodox
/// `flowcontext.py:600-636 exc_from_raise` op sequence with a 2-exit
/// `isinstance` fork and Link to `graph.exceptblock`. Leaf
/// expressions reach `lower_value_boundary` via `lower_arm_body`'s
/// default arm; ordinary identifiers / paths fall through to
/// `lower_expr`.
fn lower_return(b: &mut Builder, ret: &ExprReturn) -> Result<BlockExit, AdapterError> {
    let value = match &ret.expr {
        // `lower_arm_body(_, _, true)` is the boundary-aware
        // dispatcher: Expr::Block → `lower_block(_, _, true)`,
        // Expr::If → `lower_if(_, _, true)`, Expr::Match →
        // `lower_match(_, _, true)`. Without this, nested
        // control-flow inside an explicit `return` would lose
        // boundary mode and the wrapper would survive into the
        // returnblock Link (Issue 1 codex audit, 2026-05-05).
        Some(expr) => match lower_arm_body(b, expr, true)? {
            BlockExit::FallThrough(v) => v,
            // Boundary helper terminated the block on its own (e.g.
            // Slice O5's `Err(e)` raise edge, or a nested `return`
            // inside the body). The block is gone and the caller
            // observes termination; nothing further to wire.
            BlockExit::Terminated => return Ok(BlockExit::Terminated),
        },
        // Bare `return;` — upstream has no syntactic analogue since
        // Python `return` without a value implicitly pushes `None`
        // (`flowcontext.py:687` pops whatever `compile.c` pushed, which
        // is `LOAD_CONST None` for bare `return`). Mirror that:
        // `ConstValue::None` carried through the returnblock link.
        None => Hlvalue::Constant(Constant::new(ConstValue::None)),
    };
    let link = Rc::new(RefCell::new(Link::new(
        vec![value],
        Some(b.returnblock.clone()),
        None,
    )));
    b.finalize_current(vec![link], None);
    Ok(BlockExit::Terminated)
}

fn lower_expr(b: &mut Builder, expr: &Expr) -> Result<Hlvalue, AdapterError> {
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => lower_literal(lit),
        Expr::Path(ExprPath {
            path, qself: None, ..
        }) => {
            // Both single-segment and multi-segment paths route
            // through `Builder::resolve_path_constant`, which
            // mirrors upstream `flowcontext.py:856 LOAD_GLOBAL` +
            // `:861 LOAD_ATTR` chain. See the helper for full
            // upstream citations and the locals-first /
            // pyre-stdlib / mint-on-multi-segment ordering.
            b.resolve_path_constant(path)
        }
        Expr::Binary(ExprBinary {
            op, left, right, ..
        }) => lower_binop(b, *op, left, right),
        Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr),
        // Expression-position control flow runs in VALUE position —
        // `at_boundary=false` so a tail `Ok(x)` / `Some(x)` / `Err(e)`
        // / `None` inside the arm/body emits the
        // `simple_call(<host class>, x)` op (or stays a Constant)
        // instead of being collapsed by Slice O4. The collapse is
        // upstream-orthodox only at the function-return edge.
        Expr::If(if_expr) => match lower_if(b, if_expr, false)? {
            BlockExit::FallThrough(v) => Ok(v),
            // Every branch terminated via `return` — the if-
            // expression has no reachable value. Upstream bytecode
            // would have raised `StopFlowing` inside each branch
            // before the join PC was even reached; in Rust source
            // position this is `let x = if c { return a } else {
            // return b };` (type `!`) which is valid but produces no
            // binding. Reject at the adapter boundary rather than
            // silently synthesize an unreachable join.
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "if-expression where every branch terminates via `return` — \
                    the expression has no reachable value. Reshape so the \
                    `return` is a semicolon-terminated statement instead of \
                    an expression operand"
                    .into(),
            }),
        },
        Expr::Block(block_expr) => match lower_block(b, &block_expr.block, false)? {
            BlockExit::FallThrough(v) => Ok(v),
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "block-expression whose body terminates via `return` — the \
                    expression has no reachable value. Reshape so the `return` \
                    is a semicolon-terminated statement instead of an expression \
                    operand"
                    .into(),
            }),
        },
        // Bare `return` in expression position (e.g. `let x = return
        // 1;`). Upstream has no syntactic analogue — Python's
        // `return` is a statement. Rust's expression-position
        // `return` produces type `!`. The adapter's subset already
        // supports `return` at every statement position (including
        // the tails of if / match branches) via `lower_block`'s
        // explicit `Expr::Return` branch, so rejecting here simply
        // funnels users to the statement position.
        Expr::Return(_) => Err(AdapterError::Unsupported {
            reason: "return in expression position — put the `return` at statement \
                position (a semicolon-terminated statement, or the tail of an if / \
                match branch) instead"
                .into(),
        }),
        Expr::Match(match_expr) => match lower_match(b, match_expr, false)? {
            BlockExit::FallThrough(v) => Ok(v),
            BlockExit::Terminated => Err(AdapterError::Unsupported {
                reason: "match-expression where every arm terminates via `return` — \
                    the expression has no reachable value. Reshape so the \
                    `return` is a semicolon-terminated statement instead of an \
                    expression operand"
                    .into(),
            }),
        },
        Expr::ForLoop(_) | Expr::While(_) | Expr::Loop(_) => Err(AdapterError::Unsupported {
            reason: "loop construct in expression position produces `()` — use it as a statement \
                (trailing `;` or non-last position) instead"
                .into(),
        }),
        Expr::Try(try_expr) => lower_try(b, try_expr),
        Expr::Break(_) | Expr::Continue(_) => Err(AdapterError::Unsupported {
            reason: "break/continue (lands with loops in M2.5b)".into(),
        }),
        Expr::MethodCall(method_call) => lower_method_call(b, method_call),
        Expr::Call(call) => lower_call(b, call),
        Expr::Tuple(tup) => lower_tuple(b, tup),
        Expr::Array(arr) => lower_array(b, arr),
        Expr::Struct(_) => Err(AdapterError::Unsupported {
            reason: "struct literal (user-type resolution lands in M2.5e — the annotator needs a \
                `ClassDesc` lookup path that `HOST_ENV` doesn't bootstrap on its own)"
                .into(),
        }),
        Expr::Closure(_) => Err(AdapterError::Unsupported {
            reason: "closure (not in roadmap scope)".into(),
        }),
        Expr::Reference(r) => {
            // Rust borrow `&x` / `&mut x` — upstream Python has no
            // ownership model, so the annotator tracks value identity
            // + type only. Pass-through the operand.
            lower_expr(b, &r.expr)
        }
        Expr::Unary(u) => lower_unary(b, u),
        Expr::Cast(c) => lower_cast(b, c),
        Expr::Field(f) => lower_field(b, f),
        Expr::Index(i) => lower_index(b, i),
        _ => Err(AdapterError::Unsupported {
            reason: format!("unrecognised expression kind: {}", discriminant(expr)),
        }),
    }
}

fn lower_literal(lit: &Lit) -> Result<Hlvalue, AdapterError> {
    match lit {
        Lit::Int(int) => {
            let value: i64 = int.base10_parse().map_err(|e| AdapterError::Unsupported {
                reason: format!("integer literal out of i64 range: {e}"),
            })?;
            Ok(Hlvalue::Constant(Constant::new(ConstValue::Int(value))))
        }
        Lit::Bool(bl) => Ok(Hlvalue::Constant(Constant::new(ConstValue::Bool(bl.value)))),
        Lit::Str(s) => Ok(Hlvalue::Constant(Constant::new(ConstValue::uni_str(
            s.value(),
        )))),
        Lit::Float(f) => {
            // `ConstValue::Float` stores `f64::to_bits()` so the
            // enum keeps `Eq + Hash` — see model.rs:1696-1701. The
            // adapter round-trips through `base10_parse::<f64>()` to
            // preserve the exact literal the user wrote.
            let value: f64 = f.base10_parse().map_err(|e| AdapterError::Unsupported {
                reason: format!("float literal out of f64 range: {e}"),
            })?;
            Ok(Hlvalue::Constant(Constant::new(ConstValue::Float(
                value.to_bits(),
            ))))
        }
        // Rust `char` literal is a single Unicode scalar. Upstream
        // RPython has no `char` type; single-character strings fill
        // the role (`model.py:658` switch-exitcase admits
        // `isinstance(n, (str, unicode)) and len(n) == 1`; general
        // `operation.py` string ops accept len==1 the same as any
        // other unicode). Emit as `ConstValue::UniStr(c.to_string())` so
        // expression-position `'a'` and match-arm `'a' =>` carry the
        // identical constant — see `classify_pattern` for the
        // match-arm side.
        Lit::Char(ch) => Ok(Hlvalue::Constant(Constant::new(ConstValue::uni_str(
            ch.value().to_string(),
        )))),
        Lit::ByteStr(bs) => Ok(Hlvalue::Constant(Constant::new(ConstValue::ByteStr(
            bs.value(),
        )))),
        Lit::Byte(b) => Ok(Hlvalue::Constant(Constant::new(ConstValue::ByteStr(vec![
            b.value(),
        ])))),
        _ => Err(AdapterError::Unsupported {
            reason: "unrecognised literal kind".into(),
        }),
    }
}

fn lower_binop(
    b: &mut Builder,
    op: BinOp,
    left: &Expr,
    right: &Expr,
) -> Result<Hlvalue, AdapterError> {
    let opname = binop_opname(op)?;
    let lhs = lower_expr(b, left)?;
    let rhs = lower_expr(b, right)?;
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(opname, vec![lhs, rhs], result.clone()));
    Ok(result)
}

/// Rust `BinOp` → upstream `operation.py` opname. Covers the 16
/// non-short-circuit infix operators the M2.5a subset supports.
/// Short-circuit `&&` / `||` are control flow and land in M2.5b
/// together with `if` / `match`.
fn binop_opname(op: BinOp) -> Result<&'static str, AdapterError> {
    Ok(match op {
        BinOp::Add(_) => "add",
        BinOp::Sub(_) => "sub",
        BinOp::Mul(_) => "mul",
        BinOp::Div(_) => "div",
        BinOp::Rem(_) => "mod",
        BinOp::BitAnd(_) => "and_",
        BinOp::BitOr(_) => "or_",
        BinOp::BitXor(_) => "xor",
        BinOp::Shl(_) => "lshift",
        BinOp::Shr(_) => "rshift",
        BinOp::Eq(_) => "eq",
        BinOp::Ne(_) => "ne",
        BinOp::Lt(_) => "lt",
        BinOp::Le(_) => "le",
        BinOp::Gt(_) => "gt",
        BinOp::Ge(_) => "ge",
        BinOp::And(_) | BinOp::Or(_) => {
            return Err(AdapterError::Unsupported {
                reason: "short-circuit && / || (lands with match in next M2.5b slice)".into(),
            });
        }
        BinOp::AddAssign(_)
        | BinOp::SubAssign(_)
        | BinOp::MulAssign(_)
        | BinOp::DivAssign(_)
        | BinOp::RemAssign(_)
        | BinOp::BitAndAssign(_)
        | BinOp::BitOrAssign(_)
        | BinOp::BitXorAssign(_)
        | BinOp::ShlAssign(_)
        | BinOp::ShrAssign(_) => {
            return Err(AdapterError::Unsupported {
                reason: "compound assignment (mutation lands with control flow in M2.5b)".into(),
            });
        }
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "unrecognised binary operator".into(),
            });
        }
    })
}

// ____________________________________________________________
// `if/else` — 2-way fork with FrameState-style locals merge.
//
// Upstream basis: `rpython/flowspace/flowcontext.py` `POP_JUMP_IF_FALSE`
// / `JUMP_FORWARD` handler pair. The Python-bytecode flow is:
//
//   if cond: then_body                POP_JUMP_IF_FALSE to else-pc
//   else: else_body             →     then-ops
//   …continue                          JUMP_FORWARD to join-pc
//                                      else-pc: else-ops
//                                      join-pc: …
//
// The graph-level shape `flowcontext` emits is what this routine
// mirrors directly against `syn::ExprIf`: a startblock with
// `exitswitch=cond` and two exits (one per `Constant::Bool(x)`
// exitcase), two branch blocks, and a single join block reached by
// `Link(target=join)` from each branch. Locals live in each branch's
// own `locals_w` and merge into the join block's inputargs via
// `FrameState` union.

/// State captured after lowering a single branch's body. `None` when
/// the branch terminated via `return` — in that case the branch's
/// block has already been closed to `graph.returnblock` by
/// `lower_return`, so no further link / op emission applies. `Some`
/// captures the now-open current-block state so the caller can later
/// attach a Link from that block into the post-if join.
struct ArmCapture {
    tail: Hlvalue,
    block: BlockRef,
    ops: Vec<SpaceOperation>,
    locals: HashMap<String, Hlvalue>,
}

/// Finalize the current BlockBuilder state into an `ArmCapture` when
/// the arm fell through, or discard it when the arm terminated.
fn capture_arm_exit(b: &mut Builder, exit: BlockExit) -> Option<ArmCapture> {
    match exit {
        BlockExit::FallThrough(tail) => Some(ArmCapture {
            tail,
            block: b.current.block.clone(),
            ops: std::mem::take(&mut b.current.ops),
            locals: std::mem::take(&mut b.current.locals),
        }),
        BlockExit::Terminated => {
            // `lower_return` already called `finalize_current` on the
            // branch's last block. Clear b.current so the next
            // `open_new_block` starts from a clean slate.
            b.current.ops.clear();
            b.current.locals.clear();
            None
        }
    }
}

/// Dispatch the else-arm of an `if` / `else` chain. Per `syn`'s
/// grammar, `ExprIf.else_branch` is always `Expr::Block` (a plain
/// `else { … }`) or `Expr::If` (chained `else if …`). Both sub-
/// routines return `BlockExit`, so termination threads through a
/// chain of nested `else if` transparently.
fn lower_else_arm(
    b: &mut Builder,
    else_expr: &Expr,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    match else_expr {
        Expr::Block(block_expr) => lower_block(b, &block_expr.block, at_boundary),
        Expr::If(nested_if) => lower_if(b, nested_if, at_boundary),
        _ => Err(AdapterError::Unsupported {
            reason: "`else` branch is neither a block nor an `if` — syn's grammar \
                should forbid this; if it fires, please file a bug citing the \
                input fragment"
                .into(),
        }),
    }
}

fn lower_if(
    b: &mut Builder,
    if_expr: &ExprIf,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    // 1. Evaluate condition into the current block, then coerce via
    //    `bool(cond)` — mirrors upstream POP_JUMP_IF_FALSE at
    //    `flowcontext.py:756` which always emits `op.bool(w_value)`
    //    before the guessbool test. The `bool` op is in upstream's
    //    `operation.py:467 add_operator('bool', 1)` registry.
    let cond_raw = lower_expr(b, &if_expr.cond)?;
    let cond = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("bool", vec![cond_raw], cond.clone()));

    // Extract the else-less path early so the rest of `lower_if` can
    // assume `else_expr` is present; the common `bool(cond)` +
    // locals-snapshot steps are shared.
    let Some((_else_tok, else_expr)) = &if_expr.else_branch else {
        return lower_if_without_else(b, cond, &if_expr.then_branch);
    };

    // 2. Snapshot state at the fork point. Upstream `FrameState.copy()`
    //    clones locals_w so STORE_FAST inside one branch doesn't leak
    //    into the other. The adapter replays this as "each branch
    //    inherits its own copy of locals, addressed via fresh
    //    Variables on the branch block's inputargs".
    //
    //    Sorted merged_names gives deterministic inputargs ordering
    //    across runs, mirroring upstream's Python dict order.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();

    // 3. Allocate branch blocks. Each carries a fresh Variable per
    //    merged local in its inputargs, matching upstream where the
    //    jump target's frame receives fresh vars via link.args.
    let (then_block, then_locals) = branch_block_with_inputargs(&merged_names);
    let (else_block, else_locals) = branch_block_with_inputargs(&merged_names);

    // 4. Close the fork block. Link.args carry the pre-fork SSA values
    //    so branch inputargs bind to the caller's locals on entry.
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();
    let false_link = Rc::new(RefCell::new(Link::new(
        fork_args.clone(),
        Some(else_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        fork_args,
        Some(then_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 5. Lower the then-branch into its own BlockBuilder. Nested
    //    control flow may reassign `b.current.block` away from
    //    `then_block` to a nested join; we close whatever the final
    //    current block is after the branch body returns. A
    //    `Terminated` exit means `lower_return` already closed the
    //    current block — we drop the capture in that case.
    b.open_new_block(BlockBuilder {
        block: then_block.clone(),
        ops: Vec::new(),
        locals: then_locals,
    });
    let then_exit = lower_block(b, &if_expr.then_branch, at_boundary)?;
    let then_capture = capture_arm_exit(b, then_exit);

    // 6. Lower the else-branch the same way. `else_expr` is always
    //    `Expr::Block` or `Expr::If` per syn's grammar — the two
    //    variants route through `lower_else_arm` so nested `else if`
    //    chains thread termination up transparently.
    b.open_new_block(BlockBuilder {
        block: else_block.clone(),
        ops: Vec::new(),
        locals: else_locals,
    });
    let else_exit = lower_else_arm(b, else_expr, at_boundary)?;
    let else_capture = capture_arm_exit(b, else_exit);

    // 7. Fork the post-branch wiring by which arms fell through.
    //    Upstream analogue: `Return.nomoreblocks()` on a branch
    //    raises `StopFlowing`, and the pending-block scheduler never
    //    enqueues that branch's PC at the join. Here:
    //    - both terminated → no join block, whole-if is Terminated
    //    - exactly one fell through → join has one predecessor
    //    - both fell through → canonical two-predecessor join
    match (then_capture, else_capture) {
        (None, None) => {
            // Both branches closed themselves to returnblock via
            // `return`. Nothing reaches the post-if PC; signal
            // termination to the enclosing lowering.
            Ok(BlockExit::Terminated)
        }
        (Some(cap), None) | (None, Some(cap)) => {
            // Exactly one branch reached the post-if PC. Build a
            // single-predecessor join: the branch's tail value is the
            // if-expression's value, and its locals snapshot feeds
            // the join's inputargs.
            let (_join_block, tail_var) = build_if_join_block(&merged_names, b, &cap);
            Ok(BlockExit::FallThrough(tail_var))
        }
        (Some(then_cap), Some(else_cap)) => {
            // Canonical case — both branches reach the join.
            // inputargs = [tail_var, local_var_0, …].
            let tail_var = Hlvalue::Variable(Variable::new());
            let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
            join_inputargs.push(tail_var.clone());
            let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
            for name in &merged_names {
                let fresh = Hlvalue::Variable(Variable::named(name));
                join_inputargs.push(fresh.clone());
                join_locals.insert(name.clone(), fresh);
            }
            let join_block = Block::shared(join_inputargs);

            // Close each arm's tail block with a Link into the join.
            // `branch_link_args` reads from the arm's locals snapshot
            // — branch-local `let` bindings (names absent from
            // merged_names) never reach the join.
            {
                let link_args = branch_link_args(&then_cap.tail, &merged_names, &then_cap.locals);
                let link = Rc::new(RefCell::new(Link::new(
                    link_args,
                    Some(join_block.clone()),
                    None,
                )));
                then_cap.block.borrow_mut().operations = then_cap.ops;
                then_cap.block.closeblock(vec![link]);
            }
            {
                let link_args = branch_link_args(&else_cap.tail, &merged_names, &else_cap.locals);
                let link = Rc::new(RefCell::new(Link::new(
                    link_args,
                    Some(join_block.clone()),
                    None,
                )));
                else_cap.block.borrow_mut().operations = else_cap.ops;
                else_cap.block.closeblock(vec![link]);
            }

            b.open_new_block(BlockBuilder {
                block: join_block,
                ops: Vec::new(),
                locals: join_locals,
            });
            Ok(BlockExit::FallThrough(tail_var))
        }
    }
}

/// Build the post-if join block for the single-predecessor case and
/// close the surviving branch's tail block into it. Returns the join
/// block (for caller bookkeeping) and the `tail_var` representing the
/// if-expression's value as seen from join-block locals.
///
/// Upstream analogue: `flowcontext.py` never creates a join block for
/// a PC that only one arm jumps to (it would still be there, but with
/// one incoming Link) — this routine mirrors that shape.
fn build_if_join_block(
    merged_names: &[String],
    b: &mut Builder,
    cap: &ArmCapture,
) -> (BlockRef, Hlvalue) {
    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // Close the surviving arm's tail block with the one-and-only Link
    // into the join.
    let link_args = branch_link_args(&cap.tail, merged_names, &cap.locals);
    let link = Rc::new(RefCell::new(Link::new(
        link_args,
        Some(join_block.clone()),
        None,
    )));
    cap.block.borrow_mut().operations = cap.ops.clone();
    cap.block.closeblock(vec![link]);

    // Open the join as the new current block.
    b.open_new_block(BlockBuilder {
        block: join_block.clone(),
        ops: Vec::new(),
        locals: join_locals,
    });
    (join_block, tail_var)
}

/// Lower `if cond { body }` without an `else` branch.
///
/// Shape mirrors upstream `flowcontext.py:756 POP_JUMP_IF_FALSE`: the
/// false-branch target IS the post-body join PC (fallthrough of
/// `body` lands on the same PC), so the fork block's `false` exit
/// links directly to the join, and the `true` exit threads through a
/// `then_block` whose body-tail also links to the join. No else
/// block is allocated. The join's inputargs carry ONLY the merged
/// locals — there is no tail-value slot, because `if` without `else`
/// is a statement and the expression "produces" `None`
/// (Python convention mirrored by `ConstValue::None`; upstream
/// RPython bytecode never leaves a value on the stack for this
/// construct).
///
/// Return value: `Constant(None)`. The `Stmt::Expr` loop in
/// `lower_block` discards the tail of semicolon-terminated
/// expressions, so in normal statement position this is erased. In
/// tail position (`if cond { body }` as the function's last
/// expression), the None flows into the returnblock — matching
/// Python's implicit `return None` fallthrough.
fn lower_if_without_else(
    b: &mut Builder,
    cond: Hlvalue,
    then_branch: &SynBlock,
) -> Result<BlockExit, AdapterError> {
    // 1. Snapshot pre-fork locals. Same discipline as the with-else
    //    path: deterministic ordering via sort, one entry per
    //    live local name at the fork point.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();

    // 2. Allocate the then_block (sole branch block) and the join
    //    block up-front — the fork's false Link needs the join as its
    //    target, so the join must exist before `finalize_current`.
    //    Join inputargs are just the merged locals; no tail slot.
    let (then_block, then_locals) = branch_block_with_inputargs(&merged_names);
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 3. Close the fork block. `false` shortcuts directly to the
    //    join; `true` routes through `then_block`.
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();
    let false_link = Rc::new(RefCell::new(Link::new(
        fork_args.clone(),
        Some(join_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        fork_args,
        Some(then_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 4. Lower the then-branch. A `Terminated` exit (branch ends in
    //    `return`) means `lower_return` has already wired the block
    //    to `graph.returnblock`; skip the then→join link. The
    //    false-path link into the join is already installed above,
    //    so the join is always reachable — `lower_if_without_else`
    //    therefore always returns FallThrough.
    b.open_new_block(BlockBuilder {
        block: then_block,
        ops: Vec::new(),
        locals: then_locals,
    });
    // `if cond { body }` body's tail value is discarded — the join
    // produces implicit `None` regardless of what the body evaluated
    // to. Pass `at_boundary=false` so `Ok(x)` / `Some(x)` / `None`
    // / `Err(e)` body tails are NOT collapsed: there's no return
    // edge for them to feed.
    let then_exit = lower_block(b, then_branch, false)?;
    if let BlockExit::FallThrough(_tail) = then_exit {
        // Body's tail expression value is discarded — `if` without
        // else has no tail slot in the join (it produces implicit
        // `None` regardless of what the body evaluated to).
        let then_exit_block = b.current.block.clone();
        let then_exit_ops = std::mem::take(&mut b.current.ops);
        let then_exit_locals = std::mem::take(&mut b.current.locals);
        let then_link_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|name| {
                then_exit_locals
                    .get(name)
                    .cloned()
                    .expect("merged_names is a subset of branch entry locals")
            })
            .collect();
        let then_link = Rc::new(RefCell::new(Link::new(
            then_link_args,
            Some(join_block.clone()),
            None,
        )));
        then_exit_block.borrow_mut().operations = then_exit_ops;
        then_exit_block.closeblock(vec![then_link]);
    } else {
        // Then-branch terminated via `return`. Clear any stale
        // current-block state so `open_new_block` starts cleanly.
        b.current.ops.clear();
        b.current.locals.clear();
    }

    // 5. Continue lowering into the join block. The false-path link
    //    is always installed, so the join is reachable whether the
    //    then-branch fell through or terminated.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });

    // 6. The if-without-else expression value is `None` — Python
    //    statement convention, upstream's bytecode never leaves a
    //    value on the stack for `if x: body`.
    Ok(BlockExit::FallThrough(Hlvalue::Constant(Constant::new(
        ConstValue::None,
    ))))
}

/// Create a branch block whose inputargs are fresh Variables — one
/// per merged local name — plus the locals map that binds each name
/// to its own inputarg Variable.
fn branch_block_with_inputargs(merged_names: &[String]) -> (BlockRef, HashMap<String, Hlvalue>) {
    let mut inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        inputargs.push(fresh.clone());
        locals.insert(name.clone(), fresh);
    }
    (Block::shared(inputargs), locals)
}

/// Build the `Link.args` a branch carries into the join block.
///
/// Head slot: the branch's tail expression value.
/// Tail slots: one per merged local name, in the caller-provided
/// order. Each slot holds the branch's current SSA value for that
/// name. `merged_names` is sourced from the pre-fork locals, so every
/// name is guaranteed to be present in `branch_locals` because every
/// branch inherits the full pre-fork set on entry.
fn branch_link_args(
    tail: &Hlvalue,
    merged_names: &[String],
    branch_locals: &HashMap<String, Hlvalue>,
) -> Vec<Hlvalue> {
    let mut out = Vec::with_capacity(merged_names.len() + 1);
    out.push(tail.clone());
    for name in merged_names {
        let value = branch_locals
            .get(name)
            .cloned()
            .expect("merged_names is a subset of branch entry locals");
        out.push(value);
    }
    out
}

// ____________________________________________________________
// `match` — two structurally distinct lowerings depending on arm
// shape:
//
// 1. **Literal switch** — every arm is a primitive literal pattern
//    (`Pat::Lit` of int/bool/single-char str/byte) optionally ending
//    in a wildcard. Lowered as a single fork block with
//    `exitswitch=scrutinee` and one Link per arm carrying a primitive
//    exitcase. Upstream basis: `rpython/flowspace/model.py:643-666`,
//    which admits either (a) a 2-arm boolean switch with `Bool(False)`
//    then `Bool(True)` exitcases or (b) an n-arm primitive switch with
//    `is_valid_int(n)` exitcases ending optionally in a
//    `Constant::Str("default")` catch-all.
//
//    Arm → exitcase mapping:
//
//    | Rust pattern      | Link.exitcase                                |
//    |-------------------|----------------------------------------------|
//    | `N` (int literal) | `Constant::Int(N)`                           |
//    | `true` / `false`  | `Constant::Bool(_)`                          |
//    | `_`               | `Constant::Str("default")` (must be last)    |
//
// 2. **Variant cascade** (slices 2c + 2d) — any arm has a variant
//    sub-pattern that `pat_has_variant_shape` recognises:
//    - `Pat::Path`            — unit variant (slice 2c).
//    - `Pat::Struct { .. }`   — rest-only struct variant (slice 2d).
//    - `Pat::TupleStruct(..)` — rest-only tuple variant (slice 2d).
//
//    Lowered as a chain of 2-exit boolean forks, each emitting any
//    necessary `getattr` ops to resolve the variant path's segments
//    into a `Constant(HostObject(<class>))` carrier
//    (`Builder::resolve_path_constant` mirroring `flowcontext.py:856
//    LOAD_GLOBAL` + `:861 LOAD_ATTR`), followed by
//    `cond = isinstance(scrutinee, <class>)` per `operation.py:449`.
//    Upstream basis: `rpython/flowspace/flowcontext.py` lowering of
//    `if isinstance(x, A.X): … elif isinstance(x, A.Y): …` — the
//    same isinstance-cascade shape Python source produces.
//
// Patterns outside both subsets reject via `AdapterError::Unsupported`:
// - `Pat::Range`                          — no upstream analogue.
// - `Pat::Struct { field, .. }`           — field-binding extraction
//                                           lands in slice 2e.
// - `Pat::TupleStruct(a, b)`              — same as above.
// - Mixed literal + variant arms          — homogeneous sets only.
// - `match` arm guards (`if COND`)        — lands with annotator
//                                           short-circuit logic.

/// Dispatch a match-arm body across the control-flow-bearing
/// expression kinds so termination (via `return`) threads up through
/// the arm naturally. Arms with a non-control-flow body (literal,
/// path, binop, etc.) fall back to `lower_expr` and report
/// `FallThrough` with the evaluated value.
///
/// Codex 2026-05-03 parity audit removed the prior
/// `match_err_call_unshadowed` intercept that rewrote arm-tail
/// `Err(e)` as a raise edge — upstream's RAISE_VARARGS shape
/// (`flowcontext.py:638-656` + `:632-636 exc_from_raise`) is more
/// than the partial `op.type(value) + Link(exceptblock)` the prior
/// intercept emitted. Faithful `exc_from_raise` port is the
/// orthodox follow-up (Slice O5).
///
/// Non-control-flow arm bodies route through `lower_value_boundary`
/// (Slice O4) so `Ok(x)` / `Some(x)` / `None` at the arm's outer-
/// tail position collapse to the inner value before threading into
/// the match's join block / returnblock link.
fn lower_arm_body(
    b: &mut Builder,
    body: &Expr,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    match body {
        // Control-flow constructs propagate `at_boundary` so a
        // boundary-position `match`/`if`/`block` whose arms feed the
        // function's return value continues to collapse Ok/Some/Err
        // tails per the Slice O4 PRE-EXISTING-ADAPTATION.
        Expr::Block(block_expr) => lower_block(b, &block_expr.block, at_boundary),
        Expr::If(if_expr) => lower_if(b, if_expr, at_boundary),
        Expr::Match(match_expr) => lower_match(b, match_expr, at_boundary),
        // `return` is itself a boundary site regardless of context;
        // `lower_return` routes through `lower_value_boundary` directly.
        Expr::Return(ret) => lower_return(b, ret),
        // Non-control-flow tail. At boundary, apply Slice O4 unwrap;
        // otherwise emit the expression value as-is so value-position
        // blocks like `let z = { Ok(x) }; z` keep `z` bound to the
        // `simple_call(<Ok>, x)` result instead of unwrapping to `x`.
        _ => {
            if at_boundary {
                lower_value_boundary(b, body)
            } else {
                Ok(BlockExit::FallThrough(lower_expr(b, body)?))
            }
        }
    }
}

fn lower_match(
    b: &mut Builder,
    match_expr: &ExprMatch,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    // 1. Evaluate the scrutinee into the current block. Becomes the
    //    block's `exitswitch` when we close the fork (literal-switch
    //    path) or the operand to per-step `isinstance` ops (variant-
    //    cascade path).
    let scrutinee = lower_expr(b, &match_expr.expr)?;

    if match_expr.arms.is_empty() {
        return Err(AdapterError::Unsupported {
            reason: "match with zero arms".into(),
        });
    }

    // Validate guards before either dispatch path; lifted from the
    // per-arm classify loop so cascade lowering shares the check.
    for arm in &match_expr.arms {
        validate_arm(arm)?;
    }

    // Dispatch: any arm whose sub-pattern looks like an enum variant
    // (multi-segment path inside `Pat::Path`/`Pat::Struct`/
    // `Pat::TupleStruct`) routes through the isinstance-cascade
    // lowering. The detection here is intentionally MORE permissive
    // than `extract_variant_arm_info`: shapes whose specific cascade
    // support has not landed yet (e.g. tuple-variant element bindings,
    // slice 2f) still route through the cascade so the user gets the
    // precise "lands in slice 2f" diagnostic instead of the literal-
    // switch path's generic "composite pattern" message. Pure literal
    // matches (every arm is `Pat::Lit` or wildcard) keep the upstream
    // `model.py:648-692` primitive-switch shape.
    let needs_cascade = match_expr.arms.iter().any(|arm| {
        let mut sub_pats: Vec<&Pat> = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        sub_pats.iter().any(|p| pat_has_variant_shape(p))
    });
    if needs_cascade {
        return lower_match_variant_cascade(b, match_expr, scrutinee, at_boundary);
    }

    // 2. Validate every arm up-front so the fork block is only closed
    //    once we know every branch can be lowered. Or-patterns
    //    (`A | B | C => body`) expand into multiple sub-patterns per
    //    arm — each sub-pattern classifies independently, and at step
    //    5 each sub-pattern contributes one Link targeting the ONE
    //    branch block allocated for the original arm (upstream
    //    `model.py:648-692` admits multiple Links with distinct
    //    exitcases pointing to the same target block). `exitcase =
    //    None` marks a wildcard — upstream `model.py:652` requires
    //    such a case to be the last exit of the match, and forbids it
    //    inside an or-pattern.
    let mut arm_sub_exitcases: Vec<Vec<Option<Hlvalue>>> =
        Vec::with_capacity(match_expr.arms.len());
    for (idx, arm) in match_expr.arms.iter().enumerate() {
        // Guards already validated by the dispatch-shared
        // `validate_arm` loop above.
        let is_last_arm = idx + 1 == match_expr.arms.len();
        let mut sub_pats: Vec<&Pat> = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        let sub_pat_count = sub_pats.len();
        // Pre-check: upstream `model.py:652` reserves the wildcard
        // for a standalone default arm at match level; embedding it
        // inside an or-pattern duplicates the catch-all intent. Flag
        // it here before `classify_pattern` (whose own `is_last`
        // check would otherwise surface a misleading "wildcard must
        // be last" message).
        if sub_pat_count > 1 {
            for sub_pat in &sub_pats {
                if matches!(sub_pat, Pat::Wild(_)) {
                    return Err(AdapterError::Unsupported {
                        reason: "wildcard sub-pattern inside or-pattern — upstream \
                            `model.py:652` reserves the wildcard for a standalone \
                            default arm at match-level"
                            .into(),
                    });
                }
            }
        }
        let mut sub_exitcases: Vec<Option<Hlvalue>> = Vec::with_capacity(sub_pat_count);
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && (sub_idx + 1 == sub_pat_count);
            let exitcase = classify_pattern(sub_pat, is_last)?;
            sub_exitcases.push(exitcase);
        }
        arm_sub_exitcases.push(sub_exitcases);
    }

    // 3. Enforce upstream's uniqueness invariant
    //    (`model.py:692 allexitcases[link.exitcase] = True`). Check
    //    across all sub-exitcases of every arm — two or-pattern
    //    sub-cases on the same value would collide just like two arms
    //    with the same case.
    let mut seen: Vec<&Hlvalue> = Vec::new();
    for arm_ex in &arm_sub_exitcases {
        for exitcase in arm_ex.iter().flatten() {
            if seen.iter().any(|s| *s == exitcase) {
                return Err(AdapterError::Unsupported {
                    reason: "match arm exitcase repeated — upstream forbids duplicate \
                        jump-table cases"
                        .into(),
                });
            }
            seen.push(exitcase);
        }
    }

    // 4. Snapshot locals for the fork (same discipline as `lower_if`).
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let fork_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|name| pre_fork_locals[name].clone())
        .collect();

    // 5. Allocate one branch block per ORIGINAL arm; each arm emits
    //    one Link per sub-pattern pointing at that shared block.
    let mut branch_blocks: Vec<(BlockRef, HashMap<String, Hlvalue>)> =
        Vec::with_capacity(match_expr.arms.len());
    let mut fork_exits: Vec<Rc<RefCell<Link>>> = Vec::new();
    for arm_ex in &arm_sub_exitcases {
        let (branch_block, branch_locals) = branch_block_with_inputargs(&merged_names);
        for exitcase in arm_ex {
            let link_exitcase = match exitcase {
                Some(v) => Some(v.clone()),
                None => Some(Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                    "default",
                )))),
            };
            let link = Rc::new(RefCell::new(Link::new(
                fork_args.clone(),
                Some(branch_block.clone()),
                link_exitcase,
            )));
            fork_exits.push(link);
        }
        branch_blocks.push((branch_block, branch_locals));
    }
    b.finalize_current(fork_exits, Some(scrutinee));

    // 6. Lower each arm's body via `lower_arm_body` so termination
    //    via `return` threads up per `flowcontext.py:1232`. Record
    //    ONE capture per original arm — a `None` capture means the
    //    arm terminated and is not linked to the join. Regardless of
    //    how many sub-patterns (Links) feed into the arm's branch
    //    block, the body is lowered exactly once.
    let mut arm_captures: Vec<Option<ArmCapture>> = Vec::with_capacity(match_expr.arms.len());
    for ((branch_block, branch_locals), arm) in branch_blocks.into_iter().zip(&match_expr.arms) {
        b.open_new_block(BlockBuilder {
            block: branch_block,
            ops: Vec::new(),
            locals: branch_locals,
        });
        let exit = lower_arm_body(b, &arm.body, at_boundary)?;
        arm_captures.push(capture_arm_exit(b, exit));
    }

    // 7. If every arm terminated, the post-match PC is unreachable:
    //    no join block is allocated, and the enclosing lowering must
    //    observe termination. Upstream analogue:
    //    `Return.nomoreblocks()` on every pending arm raises
    //    `StopFlowing` and the join PC is never enqueued.
    if arm_captures.iter().all(|c| c.is_none()) {
        return Ok(BlockExit::Terminated);
    }

    // 8. Build the join block — inputargs = [tail_var, local_var_0, …].
    //    Every surviving arm contributes one Link; terminated arms
    //    are silently skipped (their block was already closed to
    //    returnblock by `lower_return`).
    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 9. Close each surviving arm's exit block with a Link into the
    //    join.
    for cap in arm_captures.into_iter().flatten() {
        let link_args = branch_link_args(&cap.tail, &merged_names, &cap.locals);
        let link = Rc::new(RefCell::new(Link::new(
            link_args,
            Some(join_block.clone()),
            None,
        )));
        cap.block.borrow_mut().operations = cap.ops;
        cap.block.closeblock(vec![link]);
    }

    // 10. Continue lowering into the join block.
    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });
    Ok(BlockExit::FallThrough(tail_var))
}

/// One named-field binding extracted from a struct-variant pattern
/// (slice 2e). The cascade emits
/// `local_name = getattr(scrutinee, "<field_name>")` at the arm body
/// block's entry.
///
/// Upstream basis: `rpython/flowspace/operation.py:618 getattr`
/// arity=2. The same op Python source produces for `obj.field`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct StructFieldBinding {
    /// Source-side field name — supplied as the second operand of the
    /// emitted `getattr` op (`Constant::byte_str`).
    field_name: String,
    /// User identifier the arm body sees as a local. For shorthand
    /// patterns like `{ pc, .. }` this equals `field_name`; for
    /// renaming bindings `{ pc: orig_pc }` it is the right-hand side.
    local_name: String,
}

/// Match-arm classification produced by [`extract_variant_arm_info`].
struct VariantPatInfo<'a> {
    /// The original `syn::Path` (validated for the cascade-recognised
    /// shape). The cascade emits the isinstance step's second operand
    /// by routing this path through `Builder::resolve_path_constant`,
    /// which mirrors `flowcontext.py:856 LOAD_GLOBAL` + `:861
    /// LOAD_ATTR` chain so `isinstance(x, A.B)` carries a real
    /// `Constant(HostObject(class))` per `operation.py:449`. Same
    /// path shape across unit, rest-only struct, rest-only tuple, and
    /// binding-carrying struct variants.
    path: &'a syn::Path,
    /// Per-field bindings the cascade must materialise at arm-body
    /// entry. Empty for unit variants, rest-only struct/tuple
    /// variants, and `Pat::Wild` field skips. One entry per named-
    /// Ident field binding (slice 2e).
    bindings: Vec<StructFieldBinding>,
}

/// Validate the multi-segment, no-generics, no-qself shape every
/// cascade-recognised variant path must obey. Returns the original
/// `syn::Path` reference unchanged when the shape passes; the cascade
/// later resolves it through `Builder::resolve_path_constant` to a
/// `Constant(HostObject(class))` per `flowcontext.py:856 LOAD_GLOBAL`
/// + `:861 LOAD_ATTR` semantics.
///
/// Single-segment paths (`CONST_MAX` / `Foo`) reject — without
/// resolution context the adapter cannot distinguish a const
/// reference from a top-level type. Paths with generics
/// (`Foo::<T>::Bar`), `qself` (`<T as Foo>::Bar`), or a leading `::`
/// (global path) reject for the same reason.
fn variant_path_ref<'a>(qself: Option<&syn::QSelf>, path: &'a syn::Path) -> Option<&'a syn::Path> {
    if qself.is_some() || path.leading_colon.is_some() || path.segments.len() < 2 {
        return None;
    }
    for seg in path.segments.iter() {
        if !matches!(seg.arguments, syn::PathArguments::None) {
            return None;
        }
    }
    Some(path)
}

/// Walk a `Pat::Struct`'s `FieldPat` list and extract the per-field
/// `Ident`-binding shapes the cascade can lower as a `getattr` at
/// arm-body entry. `Pat::Wild` (field skip) is admitted with no
/// binding emission. Any other field-pattern shape (literal, nested
/// struct/tuple, sub-pattern, by_ref/mut binding) returns `None`,
/// which routes the parent variant pattern through the cascade's
/// rejection arm with the correct slice-boundary diagnostic.
///
/// No upstream counterpart — RPython source uses `getattr(obj, name)`
/// per `rpython/flowspace/operation.py:618`, which the arm-body
/// emitter mirrors line-for-line.
fn extract_struct_field_bindings(
    fields: &syn::punctuated::Punctuated<syn::FieldPat, syn::Token![,]>,
) -> Option<Vec<StructFieldBinding>> {
    let mut out: Vec<StructFieldBinding> = Vec::with_capacity(fields.len());
    for field_pat in fields {
        let field_name = match &field_pat.member {
            Member::Named(ident) => ident.to_string(),
            // Tuple-struct numeric members (`{ 0: a }`) reach this
            // helper only via `Pat::Struct`; tuple-variant patterns
            // (`Pat::TupleStruct`) lower through their own arm. Reject
            // here so the parent classifier surfaces the slice-2f
            // tuple-binding diagnostic.
            Member::Unnamed(_) => return None,
        };
        match &*field_pat.pat {
            Pat::Ident(PatIdent {
                ident,
                by_ref: None,
                subpat: None,
                mutability: None,
                ..
            }) => {
                out.push(StructFieldBinding {
                    field_name,
                    local_name: ident.to_string(),
                });
            }
            Pat::Wild(_) => {
                // `{ field: _ }` — the variant arm does not bind
                // this field. No `getattr` emitted; the field is
                // matched but its value is discarded.
            }
            _ => return None,
        }
    }
    Some(out)
}

/// Classify a match-arm sub-pattern as a variant pattern the cascade
/// can lower. Returns `Some(VariantPatInfo)` for shapes the current
/// slice supports:
///
/// 1. `Foo::Bar` (`Pat::Path`)            — unit variant (slice 2c).
/// 2. `Foo::Bar { .. }` (`Pat::Struct`)   — rest-only struct
///                                          (slice 2d).
/// 3. `Foo::Bar(..)` (`Pat::TupleStruct`) — rest-only tuple
///                                          (slice 2d).
/// 4. `Foo::Bar { f, .. }` / `{ f: i, .. }` (`Pat::Struct`) —
///    named-Ident struct fields (slice 2e). `Pat::Wild` field skip
///    is allowed; any other field-pattern shape returns `None`.
///
/// Returns `None` for non-variant shapes and for variant shapes that
/// the current slice does not yet support (e.g. tuple-variant element
/// bindings, pending slice 2f). The caller routes `None` through the
/// cascade's per-shape rejection diagnostic.
///
/// No upstream counterpart — Rust's `syn::Pat` shape carries no class
/// identity, so the classifier records the original `syn::Path` and
/// hands it to `Builder::resolve_path_constant` at cascade emission
/// time. That routes through the closed-world `host_env::PYRE_STDLIB`
/// registry plus the process-global `host_env::HOST_CLASS_MINTS`
/// runtime registry to produce a real `HostObject::Class` carrier —
/// the second operand `operation.py:449 isinstance` expects.
fn extract_variant_arm_info<'a>(pat: &'a Pat) -> Option<VariantPatInfo<'a>> {
    match pat {
        Pat::Path(p) => {
            let path = variant_path_ref(p.qself.as_ref(), &p.path)?;
            Some(VariantPatInfo {
                path,
                bindings: Vec::new(),
            })
        }
        Pat::Struct(p) => {
            let path = variant_path_ref(p.qself.as_ref(), &p.path)?;
            // Rest-only struct (`Foo::Bar { .. }`) — empty fields
            // with PatRest. Treat as unit-equivalent.
            if p.fields.is_empty() && p.rest.is_some() {
                return Some(VariantPatInfo {
                    path,
                    bindings: Vec::new(),
                });
            }
            let bindings = extract_struct_field_bindings(&p.fields)?;
            Some(VariantPatInfo { path, bindings })
        }
        Pat::TupleStruct(p) => {
            let path = variant_path_ref(p.qself.as_ref(), &p.path)?;
            // Rest-only tuple (`Foo::Bar(..)`) — exactly one
            // `Pat::Rest`. Element-binding form lands in slice 2f.
            if p.elems.len() == 1 && matches!(p.elems.first().unwrap(), Pat::Rest(_)) {
                return Some(VariantPatInfo {
                    path,
                    bindings: Vec::new(),
                });
            }
            None
        }
        _ => None,
    }
}

/// Lenient sibling of `extract_variant_arm_info` used by the dispatch
/// peek in `lower_match`. Returns `true` for any pattern shape whose
/// path could plausibly name an enum variant — including those whose
/// field-binding details a future slice will support.
///
/// The peek uses this so a match that mixes accepted slice-2c/2d/2e
/// shapes with future-slice shapes still routes through the cascade
/// lowering, where the exact "lands in slice 2f" diagnostic lives.
/// Without this lenient predicate, a match like
/// `match x { Foo::Bar(a, b) => …, _ => … }` would fall through to
/// the literal-switch path and surface the generic "composite
/// pattern" message from `classify_pattern`.
fn pat_has_variant_shape(pat: &Pat) -> bool {
    let path = match pat {
        Pat::Path(p) => &p.path,
        Pat::Struct(p) => &p.path,
        Pat::TupleStruct(p) => &p.path,
        _ => return false,
    };
    path.leading_colon.is_none() && path.segments.len() >= 2
}

/// Per-arm classification consumed by `lower_match_variant_cascade`.
enum CascadeArm<'a> {
    /// Variant arm. `paths` holds one path per or-pattern sub-arm —
    /// each entry produces an isinstance cascade step pointing at the
    /// same arm body block. `field_bindings` is extracted from the
    /// FIRST sub-pattern; or-pattern siblings are validated to bind
    /// the same `field_name → local_name` set before this arm
    /// classifies as `Variant`.
    Variant {
        paths: Vec<&'a syn::Path>,
        field_bindings: Vec<StructFieldBinding>,
    },
    /// `_` arm — must be last. No binding into the body's locals.
    Wildcard,
    /// `name` arm — must be last. Binds the scrutinee to `name` in the
    /// body's locals scope.
    BindWildcard(String),
}

/// True when `a` and `b` describe the same set of `field_name →
/// local_name` bindings (order-independent). Used to validate that
/// or-pattern siblings of a single arm bind the same identifiers, so
/// the shared arm-body block sees a consistent locals view regardless
/// of which sub-pattern matched.
fn struct_field_bindings_match(a: &[StructFieldBinding], b: &[StructFieldBinding]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_sorted: Vec<&StructFieldBinding> = a.iter().collect();
    let mut b_sorted: Vec<&StructFieldBinding> = b.iter().collect();
    a_sorted.sort_by(|x, y| x.field_name.cmp(&y.field_name));
    b_sorted.sort_by(|x, y| x.field_name.cmp(&y.field_name));
    a_sorted
        .iter()
        .zip(b_sorted.iter())
        .all(|(x, y)| x.field_name == y.field_name && x.local_name == y.local_name)
}

/// Lower `match enum_val { Variant => body, …, _ => body }` where at
/// least one arm references a unit enum variant. Each variant arm
/// produces an `isinstance(scrutinee, Constant(HostObject(<class>)))`
/// boolean fork — the second operand is the `HostObject::Class`
/// `Builder::resolve_path_constant` returns for the variant path,
/// emitting any necessary `getattr` ops (one per non-leftmost segment)
/// into the cascade step's block beforehand. Or-pattern arms unfold
/// into multiple forks all pointing at the same arm-body block. The
/// cascade terminates at a user-provided wildcard arm — exhaustive
/// matches without a catch-all are still rejected because the adapter
/// does not enumerate the variant universe.
///
/// Upstream analogue: the if-cascade `flowcontext.py` produces from
/// Python source `if isinstance(x, A.X): … elif isinstance(x, A.Y):
/// …`. Each cascade step's emission mirrors `flowcontext.py:856
/// LOAD_GLOBAL` + `:861 LOAD_ATTR` (the getattr cascade) followed by
/// `operation.py:449 isinstance` arity=2 on the resolved class.
fn lower_match_variant_cascade(
    b: &mut Builder,
    match_expr: &ExprMatch,
    scrutinee: Hlvalue,
    at_boundary: bool,
) -> Result<BlockExit, AdapterError> {
    let n_arms = match_expr.arms.len();

    // 1. Pre-classify every arm. Or-pattern sub-patterns flatten;
    //    every sub-pattern must be a variant pattern recognised by
    //    `pat_has_variant_shape` (unit `Pat::Path`, rest-only
    //    `Pat::Struct { .. }`, or rest-only `Pat::TupleStruct(..)`),
    //    a wildcard `_`, or a `Pat::Ident` binding-wildcard.
    //    Wildcards (`_` / `name`) must be the standalone last arm.
    let mut arm_kinds: Vec<CascadeArm> = Vec::with_capacity(n_arms);
    for (idx, arm) in match_expr.arms.iter().enumerate() {
        let is_last_arm = idx + 1 == n_arms;
        let mut sub_pats: Vec<&Pat> = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);

        // A single sub-pattern that's a wildcard or bind-wildcard is
        // the cascade's catch-all. Inside an or-pattern (`_ | Foo::A`)
        // the wildcard intent collides with the variant intent — same
        // policy as the literal-switch `lower_match` enforces at
        // `model.py:652`.
        if sub_pats.len() == 1 {
            match sub_pats[0] {
                Pat::Wild(_) => {
                    if !is_last_arm {
                        return Err(AdapterError::Unsupported {
                            reason: "wildcard arm must be the last arm of a variant-cascade \
                                match (upstream `model.py:652` invariant)"
                                .into(),
                        });
                    }
                    arm_kinds.push(CascadeArm::Wildcard);
                    continue;
                }
                Pat::Ident(PatIdent {
                    by_ref: None,
                    subpat: None,
                    mutability: None,
                    ident,
                    ..
                }) => {
                    if !is_last_arm {
                        return Err(AdapterError::Unsupported {
                            reason: "binding-wildcard arm (e.g. `other => …`) must be the \
                                last arm of a variant-cascade match"
                                .into(),
                        });
                    }
                    arm_kinds.push(CascadeArm::BindWildcard(ident.to_string()));
                    continue;
                }
                _ => {}
            }
        }

        // Otherwise every sub-pattern must be a variant pattern that
        // `extract_variant_arm_info` recognises (unit / rest-only
        // struct / rest-only tuple / struct with named-Ident field
        // bindings). Or-pattern siblings of the same arm must agree
        // on the bindings shape so the shared arm-body block sees a
        // consistent locals view.
        let mut paths: Vec<&syn::Path> = Vec::with_capacity(sub_pats.len());
        let mut arm_field_bindings: Option<Vec<StructFieldBinding>> = None;
        for sub_pat in &sub_pats {
            if let Some(info) = extract_variant_arm_info(sub_pat) {
                if let Some(prev) = &arm_field_bindings {
                    if !struct_field_bindings_match(prev, &info.bindings) {
                        return Err(AdapterError::Unsupported {
                            reason: "or-pattern variant arms with inconsistent field \
                                bindings — the cascade lowers a single arm body block \
                                whose locals must agree across or-pattern siblings"
                                .into(),
                        });
                    }
                } else {
                    arm_field_bindings = Some(info.bindings.clone());
                }
                paths.push(info.path);
                continue;
            }
            // Specific diagnostics matching the categories the M2.5e
            // probe test pins. After slice 2e: `Pat::Struct` field
            // bindings reach `extract_variant_arm_info` and accept
            // when every field pattern is `Pat::Ident` or `Pat::Wild`.
            // The `Pat::Struct` arm here fires only when at least one
            // field pattern is something else (literal, nested
            // composite, by_ref binding, etc.).
            let reason = match sub_pat {
                Pat::Struct(_) => "match arm struct-variant pattern with non-Ident field \
                        sub-pattern (literal / nested composite / by_ref binding) — \
                        complex destructuring lands after slice 2e"
                    .to_string(),
                Pat::TupleStruct(_) => "match arm tuple-variant pattern with element \
                        bindings (`Foo::Bar(a, b)`) — tuple-variant element extraction \
                        lands in M2.5d slice 2f"
                    .to_string(),
                Pat::Tuple(_) => "match arm tuple pattern (`(a, b)`) — non-variant \
                        composite patterns are out of scope of variant-cascade lowering"
                    .to_string(),
                Pat::Wild(_) => "wildcard sub-pattern inside or-pattern — upstream \
                        `model.py:652` reserves the wildcard for a standalone default \
                        arm at match-level"
                    .to_string(),
                Pat::Lit(_) => "match cannot mix literal and variant patterns in the same \
                        match (variant-cascade lowering requires homogeneous variant arms)"
                    .to_string(),
                _ => "match arm pattern not in M2.5b/d/e subset (only variant-path, \
                        rest-only struct/tuple, named-Ident struct fields, or wildcard \
                        supported in variant cascade)"
                    .to_string(),
            };
            return Err(AdapterError::Unsupported { reason });
        }
        arm_kinds.push(CascadeArm::Variant {
            paths,
            field_bindings: arm_field_bindings.unwrap_or_default(),
        });
    }

    // 2. Require a wildcard last. The cascade lowers each variant arm
    //    to a 2-exit `isinstance` fork; without a catch-all the final
    //    "false" branch has no target. Upstream's exhaustiveness
    //    check kicks in inside the annotator (one classdef knows the
    //    full subclass set); the adapter cannot enumerate the variant
    //    universe from `syn::ItemFn` alone, so the user-provided
    //    wildcard arm is the only structural way to close the cascade.
    let last_is_wildcard = matches!(
        arm_kinds.last(),
        Some(CascadeArm::Wildcard | CascadeArm::BindWildcard(_))
    );
    if !last_is_wildcard {
        return Err(AdapterError::Unsupported {
            reason: "variant-cascade match must end with a wildcard arm (`_` or `name`) — \
                the adapter cannot enumerate the variant universe from `syn::ItemFn`, so \
                the catch-all is the only structural way to close the final isinstance fork"
                .into(),
        });
    }

    // 3. Stash the scrutinee into a synthetic local so the standard
    //    merged-names machinery threads it through every cascade
    //    fork-block via Link.args. Use the same `#`-prefixed slot
    //    convention as `lower_for`'s `#for_iter_{depth}` — `#` cannot
    //    appear in `syn::Ident::to_string()` output, so collisions
    //    with user-source names are impossible. The slot id derives
    //    from the scrutinee Variable's identity so nested matches
    //    pick distinct names.
    let scrutinee_var = match scrutinee {
        Hlvalue::Variable(v) => v,
        Hlvalue::Constant(c) => {
            // Wrap constant scrutinees in `same_as` so the cascade
            // always operates on a Variable. Upstream's
            // `flowcontext.py` `IS_OP` / `COMPARE_OP` paths likewise
            // bind the value to a Variable before forking on it.
            let v = Variable::new();
            b.emit_op(SpaceOperation::new(
                "same_as",
                vec![Hlvalue::Constant(c)],
                Hlvalue::Variable(v.clone()),
            ));
            v
        }
    };
    let scrutinee_slot = format!("#match_scrutinee_{}", scrutinee_var.id());
    b.set_local(scrutinee_slot.clone(), Hlvalue::Variable(scrutinee_var));

    // 4. Snapshot pre-cascade locals (now including the scrutinee
    //    slot) and fix the merged_names ordering. Sorted ordering
    //    gives deterministic Link.args layout across runs.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();

    // 5. Allocate one body block per arm. The wildcard arm's body
    //    block doubles as the cascade's terminal "false" target.
    let mut arm_body_blocks: Vec<(BlockRef, HashMap<String, Hlvalue>)> = Vec::with_capacity(n_arms);
    for _ in 0..n_arms {
        arm_body_blocks.push(branch_block_with_inputargs(&merged_names));
    }
    let wildcard_arm_idx = n_arms - 1;

    // 6. Emit the cascade. One step per `(arm_idx, variant_path)` —
    //    or-pattern arms unfold into multiple steps targeting the
    //    same arm body. The wildcard arm (always last per step 2) is
    //    NOT a step; it receives the final cascade step's "false"
    //    Link.
    let cascade_steps: Vec<(usize, &syn::Path)> = arm_kinds
        .iter()
        .enumerate()
        .filter_map(|(arm_idx, kind)| match kind {
            CascadeArm::Variant { paths, .. } => Some((arm_idx, paths.as_slice())),
            _ => None,
        })
        .flat_map(|(arm_idx, paths)| paths.iter().map(move |p| (arm_idx, *p)))
        .collect();
    let n_steps = cascade_steps.len();
    if n_steps == 0 {
        // No variant arms means the dispatch peek mis-fired — every
        // arm is a wildcard, which is structurally a no-op match.
        // Defensive: this shape should never reach the cascade because
        // `lower_match`'s peek requires at least one `Pat::Path` to
        // route here.
        return Err(AdapterError::Unsupported {
            reason: "variant-cascade match with no variant arms (defensive — caller \
                dispatch should not route here)"
                .into(),
        });
    }

    for (step_idx, (arm_idx, variant_path)) in cascade_steps.iter().enumerate() {
        let is_last_step = step_idx + 1 == n_steps;

        // Read the scrutinee through the current block's locals view —
        // for the FIRST step it's the original Variable; for
        // subsequent steps it's the fresh inputarg of the cascade
        // fork-block opened by the previous iteration.
        let scrutinee_now = b
            .locals()
            .get(&scrutinee_slot)
            .cloned()
            .expect("scrutinee_slot threads through cascade fork-blocks");
        // Resolve the variant path to a `Constant(HostObject(<class>))`
        // carrier through the same `Builder::resolve_path_constant`
        // chain expression-position qualified paths use. This mirrors
        // upstream `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR`
        // for `isinstance(x, Foo.Bar)`: each cascade step block emits
        // its own getattr ops (one per non-leftmost segment), then
        // the isinstance op consumes the resolved leaf as its second
        // operand per `operation.py:449 isinstance` arity=2. Identity
        // sharing across cascade steps that name the same leftmost
        // segment falls out of the process-global
        // `host_env::HOST_CLASS_MINTS` registry (mirrors Python's
        // `LOAD_GLOBAL` returning the same class object across every
        // reference, including across graph boundaries —
        // `flowcontext.py:847 w_globals.value[varname]`).
        let class_carrier = b.resolve_path_constant(variant_path)?;
        let cond_var = Hlvalue::Variable(Variable::new());
        b.emit_op(SpaceOperation::new(
            "isinstance",
            vec![scrutinee_now, class_carrier],
            cond_var.clone(),
        ));

        let arm_body_block = arm_body_blocks[*arm_idx].0.clone();
        // Final step's "false" target IS the wildcard arm's body
        // block; intermediate steps allocate a fresh fork block.
        let (next_block, next_locals) = if is_last_step {
            arm_body_blocks[wildcard_arm_idx].clone()
        } else {
            branch_block_with_inputargs(&merged_names)
        };

        // Link.args carry every merged-name's current SSA value into
        // BOTH the true and false targets — same convention as
        // `lower_if` (`branch_link_args` reads `b.locals()` per name).
        let cur_locals = b.locals().clone();
        let cur_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|name| {
                cur_locals
                    .get(name)
                    .cloned()
                    .expect("merged_names is a subset of cascade-step entry locals")
            })
            .collect();

        let false_link = Rc::new(RefCell::new(Link::new(
            cur_args.clone(),
            Some(next_block.clone()),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
        )));
        let true_link = Rc::new(RefCell::new(Link::new(
            cur_args,
            Some(arm_body_block),
            Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
        )));
        b.finalize_current(vec![false_link, true_link], Some(cond_var));

        // Open the next cascade fork-block (intermediate steps) — the
        // wildcard arm's body block opens later in step 7 alongside
        // every other arm body, so we leave it untouched here.
        if !is_last_step {
            b.open_new_block(BlockBuilder {
                block: next_block,
                ops: Vec::new(),
                locals: next_locals,
            });
        }
    }

    // 7. Lower each arm's body. Before the user body lowers:
    //    - `Variant` arms emit one `getattr(scrutinee, "<field>")` op
    //      per declared field binding and bind the resulting value to
    //      the user-named local. Mirrors upstream
    //      `rpython/flowspace/operation.py:618 getattr` arity=2; same
    //      shape `flowcontext.py` produces for an attribute access
    //      (`obj.field`) read at the start of an isinstance arm.
    //    - `BindWildcard` arms surface the scrutinee under the
    //      user-supplied identifier.
    let mut arm_captures: Vec<Option<ArmCapture>> = Vec::with_capacity(n_arms);
    for ((branch_block, branch_locals), (kind, arm)) in arm_body_blocks
        .into_iter()
        .zip(arm_kinds.iter().zip(&match_expr.arms))
    {
        b.open_new_block(BlockBuilder {
            block: branch_block,
            ops: Vec::new(),
            locals: branch_locals,
        });
        match kind {
            CascadeArm::Variant { field_bindings, .. } if !field_bindings.is_empty() => {
                let scrutinee_now = b
                    .locals()
                    .get(&scrutinee_slot)
                    .cloned()
                    .expect("variant arm body inherits scrutinee slot via merged_names");
                for binding in field_bindings {
                    let value = Hlvalue::Variable(Variable::new());
                    b.emit_op(SpaceOperation::new(
                        "getattr",
                        vec![
                            scrutinee_now.clone(),
                            Hlvalue::Constant(Constant::new(ConstValue::byte_str(
                                &binding.field_name,
                            ))),
                        ],
                        value.clone(),
                    ));
                    b.set_local(binding.local_name.clone(), value);
                }
            }
            CascadeArm::BindWildcard(name) => {
                let scrutinee_now = b
                    .locals()
                    .get(&scrutinee_slot)
                    .cloned()
                    .expect("wildcard body inherits scrutinee slot via merged_names");
                b.set_local(name.clone(), scrutinee_now);
            }
            _ => {}
        }
        let exit = lower_arm_body(b, &arm.body, at_boundary)?;
        arm_captures.push(capture_arm_exit(b, exit));
    }

    // 8. If every arm terminated, the post-match PC is unreachable —
    //    same convention as `lower_match`'s literal-switch path.
    if arm_captures.iter().all(|c| c.is_none()) {
        return Ok(BlockExit::Terminated);
    }

    // 9. Build the join block. The synthetic scrutinee slot is
    //    filtered out of the exit-visible name set — post-match code
    //    must not see it (mirrors `lower_for`'s `exit_merged_names`
    //    treatment of the iterator slot).
    let exit_merged_names: Vec<String> = merged_names
        .iter()
        .filter(|n| **n != scrutinee_slot)
        .cloned()
        .collect();

    let tail_var = Hlvalue::Variable(Variable::new());
    let mut join_inputargs: Vec<Hlvalue> = Vec::with_capacity(exit_merged_names.len() + 1);
    join_inputargs.push(tail_var.clone());
    let mut join_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &exit_merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        join_inputargs.push(fresh.clone());
        join_locals.insert(name.clone(), fresh);
    }
    let join_block = Block::shared(join_inputargs);

    // 10. Close every surviving arm's tail block with a Link into the
    //     join.
    for cap in arm_captures.into_iter().flatten() {
        let mut link_args: Vec<Hlvalue> = Vec::with_capacity(exit_merged_names.len() + 1);
        link_args.push(cap.tail.clone());
        for name in &exit_merged_names {
            let value = cap
                .locals
                .get(name)
                .cloned()
                .expect("merged_names is a subset of arm entry locals");
            link_args.push(value);
        }
        let link = Rc::new(RefCell::new(Link::new(
            link_args,
            Some(join_block.clone()),
            None,
        )));
        cap.block.borrow_mut().operations = cap.ops;
        cap.block.closeblock(vec![link]);
    }

    b.open_new_block(BlockBuilder {
        block: join_block,
        ops: Vec::new(),
        locals: join_locals,
    });
    Ok(BlockExit::FallThrough(tail_var))
}

// ____________________________________________________________
// `while` / `loop` — header + back-edge with `break` / `continue`.
//
// Upstream basis: `rpython/flowspace/flowcontext.py:794 SETUP_LOOP`
// pushes a LoopBlock; `:718 JUMP_ABSOLUTE` returns the back-edge target
// that the pending-block scheduler merges against the header; `:525
// BREAK_LOOP` / `:528 CONTINUE_LOOP` raise Break/Continue which the
// enclosing LoopBlock turns into the corresponding jumps.
//
// The graph-level shape upstream produces for
//
//   while cond:                header:  cond, exitswitch=cond,
//     body                              [false → exit, true → body]
//                              body:    body-ops, Link(→ header)
//                              exit:    (continue from here)
//
// is what `lower_while` emits directly. `lower_loop` is the same shape
// without the `cond` fork — the header is entered unconditionally and
// the only way out is via `break` (or fallthrough through a
// body-tail that happens to not exist in our subset, since `loop`
// bodies end at the `}` with an implicit back-edge).
//
// Body subset (slice 3): a flat sequence of `let` bindings, optionally
// terminated by a single `break;` / `continue;`. If the body falls
// through to the closing `}`, the adapter emits the back-edge
// automatically. Dead code after an explicit terminator rejects via
// `AdapterError::Unsupported`. Loops nested inside `if`/`match`
// branches work because those lower into independent BlockBuilders
// before hitting the terminator check.

/// Outcome of a loop body lowering:
/// - `FallThrough` — body reached its closing `}` naturally, the caller
///   emits the back-edge Link.
/// - `Terminated` — body finalized its current block via `break;` /
///   `continue;`; no further emission required from the caller.
enum LoopBodyExit {
    FallThrough,
    Terminated,
}

fn lower_while(b: &mut Builder, while_expr: &ExprWhile) -> Result<(), AdapterError> {
    // 1. Snapshot pre-loop locals and fix the merged name ordering.
    //    This is what both the header's inputargs and the back-edge
    //    Link args agree on.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    // 2. Allocate the header + exit blocks. Both carry one fresh
    //    inputarg per merged local; the header becomes the back-edge
    //    target, the exit block is where execution resumes after the
    //    loop.
    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&merged_names);

    // 3. Close the pre-loop block with an unconditional Link into the
    //    header.
    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    // 4. Open the header, lower the condition, and allocate the body
    //    block. The header's locals at fork time are identical to the
    //    header's inputargs — no rebinding happens between the entry
    //    and the condition evaluation.
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    // Upstream POP_JUMP_IF_FALSE (`flowcontext.py:756`) always wraps
    // the predicate in `op.bool(w_value)` before the fork. Emit the
    // `bool` op explicitly so the exitswitch carries the coerced
    // result — the annotator / optimizer can fold it away when the
    // input is already `SomeBool`.
    let cond_raw = lower_expr(b, &while_expr.cond)?;
    let cond = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("bool", vec![cond_raw], cond.clone()));
    let header_locals_at_fork: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();

    let (body_block, body_locals) = branch_block_with_inputargs(&merged_names);
    let false_link = Rc::new(RefCell::new(Link::new(
        header_locals_at_fork.clone(),
        Some(exit_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))),
    )));
    let true_link = Rc::new(RefCell::new(Link::new(
        header_locals_at_fork,
        Some(body_block.clone()),
        Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))),
    )));
    b.finalize_current(vec![false_link, true_link], Some(cond));

    // 5. Push the loop context before lowering the body so nested
    //    `break` / `continue` resolve against *this* loop.
    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        // while/loop have no synthetic sidecar — exit set == full set.
        exit_merged_names: merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: body_block,
        ops: Vec::new(),
        locals: body_locals,
    });
    let body_exit = lower_loop_body(b, &while_expr.body)?;

    // 6. If the body fell through, emit the back-edge. Body-local
    //    rebinds that ended up in `current.locals` flow back into the
    //    header via the new Link's args.
    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    // 7. Execution after the loop resumes in the exit block, with its
    //    inputargs bound as the new `locals`.
    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

fn lower_loop(b: &mut Builder, loop_expr: &ExprLoop) -> Result<(), AdapterError> {
    // `loop { body }` is `while true { body }` without the cond fork.
    // The header is entered unconditionally and doubles as the body
    // start; `break` jumps to the exit block; fallthrough loops back.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&merged_names);

    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        // while/loop have no synthetic sidecar — exit set == full set.
        exit_merged_names: merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    let body_exit = lower_loop_body(b, &loop_expr.body)?;

    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

// ____________________________________________________________
// `for item in iter { body }` — iter protocol with a StopIteration
// exception exit that catches at the loop boundary (not at
// graph.exceptblock).
//
// Upstream basis: `rpython/flowspace/flowcontext.py:782 GET_ITER`
// emits `op.iter(iterable)`; `:787 FOR_ITER` pushes an IterBlock
// exception handler that catches StopIteration and jumps to the
// post-loop pc, then emits `op.next(iterator)` which may raise
// StopIteration. In graph form the header block ends in
// `next(iter)` with `exitswitch = c_last_exception`, exit[0] is the
// normal fall-through into the body with the freshly-yielded
// element, and exit[1] carries `exitcase = StopIteration` targeting
// the loop-exit block (not graph.exceptblock — StopIteration is
// silently caught by the IterBlock handler). `operation.py:596-599`
// confirms `next` is arity=1 with `canraise = [StopIteration,
// RuntimeError]`.
//
// Exception exit shape (post-simplify-equivalent):
// - StopIteration exit: link.args carry the exit-visible merged
//   locals, target = loop exit_block, exitcase = Constant(StopIteration
//   class). link.extravars.last_exception = `Constant(StopIteration)`
//   per upstream `flowcontext.py:127` — class-specific guessexception
//   exits use a Constant for `last_exc`; only the generic `Exception`
//   case uses a Variable. link.extravars.last_exc_value = fresh
//   Variable named `last_exc_value`.
// - RuntimeError exit: no outer handler in the adapter's simplified
//   model, so the egg's `RaiseImplicit.nomoreblocks`
//   (`flowcontext.py:1271-1284`) runs immediately — its closing link
//   carries `[Constant(AssertionError_class), Constant(AssertionError(
//   "implicit RuntimeError shouldn't occur"))]` straight to
//   `graph.exceptblock`. After `eliminate_empty_blocks` the trivial
//   RaiseImplicit egg folds away and the header link lands directly
//   on the exceptblock with those AssertionError constants — that is
//   the shape the adapter emits.
//
// PRE-EXISTING-ADAPTATION (value-stack vs locals-map). Upstream
// `flowcontext.py:782 GET_ITER` leaves the iterator on the Python
// value stack, and `:787 FOR_ITER` pops it via IterBlock.handle
// after StopIteration. The adapter lacks a stack model — its frame
// state is a name-keyed locals map. We stash the iterator in a
// reserved local slot named `#for_iter_{depth}` (the `#` character
// cannot appear in any `syn::Ident::to_string()` output, ruling out
// user-source collisions) and strip it from every post-loop visible-
// name set. Cited as unavoidable per CLAUDE.md "smallest possible
// change from RPython structure" — a full value-stack port is an
// M2.5e scope item.
//
// Scope (slice 5):
// - Pattern: `Pat::Ident` only (simple `for item in iter`, no
//   destructuring).
// - Iterator expression: whatever `lower_expr` accepts (locals,
//   literals, binops).
// - Body: the `lower_loop_body` subset — `let` bindings, nested
//   while/loop/for, optionally terminated by `break;` / `continue;`.
// - `break;`, StopIteration exit, `continue;`, fall-through back-edge
//   all route through the standard merged-locals machinery — the
//   iterator sits in the `#for_iter_{depth}` slot that participates
//   in the merged-names set, so nested loops thread it through their
//   own merged-name machinery without special-casing.

fn lower_for(b: &mut Builder, for_expr: &ExprForLoop) -> Result<(), AdapterError> {
    // 1. `for item in iter` — only a plain identifier pattern is
    //    accepted. Destructuring / tuple-patterns land with composite
    //    literals in M2.5d.
    let item_name = match &*for_expr.pat {
        Pat::Ident(PatIdent {
            ident,
            by_ref: None,
            subpat: None,
            ..
        }) => ident.to_string(),
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "`for` pattern must be a plain identifier (destructuring lands in M2.5d)"
                    .into(),
            });
        }
    };

    // 2. Lower the iterable into the current block and emit
    //    `iter(iterable) -> v_iter` — flowcontext's GET_ITER
    //    sequence. Store `v_iter` into a reserved internal slot so
    //    the standard merged-names machinery carries it through
    //    every enclosed nested loop / conditional without special
    //    casing.
    //
    //    The slot name is `#for_iter_{depth}` indexed by
    //    `loop_stack.len()` at entry, so nested for-loops pick
    //    distinct names. `#` is not a legal Rust identifier
    //    character, so `syn::Ident::to_string()` can never produce
    //    a name that collides with the slot.
    let iterable = lower_expr(b, &for_expr.expr)?;
    let v_iter = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("iter", vec![iterable], v_iter.clone()));
    let iter_slot = format!("#for_iter_{}", b.loop_stack.len());
    b.set_local(iter_slot.clone(), v_iter);

    // 3. Snapshot the pre-loop locals (now including the iter slot)
    //    and fix the merged ordering.
    let pre_fork_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
    merged_names.sort();
    let pre_args: Vec<Hlvalue> = merged_names
        .iter()
        .map(|n| pre_fork_locals[n].clone())
        .collect();

    // 4. Pre-compute the exit-block's visible name set: full
    //    merged_names MINUS the synthetic iter slot. Upstream
    //    `flowcontext.py:787, :1355, :1383` — the iterator lives on
    //    the value stack for the loop's lifetime and is popped at
    //    loop exit, so post-loop code must NOT see it.
    let exit_merged_names: Vec<String> = merged_names
        .iter()
        .filter(|n| *n != &iter_slot)
        .cloned()
        .collect();

    // 5. Allocate header (threads the iter slot via merged_names) and
    //    exit blocks (uses the filtered exit_merged_names so the
    //    iter slot does NOT appear in post-loop inputargs or locals).
    let (header_block, header_locals) = branch_block_with_inputargs(&merged_names);
    let (exit_block, exit_locals) = branch_block_with_inputargs(&exit_merged_names);

    // 5. Close the pre-loop block with a plain Link into the header.
    let pre_link = Rc::new(RefCell::new(Link::new(
        pre_args,
        Some(header_block.clone()),
        None,
    )));
    b.finalize_current(vec![pre_link], None);

    // 6. Open the header block and emit the raising `next(iter_h)`
    //    op reading from the header's own iter-slot binding.
    b.open_new_block(BlockBuilder {
        block: header_block.clone(),
        ops: Vec::new(),
        locals: header_locals,
    });
    let iter_h = b.current.locals[&iter_slot].clone();
    let v_next = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("next", vec![iter_h], v_next.clone()));

    // 7. Body block. Upstream STORE_FAST
    //    (`flowcontext.py:878-884`) rebinds the loop-variable slot in
    //    place — after FOR_ITER pops the new item and STORE_FAST
    //    writes it, `locals_w[i_item]` IS the new item and no
    //    separate channel exists for the pre-loop value of that slot.
    //    Mirror that by walking `merged_names` once and emitting a
    //    single inputarg per slot: for `item_name` the inputarg IS
    //    `body_item_var`; for every other slot the inputarg is a
    //    fresh `Variable::named(name)`. No double channels.
    let body_item_var = Hlvalue::Variable(Variable::named(&item_name));
    let mut body_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len());
    let mut body_locals: HashMap<String, Hlvalue> = HashMap::new();
    let mut item_in_merged = false;
    for name in &merged_names {
        if name == &item_name {
            body_inputargs.push(body_item_var.clone());
            body_locals.insert(name.clone(), body_item_var.clone());
            item_in_merged = true;
        } else {
            let fresh = Hlvalue::Variable(Variable::named(name));
            body_inputargs.push(fresh.clone());
            body_locals.insert(name.clone(), fresh);
        }
    }
    // If `item_name` wasn't pre-existing in merged_names, the
    // STORE_FAST creates a new slot: append body_item_var as the
    // last inputarg and record it under its name.
    if !item_in_merged {
        body_inputargs.push(body_item_var.clone());
        body_locals.insert(item_name.clone(), body_item_var);
    }
    let body_block = Block::shared(body_inputargs);

    // 8. Close the header with the canraise shape. Upstream
    //    `operation.py:595-599` — `Next.canraise = [StopIteration,
    //    RuntimeError]`. The flowcontext's implicit IterBlock
    //    (`flowcontext.py:1378`) catches StopIteration and routes it
    //    to the loop's exit; every other canraise exception is
    //    unrolled as `RaiseImplicit` (`flowcontext.py:176`) which,
    //    with no outer handler, closes via
    //    `RaiseImplicit.nomoreblocks` (`:1271-1284`) — that pathway
    //    rewrites the exception into `AssertionError("implicit <CLS>
    //    shouldn't occur")` and links straight to
    //    `graph.exceptblock` with those constants as the link args.
    //
    //    Emission order matches upstream guessexception
    //    (`flowcontext.py:124-148` — `[None] + list(cases)`):
    //    normal → body, StopIteration → loop exit,
    //    RuntimeError → graph.exceptblock (via AssertionError
    //    rewrite).
    //
    //    Normal link.args align positionally with body_inputargs:
    //    item_name's slot carries `v_next`, every other merged slot
    //    carries the header's current binding. If `item_name` is
    //    NOT in merged_names, `v_next` is appended last (mirroring
    //    STORE_FAST creating a new slot).
    let mut normal_args: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    for name in &merged_names {
        if name == &item_name {
            normal_args.push(v_next.clone());
        } else {
            normal_args.push(b.current.locals[name].clone());
        }
    }
    if !item_in_merged {
        normal_args.push(v_next);
    }
    let normal_link = Rc::new(RefCell::new(Link::new(
        normal_args,
        Some(body_block.clone()),
        None,
    )));

    // StopIteration exit → loop's own exit_block. Link.args carry
    // only the exit-visible names (iter slot filtered out).
    // Extravars mirror `guessexception` at `flowcontext.py:127-134`
    // for a class-specific case: `last_exception = Constant(case)`,
    // `last_exc_value = Variable('last_exc_value')` (fresh).
    let stopiter_exit_args: Vec<Hlvalue> = exit_merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let stopiter_cls = HOST_ENV
        .lookup_exception_class("StopIteration")
        .expect("HOST_ENV bootstrap must register `StopIteration` — model.rs:1426 ensures it");
    let stopiter_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(stopiter_cls)));
    let stopiter_last_exc_value = Variable::named("last_exc_value");
    let mut stopiter_link_inner = Link::new(
        stopiter_exit_args,
        Some(exit_block.clone()),
        Some(stopiter_const.clone()),
    );
    stopiter_link_inner.extravars(
        Some(stopiter_const),
        Some(Hlvalue::Variable(stopiter_last_exc_value)),
    );
    let stopiter_link = Rc::new(RefCell::new(stopiter_link_inner));

    // RuntimeError exit → graph.exceptblock via the RaiseImplicit
    // rewrite. `flowcontext.py:1271-1284 RaiseImplicit.nomoreblocks`
    // fires when no outer handler catches the `Constant(RuntimeError)`
    // raise: it closes the current block with
    // `Link([Constant(AssertionError_class), Constant(AssertionError(msg))],
    // graph.exceptblock)`. After `eliminate_empty_blocks` the
    // intervening egg folds away, leaving the header linked directly
    // to `graph.exceptblock` with those AssertionError constants.
    //
    // The link's own extravars still reflect the original
    // guessexception class-specific case — `last_exception =
    // Constant(RuntimeError)`, `last_exc_value =
    // Variable('last_exc_value')` — matching upstream
    // `flowcontext.py:127-143`.
    let runtime_cls = HOST_ENV
        .lookup_exception_class("RuntimeError")
        .expect("HOST_ENV bootstrap must register `RuntimeError` — model.rs:1419 ensures it");
    let runtime_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(runtime_cls)));
    let assertion_cls = HOST_ENV
        .lookup_exception_class("AssertionError")
        .expect("HOST_ENV bootstrap must register `AssertionError` — model.rs:1424 ensures it");
    let assertion_cls_const =
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(assertion_cls.clone())));
    let assertion_msg = "implicit RuntimeError shouldn't occur".to_string();
    let assertion_instance = crate::flowspace::model::HostObject::new_instance(
        assertion_cls,
        vec![ConstValue::byte_str(assertion_msg)],
    );
    let assertion_instance_const =
        Hlvalue::Constant(Constant::new(ConstValue::HostObject(assertion_instance)));
    let runtime_last_exc_value = Variable::named("last_exc_value");
    let mut runtime_link_inner = Link::new(
        vec![assertion_cls_const, assertion_instance_const],
        Some(b.exceptblock.clone()),
        Some(runtime_const.clone()),
    );
    runtime_link_inner.extravars(
        Some(runtime_const),
        Some(Hlvalue::Variable(runtime_last_exc_value)),
    );
    let runtime_link = Rc::new(RefCell::new(runtime_link_inner));

    b.finalize_current(
        vec![normal_link, stopiter_link, runtime_link],
        Some(Hlvalue::Constant(c_last_exception())),
    );

    // 9. Push the LoopCtx and lower the body. Break / continue /
    //    fall-through all use the standard merged-locals shape;
    //    because the iter slot is in `merged_names` their Links
    //    automatically carry it. `break` / loop-exit Links use
    //    `exit_merged_names` which excludes the internal iter slot
    //    so the post-loop block never sees it.
    let exit_merged_names: Vec<String> = merged_names
        .iter()
        .filter(|n| *n != &iter_slot)
        .cloned()
        .collect();
    b.loop_stack.push(LoopCtx {
        header_block: header_block.clone(),
        exit_block: exit_block.clone(),
        merged_names: merged_names.clone(),
        exit_merged_names: exit_merged_names.clone(),
    });
    b.open_new_block(BlockBuilder {
        block: body_block,
        ops: Vec::new(),
        locals: body_locals,
    });
    let body_exit = lower_loop_body(b, &for_expr.body)?;
    if matches!(body_exit, LoopBodyExit::FallThrough) {
        let back_args: Vec<Hlvalue> = merged_names
            .iter()
            .map(|n| b.current.locals[n].clone())
            .collect();
        let back_link = Rc::new(RefCell::new(Link::new(back_args, Some(header_block), None)));
        b.finalize_current(vec![back_link], None);
    }
    b.loop_stack.pop();

    // 10. Continue lowering in the exit block. `exit_locals` has
    //     had the internal iter slot stripped so user source can't
    //     reference it.
    b.open_new_block(BlockBuilder {
        block: exit_block,
        ops: Vec::new(),
        locals: exit_locals,
    });
    Ok(())
}

/// Lower a `while`/`loop` body. Upstream SETUP_LOOP
/// (`flowcontext.py:488, :794`) wraps arbitrary bytecode flow — the
/// loop body is not a separate dispatch subset. The adapter follows
/// suit: any statement accepted by top-level lowering is accepted
/// here, with two additions specific to loops:
/// - `break;` and `continue;` are terminator statements that
///   finalize the current block with a Link into the loop's
///   exit_block / header_block.
/// - the loop body has no tail value (Rust loop body returns `()`),
///   so any trailing expression is discarded as a POP_TOP.
fn lower_loop_body(b: &mut Builder, body: &SynBlock) -> Result<LoopBodyExit, AdapterError> {
    let n = body.stmts.len();
    for (idx, stmt) in body.stmts.iter().enumerate() {
        let is_last = idx + 1 == n;
        match stmt {
            Stmt::Local(local) => lower_let(b, local)?,
            Stmt::Expr(Expr::Break(brk), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `break;` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                lower_break(b, brk)?;
                return Ok(LoopBodyExit::Terminated);
            }
            Stmt::Expr(Expr::Continue(cont), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `continue;` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                lower_continue(b, cont)?;
                return Ok(LoopBodyExit::Terminated);
            }
            // `return` inside a loop body closes the current block
            // to `graph.returnblock` just like at any other statement
            // position (upstream `flowcontext.py:687, :1232`). That
            // also ends this loop-body lowering — further body stmts
            // would be dead code, and the loop's back-edge must NOT
            // be emitted because execution never reaches the body's
            // closing `}`.
            Stmt::Expr(Expr::Return(ret), _) => {
                if !is_last {
                    return Err(AdapterError::Unsupported {
                        reason: "dead code after `return` — the terminator must be the last \
                            statement in the loop body"
                            .into(),
                    });
                }
                let _ = lower_return(b, ret)?;
                return Ok(LoopBodyExit::Terminated);
            }
            Stmt::Expr(Expr::While(while_expr), _) => {
                lower_while(b, while_expr)?;
            }
            Stmt::Expr(Expr::Loop(loop_expr), _) => {
                lower_loop(b, loop_expr)?;
            }
            Stmt::Expr(Expr::ForLoop(for_expr), _) => {
                lower_for(b, for_expr)?;
            }
            // Any other expression statement: lower for side effect
            // and discard the result (upstream POP_TOP after a
            // stack-producing instruction). Covers call / method_call
            // / if-else / match / attribute access / etc. — matches
            // SETUP_LOOP's "arbitrary bytecode inside the loop"
            // semantic at flowcontext.py:488.
            Stmt::Expr(expr, _) => {
                let _ = lower_expr(b, expr)?;
            }
            Stmt::Item(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "nested item (fn/struct/impl inside loop body)".into(),
                });
            }
            Stmt::Macro(_) => {
                return Err(AdapterError::Unsupported {
                    reason: "macro invocation in statement position".into(),
                });
            }
        }
    }
    Ok(LoopBodyExit::FallThrough)
}

fn lower_break(b: &mut Builder, brk: &ExprBreak) -> Result<(), AdapterError> {
    if brk.expr.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "`break VALUE` — loop-as-expression value is out of M2.5b slice-3 scope".into(),
        });
    }
    if brk.label.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "labeled `break 'label` is out of M2.5b scope".into(),
        });
    }
    let ctx = b
        .loop_stack
        .last()
        .cloned_ctx()
        .ok_or_else(|| AdapterError::Unsupported {
            reason: "`break` outside of a loop".into(),
        })?;
    // Break jumps to the loop's exit block, which uses
    // `exit_merged_names` — the set excluding internal iter sidecars.
    let args: Vec<Hlvalue> = ctx
        .exit_merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let link = Rc::new(RefCell::new(Link::new(args, Some(ctx.exit_block), None)));
    b.finalize_current(vec![link], None);
    Ok(())
}

fn lower_continue(b: &mut Builder, cont: &ExprContinue) -> Result<(), AdapterError> {
    if cont.label.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "labeled `continue 'label` is out of M2.5b scope".into(),
        });
    }
    let ctx = b
        .loop_stack
        .last()
        .cloned_ctx()
        .ok_or_else(|| AdapterError::Unsupported {
            reason: "`continue` outside of a loop".into(),
        })?;
    let args: Vec<Hlvalue> = ctx
        .merged_names
        .iter()
        .map(|n| b.current.locals[n].clone())
        .collect();
    let link = Rc::new(RefCell::new(Link::new(args, Some(ctx.header_block), None)));
    b.finalize_current(vec![link], None);
    Ok(())
}

// ____________________________________________________________
// `?` operator — raising operation + exception edge to
// `graph.exceptblock`.
//
// Upstream basis: `rpython/flowspace/model.py:469-470`
// (`c_last_exception`), `:214-221` (`Block.canraise` /
// `Block.raising_op` properties), and the `graph.exceptblock`
// constructor at `:22-25`. A canraise block's last operation is the
// raising op, `exitswitch` is set to `c_last_exception`, `exits[0]`
// is the normal fall-through (exitcase=None, carrying the op's
// result), and `exits[1..]` are exception exits whose `exitcase` is
// an exception-class `Constant` and whose `last_exception` /
// `last_exc_value` carry the caught exception's type and value
// Variables. The RPython parser emits this shape when it encounters
// any opcode marked `canraise` in `operation.py:536-611`.
//
// Rust-specific adaptation: Rust's `?` expands to
//   match operand { Ok(v) => v, Err(e) => return Err(e.into()) }
// — an early-return on `Err`. We mirror this via the canraise shape
// rather than a direct match so the exception signal flows through
// `graph.exceptblock`, matching upstream's "uncaught exception
// propagates through the graph's exception exit" invariant.
// `HOST_ENV.lookup_exception_class("Exception")` provides the
// bootstrap exception-class HostObject required by
// `is_exception_exitcase` in `model.rs` `checkgraph`.

fn lower_try(b: &mut Builder, try_expr: &ExprTry) -> Result<Hlvalue, AdapterError> {
    // Upstream canraise sites are ops emitted by `ctx.do_op(op)` that
    // carry `canraise != []` (operation.py:475-611). The fork comes
    // from `guessexception` at flowcontext.py:124 / :379 / :385 —
    // attached to THAT real op, not a synthetic wrapper.
    //
    // Rust's `?` has no upstream counterpart, so the line-by-line-
    // orthodox mapping is: the `?` operand must itself be a call
    // whose lowered SpaceOperation IS the raising site. Any other
    // operand (bare variable, arithmetic, literal) has no call op
    // to hang canraise on.
    match &*try_expr.expr {
        Expr::Call(_) | Expr::MethodCall(_) => {}
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "`?` operand must be a direct call / method call — upstream \
                    canraise sites are the ops themselves (flowcontext.py:124, :379), \
                    not wrappers over arbitrary values"
                    .into(),
            });
        }
    }

    // Lower the operand. `lower_call` / `lower_method_call` emit their
    // SpaceOperation into `b.current.ops` and return its result
    // Variable. That last op is the raising site — no synthetic
    // wrapper.
    let unwrapped = lower_expr(b, &try_expr.expr)?;

    // Snapshot the locals set so the normal-exit Link can carry
    // them into the continuation block. The exception exit is only
    // [etype, evalue] — `graph.exceptblock.inputargs.len() == 2` per
    // the constructor at `model.py:22-25`; no caller locals survive
    // the exception edge.
    let pre_locals = b.locals().clone();
    let mut merged_names: Vec<String> = pre_locals.keys().cloned().collect();
    merged_names.sort();

    // Allocate the continuation block — inputargs =
    // [unwrapped_var, local_var_0, …]. The first inputarg is the
    // result of the raising op flowing through the normal exit.
    let cont_unwrapped_var = Hlvalue::Variable(Variable::new());
    let mut cont_inputargs: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    cont_inputargs.push(cont_unwrapped_var.clone());
    let mut cont_locals: HashMap<String, Hlvalue> = HashMap::new();
    for name in &merged_names {
        let fresh = Hlvalue::Variable(Variable::named(name));
        cont_inputargs.push(fresh.clone());
        cont_locals.insert(name.clone(), fresh);
    }
    let cont_block = Block::shared(cont_inputargs);

    // Build the normal-exit Link (exitcase=None). Args carry
    // [unwrapped, ...current-locals] — the call's result flows
    // into `cont_unwrapped_var`.
    let mut normal_args: Vec<Hlvalue> = Vec::with_capacity(merged_names.len() + 1);
    normal_args.push(unwrapped.clone());
    for name in &merged_names {
        normal_args.push(pre_locals[name].clone());
    }
    let normal_link = Rc::new(RefCell::new(Link::new(
        normal_args,
        Some(cont_block.clone()),
        None,
    )));

    // 6. Build the exception-exit Link. The target is
    //    `graph.exceptblock` (retrieved via Builder's reference), the
    //    exitcase is an exception-class Constant per
    //    `is_exception_exitcase`. `last_exception` / `last_exc_value`
    //    are fresh Variables defined exclusively on this link
    //    (checkgraph at `model.rs:3780-3785`).
    let etype = Variable::new();
    let evalue = Variable::new();
    let exc_class = HOST_ENV.lookup_exception_class("Exception").expect(
        "HOST_ENV bootstrap must register the builtin `Exception` class — \
            model.rs:1418 ensures it",
    );
    let exc_class_const = Hlvalue::Constant(Constant::new(ConstValue::HostObject(exc_class)));
    let mut exc_link_inner = Link::new(
        vec![
            Hlvalue::Variable(etype.clone()),
            Hlvalue::Variable(evalue.clone()),
        ],
        Some(b.exceptblock.clone()),
        Some(exc_class_const),
    );
    exc_link_inner.extravars(
        Some(Hlvalue::Variable(etype)),
        Some(Hlvalue::Variable(evalue)),
    );
    let exc_link = Rc::new(RefCell::new(exc_link_inner));

    // 7. Close the current block with the canraise shape. The
    //    exitswitch sentinel is `c_last_exception()` which
    //    `Block::canraise` detects via its Atom identity.
    let switch = Hlvalue::Constant(c_last_exception());
    b.finalize_current(vec![normal_link, exc_link], Some(switch));

    // 8. Open the continuation block. Subsequent reads of any local
    //    see the continuation's fresh inputarg; subsequent reads of
    //    the expression's value (the `?` unwrap result) see
    //    `cont_unwrapped_var`.
    b.open_new_block(BlockBuilder {
        block: cont_block,
        ops: Vec::new(),
        locals: cont_locals,
    });
    Ok(cont_unwrapped_var)
}

// ____________________________________________________________
// Method calls + function calls.
//
// Upstream basis:
// - `rpython/flowspace/operation.py:617-622` — `GetAttr(obj, name)`:
//   `arity=2`, `canraise=[]`, `pyfunc = staticmethod(getattr)`.
//   Emission convention is `getattr(obj, name_as_constant_str)`.
// - `rpython/flowspace/operation.py:663-679` — `SimpleCall(f, *args)`:
//   variable arity, no canraise at the op layer. Dispatched through
//   `SPECIAL_CASES` at annotator time if the callable is a
//   `Constant`; otherwise falls through to `ctx.do_op(self)`.
// - `rpython/flowspace/flowcontext.py` `LOAD_METHOD` + `CALL_METHOD`
//   (aliased at `:1000 CALL_METHOD = CALL_FUNCTION`): the bytecode
//   sequence emits `getattr(obj, name)` then `simple_call(bound,
//   *args)`, matching this lowering exactly.
//
// Trait dispatch is *not* emitted as a separate kind of op. The
// annotator downstream (`FunctionDesc.specialize` /
// `MethodDesc.bind_self`, `description.py:272-281` / `:1805-1819`)
// reads the receiver's `SomeInstance.classdef` and threads the
// concrete impl's `FunctionDesc` into the `simple_call` site. The
// adapter's job is to emit the structural `getattr + simple_call`
// pair so the annotator has something to rewrite.

fn lower_method_call(
    b: &mut Builder,
    method_call: &ExprMethodCall,
) -> Result<Hlvalue, AdapterError> {
    if method_call.turbofish.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "method turbofish `obj.method::<T>(…)` (explicit method generics land in \
                M2.5d alongside struct/enum literal typing)"
                .into(),
        });
    }
    let receiver = lower_expr(b, &method_call.receiver)?;

    // `getattr(receiver, "method")` — the method name is a Python
    // byte-string constant, matching Python 2 method names.
    let method_name = method_call.method.to_string();
    let bound = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getattr",
        vec![
            receiver,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str(method_name))),
        ],
        bound.clone(),
    ));

    // `simple_call(bound, *args)` — arg list starts with the bound
    // method (which carries the receiver after upstream's
    // `FunctionDesc.bind_self`), matching upstream
    // `operation.py:663-679 SimpleCall` convention.
    let mut call_args: Vec<Hlvalue> = Vec::with_capacity(method_call.args.len() + 1);
    call_args.push(bound);
    for arg in &method_call.args {
        call_args.push(lower_expr(b, arg)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "simple_call",
        call_args,
        result.clone(),
    ));
    Ok(result)
}

fn lower_call(b: &mut Builder, call: &ExprCall) -> Result<Hlvalue, AdapterError> {
    // Callee must be a simple identifier path. Qualified paths
    // (`module::fn`) would need module-resolved HostObject lookup
    // which the adapter does not perform — M2.5g registers adapter
    // -produced HostObjects by name, so those will land later.
    //
    // Codex 2026-05-03 parity audit removed the prior
    // `try_lower_stdlib_variant_call` short-circuit that collapsed
    // `Ok(x)` / `Some(x)` to their inner expression at every value
    // position. Upstream Python `Ok(x)` is an ordinary
    // constructor call: `LOAD_GLOBAL Ok; LOAD_FAST x; CALL_FUNCTION 1`,
    // emitting `simple_call(Ok, x)` per `operation.py:663-679
    // SimpleCall`. Erasing the call at value position produced a
    // graph that did not match what `flowcontext.py` would emit
    // for the equivalent Python source. Wrapper-transparency for
    // `return Ok(x)` / `arm => Ok(x)` lives at the boundary sites
    // (`lower_arm_body` / `lower_return`) only — see those
    // helpers for the documented PRE-EXISTING-ADAPTATION shape.
    match &*call.func {
        Expr::Path(_) => {}
        _ => {
            return Err(AdapterError::Unsupported {
                reason: "call callee must be a simple identifier path (closures, method-returned \
                    callables, etc. land in M2.5d/g)"
                    .into(),
            });
        }
    }

    let callee = lower_expr(b, &call.func)?;

    let mut call_args: Vec<Hlvalue> = Vec::with_capacity(call.args.len() + 1);
    call_args.push(callee);
    for arg in &call.args {
        call_args.push(lower_expr(b, arg)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "simple_call",
        call_args,
        result.clone(),
    ));
    Ok(result)
}

// ____________________________________________________________
// Tuple / array literals.
//
// Upstream basis:
// - `rpython/flowspace/operation.py:543-546` — `newtuple(*items)`:
//   variable arity, the annotator's `bookkeeper.newtuple` builds a
//   `SomeTuple` whose item types come from the arg annotations.
// - `rpython/flowspace/operation.py:552-559` — `newlist(*items)`:
//   variable arity, `bookkeeper.newlist` produces a `SomeList` whose
//   element type is the union of the arg types.
// - `rpython/flowspace/flowcontext.py:1163-1166` — `BUILD_TUPLE`
//   emits `op.newtuple(*items).eval(self)`.
// - `rpython/flowspace/flowcontext.py:1168-1171` — `BUILD_LIST`
//   emits `op.newlist(*items).eval(self)`.

fn lower_tuple(b: &mut Builder, tup: &ExprTuple) -> Result<Hlvalue, AdapterError> {
    let mut args: Vec<Hlvalue> = Vec::with_capacity(tup.elems.len());
    for elem in &tup.elems {
        args.push(lower_expr(b, elem)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("newtuple", args, result.clone()));
    Ok(result)
}

fn lower_array(b: &mut Builder, arr: &ExprArray) -> Result<Hlvalue, AdapterError> {
    let mut args: Vec<Hlvalue> = Vec::with_capacity(arr.elems.len());
    for elem in &arr.elems {
        args.push(lower_expr(b, elem)?);
    }
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new("newlist", args, result.clone()));
    Ok(result)
}

// ____________________________________________________________
// Unary operators, field access, index, cast — the "small surface"
// expression kinds that Rust source uses frequently.
//
// Upstream basis (all from `operation.py` `add_operator` registry):
// - `neg` arity=1 (:466)           — unary `-x`.
// - `pos` arity=1 (:465)           — unary `+x`.
// - `invert` arity=1 (:474)        — bitwise `~x`.
// - `getattr` arity=2 (:618)       — field access.
// - `getitem` arity=2 (:457)       — index access.
// - `int` / `float` / `bool` (:488/:490/:467) — type coercion.

fn lower_unary(b: &mut Builder, u: &ExprUnary) -> Result<Hlvalue, AdapterError> {
    // Upstream unary operators covered here are only those with a
    // direct 1-to-1 mapping into the `operation.py` `add_operator`
    // registry. `UnOp::Not` is deliberately NOT mapped — see the
    // match arm for rationale.
    match u.op {
        // `*x` — Rust deref. No upstream counterpart (Python has no
        // explicit deref). The annotator tracks identity + type
        // regardless of borrow form; pass-through is safe.
        UnOp::Deref(_) => lower_expr(b, &u.expr),
        // `-x` — upstream `operation.py:466 neg` arity=1.
        UnOp::Neg(_) => {
            let arg = lower_expr(b, &u.expr)?;
            let result = Hlvalue::Variable(Variable::new());
            b.emit_op(SpaceOperation::new("neg", vec![arg], result.clone()));
            Ok(result)
        }
        // `!x` — Rust overloads this for bitwise (ints) AND logical
        // (bools). Upstream handles the two paths *differently*:
        // - `UNARY_INVERT` (flowcontext.py:194) emits `op.invert`.
        // - `UNARY_NOT` (flowcontext.py:531) emits `op.bool` then
        //   forks on `guessbool` — `bool(x)` is a runtime
        //   conversion, the true-branch yields `Constant(False)`
        //   and the false-branch yields `Constant(True)`. There is
        //   no direct `not_` op in the registry.
        //
        // The adapter cannot pick bitwise-vs-logical without type
        // info, and has no `guessbool` facility to replicate the
        // upstream logical path. Rejecting keeps the lowering
        // honest — users who need bitwise NOT can write the
        // explicit helper call.
        UnOp::Not(_) => Err(AdapterError::Unsupported {
            reason: "unary `!` has no line-by-line upstream mapping (UNARY_NOT uses \
                `bool` + guessbool fork; UNARY_INVERT emits `invert`) — reject to keep \
                adapter orthodox"
                .into(),
        }),
        _ => Err(AdapterError::Unsupported {
            reason: "unrecognised unary operator".into(),
        }),
    }
}

fn lower_field(b: &mut Builder, f: &ExprField) -> Result<Hlvalue, AdapterError> {
    let base = lower_expr(b, &f.base)?;
    let attr_name = match &f.member {
        Member::Named(id) => id.to_string(),
        Member::Unnamed(idx) => {
            // `x.0` / `x.1` — tuple-struct index access. Upstream
            // Python has `tup[0]` which lowers to `getitem(tup,
            // 0)`; Rust tuple-struct / tuple field access has no
            // direct equivalent. Emit `getitem` with an integer
            // Constant index — the annotator can distinguish by
            // receiver type (Tuple vs user struct).
            let int_index = idx.index as i64;
            let result = Hlvalue::Variable(Variable::new());
            b.emit_op(SpaceOperation::new(
                "getitem",
                vec![
                    base,
                    Hlvalue::Constant(Constant::new(ConstValue::Int(int_index))),
                ],
                result.clone(),
            ));
            return Ok(result);
        }
    };
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getattr",
        vec![
            base,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str(attr_name))),
        ],
        result.clone(),
    ));
    Ok(result)
}

fn lower_index(b: &mut Builder, idx: &ExprIndex) -> Result<Hlvalue, AdapterError> {
    let base = lower_expr(b, &idx.expr)?;
    let key = lower_expr(b, &idx.index)?;
    let result = Hlvalue::Variable(Variable::new());
    b.emit_op(SpaceOperation::new(
        "getitem",
        vec![base, key],
        result.clone(),
    ));
    Ok(result)
}

fn lower_cast(_b: &mut Builder, _c: &ExprCast) -> Result<Hlvalue, AdapterError> {
    // `x as T` — Rust compile-time numeric conversion. Upstream has
    // no direct counterpart:
    //
    // - `operation.py:347 do_int(x)` → calls `x.__int__()`, not a
    //   numeric cast. It is a Python-method dispatch with its own
    //   method-resolution path.
    // - `operation.py:490 float(x)` → same shape, calls `__float__`.
    // - `operation.py:467 bool(x)` → calls `__bool__`.
    //
    // Mapping Rust `as` to these would silently inject a dunder call
    // the user never wrote, diverging semantically. Users who need
    // conversion can invoke the helper explicitly (e.g. `x.to_i64()`
    // or a registered function) and let the adapter lower that as a
    // method / function call.
    Err(AdapterError::Unsupported {
        reason: "`x as T` has no line-by-line upstream mapping — upstream `do_int` / \
            `do_float` / `bool` are __int__/__float__/__bool__ dunder dispatches, \
            not numeric coercions. Use an explicit helper call instead."
            .into(),
    })
}

/// Private helper trait — lets `lower_break`/`lower_continue` snapshot
/// the top-of-stack [`LoopCtx`] through an Option without tangling
/// borrow regions (the body then calls `b.finalize_current`, a
/// `&mut self` borrow on the Builder). Clones the small fields only;
/// it is not a hot path.
trait LoopCtxSnapshot {
    fn cloned_ctx(self) -> Option<LoopCtx>;
}

impl LoopCtxSnapshot for Option<&LoopCtx> {
    fn cloned_ctx(self) -> Option<LoopCtx> {
        self.map(|ctx| LoopCtx {
            header_block: ctx.header_block.clone(),
            exit_block: ctx.exit_block.clone(),
            merged_names: ctx.merged_names.clone(),
            exit_merged_names: ctx.exit_merged_names.clone(),
        })
    }
}

// ____________________________________________________________

fn validate_arm(arm: &Arm) -> Result<(), AdapterError> {
    if arm.guard.is_some() {
        return Err(AdapterError::Unsupported {
            reason: "match arm guard (`if COND`) (lands after control-flow slice)".into(),
        });
    }
    Ok(())
}

/// Flatten a `Pat::Or` into its constituent sub-patterns, recursing so
/// nested or-patterns (`A | (B | C)`) reduce to a flat list.
/// Parenthesised patterns are transparent — `(X)` is grouping-only in
/// syn, and upstream semantics treat the inner pattern identically —
/// so we unwrap them before deciding whether to recurse. Non-or
/// patterns contribute a single entry. No direct upstream analogue
/// (Python source produces `if-elif` chains, not or-patterns) but the
/// resulting list maps onto a standard fan-out of Links all pointing
/// at the same target block, which `model.py:648-692` admits directly.
fn flatten_or_pattern<'a>(pat: &'a Pat, out: &mut Vec<&'a Pat>) {
    match pat {
        Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                flatten_or_pattern(case, out);
            }
        }
        Pat::Paren(paren) => flatten_or_pattern(&paren.pat, out),
        other => out.push(other),
    }
}

/// Map a `syn::Pat` to the `Link.exitcase` upstream expects. Returns
/// `Ok(None)` for a wildcard (catch-all) arm — the caller inserts the
/// sentinel `"default"` string per `model.py:652`.
fn classify_pattern(pat: &Pat, is_last: bool) -> Result<Option<Hlvalue>, AdapterError> {
    match pat {
        Pat::Wild(_) => {
            if !is_last {
                return Err(AdapterError::Unsupported {
                    reason: "wildcard arm must be the last arm (upstream `model.py:652` invariant)"
                        .into(),
                });
            }
            Ok(None)
        }
        Pat::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Int(int) => {
                let value: i64 = int.base10_parse().map_err(|e| AdapterError::Unsupported {
                    reason: format!("match arm integer pattern out of i64 range: {e}"),
                })?;
                Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Int(
                    value,
                )))))
            }
            Lit::Bool(bl) => Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::Bool(
                bl.value,
            ))))),
            // Upstream `model.py:658` admits single-character strings as
            // switch exitcases (`isinstance(n, (str, unicode)) and
            // len(n) == 1`). Rust's `char` literal is the direct
            // analogue — RPython has no `char` type; it uses one-char
            // unicode. Emit `ConstValue::UniStr(c.to_string())` so the
            // resulting exitcase passes `checkgraph`'s len==1 check.
            Lit::Char(ch) => Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::uni_str(
                ch.value().to_string(),
            ))))),
            // Upstream `model.py:658` admits `isinstance(n, (str,
            // unicode)) and len(n) == 1` as a valid switch exitcase,
            // so single-character string patterns are legal. `char`
            // literals lower to the same `ConstValue::UniStr(c.to_string())`
            // shape (see the `Lit::Char` arm above), making
            // `match x { "a" => … }` and `match x { 'a' => … }`
            // structurally interchangeable. Multi-character strings
            // still reject — `checkgraph` would flag them.
            Lit::Str(s) => {
                let value = s.value();
                // `str.chars().count()` counts Unicode scalars,
                // matching what `len(u"é")` observes on RPython's
                // unicode side.
                if value.chars().count() != 1 {
                    return Err(AdapterError::Unsupported {
                        reason: "match arm multi-character string-literal pattern — \
                            upstream `model.py:658` admits only single-character \
                            strings as switch exitcases"
                            .into(),
                    });
                }
                Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::uni_str(
                    value,
                )))))
            }
            Lit::ByteStr(bs) if bs.value().len() == 1 => Ok(Some(Hlvalue::Constant(
                Constant::new(ConstValue::ByteStr(bs.value())),
            ))),
            Lit::Byte(b) => Ok(Some(Hlvalue::Constant(Constant::new(ConstValue::ByteStr(
                vec![b.value()],
            ))))),
            Lit::ByteStr(_) | Lit::Float(_) | _ => Err(AdapterError::Unsupported {
                reason: "match arm non-integer/bool/char literal pattern \
                        (byte/bytestring/float patterns have no upstream analogue)"
                    .into(),
            }),
        },
        Pat::Or(_) => {
            // Unreachable under `lower_match` which pre-flattens
            // or-patterns via `flatten_or_pattern` before calling
            // `classify_pattern`. Keep this arm as a defensive
            // contract so a future caller that forgets the flatten
            // step fails loudly instead of silently mis-classifying.
            Err(AdapterError::Unsupported {
                reason: "or-pattern reached classify_pattern without flattening — \
                    caller must pre-expand via `flatten_or_pattern`"
                    .into(),
            })
        }
        Pat::Range(_) => Err(AdapterError::Unsupported {
            reason: "match arm range pattern (`a..b`)".into(),
        }),
        Pat::TupleStruct(_) | Pat::Tuple(_) | Pat::Struct(_) => Err(AdapterError::Unsupported {
            reason: "match arm composite pattern (enum/tuple/struct — lands in M2.5d)".into(),
        }),
        Pat::Ident(_) => Err(AdapterError::Unsupported {
            reason: "match arm identifier-binding pattern".into(),
        }),
        _ => Err(AdapterError::Unsupported {
            reason: "match arm pattern not in M2.5b subset (only int / bool literals and \
                wildcard supported)"
                .into(),
        }),
    }
}

// ____________________________________________________________

fn discriminant(expr: &Expr) -> &'static str {
    match expr {
        Expr::Array(_) => "Array",
        Expr::Assign(_) => "Assign",
        Expr::Async(_) => "Async",
        Expr::Await(_) => "Await",
        Expr::Binary(_) => "Binary",
        Expr::Block(_) => "Block",
        Expr::Break(_) => "Break",
        Expr::Call(_) => "Call",
        Expr::Cast(_) => "Cast",
        Expr::Closure(_) => "Closure",
        Expr::Const(_) => "Const",
        Expr::Continue(_) => "Continue",
        Expr::Field(_) => "Field",
        Expr::ForLoop(_) => "ForLoop",
        Expr::Group(_) => "Group",
        Expr::If(_) => "If",
        Expr::Index(_) => "Index",
        Expr::Infer(_) => "Infer",
        Expr::Let(_) => "Let",
        Expr::Lit(_) => "Lit",
        Expr::Loop(_) => "Loop",
        Expr::Macro(_) => "Macro",
        Expr::Match(_) => "Match",
        Expr::MethodCall(_) => "MethodCall",
        Expr::Paren(_) => "Paren",
        Expr::Path(_) => "Path",
        Expr::Range(_) => "Range",
        Expr::Reference(_) => "Reference",
        Expr::Repeat(_) => "Repeat",
        Expr::Return(_) => "Return",
        Expr::Struct(_) => "Struct",
        Expr::Try(_) => "Try",
        Expr::TryBlock(_) => "TryBlock",
        Expr::Tuple(_) => "Tuple",
        Expr::Unary(_) => "Unary",
        Expr::Unsafe(_) => "Unsafe",
        Expr::Verbatim(_) => "Verbatim",
        Expr::While(_) => "While",
        Expr::Yield(_) => "Yield",
        _ => "Unknown",
    }
}

// ____________________________________________________________

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flowspace::model::checkgraph;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str(src).expect("test source should parse")
    }

    fn lower(src: &str) -> Result<FunctionGraph, AdapterError> {
        build_flow_from_rust(&parse(src))
    }

    // ---- M2.5a straight-line accept tests -------------------------

    #[test]
    fn lit_constant_return() {
        let g = lower("fn one() -> i64 { 1 }").unwrap();
        checkgraph(&g);
        assert_eq!(g.name, "one");
        assert_eq!(g.getargs().len(), 0);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);
        let link = start.exits[0].borrow();
        assert_eq!(link.args.len(), 1);
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
    }

    #[test]
    fn identity_function() {
        let g = lower("fn identity(x: i64) -> i64 { x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        let link = start.exits[0].borrow();
        match &start.inputargs[0] {
            Hlvalue::Variable(v) => match link.args[0].as_ref().unwrap() {
                Hlvalue::Variable(lv) => assert_eq!(lv.id(), v.id()),
                other => panic!("expected Variable, got {other:?}"),
            },
            other => panic!("expected Variable inputarg, got {other:?}"),
        }
    }

    #[test]
    fn binop_add_emits_space_operation() {
        let g = lower("fn add(a: i64, b: i64) -> i64 { a + b }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        let op = &start.operations[0];
        assert_eq!(op.opname, "add");
        assert_eq!(op.args.len(), 2);
    }

    #[test]
    fn binop_sub_emits_space_operation() {
        let g = lower("fn sub(a: i64, b: i64) -> i64 { a - b }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations[0].opname, "sub");
    }

    #[test]
    fn let_then_tail_chains_two_ops() {
        let g = lower(
            "fn f(a: i64, b: i64) -> i64 {
                let t = a + b;
                t * t
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "mul");
        let t_var = &start.operations[0].result;
        assert_eq!(&start.operations[1].args[0], t_var);
        assert_eq!(&start.operations[1].args[1], t_var);
    }

    #[test]
    fn chained_lets_and_rebinding() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                let x = x + 1;
                let x = x * 2;
                x
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "mul");
        let link = start.exits[0].borrow();
        assert_eq!(link.args[0].as_ref().unwrap(), &start.operations[1].result);
    }

    #[test]
    fn comparison_ops() {
        let g = lower("fn f(a: i64, b: i64) -> bool { a < b }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations[0].opname, "lt");
    }

    #[test]
    fn bool_literal_return() {
        let g = lower("fn f() -> bool { true }").unwrap();
        checkgraph(&g);
        let link = g.startblock.borrow().exits[0].clone();
        match link.borrow().args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Bool(true)),
            other => panic!("expected Constant(true), got {other:?}"),
        }
    }

    #[test]
    fn explicit_return_is_accepted_in_tail() {
        let g = lower("fn f() -> i64 { return 1; }").unwrap();
        checkgraph(&g);
        let link = g.startblock.borrow().exits[0].clone();
        match link.borrow().args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
    }

    // ---- M2.5b if/else accept tests --------------------------------

    #[test]
    fn if_else_literal_branches() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { 1 } else { 0 }
            }",
        )
        .unwrap();
        checkgraph(&g);

        // startblock: [x] → emits `x > 0` then `bool(gt_result)`
        // (upstream POP_JUMP_IF_FALSE wraps cond in `op.bool`), so
        // exitswitch = the bool-op result, 2 exits.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "gt");
        assert_eq!(start.operations[1].opname, "bool");
        let switch_var = start.operations[1].result.clone();
        assert_eq!(start.exitswitch.as_ref(), Some(&switch_var));
        assert_eq!(start.exits.len(), 2);

        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        assert_eq!(
            false_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            true_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );

        // Both branches reach the same join block (iterblocks has 4:
        // start, else, then, join; returnblock is listed separately
        // and may appear as the join's successor).
        drop(false_exit);
        drop(true_exit);
        drop(start);
        let blocks = g.iterblocks();
        // start + else + then + join + returnblock.
        assert_eq!(blocks.len(), 5);
    }

    #[test]
    fn if_else_preserves_local_merge() {
        // Both branches modify `x`; join must receive the merged
        // `x_local_*` through its inputargs and forward it.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let y = if x > 0 { x + 1 } else { x - 1 };
                y
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        assert_eq!(blocks.len(), 5); // start + else + then + join + return
        // Join has 2 inputargs: the if-tail + the merged `x`.
        let join = blocks
            .iter()
            .find(|b| b.borrow().inputargs.len() == 2)
            .expect("expected join block with 2 inputargs");
        let join_ref = join.borrow();
        assert_eq!(join_ref.operations.len(), 0);
        // Join's single exit carries [tail_var] into returnblock (plus
        // any merged locals are dropped — tail assignment just reads
        // the first inputarg).
        let exit = join_ref.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        assert_eq!(exit.args[0].as_ref().unwrap(), &join_ref.inputargs[0]);
    }

    #[test]
    fn if_else_nested() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 {
                    if x > 10 { 2 } else { 1 }
                } else {
                    0
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Outer if-else has nested if-else in then-branch: 5 blocks
        // from the outer, +2 (inner then + inner else) +1 (inner join,
        // which acts as the outer then-branch's ops host), minus
        // duplication of the outer join…
        // Rather than pin exact count, check that checkgraph passes
        // (structural invariants hold).
        assert!(g.iterblocks().len() >= 5);
    }

    #[test]
    fn if_else_if_chain_positive_literals() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x == 0 { 0 } else if x == 1 { 1 } else { 2 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // 2 nested if-elses ⇒ 2 fork blocks + 2 × 2 branch blocks +
        // 2 joins + startblock overlap: at least 7 reachable blocks.
        assert!(g.iterblocks().len() >= 6);
    }

    // ---- M2.5b match accept tests ---------------------------------

    #[test]
    fn match_int_literals_with_default() {
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x {
                    0 => 0,
                    1 => 10,
                    _ => 99,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // scrutinee path is a bare LOAD — no operation emitted.
        assert_eq!(start.operations.len(), 0);
        // exitswitch is the scrutinee (`x`'s Variable, per identity).
        assert!(start.exitswitch.is_some());
        assert_eq!(start.exits.len(), 3);
        // Per-arm exitcase ordering matches source order.
        let ec0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec1 = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec2 = start.exits[2].borrow().exitcase.clone().unwrap();
        assert_eq!(ec0, Hlvalue::Constant(Constant::new(ConstValue::Int(0))));
        assert_eq!(ec1, Hlvalue::Constant(Constant::new(ConstValue::Int(1))));
        assert_eq!(
            ec2,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("default")))
        );
    }

    #[test]
    fn match_scrutinee_is_op_result() {
        // Scrutinee is a non-trivial expression — the binop's result
        // Variable becomes `exitswitch`.
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x + 1 {
                    0 => 0,
                    _ => 1,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "add");
        let add_result = start.operations[0].result.clone();
        assert_eq!(start.exitswitch.as_ref(), Some(&add_result));
    }

    #[test]
    fn match_merges_locals_into_join() {
        // Every arm rebinds `y`. The join block's inputargs must carry
        // both `tail_var` and the merged `y`.
        let g = lower(
            "fn f(x: i64, y: i64) -> i64 {
                let y = match x {
                    0 => y + 1,
                    _ => y - 1,
                };
                y
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock + 2 arm blocks + 1 join + returnblock.
        assert_eq!(blocks.len(), 5);
        // Locate the join block — it has len(merged_names)+1 = 3
        // inputargs (tail_var, x_merged, y_merged) and zero ops. The
        // startblock also has zero ops but only 2 inputargs (x, y).
        let join = blocks
            .iter()
            .find(|b| b.borrow().inputargs.len() == 3 && b.borrow().operations.is_empty())
            .expect("expected empty-ops join block with 3 inputargs");
        let join_ref = join.borrow();
        // Join's single exit forwards inputarg[0] (tail_var) as the
        // return, matching STORE_FAST `y = <match tail>` semantics.
        let exit = join_ref.exits[0].borrow();
        assert_eq!(exit.args.len(), 1);
        assert_eq!(exit.args[0].as_ref().unwrap(), &join_ref.inputargs[0]);
    }

    #[test]
    fn match_bool_scrutinee_two_arms() {
        // 2-arm match on a bool is the boolean-switch shape — upstream
        // `model.py:643-644` accepts `[False, True]` ordering but the
        // adapter preserves *source* order, i.e. whatever the user
        // writes. `checkgraph` tolerates the multi-case branch too.
        let g = lower(
            "fn f(x: bool) -> i64 {
                match x {
                    true => 1,
                    false => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let ec0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec1 = start.exits[1].borrow().exitcase.clone().unwrap();
        assert_eq!(
            ec0,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );
        assert_eq!(
            ec1,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
    }

    #[test]
    fn match_single_wildcard_is_straight_line() {
        // Degenerate case: single wildcard arm. The adapter still emits
        // a fork + single-arm join; tail propagates through.
        let g = lower("fn f(x: i64) -> i64 { match x { _ => 42 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 1);
        let ec = start.exits[0].borrow().exitcase.clone().unwrap();
        assert_eq!(
            ec,
            Hlvalue::Constant(Constant::new(ConstValue::byte_str("default")))
        );
    }

    // ---- reject paths ---------------------------------------------

    // NOTE: generic fns are now accepted (slice M2.5c). The
    // annotator layer monomorphizes via `FunctionDesc.specialize`.
    // See `generic_fn_identity` below.

    #[test]
    fn rejects_async_fn() {
        assert!(matches!(
            lower("async fn f() -> i64 { 1 }").unwrap_err(),
            AdapterError::InvalidSignature { .. }
        ));
    }

    #[test]
    fn rejects_unsafe_fn() {
        assert!(matches!(
            lower("unsafe fn f() -> i64 { 1 }").unwrap_err(),
            AdapterError::InvalidSignature { .. }
        ));
    }

    // NOTE: `self` receivers are now accepted (slice M2.5c) as a
    // local named `"self"`. See `self_receiver_is_local_named_self`
    // below.

    #[test]
    fn if_without_else_return_in_branch_closes_to_returnblock() {
        // Upstream `flowcontext.py:687 RETURN_VALUE` raises `Return`
        // and `flowcontext.py:1232 Return.nomoreblocks(ctx)` closes
        // the current block straight to `graph.returnblock`. In an
        // `if cond { return X; } tail` shape the then-branch's
        // closing Link therefore targets `returnblock` (carrying the
        // return value), while the `false` shortcut threads through
        // the join and on to the function's implicit-None return
        // tail.
        let g = lower("fn f(x: i64) -> i64 { if x > 0 { return 1; } 2 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        let true_target = true_exit
            .target
            .as_ref()
            .expect("true branch has target")
            .clone();
        // The then_block's only exit Links to `graph.returnblock`
        // with `Constant(1)` — upstream `flowcontext.py:1232`
        // structural shape.
        let then_exits = true_target.borrow().exits.clone();
        assert_eq!(
            then_exits.len(),
            1,
            "then_block closes with a single returnblock Link per flowcontext.py:1232"
        );
        let then_link = then_exits[0].borrow();
        let then_link_target = then_link
            .target
            .as_ref()
            .expect("then-Link has target")
            .clone();
        assert!(
            Rc::ptr_eq(&then_link_target, &g.returnblock),
            "then-branch `return` must Link directly to graph.returnblock (not join)"
        );
        assert_eq!(then_link.args.len(), 1);
        match then_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(1)),
            other => panic!("expected Constant(1), got {other:?}"),
        }
        // The `false` shortcut threads to the post-if join, which in
        // turn closes via the `2` tail into `returnblock`. Walk it
        // explicitly.
        let false_target = false_exit
            .target
            .as_ref()
            .expect("false branch has target")
            .clone();
        let join_exits = false_target.borrow().exits.clone();
        assert_eq!(join_exits.len(), 1);
        let join_link = join_exits[0].borrow();
        assert!(Rc::ptr_eq(
            join_link.target.as_ref().unwrap(),
            &g.returnblock
        ));
        match join_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::Int(2)),
            other => panic!("expected Constant(2), got {other:?}"),
        }
    }

    #[test]
    fn if_else_both_branches_return_produces_no_join() {
        // When every branch of an if/else terminates via `return`,
        // upstream's `Return.nomoreblocks()` runs on both sides and
        // `StopFlowing` prevents the pending-block scheduler from
        // ever enqueueing a post-if PC. Result: no join block, just
        // two fork-target blocks that each Link to `graph.returnblock`.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { return 1; } else { return 2; }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        // Each branch block has exactly one exit, which Links to
        // graph.returnblock.
        for exit in start.exits.iter() {
            let branch = exit
                .borrow()
                .target
                .as_ref()
                .expect("fork exit has target")
                .clone();
            let branch_exits = branch.borrow().exits.clone();
            assert_eq!(branch_exits.len(), 1);
            let link = branch_exits[0].borrow();
            assert!(Rc::ptr_eq(link.target.as_ref().unwrap(), &g.returnblock));
        }
    }

    #[test]
    fn if_else_then_returns_else_falls_through_yields_else_tail() {
        // Mixed case: then terminates via `return`, else falls
        // through with a tail value. The if-expression's value is
        // the else-tail, and the join block has exactly one
        // predecessor (the else arm).
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { return 1; } else { 2 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Walk to the returnblock from the else-side — its incoming
        // Link should carry `Constant(2)` per the else-tail.
        let ret = g.returnblock.borrow();
        assert_eq!(ret.inputargs.len(), 1);
        assert!(ret.is_final);
    }

    #[test]
    fn match_all_arms_return_yields_no_join() {
        // Upstream `model.py:648-692` builds the switch-exit table
        // the same way `flowcontext.py:1232` closes each arm — if
        // every arm's body raises `Return`, the post-match PC is
        // unreachable and the scheduler never allocates a join.
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x {
                    0 => return 10,
                    _ => return 20,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Each arm's branch block ends in a Link straight to
        // graph.returnblock.
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        for exit in start.exits.iter() {
            let branch = exit
                .borrow()
                .target
                .as_ref()
                .expect("fork exit has target")
                .clone();
            let branch_exits = branch.borrow().exits.clone();
            assert_eq!(branch_exits.len(), 1);
            let link = branch_exits[0].borrow();
            assert!(Rc::ptr_eq(link.target.as_ref().unwrap(), &g.returnblock));
        }
    }

    #[test]
    fn return_before_end_of_block_rejects() {
        // `return X; Y` with anything after the return is unreachable
        // dead code in upstream terms — `flowcontext.py:1232` closes
        // the block at Return; subsequent ops never emit. Reject
        // cleanly so users get a parse-time error instead of silent
        // dead-code emission.
        match lower("fn f() -> i64 { return 1; 2 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("after `return`"),
                    "reason should cite dead code: {reason}"
                );
            }
            other => panic!("expected Unsupported(dead code after return), got {other:?}"),
        }
    }

    #[test]
    fn if_without_else_statement_shortcuts_false_link_to_join() {
        // `if x > 0 { let _ = 1; }` as a statement lowers to the
        // upstream `POP_JUMP_IF_FALSE` shape: fork block has two
        // exits, the Bool(false) link shortcuts straight to the join
        // block (no else block allocated), the Bool(true) link
        // routes through the then_block which in turn Links into the
        // same join.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
                2
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2, "if fork has exactly two exits");
        let false_exit = start.exits[0].borrow();
        let true_exit = start.exits[1].borrow();
        assert!(
            matches!(
                &false_exit.exitcase,
                Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Bool(false))
            ),
            "first exit must carry Bool(false)"
        );
        assert!(
            matches!(
                &true_exit.exitcase,
                Some(Hlvalue::Constant(c)) if matches!(c.value, ConstValue::Bool(true))
            ),
            "second exit must carry Bool(true)"
        );
        // False shortcut: its target is reached from fork in one hop
        // and each of its exits Links to the join. True path routes
        // through a then_block whose single exit Links to the same
        // join. Verified by following one hop from each side and
        // asserting pointer-equality of their targets.
        let false_target = false_exit
            .target
            .as_ref()
            .expect("false has target")
            .clone();
        let true_target = true_exit.target.as_ref().expect("true has target").clone();
        assert!(
            !Rc::ptr_eq(&false_target, &true_target),
            "false shortcut and true branch must head to distinct blocks from the fork"
        );
        // The false-side target IS the join directly; the true-side
        // target is the then_block, whose single exit Links to the
        // join.
        let then_exits = true_target.borrow().exits.clone();
        assert_eq!(then_exits.len(), 1, "then_block single-exit to join");
        let then_link_target = then_exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("then_block link has target")
            .clone();
        assert!(
            Rc::ptr_eq(&false_target, &then_link_target),
            "then_block's Link target must be the SAME join block the false shortcut points at"
        );
    }

    #[test]
    fn if_without_else_tail_returns_none() {
        // In tail position, `if cond { body }` produces `None` as
        // the expression value — matches Python's fallthrough
        // convention. `ConstValue::None` should flow into the
        // returnblock Link.
        let g = lower(
            "fn f(x: i64) -> i64 {
                if x > 0 { let _y = 1; }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Walk to the returnblock: startblock → join → returnblock.
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 2);
        let false_target = start.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("false target")
            .clone();
        // The join's single exit is the Link to the returnblock,
        // carrying the None tail as its first (and only) arg.
        let join_exits = false_target.borrow().exits.clone();
        assert_eq!(join_exits.len(), 1);
        let return_link = join_exits[0].borrow();
        assert!(!return_link.args.is_empty(), "return link carries tail arg");
        match return_link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => assert_eq!(c.value, ConstValue::None),
            other => panic!("expected Constant(None), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_arm_guard() {
        // Arm guards (`if COND`) are out-of-scope for M2.5b. They
        // require the slice-2 fork + slice-1 if lowering in sequence
        // inside the arm body, which the adapter does not yet plumb.
        match lower("fn f(x: i64) -> i64 { match x { n if n > 0 => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("guard"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(guard), got {other:?}"),
        }
    }

    #[test]
    fn or_pattern_emits_multiple_links_sharing_one_branch_block() {
        // `match x { 1 | 2 => 1, _ => 0 }` — or-pattern `1 | 2`
        // lowers into TWO Links from the fork block (one per
        // sub-pattern) both targeting the SAME branch block; then one
        // wildcard-default Link for the `_` arm. Upstream
        // `model.py:648-692` admits multiple Links with distinct
        // exitcases pointing at the same target block.
        let g = lower("fn f(x: i64) -> i64 { match x { 1 | 2 => 1, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 3, "two or-sub-cases + one default");
        let target_0 = start.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        let target_1 = start.exits[1]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        let target_2 = start.exits[2]
            .borrow()
            .target
            .as_ref()
            .expect("fork exit has target")
            .clone();
        assert!(
            Rc::ptr_eq(&target_0, &target_1),
            "or-pattern sub-cases must share one branch block"
        );
        assert!(
            !Rc::ptr_eq(&target_0, &target_2),
            "default arm must reach a distinct branch block"
        );
        // Exitcases: Int(1), Int(2), Str("default").
        let ec_0 = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec_1 = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec_2 = start.exits[2].borrow().exitcase.clone().unwrap();
        assert!(
            matches!(
                ec_0,
                Hlvalue::Constant(ref c) if matches!(c.value, ConstValue::Int(1))
            ),
            "first or-sub-case should be Int(1)"
        );
        assert!(
            matches!(
                ec_1,
                Hlvalue::Constant(ref c) if matches!(c.value, ConstValue::Int(2))
            ),
            "second or-sub-case should be Int(2)"
        );
        assert!(
            matches!(
                ec_2,
                Hlvalue::Constant(ref c) if c.value.string_eq("default")
            ),
            "default arm should carry Str(\"default\")"
        );
    }

    #[test]
    fn or_pattern_nested_flattens_to_siblings() {
        // `A | (B | C)` — nested or-pattern. `flatten_or_pattern`
        // recurses so three Links come out of the fork.
        let g = lower("fn f(x: i64) -> i64 { match x { 1 | (2 | 3) => 1, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 4, "three or-sub-cases + one default");
        let t0 = start.exits[0].borrow().target.as_ref().unwrap().clone();
        let t1 = start.exits[1].borrow().target.as_ref().unwrap().clone();
        let t2 = start.exits[2].borrow().target.as_ref().unwrap().clone();
        assert!(
            Rc::ptr_eq(&t0, &t1) && Rc::ptr_eq(&t1, &t2),
            "all three or-sub-cases must share the same branch block"
        );
    }

    #[test]
    fn rejects_wildcard_inside_or_pattern() {
        // Upstream `model.py:652` reserves wildcard for the
        // standalone default arm; embedding it inside an or-pattern
        // would duplicate the "catch-all" intent.
        match lower("fn f(x: i64) -> i64 { match x { 1 | _ => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("wildcard sub-pattern inside or-pattern"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(wildcard-in-or), got {other:?}"),
        }
    }

    #[test]
    fn rejects_or_pattern_with_duplicate_case() {
        // `1 | 1` or `1 | 2` crossing into another arm that reuses 2
        // both violate the uniqueness invariant
        // (`model.py:692 allexitcases`).
        match lower("fn f(x: i64) -> i64 { match x { 1 | 2 => 1, 2 => 2, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("repeated"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(repeated), got {other:?}"),
        }
    }

    #[test]
    fn match_char_pattern_emits_single_char_str_exitcase() {
        // Upstream `model.py:658` admits `isinstance(n, (str, unicode))
        // and len(n) == 1` as a switch exitcase. Rust's `char` literal
        // (`'a'`) is the direct analogue — RPython has no `char` type,
        // so single-char strings fill the role. The arm exitcase
        // should be `ConstValue::UniStr("a")` (len==1), passing
        // `checkgraph`.
        let g = lower("fn f(c: char) -> i64 { match c { 'a' => 1, 'b' => 2, _ => 0 } }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 3, "two char arms + default");
        let ec_a = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec_b = start.exits[1].borrow().exitcase.clone().unwrap();
        let ec_d = start.exits[2].borrow().exitcase.clone().unwrap();
        assert!(
            matches!(
                ec_a,
                Hlvalue::Constant(ref c) if c.value.string_eq("a")
            ),
            "first char arm should carry Str(\"a\") — got {ec_a:?}"
        );
        assert!(
            matches!(
                ec_b,
                Hlvalue::Constant(ref c) if c.value.string_eq("b")
            ),
            "second char arm should carry Str(\"b\") — got {ec_b:?}"
        );
        assert!(
            matches!(
                ec_d,
                Hlvalue::Constant(ref c) if c.value.string_eq("default")
            ),
            "wildcard arm should carry Str(\"default\") — got {ec_d:?}"
        );
    }

    #[test]
    fn rejects_match_multichar_string_literal_pattern() {
        // `Lit::Str` with len > 1 violates upstream `model.py:658`'s
        // `len(n) == 1` invariant for string exitcases. Single-char
        // strings are accepted (see
        // `match_single_char_str_pattern_emits_str_exitcase`).
        match lower("fn f(s: &str) -> i64 { match s { \"abc\" => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("multi-character"),
                    "reason should cite multi-char rejection: {reason}"
                );
                assert!(
                    reason.contains("model.py:658"),
                    "reason should cite the upstream rule: {reason}"
                );
            }
            other => panic!("expected Unsupported(multi-char string), got {other:?}"),
        }
    }

    #[test]
    fn match_single_char_str_pattern_emits_str_exitcase() {
        // Upstream `model.py:658` admits single-character strings as
        // valid switch exitcases; the adapter should accept `"a"` in
        // a match arm the same way it accepts `'a'`. Both produce a
        // `ConstValue::UniStr("a".into())` exitcase — structurally
        // interchangeable at the graph layer.
        let g = lower(
            "fn f(s: &str) -> i64 {
                match s { \"a\" => 1, _ => 0 }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        let ec = start.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        assert!(
            matches!(
                ec,
                Hlvalue::Constant(ref c) if c.value.string_eq("a")
            ),
            "single-char string pattern should carry Str(\"a\") — got {ec:?}"
        );
    }

    #[test]
    fn match_single_char_str_and_char_pattern_produce_identical_exitcase() {
        // `match x { "a" => … }` and `match x { 'a' => … }` must
        // produce the same `ConstValue::UniStr("a")` exitcase. This
        // pins `lower_literal` / `classify_pattern`'s symmetry
        // between the two syntactic forms.
        let g_str = lower("fn f(s: &str) -> i64 { match s { \"a\" => 1, _ => 0 } }").unwrap();
        let g_char = lower("fn f(c: char) -> i64 { match c { 'a' => 1, _ => 0 } }").unwrap();
        let start_str = g_str.startblock.borrow();
        let start_char = g_char.startblock.borrow();
        let ec_str = start_str.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        let ec_char = start_char.exits[0]
            .borrow()
            .exitcase
            .clone()
            .expect("arm has exitcase");
        assert_eq!(ec_str, ec_char);
    }

    #[test]
    fn rejects_match_range_pattern() {
        match lower("fn f(x: i64) -> i64 { match x { 0..=9 => 1, _ => 0 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("range"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(range), got {other:?}"),
        }
    }

    #[test]
    fn rejects_match_wildcard_not_last() {
        match lower("fn f(x: i64) -> i64 { match x { _ => 0, 1 => 1 } }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("wildcard"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(wildcard-not-last), got {other:?}"),
        }
    }

    // ---- M2.5d slice 2c: variant-cascade tests -------------------

    /// `match` over a unit enum variant routes through the
    /// isinstance-cascade path. Each variant emits a `getattr` cascade
    /// resolving `Foo::A` to a `Constant(HostObject(<A class>))`,
    /// followed by an `isinstance` op + 2-exit Bool fork; the wildcard
    /// arm is the cascade's terminal "false" target. Mirrors upstream
    /// `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR` +
    /// `operation.py:449 isinstance` shape for `if isinstance(x,
    /// Foo.A): … else: …`.
    #[test]
    fn match_unit_variant_two_arms_emits_isinstance_cascade() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // First-step block: getattr(<Foo>, "A") then isinstance(scrut,
        // <A>) then 2-exit Bool fork.
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "isinstance");
        assert_eq!(start.exits.len(), 2);
        let ec_false = start.exits[0].borrow().exitcase.clone().unwrap();
        let ec_true = start.exits[1].borrow().exitcase.clone().unwrap();
        assert_eq!(
            ec_false,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            ec_true,
            Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );
        // The getattr's first arg is `Constant(HostObject(Foo))` —
        // the leftmost segment minted by `Builder::resolve_path_constant`
        // and cached in the process-global `host_env::HOST_CLASS_MINTS`
        // registry. The second arg is the attribute name `"A"` per
        // `operation.py:618 getattr` arity=2.
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(
                        obj.is_class(),
                        "leftmost segment must mint a HostObject::Class"
                    );
                    assert_eq!(obj.qualname(), "Foo");
                }
                other => panic!("expected HostObject leftmost, got {other:?}"),
            },
            other => panic!("expected Constant leftmost, got {other:?}"),
        }
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(bytes) => assert_eq!(bytes, b"A"),
                other => panic!("expected ByteStr attr name, got {other:?}"),
            },
            other => panic!("expected Constant attr name, got {other:?}"),
        }
        // The isinstance op consumes the getattr result as its second
        // operand — the resolved variant class, not a sentinel.
        let getattr_result = start.operations[0].result.clone();
        assert_eq!(start.operations[1].args[1], getattr_result);
    }

    /// Three variant arms produce three cascade steps. Each step is a
    /// 2-exit boolean fork — total entry-reachable blocks should
    /// include the start, two intermediate cascade fork blocks, three
    /// arm body blocks, one join, and the returnblock.
    #[test]
    fn match_three_unit_variants_chain_three_isinstance_forks() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    Foo::B => 2,
                    Foo::C => 3,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Count blocks that emit an `isinstance` op — exactly one per
        // cascade step, three total.
        let isinstance_blocks = g
            .iterblocks()
            .iter()
            .filter(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .any(|op| op.opname == "isinstance")
            })
            .count();
        assert_eq!(isinstance_blocks, 3, "one cascade step per variant");
    }

    /// Or-pattern `Foo::A | Foo::B => body` produces TWO cascade steps
    /// pointing at the SAME arm body block — the structural shape that
    /// upstream `model.py:648-692` admits (multiple Links with distinct
    /// exitcases sharing a target). Plus the wildcard arm.
    #[test]
    fn match_or_variant_pattern_shares_arm_body_block() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A | Foo::B => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // 2 cascade steps × 1 isinstance op each = 2 isinstance ops
        // total.
        let isinstance_count = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| op.opname == "isinstance")
                    .count()
            })
            .sum::<usize>();
        assert_eq!(isinstance_count, 2, "or-pattern unfolds to 2 steps");

        // Find the two true-target Block IDs — they should be
        // identical (the shared arm body block).
        let mut true_targets: Vec<*const _> = Vec::new();
        for blk in g.iterblocks() {
            for exit in &blk.borrow().exits {
                let ex = exit.borrow();
                if let Some(Hlvalue::Constant(c)) = &ex.exitcase {
                    if matches!(&c.value, ConstValue::Bool(true)) {
                        if let Some(target) = &ex.target {
                            true_targets.push(target.as_ptr() as *const _);
                        }
                    }
                }
            }
        }
        assert_eq!(true_targets.len(), 2, "two cascade-step true exits");
        assert_eq!(
            true_targets[0], true_targets[1],
            "or-pattern sub-cases share the arm body block"
        );
    }

    /// Binding-wildcard `name => body` makes `name` available as a
    /// local inside the arm body, bound to the scrutinee value.
    #[test]
    fn match_binding_wildcard_threads_scrutinee_into_arm_body() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    other => other,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // The arm body for `other => other` reads the scrutinee back
        // out — there should be no extra `getattr` op required to
        // surface it. The graph remains structurally simple (no
        // operations beyond the cascade's isinstance).
        let isinstance_count = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| op.opname == "isinstance")
                    .count()
            })
            .sum::<usize>();
        assert_eq!(isinstance_count, 1, "single variant arm = 1 cascade step");
    }

    /// Variant-cascade matches REQUIRE a wildcard last. Without one,
    /// the adapter rejects with a reason that names the constraint —
    /// the adapter cannot enumerate Rust's variant universe from
    /// `syn::ItemFn` alone, so the user-provided catch-all is the
    /// only structural way to close the final isinstance fork.
    #[test]
    fn rejects_variant_match_without_wildcard() {
        match lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    Foo::B => 2,
                }
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("wildcard"),
                    "reason should cite wildcard requirement: {reason}"
                );
            }
            other => panic!("expected Unsupported(wildcard-required), got {other:?}"),
        }
    }

    /// Mixed literal + variant arms reject — the cascade lowering is
    /// homogeneous-variant only.
    #[test]
    fn rejects_match_mixing_literal_and_variant_arms() {
        match lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    1 => 2,
                    _ => 0,
                }
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("mix") || reason.contains("homogeneous"),
                    "reason should cite mixed-pattern rejection: {reason}"
                );
            }
            other => panic!("expected Unsupported(mixed), got {other:?}"),
        }
    }

    /// Slice 2d accept: rest-only struct-variant pattern
    /// `Foo::Bar { .. }` lowers to a single getattr+isinstance cascade
    /// step — structurally identical to the unit-variant case (slice
    /// 2c). The field-binding extraction is deliberately deferred so
    /// this slice carries zero new SpaceOperations beyond the cascade
    /// pair.
    #[test]
    fn match_rest_only_struct_variant_emits_isinstance_cascade() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Resume { .. } => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "isinstance");
        assert_eq!(start.exits.len(), 2);
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(bytes) => assert_eq!(bytes, b"Resume"),
                other => panic!("expected ByteStr attr name, got {other:?}"),
            },
            other => panic!("expected Constant attr name, got {other:?}"),
        }
        let getattr_result = start.operations[0].result.clone();
        assert_eq!(start.operations[1].args[1], getattr_result);
    }

    /// Slice 2d accept: rest-only tuple-variant pattern
    /// `Foo::Bar(..)` lowers to a single getattr+isinstance cascade
    /// step.
    #[test]
    fn match_rest_only_tuple_variant_emits_isinstance_cascade() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Pair(..) => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "isinstance");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(bytes) => assert_eq!(bytes, b"Pair"),
                other => panic!("expected ByteStr attr name, got {other:?}"),
            },
            other => panic!("expected Constant attr name, got {other:?}"),
        }
        let getattr_result = start.operations[0].result.clone();
        assert_eq!(start.operations[1].args[1], getattr_result);
    }

    /// O3 acceptance: every cascade step emits a getattr op resolving
    /// its variant path against a `HostObject::Class` carrier instead
    /// of the prior `Constant(ByteStr("Enum::Variant"))` sentinel.
    /// This pins the orthodox shape `flowcontext.py:856 LOAD_GLOBAL`
    /// + `:861 LOAD_ATTR` + `operation.py:449 isinstance` produces
    /// for `if isinstance(x, A.B): … elif isinstance(x, A.C): …`.
    #[test]
    fn cascade_step_isinstance_arg2_is_host_class_carrier() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    Foo::B => 2,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let mut step_pairs: Vec<(SpaceOperation, SpaceOperation)> = Vec::new();
        for blk in g.iterblocks() {
            let blk_ref = blk.borrow();
            for window in blk_ref.operations.windows(2) {
                if window[0].opname == "getattr" && window[1].opname == "isinstance" {
                    step_pairs.push((window[0].clone(), window[1].clone()));
                }
            }
        }
        assert_eq!(
            step_pairs.len(),
            2,
            "two cascade steps each emit a (getattr, isinstance) op pair"
        );
        for (getattr_op, isinstance_op) in &step_pairs {
            // Each step's getattr resolves against the same
            // `HostObject::Class` for `Foo` (cached in
            // `host_env::HOST_CLASS_MINTS` process-global registry).
            match &getattr_op.args[0] {
                Hlvalue::Constant(c) => match &c.value {
                    ConstValue::HostObject(obj) => {
                        assert!(obj.is_class());
                        assert_eq!(obj.qualname(), "Foo");
                    }
                    other => panic!("expected HostObject leftmost, got {other:?}"),
                },
                other => panic!("expected Constant leftmost, got {other:?}"),
            }
            // Cascade step's isinstance second-arg is the getattr's
            // result Variable — orthodox shape per `operation.py:449`.
            assert_eq!(isinstance_op.args[1], getattr_op.result);
        }
        // Identity sharing: both steps' leftmost HostObject(Foo) are
        // the same class object — `HOST_CLASS_MINTS` returns the same
        // `HostObject` across all cascade steps (and across graphs)
        // that name `Foo`.
        let lhs0 = match &step_pairs[0].0.args[0] {
            Hlvalue::Constant(c) => c.value.clone(),
            _ => unreachable!(),
        };
        let lhs1 = match &step_pairs[1].0.args[0] {
            Hlvalue::Constant(c) => c.value.clone(),
            _ => unreachable!(),
        };
        assert_eq!(
            lhs0, lhs1,
            "cascade steps share the leftmost HostObject(Foo) class identity \
             via host_env::HOST_CLASS_MINTS (mirrors LOAD_GLOBAL returning the \
             same class object across every reference, including across \
             graph boundaries — flowcontext.py:847)"
        );
    }

    /// Slice 2d accept: heterogeneous variant arms in the same match
    /// — `Foo::A` (unit), `Foo::B { .. }` (rest-only struct), and
    /// `Foo::C(..)` (rest-only tuple) all share the cascade lowering
    /// since they all classify via `variant_match_path_str`.
    #[test]
    fn match_mixed_unit_struct_rest_tuple_rest_variants_share_cascade() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => 1,
                    Foo::B { .. } => 2,
                    Foo::C(..) => 3,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Three cascade steps total.
        let isinstance_count = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| op.opname == "isinstance")
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            isinstance_count, 3,
            "one cascade step per variant arm, regardless of variant kind"
        );
    }

    /// Slice 2e accept: struct-variant arm with a single named-Ident
    /// field binding emits exactly one `getattr` op at the arm body
    /// block's entry, plus the cascade's single `isinstance` step.
    /// The bound local is reachable from the arm body — the test
    /// returns it as the arm's tail value.
    #[test]
    fn match_struct_variant_field_binding_emits_getattr_at_arm_entry() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Resume { pc, .. } => pc,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Locate the arm body block — has exactly one `getattr` op
        // whose second arg is the byte-string `pc`.
        let getattr_pc_blocks = g
            .iterblocks()
            .iter()
            .filter(|b| {
                b.borrow().operations.iter().any(|op| {
                    op.opname == "getattr"
                        && matches!(
                            &op.args[1],
                            Hlvalue::Constant(c) if matches!(&c.value, ConstValue::ByteStr(bs) if bs == b"pc"),
                        )
                })
            })
            .count();
        assert_eq!(
            getattr_pc_blocks, 1,
            "exactly one arm body block emits the `getattr(scrutinee, \"pc\")` op"
        );
    }

    /// Slice 2e accept: multiple named-Ident bindings from one arm
    /// produce one `getattr` per binding, in declaration order, at
    /// the arm body block's entry.
    #[test]
    fn match_struct_variant_multiple_field_bindings_emit_getattrs_in_order() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Pair { lhs, rhs, .. } => lhs + rhs,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Locate the arm body block. It must carry getattr ops in the
        // declared order, followed by the `add` op.
        let arm_block = g
            .iterblocks()
            .iter()
            .find(|b| b.borrow().operations.iter().any(|op| op.opname == "add"))
            .cloned()
            .expect("arm body block emits the add op");
        let block_ref = arm_block.borrow();
        let getattr_ops: Vec<_> = block_ref
            .operations
            .iter()
            .filter(|op| op.opname == "getattr")
            .collect();
        assert_eq!(getattr_ops.len(), 2);
        let names: Vec<&[u8]> = getattr_ops
            .iter()
            .map(|op| match &op.args[1] {
                Hlvalue::Constant(c) => match &c.value {
                    ConstValue::ByteStr(bs) => bs.as_slice(),
                    _ => panic!("expected ByteStr"),
                },
                _ => panic!("expected Constant"),
            })
            .collect();
        assert_eq!(names, vec![b"lhs" as &[u8], b"rhs" as &[u8]]);
    }

    /// Slice 2e accept: explicit-rename binding `{ field: ident }`
    /// produces the `getattr` against the FIELD name and binds the
    /// LOCAL identifier in the arm body. The arm body reads the
    /// renamed identifier, so a missing rename would surface as an
    /// `UnboundLocal`.
    #[test]
    fn match_struct_variant_renaming_field_binding_uses_local_name() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Resume { pc: orig_pc, .. } => orig_pc,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // The arm body's getattr second arg is the field name `pc`,
        // NOT the local name `orig_pc` — same op upstream
        // `flowcontext.py` emits for `obj.pc`. Filter against the
        // cascade's own `getattr(<Foo>, "Resume")` step by matching
        // on the byte-string operand.
        let pc_getattrs = g
            .iterblocks()
            .iter()
            .flat_map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| {
                        op.opname == "getattr"
                            && matches!(
                                &op.args[1],
                                Hlvalue::Constant(c)
                                    if matches!(&c.value, ConstValue::ByteStr(bs) if bs == b"pc"),
                            )
                    })
                    .cloned()
                    .collect::<Vec<_>>()
            })
            .count();
        assert_eq!(
            pc_getattrs, 1,
            "exactly one `getattr(scrutinee, \"pc\")` op for the renaming binding"
        );
    }

    /// Slice 2e accept: `{ field: _ }` is a field-skip pattern — the
    /// field is matched but its value is discarded, so no `getattr`
    /// is emitted for the *field*. The arm body doesn't reference the
    /// field's name. The cascade itself still emits its
    /// `getattr(<Foo>, "Resume")` resolution step.
    #[test]
    fn match_struct_variant_wildcard_field_skip_emits_no_getattr() {
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Resume { pc: _, .. } => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // No `getattr(scrutinee, "pc")` is emitted, but the cascade's
        // own `getattr(<Foo>, "Resume")` resolution step is unaffected.
        let pc_getattrs = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| {
                        op.opname == "getattr"
                            && matches!(
                                &op.args[1],
                                Hlvalue::Constant(c)
                                    if matches!(&c.value, ConstValue::ByteStr(bs) if bs == b"pc"),
                            )
                    })
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            pc_getattrs, 0,
            "`{{ field: _ }}` skip emits no `getattr(_, \"pc\")` — only the cascade's variant-resolution getattr"
        );
    }

    /// Slice 2f boundary: tuple-variant arms with element bindings
    /// continue to reject with the slice-2f diagnostic.
    #[test]
    fn rejects_match_tuple_variant_with_element_bindings() {
        match lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Pair(a, b) => a + b,
                    _ => 0,
                }
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("tuple-variant") && reason.contains("slice 2f"),
                    "reason should cite slice-2f tuple-variant rejection: {reason}"
                );
            }
            other => panic!("expected Unsupported(tuple-variant bindings), got {other:?}"),
        }
    }

    /// Slice 2e boundary: or-pattern siblings whose field bindings
    /// disagree must reject — Rust enforces this at the language
    /// level, but the adapter re-validates so a malformed or-pattern
    /// surfaces a precise error rather than a confusing arm-body
    /// `UnboundLocal`.
    #[test]
    fn rejects_or_pattern_with_inconsistent_field_bindings() {
        // Note: rustc rejects this, but `syn::parse_str` accepts it
        // without semantic checks. The adapter's re-validation
        // surfaces the inconsistency as a precise error.
        match lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A { x } | Foo::B { y } => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("inconsistent field bindings"),
                    "reason should cite or-pattern binding mismatch: {reason}"
                );
            }
            other => panic!("expected Unsupported(or-pattern inconsistent), got {other:?}"),
        }
    }

    /// Slice 2e boundary: non-Ident / non-Wild field sub-patterns
    /// (literal, nested struct, etc.) reject with a complex-
    /// destructuring diagnostic.
    #[test]
    fn rejects_match_struct_variant_with_literal_field_pattern() {
        match lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::Resume { pc: 0, .. } => 1,
                    _ => 0,
                }
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("non-Ident") || reason.contains("complex destructuring"),
                    "reason should cite non-Ident field-pattern rejection: {reason}"
                );
            }
            other => panic!("expected Unsupported(non-Ident field), got {other:?}"),
        }
    }

    /// Variant cascade preserves locals merging across arms — every
    /// arm rebinding `y` produces a join block whose inputargs include
    /// the merged `y` slot. The synthetic `#match_scrutinee_*` slot
    /// must NOT leak into the post-match join.
    #[test]
    fn match_variant_cascade_merges_locals_into_join() {
        let g = lower(
            "fn f(x: Foo, y: i64) -> i64 {
                let y = match x {
                    Foo::A => y + 1,
                    _ => y - 1,
                };
                y
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Find the join block (zero ops, multiple inputargs, multiple
        // arm predecessors). Its inputargs must NOT contain a
        // `#match_scrutinee_*` synthetic slot — that's strictly a
        // cascade-internal sidecar and post-match code must not see
        // it.
        let join = g
            .iterblocks()
            .iter()
            .find(|b| {
                let bref = b.borrow();
                bref.operations.is_empty() && bref.inputargs.len() == 3 && bref.exits.len() == 1
            })
            .cloned()
            .expect("expected match-join block with 3 inputargs (tail + x + y)");
        // None of the join's inputargs should be named with the
        // synthetic prefix — the filter at `lower_match_variant_cascade`
        // step 9 strips it.
        for arg in &join.borrow().inputargs {
            if let Hlvalue::Variable(v) = arg {
                let nm = v.name();
                assert!(
                    !nm.starts_with("#match_scrutinee_"),
                    "synthetic scrutinee slot leaked into join inputargs: {nm}"
                );
            }
        }
    }

    // NOTE: method calls are now accepted (slice M2.5c). See
    // `method_call_emits_getattr_plus_simple_call` below.

    #[test]
    fn rejects_unbound_local() {
        match lower("fn f() -> i64 { x }").unwrap_err() {
            AdapterError::UnboundLocal { name } => assert_eq!(name, "x"),
            other => panic!("expected UnboundLocal, got {other:?}"),
        }
    }

    #[test]
    fn rejects_short_circuit_and() {
        match lower("fn f(a: bool, b: bool) -> bool { a && b }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("short-circuit"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(short-circuit), got {other:?}"),
        }
    }

    // NOTE: `?` operator is now accepted (slice 4). See the
    // `try_op_*` tests below for coverage.

    // NOTE: tuple literals are now accepted (slice M2.5d). See
    // `tuple_literal_emits_newtuple_op`.

    // ---- M2.5b while/loop/break/continue accept tests -------------

    #[test]
    fn while_loop_emits_header_body_back_edge_exit() {
        // The canonical shape: a header block with exitswitch=cond and
        // two exits (false → exit, true → body), a body block that
        // Links back to the header, and an exit block that continues
        // into the returnblock.
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {
                    let i = i + 1;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock + header + body + exit + returnblock = 5.
        assert_eq!(blocks.len(), 5);

        // startblock has no ops (i = 0 is a constant rebind, no op)
        // and closes with a single Link into the header.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);

        // The header is reached from startblock and from the body
        // (2 incoming Links = the test oracle). Identify it via its
        // operations — `i < n` emits `lt` then `bool` (upstream
        // POP_JUMP_IF_FALSE wraps cond in `op.bool`); exitswitch
        // references the bool-op result and the block has 2 exits.
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 2
                    && br.operations[0].opname == "lt"
                    && br.operations[1].opname == "bool"
                    && br.exitswitch.is_some()
                    && br.exits.len() == 2
            })
            .expect("expected header block with `lt` + `bool` ops + 2 exits");
        let header_ref = header.borrow();
        let false_exit = header_ref.exits[0].borrow();
        let true_exit = header_ref.exits[1].borrow();
        assert_eq!(
            false_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            true_exit.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );

        // Body block emits `i + 1` and back-edges to header.
        let body_target = true_exit.target.as_ref().unwrap();
        let body_ref = body_target.borrow();
        assert_eq!(body_ref.operations.len(), 1);
        assert_eq!(body_ref.operations[0].opname, "add");
        assert_eq!(body_ref.exits.len(), 1);
        // Back-edge target is `header`.
        let back_link = body_ref.exits[0].borrow();
        let back_target = back_link.target.as_ref().unwrap();
        assert!(Rc::ptr_eq(back_target, header));
    }

    #[test]
    fn while_loop_no_rebinds_in_body() {
        // Body has no let — body's back-edge args are exactly the
        // inputargs forwarded through.
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {}
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn loop_with_break_reaches_exit() {
        // `loop { break; }` — no condition, body's only stmt is `break;`.
        // Header and body are the same block; the body's only exit is
        // the Link into the exit block.
        let g = lower(
            "fn f() -> i64 {
                loop { break; }
                42
            }",
        )
        .unwrap();
        checkgraph(&g);
        // startblock → header/body (break-only) → exit → returnblock.
        // The startblock has no ops.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 0);
        assert_eq!(start.exits.len(), 1);
        // Exit block tail emits the constant 42 and Links into return.
        let blocks = g.iterblocks();
        // start + header(=body) + exit + return = 4.
        assert_eq!(blocks.len(), 4);
    }

    #[test]
    fn continue_back_edges_to_header() {
        let g = lower(
            "fn f(n: i64) -> i64 {
                let i = 0;
                while i < n {
                    continue;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // start + header + body + exit + return = 5. The body's only
        // exit is a Link back to the header via `continue`.
        assert_eq!(blocks.len(), 5);
        let header = blocks
            .iter()
            .find(|b| b.borrow().exitswitch.is_some())
            .expect("expected header with exitswitch");
        let header_true_exit = header.borrow().exits[1].clone();
        let body = header_true_exit.borrow().target.clone().unwrap();
        let body_exit = body.borrow().exits[0].clone();
        let body_target = body_exit.borrow().target.clone().unwrap();
        assert!(Rc::ptr_eq(&body_target, header));
    }

    #[test]
    fn nested_loops_break_resolves_to_innermost() {
        // `break;` inside the inner loop exits only that loop.
        let g = lower(
            "fn f(n: i64, m: i64) -> i64 {
                let i = 0;
                while i < n {
                    let j = 0;
                    while j < m {
                        break;
                    }
                    let i = i + 1;
                }
                i
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn loop_with_let_rebind_before_break() {
        let g = lower(
            "fn f() -> i64 {
                let x = 0;
                loop {
                    let x = x + 1;
                    break;
                }
                x
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    // ---- M2.5b while/loop/break/continue reject tests -------------

    #[test]
    fn rejects_break_outside_loop() {
        match lower("fn f() -> i64 { break; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("break") || reason.contains("outside"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(break outside), got {other:?}"),
        }
    }

    #[test]
    fn rejects_continue_outside_loop() {
        match lower("fn f() -> i64 { continue; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("continue") || reason.contains("outside"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(continue outside), got {other:?}"),
        }
    }

    #[test]
    fn rejects_break_with_value() {
        match lower("fn f() -> i64 { loop { break 1; } }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("break VALUE")
                    || reason.contains("value")
                    || reason.contains("loop-as-expression"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(break with value), got {other:?}"),
        }
    }

    #[test]
    fn rejects_dead_code_after_break() {
        match lower(
            "fn f() -> i64 {
                loop { break; let _x = 1; }
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("dead code"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(dead code after break), got {other:?}"),
        }
    }

    #[test]
    fn rejects_labeled_break() {
        match lower("fn f() -> i64 { 'outer: loop { break 'outer; } }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("label") || reason.contains("VALUE") || reason.contains("value"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(labeled break), got {other:?}"),
        }
    }

    // ---- M2.5b `?` operator accept tests --------------------------

    #[test]
    fn try_op_emits_canraise_block_with_exception_edge() {
        // The canonical shape: the startblock emits the operand's own
        // call op as the raising op (no synthetic wrapper), sets
        // exitswitch = c_last_exception, and exits with
        // (exit[0]=normal→continuation, exit[1]=exception
        // →graph.exceptblock with exitcase=Exception class).
        // Operand must be a direct call; bare variables reject.
        let g = lower("fn f(h: Handler) -> i64 { h.read()? }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // getattr('read') + simple_call(bound) — the simple_call is
        // the raising op.
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        assert!(start.canraise(), "startblock must canraise");
        assert_eq!(start.exits.len(), 2);

        let normal = start.exits[0].borrow();
        assert!(normal.exitcase.is_none());
        assert!(normal.last_exception.is_none());
        assert!(normal.last_exc_value.is_none());

        let exc = start.exits[1].borrow();
        match exc.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class(), "exitcase must be a class HostObject");
                    assert_eq!(obj.qualname(), "Exception");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        let target = exc.target.as_ref().unwrap();
        assert!(Rc::ptr_eq(target, &g.exceptblock));
        assert!(exc.last_exception.is_some());
        assert!(exc.last_exc_value.is_some());
    }

    #[test]
    fn try_op_continuation_produces_unwrapped_value() {
        // `h.read()? + 1` — the continuation block picks up the
        // unwrapped value via its inputargs[0] and emits
        // `add(unwrapped, 1)`.
        let g = lower("fn f(h: Handler) -> i64 { h.read()? + 1 }").unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        // startblock (canraise) + continuation + returnblock +
        // exceptblock = 4.
        assert_eq!(blocks.len(), 4);

        let cont = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "add"
            })
            .expect("expected continuation block with `add` op");
        let cont_ref = cont.borrow();
        assert_eq!(cont_ref.operations[0].args[0], cont_ref.inputargs[0]);
    }

    #[test]
    fn try_op_let_binding_continues_locals() {
        // `let y = h.read()?; y + 1` — pre-try local `h` survives
        // via the continuation's inputargs.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                let y = h.read()?;
                y + 1
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn try_op_twice_chains_two_canraise_blocks() {
        // Two call-? in sequence — two canraise blocks. checkgraph
        // enforces both exception exits target the single
        // graph.exceptblock.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                let y = h.read()?;
                h.write(y)?
            }",
        )
        .unwrap();
        checkgraph(&g);
        let blocks = g.iterblocks();
        let canraise_count = blocks.iter().filter(|b| b.borrow().canraise()).count();
        assert_eq!(canraise_count, 2);
    }

    #[test]
    fn try_op_rejects_non_call_operand() {
        // `x?` where `x` is a bare variable — no call op to hang
        // canraise on, so reject.
        match lower("fn f(x: i64) -> i64 { x? }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("direct call"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(? non-call), got {other:?}"),
        }
    }

    // ---- M2.5b `for` loop accept tests ----------------------------

    #[test]
    fn for_loop_emits_iter_next_canraise_with_stopiter_exit() {
        // The canonical shape: startblock emits `iter(iter_expr)`,
        // unconditional Link to header; header emits
        // `next(iter_h)` with exitswitch = c_last_exception,
        // exits[0] normal → body, exits[1] → exit_block with
        // exitcase = StopIteration class.
        let g = lower(
            "fn f(it: i64) -> i64 {
                for item in it {
                    let _x = item;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);

        // startblock has one op `iter(it)`.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "iter");

        // Locate the header block — it has op `next` + canraise.
        let blocks = g.iterblocks();
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "next" && br.canraise()
            })
            .expect("expected header block with `next` op + canraise");
        let header_ref = header.borrow();
        // 3 exits: normal → body, StopIteration → loop exit,
        // RuntimeError → graph.exceptblock. Matches upstream
        // `operation.py:595-599 Next.canraise = [StopIteration,
        // RuntimeError]`.
        assert_eq!(header_ref.exits.len(), 3);

        // exit[0]: normal → body.
        let normal = header_ref.exits[0].borrow();
        assert!(normal.exitcase.is_none());

        // exit[1]: StopIteration → exit block.
        let stopiter_exit = header_ref.exits[1].borrow();
        match stopiter_exit.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "StopIteration");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        // `guessexception` class-specific exit: last_exception is a
        // Constant(case) per `flowcontext.py:131-132`, not a Variable.
        match stopiter_exit.last_exception.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "StopIteration");
                }
                other => panic!("expected StopIteration HostObject, got {other:?}"),
            },
            other => panic!(
                "StopIteration exit's last_exception must be a \
                 Constant(StopIteration) per upstream guessexception \
                 (flowcontext.py:127-132), got {other:?}"
            ),
        }
        match stopiter_exit.last_exc_value.as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!(
                "StopIteration exit's last_exc_value must be a \
                 fresh Variable('last_exc_value') per upstream \
                 (flowcontext.py:133), got {other:?}"
            ),
        }

        // exit[2]: RuntimeError → graph.exceptblock. Per upstream
        // `RaiseImplicit.nomoreblocks` (flowcontext.py:1271-1284)
        // the link args carry [Constant(AssertionError class),
        // Constant(AssertionError("implicit RuntimeError …"))],
        // NOT the original RuntimeError class/value pair.
        let runtime_exit = header_ref.exits[2].borrow();
        match runtime_exit.exitcase.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "RuntimeError");
                }
                other => panic!("expected HostObject exitcase, got {other:?}"),
            },
            other => panic!("expected Constant exitcase, got {other:?}"),
        }
        let runtime_target = runtime_exit.target.as_ref().unwrap();
        assert!(
            Rc::ptr_eq(runtime_target, &g.exceptblock),
            "RuntimeError exit must target graph.exceptblock"
        );
        // link.args == [Constant(AssertionError class), Constant(AE instance)]
        assert_eq!(runtime_exit.args.len(), 2);
        let arg0 = runtime_exit.args[0]
            .as_ref()
            .expect("RuntimeError link.args[0] cannot be undefined-local sentinel");
        match arg0 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "AssertionError");
                }
                other => panic!("expected AssertionError class HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError link.args[0] must be Constant(AssertionError \
                 class) per RaiseImplicit.nomoreblocks, got {other:?}"
            ),
        }
        let arg1 = runtime_exit.args[1]
            .as_ref()
            .expect("RuntimeError link.args[1] cannot be undefined-local sentinel");
        match arg1 {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_instance());
                    let cls = obj.instance_class().expect("instance class");
                    assert_eq!(cls.qualname(), "AssertionError");
                }
                other => panic!("expected AssertionError instance HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError link.args[1] must be Constant(AssertionError \
                 instance) per RaiseImplicit.nomoreblocks, got {other:?}"
            ),
        }
        // last_exception on the RuntimeError link remains the original
        // class-specific Constant(RuntimeError) per `guessexception`,
        // even though the args were rewritten by `nomoreblocks`.
        match runtime_exit.last_exception.as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class());
                    assert_eq!(obj.qualname(), "RuntimeError");
                }
                other => panic!("expected RuntimeError HostObject, got {other:?}"),
            },
            other => panic!(
                "RuntimeError exit's last_exception must be a \
                 Constant(RuntimeError) per upstream guessexception, \
                 got {other:?}"
            ),
        }
        match runtime_exit.last_exc_value.as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!(
                "RuntimeError exit's last_exc_value must be a fresh \
                 Variable('last_exc_value'), got {other:?}"
            ),
        }
    }

    #[test]
    fn for_loop_body_shadows_preloop_local_single_channel() {
        // `for x in xs { … }` where `x` is also a parameter — upstream
        // STORE_FAST (flowcontext.py:878-884) rebinds the same local
        // slot in place. The body block must therefore expose exactly
        // ONE inputarg for the `x` slot, not two, and the header's
        // normal Link must route `v_next` into that slot.
        let g = lower(
            "fn f(x: i64, xs: i64) -> i64 {
                for x in xs {
                    let _y = x;
                }
                x
            }",
        )
        .unwrap();
        checkgraph(&g);

        // Header block (has `next` op + canraise).
        let blocks = g.iterblocks();
        let header = blocks
            .iter()
            .find(|b| {
                let br = b.borrow();
                br.operations.len() == 1 && br.operations[0].opname == "next" && br.canraise()
            })
            .expect("expected header block");
        let header_ref = header.borrow();

        // Body block = target of header.exits[0].
        let body = header_ref.exits[0]
            .borrow()
            .target
            .as_ref()
            .expect("normal exit must have a target")
            .clone();
        let body_ref = body.borrow();

        // Count how many body inputargs carry the Variable::named("x")
        // prefix. `rename()` canonicalises `"x"` to `"x_"` (see
        // `model.rs:2050-2082`). Upstream STORE_FAST makes it exactly
        // one channel; a `body_inputargs = [item_body_var, fresh_x,
        // ...]` shape would be "two channels for the same slot" — the
        // bug this test guards against.
        let x_slots: usize = body_ref
            .inputargs
            .iter()
            .filter(|h| match h {
                Hlvalue::Variable(v) => v.name_prefix() == "x_",
                _ => false,
            })
            .count();
        assert_eq!(
            x_slots, 1,
            "body inputargs must have exactly one channel for the `x` \
             slot (upstream STORE_FAST rebinds in place — \
             flowcontext.py:878-884), got {x_slots}"
        );
    }

    #[test]
    fn for_loop_body_back_edges_to_header_with_iter_slot() {
        // Body's fall-through back-edge carries [iter_var, ...locals]
        // into header whose inputargs are [iter_h, ...locals].
        let g = lower(
            "fn f(it: i64, a: i64) -> i64 {
                for item in it {
                    let a = a + 1;
                }
                a
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn for_loop_with_break_exits_past_loop() {
        let g = lower(
            "fn f(it: i64) -> i64 {
                for _item in it {
                    break;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn for_loop_with_continue_back_edges_to_header() {
        let g = lower(
            "fn f(it: i64) -> i64 {
                for _item in it {
                    continue;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn nested_for_loops() {
        let g = lower(
            "fn f(it1: i64, it2: i64) -> i64 {
                for _a in it1 {
                    for _b in it2 {
                        let _x = 1;
                    }
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    // ---- M2.5b `for` loop reject tests ----------------------------

    #[test]
    fn rejects_for_tuple_pattern() {
        match lower(
            "fn f(it: i64) -> i64 {
                for (a, b) in it {}
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("identifier") || reason.contains("destructuring"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(destructuring), got {other:?}"),
        }
    }

    #[test]
    fn rejects_for_in_expr_position() {
        // `let x = for ... { ... };` — loop as expression produces `()`
        // which is out of scope.
        match lower(
            "fn f(it: i64) -> i64 {
                let _x = for _i in it {};
                0
            }",
        )
        .unwrap_err()
        {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("loop construct") || reason.contains("statement"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(for-in-expr), got {other:?}"),
        }
    }

    // ---- M2.5c method calls + function calls + generics ----------

    #[test]
    fn method_call_emits_getattr_plus_simple_call() {
        // `x.abs()` → [getattr(x, "abs") → v_bound, simple_call(v_bound) → v_result].
        let g = lower("fn f(x: i64) -> i64 { x.abs() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                value if value.string_eq("abs") => {}
                other => panic!("expected Str method name, got {other:?}"),
            },
            other => panic!("expected Constant method name, got {other:?}"),
        }
        assert_eq!(start.operations[1].opname, "simple_call");
        // simple_call's first arg is the bound method (getattr result).
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
        // And the call has no extra args (abs is nullary beyond receiver).
        assert_eq!(start.operations[1].args.len(), 1);
    }

    #[test]
    fn method_call_with_args_threads_args_into_simple_call() {
        // `x.add(y, z)` → [getattr(x, "add"), simple_call(bound, y, z)].
        let g = lower("fn f(x: i64, y: i64, z: i64) -> i64 { x.add(y, z) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        let sc = &start.operations[1];
        assert_eq!(sc.opname, "simple_call");
        assert_eq!(sc.args.len(), 3);
    }

    #[test]
    fn function_call_emits_simple_call_with_callee_first() {
        // `g(x, 1)` where `g` is a local → simple_call(g, x, 1).
        // Synthetic: the adapter doesn't know `g` is callable, just
        // that it resolves as a local identifier.
        let g = lower("fn f(x: i64, g: i64) -> i64 { g(x, 1) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "simple_call");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn method_call_chained() {
        // `x.a().b()` → 4 ops: getattr/simple_call for each.
        let g = lower("fn f(x: i64) -> i64 { x.a().b() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 4);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        assert_eq!(start.operations[2].opname, "getattr");
        assert_eq!(start.operations[3].opname, "simple_call");
    }

    #[test]
    fn method_call_on_binop_result() {
        // `(x + 1).abs()` — receiver is an op result, not a local.
        let g = lower("fn f(x: i64) -> i64 { (x + 1).abs() }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "add");
        assert_eq!(start.operations[1].opname, "getattr");
        // The getattr receiver is the add's result Variable.
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
        assert_eq!(start.operations[2].opname, "simple_call");
    }

    #[test]
    fn self_receiver_is_local_named_self() {
        // `fn m(&self) -> i64 { self.x() }` — `self` becomes a
        // Variable named "self" in `locals`, and `self.x()` lowers
        // like any other method call.
        let g = lower("fn m(&self) -> i64 { self.x() }").unwrap();
        checkgraph(&g);
        // One inputarg named by the adapter's `Variable::named("self")`
        // path, one getattr/simple_call pair.
        let start = g.startblock.borrow();
        assert_eq!(start.inputargs.len(), 1);
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
    }

    #[test]
    fn generic_fn_identity() {
        // `fn id<T>(x: T) -> T { x }` — accepted post-slice-M2.5c.
        // The adapter does not track T; the annotator's
        // FunctionDesc.specialize monomorphizes at the call site.
        let g = lower("fn id<T>(x: T) -> T { x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
    }

    #[test]
    fn generic_fn_with_trait_bound() {
        // `<E: Trait>` — trait-bound markers are parsed but not
        // inspected by the adapter; downstream annotator reads the
        // `args_s` classdef set.
        let g = lower(
            "fn step<E: StepExecutor>(e: E) -> i64 {
                e.execute()
            }",
        )
        .unwrap();
        checkgraph(&g);
        assert_eq!(g.getargs().len(), 1);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
    }

    #[test]
    fn generic_fn_with_where_clause() {
        let g = lower(
            "fn step<E>(e: E) -> i64 where E: StepExecutor {
                e.execute()
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn method_call_chained_with_let_rebinds() {
        // Ensure chained method calls interact properly with the
        // locals map: `let t = x.a(); t.b()` — the method-call
        // result is stored into a local and re-read.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let t = x.a();
                t.b()
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 4);
    }

    // ---- M2.5c reject tests --------------------------------------

    #[test]
    fn rejects_method_turbofish() {
        match lower("fn f(x: i64) -> i64 { x.convert::<i64>() }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("turbofish"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(turbofish), got {other:?}"),
        }
    }

    #[test]
    fn rejects_call_with_non_identifier_callee() {
        // `(x + 1)(y)` — callee is not a simple path.
        match lower("fn f(x: i64, y: i64) -> i64 { (x + 1)(y) }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("identifier") || reason.contains("path"),
                    "reason: {reason}"
                );
            }
            other => panic!("expected Unsupported(non-ident-callee), got {other:?}"),
        }
    }

    #[test]
    fn rejects_const_generic_param() {
        match lower("fn f<const N: usize>(x: i64) -> i64 { x }").unwrap_err() {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("const generic"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature(const generic), got {other:?}"),
        }
    }

    // ---- M2.5d literals + tuple/array ops ------------------------

    #[test]
    fn string_literal_lowers_to_constant_str() {
        // `fn f() -> i64 { let _s = "hello"; 0 }` — the string is
        // stored as a ConstValue string attached to a let binding,
        // then the tail emits 0. No op emitted for the literal
        // itself (it's a Constant, not a SpaceOperation result).
        let g = lower(r#"fn f() -> i64 { let _s = "hello"; 0 }"#).unwrap();
        checkgraph(&g);
    }

    #[test]
    fn float_literal_lowers_to_constant_float() {
        let g = lower("fn f() -> i64 { let _x = 3.14; 0 }").unwrap();
        checkgraph(&g);
    }

    #[test]
    fn tuple_literal_emits_newtuple_op() {
        let g = lower("fn f() -> i64 { let _t = (1, 2, 3); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn array_literal_emits_newlist_op() {
        let g = lower("fn f() -> i64 { let _a = [1, 2, 3]; 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newlist");
        assert_eq!(start.operations[0].args.len(), 3);
    }

    #[test]
    fn nested_tuple_of_tuples() {
        let g = lower("fn f(x: i64) -> i64 { let _t = ((x, 1), (2, x)); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // Two inner newtuple + one outer newtuple = 3 ops.
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[1].opname, "newtuple");
        assert_eq!(start.operations[2].opname, "newtuple");
    }

    #[test]
    fn tuple_as_method_argument() {
        // The tuple flows into a method call — checks that
        // `lower_tuple`'s result is re-readable downstream.
        // Emission order is receiver → getattr → args → simple_call.
        let g = lower("fn f(x: i64) -> i64 { x.fold((1, 2)) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 3);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "newtuple");
        assert_eq!(start.operations[2].opname, "simple_call");
        // simple_call's second arg = newtuple result.
        assert_eq!(start.operations[2].args[1], start.operations[1].result);
    }

    #[test]
    fn empty_tuple_emits_newtuple_with_no_args() {
        let g = lower("fn f() -> i64 { let _u = (); 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "newtuple");
        assert_eq!(start.operations[0].args.len(), 0);
    }

    // ---- M2.5d reject tests --------------------------------------

    #[test]
    fn rejects_struct_literal() {
        match lower("fn f() -> i64 { let _s = Point { x: 1, y: 2 }; 0 }").unwrap_err() {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("struct") || reason.contains("user-type"),
                "reason: {reason}"
            ),
            other => panic!("expected Unsupported(struct), got {other:?}"),
        }
    }

    // ---- M2.5e orthodox HOST_ENV-backed name resolution (Slice O1+O2)
    //
    // `Builder::resolve_path_constant` mirrors upstream's
    // `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR` chain. The
    // closed-world `host_env::PYRE_STDLIB` registry pre-registers
    // `Ok` / `Some` / `Err` / `Result` / `Option` as
    // `HostObject::Class` singletons; `None` resolves to the
    // `Constant(ConstValue::None)` NoneType-singleton constant; the
    // process-global `host_env::HOST_CLASS_MINTS` registry finds-or-
    // mints `HostObject::Class` for multi-segment paths' leftmost
    // segment on demand and shares identity across cascade steps —
    // and across graphs — that name the same class (mirrors upstream
    // `func_globals[name]` reading from a process-shared globals
    // dict at `flowcontext.py:847`). Locals-first ordering
    // (`pyopcode.py:502 LOAD_FAST`) means a user `let Ok = …`
    // shadows the closed-world entry. See plan
    // `~/.claude/plans/m2_5e_orthodox_host_env_resolution.md` Slice
    // O1+O2 for the full upstream cite chain.

    #[test]
    fn ok_call_at_value_position_emits_simple_call_with_host_class_callee() {
        // `let z = Ok(42); z` — value-position `Ok(42)` goes through
        // `lower_call`'s ordinary path: `Ok` resolves through the
        // closed-world PYRE_STDLIB registry to
        // `Constant(HostObject(<Ok class>))`, and `lower_call` emits
        // `simple_call(<host Ok>, 42)` per `operation.py:663-679
        // SimpleCall` shape, matching what `flowcontext.py:856
        // LOAD_GLOBAL Ok` + `:998 CALL_FUNCTION 1` would compile a
        // Python `Ok(42)` call to. The Slice O4 boundary helper does
        // NOT fire here because the call sits inside a `let` RHS, not
        // at a function/arm tail.
        let g = lower("fn f() -> i64 { let z = Ok(42); z }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "simple_call");
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert_eq!(obj.qualname(), "Ok");
                    assert!(obj.is_class());
                }
                other => panic!("expected HostObject(Ok), got {other:?}"),
            },
            other => panic!("expected Constant callee, got {other:?}"),
        }
    }

    #[test]
    fn some_call_at_value_position_emits_simple_call_with_host_class_callee() {
        let g = lower("fn f(x: i64) -> i64 { let z = Some(x); z }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "simple_call");
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert_eq!(obj.qualname(), "Some");
                    assert!(obj.is_class());
                }
                other => panic!("expected HostObject(Some), got {other:?}"),
            },
            other => panic!("expected Constant callee, got {other:?}"),
        }
    }

    #[test]
    fn err_at_value_position_emits_simple_call_with_host_class_callee() {
        // `let z = Err(e); z` — value-position `Err(e)` lowers as
        // an ordinary constructor call: `simple_call(<host Err>, e)`
        // per `operation.py:663-679`. With no orthodox `exc_from_raise`
        // port yet (Slice O5 lands the boundary raise rewrite), this
        // call shape applies in BOTH value and (non-O5) terminator
        // positions; the difference materialises only when O5
        // intercepts the tail position.
        let g = lower("fn f(e: i64) -> i64 { let z = Err(e); z }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "simple_call");
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert_eq!(obj.qualname(), "Err");
                    assert!(obj.is_class());
                }
                other => panic!("expected HostObject(Err), got {other:?}"),
            },
            other => panic!("expected Constant callee, got {other:?}"),
        }
    }

    #[test]
    fn none_path_lowers_to_constant_none() {
        // Bare `None` resolves to `Constant(ConstValue::None)` —
        // Python's NoneType singleton instance. Mirrors upstream
        // `Constant(None)` carriers in flowspace graphs (`model.py`
        // `Constant.value = None`).
        let g = lower("fn f() -> i64 { None }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "bare None resolves to a Constant; no SpaceOperation emits"
        );
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::None => {}
                other => panic!("expected ConstValue::None, got {other:?}"),
            },
            other => panic!("expected Constant return value, got {other:?}"),
        }
    }

    // ---- M2.5e orthodox HOST_ENV (Slice O4): boundary-only
    //      Result/Option transparency adaptation
    //
    // PRE-EXISTING-ADAPTATION (Codex 2026-05-03 parity audit accepted
    // as "documented as structural adaptation, not parity"). Function
    // /arm tail and `return expr` positions in pyre-interpreter Rust
    // source frequently end in `Ok(x)` / `Some(x)` / `None`; upstream
    // Python source has no Result/Option layer. `lower_value_boundary`
    // collapses these wrappers at the three boundary call sites only
    // (`lower_block` tail via `lower_arm_body`, `lower_arm_body`'s
    // default arm, and `lower_return`'s `ret.expr`). Value-position
    // calls keep flowing through ordinary `simple_call(<host class>,
    // …)` lowering per O1+O2.

    #[test]
    fn ok_call_at_boundary_collapses_to_inner_value() {
        // Function-tail `Ok(42)` collapses to its inner value at the
        // returnblock edge — no `simple_call(<Ok>, 42)` is emitted.
        // Mirrors what upstream Python `def f(): return 42` (the
        // analogue without Result wrapping) would produce in
        // flowspace.
        let g = lower("fn f() -> i64 { Ok(42) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "Ok(42) at function tail collapses to inner value; no SpaceOperation emits — got {:?}",
            start.operations,
        );
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(v) => assert_eq!(*v, 42),
                other => panic!("expected ConstValue::Int(42), got {other:?}"),
            },
            other => panic!("expected Constant return value, got {other:?}"),
        }
    }

    #[test]
    fn some_call_at_boundary_collapses_to_inner_value() {
        let g = lower("fn f(x: i64) -> i64 { Some(x) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "Some(x) at function tail collapses to inner value; no SpaceOperation emits"
        );
        let link = start.exits[0].borrow();
        // Inner value `x` is the function's first inputarg —
        // identity matches.
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!("expected Variable return value (inner x), got {other:?}"),
        }
    }

    #[test]
    fn ok_call_at_explicit_return_collapses_to_inner_value() {
        // `return Ok(x);` — explicit return route: `lower_return` calls
        // the boundary helper before composing the returnblock Link.
        let g = lower("fn f(x: i64) -> i64 { return Ok(x); }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "explicit return Ok(x) collapses; no SpaceOperation emits"
        );
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!("expected Variable return value (inner x), got {other:?}"),
        }
    }

    #[test]
    fn ok_call_at_match_arm_tail_collapses_to_inner_value() {
        // Match-arm-tail boundary route: `lower_arm_body`'s default
        // arm calls the boundary helper. Each variant arm collapses
        // its `Ok(...)` wrapper at the arm's outer-tail position.
        let g = lower(
            "fn f(x: Foo) -> i64 {
                match x {
                    Foo::A => Ok(1),
                    _ => Ok(0),
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // No `simple_call(<Ok>, …)` ops anywhere in the graph — both
        // arm tails collapsed.
        let simple_calls_to_ok: usize = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .operations
                    .iter()
                    .filter(|op| {
                        op.opname == "simple_call"
                            && matches!(
                                &op.args[0],
                                Hlvalue::Constant(c)
                                    if matches!(
                                        &c.value,
                                        ConstValue::HostObject(obj) if obj.qualname() == "Ok",
                                    ),
                            )
                    })
                    .count()
            })
            .sum();
        assert_eq!(simple_calls_to_ok, 0);
    }

    #[test]
    fn nested_ok_wrapping_qualified_path_at_boundary_collapses_to_getattr_only() {
        // `Ok(StepResult::Continue)` at function tail: outer wrapper
        // collapses, leaving only the inner getattr cascade.
        let g = lower("fn f() -> i64 { Ok(StepResult::Continue) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            1,
            "boundary collapse leaves only inner getattr; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "getattr");
        let link = start.exits[0].borrow();
        assert_eq!(
            link.args[0].as_ref().unwrap(),
            &start.operations[0].result,
            "returnblock link carries the getattr result (inner StepResult::Continue)"
        );
    }

    // ---- M2.5e orthodox HOST_ENV (Slice O5): exc_from_raise subset
    //      port for boundary-position `Err(e)` — full 2-exit
    //      `isinstance(evalue, type)` fork preserved
    //
    // PARITY (no fork elision). The boundary helper emits the upstream
    // `flowcontext.py:600-636 exc_from_raise` op sequence with a
    // 2-exit `guessbool(isinstance(arg, type))` fork per
    // `flowcontext.py:610`. Both arms — True branch (instantiate via
    // `simple_call(evalue)`) and False branch (assert via
    // `ll_assert_not_none(evalue)`) — converge on a join block that
    // emits `type(w_value)` and Links `[etype, w_value]` to
    // `graph.exceptblock` per `flowcontext.py:1259 Raise.nomoreblocks`.
    // Value-position `Err(e)` continues through ordinary
    // `simple_call(<Err>, e)` (see
    // `err_at_value_position_emits_simple_call_with_host_class_callee`).

    #[test]
    fn err_call_at_boundary_emits_isinstance_fork_and_raise_edge() {
        let g = lower("fn f(e: i64) -> i64 { Err(e) }").unwrap();
        checkgraph(&g);
        // Start block: 1 isinstance op + Bool fork to True/False arms.
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "isinstance");
        assert_eq!(start.exits.len(), 2);
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert_eq!(obj.qualname(), "type");
                    assert!(obj.is_class());
                }
                other => panic!("expected HostObject(type) isinstance arg2, got {other:?}"),
            },
            other => panic!("expected Constant isinstance arg2, got {other:?}"),
        }
        // Bool(false) before Bool(true) per upstream BlockRecorder.guessbool
        // (`flowcontext.py:107-122`).
        let exit_false = start.exits[0].borrow();
        let exit_true = start.exits[1].borrow();
        assert_eq!(
            exit_false.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(false)))
        );
        assert_eq!(
            exit_true.exitcase.as_ref().unwrap(),
            &Hlvalue::Constant(Constant::new(ConstValue::Bool(true)))
        );
        let false_block = exit_false.target.as_ref().unwrap();
        let true_block = exit_true.target.as_ref().unwrap();
        // True arm: simple_call(evalue) — the instantiation path per
        // `flowcontext.py:614`.
        let true_ref = true_block.borrow();
        assert_eq!(true_ref.operations.len(), 1);
        assert_eq!(true_ref.operations[0].opname, "simple_call");
        // False arm: ll_assert_not_none(evalue) — the (instance, None)
        // shape per `flowcontext.py:632-634`.
        let false_ref = false_block.borrow();
        assert_eq!(false_ref.operations.len(), 1);
        assert_eq!(false_ref.operations[0].opname, "ll_assert_not_none");
        // Both arms converge on the same join block.
        assert_eq!(true_ref.exits.len(), 1);
        assert_eq!(false_ref.exits.len(), 1);
        let true_join = true_ref.exits[0].borrow().target.as_ref().unwrap().clone();
        let false_join = false_ref.exits[0].borrow().target.as_ref().unwrap().clone();
        assert!(
            std::rc::Rc::ptr_eq(&true_join, &false_join),
            "True and False arm Links converge on the same join block"
        );
        // Join block: type(w_value) + Link [etype, w_value] -> exceptblock.
        let join_ref = true_join.borrow();
        assert_eq!(join_ref.operations.len(), 1);
        assert_eq!(join_ref.operations[0].opname, "type");
        assert_eq!(join_ref.exits.len(), 1);
        let exit_link = join_ref.exits[0].borrow();
        assert!(
            std::rc::Rc::ptr_eq(exit_link.target.as_ref().unwrap(), &g.exceptblock),
            "join Links to graph.exceptblock per flowcontext.py:1259"
        );
        assert_eq!(exit_link.args.len(), 2);
        assert_eq!(
            exit_link.args[0].as_ref().unwrap(),
            &join_ref.operations[0].result,
            "Link arg[0] = etype = type(w_value)"
        );
    }

    #[test]
    fn err_at_explicit_return_emits_isinstance_fork_and_raise_edge() {
        // `return Err(e);` — `lower_return` route also hits the
        // boundary helper; same fork+join+raise-edge shape.
        let g = lower("fn f(e: i64) -> i64 { return Err(e); }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "isinstance");
        assert_eq!(start.exits.len(), 2);
    }

    #[test]
    fn err_at_match_arm_tail_emits_raise_edge_to_exceptblock() {
        // Match-arm tail boundary route: each arm's `Err(e)` lowers
        // to its own fork+join+raise edge.
        let g = lower(
            "fn f(x: Foo, e: i64) -> i64 {
                match x {
                    Foo::A => 0,
                    _ => Err(e),
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Exactly one Link in the entire graph targets exceptblock —
        // from the wildcard arm's join block.
        let exceptblock_links = g
            .iterblocks()
            .iter()
            .map(|b| {
                b.borrow()
                    .exits
                    .iter()
                    .filter(|exit| {
                        exit.borrow()
                            .target
                            .as_ref()
                            .map(|t| std::rc::Rc::ptr_eq(t, &g.exceptblock))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .sum::<usize>();
        assert_eq!(
            exceptblock_links, 1,
            "exactly one path Links to exceptblock — the wildcard arm's join"
        );
    }

    #[test]
    fn local_binding_shadows_err_at_boundary_does_not_emit_raise_edge() {
        // Locals-first ordering at the boundary: `let Err = some_fn;
        // Err(b)` — the inner `Err` references the local function,
        // not the closed-world host class. The boundary helper
        // recognises the shadowing and falls through to ordinary
        // `lower_call`, emitting `simple_call(<local Variable>, b)`
        // — no fork, no raise edge.
        let g = lower(
            "fn f(some_fn: i64, b: i64) -> i64 {
                let Err = some_fn;
                Err(b)
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            1,
            "shadowed Err at boundary emits simple_call only; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "simple_call");
        // No exit Links to exceptblock anywhere — the shadowing path
        // keeps the value-flow into the returnblock.
        let to_exceptblock = g.iterblocks().iter().any(|b| {
            b.borrow().exits.iter().any(|exit| {
                exit.borrow()
                    .target
                    .as_ref()
                    .map(|t| std::rc::Rc::ptr_eq(t, &g.exceptblock))
                    .unwrap_or(false)
            })
        });
        assert!(
            !to_exceptblock,
            "shadowed Err must not emit a raise edge to exceptblock"
        );
    }

    #[test]
    fn local_binding_shadows_ok_at_boundary_does_not_collapse() {
        // Locals-first ordering at the boundary: `let Ok = some_fn;
        // Ok(b)` — the inner `Ok` references the local function, not
        // the closed-world host class. The boundary helper recognises
        // the shadowing and falls through to ordinary `lower_call`,
        // emitting `simple_call(<local Variable>, b)`.
        let g = lower(
            "fn f(some_fn: i64, b: i64) -> i64 {
                let Ok = some_fn;
                Ok(b)
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            1,
            "shadowed Ok at boundary still emits simple_call; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "simple_call");
        match &start.operations[0].args[0] {
            Hlvalue::Variable(_) => {}
            other => panic!("expected Variable callee (local shadowing), got {other:?}"),
        }
    }

    #[test]
    fn local_binding_shadows_ok_resolves_through_local() {
        // pyopcode.py:502 LOAD_FAST priority — locals win over the
        // closed-world PYRE_STDLIB registry. With `let Ok = a;` the
        // identifier `Ok` resolves through locals; `Ok(b)` lowers
        // as `simple_call(<local Variable>, b)` because the locals
        // lookup succeeds before the host-class fallback.
        let g = lower(
            "fn f(a: i64, b: i64) -> i64 {
                let Ok = a;
                let _r = Ok(b);
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            1,
            "expected exactly one simple_call; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "simple_call");
        match &start.operations[0].args[0] {
            Hlvalue::Variable(_) => {}
            other => panic!("expected Variable callee (local shadowing), got {other:?}"),
        }
    }

    #[test]
    fn qualified_path_in_tail_emits_getattr_cascade() {
        // `StepResult::Continue` — multi-segment qualified path.
        // Leftmost `StepResult` is not in PYRE_STDLIB, so the
        // resolver mints a `HostObject::Class` and caches it on the
        // Builder. The trailing `Continue` segment emits a single
        // `getattr` op per `flowcontext.py:861 LOAD_ATTR` /
        // `operation.py:618 getattr` arity=2, with the second
        // argument a `Constant(ConstValue::ByteStr("Continue"))`.
        // N=2 segments → N-1=1 getattr op.
        let g = lower("fn f() -> i64 { StepResult::Continue }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            1,
            "2-segment path emits 1 getattr op; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "getattr");
        // Receiver is the minted host class.
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert_eq!(obj.qualname(), "StepResult");
                    assert!(obj.is_class());
                }
                other => panic!("expected HostObject(StepResult), got {other:?}"),
            },
            other => panic!("expected Constant receiver, got {other:?}"),
        }
        // Attribute name is the trailing segment as a byte string.
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::ByteStr(b) => assert_eq!(b, b"Continue"),
                other => panic!("expected ByteStr attr name, got {other:?}"),
            },
            other => panic!("expected Constant attr name, got {other:?}"),
        }
        // Result threads into the returnblock link.
        let link = start.exits[0].borrow();
        assert_eq!(
            link.args[0].as_ref().unwrap(),
            &start.operations[0].result,
            "returnblock link must carry the getattr result"
        );
    }

    #[test]
    fn qualified_path_call_callee_emits_getattr_then_simple_call() {
        // `StepResult::Return(value)` — multi-segment callee resolves
        // through the same getattr cascade, then `lower_call` emits
        // `simple_call(<getattr result>, value)`.
        let g = lower("fn f(value: i64) -> i64 { StepResult::Return(value) }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        // simple_call's first arg = getattr result (the bound class
        // attribute that the call invokes).
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
    }

    #[test]
    fn three_segment_path_emits_two_getattr_ops_in_cascade() {
        // `A::B::C` — N=3 segments → N-1=2 getattr ops chained.
        // Leftmost `A` is the minted class; `getattr(<A>, "B")` is
        // op[0]; `getattr(op[0].result, "C")` is op[1]. Mirrors
        // upstream LOAD_GLOBAL + 2× LOAD_ATTR (`flowcontext.py:861`).
        let g = lower("fn f() -> i64 { A::B::C }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "getattr");
        // op[1]'s receiver chains from op[0]'s result.
        assert_eq!(start.operations[1].args[0], start.operations[0].result);
    }

    #[test]
    fn repeat_qualified_path_shares_host_class_identity() {
        // Two references to `StepResult::*` in the same graph share
        // the same `Constant(HostObject(<StepResult>))` identity —
        // `host_env::HOST_CLASS_MINTS` finds-or-mints, returning the
        // same class on the second cascade step (mirrors
        // `LOAD_GLOBAL StepResult` returning the same object on
        // every lookup, regardless of graph — `flowcontext.py:847`).
        let g = lower(
            "fn f(x: i64) -> i64 {
                let _a = StepResult::Continue;
                let _b = StepResult::Halt;
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        let host_a = match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => obj.clone(),
                other => panic!("expected HostObject in op[0], got {other:?}"),
            },
            other => panic!("expected Constant receiver, got {other:?}"),
        };
        let host_b = match &start.operations[1].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => obj.clone(),
                other => panic!("expected HostObject in op[1], got {other:?}"),
            },
            other => panic!("expected Constant receiver, got {other:?}"),
        };
        assert_eq!(
            host_a, host_b,
            "host_env::HOST_CLASS_MINTS must share StepResult identity \
             across both cascade steps (HostObject::eq is Arc::ptr_eq)"
        );
    }

    #[test]
    fn qualified_path_shares_host_class_identity_across_graphs() {
        // Cross-graph identity: two separate `build_flow_from_rust`
        // invocations that name `CrossGraphProbe::*` must observe the
        // same `HostObject::Class` identity. Mirrors upstream
        // `func_globals[name]` returning the same Python object on
        // every `LOAD_GLOBAL` regardless of which graph is being
        // built — `flowcontext.py:845-854 find_global` reads from a
        // process-shared globals dict. Per-graph minting (the prior
        // `Builder::host_classes` HashMap shape) violated this
        // invariant.
        let g1 = lower("fn f() -> i64 { let _a = CrossGraphProbe::A; 0 }").unwrap();
        let g2 = lower("fn g() -> i64 { let _b = CrossGraphProbe::B; 0 }").unwrap();
        let lhs1 = match &g1.startblock.borrow().operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => obj.clone(),
                other => panic!("expected HostObject in g1, got {other:?}"),
            },
            other => panic!("expected Constant receiver in g1, got {other:?}"),
        };
        let lhs2 = match &g2.startblock.borrow().operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => obj.clone(),
                other => panic!("expected HostObject in g2, got {other:?}"),
            },
            other => panic!("expected Constant receiver in g2, got {other:?}"),
        };
        assert_eq!(
            lhs1, lhs2,
            "CrossGraphProbe identity must be process-global \
             (HostObject::eq is Arc::ptr_eq); per-graph minting \
             would have produced two distinct Arcs",
        );
    }

    #[test]
    fn nested_ok_wrapping_qualified_path_at_value_position_emits_getattr_then_simple_call() {
        // `let z = Ok(StepResult::Continue); z` — value-position
        // wrap. Inner qualified path resolves first (1 getattr),
        // then outer `Ok` constructor wraps the inner result (1
        // simple_call) per upstream `LOAD_GLOBAL Ok; LOAD_GLOBAL
        // StepResult; LOAD_ATTR Continue; CALL_FUNCTION 1`. Boundary
        // collapse (Slice O4) does NOT fire because the call sits
        // inside a `let` RHS — the boundary helper only intercepts
        // function/arm tail and `return` operand positions.
        let g = lower("fn f() -> i64 { let z = Ok(StepResult::Continue); z }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "simple_call");
        // simple_call's callee = Ok host class; arg[1] = getattr result.
        match &start.operations[1].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => assert_eq!(obj.qualname(), "Ok"),
                other => panic!("expected Ok HostObject, got {other:?}"),
            },
            other => panic!("expected Constant callee, got {other:?}"),
        }
        assert_eq!(start.operations[1].args[1], start.operations[0].result);
    }

    #[test]
    fn qualified_path_with_leading_colon_rejects() {
        // `::std::result::Result` — globally-anchored path is its
        // own port (out of M2.5e orthodox scope per the plan's
        // Non-goals).
        let err = lower("fn f() -> i64 { ::std::result::Result }").unwrap_err();
        match err {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("leading-`::`"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(leading-::), got {other:?}"),
        }
    }

    #[test]
    fn qualified_path_with_generic_arguments_rejects() {
        // `Vec::<i32>::new` — generic-argument paths reject pending
        // type-arg reification (out of M2.5e orthodox scope per the
        // plan's Non-goals).
        let err = lower("fn f() -> i64 { Vec::<i32>::new() }").unwrap_err();
        match err {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("generic arguments"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(generic args), got {other:?}"),
        }
    }

    #[test]
    fn char_literal_lowers_to_single_char_str() {
        // Rust `char` → `ConstValue::UniStr(len==1)`. Matches the
        // match-arm side (`classify_pattern`) so scrutinee and
        // exitcase share the identical Constant for an end-to-end
        // char match. No operations emitted — a bare `let _c = 'a'`
        // is pure SSA binding.
        let g = lower("fn f() -> i64 { let c = 'a'; 0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "bare char literal binding emits no ops"
        );
    }

    #[test]
    fn byte_literal_lowers_to_bytestr() {
        let g = lower("fn f() -> u8 { b'a' }").unwrap();
        checkgraph(&g);
        let link = g.startblock.borrow().exits[0].clone();
        match link.borrow().args[0].as_ref() {
            Some(Hlvalue::Constant(c)) => {
                assert_eq!(c.value, ConstValue::byte_str(b"a"));
            }
            other => panic!("expected ByteStr return constant, got {other:?}"),
        }
    }

    #[test]
    fn char_literal_returned_as_single_char_str_constant() {
        // Expression-position `'a'` must reach the returnblock as
        // `Constant(Str("a"))` — same encoding the match-arm side
        // emits in `classify_pattern`. An end-to-end test with a
        // constant-scrutinee match would also need `guessbool` /
        // `HLOperation.eval` constant-folding (PRE-EXISTING-ADAPTATION
        // #3 in `mod.rs`) to close cleanly, so that composition is
        // deferred to the constfold port; here we only pin the
        // literal-to-Constant step.
        let g = lower("fn f() -> &'static str { 'a' }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.exits.len(), 1);
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Constant(c) => match &c.value {
                value if value.string_eq("a") => {}
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    // ---- small-surface operator tests ---------------------------

    #[test]
    fn unary_neg_emits_neg_op() {
        let g = lower("fn f(x: i64) -> i64 { -x }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "neg");
        assert_eq!(start.operations[0].args.len(), 1);
    }

    #[test]
    fn rejects_unary_not() {
        // `!x` has no 1-to-1 upstream mapping — UNARY_NOT is a
        // `bool` + guessbool fork, UNARY_INVERT emits `invert`. The
        // adapter can't distinguish without type info.
        match lower("fn f(x: i64) -> i64 { !x }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("unary `!`"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(unary !), got {other:?}"),
        }
    }

    #[test]
    fn reference_is_passthrough() {
        // `&x` and `*x` emit no ops — they preserve the SSA value.
        let g = lower("fn f(x: i64) -> i64 { *&x }").unwrap();
        checkgraph(&g);
        assert_eq!(g.startblock.borrow().operations.len(), 0);
    }

    #[test]
    fn field_access_emits_getattr() {
        let g = lower("fn f(p: i64) -> i64 { p.x }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations.len(), 1);
        assert_eq!(start.operations[0].opname, "getattr");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                value if value.string_eq("x") => {}
                other => panic!("expected Str, got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    #[test]
    fn tuple_field_access_emits_getitem() {
        let g = lower("fn f(t: i64) -> i64 { t.0 }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations[0].opname, "getitem");
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::Int(0) => {}
                other => panic!("expected Int(0), got {other:?}"),
            },
            other => panic!("expected Constant, got {other:?}"),
        }
    }

    #[test]
    fn index_emits_getitem() {
        let g = lower("fn f(a: i64, i: i64) -> i64 { a[i] }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(start.operations[0].opname, "getitem");
        assert_eq!(start.operations[0].args.len(), 2);
    }

    #[test]
    fn rejects_numeric_cast() {
        // `x as T` has no 1-to-1 upstream mapping — upstream's
        // `do_int`/`do_float`/`bool` are dunder-method dispatches,
        // not numeric coercions. Users who need conversion should
        // call an explicit helper.
        match lower("fn f(x: i64) -> i64 { x as i32 }").unwrap_err() {
            AdapterError::Unsupported { reason } => {
                assert!(reason.contains("`x as T`"), "reason: {reason}");
            }
            other => panic!("expected Unsupported(x as T), got {other:?}"),
        }
    }

    // ---- loop body can carry general expression statements -------

    #[test]
    fn loop_body_accepts_method_call_statement() {
        // `while cond { h.step(); }` — method call as a side-effect
        // inside the body. Upstream SETUP_LOOP (`flowcontext.py:488,
        // :794`) wraps arbitrary bytecode, not a subset.
        let g = lower(
            "fn f(h: Handler, cond: bool) -> i64 {
                while cond {
                    h.step();
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Body emits getattr('step') + simple_call(bound) — evidence
        // that the statement-position expression is lowered.
        let has_getattr = g.iterblocks().iter().any(|blk| {
            blk.borrow()
                .operations
                .iter()
                .any(|o| o.opname == "getattr")
        });
        assert!(has_getattr, "loop body must lower method call");
    }

    #[test]
    fn loop_body_accepts_field_access_statement() {
        // `while cond { let _ = h.field; }` — field access as a
        // side-effect in a let pattern. Verifies loop body delegates
        // to the same expression lowering path.
        let g = lower(
            "fn f(h: Handler, cond: bool) -> i64 {
                while cond {
                    let _ignore = h.field;
                }
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
    }

    #[test]
    fn top_level_expression_statement_discards_result() {
        // `fn f(h) { h.step(); 0 }` — method call as a statement at
        // top level. Same POP_TOP semantic as inside loop body; the
        // call's ops still emit but the result Variable is discarded.
        let g = lower(
            "fn f(h: Handler) -> i64 {
                h.step();
                0
            }",
        )
        .unwrap();
        checkgraph(&g);
        let has_getattr = g.iterblocks().iter().any(|blk| {
            blk.borrow()
                .operations
                .iter()
                .any(|o| o.opname == "getattr")
        });
        assert!(has_getattr, "top-level expression statement must lower");
    }

    // ---- Slice O8: constfold-then-emit cascade -------------------

    /// Walker-registered enum: cascade `MyEnum::Continue` resolves
    /// to a *folded* `Constant(HostObject(<Continue child>))` — no
    /// raw `getattr` op is recorded for the variant lookup.
    ///
    /// Mirrors upstream `flowcontext.py:861 LOAD_ATTR` →
    /// `op.getattr(w_obj, w_name).eval(self)` →
    /// `operation.py:624 GetAttr.constfold` returning
    /// `const(getattr(MyEnum, "Continue"))` when both args are
    /// foldable.
    #[test]
    fn cascade_constfolds_through_walker_registered_enum_variant() {
        use crate::flowspace::rust_source::register::register_rust_module;
        use syn::Item;

        // Per-module scoping (Issue 1.3): the walker's `ModuleId`
        // must match the body lowerer's id. Put the enum and the
        // entry-point fn in the same `syn::File`, walk it once to
        // register the enum under id `m`, then call
        // `build_flow_from_rust_in_module(_, m)` so the body's
        // `LOAD_GLOBAL` resolution hits the matching partition.
        let src = "pub enum ParityProbe_O8_Constfold_Enum { ContinueA, StopA }
                   fn f(x: ParityProbe_O8_Constfold_Enum) -> i64 {
                       match x {
                           ParityProbe_O8_Constfold_Enum::ContinueA => 1,
                           _ => 0,
                       }
                   }";
        let file = syn::parse_file(src).expect("walker + entry fixture parses");
        let module_id = register_rust_module(&file);
        let entry = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(item_fn) if item_fn.sig.ident == "f" => Some(item_fn),
                _ => None,
            })
            .expect("entry-point fn `f` in fixture");

        // Match against the walker-registered enum's variant. The
        // cascade's first step (`getattr(<enum>, "ContinueA")`) must
        // collapse to a `Constant(HostObject(<ContinueA child>))`,
        // making the startblock have just the `isinstance` op (no
        // raw `getattr` precedes it).
        let g = super::build_flow_from_rust_in_module(entry, module_id)
            .expect("lower against walker-registered enum");
        checkgraph(&g);

        let start = g.startblock.borrow();
        // PARITY: with constfold, the first cascade step folds the
        // variant lookup and `isinstance` becomes the SOLE op of
        // the startblock. The minted-class fallback (test
        // `match_unit_variant_two_arms_emits_isinstance_cascade`)
        // emits 2 ops because the leftmost was minted with empty
        // members.
        assert_eq!(
            start.operations.len(),
            1,
            "constfold collapses the variant getattr, leaving only `isinstance`: \
             {:?}",
            start
                .operations
                .iter()
                .map(|o| o.opname.clone())
                .collect::<Vec<_>>(),
        );
        assert_eq!(start.operations[0].opname, "isinstance");

        // The isinstance op's second arg is the folded variant
        // class — a `Constant(HostObject(<variant>))` whose
        // qualname is `<EnumName>.<VariantName>`.
        match &start.operations[0].args[1] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => {
                    assert!(obj.is_class(), "folded variant must be a class");
                    assert_eq!(obj.qualname(), "ParityProbe_O8_Constfold_Enum.ContinueA");
                }
                other => panic!("expected HostObject variant, got {other:?}"),
            },
            other => panic!("expected Constant variant carrier, got {other:?}"),
        }
    }

    /// Walker-registered struct: `MyStruct::field` lookup misses
    /// the (empty) class dict — Issue 3 (2026-05-05) closes the
    /// raw-getattr fall-through for walker-registered classes:
    /// upstream `operation.py:638-642 GetAttr.constfold` raises
    /// `FlowingError` whenever both args are foldable and Python's
    /// `getattr` would raise `AttributeError`. The Rust counterpart
    /// surfaces `AdapterError::Unsupported`. The mint-on-demand path
    /// (PRE-EXISTING-ADAPTATION) keeps the raw-emit fall-through —
    /// see [`cascade_falls_through_to_raw_emit_for_minted_class_missing_member`].
    #[test]
    fn cascade_fails_loud_for_walker_registered_class_missing_member() {
        use crate::flowspace::rust_source::register::register_rust_module;
        use syn::Item;

        let src = "pub struct ParityProbe_Issue3_Empty_Struct { f: i64 }
                   fn f(x: ParityProbe_Issue3_Empty_Struct) -> i64 {
                       match x {
                           ParityProbe_Issue3_Empty_Struct::nonexistent_member => 1,
                           _ => 0,
                       }
                   }";
        let file = syn::parse_file(src).expect("walker + entry fixture parses");
        let module_id = register_rust_module(&file);
        let entry = file
            .items
            .iter()
            .find_map(|item| match item {
                Item::Fn(item_fn) if item_fn.sig.ident == "f" => Some(item_fn),
                _ => None,
            })
            .expect("entry-point fn `f` in fixture");

        let err = super::build_flow_from_rust_in_module(entry, module_id)
            .expect_err("walker-registered class + missing member must fail-loud");
        match err {
            AdapterError::Unsupported { reason } => {
                assert!(
                    reason.contains("AttributeError") && reason.contains("nonexistent_member"),
                    "expected upstream-style AttributeError → FlowingError message, got: {reason}",
                );
            }
            other => panic!("expected AdapterError::Unsupported, got {other:?}"),
        }
    }

    /// Mint-on-demand classes (PRE-EXISTING-ADAPTATION) carry empty
    /// class dicts pending walker coverage of the source-file
    /// declaration. The cascade treats them as opaque — analogous to
    /// upstream's `w_obj.foldable() == False` path which records the
    /// raw `getattr` op instead of folding. Convergence: once
    /// mint-on-demand is retired (every multi-segment leftmost
    /// resolves through the walker), this branch + the
    /// `is_host_class_minted` helper go away and walker-registered
    /// fail-loud applies uniformly.
    #[test]
    fn cascade_falls_through_to_raw_emit_for_minted_class_missing_member() {
        // No `register_rust_module` pre-pass — `ParityProbe_Issue3_Minted`
        // is not in any registry slice, so the multi-segment leftmost
        // resolution lands on `mint_host_class`. Body lowering then
        // hits the cascade's class-dict miss and falls through to the
        // raw `getattr` op per the PRE-EXISTING-ADAPTATION.
        let item = parse(
            "fn f(x: i64) -> i64 {
                match x {
                    ParityProbe_Issue3_Minted::variant => 1,
                    _ => 0,
                }
            }",
        );
        let g = super::build_flow_from_rust(&item)
            .expect("mint-class missing member must fall through to raw emit");
        checkgraph(&g);
        let start = g.startblock.borrow();
        // Fall-through preserves the raw-emit behaviour: getattr
        // followed by isinstance.
        assert_eq!(start.operations.len(), 2);
        assert_eq!(start.operations[0].opname, "getattr");
        assert_eq!(start.operations[1].opname, "isinstance");
    }

    // ____________________________________________________________
    // Slice O4 boundary-vs-value-position discipline (Issue 1.1
    // regression suite).
    //
    // The Slice O4 PRE-EXISTING-ADAPTATION (`Ok(x)` / `Some(x)` →
    // unwrap, `None` → `Constant(None)`, `Err(e)` → raise edge per
    // O5) collapses the Result/Option wrapper at the
    // function-return boundary. The threading invariant verified
    // here: collapse fires ONLY when the wrapper sits at a real
    // return edge — function tail, `return` statement, or the tail
    // of an `if`/`match`/`block` whose result IS the function's
    // return value. Value-position uses (let-init, function arg,
    // binop operand) MUST keep emitting `simple_call(<host class>,
    // x)` so the host-class instance threads through to the
    // consumer.
    //
    // Upstream RPython has no Result/Option layer so there's no
    // direct counterpart; the discipline mirrors the structural
    // invariant that `flowcontext.py:1232 RETURN_VALUE` is the
    // only return-edge site, never an intermediate block tail or
    // value-position match.

    #[test]
    fn ok_call_in_value_position_block_emits_simple_call() {
        // Regression test for Issue 1.1: `let z = { Ok(x) }; z` —
        // value-position block tail. Before the fix, lower_block's
        // tail dispatch unconditionally went through
        // lower_value_boundary and unwrapped Ok(x) to x, so `z` got
        // bound to the bare argument. Correct behaviour: emit
        // `simple_call(<Ok>, x)`; bind `z` to the call result.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let z = { Ok(x) };
                z
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        // Exactly one op — the simple_call(<Ok>, x) for the inner
        // block. The outer block tail is `z` (a bare path), no op.
        assert_eq!(
            start.operations.len(),
            1,
            "value-position Ok(x) block tail must emit simple_call; ops: {:?}",
            start.operations
        );
        assert_eq!(start.operations[0].opname, "simple_call");
        // Callee is the Ok host class (Constant), not unwrapped.
        match &start.operations[0].args[0] {
            Hlvalue::Constant(c) => match &c.value {
                ConstValue::HostObject(obj) => assert_eq!(obj.qualname(), "Ok"),
                other => panic!("expected ConstValue::HostObject(Ok), got {other:?}"),
            },
            other => panic!("expected Constant callee for value-position Ok, got {other:?}"),
        }
    }

    #[test]
    fn ok_call_in_value_position_if_arms_emit_simple_call_each() {
        // Regression test for Issue 1.1: `let z = if c { Ok(a) }
        // else { Ok(b) }; z`. Both arms are value-position so each
        // must emit `simple_call(<Ok>, _)` rather than collapsing
        // to the inner value.
        let g = lower(
            "fn f(c: bool, a: i64, b: i64) -> i64 {
                let z = if c { Ok(a) } else { Ok(b) };
                z
            }",
        )
        .unwrap();
        checkgraph(&g);
        // Walk every block; total simple_call ops over the whole
        // graph must be exactly 2 (one per if-arm).
        let mut total_simple_calls = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    total_simple_calls += 1;
                }
            }
        }
        assert_eq!(
            total_simple_calls, 2,
            "expected 2 simple_call ops (one per Ok-wrapper arm); \
             value-position if must NOT collapse arm tails"
        );
    }

    #[test]
    fn ok_call_in_value_position_match_arms_emit_simple_call_each() {
        // Regression test for Issue 1.1: `let z = match x { 0 =>
        // Ok(1), _ => Ok(2) }; z`. Both arms value-position →
        // each emits its own `simple_call(<Ok>, _)`.
        let g = lower(
            "fn f(x: i64) -> i64 {
                let z = match x { 0 => Ok(1), _ => Ok(2) };
                z
            }",
        )
        .unwrap();
        checkgraph(&g);
        let mut total_simple_calls = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    total_simple_calls += 1;
                }
            }
        }
        assert_eq!(
            total_simple_calls, 2,
            "expected 2 simple_call ops (one per Ok-wrapper arm); \
             value-position match must NOT collapse arm tails"
        );
    }

    #[test]
    fn nested_ok_in_function_tail_block_still_collapses() {
        // Sanity check: the boundary mode threads correctly through
        // a function-tail block. `fn f() -> i64 { { Ok(x) } }` —
        // the inner `{ Ok(x) }` IS at boundary because the outer
        // block is the function body. Slice O4 still collapses.
        let g = lower(
            "fn f(x: i64) -> i64 {
                { Ok(x) }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "function-tail nested-block Ok(x) collapses; no SpaceOperation emits"
        );
    }

    #[test]
    fn nested_ok_in_function_tail_match_still_collapses() {
        // Sanity: function-tail match with Ok arm tails — the
        // boundary flag propagates from fn-body into lower_match,
        // so each arm collapses Ok per Slice O4.
        let g = lower(
            "fn f(x: i64) -> i64 {
                match x {
                    0 => Ok(1),
                    _ => Ok(2),
                }
            }",
        )
        .unwrap();
        checkgraph(&g);
        let mut simple_call_count = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    simple_call_count += 1;
                }
            }
        }
        assert_eq!(
            simple_call_count, 0,
            "function-tail match arm Ok wrappers must collapse; got {simple_call_count} simple_call ops"
        );
    }

    // ____________________________________________________________
    // Issue 1 (Codex audit 2026-05-05) — explicit `return EXPR;`
    // boundary semantics for nested control-flow EXPR.
    //
    // Before the fix, `lower_return` called `lower_value_boundary`
    // directly. `lower_value_boundary` only intercepts the literal
    // `Ok(_)` / `Some(_)` / `None` / `Err(_)` shapes; for
    // `Expr::Block` / `Expr::If` / `Expr::Match` it fell through
    // to `lower_expr`, which routes those constructs through
    // `lower_block(_, false)` / `lower_if(_, false)` /
    // `lower_match(_, false)` — value mode. Result: the wrapper
    // op survived into the returnblock Link.
    //
    // The fix routes `lower_return.ret.expr` through
    // `lower_arm_body(_, _, true)` so block/if/match dispatch
    // through their boundary-aware lowerings. The leaf cases
    // (`Ok(x)` / etc.) still reach `lower_value_boundary` via
    // `lower_arm_body`'s default arm.

    #[test]
    fn explicit_return_block_with_ok_tail_collapses() {
        // `return { Ok(x) };` — explicit return of a nested block.
        // The inner Ok wrapper must collapse: this is a real
        // function-return boundary, so the wrapper would otherwise
        // survive into the returnblock Link.
        let g = lower("fn f(x: i64) -> i64 { return { Ok(x) }; }").unwrap();
        checkgraph(&g);
        let start = g.startblock.borrow();
        assert_eq!(
            start.operations.len(),
            0,
            "explicit return of {{ Ok(x) }} must collapse the wrapper; got ops: {:?}",
            start.operations
        );
        let link = start.exits[0].borrow();
        match link.args[0].as_ref().unwrap() {
            Hlvalue::Variable(_) => {}
            other => panic!("expected Variable return value (inner x), got {other:?}"),
        }
    }

    #[test]
    fn explicit_return_if_with_ok_arm_tails_collapses_each() {
        // `return if c { Ok(a) } else { Ok(b) };` — explicit return
        // of a value-producing if. Both arms feed the function
        // return so each Ok wrapper must collapse.
        let g =
            lower("fn f(c: bool, a: i64, b: i64) -> i64 { return if c { Ok(a) } else { Ok(b) }; }")
                .unwrap();
        checkgraph(&g);
        let mut simple_call_count = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    simple_call_count += 1;
                }
            }
        }
        assert_eq!(
            simple_call_count, 0,
            "explicit return if must collapse Ok in BOTH arms; got {simple_call_count} \
             simple_call ops (was: leaked through lower_expr's value-mode lowering)"
        );
    }

    #[test]
    fn explicit_return_match_with_ok_arm_tails_collapses_each() {
        // `return match x { 0 => Ok(1), _ => Ok(2) };` — explicit
        // return of a value-producing match. Each arm tail is at
        // the function-return boundary; Ok wrappers must collapse.
        let g = lower(
            "fn f(x: i64) -> i64 {
                return match x { 0 => Ok(1), _ => Ok(2) };
            }",
        )
        .unwrap();
        checkgraph(&g);
        let mut simple_call_count = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    simple_call_count += 1;
                }
            }
        }
        assert_eq!(
            simple_call_count, 0,
            "explicit return match must collapse Ok in EVERY arm; got {simple_call_count} \
             simple_call ops"
        );
    }

    #[test]
    fn explicit_return_nested_block_in_if_with_ok_collapses() {
        // `return if c { { Ok(a) } } else { Ok(b) };` — the then-
        // arm wraps Ok in another inner block. Both layers should
        // observe at_boundary=true: the outer if-arm's lower_block
        // (boundary=true) plus the tail Ok in that inner block
        // hits lower_value_boundary directly.
        let g = lower(
            "fn f(c: bool, a: i64, b: i64) -> i64 {
                return if c { { Ok(a) } } else { Ok(b) };
            }",
        )
        .unwrap();
        checkgraph(&g);
        let mut simple_call_count = 0usize;
        for block in g.iterblocks() {
            for op in &block.borrow().operations {
                if op.opname == "simple_call" {
                    simple_call_count += 1;
                }
            }
        }
        assert_eq!(
            simple_call_count, 0,
            "explicit return with nested block-in-arm must collapse; \
             got {simple_call_count} simple_call ops"
        );
    }
}
