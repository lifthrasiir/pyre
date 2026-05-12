//! Bundle an adapter-produced `FunctionGraph` into the
//! `(HostObject, PyGraph)` pair the annotator pipeline expects.
//!
//! Upstream analogue — `rpython/translator/interactive.py:25-26`:
//!
//! ```python
//! graph = self.context.buildflowgraph(entry_point)
//! self.context._prebuilt_graphs[entry_point] = graph
//! ```
//!
//! Line 25 runs upstream `build_flow` on Python bytecode and wraps the
//! resulting `FunctionGraph` inside a `PyGraph`. Line 26 seeds the
//! translator's prebuilt-graph cache so subsequent
//! `buildflowgraph(same entry_point)` calls short-circuit without
//! re-building.
//!
//! The Rust-source counterpart has no bytecode, so
//! `build_flow_from_rust` replaces line 25's work; this helper packages
//! the same `(host, pygraph)` pair that line 26 inserts into the cache.
//! Seeding the cache stays the caller's responsibility so this module
//! does not need to depend on `TranslationContext`.
//!
//! The synthetic [`HostCode`] populated here is the minimum needed for
//! upstream `cpython_code_signature` (`flowspace/bytecode.py`) to read
//! back the right argnames — `co_argcount`, `co_varnames`, `co_flags`.
//! `co_code` is empty because the function has no bytecode. Callers
//! that later introspect the code object (e.g. `is_generator`) will
//! see `CO_GENERATOR` unset, which is the correct Rust-fn answer.
//!
//! Upstream RPython's `_assert_rpythonic` (`objspace.py:33-35`) requires
//! `CO_NEWLOCALS` on any RPython function's code object, so we set it
//! here even though the adapter itself bypasses `build_flow` /
//! `_assert_rpythonic`; downstream consumers that re-run
//! `_assert_rpythonic` on the pair (e.g. a later `PyGraph::new` rebuild)
//! must see a structurally valid code object.
//!
//! `co_nlocals` / `co_varnames` cover formal arguments **and** every
//! `let`-bound / `for`-pattern identifier that [`build_flow_from_rust`]
//! may have introduced as an extra local. Upstream `pygraph.py:14-16`
//! sizes the initial `locals = [None] * co_nlocals` array by the full
//! local count; synthesizing only the formal-arg prefix here would let
//! a downstream `PyGraph::new` rebuild produce an under-sized locals
//! array that disagrees with the adapter's by-name `HashMap`.
//!
//! `co_firstlineno` reads `syn::ItemFn`'s `fn_token` span (requires
//! the `proc-macro2/span-locations` feature — see this crate's
//! `Cargo.toml`). `co_filename` is supplied by the caller via the
//! `source_filename: Option<&str>` parameter — `syn::Span` has no
//! stable accessor for the source file path, so the caller (who
//! performed the `parse_file` / `parse_str` call in the first place)
//! is the authoritative source. When the caller has no file context
//! (e.g., `parse_str` on a fixture), passing `None` falls back to
//! the `<rust-source>` sentinel upstream would never emit but the
//! error-rendering code (`tool/error.rs:304`) handles gracefully.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use std::collections::HashMap as StdHashMap;
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    BinOp, Expr, ExprBinary, ExprForLoop, ExprLit, ExprPath, ExprUnary, File, FnArg, Item,
    ItemEnum, ItemFn, ItemStruct, Lit, Local, Pat, PatIdent, UnOp,
};

use super::build_flow::{AdapterError, build_flow_from_rust_in_module};
use super::host_env::{
    ModuleId, module_globals_lookup, module_globals_snapshot, pyre_stdlib_lookup,
    register_module_global,
};
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::objspace::CO_NEWLOCALS;
use crate::flowspace::operation::{
    ArithOps, cmp_fold, coerce_arith, coerce_int_pair, float_py_mod, int_py_floor_div, int_py_mod,
    is_foldable_numeric, python_eq_const,
};
use crate::flowspace::pygraph::PyGraph;

/// Walk `item_fn`, run the Rust-AST adapter, and return the
/// `(HostObject, PyGraph)` pair that the upstream translator cache
/// expects. The caller is responsible for seeding
/// `TranslationContext._prebuilt_graphs` with the returned pair, exactly
/// as `interactive.py:26` does:
///
/// ```ignore
/// let (host, pygraph) = build_host_function_from_rust(
///     &item_fn,
///     Some("pyre/src/pyopcode.rs"),
///     Some(src),
/// )?;
/// translator
///     ._prebuilt_graphs
///     .borrow_mut()
///     .insert(host.clone(), pygraph);
/// ```
///
/// - `source_filename` populates `HostCode.co_filename` — upstream reads
///   `func.__code__.co_filename` at `model.py:54` for graph-rendering
///   error messages (`tool/error.rs:304`). `syn::Span` has no stable
///   file-path accessor, so the caller (who originally invoked
///   `syn::parse_file` / `parse_str`) is the authoritative source.
///   Passing `None` falls back to the `<rust-source>` sentinel.
/// - `source_text` populates `GraphFunc.source` (upstream
///   `inspect.getsource(func)` at `flowspace/bytecode.py:50`) **and**
///   `FunctionGraph._source` (upstream `model.py:35-47` `source`
///   setter). When `None`, `graph.source()` falls back to the GraphFunc
///   setting, then to the `"source not found"` error surfaced by
///   `tool/error.rs:300`.
pub fn build_host_function_from_rust(
    item_fn: &ItemFn,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    // Single-`ItemFn` entry mints a fresh ModuleId — no walker
    // pre-pass means the registry slice is empty for this id, so
    // every `LOAD_GLOBAL` lookup falls through to
    // `pyre_stdlib_lookup` / mint exactly as the pre-Issue-1.3
    // process-global path did. Callers that want sibling-item
    // resolution should route through
    // [`build_host_function_from_rust_file`] instead.
    build_host_function_from_rust_in_module(
        item_fn,
        ModuleId::fresh(),
        source_filename,
        source_text,
    )
}

/// Internal helper used by [`build_host_function_from_rust`] and
/// [`build_host_function_from_rust_file`] — both lower the body
/// under an explicit `module_id` so the body's `LOAD_GLOBAL`
/// lookups resolve against the matching registry partition.
fn build_host_function_from_rust_in_module(
    item_fn: &ItemFn,
    module_id: ModuleId,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    let mut graph = build_flow_from_rust_in_module(item_fn, module_id)?;
    let HostMetadataParts {
        host,
        host_code,
        gf,
    } = build_host_metadata_parts(item_fn, module_id, source_filename, source_text)?;

    // upstream `PyGraph.__init__` (pygraph.py:20) assigns
    // `FunctionGraph.func = func` via `super().__init__`. Mirror that so
    // downstream helpers (`FlowContext::new`, `FunctionDesc.getuniquegraph`)
    // see the same GraphFunc the HostObject exposes.
    graph.func = Some(gf.clone());
    // upstream `model.py:35-47` exposes `FunctionGraph.source` as a
    // property-with-setter backed by `_source`. The Translation
    // constructor at `interactive.py:25` delegates to
    // `buildflowgraph`, whose non-prebuilt branch leaves
    // `graph._source` untouched — but `inspect.getsource(func)` has
    // already populated `GraphFunc.source`, and the `FunctionGraph.source`
    // property returns it via the `func.source` fallback at
    // `model.py:42`. We mirror the same pair assignment explicitly
    // so `graph.source()` at `model.rs:3207-3216` hits `_source`
    // first (fast path for graph-render error messages).
    if let Some(src) = source_text {
        graph._source = Some(src.to_owned());
    }

    let pygraph = Rc::new(PyGraph {
        graph: Rc::new(RefCell::new(graph)),
        signature: RefCell::new(host_code.signature.clone()),
        // upstream `PyGraph.__init__`: `self.defaults =
        // func.__defaults__ or ()`. Rust-source adapter does not yet
        // surface default values; use the empty tuple shape.
        defaults: RefCell::new(Some(Vec::new())),
        access_directly: Cell::new(false),
        func: gf,
    });
    Ok((host, pygraph))
}

/// File-aware sibling of [`build_host_function_from_rust`]: walk
/// every top-level item in `file` through [`register_rust_module`]
/// FIRST (so sibling enums/structs/fns are seeded into the
/// module-globals registry), then locate the `entry_point_name`
/// `Item::Fn` and run the body lowerer on it.
///
/// This is the upstream-orthodox shape for the
/// `interactive.py:14 def __init__(self, entry_point, ...)` →
/// `:25 buildflowgraph(entry_point)` chain: by the time
/// `build_flow(entry_point)` runs upstream, `entry_point.func_globals`
/// already contains every other top-level definition in the same
/// source module (Python's module-import bound them at `def` /
/// `class` time — `flowcontext.py:847 w_globals.value[varname]`).
/// The Rust analogue is the walker pre-pass over `file.items`.
///
/// `entry_point_name` is the bare ident of the target fn — matches
/// the upstream `Translation(entry_point=funcobj)` carrier where
/// `funcobj.__name__` identifies which fn is the build target.
///
/// Returns `AdapterError::Unsupported` if the entry-point name is
/// not a top-level `Item::Fn` in `file`. Other items (enums,
/// structs, etc.) are walked unconditionally — only the entry
/// point's body lowering is gated.
pub fn build_host_function_from_rust_file(
    file: &File,
    entry_point_name: &str,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<(HostObject, Rc<PyGraph>), AdapterError> {
    // Walker pre-pass — register every top-level item under a
    // `ModuleId` keyed on `source_filename`. The same id is then
    // threaded into body lowering so the entry-point's
    // `LOAD_GLOBAL` resolutions see exactly the bindings the
    // walker just wrote (Issue 1.3 per-module scoping). When the
    // caller threads in a path, the id is path-keyed (Issue 2,
    // 2026-05-05): two walks of the same source file converge on
    // the same id, mirroring upstream
    // `entry_point_a.__globals__ is entry_point_b.__globals__`
    // for two functions defined in the same Python module.
    let module_id = register_rust_module_at(file, source_filename)?;

    // Locate the entry-point fn. Upstream `interactive.py:14` takes
    // the function object directly; here the caller names it because
    // a `&syn::File` carries multiple items.
    let item_fn = file
        .items
        .iter()
        .find_map(|item| match item {
            Item::Fn(item_fn) if item_fn.sig.ident == entry_point_name => Some(item_fn),
            _ => None,
        })
        .ok_or_else(|| AdapterError::Unsupported {
            reason: format!(
                "entry-point fn `{entry_point_name}` not found among top-level items in the \
                 supplied `syn::File` — `interactive.py:14 entry_point` requires a real function \
                 object as the build target"
            ),
        })?;

    build_host_function_from_rust_in_module(item_fn, module_id, source_filename, source_text)
}

/// Build a `HostObject::UserFunction` for `item_fn` carrying the
/// synthetic `HostCode` (signature, co_varnames, co_firstlineno) but
/// **without** running [`build_flow_from_rust`] on the body. The
/// embedded `GraphFunc.prebuilt_flow_graph` stays `None`.
///
/// This is the Rust-source analogue of Python's module-import-time
/// function creation: at `import` time, the Python interpreter binds
/// the name in `module.__dict__` to a function object whose
/// `__code__` is set but whose flowspace `FunctionGraph` has not been
/// built yet.
///
/// **Status**: as of Issue 1.2 (PRE-EXISTING-ADAPTATION), this helper
/// is no longer the walker's body-deferral path —
/// [`register_rust_module`] does not register `Item::Fn` because
/// the rebuild path between `FunctionDesc.buildgraph`
/// (`description.py:140`) and the Rust-AST adapter is missing,
/// so a deferred-body `HostObject` would supply empty bytecode at
/// lowering time. The helper remains exported as a public utility
/// for callers that explicitly want metadata-only construction
/// (e.g. a future M2.5g side-table walker that pairs the metadata
/// HostObject with a stored `&syn::ItemFn` for later replay).
pub fn build_host_function_metadata_from_rust(
    item_fn: &ItemFn,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<HostObject, AdapterError> {
    // No walker pre-pass on the metadata-only path — module dict is
    // empty, matching upstream `func.__globals__ == {}` for a
    // function defined with no module bindings yet visible.
    Ok(build_host_metadata_parts(item_fn, ModuleId::fresh(), source_filename, source_text)?.host)
}

/// Walk a parsed Rust source `file` and register every top-level
/// **class-shaped** item (`Item::Enum` / `Item::Struct`) and
/// **literal const** (`Item::Const`) into the process-global
/// module-globals registry (`HOST_RUST_MODULE_GLOBALS`).
///
/// Mirrors Python module import: when the Python interpreter executes
/// a `class` statement or a top-level constant assignment at module
/// scope, it binds the name in `module.__dict__` to the freshly-built
/// class object / value. This walker is the Rust-source counterpart
/// for the *bindable-without-body* subset.
///
/// Subsequent `Builder::resolve_path_constant` lookups route through
/// `module_globals_lookup` and return the registered value directly,
/// matching upstream `flowcontext.py:847 w_globals.value[varname]`.
///
/// ### Why no `Item::Fn`?
///
/// Upstream Python `def` populates `module.__dict__[name]` with a
/// function object whose body lowering is deferred to
/// `FunctionDesc.buildgraph` (`description.py:140`) at the first
/// annotator-driven call site. The deferred lowering ALWAYS routes
/// through `build_flow(GraphFunc)` which consumes Python bytecode
/// from `func.__code__.co_code`.
///
/// pyre's `HostCode` for an `Item::Fn` is constructed at
/// `register.rs::build_host_metadata_parts` with **empty bytecode**
/// (`CodeUnits::from(Vec::new())`) because the Rust-AST adapter is
/// the only path that can actually lower the body. There is no
/// connection from `FunctionDesc.buildgraph` back to
/// `build_flow_from_rust` (the AST is not stored in `HostCode`,
/// only the syntactic skeleton — `co_varnames` / `co_firstlineno` /
/// `co_filename`). So a sibling-fn `HostObject` registered here
/// would masquerade as a callable function but, on resolution, hand
/// the annotator empty bytecode to "lower", silently producing a
/// no-op graph or panicking.
///
/// **PRE-EXISTING-ADAPTATION**: drop `Item::Fn` registration until
/// the walker can either (a) eagerly build the
/// `prebuilt_flow_graph` per Slice M2.5f and bind the registered
/// `HostObject` to it, or (b) store the original `&syn::ItemFn` in
/// a side table that `FunctionDesc.buildgraph` can consult. Both
/// paths are multi-session work — see plan
/// `~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`
/// (Phase M2.5g extern-Rust-helper registry walker epic). Until
/// either lands, sibling fn name resolution falls through to the
/// same mint-or-fail path that pre-O9 main exercised.
///
/// The single entry-point fn that production callers actually want
/// to lower is found directly via `file.items.iter().find_map(...)`
/// in [`build_host_function_from_rust_file`] — that path bypasses
/// the registry entirely and feeds the `&ItemFn` to
/// `build_host_function_from_rust`, which DOES run the Rust-AST
/// adapter and produce a real `prebuilt_flow_graph`.
///
/// ### Re-registration semantics
///
/// `register_module_global` is **last-writer-wins** under a fixed
/// `(module_id, name)`: every call unconditionally overwrites the
/// prior entry. This mirrors upstream `module.__dict__[name] =
/// value` — every top-level binding statement is an unconditional
/// assignment, whether on first import, `exec(source, dict)`, or
/// `importlib.reload`. Within a single walk Rust syntax does not
/// allow duplicate top-level item names, so the observable effect
/// is across walks of the same path-keyed module: a second walk's
/// bindings supersede the first. Callers who want `sys.modules`
/// cache-hit semantics ("don't re-execute when already loaded")
/// must gate the call themselves on a prior `module_globals_lookup`
/// — the registry does not implement that gate. Production
/// callsites (`from_rust_file_entry_point_with_source`) invoke
/// the walker once per `Translation`, so the re-walk window is
/// open only in tests and cross-`Translation` workflows.
///
/// ### Scope (Slice O10 walker — Item::Enum / Item::Struct / Item::Const)
///
/// - **`Item::Enum`** → `class StepResult: ...` with each variant
///   populated as a class-dict entry (`class_set(variant_name,
///   ConstValue::HostObject(variant_class))`). The variant class is
///   a subclass of the parent enum class — Rust's `match` semantics
///   line up with Python `isinstance(x, StepResult.Continue)`.
///   Stored as `ConstValue::HostObject(<class>)`.
/// - **`Item::Struct`** → `class Foo: ...` with empty class dict.
///   Struct fields live on instances, not the class object.
///   Stored as `ConstValue::HostObject(<class>)`.
/// - **`Item::Const`** → `MODULE_NAME = <expr>` at module top
///   level. Bound to `module.__dict__[MODULE_NAME]` as the
///   evaluated value. Stored as `ConstValue::Int/Bool/UniStr/ByteStr/Float`
///   directly (no HostObject wrapper) — mirrors upstream
///   `find_global` returning `const(value)` regardless of value
///   type. RHS evaluation is delegated to [`eval_const_expr`],
///   which covers literal forms (`Lit::Int` / `Lit::Bool` /
///   `Lit::Str` / `Lit::ByteStr` / `Lit::Float` / `Lit::Char` /
///   `Lit::Byte`), unary `Neg` / `Not`, single-segment Path
///   lookups against prior `bindings` and the registry partition
///   (forward-ref + re-walk paths), and binary ops (`+` / `-` /
///   `*` / `/` / `%` / `<<` / `>>` / `&` / `|` / `^` / `==` /
///   `!=` / `<` / `<=` / `>` / `>=` / `&&` / `||`) over
///   evaluated operands. Shapes [`eval_const_expr`] does not yet
///   cover (function calls, struct literals, multi-segment paths)
///   decline-fold to `Ok(None)` and the walker silently skips —
///   those bindings then fall through to
///   `Builder::resolve_path_constant`'s mint-or-fail path at
///   call sites.
///
/// Other `Item::*` kinds (`Item::Fn`, `Item::Use`, `Item::Mod`,
/// `Item::Impl`, …) are silently skipped. `Item::Fn` for the
/// parity reason above; the others as upstream-walker follow-ups
/// (each populates `module.__dict__` at Python import time too).
/// **Immutable** `Item::Static` is admitted alongside `Item::Const`
/// (Slice O12) — upstream Python's `module.__dict__` sees both
/// shapes identically. `static mut FOO` is **NOT** registered
/// (Slice O15 / codex audit 2026-05-06): runtime mutation makes
/// the initial-value snapshot unsound for constfold reads, so the
/// adapter skips until a live-store path lands. Each future slice
/// extends the dispatch match without changing the call sites.
///
/// ### Per-module scoping (Issue 1.3, 2026-05-05)
///
/// Returns a fresh [`ModuleId`] every time — anonymous walks
/// never merge. Mirrors upstream Python's `exec(source,
/// fresh_dict)` semantic: each call runs the source against an
/// independent `__dict__`, even if the source bytes are
/// byte-identical to a prior `exec`.
///
/// This BC entry routes through [`register_rust_module_at`] with
/// `None` path. Callers that need shared registry slices across
/// multiple walks (e.g. two entry points from the same source
/// file sharing `func.__globals__`) MUST thread a stable path
/// through [`register_rust_module_at`] — that's the only orthodox
/// way to opt into upstream `sys.modules[name]` import-cache
/// behavior.
pub fn register_rust_module(file: &File) -> Result<ModuleId, AdapterError> {
    register_rust_module_at(file, None)
}

/// Path-aware sibling of [`register_rust_module`]. When
/// `source_filename` is `Some(path)`, the registry id is keyed on
/// `path` so that two walks of files at the same path converge on
/// the same [`ModuleId`] (scoped sharing of the `(module_id, name)`
/// partition; **not** upstream `sys.modules` import-cache, which
/// short-circuits the second `import` entirely — the walker
/// re-executes every time and overwrites entries last-writer-wins).
/// When `None`, the call mints a fresh [`ModuleId`] every time —
/// anonymous walks NEVER merge.
///
/// Two walk shapes:
///
/// - **Path-keyed** (`Some(path)`): every walk re-executes against
///   the shared `(module_id, name)` partition under
///   `exec(source, shared_dict)` semantics — entries from the
///   second walk overwrite those from the first per the
///   [`register_module_global`] last-writer-wins rule. Mid-walk
///   failure (an `Item::Const` RHS that surfaces `Err`) leaves
///   prior bindings of THIS walk in the partition. Callers that
///   want `sys.modules`-style "skip the second import" semantics
///   must gate the call themselves on a prior
///   [`module_globals_lookup`] for a known sentinel name. Two
///   `Translation` instances built against entry points from the
///   same file see identical registry partitions only when both
///   re-walks succeed end-to-end.
/// - **Anonymous** (`None`) ↔ `exec(source, fresh_dict)`: each
///   `exec` runs the code against a fresh namespace, even if the
///   source string is byte-identical to a prior `exec`. Two
///   anonymous walks of structurally-identical content therefore
///   produce **independent** module identities — upstream Python
///   never merges two `exec` calls into one `__dict__` based on
///   content.
///
/// **Why not content-hash anonymous?** A prior revision (Issue 2.2,
/// 2026-05-05) keyed the `None` branch on a token-stream content
/// hash so two walks of the same source string converged on one
/// `ModuleId`. Codex parity audit (2026-05-05) flagged that as a
/// regression: `exec(source, dict_a)` and `exec(source, dict_b)`
/// in upstream Python produce **distinct** `__dict__`s — content
/// equality does not imply module identity. Path-keyed sharing
/// is the only way to opt into a shared registry slice; callers
/// who need that contract MUST thread a stable path (filesystem
/// path, fixture label, anything) through `Some(...)`.
pub fn register_rust_module_at(
    file: &File,
    source_filename: Option<&str>,
) -> Result<ModuleId, AdapterError> {
    let module_id = match source_filename {
        Some(path) => ModuleId::for_path(path),
        None => ModuleId::fresh(),
    };
    // Source-order accumulator of `Item::Const` bindings produced
    // during this walk. Mirrors Python module-import semantics:
    // top-level statements run in order and each binding is visible
    // to subsequent ones via `module.__dict__`. The walker passes
    // this dict to `eval_const_expr` so compound consts (`const Y =
    // X + 1`) resolve their forward dependencies through prior
    // entries.
    let mut const_bindings: StdHashMap<String, ConstValue> = StdHashMap::new();
    for item in &file.items {
        match item {
            Item::Enum(item_enum) => {
                let name = item_enum.ident.to_string();
                // Last-writer-wins per upstream
                // `module.__dict__[name] = value` (every top-level
                // binding statement is an unconditional assignment;
                // `exec(source, dict)` / `importlib.reload`
                // semantics). Rust syntax does not allow duplicate
                // top-level item names within a single source file,
                // so the observable effect is across walks of the
                // same path-keyed `module_id`: the second walk's
                // bindings supersede the first.
                let host = build_host_class_from_enum(item_enum);
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Struct(item_struct) => {
                let name = item_struct.ident.to_string();
                let host = build_host_class_from_struct(item_struct);
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Const(item_const) => {
                let name = item_const.ident.to_string();
                // upstream Python import-time evaluation: the RHS
                // runs against the partially-built `module.__dict__`,
                // so compound expressions like `const Y: i64 = X + 1`
                // resolve `X` through the prior binding. The walker
                // threads `const_bindings` (the local source-order
                // accumulator) into the evaluator so forward
                // dependencies between sibling consts work; the
                // evaluator ALSO consults `module_globals_lookup` as
                // fallback when the path-keyed `module_id` was already
                // populated by a prior walk (Issue 2.3, 2026-05-05) —
                // mirrors upstream `module.__dict__` being the live
                // reference visible across re-imports.
                // Failure modes:
                //
                // - **Type mismatch / zero divisor** (`true + 1`,
                //   `1 / 0`) → `Err`. Walker aborts the file, matching
                //   upstream Python's import-time exception.
                // - **`i64` overflow** (`MAX + 1`, `MIN / -1`,
                //   `1 << 64`) → `Unsupported`. Upstream Python
                //   would bind a bignum at module top level; pyre
                //   adapter has no bignum carrier, and this is not an
                //   upstream `FlowingError`. Surface the unsupported
                //   carrier loudly instead of pretending Python raised.
                //   (Function-body constfold takes a different path —
                //   it declines per `operation.py:140-142` and lets
                //   runtime emit. Module top level has no analogous
                //   "let runtime do it" stage.)
                // - **Unsupported shapes** (function calls, struct
                //   literals, multi-segment paths) → `Ok(None)`.
                //   Walker-coverage gaps (Issue 2.3 follow-up).
                // - **Unresolved single-segment Path** → `Err(NameError)`,
                //   matching `flowcontext.py:853 find_global` which
                //   raises `FlowingError("global name '...' is not
                //   defined")` after both globals and builtins miss.
                //   Walker aborts the file. Walker-coverage gaps for
                //   `Item::Fn` / `Item::Use` / `Item::Mod` / `Item::Impl`
                //   surface as parity-correct hard errors here rather
                //   than silent dropped bindings.
                if let Some(value) = eval_const_expr(&item_const.expr, &const_bindings, module_id)?
                {
                    register_module_global(module_id, &name, value.clone());
                    const_bindings.insert(name, value);
                }
            }
            // `static FOO: T = <expr>;` (immutable static, Slice
            // O12 + O15). Module-globals registry binding parallels
            // `Item::Const`: the import-time RHS folds through the
            // same `eval_const_expr` and the same source-order
            // `const_bindings` accumulator. Mirrors `flowcontext.py:847
            // w_globals.value[varname]` which sees `def`-bound and
            // assignment-bound names identically.
            //
            // **Mutable static (`static mut FOO: T = <expr>;`) is
            // NOT registered** (Slice O15, codex parity audit
            // 2026-05-06). PRE-EXISTING-ADAPTATION: Rust's `mut`
            // marks the global as runtime-mutated; the registry
            // entry would be a stale snapshot of the *initial*
            // value the moment any code writes to it, and folding
            // reads against the initial value would silently
            // produce wrong constfold results. Upstream Python does
            // not have this problem (its flow analysis tracks each
            // `STORE_GLOBAL` at function-body time); pyre's adapter
            // has no analogous live-store path at module-globals
            // prepass. Until that lands, skipping mutable statics
            // entirely is the parity-safe choice — downstream
            // lookups fall through to `Builder::resolve_path_constant`'s
            // mint-or-fail path, which is the right shape for
            // "opaque runtime-mutated symbol".
            Item::Static(item_static)
                if matches!(item_static.mutability, syn::StaticMutability::None) =>
            {
                let name = item_static.ident.to_string();
                if let Some(value) = eval_const_expr(&item_static.expr, &const_bindings, module_id)?
                {
                    register_module_global(module_id, &name, value.clone());
                    const_bindings.insert(name, value);
                }
            }
            _ => {
                // PRE-EXISTING-ADAPTATION (Issue 2.3): walker
                // coverage is incomplete vs upstream
                // `module.__dict__`. Upstream Python module import
                // populates the dict for every binding statement
                // (`def`, `class`, top-level assignment,
                // `from ... import ...`, nested `import`, …).
                // Currently skipped:
                //
                // - **`Item::Fn`** — see "Why no Item::Fn?" doc on
                //   this fn for the parity reason; convergence is
                //   the M2.5g side-table walker epic.
                // - **`Item::Use`** — re-export of another item's
                //   binding. Upstream Python's `from x import y`
                //   binds `module.__dict__["y"]` to the imported
                //   value. Walker dispatch needs cross-file lookup
                //   (which itself depends on per-module scoping —
                //   see Issue 1.3).
                // - **`Item::Mod`** — submodule. Upstream
                //   `import x.y` binds `module.__dict__["x"]` to
                //   the submodule. Walker dispatch needs nested
                //   walking + module-object construction.
                // - **`Item::Impl`** — Rust associates methods with
                //   the type via `impl Foo { fn bar(&self) {} }`
                //   instead of putting them in the class dict like
                //   Python's `class Foo: def bar(self): ...`. The
                //   walker needs to redirect `bar` into the
                //   already-registered `Foo` class's class dict.
                //
                // Each follow-up slice extends this dispatch match
                // without changing the call sites.
            }
        }
    }
    Ok(module_id)
}

/// Build the `HostObject::Class` corresponding to `item_enum` and
/// populate its class dict with every variant as a child class.
///
/// Mirrors the closest Python analogue:
///
/// ```python
/// class StepResult: pass
/// class StepResult_Continue(StepResult): pass
/// class StepResult_Return(StepResult): pass
/// StepResult.Continue = StepResult_Continue
/// StepResult.Return = StepResult_Return
/// ```
///
/// Each variant's child class carries the parent in its `bases`
/// vector so `is_subclass_of(parent)` returns `true` — matches
/// upstream `classdef.py:336 ClassDef.lookup_filter` walking the
/// `__bases__` chain when computing `isinstance(x, StepResult)`
/// against an instance of `StepResult_Continue`.
///
/// The variant carrier qualname is `"<EnumName>.<VariantName>"`
/// (dot-separator) matching upstream Python's `cls.__qualname__`
/// shape for nested classes.
fn build_host_class_from_enum(item_enum: &ItemEnum) -> HostObject {
    let parent_name = item_enum.ident.to_string();
    let parent = HostObject::new_class(&parent_name, vec![]);
    for variant in &item_enum.variants {
        let v_name = variant.ident.to_string();
        let v_qualname = format!("{}.{}", parent_name, v_name);
        let v_class = HostObject::new_class(v_qualname, vec![parent.clone()]);
        parent.class_set(&v_name, ConstValue::HostObject(v_class));
    }
    parent
}

/// Evaluate a `const` RHS expression to a [`ConstValue`] using
/// `bindings` as the lookup environment for prior `const` names in
/// the same module walk.
///
/// Tri-state return:
///
/// - `Ok(Some(v))` — RHS evaluated successfully; bind `name = v`.
/// - `Ok(None)` — decline-fold. Used for **shape unsupported** at
///   this evaluator: function call, struct literal, multi-segment
///   path. Walker silently skips (walker-coverage gap, Issue 2.3
///   follow-up) and the const stays unbound; downstream lookups
///   fall through to mint-or-fail.
/// - `Err(e)` — supported shape but the operation cannot produce
///   a representable binding at module top level:
///   - **Type mismatch** in unary / binary op (`!"x"`,
///     `1 < "x"` ordering).
///   - **Zero-divisor** (`1 / 0`, `1 % False`, `1.0 / 0.0`).
///   - **Negative shift count** (`1 << -1`, `1 >> -1`) per
///     `operator.lshift/rshift` raising `ValueError` upstream.
///   - **`i64` overflow** on `Lit::Int` / unary `Neg` / binop
///     (`MAX + 1`, `MIN / -1`, `1 << 64`). Upstream Python module
///     execution would bind a bignum here and continue; pyre
///     adapter has no bignum carrier so surfaces the divergence as
///     `AdapterError::Unsupported`, not an upstream-shaped
///     `FlowingError`. **Note**: this is structurally distinct from
///     `PureOperation.constfold`'s
///     `can_overflow and type(result) is long` decline arm at
///     `operation.py:140-142`. THAT rule lives at function-body
///     constfold time — it lets the runtime evaluate the op as
///     bignum after declining. There is no analogous "let the
///     runtime do it" at module-import-time prepass.
///   - **Unresolved single-segment Path** — name not in `bindings`
///     and not in the registry partition. Mirrors
///     `flowcontext.py:853 find_global` raising
///     `FlowingError("global name '...' is not defined")` after
///     both globals and builtins miss. Walker-coverage gaps for
///     `Item::Fn` / `Item::Use` / `Item::Mod` / `Item::Impl`
///     surface here — those are fixed by extending walker
///     coverage, not by silently dropping the binding.
///   Walker propagates the error to abort the walk, matching
///   upstream Python's module-execution-aborts-on-exception
///   semantics (`Y = X + 1; X = 1` fails on `Y` with NameError;
///   the `X = 1` statement never runs).
///
/// Supported shapes:
///
/// - `Lit::Int(n)` → `ConstValue::Int(n)` (parsed as `i64`).
/// - `Lit::Bool(b)` → `ConstValue::Bool(b)`.
/// - `Lit::Str(s)` → `ConstValue::uni_str(s)`. Matches the in-body
///   `Lit::Str` lowering at `build_flow.rs::lower_literal` and
///   Python 3 unicode-string semantics — every `"..."` literal is
///   unicode regardless of where it appears.
/// - `Lit::ByteStr(s)` → `ConstValue::byte_str(s)` (Rust `b"..."`
///   bytes literal stays bytes).
/// - `Lit::Float(f)` → `ConstValue::Float(f.to_bits())`. Same shape
///   as `build_flow.rs::lower_literal::Lit::Float`. Out-of-`f64`-range
///   raises `AdapterError::Unsupported` at module top level (no carrier).
/// - `Lit::Char(c)` → `ConstValue::uni_str(c.to_string())`. Single-
///   char unicode string — upstream RPython has no `char` type,
///   single-char strings fill the role.
/// - `Lit::Byte(b)` → `ConstValue::ByteStr(vec![b])`. Mirrors
///   `build_flow.rs::lower_literal::Lit::Byte`.
/// - `-<Lit::Int>` / `-<Lit::Float>` (unary negation over a
///   numeric literal) → `ConstValue::Int(-n)` /
///   `ConstValue::Float(-f).to_bits()`. `syn` parses `const X:
///   i64 = -1` as `Expr::Unary { op: Neg, expr: Lit(1) }` (and
///   likewise for floats), not a signed literal, so unwrap one
///   level. Mirrors `operation.rs::pyfunc`'s
///   `(OpKind::Neg, [ConstValue::Int])` and
///   `(OpKind::Neg, [ConstValue::Float])` arms so a top-level
///   const and an in-body unary `-` agree on the value.
/// - **`Expr::Path` (single segment)** → resolution chain mirrors
///   upstream `flowcontext.py:845-853 find_global`:
///   1. `bindings.get(name)` — per-walk source-order accumulator,
///      stand-in for the in-progress module dict.
///   2. `module_globals_lookup(module_id, name)` — registry
///      partition; covers re-walks where the second walk's fresh
///      `bindings` is empty but the partition was populated by the
///      first walk.
///   3. `pyre_stdlib_lookup(name)` — closed-world builtins
///      (`Ok` / `Some` / `Err` / `Result` / `Option`). Mirrors
///      upstream's `getattr(__builtin__, varname)` second arm.
///   4. NameError. All three channels miss matches upstream's
///      `AttributeError → FlowingError` final step.
///   Multi-segment paths fall through to `None` (a path like
///   `mod::CONST_X` would require cross-file lookup and is out
///   of scope for this slice).
/// - **`Expr::Binary { Add | Sub | Mul | Div | Rem | Shl | Shr |
///   BitAnd | BitOr | BitXor, lhs, rhs }`** over `Int` operands →
///   `Int(a OP b)`. `Div` / `Rem` use Python floor-div / floor-mod
///   semantics (negative-divisor sign-flip) line-by-line per
///   `rpython/rtyper/rint.py:398 ll_int_py_div` /
///   `:496 ll_int_py_mod`, matching `flowspace::operation`'s
///   `int_py_floor_div` / `int_py_mod` so a `const Y = 3 / -2` at
///   module top level and `3 / -2` inside a function body produce
///   the same value. Zero divisor surfaces as
///   `Err(ZeroDivisionError)` per upstream `operator.floordiv(_, 0)`
///   raising at import time. `i64` overflow on `+` / `-` / `*` /
///   `i64::MIN / -1` / shifts that don't fit in `u32` / `>= 64`
///   surface as `AdapterError::Unsupported` (pyre adapter has no bignum
///   carrier; the function-body `PureOperation.constfold` decline
///   path at `operation.py:140-142` does not apply at import-time
///   prepass — see fn-level doc).
/// - **`Expr::Binary { Add | Sub | Mul | Div, lhs, rhs }`** over
///   `Float` operands → `Float(a OP b)`. Mirrors `operation.rs::pyfunc`
///   Float arms (`operation.rs:1367-1383`). Rust `BinOp::Div` over
///   `f64` is true division (matches Python `/`). `Float / 0.0`
///   raises `Err(ZeroDivisionError)` per Python semantics.
/// - **`Expr::Binary { Eq | Ne, lhs, rhs }`** over any operand
///   pair → `Bool(...)`. Same-type pairs use the matching arm's
///   structural equality; mixed-type pairs (e.g. `Int` vs `UniStr`)
///   return `Bool(false)` for `Eq` / `Bool(true)` for `Ne` — Python
///   3 does NOT raise on `==` / `!=` between distinct primitive
///   types. Numeric coercion for `True == 1` etc. is folded by
///   `HLOperation::constfold` once the const reaches the SSA layer.
/// - **`Expr::Binary { Lt | Le | Gt | Ge, lhs, rhs }`** over
///   same-type operands → `Bool(...)`. Mixed-type ordering raises
///   `Err(TypeError)` matching Python 3's
///   `'<' not supported between instances of …`.
/// - **`Expr::Binary { And | Or, lhs, rhs }`** over `Bool`
///   operands → `Bool(...)`. Rust `&&`/`||` are typed `bool ->
///   bool -> bool` so unlike Python's value-returning
///   short-circuit, the result is always `Bool`.
/// - **`Expr::Unary { Not, expr }`** over `Bool` → `Bool(!b)` (Rust
///   `!bool` is logical negation), or over `Int` → `Int(!n)` (Rust
///   `!int` is bitwise complement, mirroring Python `~`).
/// - **`Expr::Paren(expr)` / `Expr::Group(expr)`** — transparent
///   delegation to the inner expression. Upstream Python has no
///   parenthesisation node (parens evaporate at parse time and only
///   affect operator precedence) so a `const X: bool = (1 == 2)`
///   resolves identically to `const X: bool = 1 == 2`. Mirrors the
///   body lowerer's transparency at `build_flow.rs:1317`.
fn eval_const_expr(
    expr: &Expr,
    bindings: &StdHashMap<String, ConstValue>,
    module_id: ModuleId,
) -> Result<Option<ConstValue>, AdapterError> {
    let raise = |reason: String| AdapterError::Flowing { reason };
    let unsupported = |reason: String| AdapterError::Unsupported { reason };
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            // Literal that does not fit in `i64`. Upstream Python
            // module-top-level `X = 9223372036854775808` would bind
            // `X` to a Python long (bignum) without raising — Python
            // ints are arbitrary precision. Pyre's adapter has no
            // bignum carrier, so the binding cannot be created. The
            // orthodox response is `AdapterError::Unsupported`:
            // declining to `Ok(None)` would silently produce a
            // `module.__dict__` missing a name that upstream Python
            // would have bound, creating worse divergence than
            // aborting the walk. This is not a `FlowingError` because
            // upstream Python does not raise.
            //
            // Note: this is structurally distinct from
            // `PureOperation.constfold`'s `can_overflow and type(result)
            // is long` decline arm at `operation.py:140-142`. THAT
            // rule lives at function-body constfold time — it lets the
            // runtime evaluate the op as bignum. There is no analogous
            // "let the runtime do it" at module-import-time prepass:
            // upstream Python would have already produced the bignum
            // by the time this prepass ran.
            Lit::Int(n) => match n.base10_parse::<i64>() {
                Ok(v) => Ok(Some(ConstValue::Int(v))),
                Err(_) => Err(unsupported(format!(
                    "integer literal {} exceeds i64 carrier; bignum constants are not supported",
                    n
                ))),
            },
            Lit::Bool(b) => Ok(Some(ConstValue::Bool(b.value))),
            // `"..."` literal — unicode. Same shape as
            // `build_flow.rs::lower_literal::Lit::Str` so the
            // identical `"abc"` source carries the identical
            // ConstValue regardless of position.
            Lit::Str(s) => Ok(Some(ConstValue::uni_str(s.value()))),
            // `b"..."` literal — bytes. Mirrors
            // `build_flow.rs::lower_literal::Lit::ByteStr`.
            Lit::ByteStr(s) => Ok(Some(ConstValue::ByteStr(s.value()))),
            // Float literal — `ConstValue::Float` stores
            // `f64::to_bits()` so the enum keeps `Eq + Hash`
            // (`model.rs:1696-1701`). Matches the in-body shape
            // at `build_flow.rs::lower_literal::Lit::Float`. Out-
            // of-`f64`-range surfaces `AdapterError::Unsupported` for
            // the same reason `Lit::Int` overflow does: upstream
            // Python would still bind a value (even if `inf`), but
            // the representation we encode through `f64::to_bits()`
            // cannot carry the source-text the user wrote.
            Lit::Float(f) => match f.base10_parse::<f64>() {
                Ok(v) => Ok(Some(ConstValue::Float(v.to_bits()))),
                Err(e) => Err(unsupported(format!(
                    "float literal out of f64 range; float carrier cannot represent source literal: {e}"
                ))),
            },
            // Rust `'a'` char literal — single Unicode scalar.
            // Upstream RPython has no `char` type; single-char
            // strings fill the role (`model.py:658`,
            // `operation.py` string ops accept `len == 1` like
            // any other unicode). Emit as `ConstValue::UniStr`
            // matching `build_flow.rs::lower_literal::Lit::Char`.
            Lit::Char(ch) => Ok(Some(ConstValue::uni_str(ch.value().to_string()))),
            // Rust `b'a'` byte literal — single byte. Mirrors
            // `build_flow.rs::lower_literal::Lit::Byte` which
            // emits `ConstValue::ByteStr(vec![b])`.
            Lit::Byte(b) => Ok(Some(ConstValue::ByteStr(vec![b.value()]))),
            _ => Ok(None),
        },
        // `const X: i64 = -1` — `syn` parses as `Unary { op: Neg,
        // expr: Lit(1) }` rather than a signed literal. Unwrap one
        // level so the common signed-int form is recognised.
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => {
            let Some(v) = eval_const_expr(expr, bindings, module_id)? else {
                return Ok(None);
            };
            match v {
                // `-(i64::MIN)` overflows the `i64` carrier. Upstream
                // Python would bind `2**63` (bignum) without raising;
                // pyre adapter has no bignum carrier, so surface as
                // `AdapterError::Unsupported`. See `Lit::Int` doc
                // above for why decline-fold (Ok(None)) is wrong at
                // module top level (it lives at function-body constfold
                // time per `operation.py:140-142`, not import-time
                // prepass).
                ConstValue::Int(n) => match n.checked_neg() {
                    Some(neg) => Ok(Some(ConstValue::Int(neg))),
                    None => Err(unsupported(format!(
                        "-i64::MIN ({}) exceeds i64 carrier; bignum constants are not supported",
                        n
                    ))),
                },
                // `-3.14` / `-2.5` etc. — `syn` parses the leading
                // minus as `Expr::Unary { Neg, Lit::Float(...) }`,
                // not a signed float literal. Mirrors
                // `operation.rs::pyfunc`'s
                // `(OpKind::Neg, [ConstValue::Float(bits)]) =>
                //   Some(ConstValue::float(-f64::from_bits(*bits)))`
                // so a module-top `const X: f64 = -3.14` and an
                // in-body `-3.14` agree on the bit pattern.
                ConstValue::Float(bits) => Ok(Some(ConstValue::float(-f64::from_bits(bits)))),
                other => Err(raise(format!(
                    "TypeError: unary `-` operand must be Int or Float, got {other:?}"
                ))),
            }
        }
        // `const X: bool = !true` over Bool → logical negation.
        // `const X: i64 = !0` over Int → bitwise complement
        // (Rust's `!` on integers is the same as Python's `~`).
        Expr::Unary(ExprUnary {
            op: UnOp::Not(_),
            expr,
            ..
        }) => {
            let Some(v) = eval_const_expr(expr, bindings, module_id)? else {
                return Ok(None);
            };
            match v {
                ConstValue::Bool(b) => Ok(Some(ConstValue::Bool(!b))),
                ConstValue::Int(n) => Ok(Some(ConstValue::Int(!n))),
                other => Err(raise(format!(
                    "TypeError: unary `!` operand must be Bool or Int, got {other:?}"
                ))),
            }
        }
        // `const Y: i64 = X` — single-segment path. Resolution
        // order matches upstream Python's import-time name
        // resolution against `module.__dict__`: per-walk source-
        // order `bindings` first, then the registry partition
        // (Issue 2.3 — covers re-walk: a path-keyed `module_id`
        // already populated by a prior walk has the entry in
        // the registry but not in this walk's fresh `bindings`).
        // Multi-segment paths fall through to `Ok(None)` (cross-
        // file lookup is a separate slice).
        //
        // Unresolved single-segment names raise
        // `FlowingError("global name '...' is not defined")`,
        // matching upstream `flowcontext.py:853 find_global` which
        // raises after both globals and builtins miss. Aborts the
        // walker — same as Python module execution where
        // `Y = X + 1; X = 1` fails with NameError on Y and the
        // remaining statements never execute. Walker gaps for
        // `Item::Use` / `Item::Mod` / `Item::Impl` (and `Item::Fn`)
        // surface here as parity-correct hard errors rather than
        // silent dropped bindings — pyre adapter cannot pretend a
        // name resolves when upstream Python's same execution would
        // have aborted.
        Expr::Path(ExprPath {
            qself: None, path, ..
        }) if path.segments.len() == 1 => {
            let seg = &path.segments[0];
            if !seg.arguments.is_empty() {
                return Ok(None);
            }
            let name = seg.ident.to_string();
            // Resolution order mirrors `flowcontext.py:845-853 find_global`:
            //
            //     try:
            //         value = w_globals.value[varname]   # 1. globals
            //     except KeyError:
            //         try:
            //             value = getattr(__builtin__, varname)  # 2. builtins
            //         except AttributeError:
            //             raise FlowingError("global name '%s' is not defined")
            //
            // pyre's adapter has two channels in place of one Python
            // dict: per-walk source-order `bindings` (the in-progress
            // module dict) and `module_globals_lookup` (re-walks /
            // pre-walk registry partition). Both stand in for the
            // `w_globals.value[varname]` lookup. The closed-world
            // `pyre_stdlib_lookup` is the `__builtin__` analogue
            // (`Ok` / `Some` / `Err` / `Result` / `Option` HostObject
            // singletons; documented at `host_env.rs:170-191` as the
            // adapter's auxiliary builtins). NameError fires only when
            // all three channels miss, matching upstream's final
            // `AttributeError → FlowingError` step.
            match bindings
                .get(&name)
                .cloned()
                .or_else(|| module_globals_lookup(module_id, &name))
                .or_else(|| pyre_stdlib_lookup(&name).map(ConstValue::HostObject))
            {
                Some(v) => Ok(Some(v)),
                None => Err(raise(format!("global name '{name}' is not defined"))),
            }
        }
        // `const Y = X + 1` etc — operator dispatch over evaluated
        // operands. Type-mismatch and zero-divisor surface as
        // `Err`, matching upstream Python's import-time
        // exception. `i64` overflow at module top level is a local
        // carrier limitation: upstream Python would bind a bignum, so
        // the adapter returns `Unsupported` rather than silently
        // dropping the binding or pretending Python raised. Truly
        // unsupported operator/operand combinations return `Ok(None)`
        // (silent skip).
        //
        // **Short-circuit `&&` / `||`** (codex parity audit,
        // 2026-05-05): both Rust source and upstream Python's
        // `and`/`or` are short-circuit at the source level —
        // `false && BAD` evaluates to `false` without touching
        // `BAD`, and `true || BAD` evaluates to `true` similarly.
        // The naive "evaluate both operands then dispatch" path
        // would force-evaluate the RHS, diverging in cases where
        // the RHS would itself raise (e.g. unbound name, division
        // by zero). LHS is evaluated first; if it's a `Bool` whose
        // value determines the result, the RHS is skipped.
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let Some(lhs) = eval_const_expr(left, bindings, module_id)? else {
                return Ok(None);
            };
            if let ConstValue::Bool(b) = lhs {
                match op {
                    BinOp::And(_) if !b => return Ok(Some(ConstValue::Bool(false))),
                    BinOp::Or(_) if b => return Ok(Some(ConstValue::Bool(true))),
                    _ => {}
                }
            }
            let Some(rhs) = eval_const_expr(right, bindings, module_id)? else {
                return Ok(None);
            };
            eval_binop(op, &lhs, &rhs)
        }
        // `Expr::Paren` / `Expr::Group` are transparent. Upstream
        // Python has no parenthesisation node — parens evaporate at
        // parse time and only affect operator precedence — so a
        // `const X: bool = (1 == 2)` resolves identically to
        // `const X: bool = 1 == 2`. Mirrors the body lowerer's
        // transparency at `build_flow.rs:1317
        // Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr)`.
        // `Expr::Group` is the proc-macro-emitted invisible grouping
        // wrapper; same semantic.
        Expr::Paren(syn::ExprParen { expr, .. }) | Expr::Group(syn::ExprGroup { expr, .. }) => {
            eval_const_expr(expr, bindings, module_id)
        }
        _ => Ok(None),
    }
}

/// Evaluate a Rust `BinOp` over two evaluated `ConstValue` operands.
///
/// Tri-state return mirrors [`eval_const_expr`]:
///
/// - `Ok(Some(v))` — success.
/// - `Ok(None)` — decline-fold for **shape unsupported** by this
///   evaluator (operator/operand pair never reachable from typed
///   Rust source, e.g. `BinOp::Add` over `(UniStr, Int)`). Walker
///   silently skips so the const stays unbound and downstream
///   lookups fall through to mint-or-fail. **Note**: `i64` overflow
///   on `+` / `-` / `*` / `/` / `%` / `<<` does NOT decline-fold at
///   module top level — it returns `AdapterError::Unsupported`
///   instead. The
///   `PureOperation.constfold` decline arm at `operation.py:140-142`
///   lives at *function-body* constfold time, where the runtime
///   evaluates the op as bignum after the fold is declined; there
///   is no analogous "let the runtime do it" at module-import-time
///   prepass — upstream Python would have already produced the
///   bignum and bound it. Pyre adapter surfaces the divergence
///   (no bignum carrier) loudly.
///   `>>` count `>= 64` is the one exception: `operation.py:484`
///   registers `rshift` without `ovf=True`, and Python's arithmetic
///   right shift saturates rather than producing a long, so the
///   constfold *can* still produce a representable result.
/// - `Err(e)` — supported shape but operation cannot produce a
///   representable binding:
///   - **Type mismatch** in unary / binary op
///     (`!"x"`, `1 < "x"` ordering).
///   - **Zero-divisor** (`1 / 0`, `1 // False`, `1 % 0`,
///     `1.0 / 0.0`).
///   - **Negative shift count** (`1 << -1`, `1 >> -1`).
///     Upstream `operator.lshift` / `operator.rshift` raise
///     `ValueError: negative shift count` directly; the
///     constfold's `try/except Exception` (`operation.py:120-127`)
///     captures and re-raises as `FlowingError`.
///   - **`i64` overflow** on `Add` / `Sub` / `Mul` / `Div` (`MIN /
///     -1`) / `Shl` (`count >= 64`). Upstream Python would bind
///     a bignum; pyre adapter has no bignum carrier so surfaces
///     the divergence as `AdapterError::Unsupported`.
///   Walker propagates the error to abort the walk, matching
///   upstream Python's module-execution-aborts-on-exception
///   semantics.
fn eval_binop(
    op: &BinOp,
    lhs: &ConstValue,
    rhs: &ConstValue,
) -> Result<Option<ConstValue>, AdapterError> {
    let raise = |reason: String| AdapterError::Flowing { reason };
    let unsupported = |reason: String| AdapterError::Unsupported { reason };
    let int_zerodiv = |op_name: &str| {
        Err(raise(format!(
            "ZeroDivisionError: integer {op_name} by zero"
        )))
    };
    let int_overflow = |op_name: &str, a: i64, b: i64| {
        Err(unsupported(format!(
            "i64 {op_name} ({a} {op_name} {b}) exceeds i64 carrier; bignum constants are not supported"
        )))
    };
    match (lhs, rhs) {
        (ConstValue::Int(a), ConstValue::Int(b)) => match op {
            // `+` / `-` / `*` overflow → `Unsupported`. At
            // module top level upstream Python would bind a bignum;
            // pyre adapter has no bignum carrier so the divergence
            // surfaces loudly. (Function-body constfold is a
            // *different* path — it declines per
            // `operation.py:140-142`.)
            BinOp::Add(_) => match a.checked_add(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("+", *a, *b),
            },
            BinOp::Sub(_) => match a.checked_sub(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("-", *a, *b),
            },
            BinOp::Mul(_) => match a.checked_mul(*b) {
                Some(v) => Ok(Some(ConstValue::Int(v))),
                None => int_overflow("*", *a, *b),
            },
            // `Div` and `Rem` use Python floor-div / floor-mod
            // semantics (negative-divisor sign-flip) — same helpers
            // as `flowspace::operation::int_py_floor_div` /
            // `int_py_mod` so a `const Y: i64 = 3 / -2` and the
            // function-body `3 / -2` produce the same value. Zero
            // divisor → `Err(ZeroDivisionError)` per upstream
            // `operator.floordiv(_, 0)` raising. `i64::MIN / -1`
            // overflow → `Unsupported` at module top level
            // (upstream binds bignum `2**63`).
            BinOp::Div(_) => {
                if *b == 0 {
                    return int_zerodiv("division");
                }
                match int_py_floor_div(*a, *b) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("/", *a, *b),
                }
            }
            BinOp::Rem(_) => {
                if *b == 0 {
                    return int_zerodiv("modulo");
                }
                // `x % -1 == 0` for any `x` upstream — fold
                // unconditionally. `int_py_mod` would otherwise
                // route through `i64::MIN.checked_rem(-1)` which
                // returns `None` (the *quotient* `2^63` overflows,
                // even though the remainder `0` itself fits),
                // declining a fold that has a well-defined value.
                if *b == -1 {
                    return Ok(Some(ConstValue::Int(0)));
                }
                // `int_py_mod` only returns `None` on
                // `i64::MIN.checked_rem(-1)` (handled above) —
                // every other `(x, y)` with `y != 0` produces a
                // well-defined result. Unwrap defensively, but if
                // a future helper change introduces another
                // overflow path, surface it as Err.
                match int_py_mod(*a, *b) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("%", *a, *b),
                }
            }
            // `<<` — Negative `count` raises `ValueError: negative
            // shift count` upstream (`operator.lshift(_, -1)`).
            // `count >= 64` overflows the i64 carrier; upstream
            // Python would bind a bignum, pyre adapter cannot, so
            // surface as `Unsupported`.
            BinOp::Shl(_) => {
                if *b < 0 {
                    return Err(raise(format!("ValueError: negative shift count {b}")));
                }
                match u32::try_from(*b).ok().and_then(|n| a.checked_shl(n)) {
                    Some(v) => Ok(Some(ConstValue::Int(v))),
                    None => int_overflow("<<", *a, *b),
                }
            }
            // `>>` — Python's arithmetic right shift saturates
            // on counts `>= 64`: any non-negative `a` shifts to
            // `0`, any negative `a` sign-extends to `-1`. Rust's
            // `i64 >> 64` panics in debug, so we explicit-saturate.
            // `rshift` is NOT registered with `ovf=True` upstream
            // (`operation.py:484 add_operator('rshift', 2,
            // dispatch=2, pure=True)`), so the long-decline arm
            // doesn't apply — we MUST fold the saturation result.
            // Negative count surfaces as `ValueError` like `Shl`.
            BinOp::Shr(_) => {
                if *b < 0 {
                    return Err(raise(format!("ValueError: negative shift count {b}")));
                }
                let saturated = if *a < 0 { -1 } else { 0 };
                let folded = u32::try_from(*b)
                    .ok()
                    .and_then(|n| a.checked_shr(n))
                    .unwrap_or(saturated);
                Ok(Some(ConstValue::Int(folded)))
            }
            BinOp::BitAnd(_) => Ok(Some(ConstValue::Int(a & b))),
            BinOp::BitOr(_) => Ok(Some(ConstValue::Int(a | b))),
            BinOp::BitXor(_) => Ok(Some(ConstValue::Int(a ^ b))),
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        // Float-Float arithmetic mirrors `operation.rs::pyfunc`'s
        // `(OpKind::{Add,Sub,Mul,TrueDiv,Mod}, [ConstValue::Float,
        // ConstValue::Float])` arms (`operation.rs:1336-1393`). Rust
        // `BinOp::Div` over `f64` IS Python's `/` (true division),
        // not floor-division — Rust has no integer-floor `/` for
        // floats. `BinOp::Rem` is Rust's `%`, which we route through
        // `float_py_mod` so the const-evaluator and the function-body
        // constfold produce bit-identical results (signed-zero
        // copysign + sign-of-denominator correction per upstream
        // `pypy/objspace/std/floatobject.py:543-563 descr_mod`).
        // Comparison ops use `partial_cmp` semantics on the underlying
        // `f64` (NaN compares as false everywhere, matching Python
        // `float('nan') < 1.0 == False`).
        (ConstValue::Float(a_bits), ConstValue::Float(b_bits)) => {
            let a = f64::from_bits(*a_bits);
            let b = f64::from_bits(*b_bits);
            match op {
                BinOp::Add(_) => Ok(Some(ConstValue::float(a + b))),
                BinOp::Sub(_) => Ok(Some(ConstValue::float(a - b))),
                BinOp::Mul(_) => Ok(Some(ConstValue::float(a * b))),
                BinOp::Div(_) => {
                    if b == 0.0 {
                        return Err(raise(
                            "ZeroDivisionError: float division by zero".to_string(),
                        ));
                    }
                    Ok(Some(ConstValue::float(a / b)))
                }
                BinOp::Rem(_) => {
                    if b == 0.0 {
                        return Err(raise("ZeroDivisionError: float modulo".to_string()));
                    }
                    Ok(Some(ConstValue::float(float_py_mod(a, b))))
                }
                // `==` / `!=` use IEEE 754 semantics on `f64`
                // (`NaN != NaN`), matching Python 3's
                // `float('nan') == float('nan') is False`.
                BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
                BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
                BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
                BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
                BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
                BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
                _ => Ok(None),
            }
        }
        (ConstValue::Bool(a), ConstValue::Bool(b)) => match op {
            // Rust `&&` / `||` are short-circuit at the source
            // level — the lowerer in `eval_const_expr` handles the
            // short-circuit semantics before we reach here. By the
            // time `eval_binop` runs both operands have been fully
            // evaluated; the operator semantic is then `bool && bool`
            // / `bool || bool` (boolean conjunction / disjunction
            // returning bool, not Python's value-returning short-
            // circuit).
            BinOp::And(_) => Ok(Some(ConstValue::Bool(*a && *b))),
            BinOp::Or(_) => Ok(Some(ConstValue::Bool(*a || *b))),
            BinOp::BitAnd(_) => Ok(Some(ConstValue::Bool(*a & *b))),
            BinOp::BitOr(_) => Ok(Some(ConstValue::Bool(*a | *b))),
            BinOp::BitXor(_) => Ok(Some(ConstValue::Bool(*a ^ *b))),
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            _ => Ok(None),
        },
        (ConstValue::UniStr(a), ConstValue::UniStr(b)) => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        (ConstValue::ByteStr(a), ConstValue::ByteStr(b)) => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(a == b))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(a != b))),
            BinOp::Lt(_) => Ok(Some(ConstValue::Bool(a < b))),
            BinOp::Le(_) => Ok(Some(ConstValue::Bool(a <= b))),
            BinOp::Gt(_) => Ok(Some(ConstValue::Bool(a > b))),
            BinOp::Ge(_) => Ok(Some(ConstValue::Bool(a >= b))),
            _ => Ok(None),
        },
        // Mixed-numeric arm: cross-type pairs over `Int` / `Bool` /
        // `Float` follow Python's numeric tower (`bool ⊂ int ⊂ float`)
        // via the same `coerce_arith` / `coerce_int_pair` helpers
        // `flowspace::operation::pyfunc` uses, so the const-evaluator
        // and the function-body constfold agree on values for
        // `Int + Float`, `Bool + Int`, `Int % Float`, `True / 0.0`,
        // etc. Mirrors upstream `PureOperation.constfold`
        // (`operation.py:120-127`) which calls `operator.<op>(*args)`
        // directly — Python's runtime coerces at the operator level.
        //
        // Reached only for cross-type pairs (same-type pairs match
        // typed arms above). Bool/Bool arithmetic is intentionally
        // NOT routed here; the typed `(Bool, Bool)` arm declines to
        // `Ok(None)` for arithmetic ops, matching the prior behavior
        // where the walker silent-skips that combination.
        (_, _) if is_foldable_numeric(lhs) && is_foldable_numeric(rhs) => match op {
            BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match (op, pair) {
                    (BinOp::Add(_), ArithOps::Int(x, y)) => match x.checked_add(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("+", x, y),
                    },
                    (BinOp::Add(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x + y))),
                    (BinOp::Sub(_), ArithOps::Int(x, y)) => match x.checked_sub(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("-", x, y),
                    },
                    (BinOp::Sub(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x - y))),
                    (BinOp::Mul(_), ArithOps::Int(x, y)) => match x.checked_mul(y) {
                        Some(v) => Ok(Some(ConstValue::Int(v))),
                        None => int_overflow("*", x, y),
                    },
                    (BinOp::Mul(_), ArithOps::Float(x, y)) => Ok(Some(ConstValue::float(x * y))),
                    _ => unreachable!(),
                }
            }
            BinOp::Div(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match pair {
                    ArithOps::Int(x, y) => {
                        if y == 0 {
                            return int_zerodiv("division");
                        }
                        match int_py_floor_div(x, y) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("/", x, y),
                        }
                    }
                    ArithOps::Float(x, y) => {
                        if y == 0.0 {
                            return Err(raise(
                                "ZeroDivisionError: float division by zero".to_string(),
                            ));
                        }
                        Ok(Some(ConstValue::float(x / y)))
                    }
                }
            }
            BinOp::Rem(_) => {
                let pair = coerce_arith(lhs, rhs).expect("both operands numeric");
                match pair {
                    ArithOps::Int(x, y) => {
                        if y == 0 {
                            return int_zerodiv("modulo");
                        }
                        if y == -1 {
                            return Ok(Some(ConstValue::Int(0)));
                        }
                        match int_py_mod(x, y) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("%", x, y),
                        }
                    }
                    ArithOps::Float(x, y) => {
                        if y == 0.0 {
                            return Err(raise("ZeroDivisionError: float modulo".to_string()));
                        }
                        Ok(Some(ConstValue::float(float_py_mod(x, y))))
                    }
                }
            }
            // Shifts and bitwise: int-only coercion; Float operand
            // surfaces `TypeError` per Python `1 << 1.0`. `coerce_int_pair`
            // returns `None` when either side is a Float.
            BinOp::Shl(_)
            | BinOp::Shr(_)
            | BinOp::BitAnd(_)
            | BinOp::BitOr(_)
            | BinOp::BitXor(_) => {
                let Some((x, y)) = coerce_int_pair(lhs, rhs) else {
                    return Err(raise(format!(
                        "TypeError: unsupported operand types for binop: {lhs:?} OP {rhs:?}"
                    )));
                };
                match op {
                    BinOp::Shl(_) => {
                        if y < 0 {
                            return Err(raise(format!("ValueError: negative shift count {y}")));
                        }
                        match u32::try_from(y).ok().and_then(|n| x.checked_shl(n)) {
                            Some(v) => Ok(Some(ConstValue::Int(v))),
                            None => int_overflow("<<", x, y),
                        }
                    }
                    BinOp::Shr(_) => {
                        if y < 0 {
                            return Err(raise(format!("ValueError: negative shift count {y}")));
                        }
                        Ok(Some(ConstValue::Int(if y >= 64 {
                            if x < 0 { -1 } else { 0 }
                        } else {
                            x >> (y as u32)
                        })))
                    }
                    BinOp::BitAnd(_) => Ok(Some(ConstValue::Int(x & y))),
                    BinOp::BitOr(_) => Ok(Some(ConstValue::Int(x | y))),
                    BinOp::BitXor(_) => Ok(Some(ConstValue::Int(x ^ y))),
                    _ => unreachable!(),
                }
            }
            // Eq / Ne / Lt / Le / Gt / Ge over numeric pairs go through
            // the precision-correct helpers (handle ints beyond the
            // f64 53-bit mantissa boundary per upstream
            // `floatobject.py:135-148`).
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(python_eq_const(lhs, rhs)))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(!python_eq_const(lhs, rhs)))),
            BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => match cmp_fold(lhs, rhs) {
                Some(o) => {
                    let v = match op {
                        BinOp::Lt(_) => o.is_lt(),
                        BinOp::Le(_) => o.is_le(),
                        BinOp::Gt(_) => o.is_gt(),
                        BinOp::Ge(_) => o.is_ge(),
                        _ => unreachable!(),
                    };
                    Ok(Some(ConstValue::Bool(v)))
                }
                None => Ok(None),
            },
            _ => Ok(None),
        },
        // Catch-all for non-numeric cross-type pairs.
        //
        // **Eq / Ne / Lt / Le / Gt / Ge** delegate to the same helpers
        // that `pyfunc` uses for function-body constfold so the
        // const-evaluator and the in-flow constfold agree on Python
        // semantics. Concretely:
        //
        //   `Int(1) == "1"`         → False  (`python_eq_const` falls
        //                                     through to derived PartialEq)
        //
        // Upstream `PureOperation.constfold` calls
        // `operator.eq(*args)` / `operator.lt(*args)` directly, so
        // Python's runtime drives the result.
        //
        // **Cross-type ordering** — `cmp_fold` returns `None` for
        // genuinely-incomparable pairs (`Int <-> UniStr` etc.).
        // Python 3 raises `TypeError("'<' not supported between
        // instances of …")` for those; surface as walker error.
        //
        // **All other ops** (`+`, `-`, `*`, `/`, etc.) over distinct
        // types: `TypeError` per upstream `operator.add` etc.
        _ => match op {
            BinOp::Eq(_) => Ok(Some(ConstValue::Bool(python_eq_const(lhs, rhs)))),
            BinOp::Ne(_) => Ok(Some(ConstValue::Bool(!python_eq_const(lhs, rhs)))),
            BinOp::Lt(_) | BinOp::Le(_) | BinOp::Gt(_) | BinOp::Ge(_) => match cmp_fold(lhs, rhs) {
                Some(o) => {
                    let v = match op {
                        BinOp::Lt(_) => o.is_lt(),
                        BinOp::Le(_) => o.is_le(),
                        BinOp::Gt(_) => o.is_gt(),
                        BinOp::Ge(_) => o.is_ge(),
                        _ => unreachable!(),
                    };
                    Ok(Some(ConstValue::Bool(v)))
                }
                None => Err(raise(format!(
                    "TypeError: ordering comparison not supported between {lhs:?} and {rhs:?}"
                ))),
            },
            _ => Err(raise(format!(
                "TypeError: unsupported operand types for binop: {lhs:?} OP {rhs:?}"
            ))),
        },
    }
}

/// Build the `HostObject::Class` corresponding to `item_struct`.
/// The class dict is left empty — Rust struct fields are accessed on
/// *instances*, not the class object, so `Foo.x` is a meaningful
/// expression only when `Foo` is a value (e.g. an enum variant
/// constructor with named fields like `Foo::Variant { x }`).
///
/// `pyre`'s match-arm cascade (`build_flow.rs::lower_match_variant_cascade`)
/// uses the class identity for `isinstance(scrutinee, Foo)` at the
/// fork; named-field bindings then emit `getattr(scrutinee, "x")` on
/// the *instance* (not the class object) — the empty class dict
/// matches that semantic exactly.
fn build_host_class_from_struct(item_struct: &ItemStruct) -> HostObject {
    let name = item_struct.ident.to_string();
    HostObject::new_class(&name, vec![])
}

/// Test-only accessor for the per-`ModuleId` slice of the
/// module-globals registry. Used by `interactive.rs::tests` to
/// verify that the file-aware entry's walker pre-pass registered
/// sibling items before the entry-point body lowered. Re-exports
/// `module_globals_lookup` under a `pub(crate)` name so cross-
/// module tests can read the registry without exposing the
/// `pub(super)` API surface.
#[allow(dead_code)]
pub(crate) fn module_globals_for_test(module_id: ModuleId, name: &str) -> Option<ConstValue> {
    module_globals_lookup(module_id, name)
}

/// Inner builder shared by [`build_host_function_from_rust`] (full
/// body-lowering path) and
/// [`build_host_function_metadata_from_rust`] (import-time-only
/// path). Returns the `HostObject` plus the underlying `HostCode` /
/// `GraphFunc` so the full path can wire `graph.func` after running
/// `build_flow_from_rust`.
struct HostMetadataParts {
    host: HostObject,
    host_code: HostCode,
    gf: GraphFunc,
}

fn build_host_metadata_parts(
    item_fn: &ItemFn,
    module_id: ModuleId,
    source_filename: Option<&str>,
    source_text: Option<&str>,
) -> Result<HostMetadataParts, AdapterError> {
    let argnames = extract_argnames(item_fn)?;
    let name = item_fn.sig.ident.to_string();
    // upstream `pygraph.py:14-16`: `locals = [None] * code.co_nlocals;
    //   for i in range(code.formalargcount): locals[i] = Variable(...)`.
    // Synthesize the same shape by extending `co_varnames` with every
    // extra local the body walker introduced (let-pattern / for-pattern
    // identifiers), so `co_nlocals = formalargcount + extras`.
    let extras = collect_local_names(item_fn, &argnames);
    let mut co_varnames = argnames.clone();
    co_varnames.extend(extras.iter().cloned());
    let nlocals = co_varnames.len() as u32;

    // upstream `objspace.py:33-35` `_assert_rpythonic`: any RPython
    // function's code object must carry `CO_NEWLOCALS`. The adapter
    // bypasses `_assert_rpythonic` (no `build_flow` call) but the
    // synthetic HostCode must still satisfy the invariant so later
    // consumers can re-verify.
    let co_flags = CO_NEWLOCALS;

    // upstream `bytecode.py:46-60` stores `co_firstlineno` from the
    // source code object. `syn::Span::start().line` is 1-based within
    // the span's source input — `parse_file` seeds this as the file
    // line, `parse_str` as the offset within the string (usually 1
    // for a single-fn fixture). The `proc-macro2/span-locations`
    // feature (pulled in via this crate's `Cargo.toml`) is what
    // exposes `start()` outside of a proc-macro runtime.
    let co_firstlineno = item_fn.sig.fn_token.span().start().line as u32;

    // PRE-EXISTING-ADAPTATION: upstream `model.py:54 FunctionGraph.filename`
    // surfaces `func.__code__.co_filename` (a real filesystem path).
    // `syn::Span::source_file()` is nightly-only in `proc_macro2`, so
    // stable Rust cannot recover the path the ItemFn parsed from.
    // Caller threading through `source_filename` is the parity-
    // preserving channel; when the caller has no filename (typical
    // `syn::parse_str` fixtures, or ingestion paths that haven't been
    // taught to thread the path yet), fall back to the `<rust-source>`
    // sentinel. `tool/error.rs:304` renders this sentinel gracefully
    // on the graph-error path.
    //
    // *Convergence path*: when `proc_macro2`'s `span-locations`
    // feature exposes source-file accessors on stable Rust (or we
    // wrap `parse_file` in a helper that preserves the path itself),
    // drop the sentinel and derive from `Span` directly.
    let co_filename = source_filename
        .map(str::to_owned)
        .unwrap_or_else(|| "<rust-source>".to_string());
    let host_code = HostCode::new(
        argnames.len() as u32,
        nlocals,
        0,
        co_flags,
        rustpython_compiler_core::bytecode::CodeUnits::from(Vec::new()),
        Vec::new(),
        Vec::new(),
        co_varnames,
        co_filename,
        name.clone(),
        co_firstlineno,
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new().into_boxed_slice(),
    );
    // upstream `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
    // wraps the function's owning module dict — every entry the
    // module bound at import time is visible. Mirror that by
    // snapshotting `module_id`'s slice of the registry. When the
    // walker registered no entries (anonymous fixture / metadata-
    // only entry), the snapshot is an empty dict, matching upstream
    // `func.__globals__ == {}` for a function defined with no
    // enclosing module bindings.
    let func_globals = Constant::new(ConstValue::Dict(module_globals_snapshot(module_id)));
    let mut gf = GraphFunc::from_host_code(host_code.clone(), func_globals, Vec::new());
    // Issue 2.1 (2026-05-05): record `module_id` on `GraphFunc` so
    // `func.__globals__` introspection paths
    // (`HostObject::class_get("__globals__")` /
    // `getattr(func, "__globals__")`) re-snapshot the registry at
    // access time — mirrors upstream `flowcontext.py:284
    // self.w_globals = Constant(func.__globals__)` where
    // `func.__globals__` is a *live* reference to the module
    // dict. The static snapshot stored in `gf.globals` above
    // remains as the canonical `module = __name__`-extraction
    // source per `from_host_code` and as the static fallback
    // for non-rust-source callers.
    gf.module_globals_id = Some(module_id);
    // upstream `bytecode.py:46-60` populates `GraphFunc.source` from
    // `inspect.getsource(func)`. When the caller threads in the
    // source text, mirror that — downstream readers (`model.rs:3210
    // FunctionGraph::source`, `tool/error.rs:300-320`) walk
    // `func.source` as a fallback when `graph._source` is unset, so
    // one assignment covers both paths.
    if let Some(src) = source_text {
        gf.source = Some(src.to_owned());
    }
    let host = HostObject::new_user_function(gf.clone());
    Ok(HostMetadataParts {
        host,
        host_code,
        gf,
    })
}

/// Walk the function body and return the ordered unique set of
/// `let`-bound / `for`-pattern identifiers that the adapter's builder
/// introduces as extra locals beyond the formal arguments.
///
/// Mirrors what the Python compiler would emit into `co_varnames`
/// after the formal-arg prefix: one entry per distinct local name
/// assigned anywhere inside the function (`compile.c:compiler_nameop`
/// on the CPython side; `pygraph.py:14-16` reads the resulting
/// `co_nlocals` back when seeding the initial `FrameState`).
///
/// The adapter's `BlockBuilder::locals` also carries synthetic slots
/// named `#for_iter_{depth}` (`build_flow.rs:1266`) — those are *not*
/// upstream `co_varnames` entries (Python would have kept the
/// iterator on the value stack) so they are filtered out by rejecting
/// names starting with `#`.
///
/// Formals are excluded via `argnames_in_order` so the caller can
/// simply append `extras` after `argnames` without deduping again.
fn collect_local_names(item_fn: &ItemFn, argnames_in_order: &[String]) -> Vec<String> {
    struct LocalCollector<'a> {
        argnames: &'a [String],
        seen: std::collections::HashSet<String>,
        order: Vec<String>,
    }

    impl<'a> LocalCollector<'a> {
        fn record(&mut self, pat: &Pat) {
            let ident = match pat {
                Pat::Ident(PatIdent {
                    ident,
                    by_ref: None,
                    subpat: None,
                    ..
                }) => ident.to_string(),
                Pat::Type(pat_type) => {
                    if let Pat::Ident(PatIdent {
                        ident,
                        by_ref: None,
                        subpat: None,
                        ..
                    }) = &*pat_type.pat
                    {
                        ident.to_string()
                    } else {
                        return;
                    }
                }
                _ => return,
            };
            if ident.starts_with('#') || self.argnames.iter().any(|a| a == &ident) {
                return;
            }
            if self.seen.insert(ident.clone()) {
                self.order.push(ident);
            }
        }
    }

    impl<'ast, 'a> Visit<'ast> for LocalCollector<'a> {
        fn visit_local(&mut self, node: &'ast Local) {
            self.record(&node.pat);
            visit::visit_local(self, node);
        }

        fn visit_expr_for_loop(&mut self, node: &'ast ExprForLoop) {
            self.record(&node.pat);
            visit::visit_expr_for_loop(self, node);
        }
    }

    let mut collector = LocalCollector {
        argnames: argnames_in_order,
        seen: std::collections::HashSet::new(),
        order: Vec::new(),
    };
    collector.visit_block(&item_fn.block);
    collector.order
}

/// Extract the formal-parameter identifiers from a `syn::ItemFn`,
/// mirroring `collect_params` in `build_flow.rs`. Duplicated rather
/// than shared because the two callers consume different outputs — the
/// adapter needs `Hlvalue`s for the startblock, while this helper needs
/// the plain `String` names for `HostCode::co_varnames`.
fn extract_argnames(item_fn: &ItemFn) -> Result<Vec<String>, AdapterError> {
    let mut out = Vec::new();
    for input in &item_fn.sig.inputs {
        let ident = match input {
            FnArg::Receiver(_) => "self".to_string(),
            FnArg::Typed(pat_type) => match &*pat_type.pat {
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
        out.push(ident);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> ItemFn {
        syn::parse_str::<ItemFn>(src).expect("test fixture must parse")
    }

    /// Test helper: lookup `name` in `module_id`'s slice of the
    /// module-globals registry and unwrap the expected
    /// `ConstValue::HostObject` shape. Per-module scoping (Issue
    /// 1.3) makes the lookup id-aware; tests pass the id returned
    /// by `register_rust_module`.
    fn lookup_host(module_id: ModuleId, name: &str) -> Option<HostObject> {
        match module_globals_lookup(module_id, name)? {
            ConstValue::HostObject(h) => Some(h),
            other => panic!("expected HostObject for {name}, got {other:?}"),
        }
    }

    #[test]
    fn zero_arg_function_produces_matching_signature() {
        let item = parse("fn zero() -> i64 { 1 }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        assert_eq!(host.qualname(), "zero");
        assert!(host.is_user_function());

        let sig = pygraph.signature.borrow();
        assert!(sig.argnames.is_empty());
        assert!(sig.varargname.is_none());
        assert!(sig.kwargname.is_none());

        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 0);
        assert_eq!(code.co_nlocals, 0);
        assert!(code.co_varnames.is_empty());
        // upstream `objspace.py:33-35` — any RPython function's code
        // object must carry `CO_NEWLOCALS`.
        assert_ne!(code.co_flags & CO_NEWLOCALS, 0);
    }

    #[test]
    fn let_bindings_extend_co_varnames_and_co_nlocals() {
        // upstream `pygraph.py:14-16` — `co_nlocals` must size the
        // full locals array (formals + extras); `co_varnames` names
        // each slot in order.
        let item = parse("fn f(a: i64, b: i64) -> i64 { let x = a + b; let y = x + 1; y }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 2);
        assert_eq!(code.co_nlocals, 4);
        assert_eq!(
            code.co_varnames,
            vec![
                "a".to_string(),
                "b".to_string(),
                "x".to_string(),
                "y".to_string(),
            ],
        );
    }

    #[test]
    fn duplicate_let_names_appear_once() {
        // Shadowing `let x` twice still records one slot; upstream
        // Python compilers likewise collapse repeated assignments to
        // the same name into one `co_varnames` entry.
        let item = parse("fn f(a: i64) -> i64 { let x = a; let x = x + 1; x }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_nlocals, 2);
        assert_eq!(code.co_varnames, vec!["a".to_string(), "x".to_string()],);
    }

    #[test]
    fn for_pattern_identifier_is_recorded_as_local() {
        // upstream Python `for item in iter:` introduces `item` as a
        // fast local. Mirror that so the `co_varnames` collector
        // picks the loop variable up even when the adapter itself
        // can't yet lower assignments (`Expr::Assign` is
        // M2.5b-subset-rejected at `build_flow.rs:2145`), so we call
        // the helper directly instead of routing through
        // `build_host_function_from_rust`.
        //
        // The `#for_iter_N` synthetic slot from `build_flow.rs:1266`
        // stays out of `co_varnames` because `#` is not a valid
        // Python identifier character — the collector filters on
        // that prefix.
        let item = parse("fn f(xs: i64) -> i64 { for x in xs { let y = x; } xs }");
        let argnames = extract_argnames(&item).expect("formal args");
        let extras = collect_local_names(&item, &argnames);
        assert!(extras.contains(&"x".to_string()));
        assert!(extras.contains(&"y".to_string()));
        assert!(
            !extras.iter().any(|n| n.starts_with('#')),
            "synthetic iter slot leaked: {:?}",
            extras,
        );
    }

    #[test]
    fn two_arg_function_preserves_order_and_identity() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        let sig = pygraph.signature.borrow();
        assert_eq!(sig.argnames, vec!["a".to_string(), "b".to_string()]);

        // FunctionGraph.func points at the same GraphFunc the
        // HostObject wraps — parity with upstream PyGraph.__init__.
        let graph_func_id = pygraph
            .graph
            .borrow()
            .func
            .as_ref()
            .expect("graph.func set")
            .id;
        let host_func_id = host.user_function().expect("user function").id;
        assert_eq!(graph_func_id, host_func_id);
    }

    #[test]
    fn startblock_inputargs_match_argnames() {
        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (_host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let inputargs = pygraph.graph.borrow().startblock.borrow().inputargs.clone();
        assert_eq!(inputargs.len(), 2);
        // Adapter builds startblock with named Variables — the names
        // come from the Rust parameter identifiers via `collect_params`.
        // `Variable::rename` (model.rs:2050) always trails the prefix
        // with `_` for valid-Python-identifier parity.
        for (expected, arg) in ["a_", "b_"].iter().zip(inputargs.iter()) {
            match arg {
                crate::flowspace::model::Hlvalue::Variable(v) => {
                    assert_eq!(v.name_prefix(), *expected);
                }
                other => panic!("expected Variable, got {other:?}"),
            }
        }
    }

    #[test]
    fn co_firstlineno_reflects_fn_span() {
        // `span-locations` (Cargo.toml) gives `Span::start().line`
        // a non-zero 1-based reading. A leading newline pushes the
        // `fn` token to line 2; assert that the synthetic HostCode
        // picks that up rather than keeping the prior `0` placeholder.
        let item = parse("\n    fn shifted() -> i64 { 1 }");
        let (host, _pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_firstlineno, 2);
    }

    #[test]
    fn rejects_tuple_pattern_parameter() {
        // Matches `collect_params` in `build_flow.rs` — only plain
        // identifier patterns are accepted.
        let item = parse("fn f((a, b): (i64, i64)) -> i64 { a + b }");
        let err = build_host_function_from_rust(&item, None, None).unwrap_err();
        match err {
            AdapterError::InvalidSignature { reason } => {
                assert!(reason.contains("plain identifier"), "reason: {reason}");
            }
            other => panic!("expected InvalidSignature, got {other:?}"),
        }
    }

    #[test]
    fn seeds_into_translator_prebuilt_graphs_roundtrip() {
        use crate::translator::translator::TranslationContext;

        let item = parse("fn add(a: i64, b: i64) -> i64 { a + b }");
        let (host, pygraph) = build_host_function_from_rust(&item, None, None).expect("adapter");

        let ctx = TranslationContext::new();
        ctx._prebuilt_graphs
            .borrow_mut()
            .insert(host.clone(), pygraph.clone());

        // `buildflowgraph` must return the prebuilt graph unchanged
        // and leave no residual entry in the cache (upstream
        // `translator.py:50-51` pops).
        let retrieved = ctx.buildflowgraph(host.clone(), false).expect("prebuilt");
        assert!(Rc::ptr_eq(&retrieved, &pygraph));
        assert!(!ctx._prebuilt_graphs.borrow().contains_key(&host));
    }

    // ---- Slice O7 — module-globals walker (RPython parity for
    //      `flowcontext.py:847 w_globals.value[varname]` /
    //      `interactive.py:25-26 buildflowgraph` import-time shape).

    #[test]
    fn metadata_only_helper_does_not_lower_body() {
        // upstream Python module import: `def f(...): <body>` creates
        // a function object with `__code__` set; the flowspace graph
        // is NOT built at import time. The metadata-only helper must
        // mirror that — no `build_flow_from_rust` call, no graph in
        // hand, no PyGraph wrapped.
        //
        // A body using a construct the body lowerer rejects (a
        // leading-`::` globally-anchored path — see
        // `build_flow.rs::resolve_path_constant` Unsupported branch)
        // demonstrates this directly: the body lowerer would surface
        // `AdapterError::Unsupported` if invoked, but the metadata
        // path bypasses the body and succeeds.
        let item = parse("fn helper(x: i64) -> i64 { ::std::result::Result::Ok(x); x }");
        // First confirm the body lowerer rejects this fixture so the
        // bypass is actually load-bearing — if leading-`::` paths
        // ever land in `build_flow_from_rust`'s subset, this
        // assertion will fail loudly and the test author can refresh
        // the body to a still-rejected construct.
        assert!(
            super::super::build_flow::build_flow_from_rust(&item).is_err(),
            "fixture body must be rejected by build_flow_from_rust so the \
             metadata-only path is the load-bearing reason this test passes",
        );
        let host = build_host_function_metadata_from_rust(&item, None, None)
            .expect("metadata path skips body lowering");
        assert_eq!(host.qualname(), "helper");
        assert!(host.is_user_function());
        let gf = host.user_function().expect("user function");
        let code = gf.code.as_ref().expect("synthetic HostCode");
        assert_eq!(code.co_argcount, 1);
        assert_eq!(code.co_varnames, vec!["x".to_string()]);
        // upstream `objspace.py:33-35` invariant — code object must
        // carry CO_NEWLOCALS even on the metadata-only path so any
        // later `_assert_rpythonic` re-verify succeeds.
        assert_ne!(code.co_flags & CO_NEWLOCALS, 0);
    }

    #[test]
    fn register_rust_module_does_not_register_item_fn() {
        // PRE-EXISTING-ADAPTATION (Issue 1.2 fix): top-level
        // `Item::Fn` is INTENTIONALLY NOT registered into the
        // module-globals registry. Upstream Python `def` would bind
        // a function object whose `func.__code__.co_code` carries the
        // body — `FunctionDesc.buildgraph` (`description.py:140`)
        // calls `build_flow(GraphFunc)` to lower it on first call.
        //
        // pyre's `HostCode` for an `Item::Fn` is built with empty
        // bytecode (`build_host_metadata_parts` →
        // `CodeUnits::from(Vec::new())`) because the Rust-AST adapter
        // is the only path that lowers Rust source. There is no
        // wire-back from `FunctionDesc.buildgraph` to
        // `build_flow_from_rust`, so a registered sibling-fn
        // `HostObject` would masquerade as callable but supply
        // empty bytecode at lowering time. Until the walker can
        // either eagerly build the prebuilt graph (Slice M2.5f) or
        // store the AST in a side table for later replay (M2.5g),
        // we leave sibling fns unresolved — same shape pre-O9 main
        // exhibited.
        //
        // The single entry-point fn that production callers want is
        // located directly via `file.items.iter().find_map(...)` in
        // `build_host_function_from_rust_file`, bypassing the
        // registry entirely. So this opt-out is invisible to the
        // production path.

        let src = "fn parity_probe_walker_alpha() -> i64 { 1 }
                   fn parity_probe_walker_beta(a: i64) -> i64 { a }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");

        assert!(
            module_globals_lookup(module_id, "parity_probe_walker_alpha").is_none(),
            "Item::Fn must NOT be registered (sibling-fn body-rebuild path missing)",
        );
        assert!(
            module_globals_lookup(module_id, "parity_probe_walker_beta").is_none(),
            "Item::Fn must NOT be registered (sibling-fn body-rebuild path missing)",
        );
    }

    #[test]
    fn register_rust_module_skip_extends_to_unsupported_bodies() {
        // Same Issue 1.2 invariant: even if a fn body is something
        // the lowerer would reject (`as T` cast — task #94), the
        // walker still does not register it. The skip is uniform
        // across `Item::Fn` regardless of body shape.

        let src = "fn parity_probe_walker_with_cast(x: u32) -> i64 { x as i64 }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert!(
            module_globals_lookup(module_id, "parity_probe_walker_with_cast").is_none(),
            "Item::Fn skip is unconditional regardless of body lowerability",
        );
    }

    #[test]
    fn register_rust_module_skips_non_walked_item_kinds() {
        // Walker dispatches `Item::Enum`, `Item::Struct`, `Item::Const`
        // (Slice O10), and `Item::Static` (Slice O12). Remaining kinds
        // (`Item::Use`, `Item::Mod`, `Item::Impl`, `Item::Fn`) are
        // follow-up slices — they must NOT pollute the module-globals
        // registry until their dispatch is added.
        use super::super::host_env::module_globals_lookup;

        // `use` re-export — upstream `from x import y` would bind
        // `y` in `module.__dict__`; pyre walker doesn't yet resolve
        // cross-module references, so the binding is silently
        // skipped.
        let src = "use std::collections::HashMap as PARITY_PROBE_WALKER_USE_ONLY;";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert!(module_globals_lookup(module_id, "PARITY_PROBE_WALKER_USE_ONLY").is_none());
    }

    #[test]
    fn register_rust_module_walks_item_static_like_item_const() {
        // Slice O12: immutable `static FOO: T = <expr>;` populates
        // `module.__dict__` identically to `const FOO`.
        use super::super::host_env::module_globals_lookup;

        let src = "static ParityProbe_O12_Static: i64 = 5;
                   static ParityProbe_O12_StaticStr: &str = \"hello\";
                   static ParityProbe_O12_StaticBool: bool = true;";
        let file = syn::parse_file(src).expect("static fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Static"),
            Some(ConstValue::Int(5)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_StaticStr"),
            Some(ConstValue::uni_str("hello".to_string())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_StaticBool"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_skips_mutable_static() {
        // Slice O15 / codex parity audit (2026-05-06): `static mut`
        // is NOT registered. Runtime mutation makes the initial-
        // value snapshot unsound for constfold reads. PRE-EXISTING-
        // ADAPTATION until a live-store path lands. Downstream
        // lookups for the name fall through to the mint-or-fail
        // resolver — the registry intentionally has no entry.
        use super::super::host_env::module_globals_lookup;

        let src = "static mut ParityProbe_O15_StaticMut: i64 = 7;
                   static ParityProbe_O15_StaticImmut: i64 = 9;";
        let file = syn::parse_file(src).expect("static-mut fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert!(
            module_globals_lookup(module_id, "ParityProbe_O15_StaticMut").is_none(),
            "`static mut` must NOT be registered (runtime mutation makes snapshot unsound)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O15_StaticImmut"),
            Some(ConstValue::Int(9)),
            "immutable `static` next to a `static mut` still walks normally",
        );
    }

    #[test]
    fn register_rust_module_static_resolves_compound_rhs_against_prior_bindings() {
        // Static RHS goes through the same `eval_const_expr` as
        // `Item::Const`, so forward-ref to a sibling `const` (or
        // sibling `static`) works the same way: source-order
        // `bindings` accumulator is populated before the next item
        // is evaluated. Mirrors upstream Python's import-time linear
        // execution: `X = 1; Y = X + 1` binds both, in order.
        use super::super::host_env::module_globals_lookup;

        let src = "const ParityProbe_O12_X: i64 = 10;
                   static ParityProbe_O12_Y: i64 = ParityProbe_O12_X + 5;
                   static ParityProbe_O12_Z: i64 = ParityProbe_O12_Y * 2;";
        let file = syn::parse_file(src).expect("compound static fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_X"),
            Some(ConstValue::Int(10)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Y"),
            Some(ConstValue::Int(15)),
            "`Y` resolves prior `const X` through the source-order accumulator",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O12_Z"),
            Some(ConstValue::Int(30)),
            "`Z` resolves prior `static Y` (Slice O12 admits Item::Static into the accumulator)",
        );
    }

    #[test]
    fn register_rust_module_walks_item_enum_with_variants_as_children() {
        // upstream Python analogue: `class StepResult: pass; class
        // StepResult_Continue(StepResult): pass; StepResult.Continue
        // = StepResult_Continue`. The walker's enum dispatch produces
        // the same shape — parent class with each variant bound in
        // the class dict to a child class whose bases include the
        // parent.

        let src = "pub enum ParityProbeEnum_Slice_O8 { Alpha, Beta, Gamma }";
        let file = syn::parse_file(src).expect("enum fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");

        let parent =
            lookup_host(module_id, "ParityProbeEnum_Slice_O8").expect("enum registered after walk");
        assert!(parent.is_class());
        assert_eq!(parent.qualname(), "ParityProbeEnum_Slice_O8");

        for variant in ["Alpha", "Beta", "Gamma"] {
            let entry = parent
                .class_get(variant)
                .unwrap_or_else(|| panic!("variant {variant} bound in parent class dict"));
            let child = match entry {
                ConstValue::HostObject(h) => h,
                other => panic!("variant carrier must be HostObject, got {other:?}"),
            };
            assert!(child.is_class(), "variant {variant} must be a class");
            assert!(
                child.is_subclass_of(&parent),
                "variant {variant} must be a subclass of the parent enum class \
                 (matches upstream `class V(Parent): pass` shape)",
            );
            assert_eq!(
                child.qualname(),
                format!("ParityProbeEnum_Slice_O8.{variant}")
            );
        }
    }

    #[test]
    fn register_rust_module_walks_item_struct_with_empty_class_dict() {
        // Rust struct field access `instance.x` reads from the
        // instance, not the class object — upstream `class Foo: pass`
        // likewise leaves `Foo.__dict__` empty for instance
        // attributes. The walker's struct dispatch produces the
        // identity-carrier class with an empty class dict.

        let src = "pub struct ParityProbeStruct_Slice_O8 { x: i64, y: i64 }";
        let file = syn::parse_file(src).expect("struct fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let host = lookup_host(module_id, "ParityProbeStruct_Slice_O8")
            .expect("struct registered after walk");
        assert!(host.is_class());
        assert_eq!(host.qualname(), "ParityProbeStruct_Slice_O8");
        // No instance fields populate the class dict — they live on
        // instances. `class_dict_keys()` returns the empty set.
        assert!(
            host.class_dict_keys().is_empty(),
            "struct class dict must be empty (instance fields belong on instances), \
             got keys: {:?}",
            host.class_dict_keys(),
        );
    }

    #[test]
    fn graph_func_globals_reflects_module_globals_partition_after_walk() {
        // Issue 1 (2026-05-05): `GraphFunc.globals` must surface
        // the module-globals registry slice for the active
        // `ModuleId`, not an empty dict. Mirrors upstream
        // `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
        // — `func.__globals__` is the function's owning module's
        // `__dict__`, whose entries the walker has just bound.
        //
        // Path-keyed walk (Issue 2 share) so the body lowering uses
        // the same id the walker registered under. The fixture must
        // be lowerable by the current adapter subset — `fn entry()
        // -> i64 { 1 }` is enough; what we're verifying is the
        // GraphFunc-side carrier, not body resolution.
        let src = "pub struct ParityProbe_Issue1_sibling;
                   pub const ParityProbe_Issue1_const: i64 = 7;
                   fn parity_probe_issue1_entry() -> i64 { 1 }";
        let file = syn::parse_file(src).expect("file fixture parses");
        let (host, _pygraph) = build_host_function_from_rust_file(
            &file,
            "parity_probe_issue1_entry",
            Some("/parity_probe/issue1_globals.rs"),
            None,
        )
        .expect("file-aware entry succeeds");
        let gf = host.user_function().expect("user function");
        // GraphFunc.globals is a Constant; the inner ConstValue must
        // be Dict carrying the registered struct + const.
        let dict = match &gf.globals.value {
            ConstValue::Dict(items) => items,
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };
        let struct_key = ConstValue::byte_str(b"ParityProbe_Issue1_sibling");
        let const_key = ConstValue::byte_str(b"ParityProbe_Issue1_const");
        assert!(
            dict.contains_key(&struct_key),
            "module-globals dict must contain registered struct, got keys: {:?}",
            dict.keys().collect::<Vec<_>>(),
        );
        assert_eq!(
            dict.get(&const_key),
            Some(&ConstValue::Int(7)),
            "module-globals dict must contain registered Item::Const value",
        );
    }

    #[test]
    fn graph_func_live_globals_reflects_post_construction_walker_writes() {
        // Issue 2.1 (2026-05-05): `func.__globals__` must surface
        // entries added to the registry partition AFTER `GraphFunc`
        // construction, mirroring upstream `flowcontext.py:284
        // self.w_globals = Constant(func.__globals__)` where
        // `func.__globals__` is a *live* reference to the module
        // dict. `live_globals()` re-snapshots the partition keyed
        // on `module_globals_id`.
        //
        // Construction sequence: build entry-point first (snapshot
        // taken with only items the walker has registered so far),
        // then register additional items into the same `module_id`.
        // The static `gf.globals.value` snapshot stays stale; the
        // `live_globals()` accessor returns the post-write state.
        use super::super::host_env::register_module_global;
        let path = "/parity_probe/issue21_live_globals.rs";
        let src = "fn parity_probe_issue21_entry() -> i64 { 1 }";
        let file = syn::parse_file(src).expect("entry-point fixture parses");
        let (host, _pygraph) = build_host_function_from_rust_file(
            &file,
            "parity_probe_issue21_entry",
            Some(path),
            None,
        )
        .expect("file-aware entry succeeds");
        let gf = host.user_function().expect("user function");
        let module_id = gf.module_globals_id.expect("rust-source built sets the id");

        // Static snapshot is the dict at construction time — empty
        // beyond what the walker pre-pass wrote. Sanity-check the
        // construction-time invariant first.
        let static_dict = match &gf.globals.value {
            ConstValue::Dict(items) => items.clone(),
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };

        // Now register a fresh entry into the same partition,
        // simulating either a follow-up walker pass or a deferred
        // `Item::*` registration. Upstream Python: a later
        // `module.NEW_NAME = ...` assignment would be visible to
        // any subsequent `func.__globals__["NEW_NAME"]` read.
        register_module_global(
            module_id,
            "ParityProbe_Issue21_late_addition",
            ConstValue::Int(99),
        );

        // Static snapshot is unchanged (snapshot semantic).
        let static_dict_after = match &gf.globals.value {
            ConstValue::Dict(items) => items.clone(),
            other => panic!("expected ConstValue::Dict, got {other:?}"),
        };
        assert_eq!(
            static_dict, static_dict_after,
            "static `gf.globals.value` snapshot stays frozen at construction time",
        );

        // Live snapshot via `live_globals()` reflects the new entry.
        let live = gf.live_globals();
        let live_dict = match &live {
            ConstValue::Dict(items) => items,
            other => panic!("expected live_globals() Dict, got {other:?}"),
        };
        assert_eq!(
            live_dict.get(&ConstValue::byte_str(b"ParityProbe_Issue21_late_addition")),
            Some(&ConstValue::Int(99)),
            "live_globals must surface entries added to the partition \
             post-construction (upstream `func.__globals__` is a live ref)",
        );
    }

    #[test]
    fn register_rust_module_at_with_same_path_returns_same_id() {
        // Issue 2 (2026-05-05): two walks of the same source path
        // converge on the same `ModuleId` — mirrors upstream
        // `sys.modules[path]` import-cache identity. The second
        // walk's registrations clobber the first under
        // last-writer-wins (`module.__dict__[name] = value`
        // semantics) — the cross-id lookup observes the second
        // walk's binding regardless of whether file1 / file2 are
        // identical or distinct.

        let src = "pub struct ParityProbe_Issue2_path_share;";
        let file1 = syn::parse_file(src).expect("file 1 parses");
        let file2 = syn::parse_file(src).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_share.rs"))
            .expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_share.rs"))
            .expect("walker must succeed");
        assert_eq!(
            id1, id2,
            "same path must yield the same ModuleId (sys.modules cache parity)",
        );
        let host = lookup_host(id1, "ParityProbe_Issue2_path_share")
            .expect("walk's binding visible from shared id");
        let cross = lookup_host(id2, "ParityProbe_Issue2_path_share").unwrap();
        assert_eq!(
            host, cross,
            "shared id must serve identical HostObject identity across walks \
             (last-writer-wins: the second walk's HostObject is what both ids see)",
        );
    }

    #[test]
    fn register_rust_module_at_with_distinct_paths_isolates_partitions() {
        // Issue 2 (2026-05-05): different paths mint distinct ids,
        // matching upstream's per-module `__dict__` isolation. Two
        // modules at distinct paths binding the same top-level name
        // see independent values — the cross-id lookup misses.

        let src1 = "pub struct ParityProbe_Issue2_path_distinct;";
        let src2 = "pub struct ParityProbe_Issue2_path_distinct { x: i64 }";
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_distinct_a.rs"))
            .expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_distinct_b.rs"))
            .expect("walker must succeed");
        assert_ne!(id1, id2, "distinct paths must mint distinct ids");
        let host1 = lookup_host(id1, "ParityProbe_Issue2_path_distinct").unwrap();
        let host2 = lookup_host(id2, "ParityProbe_Issue2_path_distinct").unwrap();
        assert_ne!(
            host1, host2,
            "distinct-path walks must produce independent class identities",
        );
    }

    #[test]
    fn register_rust_module_at_with_none_path_mints_fresh_per_call() {
        // Codex parity revert (2026-05-05): the `None` branch was
        // previously content-hashed (Issue 2.2), but that merged
        // distinct anonymous walks of identical source whenever the
        // bytes matched — diverging from upstream Python's
        // `exec(source, dict_a)` / `exec(source, dict_b)` semantic
        // (each `exec` runs against an independent `__dict__`). The
        // fix mints a fresh ModuleId per anonymous call. Callers
        // who need shared identity opt in via `Some(path)`.
        let src = "pub struct ParityProbe_Anonymous_FreshPerCall;";
        let file = syn::parse_file(src).expect("anonymous walk parses");
        let id1 = register_rust_module_at(&file, None).expect("walker must succeed");
        let id2 = register_rust_module_at(&file, None).expect("walker must succeed");
        assert_ne!(
            id1, id2,
            "anonymous walks (None path) MUST mint independent ModuleIds even \
             when the source bytes are identical — content equality does not \
             imply module identity in upstream Python",
        );
    }

    #[test]
    fn register_rust_module_at_with_none_path_distinguishes_distinct_content() {
        // Anonymous walks of distinct content produce distinct ids
        // (trivially follows from "every None call mints fresh").
        // Kept as a separate oracle so future regressions to
        // content-hash sharing fail this test too.
        let src1 = "pub struct ParityProbe_Anon_Distinct_A;";
        let src2 = "pub struct ParityProbe_Anon_Distinct_B;";
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, None).expect("walker must succeed");
        let id2 = register_rust_module_at(&file2, None).expect("walker must succeed");
        assert_ne!(id1, id2, "distinct anonymous walks mint distinct ModuleIds",);
    }

    #[test]
    fn register_rust_module_isolates_distinct_walks_with_shared_top_level_name() {
        // Per-module scoping (Issue 1.3, 2026-05-05): two walks of
        // files containing the same top-level name now produce
        // INDEPENDENT registry partitions, not a shared first-writer
        // entry. Mirrors upstream `flowcontext.py:284 self.w_globals
        // = Constant(func.__globals__)` per-module scoping — two
        // distinct modules with identically-named top-level
        // bindings see independent values.
        //
        // (The pre-Issue-1.3 cross-walk first-writer-wins test was
        // a workaround for the missing per-module scoping; it no
        // longer applies once the registry partitions properly.)

        let src1 = "pub struct ParityProbeStruct_isolate;";
        let src2 = "pub struct ParityProbeStruct_isolate { x: i64 }"; // distinct shape, same name
        let file1 = syn::parse_file(src1).expect("file 1 parses");
        let file2 = syn::parse_file(src2).expect("file 2 parses");
        let id1 = register_rust_module(&file1).expect("walker must succeed");
        let id2 = register_rust_module(&file2).expect("walker must succeed");
        assert_ne!(id1, id2, "fresh walks must produce distinct ModuleIds");
        let host1 = lookup_host(id1, "ParityProbeStruct_isolate")
            .expect("file 1's binding visible from id1");
        let host2 = lookup_host(id2, "ParityProbeStruct_isolate")
            .expect("file 2's binding visible from id2");
        assert_ne!(
            host1, host2,
            "isolated walks must NOT share class identity \
             (each file mints its own HostObject under its own ModuleId)",
        );
        // Cross-id lookup remains scoped to its own partition: id1
        // does not see id2's struct shape, even though the names
        // are identical.
        let cross = lookup_host(id2, "ParityProbeStruct_isolate").unwrap();
        assert_eq!(cross, host2, "id2's lookup returns id2's binding");
        assert_ne!(cross, host1, "id2's lookup must NOT see id1's binding");
    }

    #[test]
    fn register_rust_module_last_writer_wins_under_same_module_id() {
        // Last-writer-wins under a fixed `ModuleId` mirrors upstream
        // `module.__dict__[name] = value` semantics: every top-level
        // binding statement is an unconditional assignment. Within a
        // single `register_rust_module` call Rust syntax does not
        // allow duplicate top-level item names, so the observable
        // effect is across walks of the same path-keyed module — the
        // second walk's bindings supersede the first.
        //
        // We can't feed two `pub struct Foo;` to syn::parse_file (it
        // errors on duplicate items), so we exercise
        // `register_module_global` directly under a fixed id to
        // model the cross-walk re-binding.
        let id = ModuleId::fresh();
        let first = HostObject::new_class("ParityProbeStruct_within_walk", vec![]);
        let second = HostObject::new_class("ParityProbeStruct_within_walk", vec![]);
        super::super::host_env::register_module_global(
            id,
            "ParityProbeStruct_within_walk",
            ConstValue::HostObject(first.clone()),
        );
        super::super::host_env::register_module_global(
            id,
            "ParityProbeStruct_within_walk",
            ConstValue::HostObject(second.clone()),
        );
        let observed = lookup_host(id, "ParityProbeStruct_within_walk").unwrap();
        assert_eq!(observed, second, "last registration wins within same id");
        assert_ne!(
            observed, first,
            "first registration must be clobbered by the second under same id",
        );
    }

    // ---- Slice O10 — Item::Const walker dispatch -----------------

    #[test]
    fn register_rust_module_walks_item_const_integer_literal() {
        // upstream Python `MODULE.MAX_SIZE` reads
        // `module.__dict__["MAX_SIZE"]` which holds the int the
        // top-level assignment bound. The walker mirrors this for
        // `const MAX_SIZE: i64 = 42` — registers the integer value
        // directly as `ConstValue::Int(42)`.
        let src = "pub const ParityProbe_O10_const_int: i64 = 42;";
        let file = syn::parse_file(src).expect("const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_int")
            .expect("integer const registered after walk");
        assert_eq!(value, ConstValue::Int(42));
    }

    #[test]
    fn register_rust_module_walks_item_const_negative_integer_literal() {
        // `const X: i64 = -7` parses through `Expr::Unary { op:
        // Neg, expr: Lit(7) }` — the walker must unwrap one level
        // of unary minus to recognise the signed-int form.
        let src = "pub const ParityProbe_O10_const_neg_int: i64 = -7;";
        let file = syn::parse_file(src).expect("negated const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_neg_int")
            .expect("negated integer const registered after walk");
        assert_eq!(value, ConstValue::Int(-7));
    }

    #[test]
    fn register_rust_module_walks_item_const_bool_literal() {
        let src = "pub const ParityProbe_O10_const_bool: bool = true;";
        let file = syn::parse_file(src).expect("const bool parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_bool")
            .expect("bool const registered after walk");
        assert_eq!(value, ConstValue::Bool(true));
    }

    #[test]
    fn register_rust_module_walks_item_const_str_literal() {
        // `Lit::Str` lowers to `ConstValue::UniStr` matching
        // `build_flow.rs::lower_literal::Lit::Str` and Python 3
        // unicode-string semantics. The same `"abc"` literal at
        // body position would lower identically — no shape drift
        // between expression and module-const positions.
        let src = "pub const ParityProbe_O10_const_str: &str = \"abc\";";
        let file = syn::parse_file(src).expect("const str parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_str")
            .expect("str const registered after walk");
        assert_eq!(value, ConstValue::uni_str("abc"));
    }

    #[test]
    fn register_rust_module_walks_item_const_compound_expression() {
        // Issue 4 (2026-05-05): the walker resolves compound const
        // RHS expressions through prior bindings in the same source-
        // order walk. Mirrors upstream Python module-import: by the
        // time `Y = X + 1` runs at top level, `X = 1` has already
        // bound `module.__dict__["X"]`, and the binary op evaluates
        // `module.__dict__["X"] + 1` against that.
        let src = "pub const ParityProbe_Issue4_const_X: i64 = 1;
                   pub const ParityProbe_Issue4_const_Y: i64 = ParityProbe_Issue4_const_X + 1;
                   pub const ParityProbe_Issue4_const_Z: i64 = ParityProbe_Issue4_const_Y * 3;
                   pub const ParityProbe_Issue4_const_NEG: i64 = -ParityProbe_Issue4_const_Z;";
        let file = syn::parse_file(src).expect("compound const fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_X"),
            Some(ConstValue::Int(1)),
            "X registers as literal Int(1)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_Y"),
            Some(ConstValue::Int(2)),
            "Y registers as X + 1 = 2 via prior-bindings env",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_Z"),
            Some(ConstValue::Int(6)),
            "Z registers as Y * 3 = 6 via prior-bindings env",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_NEG"),
            Some(ConstValue::Int(-6)),
            "NEG registers as -Z = -6 (unary neg over evaluated path)",
        );
    }

    #[test]
    fn register_rust_module_walks_compound_const_with_shift_and_bitwise() {
        // Issue 2.4 (2026-05-05): `eval_const_expr` covers shifts
        // and bitwise ops over Int operands. Mirrors upstream
        // Python import-time evaluation: `MASK = 1 << 4 | 0x3` would
        // bind `module.__dict__["MASK"]` to the evaluated integer.
        let src = "pub const ParityProbe_Issue24_BASE: i64 = 1;
                   pub const ParityProbe_Issue24_SHIFT: i64 = ParityProbe_Issue24_BASE << 4;
                   pub const ParityProbe_Issue24_RSHIFT: i64 = 64 >> 2;
                   pub const ParityProbe_Issue24_AND: i64 = 0x1F & 0x07;
                   pub const ParityProbe_Issue24_OR: i64 = 0x10 | 0x01;
                   pub const ParityProbe_Issue24_XOR: i64 = 0x0F ^ 0x05;
                   pub const ParityProbe_Issue24_NOT: i64 = !0;";
        let file = syn::parse_file(src).expect("shift+bitwise fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_SHIFT"),
            Some(ConstValue::Int(16)),
            "1 << 4 = 16",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_RSHIFT"),
            Some(ConstValue::Int(16)),
            "64 >> 2 = 16",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_AND"),
            Some(ConstValue::Int(0x07)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_OR"),
            Some(ConstValue::Int(0x11)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_XOR"),
            Some(ConstValue::Int(0x0A)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_NOT"),
            Some(ConstValue::Int(!0_i64)),
            "Rust `!` on Int is bitwise complement (Python `~`)",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_shift_parity() {
        // `<<` / `>>` over int operands must match upstream
        // `operator.lshift` / `operator.rshift` semantics so a
        // module-top `const X = 1 >> 64` and a body `1 >> 64`
        // produce the same value.
        //
        // - **Negative shift count** raises `ValueError: negative
        //   shift count` upstream — propagate as `Err` (walker
        //   aborts the file).
        // - **`<<` count >= 64** declines per the
        //   `can_overflow and type(result) is long` arm
        //   (`operation.py:140-142`): the bignum result doesn't
        //   fit in `i64`, walker leaves the const unbound.
        // - **`>>` count >= 64** saturates: `1 >> 64 == 0`,
        //   `-1 >> 64 == -1`. RShift is not in the `can_overflow`
        //   set so the long-decline arm doesn't apply.
        let neg_shl = "pub const X: i64 = 1 << -1;";
        let file = syn::parse_file(neg_shl).expect("negative shl fixture parses");
        let err =
            register_rust_module(&file).expect_err("negative shift count must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ValueError") && reason.contains("negative shift count"),
                "expected ValueError on negative shift, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
        let neg_shr = "pub const Y: i64 = 1 >> -1;";
        let file = syn::parse_file(neg_shr).expect("negative shr fixture parses");
        register_rust_module(&file).expect_err("negative shr count must abort the walker");
        // Large positive shr — saturating fold.
        let big_shr = "pub const ParityProbe_BigShr_pos: i64 = 1 >> 64;
                       pub const ParityProbe_BigShr_neg: i64 = -1 >> 64;
                       pub const ParityProbe_BigShr_100: i64 = 100 >> 128;";
        let file = syn::parse_file(big_shr).expect("big shr fixture parses");
        let module_id = register_rust_module(&file).expect("big positive shr saturates, no abort");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_pos"),
            Some(ConstValue::Int(0)),
            "1 >> 64 == 0 (saturating non-negative)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_neg"),
            Some(ConstValue::Int(-1)),
            "-1 >> 64 == -1 (sign-extending)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_BigShr_100"),
            Some(ConstValue::Int(0)),
        );
        // Large positive shl — unsupported bignum carrier at module top level.
        // Upstream Python `1 << 64` binds bignum `2**64`; pyre
        // adapter has no bignum carrier so surfaces as
        // `AdapterError::Unsupported`. The function-body constfold path
        // (`operation.py:140-142`) declines with Ok(None) instead,
        // but THAT path is at function evaluation time where the
        // runtime can produce the bignum — there is no analogous
        // path at module-import-time prepass.
        let big_shl = "pub const ParityProbe_BigShl: i64 = 1 << 64;";
        let file = syn::parse_file(big_shl).expect("big shl fixture parses");
        let err = register_rust_module(&file)
            .expect_err("1 << 64 requires unsupported bignum carrier at module top level");
        match err {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("bignum") && reason.contains("i64"),
                "expected bignum unsupported error on 1 << 64, got: {reason}",
            ),
            other => panic!("expected AdapterError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mod_minus_one() {
        // `x % -1 == 0` for every `x` upstream. The `int_py_mod`
        // closure short-circuits `y == -1` so that
        // `i64::MIN.checked_rem(-1)` (which Rust returns `None`
        // for, because the *quotient* `2^63` overflows even though
        // the remainder `0` itself fits) does not silently decline
        // a well-defined fold. Source `-9223372036854775808` lies
        // outside what `syn` can parse as a single `Neg(Lit)`
        // (the inner literal `9223372036854775808` is `i64::MAX +
        // 1`), so the operation.rs `int_py_mod`-level test pins
        // the `MIN` boundary directly. Here we pin the
        // representable-source cases.
        let src = "pub const ParityProbe_ModMinusOne_pos: i64 = 7 % -1;
                   pub const ParityProbe_ModMinusOne_neg: i64 = -7 % -1;
                   pub const ParityProbe_ModMinusOne_max: i64 = 9223372036854775807 % -1;";
        let file = syn::parse_file(src).expect("x % -1 fixture parses");
        let module_id = register_rust_module(&file).expect("x % -1 folds, no abort");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_pos"),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_neg"),
            Some(ConstValue::Int(0)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ModMinusOne_max"),
            Some(ConstValue::Int(0)),
            "i64::MAX % -1 == 0 (no overflow path)",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_uses_python_floor_div_and_mod() {
        // The const evaluator's `Div` / `Rem` arms must agree with
        // `flowspace::operation`'s `int_py_floor_div` / `int_py_mod`
        // over negative-divisor pairs so a `const Y: i64 = 3 / -2`
        // at module top level produces the same value the function-
        // body `3 / -2` would after constfold. C-style truncation
        // (`a.checked_div(b)`) gives `-1` for `3 / -2` and `1` for
        // `3 % -2`, which would diverge from the body result.
        // Python floor-div toward `-inf` gives `-2` and `-1`
        // respectively (line-by-line port of
        // `rpython/rtyper/rint.py:398 ll_int_py_div`,
        // `:496 ll_int_py_mod`).
        let src = "pub const ParityProbe_FloorDiv_pos_div_neg: i64 = 3 / -2;
                   pub const ParityProbe_FloorMod_pos_mod_neg: i64 = 3 % -2;
                   pub const ParityProbe_FloorDiv_neg_div_pos: i64 = -3 / 2;
                   pub const ParityProbe_FloorMod_neg_mod_pos: i64 = -3 % 2;
                   pub const ParityProbe_FloorDiv_neg_div_neg: i64 = -3 / -2;
                   pub const ParityProbe_FloorMod_neg_mod_neg: i64 = -3 % -2;";
        let file = syn::parse_file(src).expect("floor-div fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_pos_div_neg"),
            Some(ConstValue::Int(-2)),
            "3 // -2 == -2 (Python floor toward -inf, not C trunc -1)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_pos_mod_neg"),
            Some(ConstValue::Int(-1)),
            "3 % -2 == -1 (sign matches divisor)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_neg_div_pos"),
            Some(ConstValue::Int(-2)),
            "-3 // 2 == -2",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_neg_mod_pos"),
            Some(ConstValue::Int(1)),
            "-3 % 2 == 1",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorDiv_neg_div_neg"),
            Some(ConstValue::Int(1)),
            "-3 // -2 == 1",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_FloorMod_neg_mod_neg"),
            Some(ConstValue::Int(-1)),
            "-3 % -2 == -1",
        );
    }

    #[test]
    fn register_rust_module_walks_compound_const_with_comparison_and_bool() {
        // Issue 2.4 (2026-05-05): comparison ops over Int / Bool /
        // strings produce Bool; `&&`/`||` over Bool stay Bool.
        // Mirrors Python's `MAX_NEG = MAX < 0` / `BOTH = A && B`
        // import-time evaluation.
        let src = "pub const ParityProbe_Issue24_MAX: i64 = 100;
                   pub const ParityProbe_Issue24_IS_BIG: bool = ParityProbe_Issue24_MAX > 50;
                   pub const ParityProbe_Issue24_EQ: bool = ParityProbe_Issue24_MAX == 100;
                   pub const ParityProbe_Issue24_AND_BOOL: bool = true && false;
                   pub const ParityProbe_Issue24_OR_BOOL: bool = false || true;
                   pub const ParityProbe_Issue24_NOT_BOOL: bool = !false;";
        let file = syn::parse_file(src).expect("comparison fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_IS_BIG"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_EQ"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_AND_BOOL"),
            Some(ConstValue::Bool(false)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_OR_BOOL"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue24_NOT_BOOL"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_short_circuits_and_or_at_compile_time() {
        // Codex parity audit (2026-05-05): both Rust source and
        // upstream Python's `and`/`or` are short-circuit.
        // `false && BAD` evaluates to `false` without ever touching
        // `BAD`; `true || BAD` evaluates to `true`. Without
        // short-circuit, the RHS reference to an unbound name would
        // surface as `NameError` and abort the walker.
        let src = "pub const ParityProbe_ShortCircuit_AND: bool = false && ParityProbe_UnboundRhs;
                   pub const ParityProbe_ShortCircuit_OR: bool = true || ParityProbe_UnboundRhs;";
        let file = syn::parse_file(src).expect("short-circuit fixture parses");
        let module_id = register_rust_module(&file).expect(
            "short-circuit must skip RHS evaluation — unbound name in RHS \
             does not abort the walker when LHS already determines result",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_AND"),
            Some(ConstValue::Bool(false)),
            "`false && X` short-circuits to false without evaluating X",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_OR"),
            Some(ConstValue::Bool(true)),
            "`true || X` short-circuits to true without evaluating X",
        );
    }

    #[test]
    fn register_rust_module_evaluates_rhs_when_lhs_does_not_short_circuit() {
        // Negative-direction oracle for short-circuit: when LHS does
        // NOT determine the result, RHS must be evaluated. `true &&
        // X` and `false || X` both depend on X.
        let src = "pub const ParityProbe_ShortCircuit_RHS_AND: bool = true && false;
                   pub const ParityProbe_ShortCircuit_RHS_OR: bool = false || true;";
        let file = syn::parse_file(src).expect("non-short-circuit fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_RHS_AND"),
            Some(ConstValue::Bool(false)),
            "`true && false` reads the RHS",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_ShortCircuit_RHS_OR"),
            Some(ConstValue::Bool(true)),
            "`false || true` reads the RHS",
        );
    }

    #[test]
    fn register_rust_module_aborts_when_short_circuit_does_not_apply() {
        // Without LHS short-circuit, the RHS is evaluated; an
        // unresolved single-segment Path raises NameError per
        // upstream `flowcontext.py:853 find_global`'s
        // `FlowingError("global name '...' is not defined")`.
        // Walker aborts the file matching upstream Python's
        // module-execution-aborts-on-exception semantic.
        let src = "pub const ParityProbe_ShortCircuit_NoShort_AND: bool = true && ParityProbe_UnboundRhs;";
        let file = syn::parse_file(src).expect("non-short-circuit-failure fixture parses");
        let err = register_rust_module(&file)
            .expect_err("unbound RHS at module top must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_UnboundRhs' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_aborts_on_overflow_const() {
        // `i64::MAX + 1` at module top level. Upstream Python would
        // bind a bignum `2**63` without raising; pyre adapter has no
        // bignum carrier so surfaces as `AdapterError::Unsupported`.
        // Walker aborts the file. (NOT the function-body
        // `PureOperation.constfold` decline path at
        // `operation.py:140-142` — that path lives at function
        // evaluation time where the runtime can produce the bignum.)
        let src = "pub const ParityProbe_Issue24_OVERFLOW: i64 = 9223372036854775807 + 1;";
        let file = syn::parse_file(src).expect("overflow fixture parses");
        let err = register_rust_module(&file)
            .expect_err("i64 overflow at module top requires unsupported bignum carrier");
        match err {
            AdapterError::Unsupported { reason } => assert!(
                reason.contains("bignum") && reason.contains("i64"),
                "expected bignum unsupported error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_rewalks_path_keyed_module_resolves_via_registry() {
        // Issue 2.3 (2026-05-05): when a path-keyed `module_id` is
        // re-walked (e.g. two `Translation::from_rust_file_entry_point_with_source`
        // calls against the same source file), the second walk's
        // local `const_bindings` is fresh-empty. `eval_const_expr`
        // must fall back to `module_globals_lookup(module_id, ...)`
        // to surface the registered prior binding — mirrors upstream
        // `module.__dict__` being the live reference visible across
        // re-imports.
        //
        // Walk 1 binds X. Walk 2 has only Y = X + 1; Y must resolve.
        let path = "issue_2_3_const_rewalk_fixture.rs";
        let src1 = "pub const ParityProbe_Issue23_X: i64 = 7;";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(id1, "ParityProbe_Issue23_X"),
            Some(ConstValue::Int(7)),
        );
        // Walk 2 — same path → same module_id. New const Y references X.
        let src2 = "pub const ParityProbe_Issue23_Y: i64 = ParityProbe_Issue23_X + 1;";
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        assert_eq!(
            module_globals_lookup(id2, "ParityProbe_Issue23_Y"),
            Some(ConstValue::Int(8)),
            "Y = X + 1 resolves X via registry fallback (X bound in prior walk, \
             absent from this walk's local const_bindings)",
        );
    }

    #[test]
    fn register_rust_module_path_keyed_rewalk_is_last_writer_wins() {
        // Codex parity audit (2026-05-05): a path-keyed re-walk that
        // re-binds the SAME name MUST clobber the prior entry, not
        // skip it. Mirrors upstream `module.__dict__[name] = value`:
        // every top-level binding statement is an unconditional
        // assignment (`exec(source, dict)` / `importlib.reload`
        // semantics). The pre-fix walker silently dropped the second
        // walk's binding under first-writer-wins, diverging from
        // Python module-execution semantics.
        let path = "issue6_lastwriter_rewalk_fixture.rs";
        let src1 = "pub const ParityProbe_LastWriter_X: i64 = 1;";
        let src2 = "pub const ParityProbe_LastWriter_X: i64 = 2;";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        assert_eq!(
            module_globals_lookup(id1, "ParityProbe_LastWriter_X"),
            Some(ConstValue::Int(1)),
            "walk 1 binds X = 1",
        );
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        assert_eq!(
            module_globals_lookup(id2, "ParityProbe_LastWriter_X"),
            Some(ConstValue::Int(2)),
            "walk 2 last-writer-wins: X is now 2 (was 1 from walk 1)",
        );
    }

    #[test]
    fn register_rust_module_path_keyed_rewalk_overwrites_struct_class_identity() {
        // Same last-writer-wins applies to `Item::Struct` / `Item::Enum`
        // class registrations: a path-keyed re-walk re-mints the
        // `HostObject::Class` carrier and overwrites the prior entry,
        // mirroring upstream `module.__dict__[ClassName] = <new
        // class>` after re-executing the module body.
        let path = "issue6_lastwriter_class_rewalk_fixture.rs";
        let src1 = "pub struct ParityProbe_LastWriter_Cls;";
        let src2 = "pub struct ParityProbe_LastWriter_Cls { x: i64 }";
        let file1 = syn::parse_file(src1).expect("walk-1 fixture parses");
        let file2 = syn::parse_file(src2).expect("walk-2 fixture parses");
        let id1 = register_rust_module_at(&file1, Some(path)).expect("walker must succeed");
        let host1 = lookup_host(id1, "ParityProbe_LastWriter_Cls").expect("walk 1 binds class");
        let id2 = register_rust_module_at(&file2, Some(path)).expect("walker must succeed");
        assert_eq!(id1, id2, "path-keyed re-walk reuses same ModuleId");
        let host2 = lookup_host(id2, "ParityProbe_LastWriter_Cls").expect("walk 2 binds class");
        assert_ne!(
            host1, host2,
            "walk 2 mints a fresh HostObject::Class carrier and clobbers walk 1's binding",
        );
    }

    #[test]
    fn register_rust_module_eq_ne_mixed_types_returns_bool_not_typeerror() {
        // Codex parity audit (2026-05-05): Python 3 `==` / `!=` over
        // distinct primitive types do NOT raise — `1 == "1"` returns
        // `False`, `1 != "1"` returns `True`. Only ordering
        // (`<`, `<=`, `>`, `>=`) raises `TypeError`. The walker's
        // `eval_binop` previously routed every mixed-type pair to
        // `TypeError` via the catch-all, mis-aborting valid module
        // top-level constants like `pub const X: bool = 1 == "1";`.
        //
        // Bare-binop and paren-wrapped fixtures must agree per
        // `Expr::Paren` transparency — `(1 == "a")` and `1 == "a"`
        // produce the same binding (matches `build_flow.rs:1317`).
        let src = "pub const ParityProbe_Issue3_eq_int_str: bool = 1 == \"a\";
                   pub const ParityProbe_Issue3_ne_int_str: bool = 1 != \"a\";";
        let file = syn::parse_file(src).expect("eq-mixed fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        assert_eq!(
            super::super::host_env::module_globals_lookup(
                module_id,
                "ParityProbe_Issue3_eq_int_str"
            ),
            Some(ConstValue::Bool(false)),
            "Python 3: `1 == \"a\"` is False, NOT TypeError",
        );
        assert_eq!(
            super::super::host_env::module_globals_lookup(
                module_id,
                "ParityProbe_Issue3_ne_int_str"
            ),
            Some(ConstValue::Bool(true)),
            "Python 3: `1 != \"a\"` is True, NOT TypeError",
        );
        // Ordering on mixed types DOES raise TypeError — that arm of
        // `eval_binop` must continue to abort the walker.
        let bad_src = "pub const ParityProbe_Issue3_lt_mixed: bool = 1 < \"a\";";
        let bad_file = syn::parse_file(bad_src).expect("lt-mixed fixture parses");
        let err = register_rust_module(&bad_file)
            .expect_err("Python 3 `1 < \"a\"` raises TypeError — walker must abort");
        match err {
            AdapterError::Flowing { reason } => {
                assert!(
                    reason.contains("TypeError"),
                    "expected TypeError on mixed-type ordering, got: {reason}",
                );
            }
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_paren_is_transparent() {
        // Upstream Python parens evaporate at parse time, so
        // `(1 + 2)` and `1 + 2` produce identical bindings. `syn`
        // surfaces `Expr::Paren` and `Expr::Group` as wrapper
        // nodes; `eval_const_expr` must delegate transparently
        // to mirror `build_flow.rs:1317
        // Expr::Paren(ExprParen { expr, .. }) => lower_expr(b, expr)`.
        let src = "pub const ParityProbe_Paren_BareSum: i64 = 1 + 2;
                   pub const ParityProbe_Paren_WrappedSum: i64 = (1 + 2);
                   pub const ParityProbe_Paren_NestedWrap: i64 = (((1 + 2)));
                   pub const ParityProbe_Paren_WrappedEq: bool = (1 == 1);";
        let file = syn::parse_file(src).expect("paren-transparency fixture parses");
        let module_id = register_rust_module(&file).expect("walker must succeed");
        let lookup = |name: &str| super::super::host_env::module_globals_lookup(module_id, name);
        assert_eq!(
            lookup("ParityProbe_Paren_BareSum"),
            Some(ConstValue::Int(3))
        );
        assert_eq!(
            lookup("ParityProbe_Paren_WrappedSum"),
            Some(ConstValue::Int(3)),
            "`(1 + 2)` must fold identically to `1 + 2`",
        );
        assert_eq!(
            lookup("ParityProbe_Paren_NestedWrap"),
            Some(ConstValue::Int(3)),
            "nested parens are transparent at every level",
        );
        assert_eq!(
            lookup("ParityProbe_Paren_WrappedEq"),
            Some(ConstValue::Bool(true)),
            "paren around comparison stays transparent",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_path_falls_back_to_builtins() {
        // Upstream `flowcontext.py:845-853 find_global`: globals miss
        // → `getattr(__builtin__, varname)` → NameError on builtins
        // miss. pyre's adapter has `pyre_stdlib_lookup` as the
        // closed-world `__builtin__` analogue (`Ok` / `Some` / `Err` /
        // `Result` / `Option`). The const evaluator must consult it
        // before raising NameError, so a top-level
        // `pub const X: Result = Ok;` (where `Ok` is not in walker
        // bindings or registry) resolves through builtins.
        let src = "pub const ParityProbe_Builtins_Ok: i64 = 0;
                   pub const ParityProbe_Builtins_Forward: bool = Ok == Ok;";
        let file = syn::parse_file(src).expect("builtins-fallback fixture parses");
        let module_id = register_rust_module(&file).expect(
            "walker must succeed — `Ok` resolves through pyre_stdlib_lookup, not NameError",
        );
        // The eq comparison succeeds because both sides resolve to
        // the same `HostObject(Ok)` singleton (PYRE_STDLIB returns
        // identical `HostObject` clones across lookups; structural
        // equality holds).
        let v = super::super::host_env::module_globals_lookup(
            module_id,
            "ParityProbe_Builtins_Forward",
        );
        assert_eq!(
            v,
            Some(ConstValue::Bool(true)),
            "`Ok == Ok` must fold to True via builtins fallback (Ok singleton)",
        );

        // Negative control: a bare identifier that is neither in
        // bindings nor in the registry nor in PYRE_STDLIB still
        // raises NameError per upstream's final
        // AttributeError → FlowingError step.
        let bad_src = "pub const ParityProbe_Builtins_NotABuiltin: i64 =
            ParityProbe_Builtins_Unbound + 1;";
        let bad_file = syn::parse_file(bad_src).expect("bad fixture parses");
        let err = register_rust_module(&bad_file)
            .expect_err("name absent from all three channels must raise NameError");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_Builtins_Unbound' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_aborts_on_unbound_name() {
        // Forward-reference to a name not yet bound surfaces as
        // `FlowingError("global name '...' is not defined")` per
        // `flowcontext.py:853 find_global`. Mirrors upstream
        // Python's module-execution-aborts-on-exception semantic:
        // `Y = X + 1; X = 1` fails on Y with NameError and the
        // `X = 1` statement never executes. Walker therefore aborts
        // before reaching the LATER binding.
        let src =
            "pub const ParityProbe_Issue4_const_FORWARD: i64 = ParityProbe_Issue4_const_LATER + 1;
                   pub const ParityProbe_Issue4_const_LATER: i64 = 1;";
        let file = syn::parse_file(src).expect("forward-ref fixture parses");
        let err = register_rust_module(&file)
            .expect_err("forward-ref to unbound name aborts module execution");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("global name 'ParityProbe_Issue4_const_LATER' is not defined"),
                "expected NameError-shaped Flowing error, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_float_char_byte_literals() {
        // Slice O11: `Lit::Float` / `Lit::Char` / `Lit::Byte` at the
        // module top level resolve to the same `ConstValue` shape
        // as their in-body counterparts at
        // `build_flow.rs::lower_literal`. A `const PI: f64 = 3.14`
        // and a body `3.14` must therefore agree on the bit pattern.
        let src = r#"
pub const ParityProbe_O11_Float: f64 = 3.14;
pub const ParityProbe_O11_FloatNeg: f64 = -2.5;
pub const ParityProbe_O11_FloatZero: f64 = 0.0;
pub const ParityProbe_O11_Char: char = 'a';
pub const ParityProbe_O11_CharUnicode: char = '한';
pub const ParityProbe_O11_Byte: u8 = b'a';
"#;
        let file = syn::parse_file(src).expect("float/char/byte fixture parses");
        let module_id = register_rust_module(&file).expect("float/char/byte literals walk");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Float"),
            Some(ConstValue::Float(3.14_f64.to_bits())),
            "Lit::Float folds to ConstValue::Float(to_bits())",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_FloatNeg"),
            Some(ConstValue::Float((-2.5_f64).to_bits())),
            "syn parses `-2.5` as `Expr::Unary(Neg, Lit::Float(2.5))`, \
             so the Neg arm folds via operation.rs::pyfunc's \
             (OpKind::Neg, [ConstValue::Float]) parity",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_FloatZero"),
            Some(ConstValue::Float(0.0_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Char"),
            Some(ConstValue::uni_str("a".to_string())),
            "Lit::Char folds to single-codepoint UniStr",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_CharUnicode"),
            Some(ConstValue::uni_str("한".to_string())),
            "non-ASCII char literal preserves the codepoint",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O11_Byte"),
            Some(ConstValue::ByteStr(vec![b'a'])),
            "Lit::Byte folds to single-byte ByteStr",
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_float_arithmetic() {
        // Slice O14: `eval_binop` admits `(Float, Float)` operand
        // pairs for arithmetic + comparison, mirroring
        // `operation.rs::pyfunc`'s Float arms (`operation.rs:1367-1383`).
        // Compound `const Z: f64 = X + Y` therefore folds at module
        // top level the same way the function-body `X + Y` would.
        // Rust `BinOp::Div` over `f64` is true division (matches
        // Python `/`). `Float / 0.0` raises `ZeroDivisionError`.
        let src = r#"
pub const ParityProbe_O14_FAdd: f64 = 1.0 + 2.5;
pub const ParityProbe_O14_FSub: f64 = 5.0 - 1.5;
pub const ParityProbe_O14_FMul: f64 = 2.0 * 3.5;
pub const ParityProbe_O14_FDiv: f64 = 7.0 / 2.0;
pub const ParityProbe_O14_FEq: bool = 1.5 == 1.5;
pub const ParityProbe_O14_FNe: bool = 1.5 != 2.0;
pub const ParityProbe_O14_FLt: bool = 1.5 < 2.0;
pub const ParityProbe_O14_FLe: bool = 1.5 <= 1.5;
pub const ParityProbe_O14_FGt: bool = 2.5 > 1.5;
pub const ParityProbe_O14_FGe: bool = 1.5 >= 1.5;
"#;
        let file = syn::parse_file(src).expect("float-arith fixture parses");
        let module_id = register_rust_module(&file).expect("Float arithmetic walks");
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FAdd"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FSub"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FMul"),
            Some(ConstValue::Float(7.0_f64.to_bits())),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FDiv"),
            Some(ConstValue::Float(3.5_f64.to_bits())),
            "Rust `/` over f64 is true division (matches Python `/`)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FEq"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FNe"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FLt"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FLe"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FGt"),
            Some(ConstValue::Bool(true)),
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_O14_FGe"),
            Some(ConstValue::Bool(true)),
        );
    }

    #[test]
    fn register_rust_module_const_evaluator_float_div_by_zero_raises() {
        // Float `/ 0.0` raises `ZeroDivisionError` per upstream
        // `operator.truediv(_, 0.0)`. Walker aborts the file.
        let src = "pub const ParityProbe_O14_FDivZero: f64 = 1.0 / 0.0;";
        let file = syn::parse_file(src).expect("float-zerodiv fixture parses");
        let err =
            register_rust_module(&file).expect_err("float division by zero must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError"),
                "expected ZeroDivisionError, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_float_mod_binds_and_zero_raises() {
        // Rust `%` is `BinOp::Rem`. The const-evaluator routes
        // `Float % Float` through `flowspace::operation::float_py_mod`
        // so the bound value matches the function-body constfold,
        // including the signed-zero copysign and sign-of-denominator
        // correction from `pypy/objspace/std/floatobject.py:543-563
        // descr_mod`. Mirror of
        // `register_rust_module_const_evaluator_float_arithmetic` for
        // the `%` operator.
        let src = "\
            pub const ParityProbe_FModOK: f64 = 7.0 % 2.0;
            pub const ParityProbe_FModSign: f64 = -7.0 % 2.0;
            pub const ParityProbe_FModNegDiv: f64 = 7.0 % -2.0;
        ";
        let file = syn::parse_file(src).expect("float-mod fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        match module_globals_for_test(module_id, "ParityProbe_FModOK") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), 1.0),
            other => panic!("expected Float(1.0), got {other:?}"),
        }
        // `(-7.0) % 2.0`: divisor is positive, remainder follows
        // divisor sign → +1.0.
        match module_globals_for_test(module_id, "ParityProbe_FModSign") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), 1.0),
            other => panic!("expected Float(1.0), got {other:?}"),
        }
        // `7.0 % (-2.0)`: divisor is negative, remainder follows
        // divisor sign → -1.0.
        match module_globals_for_test(module_id, "ParityProbe_FModNegDiv") {
            Some(ConstValue::Float(bits)) => assert_eq!(f64::from_bits(bits), -1.0),
            other => panic!("expected Float(-1.0), got {other:?}"),
        }

        // Zero-divisor surfaces as FlowingError matching upstream
        // `operator.mod(_, 0.0)` ZeroDivisionError. Aborts walker.
        let src = "pub const ParityProbe_FModZero: f64 = 1.0 % 0.0;";
        let file = syn::parse_file(src).expect("float-mod-zero fixture parses");
        let err =
            register_rust_module(&file).expect_err("float modulo by zero must abort the walker");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError") && reason.contains("float modulo"),
                "expected ZeroDivisionError float modulo, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mixed_numeric_arithmetic_uses_python_coercion() {
        // Mirrors `flowspace::operation::pyfunc`'s mixed-numeric
        // arithmetic so the const-evaluator and the function-body
        // constfold agree on Python `bool ⊂ int ⊂ float`. Cross-type
        // pairs (Int+Float, Bool+Int, Int%Float, etc.) MUST bind a
        // value, not abort with TypeError.
        //
        // Upstream `PureOperation.constfold` (operation.py:120-127)
        // calls `operator.<op>(*args)` directly; Python's runtime
        // coerces `1 + 1.0 == 2.0`, `True * 3 == 3`, `7 % 2.5 == 2.0`,
        // etc. The mixed-numeric arm in `eval_binop` ports that
        // behavior.
        let src = "\
            pub const INT_ONE: i64 = 1;
            pub const FLOAT_HALF: f64 = 0.5;
            pub const BOOL_TRUE: bool = true;
            pub const ParityProbe_IntPlusFloat: f64 = INT_ONE + FLOAT_HALF;
            pub const ParityProbe_FloatTimesInt: f64 = FLOAT_HALF * INT_ONE;
            pub const ParityProbe_BoolPlusInt: i64 = BOOL_TRUE + INT_ONE;
            pub const ParityProbe_IntDivFloat: f64 = INT_ONE / FLOAT_HALF;
            pub const ParityProbe_IntModFloat: f64 = INT_ONE % FLOAT_HALF;
            pub const ParityProbe_BoolMulFloat: f64 = BOOL_TRUE * FLOAT_HALF;
        ";
        let file = syn::parse_file(src).expect("mixed-numeric fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        let f = |name: &str, expected: f64| match module_globals_for_test(module_id, name) {
            Some(ConstValue::Float(bits)) => assert_eq!(
                f64::from_bits(bits),
                expected,
                "{name}: expected Float({expected})",
            ),
            other => panic!("{name}: expected Float({expected}), got {other:?}"),
        };
        f("ParityProbe_IntPlusFloat", 1.5);
        f("ParityProbe_FloatTimesInt", 0.5);
        f("ParityProbe_IntDivFloat", 2.0);
        // 1 % 0.5 == 0.0 (1.0 - 0.5 * floor(1.0/0.5) == 0.0).
        f("ParityProbe_IntModFloat", 0.0);
        f("ParityProbe_BoolMulFloat", 0.5);

        match module_globals_for_test(module_id, "ParityProbe_BoolPlusInt") {
            Some(ConstValue::Int(2)) => {}
            other => panic!("expected Int(2), got {other:?}"),
        }

        // Float zero divisor through mixed-numeric Div is float-specific
        // ZeroDivisionError per upstream Python.
        let src = "\
            pub const FLOAT_ZERO: f64 = 0.0;
            pub const ParityProbe_IntDivFloatZero: f64 = 1 / FLOAT_ZERO;
        ";
        let file = syn::parse_file(src).expect("int/float-zero fixture parses");
        let err = register_rust_module(&file).expect_err("Int / Float(0.0) must raise float ZDE");
        match err {
            AdapterError::Flowing { reason } => assert!(
                reason.contains("ZeroDivisionError") && reason.contains("float division by zero"),
                "expected float ZDE, got: {reason}",
            ),
            other => panic!("expected AdapterError::Flowing, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_int_float_compare_handles_mantissa_boundary() {
        // Port test for `pypy/objspace/std/floatobject.py:135-148`'s
        // bigint-aware Int↔Float comparison. f64 mantissa is 53 bits;
        // ints with `|i| > 2^53` are not exactly representable as f64.
        // Upstream's `_compare` routes through `do_compare_bigint` for
        // these values, comparing against `floor(f)` exactly. Naive
        // `*x as f64` would round `Int(2^53 + 1)` to `2^53`,
        // misclassifying it as equal to `Float(2^53)`.
        //
        // 2^53 = 9007199254740992 (exactly representable as f64)
        // 2^53 + 1 = 9007199254740993 (NOT representable; f64 rounds
        //   to 2^53)
        let src = "\
            pub const I_2_TO_53_PLUS_1: i64 = 9007199254740993;
            pub const F_2_TO_53: f64 = 9007199254740992.0;
            pub const ParityProbe_BigIntEqFloat: bool = I_2_TO_53_PLUS_1 == F_2_TO_53;
            pub const ParityProbe_BigIntGtFloat: bool = I_2_TO_53_PLUS_1 > F_2_TO_53;
            pub const ParityProbe_FloatLtBigInt: bool = F_2_TO_53 < I_2_TO_53_PLUS_1;
        ";
        let file = syn::parse_file(src).expect("mantissa-boundary fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        // Eq: i = 2^53 + 1, f = 2^53. They are NOT equal; naive cast
        // would say True (because (2^53 + 1) as f64 == 2^53).
        match module_globals_for_test(module_id, "ParityProbe_BigIntEqFloat") {
            Some(ConstValue::Bool(false)) => {}
            other => {
                panic!("Int(2^53+1) == Float(2^53) must be False (bigint compare), got {other:?}",)
            }
        }
        // Gt: i > f, since i = 2^53 + 1 > f = 2^53.
        match module_globals_for_test(module_id, "ParityProbe_BigIntGtFloat") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("Int(2^53+1) > Float(2^53) must be True, got {other:?}"),
        }
        match module_globals_for_test(module_id, "ParityProbe_FloatLtBigInt") {
            Some(ConstValue::Bool(true)) => {}
            other => panic!("Float(2^53) < Int(2^53+1) must be True, got {other:?}"),
        }
    }

    #[test]
    fn register_rust_module_const_evaluator_mixed_numeric_eq_uses_python_coercion() {
        // Cross-numeric `==` / `!=` / `<` / `<=` / `>` / `>=` follow
        // Python's `bool ⊂ int ⊂ float` coercion, matching upstream
        // `PureOperation.constfold` calling `operator.eq(*args)`
        // directly. Without this, `Int(1) == Float(1.0)` would fold
        // to `False` via structural ConstValue equality, diverging
        // from Python.
        //
        // Cross-numeric pairs reach the const-evaluator via the
        // bindings/registry/builtins resolver chain (e.g. an `Int`
        // const compared to a `Float` const). Rust's typed source
        // can't write `1 == 1.0` directly, but it can write
        // `INT_ONE == FLOAT_ONE` after binding both names — that's
        // the path this test exercises.
        let src = "\
            pub const INT_ONE: i64 = 1;
            pub const FLOAT_ONE: f64 = 1.0;
            pub const BOOL_TRUE: bool = true;
            pub const ParityProbe_IntEqFloat: bool = INT_ONE == FLOAT_ONE;
            pub const ParityProbe_BoolEqInt: bool = BOOL_TRUE == INT_ONE;
            pub const ParityProbe_BoolEqFloat: bool = BOOL_TRUE == FLOAT_ONE;
            pub const ParityProbe_FloatLtInt: bool = FLOAT_ONE < INT_ONE;
        ";
        let file = syn::parse_file(src).expect("mixed-numeric fixture parses");
        let module_id = register_rust_module(&file).expect("walk succeeds");

        for (name, expected) in [
            ("ParityProbe_IntEqFloat", true),
            ("ParityProbe_BoolEqInt", true),
            ("ParityProbe_BoolEqFloat", true),
            ("ParityProbe_FloatLtInt", false), // 1.0 < 1 == False
        ] {
            match module_globals_for_test(module_id, name) {
                Some(ConstValue::Bool(v)) => assert_eq!(
                    v, expected,
                    "{name}: expected Bool({expected}), got Bool({v})",
                ),
                other => panic!("{name}: expected Bool, got {other:?}"),
            }
        }
    }
}
