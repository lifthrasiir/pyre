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
    ItemConst, ItemEnum, ItemFn, ItemStruct, Lit, Local, Pat, PatIdent, UnOp,
};

use super::build_flow::{AdapterError, build_flow_from_rust_in_module};
use super::host_env::{
    ModuleId, module_globals_lookup, module_globals_snapshot, register_module_global,
};
use crate::flowspace::bytecode::HostCode;
use crate::flowspace::model::{ConstValue, Constant, GraphFunc, HostObject};
use crate::flowspace::objspace::CO_NEWLOCALS;
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
    let module_id = register_rust_module_at(file, source_filename);

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
/// ### Idempotence
///
/// Re-calling on the same file (or a file containing a name already
/// registered from another walk) is a no-op for already-bound names —
/// `register_module_global` keeps the first registration to preserve
/// the per-process identity invariant. This matches Python's
/// `sys.modules` cache: re-importing a module does not rebuild the
/// class objects, it returns the cached module unchanged.
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
/// - **`Item::Const`** → `MODULE_NAME = <literal>` at module top
///   level. Bound to `module.__dict__[MODULE_NAME]` as the literal
///   value itself. Stored as `ConstValue::Int/Bool/UniStr/ByteStr`
///   directly (no HostObject wrapper) — mirrors upstream
///   `find_global` returning `const(value)` regardless of value
///   type. Only literal RHS exprs are supported in this slice
///   (`Lit::Int` / `Lit::Bool` / `Lit::Str` / `Lit::ByteStr` /
///   unary-`Neg` over `Lit::Int`); compound const expressions
///   (`const Y: i64 = X + 1;` referring to other consts, calls to
///   `const fn`, etc.) require a richer evaluator and are skipped
///   here — each falls through to the
///   `Builder::resolve_path_constant` mint-or-fail path.
///
/// Other `Item::*` kinds (`Item::Fn`, `Item::Static`, `Item::Use`,
/// `Item::Mod`, `Item::Impl`, …) are silently skipped. `Item::Fn`
/// for the parity reason above; the others as upstream-walker
/// follow-ups (each populates `module.__dict__` at Python import
/// time too). Each future slice extends the dispatch match without
/// changing the call sites.
///
/// ### Per-module scoping (Issue 1.3, 2026-05-05)
///
/// Returns a fresh [`ModuleId`] minted at the start of the walk;
/// every `register_module_global` call inside this function tags
/// its entry with that id. Callers thread the returned id into
/// `Builder.module_id` (via `build_flow_from_rust_in_module` /
/// `build_host_function_from_rust_in_module`) so the body lowerer's
/// `LOAD_GLOBAL` lookups resolve against the matching partition.
/// Two separate walks of files that share top-level names mint
/// distinct ids and never collide — mirroring upstream
/// `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
/// per-module scoping.
///
/// This BC entry routes through [`register_rust_module_at`] with
/// no path. Callers that want re-walks of the same source file to
/// converge on a single id (matching upstream `sys.modules`) call
/// [`register_rust_module_at`] directly with `Some(path)`.
pub fn register_rust_module(file: &File) -> ModuleId {
    register_rust_module_at(file, None)
}

/// Path-aware sibling of [`register_rust_module`]. When
/// `source_filename` is `Some(path)`, the registry id is keyed on
/// `path`: two walks of files at the same path converge on the
/// same [`ModuleId`] and the second walk is a no-op (entries
/// already first-writer-wins-bound from the first walk). When
/// `None`, mints a fresh id per call.
///
/// Mirrors upstream's two ways of obtaining a module dict:
///
/// - **Path-keyed** ↔ `sys.modules[modulename]`: every `def` /
///   `class` statement in the same source file binds into the same
///   `__dict__` via the import-cache lookup. Two `Translation`
///   instances built against entry points from the same file see
///   identical `func.__globals__` references.
/// - **Anonymous** ↔ `exec(source, dict={})`: the module dict is a
///   throwaway, so two walks against the same source create two
///   independent dicts. Right answer for unit-test fixtures and the
///   bare [`build_host_function_from_rust`] single-`ItemFn` entry.
pub fn register_rust_module_at(file: &File, source_filename: Option<&str>) -> ModuleId {
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
                if module_globals_lookup(module_id, &name).is_some() {
                    // First-writer-wins idempotence within this
                    // module's partition (defensive — Rust does
                    // not allow two top-level enums with the same
                    // name in one source file, so this branch
                    // is normally unreachable).
                    continue;
                }
                let host = build_host_class_from_enum(item_enum);
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Struct(item_struct) => {
                let name = item_struct.ident.to_string();
                if module_globals_lookup(module_id, &name).is_some() {
                    continue;
                }
                let host = build_host_class_from_struct(item_struct);
                register_module_global(module_id, &name, ConstValue::HostObject(host));
            }
            Item::Const(item_const) => {
                let name = item_const.ident.to_string();
                if module_globals_lookup(module_id, &name).is_some() {
                    continue;
                }
                // upstream Python import-time evaluation: the RHS
                // runs against the partially-built `module.__dict__`,
                // so compound expressions like `const Y: i64 = X + 1`
                // resolve `X` through the prior binding. The walker
                // threads `const_bindings` (the local source-order
                // accumulator) into the evaluator so forward
                // dependencies between sibling consts work without
                // a process-global registry round-trip.
                if let Some(value) = eval_const_expr(&item_const.expr, &const_bindings) {
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
                // - **`Item::Static`** — module-level mutable
                //   bindings; upstream `MUTABLE = []` at module
                //   top level. Walker dispatch needs the same
                //   literal evaluator as `Item::Const` plus a
                //   mutability marker on the `ConstValue`.
                // - **Compound `Item::Const`** (`const Y = X + 1`)
                //   — needs a const-expression evaluator capable of
                //   threading prior registry entries through binop
                //   / call ops.
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
    module_id
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
/// the same module walk. Returns `None` for unsupported shapes
/// (`Lit::Float`, `Lit::Char`, calls, struct literals, …) so the
/// walker can skip the entry, leaving the name unresolved exactly
/// as upstream Python raises `FlowingError` only when a missing
/// global is referenced (lazy resolution).
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
/// - `-<Lit::Int>` (unary negation over an integer literal) →
///   `ConstValue::Int(-n)`. `syn` parses `const X: i64 = -1` as
///   `Expr::Unary { op: Neg, expr: Lit(1) }`, not a single
///   `Lit::Int(-1)`, so unwrap one level for the common signed-int
///   form.
/// - **`Expr::Path` (single segment)** → `bindings.get(name)`
///   lookup. Mirrors upstream Python's name resolution against
///   `module.__dict__` at import time: by the time the RHS of
///   `Y = X + 1` runs, the prior `X = 1` has already bound
///   `module.__dict__["X"]`. Multi-segment paths fall through to
///   `None` (a path like `mod::CONST_X` would require cross-file
///   lookup and is out of scope for this slice).
/// - **`Expr::Binary { Add | Sub | Mul | Div | Rem, lhs, rhs }`**
///   over `Int` operands → `Int(a OP b)`. Uses Rust's checked
///   arithmetic so overflow returns `None` (and the const skips
///   silently — same outcome as upstream raising on integer
///   overflow at import time). `Div` / `Rem` returns `None` on
///   zero divisor.
fn eval_const_expr(expr: &Expr, bindings: &StdHashMap<String, ConstValue>) -> Option<ConstValue> {
    match expr {
        Expr::Lit(ExprLit { lit, .. }) => match lit {
            Lit::Int(n) => n.base10_parse::<i64>().ok().map(ConstValue::Int),
            Lit::Bool(b) => Some(ConstValue::Bool(b.value)),
            // `"..."` literal — unicode. Same shape as
            // `build_flow.rs::lower_literal::Lit::Str` so the
            // identical `"abc"` source carries the identical
            // ConstValue regardless of position.
            Lit::Str(s) => Some(ConstValue::uni_str(s.value())),
            // `b"..."` literal — bytes. Mirrors
            // `build_flow.rs::lower_literal::Lit::ByteStr`.
            Lit::ByteStr(s) => Some(ConstValue::ByteStr(s.value())),
            _ => None,
        },
        // `const X: i64 = -1` — `syn` parses as `Unary { op: Neg,
        // expr: Lit(1) }` rather than a signed literal. Unwrap one
        // level so the common signed-int form is recognised.
        Expr::Unary(ExprUnary {
            op: UnOp::Neg(_),
            expr,
            ..
        }) => match eval_const_expr(expr, bindings)? {
            ConstValue::Int(n) => n.checked_neg().map(ConstValue::Int),
            _ => None,
        },
        // `const Y: i64 = X` — single-segment path resolves to a
        // prior binding in the same walk. Multi-segment paths are
        // out of scope.
        Expr::Path(ExprPath {
            qself: None, path, ..
        }) if path.segments.len() == 1 => {
            let seg = &path.segments[0];
            if !seg.arguments.is_empty() {
                return None;
            }
            bindings.get(&seg.ident.to_string()).cloned()
        }
        // `const Y: i64 = X + 1` — checked arithmetic over
        // `ConstValue::Int` operands. Other operand kinds (float,
        // bool, str) and other binops (shifts, bitwise, comparisons)
        // are out of scope until pyre exercises them.
        Expr::Binary(ExprBinary {
            left, op, right, ..
        }) => {
            let lhs = eval_const_expr(left, bindings)?;
            let rhs = eval_const_expr(right, bindings)?;
            let (ConstValue::Int(a), ConstValue::Int(b)) = (lhs, rhs) else {
                return None;
            };
            match op {
                BinOp::Add(_) => a.checked_add(b).map(ConstValue::Int),
                BinOp::Sub(_) => a.checked_sub(b).map(ConstValue::Int),
                BinOp::Mul(_) => a.checked_mul(b).map(ConstValue::Int),
                BinOp::Div(_) => a.checked_div(b).map(ConstValue::Int),
                BinOp::Rem(_) => a.checked_rem(b).map(ConstValue::Int),
                _ => None,
            }
        }
        _ => None,
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
        // A body using a construct the body lowerer rejects (`x as
        // i64` — task #94, `as T` cast removal epic) demonstrates
        // this directly: the body lowerer would surface
        // `AdapterError::Unsupported` if invoked, but the metadata
        // path bypasses the body and succeeds.
        let item = parse("fn helper(x: u32) -> i64 { x as i64 }");
        // First confirm the body lowerer rejects this fixture so the
        // bypass is actually load-bearing — if `x as i64` ever lands
        // in `build_flow_from_rust`'s subset, this assertion will
        // fail loudly and the test author can refresh the body to a
        // still-rejected construct.
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
        let module_id = register_rust_module(&file);

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
        let module_id = register_rust_module(&file);
        assert!(
            module_globals_lookup(module_id, "parity_probe_walker_with_cast").is_none(),
            "Item::Fn skip is unconditional regardless of body lowerability",
        );
    }

    #[test]
    fn register_rust_module_skips_non_walked_item_kinds() {
        // Walker dispatches `Item::Fn`, `Item::Enum`, `Item::Struct`,
        // and `Item::Const` (Slice O10). Other kinds (`Item::Static`,
        // `Item::Use`, …) are follow-up slices — they must NOT
        // pollute the module-globals registry until their dispatch
        // is added.
        use super::super::host_env::module_globals_lookup;

        let src = "static PARITY_PROBE_WALKER_STATIC_ONLY: i64 = 1;";
        let file = syn::parse_file(src).expect("file fixture parses");
        let module_id = register_rust_module(&file);
        assert!(module_globals_lookup(module_id, "PARITY_PROBE_WALKER_STATIC_ONLY").is_none());
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
        let module_id = register_rust_module(&file);

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
        let module_id = register_rust_module(&file);
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
    fn register_rust_module_at_with_same_path_returns_same_id() {
        // Issue 2 (2026-05-05): two walks of the same source path
        // converge on the same `ModuleId` — mirrors upstream
        // `sys.modules[path]` import-cache. Second walk's
        // registrations are noops (first-writer-wins inside the
        // partition) but the id matches so a downstream
        // `Builder.module_id` lookup against either id sees the
        // first walk's bindings.

        let src = "pub struct ParityProbe_Issue2_path_share;";
        let file1 = syn::parse_file(src).expect("file 1 parses");
        let file2 = syn::parse_file(src).expect("file 2 parses");
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_share.rs"));
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_share.rs"));
        assert_eq!(
            id1, id2,
            "same path must yield the same ModuleId (sys.modules cache parity)",
        );
        let host = lookup_host(id1, "ParityProbe_Issue2_path_share")
            .expect("first walk's binding visible from shared id");
        let cross = lookup_host(id2, "ParityProbe_Issue2_path_share").unwrap();
        assert_eq!(
            host, cross,
            "shared id must serve identical HostObject identity across walks",
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
        let id1 = register_rust_module_at(&file1, Some("/parity_probe/issue2_distinct_a.rs"));
        let id2 = register_rust_module_at(&file2, Some("/parity_probe/issue2_distinct_b.rs"));
        assert_ne!(id1, id2, "distinct paths must mint distinct ids");
        let host1 = lookup_host(id1, "ParityProbe_Issue2_path_distinct").unwrap();
        let host2 = lookup_host(id2, "ParityProbe_Issue2_path_distinct").unwrap();
        assert_ne!(
            host1, host2,
            "distinct-path walks must produce independent class identities",
        );
    }

    #[test]
    fn register_rust_module_at_with_none_path_mints_fresh_each_call() {
        // Issue 2 (2026-05-05): `None` path falls back to
        // `ModuleId::fresh()` (anonymous module, like
        // `exec(source, dict={})`). Two anonymous walks of the same
        // source remain isolated — caller didn't ask for caching, so
        // they shouldn't get it.
        let src = "pub struct ParityProbe_Issue2_anonymous;";
        let file = syn::parse_file(src).expect("anonymous walk parses");
        let id1 = register_rust_module_at(&file, None);
        let id2 = register_rust_module_at(&file, None);
        assert_ne!(
            id1, id2,
            "None path must mint a fresh id per call (anonymous walks stay isolated)",
        );
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
        let id1 = register_rust_module(&file1);
        let id2 = register_rust_module(&file2);
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
    fn register_rust_module_idempotent_within_single_walk_for_duplicate_name() {
        // Within a single `register_rust_module` call, a duplicate
        // top-level name is defensive (Rust would fail to compile)
        // — but if such a fixture is parsed, the first registration
        // wins to preserve the identity invariant for whoever already
        // observed the binding inside the same partition. Mirrors
        // upstream Python semantics where a `class` statement
        // executed twice in the same module body would re-bind the
        // name, but our walker treats the partition as immutable
        // once written so observers stay coherent.
        //
        // We can't actually feed two `pub struct Foo;` to syn::parse_file
        // (it errors on duplicate items), so we instead exercise
        // `register_module_global` directly under a fixed id.
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
        assert_eq!(observed, first, "first registration wins within same id");
        assert_ne!(
            observed, second,
            "second registration must NOT clobber within same id",
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
        let module_id = register_rust_module(&file);
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
        let module_id = register_rust_module(&file);
        let value = module_globals_lookup(module_id, "ParityProbe_O10_const_neg_int")
            .expect("negated integer const registered after walk");
        assert_eq!(value, ConstValue::Int(-7));
    }

    #[test]
    fn register_rust_module_walks_item_const_bool_literal() {
        let src = "pub const ParityProbe_O10_const_bool: bool = true;";
        let file = syn::parse_file(src).expect("const bool parses");
        let module_id = register_rust_module(&file);
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
        let module_id = register_rust_module(&file);
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
        let module_id = register_rust_module(&file);
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
    fn register_rust_module_skips_compound_const_with_unsupported_op() {
        // Compound const with an op the evaluator does not yet
        // handle (shifts, comparisons, …) skips silently. Mirrors
        // upstream lazy-resolution: the name stays unresolved, and
        // a later call site raises `FlowingError` if it tries to
        // reference the missing global.
        let src = "pub const ParityProbe_Issue4_const_BASE: i64 = 1;
                   pub const ParityProbe_Issue4_const_SHIFT: i64 = ParityProbe_Issue4_const_BASE << 4;";
        let file = syn::parse_file(src).expect("compound shift fixture parses");
        let module_id = register_rust_module(&file);
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_BASE"),
            Some(ConstValue::Int(1)),
        );
        assert!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_SHIFT").is_none(),
            "unsupported binop must keep the const skipped per lazy-resolution semantics",
        );
    }

    #[test]
    fn register_rust_module_skips_compound_const_referencing_unbound_name() {
        // Forward-reference to a name not yet bound (or never bound)
        // skips silently. Upstream Python would raise `NameError` at
        // import time; the walker treats it as "evaluator returned
        // None" and skips, deferring the error to lazy resolution.
        let src =
            "pub const ParityProbe_Issue4_const_FORWARD: i64 = ParityProbe_Issue4_const_LATER + 1;
                   pub const ParityProbe_Issue4_const_LATER: i64 = 1;";
        let file = syn::parse_file(src).expect("forward-ref fixture parses");
        let module_id = register_rust_module(&file);
        assert!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_FORWARD").is_none(),
            "forward reference to LATER must skip (LATER not yet bound when FORWARD evaluates)",
        );
        assert_eq!(
            module_globals_lookup(module_id, "ParityProbe_Issue4_const_LATER"),
            Some(ConstValue::Int(1)),
            "LATER registers normally once its turn comes",
        );
    }
}
