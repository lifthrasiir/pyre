//! Rust-AST adapter's host-namespace resolver.
//!
//! Position-2 adaptation per
//! `~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`:
//! pyre-interpreter is Rust source, not Python, so the upstream
//! `flowcontext.py:856 LOAD_GLOBAL` chain (co_names → func_globals →
//! module dict) has no direct analogue. The adapter routes name
//! resolution through this module.
//!
//! Two layers:
//!
//! 1. `PYRE_STDLIB` — closed-world registry of Rust-stdlib identifiers
//!    pyre source uses as control-flow constructors (`Ok`, `Some`,
//!    `Err`, `Result`, `Option`). Each maps to a process-singleton
//!    `HostObject::Class` so two graphs that mention `Ok` see the same
//!    identity (mirrors Python's `__builtin__.Ok` returning the same
//!    object across every `LOAD_GLOBAL`).
//! 2. `HOST_CLASS_MINTS` — process-global cache of `HostObject::Class`
//!    minted on demand for path-prefix names the closed-world registry
//!    does not cover (e.g. `StepResult` in `StepResult::Continue`).
//!    Same identity invariant as PYRE_STDLIB: two graphs that name
//!    `StepResult` share the same `HostObject` (`Arc::ptr_eq` per
//!    `model.rs:208`), mirroring upstream `func_globals[name]` at
//!    `flowcontext.py:847` returning the same Python object on every
//!    `LOAD_GLOBAL` regardless of which graph is being built.
//!
//! `None` is special-cased away from this layer because Python's
//! `None` is the NoneType *singleton instance*, not a class — it
//! resolves to `Constant(ConstValue::None)` directly at the lower_expr
//! site, matching upstream `Constant(None)` in flowspace graphs.
//!
//! ## `HOST_RUST_MODULE_GLOBALS` — module-globals layer
//!
//! Upstream `flowcontext.py:845-854 find_global` reads
//! `w_globals.value[varname]` where `self.w_globals =
//! Constant(func.__globals__)` is set at `flowcontext.py:284` —
//! the function's `func_globals` dict, i.e. its owning module's
//! `__dict__`. Two different modules with the same top-level name
//! (`MAX`, `helper`, `StepResult`, …) carry distinct values in their
//! respective `func.__globals__`; the lookup at `:847` reads the
//! *current function's* globals, not a shared one.
//!
//! The Rust-source counterpart populates this layer through
//! [`super::register::register_rust_module`] which walks a
//! `syn::File` (the analogue of "module import") and inserts each
//! top-level `Item::Enum` / `Item::Struct` / `Item::Const` here.
//! The registry's value type is [`ConstValue`] to mirror upstream's
//! any-Python-value semantics — classes wrap as
//! `ConstValue::HostObject(...)`, consts as `ConstValue::Int` /
//! `Bool` / `UniStr` / `ByteStr` directly.
//!
//! ## Per-module scoping (Issue 1.3 closed 2026-05-05)
//!
//! Each [`register_rust_module`](super::register::register_rust_module)
//! call mints a fresh [`ModuleId`] and partitions the registry
//! entries it writes under that id. A `Builder` carries the same
//! `ModuleId` so its lookups only see entries from the file the
//! walker walked — matching upstream `func.__globals__` per-function
//! scoping. Two `register_rust_module` walks of files that share a
//! top-level name no longer collide: each walk's entries live under
//! a distinct id, mirroring two Python modules whose `__dict__`s
//! happen to share a key.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::flowspace::model::{ConstValue, HostObject};

/// Opaque module identity used to partition
/// [`HOST_RUST_MODULE_GLOBALS`] entries by the file that registered
/// them. Mirrors upstream `func.__globals__`: a function's globals
/// reference is a per-module dict, so two distinct modules with
/// identically-named top-level bindings see independent values.
///
/// Two construction shapes mirror Python's two ways of obtaining a
/// module dict:
///
/// - [`Self::fresh`] — anonymous, monotonic, never re-used. The
///   counterpart of running `exec(source)` against a brand-new
///   throwaway dict: every call mints a distinct id. Right answer
///   for one-shot fixtures, throwaway probe harnesses, and the
///   bare [`super::register::build_host_function_from_rust`]
///   single-`ItemFn` entry that has no enclosing file.
/// - [`Self::for_path`] — find-or-mint keyed on a stable path
///   string. Path-keyed `exec(source, shared_dict)` shape: every
///   walk of the same path re-executes against the **same**
///   `(module_id, _)` partition with last-writer-wins semantics
///   (see [`Self::for_path`]'s docstring). This is **not**
///   upstream `sys.modules[path]` import-cache behaviour — that
///   cache short-circuits the second `import` entirely; the
///   pyre walker has no such short-circuit and re-runs every
///   walk. The two-walks-converge invariant is the registry id
///   only, not the registry contents. Right answer for the
///   file-aware
///   [`super::register::build_host_function_from_rust_file`]
///   path: two walks of the same source file (e.g. one driver
///   test asks for entry-point A, another asks for entry-point B,
///   both inside the same `pyopcode.rs`) reach the same
///   partition so the entry points share their sibling-item
///   view as long as both walks succeed end-to-end. Callers
///   that want `sys.modules`-style "skip the second import"
///   semantics must gate the call themselves on a prior
///   [`module_globals_lookup`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(u64);

impl ModuleId {
    /// Mint a fresh `ModuleId`. Each call produces a distinct id —
    /// callers that need shared identity must store the id and pass
    /// it explicitly, or use [`Self::for_path`] for path-keyed
    /// caching.
    pub fn fresh() -> Self {
        Self(MODULE_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Find-or-mint a `ModuleId` keyed on `path`. Two calls with the
    /// same path return the same `id` so multiple walks of the same
    /// source file share one `(module_id, name)` partition in
    /// [`HOST_RUST_MODULE_GLOBALS`]. Each walk is `exec(source,
    /// shared_dict)`-shaped — every `Item::Const` / `Item::Enum` /
    /// `Item::Struct` re-runs and overwrites under the fixed
    /// `module_id` ([`register_module_global`] is last-writer-wins).
    /// This is **not** upstream `sys.modules[name]` import-cache
    /// behaviour: that cache short-circuits the re-execution
    /// entirely on the second `import`. The pyre walker has no such
    /// short-circuit, so callers that want import-cache semantics
    /// must gate the call themselves on a prior
    /// [`module_globals_lookup`] (or first-class
    /// `Translation`-level memo). The two-walks-converge invariant
    /// here is only the registry id, not the registry contents.
    ///
    /// `path` is the caller-supplied identity string. For a
    /// `syn::File` produced by `syn::parse_file(p)`, callers thread
    /// `p` (the filesystem path) here. For a `parse_str` fixture the
    /// caller picks any unique label they want re-walks to converge
    /// on. The cache is a process-global `Mutex<HashMap<String,
    /// ModuleId>>` so two callers (different threads, different
    /// `Translation` instances, …) sharing the same path see the
    /// same id.
    pub fn for_path(path: &str) -> Self {
        let mut map = MODULE_ID_BY_PATH
            .lock()
            .expect("MODULE_ID_BY_PATH Mutex poisoned");
        if let Some(id) = map.get(path) {
            return *id;
        }
        let id = Self(MODULE_ID_COUNTER.fetch_add(1, Ordering::Relaxed));
        map.insert(path.to_string(), id);
        id
    }
}

static MODULE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Path-keyed cache of [`ModuleId`]s. See [`ModuleId::for_path`].
static MODULE_ID_BY_PATH: LazyLock<Mutex<HashMap<String, ModuleId>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Rust-stdlib identifiers pre-registered as `HostObject::Class`
/// singletons.
///
/// Population rule: a name lands here when (a) it is structurally
/// PascalCase / capitalised so it cannot collide with pyre's
/// snake_case helper functions, AND (b) the adapter needs identity
/// for it before any source files are walked. New names join the
/// list as adapter coverage grows. Anything not in this list and not
/// in locals reaches the `HOST_CLASS_MINTS` mint path (multi-segment)
/// or rejects as `UnboundLocal` (single-segment).
///
/// The closed-world choice mirrors `HostEnv::bootstrap_builtin_types`
/// in `flowspace/model.rs:1471` — that table also pre-registers a
/// fixed set of class objects (`int`, `float`, `Exception`, …)
/// because the Rust port has no Python runtime to walk
/// `__builtin__.__dict__` dynamically.
pub(super) static PYRE_STDLIB: LazyLock<HashMap<String, HostObject>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for name in ["Ok", "Some", "Err", "Result", "Option"] {
        map.insert(name.to_string(), HostObject::new_class(name, vec![]));
    }
    map
});

/// Look up `name` in the closed-world Rust-stdlib registry. Returns
/// the singleton `HostObject::Class` if registered, `None` otherwise.
///
/// This is the analogue of `HostEnv::lookup_builtin` for the
/// Rust-AST adapter's auxiliary namespace; it does not consult
/// HOST_ENV (which is Python's `__builtin__`).
pub(super) fn pyre_stdlib_lookup(name: &str) -> Option<HostObject> {
    PYRE_STDLIB.get(name).cloned()
}

/// Process-global cache of `HostObject::Class` minted on demand for
/// path-prefix names that the closed-world `PYRE_STDLIB` registry
/// does not cover (e.g. `StepResult` in `StepResult::Continue`).
///
/// Lives for the duration of the process, mirroring upstream
/// `func_globals[name]` returning the same Python object on every
/// `LOAD_GLOBAL` (`flowcontext.py:847`). Two graphs that mention
/// `StepResult` therefore share the same `HostObject::Class` identity
/// — downstream annotator/codewriter compare class identity via
/// `Arc::ptr_eq` (`model.rs:208`) without a per-graph qualifier.
///
/// Population is incidental: `mint_host_class(name)` finds-or-mints,
/// the registry grows as new path prefixes are encountered. Concurrent
/// mints are serialized through the `Mutex`.
static HOST_CLASS_MINTS: LazyLock<Mutex<HashMap<String, HostObject>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Find-or-mint a `HostObject::Class` for `name` in the process-global
/// runtime registry. Returns the same `HostObject` (Arc identity) for
/// the same name across all calls and all graphs, mirroring upstream
/// `LOAD_GLOBAL` returning the same Python object on every lookup of
/// the same global name (`flowcontext.py:845-854 find_global` →
/// `w_globals.value[varname]` reads from a process-shared globals
/// dict).
///
/// Caller is expected to consult `pyre_stdlib_lookup` first if the
/// name might be in the closed-world registry — keeping the two
/// layers separate matches `HostEnv::bootstrap_builtin_types` (closed)
/// vs `HostEnv::lookup` (open) in `flowspace/model.rs`.
/// Return `true` if `host` is identical (by `Arc::ptr_eq`) to a
/// class minted via [`mint_host_class`]. Used by the constfold
/// path in [`super::build_flow`] to distinguish walker-registered
/// classes (whose class dicts are authoritative — a member miss
/// is a real `AttributeError` per upstream
/// `operation.py:638-642 GetAttr.constfold`) from mint-on-demand
/// classes (whose dicts are intentionally empty pending walker
/// coverage — a miss must keep falling through to the raw
/// `getattr` op until mint-on-demand is retired).
///
/// PRE-EXISTING-ADAPTATION: this helper exists *only* because the
/// mint-on-demand layer is a stand-in for missing walker coverage;
/// once mint is retired, every Constant<HostObject::Class> reaching
/// the constfold path will be walker-registered and the helper is
/// removed.
pub(super) fn is_host_class_minted(host: &HostObject) -> bool {
    let map = HOST_CLASS_MINTS
        .lock()
        .expect("HOST_CLASS_MINTS Mutex poisoned");
    map.values().any(|m| m == host)
}

pub(super) fn mint_host_class(name: &str) -> HostObject {
    let mut map = HOST_CLASS_MINTS
        .lock()
        .expect("HOST_CLASS_MINTS Mutex poisoned");
    if let Some(class) = map.get(name) {
        return class.clone();
    }
    let class = HostObject::new_class(name, vec![]);
    map.insert(name.to_string(), class.clone());
    class
}

/// Module-globals registry keyed on `(ModuleId, name)`. Mirrors
/// upstream `func_globals.value[varname]` at `flowcontext.py:847`:
/// each function carries a per-module reference to its owning
/// module's `__dict__`, and the lookup reads the *current
/// function's* globals — not a process-wide shared dict.
///
/// The outer `HashMap<ModuleId, _>` partitions entries by the
/// `register_rust_module` walk that wrote them, so two walks of
/// files that share a top-level name no longer collide. Each walk
/// mints a fresh [`ModuleId`] (via [`ModuleId::fresh`]) and
/// `Builder` threads the same id through body lowering so its
/// `LOAD_GLOBAL` lookups only see entries from the file the walker
/// walked.
///
/// The inner value type is [`ConstValue`] so the registry can hold
/// any of the upstream-bindable shapes:
///
/// - `ConstValue::HostObject(...)` — class objects
///   (`Item::Enum`, `Item::Struct`). `Item::Fn` is intentionally
///   NOT registered — the body-rebuild path between
///   `FunctionDesc.buildgraph` (`description.py:140`) and the Rust
///   AST adapter is missing, so a sibling-fn HostObject would
///   masquerade as callable while routing to empty bytecode. See
///   `register_rust_module`'s "Why no Item::Fn?" docstring.
/// - `ConstValue::Int(...)` — integer constants (`Item::Const`
///   with `Lit::Int` rhs).
/// - `ConstValue::Bool(...)` — boolean constants.
/// - `ConstValue::UniStr(...)` — unicode string constants
///   (`Lit::Str`); shape matches in-body `build_flow.rs::lower_literal`
///   so the same `"..."` source carries the identical
///   `ConstValue` regardless of position.
/// - `ConstValue::ByteStr(...)` — byte-string constants
///   (`Lit::ByteStr`).
///
/// Mirrors upstream's any-value semantic at `find_global` line 854:
/// `return const(value)` wraps whatever Python object was looked up.
///
/// Population is the responsibility of
/// [`super::register::register_rust_module`], which walks a
/// `syn::File` and registers each top-level item here under a
/// `ModuleId`. Re-registration of the same `(module_id, name)` is
/// last-writer-wins, mirroring upstream `module.__dict__[name] =
/// value` semantics: every top-level binding statement
/// unconditionally overwrites any prior entry. Within a single
/// walk Rust syntax does not allow duplicate top-level item names,
/// so the observable difference is across walks of the same
/// `module_id` (path-keyed re-walk after `Translation` rebuilds
/// against the same source file) — the second walk's bindings
/// supersede the first, matching `exec(source, dict)` /
/// `importlib.reload` semantics.
///
/// **Model boundary** (Codex audit, 2026-05-05): a path-keyed
/// re-walk is treated like `exec(source, dict)` / `reload`, NOT
/// like `import` returning a cached `sys.modules` entry. This is
/// the parity-correct framing for the *registration* path: every
/// top-level statement is an unconditional `__dict__[name] =
/// value` assignment in upstream Python, regardless of whether the
/// module is being imported for the first time or reloaded. A
/// caller who wants `sys.modules` cache-hit semantics ("don't
/// re-execute the body if the module is already loaded") must
/// gate the call on a prior `module_globals_lookup` of any
/// expected name and short-circuit themselves — the registry does
/// not implement that gate. Production callsites
/// (`from_rust_file_entry_point_with_source`) currently invoke
/// the walker once per `Translation`, so the re-walk window only
/// opens in tests and in cross-`Translation` workflows where the
/// caller is expected to know whether a refresh is desired.
/// Already-built `GraphFunc` instances observe the registry through
/// `GraphFunc::live_globals()` so a deliberate refresh propagates
/// through the same live-`__dict__` channel upstream uses for
/// post-import mutations.
static HOST_RUST_MODULE_GLOBALS: LazyLock<Mutex<HashMap<ModuleId, HashMap<String, ConstValue>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Look `name` up in `module_id`'s slice of the module-globals
/// registry. Returns the registered `ConstValue` if a prior
/// `register_rust_module(file)` walk for this id inserted it,
/// `None` otherwise.
///
/// Mirrors the Python expression `w_globals.value.get(varname)`
/// where `w_globals = Constant(func.__globals__)`
/// (`flowcontext.py:284`) — the per-module dict, not a shared one.
/// Callers fall back to `pyre_stdlib_lookup` (`__builtin__`
/// analogue) on `None`.
pub(super) fn module_globals_lookup(module_id: ModuleId, name: &str) -> Option<ConstValue> {
    HOST_RUST_MODULE_GLOBALS
        .lock()
        .expect("HOST_RUST_MODULE_GLOBALS Mutex poisoned")
        .get(&module_id)
        .and_then(|m| m.get(name).cloned())
}

/// Register `value` as the module-globals entry for `name` under
/// `module_id`. Last-writer-wins: re-registering the same
/// `(module_id, name)` with a different value clobbers the prior
/// entry, mirroring upstream `module.__dict__[name] = value`
/// (each top-level binding statement is an unconditional
/// assignment — `exec(source, dict)` / `importlib.reload`
/// semantics). Cross-`module_id` same-name registration is
/// independent: each id partitions its own bindings, mirroring
/// two Python modules whose `__dict__`s happen to share a key.
///
/// For class entries, the wrapped `HostObject::Class` is expected
/// to be the freshly-minted carrier from
/// [`super::register::register_rust_module`]; for `Item::Const`
/// entries, the value is a plain `ConstValue::Int` / `Bool` /
/// `UniStr` / `ByteStr` per upstream `find_global` returning
/// `const(value)` for any Python object.
/// Clone `module_id`'s slice of the registry into a fresh
/// `HashMap`, suitable for embedding in a `ConstValue::Dict` carrier.
/// Keys are byte-string `ConstValue`s matching the `__name__`-style
/// shape that [`crate::flowspace::model::GraphFunc::from_host_code`]
/// already handles at `model.rs:3331`.
///
/// Mirrors upstream `func.__globals__`: when `flowcontext.py:284
/// self.w_globals = Constant(func.__globals__)` runs, the Constant
/// wraps a snapshot of the module's `__dict__` at function-creation
/// time. The Rust counterpart snapshots the registry at
/// `build_host_metadata_parts` time so downstream consumers
/// (`flowcontext.py:847 w_globals.value[varname]`) read the same
/// content the body lowerer's `module_globals_lookup` would have
/// resolved against. Returns an empty map when `module_id` has no
/// registered entries (matches `func.__globals__ == {}` for a
/// function defined with no enclosing module bindings).
/// Public-via-`mod.rs` accessor (`module_globals_snapshot_for_id`)
/// for the `pub` field carrier `GraphFunc.module_globals_id` —
/// keeps `host_env` itself private while exposing the read path
/// through a single re-export.
pub(crate) fn module_globals_snapshot_pub(module_id: ModuleId) -> HashMap<ConstValue, ConstValue> {
    module_globals_snapshot(module_id)
}

pub(super) fn module_globals_snapshot(module_id: ModuleId) -> HashMap<ConstValue, ConstValue> {
    let map = HOST_RUST_MODULE_GLOBALS
        .lock()
        .expect("HOST_RUST_MODULE_GLOBALS Mutex poisoned");
    let Some(slice) = map.get(&module_id) else {
        return HashMap::new();
    };
    slice
        .iter()
        .map(|(name, value)| (ConstValue::byte_str(name.as_bytes()), value.clone()))
        .collect()
}

pub(super) fn register_module_global(module_id: ModuleId, name: &str, value: ConstValue) {
    let mut map = HOST_RUST_MODULE_GLOBALS
        .lock()
        .expect("HOST_RUST_MODULE_GLOBALS Mutex poisoned");
    map.entry(module_id)
        .or_default()
        .insert(name.to_string(), value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_some_err_result_option_are_registered_as_classes() {
        for name in ["Ok", "Some", "Err", "Result", "Option"] {
            let obj = pyre_stdlib_lookup(name)
                .unwrap_or_else(|| panic!("PYRE_STDLIB must register {name} as HostObject::Class"));
            assert!(
                obj.is_class(),
                "{name} must be a HostObject::Class, got {obj:?}",
            );
            assert_eq!(obj.qualname(), name);
        }
    }

    #[test]
    fn unregistered_names_return_none() {
        // pyre-specific identifiers (`StepResult`, `Instruction`) are
        // NOT in the closed-world registry — they reach the per-
        // Builder mint path via `Builder::resolve_path_constant`.
        assert!(pyre_stdlib_lookup("StepResult").is_none());
        assert!(pyre_stdlib_lookup("Instruction").is_none());
        // Snake-case helper names are never in the registry.
        assert!(pyre_stdlib_lookup("u32_as_i64").is_none());
        // `None` is special-cased outside this layer (it resolves to
        // `Constant(ConstValue::None)`, not a class).
        assert!(pyre_stdlib_lookup("None").is_none());
    }

    #[test]
    fn lookup_returns_same_identity_across_calls() {
        // Singleton invariant: two lookups of the same name share
        // pointer identity (mirrors Python's `__builtin__.Ok` being
        // the same object across every `LOAD_GLOBAL`). `HostObject`'s
        // `PartialEq` impl is `Arc::ptr_eq` (model.rs:208), so `==`
        // is pointer-equality.
        let a = pyre_stdlib_lookup("Ok").unwrap();
        let b = pyre_stdlib_lookup("Ok").unwrap();
        assert_eq!(a, b, "PYRE_STDLIB[Ok] must be a singleton across lookups");
    }

    #[test]
    fn mint_host_class_returns_same_identity_across_calls() {
        // `HOST_CLASS_MINTS` is process-global so two `mint_host_class`
        // calls for the same name (whether they originate from the
        // same `build_flow_from_rust` invocation or from different
        // ones) must return the same `HostObject`. Mirrors upstream
        // `func_globals[name]` (`flowcontext.py:847`) returning the
        // same Python object on every `LOAD_GLOBAL` regardless of
        // which graph is being built.
        let a = mint_host_class("ParityProbe_StepResult");
        let b = mint_host_class("ParityProbe_StepResult");
        assert_eq!(
            a, b,
            "HOST_CLASS_MINTS[ParityProbe_StepResult] must return the \
             same HostObject across calls (Arc::ptr_eq identity)",
        );
        assert!(a.is_class());
        assert_eq!(a.qualname(), "ParityProbe_StepResult");
    }

    #[test]
    fn mint_host_class_distinguishes_names() {
        // Distinct names mint distinct `HostObject` instances —
        // upstream `func_globals[a]` and `func_globals[b]` are
        // separate dict entries.
        let a = mint_host_class("ParityProbe_NameA");
        let b = mint_host_class("ParityProbe_NameB");
        assert_ne!(
            a, b,
            "distinct names must produce distinct HostObject identities",
        );
    }

    #[test]
    fn module_globals_lookup_returns_none_for_unregistered_names() {
        // Until `register_rust_module` registers a name, the
        // `func_globals` lookup misses and the resolver falls through
        // to `pyre_stdlib_lookup` per upstream
        // `flowcontext.py:845-854 find_global` ordering. Per-module
        // scoping (Issue 1.3): every test mints a fresh ModuleId so
        // observed registry state is isolated from cross-test pollution.
        let id = ModuleId::fresh();
        assert!(module_globals_lookup(id, "ParityProbe_FuncGlobals_unset").is_none());
    }

    #[test]
    fn module_globals_lookup_isolates_distinct_module_ids() {
        // Per-module scoping (Issue 1.3): two `ModuleId::fresh` ids
        // are independent partitions. Registering `name` under id1
        // does NOT make it visible from id2 — mirrors upstream
        // `flowcontext.py:284 self.w_globals = Constant(func.__globals__)`
        // where two distinct modules have independent
        // `__dict__`s and a name bound in one is invisible from the
        // other.
        let id1 = ModuleId::fresh();
        let id2 = ModuleId::fresh();
        assert_ne!(id1, id2, "ModuleId::fresh must produce distinct ids");
        register_module_global(id1, "ParityProbe_FuncGlobals_isolation", ConstValue::Int(1));
        assert_eq!(
            module_globals_lookup(id1, "ParityProbe_FuncGlobals_isolation"),
            Some(ConstValue::Int(1)),
            "id1's binding visible from id1",
        );
        assert!(
            module_globals_lookup(id2, "ParityProbe_FuncGlobals_isolation").is_none(),
            "id1's binding must NOT leak into id2 (cross-module isolation)",
        );
    }

    #[test]
    fn register_module_global_round_trips_and_preserves_identity() {
        // upstream `func_globals.value[varname]` returns the SAME
        // Python value on every lookup. The registry's contract is
        // the same: two `module_globals_lookup` calls of the same
        // (module_id, name) return `ConstValue` instances that
        // compare equal (HostObject equality is `Arc::ptr_eq`,
        // `model.rs:208`).
        let id = ModuleId::fresh();
        let host = HostObject::new_class("ParityProbe_FuncGlobals_round_trip", vec![]);
        let registered = ConstValue::HostObject(host.clone());
        register_module_global(id, "ParityProbe_FuncGlobals_round_trip", registered.clone());
        let a = module_globals_lookup(id, "ParityProbe_FuncGlobals_round_trip")
            .expect("registered name must look up");
        let b = module_globals_lookup(id, "ParityProbe_FuncGlobals_round_trip")
            .expect("registered name must look up");
        assert_eq!(a, b, "two lookups must yield identical ConstValue");
        assert_eq!(
            a, registered,
            "lookup result must match the registered value"
        );
    }

    #[test]
    fn register_module_global_is_last_writer_wins_within_module() {
        // upstream `module.__dict__[name] = value` is an
        // unconditional assignment: every top-level binding statement
        // (whether on first import, `exec(source, dict)`, or
        // `importlib.reload`) overwrites any prior entry. Within a
        // single `ModuleId` partition, the most recent registration
        // wins.
        let id = ModuleId::fresh();
        let first = HostObject::new_class("ParityProbe_FuncGlobals_lastwriter", vec![]);
        let second = HostObject::new_class("ParityProbe_FuncGlobals_lastwriter", vec![]);
        assert_ne!(
            first, second,
            "two fresh classes with the same qualname must NOT share identity \
             (sanity check — `new_class` mints a fresh Arc each call)",
        );
        register_module_global(
            id,
            "ParityProbe_FuncGlobals_lastwriter",
            ConstValue::HostObject(first.clone()),
        );
        register_module_global(
            id,
            "ParityProbe_FuncGlobals_lastwriter",
            ConstValue::HostObject(second.clone()),
        );
        let observed = module_globals_lookup(id, "ParityProbe_FuncGlobals_lastwriter").unwrap();
        assert_eq!(
            observed,
            ConstValue::HostObject(second),
            "last registration wins within a single ModuleId",
        );
        assert_ne!(
            observed,
            ConstValue::HostObject(first),
            "first registration must be clobbered by the second",
        );
    }

    #[test]
    fn register_module_global_accepts_const_value_variants() {
        // upstream `find_global` returns `const(value)` for any
        // Python object — int, bool, str, etc. The registry mirrors
        // that: any `ConstValue` variant round-trips intact. This
        // is what unblocks the `Item::Const` walker dispatch in
        // Slice O10.
        let id = ModuleId::fresh();
        register_module_global(id, "ParityProbe_FuncGlobals_const_int", ConstValue::Int(42));
        register_module_global(
            id,
            "ParityProbe_FuncGlobals_const_bool",
            ConstValue::Bool(true),
        );
        register_module_global(
            id,
            "ParityProbe_FuncGlobals_const_bytes",
            ConstValue::byte_str("hello"),
        );

        assert_eq!(
            module_globals_lookup(id, "ParityProbe_FuncGlobals_const_int").unwrap(),
            ConstValue::Int(42),
        );
        assert_eq!(
            module_globals_lookup(id, "ParityProbe_FuncGlobals_const_bool").unwrap(),
            ConstValue::Bool(true),
        );
        assert_eq!(
            module_globals_lookup(id, "ParityProbe_FuncGlobals_const_bytes").unwrap(),
            ConstValue::byte_str("hello"),
        );
    }
}
