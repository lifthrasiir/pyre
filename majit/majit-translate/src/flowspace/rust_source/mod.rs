//! `flowspace/rust_source/` — Rust AST → `flowspace::FunctionGraph`
//! adapter.
//!
//! pyre-interpreter's source is Rust, not Python, so upstream
//! `rpython/flowspace/objspace.py:38-53 build_flow(func)` cannot consume
//! it directly. This module is the Position-2 adaptation (see
//! `.claude/plans/annotator-monomorphization-tier1-abstract-lake.md` —
//! "Position 2 — Rust AST adapter into unchanged flowspace"): walk a
//! `syn::ItemFn` and emit the same `flowspace::FunctionGraph` shape
//! `build_flow(GraphFunc)` would have emitted for the equivalent
//! Python bytecode.
//!
//! The downstream consumers (`annotator/*`, `classdesc`, rtyper) run
//! unchanged — the adapter is the only place the "input side is Rust"
//! divergence lives.
//!
//! ## Scope of the M2.5a skeleton
//!
//! This is the initial `build_flow_from_rust` entry point per the
//! plan's M2.5a bullet. It covers the "no control flow" core:
//!
//! - Function signature: non-generic, non-async, non-unsafe,
//!   identifier-only parameters, no `self` receiver.
//! - Body: sequence of `let <ident> = <expr>;` statements followed by
//!   an expression-tail `return`.
//! - Expressions: integer / bool literals, identifier paths (local
//!   reference), 16 `BinOp`s matching upstream `operation.py:475
//!   add_operator` entries.
//!
//! Anything outside that scope (control flow, method calls, struct
//! literals, closures, …) rejects via [`AdapterError::Unsupported`].
//! Control flow lands in M2.5b; method calls + trait dispatch in
//! M2.5c; struct/enum/tuple literals in M2.5d; full
//! `execute_opcode_step` in M2.5e.
//!
//! ## Output shape
//!
//! A `FunctionGraph` with exactly one non-terminal block:
//!
//! ```text
//!   startblock([p_0, p_1, …, p_n])
//!     op_0
//!     op_1
//!     …
//!     op_m
//!     → returnblock(tail_value)
//! ```
//!
//! Matching upstream `test_model.py:13-43` + `test_ssa.py:55-88`
//! construction idioms for straight-line functions.
//!
//! ## PRE-EXISTING-ADAPTATIONs (do not accumulate new ones)
//!
//! Three structural divergences exist from upstream
//! `rpython/flowspace/*.py`. Each is load-bearing for the "no Python
//! bytecode on the input side" constraint and is listed here with a
//! concrete convergence path. Re-evaluate on every `/parity` pass.
//!
//! 1. **By-name locals `HashMap<String, Hlvalue>`** in
//!    [`build_flow::Builder`] (`build_flow.rs:189-199`) replaces
//!    upstream `FrameState(locals_w, stack, last_exception, blocklist,
//!    next_offset)` (`rpython/flowspace/framestate.py:18`). Upstream
//!    tracks slot-indexed locals; the Rust adapter has no Python
//!    bytecode stack and no slot indices, so it uses a name-keyed map.
//!    The `for`-loop iterator survives joins as a synthetic
//!    `#for_iter_{depth}` name rather than a stack slot
//!    (`build_flow.rs:1266, flowcontext.py:782`).
//!    *Convergence path*: port `FrameState` + slot-indexed locals
//!    through the adapter once all Rust-AST constructs the adapter
//!    emits have an upstream slot-space equivalent. Multi-session.
//!
//! 2. **Direct post-simplify graph emission**
//!    (`build_flow.rs:1189, :1272, :641`) replaces upstream's
//!    `flowcontext.py:124 pendingblocks` + `SpamBlock`/`EggBlock`
//!    creation loop and `rpython/translator/simplify.py:52` empty-block
//!    folding. The adapter's `branch_block_with_inputargs` machinery
//!    produces an already-simplified shape.
//!    *Convergence path*: implement `SpamBlock`/`EggBlock` abstractions
//!    + pendingblocks loop inside the adapter, then let upstream
//!    `simplify.py` collapse the empty blocks. Multi-session.
//!
//! 3. **`HLOperation.eval` constfold side wired (2026-05-05);
//!    `guessbool` early-resolution wired (2026-05-05).** Pure-op
//!    emission routes through [`build_flow::Builder::record_pure_op`]
//!    which mirrors `flowspace/flowcontext.rs:1734 record_pure_op`:
//!    every recognised opname runs through `HLOperation::constfold`
//!    (`operation.py:92, :120`) before `emit_op`, so all-foldable args
//!    collapse to a `Constant` at flow-build time exactly as upstream
//!    `op.<name>(*args).eval(self)` would. Control-flow lowering then
//!    consults the post-fold `Constant(Bool)` and follows upstream
//!    `flowcontext.py:341 guessbool`'s `if isinstance(w_condition,
//!    Constant): return w_condition.value` short-circuit — only the
//!    chosen arm flows; no 2-exit fork materializes. Wired in
//!    `lower_if`, `lower_if_without_else`, and `lower_while`.
//!    One pure-op callsite stays on raw `emit_op` as a real
//!    PRE-EXISTING-ADAPTATION:
//!    - `build_flow.rs:651` cascade `getattr` in
//!      `Builder::resolve_path_constant`: keeps an explicit
//!      `host.class_get` / `is_host_class_minted` path because
//!      `const_runtime_getattr` raises `FlowingError` on minted-class
//!      misses, while the adapter's mint-on-demand stand-in needs the
//!      "treat as opaque, fall through to raw `getattr`" semantics
//!      pending the M2.5g extern-Rust-helper walker. Convergence
//!      lands when minting retires.
//!
//!    A second raw `emit_op` callsite at `build_flow.rs:2658`
//!    (`same_as` in `lower_match`) is NOT a PRE-EXISTING-ADAPTATION
//!    — `same_as` is a synthetic pseudo-op in upstream too,
//!    referenced only at `rpython/flowspace/model.py:634`
//!    (`raising_op.opname not in ("keepalive", "cast_pointer",
//!    "same_as")`) and `:707 summary` (excluded opname); it never
//!    appears in `operation.py`'s `add_operator` registry. Raw
//!    `emit_op` IS the orthodox shape; the surrounding code also
//!    needs the raw `Variable` (not an `Hlvalue` wrapper) to derive
//!    the `#match_scrutinee_{id}` slot.

pub mod build_flow;
mod host_env;
pub mod register;

pub use build_flow::{AdapterError, build_flow_from_rust, build_flow_from_rust_in_module};
pub use host_env::ModuleId;
pub use register::{
    build_host_function_from_rust, build_host_function_from_rust_file,
    build_host_function_metadata_from_rust, register_rust_module, register_rust_module_at,
};

/// Live snapshot of the module-globals partition keyed on
/// `module_id`. Issue 2.1 (2026-05-05): `GraphFunc.module_globals_id`
/// uses this to mirror upstream `flowcontext.py:284 self.w_globals =
/// Constant(func.__globals__)` — `func.__globals__` is a *live*
/// reference, so introspection paths
/// (`HostObject::class_get("__globals__")` at `model.rs:1228`,
/// `getattr(func, "__globals__")` at `model.rs:928`) re-snapshot
/// the registry at access time rather than reading the snapshot
/// taken at GraphFunc construction.
pub fn module_globals_snapshot_for_id(
    module_id: ModuleId,
) -> std::collections::HashMap<
    crate::flowspace::model::ConstValue,
    crate::flowspace::model::ConstValue,
> {
    host_env::module_globals_snapshot_pub(module_id)
}
