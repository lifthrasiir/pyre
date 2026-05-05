//! Probe: run the Rust-AST adapter on the real
//! `pyre-interpreter::execute_opcode_step` portal and capture the
//! current rejection point.
//!
//! Serves the `M2.5e — pass the real pyopcode.rs through the adapter`
//! milestone from the annotator-monomorphization plan (see
//! `~/.claude/plans/annotator-monomorphization-tier1-abstract-lake.md`).
//! The plan's acceptance criterion for M2.5e is that the adapter
//! produces a complete `FunctionGraph` for `execute_opcode_step<E>`,
//! with every opcode branch represented and every method call carrying
//! a resolvable receiver classdef.
//!
//! Today the adapter stops at the first `AdapterError::Unsupported`
//! that comes out of walking the function body. This test pins the
//! exact category of that stop so regressions surface early, and so
//! future adapter extensions have a visible "does it get further now?"
//! signal: every M2.5d/e slice that lands should either move the
//! rejection point deeper into the body or eliminate it.
//!
//! RPython parity note: upstream `flowspace/objspace.py:38-53
//! build_flow(func)` consumes Python bytecode end-to-end. The Rust-AST
//! adapter is the Position-2 adaptation for pyre's Rust-source
//! interpreter; the rejection surface is inherent to the
//! "implementation incomplete" state, not a parity gap on the
//! flowspace side.

use majit_translate::flowspace::rust_source::{
    AdapterError, build_flow_from_rust, build_flow_from_rust_in_module, register_rust_module,
};
use syn::{File, Item};

const PYOPCODE_SRC: &str = include_str!("../../../pyre/pyre-interpreter/src/pyopcode.rs");

fn parse_pyopcode() -> File {
    syn::parse_file(PYOPCODE_SRC).expect("pyopcode.rs must parse")
}

fn find_fn<'a>(file: &'a File, name: &str) -> &'a syn::ItemFn {
    file.items
        .iter()
        .find_map(|item| match item {
            Item::Fn(func) if func.sig.ident == name => Some(func),
            _ => None,
        })
        .unwrap_or_else(|| panic!("expected `fn {name}` in pyopcode.rs"))
}

#[test]
fn adapter_rejects_execute_opcode_step_on_composite_match_pattern() {
    let file = parse_pyopcode();
    let func = find_fn(&file, "execute_opcode_step");

    // The function parses and runs through `validate_signature`
    // successfully — generics + where-clause on `E: …OpcodeHandler`
    // traits are accepted per `build_flow.rs:134-143` (the annotator's
    // `FunctionDesc.specialize` is what monomorphizes `E` into a
    // classdef, so the adapter itself can admit the generic shape).
    //
    // ### Rejection timeline
    //
    // - Before M2.5d slice 1 (or-pattern splitting): the first match
    //   arm
    //   `Instruction::ExtendedArg | Instruction::Resume {..} | ...`
    //   rejected at the outer `Pat::Or` classifier
    //   (`build_flow.rs:classify_pattern` — or-pattern arm).
    // - After M2.5d slice 1: or-pattern flattens, surfacing the
    //   first composite / variant sub-pattern. `Instruction::ExtendedArg`
    //   is a unit enum variant (`Pat::Path`), rejected today via the
    //   `_` catch-all of `classify_pattern` with
    //   "match arm pattern not in M2.5b subset".
    // - After M2.5d slice 2c (`Pat::Path` accepted): the first
    //   rejection moves to `Pat::Struct {..}` (e.g.
    //   `Instruction::Resume {..}`) with
    //   "composite pattern (enum/tuple/struct — lands in M2.5d)".
    // - After M2.5d slice 2d (rest-only `Pat::Struct {..}` and
    //   `Pat::TupleStruct(..)` accepted): the first rejection moves
    //   to `Pat::Struct { field, .. }` (a struct variant whose match
    //   arm binds at least one field, e.g.
    //   `Instruction::LoadConst { consti }`) with
    //   "match arm struct-variant pattern with field bindings (…) —
    //   field-binding extraction lands in M2.5d slice 2e".
    // - After M2.5d slice 2e (struct-variant named-Ident field
    //   bindings accepted): the cascade lowers every match-arm
    //   pattern in `execute_opcode_step`. Lowering then progresses
    //   INTO the arm bodies and rejects on the first un-resolved
    //   identifier — the `Result::Ok(...)` constructor reference at
    //   `Ok(StepResult::Continue)`. Surfaces as
    //   `AdapterError::UnboundLocal { name: "Ok" }` because the
    //   adapter has no host-environment registry for the standard
    //   library `Result` constructors. Resolving these is a separate
    //   M2.5g intake task.
    // - The M2.5e Result/Option wrapper-transparency slice trio
    //   (Ok/Some/None value-position rewrite, qualified-path
    //   expression-position sentinel, terminator-position Err raise
    //   edge) landed in `e7e168c29f7` and was REVERTED 2026-05-03
    //   per Codex parity audit. Each rewrite was a NEW-DEVIATION:
    //   value-position `Ok(x)` collapse erased the
    //   `simple_call(<host>, x)` op upstream emits; qualified-path
    //   ByteStr sentinel produced a graph that did not match
    //   `getattr(…, attr)` cascade upstream emits; terminator Err
    //   raise rewrite was an incomplete `exc_from_raise` shape
    //   missing the isinstance check, optional class instantiation
    //   and `ll_assert_not_none` from
    //   `flowcontext.py:632-636 exc_from_raise`. The orthodox
    //   replacement plan is at
    //   `~/.claude/plans/m2_5e_orthodox_host_env_resolution.md`.
    // - 2026-05-03 — Slice O1+O2 of the orthodox replacement landed:
    //   `Builder::resolve_path_constant` mirrors upstream
    //   `flowcontext.py:856 LOAD_GLOBAL` + `:861 LOAD_ATTR` chain.
    //   Closed-world `host_env::PYRE_STDLIB` registry resolves bare
    //   `Ok` / `Some` / `Err` / `Result` / `Option` to
    //   `Constant(HostObject(<class>))`; bare `None` resolves to
    //   `Constant(ConstValue::None)`; multi-segment paths emit a
    //   `getattr` cascade per `operation.py:618 getattr`, with the
    //   leftmost segment minted on demand and cached on the Builder
    //   so two cascade steps that name the same class share
    //   identity. Probe rejection advanced from
    //   `UnboundLocal { name: "Ok" }` to
    //   `UnboundLocal { name: "u32_as_i64" }` — the
    //   pyre-as-cast-removal-epic Slice 3 helper that
    //   `Bookkeeper::register_rust_function` (M2.5g) resolves.
    // - 2026-05-04 — Slices O3 + O4 + O5 of the orthodox replacement
    //   landed (O5 attempted+reverted+re-landed same day with the
    //   fork-elision NEW-DEVIATION addressed):
    //     * O3: `lower_match_variant_cascade` isinstance arg2 routes
    //       through `Builder::resolve_path_constant`. Each cascade
    //       step block emits its own `getattr` op per non-leftmost
    //       segment of the variant path, then `isinstance(scrutinee,
    //       <leaf>)` per `operation.py:449`; identity sharing across
    //       cascade steps (and across graphs) via the process-global
    //       `host_env::HOST_CLASS_MINTS` registry. Replaces the
    //       prior `Constant(ByteStr(joined_path))` sentinel.
    //     * O4: `lower_value_boundary` collapses `Ok(x)` / `Some(x)`
    //       / `None` AT BOUNDARY positions only (function/arm tail,
    //       `return` operand). Documented PRE-EXISTING-ADAPTATION;
    //       value-position calls keep `simple_call(<host>, …)` per
    //       O1+O2.
    //     * O5 (full fork, PARITY): `emit_err_raise_boundary` lowers
    //       boundary-position `Err(e)` to the upstream
    //       `flowcontext.py:600-636 exc_from_raise` op sequence with
    //       the 2-exit `guessbool(isinstance(arg, type))` fork at
    //       `flowcontext.py:610` preserved. True arm:
    //       `w_value = simple_call(evalue)` (`flowcontext.py:614`,
    //       instantiate). False arm:
    //       `w_value = ll_assert_not_none(evalue)`
    //       (`flowcontext.py:632-634`, instance shape; the TypeError
    //       sub-arm is constant-folded out by upstream's
    //       `guessbool(is_(w_arg2, const(None)))` since w_arg2 is
    //       `const(None)` from `RAISE_VARARGS(1)`). Both arms
    //       converge on a join block that emits `type(w_value)` and
    //       Links `[etype, w_value]` to `graph.exceptblock` per
    //       `flowcontext.py:1259 Raise.nomoreblocks`. The prior
    //       reverted attempt (`16ebcd497b0`) elided the fork on the
    //       unenforced "Err always carries an instance" claim;
    //       re-landed via `f296dfdc490` after the orthodox port.
    //   Probe rejection unchanged: still
    //   `UnboundLocal { name: "u32_as_i64" }`. The cascade walks
    //   INTO arm bodies; the first body-level free identifier is the
    //   pyre-as-cast-removal helper, which resolves through the
    //   M2.5g `Bookkeeper::register_rust_function` intake (a
    //   separate epic).
    // - 2026-05-04 — Slice O7 (module-globals walker) landed:
    //   `register_rust_module(&syn::File)` walks `pyopcode.rs` once
    //   and registered every top-level `Item::Fn` into the per-process
    //   `host_env::HOST_RUST_MODULE_FUNCS` registry as a
    //   `HostObject::UserFunction` whose `GraphFunc.prebuilt_flow_graph`
    //   stays `None`. With the helpers resolving, the adapter walked
    //   deeper into `execute_opcode_step` and surfaced a closure
    //   expression as the next un-roadmapped construct.
    // - 2026-05-05 — Issue 1.2 PRE-EXISTING-ADAPTATION: Slice O7's
    //   `Item::Fn` registration is REVERTED. The deferred-body
    //   `HostObject` had no path back to the Rust-AST adapter
    //   (`FunctionDesc.buildgraph` at `description.py:140` only
    //   knows how to call `build_flow(GraphFunc)` against
    //   `func.__code__.co_code`, but pyre's `HostCode` for an
    //   Item::Fn carries empty bytecode), so registered sibling
    //   fns would supply empty bodies at lowering time. The
    //   walker now skips Item::Fn entirely; the entry-point fn is
    //   located via `file.items.iter().find_map(...)` in
    //   `build_host_function_from_rust_file` instead. **Probe
    //   rejection rolls back** from `Unsupported(closure)` to the
    //   pre-O7 `UnboundLocal { name: "u32_as_i64" }` state, since
    //   the cast-removal helpers no longer resolve through the
    //   registry. Convergence path: M2.5g side-table walker that
    //   pairs the metadata HostObject with a stored `&syn::ItemFn`
    //   for replay, OR M2.5f-style eager prebuilt-graph
    //   construction at walker time.
    //

    // Current rejection state (post-Issue-1.2): walker registers
    // only `Item::Enum` / `Item::Struct` / literal `Item::Const`.
    // The cast-removal Slice 3 helpers (`u32_as_i64` /
    // `u32_as_usize` / `op_arg_as_usize` / `raise_kind_as_usize`)
    // are top-level `Item::Fn` and therefore unresolved at call
    // sites — adapter rejects with `UnboundLocal { name }` on the
    // first cascade-driven helper reference.
    //
    // The probe pins this state strictly:
    //
    // - `UnboundLocal { name }` in the cast-removal helper set is
    //   the expected post-Issue-1.2 state.
    // - `Unsupported(closure)` would mean Item::Fn registration
    //   was silently re-introduced (the walker registered fn names
    //   again, masquerading as callables with empty bodies).
    // - Any earlier `Unsupported` reason (or-pattern, variant-
    //   path, composite, field-bindings, qualified-path, numeric-
    //   cast) means an O3/O4/O5/M2.5d slice silently regressed.
    //
    // Either way the change is a deliberate, audited progression
    // and not a silent slip.
    // Issue 1.3 (per-module scoping): `register_rust_module` mints
    // a fresh `ModuleId` and returns it; `build_flow_from_rust_in_module`
    // threads the same id through body lowering so the cascade's
    // `LOAD_GLOBAL` resolutions hit the just-walked partition.
    // (Pre-Issue-1.3, the registry was process-global, so a separate
    // `build_flow_from_rust(func)` call after `register_rust_module(&file)`
    // saw the same registry — that path no longer exists; the id
    // must be threaded explicitly.)
    let file_for_walker = parse_pyopcode();
    let module_id = register_rust_module(&file_for_walker);
    let err = build_flow_from_rust_in_module(func, module_id)
        .err()
        .expect(
            "adapter still has un-roadmapped constructs to walk past — see post-Issue-1.2 timeline",
        );
    match err {
        AdapterError::UnboundLocal { name } => {
            const CAST_REMOVAL_SLICE3_HELPERS: &[&str] = &[
                "u32_as_i64",
                "u32_as_usize",
                "op_arg_as_usize",
                "raise_kind_as_usize",
            ];
            assert!(
                CAST_REMOVAL_SLICE3_HELPERS.contains(&name.as_str()),
                "post-Issue-1.2 state expects an unresolved cast-removal helper from \
                 {CAST_REMOVAL_SLICE3_HELPERS:?}, got {name:?}. If a different name \
                 surfaces, the cascade walked past the cast-removal helper layer — \
                 re-pin this list explicitly with the new helper name."
            );
            eprintln!("adapter rejection at post-Issue-1.2 probe: UnboundLocal({name}) — expected");
        }
        AdapterError::Unsupported { reason } => {
            // If this branch hits, Item::Fn registration regressed
            // — the closure rejection is downstream of helper
            // resolution and should not be reachable without it.
            panic!(
                "PARITY REGRESSION: probe expected `UnboundLocal(<cast-removal helper>)` \
                 (post-Issue-1.2 state) but adapter rejected with `Unsupported({reason})`. \
                 Item::Fn registration in `register_rust_module` may have been \
                 re-introduced — see Issue 1.2 PRE-EXISTING-ADAPTATION."
            );
        }
        other => panic!("expected AdapterError::UnboundLocal, got {other:?}"),
    }
}

#[test]
fn adapter_rejects_execute_opcode_step_without_walker_at_cast_removal_helper() {
    // Sister oracle: WITHOUT the Slice O7 walker call, the rejection
    // state is the pre-O7 `UnboundLocal { name }` at one of the
    // cast-removal Slice 3 helpers (`u32_as_i64` / `u32_as_usize` /
    // `op_arg_as_usize` / `raise_kind_as_usize`). Documented in
    // `pyre/pyre-interpreter/src/pyopcode.rs:1302..1336`.
    //
    // Per-module scoping (Issue 1.3, 2026-05-05): `build_flow_from_rust`
    // mints a fresh `ModuleId` internally, so this test's lookup
    // partition is isolated from any other test's
    // `register_rust_module` walk. The rejection is therefore
    // strictly pre-walker — sibling tests' registry writes live
    // under different ids and cannot leak in. The pre-Issue-1.3
    // process-global pollution caveat (which forced this test to
    // accept a post-walker `Unsupported(closure)` outcome too) no
    // longer applies.
    let file = parse_pyopcode();
    let func = find_fn(&file, "execute_opcode_step");
    let err = build_flow_from_rust(func)
        .err()
        .expect("adapter still has un-roadmapped constructs to walk past");
    match err {
        AdapterError::UnboundLocal { name } => {
            const CAST_REMOVAL_SLICE3_HELPERS: &[&str] = &[
                "u32_as_i64",
                "u32_as_usize",
                "op_arg_as_usize",
                "raise_kind_as_usize",
            ];
            assert!(
                CAST_REMOVAL_SLICE3_HELPERS.contains(&name.as_str()),
                "without walker: expected an unresolved cast-removal Slice 3 \
                 helper from {CAST_REMOVAL_SLICE3_HELPERS:?}, got {name:?}",
            );
        }
        other => panic!(
            "post-Issue-1.3 per-module scoping: a fresh ModuleId means no cross-test \
             pollution from sibling walks, so the rejection MUST be \
             `UnboundLocal(<cast-removal helper>)`. Got {other:?}"
        ),
    }
}

#[test]
fn adapter_accepts_execute_opcode_step_signature_shape() {
    // Sanity partition: the signature shape alone is fine (generic
    // `<E: Trait>`, where-clause, plain-identifier params). If this
    // ever fails, the regression is in `validate_signature` +
    // `collect_params`, not in the body walker.
    //
    // Exercised indirectly via a synthetic fixture that copies only
    // the outer signature shape so the test stays independent of the
    // large body's content churn.
    let synthetic: syn::ItemFn = syn::parse_str(
        "fn execute_opcode_step<E>(
             executor: E,
             code: i64,
             instruction: i64,
             op_arg: i64,
             next_instr: i64,
         ) -> i64 where E: Handler { 0 }",
    )
    .expect("synthetic fixture parses");
    let g = build_flow_from_rust(&synthetic).expect("signature shape must be accepted");
    majit_translate::flowspace::model::checkgraph(&g);
}
