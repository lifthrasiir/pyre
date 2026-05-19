//! Phase F acceptance: the embedded pyre-interpreter JitCode registry
//! materialises every `opcode_*` handler graph and the trait impls those
//! handlers reach via `direct_call`.
//!
//! RPython parity oracle:
//! - `rpython/jit/codewriter/codewriter.py:74 make_jitcodes` produces one
//!   `JitCode` per graph it encounters (portals + everything reachable).
//! - `rpython/jit/codewriter/call.py:87 self.jitcodes` is the graph-keyed
//!   dict that `all_jitcodes()` mirrors via `AllJitCodes::by_path`.
//! - `rpython/jit/codewriter/codewriter.py:80
//!   assert self.all_jitcodes[i].index == i` is enforced inside
//!   `CallControl::collect_jitcodes_in_alloc_order`, which feeds
//!   `AllJitCodes::in_order`. The test re-checks the surface invariant.
//!
//! The test is READ-ONLY against pyre-jit; no pyre-jit binary links
//! against `generated` yet (Phase G), so baseline `./pyre/check.py`
//! 14/14+14/14 is unaffected.

use majit_translate::generated::all_jitcodes;

#[test]
fn all_jitcodes_covers_every_opcode_handler() {
    let reg = all_jitcodes();

    // Every `opcode_*` freestanding handler in pyopcode.rs must appear
    // in the registry. The Phase E.0.1 harness confirmed the transform
    // pipeline covers 28+; Phase F must preserve that count.
    assert!(
        reg.len() >= 28,
        "expected >=28 JitCodes in the registry, got {}",
        reg.len()
    );
}

#[test]
fn all_jitcodes_indices_match_alloc_order() {
    // Mirrors RPython `codewriter.py:80` invariant: in_order[i].index == i.
    for (i, jc) in all_jitcodes().in_order.iter().enumerate() {
        assert_eq!(
            jc.try_index(),
            Some(i),
            "in_order[{}].index = {:?} (expected {})",
            i,
            jc.try_index(),
            i
        );
    }
}

#[test]
fn all_jitcodes_is_memoised() {
    // `OnceLock`-backed: two calls must return the same `&'static`.
    let a: *const _ = all_jitcodes();
    let b: *const _ = all_jitcodes();
    assert!(
        std::ptr::eq(a, b),
        "all_jitcodes() returned different pointers on successive calls"
    );
}

#[test]
fn all_jitcodes_entries_have_bodies() {
    // Every emitted JitCode must have a non-empty code body —
    // `transform_graph_to_jitcode` calls `jitcode.set_body(body)` with
    // a populated `body.code`. An empty body would mean the assembler
    // ran without emitting anything, which never happens on a non-trivial
    // handler graph.
    for jc in &all_jitcodes().in_order {
        assert!(
            !jc.code.is_empty(),
            "JitCode `{}` has an empty code body",
            jc.name
        );
    }
}

#[test]
fn all_jitcodes_by_path_matches_in_order() {
    let reg = all_jitcodes();
    // Every entry of `in_order` must be reachable through `by_path`
    // (same Arc, so pointer-equal). `by_path` may contain strictly
    // more entries if the registry later tracks graph-only records
    // without JitCodes, but today the two should agree on the JitCode
    // set.
    for jc in &reg.in_order {
        let mut found = false;
        for other in reg.by_path.values() {
            if std::sync::Arc::ptr_eq(jc, other) {
                found = true;
                break;
            }
        }
        assert!(
            found,
            "in_order entry `{}` is not reachable through by_path",
            jc.name
        );
    }
}

#[test]
fn all_jitcodes_registry_contains_inherent_impl_methods() {
    // Regression guard against the earlier draft of `generated.rs`,
    // which registered only `opcode_*` free functions + trait impl
    // methods and therefore missed every `impl PyFrame { ... }` method
    // body. RPython's `translator.graphs` contains inherent methods
    // (they are ordinary reachable graphs), so the pyre registry must
    // match. Concrete witness: `PyFrame::check_exc_match` in
    // `pyre/pyre-interpreter/src/eval.rs` calls `self.pop()` /
    // `self.peek()` / `self.push()` (inherent helpers defined on
    // `PyFrame` in `pyre/pyre-interpreter/src/pyframe.rs`); all four
    // graphs must appear in the registry.
    let reg = all_jitcodes();
    let names: std::collections::HashSet<&str> =
        reg.by_path.values().map(|jc| jc.name.as_str()).collect();
    for required in ["push", "pop", "peek", "check_exc_match"] {
        assert!(
            names.contains(required),
            "inherent impl method `{}` is missing from the JitCode registry \
             — `generated::build()` lost the full-context pipeline output",
            required
        );
    }
}

/// Loop-free `opcode_*` helpers defined in
/// `pyre-interpreter/src/shared_opcode.rs:50-172` — these pass upstream
/// `rpython/jit/codewriter/policy.py:53-62 look_inside_graph`
/// (no backedges → inlinable). Task #101 wired the BFS reachability so
/// all of these land in the JitCode registry.
const SHARED_OPCODE_HELPERS_INLINABLE: &[&str] = &[
    "opcode_make_function",
    "opcode_build_list",
    "opcode_build_tuple",
    "opcode_build_map",
    "opcode_store_subscr",
    "opcode_list_append",
    "opcode_load_attr",
    "opcode_store_attr",
];

/// Helpers with loops that upstream policy rejects unless
/// `_jit_unroll_safe_` is declared. pyre mirrors the rejection
/// verbatim: these bodies stay out of `candidate_graphs` and the
/// portal reaches them through `residual_call_*` at runtime.
/// `opcode_call` has the phi/diamond from its `match nargs` default
/// arm; `opcode_unpack_sequence` has a literal `for item in items`
/// at shared_opcode.rs:157. Absence from the registry IS the parity
/// signal.
const SHARED_OPCODE_HELPERS_LOOP_REJECTED: &[&str] = &["opcode_call", "opcode_unpack_sequence"];

#[test]
fn all_jitcodes_registry_contains_loop_free_shared_opcode_helpers() {
    // Task #101 fix anchor: every loop-free `shared_opcode.rs` helper
    // must reach `AllJitCodes::by_path`. Before the fix (lib.rs:459-480
    // parallel registration under `[<Trait>, <method>]`), `PyFrame`'s
    // empty `impl OpcodeStepExecutor for PyFrame {}` block failed to
    // surface the trait's default-method bodies, so the portal's
    // `FunctionPath{[OpcodeStepExecutor, make_function]}` call lowered
    // to `CallResidual` with a symbolic fnaddr — never reaching
    // `candidate_graphs`, never allocating a jitcode, never following
    // the direct_call into `opcode_make_function`.
    //
    // Upstream parity anchor: `rpython/annotator/classdesc.py:749
    // ClassDesc.lookup(name)` walks the basedesc chain until finding
    // a class whose `classdict` contains the name. Rust trait defaults
    // are the "base-class methods" for impls that don't override; the
    // fix binds them to a direct `[<Trait>, <method>]` key so
    // `target_to_path` resolves identically to the upstream MRO walk.
    let reg = all_jitcodes();
    let names: std::collections::HashSet<&str> =
        reg.by_path.values().map(|jc| jc.name.as_str()).collect();
    let missing: Vec<&&str> = SHARED_OPCODE_HELPERS_INLINABLE
        .iter()
        .filter(|n| !names.contains(*n))
        .collect();
    assert!(
        missing.is_empty(),
        "Loop-free shared_opcode helpers missing from JitCode registry: \
         {missing:?}. Upstream `codewriter.py:74 make_jitcodes` would \
         have transformed these as inline graphs. Check that \
         `lib.rs::analyze_pipeline_from_parsed` still calls \
         `call_control.register_function_graph(direct_path, graph)` for \
         every `impl_info` whose `for_type` starts with `<default \
         methods of `."
    );
}

#[test]
fn loopy_shared_opcode_helpers_rejected_by_policy() {
    // Upstream parity: `rpython/jit/codewriter/policy.py:56
    // look_inside_graph` returns False when the graph has a backedge
    // without `_jit_unroll_safe_`. `opcode_call` and
    // `opcode_unpack_sequence` both contain loops; pyre's
    // `policy.rs::find_backedges` + `DefaultJitPolicy::look_inside_graph`
    // reject them verbatim. Their absence from the registry is the
    // expected parity outcome — the portal reaches them through
    // `residual_call_*` instead.
    let reg = all_jitcodes();
    let names: std::collections::HashSet<&str> =
        reg.by_path.values().map(|jc| jc.name.as_str()).collect();
    for n in SHARED_OPCODE_HELPERS_LOOP_REJECTED {
        assert!(
            !names.contains(n),
            "Loop-carrying shared_opcode helper `{n}` unexpectedly \
             reached the JitCode registry. Upstream \
             `rpython/jit/codewriter/policy.py:56 look_inside_graph` \
             rejects looped graphs without `_jit_unroll_safe_`. If pyre \
             now admits this helper, verify that either (a) the helper \
             gained an `unroll_safe` hint deliberately, or (b) \
             `find_backedges` lost its loop detection — the latter is a \
             parity regression."
        );
    }
}

#[test]
fn all_jitcodes_source_manifest_covers_shared_opcode_helpers() {
    // Acceptance invariant that Step 6 DOES satisfy: each helper's
    // graph is present in `program.functions` once `shared_opcode.rs`
    // joins the `PYRE_JIT_GRAPH_SOURCES` manifest. The sibling test
    // above then pinpoints the BFS resolution gap without this test
    // masking the manifest-level contribution.
    use majit_translate::front::build_semantic_program_from_parsed_files;
    use majit_translate::parse_source;
    let parsed = [
        parse_source(include_str!(
            "../../../pyre/pyre-interpreter/src/pyopcode.rs"
        )),
        parse_source(include_str!("../../../pyre/pyre-interpreter/src/eval.rs")),
        parse_source(include_str!(
            "../../../pyre/pyre-interpreter/src/pyframe.rs"
        )),
        parse_source(include_str!(
            "../../../pyre/pyre-interpreter/src/shared_opcode.rs"
        )),
    ];
    let program = build_semantic_program_from_parsed_files(&parsed)
        .expect("pyre source must lower without FlowingError");
    let names: std::collections::HashSet<&str> =
        program.functions.iter().map(|f| f.name.as_str()).collect();
    // All 10 helpers (both the 8 inlinable + the 2 loop-rejected) must
    // exist in `program.functions` — the Semantic Program layer is
    // source-driven and has no loop-policy filter.
    let all_ten: Vec<&str> = SHARED_OPCODE_HELPERS_INLINABLE
        .iter()
        .chain(SHARED_OPCODE_HELPERS_LOOP_REJECTED.iter())
        .copied()
        .collect();
    let missing: Vec<&&str> = all_ten.iter().filter(|n| !names.contains(*n)).collect();
    assert!(
        missing.is_empty(),
        "shared_opcode helpers missing from SemanticProgram functions: \
         {:?}. `PYRE_JIT_GRAPH_SOURCES` must include `shared_opcode.rs` \
         so `build_semantic_program_from_parsed_files` ingests their \
         Rust source; their `direct_call`s from trait methods will \
         otherwise have no resolvable target in `function_graphs`.",
        missing
    );
}
