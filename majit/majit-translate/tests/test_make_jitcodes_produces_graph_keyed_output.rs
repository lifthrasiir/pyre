//! Phase E acceptance anchor (lucky-growing-puzzle).
//!
//! Test name matches the plan's stated acceptance command
//! (`cargo test -p majit-translate test_make_jitcodes_produces_graph_keyed_output`).
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/codewriter.py:74-89 make_jitcodes` — the
//!   main driver that pops from `callcontrol.enum_pending_graphs()`
//!   and produces one `JitCode` per graph.
//! - `rpython/jit/codewriter/codewriter.py:33-72 transform_graph_to_jitcode` —
//!   the 4-step pipeline (jtransform → regalloc → flatten → assemble)
//!   that turns one FunctionGraph into one JitCode.
//! - `rpython/jit/codewriter/call.py:87 self.jitcodes = {}` — the
//!   graph-keyed dict that `AllJitCodes::by_path` mirrors.
//! - `rpython/jit/codewriter/call.py:88 self.all_jitcodes = []` — the
//!   alloc-order list that `AllJitCodes::in_order` mirrors, with the
//!   `all_jitcodes[i].index == i` invariant from `codewriter.py:80`.
//!
//! ## What this test anchors
//!
//! Phase E's structural claim is that `make_jitcodes` output is
//! **graph-keyed** — one `JitCode` per `CallPath`, with no
//! `Instruction`-variant tables, no opcode-to-fragment lookups, no
//! anything Python bytecode-shaped. The plan's explicit prohibition is:
//! > 산출물은 `{graph: JitCode}` 만 — variant-keyed map 도입 금지.
//!
//! `test_phase_f_all_jitcodes.rs` already covers most of the
//! behavioural acceptance; this file adds the **structural** anchor
//! the plan specifies by name, with focused assertions that detect
//! any future drift toward variant-keyed output schemas.

use majit_translate::{CallPath, generated::all_jitcodes, jitcode::JitCode};
use std::collections::HashSet;
use std::sync::Arc;

#[test]
fn test_make_jitcodes_produces_graph_keyed_output() {
    // Phase E contract: `AllJitCodes::by_path` is keyed by `CallPath`
    // (graph identity), matching upstream `call.py:87 self.jitcodes`.
    // The `by_path` field's type ensures at compile time that no
    // Instruction-variant key can ever land here — this test only has
    // to verify that the live registry respects the contract without
    // structural surprises.
    let reg = all_jitcodes();

    // Invariant 1: every `in_order` entry is also reachable through
    // `by_path`. Together with invariant 2 they pin the 1:1 mapping
    // between CallPath and JitCode (upstream `codewriter.py:80-81`).
    let in_order_ptrs: HashSet<usize> = reg
        .in_order
        .iter()
        .map(|jc: &Arc<JitCode>| Arc::as_ptr(jc) as usize)
        .collect();
    let by_path_ptrs: HashSet<usize> = reg
        .by_path
        .values()
        .map(|jc| Arc::as_ptr(jc) as usize)
        .collect();
    assert!(
        in_order_ptrs.is_subset(&by_path_ptrs),
        "every JitCode in `in_order` must be reachable via `by_path` \
         (RPython `call.py:87-88` parity)"
    );

    // Invariant 2: each JitCode Arc appears in `by_path` under exactly
    // one CallPath. Duplicate entries would mean a graph was registered
    // under two different paths, which breaks upstream's `{graph:
    // JitCode}` identity contract at `call.py:157-165`.
    let mut seen: HashSet<usize> = HashSet::new();
    for (path, jc) in &reg.by_path {
        let ptr = Arc::as_ptr(jc) as usize;
        assert!(
            seen.insert(ptr),
            "JitCode `{}` appears in `by_path` under multiple CallPath \
             keys — last seen at {path:?}",
            jc.name
        );
    }

    // Invariant 3: registry size reflects the Phase A + B expected
    // closure. Phase A confirmed 28 `opcode_*` handlers lower to
    // FunctionGraphs. Phase B confirmed every PyFrame trait method has
    // a graph. Phase E must produce at least that many JitCodes (plus
    // whatever shared_opcode / inherent method closure BFS pulls in).
    //
    // Lower floor is intentional — the upper bound drifts as more
    // helpers get pulled into the closure by Phase D's BFS. A count
    // regression is the signal; any growth is fine.
    assert!(
        reg.in_order.len() >= 28,
        "expected at least 28 JitCodes (Phase A opcode_* floor); got {}",
        reg.in_order.len()
    );

    // Invariant 4: registry contains the portal (`execute_opcode_step`).
    // RPython `call.py:145-148 grab_initial_jitcodes` seeds it first.
    let execute_opcode_step_path = CallPath::from_segments(["execute_opcode_step"]);
    assert!(
        reg.by_path.contains_key(&execute_opcode_step_path),
        "portal `execute_opcode_step` missing from `by_path` — \
         `grab_initial_jitcodes` (call.py:145-148) did not seed it"
    );

    // Invariant 5: registry is not keyed by Instruction. This is a
    // structural assertion: `by_path` is `HashMap<CallPath, Arc<JitCode>>`
    // at the type level, so we can't even construct a variant-keyed
    // view. The assertion below verifies the grep floor agrees — a
    // compile-time type enforces no variant-key lookup ever compiles.
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut offenders: Vec<String> = Vec::new();
    walk_rs_files(&root, &mut |path, contents| {
        for (i, line) in contents.lines().enumerate() {
            let trimmed = line.trim_start();
            // Skip doc comments and regular comments — they may
            // legitimately reference the negative form.
            if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*") {
                continue;
            }
            if line.contains("HashMap<Instruction") {
                offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
            }
        }
    });
    assert!(
        offenders.is_empty(),
        "variant-keyed output forbidden — RPython `call.py:87 self.jitcodes` \
         is graph-keyed only:\n{}",
        offenders.join("\n")
    );
}

fn walk_rs_files<F: FnMut(&std::path::Path, &str)>(dir: &std::path::Path, f: &mut F) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_rs_files(&path, f);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(contents) = std::fs::read_to_string(&path) {
                f(&path, &contents);
            }
        }
    }
}
