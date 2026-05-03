//! Task #98 Slice 5 — end-to-end `jit_marker` pipeline integration.
//!
//! Upstream `jtransform.py:1658-1723 rewrite_op_jit_marker` dispatches to
//! `handle_jit_marker__jit_merge_point` / `handle_jit_marker__loop_header` /
//! `handle_jit_marker__can_enter_jit` when a graph contains
//! `SpaceOperation('jit_marker', [key, jitdriver, *args])`. Pyre keys on the
//! `CallTarget::Method` shape instead (front/ast.rs lowers
//! `pypyjitdriver.jit_merge_point(...)` as a method call), with the full
//! emission handled inside `Transformer::try_handle_jit_marker`
//! (`jit_codewriter/jtransform.rs:1786`).
//!
//! Unit tests in `jtransform.rs` cover the dispatch identity, the args
//! partition and the per-handler emission shape. This integration test
//! flows the new `OpKind::JitMergePoint` / `OpKind::LoopHeader` variants
//! through the full downstream pipeline —
//! `CodeWriter::transform_graph_to_jitcode` →  annotate → rtype →
//! `lower_indirect_calls` → jtransform → regalloc → flatten → liveness →
//! assembler — and asserts the final `SSARepr` carries both opnames along
//! with the canonical `promote_greens` preamble.
//!
//! A panic anywhere in the pipeline (e.g. a pass that forgets to handle
//! the new variants) fails the test with a concrete stack trace.

use std::sync::Arc;

use majit_translate::CallPath;
use majit_translate::call::CallControl;
use majit_translate::codewriter::CodeWriter;
use majit_translate::flatten::FlatOp;
use majit_translate::front::ast::build_function_graph_pub;
use majit_translate::jitcode::JitCode;
use majit_translate::jtransform::GraphTransformConfig;
use majit_translate::model::OpKind;
use syn::{Item, ItemFn};

const SOURCE: &str = r#"
struct PyPyJitDriver;

impl PyPyJitDriver {
    pub fn jit_merge_point(&self, next_instr: i64, frame: i64) {
        let _ = (next_instr, frame);
    }
    pub fn loop_header(&self) {}
    pub fn can_enter_jit(&self, next_instr: i64, frame: i64) {
        let _ = (next_instr, frame);
    }
}

pub fn portal(driver: &PyPyJitDriver, next_instr: i64, frame: i64) -> i64 {
    driver.jit_merge_point(next_instr, frame);
    driver.loop_header();
    driver.can_enter_jit(next_instr, frame);
    next_instr
}
"#;

fn find_fn<'a>(file: &'a syn::File, name: &str) -> Option<&'a ItemFn> {
    file.items.iter().find_map(|item| match item {
        Item::Fn(func) if func.sig.ident == name => Some(func),
        _ => None,
    })
}

#[test]
fn jit_marker_emissions_reach_ssarepr_through_full_pipeline() {
    let file = syn::parse_file(SOURCE).expect("SOURCE must parse");
    let portal_fn = find_fn(&file, "portal").expect("portal fn present");
    let sf = build_function_graph_pub(portal_fn).expect("portal fn must lower");
    let portal_path = CallPath::from_segments([sf.name.clone()]);

    let mut cc = CallControl::new();
    cc.register_function_graph(portal_path.clone(), sf.graph.clone());
    // jtransform.py:1691-1692 `assert self.portal_jd is not None` — the
    // handler branch runs only when the current graph's driver is set.
    // setup_jitdriver records the driver against `portal_path` and pushes
    // an entry into `jitdrivers_sd` whose `greens = ["next_instr"]` and
    // `reds = ["frame"]` drive the args partition inside
    // `try_handle_jit_marker`.
    cc.setup_jitdriver(
        portal_path.clone(),
        vec!["next_instr".into()],
        vec!["frame".into()],
        Vec::new(),
        vec!["i64".into()],
    );
    let jitcode: Arc<JitCode> = cc.get_jitcode(&portal_path);

    let mut cw = CodeWriter::new();
    let config = GraphTransformConfig::default();
    let idx = cc.finished_jitcodes_len();
    cw.transform_graph_to_jitcode(
        &sf.graph,
        &portal_path,
        &mut cc,
        &config,
        &jitcode,
        /* verbose = */ false,
        idx,
    );

    let body = jitcode.body();
    let ssarepr = body
        ._ssarepr
        .as_ref()
        .expect("assembler.py:49 stashes SSARepr on JitCode body");

    let mut merge_points = 0usize;
    let mut loop_headers = 0usize;
    for insn in &ssarepr.insns {
        if let FlatOp::Op(op) = insn {
            match &op.kind {
                OpKind::JitMergePoint {
                    jitdriver_index,
                    greens_i,
                    greens_r,
                    greens_f,
                    reds_i,
                    reds_r,
                    reds_f,
                } => {
                    assert_eq!(*jitdriver_index, 0, "single portal uses jitdriver index 0");
                    let total_greens = greens_i.len() + greens_r.len() + greens_f.len();
                    let total_reds = reds_i.len() + reds_r.len() + reds_f.len();
                    assert_eq!(
                        total_greens, 1,
                        "next_instr is the one green — receiver must not leak into greens \
                         (greens_i={greens_i:?} greens_r={greens_r:?} greens_f={greens_f:?})"
                    );
                    assert_eq!(
                        total_reds, 1,
                        "frame is the one red (reds_i={reds_i:?} reds_r={reds_r:?} reds_f={reds_f:?})"
                    );
                    // The exact (i,r,f) bucket split depends on annotator /
                    // rtyper concrete-type propagation for function
                    // parameters. Today pyre's legacy annotator does not
                    // always thread `i64` parameter types through
                    // `concrete_types`, so `split_args_by_kind` falls back
                    // to `'r'` — matching upstream's "GcRef or Unknown"
                    // branch at `jtransform.py::make_three_lists`. The
                    // parity contract locked down here is the cardinality
                    // (one green, one red, receiver stripped), not the
                    // bucketing. Task #107 (`rpbc.py` PBCRepr
                    // specialization) tightens the concrete-type surface;
                    // once it lands the expected split becomes i=1/r=0.
                    eprintln!(
                        "[task-98 slice 5] greens: i={} r={} f={}; reds: i={} r={} f={}",
                        greens_i.len(),
                        greens_r.len(),
                        greens_f.len(),
                        reds_i.len(),
                        reds_r.len(),
                        reds_f.len(),
                    );
                    merge_points += 1;
                }
                OpKind::LoopHeader { jitdriver_index } => {
                    assert_eq!(*jitdriver_index, 0);
                    loop_headers += 1;
                }
                _ => {}
            }
        }
    }
    assert_eq!(
        merge_points, 1,
        "jit_merge_point call must emit exactly one OpKind::JitMergePoint"
    );
    assert_eq!(
        loop_headers, 2,
        "loop_header + can_enter_jit aliases to two OpKind::LoopHeader emissions \
         (jtransform.py:1723 `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`)"
    );

    // jtransform.py:526 `promote_greens` prepends `-live-` + `{kind}_guard_value`
    // pairs per green. With greens=["next_instr"] the merge point should be
    // preceded by a Live marker and an IntGuardValue against the green
    // variable.
    let merge_pos = ssarepr
        .insns
        .iter()
        .position(|insn| matches!(insn, FlatOp::Op(op) if matches!(op.kind, OpKind::JitMergePoint { .. })))
        .expect("JitMergePoint must be present in SSARepr");
    assert!(
        merge_pos >= 2,
        "promote_greens must prepend at least (-live-, {{kind}}_guard_value) before jit_merge_point"
    );
    let has_guard_value = (0..merge_pos).any(|i| match &ssarepr.insns[i] {
        FlatOp::Op(op) => matches!(op.kind, OpKind::GuardValue { .. }),
        _ => false,
    });
    assert!(
        has_guard_value,
        "promote_greens should emit an {{kind}}_guard_value for the single green"
    );

    // The assembled body must be non-empty — proves flatten / regalloc /
    // liveness / assembler all coped with the new OpKind variants.
    assert!(
        !body.code.is_empty(),
        "JitCode body must contain assembled bytes after full pipeline"
    );

    // ------------------------------------------------------------------
    // Canonical payload check: verify each JitMergePoint / LoopHeader
    // SSA op produces the upstream-shaped bytecode slice. Without this,
    // the generic fallback (which only serialises op_value_refs()) would
    // silently pass the assembler without emitting jdindex + typed lists,
    // misaligning blackhole.rs:2012 / pyjitpl/dispatch.rs:1089 cursors.
    //
    // Shape per `majit-metainterp/src/jitcode/assembler.rs:692-729` +
    // blackhole.py:1066 `@arguments("self", "i", "I", "R", "F", "I",
    // "R", "F")`:
    //   jit_merge_point: [opcode][jdindex:u8]
    //                    [gi_len][gi_regs..][gr_len][gr_regs..]
    //                    [gf_len][gf_regs..][ri_len][ri_regs..]
    //                    [rr_len][rr_regs..][rf_len][rf_regs..]
    //   loop_header:     [opcode][jdindex:u8]
    // ------------------------------------------------------------------
    let insns_pos = ssarepr
        .insns_pos
        .as_ref()
        .expect("assembler must populate ssarepr.insns_pos");
    let code = &body.code;

    // Helper: resolve a `/i`-encoded byte (register-file index, upstream
    // `@arguments("i")`) to the constant value it points at — same model
    // as `init_register_files_from_runtime_jitcode` (`blackhole.rs:1054`)
    // and `MIFrame::copy_constants` (`pyjitpl/frame.rs:196`). Used to
    // validate the pool-routed jdindex byte without running the runtime.
    let num_regs_i = body.c_num_regs_i as usize;
    let resolve_int_const = |reg_byte: u8| -> i64 {
        let idx = reg_byte as usize;
        assert!(
            idx >= num_regs_i,
            "jdindex byte {idx} points at a real register, not the constants pool \
             (num_regs_i={num_regs_i})"
        );
        let pool_idx = idx - num_regs_i;
        assert!(
            pool_idx < body.constants_i.len(),
            "jdindex byte {idx} points past constants_i pool \
             (pool_idx={pool_idx}, len={})",
            body.constants_i.len()
        );
        body.constants_i[pool_idx]
    };
    let resolve_jdindex_byte = |jdindex_byte: u8, expected: i64| -> i64 {
        if jdindex_byte as usize >= num_regs_i {
            resolve_int_const(jdindex_byte)
        } else {
            assert_eq!(
                (jdindex_byte as i8) as i64,
                expected,
                "jdindex byte {jdindex_byte} must equal jitdriver_index {expected} in `c` form"
            );
            (jdindex_byte as i8) as i64
        }
    };

    let mut canonical_merge_points = 0usize;
    let mut canonical_loop_headers = 0usize;
    for (idx, insn) in ssarepr.insns.iter().enumerate() {
        let FlatOp::Op(op) = insn else { continue };
        match &op.kind {
            OpKind::JitMergePoint {
                jitdriver_index,
                greens_i,
                greens_r,
                greens_f,
                reds_i,
                reds_r,
                reds_f,
            } => {
                let pos = insns_pos[idx];
                // Skip opcode byte; jdindex is byte 1.
                assert!(
                    pos + 1 < code.len(),
                    "jit_merge_point at {pos} truncated before jdindex"
                );
                assert_eq!(
                    resolve_jdindex_byte(code[pos + 1], *jitdriver_index as i64) as usize,
                    *jitdriver_index,
                    "jit_merge_point jdindex byte must resolve to OpKind.jitdriver_index"
                );
                let mut cursor = pos + 2;
                let expected_counts = [
                    greens_i.len(),
                    greens_r.len(),
                    greens_f.len(),
                    reds_i.len(),
                    reds_r.len(),
                    reds_f.len(),
                ];
                for (list_idx, expected) in expected_counts.iter().enumerate() {
                    assert!(
                        cursor < code.len(),
                        "jit_merge_point list {list_idx} length byte past end of code"
                    );
                    let len = code[cursor] as usize;
                    assert_eq!(
                        len, *expected,
                        "jit_merge_point list {list_idx} length mismatch (bytecode={len} vs SSA={expected})"
                    );
                    cursor += 1 + len;
                    assert!(
                        cursor <= code.len(),
                        "jit_merge_point list {list_idx} body overruns code"
                    );
                }
                canonical_merge_points += 1;
            }
            OpKind::LoopHeader { jitdriver_index } => {
                let pos = insns_pos[idx];
                assert!(
                    pos + 1 < code.len(),
                    "loop_header at {pos} truncated before jdindex"
                );
                assert_eq!(
                    resolve_int_const(code[pos + 1]) as usize,
                    *jitdriver_index,
                    "loop_header jdindex byte (pool index) must resolve \
                     to OpKind.jitdriver_index"
                );
                canonical_loop_headers += 1;
            }
            _ => {}
        }
    }
    assert_eq!(
        canonical_merge_points, merge_points,
        "bytecode must preserve every JitMergePoint"
    );
    assert_eq!(
        canonical_loop_headers, loop_headers,
        "bytecode must preserve every LoopHeader"
    );

    eprintln!(
        "[task-98 slice 5] portal ssarepr.insns={} merge_points={} loop_headers={}",
        ssarepr.insns.len(),
        merge_points,
        loop_headers
    );
}
