//! End-to-end analysis pipeline.
//!
//! **LEGACY.** Majit-local driver that sequences the ad-hoc
//! `annotate → resolve_types → flatten_with_types` chain. Not the
//! same as RPython's `translator/driver.py` (TranslationDriver +
//! SimpleTaskEngine); re-identification was considered and rejected.
//! The rtyper-pipeline cutover (roadmap Phase 8) routes pyre compiles
//! through `majit-rtyper` directly and bypasses this file; this file
//! is deleted at roadmap commit P8.11.
//!
//! RPython-orthodox chain (when fully ported):
//!   flowspace → annotator → rtyper → jtransform → flatten
//!
//! This module provides a single entry point that runs all passes
//! in sequence on a SemanticProgram.

use crate::call::CallControl;
use crate::flatten;
use crate::front::SemanticFunction;
use crate::jtransform::rewrite_graph_with_callcontrol;
use crate::pipeline::{PipelineConfig, PipelineResult, ProgramPipelineResult};
use crate::translate_legacy::annotator::annrpython::annotate;
use crate::translate_legacy::rtyper::rtyper::resolve_types;

/// Run the full analysis pipeline on a single function.
///
/// RPython equivalent: translate a single function through
/// flowspace → annotator → rtyper → jtransform → flatten.
pub fn analyze_function(func: &SemanticFunction, config: &PipelineConfig) -> PipelineResult {
    let graph = &func.graph;
    let original_blocks = graph.blocks.len();

    // Pass 1: Annotation (RPython annotator)
    let annotations = annotate(graph);
    let annotations_count = annotations.types.len();

    // Pass 2: Type resolution (RPython rtyper)
    let mut types = resolve_types(graph, &annotations);
    let concrete_types_count = types.concrete_types.len();

    // Pass 2b: rtyper-equivalent indirect_call lowering. RPython's rtyper
    // (rpbc.py:199-217) always emits `indirect_call(funcptr, *args,
    // c_graphs)` before jtransform sees the graph. Pyre's canonical
    // `codewriter::transform_graph_to_jitcode` runs this pass before
    // `rewrite_graph`; the legacy driver must do the same so callers that
    // consume `&dyn Trait` receivers (e.g. `pyre-jit/src/eval.rs`'s
    // `allocate_struct(typedescr: &dyn majit_ir::SizeDescr)`) do not trip
    // the `assert_no_indirect_call_targets` debug invariant inside
    // `rewrite_graph`. Legacy analyze is not plugged into CallControl, so
    // pass an empty one — `lower_indirect_calls` treats the resulting
    // empty `all_impls_for_indirect` family as "unknown" (graphs = None),
    // which is the conservative RPython-orthodox fallback.
    let mut legacy_callcontrol = CallControl::new();
    let mut graph_owned = graph.clone();
    crate::translator::rtyper::rpbc::lower_indirect_calls(
        &mut graph_owned,
        &mut types,
        &legacy_callcontrol,
    );

    // Pass 3: JIT transform (RPython jtransform) — thread the same empty
    // CallControl so `lower_indirect_call_op` has access to `getcalldescr`
    // / `guess_call_kind` / `graphs_from`. With no registered candidates
    // the op resolves to `CallKind::Residual`, matching upstream's
    // conservative fallback for `indirect_call` with unknown family.
    // Also thread `types` so jtransform's `get_value_kind` sees each
    // operand's `concretetype` — RPython's rtyper resolves
    // `Variable.concretetype` once at typing time and every later pass
    // (including jtransform) reads it directly.  Without `with_type_state`
    // the legacy snapshot path defaulted every operand to `'r'`, which
    // forced the kind-coercion arms in `jtransform` to manufacture
    // `cast_ptr_to_int` ops on graphs that the canonical pipeline
    // already processed correctly.
    let transform_result = {
        let mut transformer = crate::jtransform::Transformer::new(&config.transform)
            .with_callcontrol(&mut legacy_callcontrol)
            .with_type_state(&types);
        transformer.transform(&graph_owned)
    };
    let vable_rewrites = transform_result.vable_rewrites;
    let calls_classified = transform_result.calls_classified;
    let transform_notes = transform_result.notes.clone();
    let rewritten_types = crate::translate_legacy::rtyper::rtyper::resolve_rewritten_types(
        &types,
        &transform_result.graph,
        &annotations,
        &transform_result.synth_kinds,
    );

    // Pass 4: Flatten with type info (RPython flatten + regalloc)
    let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&rewritten_types);
    let regallocs =
        crate::regalloc::perform_all_register_allocations(&transform_result.graph, &value_kinds);
    let flattened =
        flatten::flatten_with_types(&transform_result.graph, &rewritten_types, &regallocs);

    PipelineResult {
        name: func.name.clone(),
        original_blocks,
        annotations_count,
        concrete_types_count,
        vable_rewrites,
        calls_classified,
        transform_notes,
        flattened,
    }
}

/// Run the full pipeline on all functions in a program.
pub fn analyze_program(
    program: &crate::front::SemanticProgram,
    config: &PipelineConfig,
) -> ProgramPipelineResult {
    let mut functions = Vec::new();
    let mut total_blocks = 0;
    let mut total_ops = 0;
    let mut total_vable_rewrites = 0;

    for func in &program.functions {
        let result = analyze_function(func, config);
        total_blocks += result.original_blocks;
        total_ops += result.flattened.insns.len();
        total_vable_rewrites += result.vable_rewrites;
        functions.push(result);
    }

    ProgramPipelineResult {
        functions,
        opcode_dispatch: Vec::new(),
        jitcodes: Vec::new(),
        jitcodes_by_path: std::collections::HashMap::new(),
        insns: std::collections::HashMap::new(),
        descrs: Vec::new(),
        total_blocks,
        total_ops,
        total_vable_rewrites,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::OpcodeDispatchSelector;
    use crate::front;
    use crate::jitcode::JitCode;
    use crate::jtransform::GraphTransformConfig;
    use crate::opcode_dispatch::PipelineOpcodeArm;
    use crate::pipeline::{PipelineConfig, PipelineResult, ProgramPipelineResult};
    use crate::{
        flatten::{FlatOp, SSARepr},
        flowspace::model::ConstValue,
        model::LinkArg,
    };

    #[test]
    fn pipeline_e2e_simple_function() {
        let parsed = crate::parse::parse_source(
            r#"
            fn add(a: i64, b: i64) -> i64 {
                a + b
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig::default();
        let result = analyze_program(&program, &config);

        assert_eq!(result.functions.len(), 1);
        assert_eq!(result.functions[0].name, "add");
        assert!(result.functions[0].annotations_count > 0);
        assert!(result.functions[0].flattened.insns.len() > 0);
    }

    #[test]
    fn pipeline_e2e_with_virtualizable() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Frame { next_instr: usize, locals_w: Vec<i64> }
            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig {
            transform: GraphTransformConfig {
                vable_fields: vec![crate::jtransform::VirtualizableFieldDescriptor::new(
                    "next_instr",
                    Some("Frame".into()),
                    0,
                )],
                vable_arrays: vec![crate::jtransform::VirtualizableFieldDescriptor::new(
                    "locals_w",
                    Some("Frame".into()),
                    0,
                )],
                ..Default::default()
            },
            ..Default::default()
        };
        let result = analyze_program(&program, &config);
        assert_eq!(result.functions.len(), 1);
        assert!(
            result.total_vable_rewrites > 0,
            "should rewrite next_instr to VableFieldRead, notes: {:?}",
            result.functions[0].transform_notes
        );
        // Should rewrite BOTH the field read (next_instr) AND the array read (locals_w[idx])
        let notes_str = format!("{:?}", result.functions[0].transform_notes);
        assert!(
            notes_str.contains("VableFieldRead"),
            "should contain VableFieldRead rewrite note"
        );
        assert!(
            notes_str.contains("VableArrayRead"),
            "should contain VableArrayRead rewrite note, got: {notes_str}"
        );
    }

    #[test]
    fn pipeline_e2e_with_control_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            fn fib(n: i64) -> i64 {
                if n <= 1 {
                    return n;
                }
                let a = n - 1;
                let b = n - 2;
                a + b
            }
        "#,
        );
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        let config = PipelineConfig::default();
        let result = analyze_program(&program, &config);

        let func = &result.functions[0];
        assert!(
            func.original_blocks >= 4,
            "if/else should create >=4 blocks, got {}",
            func.original_blocks
        );
        // Flattened should have jumps
        let has_jump = func.flattened.insns.iter().any(|op| {
            matches!(
                op,
                crate::flatten::FlatOp::Jump(_) | crate::flatten::FlatOp::GotoIfNot { .. }
            )
        });
        assert!(has_jump, "flattened fib should have conditional jumps");
    }

    #[test]
    fn serialized_program_pipeline_skips_flattened_ssa_consts() {
        let flattened = SSARepr {
            name: "consts".into(),
            insns: vec![FlatOp::RefReturn(LinkArg::Const(ConstValue::byte_str(
                "hello",
            )))],
            num_values: 0,
            num_blocks: 1,
            value_kinds: Default::default(),
            insns_pos: None,
        };
        let program = ProgramPipelineResult {
            functions: vec![PipelineResult {
                name: "consts".into(),
                original_blocks: 1,
                annotations_count: 0,
                concrete_types_count: 0,
                vable_rewrites: 0,
                calls_classified: 0,
                transform_notes: Vec::new(),
                flattened: flattened.clone(),
            }],
            opcode_dispatch: vec![PipelineOpcodeArm {
                arm_id: 7,
                selector: OpcodeDispatchSelector::Unsupported,
                entry_jitcode_index: Some(0),
                flattened: Some(flattened),
            }],
            jitcodes: vec![Arc::new(JitCode::new("consts"))],
            jitcodes_by_path: std::collections::HashMap::new(),
            insns: std::collections::HashMap::new(),
            descrs: Vec::new(),
            total_blocks: 1,
            total_ops: 1,
            total_vable_rewrites: 0,
        };

        let json = serde_json::to_string(&program).expect("program pipeline should serialize");
        assert!(
            !json.contains("flattened"),
            "serialized artifact should not persist debug SSA payloads"
        );
        serde_json::to_string(&program.opcode_dispatch)
            .expect("opcode dispatch artifact should serialize without SSARepr");
    }
}
