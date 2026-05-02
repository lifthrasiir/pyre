//! Annotation propagation pass.
//!
//! **LEGACY.** Ad-hoc `ValueType`-flat-enum annotator.
//! Line-by-line port of `annotator/annrpython.py:RPythonAnnotator` is
//! landing at `majit-annotator/src/annrpython.rs` (roadmap Phase 5).
//! This file is deleted at roadmap commit P8.11.
//!
//! Propagates ValueType annotations through the graph by analyzing
//! each op's inputs and computing the output type. Iterates to
//! fixpoint when Block.inputargs (Phi nodes) need widening.

use crate::flowspace::model::ConstValue;
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::model::{FunctionGraph, Link, LinkArg, OpKind, ValueId, ValueType};

/// Run annotation propagation to fixpoint.
///
/// RPython equivalent: `RPythonAnnotator.complete()` — processes all
/// blocks until no annotation changes.
pub fn annotate(graph: &FunctionGraph) -> AnnotationState {
    let mut state = AnnotationState::new();

    // Seed the terminal pseudo-block inputargs. RPython `flowspace/
    // model.py:17-18` `returnblock = Block([return_var])` and
    // `exceptblock = Block([etype, evalue])` carry implicit types that
    // later Link propagation confirms but never introduces for Links
    // that never reach the block.  Raising-only or void functions can
    // leave these args untyped, which later drops them from
    // `build_value_kinds` and trips the assembler's
    // `lookup_reg_with_kind` panic.  `rpython/jit/codewriter/flatten.py:
    // 169-172` assumes `etype: Int, evalue: Ref` and `return_var:
    // <result_type>` unconditionally; pyre mirrors that by seeding the
    // annotator state up front.
    let exceptblock = graph.block(graph.exceptblock);
    if let Some(&etype) = exceptblock.inputargs.first() {
        state.set(etype, ValueType::Int);
    }
    if let Some(&evalue) = exceptblock.inputargs.get(1) {
        state.set(evalue, ValueType::Ref);
    }
    // `returnblock.inputargs[0]` must not be pre-seeded to `Ref`.
    // Doing so collapses a real `Float`/`Int` return into
    // `union_type(Ref, Float|Int) == Unknown`, which the legacy rtyper
    // then backfills to `GcRef`.  Seed only the pyre-only synthetic
    // `return None` placeholder values here; normal non-void returns
    // are inferred from the incoming Link args.
    let returnblock = graph.block(graph.returnblock);
    if let Some(&ret) = returnblock.inputargs.first() {
        for block in &graph.blocks {
            for link in &block.exits {
                if link.target != graph.returnblock {
                    continue;
                }
                if let Some(LinkArg::Value(src)) = link.args.first()
                    && is_synthetic_return_void_value(graph, *src)
                {
                    state.set(*src, ValueType::Void);
                    state.set(ret, ValueType::Void);
                }
            }
        }
    }

    // Process all blocks (simple single-pass for acyclic; loops need fixpoint)
    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 20;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for block in &graph.blocks {
            // Propagate annotations through ops in this block
            for op in &block.operations {
                if let Some(result) = op.result {
                    let inferred = infer_op_type(&op.kind, &state);
                    let current = state.get(result).clone();
                    let merged = union_type(&current, &inferred);
                    if merged != current {
                        state.set(result, merged);
                        changed = true;
                    }
                }
            }

            // Cross-block propagation: Link args → target inputargs, per
            // upstream `rpython/annotator/annrpython.rs` fold pass which
            // iterates `for link in block.exits` and unions each
            // `link.args[i]` annotation into `link.target.inputargs[i]`.
            for link in &block.exits {
                let link_changed = if link_is_raise_like(link) {
                    follow_raise_link(&mut state, graph, link)
                } else {
                    follow_link(&mut state, graph, link)
                };
                changed |= link_changed;
            }
        }
    }

    state
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn follow_link(state: &mut AnnotationState, graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        changed |= merge_value_type(state, *dst, link_arg_type(state, src));
    }
    changed
}

fn follow_raise_link(state: &mut AnnotationState, graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    if let Some(LinkArg::Value(value)) = link.last_exc_value.as_ref() {
        changed |= merge_value_type(state, *value, ValueType::Ref);
    }
    if let Some(LinkArg::Value(value)) = link.last_exception.as_ref() {
        changed |= merge_value_type(state, *value, ValueType::Int);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ValueType::Int
        } else if Some(src) == link.last_exc_value.as_ref() {
            ValueType::Ref
        } else {
            link_arg_type(state, src)
        };
        changed |= merge_value_type(state, *dst, src_ty);
    }
    changed
}

fn link_arg_type(state: &AnnotationState, src: &LinkArg) -> ValueType {
    match src {
        LinkArg::Value(src) => state.get(*src).clone(),
        LinkArg::Const(value) => const_value_type(value),
    }
}

fn merge_value_type(state: &mut AnnotationState, dst: ValueId, src_ty: ValueType) -> bool {
    let current = state.get(dst).clone();
    let merged = union_type(&current, &src_ty);
    if merged != current {
        state.set(dst, merged);
        true
    } else {
        false
    }
}

fn const_value_type(value: &ConstValue) -> ValueType {
    match value {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::LLAddress(_) => ValueType::Int,
        ConstValue::Float(_) => ValueType::Float,
        ConstValue::Placeholder => ValueType::Unknown,
        ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::None
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_) => ValueType::Ref,
    }
}

fn is_synthetic_return_void_value(graph: &FunctionGraph, value: ValueId) -> bool {
    for block in &graph.blocks {
        if block.inputargs.contains(&value) {
            return false;
        }
        if block.operations.iter().any(|op| op.result == Some(value)) {
            return false;
        }
    }
    true
}

/// Infer the output type of an operation from its inputs.
///
/// RPython equivalent: annotator dispatch (e.g., `annotate_int_add`
/// returns `SomeInteger()`).
fn infer_op_type(kind: &OpKind, state: &AnnotationState) -> ValueType {
    match kind {
        OpKind::Input { ty, .. } => ty.clone(),
        OpKind::ConstInt(_) => ValueType::Int,
        OpKind::ConstFloat(_) => ValueType::Float,
        OpKind::FieldRead { ty, .. } => ty.clone(),
        OpKind::FieldWrite { .. } => ValueType::Void,
        OpKind::ArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::ArrayWrite { .. } => ValueType::Void,
        OpKind::InteriorFieldRead { item_ty, .. } => item_ty.clone(),
        OpKind::InteriorFieldWrite { .. } => ValueType::Void,
        OpKind::Call {
            result_ty,
            target,
            args,
            ..
        } => {
            if result_ty != &ValueType::Unknown {
                return result_ty.clone();
            }
            infer_call_result_type(target, args, state)
        }
        OpKind::GuardTrue { .. } | OpKind::GuardFalse { .. } => ValueType::Void,
        OpKind::VableFieldRead { ty, .. } => ty.clone(),
        OpKind::VableFieldWrite { .. } => ValueType::Void,
        OpKind::VableArrayRead { item_ty, .. } => item_ty.clone(),
        OpKind::VableArrayWrite { .. } => ValueType::Void,
        OpKind::UnaryOp {
            op,
            operand,
            result_ty,
        } if op == "same_as" => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                state
                    .types
                    .get(operand)
                    .cloned()
                    .unwrap_or(ValueType::Unknown)
            }
        }
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            if result_ty != &ValueType::Unknown {
                result_ty.clone()
            } else {
                ValueType::Int // Arithmetic defaults to Int
            }
        }
        OpKind::VableForce { .. }
        | OpKind::Live
        | OpKind::GuardValue { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::RecordKnownResult { .. }
        // jtransform.py:901-903 — `record_quasiimmut_field` has no result.
        | OpKind::RecordQuasiImmutField { .. }
        // jtransform.py:1707,1718 — jit_merge_point / loop_header have no
        // result; upstream emits them with `op1 = SpaceOperation(..., None)`.
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. } => ValueType::Void,
        OpKind::CurrentTraceLength => ValueType::Int,
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } => ValueType::Int,
        // RPython: vtable entry is a `Ptr(FuncType)` address.
        OpKind::VtableMethodPtr { .. } => ValueType::Int,
        OpKind::IndirectCall { result_ty, .. } => result_ty.clone(),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. }
        | OpKind::ConditionalCallValue { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::ConditionalCall { .. } => ValueType::Void,
        OpKind::Abort { .. } => ValueType::Unknown,
    }
}

fn kind_char_to_value_type(kind: char) -> ValueType {
    match kind {
        'i' => ValueType::Int,
        'r' => ValueType::Ref,
        'f' => ValueType::Float,
        'v' => ValueType::Void,
        _ => ValueType::Unknown,
    }
}

fn infer_call_result_type(
    target: &crate::model::CallTarget,
    _args: &[ValueId],
    _state: &AnnotationState,
) -> ValueType {
    if crate::call::is_int_arithmetic_target(target) {
        return ValueType::Int;
    }
    ValueType::Unknown
}

/// Merge two annotations (RPython `unionof()`).
///
/// Returns the wider type. Unknown absorbs everything (top of lattice).
fn union_type(a: &ValueType, b: &ValueType) -> ValueType {
    if a == b {
        return a.clone();
    }
    match (a, b) {
        (ValueType::Unknown, other) | (other, ValueType::Unknown) => {
            if other == &ValueType::Unknown {
                ValueType::Unknown
            } else {
                other.clone()
            }
        }
        _ => ValueType::Unknown, // Conflicting types → Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        CallTarget, ExitSwitch, FunctionGraph, Link, OpKind, ValueType, exception_exitcase,
    };

    #[test]
    fn annotates_const_int() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(entry, Some(v));

        let state = annotate(&graph);
        assert_eq!(state.get(v), &ValueType::Int);
    }

    #[test]
    fn annotates_field_read_type() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("x", None),
                    ty: ValueType::Int,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v));

        let state = annotate(&graph);
        assert_eq!(state.get(v), &ValueType::Int);
    }

    #[test]
    fn annotates_call_with_int_args() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let b = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["w_int_add"]),
                    args: vec![a, b],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let state = annotate(&graph);
        assert_eq!(state.get(a), &ValueType::Int);
        assert_eq!(state.get(b), &ValueType::Int);
        assert_eq!(state.get(result), &ValueType::Int);
    }

    #[test]
    fn annotates_path_like_int_helper_call() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let a = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let b = graph.push_op(entry, OpKind::ConstInt(2), true).unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::Call {
                    target: CallTarget::function_path(["crate", "math", "w_int_add"]),
                    args: vec![a, b],
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let state = annotate(&graph);
        assert_eq!(state.get(result), &ValueType::Int);
    }

    #[test]
    fn propagates_across_blocks_via_phi() {
        // Test cross-block annotation propagation through Link args → inputargs
        let mut graph = FunctionGraph::new("phi_test");
        let entry = graph.startblock;

        // Entry: produce an Int value
        let val = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();

        // Create target block with one inputarg (Phi node)
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];

        // Link: entry → target, passing val as the Phi arg
        graph.set_goto(entry, target, vec![val]);
        graph.set_return(target, Some(phi));

        let state = annotate(&graph);
        // Phi should inherit Int from val via Link propagation
        assert_eq!(
            state.get(phi),
            &ValueType::Int,
            "Phi node should receive Int annotation from Link args"
        );
    }

    #[test]
    fn raise_link_propagates_exception_pair_with_special_types() {
        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let (exc_block, etype, evalue) = graph.exceptblock_args();
        let last_exception = graph.alloc_value();
        let last_exc_value = graph.alloc_value();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
                Link::new(
                    vec![last_exception, last_exc_value],
                    exc_block,
                    Some(exception_exitcase()),
                )
                .extravars(
                    Some(LinkArg::from(last_exception)),
                    Some(LinkArg::from(last_exc_value)),
                ),
            ],
        );

        let state = annotate(&graph);
        assert_eq!(state.get(last_exception), &ValueType::Int);
        assert_eq!(state.get(last_exc_value), &ValueType::Ref);
        assert_eq!(state.get(etype), &ValueType::Int);
        assert_eq!(state.get(evalue), &ValueType::Ref);
    }

    #[test]
    fn propagates_float_return_into_returnblock() {
        let mut graph = FunctionGraph::new("float_return");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let result = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("floatval", None),
                    ty: ValueType::Float,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let state = annotate(&graph);
        let ret = graph.block(graph.returnblock).inputargs[0];
        assert_eq!(state.get(result), &ValueType::Float);
        assert_eq!(state.get(ret), &ValueType::Float);
    }

    #[test]
    fn synthetic_void_return_stays_void() {
        let mut graph = FunctionGraph::new("void_return");
        let entry = graph.startblock;
        graph.set_return(entry, None);

        let state = annotate(&graph);
        let ret = graph.block(graph.returnblock).inputargs[0];
        assert_eq!(state.get(ret), &ValueType::Void);
    }
}
