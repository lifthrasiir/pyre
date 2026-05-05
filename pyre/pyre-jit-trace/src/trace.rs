//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! RPython MetaInterp._interpret() parity: trace_bytecode creates a
//! PyreMetaInterp and delegates to interpret(). The interpret loop
//! calls MIFrame::trace_code_step() for each bytecode, combining
//! concrete execution and symbolic IR recording.

use majit_metainterp::{TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;

use crate::metainterp::{MetaInterpFrame, PyreMetaInterp};
use crate::state::PyreSym;

/// Trace an entire loop body starting at `start_pc`.
///
/// RPython MetaInterp._interpret() parity: creates a PyreMetaInterp
/// with a single frame and delegates to interpret(). The interpret
/// loop calls MIFrame::trace_code_step() for each bytecode, combining
/// concrete execution and symbolic IR recording.
pub fn trace_bytecode(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    _code: &CodeObject,
    start_pc: usize,
    mut concrete_frame: Box<pyre_interpreter::pyframe::PyFrame>,
) -> (TraceAction, Box<pyre_interpreter::pyframe::PyFrame>) {
    // RPython MetaInterp._interpret() parity: root frame owns a concrete
    // PyFrame snapshot. MetaInterp drives both symbolic tracing AND
    // concrete execution — the interpreter does not run during tracing.
    concrete_frame.set_last_instr_from_next_instr(start_pc);
    let w_code = concrete_frame.pycode;
    let cf_addr = &*concrete_frame as *const pyre_interpreter::pyframe::PyFrame as usize;
    // pyjitpl.py:65 MIFrame.__init__: sym fields populated once at frame
    // construction. Callee (inline) frames are set up by perform_call
    // (trace_opcode.rs:3323-3424) and don't call init_symbolic; this path
    // handles the root frame push.
    sym.init_symbolic(ctx, cf_addr);
    let frame = MetaInterpFrame {
        sym: sym as *mut PyreSym,
        owned_sym: None,
        jitcode: w_code,
        pc: start_pc,
        greenkey: None,
        concrete_frame: cf_addr,
        owned_concrete_frame: Some(concrete_frame),
        parent_frames: Vec::new(),
        drop_frame_opref: None,
        caller_result_stack_idx: None,
        caller_result_type: None,
        arg_state: pyre_interpreter::bytecode::OpArgState::default(),
    };

    let mut metainterp = PyreMetaInterp::new(w_code, std::ptr::null_mut());
    metainterp.framestack.push(frame);

    // pyjitpl.py:2971-2973: register the initial merge point so
    // reached_loop_header recognizes the trace start backedge and closes
    // the loop instead of unrolling it as a first-visit inner loop.
    let start_key = crate::driver::make_green_key(w_code, start_pc);
    {
        // resoperation.py:719/727/739 InputArg{Int,Ref,Float}: each input
        // arg's typed Box is intrinsic to its `box.type`. Mint typed
        // OpRefs from `inputarg_types()` so the merge-point's args carry
        // RPython Box identity (variant-aware Eq) rather than collapsing
        // through an Untyped position (history.py:182 `box.type`).
        let input_types = ctx.inputarg_types();
        let input_args: Vec<majit_ir::OpRef> = input_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| majit_ir::OpRef::input_arg_typed(i as u32, tp))
            .collect();
        ctx.add_merge_point(start_key, input_args, input_types, start_pc);
    }

    let action = metainterp.interpret(ctx);

    // Recover the root frame's owned_concrete_frame for writeback.
    let executed_frame = metainterp
        .framestack
        .pop()
        .and_then(|f| f.owned_concrete_frame);

    // pyjitpl.py:3160: greenkey = original_boxes[:num_green_args]
    // original_boxes comes from the merge point where the loop closes
    // (pyjitpl.py:2995), which may differ from start_pc when
    // cut_trace_from retargets to an inner loop.
    match &action {
        TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(target_pc),
            ..
        } if *target_pc != start_pc => {
            let target_key = crate::driver::make_green_key(w_code, *target_pc);
            ctx.set_green_key(target_key, (w_code as usize, *target_pc));
            ctx.header_pc = *target_pc;
            ctx.cut_inner_green_key = Some(target_key);
        }
        TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
            let key = crate::driver::make_green_key(w_code, start_pc);
            ctx.set_green_key(key, (w_code as usize, start_pc));
            ctx.header_pc = start_pc;
        }
        _ => {}
    }

    // On abort, root frame may still be on the stack.
    let root_frame = if let Some(frame) = executed_frame {
        frame
    } else {
        metainterp
            .framestack
            .pop()
            .and_then(|frame| frame.owned_concrete_frame)
            .expect("trace_bytecode must return the root concrete frame")
    };
    (action, root_frame)
}

#[cfg(test)]
mod tests {
    use crate::metainterp::semantic_fallthrough_pc;
    use pyre_interpreter::bytecode::Instruction;
    use pyre_interpreter::compile_exec;
    use pyre_interpreter::decode_instruction_at;

    #[test]
    fn test_semantic_fallthrough_pc_skips_branch_trivia() {
        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let branch_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::PopJumpIfFalse { .. }, _))
                )
            })
            .expect("test bytecode should contain POP_JUMP_IF_FALSE");

        let fallthrough_pc = semantic_fallthrough_pc(&code, branch_pc);
        let fallthrough_instruction = decode_instruction_at(&code, fallthrough_pc)
            .map(|(instruction, _)| instruction)
            .expect("semantic fallthrough should decode");

        assert!(
            !matches!(
                fallthrough_instruction,
                Instruction::ExtendedArg
                    | Instruction::Resume { .. }
                    | Instruction::Nop
                    | Instruction::Cache
                    | Instruction::NotTaken
            ),
            "semantic fallthrough must skip bytecode trivia"
        );
    }
}
