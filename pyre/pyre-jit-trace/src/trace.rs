//! Public trace entrypoint for `pyre`'s JIT portal.
//!
//! RPython MetaInterp._interpret() parity: trace_bytecode creates a
//! PyreMetaInterp and delegates to interpret(). The interpret loop
//! calls MIFrame::trace_code_step() for each bytecode, combining
//! concrete execution and symbolic IR recording.

use majit_metainterp::{TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;

use crate::metainterp::PyreMetaInterp;
use crate::state::{MIFrame, PyreSym};

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
    let frame = MIFrame {
        // Persistent fields (pyjitpl.py:65-100).
        sym: sym as *mut PyreSym,
        owned_sym: None,
        jitcode: w_code,
        pc: start_pc,
        greenkey: None,
        owned_concrete_frame: Some(concrete_frame),
        parent_frames: Vec::new(),
        drop_frame_opref: None,
        caller_result_stack_idx: None,
        arg_state: pyre_interpreter::bytecode::OpArgState::default(),
        // Per-instruction fields default-init; populated each step.
        ctx: std::ptr::null_mut(),
        meta: std::ptr::null_mut(),
        fallthrough_pc: 0,
        concrete_frame_addr: cf_addr,
        orgpc: start_pc,
        pre_opcode_registers_r: None,
        pending_result_stack_idx: None,
        pending_inline_frame: None,
    };

    let mut metainterp = PyreMetaInterp::new(w_code, std::ptr::null_mut());
    metainterp.framestack.push(frame);

    // pyjitpl.py:2971-2973: register the initial merge point so
    // reached_loop_header recognizes the trace start backedge and closes
    // the loop instead of unrolling it as a first-visit inner loop.
    //
    // interp_jit.py:118 reads `frame.is_being_profiled` live at every
    // backedge, so derive the start key from the live concrete frame
    // rather than caching it across the trace.
    let root_frame_ref = metainterp
        .framestack
        .first()
        .expect("trace_bytecode root frame must be live before interpret()");
    let start_key = root_frame_ref
        .green_key_hash_for_pc(start_pc)
        .expect("trace_bytecode root frame must hold a concrete PyFrame");
    {
        let input_args: Vec<majit_ir::OpRef> = (0..ctx.num_inputs())
            .map(|i| majit_ir::OpRef(i as u32))
            .collect();
        let input_types = ctx.inputarg_types();
        ctx.add_merge_point(start_key, input_args, input_types, start_pc);
    }

    let action = metainterp.interpret(ctx);

    // pyjitpl.py:3160: greenkey = original_boxes[:num_green_args]
    // original_boxes comes from the merge point where the loop closes
    // (pyjitpl.py:2995), which may differ from start_pc when
    // cut_trace_from retargets to an inner loop.
    //
    // Compute the close-loop key BEFORE popping the root frame so the
    // live `is_being_profiled` is observed (interp_jit.py:118), instead
    // of the trace-start snapshot.
    match &action {
        TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(target_pc),
            ..
        } if *target_pc != start_pc => {
            let target_key = metainterp
                .framestack
                .last()
                .and_then(|frame| frame.green_key_hash_for_pc(*target_pc))
                .unwrap_or(start_key);
            ctx.set_green_key(target_key, (w_code as usize, *target_pc));
            ctx.header_pc = *target_pc;
            ctx.cut_inner_green_key = Some(target_key);
        }
        TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
            let key = metainterp
                .framestack
                .first()
                .and_then(|frame| frame.green_key_hash_for_pc(start_pc))
                .unwrap_or(start_key);
            ctx.set_green_key(key, (w_code as usize, start_pc));
            ctx.header_pc = start_pc;
        }
        _ => {}
    }

    // Recover the root frame's owned_concrete_frame for writeback.
    let executed_frame = metainterp
        .framestack
        .pop()
        .and_then(|f| f.owned_concrete_frame);

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
