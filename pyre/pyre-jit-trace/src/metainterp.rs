//! PyreMetaInterp — RPython MetaInterp (pyjitpl.py:2371) parity.
//!
//! Single interpret() loop with a single framestack for both root
//! and inline frames. CALL → push_inline_frame (RPython perform_call),
//! RETURN → finishframe_inline (RPython finishframe).

use majit_ir::{OpRef, Type};
use majit_metainterp::{TraceAction, TraceCtx};
use pyre_interpreter::CodeObject;
use pyre_interpreter::bytecode::Instruction;

use super::state::{
    ConcreteValue, MIFrame, PendingInlineFrame, PyreSym, materialize_pending_inline_result,
    pending_inline_result_from_concrete,
};

/// RPython MetaInterp (pyjitpl.py:2371).
pub struct PyreMetaInterp {
    pub framestack: Vec<MIFrame>,
    pub portal_call_depth: i32,
    pub jitcode: *const (),
    pub namespace: *mut pyre_interpreter::DictStorage,
    inline_trace_base: usize,
}

impl PyreMetaInterp {
    pub fn new(jitcode: *const (), namespace: *mut pyre_interpreter::DictStorage) -> Self {
        Self {
            framestack: Vec::new(),
            portal_call_depth: -1,
            jitcode,
            namespace,
            inline_trace_base: 0,
        }
    }

    /// RPython MetaInterp._interpret() — single loop, single framestack.
    pub fn interpret(&mut self, ctx: &mut TraceCtx) -> TraceAction {
        self.inline_trace_base = ctx.inline_trace_depth();

        loop {
            let Some(top) = self.framestack.last_mut() else {
                return TraceAction::Abort;
            };
            let code = unsafe {
                &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject)
            };
            // RPython _interpret() parity: all frames use cf.next_instr as PC.
            // Skip trivia (Cache, ExtendedArg, etc.) — RPython's trace loop
            // only processes real opcodes. Inline frame returns or branch
            // targets can land on Cache instructions; skip forward to the
            // next semantic opcode so orgpc is always a real opcode PC.
            let mut pc = top
                .owned_concrete_frame
                .as_ref()
                .map(|cf| cf.next_instr())
                .unwrap_or(top.pc);
            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }
            // Advance past Cache/ExtendedArg/Nop/NotTaken/Resume trivia.
            while pc < code.instructions.len() {
                match pyre_interpreter::decode_instruction_at(code, pc) {
                    Some((
                        pyre_interpreter::bytecode::Instruction::Cache
                        | pyre_interpreter::bytecode::Instruction::ExtendedArg
                        | pyre_interpreter::bytecode::Instruction::Nop
                        | pyre_interpreter::bytecode::Instruction::NotTaken,
                        _,
                    )) => {
                        pc += 1;
                        // Sync the concrete frame so subsequent steps see the
                        // corrected PC.
                        if let Some(ref mut cf) = top.owned_concrete_frame {
                            cf.set_last_instr_from_next_instr(pc);
                        }
                        top.pc = pc;
                    }
                    _ => break,
                }
            }
            if pc >= code.instructions.len() {
                return TraceAction::Abort;
            }

            let action = if top.is_inline() {
                self.step_inline_frame(ctx, pc)
            } else {
                self.step_root_frame(ctx, pc)
            };

            match action {
                LoopAction::Continue => {}
                LoopAction::Return(ta) => return ta,
            }
            // pyjitpl.py:2837-2843 _interpret parity:
            //     while True:
            //         self.framestack[-1].run_one_step()
            //         self.blackhole_if_trace_too_long()
            //
            // The full `blackhole_if_trace_too_long` bookkeeping
            // (find_biggest_function / disable_noninlinable_function /
            // prepare_trace_segmenting) lives on `MetaInterp` and runs
            // in the jitdriver when it sees `TraceAction::Abort`. Here
            // we only mirror the trigger: stop dispatch the moment the
            // recorder exceeds `trace_limit`, which RPython signals via
            // `raise SwitchToBlackhole(ABORT_TOO_LONG)` inside
            // `blackhole_if_trace_too_long`.
            if ctx.is_too_long() {
                return TraceAction::Abort;
            }
        }
    }

    // ── Root frame step ──────────────────────────────────────────

    fn step_root_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        // pyjitpl.py:74 `self.metainterp` back-pointer — capture the
        // raw pointer before reborrowing through `self.framestack`
        // so MIFrame's recursion-depth walk can reach the same
        // framestack RPython's `_opimpl_recursive_call` consults.
        let meta_ptr = self as *mut Self;
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let cf_addr = top.live_concrete_frame_addr();
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        // pyjitpl.py:65-100 MIFrame.run_one_step: per-instruction state
        // lives on the same MIFrame instance as the persistent fields.
        // Rebind the per-step pointers / pcs in place rather than spinning
        // up a fresh `from_sym` view.
        top.ctx = ctx;
        top.meta = meta_ptr;
        top.fallthrough_pc = fallthrough_pc;
        top.concrete_frame_addr = cf_addr;
        top.orgpc = pc;
        top.pre_opcode_registers_r = None;
        top.pending_result_stack_idx = None;
        top.pending_inline_frame = None;

        let action = top.trace_code_step(code, pc);
        let pending = top.pending_inline_frame.take();

        if let Some(pending) = pending {
            // pyjitpl.py:1415 `self.metainterp.perform_call(portal_code,
            // allboxes)` — push callee frame onto framestack, trace
            // into callee. The recursion-limit check lives at the
            // decision site (trace_opcode.rs `_opimpl_recursive_call`
            // parity block); over-limit callees never produce a
            // pending here.
            let top = self.framestack.last_mut().unwrap();
            let sym = unsafe { &mut *top.sym };
            let next_pc = sym
                .pending_next_instr
                .take()
                .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
            top.pc = next_pc;
            let result_idx = sym
                .valuestackdepth
                .saturating_sub(sym.nlocals)
                .checked_sub(1);
            if let Some(ref mut cf) = top.owned_concrete_frame {
                cf.set_last_instr_from_next_instr(next_pc);
            }
            self.push_inline_frame(ctx, pending, result_idx);
            return LoopAction::Continue;
        }

        match action {
            TraceAction::Continue => {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let next_pc = sym
                    .pending_next_instr
                    .take()
                    .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
                top.pc = next_pc;
                // Sync concrete frame PC with symbolic PC.
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    cf.set_last_instr_from_next_instr(next_pc);
                }
                LoopAction::Continue
            }
            TraceAction::CloseLoop | TraceAction::CloseLoopWithArgs { .. } => {
                // pyjitpl.py:2970-2975 reached_loop_header:
                //     ptoken = self.get_procedure_token(greenboxes)
                //     if has_compiled_targets(ptoken):
                //         self.compile_trace(...)   # close the trace
                //     else:
                //         return  # continue tracing past this merge point
                //
                // Bridge traces skip loop headers that don't have compiled
                // code, continuing until they reach a COMPILED loop's header.
                // Without this, nested loop bridges close at the outer loop's
                // header instead of the inner loop's.
                if let Some(ref has_compiled) = ctx.has_compiled_targets_fn {
                    // pyjitpl.py:2979: ptoken = self.get_procedure_token(greenboxes)
                    // greenboxes = (code, pc) for the CURRENT frame's loop header.
                    let top_frame = self.framestack.last().unwrap();
                    let target_pc = match &action {
                        TraceAction::CloseLoopWithArgs {
                            loop_header_pc: Some(tp),
                            ..
                        } => *tp,
                        _ => pc,
                    };
                    let Some(gk) = top_frame.green_key_hash_for_pc(target_pc) else {
                        return LoopAction::Continue;
                    };
                    if !has_compiled(gk) {
                        // No compiled targets for this loop — continue tracing.
                        // RPython: reached_loop_header returns without closing.
                        let top = self.framestack.last_mut().unwrap();
                        top.pc = pc;
                        return LoopAction::Continue;
                    }
                }
                self.handle_close_loop(ctx, &action, pc);
                LoopAction::Return(action)
            }
            other => LoopAction::Return(other),
        }
    }

    // ── Inline frame step ────────────────────────────────────────

    fn step_inline_frame(&mut self, ctx: &mut TraceCtx, pc: usize) -> LoopAction {
        // pyjitpl.py:74 metainterp back-pointer (see step_root_frame).
        let meta_ptr = self as *mut Self;
        let top = self.framestack.last_mut().unwrap();
        let code = unsafe {
            &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                as *const pyre_interpreter::CodeObject)
        };
        let cf_addr = top.live_concrete_frame_addr();
        let fallthrough_pc = semantic_fallthrough_pc(code, pc);

        let inline_action = {
            top.ctx = ctx;
            top.meta = meta_ptr;
            top.fallthrough_pc = fallthrough_pc;
            top.concrete_frame_addr = cf_addr;
            top.orgpc = pc;
            top.pre_opcode_registers_r = None;
            top.pending_result_stack_idx = None;
            top.pending_inline_frame = None;
            // `parent_frames` already lives on the persistent frame; no
            // per-step clone needed — `trace_code_step_inline` reads it
            // directly from `self.parent_frames`.
            let result = top.trace_code_step_inline(code, pc);
            let pending = top.pending_inline_frame.take();

            if let Some(pending) = pending {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym
                    .valuestackdepth
                    .saturating_sub(sym.nlocals)
                    .checked_sub(1);
                // Pop concrete call args from parent owned_concrete_frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.set_last_instr_from_next_instr(sfall);
                }
                self.push_inline_frame(ctx, pending, result_idx);
                return LoopAction::Continue;
            }
            result
        };

        match inline_action {
            super::state::InlineTraceStepAction::Trace(TraceAction::Continue) => {
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                top.pc = sym
                    .pending_next_instr
                    .take()
                    .unwrap_or_else(|| semantic_fallthrough_pc(code, pc));
                // Concrete execution step
                self.concrete_execute_step();
                LoopAction::Continue
            }
            super::state::InlineTraceStepAction::Trace(TraceAction::Finish {
                finish_args,
                finish_arg_types,
                // pyjitpl.py:2489-2502 finishframe path: inline return/yield
                // never raises through this branch (exception exits take the
                // finishframe_exception route above), so the flag is ignored.
                exit_with_exception: _,
            }) => {
                self.finishframe_inline(ctx, &finish_args, &finish_arg_types);
                LoopAction::Continue
            }
            super::state::InlineTraceStepAction::Trace(
                TraceAction::Abort | TraceAction::AbortPermanent,
            ) => {
                // RPython finishframe_exception: if exception pending,
                // try multi-frame unwind before giving up.
                let has_exc = {
                    let top = self.framestack.last().unwrap();
                    let sym = unsafe { &*top.sym };
                    // pyjitpl.py:2506 finishframe_exception is reached
                    // only after opimpl_raise/handle_possible_exception has
                    // populated both last_exc_value and last_exc_box.  A
                    // concrete-only exception without a box cannot be encoded
                    // as compile_exit_frame_with_exception(valuebox).
                    !sym.last_exc_value.is_null() && sym.last_exc_box != OpRef::NONE
                };
                if has_exc {
                    if let Some(action) = self.finishframe_exception(ctx) {
                        return action;
                    }
                }
                ctx.truncate_inline_trace_positions(self.inline_trace_base);
                LoopAction::Return(TraceAction::Abort)
            }
            super::state::InlineTraceStepAction::Trace(action) => LoopAction::Return(action),
            super::state::InlineTraceStepAction::PushFrame(pending) => {
                // trace_code_step_inline already took the pending frame
                let top = self.framestack.last_mut().unwrap();
                let code = unsafe {
                    &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject)
                };
                let sym = unsafe { &mut *top.sym };
                let sfall = semantic_fallthrough_pc(code, pc);
                top.pc = sym.pending_next_instr.take().unwrap_or(sfall);
                let result_idx = sym
                    .valuestackdepth
                    .saturating_sub(sym.nlocals)
                    .checked_sub(1);
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    for _ in 0..pending.nargs {
                        cf.pop();
                    }
                    let _ = cf.pop(); // null_or_self
                    let _ = cf.pop(); // callable
                    cf.set_last_instr_from_next_instr(sfall);
                }
                self.push_inline_frame(ctx, pending, result_idx);
                LoopAction::Continue
            }
        }
    }

    // ── Frame management ─────────────────────────────────────────

    /// PRE-EXISTING-ADAPTATION: equivalent to RPython
    /// `MetaInterp.perform_call` (`rpython/jit/metainterp/pyjitpl.py`)
    /// which constructs and pushes the callee `MIFrame` directly.  Pyre
    /// splits trace-step output (returns `PendingInlineFrame`) from the
    /// frame-stack mutation (this method) so the trace handler stays
    /// borrow-isolated from the metainterp framestack.  The
    /// `PendingInlineFrame` struct itself has no upstream counterpart.
    fn push_inline_frame(
        &mut self,
        ctx: &mut TraceCtx,
        mut pending: PendingInlineFrame,
        caller_result_idx: Option<usize>,
    ) {
        // pyjitpl.py:1418-1420 perform_call → newframe: the inline-depth
        // counter and the recursion-walk both read `MetaInterp.framestack`
        // directly, so the legacy `TraceCtx::inline_frames` side table is
        // no longer fed from the pyre-tracer push site.
        ctx.push_inline_trace_position(pending.green_key);

        if let Some(parent) = pending.parent_frames.first_mut() {
            parent.pending_result_stack_idx = caller_result_idx;
        }

        let callee_code = pending.concrete_frame.pycode;
        let mut owned_sym = Box::new(pending.sym);
        let sym_ptr = owned_sym.as_mut() as *mut PyreSym;
        let owned_cf = Box::new(pending.concrete_frame);
        let cf_addr = &*owned_cf as *const pyre_interpreter::pyframe::PyFrame as usize;

        // executioncontext.py:76 / pyjitpl.py:1789: virtual_ref for callee frame.
        if let Some(frame_opref) = pending.drop_frame_opref {
            if let Some(caller) = self.framestack.last() {
                let caller_sym = unsafe { &mut *caller.sym };
                super::state::opimpl_virtual_ref(ctx, caller_sym, frame_opref, cf_addr);
            }
        }

        let frame = MIFrame {
            // Persistent fields (pyjitpl.py:65-100, perform_call).
            sym: sym_ptr,
            owned_sym: Some(owned_sym),
            jitcode: callee_code,
            pc: 0,
            greenkey: Some(pending.greenkey),
            owned_concrete_frame: Some(owned_cf),
            parent_frames: pending.parent_frames,
            drop_frame_opref: pending.drop_frame_opref,
            caller_result_stack_idx: caller_result_idx,
            arg_state: pyre_interpreter::bytecode::OpArgState::default(),
            // Per-instruction fields default-init; populated each step.
            ctx: std::ptr::null_mut(),
            meta: std::ptr::null_mut(),
            fallthrough_pc: 0,
            concrete_frame_addr: cf_addr,
            orgpc: 0,
            pre_opcode_registers_r: None,
            pending_result_stack_idx: None,
            pending_inline_frame: None,
        };

        self.portal_call_depth += 1;
        self.framestack.push(frame);
    }

    /// RPython finishframe: pop inline frame, store result in parent.
    fn finishframe_inline(
        &mut self,
        ctx: &mut TraceCtx,
        finish_args: &[OpRef],
        finish_arg_types: &[Type],
    ) {
        let result_opref = finish_args.first().copied().unwrap_or(OpRef::NONE);
        let result_type = finish_arg_types.first().copied().unwrap_or(Type::Ref);

        // Concrete RETURN_VALUE execution on inline frame
        let concrete_result = self.concrete_execute_return();

        // Pop frame
        let popped = self.framestack.pop().unwrap();
        self.portal_call_depth -= 1;
        ctx.pop_inline_trace_position();

        // executioncontext.py:89 / pyjitpl.py:1819: virtual_ref_finish for callee frame.
        if let Some(frame_opref) = popped.drop_frame_opref {
            if let Some(parent) = self.framestack.last() {
                let parent_sym = unsafe { &mut *parent.sym };
                super::state::opimpl_virtual_ref_finish(ctx, parent_sym, frame_opref);
            }
        }

        // Drop callee frame in trace
        if let Some(frame_opref) = popped.drop_frame_opref {
            ctx.call_void(
                crate::callbacks::get().jit_drop_callee_frame,
                &[frame_opref],
            );
        }

        // pyjitpl.py:2462-2477 popframe: the framestack pop above is the
        // single source of truth for the inline-depth counter; no
        // matching `TraceCtx::inline_frames` pop is needed.

        if self.framestack.is_empty() {
            return; // shouldn't happen: root never produces Finish
        }

        // make_result_of_lastop: store in parent
        let parent = self.framestack.last_mut().unwrap();
        let parent_sym = unsafe { &mut *parent.sym };

        if let Some(result_idx) = popped.caller_result_stack_idx {
            // Update the unified register file. Stack slots live at
            // `registers_r[nlocals + stack_idx]`; re-derive the abs
            // index and write the call result there.
            let parent_nlocals = parent_sym.nlocals;
            let abs_idx = parent_nlocals + result_idx;
            if abs_idx >= parent_sym.registers_r.len() {
                parent_sym.registers_r.resize(abs_idx + 1, OpRef::NONE);
            }
            parent_sym.registers_r[abs_idx] = result_opref;
            if result_idx >= parent_sym.symbolic_stack_types.len() {
                parent_sym
                    .symbolic_stack_types
                    .resize(result_idx + 1, Type::Ref);
            }
            parent_sym.symbolic_stack_types[result_idx] = result_type;

            // Update concrete stack. history.py:307 ConstPtr.getref_base
            // parity: a Ref-typed return value carries the raw GCREF —
            // the JIT never eagerly unboxes W_IntObject / W_FloatObject
            // at shadow-set time. For Int/Float-typed results we extract
            // the scalar because the opcode's signature said so.
            let cv = match result_type {
                Type::Int => ConcreteValue::from_pyobj(concrete_result),
                Type::Float => ConcreteValue::from_pyobj(concrete_result),
                Type::Ref => {
                    if concrete_result.is_null() {
                        ConcreteValue::Null
                    } else {
                        ConcreteValue::Ref(concrete_result)
                    }
                }
                Type::Void => ConcreteValue::Null,
            };
            if result_idx < parent_sym.concrete_stack.len() {
                parent_sym.concrete_stack[result_idx] = cv;
            } else {
                parent_sym
                    .concrete_stack
                    .resize(result_idx + 1, ConcreteValue::Null);
                parent_sym.concrete_stack[result_idx] = cv;
            }
        }

        // Keep concrete stack in sync only for inline parents.
        // Root frames are traced symbolically against a snapshot and are
        // not concrete-stepped instruction-by-instruction.
        if parent.is_inline() {
            if let Some(ref mut pcf) = parent.owned_concrete_frame {
                pcf.push(concrete_result);
            }
        }

        // No pending_concrete_push needed — concrete_stack[result_idx] already updated above.
    }

    // ── Concrete execution helpers ───────────────────────────────

    fn concrete_execute_step(&mut self) {
        let top = self.framestack.last_mut().unwrap();
        let Some(cf) = top.owned_concrete_frame.as_mut() else {
            return;
        };
        let cf = &mut **cf;
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(cf) };
        let ni = cf.next_instr();
        if ni >= code.instructions.len() {
            return;
        }

        let (opcode_pc, instruction, op_arg) =
            pyre_interpreter::decode_instruction_for_dispatch(code, ni)
                .expect("concrete_execute_step: bytecode corruption");
        cf.set_last_instr_from_next_instr(opcode_pc + 1);
        let next = cf.next_instr();

        if let Instruction::Call { argc } = instruction {
            let nargs = argc.get(op_arg) as usize;
            let _ = super::state::execute_inline_residual_call(cf, nargs);
            return;
        }

        let _ = pyre_interpreter::execute_opcode_step(cf, code, instruction, op_arg, next);
    }

    fn concrete_execute_return(&mut self) -> pyre_object::PyObjectRef {
        let top = self.framestack.last_mut().unwrap();
        let Some(cf) = top.owned_concrete_frame.as_mut() else {
            return pyre_object::PY_NULL;
        };
        let cf = &mut **cf;
        let code = unsafe { &*pyre_interpreter::pyframe_get_pycode(cf) };
        let ni = cf.next_instr();
        if ni >= code.instructions.len() {
            return pyre_object::PY_NULL;
        }

        let (opcode_pc, instruction, op_arg) =
            pyre_interpreter::decode_instruction_for_dispatch(code, ni)
                .expect("concrete_execute_return: bytecode corruption");
        cf.set_last_instr_from_next_instr(opcode_pc + 1);
        let next = cf.next_instr();

        match pyre_interpreter::execute_opcode_step(cf, code, instruction, op_arg, next) {
            Ok(pyre_interpreter::StepResult::Return(value)) => materialize_pending_inline_result(
                pending_inline_result_from_concrete(Type::Ref, value),
            ),
            _ => pyre_object::PY_NULL,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn handle_close_loop(&self, ctx: &mut TraceCtx, action: &TraceAction, pc: usize) {
        let w_code = self.framestack[0].jitcode;
        let target_pc = if let TraceAction::CloseLoopWithArgs {
            loop_header_pc: Some(tp),
            ..
        } = action
        {
            Some(*tp)
        } else if matches!(action, TraceAction::CloseLoop) {
            Some(pc)
        } else {
            None
        };
        if let Some(tp) = target_pc {
            // interp_jit.py:118 reads `frame.is_being_profiled` live at
            // every backedge; fall back to the root frame's live state
            // rather than hardcoding `false` so the resulting JitCell
            // matches the eval-side `make_green_key_for_frame` key.
            let key = self
                .framestack
                .last()
                .and_then(|frame| frame.green_key_hash_for_pc(tp))
                .or_else(|| {
                    self.framestack
                        .first()
                        .and_then(|f| f.green_key_hash_for_pc(tp))
                })
                .unwrap_or_else(|| crate::driver::make_green_key(w_code, tp, false));
            ctx.set_green_key(key, (w_code as usize, tp));
        }
    }

    /// RPython pyjitpl.py:2506 finishframe_exception (multi-frame).
    ///
    /// Walks the framestack looking for an exception handler, popping
    /// frames that don't have one. Structurally matches RPython's
    /// `while self.framestack: ... self.popframe()` loop.
    ///
    /// Returns Some(LoopAction) if handled, None if all frames exhausted.
    fn finishframe_exception(&mut self, ctx: &mut TraceCtx) -> Option<LoopAction> {
        // pyjitpl.py:74 metainterp back-pointer (see step_root_frame).
        let meta_ptr = self as *mut Self;
        // RPython pyjitpl.py:2506: while self.framestack:
        while let Some(top) = self.framestack.last() {
            let code = unsafe {
                &*(pyre_interpreter::w_code_get_ptr(top.jitcode as pyre_object::PyObjectRef)
                    as *const pyre_interpreter::CodeObject)
            };
            let sym = unsafe { &*top.sym };
            let pc = top.pc;

            // RPython: if opcode == op_catch_exception → handler found
            if let Some(entry) =
                pyre_interpreter::bytecode::find_exception_handler(&code.exceptiontable, pc as u32)
            {
                let handler_pc = entry.target as usize;
                let handler_depth = entry.depth as usize;
                let exc_opref = sym.last_exc_box;
                let exc_obj = sym.last_exc_value;

                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][finishframe_exception] found handler in frame depth={} handler_pc={}",
                        self.framestack.len(),
                        handler_pc
                    );
                }

                // Unwind symbolic + concrete state to handler.
                let top = self.framestack.last_mut().unwrap();
                let sym = unsafe { &mut *top.sym };
                let ncells = code.cellvars.len() + code.freevars.len();
                let nlocals = sym.nlocals;
                let target_stack_len = ncells + handler_depth;

                sym.symbolic_stack_types.truncate(target_stack_len);
                sym.concrete_stack.truncate(target_stack_len);
                sym.valuestackdepth = nlocals + target_stack_len;
                // Unified register file: truncate the stack tail of
                // `registers_r` in lockstep with the types / concrete
                // mirrors.
                if sym.registers_r.len() > nlocals + target_stack_len {
                    sym.registers_r.truncate(nlocals + target_stack_len);
                }
                if entry.push_lasti {
                    sym.symbolic_stack_types.push(Type::Ref);
                    sym.concrete_stack
                        .push(ConcreteValue::Ref(pyre_object::w_int_new(pc as i64)));
                    sym.valuestackdepth += 1;
                    sym.registers_r.push(OpRef::NONE);
                }
                sym.symbolic_stack_types.push(Type::Ref);
                sym.concrete_stack.push(ConcreteValue::Ref(exc_obj));
                sym.valuestackdepth += 1;
                sym.registers_r.push(exc_opref);
                sym.pending_next_instr = Some(handler_pc);

                // Sync concrete frame
                if let Some(ref mut cf) = top.owned_concrete_frame {
                    let target_depth = cf.nlocals() + cf.ncells() + handler_depth;
                    while cf.valuestackdepth > target_depth {
                        cf.pop();
                    }
                    if entry.push_lasti {
                        cf.push(pyre_object::w_int_new(pc as i64));
                    }
                    cf.push(exc_obj);
                    cf.set_last_instr_from_next_instr(handler_pc);
                }

                // pyjitpl.py:2518: frame.pc = target; raise ChangeFrame
                return Some(LoopAction::Continue);
            }

            // RPython: self.popframe() — no handler in this frame
            if top.is_inline() {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][finishframe_exception] pop inline frame, depth={}",
                        self.framestack.len()
                    );
                }
                // Propagate last_exc_value/last_exc_box to parent before popping
                let exc_value = sym.last_exc_value;
                let exc_box = sym.last_exc_box;
                let exc_const = sym.class_of_last_exc_is_const;

                // Pop the inline frame
                let popped = self.framestack.pop().unwrap();
                self.portal_call_depth -= 1;
                ctx.pop_inline_trace_position();

                // executioncontext.py:89 / pyjitpl.py:1819: virtual_ref_finish on exception leave.
                if let Some(frame_opref) = popped.drop_frame_opref {
                    if let Some(parent) = self.framestack.last() {
                        let parent_sym = unsafe { &mut *parent.sym };
                        super::state::opimpl_virtual_ref_finish(ctx, parent_sym, frame_opref);
                    }
                }

                // Drop callee frame in trace
                if let Some(frame_opref) = popped.drop_frame_opref {
                    ctx.call_void(
                        crate::callbacks::get().jit_drop_callee_frame,
                        &[frame_opref],
                    );
                }
                // pyjitpl.py:2506 finishframe_exception: framestack pop
                // above is sufficient — no `inline_frames` side-table
                // pop required (see push_inline_frame for why).

                // Propagate exception state to parent sym
                if let Some(parent) = self.framestack.last() {
                    let parent_sym = unsafe { &mut *parent.sym };
                    parent_sym.last_exc_value = exc_value;
                    parent_sym.last_exc_box = exc_box;
                    parent_sym.class_of_last_exc_is_const = exc_const;
                }

                continue; // check parent frame
            }

            // Root frame with no handler → compile_exit_frame_with_exception
            // pyjitpl.py:3238-3242 compile_exit_frame_with_exception:
            //   self.store_token_in_vable()
            //   token = sd.exit_frame_with_exception_descr_ref
            //   self.history.record1(rop.FINISH, valuebox, None, descr=token)
            let exc_opref = sym.last_exc_box;
            if majit_metainterp::majit_log_enabled() {
                eprintln!("[jit][finishframe_exception] root frame, no handler → FINISH");
            }
            // pyjitpl.py:3239: self.store_token_in_vable()
            // Record SetfieldGc on vable_token + GUARD_NOT_FORCED_2 with
            // proper resumedata, matching the return/yield path in
            // trace_step_result_to_action.
            {
                let sym_mut = unsafe { &mut *top.sym };
                let concrete_frame = sym_mut.concrete_vable_ptr as usize;
                let pc = sym_mut.pending_next_instr.unwrap_or(0);
                let mut mi =
                    super::state::MIFrame::from_sym(ctx, sym_mut, meta_ptr, concrete_frame, pc, pc);
                mi.store_token_in_vable(ctx);
            }
            return Some(LoopAction::Return(TraceAction::Finish {
                finish_args: vec![exc_opref],
                finish_arg_types: vec![Type::Ref],
                // pyjitpl.py:3241 token = sd.exit_frame_with_exception_descr_ref
                exit_with_exception: true,
            }));
        }
        None
    }
}

/// Internal loop control — not exposed outside interpret().
enum LoopAction {
    Continue,
    Return(TraceAction),
}

pub(crate) fn semantic_fallthrough_pc(code: &CodeObject, pc: usize) -> usize {
    let mut next_pc = pc.saturating_add(1);
    loop {
        match pyre_interpreter::decode_instruction_at(code, next_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => next_pc += 1,
            _ => return next_pc,
        }
    }
}
