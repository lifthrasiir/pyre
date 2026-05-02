impl pyre_interpreter::ControlFlowOpcodeHandler for crate::state::MIFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.fallthrough_pc()
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::set_next_instr(this, ctx, target);
            Ok(())
        })
    }

    #[inline(never)]
    fn close_loop_args(
        &mut self,
        target: usize,
    ) -> Result<Option<Vec<Self::Value>>, pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            // pyjitpl.py:2950-3036 reached_loop_header
            let code_ptr = unsafe { (*this.sym().jitcode).code };
            let back_edge_key = crate::driver::make_green_key(code_ptr, target);
            // pyjitpl.py:2951 self.heapcache.reset()
            ctx.heap_cache_mut().reset();
            // pyjitpl.py:2957-2965 build live_arg_boxes ONCE.
            let live_args =
                crate::state::MIFrame::close_loop_args_at(this, ctx, Some(target));
            let live_types = {
                live_args
                    .iter()
                    .map(|opref| {
                        ctx.get_opref_type(*opref).unwrap_or_else(|| {
                            panic!(
                                "live_arg {opref:?} has no type in OptContext; \
                                 RPython Box always carries its type"
                            )
                        })
                    })
                    .collect()
            };

            // pyjitpl.py:2978-2983 compile_trace attempt.
            {
                let (driver, _) = crate::driver::driver_pair();
                let has_partial = driver.meta_interp().partial_trace().is_some();
                let bridge_origin = driver
                    .meta_interp()
                    .bridge_info()
                    .map(|b| (b.trace_id, b.fail_index));
                let has_targets = driver.meta_interp().has_compiled_targets(back_edge_key);
                if !has_partial && has_targets {
                    let outcome = driver.meta_interp_mut().compile_trace(
                        back_edge_key,
                        &live_args,
                        bridge_origin,
                    );
                    if matches!(outcome, majit_metainterp::CompileOutcome::Compiled { .. }) {
                        if majit_metainterp::majit_log_enabled() {
                            eprintln!(
                                "[jit][reached_loop_header] compile_trace success: key={} pc={} bridge={:?}",
                                back_edge_key, target, bridge_origin
                            );
                        }
                        // pyjitpl.py:3095 raise_if_successful() — successful
                        // compile_trace raises ContinueRunningNormally and
                        // unwinds; no value propagates to the caller. Pyre
                        // ports that as an early `return Ok(None)` so the
                        // tracer stops at this point rather than falling
                        // through to the loop-close path.
                        driver.note_compile_trace_success();
                        return Ok(None);
                    }
                }
            }
            // pyjitpl.py:2994-3036 search current_merge_points
            if !ctx.has_merge_point(back_edge_key) {
                // pyjitpl.py:3034-3036 first visit, register & continue
                ctx.add_merge_point(back_edge_key, live_args, live_types, target);
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][reached_loop_header] first visit, unroll: key={} pc={}",
                        back_edge_key, target
                    );
                }
                return Ok(None);
            }
            // pyjitpl.py:3002-3030 found, compile as loop.
            Ok(Some(
                live_args
                    .into_iter()
                    .map(crate::state::FrontendOp::void)
                    .collect(),
            ))
        })
    }
}

impl pyre_interpreter::BranchOpcodeHandler for crate::state::MIFrame {
    fn concrete_truth_as_bool(
        &mut self,
        value: Self::Value,
        _truth: Self::Truth,
    ) -> Result<bool, pyre_interpreter::PyError> {
        crate::state::MIFrame::concrete_branch_truth_for_value(
            self,
            value.opref,
            value.concrete.to_pyobj(),
        )
    }

    fn guard_truth_value(
        &mut self,
        truth: Self::Truth,
        expect_true: bool,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            let opcode = if expect_true {
                majit_ir::OpCode::GuardTrue
            } else {
                majit_ir::OpCode::GuardFalse
            };
            crate::state::MIFrame::generate_guard(this, ctx, opcode, &[truth]);
            Ok(())
        })
    }

    fn record_branch_guard(
        &mut self,
        value: Self::Value,
        truth: Self::Truth,
        concrete_truth: bool,
        other_target: usize,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::record_branch_guard(
                this,
                ctx,
                value.opref,
                truth,
                concrete_truth,
                other_target,
            );
            Ok(())
        })
    }
}

impl pyre_interpreter::NamespaceOpcodeHandler for crate::state::MIFrame {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let ns = self.sym().concrete_namespace;
        let Some(slot) = crate::state::dict_storage_slot_direct(ns, name) else {
            let globals = self.trace_globals_ptr();
            let opref = self.with_trace_ctx(|ctx| {
                crate::helpers::emit_trace_load_name_from_namespace(ctx, globals, name)
            });
            let result_concrete = if ns.is_null() {
                crate::state::ConcreteValue::Null
            } else {
                unsafe {
                    let obj = pyre_interpreter::dict_storage_get(&*ns, name)
                        .unwrap_or(std::ptr::null_mut());
                    crate::state::ConcreteValue::from_pyobj(obj)
                }
            };
            return Ok(crate::state::FrontendOp::new(opref, result_concrete));
        };
        let concrete_cv = crate::state::dict_storage_value_direct(ns, slot);
        let result_concrete = concrete_cv
            .map(crate::state::ConcreteValue::from_pyobj)
            .unwrap_or(crate::state::ConcreteValue::Null);
        if let Some(concrete_value) = concrete_cv {
            if !concrete_value.is_null() {
                // celldict.py @elidable_promote + quasiimmut.py parity:
                // 1. QUASIIMMUT_FIELD(ns, slot) — collected as quasi_immutable_deps
                //    + GUARD_NOT_INVALIDATED.
                // 2. RECORD_KNOWN_RESULT(result, ns, slot) — cache trace-time
                //    lookup result (call_pure_results).
                // 3. CALL_PURE_R(ns, slot) — elidable lookup; record_result_of_call_pure
                //    folds via OptPure lookup_known_result.
                let opref = self.with_ctx(|_this, ctx| {
                    let ns_const = ctx.const_ref(ns as i64);
                    let slot_const = ctx.const_int(slot as i64);
                    ctx.record_op(majit_ir::OpCode::QuasiimmutField, &[ns_const, slot_const]);
                    let result_const = ctx.const_ref(concrete_value as i64);
                    ctx.record_op(
                        majit_ir::OpCode::RecordKnownResult,
                        &[result_const, ns_const, slot_const],
                    );
                    let call_result =
                        ctx.record_op(majit_ir::OpCode::CallPureR, &[ns_const, slot_const]);
                    Ok::<_, pyre_interpreter::PyError>(call_result)
                })?;
                return Ok(crate::state::FrontendOp::new(opref, result_concrete));
            }
        }
        let opref = self
            .with_ctx(|this, ctx| crate::state::MIFrame::load_namespace_value(this, ctx, slot))?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn store_name_value(
        &mut self,
        name: &str,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let ns = self.sym().concrete_namespace;
        let Some(slot) = crate::state::dict_storage_slot_direct(ns, name) else {
            let globals = self.trace_globals_ptr();
            return self.with_trace_ctx(|ctx| {
                crate::helpers::emit_trace_store_name_to_namespace(ctx, globals, name, value.opref);
                Ok(())
            });
        };
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::store_namespace_value(this, ctx, slot, value.opref)
        })
    }

    fn null_value(&mut self) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let opref = self.with_trace_ctx(|ctx| ctx.const_int(0));
        Ok(crate::state::FrontendOp::new(
            opref,
            crate::state::ConcreteValue::Ref(pyre_object::PY_NULL),
        ))
    }
}

impl pyre_interpreter::ArithmeticOpcodeHandler for crate::state::MIFrame {
    fn binary_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        // Concrete result via interpreter dispatch (baseobjspace).
        let result_concrete = crate::concrete_binary_value(op, lhs_obj, rhs_obj);
        if matches!(op, BinaryOperator::Subscr) {
            let fop = self.binary_subscr_value(a, b, lhs_obj, rhs_obj)?;
            let concrete = if result_concrete.is_null() {
                fop.concrete
            } else {
                result_concrete
            };
            return Ok(crate::state::FrontendOp::new(fop.opref, concrete));
        }
        let is_float_path = (!lhs_obj.is_null()
            && !rhs_obj.is_null()
            && unsafe { pyre_object::is_float(lhs_obj) || pyre_object::is_float(rhs_obj) })
            || self.value_type(a) == majit_ir::Type::Float
            || self.value_type(b) == majit_ir::Type::Float;
        let opref = if is_float_path {
            self.binary_float_value(a, b, op, lhs_obj, rhs_obj)?
        } else {
            self.binary_int_value(a, b, op, lhs_obj, rhs_obj)?
        };
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn compare_value(
        &mut self,
        a_fop: Self::Value,
        b_fop: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let a = a_fop.opref;
        let b = b_fop.opref;
        let lhs_obj = a_fop.concrete.to_pyobj();
        let rhs_obj = b_fop.concrete.to_pyobj();
        // Concrete result via interpreter dispatch (baseobjspace::compare).
        let result_concrete = crate::concrete_compare_value(op, lhs_obj, rhs_obj);
        let opref = self.compare_value_direct(a, b, op, lhs_obj, rhs_obj)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn unary_negative_value(
        &mut self,
        value: Self::Value,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { pyre_object::is_int(concrete_val) } {
            let v = unsafe { pyre_object::w_int_get_value(concrete_val) };
            result_concrete = crate::state::ConcreteValue::Int(v.wrapping_neg());
        }
        let opref = self.unary_int_value(value.opref, majit_ir::OpCode::IntNeg, concrete_val)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn unary_invert_value(
        &mut self,
        value: Self::Value,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let concrete_val = value.concrete.to_pyobj();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if !concrete_val.is_null() && unsafe { pyre_object::is_int(concrete_val) } {
            let v = unsafe { pyre_object::w_int_get_value(concrete_val) };
            result_concrete = crate::state::ConcreteValue::Int(!v);
        }
        let opref = self.unary_int_value(value.opref, majit_ir::OpCode::IntInvert, concrete_val)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }
}
