// `OpcodeHandler` trait impls for `MIFrame` — variant section, part 1.
//
// `pyre/pyre-jit-trace/build.rs` assembles `OUT_DIR/jit_trace_trait_impls.rs`
// from THREE pieces:
//
//   1. this file (`opcode_handler_impls_pre.template.rs`) — header +
//      `SharedOpcodeHandler` impl. Variant trait, hand-maintained.
//   2. `majit_translate::handler_spec::emit_simple_trait_impls()` — the 5
//      simple traits (`Constant/Stack/Truth/Iter/Local`) emitted from the
//      spec table in `majit-translate/src/handler_spec.rs`.
//   3. `opcode_handler_impls_post.template.rs` — the remaining variant
//      traits (`ControlFlow/Branch/Namespace/Arithmetic`).
//
// `lib.rs` `include!`s the assembled file once at crate root. Phase D will
// replace the simple-trait emit with codegen-derived dispatch through
// `BlackholeInterpreter::dispatch_one()` against the per-opcode jitcode
// produced by Phase C.
//
// **drift warning**: this file is a *transcription* of variant `MIFrame`
// helper logic. The helpers it calls (e.g. `MIFrame::push_value`,
// `MIFrame::call_callable_value`) live in the same crate; if you change
// those, you almost certainly need to change the matching impl block here
// too. `tests/trait_impls_snapshot.rs` guards against unintended drift by
// comparing the assembled output against a checked-in snapshot.

impl pyre_interpreter::SharedOpcodeHandler for crate::state::MIFrame {
    type Value = crate::state::FrontendOp;

    fn push_value(&mut self, value: Self::Value) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::push_value(this, ctx, value.opref, value.concrete);
            Ok(())
        })
    }

    fn pop_value(&mut self) -> Result<Self::Value, pyre_interpreter::PyError> {
        let s = self.sym();
        let stack_idx = s.valuestackdepth.checked_sub(s.nlocals + 1).unwrap_or(0);
        let concrete = s
            .concrete_stack
            .get(stack_idx)
            .copied()
            .unwrap_or(crate::state::ConcreteValue::Null);
        let opref =
            self.with_ctx(|this, ctx| crate::state::MIFrame::pop_value(this, ctx))?;
        Ok(crate::state::FrontendOp::new(opref, concrete))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, pyre_interpreter::PyError> {
        let s = self.sym();
        let stack_idx = s.valuestackdepth.checked_sub(s.nlocals + depth + 1);
        let concrete = stack_idx
            .and_then(|idx| s.concrete_stack.get(idx).copied())
            .unwrap_or(crate::state::ConcreteValue::Null);
        let opref = self
            .with_ctx(|this, ctx| crate::state::MIFrame::peek_value(this, ctx, depth))?;
        Ok(crate::state::FrontendOp::new(opref, concrete))
    }

    fn guard_nonnull_value(
        &mut self,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        self.with_ctx(|this, ctx| {
            crate::state::MIFrame::guard_nonnull(this, ctx, value.opref);
            Ok(())
        })
    }

    fn make_function(
        &mut self,
        code_obj: Self::Value,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let opref = self.trace_make_function(code_obj.opref)?;
        Ok(crate::state::FrontendOp::void(opref))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        // executor.execute_varargs parity: compute concrete result.
        let concrete_callable = callable.concrete.to_pyobj();
        let concrete_args: Vec<pyre_object::PyObjectRef> =
            args.iter().map(|a| a.concrete.to_pyobj()).collect();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if !concrete_callable.is_null() && concrete_args.iter().all(|v| !v.is_null()) {
            unsafe {
                if pyre_interpreter::is_function(concrete_callable)
                    && pyre_interpreter::is_builtin_code(
                        pyre_interpreter::getcode(concrete_callable)
                            as pyre_object::PyObjectRef,
                    )
                {
                    let code = pyre_interpreter::getcode(concrete_callable);
                    let func = pyre_interpreter::builtin_code_get(
                        code as pyre_object::PyObjectRef,
                    );
                    let result = func(&concrete_args).unwrap_or(pyre_object::PY_NULL);
                    result_concrete = crate::state::ConcreteValue::from_pyobj(result);
                } else if pyre_interpreter::is_function(concrete_callable) {
                    // pyjitpl.py:2025 concrete execution only.
                    use std::cell::Cell;
                    thread_local! {
                        static CONCRETE_CALL_DEPTH: Cell<u32> = Cell::new(0);
                    }
                    let depth = CONCRETE_CALL_DEPTH.with(|d| d.get());
                    if depth < 32 {
                        CONCRETE_CALL_DEPTH.with(|d| d.set(depth + 1));
                        let exec_ctx = self.sym().concrete_execution_context;
                        let result =
                            pyre_interpreter::call::call_user_function_plain_with_ctx(
                                exec_ctx,
                                concrete_callable,
                                &concrete_args,
                            );
                        CONCRETE_CALL_DEPTH.with(|d| d.set(depth));
                        if let Ok(result) = result {
                            result_concrete = crate::state::ConcreteValue::from_pyobj(result);
                        }
                    }
                }
            }
        }
        let arg_oprefs: Vec<majit_ir::OpRef> = args.iter().map(|a| a.opref).collect();
        let opref = self.call_callable_value(
            callable.opref,
            &arg_oprefs,
            concrete_callable,
            &concrete_args,
        )?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn build_list(
        &mut self,
        items: &[Self::Value],
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let concrete_items: Vec<pyre_object::PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let list = pyre_interpreter::build_list_from_refs(&concrete_items);
            result_concrete = crate::state::ConcreteValue::from_pyobj(list);
        }
        let item_oprefs: Vec<majit_ir::OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_list(&item_oprefs)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn build_tuple(
        &mut self,
        items: &[Self::Value],
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        let concrete_items: Vec<pyre_object::PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let tuple = pyre_interpreter::build_tuple_from_refs(&concrete_items);
            result_concrete = crate::state::ConcreteValue::from_pyobj(tuple);
        }
        let item_oprefs: Vec<majit_ir::OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_tuple_value(&item_oprefs, &concrete_items)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn build_map(
        &mut self,
        items: &[Self::Value],
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let concrete_items: Vec<pyre_object::PyObjectRef> =
            items.iter().map(|i| i.concrete.to_pyobj()).collect();
        let mut result_concrete = crate::state::ConcreteValue::Null;
        if concrete_items.iter().all(|v| !v.is_null()) {
            let dict = pyre_interpreter::build_map_from_refs(&concrete_items);
            result_concrete = crate::state::ConcreteValue::from_pyobj(dict);
        }
        let item_oprefs: Vec<majit_ir::OpRef> = items.iter().map(|i| i.opref).collect();
        let opref = self.trace_build_map(&item_oprefs)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        // MIFrame parity: trace-only, no concrete mutation.
        // Root frame: interpreter executes the real STORE_SUBSCR.
        // Inline frame: MetaInterp.concrete_execute_step handles it.
        self.store_subscr_value(
            obj.opref,
            key.opref,
            value.opref,
            obj.concrete.to_pyobj(),
            key.concrete.to_pyobj(),
            value.concrete.to_pyobj(),
        )
    }

    fn list_append(
        &mut self,
        list: Self::Value,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        // MIFrame parity: trace-only, no concrete mutation.
        self.list_append_value(
            list.opref,
            value.opref,
            list.concrete.to_pyobj(),
            value.concrete.to_pyobj(),
        )
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, pyre_interpreter::PyError> {
        self.unpack_sequence_value(seq.opref, count, seq.concrete.to_pyobj())
    }

    fn load_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
    ) -> Result<Self::Value, pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        let mut result_concrete = crate::state::ConcreteValue::Null;
        let c_obj = obj.concrete.to_pyobj();
        if !c_obj.is_null() {
            if let Ok(result) = pyre_interpreter::baseobjspace::getattr(c_obj, name) {
                result_concrete = crate::state::ConcreteValue::from_pyobj(result);
            }
        }
        let opref = self.trace_load_attr(obj.opref, name)?;
        Ok(crate::state::FrontendOp::new(opref, result_concrete))
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), pyre_interpreter::PyError> {
        use crate::helpers::TraceHelperAccess;
        self.trace_store_attr(obj.opref, name, value.opref)
    }
}
