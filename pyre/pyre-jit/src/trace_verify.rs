//! Verification: auto-generated trace functions produce correct IR.

#[cfg(test)]
mod tests {
    use crate::jit::descr::{
        make_field_descr, make_immutable_field_descr, make_size_descr_with_type_and_vtable,
    };
    use majit_ir::{OpCode, OpRef};
    use majit_metainterp::TraceCtx;

    fn ob_type_descr() -> majit_ir::DescrRef {
        make_field_descr(0, 8, majit_ir::Type::Int, false)
    }
    fn intval_descr() -> majit_ir::DescrRef {
        make_field_descr(8, 8, majit_ir::Type::Int, true)
    }
    fn immutable_ob_type_descr() -> majit_ir::DescrRef {
        make_immutable_field_descr(0, 8, majit_ir::Type::Int, false)
    }
    fn immutable_intval_descr() -> majit_ir::DescrRef {
        make_immutable_field_descr(8, 8, majit_ir::Type::Int, true)
    }
    fn floatval_descr() -> majit_ir::DescrRef {
        make_field_descr(8, 8, majit_ir::Type::Float, false)
    }
    fn immutable_floatval_descr() -> majit_ir::DescrRef {
        make_immutable_field_descr(8, 8, majit_ir::Type::Float, false)
    }
    fn w_int_size_descr() -> majit_ir::DescrRef {
        make_size_descr_with_type_and_vtable(16, 0, 0)
    }
    fn w_float_size_descr() -> majit_ir::DescrRef {
        make_size_descr_with_type_and_vtable(16, 0, 0)
    }
    const FAKE_INT_TYPE: i64 = 0x1234_5678;
    const FAKE_FLOAT_TYPE: i64 = 0x1234_9876;

    /// Get ops from TraceCtx, excluding the dummy Finish we add for finalization.
    fn get_ops(ctx: TraceCtx) -> Vec<OpCode> {
        let mut recorder = ctx.into_recorder();
        let dummy_descr = make_size_descr_with_type_and_vtable(0, 0, 0);
        recorder.finish(&[], dummy_descr);
        let trace = recorder.get_trace();
        trace
            .ops
            .iter()
            .map(|o| o.opcode)
            .filter(|op| *op != OpCode::Finish)
            .collect()
    }

    #[test]
    fn test_unbox_int_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef::input_arg_ref(0);
        let _intval = crate::trace_unbox_int(
            &mut ctx,
            obj,
            FAKE_INT_TYPE,
            ob_type_descr(),
            intval_descr(),
        );
        let ops = get_ops(ctx);
        // GUARD_CLASS(box, cls): backend loads typeptr from obj at offset 0
        // (llgraph/runner.py:1245), no explicit pre-read. Followed by the
        // intval field read.
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcI]);
        eprintln!("✓ trace_unbox_int: {:?}", ops);
    }

    #[test]
    fn test_box_int_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let _obj = crate::trace_box_int(
            &mut ctx,
            OpRef::input_arg_ref(0),
            w_int_size_descr(),
            ob_type_descr(),
            intval_descr(),
            FAKE_INT_TYPE,
        );
        let ops = get_ops(ctx);
        // rewrite.py:479-484 GC rewriter: NEW_WITH_VTABLE emits the vtable
        // field via fielddescr_vtable, so the trace records
        // NEW_WITH_VTABLE(size) + only the non-type field setter.
        assert_eq!(ops, vec![OpCode::NewWithVtable, OpCode::SetfieldGc]);
        eprintln!("✓ trace_box_int: {:?}", ops);
    }

    #[test]
    fn test_int_binop_ovf_ops() {
        let mut ctx = TraceCtx::for_test(2);
        let _result = crate::trace_int_binop_ovf(
            &mut ctx,
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpCode::IntAddOvf,
            FAKE_INT_TYPE,
            ob_type_descr(),
            intval_descr(),
            w_int_size_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GuardClass,
                OpCode::GetfieldGcI, // unbox a
                OpCode::GuardClass,
                OpCode::GetfieldGcI, // unbox b
                OpCode::IntAddOvf,
                OpCode::GuardNoOverflow, // add + guard
                OpCode::NewWithVtable,
                OpCode::SetfieldGc, // box (NEW_WITH_VTABLE + intval setter)
            ]
        );
        eprintln!("✓ trace_int_binop_ovf: {} ops", ops.len());
    }

    #[test]
    fn test_unbox_float_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef::input_arg_ref(0);
        let _floatval = crate::trace_unbox_float(
            &mut ctx,
            obj,
            FAKE_FLOAT_TYPE,
            ob_type_descr(),
            floatval_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcF]);
        eprintln!("✓ trace_unbox_float: {:?}", ops);
    }

    #[test]
    fn test_unbox_int_ops_use_pure_reads_for_immutable_descrs() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef::input_arg_ref(0);
        let _intval = crate::trace_unbox_int(
            &mut ctx,
            obj,
            FAKE_INT_TYPE,
            immutable_ob_type_descr(),
            immutable_intval_descr(),
        );
        let ops = get_ops(ctx);
        // Immutable intval descr → GetfieldGcPureI for the value read.
        // GuardClass takes the object directly (no pre-read of ob_type).
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcPureI]);
    }

    #[test]
    fn test_unbox_float_ops_use_pure_reads_for_immutable_descrs() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef::input_arg_ref(0);
        let _floatval = crate::trace_unbox_float(
            &mut ctx,
            obj,
            FAKE_FLOAT_TYPE,
            immutable_ob_type_descr(),
            immutable_floatval_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::GuardClass, OpCode::GetfieldGcPureF]);
    }

    #[test]
    fn test_box_float_ops() {
        let mut ctx = TraceCtx::for_test(1);
        let _obj = crate::trace_box_float(
            &mut ctx,
            OpRef::input_arg_ref(0),
            w_float_size_descr(),
            ob_type_descr(),
            floatval_descr(),
            FAKE_FLOAT_TYPE,
        );
        let ops = get_ops(ctx);
        assert_eq!(ops, vec![OpCode::NewWithVtable, OpCode::SetfieldGc]);
        eprintln!("✓ trace_box_float: {:?}", ops);
    }

    #[test]
    fn test_float_binop_ops() {
        let mut ctx = TraceCtx::for_test(2);
        let _result = crate::trace_float_binop(
            &mut ctx,
            OpRef::input_arg_ref(0),
            OpRef::input_arg_ref(1),
            OpCode::FloatAdd,
            FAKE_FLOAT_TYPE,
            ob_type_descr(),
            floatval_descr(),
            w_float_size_descr(),
        );
        let ops = get_ops(ctx);
        assert_eq!(
            ops,
            vec![
                OpCode::GuardClass,
                OpCode::GetfieldGcF, // unbox a
                OpCode::GuardClass,
                OpCode::GetfieldGcF, // unbox b
                OpCode::FloatAdd,
                OpCode::NewWithVtable,
                OpCode::SetfieldGc, // box
            ]
        );
        eprintln!("✓ trace_float_binop: {} ops", ops.len());
    }
}
