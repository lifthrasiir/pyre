//! Spec-driven derivation for the simple `OpcodeHandler` trait impls.
//!
//! Phase B of the eval-loop automation plan. The 5 "simple" traits
//! (`Constant/Stack/Truth/Iter/Local`) have method bodies regular enough
//! to describe with a small enum. `emit_simple_trait_impls()` walks the
//! spec table and produces the exact Rust source bytes that used to live
//! in `pyre/pyre-jit-trace/src/opcode_handler_impls.template.rs`.
//!
//! The spec table is intentionally narrow — each variant captures one
//! body *shape*, not one method. Adding a new constant kind only requires
//! a new row in `CONSTANT_METHODS`, not 8 lines of Rust.
//!
//! The variant traits (`Shared/Namespace/Branch/ControlFlow/Arithmetic`)
//! are too irregular for spec abstraction and remain in the template file.
//! Phase D will replace both layers with jitcode dispatch from the per-opcode
//! `JitCode` artifact, so this transcription does not have to live forever.

/// Description of a `*_constant` method on `ConstantOpcodeHandler`.
struct ConstantMethod {
    /// Method name, e.g. `"int_constant"`.
    name: &'static str,
    /// Argument list rendered into the `fn`. Empty string for `none_constant`.
    args_decl: &'static str,
    /// `ConcreteValue::*` constructor expression. Receives `value` (or whatever
    /// the arg is named) in scope.
    concrete_expr: &'static str,
    /// Trace helper invocation expression. Same in-scope variables as above.
    trace_call: &'static str,
}

const CONSTANT_METHODS: &[ConstantMethod] = &[
    ConstantMethod {
        name: "int_constant",
        args_decl: "value: i64",
        concrete_expr: "crate::state::ConcreteValue::Int(value)",
        trace_call: "self.trace_int_constant(value)?",
    },
    ConstantMethod {
        name: "float_constant",
        args_decl: "value: f64",
        concrete_expr: "crate::state::ConcreteValue::Float(value)",
        trace_call: "self.trace_float_constant(value)?",
    },
    ConstantMethod {
        name: "bool_constant",
        args_decl: "value: bool",
        concrete_expr: "crate::state::ConcreteValue::Int(value as i64)",
        trace_call: "self.trace_bool_constant(value)?",
    },
    ConstantMethod {
        name: "str_constant",
        args_decl: "value: &str",
        concrete_expr: "crate::state::ConcreteValue::Ref(pyre_object::w_str_new(value))",
        trace_call: "self.trace_str_constant(value)?",
    },
    ConstantMethod {
        name: "none_constant",
        args_decl: "",
        concrete_expr: "crate::state::ConcreteValue::Ref(pyre_object::w_none())",
        trace_call: "self.trace_none_constant()?",
    },
    ConstantMethod {
        name: "ellipsis_constant",
        args_decl: "",
        concrete_expr: "crate::state::ConcreteValue::Ref(pyre_object::noneobject::w_ellipsis())",
        trace_call: "self.trace_ellipsis_constant()?",
    },
];

/// Render a single trivial-constant method (the `int/float/bool/str/none` shape).
fn emit_constant_method(out: &mut String, m: &ConstantMethod) {
    let args_part = if m.args_decl.is_empty() {
        String::new()
    } else {
        format!(", {}", m.args_decl)
    };
    out.push_str(&format!(
        "    fn {name}(&mut self{args}) -> Result<Self::Value, pyre_interpreter::PyError> {{\n",
        name = m.name,
        args = args_part,
    ));
    out.push_str("        use crate::helpers::TraceHelperAccess;\n");
    out.push_str(&format!(
        "        let concrete = {expr};\n",
        expr = m.concrete_expr
    ));
    out.push_str(&format!(
        "        let opref = {call};\n",
        call = m.trace_call
    ));
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");
}

/// Emit `impl ConstantOpcodeHandler for crate::state::MIFrame { ... }` —
/// the 8 constant methods. Trivial-shape methods come from `CONSTANT_METHODS`;
/// the three special-shape methods (`bigint`, `bytes`, `code`) are emitted
/// inline because their concrete constructor / arg type doesn't fit the
/// generic shape.
fn emit_constant_impl(out: &mut String) {
    out.push_str("impl pyre_interpreter::ConstantOpcodeHandler for crate::state::MIFrame {\n");

    // Emit `int_constant` first (matches the template ordering).
    emit_constant_method(out, &CONSTANT_METHODS[0]);

    // bigint_constant — special: by-ref arg + clone in concrete ctor.
    out.push('\n');
    out.push_str("    fn bigint_constant(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        value: &pyre_interpreter::PyBigInt,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        use crate::helpers::TraceHelperAccess;\n");
    out.push_str(
        "        let concrete = crate::state::ConcreteValue::Ref(pyre_object::w_long_new(value.clone()));\n",
    );
    out.push_str("        let opref = self.trace_bigint_constant(value)?;\n");
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");

    // float_constant, bool_constant, str_constant — trivial shape.
    for m in &CONSTANT_METHODS[1..4] {
        out.push('\n');
        emit_constant_method(out, m);
    }

    // bytes_constant — special: comment + named local before concrete ctor.
    out.push('\n');
    out.push_str("    fn bytes_constant(&mut self, value: &[u8]) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        use crate::helpers::TraceHelperAccess;\n");
    out.push_str("        // pyre lacks a separate bytes type — bytes literals materialise as\n");
    out.push_str("        // bytearray. Call sites that need true immutability must copy.\n");
    out.push_str(
        "        let bytes_ref = pyre_object::bytearrayobject::w_bytearray_from_bytes(value);\n",
    );
    out.push_str("        let concrete = crate::state::ConcreteValue::Ref(bytes_ref);\n");
    out.push_str("        let opref = self.trace_bytes_constant(value)?;\n");
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");

    // code_constant — special: pointer cast in concrete ctor.
    out.push('\n');
    out.push_str("    fn code_constant(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        code: &pyre_interpreter::CodeObject,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        use crate::helpers::TraceHelperAccess;\n");
    out.push_str("        let concrete = crate::state::ConcreteValue::Ref(\n");
    out.push_str(
        "            code as *const pyre_interpreter::CodeObject as pyre_object::PyObjectRef,\n",
    );
    out.push_str("        );\n");
    out.push_str("        let opref = self.trace_code_constant(code)?;\n");
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");

    // none_constant — trivial shape.
    out.push('\n');
    emit_constant_method(out, &CONSTANT_METHODS[4]);

    // ellipsis_constant — trivial shape (last).
    out.push('\n');
    emit_constant_method(out, &CONSTANT_METHODS[5]);

    // slice_constant — `pypy/objspace/std/objspace.py:385` `space.newslice`
    // emits a fresh W_SliceObject per call. The IR shape mirrors
    // `pypy/interpreter/pyopcode.py:1463 BUILD_SLICE` →
    // `op.newslice(w_start, w_end, w_step)` so the three operands stay
    // live in the trace as field dependencies (NewWithVtable + 3
    // SetfieldGc per `emit_box_slice_inline`). The concrete shadow mirrors
    // the root interpreter's W_SliceObject so STORE_SUBSCR can take the same
    // list slice-assignment specialization decision PyPy takes after
    // `space.newslice(...)`.
    out.push('\n');
    out.push_str("    fn slice_constant(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        start: Self::Value,\n");
    out.push_str("        stop: Self::Value,\n");
    out.push_str("        step: Self::Value,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        let opref = self.with_ctx(|_this, ctx| {\n");
    out.push_str("            let new_op = crate::helpers::emit_box_slice_inline(\n");
    out.push_str("                ctx,\n");
    out.push_str("                start.opref,\n");
    out.push_str("                stop.opref,\n");
    out.push_str("                step.opref,\n");
    out.push_str("                crate::descr::w_slice_size_descr(),\n");
    out.push_str("                crate::descr::slice_w_start_descr(),\n");
    out.push_str("                crate::descr::slice_w_stop_descr(),\n");
    out.push_str("                crate::descr::slice_w_step_descr(),\n");
    out.push_str("            );\n");
    out.push_str("            Ok::<_, pyre_interpreter::PyError>(new_op)\n");
    out.push_str("        })?;\n");
    out.push_str("        let concrete = crate::state::ConcreteValue::Ref(\n");
    out.push_str("            pyre_object::w_slice_new(\n");
    out.push_str("                start.concrete.to_pyobj(),\n");
    out.push_str("                stop.concrete.to_pyobj(),\n");
    out.push_str("                step.concrete.to_pyobj(),\n");
    out.push_str("            ),\n");
    out.push_str("        );\n");
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");

    out.push_str("}\n");
}

/// Emit `impl StackOpcodeHandler for crate::state::MIFrame { ... }`.
/// One method, pure delegation to `MIFrame::swap_values`.
fn emit_stack_impl(out: &mut String) {
    out.push_str("impl pyre_interpreter::StackOpcodeHandler for crate::state::MIFrame {\n");
    out.push_str(
        "    fn swap_values(&mut self, depth: usize) -> Result<(), pyre_interpreter::PyError> {\n",
    );
    out.push_str(
        "        self.with_ctx(|this, ctx| crate::state::MIFrame::swap_values(this, ctx, depth))\n",
    );
    out.push_str("    }\n");
    out.push_str("}\n");
}

/// Emit `impl TruthOpcodeHandler for crate::state::MIFrame { ... }`.
fn emit_truth_impl(out: &mut String) {
    out.push_str("impl pyre_interpreter::TruthOpcodeHandler for crate::state::MIFrame {\n");
    out.push_str("    type Truth = majit_ir::OpRef;\n");
    out.push_str("\n");
    out.push_str("    fn truth_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        value: Self::Value,\n");
    out.push_str("    ) -> Result<Self::Truth, pyre_interpreter::PyError> {\n");
    out.push_str("        self.truth_value_direct(value.opref, value.concrete.to_pyobj())\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn bool_value_from_truth(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        truth: Self::Truth,\n");
    out.push_str("        negate: bool,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        use crate::helpers::TraceHelperAccess;\n");
    out.push_str("        let opref = self.trace_bool_value_from_truth(truth, negate)?;\n");
    out.push_str(
        "        Ok(crate::state::FrontendOp::new(opref, crate::state::ConcreteValue::Null))\n",
    );
    out.push_str("    }\n");
    out.push_str("}\n");
}

/// Emit `impl IterOpcodeHandler for crate::state::MIFrame { ... }`.
fn emit_iter_impl(out: &mut String) {
    out.push_str("impl pyre_interpreter::IterOpcodeHandler for crate::state::MIFrame {\n");
    out.push_str("    fn ensure_iter_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        iter: Self::Value,\n");
    out.push_str("    ) -> Result<(), pyre_interpreter::PyError> {\n");
    out.push_str("        self.with_ctx(|this, ctx| {\n");
    out.push_str("            crate::state::MIFrame::guard_range_iter(this, ctx, iter.opref);\n");
    out.push_str("            Ok(())\n");
    out.push_str("        })\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn concrete_iter_continues(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        iter: Self::Value,\n");
    out.push_str("    ) -> Result<bool, pyre_interpreter::PyError> {\n");
    out.push_str("        let concrete_iter = iter.concrete.to_pyobj();\n");
    out.push_str("        crate::state::MIFrame::concrete_iter_continues(self, concrete_iter)\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn iter_next_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        iter: Self::Value,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        let concrete_iter = iter.concrete.to_pyobj();\n");
    out.push_str(
        "        crate::state::MIFrame::iter_next_value(self, iter.opref, concrete_iter)\n",
    );
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn guard_optional_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        next: Self::Value,\n");
    out.push_str("        continues: bool,\n");
    out.push_str("    ) -> Result<(), pyre_interpreter::PyError> {\n");
    out.push_str("        self.with_ctx(|this, ctx| {\n");
    out.push_str(
        "            crate::state::MIFrame::record_for_iter_guard(this, ctx, next.opref, continues);\n",
    );
    out.push_str("            Ok(())\n");
    out.push_str("        })\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str(
        "    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), pyre_interpreter::PyError> {\n",
    );
    out.push_str("        self.with_ctx(|this, ctx| {\n");
    out.push_str("            crate::state::MIFrame::set_next_instr(this, ctx, target);\n");
    out.push_str("            Ok(())\n");
    out.push_str("        })\n");
    out.push_str("    }\n");
    out.push_str("}\n");
}

/// Emit `impl LocalOpcodeHandler for crate::state::MIFrame { ... }`.
fn emit_local_impl(out: &mut String) {
    out.push_str("impl pyre_interpreter::LocalOpcodeHandler for crate::state::MIFrame {\n");
    out.push_str("    fn load_local_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        idx: usize,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        let concrete = self\n");
    out.push_str("            .sym()\n");
    out.push_str("            .concrete_locals\n");
    out.push_str("            .get(idx)\n");
    out.push_str("            .copied()\n");
    out.push_str("            .unwrap_or(crate::state::ConcreteValue::Null);\n");
    out.push_str(
        "        let opref = self.with_ctx(|this, ctx| crate::state::MIFrame::load_local_value(this, ctx, idx))?;\n",
    );
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn load_local_checked_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        idx: usize,\n");
    out.push_str("        name: &str,\n");
    out.push_str("    ) -> Result<Self::Value, pyre_interpreter::PyError> {\n");
    out.push_str("        let _ = name;\n");
    out.push_str("        let concrete = self\n");
    out.push_str("            .sym()\n");
    out.push_str("            .concrete_locals\n");
    out.push_str("            .get(idx)\n");
    out.push_str("            .copied()\n");
    out.push_str("            .unwrap_or(crate::state::ConcreteValue::Null);\n");
    out.push_str(
        "        let opref = self.with_ctx(|this, ctx| crate::state::MIFrame::load_local_value(this, ctx, idx))?;\n",
    );
    out.push_str("        if self.value_type(opref) == majit_ir::Type::Ref {\n");
    out.push_str("            self.with_ctx(|this, ctx| {\n");
    out.push_str("                crate::state::MIFrame::guard_nonnull(this, ctx, opref);\n");
    out.push_str("            });\n");
    out.push_str("        }\n");
    out.push_str("        Ok(crate::state::FrontendOp::new(opref, concrete))\n");
    out.push_str("    }\n");
    out.push_str("\n");
    out.push_str("    fn store_local_value(\n");
    out.push_str("        &mut self,\n");
    out.push_str("        idx: usize,\n");
    out.push_str("        value: Self::Value,\n");
    out.push_str("    ) -> Result<(), pyre_interpreter::PyError> {\n");
    out.push_str("        if idx < self.sym().concrete_locals.len() {\n");
    out.push_str("            self.sym_mut().concrete_locals[idx] = value.concrete;\n");
    out.push_str("        }\n");
    out.push_str("        let concrete = value.concrete;\n");
    out.push_str("        self.with_ctx(|this, ctx| {\n");
    out.push_str(
        "            crate::state::MIFrame::store_local_value(this, ctx, idx, value.opref, concrete)\n",
    );
    out.push_str("        })\n");
    out.push_str("    }\n");
    out.push_str("}\n");
}

/// Emit the simple `OpcodeHandler` trait impls (Constant, Stack, Truth,
/// Iter, Local) as Rust source code.
///
/// The output is concatenated by `pyre/pyre-jit-trace/build.rs` with the
/// raw template file (which holds the variant traits) to produce the
/// final `OUT_DIR/jit_trace_trait_impls.rs`.
pub fn emit_simple_trait_impls() -> String {
    let mut out = String::new();
    emit_constant_impl(&mut out);
    out.push('\n');
    emit_stack_impl(&mut out);
    out.push('\n');
    emit_truth_impl(&mut out);
    out.push('\n');
    emit_iter_impl(&mut out);
    out.push('\n');
    emit_local_impl(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The emit output should contain one `impl` block per simple trait,
    /// in the order matching the legacy template (Constant → Stack →
    /// Truth → Iter → Local). We assert structural anchors rather than
    /// byte-equality here because the byte-equality check lives in
    /// `pyre/pyre-jit-trace/tests/trait_impls_snapshot.rs` (which has
    /// access to the snapshot file).
    #[test]
    fn emit_contains_each_impl_block() {
        let out = emit_simple_trait_impls();
        for trait_name in [
            "ConstantOpcodeHandler",
            "StackOpcodeHandler",
            "TruthOpcodeHandler",
            "IterOpcodeHandler",
            "LocalOpcodeHandler",
        ] {
            let needle =
                format!("impl pyre_interpreter::{trait_name} for crate::state::MIFrame {{");
            assert!(
                out.contains(&needle),
                "emit_simple_trait_impls missing block for {trait_name}"
            );
        }
    }

    #[test]
    fn emit_contains_all_constant_methods() {
        let out = emit_simple_trait_impls();
        for method in [
            "fn int_constant",
            "fn bigint_constant",
            "fn float_constant",
            "fn bool_constant",
            "fn str_constant",
            "fn bytes_constant",
            "fn code_constant",
            "fn none_constant",
            "fn ellipsis_constant",
            "fn slice_constant",
        ] {
            assert!(
                out.contains(method),
                "emit_simple_trait_impls missing constant method {method}"
            );
        }
    }
}
