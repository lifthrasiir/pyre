use super::*;

impl<'c> Lowerer<'c> {
    pub(super) fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        // State field read (register/tape machines).
        if let Some(binding) = self.lower_state_field_read(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_state_array_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:832 — virtualizable field read rewrite.
        if let Some(binding) = self.lower_vable_field_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:760 — virtualizable array read rewrite.
        if let Some(binding) = self.lower_vable_array_read(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_vable_array_len(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:655 — suppress hint_access_directly(frame) /
        // hint_fresh_virtualizable(frame) function calls as identity.
        // These return the frame unchanged, so lower the argument instead.
        if let Some(binding) = self.lower_vable_hint_identity_call(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:1687 — conditional_call_elidable!(value, func, args...)
        if let Some(binding) = self.lower_conditional_call_elidable(expr) {
            return Some(binding);
        }

        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) => {
                let value = int_lit.base10_parse::<i64>().ok()?;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                })
            }
            Expr::Path(ExprPath { path, .. }) => {
                let ident = path.get_ident()?;
                self.bindings.get(&ident.to_string()).cloned()
            }
            Expr::Cast(ExprCast { expr, ty, .. }) if is_supported_int_cast(ty) => {
                let binding = self.lower_value_expr(expr)?;
                if !matches!(binding.kind, BindingKind::Int) {
                    return None;
                }
                Some(binding)
            }
            Expr::Paren(ExprParen { expr, .. }) => self.lower_value_expr(expr),
            Expr::If(expr_if) => self.lower_if_value(expr_if),
            Expr::Match(expr_match) => self.lower_match_value(expr_match),
            Expr::Unary(ExprUnary { op, expr, .. }) => self.lower_unary(op, expr),
            Expr::Binary(binary) => self.lower_binary(binary),
            Expr::Call(call) => {
                // jtransform.py:596 rewrite_op_hint: promote → int_guard_value
                if let Some(binding) = self.lower_promote_call(call) {
                    return Some(binding);
                }
                self.lower_call_value(call)
            }
            Expr::MethodCall(call) => self.lower_method_call_value(call),
            _ => None,
        }
    }

    /// Statement-context lowering for `hint(x, promote=True)`:
    ///
    /// - `x = promote(arg);` (plain local re-assignment — `lower_state_
    ///   field_write` falls through because LHS isn't a `state.foo` field;
    ///   without this site `stmt_modifies_jit_state` returns false and
    ///   `lower_stmt`'s config-aware fallback silently consumes the stmt
    ///   without emitting the guard).
    /// - `promote(x);` (bare statement-form — same fall-through path as
    ///   plain assign).
    ///
    /// In both forms the actual guard emission is delegated to
    /// `lower_promote_call` via `lower_value_expr` so the resulting op
    /// shape (`-live-` + `<kind>_guard_value`) is identical to the
    /// value-context lowering.  For assignment form, mirror
    /// jtransform.py:613-615: the `None` sentinel means the result is
    /// considered equal to arg0, so the LHS aliases the RHS binding
    /// (`x = promote(y)` makes `x` read from y's register).
    pub(super) fn lower_promote_stmt(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Assign(assign) => {
                let Expr::Path(lhs_path) = &*assign.left else {
                    return None;
                };
                let lhs_ident = lhs_path.path.get_ident()?.to_string();
                let Expr::Call(call) = &*assign.right else {
                    return None;
                };
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                let binding = self.lower_value_expr(&assign.right)?;
                self.bindings.insert(lhs_ident, binding);
                Some(())
            }
            Expr::Call(call) => {
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                self.lower_value_expr(expr)?;
                Some(())
            }
            _ => None,
        }
    }

    /// Lower `promote(x)` → emit `-live-` + `<kind>_guard_value(x_reg)`,
    /// return x binding.
    ///
    /// RPython: `hint(x, promote=True)` rewrites to a `-live-` marker
    /// (jtransform.py:611) immediately followed by `int_guard_value(x)`
    /// (jtransform.py:612).  The leading `-live-` pins the per-marker
    /// liveness triple at the source position so the resume protocol can
    /// rebuild the live frame state if the guard fails.  Without this
    /// pair, the snapshot path falls back to the canonical "everything-
    /// alive" entry — correct for blackhole resume but a per-marker
    /// liveness parity loss vs RPython.
    ///
    /// Blackhole: no-op (the live marker is metadata, the guard a no-op
    /// at non-trace time).  Tracing: emits GUARD_VALUE to specialize on
    /// current value with the per-pc live set saved into all_liveness.
    ///
    /// Recognizes: `promote(x)`, `hint_promote(x)`, `jit::promote(x)`.
    fn lower_promote_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if !is_promote_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        let binding = self.lower_value_expr(&call.args[0])?;
        let reg = binding.reg;
        // jtransform.py:611 — emit `-live-` before the guard op so the
        // codewriter's per-marker liveness analysis records the alive
        // set at this CFG position.  `live_placeholder_with_triple` will
        // patch the BC_LIVE 2-byte slot to the dedup'd offset at
        // `finalize_liveness` time.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { __builder.live_placeholder(); },
        );
        match binding.kind {
            BindingKind::Int => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::int(reg)], vec![]),
                    quote! { __builder.int_guard_value(#reg); },
                );
            }
            BindingKind::Ref => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::ref_(reg)], vec![]),
                    quote! { __builder.ref_guard_value(#reg); },
                );
            }
            BindingKind::Float => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::float(reg)], vec![]),
                    quote! { __builder.float_guard_value(#reg); },
                );
            }
        }
        Some(binding)
    }

    pub(super) fn lower_call_value(&mut self, call: &ExprCall) -> Option<Binding> {
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        let mut depends_on_stack = false;
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding.clone());
            depends_on_stack |= binding.depends_on_stack;
        }

        let reg = self.alloc_reg();
        let func = &call.func;
        let mut result_kind = BindingKind::Int;
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::ReleaseGilInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let canonical_call = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            quote! { residual_call_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceInt => {
                            quote! { call_may_force_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilInt => {
                            quote! { call_release_gil_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                            quote! { call_loopinvariant_int_canonical_via_target }
                        }
                        _ => unreachable!(),
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                Register::ints(&arg_regs),
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #reg,
                                );
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                __arg_regs,
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.#canonical_call(__fn_idx, #typed_args, #reg);
                            },
                        );
                    }
                }
                // `call.py:303` non-elidable EF_CANNOT_RAISE int — explicit policy.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // Parity #14 Slice C.4 + Parity #20: see the stmt-form
                    // ElidableInt arm earlier in this file for the
                    // canonical migration rationale and the 3-way
                    // `_canraise(op)` pick from `call.py:292-299`.
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceFloatWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped => {
                            quote! { __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_release_gil_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        _ => unreachable!(),
                    };
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                majit_metainterp::EffectInfoSlot::CanRaise,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::InlineInt
                | crate::jit_interp::CallPolicyKind::InlineRef
                | crate::jit_interp::CallPolicyKind::InlineFloat => {
                    result_kind = binding_kind_for_inline_policy(kind).unwrap();
                    let builder_path = inline_builder_path(&call.func)?;
                    let prebuild_path = inline_prebuild_path(&call.func)?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // RPython `pyjitpl.py:2255 finish_setup` order: the
                    // helper's per-marker `-live-` triples must land in
                    // `asm.all_liveness` before the parent's
                    // `JitDriver::install_canonical_liveness` snapshot.
                    // Queue the helper's `__majit_inline_jitcode_<name>_
                    // prebuild(__asm)` call into the parent's prebuild
                    // accumulator; `liveness_prebuild_tokens` will splice
                    // it ahead of the parent arm's own
                    // `__asm._register_liveness_offset` calls.
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __sub_jitcode = #builder_path(__asm);
                            let (__sub_return_kind, _) = __sub_jitcode
                                .trailing_return_info()
                                .expect("inline helper jitcode must end in a typed return opcode");
                            let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_op(OpMeta::live_marker(), post_live);
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                let int_arg_regs = int_arg_regs(&arg_bindings);
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy only supports int-return value calls here; use an explicit inline_ref/inline_float or *_ref_wrapped/*_float_wrapped policy",
                );
                // RPython `codewriter.py:55` precomputes per-helper
                // `-live-` triples and `pyjitpl.py:2255 finish_setup`
                // snapshots `metainterp_sd.liveness_info` after every
                // helper has had its triples registered.  When the
                // inferred path's runtime policy resolves to `4u8`
                // (`call.py` `EF_INLINE_HELPER`), the inline-helper
                // builder at line 4u8's runtime arm below executes
                // `__inline_builder(__asm)` which materialises a sub-
                // JitCode containing the helper's `-live-` markers.
                // Without an install-time prebuild call queuing into
                // `inline_liveness_prebuild`, the helper's per-marker
                // triples never enter `asm.all_liveness` ahead of the
                // parent driver's `install_canonical_liveness`
                // snapshot — diverging from the
                // `codewriter.py:55 → finish_setup` order.
                //
                // Queue the prebuild path when the helper exposes one
                // (only inline-able helpers do).  At runtime, the
                // prebuild call is idempotent — it registers triples
                // even when the runtime-resolved policy is non-inline,
                // matching the explicit `InlineInt/InlineRef/InlineFloat`
                // arms above which queue unconditionally.  Non-inline
                // helpers don't expose a `inline_prebuild_path` symbol,
                // so the `.ok()` swallow here is the parity-correct
                // "no triples to register" signal.
                if let Some(prebuild_path) = inline_prebuild_path(&call.func) {
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                }
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                if let Some(_arg_regs) = int_arg_regs {
                    // Inferred path: result is Int (the explicit-Inline case is
                    // handled at line 4u8 of the runtime match below, but the
                    // emit-time OpMeta destination still tracks the call's
                    // post-condition register slot — Int matches the typed-call
                    // helper that produces it).
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs.clone(),
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #func as *const ()
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                majit_metainterp::EffectInfoSlot::CanRaise,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:299 _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:295 _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                    let __builder_fn: fn(&mut majit_metainterp::Assembler) -> majit_metainterp::JitCode =
                                        unsafe { std::mem::transmute(__inline_builder) };
                                    let __sub_jitcode = __builder_fn(__asm);
                                    let (__sub_return_kind, _) =
                                        <majit_metainterp::JitCode as majit_metainterp::jitcode::JitCodeRuntimeExt>::trailing_return_info(&__sub_jitcode)
                                        .expect("inline helper jitcode must end in a typed return opcode");
                                    let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                    #inline_call
                                }
                                #INT_MAY_FORCE => {
                                    __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                #INT_RELEASE_GIL => {
                                    __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                #INT_LOOP_INVARIANT => {
                                    __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                _ => {
                                    #unsupported
                                }
                            }
                        },
                    );
                    // jtransform.py:467/480-482 — `inline_call_*` is always
                    // followed by `-live-`; a residual call (`call_*`)
                    // appends `-live-` only when `calldescr_canraise(calldescr)`
                    // (`call.py:295-300`).  In inferred mode the policy
                    // byte selects the calldescr at runtime, so emit the
                    // marker conditional on the can-raise codes plus the
                    // inline byte (4u8) which forces emit per
                    // jtransform.py:480-482.  LoopInvariantInt (18u8) and
                    // ElidableCannotRaiseInt (19u8) skip the marker
                    // because their calldescrs are statically cannot-raise.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_RELEASE_GIL,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live.clone(),
                    );
                } else {
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #func as *const ()
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                majit_metainterp::EffectInfoSlot::CanRaise,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:299 _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:295 _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                let __builder_fn: fn(&mut majit_metainterp::Assembler) -> majit_metainterp::JitCode =
                                    unsafe { std::mem::transmute(__inline_builder) };
                                let __sub_jitcode = __builder_fn(__asm);
                                let (__sub_return_kind, _) =
                                    <majit_metainterp::JitCode as majit_metainterp::jitcode::JitCodeRuntimeExt>::trailing_return_info(&__sub_jitcode)
                                    .expect("inline helper jitcode must end in a typed return opcode");
                                let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                #inline_call
                            }
                            #INT_MAY_FORCE => {
                                __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            #INT_RELEASE_GIL => {
                                __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            #INT_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            _ => {
                                #unsupported
                            }
                        }
                    });
                    // jtransform.py:467/480-482 — see int_arg_regs branch above.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_RELEASE_GIL,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live,
                    );
                }
            }
        }

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
        })
    }

    /// Lower an `Expr::MethodCall` by mapping the receiver ident to its
    /// canonical owning type (via `LowererConfig.state_type_name` /
    /// `env_type_name`) and dispatching through the existing call-policy
    /// table keyed on `<type>::<method>` segments. The receiver is lowered
    /// as the first call argument so the owning type's `&self` parameter
    /// gets a real register binding.
    ///
    /// Receiver-resolution policy: only the env parameter (`program`) and
    /// the state parameter (`state`) are accepted; any other receiver
    /// returns `None`. RPython `call.py:282-324 getcalldescr` keys on
    /// graph identity (no naming collision possible); pyre keys on
    /// canonical path so the `<state_type|env_type>::<method>` lookup
    /// preserves that fidelity. Arbitrary receivers cannot be resolved
    /// without the owning type identity, so they fall through.
    ///
    /// Currently only the `Elidable*` Int policy family is supported
    /// (the consumer set required for `Program::get_req_size`-shaped
    /// helpers). Wrapped / inline / non-int return policies fall through;
    /// extending them mirrors the corresponding `lower_call_value` arms
    /// when needed.
    ///
    /// RPython parity: `jtransform.py:456-470 rewrite_op` (graph-identity
    /// lookup) + `call.py:282-324 getcalldescr`.
    fn lower_method_call_value(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        let receiver_ident = match &*call.receiver {
            Expr::Path(ExprPath { path, .. }) => path.get_ident()?,
            _ => return None,
        };
        let receiver_name = receiver_ident.to_string();
        let config = self.config?;
        // Receiver-name → owning-type mapping mirrors the dispatch-portal
        // input-binding convention installed at `lower_dispatch_body`
        // (jitcode_lower.rs:6948 / :6957: `bindings.insert("program", …)`,
        // `bindings.insert("pc", …)`). Other receivers cannot be resolved
        // to a canonical owning type at macro time and fall through.
        let type_name = match receiver_name.as_str() {
            "program" => config.env_type_name.clone(),
            "state" => config.state_type_name.clone(),
            _ => return None,
        };

        // Synthesize <Type>::<method> for call-policy lookup.
        let method_segments = vec![type_name.clone(), call.method.to_string()];
        let policy = self
            .call_policies
            .iter()
            .find(|(p, _)| *p == method_segments)
            .map(|(_, spec)| spec.clone())?;
        let kind = match policy {
            CallPolicySpec::Explicit(kind) => kind,
            // Method-call inference is not supported; the policy table
            // must declare the method explicitly.
            CallPolicySpec::Infer => return None,
        };

        // Receiver counts as the first call argument; RPython
        // `jtransform.py:456 rewrite_op` similarly threads `op.args[0]`
        // (the receiver / first positional) ahead of the rest.
        if call.args.len() + 1 > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let receiver_binding = self.lower_value_expr(&call.receiver)?;
        let mut arg_bindings = Vec::with_capacity(call.args.len() + 1);
        let mut depends_on_stack = receiver_binding.depends_on_stack;
        arg_bindings.push(receiver_binding);
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            depends_on_stack |= binding.depends_on_stack;
            arg_bindings.push(binding);
        }

        // Construct the `<Type>::<method>` path tokens for `add_fn_ptr`.
        let type_ident = format_ident!("{}", type_name);
        let method_ident = &call.method;
        let func_path = quote! { <#type_ident>::#method_ident };

        let reg = self.alloc_reg();
        let result_kind = BindingKind::Int;
        match kind {
            crate::jit_interp::CallPolicyKind::ElidableInt
            | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
            | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                let call_stmt = match kind {
                    crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                        __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                    },
                    crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                        __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                    },
                    crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                        __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                    },
                    _ => unreachable!(),
                };
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Call,
                        __arg_regs,
                        vec![Register::new(result_kind, reg)],
                    ),
                    quote! {
                        let __fn_idx = __builder.add_fn_ptr(#func_path as *const ());
                        #call_stmt
                    },
                );
            }
            // Other policy kinds are not yet wired for method-call RHS.
            // Consumers needing residual / may_force / wrapped / inline
            // method-call lowering must add the corresponding arm here
            // mirroring `lower_call_value`'s shape.
            _ => return None,
        }

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
        })
    }

    fn lower_if_value(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        if let Some(binding) = self.lower_bool_if(expr_if) {
            return Some(binding);
        }

        let cond = self.lower_value_expr(&expr_if.cond)?;
        if !matches!(cond.kind, BindingKind::Int) {
            return None;
        }
        let (_, else_expr) = expr_if.else_branch.as_ref()?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        let cond_reg = cond.reg;
        let (then_seq, then_binding) =
            self.lower_branch_value_expr(&Expr::Block(syn::ExprBlock {
                attrs: Vec::new(),
                label: None,
                block: expr_if.then_branch.clone(),
            }))?;
        let (else_seq, else_binding) = self.lower_branch_value_expr(else_expr)?;
        if !matches!(then_binding.kind, BindingKind::Int)
            || !matches!(else_binding.kind, BindingKind::Int)
        {
            return None;
        }
        let then_reg = then_binding.reg;
        let else_reg = else_binding.reg;

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);
        self.append_lowered_sequence(then_seq);
        self.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(then_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! { __builder.move_i(#result_reg, #then_reg); },
        );
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);
        self.append_lowered_sequence(else_seq);
        self.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(else_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! { __builder.move_i(#result_reg, #else_reg); },
        );
        self.emit_label_def(&end_label);

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: cond.depends_on_stack
                || then_binding.depends_on_stack
                || else_binding.depends_on_stack,
        })
    }

    fn lower_bool_if(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        let (then_value, else_value) = extract_bool_branch_values(expr_if)?;
        let cond = self.lower_value_expr(&expr_if.cond)?;
        if !matches!(cond.kind, BindingKind::Int) {
            return None;
        }
        match (then_value, else_value) {
            (1, 0) => Some(cond),
            (0, 1) => {
                let zero_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(zero_reg)]),
                    quote! { __builder.load_const_i_value(#zero_reg, 0); },
                );
                let reg = self.alloc_reg();
                let cond_reg = cond.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[cond_reg, zero_reg]),
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_binop_i(#reg, majit_ir::OpCode::IntEq, #cond_reg, #zero_reg); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: cond.depends_on_stack,
                })
            }
            _ => None,
        }
    }

    fn lower_unary(&mut self, op: &UnOp, expr: &Expr) -> Option<Binding> {
        match op {
            UnOp::Neg(_) => {
                let inner = self.lower_value_expr(expr)?;
                if !matches!(inner.kind, BindingKind::Int) {
                    return None;
                }
                let reg = self.alloc_reg();
                let src_reg = inner.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        vec![Register::int(src_reg)],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::IntNeg, #src_reg); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: inner.depends_on_stack,
                })
            }
            _ => None,
        }
    }

    fn lower_binary(&mut self, expr: &ExprBinary) -> Option<Binding> {
        let lhs = self.lower_value_expr(&expr.left)?;
        let rhs = self.lower_value_expr(&expr.right)?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let opcode = opcode_for_binop(&expr.op)?;
        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            quote! { __builder.record_binop_i(#reg, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
        })
    }

    pub(super) fn lower_branch_expr(&mut self, expr: &Expr) -> Option<LoweredSequence> {
        let stmts = extract_stmts(expr);
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
        };

        for stmt in &stmts {
            nested.lower_stmt(stmt)?;
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some(LoweredSequence::new(nested.statements, nested.op_metadata))
    }

    pub(super) fn lower_branch_value_expr(&mut self, expr: &Expr) -> Option<(LoweredSequence, Binding)> {
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
        };

        let binding = nested.lower_scoped_value_expr(expr)?;
        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some((
            LoweredSequence::new(nested.statements, nested.op_metadata),
            binding,
        ))
    }

    fn lower_scoped_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        match expr {
            Expr::Block(block) => self.lower_block_value(&block.block),
            _ => self.lower_value_expr(expr),
        }
    }

    pub(super) fn lower_block_value(&mut self, block: &Block) -> Option<Binding> {
        let (tail, prefix) = block.stmts.split_last()?;

        for stmt in prefix {
            self.lower_stmt(stmt)?;
        }

        match tail {
            Stmt::Expr(expr, None) => self.lower_value_expr(expr),
            _ => None,
        }
    }
}
