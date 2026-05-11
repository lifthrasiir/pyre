use super::*;

pub(super) struct Lowerer<'c> {
    pub(super) bindings: HashMap<String, Binding>,
    pub(super) statements: Vec<TokenStream>,
    /// Per-op metadata, parallel to `statements`. Populated as B.2.A.ii
    /// migrates each emit site through `emit_op`. Read by the backward
    /// walker (B.2.B). Currently sparse — only `LiveMarker` sites land.
    #[allow(dead_code)]
    pub(super) op_metadata: Vec<OpMeta>,
    pub(super) next_reg: u16,
    pub(super) next_label: u16,
    pub(super) config: Option<&'c LowererConfig>,
    pub(super) call_policies: Vec<(Vec<String>, CallPolicySpec)>,
    pub(super) inference_failure_mode: InferenceFailureMode,
    pub(super) auto_calls: bool,
    /// Prebuild tokens carried up from nested inline-helper lowerings.
    /// These get merged into the parent body's
    /// `liveness_prebuild_tokens` output so the helper's per-marker
    /// triples land in `__prebuild_jitcode_liveness_*` alongside the
    /// outer arm's triples.
    #[allow(dead_code)]
    pub(super) inline_liveness_prebuild: Vec<TokenStream>,
    /// A.2.3a fail-closed install gate signal. Set when
    /// `lower_pre_dispatch_stmts` encounters a pre-dispatch construct
    /// whose structural shape cannot be safely lowered to dispatch
    /// JitCode (currently only inner `Expr::While` that fails the
    /// EXTENDED_ARG-shape recognizer). When `Some`,
    /// `lower_dispatch_body` returns `None` so the dispatch
    /// JitCode body is empty and the runtime install gate at
    /// `codegen_state.rs:786-823` refuses to register the singleton —
    /// matching the Pre-A.2.3 codex (gpt-5.5, 2026-05-05) "fail-closed"
    /// requirement that an unrecognized inner while must NOT silently
    /// pass the existing `BC_GETARRAYITEM_GC_I`-presence gate.
    pub(super) dispatch_tainted_reason: Option<&'static str>,
    /// Name of the LHS variable that received the opcode-fetch
    /// result, set by `try_lower_opcode_fetch_stmt` when it recognises
    /// `let <name> = program[<idx>]` (or the method-call form
    /// `program.get_op(<idx>)`).  `lower_dispatch_chain` uses this name
    /// to look up the opcode reg in `bindings` so the dispatch chain
    /// emits regardless of the consumer's chosen variable name.
    /// Falls back to the literal `"opcode"` (PyPy `pyopcode.py:171`
    /// canonical name) when unset, preserving existing fixtures.
    pub(super) opcode_var_name: Option<String>,
    /// `true` when this Lowerer is producing the arm body sub-JitCode
    /// inside the dispatch JitCode (`__dispatch_jitcode_<fn>(__asm,
    /// __jdindex: i64)` — `__jdindex` is in scope here).  `false` when
    /// producing the per-arm trace JitCode (`#jitcode_fn_name(__asm,
    /// program, pc, __op)` — `__jdindex` is NOT in scope, so the
    /// `Stmt::Macro` recognition for `can_enter_jit!()` must NOT emit
    /// `__builder.loop_header(__jdindex);` to avoid a
    /// "cannot find value `__jdindex`" compile error in the consumer's
    /// macro expansion).  Set by
    /// `try_generate_jitcode_body_parts_with_caller_bindings` (the sole
    /// dispatch-arm-body lowerer entry).  Pyre's per-arm trace JitCode
    /// is a PRE-EXISTING-ADAPTATION not present in RPython, so omitting
    /// `loop_header` there is consistent with upstream's single-JitCode
    /// model where `loop_header` lives only in the dispatch-equivalent
    /// JitCode.
    pub(super) in_dispatch_arm_body: bool,
}

impl<'c> Lowerer<'c> {
    pub(super) fn new(config: Option<&'c LowererConfig>) -> Self {
        let call_policies = config.map(|cfg| cfg.calls.clone()).unwrap_or_default();
        Self::new_with_call_policies(config, call_policies, InferenceFailureMode::ReturnNone)
    }

    pub(super) fn new_with_call_policies(
        config: Option<&'c LowererConfig>,
        call_policies: Vec<(Vec<String>, CallPolicySpec)>,
        inference_failure_mode: InferenceFailureMode,
    ) -> Self {
        let mut this = Self {
            bindings: HashMap::new(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: 0,
            next_label: 0,
            config,
            call_policies,
            inference_failure_mode,
            auto_calls: config.map(|cfg| cfg.auto_calls).unwrap_or(false),
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            opcode_var_name: None,
            in_dispatch_arm_body: false,
        };
        this.install_vable_input_binding();
        this
    }

    pub(super) fn install_vable_input_binding(&mut self) {
        let Some(config) = self.config else {
            return;
        };
        let (Some(vable_var), Some(vable_reg)) =
            (config.vable_var.as_ref(), config.vable_input_ref_reg)
        else {
            return;
        };
        self.bindings.insert(
            vable_var.clone(),
            Binding {
                reg: vable_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
            },
        );
        self.next_reg = self.next_reg.max(vable_reg.saturating_add(1));
    }

    pub(super) fn vable_base_reg(&self) -> Option<u16> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let binding = self.bindings.get(vable_var)?;
        match binding.kind {
            BindingKind::Ref => Some(binding.reg),
            _ => None,
        }
    }

    pub(super) fn alloc_reg(&mut self) -> u16 {
        let reg = self.next_reg;
        self.next_reg = self.next_reg.saturating_add(1);
        reg
    }

    pub(super) fn alloc_label(&mut self) -> syn::Ident {
        let label = self.next_label;
        self.next_label = self.next_label.saturating_add(1);
        format_ident!("__jit_label_{label}")
    }

    /// Emit an op token plus its parallel `OpMeta` entry, keeping
    /// `statements` and `op_metadata` index-aligned for the backward
    /// liveness walker (Slice B.2.B).
    pub(super) fn emit_op(&mut self, meta: OpMeta, tokens: TokenStream) {
        self.statements.push(tokens);
        self.op_metadata.push(meta);
    }

    pub(super) fn append_lowered_sequence(&mut self, lowered: LoweredSequence) {
        debug_assert_eq!(
            lowered.statements.len(),
            lowered.op_metadata.len(),
            "RPython ssarepr.insns parity requires branch statements and metadata to append together"
        );
        self.statements.extend(lowered.statements);
        self.op_metadata.extend(lowered.op_metadata);
    }

    pub(super) fn emit_label_def(&mut self, label: &Ident) {
        self.emit_op(
            OpMeta::label_def(label.clone()),
            quote! { __builder.mark_label(#label); },
        );
    }

    pub(super) fn emit_jump(&mut self, target: &Ident) {
        self.emit_op(
            OpMeta::jump(target.clone()),
            quote! { __builder.jump(#target); },
        );
    }

    pub(super) fn emit_conditional_guard(&mut self, cond_reg: u16, target: &Ident) {
        // `goto_if_not_int_is_true` reads an int-banked register per
        // `assembler.py:217 'i'` argcode — encode the kind into the
        // metadata `Register` so the liveness walker keeps it under Int.
        self.emit_op(
            OpMeta::conditional_guard(Register::int(cond_reg), target.clone()),
            quote! { __builder.goto_if_not_int_is_true(#cond_reg, #target); },
        );
    }

    /// Emit a builder-side aux statement (no BC_* op, no def/use).
    pub(super) fn emit_aux(&mut self, tokens: TokenStream) {
        self.emit_op(OpMeta::aux(), tokens);
    }

    pub(super) fn inference_failure_tokens(&self, message: &str) -> TokenStream {
        match self.inference_failure_mode {
            InferenceFailureMode::ReturnNone => quote! { return None; },
            InferenceFailureMode::Panic => {
                let message = message.to_string();
                quote! { panic!(#message); }
            }
        }
    }

    pub(super) fn resolve_call_policy(&self, func: &Expr) -> Option<CallPolicySpec> {
        let func_segments = canonical_expr_segments(func)?;
        if let Some((_, policy)) = self
            .call_policies
            .iter()
            .find(|(path, _)| *path == func_segments)
        {
            return Some(policy.clone());
        }
        match self.inference_failure_mode {
            InferenceFailureMode::Panic => helper_policy_path(func).map(|_| CallPolicySpec::Infer),
            InferenceFailureMode::ReturnNone => {
                if self.auto_calls {
                    helper_policy_path(func).map(|_| CallPolicySpec::Infer)
                } else {
                    None
                }
            }
        }
    }

    /// Resolve the cond_call / record_known_result helper policy for
    /// `func`, falling back to `inferred_default` when the helper has a
    /// `helper_policy_path` but no explicit `calls={{ helper => ... }}`
    /// entry — RPython's `getcalldescr` (`call.py:282-303`) derives
    /// `extraeffect` from the call graph regardless of any user
    /// annotation, so a missing explicit policy must not crash.
    ///
    /// Returns `(kind, is_inferred)`.  The `kind` is the
    /// `CallPolicyKind` to drive expansion-time decisions (result-kind
    /// dispatch, wrapped-vs-direct registration shape, the
    /// `record_known_result!` elidable assert).  When `is_inferred` is
    /// true, the runtime `__policy` byte from the helper's
    /// `_jit_helper_policy` accessor reflects the actual analyzer
    /// outcome, and the registration code below picks the matching
    /// `EffectInfoSlot` at runtime instead of trusting the static
    /// default.
    ///
    /// Panics only when no helper-policy path exists at all — in that
    /// case the macro literally cannot register the function pointer.
    pub(super) fn cond_call_policy_or_inferred_default(
        &self,
        func: &Expr,
        macro_name: &str,
        inferred_default: crate::jit_interp::CallPolicyKind,
    ) -> (crate::jit_interp::CallPolicyKind, bool) {
        match self.resolve_call_policy(func) {
            Some(CallPolicySpec::Explicit(kind)) => (kind, false),
            Some(CallPolicySpec::Infer) => (inferred_default, true),
            None => {
                panic!(
                    "{macro_name} cannot resolve a helper policy for the callee — \
                     no `calls={{ helper => ... }}` entry and no `_jit_helper_policy` \
                     accessor on the function path"
                );
            }
        }
    }

    pub(super) fn cond_call_slot_for_policy(
        &self,
        kind: crate::jit_interp::CallPolicyKind,
        macro_name: &str,
    ) -> CondCallEffectSlot {
        call_policy_effect_slot(kind).unwrap_or_else(|| {
            panic!(
                "{macro_name} cannot lower helper policy {kind:?}: RPython \
                 jtransform.py:1677 rejects conditional_call / record_known_result \
                 callees whose calldescr forces virtuals or uses release-gil, and \
                 inline helpers do not have a direct-call calldescr"
            )
        })
    }

    pub(super) fn call_target_registration_tokens(
        &self,
        func: &Expr,
        kind: crate::jit_interp::CallPolicyKind,
        slot: CondCallEffectSlot,
        is_inferred: bool,
        inferred_policy_check: Option<TokenStream>,
    ) -> TokenStream {
        let static_slot_token = slot.token();
        // For `Infer` mode, the helper's `_jit_helper_policy` byte is
        // the macro-time stand-in for RPython's `_canraise` /
        // `_elidable_function_` / `_jit_loop_invariant_` analyzers
        // (`call.py:282-303 getcalldescr`).  Map it to the matching
        // `EffectInfoSlot` at runtime so an auto-discovered
        // `#[elidable_cannot_raise]` helper used without an explicit
        // `calls = { ... }` entry still registers an
        // `ElidableCannotRaise` slot — matching what an explicit
        // policy would have given.
        //
        // Bytes are allocated by `helper_policy_tokens_for_fn`
        // (`majit-macros/src/lib.rs`):
        //   1u8/2u8 — `dont_look_inside` Void/Int (`Plain`).
        //   3u8 — `elidable` Int.
        //   17u8/18u8 — `jit_loop_invariant` Void/Int.
        //   19u8 — `elidable_cannot_raise` Int.
        //   20u8 — `elidable_or_memerror` Int.
        //   21u8/22u8/23u8 — `elidable*` Ref.
        //   24u8 — `jit_loop_invariant` Ref.
        //   25u8 — `dont_look_inside` Ref (`Plain`).
        //   26u8 — Ref `MayForce` (rejected here). Ref `ReleaseGil` has
        //   no upstream CALL_RELEASE_GIL_R and is emitted as unsupported.
        //   9u8/10u8/13u8/14u8 — `MayForce` / `ReleaseGil` (rejected
        //   by `cond_call_slot_for_policy`'s `jtransform.py:1677`
        //   gate, but reach here at runtime — panic to match).
        // Unknown bytes (including `0u8` "unsupported") are rejected by
        // the call-site-specific inferred policy check before this slot is
        // used.  The fallback is kept only for defensive expansion.
        let slot_expr = if is_inferred {
            quote! {
                match __policy {
                    #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE => {
                        majit_metainterp::EffectInfoSlot::CanRaise
                    }
                    // `call.py:303 getcalldescr` non-elidable EF_CANNOT_RAISE
                    // (`#[dont_look_inside_cannot_raise]` opt-in for void/int/ref).
                    #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                        majit_metainterp::EffectInfoSlot::CannotRaise
                    }
                    #INT_ELIDABLE | #REF_ELIDABLE => majit_metainterp::EffectInfoSlot::ElidableCanRaise,
                    #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #REF_LOOP_INVARIANT => {
                        majit_metainterp::EffectInfoSlot::LoopInvariant
                    }
                    #INT_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_CANNOT_RAISE => {
                        majit_metainterp::EffectInfoSlot::ElidableCannotRaise
                    }
                    #INT_ELIDABLE_OR_MEMERROR | #REF_ELIDABLE_OR_MEMERROR => {
                        majit_metainterp::EffectInfoSlot::ElidableOrMemerror
                    }
                    #VOID_MAY_FORCE | #INT_MAY_FORCE | #VOID_RELEASE_GIL | #INT_RELEASE_GIL
                    | #REF_MAY_FORCE => panic!(
                        "conditional_call! / conditional_call_elidable! / record_known_result! \
                         cannot dispatch MayForce / ReleaseGil callees \
                         (jtransform.py:1677 _rewrite_op_cond_call assert)",
                    ),
                    _ => #static_slot_token,
                }
            }
        } else {
            static_slot_token
        };
        if call_policy_is_wrapped(kind) {
            let policy_path =
                helper_policy_path(func).expect("wrapped helper policy requires a path expression");
            let inferred_policy_check = inferred_policy_check.unwrap_or_else(|| quote! {});
            quote! {
                let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                #inferred_policy_check
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
                    #slot_expr,
                    __save_err,
                );
            }
        } else {
            quote! {
                let __fn_idx = __builder.add_fn_ptr_with_slot(#func as *const (), #slot_expr);
            }
        }
    }
}
