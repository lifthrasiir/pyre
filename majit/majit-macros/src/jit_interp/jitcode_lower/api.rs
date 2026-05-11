use super::*;

pub struct GeneratedJitCodeBody {
    pub body: TokenStream,
    pub liveness_prebuild: TokenStream,
    /// Slice (audit Issue #5) — green schema in declaration order,
    /// each pair is `(name, green_type_token)` where the token
    /// resolves to a `majit_ir::GreenType` variant at the install
    /// site (Int / Ref / Float / Void / Str / Unicode).  The
    /// dispatch path passes this to
    /// `JitDriver::declare_schema_typed` so
    /// `JitDriverStaticData::green_args_spec` reports STR/UNICODE
    /// where the user tagged `: str` / `: unicode` instead of
    /// collapsing to `GreenType::Ref`.  Per-arm bodies leave it
    /// empty.
    pub green_schema: Vec<(String, TokenStream)>,
    /// Red schema — `(name, ir_type_token)` resolving to a
    /// `majit_ir::Type` (Int / Ref / Float / Void).  Reds carry no
    /// upstream lltype subtype distinction (no `equal_whatever`
    /// dispatch on runtime args).
    pub red_schema: Vec<(String, TokenStream)>,
}

pub fn try_generate_jitcode_body(body: &Expr) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, None).map(|p| p.body)
}

pub fn try_generate_jitcode_body_parts(
    body: &Expr,
    _config: Option<&LowererConfig>,
) -> Option<GeneratedJitCodeBody> {
    try_generate_jitcode_body_inner(body, _config)
}

pub fn try_generate_jitcode_body_with_config(
    config: &LowererConfig,
    body: &Expr,
) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, Some(config)).map(|p| p.body)
}

pub fn try_generate_jitcode_body_with_config_parts(
    config: &LowererConfig,
    body: &Expr,
) -> Option<GeneratedJitCodeBody> {
    try_generate_jitcode_body_inner(body, Some(config))
}

/// Per-caller-local layout descriptor produced by
/// [`try_generate_jitcode_body_parts_with_caller_bindings`].
///
/// The dispatch arm parent emit needs three things to construct the
/// `inline_call_<types>_v(__sub_idx, args_i, args_r, args_f)` call:
/// - the parent's reg (read from caller's `Binding`),
/// - the callee's portal-input reg (assigned per-bank by the sub-Lowerer
///   pre-bind pass),
/// - the bank (Int / Ref / Float) so the (parent, callee) pair lands in
///   the matching `args_<kind>` vector.
#[derive(Clone, Debug)]
pub(crate) struct CallerLocalLayout {
    #[allow(dead_code)]
    pub name: String,
    pub parent_reg: u16,
    pub callee_reg: u16,
    pub kind: BindingKind,
}

/// Slice 1.2 of dispatch arm caller-local plumbing.
///
/// Variant of [`try_generate_jitcode_body_parts`] that pre-binds a
/// list of caller-locals as portal-input bindings on the sub-Lowerer
/// before lowering the body.  The caller (slice 1.3 — dispatch arm
/// emit at `lower_dispatch_chain`) collects them via
/// [`collect_arm_caller_locals`] and threads the same list through
/// `inline_call_<types>_v` as `(parent_reg, callee_reg)` pairs.
///
/// Layout convention (mirrors the existing portal-input pre-bind in
/// `lower_dispatch_body` at `:7440-7457`):
/// - per-bank packed regs starting at 0 (first Int → int_reg 0, second
///   Int → int_reg 1, …; same for Ref and Float independently);
/// - flat `next_reg` counter advanced past the highest per-bank slot
///   so subsequent `alloc_reg()` calls inside the body lowering do not
///   collide with the pre-bound caller-locals.
///
/// `pyopcode.py:179` and `jtransform.py:480` parity: PyPy's dispatch
/// inline_call passes `(opcode, oparg, ...)` as call args; the callee
/// jitcode receives them via portal-input binding slots indexed
/// per-bank.  Pyre's sub-frame uses the same `int_regs[]` / `ref_regs[]`
/// arrays per kind, so per-bank packing is the orthodox layout.
/// Layout-side helper extracted from
/// [`try_generate_jitcode_body_parts_with_caller_bindings`] so the
/// per-bank packing rule is testable without needing a body that
/// lowers cleanly.  Returns the layout descriptors plus the
/// worst-case `next_reg` advance that the caller must apply so
/// subsequent `alloc_reg()` cannot collide.
pub(crate) fn assign_caller_local_layout(
    caller_locals: &[(String, Binding)],
) -> (Vec<CallerLocalLayout>, u16) {
    let mut next_int = 0u16;
    let mut next_ref = 0u16;
    let mut next_float = 0u16;
    let mut layout = Vec::with_capacity(caller_locals.len());
    for (name, parent_binding) in caller_locals {
        let callee_reg = match parent_binding.kind {
            BindingKind::Int => {
                let r = next_int;
                next_int = next_int.saturating_add(1);
                r
            }
            BindingKind::Ref => {
                let r = next_ref;
                next_ref = next_ref.saturating_add(1);
                r
            }
            BindingKind::Float => {
                let r = next_float;
                next_float = next_float.saturating_add(1);
                r
            }
        };
        layout.push(CallerLocalLayout {
            name: name.clone(),
            parent_reg: parent_binding.reg,
            callee_reg,
            kind: parent_binding.kind,
        });
    }
    let max_pre_bound = next_int.max(next_ref).max(next_float);
    (layout, max_pre_bound)
}

pub(crate) fn try_generate_jitcode_body_parts_with_caller_bindings(
    body: &Expr,
    config: Option<&LowererConfig>,
    caller_locals: &[(String, Binding)],
) -> Option<(GeneratedJitCodeBody, Vec<CallerLocalLayout>)> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    // Sole dispatch-arm-body lowerer entry — `lower_stmt`'s `Stmt::Macro`
    // recognition for `can_enter_jit!()` emits
    // `__builder.loop_header(__jdindex);` which references the
    // `__jdindex: i64` parameter of the enclosing `__dispatch_jitcode_<fn>`
    // fn (codegen_trace.rs:309-313).  The per-arm trace JitCode lowerer
    // path (`try_generate_jitcode_body_inner` callers, including
    // `generate_jitcode_arm` at codegen_trace.rs:367) lives inside
    // `#jitcode_fn_name(__asm, program, pc, __op)` which has no
    // `__jdindex` in scope, so this flag stays `false` there and the
    // recognition gracefully returns `None` (falling back to abort).
    lowerer.in_dispatch_arm_body = true;

    let (layout, max_pre_bound) = assign_caller_local_layout(caller_locals);
    for entry in &layout {
        lowerer.bindings.insert(
            entry.name.clone(),
            Binding {
                reg: entry.callee_reg,
                kind: entry.kind,
                depends_on_stack: false,
            },
        );
    }
    // Advance the flat `next_reg` past the worst-case per-bank max so
    // body-side `alloc_reg()` cannot reuse any pre-bound slot in any
    // bank.  Mirrors the `next_reg.max(1)` advance after the
    // pc=Int(0)+program=Ref(0) pre-bind in `lower_dispatch_body`.
    lowerer.next_reg = lowerer.next_reg.max(max_pre_bound);

    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body_with_caller_bindings", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some((
        GeneratedJitCodeBody {
            body: quote! {
                #(#statements)*
            },
            liveness_prebuild,
            green_schema: Vec::new(),
            red_schema: Vec::new(),
        },
        layout,
    ))
}

pub(crate) fn generate_inline_helper_jitcode_with_calls(
    func: &ItemFn,
    calls: &[crate::jit_interp::CallEntry],
) -> syn::Result<Option<InlineHelperJitCode>> {
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "#[jit_inline] does not support generic helper functions yet",
        ));
    }

    let ReturnType::Type(_, return_ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(
            &func.sig.output,
            "#[jit_inline] requires a return type",
        ));
    };
    let return_kind = classify_param_type(return_ty).ok_or_else(|| {
        syn::Error::new_spanned(
            return_ty,
            "#[jit_inline] supports i64/isize (Int), usize/pointer (Ref), or f64 (Float) return types",
        )
    })?;

    let call_policies = calls
        .iter()
        .map(|entry| {
            let spec = match entry.policy {
                Some(kind) => CallPolicySpec::Explicit(kind),
                None => CallPolicySpec::Infer,
            };
            (canonical_path_segments(&entry.path), spec)
        })
        .collect();
    let mut lowerer =
        Lowerer::new_with_call_policies(None, call_policies, InferenceFailureMode::Panic);
    let param_layout = inline_helper_param_layout(func)?;
    let mut max_reg = 0u16;
    for (arg, (param_kind, reg)) in func.sig.inputs.iter().zip(param_layout.into_iter()) {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "#[jit_inline] parameters must be simple identifiers",
            ));
        };
        let binding_kind = match param_kind {
            InlineReturnKind::Int => BindingKind::Int,
            InlineReturnKind::Ref => BindingKind::Ref,
            InlineReturnKind::Float => BindingKind::Float,
        };
        max_reg = max_reg.max(reg.saturating_add(1));
        lowerer.bindings.insert(
            pat_ident.ident.to_string(),
            Binding {
                reg,
                kind: binding_kind,
                depends_on_stack: false,
            },
        );
    }
    lowerer.next_reg = max_reg;

    let Some(binding) = lowerer.lower_block_value(&func.block) else {
        return Ok(None);
    };

    let helper_name = func.sig.ident.to_string();
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness(&helper_name, &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Ok(Some(InlineHelperJitCode {
        body: quote! {
            #(#statements)*
        },
        return_reg: binding.reg,
        return_kind,
        liveness_prebuild,
    }))
}

pub(crate) fn inline_helper_param_layout(
    func: &ItemFn,
) -> syn::Result<Vec<(InlineReturnKind, u16)>> {
    let mut next_i = 0u16;
    let mut next_r = 0u16;
    let mut next_f = 0u16;
    let mut layout = Vec::with_capacity(func.sig.inputs.len());
    for arg in &func.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let param_kind = classify_param_type(&pat_type.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &pat_type.ty,
                "#[jit_inline] parameters must use i64/isize (Int), usize/pointer (Ref), or f64 (Float)",
            )
        })?;
        let reg = match param_kind {
            InlineReturnKind::Int => {
                let reg = next_i;
                next_i = next_i.saturating_add(1);
                reg
            }
            InlineReturnKind::Ref => {
                let reg = next_r;
                next_r = next_r.saturating_add(1);
                reg
            }
            InlineReturnKind::Float => {
                let reg = next_f;
                next_f = next_f.saturating_add(1);
                reg
            }
        };
        layout.push((param_kind, reg));
    }
    Ok(layout)
}

pub(crate) fn inline_helper_param_counts(func: &ItemFn) -> syn::Result<(u16, u16, u16)> {
    let layout = inline_helper_param_layout(func)?;
    let mut count_i = 0u16;
    let mut count_r = 0u16;
    let mut count_f = 0u16;
    for (kind, _) in layout {
        match kind {
            InlineReturnKind::Int => count_i = count_i.saturating_add(1),
            InlineReturnKind::Ref => count_r = count_r.saturating_add(1),
            InlineReturnKind::Float => count_f = count_f.saturating_add(1),
        }
    }
    Ok((count_i, count_r, count_f))
}

fn try_generate_jitcode_body_inner(
    body: &Expr,
    config: Option<&LowererConfig>,
) -> Option<GeneratedJitCodeBody> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    // RPython `compute_liveness(ssarepr) → remove_repeated_live(ssarepr)
    // → assemble()` (codewriter.py call order). `annotate_live_markers_
    // with_liveness` materialises the per-marker fixed-point alive set
    // onto each `LiveMarker.reads` so the repeated-live pass and the
    // emit-time triple rewrite both consume the same ssarepr-mutated
    // shape `liveness.py:33-79` produces.
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some(GeneratedJitCodeBody {
        body: quote! {
            #(#statements)*
        },
        liveness_prebuild,
        green_schema: Vec::new(),
        red_schema: Vec::new(),
    })
}

/// A.3.6.1 (jtransform.py:1693-1714): bind body-local `let` stmts that
/// appear in the dispatch while-body BEFORE the `jit_merge_point!()`
/// macro stmt, so that consumer-declared
/// `#[jit_interp(greens = [<body-local>])]` (e.g. aheui-jit's
/// `greens = [stackok]` with `let stackok = program.get_req_size(pc) <= ...`)
/// flow through `resolve_greens` / `emit_promote_greens` without panic.
///
/// PRE-EXISTING-ADAPTATION: RPython has no equivalent two-pass walker.
/// Its annotator-driven flowgraph SpaceOperation-lowers every stmt before
/// `jtransform.handle_jit_marker__jit_merge_point` fires, so `Variable`
/// records exist for body-locals at merge-point rewrite time. Pyre's
/// proc-macro lowerer walks `syn::Block` AST directly with limited
/// expression-lowering coverage, requiring this explicit pre-pass.
///
/// Returns `Some(())` once the `jit_merge_point!()` macro stmt is reached
/// (or the while-body ends without one). Eligibility for binding is
/// delegated to `Lowerer::lower_local`, which in turn delegates to
/// `lower_value_expr`; non-`let` stmts and `let`s with unsupported RHS
/// shapes are skipped silently here (the existing `lower_pre_dispatch_stmts`
/// post-merge-point walker / dispatch-body emit is responsible for
/// diagnostics on its own pass).
pub(super) fn bind_pre_merge_point_stmts(lowerer: &mut Lowerer, func_block: &syn::Block) -> Option<()> {
    let dispatch_match = find_dispatch_match(func_block)?;
    let loop_body = find_dispatch_loop_body(func_block, dispatch_match)?;
    for stmt in &loop_body.stmts {
        if is_jit_merge_point_macro(stmt) {
            break;
        }
        if let syn::Stmt::Local(local) = stmt {
            // lower_local delegates to lower_value_expr; failure here is
            // intentionally silent — emit_promote_greens will produce the
            // diagnostic if the green's binding is still missing.
            let _ = lowerer.lower_local(local);
        }
    }
    Some(())
}
