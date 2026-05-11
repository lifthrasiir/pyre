use super::*;

// ── Loop control detection ───────────────────────────────────────────

/// Check if a block contains break or continue at the top level (not nested in inner loops).
pub(super) fn block_has_loop_control(block: &Block) -> bool {
    block.stmts.iter().any(|stmt| stmt_has_loop_control(stmt))
}

pub(super) fn stmt_has_loop_control(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_has_loop_control(expr),
        _ => false,
    }
}

pub(super) fn expr_has_loop_control(expr: &Expr) -> bool {
    match expr {
        Expr::Break(_) | Expr::Continue(_) => true,
        Expr::If(expr_if) => {
            block_has_loop_control(&expr_if.then_branch)
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, e)| expr_has_loop_control(e))
        }
        Expr::Block(block) => block_has_loop_control(&block.block),
        Expr::Match(m) => m.arms.iter().any(|arm| expr_has_loop_control(&arm.body)),
        // Don't recurse into nested loops — they have their own break/continue scope
        Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_) => false,
        _ => false,
    }
}

// ── Helper functions ─────────────────────────────────────────────────

/// Extract the get_mut argument from a pool.get_mut(arg) expression.
pub(super) fn extract_stmts(expr: &Expr) -> Vec<Stmt> {
    match expr {
        Expr::Block(block) => block.block.stmts.clone(),
        _ => vec![Stmt::Expr(expr.clone(), None)],
    }
}

/// Extract integer literal values from a match arm pattern.
///
/// Supports `Pat::Lit` (integer literals), `Pat::Or` (multiple patterns
/// like `1 | 2 | 3`), and `Pat::Path` (constant paths — evaluated at
/// compile time via `#pat as i64`).
///
/// Returns `None` if the pattern contains unsupported constructs.
pub(super) fn extract_pat_literals(pat: &Pat) -> Option<Vec<i64>> {
    match pat {
        Pat::Lit(expr_lit) => {
            if let Lit::Int(int_lit) = &expr_lit.lit {
                Some(vec![int_lit.base10_parse::<i64>().ok()?])
            } else {
                None
            }
        }
        Pat::Or(pat_or) => {
            let mut values = Vec::new();
            for case in &pat_or.cases {
                values.extend(extract_pat_literals(case)?);
            }
            Some(values)
        }
        // Constant path pattern (e.g., `MY_CONST`): we cannot evaluate
        // this at proc-macro time, so return None to bail out.
        _ => None,
    }
}

/// Extract pattern values as token expressions for use in generated code.
///
/// Unlike `extract_pat_literals`, this accepts constant paths (`Pat::Path`)
/// that cannot be evaluated at proc-macro time. Returns each value as a
/// `TokenStream` expression (`#path as i64` or `#lit as i64`) that is valid
/// in generated Rust code where the constant is in scope.
///
/// pyopcode.py:183+ if/elif dispatch over opcode constants (e.g. `OP_NOP`,
/// `OP_INC_A`) that are defined as symbolic constants, not inline literals.
pub(super) fn extract_pat_value_tokens(pat: &Pat) -> Option<Vec<TokenStream>> {
    match pat {
        Pat::Lit(expr_lit) => {
            if let Lit::Int(int_lit) = &expr_lit.lit {
                let val: i64 = int_lit.base10_parse().ok()?;
                Some(vec![quote! { #val as i64 }])
            } else {
                None
            }
        }
        Pat::Path(pp) => {
            let path = &pp.path;
            Some(vec![quote! { #path as i64 }])
        }
        // Bare identifier in a match arm (e.g. `OP_NOP`): syn 2 parses
        // unqualified constants the same as binding patterns. Emit
        // `#ident as i64`; the Rust compiler resolves whether it is a
        // constant or a binding at compile time. The caller (`lower_dispatch_chain`)
        // skips `Pat::Wild` and delegates catch-all arms via the default
        // label, so a bare ident reaching this branch is always a constant.
        Pat::Ident(pi) if pi.subpat.is_none() && pi.mutability.is_none() && pi.by_ref.is_none() => {
            let ident = &pi.ident;
            Some(vec![quote! { #ident as i64 }])
        }
        Pat::Or(pat_or) => {
            let mut tokens = Vec::new();
            for case in &pat_or.cases {
                tokens.extend(extract_pat_value_tokens(case)?);
            }
            Some(tokens)
        }
        _ => None,
    }
}

pub(super) fn int_arg_regs(bindings: &[Binding]) -> Option<Vec<u16>> {
    bindings
        .iter()
        .map(|binding| match binding.kind {
            BindingKind::Int => Some(binding.reg),
            BindingKind::Ref | BindingKind::Float => None,
        })
        .collect()
}

pub(super) fn inline_int_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let mut next_idx = 0u16;
    let args = bindings.iter().filter_map(|binding| match binding.kind {
        BindingKind::Int => {
            let reg = binding.reg;
            let idx = next_idx;
            next_idx = next_idx.saturating_add(1);
            Some(quote! { (#reg, #idx) })
        }
        BindingKind::Ref | BindingKind::Float => None,
    });
    quote! { &[#(#args),*] }
}

pub(super) fn inline_ref_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let mut next_idx = 0u16;
    let args = bindings.iter().filter_map(|binding| match binding.kind {
        BindingKind::Ref => {
            let reg = binding.reg;
            let idx = next_idx;
            next_idx = next_idx.saturating_add(1);
            Some(quote! { (#reg, #idx) })
        }
        BindingKind::Int | BindingKind::Float => None,
    });
    quote! { &[#(#args),*] }
}

pub(super) fn inline_float_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let mut next_idx = 0u16;
    let args = bindings.iter().filter_map(|binding| match binding.kind {
        BindingKind::Float => {
            let reg = binding.reg;
            let idx = next_idx;
            next_idx = next_idx.saturating_add(1);
            Some(quote! { (#reg, #idx) })
        }
        BindingKind::Int | BindingKind::Ref => None,
    });
    quote! { &[#(#args),*] }
}

/// Returns `(call_match, post_live)` so the caller can register the
/// `BC_INLINE_CALL` and the trailing `BC_LIVE` marker as two distinct
/// `OpMeta` entries (RPython `jtransform.py:480-481` emits inline_call
/// followed by `-live-`).
pub(super) fn inline_call_tokens(bindings: &[Binding], result_reg: u16) -> (TokenStream, TokenStream) {
    let args_i = inline_int_arg_tokens(bindings);
    let args_r = inline_ref_arg_tokens(bindings);
    let args_f = inline_float_arg_tokens(bindings);
    let has_int_args = bindings
        .iter()
        .any(|binding| matches!(binding.kind, BindingKind::Int));
    let has_float_args = bindings
        .iter()
        .any(|binding| matches!(binding.kind, BindingKind::Float));

    let call_i = if has_float_args {
        quote! {
            __builder.inline_call_irf_i(
                __sub_idx,
                #args_i,
                #args_r,
                #args_f,
                Some(#result_reg),
            );
        }
    } else if has_int_args {
        quote! {
            __builder.inline_call_ir_i(
                __sub_idx,
                #args_i,
                #args_r,
                Some(#result_reg),
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_i(
                __sub_idx,
                #args_r,
                Some(#result_reg),
            );
        }
    };
    let call_r = if has_float_args {
        quote! {
            __builder.inline_call_irf_r(
                __sub_idx,
                #args_i,
                #args_r,
                #args_f,
                Some(#result_reg),
            );
        }
    } else if has_int_args {
        quote! {
            __builder.inline_call_ir_r(
                __sub_idx,
                #args_i,
                #args_r,
                Some(#result_reg),
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_r(
                __sub_idx,
                #args_r,
                Some(#result_reg),
            );
        }
    };
    let call_f = quote! {
        __builder.inline_call_irf_f(
            __sub_idx,
            #args_i,
            #args_r,
            #args_f,
            Some(#result_reg),
        );
    };

    let call_match = quote! {
        match __sub_return_kind {
            majit_metainterp::JitArgKind::Int => {
                #call_i
            }
            majit_metainterp::JitArgKind::Ref => {
                #call_r
            }
            majit_metainterp::JitArgKind::Float => {
                #call_f
            }
        }
    };
    // RPython jtransform.py:480-481 emits inline_call_* followed
    // immediately by -live-.  Parent-frame resume snapshots rely on
    // BC_INLINE_CALL leaving frame.pc at this post-call LIVE marker
    // before opencoder.py:create_snapshot calls
    // get_list_of_active_boxes(in_a_call=True).
    let post_live = quote! { let _ = __builder.live_placeholder(); };
    (call_match, post_live)
}

pub(super) fn typed_call_arg_tokens(bindings: &[Binding]) -> TokenStream {
    let args = bindings.iter().map(|binding| {
        let reg = binding.reg;
        match binding.kind {
            BindingKind::Int => quote! { majit_metainterp::JitCallArg::int(#reg) },
            BindingKind::Ref => quote! { majit_metainterp::JitCallArg::reference(#reg) },
            BindingKind::Float => quote! { majit_metainterp::JitCallArg::float(#reg) },
        }
    });
    quote! { &[#(#args),*] }
}

pub(super) fn is_supported_int_cast(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            type_path.path.is_ident("i64")
                || type_path.path.is_ident("isize")
                || type_path.path.is_ident("i32")
                || type_path.path.is_ident("u32")
                || type_path.path.is_ident("i16")
                || type_path.path.is_ident("u16")
                || type_path.path.is_ident("i8")
                || type_path.path.is_ident("u8")
        }
        _ => false,
    }
}

pub(super) fn is_supported_ref_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("usize"),
        Type::Ptr(_) => true,
        _ => false,
    }
}

pub(super) fn is_supported_float_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("f64"),
        _ => false,
    }
}

pub(crate) fn classify_param_type(ty: &Type) -> Option<InlineReturnKind> {
    if is_supported_int_cast(ty) {
        Some(InlineReturnKind::Int)
    } else if is_supported_ref_type(ty) {
        Some(InlineReturnKind::Ref)
    } else if is_supported_float_type(ty) {
        Some(InlineReturnKind::Float)
    } else {
        None
    }
}

pub(super) fn extract_bool_branch_values(expr_if: &ExprIf) -> Option<(i64, i64)> {
    let then_value = extract_block_tail_int(&expr_if.then_branch)?;
    let (_, else_expr) = expr_if.else_branch.as_ref()?;
    let else_value = extract_branch_int(else_expr)?;
    Some((then_value, else_value))
}

pub(super) fn extract_block_tail_int(block: &Block) -> Option<i64> {
    match block.stmts.as_slice() {
        [Stmt::Expr(expr, None)] => extract_branch_int(expr),
        _ => None,
    }
}

pub(super) fn extract_branch_int(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(int_lit),
            ..
        }) => int_lit.base10_parse::<i64>().ok(),
        Expr::Paren(ExprParen { expr, .. }) => extract_branch_int(expr),
        Expr::Block(block) => extract_block_tail_int(&block.block),
        _ => None,
    }
}

pub(super) fn inline_builder_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}_with_asm", last.ident);
    Some(path)
}

/// Construct the path of the per-helper liveness prebuild fn that
/// `#[jit_inline]` generates alongside `_with_asm`. The parent
/// `#[jit_interp]` calls this from its
/// `__prebuild_jitcode_liveness_*` so the helper's per-marker
/// triples land in `asm.all_liveness` before
/// `metainterp_sd.liveness_info` snapshot, matching RPython
/// `pyjitpl.py:2255 finish_setup` order.
pub(super) fn inline_prebuild_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}_prebuild", last.ident);
    Some(path)
}

pub(super) fn binding_kind_for_inline_policy(kind: crate::jit_interp::CallPolicyKind) -> Option<BindingKind> {
    match kind {
        crate::jit_interp::CallPolicyKind::InlineInt => Some(BindingKind::Int),
        crate::jit_interp::CallPolicyKind::InlineRef => Some(BindingKind::Ref),
        crate::jit_interp::CallPolicyKind::InlineFloat => Some(BindingKind::Float),
        _ => None,
    }
}

pub(crate) fn helper_policy_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_call_policy_{}", last.ident);
    Some(path)
}

pub(super) fn opcode_for_binop(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::Add(_) => "IntAdd",
        BinOp::Sub(_) => "IntSub",
        BinOp::Mul(_) => "IntMul",
        BinOp::Div(_) => "IntFloorDiv",
        BinOp::Rem(_) => "IntMod",
        BinOp::BitAnd(_) => "IntAnd",
        BinOp::BitOr(_) => "IntOr",
        BinOp::BitXor(_) => "IntXor",
        BinOp::Shl(_) => "IntLshift",
        BinOp::Shr(_) => "IntRshift",
        BinOp::Eq(_) => "IntEq",
        BinOp::Ne(_) => "IntNe",
        BinOp::Lt(_) => "IntLt",
        BinOp::Le(_) => "IntLe",
        BinOp::Gt(_) => "IntGt",
        BinOp::Ge(_) => "IntGe",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

pub(super) fn opcode_for_assign_binop(op: &BinOp) -> Option<Ident> {
    let name = match op {
        BinOp::AddAssign(_) => "IntAdd",
        BinOp::SubAssign(_) => "IntSub",
        BinOp::MulAssign(_) => "IntMul",
        BinOp::DivAssign(_) => "IntFloorDiv",
        BinOp::RemAssign(_) => "IntMod",
        BinOp::BitAndAssign(_) => "IntAnd",
        BinOp::BitOrAssign(_) => "IntOr",
        BinOp::BitXorAssign(_) => "IntXor",
        BinOp::ShlAssign(_) => "IntLshift",
        BinOp::ShrAssign(_) => "IntRshift",
        _ => return None,
    };
    Some(Ident::new(name, proc_macro2::Span::call_site()))
}

