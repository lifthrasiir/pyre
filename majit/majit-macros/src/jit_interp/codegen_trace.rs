//! Generate `JitCode` builders and the generic `__trace_*` wrapper.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use super::JitInterpConfig;
use super::classify::{ArmPattern, classify_arms};
use super::jitcode_lower::{self, LowererConfig};

pub fn generate_trace_fn(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let trace_fn_name = format_ident!("__trace_{}", fn_name);
    let jitcode_fn_name = format_ident!("__jitcode_{}", fn_name);

    let match_expr = find_dispatch_match(&func.block);
    let Some(match_expr) = match_expr else {
        return syn::Error::new_spanned(func, "could not find opcode dispatch match")
            .to_compile_error();
    };

    // jtransform.py:596 rewrite_op_hint — detect hint_promote() calls in
    // pre-dispatch code.  When present, the trace function emits GUARD_VALUE
    // before each arm's JitCode (pyjitpl.py:1916 implement_guard_value).
    let has_pre_dispatch_promote = has_promote_before_match(&func.block);
    let promote_preamble = quote! {}; // arm preamble not used; promote goes in trace fn

    let lowerer_config = LowererConfig::new(
        &config.io_shims,
        &config.calls,
        config.auto_calls,
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
    );

    let classified = classify_arms(&match_expr.arms);
    let env_type = &config.env_type;

    let jitcode_arms = classified
        .iter()
        .map(|arm| generate_jitcode_arm(arm, &lowerer_config, &promote_preamble));

    let label_closure = quote! { |_unused_pc| 0usize };
    let trace_jitcode_call = if config.virtualizable_decl.is_some() {
        quote! {
            let Some(__vable_argbox) = __ctx.standard_virtualizable_jitcode_argbox() else {
                return TraceAction::Abort;
            };
            let __jitcode_args = [__vable_argbox];
            let __result = majit_metainterp::trace_jitcode_observer_with_args(
                __ctx,
                __sym,
                &__jitcode,
                pc,
                #label_closure,
                &__jitcode_args,
            );
        }
    } else {
        quote! {
            let __result = majit_metainterp::trace_jitcode_observer(
                __ctx,
                __sym,
                &__jitcode,
                pc,
                #label_closure,
            );
        }
    };

    let _ = has_pre_dispatch_promote;
    let trace_fn_body = quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #trace_fn_name(
            __ctx: &mut majit_metainterp::TraceCtx,
            __sym: &mut __JitSym,
            program: &#env_type,
            pc: usize,
        ) -> majit_metainterp::TraceAction {
            use majit_metainterp::TraceAction;

            let __op = program.get_op(pc);
            // The lowered JitCode must see the same local state as the
            // interpreter's `match opcode { ... }` body: the opcode has
            // already been fetched and `pc` has already advanced past it.
            // RPython tracing observes the post-fetch bytecode index when
            // recording immediate operands, so pass `pc + 1` here instead
            // of the opcode's address.
            let __jit_pc = pc + 1;
            let Some(__jitcode) = #jitcode_fn_name(program, __jit_pc, __op) else {
                if majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit] no jitcode for pc={} op={}",
                        pc,
                        __op
                    );
                }
                return TraceAction::AbortPermanent;
            };

            // Observer mode: the outer Rust mainloop runs the same opcode
            // body alongside this metainterp pass. The metainterp executes
            // each residual function-pointer call (BC_CALL_INT /
            // BC_RESIDUAL_CALL_VOID etc.) and pushes (func, args[, result])
            // onto OBSERVED_CALLS; the outer body, rewritten by `rewrite_body`
            // so each registered helper is wrapped in `consume_observed_*_call`,
            // replays the queued result instead of invoking the helper a
            // second time. The IR call op recorded above runs at compiled-
            // trace runtime; the outer/metainterp pair stays single-execution
            // per recording iter.
            #trace_jitcode_call
            if majit_metainterp::majit_log_enabled() && !matches!(__result, TraceAction::Continue) {
                eprintln!(
                    "[jit] trace action at pc={} op={} -> {:?}",
                    pc,
                    __op,
                    __result
                );
            }
            __result
        }
    };

    quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #jitcode_fn_name(
            program: &#env_type,
            pc: usize,
            __op: u8,
        ) -> Option<majit_metainterp::JitCode> {
            match __op {
                #(#jitcode_arms)*
            }
        }

        #trace_fn_body
    }
}

fn generate_jitcode_arm(
    arm: &super::classify::ClassifiedArm,
    config: &LowererConfig,
    promote_preamble: &TokenStream,
) -> TokenStream {
    let pat = &arm.pat;
    let build = match &arm.pattern {
        ArmPattern::Lowerable => {
            // Try config-aware lowering first, fall back to basic lowering
            let code =
                jitcode_lower::try_generate_jitcode_body_with_config(config, &arm.original_body)
                    .or_else(|| jitcode_lower::try_generate_jitcode_body(&arm.original_body));

            match code {
                // RPython `assembler.py:146-158` emits a `live/<offset>`
                // marker ahead of every guard-bearing instruction during
                // codewriter assemble.  The placeholder offset (0) points
                // at the per-`__JitMeta` canonical liveness entry installed
                // by `__JitMeta::install_canonical_liveness`; the dispatcher
                // records BC_LIVE per `blackhole.py:950 bhimpl_live`, and
                // the snapshot path decodes that entry via
                // `MIFrame::get_list_of_active_boxes`.
                Some(code) => quote! {
                    let mut __builder = majit_metainterp::JitCodeBuilder::new();
                    let _live_offset_patch = __builder.live_placeholder();
                    #promote_preamble
                    #code
                    Some(__builder.finish())
                },
                None => quote! { None },
            }
        }
        ArmPattern::AbortPermanent | ArmPattern::Halt => quote! {
            let mut __builder = majit_metainterp::JitCodeBuilder::new();
            __builder.abort_permanent();
            Some(__builder.finish())
        },
        ArmPattern::Nop => quote! {
            Some(majit_metainterp::JitCodeBuilder::new().finish())
        },
        ArmPattern::Unsupported(_reason) => {
            // Complex CFG (loop/while/for in match arm) cannot be lowered to
            // JitCode. Instead of compile_error!, emit an abort bytecode so
            // tracing falls back to the interpreter — matching RPython's
            // dont_look_inside behavior for complex code patterns.
            quote! {
                Some({
                    let mut builder = majit_metainterp::JitCodeBuilder::new();
                    builder.abort();
                    builder.finish()
                })
            }
        }
    };

    quote! { #pat => { #build }, }
}

fn find_dispatch_match(block: &syn::Block) -> Option<&syn::ExprMatch> {
    for stmt in &block.stmts {
        if let Some(m) = find_match_in_stmt(stmt) {
            return Some(m);
        }
    }
    None
}

fn find_match_in_stmt(stmt: &syn::Stmt) -> Option<&syn::ExprMatch> {
    match stmt {
        syn::Stmt::Expr(expr, _) => find_match_in_expr(expr),
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                find_match_in_expr(&init.expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn find_match_in_expr(expr: &syn::Expr) -> Option<&syn::ExprMatch> {
    match expr {
        syn::Expr::Match(m) => Some(m),
        syn::Expr::While(w) => {
            for stmt in &w.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Loop(l) => {
            for stmt in &l.body.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::Block(b) => {
            for stmt in &b.block.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            None
        }
        syn::Expr::If(i) => {
            for stmt in &i.then_branch.stmts {
                if let Some(m) = find_match_in_stmt(stmt) {
                    return Some(m);
                }
            }
            if let Some((_, else_expr)) = &i.else_branch {
                return find_match_in_expr(else_expr);
            }
            None
        }
        _ => None,
    }
}

/// Detect whether the function body contains `promote()` calls before
/// the dispatch match.  Returns `true` as a literal for codegen.
///
/// RPython: jtransform.py:596 — `hint(x, promote=True)` becomes
/// `int_guard_value(x)`.  When detected, the trace function emits
/// GUARD_VALUE via `int_guard_value` before each arm's JitCode.
fn has_promote_before_match(block: &syn::Block) -> bool {
    let mut promotes = Vec::new();
    collect_promote_stmts(&block.stmts, &mut promotes);
    !promotes.is_empty()
}

/// Collect variable names from `x = promote(x)` patterns in statements.
fn collect_promote_stmts(stmts: &[syn::Stmt], promotes: &mut Vec<String>) {
    for stmt in stmts {
        match stmt {
            syn::Stmt::Expr(syn::Expr::While(w), _) => {
                collect_promote_stmts(&w.body.stmts, promotes);
            }
            syn::Stmt::Expr(syn::Expr::Loop(l), _) => {
                collect_promote_stmts(&l.body.stmts, promotes);
            }
            syn::Stmt::Expr(syn::Expr::Assign(assign), _) => {
                if let Some(name) = extract_promote_assign(assign) {
                    promotes.push(name);
                }
            }
            _ => {}
        }
    }
}

/// Check if `expr` is `x = promote(x)` (or legacy `hint_promote(x)`) and return
/// the variable name.
fn extract_promote_assign(assign: &syn::ExprAssign) -> Option<String> {
    let syn::Expr::Call(call) = &*assign.right else {
        return None;
    };
    if !is_promote_call_path(&call.func) {
        return None;
    }
    let syn::Expr::Path(lhs_path) = &*assign.left else {
        return None;
    };
    Some(lhs_path.path.get_ident()?.to_string())
}

/// Check if a call expression's function path is a promote call.
///
/// Matches: `promote`, `hint_promote`, `jit::promote`,
/// `majit_metainterp::jit::promote`.
pub(crate) fn is_promote_call_path(func: &syn::Expr) -> bool {
    let syn::Expr::Path(func_path) = func else {
        return false;
    };
    let segments: Vec<_> = func_path
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    match segments.as_slice() {
        [name] => name == "promote" || name == "hint_promote",
        [ns, name] => name == "promote" && ns == "jit",
        [_, ns, name] => name == "promote" && ns == "jit",
        _ => false,
    }
}
