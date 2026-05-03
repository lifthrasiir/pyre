//! Generate `JitCode` builders and the generic `__trace_*` wrapper.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{Block, Expr, ExprBlock, ExprMatch, ItemFn, Stmt};

use super::JitInterpConfig;
use super::classify::{ArmPattern, classify_arms};
use super::jitcode_lower::{self, LowererConfig};

pub fn generate_trace_fn(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let trace_fn_name = format_ident!("__trace_{}", fn_name);
    let jitcode_fn_name = format_ident!("__jitcode_{}", fn_name);
    let prebuild_fn_name = format_ident!("__prebuild_jitcode_liveness_{}", fn_name);

    let match_expr = find_dispatch_match(&func.block);
    let Some(match_expr) = match_expr else {
        return syn::Error::new_spanned(func, "could not find opcode dispatch match")
            .to_compile_error();
    };

    // jtransform.py:596 rewrite_op_hint — detect every `hint(x, promote=
    // True)` (`x = promote(x)`, `let x = promote(x)`, `promote(x)`)
    // statement that lexically dominates the dispatch match.
    //
    // RPython rewrites each occurrence to `-live-` + `<kind>_guard_value(x)`
    // (jtransform.py:611-612) at the original CFG position; the codewriter
    // then emits those ops into every JitCode reached from that position.
    // Pyre's per-arm `JitCodeBuilder` builds each arm's JitCode
    // independently, so to preserve the semantics we splice the collected
    // promote `Stmt`s onto the head of each Lowerable arm's body and let
    // the existing `lower_promote_call` lowerer (`jitcode_lower.rs:3360`)
    // emit `-live-` + `<kind>_guard_value` per arm.
    //
    // The collector stops at the dispatch match — promotes that lexically
    // appear AFTER the match in the same containing block do NOT dominate
    // the match (they execute after each iteration's match completes), so
    // hoisting them into every arm would be a parity regression vs
    // RPython's CFG-position-bound rewrite.
    let pre_dispatch_promotes = collect_pre_dispatch_promote_stmts(&func.block, match_expr);

    let lowerer_config = LowererConfig::new(
        &config.io_shims,
        &config.calls,
        config.auto_calls,
        config.virtualizable_decl.as_ref(),
        config.state_fields.as_ref(),
    );

    let classified = classify_arms(&match_expr.arms);
    let env_type = &config.env_type;

    // RPython `pyjitpl.py:2255 finish_setup` builds every JitCode and
    // stamps every `-live-` triple into `asm.all_liveness` *before*
    // snapshotting `metainterp_sd.liveness_info`. Pyre's lazy factory
    // can't eagerly build every (pc, op), so the macro pre-registers
    // each lowered arm's per-marker liveness triples into the
    // shared assembler at install time, via the generated
    // `__prebuild_jitcode_liveness_*` function. Trace-time
    // `JitCodeBuilder::finalize_liveness(asm)` then only dedups against
    // those entries, preserving the snapshot's immutability invariant
    // (asserted in `__trace_*` below).
    let generated_arms: Vec<_> = classified
        .iter()
        .map(|arm| generate_jitcode_arm(arm, &lowerer_config, &pre_dispatch_promotes))
        .collect();
    let jitcode_arms = generated_arms.iter().map(|(arm, _)| arm);
    let liveness_prebuilds = generated_arms.iter().map(|(_, prebuild)| prebuild);

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

    let trace_fn_body = quote! {
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #trace_fn_name(
            __shared_asm: &::std::sync::Arc<::std::sync::Mutex<majit_metainterp::Assembler>>,
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
            // Lock the driver-shared `Assembler` only across the
            // `JitCode` build (which calls `finalize_liveness` to register
            // per-marker triples into `all_liveness`).  RPython does not
            // hold any assembler lock during tracing — `make_jitcodes()`
            // finishes before the metainterp starts (`pyjitpl.py:2255
            // finish_setup`).  Releasing before `trace_jitcode_observer`
            // avoids a deadlock if a recursive portal/residual callback
            // re-enters this trace path on the same driver thread.
            let __jitcode_opt = {
                let mut __asm_guard = __shared_asm
                    .lock()
                    .expect("shared_asm poisoned in __trace_* JitCode build");
                // RPython `pyjitpl.py:2255-2264` builds all jitcodes before
                // `finish_setup` snapshots `metainterp_sd.liveness_info`.
                // Runtime trace-time factory calls may rebuild/dedup, but
                // must not append new liveness entries past that snapshot.
                let __liveness_len_before = __asm_guard.all_liveness().len();
                let __jitcode_opt = #jitcode_fn_name(&mut *__asm_guard, program, __jit_pc, __op);
                assert_eq!(
                    __asm_guard.all_liveness().len(),
                    __liveness_len_before,
                    "__trace_* JitCode build grew shared_asm.all_liveness past \
                     staticdata.liveness_info snapshot — pre-build every reachable \
                     (pc, op) JitCode and call JitDriver::sync_liveness_info_from_shared_asm() \
                     before tracing starts"
                );
                __jitcode_opt
            };
            let Some(__jitcode) = __jitcode_opt else {
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
            __asm: &mut majit_metainterp::Assembler,
            program: &#env_type,
            pc: usize,
            __op: u8,
        ) -> Option<majit_metainterp::JitCode> {
            match __op {
                #(#jitcode_arms)*
            }
        }

        /// Pre-register every lowered arm's per-marker liveness triple
        /// into the driver-shared `Assembler`, mirroring RPython
        /// `pyjitpl.py:2255 finish_setup`'s "all `-live-` entries land
        /// in `asm.all_liveness` before the snapshot" invariant.
        /// Invoked from `__JitMeta::install_canonical_liveness` exactly
        /// once at install time, before
        /// `JitDriver::install_canonical_liveness` snapshots
        /// `metainterp_sd.liveness_info`.
        #[allow(non_snake_case, unused_variables, unused_mut)]
        fn #prebuild_fn_name(__asm: &mut majit_metainterp::Assembler) {
            #(#liveness_prebuilds)*
        }

        #trace_fn_body
    }
}

fn generate_jitcode_arm(
    arm: &super::classify::ClassifiedArm,
    config: &LowererConfig,
    pre_dispatch_promotes: &[Stmt],
) -> (TokenStream, TokenStream) {
    let pat = &arm.pat;
    let mut liveness_prebuild = quote! {};
    let build = match &arm.pattern {
        ArmPattern::Lowerable => {
            // RPython jtransform.py:596 / pyjitpl.py:1916: each
            // pre-dispatch `x = promote(x)` rewrites to an
            // `int_guard_value(x)` op emitted into every reached JitCode.
            // Pyre's lowerer treats `state.field = promote(state.field)`
            // structurally — `lower_promote_call` (jitcode_lower.rs:3346)
            // emits the appropriate `<kind>_guard_value` and
            // `assign_to_state_field` writes the promoted value back —
            // so prepending the original promote `Stmt`s onto each arm's
            // body yields the per-arm guard_value emission RPython produces.
            let body_with_promote =
                prepend_pre_dispatch_promotes(&arm.original_body, pre_dispatch_promotes);
            let body_for_lowering: &Expr = body_with_promote.as_ref().unwrap_or(&arm.original_body);

            // Try config-aware lowering first, fall back to basic lowering
            let code = jitcode_lower::try_generate_jitcode_body_with_config_parts(
                config,
                body_for_lowering,
            )
            .or_else(|| jitcode_lower::try_generate_jitcode_body_parts(body_for_lowering, None));

            match code {
                // RPython `assembler.py:146-158` emits a `live/<offset>`
                // marker ahead of every guard-bearing instruction during
                // codewriter assemble.  Each marker's 2-byte offset is
                // patched per-marker via `JitCodeBuilder::finalize_liveness`
                // (Phase 4 / Epic B.3-B.4 deferred-patch flow): the lowered
                // body's `live_placeholder_with_triple(li, lr, lf)` records
                // each marker's per-pc liveness triple from
                // `compute_per_marker_liveness` (B.2 walker output), and
                // the post-emit `finalize_liveness(__asm)` registers each
                // triple via `Assembler::_register_liveness_offset` and
                // rewrites the BC_LIVE 2-byte slot to point at the dedup'd
                // entry.  The dispatcher then records BC_LIVE per
                // `blackhole.py:950 bhimpl_live` and the snapshot path
                // decodes the resulting entry via
                // `MIFrame::get_list_of_active_boxes`.
                //
                // The leading `live_placeholder()` (without per-pc triple)
                // sits at the very start of the JitCode and is meant to
                // satisfy the `code[orgpc - SIZE_LIVE_OP] == op_live`
                // assertion that fires before the first lowered op.  It
                // resolves to the canonical entry (offset 0) registered by
                // `__JitMeta::install_canonical_liveness`.
                Some(generated) => {
                    let body = generated.body;
                    liveness_prebuild = generated.liveness_prebuild;
                    quote! {
                        let mut __builder = majit_metainterp::JitCodeBuilder::new();
                        let _live_offset_patch = __builder.live_placeholder();
                        #body
                        __builder.finalize_liveness(__asm);
                        Some(__builder.finish())
                    }
                }
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

    (quote! { #pat => { #build }, }, liveness_prebuild)
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

/// Collect every promote `Stmt` from `block` that lexically *dominates*
/// `target_match` (i.e., lies on every CFG path leading to the match).
///
/// Three promote forms are recognised, mirroring RPython
/// `jtransform.py:596 rewrite_op_hint`'s `hint(x, promote=True)` shape
/// in any statement context:
/// 1. `x = promote(x)` (`Stmt::Expr(Expr::Assign(...), _)`)
/// 2. `let x = promote(x)` / `let x = promote(y)`
///    (`Stmt::Local` with init = promote call)
/// 3. `promote(x);` / `promote(x)` (`Stmt::Expr(Expr::Call(...), _)`)
///
/// Collection stops at the stmt whose subtree contains `target_match`:
/// after the match dispatches, control flow re-enters via the loop
/// back-edge — promotes appearing AFTER the match in the same enclosing
/// block run BEFORE the next iteration's match but AFTER this iteration's
/// arm body.  RPython binds each `hint_promote` rewrite to its source
/// CFG position, so hoisting a post-match promote into every arm body
/// would synthesise guard_value ops at a position they never appear at
/// upstream — a parity regression over main.
fn collect_pre_dispatch_promote_stmts(block: &syn::Block, target_match: &ExprMatch) -> Vec<Stmt> {
    let mut promotes = Vec::new();
    collect_promotes_until_match(&block.stmts, target_match, &mut promotes);
    promotes
}

/// Walk `stmts` lexically; for each stmt either recurse into its
/// match-containing subtree (and STOP — anything after that dominates
/// the next loop iteration's match, not this iteration's), or scan the
/// stmt locally for promote forms.  Does not recurse into while/loop
/// bodies that don't contain `target_match` — their bodies don't run on
/// every path to the match.
fn collect_promotes_until_match(
    stmts: &[Stmt],
    target_match: &ExprMatch,
    promotes: &mut Vec<Stmt>,
) {
    for stmt in stmts {
        if stmt_contains_match(stmt, target_match) {
            recurse_into_match_containing_stmt(stmt, target_match, promotes);
            return;
        }
        scan_stmt_for_promote(stmt, promotes);
    }
}

/// Recurse INTO a stmt that contains `target_match`, walking the nested
/// block that holds the match so promotes lexically before the match
/// within that block are still collected.
fn recurse_into_match_containing_stmt(
    stmt: &Stmt,
    target_match: &ExprMatch,
    promotes: &mut Vec<Stmt>,
) {
    let inner_block = match stmt {
        Stmt::Expr(expr, _) => expr_inner_match_block(expr, target_match),
        Stmt::Local(local) => local
            .init
            .as_ref()
            .and_then(|init| expr_inner_match_block(&init.expr, target_match)),
        _ => None,
    };
    if let Some(stmts) = inner_block {
        collect_promotes_until_match(stmts, target_match, promotes);
    }
}

/// Find the inner `&[Stmt]` block of `expr` that holds `target_match`.
/// Returns the stmts of the `while`/`loop`/`block`/`if`-branch that
/// transitively contains the match, so the caller can recurse and
/// collect promotes lexically preceding the match within it.
fn expr_inner_match_block<'e>(expr: &'e Expr, target_match: &ExprMatch) -> Option<&'e [Stmt]> {
    match expr {
        Expr::Match(m) if std::ptr::eq(m, target_match) => Some(&[]),
        Expr::While(w) if block_contains_match(&w.body, target_match) => Some(&w.body.stmts),
        Expr::Loop(l) if block_contains_match(&l.body, target_match) => Some(&l.body.stmts),
        Expr::Block(b) if block_contains_match(&b.block, target_match) => Some(&b.block.stmts),
        Expr::If(i) => {
            if block_contains_match(&i.then_branch, target_match) {
                Some(&i.then_branch.stmts)
            } else if let Some((_, else_expr)) = &i.else_branch {
                expr_inner_match_block(else_expr, target_match)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// `true` iff `stmt`'s expression subtree contains `target_match`.
fn stmt_contains_match(stmt: &Stmt, target_match: &ExprMatch) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_contains_match(expr, target_match),
        Stmt::Local(local) => local
            .init
            .as_ref()
            .map(|init| expr_contains_match(&init.expr, target_match))
            .unwrap_or(false),
        _ => false,
    }
}

fn block_contains_match(block: &Block, target_match: &ExprMatch) -> bool {
    block
        .stmts
        .iter()
        .any(|s| stmt_contains_match(s, target_match))
}

fn expr_contains_match(expr: &Expr, target_match: &ExprMatch) -> bool {
    match expr {
        Expr::Match(m) => std::ptr::eq(m, target_match),
        Expr::While(w) => block_contains_match(&w.body, target_match),
        Expr::Loop(l) => block_contains_match(&l.body, target_match),
        Expr::Block(b) => block_contains_match(&b.block, target_match),
        Expr::If(i) => {
            block_contains_match(&i.then_branch, target_match)
                || i.else_branch
                    .as_ref()
                    .map(|(_, e)| expr_contains_match(e, target_match))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Scan a single stmt locally (no recursion) for any of the three
/// promote forms and push if matched.
fn scan_stmt_for_promote(stmt: &Stmt, promotes: &mut Vec<Stmt>) {
    match stmt {
        // Form 1: `x = promote(x);`
        Stmt::Expr(Expr::Assign(assign), _) => {
            if is_promote_assign_rhs(assign) {
                promotes.push(stmt.clone());
            }
        }
        // Form 2: `let x = promote(x);`
        Stmt::Local(local) => {
            if local
                .init
                .as_ref()
                .map(|init| is_promote_call_expr(&init.expr))
                .unwrap_or(false)
            {
                promotes.push(stmt.clone());
            }
        }
        // Form 3: `promote(x);` / `promote(x)`
        Stmt::Expr(expr, _) => {
            if is_promote_call_expr(expr) {
                promotes.push(stmt.clone());
            }
        }
        _ => {}
    }
}

fn is_promote_assign_rhs(assign: &syn::ExprAssign) -> bool {
    is_promote_call_expr(&assign.right)
}

fn is_promote_call_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Call(call) if is_promote_call_path(&call.func))
}

/// Build a synthetic `Expr::Block` whose body is `pre_dispatch_promotes`
/// followed by the original arm body's stmts.  Returns `None` when there
/// are no promote stmts to splice (caller falls back to `&arm.original_body`
/// directly to avoid an unnecessary clone).
fn prepend_pre_dispatch_promotes(
    original_body: &Expr,
    pre_dispatch_promotes: &[Stmt],
) -> Option<Expr> {
    if pre_dispatch_promotes.is_empty() {
        return None;
    }
    let original_stmts = match original_body {
        Expr::Block(b) => b.block.stmts.clone(),
        other => vec![Stmt::Expr(other.clone(), None)],
    };
    let mut stmts = Vec::with_capacity(pre_dispatch_promotes.len() + original_stmts.len());
    stmts.extend(pre_dispatch_promotes.iter().cloned());
    stmts.extend(original_stmts);
    Some(Expr::Block(ExprBlock {
        attrs: Vec::new(),
        label: None,
        block: Block {
            brace_token: Default::default(),
            stmts,
        },
    }))
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
