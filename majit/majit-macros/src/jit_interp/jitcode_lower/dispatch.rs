use super::*;

#[cfg(test)]
mod find_dispatch_loop_body_tests {
    use super::*;

    fn fn_block_from(src: &str) -> syn::Block {
        let item: syn::ItemFn = syn::parse_str(&format!("fn f() {{ {} }}", src)).unwrap();
        *item.block
    }

    fn first_match(block: &syn::Block) -> &syn::ExprMatch {
        fn find_in(stmt: &syn::Stmt) -> Option<&syn::ExprMatch> {
            match stmt {
                syn::Stmt::Expr(syn::Expr::Match(m), _) => Some(m),
                syn::Stmt::Expr(syn::Expr::While(w), _) => w.body.stmts.iter().find_map(find_in),
                syn::Stmt::Expr(syn::Expr::Loop(l), _) => l.body.stmts.iter().find_map(find_in),
                _ => None,
            }
        }
        block
            .stmts
            .iter()
            .find_map(find_in)
            .expect("no match in block")
    }

    #[test]
    fn finds_while_body() {
        let blk = fn_block_from("while x < 10 { match op { 0 => {}, _ => {} } }");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_some());
    }

    #[test]
    fn finds_loop_body() {
        let blk = fn_block_from("loop { match op { 0 => {}, _ => break } }");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_some());
    }

    #[test]
    fn returns_none_when_no_loop() {
        let blk = fn_block_from("match op { 0 => {}, _ => {} };");
        let m = first_match(&blk);
        assert!(find_dispatch_loop_body(&blk, m).is_none());
    }
}

/// Walk a dispatch arm body and collect the parent-scope idents it
/// references, paired with the parent's [`Binding`] for each.
///
/// `pyopcode.py:179` keeps `oparg` / `next_instr` etc. as flowgraph
/// variables shared between the dispatch loop and the per-opcode
/// handler bodies; `jtransform.py:480 inline_call_<types>(jitcode,
/// args...)` then threads those variables as call args so the callee
/// jitcode sees them via its portal-input bindings.  Pyre's dispatch
/// arm lowerer emits `__builder.inline_call(__sub_idx)` with no args,
/// leaving the sub-frame without `pc` / `program` / `op` etc. — any
/// arm body that references these (e.g. `program.get_operand(pc - 1)`
/// in aheui-jit) lowers to raw Rust that fails to compile inside
/// `__dispatch_jitcode_<fn>` where the names are out of scope.
///
/// This collector is the first slice (of three) closing that gap:
///   1. (here) identify the caller-locals the arm body references;
///   2. wire the sub-Lowerer to pre-bind those names as portal-input
///      bindings so `lower_value_expr` can resolve them;
///   3. emit `inline_call_<types>_v(__sub_idx, args...)` with the
///      caller's regs paired against the callee's portal-input slots.
///
/// **Field/method recognition**: `state.selected` only contributes
/// `state` (skipped because state isn't a parent binding); `expr.field`
/// only visits `expr`; `expr.method(args)` only visits the receiver +
/// args, never the method ident.  Multi-segment paths (`lj::stack_add`)
/// are skipped — only single-segment idents can match a parent binding.
///
/// **Local-binding suppression**: `let x = expr;` adds `x` to a local
/// suppression set so subsequent `x` references in the same arm are NOT
/// reported as caller-locals (the local shadows the caller scope).
/// Scope tracking is intentionally flat — over-suppression in nested
/// blocks is acceptable since the consumer (slice 2) just gets fewer
/// args, which is safe.  Under-suppression (let in inner block missed
/// by outer scope) is fine — sub-Lowerer ignores extra portal inputs.
/// Collect every identifier the pattern would bind, recursing into
/// `Pat::TupleStruct(OP_PUSH(value))` / `Pat::Tuple((a, b))` /
/// `Pat::Struct(Foo { x })` / `Pat::Or(A | B)` / `Pat::Reference(&x)`
/// /etc. so the dispatch arm pattern's bound names — distinct
/// flowgraph variables in PyPy `flowspace`/`SpaceOperation` parlance —
/// are shadowed in caller-local probes (`collect_arm_caller_locals`)
/// and runtime-constant-fallback gates (`expr_references_any_binding`'s
/// `visit_expr_match` arm-scope handling).
pub(super) fn collect_pat_bound_idents(pat: &Pat, out: &mut HashSet<String>) {
    match pat {
        Pat::Ident(pi) => {
            out.insert(pi.ident.to_string());
            if let Some((_, sub)) = &pi.subpat {
                collect_pat_bound_idents(sub, out);
            }
        }
        Pat::TupleStruct(ts) => {
            for p in &ts.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Tuple(tu) => {
            for p in &tu.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Struct(ps) => {
            for field in &ps.fields {
                collect_pat_bound_idents(&field.pat, out);
            }
        }
        Pat::Or(po) => {
            // PyPy `pyopcode.py:179` arm patterns carrying `Or`
            // (`A | B`) require both alternatives to bind the same
            // names; visiting any one suffices, but visiting all
            // is also safe — names dedupe in the HashSet.
            for p in &po.cases {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Reference(pr) => collect_pat_bound_idents(&pr.pat, out),
        Pat::Slice(ss) => {
            for p in &ss.elems {
                collect_pat_bound_idents(p, out);
            }
        }
        Pat::Type(pt) => collect_pat_bound_idents(&pt.pat, out),
        Pat::Paren(pp) => collect_pat_bound_idents(&pp.pat, out),
        Pat::Range(_)
        | Pat::Lit(_)
        | Pat::Path(_)
        | Pat::Wild(_)
        | Pat::Rest(_)
        | Pat::Const(_)
        | Pat::Macro(_)
        | Pat::Verbatim(_) => {}
        // syn::Pat is non_exhaustive — be conservative on future
        // additions (no bindings extracted).
        _ => {}
    }
}

pub(super) fn collect_arm_caller_locals(
    arm_body: &syn::Expr,
    arm_pat: &Pat,
    parent_bindings: &HashMap<String, Binding>,
) -> Vec<(String, Binding)> {
    use syn::visit::Visit;

    struct Collector<'a> {
        parent_bindings: &'a HashMap<String, Binding>,
        local_binds: HashSet<String>,
        visited: HashSet<String>,
        result: Vec<(String, Binding)>,
    }

    impl<'ast> Visit<'ast> for Collector<'_> {
        fn visit_expr_path(&mut self, p: &'ast ExprPath) {
            if p.qself.is_some() || p.path.segments.len() != 1 {
                return;
            }
            let seg = &p.path.segments[0];
            if !seg.arguments.is_none() {
                return;
            }
            let name = seg.ident.to_string();
            if self.local_binds.contains(&name) {
                return;
            }
            if !self.visited.insert(name.clone()) {
                return;
            }
            if let Some(binding) = self.parent_bindings.get(&name) {
                self.result.push((name, binding.clone()));
            }
        }

        fn visit_expr_field(&mut self, ef: &'ast syn::ExprField) {
            // Only the base expression is a user expression; the field
            // member ident is a struct-layout name, not a free var.
            self.visit_expr(&ef.base);
        }

        fn visit_expr_method_call(&mut self, mc: &'ast ExprMethodCall) {
            self.visit_expr(&mc.receiver);
            // Skip `mc.method` (the method ident) and `mc.turbofish`
            // (which carries syn::AngleBracketedGenericArguments — pure
            // type machinery, never references caller-scope idents).
            for arg in &mc.args {
                self.visit_expr(arg);
            }
        }

        fn visit_local(&mut self, local: &'ast Local) {
            // Visit the RHS first so any caller-local it references is
            // collected before the binding name shadows them.
            if let Some(init) = &local.init {
                self.visit_expr(&init.expr);
                // `let X = expr else { … };` else branch can use names
                // from the outer scope — visit it before adding `X`.
                if let Some((_, diverge)) = &init.diverge {
                    self.visit_expr(diverge);
                }
            }
            // Record the bound name(s) so subsequent `X` references in
            // the same flat scope are treated as locally-shadowed.
            collect_pat_bound_idents(&local.pat, &mut self.local_binds);
            // No auto-recursion fallback: we already visited init above.
        }
    }

    let mut collector = Collector {
        parent_bindings,
        local_binds: HashSet::new(),
        visited: HashSet::new(),
        result: Vec::new(),
    };
    // Pre-populate `local_binds` with names bound by the arm pattern
    // (`OP_PUSH(value)` binds `value`, `OP_TUPLE(a, b)` binds `a`/`b`,
    // etc.) so the visitor treats them as locally shadowed instead of
    // as free idents that could falsely match parent bindings.
    collect_pat_bound_idents(arm_pat, &mut collector.local_binds);
    collector.visit_expr(arm_body);
    collector.result
}

#[cfg(test)]
mod collect_arm_caller_locals_tests {
    use super::*;

    fn parent_bindings(entries: &[(&str, BindingKind, u16)]) -> HashMap<String, Binding> {
        entries
            .iter()
            .map(|(name, kind, reg)| {
                (
                    (*name).to_string(),
                    Binding {
                        reg: *reg,
                        kind: *kind,
                        depends_on_stack: false,
                    },
                )
            })
            .collect()
    }

    fn arm_body(src: &str) -> syn::Expr {
        // The collector takes an Expr. Wrap as a block expression so
        // multi-stmt arm bodies (let + tail expr) parse cleanly.
        syn::parse_str::<syn::Expr>(src).expect("arm body must parse as Expr")
    }

    /// Default arm pattern for tests that don't exercise pattern
    /// bindings — `_` wildcard binds nothing.
    fn wildcard_pat() -> syn::Pat {
        syn::parse_quote!(_)
    }

    fn names(result: &[(String, Binding)]) -> Vec<String> {
        result.iter().map(|(n, _)| n.clone()).collect()
    }

    #[test]
    fn collects_single_ident_match_in_method_call_args() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 0),
            ("program", BindingKind::Ref, 0),
        ]);
        let body = arm_body("{ program.get_operand(pc - 1) }");
        let pat = wildcard_pat();
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["pc".to_string(), "program".to_string()]);
    }

    #[test]
    fn skips_field_member_idents() {
        let bindings = parent_bindings(&[("state", BindingKind::Ref, 0)]);
        let body = arm_body("{ state.selected }");
        let pat = wildcard_pat();
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        // `selected` is a field name, not a parent binding — only
        // `state` should be picked up (and it IS in parent_bindings).
        assert_eq!(collected, vec!["state".to_string()]);
    }

    #[test]
    fn skips_method_ident_keeps_receiver_and_args() {
        let bindings =
            parent_bindings(&[("state", BindingKind::Ref, 0), ("v", BindingKind::Int, 0)]);
        let body = arm_body("{ state.push(v) }");
        let pat = wildcard_pat();
        // `push` is the method ident — must NOT appear in parent_bindings
        // probes; only `state` (receiver) and `v` (arg) match.
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["state".to_string(), "v".to_string()]);
    }

    #[test]
    fn skips_multi_segment_paths() {
        let bindings = parent_bindings(&[("stack_add", BindingKind::Int, 0)]);
        let body = arm_body("{ lj::stack_add(state.selected_ref) }");
        let pat = wildcard_pat();
        // `lj::stack_add` is a 2-segment path — must NOT match the
        // single-segment `stack_add` parent binding.  `state` is also
        // not in parent_bindings here, so result is empty.
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert!(collected.is_empty(), "got {:?}", collected);
    }

    #[test]
    fn local_binding_shadows_caller_scope() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 0),
            ("program", BindingKind::Ref, 0),
        ]);
        let body = arm_body("{ let value = program.get_operand(pc - 1); state.s.push(value); }");
        let pat = wildcard_pat();
        // RHS `program.get_operand(pc - 1)` is visited BEFORE `value`
        // joins local_binds, so `program` and `pc` are picked up.
        // Subsequent `value` reference does NOT appear in
        // parent_bindings, so it never enters the result; even if it
        // did, the local-bind suppression would skip it.
        let mut collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        collected.sort();
        assert_eq!(collected, vec!["pc".to_string(), "program".to_string()]);
    }

    #[test]
    fn dedupes_repeated_idents() {
        let bindings = parent_bindings(&[("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ pc + pc + pc }");
        let pat = wildcard_pat();
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn skips_idents_not_in_parent_bindings() {
        let bindings = parent_bindings(&[("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ x + y + pc }");
        let pat = wildcard_pat();
        // `x` and `y` are not parent bindings — only `pc` is collected.
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn arm_pattern_bound_name_shadows_parent_binding() {
        // Round-3 line-by-line PyPy parity probe (`flowspace` /
        // `SpaceOperation` distinguishes pattern-bound from outer-scope
        // variables).  An arm pattern `OP_PUSH(value)` binds `value`
        // locally; even if `value` happens to be a parent binding, the
        // arm body's reference to `value` is the pattern's payload, NOT
        // the caller-frame value.  Walker must skip pattern-bound names.
        let bindings =
            parent_bindings(&[("value", BindingKind::Int, 5), ("pc", BindingKind::Int, 0)]);
        let body = arm_body("{ state.s.push(value as i64); pc }");
        // `OP_PUSH(value)` — pattern binds `value`.
        let pat: syn::Pat = syn::parse_quote!(OP_PUSH(value));
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        // `value` is pattern-bound — must NOT be collected even though
        // parent_bindings contains it.  `pc` is a free var in the body
        // matching parent_bindings — IS collected.
        assert_eq!(collected, vec!["pc".to_string()]);
    }

    #[test]
    fn arm_pattern_or_alternatives_share_bound_names() {
        // PyPy `pyopcode.py:179` `Or` patterns (`A | B`) require both
        // alternatives to bind the same name; pyre walks all to be
        // robust to syntactic variants.  Either side's binding suffices
        // to suppress the name in the body probe.
        let bindings = parent_bindings(&[("target", BindingKind::Int, 3)]);
        let body = arm_body("{ pc = target; }");
        let pat: syn::Pat = syn::parse_quote!(OP_JMP(target) | OP_BRANCH(target));
        let collected = names(&collect_arm_caller_locals(&body, &pat, &bindings));
        assert!(collected.is_empty(), "got {:?}", collected);
    }

    #[test]
    fn collected_binding_carries_kind_and_reg() {
        let bindings = parent_bindings(&[
            ("pc", BindingKind::Int, 7),
            ("program", BindingKind::Ref, 3),
        ]);
        let body = arm_body("{ program.get_op(pc) }");
        let pat = wildcard_pat();
        let result = collect_arm_caller_locals(&body, &pat, &bindings);
        let pc = result
            .iter()
            .find(|(n, _)| n == "pc")
            .expect("pc collected");
        assert_eq!(pc.1.reg, 7);
        assert!(matches!(pc.1.kind, BindingKind::Int));
        let program = result
            .iter()
            .find(|(n, _)| n == "program")
            .expect("program collected");
        assert_eq!(program.1.reg, 3);
        assert!(matches!(program.1.kind, BindingKind::Ref));
    }
}
#[cfg(test)]
mod assign_caller_local_layout_tests {
    use super::*;

    fn caller_binding(kind: BindingKind, parent_reg: u16) -> Binding {
        Binding {
            reg: parent_reg,
            kind,
            depends_on_stack: false,
        }
    }

    #[test]
    fn empty_input_yields_empty_layout_and_zero_advance() {
        let (layout, max_pre_bound) = assign_caller_local_layout(&[]);
        assert!(layout.is_empty());
        assert_eq!(max_pre_bound, 0);
    }

    #[test]
    fn per_bank_packed_callee_regs() {
        let caller_locals = vec![
            ("first_int".to_string(), caller_binding(BindingKind::Int, 5)),
            ("first_ref".to_string(), caller_binding(BindingKind::Ref, 9)),
            (
                "second_int".to_string(),
                caller_binding(BindingKind::Int, 6),
            ),
            (
                "second_ref".to_string(),
                caller_binding(BindingKind::Ref, 10),
            ),
            (
                "first_float".to_string(),
                caller_binding(BindingKind::Float, 2),
            ),
        ];
        let (layout, max_pre_bound) = assign_caller_local_layout(&caller_locals);
        // Per-bank packed: int → 0, 1; ref → 0, 1; float → 0.
        let by_name: std::collections::HashMap<String, (u16, BindingKind)> = layout
            .iter()
            .map(|l| (l.name.clone(), (l.callee_reg, l.kind)))
            .collect();
        assert_eq!(by_name["first_int"].0, 0u16);
        assert!(matches!(by_name["first_int"].1, BindingKind::Int));
        assert_eq!(by_name["second_int"].0, 1u16);
        assert!(matches!(by_name["second_int"].1, BindingKind::Int));
        assert_eq!(by_name["first_ref"].0, 0u16);
        assert!(matches!(by_name["first_ref"].1, BindingKind::Ref));
        assert_eq!(by_name["second_ref"].0, 1u16);
        assert!(matches!(by_name["second_ref"].1, BindingKind::Ref));
        assert_eq!(by_name["first_float"].0, 0u16);
        assert!(matches!(by_name["first_float"].1, BindingKind::Float));
        // Worst-case advance is the max per-bank slot count = 2 (int and ref).
        assert_eq!(max_pre_bound, 2);
    }

    #[test]
    fn parent_reg_is_preserved_in_layout() {
        let caller_locals = vec![
            ("pc".to_string(), caller_binding(BindingKind::Int, 7)),
            ("program".to_string(), caller_binding(BindingKind::Ref, 3)),
        ];
        let (layout, max_pre_bound) = assign_caller_local_layout(&caller_locals);
        let pc_layout = layout
            .iter()
            .find(|l| l.name == "pc")
            .expect("pc in layout");
        assert_eq!(pc_layout.parent_reg, 7);
        assert_eq!(pc_layout.callee_reg, 0);
        assert!(matches!(pc_layout.kind, BindingKind::Int));
        let program_layout = layout
            .iter()
            .find(|l| l.name == "program")
            .expect("program in layout");
        assert_eq!(program_layout.parent_reg, 3);
        // First Ref → ref_reg 0 (different bank from `pc` Int 0).
        assert_eq!(program_layout.callee_reg, 0);
        assert!(matches!(program_layout.kind, BindingKind::Ref));
        // One Int + one Ref pre-bound → next_reg advance is 1.
        assert_eq!(max_pre_bound, 1);
    }

    #[test]
    fn order_within_kind_matches_input() {
        let caller_locals = vec![
            ("a".to_string(), caller_binding(BindingKind::Int, 100)),
            ("b".to_string(), caller_binding(BindingKind::Int, 200)),
            ("c".to_string(), caller_binding(BindingKind::Int, 300)),
        ];
        let (layout, _) = assign_caller_local_layout(&caller_locals);
        // First-seen Int → 0, second → 1, third → 2.  The layout
        // preserves input order so the parent emit can pair
        // (parent_reg, callee_reg) deterministically.
        assert_eq!(layout[0].callee_reg, 0);
        assert_eq!(layout[0].parent_reg, 100);
        assert_eq!(layout[1].callee_reg, 1);
        assert_eq!(layout[1].parent_reg, 200);
        assert_eq!(layout[2].callee_reg, 2);
        assert_eq!(layout[2].parent_reg, 300);
    }
}

/// Walk `func_block` to find the dispatch while-loop, then lower every stmt
/// that appears before the dispatch match in source order.
///
/// interp_jit.py:91-93 — stmts between jit_merge_point and the opcode
/// dispatch (e.g. `co_code = pycode.co_code`, `valuestackdepth = promote(...)`)
/// execute unconditionally before each dispatch. We lower them into the
/// dispatch JitCode body so the JIT sees them on every loop iteration.
///
/// Returns `Some(())` if at least one pre-dispatch stmt was found; `None` if
/// no while body or dispatch match could be located (caller continues anyway
/// since later tasks fill the dispatch chain).
pub(super) fn lower_pre_dispatch_stmts(lowerer: &mut Lowerer, func_block: &syn::Block) -> Option<()> {
    // Find the dispatch match expression anywhere in the function block.
    let dispatch_match = find_dispatch_match(func_block)?;

    // Find the dispatch loop body (while or loop) whose stmts contain the
    // dispatch match.
    let loop_body = find_dispatch_loop_body(func_block, dispatch_match)?;

    // A.3.6.1: pre-merge-point body-local `let` stmts are bound by
    // `bind_pre_merge_point_stmts` (called earlier in `lower_dispatch_body`).
    // This walker latches `seen_merge_point` upon reaching the macro stmt
    // and only processes stmts that appear AFTER it (avoiding a double-bind
    // collision with the pre-pass).
    let mut seen_merge_point = false;

    // Iterate stmts in the dispatch loop body before the stmt that contains
    // the match.
    for stmt in &loop_body.stmts {
        if stmt_contains_match(stmt, dispatch_match) {
            // Reached the dispatch site; no more pre-dispatch stmts.
            break;
        }
        // A.3.6.1: latch on the `jit_merge_point!()` macro stmt. Pre-merge
        // point stmts are handled by `bind_pre_merge_point_stmts`; the
        // post-merge-point body resumes for subsequent stmts.
        if is_jit_merge_point_macro(stmt) {
            seen_merge_point = true;
            continue;
        }
        if !seen_merge_point {
            continue;
        }
        // Slice 1.4: try to lower opcode-fetch patterns before the
        // state-field filter so they are emitted as IR ops rather than
        // verbatim Rust (which would fail to compile inside
        // `__dispatch_jitcode_*` where `program`/`pc` are not in scope).
        if try_lower_opcode_fetch_stmt(lowerer, stmt) {
            continue;
        }
        // A.2.3a/b: inner `Expr::While` recognition + emission. RPython
        // `pyopcode.py:187-193` lays out the EXTENDED_ARG inner loop as
        // `while opcode == EXTENDED_ARG { ... }`. The proc macro is
        // token-only and cannot resolve constant integer values, so the
        // recognizer matches structural shape only: condition is
        // `<ident> == <ident>` (paren wrapping + reversed operands
        // accepted; one ident must already be in `lowerer.bindings` so
        // the OTHER is the const). When recognition succeeds, A.2.3b
        // emits the loop scaffold + body IR (per Pre-A.2.3 codex
        // BLOCKERs (c) merge arithmetic + (d) HAVE_ARGUMENT polarity).
        // When recognition or body emission fails, signal taint so
        // `lower_dispatch_body` returns `None` → dispatch body empty →
        // gate at `codegen_state.rs:786-823` misses
        // `BC_GETARRAYITEM_GC_I` and refuses install (fail-closed per
        // Pre-A.2.3 codex review item (a)).
        if let Some(while_expr) = stmt_as_inner_while(stmt) {
            if !is_recognized_extended_arg_while(while_expr) {
                lowerer.dispatch_tainted_reason = Some(
                    "A.2.3a fail-closed: inner Expr::While condition is not the \
                     recognized `<ident> == <ident>` shape (EXTENDED_ARG inner \
                     loop per pyopcode.py:187-193)",
                );
                return None;
            }
            if lower_extended_arg_inner_while(lowerer, while_expr).is_none() {
                lowerer.dispatch_tainted_reason = Some(
                    "A.2.3b fail-closed: inner Expr::While body could not be \
                     lowered to RPython EXTENDED_ARG IR (opcode2/arg2 fetch + \
                     HAVE_ARGUMENT range guard + pc += 2 + oparg merge + \
                     opcode reassign per pyopcode.py:188-193)",
                );
                return None;
            }
            continue;
        }
        // A.2.5.a: free-function call statement with resolvable helper
        // policy (e.g. `bytecode_only_trace_helper();` annotated with
        // `#[majit_macros::dont_look_inside]` under `auto_calls = true`).
        // RPython `pyopcode.py:174` `ec.bytecode_only_trace(self)` lowers
        // through `jtransform.py:456-470 rewrite_op` + `call.py:282-324
        // getcalldescr`'s analyzer trio at translation time. Pyre's
        // `resolve_call_policy` + `lower_config_call_stmt` is the
        // per-callsite equivalent before Task #64 plumbs the analyzer
        // trio output to runtime helper-call sites. This recognizer is
        // a third path alongside opcode-fetch and state-modifying stmts
        // — it MUST NOT extend `stmt_modifies_jit_state`, which means
        // "touches lowered JIT state" (state-place reachability), a
        // different concept from "has analyzer-classified side effects"
        // (Pre-A.2.5 codex review BLOCKER 1).
        if try_lower_pre_dispatch_policy_call_stmt(lowerer, stmt) {
            continue;
        }
        // Only lower stmts that modify JIT state (e.g. state field writes,
        // promote calls on state fields). Skip remaining runtime-only stmts
        // that are neither opcode-fetch patterns nor state-modifying.
        if !lowerer.stmt_modifies_jit_state(stmt) {
            continue;
        }
        let _ = lowerer.lower_stmt(stmt);
    }
    Some(())
}

/// A.2.5.a: lower a free-function call statement whose callee has a
/// resolvable helper policy (e.g. `#[majit_macros::dont_look_inside]`
/// under `auto_calls = true` or an explicit `calls = { ... }` entry).
/// Mirrors the dispatch-body equivalent of RPython `pyopcode.py:174`
/// `ec.bytecode_only_trace(self)`, whose effect class is at minimum
/// `DEFAULT_EFFECT_INFO` (EF_CAN_RAISE + saturated read/write descrs)
/// per Pre-A.2.5 codex review (`call.py:282-324 getcalldescr` for the
/// upstream analyzer-trio classification; pyre's per-callsite hatch
/// before Task #64 plumbs the analyzer outputs).
///
/// Returns `true` if the stmt was consumed (lowered via
/// `lower_config_call_stmt`), `false` if the caller should continue
/// with the state-modifying gate or skip-silently fallback.
///
/// Recognises `Stmt::Expr(Expr::Call(_), _)` where the callee path
/// resolves to a `CallPolicySpec` via `resolve_call_policy`. The
/// existing `lower_config_call_stmt` (`jitcode_lower.rs:2319+`) handles
/// every `CallPolicyKind` (ResidualVoid / MayForceVoid / LoopInvariant
/// / Elidable / etc.); this recognizer is the gate that lets the same
/// path fire in the dispatch JitCode body, where `lower_stmt`'s
/// state-modifying filter would otherwise silently skip the call.
fn try_lower_pre_dispatch_policy_call_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(expr, _) = stmt else {
        return false;
    };
    let Expr::Call(call) = expr else {
        return false;
    };
    if lowerer.resolve_call_policy(&call.func).is_none() {
        return false;
    }
    lowerer.lower_config_call_stmt(expr).is_some()
}

/// Return the inner `ExprWhile` if `stmt` is `Stmt::Expr(Expr::While(_), _)`.
/// Used by `lower_pre_dispatch_stmts` to detect EXTENDED_ARG inner loops in
/// the dispatch body (RPython `pyopcode.py:187-193`).
fn stmt_as_inner_while(stmt: &Stmt) -> Option<&syn::ExprWhile> {
    let Stmt::Expr(Expr::While(while_expr), _) = stmt else {
        return None;
    };
    Some(while_expr)
}

/// Recognize the `while <ident> == <ident> { ... }` structural shape used
/// by RPython's EXTENDED_ARG inner loop (`pyopcode.py:187`
/// `while opcode == opcodedesc.EXTENDED_ARG.index`). The proc macro is
/// token-only and cannot resolve constant integer values, so this helper
/// validates only the AST shape:
///
/// - condition is `Expr::Binary { op: Eq, left, right }` (after stripping
///   any number of nested `Expr::Paren` wrappers)
/// - both `left` and `right` are bare single-segment `Expr::Path` idents
///   (no qualified paths, no leading `::`, no generics)
///
/// Either operand order is accepted (`opcode == EXTENDED_ARG` or
/// `EXTENDED_ARG == opcode`) because the macro cannot tell which side is
/// the constant.
fn is_recognized_extended_arg_while(while_expr: &syn::ExprWhile) -> bool {
    let cond = unwrap_expr_paren(&while_expr.cond);
    let Expr::Binary(bin) = cond else {
        return false;
    };
    if !matches!(bin.op, syn::BinOp::Eq(_)) {
        return false;
    }
    expr_single_ident(&bin.left).is_some() && expr_single_ident(&bin.right).is_some()
}

/// Strip any number of `Expr::Paren` wrappers from `expr`, returning the
/// innermost non-parenthesized expression.
fn unwrap_expr_paren(expr: &Expr) -> &Expr {
    let mut cur = expr;
    while let Expr::Paren(p) = cur {
        cur = &p.expr;
    }
    cur
}

/// Return the bound name from `let <pat> = ...` where `<pat>` is either
/// `Pat::Ident(name)` or `Pat::Type(Pat::Ident(name), <ty>)` (i.e.
/// `let X = ...` or `let X: T = ...`). Other pattern shapes (tuple,
/// struct, ref) are out of scope for the byte-fetch lowering.
fn pat_bound_ident_name(pat: &Pat) -> Option<String> {
    let inner = match pat {
        Pat::Type(pt) => pt.pat.as_ref(),
        other => other,
    };
    let Pat::Ident(pi) = inner else { return None };
    Some(pi.ident.to_string())
}

/// A.2.3b: lower a recognized EXTENDED_ARG inner while loop to dispatch
/// JitCode IR. Mirrors RPython `pyopcode.py:187-193`:
///
/// ```text
/// while opcode == opcodedesc.EXTENDED_ARG.index:
///     opcode = ord(co_code[next_instr])
///     arg    = ord(co_code[next_instr + 1])
///     if opcode < HAVE_ARGUMENT:
///         raise BytecodeCorruption
///     next_instr += 2
///     oparg = (oparg * 256) | arg
/// ```
///
/// Emit layout (single shared `opcode` register so the back-edge
/// re-tests against the freshly-fetched value):
///
/// ```text
///   load_const_i  EXTENDED_ARG_const_reg, EXTENDED_ARG  ;; hoisted
/// inner_loop_top:                                       ;; back-edge target
///   goto_if_not_int_eq  opcode_reg, EXTENDED_ARG_const_reg, after_loop
///   <body — opcode2/arg2 fetch with opcode2 aliased to opcode_reg per
///    RPython L188 reuses the same `opcode` variable + arg2 fresh; range
///    guard via goto_if_not_int_lt + abort + ok label; pc += 2 via
///    existing fetch helper; oparg merge via load_const_i(256) +
///    int_mul + int_or>
///   jump inner_loop_top
/// after_loop:
/// ```
///
/// `goto_if_not_int_eq` branches on FALSE
/// (`flatten.py:240-260`/`pyjitpl.py:510-522`), so the top-of-loop test
/// reaches `after_loop` precisely when `opcode != EXTENDED_ARG` —
/// matching RPython's `while opcode == EXTENDED_ARG: ...` semantic.
/// The unconditional `jump` at the bottom is the back-edge per Pre-A.2.3
/// codex review item (b): the back-edge target is the inner loop header
/// where the next iteration re-fetches opcode2/arg2/merge, NOT the outer
/// `loop_start_label`. No second `BC_JIT_MERGE_POINT` is emitted.
///
/// Returns `None` if any structural check fails (mismatched const ident
/// scope, body shape mismatch, missing required outer bindings) so the
/// caller can install the fail-closed gate.
fn lower_extended_arg_inner_while(
    lowerer: &mut Lowerer,
    while_expr: &syn::ExprWhile,
) -> Option<()> {
    // Identify which side of `<ident> == <ident>` is the bound local
    // (one of `lowerer.bindings` with Int kind) and which is the const.
    let (opcode_reg, extended_arg_const) = pick_local_and_const_idents(lowerer, &while_expr.cond)?;

    // Pre-pass: scan body for `<outer_local> = <inner_local>` reassigns
    // (e.g. `opcode = opcode2`) so we can alias the inner `let` of
    // `<inner_local>` to `<outer_local>`'s register. RPython L188 reuses
    // the same `opcode` variable; the Rust fixture uses `opcode2` plus
    // a trailing `opcode = opcode2` purely because `let` in a Rust block
    // shadows rather than rebinds.
    let mut inner_alias: HashMap<String, u16> = HashMap::new();
    for stmt in &while_expr.body.stmts {
        if let Some((lhs, rhs)) = match_simple_ident_assign(stmt) {
            if let Some(b) = lowerer.bindings.get(&lhs) {
                if matches!(b.kind, BindingKind::Int) {
                    inner_alias.insert(rhs, b.reg);
                }
            }
        }
    }

    // Hoist `EXTENDED_ARG` constant load before the loop label so the
    // back-edge does not reload it every iteration. Using `as i64` lets
    // any module-level `const X: u8` (or `i32`/`u16`/etc.) widen to the
    // builder's `i64` argument cleanly.
    let extended_arg_const_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::LoadConstI,
            vec![],
            vec![Register::int(extended_arg_const_reg)],
        ),
        quote! {
            __builder.load_const_i_value(#extended_arg_const_reg as u16, #extended_arg_const as i64);
        },
    );

    let inner_loop_top = lowerer.alloc_label();
    let after_loop = lowerer.alloc_label();
    lowerer.emit_aux(quote! { let #inner_loop_top = __builder.new_label(); });
    lowerer.emit_aux(quote! { let #after_loop = __builder.new_label(); });
    lowerer.emit_label_def(&inner_loop_top);

    // jtransform.py:196-225 fuses int_eq + goto_if_not into
    // goto_if_not_int_eq/iiL. `opcode_reg` is the canonical loop reg
    // updated each iteration by the inner BC_GETARRAYITEM_GC_I aliased
    // through `inner_alias`.
    lowerer.emit_op(
        OpMeta::conditional_guard_int_eq(
            Register::int(opcode_reg),
            Register::int(extended_arg_const_reg),
            after_loop.clone(),
        ),
        quote! {
            __builder.goto_if_not_int_eq(
                #opcode_reg as u16,
                #extended_arg_const_reg as u16,
                #after_loop,
            );
        },
    );

    lower_extended_arg_inner_while_body(lowerer, &while_expr.body, &inner_alias)?;

    lowerer.emit_jump(&inner_loop_top);
    lowerer.emit_label_def(&after_loop);
    Some(())
}

/// Walk the inner-while body's stmts and emit IR for each. Returns
/// `None` on any unrecognized stmt so the caller can fail-closed.
///
/// Recognized stmt shapes (RPython `pyopcode.py:188-193`):
///
/// 1. `let X = program[<idx>]` — opcode2/arg2 byte fetch. If `X` is in
///    `inner_alias`, the BC_GETARRAYITEM_GC_I writes into the aliased
///    register (RPython orthodoxy: `opcode = ord(co_code[next_instr])`
///    reuses the outer `opcode` slot). Otherwise allocates a fresh reg.
/// 2. `pc += N` — delegated to existing `try_lower_opcode_fetch_stmt`.
/// 3. `if X < CONST { panic!(...) }` — HAVE_ARGUMENT range guard.
///    Emits `load_const_i + goto_if_not_int_lt + abort + ok_label`.
///    `goto_if_not_int_lt(a, b, L)` branches when NOT(a < b), i.e.
///    when `a >= b` — so `a < b` fall-throughs into BC_ABORT, matching
///    RPython L190-191 `raise BytecodeCorruption` (Pre-A.2.3 codex
///    BLOCKER (d) polarity correction).
/// 4. `oparg = (<oparg> * 256) | <arg> as i64` — multi-byte oparg merge.
///    Emits `load_const_i 256 + int_mul + int_or` per Pre-A.2.3 codex
///    BLOCKER (c) (no `int_lshift_imm` in pyre; `jtransform.py:363-366`
///    leaves int_mul symmetric).
/// 5. `<outer> = <inner>` where `<inner>` is in `inner_alias` — no-op
///    (the alias was set up in pass 1; both names already point to the
///    outer register).
fn lower_extended_arg_inner_while_body(
    lowerer: &mut Lowerer,
    body: &syn::Block,
    inner_alias: &HashMap<String, u16>,
) -> Option<()> {
    for stmt in &body.stmts {
        if try_lower_inner_byte_fetch(lowerer, stmt, inner_alias) {
            continue;
        }
        if try_lower_opcode_fetch_stmt(lowerer, stmt) {
            continue;
        }
        if try_lower_have_argument_guard(lowerer, stmt) {
            continue;
        }
        if try_lower_oparg_merge_stmt(lowerer, stmt) {
            continue;
        }
        if try_lower_alias_assign_stmt(stmt, inner_alias) {
            continue;
        }
        return None;
    }
    Some(())
}

/// Match `Stmt::Local { pat: Pat::Ident(X), init: Some(program[<idx>]) }`.
/// If `X` is in `inner_alias`, emit `BC_GETARRAYITEM_GC_I` writing into
/// the aliased outer register and re-bind `X → outer_reg`. Otherwise
/// fall through to the existing `try_lower_opcode_fetch_stmt` (which
/// allocates a fresh register).
fn try_lower_inner_byte_fetch(
    lowerer: &mut Lowerer,
    stmt: &Stmt,
    inner_alias: &HashMap<String, u16>,
) -> bool {
    let Stmt::Local(local) = stmt else {
        return false;
    };
    let Some(lhs_name) = pat_bound_ident_name(&local.pat) else {
        return false;
    };
    let Some(&aliased_reg) = inner_alias.get(&lhs_name) else {
        return false;
    };
    let Some(init) = &local.init else {
        return false;
    };
    // Peel an outer `Expr::Cast` per `try_lower_opcode_fetch_stmt`'s
    // Pattern 1: `let X: i64 = program[idx] as i64` (a Rust widening
    // artifact) lowers identically to the bare form.
    let init_expr = match init.expr.as_ref() {
        Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    let Expr::Index(idx) = init_expr else {
        return false;
    };
    if expr_single_ident(&idx.expr).as_deref() != Some("program") {
        return false;
    }
    let Some(prog) = lowerer.bindings.get("program").cloned() else {
        return false;
    };
    let program_reg = prog.reg;
    let Some(index_reg) = lower_array_index_expr(lowerer, &idx.index) else {
        return false;
    };
    let descr_tok = quote! { __builder.add_gc_byte_array_descr() };
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::Vable,
            vec![Register::ref_(program_reg), Register::int(index_reg)],
            vec![Register::int(aliased_reg)],
        ),
        quote! {
            let __descr_idx = #descr_tok;
            __builder.getarrayitem_gc_i(
                #aliased_reg as u16,
                #program_reg as u16,
                #index_reg as u16,
                __descr_idx,
            );
        },
    );
    lowerer.bindings.insert(
        lhs_name,
        Binding {
            reg: aliased_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
        },
    );
    true
}

/// Match `if <ident> < <ident> { panic!(...); }` and emit the
/// HAVE_ARGUMENT range guard:
///
/// ```text
///   load_const_i  HAVE_ARGUMENT_const_reg, HAVE_ARGUMENT
///   goto_if_not_int_lt  opcode_reg, HAVE_ARGUMENT_const_reg, ok_label
///   abort
/// ok_label:
/// ```
///
/// The local ident in the comparison must already be in `lowerer.bindings`
/// as Int; the other ident is treated as the module-level const. Either
/// operand order (`opcode < HAVE_ARGUMENT` or `HAVE_ARGUMENT < opcode`)
/// is rejected — RPython L190 is unambiguously `opcode < HAVE_ARGUMENT`,
/// the comparison is asymmetric, and accepting the reversed form would
/// silently invert the guard.
fn try_lower_have_argument_guard(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::If(if_expr), _) = stmt else {
        return false;
    };
    if if_expr.else_branch.is_some() {
        return false;
    }
    let cond = unwrap_expr_paren(&if_expr.cond);
    let Expr::Binary(bin) = cond else {
        return false;
    };
    if !matches!(bin.op, syn::BinOp::Lt(_)) {
        return false;
    }
    let Some(lhs_name) = expr_single_ident(&bin.left) else {
        return false;
    };
    let Some(rhs_name) = expr_single_ident(&bin.right) else {
        return false;
    };
    // pyopcode.py:190 `if opcode < HAVE_ARGUMENT` — LHS must be the
    // local (in bindings), RHS the const (out of bindings). Rejecting
    // the reversed form keeps the guard polarity unambiguous.
    let Some(local) = lowerer.bindings.get(&lhs_name).cloned() else {
        return false;
    };
    if !matches!(local.kind, BindingKind::Int) {
        return false;
    }
    if lowerer.bindings.contains_key(&rhs_name) {
        return false;
    }
    if !block_is_panic_only(&if_expr.then_branch) {
        return false;
    }
    let Expr::Path(rhs_path) = bin.right.as_ref() else {
        return false;
    };
    let Some(const_ident) = rhs_path.path.get_ident().cloned() else {
        return false;
    };

    let const_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
        quote! {
            __builder.load_const_i_value(#const_reg as u16, #const_ident as i64);
        },
    );
    let ok_label = lowerer.alloc_label();
    lowerer.emit_aux(quote! { let #ok_label = __builder.new_label(); });
    let local_reg = local.reg;
    lowerer.emit_op(
        OpMeta::conditional_guard_int_eq(
            Register::int(local_reg),
            Register::int(const_reg),
            ok_label.clone(),
        ),
        quote! {
            __builder.goto_if_not_int_lt(
                #local_reg as u16,
                #const_reg as u16,
                #ok_label,
            );
        },
    );
    // BC_ABORT is the canonical local bailout — `assembler.rs:1352-1354`
    // + `dispatch.rs:3632-3633` resume protocol. Pre-A.2.3 codex review
    // BLOCKER (d): `guard_value` is the wrong shape (range vs equality);
    // BC_ABORT preserves RPython L190-191 `raise BytecodeCorruption`
    // semantics through the existing trace-abort path.
    lowerer.emit_op(
        OpMeta::terminal(Vec::new()),
        quote! {
            __builder.abort();
        },
    );
    lowerer.emit_label_def(&ok_label);
    true
}

/// Match `<oparg_ident> = (<oparg_ident> * <int_lit>) | <arg_ident> as <ty>`
/// and emit the multi-byte oparg merge per RPython L193:
///
/// ```text
///   load_const_i  c_reg, <int_lit>
///   int_mul       oparg_reg, oparg_reg, c_reg
///   int_or        oparg_reg, oparg_reg, arg_reg
/// ```
///
/// Pre-A.2.3 codex review BLOCKER (c): pyre lacks `BC_INT_LSHIFT_IMM`
/// (only the 2-arg `BC_INT_LSHIFT`), and `jtransform.py:363-366` leaves
/// `int_mul` as the symmetric primitive for `* 256`. Treating
/// `lshift(8)` as the source-port choice would be an optimizer-level
/// equivalence, not RPython parity.
fn try_lower_oparg_merge_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    let Stmt::Expr(Expr::Assign(assign), _) = stmt else {
        return false;
    };
    let Some(lhs_name) = expr_single_ident(&assign.left) else {
        return false;
    };
    let lhs_binding = lowerer.bindings.get(&lhs_name).cloned();
    let Some(lhs_binding) = lhs_binding else {
        return false;
    };
    if !matches!(lhs_binding.kind, BindingKind::Int) {
        return false;
    }

    let rhs = unwrap_expr_paren(assign.right.as_ref());
    let Expr::Binary(or_bin) = rhs else {
        return false;
    };
    if !matches!(or_bin.op, syn::BinOp::BitOr(_)) {
        return false;
    }
    // Left of `|` must be `(<lhs_name> * <int_lit>)` (paren-wrapped).
    let mul_expr = unwrap_expr_paren(or_bin.left.as_ref());
    let Expr::Binary(mul_bin) = mul_expr else {
        return false;
    };
    if !matches!(mul_bin.op, syn::BinOp::Mul(_)) {
        return false;
    }
    let Some(mul_lhs_name) = expr_single_ident(&mul_bin.left) else {
        return false;
    };
    if mul_lhs_name != lhs_name {
        return false;
    }
    let Some(mul_lit) = expr_int_literal_value(&mul_bin.right) else {
        return false;
    };
    if mul_lit <= 0 {
        return false;
    }
    // Right of `|` is `<arg_ident> as <ty>` (RPython has no cast — the
    // fixture's `as i64` is a Rust artifact for u8 → i64 widening).
    let or_rhs = unwrap_expr_paren(or_bin.right.as_ref());
    let arg_expr = match or_rhs {
        Expr::Cast(c) => c.expr.as_ref(),
        other => other,
    };
    let Some(arg_name) = expr_single_ident(arg_expr) else {
        return false;
    };
    let Some(arg_binding) = lowerer.bindings.get(&arg_name).cloned() else {
        return false;
    };
    if !matches!(arg_binding.kind, BindingKind::Int) {
        return false;
    }

    let const_reg = lowerer.alloc_reg();
    let oparg_reg = lhs_binding.reg;
    let arg_reg = arg_binding.reg;
    lowerer.emit_op(
        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
        quote! {
            __builder.load_const_i_value(#const_reg as u16, #mul_lit as i64);
        },
    );
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(oparg_reg), Register::int(const_reg)],
            vec![Register::int(oparg_reg)],
        ),
        quote! {
            __builder.record_binop_i(
                #oparg_reg as u16,
                majit_ir::OpCode::IntMul,
                #oparg_reg as u16,
                #const_reg as u16,
            );
        },
    );
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(oparg_reg), Register::int(arg_reg)],
            vec![Register::int(oparg_reg)],
        ),
        quote! {
            __builder.record_binop_i(
                #oparg_reg as u16,
                majit_ir::OpCode::IntOr,
                #oparg_reg as u16,
                #arg_reg as u16,
            );
        },
    );
    true
}

/// Recognize `<outer> = <inner>` where `<inner>` is in `inner_alias`. The
/// alias was set up in `lower_extended_arg_inner_while`'s pre-pass, so
/// both names already resolve to the same outer register — the stmt is
/// a no-op for the IR (RPython L188's `opcode = ord(...)` is the
/// in-loop rebind; the Rust fixture's trailing `opcode = opcode2` is a
/// scoping artifact, not an RPython operation).
fn try_lower_alias_assign_stmt(stmt: &Stmt, inner_alias: &HashMap<String, u16>) -> bool {
    let Some((_lhs, rhs)) = match_simple_ident_assign(stmt) else {
        return false;
    };
    inner_alias.contains_key(&rhs)
}

/// Match `<lhs_ident> = <rhs_ident>;` where both sides are bare
/// single-segment idents. Returns `(lhs, rhs)` ident names.
fn match_simple_ident_assign(stmt: &Stmt) -> Option<(String, String)> {
    let Stmt::Expr(Expr::Assign(assign), _) = stmt else {
        return None;
    };
    let lhs = expr_single_ident(&assign.left)?;
    let rhs = expr_single_ident(&assign.right)?;
    Some((lhs, rhs))
}

/// Returns `true` if `block` consists of a single `panic!(...)` macro
/// stmt. Used to validate the corruption-bailout body of the
/// HAVE_ARGUMENT range guard so a stray non-panic body would fail
/// recognition (and the caller would mark the dispatch JitCode tainted).
fn block_is_panic_only(block: &syn::Block) -> bool {
    if block.stmts.len() != 1 {
        return false;
    }
    let stmt = &block.stmts[0];
    let mac = match stmt {
        Stmt::Macro(m) => &m.mac,
        Stmt::Expr(Expr::Macro(em), _) => &em.mac,
        _ => return false,
    };
    mac.path
        .segments
        .last()
        .map(|seg| seg.ident == "panic")
        .unwrap_or(false)
}

/// Disambiguate a recognized `<ident> == <ident>` while-condition into
/// `(local_reg, const_ident)`. The "local" side is the one already in
/// `lowerer.bindings` as `Int`; the "const" side is the other ident,
/// returned as a `syn::Ident` so the caller can interpolate it into the
/// `load_const_i` token tree. `None` if both sides or neither are bound
/// (genuinely ambiguous; fail closed per A.2.3a abandonment trigger).
fn pick_local_and_const_idents(lowerer: &Lowerer, cond: &Expr) -> Option<(u16, syn::Ident)> {
    let bin = match unwrap_expr_paren(cond) {
        Expr::Binary(b) => b,
        _ => return None,
    };
    let left_ident = match bin.left.as_ref() {
        Expr::Path(p) => p.path.get_ident().cloned()?,
        _ => return None,
    };
    let right_ident = match bin.right.as_ref() {
        Expr::Path(p) => p.path.get_ident().cloned()?,
        _ => return None,
    };
    let left_local = lowerer
        .bindings
        .get(&left_ident.to_string())
        .filter(|b| matches!(b.kind, BindingKind::Int));
    let right_local = lowerer
        .bindings
        .get(&right_ident.to_string())
        .filter(|b| matches!(b.kind, BindingKind::Int));
    match (left_local, right_local) {
        (Some(b), None) => Some((b.reg, right_ident)),
        (None, Some(b)) => Some((b.reg, left_ident)),
        _ => None,
    }
}

/// Try to lower one of the two opcode-fetch IR patterns:
///
/// 1. `let <name> = program[<index>];` where `<index>` is `pc` or `pc + N`
///    For `pc`, emits `BC_GETARRAYITEM_GC_I result_reg, program_reg(r0),
///    pc_reg(i0)`. For `pc + N` (RPython `pyopcode.py:180`
///    `co_code[next_instr + 1]`), emits an extra `load_const_i(tmp, N) +
///    int_add(offset, pc_reg, tmp)` pair to materialize the index, then
///    `BC_GETARRAYITEM_GC_I result_reg, program_reg, offset_reg`. Stores
///    `<name> → Binding { reg: result_reg, kind: Int }` in lowerer so
///    downstream arms can reference the fetched byte.
///
/// 2. `pc += N` (RPython `pyopcode.py:181` `next_instr += 2`)
///    Emits `load_const_i(tmp, N)` + `int_add(pc_reg, pc_reg, tmp)` to
///    model the pc increment without a literal const operand.
///
/// Identification uses name-based heuristics ("program" / "pc") rather
/// than type-level analysis. This matches the `#[jit_interp]` macro's
/// convention where the env parameter is named `program` and the loop
/// counter is named `pc`.
///
/// TODO: derive names from LowererConfig env/pc config fields when those
/// are added, instead of the hard-coded strings.
///
/// Returns `true` if the stmt was consumed (lowered or silently skipped),
/// `false` if the caller should continue with other lowering paths.
fn try_lower_opcode_fetch_stmt(lowerer: &mut Lowerer, stmt: &Stmt) -> bool {
    // Pattern 1: `let <name> = program[<index>];`
    // AST: Stmt::Local { pat: Pat::Ident { ident: <name> },
    //                    init: Some(LocalInit { expr: Expr::Index {
    //                        expr: Expr::Path(program_path),
    //                        index: <pc | pc + N> } }) }
    if let Stmt::Local(local) = stmt {
        if let Some(init) = &local.init {
            // Peel an outer `Expr::Cast` so `let X: i64 = program[idx] as i64`
            // matches alongside the bare `let X = program[idx]` form. The
            // cast is purely a Rust widening artifact (the byte fetch itself
            // writes into an i64-banked register either way; RPython's
            // `ord(co_code[next_instr + 1])` already returns a Python int).
            let init_expr = match init.expr.as_ref() {
                Expr::Cast(c) => c.expr.as_ref(),
                other => other,
            };
            // Recognise the index form `program[idx]` AND the method-call
            // PyPy `pyopcode.py:171 ord(co_code[next_instr])` is an
            // index form on the bytecode array; that's the ONLY shape
            // the codewriter recognises as the BC_GETARRAYITEM_GC_I
            // opcode-fetch.  Earlier pyre revisions also whitelisted
            // a method-call form `program.get_op(idx)` to accommodate
            // consumers wrapping the byte access in a method, but the
            // proc-macro cannot verify the method body actually equals
            // `code[idx]` — a wrapper that returns `code[idx] ^ key`
            // (or any non-trivial transformation) would silently lower
            // as a raw byte-array load with the wrong semantic.
            //
            // For strict line-by-line PyPy parity, only the index form
            // is recognised here.  Consumers using a method wrapper
            // must register the method as a call policy
            // (`#[jit_interp(calls = { Program::get_op =>
            // elidable_int })]`); `lower_value_expr` then emits a
            // `call_pure_int_canonical_via_target` op rather than a
            // hardcoded byte-array load.
            let opcode_fetch = match init_expr {
                Expr::Index(idx) => {
                    let array_name = expr_single_ident(&idx.expr);
                    if array_name.as_deref() == Some("program") {
                        Some(idx.index.as_ref())
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(idx_expr) = opcode_fetch {
                // Binding names: "program" → r0, "pc" → i0 (installed by
                // lower_dispatch_body before this fn is called).
                let program_binding = lowerer.bindings.get("program").cloned();
                let Some(prog) = program_binding else {
                    return false;
                };
                let program_reg = prog.reg;
                // Compute the index register: `pc` returns pc_reg directly,
                // `pc + N` emits load_const + int_add into a fresh reg.
                let Some(index_reg) = lower_array_index_expr(lowerer, idx_expr) else {
                    return false;
                };
                // Allocate a fresh Int register for the byte fetch result.
                let result_reg = lowerer.alloc_reg();
                let descr_tok = quote::quote! {
                    __builder.add_gc_byte_array_descr()
                };
                lowerer.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![Register::ref_(program_reg), Register::int(index_reg)],
                        vec![Register::int(result_reg)],
                    ),
                    quote::quote! {
                        let __descr_idx = #descr_tok;
                        __builder.getarrayitem_gc_i(
                            #result_reg as u16,
                            #program_reg as u16,
                            #index_reg as u16,
                            __descr_idx,
                        );
                    },
                );
                // Record binding for `<name>` so downstream patterns
                // (dispatch chain, Task 1.5) can reference it. Peel
                // an outer `Pat::Type` so `let X: i64 = ...` (with
                // an explicit type annotation) is recognized
                // alongside the bare `let X = ...` form.
                if let Some(name) = pat_bound_ident_name(&local.pat) {
                    lowerer.bindings.insert(
                        name.clone(),
                        Binding {
                            reg: result_reg,
                            kind: BindingKind::Int,
                            depends_on_stack: false,
                        },
                    );
                    // Slice ε.1: record the consumer's chosen
                    // opcode-result name so `lower_dispatch_chain`
                    // can find the binding regardless of whether the
                    // consumer named it `opcode` (PyPy convention,
                    // `pyopcode.py:171`) or `op` (aheui-jit
                    // `aheui.py:255`) or anything else.
                    lowerer.opcode_var_name = Some(name);
                }
                return true;
            }
        }
    }

    // Pattern 2: `pc += N`
    // syn 2 AST: Stmt::Expr(Expr::Binary { op: BinOp::AddAssign,
    //                left: pc_path, right: Expr::Lit(LitInt(N)) })
    // Also handles `pc = pc + N` (Expr::Assign with binary Add).
    if let Some((lhs_name, increment)) = match_pc_increment_stmt(stmt) {
        if lhs_name == "pc" && increment > 0 {
            let pc_binding = lowerer.bindings.get("pc").cloned();
            let Some(pc) = pc_binding else {
                return false;
            };
            let pc_reg = pc.reg;
            // Load the increment into a fresh tmp int register, then emit
            // int_add(pc_reg, pc_reg, tmp_reg). RPython `pyopcode.py:181`
            // `next_instr += 2` is the canonical N=2 case.
            let tmp_reg = lowerer.alloc_reg();
            lowerer.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(tmp_reg)]),
                quote::quote! {
                    __builder.load_const_i_value(#tmp_reg as u16, #increment as i64);
                },
            );
            lowerer.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    vec![Register::int(pc_reg), Register::int(tmp_reg)],
                    vec![Register::int(pc_reg)],
                ),
                quote::quote! {
                    __builder.record_binop_i(
                        #pc_reg as u16,
                        majit_ir::OpCode::IntAdd,
                        #pc_reg as u16,
                        #tmp_reg as u16,
                    );
                },
            );
            return true;
        }
    }

    false
}

/// Compute the register holding the array-index value for the
/// `program[<idx_expr>]` opcode-fetch pattern. Supports two shapes:
///
/// - `pc` (single ident): returns `pc_reg` directly with no ops emitted.
/// - `pc + N` (binary Add with int literal RHS): emits a fresh
///   `load_const_i(const_reg, N)` + `int_add(offset_reg, pc_reg,
///   const_reg)` pair into the lowerer and returns `offset_reg`. RPython
///   `pyopcode.py:180` `co_code[next_instr + 1]` is the canonical N=1
///   case; `pc_reg` itself is preserved (matches RPython orthodoxy where
///   the index expression does NOT mutate next_instr — that is L181's
///   `next_instr += 2`).
///
/// Returns `None` for any other shape so the caller can abort lowering
/// and fall through to the state-modifies filter.
fn lower_array_index_expr(lowerer: &mut Lowerer, idx_expr: &Expr) -> Option<u16> {
    if let Some(name) = expr_single_ident(idx_expr) {
        if name == "pc" {
            return Some(lowerer.bindings.get("pc")?.reg);
        }
        return None;
    }
    let Expr::Binary(bin) = idx_expr else {
        return None;
    };
    if !matches!(bin.op, syn::BinOp::Add(_)) {
        return None;
    }
    let lhs_name = expr_single_ident(&bin.left)?;
    if lhs_name != "pc" {
        return None;
    }
    let rhs_val = expr_int_literal_value(&bin.right)?;
    if rhs_val <= 0 {
        return None;
    }
    let pc_reg = lowerer.bindings.get("pc")?.reg;
    let const_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
        quote::quote! {
            __builder.load_const_i_value(#const_reg as u16, #rhs_val as i64);
        },
    );
    let offset_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(pc_reg), Register::int(const_reg)],
            vec![Register::int(offset_reg)],
        ),
        quote::quote! {
            __builder.record_binop_i(
                #offset_reg as u16,
                majit_ir::OpCode::IntAdd,
                #pc_reg as u16,
                #const_reg as u16,
            );
        },
    );
    Some(offset_reg)
}

/// Extract the single ident string from `Expr::Path` if it has exactly one
/// segment (no leading `::`, no generics). Returns `None` otherwise.
fn expr_single_ident(expr: &Expr) -> Option<String> {
    let Expr::Path(ep) = expr else { return None };
    if ep.qself.is_some() || ep.path.leading_colon.is_some() {
        return None;
    }
    if ep.path.segments.len() != 1 {
        return None;
    }
    Some(ep.path.segments[0].ident.to_string())
}

/// Match a `pc += N` or `pc = pc + N` statement (where N is any int literal).
/// Returns `Some((lhs_name, increment))` if the pattern matches.
///
/// The caller filters by `lhs_name == "pc"` and `increment > 0`. RPython
/// `pyopcode.py:181` `next_instr += 2` is the canonical N=2 case; the
/// previous N=1 case (`pc += 1`) is preserved verbatim by the same
/// generalized AST shape.
fn match_pc_increment_stmt(stmt: &Stmt) -> Option<(String, i64)> {
    let expr = match stmt {
        Stmt::Expr(e, _) => e,
        _ => return None,
    };
    // `pc += N` — syn 2 parses compound assignments as Expr::Binary
    // with BinOp::AddAssign(syn::token::PlusEq).
    if let Expr::Binary(bin) = expr {
        if matches!(bin.op, syn::BinOp::AddAssign(_)) {
            let lhs = expr_single_ident(&bin.left)?;
            let increment = expr_int_literal_value(&bin.right)?;
            return Some((lhs, increment));
        }
    }
    // `pc = pc + N`
    if let Expr::Assign(a) = expr {
        let lhs = expr_single_ident(&a.left)?;
        if let Expr::Binary(bin) = a.right.as_ref() {
            if matches!(bin.op, syn::BinOp::Add(_)) {
                let l_name = expr_single_ident(&bin.left)?;
                let increment = expr_int_literal_value(&bin.right)?;
                if l_name == lhs {
                    return Some((lhs, increment));
                }
            }
        }
    }
    None
}

/// Return the integer value of an `Expr::Lit(LitInt)` if it fits in i64.
fn expr_int_literal_value(expr: &Expr) -> Option<i64> {
    let Expr::Lit(el) = expr else { return None };
    let Lit::Int(li) = &el.lit else { return None };
    li.base10_parse::<i64>().ok()
}

/// Find the body block of the unique top-level loop in `func_block` whose
/// body contains `target_match`.
///
/// Recognises both `while cond { ... }` (e.g. `aheui.py:251 while pc <
/// program.size`) and `loop { ... }` (e.g. `pypy/jit/tl/tinyframe.py` and
/// other tinyframe-family interpreters whose dispatch loop is unconditional
/// with a `break`-driven exit).  Mirrors `codegen_trace.rs:520
/// expr_inner_match_block`'s recognition set.
pub(super) fn find_dispatch_loop_body<'b>(
    func_block: &'b syn::Block,
    target_match: &ExprMatch,
) -> Option<&'b syn::Block> {
    for stmt in &func_block.stmts {
        let expr = match stmt {
            Stmt::Expr(e, _) => e,
            _ => continue,
        };
        match expr {
            Expr::While(while_expr) if block_contains_match(&while_expr.body, target_match) => {
                return Some(&while_expr.body);
            }
            Expr::Loop(loop_expr) if block_contains_match(&loop_expr.body, target_match) => {
                return Some(&loop_expr.body);
            }
            _ => continue,
        }
    }
    None
}

/// Returns `true` if `stmt` is a `jit_merge_point!()` macro invocation.
pub(super) fn is_jit_merge_point_macro(stmt: &Stmt) -> bool {
    let Stmt::Macro(mac_stmt) = stmt else {
        return false;
    };
    let path = &mac_stmt.mac.path;
    path.segments
        .last()
        .map(|seg| seg.ident == "jit_merge_point")
        .unwrap_or(false)
}

/// Build the parent-side `__builder.inline_call_<types>_v(__sub_idx, ...)`
/// emit for a dispatch arm given its
/// [`CallerLocalLayout`] list (from
/// [`try_generate_jitcode_body_parts_with_caller_bindings`]).
///
/// Picks the family member by which banks are populated:
/// - any Float entry → `inline_call_irf_v` (Int+Ref+Float arg vectors);
/// - else any Int entry → `inline_call_ir_v` (Int+Ref arg vectors);
/// - else → `inline_call_r_v` (Ref-only arg vector; degenerates to the
///   no-arg form when layout is empty).
///
/// Arg pairs are `(parent_reg, callee_reg)` per `assembler.rs:1421
/// inline_call_<types>_v` API.  Mirrors `inline_call_tokens` at
/// `:5098`'s family-by-bank pattern but always selects the void-result
/// variant — dispatch arms never produce an inline_call return value
/// (the arm body's return path is always the loop back-edge to
/// `jit_merge_point` or the default exit, never a value).
pub(super) fn dispatch_arm_inline_call_tokens(layout: &[CallerLocalLayout]) -> proc_macro2::TokenStream {
    use quote::quote;
    let has_int = layout.iter().any(|l| matches!(l.kind, BindingKind::Int));
    let has_float = layout.iter().any(|l| matches!(l.kind, BindingKind::Float));
    let pair_tokens = |kind: BindingKind| -> Vec<proc_macro2::TokenStream> {
        layout
            .iter()
            .filter(|l| l.kind == kind)
            .map(|l| {
                let parent = l.parent_reg;
                let callee = l.callee_reg;
                quote! { (#parent as u16, #callee as u16) }
            })
            .collect()
    };
    let args_i = pair_tokens(BindingKind::Int);
    let args_r = pair_tokens(BindingKind::Ref);
    let args_f = pair_tokens(BindingKind::Float);
    if has_float {
        quote! {
            __builder.inline_call_irf_v(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                &[#(#args_f),*],
                None,
            );
        }
    } else if has_int {
        quote! {
            __builder.inline_call_ir_v(
                __sub_idx,
                &[#(#args_i),*],
                &[#(#args_r),*],
                None,
            );
        }
    } else {
        quote! {
            __builder.inline_call_r_v(
                __sub_idx,
                &[#(#args_r),*],
                None,
            );
        }
    }
}

#[cfg(test)]
mod dispatch_arm_inline_call_tokens_tests {
    use super::*;

    fn entry(name: &str, parent_reg: u16, callee_reg: u16, kind: BindingKind) -> CallerLocalLayout {
        CallerLocalLayout {
            name: name.to_string(),
            parent_reg,
            callee_reg,
            kind,
        }
    }

    fn render(tokens: &proc_macro2::TokenStream) -> String {
        tokens.to_string()
    }

    #[test]
    fn empty_layout_emits_inline_call_r_v_no_args() {
        let out = dispatch_arm_inline_call_tokens(&[]);
        let s = render(&out);
        assert!(s.contains("inline_call_r_v"), "got: {s}");
        // Both arg vec literal and the no-return None must appear.
        assert!(s.contains("& [] ,"), "got: {s}");
        assert!(s.contains("None"), "got: {s}");
    }

    #[test]
    fn ref_only_layout_uses_inline_call_r_v() {
        let layout = vec![entry("program", 3, 0, BindingKind::Ref)];
        let s = render(&dispatch_arm_inline_call_tokens(&layout));
        assert!(s.contains("inline_call_r_v"), "got: {s}");
        assert!(s.contains("3u16") || s.contains("3 as u16"), "got: {s}");
        assert!(!s.contains("inline_call_ir_v"), "got: {s}");
    }

    #[test]
    fn mixed_int_ref_uses_inline_call_ir_v() {
        let layout = vec![
            entry("pc", 7, 0, BindingKind::Int),
            entry("program", 3, 0, BindingKind::Ref),
        ];
        let s = render(&dispatch_arm_inline_call_tokens(&layout));
        assert!(s.contains("inline_call_ir_v"), "got: {s}");
        // Both parent regs land in their respective arg slices.
        assert!(s.contains("7"), "got: {s}");
        assert!(s.contains("3"), "got: {s}");
    }

    #[test]
    fn any_float_uses_inline_call_irf_v() {
        let layout = vec![
            entry("pc", 7, 0, BindingKind::Int),
            entry("flt", 9, 0, BindingKind::Float),
        ];
        let s = render(&dispatch_arm_inline_call_tokens(&layout));
        assert!(s.contains("inline_call_irf_v"), "got: {s}");
    }
}

/// Slice 1.5: Emit the dispatch chain for the opcode dispatch loop.
///
/// For each non-wildcard arm, emits a fused `goto_if_not_int_eq/iiL`
/// (BC_GOTO_IF_NOT_INT_EQ) that branches past the arm if the opcode does NOT
/// match. After all checks, emits an unconditional `jump` (BC_GOTO) to the
/// default label.
///
/// pyopcode.py:183+ if/elif chain over opcode constants.
/// jtransform.py:196-225 optimize_goto_if_not fuses `int_eq + goto_if_not`
/// into `goto_if_not_int_eq/iiL`.
///
/// `default_label` is bound at the typed-return emission site in
/// `lower_dispatch_body`; `loop_start_label` is the back-edge target
/// (JIT_MERGE_POINT position) emitted after each matched arm's body.
pub(super) fn lower_dispatch_chain(
    lowerer: &mut Lowerer,
    classified_arms: &[crate::jit_interp::classify::ClassifiedArm],
    config: &LowererConfig,
    loop_start_label: &syn::Ident,
) -> syn::Ident {
    // Allocate the default/exit label (BC_GOTO target when no arm matches).
    // Allocated before the opcode-reg guard so we always have a label to return.
    let default_label = lowerer.alloc_label();
    lowerer.emit_aux(quote::quote! { let #default_label = __builder.new_label(); });

    // Retrieve the opcode register installed by the opcode-fetch lowerer.
    // Slice ε.1: prefer the consumer's chosen opcode-result name (set by
    // `try_lower_opcode_fetch_stmt` when it recognised the
    // `let <name> = program[<idx>]` pattern); fall back to the literal
    // `"opcode"` (PyPy `pyopcode.py:171` canonical name) so existing
    // fixtures whose dispatch loops use that name continue to lower.
    // If neither is bound (e.g. skeleton without opcode-fetch), skip
    // chain emission.
    let opcode_lookup_name: String = lowerer
        .opcode_var_name
        .clone()
        .unwrap_or_else(|| "opcode".to_string());
    let opcode_reg = match lowerer.bindings.get(&opcode_lookup_name) {
        Some(b) if matches!(b.kind, BindingKind::Int) => b.reg,
        _ => return default_label,
    };

    for arm in classified_arms {
        // `_` wildcard: skip here; handled by the default GOTO below.
        // All other patterns (including Pat::Ident like `OP_NOP`) are
        // treated as constant tests and emitted as goto_if_not_int_eq.
        if matches!(arm.pat, Pat::Wild(_)) || is_lowercase_binding_pat(&arm.pat) {
            continue;
        }

        // Extract token expressions for each value in the pattern.
        // extract_pat_value_tokens handles Pat::Lit, Pat::Path, Pat::Or.
        let value_tokens = match extract_pat_value_tokens(&arm.pat) {
            Some(v) => v,
            None => continue, // unsupported pattern shape — skip
        };

        // Allocate a skip label: if opcode ≠ this arm's value, jump here.
        // Task 1.6 will emit the arm body between the check and this label.
        let skip_label = lowerer.alloc_label();
        lowerer.emit_aux(quote::quote! { let #skip_label = __builder.new_label(); });

        let matched_label = if value_tokens.len() > 1 {
            let label = lowerer.alloc_label();
            lowerer.emit_aux(quote::quote! { let #label = __builder.new_label(); });
            Some(label)
        } else {
            None
        };

        for (value_idx, val_tok) in value_tokens.iter().enumerate() {
            // Load the pattern constant into a fresh int register.
            let const_reg = lowerer.alloc_reg();
            lowerer.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                quote::quote! {
                    __builder.load_const_i_value(#const_reg as u16, #val_tok);
                },
            );
            let is_last_value = value_idx + 1 == value_tokens.len();
            let miss_label = if is_last_value {
                skip_label.clone()
            } else {
                let label = lowerer.alloc_label();
                lowerer.emit_aux(quote::quote! { let #label = __builder.new_label(); });
                label
            };
            // Fused goto_if_not_int_eq: branch to the next alternative (or
            // the arm skip label) if opcode != const. For `A | B`, a
            // successful early alternative jumps to the shared matched label
            // so the remaining alternatives are not tested as an accidental
            // conjunction.
            lowerer.emit_op(
                OpMeta::conditional_guard_int_eq(
                    Register::int(opcode_reg),
                    Register::int(const_reg),
                    miss_label.clone(),
                ),
                quote::quote! {
                    __builder.goto_if_not_int_eq(#opcode_reg as u16, #const_reg as u16, #miss_label);
                },
            );
            if let Some(matched_label) = matched_label.as_ref() {
                lowerer.emit_jump(matched_label);
                if !is_last_value {
                    lowerer.emit_label_def(&miss_label);
                }
            }
        }
        if let Some(matched_label) = matched_label.as_ref() {
            lowerer.emit_label_def(matched_label);
        }

        // jtransform.py:473-482 — inline_call_* + trailing -live-.
        // Build the arm sub-JitCode and register it; emit BC_INLINE_CALL.
        // This executes when the arm MATCHED (guards fell through); the
        // sub-JitCode encodes the opcode handler body.
        //
        // Slice 1.3 of dispatch-arm caller-local plumbing: walk the arm
        // body to collect parent-scope idents (via `collect_arm_caller_locals`),
        // pre-bind them on the sub-Lowerer at fresh per-bank callee regs
        // (via `try_generate_jitcode_body_parts_with_caller_bindings`), and
        // emit the typed `inline_call_<types>_v(__sub_idx, args_i, args_r,
        // args_f)` so the callee jitcode receives them as portal-input
        // bindings.  Mirrors `jtransform.py:480 inline_call_<types>(jitcode,
        // args...)`.  When the arm body has no parent-scope refs the layout
        // is empty and the emit reduces to the no-arg `inline_call_r_v`
        // (equivalent to the previous `__builder.inline_call(__sub_idx)`).
        let mut arm_inline_call_reads: Vec<Register> = Vec::new();
        let (arm_body_tokens, arm_inline_call_emit): (
            proc_macro2::TokenStream,
            proc_macro2::TokenStream,
        ) = match &arm.pattern {
            crate::jit_interp::classify::ArmPattern::Lowerable => {
                let caller_locals =
                    collect_arm_caller_locals(&arm.original_body, &arm.pat, &lowerer.bindings);
                match try_generate_jitcode_body_parts_with_caller_bindings(
                    &arm.original_body,
                    Some(config),
                    &caller_locals,
                ) {
                    Some((generated, layout)) => {
                        let body = generated.body;
                        let liveness_prebuild = generated.liveness_prebuild;
                        lowerer.inline_liveness_prebuild.push(liveness_prebuild);
                        // Carry the parent-side caller regs into the
                        // BC_INLINE_CALL OpMeta so the liveness walker
                        // accounts for them as live at the call site
                        // (assembler.py:225 get_liveness_info reads).
                        for entry in &layout {
                            arm_inline_call_reads.push(Register::new(entry.kind, entry.parent_reg));
                        }
                        let inline_call_emit = dispatch_arm_inline_call_tokens(&layout);
                        (
                            quote::quote! {
                                let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                                let _live_offset_patch = __sub_builder.live_placeholder();
                                {
                                    let __builder = &mut __sub_builder;
                                    #body
                                }
                                __sub_builder.finalize_liveness(__asm);
                                __sub_builder.finish()
                            },
                            inline_call_emit,
                        )
                    }
                    None => (
                        quote::quote! {
                            {
                                let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                                __sub_builder.abort();
                                __sub_builder.finish()
                            }
                        },
                        // Failed-lower fallback: no caller args (the abort
                        // body never reads them anyway).
                        dispatch_arm_inline_call_tokens(&[]),
                    ),
                }
            }
            crate::jit_interp::classify::ArmPattern::Nop => (
                quote::quote! { majit_metainterp::JitCodeBuilder::new().finish() },
                dispatch_arm_inline_call_tokens(&[]),
            ),
            crate::jit_interp::classify::ArmPattern::AbortPermanent
            | crate::jit_interp::classify::ArmPattern::Halt => (
                quote::quote! {
                    {
                        let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                        __sub_builder.abort_permanent();
                        __sub_builder.finish()
                    }
                },
                dispatch_arm_inline_call_tokens(&[]),
            ),
            crate::jit_interp::classify::ArmPattern::Unsupported(_) => (
                quote::quote! {
                    {
                        let mut __sub_builder = majit_metainterp::JitCodeBuilder::new();
                        __sub_builder.abort();
                        __sub_builder.finish()
                    }
                },
                dispatch_arm_inline_call_tokens(&[]),
            ),
        };
        lowerer.emit_op(
            OpMeta::linear(OpKind::InlineCall, arm_inline_call_reads, vec![]),
            quote::quote! {
                let __sub_jitcode = { #arm_body_tokens };
                let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                #arm_inline_call_emit
            },
        );
        // jtransform.py:480-482 — trailing -live- after inline_call_*.
        lowerer.emit_op(
            OpMeta::live_marker(),
            quote::quote! { let _ = __builder.live_placeholder(); },
        );
        // jtransform.py:1714-1723 `handle_jit_marker__loop_header`:
        // RPython lowers `can_enter_jit()` at the user's source-code
        // back-edge (interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)`
        // inside `jump_absolute`'s BACKWARD branch only — `interp_jit.py:104
        // if jumpto >= next_instr: return jumpto` early-out skips the
        // forward path) into a `loop_header(jd.index)` op AT the same
        // source position.  Pyre's `can_enter_jit!()` recognition lives
        // in `Lowerer::lower_stmt` (`Stmt::Macro` arm) which emits the
        // `LoopHeader` IR + `__builder.loop_header(__jdindex)` at that
        // exact stmt position INSIDE the arm body sub-JitCode — so the
        // LH op only executes when the user's source-level conditional
        // (`if backward { can_enter_jit!(); ... }`) is taken at runtime.
        // No post-INLINE_CALL emission here: doing so would over-emit
        // on every arm execution including forward-progress arms (per
        // codex strict-parity audit — arm-level existence ≠ conditional
        // call-site).
        //
        // interp_jit.py:95-100 — loop back-edge: after each matched arm,
        // jump back to jit_merge_point so the next iteration re-enters
        // the dispatch loop at the portal merge point.  The GOTO is
        // required for control-flow correctness regardless of whether
        // the arm body emitted any LH inside its sub-JitCode.
        lowerer.emit_jump(loop_start_label);

        // Bind the skip label at the end of this arm's guard sequence.
        // Jumping here means "this arm did not match; proceed to next arm".
        lowerer.emit_label_def(&skip_label);
    }

    // After all arm guards, the default/exit path: unconditional GOTO.
    // default_label is bound at the typed-return emission site in
    // lower_dispatch_body (Task 1.7).
    lowerer.emit_jump(&default_label);
    default_label
}

fn is_lowercase_binding_pat(pat: &Pat) -> bool {
    let Pat::Ident(pi) = pat else {
        return false;
    };
    if pi.subpat.is_some() || pi.mutability.is_some() || pi.by_ref.is_some() {
        return false;
    }
    pi.ident
        .to_string()
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_lowercase())
}

/// A.3.5 — emit a `-live-` + `<kind>_guard_value` pair for each declared green.
///
/// Mirrors `jtransform.py:1693-1714 promote_greens`: for every green Variable
/// (constants are already promoted and skipped at the RPython level; pyre
/// has no compile-time green constants so every entry is promoted), emit a
/// `-live-` marker followed by `<kind>_guard_value(reg)`.  The guard forces
/// the runtime value to a constant before `BC_JIT_MERGE_POINT`, satisfying
/// `pyjitpl.py:1530` which expects all greens to be constants at the merge
/// point.
///
/// Must be called after the portal-input bindings are installed (so that
/// `lowerer.bindings` maps green idents → `Binding`) and BEFORE the
/// `jit_merge_point` emit.
pub(super) fn emit_promote_greens(lowerer: &mut Lowerer, config: &LowererConfig) {
    for green in &config.greens {
        let ident = match green {
            syn::Expr::Path(p) => p.path.get_ident().unwrap_or_else(|| {
                panic!(
                    "A.3.5 (jtransform.py:1693): green expression must be a single-segment \
                     ident for promote_greens. Got: {:?}",
                    p.path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                )
            }),
            _ => panic!(
                "A.3.5 (jtransform.py:1693): green expression must be a single-segment \
                 ident for promote_greens. Got non-path expression: {}",
                quote::quote!(#green)
            ),
        };
        let ident_name = ident.to_string();
        let binding = lowerer.bindings.get(&ident_name).unwrap_or_else(|| {
            panic!(
                "A.3.5 (jtransform.py:1693): green '{}' declared in #[jit_interp(greens = ...)] \
                 but not bound at portal entry. Available bindings: {:?}",
                ident_name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        let reg = binding.reg;
        let kind = binding.kind;
        // jtransform.py:1707: emit `-live-` before each guard_value so the
        // codewriter's per-marker liveness analysis records the alive set here.
        lowerer.emit_op(
            OpMeta::live_marker(),
            quote::quote! { __builder.live_placeholder(); },
        );
        match kind {
            BindingKind::Int => {
                lowerer.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::int(reg)], vec![]),
                    quote::quote! { __builder.int_guard_value(#reg); },
                );
            }
            BindingKind::Ref => {
                lowerer.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::ref_(reg)], vec![]),
                    quote::quote! { __builder.ref_guard_value(#reg); },
                );
            }
            BindingKind::Float => {
                lowerer.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::float(reg)], vec![]),
                    quote::quote! { __builder.float_guard_value(#reg); },
                );
            }
        }
    }
}

/// A.3.2 — resolve green variable names to register-byte lists.
///
/// Mirrors `jtransform.py:1700 make_three_lists(op.args[2:2+num_green_args])`:
/// each green expression is expected to be a single-segment ident.  Dotted
/// paths (e.g. `state.pc`) are explicitly out of scope — task A.7.
///
/// Returns `(greens_i, greens_r, greens_f)` matching the
/// `jit_merge_point(..., greens_i, greens_r, greens_f, ...)` bucket order.
pub(super) fn resolve_greens(lowerer: &Lowerer, config: &LowererConfig) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut greens_i: Vec<u8> = Vec::new();
    let mut greens_r: Vec<u8> = Vec::new();
    let mut greens_f: Vec<u8> = Vec::new();

    for green in &config.greens {
        let ident = match green {
            syn::Expr::Path(p) => p.path.get_ident().unwrap_or_else(|| {
                panic!(
                    "A.3.2 (jtransform.py:1700): green expression must be a single-segment \
                     ident; dotted greens (state.pc) are scoped to follow-up task A.7. \
                     Got: {:?}",
                    p.path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                )
            }),
            _ => panic!(
                "A.3.2 (jtransform.py:1700): green expression must be a single-segment \
                 ident; dotted greens (state.pc) are scoped to follow-up task A.7. \
                 Got non-path expression: {}",
                quote::quote!(#green)
            ),
        };
        let ident_name = ident.to_string();
        let binding = lowerer.bindings.get(&ident_name).unwrap_or_else(|| {
            panic!(
                "A.3.2 (jtransform.py:1700): green '{}' declared in #[jit_interp(greens = ...)] \
                 but not bound at portal entry. Available bindings: {:?}",
                ident_name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        // `Binding.reg: u16` but `assembler.py:225` per-bank bitset addressing
        // is u8-bounded.  `Register::new()` asserts this on construction; the
        // `Binding`-shaped path does not, so we re-check at the boundary
        // before encoding the register byte into the jit_merge_point payload.
        let reg_byte = u8::try_from(binding.reg).unwrap_or_else(|_| {
            panic!(
                "A.3.2 (assembler.py:225): green register index {} for ident '{}' exceeds u8 \
                 encoding limit; jit_merge_point list-byte encoding is u8-bounded",
                binding.reg, ident_name,
            )
        });
        match binding.kind {
            BindingKind::Int => greens_i.push(reg_byte),
            BindingKind::Ref => greens_r.push(reg_byte),
            BindingKind::Float => greens_f.push(reg_byte),
        }
    }

    // Validate uniqueness within each bucket (jtransform.py:1701).
    // `make_three_lists` is invoked twice in jtransform — greens at :1693 and
    // reds at :1697 — and the `dict.fromkeys` assert runs over each return
    // value. `resolve_reds` mirrors this; backport the same check here.
    for (label, bucket) in [
        ("greens_i", &greens_i),
        ("greens_r", &greens_r),
        ("greens_f", &greens_f),
    ] {
        let mut seen = HashSet::new();
        for &b in bucket.iter() {
            if !seen.insert(b) {
                panic!(
                    "A.3.2 (jtransform.py:1701): duplicate register {} in {}",
                    b, label
                );
            }
        }
    }

    (greens_i, greens_r, greens_f)
}

/// A.3.3 — resolve red variable names to register-byte lists.
///
/// Mirrors `jtransform.py:1700 make_three_lists(op.args[2+num_green_args:])`:
/// pyre's portal inputs are `program` (Ref/r0), `pc` (Int/i0), and optionally
/// `vable_var` (Ref/r1).  The reds = portal-inputs minus declared greens.
///
/// PRE-EXISTING-ADAPTATION: `interp_jit.py:67 reds = ['frame', 'ec']` uses
/// PyPy's frame+ec pair; pyre uses `[program, pc]` (minus greens) as its
/// minimal reds set.  Consumers that want the PyPy parity declaration can set
/// `greens = [pc, program]`, leaving reds = [], which is the intended A.6
/// follow-up mapping.
///
/// Returns `(reds_i, reds_r, reds_f)` matching the
/// `jit_merge_point(..., reds_i, reds_r, reds_f)` bucket order.
pub(super) fn resolve_reds(lowerer: &Lowerer, config: &LowererConfig) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut reds_i: Vec<u8> = Vec::new();
    let mut reds_r: Vec<u8> = Vec::new();
    let mut reds_f: Vec<u8> = Vec::new();

    // Slice (audit Issue #6) — when the consumer declares
    // `#[jit_interp(reds = [...])]` explicitly, use that list as the
    // canonical reds source matching RPython
    // `jtransform.py:1700 make_three_lists(op.args[2+num_green_args:])`
    // — the marker's tail args are the reds.  Pyre's marker is
    // stateless (no tail args), so the `reds` config slot replaces
    // them.  When `config.reds` is empty, fall back to the legacy
    // candidate list `[program, pc(+ optional vable)]` minus declared
    // greens (the pre-Issue-#6 pyre default).
    let explicit_red_names: Vec<String> = config
        .reds
        .iter()
        .filter_map(|expr| match expr {
            syn::Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
            _ => panic!(
                "Issue #6: red expression must be a single-segment ident; \
                 got non-path: {}",
                quote::quote!(#expr),
            ),
        })
        .collect();

    let owned_red_names: Vec<String>;
    let reds_names: Vec<&str> = if !explicit_red_names.is_empty() {
        owned_red_names = explicit_red_names;
        owned_red_names.iter().map(|s| s.as_str()).collect()
    } else {
        // Issue 2.2 (support.py:121 _kind2count = {'int':1,'ref':2,'float':3}):
        // `pc` (Int) precedes `program` (Ref) so the int→ref→float bucket
        // emit below preserves the canonical sort order.  Bucketing
        // sorts by kind regardless, but lining up the candidate list
        // with the bucket order keeps schema (`red_schema`) and payload
        // byte-for-byte aligned.
        let mut candidates: Vec<&str> = vec!["pc", "program"];
        if let Some(vable) = config.vable_var.as_ref() {
            candidates.push(vable.as_str());
        }
        let green_names: HashSet<String> = config
            .greens
            .iter()
            .filter_map(|expr| match expr {
                syn::Expr::Path(p) => p.path.get_ident().map(|i| i.to_string()),
                _ => None,
            })
            .collect();
        candidates
            .into_iter()
            .filter(|name| !green_names.contains(*name))
            .collect()
    };

    for name in reds_names {
        let binding = lowerer.bindings.get(name).unwrap_or_else(|| {
            panic!(
                "A.3.3 (jtransform.py:1700): red '{}' is a portal-input name but has no \
                 binding at jit_merge_point emit. Available bindings: {:?}",
                name,
                lowerer.bindings.keys().collect::<Vec<_>>(),
            )
        });
        let reg_byte = u8::try_from(binding.reg).unwrap_or_else(|_| {
            panic!(
                "A.3.3 (assembler.py:225): red register index {} for ident '{}' exceeds u8 \
                 encoding limit; jit_merge_point list-byte encoding is u8-bounded",
                binding.reg, name,
            )
        });
        match binding.kind {
            BindingKind::Int => reds_i.push(reg_byte),
            BindingKind::Ref => reds_r.push(reg_byte),
            BindingKind::Float => reds_f.push(reg_byte),
        }
    }

    // Validate uniqueness within each bucket (jtransform.py:1701).
    for (label, bucket) in [
        ("reds_i", &reds_i),
        ("reds_r", &reds_r),
        ("reds_f", &reds_f),
    ] {
        let mut seen = HashSet::new();
        for &b in bucket.iter() {
            if !seen.insert(b) {
                panic!(
                    "A.3.3 (jtransform.py:1701): duplicate register {} in {}",
                    b, label
                );
            }
        }
    }

    (reds_i, reds_r, reds_f)
}

/// Slice (audit Issue #5) — extract `(name, green_type_token)` pairs
/// for the declared greens.  `BindingKind` maps to canonical
/// `majit_ir::GreenType::{Int, Ref, Float}`; an explicit `: str` /
/// `: unicode` tag in `config.green_type_tags` overrides the binding's
/// IR type with the upstream `Ptr(rstr.STR)` / `Ptr(rstr.UNICODE)`
/// distinction (`warmspot.py:663 _green_args_spec`).  Used at the
/// install path to populate `JitDriverStaticData::vars` via
/// `JitDriver::declare_schema_typed` so `green_args_spec` reports the
/// real lltype subtype instead of collapsing STR/UNICODE to `Ref`.
///
/// Output preserves declaration order.  RPython
/// `decode_hp_hint_args` (support.py:135-150) does not silently reorder
/// the JitDriver declaration; it computes `sort_vars(lst)` and asserts
/// `lst == lst2`, telling the user to reorder the greens/reds if needed.
/// Pyre mirrors that shape here: validate `int → ref → float` order, then
/// return the schema unchanged.  The bytecode payload encoder
/// (`resolve_greens`) still emits the RPython `make_three_lists` bucket
/// shape independently.
pub(super) fn green_schema(lowerer: &Lowerer, config: &LowererConfig) -> Vec<(String, TokenStream)> {
    use crate::jit_interp::green_type_tag::GreenTypeTag;
    let mut out: Vec<(u8, String, TokenStream)> = Vec::new();
    for (i, green) in config.greens.iter().enumerate() {
        // RPython `support.py:135-150 decode_hp_hint_args` strictly
        // validates greens/reds count + ordering — it never silently
        // drops a malformed marker arg.  Pyre mirrors that strength
        // here: bare-ident is the only supported form (matching
        // `JitDriver(..., greens=['name', ...])` on the upstream
        // side); anything else is a structural mismatch the install
        // path could not surface as count/payload divergence
        // downstream.  Earlier `continue` quietly shrank the schema
        // and let the macro emit a payload that disagreed with
        // `JitDriverStaticData::vars`.
        let ident_name = match green {
            syn::Expr::Path(p) => match p.path.get_ident() {
                Some(i) => i.to_string(),
                None => panic!(
                    "#[jit_interp] greens[{i}]: only bare-ident greens are \
                     supported (matching `JitDriver(greens=['name'])` upstream); \
                     got a multi-segment path: {}",
                    quote::quote!(#green),
                ),
            },
            _ => panic!(
                "#[jit_interp] greens[{i}]: only bare-ident greens are \
                 supported (matching `JitDriver(greens=['name'])` upstream); \
                 got a non-path expression: {}",
                quote::quote!(#green),
            ),
        };
        let Some(binding) = lowerer.bindings.get(&ident_name) else {
            panic!(
                "#[jit_interp] greens[{i}]: unknown identifier `{ident_name}` — \
                 not a state field bound by `state_fields!` and not a JitDriver \
                 green declared in scope",
            );
        };
        let tag = config.green_type_tags.get(i).copied().flatten();
        // Tag wins over binding (warmspot.py:663 — _green_args_spec
        // reads the lltype directly from the JitDriver signature, not
        // from the codewriter's IR-collapsed view).
        let (kind_rank, gt_tok) = match tag {
            Some(GreenTypeTag::Int) => (1u8, quote::quote!(majit_ir::GreenType::Int)),
            Some(GreenTypeTag::Float) => (3u8, quote::quote!(majit_ir::GreenType::Float)),
            Some(GreenTypeTag::Ref) => (2u8, quote::quote!(majit_ir::GreenType::Ref)),
            Some(GreenTypeTag::Str) => (2u8, quote::quote!(majit_ir::GreenType::Str)),
            Some(GreenTypeTag::Unicode) => (2u8, quote::quote!(majit_ir::GreenType::Unicode)),
            None => match binding.kind {
                BindingKind::Int => (1u8, quote::quote!(majit_ir::GreenType::Int)),
                BindingKind::Ref => (2u8, quote::quote!(majit_ir::GreenType::Ref)),
                BindingKind::Float => (3u8, quote::quote!(majit_ir::GreenType::Float)),
            },
        };
        out.push((kind_rank, ident_name, gt_tok));
    }
    assert_kind_sorted("greens", &out);
    out.into_iter().map(|(_, n, t)| (n, t)).collect()
}

/// Companion to [`green_schema`] for the dispatch path's red layout.
/// Mirrors [`resolve_reds`] — when `config.reds` is non-empty, use that
/// explicit list (Issue #6 RPython parity); otherwise default to
/// `[pc, program(+ optional vable)]` minus declared greens.
///
/// Output preserves declaration order and asserts the same
/// `int → ref → float` invariant as RPython `decode_hp_hint_args`
/// (support.py:135-150).  The default red candidate list begins with
/// `pc` (Int) before `program` (Ref), so the implicit path is already
/// in RPython-accepted order.
pub(super) fn red_schema(lowerer: &Lowerer, config: &LowererConfig) -> Vec<(String, TokenStream)> {
    // RPython `support.py:135-150 decode_hp_hint_args` parity: malformed
    // marker args panic instead of silently shrinking the schema (see
    // `green_schema` for the full rationale).
    let explicit: Vec<String> = config
        .reds
        .iter()
        .enumerate()
        .map(|(i, expr)| match expr {
            syn::Expr::Path(p) => match p.path.get_ident() {
                Some(id) => id.to_string(),
                None => panic!(
                    "#[jit_interp] reds[{i}]: only bare-ident reds are \
                     supported (matching `JitDriver(reds=['name'])` upstream); \
                     got a multi-segment path: {}",
                    quote::quote!(#expr),
                ),
            },
            _ => panic!(
                "#[jit_interp] reds[{i}]: only bare-ident reds are \
                 supported (matching `JitDriver(reds=['name'])` upstream); \
                 got a non-path expression: {}",
                quote::quote!(#expr),
            ),
        })
        .collect();
    let owned: Vec<String>;
    let explicit_was_provided = !explicit.is_empty();
    let names: Vec<&str> = if explicit_was_provided {
        owned = explicit;
        owned.iter().map(|s| s.as_str()).collect()
    } else {
        // Issue 2.2 (support.py:121 _kind2count): `pc` (Int=1) precedes
        // `program` (Ref=2).  Earlier shape `["program", "pc"]`
        // produced Ref-before-Int order; RPython would reject that in
        // `decode_hp_hint_args` instead of reordering it.
        let mut candidates: Vec<&str> = vec!["pc", "program"];
        if let Some(vable) = config.vable_var.as_ref() {
            candidates.push(vable.as_str());
        }
        // greens have already been validated at `green_schema` (which
        // runs before `red_schema` in the lowering pipeline), so any
        // non-bare-ident at this point is a programmer error worth
        // surfacing — but in `red_schema` the intent is just to filter
        // them out of the default candidate set.  Mirror the green
        // validation strictness instead of accepting silent drops:
        // assert each green entry is a bare ident.
        let green_names: HashSet<String> = config
            .greens
            .iter()
            .enumerate()
            .map(|(i, expr)| match expr {
                syn::Expr::Path(p) => match p.path.get_ident() {
                    Some(id) => id.to_string(),
                    None => panic!(
                        "#[jit_interp] greens[{i}] (re-validated in red_schema): \
                         only bare-ident greens are supported; got a \
                         multi-segment path: {}",
                        quote::quote!(#expr),
                    ),
                },
                _ => panic!(
                    "#[jit_interp] greens[{i}] (re-validated in red_schema): \
                     only bare-ident greens are supported; got a non-path \
                     expression: {}",
                    quote::quote!(#expr),
                ),
            })
            .collect();
        candidates
            .into_iter()
            .filter(|name| !green_names.contains(*name))
            .collect()
    };
    let mut out: Vec<(u8, String, TokenStream)> = Vec::new();
    for name in names {
        match lowerer.bindings.get(name) {
            Some(binding) => {
                let (rank, ty_tok) = match binding.kind {
                    BindingKind::Int => (1u8, quote::quote!(majit_ir::Type::Int)),
                    BindingKind::Ref => (2u8, quote::quote!(majit_ir::Type::Ref)),
                    BindingKind::Float => (3u8, quote::quote!(majit_ir::Type::Float)),
                };
                out.push((rank, name.to_string(), ty_tok));
            }
            None => {
                if explicit_was_provided {
                    // RPython `support.py:135-150 decode_hp_hint_args`:
                    // every declared red name must appear in the
                    // function's local bindings; an unknown name is a
                    // declaration-vs-body mismatch that upstream
                    // surfaces as `KeyError` from
                    // `decode_hp_hint_args`'s `varlist[i]` lookup.
                    // Silent drop would mask the mismatch and let a
                    // misshaped `BC_JIT_MERGE_POINT` payload propagate
                    // into the dispatch JitCode.
                    panic!(
                        "#[jit_interp] reds: declared red `{name}` is not \
                         bound in the function body. Either remove it from \
                         the explicit `reds = [...]` list or introduce a \
                         matching `let {name}` binding (support.py:135-150 \
                         decode_hp_hint_args parity).",
                    );
                }
                // Implicit-default branch: the candidate list above is
                // speculative (`pc` / `program` / vable_var); only the
                // intersection with the actual bindings becomes the red
                // schema, matching pyre's per-#[jit_interp] minimal red
                // shape (some sites use `program` only, others add a
                // virtualizable).
            }
        }
    }
    assert_kind_sorted("reds", &out);
    out.into_iter().map(|(_, n, t)| (n, t)).collect()
}

fn assert_kind_sorted(label: &str, vars: &[(u8, String, TokenStream)]) {
    for pair in vars.windows(2) {
        let (prev_rank, prev_name, _) = &pair[0];
        let (rank, name, _) = &pair[1];
        if prev_rank > rank {
            panic!(
                "support.py:135-150 decode_hp_hint_args parity: JitDriver {} \
                 must be declared in int -> ref -> float order. '{}' appears \
                 before '{}', but rank {} > {}. Reorder the variables instead \
                 of relying on pyre to sort them.",
                label, prev_name, name, prev_rank, rank
            );
        }
    }
}

/// Slice 1: Dispatch JitCode body lowerer.
///
/// Lowers a `#[jit_interp]` function's `while { jit_merge_point!(); ...
/// match opcode { ... } }` dispatch loop into a single dispatch JitCode
/// body. Mirrors RPython `pypy/module/pypyjit/interp_jit.py:82-94`
/// portal + `pypy/interpreter/pyopcode.py:168-181` dispatch_bytecode.
///
/// Output IR shape (filled in across Slice 1 tasks 1.1-1.7):
/// 1. `BC_LIVE` (canonical entry) — Task 1.1 (this task)
/// 2. `BC_JIT_MERGE_POINT(_C)` (interp_jit.py:88-90 jit_merge_point hook) — Task 1.2
/// 3. `BC_LOOP_HEADER` — Task 1.2
/// 4. pre-dispatch ops, source-order (interp_jit.py:91-93) — Task 1.3
/// 5. opcode/oparg fetch + pc advance (pyopcode.py:171-181) — Task 1.4
/// 6. dispatch chain via existing `BC_GOTO_IF_NOT_*` ops
///    (jtransform.py:196-225 conditional fusion) — Task 1.5
/// 7. per-arm `BC_INLINE_CALL sub_jitcode_idx` (jtransform.py:473-482) — Task 1.6
/// 8. loop close `BC_GOTO 0` — Task 1.7
/// 9. default arm: typed return / dispatch-exit ABI — Task 1.7
pub(crate) fn lower_dispatch_body(
    config: &LowererConfig,
    func_block: &syn::Block,
    classified_arms: &[crate::jit_interp::classify::ClassifiedArm],
) -> Option<GeneratedJitCodeBody> {
    let mut lowerer = Lowerer::new(Some(config));
    // RPython `assembler` emits exactly one `-live-` per source point;
    // the dispatch JitCode's leading-dummy `BC_LIVE` is already emitted
    // by `codegen_trace.rs`'s `__dispatch_jitcode_<fn>` wrapper as
    // `let _live_offset_patch = __builder.live_placeholder();`
    // (matching every per-arm sub-JitCode's single leading dummy).
    // Earlier pyre revisions emitted a SECOND entry placeholder here,
    // landing two consecutive `BC_LIVE` markers at the dispatch
    // JitCode's start — divergent from main's per-arm shape.  The
    // duplicate is retired; subsequent ops below are emitted in order.

    // Loop back-edge target (interp_jit.py:95-100): bound here so that the
    // back-edge GOTOs at the end of each matched arm land at the
    // jit_merge_point, re-entering the portal on the next iteration.
    let loop_start_label = lowerer.alloc_label();
    lowerer.emit_aux(quote::quote! { let #loop_start_label = __builder.new_label(); });
    lowerer.emit_label_def(&loop_start_label);

    // Register the portal-input bindings at proc-macro time before
    // resolve_greens (below) consults the binding map.  These are
    // pure proc-macro-time HashMap inserts — no runtime code is emitted
    // here; the corresponding __builder.ensure_*_regs calls that allocate
    // the register slots at runtime appear later (after loop_header).
    //
    // r0 = program (Ref). We install the binding but do NOT advance
    // next_reg (the Int-bank counter) because r0 lives in the Ref bank.
    lowerer.bindings.insert(
        "program".to_owned(),
        Binding {
            reg: 0,
            kind: BindingKind::Ref,
            depends_on_stack: false,
        },
    );
    // i0 = pc (Int). Advance next_reg past i0 so opcode_reg gets i1.
    lowerer.bindings.insert(
        "pc".to_owned(),
        Binding {
            reg: 0,
            kind: BindingKind::Int,
            depends_on_stack: false,
        },
    );
    lowerer.next_reg = lowerer.next_reg.max(1);

    // A.3.6.1 (jtransform.py:1693): bind body-local `let` stmts that
    // appear BEFORE `jit_merge_point!()` in the dispatch while-body, so
    // that consumer-declared `greens = [<body-local>]` (e.g. aheui-jit's
    // `greens = [stackok]`) resolve via `lowerer.bindings` when
    // `emit_promote_greens` and `resolve_greens` consult it below.
    let _ = bind_pre_merge_point_stmts(&mut lowerer, func_block);
    if lowerer.dispatch_tainted_reason.is_some() {
        return None;
    }

    // A.3.5 (jtransform.py:1693-1714): emit a `-live-` + `<kind>_guard_value`
    // pair for each declared green BEFORE `jit_merge_point`.  Forces every
    // green to a constant at trace time; `pyjitpl.py:1530` asserts all greens
    // are constants when the merge point is reached.
    emit_promote_greens(&mut lowerer, config);

    // jtransform.py:1707-1712 returns `[op3, op1, op2]` from
    // `handle_jit_marker__jit_merge_point`:
    //
    //     op1 = SpaceOperation('jit_merge_point', args, None)
    //     op2 = SpaceOperation('-live-', [], None)
    //     # ^^^ we need a -live- for the case of do_recursive_call()
    //     op3 = SpaceOperation('-live-', [], None)
    //     # and one for inlined short preambles
    //     return ops + [op3, op1, op2]
    //
    // i.e. `promote_greens` results, then a `-live-` (op3, used by
    // inlined short preambles), then the merge-point op (op1), then
    // another `-live-` (op2, used by recursive-call resume).  Pyre
    // emits the trailing `-live-` (op2) below; this PRE-merge-point
    // `-live-` (op3) was previously missing — restore it for parity.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote::quote! { __builder.live_placeholder(); },
    );

    // interp_jit.py:88-90 — pypyjitdriver.jit_merge_point(...) at the
    // portal entry. A.3.2 fills greens; A.3.3 fills reds;
    // jdindex is the `__jdindex: i64` runtime parameter of
    // `__dispatch_jitcode_*` (jtransform.py:1704 portal_jd.index).
    //
    // resolve_greens requires the portal-input bindings to be installed
    // (done just above) so that green ident → register-byte lookup works.
    let (greens_i, greens_r, greens_f) = resolve_greens(&lowerer, config);
    let (reds_i, reds_r, reds_f) = resolve_reds(&lowerer, config);
    let greens_i_lit: Vec<_> = greens_i.iter().map(|b| quote::quote!(#b)).collect();
    let greens_r_lit: Vec<_> = greens_r.iter().map(|b| quote::quote!(#b)).collect();
    let greens_f_lit: Vec<_> = greens_f.iter().map(|b| quote::quote!(#b)).collect();
    let reds_i_lit: Vec<_> = reds_i.iter().map(|b| quote::quote!(#b)).collect();
    let reds_r_lit: Vec<_> = reds_r.iter().map(|b| quote::quote!(#b)).collect();
    let reds_f_lit: Vec<_> = reds_f.iter().map(|b| quote::quote!(#b)).collect();
    lowerer.emit_op(
        OpMeta::linear(OpKind::JitMergePoint, vec![], vec![]),
        quote::quote! {
            // __jdindex: jtransform.py:1704 portal_jd.index threaded as runtime param.
            __builder.jit_merge_point(
                __jdindex,
                &[#(#greens_i_lit),*],
                &[#(#greens_r_lit),*],
                &[#(#greens_f_lit),*],
                &[#(#reds_i_lit),*],
                &[#(#reds_r_lit),*],
                &[#(#reds_f_lit),*],
            );
        },
    );
    // jtransform.py:1707-1712 emits a trailing `-live-` after
    // `jit_merge_point`, used by recursive-call and short-preamble resume
    // paths.
    lowerer.emit_op(
        OpMeta::live_marker(),
        quote::quote! { __builder.live_placeholder(); },
    );

    // jtransform.py:1714-1723 `handle_jit_marker__loop_header` emits the
    // `loop_header` op at the source-code `can_enter_jit()` call site
    // (interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` inside
    // `jump_absolute`).  In the lowered bytecode this lands at each
    // back-edge — NOT immediately after `jit_merge_point` at the top of
    // the dispatch loop.  pyre's `lower_dispatch_chain` emits one
    // `loop_header(__jdindex)` op per matched arm, just before the
    // `goto loop_start_label` back-edge — see the per-arm emission
    // there.  Earlier pyre revisions emitted a single `loop_header` here
    // (right after `jit_merge_point`); that placement was structurally
    // backwards relative to RPython (which has loop_header at back-edges)
    // and is now retired.

    // Slice 1.4: allocate input registers for the dispatch JitCode.
    //
    // The dispatch JitCode reads from two reds at its entry point:
    //   r0 — `program` (bytecode slice, Ref bank)
    //   i0 — `pc`      (program counter, Int bank)
    // Optional virtualizable input uses r1, because r0 is already the
    // bytecode object in the dispatch JitCode entry ABI.
    //
    // Mirrors interp_jit.py:67-70 reds=['frame', 'ec'] for PyPy's portal;
    // pyre's per-#[jit_interp] dispatch uses (program, pc) as the minimal
    // reds.  The actual value seeding (binding i0/r0 to the outer Rust
    // `program`/`pc` at trace time) is Slice 2's `__trace_*` rewrite.
    //
    // TODO: derive env_param_name / pc_var_name from LowererConfig or
    // macro config instead of hard-coding "program"/"pc".
    lowerer.emit_aux(quote::quote! {
        __builder.ensure_r_regs(2u16);  // r0 = program; r1 = virtualizable when present
        __builder.ensure_i_regs(1u16);  // i0 = pc (Int bank)
    });

    // interp_jit.py:91-93: lower stmts that appear between jit_merge_point
    // and the dispatch match in source order. This covers promote calls
    // (jtransform.py:608-615: hint(x, promote=True) → -live- + guard_value),
    // opcode/oparg fetch (Task 1.4), and any other pre-dispatch stmts.
    //
    // Walk: find the dispatch match, then find the while body that contains it,
    // then iterate stmts before the match-containing stmt.
    let _lowered_pre_dispatch = lower_pre_dispatch_stmts(&mut lowerer, func_block);
    // A.2.3a fail-closed install gate: if pre-dispatch lowering detected
    // a structurally unrecognized inner construct (currently only the
    // `Expr::While` shape mismatch path), abort dispatch JitCode body
    // generation and return None. The caller (`codegen_trace.rs:81-88`)
    // emits an empty body for the dispatch_jitcode_fn, so the runtime
    // gate at `codegen_state.rs:786-823` misses `BC_GETARRAYITEM_GC_I`
    // and refuses to register the singleton.
    if lowerer.dispatch_tainted_reason.is_some() {
        return None;
    }

    // Task 1.5: emit dispatch chain.
    // pyopcode.py:183+ if/elif chain over opcode value.
    // jtransform.py:196-225 optimize_goto_if_not fuses int_eq + goto_if_not
    // into goto_if_not_int_eq/iiL (BC_GOTO_IF_NOT_INT_EQ).
    let default_label =
        lower_dispatch_chain(&mut lowerer, classified_arms, config, &loop_start_label);

    // Task 1.7: default arm typed return.
    // Bind default_label here so the dispatch chain's fall-through GOTO lands
    // at the typed-return emission (interp_jit.py:95-100 return boundary).
    lowerer.emit_label_def(&default_label);

    // Lower the function-final return expression (the stmt after the while loop).
    // dispatch_minimal returns `state.a` (i64) → BC_INT_RETURN.
    let return_expr = func_block.stmts.iter().rev().find_map(|s| match s {
        syn::Stmt::Expr(e, None) => Some(e),
        _ => None,
    });
    match return_expr.and_then(|e| lowerer.lower_value_expr(e)) {
        Some(binding) => {
            let reg = binding.reg;
            // blackhole.py:841-857 — typed return reads the source register
            // of its declared kind. Walker keeps `reg` alive upstream via
            // OpMeta::terminal's reads list.
            let (read_reg, emitter) = match binding.kind {
                BindingKind::Int => (
                    Register::int(reg),
                    quote::quote! { __builder.int_return(#reg as u16); },
                ),
                BindingKind::Ref => (
                    Register::ref_(reg),
                    quote::quote! { __builder.ref_return(#reg as u16); },
                ),
                BindingKind::Float => (
                    Register::float(reg),
                    quote::quote! { __builder.float_return(#reg as u16); },
                ),
            };
            lowerer.emit_op(OpMeta::terminal(vec![read_reg]), emitter);
        }
        None => {
            // No lowerable return expr: emit void_return.
            // blackhole.py:859-862 — void_return has no operand and no reads.
            lowerer.emit_op(
                OpMeta::terminal(Vec::new()),
                quote::quote! { __builder.void_return(); },
            );
        }
    }

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    // Slice (audit Issue #5) — surface the dispatch JitCode's
    // (name, IR Type) green / red schemas to the install path so it
    // can populate `JitDriverStaticData::vars` via
    // `JitDriver::declare_schema`.  Computed BEFORE moving
    // `lowerer.statements` because the helpers borrow `&lowerer`.
    let green_schema_pairs = green_schema(&lowerer, config);
    let red_schema_pairs = red_schema(&lowerer, config);
    let statements = lowerer.statements;
    Some(GeneratedJitCodeBody {
        body: quote::quote! {
            #(#statements)*
        },
        liveness_prebuild,
        green_schema: green_schema_pairs,
        red_schema: red_schema_pairs,
    })
}
