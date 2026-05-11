use super::*;

impl<'c> Lowerer<'c> {
    pub(super) fn stmt_modifies_jit_state(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_modifies_jit_state(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_modifies_jit_state(&init.expr)),
            _ => false,
        }
    }

    /// Check if an expression touches the storage pool or state fields.
    pub(super) fn expr_touches_storage(&self, expr: &Expr) -> bool {
        self.expr_has_jit_state_reference(expr) || self.expr_references_unknown_local(expr)
    }

    /// Walks the expression looking for `Path` references to locals that
    /// are not visible in the generated trace function. The trace
    /// function's scope only carries `program`, `pc`, `__op`, plus the
    /// macro-managed `__builder` / `__ctx` / `__sym` handles. Any other
    /// bare identifier (e.g. user mainloop locals `op`, `stackok`,
    /// `is_queue`) cannot survive verbatim emission inside the trace
    /// function and must abort lowering instead.
    ///
    /// Type identifiers (uppercase first letter) and qualified paths are
    /// allowed — they resolve at module scope.
    fn expr_references_unknown_local(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(p) => {
                if let Some(ident) = p.path.get_ident() {
                    let s = ident.to_string();
                    // Whitelist trace-function scope.
                    if matches!(
                        s.as_str(),
                        "program" | "pc" | "__op" | "__sym" | "__ctx" | "__builder"
                    ) {
                        return false;
                    }
                    // Type / module / constant idents start uppercase or
                    // underscore-uppercase (e.g. `OP_MOV`, `VAL_PORT`).
                    let first = s.chars().next();
                    if first.map_or(false, |c| c.is_uppercase() || c == '_') {
                        return false;
                    }
                    // Bare lowercase identifier — assume it is a user
                    // local that the trace function will not have.
                    true
                } else {
                    // Qualified path (`a::b::c`) resolves at module scope.
                    false
                }
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_references_unknown_local(receiver)
                    || args.iter().any(|a| self.expr_references_unknown_local(a))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_references_unknown_local(func)
                    || args.iter().any(|a| self.expr_references_unknown_local(a))
            }
            Expr::Binary(ExprBinary { left, right, .. })
            | Expr::Assign(ExprAssign { left, right, .. }) => {
                self.expr_references_unknown_local(left)
                    || self.expr_references_unknown_local(right)
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. })
            | Expr::Try(syn::ExprTry { expr, .. }) => self.expr_references_unknown_local(expr),
            Expr::Field(syn::ExprField { base, .. }) => self.expr_references_unknown_local(base),
            Expr::Index(syn::ExprIndex { expr, index, .. }) => {
                self.expr_references_unknown_local(expr)
                    || self.expr_references_unknown_local(index)
            }
            Expr::Match(m) => {
                self.expr_references_unknown_local(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_references_unknown_local(&arm.body))
            }
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_references_unknown_local(cond)
                    || then_branch.stmts.iter().any(|s| {
                        if let Stmt::Expr(e, _) = s {
                            self.expr_references_unknown_local(e)
                        } else {
                            false
                        }
                    })
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, e)| self.expr_references_unknown_local(e))
            }
            // Literals, returns without expression, etc. are safe.
            _ => false,
        }
    }

    fn expr_modifies_jit_state(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Assign(ExprAssign { left, right, .. }) => {
                self.expr_is_jit_state_place(left)
                    || self.expr_modifies_jit_state(left)
                    || self.expr_modifies_jit_state(right)
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_modifies_jit_state(receiver)
                    || args.iter().any(|arg| self.expr_modifies_jit_state(arg))
            }
            Expr::Block(expr_block) => expr_block
                .block
                .stmts
                .iter()
                .any(|stmt| self.stmt_modifies_jit_state(stmt)),
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_modifies_jit_state(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_modifies_jit_state(stmt))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, expr)| self.expr_modifies_jit_state(expr))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_modifies_jit_state(func)
                    || args.iter().any(|arg| self.expr_modifies_jit_state(arg))
            }
            Expr::Binary(ExprBinary { left, right, .. }) => {
                self.expr_modifies_jit_state(left) || self.expr_modifies_jit_state(right)
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. }) => self.expr_modifies_jit_state(expr),
            Expr::Match(m) => {
                self.expr_modifies_jit_state(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_modifies_jit_state(&arm.body))
            }
            Expr::Field(_)
            | Expr::Index(_)
            | Expr::Path(_)
            | Expr::Lit(_)
            | Expr::Try(_)
            | Expr::Loop(_)
            | Expr::While(_)
            | Expr::ForLoop(_)
            | Expr::Break(_)
            | Expr::Continue(_)
            | Expr::Return(_)
            | Expr::Macro(_) => false,
            _ => false,
        }
    }

    fn expr_has_jit_state_reference(&self, expr: &Expr) -> bool {
        if self.expr_is_jit_state_place(expr) {
            return true;
        }
        // RPython parity: any reference to the state root (e.g.
        // `state.selected_dispatch_mut()`) touches the JIT-managed state.
        // The trace function does not have `state` in scope; without
        // this guard, lower_local's runtime-constant fallback would
        // emit the verbatim expression and fail to compile. Catching
        // the bare `state` path forces the macro to either lower the
        // expression to IR (Step 4b MethodCall lowering) or skip the
        // arm (treat as residual / not traced).
        if self.expr_is_state_root(expr) {
            return true;
        }
        match expr {
            Expr::Assign(ExprAssign { left, right, .. })
            | Expr::Binary(ExprBinary { left, right, .. }) => {
                self.expr_has_jit_state_reference(left) || self.expr_has_jit_state_reference(right)
            }
            Expr::MethodCall(ExprMethodCall { receiver, args, .. }) => {
                self.expr_has_jit_state_reference(receiver)
                    || args
                        .iter()
                        .any(|arg| self.expr_has_jit_state_reference(arg))
            }
            Expr::Call(ExprCall { func, args, .. }) => {
                self.expr_has_jit_state_reference(func)
                    || args
                        .iter()
                        .any(|arg| self.expr_has_jit_state_reference(arg))
            }
            Expr::Block(expr_block) => expr_block
                .block
                .stmts
                .iter()
                .any(|stmt| self.stmt_touches_jit_state(stmt)),
            Expr::If(ExprIf {
                cond,
                then_branch,
                else_branch,
                ..
            }) => {
                self.expr_has_jit_state_reference(cond)
                    || then_branch
                        .stmts
                        .iter()
                        .any(|stmt| self.stmt_touches_jit_state(stmt))
                    || else_branch
                        .as_ref()
                        .is_some_and(|(_, expr)| self.expr_has_jit_state_reference(expr))
            }
            Expr::Cast(ExprCast { expr, .. })
            | Expr::Paren(ExprParen { expr, .. })
            | Expr::Reference(ExprReference { expr, .. })
            | Expr::Unary(ExprUnary { expr, .. })
            | Expr::Try(syn::ExprTry { expr, .. }) => self.expr_has_jit_state_reference(expr),
            Expr::Index(syn::ExprIndex { expr, index, .. }) => {
                self.expr_has_jit_state_reference(expr) || self.expr_has_jit_state_reference(index)
            }
            Expr::Field(syn::ExprField { base, .. }) => self.expr_has_jit_state_reference(base),
            Expr::Match(m) => {
                self.expr_has_jit_state_reference(&m.expr)
                    || m.arms
                        .iter()
                        .any(|arm| self.expr_has_jit_state_reference(&arm.body))
            }
            _ => false,
        }
    }

    fn stmt_touches_jit_state(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr, _) => self.expr_has_jit_state_reference(expr),
            Stmt::Local(local) => local
                .init
                .as_ref()
                .is_some_and(|init| self.expr_has_jit_state_reference(&init.expr)),
            _ => false,
        }
    }

    fn expr_is_jit_state_place(&self, expr: &Expr) -> bool {
        let config = match self.config {
            Some(c) => c,
            None => return false,
        };
        match expr {
            Expr::Field(field) => {
                if !self.expr_is_state_root(&field.base) {
                    return false;
                }
                let member = match &field.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(idx) => idx.index.to_string(),
                };
                config.state_scalars.contains_key(&member)
                    || config.state_arrays.contains_key(&member)
                    || config.state_virt_arrays.contains_key(&member)
            }
            Expr::Index(syn::ExprIndex { expr, .. }) => self.expr_is_jit_state_place(expr),
            _ => false,
        }
    }

    fn expr_is_state_root(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Path(path) => path.path.is_ident("state"),
            Expr::Paren(ExprParen { expr, .. }) | Expr::Reference(ExprReference { expr, .. }) => {
                self.expr_is_state_root(expr)
            }
            _ => false,
        }
    }

    // ── Core lowering (unchanged logic) ──────────────────────────────


}
