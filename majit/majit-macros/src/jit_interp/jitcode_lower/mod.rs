mod api;
mod dispatch;
mod helpers;
mod jit_state_analysis;
mod liveness;
mod lower_control;
mod lower_stmt;
mod lower_vable;
mod lower_value;
mod lowerer;

pub use api::GeneratedJitCodeBody;
#[allow(unused_imports)]
pub(crate) use api::{
    CallerLocalLayout, assign_caller_local_layout, generate_inline_helper_jitcode_with_calls,
    inline_helper_param_counts, inline_helper_param_layout,
    try_generate_jitcode_body_parts_with_caller_bindings,
};
#[allow(unused_imports)]
pub use api::{
    try_generate_jitcode_body, try_generate_jitcode_body_parts,
    try_generate_jitcode_body_with_config, try_generate_jitcode_body_with_config_parts,
};
pub(crate) use dispatch::lower_dispatch_body;
pub(crate) use helpers::classify_param_type;
pub(super) use helpers::helper_policy_path;

// Re-export submodule items for sibling-submodule access via `use super::*`.
// These appear unused in mod.rs itself but are consumed by submodules and tests.
mod reexports {
    #![allow(unused_imports)]
    pub(super) use super::api::bind_pre_merge_point_stmts;
    pub(super) use super::dispatch::{
        collect_arm_caller_locals, collect_pat_bound_idents, dispatch_arm_inline_call_tokens,
        emit_promote_greens, find_dispatch_loop_body, green_schema, is_jit_merge_point_macro,
        lower_dispatch_chain, lower_pre_dispatch_stmts, red_schema, resolve_greens, resolve_reds,
    };
    pub(super) use super::helpers::{
        binding_kind_for_inline_policy, block_has_loop_control, expr_has_loop_control,
        extract_block_tail_int, extract_bool_branch_values, extract_branch_int,
        extract_pat_literals, extract_pat_value_tokens, extract_stmts, inline_builder_path,
        inline_call_tokens, inline_float_arg_tokens, inline_int_arg_tokens, inline_prebuild_path,
        inline_ref_arg_tokens, int_arg_regs, is_supported_float_type, is_supported_int_cast,
        is_supported_ref_type, opcode_for_assign_binop, opcode_for_binop, stmt_has_loop_control,
        typed_call_arg_tokens,
    };
    pub(super) use super::liveness::{
        annotate_live_markers_with_liveness, compute_per_marker_liveness, get_liveness_info,
        liveness_prebuild_tokens, liveness_triple, liveness_triple_from_reads, maybe_dump_liveness,
        remove_repeated_live, rewrite_live_marker_statements_with_triples,
    };
    pub(super) use super::lowerer::Lowerer;
}
#[allow(unused_imports)]
use reexports::*;

use std::collections::{BTreeSet, HashMap, HashSet};

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use super::call_policy_byte::{
    INT_DONT_LOOK_INSIDE, INT_DONT_LOOK_INSIDE_CANNOT_RAISE, INT_ELIDABLE,
    INT_ELIDABLE_CANNOT_RAISE, INT_ELIDABLE_OR_MEMERROR, INT_INLINE, INT_LOOP_INVARIANT,
    INT_MAY_FORCE, INT_RELEASE_GIL, REF_DONT_LOOK_INSIDE, REF_DONT_LOOK_INSIDE_CANNOT_RAISE,
    REF_ELIDABLE, REF_ELIDABLE_CANNOT_RAISE, REF_ELIDABLE_OR_MEMERROR, REF_LOOP_INVARIANT,
    REF_MAY_FORCE, VOID_DONT_LOOK_INSIDE, VOID_DONT_LOOK_INSIDE_CANNOT_RAISE, VOID_LOOP_INVARIANT,
    VOID_MAY_FORCE, VOID_RELEASE_GIL,
};
use super::codegen_trace::{
    block_contains_match, find_dispatch_match, is_promote_call_path, stmt_contains_match,
};
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprLit, ExprMatch,
    ExprMethodCall, ExprParen, ExprPath, ExprReference, ExprUnary, FnArg, Ident, ItemFn, Lit,
    Local, Pat, Path, ReturnType, Stmt, Type, UnOp,
};

// Duplicated from majit-translate::hints — proc-macro crates cannot depend
// on heavy library crates, so we inline the small enum + classifier here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VirtualizableHintKind {
    AccessDirectly,
    FreshVirtualizable,
    ForceVirtualizable,
}

pub(super) fn classify_virtualizable_hint_segments<'a, I>(
    segments: I,
) -> Option<VirtualizableHintKind>
where
    I: IntoIterator<Item = &'a str>,
{
    match segments.into_iter().last().unwrap_or_default() {
        "hint_access_directly" => Some(VirtualizableHintKind::AccessDirectly),
        "hint_fresh_virtualizable" => Some(VirtualizableHintKind::FreshVirtualizable),
        "hint_force_virtualizable" => Some(VirtualizableHintKind::ForceVirtualizable),
        _ => None,
    }
}

// ── LowererConfig ────────────────────────────────────────────────────

/// Configuration for state_fields-aware JitCode lowering.
///
/// Built from `JitInterpConfig` at proc-macro time and passed to the Lowerer
/// to recognize state-field reads/writes, virtualizable accesses, I/O shims,
/// and helper-call policies.
#[derive(Clone)]
pub struct LowererConfig {
    /// Canonical I/O func path → shim ident.
    pub(super) io_shims: Vec<(Vec<String>, Ident)>,
    /// Canonical helper func path → explicit or inferred call policy.
    pub(super) calls: Vec<(Vec<String>, CallPolicySpec)>,
    /// Whether top-level traced calls should auto-infer helper policy.
    pub(super) auto_calls: bool,
    /// Virtualizable variable name (normalized, e.g., "frame").
    /// RPython jtransform.py: `is_virtualizable_getset()` uses this to check
    /// if a field access target is the virtualizable variable.
    pub(super) vable_var: Option<String>,
    /// Ref-register assigned to the virtualizable input variable.
    ///
    /// RPython `MIFrame.setup_call(original_boxes)` distributes portal args
    /// by kind before opimpls consume `v_inst` / `v_base`.  The generated
    /// observer JitCode fragment receives the virtualizable as its first Ref
    /// input, so the line-by-line graph variable is `registers_r[0]`.
    pub(super) vable_input_ref_reg: Option<u16>,
    /// Field name → (field_index, field_type).
    /// RPython: `vinfo.static_field_to_extra_box[fieldname]` → index.
    pub(super) vable_fields: HashMap<String, (usize, ValueKind)>,
    /// Array name → (array_index, item_type).
    /// RPython: `vinfo.array_field_counter[fieldname]` → index.
    pub(super) vable_arrays: HashMap<String, (usize, ValueKind)>,
    /// State field scalars: field_name → global_field_index.
    pub(super) state_scalars: HashMap<String, usize>,
    /// State field arrays (flattened): field_name → global_array_index.
    pub(super) state_arrays: HashMap<String, usize>,
    /// State field virtualizable arrays: field_name → virt_array_index.
    /// These emit GETARRAYITEM_RAW_I/SETARRAYITEM_RAW instead of element-level tracking.
    pub(super) state_virt_arrays: HashMap<String, usize>,
    /// Green-variable expressions for `jit_merge_point` / `promote_greens`.
    ///
    /// Source: `JitInterpConfig.greens` (mod.rs:65) — the `greens = [...]` list
    /// from the `#[jit_interp]` attribute.  Consumed by A.3.2 (green register
    /// byte list emit) and A.3.5 (promote_greens pre-portal emission).
    pub greens: Vec<Expr>,
    /// Per-green explicit lltype subtype tag (`: str` / `: unicode` / etc.)
    /// from `JitInterpConfig.green_type_tags`.  Lockstep with `greens`;
    /// `None` for an untagged entry.  Consumed by `green_schema()` so the
    /// `JitDriverStaticData::green_args_spec` reflects the upstream
    /// `warmspot.py:663 _green_args_spec` STR/UNICODE distinction
    /// instead of collapsing to `GreenType::Ref`.
    pub green_type_tags: Vec<Option<crate::jit_interp::green_type_tag::GreenTypeTag>>,
    /// Slice (audit Issue #6) — explicit red declarations.  Source:
    /// `JitInterpConfig.reds` (mod.rs).  Empty = use the default
    /// `[program, pc(+ optional vable)]` candidate list.
    pub(super) reds: Vec<Expr>,
    /// Canonical state-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `state`.
    /// Source: `JitInterpConfig.state_type` Ident.
    pub(super) state_type_name: String,
    /// Canonical env-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `program`
    /// (the env parameter — convention fixed at the dispatch portal-input
    /// installer below). Source: `JitInterpConfig.env_type` Ident.
    pub(super) env_type_name: String,
}

pub(super) const MAX_HELPER_CALL_ARITY: usize = 16;

pub(super) fn classify_virtualizable_hint_syn_path(path: &Path) -> Option<VirtualizableHintKind> {
    let segments = path
        .segments
        .iter()
        .map(|seg| seg.ident.to_string())
        .collect::<Vec<_>>();
    classify_virtualizable_hint_segments(segments.iter().map(String::as_str))
}

pub(crate) struct InlineHelperJitCode {
    pub body: TokenStream,
    pub return_reg: u16,
    pub return_kind: InlineReturnKind,
    /// Helper-side per-marker liveness prebuild tokens. Threaded into the
    /// parent's `__prebuild_jitcode_liveness_*` so RPython
    /// `pyjitpl.py:2255 finish_setup`'s "all `-live-` entries land in
    /// `asm.all_liveness` before the snapshot" invariant is preserved
    /// when the helper is invoked at trace time. Without this thread, the
    /// helper's `JitCodeBuilder::finalize_liveness(asm)` at trace time
    /// would register triples the snapshot didn't see, growing
    /// `staticdata.liveness_info` past the install-time freeze and
    /// tripping the `__trace_*` snapshot-invariant assertion.
    pub liveness_prebuild: TokenStream,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlineReturnKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
pub(super) enum CallPolicySpec {
    Explicit(crate::jit_interp::CallPolicyKind),
    Infer,
}

#[derive(Clone, Copy)]
pub(super) enum InferenceFailureMode {
    ReturnNone,
    Panic,
}

#[derive(Clone, Copy)]
pub(super) enum ValueKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CondCallEffectSlot {
    CanRaise,
    /// `EF_CANNOT_RAISE` — `call.py:303 getcalldescr`'s non-elidable
    /// `else` branch.  Selected by a `residual_*_cannot_raise` policy
    /// when the producer statically knows the callee cannot raise.
    CannotRaise,
    ElidableCanRaise,
    ElidableCannotRaise,
    ElidableOrMemerror,
    LoopInvariant,
}

impl CondCallEffectSlot {
    pub(super) fn token(self) -> TokenStream {
        match self {
            Self::CanRaise => quote! { majit_metainterp::EffectInfoSlot::CanRaise },
            Self::CannotRaise => quote! { majit_metainterp::EffectInfoSlot::CannotRaise },
            Self::ElidableCanRaise => quote! { majit_metainterp::EffectInfoSlot::ElidableCanRaise },
            Self::ElidableCannotRaise => {
                quote! { majit_metainterp::EffectInfoSlot::ElidableCannotRaise }
            }
            Self::ElidableOrMemerror => {
                quote! { majit_metainterp::EffectInfoSlot::ElidableOrMemerror }
            }
            Self::LoopInvariant => quote! { majit_metainterp::EffectInfoSlot::LoopInvariant },
        }
    }

    pub(super) fn can_raise(self) -> bool {
        matches!(
            self,
            Self::CanRaise | Self::ElidableCanRaise | Self::ElidableOrMemerror
        )
    }

    pub(super) fn is_elidable(self) -> bool {
        matches!(
            self,
            Self::ElidableCanRaise | Self::ElidableCannotRaise | Self::ElidableOrMemerror
        )
    }

    /// Emit the `EffectInfoSlot` token for a statically-known wrapped
    /// `CallPolicyKind`.  Used by the `*Wrapped` lowering arms so the
    /// registered call-target descr carries the real effect classification
    /// (`call.py:282-303 getcalldescr`) rather than a blanket `CanRaise`.
    /// Falls back to `CanRaise` for `MayForce` / `ReleaseGil` / `Inline*`
    /// kinds whose conditional-call slot is `None` in
    /// `call_policy_effect_slot`; the actual call surface dispatches them
    /// through dedicated `call_may_force_*` / `call_release_gil_*` /
    /// inline-helper paths that ignore the registered slot.
    pub(super) fn for_wrapped_kind(kind: crate::jit_interp::CallPolicyKind) -> TokenStream {
        call_policy_effect_slot(kind)
            .map(|slot| slot.token())
            .unwrap_or_else(|| quote! { majit_metainterp::EffectInfoSlot::CanRaise })
    }

    /// Emit a runtime `match __policy { ... }` expression that resolves the
    /// `EffectInfoSlot` from the helper's `_jit_helper_policy` byte.  Used
    /// by the `Infer` lowering paths where the policy kind is only known at
    /// runtime (`call.py:282-303 getcalldescr` analyzer chain executed on
    /// the live byte).
    pub(super) fn slot_from_policy_tokens() -> TokenStream {
        quote! {
            match __policy {
                #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE
                | #VOID_MAY_FORCE | #INT_MAY_FORCE | #REF_MAY_FORCE
                | #VOID_RELEASE_GIL | #INT_RELEASE_GIL => {
                    majit_metainterp::EffectInfoSlot::CanRaise
                }
                #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                    majit_metainterp::EffectInfoSlot::CannotRaise
                }
                #INT_ELIDABLE | #REF_ELIDABLE => {
                    majit_metainterp::EffectInfoSlot::ElidableCanRaise
                }
                #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #REF_LOOP_INVARIANT => {
                    majit_metainterp::EffectInfoSlot::LoopInvariant
                }
                #INT_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_CANNOT_RAISE => {
                    majit_metainterp::EffectInfoSlot::ElidableCannotRaise
                }
                #INT_ELIDABLE_OR_MEMERROR | #REF_ELIDABLE_OR_MEMERROR => {
                    majit_metainterp::EffectInfoSlot::ElidableOrMemerror
                }
                _ => majit_metainterp::EffectInfoSlot::CanRaise,
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CallResultKind {
    Void,
    Int,
    Ref,
    Float,
}

impl ValueKind {
    pub(super) fn from_ident(ident: &Ident) -> Self {
        match ident.to_string().as_str() {
            "ref" => Self::Ref,
            "float" => Self::Float,
            _ => Self::Int,
        }
    }
}

pub(super) fn call_policy_effect_slot(
    kind: crate::jit_interp::CallPolicyKind,
) -> Option<CondCallEffectSlot> {
    use crate::jit_interp::CallPolicyKind as K;
    match kind {
        K::ResidualVoid
        | K::ResidualVoidWrapped
        | K::ResidualInt
        | K::ResidualIntWrapped
        | K::ResidualRefWrapped
        | K::ResidualFloatWrapped => Some(CondCallEffectSlot::CanRaise),

        K::ResidualVoidCannotRaise
        | K::ResidualVoidCannotRaiseWrapped
        | K::ResidualIntCannotRaise
        | K::ResidualIntCannotRaiseWrapped
        | K::ResidualRefCannotRaiseWrapped
        | K::ResidualFloatCannotRaiseWrapped => Some(CondCallEffectSlot::CannotRaise),

        K::LoopInvariantVoid
        | K::LoopInvariantVoidWrapped
        | K::LoopInvariantInt
        | K::LoopInvariantIntWrapped
        | K::LoopInvariantRefWrapped
        | K::LoopInvariantFloatWrapped => Some(CondCallEffectSlot::LoopInvariant),

        K::ElidableInt
        | K::ElidableIntWrapped
        | K::ElidableRefWrapped
        | K::ElidableFloatWrapped => Some(CondCallEffectSlot::ElidableCanRaise),
        K::ElidableIntCannotRaise
        | K::ElidableIntCannotRaiseWrapped
        | K::ElidableRefCannotRaiseWrapped
        | K::ElidableFloatCannotRaiseWrapped => Some(CondCallEffectSlot::ElidableCannotRaise),
        K::ElidableIntOrMemerror
        | K::ElidableIntOrMemerrorWrapped
        | K::ElidableRefOrMemerrorWrapped
        | K::ElidableFloatOrMemerrorWrapped => Some(CondCallEffectSlot::ElidableOrMemerror),

        K::MayForceVoid
        | K::MayForceVoidWrapped
        | K::MayForceInt
        | K::MayForceIntWrapped
        | K::MayForceRefWrapped
        | K::MayForceFloatWrapped
        | K::ReleaseGilVoid
        | K::ReleaseGilVoidWrapped
        | K::ReleaseGilInt
        | K::ReleaseGilIntWrapped
        | K::ReleaseGilFloatWrapped
        | K::InlineInt
        | K::InlineRef
        | K::InlineFloat => None,
    }
}

pub(super) fn call_policy_result_kind(
    kind: crate::jit_interp::CallPolicyKind,
) -> Option<CallResultKind> {
    use crate::jit_interp::CallPolicyKind as K;
    match kind {
        K::ResidualVoid
        | K::ResidualVoidWrapped
        | K::ResidualVoidCannotRaise
        | K::ResidualVoidCannotRaiseWrapped
        | K::MayForceVoid
        | K::MayForceVoidWrapped
        | K::ReleaseGilVoid
        | K::ReleaseGilVoidWrapped
        | K::LoopInvariantVoid
        | K::LoopInvariantVoidWrapped => Some(CallResultKind::Void),

        K::ResidualInt
        | K::ResidualIntWrapped
        | K::ResidualIntCannotRaise
        | K::ResidualIntCannotRaiseWrapped
        | K::MayForceInt
        | K::MayForceIntWrapped
        | K::ReleaseGilInt
        | K::ReleaseGilIntWrapped
        | K::LoopInvariantInt
        | K::LoopInvariantIntWrapped
        | K::ElidableInt
        | K::ElidableIntWrapped
        | K::ElidableIntCannotRaise
        | K::ElidableIntCannotRaiseWrapped
        | K::ElidableIntOrMemerror
        | K::ElidableIntOrMemerrorWrapped
        | K::InlineInt => Some(CallResultKind::Int),

        K::ResidualRefWrapped
        | K::ResidualRefCannotRaiseWrapped
        | K::MayForceRefWrapped
        | K::LoopInvariantRefWrapped
        | K::ElidableRefWrapped
        | K::ElidableRefCannotRaiseWrapped
        | K::ElidableRefOrMemerrorWrapped
        | K::InlineRef => Some(CallResultKind::Ref),

        K::ResidualFloatWrapped
        | K::ResidualFloatCannotRaiseWrapped
        | K::MayForceFloatWrapped
        | K::ReleaseGilFloatWrapped
        | K::LoopInvariantFloatWrapped
        | K::ElidableFloatWrapped
        | K::ElidableFloatCannotRaiseWrapped
        | K::ElidableFloatOrMemerrorWrapped
        | K::InlineFloat => Some(CallResultKind::Float),
    }
}

pub(super) fn call_policy_is_wrapped(kind: crate::jit_interp::CallPolicyKind) -> bool {
    use crate::jit_interp::CallPolicyKind as K;
    matches!(
        kind,
        K::ResidualVoidWrapped
            | K::ResidualVoidCannotRaiseWrapped
            | K::MayForceVoidWrapped
            | K::ReleaseGilVoidWrapped
            | K::LoopInvariantVoidWrapped
            | K::ResidualIntWrapped
            | K::ResidualIntCannotRaiseWrapped
            | K::MayForceIntWrapped
            | K::ReleaseGilIntWrapped
            | K::LoopInvariantIntWrapped
            | K::ElidableIntWrapped
            | K::ElidableIntCannotRaiseWrapped
            | K::ElidableIntOrMemerrorWrapped
            | K::ResidualRefWrapped
            | K::ResidualRefCannotRaiseWrapped
            | K::MayForceRefWrapped
            | K::LoopInvariantRefWrapped
            | K::ElidableRefWrapped
            | K::ElidableRefCannotRaiseWrapped
            | K::ElidableRefOrMemerrorWrapped
            | K::ResidualFloatWrapped
            | K::ResidualFloatCannotRaiseWrapped
            | K::MayForceFloatWrapped
            | K::ReleaseGilFloatWrapped
            | K::LoopInvariantFloatWrapped
            | K::ElidableFloatWrapped
            | K::ElidableFloatCannotRaiseWrapped
            | K::ElidableFloatOrMemerrorWrapped
    )
}

pub(super) fn call_result_matches_binding(
    result_kind: CallResultKind,
    binding_kind: BindingKind,
) -> bool {
    matches!(
        (result_kind, binding_kind),
        (CallResultKind::Int, BindingKind::Int)
            | (CallResultKind::Ref, BindingKind::Ref)
            | (CallResultKind::Float, BindingKind::Float)
    )
}

/// Build the runtime guard that wraps `live_placeholder*` for an
/// inferred-policy callee.  See [`LiveMarkerCondition`] for the
/// PRE-EXISTING-ADAPTATION rationale and the convergence path
/// (Tasks #146/#235) that retires this wrapper.
pub(super) fn inferred_policy_live_condition(func: &Expr, can_raise_codes: &[u8]) -> TokenStream {
    let policy_path =
        helper_policy_path(func).expect("inferred helper policy requires a path expression");
    let patterns = can_raise_codes
        .iter()
        .copied()
        .map(|code| quote! { #code })
        .collect::<Vec<_>>();
    if patterns.is_empty() {
        return quote! { false };
    }
    quote! {{
        let (__policy, _, _, _, _, _) = #policy_path();
        matches!(__policy, #(#patterns)|*)
    }}
}

pub(super) fn inferred_conditional_call_policy_check(func_args_empty: bool) -> TokenStream {
    let loop_invariant_arm = if func_args_empty {
        quote! { #VOID_LOOP_INVARIANT => {} }
    } else {
        quote! {
            #VOID_LOOP_INVARIANT => panic!(
                "conditional_call!: arguments not supported for loop-invariant function",
            )
        }
    };
    quote! {
        match __policy {
            // Void-return, non-forcing calldescrs accepted by
            // jtransform.py:1677.  `VOID_DONT_LOOK_INSIDE_CANNOT_RAISE`
            // is the EF_CANNOT_RAISE void surface; jtransform's gate
            // accepts both EF_CAN_RAISE and EF_CANNOT_RAISE — the
            // latter just skips the trailing `-live-` per
            // `jtransform.py:1681 calldescr_canraise`.
            #VOID_DONT_LOOK_INSIDE | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
            #loop_invariant_arm,
            // Void-return but rejected by jtransform.py:1677's
            // `check_forces_virtual_or_virtualizable` gate or by the
            // release-gil structural surface.
            #VOID_MAY_FORCE | #VOID_RELEASE_GIL => panic!(
                "conditional_call! cannot dispatch MayForce / ReleaseGil callees",
            ),
            // PyPy `call.py:getcalldescr` checks actual return type before
            // effect flags; `conditional_call!` is the void-result opcode.
            #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE | #INT_MAY_FORCE | #INT_RELEASE_GIL
            | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE | #INT_ELIDABLE_OR_MEMERROR
            | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE
            | #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
            | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                "conditional_call! requires a void-return helper policy",
            ),
            _ => panic!(
                "conditional_call! could not infer a PyPy-compatible helper policy",
            ),
        }
    }
}

pub(super) fn inferred_conditional_call_value_policy_check(
    value_kind: BindingKind,
    func_args_empty: bool,
) -> TokenStream {
    match value_kind {
        BindingKind::Int => {
            let loop_invariant_arm = if func_args_empty {
                quote! { #INT_LOOP_INVARIANT => {} }
            } else {
                quote! {
                    #INT_LOOP_INVARIANT => panic!(
                        "conditional_call_elidable!: arguments not supported for loop-invariant function",
                    )
                }
            };
            quote! {
                match __policy {
                    // INT_DONT_LOOK_INSIDE_CANNOT_RAISE: int residual
                    // EF_CANNOT_RAISE accepted on the int value branch
                    // per `call.py:300`.
                    #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE | #INT_ELIDABLE_CANNOT_RAISE
                    | #INT_ELIDABLE_OR_MEMERROR | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
                    #loop_invariant_arm,
                    #INT_MAY_FORCE | #INT_RELEASE_GIL => panic!(
                        "conditional_call_elidable! cannot dispatch MayForce / ReleaseGil callees",
                    ),
                    // VOID_DONT_LOOK_INSIDE_CANNOT_RAISE (28) and
                    // REF_DONT_LOOK_INSIDE_CANNOT_RAISE (30) are wrong
                    // result kind for the int value branch.
                    #VOID_DONT_LOOK_INSIDE | #VOID_MAY_FORCE | #VOID_RELEASE_GIL
                    | #VOID_LOOP_INVARIANT | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_ELIDABLE
                    | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                    | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                        "conditional_call_elidable! value/result kind mismatch for inferred helper policy",
                    ),
                    _ => panic!(
                        "conditional_call_elidable! could not infer a PyPy-compatible helper policy",
                    ),
                }
            }
        }
        BindingKind::Ref => {
            let loop_invariant_arm = if func_args_empty {
                quote! { #REF_LOOP_INVARIANT => {} }
            } else {
                quote! {
                    #REF_LOOP_INVARIANT => panic!(
                        "conditional_call_elidable!: arguments not supported for loop-invariant function",
                    )
                }
            };
            quote! {
                match __policy {
                    // REF_DONT_LOOK_INSIDE_CANNOT_RAISE: ref residual
                    // EF_CANNOT_RAISE accepted on the ref value branch.
                    #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                    | #REF_DONT_LOOK_INSIDE | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => {},
                    #loop_invariant_arm,
                    #REF_MAY_FORCE => panic!(
                        "conditional_call_elidable! cannot dispatch MayForce callees",
                    ),
                    // VOID_DONT_LOOK_INSIDE_CANNOT_RAISE (28) and
                    // INT_DONT_LOOK_INSIDE_CANNOT_RAISE (29) are wrong
                    // result kind for the ref value branch.
                    #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE
                    | #VOID_MAY_FORCE | #INT_MAY_FORCE | #VOID_RELEASE_GIL | #INT_RELEASE_GIL
                    | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                    | #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE
                    | #INT_ELIDABLE_OR_MEMERROR => panic!(
                        "conditional_call_elidable! value/result kind mismatch for inferred helper policy",
                    ),
                    _ => panic!(
                        "conditional_call_elidable! could not infer a PyPy-compatible helper policy",
                    ),
                }
            }
        }
        BindingKind::Float => quote! {
            panic!("Conditional call does not support floats");
        },
    }
}

pub(super) fn inferred_record_known_result_policy_check(result_kind: BindingKind) -> TokenStream {
    match result_kind {
        BindingKind::Int => quote! {
            match __policy {
                #INT_ELIDABLE | #INT_ELIDABLE_CANNOT_RAISE | #INT_ELIDABLE_OR_MEMERROR => {},
                // INT_DONT_LOOK_INSIDE_CANNOT_RAISE (29): int residual
                // EF_CANNOT_RAISE — not elidable, rejected here.
                #INT_DONT_LOOK_INSIDE | #INT_MAY_FORCE | #INT_RELEASE_GIL | #INT_LOOP_INVARIANT
                | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => panic!(
                    "record_known_result! requires an elidable helper policy",
                ),
                #VOID_DONT_LOOK_INSIDE | #VOID_MAY_FORCE | #VOID_RELEASE_GIL
                | #VOID_LOOP_INVARIANT | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE | #REF_ELIDABLE
                | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR
                | #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE => panic!(
                    "record_known_result! result kind mismatch for inferred helper policy",
                ),
                _ => panic!(
                    "record_known_result! could not infer a PyPy-compatible helper policy",
                ),
            }
        },
        BindingKind::Ref => quote! {
            match __policy {
                #REF_ELIDABLE | #REF_ELIDABLE_CANNOT_RAISE | #REF_ELIDABLE_OR_MEMERROR => {},
                // REF_DONT_LOOK_INSIDE_CANNOT_RAISE (30): ref residual
                // EF_CANNOT_RAISE — not elidable, rejected here.
                #REF_LOOP_INVARIANT | #REF_DONT_LOOK_INSIDE | #REF_MAY_FORCE
                | #REF_DONT_LOOK_INSIDE_CANNOT_RAISE => panic!(
                    "record_known_result! requires an elidable helper policy",
                ),
                #VOID_DONT_LOOK_INSIDE | #INT_DONT_LOOK_INSIDE | #INT_ELIDABLE
                | #VOID_MAY_FORCE | #INT_MAY_FORCE | #VOID_RELEASE_GIL | #INT_RELEASE_GIL
                | #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE | #INT_DONT_LOOK_INSIDE_CANNOT_RAISE
                | #VOID_LOOP_INVARIANT | #INT_LOOP_INVARIANT | #INT_ELIDABLE_CANNOT_RAISE
                | #INT_ELIDABLE_OR_MEMERROR => panic!(
                    "record_known_result! result kind mismatch for inferred helper policy",
                ),
                _ => panic!(
                    "record_known_result! could not infer a PyPy-compatible helper policy",
                ),
            }
        },
        BindingKind::Float => quote! {
            panic!("record_known_result does not support floats");
        },
    }
}

impl LowererConfig {
    pub fn new(
        io_shims: &[(Path, Ident)],
        calls: &[crate::jit_interp::CallEntry],
        auto_calls: bool,
        vable_decl: Option<&crate::jit_interp::VirtualizableDecl>,
        state_fields_cfg: Option<&crate::jit_interp::StateFieldsConfig>,
        greens: &[Expr],
        green_type_tags: &[Option<crate::jit_interp::green_type_tag::GreenTypeTag>],
        reds: &[Expr],
        state_type: &Ident,
        env_type: &Ident,
    ) -> Self {
        let io_shims = io_shims
            .iter()
            .map(|(p, s)| (canonical_path_segments(p), s.clone()))
            .collect();
        let calls = calls
            .iter()
            .map(|entry| {
                let spec = match entry.policy {
                    Some(kind) => CallPolicySpec::Explicit(kind),
                    None => CallPolicySpec::Infer,
                };
                (canonical_path_segments(&entry.path), spec)
            })
            .collect();
        let (vable_var, vable_input_ref_reg, vable_fields, vable_arrays) =
            if let Some(decl) = vable_decl {
                let var = Some(decl.var_name.to_string());
                let fields = decl
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        (
                            f.name.to_string(),
                            (i, ValueKind::from_ident(&f.field_type)),
                        )
                    })
                    .collect();
                let arrays = decl
                    .arrays
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (a.name.to_string(), (i, ValueKind::from_ident(&a.item_type))))
                    .collect();
                (var, Some(0), fields, arrays)
            } else {
                (None, None, HashMap::new(), HashMap::new())
            };
        let (state_scalars, state_arrays, state_virt_arrays) = if let Some(sf) = state_fields_cfg {
            use crate::jit_interp::StateFieldKind;
            let mut scalars = HashMap::new();
            let mut arrays = HashMap::new();
            let mut virt_arrays = HashMap::new();
            let mut scalar_idx = 0usize;
            let mut array_idx = 0usize;
            let mut virt_array_idx = 0usize;
            for f in &sf.fields {
                match &f.kind {
                    StateFieldKind::Scalar { .. } => {
                        scalars.insert(f.name.to_string(), scalar_idx);
                        scalar_idx += 1;
                    }
                    StateFieldKind::Array(_) => {
                        arrays.insert(f.name.to_string(), array_idx);
                        array_idx += 1;
                    }
                    StateFieldKind::VirtArray(_) => {
                        virt_arrays.insert(f.name.to_string(), virt_array_idx);
                        virt_array_idx += 1;
                    }
                    // Opaque fields are not registered in any index map —
                    // the lowering layer must not see them as state slots.
                    StateFieldKind::Opaque(_) => {}
                }
            }
            (scalars, arrays, virt_arrays)
        } else {
            (HashMap::new(), HashMap::new(), HashMap::new())
        };
        Self {
            io_shims,
            calls,
            auto_calls,
            vable_var,
            vable_input_ref_reg,
            vable_fields,
            vable_arrays,
            state_scalars,
            state_arrays,
            state_virt_arrays,
            greens: greens.to_vec(),
            green_type_tags: green_type_tags.to_vec(),
            reds: reds.to_vec(),
            state_type_name: state_type.to_string(),
            env_type_name: env_type.to_string(),
        }
    }

    pub fn with_vable_input_ref_reg(&self, reg: u16) -> Self {
        let mut cloned = self.clone();
        if cloned.vable_var.is_some() {
            cloned.vable_input_ref_reg = Some(reg);
        }
        cloned
    }
}

pub(super) fn canonical_path_segments(path: &Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect()
}

pub(super) fn canonical_member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

pub(super) fn canonical_expr_segments(expr: &Expr) -> Option<Vec<String>> {
    match expr {
        Expr::Path(path) => Some(canonical_path_segments(&path.path)),
        Expr::Field(field) => {
            let mut segments = canonical_expr_segments(&field.base)?;
            segments.push(canonical_member_name(&field.member));
            Some(segments)
        }
        Expr::Paren(paren) => canonical_expr_segments(&paren.expr),
        Expr::Reference(reference) => canonical_expr_segments(&reference.expr),
        _ => None,
    }
}

pub(super) fn unwrap_ref_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => expr,
        _ => expr,
    }
}

pub(super) fn expr_matches_local_name(expr: &Expr, expected: &str) -> bool {
    match expr {
        Expr::Path(path) => path
            .path
            .get_ident()
            .map(|ident| ident == expected)
            .unwrap_or(false),
        Expr::Reference(reference) => expr_matches_local_name(&reference.expr, expected),
        Expr::Paren(paren) => expr_matches_local_name(&paren.expr, expected),
        _ => false,
    }
}

pub(super) fn named_member(member: &syn::Member) -> Option<String> {
    match member {
        syn::Member::Named(ident) => Some(ident.to_string()),
        _ => None,
    }
}

// ── Lowerer ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum BindingKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
pub(super) struct Binding {
    pub(super) reg: u16,
    pub(super) kind: BindingKind,
    pub(super) depends_on_stack: bool,
}

/// Mirror of RPython `rpython/jit/codewriter/flatten.py:Register(kind, index)`.
/// Each emitted register carries its bank with it; the liveness walker
/// (`liveness.py:33-79`) keeps a single `set()` of `Register` objects per
/// marker, and `assembler.py:225-232 get_liveness_info(args, kind)` filters
/// by `reg.kind == kind` at encode time to split into the per-bank bitsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) struct Register {
    /// Total order is `(kind, index)` so `BTreeSet<Register>` iterates in
    /// kind-grouped order — convenient for encoders that emit per-bank
    /// bitsets.
    pub(super) kind: BindingKind,
    pub(super) index: u8,
}

impl Register {
    /// Construct a `Register` from a `(kind, u16-index)` pair, asserting that
    /// the index fits in the `assembler.py:225` bitset addressing range
    /// (0..=255). The lowerer's `Lowerer::next_reg` counter already obeys
    /// this bound; the assert traps regressions where a u16 reg leaked from
    /// outside that bound.
    #[allow(dead_code)]
    pub(super) fn new(kind: BindingKind, index: u16) -> Self {
        assert!(
            index <= u8::MAX as u16,
            "Register index {} exceeds u8 (assembler.py:225 bitset range)",
            index,
        );
        Self {
            kind,
            index: index as u8,
        }
    }

    /// Per-bank constructor shortcut — `Register::int(0)` mirrors the
    /// RPython sugar of `Register('int', 0)`.
    #[allow(dead_code)]
    pub(super) fn int(index: u16) -> Self {
        Self::new(BindingKind::Int, index)
    }

    #[allow(dead_code)]
    pub(super) fn ref_(index: u16) -> Self {
        Self::new(BindingKind::Ref, index)
    }

    #[allow(dead_code)]
    pub(super) fn float(index: u16) -> Self {
        Self::new(BindingKind::Float, index)
    }

    /// Convenience: build a typed `Register` from a `Binding`.
    #[allow(dead_code)]
    pub(super) fn from_binding(b: &Binding) -> Self {
        Self::new(b.kind, b.reg)
    }

    /// Build a `Vec<Register>` of `Int` from a slice of indices. Used by
    /// emit sites whose reads list is uniformly Int (binop, guard_value,
    /// etc.).
    #[allow(dead_code)]
    pub(super) fn ints(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::int).collect()
    }

    #[allow(dead_code)]
    pub(super) fn refs(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::ref_).collect()
    }

    #[allow(dead_code)]
    pub(super) fn floats(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::float).collect()
    }
}

// ── Op metadata for backward liveness analysis (Phase 4 Epic B) ─────
//
// `op_metadata[i]` describes the i-th emitted op so a downstream backward
// walker (Slice B.2.B) can produce per-marker live sets matching RPython
// `liveness.py:33-79 _compute_liveness_must_continue`. Currently only the
// `LiveMarker` sites are populated — remaining emit sites are migrated in
// Slice B.2.A.ii.
//
// `kind` and `control` are split because future op categories (binop,
// load_const, jump, ...) carry the same `Linear`/`UnconditionalJump`/etc
// shape as several others; control flow is the orthogonal axis the walker
// branches on.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OpKind {
    /// `BC_LIVE` marker emitted before a guard. RPython `flatten.py:259`
    /// `-live-`. Carries no def/use; the walker records the alive set at
    /// this point.
    LiveMarker,
    LoadConstI,
    LoadConstR,
    LoadConstF,
    MoveI,
    MoveR,
    MoveF,
    BinopI,
    UnaryI,
    /// Unconditional `jump` to a target label.
    Jump,
    /// `goto_if_not_*` — conditional branch with fail exit on miss.
    GotoIfNot,
    /// `mark_label` — defines a label.
    MarkLabel,
    /// Any `call_*_typed` / `call_*_args` / `residual_call_*` /
    /// `conditional_call_*` family op. `reads` carries arg regs;
    /// `writes` carries the result reg if the call is value-form.
    Call,
    /// `inline_call_*` family — sub-jitcode invocation.
    InlineCall,
    /// `vable_*` family (getfield/setfield/getarrayitem/setarrayitem/
    /// arraylen/force).
    Vable,
    /// `load_state_*` / `store_state_*` family.
    StateField,
    /// `int_guard_value` / `float_guard_value` / `ref_guard_value`.
    GuardValue,
    /// `record_known_result_*` — pure-call result hint, no real call.
    RecordKnownResult,
    /// `jit_merge_point` portal merge-point marker.
    /// interp_jit.py:88-90 pypyjitdriver.jit_merge_point(...).
    JitMergePoint,
    /// `loop_header` loop-header marker before the dispatch body.
    /// jtransform.py:1714-1718 handle_jit_marker__loop_header.
    LoopHeader,
    /// Builder-side auxiliary statement that emits no BC_* op. Examples:
    /// `let #label = __builder.new_label();` (label allocation), Rust
    /// `let` bindings injected into the generated trace body for
    /// register-side use, sub-jitcode-add helpers. Carries no def/use;
    /// the backward walker treats it as a no-op pass-through.
    Aux,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ControlFlowClass {
    /// Falls through to the next op.
    Linear,
    /// `-live-` marker. Behaves Linear but is the recording point for
    /// backward liveness analysis.
    LiveMarker,
    /// `goto_if_not_*` — emits a fail exit and falls through on no-fail.
    /// Walker treats this both as Linear (fall-through into next op) and
    /// as a branch into the named label (joins at label backward propagation).
    ConditionalGuard,
    /// `jump` — unconditional branch. Walker resets `alive` from the
    /// label's accumulated set instead of the prior fall-through.
    UnconditionalJump,
    /// `mark_label` — defines a label. Walker records the current `alive`
    /// against the label name so forward jumps backward-feed from here.
    LabelDef,
    /// `*_return` family — terminal op with no fall-through and no
    /// successor. Walker resets `alive` to empty; the op's own register
    /// reads are still added as uses so the source value stays live.
    /// blackhole.py:841-862 bhimpl_int_return / ref_return / float_return /
    /// void_return.
    Terminal,
}

/// PRE-EXISTING-ADAPTATION (no upstream counterpart).
///
/// `rpython/jit/codewriter/liveness.py:82-116`'s `-live-` is always
/// unconditional; `jtransform.py:311-312` decides whether to emit one at
/// translation time from `calldescr_canraise(calldescr)`, which is
/// statically known once the calldescr is built.  pyre's macro expansion
/// sees only a runtime helper-policy byte (`__majit_call_policy_<name>()`)
/// for inferred-policy callees because cross-crate proc macros cannot read
/// another crate's proc-macro-generated function at expand time, so the
/// emit decision is deferred to a runtime guard wrapped around
/// `live_placeholder*`.  `remove_repeated_live` merges adjacent markers
/// only when the run contains at least one unconditional marker (which
/// guarantees emit at this position, so unioning the conditional
/// siblings' reads is safe); a run consisting entirely of conditional
/// markers stays unmerged so that each marker's BC_LIVE captures only
/// its own alive set when its condition holds — unioning them would
/// over-capture vs PyPy's per-site `liveness.py:111-115`
/// `liveset.update(live[1:])` (which only sees `-live-`s that actually
/// exist).  Convergence path: once the ann/rtyper EffectInfo
/// infrastructure (Tasks #146/#235) exposes the helper's analyzer
/// outcome at expand time, this conditional surface retires and
/// `LiveMarkerCondition` plus `live_marker_if` /
/// `inferred_policy_live_condition` can be removed.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(super) struct LiveMarkerCondition {
    /// Boolean expression evaluated both in the JitCode builder body and in
    /// the liveness prebuild body.
    pub(super) emit: TokenStream,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub(super) struct OpMeta {
    pub(super) kind: OpKind,
    /// Source registers (uses). Each `Register` carries `kind` directly per
    /// `flatten.py:Register(kind, index)` so the liveness walker stays a
    /// single-bag set and the encoder (`assembler.py:225-232`) splits into
    /// per-bank bitsets on demand.
    pub(super) reads: Vec<Register>,
    /// Destination registers (defs).
    pub(super) writes: Vec<Register>,
    /// Branch target label, for control-flow ops.
    pub(super) target_label: Option<Ident>,
    /// `-live-` marker TLabel operands. RPython stores every TLabel in
    /// the instruction tuple; a marker can carry more than one.
    pub(super) live_target_labels: Vec<Ident>,
    /// Optional guard for the physical `BC_LIVE` emission.  Unconditional
    /// markers match normal RPython ssarepr.  Conditional markers are the
    /// strict-parity bridge for inferred helper policies whose can-raise
    /// answer is represented by a runtime policy byte.
    pub(super) live_condition: Option<LiveMarkerCondition>,
    pub(super) control: ControlFlowClass,
}

#[allow(dead_code)]
impl OpMeta {
    pub(super) fn live_marker() -> Self {
        Self::live_marker_with(Vec::new(), Vec::new())
    }

    /// Conditional `-live-` for inferred-policy callees.  See
    /// [`LiveMarkerCondition`] for the PRE-EXISTING-ADAPTATION rationale
    /// and convergence path.
    pub(super) fn live_marker_if(condition: TokenStream) -> Self {
        let mut marker = Self::live_marker();
        marker.live_condition = Some(LiveMarkerCondition { emit: condition });
        marker
    }

    /// `-live-` marker carrying explicit force-alive register args and/or
    /// target labels whose accumulated alive sets should fold in. Mirrors
    /// RPython `rpython/jit/codewriter/liveness.py:44-53`'s handling of
    /// `-live-` insns whose tuple tail includes Register / TLabel
    /// entries. The lowerer currently never emits such enriched markers
    /// itself, but parity-aware consumers (snapshot helpers that synth
    /// extra live regs around an inline call) can produce them through
    /// this constructor.
    #[allow(dead_code)]
    pub(super) fn live_marker_with(reads: Vec<Register>, live_target_labels: Vec<Ident>) -> Self {
        Self {
            kind: OpKind::LiveMarker,
            reads,
            writes: Vec::new(),
            target_label: None,
            live_target_labels,
            live_condition: None,
            control: ControlFlowClass::LiveMarker,
        }
    }

    /// Linear op with explicit reads/writes. The most common shape —
    /// load_const, move, binop, unary, call, vable, state-field,
    /// guard_value, record_known_result, inline_call.
    pub(super) fn linear(kind: OpKind, reads: Vec<Register>, writes: Vec<Register>) -> Self {
        Self {
            kind,
            reads,
            writes,
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Linear,
        }
    }

    /// Unconditional jump to `target`.
    pub(super) fn jump(target: Ident) -> Self {
        Self {
            kind: OpKind::Jump,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::UnconditionalJump,
        }
    }

    /// Conditional guard branching to `target` on miss. `cond_reg` is
    /// the read register feeding the guard.
    pub(super) fn conditional_guard(cond_reg: Register, target: Ident) -> Self {
        Self {
            kind: OpKind::GotoIfNot,
            reads: vec![cond_reg],
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::ConditionalGuard,
        }
    }

    /// Two-register conditional guard for `goto_if_not_int_eq(a, b, target)`.
    /// jtransform.py:196-225 `optimize_goto_if_not` fuses `int_eq + goto_if_not`
    /// into `goto_if_not_int_eq/iiL`. Both `a_reg` and `b_reg` are read uses.
    pub(super) fn conditional_guard_int_eq(
        a_reg: Register,
        b_reg: Register,
        target: Ident,
    ) -> Self {
        Self {
            kind: OpKind::GotoIfNot,
            reads: vec![a_reg, b_reg],
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::ConditionalGuard,
        }
    }

    /// Label definition site. Walker uses `target` to associate the
    /// current `alive` set with the label name.
    pub(super) fn label_def(target: Ident) -> Self {
        Self {
            kind: OpKind::MarkLabel,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: Some(target),
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::LabelDef,
        }
    }

    /// Builder-side aux op (label allocation, Rust `let` bindings,
    /// sub-jitcode add). Linear, no def/use.
    pub(super) fn aux() -> Self {
        Self {
            kind: OpKind::Aux,
            reads: Vec::new(),
            writes: Vec::new(),
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Linear,
        }
    }

    /// Terminal op (`*_return` family). No fall-through, no branch target.
    /// `reads` carries the source register so the walker still keeps it
    /// alive upstream; the walker resets the alive set on encountering
    /// this op (no successor to inherit from).
    pub(super) fn terminal(reads: Vec<Register>) -> Self {
        Self {
            kind: OpKind::Aux,
            reads,
            writes: Vec::new(),
            target_label: None,
            live_target_labels: Vec::new(),
            live_condition: None,
            control: ControlFlowClass::Terminal,
        }
    }
}

#[derive(Default)]
pub(super) struct LoweredSequence {
    pub(super) statements: Vec<TokenStream>,
    pub(super) op_metadata: Vec<OpMeta>,
}

impl LoweredSequence {
    pub(super) fn new(statements: Vec<TokenStream>, op_metadata: Vec<OpMeta>) -> Self {
        assert_eq!(
            statements.len(),
            op_metadata.len(),
            "RPython ssarepr.insns parity requires statement/op_metadata streams to stay paired"
        );
        Self {
            statements,
            op_metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_pat(code: &str) -> Pat {
        let match_code = format!("match x {{ {code} => () }}");
        let expr: syn::ExprMatch = syn::parse_str(&match_code).expect("failed to parse match");
        expr.arms.into_iter().next().unwrap().pat
    }

    #[test]
    fn liveness_records_alive_at_marker_after_use() {
        // [marker, linear reads=[5]]
        // backward: linear adds 5 → alive={5}; marker saves {5}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(5)])]);
    }

    #[test]
    fn liveness_def_kills_use_kept() {
        // [marker, linear1 reads=[3], linear2 writes=[3] reads=[5]]
        // backward: linear2 adds 5 (writes 3 first but alive empty);
        //           linear1 adds 3; marker saves {3, 5}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), Register::ints(&[3])),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(3), Register::int(5)])],
        );
    }

    #[test]
    fn liveness_jump_overwrites_alive_with_label_set() {
        // [marker, jump L1, mark_label L2, reads=[7], mark_label L1, reads=[9]]
        // backward single pass:
        //   reads(9): alive={9}
        //   mark_label L1: label_alive[L1]={9}, alive stays {9}
        //   reads(7): alive={9,7}
        //   mark_label L2: label_alive[L2]={9,7}, alive stays {9,7}
        //   jump L1: alive = label_alive[L1] = {9}
        //   marker: save {9}
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l2.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[7]), vec![]),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(9)])]);
    }

    #[test]
    fn liveness_back_edge_loop_reaches_fixed_point() {
        // [mark_label START, marker, reads=[2], jump START]
        // pass 1: jump finds label_alive[START]={}, then reads(2) → {2}, marker={2}, mark_label sets START={2}.
        // pass 2: jump finds {2}, reads(2) keeps {2}, marker {2}, label unchanged → done.
        let start = format_ident!("LOOP_START");
        let ops = vec![
            OpMeta::label_def(start.clone()),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[2]), vec![]),
            OpMeta::jump(start.clone()),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(2)])]);
    }

    #[test]
    fn liveness_conditional_guard_reads_cond_and_unions_branch_target() {
        // [marker, conditional_guard cond_reg=4 → ELSE, reads=[6], mark_label ELSE, reads=[8]]
        // backward:
        //   reads(8): alive={8}
        //   mark_label ELSE: label_alive[ELSE]={8}
        //   reads(6): alive={8, 6}  (fall-through past ELSE in backward order)
        //
        //   Wait, "fall-through past ELSE" backward direction means we already passed reads(8) and
        //   mark_label, now at reads(6). reads(6) sets alive={8,6}.
        //
        //   conditional_guard target=ELSE, reads=[4]: alive folds in label_alive[ELSE]={8}
        //   → alive={8,6} (already had 8) ∪ {8} = {8,6}; then add reads [4] → {8,6,4}.
        //   marker: save {8,6,4}.
        let else_label = format_ident!("ELSE");
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::conditional_guard(Register::int(4), else_label.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[6]), vec![]),
            OpMeta::label_def(else_label.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[8]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([
                Register::int(4),
                Register::int(6),
                Register::int(8),
            ])],
        );
    }

    #[test]
    fn liveness_marker_with_explicit_args_force_alive() {
        // [marker_with([7]), reads=[5]]
        // marker carries `7` as a force-alive register; backward walk
        // adds 5 (linear) then folds 7 into alive at marker → {5, 7}.
        let ops = vec![
            OpMeta::live_marker_with(Register::ints(&[7]), Vec::new()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[5]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(5), Register::int(7)])],
        );
    }

    #[test]
    fn liveness_marker_with_target_label_unions_label_set() {
        // [marker_with([], target=L1), reads=[3], jump L1, label_def L1, reads=[9]]
        // pass 1: reads(9) alive={9}; label L1 saved {9}; jump L1 → alive={9};
        //         reads(3) alive={9,3}; marker fold L1 alive={9,3}∪{9}={9,3}; saved.
        // pass 2: stable.
        let l1 = format_ident!("L1");
        let ops = vec![
            OpMeta::live_marker_with(Vec::new(), vec![l1.clone()]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(3), Register::int(9)])],
        );
    }

    #[test]
    fn liveness_marker_with_multiple_target_labels_unions_all_label_sets() {
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let ops = vec![
            OpMeta::live_marker_with(Vec::new(), vec![l1.clone(), l2.clone()]),
            OpMeta::jump(l1.clone()),
            OpMeta::label_def(l2.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[7]), vec![]),
            OpMeta::label_def(l1.clone()),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[9]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(
            result,
            vec![BTreeSet::from([Register::int(7), Register::int(9)])],
        );
    }

    #[test]
    fn remove_repeated_live_is_no_op_when_runs_have_one_marker() {
        // [marker, reads=[1], marker, reads=[2]] — two markers but each is
        // separated by a non-marker op, so no run to collapse.
        let mut ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), vec![]),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[2]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len())
            .map(|i| quote! { /* op #i */ let _ = #i; })
            .collect();
        let before_len = ops.len();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), before_len);
        assert_eq!(stmts.len(), before_len);
    }

    #[test]
    fn remove_repeated_live_collapses_consecutive_markers() {
        // [marker(reads=[1]), marker(reads=[2]), label_def L, marker, reads=[3]]
        // RPython: collapse the run before reads=[3] into a single marker
        // carrying union({1, 2}) (label L stays as a separate op kept in
        // place between original positions, though its position relative
        // to the merged marker shifts to before per RPython).
        let l = format_ident!("L");
        let mut ops = vec![
            OpMeta::live_marker_with(Register::ints(&[1]), Vec::new()),
            OpMeta::live_marker_with(Register::ints(&[2]), Vec::new()),
            OpMeta::label_def(l.clone()),
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        // Resulting layout: [label_def L, merged_marker(reads={1,2}), reads=[3]].
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0].control, ControlFlowClass::LabelDef));
        assert!(matches!(ops[1].control, ControlFlowClass::LiveMarker));
        assert_eq!(ops[1].reads, vec![Register::int(1), Register::int(2)]);
        assert!(matches!(ops[2].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_preserves_all_marker_target_labels() {
        let l1 = format_ident!("L1");
        let l2 = format_ident!("L2");
        let mut ops = vec![
            OpMeta::live_marker_with(Register::ints(&[1]), vec![l2.clone()]),
            OpMeta::live_marker_with(Register::ints(&[2]), vec![l1.clone()]),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert_eq!(ops[0].reads, vec![Register::int(1), Register::int(2)]);
        let labels: Vec<String> = ops[0]
            .live_target_labels
            .iter()
            .map(|label| label.to_string())
            .collect();
        assert_eq!(labels, vec!["L1".to_string(), "L2".to_string()]);
        assert!(matches!(ops[1].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_keeps_conditional_only_runs_unmerged() {
        // A run consisting entirely of conditional markers stays
        // unmerged: unioning their reads would over-capture vs PyPy's
        // per-site `liveness.py:111-115` `liveset.update(live[1:])`
        // (which only ever sees `-live-`s that actually exist).  Each
        // marker's BC_LIVE fires (or not) on its own condition and
        // captures only its own alive set.
        let mut ops = vec![
            OpMeta::live_marker_if(quote! { policy_a() }),
            OpMeta::live_marker_if(quote! { policy_b() }),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert!(matches!(ops[1].control, ControlFlowClass::LiveMarker));
        assert!(ops[0].live_condition.is_some());
        assert!(ops[1].live_condition.is_some());
        assert!(matches!(ops[2].control, ControlFlowClass::Linear));
    }

    #[test]
    fn remove_repeated_live_drops_conditions_when_run_includes_unconditional_marker() {
        // An unconditional marker mixed in with conditional ones forces
        // the merged result to be unconditional — PyPy emits at this
        // position regardless, so the conditional siblings must follow
        // suit (their alive sets fold in).
        let mut ops = vec![
            OpMeta::live_marker_if(quote! { policy_a() }),
            OpMeta::live_marker_with(Register::ints(&[1]), vec![]),
            OpMeta::live_marker_if(quote! { policy_b() }),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[3]), vec![]),
        ];
        let mut stmts: Vec<TokenStream> = (0..ops.len()).map(|_| quote! { let _ = (); }).collect();
        remove_repeated_live(&mut ops, &mut stmts);
        assert_eq!(ops.len(), 2);
        assert!(matches!(ops[0].control, ControlFlowClass::LiveMarker));
        assert!(
            ops[0].live_condition.is_none(),
            "merged marker must be unconditional when run contains an unconditional marker"
        );
        assert_eq!(ops[0].reads, vec![Register::int(1)]);
        assert!(matches!(ops[1].control, ControlFlowClass::Linear));
    }

    #[test]
    fn rewrite_live_marker_replaces_placeholder_with_triple() {
        // [marker, reads=[1], writes=[2]] — backward walk records {1}
        // alive at marker, def[2] discards 2 from alive carry-over.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), Register::ints(&[2])),
        ];
        let other_marker = quote! { let __probe = 1234; }.to_string();
        let mut stmts: Vec<TokenStream> = vec![
            quote! { let _ = __builder.live_placeholder(); },
            quote! { let __probe = 1234; },
        ];
        rewrite_live_marker_statements_with_triples(&ops, &mut stmts);
        let rendered = stmts[0].to_string();
        assert!(
            rendered.contains("live_placeholder_with_triple"),
            "post-rewrite stmt[0] still uses bare live_placeholder: {rendered}"
        );
        // The triple must reflect the walker output: live_i = [1], live_r = [], live_f = [].
        assert!(
            rendered.contains("& [1u8]"),
            "live_i array missing or wrong: {rendered}"
        );
        assert!(
            rendered.contains("& []"),
            "live_r/live_f empty arrays missing: {rendered}"
        );
        // Non-marker statement is untouched (token-stream equality, since
        // `quote!` strips comments and reformats whitespace).
        assert_eq!(stmts[1].to_string(), other_marker);
    }

    #[test]
    fn rewrite_live_marker_emits_typed_arrays_per_bank() {
        // [marker, reads_int=[3], reads_ref=[7]] — walker records both
        // banks; rewrite must emit per-bank typed arrays (live_i, live_r,
        // live_f) so `live_placeholder_with_triple` sees the right shape.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::int(3), Register::ref_(7)],
                vec![],
            ),
        ];
        let mut stmts: Vec<TokenStream> = vec![
            quote! { let _ = __builder.live_placeholder(); },
            quote! { /* op */ },
        ];
        rewrite_live_marker_statements_with_triples(&ops, &mut stmts);
        let rendered = stmts[0].to_string();
        assert!(rendered.contains("& [3u8]"), "live_i missing: {rendered}");
        assert!(rendered.contains("& [7u8]"), "live_r missing: {rendered}");
    }

    #[test]
    fn liveness_aux_is_pass_through() {
        // [marker, aux, reads=[1]] — aux carries no def/use; alive at marker = {1}.
        let ops = vec![
            OpMeta::live_marker(),
            OpMeta::aux(),
            OpMeta::linear(OpKind::BinopI, Register::ints(&[1]), vec![]),
        ];
        let result = compute_per_marker_liveness(&ops);
        assert_eq!(result, vec![BTreeSet::from([Register::int(1)])]);
    }

    #[test]
    fn get_liveness_info_filters_by_kind() {
        // RPython `assembler.py:225-232 get_liveness_info(args, kind)` parity:
        // a single set of typed Registers projected per bank yields the same
        // sorted u8 indices RPython would emit into the BC_LIVE bitset.
        let set: BTreeSet<Register> = [
            Register::int(3),
            Register::ref_(7),
            Register::int(5),
            Register::float(2),
            Register::ref_(1),
        ]
        .into_iter()
        .collect();
        assert_eq!(get_liveness_info(&set, BindingKind::Int), vec![3u8, 5u8]);
        assert_eq!(get_liveness_info(&set, BindingKind::Ref), vec![1u8, 7u8]);
        assert_eq!(get_liveness_info(&set, BindingKind::Float), vec![2u8]);
    }

    #[test]
    fn liveness_triple_keeps_per_bank_sort_order() {
        // BTreeSet<Register> orders by (kind, index); `liveness_triple` must
        // surface that ordering as three independent sorted Vec<u8> slices,
        // matching `assembler.py:147-157 _encode_liveness` which encodes each
        // bank as a sorted bitset.
        let set: BTreeSet<Register> = [
            Register::float(9),
            Register::int(2),
            Register::ref_(4),
            Register::float(1),
            Register::int(0),
        ]
        .into_iter()
        .collect();
        let (live_i, live_r, live_f) = liveness_triple(&set);
        assert_eq!(live_i, vec![0u8, 2u8]);
        assert_eq!(live_r, vec![4u8]);
        assert_eq!(live_f, vec![1u8, 9u8]);
    }

    #[test]
    fn liveness_triple_empty_when_set_is_empty() {
        let set: BTreeSet<Register> = BTreeSet::new();
        assert_eq!(liveness_triple(&set), (Vec::new(), Vec::new(), Vec::new()));
    }

    #[test]
    fn extract_pat_literals_single() {
        let pat = parse_pat("42");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, Some(vec![42]));
    }

    #[test]
    fn extract_pat_literals_or() {
        let pat = parse_pat("1 | 2 | 3");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, Some(vec![1, 2, 3]));
    }

    #[test]
    fn extract_pat_literals_wildcard_returns_none() {
        let pat = parse_pat("_");
        let lits = extract_pat_literals(&pat);
        assert_eq!(lits, None);
    }

    fn binding(reg: u16, kind: BindingKind) -> Binding {
        Binding {
            reg,
            kind,
            depends_on_stack: false,
        }
    }

    fn parse_fn(code: &str) -> ItemFn {
        syn::parse_str(code).expect("failed to parse function")
    }

    fn inline_policy_with_kind(
        path: &str,
        kind: crate::jit_interp::CallPolicyKind,
    ) -> crate::jit_interp::CallEntry {
        crate::jit_interp::CallEntry {
            path: syn::parse_str(path).expect("failed to parse path"),
            policy: Some(kind),
        }
    }

    fn inline_policy(path: &str) -> crate::jit_interp::CallEntry {
        inline_policy_with_kind(path, crate::jit_interp::CallPolicyKind::InlineInt)
    }

    fn lowerer_with_call_policy(
        path: &str,
        kind: crate::jit_interp::CallPolicyKind,
    ) -> Lowerer<'static> {
        let path: Path = syn::parse_str(path).expect("failed to parse path");
        Lowerer::new_with_call_policies(
            None,
            vec![(
                canonical_path_segments(&path),
                CallPolicySpec::Explicit(kind),
            )],
            InferenceFailureMode::ReturnNone,
        )
    }

    fn lowerer_with_inferred_call_policy(path: &str) -> Lowerer<'static> {
        let path: Path = syn::parse_str(path).expect("failed to parse path");
        Lowerer::new_with_call_policies(
            None,
            vec![(canonical_path_segments(&path), CallPolicySpec::Infer)],
            InferenceFailureMode::ReturnNone,
        )
    }

    fn parse_call(code: &str) -> ExprCall {
        syn::parse_str(code).expect("failed to parse call")
    }

    fn inline_call_tokens_combined(bindings: &[Binding], result_reg: u16) -> String {
        let (call_match, post_live) = inline_call_tokens(bindings, result_reg);
        format!("{} {}", call_match, post_live)
    }

    #[test]
    fn append_lowered_sequence_keeps_statements_and_metadata_aligned() {
        let mut lowerer = Lowerer::new(None);
        lowerer.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let seq = LoweredSequence::new(
            vec![quote! { __builder.load_const_i_value(1, 42); }],
            vec![OpMeta::linear(
                OpKind::LoadConstI,
                Vec::new(),
                Register::ints(&[1]),
            )],
        );
        lowerer.append_lowered_sequence(seq);
        assert_eq!(lowerer.statements.len(), lowerer.op_metadata.len());
        assert!(matches!(lowerer.op_metadata[1].kind, OpKind::LoadConstI));
    }

    #[test]
    fn record_known_result_metadata_reads_known_result_and_writes_nothing() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ElidableInt);
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        let record = lowerer
            .op_metadata
            .iter()
            .find(|m| matches!(m.kind, OpKind::RecordKnownResult))
            .expect("RecordKnownResult metadata emitted");
        assert_eq!(record.reads, Register::ints(&[0, 1]));
        assert!(record.writes.is_empty());
        // `jtransform.py:311-312` trailing `-live-` after a can-raise
        // record_known_result call.
        let last = lowerer.op_metadata.last().expect("metadata emitted");
        assert!(matches!(last.kind, OpKind::LiveMarker));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("add_fn_ptr_with_slot"));
        assert!(tokens.contains("ElidableCanRaise"));
    }

    #[test]
    fn record_known_result_cannot_raise_elidable_omits_live_marker() {
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise,
        );
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        assert_eq!(lowerer.op_metadata.len(), 1);
        assert!(matches!(
            lowerer.op_metadata[0].kind,
            OpKind::RecordKnownResult
        ));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("ElidableCannotRaise"));
        assert!(!tokens.contains("live_placeholder"));
    }

    #[test]
    fn record_known_result_inferred_policy_validates_and_conditions_live_marker() {
        let mut lowerer = lowerer_with_inferred_call_policy("helper");
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_record_known_result(&expr)
            .expect("record_known_result should lower");

        assert_eq!(lowerer.op_metadata.len(), 2);
        assert!(matches!(
            lowerer.op_metadata[0].kind,
            OpKind::RecordKnownResult
        ));
        assert!(lowerer.op_metadata[1].live_condition.is_some());
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("match __policy"));
        assert!(tokens.contains("requires an elidable helper policy"));
    }

    #[test]
    #[should_panic(expected = "record_known_result! requires an elidable helper policy")]
    fn record_known_result_rejects_non_elidable_policy() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .bindings
            .insert("known".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("record_known_result!(known, helper, arg)").expect("parse macro expr");

        let _ = lowerer.lower_record_known_result(&expr);
    }

    #[test]
    fn conditional_call_loopinvariant_omits_live_marker() {
        // `call.py:249-251 getcalldescr` forbids non-void args for
        // loop-invariant direct_call, so the cond_call shape must
        // also have no func args when the slot is `LoopInvariant`.
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::LoopInvariantVoid,
        );
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper)").expect("parse macro expr");

        lowerer
            .lower_conditional_call(&expr)
            .expect("conditional_call should lower");

        assert_eq!(lowerer.op_metadata.len(), 1);
        assert!(matches!(lowerer.op_metadata[0].kind, OpKind::Call));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("LoopInvariant"));
        assert!(!tokens.contains("live_placeholder"));
    }

    #[test]
    #[should_panic(expected = "arguments not supported for loop-invariant function")]
    fn conditional_call_loopinvariant_rejects_func_args() {
        // `call.py:249-251 getcalldescr` asserts `not NON_VOID_ARGS`
        // for loop-invariant direct_call.  The cond_call macro path
        // mirrors that assert at expansion time.
        let mut lowerer = lowerer_with_call_policy(
            "helper",
            crate::jit_interp::CallPolicyKind::LoopInvariantVoid,
        );
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper, arg)").expect("parse macro expr");

        let _ = lowerer.lower_conditional_call(&expr);
    }

    #[test]
    fn conditional_call_inferred_policy_keeps_runtime_loopinvariant_arg_check() {
        let mut lowerer = lowerer_with_inferred_call_policy("helper");
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper, arg)").expect("parse macro expr");

        lowerer
            .lower_conditional_call(&expr)
            .expect("conditional_call should lower");

        assert!(lowerer.op_metadata.last().unwrap().live_condition.is_some());
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("match __policy"));
        assert!(tokens.contains("arguments not supported for loop-invariant function"));
    }

    #[test]
    #[should_panic(expected = "conditional_call! cannot lower helper policy MayForceVoid")]
    fn conditional_call_rejects_may_force_policy() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::MayForceVoid);
        lowerer
            .bindings
            .insert("cond".to_string(), binding(0, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("conditional_call!(cond, helper)").expect("parse macro expr");

        let _ = lowerer.lower_conditional_call(&expr);
    }

    #[test]
    fn conditional_call_elidable_residual_policy_keeps_live_marker() {
        let mut lowerer =
            lowerer_with_call_policy("helper", crate::jit_interp::CallPolicyKind::ResidualInt);
        lowerer
            .bindings
            .insert("value".to_string(), binding(0, BindingKind::Int));
        lowerer
            .bindings
            .insert("arg".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("conditional_call_elidable!(value, helper, arg)")
            .expect("parse macro expr");

        let result = lowerer
            .lower_conditional_call_elidable(&expr)
            .expect("conditional_call_elidable should lower");

        assert_eq!(result.kind, BindingKind::Int);
        let last = lowerer.op_metadata.last().expect("metadata emitted");
        assert!(matches!(last.kind, OpKind::LiveMarker));
        let tokens = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(tokens.contains("CanRaise"));
        assert!(tokens.contains("live_placeholder"));
    }

    #[test]
    fn promote_assign_aliases_lhs_to_promoted_arg_binding() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("y".to_string(), binding(7, BindingKind::Int));
        lowerer
            .bindings
            .insert("x".to_string(), binding(1, BindingKind::Int));
        let expr: Expr = syn::parse_str("x = promote(y)").expect("parse promote assignment");

        lowerer
            .lower_promote_stmt(&expr)
            .expect("promote assignment should lower");

        let x = lowerer.bindings.get("x").expect("x binding must exist");
        assert_eq!(
            x.reg, 7,
            "jtransform.py:613-615 returns None so result aliases arg0"
        );
        assert!(matches!(x.kind, BindingKind::Int));
        assert!(matches!(
            lowerer.op_metadata[0].control,
            ControlFlowClass::LiveMarker
        ));
        assert!(matches!(lowerer.op_metadata[1].kind, OpKind::GuardValue));
        assert_eq!(lowerer.op_metadata[1].reads, Register::ints(&[7]));
    }

    #[test]
    fn liveness_prebuild_emits_parent_markers_before_inline_helpers() {
        let helper_prebuild = quote! { helper_prebuild(__asm); };
        let tokens = liveness_prebuild_tokens(
            &[OpMeta::live_marker_with(Register::ints(&[3]), Vec::new())],
            &[helper_prebuild],
        )
        .to_string();
        let parent_pos = tokens
            .find("_register_liveness_offset")
            .expect("parent live marker should register a liveness offset");
        let helper_pos = tokens
            .find("helper_prebuild")
            .expect("nested helper prebuild should be present");
        assert!(
            parent_pos < helper_pos,
            "RPython assembles the caller before pending inline callees"
        );
    }

    #[test]
    fn inline_call_tokens_use_r_family_for_ref_only_args() {
        let tokens = inline_call_tokens_combined(&[binding(0, BindingKind::Ref)], 7);
        assert!(tokens.contains("inline_call_r_i"));
        assert!(tokens.contains("inline_call_r_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_ir_i"));
        assert!(!tokens.contains("inline_call_ir_r"));
        assert!(!tokens.contains("inline_call_irf_i"));
        assert!(!tokens.contains("inline_call_irf_r"));
    }

    #[test]
    fn inline_call_tokens_use_ir_family_when_any_int_arg_is_present() {
        let tokens = inline_call_tokens_combined(
            &[binding(0, BindingKind::Ref), binding(1, BindingKind::Int)],
            9,
        );
        assert!(tokens.contains("inline_call_ir_i"));
        assert!(tokens.contains("inline_call_ir_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_r_i"));
        assert!(!tokens.contains("inline_call_r_r"));
        assert!(!tokens.contains("inline_call_irf_i"));
        assert!(!tokens.contains("inline_call_irf_r"));
    }

    #[test]
    fn inline_call_tokens_use_irf_family_when_any_float_arg_is_present() {
        let tokens = inline_call_tokens_combined(
            &[binding(0, BindingKind::Int), binding(1, BindingKind::Float)],
            11,
        );
        assert!(tokens.contains("inline_call_irf_i"));
        assert!(tokens.contains("inline_call_irf_r"));
        assert!(tokens.contains("inline_call_irf_f"));
        assert!(!tokens.contains("inline_call_r_i"));
        assert!(!tokens.contains("inline_call_r_r"));
        assert!(!tokens.contains("inline_call_ir_i"));
        assert!(!tokens.contains("inline_call_ir_r"));
    }

    #[test]
    fn inline_call_tokens_emit_post_call_live_marker() {
        let (_, post_live) = inline_call_tokens(&[binding(0, BindingKind::Ref)], 7);
        let post = post_live.to_string();
        assert!(
            post.contains("live_placeholder"),
            "RPython jtransform.py emits inline_call followed by -live-"
        );
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_r_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(arg: usize) -> usize {
                    callee(arg)
                }
                "#,
            ),
            &[inline_policy("callee")],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_r_r"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_ir_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(lhs: usize, rhs: i64) -> i64 {
                    callee(lhs, rhs)
                }
                "#,
            ),
            &[inline_policy("callee")],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_ir_i"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    #[test]
    fn inline_helper_codegen_uses_canonical_irf_surface() {
        let helper = generate_inline_helper_jitcode_with_calls(
            &parse_fn(
                r#"
                fn outer(arg: f64) -> f64 {
                    callee(arg)
                }
                "#,
            ),
            &[inline_policy("callee")],
        )
        .expect("jit_inline lowering should succeed")
        .expect("helper should lower");
        let body = helper.body.to_string();
        assert!(body.contains("inline_call_irf_f"));
        assert!(!body.contains("inline_call_with_typed_args"));
    }

    #[test]
    fn inline_helper_param_layout_uses_dense_per_kind_banks() {
        let func = parse_fn(
            r#"
            fn helper(ptr: usize, value: i64, scale: f64, other: usize, more: i64) -> i64 {
                value + more
            }
            "#,
        );
        let layout = inline_helper_param_layout(&func).expect("layout should build");
        assert_eq!(
            layout,
            vec![
                (InlineReturnKind::Ref, 0),
                (InlineReturnKind::Int, 0),
                (InlineReturnKind::Float, 0),
                (InlineReturnKind::Ref, 1),
                (InlineReturnKind::Int, 1),
            ]
        );
    }

    #[test]
    fn inline_helper_param_counts_match_dense_layout() {
        let func = parse_fn(
            r#"
            fn helper(ptr: usize, value: i64, scale: f64, other: usize, more: i64) -> i64 {
                value + more
            }
            "#,
        );
        let counts = inline_helper_param_counts(&func).expect("counts should build");
        assert_eq!(counts, (2, 2, 1));
    }

    #[test]
    fn explicit_inline_ref_policy_sets_ref_binding_kind() {
        let call = parse_call("callee()");
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["callee".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlineRef),
            )],
            InferenceFailureMode::Panic,
        );
        let binding = lowerer
            .lower_call_value(&call)
            .expect("inline ref call should lower");
        assert!(matches!(binding.kind, BindingKind::Ref));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_sub_jitcode"));
        assert!(body.contains("__sub_return_kind"));
    }

    #[test]
    fn explicit_inline_float_policy_sets_float_binding_kind() {
        let call = parse_call("callee()");
        let mut lowerer = Lowerer::new_with_call_policies(
            None,
            vec![(
                vec!["callee".to_string()],
                CallPolicySpec::Explicit(crate::jit_interp::CallPolicyKind::InlineFloat),
            )],
            InferenceFailureMode::Panic,
        );
        let binding = lowerer
            .lower_call_value(&call)
            .expect("inline float call should lower");
        assert!(matches!(binding.kind, BindingKind::Float));
        let statements = &lowerer.statements;
        let body = quote! { #(#statements)* }.to_string();
        assert!(body.contains("add_sub_jitcode"));
        assert!(body.contains("__sub_return_kind"));
    }
}
