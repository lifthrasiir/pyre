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
enum VirtualizableHintKind {
    AccessDirectly,
    FreshVirtualizable,
    ForceVirtualizable,
}

fn classify_virtualizable_hint_segments<'a, I>(segments: I) -> Option<VirtualizableHintKind>
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
    io_shims: Vec<(Vec<String>, Ident)>,
    /// Canonical helper func path → explicit or inferred call policy.
    calls: Vec<(Vec<String>, CallPolicySpec)>,
    /// Whether top-level traced calls should auto-infer helper policy.
    auto_calls: bool,
    /// Virtualizable variable name (normalized, e.g., "frame").
    /// RPython jtransform.py: `is_virtualizable_getset()` uses this to check
    /// if a field access target is the virtualizable variable.
    vable_var: Option<String>,
    /// Ref-register assigned to the virtualizable input variable.
    ///
    /// RPython `MIFrame.setup_call(original_boxes)` distributes portal args
    /// by kind before opimpls consume `v_inst` / `v_base`.  The generated
    /// observer JitCode fragment receives the virtualizable as its first Ref
    /// input, so the line-by-line graph variable is `registers_r[0]`.
    vable_input_ref_reg: Option<u16>,
    /// Field name → (field_index, field_type).
    /// RPython: `vinfo.static_field_to_extra_box[fieldname]` → index.
    vable_fields: HashMap<String, (usize, ValueKind)>,
    /// Array name → (array_index, item_type).
    /// RPython: `vinfo.array_field_counter[fieldname]` → index.
    vable_arrays: HashMap<String, (usize, ValueKind)>,
    /// State field scalars: field_name → global_field_index.
    state_scalars: HashMap<String, usize>,
    /// State field arrays (flattened): field_name → global_array_index.
    state_arrays: HashMap<String, usize>,
    /// State field virtualizable arrays: field_name → virt_array_index.
    /// These emit GETARRAYITEM_RAW_I/SETARRAYITEM_RAW instead of element-level tracking.
    state_virt_arrays: HashMap<String, usize>,
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
    pub reds: Vec<Expr>,
    /// Canonical state-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `state`.
    /// Source: `JitInterpConfig.state_type` Ident.
    pub state_type_name: String,
    /// Canonical env-parameter type name. Used by `lower_method_call_value`
    /// to synthesize `<type>::<method>` path lookups for receiver `program`
    /// (the env parameter — convention fixed at the dispatch portal-input
    /// installer below). Source: `JitInterpConfig.env_type` Ident.
    pub env_type_name: String,
}

const MAX_HELPER_CALL_ARITY: usize = 16;

fn classify_virtualizable_hint_syn_path(path: &Path) -> Option<VirtualizableHintKind> {
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
enum CallPolicySpec {
    Explicit(crate::jit_interp::CallPolicyKind),
    Infer,
}

#[derive(Clone, Copy)]
enum InferenceFailureMode {
    ReturnNone,
    Panic,
}

#[derive(Clone, Copy)]
enum ValueKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CondCallEffectSlot {
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
    fn token(self) -> TokenStream {
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

    fn can_raise(self) -> bool {
        matches!(
            self,
            Self::CanRaise | Self::ElidableCanRaise | Self::ElidableOrMemerror
        )
    }

    fn is_elidable(self) -> bool {
        matches!(
            self,
            Self::ElidableCanRaise | Self::ElidableCannotRaise | Self::ElidableOrMemerror
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallResultKind {
    Void,
    Int,
    Ref,
    Float,
}

impl ValueKind {
    fn from_ident(ident: &Ident) -> Self {
        match ident.to_string().as_str() {
            "ref" => Self::Ref,
            "float" => Self::Float,
            _ => Self::Int,
        }
    }
}

fn call_policy_effect_slot(kind: crate::jit_interp::CallPolicyKind) -> Option<CondCallEffectSlot> {
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

fn call_policy_result_kind(kind: crate::jit_interp::CallPolicyKind) -> Option<CallResultKind> {
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

fn call_policy_is_wrapped(kind: crate::jit_interp::CallPolicyKind) -> bool {
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

fn call_result_matches_binding(result_kind: CallResultKind, binding_kind: BindingKind) -> bool {
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
fn inferred_policy_live_condition(func: &Expr, can_raise_codes: &[u8]) -> TokenStream {
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
        let (__policy, _, _, _, _) = #policy_path();
        matches!(__policy, #(#patterns)|*)
    }}
}

fn inferred_conditional_call_policy_check(func_args_empty: bool) -> TokenStream {
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

fn inferred_conditional_call_value_policy_check(
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

fn inferred_record_known_result_policy_check(result_kind: BindingKind) -> TokenStream {
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

fn canonical_path_segments(path: &Path) -> Vec<String> {
    path.segments
        .iter()
        .map(|segment| segment.ident.to_string())
        .collect()
}

fn canonical_member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

fn canonical_expr_segments(expr: &Expr) -> Option<Vec<String>> {
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

fn unwrap_ref_expr(expr: &Expr) -> &Expr {
    match expr {
        Expr::Reference(ExprReference { expr, .. }) => expr,
        _ => expr,
    }
}

fn expr_matches_local_name(expr: &Expr, expected: &str) -> bool {
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

fn named_member(member: &syn::Member) -> Option<String> {
    match member {
        syn::Member::Named(ident) => Some(ident.to_string()),
        _ => None,
    }
}

// ── Lowerer ──────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BindingKind {
    Int,
    Ref,
    Float,
}

#[derive(Clone)]
struct Binding {
    reg: u16,
    kind: BindingKind,
    depends_on_stack: bool,
}

/// Mirror of RPython `rpython/jit/codewriter/flatten.py:Register(kind, index)`.
/// Each emitted register carries its bank with it; the liveness walker
/// (`liveness.py:33-79`) keeps a single `set()` of `Register` objects per
/// marker, and `assembler.py:225-232 get_liveness_info(args, kind)` filters
/// by `reg.kind == kind` at encode time to split into the per-bank bitsets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Register {
    /// Total order is `(kind, index)` so `BTreeSet<Register>` iterates in
    /// kind-grouped order — convenient for encoders that emit per-bank
    /// bitsets.
    kind: BindingKind,
    index: u8,
}

impl Register {
    /// Construct a `Register` from a `(kind, u16-index)` pair, asserting that
    /// the index fits in the `assembler.py:225` bitset addressing range
    /// (0..=255). The lowerer's `Lowerer::next_reg` counter already obeys
    /// this bound; the assert traps regressions where a u16 reg leaked from
    /// outside that bound.
    #[allow(dead_code)]
    fn new(kind: BindingKind, index: u16) -> Self {
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
    fn int(index: u16) -> Self {
        Self::new(BindingKind::Int, index)
    }

    #[allow(dead_code)]
    fn ref_(index: u16) -> Self {
        Self::new(BindingKind::Ref, index)
    }

    #[allow(dead_code)]
    fn float(index: u16) -> Self {
        Self::new(BindingKind::Float, index)
    }

    /// Convenience: build a typed `Register` from a `Binding`.
    #[allow(dead_code)]
    fn from_binding(b: &Binding) -> Self {
        Self::new(b.kind, b.reg)
    }

    /// Build a `Vec<Register>` of `Int` from a slice of indices. Used by
    /// emit sites whose reads list is uniformly Int (binop, guard_value,
    /// etc.).
    #[allow(dead_code)]
    fn ints(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::int).collect()
    }

    #[allow(dead_code)]
    fn refs(indices: &[u16]) -> Vec<Register> {
        indices.iter().copied().map(Self::ref_).collect()
    }

    #[allow(dead_code)]
    fn floats(indices: &[u16]) -> Vec<Register> {
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
enum OpKind {
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
enum ControlFlowClass {
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
struct LiveMarkerCondition {
    /// Boolean expression evaluated both in the JitCode builder body and in
    /// the liveness prebuild body.
    emit: TokenStream,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct OpMeta {
    kind: OpKind,
    /// Source registers (uses). Each `Register` carries `kind` directly per
    /// `flatten.py:Register(kind, index)` so the liveness walker stays a
    /// single-bag set and the encoder (`assembler.py:225-232`) splits into
    /// per-bank bitsets on demand.
    reads: Vec<Register>,
    /// Destination registers (defs).
    writes: Vec<Register>,
    /// Branch target label, for control-flow ops.
    target_label: Option<Ident>,
    /// `-live-` marker TLabel operands. RPython stores every TLabel in
    /// the instruction tuple; a marker can carry more than one.
    live_target_labels: Vec<Ident>,
    /// Optional guard for the physical `BC_LIVE` emission.  Unconditional
    /// markers match normal RPython ssarepr.  Conditional markers are the
    /// strict-parity bridge for inferred helper policies whose can-raise
    /// answer is represented by a runtime policy byte.
    live_condition: Option<LiveMarkerCondition>,
    control: ControlFlowClass,
}

#[allow(dead_code)]
impl OpMeta {
    fn live_marker() -> Self {
        Self::live_marker_with(Vec::new(), Vec::new())
    }

    /// Conditional `-live-` for inferred-policy callees.  See
    /// [`LiveMarkerCondition`] for the PRE-EXISTING-ADAPTATION rationale
    /// and convergence path.
    fn live_marker_if(condition: TokenStream) -> Self {
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
    fn live_marker_with(reads: Vec<Register>, live_target_labels: Vec<Ident>) -> Self {
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
    fn linear(kind: OpKind, reads: Vec<Register>, writes: Vec<Register>) -> Self {
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
    fn jump(target: Ident) -> Self {
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
    fn conditional_guard(cond_reg: Register, target: Ident) -> Self {
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
    fn conditional_guard_int_eq(a_reg: Register, b_reg: Register, target: Ident) -> Self {
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
    fn label_def(target: Ident) -> Self {
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
    fn aux() -> Self {
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
    fn terminal(reads: Vec<Register>) -> Self {
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
struct LoweredSequence {
    statements: Vec<TokenStream>,
    op_metadata: Vec<OpMeta>,
}

impl LoweredSequence {
    fn new(statements: Vec<TokenStream>, op_metadata: Vec<OpMeta>) -> Self {
        debug_assert_eq!(
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

/// Per-marker live set produced by `compute_per_marker_liveness`.
/// Index aligns with the order in which `LiveMarker` ops appear in
/// `op_metadata`. Each entry is a single `BTreeSet<Register>` matching
/// RPython `liveness.py`'s `set()` of `Register` objects — bank info
/// rides on `Register.kind` and the encoder splits at emit time per
/// `assembler.py:225-232 get_liveness_info(args, kind)`.
#[allow(dead_code)]
type LiveMarkerLiveSets = Vec<BTreeSet<Register>>;

/// Compute the live register set captured at every `LiveMarker` op in
/// `op_metadata`, mirroring RPython
/// `rpython/jit/codewriter/liveness.py:33-79
/// _compute_liveness_must_continue`.
///
/// The walk is backward (def `discard`, use `add`); branch ops fold in
/// the destination label's accumulated alive set; label definitions
/// store the current alive set for forward jumps to consume on the
/// next iteration. Iterations continue until no label or marker entry
/// changes (fixed-point), matching RPython's `must_continue` loop.
///
/// Returned `Vec<BTreeSet<Register>>` is indexed in `LiveMarker`
/// encounter order, so callers can pair entries with their
/// `live_placeholder()` emit sites.
#[allow(dead_code)]
fn compute_per_marker_liveness(op_metadata: &[OpMeta]) -> LiveMarkerLiveSets {
    let marker_indices: Vec<usize> = op_metadata
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.control, ControlFlowClass::LiveMarker))
        .map(|(i, _)| i)
        .collect();

    let mut label_alive: HashMap<String, BTreeSet<Register>> = HashMap::new();
    let mut live_at_marker: HashMap<usize, BTreeSet<Register>> = HashMap::new();

    loop {
        let mut changed = false;
        let mut alive: BTreeSet<Register> = BTreeSet::new();

        for i in (0..op_metadata.len()).rev() {
            let op = &op_metadata[i];
            match op.control {
                ControlFlowClass::LiveMarker => {
                    // RPython liveness.py:44-53 — `-live-` first folds in
                    // any explicit force-alive register args and any
                    // TLabel target's accumulated alive set, then records
                    // the resulting alive at this marker. The mutation
                    // also propagates upstream so the registers / labels
                    // the marker keeps alive stay alive in earlier ops.
                    for target in &op.live_target_labels {
                        let name = target.to_string();
                        if let Some(s) = label_alive.get(&name) {
                            alive.extend(s.iter().copied());
                        }
                    }
                    alive.extend(op.reads.iter().copied());
                    let prev = live_at_marker.get(&i);
                    if prev.is_none() || prev.unwrap() != &alive {
                        live_at_marker.insert(i, alive.clone());
                        changed = true;
                    }
                }
                ControlFlowClass::LabelDef => {
                    // RPython liveness.py:36-42 — record alive against
                    // the label name (union with prior iterations).
                    let name = op
                        .target_label
                        .as_ref()
                        .expect("label_def needs target")
                        .to_string();
                    let entry = label_alive.entry(name).or_default();
                    let before = entry.len();
                    entry.extend(alive.iter().copied());
                    if entry.len() != before {
                        changed = true;
                    }
                }
                ControlFlowClass::UnconditionalJump => {
                    // RPython follow_label (liveness.py:29-31) — `alive`
                    // becomes the label's accumulated set (overwrite,
                    // not union, since fall-through past `jump` is
                    // unreachable).
                    let name = op
                        .target_label
                        .as_ref()
                        .expect("jump needs target")
                        .to_string();
                    alive = label_alive.get(&name).cloned().unwrap_or_default();
                }
                ControlFlowClass::ConditionalGuard => {
                    // Fold the branch target's alive set into the
                    // fall-through alive set, then add the cond_reg(s)
                    // as uses. RPython treats `goto_if_not` as a
                    // normal op whose TLabel arg triggers
                    // follow_label (alive update) and whose register
                    // args (cond) become uses.
                    if let Some(target) = op.target_label.as_ref() {
                        let name = target.to_string();
                        if let Some(s) = label_alive.get(&name) {
                            alive.extend(s.iter().copied());
                        }
                    }
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
                ControlFlowClass::Linear => {
                    // RPython liveness.py:60-69 — def first
                    // (`alive.discard(reg)`) then uses (`alive.add(x)`).
                    for w in &op.writes {
                        alive.remove(w);
                    }
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
                ControlFlowClass::Terminal => {
                    // `*_return` — no successor, no fall-through. Reset
                    // the alive set (nothing is alive past a return) and
                    // then add this op's own reads so the returned value
                    // stays alive upstream.
                    alive.clear();
                    for r in &op.reads {
                        alive.insert(*r);
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    marker_indices
        .iter()
        .map(|i| live_at_marker.remove(i).unwrap_or_default())
        .collect()
}

/// Encode-time bank split, mirroring RPython
/// `rpython/jit/codewriter/assembler.py:225-232 get_liveness_info(args,
/// kind)`. Walks a marker's accumulated alive set and projects out the
/// indices belonging to a single bank, producing the per-bank u8 vector
/// the BC_LIVE encoder consumes (`assembler.py:147-157` writes the
/// `(live_i, live_r, live_f)` triple as three sorted bitsets).
///
/// The walker (`compute_per_marker_liveness`) keeps a single
/// `BTreeSet<Register>` per marker so that the analysis stays
/// structurally identical to RPython's `set()` of `Register` objects;
/// the bank split is deferred to this helper at emit time.
///
/// `BTreeSet<Register>` already iterates in `(kind, index)` order due
/// to `Register`'s derived `Ord`, so the resulting `Vec<u8>` is sorted
/// — matching `assembler.py:148 live = sorted(live)`.
#[allow(dead_code)]
fn get_liveness_info(set: &BTreeSet<Register>, kind: BindingKind) -> Vec<u8> {
    set.iter()
        .filter(|r| r.kind == kind)
        .map(|r| r.index)
        .collect()
}

/// Convenience: return the `(live_i, live_r, live_f)` triple sourced
/// from `set`. Used by `maybe_dump_liveness` and by the BC_LIVE
/// per-marker patcher (`live_placeholder_with_triple` consumers added
/// in Phase 4 Epic B.3-B.4).
#[allow(dead_code)]
fn liveness_triple(set: &BTreeSet<Register>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    (
        get_liveness_info(set, BindingKind::Int),
        get_liveness_info(set, BindingKind::Ref),
        get_liveness_info(set, BindingKind::Float),
    )
}

/// Same as [`liveness_triple`] but consuming a typed register slice
/// (post-`annotate_live_markers_with_liveness` `LiveMarker.reads`).
/// Mirrors RPython `assembler.py:225-232 get_liveness_info(args, kind)`
/// applied to the marker's args directly, which by then are the full
/// alive set per `liveness.py:52`.
#[allow(dead_code)]
fn liveness_triple_from_reads(reads: &[Register]) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut live_i = Vec::new();
    let mut live_r = Vec::new();
    let mut live_f = Vec::new();
    for reg in reads {
        match reg.kind {
            BindingKind::Int => live_i.push(reg.index),
            BindingKind::Ref => live_r.push(reg.index),
            BindingKind::Float => live_f.push(reg.index),
        }
    }
    (live_i, live_r, live_f)
}

/// RPython `compute_liveness(ssarepr)` mutates each `-live-` instruction
/// (`liveness.py:52 ssarepr.insns[i] = insn[:1] + tuple(alive) + tuple(labels)`)
/// before `remove_repeated_live(ssarepr)` runs. Mirror that order by
/// materialising the fixed-point alive set back onto each `LiveMarker`'s
/// `reads` operand; the repeated-live pass and the emit-time triple
/// rewrite both consume the ssarepr-mutated shape directly.
fn annotate_live_markers_with_liveness(op_metadata: &mut [OpMeta]) {
    let live_sets = compute_per_marker_liveness(op_metadata);
    let mut next_marker = 0usize;
    for meta in op_metadata.iter_mut() {
        if !matches!(meta.control, ControlFlowClass::LiveMarker) {
            continue;
        }
        meta.reads = live_sets[next_marker].iter().copied().collect();
        next_marker += 1;
    }
    debug_assert_eq!(
        next_marker,
        live_sets.len(),
        "compute_per_marker_liveness output count must match LiveMarker op_metadata entries"
    );
}

/// Generate the per-marker liveness prebuild tokens that
/// `__prebuild_jitcode_liveness_*` (codegen_trace.rs) replays into the
/// driver-shared `Assembler` at install time. Each `LiveMarker` op
/// emits an `__asm._register_liveness_offset(&[live_i], &[live_r],
/// &[live_f])` call so RPython `pyjitpl.py:2255 finish_setup` order is
/// preserved: every per-marker triple lands in `asm.all_liveness`
/// before `metainterp_sd.liveness_info` snapshots it. Trace-time
/// `JitCodeBuilder::finalize_liveness` then only dedups against the
/// pre-registered offsets, never grows the table past the snapshot.
///
/// `inline_prebuild` carries any nested-helper prebuild tokens that
/// were aggregated during lowering.  Emit the current body's own
/// `-live-` triples first, then nested helper prebuilds: RPython
/// `codewriter.py:74-80` assembles the caller graph that discovered
/// an inline callee before draining the pending callee graph queued by
/// `call.py:155-172 get_jitcode`.
fn liveness_prebuild_tokens(
    op_metadata: &[OpMeta],
    inline_prebuild: &[TokenStream],
) -> TokenStream {
    let live_regs = op_metadata.iter().filter_map(|m| {
        if !matches!(m.control, ControlFlowClass::LiveMarker) {
            return None;
        }
        let (live_i, live_r, live_f) = liveness_triple_from_reads(&m.reads);
        let register = quote! {
            let _ = __asm._register_liveness_offset(
                &[#(#live_i),*],
                &[#(#live_r),*],
                &[#(#live_f),*],
            );
        };
        Some(if let Some(condition) = m.live_condition.as_ref() {
            let condition = condition.emit.clone();
            quote! {
                if #condition {
                    #register
                }
            }
        } else {
            register
        })
    });
    quote! {
        #(#live_regs)*
        #(#inline_prebuild)*
    }
}

/// Collapse runs of consecutive `LiveMarker` ops (and any intervening
/// `LabelDef` ops) into a single `LiveMarker`, mirroring RPython
/// `rpython/jit/codewriter/liveness.py:82-117 remove_repeated_live`.
///
/// The lowerer currently never emits markers in succession (each
/// `live_placeholder()` site sits in front of a guard / call op so a
/// non-marker non-label always intervenes), making this function a
/// structural no-op for present `#[jit_interp]` consumers. It still
/// runs end-to-end so future lowerers (or post-processing passes that
/// inject extra markers around inline-call boundaries) inherit the
/// RPython collapse semantics for free.
///
/// `op_metadata` and `statements` must stay index-aligned; both vectors
/// are mutated in lockstep.
#[allow(dead_code)]
fn remove_repeated_live(op_metadata: &mut Vec<OpMeta>, statements: &mut Vec<TokenStream>) {
    debug_assert_eq!(op_metadata.len(), statements.len());
    let mut new_meta: Vec<OpMeta> = Vec::with_capacity(op_metadata.len());
    let mut new_stmts: Vec<TokenStream> = Vec::with_capacity(statements.len());
    let mut i = 0;
    while i < op_metadata.len() {
        if !matches!(op_metadata[i].control, ControlFlowClass::LiveMarker) {
            new_meta.push(op_metadata[i].clone());
            new_stmts.push(statements[i].clone());
            i += 1;
            continue;
        }
        // Collect the run of consecutive markers (separated by label
        // definitions only).
        let first_marker_idx = i;
        let mut markers: Vec<usize> = vec![i];
        let mut interleaved_labels: Vec<usize> = Vec::new();
        i += 1;
        while i < op_metadata.len() {
            match op_metadata[i].control {
                ControlFlowClass::LiveMarker => {
                    markers.push(i);
                    i += 1;
                }
                ControlFlowClass::LabelDef => {
                    interleaved_labels.push(i);
                    i += 1;
                }
                _ => break,
            }
        }
        if markers.len() == 1 {
            for li in &interleaved_labels {
                new_meta.push(op_metadata[*li].clone());
                new_stmts.push(statements[*li].clone());
            }
            new_meta.push(op_metadata[first_marker_idx].clone());
            new_stmts.push(statements[first_marker_idx].clone());
            continue;
        }
        // PRE-EXISTING-ADAPTATION: `liveness.py:82-116 remove_repeated_live`
        // unions the `reads` of every marker in the run because every
        // upstream marker actually fires (RPython has no conditional
        // emission).  pyre's `live_marker_if` markers exist or not at
        // runtime depending on the helper-policy byte
        // (`__majit_call_policy_<name>()`), so unioning their reads here
        // would over-capture: when only one condition holds at runtime,
        // the merged BC_LIVE would still pin the union of the
        // would-have-fired siblings' alive sets.  When the run contains
        // any unconditional marker the merged BC_LIVE is guaranteed to
        // fire (PyPy parity), so unioning is safe and the merged marker
        // becomes unconditional.  When every marker is conditional, fall
        // back to keeping them unmerged — each emits its own BC_LIVE
        // only when its own condition holds, matching PyPy's per-site
        // alive-set capture (at the cost of skipping `liveness.py:82`'s
        // dedup, which `production` doesn't trigger anyway because the
        // lowerer emits at most one marker per call/guard site).
        if markers
            .iter()
            .all(|mi| op_metadata[*mi].live_condition.is_some())
        {
            for idx in first_marker_idx..i {
                new_meta.push(op_metadata[idx].clone());
                new_stmts.push(statements[idx].clone());
            }
            continue;
        }
        // Multiple markers with at least one unconditional: union their
        // `reads` registers per RPython `liveness.py:111-115
        // liveset.update(live[1:])`.  Union typed Register reads as a
        // single bag (Ord = (kind, index)) and union
        // `live_target_labels` separately.  Result is unconditional —
        // the unconditional sibling forces emit, so the conditional
        // siblings' reads fold in at this fully-fired position.
        let mut merged_reads: Vec<Register> = Vec::new();
        let mut merged_labels: Vec<Ident> = Vec::new();
        for mi in &markers {
            let m = &op_metadata[*mi];
            merged_reads.extend(m.reads.iter().copied());
            merged_labels.extend(m.live_target_labels.iter().cloned());
        }
        merged_reads.sort();
        merged_reads.dedup();
        merged_labels.sort_by_key(|label| label.to_string());
        merged_labels.dedup_by_key(|label| label.to_string());
        let merged_marker = OpMeta::live_marker_with(merged_reads, merged_labels);
        for li in &interleaved_labels {
            new_meta.push(op_metadata[*li].clone());
            new_stmts.push(statements[*li].clone());
        }
        new_meta.push(merged_marker);
        // Reuse the first marker's statement token (a single
        // `live_placeholder()` call); the duplicated runs don't survive
        // the collapse since RPython prints just one `-live-` for the
        // whole run.  `rewrite_live_marker_statements_with_triples`
        // (later pass) overwrites the body — the merged marker's
        // `live_condition` is `None` so the rewrite emits an
        // unconditional `live_placeholder_with_triple(...)`.
        new_stmts.push(statements[first_marker_idx].clone());
    }
    *op_metadata = new_meta;
    *statements = new_stmts;
}

/// Phase 4 / Epic B.3-B.4 emit-time bridge: replace each `LiveMarker`
/// statement's `live_placeholder()` call with the triple-aware
/// `live_placeholder_with_triple(&[live_i...], &[live_r...], &[live_f...])`
/// shape, sourcing the per-marker triples from
/// [`compute_per_marker_liveness`] split per bank by [`liveness_triple`]
/// (mirrors `assembler.py:225-232 get_liveness_info(args, kind)`).
///
/// Runs after [`remove_repeated_live`] so the marker count seen by the
/// walker matches the number of statements that actually survive into
/// the lowered output.
///
/// The runtime effect is no-op until the factory closure calls
/// `JitCodeBuilder::finalize_liveness(&mut asm)` — until then,
/// `pending_live_triples` accumulates per-builder records but the
/// emitted `live/<00 00>` slot stays at offset 0, identical to the
/// `live_placeholder()` shape it replaces.  `finalize_liveness` is wired
/// in a follow-on slice (driver-shared `Arc<Mutex<Assembler>>` plumbing
/// through `register_jitcode_factory`).
///
/// Each register index must fit in `u8` per RPython
/// `rpython/jit/codewriter/assembler.py:225` — the bitset encoder
/// only addresses 0..=255 (8 register-bytes × 8 bits). The typed
/// `Register::new` constructor asserts this bound at every
/// emit site, so by the time the walker hands us a `BTreeSet<Register>`
/// the indices are guaranteed `u8`-clean.
fn rewrite_live_marker_statements_with_triples(
    op_metadata: &[OpMeta],
    statements: &mut [TokenStream],
) {
    debug_assert_eq!(op_metadata.len(), statements.len());
    let live_sets = compute_per_marker_liveness(op_metadata);
    let mut next_marker = 0usize;
    for (i, m) in op_metadata.iter().enumerate() {
        if !matches!(m.control, ControlFlowClass::LiveMarker) {
            continue;
        }
        let (live_i, live_r, live_f) = liveness_triple(&live_sets[next_marker]);
        next_marker += 1;
        let live_stmt = quote! {
            let _ = __builder.live_placeholder_with_triple(
                &[#(#live_i),*],
                &[#(#live_r),*],
                &[#(#live_f),*],
            );
        };
        statements[i] = if let Some(condition) = m.live_condition.as_ref() {
            let condition = condition.emit.clone();
            quote! {
                if #condition {
                    #live_stmt
                }
            }
        } else {
            live_stmt
        };
    }
    debug_assert_eq!(
        next_marker,
        live_sets.len(),
        "compute_per_marker_liveness output count must match LiveMarker op_metadata entries"
    );
}

/// Print per-marker live sets to stderr when `MAJIT_DUMP_LIVENESS` is
/// set in the proc-macro build environment. `label` is the lowerer
/// scope being dumped (e.g. helper name) so concurrent expansions are
/// distinguishable.
fn maybe_dump_liveness(label: &str, op_metadata: &[OpMeta]) {
    if std::env::var("MAJIT_DUMP_LIVENESS").is_err() {
        return;
    }
    let live_sets = compute_per_marker_liveness(op_metadata);
    let marker_count = op_metadata
        .iter()
        .filter(|m| matches!(m.control, ControlFlowClass::LiveMarker))
        .count();
    eprintln!(
        "=== majit liveness dump [{}] op_metadata={} markers={} ===",
        label,
        op_metadata.len(),
        marker_count
    );
    for (idx, set) in live_sets.iter().enumerate() {
        let (live_i, live_r, live_f) = liveness_triple(set);
        eprintln!(
            "  marker[{}] live_i={:?} live_r={:?} live_f={:?}",
            idx, live_i, live_r, live_f,
        );
    }
}

struct Lowerer<'c> {
    bindings: HashMap<String, Binding>,
    statements: Vec<TokenStream>,
    /// Per-op metadata, parallel to `statements`. Populated as B.2.A.ii
    /// migrates each emit site through `emit_op`. Read by the backward
    /// walker (B.2.B). Currently sparse — only `LiveMarker` sites land.
    #[allow(dead_code)]
    op_metadata: Vec<OpMeta>,
    next_reg: u16,
    next_label: u16,
    config: Option<&'c LowererConfig>,
    call_policies: Vec<(Vec<String>, CallPolicySpec)>,
    inference_failure_mode: InferenceFailureMode,
    auto_calls: bool,
    /// Prebuild tokens carried up from nested inline-helper lowerings.
    /// These get merged into the parent body's
    /// `liveness_prebuild_tokens` output so the helper's per-marker
    /// triples land in `__prebuild_jitcode_liveness_*` alongside the
    /// outer arm's triples.
    #[allow(dead_code)]
    inline_liveness_prebuild: Vec<TokenStream>,
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
    dispatch_tainted_reason: Option<&'static str>,
    /// Name of the LHS variable that received the opcode-fetch
    /// result, set by `try_lower_opcode_fetch_stmt` when it recognises
    /// `let <name> = program[<idx>]` (or the method-call form
    /// `program.get_op(<idx>)`).  `lower_dispatch_chain` uses this name
    /// to look up the opcode reg in `bindings` so the dispatch chain
    /// emits regardless of the consumer's chosen variable name.
    /// Falls back to the literal `"opcode"` (PyPy `pyopcode.py:171`
    /// canonical name) when unset, preserving existing fixtures.
    opcode_var_name: Option<String>,
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
    in_dispatch_arm_body: bool,
}

impl<'c> Lowerer<'c> {
    fn new(config: Option<&'c LowererConfig>) -> Self {
        let call_policies = config.map(|cfg| cfg.calls.clone()).unwrap_or_default();
        Self::new_with_call_policies(config, call_policies, InferenceFailureMode::ReturnNone)
    }

    fn new_with_call_policies(
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

    fn install_vable_input_binding(&mut self) {
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

    fn vable_base_reg(&self) -> Option<u16> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let binding = self.bindings.get(vable_var)?;
        match binding.kind {
            BindingKind::Ref => Some(binding.reg),
            _ => None,
        }
    }

    fn alloc_reg(&mut self) -> u16 {
        let reg = self.next_reg;
        self.next_reg = self.next_reg.saturating_add(1);
        reg
    }

    fn alloc_label(&mut self) -> syn::Ident {
        let label = self.next_label;
        self.next_label = self.next_label.saturating_add(1);
        format_ident!("__jit_label_{label}")
    }

    /// Emit an op token plus its parallel `OpMeta` entry, keeping
    /// `statements` and `op_metadata` index-aligned for the backward
    /// liveness walker (Slice B.2.B).
    fn emit_op(&mut self, meta: OpMeta, tokens: TokenStream) {
        self.statements.push(tokens);
        self.op_metadata.push(meta);
    }

    fn append_lowered_sequence(&mut self, lowered: LoweredSequence) {
        debug_assert_eq!(
            lowered.statements.len(),
            lowered.op_metadata.len(),
            "RPython ssarepr.insns parity requires branch statements and metadata to append together"
        );
        self.statements.extend(lowered.statements);
        self.op_metadata.extend(lowered.op_metadata);
    }

    fn emit_label_def(&mut self, label: &Ident) {
        self.emit_op(
            OpMeta::label_def(label.clone()),
            quote! { __builder.mark_label(#label); },
        );
    }

    fn emit_jump(&mut self, target: &Ident) {
        self.emit_op(
            OpMeta::jump(target.clone()),
            quote! { __builder.jump(#target); },
        );
    }

    fn emit_conditional_guard(&mut self, cond_reg: u16, target: &Ident) {
        // `goto_if_not_int_is_true` reads an int-banked register per
        // `assembler.py:217 'i'` argcode — encode the kind into the
        // metadata `Register` so the liveness walker keeps it under Int.
        self.emit_op(
            OpMeta::conditional_guard(Register::int(cond_reg), target.clone()),
            quote! { __builder.goto_if_not_int_is_true(#cond_reg, #target); },
        );
    }

    /// Emit a builder-side aux statement (no BC_* op, no def/use).
    fn emit_aux(&mut self, tokens: TokenStream) {
        self.emit_op(OpMeta::aux(), tokens);
    }

    fn inference_failure_tokens(&self, message: &str) -> TokenStream {
        match self.inference_failure_mode {
            InferenceFailureMode::ReturnNone => quote! { return None; },
            InferenceFailureMode::Panic => {
                let message = message.to_string();
                quote! { panic!(#message); }
            }
        }
    }

    fn resolve_call_policy(&self, func: &Expr) -> Option<CallPolicySpec> {
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
    fn cond_call_policy_or_inferred_default(
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

    fn cond_call_slot_for_policy(
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

    fn call_target_registration_tokens(
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
                let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                let __fn_idx = __builder.add_call_target_with_slot(
                    __trace_target,
                    __concrete_target,
                    #slot_expr,
                );
            }
        } else {
            quote! {
                let __fn_idx = __builder.add_fn_ptr_with_slot(#func as *const (), #slot_expr);
            }
        }
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Option<()> {
        match stmt {
            Stmt::Local(local) => {
                if let Some(()) = self.lower_local(local) {
                    return Some(());
                }
                if self.config.is_some() && !self.stmt_modifies_jit_state(stmt) {
                    return Some(());
                }
                None
            }
            Stmt::Expr(expr, _) => {
                if let Some(()) = self.lower_expr_stmt(expr) {
                    return Some(());
                }
                if self.config.is_some() && !self.stmt_modifies_jit_state(stmt) {
                    return Some(());
                }
                None
            }
            Stmt::Macro(stmt_macro) => {
                // jtransform.py:1714-1723 handle_jit_marker__loop_header —
                // a `can_enter_jit!()` call at the user's source-level
                // back-edge (interp_jit.py:118 inside `jump_absolute`'s
                // backward-jump branch) lowers to `loop_header(jd.index)`
                // at the SAME source position.  Per-arm emission at the
                // dispatch JitCode level (post-INLINE_CALL) would over-
                // emit on every arm execution including forward-jump
                // path; emitting at the call site inside the arm body
                // sub-JitCode makes the LH op execute only when control
                // reaches the conditional that contains can_enter_jit!.
                //
                // Only fire when this Lowerer is producing the dispatch
                // arm body sub-JitCode (where the surrounding
                // `__dispatch_jitcode_<fn>` provides `__jdindex` in
                // scope).  For the per-arm trace JitCode path (whose
                // surrounding fn has no `__jdindex`) the recognition
                // falls through to `None` and the body lowering aborts
                // — pyre's per-arm trace JitCode is a
                // PRE-EXISTING-ADAPTATION not present in RPython, so
                // omitting `loop_header` there is consistent with
                // upstream's single-JitCode model.
                if !self.in_dispatch_arm_body {
                    return None;
                }
                let path_str = stmt_macro
                    .mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "can_enter_jit" || path_str.ends_with("::can_enter_jit") {
                    self.emit_op(
                        OpMeta::linear(OpKind::LoopHeader, vec![], vec![]),
                        quote! {
                            // jtransform.py:1716 c_index = Constant(jd.index, ...);
                            // __jdindex is the runtime parameter of the
                            // enclosing `__dispatch_jitcode_<fn>(__asm,
                            // __jdindex: i64)` and remains in scope of
                            // the arm body sub-builder block.
                            __builder.loop_header(__jdindex);
                        },
                    );
                    return Some(());
                }
                None
            }
            Stmt::Item(_) => None,
        }
    }

    fn lower_local(&mut self, local: &Local) -> Option<()> {
        let Pat::Ident(pat_ident) = &local.pat else {
            return None;
        };
        let init = local.init.as_ref()?;

        // Try normal lowering
        if let Some(binding) = self.lower_value_expr(&init.expr) {
            // When a stack pop is lowered to a JitCode register, also emit a
            // Rust `let` binding so that subsequent un-lowered code (e.g.,
            // complex expressions referencing the variable) can still compile.
            // The value is 0 — only the JitCode register carries the real
            // runtime value, but this prevents "cannot find value" errors.
            if binding.depends_on_stack {
                let ident = &pat_ident.ident;
                self.emit_aux(quote! { let #ident: i64 = 0; });
            }
            self.bindings.insert(pat_ident.ident.to_string(), binding);
            return Some(());
        }

        // Config-aware: runtime constant (expression not touching storage).
        //
        // Slice ε.3 fail-closed: ALSO refuse this fallback when the init
        // expression references any name already bound in `self.bindings`.
        // The fallback emits the original `let X = <init_expr>;` line as
        // verbatim Rust into the surrounding `__builder` block scope, then
        // a `__builder.load_const_i_value(reg, X as i64)`.  That contract
        // assumes `init_expr` is a true compile-time constant whose
        // identifiers (if any) are Rust types / `const` items / module
        // paths — NOT JIT-level bindings (`program` Ref / `pc` Int /
        // arm-pattern bound names) which are not in scope at the
        // surrounding Rust scope.  Without this guard, dispatch arm
        // sub-JitCode bodies that contain unrecognised method calls on
        // a parent binding (e.g. aheui-jit's `program.get_operand(pc - 1)`
        // when no `Program::get_operand` call policy is registered) would
        // emit verbatim Rust referencing `program`/`pc` in the
        // `__sub_builder` block — failing to compile.  Returning `None`
        // here triggers the dispatch arm's `None` branch which substitutes
        // an `abort_permanent()` sub-JitCode (see `lower_dispatch_chain`).
        if self.config.is_some()
            && !self.expr_touches_storage(&init.expr)
            && !self.expr_references_any_binding(&init.expr)
        {
            let reg = self.alloc_reg();
            let ident = &pat_ident.ident;
            let init_expr = &init.expr;
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                quote! {
                    let #ident = #init_expr;
                    __builder.load_const_i_value(#reg, #ident as i64);
                },
            );
            self.bindings.insert(
                ident.to_string(),
                Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                },
            );
            return Some(());
        }

        None
    }

    /// Walk `expr` and return `true` if any single-segment `Expr::Path`
    /// references a name bound in `self.bindings`, **excluding** names
    /// shadowed by an inner `let` in the same expression.  Mirrors the
    /// recognition core of `collect_arm_caller_locals` but stops at the
    /// first match — used as a fail-closed gate inside `lower_local`'s
    /// runtime-constant fallback.
    ///
    /// Scope tracking: PyPy `flowspace` produces distinct flowgraph
    /// variables per lexical scope.  Pyre's probe approximates this by
    /// pushing a fresh scope frame on entering an `ExprBlock` and
    /// popping on exit; `let X = ...` inside the block adds X to the
    /// innermost frame.  An ident is "locally bound" if any frame in
    /// the stack contains it, so the inner `let pc = 42; pc + 1` shape
    /// correctly suppresses the outer `pc` parent-binding match.
    fn expr_references_any_binding(&self, expr: &Expr) -> bool {
        use syn::visit::Visit;
        struct BindingProbe<'a> {
            bindings: &'a HashMap<String, Binding>,
            hit: bool,
            /// Stack of per-block local-binding sets (innermost on top).
            scope_stack: Vec<HashSet<String>>,
        }
        impl BindingProbe<'_> {
            fn is_locally_bound(&self, name: &str) -> bool {
                self.scope_stack.iter().any(|s| s.contains(name))
            }
        }
        impl<'ast> Visit<'ast> for BindingProbe<'_> {
            fn visit_expr_path(&mut self, p: &'ast ExprPath) {
                if self.hit || p.qself.is_some() || p.path.segments.len() != 1 {
                    return;
                }
                let seg = &p.path.segments[0];
                if !seg.arguments.is_none() {
                    return;
                }
                let name = seg.ident.to_string();
                if self.is_locally_bound(&name) {
                    return;
                }
                if self.bindings.contains_key(&name) {
                    self.hit = true;
                }
            }
            fn visit_expr_field(&mut self, ef: &'ast syn::ExprField) {
                self.visit_expr(&ef.base);
            }
            fn visit_expr_method_call(&mut self, mc: &'ast ExprMethodCall) {
                self.visit_expr(&mc.receiver);
                for arg in &mc.args {
                    self.visit_expr(arg);
                }
            }
            fn visit_block(&mut self, b: &'ast Block) {
                // Cover every `Block` traversal — explicit `{ ... }`
                // (`ExprBlock`'s default impl forwards here), if/else
                // branches (`ExprIf::then_branch` / `else_branch`),
                // while / loop / for bodies — not just the explicit
                // block expression form.  Each lexical block pushes a
                // fresh scope frame so inner `let X = ...` shadows the
                // parent binding inside that block only.
                self.scope_stack.push(HashSet::new());
                for stmt in &b.stmts {
                    self.visit_stmt(stmt);
                    if self.hit {
                        break;
                    }
                }
                self.scope_stack.pop();
            }
            fn visit_expr_match(&mut self, em: &'ast ExprMatch) {
                self.visit_expr(&em.expr);
                for arm in &em.arms {
                    if self.hit {
                        break;
                    }
                    // Each match arm introduces a scope: pattern-bound
                    // names shadow outer bindings inside the arm body.
                    // Mirrors `flowspace`'s SpaceOperation scope per
                    // match arm.
                    let mut arm_scope = HashSet::new();
                    collect_pat_bound_idents(&arm.pat, &mut arm_scope);
                    self.scope_stack.push(arm_scope);
                    if let Some((_, guard)) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                    self.scope_stack.pop();
                }
            }
            fn visit_local(&mut self, local: &'ast Local) {
                // Visit init RHS BEFORE adding the bound name so the
                // init expression's references are still probed against
                // outer scope (`let X = X + 1` at scope entry uses the
                // outer X for the RHS).
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    if let Some((_, diverge)) = &init.diverge {
                        self.visit_expr(diverge);
                    }
                }
                // All bindings produced by the pattern enter the
                // innermost scope frame.  `let (a, b) = ...;`,
                // `let Foo { x } = ...;`, `let A(y) | B(y) = ...;`
                // — each pattern shape contributes its bound names.
                // Mirrors `flowspace`'s SpaceOperation per
                // pattern-extraction step.
                if let Some(top) = self.scope_stack.last_mut() {
                    collect_pat_bound_idents(&local.pat, top);
                }
            }
        }
        let mut probe = BindingProbe {
            bindings: &self.bindings,
            hit: false,
            // Seed with one root frame so `visit_local` inside the
            // top-level expression (no enclosing block) can still
            // record bindings.
            scope_stack: vec![HashSet::new()],
        };
        probe.visit_expr(expr);
        probe.hit
    }

    /// RPython jtransform.py:923 `_rewrite_op_setfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name = value` and emits vable_setfield JitCode.
    fn lower_vable_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(field_index, field_type) = config.vable_fields.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let fi = field_index as u16;
        let binding = self.lower_value_expr(&assign.right)?;
        let src = binding.reg;
        // vable_reg is always Ref (the virtualizable input register); src bank
        // follows `field_type` per `assembler.py:217` argcode mapping.
        let vable_r = Register::ref_(vable_reg);
        match field_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::ref_(src)], vec![]),
                quote! { __builder.vable_setfield_ref_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::float(src)], vec![]),
                quote! { __builder.vable_setfield_float_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::int(src)], vec![]),
                quote! { __builder.vable_setfield_int_with_base(#vable_reg, #fi, #src); },
            ),
        }
        Some(())
    }

    /// RPython jtransform.py:794 `setarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i] = val` and emits vable_setarrayitem.
    fn lower_vable_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        // LHS: frame.array_field[index]
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let ai = array_index as u16;

        // Lower index and value
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        // vable_reg: Ref. idx_reg: Int (array index). val_reg: bank by item_type.
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        match item_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::ref_(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_ref_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::float(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_float_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::int(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_int_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
        }
        Some(())
    }

    /// Recognizes `state.field = expr` for scalar state fields.
    fn lower_state_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &field_index = config.state_scalars.get(&member_name)?;
        let fi = field_index as u16;
        let binding = self.lower_value_expr(&assign.right)?;
        let src = binding.reg;
        // store_state_field/di — `src` is Int per assembler.py:217 'i' argcode.
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![Register::int(src)], vec![]),
            quote! { __builder.store_state_field(#fi, #src); },
        );
        Some(())
    }

    /// Recognizes `state.field += expr` for scalar state fields.
    fn lower_state_field_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let field = match &*binary.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &field_index = config.state_scalars.get(&member_name)?;
        let opcode = opcode_for_assign_binop(&binary.op)?;

        let lhs = self.lower_state_field_read(&binary.left)?;
        let rhs = self.lower_value_expr(&binary.right)?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let dst = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(dst)],
            ),
            quote! { __builder.record_binop_i(#dst, majit_ir::OpCode::#opcode, #lhs_reg, #rhs_reg); },
        );
        let fi = field_index as u16;
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![Register::int(dst)], vec![]),
            quote! { __builder.store_state_field(#fi, #dst); },
        );
        Some(())
    }

    /// Recognizes `state.array[index] = expr` for array state fields.
    /// Routes to `store_state_varray` for virtualizable arrays, `store_state_array` for flattened.
    fn lower_state_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        // store_state_{varray,array}/dii — both reg args are Int per
        // assembler.py:217 'i' argcode.
        let idx_r = Register::int(idx_reg);
        let val_r = Register::int(val_reg);
        if let Some(&va_idx) = config.state_virt_arrays.get(&member_name) {
            let ai = va_idx as u16;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![idx_r, val_r], vec![]),
                quote! { __builder.store_state_varray(#ai, #idx_reg, #val_reg); },
            );
            return Some(());
        }
        let &array_index = config.state_arrays.get(&member_name)?;
        let ai = array_index as u16;
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![idx_r, val_r], vec![]),
            quote! { __builder.store_state_array(#ai, #idx_reg, #val_reg); },
        );
        Some(())
    }

    /// RPython jtransform.py:650 `hint_force_virtualizable`.
    ///
    /// Recognizes `hint_force_virtualizable!(frame)` macro invocation.
    fn lower_vable_force(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let _vable_var = config.vable_var.as_ref()?;

        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let hint = classify_virtualizable_hint_syn_path(&mac.mac.path)?;
        if hint != VirtualizableHintKind::ForceVirtualizable {
            return None;
        }
        let arg: Expr = syn::parse2(mac.mac.tokens.clone()).ok()?;
        let binding = self.lower_value_expr(&arg)?;
        let vable_reg = binding.reg;
        // vable_force/r — vable_reg is Ref per assembler.py:217 'r' argcode.
        self.emit_op(
            OpMeta::linear(OpKind::Vable, vec![Register::ref_(vable_reg)], vec![]),
            quote! { __builder.vable_force_with_base(#vable_reg); },
        );
        Some(())
    }

    /// RPython jtransform.py:655 — suppress identity hint function calls.
    ///
    /// `hint_access_directly(frame)` and `hint_fresh_virtualizable(frame)`
    /// are identity functions that return their argument unchanged.
    /// The Lowerer recognizes these calls and lowers the argument directly,
    /// effectively eliminating the hint call.
    fn lower_vable_hint_identity_call(&mut self, expr: &Expr) -> Option<Binding> {
        let call = match expr {
            Expr::Call(c) => c,
            _ => return None,
        };
        let func_name = match &*call.func {
            Expr::Path(p) => classify_virtualizable_hint_syn_path(&p.path),
            _ => return None,
        };
        match func_name {
            Some(
                VirtualizableHintKind::AccessDirectly | VirtualizableHintKind::FreshVirtualizable,
            ) => {
                let arg = call.args.first()?;
                self.lower_value_expr(arg)
            }
            _ => None,
        }
    }

    /// RPython jtransform.py:655 `hint(access_directly=True)` /
    /// `hint(fresh_virtualizable=True)`.
    ///
    /// These hints are consumed by the translator — jtransform suppresses
    /// them (returns None = no opcode generated). The codewriter has already
    /// rewritten field accesses to use vable_getfield/setfield, so the
    /// access_directly hint is redundant at this point.
    ///
    /// In majit, the Lowerer recognizes these macro calls and emits nothing,
    /// which matches RPython's behavior exactly.
    fn lower_vable_hint_suppress(&self, expr: &Expr) -> Option<()> {
        let _config = self.config?;
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        match classify_virtualizable_hint_syn_path(&mac.mac.path) {
            Some(
                VirtualizableHintKind::AccessDirectly | VirtualizableHintKind::FreshVirtualizable,
            ) => Some(()),
            _ => None,
        }
    }

    // ── conditional_call / record_known_result JIT op emission ──────

    /// RPython jtransform.py:1685 — `rewrite_op_jit_conditional_call`.
    ///
    /// Recognizes `conditional_call!(condition, func, args...)` and emits
    /// `__builder.conditional_call_ir_v_typed_args`, matching
    /// `jtransform.py`'s canonical opname.
    fn lower_conditional_call(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = condition, args[1] = func path, args[2..] = function arguments
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("conditional_call does not support more than 4 arguments");
        }
        let cond_binding = self.lower_value_expr(args[0])?;
        let cond_reg = cond_binding.reg;
        // RPython make_three_lists: tag each arg with its kind (int/ref).
        let mut typed_arg_tokens = Vec::new();
        // cond_reg is Int per the conditional_call argcode prefix.
        let mut arg_regs: Vec<Register> = vec![Register::int(cond_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                // jtransform.py:1668: float → raise Exception
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // `conditional_call!` always lowers to a void residual_call.
        // Default to `ResidualVoidWrapped` for `Infer` so the
        // analyzer-absent CanRaise slot is the lowering's static slot;
        // the runtime helper-policy lookup overrides this for callees
        // whose flavor turns out otherwise.
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call!",
            crate::jit_interp::CallPolicyKind::ResidualVoidWrapped,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("conditional_call! helper policy {policy:?} has no direct-call result kind");
        };
        if result_kind != CallResultKind::Void {
            panic!("conditional_call! requires a void-return helper policy, got {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call!");
        // `call.py:249-251 getcalldescr`:
        //   if loopinvariant:
        //       assert not NON_VOID_ARGS, ("arguments not supported for "
        //                                  "loop-invariant function!")
        // The canonical `call_loopinvariant_*_canonical_via_target`
        // builders enforce the same invariant via `arg_regs.is_empty()`
        // (`jitcode/assembler.rs:1849`), but the cond_call helper
        // dispatch routes through `conditional_call_ir_v_typed_args`
        // which doesn't share that assert. Mirror the check here so
        // a `conditional_call!(cond, loop_invariant_helper, arg)`
        // panics at expansion time instead of silently registering a
        // bytecode shape RPython would reject at calldescr build.  In
        // `Infer` mode the slot is decided at runtime from `__policy`,
        // so the static check only fires when the macro-time default
        // resolves to LoopInvariant — explicit policy paths preserve
        // the original eager assert.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_policy_check(func_args.is_empty()))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        self.emit_op(
            OpMeta::linear(OpKind::Call, arg_regs, vec![]),
            quote! {
                #register_target
                __builder.conditional_call_ir_v_typed_args(__fn_idx, #cond_reg, &[#(#typed_arg_tokens),*]);
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)` for the selected calldescr.
        // In inferred mode the physical BC_LIVE is guarded by the same
        // helper-policy byte that selects the calldescr slot, preserving
        // PyPy's cannot-raise / loop-invariant no-marker shape.
        if is_inferred {
            let condition = inferred_policy_live_condition(func_path, &[1]);
            self.emit_op(
                OpMeta::live_marker_if(condition),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    /// RPython jtransform.py:1687 — `rewrite_op_jit_conditional_call_value`.
    ///
    /// Recognizes `conditional_call_elidable!(value, func, args...)` and emits
    /// the canonical `conditional_call_value_ir_{i,r}` builder entrypoint.
    fn lower_conditional_call_elidable(&mut self, expr: &Expr) -> Option<Binding> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call_elidable" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        let value_binding = self.lower_value_expr(args[0])?;
        let value_reg = value_binding.reg;
        // jtransform.py:1668: value itself must not be float
        if matches!(value_binding.kind, BindingKind::Float) {
            panic!("Conditional call does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        // value_reg is Int or Ref per the conditional_call_value_ir_{i|r} arm.
        let value_kind = value_binding.kind;
        let mut arg_regs: Vec<Register> = vec![Register::new(value_kind, value_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        let result_reg = self.alloc_reg();
        // RPython jtransform.py:1687 — conditional_call_value_ir_{i|r}
        let builder_call = match value_kind {
            BindingKind::Ref => quote! {
                __builder.conditional_call_value_ir_r_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
            _ => quote! {
                __builder.conditional_call_value_ir_i_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
        };
        // `conditional_call_elidable!` is the elidable cache helper; per
        // `rlib/jit.py:1334-1336` the callee need not be `@elidable` but
        // the cond_call_value op itself caches the result.  Default to
        // `Elidable*Wrapped` based on the leading value-kind so an
        // inferred policy still classifies as elidable.
        let inferred_default = match value_kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call_elidable!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!(
                "conditional_call_elidable! helper policy {policy:?} has no direct-call result kind"
            );
        };
        if !call_result_matches_binding(result_kind, value_kind) {
            panic!(
                "conditional_call_elidable! value/result kind mismatch for helper policy {policy:?}"
            );
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call_elidable!");
        // `call.py:249-251 getcalldescr`'s loop-invariant non-void-args
        // assert (see plain `conditional_call!` lowerer for the citation).
        // `conditional_call_elidable!` accepts non-elidable cache-computing
        // helpers per `rlib/jit.py:1334-1336`, so a `LoopInvariant` slot is
        // legal in principle and must enforce the same args-empty rule.
        // Static check applies only to explicit-policy paths; `Infer`
        // resolves slot at runtime from the `__policy` byte.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call_elidable!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_value_policy_check(
                value_kind,
                func_args.is_empty(),
            ))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        self.emit_op(
            OpMeta::linear(
                OpKind::Call,
                arg_regs,
                vec![Register::new(value_kind, result_reg)],
            ),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)`.  `conditional_call_elidable`
        // still accepts non-elidable cache-computing helpers per
        // `rlib/jit.py:1334-1336`; their explicit policy maps to
        // `EffectInfoSlot::CanRaise` and therefore keeps the marker.
        // `Infer` resolves slot at runtime; guard the physical marker with
        // the same can-raise policy cases instead of emitting a redundant
        // PyPy-invisible marker.
        if is_inferred {
            let can_raise_codes: &[u8] = match value_kind {
                BindingKind::Int => &[INT_DONT_LOOK_INSIDE, INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR, REF_DONT_LOOK_INSIDE],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(Binding {
            reg: result_reg,
            kind: value_kind,
            depends_on_stack: false,
        })
    }

    /// RPython jtransform.py:292-313 — `rewrite_op_jit_record_known_result`.
    ///
    /// Recognizes `record_known_result!(result, func, args...)` and emits
    /// the canonical `record_known_result_{i,r}_ir_v` builder entrypoint.
    fn lower_record_known_result(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "record_known_result" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = known result, args[1] = func path, args[2..] = function arguments
        let result_binding = self.lower_value_expr(args[0])?;
        let result_reg = result_binding.reg;
        // jtransform.py:293-295: float → raise Exception
        if matches!(result_binding.kind, BindingKind::Float) {
            panic!("record_known_result does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        let mut arg_regs: Vec<Register> = Vec::new();
        for arg in &args[2..] {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("record_known_result does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // RPython jtransform.py:302-307 — record_known_result_{i|r}
        let builder_call = match result_binding.kind {
            BindingKind::Ref => quote! {
                __builder.record_known_result_r_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
            _ => quote! {
                __builder.record_known_result_i_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
        };
        // RPython pyjitpl.py:413-419 passes the known result box as
        // `prepend_box=resbox`; record_known_result reads that box and
        // produces no result (`_v` suffix).
        // `record_known_result!` requires an elidable callee — the
        // `slot.is_elidable()` assert below catches non-elidable
        // policies.  Default `Infer` to `Elidable*Wrapped` so the
        // assert succeeds when the helper is registered through the
        // wrapped policy path.
        let inferred_default = match result_binding.kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "record_known_result!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("record_known_result! helper policy {policy:?} has no direct-call result kind");
        };
        if !call_result_matches_binding(result_kind, result_binding.kind) {
            panic!("record_known_result! result kind mismatch for helper policy {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "record_known_result!");
        if !slot.is_elidable() {
            panic!("record_known_result! requires an elidable helper policy, got {policy:?}");
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_record_known_result_policy_check(
                result_binding.kind,
            ))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        let result_typed = Register::new(result_binding.kind, result_reg);
        let mut reads = Vec::with_capacity(arg_regs.len() + 1);
        reads.push(result_typed);
        reads.extend(arg_regs);
        self.emit_op(
            OpMeta::linear(OpKind::RecordKnownResult, reads, Vec::new()),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:311-312`: append `-live-` exactly when the
        // elidable calldescr can raise.  In inferred mode, guard the
        // physical marker on the elidable-can-raise / memoryerror policy
        // bytes instead of emitting one for elidable_cannot_raise.
        if is_inferred {
            let can_raise_codes: &[u8] = match result_binding.kind {
                BindingKind::Int => &[INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    fn lower_expr_stmt(&mut self, expr: &Expr) -> Option<()> {
        // jtransform.py:596 rewrite_op_hint — `hint(x, promote=True)` in
        // statement context.  Routes both `x = promote(arg)` (plain local
        // re-assignment, no state-write to trigger
        // `lower_state_field_write`'s RHS recursion) and bare
        // `promote(x);` through `lower_promote_call`, which emits the
        // `-live-` + `<kind>_guard_value` pair.  Without this site the
        // statement-form promote would silently no-op when the
        // config-aware fall-through later observes `stmt_modifies_jit_
        // state(stmt) == false`.
        if let Some(()) = self.lower_promote_stmt(expr) {
            return Some(());
        }
        // State field writes (register/tape machines).
        if let Some(()) = self.lower_state_field_update(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_field_write(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_array_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:923 — virtualizable field write rewrite.
        if let Some(()) = self.lower_vable_field_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:794 — virtualizable array write rewrite.
        if let Some(()) = self.lower_vable_array_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:650 — hint_force_virtualizable rewrite.
        if let Some(()) = self.lower_vable_force(expr) {
            return Some(());
        }
        // RPython jtransform.py:655 — access_directly/fresh_virtualizable suppression.
        if let Some(()) = self.lower_vable_hint_suppress(expr) {
            return Some(());
        }
        // RPython jtransform.py:1685 — conditional_call!(condition, func, args...)
        if let Some(()) = self.lower_conditional_call(expr) {
            return Some(());
        }
        // RPython jtransform.py:292 — record_known_result!(result, func, args...)
        if let Some(()) = self.lower_record_known_result(expr) {
            return Some(());
        }

        if let Expr::If(expr_if) = expr {
            return self.lower_if_stmt(expr_if);
        }

        if let Expr::Match(expr_match) = expr {
            return self.lower_match_stmt(expr_match);
        }

        if let Expr::While(expr_while) = expr {
            return self.lower_while_loop(expr_while);
        }

        if let Expr::Loop(expr_loop) = expr {
            return self.lower_loop_expr(expr_loop);
        }

        if let Expr::ForLoop(expr_for) = expr {
            return self.lower_for_loop(expr_for);
        }

        if let Some(()) = self.lower_config_call_stmt(expr) {
            return Some(());
        }

        // Config-aware patterns
        if self.config.is_some() {
            if let Some(()) = self.lower_io_call_stmt(expr) {
                return Some(());
            }
        }

        None
    }

    // ── Config-aware lowering methods ────────────────────────────────

    fn lower_config_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding);
        }
        let func = &call.func;
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualVoid
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise => {
                    // `call.py:301-303 getcalldescr`: `EF_CAN_RAISE` for the
                    // analyzer-absent default, `EF_CANNOT_RAISE` when
                    // `_canraise(op) == False` on the non-elidable `else`
                    // branch.  Both share the residual_call dispatch byte
                    // family; only the descr's `EffectInfo` differs.
                    let cannot_raise = matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise,
                    );
                    let call_stmt = if cannot_raise {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                __typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! {
                            __builder.residual_call_void_canonical_via_target(__fn_idx, __typed_args);
                        }
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                let __typed_args = #typed_args;
                                #call_stmt
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                let __typed_args = #typed_args;
                                #call_stmt
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::MayForceVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::ReleaseGilVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::LoopInvariantVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                // Stmt-form variants of result-returning policies discard
                // the value but still need the IR call op recorded so the
                // compiled trace runs the side effect (e.g. aheui OP_POP's
                // `lj::stack_pop(state.selected_ref);` discards the popped
                // value but the pop side effect must reach compiled code).
                // Allocate a throwaway destination register; never read it.
                //
                // RPython jtransform.py:456 `handle_residual_call` lowers
                // every direct_call to a residual_call regardless of result
                // usage; majit's CallPolicyKind enum captures the effect
                // distinction (Residual / MayForce / ReleaseGil /
                // LoopInvariant / Elidable) so the dispatched bytecode
                // varies per policy here.  Wrapped variants stay deferred
                // — wrapper closure plumbing is shared with the void path
                // and not exercised by current `#[jit_interp]` users.
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::ReleaseGilInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let throwaway_reg = self.alloc_reg();
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
                                vec![Register::int(throwaway_reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #throwaway_reg,
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
                                vec![Register::int(throwaway_reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.#canonical_call(__fn_idx, #typed_args, #throwaway_reg);
                            },
                        );
                    }
                }
                // `call.py:303 getcalldescr` non-elidable EF_CANNOT_RAISE
                // for int residuals.  Dispatches via the
                // `_with_effect_info(cannot_raise_effect_info())` builder
                // method so the recorded calldescr's `EffectInfo`
                // matches PyPy's `cannot_raise_effect_info()`.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #throwaway_reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // Parity #14 Slice C.4 + Parity #20: Pure flows through
                    // the canonical `BC_RESIDUAL_CALL_*_I` family with the
                    // calldescr's `extra_info` set per `call.py:292-299
                    // _canraise(op)`'s 3-way pick.  The walker
                    // (`pyjitpl/dispatch.rs` Slice C.1) reads
                    // `effectinfo.check_is_elidable()` and routes through
                    // `record_result_of_call_pure` mirroring
                    // `pyjitpl.py:2111-2115`; the trailing
                    // `GUARD_NO_EXCEPTION` is gated on
                    // `effectinfo.check_can_raise(False)` so cannot-raise
                    // elidable callees skip it.
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ResidualVoidWrapped
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // `call.py:301-303 getcalldescr`: descr's `EffectInfo`
                    // differs by the analyzer's `_canraise` result, but the
                    // residual_call dispatch family is the same.
                    let call_stmt = if matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped,
                    ) {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! { __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args); }
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::MayForceVoidWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilVoidWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::MayForceVoidWrapped => {
                            quote! { __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilVoidWrapped => {
                            quote! { __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                            quote! { __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                            #call_stmt
                        },
                    );
                }
                // Wrapped Int / Ref / Float statement-form: result discarded,
                // but the residual_call must still execute the side effect on
                // the compiled trace.  RPython jtransform.py:456
                // handle_residual_call lowers every direct_call regardless of
                // result usage; the wrapped policy adds the trace_target /
                // concrete_target tuple resolution shared with the void
                // wrapped variants above.  Throwaway destination register is
                // allocated (per-bank slot picked by JitCodeBuilder when the
                // typed call dispatches) and never read.
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
                    let throwaway_reg = self.alloc_reg();
                    // Result bank — pick from the wrapped policy variant family.
                    let result_kind = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                        | crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            BindingKind::Int
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            BindingKind::Ref
                        }
                        _ => BindingKind::Float,
                    };
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped => {
                            quote! { __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped => {
                            quote! { __builder.call_release_gil_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        _ => unreachable!(),
                    };
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, throwaway_reg)],
                        ),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                            #call_stmt
                        },
                    );
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                self.emit_op(
                    OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                    quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                        let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
                        match __policy {
                            #VOID_DONT_LOOK_INSIDE => {
                                __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // `call.py:303` non-elidable EF_CANNOT_RAISE for void.
                            #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                __builder.residual_call_void_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                            #VOID_MAY_FORCE => {
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            #VOID_RELEASE_GIL => {
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            #VOID_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // The `_ =>` arm is a runtime invariant violation
                            // (helper policy companion fn returned an
                            // unrecognized byte), NOT a recoverable lower-time
                            // inference failure, so it panics regardless of the
                            // outer Lowerer's `InferenceFailureMode`. Earlier
                            // versions routed this through
                            // `inference_failure_tokens` which emits
                            // `return None;` in `ReturnNone` mode — wrong for
                            // dispatch-body wrappers that return `JitCode`,
                            // not `Option<_>`, surfaced as a type-check error
                            // when a `dont_look_inside` helper is called from
                            // a dispatch JitCode body (A.2.5).
                            other => panic!(
                                "inferred void-call policy returned unrecognized byte {other}; \
                                 expected one of 1 (residual), 9 (may_force), 13 (release_gil), \
                                 17 (loopinvariant)"
                            ),
                        }
                    },
                );
            }
        }
        Some(())
    }

    /// Lower I/O call: aheui_io::write_number(r, writer) → residual_call_void(shim, r)
    fn lower_io_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;

        for (io_path, shim) in &config.io_shims {
            if func_segments == *io_path {
                let arg = unwrap_ref_expr(call.args.first()?);
                let binding = self.lower_value_expr(arg)?;
                let reg = binding.reg;
                // residual_call_void_args takes int-banked args.
                self.emit_op(
                    OpMeta::linear(OpKind::Call, vec![Register::int(reg)], vec![]),
                    quote! {
                        let __fn_idx = __builder.add_fn_ptr(#shim as *const ());
                        __builder.residual_call_void_canonical_via_target(
                            __fn_idx,
                            &[majit_metainterp::JitCallArg::int(#reg)],
                        );
                    },
                );
                return Some(());
            }
        }

        None
    }

    /// Check if a statement modifies JIT-visible state (storage writes).
    fn stmt_modifies_jit_state(&self, stmt: &Stmt) -> bool {
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
    fn expr_touches_storage(&self, expr: &Expr) -> bool {
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
            Expr::Match(m) => self.expr_references_unknown_local(&m.expr),
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
            Expr::Field(_)
            | Expr::Index(_)
            | Expr::Path(_)
            | Expr::Lit(_)
            | Expr::Try(_)
            | Expr::Match(_)
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

    fn lower_if_stmt(&mut self, expr_if: &ExprIf) -> Option<()> {
        let cond = self.lower_value_expr(&expr_if.cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;
        let then_seq = self.lower_branch_expr(&Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: expr_if.then_branch.clone(),
        }))?;
        let else_seq = match expr_if.else_branch.as_ref() {
            Some((_, else_expr)) => self.lower_branch_expr(else_expr)?,
            None => LoweredSequence::default(),
        };

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        // RPython `flatten.py:259` `-live-` convention: every guard-bearing
        // instruction is *preceded* by a `live` marker (byte order:
        // `BC_LIVE+offset` then the guard op). The recorded `orgpc` (=
        // RPython `pyjitpl.py:3713 orgpc = position`, copied to the guard's
        // `resumepc` via `record_state_guard`) is the byte position of the
        // guard op itself, so the BC_LIVE marker sits at `orgpc - SIZE_LIVE_OP`
        // and blackhole's `get_current_position_info` reads liveness from
        // there.  Without this preceding marker, blackhole panics with
        // `missing liveness[N] in JitCode`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);
        self.append_lowered_sequence(then_seq);
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);
        self.append_lowered_sequence(else_seq);
        self.emit_label_def(&end_label);
        Some(())
    }

    /// Lower a standalone match expression to a chained if-else guard sequence.
    ///
    /// ```text
    /// match x { 1 => body1, 2 => body2, _ => default }
    /// ```
    /// becomes:
    /// ```text
    /// eq_1 = (x == 1); brz eq_1, next1; body1; jmp end; next1:
    /// eq_2 = (x == 2); brz eq_2, next2; body2; jmp end; next2:
    /// default; end:
    /// ```
    fn lower_match_stmt(&mut self, expr_match: &syn::ExprMatch) -> Option<()> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

        // Separate literal/path arms from the wildcard/default arm.
        let mut guarded_arms = Vec::new();
        let mut default_arm = None;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => {
                    // Catch-all binding like `x => ...` treated as default
                    default_arm = Some(&arm.body);
                }
                _ => {
                    let literals = extract_pat_literals(&arm.pat)?;
                    guarded_arms.push((literals, &arm.body));
                }
            }
        }

        let disc_reg = discriminant.reg;

        for (literals, body) in &guarded_arms {
            let next_label = self.alloc_label();
            self.emit_aux(quote! { let #next_label = __builder.new_label(); });

            if literals.len() == 1 {
                // Single literal: eq check + branch
                let value = literals[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                    quote! { __builder.load_const_i_value(#const_reg, #value); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(eq_reg, &next_label);
            } else {
                // Multiple literals (Or pattern): chain with logical OR
                // (val == lit1) | (val == lit2) | ...
                let first_val = literals[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::LoadConstI,
                        vec![],
                        vec![Register::int(first_const_reg)],
                    ),
                    quote! { __builder.load_const_i_value(#first_const_reg, #first_val); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, first_const_reg]),
                        vec![Register::int(or_reg)],
                    ),
                    quote! { __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg); },
                );
                for &lit_val in &literals[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.emit_op(
                        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                        quote! { __builder.load_const_i_value(#const_reg, #lit_val); },
                    );
                    self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::BinopI,
                            Register::ints(&[or_reg, eq_reg]),
                            vec![Register::int(new_or_reg)],
                        ),
                        quote! { __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg); },
                    );
                    or_reg = new_or_reg;
                }
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(or_reg, &next_label);
            }

            let body_seq = self.lower_branch_expr(body)?;
            self.append_lowered_sequence(body_seq);
            self.emit_jump(&end_label);
            self.emit_label_def(&next_label);
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let default_seq = self.lower_branch_expr(default_body)?;
            self.append_lowered_sequence(default_seq);
        }

        self.emit_label_def(&end_label);
        Some(())
    }

    // ── Loop lowering ────────────────────────────────────────────────

    /// Lower `while cond { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval cond
    ///   goto_if_not_int_is_true(cond, loop_end)
    ///   eval body
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    fn lower_while_loop(&mut self, expr_while: &syn::ExprWhile) -> Option<()> {
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.emit_aux(quote! { let #loop_start = __builder.new_label(); });
        self.emit_aux(quote! { let #loop_end = __builder.new_label(); });
        self.emit_label_def(&loop_start);

        // Evaluate the condition
        let cond = self.lower_value_expr(&expr_while.cond)?;
        let cond_reg = cond.reg;
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &loop_end);

        // Lower the body, with break targets pointing to loop_end
        let body_seq = self.lower_loop_body(&expr_while.body, &loop_end, &loop_start)?;
        self.append_lowered_sequence(body_seq);

        // Back-edge jump
        self.emit_jump(&loop_start);
        self.emit_label_def(&loop_end);
        Some(())
    }

    /// Lower `loop { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval body (break → jump loop_end, continue → jump loop_start)
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    fn lower_loop_expr(&mut self, expr_loop: &syn::ExprLoop) -> Option<()> {
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.emit_aux(quote! { let #loop_start = __builder.new_label(); });
        self.emit_aux(quote! { let #loop_end = __builder.new_label(); });
        self.emit_label_def(&loop_start);

        let body_seq = self.lower_loop_body(&expr_loop.body, &loop_end, &loop_start)?;
        self.append_lowered_sequence(body_seq);

        self.emit_jump(&loop_start);
        self.emit_label_def(&loop_end);
        Some(())
    }

    /// Lower `for _ in _ { body }`.
    ///
    /// For-loops involve Rust's iterator protocol which cannot be
    /// statically decomposed at proc-macro time. Return `None` so the
    /// arm falls back to opaque (not traced through by the JIT).
    fn lower_for_loop(&mut self, _expr_for: &syn::ExprForLoop) -> Option<()> {
        None
    }

    /// Lower a loop body block, translating `break` → jump to `break_label`
    /// and `continue` → jump to `continue_label`.
    fn lower_loop_body(
        &mut self,
        block: &syn::Block,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<LoweredSequence> {
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

        for stmt in &block.stmts {
            if nested
                .lower_loop_stmt(stmt, break_label, continue_label)
                .is_none()
            {
                // Fall back: try normal lowering
                nested.lower_stmt(stmt)?;
            }
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some(LoweredSequence::new(nested.statements, nested.op_metadata))
    }

    /// Lower a statement inside a loop body, handling break/continue specially.
    fn lower_loop_stmt(
        &mut self,
        stmt: &Stmt,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<()> {
        match stmt {
            Stmt::Expr(Expr::Break(_), _) => {
                self.emit_jump(&break_label);
                Some(())
            }
            Stmt::Expr(Expr::Continue(_), _) => {
                self.emit_jump(&continue_label);
                Some(())
            }
            Stmt::Expr(Expr::If(expr_if), _) => {
                self.lower_loop_if(expr_if, break_label, continue_label)
            }
            _ => None,
        }
    }

    /// Lower an if-expression inside a loop body, where branches may
    /// contain break/continue.
    fn lower_loop_if(
        &mut self,
        expr_if: &ExprIf,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<()> {
        // Check if any branch contains break or continue
        let then_has_loop_ctrl = block_has_loop_control(&expr_if.then_branch);
        let else_has_loop_ctrl = expr_if
            .else_branch
            .as_ref()
            .is_some_and(|(_, e)| expr_has_loop_control(e));

        if !then_has_loop_ctrl && !else_has_loop_ctrl {
            return None; // no break/continue, fall back to normal lowering
        }

        let cond = self.lower_value_expr(&expr_if.cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);

        // Lower then-branch with loop control
        let then_seq = self.lower_loop_body(&expr_if.then_branch, break_label, continue_label)?;
        self.append_lowered_sequence(then_seq);
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);

        // Lower else-branch with loop control
        if let Some((_, else_expr)) = &expr_if.else_branch {
            let else_block = match &**else_expr {
                Expr::Block(block) => &block.block,
                _ => return None,
            };
            let else_seq = self.lower_loop_body(else_block, break_label, continue_label)?;
            self.append_lowered_sequence(else_seq);
        }

        self.emit_label_def(&end_label);
        Some(())
    }

    /// Lower a match expression in value position to chained if-else guards
    /// that produce a value.
    fn lower_match_value(&mut self, expr_match: &syn::ExprMatch) -> Option<Binding> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

        let mut guarded_arms = Vec::new();
        let mut default_arm = None;
        let mut depends_on_stack = discriminant.depends_on_stack;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => {
                    default_arm = Some(&arm.body);
                }
                _ => {
                    let literals = extract_pat_literals(&arm.pat)?;
                    guarded_arms.push((literals, &arm.body));
                }
            }
        }

        let disc_reg = discriminant.reg;

        for (literals, body) in &guarded_arms {
            let next_label = self.alloc_label();
            self.emit_aux(quote! { let #next_label = __builder.new_label(); });

            if literals.len() == 1 {
                let value = literals[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                    quote! { __builder.load_const_i_value(#const_reg, #value); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(eq_reg, &next_label);
            } else {
                let first_val = literals[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::LoadConstI,
                        vec![],
                        vec![Register::int(first_const_reg)],
                    ),
                    quote! { __builder.load_const_i_value(#first_const_reg, #first_val); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, first_const_reg]),
                        vec![Register::int(or_reg)],
                    ),
                    quote! { __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg); },
                );
                for &lit_val in &literals[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.emit_op(
                        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                        quote! { __builder.load_const_i_value(#const_reg, #lit_val); },
                    );
                    self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::BinopI,
                            Register::ints(&[or_reg, eq_reg]),
                            vec![Register::int(new_or_reg)],
                        ),
                        quote! { __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg); },
                    );
                    or_reg = new_or_reg;
                }
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(or_reg, &next_label);
            }

            let (body_seq, binding) = self.lower_branch_value_expr(body)?;
            if !matches!(binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= binding.depends_on_stack;
            let arm_reg = binding.reg;
            self.append_lowered_sequence(body_seq);
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(arm_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #arm_reg); },
            );
            self.emit_jump(&end_label);
            self.emit_label_def(&next_label);
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let (default_seq, default_binding) = self.lower_branch_value_expr(default_body)?;
            if !matches!(default_binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= default_binding.depends_on_stack;
            let default_reg = default_binding.reg;
            self.append_lowered_sequence(default_seq);
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(default_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #default_reg); },
            );
        }

        self.emit_label_def(&end_label);

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack,
        })
    }

    /// RPython jtransform.py:832 `rewrite_op_getfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name` where `frame` is the virtualizable variable
    /// and `field_name` is a declared virtualizable field. Emits a vable_getfield
    /// JitCode instruction that will read from virtualizable_boxes at trace time.
    fn lower_vable_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        if let Expr::Field(field) = expr {
            if !expr_matches_local_name(&field.base, vable_var) {
                return None;
            }
            let member_name = named_member(&field.member)?;

            if let Some(&(field_index, field_type)) = config.vable_fields.get(&member_name) {
                let vable_reg = self.vable_base_reg()?;
                let reg = self.alloc_reg();
                let fi = field_index as u16;
                // vable_reg is Ref; result `reg` bank follows field_type.
                let vable_r = Register::ref_(vable_reg);
                let kind = match field_type {
                    ValueKind::Ref => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::ref_(reg)],
                            ),
                            quote! { __builder.vable_getfield_ref_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Ref
                    }
                    ValueKind::Float => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::float(reg)],
                            ),
                            quote! { __builder.vable_getfield_float_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Float
                    }
                    ValueKind::Int => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::int(reg)],
                            ),
                            quote! { __builder.vable_getfield_int_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Int
                    }
                };
                return Some(Binding {
                    reg,
                    kind,
                    depends_on_stack: false,
                });
            }
        }
        None
    }

    /// RPython jtransform.py:760 `getarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i]` where `frame` is the virtualizable
    /// variable and `locals_w` is a declared virtualizable array field.
    fn lower_vable_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        // Pattern: Expr::Index where base is Expr::Field on vable_var
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;

        // Lower the index expression to a register
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;

        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // vable_reg: Ref. idx_reg: Int. result `reg` bank by item_type.
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        let kind = match item_type {
            ValueKind::Ref => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::ref_(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_ref_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Ref
            }
            ValueKind::Float => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::float(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_float_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Float
            }
            ValueKind::Int => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_int_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Int
            }
        };
        Some(Binding {
            reg,
            kind,
            depends_on_stack: false,
        })
    }

    /// RPython jtransform.py:815 `arraylen_vable`.
    ///
    /// Recognizes `frame.locals_w.len()` for declared virtualizable arrays.
    fn lower_vable_array_len(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let call = match expr {
            Expr::MethodCall(call) => call,
            _ => return None,
        };
        if call.method != "len" || !call.args.is_empty() {
            return None;
        }
        let field = match &*call.receiver {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &array_index = config.vable_arrays.get(&member_name).map(|(idx, _)| idx)?;
        let vable_reg = self.vable_base_reg()?;
        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // vable_arraylen reads vable_reg (Ref) and writes the length to an int reg.
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(vable_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.vable_arraylen_with_base(#reg, #vable_reg, #ai); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
        })
    }

    /// Recognizes `state.field` for scalar state fields.
    fn lower_state_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let field = match expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &field_index = config.state_scalars.get(&member_name)?;
        let fi = field_index as u16;
        let reg = self.alloc_reg();
        // load_state_field reads the field at int index `fi` into int `reg`.
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![], vec![Register::int(reg)]),
            quote! { __builder.load_state_field(#fi, #reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
        })
    }

    /// Recognizes `state.array[index]` for array state fields.
    /// Routes to `load_state_varray` for virtualizable arrays, `load_state_array` for flattened.
    fn lower_state_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let reg = self.alloc_reg();

        // Check virtualizable arrays first, then flattened arrays.
        if let Some(&va_idx) = config.state_virt_arrays.get(&member_name) {
            let ai = va_idx as u16;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::int(idx_reg)],
                    vec![Register::int(reg)],
                ),
                quote! { __builder.load_state_varray(#ai, #idx_reg, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
            });
        }
        let &array_index = config.state_arrays.get(&member_name)?;
        let ai = array_index as u16;
        self.emit_op(
            OpMeta::linear(
                OpKind::StateField,
                vec![Register::int(idx_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.load_state_array(#ai, #idx_reg, #reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
        })
    }

    fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
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
    fn lower_promote_stmt(&mut self, expr: &Expr) -> Option<()> {
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

    fn lower_call_value(&mut self, call: &ExprCall) -> Option<Binding> {
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
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
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
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
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
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_path();
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
                            let __fn_idx = __builder.add_call_target(__trace_target, __concrete_target);
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

    fn lower_branch_expr(&mut self, expr: &Expr) -> Option<LoweredSequence> {
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

    fn lower_branch_value_expr(&mut self, expr: &Expr) -> Option<(LoweredSequence, Binding)> {
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

    fn lower_block_value(&mut self, block: &Block) -> Option<Binding> {
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

// ── Loop control detection ───────────────────────────────────────────

/// Check if a block contains break or continue at the top level (not nested in inner loops).
fn block_has_loop_control(block: &Block) -> bool {
    block.stmts.iter().any(|stmt| stmt_has_loop_control(stmt))
}

fn stmt_has_loop_control(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_has_loop_control(expr),
        _ => false,
    }
}

fn expr_has_loop_control(expr: &Expr) -> bool {
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
        // Don't recurse into nested loops — they have their own break/continue scope
        Expr::Loop(_) | Expr::While(_) | Expr::ForLoop(_) => false,
        _ => false,
    }
}

// ── Helper functions ─────────────────────────────────────────────────

/// Extract the get_mut argument from a pool.get_mut(arg) expression.
fn extract_stmts(expr: &Expr) -> Vec<Stmt> {
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
fn extract_pat_literals(pat: &Pat) -> Option<Vec<i64>> {
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
fn extract_pat_value_tokens(pat: &Pat) -> Option<Vec<TokenStream>> {
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

fn int_arg_regs(bindings: &[Binding]) -> Option<Vec<u16>> {
    bindings
        .iter()
        .map(|binding| match binding.kind {
            BindingKind::Int => Some(binding.reg),
            BindingKind::Ref | BindingKind::Float => None,
        })
        .collect()
}

fn inline_int_arg_tokens(bindings: &[Binding]) -> TokenStream {
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

fn inline_ref_arg_tokens(bindings: &[Binding]) -> TokenStream {
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

fn inline_float_arg_tokens(bindings: &[Binding]) -> TokenStream {
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
fn inline_call_tokens(bindings: &[Binding], result_reg: u16) -> (TokenStream, TokenStream) {
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

fn typed_call_arg_tokens(bindings: &[Binding]) -> TokenStream {
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

fn is_supported_int_cast(ty: &Type) -> bool {
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

fn is_supported_ref_type(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => type_path.path.is_ident("usize"),
        Type::Ptr(_) => true,
        _ => false,
    }
}

fn is_supported_float_type(ty: &Type) -> bool {
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

fn extract_bool_branch_values(expr_if: &ExprIf) -> Option<(i64, i64)> {
    let then_value = extract_block_tail_int(&expr_if.then_branch)?;
    let (_, else_expr) = expr_if.else_branch.as_ref()?;
    let else_value = extract_branch_int(else_expr)?;
    Some((then_value, else_value))
}

fn extract_block_tail_int(block: &Block) -> Option<i64> {
    match block.stmts.as_slice() {
        [Stmt::Expr(expr, None)] => extract_branch_int(expr),
        _ => None,
    }
}

fn extract_branch_int(expr: &Expr) -> Option<i64> {
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

fn inline_builder_path(expr: &Expr) -> Option<Path> {
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
fn inline_prebuild_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_inline_jitcode_{}_prebuild", last.ident);
    Some(path)
}

fn binding_kind_for_inline_policy(kind: crate::jit_interp::CallPolicyKind) -> Option<BindingKind> {
    match kind {
        crate::jit_interp::CallPolicyKind::InlineInt => Some(BindingKind::Int),
        crate::jit_interp::CallPolicyKind::InlineRef => Some(BindingKind::Ref),
        crate::jit_interp::CallPolicyKind::InlineFloat => Some(BindingKind::Float),
        _ => None,
    }
}

pub(super) fn helper_policy_path(expr: &Expr) -> Option<Path> {
    let Expr::Path(ExprPath { path, .. }) = expr else {
        return None;
    };
    let mut path = path.clone();
    let last = path.segments.last_mut()?;
    last.ident = format_ident!("__majit_call_policy_{}", last.ident);
    Some(path)
}

fn opcode_for_binop(op: &BinOp) -> Option<Ident> {
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

fn opcode_for_assign_binop(op: &BinOp) -> Option<Ident> {
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

// ── Public entry points ──────────────────────────────────────────────

/// Generated JitCode body alongside the per-marker liveness prebuild
/// tokens that `__prebuild_jitcode_liveness_*` (codegen_trace.rs)
/// replays at install time. The prebuild ensures every per-pc `-live-`
/// triple lands in `asm.all_liveness` before
/// `metainterp_sd.liveness_info` snapshot, mirroring RPython
/// `pyjitpl.py:2255 finish_setup` order.
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
fn bind_pre_merge_point_stmts(lowerer: &mut Lowerer, func_block: &syn::Block) -> Option<()> {
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
fn collect_pat_bound_idents(pat: &Pat, out: &mut HashSet<String>) {
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

fn collect_arm_caller_locals(
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
fn lower_pre_dispatch_stmts(lowerer: &mut Lowerer, func_block: &syn::Block) -> Option<()> {
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
fn find_dispatch_loop_body<'b>(
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
fn is_jit_merge_point_macro(stmt: &Stmt) -> bool {
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
fn dispatch_arm_inline_call_tokens(layout: &[CallerLocalLayout]) -> proc_macro2::TokenStream {
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
fn lower_dispatch_chain(
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
fn emit_promote_greens(lowerer: &mut Lowerer, config: &LowererConfig) {
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
fn resolve_greens(lowerer: &Lowerer, config: &LowererConfig) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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
fn resolve_reds(lowerer: &Lowerer, config: &LowererConfig) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
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
fn green_schema(lowerer: &Lowerer, config: &LowererConfig) -> Vec<(String, TokenStream)> {
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
fn red_schema(lowerer: &Lowerer, config: &LowererConfig) -> Vec<(String, TokenStream)> {
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
