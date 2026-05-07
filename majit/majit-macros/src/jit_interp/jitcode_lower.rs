use std::collections::{BTreeSet, HashMap};

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
use super::codegen_trace::is_promote_call_path;
use syn::{
    BinOp, Block, Expr, ExprAssign, ExprBinary, ExprCall, ExprCast, ExprIf, ExprLit,
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
        }
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
/// were aggregated during lowering.
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
        #(#inline_prebuild)*
        #(#live_regs)*
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
            Stmt::Item(_) | Stmt::Macro(_) => None,
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

        // Config-aware: runtime constant (expression not touching storage)
        if self.config.is_some() && !self.expr_touches_storage(&init.expr) {
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
        // statement context.  Routes both `x = promote(x)` (plain local
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
                                majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                // `_with_effect_info(CANNOT_RAISE_EFFECT_INFO)` builder
                // method so the recorded calldescr's `EffectInfo`
                // matches PyPy's `CANNOT_RAISE_EFFECT_INFO`.
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
                                majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy does not support void calls here",
                );
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                            _ => {
                                #unsupported
                            }
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
            _ => None,
        }
    }

    /// Statement-context lowering for `hint(x, promote=True)`:
    ///
    /// - `x = promote(x);` (plain local re-assignment — `lower_state_
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
    /// value-context lowering.  The local binding map already holds
    /// `x → reg`; promote-on-self leaves it unchanged.
    fn lower_promote_stmt(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Assign(assign) => {
                let Expr::Path(_) = &*assign.left else {
                    return None;
                };
                let Expr::Call(call) = &*assign.right else {
                    return None;
                };
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                self.lower_value_expr(&assign.right)?;
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
                                majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                    majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                // RPython `pyjitpl.py:2255 finish_setup` order: the
                // helper's per-marker `-live-` triples must land in
                // `asm.all_liveness` before the parent's
                // `JitDriver::install_canonical_liveness` snapshot.
                //
                // The inferred-policy path resolves the helper at runtime
                // (`__policy == 4u8` → Inline build), so we cannot
                // statically know whether the helper carries a real
                // prebuild fn or none at all (residual / elidable /
                // may-force / etc. helpers have no per-marker triples).
                // The 5th element of the helper's
                // `__majit_call_policy_<name>()` tuple
                // (`emit_helper_policy_fn`) carries the prebuild fn
                // pointer for `#[jit_inline]` helpers and `null` for
                // every other helper attribute that flows through here.
                // At parent-install time we read the pointer and
                // dispatch: non-null → transmute & call → registers the
                // helper's per-marker triples; null → skip.  Avoids a
                // stub fn generation in non-Inline helper macros (some
                // are reachable from crates that don't depend on
                // `majit_metainterp`, so a stub referencing
                // `&mut Assembler` would fail to compile).
                let policy_path_for_prebuild = helper_policy_path(&call.func)?;
                self.inline_liveness_prebuild.push(quote! {
                    {
                        let (_, _, _, _, __prebuild_ptr) = #policy_path_for_prebuild();
                        if !__prebuild_ptr.is_null() {
                            // SAFETY: `#[jit_inline]` (`lib.rs:1330`) is
                            // the only emitter of a non-null prebuild
                            // pointer, and it always points to a fn with
                            // signature `fn(&mut Assembler)`.
                            let __prebuild_fn: fn(&mut majit_metainterp::Assembler) =
                                unsafe { std::mem::transmute(__prebuild_ptr) };
                            __prebuild_fn(__asm);
                        }
                    }
                });
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
                                        majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
                                        majit_metainterp::CANNOT_RAISE_EFFECT_INFO,
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
