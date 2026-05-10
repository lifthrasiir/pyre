//! `#[jit_interp]` proc macro implementation.
//!
//! Transforms an interpreter mainloop function into a JIT-enabled version by:
//! 1. Generating `trace_instruction` from the match dispatch
//! 2. Generating `JitState` types and impl
//! 3. Replacing `jit_merge_point!()` / `can_enter_jit!()` markers

pub(crate) mod call_policy_byte;
mod classify;
mod codegen_state;
mod codegen_trace;
mod green_type_tag;
pub(crate) mod jitcode_lower;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, ItemFn, LitBool, Path, Token, braced, bracketed,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Parsed configuration from `#[jit_interp(...)]` attributes.
///
/// ## Helper discovery
///
/// Helpers (functions called from traced match arms) can be declared in
/// three ways, from most explicit to most concise:
///
/// 1. **`calls = { helper_a, helper_b => residual_int, ... }`**
///    Brace-delimited list with optional per-helper policy overrides.
///
/// 2. **`helpers = [helper_a, helper_b, helper_c]`**
///    Bracket-delimited shorthand — all helpers use auto-inferred policy.
///    Equivalent to listing each in `calls = { ... }` without a `=>` override.
///    Can be combined with `calls` for helpers that need explicit policies.
///
/// 3. **`auto_calls = true`**
///    Infer helper policies from sidecar `#[elidable]` / `#[dont_look_inside]`
///    / `#[jit_inline]` attributes on every call site in the traced arms.
///    Value-call inference is currently limited to helpers whose result bank
///    is statically int-shaped; ref/float-return helpers still need explicit
///    `calls = { helper => ... }` overrides such as `inline_ref`,
///    `inline_float`, `residual_ref_wrapped`, or `residual_float_wrapped`.
///
/// ### Module-level discovery
///
/// For automatic helper discovery, use `#[jit_module]` on the enclosing
/// `mod` block. It scans all items for JIT-annotated functions and
/// generates hidden registry constants (`__MAJIT_DISCOVERED_HELPERS`,
/// `__MAJIT_HELPER_POLICIES`). Alternatively, use `helpers` or `calls`
/// to explicitly list the functions that need JIT integration.
pub struct JitInterpConfig {
    /// The interpreter state type (e.g., `AheuiState`).
    pub state_type: Ident,
    /// The environment type (e.g., `Program`).
    pub env_type: Ident,
    /// Interpreter I/O function → JIT shim function mapping.
    pub io_shims: Vec<(Path, Ident)>,
    /// Interpreter function call policies for helper calls.
    /// Populated from both `calls = { ... }` and `helpers = [...]`.
    pub calls: Vec<CallEntry>,
    /// Whether direct helper calls should be auto-inferred from sidecar metadata.
    pub auto_calls: bool,
    /// Optional structured green-key expressions for marker rewrite.
    pub greens: Vec<Expr>,
    /// Slice (audit Issue #6) — explicit red declarations for the
    /// dispatch JitCode `BC_JIT_MERGE_POINT` payload.  RPython
    /// `jtransform.py:1700 make_three_lists(op.args[2+num_green_args:])`
    /// derives reds from the marker call's tail args; pyre's marker is
    /// stateless (no tail args), so consumers declare the reds via
    /// this config slot instead.  Empty = use the default candidate
    /// list `[program, pc(+ optional vable)]` with declared greens
    /// filtered out (the pre-Issue-#6 pyre behavior).
    pub reds: Vec<Expr>,
    /// Slice 92.2 — per-green type tag from `greens = [name: tag]` syntax.
    /// Lockstep with [`Self::greens`] (same length, same order).  `None`
    /// at a given index means the green flows through the
    /// `<_ as GreenAsI64>::__green_repr` trait dispatch unchanged;
    /// `Some(tag)` overrides the dispatch with an explicit
    /// `(value_bits, GreenType::<tag>)` pair so str / unicode greens route
    /// through the hardcoded `default_str_eq` / `default_str_hash` /
    /// `default_unicode_hash` in `majit-ir/src/value.rs`
    /// (`warmstate.py:108-128 lltype.Ptr STR/UNICODE` parity, no
    /// frontend override).
    pub green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
    /// Virtualizable frame field declaration.
    ///
    /// RPython equivalent: jtransform.py's `is_virtualizable_getset()`.
    /// When set, the proc macro rewrites field accesses on the virtualizable
    /// variable to use TraceCtx vable_* methods instead of heap operations.
    pub virtualizable_decl: Option<VirtualizableDecl>,
    /// State field declarations for register/tape machines.
    ///
    /// When set, the macro tracks state struct fields as JIT-managed values
    /// instead of requiring a storage pool. Enables `state.field` and
    /// `state.array[index]` patterns in match arms.
    pub state_fields: Option<StateFieldsConfig>,
}

/// Virtualizable frame field declaration for `#[jit_interp]`.
///
/// RPython equivalent: VirtualizableInfo from virtualizable.py, combined
/// with jtransform.py's field-to-descriptor mapping.
///
/// Syntax in attribute:
/// ```ignore
/// virtualizable_fields = {
///     var: frame,
///     token_offset: PYFRAME_VABLE_TOKEN_OFFSET,
///     fields: { next_instr: int @ NEXT_INSTR_OFFSET, ... },
///     arrays: {
///         locals_w: ref @ LOCALS_OFFSET,
///         stack: int @ (DATA_PTR_OFFSET + SLOT_OFFSET) {
///             ptr_offset: 0,
///             length_offset: LENGTH_OFFSET_MINUS_DATA_PTR,
///             items_offset: 0,
///         },
///     },
/// }
/// ```
pub struct VirtualizableDecl {
    /// Expression for the virtualizable variable in the mainloop body.
    pub var_name: Ident,
    /// Constant path for the vable_token field offset.
    pub token_offset: Path,
    /// Static fields: name, type (int/ref/float), byte offset constant.
    pub fields: Vec<VableFieldDecl>,
    /// Array fields: name, item type (int/ref/float), byte offset constant.
    pub arrays: Vec<VableArrayDecl>,
}

/// A single virtualizable static field declaration.
pub struct VableFieldDecl {
    /// Field name as it appears in the struct.
    pub name: Ident,
    /// Field type: `int`, `ref`, or `float`.
    pub field_type: Ident,
    /// Byte offset expression (e.g., `PYFRAME_NEXT_INSTR_OFFSET`,
    /// `STORAGEPOOL_LENGTHS_OFFSET + 8`).
    pub offset: Expr,
}

/// A single virtualizable array field declaration.
pub struct VableArrayDecl {
    /// Array field name as it appears in the struct.
    pub name: Ident,
    /// Item type: `int`, `ref`, or `float`.
    pub item_type: Ident,
    /// Physical layout of the array field inside the virtualizable object.
    pub layout: VableArrayLayoutDecl,
}

/// Layout description for a virtualizable array field.
pub enum VableArrayLayoutDecl {
    /// Direct pointer field: the virtualizable stores a pointer to the
    /// array object. `length_offset` and `items_offset` describe where,
    /// relative to that pointer, the length prefix and items begin.
    /// Matches RPython `Ptr(GcArray(T))`: `length_offset = 0`,
    /// `items_offset = size_of::<usize>()` (8 on 64-bit).
    ///
    /// `None` defaults to `0` for back-compat with pointer-to-items shapes
    /// where items sit directly at the field pointer.
    Direct {
        field_offset: Expr,
        length_offset: Option<Expr>,
        items_offset: Option<Expr>,
    },
    /// Embedded container layout: pointer/length live in sibling fields or
    /// an inline container relative to the declared field offset.
    Embedded {
        field_offset: Expr,
        ptr_offset: Expr,
        length_offset: Expr,
        items_offset: Expr,
    },
}

/// State field declaration for register/tape machines.
///
/// Syntax: `state_fields = { a: int, regs: [int], ... }`
///
/// Current implementation supports only `int` and `[int]`.
pub struct StateFieldsConfig {
    pub fields: Vec<StateFieldDecl>,
}

/// A single state field declaration.
pub struct StateFieldDecl {
    pub name: Ident,
    pub kind: StateFieldKind,
}

/// Whether a state field is a scalar, array, virtualizable array, or
/// an opaque pass-through (RPython parity: a field that lives on the state
/// struct but is invisible to the JIT — accessed only through explicit
/// interpreter code or via raw-pointer handles. Used for `Storage`-like
/// pools with polymorphic dispatch that can not be flattened as ints).
pub enum StateFieldKind {
    /// Scalar value.
    ///
    /// Syntax: `a: int` (default i64 storage) or `a: int(usize)` /
    /// `a: int(i32)` to declare a different Rust storage type. RPython
    /// parity: `lltype.Signed` is i64 word-sized, `lltype.Unsigned` is
    /// usize word-sized — both render to a single Int register in IR
    /// but the user-visible struct field carries the declared type. The
    /// macro emits `as i64` / `as <type>` casts at the JIT boundary so
    /// IR sees a uniform i64 while the interpreter keeps natural Rust
    /// types (e.g. aheui's `selected: usize`, `stacksize: i32`).
    Scalar {
        ir_type: Ident,
        /// `None` ⇒ Rust storage type matches `ir_type` (default i64
        /// for `int`). `Some(path)` ⇒ explicit override from
        /// `int(<path>)` syntax.
        rust_type: Option<syn::Path>,
    },
    /// Array value (e.g., `regs: [int]`) — flattened into inputargs.
    Array(Ident),
    /// Virtualizable array (e.g., `tape: [int; virt]`) — NOT flattened.
    /// Only the data pointer and length are tracked as inputargs.
    /// Element access emits GETARRAYITEM_RAW_I / SETARRAYITEM_RAW IR ops.
    VirtArray(Ident),
    /// Opaque pass-through (e.g., `storage: opaque(Storage)`) — the field
    /// keeps its declared type on the state struct but is NOT enumerated
    /// as a JIT inputarg, fail_arg, or sym slot. The codegen skips it
    /// entirely; the interpreter is responsible for accessing it directly.
    /// Used to carry pools/handles whose internal layout is not flat ints
    /// (e.g. polymorphic `Storage`). Pair with `opaque(Type)` raw-pointer
    /// handles passed via additional `int` scalars when the JIT needs to
    /// touch the underlying memory through `jit_promote!` + raw-load IR.
    Opaque(syn::Path),
}

/// One entry in the `calls = { ... }` / `helpers = [ ... ]` map.
#[derive(Clone)]
pub struct CallEntry {
    pub path: Path,
    pub policy: Option<CallPolicyKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallPolicyKind {
    ResidualVoid,
    ResidualVoidWrapped,
    /// `EF_CANNOT_RAISE` (`call.py:303 getcalldescr`'s non-elidable
    /// `else` branch).  Producers pick this when they statically know
    /// the callee cannot raise but is otherwise neither elidable nor
    /// loop-invariant — e.g. flat TLS / buffer shims.  Maps to
    /// [`CondCallEffectSlot::CannotRaise`].
    ResidualVoidCannotRaise,
    ResidualVoidCannotRaiseWrapped,
    MayForceVoid,
    MayForceVoidWrapped,
    ReleaseGilVoid,
    ReleaseGilVoidWrapped,
    LoopInvariantVoid,
    LoopInvariantVoidWrapped,
    ResidualInt,
    ResidualIntWrapped,
    /// `EF_CANNOT_RAISE` (`call.py:303 getcalldescr`'s non-elidable
    /// `else` branch) for int-returning residual helpers.  Mirrors
    /// the void-side `ResidualVoidCannotRaise` pair.  Producers pick
    /// this when the callee is statically known to be non-elidable
    /// AND cannot raise.
    ResidualIntCannotRaise,
    ResidualIntCannotRaiseWrapped,
    MayForceInt,
    MayForceIntWrapped,
    ReleaseGilInt,
    ReleaseGilIntWrapped,
    LoopInvariantInt,
    LoopInvariantIntWrapped,
    // `EF_ELIDABLE_CAN_RAISE` (call.py:297). Default elidable variant —
    // emits trailing `GUARD_NO_EXCEPTION`.
    ElidableInt,
    ElidableIntWrapped,
    // `EF_ELIDABLE_CANNOT_RAISE` (call.py:299). Skips the
    // `GUARD_NO_EXCEPTION` because `effectinfo.check_can_raise(False)`
    // is false for `extraeffect == 0`.
    ElidableIntCannotRaise,
    ElidableIntCannotRaiseWrapped,
    // `EF_ELIDABLE_OR_MEMORYERROR` (call.py:295). Same dispatch as
    // `_can_raise` but distinguishes memory-only failure modes.
    ElidableIntOrMemerror,
    ElidableIntOrMemerrorWrapped,
    ResidualRefWrapped,
    /// `EF_CANNOT_RAISE` for ref-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped`; the unwrapped variant
    /// is absent because the inferred lower path cannot recover a
    /// static ref-return BindingKind from the policy byte alone
    /// (mirrors the existing `ResidualRefWrapped`-only shape).
    ResidualRefCannotRaiseWrapped,
    MayForceRefWrapped,
    // ReleaseGilRefWrapped intentionally absent: resoperation.py:1243-1244
    // (`# no such thing`) excludes CALL_RELEASE_GIL_R from the upstream
    // opcode table, so a `'release_gil_ref_wrapped'` policy could only
    // emit an IR op the optimizer/backend cannot consume.
    LoopInvariantRefWrapped,
    ElidableRefWrapped,
    ElidableRefCannotRaiseWrapped,
    ElidableRefOrMemerrorWrapped,
    ResidualFloatWrapped,
    /// `EF_CANNOT_RAISE` for float-returning residual helpers.
    /// Mirrors `ResidualIntCannotRaiseWrapped` / `ResidualRefCannotRaiseWrapped`.
    ResidualFloatCannotRaiseWrapped,
    MayForceFloatWrapped,
    ReleaseGilFloatWrapped,
    LoopInvariantFloatWrapped,
    ElidableFloatWrapped,
    ElidableFloatCannotRaiseWrapped,
    ElidableFloatOrMemerrorWrapped,
    InlineInt,
    InlineRef,
    InlineFloat,
}

pub(crate) fn parse_call_policy_kind(kind: &Ident) -> Option<CallPolicyKind> {
    Some(match kind.to_string().as_str() {
        "residual_void" => CallPolicyKind::ResidualVoid,
        "residual_void_wrapped" => CallPolicyKind::ResidualVoidWrapped,
        "residual_void_cannot_raise" => CallPolicyKind::ResidualVoidCannotRaise,
        "residual_void_cannot_raise_wrapped" => CallPolicyKind::ResidualVoidCannotRaiseWrapped,
        "may_force_void" => CallPolicyKind::MayForceVoid,
        "may_force_void_wrapped" => CallPolicyKind::MayForceVoidWrapped,
        "release_gil_void" => CallPolicyKind::ReleaseGilVoid,
        "release_gil_void_wrapped" => CallPolicyKind::ReleaseGilVoidWrapped,
        "loopinvariant_void" => CallPolicyKind::LoopInvariantVoid,
        "loopinvariant_void_wrapped" => CallPolicyKind::LoopInvariantVoidWrapped,
        "residual_int" => CallPolicyKind::ResidualInt,
        "residual_int_wrapped" => CallPolicyKind::ResidualIntWrapped,
        "residual_int_cannot_raise" => CallPolicyKind::ResidualIntCannotRaise,
        "residual_int_cannot_raise_wrapped" => CallPolicyKind::ResidualIntCannotRaiseWrapped,
        "may_force_int" => CallPolicyKind::MayForceInt,
        "may_force_int_wrapped" => CallPolicyKind::MayForceIntWrapped,
        "release_gil_int" => CallPolicyKind::ReleaseGilInt,
        "release_gil_int_wrapped" => CallPolicyKind::ReleaseGilIntWrapped,
        "loopinvariant_int" => CallPolicyKind::LoopInvariantInt,
        "loopinvariant_int_wrapped" => CallPolicyKind::LoopInvariantIntWrapped,
        // `call.py:292-299 _canraise(op)` 3-way pick on the elidable
        // branch. `elidable_*` (no suffix) is the EF_ELIDABLE_CAN_RAISE
        // default; `_cannot_raise` / `_or_memerror` map to
        // EF_ELIDABLE_CANNOT_RAISE / EF_ELIDABLE_OR_MEMORYERROR.
        "elidable_int" => CallPolicyKind::ElidableInt,
        "elidable_int_wrapped" => CallPolicyKind::ElidableIntWrapped,
        "elidable_int_cannot_raise" => CallPolicyKind::ElidableIntCannotRaise,
        "elidable_int_cannot_raise_wrapped" => CallPolicyKind::ElidableIntCannotRaiseWrapped,
        "elidable_int_or_memerror" => CallPolicyKind::ElidableIntOrMemerror,
        "elidable_int_or_memerror_wrapped" => CallPolicyKind::ElidableIntOrMemerrorWrapped,
        "residual_ref_wrapped" => CallPolicyKind::ResidualRefWrapped,
        "residual_ref_cannot_raise_wrapped" => CallPolicyKind::ResidualRefCannotRaiseWrapped,
        "may_force_ref_wrapped" => CallPolicyKind::MayForceRefWrapped,
        // "release_gil_ref_wrapped" intentionally rejected per
        // resoperation.py:1243-1244 — see CallPolicyKind comment.
        "loopinvariant_ref_wrapped" => CallPolicyKind::LoopInvariantRefWrapped,
        "elidable_ref_wrapped" => CallPolicyKind::ElidableRefWrapped,
        "elidable_ref_cannot_raise_wrapped" => CallPolicyKind::ElidableRefCannotRaiseWrapped,
        "elidable_ref_or_memerror_wrapped" => CallPolicyKind::ElidableRefOrMemerrorWrapped,
        "residual_float_wrapped" => CallPolicyKind::ResidualFloatWrapped,
        "residual_float_cannot_raise_wrapped" => CallPolicyKind::ResidualFloatCannotRaiseWrapped,
        "may_force_float_wrapped" => CallPolicyKind::MayForceFloatWrapped,
        "release_gil_float_wrapped" => CallPolicyKind::ReleaseGilFloatWrapped,
        "loopinvariant_float_wrapped" => CallPolicyKind::LoopInvariantFloatWrapped,
        "elidable_float_wrapped" => CallPolicyKind::ElidableFloatWrapped,
        "elidable_float_cannot_raise_wrapped" => CallPolicyKind::ElidableFloatCannotRaiseWrapped,
        "elidable_float_or_memerror_wrapped" => CallPolicyKind::ElidableFloatOrMemerrorWrapped,
        "inline_int" => CallPolicyKind::InlineInt,
        "inline_ref" => CallPolicyKind::InlineRef,
        "inline_float" => CallPolicyKind::InlineFloat,
        _ => return None,
    })
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut env_type = None;
        let mut io_shims = None;
        let mut calls: Vec<CallEntry> = Vec::new();
        let mut auto_calls = None;
        let mut greens = None;
        let mut reds = None;
        let mut virtualizable_decl = None;
        let mut state_fields = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "env" => {
                    env_type = Some(input.parse::<Ident>()?);
                }
                "io_shims" => {
                    io_shims = Some(parse_io_shim_map(input)?);
                }
                "calls" => {
                    calls.extend(parse_call_map(input)?);
                }
                "helpers" => {
                    calls.extend(parse_helpers_list(input)?);
                }
                "auto_calls" => {
                    auto_calls = Some(input.parse::<LitBool>()?.value);
                }
                "greens" => {
                    let specs = green_type_tag::parse_green_spec_list(input)?;
                    greens = Some(specs);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_expr_list(input)?);
                }
                "virtualizable_fields" => {
                    virtualizable_decl = Some(parse_virtualizable_decl(input)?);
                }
                "state_fields" => {
                    state_fields = Some(parse_state_fields(input)?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_interp parameter: `{other}`"),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        let state_type =
            state_type.ok_or_else(|| syn::Error::new(input.span(), "missing `state` parameter"))?;
        let env_type =
            env_type.ok_or_else(|| syn::Error::new(input.span(), "missing `env` parameter"))?;

        if state_fields.is_none() {
            return Err(syn::Error::new(
                input.span(),
                "missing `state_fields` parameter",
            ));
        }

        let greens_specs = greens.unwrap_or_default();
        let (green_exprs, green_type_tags): (Vec<Expr>, Vec<Option<green_type_tag::GreenTypeTag>>) =
            greens_specs
                .into_iter()
                .map(|spec| (spec.expr, spec.type_tag))
                .unzip();

        Ok(JitInterpConfig {
            state_type,
            env_type,
            io_shims: io_shims.unwrap_or_default(),
            calls,
            auto_calls: auto_calls.unwrap_or(false),
            greens: green_exprs,
            reds: reds.unwrap_or_default(),
            green_type_tags,
            virtualizable_decl,
            state_fields,
        })
    }
}

fn parse_expr_list(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    bracketed!(content in input);
    let exprs: Punctuated<Expr, Token![,]> = content.parse_terminated(Expr::parse, Token![,])?;
    Ok(exprs.into_iter().collect())
}

/// Parse virtualizable_fields = { var: IDENT, token_offset: PATH, fields: { ... }, arrays: { ... } }
///
/// Parse `state_fields = { name: type, ... }` where type is `int`, `[int]`,
/// `[int; virt]`, or `opaque(TypePath)`.
fn parse_state_fields(input: ParseStream) -> syn::Result<StateFieldsConfig> {
    let content;
    braced!(content in input);
    let mut fields = Vec::new();

    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let kind = if content.peek(syn::token::Bracket) {
            // Array: [int] or virtualizable: [int; virt]
            let inner;
            bracketed!(inner in content);
            let item_type: Ident = inner.parse()?;
            if item_type != "int" {
                return Err(syn::Error::new(
                    item_type.span(),
                    format!(
                        "state_fields array `{name}` uses unsupported item type `{item_type}`; \
                         only `int` is currently supported"
                    ),
                ));
            }
            if inner.peek(Token![;]) {
                inner.parse::<Token![;]>()?;
                let flag: Ident = inner.parse()?;
                if flag == "virt" {
                    StateFieldKind::VirtArray(item_type)
                } else {
                    return Err(syn::Error::new(
                        flag.span(),
                        format!("unknown array modifier `{flag}`, expected `virt`"),
                    ));
                }
            } else {
                StateFieldKind::Array(item_type)
            }
        } else {
            // Scalar forms: `int`, `int(<TypePath>)`, or `opaque(TypePath)`.
            let head: Ident = content.parse()?;
            if head == "opaque" {
                let inner;
                syn::parenthesized!(inner in content);
                let type_path: syn::Path = inner.parse()?;
                StateFieldKind::Opaque(type_path)
            } else if head == "int" {
                // RPython parity: optional `int(<TypePath>)` declares the
                // Rust storage type when it differs from i64.
                let rust_type = if content.peek(syn::token::Paren) {
                    let inner;
                    syn::parenthesized!(inner in content);
                    Some(inner.parse::<syn::Path>()?)
                } else {
                    None
                };
                StateFieldKind::Scalar {
                    ir_type: head,
                    rust_type,
                }
            } else {
                return Err(syn::Error::new(
                    head.span(),
                    format!(
                        "state_fields scalar `{name}` uses unsupported type `{head}`; \
                         supported: `int`, `int(<TypePath>)`, `opaque(TypePath)`"
                    ),
                ));
            }
        };

        fields.push(StateFieldDecl { name, kind });
        let _ = content.parse::<Token![,]>();
    }

    Ok(StateFieldsConfig { fields })
}

/// RPython equivalent: VirtualizableInfo construction from virtualizable.py
/// + jtransform.py's field-to-descriptor mapping.
fn parse_virtualizable_decl(input: ParseStream) -> syn::Result<VirtualizableDecl> {
    let content;
    braced!(content in input);

    let mut var_name = None;
    let mut token_offset = None;
    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "var" => {
                var_name = Some(content.parse::<Ident>()?);
            }
            "token_offset" => {
                token_offset = Some(content.parse::<Path>()?);
            }
            "fields" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let field_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    fields.push(VableFieldDecl {
                        name,
                        field_type,
                        offset,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            "arrays" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let item_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let field_offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    let layout = if inner.peek(syn::token::Brace) {
                        let layout_content;
                        braced!(layout_content in inner);
                        let mut ptr_offset = None;
                        let mut length_offset = None;
                        let mut items_offset = None;
                        while !layout_content.is_empty() {
                            let layout_key: Ident = layout_content.parse()?;
                            layout_content.parse::<Token![:]>()?;
                            match layout_key.to_string().as_str() {
                                "ptr_offset" => {
                                    ptr_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "length_offset" => {
                                    length_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "items_offset" => {
                                    items_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                other => {
                                    return Err(syn::Error::new(
                                        layout_key.span(),
                                        format!(
                                            "unknown virtualizable array layout parameter: `{other}`"
                                        ),
                                    ));
                                }
                            }
                            let _ = layout_content.parse::<Token![,]>();
                        }
                        if let Some(ptr_offset) = ptr_offset {
                            VableArrayLayoutDecl::Embedded {
                                field_offset,
                                ptr_offset,
                                length_offset: length_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `length_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                                items_offset: items_offset.ok_or_else(|| {
                                    syn::Error::new(
                                        inner.span(),
                                        "missing `items_offset` in embedded virtualizable array layout",
                                    )
                                })?,
                            }
                        } else {
                            VableArrayLayoutDecl::Direct {
                                field_offset,
                                length_offset,
                                items_offset,
                            }
                        }
                    } else {
                        VableArrayLayoutDecl::Direct {
                            field_offset,
                            length_offset: None,
                            items_offset: None,
                        }
                    };
                    arrays.push(VableArrayDecl {
                        name,
                        item_type,
                        layout,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown virtualizable_fields parameter: `{other}`"),
                ));
            }
        }
        let _ = content.parse::<Token![,]>();
    }

    let var_name = var_name
        .ok_or_else(|| syn::Error::new(content.span(), "missing `var` in virtualizable_fields"))?;
    let token_offset = token_offset.ok_or_else(|| {
        syn::Error::new(
            content.span(),
            "missing `token_offset` in virtualizable_fields",
        )
    })?;

    Ok(VirtualizableDecl {
        var_name,
        token_offset,
        fields,
        arrays,
    })
}

/// Parse `{ path::func => jit_func, ... }`.
fn parse_io_shim_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let shim: Ident = content.parse()?;
        map.push((func, shim));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `{ path::func, path::func => residual_int, ... }`.
///
/// Per-entry forms:
///   - `path::func`                (default policy)
///   - `path::func => policy_kind` (explicit policy)
fn parse_call_map(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        let policy = if content.peek(Token![=>]) {
            content.parse::<Token![=>]>()?;
            let kind: Ident = content.parse()?;
            Some(parse_call_policy_kind(&kind).ok_or_else(|| {
                syn::Error::new(
                    kind.span(),
                    "call policy must be a supported residual/may_force/release_gil/loopinvariant policy or inline_int/inline_ref/inline_float",
                )
            })?)
        } else {
            None
        };
        map.push(CallEntry { path: func, policy });
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `[func_a, func_b, func_c]` — shorthand helper list with auto-inferred policies.
fn parse_helpers_list(input: ParseStream) -> syn::Result<Vec<CallEntry>> {
    let content;
    bracketed!(content in input);
    let paths: Punctuated<Path, Token![,]> = content.parse_terminated(Path::parse, Token![,])?;
    Ok(paths
        .into_iter()
        .map(|p| CallEntry {
            path: p,
            policy: None,
        })
        .collect())
}

/// Main entry point: transform the function with JIT support.
pub fn transform_jit_interp(config: JitInterpConfig, func: ItemFn) -> TokenStream {
    let trace_fn = codegen_trace::generate_trace_fn(&config, &func);
    let state_impl = codegen_state::generate_jit_state(&config, &func);
    let merge_wrapper = generate_merge_wrapper(&config, &func);
    let transformed_fn = transform_function(&config, &func);

    quote! {
        #state_impl
        #trace_fn
        #merge_wrapper
        #transformed_fn
    }
}

/// Generate a `#[cold]` out-of-line wrapper for the merge_point call.
///
/// This keeps the mainloop hot path thin — only an `is_tracing()` flag check
/// appears inline, while the closure capture and tracing logic live here.
fn generate_merge_wrapper(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    quote! {
        #[cold]
        #[inline(never)]
        #[allow(non_snake_case)]
        fn #merge_fn_name(
            __driver: &mut majit_metainterp::JitDriver<#state_type>,
            __env: &#env_type,
            __pc: usize,
        ) {
            // Phase 4 Epic B.3-B.4: clone the driver-shared `Assembler`
            // Arc before the `merge_point` mutable borrow so the trace
            // closure can forward it into `#trace_fn_name`.  The lock is
            // acquired *inside* `#trace_fn_name`, scoped tightly around
            // the `JitCode` build only — `trace_jitcode_observer` runs
            // outside the lock to mirror RPython's `pyjitpl.py:2255
            // finish_setup` ordering (assembler is frozen before any
            // tracing observation runs) and to avoid a deadlock if a
            // recursive portal/residual callback re-enters this trace
            // path on the same driver thread.
            let __shared_asm = __driver.shared_asm();
            // Slice 2.2: clone the dispatch JitCode Arc before the mutable
            // `merge_point` borrow so the closure can forward it to
            // `#trace_fn_name` without holding a `JitDriver` reference.
            let __dispatch_jitcode: Option<::std::sync::Arc<majit_metainterp::JitCode>> =
                __driver.dispatch_jitcode().cloned();
            __driver.merge_point(|__meta, __sym| {
                use majit_metainterp::JitCodeSym;
                let __ctx = __meta
                    .trace_ctx()
                    .expect("merge_point invariant: tracing must be Some");
                if __sym.trace_started && __pc == __sym.loop_header_pc() {
                    return majit_metainterp::TraceAction::CloseLoop;
                }
                let __result = #trace_fn_name(
                    &__shared_asm,
                    __ctx,
                    __sym,
                    __env,
                    __pc,
                    __dispatch_jitcode.as_ref(),
                );
                __sym.trace_started = true;
                // pyjitpl.py:2843 blackhole_if_trace_too_long — check
                // AFTER executing the step (RPython _interpret loop order).
                if __ctx.is_too_long() {
                    if majit_metainterp::majit_log_enabled() {
                        eprintln!("[jit] trace too long, aborting");
                    }
                    return majit_metainterp::TraceAction::Abort;
                }
                __result
            });
        }
    }
}

/// Transform the original function: replace jit_merge_point!() and can_enter_jit!() markers.
fn transform_function(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);

    // Rewrite the function body, replacing marker macros
    let body = rewrite_body(
        &func.block,
        &merge_fn_name,
        &config.greens,
        &config.green_type_tags,
        &config.calls,
        &config.io_shims,
    );

    quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    }
}

/// Rewrite function body: replace jit_merge_point!() and can_enter_jit!() calls.
fn rewrite_body(
    block: &syn::Block,
    merge_fn_name: &Ident,
    default_greens: &[Expr],
    default_green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    call_policies: &[CallEntry],
    io_shims: &[(Path, Ident)],
) -> TokenStream {
    use syn::visit_mut::VisitMut;

    #[derive(Default, Clone)]
    struct MergePointArgs {
        driver: Option<Expr>,
        env: Option<Expr>,
        pc: Option<Expr>,
    }

    impl Parse for MergePointArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.is_empty() {
                return Ok(Self::default());
            }
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pc: Expr = input.parse()?;
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let _: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
            }
            Ok(Self {
                driver: Some(driver),
                env: Some(env),
                pc: Some(pc),
            })
        }
    }

    struct CanEnterJitArgs {
        driver: Expr,
        target: Expr,
        state: Expr,
        env: Expr,
        pre_run: Expr,
        pc: Option<Expr>,
        stacksize: Option<Expr>,
        greens: Vec<Expr>,
    }

    impl Parse for CanEnterJitArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let target: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let state: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pre_run: Expr = input.parse()?;

            let mut pc = None;
            let mut stacksize = None;
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                pc = Some(input.parse::<Expr>()?);
                input.parse::<Token![,]>()?;
                stacksize = Some(input.parse::<Expr>()?);
            }

            let mut greens = Vec::new();
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let exprs: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                greens = exprs.into_iter().collect();
            }

            Ok(Self {
                driver,
                target,
                state,
                env,
                pre_run,
                pc,
                stacksize,
                greens,
            })
        }
    }

    /// Slice 92.2 — emit a single `(i64, majit_ir::GreenType)` pair for one
    /// green expression, dispatching on the optional per-green type tag.
    /// Untagged (`None`) greens flow through the
    /// `<_ as GreenAsI64>::__green_repr` trait — preserving the
    /// pre-Slice-92 behavior. Tagged greens emit explicit casts so
    /// `&str`-bearing greens carry `GreenType::Str` instead of being
    /// silently routed through the blanket `impl<T: ?Sized>` Ref impl.
    fn emit_green_repr(spec_expr: &Expr, tag: Option<green_type_tag::GreenTypeTag>) -> TokenStream {
        use green_type_tag::GreenTypeTag;
        match tag {
            // Untagged greens go through the `GreenAsI64` trait —
            // primitive types route to `GreenType::Int`, floats to
            // `GreenType::Float`, references to `GreenType::Ref`.  An
            // untagged `&str` lands on the blanket `impl<T: ?Sized>
            // GreenAsI64 for &T` and carries `GreenType::Ref` (raw
            // pointer identity).  Consumers wanting RPython STR /
            // UNICODE content semantics MUST tag explicitly via
            // `greens = [name: str]` / `greens = [name: unicode]`;
            // `&str` is ambiguous between `rstr.STR` (UTF-8 byte
            // string) and `rstr.UNICODE` (codepoint sequence) and the
            // macro cannot pick one without an explicit declaration.
            None => quote! {
                <_ as majit_ir::GreenAsI64>::__green_repr(#spec_expr)
            },
            Some(GreenTypeTag::Int) => quote! {
                ((#spec_expr) as i64, majit_ir::GreenType::Int)
            },
            Some(GreenTypeTag::Float) => quote! {
                {
                    let __green_f: f64 = (#spec_expr) as f64;
                    (__green_f.to_bits() as i64, majit_ir::GreenType::Float)
                }
            },
            Some(GreenTypeTag::Ref) => quote! {
                ((#spec_expr) as *const _ as *const () as usize as i64,
                 majit_ir::GreenType::Ref)
            },
            // ABI: the i64 is the address of a `'static` slot holding
            // a `&'static str`.  `majit_ir::value::default_str_eq` /
            // `default_str_hash` / `default_unicode_hash` dereference
            // the i64 as `*const &'static str` and read the fat
            // pointer (data + len) — RPython's `rstr.STR*` /
            // `rstr.UNICODE*` carry their length internally; pyre
            // mirrors that contract by storing the fat pointer at a
            // stable address rather than the bare data pointer.
            //
            // PRE-EXISTING-ADAPTATION (allocation lifetime divergence
            // from RPython, intentionally deferred):
            //
            //   * RPython: `rstr.STR*` is GC-allocated *once per
            //     JitCell* and naturally stable for the JitCell's
            //     lifetime.  `JitCell.greenargs[i]` holds the rstr
            //     pointer; subsequent merge-point hits for the same
            //     `(jitdriver, greenkey)` pass the existing pointer
            //     into `equal_whatever` / `hash_whatever`.  No
            //     re-allocation, GC frees the rstr when the cell
            //     dies.
            //
            //   * pyre: `&str` has no stable backing-storage address
            //     by default, so this macro emits a fresh slot via
            //     `Box::leak` *every merge-point hit* — not once per
            //     JitCell.  The GreenKey HashMap content-de-dupes via
            //     `default_str_eq` / `default_str_hash`, so multiple
            //     slots with the same content collapse to a single
            //     cache entry, but the slot leaks themselves grow
            //     unboundedly with merge-point hit count for
            //     long-running programs.
            //
            // A structural fix (per-JitCell owned-string field
            // instead of leaked slot, e.g. reshaping
            // `GreenKey::values` from `Vec<i64>` to a typed enum
            // carrying `Box<str>` for str/unicode greens, with the
            // macro emitting a temporary the cache promotes on
            // insertion) is a multi-session refactor and is
            // intentionally deferred — a global intern side-table
            // was rejected as non-orthodox (RPython does not
            // maintain one).  Functional behavior matches RPython
            // (content-keyed compare/hash); only the lifetime /
            // allocation profile differs.
            Some(GreenTypeTag::Str) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Str,
                )
            },
            Some(GreenTypeTag::Unicode) => quote! {
                (
                    majit_ir::make_str_slot(#spec_expr),
                    majit_ir::GreenType::Unicode,
                )
            },
        }
    }

    fn green_key_expr(
        target: &Expr,
        greens: &[Expr],
        green_type_tags: &[Option<green_type_tag::GreenTypeTag>],
    ) -> Option<TokenStream> {
        if greens.is_empty() {
            None
        } else {
            // Per-green dispatch through `majit_ir::GreenAsI64::__green_repr`
            // so the `(i64-bits, GreenType)` pair travels together for each
            // green expression. `with_types` builds the typed schema;
            // `warmstate.py:575 _green_args_spec` keys per-type
            // `equal_whatever`/`hash_whatever` off the green's lltype, so a
            // Ref-typed green must compare by pointer identity, a Float by
            // bit pattern, and an Int by raw value — collapsing all to
            // `GreenType::Int` (the previous `GreenKey::new` shape) made
            // Float / Ref greens equal under bit-equal Int values they
            // should not be equal to.
            //
            // Slice 92.2 — `green_type_tags` is the lockstep
            // `Vec<Option<GreenTypeTag>>` carried alongside `greens`
            // (`JitInterpConfig.green_type_tags`).  Tagged greens
            // bypass the trait dispatch with explicit casts so str /
            // unicode greens carry `GreenType::Str` (warmstate.py:108-128
            // ll_streq / ll_strhash routing).  Untagged greens
            // (`None`) keep the trait path unchanged.
            let green_reprs: Vec<TokenStream> = greens
                .iter()
                .enumerate()
                .map(|(i, expr)| {
                    let tag = green_type_tags.get(i).copied().flatten();
                    emit_green_repr(expr, tag)
                })
                .collect();
            Some(quote! {
                {
                    let (__values, __types): (Vec<i64>, Vec<majit_ir::GreenType>) = vec![
                        <_ as majit_ir::GreenAsI64>::__green_repr(#target),
                        #(#green_reprs),*
                    ].into_iter().unzip();
                    majit_ir::GreenKey::with_types(__values, __types)
                }
            })
        }
    }

    #[derive(Clone)]
    enum ObserverReplayKind {
        Void,
        Int,
        Ref,
        Float,
    }

    #[derive(Clone)]
    struct ObserverReplay {
        kind: ObserverReplayKind,
        observed_func: TokenStream,
        observed_arg_indices: Vec<usize>,
        unwrap_observed_refs: bool,
    }

    fn path_segments(path: &Path) -> Vec<String> {
        path.segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect()
    }

    fn call_expr_segments(expr: &Expr) -> Option<Vec<String>> {
        match expr {
            Expr::Path(expr_path) => Some(path_segments(&expr_path.path)),
            _ => None,
        }
    }

    fn unwrap_observer_ref_expr(expr: &Expr) -> &Expr {
        match expr {
            Expr::Reference(reference) => unwrap_observer_ref_expr(&reference.expr),
            Expr::Paren(paren) => unwrap_observer_ref_expr(&paren.expr),
            _ => expr,
        }
    }

    fn replay_kind_for_policy(kind: CallPolicyKind) -> Option<ObserverReplayKind> {
        // Mirror the metainterp's recording sites in
        // `pyjitpl/dispatch.rs::run_one_step`.  Plain (non-wrapped)
        // policies use the helper symbol directly; wrapped policies
        // route through `helper_policy_path` to recover the
        // `__concrete_target` wrapper symbol the metainterp records,
        // wired by `observer_replay_for_call` below.
        //
        // Elidable: not recorded by dispatch (CALL_PURE_* is exempt),
        // pure re-execution is harmless. Inline: pushes a metainterp
        // frame, never reaches call_*_function.
        match kind {
            CallPolicyKind::ResidualVoid
            | CallPolicyKind::ResidualVoidCannotRaise
            | CallPolicyKind::MayForceVoid
            | CallPolicyKind::ReleaseGilVoid
            | CallPolicyKind::LoopInvariantVoid
            | CallPolicyKind::ResidualVoidWrapped
            | CallPolicyKind::ResidualVoidCannotRaiseWrapped
            | CallPolicyKind::MayForceVoidWrapped
            | CallPolicyKind::ReleaseGilVoidWrapped
            | CallPolicyKind::LoopInvariantVoidWrapped => Some(ObserverReplayKind::Void),
            CallPolicyKind::ResidualInt
            | CallPolicyKind::ResidualIntCannotRaise
            | CallPolicyKind::MayForceInt
            | CallPolicyKind::ReleaseGilInt
            | CallPolicyKind::LoopInvariantInt
            | CallPolicyKind::ResidualIntWrapped
            | CallPolicyKind::ResidualIntCannotRaiseWrapped
            | CallPolicyKind::MayForceIntWrapped
            | CallPolicyKind::ReleaseGilIntWrapped
            | CallPolicyKind::LoopInvariantIntWrapped => Some(ObserverReplayKind::Int),
            CallPolicyKind::ResidualRefWrapped
            | CallPolicyKind::ResidualRefCannotRaiseWrapped
            | CallPolicyKind::MayForceRefWrapped
            | CallPolicyKind::LoopInvariantRefWrapped => Some(ObserverReplayKind::Ref),
            CallPolicyKind::ResidualFloatWrapped
            | CallPolicyKind::ResidualFloatCannotRaiseWrapped
            | CallPolicyKind::MayForceFloatWrapped
            | CallPolicyKind::ReleaseGilFloatWrapped
            | CallPolicyKind::LoopInvariantFloatWrapped => Some(ObserverReplayKind::Float),
            _ => None,
        }
    }

    /// Wrapped policy variants install a generated wrapper at codewriter
    /// time (`__concrete_target` from `(__policy, _, __trace_target,
    /// __concrete_target)` tuple).  The metainterp records the wrapper
    /// pointer on OBSERVED_CALLS, so the outer-side consume must use the
    /// same wrapper symbol — recovered at runtime by calling the helper's
    /// `__majit_call_policy_<name>()` accessor.  Plain policies pass
    /// through `#func as *const ()` directly.
    fn is_wrapped_policy(kind: CallPolicyKind) -> bool {
        matches!(
            kind,
            CallPolicyKind::ResidualVoidWrapped
                | CallPolicyKind::ResidualVoidCannotRaiseWrapped
                | CallPolicyKind::MayForceVoidWrapped
                | CallPolicyKind::ReleaseGilVoidWrapped
                | CallPolicyKind::LoopInvariantVoidWrapped
                | CallPolicyKind::ResidualIntWrapped
                | CallPolicyKind::ResidualIntCannotRaiseWrapped
                | CallPolicyKind::MayForceIntWrapped
                | CallPolicyKind::ReleaseGilIntWrapped
                | CallPolicyKind::LoopInvariantIntWrapped
                | CallPolicyKind::ResidualRefWrapped
                | CallPolicyKind::ResidualRefCannotRaiseWrapped
                | CallPolicyKind::MayForceRefWrapped
                | CallPolicyKind::LoopInvariantRefWrapped
                | CallPolicyKind::ResidualFloatWrapped
                | CallPolicyKind::ResidualFloatCannotRaiseWrapped
                | CallPolicyKind::MayForceFloatWrapped
                | CallPolicyKind::ReleaseGilFloatWrapped
                | CallPolicyKind::LoopInvariantFloatWrapped
        )
    }

    fn observer_replay_for_call(
        func: &Expr,
        call_policies: &[(Vec<String>, Option<CallPolicyKind>)],
        io_shims: &[(Vec<String>, Ident)],
    ) -> Option<ObserverReplay> {
        let segments = call_expr_segments(func)?;
        for (io_path, shim) in io_shims {
            if *io_path == segments {
                let observed_func = quote! { #shim as *const () };
                return Some(ObserverReplay {
                    kind: ObserverReplayKind::Void,
                    observed_func,
                    observed_arg_indices: vec![0],
                    unwrap_observed_refs: true,
                });
            }
        }
        for (path, policy) in call_policies {
            if *path == segments {
                let policy_kind = *policy.as_ref()?;
                let kind = replay_kind_for_policy(policy_kind)?;
                let observed_func = if is_wrapped_policy(policy_kind) {
                    // Wrapped policy: outer-side replay key must match the
                    // wrapper symbol (`__concrete_target`) the metainterp
                    // recorded — recover it via the helper's policy
                    // accessor at runtime.
                    let policy_path = jitcode_lower::helper_policy_path(func)?;
                    quote! {
                        {
                            let (_, _, __majit_observer_trace, __majit_observer_concrete, _) = #policy_path();
                            if __majit_observer_trace.is_null()
                                && __majit_observer_concrete.is_null()
                            {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __majit_observer_trace = if __majit_observer_trace.is_null() {
                                __majit_observer_concrete
                            } else {
                                __majit_observer_trace
                            };
                            let __majit_observer_concrete = if __majit_observer_concrete.is_null() {
                                __majit_observer_trace
                            } else {
                                __majit_observer_concrete
                            };
                            __majit_observer_concrete
                        }
                    }
                } else {
                    quote! { #func as *const () }
                };
                return Some(ObserverReplay {
                    kind,
                    observed_func,
                    observed_arg_indices: Vec::new(),
                    unwrap_observed_refs: false,
                });
            }
        }
        None
    }

    fn observer_replay_expr(call: &syn::ExprCall, replay: ObserverReplay) -> Expr {
        let func = &call.func;
        let arg_exprs: Vec<Expr> = call.args.iter().cloned().collect();
        let arg_names: Vec<Ident> = (0..arg_exprs.len())
            .map(|idx| format_ident!("__majit_observer_arg{idx}"))
            .collect();
        let arg_binds = arg_names.iter().zip(arg_exprs.iter()).map(|(name, arg)| {
            quote! {
                let #name = #arg;
            }
        });
        let observed_indices: Vec<usize> = if replay.observed_arg_indices.is_empty() {
            (0..arg_names.len()).collect()
        } else {
            replay.observed_arg_indices.clone()
        };
        let observed_args = observed_indices.iter().map(|idx| {
            if replay.unwrap_observed_refs {
                let expr = unwrap_observer_ref_expr(&arg_exprs[*idx]);
                quote! { majit_metainterp::observer_arg_to_i64(&(#expr)) }
            } else {
                let name = &arg_names[*idx];
                quote! { majit_metainterp::observer_arg_to_i64(&#name) }
            }
        });
        let observed_func = replay.observed_func;

        let tokens = match replay.kind {
            ObserverReplayKind::Void => quote! {{
                #(#arg_binds)*
                let __majit_observer_args = [#(#observed_args),*];
                if !majit_metainterp::consume_observed_void_call(
                    #observed_func,
                    &__majit_observer_args,
                ) {
                    #func(#(#arg_names),*);
                }
            }},
            ObserverReplayKind::Int => quote! {{
                #(#arg_binds)*
                let __majit_observer_args = [#(#observed_args),*];
                match majit_metainterp::consume_observed_int_call(
                    #observed_func,
                    &__majit_observer_args,
                ) {
                    Some(__majit_observer_result) => unsafe {
                        majit_metainterp::observer_i64_to_value(__majit_observer_result)
                    },
                    None => #func(#(#arg_names),*),
                }
            }},
            ObserverReplayKind::Ref => quote! {{
                #(#arg_binds)*
                let __majit_observer_args = [#(#observed_args),*];
                match majit_metainterp::consume_observed_ref_call(
                    #observed_func,
                    &__majit_observer_args,
                ) {
                    Some(__majit_observer_result) => unsafe {
                        majit_metainterp::observer_i64_to_value(__majit_observer_result)
                    },
                    None => #func(#(#arg_names),*),
                }
            }},
            ObserverReplayKind::Float => quote! {{
                #(#arg_binds)*
                let __majit_observer_args = [#(#observed_args),*];
                match majit_metainterp::consume_observed_float_call(
                    #observed_func,
                    &__majit_observer_args,
                ) {
                    Some(__majit_observer_result) => unsafe {
                        majit_metainterp::observer_i64_to_value(__majit_observer_result)
                    },
                    None => #func(#(#arg_names),*),
                }
            }},
        };
        syn::parse2(tokens).expect("failed to parse observer replay expression")
    }

    struct MarkerRewriter {
        merge_fn_name: Ident,
        default_greens: Vec<Expr>,
        default_green_type_tags: Vec<Option<green_type_tag::GreenTypeTag>>,
        call_policies: Vec<(Vec<String>, Option<CallPolicyKind>)>,
        io_shims: Vec<(Vec<String>, Ident)>,
    }

    impl VisitMut for MarkerRewriter {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // First recurse into children
            syn::visit_mut::visit_stmt_mut(self, stmt);

            // Check if this statement is a macro invocation
            if let syn::Stmt::Macro(stmt_macro) = stmt {
                let mac = &stmt_macro.mac;
                let path_str = mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");

                if path_str == "jit_merge_point" || path_str.ends_with("::jit_merge_point") {
                    let args =
                        syn::parse2::<MergePointArgs>(mac.tokens.clone()).unwrap_or_default();
                    let merge_fn = &self.merge_fn_name;
                    let driver = args.driver.unwrap_or_else(|| syn::parse_quote!(driver));
                    let env = args.env.unwrap_or_else(|| syn::parse_quote!(program));
                    let pc = args.pc.unwrap_or_else(|| syn::parse_quote!(pc));
                    // Slice 2.3: jit_merge_point!() in #[jit_interp] dispatch portals
                    // expands to a single merge_wrapper invocation.  The wrapper
                    // (generate_merge_wrapper) clones the dispatch JitCode Arc and calls
                    // `driver.merge_point(...)` exactly once — there is no additional
                    // outer merge-point hook.  The BC_JIT_MERGE_POINT(_C) IR op lives
                    // inside the dispatch JitCode body (lower_dispatch_body, Slice 1.2),
                    // not in the outer Rust source.
                    //
                    // RPython parity: source-level jit_merge_point() is a codewriter
                    // marker; the runtime hook is the JitCode IR op
                    // (interp_jit.py:88-90).
                    //
                    // The `is_tracing()` guard here is a hot-path short-circuit
                    // (avoids the cold `__merge_*` call when not tracing).  It does NOT
                    // add a second merge-point dispatch — `driver.merge_point` guards
                    // again internally, but the closure runs only once.
                    //
                    // Slice 5 will simplify further once the legacy factory is deleted.
                    let new_tokens: TokenStream = quote! {
                        if #driver.is_tracing() {
                            #merge_fn(&mut #driver, #env, #pc);
                        }
                    };
                    *stmt =
                        syn::parse2(new_tokens).expect("failed to parse merge_point replacement");
                }
            }
        }

        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            // First recurse
            syn::visit_mut::visit_expr_mut(self, expr);

            if let Expr::Call(call) = expr {
                if let Some(replay) =
                    observer_replay_for_call(&call.func, &self.call_policies, &self.io_shims)
                {
                    *expr = observer_replay_expr(call, replay);
                }
            }
        }

        fn visit_block_mut(&mut self, block: &mut syn::Block) {
            let mut new_stmts = Vec::new();
            let mut i = 0;
            while i < block.stmts.len() {
                let stmt = &block.stmts[i];

                // Check if this is can_enter_jit!(driver, target, state, env, pre_run, ...)
                if let syn::Stmt::Macro(stmt_macro) = stmt {
                    let mac = &stmt_macro.mac;
                    let path_str = mac
                        .path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");

                    if path_str == "can_enter_jit" || path_str.ends_with("::can_enter_jit") {
                        let tokens = &mac.tokens;
                        if let Ok(args) = syn::parse2::<CanEnterJitArgs>(tokens.clone()) {
                            let driver_expr = &args.driver;
                            let target_expr = &args.target;
                            let state_expr = &args.state;
                            let env_expr = &args.env;
                            let pre_run_expr = &args.pre_run;
                            let pc_expr = args
                                .pc
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(pc));
                            let stacksize_expr = args
                                .stacksize
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(stacksize));
                            let (greens, green_type_tags) = if args.greens.is_empty() {
                                (
                                    self.default_greens.clone(),
                                    self.default_green_type_tags.clone(),
                                )
                            } else {
                                // RPython `can_enter_jit` and
                                // `jit_merge_point` reference the same
                                // `JitDriver` greens spec — the marker op
                                // args sit in fixed positional slots whose
                                // lltype is fixed by declaration
                                // (`warmstate.py:564 _green_args_spec`,
                                // `support.py:126 decode_hp_hint_args`
                                // asserts on count mismatch at translation
                                // time).  Positional inheritance of
                                // declaration tags therefore matches
                                // RPython parity: a `str`-tagged green
                                // stays Str-keyed through the override
                                // path, routing through the canonical
                                // slot ABI rather than silently falling
                                // to the blanket `GreenAsI64 for &T`
                                // (Ref / pointer identity) implementation.
                                //
                                // If a downstream override expression has
                                // a different Rust type than the
                                // declaration's tag indicates (e.g. tag
                                // `Int` with `&str` override), the
                                // emitted explicit cast (`(<expr>) as
                                // i64`) fails at compile time — fail-loud
                                // at the macro / type-check boundary
                                // rather than at runtime with a
                                // misshaped key.
                                //
                                // Marker arity is structurally fixed in
                                // RPython, so a count mismatch is a
                                // hard user error. pyre fails loud at
                                // proc-macro expansion (compile-time)
                                // with a clear message rather than
                                // silently falling back to an untagged
                                // trait path that would emit a misshaped
                                // key schema.
                                if args.greens.len() != self.default_green_type_tags.len() {
                                    panic!(
                                        "can_enter_jit! override green count {} does not match \
                                         the JitDriver declaration's green count {} — RPython \
                                         marker arity is fixed (rpython/jit/codewriter/support.py \
                                         decode_hp_hint_args asserts on mismatch). Fix the \
                                         override expression list to match the declaration's \
                                         `greens=[...]` arity.",
                                        args.greens.len(),
                                        self.default_green_type_tags.len(),
                                    );
                                }
                                (args.greens.clone(), self.default_green_type_tags.clone())
                            };
                            let stacksize_update: TokenStream = quote! { #stacksize_expr = 0i32; };
                            // compile.py:711 parity: back_edge returns
                            // Some(resume_pc) on guard failure (blackhole
                            // resume) or FINISH (loop header re-entry).
                            let back_edge: TokenStream = if let Some(green_key) =
                                green_key_expr(target_expr, &greens, &green_type_tags)
                            {
                                quote! {
                                    if let Some(__resume_pc) = #driver_expr.back_edge_structured(#green_key, #target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = __resume_pc;
                                        #stacksize_update
                                        continue;
                                    }
                                }
                            } else {
                                quote! {
                                    if let Some(__resume_pc) = #driver_expr.back_edge(#target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = __resume_pc;
                                        #stacksize_update
                                        continue;
                                    }
                                }
                            };
                            let parsed: syn::Stmt =
                                syn::parse2(back_edge).expect("failed to parse back_edge");
                            new_stmts.push(parsed);
                            i += 1;
                            continue;
                        }
                    }
                }

                let mut cloned = block.stmts[i].clone();
                self.visit_stmt_mut(&mut cloned);
                new_stmts.push(cloned);
                i += 1;
            }
            block.stmts = new_stmts;
        }
    }

    let mut cloned_block = block.clone();
    let mut rewriter = MarkerRewriter {
        merge_fn_name: merge_fn_name.clone(),
        default_greens: default_greens.to_vec(),
        default_green_type_tags: default_green_type_tags.to_vec(),
        call_policies: call_policies
            .iter()
            .map(|entry| (path_segments(&entry.path), entry.policy))
            .collect(),
        io_shims: io_shims
            .iter()
            .map(|(path, shim)| (path_segments(path), shim.clone()))
            .collect(),
    };
    rewriter.visit_block_mut(&mut cloned_block);

    let stmts = &cloned_block.stmts;
    quote! { #(#stmts)* }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn parse_helpers_list_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [helper_add, helper_sub, helper_mul]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "helper_add"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "helper_sub"
        );
        assert_eq!(
            result[2].path.segments.last().unwrap().ident.to_string(),
            "helper_mul"
        );
    }

    #[test]
    fn parse_helpers_list_empty() {
        let tokens: proc_macro2::TokenStream = parse_quote! { [] };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert!(result.is_empty());
    }

    #[test]
    fn parse_helpers_list_with_path() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [module::helper_a, helper_b]
        };
        let result: Vec<CallEntry> = syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        // First has two path segments
        assert_eq!(result[0].path.segments.len(), 2);
        assert_eq!(result[0].path.segments[0].ident.to_string(), "module");
        assert_eq!(result[0].path.segments[1].ident.to_string(), "helper_a");
    }

    /// Wrapper to make `parse_helpers_list` testable via `syn::parse2`.
    struct HelpersListWrapper(Vec<CallEntry>);
    impl Parse for HelpersListWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_helpers_list(input)?))
        }
    }

    /// Wrapper to make `parse_call_map` testable via `syn::parse2`.
    struct CallMapWrapper(Vec<CallEntry>);
    impl std::fmt::Debug for CallMapWrapper {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_list()
                .entries(self.0.iter().map(|e| {
                    e.path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())
                        .unwrap_or_default()
                }))
                .finish()
        }
    }
    impl Parse for CallMapWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_call_map(input)?))
        }
    }

    #[test]
    fn parse_call_map_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { foo, bar => may_force_int }
        };
        let result: Vec<CallEntry> = syn::parse2::<CallMapWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].path.segments.last().unwrap().ident.to_string(),
            "foo"
        );
        assert!(result[0].policy.is_none());
        assert_eq!(
            result[1].path.segments.last().unwrap().ident.to_string(),
            "bar"
        );
        assert_eq!(result[1].policy, Some(CallPolicyKind::MayForceInt));
    }

    #[test]
    fn parse_call_map_rejects_slot_effect_blocks() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { pop_only { reads: [stackpos] } }
        };
        let err = syn::parse2::<CallMapWrapper>(tokens).unwrap_err();
        assert!(
            err.to_string().contains("expected identifier")
                || err.to_string().contains("expected `,`"),
            "expected call-map parser to reject slot-effect block, got: {err}",
        );
    }

    struct VirtualizableWrapper(VirtualizableDecl);
    impl Parse for VirtualizableWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_virtualizable_decl(input)?))
        }
    }

    struct StateFieldsWrapper(StateFieldsConfig);
    impl Parse for StateFieldsWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_state_fields(input)?))
        }
    }

    #[test]
    fn parse_state_fields_accepts_opaque_pass_through() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                storage: opaque(aheui_runtime::storage::Storage),
                selected: int,
                tape: [int; virt],
            }
        };
        let parsed = syn::parse2::<StateFieldsWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.fields.len(), 3);
        assert_eq!(parsed.fields[0].name.to_string(), "storage");
        match &parsed.fields[0].kind {
            StateFieldKind::Opaque(p) => {
                assert_eq!(p.segments.last().unwrap().ident.to_string(), "Storage");
            }
            _ => panic!("expected Opaque variant"),
        }
        assert!(matches!(
            parsed.fields[1].kind,
            StateFieldKind::Scalar { .. }
        ));
        assert!(matches!(
            parsed.fields[2].kind,
            StateFieldKind::VirtArray(_)
        ));
    }

    #[test]
    fn parse_state_fields_rejects_unknown_scalar_type() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            { val: float }
        };
        let err = match syn::parse2::<StateFieldsWrapper>(tokens) {
            Ok(_) => panic!("expected parse error for unknown scalar type"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("supported: `int`, `int(<TypePath>)`, `opaque(TypePath)`"),
            "msg: {msg}"
        );
    }

    #[test]
    fn parse_virtualizable_decl_keeps_direct_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: { next_instr: int @ NEXT_INSTR_OFFSET },
                arrays: { locals_w: ref @ LOCALS_OFFSET },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Direct {
                field_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(quote::quote!(#field_offset).to_string(), "LOCALS_OFFSET");
                assert!(length_offset.is_none());
                assert!(items_offset.is_none());
            }
            VableArrayLayoutDecl::Embedded { .. } => {
                panic!("expected direct array layout");
            }
        }
    }

    #[test]
    fn parse_virtualizable_decl_supports_embedded_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: {},
                arrays: {
                    stack: int @ (STORAGEPOOL_DATA_PTRS_OFFSET + SLOT) {
                        ptr_offset: 0,
                        length_offset: STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET,
                        items_offset: 0,
                    },
                },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Embedded {
                field_offset,
                ptr_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(
                    quote::quote!(#field_offset).to_string(),
                    "STORAGEPOOL_DATA_PTRS_OFFSET + SLOT"
                );
                assert_eq!(quote::quote!(#ptr_offset).to_string(), "0");
                assert_eq!(
                    quote::quote!(#length_offset).to_string(),
                    "STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET"
                );
                assert_eq!(quote::quote!(#items_offset).to_string(), "0");
            }
            VableArrayLayoutDecl::Direct { .. } => {
                panic!("expected embedded array layout");
            }
        }
    }
}
