/// Proc macros for the majit JIT framework.
///
/// rpython/rlib/jit.py decorator equivalents:
/// - #[elidable]: rlib/jit.py:13 — Mark a function as pure (constant-foldable)
/// - #[elidable_promote]: rlib/jit.py:180 — Elidable + auto-promote args
/// - #[dont_look_inside]: rlib/jit.py:132 — Prevent tracing into a function
/// - #[unroll_safe]: rlib/jit.py:150 — Safe to unroll loops
/// - #[loop_invariant]: rlib/jit.py:161 — Loop-invariant function
/// - #[not_in_trace]: rlib/jit.py:260 — Disappears from final assembler
///
/// majit-specific extensions:
/// - #[jit_driver]: Annotate an interpreter's main dispatch loop
/// - #[jit_interp]: Auto-generate trace_instruction and JitState from dispatch
/// - #[jit_inline]: Serialize a helper into a hidden sub-JitCode
/// - #[jit_may_force]: Mark a helper as a may-force call surface
/// - #[jit_release_gil]: Mark a helper as a release-GIL call surface
/// - #[jit_loop_invariant]: Alias for #[loop_invariant]
/// - #[jit_module]: Module-level automatic helper discovery
/// - virtualizable!: Standalone virtualizable field declaration
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    FnArg, Ident, ItemFn, Path, ReturnType, Token, Type, parse::Parse, parse::ParseStream,
    parse_macro_input,
};

mod jit_interp;
mod jit_struct;
mod virtualizable;

struct JitInlineArgs {
    calls: Vec<jit_interp::CallEntry>,
}

impl Parse for JitInlineArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut calls: Vec<jit_interp::CallEntry> = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "calls" => {
                    let content;
                    syn::braced!(content in input);
                    while !content.is_empty() {
                        let func: Path = content.parse()?;
                        let policy = if content.peek(Token![=>]) {
                            content.parse::<Token![=>]>()?;
                            let kind: Ident = content.parse()?;
                            Some(jit_interp::parse_call_policy_kind(&kind).ok_or_else(|| {
                                syn::Error::new(
                                    kind.span(),
                                    "#[jit_inline(calls = { ... })] supports residual/may_force/release_gil/loopinvariant call policies for void/int and wrapped int/ref/float helpers, plus inline_int/inline_ref/inline_float",
                                )
                            })?)
                        } else {
                            None
                        };
                        calls.push(jit_interp::CallEntry { path: func, policy });
                        let _ = content.parse::<Token![,]>();
                    }
                }
                "helpers" => {
                    let content;
                    syn::bracketed!(content in input);
                    let paths: syn::punctuated::Punctuated<Path, Token![,]> =
                        content.parse_terminated(Path::parse, Token![,])?;
                    calls.extend(paths.into_iter().map(|p| jit_interp::CallEntry {
                        path: p,
                        policy: None,
                    }));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_inline parameter: `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }
        Ok(Self { calls })
    }
}

fn helper_policy_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_policy_{}", last.ident))
}

fn helper_call_target_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_target_{}", last.ident))
}

fn primitive_type_ident(ty: &Type) -> Option<&Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    Some(&type_path.path.segments.last()?.ident)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HelperCallKind {
    Void,
    Int,
    Ref,
    Float,
    Unsupported,
}

fn is_gc_ref_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Path(type_path)
            if type_path.qself.is_none()
                && type_path
                    .path
                    .segments
                    .last()
                    .map(|segment| segment.ident == "GcRef")
                    .unwrap_or(false)
    )
}

fn is_raw_pointer_type(ty: &Type) -> bool {
    matches!(ty, Type::Ptr(_))
}

fn helper_call_kind_for_type(ty: &Type) -> HelperCallKind {
    if is_gc_ref_type(ty) || is_raw_pointer_type(ty) {
        return HelperCallKind::Ref;
    }
    match primitive_type_ident(ty)
        .map(|ident| ident.to_string())
        .as_deref()
    {
        Some(
            "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
            | "bool",
        ) => HelperCallKind::Int,
        Some("f64") => HelperCallKind::Float,
        _ => HelperCallKind::Unsupported,
    }
}

fn helper_call_kind_for_return(output: &ReturnType) -> HelperCallKind {
    match output {
        ReturnType::Default => HelperCallKind::Void,
        ReturnType::Type(_, ty) => helper_call_kind_for_type(ty),
    }
}

fn helper_arg_from_i64(arg_ident: &Ident, ty: &Type) -> Option<proc_macro2::TokenStream> {
    if is_gc_ref_type(ty) {
        return Some(quote! { #ty((#arg_ident) as usize) });
    }
    if is_raw_pointer_type(ty) {
        return Some(quote! { ((#arg_ident) as usize) as #ty });
    }
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            Some(quote! { (#arg_ident) as #ty })
        }
        "i64" => Some(quote! { #arg_ident }),
        "bool" => Some(quote! { (#arg_ident) != 0 }),
        "f64" => Some(quote! { f64::from_bits((#arg_ident) as u64) }),
        _ => None,
    }
}

fn helper_return_to_i64(
    value: proc_macro2::TokenStream,
    ty: &Type,
) -> Option<proc_macro2::TokenStream> {
    if is_gc_ref_type(ty) {
        return Some(quote! { (#value).0 as i64 });
    }
    if is_raw_pointer_type(ty) {
        return Some(quote! { (#value) as usize as i64 });
    }
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "u8" | "u16" | "u32" | "u64" | "usize" | "bool" => {
            Some(quote! { (#value) as i64 })
        }
        "i64" => Some(quote! { #value }),
        "isize" => Some(quote! { (#value) as i64 }),
        "f64" => Some(quote! { f64::to_bits(#value) as i64 }),
        _ => None,
    }
}

fn emit_helper_call_target_fn(
    func: &ItemFn,
) -> syn::Result<Option<(Ident, Ident, proc_macro2::TokenStream)>> {
    if !func.sig.generics.params.is_empty() {
        return Ok(None);
    }

    let trace_target_name = helper_call_target_fn_name(&Path::from(func.sig.ident.clone()))?;
    let concrete_target_name = format_ident!("{}_concrete", trace_target_name);
    let mut wrapper_params = Vec::new();
    let mut converted_args = Vec::new();
    for (index, arg) in func.sig.inputs.iter().enumerate() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Ok(None);
        };
        let arg_ident = format_ident!("__majit_arg_{index}");
        wrapper_params.push(quote! { #arg_ident: i64 });
        let Some(converted) = helper_arg_from_i64(&arg_ident, &pat_type.ty) else {
            return Ok(None);
        };
        converted_args.push(converted);
    }

    let helper_name = &func.sig.ident;
    let wrapper = match helper_call_kind_for_return(&func.sig.output) {
        HelperCallKind::Void => quote! {
            #[doc(hidden)]
            pub(crate) extern "C" fn #trace_target_name(#(#wrapper_params),*) {
                #helper_name(#(#converted_args),*);
            }
        },
        HelperCallKind::Int | HelperCallKind::Ref => {
            let ReturnType::Type(_, ty) = &func.sig.output else {
                return Ok(None);
            };
            let Some(converted_return) = helper_return_to_i64(
                quote! {
                    #helper_name(#(#converted_args),*)
                },
                ty,
            ) else {
                return Ok(None);
            };
            quote! {
                #[doc(hidden)]
                pub(crate) extern "C" fn #trace_target_name(#(#wrapper_params),*) -> i64 {
                    #converted_return
                }
            }
        }
        HelperCallKind::Float => {
            let ReturnType::Type(_, ty) = &func.sig.output else {
                return Ok(None);
            };
            let float_wrapper = quote! {
                #[doc(hidden)]
                pub(crate) extern "C" fn #trace_target_name(#(#wrapper_params),*) -> f64 {
                    #helper_name(#(#converted_args),*)
                }
            };
            let Some(concrete_return) = helper_return_to_i64(
                quote! {
                    #helper_name(#(#converted_args),*)
                },
                ty,
            ) else {
                return Ok(None);
            };
            let concrete_wrapper = quote! {
                #[doc(hidden)]
                pub(crate) extern "C" fn #concrete_target_name(#(#wrapper_params),*) -> i64 {
                    #concrete_return
                }
            };
            quote! {
                #float_wrapper
                #concrete_wrapper
            }
        }
        HelperCallKind::Unsupported => return Ok(None),
    };

    let concrete_name = if matches!(
        helper_call_kind_for_return(&func.sig.output),
        HelperCallKind::Float
    ) {
        concrete_target_name
    } else {
        trace_target_name.clone()
    };
    Ok(Some((trace_target_name, concrete_name, wrapper)))
}

fn helper_policy_tokens_for_fn(
    func: &ItemFn,
    attr_name: &str,
    trace_target_name: Option<&Ident>,
    concrete_target_name: Option<&Ident>,
) -> syn::Result<proc_macro2::TokenStream> {
    let unsupported = quote! {
        (0u8, std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null())
    };
    let (Some(trace_target_name), Some(concrete_target_name)) =
        (trace_target_name, concrete_target_name)
    else {
        return Ok(unsupported);
    };
    // 5-tuple: (policy, inline_builder, trace_target, concrete_target, prebuild).
    // `prebuild` is the per-helper liveness prebuild fn pointer or null
    // for non-Inline helpers (these have no per-marker triples to register).
    // Only `#[jit_inline]` emits a real prebuild fn at `lib.rs:1330`; every
    // other helper attribute that flows through here advertises null and
    // the parent `#[jit_interp]` lowerer's inferred-policy site
    // (`jitcode_lower.rs::CallPolicySpec::Infer`) skips the call.
    match helper_call_kind_for_return(&func.sig.output) {
        HelperCallKind::Void => Ok(match attr_name {
            "dont_look_inside" => quote! {
                (1u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_may_force" => quote! {
                (9u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_release_gil" => quote! {
                (13u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_loop_invariant" => quote! {
                (17u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            _ => unsupported,
        }),
        HelperCallKind::Int => Ok(match attr_name {
            "elidable" => quote! {
                (3u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "dont_look_inside" => quote! {
                (2u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_may_force" => quote! {
                (10u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_release_gil" => quote! {
                (14u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            "jit_loop_invariant" => quote! {
                (18u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            _ => unsupported,
        }),
        HelperCallKind::Ref => Ok(match attr_name {
            // RPython rewrite_call() bakes the result kind into the opname.
            // Our infer path cannot recover a static ref-return BindingKind,
            // so keep the call targets for explicit wrapped policies but do
            // not advertise a value-call policy code here.
            "elidable" | "dont_look_inside" | "jit_may_force" | "jit_release_gil"
            | "jit_loop_invariant" => quote! {
                (0u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            _ => unsupported,
        }),
        HelperCallKind::Float => Ok(match attr_name {
            // Same restriction as ref-return helpers: explicit wrapped float
            // policies consume these targets directly, but inferred value-call
            // lowering cannot model the static float result bank.
            "elidable" | "dont_look_inside" | "jit_may_force" | "jit_release_gil"
            | "jit_loop_invariant" => quote! {
                (0u8, std::ptr::null(), #trace_target_name as *const (), #concrete_target_name as *const (), std::ptr::null())
            },
            _ => unsupported,
        }),
        HelperCallKind::Unsupported => Ok(unsupported),
    }
}

fn emit_helper_policy_fn(
    path: &Path,
    body: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let helper_name = helper_policy_fn_name(path)?;
    Ok(quote! {
        #[doc(hidden)]
        pub(crate) fn #helper_name() -> (u8, *const (), *const (), *const (), *const ()) {
            #body
        }
    })
}

/// Parsed contents of `#[jit_driver(greens = [...], reds = [...])]`.
struct JitDriverArgs {
    greens: Vec<Ident>,
    reds: Vec<Ident>,
    virtualizable: Option<Ident>,
}

/// Parse a bracketed list of identifiers: `[a, b, c]`.
fn parse_ident_list(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    syn::bracketed!(content in input);
    let idents = content.parse_terminated(Ident::parse, Token![,])?;
    Ok(idents.into_iter().collect())
}

impl Parse for JitDriverArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut greens = None;
        let mut reds = None;
        let mut virtualizable = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "greens" => {
                    if greens.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `greens`"));
                    }
                    greens = Some(parse_ident_list(input)?);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_ident_list(input)?);
                }
                "virtualizable" => {
                    if virtualizable.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `virtualizable`"));
                    }
                    virtualizable = Some(input.parse::<Ident>()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_driver parameter: `{other}`"),
                    ));
                }
            }

            // Consume optional trailing comma between greens and reds
            let _ = input.parse::<Token![,]>();
        }

        let greens = greens
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `greens`"))?;
        let reds =
            reds.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `reds`"))?;

        Ok(JitDriverArgs {
            greens,
            reds,
            virtualizable,
        })
    }
}

/// Mark a struct as a JIT driver configuration.
///
/// Usage:
/// ```ignore
/// #[majit::jit_driver(
///     greens = [next_instr, pycode],
///     reds = [frame, ec],
/// )]
/// struct MyJitDriver;
/// ```
///
/// Generates an `impl` block with associated constants describing the green
/// and red variable names, their counts, and the total number of JIT
/// variables.
#[proc_macro_attribute]
pub fn jit_driver(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JitDriverArgs);

    let input: syn::DeriveInput = match syn::parse(item) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let struct_name = &input.ident;
    let virtualizable = args.virtualizable.clone();

    let green_strs: Vec<String> = args.greens.iter().map(|id| id.to_string()).collect();
    let red_strs: Vec<String> = args.reds.iter().map(|id| id.to_string()).collect();
    let greens_joined = green_strs.join(", ");
    let reds_joined = red_strs.join(", ");

    let num_greens = green_strs.len();
    let num_reds = red_strs.len();
    let num_vars = num_greens + num_reds;

    let mut seen = std::collections::HashSet::new();
    for green in &args.greens {
        if !seen.insert(green.to_string()) {
            return syn::Error::new(green.span(), "duplicate variable in `greens`")
                .to_compile_error()
                .into();
        }
    }
    for red in &args.reds {
        if !seen.insert(red.to_string()) {
            return syn::Error::new(red.span(), "green/red variables must be distinct")
                .to_compile_error()
                .into();
        }
    }
    if let Some(virtualizable) = &virtualizable {
        if !args.reds.iter().any(|red| red == virtualizable) {
            return syn::Error::new(
                virtualizable.span(),
                "`virtualizable` must name one of the red variables",
            )
            .to_compile_error()
            .into();
        }
    }

    let doc = format!("JIT driver: greens=[{greens_joined}], reds=[{reds_joined}]");

    let attrs = &input.attrs;
    let vis = &input.vis;
    let generics = &input.generics;
    let data = &input.data;

    // Re-emit the struct with doc annotation, then add the impl block.
    let struct_token = match data {
        syn::Data::Struct(s) => {
            let fields = &s.fields;
            let semi = &s.semi_token;
            quote! {
                #(#attrs)*
                #[doc = #doc]
                #vis struct #struct_name #generics #fields #semi
            }
        }
        _ => {
            return syn::Error::new_spanned(&input, "jit_driver can only be applied to structs")
                .to_compile_error()
                .into();
        }
    };

    let virtualizable_value = if let Some(virtualizable) = &virtualizable {
        quote! { Some(stringify!(#virtualizable)) }
    } else {
        quote! { None }
    };

    let expanded = quote! {
        #struct_token

        impl #generics #struct_name #generics {
            /// Green variable names.
            pub const GREENS: &'static [&'static str] = &[#(#green_strs),*];
            /// Red variable names.
            pub const REDS: &'static [&'static str] = &[#(#red_strs),*];
            /// Total number of JIT variables.
            pub const NUM_VARS: usize = #num_vars;
            /// Number of green variables.
            pub const NUM_GREENS: usize = #num_greens;
            /// Number of red variables.
            pub const NUM_REDS: usize = #num_reds;
            /// Name of the virtualizable red variable, if any.
            pub const VIRTUALIZABLE: Option<&'static str> = #virtualizable_value;

            pub fn descriptor(
                green_types: &[majit_ir::Type],
                red_types: &[majit_ir::Type],
            ) -> Result<majit_metainterp::JitDriverStaticData, &'static str> {
                if green_types.len() != Self::NUM_GREENS {
                    return Err("wrong number of green variable types");
                }
                if red_types.len() != Self::NUM_REDS {
                    return Err("wrong number of red variable types");
                }

                let greens = Self::GREENS
                    .iter()
                    .zip(green_types.iter().copied())
                    .map(|(name, tp)| (*name, tp))
                    .collect::<Vec<_>>();
                let reds = Self::REDS
                    .iter()
                    .zip(red_types.iter().copied())
                    .map(|(name, tp)| (*name, tp))
                    .collect::<Vec<_>>();
                let descriptor = majit_metainterp::JitDriverStaticData::with_virtualizable(
                    greens,
                    reds,
                    Self::VIRTUALIZABLE,
                );
                if let Some(virtualizable) = descriptor.virtualizable() {
                    if virtualizable.tp != majit_ir::Type::Ref {
                        return Err("virtualizable red must have Ref type");
                    }
                }
                Ok(descriptor)
            }

            pub fn green_key(values: &[i64]) -> Result<majit_ir::GreenKey, &'static str> {
                if values.len() != Self::NUM_GREENS {
                    return Err("wrong number of green key values");
                }
                Ok(majit_ir::GreenKey::new(values.to_vec()))
            }
        }

        impl #generics majit_metainterp::DeclarativeJitDriver for #struct_name #generics {
            const GREENS: &'static [&'static str] = <Self>::GREENS;
            const REDS: &'static [&'static str] = <Self>::REDS;
            const NUM_VARS: usize = <Self>::NUM_VARS;
            const NUM_GREENS: usize = <Self>::NUM_GREENS;
            const NUM_REDS: usize = <Self>::NUM_REDS;
            const VIRTUALIZABLE: Option<&'static str> = <Self>::VIRTUALIZABLE;

            fn descriptor(
                green_types: &[majit_ir::Type],
                red_types: &[majit_ir::Type],
            ) -> Result<majit_metainterp::JitDriverStaticData, &'static str> {
                <Self>::descriptor(green_types, red_types)
            }

            fn green_key(values: &[i64]) -> Result<majit_ir::GreenKey, &'static str> {
                <Self>::green_key(values)
            }
        }
    };

    expanded.into()
}

/// Mark a function as elidable (pure / constant-foldable).
///
/// The JIT can eliminate calls to this function when all arguments are constants.
/// Adds `#[inline(never)]` to prevent inlining, and a hidden `#[majit_elidable]`
/// marker attribute that the tracer can detect at compile time.
#[proc_macro_attribute]
pub fn elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(
            &func,
            "elidable",
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_ELIDABLE: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn
    };

    expanded.into()
}

/// Mark a function as opaque to the tracer.
///
/// The JIT will not trace into this function; it will be called as a black box.
/// Adds `#[inline(never)]` to prevent inlining, and a hidden `#[majit_opaque]`
/// marker constant that the tracer can detect at compile time.
#[proc_macro_attribute]
pub fn dont_look_inside(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(
            &func,
            "dont_look_inside",
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OPAQUE: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn
    };

    expanded.into()
}

fn expand_call_surface_attr(attr_name: &str, marker_name: &str, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let marker = format_ident!("{marker_name}");
    let policy_path = Path::from(sig.ident.clone());
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&func) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(
            &func,
            attr_name,
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const #marker: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn
    };

    expanded.into()
}

/// Mark a function as a may-force call surface.
#[proc_macro_attribute]
pub fn jit_may_force(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_may_force", "_MAJIT_MAY_FORCE", item)
}

/// Mark a function as a release-GIL call surface.
#[proc_macro_attribute]
pub fn jit_release_gil(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_release_gil", "_MAJIT_RELEASE_GIL", item)
}

/// Mark a function as a loop-invariant call surface.
#[proc_macro_attribute]
pub fn jit_loop_invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_loop_invariant", "_MAJIT_LOOP_INVARIANT", item)
}

/// Mark a function as loop-invariant.
///
/// RPython name parity alias for `#[jit_loop_invariant]`.
///
/// rlib/jit.py:161 — `@loop_invariant`: describes a function with no argument
/// that returns an object that is always the same in a loop.
/// Implies `@dont_look_inside`.
#[proc_macro_attribute]
pub fn loop_invariant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    expand_call_surface_attr("jit_loop_invariant", "_MAJIT_LOOP_INVARIANT", item)
}

/// Mark struct fields whose value never mutates after construction.
///
/// RPython parity: `_immutable_fields_ = ['field_a', 'field_b']` on the
/// class body (`rpython/rlib/jit.py` -> `rpython/rtyper/lltypesystem/rclass.py`).
/// The annotator pipes the list through to `cpu.fielddescrof()` which
/// stores `is_pure=True` on the field descr, allowing the JIT to fold
/// reads of those fields into constants once the receiver is known.
///
/// Usage:
/// ```ignore
/// #[majit_macros::jit_immutable_fields(pools)]
/// pub struct Storage {
///     pub pools: [*mut Stack; STORAGE_COUNT],
///     ...
/// }
/// ```
///
/// The proc-macro is a pass-through: it leaves the struct definition
/// untouched and exists solely so `rustc` accepts the attribute. The
/// codewriter front-end (`majit-translate::front::ast`) reads the
/// attribute directly from the parsed source via `syn` and feeds the
/// field list into the struct layout / descr pipeline.
#[proc_macro_attribute]
pub fn jit_immutable_fields(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a method (or free function) as elidable / pure.
///
/// RPython parity: `@jit.elidable` (`rpython/rlib/jit.py:13`). The JIT
/// can fold calls to this function once all arguments are constant.
///
/// Companion to the existing `#[elidable]` attribute, with two
/// differences:
///   1. Works on `ImplItemFn` as well as free functions — the existing
///      `#[elidable]` parses as `ItemFn` and rejects `impl` methods.
///   2. Pure pass-through, so it does not synthesize trampolines /
///      helper policy tokens. Methods on `&self` / `&mut self` are
///      called by the codewriter via `CallTarget::method` path
///      resolution, which doesn't need the trampoline that free
///      functions get.
///
/// The codewriter front-end (`majit-translate::front::ast::collect_jit_hints`)
/// already recognises the bare attribute name `jit_elidable` and
/// flips the function hint to `"elidable"`, which `mark_elidable`
/// consumes downstream.
#[proc_macro_attribute]
pub fn jit_elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// JIT can safely unroll loops in this function and this will
/// not lead to code explosion.
///
/// rlib/jit.py:150 — `@unroll_safe`.
/// Cannot be combined with `#[elidable]` or `#[dont_look_inside]`.
#[proc_macro_attribute]
pub fn unroll_safe(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_UNROLL_SAFE: bool = true;
            #block
        }
    };

    expanded.into()
}

/// A decorator for a function with no return value.  It makes the
/// function call disappear from the jit traces. It is still called in
/// interpreted mode, and by the jit tracing and blackholing, but not
/// by the final assembler.
///
/// rlib/jit.py:260 — `@not_in_trace`.
#[proc_macro_attribute]
pub fn not_in_trace(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_NOT_IN_TRACE: bool = true;
            #block
        }
    };

    expanded.into()
}

/// A decorator that promotes all arguments and then calls the supplied
/// elidable function.
///
/// rlib/jit.py:180 — `@elidable_promote(promote_args='all')`.
///
/// The decorated name **becomes** the promoting wrapper (RPython parity).
/// The original elidable body is renamed to a hidden `_orig_<name>_unlikely_name`.
///
/// Usage:
///   `#[elidable_promote]` — promote all arguments (default)
///   `#[elidable_promote(promote_args = "0,2")]` — promote args at indices 0, 2
#[proc_macro_attribute]
pub fn elidable_promote(attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);

    // Parse promote_args from attribute
    let promote_args_str = if attr.is_empty() {
        "all".to_string()
    } else {
        let config = parse_macro_input!(attr as ElidablePromoteArgs);
        config.promote_args
    };

    let arg_count = func.sig.inputs.len();
    let promote_indices: Vec<usize> = if promote_args_str == "all" {
        (0..arg_count).collect()
    } else {
        promote_args_str
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let fn_name = &sig.ident;
    // rlib/jit.py:196 — _orig_func_unlikely_name
    let orig_name = format_ident!("_orig_{}_unlikely_name", fn_name);
    let output = &sig.output;

    // Collect param info (rlib/jit.py:186 _get_args — includes self if method)
    let mut param_names: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut full_params: Vec<syn::FnArg> = Vec::new();
    let mut named_args: Vec<Ident> = Vec::new();
    let has_self = matches!(sig.inputs.first(), Some(FnArg::Receiver(_)));
    for arg in &sig.inputs {
        full_params.push(arg.clone());
        match arg {
            FnArg::Receiver(_) => {
                param_names.push(quote! { self });
            }
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = pat_ident.ident.clone();
                    param_names.push(quote! { #name });
                    named_args.push(name);
                }
            }
        }
    }

    // rlib/jit.py:191-194 — promote each selected arg with both hints.
    // promote_indices index into full arg list (including self), matching _get_args.
    let promote_stmts: Vec<_> = promote_indices
        .iter()
        .filter_map(|&idx| {
            if has_self && idx == 0 {
                // rlib/jit.py:193 — promote self by identity (guard_value on pointer)
                Some(quote! {
                    let _ = majit_metainterp::jit::promote(self as *const _ as usize);
                })
            } else {
                let named_idx = if has_self { idx - 1 } else { idx };
                named_args.get(named_idx).map(|name| {
                    quote! { let #name = majit_metainterp::jit::promote(#name); }
                })
            }
        })
        .collect();

    let call_args: Vec<_> = param_names.clone();

    // Build call_target/policy for the ORIGINAL elidable function
    let orig_sig = syn::Signature {
        ident: orig_name.clone(),
        ..sig.clone()
    };
    let orig_func = syn::ItemFn {
        sig: orig_sig,
        ..func.clone()
    };
    let (trace_target_name, concrete_target_name, call_target_fn) =
        match emit_helper_call_target_fn(&orig_func) {
            Ok(Some((trace_name, concrete_name, tokens))) => {
                (Some(trace_name), Some(concrete_name), Some(tokens))
            }
            Ok(None) => (None, None, None),
            Err(err) => return err.to_compile_error().into(),
        };
    let policy_path = Path::from(orig_name.clone());
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(
            &orig_func,
            "elidable",
            trace_target_name.as_ref(),
            concrete_target_name.as_ref(),
        ) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        // rlib/jit.py:184-185 — elidable(func); original body hidden
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        fn #orig_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_ELIDABLE: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn

        // rlib/jit.py:188-200 — the decorated name IS the promoting wrapper
        #(#attrs)*
        #vis fn #fn_name(#(#full_params),*) #output {
            #(#promote_stmts)*
            #orig_name(#(#call_args),*)
        }
    };

    expanded.into()
}

/// Parse helper for `#[elidable_promote(promote_args = "...")]`.
struct ElidablePromoteArgs {
    promote_args: String,
}

impl Parse for ElidablePromoteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let key: Ident = input.parse()?;
        if key != "promote_args" {
            return Err(syn::Error::new(key.span(), "expected `promote_args`"));
        }
        input.parse::<Token![=]>()?;
        let value: syn::LitStr = input.parse()?;
        Ok(Self {
            promote_args: value.value(),
        })
    }
}

/// The JIT compiler won't look inside this decorated function,
/// but instead during translation, rewrites it according to the handler in
/// the codewriter/jtransform.
///
/// rlib/jit.py:250 — `@oopspec(spec)`.
///
/// Usage: `#[oopspec("jit.isconstant(value)")]`
///
/// The spec string is stored as a hidden constant for the codewriter to discover.
#[proc_macro_attribute]
pub fn oopspec(attr: TokenStream, item: TokenStream) -> TokenStream {
    let spec: syn::LitStr = parse_macro_input!(attr as syn::LitStr);
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let spec_value = spec.value();

    let expanded = quote! {
        #(#attrs)*
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OOPSPEC: &str = #spec_value;
            #block
        }
    };

    expanded.into()
}

/// Look inside (including unrolling loops) the target function, if and only if
/// `predicate(args)` returns true.
///
/// rlib/jit.py:208 — `@look_inside_iff(predicate)`.
///
/// Generates three functions:
/// 1. `_orig_<name>` — original body, marked `#[unroll_safe]` (hidden)
/// 2. `<name>_trampoline` — `@dont_look_inside` wrapper calling _orig (hidden)
/// 3. `<name>` — dispatch wrapper (the public name):
///    `if !we_are_jitted() || predicate(args) { _orig(args) } else { trampoline(args) }`
///
/// Usage: `#[look_inside_iff(my_predicate)]`
/// where `my_predicate` has the same signature as the decorated function returning bool.
#[proc_macro_attribute]
pub fn look_inside_iff(attr: TokenStream, item: TokenStream) -> TokenStream {
    let predicate_path: Path = parse_macro_input!(attr as Path);
    let func = parse_macro_input!(item as ItemFn);

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let fn_name = &sig.ident;
    // rlib/jit.py:213 — func = unroll_safe(func)
    let orig_name = format_ident!("_orig_{}", fn_name);
    // rlib/jit.py:232 — trampoline.__name__ = func.__name__ + "_trampoline"
    let trampoline_name = format_ident!("{}_trampoline", fn_name);
    let output = &sig.output;

    // Collect parameter patterns and call argument expressions.
    // rlib/jit.py:221 args = _get_args(func) — includes `self` if method.
    let mut full_params: Vec<syn::FnArg> = Vec::new();
    let mut call_args: Vec<proc_macro2::TokenStream> = Vec::new();
    for arg in &sig.inputs {
        full_params.push(arg.clone());
        match arg {
            FnArg::Receiver(_) => {
                // Forward `self` as-is (works for &self, &mut self, self, Box<Self>).
                call_args.push(quote! { self });
            }
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = &*pat_type.pat {
                    let name = &pat_ident.ident;
                    call_args.push(quote! { #name });
                }
            }
        }
    }

    let expanded = quote! {
        // rlib/jit.py:213-214 — func = unroll_safe(func)
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        fn #orig_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_UNROLL_SAFE: bool = true;
            #block
        }

        // rlib/jit.py:231-233 — @dont_look_inside def trampoline(...): return func(...)
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        fn #trampoline_name(#(#full_params),*) #output {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OPAQUE: bool = true;
            #orig_name(#(#call_args),*)
        }

        // rlib/jit.py:240-244 — the decorated name becomes the dispatch wrapper
        // def f(*args):
        //     if not we_are_jitted() or predicate(*args):
        //         return func(*args)
        //     else:
        //         return trampoline(*args)
        #(#attrs)*
        #vis fn #fn_name(#(#full_params),*) #output {
            if !majit_metainterp::jit::we_are_jitted() || #predicate_path(#(#call_args),*) {
                #orig_name(#(#call_args),*)
            } else {
                #trampoline_name(#(#call_args),*)
            }
        }
    };

    expanded.into()
}

/// Serialize a helper into a hidden `JitCode` builder.
///
/// This is the proc-macro side of RPython's `codewriter.py` helper serialization:
/// the original function stays callable by the interpreter, and the macro also
/// emits a hidden `__majit_inline_jitcode_*()` function that `#[jit_interp]`
/// can use when a call policy maps the helper to `inline_int`/`inline_ref`/`inline_float`.
///
/// Supports Int (i64/isize), Ref (usize/pointer), and Float (f64) return types
/// and parameter types.
#[proc_macro_attribute]
pub fn jit_inline(attr: TokenStream, item: TokenStream) -> TokenStream {
    use jit_interp::jitcode_lower::InlineReturnKind;

    let args = parse_macro_input!(attr as JitInlineArgs);
    let func = parse_macro_input!(item as ItemFn);
    let helper = match jit_interp::jitcode_lower::generate_inline_helper_jitcode_with_calls(
        &func,
        &args.calls,
    ) {
        Ok(Some(lowered)) => lowered,
        Ok(None) => {
            return syn::Error::new_spanned(
                &func.block,
                "#[jit_inline] could not lower this helper into JitCode",
            )
            .to_compile_error()
            .into();
        }
        Err(err) => return err.to_compile_error().into(),
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let helper_with_asm_name = format_ident!("__majit_inline_jitcode_{}_with_asm", sig.ident);
    let helper_prebuild_name = format_ident!("__majit_inline_jitcode_{}_prebuild", sig.ident);
    let policy_name = format_ident!("__majit_call_policy_{}", sig.ident);
    let helper_body = helper.body;
    let helper_liveness_prebuild = helper.liveness_prebuild;
    let return_reg = helper.return_reg;
    let helper_return = match helper.return_kind {
        InlineReturnKind::Int => quote! { __builder.int_return(#return_reg); },
        InlineReturnKind::Ref => quote! { __builder.ref_return(#return_reg); },
        InlineReturnKind::Float => quote! { __builder.float_return(#return_reg); },
    };

    // RPython jtransform.py: rewrite_call() bakes the result kind into the
    // emitted opname. Our inferred helper-policy surface can only model that
    // parity for int-return inline helpers; ref/float inline helpers must go
    // through explicit `inline_ref` / `inline_float` policies.
    let inferred_policy_code: u8 = match helper.return_kind {
        InlineReturnKind::Int => 4,
        InlineReturnKind::Ref | InlineReturnKind::Float => 0,
    };
    let inferred_inline_builder = match helper.return_kind {
        InlineReturnKind::Int => quote! { #helper_with_asm_name as *const () },
        InlineReturnKind::Ref | InlineReturnKind::Float => quote! { std::ptr::null() },
    };

    // Ensure the right register file for each parameter
    let ensure_param_regs = {
        let mut stmts = Vec::new();
        let (count_i, count_r, count_f) =
            match jit_interp::jitcode_lower::inline_helper_param_counts(&func) {
                Ok(counts) => counts,
                Err(err) => return err.to_compile_error().into(),
            };
        if count_i != 0 {
            stmts.push(quote! { __builder.ensure_i_regs(#count_i); });
        }
        if count_r != 0 {
            stmts.push(quote! { __builder.ensure_r_regs(#count_r); });
        }
        if count_f != 0 {
            stmts.push(quote! { __builder.ensure_f_regs(#count_f); });
        }
        stmts
    };

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #block
        }

        // Phase 4 Epic B.3-B.4: inline helper jitcodes register
        // per-marker liveness triples through the caller-supplied
        // `Assembler`.  The production caller threads the driver-shared
        // `Assembler` (see `JitDriver::shared_asm`) so all jitcodes —
        // top-level per-pc bodies and inline helpers — share the same
        // `all_liveness` byte stream and dedup against the same cache.
        // RPython parity: `rpython/jit/codewriter/assembler.py` is a
        // single object that assembles every JitCode in the program
        // (`call.py:174-189`), so liveness offsets are always relative
        // to one shared table.
        #[doc(hidden)]
        pub(crate) fn #helper_with_asm_name(
            __asm: &mut majit_metainterp::Assembler,
        ) -> majit_metainterp::JitCode {
            let mut __builder = majit_metainterp::JitCodeBuilder::new();
            #(#ensure_param_regs)*
            #helper_body
            #helper_return
            __builder.finalize_liveness(__asm);
            __builder.finish()
        }

        // RPython `pyjitpl.py:2255 finish_setup` order: pre-register
        // the helper's per-marker `-live-` triples into the driver-
        // shared `Assembler` at install time so the trace-time
        // `__builder.finalize_liveness(__asm)` above only dedups
        // (does not grow `all_liveness` past the snapshot taken by
        // `JitDriver::install_canonical_liveness`). Called from each
        // parent `__prebuild_jitcode_liveness_*` that statically
        // resolves an inline-call to this helper (see
        // `jitcode_lower::inline_prebuild_path`).
        #[allow(non_snake_case, unused_variables, unused_mut)]
        #[doc(hidden)]
        pub(crate) fn #helper_prebuild_name(
            __asm: &mut majit_metainterp::Assembler,
        ) {
            #helper_liveness_prebuild
        }

        #[doc(hidden)]
        pub(crate) fn #policy_name() -> (u8, *const (), *const (), *const (), *const ()) {
            (
                #inferred_policy_code,
                #inferred_inline_builder,
                std::ptr::null(),
                std::ptr::null(),
                #helper_prebuild_name as *const (),
            )
        }
    };

    expanded.into()
}

/// Auto-generate trace_instruction and JitState from an interpreter's dispatch loop.
///
/// This is the Rust equivalent of RPython's meta-tracing: the proc macro analyzes
/// the interpreter's opcode dispatch match and generates the tracing code automatically.
///
/// The interpreter author writes ONLY the dispatch loop (like RPython's rpaheui).
/// The macro generates:
/// - `trace_instruction()` function (IR recording for each opcode)
/// - `JitState` impl with Meta/Sym types
/// - Replaces `jit_merge_point!()` / `can_enter_jit!()` markers with JitDriver calls
///
/// # Example
///
/// ```ignore
/// #[jit_interp(
///     state = AheuiState,
///     env = Program,
///     storage = {
///         pool: state.storage,
///         selector: state.selected,
///         untraceable: [VAL_QUEUE, VAL_PORT],
///         scan: find_used_storages,
///     },
///     io_shims = {
///         aheui_io::write_number => jit_write_number,
///         aheui_io::write_utf8 => jit_write_utf8,
///     },
///     // optional: infer direct helper calls from sidecar metadata
///     // non-int result helpers still need explicit `calls = { ... => ... }`
///     auto_calls = true,
///     calls = {
///         helper_compute,
///         helper_opaque,
///         helper_sink,
///         helper_inline,
///         // explicit overrides are still allowed:
///         helper_force_residual => residual_int,
///     },
/// )]
/// pub fn mainloop_jit(program: &Program) -> i64 {
///     // ... setup ...
///     while pc < program.size {
///         jit_merge_point!();
///         match op {
///             OP_ADD => state.storage.get_mut(state.selected).add(),
///             // ...
///         }
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn jit_interp(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_macro_input!(attr as jit_interp::JitInterpConfig);
    let func = parse_macro_input!(item as ItemFn);
    jit_interp::transform_jit_interp(config, func).into()
}

/// Register a Rust struct with `majit_ir::descr::GcCache`.
///
/// RPython parity: descr.py:105-127 `get_size_descr` + descr.py:218-239
/// `get_field_descr`. RPython's translator auto-discovers `lltype.Struct`
/// fields and emits FieldDescr/SizeDescr; this macro performs the same
/// auto-discovery for Rust structs using `offset_of!` / `size_of`.
///
/// Generated inherent methods:
/// - `__majit_type_id() -> u64`
/// - `__majit_register_descrs(&mut GcCache) -> DescrRef`
/// - `const __MAJIT_FIELD_NAMES: &'static [&'static str]`
///
/// Only named-field structs are supported in this skeleton. Field types
/// are classified by a simple heuristic (integer/float primitives vs Ref).
#[proc_macro_attribute]
pub fn jit_struct(attr: TokenStream, item: TokenStream) -> TokenStream {
    jit_struct::expand(attr.into(), item.into()).into()
}

/// JIT attribute names recognized by `#[jit_module]` for automatic helper discovery.
const JIT_HELPER_ATTRS: &[&str] = &[
    "jit_inline",
    "elidable",
    "elidable_promote",
    "dont_look_inside",
    "unroll_safe",
    "loop_invariant",
    "not_in_trace",
    "look_inside_iff",
    "oopspec",
    "jit_may_force",
    "jit_release_gil",
    "jit_loop_invariant",
];

/// Check if a syn attribute path matches one of the JIT helper attributes.
fn jit_attr_name(attr: &syn::Attribute) -> Option<String> {
    let path = attr.path();
    // Match both bare `elidable` and qualified `majit_macros::elidable`
    let last_segment = path.segments.last()?;
    let name = last_segment.ident.to_string();
    if JIT_HELPER_ATTRS.contains(&name.as_str()) {
        Some(name)
    } else {
        None
    }
}

/// Discovered helper entry: function name and its JIT attribute.
///
/// `impl_type_segments` is `Some(vec)` for inherent / trait-impl methods
/// discovered inside an `impl` block, carrying the type-path segments
/// exactly as written at the `impl` header (e.g. `[a, Foo]` for
/// `impl a::Foo { ... }`). Segments are extracted from the type path
/// (`syn::Type::Path`) so that downstream code can render the
/// `impl_type` as a joined string matching the parser's
/// `self_ty_root` canonicalization (parse.rs:702, front/ast.rs:106
/// `qualify_type_name`). RPython parity: `getfunctionptr(graph)`
/// (call.py:174-187) does not distinguish free fns from methods; pyre
/// keys methods by the `[impl_type_joined, method]` 2-segment CallPath
/// (lib.rs:406-433), so the macro emits exactly that.
struct DiscoveredHelper {
    fn_name: Ident,
    attr_name: String,
    /// `None` for free fns, `Some(segments)` for impl methods.
    impl_type_segments: Option<Vec<Ident>>,
    /// `Some(segments)` for trait impls (`impl Trait for Type { fn m }`),
    /// carrying the trait path segments. Emitted in the fnaddr cast as
    /// `<Type as Trait>::method` to disambiguate when a type has
    /// multiple inherent/trait methods named `method`. `None` for
    /// inherent impls (plain `<Type>::method`) and free fns.
    trait_type_segments: Option<Vec<Ident>>,
}

/// Extract the identifier sequence from a `syn::Type::Path`, e.g.
/// `a::b::Foo` → `[a, b, Foo]`. Returns `None` for non-path types
/// (trait objects, references, fn pointers, generics on the outer
/// path, …) — those cases are not expressible as a canonical
/// `self_ty_root` string in the parser either, so we skip them.
fn impl_type_path_segments(ty: &syn::Type) -> Option<Vec<Ident>> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() {
        return None;
    }
    // Parser's `type_root_ident` strips generic arguments from each
    // segment and joins the identifiers. Match that shape here by
    // taking `ident` only.
    Some(
        type_path
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.clone())
            .collect(),
    )
}

/// Scan a module's items for functions annotated with JIT helper attributes.
///
/// Walks both top-level `Item::Fn` and inherent / trait-impl methods
/// inside `Item::Impl` blocks. Instance methods are NOT skipped — Rust
/// allows `S::f as fn(&S)`, `S::g as fn(&mut S)`, `S::h as fn(S)` to
/// coerce to plain function pointers (verified with rustc), and RPython
/// upstream treats `getfunctionptr(graph)` uniformly across free fns and
/// methods (`call.py:174-187`).
fn discover_helpers(items: &[syn::Item]) -> Vec<DiscoveredHelper> {
    let mut discovered = Vec::new();
    for item in items {
        match item {
            syn::Item::Fn(func) => {
                for attr in &func.attrs {
                    if let Some(attr_name) = jit_attr_name(attr) {
                        discovered.push(DiscoveredHelper {
                            fn_name: func.sig.ident.clone(),
                            attr_name,
                            impl_type_segments: None,
                            trait_type_segments: None,
                        });
                        // Only record the first JIT attribute per function
                        break;
                    }
                }
            }
            // RPython parity: `impl Type { fn helper(...) }` and
            // `impl Trait for Type { fn helper(...) }` both lower to a
            // `getfunctionptr(graph)` whose canonical CallPath is
            // `[impl_type_joined, helper]`.
            syn::Item::Impl(item_impl) => {
                let Some(impl_segs) = impl_type_path_segments(&item_impl.self_ty) else {
                    continue;
                };
                // `impl Trait for Type { ... }` — carry the trait path
                // so the fnaddr cast can disambiguate via
                // `<Type as Trait>::method`. RPython `getfunctionptr(graph)`
                // uses graph identity directly so no such aliasing exists
                // upstream; in Rust a bare `<Type>::method` cast is
                // ambiguous when the Type carries multiple trait methods
                // with the same name (or a name-colliding inherent).
                let trait_segs: Option<Vec<Ident>> =
                    item_impl.trait_.as_ref().and_then(|(_, path, _)| {
                        Some(
                            path.segments
                                .iter()
                                .map(|seg| seg.ident.clone())
                                .collect::<Vec<_>>(),
                        )
                        .filter(|segs: &Vec<Ident>| !segs.is_empty())
                    });
                for impl_item in &item_impl.items {
                    let syn::ImplItem::Fn(method) = impl_item else {
                        continue;
                    };
                    for attr in &method.attrs {
                        if let Some(attr_name) = jit_attr_name(attr) {
                            discovered.push(DiscoveredHelper {
                                fn_name: method.sig.ident.clone(),
                                attr_name,
                                impl_type_segments: Some(impl_segs.clone()),
                                trait_type_segments: trait_segs.clone(),
                            });
                            break;
                        }
                    }
                }
            }
            _ => {}
        }
    }
    discovered
}

/// Module-level automatic helper discovery for JIT-annotated functions.
///
/// Place `#[jit_module]` on a `mod` block containing JIT-annotated functions:
/// `#[elidable]`, `#[elidable_promote]`, `#[dont_look_inside]`,
/// `#[unroll_safe]`, `#[loop_invariant]`, `#[not_in_trace]`,
/// `#[look_inside_iff]`, `#[oopspec]`, `#[jit_inline]`,
/// `#[jit_may_force]`, `#[jit_release_gil]`, `#[jit_loop_invariant]`.
/// The macro scans all items and generates a hidden registry constant
/// listing discovered helpers and their attributes.
///
/// # Example
///
/// ```ignore
/// #[jit_module]
/// mod my_interp {
///     #[jit_inline]
///     fn helper_add(a: i64, b: i64) -> i64 { a + b }
///
///     #[elidable]
///     fn lookup(key: i64) -> i64 { /* ... */ }
///
///     #[dont_look_inside]
///     fn opaque(x: i64) -> i64 { /* ... */ }
///
///     fn not_jit_relevant() { /* ignored */ }
/// }
///
/// // After expansion, `my_interp` contains:
/// // const __MAJIT_DISCOVERED_HELPERS: &[&str] = &["helper_add", "lookup", "opaque"];
/// // const __MAJIT_HELPER_POLICIES: &[(&str, &str)] = &[
/// //     ("helper_add", "jit_inline"),
/// //     ("lookup", "elidable"),
/// //     ("opaque", "dont_look_inside"),
/// // ];
/// ```
#[proc_macro_attribute]
pub fn jit_module(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut module = parse_macro_input!(item as syn::ItemMod);

    let Some((brace, ref items)) = module.content else {
        return syn::Error::new_spanned(
            &module.ident,
            "#[jit_module] requires an inline module body (not `mod foo;`)",
        )
        .to_compile_error()
        .into();
    };

    let discovered = discover_helpers(items);

    // Per-helper tokens: free fns route through the existing
    // `__majit_call_policy_*` trampoline indirection; impl methods take
    // their direct `<Type>::method` address since the policy macro family
    // does not yet decorate impl items.  Both shapes feed the same
    // `__majit_helper_trace_fnaddrs()` table that the codewriter reads
    // through `register_macro_helper_trace_fnaddr`.
    // For impl methods we emit the `impl_type_joined` string by
    // concat!-ing the individual `stringify!(ident)` tokens per segment;
    // `stringify!` on individual idents produces clean
    // whitespace-free identifier strings (unlike `stringify!(a::Foo)`
    // which expands with spaces around `::`).  The resulting joined
    // form matches the parser's `self_ty_root` canonicalization
    // (parse.rs:400 → `type_root_ident` + `qualify_type_name`).
    let helper_name_lits: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .map(|h| {
            let fn_name = &h.fn_name;
            match &h.impl_type_segments {
                None => quote! { stringify!(#fn_name) },
                Some(segs) => {
                    let parts = segs_joined_with_method(segs, fn_name);
                    quote! { #parts }
                }
            }
        })
        .collect();
    let helper_attr_lits: Vec<&str> = discovered.iter().map(|h| h.attr_name.as_str()).collect();
    let helper_path_lits: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .map(|h| {
            let fn_name = &h.fn_name;
            match &h.impl_type_segments {
                None => quote! {
                    concat!(module_path!(), "::", stringify!(#fn_name))
                },
                Some(segs) => {
                    let parts = segs_joined_with_method(segs, fn_name);
                    quote! {
                        concat!(module_path!(), "::", #parts)
                    }
                }
            }
        })
        .collect();
    let helper_addr_exprs: Vec<proc_macro2::TokenStream> =
        discovered.iter().map(|h| impl_addr_expr(h)).collect();

    // Structured impl-method registry:
    //   `(module_path_with_crate, impl_type_as_written, method, fnaddr)`.
    // The codewriter consumes this through
    // `CallControl::register_macro_impl_helper_trace_fnaddr` which applies
    // the parser's `qualify_type_name` rule (front/ast.rs:106) to decide
    // whether to prepend the module prefix before registering the
    // canonical 2-segment CallPath `[impl_type_joined, method]`
    // (lib.rs:406-433).
    let impl_entries: Vec<proc_macro2::TokenStream> = discovered
        .iter()
        .filter_map(|h| {
            let segs = h.impl_type_segments.as_ref()?;
            let fn_name = &h.fn_name;
            let impl_type_as_written = segs_joined(segs);
            let addr = impl_addr_expr(h);
            Some(quote! {
                (
                    module_path!(),
                    #impl_type_as_written,
                    stringify!(#fn_name),
                    #addr,
                )
            })
        })
        .collect();

    let registry_names = quote! {
        /// Hidden registry of automatically discovered JIT helpers.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub const __MAJIT_DISCOVERED_HELPERS: &[&str] = &[
            #(#helper_name_lits),*
        ];
    };

    let registry_policies = quote! {
        /// Hidden registry mapping each discovered helper to its JIT attribute.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub const __MAJIT_HELPER_POLICIES: &[(&str, &str)] = &[
            #((#helper_name_lits, #helper_attr_lits)),*
        ];
    };

    let registry_trace_fnaddrs = quote! {
        /// Hidden registry mapping each discovered helper to the compiled
        /// trace-call surface address used by `majit-codewriter`'s
        /// `getfunctionptr(graph)` parity path.
        #[doc(hidden)]
        #[allow(dead_code)]
        pub fn __majit_helper_trace_fnaddrs() -> ::std::vec::Vec<(&'static str, i64)> {
            ::std::vec![
                #((#helper_path_lits, #helper_addr_exprs)),*
            ]
        }
    };

    let registry_impl_trace_fnaddrs = quote! {
        /// Hidden registry mapping each discovered impl-method helper to
        /// its `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`
        /// 4-tuple. The codewriter consumes this through
        /// `CallControl::register_macro_impl_helper_trace_fnaddr`, which
        /// applies the parser's `qualify_type_name` rule
        /// (front/ast.rs:106) to decide whether to prepend the module
        /// prefix before storing the canonical 2-segment CallPath
        /// `[impl_type_joined, method]` — same shape the parser uses for
        /// `self_ty_root`-keyed methods (parse.rs:702, lib.rs:406-433).
        #[doc(hidden)]
        #[allow(dead_code)]
        pub fn __majit_helper_impl_trace_fnaddrs()
            -> ::std::vec::Vec<(&'static str, &'static str, &'static str, i64)>
        {
            ::std::vec![
                #(#impl_entries),*
            ]
        }
    };

    // Inject the registry constants into the module body
    let mut new_items = items.clone();
    new_items.push(syn::parse2(registry_names).expect("failed to parse registry_names"));
    new_items.push(syn::parse2(registry_policies).expect("failed to parse registry_policies"));
    new_items
        .push(syn::parse2(registry_trace_fnaddrs).expect("failed to parse registry_trace_fnaddrs"));
    new_items.push(
        syn::parse2(registry_impl_trace_fnaddrs)
            .expect("failed to parse registry_impl_trace_fnaddrs"),
    );
    module.content = Some((brace, new_items));

    quote! { #module }.into()
}

/// Emit the runtime address expression for a `DiscoveredHelper`:
///
/// * Free fn: route through `__majit_call_policy_<name>()` (no change).
/// * Inherent impl (`impl Type { ... }`): `<Type>::method as *const ()`.
/// * Trait impl (`impl Trait for Type { ... }`):
///   `<Type as Trait>::method as *const ()` — fully-qualified trait
///   method syntax, so the cast is unambiguous when the type carries
///   multiple name-colliding inherent/trait methods. RPython's
///   `getfunctionptr(graph)` uses graph identity directly so upstream
///   never needs this disambiguation.
fn impl_addr_expr(h: &DiscoveredHelper) -> proc_macro2::TokenStream {
    let fn_name = &h.fn_name;
    let Some(impl_segs) = h.impl_type_segments.as_ref() else {
        let policy_fn = format_ident!("__majit_call_policy_{}", fn_name);
        return quote! {
            {
                let (_, _inline_builder, __trace_target, __concrete_target, _prebuild) = #policy_fn();
                let __trace_target = if __trace_target.is_null() {
                    if __concrete_target.is_null() {
                        #fn_name as *const ()
                    } else {
                        __concrete_target
                    }
                } else {
                    __trace_target
                };
                __trace_target as usize as i64
            }
        };
    };
    let ty_path = path_tokens_from_segs(impl_segs);
    match h.trait_type_segments.as_ref() {
        Some(trait_segs) => {
            let trait_path = path_tokens_from_segs(trait_segs);
            quote! {
                <#ty_path as #trait_path>::#fn_name as *const () as usize as i64
            }
        }
        None => quote! {
            <#ty_path>::#fn_name as *const () as usize as i64
        },
    }
}

/// Build a `concat!(stringify!(s1), "::", stringify!(s2), …)` token
/// stream that renders the impl type's joined canonical form (e.g.
/// `"a::Foo"`). Uses `stringify!` per ident to avoid the whitespace
/// artefacts that `stringify!(a::Foo)` produces.
fn segs_joined(segs: &[Ident]) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            parts.push(quote! { "::" });
        }
        parts.push(quote! { stringify!(#seg) });
    }
    quote! { concat!(#(#parts),*) }
}

/// `segs_joined` followed by `"::"` and the method name — used for the
/// full `Type::method` rendering inside `__MAJIT_DISCOVERED_HELPERS` /
/// `__MAJIT_HELPER_POLICIES` and the path slot of
/// `__majit_helper_trace_fnaddrs`.
fn segs_joined_with_method(segs: &[Ident], method: &Ident) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for seg in segs {
        parts.push(quote! { stringify!(#seg) });
        parts.push(quote! { "::" });
    }
    parts.push(quote! { stringify!(#method) });
    quote! { concat!(#(#parts),*) }
}

/// Reconstruct the `syn::Path` tokens that `<#path>::method` can use as
/// a generic-path context. Just re-emits the ident sequence
/// `s1::s2::…::sN`.
fn path_tokens_from_segs(segs: &[Ident]) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    for (i, seg) in segs.iter().enumerate() {
        if i > 0 {
            parts.push(quote! { :: });
        }
        parts.push(quote! { #seg });
    }
    quote! { #(#parts)* }
}

/// Standalone virtualizable field declaration macro.
///
/// Generates `VirtualizableInfo` builder, field spec constants, and
/// JitState hook helper functions from a declarative specification.
///
/// # Example
///
/// ```ignore
/// majit_macros::virtualizable! {
///     state = MyState,
///     name = "frame",
///     heap_ptr = |s: &MyState| s.frame_ptr(),
///     token_offset = VABLE_TOKEN_OFFSET,
///
///     fields = {
///         next_instr: int @ NEXT_INSTR_OFFSET,
///         code: ref @ CODE_OFFSET,
///     },
///
///     arrays = {
///         stack: ref @ STACK_OFFSET {
///             embedded,
///             ptr_offset: PTR_OFFSET,
///             length_offset: LEN_OFFSET,
///             items_offset: 0,
///         },
///     },
/// }
/// ```
#[proc_macro]
pub fn virtualizable(input: TokenStream) -> TokenStream {
    virtualizable::parse_and_expand(input)
}

/// Derive macro for virtualizable symbolic state structs.
///
/// Recognizes `#[vable(...)]` attributes on fields:
/// - `#[vable(frame)]` — frame pointer OpRef
/// - `#[vable(field)]` — static virtualizable field OpRef
/// - `#[vable(array_base)]` — array base index
/// - `#[vable(locals)]` — symbolic locals Vec<OpRef>
/// - `#[vable(stack)]` — symbolic stack Vec<OpRef>
/// - `#[vable(local_types)]` / `#[vable(stack_types)]` — type vectors
/// - `#[vable(nlocals)]` / `#[vable(valuestackdepth)]` — shape fields
///
/// Generates: `flush_vable_fields`, `vable_field_oprefs`,
/// `init_vable_indices`, `vable_collect_jump_args`,
/// `vable_collect_typed_jump_args`.
#[proc_macro_derive(VirtualizableSym, attributes(vable))]
pub fn derive_virtualizable_sym(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_sym(input).into()
}

/// Derive macro for virtualizable meta structs.
///
/// Recognizes `#[vable(...)]` attributes on fields:
/// - `#[vable(num_locals)]` — number of locals
/// - `#[vable(valuestackdepth)]` — value stack depth
/// - `#[vable(slot_types)]` — slot type vector
///
/// Generates: `vable_stack_only_depth`, `vable_update_vsd_from_len`.
#[proc_macro_derive(VirtualizableMeta, attributes(vable))]
pub fn derive_virtualizable_meta(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_meta(input).into()
}

/// Derive macro for virtualizable interpreter state structs.
///
/// Recognizes `#[vable(...)]` attributes:
/// - `#[vable(frame)]` — frame pointer field (usize)
/// - `#[vable(static_field = N)]` — state-backed VirtualizableInfo field at index N
///
/// Generates: `virt_export_static_boxes`, `virt_import_static_boxes`,
/// `virt_export_all`.
#[proc_macro_derive(VirtualizableState, attributes(vable))]
pub fn derive_virtualizable_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    virtualizable::expand_state(input).into()
}

#[cfg(test)]
mod tests {
    // Proc macro crates cannot have unit tests that invoke the macros directly.
    // Integration tests and compile-time tests are used instead.
    // The parse logic is validated via the proc macro invocations in dependent crates.
}
