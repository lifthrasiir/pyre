//! Slice 92.1 — per-green type tag parsing for `#[jit_interp(greens = ...)]`.
//!
//! Extends the bracketed `greens = [pc, code: str, env: ref]` syntax so each
//! green can carry an optional type tag. Tagged greens override the trait
//! dispatch in `green_key_expr` (mod.rs:912) so a `&str` green emits
//! `(ptr_bits, GreenType::Str)` directly — letting `equal_whatever` /
//! `hash_whatever` route through the hardcoded `default_str_eq` /
//! `default_str_hash` / `default_unicode_hash` (`majit-ir/src/value.rs`)
//! which mirror `rstr.LLHelpers.ll_streq` / `ll_strhash` over the
//! `*const &'static str` slot ABI (warmstate.py:108-128 `lltype.Ptr`
//! to `rstr.STR / rstr.UNICODE` parity, hardcoded with no frontend
//! override).
//!
//! Untagged greens (the existing form) keep the
//! `<_ as majit_ir::GreenAsI64>::__green_repr(<expr>)` path unchanged.

use syn::{
    Expr, Ident, Token, bracketed,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Per-green type tag.  Maps to `majit_ir::GreenType` at codegen time
/// (Slice 92.2).  `Int / Ref / Float` are siblings of the `GreenAsI64`
/// trait's automatic dispatch (the tag forces the bucket explicitly);
/// `Str / Unicode` opt-in to content-comparison through the hardcoded
/// `default_str_eq` / `default_str_hash` / `default_unicode_hash` in
/// `majit-ir/src/value.rs` (`warmstate.py:108-128 ll_streq` /
/// `ll_strhash` parity, no frontend override).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GreenTypeTag {
    Int,
    Ref,
    Float,
    Str,
    Unicode,
}

/// One green declaration with optional type tag.
#[derive(Clone)]
pub(crate) struct GreenSpec {
    pub expr: Expr,
    pub type_tag: Option<GreenTypeTag>,
}

impl Parse for GreenSpec {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let expr: Expr = input.parse()?;
        let type_tag = if input.peek(Token![:]) {
            let _: Token![:] = input.parse()?;
            // `ref` is a Rust keyword — `Ident::parse` rejects it, so use
            // `Ident::parse_any` to accept keyword-shaped tags.
            let ident: Ident = Ident::parse_any(input)?;
            Some(match ident.to_string().as_str() {
                "int" => GreenTypeTag::Int,
                "ref" => GreenTypeTag::Ref,
                "float" => GreenTypeTag::Float,
                "str" => GreenTypeTag::Str,
                "unicode" => GreenTypeTag::Unicode,
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown green type tag '{other}' \
                             (expected int|ref|float|str|unicode)",
                        ),
                    ));
                }
            })
        } else {
            None
        };
        Ok(GreenSpec { expr, type_tag })
    }
}

/// Parse `[expr1, name: tag, expr3, ...]` into a vector of `GreenSpec`.
pub(crate) fn parse_green_spec_list(input: ParseStream) -> syn::Result<Vec<GreenSpec>> {
    let content;
    bracketed!(content in input);
    let specs: Punctuated<GreenSpec, Token![,]> =
        content.parse_terminated(GreenSpec::parse, Token![,])?;
    Ok(specs.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_specs(src: &str) -> syn::Result<Vec<GreenSpec>> {
        // Wrap in an outer parser so `bracketed!` finds the brackets.
        struct Wrapper(Vec<GreenSpec>);
        impl Parse for Wrapper {
            fn parse(input: ParseStream) -> syn::Result<Self> {
                Ok(Wrapper(parse_green_spec_list(input)?))
            }
        }
        let wrapped: Wrapper = syn::parse_str(src)?;
        Ok(wrapped.0)
    }

    #[test]
    fn untagged_idents_parse_with_no_tag() {
        let specs = parse_specs("[pc, stackok, is_queue, program]").unwrap();
        assert_eq!(specs.len(), 4);
        for s in &specs {
            assert_eq!(s.type_tag, None);
        }
    }

    #[test]
    fn typed_specs_carry_their_tags() {
        let specs = parse_specs("[pc, code: str, env: ref, val: int]").unwrap();
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].type_tag, None);
        assert_eq!(specs[1].type_tag, Some(GreenTypeTag::Str));
        assert_eq!(specs[2].type_tag, Some(GreenTypeTag::Ref));
        assert_eq!(specs[3].type_tag, Some(GreenTypeTag::Int));
    }

    #[test]
    fn unknown_tag_is_rejected_with_descriptive_error() {
        match parse_specs("[pc: void]") {
            Ok(_) => panic!("expected an error for unknown tag 'void'"),
            Err(e) => assert!(
                e.to_string().contains("unknown green type tag"),
                "unexpected error: {e}",
            ),
        }
    }

    #[test]
    fn complex_expressions_still_parse_without_tag() {
        let specs = parse_specs("[state.pc, program.get_op(idx)]").unwrap();
        assert_eq!(specs.len(), 2);
        for s in &specs {
            assert_eq!(s.type_tag, None);
        }
    }

    #[test]
    fn float_and_unicode_tags_round_trip() {
        let specs = parse_specs("[scale: float, name: unicode]").unwrap();
        assert_eq!(specs[0].type_tag, Some(GreenTypeTag::Float));
        assert_eq!(specs[1].type_tag, Some(GreenTypeTag::Unicode));
    }
}
