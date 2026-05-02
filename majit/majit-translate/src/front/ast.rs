//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into FunctionGraph ops with proper data flow (ValueId linking).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use syn::{Item, ItemFn};

use crate::ParsedInterpreter;
use crate::flowspace::model::ConstValue;
use crate::model::{
    BlockId, CallTarget, ExitCase, ExitSwitch, FunctionGraph, ImmutableRank, Link, LinkArg, OpKind,
    UnknownKind, UnsupportedExprKind, UnsupportedLiteralKind, ValueId, ValueType,
    exception_exitcase,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

/// Signal that lowering was halted due to an unsupported construct.
///
/// RPython `rpython/flowspace/flowcontext.py:258,417` raises `FlowingError`
/// when the abstract interpreter hits a bytecode it cannot model; that
/// error propagates all the way out of `build_flow_graph`, aborting the
/// current graph rather than silently continuing with a synthetic value.
///
/// Pyre's `Option<ValueId>` return conflates "expression legitimately
/// produced no value" (e.g. `return` / `break`) with "lowering halted"
/// — making the latter an explicit `Err` variant restores the RPython
/// invariant that unsupported constructs stop the walk at once.  The
/// `Unknown` op is still emitted at the failure site so downstream
/// passes see evidence of the drop; the `Err` just guarantees no
/// synthesised SSA value follows it.
///
/// PyPy distinguishes two kinds of "stop this walk":
///
/// 1. `FlowingError` — unsupported opcode encountered, the whole graph
///    is invalid and `build_flow` re-raises upward
///    (`rpython/flowspace/objspace.py:38`,
///    `rpython/flowspace/flowcontext.py:417`).  This is the `Err` arm
///    here.
/// 2. `FlowSignal::Return` / `FlowSignal::Raise` / `FlowSignal::Break`
///    / `FlowSignal::Continue` — the current block is closed (return
///    to caller, raise into exceptblock, goto loop tail/header), but
///    sibling walks (the other arm of a conditional, arms of a match)
///    continue normally
///    (`rpython/flowspace/flowcontext.py:1253`
///    `Raise.nomoreblocks`).  These are signalled via
///    `Lowered { path_closed: true }` on the `Ok` arm — the caller
///    stops lowering into the closed block but keeps walking siblings.
#[derive(Debug, Clone)]
pub enum FlowingError {
    Unsupported { kind: UnknownKind },
}

/// Result of lowering one expression or statement-list tail.
///
/// `path_closed` tracks the RPython `FlowSignal` state-machine.  When a
/// sub-expression raises `Return` / `Raise` / `Break` / `Continue`, the
/// block where the signal fires is closed with the appropriate
/// terminator and `path_closed` becomes `true`; parent walkers stop
/// lowering into that block but continue their sibling walks.
#[derive(Debug, Clone, Copy)]
pub struct Lowered {
    pub value: Option<ValueId>,
    pub path_closed: bool,
}

impl Lowered {
    pub fn value(v: ValueId) -> Self {
        Lowered {
            value: Some(v),
            path_closed: false,
        }
    }
    pub fn no_value() -> Self {
        Lowered {
            value: None,
            path_closed: false,
        }
    }
    pub fn path_closed() -> Self {
        Lowered {
            value: None,
            path_closed: true,
        }
    }
}

/// Propagate `path_closed` up the call chain, or unwrap the inner
/// `ValueId` if the child produced one.  Used in expression contexts
/// that REQUIRE a value from the sub-expression — if the sub-expr
/// returned `None` with the path still open, that is a FlowingError
/// (well-typed Rust does not produce such a state).
macro_rules! get_value {
    ($lowered:expr) => {{
        let __l = $lowered;
        if __l.path_closed {
            return Ok(Lowered::path_closed());
        }
        match __l.value {
            Some(v) => v,
            None => {
                return Err(FlowingError::Unsupported {
                    kind: UnknownKind::UnsupportedExpr {
                        variant: UnsupportedExprKind::OtherExpr,
                    },
                });
            }
        }
    }};
}

/// Legacy alias: callers that pre-date the `FlowingError` / `Lowered`
/// split in this file still reference `LoweringAbort`.  Keep the name
/// pointing at `FlowingError` until all in-crate consumers migrate.
pub type LoweringAbort = FlowingError;

#[derive(Debug, Clone)]
pub struct SemanticFunction {
    pub name: String,
    pub graph: FunctionGraph,
    /// RPython: `op.result.concretetype` — full return type string.
    /// Used for array identity resolution on Call result values.
    pub return_type: Option<String>,
    /// Owner type for impl methods (e.g. "MyStruct" for `impl MyStruct { fn foo() }`).
    /// Used to construct the full CallPath for return_type registration.
    pub self_ty_root: Option<String>,
    /// RPython: function-level hints set by GC transformer / decorators.
    /// "close_stack" → _gctransformer_hint_close_stack_
    /// "cannot_collect" → _gctransformer_hint_cannot_collect_
    /// "gc_effects" → random_effects_on_gcobjs
    /// "elidable" → _elidable_function_
    /// "loopinvariant" → _jit_loop_invariant_
    pub hints: Vec<String>,
    /// RPython `graph.access_directly` (flowspace attribute set by the
    /// annotator's `default_specialize` rewrite — see
    /// `description.rs:1333-1335` + `pygraph.rs:53-56`). Carried into
    /// `SemanticFunction` so `policy::look_inside_graph` can port the
    /// `policy.py:71-83` virtualizable safety gate without reaching back
    /// into the PyGraph layer.
    ///
    /// Today every SemanticFunction produced by `build_function_graph`
    /// defaults to `false` because the `front::ast` parser does not yet
    /// consult the annotator result; when the annotator-to-front bridge
    /// lands, the bridge assigns this field from
    /// `PyGraph.access_directly.get()` for the matching graph.
    pub access_directly: bool,
}

/// RPython: struct field type info for `heaptracker.all_interiorfielddescrs`.
/// Maps struct_name → vec of (field_name, field_element_type).
/// `field_element_type` is the array element type when the field is an
/// array container (e.g. `Vec<Point>` → `"Point"`), or the full type
/// string for non-array fields.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StructFieldRegistry {
    /// struct_name → [(field_name, full_field_type_string)]
    pub fields: HashMap<String, Vec<(String, String)>>,
}

impl StructFieldRegistry {
    /// Look up a field's type.  For array-typed fields like
    /// `Vec<Point>`, this returns the full type string `"Vec<Point>"`.
    /// Callers use `array_element_type_from_str` to extract `"Point"`.
    ///
    /// **Structural adaptation (parity rule §1):** RPython compares
    /// `lltype.Ptr(GcStruct)` objects directly; field/method lookups
    /// resolve by structural type identity, not name.  Pyre's
    /// analyser carries types as strings (the only stable handle on
    /// `syn::Type`), so identity is recovered by string match.  A
    /// receiver may surface as a fully-qualified path that adds
    /// crate-level segments the analyser's `module_prefix` did not
    /// contain at registration time (e.g. `obj as *mut
    /// pyre_object::rangeobject::W_RangeIterator` while
    /// `collect_fields_and_returns` registered the struct under the
    /// shorter analyser-relative path).  Walk the trailing segments
    /// until a registered key matches.
    ///
    /// **Limitation:** the suffix walk does not disambiguate two
    /// structs that share a leaf name in different modules of the
    /// same parsed crate.  `collect_fields_and_returns` registers
    /// the bare name via `entry().or_insert`, so the first struct
    /// with a given leaf wins the bare-key slot and a suffix walk
    /// bottoming out there returns its fields regardless of which
    /// qualified owner the caller asked about.  The parsed-
    /// interpreter crate has no leaf-name collisions today; a future
    /// collision is a structural-correctness concern that must be
    /// re-evaluated rather than papered over.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        if let Some(fields) = self.fields.get(owner) {
            return fields
                .iter()
                .find(|(name, _)| name == field_name)
                .map(|(_, ty)| ty.as_str());
        }
        let mut tail = owner;
        while let Some(rest) = tail.split_once("::").map(|(_, rest)| rest) {
            if let Some(fields) = self.fields.get(rest) {
                return fields
                    .iter()
                    .find(|(name, _)| name == field_name)
                    .map(|(_, ty)| ty.as_str());
            }
            tail = rest;
        }
        None
    }
}

#[derive(Debug, Clone, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    pub known_struct_names: std::collections::HashSet<String>,
    /// Known trait names used to canonicalize local `dyn Trait` family keys.
    pub known_trait_names: std::collections::HashSet<String>,
    /// RPython: struct field types for resolving `op.args[0].concretetype`
    /// on FieldRead-produced array bases.
    pub struct_fields: StructFieldRegistry,
    /// RPython: op.result.concretetype — whole-program function return types.
    /// Maps exact callee path (e.g. "a::helper", "Type::method") → return type.
    /// Stored here so that downstream consumers (parse.rs method graph building)
    /// can use them for array type identity resolution.
    pub fn_return_types: HashMap<String, String>,
    /// RPython: `_immutable_fields_ = [...]` declared on a class body.
    /// Maps struct name → `(field_name, rank)` pairs whose value never
    /// mutates after construction (or is quasi-immutable).  Both bare and
    /// qualified struct keys are inserted (mirroring `struct_fields`) so
    /// the same lookup logic works across module-prefix variants.  Rank
    /// encoding follows `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
    pub immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>>,
}

pub fn build_semantic_program(parsed: &ParsedInterpreter) -> Result<SemanticProgram, FlowingError> {
    build_semantic_program_with_options(parsed, &AstGraphOptions::default())
}

pub fn build_semantic_program_from_parsed_files(
    parsed_files: &[ParsedInterpreter],
) -> Result<SemanticProgram, FlowingError> {
    build_semantic_program_from_parsed_files_with_options(parsed_files, &AstGraphOptions::default())
}

/// Qualify a bare type name with module prefix.
/// "Foo" with prefix "a" → "a::Foo". Already-qualified "a::Foo" unchanged.
/// Empty prefix → return bare name as-is.
pub(crate) fn qualify_type_name(bare: &str, prefix: &str) -> String {
    if prefix.is_empty() || bare.contains("::") {
        bare.to_string()
    } else {
        format!("{}::{}", prefix, bare)
    }
}

/// RPython: annotator whole-program type collection.
/// Recursively collects struct definitions, function return types, and impl
/// method return types from items, handling `mod` blocks with qualified paths.
/// The `prefix` carries the module path (e.g. "a::b") to produce exact callee
/// identities matching what `canonical_call_target` generates at call sites.
fn collect_types_from_items(
    items: &[Item],
    prefix: &str,
    known_struct_names: &mut std::collections::HashSet<String>,
    known_trait_names: &mut std::collections::HashSet<String>,
    struct_fields: &mut StructFieldRegistry,
    fn_return_types: &mut HashMap<String, String>,
    immutable_fields: &mut HashMap<String, Vec<(String, ImmutableRank)>>,
) {
    // RPython: annotator/rtyper resolves all types in a whole-program pass.
    // Two-pass: first collect ALL struct names, then field types + return types.
    // This ensures qualified_full_type_string can identify known structs
    // regardless of source order (RPython's lltype T.TO identity).
    collect_struct_names(items, prefix, known_struct_names);
    collect_trait_names(items, prefix, known_trait_names);
    collect_fields_and_returns(
        items,
        prefix,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    );
}

/// Read `#[jit_immutable_fields("a", "b?", "c[*]", "d?[*]")]` attributes
/// off a struct declaration and return the declared field names paired
/// with their `ImmutableRank`.  Bare idents (`#[jit_immutable_fields(a, b)]`)
/// remain accepted as `ImmutableRank::Immutable` for backward compatibility.
///
/// Multiple attributes accumulate; non-recognised tokens are silently
/// skipped (matching `syn::Meta::parse` looseness).  Rank suffix encoding
/// follows RPython `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
fn collect_immutable_field_attrs(attrs: &[syn::Attribute]) -> Vec<(String, ImmutableRank)> {
    use syn::punctuated::Punctuated;
    use syn::{Expr, ExprLit, ExprPath, Lit, Token};

    let mut specs = Vec::new();
    for attr in attrs {
        let Some(ident) = attr.path().get_ident() else {
            continue;
        };
        if ident != "jit_immutable_fields" {
            continue;
        }
        // Accept a comma-separated list of string literals and/or bare
        // idents:  `#[jit_immutable_fields("foo?", bar)]`.  String form
        // carries the RPython suffix; bare ident form is
        // `ImmutableRank::Immutable`.
        let parsed = attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated);
        let Ok(items) = parsed else {
            continue;
        };
        for item in items {
            match item {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => {
                    specs.push(ImmutableRank::parse(&s.value()));
                }
                Expr::Path(ExprPath { path, .. }) => {
                    if let Some(id) = path.get_ident() {
                        specs.push((id.to_string(), ImmutableRank::Immutable));
                    }
                }
                _ => {}
            }
        }
    }
    specs
}

/// Pass 1a: collect all struct names (bare + qualified) recursively.
fn collect_struct_names(
    items: &[Item],
    prefix: &str,
    known_struct_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let bare_name = s.ident.to_string();
                known_struct_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_struct_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_struct_names(sub_items, &mod_prefix, known_struct_names);
                }
            }
            _ => {}
        }
    }
}

fn collect_trait_names(
    items: &[Item],
    prefix: &str,
    known_trait_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            Item::Trait(trait_def) => {
                let bare_name = trait_def.ident.to_string();
                known_trait_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_trait_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_names(sub_items, &mod_prefix, known_trait_names);
                }
            }
            _ => {}
        }
    }
}

/// Pass 1b: collect field types + fn return types using known_struct_names.
fn collect_fields_and_returns(
    items: &[Item],
    prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    struct_fields: &mut StructFieldRegistry,
    fn_return_types: &mut HashMap<String, String>,
    immutable_fields: &mut HashMap<String, Vec<(String, ImmutableRank)>>,
) {
    for item in items {
        match item {
            Item::Struct(s) => {
                let bare_name = s.ident.to_string();
                // RPython: T.TO gives the actual lltype object.
                // qualified_full_type_string uses known_struct_names to identify
                // which inner types are user structs (not heuristic).
                let fields: Vec<(String, String)> = s
                    .fields
                    .iter()
                    .filter_map(|f| {
                        let field_name = f.ident.as_ref()?.to_string();
                        let field_type = qualified_full_type_string(
                            &f.ty,
                            prefix,
                            known_struct_names,
                            known_trait_names,
                        )?;
                        Some((field_name, field_type))
                    })
                    .collect();
                // RPython: `_immutable_fields_ = ['a', 'b']` on the class
                // body. We accept `#[jit_immutable_fields(a, b)]` on the
                // struct declaration (proc-macro pass-through in
                // `majit_macros::jit_immutable_fields`). Multiple
                // attributes accumulate.
                let immutables = collect_immutable_field_attrs(&s.attrs);
                if !immutables.is_empty() {
                    if prefix.is_empty() {
                        immutable_fields
                            .entry(bare_name.clone())
                            .or_default()
                            .extend(immutables.iter().cloned());
                    } else {
                        let qualified = format!("{}::{}", prefix, bare_name);
                        immutable_fields
                            .entry(qualified)
                            .or_default()
                            .extend(immutables.iter().cloned());
                        immutable_fields
                            .entry(bare_name.clone())
                            .or_default()
                            .extend(immutables.iter().cloned());
                    }
                }
                if prefix.is_empty() {
                    struct_fields.fields.insert(bare_name, fields);
                } else {
                    let qualified = format!("{}::{}", prefix, bare_name);
                    struct_fields.fields.insert(qualified, fields.clone());
                    struct_fields.fields.entry(bare_name).or_insert(fields);
                }
            }
            Item::Fn(func) => {
                // RPython: op.result.concretetype — module-qualified return type.
                let ret_ty = match &func.sig.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                        ty,
                        prefix,
                        known_struct_names,
                        known_trait_names,
                    ),
                    syn::ReturnType::Default => Some("()".to_string()),
                };
                if let Some(ret_ty) = ret_ty {
                    let key = if prefix.is_empty() {
                        func.sig.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, func.sig.ident)
                    };
                    fn_return_types.insert(key, ret_ty);
                }
            }
            Item::Impl(impl_block) => {
                let self_ty_root = type_root_ident(&impl_block.self_ty);
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let ret_ty = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                                ty,
                                prefix,
                                known_struct_names,
                                known_trait_names,
                            ),
                            syn::ReturnType::Default => Some("()".to_string()),
                        };
                        if let Some(ret_ty) = ret_ty {
                            if let Some(ref ty_root) = self_ty_root {
                                let qualified_ty = qualify_type_name(ty_root, prefix);
                                fn_return_types.insert(
                                    format!("{}::{}", qualified_ty, method.sig.ident),
                                    ret_ty,
                                );
                            }
                        }
                    }
                }
            }
            Item::Trait(trait_def) => {
                let trait_root = qualify_type_name(&trait_def.ident.to_string(), prefix);
                for sub in &trait_def.items {
                    if let syn::TraitItem::Fn(method) = sub {
                        let ret_ty = match &method.sig.output {
                            syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                                ty,
                                prefix,
                                known_struct_names,
                                known_trait_names,
                            ),
                            syn::ReturnType::Default => Some("()".to_string()),
                        };
                        if let Some(ret_ty) = ret_ty {
                            fn_return_types
                                .insert(format!("{}::{}", trait_root, method.sig.ident), ret_ty);
                        }
                    }
                }
            }
            Item::Enum(e) => {
                // RPython sum types are multiple subclasses inheriting from a
                // common ancestor; each subclass owns its own
                // `concretetype` field set keyed by the lltype object
                // identity.  Pyre carries identity as a flat string
                // table, and the only stable identity for a variant is
                // the fully-qualified `prefix::Enum::Variant` path.
                // Earlier drafts also registered the bare `Variant`
                // fallback and the bare-enum `Enum::Variant` (without
                // module prefix), but both forms collided across
                // unrelated enums with the same variant name (e.g.
                // `a::Foo::Empty` vs `b::Foo::Empty`, or
                // `Foo::Empty` vs `Bar::Empty`) — `field_type`'s suffix
                // walk would then bind the wrong field set.  Register
                // only the fully-qualified key.  The walk still
                // resolves shorter callers like
                // `majit_ir::RdVirtualInfo::VirtualInfo` →
                // `RdVirtualInfo::VirtualInfo` because the strip
                // happens left-to-right on the caller's owner string,
                // and the registered fully-qualified key is matched
                // when the caller's tail equals it.  Tuple/unit
                // variants carry no named fields and need no entry.
                let bare_enum = e.ident.to_string();
                let qualified_enum = if prefix.is_empty() {
                    bare_enum.clone()
                } else {
                    format!("{}::{}", prefix, bare_enum)
                };
                for variant in &e.variants {
                    if let syn::Fields::Named(named) = &variant.fields {
                        let var_name = variant.ident.to_string();
                        let fields: Vec<(String, String)> = named
                            .named
                            .iter()
                            .filter_map(|f| {
                                let field_name = f.ident.as_ref()?.to_string();
                                let field_type = qualified_full_type_string(
                                    &f.ty,
                                    prefix,
                                    known_struct_names,
                                    known_trait_names,
                                )?;
                                Some((field_name, field_type))
                            })
                            .collect();
                        struct_fields
                            .fields
                            .insert(format!("{}::{}", qualified_enum, var_name), fields);
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_fields_and_returns(
                        sub_items,
                        &mod_prefix,
                        known_struct_names,
                        known_trait_names,
                        struct_fields,
                        fn_return_types,
                        immutable_fields,
                    );
                }
            }
            _ => {}
        }
    }
}

/// RPython: pass 2 graph building with Item::Mod recursion.
/// Mirrors collect_types_from_items traversal so that module-internal
/// functions get proper SemanticFunction entries with qualified names.
///
/// RPython `flowspace/objspace.py:49` + `flowspace/flowcontext.py:417`
/// + `translator/translator.py:55` — `build_flow()` / `buildflowgraph()`
/// re-raise `FlowingError`, and the top-level translator observes the
/// unsupported construct as a hard failure.  This batch collector
/// propagates `FlowingError` the same way rather than silently dropping
/// a graph whose body hit an unsupported construct.
fn build_graphs_from_items(
    items: &[Item],
    prefix: &str,
    options: &AstGraphOptions,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    functions: &mut Vec<SemanticFunction>,
) -> Result<(), FlowingError> {
    for item in items {
        match item {
            Item::Fn(func) => {
                let mut sf = build_function_graph(
                    func,
                    options,
                    None,
                    struct_fields,
                    fn_return_types,
                    prefix,
                    known_struct_names,
                    known_trait_names,
                )?;
                // RPython: exact graph identity — module-qualified name.
                if !prefix.is_empty() {
                    sf.name = format!("{}::{}", prefix, sf.name);
                }
                functions.push(sf);
            }
            Item::Impl(impl_block) => {
                // Qualify bare self type with module prefix (RPython: unique type identity).
                let self_ty_root =
                    type_root_ident(&impl_block.self_ty).map(|t| qualify_type_name(&t, prefix));
                for sub in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = sub {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        let sf = build_function_graph(
                            &fake_fn,
                            options,
                            self_ty_root.clone(),
                            struct_fields,
                            fn_return_types,
                            prefix,
                            known_struct_names,
                            known_trait_names,
                        )?;
                        functions.push(sf);
                    }
                }
            }
            Item::Mod(m) => {
                if let Some((_, ref items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    build_graphs_from_items(
                        items,
                        &mod_prefix,
                        options,
                        struct_fields,
                        fn_return_types,
                        known_struct_names,
                        known_trait_names,
                        functions,
                    )?;
                }
            }
            _ => {}
        }
    }
    Ok(())
}

pub fn build_semantic_program_with_options(
    parsed: &ParsedInterpreter,
    options: &AstGraphOptions,
) -> Result<SemanticProgram, FlowingError> {
    let mut functions = Vec::new();
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();

    // Pass 1: collect all struct definitions and function return types.
    // RPython: annotator/rtyper resolves all types in a whole-program pass.
    // We recursively traverse Item::Mod to register module-qualified paths
    // matching the exact callee identity that canonical_call_target produces.
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
    collect_types_from_items(
        &parsed.file.items,
        "",
        &mut known_struct_names,
        &mut known_trait_names,
        &mut struct_fields,
        &mut fn_return_types,
        &mut immutable_fields,
    );

    // Pass 2: build function graphs with struct_fields + fn_return_types.
    // Field types are already module-qualified at the source (via
    // qualified_full_type_string), matching RPython's lltype identity.
    build_graphs_from_items(
        &parsed.file.items,
        "",
        options,
        &struct_fields,
        &fn_return_types,
        &known_struct_names,
        &known_trait_names,
        &mut functions,
    )?;

    Ok(SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    })
}

pub fn build_semantic_program_from_parsed_files_with_options(
    parsed_files: &[ParsedInterpreter],
    options: &AstGraphOptions,
) -> Result<SemanticProgram, FlowingError> {
    // RPython: annotator/rtyper provides whole-program type info before
    // the codewriter runs. We emulate this with a 2-pass approach:
    // Pass 1: collect ALL struct definitions and function return types across ALL files.
    // Uses collect_types_from_items to handle Item::Mod recursively with
    // qualified paths matching canonical_call_target identity.
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
    // RPython: whole-program — ALL types visible everywhere.
    // Collect struct names from ALL files first, then fields+returns.
    for parsed in parsed_files {
        collect_struct_names(&parsed.file.items, "", &mut known_struct_names);
        collect_trait_names(&parsed.file.items, "", &mut known_trait_names);
    }
    for parsed in parsed_files {
        collect_fields_and_returns(
            &parsed.file.items,
            "",
            &known_struct_names,
            &known_trait_names,
            &mut struct_fields,
            &mut fn_return_types,
            &mut immutable_fields,
        );
    }
    // Pass 2: build function graphs with merged struct_fields + fn_return_types visible.
    // Field types already module-qualified at source (qualified_full_type_string).
    let mut functions = Vec::new();
    for parsed in parsed_files {
        build_graphs_from_items(
            &parsed.file.items,
            "",
            options,
            &struct_fields,
            &fn_return_types,
            &known_struct_names,
            &known_trait_names,
            &mut functions,
        )?;
    }
    Ok(SemanticProgram {
        functions,
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
        immutable_fields,
    })
}

/// Public entry for building a graph from a single function AST node.
/// Lower a standalone expression into an existing graph.
/// Used to build semantic graphs from opcode match arm bodies.
///
/// RPython `flowspace/objspace.py:38` — `build_flow()` re-raises
/// `FlowingError` so callers observe the unsupported construct as an
/// error rather than receiving a partially-constructed graph.  The
/// `Unknown` marker op that `stop_unsupported` already emitted stays in
/// the graph; the caller decides whether to keep, discard, or close it.
pub fn lower_expr_into_graph(
    graph: &mut FunctionGraph,
    expr: &syn::Expr,
) -> Result<(), FlowingError> {
    let mut block = graph.startblock;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    );
    let lowered = lower_expr(
        graph,
        &mut block,
        expr,
        &AstGraphOptions::default(),
        &mut ctx,
    )?;
    if graph.block(block).is_open() {
        graph.set_return(block, lowered.value);
    }
    Ok(())
}

pub fn build_function_graph_pub(func: &ItemFn) -> Result<SemanticFunction, FlowingError> {
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        None,
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    )
}

pub fn build_function_graph_with_self_ty_pub(
    func: &ItemFn,
    self_ty_root: Option<String>,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    module_prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Result<SemanticFunction, FlowingError> {
    build_function_graph(
        func,
        &AstGraphOptions::default(),
        self_ty_root,
        struct_fields,
        fn_return_types,
        module_prefix,
        known_struct_names,
        known_trait_names,
    )
}

/// Expose `collect_jit_hints` so `parse.rs` can hoist trait-method
/// hints from AST attributes when the strict graph build returns Err.
pub fn collect_jit_hints_pub(attrs: &[syn::Attribute]) -> Vec<String> {
    collect_jit_hints(attrs)
}

#[derive(Debug, Clone)]
struct GraphBuildContext<'a> {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    local_trait_bound_roots: HashMap<String, String>,
    generic_trait_roots: HashMap<String, String>,
    /// RPython: ARRAY element type identity — maps variable name to the
    /// element type of its array (e.g. "arr" → "Point" for `arr: Vec<Point>`).
    /// This is the Rust equivalent of RPython's `GcArray(T)` where T is the
    /// element type that determines the ARRAY identity for `cpu.arraydescrof()`.
    local_array_types: HashMap<String, String>,
    /// Receiver-trait lookup for locals/parameters bound to `&mut dyn T`
    /// / `&dyn T` / `Box<dyn T>` / `dyn T`.  Populated at let-statement
    /// and fn-parameter binding time; consumed by `dyn_trait_for_receiver`
    /// so method-call lowering can emit `CallTarget::Indirect`
    /// (`jtransform.py:410-412`).
    local_dyn_trait_roots: HashMap<String, String>,
    /// RPython: program-level struct field types, available for resolving
    /// field access array identity (e.g. `self.array[i]` → owner.field_type).
    struct_fields: &'a StructFieldRegistry,
    /// RPython: op.result.concretetype — function return types from the annotator.
    /// Maps function name (or "Type::method") → return type string.
    /// Used by array_type_id_from_expr to resolve Call/MethodCall expressions.
    fn_return_types: &'a HashMap<String, String>,
    /// Module path prefix for qualifying bare type names.
    /// RPython: lltype identity is globally unique — bare "Foo" in mod "a"
    /// must resolve to "a::Foo" in struct_fields lookups.
    module_prefix: String,
    known_struct_names: &'a std::collections::HashSet<String>,
    known_trait_names: &'a std::collections::HashSet<String>,
    /// Loop targets active at the current lowering point.  Pushed on
    /// entry to `Loop` / `While` / `ForLoop` and popped after the body
    /// is walked.  `break` closes the current block with a goto to the
    /// innermost `break_target`; `continue` goes to `continue_target`.
    /// RPython: `flowspace/flowcontext.py:525` BreakLoop signal +
    /// `:1341` LoopBlock.handle_signal dispatches to end/header.
    loop_stack: Vec<LoopFrame>,
}

#[derive(Debug, Clone, Copy)]
struct LoopFrame {
    /// Block that `continue` jumps to.  For `while` / `for` this is
    /// the header; for `loop` this is the body entry (which also acts
    /// as the loop head).
    continue_target: BlockId,
    /// Block that `break` jumps to — the loop's exit block.
    break_target: BlockId,
}

impl<'a> GraphBuildContext<'a> {
    fn new(
        struct_fields: &'a StructFieldRegistry,
        fn_return_types: &'a HashMap<String, String>,
        module_prefix: &str,
        known_struct_names: &'a std::collections::HashSet<String>,
        known_trait_names: &'a std::collections::HashSet<String>,
    ) -> Self {
        Self {
            local_type_roots: HashMap::new(),
            local_type_strings: HashMap::new(),
            local_value_types: HashMap::new(),
            local_trait_bound_roots: HashMap::new(),
            generic_trait_roots: HashMap::new(),
            local_array_types: HashMap::new(),
            local_dyn_trait_roots: HashMap::new(),
            struct_fields,
            fn_return_types,
            module_prefix: module_prefix.to_string(),
            known_struct_names,
            known_trait_names,
            loop_stack: Vec::new(),
        }
    }
}

struct LocalBindingSnapshot {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    local_trait_bound_roots: HashMap<String, String>,
    local_array_types: HashMap<String, String>,
    local_dyn_trait_roots: HashMap<String, String>,
}

impl LocalBindingSnapshot {
    fn capture(ctx: &GraphBuildContext<'_>) -> Self {
        Self {
            local_type_roots: ctx.local_type_roots.clone(),
            local_type_strings: ctx.local_type_strings.clone(),
            local_value_types: ctx.local_value_types.clone(),
            local_trait_bound_roots: ctx.local_trait_bound_roots.clone(),
            local_array_types: ctx.local_array_types.clone(),
            local_dyn_trait_roots: ctx.local_dyn_trait_roots.clone(),
        }
    }

    fn restore(self, ctx: &mut GraphBuildContext<'_>) {
        ctx.local_type_roots = self.local_type_roots;
        ctx.local_type_strings = self.local_type_strings;
        ctx.local_value_types = self.local_value_types;
        ctx.local_trait_bound_roots = self.local_trait_bound_roots;
        ctx.local_array_types = self.local_array_types;
        ctx.local_dyn_trait_roots = self.local_dyn_trait_roots;
    }
}

/// Build a SemanticFunction from a Rust function AST.  Mirrors
/// RPython `flowspace/objspace.py:38` `build_flow()` — `FlowingError`
/// propagates to the caller rather than producing a partial graph.
thread_local! {
    /// MAJIT_UNKNOWN_DUMP diagnostic context: name of the function
    /// currently being lowered. Set on `build_function_graph` entry
    /// and restored on exit so the per-`syn::Expr` Unknown emit sites
    /// can attribute their stub to the source function. Read-only
    /// elsewhere — purely cosmetic for the dump output.
    static CURRENT_LOWERING_FN_NAME: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

/// RAII guard for `CURRENT_LOWERING_FN_NAME` — restores the previous
/// fn name on Drop so a `?` early-exit inside `build_function_graph`
/// still leaves the thread-local in a sane state for sibling lowerings.
struct LoweringFnNameGuard {
    previous: Option<String>,
}

impl Drop for LoweringFnNameGuard {
    fn drop(&mut self) {
        let prev = self.previous.take();
        CURRENT_LOWERING_FN_NAME.with(|c| *c.borrow_mut() = prev);
    }
}

fn build_function_graph(
    func: &ItemFn,
    options: &AstGraphOptions,
    self_ty_root: Option<String>,
    struct_fields: &StructFieldRegistry,
    fn_return_types: &HashMap<String, String>,
    module_prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Result<SemanticFunction, FlowingError> {
    let fn_name = func.sig.ident.to_string();
    let previous = CURRENT_LOWERING_FN_NAME.with(|c| c.borrow_mut().replace(fn_name.clone()));
    let _restore_fn = LoweringFnNameGuard { previous };
    let mut graph = FunctionGraph::new(fn_name);
    let mut entry = graph.startblock;
    let mut ctx = GraphBuildContext::new(
        struct_fields,
        fn_return_types,
        module_prefix,
        known_struct_names,
        known_trait_names,
    );
    ctx.generic_trait_roots =
        collect_generic_trait_roots(&func.sig.generics, module_prefix, known_trait_names);

    // Register function parameters as Input ops AND on `Block.inputargs`.
    //
    // RPython parity: `Block.inputargs` is the function's formal parameter
    // list for the startblock (`flowspace/model.py` Block class).  Pyre
    // originally only emitted `OpKind::Input` ops here — but because body
    // `Expr::Path` lowering also emits `OpKind::Input` for plain variable
    // references (`front/ast.rs:1271-1287`), counting startblock `Input`
    // ops after lowering can no longer tell "parameter" from "body
    // reference" apart.  Populating `inputargs` during parameter
    // registration preserves the RPython `startblock.inputargs == params`
    // invariant and is what `getcalldescr`'s `FUNC.ARGS` check reads
    // (RPython `call.py:220-221`).
    for param in &func.sig.inputs {
        match param {
            syn::FnArg::Receiver(recv) => {
                if let Some(self_ty_root) = &self_ty_root {
                    ctx.local_type_roots
                        .insert("self".to_string(), self_ty_root.clone());
                    ctx.local_type_strings
                        .insert("self".to_string(), self_ty_root.clone());
                }
                // `self`, `&self`, `&mut self` — all three correspond to
                // an `lltype.Ptr(<Self>)` register in RPython, so the
                // formal parameter always lands in the Ref class.
                let self_ty = classify_fn_arg_ty(&recv.ty);
                ctx.local_value_types
                    .insert("self".to_string(), self_ty.clone());
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: "self".to_string(),
                        ty: self_ty,
                    },
                    true,
                ) {
                    graph.name_value(vid, "self".to_string());
                    graph.block_mut(entry).inputargs.push(vid);
                }
            }
            syn::FnArg::Typed(pat_type) => {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    // Qualify bare type with module prefix for exact identity.
                    let qualified = qualify_type_name(&type_root, &ctx.module_prefix);
                    ctx.local_type_roots.insert(name.clone(), qualified);
                    if let Some(trait_root) = ctx.generic_trait_roots.get(&type_root) {
                        ctx.local_trait_bound_roots
                            .insert(name.clone(), trait_root.clone());
                    }
                }
                if let Some(full_type) = qualified_full_type_string(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_struct_names,
                    ctx.known_trait_names,
                ) {
                    ctx.local_type_strings
                        .insert(name.clone(), full_type.clone());
                    ctx.local_array_types.insert(name.clone(), full_type);
                }
                if let Some(trait_root) = extract_dyn_trait_root_with_context(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_trait_names,
                ) {
                    ctx.local_dyn_trait_roots.insert(name.clone(), trait_root);
                }
                // RPython `rpython/jit/codewriter/support.py:getkind`
                // mapping: classify the Rust parameter type to one of
                // the three register classes so the annotator + rtyper
                // receive a non-`Unknown` seed. Upstream's rtyper
                // assigns a concretetype to every `Variable`, and
                // `assembler.py:write_insn` relies on every operand
                // having a coloring. Using `ValueType::Unknown` here
                // used to cascade into `build_value_kinds` dropping
                // the value, which produced the `(0, 'i')` fallback at
                // `lookup_reg_with_kind` — the source of the pyre-only
                // `getfield_gc_*/id>*` `_intbase` aliases.
                let arg_ty = classify_fn_arg_ty(&pat_type.ty);
                ctx.local_value_types.insert(name.clone(), arg_ty.clone());
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: arg_ty,
                    },
                    true,
                ) {
                    graph.name_value(vid, name);
                    graph.block_mut(entry).inputargs.push(vid);
                }
            }
        }
    }

    // Lower function body.  RPython `flowspace/flowcontext.py` stops
    // abstract-interpreting the current graph on `FlowingError`
    // (unsupported opcode) — the exception propagates out of
    // `build_flow()` (`flowspace/objspace.py:38`) so the translator
    // observes the failure instead of receiving a partial graph.  A
    // path-closing `FlowSignal::Return` / `Raise` at the top level is
    // orderly termination: after `return x` there's nothing more to
    // walk but the graph is well-formed, so we break without
    // propagating.
    let lowered = lower_stmt_list_with_tail_value(
        &mut graph,
        &mut entry,
        &func.block.stmts,
        options,
        &mut ctx,
    )?;

    // Default terminator if none was set. RPython `RETURN_VALUE`
    // carries the evaluated tail expression into `graph.returnblock`;
    // only statement-only / empty bodies synthesize the void return
    // value.
    if !lowered.path_closed && graph.block(entry).is_open() {
        graph.set_return(entry, lowered.value);
    }

    // RPython: op.result.concretetype — module-qualified for exact type identity.
    let return_type = match &func.sig.output {
        syn::ReturnType::Type(_, ty) => {
            qualified_full_type_string(ty, module_prefix, known_struct_names, known_trait_names)
        }
        syn::ReturnType::Default => Some("()".to_string()),
    };

    // RPython: function-level hints from decorators / GC transformer.
    // Scan #[jit_*] attributes to detect elidable, loopinvariant,
    // close_stack, cannot_collect, gc_effects.
    let hints = collect_jit_hints(&func.attrs);

    Ok(SemanticFunction {
        name: func.sig.ident.to_string(),
        graph,
        return_type,
        self_ty_root,
        hints,
        access_directly: false,
    })
}

/// RPython: extract function-level JIT hints from attributes.
/// Maps JIT hint attributes to effectinfo classification strings.
///
/// Recognizes both legacy `jit_*` and RPython-parity names.
///
/// For `#[oopspec("spec")]`, returns `"oopspec:spec_string"` so the hint
/// consumer can extract the spec via `hint.strip_prefix("oopspec:")`.
fn collect_jit_hints(attrs: &[syn::Attribute]) -> Vec<String> {
    let mut hints = Vec::new();
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            match name.as_str() {
                // RPython-parity names (rlib/jit.py)
                "elidable" | "jit_elidable" => hints.push("elidable".into()),
                "elidable_promote" => hints.push("elidable".into()),
                "dont_look_inside" => hints.push("dont_look_inside".into()),
                "unroll_safe" => hints.push("unroll_safe".into()),
                "loop_invariant" | "jit_loop_invariant" => {
                    hints.push("loopinvariant".into());
                }
                "not_in_trace" => hints.push("not_in_trace".into()),
                // rlib/jit.py:250 — `@oopspec(spec)`: extract spec string.
                "oopspec" => {
                    if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                        hints.push(format!("oopspec:{}", lit.value()));
                    } else {
                        hints.push("oopspec".into());
                    }
                }
                // majit-specific
                "jit_close_stack" => hints.push("close_stack".into()),
                "jit_cannot_collect" => hints.push("cannot_collect".into()),
                "jit_gc_effects" => hints.push("gc_effects".into()),
                _ => {}
            }
        }
    }
    hints
}

// ── Statement lowering ──────────────────────────────────────────

/// Public entry point for lowering a single statement into a graph.
/// Used by the graph-based classifier in lib.rs to analyze resolved method bodies.
///
/// RPython `flowspace/objspace.py:38` — `FlowingError` propagates.  The
/// caller is responsible for handling the unsupported-construct signal
/// (typically by discarding the partially-built graph).  The boolean
/// result mirrors `lower_stmt`: `true` means the path terminated
/// (return/break/continue/raise) and the enclosing walker should stop.
pub fn lower_stmt_pub(
    graph: &mut FunctionGraph,
    block: BlockId,
    stmt: &syn::Stmt,
) -> Result<bool, FlowingError> {
    let mut block = block;
    let empty_registry = StructFieldRegistry::default();
    let empty_fn_ret = HashMap::new();
    let empty_names = std::collections::HashSet::new();
    let empty_trait_names = std::collections::HashSet::new();
    let mut ctx = GraphBuildContext::new(
        &empty_registry,
        &empty_fn_ret,
        "",
        &empty_names,
        &empty_trait_names,
    );
    lower_stmt(
        graph,
        &mut block,
        stmt,
        &AstGraphOptions::default(),
        &mut ctx,
    )
}

/// Lower a sequence of statements whose final element may be a tail
/// expression (Rust block-value form: `{ stmt; stmt; expr }`).
///
/// RPython flow-space guarantee: every source expression is walked
/// exactly once (`rpython/flowspace/flowcontext.py::FlowContext.record`
/// appends each bytecode op once). Rust `syn::Block` / `ExprBlock` /
/// `ExprUnsafe` / `ExprIf.then_branch` all carry `Vec<Stmt>` with an
/// optional `Stmt::Expr(_, None)` tail whose value becomes the block's
/// value — lowering that tail via both `lower_stmt` (which delegates to
/// `lower_expr`) and a second `lower_expr` call would emit the op
/// twice and break the "walk once" invariant.
fn lower_stmt_list_with_tail_value(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmts: &[syn::Stmt],
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<Lowered, FlowingError> {
    let Some((last, prefix)) = stmts.split_last() else {
        return Ok(Lowered::no_value());
    };
    // Prefix stmts: walk each; if one closes the path
    // (`return x;`, `panic!();`, ...), stop — remaining stmts are
    // unreachable, mirroring RPython `flowspace/flowcontext.py`'s
    // `FlowSignal` propagation where `Return`/`Raise` halts the
    // current recorder before the next bytecode runs.
    for stmt in prefix {
        let path_closed = lower_stmt(graph, block, stmt, options, ctx)?;
        if path_closed {
            return Ok(Lowered::path_closed());
        }
    }
    match last {
        syn::Stmt::Expr(expr, None) => lower_expr(graph, block, expr, options, ctx),
        _ => {
            let path_closed = lower_stmt(graph, block, last, options, ctx)?;
            Ok(Lowered {
                value: None,
                path_closed,
            })
        }
    }
}

fn lower_stmt(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<bool, FlowingError> {
    match stmt {
        syn::Stmt::Expr(expr, _) => {
            let lowered = lower_expr(graph, block, expr, options, ctx)?;
            return Ok(lowered.path_closed);
        }
        syn::Stmt::Local(local) => {
            // RPython: rtyper assigns concretetype to let-bound variables.
            // Extract array element type from type annotations on let bindings.
            if let syn::Pat::Type(pat_type) = &local.pat {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    let qualified = qualify_type_name(&type_root, &ctx.module_prefix);
                    ctx.local_type_roots.insert(name.clone(), qualified);
                }
                ctx.local_value_types
                    .insert(name.clone(), classify_fn_arg_ty(&pat_type.ty));
                if let Some(full_type) = qualified_full_type_string(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_struct_names,
                    ctx.known_trait_names,
                ) {
                    ctx.local_type_strings
                        .insert(name.clone(), full_type.clone());
                    ctx.local_array_types.insert(name.clone(), full_type);
                }
                if let Some(trait_root) = extract_dyn_trait_root_with_context(
                    &pat_type.ty,
                    &ctx.module_prefix,
                    ctx.known_trait_names,
                ) {
                    ctx.local_dyn_trait_roots.insert(name.clone(), trait_root);
                }
            }
            if let Some(init) = &local.init {
                let init_type_string = infer_init_type_string(&init.expr, ctx);
                let lowered = lower_expr(graph, block, &init.expr, options, ctx)?;
                if lowered.path_closed {
                    return Ok(true);
                }
                // Record variable name (RPython Variable._name)
                if let Some(vid) = lowered.value {
                    let name = if let syn::Pat::Ident(pat_ident) = &local.pat {
                        Some(pat_ident.ident.to_string())
                    } else if let syn::Pat::Type(pat_type) = &local.pat {
                        Some(canonical_pat_name(&pat_type.pat))
                    } else {
                        None
                    };
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        graph.name_value(vid, pat_ident.ident.to_string());
                    } else if let syn::Pat::Type(pat_type) = &local.pat {
                        let name = canonical_pat_name(&pat_type.pat);
                        graph.name_value(vid, name);
                    }
                    if let Some(name) = name {
                        if let Some(ty) = graph_value_type(graph, vid) {
                            ctx.local_value_types.insert(name.clone(), ty);
                        }
                        if let Some(type_string) = init_type_string {
                            // Mirror `bind_ident_type` on let-with-annotation:
                            // record the receiver root so subsequent
                            // `receiver_type_root` lookups for field access
                            // can resolve `(*x).field` against
                            // `ctx.struct_fields`.  Without this, lets bound
                            // by inference from a Cast/Call init lose their
                            // owner root and field reads land with `ty:
                            // Unknown` → cast arms fire downstream.
                            if let Some(root) = type_root_from_type_string(&type_string) {
                                ctx.local_type_roots.insert(name.clone(), root);
                            }
                            ctx.local_type_strings.insert(name, type_string);
                        }
                    } else if !matches!(&local.pat, syn::Pat::Ident(_) | syn::Pat::Type(_)) {
                        // Destructure let (`let Some(x) = ...;`,
                        // `let Foo { a, b } = ...;`, `let A | B { f, .. }
                        // = ...;`) introduces names that the simple
                        // Pat::Ident / Pat::Type binding above misses.
                        // RPython parity: `flowspace/flowcontext.py` walks
                        // the BUILD_TUPLE_UNPACK / unpack_sequence paths
                        // and binds each leaf name with its rtyped
                        // concretetype.  Pyre routes the same shape
                        // through `bind_pattern_locals`, which already
                        // unwraps `Some(_)` / `Ok(_)` / `Err(_)` and
                        // recurses into struct / or patterns.
                        bind_pattern_locals(&local.pat, init_type_string.as_deref(), ctx);
                    }
                }
            }
        }
        syn::Stmt::Macro(stmt_macro) => {
            // Rust macros are syntactic, not part of the flow graph —
            // RPython has no construct counterpart.  Only forward
            // macros whose Rust semantics have an explicit RPython
            // mapping through `lower_expr`:
            //   * abort-family (`panic!`, `unreachable!`, `todo!`,
            //     `unimplemented!`) → `set_raise` (canonical
            //     exceptblock Link per `flowspace/model.py:21-25`).
            //   * assert-family (`assert!`, `assert_eq!`, `assert_ne!`,
            //     and `debug_` variants) → conditional `set_branch` +
            //     `set_raise` on the failing arm.
            // Other statement-position macros (`dbg!`, `println!`,
            // `vec!`, `format!`, `write!`, `writeln!`, ...) are
            // skipped, matching the pre-`92725722af` behaviour.
            let name = stmt_macro
                .mac
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if matches!(
                name.as_str(),
                "panic"
                    | "unreachable"
                    | "todo"
                    | "unimplemented"
                    | "assert"
                    | "assert_eq"
                    | "assert_ne"
                    | "debug_assert"
                    | "debug_assert_eq"
                    | "debug_assert_ne"
            ) {
                let expr_macro = syn::ExprMacro {
                    attrs: stmt_macro.attrs.clone(),
                    mac: stmt_macro.mac.clone(),
                };
                let expr = syn::Expr::Macro(expr_macro);
                let lowered = lower_expr(graph, block, &expr, options, ctx)?;
                return Ok(lowered.path_closed);
            }
        }
        syn::Stmt::Item(_) => {}
    }
    Ok(false)
}

// ── Expression lowering (block-splitting for control flow) ───────

/// Lower an expression, potentially splitting blocks for control flow.
///
/// RPython equivalent: FlowContext.handle_bytecode() + guessbool().
/// When `if`/`match` is encountered, the current block is terminated
/// with a Branch, new blocks are created for each arm, and `block`
/// is updated to the merge/continuation block.
fn lower_expr(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    expr: &syn::Expr,
    options: &AstGraphOptions,
    ctx: &mut GraphBuildContext,
) -> Result<Lowered, FlowingError> {
    // RPython `flowspace/flowcontext.py:258,417` — when the abstract
    // interpreter hits an unsupported bytecode it raises `FlowingError`
    // and the walk stops at once.  Pyre's analogue: emit an
    // `UnsupportedExpr` marker op in *block (so downstream passes see
    // evidence of the drop) and return `Err(FlowingError::Unsupported)`
    // so every caller in the chain aborts via `?` rather than
    // synthesising a fabricated SSA value.  The helper centralises
    // that pair so every failure site emits exactly one Unknown.
    let stop_unsupported = |graph: &mut FunctionGraph,
                            block: BlockId,
                            variant: UnsupportedExprKind|
     -> Result<Lowered, FlowingError> {
        graph.push_op(
            block,
            OpKind::Abort {
                kind: UnknownKind::UnsupportedExpr { variant },
            },
            true,
        );
        Err(FlowingError::Unsupported {
            kind: UnknownKind::UnsupportedExpr { variant },
        })
    };
    // Non-fatal counterpart of `stop_unsupported`: emit the `Unknown`
    // marker so coverage auditing still flags the gap, but hand its
    // ValueId back so the enclosing walker keeps going.  Matches
    // RPython `LOAD_CONST` (`flowspace/flowcontext.py:841`) — the
    // bytecode pushes a value of an un-modelled shape and the flow
    // walk continues without raising `FlowingError`.
    let continue_with_unknown =
        |graph: &mut FunctionGraph, block: BlockId, variant: UnsupportedExprKind| -> Lowered {
            let v = graph.push_op(
                block,
                OpKind::Abort {
                    kind: UnknownKind::UnsupportedExpr { variant },
                },
                true,
            );
            Lowered {
                value: v,
                path_closed: false,
            }
        };
    let continue_with_unknown_literal =
        |graph: &mut FunctionGraph, block: BlockId, variant: UnsupportedLiteralKind| -> Lowered {
            let v = graph.push_op(
                block,
                OpKind::Abort {
                    kind: UnknownKind::UnsupportedLiteral { variant },
                },
                true,
            );
            Lowered {
                value: v,
                path_closed: false,
            }
        };
    match expr {
        // ── receiver.field / arr[i].field ──
        syn::Expr::Field(field) => {
            if let syn::Expr::Index(idx) = &*field.base {
                // RPython: getinteriorfield_gc — arr[i].field as a single op.
                let base = get_value!(lower_expr(graph, block, &idx.expr, options, ctx)?);
                let index = get_value!(lower_expr(graph, block, &idx.index, options, ctx)?);
                let field_name = member_name(&field.member);
                let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                // Element struct type is the field owner for interiorfield descriptors.
                let elem_type = array_type_id
                    .as_ref()
                    .and_then(|atid| extract_element_type_from_str(atid));
                // RPython: getkind(op.result.concretetype) — resolve field type
                // from struct field registry for the kind suffix (i/r/f).
                let item_ty = elem_type
                    .as_ref()
                    .and_then(|owner| ctx.struct_fields.field_type(owner, &field_name))
                    .map(type_string_to_value_type)
                    .unwrap_or(ValueType::Unknown);
                Ok(Lowered {
                    value: graph.push_op(
                        *block,
                        OpKind::InteriorFieldRead {
                            base,
                            index,
                            field: crate::model::FieldDescriptor::new(field_name, elem_type),
                            item_ty,
                            array_type_id,
                        },
                        true,
                    ),
                    path_closed: false,
                })
            } else {
                let base = get_value!(lower_expr(graph, block, &field.base, options, ctx)?);
                let field_name = member_name(&field.member);
                let ty = field_value_type_from_expr(&field.base, &field.member, ctx)
                    .unwrap_or(ValueType::Unknown);
                Ok(Lowered {
                    value: graph.push_op(
                        *block,
                        OpKind::FieldRead {
                            base,
                            field: crate::model::FieldDescriptor::new(
                                field_name,
                                receiver_type_root(&field.base, ctx),
                            ),
                            ty,
                            pure: false,
                        },
                        true,
                    ),
                    path_closed: false,
                })
            }
        }

        // ── base[index] ──
        syn::Expr::Index(idx) => {
            let base = get_value!(lower_expr(graph, block, &idx.expr, options, ctx)?);
            let index = get_value!(lower_expr(graph, block, &idx.index, options, ctx)?);
            let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
            let item_ty = array_item_value_type_from_array_type_id(array_type_id.as_deref())
                .unwrap_or(ValueType::Unknown);
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::ArrayRead {
                        base,
                        index,
                        item_ty,
                        array_type_id,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── lhs = rhs ──
        syn::Expr::Assign(assign) => {
            // RPython `flowcontext.py` evaluates rhs first; if it raises
            // `FlowingError`, the whole assignment is dropped.  `get_value!`
            // propagates both `FlowingError` (`Err(..)`) and `path_closed`
            // (`Ok(Lowered { path_closed: true })`) up the walk.
            let value = get_value!(lower_expr(graph, block, &assign.right, options, ctx)?);

            match &*assign.left {
                syn::Expr::Field(field) => {
                    if let syn::Expr::Index(idx) = &*field.base {
                        // RPython: setinteriorfield_gc — arr[i].field = value.
                        let base = get_value!(lower_expr(graph, block, &idx.expr, options, ctx)?);
                        let index = get_value!(lower_expr(graph, block, &idx.index, options, ctx)?);
                        let field_name = member_name(&field.member);
                        let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                        let elem_type = array_type_id
                            .as_ref()
                            .and_then(|atid| extract_element_type_from_str(atid));
                        // RPython: getkind(v_value.concretetype) — resolve field type
                        // from struct field registry for the kind suffix (i/r/f).
                        let item_ty = elem_type
                            .as_ref()
                            .and_then(|owner| ctx.struct_fields.field_type(owner, &field_name))
                            .map(type_string_to_value_type)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op(
                            *block,
                            OpKind::InteriorFieldWrite {
                                base,
                                index,
                                field: crate::model::FieldDescriptor::new(field_name, elem_type),
                                value,
                                item_ty,
                                array_type_id,
                            },
                            false,
                        );
                    } else {
                        let base = get_value!(lower_expr(graph, block, &field.base, options, ctx)?);
                        let field_name = member_name(&field.member);
                        let ty = field_value_type_from_expr(&field.base, &field.member, ctx)
                            .unwrap_or(ValueType::Unknown);
                        graph.push_op(
                            *block,
                            OpKind::FieldWrite {
                                base,
                                field: crate::model::FieldDescriptor::new(
                                    field_name,
                                    receiver_type_root(&field.base, ctx),
                                ),
                                value,
                                ty,
                            },
                            false,
                        );
                    }
                }
                syn::Expr::Index(idx) => {
                    let base = get_value!(lower_expr(graph, block, &idx.expr, options, ctx)?);
                    let index = get_value!(lower_expr(graph, block, &idx.index, options, ctx)?);
                    let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                    let item_ty =
                        array_item_value_type_from_array_type_id(array_type_id.as_deref())
                            .unwrap_or(ValueType::Unknown);
                    graph.push_op(
                        *block,
                        OpKind::ArrayWrite {
                            base,
                            index,
                            value,
                            item_ty,
                            array_type_id,
                        },
                        false,
                    );
                }
                _ => {
                    // Generic assignment — value already lowered
                }
            }
            Ok(Lowered::no_value())
        }

        // ── function call ──
        syn::Expr::Call(call) => {
            let mut args: Vec<ValueId> = Vec::with_capacity(call.args.len());
            for a in &call.args {
                let v = get_value!(lower_expr(graph, block, a, options, ctx)?);
                args.push(v);
            }
            let target = canonical_call_target(&call.func, ctx);
            // RPython parity: same rationale as the MethodCall arm above
            // — `op.result.concretetype` is set from the registered
            // FuncDesc.  Look up the qualified function path in
            // `ctx.fn_return_types` (populated in pass 1) so calls to
            // free functions returning `usize` / `bool` / `i64` propagate
            // a `Signed` result kind through rtyper instead of defaulting
            // to GcRef.
            let result_ty = if let syn::Expr::Path(p) = &*call.func {
                let segments: Vec<String> = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                intrinsic_call_result_type(&segments)
                    .or_else(|| {
                        lookup_function_return_type(ctx, &segments)
                            .map(|s| type_string_to_value_type(s))
                    })
                    .unwrap_or(ValueType::Unknown)
            } else {
                ValueType::Unknown
            };
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::Call {
                        target,
                        args,
                        result_ty,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── method call ──
        syn::Expr::MethodCall(mc) => {
            let mut args = Vec::new();
            let recv = get_value!(lower_expr(graph, block, &mc.receiver, options, ctx)?);
            args.push(recv);
            for a in &mc.args {
                let v = get_value!(lower_expr(graph, block, a, options, ctx)?);
                args.push(v);
            }
            // RPython `jtransform.py:410-412`: a polymorphic receiver
            // (dyn Trait) lowers to `indirect_call`, not `direct_call`.
            // Detect via the collected local_dyn_trait_roots map so
            // locals / params / Box<dyn> receivers all participate
            // (Issue 3 coverage).
            let receiver_root = receiver_type_root(&mc.receiver, ctx);
            let trait_bound_root = trait_bound_root_for_receiver(&mc.receiver, ctx);
            let target = if let Some(trait_root) = dyn_trait_root_for_receiver(&mc.receiver, ctx) {
                CallTarget::indirect(trait_root, mc.method.to_string())
            } else {
                CallTarget::method(mc.method.to_string(), receiver_root.clone())
            };
            // RPython parity: `op.result.concretetype` is set from the
            // callee graph's return signature at flowspace time
            // (`flowspace/objspace.py` consults the registered FuncDesc).
            // Pyre's pass 1 collected method return types into
            // `ctx.fn_return_types` keyed by `Type::method`; resolving
            // here lets the rtyper produce `Signed` operands for pure
            // integer ops (otherwise `value_type_to_kind` defaults to
            // `'r'` and the result reaches the assembler as a Ref-kind
            // operand, surfacing as `int_ge/ir>i` etc.).
            let result_ty = primitive_method_result_type(graph, &args, &mc.method)
                .or_else(|| transparent_option_method_result_type(graph, &args, &mc.method))
                .or_else(|| {
                    lookup_method_return_type(ctx, receiver_root.as_deref(), &mc.method)
                        .map(|s| type_string_to_value_type(s))
                })
                .or_else(|| {
                    lookup_method_return_type(ctx, trait_bound_root.as_deref(), &mc.method)
                        .map(|s| type_string_to_value_type(s))
                })
                .unwrap_or(ValueType::Unknown);
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::Call {
                        target,
                        args,
                        result_ty,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── if/else → block split (RPython FlowContext.guessbool) ──
        //
        // Creates: then_block, else_block, merge_block
        // If both branches produce a value, merge_block gets an inputarg
        // (Phi node) that receives the value from each branch via Link args.
        syn::Expr::If(if_expr) => {
            // ── if-let desugaring ──
            // `if let pat = scrutinee { then } else { else }` is exact
            // syntactic sugar for `match scrutinee { pat => then, _ =>
            // else }` (Rust Reference, "If let expressions"). We build
            // the synthetic `Expr::Match` AST and recurse so the
            // existing `Expr::Match` lowering (the path immediately
            // below at `syn::Expr::Match(m)`) handles the pattern
            // dispatch — keeps a single match-emit codepath rather than
            // duplicating the merge / phi / arm-entry logic.
            //
            // Without this desugar, `if_expr.cond` would be lowered as
            // a regular expression and trip the catch-all `Expr::Let`
            // arm below, emitting `OpKind::Abort { Let }`. That stub
            // makes any function carrying an `if let` un-portal-able
            // (Phase G G.4.4 path A.1) since a BH resume could land on
            // it and crash on "unknown bhimpl_*".
            if let syn::Expr::Let(let_expr) = if_expr.cond.as_ref() {
                let then_expr = syn::Expr::Block(syn::ExprBlock {
                    attrs: vec![],
                    label: None,
                    block: if_expr.then_branch.clone(),
                });
                let else_expr: syn::Expr = match &if_expr.else_branch {
                    Some((_, else_branch)) => (**else_branch).clone(),
                    None => syn::parse_quote!({}),
                };
                let then_arm = syn::Arm {
                    attrs: vec![],
                    pat: (*let_expr.pat).clone(),
                    guard: None,
                    fat_arrow_token: Default::default(),
                    body: Box::new(then_expr),
                    comma: Some(Default::default()),
                };
                let else_arm = syn::Arm {
                    attrs: vec![],
                    pat: syn::parse_quote!(_),
                    guard: None,
                    fat_arrow_token: Default::default(),
                    body: Box::new(else_expr),
                    comma: None,
                };
                let synthetic = syn::Expr::Match(syn::ExprMatch {
                    attrs: vec![],
                    match_token: Default::default(),
                    expr: let_expr.expr.clone(),
                    brace_token: Default::default(),
                    arms: vec![then_arm, else_arm],
                });
                return lower_expr(graph, block, &synthetic, options, ctx);
            }

            // RPython `flowspace/flowcontext.py:91,107,364`: unsupported
            // cond raises `FlowingError`, halting the walk.  A child
            // that closed its path (`if return_early { ... } else ...`)
            // also has no truth value — propagate via `get_value!`.
            let cond = get_value!(lower_expr(graph, block, &if_expr.cond, options, ctx)?);

            let mut then_block = graph.create_block();
            let mut else_block = graph.create_block();

            graph.set_branch(*block, cond, then_block, vec![], else_block, vec![]);

            // Lower then branch — collect result value
            let then_lowered = lower_stmt_list_with_tail_value(
                graph,
                &mut then_block,
                &if_expr.then_branch.stmts,
                options,
                ctx,
            )?;

            // Lower else branch
            let mut else_lowered = Lowered::no_value();
            if let Some((_, else_branch)) = &if_expr.else_branch {
                else_lowered = lower_expr(graph, &mut else_block, else_branch, options, ctx)?;
            }

            // RPython `flowspace/flowcontext.py` merges via Link: a
            // branch whose path is closed (`return`/`raise`/`break`)
            // does not `goto` the merge — the `is_open` check below
            // already skips it.  A phi inputarg is introduced when both
            // arms *produced a value*, mirroring the old all-or-nothing
            // shape; arity is kept consistent by skipping the closed
            // arm's goto so only the open arm sends a `vec![value]` to
            // the one-inputarg merge block.
            let then_value = then_lowered.value;
            let else_value = else_lowered.value;
            let then_open = graph.block(then_block).is_open();
            let else_open = graph.block(else_block).is_open();
            let want_phi = then_value.is_some() && else_value.is_some();

            let (merge_block, phi_result) = if want_phi {
                let (merge, phi_args) = graph.create_block_with_args(1);
                if then_open {
                    graph.set_goto(then_block, merge, vec![then_value.unwrap()]);
                }
                if else_open {
                    graph.set_goto(else_block, merge, vec![else_value.unwrap()]);
                }
                (merge, Some(phi_args[0]))
            } else {
                let merge = graph.create_block();
                if then_open {
                    graph.set_goto(then_block, merge, vec![]);
                }
                if else_open {
                    graph.set_goto(else_block, merge, vec![]);
                }
                (merge, None)
            };

            *block = merge_block;
            // If NEITHER arm remains open, the merge block is
            // unreachable — mark the enclosing path as closed so the
            // caller stops lowering into it.  RPython parity:
            // `flowspace/flowcontext.py` never keeps a merge block
            // reachable when all incoming links closed with
            // `FlowSignal::Return` / `Raise`.
            if !then_open && !else_open {
                Ok(Lowered::path_closed())
            } else {
                Ok(Lowered {
                    value: phi_result,
                    path_closed: false,
                })
            }
        }

        // ── return ──
        syn::Expr::Return(ret) => {
            // RPython `RETURN_VALUE` (`flowspace/flowcontext.py`):
            // `popvalue()` then `raise Return(w_result)`.  Pyre
            // equivalent: evaluate the return value (propagating
            // path_closed / FlowingError), then `set_return(..)` closes
            // the block and `Lowered::path_closed()` tells the caller
            // to stop walking this path.
            let val = if let Some(e) = &ret.expr {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
                lowered.value
            } else {
                None
            };
            graph.set_return(*block, val);
            Ok(Lowered::path_closed())
        }

        // ── block { stmts } ──
        syn::Expr::Block(blk) => {
            lower_stmt_list_with_tail_value(graph, block, &blk.block.stmts, options, ctx)
        }

        // ── literals ──
        // RPython `rpython/annotator/model.py` + `rtyper/rclass.py` resolve
        // every literal to a concrete SSA value at annotation time.  pyre
        // handles the common RPython-usable cases here; cases that RPython
        // itself does not support (f64 literals, char/str/byte literals
        // inside annotated code) still fall through to `OpKind::Abort`
        // and are tracked as rtyper follow-ups.
        syn::Expr::Lit(lit) => {
            match &lit.lit {
                syn::Lit::Int(int_lit) => {
                    if let Ok(v) = int_lit.base10_parse::<i64>() {
                        return Ok(Lowered {
                            value: graph.push_op(*block, OpKind::ConstInt(v), true),
                            path_closed: false,
                        });
                    }
                }
                // RPython lowers `True`/`False` to `Constant(1)`/`Constant(0)`
                // of `lltype.Bool`; at the codewriter level `getkind(Bool)`
                // returns `'int'` (`rpython/jit/codewriter/flatten.py getkind`)
                // so the value lives in an int register exactly like a
                // regular integer constant.  Emit as `ConstInt` to match.
                syn::Lit::Bool(b) => {
                    return Ok(Lowered {
                        value: graph.push_op(*block, OpKind::ConstInt(b.value as i64), true),
                        path_closed: false,
                    });
                }
                // RPython treats `chr(x)` / single-char byte literals as
                // `lltype.Char` which is also kind `'int'` (single unsigned
                // byte).  Rust `b'x'` (syn::Lit::Byte) and `'x'`
                // (syn::Lit::Char as u32) map to the same shape.
                syn::Lit::Byte(b) => {
                    return Ok(Lowered {
                        value: graph.push_op(*block, OpKind::ConstInt(b.value() as i64), true),
                        path_closed: false,
                    });
                }
                syn::Lit::Char(c) => {
                    return Ok(Lowered {
                        value: graph.push_op(*block, OpKind::ConstInt(c.value() as i64), true),
                        path_closed: false,
                    });
                }
                // RPython `flowmodel.py:Constant(rfloat)`: float literals
                // become `Constant` nodes with `lltype.Float` concretetype.
                // Pyre stores the bit pattern (`history.py:265
                // ConstFloat.getfloatstorage`) so PartialEq/Hash stay
                // derivable; the assembler materialises this through the
                // existing `constants_f` pool with a `float_copy` op.
                syn::Lit::Float(f) => {
                    if let Ok(v) = f.base10_parse::<f64>() {
                        return Ok(Lowered {
                            value: graph.push_op(*block, OpKind::ConstFloat(v.to_bits()), true),
                            path_closed: false,
                        });
                    }
                }
                _ => {}
            }
            // Unsupported literal kind — tag the specific variant so
            // the `Unknown` marker + diagnostics still identify the
            // remaining rtyper-side port gap (Str / Float / ByteStr /
            // Verbatim).  RPython `LOAD_CONST`
            // (`flowspace/flowcontext.py:841`) pushes the constant and
            // the flow walk continues; Err here would abort the whole
            // function graph and cascade through consumers like
            // `assert!("...")` / `panic!("...")` (which can legitimately
            // carry string literals next to side-effecting args).
            let variant = match &lit.lit {
                syn::Lit::Str(_) => UnsupportedLiteralKind::Str,
                syn::Lit::Float(_) => UnsupportedLiteralKind::Float,
                syn::Lit::ByteStr(_) => UnsupportedLiteralKind::ByteStr,
                syn::Lit::Verbatim(_) => UnsupportedLiteralKind::Verbatim,
                _ => UnsupportedLiteralKind::Other,
            };
            if std::env::var("MAJIT_UNKNOWN_DUMP").is_ok() {
                println!("cargo:warning=[UnsupportedLit] variant={variant:?}");
            }
            Ok(continue_with_unknown_literal(graph, *block, variant))
        }

        // ── path (variable reference) ──
        syn::Expr::Path(path) => {
            let name = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            let ty = ctx
                .local_value_types
                .get(&name)
                .cloned()
                .unwrap_or(ValueType::Unknown);
            Ok(Lowered {
                value: graph.push_op(*block, OpKind::Input { name, ty }, true),
                path_closed: false,
            })
        }

        // ── reference &expr ──
        syn::Expr::Reference(r) => lower_expr(graph, block, &r.expr, options, ctx),

        // `&raw const/mut expr` (`syn::Expr::RawAddr`) is intentionally
        // *not* pass-through here.  Unlike `&expr`, the raw-address
        // operator yields the *address* of the inner expr rather than
        // its value, so reusing the inner lowering would silently
        // misrepresent semantics (a downstream `as usize` cast would
        // see the dereferenced value instead of the pointer).  Falling
        // through to the `_ => other` unsupported handler classifies
        // it as `UnsupportedExprKind::RawAddr` (data-creation arm),
        // walks the inner expr for side effects via the `match other`
        // RawAddr branch below, and emits an `Unknown` marker so the
        // graph remains opaque rather than incorrect.

        // ── parenthesized (expr) ──
        syn::Expr::Paren(p) => lower_expr(graph, block, &p.expr, options, ctx),

        // ── unary !x, -x ──
        syn::Expr::Unary(u) => {
            let operand = get_value!(lower_expr(graph, block, &u.expr, options, ctx)?);
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::UnaryOp {
                        op: unary_op_name(&u.op).into(),
                        operand,
                        result_ty: ValueType::Unknown,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── binary a + b ──
        syn::Expr::Binary(bin) => {
            let lhs = get_value!(lower_expr(graph, block, &bin.left, options, ctx)?);
            let rhs = get_value!(lower_expr(graph, block, &bin.right, options, ctx)?);
            let op_name = binary_op_name(&bin.op);
            let result_ty = binary_result_value_type(graph, lhs, rhs, op_name);
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::BinOp {
                        op: op_name.into(),
                        lhs,
                        rhs,
                        result_ty,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── cast: expr as T ──
        syn::Expr::Cast(cast) => {
            let operand = get_value!(lower_expr(graph, block, &cast.expr, options, ctx)?);
            let result_ty = classify_fn_arg_ty(&cast.ty);
            if result_ty == ValueType::Unknown {
                return Ok(Lowered::value(operand));
            }
            if result_ty == ValueType::Void {
                return Ok(Lowered::no_value());
            }
            let source_ty = graph_value_type(graph, operand);
            let op = cast_op_name(source_ty.as_ref(), &result_ty).to_string();
            Ok(Lowered {
                value: graph.push_op(
                    *block,
                    OpKind::UnaryOp {
                        op,
                        operand,
                        result_ty,
                    },
                    true,
                ),
                path_closed: false,
            })
        }

        // ── match expr { arms } → multi-block (RPython switch) ──
        syn::Expr::Match(m) => {
            let scrutinee = get_value!(lower_expr(graph, block, &m.expr, options, ctx)?);
            let scrutinee_type_string = expression_type_string(&m.expr, ctx);

            if m.arms.is_empty() {
                return Ok(Lowered::no_value());
            }
            let bool_exitcases = classify_match_bool_exitcases(&m.arms);
            let switch_exitcases = if bool_exitcases.is_none() {
                classify_match_switch_exitcases(&m.arms)
            } else {
                None
            };

            // Lower each arm body into its own block, collecting both
            // the ENTRY block (what the outer Branch/Goto jumps to)
            // and the TAIL block (what jumps to merge). lower_expr
            // takes `&mut arm_block` and may rewire arm_block to the
            // arm's tail (e.g., nested if/match's merge). We capture
            // the entry before calling so the outer terminator targets
            // the right landing pad.
            //
            // The merge block's inputarg list must have the same
            // length as every outgoing Goto's args (flatten.py:308
            // assumption), so we defer merge creation until we know
            // whether any arm actually produced a value.
            //
            // RPython `flowspace/flowcontext.py:417` — `FlowingError`
            // from any arm aborts the whole function graph, not just
            // the current arm.  `?` here propagates that out of the
            // whole match so the enclosing `build_function_graph` body
            // loop breaks at the first unsupported construct, matching
            // upstream's all-or-nothing flowgraph semantics.
            let mut arm_entries: Vec<BlockId> = Vec::with_capacity(m.arms.len());
            let mut arm_tails: Vec<(BlockId, Option<ValueId>)> = Vec::with_capacity(m.arms.len());
            for arm in &m.arms {
                let entry = graph.create_block();
                let mut tail = entry;
                let saved_locals = LocalBindingSnapshot::capture(ctx);
                bind_pattern_locals(&arm.pat, scrutinee_type_string.as_deref(), ctx);
                let arm_lowered_result = lower_expr(graph, &mut tail, &arm.body, options, ctx);
                saved_locals.restore(ctx);
                let arm_lowered = arm_lowered_result?;
                // A closed arm (body is `return x` / `break` / `panic!`
                // / `raise`) does not contribute a value to the merge —
                // its path terminates inside `tail` and no outgoing
                // goto is synthesised.  Per RPython
                // `flowspace/flowcontext.py:1253` `Raise.nomoreblocks`,
                // sibling walks continue irrespective of this arm's
                // closure.
                arm_entries.push(entry);
                arm_tails.push((tail, arm_lowered.value));
            }

            // Merge gets a Phi inputarg iff every arm that actually
            // reaches the merge carries a value.  Closed arms (early
            // `return` / `break`) don't emit a goto to merge, so they
            // contribute nothing to the phi arity.  Mixing some-value
            // and no-value open arms would require a fake phi arg for
            // the no-value arms (RPython `jit/codewriter/flatten.py:308`
            // — every outgoing goto's arg list must match the target's
            // inputarg arity), so in that case we emit no phi at all.
            let all_open_arms_have_value = arm_tails
                .iter()
                .all(|(tail, r)| !graph.block(*tail).is_open() || r.is_some());
            let (merge, merge_phi) = if all_open_arms_have_value {
                let (m_block, phi_args) = graph.create_block_with_args(1);
                (m_block, Some(phi_args[0]))
            } else {
                (graph.create_block(), None)
            };

            let mut any_open = false;
            for (tail, result) in &arm_tails {
                if !graph.block(*tail).is_open() {
                    continue;
                }
                any_open = true;
                let goto_args = if all_open_arms_have_value {
                    // Safe: the filter above guarantees every open arm's
                    // `result` is `Some`.
                    vec![result.unwrap()]
                } else {
                    Vec::new()
                };
                graph.set_goto(*tail, merge, goto_args);
            }

            if let Some(arm_exitcases) = bool_exitcases {
                // RPython `flatten.py:240-267` lowers Bool exitswitch
                // blocks through the two-link `goto_if_not` path, not the
                // integer `switch` path.  A Rust `match flag { true => ...,
                // _ => ... }` is therefore expanded to explicit
                // True/False exitcases instead of using the switch
                // `"default"` sentinel.
                let mut exits = Vec::new();
                for (entry, exitcases) in arm_entries.iter().zip(arm_exitcases.iter()) {
                    for exitcase in exitcases {
                        exits.push(
                            Link::new_mixed(Vec::new(), *entry, Some(exitcase.clone()))
                                .with_llexitcase_from_exitcase(),
                        );
                    }
                }
                graph.set_control_flow_metadata(*block, Some(ExitSwitch::Value(scrutinee)), exits);
            } else if let Some(arm_exitcases) = switch_exitcases {
                // RPython `flatten.py:278-308` switch shape:
                // `exitswitch` is the scrutinee and each primitive arm
                // contributes one Link with a concrete `exitcase`.  A
                // wildcard arm uses the same `"default"` sentinel that
                // upstream treats as the fall-through switch path.
                let mut exits = Vec::new();
                for (entry, exitcases) in arm_entries.iter().zip(arm_exitcases.iter()) {
                    for exitcase in exitcases {
                        exits.push(
                            Link::new_mixed(Vec::new(), *entry, Some(exitcase.clone()))
                                .with_llexitcase_from_exitcase(),
                        );
                    }
                }
                graph.set_control_flow_metadata(*block, Some(ExitSwitch::Value(scrutinee)), exits);
            } else if m.arms.len() == 1 {
                graph.set_goto(*block, arm_entries[0], vec![]);
            } else {
                // Structural adaptation for Rust composite patterns
                // (`if let Some(_)`, `Err(_)`, tuple/struct variants).
                // This front-end lacks a typed enum-discriminant op; keep
                // the existing two-arm truthy split for those cases rather
                // than inventing a fake switch key. Primitive literal
                // patterns use the switch path above.
                graph.set_branch(
                    *block,
                    scrutinee,
                    arm_entries[0],
                    vec![],
                    arm_entries[1],
                    vec![],
                );
            }

            *block = merge;
            if !any_open {
                // All arms terminated — the enclosing walk has no open
                // path to continue.
                Ok(Lowered::path_closed())
            } else {
                Ok(Lowered {
                    value: merge_phi,
                    path_closed: false,
                })
            }
        }

        // ── while → header block + body block + exit block ──
        syn::Expr::While(w) => {
            let header_entry = graph.create_block();
            let exit = graph.create_block();

            // Current block → header_entry (loop-head, 0 inputargs).
            graph.set_goto(*block, header_entry, vec![]);

            // Header: evaluate condition, branch to body or exit.
            // `lower_expr(&mut header_tail, ...)` may rewire to a
            // sub-merge; the cond-branch attaches to header_tail so
            // the branch lives at the header's actual end.
            //
            // RPython `flowspace/flowcontext.py:91,107,364`: unsupported
            // cond raises `FlowingError`.  We propagate that via `?` —
            // no fake cond, no fallback goto-exit.  The exit block we
            // pre-created above becomes dead; simplify prunes it.
            let mut header_tail = header_entry;
            let cond = get_value!(lower_expr(graph, &mut header_tail, &w.cond, options, ctx)?);
            let body_entry = graph.create_block();
            graph.set_branch(header_tail, cond, body_entry, vec![], exit, vec![]);

            // Body → back to header_entry (entry, not tail —
            // header_entry is the 0-inputarg loop-head).  Each stmt may
            // close its path (inner `return` / `break` / `panic!`); on
            // closure we stop walking the body and the back-edge is
            // skipped via the `is_open` check below.  The loop frame
            // makes `break` / `continue` in the body route to exit /
            // header.
            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &w.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, header_entry, vec![]);
            }

            *block = exit;
            Ok(Lowered::no_value())
        }
        syn::Expr::Loop(l) => {
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            graph.set_goto(*block, body_entry, vec![]);

            ctx.loop_stack.push(LoopFrame {
                continue_target: body_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &l.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, body_entry, vec![]);
            }

            *block = exit;
            Ok(Lowered::no_value())
        }
        syn::Expr::ForLoop(f) => {
            // RPython `for` lowers to the iterator protocol: `GET_ITER`
            // on the iterable, then a `FOR_ITER` at the header whose
            // true arm binds the next item into the body and whose
            // false arm falls through (`rpython/flowspace/
            // flowcontext.py:782,787,1378`).  Pyre has NO `Iter` /
            // `Next` op yet (Slice 6 port).  The shape below is
            // deliberately NOT claiming op-level equivalence with
            // upstream's iter/next — it emits a SINGLE `Unknown`
            // marker tagged `ForLoop` at the header that stands for
            // the whole iterator protocol, and walks the iterable
            // sub-expression for its side effects so the
            // `build_flow`-visible part of the construct is complete
            // even when the loop ops themselves are stubbed.
            let iterable = get_value!(lower_expr(graph, block, &f.expr, options, ctx)?);
            let _ = iterable;

            let header_entry = graph.create_block();
            let body_entry = graph.create_block();
            let exit = graph.create_block();
            graph.set_goto(*block, header_entry, vec![]);

            // Single iterator-protocol placeholder, NOT two separate
            // iter/next markers.  The branch shape is required to
            // make `exit` reachable from the normal control-flow
            // fallthrough (without it, loops without `break` would
            // leave every statement after the `for` unreachable).
            let for_cond = graph.push_op(
                header_entry,
                OpKind::Abort {
                    kind: UnknownKind::UnsupportedExpr {
                        variant: UnsupportedExprKind::ForLoop,
                    },
                },
                true,
            );
            if let Some(cond) = for_cond {
                graph.set_branch(header_entry, cond, body_entry, vec![], exit, vec![]);
            } else {
                graph.set_goto(header_entry, body_entry, vec![]);
            }

            ctx.loop_stack.push(LoopFrame {
                continue_target: header_entry,
                break_target: exit,
            });
            let mut body_tail = body_entry;
            for stmt in &f.body.stmts {
                let closed = lower_stmt(graph, &mut body_tail, stmt, options, ctx)?;
                if closed {
                    break;
                }
            }
            ctx.loop_stack.pop();
            if graph.block(body_tail).is_open() {
                graph.set_goto(body_tail, header_entry, vec![]);
            }

            *block = exit;
            Ok(Lowered::no_value())
        }

        // ── break/continue ──
        //
        // RPython `flowspace/flowcontext.py:525` models these as
        // `Break` / `Continue` `FlowSignal`s; `LoopBlock.handle_signal`
        // (`:1341`) rewrites the current block with a Link to the
        // loop's end / header.  Pyre's port: look up the enclosing
        // `LoopFrame` on `ctx.loop_stack` and close the current block
        // with `set_goto(*block, target)`, then report path_closed so
        // the surrounding walker stops emitting ops into a
        // now-terminated block.  A break/continue outside any loop is
        // orphaned — `path_closed` alone gives the surrounding walker
        // the stop signal without corrupting the graph.
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
            }
            if let Some(frame) = ctx.loop_stack.last().copied() {
                if graph.block(*block).is_open() {
                    graph.set_goto(*block, frame.break_target, vec![]);
                }
            }
            Ok(Lowered::path_closed())
        }
        syn::Expr::Continue(_) => {
            if let Some(frame) = ctx.loop_stack.last().copied() {
                if graph.block(*block).is_open() {
                    graph.set_goto(*block, frame.continue_target, vec![]);
                }
            }
            Ok(Lowered::path_closed())
        }

        // ── closure ──
        //
        // RPython `MAKE_FUNCTION` (`pypy/interpreter/pyopcode.py:1144`,
        // `flowspace/flowcontext.py:1177`) pushes a fresh function
        // value onto the stack — the `def`/`lambda` body becomes a
        // separate graph, NOT inlined into the enclosing flow.  Pyre
        // has no per-closure graph yet (the closure body is usually
        // captured via an `fn` pointer arg), so the expression lowers
        // to a single `Unknown` marker representing the closure
        // value.  The body is NOT walked here: inlining it into the
        // caller's flow was a New-Deviation that treated the closure
        // as a synchronous block, which broke callers that pass the
        // closure itself as a function-typed argument (e.g. empty
        // `|_| {}` body produced no value for `get_value!`).
        syn::Expr::Closure(_) => Ok(continue_with_unknown(
            graph,
            *block,
            UnsupportedExprKind::OtherExpr,
        )),

        // ── tuple (a, b, c) ──
        syn::Expr::Tuple(t) => {
            // RPython `BUILD_TUPLE` (`pypy/interpreter/pyopcode.py:955`,
            // `flowspace/flowcontext.py:1163`) always pushes a fresh
            // tuple object — the result is a NEW value distinct from
            // any individual element.  Pyre has no `NewTuple` op yet
            // (Slice 5 port), so the construct lowers to a single
            // `Unknown` marker tagged `Tuple` that stands in for the
            // whole tuple-builder; callers that read the result get a
            // well-formed ValueId but coverage audits still flag the
            // port gap.  Elements lower for their side effects and
            // path-closed propagation but do NOT feed the result.
            for e in &t.elems {
                let lowered = lower_expr(graph, block, e, options, ctx)?;
                if lowered.path_closed {
                    return Ok(Lowered::path_closed());
                }
            }
            Ok(continue_with_unknown(
                graph,
                *block,
                UnsupportedExprKind::Tuple,
            ))
        }

        // ── try expr? ──
        //
        // RPython `flowspace/flowcontext.py:127-148 guessexception` port.
        // A can-raise op closes its containing block with
        // `block.exitswitch = c_last_exception` and two Links: the
        // normal fall-through Link and the exception Link whose
        // `args`/`extravars` both reference fresh prevblock-side
        // `Variable('last_exception')` / `Variable('last_exc_value')`
        // (`flowcontext.py:130-134`).  These fresh variables flow into
        // the exceptblock's own inputargs via `insert_renamings` — the
        // target side has its own distinct Variables
        // (`flowcontext.py:135 vars2`), matching upstream's "Link.args
        // are prevblock-side values" invariant at
        // `flowspace/model.py:114`.
        syn::Expr::Try(t) => {
            let ok_ty = expression_type_string(&t.expr, ctx)
                .as_deref()
                .and_then(transparent_result_ok_type)
                .map(type_string_to_value_type);
            let inner = get_value!(lower_expr(graph, block, &t.expr, options, ctx)?);
            if let Some(ok_ty) = ok_ty {
                retag_result_value_type(graph, inner, ok_ty);
            }
            let continuation = graph.create_block();
            let continuation_arg = graph.alloc_value();
            graph
                .block_mut(continuation)
                .inputargs
                .push(continuation_arg);
            // RPython `flowcontext.py:130-133` — fresh prevblock-side
            // `Variable('last_exception')` + `Variable('last_exc_value')`.
            let last_exception = graph.alloc_value();
            let last_exc_value = graph.alloc_value();
            let exc_block = graph.exceptblock;
            graph.set_goto(*block, continuation, vec![inner]);
            graph.set_control_flow_metadata(
                *block,
                Some(ExitSwitch::LastException),
                vec![
                    // RPython `flowcontext.py:141` `Link(vars=[], egg, case=None)`
                    // for the normal fall-through.
                    Link::new(vec![inner], continuation, None),
                    // RPython `flowcontext.py:141-143` `link = Link(vars, egg, case)`
                    // + `link.extravars(last_exception=..., last_exc_value=...)`
                    // with `case is Exception`.
                    Link::new(
                        vec![last_exception, last_exc_value],
                        exc_block,
                        Some(exception_exitcase()),
                    )
                    .extravars(
                        Some(LinkArg::from(last_exception)),
                        Some(LinkArg::from(last_exc_value)),
                    ),
                ],
            );
            *block = continuation;
            Ok(Lowered::value(continuation_arg))
        }

        // ── unsafe { stmts } ──
        //
        // RPython flow-space has no concept of `unsafe` — in Python every
        // load/store already has the same aliasing model.  In the Rust
        // port `unsafe { stmts }` wraps raw-pointer / transmute helpers
        // whose **body** is still a regular Rust block; the `unsafe`
        // keyword is a type-system marker, not runtime semantics.  Lower
        // it by reusing the same `Block` path so the contained
        // statements + tail expression flow through normally.
        syn::Expr::Unsafe(u) => {
            lower_stmt_list_with_tail_value(graph, block, &u.block.stmts, options, ctx)
        }

        // ── fallback ──
        //
        // RPython `flowspace/flowcontext.py` evaluates sub-expressions
        // eagerly as bytecode streams in; `FlowingError` halts the
        // walk AT the unsupported op, not BEFORE the sub-expression
        // push operations.  For Rust variants whose AST carries named
        // sub-expressions (Range endpoints, Struct field values, Array
        // / Repeat elements, `if let` scrutinee) we walk those first
        // so their Call / FieldRead / etc. ops land in the graph
        // before the Unknown marker + abort.
        other => {
            // Conditional-raise macro family (assert!, debug_assert!,
            // assert_eq!, assert_ne!, debug_assert_eq!,
            // debug_assert_ne!) expand to `if !cond { panic }` — a
            // runtime check that either continues or unconditionally
            // raises.  Port to the RPython-canonical shape of a
            // `set_branch` whose false side routes through the
            // exceptblock via `set_raise`
            // (`rpython/flowspace/model.py:21-25`).  Unlike panic!, the
            // macro expression itself has type `()` — on the pass side
            // the enclosing walk continues normally.
            if let syn::Expr::Macro(m) = other {
                let macro_name = m
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                let is_assert = matches!(macro_name.as_str(), "assert" | "debug_assert");
                let is_assert_cmp = matches!(
                    macro_name.as_str(),
                    "assert_eq" | "assert_ne" | "debug_assert_eq" | "debug_assert_ne"
                );

                // ── matches! desugaring ──
                // `matches!(scrutinee, pat)` and `matches!(scrutinee,
                // pat if guard)` desugar (per std::matches docs) to
                // `match scrutinee { pat => true, _ => false }` (with
                // guard inlined onto the arm if present). We build the
                // synthetic `Expr::Match` AST and recurse so the
                // existing match lowering handles the dispatch — same
                // shape as the `if let` desugar above.
                //
                // Without this desugar, `matches!` flows through the
                // catch-all `Expr::Macro` arm below and emits
                // `OpKind::Abort { Macro }`. Phase G G.4.4 Path A.B.
                if macro_name == "matches" {
                    let tokens = m.mac.tokens.clone();
                    if let Some((scrutinee_tokens, rest_tokens)) =
                        split_macro_args_at_first_top_comma(tokens)
                    {
                        if let (Ok(scrutinee_expr), Ok((pat, guard))) = (
                            syn::parse2::<syn::Expr>(scrutinee_tokens),
                            syn::parse::Parser::parse2(parse_matches_pat_and_guard, rest_tokens),
                        ) {
                            let arm_then_body: syn::Expr = syn::parse_quote!(true);
                            let arm_else_body: syn::Expr = syn::parse_quote!(false);
                            let then_arm = syn::Arm {
                                attrs: vec![],
                                pat,
                                guard: guard.map(|g| (Default::default(), Box::new(g))),
                                fat_arrow_token: Default::default(),
                                body: Box::new(arm_then_body),
                                comma: Some(Default::default()),
                            };
                            let else_arm = syn::Arm {
                                attrs: vec![],
                                pat: syn::parse_quote!(_),
                                guard: None,
                                fat_arrow_token: Default::default(),
                                body: Box::new(arm_else_body),
                                comma: None,
                            };
                            let synthetic = syn::Expr::Match(syn::ExprMatch {
                                attrs: vec![],
                                match_token: Default::default(),
                                expr: Box::new(scrutinee_expr),
                                brace_token: Default::default(),
                                arms: vec![then_arm, else_arm],
                            });
                            return lower_expr(graph, block, &synthetic, options, ctx);
                        }
                    }
                    // Parse failure falls through to the catch-all
                    // Macro arm — preserves the `OpKind::Abort`
                    // diagnostic for un-portable `matches!` shapes
                    // rather than silently mis-lowering.
                }

                if is_assert || is_assert_cmp {
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        let mut it = args.iter();
                        let cond_opt: Option<ValueId> = if is_assert {
                            if let Some(cond_expr) = it.next() {
                                let lowered = lower_expr(graph, block, cond_expr, options, ctx)?;
                                if lowered.path_closed {
                                    return Ok(Lowered::path_closed());
                                }
                                lowered.value
                            } else {
                                None
                            }
                        } else {
                            let lhs_expr = it.next();
                            let rhs_expr = it.next();
                            match (lhs_expr, rhs_expr) {
                                (Some(le), Some(re)) => {
                                    let lhs =
                                        get_value!(lower_expr(graph, block, le, options, ctx)?);
                                    let rhs =
                                        get_value!(lower_expr(graph, block, re, options, ctx)?);
                                    let op_name = if macro_name.contains("_ne") {
                                        "ne"
                                    } else {
                                        "eq"
                                    };
                                    graph.push_op(
                                        *block,
                                        OpKind::BinOp {
                                            op: op_name.into(),
                                            lhs,
                                            rhs,
                                            result_ty: ValueType::Unknown,
                                        },
                                        true,
                                    )
                                }
                                _ => None,
                            }
                        };
                        if let Some(cond) = cond_opt {
                            // Split into pass/fail arms BEFORE walking
                            // the message expressions.  Per RPython
                            // `flowspace/flowcontext.py:107`
                            // (`BlockRecorder.guessbool`), the two
                            // arms of a conditional are independent
                            // walks — the message-format arguments
                            // are only reachable on the failing path
                            // and must not land ops on the pass path.
                            //
                            // Message format arguments walk on the
                            // fail branch: RPython `LOAD_CONST`
                            // (`flowspace/flowcontext.py:841`) pushes a
                            // constant and the walk continues, so Str /
                            // Float / ByteStr literals are no longer
                            // fatal — the non-fatal Lit handler above
                            // emits an `Unknown` marker and returns a
                            // value.  We therefore walk every rest arg
                            // unconditionally (side-effect-preserving
                            // order).
                            let pass_block = graph.create_block();
                            let mut fail_block = graph.create_block();
                            graph.set_branch(*block, cond, pass_block, vec![], fail_block, vec![]);
                            // Walk every message-expr on the fail
                            // branch to preserve its side effects
                            // (Call / FieldRead / …) on the graph,
                            // then hand the evaluated ValueIds to the
                            // shared `exc_from_raise` lowering as the
                            // positional args of `simple_call(AssertionError,
                            // *args)`.  Upstream parity: RPython
                            // `RAISE_VARARGS` (`flowcontext.py:638-656`)
                            // popvalue's all args before reaching
                            // `exc_from_raise`; the adapter here picks
                            // `AssertionError` as the `w_arg1` for
                            // every assert-family macro so
                            // `front::raise::lower_exc_from_raise`
                            // walks the same op sequence as the
                            // flowspace port at
                            // `flowspace/flowcontext.rs:1189`.
                            let mut message_args: Vec<ValueId> = Vec::new();
                            for rest in it {
                                // The fail-branch walk is independent
                                // of the pass-branch walk; a
                                // path-closing construct inside the
                                // message format (`panic!` nested
                                // inside the format arg) still leaves
                                // the pass branch open, so we don't
                                // propagate path_closed out here.
                                // FlowingError still propagates via
                                // `?`.
                                let lowered =
                                    lower_expr(graph, &mut fail_block, rest, options, ctx)?;
                                if let Some(v) = lowered.value {
                                    message_args.push(v);
                                }
                            }
                            let _ = &macro_name; // name is only used for diagnostics; class is fixed.
                            crate::front::raise::lower_exc_from_raise(
                                graph,
                                fail_block,
                                "AssertionError",
                                message_args,
                            );
                            *block = pass_block;
                            // Pass block is still open — the assert
                            // expression itself has type `()`, no value.
                            return Ok(Lowered::no_value());
                        }
                    }
                }
            }
            // Abort-family macros (`panic!`, `unreachable!`, `todo!`,
            // `unimplemented!`) have type `!` and terminate the current
            // control-flow path with an unconditional raise.  Matches
            // RPython `flowspace/flowcontext.py:1253` `Raise.nomoreblocks`
            // where the enclosing block is closed with a Link to
            // `exceptblock` regardless of the exception argument shape.
            //
            // Per RPython `RAISE_VARARGS`
            // (`flowspace/flowcontext.py:638-656`), the raise target /
            // arguments are `popvalue()`'d off the stack — they have
            // already been evaluated before the Raise.  The same
            // happens in Rust: `panic!("{}", side_effect())` evaluates
            // `side_effect()` before panicking.  Walk every macro arg
            // before `set_raise` so its side effects land in the graph.
            // Literal args are no longer fatal (Lit handler above emits
            // `Unknown` + returns a value), so no skip is needed.
            if let syn::Expr::Macro(m) = other {
                let name = m
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                if matches!(
                    name.as_str(),
                    "panic" | "unreachable" | "todo" | "unimplemented"
                ) {
                    // Walk every message-arg for its side effects
                    // (popvalue-before-raise semantic of RPython
                    // `RAISE_VARARGS`, `flowcontext.py:638-656`), then
                    // forward the evaluated ValueIds as the positional
                    // args of `simple_call(PanicError, *args)` inside
                    // the shared `exc_from_raise` lowering
                    // (`front::raise::lower_exc_from_raise` →
                    // `flowcontext.rs:1189` parity).
                    let mut message_args: Vec<ValueId> = Vec::new();
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        for arg in args.iter() {
                            let lowered = lower_expr(graph, block, arg, options, ctx)?;
                            if lowered.path_closed {
                                // A path-closing sub-expression already
                                // terminated `*block`; the outer panic!
                                // has nothing more to do — propagate
                                // path_closed so the enclosing walker
                                // stops.
                                return Ok(Lowered::path_closed());
                            }
                            if let Some(v) = lowered.value {
                                message_args.push(v);
                            }
                        }
                    }
                    // RPython `flowspace/flowcontext.py:1253`
                    // `Raise.nomoreblocks`: close the current block
                    // with a Link to `exceptblock`, then signal the
                    // path terminated.  RPython raises `StopFlowing`,
                    // which is the same kind of FlowSignal as Return —
                    // sibling walks continue normally.  Pyre equivalent:
                    // `Lowered::path_closed()` on the `Ok` arm, NOT
                    // `Err(FlowingError)` (which would abort the whole
                    // function graph).  The Rust panic-family macros
                    // (panic!, unreachable!, todo!, unimplemented!)
                    // all share the `PanicError` adapter class — their
                    // runtime-distinct PanicInfo shape is not modelled
                    // at the flow-graph layer, mirroring reviewer
                    // guidance (`flowcontext.py:2861 Raise` bytecode
                    // adapter — version-specific variants converge on
                    // `exc_from_raise(w_arg1, w_arg2)`).
                    let _ = &name; // macro name carried for diagnostics only.
                    crate::front::raise::lower_exc_from_raise(
                        graph,
                        *block,
                        "PanicError",
                        message_args,
                    );
                    return Ok(Lowered::path_closed());
                }
            }
            let variant = match other {
                syn::Expr::Array(_) => UnsupportedExprKind::Array,
                syn::Expr::Async(_) => UnsupportedExprKind::Async,
                syn::Expr::Await(_) => UnsupportedExprKind::Await,
                syn::Expr::Const(_) => UnsupportedExprKind::Const,
                syn::Expr::Group(_) => UnsupportedExprKind::Group,
                syn::Expr::Infer(_) => UnsupportedExprKind::Infer,
                syn::Expr::Let(_) => UnsupportedExprKind::Let,
                syn::Expr::Macro(_) => UnsupportedExprKind::Macro,
                syn::Expr::Range(_) => UnsupportedExprKind::Range,
                syn::Expr::RawAddr(_) => UnsupportedExprKind::RawAddr,
                syn::Expr::Repeat(_) => UnsupportedExprKind::Repeat,
                syn::Expr::Struct(_) => UnsupportedExprKind::Struct,
                syn::Expr::TryBlock(_) => UnsupportedExprKind::TryBlock,
                syn::Expr::Verbatim(_) => UnsupportedExprKind::Verbatim,
                syn::Expr::Yield(_) => UnsupportedExprKind::Yield,
                _ => UnsupportedExprKind::OtherExpr,
            };
            // The diagnostic emit decision is made later in the
            // `is_data_creation` / `stop_unsupported` chain below;
            // the dump probe runs there so `[UnsupportedExpr]` covers
            // the data-creation default-arm path and
            // `[UnsupportedExpr/stop]` covers the abort path.
            // Helper: walk a sub-expression purely for its side effects
            // (the parent composite is about to be marked unsupported,
            // so the returned value is unused).  Propagate FlowingError
            // via `?`; on path_closed, bail out of the parent walk too
            // — the enclosing block is already terminated and a later
            // `stop_unsupported` would push into a closed block.
            macro_rules! walk_for_side_effects {
                ($e:expr) => {{
                    let lowered = lower_expr(graph, block, $e, options, ctx)?;
                    if lowered.path_closed {
                        return Ok(Lowered::path_closed());
                    }
                }};
            }
            // Non-fatal families mirror RPython bytecodes that push a
            // value and continue the flow walk:
            //   • Data constructors — `BUILD_LIST` / `BUILD_TUPLE` /
            //     `newslice` (`flowspace/flowcontext.py:1168`,
            //     `pypy/interpreter/pyopcode.py:960`).  Pyre does not
            //     yet emit `NewList` / `NewStruct` / `NewRange` IR
            //     ops, so element walks land in the graph and a
            //     single `Unknown` marker stands in for the
            //     allocation.  The local Rust-parity adapter
            //     `flowspace/rust_source/build_flow.rs:1889`
            //     (`lower_array -> newlist`) uses the same shape.
            //   • Generic (non-abort) macros — `format!`, `write!`,
            //     `vec!`, `matches!`, …  treat these as opaque ops
            //     whose result is an opaque value; sub-expr walks
            //     still capture side effects before the marker.
            //     Abort-family macros are handled separately by the
            //     `set_raise` branch earlier in the Macro arm above.
            let is_data_creation = matches!(
                other,
                syn::Expr::Array(_)
                    | syn::Expr::Repeat(_)
                    | syn::Expr::Struct(_)
                    | syn::Expr::Range(_)
                    | syn::Expr::Let(_)
                    | syn::Expr::Macro(_)
                    | syn::Expr::RawAddr(_)
            );
            match other {
                // `a..b` / `a..=b` / `..b` / `a..` / `..` — evaluate
                // the endpoint expressions so side effects in them are
                // captured.  Per RPython `newslice` (implicit in
                // `BUILD_SLICE` at `pypy/interpreter/pyopcode.py`), the
                // endpoints land as separate pushes before the slice
                // is constructed.
                syn::Expr::Range(r) => {
                    if let Some(from) = &r.start {
                        walk_for_side_effects!(from);
                    }
                    if let Some(to) = &r.end {
                        walk_for_side_effects!(to);
                    }
                }
                // `[a, b, c]` — evaluate each element.  RPython
                // `BUILD_LIST` (`flowspace/flowcontext.py:1168`) pops
                // N items and pushes `space.newlist(items)`; we emit
                // an `Unknown` marker for the `newlist` step, which
                // matches the local Rust-parity adapter in
                // `flowspace/rust_source/build_flow.rs:1889`
                // (`lower_array -> newlist`) until a proper
                // `OpKind::NewList` lands.
                syn::Expr::Array(a) => {
                    for e in &a.elems {
                        walk_for_side_effects!(e);
                    }
                }
                // `[v; N]` — evaluate the element expression and the
                // repeat count expression.  N is commonly a literal
                // integer; walking it emits a `ConstInt` op that the
                // annotator can still see.
                syn::Expr::Repeat(r) => {
                    walk_for_side_effects!(&r.expr);
                    walk_for_side_effects!(&r.len);
                }
                // `S { f: v, g: w, ..rest }` — evaluate each field
                // value, then any `..rest` base.  Parallels RPython
                // `newstruct` / `BUILD_MAP`-style constructors.
                syn::Expr::Struct(s) => {
                    for field in &s.fields {
                        walk_for_side_effects!(&field.expr);
                    }
                    if let Some(rest) = &s.rest {
                        walk_for_side_effects!(rest);
                    }
                }
                // `let PAT = EXPR` (only reachable as the cond of an
                // `if let` / `while let`).  Evaluate the scrutinee so
                // side effects are captured; the pattern match itself
                // remains opaque until enum-variant dispatch lands.
                syn::Expr::Let(l) => {
                    walk_for_side_effects!(&l.expr);
                }
                // `&raw const/mut EXPR` — the address operator
                // produces a pointer rather than the inner value, so
                // we emit an `Unknown` marker for the address itself
                // (handled by the data-creation arm below) but still
                // walk the inner expr so any side effects are
                // captured before the pointer is taken.
                syn::Expr::RawAddr(r) => {
                    walk_for_side_effects!(&r.expr);
                }
                // `foo!(a, b, c)` / `foo![a, b, c]` / `foo!{a, b, c}`
                // — most Rust macros whose bodies reach this point
                // (vec!, format!, matches!, write!, writeln!, ...)
                // accept comma-separated expressions as arguments.
                // Parse the token stream as `Punctuated<Expr, ,>` and
                // walk each; on parse failure (e.g. macros with
                // non-expression syntax), fall through to bare abort.
                // Matches the RPython FlowingError convention at
                // `rpython/flowspace/flowcontext.py:258` where
                // sub-expression push ops land BEFORE the abort point.
                syn::Expr::Macro(m) => {
                    if let Ok(args) = m.mac.parse_body_with(
                        syn::punctuated::Punctuated::<syn::Expr, syn::Token![,]>::parse_terminated,
                    ) {
                        for arg in args.iter() {
                            walk_for_side_effects!(arg);
                        }
                    }
                }
                _ => {}
            }
            let dump_enabled = std::env::var("MAJIT_UNKNOWN_DUMP").is_ok();
            if is_data_creation {
                if dump_enabled {
                    let fn_name = CURRENT_LOWERING_FN_NAME
                        .with(|c| c.borrow().clone())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    println!("cargo:warning=[UnsupportedExpr] fn={fn_name} variant={variant:?}");
                }
                Ok(continue_with_unknown(graph, *block, variant))
            } else {
                if dump_enabled {
                    let fn_name = CURRENT_LOWERING_FN_NAME
                        .with(|c| c.borrow().clone())
                        .unwrap_or_else(|| "<unknown>".to_string());
                    println!(
                        "cargo:warning=[UnsupportedExpr/stop] fn={fn_name} variant={variant:?}"
                    );
                }
                stop_unsupported(graph, *block, variant)
            }
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Split a macro token stream at the first top-level (depth-0) comma,
/// returning `(prefix, suffix)` token streams without the comma. Used
/// by the `matches!` desugar to peel `(scrutinee, pat [if guard])` —
/// the first comma separates `scrutinee` from the rest, but commas
/// inside `(...)` / `[...]` / `{...}` (e.g. tuple struct patterns,
/// inner expression lists) must not split. Returns `None` when no
/// top-level comma is present.
fn split_macro_args_at_first_top_comma(
    tokens: proc_macro2::TokenStream,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut prefix: Vec<proc_macro2::TokenTree> = Vec::new();
    let mut iter = tokens.into_iter();
    for tt in iter.by_ref() {
        if let proc_macro2::TokenTree::Punct(ref p) = tt {
            if p.as_char() == ',' && p.spacing() == proc_macro2::Spacing::Alone {
                let suffix: proc_macro2::TokenStream = iter.collect();
                return Some((prefix.into_iter().collect(), suffix));
            }
        }
        prefix.push(tt);
    }
    None
}

/// `syn::parse::Parser` adapter for the `pat [if guard]` tail of a
/// `matches!` invocation. Mirrors the std `matches!` macro grammar
/// (`std::macros::matches`): a single `Pat`, optionally followed by
/// `if Expr`. The `Pat` parse uses `Pat::parse_multi_with_leading_vert`
/// so top-level `|` alternations (`Some(_) | None`) are accepted.
fn parse_matches_pat_and_guard(
    input: syn::parse::ParseStream,
) -> syn::Result<(syn::Pat, Option<syn::Expr>)> {
    let pat = syn::Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(syn::Token![if]) {
        let _: syn::Token![if] = input.parse()?;
        Some(input.parse::<syn::Expr>()?)
    } else {
        None
    };
    Ok((pat, guard))
}

fn classify_match_bool_exitcases(arms: &[syn::Arm]) -> Option<Vec<Vec<ExitCase>>> {
    if arms.len() != 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen_bools = Vec::<bool>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            match classify_bool_pattern(sub_pat, is_last, &seen_bools)? {
                BoolPatternCase::Value(value) => {
                    if seen_bools.contains(&value) {
                        return None;
                    }
                    seen_bools.push(value);
                    arm_cases.push(ExitCase::Bool(value));
                }
                BoolPatternCase::Default(values) => {
                    if values.is_empty() {
                        return None;
                    }
                    for value in values {
                        seen_bools.push(value);
                        arm_cases.push(ExitCase::Bool(value));
                    }
                }
            }
        }
        all.push(arm_cases);
    }
    let mut sorted = seen_bools;
    sorted.sort();
    sorted.dedup();
    if sorted.as_slice() == &[false, true] {
        Some(all)
    } else {
        None
    }
}

enum BoolPatternCase {
    Value(bool),
    Default(Vec<bool>),
}

fn classify_bool_pattern(
    pat: &syn::Pat,
    is_last: bool,
    seen_bools: &[bool],
) -> Option<BoolPatternCase> {
    match pat {
        syn::Pat::Wild(_) if is_last => {
            let mut values = Vec::new();
            if !seen_bools.contains(&false) {
                values.push(false);
            }
            if !seen_bools.contains(&true) {
                values.push(true);
            }
            Some(BoolPatternCase::Default(values))
        }
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Bool(value) => Some(BoolPatternCase::Value(value.value)),
            _ => None,
        },
        _ => None,
    }
}

fn classify_match_switch_exitcases(arms: &[syn::Arm]) -> Option<Vec<Vec<ExitCase>>> {
    if arms.len() < 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen = Vec::<ExitCase>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            let exitcase = classify_switch_pattern(sub_pat, is_last)?;
            if !is_default_exitcase(&exitcase) {
                if seen.iter().any(|existing| existing == &exitcase) {
                    return None;
                }
                seen.push(exitcase.clone());
            }
            arm_cases.push(exitcase);
        }
        all.push(arm_cases);
    }
    Some(all)
}

fn flatten_or_pattern<'a>(pat: &'a syn::Pat, out: &mut Vec<&'a syn::Pat>) {
    match pat {
        syn::Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                flatten_or_pattern(case, out);
            }
        }
        syn::Pat::Paren(paren) => flatten_or_pattern(&paren.pat, out),
        other => out.push(other),
    }
}

fn classify_switch_pattern(pat: &syn::Pat, is_last: bool) -> Option<ExitCase> {
    match pat {
        syn::Pat::Wild(_) if is_last => Some(default_exitcase()),
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit
                .base10_parse::<i64>()
                .ok()
                .map(|value| ExitCase::Const(ConstValue::Int(value))),
            syn::Lit::Byte(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            syn::Lit::Char(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            _ => None,
        },
        _ => None,
    }
}

fn default_exitcase() -> ExitCase {
    ExitCase::Const(ConstValue::byte_str("default"))
}

fn is_default_exitcase(exitcase: &ExitCase) -> bool {
    matches!(
        exitcase,
        ExitCase::Const(ConstValue::ByteStr(bytes)) if bytes.as_slice() == b"default"
    )
}

fn unary_op_name(op: &syn::UnOp) -> &'static str {
    match op {
        syn::UnOp::Deref(_) => "deref",
        syn::UnOp::Not(_) => "not",
        syn::UnOp::Neg(_) => "neg",
        _ => "unknown_unary",
    }
}

fn binary_op_name(op: &syn::BinOp) -> &'static str {
    match op {
        syn::BinOp::Add(_) => "add",
        syn::BinOp::Sub(_) => "sub",
        syn::BinOp::Mul(_) => "mul",
        syn::BinOp::Div(_) => "div",
        syn::BinOp::Rem(_) => "mod",
        syn::BinOp::And(_) => "and",
        syn::BinOp::Or(_) => "or",
        syn::BinOp::BitXor(_) => "bitxor",
        syn::BinOp::BitAnd(_) => "bitand",
        syn::BinOp::BitOr(_) => "bitor",
        syn::BinOp::Shl(_) => "lshift",
        syn::BinOp::Shr(_) => "rshift",
        syn::BinOp::Eq(_) => "eq",
        syn::BinOp::Lt(_) => "lt",
        syn::BinOp::Le(_) => "le",
        syn::BinOp::Ne(_) => "ne",
        syn::BinOp::Ge(_) => "ge",
        syn::BinOp::Gt(_) => "gt",
        syn::BinOp::AddAssign(_) => "add_assign",
        syn::BinOp::SubAssign(_) => "sub_assign",
        syn::BinOp::MulAssign(_) => "mul_assign",
        syn::BinOp::DivAssign(_) => "div_assign",
        syn::BinOp::RemAssign(_) => "mod_assign",
        syn::BinOp::BitXorAssign(_) => "bitxor_assign",
        syn::BinOp::BitAndAssign(_) => "bitand_assign",
        syn::BinOp::BitOrAssign(_) => "bitor_assign",
        syn::BinOp::ShlAssign(_) => "lshift_assign",
        syn::BinOp::ShrAssign(_) => "rshift_assign",
        _ => "unknown_binop",
    }
}

fn binary_result_value_type(
    graph: &FunctionGraph,
    lhs: ValueId,
    rhs: ValueId,
    op: &str,
) -> ValueType {
    if matches!(op, "eq" | "ne" | "lt" | "le" | "gt" | "ge") {
        return ValueType::Int;
    }

    let lhs_ty = graph_value_type(graph, lhs);
    let rhs_ty = graph_value_type(graph, rhs);
    match (lhs_ty, rhs_ty) {
        (Some(ValueType::Float), Some(ValueType::Float))
        | (Some(ValueType::Float), Some(ValueType::Int))
        | (Some(ValueType::Int), Some(ValueType::Float))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
            ) =>
        {
            ValueType::Float
        }
        (Some(ValueType::Int), Some(ValueType::Int))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "bitand"
                    | "bitor"
                    | "bitxor"
                    | "lshift"
                    | "rshift"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
                    | "bitand_assign"
                    | "bitor_assign"
                    | "bitxor_assign"
                    | "lshift_assign"
                    | "rshift_assign"
            ) =>
        {
            ValueType::Int
        }
        _ => ValueType::Unknown,
    }
}

fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

/// RPython: direct_call carries the exact callee graph identity.
/// Qualify single-segment bare function names with module prefix so that
/// `helper()` inside `mod a` produces `["a", "helper"]`, matching the
/// registered graph path.
fn canonical_call_target(expr: &syn::Expr, ctx: &GraphBuildContext) -> CallTarget {
    match expr {
        syn::Expr::Path(path) => {
            let mut segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if is_transparent_result_option_ctor(&segments)
                && !registered_function_path(&segments, ctx)
            {
                return CallTarget::synthetic_transparent_ctor(
                    segments
                        .last()
                        .expect("transparent ctor path is non-empty")
                        .clone(),
                );
            }
            if segments.len() == 1 && !ctx.module_prefix.is_empty() {
                let mut qualified = ctx
                    .module_prefix
                    .split("::")
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>();
                qualified.extend(segments);
                segments = qualified;
            }
            CallTarget::function_path(segments)
        }
        _ => CallTarget::UnsupportedExpr,
    }
}

fn is_transparent_result_option_ctor(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["Ok"]
            | ["Err"]
            | ["Some"]
            | ["Result", "Ok"]
            | ["Result", "Err"]
            | ["Option", "Some"]
            | ["result", "Result", "Ok"]
            | ["result", "Result", "Err"]
            | ["option", "Option", "Some"]
            | ["std", "result", "Result", "Ok"]
            | ["std", "result", "Result", "Err"]
            | ["std", "option", "Option", "Some"]
            | ["core", "result", "Result", "Ok"]
            | ["core", "result", "Result", "Err"]
            | ["core", "option", "Option", "Some"]
    )
}

fn registered_function_path(segments: &[String], ctx: &GraphBuildContext) -> bool {
    let unqualified = segments.join("::");
    if ctx.fn_return_types.contains_key(&unqualified) {
        return true;
    }
    if segments.len() == 1 && !ctx.module_prefix.is_empty() {
        let qualified = format!("{}::{}", ctx.module_prefix, segments[0]);
        return ctx.fn_return_types.contains_key(&qualified);
    }
    false
}

fn receiver_type_root(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Cast(cast) => {
            type_root_ident(&cast.ty).map(|root| qualify_type_name(&root, &ctx.module_prefix))
        }
        syn::Expr::Reference(reference) => receiver_type_root(&reference.expr, ctx),
        syn::Expr::Paren(paren) => receiver_type_root(&paren.expr, ctx),
        syn::Expr::Unary(unary) => match &unary.op {
            syn::UnOp::Deref(_) => receiver_type_root(&unary.expr, ctx),
            _ => None,
        },
        syn::Expr::Field(field) => receiver_type_root(&field.base, ctx),
        syn::Expr::Index(index) => receiver_type_root(&index.expr, ctx),
        _ => None,
    }
}

fn graph_result_value_type(graph: &FunctionGraph, value: ValueId) -> Option<ValueType> {
    graph
        .blocks
        .iter()
        .flat_map(|block| block.operations.iter())
        .find_map(|op| {
            if op.result == Some(value) {
                op_result_value_type(&op.kind)
            } else {
                None
            }
        })
}

fn graph_value_type(graph: &FunctionGraph, value: ValueId) -> Option<ValueType> {
    graph_result_value_type(graph, value).or_else(|| graph_link_input_value_type(graph, value))
}

fn retag_result_value_type(graph: &mut FunctionGraph, value: ValueId, ty: ValueType) {
    for block in &mut graph.blocks {
        for op in &mut block.operations {
            if op.result != Some(value) {
                continue;
            }
            match &mut op.kind {
                OpKind::Input { ty: result_ty, .. }
                | OpKind::FieldRead { ty: result_ty, .. }
                | OpKind::VableFieldRead { ty: result_ty, .. }
                | OpKind::BinOp { result_ty, .. }
                | OpKind::UnaryOp { result_ty, .. }
                | OpKind::Call { result_ty, .. }
                | OpKind::IndirectCall { result_ty, .. } => *result_ty = ty,
                OpKind::ArrayRead { item_ty, .. }
                | OpKind::InteriorFieldRead { item_ty, .. }
                | OpKind::VableArrayRead { item_ty, .. } => *item_ty = ty,
                _ => {}
            }
            return;
        }
    }
}

fn graph_link_input_value_type(graph: &FunctionGraph, value: ValueId) -> Option<ValueType> {
    for target_block in &graph.blocks {
        let Some(arg_index) = target_block
            .inputargs
            .iter()
            .position(|&inputarg| inputarg == value)
        else {
            continue;
        };
        let mut inferred: Option<ValueType> = None;
        for predecessor in &graph.blocks {
            for link in &predecessor.exits {
                if link.target != target_block.id {
                    continue;
                }
                let source_ty = match link.args.get(arg_index)? {
                    LinkArg::Value(source) => match graph_result_value_type(graph, *source) {
                        Some(ty) => ty,
                        None => continue,
                    },
                    // RPython `flowspace/model.py:Constant.concretetype`
                    // — `Link.args` may carry constants whose lltype is
                    // determined by the constant's Python class; the
                    // inputarg's concretetype is unified across all
                    // predecessor links the same way variable sources
                    // are.  Skipping constants leaves the inputarg
                    // Unknown, which the rtyper backfills with GcRef
                    // and forces synthetic casts at int/float
                    // operations downstream.
                    LinkArg::Const(c) => match const_value_value_type(c) {
                        Some(ty) => ty,
                        None => continue,
                    },
                };
                match &inferred {
                    None => inferred = Some(source_ty),
                    Some(existing) if *existing == source_ty => {}
                    Some(_) => return None,
                }
            }
        }
        if inferred.is_some() {
            return inferred;
        }
    }
    None
}

/// RPython `flowspace/model.py:Constant.concretetype` — map a Python
/// constant value to its lltype kind.  Used by
/// `graph_link_input_value_type` to infer phi-input concretetype from
/// constant link args.  `Placeholder` is unmaterialised by definition
/// and never appears in production link args.
fn const_value_value_type(c: &ConstValue) -> Option<ValueType> {
    match c {
        // RPython int/bool both map to `lltype.Signed`
        // (`rpython/rtyper/rint.py`); booleans flow through int handlers.
        ConstValue::Int(_) | ConstValue::Bool(_) => Some(ValueType::Int),
        ConstValue::Float(_) => Some(ValueType::Float),
        // GC-managed Python objects → `lltype.Ptr(GcStruct)` (Ref bank).
        ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::None
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::Atom(_)
        | ConstValue::LLPtr(_)
        | ConstValue::HostObject(_) => Some(ValueType::Ref),
        // `LowLevelType` constants are `lltype.Void` carriers (the
        // value IS a TYPE object); RPython flow `Constant(TYPE,
        // lltype.Void)` — Void register class.
        ConstValue::LowLevelType(_) => Some(ValueType::Void),
        // `_address` is RPython's `Address` lowleveltype — distinct
        // from GcRef and Signed.  Pyre has no Address bank, but the
        // value is a raw pointer-sized integer in practice; conservative
        // None lets the rtyper Unknown→GcRef fallback handle it as it
        // does today.
        ConstValue::LLAddress(_) => None,
        // SpecTag identity carrier — never feeds an int/float/ref op.
        ConstValue::SpecTag(_) => None,
        ConstValue::Placeholder => None,
    }
}

fn op_result_value_type(kind: &OpKind) -> Option<ValueType> {
    match kind {
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. }
        | OpKind::IndirectCall { result_ty: ty, .. } => {
            if *ty == ValueType::Unknown {
                None
            } else {
                Some(ty.clone())
            }
        }
        OpKind::ConstInt(_) | OpKind::VtableMethodPtr { .. } | OpKind::CurrentTraceLength => {
            Some(ValueType::Int)
        }
        OpKind::ConstFloat(_) => Some(ValueType::Float),
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => {
            if *item_ty == ValueType::Unknown {
                None
            } else {
                Some(item_ty.clone())
            }
        }
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } => Some(ValueType::Int),
        _ => None,
    }
}

fn transparent_option_method_result_type(
    graph: &FunctionGraph,
    args: &[ValueId],
    method: &syn::Ident,
) -> Option<ValueType> {
    match method.to_string().as_str() {
        "as_usize" | "len" | "is_empty" | "is_null" | "wrapping_mul" => Some(ValueType::Int),
        "unwrap_or" => args
            .get(1)
            .and_then(|&default| graph_value_type(graph, default)),
        _ => None,
    }
}

/// Inference for Rust `f64`/i64 method calls used as low-level
/// floating-point/integer helpers in pyre's port of
/// `pypy/objspace/std/floatobject.py`.
///
/// **Structural adaptation (parity rule §1):** RPython spells the
/// same low-level operations as `r_float`/`r_uint`/`r_longlong`
/// helpers (`rpython/rlib/rfloat.py`, `rpython/rlib/rarithmetic.py`),
/// not as method calls.  The Rust port relays them through `f64::*`
/// and `i64::*` because that is the only way to express the same
/// arithmetic in stable Rust.  The receiver-typed match below stamps
/// the result class so jtransform sees the same concretetype the
/// rtyper would have written onto a Variable for an `r_float` /
/// `r_uint` operand.
///
/// This mapping is for the LOW-LEVEL helpers only — it does NOT
/// translate Python-level `float.__floor__` / `int.__abs__`.  The
/// PyPy descriptors (`pypy/objspace/std/floatobject.py:descr_floor`)
/// box-return `int`, whereas Rust `f64::floor` returns `f64`.
/// Callers must already have lowered to the helper level (e.g.
/// `floatobject.py` body, not the unboxed Python descriptor) before
/// this inference applies.
fn primitive_method_result_type(
    graph: &FunctionGraph,
    args: &[ValueId],
    method: &syn::Ident,
) -> Option<ValueType> {
    let receiver = args
        .first()
        .and_then(|&recv| graph_value_type(graph, recv))?;
    match (receiver, method.to_string().as_str()) {
        (ValueType::Float, "abs" | "floor" | "ceil" | "trunc" | "round" | "sqrt" | "powf") => {
            Some(ValueType::Float)
        }
        (ValueType::Float, "is_nan" | "is_infinite" | "is_finite" | "is_sign_negative") => {
            Some(ValueType::Int)
        }
        (
            ValueType::Int,
            "abs" | "wrapping_abs" | "wrapping_mul" | "wrapping_add" | "wrapping_sub",
        ) => Some(ValueType::Int),
        _ => None,
    }
}

fn intrinsic_call_result_type(segments: &[String]) -> Option<ValueType> {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    match path.as_slice() {
        // Rust's `size_of::<T>()` is a compile-time `usize`.  RPython
        // carries such layout sizes as `lltype.Signed` constants before
        // codewriter, so the equivalent register class is Int.
        ["std", "mem", "size_of"] | ["core", "mem", "size_of"] | ["mem", "size_of"] => {
            Some(ValueType::Int)
        }
        // Associated f64 helpers used by the floatobject.py port.
        ["f64", "copysign"] | ["std", "f64", "copysign"] | ["core", "f64", "copysign"] => {
            Some(ValueType::Float)
        }
        _ => None,
    }
}

fn kind_char_to_value_type(kind: char) -> Option<ValueType> {
    match kind {
        'i' => Some(ValueType::Int),
        'r' => Some(ValueType::Ref),
        'f' => Some(ValueType::Float),
        'v' => Some(ValueType::Void),
        _ => None,
    }
}

fn cast_op_name(source_ty: Option<&ValueType>, target_ty: &ValueType) -> &'static str {
    match (source_ty, target_ty) {
        (Some(ValueType::Int), ValueType::Float) => "cast_int_to_float",
        (Some(ValueType::Float), ValueType::Int) => "cast_float_to_int",
        (Some(ValueType::Ref), ValueType::Int) => "cast_ptr_to_int",
        (Some(ValueType::Int), ValueType::Ref) => "cast_int_to_ptr",
        _ => "same_as",
    }
}

fn lookup_method_return_type<'a>(
    ctx: &'a GraphBuildContext,
    receiver_root: Option<&str>,
    method: &syn::Ident,
) -> Option<&'a String> {
    let receiver_root = receiver_root?;
    let exact = format!("{}::{}", receiver_root, method);
    if let Some(ret) = ctx.fn_return_types.get(&exact) {
        return Some(ret);
    }

    let receiver_leaf = receiver_root.rsplit("::").next().unwrap_or(receiver_root);
    let method_name = method.to_string();
    // Rust imports can make the call-site owner path shorter or longer
    // than the impl key. Use the leaf owner only when it is unambiguous.
    let mut matches = ctx.fn_return_types.iter().filter_map(|(key, ret)| {
        let (owner, candidate_method) = key.rsplit_once("::")?;
        if candidate_method == method_name.as_str()
            && owner.rsplit("::").next().unwrap_or(owner) == receiver_leaf
        {
            Some(ret)
        } else {
            None
        }
    });
    let first = matches.next()?;
    if matches.next().is_none() {
        Some(first)
    } else {
        None
    }
}

/// Extract the trait root from a type-string when the type is a trait
/// object — direct (`"dyn Foo"`) or wrapped (`"Box<dyn Foo>"`,
/// `"Rc<dyn Foo>"`, `"Arc<dyn Foo>"`).  The trailing `+ 'a` lifetime
/// bound is stripped.  Returns `None` for non-dyn types.
fn dyn_trait_root_from_type_str(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("dyn ") {
        // `dyn Trait + 'a` — drop everything after the first `+`.
        let head = rest.split('+').next()?.trim();
        if head.is_empty() {
            return None;
        }
        return Some(head.to_string());
    }
    for wrapper in ["Box", "Rc", "Arc"] {
        let prefix = format!("{wrapper}<");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str())
            && let Some(inner) = rest.strip_suffix('>')
        {
            return dyn_trait_root_from_type_str(inner);
        }
    }
    None
}

/// Return the trait root when the receiver's static type is a
/// `dyn Trait` (including `&dyn T` / `&mut dyn T` / `Box<dyn T>`),
/// otherwise `None`.  Looks up local/parameter bindings via
/// `ctx.local_dyn_trait_roots`, struct field types via
/// `ctx.struct_fields`, array element types via
/// `ctx.local_array_types`, and chained method-call / free-call
/// return types via `ctx.fn_return_types`.
fn dyn_trait_root_for_receiver(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        // Local/parameter bound to `dyn Trait` — directly mapped in
        // `local_dyn_trait_roots`.
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_dyn_trait_roots.get(&ident.to_string()).cloned()),
        // Strip wrappers that don't change the static type's trait-ness.
        syn::Expr::Reference(reference) => dyn_trait_root_for_receiver(&reference.expr, ctx),
        syn::Expr::Paren(paren) => dyn_trait_root_for_receiver(&paren.expr, ctx),
        syn::Expr::Group(group) => dyn_trait_root_for_receiver(&group.expr, ctx),
        // `self.handler.run()` — resolve `self.handler`'s declared field
        // type via `struct_fields[owner_type][handler]`, then check for
        // `dyn` / `Box<dyn>` / wrapper.
        syn::Expr::Field(field) => {
            let owner = receiver_type_root(&field.base, ctx)?;
            let field_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(_) => return None,
            };
            let field_type = ctx.struct_fields.field_type(&owner, &field_name)?;
            dyn_trait_root_from_type_str(field_type)
        }
        // `handlers[i].run()` — `handlers`'s declared full type is
        // tracked in `local_array_types` (e.g. `"Vec<Box<dyn T>>"`);
        // strip the container wrapper to get the element type, then
        // check whether that element is a trait object.
        syn::Expr::Index(index) => {
            let container = match &*index.expr {
                syn::Expr::Path(path) => path
                    .path
                    .get_ident()
                    .and_then(|ident| ctx.local_array_types.get(&ident.to_string()).cloned()),
                _ => None,
            }?;
            let elem = extract_element_type_from_str(&container)?;
            dyn_trait_root_from_type_str(&elem)
        }
        // Chained `x.foo().bar()` — look up `x.foo`'s declared return
        // type, accepting plain `dyn T` AND wrapped (`Box<dyn T>`).
        syn::Expr::MethodCall(mc) => {
            let owner = receiver_type_root(&mc.receiver, ctx)?;
            let key = format!("{}::{}", owner, mc.method);
            let ret = ctx.fn_return_types.get(&key)?;
            dyn_trait_root_from_type_str(ret)
        }
        // Chained `foo().bar()` — free function return type, same wrapper
        // recognition as the method-call branch.
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(p) = &*call.func {
                let key = p
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                let ret = ctx.fn_return_types.get(&key)?;
                return dyn_trait_root_from_type_str(ret);
            }
            None
        }
        _ => None,
    }
}

fn canonical_pat_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(ident) => ident.ident.to_string(),
        syn::Pat::Reference(reference) => canonical_pat_name(&reference.pat),
        syn::Pat::Type(typed) => canonical_pat_name(&typed.pat),
        syn::Pat::TupleStruct(tuple_struct) => tuple_struct
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Struct(strukt) => strukt
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Tuple(_) => "tuple_pat".into(),
        syn::Pat::Slice(_) => "slice_pat".into(),
        syn::Pat::Lit(_) => "lit_pat".into(),
        syn::Pat::Path(_) => "path_pat".into(),
        syn::Pat::Wild(_) => "_".into(),
        syn::Pat::Or(_) => "or_pat".into(),
        syn::Pat::Range(_) => "range_pat".into(),
        syn::Pat::Macro(_) => "macro_pat".into(),
        syn::Pat::Paren(paren) => canonical_pat_name(&paren.pat),
        _ => "unsupported_pat".into(),
    }
}

fn bind_pattern_locals(
    pat: &syn::Pat,
    matched_type: Option<&str>,
    ctx: &mut GraphBuildContext<'_>,
) {
    match pat {
        syn::Pat::Ident(ident) => {
            if let Some(type_str) = matched_type {
                bind_ident_type(&ident.ident, type_str, ctx);
            }
        }
        syn::Pat::Reference(reference) => bind_pattern_locals(&reference.pat, matched_type, ctx),
        syn::Pat::Paren(paren) => bind_pattern_locals(&paren.pat, matched_type, ctx),
        syn::Pat::Type(typed) => {
            let explicit_type = qualified_full_type_string(
                &typed.ty,
                &ctx.module_prefix,
                ctx.known_struct_names,
                ctx.known_trait_names,
            );
            bind_pattern_locals(&typed.pat, explicit_type.as_deref().or(matched_type), ctx);
        }
        syn::Pat::TupleStruct(tuple_struct) => {
            let path: Vec<String> = tuple_struct
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if matches_constructor_path(&path, "Some") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_option_inner_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            } else if matches_constructor_path(&path, "Ok") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_result_ok_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            } else if matches_constructor_path(&path, "Err") {
                if let (Some(inner), Some(inner_pat)) = (
                    matched_type.and_then(transparent_result_err_type),
                    tuple_struct.elems.first(),
                ) {
                    bind_pattern_locals(inner_pat, Some(inner), ctx);
                }
            }
        }
        syn::Pat::Struct(pat_struct) => {
            // RPython rtyper field-access shape: pattern destructure on
            // `Enum::Variant { f, .. }` resolves each field's concretetype
            // through the per-class field table (see Item::Enum branch of
            // `collect_fields_and_returns`).  Pyre's
            // `StructFieldRegistry::field_type` runs a suffix walk on the
            // owner key, so both fully-qualified
            // (`majit_ir::RdVirtualInfo::VirtualInfo`) and bare
            // (`VirtualInfo`) destructure paths land on the same field set.
            let owner: String = pat_struct
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            for field_pat in &pat_struct.fields {
                let field_name = match &field_pat.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                let field_type = ctx
                    .struct_fields
                    .field_type(&owner, &field_name)
                    .map(|s| s.to_string());
                bind_pattern_locals(&field_pat.pat, field_type.as_deref(), ctx);
            }
        }
        syn::Pat::Or(pat_or) => {
            // RPython `flowspace/flowcontext.py` does not have or-patterns;
            // `match A | B { kind, .. } => ...` desugars to two parallel
            // arms in upstream's translator.  Recurse into each case so
            // shared destructure names get bound under each variant's
            // concretetype — both variants are required to expose the
            // same field set, so the resulting concretetype is consistent
            // across the cases.
            for case in &pat_or.cases {
                bind_pattern_locals(case, matched_type, ctx);
            }
        }
        syn::Pat::Tuple(pat_tuple) => {
            // RPython `BUILD_TUPLE_UNPACK` (`flowspace/flowcontext.py`)
            // unpacks each element with the per-position concretetype
            // recorded on the source `SomeTuple`.  The matched type
            // string carries the parenthesised list of element types
            // (`(VirtualKind, &[FieldDescrInfo], &[i16], usize)`);
            // `split_tuple_type_elements` walks balanced angle / paren /
            // bracket depth so nested generics (`Option<Result<T, E>>`)
            // do not split prematurely.  When the element count
            // disagrees with the type-string arity, fall back to
            // unbound recursion — better than asserting on a
            // tuple-typed value the inference rules do not yet handle.
            let elem_types = matched_type.and_then(split_tuple_type_elements);
            for (idx, elem_pat) in pat_tuple.elems.iter().enumerate() {
                let elem_type = elem_types
                    .as_ref()
                    .and_then(|v| v.get(idx))
                    .map(|s| s.as_str());
                bind_pattern_locals(elem_pat, elem_type, ctx);
            }
        }
        _ => {}
    }
}

/// Split a parenthesised tuple type string into its element types.
/// Mirrors `extract_element_type_from_str`'s prefix walk for `&` /
/// `*const` / `*mut` so a `&(A, B)` reference is treated as `(A, B)`,
/// then walks balanced `<>` / `()` / `[]` depth so nested generics
/// (`Option<Result<T, E>>`) and inner tuples (`Vec<(A, B)>`) survive
/// the split intact.
fn split_tuple_type_elements(type_str: &str) -> Option<Vec<String>> {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    let inner = s.strip_prefix('(')?.strip_suffix(')')?;
    let mut elements: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;
    for ch in inner.chars() {
        match ch {
            '(' | '<' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    elements.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let last = current.trim();
    if !last.is_empty() {
        elements.push(last.to_string());
    }
    Some(elements)
}

fn bind_ident_type(ident: &syn::Ident, type_str: &str, ctx: &mut GraphBuildContext<'_>) {
    let name = ident.to_string();
    ctx.local_value_types
        .insert(name.clone(), type_string_to_value_type(type_str));
    let trimmed = type_str.trim().to_string();
    ctx.local_type_strings.insert(name.clone(), trimmed.clone());
    // Mirror Stmt::Local Pat::Type binding (line 1085-1087 / 1318-1320):
    // every named local with a known full-type string seeds
    // `local_array_types` so that downstream
    // `array_type_id_from_expr` / `extract_element_type_from_str`
    // resolve the element type via the same channel an explicit
    // `let x: Vec<T> = ...` would.
    ctx.local_array_types.insert(name.clone(), trimmed);
    if let Some(root) = type_root_from_type_string(type_str) {
        ctx.local_type_roots.insert(name, root);
    }
}

fn matches_constructor_path(path: &[String], leaf: &str) -> bool {
    path.last().is_some_and(|last| last == leaf)
}

/// Extract the head identifier from a Rust type string for
/// receiver / method / field-owner lookups.
///
/// **Structural adaptation (parity rule §1):** RPython carries
/// `concretetype` as an `lltype.Ptr(GcStruct)` object whose identity
/// is structural; field/method lookups compare type objects directly.
/// Rust's `syn` AST surfaces types as strings, and the analyser
/// keeps them as strings throughout, so type identity resolves by
/// head-identifier match.  Wrapper info (`Box<T>`, `Vec<T>`,
/// generic args, lifetime params) is discarded here — the caller is
/// expected to have already applied the appropriate transparent-
/// container unwrapping (`extract_element_type_from_str` for arrays,
/// `transparent_option_inner_type` etc.) when the wrapper carries
/// payload type information.  Two distinct types that happen to share
/// a head identifier (e.g. via `use crate::foo::Bar` and
/// `use crate::baz::Bar`) will collide; pyre does not currently
/// disambiguate, mirroring the analyser's flat name table.  The
/// raw-pointer / reference prefix strip preserves single-identity
/// behaviour for `let x = obj as *mut Foo;` bindings.
fn type_root_from_type_string(type_str: &str) -> Option<String> {
    let mut trimmed = type_str.trim();
    loop {
        let stripped = trimmed
            .strip_prefix("*const ")
            .or_else(|| trimmed.strip_prefix("*mut "))
            .or_else(|| trimmed.strip_prefix("&mut "))
            .or_else(|| trimmed.strip_prefix("&"));
        match stripped {
            Some(rest) => trimmed = rest.trim_start(),
            None => break,
        }
    }
    if trimmed.is_empty() || is_primitive_type_string(trimmed) {
        return None;
    }
    let head = trimmed
        .split(['<', ' ', '('])
        .next()
        .unwrap_or(trimmed)
        .trim();
    if head.is_empty() {
        None
    } else {
        Some(head.to_string())
    }
}

fn is_primitive_type_string(type_str: &str) -> bool {
    matches!(
        type_str,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "char"
            | "f32"
            | "f64"
            | "()"
    )
}

/// RPython: lltype graph identity — returns the full type path.
/// For `Foo` → "Foo", for `a::Foo` → "a::Foo".
/// Classify a Rust parameter/return `syn::Type` into one of the three
/// RPython `lltype` register classes (`Int`/`Ref`/`Float`).  This is the
/// pyre-side bridge for what RPython does implicitly: each `Variable`
/// carries `concretetype`, and `getkind(concretetype)` picks the class
/// (`rpython/jit/codewriter/support.py:getkind`).  pyre's front-end
/// records only a `syn::Type` so we reproduce the mapping here.
///
/// Returned value is assigned to `OpKind::Input { ty }` so the annotator
/// + rtyper reach every function parameter with a concrete class; the
/// assembler's `lookup_reg_with_kind` then finds a coloring for every
/// operand it encounters, matching upstream's invariant that every
/// Variable reaching `assembler.py:write_insn` has a `concretetype`.
pub(crate) fn classify_fn_arg_ty(ty: &syn::Type) -> crate::model::ValueType {
    use crate::model::ValueType;
    match ty {
        syn::Type::Path(path) => {
            let last = match path.path.segments.last() {
                Some(s) => s,
                None => return ValueType::Ref,
            };
            if path.path.segments.len() == 2
                && path.path.segments[0].ident == "Self"
                && path.path.segments[1].ident == "Truth"
            {
                return ValueType::Int;
            }
            let name = last.ident.to_string();
            // `Box<T>` / `Rc<T>` / `Arc<T>` — classify on the inner type
            // so `Box<i64>` stays Int (RPython `lltype.Ptr(Signed)`
            // collapses to the primitive), matching the downstream
            // `ValueType::Ref` vs `Int` distinction the assembler keys
            // off.
            if matches!(name.as_str(), "Box" | "Rc" | "Arc") {
                if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            return classify_fn_arg_ty(inner);
                        }
                    }
                }
                return ValueType::Ref;
            }
            match name.as_str() {
                // RPython `rlib/rarithmetic.py` + `lltype.Signed` family.
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
                | "bool" | "char" => ValueType::Int,
                // `lltype.Float` — `f32` widens up to f64 at the SSA
                // level but stays in the Float class either way.
                "f32" | "f64" => ValueType::Float,
                // Anything else is a user type / GC ref / opaque struct.
                _ => ValueType::Ref,
            }
        }
        // `&T` / `&mut T` — pointer → Ref (lltype.Ptr in RPython).
        syn::Type::Reference(_) => ValueType::Ref,
        // `*const T` / `*mut T` — raw pointer, same class as Ref.  pyre
        // often stores GC objects as `*mut PyObject`; classify as Ref
        // so field/array bases reach the canonical `/rd>X` encoding
        // rather than the pyre-only `*_intbase` aliases.
        syn::Type::Ptr(_) => ValueType::Ref,
        syn::Type::Paren(paren) => classify_fn_arg_ty(&paren.elem),
        syn::Type::Group(group) => classify_fn_arg_ty(&group.elem),
        // `dyn Trait` — GC pointer to a trait object.
        syn::Type::TraitObject(_) => ValueType::Ref,
        // Tuple/array/slice: treat as Ref (bulk data, not a register
        // primitive).  RPython `lltype.Array` + `lltype.Struct` both
        // flatten to `lltype.Ptr` at the call-site boundary.
        syn::Type::Tuple(_) | syn::Type::Array(_) | syn::Type::Slice(_) => ValueType::Ref,
        // `fn(T) -> T`, `impl Trait`, never — no runtime
        // representation reaches the SSA level; default to Ref for
        // safe-by-default classification.
        _ => ValueType::Ref,
    }
}

/// RPython's lltype.Struct objects have globally unique identities;
/// returning all path segments ensures `a::Foo` and `b::Foo` don't alias.
fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            // `Box<dyn Trait>` / `Rc<dyn Trait>` / `Arc<dyn Trait>` —
            // unwrap the first generic arg and try again; the resulting
            // root identifies the trait, not the container.
            if let Some(last) = path.path.segments.last() {
                let wrapper = last.ident.to_string();
                if matches!(wrapper.as_str(), "Box" | "Rc" | "Arc") {
                    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(inner) = arg {
                                if let Some(root) = type_root_ident(inner) {
                                    return Some(root);
                                }
                            }
                        }
                    }
                }
            }
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("::"))
            }
        }
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Ptr(ptr) => type_root_ident(&ptr.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        // `dyn Trait + 'a` / `&mut dyn Trait` (after deref) — return the
        // first trait bound's canonical path, rendered as `dyn <Trait>` so
        // callers can tell this is a trait object.
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque type (compiler monomorphizes
        // each call site to a single concrete impl), not runtime
        // family-dispatch.  RPython `indirect_call` is reserved for
        // truly polymorphic callees (`rpython/jit/codewriter/call.py:103
        // graphs_from`); treat impl Trait the same way concrete-type
        // method calls are treated and bail out so downstream emits
        // CallTarget::Method, not CallTarget::Indirect.
        syn::Type::ImplTrait(_) => None,
        _ => None,
    }
}

fn trait_bound_root_for_receiver(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_trait_bound_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => trait_bound_root_for_receiver(&reference.expr, ctx),
        syn::Expr::Paren(paren) => trait_bound_root_for_receiver(&paren.expr, ctx),
        syn::Expr::Group(group) => trait_bound_root_for_receiver(&group.expr, ctx),
        syn::Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            trait_bound_root_for_receiver(&unary.expr, ctx)
        }
        _ => None,
    }
}

/// Extract the declaring trait name from a `dyn T + 'a` bound list:
/// returns the first `T::Trait`-style bound's canonical path.
/// Used by `type_root_ident` / `full_type_string` / `extract_dyn_trait_root`
/// to identify the indirect-call family key.
fn trait_object_root_name(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
) -> Option<String> {
    bounds.iter().find_map(|b| match b {
        syn::TypeParamBound::Trait(t) => Some(
            t.path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        ),
        _ => None,
    })
}

fn collect_generic_trait_roots(
    generics: &syn::Generics,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> HashMap<String, String> {
    let mut roots = HashMap::new();
    for param in &generics.params {
        if let syn::GenericParam::Type(type_param) = param {
            if let Some(root) =
                trait_object_root_name_qualified(&type_param.bounds, prefix, known_trait_names)
            {
                roots.insert(type_param.ident.to_string(), root);
            }
        }
    }
    if let Some(where_clause) = &generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pred_ty) = predicate {
                let Some(type_name) = type_root_ident(&pred_ty.bounded_ty) else {
                    continue;
                };
                if let Some(root) =
                    trait_object_root_name_qualified(&pred_ty.bounds, prefix, known_trait_names)
                {
                    roots.insert(type_name, root);
                }
            }
        }
    }
    roots
}

fn qualify_known_trait_name(
    bare: &str,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let qualified = if prefix.is_empty() || bare.contains("::") {
        None
    } else {
        Some(format!("{}::{}", prefix, bare))
    };
    if let Some(qualified) = qualified {
        if known_trait_names.contains(&qualified) {
            qualified
        } else {
            bare.to_string()
        }
    } else {
        bare.to_string()
    }
}

fn trait_object_root_name_qualified(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    trait_object_root_name(bounds)
        .map(|name| qualify_known_trait_name(&name, prefix, known_trait_names))
}

/// Returns the bare trait root (no `dyn ` prefix) when `ty` denotes a
/// `dyn Trait` / `&dyn Trait` / `Box<dyn Trait>` receiver; `None`
/// otherwise.  Used by method-call lowering to decide whether the call
/// should be modeled as an RPython `indirect_call`
/// (`rewrite_op_indirect_call` entrypoint).
pub fn extract_dyn_trait_root(ty: &syn::Type) -> Option<String> {
    extract_dyn_trait_root_with_context(ty, "", &std::collections::HashSet::new())
}

fn extract_dyn_trait_root_with_context(
    ty: &syn::Type,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    match ty {
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        // `impl Trait` is a static opaque type — no runtime family-dispatch.
        // See `type_root_ident`'s ImplTrait arm for the rationale + RPython cite.
        syn::Type::ImplTrait(_) => None,
        syn::Type::Reference(r) => {
            extract_dyn_trait_root_with_context(&r.elem, prefix, known_trait_names)
        }
        syn::Type::Paren(p) => {
            extract_dyn_trait_root_with_context(&p.elem, prefix, known_trait_names)
        }
        syn::Type::Group(g) => {
            extract_dyn_trait_root_with_context(&g.elem, prefix, known_trait_names)
        }
        syn::Type::Path(path) => {
            // `Box<dyn Trait>` / `Rc<dyn Trait>` / `Arc<dyn Trait>`.
            let last = path.path.segments.last()?;
            if !matches!(last.ident.to_string().as_str(), "Box" | "Rc" | "Arc") {
                return None;
            }
            if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        if let Some(r) =
                            extract_dyn_trait_root_with_context(inner, prefix, known_trait_names)
                        {
                            return Some(r);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}

/// Canonical type string for a syn::Type.
///
/// Produces a string that includes generic arguments,
/// e.g. `Vec<Point>` → `"Vec<Point>"`, `Point` → `"Point"`.
pub fn full_type_string(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => name,
                        syn::PathArguments::AngleBracketed(args) => {
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => full_type_string(t),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => full_type_string(&r.elem),
        syn::Type::Ptr(p) => {
            let inner = full_type_string(&p.elem)?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => full_type_string(&p.elem),
        syn::Type::Group(g) => full_type_string(&g.elem),
        syn::Type::Slice(s) => full_type_string(&s.elem).map(|t| format!("[{}]", t)),
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque type — render as the underlying
        // bound name without the `dyn ` prefix so downstream consumers
        // do not mistake it for a trait object (see `type_root_ident`).
        syn::Type::ImplTrait(obj) => trait_object_root_name(&obj.bounds),
        // RPython: ARRAY identity preserves full type including length.
        // [Point; 4] and [Point; 8] are different ARRAY types.
        syn::Type::Array(a) => {
            let elem = full_type_string(&a.elem)?;
            // Extract length from Expr::Lit if possible.
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t.elems.iter().map(full_type_string).collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        _ => None,
    }
}

/// RPython: lltype identity — `full_type_string` with module-prefix qualification.
///
/// RPython's `T.TO` always returns the actual lltype object.
/// This function qualifies single-segment leaf types that are KNOWN structs
/// (in `known_struct_names`) with the module prefix, so `Bar` in `mod a`
/// becomes `a::Bar`. Uses the actual struct name set, not a heuristic.
pub(crate) fn qualified_full_type_string(
    ty: &syn::Type,
    prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    if prefix.is_empty() {
        return full_type_string(ty);
    }
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => {
                            // Leaf type (no generics). Qualify if it looks like
                            // a user struct: starts with uppercase, single segment.
                            if path.path.segments.len() == 1 && known_struct_names.contains(&name) {
                                qualify_type_name(&name, prefix)
                            } else {
                                name
                            }
                        }
                        syn::PathArguments::AngleBracketed(args) => {
                            // Container<T,...> — qualify inner types, not the container.
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => qualified_full_type_string(
                                        t,
                                        prefix,
                                        known_struct_names,
                                        known_trait_names,
                                    ),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => {
            qualified_full_type_string(&r.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Ptr(p) => {
            let inner =
                qualified_full_type_string(&p.elem, prefix, known_struct_names, known_trait_names)?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => {
            qualified_full_type_string(&p.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Group(g) => {
            qualified_full_type_string(&g.elem, prefix, known_struct_names, known_trait_names)
        }
        syn::Type::Slice(s) => {
            qualified_full_type_string(&s.elem, prefix, known_struct_names, known_trait_names)
                .map(|t| format!("[{}]", t))
        }
        syn::Type::Array(a) => {
            let elem =
                qualified_full_type_string(&a.elem, prefix, known_struct_names, known_trait_names)?;
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t
                .elems
                .iter()
                .map(|elem| {
                    qualified_full_type_string(elem, prefix, known_struct_names, known_trait_names)
                })
                .collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
                .map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque — render the bound name without
        // the `dyn ` marker.  See `type_root_ident` for the full rationale.
        syn::Type::ImplTrait(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        _ => None,
    }
}

/// RPython: resolve ARRAY identity from an expression.
///
/// RPython: `getkind(TYPE)[0]` — map type string to ValueType for kind suffix.
/// Used by InteriorFieldRead/Write to determine the i/r/f suffix.
fn type_string_to_value_type(type_str: &str) -> ValueType {
    let type_str = type_str.trim();
    match type_str {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "bool" | "char" | "Self::Truth" => ValueType::Int,
        "f32" | "f64" => ValueType::Float,
        "()" => ValueType::Void,
        _ => ValueType::Ref,
    }
}

fn transparent_result_ok_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let ok_type = first_top_level_generic_arg(inner).map(str::trim)?;
        if ok_type == "()" {
            return None;
        }
        return Some(ok_type);
    }
    None
}

fn transparent_result_err_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let err_type = second_top_level_generic_arg(inner).map(str::trim)?;
        if err_type == "()" {
            return None;
        }
        return Some(err_type);
    }
    None
}

fn transparent_option_inner_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Option<", "std::option::Option<", "core::option::Option<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        return first_top_level_generic_arg(inner).map(str::trim);
    }
    None
}

fn first_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Some(&args[..idx]),
            _ => {}
        }
    }
    if args.is_empty() { None } else { Some(args) }
}

fn second_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let rest = args[idx + 1..].trim();
                return if rest.is_empty() { None } else { Some(rest) };
            }
            _ => {}
        }
    }
    None
}

fn field_value_type_from_expr(
    base: &syn::Expr,
    member: &syn::Member,
    ctx: &GraphBuildContext,
) -> Option<ValueType> {
    field_type_string_from_expr(base, member, ctx)
        .map(|field_type| type_string_to_value_type(&field_type))
}

fn field_type_string_from_expr(
    base: &syn::Expr,
    member: &syn::Member,
    ctx: &GraphBuildContext,
) -> Option<String> {
    let field_name = member_name(member);
    let owner = receiver_type_root(base, ctx)?;
    ctx.struct_fields
        .field_type(&owner, &field_name)
        .map(ToOwned::to_owned)
}

/// Type-string inference for `let pat = init;` initialisers, with the
/// extra power of binding match-arm patterns into a borrowed snapshot of
/// `ctx` so an arm body's expression can be typed against the
/// destructured names the body actually uses
/// (`fielddescrs.as_slice()` after `VirtualInfo { fielddescrs, .. }`).
/// Non-match initialisers fall through to the immutable
/// `expression_type_string`.
///
/// RPython's annotator handles the same shape via `SomeBuiltin.unionof`
/// over per-branch annotations; pyre takes the first arm whose body
/// types successfully and trusts the type-checker to keep the
/// remaining arms in sync.
fn infer_init_type_string(expr: &syn::Expr, ctx: &mut GraphBuildContext<'_>) -> Option<String> {
    if let syn::Expr::Match(m) = expr {
        let scrutinee_type = expression_type_string(&m.expr, ctx);
        for arm in &m.arms {
            let saved = LocalBindingSnapshot::capture(ctx);
            bind_pattern_locals(&arm.pat, scrutinee_type.as_deref(), ctx);
            let body_type = expression_type_string(&arm.body, ctx);
            saved.restore(ctx);
            if body_type.is_some() {
                return body_type;
            }
        }
        return None;
    }
    expression_type_string(expr, ctx)
}

fn expression_type_string(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_strings.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => expression_type_string(&reference.expr, ctx),
        syn::Expr::Paren(paren) => expression_type_string(&paren.expr, ctx),
        syn::Expr::Unary(unary) => match &unary.op {
            syn::UnOp::Deref(_) => expression_type_string(&unary.expr, ctx),
            _ => None,
        },
        syn::Expr::Cast(cast) => qualified_full_type_string(
            &cast.ty,
            &ctx.module_prefix,
            ctx.known_struct_names,
            ctx.known_trait_names,
        ),
        syn::Expr::Field(field) => field_type_string_from_expr(&field.base, &field.member, ctx),
        syn::Expr::Call(call) => {
            let syn::Expr::Path(path) = &*call.func else {
                return None;
            };
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            lookup_function_return_type(ctx, &segments).cloned()
        }
        syn::Expr::MethodCall(mc) => {
            let receiver_root = receiver_type_root(&mc.receiver, ctx);
            let trait_bound_root = trait_bound_root_for_receiver(&mc.receiver, ctx);
            if let Some(ret) = lookup_method_return_type(ctx, receiver_root.as_deref(), &mc.method)
                .or_else(|| lookup_method_return_type(ctx, trait_bound_root.as_deref(), &mc.method))
                .cloned()
            {
                return Some(ret);
            }
            // RPython annotator `SomeList.method_get / .method_first /
            // .method_last`-style result inference, narrowed to the
            // stdlib `Vec<T>` / `[T]` accessors that `let Some(x) =
            // lst.get(i) else { ... }` desugars from.  Pyre's
            // `local_array_types` carries the full container type
            // (`Vec<FieldDescrInfo>`); the `Option<&T>` shape is the
            // Rust-language adaptation of RPython's `lst[i]` access.
            let method = mc.method.to_string();
            if matches!(method.as_str(), "get" | "first" | "last") {
                if let Some(arr_ty) = array_type_id_from_expr(&mc.receiver, ctx)
                    && let Some(elem) = extract_element_type_from_str(&arr_ty)
                {
                    return Some(format!("Option<&{}>", elem));
                }
            }
            if method == "as_ref" {
                if let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
                    && let Some(ret) = method_as_ref_return_type(&receiver_ty)
                {
                    return Some(ret);
                }
            }
            // `Vec::as_slice` / `slice::as_ref` view the receiver as
            // `&[T]` while preserving element identity.  Same parity
            // rationale as the `get` arm — RPython's `lst.tolist()` /
            // `lst[:]` aliases keep the underlying `GcArray(T)` type
            // identity for downstream `getarrayitem` lookups.
            if matches!(method.as_str(), "as_slice" | "as_ref") {
                if let Some(arr_ty) = array_type_id_from_expr(&mc.receiver, ctx)
                    && let Some(elem) = extract_element_type_from_str(&arr_ty)
                {
                    return Some(format!("&[{}]", elem));
                }
            }
            None
        }
        // RPython `BUILD_TUPLE` produces a `SomeTuple(elems)`; the type
        // surfaces as a parenthesised list of element annotations.  Pyre
        // mirrors this with a flat string so `Pat::Tuple` /
        // `split_tuple_type_elements` can route each element back to its
        // originating concretetype.
        syn::Expr::Tuple(t) => {
            let mut element_types: Vec<String> = Vec::with_capacity(t.elems.len());
            for elem in &t.elems {
                let ty = expression_type_string(elem, ctx)?;
                element_types.push(ty);
            }
            Some(format!("({})", element_types.join(", ")))
        }
        // `VirtualKind::Instance { ... }` — Rust struct-init for an enum
        // variant.  RPython's annotator returns `SomeInstance(cls)` for
        // class-instantiation; here the parent enum name is the
        // closest analog (the variant's per-class subclass identity is
        // not carried as a separate type root).  Falls back to the
        // joined path for plain struct constructors.
        syn::Expr::Struct(es) => {
            let segments: Vec<String> = es
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect();
            if segments.len() >= 2 {
                Some(segments[..segments.len() - 1].join("::"))
            } else {
                Some(segments.join("::"))
            }
        }
        _ => None,
    }
}

fn lookup_function_return_type<'a>(
    ctx: &'a GraphBuildContext,
    segments: &[String],
) -> Option<&'a String> {
    let key = if segments.len() == 1 && !ctx.module_prefix.is_empty() {
        format!("{}::{}", ctx.module_prefix, segments[0])
    } else {
        segments.join("::")
    };
    ctx.fn_return_types.get(&key)
}

fn array_item_value_type_from_array_type_id(array_type_id: Option<&str>) -> Option<ValueType> {
    let elem_type = extract_element_type_from_str(array_type_id?)?;
    Some(type_string_to_value_type(&elem_type))
}

/// For `arr[idx]`, returns the ELEMENT TYPE of `arr` from context.
/// This is the Rust equivalent of RPython's `op.args[0].concretetype.TO`
/// which gives `GcArray(T)` — the `T` is what distinguishes array types.
/// RPython: `ARRAY.OF` — extract element type from full ARRAY type string.
///
/// Handles all Rust array/container notations:
/// - `Vec<Point>` → `"Point"` (angle brackets)
/// - `[i64]` → `"i64"` (slice)
/// - `[Point; 10]` → `"Point"` (fixed-size array)
/// - `&[Point]` / `&mut [Point]` — reference / mut-reference prefixes are
///   stripped before the slice form is matched, mirroring
///   `type_root_from_type_string`'s prefix walk so chained
///   `Vec::as_slice` results retain their element type.
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    // Square brackets: [T] or [T; N]
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        // [T; N] → T (strip "; N" suffix)
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    // Angle brackets: Vec<T>, Box<T>, etc.  This is checked after the
    // outer slice form so `[Rc<T>]` yields `Rc<T>`, not `T`.
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return first_top_level_generic_arg(&s[start + 1..end])
                .map(str::trim)
                .filter(|elem| !elem.is_empty())
                .map(ToOwned::to_owned);
        }
    }
    None
}

fn strip_pointer_like_prefixes(type_str: &str) -> &str {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    s
}

fn outer_generic_inner_type(type_str: &str, wrappers: &[&str]) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    let start = s.find('<')?;
    let inner = s[start + 1..].strip_suffix('>')?;
    let head = s[..start].trim().rsplit("::").next().unwrap_or("").trim();
    if !wrappers.contains(&head) {
        return None;
    }
    first_top_level_generic_arg(inner)
        .map(str::trim)
        .filter(|inner| !inner.is_empty())
        .map(ToOwned::to_owned)
}

fn method_as_ref_return_type(receiver_ty: &str) -> Option<String> {
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Rc", "Arc", "Box", "NonNull"]) {
        return Some(format!("&{}", strip_pointer_like_prefixes(&inner)));
    }
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Option"]) {
        return Some(format!("Option<&{}>", strip_pointer_like_prefixes(&inner)));
    }
    extract_element_type_from_str(receiver_ty).map(|elem| format!("&[{}]", elem))
}

fn array_type_id_from_expr(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_array_types.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(r) => array_type_id_from_expr(&r.expr, ctx),
        syn::Expr::Paren(p) => array_type_id_from_expr(&p.expr, ctx),
        // RPython: op.args[0].concretetype — for field access like `self.array`,
        // resolve the field's type from struct_fields to get element type.
        syn::Expr::Field(field) => {
            let owner_type = receiver_type_root(&field.base, ctx)?;
            let field_name = member_name(&field.member);
            // RPython: op.args[0].concretetype — returns full ARRAY type.
            let field_type_str = ctx.struct_fields.field_type(&owner_type, &field_name)?;
            Some(field_type_str.to_string())
        }
        // RPython: op.result.concretetype — for call expressions like `make_points()[i]`,
        // resolve the return type from the exact callee graph (fn_return_types in pass 1).
        syn::Expr::Call(call) => {
            if let syn::Expr::Path(path) = &*call.func {
                // RPython: exact graph identity — join path segments to match
                // the key format produced by collect_types_from_items.
                // RPython: exact graph identity — qualify bare single-segment
                // calls with module prefix to match registered keys.
                let segments: Vec<String> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                lookup_function_return_type(ctx, &segments).cloned()
            } else {
                None
            }
        }
        // RPython: op.result.concretetype — for method calls like `self.make_points()[i]`.
        // RPython resolves via the exact callee graph — no bare name fallback.
        syn::Expr::MethodCall(mc) => {
            let method_name = mc.method.to_string();
            if matches!(method_name.as_str(), "as_slice" | "as_ref")
                && let Some(ret) = expression_type_string(expr, ctx)
                && extract_element_type_from_str(&ret).is_some()
            {
                return Some(ret);
            }
            let receiver_ty = receiver_type_root(&mc.receiver, ctx)?;
            let key = format!("{}::{}", receiver_ty, method_name);
            ctx.fn_return_types.get(&key).cloned()
        }
        // RPython: op.result.concretetype — for nested index like `matrix[i][j]`,
        // resolve the outer array's element type.
        syn::Expr::Index(idx) => {
            let outer_type = array_type_id_from_expr(&idx.expr, ctx)?;
            let elem = extract_element_type_from_str(&outer_type)?;
            // If the element type is itself an array type, return it
            if elem.starts_with("Vec<") || elem.starts_with('[') {
                Some(elem)
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_function_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                let z = x + y;
                z
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(program.functions.len(), 1);
        let graph = &program.functions[0].graph;
        // Should have Input ops for params + ops for body
        assert!(graph.block(graph.startblock).operations.len() >= 2);
    }

    #[test]
    fn lowers_field_access_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_field(s: S) -> i64 {
                s.x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        // Should contain a FieldRead op
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "x"
            )),
            "expected FieldRead for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn type_string_helpers_preserve_outer_slice_and_as_ref_wrapper_identity() {
        assert_eq!(
            extract_element_type_from_str("[std::rc::Rc<majit_ir::RdVirtualInfo>]"),
            Some("std::rc::Rc<majit_ir::RdVirtualInfo>".to_string())
        );
        assert_eq!(
            extract_element_type_from_str("&[FieldDescrInfo]"),
            Some("FieldDescrInfo".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("&std::rc::Rc<majit_ir::RdVirtualInfo>"),
            Some("&majit_ir::RdVirtualInfo".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("Vec<FieldDescrInfo>"),
            Some("&[FieldDescrInfo]".to_string())
        );
        assert_eq!(
            method_as_ref_return_type("Option<FieldDescrInfo>"),
            Some("Option<&FieldDescrInfo>".to_string())
        );
    }

    #[test]
    fn binds_slice_get_result_from_enum_variant_tuple_destructure() {
        let ir = crate::parse::parse_source(
            r#"
            enum Type {
                Ref,
                Float,
                Int,
            }
            struct FieldDescrInfo {
                field_type: Type,
                field_size: usize,
            }
            enum RdVirtualInfo {
                VirtualInfo {
                    fielddescrs: Vec<FieldDescrInfo>,
                    descr_size: usize,
                },
                Empty,
            }
        "#,
        );
        let eval = crate::parse::parse_source(
            r#"
            fn read_field_size(
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                vidx: usize,
                i: usize,
            ) -> i64 {
                let Some(virtuals) = rd_virtuals else {
                    return 0;
                };
                let Some(entry) = virtuals.get(vidx) else {
                    return 0;
                };
                let (fielddescrs, _descr_size) = match entry.as_ref() {
                    majit_ir::RdVirtualInfo::VirtualInfo {
                        fielddescrs,
                        descr_size,
                    } => (fielddescrs.as_slice(), *descr_size),
                    _ => return 0,
                };
                let Some(descr) = fielddescrs.get(i) else {
                    return 0;
                };
                match descr.field_size {
                    1 => 1,
                    _ => 8,
                }
            }
        "#,
        );
        let program =
            build_semantic_program_from_parsed_files(&[ir, eval]).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "read_field_size")
            .expect("read_field_size graph")
            .graph;
        let field_size_read = graph
            .blocks
            .iter()
            .flat_map(|block| block.operations.iter())
            .find_map(|op| match &op.kind {
                OpKind::FieldRead { field, ty, .. } if field.name == "field_size" => {
                    Some((field.owner_root.clone(), ty.clone()))
                }
                _ => None,
            })
            .expect("field_size FieldRead");
        assert_eq!(
            field_size_read,
            (Some("FieldDescrInfo".to_string()), ValueType::Int)
        );
    }

    #[test]
    fn binds_option_some_inner_type_in_match_arm() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Meta { tracing_call_depth: Option<u32> }
            fn call_depth() -> u32 { 0 }
            fn example(meta: Meta) -> bool {
                let tracing_depth = meta.tracing_call_depth;
                if let Some(depth) = tracing_depth {
                    call_depth() == depth
                } else {
                    false
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|func| func.name == "example")
            .expect("example graph")
            .graph;
        assert!(
            graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Input { name, ty }
                            if name == "depth" && *ty == ValueType::Int
                    )
                })
            }),
            "expected Some(depth) to bind depth as Int; graph:\n{}",
            graph.dump()
        );
    }

    #[test]
    fn binds_result_err_inner_type_in_if_let_arm() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(result: Result<i64, f64>) -> f64 {
                if let Err(err) = result {
                    err
                } else {
                    0.0
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;

        assert!(
            graph.blocks.iter().any(|block| {
                block.operations.iter().any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Input { name, ty }
                            if name == "err" && *ty == ValueType::Float
                    )
                })
            }),
            "expected Err(err) to bind err as Float; graph:\n{}",
            graph.dump()
        );
    }

    fn count_field_reads(graph: &FunctionGraph, field_name: &str) -> usize {
        let mut n = 0;
        for bid in 0..graph.blocks.len() {
            for op in &graph.blocks[bid].operations {
                if let OpKind::FieldRead { field, .. } = &op.kind {
                    if field.name == field_name {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    /// Block tail expression must be lowered exactly once.
    ///
    /// RPython flow-space invariant: every source expression is walked
    /// once. Before the `lower_stmt_list_with_tail_value` refactor, a
    /// block's tail was lowered via `lower_stmt` (which dispatches to
    /// `lower_expr`) AND a second explicit `lower_expr` call, emitting
    /// the op twice.
    #[test]
    fn block_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S) -> i64 {
                let y = { s.x };
                y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "block tail `s.x` must produce exactly one FieldRead"
        );
    }

    /// `unsafe { .. }` lowers through the same single-walk path as a
    /// plain block — `unsafe` is a type-system marker, not a runtime
    /// wrapper, so the tail expression is walked once.
    #[test]
    fn unsafe_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S) -> i64 {
                let y = unsafe { s.x };
                y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "unsafe tail `s.x` must produce exactly one FieldRead"
        );
    }

    /// `if` then-branch tail expression is walked once. Counts
    /// FieldReads of `s.x` across every block in the graph so the
    /// assertion is independent of how the then/else blocks are laid
    /// out.
    #[test]
    fn if_then_tail_expression_lowered_once() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_once(s: S, c: bool) -> i64 {
                if c { s.x } else { 0 }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert_eq!(
            count_field_reads(graph, "x"),
            1,
            "if-then tail `s.x` must produce exactly one FieldRead"
        );
    }

    #[test]
    fn lowers_field_access_with_typed_fieldread_and_fieldwrite() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64, y: f64 }
            fn mutate(s: S) -> i64 {
                s.x = 1;
                s.x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldWrite { field, ty, .. }
                    if field.name == "x" && *ty == ValueType::Int
            )),
            "expected typed FieldWrite for 'x', got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "x" && *ty == ValueType::Int
            )),
            "expected typed FieldRead for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_array_access_with_typed_arrayread_and_arraywrite() {
        let parsed = crate::parse::parse_source(
            r#"
            fn mutate(xs: Vec<i64>, i: usize) -> i64 {
                xs[i] = 1;
                xs[i]
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::ArrayWrite { item_ty, .. } if *item_ty == ValueType::Int
            )),
            "expected typed ArrayWrite, got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::ArrayRead { item_ty, .. } if *item_ty == ValueType::Int
            )),
            "expected typed ArrayRead, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_cast_deref_field_access_with_typed_float_fieldread() {
        let parsed = crate::parse::parse_source(
            r#"
            struct PyObject { ob_type: i64, w_class: i64 }
            struct W_FloatObject { ob_header: PyObject, floatval: f64 }
            type PyObjectRef = *mut PyObject;

            unsafe fn w_float_get_value(obj: PyObjectRef) -> f64 {
                (*(obj as *const W_FloatObject)).floatval
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, ty, .. }
                    if field.name == "floatval" && *ty == ValueType::Float
            )),
            "expected typed float FieldRead for 'floatval', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_method_call_with_args() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(v: Vec<i64>) {
                v.push(42);
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. } if target == &CallTarget::method("push", Some("Vec".into()))
            )),
            "expected Call to 'push', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_impl_self_method_call_with_concrete_self_type() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn helper(&self) {}
                fn run(&self) {
                    self.helper();
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let run = program
            .functions
            .iter()
            .find(|func| func.name == "run")
            .expect("run graph");
        let ops = &run.graph.block(run.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::method("helper", Some("Foo".into()))
            )),
            "expected helper call with concrete self type, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_path_call_to_canonical_symbol() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(x: i64) -> i64 {
                crate::math::w_int_add(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::function_path(["crate", "math", "w_int_add"])
            )),
            "expected canonical Call target path, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_unregistered_ok_call_to_synthetic_transparent_ctor() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(x: i64) -> i64 {
                Ok(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::synthetic_transparent_ctor("Ok")
            )),
            "expected synthetic transparent Ok ctor, got {:?}",
            ops
        );
    }

    #[test]
    fn registered_function_named_ok_does_not_lower_to_synthetic_ctor() {
        let parsed = crate::parse::parse_source(
            r#"
            fn Ok(x: i64) -> i64 { x }
            fn call_example(x: i64) -> i64 {
                Ok(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|func| func.name == "call_example")
            .expect("call_example present");
        let ops = &func.graph.block(func.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::FunctionPath { segments },
                    ..
                } if segments.last().map(String::as_str) == Some("Ok")
            )),
            "expected registered Ok function path, got {:?}",
            ops
        );
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call {
                    target: CallTarget::SyntheticTransparentCtor { .. },
                    ..
                }
            )),
            "registered Ok must not be treated as a synthetic ctor: {:?}",
            ops
        );
    }

    #[test]
    fn err_and_some_calls_lower_to_synthetic_transparent_ctors() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_err(x: i64) -> i64 {
                Err(x)
            }
            fn call_some(x: i64) -> i64 {
                Some(x)
            }
            fn call_qualified_ok(x: i64) -> i64 {
                Result::Ok(x)
            }
            fn call_std_some(x: i64) -> i64 {
                std::option::Option::Some(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        for func_name in [
            "call_err",
            "call_some",
            "call_qualified_ok",
            "call_std_some",
        ] {
            let func = program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .expect("function present");
            let ops = &func.graph.block(func.graph.startblock).operations;
            assert!(
                ops.iter().any(|op| matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::SyntheticTransparentCtor { .. },
                        ..
                    }
                )),
                "{func_name} must lower to a synthetic transparent ctor: {:?}",
                ops
            );
        }
    }

    #[test]
    fn builds_impl_methods() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn bar(&self) { }
                fn baz(&self, x: i64) -> i64 { x }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(program.functions.len(), 2);
        assert_eq!(program.functions[0].name, "bar");
        assert_eq!(program.functions[1].name, "baz");
    }

    #[test]
    fn if_creates_multiple_blocks() {
        let parsed = crate::parse::parse_source(
            r#"
            fn branch(x: bool) -> i64 {
                if x { 1 } else { 2 }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        // entry + then + else + merge = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "if/else should create >=4 blocks, got {}",
            graph.blocks.len()
        );
        // Upstream `flowspace/model.py:175-180` tags a bool branch by
        // `block.exitswitch == Variable` with two exits whose
        // `exitcase` values are True / False respectively.
        let entry = graph.block(graph.startblock);
        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "entry exitswitch should name the branch condition, got {:?}",
            entry.exitswitch,
        );
        assert_eq!(entry.exits.len(), 2, "bool branch has two exits");
    }

    #[test]
    fn match_literals_emit_switch_exitcases_for_all_arms() {
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(x: i64) -> i64 {
                match x {
                    0 => 11,
                    1 => 22,
                    _ => 33,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "primitive match should switch on the scrutinee, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![
                Some(ExitCase::Const(ConstValue::Int(0))),
                Some(ExitCase::Const(ConstValue::Int(1))),
                Some(ExitCase::Const(ConstValue::byte_str("default"))),
            ],
        );
        let llexitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.llexitcase.clone())
            .collect();
        assert_eq!(
            llexitcases,
            vec![Some(ConstValue::Int(0)), Some(ConstValue::Int(1)), None],
        );
    }

    #[test]
    fn match_negative_int_literals_are_classified_as_switch() {
        // RPython `flatten.py:269` reads `link.llexitcase` as Signed,
        // so `match x { -1 => ... }` lands in the switch path the same
        // way `match x { 1 => ... }` does.  syn 2.x represents `-1` as
        // `Pat::Lit(ExprLit { lit: Lit::Int })` whose token text
        // includes the `-`; `LitInt::base10_parse::<i64>` returns
        // `Ok(-1)` for that token.  This guard catches a future
        // regression where `classify_switch_pattern`'s `Pat::Lit` arm
        // narrows to non-negative literals only.
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(x: i64) -> i64 {
                match x {
                    -1 => 11,
                    -2 | -3 => 22,
                    5 => 33,
                    _ => 44,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "negative int literals must remain on the switch path, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![
                Some(ExitCase::Const(ConstValue::Int(-1))),
                Some(ExitCase::Const(ConstValue::Int(-2))),
                Some(ExitCase::Const(ConstValue::Int(-3))),
                Some(ExitCase::Const(ConstValue::Int(5))),
                Some(ExitCase::Const(ConstValue::byte_str("default"))),
            ],
        );
    }

    #[test]
    fn bool_match_wildcard_emits_true_false_exitcases() {
        let parsed = crate::parse::parse_source(
            r#"
            fn pick(flag: bool) -> i64 {
                match flag {
                    true => 11,
                    _ => 22,
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let entry = graph.block(graph.startblock);

        assert!(
            matches!(entry.exitswitch, Some(crate::model::ExitSwitch::Value(_))),
            "bool match should branch on the scrutinee, got {:?}",
            entry.exitswitch,
        );
        let exitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.exitcase.clone())
            .collect();
        assert_eq!(
            exitcases,
            vec![Some(ExitCase::Bool(true)), Some(ExitCase::Bool(false))],
        );
        let llexitcases: Vec<_> = entry
            .exits
            .iter()
            .map(|link| link.llexitcase.clone())
            .collect();
        assert_eq!(
            llexitcases,
            vec![Some(ConstValue::Bool(true)), Some(ConstValue::Bool(false))],
        );
    }

    #[test]
    fn while_creates_header_body_exit() {
        let parsed = crate::parse::parse_source(
            r#"
            fn loop_fn(mut x: i64) -> i64 {
                while x > 0 { x = x - 1; }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        // entry + header + body + exit = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "while should create >=4 blocks, got {}",
            graph.blocks.len()
        );
    }

    #[test]
    fn lowers_binary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                x + y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("binop");
        assert_eq!(op, "add");
    }

    #[test]
    fn lowers_array_field_to_interiorfield_read() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Point { x: i64, y: i64 }
            fn read_point(points: Vec<Point>, i: usize) -> i64 {
                points[i].x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::InteriorFieldRead { field, item_ty, .. }
                    if field.name == "x" && *item_ty == ValueType::Int
            )),
            "expected InteriorFieldRead for 'x' with item_ty=Int, got {:?}",
            ops
        );
        // Should NOT generate a separate ArrayRead + FieldRead pair.
        assert!(
            !ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "x"
            )),
            "should not have FieldRead for 'x' when InteriorFieldRead is present"
        );
    }

    #[test]
    fn lowers_array_field_to_interiorfield_write() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Point { x: i64, y: i64 }
            fn write_point(points: Vec<Point>, i: usize) {
                points[i].x = 42;
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::InteriorFieldWrite { field, .. } if field.name == "x"
            )),
            "expected InteriorFieldWrite for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_unary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64) -> i64 {
                -x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("unary op");
        assert_eq!(op, "neg");
    }

    /// RPython `jtransform.py:410-412`: `dyn Trait` receivers from
    /// parameter bindings and `Box<dyn Trait>` locals must lower to
    /// `CallTarget::Indirect`, not to `CallTarget::Method`.
    /// Covers Issue 3 (detection too narrow).
    #[test]
    fn dyn_trait_receiver_detection_local_binding() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_via_param(h: &mut dyn Handler) {
                h.run();
            }
            fn call_via_box_local() {
                let h: Box<dyn Handler> = make_handler();
                h.run();
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");

        for func in &program.functions {
            let graph = &func.graph;
            let saw_indirect = graph
                .block(graph.startblock)
                .operations
                .iter()
                .any(|op| match &op.kind {
                    OpKind::Call {
                        target:
                            CallTarget::Indirect {
                                trait_root,
                                method_name,
                            },
                        ..
                    } => trait_root == "Handler" && method_name == "run",
                    _ => false,
                });
            assert!(
                saw_indirect,
                "expected CallTarget::Indirect in {}, got {:?}",
                func.graph.name,
                graph.block(graph.startblock).operations
            );
        }
    }

    #[test]
    fn dyn_trait_receiver_uses_module_qualified_trait_family_key() {
        let parsed = crate::parse::parse_source(
            r#"
            mod a {
                pub trait Handler { fn run(&mut self); }
                pub fn call_a(h: &mut dyn Handler) { h.run(); }
            }
            mod b {
                pub trait Handler { fn run(&mut self); }
                pub fn call_b(h: &mut dyn Handler) { h.run(); }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let mut seen = std::collections::HashMap::<String, String>::new();
        for func in &program.functions {
            let Some(trait_root) = func
                .graph
                .block(func.graph.startblock)
                .operations
                .iter()
                .find_map(|op| match &op.kind {
                    OpKind::Call {
                        target: CallTarget::Indirect { trait_root, .. },
                        ..
                    } => Some(trait_root.clone()),
                    _ => None,
                })
            else {
                continue;
            };
            seen.insert(func.name.clone(), trait_root);
        }
        assert_eq!(seen.get("a::call_a"), Some(&"a::Handler".to_string()));
        assert_eq!(seen.get("b::call_b"), Some(&"b::Handler".to_string()));
    }

    /// RPython `rpython/rtyper/rclass.py:644-678 _parse_field_list` — the
    /// `?`, `[*]`, and `?[*]` suffixes must resolve to `IR_QUASIIMMUTABLE`,
    /// `IR_IMMUTABLE_ARRAY`, and `IR_QUASIIMMUTABLE_ARRAY` respectively.
    /// Covers Issue 5 (partial port).
    #[test]
    fn parse_immutable_fields_accepts_string_literal_suffixes() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields("plain", "quasi?", "arr[*]", "qarr?[*]")]
            struct S { plain: i64, quasi: i64, arr: i64, qarr: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        let by_name: std::collections::HashMap<&str, ImmutableRank> =
            entries.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        assert_eq!(by_name.get("plain"), Some(&ImmutableRank::Immutable));
        assert_eq!(by_name.get("quasi"), Some(&ImmutableRank::QuasiImmutable));
        assert_eq!(by_name.get("arr"), Some(&ImmutableRank::ImmutableArray));
        assert_eq!(
            by_name.get("qarr"),
            Some(&ImmutableRank::QuasiImmutableArray)
        );
    }

    /// Bare ident entries in `#[jit_immutable_fields(foo, bar)]` continue
    /// to resolve to `IR_IMMUTABLE` — backward compatibility with pre-rank
    /// usage sites.
    #[test]
    fn parse_immutable_fields_preserves_bare_ident_backward_compat() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields(foo, bar)]
            struct S { foo: i64, bar: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        for (name, rank) in entries {
            assert!(
                matches!(rank, ImmutableRank::Immutable),
                "bare ident `{}` expected Immutable rank, got {:?}",
                name,
                rank,
            );
        }
        let names: std::collections::HashSet<&str> =
            entries.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains("foo"));
        assert!(names.contains("bar"));
    }

    /// Multiple `#[jit_immutable_fields(...)]` attributes on the same
    /// struct should accumulate — `rpython/rtyper/rclass.py:638-641` rbase
    /// walk iterates ancestor `_immutable_fields_` unions similarly.
    #[test]
    fn parse_immutable_fields_merge_across_multiple_attributes() {
        let parsed = crate::parse::parse_source(
            r#"
            #[jit_immutable_fields("a?")]
            #[jit_immutable_fields("b[*]")]
            struct S { a: i64, b: i64 }
            fn noop() {}
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let entries = program
            .immutable_fields
            .get("S")
            .expect("S should have immutable_fields entries");
        let by_name: std::collections::HashMap<&str, ImmutableRank> =
            entries.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        assert_eq!(by_name.get("a"), Some(&ImmutableRank::QuasiImmutable));
        assert_eq!(by_name.get("b"), Some(&ImmutableRank::ImmutableArray));
    }

    /// Rust `impl Trait` is a static opaque type — the compiler
    /// monomorphizes each call site to a single concrete impl.  RPython
    /// `indirect_call` is reserved for truly polymorphic callees
    /// (`rpython/jit/codewriter/call.py:103 graphs_from`).  An `impl
    /// Trait` parameter must therefore lower to `CallTarget::Method`,
    /// not `CallTarget::Indirect`.
    #[test]
    fn impl_trait_param_does_not_lower_to_indirect_call() {
        let parsed = crate::parse::parse_source(
            r#"
            pub trait Handler { fn run(&mut self); }
            pub fn call_via_impl(mut h: impl Handler) {
                h.run();
            }
            "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "call_via_impl")
            .expect("call_via_impl present");
        let saw_indirect = func
            .graph
            .block(func.graph.startblock)
            .operations
            .iter()
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call {
                        target: CallTarget::Indirect { .. },
                        ..
                    }
                )
            });
        assert!(
            !saw_indirect,
            "impl Trait must not lower to CallTarget::Indirect, got {:?}",
            func.graph.block(func.graph.startblock).operations
        );
    }

    /// Issue #5 — receiver detection beyond simple bindings.  Field
    /// access (`self.handler.run()`), index (`handlers[i].run()`), and
    /// `Box<dyn T>`-returning calls (`make_boxed().run()`) must all
    /// reach `CallTarget::Indirect`.
    #[test]
    fn dyn_receiver_via_field_index_and_box_return() {
        let parsed = crate::parse::parse_source(
            r#"
            pub trait Handler { fn run(&mut self); }
            struct Owner { handler: Box<dyn Handler> }
            impl Owner {
                fn dispatch(&mut self) { self.handler.run(); }
            }
            fn list_dispatch(handlers: Vec<Box<dyn Handler>>, idx: usize) {
                handlers[idx].run();
            }
            fn make_boxed() -> Box<dyn Handler> { panic!() }
            fn ret_dispatch() { make_boxed().run(); }
            "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        for fname in &["dispatch", "list_dispatch", "ret_dispatch"] {
            let func = program
                .functions
                .iter()
                .find(|f| f.graph.name == *fname)
                .unwrap_or_else(|| panic!("function {fname} present"));
            let saw_indirect = func
                .graph
                .block(func.graph.startblock)
                .operations
                .iter()
                .any(|op| {
                    matches!(
                        &op.kind,
                        OpKind::Call {
                            target: CallTarget::Indirect { method_name, .. },
                            ..
                        } if method_name == "run"
                    )
                });
            assert!(
                saw_indirect,
                "expected CallTarget::Indirect{{method=run}} in {fname}, got {:?}",
                func.graph.block(func.graph.startblock).operations
            );
        }
    }

    #[test]
    fn value_return_routes_through_canonical_returnblock() {
        let parsed = crate::parse::parse_source(
            r#"
            fn returns_one() -> i64 { return 1; }
            "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "returns_one")
            .expect("returns_one present");
        let entry = func.graph.block(func.graph.startblock);
        // rpython/flowspace/model.py:171-180 Block is characterized by
        // exits + exitswitch; a non-void return is Link(
        // [return_value], graph.returnblock) with exitswitch=None.
        assert!(entry.exitswitch.is_none());
        assert_eq!(entry.exits.len(), 1);
        assert_eq!(entry.exits[0].prevblock, Some(func.graph.startblock));
        assert_eq!(entry.exits[0].target, func.graph.returnblock);
        assert_eq!(
            entry.exits[0].args,
            vec![crate::model::LinkArg::from(
                entry.operations[0].result.expect("const result"),
            )],
        );
    }

    #[test]
    fn void_return_routes_through_canonical_returnblock() {
        let parsed = crate::parse::parse_source(
            r#"
            fn returns_unit() { return; }
            "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let func = program
            .functions
            .iter()
            .find(|f| f.graph.name == "returns_unit")
            .expect("returns_unit present");
        let entry = func.graph.block(func.graph.startblock);
        // RPython `flowcontext.py` emits a fresh Variable on the
        // prevblock side for `return None`; the returnblock's own
        // inputarg stays distinct.
        let returnblock_arg = func.graph.block(func.graph.returnblock).inputargs[0];
        // Upstream `flowspace/model.py:171-180` keeps the void return shape
        // in Block.exits: a single Link([fresh_void], graph.returnblock)
        // with exitswitch=None.
        assert!(entry.exitswitch.is_none());
        assert_eq!(entry.exits.len(), 1);
        assert_eq!(entry.exits[0].prevblock, Some(func.graph.startblock));
        assert_eq!(entry.exits[0].target, func.graph.returnblock);
        assert_eq!(entry.exits[0].args.len(), 1);
        assert_ne!(
            entry.exits[0].args[0].as_value(),
            Some(returnblock_arg),
            "void return must allocate a fresh prevblock-side ValueId (`flowspace/model.py:114`), \
             not reuse the returnblock's own inputarg"
        );
    }
}
