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
    BlockId, CallTarget, ExitCase, ExitSwitch, FrameState, FrameStateEntry, FunctionGraph,
    ImmutableRank, Link, LinkArg, OpKind, UnknownKind, UnsupportedExprKind, UnsupportedLiteralKind,
    ValueId, ValueType, exception_exitcase,
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
    /// resolve by structural type identity. Pyre's analyser carries
    /// type identity as strings, so exact registered keys are the
    /// normal path. A receiver can still surface with an unavoidable
    /// Rust crate prefix that was not present at registration time
    /// (for example `pyre_object::rangeobject::W_RangeIterator` vs
    /// `rangeobject::W_RangeIterator`). In that case, accept a suffix
    /// recovery only when it is unique. Ambiguous leaf-name matches
    /// return `None` instead of picking the first registered struct.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.lookup_fields(owner)?
            .iter()
            .find(|(name, _)| name == field_name)
            .map(|(_, ty)| ty.as_str())
    }

    fn lookup_fields(&self, owner: &str) -> Option<&[(String, String)]> {
        if let Some(fields) = self.fields.get(owner) {
            return Some(fields.as_slice());
        }
        let key = self.unique_suffix_owner_key(owner)?;
        self.fields.get(key).map(Vec::as_slice)
    }

    fn unique_suffix_owner_key<'a>(&'a self, owner: &str) -> Option<&'a str> {
        let mut found: Option<&str> = None;
        for key in self.fields.keys() {
            let matches =
                is_path_suffix(owner, key.as_str()) || is_path_suffix(key.as_str(), owner);
            if !matches {
                continue;
            }
            if found.is_some() {
                return None;
            }
            found = Some(key.as_str());
        }
        found
    }
}

fn is_path_suffix(longer: &str, shorter: &str) -> bool {
    if longer.len() <= shorter.len() || !longer.ends_with(shorter) {
        return false;
    }
    let prefix_len = longer.len() - shorter.len();
    longer[..prefix_len].ends_with("::")
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

/// Pre-walk metadata produced by `collect_program_metadata_pub` —
/// the four registries that `build_function_graph` /
/// `build_function_graph_with_self_ty_pub` need before per-function
/// graph build can resolve typed call shapes.
pub struct ProgramMetadata {
    pub known_struct_names: std::collections::HashSet<String>,
    pub known_trait_names: std::collections::HashSet<String>,
    pub struct_fields: StructFieldRegistry,
    pub fn_return_types: HashMap<String, String>,
}

/// RPython `annrpython.py:103-150 build_types` whole-program walk —
/// runs `collect_struct_names` + `collect_trait_names` +
/// `collect_fields_and_returns` over the items of every parsed
/// file in `parsed_files`.  Public counterpart of the per-pipeline
/// collectors at
/// `build_semantic_program_from_parsed_files_with_options:744-764`,
/// for test-only entry points (`parse::collect_function_graphs`)
/// that need the same registries before invoking
/// `build_function_graph_with_self_ty_pub`.  Accepts a slice so a
/// callsite in one file can resolve a free function defined in
/// another file — single-file metadata leaves cross-file calls
/// (`!crate::is_str(...)` from dictobject.rs against `is_str`
/// defined in strobject.rs) unclassified.
pub fn collect_program_metadata_pub(parsed_files: &[ParsedInterpreter]) -> ProgramMetadata {
    let mut known_struct_names = std::collections::HashSet::new();
    let mut known_trait_names = std::collections::HashSet::new();
    let mut struct_fields = StructFieldRegistry::default();
    let mut fn_return_types: HashMap<String, String> = HashMap::new();
    let mut immutable_fields: HashMap<String, Vec<(String, ImmutableRank)>> = HashMap::new();
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
    ProgramMetadata {
        known_struct_names,
        known_trait_names,
        struct_fields,
        fn_return_types,
    }
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
                    struct_fields.fields.insert(qualified, fields);
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
                // Nested `fn`s declared as `Stmt::Item(Item::Fn(_))`
                // inside this fn's body.  Rust's lexical scoping makes
                // them callable only from within the parent body, but
                // the classifier (`expr_unary_not_operand_kind`) sees
                // them as bare-name `Expr::Call` paths and needs the
                // signature to disambiguate `!nested_pred(arg)` between
                // UNARY_NOT and UNARY_INVERT.  RPython parity:
                // `bookkeeper.getdesc(value)` resolves any callable in
                // scope by host-identity (`annrpython.py` callee
                // resolution); pyre's static walker substitutes by
                // registering the nested signature under the bare ident.
                collect_nested_fn_returns(
                    &func.block.stmts,
                    prefix,
                    known_struct_names,
                    known_trait_names,
                    fn_return_types,
                );
            }
            // RPython has no `Item::Const` analogue — Python module-level
            // constants reach `flowcontext.py` as `LOAD_GLOBAL(name)`
            // followed by the bookkeeper's PBC table lookup
            // (`bookkeeper.py:329-340 immutablevalue`).  Pyre's walker
            // doesn't model PBCs, so consts surface as plain
            // `Expr::Path` identifier references; the classifier needs
            // a typed-name entry to resolve `!FOO`-shape uses.  Reuse
            // `fn_return_types` keyed by ident — Rust convention
            // (SCREAMING_SNAKE_CASE consts vs snake_case fns) keeps the
            // namespaces separate in pyre's source.
            Item::Const(c) => {
                if let Some(ty) =
                    qualified_full_type_string(&c.ty, prefix, known_struct_names, known_trait_names)
                {
                    let key = if prefix.is_empty() {
                        c.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, c.ident)
                    };
                    fn_return_types.insert(key, ty);
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
                    // Impl-block associated consts — `impl Foo { const
                    // CONST_BIT: u32 = 1 << 31; }`. RPython peer:
                    // `bookkeeper.getdesc(value)` resolves class-level
                    // descriptors (`bookkeeper.py:329-340 immutablevalue`)
                    // by host-identity; pyre's static walker registers
                    // them under `Type::CONST_NAME` so `Type::CONST` /
                    // `Self::CONST` references can resolve via the
                    // last-two-segments fallback in
                    // `lookup_function_return_type` /
                    // `expr_unary_not_operand_kind`.
                    if let syn::ImplItem::Const(item_const) = sub {
                        if let Some(ty) = qualified_full_type_string(
                            &item_const.ty,
                            prefix,
                            known_struct_names,
                            known_trait_names,
                        ) && let Some(ref ty_root) = self_ty_root
                        {
                            let qualified_ty = qualify_type_name(ty_root, prefix);
                            fn_return_types.insert(
                                format!("{}::{}", qualified_ty, item_const.ident),
                                ty.clone(),
                            );
                            // Bare-key alias for `Self::CONST_BIT`-shape
                            // references whose qualifier strips to the
                            // last segment. Mirrors `Item::Const`'s
                            // file-level registration.
                            fn_return_types.insert(item_const.ident.to_string(), ty);
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
                // `Foo::Empty` vs `Bar::Empty`). Register only the
                // fully-qualified key. `field_type` accepts a shorter
                // registered key for callers with extra crate prefixes
                // only when that suffix match is unique. Tuple/unit
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

/// `promote_args` selector for `#[elidable_promote(...)]` synthesis.
///
/// Mirrors `rlib/jit.py:189-191` — RPython splits a literal string at
/// `,` when the value is not the special `"all"` marker:
///
/// ```python
/// if promote_args != 'all':
///     args = [args[int(i)] for i in promote_args.split(",")]
/// ```
#[derive(Debug, Clone)]
enum PromoteArgsSelector {
    /// `jit.py:180` default `promote_args='all'` — every non-self
    /// positional arg flows through `hint(..., promote=True)`.
    All,
    /// `jit.py:189-191` index list (`"0,2"` → `[0, 2]`).  Indices are
    /// 0-based and point into the **positional arg list including
    /// `self`** when the decorated function is a method (RPython's
    /// `_get_args(func)` reads the raw co_varnames; pyre mirrors the
    /// same convention by treating `self` as index 0 when present).
    Indices(Vec<usize>),
}

/// `rlib/jit.py:180-191` — parse the literal attribute value attached
/// to `#[elidable_promote(promote_args = "...")]`.  Bare
/// `#[elidable_promote]` defaults to `"all"` per the upstream default
/// argument at jit.py:180.
fn parse_elidable_promote_args(attr: &syn::Attribute) -> syn::Result<PromoteArgsSelector> {
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(PromoteArgsSelector::All);
    }
    let mut selector = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("promote_args") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            selector = Some(if lit.value() == "all" {
                PromoteArgsSelector::All
            } else {
                let mut indices = Vec::new();
                for piece in lit.value().split(',') {
                    indices.push(piece.trim().parse::<usize>().map_err(|err| {
                        syn::Error::new(lit.span(), format!("promote_args: {err}"))
                    })?);
                }
                PromoteArgsSelector::Indices(indices)
            });
            Ok(())
        } else {
            Err(meta.error("unsupported elidable_promote argument"))
        }
    })?;
    Ok(selector.unwrap_or(PromoteArgsSelector::All))
}

/// Expand a `syn::ItemFn` into `[orig, wrapper]` when it carries
/// `#[elidable_promote]` / `#[purefunction_promote]`, else return the
/// single function unchanged.  Mirrors `rlib/jit.py:184-201`'s "module
/// import installs two callables" semantics at every entry point that
/// hands a `syn::ItemFn` to `build_function_graph*` — free functions
/// (`build_graphs_from_items`), inherent / trait-impl methods
/// (`parse::extract_inherent_impl_methods`,
/// `parse::extract_trait_impls`), and trait default methods
/// (`parse::extract_trait_impls`).  Centralising the expansion here
/// keeps the decorator behaviour uniform across all four lowering
/// surfaces.
///
/// `impl_type` carries the qualified type root (`"S"`, `"a::S"`) when
/// the source item lives inside an `impl` block, so the synthesizer can
/// emit a type-qualified tail call (`a::S::_orig_<name>_unlikely_name(
/// self, args)`) that matches the impl-method registration path built
/// by `parse::CallPath::for_impl_method` at `lib.rs:531-537`.  Free
/// functions and trait-default methods (which have no concrete `Self`
/// type at synthesis time) pass `None` and fall back to the bare-path
/// tail call.
pub fn synthesize_or_passthrough(fake_fn: ItemFn, impl_type: Option<&str>) -> Vec<ItemFn> {
    match extract_elidable_promote_selector(&fake_fn.attrs) {
        Some(selector) => {
            let (orig, wrapper) = synthesize_elidable_promote_pair(&fake_fn, &selector, impl_type);
            vec![orig, wrapper]
        }
        None => vec![fake_fn],
    }
}

/// Locate `#[elidable_promote]` (or its deprecated `#[purefunction_promote]`
/// alias from `rlib/jit.py:203-205`) and return its parsed
/// `promote_args` selector, or `None` if neither attribute is present.
///
/// `rlib/jit.py:189-191` `args[int(i)]` propagates `ValueError` /
/// `IndexError` to the caller — the decorator does not silently drop
/// malformed input.  Pyre mirrors that fail-loud behaviour here: a
/// malformed `promote_args = "..."` literal panics with the
/// underlying `syn::Error` rather than falling through.
fn extract_elidable_promote_selector(attrs: &[syn::Attribute]) -> Option<PromoteArgsSelector> {
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            if name == "elidable_promote" || name == "purefunction_promote" {
                return Some(parse_elidable_promote_args(attr).unwrap_or_else(|err| {
                    panic!("#[{name}(...)]: {err}");
                }));
            }
        }
    }
    None
}

/// Synthesize the wrapper / original function pair from a single
/// `#[elidable_promote] fn foo(...)` source item.
///
/// Line-by-line port of `rlib/jit.py:184-201`:
///
/// ```python
/// def decorator(func):
///     elidable(func)                                # func._elidable_function_ = True
///     args = _get_args(func)
///     code = ["def f(%s):\n" % (argstring,)]
///     if promote_args != 'all':
///         args = [args[int(i)] for i in promote_args.split(",")]
///     for arg in args:
///         code.append("    %s = hint(%s, promote=True, promote_string=True)\n" % …)
///     code.append("    return _orig_func_unlikely_name(%s)\n" % …)
///     d = {"_orig_func_unlikely_name": func, "hint": hint}
///     exec py.code.Source("\n".join(code)).compile() in d
///     result = d["f"]
///     result.__name__ = func.__name__ + "_promote"
///     return result
/// ```
///
/// Pyre's syn-tree adaptation:
///
///   * RPython's `_orig_func_unlikely_name` closure capture becomes a
///     module-scope sibling `fn _orig_<name>_unlikely_name` carrying
///     the original body and the `#[elidable]` attribute (so
///     `collect_jit_hints` registers it via `mark_elidable`).
///   * RPython's `exec compile` synthesis of `def f(...)` becomes a
///     `syn::Block` whose stmts are `let <arg> = hint_promote(<arg>);`
///     for each selected arg, followed by the tail call.
///   * The user-facing name keeps its identity (matching pyre's
///     proc-macro naming convention `_orig_<name>_unlikely_name` at
///     `majit-macros::elidable_promote`); RPython's `__name__ + "_promote"`
///     rename is the symmetric choice and equivalently produces two
///     distinct callable identifiers.
fn synthesize_elidable_promote_pair(
    func: &ItemFn,
    selector: &PromoteArgsSelector,
    impl_type: Option<&str>,
) -> (ItemFn, ItemFn) {
    use quote::format_ident;
    // jit.py:186 args = _get_args(func) — positional names, self included.
    let arg_names: Vec<syn::Ident> = func
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(pt) => match &*pt.pat {
                syn::Pat::Ident(pi) => pi.ident.clone(),
                _ => panic!(
                    "#[elidable_promote] on `fn {}`: unsupported binder \
                     pattern in arg position — RPython `_get_args(func)` \
                     (jit.py:172-178) reads positional names off \
                     `co_varnames`, which never include destructured \
                     binders.  Rewrite the parameter as a plain `name: \
                     Ty` instead.",
                    func.sig.ident
                ),
            },
            // `&self` / `&mut self` map to a positional name "self" so
            // index 0 in `Indices` continues to address the receiver as
            // RPython does (`_get_args(func)` reads co_varnames raw).
            syn::FnArg::Receiver(_) => format_ident!("self"),
        })
        .collect();

    let orig_ident = format_ident!("_orig_{}_unlikely_name", func.sig.ident);

    // jit.py:184-185 — original keeps the body, gains `_elidable_function_`.
    let orig_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let elidable_attr: syn::Attribute = syn::parse_quote!(#[elidable]);
    let mut orig_fn = func.clone();
    orig_fn.attrs = orig_attrs;
    orig_fn.attrs.push(elidable_attr);
    orig_fn.sig.ident = orig_ident.clone();

    // jit.py:189-191 — promote_args=='all' → all indices; else parsed list.
    // RPython `args[int(i)]` raises `IndexError` on out-of-range
    // indices; pyre fails loudly here for the same reason.
    let promote_indices: Vec<usize> = match selector {
        PromoteArgsSelector::All => (0..arg_names.len()).collect(),
        PromoteArgsSelector::Indices(ix) => {
            for &i in ix {
                if i >= arg_names.len() {
                    panic!(
                        "#[elidable_promote(promote_args = ...)] on `fn {}`: \
                         index {} is out of range for a {}-arg function \
                         (jit.py:191 `args[int(i)]` would IndexError)",
                        func.sig.ident,
                        i,
                        arg_names.len()
                    );
                }
            }
            ix.clone()
        }
    };
    // jit.py:191-194 — `for arg in args: hint(arg, ...)`; `args` includes
    // `self` since `_get_args(func)` reads `co_varnames` raw.  Rust forbids
    // re-binding the `self` keyword, so the receiver is routed through a
    // fresh `__self_promoted` local; non-receiver args keep RPython's
    // shadow pattern (`let arg = hint_promote(arg);`).
    let promote_self = promote_indices.iter().any(|&i| arg_names[i] == "self");
    let promote_stmts: Vec<syn::Stmt> = promote_indices
        .iter()
        .map(|&i| {
            let id = &arg_names[i];
            if id == "self" {
                syn::parse_quote!(let __self_promoted = hint_promote(self);)
            } else {
                syn::parse_quote!(let #id = hint_promote(#id);)
            }
        })
        .collect();

    // jit.py:195 — return _orig_func_unlikely_name(args).  When `self` was
    // promoted above, the tail call uses the promoted local; otherwise it
    // threads the unmodified receiver through.
    let call_args = arg_names.iter().map(|id| -> syn::Expr {
        if id == "self" {
            if promote_self {
                syn::parse_quote!(__self_promoted)
            } else {
                syn::parse_quote!(self)
            }
        } else {
            syn::parse_quote!(#id)
        }
    });
    // jit.py:195-197 — `_orig_func_unlikely_name` is bound in the
    // wrapper's `exec` namespace (`d = {"_orig_func_unlikely_name":
    // func, ...}`), so the wrapper invokes it as a closure-local name.
    // Pyre's lowering surface does not have an analogue closure scope;
    // for free fns the orig is registered as a bare-path function (so a
    // bare-path call resolves), but for impl methods the orig is
    // registered under `<ImplType>::_orig_<name>_unlikely_name` via
    // `parse::CallPath::for_impl_method` at `lib.rs:531-537`.  Emit a
    // type-qualified tail call in that case so the wrapper's IR `Call`
    // target matches the registration path and the elidable flag bound
    // to the orig graph is visible at the call-site.
    let tail_call: syn::Expr = match impl_type {
        Some(ty_str) => {
            let ty_path: syn::Path = syn::parse_str(ty_str).unwrap_or_else(|err| {
                panic!(
                    "synthesize_elidable_promote_pair: failed to parse impl type `{ty_str}`: {err}"
                )
            });
            syn::parse_quote!(#ty_path::#orig_ident(#(#call_args),*))
        }
        None => syn::parse_quote!(#orig_ident(#(#call_args),*)),
    };
    let tail_stmt = syn::Stmt::Expr(tail_call, None);

    // jit.py:198-201 — wrapper is the user-facing decorated name with
    // the promote+forward body; `#[elidable_promote]` is stripped so
    // `collect_jit_hints` does not register the wrapper itself as
    // elidable (the binary flag belongs on the original alone, per
    // jit.py:185 `elidable(func)`).
    let wrapper_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let mut wrapper_block = syn::Block {
        brace_token: Default::default(),
        stmts: Vec::new(),
    };
    wrapper_block.stmts.extend(promote_stmts);
    wrapper_block.stmts.push(tail_stmt);

    let mut wrapper_fn = func.clone();
    wrapper_fn.attrs = wrapper_attrs;
    wrapper_fn.block = Box::new(wrapper_block);

    (orig_fn, wrapper_fn)
}

/// Walk the statements of a fn body and register the return types of
/// any nested `fn` items declared inside.  Used by `collect_fields_and_
/// returns`'s `Item::Fn` arm so the classifier sees nested-fn
/// signatures.  Recurses through `Stmt::Item(Item::Fn(_))` only
/// (nested-mod / nested-impl inside a fn body are vanishingly rare in
/// pyre source and would require their own qualified prefix).
fn collect_nested_fn_returns(
    stmts: &[syn::Stmt],
    prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
    fn_return_types: &mut HashMap<String, String>,
) {
    for stmt in stmts {
        if let syn::Stmt::Item(Item::Fn(nested)) = stmt {
            let ret_ty = match &nested.sig.output {
                syn::ReturnType::Type(_, ty) => {
                    qualified_full_type_string(ty, prefix, known_struct_names, known_trait_names)
                }
                syn::ReturnType::Default => Some("()".to_string()),
            };
            if let Some(ret_ty) = ret_ty {
                let key = if prefix.is_empty() {
                    nested.sig.ident.to_string()
                } else {
                    format!("{}::{}", prefix, nested.sig.ident)
                };
                fn_return_types.entry(key).or_insert(ret_ty);
            }
            collect_nested_fn_returns(
                &nested.block.stmts,
                prefix,
                known_struct_names,
                known_trait_names,
                fn_return_types,
            );
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
                // `rlib/jit.py:184-201 elidable_promote.decorator(func)`
                // installs two callable objects on module import
                // (closure-captured original + `exec`-built wrapper).
                // `synthesize_or_passthrough` mirrors that for both
                // free fns and impl methods so the lowering layer below
                // stays wrapper-blind, exactly like RPython's flow-graph
                // builder.
                for synth in synthesize_or_passthrough(func.clone(), None) {
                    let mut sf = build_function_graph(
                        &synth,
                        options,
                        None,
                        struct_fields,
                        fn_return_types,
                        prefix,
                        known_struct_names,
                        known_trait_names,
                    )?;
                    if !prefix.is_empty() {
                        sf.name = format!("{}::{}", prefix, sf.name);
                    }
                    functions.push(sf);
                }
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
                        // jit.py:184-201 — method-level `@elidable_promote`
                        // gets the same wrapper/orig synthesis as free
                        // fns; RPython decorators apply uniformly to any
                        // callable.  The qualified `self_ty_root` lets
                        // the wrapper's tail call hit the impl-method
                        // registration path built by
                        // `CallPath::for_impl_method`.
                        for synth in synthesize_or_passthrough(fake_fn, self_ty_root.as_deref()) {
                            let sf = build_function_graph(
                                &synth,
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
    collect_jit_hints(attrs, None)
}

/// `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
/// — companion to `collect_jit_hints` that exposes the function's
/// positional parameter names when `#[oopspec(...)]` is present.
///
/// Used by the walker to populate `CallControl::oopspec_argnames`
/// (`lib.rs:598-600` consumes the `"oopspec_argnames:..."` hint
/// alongside the `"oopspec:..."` hint).  Pyre's `parse_oopspec`
/// (`support.py:701-715` port) needs argnames to resolve identifier
/// slots in the spec's `(...)` pattern to `Index(n)` placeholders.
pub fn collect_jit_hints_with_sig(attrs: &[syn::Attribute], sig: &syn::Signature) -> Vec<String> {
    collect_jit_hints(attrs, Some(sig))
}

#[derive(Debug, Clone)]
struct GraphBuildContext<'a> {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    /// RPython `LOAD_FAST` parity: locals' definition sites carried as
    /// `(ValueId, defining BlockId)` so a body `Expr::Path` reference
    /// can reuse the existing definition's `ValueId` instead of
    /// emitting a fresh `OpKind::Input`. Same-block reuse only —
    /// cross-block reads keep the legacy fresh-`Input` behaviour
    /// because pyre does not yet thread the locals stack across
    /// `Link.args` / `inputarg` the way RPython
    /// `flowspace/flowcontext.py:835 LOAD_FAST` does. Closing the
    /// cross-block gap is a deferred Cat 3.2 follow-up; this field
    /// owns the same-block half of the parity.
    local_value_ids: HashMap<String, (ValueId, BlockId)>,
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
    /// Closure-bound locals: `let f = |args| body` registers `f` →
    /// closure body's return type so a downstream `f(...)` call gets a
    /// known return type. Pyre's walker has no closure visibility
    /// (`fn_return_types` only registers `Item::Fn` / `Item::Const` /
    /// `Item::Impl` methods); RPython's `bookkeeper.getdesc(value)`
    /// resolves any callable in scope by host-identity, so this side
    /// table substitutes by registering closure return types under the
    /// bare local ident.  Read by `lookup_function_return_type`'s
    /// bare-key fallback chain.
    local_closure_returns: HashMap<String, String>,
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
    /// First-bind positional order of local names — graph-wide
    /// append-only.  Each name is appended at the moment it is first
    /// bound anywhere in the function (`bind_local_id`); subsequent
    /// rebinds update `local_value_ids` in place but do not move the
    /// entry, and `LocalBindingSnapshot::restore` does NOT roll the
    /// order back so a name's slot index is invariant across the
    /// entire lowering even when a sibling arm's bindings get
    /// restored.  RPython parity: `co_varnames` slot order is
    /// assigned at compile time and never reshuffled
    /// (`flowspace/flowcontext.py:835 LOAD_FAST` reads slots by
    /// index); `FrameState::union`'s positional zip
    /// (`framestate.py:14 _union`) relies on this graph-wide
    /// invariant so two predecessors of the same merge point line
    /// up slot-by-slot.
    local_first_bind_order: Vec<String>,
    /// Graph-wide membership set complementing
    /// `local_first_bind_order`.  Used by `bind_local_id` to detect
    /// the *first ever* bind of a name in this function — distinct
    /// from "currently bound" (`local_value_ids.contains_key`)
    /// because a `LocalBindingSnapshot::restore` may unbind a name
    /// from `local_value_ids` while leaving its slot in
    /// `local_first_bind_order`.  Without this set a re-bind after
    /// restore would double-append the name and violate the slot
    /// invariant.
    local_first_bind_seen: std::collections::HashSet<String>,
}

#[derive(Debug, Clone)]
struct LoopFrame {
    /// Block that `continue` jumps to.  For `while` / `for` this is
    /// the header; for `loop` this is the body entry (which also acts
    /// as the loop head).  The current header-phi name list is
    /// recomputed on demand from `continue_target.inputargs` at
    /// every close site (back-edge / `Expr::Continue`) via
    /// `header_phi_name_list(graph, continue_target)` — Cat 2-2
    /// Phase B α.1 (`audit_cat2_2_loop_header_phi_workl_fixpoint_*`).
    /// RPython parity: `flowspace/flowcontext.py:399-465 mergeblock`
    /// queries the merge target's current state at close time, not
    /// at frame push, so any phi that lazy-install added to the
    /// header during body walk is automatically threaded.
    continue_target: BlockId,
    /// Block that `break` jumps to — the loop's exit block.
    break_target: BlockId,
}

/// Names statically reachable as locals from inside a loop body —
/// produced by `loop_body_locals` ahead of the loop walk so the eager
/// phi allocator (Slice 5b.3) can pre-build the loop header's
/// inputarg list without running a `framestate.union` work-list to
/// fixpoint.
///
/// `read_names` collects names referenced as a single-ident
/// `Expr::Path` (a use) or as the LHS of a compound-assign
/// `Expr::Binary` (which both reads and rebinds the slot).
/// `rebound_names` collects names appearing as the LHS of a simple
/// `Expr::Assign`, the LHS of any compound-assign `Expr::Binary`, or
/// the binding of a `Stmt::Local` whose pattern surfaces idents.  The
/// scan recurses through `Block` (including `if` arms, `match` arms,
/// nested loop bodies, and `unsafe` blocks) but explicitly skips
/// `Expr::Closure` bodies, `Stmt::Item` definitions, and `Stmt::Macro`
/// invocations.
///
/// RPython parity: there is no single-line counterpart in
/// `flowspace/flowcontext.py` because RPython infers reads/writes from
/// `LOAD_FAST` / `STORE_FAST` bytecodes implicit in the per-bytecode
/// dispatch (`flowcontext.py:780-820`).  This pre-scan is the
/// static-AST analogue, computing the must-merge name set ahead of the
/// loop walk so the eager allocator can replace RPython's iterative
/// `mergeblock` widening at `flowcontext.py:430` with a single pass.
/// The None-kill at `framestate.py:110-111` is realised by
/// intersecting the pre-scan names with the pre-loop framestate inside
/// the allocator — body-only locals never enter the header phi list.
#[derive(Debug, Default, Clone)]
struct LoopBodyLocals {
    read_names: std::collections::HashSet<String>,
    rebound_names: std::collections::HashSet<String>,
}

/// Statically scan `body` and partition every locally-referenced name
/// into `read_names` and `rebound_names` for the eager loop-header
/// phi allocator.  See `LoopBodyLocals` for the exact contract.
fn loop_body_locals(body: &syn::Block) -> LoopBodyLocals {
    let mut state = LoopBodyLocals::default();
    visit_block_for_loop_locals(body, &mut state);
    state
}

fn visit_block_for_loop_locals(block: &syn::Block, state: &mut LoopBodyLocals) {
    for stmt in &block.stmts {
        visit_stmt_for_loop_locals(stmt, state);
    }
}

fn visit_stmt_for_loop_locals(stmt: &syn::Stmt, state: &mut LoopBodyLocals) {
    match stmt {
        syn::Stmt::Local(local) => {
            // `let pat [: ty] [= init];` — pattern bindings are
            // rebinds.  Inner-scope-only bindings (a `let` inside a
            // nested block whose name shadows nothing in the
            // pre-loop snapshot) are filtered by the allocator: a
            // name not present in the pre-loop framestate is not a
            // header phi candidate.
            collect_pat_idents(&local.pat, &mut state.rebound_names);
            if let Some(init) = &local.init {
                visit_expr_for_loop_locals(&init.expr, state);
                if let Some((_, diverge)) = &init.diverge {
                    visit_expr_for_loop_locals(diverge, state);
                }
            }
        }
        syn::Stmt::Expr(expr, _semi) => visit_expr_for_loop_locals(expr, state),
        syn::Stmt::Item(_) => {
            // `fn`, `struct`, `use`, etc. — item definitions live in
            // their own scope and do not refer to enclosing locals.
            // Slice 5b.2 contract: skip.
        }
        syn::Stmt::Macro(_) => {
            // Macro invocations cannot be statically introspected
            // for local references.  Conservatively skip; if a macro
            // mutates a local that later participates in the
            // back-edge, the `Link.__init__` arity check at build
            // time fails loud (Slice 5 risk #1).
        }
    }
}

fn visit_expr_for_loop_locals(expr: &syn::Expr, state: &mut LoopBodyLocals) {
    match expr {
        syn::Expr::Path(p) => {
            if let Some(ident) = path_as_single_ident(&p.path) {
                state.read_names.insert(ident.to_string());
            }
        }
        syn::Expr::Assign(a) => {
            // Simple LHS `name = rhs` is a pure rebind; complex LHS
            // (e.g. `obj.field = ...`, `arr[i] = ...`) reads the
            // base/index instead.
            if let syn::Expr::Path(p) = a.left.as_ref()
                && let Some(ident) = path_as_single_ident(&p.path)
            {
                state.rebound_names.insert(ident.to_string());
            } else {
                visit_expr_for_loop_locals(&a.left, state);
            }
            visit_expr_for_loop_locals(&a.right, state);
        }
        syn::Expr::Binary(b) => {
            // Compound assign (`a += b`, `a |= b`, …) lowers to
            // `Expr::Binary` with one of the `BinOp::*Assign(_)`
            // variants; a simple-ident LHS is BOTH read and rebound.
            if is_compound_assign(b.op) {
                if let syn::Expr::Path(p) = b.left.as_ref()
                    && let Some(ident) = path_as_single_ident(&p.path)
                {
                    let name = ident.to_string();
                    state.read_names.insert(name.clone());
                    state.rebound_names.insert(name);
                } else {
                    visit_expr_for_loop_locals(&b.left, state);
                }
            } else {
                visit_expr_for_loop_locals(&b.left, state);
            }
            visit_expr_for_loop_locals(&b.right, state);
        }
        syn::Expr::Block(b) => visit_block_for_loop_locals(&b.block, state),
        syn::Expr::Unsafe(u) => visit_block_for_loop_locals(&u.block, state),
        syn::Expr::Async(a) => visit_block_for_loop_locals(&a.block, state),
        syn::Expr::If(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.then_branch, state);
            if let Some((_, else_branch)) = &e.else_branch {
                visit_expr_for_loop_locals(else_branch, state);
            }
        }
        syn::Expr::Match(m) => {
            visit_expr_for_loop_locals(&m.expr, state);
            for arm in &m.arms {
                if let Some((_, guard)) = &arm.guard {
                    visit_expr_for_loop_locals(guard, state);
                }
                visit_expr_for_loop_locals(&arm.body, state);
            }
        }
        syn::Expr::While(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Loop(e) => visit_block_for_loop_locals(&e.body, state),
        syn::Expr::ForLoop(e) => {
            // The for-loop's pat introduces a new inner-scope binding;
            // the allocator filters it out by intersecting against the
            // pre-loop framestate.  Visit iterable for reads + body
            // for any rebinds of *outer* locals.
            visit_expr_for_loop_locals(&e.expr, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Let(l) => {
            // `if let Some(x) = expr { .. }` — pattern bindings are
            // inner-scope; only the scrutinee is a read of the
            // enclosing names.
            visit_expr_for_loop_locals(&l.expr, state);
        }
        syn::Expr::Closure(_) => {
            // Slice 5b.2 contract: skip closure body.  A closure
            // captures enclosing locals via a synthetic capture list;
            // those captures do not flow through the loop's
            // straight-line control path, so bindings inside the
            // closure must not influence the header phi set.  RPython
            // parity: a closure compiles to its own bytecode with
            // `LOAD_DEREF` / `STORE_DEREF`, distinct from the outer
            // `LOAD_FAST` / `STORE_FAST` sequence.
        }
        syn::Expr::Call(c) => {
            visit_expr_for_loop_locals(&c.func, state);
            for arg in &c.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::MethodCall(m) => {
            visit_expr_for_loop_locals(&m.receiver, state);
            for arg in &m.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::Field(f) => visit_expr_for_loop_locals(&f.base, state),
        syn::Expr::Index(i) => {
            visit_expr_for_loop_locals(&i.expr, state);
            visit_expr_for_loop_locals(&i.index, state);
        }
        syn::Expr::Reference(r) => visit_expr_for_loop_locals(&r.expr, state),
        syn::Expr::Unary(u) => visit_expr_for_loop_locals(&u.expr, state),
        syn::Expr::Paren(p) => visit_expr_for_loop_locals(&p.expr, state),
        syn::Expr::Try(t) => visit_expr_for_loop_locals(&t.expr, state),
        syn::Expr::TryBlock(t) => visit_block_for_loop_locals(&t.block, state),
        syn::Expr::Tuple(t) => {
            for elem in &t.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Cast(c) => visit_expr_for_loop_locals(&c.expr, state),
        syn::Expr::Array(a) => {
            for elem in &a.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Range(r) => {
            if let Some(s) = &r.start {
                visit_expr_for_loop_locals(s, state);
            }
            if let Some(e) = &r.end {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Return(r) => {
            if let Some(e) = &r.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Yield(y) => {
            if let Some(e) = &y.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Await(a) => visit_expr_for_loop_locals(&a.base, state),
        syn::Expr::Group(g) => visit_expr_for_loop_locals(&g.expr, state),
        syn::Expr::Struct(s) => {
            for field in &s.fields {
                visit_expr_for_loop_locals(&field.expr, state);
            }
            if let Some(rest) = &s.rest {
                visit_expr_for_loop_locals(rest, state);
            }
        }
        syn::Expr::Repeat(r) => {
            visit_expr_for_loop_locals(&r.expr, state);
            visit_expr_for_loop_locals(&r.len, state);
        }
        // Read-free leaves and constructs that cannot be statically
        // introspected for reads.  Macro args fall here (same
        // fail-loud rationale as `Stmt::Macro`).
        _ => {}
    }
}

fn collect_pat_idents(pat: &syn::Pat, out: &mut std::collections::HashSet<String>) {
    match pat {
        syn::Pat::Ident(i) => {
            out.insert(i.ident.to_string());
            if let Some((_, sub)) = &i.subpat {
                collect_pat_idents(sub, out);
            }
        }
        syn::Pat::Type(t) => collect_pat_idents(&t.pat, out),
        syn::Pat::Tuple(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::TupleStruct(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::Struct(s) => {
            for field in &s.fields {
                collect_pat_idents(&field.pat, out);
            }
        }
        syn::Pat::Or(o) => {
            for case in &o.cases {
                collect_pat_idents(case, out);
            }
        }
        syn::Pat::Reference(r) => collect_pat_idents(&r.pat, out),
        syn::Pat::Paren(p) => collect_pat_idents(&p.pat, out),
        syn::Pat::Slice(s) => {
            for elem in &s.elems {
                collect_pat_idents(elem, out);
            }
        }
        // Other variants (Lit, Path, Range, Rest, Wild, Macro, …)
        // bind no local names directly.
        _ => {}
    }
}

fn path_as_single_ident(path: &syn::Path) -> Option<&syn::Ident> {
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        let seg = &path.segments[0];
        if matches!(seg.arguments, syn::PathArguments::None) {
            return Some(&seg.ident);
        }
    }
    None
}

fn is_compound_assign(op: syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// Eagerly allocate the loop header's phi inputargs for every name in
/// `must_merge.read_names ∪ must_merge.rebound_names` that is also
/// present in `pre_loop_snapshot`.  The walk visits
/// `pre_loop_snapshot.entries` in order so the resulting header
/// inputarg slot order mirrors Stage A2's first-bind positional
/// invariant — RPython parity with `flowspace/framestate.py:14
/// _union`'s positional zip over `locals_w`.
///
/// Mirrors `flowcontext.py:430 mergeblock`'s "create phi for every
/// mergeable local, then let `simplify.transform_dead_op_vars` DCE
/// the unread ones".  Names rebound in the body but never read still
/// receive a transient header phi; `model::prune_dead_phis`
/// (`simplify.py:484-524` blanket DCE) removes the phi + matching
/// link.args slot when no graph-level reader appears.
///
/// For each surviving name this routine:
///   1. emits `OpKind::Input { name, ty: pre_loop_entry.value_type }`
///      at `header_entry`, allocating a fresh `ValueId` (the phi vid),
///   2. pushes that phi vid onto `header_entry.inputargs`,
///   3. pushes the **pre-loop** vid onto `pre_loop_block.exits[0].args`
///      so the forward-edge `Link.args` matches the new header arity,
///   4. updates `ctx.local_value_ids[name]` to point at the phi vid
///      (with `header_entry` as the new defining block) so the body
///      walk reads the phi rather than the pre-loop vid,
///   5. updates `ctx.local_value_types[name]` to the carried
///      `value_type`.
///
/// Names in `must_merge.read_names ∪ must_merge.rebound_names` that are
/// absent from `pre_loop_snapshot` are skipped — they are body-only locals
/// (e.g. the binding pattern of an inner `for` / `let` inside the body) that
/// have no pre-loop counterpart to merge with.  RPython parity:
/// `framestate.py:110-111` "if w1 is None or w2 is None: return None"
/// drops one-sided slots; here the pre-scan saw the binding but the
/// snapshot says it never existed pre-loop, so it cannot be a header
/// phi candidate.  A future body lowering pass that needs to read
/// such a name allocates it as a fresh `OpKind::Input` at its def
/// block as it does today.
///
/// The caller (Slice 5c.1+) is responsible for closing the back-edge
/// from `body_tail` to `header_entry` by fetching, per name in the
/// returned `Vec<String>`, the body's current `ctx.local_value_ids[name]`
/// and pushing those vids onto a `set_goto(body_tail, header_entry,
/// args)` call.  RPython parity for the back-edge close: the same
/// slot-by-slot mapping `framestate.py:92 getoutputargs` produces for
/// the closing predecessor's link.
///
/// Pre-conditions:
///   - `pre_loop_block` is already closed with a single goto-style
///     exit to `header_entry` (e.g. via `graph.set_goto(pre_loop_block,
///     header_entry, vec![])`).  This routine pushes onto that
///     existing exit's `args` rather than re-closing the block.
///   - `pre_loop_snapshot` was produced by
///     `ctx.snapshot_locals_for_link()` (or constructed from the same
///     ctx state) so its entries are in first-bind positional order.
///
/// Returns the ordered list of header-phi names.
fn allocate_loop_header_phis(
    graph: &mut FunctionGraph,
    ctx: &mut GraphBuildContext<'_>,
    pre_loop_block: BlockId,
    header_entry: BlockId,
    pre_loop_snapshot: &FrameState,
    must_merge: &LoopBodyLocals,
) -> Vec<String> {
    let mut header_phi_names = Vec::new();
    for entry in pre_loop_snapshot.entries.iter().filter_map(|e| e.as_ref()) {
        // `flowcontext.py:430 mergeblock` allocates a phi for every
        // mergeable local; `simplify.transform_dead_op_vars` then
        // DCEs the unread ones.  Pyre mirrors that: predicate is
        // `read_names ∪ rebound_names`, prune_dead_phis prunes the
        // transient `rebound-only-no-read` phi.
        if !must_merge.read_names.contains(&entry.name)
            && !must_merge.rebound_names.contains(&entry.name)
        {
            continue;
        }
        let value_type = entry.value_type.clone();
        let phi_vid = graph
            .push_op(
                header_entry,
                OpKind::Input {
                    name: entry.name.clone(),
                    ty: value_type.clone(),
                },
                true,
            )
            .expect("OpKind::Input always produces a result");
        graph.name_value(phi_vid, entry.name.clone());
        graph.block_mut(header_entry).inputargs.push(phi_vid);
        graph.block_mut(pre_loop_block).exits[0]
            .args
            .push(LinkArg::from(entry.value_id));
        ctx.bind_local_id(entry.name.clone(), phi_vid, header_entry);
        ctx.local_value_types.insert(entry.name.clone(), value_type);
        header_phi_names.push(entry.name.clone());
    }
    header_phi_names
}

/// Walk `header.inputargs` and recover, in inputarg order, the local
/// name attached to each `OpKind::Input { name, .. }` op whose result
/// is that inputarg.  Used by `Expr::While` / `Expr::Loop` (Slice
/// 5c.1+) to capture the FROZEN header-phi name list once the loop
/// header is fully populated — both by `allocate_loop_header_phis`'s
/// eager phis and by any cond-driven cross-block lazy installs.  The
/// returned list drives the back-edge close and `Expr::Continue` per-
/// name link-arg threading; it must match `header.inputargs.len()`.
fn header_phi_name_list(graph: &FunctionGraph, header: BlockId) -> Vec<String> {
    let header_block = graph.block(header);
    header_block
        .inputargs
        .iter()
        .filter_map(|&iarg_vid| {
            header_block
                .operations
                .iter()
                .find_map(|op| match (&op.kind, op.result) {
                    (OpKind::Input { name, .. }, Some(r)) if r == iarg_vid => Some(name.clone()),
                    _ => None,
                })
        })
        .collect()
}

/// Materialise a `Vec<ValueId>` of per-name link args for a back-edge
/// or `continue` close: each name's `ctx.local_value_ids[name].0`
/// supplies the value the loop header should observe on this edge.
/// Used at the closing predecessor of `Expr::While` / `Expr::Loop`'s
/// loop header.  RPython parity for the slot-by-slot mapping:
/// `flowspace/framestate.py:92 getoutputargs`.  Panics if any name is
/// absent from `ctx.local_value_ids` — `header_phi_names` is captured
/// from already-installed `OpKind::Input` ops and the eager allocator
/// (or the lazy installer that produced any extra phis) bound each
/// name into ctx, so a missing entry indicates a broken invariant
/// upstream of this call.
fn link_args_from_ctx(ctx: &GraphBuildContext<'_>, header_phi_names: &[String]) -> Vec<ValueId> {
    header_phi_names
        .iter()
        .map(|name| {
            let &(vid, _def_block) = ctx.local_value_ids.get(name).unwrap_or_else(|| {
                panic!(
                    "header phi name {:?} must have a current ctx binding at the closing \
                     predecessor",
                    name
                )
            });
            vid
        })
        .collect()
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
            local_value_ids: HashMap::new(),
            local_trait_bound_roots: HashMap::new(),
            generic_trait_roots: HashMap::new(),
            local_array_types: HashMap::new(),
            local_dyn_trait_roots: HashMap::new(),
            local_closure_returns: HashMap::new(),
            struct_fields,
            fn_return_types,
            module_prefix: module_prefix.to_string(),
            known_struct_names,
            known_trait_names,
            loop_stack: Vec::new(),
            local_first_bind_order: Vec::new(),
            local_first_bind_seen: std::collections::HashSet::new(),
        }
    }

    /// Bind a local name to a `(ValueId, defining_block)` pair, updating
    /// `local_value_ids` in place.  On *first* bind (name never seen by
    /// this graph) the name is also appended to
    /// `local_first_bind_order` and recorded in
    /// `local_first_bind_seen` so its slot position is fixed for the
    /// remainder of the build, even across `LocalBindingSnapshot::
    /// restore`.  On rebind the slot position is preserved.  RPython
    /// parity: `co_varnames` slot indices are assigned at compile time
    /// and never reshuffled.
    fn bind_local_id(&mut self, name: String, vid: ValueId, defining_block: BlockId) {
        if !self.local_first_bind_seen.contains(&name) {
            self.local_first_bind_seen.insert(name.clone());
            self.local_first_bind_order.push(name.clone());
        }
        self.local_value_ids.insert(name, (vid, defining_block));
    }
}

struct LocalBindingSnapshot {
    local_type_roots: HashMap<String, String>,
    local_type_strings: HashMap<String, String>,
    local_value_types: HashMap<String, ValueType>,
    local_value_ids: HashMap<String, (ValueId, BlockId)>,
    local_trait_bound_roots: HashMap<String, String>,
    local_array_types: HashMap<String, String>,
    local_dyn_trait_roots: HashMap<String, String>,
    local_closure_returns: HashMap<String, String>,
    // NB: `local_first_bind_order` and `local_first_bind_seen` are
    // intentionally NOT captured here — they are graph-wide
    // append-only and survive sibling-arm restores so the slot index
    // of any name stays invariant across the entire lowering.
    // RPython parity: `co_varnames` is fixed at compile time, so
    // the slot map cannot be rolled back by control flow.
}

impl LocalBindingSnapshot {
    fn capture(ctx: &GraphBuildContext<'_>) -> Self {
        Self {
            local_type_roots: ctx.local_type_roots.clone(),
            local_type_strings: ctx.local_type_strings.clone(),
            local_value_types: ctx.local_value_types.clone(),
            local_value_ids: ctx.local_value_ids.clone(),
            local_trait_bound_roots: ctx.local_trait_bound_roots.clone(),
            local_array_types: ctx.local_array_types.clone(),
            local_dyn_trait_roots: ctx.local_dyn_trait_roots.clone(),
            local_closure_returns: ctx.local_closure_returns.clone(),
        }
    }

    fn restore(self, ctx: &mut GraphBuildContext<'_>) {
        ctx.local_type_roots = self.local_type_roots;
        ctx.local_type_strings = self.local_type_strings;
        ctx.local_value_types = self.local_value_types;
        ctx.local_value_ids = self.local_value_ids;
        ctx.local_trait_bound_roots = self.local_trait_bound_roots;
        ctx.local_array_types = self.local_array_types;
        ctx.local_dyn_trait_roots = self.local_dyn_trait_roots;
        ctx.local_closure_returns = self.local_closure_returns;
        // local_first_bind_order / local_first_bind_seen NOT
        // restored — see struct doc comment.
    }
}

impl<'a> GraphBuildContext<'a> {
    /// Capture the current locals as a `FrameState` suitable for
    /// threading through `Link.args` to a successor block.  Locals walk
    /// `local_first_bind_order` (Stage A2) so the slot index of a name
    /// is invariant across the lowering — every predecessor emit and
    /// every successor consume line up at the same slot position.
    /// Names that have been unbound by an enclosing `LocalBindingSnapshot
    /// ::restore` are filtered out.  RPython parity: `co_varnames`
    /// slot order (`flowspace/flowcontext.py:835 LOAD_FAST` reads
    /// slots by index).
    ///
    /// Slice 2 / Stage A1 calls this at every `set_branch` / `set_goto`
    /// site in `Expr::If` and stores the result on the closing block's
    /// `Block.framestate` field; Slices 3-6 extend the callers to match
    /// arms, loop headers, and `break` / `continue` joins.
    fn snapshot_locals_for_link(&self) -> FrameState {
        let entries = self
            .local_first_bind_order
            .iter()
            .map(|name| {
                self.local_value_ids
                    .get(name)
                    .map(|&(value_id, _defining_block)| FrameStateEntry {
                        name: name.clone(),
                        value_id,
                        value_type: self
                            .local_value_types
                            .get(name)
                            .cloned()
                            .unwrap_or(ValueType::Unknown),
                    })
            })
            .collect();
        FrameState { entries }
    }
}

/// Lazy cross-block local installer — Slice 2 of the Cat 2.1 epic.
///
/// Triggered from `Expr::Path`'s cross-block branch when the local
/// `name` is bound in a block other than `current_block`.  Allocates a
/// fresh `OpKind::Input { name, ty }` in `current_block`, registers it
/// as `current_block.inputargs`, rewrites `ctx.local_value_ids[name]`
/// to point at the new inputarg `ValueId`, and **threads back** the
/// predecessor side of the join: for every immediate predecessor edge
/// `(pred_block, exit_idx)` landing at `current_block`, the snapshot
/// recorded in `pred_block.framestate` supplies a candidate
/// predecessor-side `ValueId` for `name`.  When that
/// candidate is itself defined in `pred_block` (its inputarg or an
/// op result), it is appended to `pred_block.exits[exit_idx].args`
/// directly.  When it was inherited from a dominator (not defined
/// in `pred_block` itself — e.g. an empty intermediate merge block
/// that just forwards a parameter), the installer **recurses** to
/// install `name` as an inputarg of `pred_block` first, walking the
/// predecessor chain back until the recursion lands on a block that
/// defines the local.  RPython: equivalent of
/// `flowspace/flowcontext.py:835 LOAD_FAST` pulling a fresh
/// `Variable` into the merge block while `flowspace/model.py:114
/// Link.__init__` keeps `len(args) == len(target.inputargs)`
/// invariant — RPython does this work implicitly because every basic
/// block edge in the recorder threads the `frame.locals_w` slot
/// array; pyre's frontend builds the same shape on demand only at
/// cross-block reads (lazy) instead of preemptively at every block
/// boundary (eager) so blocks with no cross-block readers stay
/// zero-arity.
///
/// **Stage B2 (final)**: the Slice 2 conservative fence is gone — the
/// installer fires for every cross-block local read regardless of
/// `current_block`'s op count.  The hazard the fence used to mask was
/// (a) duplicate phi inputargs for the same name (closed by the
/// idempotency check below — a second read of `name` in
/// `current_block` reuses the inputarg from an earlier install), and
/// (b) UnaryOp("neg", Float) annotating to Int by default and
/// poisoning downstream phi-merge inputargs through union(Int,
/// Float) → Unknown → GcRef backfill (closed by the
/// `rfloat.py:rtype_neg` parity arm in
/// `translator/rtyper/legacy_annotator.rs` and the matching
/// `infer_concrete_from_op` Unknown-pass-through in
/// `translator/rtyper/legacy_resolve.rs`).
///
/// Returns `Some(new_vid)` on success, `None` if any predecessor lacks
/// a recorded snapshot or whose snapshot lacks `name`, or the
/// snapshots disagree on `value_type`.  The call site falls back to
/// the legacy naked-`Input` emit when `None` is returned.
/// `pre_allocated_vid`: when `Some(vid)`, use the caller-supplied
/// ValueId for the fresh phi instead of allocating a new one.  Used
/// by union callers (`Expr::If`, `Expr::Match`) that pre-allocate
/// phi vids inside `FrameState::union(_into)` so the merged state
/// can be returned with vids materialised; the install is then
/// emitted with the same vid the merged state already carries.
/// `None` preserves the legacy behaviour (allocate inside).
fn lazy_install_local_at_current_block(
    graph: &mut crate::model::FunctionGraph,
    ctx: &mut GraphBuildContext<'_>,
    current_block: BlockId,
    name: &str,
    pre_allocated_vid: Option<ValueId>,
) -> Option<ValueId> {
    // Reuse — `name` may already have been installed at `current_block`
    // by an earlier read in the same block (prior recursion into a
    // shared predecessor, etc.).  Treat the same-block hit as the
    // canonical answer.
    if let Some(&(vid, def_block)) = ctx.local_value_ids.get(name)
        && def_block == current_block
    {
        return Some(vid);
    }

    // Idempotency by graph state: if a prior lazy install for `name`
    // already added an inputarg-anchored `OpKind::Input { name }` to
    // `current_block`, reuse it.  RPython parity:
    // `flowcontext.py:407 setstate(block.framestate)` makes the same
    // local slot read multiple times in a block resolve to the same
    // Variable; the duplicate-install hazard arises because pyre's
    // `LocalBindingSnapshot::restore` (Stage B1) wipes the ctx-side
    // `local_value_ids` cache between then/else arms while the
    // graph-side inputarg from the then-arm's lazy install still
    // exists, so the else-arm's recursion would otherwise allocate a
    // second phi slot for the same name.  Checking the graph's
    // inputargs list directly closes the gap without depending on
    // ctx state.
    {
        let block = graph.block(current_block);
        for op in &block.operations {
            if let (Some(result), OpKind::Input { name: op_name, .. }) = (op.result, &op.kind)
                && op_name == name
                && block.inputargs.contains(&result)
            {
                return Some(result);
            }
        }
    }

    let pred_edges: Vec<(BlockId, usize)> = graph
        .blocks
        .iter()
        .flat_map(|b| {
            let bid = b.id;
            b.exits.iter().enumerate().filter_map(move |(i, exit)| {
                if exit.target == current_block {
                    Some((bid, i))
                } else {
                    None
                }
            })
        })
        .collect();
    if pred_edges.is_empty() {
        return None;
    }

    // Audit Cat 2-1 cycle-safe shape: split into two phases.  The
    // outer `OpKind::Input` is pushed and the inputarg slot is
    // reserved BEFORE any recursive predecessor walk, so a recursion
    // back into `current_block` (e.g. a back-edge predecessor whose
    // own pred-walk reaches `current_block`) finds the in-progress
    // phi via the graph-state idempotency check above and short-
    // circuits with the pre-allocated vid.  RPython
    // `flowspace/flowcontext.py:438-451 mergeblock` reaches the same
    // fixpoint by iteratively applying `union` until predecessor
    // links converge — pyre's static AST collapses that to a single
    // pass with the in-flight inputarg slot acting as the iteration
    // checkpoint.
    //
    // Phase 1 (read-only): collect each predecessor's `(snap_vid,
    // snap_type)` and decide whether the snapshot's vid is directly
    // usable in `pred_block` or whether `pred_block` will need its
    // own recursive install.  All predecessor snapshots must agree
    // on `value_type` via the wildcard rule (`Unknown` is a wildcard
    // — concrete-vs-concrete disagreement aborts before we mutate
    // graph state).
    struct PredSnap {
        pred_block: BlockId,
        exit_idx: usize,
        snap_vid: ValueId,
        snap_type: ValueType,
        needs_recurse: bool,
    }
    let mut pred_snaps: Vec<PredSnap> = Vec::with_capacity(pred_edges.len());
    let mut shared_value_type: Option<ValueType> = None;
    for (pred_block, exit_idx) in &pred_edges {
        // Stage A1: predecessor framestate is now per-block, captured
        // at close time (`Block.framestate`).  All exits of one
        // `set_branch` share the same close-time state; the per-edge
        // `ctx.exit_snapshots` HashMap is gone.  RPython parity:
        // `flowspace/flowcontext.py:38 SpamBlock.framestate`.
        let (snap_vid, snap_type) = {
            let snap = graph.block(*pred_block).framestate.as_ref()?;
            let entry = snap.entries.iter().find_map(|slot| match slot {
                Some(e) if e.name == name => Some(e),
                _ => None,
            })?;
            (entry.value_id, entry.value_type.clone())
        };
        let needs_recurse = !value_id_defined_in_block(graph, snap_vid, *pred_block);
        // Tentative type from snap-only fold for non-recursive preds
        // we can also consult `graph_value_type(snap_vid)`; for
        // recursive preds the recursive install's value_type isn't
        // known yet and we use snap_type as the best estimate.  The
        // wildcard rule (`Unknown` absorbs concrete) is symmetric and
        // never produces a kind disagreement when one side is
        // Unknown.
        let observed_type = if needs_recurse {
            snap_type.clone()
        } else {
            graph_value_type(graph, snap_vid).unwrap_or_else(|| snap_type.clone())
        };
        match (&shared_value_type, &observed_type) {
            (None, _) => shared_value_type = Some(observed_type.clone()),
            (Some(prior), new) if prior == new => {}
            (Some(ValueType::Unknown), _) => shared_value_type = Some(observed_type.clone()),
            (Some(_), ValueType::Unknown) => {}
            (Some(_), _) => return None,
        }
        pred_snaps.push(PredSnap {
            pred_block: *pred_block,
            exit_idx: *exit_idx,
            snap_vid,
            snap_type,
            needs_recurse,
        });
    }

    let value_type = shared_value_type.unwrap_or(ValueType::Unknown);

    // Phase 2 (graph mutation): push the outer `OpKind::Input` +
    // inputarg slot first, snapshot the prior ctx binding for
    // rollback, then walk predecessors (recursively installing where
    // needed).  The recursion's same-block idempotency check on
    // `block.inputargs` finds this outer call's freshly-installed
    // inputarg and returns its vid, breaking the
    // `current_block → pred → ... → current_block` cycle that
    // arises when a back-edge predecessor's snap_vid is defined in
    // `current_block` (e.g. body_tail / continue source whose
    // framestate snap for a header-phi name points back at the
    // header).
    let prior_ctx_lvi = ctx.local_value_ids.get(name).copied();
    let prior_ctx_lvt = ctx.local_value_types.get(name).cloned();
    let new_vid = if let Some(vid) = pre_allocated_vid {
        graph.push_op_with_result(
            current_block,
            OpKind::Input {
                name: name.to_string(),
                ty: value_type.clone(),
            },
            vid,
        );
        vid
    } else {
        graph.push_op(
            current_block,
            OpKind::Input {
                name: name.to_string(),
                ty: value_type.clone(),
            },
            true,
        )?
    };
    graph.name_value(new_vid, name.to_string());
    graph.block_mut(current_block).inputargs.push(new_vid);
    ctx.bind_local_id(name.to_string(), new_vid, current_block);
    ctx.local_value_types
        .insert(name.to_string(), value_type.clone());
    // Predecessor source vids whose graph result type was Unknown when
    // observed.  If the wildcard rule above promoted `value_type` to
    // a concrete kind from a sibling predecessor, every Unknown
    // source must be retagged so the freshly-installed merge
    // inputarg's `i`/`r`/`f` kind matches every incoming link arg at
    // codegen.  Without retag, the merge inputarg gets concrete `ty`
    // but a predecessor's source op still produces an Unknown-banked
    // value, tripping the assembler's `int_copy` / `ref_copy` /
    // `float_copy` same-kind invariant.
    let mut unknown_predecessor_vids: Vec<ValueId> = Vec::new();
    let mut pred_link_args: Vec<(BlockId, usize, ValueId)> = Vec::with_capacity(pred_snaps.len());
    let mut rollback = false;
    for snap in pred_snaps {
        let resolved_vid = if snap.needs_recurse {
            match lazy_install_local_at_current_block(graph, ctx, snap.pred_block, name, None) {
                Some(v) => v,
                None => {
                    rollback = true;
                    break;
                }
            }
        } else {
            snap.snap_vid
        };
        let resolved_type =
            graph_value_type(graph, resolved_vid).unwrap_or_else(|| snap.snap_type.clone());
        // Phase 2 re-validation: Phase 1's `value_type` for needs_recurse
        // preds was estimated from `snap_type` because the recursive
        // install hadn't run yet.  The actual `resolved_type` after
        // recursion may surface a concrete kind that disagrees with
        // `value_type` (e.g. a deeper predecessor's snap stored
        // `Unknown` but the install resolved to `Int`, or two recursive
        // chains converged on disagreeing concrete kinds).  Concrete-
        // vs-concrete disagreement here MUST roll back the outer
        // Phase-2 push: otherwise the freshly-installed inputarg's
        // `ty` mismatches the incoming link arg's kind classification
        // and the assembler's `int_copy` / `ref_copy` / `float_copy`
        // same-kind invariant trips at codegen.  `Unknown` remains a
        // wildcard on either side — matches Phase 1's wildcard rule
        // and `FrameState::union`'s per-slot semantics.
        let kind_clash = match (&value_type, &resolved_type) {
            (a, b) if a == b => false,
            (ValueType::Unknown, _) | (_, ValueType::Unknown) => false,
            _ => true,
        };
        if kind_clash {
            rollback = true;
            break;
        }
        if matches!(resolved_type, ValueType::Unknown) {
            unknown_predecessor_vids.push(resolved_vid);
        }
        pred_link_args.push((snap.pred_block, snap.exit_idx, resolved_vid));
    }

    if rollback {
        // Undo Phase 2's graph + ctx mutations so the caller's
        // naked-`Input` fallback emits without a dangling inputarg
        // at `current_block`.  Recursive installs that succeeded at
        // sibling pred_blocks before the failing one are NOT rolled
        // back — those are valid installs at their own blocks and
        // remain useful for any later read.  The only orphan to
        // clean is `current_block`'s own outer inputarg.
        let block = graph.block_mut(current_block);
        let popped_op = block.operations.pop();
        debug_assert!(
            matches!(popped_op.as_ref().map(|op| &op.kind), Some(OpKind::Input { name: n, .. }) if n == name),
            "rollback expected Input op for {name:?} at the operations tail",
        );
        let popped_inputarg = block.inputargs.pop();
        debug_assert_eq!(popped_inputarg, Some(new_vid));
        match prior_ctx_lvi {
            Some((vid, def_block)) => {
                ctx.bind_local_id(name.to_string(), vid, def_block);
            }
            None => {
                ctx.local_value_ids.remove(name);
            }
        }
        match prior_ctx_lvt {
            Some(vt) => {
                ctx.local_value_types.insert(name.to_string(), vt);
            }
            None => {
                ctx.local_value_types.remove(name);
            }
        }
        return None;
    }

    // If the wildcard rule promoted `value_type` to a concrete kind,
    // retag every Unknown-typed predecessor source so the assembler's
    // same-kind copy invariant holds across each incoming link arg.
    // Mirrors the `Expr::Try` `retag_result_value_type` widening used
    // after `?` unwraps a `Result<T, E>` — same operation, here
    // driven by cross-arm wildcard widening rather than the
    // type-extractor's `Ok`-arm narrowing.
    if !matches!(value_type, ValueType::Unknown) {
        for vid in &unknown_predecessor_vids {
            retag_result_value_type(graph, *vid, value_type.clone());
        }
    }

    for (pred_block, exit_idx, pred_vid) in pred_link_args {
        graph.block_mut(pred_block).exits[exit_idx]
            .args
            .push(crate::model::LinkArg::Value(pred_vid));
    }

    Some(new_vid)
}

/// `true` iff `vid` is defined inside `block_id` — either as an entry
/// in the block's `inputargs` or as the result of one of its
/// `operations`.  RPython's `flowspace/model.py:checkgraph`
/// "variable used in more than one block" assertion (lines
/// 3886-3893 in the Rust port) requires every operand referenced from
/// a block to be defined in that block, so the lazy installer uses
/// this predicate to decide whether a snapshot vid can be threaded
/// directly or whether the predecessor needs its own inputarg
/// allocation first.
fn value_id_defined_in_block(
    graph: &crate::model::FunctionGraph,
    vid: ValueId,
    block_id: BlockId,
) -> bool {
    let block = graph.block(block_id);
    if block.inputargs.contains(&vid) {
        return true;
    }
    block.operations.iter().any(|op| op.result == Some(vid))
}

// Build a SemanticFunction from a Rust function AST. Mirrors RPython
// `flowspace/objspace.py:38` `build_flow()` — `FlowingError` propagates to
// the caller rather than producing a partial graph.
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
                    // RPython `LOAD_FAST` parity: record the receiver
                    // binding so a body `Expr::Path` reference to
                    // `self` within the entry block reuses this
                    // `ValueId` instead of emitting a fresh
                    // `OpKind::Input` — same treatment as typed
                    // parameters on the `FnArg::Typed` arm below
                    // (`flowspace/flowcontext.py:835`).
                    ctx.bind_local_id("self".to_string(), vid, entry);
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
                        ty: arg_ty.clone(),
                    },
                    true,
                ) {
                    graph.name_value(vid, name.clone());
                    graph.block_mut(entry).inputargs.push(vid);
                    // RPython `LOAD_FAST` parity: record the parameter
                    // binding so a body `Expr::Path` reference within
                    // the entry block reuses this `ValueId` instead of
                    // emitting a fresh `OpKind::Input`
                    // (`flowspace/flowcontext.py:835`).
                    ctx.bind_local_id(name.clone(), vid, entry);
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
    let hints = collect_jit_hints(&func.attrs, Some(&func.sig));

    // Direct port of `simplify.transform_dead_op_vars_in_blocks`
    // (`rpython/translator/simplify.py:422`) — the dead-op subset of
    // upstream's `simplify_graph` (`simplify.py:1075`).  The
    // upstream algorithm uses backward dataflow with
    // link-args-as-dependencies (not direct reads), which correctly
    // retires body-dead phis whose only "reader" is the back-edge
    // itself.  Running it here ensures downstream rtyper / jtransform
    // / flatten / regalloc never see an unread phi or its dangling
    // `Link.args` slot.
    //
    // TODO(simplify_graph-full-port): pyre runs only the
    // `transform_dead_op_vars` subset here.  The full upstream
    // `simplify_graph` runs `all_passes`
    // (`simplify.py:1060-1073`):
    //
    //     transform_dead_op_vars, eliminate_empty_blocks,
    //     remove_assertion_errors, remove_identical_vars_SSA,
    //     constfold_exitswitch, remove_trivial_links, SSA_to_SSI,
    //     coalesce_bool, transform_ovfcheck, simplify_exceptions,
    //     transform_xxxitem, remove_dead_exceptions
    //
    // pyre only ports the first.  The remainder are either deferred
    // (no-op pyre IR doesn't need them yet) or handled by later
    // phases under different names (e.g. constant folding lives in
    // optimizeopt rather than as a flowspace-level
    // constfold_exitswitch).  Convergence path: port each remaining
    // pass under its upstream name as the corresponding pyre IR
    // construct lands.
    crate::model::prune_dead_phis(&mut graph);

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
fn collect_jit_hints(attrs: &[syn::Attribute], sig: Option<&syn::Signature>) -> Vec<String> {
    let mut hints = Vec::new();
    let mut saw_oopspec = false;
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            match name.as_str() {
                // RPython-parity names (rlib/jit.py)
                "elidable" | "jit_elidable" => hints.push("elidable".into()),
                // `rlib/jit.py:184-201 elidable_promote` is no longer
                // collapsed onto the user-facing function's hints here.
                // `build_graphs_from_items` synthesizes the
                // (`_orig_<NAME>_unlikely_name`, wrapper) pair before
                // this collector ever runs, attaches the synthetic
                // `#[elidable]` attribute to the original, and strips
                // `#[elidable_promote]` from the wrapper's `attrs`.
                // The orig is what RPython's `elidable(func)` at
                // jit.py:185 marks with `_elidable_function_`; the
                // wrapper (`result` at jit.py:198-201) carries no
                // binary flag.  `synthesize_elidable_promote_pair`
                // always succeeds — unrecognised binder patterns (which
                // Python signatures cannot express anyway) panic with
                // a citation to `jit.py:172-178 _get_args(func)`, so
                // there is no silent single-graph fallback.
                //
                // `call.py:292-299 getcalldescr` still runs the
                // `_canraise(op)` analysis on every elidable callsite,
                // so collapsing `elidable_cannot_raise` /
                // `elidable_or_memerror` to a single `"elidable"` hint
                // remains parity-correct — `getcalldescr` recovers the
                // `EF_ELIDABLE_*` 3-way from the per-op raise analysis
                // at `jit_codewriter/call.rs:2773-2782`.
                "elidable_cannot_raise" | "elidable_or_memerror" => {
                    hints.push("elidable".into());
                }
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
                    saw_oopspec = true;
                }
                // majit-specific
                "jit_close_stack" => hints.push("close_stack".into()),
                "jit_cannot_collect" => hints.push("cannot_collect".into()),
                "jit_gc_effects" => hints.push("gc_effects".into()),
                _ => {}
            }
        }
    }
    // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
    // — when `#[oopspec(...)]` is present and the function signature
    // is available, emit a paired `"oopspec_argnames:arg1,arg2,..."`
    // hint so `lib.rs:598-600` can populate
    // `CallControl::mark_oopspec_argnames` alongside `mark_oopspec`.
    // `support.py:713 argname2index = dict(zip(argnames, [Index(n) for n in range(nb_args)]))`
    // requires the declaration-order names; `self` is skipped because
    // upstream `co_varnames` for a method's lifted free function would
    // not include it (RPython doesn't have `self`-as-receiver in
    // oopspec helpers, and pyre's `#[oopspec(...)]` macro is only
    // applied to free fns / static methods per
    // `majit-macros/src/lib.rs:1316`).
    if saw_oopspec {
        if let Some(sig) = sig {
            let argnames: Vec<String> = sig
                .inputs
                .iter()
                .filter_map(|arg| match arg {
                    syn::FnArg::Typed(pat_type) => match &*pat_type.pat {
                        syn::Pat::Ident(ident) => Some(ident.ident.to_string()),
                        _ => None,
                    },
                    // `&self` / `self` receivers carry no positional
                    // argname for the oopspec parser to bind against.
                    syn::FnArg::Receiver(_) => None,
                })
                .collect();
            if !argnames.is_empty() {
                hints.push(format!("oopspec_argnames:{}", argnames.join(",")));
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
                        // Prefer the statically-bool classification when
                        // the init expression is a `Lit::Bool` / `!x` /
                        // comparison / `&&`/`||` / registered
                        // bool-returning call, etc. (`expr_is_statically_bool`).
                        // `graph_value_type` would otherwise return
                        // `ValueType::Int` for a `Lit::Bool` lowered as
                        // `OpKind::ConstInt(0/1)`, which would make the
                        // next `!b` classifier choose the bitwise-invert
                        // path — `let b = true; !b` would emit
                        // `int_invert` instead of bool+branch. RPython
                        // annotates `Constant(True)` with `SomeBool`
                        // (`annotator/model.py:185-227`) so the let-bind
                        // here records `ValueType::Bool` to keep the
                        // lattice node distinct from Int.
                        let bool_override = expr_is_statically_bool(&init.expr, ctx);
                        if bool_override {
                            ctx.local_value_types.insert(name.clone(), ValueType::Bool);
                        } else if let Some(ty) = graph_value_type(graph, vid) {
                            ctx.local_value_types.insert(name.clone(), ty);
                        }
                        // RPython `LOAD_FAST` parity: record the
                        // let-binding's `(ValueId, defining BlockId)`
                        // so a same-block `Expr::Path` reference
                        // reuses this `ValueId` instead of emitting a
                        // fresh `OpKind::Input`
                        // (`flowspace/flowcontext.py:835`).
                        ctx.bind_local_id(name.clone(), vid, *block);
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
                            ctx.local_type_strings.insert(name.clone(), type_string);
                        }
                        // `let f = |args| body;` / `let f = |args| ->
                        // RetTy body;` — the rhs is a closure, which
                        // pyre's walker doesn't surface as a graph
                        // function. Register the closure's return type
                        // under the local ident so a downstream Call
                        // `f(...)` resolves through `lookup_function_
                        // return_type`'s bare-key fallback and
                        // classifier sites (`expr_unary_not_operand_
                        // kind`) get a kind. RPython peer:
                        // `bookkeeper.getdesc(value)` resolves any
                        // callable in scope by host-identity; the
                        // static walker substitutes by registering the
                        // closure return type under the bare ident.
                        if let syn::Expr::Closure(closure) = &*init.expr {
                            let closure_ret = match &closure.output {
                                syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                                    ty,
                                    &ctx.module_prefix,
                                    ctx.known_struct_names,
                                    ctx.known_trait_names,
                                ),
                                syn::ReturnType::Default => {
                                    expression_type_string(&closure.body, ctx)
                                }
                            };
                            if let Some(ret_ty) = closure_ret {
                                ctx.local_closure_returns.insert(name, ret_ty);
                            }
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
                        nolength: nolength_from_array_type_id(array_type_id.as_deref()),
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
                            nolength: nolength_from_array_type_id(array_type_id.as_deref()),
                            array_type_id,
                        },
                        false,
                    );
                }
                syn::Expr::Path(path) if path.path.segments.len() == 1 && path.qself.is_none() => {
                    // Generic local assignment `x = rhs` — RPython STORE_FAST
                    // parity (`flowspace/flowcontext.py:878-885`):
                    //
                    //     w_newvalue = self.popvalue()
                    //     ...
                    //     self.locals_w[varindex] = w_newvalue
                    //     if isinstance(w_newvalue, Variable):
                    //         w_newvalue.rename(self.getlocalvarname(varindex))
                    //
                    // Two effects: replace the locals slot for `x`
                    // with the rhs `ValueId`, and rename the rhs
                    // `Variable` to the local name so diagnostics and
                    // the adapter's `name_to_value` lookup pick the
                    // rhs up under that name.  Same-block dedup
                    // machinery installed at `lower_stmt`'s let arm
                    // (`ast.rs:1389 local_value_ids.insert`) caches
                    // `(let-rhs ValueId, defining block)`; without
                    // this STORE_FAST update a later `x` read returns
                    // the stale let value.  RPython only renames
                    // when the rhs `is Variable`; the
                    // `is_constant_define_value` gate skips the
                    // ValueId-keyed `name_value` for `ConstInt`/
                    // `ConstFloat` define-ops (RPython `Constant`).
                    let name = path
                        .path
                        .segments
                        .iter()
                        .map(|seg| seg.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    ctx.bind_local_id(name, value, *block);
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

            // Cat 2.1 Slice 2 / Stage A1: capture the locals frame as it
            // was when `*block` closed via `set_branch` so a later
            // cross-block read in the merge block can thread back
            // through either arm's `Link.args` even when the arm itself
            // rebinds nothing.  Stored on `Block.framestate` (per-block,
            // captured at close time) — both exits of one set_branch
            // share the same pre-branch snapshot, so the per-edge
            // duplication of Slice 2 collapses into a single field.
            // RPython parity: `flowspace/flowcontext.py:38
            // SpamBlock.framestate`.
            let pre_branch_snapshot = ctx.snapshot_locals_for_link();
            graph.set_branch(*block, cond, then_block, vec![], else_block, vec![]);
            graph.block_mut(*block).framestate = Some(pre_branch_snapshot);

            // Stage B1: capture the pre-branch ctx state BEFORE
            // lowering the then-arm so the else-arm can re-enter
            // `*block`'s scope.  RPython parity:
            // `flowspace/flowcontext.py:407-408 record_block(block)`
            // calls `setstate(block.framestate)` at every block
            // entry; pyre snapshots the analogue `LocalBindingSnapshot`
            // here and restores it before the else-arm.  Without this
            // restore the else-arm sees the then-arm's mutations to
            // `ctx.local_value_ids` / `local_value_types` etc.
            let pre_branch_ctx = LocalBindingSnapshot::capture(ctx);

            // Lower then branch — collect result value
            let then_lowered = lower_stmt_list_with_tail_value(
                graph,
                &mut then_block,
                &if_expr.then_branch.stmts,
                options,
                ctx,
            )?;
            // Cat 2.1 Slice 2: snapshot then-arm's locals state BEFORE
            // else-arm lowering mutates `ctx.local_value_ids`.  Used
            // only if then-arm is open (will `set_goto` to merge); a
            // closed arm's snapshot is unused.
            let then_exit_snapshot = ctx.snapshot_locals_for_link();

            // Stage B1: restore pre-branch ctx state before lowering
            // the else-arm so its `LOAD_FAST`-style reads see the
            // pre-If bindings, not the then-arm's rebinds.
            pre_branch_ctx.restore(ctx);

            // Lower else branch
            let mut else_lowered = Lowered::no_value();
            if let Some((_, else_branch)) = &if_expr.else_branch {
                else_lowered = lower_expr(graph, &mut else_block, else_branch, options, ctx)?;
            }
            let else_exit_snapshot = ctx.snapshot_locals_for_link();

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
                    graph.block_mut(then_block).framestate = Some(then_exit_snapshot.clone());
                }
                if else_open {
                    graph.set_goto(else_block, merge, vec![else_value.unwrap()]);
                    graph.block_mut(else_block).framestate = Some(else_exit_snapshot.clone());
                }
                (merge, Some(phi_args[0]))
            } else {
                let merge = graph.create_block();
                if then_open {
                    graph.set_goto(then_block, merge, vec![]);
                    graph.block_mut(then_block).framestate = Some(then_exit_snapshot.clone());
                }
                if else_open {
                    graph.set_goto(else_block, merge, vec![]);
                    graph.block_mut(else_block).framestate = Some(else_exit_snapshot.clone());
                }
                (merge, None)
            };

            // FrameState::union-driven merge when both arms reach the
            // merge block.  Routes through `FrameState::union` for
            // explicit per-slot classification per RPython
            // `flowspace/framestate.py:105-128 union`:
            //   - One-sided None → None-kill (`framestate.py:110-111`):
            //     the slot is dropped from `ctx` so post-merge reads
            //     of that name surface as undefined-local.
            //   - CarryThrough (same vid both arms): kept; the merged
            //     entry's `value_type` may have widened from `Unknown`
            //     to a concrete kind via the wildcard rule and the
            //     source `OpKind`'s `ty` is retagged below to keep
            //     `graph_value_type` in agreement with the framestate.
            //   - NeedsPhi (disagreeing vids): eager phi install at
            //     union time per `framestate.py:113-114 union`'s
            //     fresh `Variable()` semantics —
            //     `lazy_install_local_at_current_block` allocates the
            //     merge-block inputarg, threads per-arm vids onto
            //     each predecessor's goto args, and rebinds ctx so
            //     post-merge reads of the name resolve to the new
            //     phi vid without re-driving the lazy installer.
            //   - Concrete-vs-concrete kind disagreement returns
            //     `Err(UnionError::TypeMismatch)` from
            //     `FrameState::union` — a whole-state failure mirroring
            //     PyPy `framestate.py:88 try/except UnionError: return
            //     None` (see the `model::FrameState::union` doc comment
            //     for the parity rationale).  rustc's type checker
            //     rejects any valid Rust source that binds the same
            //     local to two different concrete kinds across `if` /
            //     `else` arms, so the `Err` arm is unreachable from
            //     real input; `.expect(...)` documents the contract
            //     and panics loudly if a hand-built fixture or a
            //     frontend change ever breaks it.
            if then_open && else_open {
                let merged = then_exit_snapshot.union(&else_exit_snapshot, graph).expect(
                    "rustc must have rejected differing concrete kinds across if/else arms",
                );
                // Carry-through Unknown→concrete widening at the
                // union (`model::FrameState::union` wildcard
                // rule) leaves the merged `value_type` concrete while
                // the shared source `OpKind`'s `ty` may still be
                // `Unknown` — `local_value_types` snapshots can
                // disagree across arms even for the same vid (one
                // arm bound the slot with an annotation, the other
                // inferred-only).  Retag the source op now so
                // `graph_value_type(vid)` agrees with the merged
                // framestate at every consumer (mirrors the
                // `Expr::Try` widening pattern used after `?`
                // unwraps a `Result<T, E>`).  Fresh-phi slots
                // (vid disagrees with then-arm's predecessor) are
                // handled by the lazy installer's own retag pass at
                // first post-merge read; they intentionally do not
                // mutate the predecessor source here.
                //
                // Detection: a slot's vid was carry-through iff it
                // matches the then-arm predecessor's vid for the
                // same slot (`union` keeps the predecessor's
                // vid on agreement, allocates a fresh one on
                // disagreement).  Detection is symmetric — checking
                // either predecessor works because agreement implies
                // both predecessors share the vid.
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    let Some(slot) = slot.as_ref() else {
                        continue;
                    };
                    let then_vid = then_exit_snapshot
                        .entries
                        .get(slot_idx)
                        .and_then(|e| e.as_ref())
                        .map(|e| e.value_id);
                    let is_carry_through = then_vid == Some(slot.value_id);
                    if is_carry_through
                        && !matches!(slot.value_type, ValueType::Unknown)
                        && !matches!(
                            graph_value_type(graph, slot.value_id),
                            Some(ref t) if t == &slot.value_type,
                        )
                    {
                        retag_result_value_type(graph, slot.value_id, slot.value_type.clone());
                    }
                }
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    if slot.is_some() {
                        continue;
                    }
                    // None-kill: name present on at most one side —
                    // drop from ctx so post-merge reads do not return
                    // a stale binding.  `local_first_bind_order` /
                    // `local_first_bind_seen` are graph-wide
                    // append-only and stay untouched (the slot index
                    // remains valid; future snapshots push `None` for
                    // it).
                    let name = then_exit_snapshot
                        .entries
                        .get(slot_idx)
                        .and_then(|e| e.as_ref())
                        .or_else(|| {
                            else_exit_snapshot
                                .entries
                                .get(slot_idx)
                                .and_then(|e| e.as_ref())
                        })
                        .map(|e| e.name.clone());
                    if let Some(name) = name {
                        ctx.local_value_ids.remove(&name);
                        ctx.local_value_types.remove(&name);
                    }
                }

                // Eager phi install for every fresh-phi slot.
                // `flowspace/framestate.py:113-114 union` returns a
                // fresh `Variable()` whenever per-slot vids
                // disagree; pyre's `union` already allocated
                // the fresh ValueId, so we drive the lazy installer
                // with `Some(slot.value_id)` to emit the Input op +
                // inputarg + predecessor link args using the same
                // vid the merged state already refers to.
                // Carry-through slots agreed on a single vid so no
                // fresh phi is emitted (`framestate.py:108-109 if w1
                // == w2: return w1`); first post-merge read of those
                // names still drives the lazy installer's same-block
                // reuse path.
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    let Some(slot) = slot.as_ref() else {
                        continue;
                    };
                    let then_vid = then_exit_snapshot
                        .entries
                        .get(slot_idx)
                        .and_then(|e| e.as_ref())
                        .map(|e| e.value_id);
                    let is_fresh_phi = then_vid != Some(slot.value_id);
                    if is_fresh_phi {
                        let _ = lazy_install_local_at_current_block(
                            graph,
                            ctx,
                            merge_block,
                            &slot.name,
                            Some(slot.value_id),
                        );
                    }
                }
            }

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
                // RPython lowers `True`/`False` to `Constant(True/False)`
                // of `lltype.Bool` (annotator/model.py:227 SomeBool).  At
                // the codewriter level `getkind(Bool)` returns `'int'`
                // (`rpython/jit/codewriter/flatten.py:getkind`) so the
                // value lives in an int register, but the annotator-side
                // distinction (SomeBool vs SomeInteger) is preserved by
                // emitting the dedicated `OpKind::ConstBool` variant.
                syn::Lit::Bool(b) => {
                    return Ok(Lowered {
                        value: graph.push_op(*block, OpKind::ConstBool(b.value), true),
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
            // RPython `flowspace/flowcontext.py:835 LOAD_FAST`: the
            // bytecode reads the existing local-stack entry rather
            // than introducing a new `Variable`.  Pyre's analogue:
            // when a single-segment path names a local whose
            // definition lives in the *same* block as the read,
            // forward the bound `ValueId` directly so downstream
            // passes see a single SSA definition with multiple uses
            // — matching upstream's frame-locals model.
            if path.path.segments.len() == 1
                && path.qself.is_none()
                && let Some(&(vid, defining_block)) = ctx.local_value_ids.get(&name)
                && defining_block == *block
            {
                return Ok(Lowered {
                    value: Some(vid),
                    path_closed: false,
                });
            }
            // Cat 2.1 Slice 2 / Stage A1: cross-block read of a
            // single-segment local — try lazy install first (allocates
            // an inputarg in `*block` + threads `Link.args` back to
            // every predecessor whose closing site recorded a snapshot
            // in `Block.framestate`).  Falls back to the legacy naked
            // `OpKind::Input` emit when any predecessor lacks a
            // recorded snapshot, so non-if/else closing sites that
            // have not yet opted into Slice 2-6 wiring still build a
            // graph (just one whose cross-block reads remain pinned
            // by the cutover anchor for that flavour).
            if path.path.segments.len() == 1
                && path.qself.is_none()
                && ctx
                    .local_value_ids
                    .get(&name)
                    .is_some_and(|&(_, defining_block)| defining_block != *block)
                && let Some(threaded_vid) =
                    lazy_install_local_at_current_block(graph, ctx, *block, &name, None)
            {
                return Ok(Lowered {
                    value: Some(threaded_vid),
                    path_closed: false,
                });
            }
            let ty = ctx
                .local_value_types
                .get(&name)
                .cloned()
                .unwrap_or(ValueType::Unknown);
            let value = graph.push_op(
                *block,
                OpKind::Input {
                    name: name.clone(),
                    ty: ty.clone(),
                },
                true,
            );
            // RPython `LOAD_FAST` parity (`flowspace/flowcontext.py:835`):
            // once the local-stack slot is read into a Variable in the
            // current block, subsequent reads of the same name in the
            // same block must return the same Variable — RPython's
            // bytecode reads the slot, not a fresh copy.  Pyre's
            // single-segment `Expr::Path` reaches this fallback only
            // when the same-block reuse and cross-block lazy install
            // both decline (no recorded predecessor framestate, or
            // graph-recoverable kind disagreement at
            // `lazy_install_local_at_current_block:1751-1757`); we
            // register the freshly-emitted `Input` result as the
            // authoritative `(ValueId, current_block)` so further reads
            // of `name` within the same block dedup against this
            // synthetic Input.
            //
            // `LocalBindingSnapshot` saves and restores
            // `ctx.local_value_ids` across `If` / `Match` / `Loop` /
            // `While` / `ForLoop` boundaries, so the cached `(vid,
            // block)` does not leak into a sibling control-flow arm.
            if let Some(vid) = value {
                if path.path.segments.len() == 1 && path.qself.is_none() {
                    ctx.bind_local_id(name.clone(), vid, *block);
                }
            }
            Ok(Lowered {
                value,
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

        // ── unary *x, !x, -x ──
        syn::Expr::Unary(u) => {
            // `*x` (Rust deref) has no flowspace counterpart —
            // `flowspace/operation.py:465-474` registers only `pos` /
            // `neg` / `invert` / `bool` as unary ops.  The
            // RPython-parity `build_flow.rs::lower_unary`
            // (`flowspace/rust_source/build_flow.rs:3301`) treats
            // `UnOp::Deref` as `lower_expr(b, &u.expr)` pass-through:
            // the annotator tracks identity + type regardless of
            // borrow form, so emitting an aliasing op here is
            // redundant.  Pyre's codewriter independently aliases
            // `OpKind::UnaryOp { op: "deref", .. }` to its operand
            // at `jit_codewriter/jtransform.rs:711` (same arm as
            // `same_as`), confirming no semantic load is lost.
            // Pass-through here lets the rtyper-side adapter
            // (`translator/rtyper/flowspace_adapter.rs:359`) skip
            // the `deref` Skip category — the production graph
            // never carries `deref` ops past this point.  The
            // fail-loud invariant at adapter level remains: any
            // synthetic graph that injects `OpKind::UnaryOp {
            // op: "deref", .. }` directly still surfaces a
            // `TyperError` (anchor test
            // `cutover.rs:anchor_unary_deref_surfaces_failloud_no_flowspace_peer`).
            if matches!(u.op, syn::UnOp::Deref(_)) {
                return lower_expr(graph, block, &u.expr, options, ctx);
            }
            // ── Rust `!x` lowering — RPython has TWO opcodes; pyre
            //    folds them at this single site.
            //    TODO(unary-not-split): when the front-end gains
            //    receiver-typed dispatch, split this back into the
            //    UNARY_NOT (`flowcontext.py:531-538`) vs UNARY_INVERT
            //    (`flowcontext.py:188-191`) shape so each surfaces
            //    its own opname.
            //
            // Upstream RPython distinguishes two unary-not operators:
            //
            //   * `UNARY_NOT`   (`flowcontext.py:531-538`) — *logical* not
            //     on booleans / truthy values.  Lowered as `op.bool(w_value)`
            //     followed by `guessbool` + constant-tail join.
            //
            //   * `UNARY_INVERT` (`flowcontext.py:188-191`) — *bitwise* not
            //     on integers.  Lowered as `op.invert(w_value)`, registered
            //     at `operation.py:474 add_operator('invert', 1, ..)` /
            //     `lloperation.py int_invert`.
            //
            // Rust's `!` is overloaded by the `std::ops::Not` trait:
            // `!bool` → logical not, `!i64` (and other integer types) →
            // *bitwise* not.  The frontend must classify the operand
            // before lowering: bool goes through the `UNARY_NOT` shape,
            // int goes through `UNARY_INVERT`.  Unknown operands
            // fail-loud (`stop_unsupported`) since RPython's
            // `flowcontext.py:194,535-538` dispatches strictly at the
            // bytecode token; guessing would collapse two distinct
            // RPython bytecodes.
            //
            // The `UNARY_NOT` branch desugars via `bool(x)` + branch +
            // constant tail, mirroring `flowcontext.py:531-538`:
            //
            //     w_value = self.popvalue()
            //     w_bool  = op.bool(w_value).eval(self)
            //     self.pushvalue(const(not self.guessbool(w_bool)))
            //
            // Twin of `flowspace/rust_source/build_flow.rs:1337
            // lower_unary_not`.  Both arms Link straight to the join with
            // a constant tail — no separate evaluation block (the simpler
            // twin of `&&`/`||`'s `lower_short_circuit`).
            //
            if matches!(u.op, syn::UnOp::Not(_)) {
                // ── Statically-obvious int operand: emit `invert` op ──
                // RPython `flowcontext.py:188-191 UNARY_INVERT` dispatches
                // to `op.invert(w_arg)` (`operation.py:474
                // add_operator('invert', 1, ..)`).  Pyre's rtyper has the
                // matching `"invert"` arm in `RPythonTyper::translate_op`
                // routing to `Repr::rtype_invert`.  Lowering an
                // int-literal `!` directly to `OpKind::UnaryOp { op:
                // "invert", .. }` skips the bool-branch detour entirely
                // — the result is the bitwise complement, matching
                // `~lit` in Python.
                // ── Dynamic operand-type dispatch ──
                // Rust's `!` is overloaded via `std::ops::Not`:
                // `!T where T: Not` is bitwise complement for integer
                // types and logical negation for `bool`.  Mirror PyPy's
                // bytecode-level distinction (`UNARY_INVERT` vs
                // `UNARY_NOT`) by inspecting the operand's static type
                // through `local_type_strings`/`local_value_types`
                // tracking populated at let-binding / fn-parameter
                // time.
                //
                // Statically detected as int → emit `invert` op
                // (UNARY_INVERT). Statically detected as bool → fall
                // through to the UNARY_NOT bool+branch desugar.
                // Unknown fail-louds via `stop_unsupported`.
                match expr_unary_not_operand_kind(&u.expr, ctx) {
                    UnaryNotOperandKind::Int => {
                        let operand = get_value!(lower_expr(graph, block, &u.expr, options, ctx)?);
                        // The classifier returns `Int` for both
                        // primitive integer kinds (lowered as
                        // `ValueType::Int`) and arbitrary-precision
                        // integers like `BigInt` (lowered as
                        // `ValueType::Ref`). RPython's
                        // `IntegerRepr.rtype_invert` /
                        // `LongRepr.rtype_invert` dispatch on the
                        // operand's lattice node; pyre projects that
                        // through `graph_value_type(operand)` so the
                        // emitted `OpKind::UnaryOp.result_ty` matches
                        // the operand's actual lowered shape and the
                        // function's declared return type
                        // (`bigint_invert(a: BigInt) -> BigInt` →
                        // `Ref`).
                        let result_ty = graph_value_type(graph, operand)
                            .filter(|ty| matches!(ty, ValueType::Int | ValueType::Ref))
                            .unwrap_or(ValueType::Int);
                        return Ok(Lowered {
                            value: graph.push_op(
                                *block,
                                OpKind::UnaryOp {
                                    op: "invert".into(),
                                    operand,
                                    result_ty,
                                },
                                true,
                            ),
                            path_closed: false,
                        });
                    }
                    UnaryNotOperandKind::Bool => {}
                    UnaryNotOperandKind::Unknown => {
                        // RPython `flowcontext.py:194,535-538`
                        // dispatches `UNARY_NOT` vs `UNARY_INVERT`
                        // strictly at the Python bytecode token; pyre
                        // mirrors that contract by fail-louding when
                        // the operand kind cannot be recovered.
                        // `build_flow.rs:4404-4416` is fail-loud on the
                        // same shape. The classifier
                        // (`expr_unary_not_operand_kind`,
                        // `front/ast.rs:5582`) handles the production
                        // patterns surfaced in
                        // `pyre-{object,interpreter,jit}/src/` plus
                        // `majit-ir/src/resoperation.rs`.
                        return stop_unsupported(
                            graph,
                            *block,
                            UnsupportedExprKind::UnaryNotUnknownOperand,
                        );
                    }
                }
                let operand = get_value!(lower_expr(graph, block, &u.expr, options, ctx)?);
                let cond = graph
                    .push_op(
                        *block,
                        OpKind::UnaryOp {
                            op: "bool".into(),
                            operand,
                            result_ty: ValueType::Bool,
                        },
                        true,
                    )
                    .expect("UnaryOp { op: \"bool\", .. } produces a value");
                // RPython pushes `Constant(not python_bool)` per arm of
                // `lltype.Bool` — the false arm produces `True`
                // (`!false == true`); the true arm produces `False`
                // (`!true == false`).  Emit as `ConstBool` so the
                // annotator picks `SomeBool` rather than `SomeInteger`.
                let const_true = graph
                    .push_op(*block, OpKind::ConstBool(true), true)
                    .expect("ConstBool produces a value");
                let const_false = graph
                    .push_op(*block, OpKind::ConstBool(false), true)
                    .expect("ConstBool produces a value");
                // ── Locals threading ──
                // RPython `flowcontext.py:531-538 UNARY_NOT` propagates
                // frame locals across the bool-fork via `Link.args` ↔
                // `inputargs`.  Mirror of build_flow.rs:1363
                // lower_unary_not's `[tail, ...locals]` join shape.
                // Both arms' Links carry pre-fork local values plus the
                // constant tail; ctx.local_value_ids rebinds to join
                // inputargs after the merge so post-join reads resolve
                // to the merged values.
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                let pre_fork_local_args: Vec<ValueId> = merged_names
                    .iter()
                    .map(|name| pre_fork_locals[name].0)
                    .collect();

                let (join_block, join_args) = graph.create_block_with_args(merged_names.len() + 1);
                let tail = join_args[0];
                let join_local_args: Vec<ValueId> = join_args[1..].to_vec();

                let mut false_arm_args: Vec<ValueId> = Vec::with_capacity(merged_names.len() + 1);
                false_arm_args.push(const_false);
                false_arm_args.extend(pre_fork_local_args.iter().cloned());
                let mut true_arm_args: Vec<ValueId> = Vec::with_capacity(merged_names.len() + 1);
                true_arm_args.push(const_true);
                true_arm_args.extend(pre_fork_local_args.iter().cloned());

                // Two Links into the same join: cond truthy → tail
                // is `0` (false); cond falsy → tail is `1` (true).
                graph.set_branch(
                    *block,
                    cond,
                    join_block,
                    false_arm_args,
                    join_block,
                    true_arm_args,
                );

                // Rebind locals to join_block's inputargs.  Same
                // pattern as the `&&`/`||` arm above.
                for (name, &arg_vid) in merged_names.iter().zip(join_local_args.iter()) {
                    ctx.local_value_ids
                        .insert(name.clone(), (arg_vid, join_block));
                }

                *block = join_block;
                return Ok(Lowered {
                    value: Some(tail),
                    path_closed: false,
                });
            }
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
            // Short-circuit `&&` / `||` are control flow in RPython
            // (`flowspace/operation.py:475-510` does NOT register
            // short-circuit `and`/`or` as binary operators).  Mirror
            // `flowspace/rust_source/build_flow.rs:1191
            // lower_short_circuit` line-by-line: emit `bool(lhs)` as the
            // exitswitch discriminator, fork the block via
            // `set_branch`, evaluate rhs in a separate block, and merge
            // both arms into a `[tail, ...locals]`-shaped join carrying
            // either `lhs_raw` (short-circuit arm) or `rhs_raw` (full
            // eval) as `tail` plus every pre-fork frame local threaded
            // through (so an `STORE_FAST` inside rhs propagates past the
            // join — `build_flow.rs:1218-1232 / :1281-1292`).  Upstream
            // bytecode basis — `flowcontext.py:766-777
            // JUMP_IF_FALSE_OR_POP` (`&&`) / `JUMP_IF_TRUE_OR_POP`
            // (`||`):
            //
            //     w_value = self.peekvalue()
            //     if not self.guessbool(op.bool(w_value).eval(self)):
            //         return target          # short-circuit on False
            //     self.popvalue()             # else evaluate rhs
            //
            // The surviving value is the original `lhs_raw` (matching
            // `peekvalue()` then `popvalue()`), not `bool(lhs_raw)` —
            // `bool` is only the switch discriminator.  `&&`
            // short-circuits on False (lhs falsy is the result; rhs is
            // dead); `||` short-circuits on True.  Without this
            // desugar, `binary_op_name(&bin.op)` emits the literal
            // `"and"` / `"or"` opname which then trips
            // `flowspace_adapter.rs:422 normalize_binop_name`'s
            // fail-loud arm, blocking Slice 10 of the rtyper cutover.
            if matches!(bin.op, syn::BinOp::And(_) | syn::BinOp::Or(_)) {
                let is_and = matches!(bin.op, syn::BinOp::And(_));

                let lhs_raw = get_value!(lower_expr(graph, block, &bin.left, options, ctx)?);
                let cond = graph
                    .push_op(
                        *block,
                        OpKind::UnaryOp {
                            op: "bool".into(),
                            operand: lhs_raw,
                            result_ty: ValueType::Bool,
                        },
                        true,
                    )
                    .expect("UnaryOp { op: \"bool\", .. } produces a value");

                // ── Locals threading through fork/join ──
                //
                // RPython's frame-locals model (`flowcontext.py:835
                // LOAD_FAST` / `:872-884 STORE_FAST`) propagates locals
                // across every fork via `Link.args` ↔ target
                // `inputargs`.  Pyre's
                // `flowspace/rust_source/build_flow.rs:1191
                // lower_short_circuit` mirrors this with a
                // `[tail, ...merged_names]` join shape — the parallel
                // port done here for `front/ast.rs::Expr::Binary`
                // `&&`/`||`.
                //
                // Pre-fork snapshot of `ctx.local_value_ids` provides
                // the names that must thread through the join; rhs
                // lowers against rhs_block's inputargs so any `STORE_
                // FAST` inside rhs writes to a local already-in-flight
                // through the join.
                //
                // Body-input-retirement-epic.md Phase 2 brings the
                // same pattern to `Expr::If` / `Expr::Match` so all
                // fork/join shapes use the consistent
                // `[result, ...locals]` Link.args contract; the
                // single-fork forms (`!`-bool desugar) follow as a
                // sibling slice.
                let pre_fork_locals = ctx.local_value_ids.clone();
                let mut merged_names: Vec<String> = pre_fork_locals.keys().cloned().collect();
                merged_names.sort();
                let pre_fork_local_args: Vec<ValueId> = merged_names
                    .iter()
                    .map(|name| pre_fork_locals[name].0)
                    .collect();

                let (mut rhs_block, rhs_local_args) =
                    graph.create_block_with_args(merged_names.len());
                let (join_block, join_args) = graph.create_block_with_args(merged_names.len() + 1);
                let tail = join_args[0];
                let join_local_args: Vec<ValueId> = join_args[1..].to_vec();

                // Short-circuit Link.args = [lhs_raw, ...pre_fork_locals];
                // rhs Link.args         = [...pre_fork_locals].
                let mut shortcut_link_args: Vec<ValueId> =
                    Vec::with_capacity(merged_names.len() + 1);
                shortcut_link_args.push(lhs_raw);
                shortcut_link_args.extend(pre_fork_local_args.iter().cloned());
                let rhs_link_args = pre_fork_local_args.clone();

                if is_and {
                    // `&&`: cond truthy → eval rhs; cond falsy →
                    // short-circuit `lhs_raw` straight to the join.
                    graph.set_branch(
                        *block,
                        cond,
                        rhs_block,
                        rhs_link_args,
                        join_block,
                        shortcut_link_args,
                    );
                } else {
                    // `||`: cond truthy → short-circuit `lhs_raw`
                    // straight to the join; cond falsy → eval rhs.
                    graph.set_branch(
                        *block,
                        cond,
                        join_block,
                        shortcut_link_args,
                        rhs_block,
                        rhs_link_args,
                    );
                }

                // Rebind locals to rhs_block's inputargs so rhs
                // lowering sees them via same-block reads.
                for (name, &arg_vid) in merged_names.iter().zip(rhs_local_args.iter()) {
                    ctx.local_value_ids
                        .insert(name.clone(), (arg_vid, rhs_block));
                }

                // Lower rhs in `rhs_block`; if the path remains open,
                // link to the join carrying `rhs_raw` plus current
                // local values (which rhs may have rebound through
                // assigns / nested control flow).  An rhs that closes
                // its path (`return` / `raise` / `break` inside the
                // right operand) leaves only the short-circuit arm
                // reaching the join — same open-arm-only pattern as
                // `lower_if` above.
                let rhs_lowered = lower_expr(graph, &mut rhs_block, &bin.right, options, ctx)?;
                if graph.block(rhs_block).is_open() {
                    // `rhs_lowered.value` is `None` only when rhs was a
                    // statement-form sub-expression with no value;
                    // `lhs_raw` keeps the join arity-correct in that
                    // unusual case (defensive, mirrors `lower_if`'s
                    // arity guard).
                    let rhs_raw = rhs_lowered.value.unwrap_or(lhs_raw);
                    let rhs_exit_local_args: Vec<ValueId> = merged_names
                        .iter()
                        .map(|name| {
                            ctx.local_value_ids
                                .get(name)
                                .map(|&(vid, _)| vid)
                                .unwrap_or_else(|| {
                                    pre_fork_locals
                                        .get(name)
                                        .map(|&(vid, _)| vid)
                                        .expect("local must remain in scope after rhs lower")
                                })
                        })
                        .collect();
                    let mut rhs_to_join_args: Vec<ValueId> =
                        Vec::with_capacity(merged_names.len() + 1);
                    rhs_to_join_args.push(rhs_raw);
                    rhs_to_join_args.extend(rhs_exit_local_args);
                    graph.set_goto(rhs_block, join_block, rhs_to_join_args);
                }

                // Rebind locals to join_block's inputargs so post-join
                // reads of each name resolve to the merged phi value
                // — `(join_inputarg, join_block)` is the same-block
                // tuple `Expr::Path` checks at line 2114 to elide the
                // `OpKind::Input` emit.  Mirror of build_flow.rs:1294-
                // 1300's `b.open_new_block(... join_locals ...)`.
                for (name, &arg_vid) in merged_names.iter().zip(join_local_args.iter()) {
                    ctx.local_value_ids
                        .insert(name.clone(), (arg_vid, join_block));
                }

                *block = join_block;
                return Ok(Lowered {
                    value: Some(tail),
                    path_closed: false,
                });
            }

            let lhs = get_value!(lower_expr(graph, block, &bin.left, options, ctx)?);
            let rhs = get_value!(lower_expr(graph, block, &bin.right, options, ctx)?);
            let op_name = binary_op_name(&bin.op);
            let result_ty = binary_result_value_type(graph, lhs, rhs, op_name);
            let value = graph.push_op(
                *block,
                OpKind::BinOp {
                    op: op_name.into(),
                    lhs,
                    rhs,
                    result_ty,
                },
                true,
            );
            // RPython INPLACE_* + STORE_FAST parity
            // (`flowspace/flowcontext.py:878-885`): compound assignment
            // `x += y` (and -=, *=, /=, %=, &=, |=, ^=, <<=, >>=) push
            // the inplace result and immediately replace the locals
            // slot for `x`, then renames the resulting `Variable` to
            // the local name.  Without the local_value_ids update,
            // the same-block dedup cache (`ast.rs:1389` let arm,
            // `:1724` simple-assign arm) still points at the
            // pre-inplace ValueId, so a later same-block read of
            // `x` returns the stale value.  Without the
            // `graph.name_value` rename, the adapter's
            // `name_to_value` lookup continues to resolve `x` to the
            // pre-inplace Variable.  Simple assignment `x = y` is
            // handled at the Expr::Assign arm above; this branch
            // owns the compound path that lowers as Expr::Binary.
            // The compound BinOp result is always a Variable (not a
            // `ConstInt`/`ConstFloat` define-op), but the
            // `is_constant_define_value` gate is kept symmetrical
            // with the simple-assign arm to mirror upstream's
            // `if isinstance(w_newvalue, Variable):` predicate.
            if op_name.ends_with("_assign") {
                if let (Some(vid), syn::Expr::Path(path)) = (value, &*bin.left) {
                    if path.path.segments.len() == 1 && path.qself.is_none() {
                        let name = path
                            .path
                            .segments
                            .iter()
                            .map(|seg| seg.ident.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        ctx.bind_local_id(name, vid, *block);
                    }
                }
            }
            Ok(Lowered {
                value,
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
            // No silent elision: every `as T` reaches the adapter as a
            // real `OpKind::UnaryOp { op, .. }`.  `cast_op_name` may
            // return `"same_as"` for transparent (category-matching or
            // source-unknown) casts; the cast handler family
            // (`rbuiltin.rs::rtype_same_as` /
            // `rtype_cast_int_to_float` / `rtype_cast_float_to_int` /
            // `rtype_cast_int_to_ptr` / `rtype_cast_bool_to_int` /
            // `rtype_cast_bool_to_float` / `rtype_int_is_true` /
            // `rtype_float_is_true`) is wired in
            // `RPythonTyper::translate_operation` (`rtyper.rs:2122-`)
            // so each opname routes through the rtyper directly; the
            // adapter at `flowspace_adapter.rs::normalize_unary_op_name`
            // passes them through unchanged.
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
            let mut arm_tails: Vec<(BlockId, Option<ValueId>, FrameState)> =
                Vec::with_capacity(m.arms.len());
            for arm in &m.arms {
                let entry = graph.create_block();
                let mut tail = entry;
                let saved_locals = LocalBindingSnapshot::capture(ctx);
                bind_pattern_locals(&arm.pat, scrutinee_type_string.as_deref(), ctx);
                let arm_lowered_result = lower_expr(graph, &mut tail, &arm.body, options, ctx);
                // Slice 4a: snapshot this arm's exit framestate BEFORE
                // `restore` wipes the per-arm rebinds.  The merge
                // block's lazy installer uses
                // `Block.framestate` on each predecessor to thread
                // `Link.args` back through the arm tail (RPython
                // parity: `flowspace/flowcontext.py:407-408
                // record_block(block)` calls `setstate(block.framestate)`
                // and `flowspace/flowcontext.py:449
                // currentstate.getoutputargs(newstate)` reads the
                // predecessor's snapshot to produce link args).
                // Without this, the predecessor walk in
                // `lazy_install_local_at_current_block` finds no
                // snapshot and falls back to the legacy naked-`Input`
                // emit, which loses the per-arm rebind information.
                let arm_exit_snapshot = ctx.snapshot_locals_for_link();
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
                arm_tails.push((tail, arm_lowered.value, arm_exit_snapshot));
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
                .all(|(tail, r, _)| !graph.block(*tail).is_open() || r.is_some());
            let (merge, merge_phi) = if all_open_arms_have_value {
                let (m_block, phi_args) = graph.create_block_with_args(1);
                (m_block, Some(phi_args[0]))
            } else {
                (graph.create_block(), None)
            };

            let mut any_open = false;
            // Collect open-arm exit snapshots before `set_goto` closes
            // each arm, so the post-loop iterative-fold over 2-way
            // `FrameState::union` sees every reaching predecessor's
            // locals state.  Parity port of Expr::If's both-open-arm
            // merge generalised to N arms — see the union block below
            // for the per-slot semantics.
            let mut open_arm_snapshots: Vec<FrameState> = Vec::new();
            for (tail, result, exit_snapshot) in &arm_tails {
                if !graph.block(*tail).is_open() {
                    continue;
                }
                any_open = true;
                open_arm_snapshots.push(exit_snapshot.clone());
                let goto_args = if all_open_arms_have_value {
                    // Safe: the filter above guarantees every open arm's
                    // `result` is `Some`.
                    vec![result.unwrap()]
                } else {
                    Vec::new()
                };
                graph.set_goto(*tail, merge, goto_args);
                // Slice 4a: stamp the arm tail's framestate so the
                // merge block's lazy installer can thread `Link.args`
                // back through this predecessor.  Mirrors the
                // `Expr::If` then/else stamp at the equivalent
                // `set_goto` site.
                graph.block_mut(*tail).framestate = Some(exit_snapshot.clone());
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

            // Iterative-fold-driven merge when 2+ arms reach the
            // merge block.  Per-slot semantics mirror Expr::If's
            // both-open-arm merge, generalised to N arms by left-
            // folding `acc.union(arm)` against each open-arm
            // snapshot — direct port of `flowspace/flowcontext.py:
            // 430-436 mergeblock`'s repeated 2-way union over
            // arriving candidates.
            //
            //   - Carry-through Unknown→concrete retag keeps
            //     `graph_value_type(vid)` in agreement with the
            //     merged framestate when an inferred-only arm is
            //     unioned with an annotated arm.
            //   - None-kill drops slots that any arm left unbound
            //     (`framestate.py:110-111`); post-merge reads of
            //     those names surface as undefined-local.
            //   - Fresh-phi install (per-slot vid disagreement)
            //     allocates the merge-block inputarg, threads
            //     predecessor link args via
            //     `lazy_install_local_at_current_block`, and
            //     rebinds ctx so post-merge reads resolve to the
            //     new phi.
            //
            // Single-open-arm case is intentionally NOT handled
            // here: that's the audit's pre-existing one-open-arm
            // fragility shared with Expr::If, requiring a separate
            // ctx-restore strategy that lives outside this slice.
            if open_arm_snapshots.len() >= 2 {
                // Concrete-vs-concrete kind disagreement returns
                // `Err(UnionError::TypeMismatch)` (whole-state failure
                // mirroring PyPy `framestate.py:88 try/except
                // UnionError: return None`).  rustc's type checker
                // rejects any valid Rust source that binds the same
                // local to two different concrete kinds across match
                // arms, so the `Err` arm is unreachable from real
                // input; `.expect(...)` documents the contract.
                // Iterative left-fold over 2-way `FrameState::union`
                // — direct port of upstream's `flowspace/
                // flowcontext.py:430-436 mergeblock` loop:
                //
                //     for block in candidates:
                //         newstate = block.framestate.union(currentstate)
                //         if newstate is not None:
                //             break
                //
                // Pyre's static AST shape knows every open arm at
                // lowering time, so the fold runs them in order
                // (first arm = initial running state; each
                // subsequent arm = `acc.union(arm)`).  rustc has
                // already rejected source whose arms bind the same
                // local to two different concrete kinds, so the
                // `.expect(...)` documents the contract.
                //
                // TODO(spamblock-recloseblock-port): upstream's
                // mergeblock
                // generalises by creating a fresh `SpamBlock(newstate)`
                // (`flowcontext.py:443`), marking the prior block
                // dead via `block.dead = True` + `block.operations =
                // ()` (`:455-456`), and patching the dead block's
                // exits to forward to the new block via
                // `block.recloseblock(Link(outputargs, newblock))`
                // (`:458-459`).  The post-cleanup CFG (after
                // `simplify.eliminate_empty_blocks`) collapses the
                // dead-block forwarding chain into a single
                // multi-incoming merge block — which is exactly the
                // shape pyre's static AST produces directly.
                //
                // Convergence path: introduce a pendingblock-style
                // worklist at `Expr::Match` lowering, emit one
                // SpamBlock per fold step, mark intermediate blocks
                // dead, and run an `eliminate_empty_blocks` pass
                // before downstream rtyper / flatten / regalloc see
                // the graph.  Multi-session — touches Expr::Match
                // traversal, block lifecycle, and adds a
                // `simplify_graph` pass equivalent.  Until then,
                // pyre produces the post-cleanup CFG directly and
                // the dataflow is parity-correct.
                let mut acc = open_arm_snapshots[0].clone();
                for arm in &open_arm_snapshots[1..] {
                    acc = acc.union(arm, graph).expect(
                        "rustc must have rejected differing concrete kinds across match arms",
                    );
                }
                let merged = acc;
                // Detection of "fresh phi" (vid allocated by `union`
                // because per-arm vids disagreed): the merged slot's
                // vid does NOT match the first arm's vid for the
                // same slot.  When all arms agreed, the fold carried
                // the agreed vid through, so the comparison is
                // False and no fresh phi is emitted.
                let first_arm = &open_arm_snapshots[0];
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    let Some(slot) = slot.as_ref() else {
                        continue;
                    };
                    let first_vid = first_arm
                        .entries
                        .get(slot_idx)
                        .and_then(|e| e.as_ref())
                        .map(|e| e.value_id);
                    let is_carry_through = first_vid == Some(slot.value_id);
                    if is_carry_through
                        && !matches!(slot.value_type, ValueType::Unknown)
                        && !matches!(
                            graph_value_type(graph, slot.value_id),
                            Some(ref t) if t == &slot.value_type,
                        )
                    {
                        retag_result_value_type(graph, slot.value_id, slot.value_type.clone());
                    }
                }
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    if slot.is_some() {
                        continue;
                    }
                    let name = open_arm_snapshots.iter().find_map(|s| {
                        s.entries
                            .get(slot_idx)
                            .and_then(|e| e.as_ref())
                            .map(|e| e.name.clone())
                    });
                    if let Some(name) = name {
                        ctx.local_value_ids.remove(&name);
                        ctx.local_value_types.remove(&name);
                    }
                }
                for (slot_idx, slot) in merged.entries.iter().enumerate() {
                    let Some(slot) = slot.as_ref() else {
                        continue;
                    };
                    let first_vid = first_arm
                        .entries
                        .get(slot_idx)
                        .and_then(|e| e.as_ref())
                        .map(|e| e.value_id);
                    let is_fresh_phi = first_vid != Some(slot.value_id);
                    if is_fresh_phi {
                        let _ = lazy_install_local_at_current_block(
                            graph,
                            ctx,
                            merge,
                            &slot.name,
                            Some(slot.value_id),
                        );
                    }
                }
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

            // Slice 5c.1 eager phi pre-allocation.  Capture the
            // pre-loop local-binding snapshot, close pre-loop's exit
            // to the header, then statically pre-scan the body for
            // its read/rebound names so `allocate_loop_header_phis`
            // can install header phis BEFORE the body walk.  This
            // replaces Slice 5a's lazy back-edge install (which blew
            // up with cycle / arity / kind-mismatch errors), in line
            // with RPython's work-list `mergeblock`+`union` fixpoint
            // semantics adapted for pyre's static AST.
            //
            // Framestate stamps on `pre_loop_block` (after allocator
            // pushes pre-loop link args) and `header_tail` (after
            // cond) keep cross-block lazy installs working for
            // forward (non-back-edge) reads — RPython parity for the
            // stamp shape:
            // `flowspace/flowcontext.py:407-408 record_block(block)`.
            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.snapshot_locals_for_link();
            graph.set_goto(pre_loop_block, header_entry, vec![]);

            let must_merge = loop_body_locals(&w.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

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
            let header_branch_snapshot = ctx.snapshot_locals_for_link();
            graph.set_branch(header_tail, cond, body_entry, vec![], exit, vec![]);
            graph.block_mut(header_tail).framestate = Some(header_branch_snapshot);

            // Body → back to header_entry (entry, not tail —
            // header_entry is the back-edge target).  Each stmt may
            // close its path (inner `return` / `break` / `panic!`); on
            // closure we stop walking the body and the back-edge is
            // skipped via the `is_open` check below.  The loop frame
            // makes `break` / `continue` in the body route to exit /
            // header.  Cat 2-2 Phase B α.1: header-phi name list is
            // computed on-demand at each close site (back-edge below,
            // `Expr::Continue`) via `header_phi_name_list(graph, ...)`
            // so any lazy-install addition to `header_entry.inputargs`
            // during the body walk is automatically threaded.
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
                // Each phi name's current ctx binding — body's new
                // write on rebind, or a body-side inputarg installed
                // by the lazy cross-block reader on first read, or the
                // header phi itself for read-only names — supplies the
                // back-edge arg.  RPython parity:
                // `flowspace/framestate.py:92 getoutputargs` produces
                // the same slot-by-slot mapping for the closing
                // predecessor link.
                let body_tail_snapshot = ctx.snapshot_locals_for_link();
                let header_phi_names = header_phi_name_list(graph, header_entry);
                let back_edge_args = link_args_from_ctx(ctx, &header_phi_names);
                graph.set_goto(body_tail, header_entry, back_edge_args);
                // Audit Cat 2-1: stamp the body-tail's framestate
                // so the post-loop lazy installer can thread reads
                // of pre-loop locals through this back-edge.  Same
                // role as the break / continue stamps; the cycle
                // hazard cited by the prior cycle-breaker comment
                // is now closed structurally by the lazy installer's
                // push-first refactor (`lazy_install_local_at_current_block`
                // installs the outer phi's inputarg slot before
                // recursing into predecessors, so a back-edge
                // recursion finds it via the same-block graph-state
                // idempotency check).
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
            }

            *block = exit;
            Ok(Lowered::no_value())
        }
        syn::Expr::Loop(l) => {
            // Slice 5c.2 eager phi pre-allocation.  `Expr::Loop` has
            // no cond block — `body_entry` IS the loop head and the
            // `continue` target.  Otherwise the shape mirrors
            // `Expr::While` (5c.1): pre-loop snapshot capture, body
            // pre-scan, eager phi install at the head before the body
            // walk, frozen header_phi_names captured for the LoopFrame
            // and back-edge close.  RPython parity: same as the
            // `Expr::While` justification above.
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.snapshot_locals_for_link();
            graph.set_goto(pre_loop_block, body_entry, vec![]);

            let must_merge = loop_body_locals(&l.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                body_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

            // Cat 2-2 Phase B α.1: header-phi name list recomputed on
            // demand at the back-edge close below, so any lazy install
            // that adds an inputarg to `body_entry` during the body
            // walk is automatically threaded.
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
                // Audit Cat 2-1: stamp body-tail's framestate (see
                // the matching `Expr::While` body-tail stamp for the
                // cycle-safety rationale).
                let body_tail_snapshot = ctx.snapshot_locals_for_link();
                let header_phi_names = header_phi_name_list(graph, body_entry);
                let back_edge_args = link_args_from_ctx(ctx, &header_phi_names);
                graph.set_goto(body_tail, body_entry, back_edge_args);
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
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
            //
            // Slice 5c.3 eager phi pre-allocation: applied identically
            // to `Expr::Loop` / `Expr::While`.  The iterable is
            // single-evaluation (RPython `flowcontext.py:1378
            // GET_ITER` evaluates it once before the loop), so its
            // result vid is bound in `pre_loop_block` and reads of
            // it inside the header are forward edges covered by lazy
            // install — no special-casing needed.
            let iterable = get_value!(lower_expr(graph, block, &f.expr, options, ctx)?);
            let _ = iterable;

            let header_entry = graph.create_block();
            let body_entry = graph.create_block();
            let exit = graph.create_block();

            let pre_loop_block = *block;
            let pre_loop_snapshot = ctx.snapshot_locals_for_link();
            graph.set_goto(pre_loop_block, header_entry, vec![]);

            let must_merge = loop_body_locals(&f.body);
            let _ = allocate_loop_header_phis(
                graph,
                ctx,
                pre_loop_block,
                header_entry,
                &pre_loop_snapshot,
                &must_merge,
            );
            graph.block_mut(pre_loop_block).framestate = Some(pre_loop_snapshot);

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
            // Stamp header_entry.framestate before the cond-branch
            // close — mirrors `Expr::While`.  Reads inside the body
            // or post-loop exit recurse back to the header and find
            // its exit-time snapshot (which already includes the
            // eager phis bound to ctx).
            let header_branch_snapshot = ctx.snapshot_locals_for_link();
            if let Some(cond) = for_cond {
                graph.set_branch(header_entry, cond, body_entry, vec![], exit, vec![]);
            } else {
                graph.set_goto(header_entry, body_entry, vec![]);
            }
            graph.block_mut(header_entry).framestate = Some(header_branch_snapshot);

            // Cat 2-2 Phase B α.1: header-phi name list recomputed on
            // demand at the back-edge close below.
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
                // Audit Cat 2-1: stamp body-tail's framestate (see
                // the matching `Expr::While` body-tail stamp for the
                // cycle-safety rationale).
                let body_tail_snapshot = ctx.snapshot_locals_for_link();
                let header_phi_names = header_phi_name_list(graph, header_entry);
                let back_edge_args = link_args_from_ctx(ctx, &header_phi_names);
                graph.set_goto(body_tail, header_entry, back_edge_args);
                graph.block_mut(body_tail).framestate = Some(body_tail_snapshot);
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
            if let Some(frame) = ctx.loop_stack.last().cloned() {
                if graph.block(*block).is_open() {
                    // `break` jumps to the loop's exit block, which has
                    // no eager-allocated phi inputargs (post-loop reads
                    // lazy-install at the exit consumer side).  Empty
                    // args therefore matches `exit.inputargs` arity at
                    // close time; any later-installed exit inputarg is
                    // accompanied by a corresponding lazy-installed
                    // arg push onto this exit's link by the lazy
                    // installer's predecessor walk.
                    let pre_break_snapshot = ctx.snapshot_locals_for_link();
                    graph.set_goto(*block, frame.break_target, vec![]);
                    // Stamp the break source's framestate so the
                    // post-loop lazy installer can read the locals
                    // visible on this predecessor edge — same role as
                    // the then/else arm-tail and loop body-tail
                    // stamps.  RPython parity:
                    // `flowspace/flowcontext.py:399-465 mergeblock`
                    // requires every closing predecessor to carry a
                    // FrameState so `getoutputargs` can resolve the
                    // target's inputargs slot-by-slot.
                    graph.block_mut(*block).framestate = Some(pre_break_snapshot);
                }
            }
            Ok(Lowered::path_closed())
        }
        syn::Expr::Continue(_) => {
            if let Some(frame) = ctx.loop_stack.last().cloned() {
                if graph.block(*block).is_open() {
                    // `continue` jumps to the loop's continue_target —
                    // for `while` / `loop` (Slice 5c.1+) this is the
                    // header with its eager-phi inputargs.  Thread
                    // per-name args from `ctx.local_value_ids[name]`
                    // using the header's CURRENT inputarg name list
                    // (Cat 2-2 Phase B α.1: recomputed on demand from
                    // `frame.continue_target.inputargs` at close time
                    // so any lazy install that added an inputarg
                    // during body walk is automatically threaded).
                    // RPython parity for the slot-by-slot mapping:
                    // `flowspace/framestate.py:92 getoutputargs`.
                    //
                    // Audit Cat 2-1: stamp the continue source's
                    // framestate so the post-loop lazy installer can
                    // thread reads of pre-loop locals (NOT in
                    // `must_merge`) back through this back-edge —
                    // same role as the break source's framestate
                    // stamp at line 3872.  RPython parity:
                    // `flowspace/flowcontext.py:399-465 mergeblock`
                    // requires every closing predecessor of the
                    // merge target to carry a FrameState so
                    // `getoutputargs` can resolve the target's
                    // inputargs slot-by-slot.  The earlier "stamping
                    // would overflow the stack" cycle hazard is now
                    // closed structurally by the lazy installer's
                    // Phase-2 push-first refactor: the back-edge's
                    // snap_vid for a header-phi name points at the
                    // header's already-installed phi inputarg, so
                    // the recursive install at the header short-
                    // circuits via the same-block graph-state
                    // idempotency check on `block.inputargs`.
                    let pre_continue_snapshot = ctx.snapshot_locals_for_link();
                    let header_phi_names = header_phi_name_list(graph, frame.continue_target);
                    let args = link_args_from_ctx(ctx, &header_phi_names);
                    graph.set_goto(*block, frame.continue_target, args);
                    graph.block_mut(*block).framestate = Some(pre_continue_snapshot);
                }
            }
            Ok(Lowered::path_closed())
        }

        // ── closure ──
        //
        // TODO(closure-body-compilation): the closure body is NOT
        // walked here.  `MAKE_FUNCTION`
        // (`pypy/interpreter/pyopcode.py:1144`,
        // `flowspace/flowcontext.py:1177`) materialises a *separate*
        // graph for the `def`/`lambda` body and pushes a fresh
        // function value onto the stack — the body never inlines into
        // the enclosing flow.  Pyre currently lowers the whole
        // expression to a single `Unknown` placeholder for the
        // closure *value* and leaves the body uncompiled.  An earlier
        // attempt to walk the body in-place was a NEW-DEVIATION (it
        // treated the closure as a synchronous block, which broke
        // callers that pass the closure itself as a function-typed
        // argument — e.g. `|_| {}` produced no value for
        // `get_value!`).
        //
        // The full port needs three pieces:
        //   1. Synthesise a fresh `FunctionGraph` per closure body
        //      (parameters from `closure.inputs`, captures plumbed
        //      through inputargs as upvars).
        //   2. Register it with the surrounding `PyreCallRegistry`
        //      under a synthetic
        //      `CallPath::Closure(<host-fn>::<call-site-index>)` so
        //      `simple_call` resolution and `target_to_path` can find
        //      it.
        //   3. Replace this `Unknown` emit with an `OpKind` that
        //      pushes the synthetic graph's host identity onto the
        //      value stack — mirroring `MAKE_FUNCTION`
        //      (`flowspace/flowcontext.py:1177`).
        //
        // Downstream call sites
        // (`expression_type_string` closure-passthrough at line 7953)
        // already project the closure body's return type for the
        // common method-call patterns (`map` / `unwrap_or_else` /
        // `and_then` / `filter`), so the *call-site* analysis is
        // type-coherent even without the synthetic-graph port; the
        // *body* itself stays uncompiled.  Multi-session epic
        // (captures, multi-shot calls, indirect-call dispatch).
        syn::Expr::Closure(_) => Ok(continue_with_unknown(
            graph,
            *block,
            UnsupportedExprKind::Closure,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnaryNotOperandKind {
    Bool,
    Int,
    Unknown,
}

/// Classify Rust's overloaded `!` operand into the RPython opcode it
/// can be lowered to. RPython/PyPy has distinct bytecode shapes:
/// `UNARY_NOT` lowers through `op.bool` plus a branch, while
/// `UNARY_INVERT` lowers directly to `op.invert`.  `Unknown` triggers
/// `stop_unsupported(UnaryNotUnknownOperand)` at the call site
/// (`lower_expr`'s `Expr::Unary UnOp::Not` arm, ~line 3737), mirroring
/// RPython `flowcontext.py:194,535-538` and `build_flow.rs:4404-4416`
/// which fail-loud on the same shape; the classifier extensions below
/// ensure every production-source operand resolves to `Bool` or `Int`.
///
/// TODO(receiver-typed-dispatch): RPython
/// `bookkeeper.py:353-409 getdesc(value)` keys on Python object
/// identity — same-named methods on different types cannot alias
/// because each host callable is a distinct PyObject.  Pyre's
/// classifier operates on the AST surface (no callable-identity
/// resolution available pre-rtyper), so it falls back to
/// `local_type_strings` / `fn_return_types` lookup, last-segment
/// fallback for multi-segment `Path`s, and a receiver-independent
/// method shortlist for known-shape predicates (`is_empty`,
/// `is_constant`, ...).  Two methods of the same name returning
/// different types would collide here even if they belong to
/// different types — the classifier is structurally not 1:1 with
/// `getdesc`.  Retire when pyre's surface DSL learns to emit a
/// callable-resolved Path (via the M2.c `classdef_impl_types`
/// registry, `description.py:407-519` parity); the classifier
/// can then route through the resolved callable and the shortlist
/// retires.  Until then, name-shortlist hits that diverge
/// from upstream behaviour surface as dual-gate `Skip` (typed-Ref
/// classdef-less SomeInstance) or production divergence panic.
fn expr_unary_not_operand_kind(expr: &syn::Expr, ctx: &GraphBuildContext) -> UnaryNotOperandKind {
    match expr {
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Bool(_),
            ..
        }) => UnaryNotOperandKind::Bool,
        syn::Expr::Lit(syn::ExprLit {
            lit: syn::Lit::Int(_) | syn::Lit::Byte(_) | syn::Lit::Char(_),
            ..
        }) => UnaryNotOperandKind::Int,
        // Multi-segment Path expressions — `Self::CONST_BIT`,
        // `Type::CONST`, `Module::CONST`. Pyre's walker registers
        // `Item::Impl::Const` under both `Type::CONST` and bare `CONST`
        // (front/ast.rs Item::Impl arm), so the last-segment fallback
        // resolves `Self::CONST_BIT` references via the bare alias.
        // TODO(receiver-typed-dispatch): replace `fn_return_types`'s
        // string map with a host-identity-keyed `Bookkeeper.descs`
        // populated by the walker so the last-segment lookup at line
        // 5602 collapses into a single qualified lookup
        // (`bookkeeper.py:353` parity).  Until then the bare-name
        // fallback can alias when same-named symbols live on
        // different scopes/types.
        syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() >= 2 => {
            let n = path.path.segments.len();
            let last = path.path.segments[n - 1].ident.to_string();
            if let Some(ty) = ctx.fn_return_types.get(&last) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            let bare_impl = format!(
                "{}::{}",
                path.path.segments[n - 2].ident,
                path.path.segments[n - 1].ident
            );
            if let Some(ty) = ctx.fn_return_types.get(&bare_impl) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Path(path)
            if path.qself.is_none()
                && path.path.leading_colon.is_none()
                && path.path.segments.len() == 1 =>
        {
            let segment = &path.path.segments[0];
            if !matches!(segment.arguments, syn::PathArguments::None) {
                return UnaryNotOperandKind::Unknown;
            }
            let name = segment.ident.to_string();
            if let Some(ty) = ctx.local_type_strings.get(&name) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            if let Some(kind) = ctx
                .local_value_types
                .get(&name)
                .map(value_type_to_unary_not_kind)
                && kind != UnaryNotOperandKind::Unknown
            {
                return kind;
            }
            // Module-level `pub const FOO: bool = ...` — the walker
            // registers `Item::Const` typed names into
            // `fn_return_types` (front/ast.rs:535+ `Item::Const`
            // arm; key namespace is shared but Rust convention
            // (SCREAMING_SNAKE_CASE consts vs snake_case fns) keeps
            // the lookups disjoint in pyre source).  RPython resolves
            // these via the bookkeeper's PBC table at LOAD_GLOBAL
            // time (`bookkeeper.py:329-340 immutablevalue`).
            if let Some(ty) = ctx.fn_return_types.get(&name) {
                let kind = type_string_to_unary_not_kind(ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Binary(bin) => expr_binary_unary_not_operand_kind(ctx, bin),
        syn::Expr::Unary(unary) => match unary.op {
            syn::UnOp::Not(_) => expr_unary_not_operand_kind(&unary.expr, ctx),
            syn::UnOp::Neg(_) => {
                if expr_unary_not_operand_kind(&unary.expr, ctx) == UnaryNotOperandKind::Int {
                    UnaryNotOperandKind::Int
                } else {
                    UnaryNotOperandKind::Unknown
                }
            }
            syn::UnOp::Deref(_) => UnaryNotOperandKind::Unknown,
            _ => UnaryNotOperandKind::Unknown,
        },
        syn::Expr::Paren(paren) => expr_unary_not_operand_kind(&paren.expr, ctx),
        syn::Expr::Group(group) => expr_unary_not_operand_kind(&group.expr, ctx),
        // `unsafe { expr }` — Rust's unsafe block is a transparent
        // wrapper for analyser purposes; RPython has no syntactic
        // analogue so the inner expression's classification is what
        // matters.  Mirror of the Paren/Group arms.
        syn::Expr::Unsafe(u) => {
            if let Some(syn::Stmt::Expr(tail, None)) = u.block.stmts.last() {
                expr_unary_not_operand_kind(tail, ctx)
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        syn::Expr::Block(b) => {
            if let Some(syn::Stmt::Expr(tail, None)) = b.block.stmts.last() {
                expr_unary_not_operand_kind(tail, ctx)
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        // `if cond { a } else { b }` — RPython `flowcontext.py:413`
        // joins the two arms at the merge block; if both arms project
        // to the same `SomeBool` / `SomeInteger`, the join carries
        // that lattice node forward.  Mirror by classifying both arm
        // tails and accepting only when they agree.  Missing else (or
        // non-Block arms) collapses to Unknown.
        syn::Expr::If(if_expr) => {
            let then_kind =
                if let Some(syn::Stmt::Expr(tail, None)) = if_expr.then_branch.stmts.last() {
                    expr_unary_not_operand_kind(tail, ctx)
                } else {
                    UnaryNotOperandKind::Unknown
                };
            if then_kind == UnaryNotOperandKind::Unknown {
                return UnaryNotOperandKind::Unknown;
            }
            match &if_expr.else_branch {
                Some((_, else_expr)) => {
                    let else_kind = expr_unary_not_operand_kind(else_expr, ctx);
                    if else_kind == then_kind {
                        then_kind
                    } else {
                        UnaryNotOperandKind::Unknown
                    }
                }
                None => UnaryNotOperandKind::Unknown,
            }
        }
        // `match e { arm => ..., ... }` — `flowcontext.py:413` join-
        // shape parity, but with N arms.  Accept only when every arm
        // projects to the same kind.
        syn::Expr::Match(match_expr) => {
            let mut acc: Option<UnaryNotOperandKind> = None;
            for arm in &match_expr.arms {
                let kind = expr_unary_not_operand_kind(&arm.body, ctx);
                if kind == UnaryNotOperandKind::Unknown {
                    return UnaryNotOperandKind::Unknown;
                }
                match acc {
                    None => acc = Some(kind),
                    Some(prev) if prev == kind => {}
                    _ => return UnaryNotOperandKind::Unknown,
                }
            }
            acc.unwrap_or(UnaryNotOperandKind::Unknown)
        }
        syn::Expr::Macro(mac)
            if mac.mac.path.segments.len() == 1
                && mac.mac.path.segments[0].ident.to_string() == "matches" =>
        {
            UnaryNotOperandKind::Bool
        }
        // `expr?` — Rust try operator desugars to `match expr { Ok(v)
        // => v, Err(e) => return Err(e.into()) }` (or the Option
        // counterpart). Mirror the `expression_type_string` Try arm
        // by classifying the inner expression's kind through the
        // Result/Option unwrap. RPython peer: `flowcontext.py`'s
        // POP_BLOCK / END_FINALLY exception-channel join, projected to
        // the success-arm tail.
        syn::Expr::Try(t) => {
            if let Some(inner_ty) = expression_type_string(&t.expr, ctx)
                && let Some(unwrapped) = unwrap_result_or_option(&inner_ty)
            {
                let kind = type_string_to_unary_not_kind(unwrapped);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        // `(*func).can_change_code` / `frame.is_root` — struct field
        // access whose declared type is recorded in
        // `ctx.struct_fields`.  RPython resolves this through
        // `SomeInstance.find_attribute` (`annotator/model.py:430+`);
        // pyre's `expression_type_string` already routes
        // `Expr::Field` to `field_type_string_from_expr`, so the
        // classifier just needs to project the resulting type string.
        syn::Expr::Field(_) => {
            if let Some(ty) = expression_type_string(expr, ctx) {
                let kind = type_string_to_unary_not_kind(&ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            UnaryNotOperandKind::Unknown
        }
        syn::Expr::Call(_) | syn::Expr::MethodCall(_) => {
            if let Some(ty) = expression_type_string(expr, ctx) {
                let kind = type_string_to_unary_not_kind(&ty);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            // `.unwrap()` / `.expect(_)` on a `Result<T, _>` / `Option<T>`
            // receiver projects to `T`.  RPython parity: the annotator
            // walks the method-resolution and sees `T` directly
            // (`rtyper/rmodel.py:rtype_method_unwrap`); pyre lacks
            // generic-method visibility, so synthesise the projection
            // by inspecting the receiver's return type and unwrapping
            // the `Result<...>` / `Option<...>` wrapper.
            if let syn::Expr::MethodCall(mc) = expr
                && (mc.method == "unwrap" || mc.method == "expect")
                && let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
                && let Some(inner) = unwrap_result_or_option(&receiver_ty)
            {
                let kind = type_string_to_unary_not_kind(inner);
                if kind != UnaryNotOperandKind::Unknown {
                    return kind;
                }
            }
            // TODO(receiver-typed-dispatch): the receiver-independent
            // bool shortlist below is a heuristic substitute for
            // `bookkeeper.getdesc(receiver).find_method` (RPython
            // `unaryop.py:206-213`).  RPython matches host-stdlib
            // trait methods by *receiver class* identity; pyre's
            // annotator does not yet track stdlib/cross-crate trait
            // impls, so receiver-typed lookup
            // (`primitive_method_result_type` /
            // `lookup_method_return_type`) misses these names and
            // they reach the catch-all here.  Retire when
            // `fn_return_types` is populated from a metadata-only
            // walk over `pyre-{object,interpreter,jit}` + stdlib
            // trait impls — this entire match arm goes away.
            // Aliasing risk: a user-source method with the same name
            // on an unrelated type that does NOT return bool (e.g.
            // `eq`/`ne`/`lt`/`gt`/`ge`/`le` from `PartialOrd`,
            // `contains`/`any`/`all` from various collection traits)
            // would mis-classify here.  Trim attempts on those names
            // 2026-05-11 broke pyre source build (some user-source
            // call sites depend on the bool fallback) — full removal
            // requires receiver-typed dispatch landing first.
            if let syn::Expr::MethodCall(mc) = expr {
                let method_name = mc.method.to_string();
                if matches!(
                    method_name.as_str(),
                    "is_null"
                        | "is_some"
                        | "is_none"
                        | "is_ok"
                        | "is_err"
                        | "is_empty"
                        | "contains"
                        | "starts_with"
                        | "ends_with"
                        | "eq"
                        | "ne"
                        | "lt"
                        | "le"
                        | "gt"
                        | "ge"
                        // Float-bool predicates (`f64::is_nan`,
                        // `f64::is_infinite`, etc.) — receiver-typed
                        // arm `primitive_method_result_type:5576`
                        // already maps these to `ValueType::Bool` for
                        // typed-call resolution, but
                        // `lookup_method_return_type` only consults
                        // user-source `fn_return_types`.  Mirror the
                        // shortlist here so the overloaded `!` over
                        // a stdlib float predicate classifies as
                        // Bool without depending on receiver-type
                        // tracking propagating to the caller.
                        | "is_nan"
                        | "is_infinite"
                        | "is_finite"
                        | "is_sign_negative"
                        | "is_sign_positive"
                        // Integer sign predicates (`i64::is_positive`,
                        // `i64::is_negative`).  Same rationale as
                        // the float predicates above — RPython's
                        // `bookkeeper.getdesc(receiver).find_method`
                        // would resolve via the host stdlib.
                        | "is_positive"
                        | "is_negative"
                        // `char::is_alphabetic` / `char::is_digit` /
                        // `char::is_alphanumeric` / `char::is_whitespace`
                        // / `char::is_ascii*` — `core::char` predicates
                        // returning `bool`.  Same rationale as the
                        // numeric predicates.
                        | "is_alphabetic"
                        | "is_alphanumeric"
                        | "is_digit"
                        | "is_whitespace"
                        | "is_ascii"
                        | "is_ascii_alphabetic"
                        | "is_ascii_alphanumeric"
                        | "is_ascii_digit"
                        | "is_ascii_whitespace"
                        | "is_ascii_uppercase"
                        | "is_ascii_lowercase"
                        | "is_uppercase"
                        | "is_lowercase"
                        // `std::process::ExitStatus::success` and the
                        // related `Path::exists` family — used by
                        // pyre's stdlib-detection helpers
                        // (`pyre-interpreter/src/importing.rs`).
                        | "success"
                        | "exists"
                        // Cross-crate JIT-driver / descriptor methods
                        // declared on types defined outside
                        // `PYRE_JIT_GRAPH_SOURCES` (e.g. `WarmState`
                        // in `majit-metainterp/src/warmstate.rs:143`,
                        // `Descr::is_array_of_pointers` in
                        // `majit-ir/src/descr.rs:1551`,
                        // `Signature::has_kwarg` in
                        // `pyre-interpreter/src/gateway.rs:235`).  The
                        // walker registers these methods only when
                        // their owner type is in the analyser source
                        // set — for cross-crate types it isn't, so
                        // `lookup_method_return_type` returns None and
                        // the surface `!driver.is_tracing()` /
                        // `!bh_descr.is_array_of_pointers()` /
                        // `!sig.has_kwarg()` falls through.  Same
                        // rationale as the stdlib-method shortlist
                        // above (`is_null` / `is_some` / ...): RPython
                        // `bookkeeper.getdesc(receiver).find_method`
                        // resolves these by host-identity; pyre's
                        // static shortlist substitutes for the missing
                        // whole-program annotator visibility.
                        // Convergence with the cross-crate
                        // `pyre_object::*` Call shortlist: a
                        // metadata-only walk over the host crates
                        // retires this list.
                        | "is_tracing"
                        | "has_compiled_loop"
                        | "is_array_of_pointers"
                        | "has_kwarg"
                        | "has_vararg"
                        // `Iterator::any` / `Iterator::all` — stdlib
                        // iterator predicates that always return
                        // `bool`.  See TODO(receiver-typed-dispatch)
                        // block above.
                        | "any"
                        | "all"
                        // `majit-ir`-side predicate methods on
                        // `OpCode` / `AbstractValue` / `OpRef` /
                        // `ResOperation` — all return `bool` and are
                        // receiver-independent shape introspectors.
                        // Walker registers `Type::method` entries when
                        // `majit-ir/src/resoperation.rs` is in scope,
                        // but multi-segment / variant-call receivers
                        // (`OpCode::IntAdd.is_jit_debug()`,
                        // `AbstractValue::ConstInt(7).is_input_arg()`)
                        // bypass `receiver_type_root`. Same RPython
                        // parity rationale as the stdlib `is_*`
                        // shortlist above.
                        | "is_input_arg"
                        | "is_res_op"
                        | "is_constant"
                        | "is_always_pure"
                        | "is_foldable_guard"
                        | "is_malloc"
                        | "is_memory_access"
                        | "is_comparison"
                        | "is_setarrayitem"
                        | "is_getfield"
                        | "is_setfield"
                        | "is_getarrayitem"
                        | "is_setinteriorfield"
                        | "is_getinteriorfield"
                        | "can_malloc"
                        | "can_raise"
                        | "is_guard_exception"
                        | "is_guard_overflow"
                        | "is_call"
                        | "is_jit_debug"
                        | "is_guard"
                        | "is_ovf"
                        | "is_same_as"
                        | "is_vector_arithmetic"
                        | "is_label"
                        | "is_final"
                        | "should_trace_function_entry"
                ) {
                    return UnaryNotOperandKind::Bool;
                }
            }
            // Stdlib free-function path calls that unambiguously return
            // `bool` — the path-call analogue of the `is_null`/`is_some`
            // / etc. method shortlist above.  Pyre's walker only
            // registers user-source signatures into `fn_return_types`
            // (`front/ast.rs:462+`), so stdlib paths like `std::ptr::eq`
            // never resolve through `lookup_function_return_type`.
            // RPython resolves these via `bookkeeper.getdesc(value)`
            // keyed on the host-stdlib function object identity
            // (`annrpython.py` callee resolution); pyre has no parallel
            // descriptor, so the shortlist stays until the annotator
            // gains stdlib visibility.
            if let syn::Expr::Call(call) = expr
                && let syn::Expr::Path(path) = &*call.func
            {
                let segments: Vec<String> = path
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect();
                let joined = segments.join("::");
                if matches!(
                    joined.as_str(),
                    "std::ptr::eq" | "core::ptr::eq" | "ptr::eq"
                ) {
                    return UnaryNotOperandKind::Bool;
                }
                // `Self::raw_is_constant` / `Self::raw_is_*` —
                // `majit-ir/src/resoperation.rs:185 raw_is_constant(raw: u32) -> bool`
                // and siblings declare bool predicates as `Self::method`
                // calls inside an impl block (function-call shape, not
                // method-call).  TODO(receiver-typed-dispatch): when
                // `Self`-relative paths resolve through the bookkeeper's
                // class identity, this falls out automatically.
                if segments.len() == 2 && segments[0] == "Self" {
                    let last = segments[1].as_str();
                    if matches!(
                        last,
                        "raw_is_constant"
                            | "raw_is_input_arg"
                            | "raw_is_res_op"
                            | "raw_is_const_int"
                            | "raw_is_const_float"
                    ) {
                        return UnaryNotOperandKind::Bool;
                    }
                }
                // Cross-crate `pyre_object::*` predicate family —
                // `pyre/pyre-object/src/{pyobject,typeobject,strobject,
                // excobject}.rs` declares a stable set of `pub unsafe
                // fn is_<...>(obj) -> bool` visibility helpers
                // (`pyobject.rs:308-368`: `is_int`, `is_bool`,
                // `is_float`, `is_long`, `is_int_or_long`, `is_list`,
                // `is_tuple`, `is_dict`, `is_none`, `is_not_implemented`,
                // …; plus `is_exception` / `is_str` / `is_module` /
                // `is_instance` / `is_set_or_frozenset` / `is_type` /
                // `is_bool` / `is_builtin_code` exported from sibling
                // modules).  These are the cross-crate analogue of
                // `std::ptr::eq` — pyre's analyser source set
                // (`generated::PYRE_JIT_GRAPH_SOURCES`) does not include
                // `pyre/pyre-object/src/*.rs`, so `lookup_function_return_
                // type` has no entry to project.  RPython parity:
                // `bookkeeper.getdesc(value)` resolves these helpers by
                // host-stdlib identity (`annrpython.py` callee
                // resolution); pyre's static shortlist substitutes for
                // the missing whole-program annotator visibility.
                // Convergence path: emit a metadata-only walk over
                // `pyre/pyre-object/src/*.rs` that registers
                // `fn_return_types` without subjecting raw-pointer
                // `unsafe fn` bodies to graph analysis (multi-session;
                // the shortlist retires once the metadata-only walk
                // lands).
                if let Some(joined_str) = joined.strip_prefix("pyre_object::") {
                    let last = joined_str.rsplit("::").next().unwrap_or(joined_str);
                    if matches!(
                        last,
                        "is_int"
                            | "is_bool"
                            | "is_float"
                            | "is_long"
                            | "is_int_or_long"
                            | "is_list"
                            | "is_tuple"
                            | "is_dict"
                            | "is_none"
                            | "is_not_implemented"
                            | "is_str"
                            | "is_exception"
                            | "is_module"
                            | "is_instance"
                            | "is_type"
                            | "is_set_or_frozenset"
                            | "is_builtin_code"
                            | "ll_issubclass"
                            | "w_set_contains"
                            | "w_type_get_hasdict"
                            | "w_type_get_weakrefable"
                            | "w_type_get_acceptable_as_base_class"
                    ) {
                        return UnaryNotOperandKind::Bool;
                    }
                }
                // Multi-segment cross-crate / `crate::`-rooted user paths
                // (`pyre_object::is_exception`, `crate::typeobject::is_type`,
                // ...) — the walker registers free functions under the
                // file-local prefix
                // (`build_semantic_program_from_parsed_files_with_options`
                // currently passes `prefix=""` per file,
                // `front/ast.rs:751-780`), so a multi-segment lookup
                // through `lookup_function_return_type` misses the
                // single-name registration.  Cross-crate / cross-module
                // `pyre_object::`-rooted paths are functionally
                // equivalent to bare references in pyre's whole-program
                // mode (RPython `annrpython.py` resolves callees via
                // `bookkeeper.getdesc(value)` keyed on object identity,
                // not on the `module.name`).  Re-attempt the lookup
                // with the trailing single segment so the bare-key
                // fallback (`lookup_function_return_type`'s
                // `segments.len() == 1` branch) finds the registered
                // entry.  Scoped to `!` classification — wider use of
                // last-segment fallback risks cross-module name
                // collisions.
                if segments.len() > 1
                    && let Some(last) = segments.last()
                    && let Some(ret) = ctx.fn_return_types.get(last)
                {
                    let kind = type_string_to_unary_not_kind(ret);
                    if kind != UnaryNotOperandKind::Unknown {
                        return kind;
                    }
                }
                // `pyre_object::typeobject::Layout::expands_equal(...)`
                // — multi-segment Impl-method paths.  The walker
                // registers Impl methods under the bare `Type::method`
                // shape (`Item::Impl` arm at front/ast.rs:591), so a
                // crate-relative call site referencing the method
                // through its module path falls back to the last two
                // segments.  Same RPython parity rationale as the
                // last-single-segment fallback above (whole-program
                // visibility via `bookkeeper.getdesc(value)` keyed on
                // host identity, not on source path).
                if segments.len() >= 2 {
                    let n = segments.len();
                    let bare_impl = format!("{}::{}", segments[n - 2], segments[n - 1]);
                    if let Some(ret) = ctx.fn_return_types.get(&bare_impl) {
                        let kind = type_string_to_unary_not_kind(ret);
                        if kind != UnaryNotOperandKind::Unknown {
                            return kind;
                        }
                    }
                }
                // External numeric type-conversion constructors —
                // `BigInt::from(...)`, `BigUint::from(...)`,
                // `i64::from_str_radix(...)`, etc.  Pyre's walker has no
                // visibility into `num_bigint`, so the constructor's
                // return type isn't in `fn_return_types`.  RPython peer:
                // `bookkeeper.getdesc(value)` resolves host-stdlib
                // arithmetic types (`int`, `float`, `long`) to integer
                // / float annotations; pyre's static shortlist
                // mirrors that for the BigInt family used by the long
                // bytestring carrier (`pyre-interpreter/src/baseobjspace.rs`
                // BigInt arithmetic).
                if segments.len() >= 2 {
                    let n = segments.len();
                    if matches!(segments[n - 2].as_str(), "BigInt" | "BigUint") {
                        return UnaryNotOperandKind::Int;
                    }
                }
            }
            UnaryNotOperandKind::Unknown
        }
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn expr_binary_unary_not_operand_kind(
    ctx: &GraphBuildContext,
    bin: &syn::ExprBinary,
) -> UnaryNotOperandKind {
    match bin.op {
        syn::BinOp::Eq(_)
        | syn::BinOp::Ne(_)
        | syn::BinOp::Lt(_)
        | syn::BinOp::Le(_)
        | syn::BinOp::Gt(_)
        | syn::BinOp::Ge(_)
        | syn::BinOp::And(_)
        | syn::BinOp::Or(_) => UnaryNotOperandKind::Bool,
        syn::BinOp::Add(_)
        | syn::BinOp::Sub(_)
        | syn::BinOp::Mul(_)
        | syn::BinOp::Div(_)
        | syn::BinOp::Rem(_)
        | syn::BinOp::Shl(_)
        | syn::BinOp::Shr(_) => {
            if expr_unary_not_operand_kind(&bin.left, ctx) == UnaryNotOperandKind::Int
                && expr_unary_not_operand_kind(&bin.right, ctx) == UnaryNotOperandKind::Int
            {
                UnaryNotOperandKind::Int
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        syn::BinOp::BitAnd(_) | syn::BinOp::BitOr(_) | syn::BinOp::BitXor(_) => {
            let lhs = expr_unary_not_operand_kind(&bin.left, ctx);
            let rhs = expr_unary_not_operand_kind(&bin.right, ctx);
            if lhs == rhs {
                lhs
            } else {
                UnaryNotOperandKind::Unknown
            }
        }
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn value_type_to_unary_not_kind(ty: &ValueType) -> UnaryNotOperandKind {
    match ty {
        ValueType::Bool => UnaryNotOperandKind::Bool,
        // `lltype.Unsigned`'s `UNARY_INVERT` dispatch routes through
        // the same `int_invert` opname as `lltype.Signed` — see
        // `flowspace/operation.py:521 invert.dispatch_to_register
        // _class('int')` and `rint.py:rtype_int__invert` which both
        // Signed and Unsigned IntegerRepr inherit.  `UnaryNotOperand
        // Kind::Int` here drives the bytecode dispatch decision; the
        // result-type carrier is computed from the operand's actual
        // lowered type at the emit site, so Unsigned operands stay
        // Unsigned through the rtyper.
        ValueType::Int | ValueType::Unsigned => UnaryNotOperandKind::Int,
        _ => UnaryNotOperandKind::Unknown,
    }
}

fn type_string_to_unary_not_kind(type_str: &str) -> UnaryNotOperandKind {
    let trimmed = type_str.trim();
    // Arbitrary-precision integer types — `BigInt` / `BigUint`. Routed
    // through `UNARY_INVERT` (bitwise NOT) like primitive integers,
    // even though their lattice lowering is `ValueType::Ref`. RPython
    // peer: `LongRepr.rtype_invert` (`rtyper/rlong.py:..`) dispatches
    // bigint invert at the rtyper layer; pyre's `OpKind::UnaryOp.
    // result_ty` is computed from the operand's actual lowered type
    // at the emit site (`front/ast.rs:3667 UnaryNotOperandKind::Int`
    // arm), so this kind only drives the bytecode dispatch decision,
    // not the result-type carrier.
    if matches!(trimmed, "BigInt" | "BigUint") {
        return UnaryNotOperandKind::Int;
    }
    value_type_to_unary_not_kind(&type_string_to_value_type(type_str))
}

/// Strip the outer `Result<_, _>` or `Option<_>` wrapper from a type
/// string and return the inner ok / some payload type.  Used by the
/// `.unwrap()` / `.expect()` projection in `expr_unary_not_operand_kind`
/// so a `let x = foo().unwrap()` whose `foo() -> Result<bool, _>`
/// classifies `x` as `Bool`, not `Unknown`.
fn unwrap_result_or_option(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    let inner = trimmed
        .strip_prefix("Result<")
        .or_else(|| trimmed.strip_prefix("Option<"))?
        .strip_suffix('>')?;
    // For `Result<T, E>` keep `T` (split on the top-level comma).  Track
    // angle-bracket depth so nested generics like `Result<Vec<T>, E>` /
    // `Option<Result<bool, _>>` survive the split.
    let mut depth = 0_i32;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(inner[..i].trim()),
            _ => {}
        }
    }
    Some(inner.trim())
}

fn expr_is_statically_bool(expr: &syn::Expr, ctx: &GraphBuildContext) -> bool {
    expr_unary_not_operand_kind(expr, ctx) == UnaryNotOperandKind::Bool
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
    // RPython `flowspace/operation.py:505-510` registers `lt`, `le`,
    // `eq`, `ne`, `ge`, `gt` as 2-arg operators returning lltype.Bool;
    // the annotator stamps the result `SomeBool(SomeInteger)`
    // (`annotator/model.py:185-198` — distinct lattice node from
    // SomeInteger).  Pyre mirrors that with `ValueType::Bool`, which
    // `valuetype_to_someshell` projects to `SomeValue::Bool` and the
    // rtyper picks `BoolRepr` for (`rmodel.rs:2204`).  Downstream
    // jit_codewriter sites that key off `ValueType::Int` already
    // alias Bool to Int (commit 4318ebb51b2 added the 9 wildcard /
    // explicit arms — assembler getkind, call array-descr / ir_type,
    // jtransform stamp/kind/ir).
    if matches!(op, "eq" | "ne" | "lt" | "le" | "gt" | "ge") {
        return ValueType::Bool;
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
        ConstValue::Int(_) => Some(ValueType::Int),
        // RPython annotates `Constant(True)`/`Constant(False)` with
        // `SomeBool(SomeInteger)` (`annotator/model.py:185-227`); the
        // rtyper picks `BoolRepr` and `getkind(lltype.Bool) == 'int'`
        // so the register class merges with Int downstream.  The
        // annotation-stage type is Bool — not Int — so propagate Bool
        // here to keep the lattice node distinct.
        ConstValue::Bool(_) => Some(ValueType::Bool),
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
        OpKind::ConstBool(_) => Some(ValueType::Bool),
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
        // Rust `usize`/`*const T::len` etc — RPython `lltype.Signed`.
        "as_usize" | "len" | "wrapping_mul" => Some(ValueType::Int),
        // Bool-returning predicates: RPython `SomeBool` (`annotator/
        // model.py:185-198`). Was `Int` until the Bool lattice landed
        // (`model.rs:18-42`); split out so the call result reaches
        // downstream `valuetype_to_someshell` as `SomeBool` instead of
        // `SomeInteger`.
        "is_empty" | "is_null" => Some(ValueType::Bool),
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
        // Float bool predicates: RPython `SomeBool` (`annotator/
        // model.py:185-198`) — `math.isnan` / `math.isinf` etc lower
        // through `rfloat.py` and surface as `Bool` annotations.
        (ValueType::Float, "is_nan" | "is_infinite" | "is_finite" | "is_sign_negative") => {
            Some(ValueType::Bool)
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

pub(crate) fn cast_op_name(source_ty: Option<&ValueType>, target_ty: &ValueType) -> &'static str {
    match (source_ty, target_ty) {
        (Some(ValueType::Int), ValueType::Float) => "cast_int_to_float",
        (Some(ValueType::Float), ValueType::Int) => "cast_float_to_int",
        (Some(ValueType::Ref), ValueType::Int) => "cast_ptr_to_int",
        (Some(ValueType::Int), ValueType::Ref) => "cast_int_to_ptr",
        // RPython `rbool.py:49` — Bool widens via dedicated cast ops:
        // `cast_bool_to_int` / `cast_bool_to_float`.  The reverse
        // direction (Int / Float → Bool) goes through truthiness
        // predicates `int_is_true` / `float_is_true`
        // (`rint.py:rtype_int__Bool` / `rfloat.py:rtype_Float__Bool`).
        (Some(ValueType::Bool), ValueType::Int) => "cast_bool_to_int",
        (Some(ValueType::Bool), ValueType::Float) => "cast_bool_to_float",
        (Some(ValueType::Int), ValueType::Bool) => "int_is_true",
        (Some(ValueType::Float), ValueType::Bool) => "float_is_true",
        // RPython `rint.py` cross-signedness casts.  `rbool.py:77-83`
        // `uint_is_true` is the unsigned counterpart to `int_is_true`.
        (Some(ValueType::Unsigned), ValueType::Int) => "cast_uint_to_int",
        (Some(ValueType::Int), ValueType::Unsigned) => "cast_int_to_uint",
        (Some(ValueType::Unsigned), ValueType::Float) => "cast_uint_to_float",
        (Some(ValueType::Float), ValueType::Unsigned) => "cast_float_to_uint",
        (Some(ValueType::Bool), ValueType::Unsigned) => "cast_bool_to_uint",
        (Some(ValueType::Unsigned), ValueType::Bool) => "uint_is_true",
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
            // `StructFieldRegistry::field_type` resolves exact owner keys and
            // accepts crate-prefix suffix recovery only when it is unique, so
            // fully-qualified destructures such as
            // `majit_ir::RdVirtualInfo::VirtualInfo` can still resolve to the
            // registered `RdVirtualInfo::VirtualInfo` identity without
            // first-wins collisions across unrelated variants.
            let owner: String = pat_struct
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            let matched_owner = matched_type.and_then(type_root_from_type_string);
            for field_pat in &pat_struct.fields {
                let field_name = match &field_pat.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(_) => continue,
                };
                let field_type = ctx
                    .struct_fields
                    .field_type(&owner, &field_name)
                    .or_else(|| {
                        matched_owner
                            .as_deref()
                            .and_then(|owner| ctx.struct_fields.field_type(owner, &field_name))
                    })
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

/// `descr.py:359 ARRAY_INSIDE._hints.get('nolength', False)` source-level
/// reader. PyPy's flowgraph carries the lltype object on every array op
/// and the JIT consults `_hints` directly; pyre's source-level analysis
/// has only the Rust type spelling, so we approximate the bit by
/// inspecting the pointee after stripping pointer-like prefixes.
///
/// Default is `False` to match PyPy's `_hints.get('nolength', False)`.
/// We only return `true` when the pointee is unambiguously a contiguous
/// item run with no length header:
///
/// - `[T]` / `[T; N]` — Rust slice / fixed-size array syntax. Pointer
///   addresses items[0]; no length word stored in the block.
/// - `*const T` / `*mut T` / `&T` / `&mut T` where the pointee `T` has
///   no generic parameters (`<…>`) and no parenthesised wrapper
///   (`Ptr(…)`). A bare identifier (`i64`, `Point`, `usize`, …) is
///   read as the *element* type and the pointer addresses items[0].
///
/// All other shapes — `Vec<T>`, `GcArray<T>`, `Ptr(GcArray(T))`,
/// `*const GcArray<T>`, … — keep PyPy's default `False`. The wrapper
/// retains a length header at offset 0; a `*const GcArray<T>` is a
/// pointer to that header block, not to items[0], so its descr must
/// carry `lendescr` per `descr.py:359-362`.
pub(crate) fn nolength_from_array_type_id(array_type_id: Option<&str>) -> bool {
    let Some(s) = array_type_id else {
        return false;
    };
    let mut inner = s.trim();
    loop {
        let stripped = inner
            .strip_prefix("*const ")
            .or_else(|| inner.strip_prefix("*mut "))
            .or_else(|| inner.strip_prefix("&mut "))
            .or_else(|| inner.strip_prefix('&'));
        match stripped {
            Some(rest) => inner = rest.trim_start(),
            None => break,
        }
    }
    // `[T]` / `[T; N]` are unambiguous headerless item runs.
    if inner.starts_with('[') && inner.ends_with(']') {
        return true;
    }
    // Length-prefixed wrappers carry `<` (generic) or `(` (paren-style
    // lltype spelling such as `Ptr(GcArray(...))`).  Keep the PyPy
    // default `False` for those — a pointer to a wrapper still
    // dereferences a length header.
    if inner.contains('<') || inner.contains('(') {
        return false;
    }
    // Bare identifier pointee (`*const i64`, `*const Point`) means the
    // pointer addresses items[0] of a primitive / struct item type.
    // A bare identifier with NO pointer prefix is a value-type binding
    // (e.g. an `array_type_id` directly naming a struct that contains
    // an embedded array); preserve the PyPy default `False` for that.
    s.trim() != inner
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
                // `rlib/rarithmetic.py` + `lltype.Signed` family.
                //
                // TODO(unsigned-producer-flip): `u8`/`u16`/`u32`/`u64`/
                // `usize` should lift to `ValueType::Unsigned` to match
                // `lltype.Unsigned`'s rtyper dispatch (`rbool.py:77-83
                // uint_is_true`, `rint.py` cast family).  The flip
                // currently breaks downstream pyre-source analysis —
                // `annotator/unaryop.rs:445 getattr` raises
                // `AnnotatorError: Cannot find attribute` on
                // const-folded paths, and `setinteriorfield_gc_r` join
                // points mix `'r'` (Ref) and `'i'` (Int) kinds causing
                // an `assembler.rs:581 int_copy` kind-mismatch panic.
                // Reverting to `ValueType::Int` keeps the surface DSL
                // monomorphic-int while the `cast_op_name` / rtyper
                // Unsigned arms (`rtype_cast_uint_to_*`,
                // `rtype_cast_*_to_uint`) remain scaffolded for the
                // eventual flip.  Convergence path: walk all
                // `OpKind::Input { ty: Int }` consumers reachable from
                // `PYRE_JIT_GRAPH_SOURCES`, narrow the ones that take
                // u* (currently widened-to-i64 via `as i64`) onto a
                // dedicated `Unsigned` arm, then flip the producer.
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
                | "char" => ValueType::Int,
                // `lltype.Bool` annotates as `SomeBool(SomeInteger)`
                // (`annotator/model.py:185-198`, distinct lattice node).
                // `getkind(Bool) == 'int'` so register-class code paths
                // alias Bool to Int (jit_codewriter sites added in
                // commit 4318ebb51b2); the producer-side type tag
                // remains Bool so the rtyper picks `BoolRepr`.
                "bool" => ValueType::Bool,
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
        // u* folded into Int alongside i* — see
        // `classify_fn_arg_ty`'s TODO(unsigned-producer-flip) for the
        // cascade list.  Must stay in sync with the producer there so
        // the InteriorFieldRead/Write kind suffix agrees with the
        // inputarg type.
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "char" | "Self::Truth" => ValueType::Int,
        // `lltype.Bool` — see `classify_fn_arg_ty`'s `"bool"` arm for
        // the SomeBool/BoolRepr rationale.
        "bool" => ValueType::Bool,
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
            if let Some(ret) = lookup_function_return_type(ctx, &segments).cloned() {
                return Some(ret);
            }
            // TODO(name-based-callee-resolution): receiver-/host-
            // identity-blind name shortlist below; same convergence
            // path as `lookup_function_return_type`'s TODO marker.
            // Cross-module pyre-interpreter bool / Result<bool>
            // predicate shortlist.  Pyre's `PYRE_JIT_GRAPH_SOURCES`
            // (`generated.rs:149`) is intentionally narrow so the
            // canonical-pipeline analysis time stays bounded; the
            // closure's source set excludes `baseobjspace.rs` /
            // `runtime_ops.rs` / `boolobject.rs` even though the
            // in-scope sources reference their predicates.  Aliasing
            // risk: any user-source function whose last segment
            // matches one of these names but does NOT return `bool`
            // is mis-classified here — convergence retires this
            // arm.
            let last = segments.last()?;
            // Direct `-> bool` returns.
            if matches!(
                last.as_str(),
                "is_true"
                    | "is_iterable"
                    | "w_bool_get_value"
                    | "exception_is_valid_class_w"
                    | "exception_is_valid_obj_as_class_w"
                    | "dict_storage_delete"
            ) {
                return Some("bool".to_string());
            }
            // `-> Result<bool, PyError>` returns — `?` peels to bool at
            // the let-binding's `Expr::Try` arm.
            if matches!(last.as_str(), "contains") {
                return Some("Result<bool, PyError>".to_string());
            }
            None
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
            // Stdlib method shortlist that always returns `bool` —
            // mirrored from `expr_unary_not_operand_kind`'s MethodCall
            // arm so closure-body / chained-receiver type inference
            // resolves through the same identity-by-name shortcut.
            // RPython peer: `bookkeeper.getdesc(receiver).find_method`
            // resolves these by host-stdlib identity; pyre's static
            // shortlist substitutes for the missing whole-program
            // annotator visibility.  Receiver-type independence is the
            // distinguishing property — predicate methods do not
            // propagate receiver identity, so they don't need
            // `local_array_types` / `local_dyn_trait_roots` lookups.
            let method = mc.method.to_string();
            if matches!(
                method.as_str(),
                "is_null"
                    | "is_some"
                    | "is_none"
                    | "is_ok"
                    | "is_err"
                    | "is_empty"
                    | "contains"
                    | "starts_with"
                    | "ends_with"
                    | "eq"
                    | "ne"
                    | "lt"
                    | "le"
                    | "gt"
                    | "ge"
                    | "is_nan"
                    | "is_infinite"
                    | "is_finite"
                    | "is_sign_negative"
                    | "is_sign_positive"
                    | "is_positive"
                    | "is_negative"
                    | "is_alphabetic"
                    | "is_alphanumeric"
                    | "is_digit"
                    | "is_whitespace"
                    | "is_ascii"
                    | "is_ascii_alphabetic"
                    | "is_ascii_alphanumeric"
                    | "is_ascii_digit"
                    | "is_ascii_whitespace"
                    | "is_ascii_uppercase"
                    | "is_ascii_lowercase"
                    | "is_uppercase"
                    | "is_lowercase"
                    | "success"
                    | "exists"
                    | "is_tracing"
                    | "has_compiled_loop"
                    | "is_array_of_pointers"
                    | "has_kwarg"
                    | "has_vararg"
                    | "any"
                    | "all"
            ) {
                return Some("bool".to_string());
            }
            // Closure-passthrough methods — `LocalKey::with(closure)`,
            // `Option::unwrap_or_else(closure)`, `Result::unwrap_or_else
            // (closure)`, `Option::map(closure)`, `Result::map_err(
            // closure)`, `Option::and_then(closure)`,
            // `Option::or_else(closure)`, `Result::ok_or_else(closure)`,
            // `Iterator::filter_map(closure)` etc.  The receiver
            // method's return type IS the closure's return type
            // (modulo `Option<_>` / `Result<_,_>` wrapping), so project
            // the last argument's body type.
            //
            // RPython peer: `bookkeeper.getdesc(method).consider_call_
            // site` reads the callable's return annotation
            // (`bookkeeper.py:355-409`); pyre's static walker
            // substitutes by inspecting the closure body directly.
            //
            // TODO(closure-makefunction-port): RPython
            // `flowspace/flowcontext.py:1177 MAKE_FUNCTION` materialises
            // the closure body as a separate function-graph keyed by
            // the lambda's host identity, and `CALL_FUNCTION` invokes
            // it.  Pyre's closure body still lowers to a single
            // `OpKind::Abort { Closure }` placeholder at
            // `front/ast.rs:4785`, so closure side effects are not
            // analysed.  The return-type projection here is the
            // minimum-viable substitute that keeps method-call sites
            // type-coherent; full MAKE_FUNCTION parity requires
            // routing each closure through a synthetic FunctionGraph
            // + `PyreCallRegistry` entry (multi-session epic).
            if matches!(
                method.as_str(),
                "with" | "with_borrow" | "with_borrow_mut" | "unwrap_or_else"
            ) && let Some(last) = mc.args.last()
                && let syn::Expr::Closure(closure) = last
            {
                let ret = match &closure.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                        ty,
                        &ctx.module_prefix,
                        ctx.known_struct_names,
                        ctx.known_trait_names,
                    ),
                    syn::ReturnType::Default => expression_type_string(&closure.body, ctx),
                };
                if ret.is_some() {
                    return ret;
                }
            }
            // `Option<T>::map(F: FnOnce(T) -> U) -> Option<U>` and
            // `Result<T,E>::map(F: FnOnce(T) -> U) -> Result<U,E>` —
            // the receiver-method return is the *wrapper* of the
            // closure body's return.  Pyre's downstream
            // `expression_type_string` only consults the receiver
            // type-string for shape-matching (`Option<_>` / `Result<_,
            // _>`), so projecting `Option<closure_body>` /
            // `Result<closure_body, _>` keeps the wrapper visible at
            // the call site.
            if matches!(method.as_str(), "map" | "and_then" | "filter")
                && let Some(last) = mc.args.last()
                && let syn::Expr::Closure(closure) = last
                && let Some(receiver_ty) = expression_type_string(&mc.receiver, ctx)
            {
                let body_ty = match &closure.output {
                    syn::ReturnType::Type(_, ty) => qualified_full_type_string(
                        ty,
                        &ctx.module_prefix,
                        ctx.known_struct_names,
                        ctx.known_trait_names,
                    ),
                    syn::ReturnType::Default => expression_type_string(&closure.body, ctx),
                };
                if let Some(body) = body_ty {
                    // `Option<X>.map(|t| body) → Option<body>`,
                    // `Option<X>.and_then(|t| body) → body` (assuming
                    // body is already `Option<_>`),
                    // `Option<X>.filter(|t| <bool>) → Option<X>`
                    // (preserves receiver).  RPython parity:
                    // upstream's `SomeOption` is just a tag —
                    // pyre keeps shape via type-string.
                    if method == "filter" {
                        return Some(receiver_ty);
                    }
                    if method == "and_then" {
                        return Some(body);
                    }
                    // `map` — wrap the body in the receiver's
                    // Option/Result shape.  Strip the existing
                    // `<...>` and re-emit `Wrapper<body>`.
                    if let Some(wrapper) = receiver_ty.split('<').next()
                        && matches!(wrapper, "Option" | "Result")
                    {
                        return Some(format!("{wrapper}<{body}>"));
                    }
                }
            }
            // RPython annotator `SomeList.method_get / .method_first /
            // .method_last`-style result inference, narrowed to the
            // stdlib `Vec<T>` / `[T]` accessors that `let Some(x) =
            // lst.get(i) else { ... }` desugars from.  Pyre's
            // `local_array_types` carries the full container type
            // (`Vec<FieldDescrInfo>`); the `Option<&T>` shape is the
            // Rust-language adaptation of RPython's `lst[i]` access.
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
        // `unsafe { tail }` — RPython has no peer; transparent on the
        // analyser side.  Mirrors the `Expr::Unsafe` arm in
        // `expr_unary_not_operand_kind`.
        syn::Expr::Unsafe(u) => block_tail_type_string(&u.block, ctx),
        // `expr?` — Rust try operator; lowers to:
        //   match expr { Ok(v) => v, Err(e) => return Err(e.into()) }
        // (or the Option counterpart).  Result/Option type carriers
        // are not part of RPython's annotator surface; the closest
        // peer is `flowcontext.py:194-198` POP_BLOCK / END_FINALLY's
        // exception-channel join, which the front-end represents as
        // an `OpKind::Call` with the question-mark desugar happening
        // at the rtyper level.  For the type-string carrier we just
        // need the success-arm projection: `Result<T, E>` → `T`,
        // `Option<T>` → `T`.  Mirrors the unwrapping `outer_generic_
        // inner_type` does for `Rc`/`Arc`/`Box` in
        // `method_as_ref_return_type`.
        syn::Expr::Try(t) => {
            let inner_ty = expression_type_string(&t.expr, ctx)?;
            outer_generic_inner_type(&inner_ty, &["Result", "Option"])
        }
        // `{ ...; tail }` — same transparent handling.
        syn::Expr::Block(b) => block_tail_type_string(&b.block, ctx),
        // `if cond { a } else { b }` — RPython's annotator unifies
        // arm types via `unionof(s_then, s_else)`
        // (`annotator/model.py:UnionedSomeObject`); pyre's frontend
        // lacks a SomeObject lattice, so handle only the case where
        // the two arms agree on a single primitive type string.  The
        // else branch is required (Rust `if` without else is `()` and
        // wouldn't be the rhs of a `let` bound to a useful type).
        syn::Expr::If(if_expr) => {
            let then_ty = block_tail_type_string(&if_expr.then_branch, ctx)?;
            let (_, else_expr) = if_expr.else_branch.as_ref()?;
            let else_ty = expression_type_string(else_expr, ctx)?;
            if then_ty == else_ty {
                Some(then_ty)
            } else {
                None
            }
        }
        // `match scrut { arm => body, ... }` — same unionof rationale
        // as `Expr::If`.  All arm bodies must agree on one type
        // string for the bind site to record a known type.
        syn::Expr::Match(m) => {
            let mut arms = m.arms.iter();
            let first_ty = expression_type_string(&arms.next()?.body, ctx)?;
            for arm in arms {
                let arm_ty = expression_type_string(&arm.body, ctx)?;
                if arm_ty != first_ty {
                    return None;
                }
            }
            Some(first_ty)
        }
        _ => None,
    }
}

/// Tail-expression type string of a Rust block — `{ ...; tail }` ⇒
/// `expression_type_string(tail)`.  RPython's flowspace has no
/// block-tail concept (each opcode pushes onto a flat stack), but
/// pyre's surface DSL wraps tail-yielding blocks under `Expr::If` /
/// `Expr::Match` arms and `Expr::Block` / `Expr::Unsafe` operand
/// expressions.  Returns `None` when the block ends in a statement
/// (no tail) — e.g. `{ x = 1; }`.
fn block_tail_type_string(block: &syn::Block, ctx: &GraphBuildContext) -> Option<String> {
    if let Some(syn::Stmt::Expr(tail, None)) = block.stmts.last() {
        expression_type_string(tail, ctx)
    } else {
        None
    }
}

// TODO(name-based-callee-resolution): RPython resolves callees via
// `bookkeeper.getdesc(value)` keyed by Python object identity
// (`annrpython.py` callee resolution), so the name-based lookup
// below is a textual substitute.  Pyre's walker has no Rust `use`
// chain visibility, so a single bare ident at a call site is
// resolved through three text-keyed fallbacks (bare key →
// last-segment → last-two-segment) any of which can mis-route to
// a similarly-named unrelated function.  Convergence path: feed
// the `bookkeeper`-bound `FunctionDesc` directly through
// `GraphBuildContext` so the same host-identity dispatch
// `simple_call` uses at rtyper time also drives return-type
// lookup at AST lowering time, then retire `fn_return_types`
// entirely.  Multi-session — entry conditions are
// `PyreCallRegistry::ensure_session` reaching every call site
// before AST lowering.
fn lookup_function_return_type<'a>(
    ctx: &'a GraphBuildContext,
    segments: &[String],
) -> Option<&'a String> {
    let key = if segments.len() == 1 && !ctx.module_prefix.is_empty() {
        format!("{}::{}", ctx.module_prefix, segments[0])
    } else {
        segments.join("::")
    };
    if let Some(ret) = ctx.fn_return_types.get(&key) {
        return Some(ret);
    }
    // Fallback for cross-module calls reached via `use crate::other::*`:
    // pyre's walker inserts `fn_return_types` entries with the defining
    // module's prefix (e.g. `pyobject::is_int` for
    // `pyre-object/src/pyobject.rs::is_int`), but Rust call sites that
    // imported the function via `use crate::pyobject::*;` invoke it
    // unqualified.  The frontend does not yet track `use` chains, so
    // resolve a bare single-segment lookup against the bare key as a
    // final fallback. Matches RPython's annotator-level whole-program
    // visibility: a globally-visible `def is_int(...) -> bool` is
    // reachable from any module that imports it without re-qualifying
    // the call site (`annrpython.py` resolves callees via `bookkeeper.
    // getdesc(value)` keyed by Python object identity, not source path).
    if segments.len() == 1
        && let Some(ret) = ctx.fn_return_types.get(&segments[0])
    {
        return Some(ret);
    }
    // Closure-bound locals — `let f = |args| body` registers `f` in
    // `local_closure_returns`. Same RPython parity rationale as the
    // bare-key fallback above; the Rust adaptation is needed because
    // pyre's walker has no closure visibility, so the closure return
    // type is recorded at let-binding time.
    if segments.len() == 1
        && let Some(ret) = ctx.local_closure_returns.get(&segments[0])
    {
        return Some(ret);
    }
    // Cross-module call qualified by `crate::module::fn` /
    // `pyre_object::fn` shapes — pyre's walker registers free
    // functions under bare ident (file walker passes `prefix=""`),
    // so a multi-segment crate-relative call falls back to the last
    // segment. Same RPython parity rationale as the bare-single-
    // segment fallback above; whole-program visibility is keyed on
    // host identity, not on source path.
    if segments.len() >= 2
        && let Some(last) = segments.last()
        && let Some(ret) = ctx.fn_return_types.get(last)
    {
        return Some(ret);
    }
    // Multi-segment Impl-method paths — `pyre_object::typeobject::
    // Layout::expands_equal(...)` is registered under the bare
    // `Type::method` shape (`Item::Impl` arm at front/ast.rs:591).
    // Mirror of the same fallback in `expr_unary_not_operand_kind`'s
    // `Expr::Call` arm.
    if segments.len() >= 2 {
        let n = segments.len();
        let bare_impl = format!("{}::{}", segments[n - 2], segments[n - 1]);
        if let Some(ret) = ctx.fn_return_types.get(&bare_impl) {
            return Some(ret);
        }
    }
    None
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
    fn struct_field_registry_does_not_fall_back_to_ambiguous_leaf_names() {
        let parsed = crate::parse::parse_source(
            r#"
            mod a {
                pub struct Foo { pub x: i64 }
                pub fn read(foo: Foo) -> i64 { foo.x }
                pub fn destructure(foo: Foo) -> i64 {
                    let Foo { x } = foo;
                    x
                }
            }
            mod b {
                pub struct Foo { pub x: f64 }
                pub fn read(foo: Foo) -> f64 { foo.x }
                pub fn destructure(foo: Foo) -> f64 {
                    let Foo { x } = foo;
                    x
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");

        assert_eq!(program.struct_fields.field_type("a::Foo", "x"), Some("i64"));
        assert_eq!(program.struct_fields.field_type("b::Foo", "x"), Some("f64"));
        assert_eq!(
            program.struct_fields.field_type("Foo", "x"),
            None,
            "bare Foo is ambiguous and must not pick the first registered module"
        );

        let field_read = |func_name: &str| {
            let graph = &program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .unwrap_or_else(|| panic!("{func_name} graph"))
                .graph;
            graph
                .blocks
                .iter()
                .flat_map(|block| block.operations.iter())
                .find_map(|op| match &op.kind {
                    OpKind::FieldRead { field, ty, .. } if field.name == "x" => {
                        Some((field.owner_root.clone(), ty.clone()))
                    }
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{func_name} FieldRead"))
        };

        assert_eq!(
            field_read("a::read"),
            (Some("a::Foo".to_string()), ValueType::Int)
        );
        assert_eq!(
            field_read("b::read"),
            (Some("b::Foo".to_string()), ValueType::Float)
        );

        let destructured_input_ty = |func_name: &str| {
            let graph = &program
                .functions
                .iter()
                .find(|func| func.name == func_name)
                .unwrap_or_else(|| panic!("{func_name} graph"))
                .graph;
            graph.blocks.iter().find_map(|block| {
                block.operations.iter().find_map(|op| match &op.kind {
                    OpKind::Input { name, ty } if name == "x" => Some(ty.clone()),
                    _ => None,
                })
            })
        };

        assert_eq!(
            destructured_input_ty("a::destructure"),
            Some(ValueType::Int)
        );
        assert_eq!(
            destructured_input_ty("b::destructure"),
            Some(ValueType::Float)
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
    fn match_three_arm_rebind_routes_through_iterative_fold_merge() {
        // Audit Cat 2-3 fix: a 3-arm match whose arms each rebind a
        // pre-existing local must produce a single merge-block phi
        // that unifies all three arm vids — exercising the iterative
        // left-fold over 2-way `FrameState::union` (vid disagreement
        // at any fold step allocates a fresh phi) plus eager phi
        // install at union time.  RPython parity:
        // `flowspace/flowcontext.py:430-436 mergeblock` repeatedly
        // 2-way-unions arriving candidates against the running
        // state; pyre's static AST shape applies the same fold over
        // the open-arm exit snapshots in a single Expr::Match
        // lowering visit.
        //
        // Without the iterative fold, the merge would carry no ctx
        // update at union time and the post-merge `x` read would
        // depend on the lazy installer firing post-hoc — the audit's
        // exact "PyPy-style merge state not reflected in ctx" gap.
        let parsed = crate::parse::parse_source(
            r#"
            fn example(tag: i64) -> i64 {
                let mut x = 0;
                match tag {
                    1 => { x = 10; }
                    2 => { x = 20; }
                    _ => { x = 30; }
                }
                x
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
        // The merge block must own an `x` phi inputarg installed
        // eagerly by the iterative fold's fresh-phi step (the
        // disagreeing-vid branch of 2-way `union`).
        let phi_block = graph.blocks.iter().find(|block| {
            block.operations.iter().any(|op| {
                op.result.is_some_and(|r| block.inputargs.contains(&r))
                    && matches!(
                        &op.kind,
                        OpKind::Input { name, .. } if name == "x"
                    )
            })
        });
        assert!(
            phi_block.is_some(),
            "expected 3-arm match merge to allocate an `x` phi inputarg via the iterative fold; graph:\n{}",
            graph.dump()
        );
        // All three arms must thread their per-arm rebind value as
        // `Link.args` into the merge — each predecessor edge must
        // carry ≥1 arg (the rebound `x`).  ≥3 predecessors covers
        // the n-way generalisation; the ≥1 per-edge contract pins
        // that the eager install threaded through every arm rather
        // than only a subset.
        let phi_block_id = phi_block.unwrap().id;
        let pred_arg_count: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == phi_block_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_count.len() >= 3 && pred_arg_count.iter().all(|&n| n >= 1),
            "expected ≥3 predecessor links (3-arm match) with ≥1 Link.args each; got {:?}\ngraph:\n{}",
            pred_arg_count,
            graph.dump()
        );
    }

    #[test]
    fn match_arm_rebind_threads_phi_inputarg_through_lazy_installer() {
        // Slice 4 / Stage D parity test: a pre-match local rebound on
        // every match arm must resolve to a merge-block phi inputarg
        // when read after the match, with both predecessor links
        // carrying the arm-specific rebind value.  RPython parity:
        // `flowspace/framestate.py:113-114 union` returns a fresh
        // `Variable` when both incoming `locals_w` slots are
        // `Variable`s, and `flowspace/flowcontext.py:449
        // currentstate.getoutputargs(newstate)` produces the
        // predecessor-side `Link.args`.
        //
        // Without Slice 4a's per-arm `Block.framestate` stamp, the
        // lazy installer's predecessor walk would find no snapshot
        // on the arm tails and fall back to a naked `OpKind::Input`
        // emit — the resulting graph would still type-check but the
        // arm-specific rebinds would not flow into the merge.
        let parsed = crate::parse::parse_source(
            r#"
            fn example(cond: bool) -> i64 {
                let mut x = 0;
                match cond {
                    true => { x = 10; }
                    false => { x = 20; }
                }
                x
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
        // Some block must carry `OpKind::Input { name: "x" }` as an
        // inputarg — that's the merge-block phi the lazy installer
        // allocates when the post-match `x` read fires.
        let phi_block = graph.blocks.iter().find(|block| {
            block.operations.iter().any(|op| {
                op.result.is_some_and(|r| block.inputargs.contains(&r))
                    && matches!(
                        &op.kind,
                        OpKind::Input { name, .. } if name == "x"
                    )
            })
        });
        assert!(
            phi_block.is_some(),
            "expected match-arm-rebind merge to allocate an `x` phi inputarg; graph:\n{}",
            graph.dump()
        );
        // Each predecessor edge into the phi block must carry one
        // extra `Link.args` value — the per-arm rebind threaded back
        // by the lazy installer.
        let phi_block_id = phi_block.unwrap().id;
        let pred_arg_count: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == phi_block_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_count.len() >= 2 && pred_arg_count.iter().all(|&n| n >= 1),
            "expected ≥2 predecessor links with ≥1 Link.args each; got {:?}\ngraph:\n{}",
            pred_arg_count,
            graph.dump()
        );
    }

    #[test]
    fn while_loop_back_edge_threads_modified_local_through_header_phi() {
        // Slice 5a parity test: a while loop whose body rebinds a
        // pre-loop local must produce a header inputarg phi for that
        // local with link args from BOTH the pre-loop block (forward
        // edge, supplying the pre-loop value) AND the body tail
        // (back-edge, supplying the body's rebound value).  RPython
        // parity: `flowspace/flowcontext.py:438-451 mergeblock` opens
        // a fresh `SpamBlock(newstate)` whose `framestate.locals_w`
        // slot for a body-modified name becomes a `Variable`; the
        // closing forward-edge link's `getoutputargs` carries the
        // pre-loop value and each subsequent merge (back-edge in
        // pyre's single-pass) carries the body's value.
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut i: i64 = 0;
                while i < 10 {
                    i = i + 1;
                }
                i
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
        // Find the header block — the one whose inputarg is
        // `Input { name: "i" }` and that has TWO predecessor edges
        // (forward + back-edge).
        let mut header_id: Option<BlockId> = None;
        for block in &graph.blocks {
            if block.inputargs.is_empty() {
                continue;
            }
            let has_i_input = block.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "i")
                    && op.result.is_some_and(|r| block.inputargs.contains(&r))
            });
            if !has_i_input {
                continue;
            }
            let pred_count = graph
                .blocks
                .iter()
                .flat_map(|b| b.exits.iter().filter(|e| e.target == block.id).map(|_| ()))
                .count();
            if pred_count >= 2 {
                header_id = Some(block.id);
                break;
            }
        }
        let header_id = header_id.unwrap_or_else(|| {
            panic!(
                "expected header block with `i` inputarg + ≥2 preds; graph:\n{}",
                graph.dump()
            )
        });
        // Each predecessor edge into the header must carry exactly
        // one Link.arg (the per-iteration value of `i`).
        let pred_arg_counts: Vec<usize> = graph
            .blocks
            .iter()
            .flat_map(|b| {
                b.exits.iter().filter_map(move |exit| {
                    if exit.target == header_id {
                        Some(exit.args.len())
                    } else {
                        None
                    }
                })
            })
            .collect();
        assert!(
            pred_arg_counts.iter().all(|&n| n == 1),
            "every predecessor link into the header must carry 1 arg \
             (the `i` value); got {:?}; graph:\n{}",
            pred_arg_counts,
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

    #[test]
    fn unary_not_on_int_param_lowers_to_invert() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64) -> i64 {
                !x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        assert!(
            graph
                .block(graph.startblock)
                .operations
                .iter()
                .any(|op| matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, .. } if op == "invert"
                )),
            "expected `!i64` to lower as RPython UNARY_INVERT"
        );
    }

    #[test]
    fn unary_not_on_bool_param_lowers_to_bool_branch() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(flag: bool) -> bool {
                !flag
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let start = graph.block(graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!bool` to lower through RPython UNARY_NOT"
        );
        assert!(
            start.exitswitch.is_some(),
            "UNARY_NOT must branch on bool(x)"
        );
    }

    /// `!x` on a statically unclassified operand fail-louds, mirroring
    /// `flowcontext.py:194,535-538`'s strict UNARY_NOT vs UNARY_INVERT
    /// dispatch and the `build_flow.rs:4404-4416` fail-loud peer.
    /// The `expr_unary_not_operand_kind` classifier
    /// (`front/ast.rs:5582`) covers the production patterns surfaced
    /// in `pyre-{object,interpreter,jit}/src/` plus
    /// `majit-ir/src/resoperation.rs`; an opaque operand with no type
    /// information remains a hard error.
    #[test]
    fn unary_not_on_unknown_operand_fail_louds() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Opaque;
            fn example(x: Opaque) -> bool {
                !x
            }
        "#,
        );
        let err =
            build_semantic_program(&parsed).expect_err("`!x` on opaque operand must fail-loud");
        assert!(
            format!("{err:?}").contains("UnaryNotUnknownOperand"),
            "expected UnaryNotUnknownOperand variant, got {err:?}",
        );
    }

    /// `!std::ptr::eq(a, b)` — stdlib free-function path call whose
    /// `bool` return is known by shortlist (parity with the
    /// `is_null`/`is_some`/... method-call shortlist; pyre's walker
    /// has no stdlib visibility, so RPython
    /// `bookkeeper.getdesc(value)` host-stdlib lookup is mirrored
    /// statically here).  Must lower through the UNARY_NOT
    /// `op.bool` + branch shape, not pop out as Unknown.
    #[test]
    fn unary_not_on_stdlib_ptr_eq_lowers_to_bool_branch() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(a: *const u8, b: *const u8) -> bool {
                !std::ptr::eq(a, b)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program.functions[0].graph;
        let start = graph.block(graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!std::ptr::eq(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// Multi-segment user path call (e.g. `crate::predicate(...)`,
    /// `pyre_object::is_exception(...)`) — the analyser registers
    /// free functions under their file-local prefix
    /// (`build_semantic_program_from_parsed_files_with_options`
    /// passes `prefix=""` per file at front/ast.rs:751-780), so a
    /// multi-segment crate-relative lookup misses the keyed
    /// `lookup_function_return_type`.  The last-segment fallback in
    /// `expr_unary_not_operand_kind`'s `Expr::Call` arm recovers
    /// the bool return so `!` classifies correctly.
    #[test]
    fn unary_not_on_multi_segment_user_path_classifies_via_last_segment() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn predicate(x: i64) -> bool { x == 0 }
            fn example(x: i64) -> bool {
                !crate::predicate(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let start = example.graph.block(example.graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!crate::predicate(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// `let local = if cond { call_a() } else { call_b() };
    /// if !local { ... }` — `expression_type_string` unifies the two
    /// arms' tail-expression types and stamps `local`'s recorded
    /// type, so the subsequent `!local` classifies via the bool
    /// `Path` arm rather than falling through.  RPython parity:
    /// `annotator/model.py` `unionof(s_then, s_else)` resolves the
    /// merged binding's annotation; pyre handles the narrow case
    /// where both arms agree on a single primitive type string.
    #[test]
    fn unary_not_on_let_bound_if_else_classifies_via_arm_unification() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn pick_a(x: i64) -> bool { x == 0 }
            pub fn pick_b(x: i64) -> bool { x != 0 }
            fn example(cond: bool, x: i64) -> bool {
                let local = if cond { pick_a(x) } else { pick_b(x) };
                !local
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!local` (bound from if-else) to lower through RPython UNARY_NOT"
        );
    }

    /// Companion to the if-else case for `match` arms.  Each match
    /// arm returns a known-bool call; `expression_type_string`'s
    /// `Expr::Match` arm unifies them so the let-bound `flag`
    /// records `bool`, classifying the downstream `!flag`.
    #[test]
    fn unary_not_on_let_bound_match_classifies_via_arm_unification() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn pick_a(x: i64) -> bool { x == 0 }
            pub fn pick_b(x: i64) -> bool { x != 0 }
            pub fn pick_c(x: i64) -> bool { x > 0 }
            fn example(tag: i64, x: i64) -> bool {
                let flag = match tag {
                    0 => pick_a(x),
                    1 => pick_b(x),
                    _ => pick_c(x),
                };
                !flag
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!flag` (bound from match) to lower through RPython UNARY_NOT"
        );
    }

    /// `!driver.is_tracing()` — cross-crate method on a receiver
    /// whose owner type (`WarmState` in
    /// `majit-metainterp/src/warmstate.rs:143`) is outside the
    /// analyser source set.  `lookup_method_return_type` returns
    /// None because `fn_return_types` has no entry for the
    /// cross-crate owner; the method-name shortlist in
    /// `expr_unary_not_operand_kind`'s `Expr::Call` /
    /// `Expr::MethodCall` arm substitutes for the missing
    /// whole-program annotator visibility.  RPython parity:
    /// `bookkeeper.getdesc(receiver).find_method`
    /// (`unaryop.py:206-213`) resolves cross-module methods by
    /// host-identity; pyre's static shortlist mirrors that.
    #[test]
    fn unary_not_on_cross_crate_method_classifies_via_method_shortlist() {
        let parsed = crate::parse::parse_source(
            r#"
            type Driver = i64;
            // No `impl` block — the receiver type's `is_tracing`
            // method lives in a crate not visible to the analyser.
            // The shortlist mirrors RPython's host-stdlib resolution.
            fn example(driver: &Driver) -> bool {
                !driver.is_tracing()
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!driver.is_tracing()` to lower through RPython UNARY_NOT"
        );
    }

    /// `!pyre_object::is_exception(obj)` — cross-crate predicate
    /// path that the analyser source set
    /// (`generated::PYRE_JIT_GRAPH_SOURCES`) does not visit.  The
    /// shortlist in `expr_unary_not_operand_kind`'s `Expr::Call` arm
    /// substitutes for the missing whole-program visibility, the
    /// way RPython's `bookkeeper.getdesc(value)` would resolve the
    /// helper by host-identity.  Pinned to bool so the surface
    /// `!is_exception(...)` lowers through UNARY_NOT.
    #[test]
    fn unary_not_on_pyre_object_predicate_classifies_via_crate_shortlist() {
        let parsed = crate::parse::parse_source(
            r#"
            type Obj = i64;
            fn example(obj: Obj) -> bool {
                !pyre_object::is_exception(obj)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!pyre_object::is_exception(...)` to lower through RPython UNARY_NOT"
        );
    }

    /// `let v = call_returning_result()?; if !v { ... }` —
    /// `Expr::Try` projects the success arm's `T` out of
    /// `Result<T, E>` so the let-binding records the inner type.
    /// RPython has no peer for `?`; the closest is
    /// `flowcontext.py:194-198` exception-channel join, which would
    /// surface as a separate `SomeBool` annotation on the success
    /// arm via the rtyper's POP_BLOCK / END_FINALLY shape.  Pyre's
    /// surface lowers `?` at the front-end via the analyser's
    /// type-string carrier, mirroring the unwrapping that
    /// `method_as_ref_return_type` does for `Rc`/`Arc`/`Box`.
    #[test]
    fn unary_not_on_let_bound_try_classifies_via_result_inner() {
        let parsed = crate::parse::parse_source(
            r#"
            pub fn maybe_pred(x: i64) -> Result<bool, i64> { Ok(x == 0) }
            fn example(x: i64) -> Result<bool, i64> {
                let flag = maybe_pred(x)?;
                Ok(!flag)
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!flag` (let-bound from `expr?`) to lower through RPython UNARY_NOT"
        );
    }

    /// `!frame.is_root` — struct field access whose declared type
    /// is in `ctx.struct_fields`.  Mirror of the Call/MethodCall
    /// shortcut: the classifier projects `Expr::Field` through
    /// `expression_type_string → field_type_string_from_expr` and
    /// classifies the resulting type string.  RPython parity:
    /// `SomeInstance.find_attribute` (`annotator/model.py:430+`)
    /// resolves the field's annotation; pyre uses the static
    /// struct-field registry as the equivalent lookup table.
    #[test]
    fn unary_not_on_struct_field_classifies_via_field_type_registry() {
        let parsed = crate::parse::parse_source(
            r#"
            pub struct Frame { pub is_root: bool, pub depth: i64 }
            fn example(frame: &Frame) -> bool {
                !frame.is_root
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let saw_bool = example
            .graph
            .iter_blocks()
            .flat_map(|b| b.operations.iter())
            .any(|op| {
                matches!(
                    &op.kind,
                    OpKind::UnaryOp { op, result_ty, .. }
                        if op == "bool" && *result_ty == ValueType::Bool
                )
            });
        assert!(
            saw_bool,
            "expected `!frame.is_root` to lower through RPython UNARY_NOT"
        );
    }

    /// `!unsafe { f(...) }` — the unsafe block is a transparent
    /// wrapper for the classifier (mirror of the Paren / Group
    /// arms).  The inner call's bool return must propagate so `!`
    /// resolves to UNARY_NOT rather than falling through to the
    /// Unknown arm flagged by `TODO(receiver-typed-dispatch)`.
    #[test]
    fn unary_not_on_unsafe_block_unwraps_inner() {
        let parsed = crate::parse::parse_source(
            r#"
            unsafe fn predicate(x: i64) -> bool { x == 0 }
            fn example(x: i64) -> bool {
                !unsafe { predicate(x) }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let example = program
            .functions
            .iter()
            .find(|f| f.graph.name == "example")
            .expect("example graph must be present");
        let start = example.graph.block(example.graph.startblock);
        assert!(
            start.operations.iter().any(|op| matches!(
                &op.kind,
                OpKind::UnaryOp { op, result_ty, .. }
                    if op == "bool" && *result_ty == ValueType::Bool
            )),
            "expected `!unsafe {{ ... }}` to lower through RPython UNARY_NOT"
        );
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

    // ── FrameState — Cat 2.1 cross-block locals threading scaffold ──
    //
    // Slice 1: data-type + pure-function tests only.  The
    // capture/install methods are exercised through Slices 2-6 when
    // wired into the lowering path; here we pin the storage-order
    // contract (Stage A2: first-bind positional, mirroring RPython
    // `co_varnames` slot order) + the rebind detection contract that
    // later slices depend on.

    fn frame_entry(name: &str, vid: usize, ty: ValueType) -> Option<FrameStateEntry> {
        Some(FrameStateEntry {
            name: name.to_string(),
            value_id: ValueId(vid),
            value_type: ty,
        })
    }

    #[test]
    fn locals_frame_link_args_preserves_storage_order() {
        // RPython `flowcontext.py:835 LOAD_FAST` reads `frame.locals_w`
        // by slot index.  Pyre stores entries densely at graph-wide
        // first-bind slot positions (`co_varnames` slot order parity),
        // so `link_args()` must walk that order verbatim, yielding only
        // bound (non-None) slots — every predecessor link feeding a
        // merge block sees the same slot order, so `Link.args[i]` lines
        // up with `inputargs[i]` at the successor.
        let frame = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int),
                frame_entry("a", 1, ValueType::Int),
                frame_entry("b", 2, ValueType::Ref),
            ],
        };
        assert_eq!(
            frame.link_args(),
            vec![ValueId(3), ValueId(1), ValueId(2)],
            "link_args must walk entries in storage (first-bind) order; \
             alphabetisation would break slot-position parity at merges"
        );
    }

    #[test]
    fn locals_frame_iter_yields_storage_order() {
        // The `iter` API is what successor blocks consume to allocate
        // matching `inputargs`; slot order must be stable so the i-th
        // `Link.args` entry from a predecessor lines up with the i-th
        // `inputargs` slot at the successor.  Pyre stores entries
        // densely at graph-wide first-bind slot positions, so iteration
        // walks the stored order verbatim, skipping unbound (None)
        // slots.
        let frame = FrameState {
            entries: vec![
                frame_entry("b", 2, ValueType::Ref),
                frame_entry("a", 1, ValueType::Int),
            ],
        };
        let collected: Vec<(&str, ValueId)> = frame.iter().map(|(n, v, _)| (n, v)).collect();
        assert_eq!(
            collected,
            vec![("b", ValueId(2)), ("a", ValueId(1))],
            "iter must walk stored (first-bind positional) order"
        );
    }

    #[test]
    fn frame_state_union_carries_through_when_predecessors_agree() {
        // Both predecessors bound `x` to the same ValueId — successor
        // need not allocate a phi.  RPython `framestate.py:108
        // if w1 == w2: return w1`.
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = a
            .union(&b, &mut graph)
            .expect("matching states must union cleanly");
        assert_eq!(
            merged.entries,
            vec![Some(crate::model::FrameStateEntry {
                name: "x".to_string(),
                value_id: ValueId(7),
                value_type: ValueType::Int,
            })]
        );
        // No fresh allocation: `next_value` cursor stays at 100.
        assert_eq!(graph.next_value(), 100);
    }

    #[test]
    fn frame_state_union_needs_phi_on_value_id_disagreement() {
        // Predecessors disagree on `x`'s ValueId — successor must
        // allocate a fresh phi inputarg.  RPython `framestate.py:
        // 113-114 return Variable()`.
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 8, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = a
            .union(&b, &mut graph)
            .expect("disagreeing vids must produce a fresh phi vid without TypeMismatch");
        assert_eq!(
            merged.entries,
            vec![Some(crate::model::FrameStateEntry {
                name: "x".to_string(),
                value_id: ValueId(100),
                value_type: ValueType::Int,
            })]
        );
        // Fresh allocation consumed exactly one vid.
        assert_eq!(graph.next_value(), 101);
    }

    #[test]
    fn frame_state_union_kills_one_sided_slots() {
        // RPython `framestate.py:110-111` None-kill: a slot present in
        // only one predecessor is dropped from the merged state.  Pyre
        // realises this via dense positional entries — graph-wide
        // first-bind order assigns slots [survivor=0, only_a=1,
        // only_b=2]; predecessor A bound only_a but never only_b, B
        // bound only_b but never only_a, so each one-sided slot
        // collapses to None-kill at union.  The merged state preserves
        // slot positions: [Some(survivor), None, None].
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int),
                frame_entry("only_a", 2, ValueType::Int),
                None,
            ],
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int),
                None,
                frame_entry("only_b", 3, ValueType::Int),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = a
            .union(&b, &mut graph)
            .expect("type-compatible slots must union cleanly");
        assert_eq!(
            merged.entries.len(),
            3,
            "merged state must preserve positional length = max(len)"
        );
        let names: Vec<&str> = merged
            .entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| e.name.as_str()))
            .collect();
        assert_eq!(
            names,
            vec!["survivor"],
            "one-sided slots must be killed (None-kill semantics)"
        );
        assert!(
            merged.entries[1].is_none() && merged.entries[2].is_none(),
            "one-sided slots stay positionally as None"
        );
    }

    #[test]
    fn frame_state_union_pads_shorter_side_with_implicit_none() {
        // RPython `framestate.py:14 _union` zips equal-length
        // `locals_w`.  Pyre's graph-wide first-bind order is
        // append-only, so a predecessor whose snapshot was taken before
        // a later name was first bound has a shorter `entries` Vec; the
        // shorter side is treated as `None` for the missing tail slots
        // (None-kill at union).  The merged state's length matches the
        // wider side (`max(len)`).
        let early = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Int)],
        };
        let late = FrameState {
            entries: vec![
                frame_entry("x", 1, ValueType::Int),
                frame_entry("y_added_later", 2, ValueType::Int),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = early
            .union(&late, &mut graph)
            .expect("type-compatible slots must union cleanly");
        assert_eq!(
            merged.entries.len(),
            2,
            "merged state must extend to wider predecessor's length"
        );
        let names: Vec<&str> = merged
            .entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| e.name.as_str()))
            .collect();
        assert_eq!(names, vec!["x"]);
        assert!(merged.entries[1].is_none(), "padded slot stays as None");
    }

    #[test]
    fn frame_state_union_concrete_kind_disagreement_returns_type_mismatch() {
        // RPython `flowspace/framestate.py:88 FrameState.union` wraps
        // its inner `_union` in `try/except UnionError: return None`,
        // and `flowcontext.py:430-436 mergeblock` reads that `None` as
        // a **whole-state** "this candidate did not unify, retry /
        // make_next_block" signal — NOT a per-slot drop.  Silently
        // None-killing the slot would let post-merge reads of `x`
        // surface as undefined-local instead of blocking the merge,
        // which would reorder PyPy's mergeblock contract.
        //
        // Pyre surfaces the same shape as `Err(UnionError::TypeMismatch)`
        // for the whole call.  Under valid Rust input rustc rejects
        // any source that binds the same local to two different
        // concrete kinds across arms (`Int` vs `Ref` here), so
        // production callers `.expect(...)` the `Err` arm; this
        // fixture bypasses the Rust frontend by constructing the
        // framestates by hand to verify the failure path.
        use crate::model::UnionError;
        let a = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Int)],
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 2, ValueType::Ref)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let err = a
            .union(&b, &mut graph)
            .expect_err("concrete kind disagreement must surface UnionError::TypeMismatch");
        assert_eq!(
            err,
            UnionError::TypeMismatch {
                name: "x".to_string(),
                self_type: ValueType::Int,
                other_type: ValueType::Ref,
            },
        );
        // `framestate.py:73-89 try/except UnionError: return None`
        // discards a partially-built result; pyre's atomic two-pass
        // implementation must not advance `graph.next_value()` when
        // the union ultimately fails.  Slot 0 disagrees on vid (1 vs
        // 2) which would have allocated a fresh vid in Pass 2, but
        // Pass 1's TypeMismatch detection runs before any
        // `graph.alloc_value()` call.
        assert_eq!(
            graph.next_value(),
            100,
            "no fresh allocation must occur when union returns Err"
        );
    }

    #[test]
    fn frame_state_union_does_not_advance_vid_counter_on_late_type_mismatch() {
        // Slot 0 disagrees on vid only (would allocate a fresh vid in
        // Pass 2); slot 1 disagrees on concrete kind (TypeMismatch).
        // The atomic-on-error contract requires that slot 0's
        // hypothetical fresh allocation is NOT committed because
        // slot 1's failure aborts before Pass 2 runs.
        use crate::model::UnionError;
        let a = FrameState {
            entries: vec![
                frame_entry("x", 1, ValueType::Int),
                frame_entry("y", 3, ValueType::Int),
            ],
        };
        let b = FrameState {
            entries: vec![
                frame_entry("x", 2, ValueType::Int),
                frame_entry("y", 4, ValueType::Ref),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let err = a
            .union(&b, &mut graph)
            .expect_err("late-slot kind disagreement must surface UnionError::TypeMismatch");
        assert!(matches!(err, UnionError::TypeMismatch { .. }));
        assert_eq!(
            graph.next_value(),
            100,
            "no fresh allocation must occur when union returns Err — \
             even when an earlier slot would have triggered one in Pass 2"
        );
    }

    #[test]
    fn frame_state_union_treats_unknown_as_wildcard() {
        // RPython `flowspace/Variable` is untyped at flowspace level,
        // so `framestate.py:union` never raises on type mismatch — it
        // produces a fresh `Variable` for any disagreement.  Pyre
        // tracks ValueType to pin int/ref/float register banks for
        // the typed pipeline, but `Unknown` is the explicit "no
        // annotation captured" sentinel and is structurally a
        // wildcard: an inferred-only bind on one arm and an annotated
        // bind on the other are still the same slot.  TypeMismatch
        // fires only when both sides are concrete and disagree.
        let inferred = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Unknown)],
        };
        let annotated = FrameState {
            entries: vec![frame_entry("x", 1, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        // Carry-through: same vid, Unknown vs Int → resolves to Int.
        let merged = inferred
            .union(&annotated, &mut graph)
            .expect("Unknown wildcard against concrete kind must union cleanly");
        assert_eq!(
            merged.entries,
            vec![Some(crate::model::FrameStateEntry {
                name: "x".to_string(),
                value_id: ValueId(1),
                value_type: ValueType::Int,
            })]
        );
        // Symmetric: concrete on self, Unknown on other → still Int.
        let merged_swap = annotated
            .union(&inferred, &mut graph)
            .expect("Unknown wildcard against concrete kind must union cleanly");
        assert_eq!(
            merged_swap.entries[0].as_ref().unwrap().value_type,
            ValueType::Int
        );
    }

    #[test]
    fn frame_state_union_three_arm_fold_carries_through_when_all_agree() {
        // Three-arm iterative fold mirroring upstream `flowspace/
        // flowcontext.py:430-436 mergeblock`'s repeated 2-way union
        // against each arriving candidate.  All three arms bind `x`
        // to the same vid → every fold step carries the vid through
        // (`framestate.py:108 if w1 == w2: return w1`).  No fresh
        // allocation along the chain.
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let c = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let acc = a
            .union(&b, &mut graph)
            .expect("matching arms must union cleanly")
            .union(&c, &mut graph)
            .expect("matching arms must union cleanly");
        assert_eq!(
            acc.entries,
            vec![Some(crate::model::FrameStateEntry {
                name: "x".to_string(),
                value_id: ValueId(7),
                value_type: ValueType::Int,
            })]
        );
        assert_eq!(graph.next_value(), 100, "no fresh allocation expected");
    }

    #[test]
    fn frame_state_union_fold_allocates_fresh_vid_at_disagreement_step() {
        // Iterative fold: a + b agree on vid=7, c brings vid=8 — the
        // SECOND fold step (`acc.union(c)`) is where vids disagree, so
        // the fresh vid is allocated there (mirrors upstream's
        // `framestate.py:113-114 return Variable()` triggered by the
        // generalising `mergeblock` step that introduces a new
        // SpamBlock).
        let a = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let b = FrameState {
            entries: vec![frame_entry("x", 7, ValueType::Int)],
        };
        let c = FrameState {
            entries: vec![frame_entry("x", 8, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let acc = a
            .union(&b, &mut graph)
            .expect("first fold step (carry-through) must succeed");
        assert_eq!(
            graph.next_value(),
            100,
            "carry-through step must not allocate"
        );
        let acc = acc
            .union(&c, &mut graph)
            .expect("disagreeing vids must produce a fresh phi without TypeMismatch");
        assert_eq!(
            acc.entries,
            vec![Some(crate::model::FrameStateEntry {
                name: "x".to_string(),
                value_id: ValueId(100),
                value_type: ValueType::Int,
            })]
        );
        assert_eq!(
            graph.next_value(),
            101,
            "disagreement step allocates exactly one vid"
        );
    }

    #[test]
    fn frame_state_union_fold_kills_slot_if_any_arm_unbound() {
        // `framestate.py:110-111` None-kill propagates through the
        // iterative fold: once any fold step encounters a side with
        // `None` at slot `i`, the merged slot is None for the rest of
        // the chain.  Here slot 1 (`only_ab`) is bound on a + b but
        // unbound on c — the second fold step collapses it to None.
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int),
                frame_entry("only_ab", 2, ValueType::Int),
            ],
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 1, ValueType::Int),
                frame_entry("only_ab", 2, ValueType::Int),
            ],
        };
        let c = FrameState {
            entries: vec![frame_entry("survivor", 1, ValueType::Int)],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let acc = a
            .union(&b, &mut graph)
            .expect("type-compatible slots must union cleanly")
            .union(&c, &mut graph)
            .expect("type-compatible slots must union cleanly");
        assert_eq!(acc.entries.len(), 2);
        assert_eq!(
            acc.entries[0].as_ref().map(|e| e.name.as_str()),
            Some("survivor"),
        );
        assert!(
            acc.entries[1].is_none(),
            "any-None slot collapses to None-kill across the fold chain"
        );
    }

    #[test]
    fn frame_state_union_preserves_graph_wide_slot_order() {
        // Both predecessors share the same graph-wide first-bind order
        // [c=0, a=1, b=2] (the order they were first bound anywhere in
        // the function).  Their dense `entries` vectors line up
        // positionally — slot 0 = c, slot 1 = a, slot 2 = b — so the
        // positional zip merges each slot with its name-mate.  The
        // merged state preserves this slot order.
        let a = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int),
                frame_entry("a", 1, ValueType::Int),
                frame_entry("b", 2, ValueType::Int),
            ],
        };
        let b = FrameState {
            entries: vec![
                frame_entry("c", 3, ValueType::Int),
                frame_entry("a", 1, ValueType::Int),
                frame_entry("b", 2, ValueType::Int),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = a
            .union(&b, &mut graph)
            .expect("type-compatible slots must union cleanly");
        let order: Vec<&str> = merged
            .entries
            .iter()
            .filter_map(|e| e.as_ref().map(|e| e.name.as_str()))
            .collect();
        assert_eq!(
            order,
            vec!["c", "a", "b"],
            "merged slot order must follow graph-wide first-bind order"
        );
    }

    #[test]
    fn frame_state_getoutputargs_walks_target_in_slot_order() {
        // RPython `framestate.py:92 getoutputargs` walks the target
        // (merged) state's slot order and picks the corresponding
        // self-side ValueId at each position.  Pyre's analogue is a
        // direct positional zip — `target.entries[i]` lines up with
        // `self.entries[i]` because both share the graph-wide
        // first-bind slot order.
        let pred = FrameState {
            entries: vec![
                frame_entry("a", 10, ValueType::Int),
                frame_entry("b", 20, ValueType::Int),
                frame_entry("c", 30, ValueType::Int),
            ],
        };
        let other = FrameState {
            entries: vec![
                frame_entry("a", 10, ValueType::Int),
                frame_entry("b", 99, ValueType::Int),
                frame_entry("c", 30, ValueType::Int),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = pred
            .union(&other, &mut graph)
            .expect("type-compatible slots must union cleanly");
        let link_args = pred.getoutputargs(&merged);
        assert_eq!(
            link_args,
            vec![ValueId(10), ValueId(20), ValueId(30)],
            "getoutputargs must yield self's ValueIds in target slot order"
        );
    }

    #[test]
    fn frame_state_getoutputargs_skips_none_killed_slots_positionally() {
        // After None-kill at union, `merged.entries[i]` is `None` at the
        // killed slot; `getoutputargs` walks positionally and skips
        // those, preserving the source ValueIds at surviving slots
        // without a name lookup.
        let a = FrameState {
            entries: vec![
                frame_entry("survivor", 11, ValueType::Int),
                frame_entry("only_a", 22, ValueType::Int),
                None,
            ],
        };
        let b = FrameState {
            entries: vec![
                frame_entry("survivor", 11, ValueType::Int),
                None,
                frame_entry("only_b", 33, ValueType::Int),
            ],
        };
        let mut graph = crate::model::FunctionGraph::new("test");
        graph.set_next_value(100);
        let merged = a
            .union(&b, &mut graph)
            .expect("type-compatible slots must union cleanly");
        let link_args = a.getoutputargs(&merged);
        assert_eq!(
            link_args,
            vec![ValueId(11)],
            "only surviving slots emit link args; None-killed slots are skipped"
        );
    }

    #[test]
    fn loop_body_locals_excludes_closure_captures() {
        // Slice 5b.2 contract: the static pre-scan reaches every
        // straight-line statement in the loop body (including nested
        // blocks, `if` / `match` / nested loops, and `unsafe` blocks)
        // but does NOT descend into `Expr::Closure` bodies.  A name
        // referenced only inside a closure must not appear in either
        // `read_names` or `rebound_names`, because the closure's
        // capture is not part of the outer loop's straight-line
        // control flow that drives header phi allocation.
        let body: syn::Block = syn::parse_quote! {{
            let local_let = 0;
            outer_assign = 1;
            outer_compound += 2;
            let _read_via_path = outer_read;
            let _capture = || {
                closure_only_read;
                closure_only_assigned = 7;
            };
        }};
        let result = loop_body_locals(&body);

        // `let local_let = 0` and `let _read_via_path = outer_read`
        // both rebind their pattern names; `_capture` is also a
        // pattern name on a closure-bound `let`.
        assert!(result.rebound_names.contains("local_let"));
        assert!(result.rebound_names.contains("_read_via_path"));
        assert!(result.rebound_names.contains("_capture"));
        // Simple `outer_assign = 1` rebinds `outer_assign`.
        assert!(result.rebound_names.contains("outer_assign"));
        // Compound `outer_compound += 2` is BOTH read and rebound.
        assert!(result.read_names.contains("outer_compound"));
        assert!(result.rebound_names.contains("outer_compound"));
        // `outer_read` appears as the RHS of a `let` init — read.
        assert!(result.read_names.contains("outer_read"));

        // Closure body must be fully invisible to the pre-scan.
        assert!(
            !result.read_names.contains("closure_only_read"),
            "closure body reads must be excluded from outer loop pre-scan"
        );
        assert!(
            !result.rebound_names.contains("closure_only_assigned"),
            "closure body rebinds must be excluded from outer loop pre-scan"
        );
    }

    #[test]
    fn allocate_loop_header_phis_eager_install_and_pre_loop_link_args() {
        // Slice 5b.3 contract: the eager allocator must (a) emit one
        // `OpKind::Input` per surviving name at `header_entry` and
        // append its phi vid to `header_entry.inputargs`, (b) push the
        // pre-loop vid onto `pre_loop_block.exits[0].args` so the
        // forward-edge `Link.args` arity matches the new header arity,
        // and (c) rewire `ctx.local_value_ids[name]` to point at the
        // freshly-allocated phi vid (with `header_entry` as the new
        // defining block) so subsequent body lowering reads the phi.
        //
        // The walk filters by intersecting the must-merge set with
        // `pre_loop_snapshot.entries`: a pre-loop name not referenced
        // in the body is skipped (no header phi), and a body-only
        // name (referenced but absent from the snapshot) is killed
        // per RPython `framestate.py:110-111` None-kill semantics.
        let mut graph = FunctionGraph::new("loop_header_phi_demo");
        let pre_loop_block = graph.startblock;
        let header_entry = graph.create_block();

        // Seed pre-loop bindings for `x` and `y` by emitting
        // `OpKind::Input` at the start block; this approximates the
        // pre-loop state Slice 5c.1 will hand to the allocator.
        let pre_x = graph
            .push_op(
                pre_loop_block,
                OpKind::Input {
                    name: "x".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .expect("OpKind::Input must produce a vid");
        let pre_y = graph
            .push_op(
                pre_loop_block,
                OpKind::Input {
                    name: "y".into(),
                    ty: ValueType::Int,
                },
                true,
            )
            .expect("OpKind::Input must produce a vid");
        // Close pre-loop block with the empty-args goto Slice 5c.1
        // installs before calling the allocator.
        graph.set_goto(pre_loop_block, header_entry, vec![]);

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
        ctx.bind_local_id("x".into(), pre_x, pre_loop_block);
        ctx.local_value_types.insert("x".into(), ValueType::Int);
        ctx.bind_local_id("y".into(), pre_y, pre_loop_block);
        ctx.local_value_types.insert("y".into(), ValueType::Int);

        // `pre_loop_snapshot` is produced by ctx in the real lowering;
        // mirror that here so the allocator walks the same first-bind
        // positional order Slice 5c.1 will feed it.
        let pre_loop_snapshot = ctx.snapshot_locals_for_link();
        assert_eq!(pre_loop_snapshot.entries.len(), 2);

        // `x` is read inside the body, `z` is body-only (rebound
        // without a pre-loop counterpart).  `y` is in the pre-loop
        // snapshot but never referenced in the body.
        let must_merge = LoopBodyLocals {
            read_names: ["x".to_string()].into_iter().collect(),
            rebound_names: ["z".to_string()].into_iter().collect(),
        };

        let header_phi_names = allocate_loop_header_phis(
            &mut graph,
            &mut ctx,
            pre_loop_block,
            header_entry,
            &pre_loop_snapshot,
            &must_merge,
        );

        // Only `x` survives — `y` filtered out (not referenced), `z`
        // filtered out (None-killed: not in pre-loop snapshot).
        assert_eq!(header_phi_names, vec!["x".to_string()]);

        let header = graph.block(header_entry);
        assert_eq!(header.inputargs.len(), 1);
        let phi_vid = header.inputargs[0];
        assert_eq!(header.operations.len(), 1);
        let phi_op = &header.operations[0];
        match &phi_op.kind {
            OpKind::Input { name, ty } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, ValueType::Int);
            }
            other => panic!("expected OpKind::Input, got {:?}", other),
        }
        assert_eq!(phi_op.result, Some(phi_vid));

        let pre_exit = &graph.block(pre_loop_block).exits[0];
        assert_eq!(
            pre_exit.args,
            vec![LinkArg::Value(pre_x)],
            "forward-edge link arg for `x` must carry the pre-loop vid"
        );

        let (current_x_vid, current_x_block) = ctx.local_value_ids["x"];
        assert_eq!(
            current_x_vid, phi_vid,
            "ctx.local_value_ids[x] must point at the header phi"
        );
        assert_eq!(
            current_x_block, header_entry,
            "ctx.local_value_ids[x].defining_block must be the header"
        );

        // `y` was not referenced, so its ctx binding still points at
        // the pre-loop vid; no header phi was allocated for it.
        let (current_y_vid, _) = ctx.local_value_ids["y"];
        assert_eq!(current_y_vid, pre_y);
    }

    /// Slice 5d nested-pattern coverage #1: nested while loops where
    /// the inner loop's `continue` re-enters the inner header.  The
    /// outer header's phi for the outer counter must remain
    /// independent of the inner control flow — every link landing on
    /// the outer header must carry exactly the names the outer
    /// header's `LoopFrame::header_phi_names` recorded, and every
    /// link landing on the inner header must carry the inner's.
    #[test]
    fn nested_loops_per_header_phi_arity_consistent() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut i: i64 = 0;
                let mut j: i64 = 0;
                while i < 10 {
                    j = 0;
                    while j < 5 {
                        j = j + 1;
                        if j == 3 { continue; }
                    }
                    i = i + 1;
                }
                i + j
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Every block whose `inputargs` is non-empty corresponds to a
        // loop header.  For each such header, every predecessor edge
        // must carry exactly `inputargs.len()` link args.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(
                            exit.args.len(),
                            arity,
                            "predecessor {:?}→header {:?}: link.args len {} ≠ \
                             header.inputargs len {}",
                            pred.id,
                            header.id,
                            exit.args.len(),
                            arity,
                        );
                    }
                }
            }
        }
    }

    /// A name rebound in the loop body but never read (neither in the
    /// body nor post-loop) gets a transient loop-header phi during
    /// lowering, then `prune_dead_phis` removes that phi and its matching
    /// link-arg before downstream passes see the graph.  This mirrors
    /// `flowcontext.py:430 mergeblock` + `simplify.transform_dead_op_vars`.
    #[test]
    fn rebound_only_no_post_read_skips_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut x: i64 = 0;
                let mut count: i64 = 0;
                while count < n {
                    count = count + 1;
                    x = 5;
                }
                count
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // The single loop header is the block with non-empty inputargs
        // that is not the start / return / except.
        let header = graph
            .blocks
            .iter()
            .find(|b| {
                !b.inputargs.is_empty()
                    && b.id != graph.startblock
                    && b.id != graph.returnblock
                    && b.id != graph.exceptblock
            })
            .expect("loop header must exist");
        let header_phi_names: Vec<&str> = header
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Input { name, .. } if header.inputargs.contains(&op.result.unwrap()) => {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            header_phi_names.contains(&"count"),
            "`count` is read in the body, so its header phi must exist; got {:?}",
            header_phi_names,
        );
        assert!(
            !header_phi_names.contains(&"x"),
            "`x` is rebound-only-no-read; prune_dead_phis must remove the \
             transient header phi before the final graph. Got {:?}",
            header_phi_names,
        );
    }

    /// A name rebound in the body that is read post-loop (but not
    /// inside the body) must still get a header phi.
    /// `flowcontext.py:430 mergeblock`'s `union` of pre-loop and
    /// body-tail framestates reports `NeedsPhi` (different vids),
    /// creating the phi at union time.  Pyre's read∪rebound
    /// allocator emits the phi at the header eagerly; the post-loop
    /// reader of `x` resolves to the eager phi vid.
    #[test]
    fn rebound_only_with_post_loop_read_lazy_installs_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut x: i64 = 0;
                let mut count: i64 = 0;
                while count < n {
                    count = count + 1;
                    x = 5;
                }
                x + count
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        let header = graph
            .blocks
            .iter()
            .find(|b| {
                !b.inputargs.is_empty()
                    && b.id != graph.startblock
                    && b.id != graph.returnblock
                    && b.id != graph.exceptblock
            })
            .expect("loop header must exist");
        let header_phi_names: Vec<&str> = header
            .operations
            .iter()
            .filter_map(|op| match &op.kind {
                OpKind::Input { name, .. } if header.inputargs.contains(&op.result.unwrap()) => {
                    Some(name.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(
            header_phi_names.contains(&"count"),
            "`count` is read in the body; got {:?}",
            header_phi_names,
        );
        assert!(
            header_phi_names.contains(&"x"),
            "`x` is rebound in the body and read post-loop; the eager \
             allocator must emit a header phi for it. Got {:?}",
            header_phi_names,
        );
    }

    /// Nested while loops where the inner loop reads outer-loop
    /// locals.  `loop_body_locals`'s recursive AST visit propagates
    /// the inner loop's reads to the outer body's `read_names`, so
    /// both loops allocate header phis for shared locals
    /// consistently.  Positive-path probe — the existing
    /// `nested_loops_per_header_phi_arity_consistent` covers the
    /// arity invariant; this one pins the
    /// `read_names ∪ rebound_names` allocation scope.
    #[test]
    fn nested_while_inner_reads_outer_local_threads_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let mut outer: i64 = 0;
                let mut inner: i64 = 0;
                while outer < n {
                    inner = 0;
                    while inner < outer {
                        inner = inner + 1;
                    }
                    outer = outer + 1;
                }
                outer + inner
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Two non-special blocks should carry inputargs (the two loop
        // headers).  Both must include `inner` and `outer` per
        // `read_names` (inner loop's `inner < outer` reads outer; outer
        // loop's `while outer < n` reads outer; both reads propagate).
        // A "loop header" is any block whose inputargs include a phi
        // for `outer` AND has a back-edge predecessor (a predecessor
        // whose block id is greater — back-edges close to lower-id
        // blocks).  At least two such blocks must exist (outer and
        // inner loop headers).
        let mut loop_headers_with_outer = 0;
        for header in &graph.blocks {
            let has_outer_phi = header.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "outer")
                    && op
                        .result
                        .map(|r| header.inputargs.contains(&r))
                        .unwrap_or(false)
            });
            if !has_outer_phi {
                continue;
            }
            let has_back_edge = graph.blocks.iter().any(|pred| {
                pred.id.0 > header.id.0 && pred.exits.iter().any(|e| e.target == header.id)
            });
            if has_back_edge {
                loop_headers_with_outer += 1;
            }
        }
        assert!(
            loop_headers_with_outer >= 2,
            "expected at least two loop headers carrying `outer` phi; got {}",
            loop_headers_with_outer,
        );
    }

    /// Cat 2-2 Phase A.2 probe: closure inside the loop body must NOT
    /// influence header phi allocation.  `visit_expr_for_loop_locals`
    /// explicitly skips `Expr::Closure`; a name read only inside a
    /// closure body should not appear in `read_names` and therefore
    /// should not get a header phi.  RPython parity: a closure
    /// compiles to its own bytecode with `LOAD_DEREF` / `STORE_DEREF`,
    /// distinct from the outer `LOAD_FAST` / `STORE_FAST` sequence, so
    /// captures do not flow through the loop's straight-line control
    /// path.
    ///
    /// Positive-path version: reads of a captured-only name inside a
    /// closure body do NOT count as read on the loop's header.  We
    /// pin the negative — closure-only-captured name is absent from
    /// `read_names` and therefore not a phi candidate.
    #[test]
    fn closure_captured_name_excluded_from_loop_header_phi() {
        // `loop_body_locals` is the unit under test; build a body
        // directly without going through `build_semantic_program` so
        // the probe stays orthogonal to the rest of the pipeline.
        let body: syn::Block = syn::parse_quote! {
            {
                let _f = || {
                    captured_only
                };
                straight_local = 1;
            }
        };
        let result = loop_body_locals(&body);
        assert!(
            !result.read_names.contains("captured_only"),
            "`captured_only` read only inside the closure body must not \
             appear in read_names; got {:?}",
            result.read_names,
        );
        assert!(
            result.rebound_names.contains("straight_local"),
            "`straight_local` rebound on the body's straight path must \
             appear in rebound_names; got {:?}",
            result.rebound_names,
        );
    }

    /// Cat 2-2 Phase A.2 probe: Rust lexical shadowing — the audit's
    /// primary Phase B motivator.  `let x: i64 = count + 100` inside
    /// a loop body shadows the pre-loop `let x: i64 = 7` for the
    /// duration of the body's scope, but pyre's `loop_body_locals`
    /// AST scan collapses both into the single name `"x"`.  Under
    /// the current shape, the loop header allocates a phi for `x`
    /// (because the body rebinds + reads `x`), and the back-edge
    /// close threads the inner `x`'s vid through the same slot as
    /// the outer pre-loop binding.  The post-loop read of outer `x`
    /// then resolves to the merged loop-header phi instead of the
    /// pre-loop `ConstInt(7)`.
    ///
    /// Marked `#[ignore]` because the current shape DOES NOT match
    /// Rust semantics: the test documents the gap until Phase B
    /// (back-edge worklist + scope-aware tracking) closes it.
    /// Removing `#[ignore]` is the regression-anchor handshake when
    /// Phase B lands.
    ///
    /// The Phase B post-fix expectation: NO loop-header phi for `x`
    /// exists in the final graph, because the inner `let x` lives in
    /// its own lexical scope that ends at the body block boundary
    /// and therefore has no back-edge.
    #[test]
    #[ignore = "Cat 2-2 Phase B: lexical shadowing not yet supported \
                — current AST scan collapses outer/inner `x` into one \
                slot, so the loop header allocates a phi for `x`."]
    fn lexical_shadowing_in_loop_body_does_not_emit_x_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let x: i64 = 7;
                let mut count: i64 = 0;
                while count < n {
                    let x: i64 = count + 100;
                    count = count + x;
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Phase B post-fix expectation: no block carries a phi named
        // `x`.  Today the loop header carries one because the AST
        // scan collapses both `let x` bindings into the same name.
        let any_x_phi = graph.blocks.iter().any(|b| {
            b.operations.iter().any(|op| {
                matches!(&op.kind, OpKind::Input { name, .. } if name == "x")
                    && op.result.map(|r| b.inputargs.contains(&r)).unwrap_or(false)
            })
        });
        assert!(
            !any_x_phi,
            "Phase B post-fix: inner `let x` is a lexically-scoped \
             shadow that must not produce a loop-header phi for `x`; \
             currently the AST scan collapses outer/inner into one slot.",
        );
    }

    /// Slice 5d nested-pattern coverage #2: `break` from a deeply
    /// nested `if` arm whose framestate has been mutated by a sibling
    /// arm's `LocalBindingSnapshot::restore`.  The lowering must close
    /// the break source's block with a goto to the loop's exit and the
    /// resulting graph must build without panic — `break` to a
    /// loop-exit block does not need header-phi args (exit's
    /// inputargs are determined by post-loop reads, threaded by the
    /// lazy cross-block installer).
    #[test]
    fn break_from_nested_if_arm_lowers_without_panic() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                while x < 100 {
                    if x > 50 {
                        if x % 2 == 0 {
                            break;
                        } else {
                            x = x + 1;
                        }
                    } else {
                        x = x + 2;
                    }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Sanity: header-arity invariant from #1 must still hold even
        // with a nested break.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(exit.args.len(), arity);
                    }
                }
            }
        }
    }

    /// Slice 5d nested-pattern coverage #3: a `loop` with NO `break`
    /// (back-edge always closes; exit is dead).  The lowering should
    /// succeed; the unreachable `*block = exit` post-loop is left for
    /// the simplifier.  RPython parity for "endless" loops:
    /// `flowspace/flowcontext.py:1378` — every `for`/`while`/`loop`
    /// without an exit signal builds a graph whose post-loop block is
    /// pruned by `simplify`.  This test does NOT assert pruning (that
    /// is `simplify_graph`'s contract), only that `build_semantic_
    /// program` succeeds.
    #[test]
    fn loop_with_no_break_lowers_without_panic() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                loop { x = x + 1; }
            }
        "#,
        );
        let _program = build_semantic_program(&parsed).expect("source must lower");
    }

    /// Slice 5d nested-pattern coverage #4: a `loop` where the only
    /// exit is `break` — body_tail is closed by the break (or the
    /// `is_open` check skips an empty back-edge).  Exit is reachable
    /// post-loop.  Smoke test for the back-edge `is_open` guard at
    /// `Expr::Loop`'s tail close in `front/ast.rs::lower_expr`.
    #[test]
    fn loop_with_only_break_lowers_without_panic() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example() -> i64 {
                let mut x: i64 = 0;
                loop {
                    x = x + 1;
                    if x > 10 { break; }
                }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let graph = &program
            .functions
            .iter()
            .find(|f| f.name == "example")
            .expect("example graph")
            .graph;
        // Header-arity invariant must still hold.
        for header in &graph.blocks {
            if header.inputargs.is_empty() {
                continue;
            }
            let arity = header.inputargs.len();
            for pred in &graph.blocks {
                for exit in &pred.exits {
                    if exit.target == header.id {
                        assert_eq!(exit.args.len(), arity);
                    }
                }
            }
        }
    }

    /// Audit Cat 2-1 probe: a `while` loop whose body contains
    /// `continue`, with a pre-loop local read after the loop that the
    /// body never touches.  Lazy installer must thread the post-loop
    /// read all the way back to the pre-loop binding via the loop
    /// header — and through each header predecessor (pre-loop edge,
    /// body-tail back-edge, AND continue back-edge).  RPython parity:
    /// `flowspace/flowcontext.py:399-465 mergeblock` requires every
    /// closing predecessor of the merge target to carry a FrameState
    /// so `getoutputargs` can resolve the target's inputargs slot-by-
    /// slot.  Pre-fix, body_tail and continue source both have
    /// `framestate = None` (intentional cycle breaker), so the
    /// installer's `framestate.as_ref()?` short-circuit aborts at
    /// the header's predecessor walk and the post-loop read drops to
    /// the naked-`Input` fallback (disconnected from pre-loop's vid).
    #[test]
    fn while_with_continue_threads_post_loop_read_through_header_phi() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(n: i64) -> i64 {
                let y: i64 = 7;
                let mut count: i64 = 0;
                while count < n {
                    if count % 2 == 0 {
                        count = count + 1;
                        continue;
                    }
                    count = count + 1;
                }
                y
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
        // Find the return block: ends with `Return` and reads `y`.
        let (returnblock_id, _ret_arg) = graph.returnblock_arg();
        // The closing predecessor of the returnblock is the post-loop
        // exit block that produces the return value (`y`).  In the
        // canonical Slice-2 lazy-install path, the post-loop block
        // owns `y` as an inputarg threaded back through the loop
        // header phi via every header predecessor (pre-loop, body-
        // tail, continue source).  In the pre-fix path, the
        // installer short-circuits at the header predecessor walk
        // because body_tail and continue source both have
        // `framestate = None`, and the post-loop block emits a naked
        // `OpKind::Input { name: "y" }` op as a synthetic disconnected
        // Variable.
        //
        // Probe: walk the returnblock's predecessor link.  Its single
        // arg is the post-loop block's `y` value.  Verify that vid is
        // an inputarg of the post-loop block (i.e. lazy install
        // succeeded).  Pre-fix, the vid would be the result of a
        // standalone `OpKind::Input` op NOT in inputargs.
        let pred_link: Option<(BlockId, ValueId)> = graph.blocks.iter().find_map(|b| {
            b.exits.iter().find_map(|exit| {
                if exit.target == returnblock_id {
                    let arg_vid = match &exit.args[0] {
                        crate::model::LinkArg::Value(v) => *v,
                        _ => return None,
                    };
                    Some((b.id, arg_vid))
                } else {
                    None
                }
            })
        });
        let (post_loop_id, ret_value_vid) =
            pred_link.expect("returnblock must have one closing predecessor");
        let post_loop_block = graph.block(post_loop_id);
        let is_inputarg = post_loop_block.inputargs.contains(&ret_value_vid);
        assert!(
            is_inputarg,
            "post-loop block must own `y` as an inputarg threaded back to \
             pre-loop; got naked-`Input` fallback (vid {:?} not in \
             inputargs {:?}). graph:\n{}",
            ret_value_vid,
            post_loop_block.inputargs,
            graph.dump()
        );
        // Audit Cat 2-1 also stamps continue + body_tail framestate
        // on the SOURCE blocks of every back-edge link into the loop
        // header.  Confirm directly that every header predecessor
        // has a stamped framestate; pre-fix at least one of them
        // (the continue source AND the body_tail) would be `None`,
        // and the residual fallback at the post-loop block was the
        // observable consequence.
        let header_id = graph
            .blocks
            .iter()
            .find_map(|b| {
                let owns_count_phi = b.operations.iter().any(|op| {
                    matches!(&op.kind, OpKind::Input { name, .. } if name == "count")
                        && op.result.is_some_and(|r| b.inputargs.contains(&r))
                });
                let pred_count = graph
                    .blocks
                    .iter()
                    .flat_map(|p| p.exits.iter().filter(|e| e.target == b.id).map(|_| ()))
                    .count();
                (owns_count_phi && pred_count >= 2).then_some(b.id)
            })
            .expect("loop header with `count` phi + ≥2 preds");
        let header_pred_ids: Vec<BlockId> = graph
            .blocks
            .iter()
            .filter_map(|b| {
                b.exits
                    .iter()
                    .any(|e| e.target == header_id)
                    .then_some(b.id)
            })
            .collect();
        for pred_id in &header_pred_ids {
            let pred = graph.block(*pred_id);
            assert!(
                pred.framestate.is_some(),
                "header predecessor block {:?} must have framestate \
                 stamped (continue + body_tail Cat 2-1 stamps); \
                 graph:\n{}",
                pred_id,
                graph.dump()
            );
        }
    }

    /// `rlib/jit.py:184-201 elidable_promote.decorator(func)` produces
    /// **two** function objects after module import: the closure-
    /// captured `func` (which carries `_elidable_function_`) and the
    /// `exec`-built `result` (which does not).  Pyre's parser-level
    /// synthesizer is the line-by-line mirror: a single
    /// `#[elidable_promote] fn foo(...)` source item must produce two
    /// `SemanticFunction`s — `_orig_foo_unlikely_name` carrying the
    /// `elidable` hint, and the user-facing wrapper `foo` whose body
    /// is `hint_promote(arg); …; _orig_foo_unlikely_name(args)`.
    /// `rlib/jit.py:191` evaluates `args[int(i)]` in
    /// `elidable_promote.decorator`, which raises `IndexError` when
    /// the literal points past the function's argument list.  Pyre
    /// mirrors the fail-loud behaviour through
    /// `synthesize_elidable_promote_pair`'s index bounds check.
    #[test]
    #[should_panic(expected = "out of range")]
    fn elidable_promote_panics_on_out_of_range_promote_index() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "0,5")]
            pub fn foo(x: i64) -> i64 { x }
        "#,
        );
        let _ = build_semantic_program(&parsed);
    }

    /// `rlib/jit.py:189-191 promote_args.split(",")` propagates
    /// `ValueError` on a non-integer piece via `int(i)`.  Pyre matches
    /// that by panicking on a `usize::parse` failure rather than
    /// silently dropping the malformed literal.
    #[test]
    #[should_panic(expected = "elidable_promote")]
    fn elidable_promote_panics_on_malformed_promote_args() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "0,not_a_number")]
            pub fn foo(x: i64) -> i64 { x }
        "#,
        );
        let _ = build_semantic_program(&parsed);
    }

    /// `rlib/jit.py:184-201 elidable_promote.decorator(func)` applies
    /// uniformly to any callable — module-level fn, instance method,
    /// classmethod.  Pyre's `synthesize_or_passthrough` must produce
    /// the (`_orig_<NAME>_unlikely_name`, wrapper) pair for impl
    /// methods too, not just `Item::Fn`s, otherwise method callers
    /// silently drop the elidable flag.
    #[test]
    fn elidable_promote_on_inherent_method_synthesizes_pair() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        // Two SemanticFunctions emitted from the decorated method —
        // the synthesized orig and the wrapper.
        let names: Vec<_> = program
            .functions
            .iter()
            .map(|sf| sf.name.as_str())
            .collect();
        assert!(
            names.contains(&"_orig_double_unlikely_name"),
            "expected synthesized orig in {names:?}"
        );
        assert!(
            names.contains(&"double"),
            "expected wrapper retaining user-facing name in {names:?}"
        );
        // jit.py:185 — elidable flag lives on the orig only.
        let orig = program
            .functions
            .iter()
            .find(|sf| sf.name == "_orig_double_unlikely_name")
            .expect("orig graph");
        assert!(orig.hints.iter().any(|h| h == "elidable"));
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        assert!(!wrapper.hints.iter().any(|h| h == "elidable"));
    }

    /// `lib.rs:531-537` registers inherent-method graphs under
    /// `CallPath::for_impl_method(impl_type, name)` which produces
    /// `[<impl_type segments...>, name]`.  The synthesized wrapper's
    /// tail call must therefore be a type-qualified path so the IR
    /// `Call`-target segments match the registered callee.  A bare
    /// `_orig_<name>_unlikely_name(self, args)` would resolve to a
    /// non-existent free function and silently drop the elidable
    /// binding bound to the orig graph.
    #[test]
    fn elidable_promote_wrapper_tail_call_is_type_qualified_for_inherent_method() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        let target = ops
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied())
                        == Some("_orig_double_unlikely_name") =>
                {
                    Some(target.clone())
                }
                _ => None,
            })
            .expect("wrapper must tail-call _orig_double_unlikely_name");
        let segments = target
            .path_segments()
            .expect("function-path tail call must carry path segments");
        assert_eq!(
            segments,
            vec!["S", "_orig_double_unlikely_name"],
            "wrapper tail call must be type-qualified \
             [\"S\", \"_orig_double_unlikely_name\"] to match \
             CallPath::for_impl_method; got {segments:?}"
        );
    }

    #[test]
    fn elidable_promote_synthesizes_orig_and_wrapper_graphs() {
        let parsed = crate::parse::parse_source(
            r#"
            #[elidable_promote(promote_args = "all")]
            pub fn foo(x: i64, y: i64) -> i64 {
                x + y
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(
            program.functions.len(),
            2,
            "expected exactly two graphs after synthesis, got {:?}",
            program
                .functions
                .iter()
                .map(|sf| sf.name.as_str())
                .collect::<Vec<_>>()
        );

        // jit.py:185 elidable(func) — the renamed original carries the
        // `elidable` hint that downstream `mark_elidable` reads.
        let orig = program
            .functions
            .iter()
            .find(|sf| sf.name == "_orig_foo_unlikely_name")
            .expect("orig graph");
        assert!(
            orig.hints.iter().any(|h| h == "elidable"),
            "orig hints must include 'elidable', got {:?}",
            orig.hints
        );

        // jit.py:198-201 — wrapper `result` has no `_elidable_function_`.
        // After Slice C retired the `collect_jit_hints` fallback the
        // wrapper's hints are empty unless the user explicitly added
        // another decorator alongside `#[elidable_promote]`.
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "foo")
            .expect("wrapper graph");
        assert!(
            !wrapper.hints.iter().any(|h| h == "elidable"),
            "wrapper must not carry the 'elidable' hint — RPython places \
             `_elidable_function_` only on the orig (jit.py:185), not \
             on the wrapper (jit.py:198-201).  Got hints: {:?}",
            wrapper.hints
        );
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        // jit.py:192-194 — one `hint(arg, promote=True)` per selected
        // arg.  With promote_args="all" and two args, exactly two
        // hint_promote calls.
        let hint_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote")
                )
            })
            .count();
        assert_eq!(
            hint_count, 2,
            "wrapper must emit hint_promote per arg; ops:\n{ops:#?}"
        );
        // jit.py:195 — tail call to the renamed original.
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target.path_segments().and_then(|s| s.last().copied()) == Some("_orig_foo_unlikely_name")
            )),
            "wrapper must tail-call _orig_foo_unlikely_name; ops:\n{ops:#?}"
        );
    }

    /// `rlib/jit.py:186, 191` — `_get_args(func)` reads `co_varnames`
    /// raw, so `self` is at index 0 and `promote_args='all'` covers it.
    /// Pyre can't shadow `self` with `let self = ...`, so the
    /// synthesizer routes the receiver through a fresh
    /// `__self_promoted` local and rewrites the tail call accordingly.
    /// The wrapper graph must emit one `hint_promote` per argument
    /// including the receiver.
    #[test]
    fn elidable_promote_promotes_self_receiver_for_inherent_method() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            impl S {
                #[elidable_promote(promote_args = "all")]
                pub fn double(&self, n: i64) -> i64 { n + n }
            }
        "#,
        );
        let program = build_semantic_program(&parsed).expect("source must lower");
        let wrapper = program
            .functions
            .iter()
            .find(|sf| sf.name == "double")
            .expect("wrapper graph");
        let ops = &wrapper.graph.block(wrapper.graph.startblock).operations;
        let hint_count = ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.kind,
                    OpKind::Call { target, .. } if target.path_segments().and_then(|s| s.last().copied()) == Some("hint_promote")
                )
            })
            .count();
        assert_eq!(
            hint_count, 2,
            "wrapper must emit hint_promote for self and n (2 total); ops:\n{ops:#?}"
        );
    }
}
