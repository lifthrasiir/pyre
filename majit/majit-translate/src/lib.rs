//! majit-translate: RPython translation pipeline.
//!
//! Upstream counterparts:
//! - `jit_codewriter/` ← `rpython/jit/codewriter/`
//! - `translate_legacy/` — pre-roadmap ad-hoc annotator+rtyper, deleted at P8.11
//! - `flowspace/`, `annotator/`, `rtyper/`, `translator/` — future line-by-line
//!   ports of `rpython/{flowspace,annotator,rtyper,translator}/`
//!
//! Everything under one crate because upstream has circular imports
//! (flowspace/operation.py ↔ annotator, annotator/annrpython.py ↔
//! rtyper/normalizecalls.py) that Cargo's DAG crate boundary cannot model.

pub mod annotator;
pub mod config;
pub mod flowspace;
pub mod jit_codewriter;
pub mod tool;
pub use jit_codewriter::{
    assembler, call, codewriter, flatten, format, jitcode, jtransform, liveness, policy, regalloc,
    support,
};

mod codegen;
pub mod front;
// Phase F: pyre-interpreter handler JitCode registry
// (PRE-EXISTING-ADAPTATION per parity rule #1 — Rust source → FunctionGraph
// bridge with no RPython counterpart; upstream assumes rtyper-produced
// `translator.graphs` is already in memory at codewriter entry).
pub mod generated;
pub mod handler_spec;
pub mod hints;
pub mod inline;
pub mod layout;
pub mod model;
pub mod opcode_dispatch;
mod parse;
pub mod pipeline;
#[cfg(test)]
mod test_support;
// Phase 0 P1 (roadmap: line-by-line port of rpython/{flowspace,annotator,rtyper}):
// `translator/` renamed to `translate_legacy/` because this tree is the
// pre-roadmap ad-hoc majit implementation, not a line-by-line port of
// `rpython/translator/`. The identifier `translator` is reserved for that
// future port; legacy consumers must spell `translate_legacy::…`
// explicitly. The legacy tree is deleted at roadmap commit P8.11.
pub mod translate_legacy;

// `translator/` is the non-LEGACY home for RPython-orthodox port files —
// see `translator/mod.rs` for the contract. Currently hosts
// `translator/rtyper/{rclass.rs, rpbc.rs}`, the `rpython/rtyper/` 1:1
// port. The matching `TypeResolutionState` infra still lives under
// `translator_legacy/rtyper/rtyper.rs` until majit-rtyper Phase 6.
pub mod translator;

pub use call::{CallDescriptor, StructFieldLayout, StructLayout};
pub use flatten::{FlatOp, Label, RegKind, SSARepr, flatten, flatten_with_types};
pub use front::{
    AstGraphOptions, SemanticFunction, SemanticProgram, build_semantic_program,
    build_semantic_program_from_parsed_files,
};
pub use jit_codewriter::annotation_state::AnnotationState;
pub use jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
pub use jtransform::{
    CallEffectKind, CallEffectOverride, GraphTransformConfig, GraphTransformResult,
    VirtualizableFieldDescriptor, rewrite_graph,
};
pub use layout::{HeuristicLayoutProvider, LayoutProvider};
pub use model::{
    Block, BlockId, CallTarget, FunctionGraph, OpKind, SpaceOperation, ValueId, ValueType,
};
pub use opcode_dispatch::PipelineOpcodeArm;
pub use parse::{
    CallPath, ExtractedHandlerCall, ExtractedOpcodeArm, OpcodeDispatchSelector, ParsedInterpreter,
    ReceiverTraitBindings, extract_inherent_impl_methods, extract_opcode_dispatch_arms,
    extract_opcode_dispatch_receiver_traits, extract_trait_impls, find_opcode_dispatch_match,
    parse_source,
};
pub use pipeline::{PipelineConfig, PipelineResult, PortalSpec, ProgramPipelineResult};
pub use translate_legacy::annotator::annrpython::annotate as annotate_graph;
pub use translate_legacy::pipeline::{analyze_function, analyze_program};
pub use translate_legacy::rtyper::rtyper::resolve_types;

use serde::{Deserialize, Serialize};

use crate::translate_legacy::{
    annotator::annrpython as annotate, pipeline as legacy_pipeline, rtyper::rtyper as rtype,
};

/// Configuration for the canonical graph/pipeline analyzer.
///
/// Consumers supply graph-rewrite metadata such as virtualizable
/// field/array mappings before the codewriter-style passes run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalyzeConfig {
    pub pipeline: PipelineConfig,
}

/// A resolved method/function call with source context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedCall {
    pub name: String,
    pub impl_type: Option<String>,
    pub trait_name: Option<String>,
    /// Canonical semantic graph for this resolved target.
    #[serde(skip, default)]
    pub graph: Option<model::FunctionGraph>,
}

/// Trait implementation info
#[derive(Debug, Serialize, Deserialize)]
pub struct TraitImplInfo {
    pub trait_name: String,
    pub for_type: String,
    #[serde(default)]
    pub self_ty_root: Option<String>,
    pub methods: Vec<MethodInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MethodInfo {
    pub name: String,
    /// Canonical semantic graph for this method when available.
    #[serde(skip, default)]
    pub graph: Option<model::FunctionGraph>,
    /// RPython: op.result.concretetype — return type for array identity.
    #[serde(default)]
    pub return_type: Option<String>,
    /// RPython: function-level JIT hints (elidable, close_stack, etc.).
    #[serde(default)]
    pub hints: Vec<String>,
}

/// Canonical single-file analysis entry point.
pub fn analyze_pipeline(source: &str) -> pipeline::ProgramPipelineResult {
    analyze_pipeline_with_config(source, &AnalyzeConfig::default())
}

/// Configurable canonical single-file analysis entry point.
pub fn analyze_pipeline_with_config(
    source: &str,
    config: &AnalyzeConfig,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(&[source], config)
}

/// Canonical multi-file analysis entry point.
///
/// This returns only the graph/pipeline result and is the preferred API for
/// RPython-like translator consumers.
pub fn analyze_multiple_pipeline(sources: &[&str]) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_config(sources, &AnalyzeConfig::default())
}

/// Configurable canonical multi-file analysis entry point.
///
/// This is the canonical graph/pipeline translator entry point.
/// Uses `HeuristicLayoutProvider` for struct layouts (type-string approximation).
pub fn analyze_multiple_pipeline_with_config(
    sources: &[&str],
    config: &AnalyzeConfig,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(&parsed_files, config, None, &|_, _| None, &[], &[])
}

/// Multi-file analysis with explicit layout provider.
///
/// RPython equivalent: the translator resolves struct layouts via
/// `symbolic.get_field_token()` / `symbolic.get_size()`. The layout
/// provider supplies these values. Pass `None` to use the heuristic default.
pub fn analyze_multiple_pipeline_with_layout(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: &dyn layout::LayoutProvider,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        Some(layout_provider),
        &|_, _| None,
        &[],
        &[],
    )
}

/// `make_virtualizable_infos` constructor closure type — mirrors the
/// upstream `VirtualizableInfo(self, VTYPEPTR)` call (warmspot.py:543).
/// `(jd_idx, vtypeptr_token) -> Option<handle>`.  Hosts that own a
/// runtime `VirtualizableInfo` impl supply the constructor here.
pub type VirtualizableInfoFactory<'a> =
    dyn Fn(usize, &str) -> Option<std::sync::Arc<dyn call::VirtualizableInfoHandle>> + 'a;

/// Optional binding table from macro-generated helper path
/// (`module_path!()::helper_name`) to the compiled trace-call address.
///
/// `#[jit_module]::__majit_helper_trace_fnaddrs()` produces this shape.
/// `analyze_pipeline_from_parsed` strips the crate-name prefix and binds
/// both canonical aliases (`helpers::foo` and `crate::helpers::foo`) on
/// `CallControl` before `get_jitcode()` / `jtransform` query fnaddrs.
pub type FnAddrBindings<'a> = [(&'a str, i64)];

/// Structured binding table for impl-method helpers.  Each entry is
/// `(module_path_with_crate, impl_type_as_written, method_name, fnaddr)`.
/// The codewriter applies the parser's `qualify_type_name` rule
/// (front/ast.rs:106) — bare types get the module prefix (minus crate
/// name) prepended, qualified types are kept verbatim — before storing
/// the canonical `[impl_type_joined, method]` 2-segment CallPath
/// (lib.rs:406-433).
///
/// `#[jit_module]::__majit_helper_impl_trace_fnaddrs()` produces this
/// shape and `analyze_pipeline_from_parsed` feeds it through
/// `CallControl::register_macro_impl_helper_trace_fnaddr`.
pub type ImplFnAddrBindings<'a> = [(&'a str, &'a str, &'a str, i64)];

/// Multi-file analysis with explicit layout provider AND a
/// `VirtualizableInfo` factory wired into
/// `CallControl::make_virtualizable_infos` (warmspot.py:516).  The
/// factory delegates the runtime constructor call from the codewriter
/// (which sits below metainterp in the crate graph) back to the host.
pub fn analyze_multiple_pipeline_with_vinfo_factory(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        layout_provider,
        vinfo_factory,
        &[],
        &[],
    )
}

/// Multi-file analysis with explicit layout provider, optional
/// `VirtualizableInfo` factory, and host-supplied compiled helper
/// addresses.
///
/// This is the line-by-line `getfunctionptr(graph)` adapter for source-only
/// codewriter consumers: pass the output of
/// `#[jit_module]::__majit_helper_trace_fnaddrs()` here so `JitCode.fnaddr`
/// and residual-call lowering use the real helper surface instead of the
/// symbolic fallback.
pub fn analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_vinfo_and_all_fnaddr_bindings(
        sources,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        &[],
    )
}

/// Like `analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings` but
/// additionally accepts an `impl_fnaddr_bindings` table produced by the
/// macro's `__majit_helper_impl_trace_fnaddrs()` registry. Entries bind
/// impl-method helpers via `register_macro_impl_helper_trace_fnaddr`,
/// resolving the structural `[impl_type_joined, method]` CallPath that
/// the string-split helper entry point cannot express.
pub fn analyze_multiple_pipeline_with_vinfo_and_all_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    let parsed_files: Vec<_> = sources.iter().map(|s| parse::parse_source(s)).collect();
    analyze_pipeline_from_parsed(
        &parsed_files,
        config,
        layout_provider,
        vinfo_factory,
        fnaddr_bindings,
        impl_fnaddr_bindings,
    )
}

/// Multi-file analysis with compiled helper fnaddr bindings but without a
/// virtualizable-info factory.
pub fn analyze_multiple_pipeline_with_fnaddr_bindings(
    sources: &[&str],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    fnaddr_bindings: &FnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings(
        sources,
        config,
        layout_provider,
        &|_, _| None,
        fnaddr_bindings,
    )
}

fn analyze_pipeline_from_parsed(
    parsed_files: &[parse::ParsedInterpreter],
    config: &AnalyzeConfig,
    layout_provider: Option<&dyn layout::LayoutProvider>,
    vinfo_factory: &VirtualizableInfoFactory<'_>,
    fnaddr_bindings: &FnAddrBindings<'_>,
    impl_fnaddr_bindings: &ImplFnAddrBindings<'_>,
) -> pipeline::ProgramPipelineResult {
    // RPython `translator/translator.py:55 buildflowgraph` — FlowingError
    // propagates out and translation halts.  Pyre's top-level analyzer
    // requires a complete program; a FlowingError here means a user-
    // facing source file contains a construct we cannot yet lower, and
    // the correct response is to abort loudly so the coverage audit
    // surfaces the unsupported expression rather than silently dropping
    // a graph.
    let program = front::build_semantic_program_from_parsed_files(parsed_files)
        .expect("pyre-interpreter source must lower without FlowingError");
    // legacy_pipeline::analyze_program is now invoked after
    // `call_control.find_all_graphs` so the BFS-from-portal candidate
    // set drives the per-function annotate → rtype → jtransform →
    // flatten chain.  RPython parity: `annrpython.py:215
    // schedule_pending` walks pending blocks reachable from the
    // entry-point closure; unreachable helpers stay outside the
    // translator and never trip the strict-typing assertions the JIT
    // path enforces.  See the
    // `legacy_pipeline::analyze_program_filtered` call site below.
    let mut canonical_trait_impls = Vec::new();
    let mut canonical_inherent_methods = Vec::new();
    let mut canonical_function_graphs = std::collections::HashMap::new();

    for parsed in parsed_files {
        canonical_trait_impls.extend(
            parse::extract_trait_impls(
                parsed,
                &program.struct_fields,
                &program.fn_return_types,
                &program.known_struct_names,
            )
            .expect("trait impls must lower without FlowingError"),
        );
        canonical_inherent_methods.extend(
            parse::extract_inherent_impl_methods(
                parsed,
                &program.struct_fields,
                &program.fn_return_types,
                &program.known_struct_names,
            )
            .expect("inherent methods must lower without FlowingError"),
        );
    }
    // RPython: use the rtyped graphs (with concretetype info) for all analysis.
    // Use program.functions' graphs which were built with full struct_fields
    // context, NOT re-parsed graphs (which lose array_type_id etc.).
    for func in &program.functions {
        if func.self_ty_root.is_none() {
            // Free function: register under qualified segments.
            // func.name may be "a::helper" for module-internal functions.
            let segments: Vec<&str> = func.name.split("::").collect();
            let path = parse::CallPath::from_segments(segments.iter().copied());
            canonical_function_graphs.insert(path, func.graph.clone());
            // Also register under ["crate", ...segments].
            let mut crate_segs = vec!["crate"];
            crate_segs.extend(segments.iter().copied());
            let crate_path = parse::CallPath::from_segments(crate_segs);
            canonical_function_graphs.insert(crate_path, func.graph.clone());
        }
    }

    // ── Build CallControl (RPython call.py) ──
    // Populate with all discovered function graphs and trait impl methods.
    let mut call_control = call::CallControl::new();
    // RPython: known struct types for get_type_flag(ARRAY.OF) → FLAG_STRUCT.
    call_control.set_known_struct_names(program.known_struct_names.clone());
    // RPython: struct field types for op.args[0].concretetype resolution.
    call_control.set_struct_fields(program.struct_fields.clone());
    // RPython: symbolic.get_field_token / get_size — resolve struct layouts
    // through the LayoutProvider. If no provider is given, use the heuristic
    // (type-string-based approximation of #[repr(C)] layout).
    let heuristic;
    let provider: &dyn layout::LayoutProvider = match layout_provider {
        Some(p) => p,
        None => {
            heuristic = layout::HeuristicLayoutProvider::from_struct_fields(
                &program.struct_fields.fields,
                &program.known_struct_names,
                &program.immutable_fields,
            );
            &heuristic
        }
    };
    // RPython: per-class `_immutable_fields_` declaration. Drives
    // `FieldDescr.is_pure` for the heuristic-fallback path inside
    // `all_interiorfielddescrs` (the layout-provider path already carries
    // `is_immutable` on `StructFieldLayout`).
    call_control.immutable_fields_by_struct = program.immutable_fields.clone();
    // Populate CallControl with layouts from the provider.
    for struct_name in program.struct_fields.fields.keys() {
        if let Some(layout) = provider.get_struct_layout(struct_name) {
            call_control.set_struct_layout(struct_name.clone(), layout);
        }
    }
    // Register graphs collected above (free functions only — trait
    // methods are handled separately via register_trait_method).
    for (path, graph) in &canonical_function_graphs {
        call_control.register_function_graph(path.clone(), graph.clone());
    }
    // Re-register free functions with their RPython-equivalent hints
    // (`elidable`, `loop_invariant`, `unroll_safe`, `jit_look_inside`)
    // so `JitPolicy::look_inside_graph` sees the same metadata RPython
    // reads off `func._jit_*_` / `_elidable_function_`.
    for func in &program.functions {
        if !func.self_ty_root.is_none() || func.hints.is_empty() {
            continue;
        }
        let segments: Vec<&str> = func.name.split("::").collect();
        let path = parse::CallPath::from_segments(segments.iter().copied());
        call_control.register_function_graph_with_hints(
            path,
            func.graph.clone(),
            func.hints.clone(),
        );
        let mut crate_segs = vec!["crate"];
        crate_segs.extend(segments.iter().copied());
        let crate_path = parse::CallPath::from_segments(crate_segs);
        call_control.register_function_graph_with_hints(
            crate_path,
            func.graph.clone(),
            func.hints.clone(),
        );
    }
    // RPython: op.result.concretetype — register return types per function.
    // Each function's return type is registered under its exact canonical path(s).
    // No name-based reverse lookup — avoids collision between same-named
    // functions in different modules.
    {
        for func in &program.functions {
            let Some(ref ret_type) = func.return_type else {
                continue;
            };
            if let Some(ref owner) = func.self_ty_root {
                // impl method: ["owner", "method_name"]
                let path =
                    crate::parse::CallPath::from_segments([owner.as_str(), func.name.as_str()]);
                call_control.return_types.insert(path, ret_type.clone());
            } else {
                // free function: register under qualified segments.
                let segments: Vec<&str> = func.name.split("::").collect();
                let path = crate::parse::CallPath::from_segments(segments.iter().copied());
                call_control.return_types.insert(path, ret_type.clone());
                let mut crate_segs = vec!["crate"];
                crate_segs.extend(segments.iter().copied());
                let crate_path = crate::parse::CallPath::from_segments(crate_segs);
                call_control
                    .return_types
                    .insert(crate_path, ret_type.clone());
            }
        }
    }
    for impl_info in &canonical_trait_impls {
        let impl_type = impl_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&impl_info.for_type);
        // RPython parity: `trait_root=Some(trait_name)` for real trait impls,
        // `None` for inherent impls (impl SomeType { ... } without `for Trait`).
        // `parse.rs:237` always writes `trait_name`; a sentinel empty string
        // from the inherent branch (see parse.rs:357-389) needs special-casing.
        let trait_root = if impl_info.trait_name.is_empty() {
            None
        } else {
            Some(impl_info.trait_name.as_str())
        };
        for method in &impl_info.methods {
            if let Some(graph) = &method.graph {
                call_control.register_trait_method(
                    &method.name,
                    trait_root,
                    impl_type,
                    graph.clone(),
                );
                // Parity with upstream `rpython/annotator/classdesc.py:749
                // lookup` MRO walk: a trait default body is the
                // "base-class method" for every impl that does not
                // override it. Rust-idiomatic call sites emit the call
                // as `<Trait>::<method>(receiver, ...)` —
                // `front/ast.rs::canonical_call_target` turns that into
                // `CallTarget::FunctionPath { segments: [<Trait>,
                // <method>] }`. The upstream-equivalent registration key
                // is therefore `[<Trait>, <method>]`. The pseudo-type
                // path `[<default methods of Trait>, <method>]` set by
                // `register_trait_method` is retained so the filter logic
                // at `call.rs:1921,1970 resolve_method*` and
                // `lib.rs:935 push_matching_trait_methods` can continue
                // to distinguish "trait default" from "concrete impl".
                if impl_info.for_type.starts_with("<default methods of ") {
                    let direct_path = crate::parse::CallPath::from_segments([
                        impl_info.trait_name.as_str(),
                        method.name.as_str(),
                    ]);
                    call_control.register_function_graph(direct_path, graph.clone());
                }
            }
            // RPython: op.result.concretetype for trait/default method calls.
            let path = crate::parse::CallPath::for_impl_method(impl_type, method.name.as_str());
            if let Some(ref ret_type) = method.return_type {
                call_control
                    .return_types
                    .insert(path.clone(), ret_type.clone());
            }
            // RPython: hints bound to graph identity.
            for hint in &method.hints {
                match hint.as_str() {
                    "elidable" => call_control.mark_elidable(path.clone()),
                    "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                    "close_stack" => call_control.mark_close_stack(path.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                    "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                    _ => {}
                }
            }
        }
    }
    // RPython: direct_call → funcobj.graph — inherent methods are resolved
    // by direct graph linkage, not name-based trait method lookup.
    // Register only in function_graphs under qualified path (Type::method).
    for method_info in &canonical_inherent_methods {
        let impl_type = method_info
            .self_ty_root
            .as_deref()
            .unwrap_or(&method_info.for_type);
        let path = crate::parse::CallPath::for_impl_method(impl_type, method_info.name.as_str());
        call_control.register_function_graph(path.clone(), method_info.graph.clone());
        if let Some(ref ret_type) = method_info.return_type {
            call_control
                .return_types
                .insert(path.clone(), ret_type.clone());
        }
        // RPython: hints bound to graph identity.
        for hint in &method_info.hints {
            match hint.as_str() {
                "elidable" => call_control.mark_elidable(path.clone()),
                "loopinvariant" => call_control.mark_loopinvariant(path.clone()),
                "close_stack" => call_control.mark_close_stack(path.clone()),
                "cannot_collect" => call_control.mark_cannot_collect(path.clone()),
                "gc_effects" => call_control.mark_external_gc_effects(path.clone()),
                _ => {}
            }
        }
    }
    for &(full_path, fnaddr) in fnaddr_bindings {
        call_control.register_macro_helper_trace_fnaddr(full_path, fnaddr);
    }
    for &(module_path_with_crate, impl_type_as_written, method, fnaddr) in impl_fnaddr_bindings {
        call_control.register_macro_impl_helper_trace_fnaddr(
            module_path_with_crate,
            impl_type_as_written,
            method,
            fnaddr,
        );
    }
    // RPython: GC transformer sets _gctransformer_hint_close_stack_,
    // _gctransformer_hint_cannot_collect_ on functions, and
    // random_effects_on_gcobjs on external function objects.
    // In majit, detect these from #[jit_close_stack], #[jit_cannot_collect],
    // #[jit_gc_effects] attributes on functions in the parsed source.
    // RPython: hints are bound to the function/graph object, not the name.
    // Register under ALL canonical paths so any call-site lookup finds them.
    for func in &program.functions {
        if func.hints.is_empty() {
            continue;
        }
        // Build all canonical paths for this function.
        let mut paths = Vec::new();
        if let Some(ref owner) = func.self_ty_root {
            // impl method: ["owner", "method"]
            paths.push(crate::parse::CallPath::from_segments([
                owner.as_str(),
                func.name.as_str(),
            ]));
        } else {
            // free function: register under qualified segments + crate:: alias
            let segments: Vec<&str> = func.name.split("::").collect();
            paths.push(crate::parse::CallPath::from_segments(
                segments.iter().copied(),
            ));
            let mut crate_segs = vec!["crate"];
            crate_segs.extend(segments.iter().copied());
            paths.push(crate::parse::CallPath::from_segments(crate_segs));
        }
        for hint in &func.hints {
            for p in &paths {
                // rlib/jit.py:250 — `@oopspec(spec)` registers func.oopspec = spec.
                if let Some(spec) = hint.strip_prefix("oopspec:") {
                    call_control.mark_oopspec(p.clone(), spec.to_string());
                    continue;
                }
                match hint.as_str() {
                    "elidable" => call_control.mark_elidable(p.clone()),
                    "loopinvariant" => call_control.mark_loopinvariant(p.clone()),
                    "close_stack" => call_control.mark_close_stack(p.clone()),
                    "cannot_collect" => call_control.mark_cannot_collect(p.clone()),
                    // rlib/jit.py:260 — @not_in_trace sets func.oopspec = "jit.not_in_trace()"
                    "not_in_trace" => {
                        call_control.mark_oopspec(p.clone(), "jit.not_in_trace".to_string());
                    }
                    // RPython: random_effects_on_gcobjs is on external funcobj only.
                    // Only register for paths WITHOUT a graph (external functions).
                    "gc_effects" => {
                        if !call_control.function_graphs().contains_key(p) {
                            call_control.mark_external_gc_effects(p.clone());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // RPython: setup_jitdriver(jitdriver_sd) — register portal + green/red layout.
    // PyPy interp_jit.py: greens = ['next_instr', 'is_being_profiled', 'pycode'],
    //                      reds = ['frame', 'ec'], virtualizables = ['frame']
    // Callers can override the portal binding via `PipelineConfig::portal`.
    //
    // Default portal identity: prefer `eval_loop_jit`
    // (pyre-jit/src/eval.rs:1187), the pyre analogue of upstream's
    // `warmspot.py::portal_runner` and the single graph seeded into
    // `find_all_graphs(portal, policy)` at `call.py:57`. When the graph
    // set does not include pyre-jit/src/eval.rs (e.g. compact test
    // inputs whose PYRE_JIT_GRAPH_SOURCES stops at pyre-interpreter),
    // fall back to `execute_opcode_step` so those tests retain a portal
    // target. `execute_opcode_step` itself is a handler reached from the
    // real portal's match arm, so seeding BFS from it treats a handler
    // as an entry point — tolerable only for the legacy test
    // configurations that have no `eval_loop_jit` at all; once those
    // tests feed the full Phase D0 source set the fallback is never
    // exercised and the eval_loop_jit-only identity locks in.
    let default_portal_name = {
        let eval_loop_jit_path = parse::CallPath::from_segments(["eval_loop_jit"]);
        if call_control
            .function_graphs()
            .contains_key(&eval_loop_jit_path)
        {
            "eval_loop_jit"
        } else {
            "execute_opcode_step"
        }
    };
    let (portal_name, portal_greens, portal_reds, portal_virtualizables, portal_red_types) =
        match &config.pipeline.portal {
            Some(spec) => (
                spec.name.clone(),
                spec.greens.clone(),
                spec.reds.clone(),
                spec.virtualizables.clone(),
                spec.red_types.clone(),
            ),
            None => (
                default_portal_name.to_string(),
                vec![
                    "next_instr".to_string(),
                    "is_being_profiled".to_string(),
                    "pycode".to_string(),
                ],
                vec!["frame".to_string(), "ec".to_string()],
                // PyPy interp_jit.py: virtualizables = ['frame'] —
                // jitdriver.virtualizables drives warmspot.py:527-545
                // make_virtualizable_infos selection.
                vec!["frame".to_string()],
                // jit.py / interp_jit.py: red types parallel to reds.
                // 'frame' is the virtualizable PyFrame; 'ec' is the
                // ExecutionContext (the wrapping struct in pyre).
                vec!["PyFrame".to_string(), "ExecutionContext".to_string()],
            ),
        };
    let portal = parse::CallPath::from_segments([portal_name.as_str()]);
    if call_control.function_graphs().contains_key(&portal) {
        call_control.setup_jitdriver(
            portal,
            portal_greens,
            portal_reds,
            portal_virtualizables,
            portal_red_types,
        );
        // warmspot.py:515-545 WarmRunnerDesc.make_virtualizable_infos —
        // assigns jd.index_of_virtualizable, builds GreenFieldInfo
        // when any green name contains '.', and delegates the upstream
        // `VirtualizableInfo(self, VTYPEPTR)` constructor call
        // (warmspot.py:543) to `vinfo_factory`.  Hosts entering through
        // `analyze_multiple_pipeline_with_vinfo_factory` supply a real
        // closure that builds the metainterp-side `VirtualizableInfo`
        // (e.g. via `majit_metainterp::virtualizable::VirtualizableInfo::
        // build_for(VTYPEPTR)`); the default entry points pass
        // `|_, _| None` so the codewriter slot stays empty until
        // `MetaInterp::set_virtualizable_info` (jitdriver.rs:285) wires
        // it at runtime.
        call_control.make_virtualizable_infos(|jd_idx, vtypeptr_token| {
            vinfo_factory(jd_idx, vtypeptr_token)
        });
    }
    // Mark known builtins (elidable helpers).
    // RPython: detected via funcobj.graph.func.oopspec attribute.
    for builtin_name in &[
        "w_int_add",
        "w_int_sub",
        "w_int_mul",
        "w_float_add",
        "w_float_sub",
    ] {
        let path = parse::CallPath::from_segments([*builtin_name]);
        if call_control.function_graphs().contains_key(&path) {
            call_control.mark_builtin(path);
        }
    }
    // Register oopspecs for jit.* builtin functions.
    // rlib/jit.py: these functions carry @oopspec("jit.*") decorators;
    // the codewriter converts calls to them into dedicated opcodes.
    for (func_name, spec) in &[
        // rlib/jit.py:269-292 — @oopspec("jit.*") decorated functions
        ("isconstant", "jit.isconstant"),
        ("isvirtual", "jit.isvirtual"),
        ("current_trace_length", "jit.current_trace_length"),
        ("jit_debug", "jit.debug"),
        ("assert_green", "jit.assert_green"),
        // NOTE: conditional_call!/conditional_call_elidable!/record_known_result!
        // are handled by jitcode_lower (proc-macro level), not here.
        // They are macro_rules! that the codewriter AST parser does not expand.
    ] {
        // Register under common call-site path patterns.
        for path in [
            parse::CallPath::from_segments(["jit", func_name]),
            parse::CallPath::from_segments(["crate", "jit", func_name]),
            parse::CallPath::from_segments(["majit_metainterp", "jit", func_name]),
        ] {
            call_control.mark_oopspec(path, spec.to_string());
        }
    }
    let mut policy = policy::DefaultJitPolicy::new();
    call_control.find_all_graphs(&mut policy);

    // RPython parity: `RPythonAnnotator.build_types(entry_point, args)`
    // only analyses functions reachable from the entry-point closure.
    // Pyre's legacy whole-program iteration ran annotate / rtype /
    // jtransform / flatten on every parsed function, including
    // runtime-only resume helpers (`materialize_virtual_from_rd`,
    // `replay_pending_fields`) the JIT never traces.  Filtering by
    // `call_control.is_candidate` mirrors annrpython's BFS-from-entry
    // shape so unreachable helpers stay outside the analyser the same
    // way `annotator/translator/translator.py:55 buildflowgraph`
    // skips them upstream.
    let mut pipeline =
        legacy_pipeline::analyze_program_filtered(&program.functions, &config.pipeline, |func| {
            function_is_candidate_for_analysis(func, &call_control)
        });

    let (opcode_dispatch, jitcodes, insns, descrs) = build_canonical_opcode_dispatch(
        parsed_files,
        &canonical_trait_impls,
        &canonical_inherent_methods,
        &canonical_function_graphs,
        &config.pipeline,
        &mut call_control,
    );
    pipeline.opcode_dispatch = opcode_dispatch;
    pipeline.jitcodes = jitcodes;
    // Mirror of `CallControl::jitcodes` (RPython `call.py:87 self.jitcodes`)
    // captured before `call_control` is dropped. Needed because consumers
    // that look up a JitCode by graph identity cannot reconstruct the key
    // from the alloc-ordered `pipeline.jitcodes` vector alone.
    pipeline.jitcodes_by_path = call_control.jitcodes().clone();
    pipeline.insns = insns;
    pipeline.descrs = descrs;

    pipeline
}

/// Decide whether `func` falls inside the BFS-from-portal closure that
/// the canonical jitcode emitter consumes.  RPython's
/// `RPythonAnnotator.build_types` schedules pending graphs from
/// `entry_point` outwards via `annrpython.py:215 schedule_pending`;
/// pyre mirrors that closure with `call_control.candidate_graphs`,
/// populated by `find_all_graphs_bfs`.  We probe both unprefixed and
/// `crate::`-prefixed forms because `register_function_graph` records
/// free functions under both keys (see the `for func in
/// &program.functions` loop earlier in this function).  Impl methods
/// surface as `(self_ty_root, method_name)` paths via
/// `canonical_trait_impls` registration; the same two-segment lookup
/// covers both inherent and trait impls.
fn function_is_candidate_for_analysis(
    func: &front::SemanticFunction,
    call_control: &call::CallControl,
) -> bool {
    if let Some(ref owner) = func.self_ty_root {
        let path = parse::CallPath::from_segments([owner.as_str(), func.name.as_str()]);
        if call_control.is_candidate(&path) {
            return true;
        }
        let crate_path =
            parse::CallPath::from_segments(["crate", owner.as_str(), func.name.as_str()]);
        return call_control.is_candidate(&crate_path);
    }
    let segments: Vec<&str> = func.name.split("::").collect();
    let path = parse::CallPath::from_segments(segments.iter().copied());
    if call_control.is_candidate(&path) {
        return true;
    }
    let mut crate_segs = vec!["crate"];
    crate_segs.extend(segments.iter().copied());
    let crate_path = parse::CallPath::from_segments(crate_segs);
    call_control.is_candidate(&crate_path)
}

/// Build opcode dispatch arms and produce JitCodes for discovered callee graphs.
///
/// RPython parity (`rpython/jit/codewriter/codewriter.py:74-89` `make_jitcodes`):
///
/// ```python
/// def make_jitcodes(self, verbose=False):
///     self.callcontrol.grab_initial_jitcodes()
///     all_jitcodes = []
///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
///         all_jitcodes.append(jitcode)
///     self.assembler.finished(self.callcontrol.callinfocollection)
///     return all_jitcodes
/// ```
///
/// PyPy registers each opcode handler as its own Python method, so PyPy's
/// codewriter naturally gets one jitcode per opcode via the discovery loop.
/// pyre's interpreter dispatches inside one big match instead of separate
/// methods, so we register each match arm body as a synthetic graph here
/// (`CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`) and the
/// orthodox `drain_pending_graphs` loop picks them up exactly the same way
/// it picks up callee graphs discovered during jtransform.
fn build_canonical_opcode_dispatch(
    parsed_files: &[parse::ParsedInterpreter],
    trait_impls: &[TraitImplInfo],
    inherent_methods: &[parse::InherentMethodInfo],
    function_graphs: &std::collections::HashMap<parse::CallPath, model::FunctionGraph>,
    pipeline_config: &pipeline::PipelineConfig,
    call_control: &mut call::CallControl,
) -> (
    Vec<opcode_dispatch::PipelineOpcodeArm>,
    Vec<std::sync::Arc<jitcode::JitCode>>,
    std::collections::HashMap<String, u8>,
    Vec<jitcode::BhDescr>,
) {
    let mut opcode_arms = Vec::new();
    let mut receiver_traits = parse::ReceiverTraitBindings::default();

    for parsed in parsed_files {
        let file_opcodes = parse::extract_opcode_dispatch_arms(parsed);
        if !file_opcodes.is_empty() {
            opcode_arms = file_opcodes;
            receiver_traits = parse::extract_opcode_dispatch_receiver_traits(parsed);
            break;
        }
    }

    // RPython codewriter.py:74-89: make_jitcodes().
    //
    // `Arc<JitCode>` shells live in `CallControl::jitcodes`; the drain loop
    // commits each shell's body via `JitCode::set_body`. After all phases,
    // `collect_jitcodes_in_alloc_order` materialises the `all_jitcodes[]`
    // vector with `all_jitcodes[i].index == i` (RPython codewriter.py:80
    // invariant).
    let mut codewriter = codewriter::CodeWriter::new();

    // Phase 1: RPython grab_initial_jitcodes + drain portal + callees.
    // RPython call.py:145-148.
    call_control.grab_initial_jitcodes();
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);

    // Phase 2: register each opcode arm body as a synthetic graph.
    //
    // PyPy's interpreter has one Python method per opcode and `find_all_graphs`
    // discovers them naturally; pyre dispatches inside one match, so we walk
    // the parser-extracted arms and call `register_function_graph` +
    // `get_jitcode` ourselves. Each arm gets a stable `arm_id` (extract order)
    // and a synthetic `CallPath::["__opcode_dispatch__", "<selector>#<arm_id>"]`
    // which decouples display label (selector) from identity (path/index).
    //
    // The parser-level `flattened` field is computed eagerly here so existing
    // snapshot tests / debug dumps keep working. The orthodox pipeline inside
    // `drain_pending_graphs` will recompute its own SSARepr+JitCode below
    // and place the result at `jitcodes[entry_jitcode_index]`.
    let mut dispatch: Vec<opcode_dispatch::PipelineOpcodeArm> =
        Vec::with_capacity(opcode_arms.len());
    let mut dispatch_paths: Vec<Option<parse::CallPath>> = Vec::with_capacity(opcode_arms.len());
    for (arm_id, arm) in opcode_arms.into_iter().enumerate() {
        let _resolved_calls = resolve_handler_calls(
            &arm.handler_calls,
            trait_impls,
            inherent_methods,
            function_graphs,
            &receiver_traits,
        );

        // Parser-level flatten — kept for snapshot tests / debug.
        let flattened = arm.body_graph.as_ref().map(|handler_graph| {
            let annotations = annotate::annotate(handler_graph);
            let mut type_state = rtype::resolve_types(handler_graph, &annotations);
            // Run the rtyper-equivalent indirect_call lowering before
            // jtransform so this snapshot path matches the canonical
            // codewriter pipeline (post-rtyper graph has no
            // `CallTarget::Indirect`). RPython rpbc.py:199-217.
            let mut handler_graph_owned = handler_graph.clone();
            crate::translator::rtyper::rpbc::lower_indirect_calls(
                &mut handler_graph_owned,
                &mut type_state,
                &mut *call_control,
            );
            #[cfg(debug_assertions)]
            crate::translator::rtyper::rpbc::assert_no_indirect_call_targets(&handler_graph_owned);
            let mut transformer = jtransform::Transformer::new(&pipeline_config.transform)
                .with_callcontrol(&mut *call_control)
                .with_type_state(&type_state);
            let rewritten = transformer.transform(&handler_graph_owned);
            let rewritten_type_state = rtype::resolve_rewritten_types(
                &type_state,
                &rewritten.graph,
                &annotations,
                &rewritten.synth_kinds,
            );
            let value_kinds =
                crate::jit_codewriter::type_state::build_value_kinds(&rewritten_type_state);
            let regallocs =
                crate::regalloc::perform_all_register_allocations(&rewritten.graph, &value_kinds);
            flatten::flatten_with_types(&rewritten.graph, &rewritten_type_state, &regallocs)
        });

        // Register the arm body in CallControl. RPython call.py:155-172
        // `get_jitcode(graph)` returns the callee object; the final
        // `jitcode.index` is assigned only after assembly completes.
        let entry_jitcode_path = arm.body_graph.map(|body_graph| {
            let synthetic_path = synthetic_opcode_arm_path(&arm.selector, arm_id);
            call_control.register_function_graph(synthetic_path.clone(), body_graph);
            let _ = call_control.get_jitcode(&synthetic_path);
            synthetic_path
        });

        dispatch.push(opcode_dispatch::PipelineOpcodeArm {
            arm_id,
            selector: arm.selector,
            entry_jitcode_index: None,
            flattened,
        });
        dispatch_paths.push(entry_jitcode_path);
    }

    // Phase 3: Drain pending graphs.
    //
    // RPython codewriter.py:79-84: `for graph, jitcode in enum_pending_graphs()`.
    // After Phase 2 every opcode arm body is on `unfinished_graphs`, plus any
    // callees discovered during the parser-level transform pass above. Each
    // gets transformed → assembled in turn, and any *new* callees they reach
    // are added to the queue and picked up by the same loop.
    codewriter.drain_pending_graphs(call_control, &pipeline_config.transform);

    // RPython codewriter.py:85: self.assembler.finished(callinfocollection).
    codewriter
        .assembler
        .finished(&call_control.callinfocollection);

    for (arm, path) in dispatch.iter_mut().zip(dispatch_paths.into_iter()) {
        if let Some(path) = path {
            let jitcode = call_control
                .jitcode_handle(&path)
                .expect("opcode arm jitcode handle must exist after registration");
            arm.entry_jitcode_index = Some(jitcode.index());
        }
    }

    // Materialise `all_jitcodes[]` from the completed jitcodes. Each
    // jitcode receives its dense index when appended, matching RPython
    // `make_jitcodes()`.
    let jitcodes = call_control.collect_jitcodes_in_alloc_order();

    // RPython codewriter.py + assembler.py: `Assembler.insns` grows as
    // `write_insn` encounters new keys.  We snapshot the final table
    // here so the runtime can map `JitCode.code[i]` bytes back to
    // opnames — the key consumed by `BlackholeInterpBuilder::setup_insns`.
    let insns = codewriter.assembler.insns().clone();

    // RPython blackhole.py:59 `self.setup_descrs(asm.descrs)` — the
    // shared descr table every 'd'/'j' argcode indexes into at runtime.
    // Snapshotted here so the build artifact carries it alongside
    // `insns`, mirroring RPython's single-store model.
    let descrs: Vec<jitcode::BhDescr> = codewriter.assembler.snapshot_descrs();

    (dispatch, jitcodes, insns, descrs)
}

/// Synthetic CallPath for an opcode-dispatch arm body.
///
/// PyPy uses `graph.name` (which is the Python method name) as the
/// debug label and `graph` object identity as the dict key in
/// `CallControl.jitcodes`. pyre uses `CallPath` as the dict key, so we
/// build a 2-segment synthetic path:
///   `["__opcode_dispatch__", "<selector_canonical>#<arm_id>"]`
/// The `#<arm_id>` suffix guarantees collision-free keys even if two
/// arms shared a selector string (parser already rejects that, but the
/// suffix makes the invariant local to this function).
fn synthetic_opcode_arm_path(
    selector: &parse::OpcodeDispatchSelector,
    arm_id: usize,
) -> parse::CallPath {
    parse::CallPath::from_segments([
        "__opcode_dispatch__".to_string(),
        format!("{}#{}", selector.canonical_key(), arm_id),
    ])
}

/// Classify a resolved call by jtransform-rewriting its graph.

fn resolve_handler_calls(
    handler_calls: &[parse::ExtractedHandlerCall],
    trait_impls: &[TraitImplInfo],
    inherent_methods: &[parse::InherentMethodInfo],
    function_graphs: &std::collections::HashMap<parse::CallPath, model::FunctionGraph>,
    receiver_traits: &parse::ReceiverTraitBindings,
) -> Vec<ResolvedCall> {
    fn push_unique(resolved_calls: &mut Vec<ResolvedCall>, call: ResolvedCall) {
        if !resolved_calls.iter().any(|existing| {
            existing.name == call.name
                && existing.impl_type == call.impl_type
                && existing.trait_name == call.trait_name
        }) {
            resolved_calls.push(call);
        }
    }

    fn receiver_matches_root(
        receiver_type_root: Option<&String>,
        candidate_root: &Option<String>,
    ) -> bool {
        match (receiver_type_root, candidate_root.as_ref()) {
            (Some(receiver), Some(candidate)) => receiver == candidate,
            _ => false,
        }
    }

    fn push_matching_trait_methods(
        resolved_calls: &mut Vec<ResolvedCall>,
        trait_impls: &[TraitImplInfo],
        name: &str,
        receiver_type_root: Option<&String>,
        allowed_traits: Option<&[String]>,
    ) -> bool {
        let mut matched = false;
        for impl_info in trait_impls {
            if let Some(allowed_traits) = allowed_traits {
                if !allowed_traits
                    .iter()
                    .any(|bound| bound == &impl_info.trait_name)
                {
                    continue;
                }
            }

            let is_default_methods = impl_info.for_type.starts_with("<default methods of ");
            let is_generic = receiver_type_root
                .is_some_and(|r| crate::call::is_generic_receiver(r))
                || receiver_type_root.is_none();
            let applies = if is_default_methods {
                is_generic
                    || receiver_type_root.is_some_and(|receiver_ty| {
                        trait_impls.iter().any(|candidate| {
                            candidate.trait_name == impl_info.trait_name
                                && candidate.self_ty_root.as_ref() == Some(receiver_ty)
                        })
                    })
            } else {
                // Concrete impls: match if the receiver type matches,
                // OR if receiver is a generic type parameter (e.g. "E", "H")
                receiver_type_root.is_some_and(|r| crate::call::is_generic_receiver(r))
                    || receiver_type_root.is_none()
                    || receiver_matches_root(receiver_type_root, &impl_info.self_ty_root)
            };

            if !applies {
                continue;
            }

            for method in &impl_info.methods {
                if method.name == name {
                    matched = true;
                    push_unique(
                        resolved_calls,
                        ResolvedCall {
                            name: name.to_string(),
                            impl_type: Some(impl_info.for_type.clone()),
                            trait_name: Some(impl_info.trait_name.clone()),
                            graph: method.graph.clone(),
                        },
                    );
                }
            }
        }
        matched
    }

    let mut resolved_calls = Vec::new();

    for call in handler_calls {
        match call {
            parse::ExtractedHandlerCall::Method {
                name,
                receiver_root,
            } => {
                let receiver_binding = receiver_root.as_ref();
                let receiver_type_root = receiver_binding
                    .and_then(|receiver| receiver_traits.type_root_by_receiver.get(receiver));
                let receiver_bound_traits = receiver_binding
                    .and_then(|receiver| receiver_traits.traits_by_receiver.get(receiver));

                let mut matched_specific_target = false;
                if receiver_type_root.is_some() {
                    for method in inherent_methods {
                        if method.name == *name
                            && receiver_matches_root(receiver_type_root, &method.self_ty_root)
                        {
                            matched_specific_target = true;
                            push_unique(
                                &mut resolved_calls,
                                ResolvedCall {
                                    name: name.clone(),
                                    impl_type: Some(method.for_type.clone()),
                                    trait_name: None,
                                    graph: Some(method.graph.clone()),
                                },
                            );
                        }
                    }

                    matched_specific_target |= push_matching_trait_methods(
                        &mut resolved_calls,
                        trait_impls,
                        name,
                        receiver_type_root,
                        receiver_bound_traits.map(Vec::as_slice),
                    );
                }

                if !matched_specific_target {
                    if let Some(bound_traits) = receiver_bound_traits {
                        for trait_name in bound_traits {
                            let mut default_match: Option<ResolvedCall> = None;
                            let mut concrete_matches = Vec::new();

                            for impl_info in trait_impls {
                                if &impl_info.trait_name != trait_name {
                                    continue;
                                }
                                for method in &impl_info.methods {
                                    if method.name != *name {
                                        continue;
                                    }
                                    let resolved = ResolvedCall {
                                        name: name.clone(),
                                        impl_type: Some(impl_info.for_type.clone()),
                                        trait_name: Some(impl_info.trait_name.clone()),
                                        graph: method.graph.clone(),
                                    };
                                    if impl_info.for_type.starts_with("<default methods of ") {
                                        default_match = Some(resolved);
                                    } else {
                                        concrete_matches.push(resolved);
                                    }
                                }
                            }

                            // Push default method first (simple delegation like
                            // `opcode_build_list(self, size).map_err(Into::into)`)
                            // so the classifier tries it before complex concrete impls.
                            if let Some(default_match) = default_match {
                                push_unique(&mut resolved_calls, default_match);
                            }
                            for concrete in concrete_matches {
                                push_unique(&mut resolved_calls, concrete);
                            }
                        }
                    }
                }
            }
            parse::ExtractedHandlerCall::FunctionPath(path) => {
                // Handle qualified trait calls like OpcodeStepExecutor::build_list(executor, ...)
                // which parse as FunctionPath(["OpcodeStepExecutor", "build_list"]).
                // Convert to a trait method resolution.
                if path.segments.len() == 2 {
                    let trait_name = &path.segments[0];
                    let method_name = &path.segments[1];
                    // Check if the first segment is a known trait name
                    let is_trait = trait_impls.iter().any(|imp| imp.trait_name == *trait_name);
                    if is_trait {
                        let allowed = &[trait_name.clone()];
                        push_matching_trait_methods(
                            &mut resolved_calls,
                            trait_impls,
                            method_name,
                            None, // generic receiver
                            Some(allowed),
                        );
                        continue; // skip the function_graphs lookup
                    }
                }
                if let Some(graph) = function_graphs.get(path) {
                    resolved_calls.push(ResolvedCall {
                        name: path.canonical_key(),
                        impl_type: None,
                        trait_name: None,
                        graph: Some(graph.clone()),
                    });
                }
            }
            parse::ExtractedHandlerCall::UnsupportedFunctionExpr => {}
        }
    }

    resolved_calls
}

/// Generate tracing code directly from the canonical pipeline result.
pub fn generate_trace_code_from_pipeline(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_pipeline(result)
}

/// Like [`generate_trace_code_from_pipeline`] but respects the selected
/// [`codegen::CodegenFlavor`]. Callers outside of pyre (e.g. `aheui-jit`)
/// opt for [`codegen::CodegenFlavor::Minimal`] to skip emission of
/// pyre-specific helpers.
pub fn generate_trace_code_from_pipeline_with_flavor(
    result: &pipeline::ProgramPipelineResult,
    flavor: codegen::CodegenFlavor,
) -> String {
    codegen::generate_from_pipeline_with_flavor(result, flavor)
}

pub use codegen::CodegenFlavor;

/// Produce a recognition report: how much the pipeline understands.
pub fn recognition_report(result: &pipeline::ProgramPipelineResult) -> codegen::RecognitionReport {
    codegen::recognition_report(result)
}
pub use codegen::{OpcodeRecognition, RecognitionReport};

/// Generate code from graph pipeline results.
#[cfg(test)]
pub fn generate_graph_code(result: &pipeline::ProgramPipelineResult) -> String {
    codegen::generate_from_graph(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use walkdir::WalkDir;

    fn read_pyre_file(name: &str) -> String {
        let base = concat!(env!("CARGO_MANIFEST_DIR"), "/../../pyre/");
        std::fs::read_to_string(format!("{base}{name}"))
            .unwrap_or_else(|_| panic!("failed to read {name}"))
    }

    fn collect_rs_files(dir: &Path, sources: &mut Vec<String>) {
        for entry in WalkDir::new(dir) {
            let entry = entry.unwrap_or_else(|_| panic!("failed to walk dir {}", dir.display()));
            let path = entry.path();
            if entry.file_type().is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                sources.push(
                    std::fs::read_to_string(path)
                        .unwrap_or_else(|_| panic!("failed to read {}", path.display())),
                );
            }
        }
    }

    fn read_all_pyre_sources() -> Vec<String> {
        let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pyre");
        let mut sources = Vec::new();
        for dir in [
            base.join("pyre-object/src"),
            base.join("pyre-interpreter/src"),
        ] {
            collect_rs_files(&dir, &mut sources);
        }
        sources
    }

    #[test]
    fn test_analyze_pyopcode() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let result = analyze_multiple_pipeline_with_config(
            &[&source],
            &crate::test_support::pyre_analyze_config(),
        );

        assert!(
            result.opcode_dispatch.len() > 20,
            "expected >20 opcode arms, got {}",
            result.opcode_dispatch.len()
        );

        eprintln!("=== Single-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        for (i, arm) in result.opcode_dispatch.iter().enumerate() {
            eprintln!(
                "  [{i}] {} → {:?}",
                arm.selector.canonical_key(),
                arm.flattened.as_ref().map(|f| f.insns.len())
            );
        }
    }

    #[test]
    fn test_multi_file_analysis() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<_> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = sources
            .iter()
            .map(|source| parse::parse_source(source))
            .collect();
        let result = analyze_multiple_pipeline_with_config(
            &source_refs,
            &crate::test_support::pyre_analyze_config(),
        );
        let empty_sf = crate::front::StructFieldRegistry::default();
        let empty_frt = std::collections::HashMap::new();
        let trait_impls: Vec<_> = parsed_files
            .iter()
            .flat_map(|p| {
                parse::extract_trait_impls(
                    p,
                    &empty_sf,
                    &empty_frt,
                    &std::collections::HashSet::new(),
                )
                .expect("trait impls must lower")
            })
            .collect();

        eprintln!("=== Multi-file Analysis ===");
        eprintln!("Opcodes: {}", result.opcode_dispatch.len());
        eprintln!("Functions: {}", result.functions.len());
        eprintln!("Trait impls: {}", trait_impls.len());

        // Should have trait impls from eval.rs (PyFrame impls)
        let pyframe_impls: Vec<_> = trait_impls
            .iter()
            .filter(|i| i.for_type.contains("PyFrame"))
            .collect();
        eprintln!("\nPyFrame trait impls: {}", pyframe_impls.len());
        for impl_info in &pyframe_impls {
            eprintln!(
                "  impl {} for PyFrame — {} methods",
                impl_info.trait_name,
                impl_info.methods.len()
            );
            for m in &impl_info.methods {
                eprintln!("    {}", m.name);
            }
        }

        // Should have resolved opcode patterns (flattened op counts)
        eprintln!("\nOpcode patterns:");
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.insns.len()
                );
            }
        }

        // Report flattened (inline→jtransform→flatten) stats
        let flattened_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        eprintln!(
            "\nFlattened (inline pipeline): {flattened_count}/{}",
            result.opcode_dispatch.len()
        );
        for arm in &result.opcode_dispatch {
            if let Some(ref flat) = arm.flattened {
                eprintln!(
                    "  {} → {} flat ops",
                    arm.selector.canonical_key(),
                    flat.insns.len()
                );
            }
        }

        // Verify canonical graph/pipeline dispatch flattens a useful subset.
        let flattened_dispatch_count = result
            .opcode_dispatch
            .iter()
            .filter(|a| a.flattened.is_some())
            .count();
        assert!(
            flattened_dispatch_count >= 10,
            "expected >=10 flattened opcode arms, got {}",
            flattened_dispatch_count
        );

        // Verify flattened arms produce non-empty op sequences.
        assert!(
            result
                .opcode_dispatch
                .iter()
                .filter_map(|arm| arm.flattened.as_ref())
                .all(|f| f.insns.len() > 0),
            "all flattened arms should have non-empty op sequences"
        );

        // RPython: CodeWriter.make_jitcodes() produces JitCode for each graph.
        // Verify the full pipeline (regalloc + liveness + assemble) runs.
        //
        // `collect_jitcodes_in_alloc_order` preserves the dense invariant
        // `all_jitcodes[i].index == i` (call.rs:1620-1633, matching RPython
        // codewriter.py:80), and by construction every slot in the vec is
        // a legitimate shell produced by `get_jitcode`. Shells whose body
        // was never committed (e.g. graph registered by a caller but never
        // drained because the test harness doesn't wire an fnaddr binding)
        // round-trip as body-less entries — `try_body()` is the documented
        // probe for distinguishing them.
        eprintln!("\nJitCodes: {}", result.jitcodes.len());
        let mut bodied = 0usize;
        for (i, jitcode) in result.jitcodes.iter().enumerate() {
            match jitcode.try_body() {
                Some(body) => {
                    eprintln!(
                        "  [{}] {} → {} bytes, regs i={} r={} f={}",
                        i,
                        jitcode.name,
                        body.code.len(),
                        body.c_num_regs_i,
                        body.c_num_regs_r,
                        body.c_num_regs_f,
                    );
                    bodied += 1;
                }
                None => eprintln!("  [{}] {} → <shell: body not committed>", i, jitcode.name),
            }
        }
        assert!(
            !result.jitcodes.is_empty(),
            "CodeWriter should produce JitCodes from opcode arms"
        );
        assert!(
            bodied > 0,
            "at least one JitCode should have a committed body"
        );
        assert!(
            result
                .jitcodes
                .iter()
                .filter_map(|jc| jc.try_body())
                .all(|body| !body.code.is_empty()),
            "every committed JitCode body must have non-empty bytecode"
        );
    }

    #[test]
    fn test_codegen_output() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<_> = sources.iter().map(String::as_str).collect();
        let result = analyze_multiple_pipeline_with_config(
            &source_refs,
            &crate::test_support::pyre_analyze_config(),
        );
        let code = generate_trace_code_from_pipeline(&result);
        let flattened_arms: Vec<_> = result
            .opcode_dispatch
            .iter()
            .filter(|arm| arm.flattened.is_some())
            .collect();

        // Should contain canonical dispatch table
        assert!(
            code.contains("CANONICAL_TRACE_PATTERNS"),
            "missing CANONICAL_TRACE_PATTERNS"
        );
        assert!(
            !code.contains("pub const TRACE_PATTERNS"),
            "canonical output should not emit legacy TRACE_PATTERNS alias"
        );
        assert!(
            code.contains("Canonical analysis summary:"),
            "missing canonical summary"
        );
        assert!(!flattened_arms.is_empty(), "expected flattened opcode arms");

        eprintln!("=== Generated Code ({} bytes) ===", code.len());
        // Print first 50 lines
        for (i, line) in code.lines().enumerate().take(50) {
            eprintln!("{:3}: {}", i + 1, line);
        }
    }

    #[test]
    fn test_recognition_report() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<_> = sources.iter().map(String::as_str).collect();
        let result = analyze_multiple_pipeline_with_config(
            &source_refs,
            &crate::test_support::pyre_analyze_config(),
        );
        let report = recognition_report(&result);

        eprintln!("=== Recognition Report ===");
        eprintln!(
            "Total opcodes: {}, Flattened: {} ({:.0}%)",
            report.total_opcodes,
            report.flattened,
            if report.total_opcodes > 0 {
                report.flattened as f64 / report.total_opcodes as f64 * 100.0
            } else {
                0.0
            }
        );
        eprintln!(
            "Total flat ops: {}, Unknown: {}, Unresolved calls: {}",
            report.total_flat_ops, report.unknown_ops, report.unresolved_calls
        );
        eprintln!("\nPer-opcode:");
        for opc in &report.per_opcode {
            let status = if opc.flat_ops > 0 {
                format!(
                    "{} ops ({}U {}C)",
                    opc.flat_ops, opc.unknowns, opc.unresolved
                )
            } else {
                "unflattened".to_string()
            };
            eprintln!("  {:40} {}", opc.selector, status);
        }

        // Scoreboard assertions
        assert!(
            report.total_opcodes > 20,
            "expected >20 opcodes, got {}",
            report.total_opcodes
        );
        assert!(
            report.flattened >= 10,
            "expected >=10 flattened, got {}",
            report.flattened
        );
    }

    #[test]
    fn test_graph_pipeline_e2e() {
        // E2E test: source → AST front-end → semantic graph → graph transform → classify
        let parsed = parse::parse_source(
            r#"
            struct Frame { next_instr: usize, locals_w: Vec<i64> }
            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
                fn store_fast(&mut self, val: i64) {
                    let idx = self.next_instr;
                    self.locals_w[idx] = val;
                }
            }
        "#,
        );

        // Step 1: AST → semantic graph
        let program = front::build_semantic_program(&parsed).expect("source must lower");
        assert_eq!(
            program.functions.len(),
            2,
            "should have load_fast + store_fast"
        );

        // Step 2: graph transform (with virtualizable config)
        let config = GraphTransformConfig {
            vable_fields: vec![VirtualizableFieldDescriptor::new(
                "next_instr",
                Some("Frame".into()),
                0,
            )],
            vable_arrays: vec![VirtualizableFieldDescriptor::new(
                "locals_w",
                Some("Frame".into()),
                0,
            )],
            ..Default::default()
        };

        let load_fast_graph = &program.functions[0].graph;
        let result = rewrite_graph(load_fast_graph, &config);
        assert!(
            result.vable_rewrites > 0,
            "load_fast should have vable rewrites, got notes: {:?}",
            result.notes
        );

        // Step 3: flatten the rewritten graph
        let types = resolve_types(&result.graph, &annotate_graph(&result.graph));
        let value_kinds = crate::jit_codewriter::type_state::build_value_kinds(&types);
        let regallocs =
            crate::regalloc::perform_all_register_allocations(&result.graph, &value_kinds);
        let flattened = flatten_with_types(&result.graph, &types, &regallocs);
        eprintln!(
            "load_fast graph ops: {:?}",
            load_fast_graph.block(load_fast_graph.startblock).operations
        );
        eprintln!("load_fast flattened: {} ops", flattened.insns.len());
        assert!(
            flattened.insns.len() > 0,
            "load_fast should produce flat ops"
        );
    }

    #[test]
    fn test_graph_pipeline_on_pyre_source() {
        // Run the full graph pipeline on actual pyre interpreter source.
        // This validates that the pipeline handles real-world Rust code.
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let parsed = parse::parse_source(&source);
        let program = front::build_semantic_program(&parsed).expect("pyopcode.rs must lower");

        let config = PipelineConfig::default();
        let result = analyze_program(&program, &config);

        eprintln!("=== Graph Pipeline on pyre-interpreter ===");
        eprintln!("Functions analyzed: {}", result.functions.len());
        eprintln!("Total blocks: {}", result.total_blocks);
        eprintln!("Total flat ops: {}", result.total_ops);

        // Should analyze many functions from the real source
        assert!(
            result.functions.len() >= 5,
            "expected >=5 functions from pyopcode.rs, got {}",
            result.functions.len()
        );

        // Should produce multi-block CFGs for functions with control flow
        let multi_block = result
            .functions
            .iter()
            .filter(|f| f.original_blocks > 1)
            .count();
        eprintln!("Functions with multi-block CFG: {multi_block}");

        // Should produce flat ops
        assert!(
            result.total_ops > 0,
            "should produce flat ops from real source"
        );
    }

    #[test]
    fn test_analyze_pipeline_runs_canonical_graph_path() {
        let source = read_pyre_file("pyre-interpreter/src/pyopcode.rs");
        let graph_result = analyze_pipeline(&source);

        // Graph pipeline should analyze functions
        assert!(graph_result.functions.len() >= 5);

        eprintln!(
            "analyze_pipeline: graph {} functions / {} blocks / {} ops",
            graph_result.functions.len(),
            graph_result.total_blocks,
            graph_result.total_ops,
        );
    }

    #[test]
    fn test_analyze_multiple_with_config_rewrites_virtualizable_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );

        let load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            load_fast.flattened.is_some(),
            "LoadFast should be flattened"
        );
        assert!(
            load_fast.flattened.as_ref().unwrap().insns.len() > 0,
            "LoadFast flattened should have ops"
        );
    }

    #[test]
    fn test_analyze_multiple_pipeline_with_config_produces_canonical_vable_dispatch() {
        let source = r#"
            enum Instruction { LoadFast }

            struct Frame {
                next_instr: usize,
                locals_w: Vec<i64>,
            }

            impl Frame {
                fn load_fast(&mut self) -> i64 {
                    let idx = self.next_instr;
                    self.locals_w[idx]
                }
            }

            fn execute_opcode_step(frame: &mut Frame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => {
                        let _ = frame.load_fast();
                    }
                }
            }
        "#;

        let result = analyze_multiple_pipeline_with_config(
            &[source],
            &AnalyzeConfig {
                pipeline: PipelineConfig {
                    transform: GraphTransformConfig {
                        vable_fields: vec![VirtualizableFieldDescriptor::new(
                            "next_instr",
                            Some("Frame".into()),
                            0,
                        )],
                        vable_arrays: vec![VirtualizableFieldDescriptor::new(
                            "locals_w",
                            Some("Frame".into()),
                            0,
                        )],
                        ..Default::default()
                    },
                    ..Default::default()
                },
            },
        );
        let canonical_load_fast = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("canonical LoadFast opcode arm");
        assert!(
            result.total_vable_rewrites > 0,
            "graph pipeline should perform vable rewrites"
        );
        assert!(
            canonical_load_fast.flattened.is_some(),
            "canonical LoadFast should be flattened"
        );
        assert!(
            canonical_load_fast.flattened.as_ref().unwrap().insns.len() > 0,
            "canonical LoadFast flattened should have ops"
        );
    }

    #[test]
    fn test_analyze_multiple_pipeline_with_fnaddr_bindings_stamps_real_jitcode_fnaddr() {
        let source = r#"
            fn helper_opaque(a: i64, b: i64) -> i64 {
                a + b
            }

            fn execute_opcode_step() -> i64 {
                helper_opaque(2, 3)
            }
        "#;

        let result = analyze_multiple_pipeline_with_fnaddr_bindings(
            &[source],
            &AnalyzeConfig::default(),
            None,
            &[("testcrate::helper_opaque", 0x1234_5678)],
        );

        let helper = result
            .jitcodes
            .iter()
            .find(|jitcode| jitcode.name == "helper_opaque")
            .expect("helper_opaque jitcode");
        assert_eq!(helper.fnaddr, 0x1234_5678);
    }

    #[test]
    fn test_opcode_dispatch_uses_trait_bound_default_method_graphs() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let result = analyze_multiple_pipeline(&[source]);
        let arm = result
            .opcode_dispatch
            .iter()
            .find(|arm| arm.selector.canonical_key() == "Instruction::LoadFast")
            .expect("LoadFast opcode arm");
        assert!(
            arm.flattened.is_some(),
            "trait-bound default method should produce a flattened result"
        );
    }

    #[test]
    fn test_generic_trait_bound_does_not_sweep_concrete_impls() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            struct PyFrame;
            impl OpcodeStepExecutor for PyFrame {}

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let inherent_methods = parse::extract_inherent_impl_methods(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("inherent methods must lower");
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs)
            .expect("free functions must lower");
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "generic trait bound should resolve only default method body"
        );
        assert_eq!(
            resolved[0].impl_type.as_deref(),
            Some("<default methods of OpcodeStepExecutor>")
        );
        assert_eq!(
            resolved[0].trait_name.as_deref(),
            Some("OpcodeStepExecutor")
        );
    }

    #[test]
    fn test_generic_trait_bound_uses_unique_concrete_impl() {
        let source = r#"
            enum Instruction { LoadFast }

            trait OpcodeStepExecutor {
                fn load_fast_checked(&mut self, idx: usize);
            }

            struct PyFrame;
            impl OpcodeStepExecutor for PyFrame {
                fn load_fast_checked(&mut self, idx: usize) {
                    let _ = idx;
                }
            }

            fn execute_opcode_step<E: OpcodeStepExecutor>(executor: &mut E, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => executor.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let inherent_methods = parse::extract_inherent_impl_methods(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("inherent methods must lower");
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs)
            .expect("free functions must lower");
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "generic trait bound should use the unique concrete impl"
        );
        assert_eq!(resolved[0].impl_type.as_deref(), Some("PyFrame"));
        assert_eq!(
            resolved[0].trait_name.as_deref(),
            Some("OpcodeStepExecutor")
        );
    }

    #[test]
    fn test_function_path_resolution_uses_exact_call_path() {
        let source = r#"
            enum Instruction { LoadFast }

            fn helper() {}

            fn execute_opcode_step(instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => crate::helper(),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let inherent_methods = parse::extract_inherent_impl_methods(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("inherent methods must lower");
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs)
            .expect("free functions must lower");
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(resolved.len(), 1, "exact crate::helper path should resolve");
        assert_eq!(resolved[0].name, "crate::helper");
        assert!(
            resolved[0].graph.is_some(),
            "resolved helper should carry a graph"
        );
    }

    #[test]
    fn test_opcode_dispatch_prefers_concrete_receiver_impl_root() {
        let source = r#"
            enum Instruction { LoadFast }

            trait TraitA {
                fn load_fast_checked(&mut self, idx: usize) { let _ = idx; }
            }
            trait TraitB {
                fn load_fast_checked(&mut self, idx: usize) { let _ = idx + 1; }
            }

            struct PyFrame;
            struct OtherFrame;

            impl TraitA for PyFrame {}
            impl TraitB for OtherFrame {}

            fn execute_opcode_step(frame: &mut PyFrame, instruction: Instruction) {
                match instruction {
                    Instruction::LoadFast => frame.load_fast_checked(0),
                }
            }
        "#;

        let parsed = parse::parse_source(source);
        let trait_impls = parse::extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let inherent_methods = parse::extract_inherent_impl_methods(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("inherent methods must lower");
        let mut function_graphs = std::collections::HashMap::new();
        parse::collect_function_graphs(&parsed, &mut function_graphs)
            .expect("free functions must lower");
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let arms = parse::extract_opcode_dispatch_arms(&parsed);
        let resolved = resolve_handler_calls(
            &arms[0].handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "expected only concrete receiver impl match"
        );
        assert_eq!(
            resolved[0].impl_type.as_deref(),
            Some("<default methods of TraitA>")
        );
        assert_eq!(resolved[0].trait_name.as_deref(), Some("TraitA"));
    }

    #[test]
    fn test_opcode_dispatch_does_not_broad_match_known_receiver_identity() {
        let source = r#"
            trait TraitA { fn helper(&mut self); }
            trait TraitB { fn helper(&mut self) {} }
            struct PyFrame;
            impl TraitA for PyFrame { fn helper(&mut self) {} }
            struct OtherFrame;
            impl TraitB for OtherFrame {}
            pub fn execute_opcode_step(frame: &mut PyFrame) { frame.helper(); }
        "#;

        let parsed = parse::parse_source(source);
        let receiver_traits = parse::extract_opcode_dispatch_receiver_traits(&parsed);
        let trait_impls = parse::extract_trait_impls(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("trait impls must lower");
        let inherent_methods = parse::extract_inherent_impl_methods(
            &parsed,
            &crate::front::StructFieldRegistry::default(),
            &std::collections::HashMap::new(),
            &std::collections::HashSet::new(),
        )
        .expect("inherent methods must lower");
        let function_graphs = std::collections::HashMap::new();
        let handler_calls = vec![parse::ExtractedHandlerCall::Method {
            name: "helper".into(),
            receiver_root: Some("frame".into()),
        }];

        let resolved = resolve_handler_calls(
            &handler_calls,
            &trait_impls,
            &inherent_methods,
            &function_graphs,
            &receiver_traits,
        );

        assert_eq!(
            resolved.len(),
            1,
            "known receiver should not broad-match unrelated impls"
        );
        assert_eq!(resolved[0].impl_type.as_deref(), Some("PyFrame"));
        assert_eq!(resolved[0].trait_name.as_deref(), Some("TraitA"));
    }

    /// Integration test: CallControl + inline on real pyre sources.
    ///
    /// Verifies that the inline pass produces graphs with low-level ops
    /// (FieldRead, ArrayRead) from inlined handler method bodies.
    #[test]
    fn test_inline_pipeline_integration() {
        let sources = read_all_pyre_sources();
        let source_refs: Vec<&str> = sources.iter().map(String::as_str).collect();
        let parsed_files: Vec<_> = source_refs.iter().map(|s| parse::parse_source(s)).collect();

        // Build CallControl from parsed sources
        let mut call_control = call::CallControl::new();
        let mut function_graphs = std::collections::HashMap::new();
        for parsed in &parsed_files {
            parse::collect_function_graphs(parsed, &mut function_graphs)
                .expect("collect_function_graphs: FlowingError must propagate");
        }
        for (path, graph) in &function_graphs {
            call_control.register_function_graph(path.clone(), graph.clone());
        }
        let empty_sf2 = crate::front::StructFieldRegistry::default();
        let empty_frt2 = std::collections::HashMap::new();
        let trait_impls: Vec<TraitImplInfo> = parsed_files
            .iter()
            .flat_map(|p| {
                parse::extract_trait_impls(
                    p,
                    &empty_sf2,
                    &empty_frt2,
                    &std::collections::HashSet::new(),
                )
                .expect("trait impls must lower")
            })
            .collect();
        for impl_info in &trait_impls {
            let impl_type = impl_info
                .self_ty_root
                .as_deref()
                .unwrap_or(&impl_info.for_type);
            let trait_root = if impl_info.trait_name.is_empty() {
                None
            } else {
                Some(impl_info.trait_name.as_str())
            };
            for method in &impl_info.methods {
                if let Some(graph) = &method.graph {
                    call_control.register_trait_method(
                        &method.name,
                        trait_root,
                        impl_type,
                        graph.clone(),
                    );
                }
            }
        }
        call_control.find_all_graphs_for_tests();

        // Get opcode_load_fast_checked graph and inline it
        let path = parse::CallPath::from_segments(["opcode_load_fast_checked"]);
        let graph = function_graphs.get(&path);
        assert!(
            graph.is_some(),
            "opcode_load_fast_checked should exist in function_graphs"
        );
        let mut graph = graph.unwrap().clone();

        let pre_inline_blocks = graph.blocks.len();
        let inlined = inline::inline_graph(&mut graph, &call_control, 3);

        eprintln!("=== Inline Integration Test ===");
        eprintln!(
            "  opcode_load_fast_checked: {pre_inline_blocks} blocks → {} blocks, {inlined} call sites inlined",
            graph.blocks.len()
        );
        for block in &graph.blocks {
            for op in &block.operations {
                eprintln!("    {:?}", op.kind);
            }
        }

        // inline.rs is a graph utility, NOT part of the RPython-orthodox
        // pipeline. Method calls are now correctly Residual (not auto-Regular),
        // so fewer call sites may be inlined. This is expected.
        eprintln!("  inlined count: {inlined}");

        // Check if any low-level ops emerged from inlining FunctionPath calls
        let all_ops: Vec<_> = graph.blocks.iter().flat_map(|b| &b.operations).collect();
        let has_low_level = all_ops.iter().any(|op| {
            matches!(
                &op.kind,
                OpKind::FieldRead { .. }
                    | OpKind::ArrayRead { .. }
                    | OpKind::ArrayWrite { .. }
                    | OpKind::FieldWrite { .. }
            )
        });
        eprintln!("  has low-level ops after inline: {has_low_level}");
    }
}

/// `rlib` — Rust port of `rpython/rlib/` helpers pulled in on demand.
/// Currently only the pieces required by the annotator port are
/// present (rarithmetic subset for `compute_restype`).
pub mod rlib;
