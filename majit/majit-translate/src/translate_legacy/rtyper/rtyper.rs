//! Type resolution pass.
//!
//! **LEGACY.** Flat `ConcreteType` enum with ad-hoc lowering.
//! Line-by-line port of `rtyper/rtyper.py:RPythonTyper` +
//! `rtyper/rmodel.py:Repr` hierarchy is landing at
//! `majit-rtyper/src/{rtyper,rmodel}.rs` (roadmap Phase 6). This file
//! is deleted at roadmap commit P8.11.
//!
//! Transforms annotated ValueTypes into concrete low-level types
//! and specializes operations accordingly.

use std::collections::HashMap;

use crate::flowspace::model::ConstValue;
use crate::jit_codewriter::annotation_state::AnnotationState;
use crate::jit_codewriter::type_state::{ConcreteType, TypeResolutionState};
use crate::model::{FunctionGraph, Link, LinkArg, OpKind, ValueId, ValueType};

/// Resolve annotations to concrete types.
///
/// RPython equivalent: `RPythonTyper.specialize_block()` — walks
/// each block and converts annotation → Repr → lowleveltype.
pub fn resolve_types(graph: &FunctionGraph, annotations: &AnnotationState) -> TypeResolutionState {
    let mut state = TypeResolutionState::new();

    for (&vid, vtype) in &annotations.types {
        let concrete = valuetype_to_concrete(vtype);
        state.concrete_types.insert(vid, concrete);
    }

    // Resolve from ops with explicit type info
    for block in &graph.blocks {
        // Resolve inputargs (Phi nodes) from annotations
        for &vid in &block.inputargs {
            if state.get(vid) == &ConcreteType::Unknown {
                let vtype = annotations.types.get(&vid).unwrap_or(&ValueType::Unknown);
                let concrete = valuetype_to_concrete(vtype);
                if concrete != ConcreteType::Unknown {
                    state.concrete_types.insert(vid, concrete);
                }
            }
        }
        for op in &block.operations {
            if let Some(result) = op.result {
                let inferred = infer_concrete_from_op(&op.kind);
                if inferred != ConcreteType::Unknown {
                    // The op carries authoritative result kind (post-
                    // jtransform `float_*` rewrite, explicit Call /
                    // FieldRead / etc. result_ty).  Override any
                    // annotation default — annrpython defaults BinOp /
                    // UnaryOp result to Int, which would shadow a later
                    // Float inference for `float_add` / `float_neg` /
                    // etc. produced by jtransform's float-operand
                    // rewrite arm.
                    state.concrete_types.insert(result, inferred);
                }
            }
        }
    }

    // Cross-block: propagate through Link args → target inputargs.
    // Keep the exception-link split explicit, mirroring upstream's
    // `_convert_link()` handling of `last_exception` /
    // `last_exc_value` before the per-arg conversion loop.
    for block in &graph.blocks {
        for link in &block.exits {
            if link_is_raise_like(link) {
                convert_raise_link(&mut state, graph, link);
            } else {
                convert_link(&mut state, graph, link);
            }
        }
    }

    // Backward-constraint: an op whose RPython decorator declares
    // integer operands (`@arguments("i", ...)` in `blackhole.py`) cannot
    // run on `Ref`-classified operands — upstream `rtyper.py:specialize`
    // determines operand kind from the op's `concretetype`, not from a
    // fallback.  Pyre's front-end leaves many `OpKind::Input` values
    // with `ty: Unknown` (path references not yet bound by the annotator);
    // without this pass they end up defaulted to `GcRef` by the backfill
    // below and produce pyre-only `int_*/{ri,ir,rr,r}` keys. Upgrade
    // Unknown operands of pure integer ops to `Signed`, matching
    // RPython's mandatory kind assignment.  `ptr_eq` / `ptr_ne` remain
    // Ref-Ref (rewritten in `jtransform` per `rpython/jit/codewriter/
    // jtransform.py:1243-1255 _rewrite_cmp_ptrs`) and are skipped.
    /// Operations whose RPython decorator requires integer operands on
    /// both sides (`blackhole.py:459+ @arguments("i", "i", returns="i")`).
    /// Unlike the comparison ops `lt`/`le`/`gt`/`ge`, no ptr-specific
    /// variant exists in RPython — these must see Signed operands.
    fn canonical_int_binop(op: &str) -> Option<&'static str> {
        match op {
            "add" | "add_assign" => Some("add"),
            "sub" | "sub_assign" => Some("sub"),
            "mul" | "mul_assign" => Some("mul"),
            "div" | "div_assign" => Some("div"),
            "mod" | "mod_assign" => Some("mod"),
            "and" | "bitand" | "bitand_assign" => Some("and"),
            "or" | "bitor" | "bitor_assign" => Some("or"),
            "xor" | "bitxor" | "bitxor_assign" => Some("xor"),
            "rshift" | "shr" | "rshift_assign" => Some("rshift"),
            "lshift" | "shl" | "lshift_assign" => Some("lshift"),
            "lt" => Some("lt"),
            "le" => Some("le"),
            "gt" => Some("gt"),
            "ge" => Some("ge"),
            _ => None,
        }
    }
    /// `eq`/`ne` may still be rewritten to `ptr_eq`/`ptr_ne` in
    /// `jtransform.rs` when both operands are Ref (RPython
    /// `_rewrite_cmp_ptrs`). Skip them here so the jtransform pass
    /// observes the original Ref-Ref shape.
    fn is_int_unop(op: &str) -> bool {
        matches!(op, "neg" | "invert" | "not")
    }
    fn is_identity_unop(op: &str) -> bool {
        matches!(op, "same_as")
    }
    // Iterate: a backward-inferred Signed on one ValueId may feed
    // another op's operand through Link propagation, so run until
    // fixed-point.  RPython's `rtyper.py:specialize` is a single
    // forward pass because every Variable has a `concretetype` from
    // annotation; pyre's partial typing means downstream ops can
    // depend on upstream inferences that only this pass supplies.
    loop {
        let mut changed = false;
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    OpKind::BinOp {
                        op: opname,
                        lhs,
                        rhs,
                        ..
                    } if canonical_int_binop(opname).is_some() => {
                        // Only upgrade Unknown operands.  Forcing
                        // GcRef → Signed here is unsound: a value
                        // classified Ref may alias with pointer uses
                        // elsewhere in the graph, and demoting it to
                        // Int would misroute those other uses
                        // (regalloc puts the value in an int register,
                        // subsequent field reads dereference garbage).
                        // The remaining pyre-only `int_*/{ri,ir,rr}`
                        // keys need `cast_ptr_to_int` insertion in
                        // jtransform, not a type override in the
                        // rtyper.
                        changed |= maybe_seed_concrete_type(&mut state, *lhs, ConcreteType::Signed);
                        changed |= maybe_seed_concrete_type(&mut state, *rhs, ConcreteType::Signed);
                        if let Some(result) = op.result {
                            // RPython rtyper resolves an `add` whose
                            // operands are both `lltype.Float` to
                            // `float_add` directly (`rfloat.py
                            // rtype_add`); the result inherits Float
                            // (or Int for comparisons).  When pyre's
                            // upstream typing already classified both
                            // operands as Float, mirror that: seed the
                            // result as Float for arithmetic ops and
                            // Int for comparisons.  jtransform will
                            // then rewrite the BinOp's `op` from `add`
                            // to `float_add` (etc.) and insert
                            // `cast_int_to_float` for mixed int/float
                            // operands, keeping IR/regalloc consistent.
                            let lhs_float = *state.get(*lhs) == ConcreteType::Float;
                            let rhs_float = *state.get(*rhs) == ConcreteType::Float;
                            let any_float = lhs_float || rhs_float;
                            let both_numeric = matches!(
                                *state.get(*lhs),
                                ConcreteType::Signed | ConcreteType::Float
                            ) && matches!(
                                *state.get(*rhs),
                                ConcreteType::Signed | ConcreteType::Float
                            );
                            // Mirror jtransform's float-rewrite set
                            // (`jit_codewriter/jtransform.rs`):
                            // arithmetic `add/sub/mul/div` →
                            // `float_*` returns Float; comparisons
                            // `lt/le/gt/ge/eq/ne` → `float_*` returns
                            // Int (Bool — RPython `rfloat.py:133`,
                            // `blackhole.py:731 bhimpl_float_eq`).
                            // `mod` is also Float when a Float operand is
                            // present, but jtransform lowers it to the
                            // residual `ll_math_fmod` helper rather than a
                            // non-RPython `float_mod` opcode.
                            let is_compare =
                                matches!(opname.as_str(), "lt" | "le" | "gt" | "ge" | "eq" | "ne");
                            let is_arith =
                                matches!(opname.as_str(), "add" | "sub" | "mul" | "div" | "mod");
                            if any_float && both_numeric && (is_arith || is_compare) {
                                let target = if is_compare {
                                    ConcreteType::Signed
                                } else {
                                    ConcreteType::Float
                                };
                                if state.get(result) != &target {
                                    state.concrete_types.insert(result, target);
                                    changed = true;
                                }
                            } else if !any_float || !both_numeric {
                                changed |= maybe_seed_concrete_type(
                                    &mut state,
                                    result,
                                    ConcreteType::Signed,
                                );
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_int_unop(opname) => {
                        changed |=
                            maybe_seed_concrete_type(&mut state, *operand, ConcreteType::Signed);
                        if let Some(result) = op.result {
                            // Same Float-operand override as the BinOp
                            // arm above.  Unary `neg` on a Float
                            // returns Float (`float_neg`).
                            let operand_float = *state.get(*operand) == ConcreteType::Float;
                            if operand_float && opname == "neg" {
                                if state.get(result) != &ConcreteType::Float {
                                    state.concrete_types.insert(result, ConcreteType::Float);
                                    changed = true;
                                }
                            } else {
                                changed |= maybe_seed_concrete_type(
                                    &mut state,
                                    result,
                                    ConcreteType::Signed,
                                );
                            }
                        }
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if opname == "cast_int_to_float" => {
                        // RPython `rfloat.py` rtype_int → calls
                        // `hop.inputarg(Signed, ...)` before emitting
                        // `cast_int_to_float`, so the operand's
                        // concretetype is Signed by construction.
                        // jtransform's `coerce_operand_to_float`
                        // (jit_codewriter/jtransform.rs) emits this
                        // op only when get_value_kind(operand) == 'i'
                        // in pass 1.  Pass 2 runs on the rewritten
                        // graph from a fresh state and may lose that
                        // backward-constraint upgrade if the operand
                        // lacks a definitive def-site classification
                        // (e.g., Call with Unknown result_ty), which
                        // would surface as `cast_int_to_float/r>f`
                        // at the assembler.  Re-seed Signed here so
                        // pass 2 converges to the same operand kind.
                        changed |=
                            maybe_seed_concrete_type(&mut state, *operand, ConcreteType::Signed);
                    }
                    OpKind::UnaryOp {
                        op: opname,
                        operand,
                        ..
                    } if is_identity_unop(opname) => {
                        if let Some(result) = op.result {
                            let operand_ty = state.get(*operand).clone();
                            let result_ty = state.get(result).clone();
                            changed |= maybe_seed_concrete_type(&mut state, result, operand_ty);
                            changed |= maybe_seed_concrete_type(&mut state, *operand, result_ty);
                        }
                    }
                    _ => {}
                }
            }
            // Re-propagate along links after each op round so backward
            // inferences reach the linked ValueIds. RPython has concrete
            // types on both ends by this point and `_convert_link()` emits
            // conversions where needed; this legacy pass has no conversion
            // insertion, so Unknown values joined by a link must converge to
            // the known low-level type before regalloc/flatten.
            for link in &block.exits {
                changed |= if link_is_raise_like(link) {
                    converge_raise_link(&mut state, graph, link)
                } else {
                    converge_link(&mut state, graph, link)
                };
            }
        }
        if !changed {
            break;
        }
    }

    // Backfill any `Variable` that's referenced by the graph but was
    // never typed (e.g. Link args pointing at values with no producer
    // op, synthetic values whose declaring site didn't seed a
    // concretetype).  RPython's rtyper cannot leave any `Variable`
    // untyped — `lltype.Signed / Ptr / Float` is mandatory — so the
    // assembler is entitled to assume every register has a class.
    // Default untyped values to `GcRef` (the same safe default
    // jtransform's `get_value_kind` picks) so `build_value_kinds` +
    // `perform_register_allocation` always produce a coloring for
    // every reachable value.
    let mut seen: std::collections::HashSet<ValueId> = std::collections::HashSet::new();
    for block in &graph.blocks {
        for v in &block.inputargs {
            seen.insert(*v);
        }
        for op in &block.operations {
            for v in crate::inline::op_value_refs(&op.kind) {
                seen.insert(v);
            }
            if let Some(r) = op.result {
                seen.insert(r);
            }
        }
        for link in &block.exits {
            for arg in &link.args {
                if let crate::model::LinkArg::Value(v) = arg {
                    seen.insert(*v);
                }
            }
            for arg in link.last_exception.iter().chain(link.last_exc_value.iter()) {
                if let crate::model::LinkArg::Value(v) = arg {
                    seen.insert(*v);
                }
            }
        }
    }
    for v in seen {
        state
            .concrete_types
            .entry(v)
            .and_modify(|c| {
                if *c == ConcreteType::Unknown {
                    *c = ConcreteType::GcRef;
                }
            })
            .or_insert(ConcreteType::GcRef);
    }

    state
}

/// Merge the pre-jtransform rtyper state with the rewritten graph's
/// concrete result kinds (legacy entry point).
///
/// Slice 3 relocated the merge logic itself to
/// [`crate::jit_codewriter::type_state::merge_synth_kinds`] —
/// `merge_synth_kinds` is the pyre-divergence piece that survives the
/// rtyper cutover. This function remains as the legacy-pipeline entry
/// point because it bundles two legacy-only graph walks
/// (`resolve_types` + `authoritative_result_types`) with the merge.
/// Real-path callers (Slice 4 onwards) call `merge_synth_kinds`
/// directly with their own `post_resolve` / `post_result` inputs.
pub fn resolve_rewritten_types(
    original_types: &TypeResolutionState,
    rewritten_graph: &FunctionGraph,
    annotations: &AnnotationState,
    stamped_kinds: &HashMap<ValueId, ConcreteType>,
) -> TypeResolutionState {
    let post_result_types = authoritative_result_types(rewritten_graph);
    let post_resolve = resolve_types(rewritten_graph, annotations);
    crate::jit_codewriter::type_state::merge_synth_kinds(
        original_types,
        post_resolve,
        post_result_types,
        stamped_kinds,
    )
}

fn const_value_to_concrete(value: &ConstValue) -> ConcreteType {
    match value {
        ConstValue::Int(_)
        | ConstValue::Bool(_)
        | ConstValue::SpecTag(_)
        | ConstValue::LLAddress(_) => ConcreteType::Signed,
        ConstValue::Float(_) => ConcreteType::Float,
        ConstValue::Placeholder => ConcreteType::Unknown,
        ConstValue::Atom(_)
        | ConstValue::Dict(_)
        | ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Graphs(_)
        | ConstValue::LowLevelType(_)
        | ConstValue::None
        | ConstValue::Code(_)
        | ConstValue::LLPtr(_)
        | ConstValue::Function(_)
        | ConstValue::HostObject(_) => ConcreteType::GcRef,
    }
}

fn link_is_raise_like(link: &Link) -> bool {
    link.last_exception.is_some() && link.last_exc_value.is_some()
}

fn convert_link(state: &mut TypeResolutionState, graph: &FunctionGraph, link: &Link) {
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let _ = maybe_seed_concrete_type(state, *dst, link_arg_concrete_type(state, src));
    }
}

fn convert_raise_link(state: &mut TypeResolutionState, graph: &FunctionGraph, link: &Link) {
    if let Some(LinkArg::Value(value)) = link.last_exception.as_ref() {
        let _ = maybe_seed_concrete_type(state, *value, ConcreteType::Signed);
    }
    if let Some(LinkArg::Value(value)) = link.last_exc_value.as_ref() {
        let _ = maybe_seed_concrete_type(state, *value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(state, src)
        };
        let _ = maybe_seed_concrete_type(state, *dst, src_ty);
    }
}

fn converge_link(state: &mut TypeResolutionState, graph: &FunctionGraph, link: &Link) -> bool {
    let mut changed = false;
    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        match src {
            LinkArg::Value(src) => {
                let src_ty = state.get(*src).clone();
                let dst_ty = state.get(*dst).clone();
                changed |= maybe_seed_concrete_type(state, *dst, src_ty);
                changed |= maybe_seed_concrete_type(state, *src, dst_ty);
            }
            LinkArg::Const(value) => {
                changed |= maybe_seed_concrete_type(state, *dst, const_value_to_concrete(value));
            }
        }
    }
    changed
}

fn converge_raise_link(
    state: &mut TypeResolutionState,
    graph: &FunctionGraph,
    link: &Link,
) -> bool {
    let mut changed = false;
    if let Some(LinkArg::Value(value)) = link.last_exception.as_ref() {
        changed |= maybe_seed_concrete_type(state, *value, ConcreteType::Signed);
    }
    if let Some(LinkArg::Value(value)) = link.last_exc_value.as_ref() {
        changed |= maybe_seed_concrete_type(state, *value, ConcreteType::GcRef);
    }

    let target_block = graph.block(link.target);
    for (dst, src) in target_block.inputargs.iter().zip(link.args.iter()) {
        let src_ty = if Some(src) == link.last_exception.as_ref() {
            ConcreteType::Signed
        } else if Some(src) == link.last_exc_value.as_ref() {
            ConcreteType::GcRef
        } else {
            link_arg_concrete_type(state, src)
        };
        changed |= maybe_seed_concrete_type(state, *dst, src_ty);
        if let LinkArg::Value(src) = src {
            let dst_ty = state.get(*dst).clone();
            changed |= maybe_seed_concrete_type(state, *src, dst_ty);
        }
    }
    changed
}

fn link_arg_concrete_type(state: &TypeResolutionState, src: &LinkArg) -> ConcreteType {
    match src {
        LinkArg::Value(src) => state.get(*src).clone(),
        LinkArg::Const(value) => const_value_to_concrete(value),
    }
}

fn maybe_seed_concrete_type(
    state: &mut TypeResolutionState,
    dst: ValueId,
    src_ty: ConcreteType,
) -> bool {
    if state.get(dst) == &ConcreteType::Unknown && src_ty != ConcreteType::Unknown {
        state.concrete_types.insert(dst, src_ty);
        true
    } else {
        false
    }
}

fn kind_char_to_concrete(kind: char) -> ConcreteType {
    match kind {
        'i' => ConcreteType::Signed,
        'r' => ConcreteType::GcRef,
        'f' => ConcreteType::Float,
        'v' => ConcreteType::Void,
        _ => ConcreteType::Unknown,
    }
}

fn valuetype_to_concrete(vt: &ValueType) -> ConcreteType {
    match vt {
        ValueType::Int => ConcreteType::Signed,
        ValueType::Ref => ConcreteType::GcRef,
        ValueType::Float => ConcreteType::Float,
        ValueType::Void => ConcreteType::Void,
        ValueType::State | ValueType::Unknown => ConcreteType::Unknown,
    }
}

fn infer_concrete_from_op(kind: &OpKind) -> ConcreteType {
    match kind {
        OpKind::ConstInt(_) => ConcreteType::Signed,
        OpKind::ConstFloat(_) => ConcreteType::Float,
        // RPython `rpython/annotator/annrpython.py` types every Variable
        // at annotation time, so `OpKind::Input` reaching rtyper has a
        // concrete type.  pyre's front-end (`front/ast.rs` Expr::Path
        // lowering) re-emits a fresh `OpKind::Input { ty: Unknown }` for
        // each source-level identifier reference instead of binding the
        // name to the inputarg's Variable.  Leave Unknown so the
        // integer-op backward-constraint pass (`resolve_types`) can
        // upgrade operands of pure integer ops to `Signed` before the
        // final GcRef backfill.
        OpKind::Input { ty, .. } => valuetype_to_concrete(ty),
        // Field / array reads whose declared `ty` is pyre-only Unknown
        // default to `ConcreteType::GcRef` so the value reaches
        // regalloc. RPython's rtyper resolves the field / element type
        // from the struct's `concretetype`; pyre's struct registry can
        // lack the entry (Rust generics, unsupported declarations),
        // leaving `ValueType::Unknown`.  Ref is the conservative
        // default — if the underlying value is actually Int, the
        // canonical `getfield_gc_i/rd>i` key still carries the
        // correct result kind at assembler emit time.
        OpKind::FieldRead { ty, .. } => {
            let c = valuetype_to_concrete(ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::ArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::InteriorFieldRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::Call { result_ty, .. } => valuetype_to_concrete(result_ty),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_concrete(*result_kind),
        OpKind::UnaryOp { op, result_ty, .. } if op == "same_as" => {
            valuetype_to_concrete(result_ty)
        }
        // For `BinOp` / `UnaryOp` whose `result_ty` is still
        // `ValueType::Unknown`, return `ConcreteType::Unknown` rather
        // than defaulting to `Signed`.  The override at line ~55 only
        // fires when `inferred != Unknown`; previously the `Signed`
        // fallback overrode the annotator's value (e.g. the `neg(Float)
        // → Float` annotation produced by the dedicated `op == "neg"`
        // arm in `infer_op_type`), poisoning downstream phi-merge
        // inputargs whose link arg types are correctly Float.  With
        // `Unknown` here the override is skipped and the annotator's
        // value persists; the loop below still upgrades the concrete
        // type to `Float` for `neg(Float)` etc. via the special arms.
        OpKind::BinOp { result_ty, .. } | OpKind::UnaryOp { result_ty, .. } => {
            valuetype_to_concrete(result_ty)
        }
        // Vtable funcptr extraction returns an integer pointer (RPython
        // `op.args[0]` of `indirect_call` is `Ptr(FuncType)`).
        OpKind::VtableMethodPtr { .. } => ConcreteType::Signed,
        OpKind::IndirectCall { result_ty, .. } => valuetype_to_concrete(result_ty),
        // Virtualizable field / array reads inherit the RPython
        // canonical result kind from the declared field/element type.
        OpKind::VableFieldRead { ty, .. } => {
            let c = valuetype_to_concrete(ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        OpKind::VableArrayRead { item_ty, .. } => {
            let c = valuetype_to_concrete(item_ty);
            if c == ConcreteType::Unknown {
                ConcreteType::GcRef
            } else {
                c
            }
        }
        // pyre-only `OpKind::Abort` (`front/ast.rs` lowering of Rust
        // syntax not yet ported — macros, unsupported literals,
        // fallback expressions).  Fall back to GcRef so these values
        // still get a regalloc coloring and the assembler's
        // `lookup_reg_with_kind` covers every operand. RPython has no
        // analogue; porting each producer path individually eliminates
        // the `abort/` wire keys.
        OpKind::Abort { .. } => ConcreteType::GcRef,
        _ => ConcreteType::Unknown,
    }
}

fn authoritative_result_types(graph: &FunctionGraph) -> HashMap<ValueId, ConcreteType> {
    let mut result = HashMap::new();
    for block in &graph.blocks {
        for op in &block.operations {
            let Some(value) = op.result else {
                continue;
            };
            if let Some(concrete) = authoritative_result_type_from_op(&op.kind) {
                result.insert(value, concrete);
            }
        }
    }
    result
}

fn concrete_if_known(concrete: ConcreteType) -> Option<ConcreteType> {
    if concrete == ConcreteType::Unknown {
        None
    } else {
        Some(concrete)
    }
}

fn authoritative_result_type_from_op(kind: &OpKind) -> Option<ConcreteType> {
    match kind {
        OpKind::ConstInt(_) => Some(ConcreteType::Signed),
        OpKind::ConstFloat(_) => Some(ConcreteType::Float),
        OpKind::Input { ty, .. } => concrete_if_known(valuetype_to_concrete(ty)),
        OpKind::FieldRead { ty, .. } | OpKind::VableFieldRead { ty, .. } => {
            concrete_if_known(valuetype_to_concrete(ty))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => {
            concrete_if_known(valuetype_to_concrete(item_ty))
        }
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. } => concrete_if_known(valuetype_to_concrete(result_ty)),
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => {
            concrete_if_known(kind_char_to_concrete(*result_kind))
        }
        OpKind::VtableMethodPtr { .. } => Some(ConcreteType::Signed),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{
        ExitSwitch, FunctionGraph, Link, LinkArg, OpKind, ValueType, exception_exitcase,
    };
    use crate::translate_legacy::annotator::annrpython as annotate;

    #[test]
    fn resolves_int_types() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_return(entry, Some(v));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(v), &ConcreteType::Signed);
    }

    #[test]
    fn resolves_ref_field() {
        let mut graph = FunctionGraph::new("test");
        let entry = graph.startblock;
        let base = graph.alloc_value();
        let v = graph
            .push_op(
                entry,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new("obj", None),
                    ty: ValueType::Ref,
                    pure: false,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(v));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(v), &ConcreteType::GcRef);
    }

    #[test]
    fn resolves_phi_through_link_args() {
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];
        graph.set_goto(entry, target, vec![val]);
        graph.set_return(target, Some(phi));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(phi), &ConcreteType::Signed);
    }

    #[test]
    fn backward_constraint_types_unknown_int_binop_operands_as_signed() {
        let mut graph = FunctionGraph::new("int_backprop");
        let entry = graph.startblock;
        let lhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "lhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "rhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "add".to_string(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(lhs), &ConcreteType::Signed);
        assert_eq!(types.get(rhs), &ConcreteType::Signed);
        assert_eq!(types.get(result), &ConcreteType::Signed);
    }

    #[test]
    fn backward_constraint_types_frontend_bitop_operands_as_signed() {
        let mut graph = FunctionGraph::new("bitxor_backprop");
        let entry = graph.startblock;
        let lhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "lhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let rhs = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "rhs".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::BinOp {
                    op: "bitxor".to_string(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(lhs), &ConcreteType::Signed);
        assert_eq!(types.get(rhs), &ConcreteType::Signed);
        assert_eq!(types.get(result), &ConcreteType::Signed);
    }

    #[test]
    fn same_as_preserves_ref_classification() {
        let mut graph = FunctionGraph::new("same_as_ref");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Ref,
                },
                true,
            )
            .unwrap();
        let alias = graph
            .push_op(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(alias));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(value), &ConcreteType::GcRef);
        assert_eq!(types.get(alias), &ConcreteType::GcRef);
    }

    #[test]
    fn same_as_propagates_signed_without_forcing_unknown_identity_to_int() {
        let mut graph = FunctionGraph::new("same_as_int");
        let entry = graph.startblock;
        let value = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let alias = graph
            .push_op(
                entry,
                OpKind::UnaryOp {
                    op: "same_as".to_string(),
                    operand: value,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(alias));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(value), &ConcreteType::Signed);
        assert_eq!(types.get(alias), &ConcreteType::Signed);
    }

    #[test]
    fn rewritten_call_result_kind_overrides_stale_original_type() {
        use crate::call::CallDescriptor;
        use crate::model::CallFuncPtr;
        use majit_ir::descr::EffectInfo;

        let mut graph = FunctionGraph::new("rewritten_call_result");
        let entry = graph.startblock;
        let funcptr = graph.push_op(entry, OpKind::ConstInt(0), true).unwrap();
        let result = graph
            .push_op(
                entry,
                OpKind::CallResidual {
                    funcptr: CallFuncPtr::Value(funcptr),
                    descriptor: CallDescriptor::known(EffectInfo::default()),
                    args_i: vec![],
                    args_r: vec![],
                    args_f: vec![],
                    result_kind: 'r',
                    indirect_targets: None,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(result));

        let annotations = annotate::annotate(&graph);
        let mut original = TypeResolutionState::new();
        original.concrete_types.insert(result, ConcreteType::Void);

        let types = resolve_rewritten_types(
            &original,
            &graph,
            &annotations,
            &std::collections::HashMap::new(),
        );

        assert_eq!(types.get(result), &ConcreteType::GcRef);
    }

    #[test]
    fn backward_constraint_propagates_signed_back_through_link_source() {
        let mut graph = FunctionGraph::new("link_backprop");
        let entry = graph.startblock;
        let src = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "x".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        let one = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let (target, phi_args) = graph.create_block_with_args(1);
        let phi = phi_args[0];
        graph.set_goto(entry, target, vec![src]);
        let result = graph
            .push_op(
                target,
                OpKind::BinOp {
                    op: "add".to_string(),
                    lhs: phi,
                    rhs: one,
                    result_ty: ValueType::Int,
                },
                true,
            )
            .unwrap();
        graph.set_return(target, Some(result));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(phi), &ConcreteType::Signed);
        assert_eq!(types.get(src), &ConcreteType::Signed);
        assert_eq!(types.get(result), &ConcreteType::Signed);
    }

    #[test]
    fn unknown_input_without_integer_constraint_backfills_as_gcref() {
        let mut graph = FunctionGraph::new("unknown_backfill");
        let entry = graph.startblock;
        let value = graph
            .push_op(
                entry,
                OpKind::Input {
                    name: "obj".to_string(),
                    ty: ValueType::Unknown,
                },
                true,
            )
            .unwrap();
        graph.set_return(entry, Some(value));

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(value), &ConcreteType::GcRef);
    }

    #[test]
    fn resolves_raise_link_exception_pair() {
        let mut graph = FunctionGraph::new("raise_link");
        let entry = graph.startblock;
        let (exc_block, etype, evalue) = graph.exceptblock_args();
        let last_exception = graph.alloc_value();
        let last_exc_value = graph.alloc_value();
        graph.set_control_flow_metadata(
            entry,
            Some(ExitSwitch::LastException),
            vec![
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

        let annotations = annotate::annotate(&graph);
        let types = resolve_types(&graph, &annotations);
        assert_eq!(types.get(last_exception), &ConcreteType::Signed);
        assert_eq!(types.get(last_exc_value), &ConcreteType::GcRef);
        assert_eq!(types.get(etype), &ConcreteType::Signed);
        assert_eq!(types.get(evalue), &ConcreteType::GcRef);
    }
}
