//! ValueId → ConcreteType side table for the jit_codewriter IR.
//!
//! PRE-EXISTING-ADAPTATION. RPython stores `.concretetype` inline on each
//! `Variable` after `RPythonTyper.specialize()` rewrites the graph, so no
//! side table exists upstream. Pyre's current jit_codewriter consumes a
//! `crate::model::FunctionGraph` (value-id-based, not variable-based),
//! so the post-rtyper kind information lives in this separate table.
//!
//! `build_value_kinds` (pure `ConcreteType → RegKind` projection) and
//! `merge_synth_kinds` (post-jtransform 4-source merge) live here
//! beside the data type — these are pyre-only divergences from RPython
//! that survive the rtyper cutover (`~/.claude/plans/0-warm-raccoon.md`
//! Slice 3). The graph-walking algorithm `resolve_types` remains in
//! `translate_legacy/rtyper/rtyper.rs` until the real rtyper
//! (`translator/rtyper/`) produces per-Variable concretetypes
//! end-to-end and replaces it.

use std::collections::HashMap;

use crate::model::ValueId;

/// Concrete low-level type. RPython `Repr.lowleveltype` collapsed to the
/// four kinds the jit_codewriter needs (Signed / GcRef / Float / Void)
/// plus `Unknown` for pre-resolution slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConcreteType {
    /// Signed integer (RPython `Signed` / i64).
    Signed,
    /// GC reference (RPython `Ptr(GcStruct)`).
    GcRef,
    /// Float (RPython `Float` / f64).
    Float,
    /// Void (RPython `Void`).
    Void,
    /// Unknown / unresolved.
    Unknown,
}

/// Type resolution state: `ValueId → ConcreteType`.
#[derive(Debug)]
pub struct TypeResolutionState {
    pub concrete_types: HashMap<ValueId, ConcreteType>,
}

impl TypeResolutionState {
    pub fn new() -> Self {
        TypeResolutionState {
            concrete_types: HashMap::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &ConcreteType {
        self.concrete_types
            .get(&id)
            .unwrap_or(&ConcreteType::Unknown)
    }
}

/// Post-jtransform 4-source merge of `ConcreteType` views into a
/// single authoritative `TypeResolutionState`.
///
/// **Why this exists** — pyre divergence, not RPython parity.
///
/// RPython has no merge step: every `Variable` carries `.concretetype`
/// inline, and jtransform-created operations preserve or assign that
/// type on the result variable. Pyre's legacy jit_codewriter graph
/// uses a `ValueId -> ConcreteType` side table, so the codewriter has
/// up to four partial sources after jtransform that must be reconciled
/// into a single `TypeResolutionState` before regalloc / flatten /
/// assemble:
///
/// 1. `original` — the pre-jtransform rtyper output. Carries backward
///    inferences that may disappear after jtransform removed the
///    consumer op that caused them.
/// 2. `post_resolve` — a fresh `resolve_types(rewritten_graph,
///    annotations)` walk over the post-jtransform graph. Picks up
///    anything visible on the rewritten shape.
/// 3. `post_result` — the per-op `result_ty` / `result_kind` declared
///    on each rewritten op. Authoritative for op-result kinds (since
///    the rewriter declares them), so it wins over `original`'s
///    operand inferences.
/// 4. `stamped` — jtransform's `synth_kinds` map: kinds the rewriter
///    explicitly stamped on synthesized values during lowering
///    (e.g. `cast_ptr_to_int`'s int-typed result). Wins outright
///    because the rewriter knows the ground truth.
///
/// Precedence: `stamped` > `post_result` > `post_resolve` > `original`.
/// `original` only fills in slots that `post_result` doesn't claim,
/// matching RPython's "single authoritative `op.result.concretetype`"
/// invariant: a stale pre-rewrite operand kind never overrides a
/// freshly-declared post-rewrite result kind.
///
/// Survives the rtyper cutover (Slice 3): once the real `RPythonTyper`
/// path produces `original` / `post_resolve`, this function still
/// reconciles them with `post_result` / `stamped`. Slice 3 relocated
/// this body from `translate_legacy/rtyper/rtyper.rs:360-382` to
/// keep the legacy graph-walk (`resolve_types`) callable separately
/// from the merge logic.
pub fn merge_synth_kinds(
    original: &TypeResolutionState,
    post_resolve: TypeResolutionState,
    post_result: HashMap<ValueId, ConcreteType>,
    stamped: &HashMap<ValueId, ConcreteType>,
) -> TypeResolutionState {
    let mut merged = post_resolve;

    // (1) `original` operand inferences fill unresolved slots, but
    // never override `post_result`. Skip `Unknown` entries — they are
    // placeholders, not actual inferences.
    for (&value, kind) in &original.concrete_types {
        if *kind != ConcreteType::Unknown && !post_result.contains_key(&value) {
            merged.concrete_types.insert(value, kind.clone());
        }
    }
    // (2) Op-result kinds win over operand inferences.
    for (value, kind) in post_result {
        merged.concrete_types.insert(value, kind);
    }
    // (3) Synth kinds (jtransform-stamped) override everything — the
    // rewriter declares these at lowering time with full ground truth.
    for (&value, kind) in stamped {
        merged.concrete_types.insert(value, kind.clone());
    }

    merged
}

/// Build value kind map from type resolution state.
///
/// RPython: `getkind(v.concretetype)` — in RPython, types live directly
/// on variables. In majit, we extract them from TypeResolutionState.
///
/// Used by both `perform_all_register_allocations()` (before flatten)
/// and `flatten_with_types()` (populates SSARepr.value_kinds).
pub fn build_value_kinds(types: &TypeResolutionState) -> HashMap<ValueId, crate::flatten::RegKind> {
    use crate::flatten::RegKind;
    types
        .concrete_types
        .iter()
        .filter_map(|(&vid, ct)| {
            let kind = match ct {
                ConcreteType::Signed => RegKind::Int,
                ConcreteType::GcRef => RegKind::Ref,
                ConcreteType::Float => RegKind::Float,
                _ => return None,
            };
            Some((vid, kind))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_from(pairs: &[(ValueId, ConcreteType)]) -> TypeResolutionState {
        let mut s = TypeResolutionState::new();
        for (v, k) in pairs {
            s.concrete_types.insert(*v, k.clone());
        }
        s
    }

    fn map_from(pairs: &[(ValueId, ConcreteType)]) -> HashMap<ValueId, ConcreteType> {
        pairs.iter().cloned().collect()
    }

    #[test]
    fn merge_synth_kinds_post_resolve_starts_as_base() {
        // No original / post_result / stamped overrides — the merged
        // state is just `post_resolve`.
        let post_resolve = state_from(&[
            (ValueId(1), ConcreteType::Signed),
            (ValueId(2), ConcreteType::Float),
        ]);
        let original = TypeResolutionState::new();
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();
        let stamped: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Signed);
        assert_eq!(merged.get(ValueId(2)), &ConcreteType::Float);
    }

    #[test]
    fn merge_synth_kinds_original_fills_unresolved_slots() {
        // post_resolve missing a value; original.concrete_types
        // supplies it (since post_result does not claim it).
        let post_resolve = TypeResolutionState::new();
        let original = state_from(&[(ValueId(7), ConcreteType::Signed)]);
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();
        let stamped: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(7)), &ConcreteType::Signed);
    }

    #[test]
    fn merge_synth_kinds_original_unknown_does_not_propagate() {
        // Unknown is a placeholder, not an inference. It must not fill
        // a slot — leaving the merged state's `get` returning the
        // default Unknown via `concrete_types` miss.
        let post_resolve = TypeResolutionState::new();
        let original = state_from(&[(ValueId(7), ConcreteType::Unknown)]);
        let post_result: HashMap<ValueId, ConcreteType> = HashMap::new();
        let stamped: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert!(
            !merged.concrete_types.contains_key(&ValueId(7)),
            "Unknown originals must not seed the merged state"
        );
    }

    #[test]
    fn merge_synth_kinds_post_result_overrides_original() {
        // original infers Signed; post_result declares Float.
        // post_result wins (rewriter ground truth).
        let post_resolve = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let original = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let post_result = map_from(&[(ValueId(1), ConcreteType::Float)]);
        let stamped: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Float);
    }

    #[test]
    fn merge_synth_kinds_original_skipped_when_post_result_claims_value() {
        // Even non-Unknown original is skipped if post_result claims
        // the slot — the rewriter's authoritative declaration takes
        // precedence (mirrors RPython's single-source-of-truth shape).
        let post_resolve = TypeResolutionState::new();
        let original = state_from(&[(ValueId(1), ConcreteType::GcRef)]);
        let post_result = map_from(&[(ValueId(1), ConcreteType::Signed)]);
        let stamped: HashMap<ValueId, ConcreteType> = HashMap::new();

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Signed);
    }

    #[test]
    fn merge_synth_kinds_stamped_overrides_everything() {
        // stamped is jtransform's ground truth — wins over post_result
        // (and therefore over post_resolve / original).
        let post_resolve = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let original = state_from(&[(ValueId(1), ConcreteType::Signed)]);
        let post_result = map_from(&[(ValueId(1), ConcreteType::Float)]);
        let stamped = map_from(&[(ValueId(1), ConcreteType::GcRef)]);

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::GcRef);
    }

    #[test]
    fn merge_synth_kinds_full_precedence_chain() {
        // Four ValueIds, each contributed by a different source —
        // confirm each source's lane reaches the merged state.
        let post_resolve = state_from(&[(ValueId(1), ConcreteType::Float)]);
        let original = state_from(&[(ValueId(2), ConcreteType::Signed)]);
        let post_result = map_from(&[(ValueId(3), ConcreteType::GcRef)]);
        let stamped = map_from(&[(ValueId(4), ConcreteType::Signed)]);

        let merged = merge_synth_kinds(&original, post_resolve, post_result, &stamped);
        assert_eq!(merged.get(ValueId(1)), &ConcreteType::Float);
        assert_eq!(merged.get(ValueId(2)), &ConcreteType::Signed);
        assert_eq!(merged.get(ValueId(3)), &ConcreteType::GcRef);
        assert_eq!(merged.get(ValueId(4)), &ConcreteType::Signed);
    }
}
