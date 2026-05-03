//! ValueId-keyed annotator output for the jit_codewriter IR.
//!
//! PRE-EXISTING-ADAPTATION.  RPython's `RPythonAnnotator.complete()`
//! attaches a `SomeValue` directly to each `Variable.annotation` slot
//! on the flowgraph (`rpython/annotator/annrpython.py:54-66`,
//! `rpython/flowspace/model.py: Variable.annotation`).  Pyre's
//! jit_codewriter consumes `crate::model::FunctionGraph`
//! (value-id-based), so the annotator output lives in this side table
//! instead.
//!
//! The carrier comes in two layers, ordered from RPython-orthodox to
//! pyre-only:
//!
//! 1. `some_values: HashMap<ValueId, Rc<SomeValue>>` — direct
//!    counterpart of `Variable.annotation`.  When the producer can
//!    determine the precise lattice node (`SomeInteger`, `SomeFloat`,
//!    `SomeInstance(classdef)`, `SomePtr(ll_ptrtype)`, `SomePBC`, ...)
//!    it sets the entry here; downstream consumers (the flowspace
//!    adapter) clone it onto `Variable.annotation` unchanged, exactly
//!    as `_setbinding` does upstream.
//!
//! 2. `types: HashMap<ValueId, ValueType>` — pyre-legacy fallback that
//!    only carries the lattice family discriminator
//!    (`Int`/`Float`/`Ref`/`Void`/`State`/`Unknown`).  Used when the
//!    producer cannot synthesize a concrete `SomeValue` shell.  The
//!    adapter projects through `valuetype_to_someshell`; `Unknown`
//!    deliberately has no shell, so annotation gaps surface fail-loud
//!    instead of being bridged to a fabricated GC reference.
//!    Convergence path is to populate `some_values` for every `ValueId`
//!    and drop this fallback together with the projection.
//!
//! The `annotate()` algorithm in
//! `translate_legacy/annotator/annrpython.rs` populates `types` only;
//! producers can populate `some_values` opportunistically (e.g. the
//! frontend's struct/PBC/Ptr cases) without breaking back-compat.

use std::collections::HashMap;
use std::rc::Rc;

use std::collections::BTreeMap;

use crate::annotator::model::{SomeFloat, SomeInstance, SomeInteger, SomeValue};
use crate::model::{ValueId, ValueType};

/// RPython `SomeValue` lattice projection of the legacy `ValueType`.
///
/// `RPythonTyper.bindingrepr` (`rtyper.rs:961`) dispatches purely on the
/// `SomeValue` shape via `rtyper_makekey` / `rtyper_makerepr`.  `Int`
/// and `Float` resolve cleanly through `rint::IntegerRepr` /
/// `rfloat::FloatRepr`.
///
/// Mapping:
///
/// | legacy `ValueType` | `SomeValue` shell      | RPython source / status |
/// |--------------------|------------------------|--------------------------|
/// | `Int`              | `Integer(SomeInteger)` | `model.py:206-224` -> `SomeValue::Integer` arm. |
/// | `Float`            | `Float(SomeFloat)`     | `model.py:164-183` -> `SomeValue::Float` arm. |
/// | `Ref`              | `Instance(SomeInstance{classdef:None,..})` | `model.py:438` `SomeInstance.__init__(classdef, can_be_None=False, flags={})`.  `classdef=None` denotes the abstract `object`-only instance — the legitimate RPython placeholder when the bookkeeper has not narrowed the ClassDef.  Routes through `rclass.py:445-447 SomeInstance.rtyper_makerepr` -> `getinstancerepr(rtyper, None, Gc)` -> `InstanceRepr::new_rootinstance` -> `Ptr(GcStruct(OBJECT))` -> `lowleveltype_to_concrete` GC-arm -> `ConcreteType::GcRef`, matching legacy `resolve_types(Ref) -> GcRef`.  Convergence path: a producer (frontend / annotator with bookkeeper) attaches a richer `SomeInstance(classdef=Some(..))` via [`AnnotationState::set_some`] once the ClassDef is known. |
/// | `Void`             | `Impossible`           | `model.py:627` -> `SomeValue::Impossible` arm. |
/// | `State`            | `Instance(SomeInstance{classdef:None,..})` | **PRE-EXISTING-ADAPTATION (no upstream)**.  Pyre-only `State` carries the JIT state pointer (a struct pointer to interpreter state).  RPython has no analogue; the `SomeInstance(classdef=None)` projection is a temporary fallback that lets the rtyper proceed without a real bookkeeper-attached pyre `ClassDef`.  Projects to `GcRef` via the same chain as `Ref`. |
/// | `Unknown`          | `None`                 | **Fail-loud — annotation gap with no annotation-stage shell.**  RPython's annotator never produces an unknown lattice node — every Variable is annotated with a definite `SomeValue`, and unreachable code stays at `SomeImpossible`.  Pyre's `Unknown` is a coverage gap (annotator did not narrow / producer did not call `set_some`).  Returning `None` leaves `Variable.annotation` empty so `bindingrepr` panics with `KeyError: no binding for arg` (`annotator/annrpython.rs:418`) on the first attempt to lower the affected `ValueId`, surfacing the producer-side gap rather than silently bridging it to `GcRef` via a fabricated `SomeInstance(None)` shell — that bridging conflated an *annotation-stage* lattice node with the **legacy** `resolve_types(Unknown) -> ConcreteType::Unknown -> GcRef` resolver-stage backfill. |
///
/// Returns `None` only for `ValueType::Unknown`; every other variant
/// projects to a definite `SomeValue` shell.
pub fn valuetype_to_someshell(vt: &ValueType) -> Option<SomeValue> {
    match vt {
        ValueType::Int => Some(SomeValue::Integer(SomeInteger::default())),
        ValueType::Float => Some(SomeValue::Float(SomeFloat::default())),
        ValueType::Ref => {
            // RPython `SomeInstance(classdef=None, can_be_None=False,
            // flags={})` (model.py:438).  Upstream-orthodox abstract
            // instance for cases where the bookkeeper has not yet
            // narrowed the ClassDef.
            Some(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                BTreeMap::new(),
            )))
        }
        ValueType::State => {
            // PRE-EXISTING-ADAPTATION (no upstream).  Pyre's `State`
            // carries the JIT state pointer; RPython has no analogue.
            // `SomeInstance(classdef=None)` is a temporary fallback
            // that lets the rtyper resolve to `GcRef` without a real
            // bookkeeper-attached pyre `ClassDef`.
            Some(SomeValue::Instance(SomeInstance::new(
                None,
                false,
                BTreeMap::new(),
            )))
        }
        ValueType::Unknown => {
            // Fail-loud — annotation gap, NOT annotation-stage parity.
            // RPython's annotator never produces an unknown lattice
            // node; pyre's `Unknown` is a coverage gap.  Returning
            // `None` leaves `Variable.annotation` empty so the rtyper
            // panics at `bindingrepr` (annotator/annrpython.rs:418
            // "KeyError: no binding for arg") on the first attempt to
            // lower an Unknown ValueId.  This surfaces the producer-
            // side gap rather than silently bridging it to `GcRef` via
            // a fabricated `SomeInstance(None)` shell — that bridging
            // conflated the annotation-stage lattice node with the
            // **legacy** resolver-stage backfill
            // (`resolve_types(Unknown) -> ConcreteType::Unknown ->
            // GcRef`).  Convergence path: precise producer-side
            // `set_some` for every `ValueId`.
            None
        }
        ValueType::Void => Some(SomeValue::Impossible),
    }
}

/// Annotation state: `ValueId -> SomeValue` (orthodox) /
/// `ValueId -> ValueType` (legacy fallback).
#[derive(Debug, Clone)]
pub struct AnnotationState {
    pub types: HashMap<ValueId, ValueType>,
    /// Precise per-`ValueId` `SomeValue` — RPython's
    /// `Variable.annotation` analogue.  When present this overrides
    /// `types` for any consumer that wants the full lattice node.
    pub some_values: HashMap<ValueId, Rc<SomeValue>>,
}

impl AnnotationState {
    pub fn new() -> Self {
        Self {
            types: HashMap::new(),
            some_values: HashMap::new(),
        }
    }

    pub fn get(&self, id: ValueId) -> &ValueType {
        self.types.get(&id).unwrap_or(&ValueType::Unknown)
    }

    /// Set the legacy `ValueType` discriminator for `id` and update the
    /// matching `SomeValue` shell via [`valuetype_to_someshell`].
    ///
    /// RPython `RPythonAnnotator.setbinding(arg, s_value)`
    /// (`annrpython.py:289-294`) requires a re-binding to be a widening
    /// of the previous lattice node (`s_new ⊇ s_old`), and then
    /// replaces the binding.  Match that shape: non-`Unknown` shells are
    /// routed through [`Self::set_some`], which performs the containment
    /// assertion and overwrites the previous shell.
    ///
    /// `ValueType::Unknown` has no RPython annotation-stage
    /// counterpart.  If a producer sets it, clear any previous
    /// `some_values` entry so no stale shell survives; downstream
    /// `seed_variable` then leaves `Variable.annotation` empty and the
    /// rtyper fails at `bindingrepr`, surfacing the producer-side gap.
    pub fn set(&mut self, id: ValueId, ty: ValueType) {
        let shell = valuetype_to_someshell(&ty);
        self.types.insert(id, ty);
        if let Some(shell) = shell {
            self.set_some(id, Rc::new(shell));
        } else {
            self.some_values.remove(&id);
        }
    }

    /// Fetch the precise `SomeValue` lattice node for `id` if a
    /// producer attached one.  Mirrors `Variable.annotation` lookup:
    /// returns `None` when the producer left the slot empty (callers
    /// fall back to `get` + `valuetype_to_someshell`).
    pub fn some(&self, id: ValueId) -> Option<&Rc<SomeValue>> {
        self.some_values.get(&id)
    }

    /// Attach a precise `SomeValue` lattice node for `id`.  RPython
    /// equivalent: `RPythonAnnotator.setbinding(v, s_value)`
    /// (`rpython/annotator/annrpython.py:289-294`):
    ///
    /// ```python
    /// def setbinding(self, arg, s_value):
    ///     if arg in self.bindings:
    ///         assert s_value.contains(self.bindings[arg])
    ///     self.bindings[arg] = s_value
    /// ```
    ///
    /// The containment check enforces monotonicity: a re-binding must
    /// be a widening of the previous lattice node (`s_new ⊇ s_old`).
    /// A non-monotone re-binding indicates the producer has lost
    /// information between calls, which RPython treats as a hard
    /// error.
    pub fn set_some(&mut self, id: ValueId, value: Rc<SomeValue>) {
        if let Some(existing) = self.some_values.get(&id) {
            assert!(
                value.contains(existing.as_ref()),
                "AnnotationState::set_some: non-monotone re-binding at \
                 ValueId({:?}); new value {:?} does not contain previous \
                 value {:?} (annrpython.py:292)",
                id.0,
                value,
                existing.as_ref(),
            );
        }
        self.some_values.insert(id, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_replaces_existing_somevalue_when_new_shell_contains_old() {
        let mut state = AnnotationState::new();
        state.set(ValueId(1), ValueType::Ref);
        state.set(ValueId(1), ValueType::Ref);

        assert!(
            matches!(
                state.some(ValueId(1)).map(|s| s.as_ref()),
                Some(SomeValue::Instance(_))
            ),
            "Ref binding must remain represented by SomeInstance(classdef=None)"
        );
    }

    #[test]
    fn set_unknown_clears_existing_somevalue() {
        let mut state = AnnotationState::new();
        state.set(ValueId(1), ValueType::Int);
        assert!(state.some(ValueId(1)).is_some());

        state.set(ValueId(1), ValueType::Unknown);

        assert_eq!(state.get(ValueId(1)), &ValueType::Unknown);
        assert!(
            state.some(ValueId(1)).is_none(),
            "Unknown must clear the previous annotation shell so stale \
             SomeValue cannot reach Variable.annotation"
        );
    }

    #[test]
    #[should_panic(expected = "non-monotone re-binding")]
    fn set_panics_on_non_monotone_rebinding() {
        let mut state = AnnotationState::new();
        state.set(ValueId(1), ValueType::Int);
        state.set(ValueId(1), ValueType::Ref);
    }
}
