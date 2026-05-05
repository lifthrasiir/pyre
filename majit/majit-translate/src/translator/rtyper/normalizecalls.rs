//! Call-family signature / annotation normalization.
//!
//! RPython upstream: `rpython/rtyper/normalizecalls.py` (414 LOC, of
//! which lines 14-204, 266-295, and 373-389 cover the pieces this
//! module ports). The remaining class-PBC halves
//! (`merge_classpbc_getattr_into_classdef`, `create_class_constructors`)
//! land with the rtyper specialization epic; they depend on
//! `rpython/rtyper/rclass.py` infrastructure that pyre still has as
//! scaffolding only. [`perform_normalizations`] is the driver entry
//! point and runs the call-family, inheritance-id, and
//! instantiate-function halves that are ported in this module.
//!
//! ## What is ported here (upstream lines 14-204 + 373-389)
//!
//! - [`normalize_call_familes`] — outer loop over
//!   `bookkeeper.pbc_maximal_call_families.infos()` (upstream line 14-21).
//! - [`normalize_calltable`] — iterate shapes/rows, delegate to the
//!   row-level normalizers (upstream line 23-42).
//! - [`raise_call_table_too_complex_error`] — build TyperError message
//!   for families that span multiple shapes (upstream line 44-76).
//! - [`normalize_calltable_row_signature`] — argument-order
//!   normalization across a row (upstream line 78-154).
//! - [`normalize_calltable_row_annotation`] — annotation-union
//!   generalization across a row (upstream line 156-204).
//! - [`assign_inheritance_ids`] — reversed-MRO witness ordering for the
//!   `classdef.minid` / `classdef.maxid` subclass-range brackets
//!   consumed by `rclass.py:ll_issubclass_const` (upstream line 373-389).
//! - [`create_instantiate_function`] / [`create_instantiate_functions`]
//!   — build the `my_instantiate_graph` helper consumed by
//!   `ClassRepr.fill_vtable_root` (upstream line 266-295). The source
//!   of the `bookkeeper.needs_generic_instantiate` set is
//!   [`merge_classpbc_getattr_into_classdef`] (this module), invoked
//!   from the [`perform_normalizations`] driver. Tests can also
//!   populate the set manually to exercise this entry point in
//!   isolation.
//!
//! ## Upstream call-family clarification
//!
//! Per the user's plan correction (session transcript):
//!
//! > `normalizecalls.py`는 graph monomorphization이 아니라
//! > signature/annotation normalization만 한다.
//!
//! The job of this module is _signature agreement_ across the families
//! that the annotator has already built — e.g. two `__init__` methods on
//! different classes in the same `pbc_maximal_call_families` bucket need
//! matching argnames so the shared PBC dispatcher can call either one.
//! **It does NOT specialize generic graphs.** Specialization (what
//! actually turns `opcode_*<H>` into `opcode_*::<PyFrame>`) lives in
//! `rpython/rtyper/rpbc.py` `SingleFrozenPBCRepr` /
//! `MultipleFrozenPBCRepr` / `SmallFunctionSetPBCRepr` — that is the
//! Commit 4 target.

use crate::annotator::annrpython::RPythonAnnotator;
use crate::annotator::classdesc::{ClassDef, ClassDesc};
use crate::annotator::description::{CallFamily, CallTableRow, ClassAttrFamily, DescEntry};
use crate::annotator::model::AnnotatorError;
use crate::flowspace::argument::{CallShape, Signature};
use crate::flowspace::model::{
    Block, BlockKey, BlockRefExt, ConstValue, Constant, FunctionGraph, GraphKey, Hlvalue, Link,
    Variable, checkgraph,
};
use crate::flowspace::pygraph::PyGraph;

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// RPython `normalize_call_familes(annotator)` (normalizecalls.py:14-21).
///
/// ```python
/// def normalize_call_familes(annotator):
///     for callfamily in annotator.bookkeeper.pbc_maximal_call_families.infos():
///         if not callfamily.modified:
///             assert callfamily.normalized
///             continue
///         normalize_calltable(annotator, callfamily)
///         callfamily.normalized = True
///         callfamily.modified = False
/// ```
pub fn normalize_call_familes(annotator: &RPythonAnnotator) -> Result<(), AnnotatorError> {
    // RPython: `annotator.bookkeeper.pbc_maximal_call_families.infos()`.
    // Pyre `UnionFind::infos()` yields `&Rc<RefCell<CallFamily>>`; we
    // clone the Rcs into a local Vec so the borrow on
    // `pbc_maximal_call_families` can be released before the inner
    // `normalize_calltable` call (which itself may invoke bookkeeper
    // helpers that re-borrow the UnionFind).
    let families: Vec<_> = {
        let bk = &annotator.bookkeeper;
        let pbc = bk.pbc_maximal_call_families.borrow();
        pbc.infos().cloned().collect()
    };
    for callfamily_rc in families {
        // RPython: `if not callfamily.modified: assert callfamily.normalized; continue`.
        {
            let cf = callfamily_rc.borrow();
            if !cf.modified {
                debug_assert!(
                    cf.normalized,
                    "CallFamily invariant: modified=false implies normalized=true"
                );
                continue;
            }
        }
        // RPython: `normalize_calltable(annotator, callfamily)`.
        normalize_calltable(annotator, &callfamily_rc)?;
        // RPython: `callfamily.normalized = True; callfamily.modified = False`.
        let mut cf = callfamily_rc.borrow_mut();
        cf.normalized = true;
        cf.modified = false;
    }
    Ok(())
}

/// RPython `perform_normalizations(annotator)` (normalizecalls.py:404-413).
///
/// ```python
/// def perform_normalizations(annotator):
///     create_class_constructors(annotator)
///     annotator.frozen += 1
///     try:
///         normalize_call_familes(annotator)
///         merge_classpbc_getattr_into_classdef(annotator)
///         assign_inheritance_ids(annotator)
///     finally:
///         annotator.frozen -= 1
///     create_instantiate_functions(annotator)
/// ```
///
/// The Rust port wires the driver entry point to the call-family
/// normalization pass + [`assign_inheritance_ids`] +
/// [`create_instantiate_functions`]. The class-constructor and
/// `merge_classpbc_getattr_into_classdef` phases still depend on rclass
/// / PBC infrastructure that is not in this tree yet — the
/// `bookkeeper.needs_generic_instantiate` set therefore stays empty
/// until a caller populates it explicitly, which makes
/// `create_instantiate_functions` a no-op in the current pipeline.
pub fn perform_normalizations(annotator: &RPythonAnnotator) -> Result<(), AnnotatorError> {
    struct FrozenGuard<'a> {
        annotator: &'a RPythonAnnotator,
        saved: bool,
    }

    impl<'a> Drop for FrozenGuard<'a> {
        fn drop(&mut self) {
            *self.annotator.frozen.borrow_mut() = self.saved;
        }
    }

    // Upstream increments `annotator.frozen` around the middle
    // normalization section and restores it in `finally`. Rust models
    // the field as a bool, so preserve the old value for nested callers.
    {
        let old_frozen = std::mem::replace(&mut *annotator.frozen.borrow_mut(), true);
        let _guard = FrozenGuard {
            annotator,
            saved: old_frozen,
        };
        normalize_call_familes(annotator)?;
        merge_classpbc_getattr_into_classdef(annotator)?;
        assign_inheritance_ids(annotator);
    }
    // upstream: `create_instantiate_functions(annotator)` runs AFTER
    // the frozen-try block so the newly-built graphs register with the
    // annotator's non-frozen state.
    create_instantiate_functions(annotator)?;
    Ok(())
}

/// RPython `assign_inheritance_ids(annotator)` (normalizecalls.py:373-389).
///
/// ```python
/// def assign_inheritance_ids(annotator):
///     bk = annotator.bookkeeper
///     try:
///         lst = bk._inheritance_id_symbolics
///     except AttributeError:
///         lst = bk._inheritance_id_symbolics = []
///     for classdef in annotator.bookkeeper.classdefs:
///         if not hasattr(classdef, 'minid'):
///             witness = [get_unique_cdef_id(cdef) for cdef in classdef.getmro()]
///             witness.reverse()
///             classdef.minid = TotalOrderSymbolic(witness, lst)
///             classdef.maxid = TotalOrderSymbolic(witness + [MAX], lst)
/// ```
///
/// Upstream stores `TotalOrderSymbolic` objects, not eager integers. A
/// later numeric comparison sorts all start/end markers by the reversed
/// MRO witness `([root_id, ..., self_id], [root_id, ..., self_id, MAX])`.
///
/// The Rust port still stores plain `i64`s on `ClassDef`, but it keeps
/// the upstream ordering rule: assign a stable per-classdef unique id,
/// build the reversed-MRO witness for every classdef in the current
/// snapshot, sort all start/end markers lexicographically, then write
/// the resulting integer positions back to `minid` / `maxid`. This
/// preserves the bracket invariant `c.minid <= d.minid <= d.maxid <=
/// c.maxid` for every descendant `d` of `c`, including when a later run
/// adds a new subclass under an already-seen parent.
/// RPython `merge_classpbc_getattr_into_classdef(annotator)`
/// (normalizecalls.py:208-235).
///
/// ```python
/// def merge_classpbc_getattr_into_classdef(annotator):
///     all_families = annotator.bookkeeper.classpbc_attr_families
///     for attrname, access_sets in all_families.items():
///         for access_set in access_sets.infos():
///             descs = access_set.descs
///             if len(descs) <= 1:
///                 continue
///             if not isinstance(descs.iterkeys().next(), ClassDesc):
///                 continue
///             classdefs = [desc.getuniqueclassdef() for desc in descs]
///             commonbase = classdefs[0]
///             for cdef in classdefs[1:]:
///                 commonbase = commonbase.commonbase(cdef)
///                 if commonbase is None:
///                     raise TyperError("reading attribute %r: no common base "
///                                      "class for %r" % (attrname, descs.keys()))
///             extra_access_sets = commonbase.extra_access_sets
///             if commonbase.repr is not None:
///                 assert access_set in extra_access_sets
///                 continue
///             access_set.commonbase = commonbase
///             if access_set not in extra_access_sets:
///                 counter = len(extra_access_sets)
///                 extra_access_sets[access_set] = attrname, counter
/// ```
///
/// For every class-PBC attribute access set this walks the matching
/// classdefs, computes the deepest common ancestor, and stamps that
/// `commonbase` onto both sides of the connection — onto the
/// `ClassAttrFamily` itself (consumed by `ClassesPBCRepr.get_access_set`)
/// and into `commonbase.extra_access_sets` (consumed by
/// `ClassRepr._setup_repr_pbcfields`, rclass.py:280-285).
///
/// Pyre lookup deviation: upstream `descs.iterkeys().next()` exploits
/// Python's hashed `Desc` identity; in Rust the family stores
/// [`crate::annotator::description::DescKey`] values only, so we
/// resolve them back to live `ClassDesc` `Rc`s by walking
/// `bookkeeper.descs` (the by-`HostObject` registry) and matching
/// keys. This is O(|bk.descs| * |access_set.descs|) per family, which
/// is fine for the small per-attr access set populations and runs
/// once per `perform_normalizations` invocation.
pub fn merge_classpbc_getattr_into_classdef(
    annotator: &RPythonAnnotator,
) -> Result<(), AnnotatorError> {
    use std::collections::HashSet;
    let bk = &annotator.bookkeeper;

    // Snapshot `(attrname, access_set)` pairs so we can release the
    // outer `classpbc_attr_families` borrow before mutating individual
    // families / commonbases below — the live `infos()` iterator holds
    // a borrow on the UnionFind storage.
    let pairs: Vec<(String, Rc<RefCell<ClassAttrFamily>>)> = {
        let map = bk.classpbc_attr_families.borrow();
        let mut out = Vec::new();
        for (attrname, uf) in map.iter() {
            for family in uf.infos() {
                out.push((attrname.clone(), family.clone()));
            }
        }
        out
    };

    for (attrname, access_set) in pairs {
        // upstream `descs = access_set.descs; if len(descs) <= 1: continue`.
        let desc_keys: HashSet<crate::annotator::description::DescKey> = {
            let caf = access_set.borrow();
            caf.descs.keys().copied().collect()
        };
        if desc_keys.len() <= 1 {
            continue;
        }

        // upstream `if not isinstance(descs.iterkeys().next(), ClassDesc):
        // continue`. Walk `bookkeeper.descs` to resolve each DescKey
        // back to its live `Rc<RefCell<ClassDesc>>`. Non-Class descs
        // that share the family are filtered out — but in practice the
        // family is built from a uniform-kind PBC, so either all match
        // or none does.
        let class_descs: Vec<Rc<RefCell<ClassDesc>>> = {
            let descs_map = bk.descs.borrow();
            let mut out = Vec::new();
            for entry in descs_map.values() {
                if let DescEntry::Class(rc) = entry {
                    let key = crate::annotator::description::DescKey::from_rc(rc);
                    if desc_keys.contains(&key) {
                        out.push(rc.clone());
                    }
                }
            }
            out
        };
        // Strictness: upstream rpython/rtyper/normalizecalls.py:217-219
        // peeks at `descs.iterkeys().next()`. If the first desc is not
        // a `ClassDesc`, the whole family is skipped — non-class
        // PBCs share their access families through different reprs
        // (FrozenAttrFamily etc.), so this branch is the right
        // skip-path. The `class_descs.is_empty()` arm models that.
        //
        // The pyre port resolves DescKeys → live `ClassDesc`s by
        // walking `bookkeeper.descs`; if every key matched a non-class
        // entry we end up with `class_descs.is_empty()` and skip
        // matches upstream. But if SOME keys matched class entries
        // and others didn't, we'd be silently dropping descs the
        // family thinks belong to it. That is upstream-impossible:
        // an attr family is built from descs of a single kind.
        // Surface a structured error rather than silently skipping
        // the partial-match.
        if class_descs.is_empty() {
            continue;
        }
        if class_descs.len() != desc_keys.len() {
            return Err(AnnotatorError::new(format!(
                "merge_classpbc_getattr_into_classdef: mixed-kind family for \
                 attr {:?} — {} descs in family, {} resolve to ClassDesc \
                 in bookkeeper.descs",
                attrname,
                desc_keys.len(),
                class_descs.len(),
            )));
        }
        if class_descs.len() <= 1 {
            continue;
        }

        // upstream `classdefs = [desc.getuniqueclassdef() for desc in descs]`
        // followed by `commonbase = classdefs[0]; for cdef in classdefs[1:]:
        //     commonbase = commonbase.commonbase(cdef); if commonbase is None: raise TyperError(...)`.
        let mut commonbase = ClassDesc::getuniqueclassdef(&class_descs[0])?;
        for other in &class_descs[1..] {
            let cdef = ClassDesc::getuniqueclassdef(other)?;
            commonbase = match ClassDef::commonbase(&commonbase, &cdef) {
                Some(b) => b,
                None => {
                    return Err(AnnotatorError::new(format!(
                        "reading attribute {:?}: no common base class for {} class descriptions",
                        attrname,
                        class_descs.len()
                    )));
                }
            };
        }

        // upstream `extra_access_sets = commonbase.extra_access_sets;
        // if commonbase.repr is not None: assert access_set in extra_access_sets; continue`.
        let access_set_key = Rc::as_ptr(&access_set) as usize;
        {
            let cb = commonbase.borrow();
            if cb.repr.is_some() {
                // upstream rpython/rtyper/normalizecalls.py:230 —
                // hard `assert access_set in extra_access_sets`. The
                // pyre port mirrors that with a runtime `assert!`
                // (not `debug_assert!`) so release builds preserve
                // the invariant.
                assert!(
                    cb.extra_access_sets.contains_key(&access_set_key),
                    "merge_classpbc_getattr_into_classdef invariant: commonbase \
                     {:?} already has a repr but access_set is not in its \
                     extra_access_sets",
                    cb.name
                );
                continue;
            }
        }

        // upstream `access_set.commonbase = commonbase`.
        access_set.borrow_mut().commonbase = Some(commonbase.clone());

        // upstream `if access_set not in extra_access_sets:
        //     counter = len(extra_access_sets);
        //     extra_access_sets[access_set] = attrname, counter`.
        let mut cb_mut = commonbase.borrow_mut();
        if !cb_mut.extra_access_sets.contains_key(&access_set_key) {
            let counter = cb_mut.extra_access_sets.len();
            cb_mut.extra_access_sets.insert(
                access_set_key,
                (access_set.clone(), attrname.clone(), counter),
            );
        }
    }

    Ok(())
}

pub fn assign_inheritance_ids(annotator: &RPythonAnnotator) {
    let snapshot: Vec<Rc<RefCell<ClassDef>>> = annotator
        .bookkeeper
        .classdefs
        .borrow()
        .iter()
        .cloned()
        .collect();

    if snapshot.is_empty() {
        return;
    }

    let mut markers: Vec<InheritanceMarker> = Vec::with_capacity(snapshot.len() * 2);
    for classdef in &snapshot {
        let mut witness = classdef_order_witness(classdef);
        markers.push(InheritanceMarker {
            orderwitness: witness.clone(),
            classdef: classdef.clone(),
            is_max: false,
        });
        witness.push(OrderWitnessAtom::Max);
        markers.push(InheritanceMarker {
            orderwitness: witness,
            classdef: classdef.clone(),
            is_max: true,
        });
    }
    markers.sort_by(|left, right| left.orderwitness.cmp(&right.orderwitness));

    for (index, marker) in markers.into_iter().enumerate() {
        let mut classdef = marker.classdef.borrow_mut();
        if marker.is_max {
            classdef.maxid = Some(index as i64);
        } else {
            classdef.minid = Some(index as i64);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum OrderWitnessAtom {
    Id(usize),
    Max,
}

#[derive(Clone)]
struct InheritanceMarker {
    orderwitness: Vec<OrderWitnessAtom>,
    classdef: Rc<RefCell<ClassDef>>,
    is_max: bool,
}

fn classdef_order_witness(classdef: &Rc<RefCell<ClassDef>>) -> Vec<OrderWitnessAtom> {
    let mut witness: Vec<OrderWitnessAtom> = ClassDef::getmro(classdef)
        .into_iter()
        .map(|cdef| OrderWitnessAtom::Id(get_unique_cdef_id(&cdef)))
        .collect();
    witness.reverse();
    witness
}

fn get_unique_cdef_id(classdef: &Rc<RefCell<ClassDef>>) -> usize {
    static NEXT_CLASSDEF_ID: AtomicUsize = AtomicUsize::new(0);

    if let Some(existing) = classdef.borrow().unique_cdef_id {
        return existing;
    }
    let fresh = NEXT_CLASSDEF_ID.fetch_add(1, Ordering::Relaxed);
    classdef.borrow_mut().unique_cdef_id = Some(fresh);
    fresh
}

/// RPython `normalize_calltable(annotator, callfamily)`
/// (normalizecalls.py:23-42).
///
/// ```python
/// def normalize_calltable(annotator, callfamily):
///     """Try to normalize all rows of a table."""
///     nshapes = len(callfamily.calltables)
///     for shape, table in callfamily.calltables.items():
///         for row in table:
///             did_something = normalize_calltable_row_signature(annotator, shape, row)
///             if did_something:
///                 assert not callfamily.normalized, "change in call family normalisation"
///                 if nshapes != 1:
///                     raise_call_table_too_complex_error(callfamily, annotator)
///     while True:
///         progress = False
///         for shape, table in callfamily.calltables.items():
///             for row in table:
///                 progress |= normalize_calltable_row_annotation(annotator, row.values())
///         if not progress:
///             return   # done
///         assert not callfamily.normalized, "change in call family normalisation"
/// ```
fn normalize_calltable(
    annotator: &RPythonAnnotator,
    callfamily_rc: &std::rc::Rc<std::cell::RefCell<CallFamily>>,
) -> Result<(), AnnotatorError> {
    // upstream line 25: `nshapes = len(callfamily.calltables)`. Clone the
    // (shape, row) pairs up front so the inner normalize_* passes can
    // receive a `&RPythonAnnotator` without the bookkeeper borrow still
    // held through the iteration.
    let (nshapes, shaped_rows): (usize, Vec<(CallShape, Vec<CallTableRow>)>) = {
        let cf = callfamily_rc.borrow();
        let pairs: Vec<_> = cf
            .calltables
            .iter()
            .map(|(shape, table)| (shape.clone(), table.clone()))
            .collect();
        (cf.calltables.len(), pairs)
    };

    // upstream line 26-33: per-row signature normalization.
    for (shape, table) in &shaped_rows {
        for row in table {
            let did_something = normalize_calltable_row_signature(annotator, shape, row)?;
            if did_something {
                // upstream line 31: `assert not callfamily.normalized, ...`.
                debug_assert!(
                    !callfamily_rc.borrow().normalized,
                    "change in call family normalisation"
                );
                if nshapes != 1 {
                    return Err(raise_call_table_too_complex_error(
                        &callfamily_rc.borrow(),
                        annotator,
                    ));
                }
            }
        }
    }

    // upstream line 34-42: `while True` fixpoint on row annotations.
    loop {
        let mut progress = false;
        // Refresh the shaped_rows snapshot each iteration — upstream
        // iterates `callfamily.calltables.items()` live, which does not
        // change during annotation normalization but may on signature
        // rewrite (row_signature can run before row_annotation in the
        // same call).
        let snapshot: Vec<_> = {
            let cf = callfamily_rc.borrow();
            cf.calltables
                .iter()
                .map(|(shape, table)| (shape.clone(), table.clone()))
                .collect()
        };
        for (_shape, table) in &snapshot {
            for row in table {
                let graphs: Vec<Rc<PyGraph>> = row.values().cloned().collect();
                progress |= normalize_calltable_row_annotation(annotator, &graphs)?;
            }
        }
        if !progress {
            return Ok(());
        }
        // upstream line 42: `assert not callfamily.normalized, ...`.
        debug_assert!(
            !callfamily_rc.borrow().normalized,
            "change in call family normalisation"
        );
    }
}

/// RPython `raise_call_table_too_complex_error(callfamily, annotator)`
/// (normalizecalls.py:44-76).
///
/// Called when a single call family carries rows with differing
/// `CallShape`s — the PBC dispatcher cannot bridge them, so the
/// annotator must stop and report the problem. Upstream raises
/// `TyperError`; pyre returns [`AnnotatorError`] carrying the same
/// multi-line message shape.
pub(crate) fn raise_call_table_too_complex_error(
    callfamily: &CallFamily,
    annotator: &RPythonAnnotator,
) -> AnnotatorError {
    let mut msg: Vec<String> = Vec::new();
    // upstream: `items = callfamily.calltables.items()` — order is
    // python-dict-insertion but the Rust HashMap does not preserve
    // that. Sort by shape to get a deterministic diagnostic output
    // that still covers every pair.
    let mut items: Vec<(&CallShape, &Vec<CallTableRow>)> = callfamily.calltables.iter().collect();
    items.sort_by(|(a, _), (b, _)| {
        a.shape_cnt
            .cmp(&b.shape_cnt)
            .then_with(|| a.shape_keys.cmp(&b.shape_keys))
            .then_with(|| a.shape_star.cmp(&b.shape_star))
    });
    for (i, (shape1, table1)) in items.iter().enumerate() {
        for (shape2, table2) in items.iter().skip(i + 1) {
            // upstream: `if shape1 == shape2: continue`.
            if shape1 == shape2 {
                continue;
            }
            // upstream: `row1 = table1[0]; row2 = table2[0]`.
            let Some(row1) = table1.first() else { continue };
            let Some(row2) = table2.first() else { continue };
            // upstream:
            //   problematic_function_graphs = set(row1.values()).union(set(row2.values()))
            //   pfg = [str(graph) for graph in problematic_function_graphs]; pfg.sort()
            let mut pfg: Vec<String> = row1
                .values()
                .chain(row2.values())
                .map(|g| g.graph.borrow().name.clone())
                .collect();
            pfg.sort();
            pfg.dedup();
            msg.push("the following functions:".to_string());
            msg.push(format!("    {}", pfg.join("\n    ")));
            msg.push("are called with inconsistent numbers of arguments".to_string());
            msg.push(
                "(and/or the argument names are different, which is not \
                 supported in this case)"
                    .to_string(),
            );
            if shape1.shape_cnt != shape2.shape_cnt {
                msg.push(format!(
                    "sometimes with {} arguments, sometimes with {}",
                    shape1.shape_cnt, shape2.shape_cnt
                ));
            }
            // upstream: `callers = []; ...` — iterate
            // `annotator.translator.callgraph.iteritems()` and collect
            // unique caller names whose callee is in
            // `problematic_function_graphs`.
            let problematic_graphs: Vec<GraphKey> = row1
                .values()
                .chain(row2.values())
                .map(|g| GraphKey::of(&g.graph))
                .collect();
            let mut callers: Vec<String> = annotator
                .translator
                .callgraph
                .borrow()
                .values()
                .filter(|edge| problematic_graphs.contains(&GraphKey::of(&edge.callee)))
                .map(|edge| edge.caller.borrow().name.clone())
                .collect();
            callers.sort();
            callers.dedup();
            msg.push("the callers of these functions are:".to_string());
            for caller in callers {
                msg.push(format!("    {caller}"));
            }
        }
    }
    AnnotatorError::new(msg.join("\n"))
}

/// RPython `normalize_calltable_row_signature(annotator, shape, row)`
/// (normalizecalls.py:78-154).
///
/// Returns `true` if the row was rewritten (triggering
/// `callfamily.normalized = False` re-check upstream).
///
/// Upstream short-circuits on line 83-89 via the `for...else` idiom:
/// if every graph in the row has the same `signature` and `defaults`
/// as the first, return `False` immediately. When a graph disagrees,
/// line 91-153 rebuilds its startblock so call-family peers expose the
/// same argument order/default surface. This port now performs the
/// same rewrite.
pub(crate) fn normalize_calltable_row_signature(
    annotator: &RPythonAnnotator,
    shape: &CallShape,
    row: &CallTableRow,
) -> Result<bool, AnnotatorError> {
    let _ = annotator;
    // upstream line 79: `graphs = row.values(); assert graphs, "no graph??"`.
    let graphs: Vec<&Rc<PyGraph>> = row.values().collect();
    assert!(!graphs.is_empty(), "no graph??");

    // upstream line 81-89: fast path — all signatures + defaults match.
    let sig0 = graphs[0].signature.borrow().clone();
    let defaults0 = graphs[0].defaults.borrow().clone();
    let all_match = graphs
        .iter()
        .skip(1)
        .all(|g| *g.signature.borrow() == sig0 && *g.defaults.borrow() == defaults0);
    if all_match {
        // upstream line 89: `return False`.
        return Ok(false);
    }

    // upstream line 91-92: `shape_cnt, shape_keys, shape_star = shape;
    // assert not shape_star`.
    assert!(
        !shape.shape_star,
        "shape_star should have been removed at this stage"
    );

    let call_nbargs = shape.shape_cnt + shape.shape_keys.len();
    let mut did_something = false;

    for graph in graphs {
        let signature = graph.signature.borrow().clone();
        let argnames = signature.argnames;
        assert!(
            signature.varargname.is_none(),
            "normalize_calltable_row_signature: vararg not implemented"
        );
        assert!(
            signature.kwargname.is_none(),
            "normalize_calltable_row_signature: kwarg not implemented"
        );

        let graph_args = graph.graph.borrow().getargs();
        let inputargs_s: Vec<_> = graph_args.iter().map(|v| annotator.binding(v)).collect();
        let mut argorder: Vec<usize> = (0..shape.shape_cnt).collect();
        for key in &shape.shape_keys {
            let i = argnames
                .iter()
                .position(|name| name == key)
                .ok_or_else(|| {
                    AnnotatorError::new(format!(
                        "normalize_calltable_row_signature: arg {key:?} not found in graph"
                    ))
                })?;
            assert!(!argorder.contains(&i));
            argorder.push(i);
        }
        let need_reordering = argorder != (0..call_nbargs).collect::<Vec<_>>();
        if !need_reordering && graph_args.len() == call_nbargs {
            continue;
        }

        let oldblock = graph.graph.borrow().startblock.clone();
        let old_annotated = annotator
            .annotated
            .borrow()
            .get(&BlockKey::of(&oldblock))
            .cloned()
            .unwrap_or(None);
        let defaults = graph.defaults.borrow().clone().unwrap_or_default();
        let num_nondefaults = inputargs_s.len().saturating_sub(defaults.len());
        let mut padded_defaults = vec![Constant::new(ConstValue::Placeholder); num_nondefaults];
        padded_defaults.extend(defaults);
        let mut newdefaults: Vec<Constant> = Vec::new();
        let mut inlist: Vec<Hlvalue> = Vec::new();

        for &j in &argorder {
            let mut v = match &graph_args[j] {
                Hlvalue::Variable(v) => {
                    let mut fresh = Variable::new();
                    fresh.set_name_from(v);
                    fresh
                }
                Hlvalue::Constant(_) => {
                    return Err(AnnotatorError::new(
                        "normalize_calltable_row_signature: expected Variable arg",
                    ));
                }
            };
            annotator.setbinding(&mut v, inputargs_s[j].clone());
            inlist.push(Hlvalue::Variable(v));
            newdefaults.push(padded_defaults[j].clone());
        }

        let newblock = Block::shared(inlist.clone());
        let mut outlist = inlist[..shape.shape_cnt].to_vec();
        for j in shape.shape_cnt..inputargs_s.len() {
            if let Some(i) = argorder.iter().position(|idx| *idx == j) {
                outlist.push(inlist[i].clone());
                continue;
            }
            let default = padded_defaults[j].clone();
            if matches!(default.value, ConstValue::Placeholder) {
                return Err(AnnotatorError::new(format!(
                    "call pattern has {} positional arguments, but {:?} takes at least {} arguments",
                    shape.shape_cnt,
                    graph.graph.borrow().name,
                    num_nondefaults
                )));
            }
            outlist.push(Hlvalue::Constant(default));
        }

        let link = Rc::new(std::cell::RefCell::new(Link::new(
            outlist,
            Some(oldblock.clone()),
            None,
        )));
        newblock.closeblock(vec![link]);
        graph.graph.borrow_mut().startblock = newblock.clone();

        let mut trimmed_defaults = newdefaults;
        for i in (0..trimmed_defaults.len()).rev() {
            if matches!(trimmed_defaults[i].value, ConstValue::Placeholder) {
                trimmed_defaults = trimmed_defaults[i..].to_vec();
                break;
            }
        }
        *graph.signature.borrow_mut() = Signature::new(
            argorder.iter().map(|&j| argnames[j].clone()).collect(),
            None,
            None,
        );
        *graph.defaults.borrow_mut() = Some(trimmed_defaults);

        checkgraph(&graph.graph.borrow());
        let new_key = BlockKey::of(&newblock);
        annotator
            .annotated
            .borrow_mut()
            .insert(new_key.clone(), old_annotated.clone());
        annotator.all_blocks.borrow_mut().insert(new_key, newblock);
        did_something = true;
    }

    Ok(did_something)
}

/// RPython `normalize_calltable_row_annotation(annotator, graphs)`
/// (normalizecalls.py:156-204).
///
/// Returns `true` if any graph's argument or return binding was
/// widened (triggering the outer `while True` to re-run).
///
/// Upstream line 180-198 writes a replacement startblock for every
/// graph whose bindings lost information relative to the union. This
/// port now performs the same graph rewrite and returnvar widening.
pub(crate) fn normalize_calltable_row_annotation(
    annotator: &RPythonAnnotator,
    graphs: &[Rc<PyGraph>],
) -> Result<bool, AnnotatorError> {
    use crate::annotator::model::unionof;

    // upstream line 157: `if len(graphs) <= 1: return False`.
    if graphs.len() <= 1 {
        return Ok(false);
    }

    // upstream line 159-162:
    //   graph_bindings = {}
    //   for graph in graphs:
    //       graph_bindings[graph] = [annotator.binding(v) for v in graph.getargs()]
    let graph_bindings: Vec<Vec<_>> = graphs
        .iter()
        .map(|g| {
            g.graph
                .borrow()
                .getargs()
                .iter()
                .map(|v| annotator.binding(v))
                .collect()
        })
        .collect();

    // upstream line 163-166: `nbargs = len(...); assert len(binding) == nbargs`.
    let nbargs = graph_bindings[0].len();
    for binding in &graph_bindings {
        assert_eq!(binding.len(), nbargs, "inconsistent arg counts in row");
    }

    // upstream line 168-174: `generalizedargs` is the union of each
    // column. Pyre's `unionof` is N-ary (accepts any iterator) and
    // returns `Result<SomeValue, UnionError>` — the `UnionError → AnnotatorError`
    // bridge is spelled explicitly because there is no blanket `From`
    // impl between the two (UnionError is annotator-internal).
    let mut generalizedargs = Vec::with_capacity(nbargs);
    for i in 0..nbargs {
        let column: Vec<&_> = graph_bindings.iter().map(|b| &b[i]).collect();
        let s_value =
            unionof(column.into_iter()).map_err(|e| AnnotatorError::new(e.to_string()))?;
        generalizedargs.push(s_value);
    }

    // upstream line 175-177: `generalizedresult = unionof(*result_s)`.
    let result_s: Vec<_> = graphs
        .iter()
        .map(|g| annotator.binding(&g.graph.borrow().getreturnvar()))
        .collect();
    let generalizedresult =
        unionof(result_s.iter()).map_err(|e| AnnotatorError::new(e.to_string()))?;

    // upstream line 179-204: `conversion = False; for graph in graphs: ...`.
    let mut conversion = false;
    for (graph, bindings) in graphs.iter().zip(&graph_bindings) {
        if *bindings != generalizedargs {
            conversion = true;
            let oldblock = graph.graph.borrow().startblock.clone();
            let old_annotated = annotator
                .annotated
                .borrow()
                .get(&BlockKey::of(&oldblock))
                .cloned()
                .unwrap_or(None);
            let graph_args = graph.graph.borrow().getargs();
            let mut inlist = Vec::with_capacity(generalizedargs.len());
            for (j, s_value) in generalizedargs.iter().enumerate() {
                let mut v = match &graph_args[j] {
                    Hlvalue::Variable(v) => {
                        let mut fresh = Variable::new();
                        fresh.set_name_from(v);
                        fresh
                    }
                    Hlvalue::Constant(_) => {
                        return Err(AnnotatorError::new(
                            "normalize_calltable_row_annotation: expected Variable arg",
                        ));
                    }
                };
                annotator.setbinding(&mut v, s_value.clone());
                inlist.push(Hlvalue::Variable(v));
            }
            let newblock = Block::shared(inlist.clone());
            let link = Rc::new(std::cell::RefCell::new(Link::new(
                inlist,
                Some(oldblock.clone()),
                None,
            )));
            newblock.closeblock(vec![link]);
            graph.graph.borrow_mut().startblock = newblock.clone();
            checkgraph(&graph.graph.borrow());
            let new_key = BlockKey::of(&newblock);
            annotator
                .annotated
                .borrow_mut()
                .insert(new_key.clone(), old_annotated.clone());
            annotator.all_blocks.borrow_mut().insert(new_key, newblock);
        }
        let fg = graph.graph.borrow_mut();
        if let Hlvalue::Variable(v) = &mut fg.returnblock.borrow_mut().inputargs[0] {
            if annotator.binding(&Hlvalue::Variable(v.clone())) != generalizedresult {
                conversion = true;
                annotator.setbinding(v, generalizedresult.clone());
            }
        }
    }

    Ok(conversion)
}

/// RPython `create_instantiate_functions(annotator)`
/// (normalizecalls.py:266-273).
///
/// ```python
/// def create_instantiate_functions(annotator):
///     needs_generic_instantiate = annotator.bookkeeper.needs_generic_instantiate
///     for classdef in needs_generic_instantiate:
///         assert getgcflavor(classdef) == 'gc'   # only gc-case
///         create_instantiate_function(annotator, classdef)
/// ```
pub fn create_instantiate_functions(annotator: &RPythonAnnotator) -> Result<(), AnnotatorError> {
    // Snapshot the pending classdefs so the inner helper can mutate
    // `annotator.bookkeeper.needs_generic_instantiate` without
    // invalidating the iteration (upstream's Python dict iteration
    // keeps pointing at the original keys even if callers push more).
    let pending: Vec<Rc<RefCell<ClassDef>>> = annotator
        .bookkeeper
        .needs_generic_instantiate
        .borrow()
        .values()
        .cloned()
        .collect();
    for classdef in pending {
        // upstream: `assert getgcflavor(classdef) == 'gc'`.
        let flavor = super::rclass::getgcflavor(&classdef)
            .map_err(|e| AnnotatorError::new(e.to_string()))?;
        if flavor != super::rclass::Flavor::Gc {
            return Err(AnnotatorError::new(format!(
                "create_instantiate_functions: classdef {:?} has gc flavor {:?}, expected Gc",
                classdef.borrow().name,
                flavor,
            )));
        }
        create_instantiate_function(annotator, &classdef)?;
    }
    Ok(())
}

/// RPython `create_instantiate_function(annotator, classdef)`
/// (normalizecalls.py:275-295).
///
/// ```python
/// def create_instantiate_function(annotator, classdef):
///     if hasattr(classdef, 'my_instantiate_graph'):
///         return
///     v = Variable()
///     block = Block([])
///     block.operations.append(SpaceOperation('instantiate1', [], v))
///     name = valid_identifier('instantiate_' + classdef.name)
///     graph = FunctionGraph(name, block)
///     block.closeblock(Link([v], graph.returnblock))
///     annotator.setbinding(v, annmodel.SomeInstance(classdef))
///     annotator.annotated[block] = graph
///     generalizedresult = annmodel.SomeInstance(classdef=None)
///     annotator.setbinding(graph.getreturnvar(), generalizedresult)
///     classdef.my_instantiate_graph = graph
///     annotator.translator.graphs.append(graph)
/// ```
pub fn create_instantiate_function(
    annotator: &RPythonAnnotator,
    classdef: &Rc<RefCell<ClassDef>>,
) -> Result<(), AnnotatorError> {
    // upstream: `if hasattr(classdef, 'my_instantiate_graph'): return`.
    if classdef.borrow().my_instantiate_graph.is_some() {
        return Ok(());
    }

    // upstream: `v = Variable()` + `SpaceOperation('instantiate1', [], v)`.
    let mut v = Variable::new();
    let v_result = Hlvalue::Variable(v.clone());
    let op = crate::flowspace::model::SpaceOperation::new("instantiate1", vec![], v_result.clone());

    // upstream: `block = Block([])` and append op.
    let block = Block::shared(vec![]);
    block.borrow_mut().operations.push(op);

    // upstream: `name = valid_identifier('instantiate_' + classdef.name)`.
    let classdef_name = classdef.borrow().name.clone();
    let name = crate::tool::sourcetools::valid_identifier(format!("instantiate_{classdef_name}"));

    // upstream: `graph = FunctionGraph(name, block)` — fresh
    // returnblock/exceptblock allocated inside the ctor.
    let graph = FunctionGraph::new(name, block.clone());
    let returnblock = graph.returnblock.clone();
    let graph_rc = Rc::new(RefCell::new(graph));

    // upstream: `block.closeblock(Link([v], graph.returnblock))`.
    let link = Link::new(vec![v_result.clone()], Some(returnblock.clone()), None);
    block
        .borrow_mut()
        .closeblock(vec![Rc::new(RefCell::new(link))]);

    // upstream: `annotator.setbinding(v, annmodel.SomeInstance(classdef))`.
    annotator.setbinding(
        &mut v,
        crate::annotator::model::SomeValue::Instance(crate::annotator::model::SomeInstance::new(
            Some(classdef.clone()),
            false,
            Default::default(),
        )),
    );

    // upstream: `annotator.annotated[block] = graph`. Pyre also tracks
    // the reverse BlockKey → BlockRef index in `all_blocks` so
    // `perform_normalizations` / `specialize_more_blocks` can resurface
    // the block from the key alone.
    let block_key = BlockKey::of(&block);
    annotator
        .annotated
        .borrow_mut()
        .insert(block_key.clone(), Some(graph_rc.clone()));
    annotator
        .all_blocks
        .borrow_mut()
        .insert(block_key, block.clone());

    // upstream: `generalizedresult = annmodel.SomeInstance(classdef=None)`
    // + `annotator.setbinding(graph.getreturnvar(), generalizedresult)`.
    let generalized = crate::annotator::model::SomeValue::Instance(
        crate::annotator::model::SomeInstance::new(None, false, Default::default()),
    );
    if let Hlvalue::Variable(mut retvar) = graph_rc.borrow().getreturnvar() {
        annotator.setbinding(&mut retvar, generalized);
    }

    // upstream: `classdef.my_instantiate_graph = graph`.
    classdef.borrow_mut().my_instantiate_graph = Some(graph_rc.clone());

    // upstream: `annotator.translator.graphs.append(graph)`.
    annotator.translator.graphs.borrow_mut().push(graph_rc);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::annotator::classdesc::ClassDesc;
    use crate::annotator::description::DescKey;
    use crate::annotator::model::{SomeInteger, SomeValue};
    use crate::flowspace::model::{FunctionGraph, GraphFunc, HostObject};

    fn empty_shape() -> CallShape {
        CallShape {
            shape_cnt: 0,
            shape_keys: Vec::new(),
            shape_star: false,
        }
    }

    fn make_pygraph(name: &str, argnames: &[&str], defaults: Option<Vec<Constant>>) -> Rc<PyGraph> {
        let inputargs: Vec<_> = argnames
            .iter()
            .map(|name| Hlvalue::Variable(Variable::named(*name)))
            .collect();
        let startblock = Block::shared(inputargs);
        let graph = FunctionGraph::new(name, startblock.clone());
        let ret_arg = graph.startblock.borrow().inputargs[0].clone();
        let link = Rc::new(std::cell::RefCell::new(Link::new(
            vec![ret_arg],
            Some(graph.returnblock.clone()),
            None,
        )));
        startblock.closeblock(vec![link]);
        Rc::new(PyGraph {
            graph: Rc::new(std::cell::RefCell::new(graph)),
            func: GraphFunc::new(name, Constant::new(ConstValue::Dict(Default::default()))),
            signature: std::cell::RefCell::new(Signature::new(
                argnames.iter().map(|s| (*s).to_string()).collect(),
                None,
                None,
            )),
            defaults: std::cell::RefCell::new(defaults),
            access_directly: std::cell::Cell::new(false),
        })
    }

    fn bind_graph_inputs(
        ann: &RPythonAnnotator,
        graph: &Rc<PyGraph>,
        inputs: &[SomeValue],
        result: SomeValue,
    ) {
        {
            let startblock = graph.graph.borrow().startblock.clone();
            let mut blk = startblock.borrow_mut();
            for (arg, value) in blk.inputargs.iter_mut().zip(inputs.iter()) {
                let Hlvalue::Variable(v) = arg else {
                    panic!("expected variable input");
                };
                ann.setbinding(v, value.clone());
            }
        }
        {
            let returnblock = graph.graph.borrow().returnblock.clone();
            let mut blk = returnblock.borrow_mut();
            let Hlvalue::Variable(v) = &mut blk.inputargs[0] else {
                panic!("expected variable return");
            };
            ann.setbinding(v, result);
        }
        let startblock = graph.graph.borrow().startblock.clone();
        let graphref = graph.graph.clone();
        let bkey = BlockKey::of(&startblock);
        ann.annotated
            .borrow_mut()
            .insert(bkey.clone(), Some(graphref));
        ann.all_blocks.borrow_mut().insert(bkey, startblock);
    }

    /// `raise_call_table_too_complex_error` is the only pure function in
    /// this module that does not require an `RPythonAnnotator`. Build a
    /// minimal `CallFamily` with two shapes and verify the message
    /// payload carries the upstream-style multi-line structure.
    #[test]
    fn raise_error_with_two_shapes_lists_inconsistent_arity() {
        let fake_desc = DescKey(1);
        let mut fam = CallFamily::new(fake_desc);
        // Inject two empty tables under distinct shapes so the
        // enumerate-cross loop runs once and produces at least the
        // inconsistent-arity message.
        fam.calltables
            .insert(empty_shape(), vec![CallTableRow::new()]);
        let mut shape2 = empty_shape();
        shape2.shape_cnt = 1;
        fam.calltables.insert(shape2, vec![CallTableRow::new()]);
        // Build a throwaway RPythonAnnotator via the public `new()`
        // helper to satisfy the signature; the function does not
        // dereference it (the callgraph lookup is deferred).
        let ann = RPythonAnnotator::new(None, None, None, false);
        let err = raise_call_table_too_complex_error(&fam, &ann);
        let msg = err.to_string();
        assert!(
            msg.contains("are called with inconsistent numbers of arguments"),
            "missing upstream phrase; got: {msg}"
        );
        assert!(
            msg.contains("sometimes with 0 arguments, sometimes with 1")
                || msg.contains("sometimes with 1 arguments, sometimes with 0"),
            "missing shape-cnt diff hint; got: {msg}"
        );
    }

    #[test]
    fn raise_error_lists_callers_from_translator_callgraph() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let callee = make_pygraph("callee", &["x"], None);
        let caller = Rc::new(std::cell::RefCell::new(FunctionGraph::new(
            "caller",
            Block::shared(vec![]),
        )));
        let caller_block = caller.borrow().startblock.clone();
        ann.translator
            .update_call_graph(&caller, &callee.graph, (BlockKey::of(&caller_block), 0));

        let fake_desc = DescKey(1);
        let mut fam = CallFamily::new(fake_desc);
        let mut row1 = CallTableRow::new();
        row1.insert(DescKey(2), callee.clone());
        fam.calltables.insert(empty_shape(), vec![row1]);
        let mut row2 = CallTableRow::new();
        row2.insert(DescKey(3), callee);
        let mut shape2 = empty_shape();
        shape2.shape_cnt = 1;
        fam.calltables.insert(shape2, vec![row2]);

        let err = raise_call_table_too_complex_error(&fam, &ann);
        let msg = err.to_string();
        assert!(msg.contains("the callers of these functions are:"));
        assert!(msg.contains("    caller"));
    }

    #[test]
    fn perform_normalizations_runs_call_family_pass_and_restores_frozen() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let desc = DescKey(123);
        let family = {
            let mut families = ann.bookkeeper.pbc_maximal_call_families.borrow_mut();
            families.find_rep(desc);
            families
                .get_mut(&desc)
                .expect("family should be materialized")
                .clone()
        };
        {
            let mut fam = family.borrow_mut();
            fam.modified = true;
            fam.normalized = false;
        }

        perform_normalizations(&ann).expect("normalization should succeed");

        assert!(!*ann.frozen.borrow(), "frozen flag must be restored");
        let fam = family.borrow();
        assert!(fam.normalized);
        assert!(!fam.modified);
    }

    #[test]
    fn row_signature_rewrites_startblock_and_signature() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph = make_pygraph(
            "f1",
            &["a", "b", "c"],
            Some(vec![
                Constant::new(ConstValue::Int(10)),
                Constant::new(ConstValue::Int(20)),
            ]),
        );
        let graph2 = make_pygraph(
            "f2",
            &["a", "c"],
            Some(vec![Constant::new(ConstValue::Int(20))]),
        );
        bind_graph_inputs(
            &ann,
            &graph,
            &[
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ],
            SomeValue::Integer(SomeInteger::default()),
        );
        bind_graph_inputs(
            &ann,
            &graph2,
            &[
                SomeValue::Integer(SomeInteger::default()),
                SomeValue::Integer(SomeInteger::default()),
            ],
            SomeValue::Integer(SomeInteger::default()),
        );
        let mut row = CallTableRow::new();
        row.insert(DescKey(1), graph.clone());
        row.insert(DescKey(2), graph2);
        let shape = CallShape {
            shape_cnt: 1,
            shape_keys: vec!["c".to_string()],
            shape_star: false,
        };

        assert!(normalize_calltable_row_signature(&ann, &shape, &row).unwrap());
        assert_eq!(
            graph.signature.borrow().argnames,
            vec!["a".to_string(), "c".to_string()]
        );
        assert_eq!(graph.graph.borrow().startblock.borrow().inputargs.len(), 2);
        let graph_ref = graph.graph.borrow();
        let startblock = graph_ref.startblock.clone();
        let exits_len = startblock.borrow().exits.len();
        assert_eq!(exits_len, 1);
        assert_eq!(startblock.borrow().exits[0].borrow().args.len(), 3);
    }

    #[test]
    fn row_annotation_rewrites_inputs_and_widens_return() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let graph_int = make_pygraph("g1", &["x"], None);
        let graph_bool = make_pygraph("g2", &["x"], None);
        bind_graph_inputs(
            &ann,
            &graph_int,
            &[SomeValue::Integer(SomeInteger::default())],
            SomeValue::Integer(SomeInteger::default()),
        );
        bind_graph_inputs(
            &ann,
            &graph_bool,
            &[crate::annotator::model::s_bool()],
            crate::annotator::model::s_bool(),
        );

        assert!(
            normalize_calltable_row_annotation(&ann, &[graph_int.clone(), graph_bool.clone()])
                .unwrap()
        );

        let startblock = graph_int.graph.borrow().startblock.clone();
        let arg = startblock.borrow().inputargs[0].clone();
        assert!(matches!(ann.binding(&arg), SomeValue::Integer(_)));
        let ret = graph_int.graph.borrow().getreturnvar();
        assert!(matches!(ann.binding(&ret), SomeValue::Integer(_)));
    }

    fn register_classdef(
        ann: &RPythonAnnotator,
        name: &str,
        base: Option<&Rc<RefCell<ClassDef>>>,
    ) -> Rc<RefCell<ClassDef>> {
        let base_desc = base.map(|b| b.borrow().classdesc.clone());
        let base_host_list = base_desc
            .as_ref()
            .map(|cd| vec![cd.borrow().pyobj.clone()])
            .unwrap_or_default();
        let pyobj = HostObject::new_class(name, base_host_list);
        let desc = Rc::new(RefCell::new(ClassDesc::new_shell(
            &ann.bookkeeper,
            pyobj,
            name.to_string(),
        )));
        desc.borrow_mut().basedesc = base_desc;
        let cd = ClassDef::new(&ann.bookkeeper, &desc);
        ann.bookkeeper.classdefs.borrow_mut().push(cd.clone());
        cd
    }

    #[test]
    fn assign_inheritance_ids_single_root_brackets_children_in_witness_order() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let root = register_classdef(&ann, "pkg.Root", None);
        let left = register_classdef(&ann, "pkg.Left", Some(&root));
        let right = register_classdef(&ann, "pkg.Right", Some(&root));

        assign_inheritance_ids(&ann);

        let root_min = root.borrow().minid.unwrap();
        let root_max = root.borrow().maxid.unwrap();
        let left_min = left.borrow().minid.unwrap();
        let left_max = left.borrow().maxid.unwrap();
        let right_min = right.borrow().minid.unwrap();
        let right_max = right.borrow().maxid.unwrap();

        // subclass-range invariant: parent brackets every descendant.
        assert!(root_min <= left_min && left_max <= root_max);
        assert!(root_min <= right_min && right_max <= root_max);
        // children do not overlap.
        assert!(left_max < right_min || right_max < left_min);
        // Reversed-MRO witness order follows stable unique classdef ids;
        // `register_classdef()` created Left before Right.
        assert!(left_min < right_min);
    }

    #[test]
    fn assign_inheritance_ids_repeat_run_is_idempotent_without_new_classes() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let root = register_classdef(&ann, "pkg.Root", None);

        assign_inheritance_ids(&ann);
        let first_min = root.borrow().minid;
        let first_max = root.borrow().maxid;

        assign_inheritance_ids(&ann);

        assert_eq!(root.borrow().minid, first_min);
        assert_eq!(root.borrow().maxid, first_max);
    }

    #[test]
    fn assign_inheritance_ids_deep_chain_produces_nested_range_encoding() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let a = register_classdef(&ann, "pkg.A", None);
        let b = register_classdef(&ann, "pkg.B", Some(&a));
        let c = register_classdef(&ann, "pkg.C", Some(&b));

        assign_inheritance_ids(&ann);

        let a_min = a.borrow().minid.unwrap();
        let a_max = a.borrow().maxid.unwrap();
        let b_min = b.borrow().minid.unwrap();
        let b_max = b.borrow().maxid.unwrap();
        let c_min = c.borrow().minid.unwrap();
        let c_max = c.borrow().maxid.unwrap();

        assert!(a_min < b_min && b_max < a_max);
        assert!(b_min < c_min && c_max < b_max);
        // minid of C is inside B's range (ll_issubclass_const invariant).
        assert!(b_min <= c_min && c_min <= b_max);
        assert!(a_min <= c_min && c_min <= a_max);
    }

    #[test]
    fn assign_inheritance_ids_no_op_when_bookkeeper_has_no_classdefs() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        assign_inheritance_ids(&ann);
        // no-op; assert we didn't panic and classdefs stays empty.
        assert!(ann.bookkeeper.classdefs.borrow().is_empty());
    }

    #[test]
    fn assign_inheritance_ids_later_root_sorts_after_existing_root_range() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let existing = register_classdef(&ann, "pkg.Existing", None);
        assign_inheritance_ids(&ann);
        let existing_max = existing.borrow().maxid.unwrap();

        let fresh = register_classdef(&ann, "pkg.Fresh", None);

        assign_inheritance_ids(&ann);

        let fresh_min = fresh.borrow().minid.unwrap();
        let fresh_max = fresh.borrow().maxid.unwrap();
        assert!(
            fresh_min > existing_max,
            "fresh minid {fresh_min} must sort after existing maxid {existing_max}"
        );
        assert!(fresh_min < fresh_max);
    }

    #[test]
    fn assign_inheritance_ids_later_subclass_stays_inside_existing_parent_range() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let root = register_classdef(&ann, "pkg.Root", None);

        assign_inheritance_ids(&ann);

        let child = register_classdef(&ann, "pkg.Child", Some(&root));
        assign_inheritance_ids(&ann);

        let root_min = root.borrow().minid.unwrap();
        let root_max = root.borrow().maxid.unwrap();
        let child_min = child.borrow().minid.unwrap();
        let child_max = child.borrow().maxid.unwrap();

        assert!(root_min <= child_min && child_max <= root_max);
    }

    #[test]
    fn create_instantiate_function_builds_my_instantiate_graph_and_registers_with_translator() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let classdef = register_classdef(&ann, "pkg.Leaf", None);
        let before = ann.translator.graphs.borrow().len();

        create_instantiate_function(&ann, &classdef).expect("create_instantiate_function");

        // upstream: `classdef.my_instantiate_graph = graph`.
        let graph_rc = classdef
            .borrow()
            .my_instantiate_graph
            .clone()
            .expect("my_instantiate_graph must be set");
        assert!(graph_rc.borrow().name.starts_with("instantiate_"));

        // upstream: `annotator.translator.graphs.append(graph)`.
        assert_eq!(ann.translator.graphs.borrow().len(), before + 1);
        assert!(Rc::ptr_eq(
            ann.translator.graphs.borrow().last().unwrap(),
            &graph_rc,
        ));

        // upstream: `block.operations.append(SpaceOperation('instantiate1', [], v))`.
        let startblock = graph_rc.borrow().startblock.clone();
        let block_ref = startblock.borrow();
        assert_eq!(block_ref.operations.len(), 1);
        assert_eq!(block_ref.operations[0].opname, "instantiate1");
        assert_eq!(block_ref.exits.len(), 1);

        // upstream: annotator.annotated[block] = graph.
        let block_key = BlockKey::of(&startblock);
        let annotated = ann.annotated.borrow();
        let graph_in_annotated = annotated.get(&block_key).cloned().flatten();
        assert!(graph_in_annotated.is_some());
        assert!(Rc::ptr_eq(&graph_in_annotated.unwrap(), &graph_rc));
    }

    #[test]
    fn create_instantiate_function_is_idempotent_on_second_call() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let classdef = register_classdef(&ann, "pkg.Once", None);
        create_instantiate_function(&ann, &classdef).expect("first call");
        let first = classdef.borrow().my_instantiate_graph.clone().unwrap();
        let count_after_first = ann.translator.graphs.borrow().len();

        create_instantiate_function(&ann, &classdef).expect("second call");
        let second = classdef.borrow().my_instantiate_graph.clone().unwrap();

        assert!(Rc::ptr_eq(&first, &second));
        assert_eq!(ann.translator.graphs.borrow().len(), count_after_first);
    }

    #[test]
    fn create_instantiate_functions_drains_needs_generic_instantiate() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let a = register_classdef(&ann, "pkg.A", None);
        let b = register_classdef(&ann, "pkg.B", None);
        ann.bookkeeper.push_needs_generic_instantiate(&a);
        ann.bookkeeper.push_needs_generic_instantiate(&b);

        create_instantiate_functions(&ann).expect("create_instantiate_functions");

        assert!(a.borrow().my_instantiate_graph.is_some());
        assert!(b.borrow().my_instantiate_graph.is_some());
    }

    #[test]
    fn create_instantiate_functions_noop_when_bookkeeper_set_empty() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        register_classdef(&ann, "pkg.Unused", None);
        let before = ann.translator.graphs.borrow().len();
        create_instantiate_functions(&ann).expect("empty drain");
        assert_eq!(ann.translator.graphs.borrow().len(), before);
    }

    #[test]
    fn perform_normalizations_runs_create_instantiate_functions_after_frozen_drops() {
        let ann = RPythonAnnotator::new(None, None, None, false);
        let classdef = register_classdef(&ann, "pkg.Chain", None);
        ann.bookkeeper.push_needs_generic_instantiate(&classdef);

        perform_normalizations(&ann).expect("perform_normalizations");

        // upstream executes the create_instantiate_functions pass AFTER
        // the frozen decrement, so by the time it writes my_instantiate_graph
        // the annotator is already unfrozen again.
        assert!(!*ann.frozen.borrow());
        assert!(classdef.borrow().my_instantiate_graph.is_some());
    }

    /// `merge_classpbc_getattr_into_classdef` walks every per-attr
    /// access set, computes the deepest common base across the
    /// classdescs that share it, and stamps that ClassDef both onto
    /// `family.commonbase` and into `commonbase.extra_access_sets`.
    ///
    /// Two-class scenario: `Root <- Leaf` sharing attr "shared". The
    /// common base must be `Root`, and `Root.extra_access_sets` must
    /// carry one entry referencing the family.
    #[test]
    fn merge_classpbc_getattr_into_classdef_attaches_commonbase_and_extra_access_set() {
        use crate::annotator::description::DescEntry as DE;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let root = register_classdef(&ann, "pkg.Root", None);
        let leaf = register_classdef(&ann, "pkg.Leaf", Some(&root));

        // Surface the classdescs in `bookkeeper.descs` so the merge
        // pass can resolve `family.descs` keys back to live ClassDescs.
        // Upstream populates this lazily via `getdesc`; the test bypass
        // mirrors that side-effect.
        let root_desc = root.borrow().classdesc.clone();
        let leaf_desc = leaf.borrow().classdesc.clone();
        {
            let mut descs = ann.bookkeeper.descs.borrow_mut();
            let root_pyobj = root_desc.borrow().pyobj.clone();
            let leaf_pyobj = leaf_desc.borrow().pyobj.clone();
            descs.insert(root_pyobj, DE::Class(root_desc.clone()));
            descs.insert(leaf_pyobj, DE::Class(leaf_desc.clone()));
        }

        // Build a single ClassAttrFamily containing both descs by
        // unioning them under the "shared" attr name.
        let root_key = DescKey::from_rc(&root_desc);
        let leaf_key = DescKey::from_rc(&leaf_desc);
        ann.bookkeeper.with_classpbc_attr_families("shared", |uf| {
            uf.find(root_key);
            uf.union(root_key, leaf_key);
            Some(())
        });

        merge_classpbc_getattr_into_classdef(&ann)
            .expect("merge_classpbc_getattr_into_classdef must succeed");

        // Find the family — there is exactly one root info in the UF.
        let access_set = ann
            .bookkeeper
            .with_classpbc_attr_families("shared", |uf| uf.infos().next().cloned())
            .expect("union must produce exactly one family");

        // Side 1: family.commonbase is the root classdef.
        let commonbase = access_set
            .borrow()
            .commonbase
            .clone()
            .expect("commonbase must be set after merge_classpbc_getattr_into_classdef");
        assert!(
            Rc::ptr_eq(&commonbase, &root),
            "commonbase must equal the deepest common ancestor (Root)"
        );

        // Side 2: commonbase.extra_access_sets carries the family.
        let key = Rc::as_ptr(&access_set) as usize;
        let entry = root
            .borrow()
            .extra_access_sets
            .get(&key)
            .cloned()
            .expect("extra_access_sets must carry the access_set keyed by Rc identity");
        assert_eq!(entry.1, "shared", "stored attrname must match");
        assert_eq!(entry.2, 0, "first registered family gets counter 0");
        assert!(
            Rc::ptr_eq(&entry.0, &access_set),
            "stored Rc must equal the original family"
        );
    }

    /// Single-desc families are skipped (no commonbase needed) — they
    /// take the upstream `if len(descs) <= 1: continue` short-circuit.
    #[test]
    fn merge_classpbc_getattr_into_classdef_skips_singleton_family() {
        use crate::annotator::description::DescEntry as DE;
        let ann = RPythonAnnotator::new(None, None, None, false);
        let only = register_classdef(&ann, "pkg.Only", None);
        let only_desc = only.borrow().classdesc.clone();
        {
            let mut descs = ann.bookkeeper.descs.borrow_mut();
            let pyobj = only_desc.borrow().pyobj.clone();
            descs.insert(pyobj, DE::Class(only_desc.clone()));
        }
        let only_key = DescKey::from_rc(&only_desc);
        ann.bookkeeper.with_classpbc_attr_families("alone", |uf| {
            uf.find(only_key);
            Some(())
        });

        merge_classpbc_getattr_into_classdef(&ann).expect("must succeed");

        // Singleton family must NOT have commonbase populated.
        let access_set = ann
            .bookkeeper
            .with_classpbc_attr_families("alone", |uf| uf.infos().next().cloned())
            .expect("singleton family present");
        assert!(
            access_set.borrow().commonbase.is_none(),
            "singleton family must not get commonbase set"
        );
        assert!(
            only.borrow().extra_access_sets.is_empty(),
            "singleton class must not be touched"
        );
    }
}
