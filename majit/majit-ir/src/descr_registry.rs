//! Process-global descriptor registry — thin facade over [`crate::descr::gc_cache`].
//!
//! PyPy `pyjitpl.py:2287-2290 finish_setup_descrs`:
//!
//! ```python
//! def finish_setup_descrs(self):
//!     from rpython.jit.codewriter import effectinfo
//!     self.all_descrs = self.cpu.setup_descrs()
//!     effectinfo.compute_bitstrings(self.all_descrs)
//! ```
//!
//! `cpu.setup_descrs()` walks `gc_cache._cache_*` (`backend/llsupport/descr.py:25-47`)
//! across the six categories — size / field / array / arraylen / call /
//! interiorfield — and returns the concatenated descr list in fixed
//! group order.  Pyre's lift routes every mint site through the
//! process-global [`crate::descr::gc_cache`] (lift of PyPy's per-CPU
//! `gc_ll_descr.gc_cache`); this module is a backwards-compatible
//! `register_*` / `snapshot_*` shim so callers that pre-date the
//! `gc_cache.get_*_descr(LLType, ...)` cache-or-mint API can still
//! publish their freshly-minted descrs.
//!
//! PRE-EXISTING-ADAPTATION: this module is the temporary surface for
//! mint sites that bypass the keyed cache.  Each migrated site drops
//! its `register_*` call in favour of `gc_cache.get_*_descr(LLType, ...)`;
//! the end state retires this module entirely once every production
//! mint flows through the keyed path.

use crate::descr::{DescrRef, gc_cache};

/// `descr.py:236-247 get_size_descr` cache-miss publication — register
/// a freshly-minted size descr.  PRE-EXISTING-ADAPTATION: prefer
/// `gc_cache.get_size_descr(LLType::Struct(...), ...)`.
pub fn register_size(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_size(descr);
}

/// `descr.py:225-235 get_field_descr` cache-miss publication.
/// PRE-EXISTING-ADAPTATION: prefer
/// `gc_cache.get_field_descr(LLType::Struct(...), ...)`.
pub fn register_field(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_field(descr);
}

/// `descr.py:354-364 get_array_descr` cache-miss publication.
/// PRE-EXISTING-ADAPTATION: prefer
/// `gc_cache.get_array_descr(LLType::Array(...), ...)`.
pub fn register_array(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_array(descr);
}

/// `descr.py:374-385 get_arraylen_descr` cache-miss publication.
/// PRE-EXISTING-ADAPTATION: prefer
/// `gc_cache.get_field_arraylen_descr(LLType::Array(...), ...)`.
pub fn register_array_len(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_arraylen(descr);
}

/// `descr.py:404-414 get_interiorfield_descr` cache-miss publication.
/// PRE-EXISTING-ADAPTATION: prefer the keyed cache-or-mint path once
/// `gc_cache.get_interiorfield_descr` lands.
pub fn register_interior_field(descr: DescrRef) {
    gc_cache()
        .lock()
        .unwrap()
        .register_external_interiorfield(descr);
}

/// `descr.py:647-675 get_call_descr` cache-miss publication.  Sole
/// caller is `call_descr::make_call_descr_with_effect`'s mint path
/// (the production call-descr factory in majit-metainterp); the
/// dedicated `CALL_DESCR_CACHE` keys on the structural call signature
/// to enforce identity per call shape, and this hook publishes the
/// minted descr into `gc_cache._cache_call_order` so
/// `MetaInterpStaticData::finish_setup_descrs` enumerates the call
/// category alongside the field / array / interiorfield categories.
pub fn register_call(descr: DescrRef) {
    gc_cache().lock().unwrap().register_external_call(descr);
}

/// `descr.py:25-47 setup_descrs` snapshot.  Sole production caller is
/// `MetaInterpStaticData::finish_setup_descrs`, which interleaves the
/// pyre-side `cached_call_descrs()` (which still lives outside
/// `gc_cache._cache_call_order`) between arrays and interior_fields
/// per PyPy group order.
pub fn snapshot_all() -> Vec<DescrRef> {
    let gc = gc_cache().lock().unwrap();
    let mut out = Vec::with_capacity(
        gc.snapshot_sizes().len()
            + gc.snapshot_fields().len()
            + gc.snapshot_arrays().len()
            + gc.snapshot_arraylens().len()
            + gc.snapshot_interiorfields().len(),
    );
    out.extend(gc.snapshot_sizes());
    out.extend(gc.snapshot_fields());
    out.extend(gc.snapshot_arrays());
    out.extend(gc.snapshot_arraylens());
    out.extend(gc.snapshot_interiorfields());
    out
}

/// `descr.py:28-29 _cache_size` snapshot.
pub fn snapshot_sizes() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_sizes()
}

/// `descr.py:30-33 _cache_field` snapshot.
pub fn snapshot_fields() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_fields()
}

/// `descr.py:34-36 _cache_array` snapshot.
pub fn snapshot_arrays() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_arrays()
}

/// `descr.py:37-39 _cache_arraylen` snapshot.
pub fn snapshot_array_lens() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_arraylens()
}

/// `descr.py:43-45 _cache_interiorfield` snapshot.
pub fn snapshot_interior_fields() -> Vec<DescrRef> {
    gc_cache().lock().unwrap().snapshot_interiorfields()
}

/// Per-category counts for diagnostic asserts.
/// Returns `(sizes, fields, arrays, array_lens, interior_fields)` —
/// the call category lives outside this facade; use
/// `gc_cache().lock().unwrap().category_counts()` for the full tuple.
pub fn category_counts() -> (usize, usize, usize, usize, usize) {
    let (s, f, a, al, _c, ifs) = gc_cache().lock().unwrap().category_counts();
    (s, f, a, al, ifs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descr::SimpleFieldDescr;
    use crate::value::Type;
    use std::sync::Arc;

    fn fresh_field(idx: u32) -> DescrRef {
        Arc::new(SimpleFieldDescr::new(idx, 0, 8, Type::Int, false))
    }

    fn count_arc(haystack: &[DescrRef], needle: &DescrRef) -> usize {
        haystack
            .iter()
            .filter(|descr| Arc::ptr_eq(descr, needle))
            .count()
    }

    /// Same `Arc` clone re-registered via the facade collapses to a
    /// single `_cache_field_order` entry — `Arc::ptr_eq` dedup at the
    /// `gc_cache.register_external_*` level.
    #[test]
    fn dedup_by_arc_identity_within_category() {
        let f = fresh_field(42);
        register_field(f.clone());
        register_field(f.clone());
        let fields = gc_cache().lock().unwrap().snapshot_fields();
        assert_eq!(
            count_arc(&fields, &f),
            1,
            "same Arc should collapse to one entry"
        );
    }

    /// Distinct Arcs that share `descr.index() == 0` (e.g. two
    /// different structs' `index_in_parent = 0` fields at
    /// `call.rs:3849`) must NOT collapse — `Arc::ptr_eq` is by
    /// pointer, not by content.  PyPy `_cache_field` parity:
    /// `(STRUCT, fieldname)` keyed dict.
    #[test]
    fn distinct_arcs_share_index_stay_separate() {
        let f_a = fresh_field(0);
        let f_b = fresh_field(0);
        register_field(f_a.clone());
        register_field(f_b.clone());
        let fields = gc_cache().lock().unwrap().snapshot_fields();
        assert_eq!(count_arc(&fields, &f_a), 1);
        assert_eq!(count_arc(&fields, &f_b), 1);
        assert!(
            fields
                .iter()
                .filter(|descr| Arc::ptr_eq(descr, &f_a) || Arc::ptr_eq(descr, &f_b))
                .count()
                == 2
        );
    }
}
