//! Visitor trait for walking virtual object structures.
//!
//! Provides consistent traversal of all virtual object types during
//! resume data construction, unrolling, and other analyses.
//!
//! Translated from rpython/jit/metainterp/walkvirtual.py.

use majit_ir::{DescrRef, OpRef};

/// walkvirtual.py:4-39: VirtualVisitor
///
/// Abstract base class. Every method raises NotImplementedError in RPython;
/// in Rust this is expressed as required trait methods with no defaults.
/// Implementors must provide all methods — the compiler enforces this
/// at the same strength as RPython's runtime NotImplementedError.
///
/// `VInfo` is the associated return type for `visit_*` dispatch methods.
/// Both RPython subclasses return values:
/// - `ResumeDataVirtualAdder` (resume.py:320-357) returns `VirtualInfo` subclasses
/// - `VirtualStateConstructor` (virtualstate.py:743-760) returns `VirtualStateInfo` subclasses
/// RPython relies on dynamic dispatch; Rust models this with an associated
/// type on the trait (the single minimal adaptation due to static typing).
pub trait VirtualVisitor {
    /// Return type of visit_* dispatch (see trait doc).
    type VInfo;

    /// walkvirtual.py:5
    fn visit_not_virtual(&mut self, value: OpRef) -> Self::VInfo;

    /// walkvirtual.py:8; info.py:331-334.
    ///
    /// `fielddescr_indices` is a remaining Rust-side compatibility shim for
    /// call sites that still thread slot numbers explicitly. The canonical
    /// shape now matches RPython again: `fielddescrs` contains the full
    /// descriptor-order slot list and the paired `fieldnums` list uses
    /// UNINITIALIZED tags for holes.
    fn visit_virtual(
        &mut self,
        descr: &DescrRef,
        fielddescr_indices: &[u32],
        fielddescrs: &[DescrRef],
    ) -> Self::VInfo;

    /// walkvirtual.py:11; info.py:368-372. See `visit_virtual` for the
    /// compatibility note on `fielddescr_indices`.
    fn visit_vstruct(
        &mut self,
        typedescr: &DescrRef,
        fielddescr_indices: &[u32],
        fielddescrs: &[DescrRef],
    ) -> Self::VInfo;

    /// walkvirtual.py:14; info.py:597-599
    fn visit_varray(&mut self, arraydescr: &DescrRef, clear: bool) -> Self::VInfo;

    /// walkvirtual.py:17; info.py:700-704. See `visit_virtual` for the
    /// compatibility note on `fielddescr_indices`.
    fn visit_varraystruct(
        &mut self,
        arraydescr: &DescrRef,
        length: usize,
        fielddescr_indices: &[u32],
        fielddescrs: &[DescrRef],
    ) -> Self::VInfo;

    /// walkvirtual.py:20; info.py:444-450
    fn visit_vrawbuffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: &[i64],
        descrs: &[DescrRef],
    ) -> Self::VInfo;

    /// walkvirtual.py:23; info.py:484-486
    fn visit_vrawslice(&mut self, offset: i64) -> Self::VInfo;

    /// walkvirtual.py:26; vstring.py:210-212
    fn visit_vstrplain(&mut self, is_unicode: bool) -> Self::VInfo;

    /// walkvirtual.py:29; vstring.py:332-334
    fn visit_vstrconcat(&mut self, is_unicode: bool) -> Self::VInfo;

    /// walkvirtual.py:32; vstring.py:262-264
    fn visit_vstrslice(&mut self, is_unicode: bool) -> Self::VInfo;

    /// walkvirtual.py:35; info.py:298-302 _visitor_walk_recursive
    fn register_virtual_fields(&mut self, virtualbox: OpRef, fieldboxes: &[OpRef]);

    /// walkvirtual.py:38
    fn already_seen_virtual(&mut self, virtualbox: OpRef) -> bool;
}
