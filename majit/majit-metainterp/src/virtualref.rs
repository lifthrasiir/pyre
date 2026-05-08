//! Virtual references: lazy materialization of virtualized objects.
//!
//! When JIT-compiled code virtualizes an object (e.g., an interpreter frame),
//! external code may still hold a reference to it. A virtual reference is a
//! lightweight wrapper that defers materialization until someone actually
//! accesses (forces) it.
//!
//! The `JitVirtualRef` struct has two fields:
//! - `virtual_token`: while the JIT is running, this is the force_token
//!   (i.e., the JIT frame address). Set to TOKEN_NONE when forced or
//!   when the frame is no longer active.
//! - `forced`: once forced, this points to the materialized object.
//!
//! Mirrors `rpython/jit/metainterp/virtualref.py`.

use std::sync::atomic::{AtomicU32, Ordering};

/// GC type id for JitVirtualRef, set by `set_vref_gc_type_id()` at startup.
/// RPython registers JIT_VIRTUAL_REF as a real GC type; pyre does the same
/// via `gc.register_type(TypeInfo::with_gc_ptrs(...))` in eval.rs.
static VREF_GC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);

/// Set the GC type id for JitVirtualRef. Called once at startup after
/// `gc.register_type()` returns the assigned id.
pub fn set_vref_gc_type_id(type_id: u32) {
    VREF_GC_TYPE_ID.store(type_id, Ordering::Relaxed);
}

/// Get the registered GC type id for JitVirtualRef.
pub fn vref_gc_type_id() -> u32 {
    VREF_GC_TYPE_ID.load(Ordering::Relaxed)
}

/// `rpython/rlib/jit.py JitVirtualRef`: heap-allocated virtual reference.
/// Contains a type tag for identity checking (typeptr), a force token
/// (active JIT frame), and a forced pointer (materialized object,
/// initially null).
///
/// `virtualref.py:17-20` registers `virtual_token` and `forced` both as
/// `llmemory.GCREF`/`OBJECTPTR` slots — pointer fields, not int.  Pyre
/// stores `virtual_token` as `*mut u8` so the runtime layout, the
/// `VREF_FIELD_VIRTUAL_TOKEN` descriptor encoding (Ref bit), and the
/// optimizer-side `make_vref_field_descr` (`Type::Ref` per
/// `optimizeopt/virtualize.rs:1659`) all agree.
///
/// PRE-EXISTING-ADAPTATION (GC trace).  Upstream traces both fields as
/// real GC pointers; pyre traces only `forced`.  `eval.rs:241-247`
/// registers JIT_VIRTUAL_REF with `gc_ptr_offsets = [16]` (forced
/// only).  The reason is that every value `virtual_token` ever holds
/// at runtime falls outside the GC heap:
///   - `TOKEN_NONE` — null, safe to walk.
///   - `token_tracing_rescall()` — static `u64` storage address (see
///     below), not a GC-allocated `_dummy` GcStruct.
///   - an active JITFRAME address — `libc::calloc`'d on a host-side
///     pool (eval.rs:232-240), not nursery/oldgen.
/// Routing it through `trace_and_update_object` would either be a
/// no-op or trip a poison-address check.  The optimizer-side
/// `Type::Ref` is intentionally retained so that
/// `setfield_gc_r` / `getfield_gc_r` ops emit correctly during
/// tracing; only the collector's view of the slot diverges.
/// Convergence path: would require allocating `_dummy` via the GC
/// AND routing JITFRAMEs through GC-managed allocation, both outside
/// the current parity scope.
#[repr(C)]
pub struct JitVirtualRef {
    /// Type identity tag — `inst.typeptr == jit_virtual_ref_vtable` peer.
    pub type_tag: u64,
    pub virtual_token: *mut u8,
    pub forced: *mut u8,
}

/// Magic value stored in JitVirtualRef.type_tag for type identity.
/// `virtualref.py:94-98 is_virtual_ref` checks
/// `inst.typeptr == jit_virtual_ref_vtable`.
pub const VREF_TYPE_TAG: u64 = 0x4A49_5456_5245_4621; // "JITVREF!"

/// Allocate a concrete JitVirtualRef on the heap.
/// `virtualref.py:85-91 virtual_ref_during_tracing(real_object)`.
/// Initializes virtual_token = TOKEN_NONE, forced = real_object.
/// Returns raw pointer; caller owns the allocation.
pub fn alloc_virtual_ref(real_object: *mut u8) -> *mut u8 {
    let vref = Box::new(JitVirtualRef {
        type_tag: VREF_TYPE_TAG,
        virtual_token: TOKEN_NONE,
        forced: real_object,
    });
    Box::into_raw(vref) as *mut u8
}

/// Token value indicating no JIT frame is active.
/// `virtualizable.py:329 TOKEN_NONE = lltype.nullptr(llmemory.GCREF.TO)`.
pub const TOKEN_NONE: *mut u8 = std::ptr::null_mut();

/// Static dummy backing `TOKEN_TRACING_RESCALL`.
///
/// `virtualizable.py:326-327`:
/// ```python
/// _DUMMY = lltype.GcStruct('JITFRAME_DUMMY')
/// _dummy = lltype.malloc(_DUMMY)
/// ```
/// allocates a real GC object whose address serves as the tracing
/// sentinel.  A `static u64` gives us a stable lifetime-of-program
/// address that is non-null, distinguishable from any heap allocation,
/// and (unlike `usize::MAX as *mut u8`) actually points at memory we
/// own — so any non-GC machinery that loads the slot and chases it
/// sees a valid pointer rather than tripping on a poison address.
///
/// PRE-EXISTING-ADAPTATION.  This is NOT 1:1 with upstream's
/// `lltype.malloc(_DUMMY)`: PyPy's `_dummy` is a real GcStruct that
/// the collector knows about, while pyre's static is host-process
/// memory the GC has no record of.  The adaptation is internally
/// consistent because `virtual_token` is also not GC-traced (see
/// `JitVirtualRef` doc-comment); if either side flips — making the
/// slot GC-traced or making `_dummy` a GcStruct — both must flip
/// together.
static TRACING_RESCALL_DUMMY: u64 = 0;

/// Token value used during tracing when a residual call is in progress.
/// `virtualizable.py:330`:
/// ```python
/// TOKEN_TRACING_RESCALL = lltype.cast_opaque_ptr(llmemory.GCREF, _dummy)
/// ```
#[inline]
pub fn token_tracing_rescall() -> *mut u8 {
    &TRACING_RESCALL_DUMMY as *const u64 as *mut u8
}

/// Well-known field descriptor indices for JitVirtualRef fields.
/// Properly encoded: FIELD_DESCR_TAG | (byte_offset << 4) | type_bits
///
/// Layout: type_tag(0) | virtual_token(8) | forced(16)
pub const VREF_FIELD_TYPE_TAG: u32 = 0x1000_0000; // offset=0, Int
/// `virtualref.py:17-20` declares `virtual_token` as `llmemory.GCREF`.
pub const VREF_FIELD_VIRTUAL_TOKEN: u32 = 0x1000_0081; // offset=8, Ref
pub const VREF_FIELD_FORCED: u32 = 0x1000_0101; // offset=16, Ref

/// Descriptor indices for the virtual ref struct fields.
pub mod descr {
    /// Field descriptor index for `type_tag` (RPython typeptr equivalent).
    pub const TYPE_TAG: u32 = super::VREF_FIELD_TYPE_TAG;
    /// Field descriptor index for `virtual_token`.
    pub const VIRTUAL_TOKEN: u32 = super::VREF_FIELD_VIRTUAL_TOKEN;
    /// Field descriptor index for `forced`.
    pub const FORCED: u32 = super::VREF_FIELD_FORCED;
    /// Size descriptor index for the JitVirtualRef struct itself.
    pub const VREF_SIZE: u32 = 0x7F10;
}

/// Virtual reference state for a single reference.
///
/// A virtual reference wraps a virtualizable object during JIT execution.
/// When code outside the JIT tries to access the virtual object, the
/// reference is "forced" -- the virtual object is materialized on the heap.
#[derive(Debug, Clone)]
pub struct VirtualRefInfo {
    /// Field descriptor index for the `virtual_token` field.
    pub descr_virtual_token: u32,
    /// Field descriptor index for the `forced` field.
    pub descr_forced: u32,
    /// Size descriptor index for the JitVirtualRef struct.
    pub descr_size: u32,
}

impl Default for VirtualRefInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::resume::VRefInfo for VirtualRefInfo {
    fn continue_tracing(&self, vref: i64, virtual_ref: i64) {
        // virtualref.py:122-127 continue_tracing(vref, virtual)
        if vref == 0 {
            return;
        }
        let vref_ptr = vref as *mut JitVirtualRef;
        // vref.virtual_token = vr_virtualtoken_none
        // vref.forced = virtual
        unsafe {
            (*vref_ptr).virtual_token = TOKEN_NONE;
            (*vref_ptr).forced = virtual_ref as *mut u8;
        }
    }
}

impl VirtualRefInfo {
    /// Create a VirtualRefInfo with the standard descriptor indices.
    pub fn new() -> Self {
        VirtualRefInfo {
            descr_virtual_token: descr::VIRTUAL_TOKEN,
            descr_forced: descr::FORCED,
            descr_size: descr::VREF_SIZE,
        }
    }

    /// virtualref.py: force_virtual()
    ///
    /// Force a virtual reference: materialize the virtual object.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual(
        &self,
        vref_ptr: *mut u8,
        force_fn: impl FnOnce(*mut u8) -> *mut u8,
    ) -> *mut u8 {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            if vref.virtual_token == TOKEN_NONE {
                return vref.forced;
            }
            if vref.virtual_token == token_tracing_rescall() {
                vref.virtual_token = TOKEN_NONE;
                return vref.forced;
            }
            let materialized = force_fn(vref.virtual_token);
            vref.forced = materialized;
            vref.virtual_token = TOKEN_NONE;
            materialized
        }
    }

    /// Create a virtual reference during tracing.
    ///
    /// virtualref.py:85-91: `virtual_ref_during_tracing(real_object)`
    ///
    /// Allocates a concrete JitVirtualRef on the heap with
    /// virtual_token = TOKEN_NONE, forced = real_object.
    pub fn virtual_ref_during_tracing(&self, real_object: *mut u8) -> *mut u8 {
        alloc_virtual_ref(real_object)
    }

    /// virtualref.py: tracing_before_residual_call(vref)
    ///
    /// Sets token to TOKEN_TRACING_RESCALL so that if the callee
    /// forces the vref, we detect it after the call.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn tracing_before_residual_call(&self, vref_ptr: *mut u8) {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            vref.virtual_token = token_tracing_rescall();
        }
    }

    /// virtualref.py:107-119 tracing_after_residual_call(vref)
    ///
    /// Returns true if the vref was forced during the residual call.
    /// If not forced, sets token to TOKEN_NONE (RPython clears it).
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn tracing_after_residual_call(&self, vref_ptr: *mut u8) -> bool {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            if vref.virtual_token != token_tracing_rescall() {
                // Token was modified during the call — vref was forced
                return true;
            }
            // Not forced: clear to TOKEN_NONE (virtualref.py:118)
            vref.virtual_token = TOKEN_NONE;
            false
        }
    }

    /// virtualref.py: continue_tracing(vref, real_object)
    ///
    /// Updates the forced field and clears the token.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn continue_tracing(&self, vref_ptr: *mut u8, real_object: *mut u8) {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            vref.forced = real_object;
            vref.virtual_token = TOKEN_NONE;
        }
    }

    /// Check if a virtual reference is currently active (has a JIT frame token).
    ///
    /// virtualref.py: token != TOKEN_NONE and token != TOKEN_TRACING_RESCALL
    pub fn is_active(token: *mut u8) -> bool {
        token != TOKEN_NONE && token != token_tracing_rescall()
    }

    /// Check if a virtual reference is forced (token == TOKEN_NONE).
    pub fn is_forced(token: *mut u8) -> bool {
        token == TOKEN_NONE
    }

    /// Check if a virtual reference is in a residual call.
    pub fn is_in_residual_call(token: *mut u8) -> bool {
        token == token_tracing_rescall()
    }

    /// virtualref.py:94-98 is_virtual_ref(gcref)
    ///
    /// RPython checks `inst.typeptr == jit_virtual_ref_vtable`.
    /// pyre checks JitVirtualRef.type_tag == VREF_TYPE_TAG as the
    /// equivalent type identity mechanism.
    ///
    /// # Safety
    /// `ptr` must point to a valid object or be null.
    pub unsafe fn is_virtual_ref(&self, ptr: *const u8) -> bool {
        unsafe {
            if ptr.is_null() {
                return false;
            }
            let tag = *(ptr as *const u64);
            tag == VREF_TYPE_TAG
        }
    }

    /// virtualref.py: force_virtual(inst)
    ///
    /// Force a virtual reference: materialize the virtual object if needed.
    /// This is the full force path that handles all token states.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual_full(
        &self,
        vref_ptr: *mut u8,
        force_fn: impl FnOnce(*mut u8) -> *mut u8,
    ) -> *mut u8 {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            if vref.virtual_token == TOKEN_NONE {
                return vref.forced;
            }
            if vref.virtual_token == token_tracing_rescall() {
                vref.virtual_token = TOKEN_NONE;
                return vref.forced;
            }
            let materialized = force_fn(vref.virtual_token);
            vref.forced = materialized;
            vref.virtual_token = TOKEN_NONE;
            materialized
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_constants() {
        assert!(
            TOKEN_NONE.is_null(),
            "TOKEN_NONE = nullptr(GCREF) per virtualizable.py:329"
        );
        let rescall = token_tracing_rescall();
        assert!(
            !rescall.is_null(),
            "token_tracing_rescall() = cast_opaque_ptr(_dummy) per virtualizable.py:330 — non-null sentinel",
        );
        assert_ne!(
            TOKEN_NONE, rescall,
            "TOKEN_NONE and token_tracing_rescall() must be distinguishable",
        );
        // Stable across calls — `_dummy = lltype.malloc(_DUMMY)` is a single
        // program-lifetime allocation, so repeated reads must yield the
        // same address.
        assert_eq!(rescall, token_tracing_rescall());
        // Address must point at memory we own (the static dummy), not a
        // poison value like `usize::MAX as *mut u8`.  Read-back is the
        // simplest probe that a real `_dummy` allocation backs the token.
        unsafe {
            assert_eq!(*(rescall as *const u64), 0);
        }
    }

    #[test]
    fn test_vref_info_default() {
        let info = VirtualRefInfo::new();
        assert_eq!(info.descr_virtual_token, descr::VIRTUAL_TOKEN);
        assert_eq!(info.descr_forced, descr::FORCED);
        assert_eq!(info.descr_size, descr::VREF_SIZE);
    }
}
