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

/// `rpython/rlib/jit.py:487 class InvalidVirtualRef(Exception)` —
/// `force_virtual` raises this when `virtual_token == TOKEN_NONE`
/// but `forced` is null (`virtualref.py:174-176`).  Pyre's single
/// canonical definition lives in `crate::jit::InvalidVirtualRef`
/// (mirrors `virtualref.py:9 from rpython.rlib.jit import
/// InvalidVirtualRef`); re-exported here for convenience.
pub use crate::jit::InvalidVirtualRef;

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
    /// virtualref.py:122-129 continue_tracing(gcref, real_object)
    ///
    /// Mirrors RPython:
    ///   if not self.is_virtual_ref(gcref): return
    ///   assert real_object
    ///   vref = ...; assert vref.virtual_token != TOKEN_TRACING_RESCALL
    ///   vref.virtual_token = TOKEN_NONE
    ///   vref.forced = real_object
    fn continue_tracing(&self, vref: i64, virtual_ref: i64) {
        // `virtualref.py:122-129 continue_tracing(gcref, real_object)`
        // — delegate to the inherent helper which carries the
        // `is_virtual_ref` guard, `assert real_object`, and
        // `assert vref.virtual_token != TOKEN_TRACING_RESCALL`
        // post-conditions.
        unsafe {
            self.continue_tracing(vref as *mut u8, virtual_ref as *mut u8);
        }
    }
}

// Project convention: RPython's plain `assert ...` ports to
// `debug_assert!(...)`.  Both elide under the production translation
// (RPython `-O`, pyre `--release`); both fire under `assert ...` /
// `cargo test` (RPython untranslated, pyre debug build).  Where
// upstream uses `if not we_are_translated(): assert ...` the port is
// `#[cfg(debug_assertions)] { assert!(...) }` (always-on under
// debug_assertions, elided in release).  See `~/.claude/projects/.../
// memory/feedback_rpython_assert_parity_always_on_2026_05_05.md`.
//
// The `debug_assert!` calls below correspond to upstream `assert`
// statements at:
//   virtualref.py:104  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:111  (assert vref.forced)
//   virtualref.py:115  (assert vref.virtual_token == TOKEN_TRACING_RESCALL)
//   virtualref.py:125  (assert real_object)
//   virtualref.py:127  (assert vref.virtual_token != TOKEN_TRACING_RESCALL)
//   virtualref.py:166  (assert vref.forced)
//   virtualref.py:169  (assert not vref.forced)
//   virtualref.py:172  (assert vref.virtual_token == TOKEN_NONE)
//   virtualref.py:173  (assert vref.forced)
impl VirtualRefInfo {
    /// Create a VirtualRefInfo with the standard descriptor indices.
    pub fn new() -> Self {
        VirtualRefInfo {
            descr_virtual_token: descr::VIRTUAL_TOKEN,
            descr_forced: descr::FORCED,
            descr_size: descr::VREF_SIZE,
        }
    }

    /// `virtualref.py:157-177 force_virtual(inst)` — force a virtual
    /// reference: materialise the virtual object if needed.
    ///
    /// Returns `Err(InvalidVirtualRef)` when `virtual_token ==
    /// TOKEN_NONE` and `forced` is null — the vref was tracked but
    /// never properly initialised (`virtualref.py:174-176`).  In the
    /// `TOKEN_TRACING_RESCALL` and active-frame branches upstream
    /// `assert vref.forced` immediately after force, mirrored here as
    /// `debug_assert!(!vref.forced.is_null())`.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual(
        &self,
        vref_ptr: *mut u8,
        force_fn: impl FnOnce(*mut u8) -> *mut u8,
    ) -> Result<*mut u8, InvalidVirtualRef> {
        unsafe {
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            let token = vref.virtual_token;
            if token != TOKEN_NONE {
                if token == token_tracing_rescall() {
                    // `virtualref.py:161-167`: "virtual" is not a virtual
                    // at all during tracing; reset token as the marker
                    // that this "virtual" escapes.
                    debug_assert!(!vref.forced.is_null());
                    vref.virtual_token = TOKEN_NONE;
                } else {
                    // `virtualref.py:168-173`: active-frame token —
                    // run the materialiser, then assert post-conditions.
                    debug_assert!(vref.forced.is_null());
                    let materialized = force_fn(token);
                    vref.forced = materialized;
                    vref.virtual_token = TOKEN_NONE;
                    debug_assert!(!vref.forced.is_null());
                }
            } else if vref.forced.is_null() {
                // `virtualref.py:174-176`: token == TOKEN_NONE and the
                // vref was not forced — invalid.
                return Err(InvalidVirtualRef);
            }
            Ok(vref.forced)
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

    /// `virtualref.py:100-105 tracing_before_residual_call(gcref)` —
    /// sets token to `TOKEN_TRACING_RESCALL` so that if the callee
    /// forces the vref, we detect it after the call.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn tracing_before_residual_call(&self, vref_ptr: *mut u8) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:104 assert vref.virtual_token == TOKEN_NONE`
            debug_assert_eq!(vref.virtual_token, TOKEN_NONE);
            vref.virtual_token = token_tracing_rescall();
        }
    }

    /// `virtualref.py:107-120 tracing_after_residual_call(gcref)` —
    /// returns `true` if the vref was forced during the residual
    /// call.  If not forced, resets token to `TOKEN_NONE`.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns `False`
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn tracing_after_residual_call(&self, vref_ptr: *mut u8) -> bool {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return false;
            }
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:111 assert vref.forced` — by construction
            // a JitVirtualRef has its `forced` field initialised to a
            // non-null real_object at allocation time.
            debug_assert!(!vref.forced.is_null());
            if vref.virtual_token != TOKEN_NONE {
                // `virtualref.py:113-117` not modified by the residual
                // call; assert it is still TOKEN_TRACING_RESCALL and
                // clear it.
                debug_assert_eq!(vref.virtual_token, token_tracing_rescall());
                vref.virtual_token = TOKEN_NONE;
                false
            } else {
                // `virtualref.py:118-120` token was cleared by the
                // callee — "modified during residual call" marker.
                true
            }
        }
    }

    /// `virtualref.py:122-129 continue_tracing(gcref, real_object)` —
    /// updates the `forced` field and clears the token.  Upstream
    /// guards on `is_virtual_ref(gcref)` first and returns silently
    /// when the gcref is not a JitVirtualRef.
    ///
    /// # Safety
    /// `vref_ptr` must be null or point to a valid GCREF object
    /// whose first 8 bytes are the type-tag word.
    pub unsafe fn continue_tracing(&self, vref_ptr: *mut u8, real_object: *mut u8) {
        unsafe {
            if !self.is_virtual_ref(vref_ptr) {
                return;
            }
            // `virtualref.py:125 assert real_object` — caller must
            // supply a non-null materialised object.
            debug_assert!(!real_object.is_null());
            let vref = &mut *(vref_ptr as *mut JitVirtualRef);
            // `virtualref.py:127 assert vref.virtual_token != TOKEN_TRACING_RESCALL`
            debug_assert_ne!(vref.virtual_token, token_tracing_rescall());
            vref.virtual_token = TOKEN_NONE;
            vref.forced = real_object;
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

    /// `virtualref.py:174-176`: `force_virtual` raises
    /// `InvalidVirtualRef` when `virtual_token == TOKEN_NONE` and
    /// `forced` is null.  pyre surfaces the same shape as
    /// `Err(InvalidVirtualRef)`.
    #[test]
    fn force_virtual_invalid_when_token_none_and_forced_null() {
        let info = VirtualRefInfo::new();
        let mut vref = JitVirtualRef {
            type_tag: VREF_TYPE_TAG,
            virtual_token: TOKEN_NONE,
            forced: std::ptr::null_mut(),
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        let err = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_fn must not be called when token == TOKEN_NONE")
            })
        }
        .expect_err(
            "TOKEN_NONE + null forced must surface Err(InvalidVirtualRef) per virtualref.py:174-176",
        );
        assert_eq!(err, InvalidVirtualRef);
    }

    /// `virtualref.py:160-167`: when `virtual_token ==
    /// TOKEN_TRACING_RESCALL`, `force_virtual` resets the token to
    /// `TOKEN_NONE` and returns the existing `forced` value
    /// without calling the materialiser.
    #[test]
    fn force_virtual_tracing_rescall_resets_token() {
        let info = VirtualRefInfo::new();
        let real_obj: *mut u8 = 0xCAFE_F00Du64 as *mut u8;
        let mut vref = JitVirtualRef {
            type_tag: VREF_TYPE_TAG,
            virtual_token: token_tracing_rescall(),
            forced: real_obj,
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        let result = unsafe {
            info.force_virtual(vref_ptr, |_| {
                panic!("force_fn must not be called for TOKEN_TRACING_RESCALL branch")
            })
        }
        .expect("tracing-rescall branch must succeed");
        assert_eq!(result, real_obj);
        assert!(
            vref.virtual_token.is_null(),
            "token must be reset to TOKEN_NONE"
        );
    }

    /// `virtualref.py:101-102`: `tracing_before_residual_call`
    /// returns silently when the gcref is not a JitVirtualRef.
    /// pyre's `is_virtual_ref` guard mirrors the shape — a
    /// non-vref pointer (here a `u64` whose first 8 bytes are NOT
    /// `VREF_TYPE_TAG`) must be left untouched.
    #[test]
    fn tracing_before_residual_call_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        unsafe {
            info.tracing_before_residual_call(ptr);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py:108-109`: `tracing_after_residual_call`
    /// returns `false` for a non-vref input.
    #[test]
    fn tracing_after_residual_call_returns_false_for_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        let forced = unsafe { info.tracing_after_residual_call(ptr) };
        assert!(!forced, "non-vref input must surface as not-forced");
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// `virtualref.py:123-124`: `continue_tracing` returns silently
    /// when the gcref is not a JitVirtualRef.
    #[test]
    fn continue_tracing_skips_non_vref() {
        let info = VirtualRefInfo::new();
        let mut not_a_vref: u64 = 0x1234_5678_9ABC_DEF0;
        let ptr = &mut not_a_vref as *mut u64 as *mut u8;
        let real_obj: *mut u8 = 0xDEADBEEFu64 as *mut u8;
        unsafe {
            info.continue_tracing(ptr, real_obj);
        }
        assert_eq!(
            not_a_vref, 0x1234_5678_9ABC_DEF0,
            "non-vref input must not be mutated",
        );
    }

    /// Sanity: a real `JitVirtualRef` IS recognised by
    /// `is_virtual_ref` and the tracing helpers DO mutate it
    /// (`virtualref.py:103-105` happy path).
    #[test]
    fn tracing_before_residual_call_mutates_real_vref() {
        let info = VirtualRefInfo::new();
        let real_obj: *mut u8 = 0xCAFE_F00Du64 as *mut u8;
        let mut vref = JitVirtualRef {
            type_tag: VREF_TYPE_TAG,
            virtual_token: TOKEN_NONE,
            forced: real_obj,
        };
        let vref_ptr = &mut vref as *mut JitVirtualRef as *mut u8;
        unsafe {
            info.tracing_before_residual_call(vref_ptr);
        }
        assert_eq!(
            vref.virtual_token,
            token_tracing_rescall(),
            "real vref must transition to TOKEN_TRACING_RESCALL",
        );
    }
}
