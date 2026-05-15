//! Thread-local GC allocation hook for host-side Python object allocators.
//!
//! `pyre-object` sits below `majit-gc` in the dependency graph and must
//! not depend on it. Host-side allocators (`w_int_new`, `w_float_new`,
//! …) that want to route through the real GC instead of `Box::into_raw`
//! go through the callback registered here. `pyre-jit::eval` installs
//! the concrete trampoline on `JitDriver` init so the callback reaches
//! the backend-owned GC allocator via `majit_gc` TLS hooks.
//!
//! Callers use [`try_gc_alloc`] which returns `None` when no hook is
//! installed — they fall back to the `Box::into_raw` Phase 1 path in
//! that case. Session-by-session migration drops the `Box::into_raw`
//! fallback at each call site as the hook's reliability is verified
//! under the full bench suite.
//!
//! Layering: this module has no external dependencies. It defines the
//! function-pointer slot only. Wire-up lives in `pyre-jit`.

use std::cell::Cell;

/// Signature of the host-side GC allocation callback.
///
/// `type_id` is the backend-registered GC type id (same id used by
/// JIT-compiled `NewWithVtable`). `payload_size` is the number of
/// payload bytes requested. The callback returns an uninitialised
/// pointer to managed memory of exactly that size, ready for raw
/// field writes. On allocation failure the callback returns
/// `std::ptr::null_mut()`.
pub type GcAllocHookFn = fn(type_id: u32, payload_size: usize) -> *mut u8;

thread_local! {
    static GC_ALLOC_HOOK: Cell<Option<GcAllocHookFn>> = const { Cell::new(None) };
    static GC_ALLOC_STABLE_HOOK: Cell<Option<GcAllocHookFn>> = const { Cell::new(None) };
}

/// Install the allocation callback for this thread. Overwrites any
/// previously-installed hook.
pub fn register_gc_alloc_hook(hook: GcAllocHookFn) {
    GC_ALLOC_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the callback on this thread. Subsequent [`try_gc_alloc`]
/// returns `None` until a new hook is registered.
pub fn clear_gc_alloc_hook() {
    GC_ALLOC_HOOK.with(|cell| cell.set(None));
}

/// Attempt a GC allocation via the installed hook. Returns `None`
/// when no hook is installed on this thread, or `Some(null)` when the
/// hook itself returned null.
#[inline]
pub fn try_gc_alloc(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_HOOK.with(|cell| cell.get().map(|f| f(type_id, payload_size)))
}

/// Install the stable (old-gen) allocation callback for this thread.
///
/// Used by host-side allocators (`w_int_new`, `w_float_new`, …)
/// whose callers hold the returned pointer on the Rust stack across
/// subsequent allocations without registering it as a GC root
/// (Task #141). The backend routes this to an old-gen allocator
/// whose returned pointer is stable across minor and major
/// collections (MiniMark mark-sweep does not move old-gen objects).
pub fn register_gc_alloc_stable_hook(hook: GcAllocHookFn) {
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the stable-allocation callback on this thread.
pub fn clear_gc_alloc_stable_hook() {
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.set(None));
}

/// Attempt a stable-address GC allocation via the installed hook.
/// See [`register_gc_alloc_stable_hook`] for semantics. Returns
/// `None` when no hook is installed on this thread.
#[inline]
pub fn try_gc_alloc_stable(type_id: u32, payload_size: usize) -> Option<*mut u8> {
    GC_ALLOC_STABLE_HOOK.with(|cell| cell.get().map(|f| f(type_id, payload_size)))
}

/// Signature of the host-side root-register callbacks (Task #141
/// option a). `slot` is a pointer to a slot holding a `PyObjectRef`
/// (equivalently `*mut u8`); the GC treats it as a live root until
/// [`try_gc_remove_root`] is called with the same pointer.
///
/// Used around host-side allocator calls that may trigger a minor
/// collection — the nursery-moving collector needs the caller's slot
/// registered so the live pointer is traced and updated.
///
/// RPython accomplishes this automatically via its GC transform
/// pass (shadowstack save/restore around safepoints). pyre has no
/// such pass, so root registration is explicit at the call site.
/// This is a documented PRE-EXISTING-ADAPTATION.
pub type GcAddRootHookFn = unsafe fn(slot: *mut *mut u8);
pub type GcRemoveRootHookFn = fn(slot: *mut *mut u8);

thread_local! {
    static GC_ADD_ROOT_HOOK: Cell<Option<GcAddRootHookFn>> = const { Cell::new(None) };
    static GC_REMOVE_ROOT_HOOK: Cell<Option<GcRemoveRootHookFn>> = const { Cell::new(None) };
}

/// Install the root-register / remove callbacks for this thread.
pub fn register_gc_root_hooks(add: GcAddRootHookFn, remove: GcRemoveRootHookFn) {
    GC_ADD_ROOT_HOOK.with(|cell| cell.set(Some(add)));
    GC_REMOVE_ROOT_HOOK.with(|cell| cell.set(Some(remove)));
}

/// Remove the root-register callbacks on this thread.
pub fn clear_gc_root_hooks() {
    GC_ADD_ROOT_HOOK.with(|cell| cell.set(None));
    GC_REMOVE_ROOT_HOOK.with(|cell| cell.set(None));
}

/// Register `slot` as a live GC root via the installed callback.
/// Returns `true` when the callback was invoked.
///
/// # Safety
/// Caller must keep `slot` valid until [`try_gc_remove_root`] is
/// called with the same pointer.
#[inline]
pub unsafe fn try_gc_add_root(slot: *mut *mut u8) -> bool {
    GC_ADD_ROOT_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            unsafe { f(slot) };
            true
        }
        None => false,
    })
}

/// Remove a previously-registered root via the installed callback.
/// Returns `true` when the callback was invoked.
#[inline]
pub fn try_gc_remove_root(slot: *mut *mut u8) -> bool {
    GC_REMOVE_ROOT_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            f(slot);
            true
        }
        None => false,
    })
}

/// Signature of the host-side "is GC-managed" predicate. Callers
/// (host-side allocators with mixed `try_gc_alloc_stable` /
/// `std::alloc` allocation paths during the L1/L2 stepping-stone
/// window) use this to discriminate GC-managed blocks from
/// `std::alloc`-backed ones at dealloc time.
pub type GcOwnsObjectHookFn = fn(addr: usize) -> bool;
pub type GcCurrentObjectAddressHookFn = fn(addr: usize) -> usize;

thread_local! {
    static GC_OWNS_OBJECT_HOOK: Cell<Option<GcOwnsObjectHookFn>> = const { Cell::new(None) };
    static GC_CURRENT_OBJECT_ADDRESS_HOOK: Cell<Option<GcCurrentObjectAddressHookFn>> =
        const { Cell::new(None) };
}

/// Install the GC-ownership predicate. Overwrites any previously-
/// installed hook.
pub fn register_gc_owns_object_hook(hook: GcOwnsObjectHookFn) {
    GC_OWNS_OBJECT_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the GC-ownership predicate on this thread.
pub fn clear_gc_owns_object_hook() {
    GC_OWNS_OBJECT_HOOK.with(|cell| cell.set(None));
}

/// Install the non-rooting current-address lookup hook.
pub fn register_gc_current_object_address_hook(hook: GcCurrentObjectAddressHookFn) {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the current-address lookup hook on this thread.
pub fn clear_gc_current_object_address_hook() {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| cell.set(None));
}

/// Whether `addr` lies inside the active backend's managed GC heap.
/// Returns `false` when no hook is installed — callers treat that as
/// "no GC owns this pointer" and fall through to their non-GC
/// dealloc path. This is the host-side mirror of
/// `majit_gc::gc_owns_object`.
#[inline]
pub fn try_gc_owns_object(addr: *mut u8) -> bool {
    GC_OWNS_OBJECT_HOOK.with(|cell| match cell.get() {
        Some(f) => f(addr as usize),
        None => false,
    })
}

/// Return the current address for `addr` without registering it as a root.
/// When no hook is installed, or the active GC does not know the object, the
/// address is unchanged.
#[inline]
pub fn try_gc_current_object_address(addr: *mut u8) -> *mut u8 {
    GC_CURRENT_OBJECT_ADDRESS_HOOK.with(|cell| match cell.get() {
        Some(f) => f(addr as usize) as *mut u8,
        None => addr,
    })
}

/// Signature of the host-side write barrier callback. `obj` is the
/// GC-managed object whose field is being updated with a possible young
/// pointer. The backend decides whether `obj` is old enough to require
/// remembering.
pub type GcWriteBarrierHookFn = fn(obj: *mut u8);

thread_local! {
    static GC_WRITE_BARRIER_HOOK: Cell<Option<GcWriteBarrierHookFn>> = const { Cell::new(None) };
}

/// Install the write-barrier callback for this thread.
pub fn register_gc_write_barrier_hook(hook: GcWriteBarrierHookFn) {
    GC_WRITE_BARRIER_HOOK.with(|cell| cell.set(Some(hook)));
}

/// Remove the write-barrier callback on this thread.
pub fn clear_gc_write_barrier_hook() {
    GC_WRITE_BARRIER_HOOK.with(|cell| cell.set(None));
}

/// Run the active GC write barrier for `obj` when one is installed.
#[inline]
pub fn try_gc_write_barrier(obj: *mut u8) -> bool {
    GC_WRITE_BARRIER_HOOK.with(|cell| match cell.get() {
        Some(f) => {
            f(obj);
            true
        }
        None => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static LAST_TYPE_ID: AtomicUsize = AtomicUsize::new(0);
    static LAST_SIZE: AtomicUsize = AtomicUsize::new(0);

    fn mock_hook(type_id: u32, payload_size: usize) -> *mut u8 {
        LAST_TYPE_ID.store(type_id as usize, Ordering::Relaxed);
        LAST_SIZE.store(payload_size, Ordering::Relaxed);
        // Return a non-null dummy pointer. Tests don't dereference it.
        payload_size as *mut u8
    }

    fn null_hook(_type_id: u32, _payload_size: usize) -> *mut u8 {
        std::ptr::null_mut()
    }

    #[test]
    fn returns_none_when_unregistered() {
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 16).is_none());
    }

    #[test]
    fn invokes_registered_hook_with_args() {
        register_gc_alloc_hook(mock_hook);
        let ptr = try_gc_alloc(7, 24);
        assert!(ptr.is_some());
        assert_eq!(ptr.unwrap() as usize, 24);
        assert_eq!(LAST_TYPE_ID.load(Ordering::Relaxed), 7);
        assert_eq!(LAST_SIZE.load(Ordering::Relaxed), 24);
        clear_gc_alloc_hook();
    }

    #[test]
    fn clear_removes_hook() {
        register_gc_alloc_hook(mock_hook);
        assert!(try_gc_alloc(1, 8).is_some());
        clear_gc_alloc_hook();
        assert!(try_gc_alloc(1, 8).is_none());
    }

    #[test]
    fn hook_returning_null_propagates_some_null() {
        register_gc_alloc_hook(null_hook);
        let ptr = try_gc_alloc(1, 8);
        assert!(ptr.is_some());
        assert!(ptr.unwrap().is_null());
        clear_gc_alloc_hook();
    }

    static LAST_ROOT_PTR: AtomicUsize = AtomicUsize::new(0);
    static REMOVE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static WRITE_BARRIER_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe fn mock_add_root(slot: *mut *mut u8) {
        LAST_ROOT_PTR.store(slot as usize, Ordering::Relaxed);
    }
    fn mock_remove_root(slot: *mut *mut u8) {
        let _ = slot;
        REMOVE_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    fn mock_write_barrier(obj: *mut u8) {
        let _ = obj;
        WRITE_BARRIER_CALLS.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn root_hooks_register_and_remove_round_trip() {
        clear_gc_root_hooks();
        let mut slot: *mut u8 = std::ptr::null_mut();
        assert!(!unsafe { try_gc_add_root(&mut slot as *mut *mut u8) });
        assert!(!try_gc_remove_root(&mut slot as *mut *mut u8));

        LAST_ROOT_PTR.store(0, Ordering::Relaxed);
        REMOVE_CALLS.store(0, Ordering::Relaxed);
        register_gc_root_hooks(mock_add_root, mock_remove_root);

        let slot_ptr = &mut slot as *mut *mut u8;
        assert!(unsafe { try_gc_add_root(slot_ptr) });
        assert_eq!(LAST_ROOT_PTR.load(Ordering::Relaxed), slot_ptr as usize);
        assert!(try_gc_remove_root(slot_ptr));
        assert_eq!(REMOVE_CALLS.load(Ordering::Relaxed), 1);

        clear_gc_root_hooks();
        assert!(!unsafe { try_gc_add_root(slot_ptr) });
        assert!(!try_gc_remove_root(slot_ptr));
    }

    #[test]
    fn stable_hook_is_independent_from_nursery_hook() {
        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
        assert!(try_gc_alloc(1, 8).is_none());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_hook(mock_hook);
        // Stable hook still not installed.
        assert!(try_gc_alloc(1, 8).is_some());
        assert!(try_gc_alloc_stable(1, 8).is_none());

        register_gc_alloc_stable_hook(mock_hook);
        let ptr = try_gc_alloc_stable(3, 32);
        assert!(ptr.is_some());
        assert_eq!(ptr.unwrap() as usize, 32);
        assert_eq!(LAST_TYPE_ID.load(Ordering::Relaxed), 3);
        assert_eq!(LAST_SIZE.load(Ordering::Relaxed), 32);

        clear_gc_alloc_hook();
        clear_gc_alloc_stable_hook();
    }

    #[test]
    fn write_barrier_hook_registers_invokes_and_clears() {
        clear_gc_write_barrier_hook();
        let obj = 0x1000usize as *mut u8;
        assert!(!try_gc_write_barrier(obj));

        WRITE_BARRIER_CALLS.store(0, Ordering::Relaxed);
        register_gc_write_barrier_hook(mock_write_barrier);
        assert!(try_gc_write_barrier(obj));
        assert_eq!(WRITE_BARRIER_CALLS.load(Ordering::Relaxed), 1);

        clear_gc_write_barrier_hook();
        assert!(!try_gc_write_barrier(obj));
    }
}
