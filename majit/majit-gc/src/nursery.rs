/// Bump-pointer nursery allocator.
///
/// A fixed-size memory region where young objects are allocated by
/// advancing a free pointer. When the nursery is full, a minor
/// collection copies live objects out.
///
/// Layout: [header0|payload0|header1|payload1|...|free...top]
///          ^nursery_start                       ^free  ^top
use std::alloc::{self, Layout};
use std::ptr;

use crate::header::GcHeader;

// ── Heap-allocated nursery pointers ──
//
// incminimark.py:324-325 parity: nursery_free and nursery_top are fields
// in the GC object. The JIT reads their addresses via gc_adr_of_nursery_free
// / gc_adr_of_nursery_top (framework.py:994-997, gc.py:525-531).
//
// In Rust, we allocate these two pointers on the heap via Box<NurseryPtrs>.
// The JIT and the runtime both read/write the SAME memory — no separate
// global statics, no dual-state synchronization.
//
// RPython x86/assembler.py:2567 malloc_cond_varsize_frame inline path:
//   ecx = load(nursery_free_adr)
//   edx = ecx + size
//   cmp edx, load(nursery_top_adr)
//   ja slow_path
//   store(nursery_free_adr, edx)

/// incminimark.py:324-325 parity: the two mutable nursery pointers
/// live at a stable heap address so the JIT can hardcode their addresses.
#[repr(C)]
pub struct NurseryPtrs {
    /// incminimark.py:324 self.nursery_free
    pub free: *mut u8,
    /// incminimark.py:325 self.nursery_top
    pub top: *const u8,
}

/// incminimark.py:240 fallback `TRANSLATION_PARAMS["nursery_size"]`.
///
/// This is not the usual runtime nursery size when `read_from_env=True`:
/// incminimark.py:467-470 first asks the environment/cache estimator and
/// only falls back to this value if that fails.
pub const TRANSLATION_NURSERY_SIZE: usize = 896 * 1024;

/// env.py:441 `NURSERY_SIZE_UNKNOWN_CACHE`.
///
/// PyPy's translated incminimark normally estimates the nursery from cache
/// size, but its documented fallback is 4MB.  majit does not yet implement
/// the platform-specific cache probe, so `GcConfig::default` uses this after
/// checking `PYPY_GC_NURSERY`.
pub const DEFAULT_NURSERY_SIZE: usize = 4 * 1024 * 1024;

/// Nursery memory region with bump-pointer allocation.
///
/// incminimark.py:324-325 parity: nursery_free and nursery_top live in
/// a heap-allocated NurseryPtrs struct at a stable address. Both the JIT
/// inline fast path and the runtime slow path read/write the same fields.
pub struct Nursery {
    /// Start of the nursery memory region.
    start: *mut u8,
    /// Total size of the nursery.
    size: usize,
    /// Heap-allocated free/top pointers. The Box ensures stable addresses
    /// that the JIT can hardcode into compiled code.
    ptrs: Box<NurseryPtrs>,
}

// Safety: The nursery owns its memory exclusively and only one thread accesses it.
unsafe impl Send for Nursery {}

impl Nursery {
    /// incminimark.py:553-560 allocate_nursery parity:
    ///   self.nursery_free = self.nursery
    ///   self.nursery_top = self.nursery + self.nursery_size
    pub fn new(size: usize) -> Self {
        let layout = Layout::from_size_align(size, 16).expect("invalid nursery layout");
        let start = unsafe { alloc::alloc_zeroed(layout) };
        if start.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let top = unsafe { start.add(size) };
        let ptrs = Box::new(NurseryPtrs { free: start, top });
        Nursery { start, size, ptrs }
    }

    /// incminimark.py:676-680 malloc_fixedsize parity:
    ///   result = self.nursery_free
    ///   self.nursery_free = new_free = result + totalsize
    ///   if new_free > self.nursery_top: collect_and_reserve()
    ///
    /// Returns null when the nursery is full (caller must collect & retry).
    #[inline]
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        // Ensure minimum size for forwarding during collection.
        let total_size = total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE);
        // Align to 8 bytes.
        let total_size = (total_size + 7) & !7;

        let result = self.ptrs.free;
        // Use wrapping_add for the bound probe: when the nursery is already
        // full `result` sits at `top` (one-past-end), so an in-bounds `.add`
        // would be UB. RPython's integer arithmetic cannot hit this because
        // it operates on untyped addresses.
        let new_free = result.wrapping_add(total_size);
        if new_free as *const u8 > self.ptrs.top {
            return ptr::null_mut();
        }
        self.ptrs.free = new_free;
        result
    }

    /// incminimark.py:1946 parity: reset nursery after minor collection.
    ///   self.nursery_free = self.nursery
    pub fn reset(&mut self) {
        unsafe {
            ptr::write_bytes(self.start, 0, self.size);
        }
        self.ptrs.free = self.start;
    }

    /// incminimark.py:676: current nursery_free.
    #[inline]
    pub fn free_ptr(&self) -> *mut u8 {
        self.ptrs.free
    }

    /// gc.py:525-531 get_nursery_free_addr parity.
    #[inline]
    pub fn free_addr(&self) -> usize {
        std::ptr::addr_of!(self.ptrs.free) as usize
    }

    /// Set the free pointer (used after collection with pinned objects).
    ///
    /// # Safety
    /// `ptr` must be within the nursery bounds.
    pub unsafe fn set_free_ptr(&mut self, ptr: *mut u8) {
        debug_assert!(ptr as usize >= self.start as usize);
        debug_assert!(ptr as usize <= self.ptrs.top as usize);
        self.ptrs.free = ptr;
    }

    /// incminimark.py:910,1947: set nursery_top (pinned object barriers).
    ///
    /// # Safety
    /// `ptr` must be within the nursery bounds.
    pub unsafe fn set_top_ptr(&mut self, ptr: *const u8) {
        debug_assert!(ptr as usize >= self.start as usize);
        debug_assert!(ptr as usize <= self.start as usize + self.size);
        self.ptrs.top = ptr;
    }

    /// incminimark.py:325: current nursery_top.
    #[inline]
    pub fn top_ptr(&self) -> *const u8 {
        self.ptrs.top
    }

    /// gc.py:525-531 get_nursery_top_addr parity.
    #[inline]
    pub fn top_addr(&self) -> usize {
        std::ptr::addr_of!(self.ptrs.top) as usize
    }

    /// Start of nursery.
    #[inline]
    pub fn start_ptr(&self) -> *const u8 {
        self.start
    }

    /// Total nursery size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Bytes currently used.
    #[inline]
    pub fn used(&self) -> usize {
        self.ptrs.free as usize - self.start as usize
    }

    /// Bytes remaining.
    #[inline]
    pub fn remaining(&self) -> usize {
        self.ptrs.top as usize - self.ptrs.free as usize
    }

    /// Check if an address is within the nursery.
    #[inline]
    pub fn contains(&self, addr: usize) -> bool {
        addr >= self.start as usize && addr < (self.start as usize + self.size)
    }
}

impl Drop for Nursery {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.size, 16).unwrap();
        unsafe {
            alloc::dealloc(self.start, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nursery_create() {
        let nursery = Nursery::new(4096);
        assert_eq!(nursery.size(), 4096);
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_alloc() {
        let mut nursery = Nursery::new(4096);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());
        // At least MIN_NURSERY_OBJ_SIZE, aligned to 8
        assert!(nursery.used() >= 32);

        let p2 = nursery.alloc(64);
        assert!(!p2.is_null());
        assert!(p2 > p1);
    }

    #[test]
    fn test_nursery_full() {
        let mut nursery = Nursery::new(64);

        let p1 = nursery.alloc(32);
        assert!(!p1.is_null());

        // The nursery is 64 bytes; after allocating 32 (aligned up to at least 16),
        // we may or may not have room for another 32.
        // But asking for the full size should eventually fail.
        let p_big = nursery.alloc(64);
        // Should fail: not enough space
        assert!(p_big.is_null());
    }

    #[test]
    fn test_nursery_reset() {
        let mut nursery = Nursery::new(4096);

        nursery.alloc(100);
        nursery.alloc(100);
        assert!(nursery.used() > 0);

        nursery.reset();
        assert_eq!(nursery.used(), 0);
        assert_eq!(nursery.remaining(), 4096);
    }

    #[test]
    fn test_nursery_contains() {
        let nursery = Nursery::new(4096);
        let start = nursery.start_ptr() as usize;
        let top = nursery.top_ptr() as usize;

        assert!(nursery.contains(start));
        assert!(nursery.contains(start + 100));
        assert!(!nursery.contains(top));
        assert!(!nursery.contains(start.wrapping_sub(1)));
    }

    #[test]
    fn test_nursery_zero_filled() {
        let mut nursery = Nursery::new(4096);
        let p = nursery.alloc(64);
        assert!(!p.is_null());
        // Check memory is zero-filled
        for i in 0..64 {
            assert_eq!(unsafe { *p.add(i) }, 0);
        }
    }

    #[test]
    fn test_nursery_alignment() {
        let mut nursery = Nursery::new(4096);
        for _ in 0..10 {
            let p = nursery.alloc(17); // unaligned request
            assert!(!p.is_null());
            assert_eq!(p as usize % 8, 0); // result is 8-byte aligned
        }
    }

    #[test]
    fn test_nursery_instance_addrs_stable() {
        // gc.py:525-531 parity: nursery free/top addresses are per-instance
        // fields of the live GC descriptor, not a process-global singleton.
        let nursery = Nursery::new(4096);
        let free_addr = nursery.free_addr();
        let top_addr = nursery.top_addr();
        assert_ne!(free_addr, 0);
        assert_ne!(top_addr, 0);
        assert_ne!(free_addr, top_addr);
        let free_val = unsafe { *(free_addr as *const *mut u8) };
        assert_eq!(free_val, nursery.free_ptr());
    }
}
