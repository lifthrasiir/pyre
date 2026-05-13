/// Old-generation allocator.
///
/// Objects that survive nursery collection are copied here. For Phase 0,
/// this is a simple allocation scheme using the system allocator. Each
/// allocation is tracked in a list so we can iterate all old-gen objects
/// during major (mark-sweep) collection.
use std::alloc::{self, Layout};
use std::ptr;

use crate::flags;
use crate::header::{GcHeader, header_of};

/// A single old-generation allocation record.
struct OldObject {
    /// Start of the raw allocation (for dealloc).
    /// For objects with card marking, this is before the card header bytes.
    /// incminimark.py:1046-1067: allocsize includes cardheadersize.
    alloc_start: usize,
    /// Address of the GcHeader.
    /// = alloc_start + card_header_bytes for card-marked objects,
    /// = alloc_start for regular objects.
    header_addr: usize,
    /// Layout used for deallocation.
    layout: Layout,
}

/// Simple old-generation allocator backed by the system allocator.
///
/// Tracks all allocations for mark-sweep collection.
pub struct OldGen {
    /// All live old-gen objects.
    objects: Vec<OldObject>,
    /// Sorted payload-address index for hot membership checks.
    payload_index: Vec<usize>,
    payload_index_dirty: bool,
    /// Total bytes allocated in old gen.
    total_bytes: usize,
}

// Safety: OldGen owns all its allocations, single-threaded access.
unsafe impl Send for OldGen {}

impl OldGen {
    pub fn new() -> Self {
        OldGen {
            objects: Vec::new(),
            payload_index: Vec::new(),
            payload_index_dirty: false,
            total_bytes: 0,
        }
    }

    /// Allocate space for an object of `total_size` bytes (header + payload).
    /// Returns a pointer to the header location. Memory is zero-filled.
    pub fn alloc(&mut self, total_size: usize) -> *mut u8 {
        self.alloc_with_card_header(total_size, 0)
    }

    /// incminimark.py:1046-1067: allocate with card header bytes prepended.
    ///
    /// Layout: `[card_bytes (card_header_bytes)] [GcHeader] [payload]`
    /// Returns pointer to the GcHeader (NOT the card bytes).
    /// Card bytes are zero-filled.
    pub fn alloc_with_card_header(
        &mut self,
        total_size: usize,
        card_header_bytes: usize,
    ) -> *mut u8 {
        let obj_size = total_size.max(GcHeader::MIN_NURSERY_OBJ_SIZE);
        let alloc_size = card_header_bytes + obj_size;
        let layout = Layout::from_size_align(alloc_size, 8).expect("invalid layout");
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        let header_ptr = unsafe { ptr.add(card_header_bytes) };
        let obj_record = OldObject {
            alloc_start: ptr as usize,
            header_addr: header_ptr as usize,
            layout,
        };
        self.payload_index
            .push(obj_record.header_addr + GcHeader::SIZE);
        self.payload_index_dirty = true;
        self.objects.push(obj_record);
        self.total_bytes += alloc_size;
        header_ptr
    }

    /// Allocate and copy data from a source address.
    /// `total_size` is the total size including header.
    /// Returns a pointer to the header of the new copy.
    ///
    /// # Safety
    /// `src` must point to at least `total_size` bytes of readable memory.
    pub unsafe fn alloc_and_copy(&mut self, src: *const u8, total_size: usize) -> *mut u8 {
        let dst = self.alloc(total_size);
        unsafe { ptr::copy_nonoverlapping(src, dst, total_size) };
        dst
    }

    /// Total bytes currently allocated in old gen.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of objects in old gen.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Perform mark-sweep collection.
    /// Before calling this, the caller must have set VISITED on all reachable objects.
    /// This frees all objects that do NOT have the VISITED flag, and clears
    /// VISITED on surviving objects.
    pub fn sweep(&mut self) {
        let mut surviving = Vec::new();
        let mut freed_bytes = 0usize;

        for obj_record in self.objects.drain(..) {
            let hdr = unsafe { &mut *(obj_record.header_addr as *mut GcHeader) };
            if hdr.has_flag(flags::VISITED) {
                // Survived: clear VISITED for next cycle.
                hdr.clear_flag(flags::VISITED);
                surviving.push(obj_record);
            } else {
                // Dead: free it. Dealloc from alloc_start (includes card header).
                freed_bytes += obj_record.layout.size();
                unsafe {
                    alloc::dealloc(obj_record.alloc_start as *mut u8, obj_record.layout);
                }
            }
        }

        self.total_bytes -= freed_bytes;
        self.objects = surviving;
        self.rebuild_payload_index();
    }

    /// Iterate all old-gen object addresses (payload address, after header).
    /// The callback receives the object payload address.
    pub fn for_each_object(&self, mut f: impl FnMut(usize)) {
        for obj_record in &self.objects {
            f(obj_record.header_addr + GcHeader::SIZE);
        }
    }

    /// Check whether `obj_addr` is the payload address of a tracked old-gen object.
    pub fn contains(&self, obj_addr: usize) -> bool {
        self.objects
            .iter()
            .any(|obj_record| obj_record.header_addr + GcHeader::SIZE == obj_addr)
    }

    /// Fast membership check for write-barrier hot paths.
    ///
    /// The authoritative shape remains `objects`, matching the simple old-gen
    /// allocation list. This index is rebuilt lazily so allocation stays a
    /// push, while repeated barrier checks avoid scanning every old object.
    pub fn contains_fast(&mut self, obj_addr: usize) -> bool {
        if self.payload_index_dirty {
            self.payload_index.sort_unstable();
            self.payload_index.dedup();
            self.payload_index_dirty = false;
        }
        self.payload_index.binary_search(&obj_addr).is_ok()
    }

    fn rebuild_payload_index(&mut self) {
        self.payload_index.clear();
        self.payload_index.extend(
            self.objects
                .iter()
                .map(|obj_record| obj_record.header_addr + GcHeader::SIZE),
        );
        self.payload_index.sort_unstable();
        self.payload_index.dedup();
        self.payload_index_dirty = false;
    }

    /// Mark an old-gen object as visited (for major collection).
    pub fn mark_visited(obj_addr: usize) {
        unsafe {
            (*header_of(obj_addr)).set_flag(flags::VISITED);
        }
    }
}

impl Default for OldGen {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for OldGen {
    fn drop(&mut self) {
        for obj_record in self.objects.drain(..) {
            unsafe {
                alloc::dealloc(obj_record.alloc_start as *mut u8, obj_record.layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oldgen_alloc() {
        let mut oldgen = OldGen::new();
        let ptr = oldgen.alloc(32);
        assert!(!ptr.is_null());
        assert_eq!(oldgen.object_count(), 1);
        assert!(oldgen.total_bytes() >= 32);
    }

    #[test]
    fn test_oldgen_alloc_and_copy() {
        let mut oldgen = OldGen::new();
        let src = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let dst = unsafe { oldgen.alloc_and_copy(src.as_ptr(), 16) };
        assert!(!dst.is_null());
        for i in 0..16 {
            assert_eq!(unsafe { *dst.add(i) }, src[i]);
        }
    }

    #[test]
    fn test_oldgen_sweep() {
        let mut oldgen = OldGen::new();

        // Allocate 3 objects
        let p1 = oldgen.alloc(GcHeader::SIZE + 16);
        let p2 = oldgen.alloc(GcHeader::SIZE + 16);
        let p3 = oldgen.alloc(GcHeader::SIZE + 16);
        assert_eq!(oldgen.object_count(), 3);

        // Mark p1 and p3 as visited (reachable), leave p2 unmarked (dead)
        let hdr1 = unsafe { &mut *(p1 as *mut GcHeader) };
        *hdr1 = GcHeader::new(0);
        hdr1.set_flag(flags::VISITED);

        let hdr2 = unsafe { &mut *(p2 as *mut GcHeader) };
        *hdr2 = GcHeader::new(0);
        // Not visited -> dead

        let hdr3 = unsafe { &mut *(p3 as *mut GcHeader) };
        *hdr3 = GcHeader::new(0);
        hdr3.set_flag(flags::VISITED);

        oldgen.sweep();

        assert_eq!(oldgen.object_count(), 2);

        // Verify VISITED is cleared on survivors
        let hdr1 = unsafe { &mut *(p1 as *mut GcHeader) };
        assert!(!hdr1.has_flag(flags::VISITED));
        let hdr3 = unsafe { &mut *(p3 as *mut GcHeader) };
        assert!(!hdr3.has_flag(flags::VISITED));
    }

    #[test]
    fn test_oldgen_contains_fast_rebuilds_after_sweep() {
        let mut oldgen = OldGen::new();

        let p1 = oldgen.alloc(GcHeader::SIZE + 16);
        let p2 = oldgen.alloc(GcHeader::SIZE + 16);
        let p3 = oldgen.alloc(GcHeader::SIZE + 16);
        let o1 = p1 as usize + GcHeader::SIZE;
        let o2 = p2 as usize + GcHeader::SIZE;
        let o3 = p3 as usize + GcHeader::SIZE;

        assert!(oldgen.contains_fast(o1));
        assert!(oldgen.contains_fast(o2));
        assert!(oldgen.contains_fast(o3));

        unsafe {
            *(p1 as *mut GcHeader) = GcHeader::new(0);
            *(p2 as *mut GcHeader) = GcHeader::new(0);
            *(p3 as *mut GcHeader) = GcHeader::new(0);
            (*(p1 as *mut GcHeader)).set_flag(flags::VISITED);
            (*(p3 as *mut GcHeader)).set_flag(flags::VISITED);
        }

        oldgen.sweep();

        assert!(oldgen.contains_fast(o1));
        assert!(!oldgen.contains_fast(o2));
        assert!(oldgen.contains_fast(o3));
    }

    #[test]
    fn test_oldgen_for_each() {
        let mut oldgen = OldGen::new();
        oldgen.alloc(GcHeader::SIZE + 8);
        oldgen.alloc(GcHeader::SIZE + 8);

        let mut count = 0;
        oldgen.for_each_object(|_addr| count += 1);
        assert_eq!(count, 2);
    }
}
