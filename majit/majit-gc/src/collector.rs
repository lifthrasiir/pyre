/// MiniMarkGC — the core collector implementing the GcAllocator trait.
///
/// A generational copying collector with:
/// - Bump-pointer nursery for young objects
/// - System-allocator old gen with mark-sweep for major collection
/// - Write barrier with remembered set for old-to-young pointers
///
/// Modeled after incminimark's minor/major collection.
use std::collections::{HashMap, HashSet};

use majit_ir::GcRef;

use crate::GcAllocator;
use crate::flags;
use crate::header::{GcHeader, header_of};
use crate::nursery::{DEFAULT_NURSERY_SIZE, Nursery};
use crate::oldgen::OldGen;
use crate::trace::{ClassTypeLayout, TypeEntry, TypeInfo, TypeInfoLayout, TypeRegistry};

/// Configuration for the MiniMarkGC.
pub struct GcConfig {
    /// Nursery size in bytes.
    pub nursery_size: usize,
    /// Maximum object size that can be allocated in the nursery.
    /// Larger objects go directly to old gen.
    pub large_object_threshold: usize,
    /// incminimark.py:275: card_page_indices (0 disables card marking).
    /// Must be a power of two.
    pub card_page_indices: u32,
}

fn read_size_from_env(varname: &str) -> Option<usize> {
    // rpython/memory/gc/env.py:17-40 `_read_float_and_factor_from_env`
    // accepts plain numbers plus K/M/G suffixes, optionally followed by B.
    let raw = std::env::var(varname).ok()?;
    let mut value = raw.trim();
    if value.is_empty() {
        return None;
    }
    if value.len() > 1 && matches!(value.as_bytes().last(), Some(b'b' | b'B')) {
        value = &value[..value.len() - 1];
    }
    if value.is_empty() {
        return None;
    }
    let (number, factor) = match value.as_bytes().last().copied() {
        Some(b'k') | Some(b'K') => (&value[..value.len() - 1], 1024.0),
        Some(b'm') | Some(b'M') => (&value[..value.len() - 1], 1024.0 * 1024.0),
        Some(b'g') | Some(b'G') => (&value[..value.len() - 1], 1024.0 * 1024.0 * 1024.0),
        Some(_) => (value, 1.0),
        None => return None,
    };
    let parsed = number.parse::<f64>().ok()?;
    let bytes = parsed * factor;
    (bytes > 0.0).then_some(bytes as usize)
}

fn default_nursery_size() -> usize {
    // incminimark.py:459-470:
    //   newsize = env.read_from_env('PYPY_GC_NURSERY')
    //   if newsize <= 0: newsize = env.estimate_best_nursery_size()
    //   if newsize <= 0: newsize = defaultsize
    //
    // Pyre does not yet port the platform-specific cache estimator, so use
    // env.py:441's 4MB unknown-cache fallback.
    read_size_from_env("PYPY_GC_NURSERY").unwrap_or(DEFAULT_NURSERY_SIZE)
}

impl Default for GcConfig {
    fn default() -> Self {
        // large_object = (16384+512)*8 from incminimark for 64-bit
        GcConfig {
            nursery_size: default_nursery_size(),
            large_object_threshold: (16384 + 512) * 8,
            card_page_indices: 128,
        }
    }
}

/// Root set: a list of locations that hold GcRef values the GC must trace.
///
/// Each root is a pointer to a GcRef-sized slot. During collection,
/// the GC reads the GcRef from this slot, traces it, and writes back
/// the (possibly updated) value.
pub struct RootSet {
    /// Stack roots: mutable pointers to GcRef slots on the stack or in frames.
    roots: Vec<*mut GcRef>,
}

unsafe impl Send for RootSet {}

impl RootSet {
    pub fn new() -> Self {
        RootSet { roots: Vec::new() }
    }

    /// Add a root. The pointer must remain valid until removed.
    ///
    /// # Safety
    /// The caller must ensure the pointer remains valid for the lifetime of the root.
    pub unsafe fn add(&mut self, root: *mut GcRef) {
        self.roots.push(root);
    }

    /// Remove a root.
    pub fn remove(&mut self, root: *mut GcRef) {
        if let Some(pos) = self.roots.iter().position(|r| *r == root) {
            self.roots.swap_remove(pos);
        }
    }

    /// Clear all roots.
    pub fn clear(&mut self) {
        self.roots.clear();
    }

    /// Number of roots.
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

impl Default for RootSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Default card page shift: each card covers 2^7 = 128 array elements.
pub const DEFAULT_CARD_PAGE_SHIFT: u32 = 7;

/// State for incremental major collection.
///
/// Instead of doing a full mark-sweep in one pause, the marking work
/// is spread across multiple minor collections. Each minor collection
/// piggybacks an incremental marking step that processes a bounded
/// number of objects from the gray stack.
struct IncrementalMarkState {
    /// Objects still to be scanned (gray set).
    gray_stack: Vec<usize>,
    /// Whether an incremental cycle is in progress.
    marking_in_progress: bool,
    /// Number of objects marked so far in this cycle.
    objects_marked: usize,
    /// Target number of bytes to trace per increment.
    ///
    /// Mirrors incminimark's `gc_increment_step`, which defaults to
    /// `nursery_size * 2`.
    mark_budget_per_step: usize,
}

impl IncrementalMarkState {
    fn new(nursery_size: usize) -> Self {
        IncrementalMarkState {
            gray_stack: Vec::new(),
            marking_in_progress: false,
            objects_marked: 0,
            mark_budget_per_step: (nursery_size.saturating_mul(2)).max(1),
        }
    }
}

/// Default ratio of old-gen growth that triggers an incremental cycle.
/// When old-gen bytes exceed `last_major_bytes * MAJOR_COLLECT_RATIO`,
/// a new incremental cycle starts.
const MAJOR_COLLECT_RATIO: f64 = 1.82;

/// The MiniMark generational GC.
#[allow(non_snake_case)] // _T_IS_RPYTHON_INSTANCE_BYTE keeps the RPython spelling
pub struct MiniMarkGC {
    /// The nursery (young generation).
    nursery: Nursery,
    /// The old generation.
    oldgen: OldGen,
    /// Type registry for tracing objects.
    pub types: TypeRegistry,
    /// Root set.
    pub roots: RootSet,
    /// Remembered set: old objects that may point to young objects.
    /// These are old-gen object payload addresses whose TRACK_YOUNG_PTRS
    /// flag has been cleared by the write barrier.
    remembered_set: Vec<usize>,
    /// incminimark.py:346-352: objects with GCFLAG_CARDS_SET bit.
    /// Card bits are stored inline before each object's GcHeader.
    /// This list tracks which objects have at least one card bit set.
    old_objects_with_cards_set: Vec<usize>,
    /// Configuration.
    config: GcConfig,
    /// Count of minor collections performed.
    pub minor_collections: usize,
    /// Count of major collections performed.
    pub major_collections: usize,
    /// State for incremental major collection.
    incr_state: IncrementalMarkState,
    /// Old-gen bytes at the end of the last completed major collection.
    /// Used to decide when to start the next incremental cycle.
    last_major_bytes: usize,
    /// Bytes promoted to old gen since the current incremental cycle started.
    ///
    /// Mirrors incminimark's `size_objects_made_old`.
    bytes_made_old_since_cycle: usize,
    /// Promotion credit granted by completed major-GC steps within the current
    /// incremental cycle.
    ///
    /// Mirrors incminimark's `threshold_objects_made_old`.
    threshold_bytes_made_old: usize,
    /// Pinned nursery objects that must not be moved during minor collection.
    pinned_objects: HashSet<usize>,
    /// Registry of compiled code regions for GC root scanning.
    pub compiled_code_registry: CompiledCodeRegistry,
    /// llsupport/gc.py:563 vtable→typeid mapping. RPython derives this
    /// arithmetically from the GC `type_info_group` base; pyre's GC
    /// keeps an explicit table because frontends register vtables
    /// independently of any translator pipeline.
    vtable_to_type_id: HashMap<usize, u32>,
    /// `gc.py:603-622 _setup_guard_is_object` instance state.
    /// `_infobits_offset` is the byte offset of `infobits` inside
    /// `TYPE_INFO`; `_infobits_offset_plus` is the additional offset
    /// past leading zero bytes of the little-endian flag word.
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` is the single non-zero byte
    /// pulled out of the flag word.
    ///
    /// Mirrors RPython's GcLLDescr_framework instance fields exactly
    /// — they're computed once at construction by `_setup_guard_is_object`
    /// and read by `get_translated_info_for_guard_is_object`. The
    /// `_T_IS_RPYTHON_INSTANCE_BYTE` field name keeps the RPython
    /// upper-case spelling so a `git grep` across the two trees
    /// finds the same name on both sides.
    _infobits_offset: usize,
    _infobits_offset_plus: usize,
    _T_IS_RPYTHON_INSTANCE_BYTE: u8,
    /// incminimark.py:315-317: log2(card_page_indices).
    /// 0 if card marking is disabled.
    card_page_shift: u32,
}

impl MiniMarkGC {
    /// Create a new GC with default configuration.
    pub fn new() -> Self {
        Self::with_config(GcConfig::default())
    }

    /// Create a new GC with custom configuration.
    pub fn with_config(config: GcConfig) -> Self {
        let nursery_size = config.nursery_size;
        let mut gc = MiniMarkGC {
            nursery: Nursery::new(config.nursery_size),
            oldgen: OldGen::new(),
            types: TypeRegistry::new(),
            roots: RootSet::new(),
            remembered_set: Vec::new(),
            old_objects_with_cards_set: Vec::new(),
            config,
            minor_collections: 0,
            major_collections: 0,
            incr_state: IncrementalMarkState::new(nursery_size),
            last_major_bytes: 0,
            bytes_made_old_since_cycle: 0,
            threshold_bytes_made_old: 0,
            pinned_objects: HashSet::new(),
            compiled_code_registry: CompiledCodeRegistry::new(),
            vtable_to_type_id: HashMap::new(),
            _infobits_offset: 0,
            _infobits_offset_plus: 0,
            _T_IS_RPYTHON_INSTANCE_BYTE: 0,
            card_page_shift: 0,
        };
        // incminimark.py:314-317
        if gc.config.card_page_indices > 0 {
            gc.card_page_shift = 0;
            while (1u32 << gc.card_page_shift) < gc.config.card_page_indices {
                gc.card_page_shift += 1;
            }
        }
        gc._setup_guard_is_object();
        gc
    }

    // ── incminimark.py:1292-1308 card marking geometry ──

    /// incminimark.py:1292-1299: number of machine words needed for card bits.
    fn card_marking_words_for_length(&self, length: usize) -> usize {
        const LONG_BIT: usize = 64;
        const LONG_BIT_SHIFT: usize = 6;
        (length + (LONG_BIT << self.card_page_shift) - 1)
            >> (self.card_page_shift as usize + LONG_BIT_SHIFT)
    }

    /// incminimark.py:1301-1308: number of bytes needed for card bits.
    fn card_marking_bytes_for_length(&self, length: usize) -> usize {
        (length + (8 << self.card_page_shift) - 1) >> (self.card_page_shift as usize + 3)
    }

    /// incminimark.py:1622-1625: address of card byte at `byteindex`.
    /// Card bytes are stored in reverse order before the GcHeader.
    /// `obj` is the payload address (after GcHeader).
    #[inline]
    fn get_card_ptr(obj: usize, byteindex: usize) -> *mut u8 {
        // addr_byte = obj - size_gc_header  (= GcHeader address)
        // return addr_byte + (~byteindex)   (= addr_byte - 1 - byteindex)
        (obj - GcHeader::SIZE - 1 - byteindex) as *mut u8
    }

    /// incminimark.py:955-1088: allocate a large object with optional card
    /// marker bits prepended before the GcHeader.
    ///
    /// `has_gc_ptrs_in_var` should be true for arrays containing GC pointers
    /// (i.e. `has_gcptr_in_varsize(typeid)` in RPython).
    pub fn alloc_in_oldgen_with_cards(
        &mut self,
        type_id: u32,
        total_size: usize,
        length: usize,
        has_gc_ptrs_in_var: bool,
    ) -> GcRef {
        // incminimark.py:1017-1030
        let (card_header_bytes, extra_flags) =
            if self.card_page_shift > 0 && has_gc_ptrs_in_var && length > 0 {
                let extra_words = self.card_marking_words_for_length(length);
                let chs = 8 * extra_words; // WORD * extra_words
                (chs, flags::HAS_CARDS | flags::TRACK_YOUNG_PTRS)
            } else {
                (0, flags::TRACK_YOUNG_PTRS)
            };

        let ptr = self
            .oldgen
            .alloc_with_card_header(total_size, card_header_bytes);
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        *hdr = GcHeader::with_flags(type_id, extra_flags);
        self.bytes_made_old_since_cycle = self
            .bytes_made_old_since_cycle
            .saturating_add(card_header_bytes + total_size);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// `gc.py:603-617 _setup_guard_is_object` parity. Computes
    /// `(_infobits_offset, _infobits_offset_plus, _T_IS_RPYTHON_INSTANCE_BYTE)`
    /// once at construction time and stores them as instance state.
    /// Mirrors line-by-line:
    ///
    /// ```python
    /// def _setup_guard_is_object(self):
    ///     from rpython.memory.gctypelayout import GCData, T_IS_RPYTHON_INSTANCE
    ///     import struct
    ///     infobits_offset, _ = symbolic.get_field_token(
    ///         GCData.TYPE_INFO, 'infobits', True)
    ///     mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)
    ///     assert mask.count('\x00') == len(mask) - 1
    ///     infobits_offset_plus = 0
    ///     while mask.startswith('\x00'):
    ///         infobits_offset_plus += 1
    ///         mask = mask[1:]
    ///     self._infobits_offset = infobits_offset
    ///     self._infobits_offset_plus = infobits_offset_plus
    ///     self._T_IS_RPYTHON_INSTANCE_BYTE = ord(mask[0])
    /// ```
    fn _setup_guard_is_object(&mut self) {
        // `symbolic.get_field_token(TYPE_INFO, 'infobits', True)`.
        let infobits_offset = TypeInfoLayout::INFOBITS_OFFSET;
        // `mask = struct.pack("l", T_IS_RPYTHON_INSTANCE)`.
        let mask = TypeInfoLayout::T_IS_RPYTHON_INSTANCE.to_le_bytes();
        // `assert mask.count('\x00') == len(mask) - 1`.
        let nonzero = mask.iter().filter(|&&b| b != 0).count();
        assert_eq!(
            nonzero, 1,
            "T_IS_RPYTHON_INSTANCE must occupy exactly one byte of \
             the packed Signed word (gc.py:610)"
        );
        // `while mask.startswith('\x00'): infobits_offset_plus += 1; mask = mask[1:]`.
        let mut infobits_offset_plus = 0usize;
        while infobits_offset_plus < mask.len() && mask[infobits_offset_plus] == 0 {
            infobits_offset_plus += 1;
        }
        self._infobits_offset = infobits_offset;
        self._infobits_offset_plus = infobits_offset_plus;
        self._T_IS_RPYTHON_INSTANCE_BYTE = mask[infobits_offset_plus];
    }

    /// Register a type and return its ID.
    pub fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    /// Check if an address is in the nursery.
    #[inline]
    pub fn is_in_nursery(&self, addr: usize) -> bool {
        self.nursery.contains(addr)
    }

    /// incminimark.py range-check parity: nursery membership is a pure
    /// range check; JIT inline nursery bump-alloc must produce GcRefs
    /// indistinguishable from the slow path's, so a side table would
    /// not stay in sync.
    #[inline]
    fn is_managed_heap_object(&self, addr: usize) -> bool {
        self.nursery.contains(addr) || self.oldgen.contains(addr)
    }

    /// incminimark.py:1208 is_in_nursery parity.
    #[inline]
    fn is_nursery_object_start(&self, addr: usize) -> bool {
        self.nursery.contains(addr)
    }

    /// Allocate a fixed-size object with the given type ID and size (excluding header).
    /// Returns a GcRef pointing to the object payload (after the header).
    pub fn alloc_with_type(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let total_size = GcHeader::SIZE + payload_size;

        // Large objects go directly to old gen.
        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            // Nursery full: trigger minor collection and retry.
            // minimark.py:1282 collect_and_reserve parity.
            self.do_collect_nursery();
            let ptr = self.nursery.alloc(total_size);
            if ptr.is_null() {
                // Still no space after collection. Allocate in old gen as fallback.
                return self.alloc_in_oldgen(type_id, total_size);
            }
            Self::init_nursery_object(ptr, type_id);
            return GcRef((ptr as usize) + GcHeader::SIZE);
        }

        Self::init_nursery_object(ptr, type_id);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Allocate without triggering collection.
    ///
    /// If the nursery cannot satisfy the request, this falls back directly to
    /// old-gen allocation so compiled code can keep running without needing
    /// stack-map-mediated collection.
    pub fn alloc_with_type_no_collect(&mut self, type_id: u32, payload_size: usize) -> GcRef {
        let total_size = GcHeader::SIZE + payload_size;

        if total_size > self.config.large_object_threshold {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        let ptr = self.nursery.alloc(total_size);
        if ptr.is_null() {
            return self.alloc_in_oldgen(type_id, total_size);
        }

        Self::init_nursery_object(ptr, type_id);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Initialize a nursery object's header.
    fn init_nursery_object(header_ptr: *mut u8, type_id: u32) {
        let hdr = unsafe { &mut *(header_ptr as *mut GcHeader) };
        // Young objects do NOT have TRACK_YOUNG_PTRS (we assume any
        // young object can point to any other young object).
        *hdr = GcHeader::new(type_id);
    }

    /// Allocate directly in old gen (for large objects or post-collection fallback).
    fn alloc_in_oldgen(&mut self, type_id: u32, total_size: usize) -> GcRef {
        let ptr = self.oldgen.alloc(total_size);
        let hdr = unsafe { &mut *(ptr as *mut GcHeader) };
        // Old objects start with TRACK_YOUNG_PTRS set (they need write barrier).
        *hdr = GcHeader::with_flags(type_id, flags::TRACK_YOUNG_PTRS);
        self.bytes_made_old_since_cycle =
            self.bytes_made_old_since_cycle.saturating_add(total_size);
        GcRef((ptr as usize) + GcHeader::SIZE)
    }

    /// Perform a minor (nursery) collection.
    ///
    /// 1. Scan roots: copy referenced nursery objects to old gen.
    /// 2. Process remembered set: copy nursery objects referenced by old-gen objects.
    /// 3. Iteratively process newly discovered references until stable.
    /// 4. Reset nursery.
    pub fn do_collect_nursery(&mut self) {
        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[gc][minor] start count={} remembered={} cards_set={}",
                self.minor_collections + 1,
                self.remembered_set.len(),
                self.old_objects_with_cards_set.len(),
            );
        }
        self.minor_collections += 1;

        // Phase 1: Process roots — copy nursery objects they point to.
        // We use raw pointers to avoid borrow checker issues since
        // copy_nursery_object mutates oldgen/nursery.
        // Pinned objects are left in place (not copied to old gen).
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for (_root_idx, root_ptr) in roots.iter().enumerate() {
            let root_ptr = *root_ptr;
            let gcref = unsafe { *root_ptr };
            if !gcref.is_null() && self.is_nursery_object_start(gcref.0) {
                if self.pinned_objects.contains(&gcref.0) {
                    continue;
                }
                let new_ref = self.copy_nursery_object(gcref.0);
                unsafe {
                    *root_ptr = new_ref;
                }
            }
        }

        // Phase 1b: Process shadow stack roots.
        // RPython gc.py: GcRootMap_shadowstack — walk the thread-local
        // shadow stack to find GC refs pushed by compiled JIT code.
        crate::shadow_stack::walk_roots(|gcref| {
            if self.is_nursery_object_start(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            }
        });

        // Phase 1c: Process jitframe shadow stack roots.
        // RPython root_walker.walk_roots with jitframe entries —
        // reads jf_gcmap from each jitframe and traces ref slots.
        // assembler.py:1122 (_call_header_shadowstack) pushes jf_ptr;
        // callbuilder.py:93 (push_gcmap) writes per-call gcmap to jf_gcmap.
        // Collect libc-jitframe slots so we can traverse/update them
        // after the walk finishes without reborrowing `self` inside the
        // tracer callback.
        let mut libc_jf_slots: Vec<*mut majit_ir::GcRef> = Vec::new();
        crate::shadow_stack::walk_jf_roots(|gcref| {
            if self.is_nursery_object_start(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            } else if !gcref.is_null() && self.oldgen.contains(gcref.0) {
                // RPython parity: old-gen jitframes need their interior
                // nursery refs traced directly. The custom_trace hook
                // walks gcmap bits to find Ref slots.
                self.trace_and_update_object(gcref.0);
            } else if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                // pyre dynasm extension: jitframes allocated via
                // `libc::calloc` in execute_token are neither in the
                // nursery nor the old gen. The registered libc-jitframe
                // tracer walks `jf_gcmap` bits to expose ref slots.
                crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                    libc_jf_slots.push(slot_ptr);
                });
            }
        });
        for slot_ptr in libc_jf_slots {
            let field_ref = unsafe { *slot_ptr };
            if !field_ref.is_null() && self.is_nursery_object_start(field_ref.0) {
                if !self.pinned_objects.contains(&field_ref.0) {
                    let new_ref = self.copy_nursery_object(field_ref.0);
                    unsafe {
                        *slot_ptr = new_ref;
                    }
                }
            }
        }

        // Phase 1d: Process blackhole interpreter register banks.
        // blackhole.py BlackholeInterpreter.registers_r parity: each
        // active blackhole frame's ref register file is part of the GC
        // root set. RPython traces these via the RPython object graph
        // (Box arrays); pyre stores raw i64 in Vec<i64> so we walk the
        // explicit thread-local stack of register banks.
        crate::shadow_stack::walk_bh_regs(|gcref| {
            if self.is_nursery_object_start(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            }
        });

        // Phase 1e: framework.py `root_walker.walk_roots` parity — the
        // embedding runtime plugs a walker that visits
        // `PyFrame.locals_cells_stack_w` across the active f_backref chain
        // so nursery refs held only by a Python frame local or operand-stack
        // slot survive collection.
        crate::walk_active_extra_roots(&mut |gcref| {
            if self.is_nursery_object_start(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            }
        });

        // Multi-registrar walker fan-out (rd_consts const-pool, etc.).
        crate::shadow_stack::walk_extra_roots(|gcref| {
            if self.is_nursery_object_start(gcref.0) {
                if !self.pinned_objects.contains(&gcref.0) {
                    *gcref = self.copy_nursery_object(gcref.0);
                }
            }
        });

        // incminimark parity: during an active marking cycle, old objects
        // remembered by the write barrier may already be black. Requeue
        // those black objects so the major collector rescans their new
        // outgoing references before sweep.
        if self.incr_state.marking_in_progress {
            let remembered_now: Vec<usize> = self.remembered_set.iter().copied().collect();
            for obj_addr in remembered_now {
                let hdr = unsafe { header_of(obj_addr) };
                if unsafe { (*hdr).has_flag(flags::VISITED) } {
                    self.incr_state.gray_stack.push(obj_addr);
                }
            }
        }

        // incminimark.py:1843-1862: while True loop —
        // collect_cardrefs_to_nursery, then collect_oldrefs_to_nursery.
        // collect_oldrefs_to_nursery may cause new card-marked objects to
        // appear, so we must re-enter if old_objects_with_cards_set is
        // non-empty after processing the remembered set.
        loop {
            // incminimark.py:1846-1847
            if self.card_page_shift > 0 {
                self.collect_cardrefs_to_nursery();
            }

            // incminimark.py:1855: collect_oldrefs_to_nursery.
            let mut idx = 0;
            loop {
                if idx >= self.remembered_set.len() {
                    break;
                }
                let obj_addr = self.remembered_set[idx];
                idx += 1;

                // incminimark.py:2095-2098: re-set TRACK_YOUNG_PTRS.
                unsafe {
                    (*header_of(obj_addr)).set_flag(flags::TRACK_YOUNG_PTRS);
                }

                // Trace this old-gen object's fields and copy any nursery
                // objects they reference.
                self.trace_and_update_object(obj_addr);
            }
            self.remembered_set.clear();

            // incminimark.py:1859-1862: loop back if card-marked objects appeared.
            if self.card_page_shift > 0 && !self.old_objects_with_cards_set.is_empty() {
                continue;
            }
            break;
        }

        // Reset nursery for new allocations, preserving pinned objects.
        if self.pinned_objects.is_empty() {
            self.nursery.reset();
        } else {
            self.reset_nursery_with_pinned();
        }

        // Minor collections must also drive incremental major-collection
        // progress. Like incminimark, take one or more major steps until
        // promoted bytes are back under the current step credit.
        self.run_major_progress_after_minor();
    }

    /// Copy a single nursery object to old gen.
    /// If already forwarded, returns the forwarding address.
    /// Pinned objects are left in place and returned as-is.
    fn copy_nursery_object(&mut self, obj_addr: usize) -> GcRef {
        // Pinned objects must stay in the nursery.
        if self.pinned_objects.contains(&obj_addr) {
            return GcRef(obj_addr);
        }

        // Keep the header access as a raw pointer rather than `&mut GcHeader`.
        // The subsequent `alloc_and_copy` performs a raw read over this same
        // byte range, which would invalidate a long-lived `&mut` under Rust's
        // aliasing rules; re-materialize the reference only for each scoped
        // read/write.
        let hdr_ptr = (obj_addr - GcHeader::SIZE) as *mut GcHeader;

        // Already forwarded?
        if unsafe { (*hdr_ptr).is_forwarded() } {
            let fwd_addr = unsafe { GcHeader::forwarding_address(hdr_ptr) };
            return GcRef(fwd_addr);
        }

        let type_id = unsafe { (*hdr_ptr).type_id() };
        if type_id as usize >= self.types.len() {
            panic!(
                "GC BUG: invalid type_id={} at obj_addr={:#x} (header_addr={:#x}, nursery_start={:#x}, forwarded={})",
                type_id,
                obj_addr,
                obj_addr - GcHeader::SIZE,
                self.nursery.start_ptr() as usize,
                unsafe { (*hdr_ptr).is_forwarded() },
            );
        }
        let type_info = self.types.get(type_id);

        // Compute the actual payload size (for varsize objects, read the length).
        let actual_payload_size = if type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            type_info.total_instance_size(length)
        } else {
            type_info.size
        };

        let total_size = GcHeader::SIZE + actual_payload_size;
        let has_gc_ptrs = type_info.has_gc_ptrs;

        // Allocate in old gen and copy.
        let header_ptr = obj_addr - GcHeader::SIZE;
        // Safety: header_ptr points to a valid nursery object of total_size bytes.
        let new_header_ptr = unsafe {
            self.oldgen
                .alloc_and_copy(header_ptr as *const u8, total_size)
        };
        let new_obj_addr = new_header_ptr as usize + GcHeader::SIZE;
        self.bytes_made_old_since_cycle =
            self.bytes_made_old_since_cycle.saturating_add(total_size);

        // Set TRACK_YOUNG_PTRS on the new old-gen object.
        unsafe {
            (*(new_header_ptr as *mut GcHeader)).set_flag(flags::TRACK_YOUNG_PTRS);
        }

        // Install forwarding pointer in the nursery copy.
        unsafe {
            GcHeader::set_forwarding_address(hdr_ptr, new_obj_addr);
        }

        // If this object has GC pointers, add it to the work list so we
        // trace its fields and update nursery references.
        if has_gc_ptrs {
            // Clear TRACK_YOUNG_PTRS temporarily so the processing loop
            // can re-set it after processing.
            unsafe {
                (*(new_header_ptr as *mut GcHeader)).clear_flag(flags::TRACK_YOUNG_PTRS);
            }
            self.remembered_set.push(new_obj_addr);
        }

        GcRef(new_obj_addr)
    }

    /// Trace an object's GC pointer fields and update any that point
    /// into the nursery by copying the target.
    fn trace_and_update_object(&mut self, obj_addr: usize) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        let type_info = self.types.get(type_id);

        // custom_trace_hook parity: use custom trace function if registered.
        if let Some(trace_fn) = type_info.custom_trace {
            let mut slots: Vec<*mut GcRef> = Vec::new();
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    slots.push(slot_ptr);
                });
            }
            for slot_ptr in slots {
                let field_ref = unsafe { *slot_ptr };
                if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                    let new_ref = self.copy_nursery_object(field_ref.0);
                    unsafe {
                        *slot_ptr = new_ref;
                    }
                }
            }
            return;
        }

        let gc_ptr_offsets: Vec<usize> = type_info.gc_ptr_offsets.clone();
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let item_size = type_info.item_size;
        let length_offset = type_info.length_offset;
        let base_size = type_info.size;

        // Process fixed-part GC pointer fields.
        for &offset in &gc_ptr_offsets {
            let slot = (obj_addr + offset) as *mut GcRef;
            let field_ref = unsafe { *slot };
            if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                let new_ref = self.copy_nursery_object(field_ref.0);
                unsafe {
                    *slot = new_ref;
                }
            }
        }

        // Process variable-part items if they contain GC pointers.
        if items_have_gc_ptrs && item_size > 0 {
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + base_size;
            for i in 0..length {
                let slot = (items_start + i * item_size) as *mut GcRef;
                let field_ref = unsafe { *slot };
                if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                    let new_ref = self.copy_nursery_object(field_ref.0);
                    unsafe {
                        *slot = new_ref;
                    }
                }
            }
        }
    }

    // ── Incremental marking ──

    /// Check whether old-gen growth warrants starting a new incremental
    /// major collection cycle.
    fn should_start_major_cycle(&self) -> bool {
        if self.last_major_bytes == 0 {
            // Never done a major collection. Use a minimum threshold so we
            // don't start an incremental cycle with a nearly empty old gen.
            return self.oldgen.total_bytes() > self.config.nursery_size;
        }
        self.oldgen.total_bytes() as f64 > self.last_major_bytes as f64 * MAJOR_COLLECT_RATIO
    }

    /// Begin a new incremental marking cycle.
    ///
    /// Seeds the gray stack with all root-reachable old-gen objects.
    pub fn start_incremental_cycle(&mut self) {
        self.incr_state.marking_in_progress = true;
        self.incr_state.gray_stack.clear();
        self.incr_state.objects_marked = 0;

        self.seed_major_roots();
    }

    fn seed_major_root(&mut self, gcref: GcRef) {
        if !gcref.is_null() && self.is_managed_heap_object(gcref.0) {
            let hdr = unsafe { header_of(gcref.0) };
            // SAFETY: header_of returns a raw pointer; keep each access
            // short-lived to avoid creating overlapping exclusive borrows.
            unsafe {
                if !(*hdr).has_flag(flags::VISITED) {
                    (*hdr).set_flag(flags::VISITED);
                    self.incr_state.gray_stack.push(gcref.0);
                }
            }
        }
    }

    fn seed_major_roots(&mut self) {
        // incminimark.py:2717 collect_roots: root_walker.walk_roots()
        // seeds stack roots for a major marking cycle. Mirror the same
        // root sets as minor collection, but mark old objects instead of
        // copying nursery objects.
        let roots: Vec<*mut GcRef> = self.roots.roots.iter().copied().collect();
        for root_ptr in roots {
            let gcref = unsafe { *root_ptr };
            self.seed_major_root(gcref);
        }

        crate::shadow_stack::walk_roots(|gcref| {
            self.seed_major_root(*gcref);
        });

        let mut libc_jf_refs: Vec<GcRef> = Vec::new();
        crate::shadow_stack::walk_jf_roots(|gcref| {
            if !gcref.is_null() && crate::shadow_stack::is_libc_jitframe(gcref.0) {
                crate::shadow_stack::trace_libc_jitframe(gcref.0, &mut |slot_ptr| {
                    let field_ref = unsafe { *slot_ptr };
                    if !field_ref.is_null() {
                        libc_jf_refs.push(field_ref);
                    }
                });
            } else {
                self.seed_major_root(*gcref);
            }
        });
        for gcref in libc_jf_refs {
            self.seed_major_root(gcref);
        }

        crate::shadow_stack::walk_bh_regs(|gcref| {
            self.seed_major_root(*gcref);
        });

        crate::walk_active_extra_roots(&mut |gcref| {
            self.seed_major_root(*gcref);
        });

        crate::shadow_stack::walk_extra_roots(|gcref| {
            self.seed_major_root(*gcref);
        });
    }

    /// Drive incremental major-collection progress after a minor collection.
    ///
    /// This follows incminimark's accounting rule: each major step grants
    /// `nursery_size / 2` bytes of promotion credit, and allocation-heavy
    /// minors may need multiple consecutive steps so old-gen growth does not
    /// outrun marking.
    fn run_major_progress_after_minor(&mut self) {
        if !self.incr_state.marking_in_progress && !self.should_start_major_cycle() {
            return;
        }

        loop {
            self.threshold_bytes_made_old = self
                .threshold_bytes_made_old
                .saturating_add(self.config.nursery_size / 2);

            if !self.incr_state.marking_in_progress {
                self.bytes_made_old_since_cycle = 0;
                self.threshold_bytes_made_old = self.config.nursery_size / 2;
                self.start_incremental_cycle();
            }

            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
                break;
            }

            if self.bytes_made_old_since_cycle <= self.threshold_bytes_made_old {
                break;
            }
        }
    }

    /// Perform one incremental marking step.
    ///
    /// Processes up to `mark_budget_per_step` bytes from the gray stack.
    /// Like incminimark, this is a byte budget, but we always process at least
    /// one object so very small budgets still make forward progress.
    /// Returns `true` if marking is complete (gray stack exhausted).
    pub fn incremental_mark_step(&mut self) -> bool {
        let mut budget = self.incr_state.mark_budget_per_step;
        let mut processed_any = false;
        while budget > 0 || !processed_any {
            let Some(obj_addr) = self.incr_state.gray_stack.pop() else {
                self.incr_state.marking_in_progress = false;
                return true; // marking complete
            };
            let obj_size = self.object_total_size(obj_addr);
            self.mark_object(obj_addr);
            self.incr_state.objects_marked += 1;
            budget = budget.saturating_sub(obj_size.max(1));
            processed_any = true;
        }
        false // more work to do
    }

    fn object_total_size(&self, obj_addr: usize) -> usize {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        let type_info = self.types.get(type_id);
        let payload_size = if type_info.item_size > 0 {
            let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
            type_info.total_instance_size(length)
        } else {
            type_info.size
        };
        GcHeader::SIZE + payload_size
    }

    /// Mark a single object: trace its GC pointer fields and push
    /// unmarked children onto the gray stack.
    fn mark_object(&mut self, obj_addr: usize) {
        let type_id = unsafe { (*header_of(obj_addr)).type_id() };
        let type_info = self.types.get(type_id);

        // custom_trace_hook parity for major GC marking.
        if let Some(trace_fn) = type_info.custom_trace {
            let mut refs: Vec<GcRef> = Vec::new();
            unsafe {
                trace_fn(obj_addr, &mut |slot_ptr: *mut GcRef| {
                    let field_ref = *slot_ptr;
                    if !field_ref.is_null() {
                        refs.push(field_ref);
                    }
                });
            }
            for field_ref in refs {
                if self.is_managed_heap_object(field_ref.0) {
                    let hdr = unsafe { header_of(field_ref.0) };
                    if unsafe { !(*hdr).has_flag(flags::VISITED) } {
                        unsafe { (*hdr).set_flag(flags::VISITED) };
                        self.incr_state.gray_stack.push(field_ref.0);
                    }
                }
            }
            return;
        }

        let gc_ptr_offsets = type_info.gc_ptr_offsets.clone();
        let items_have_gc_ptrs = type_info.items_have_gc_ptrs;
        let item_size = type_info.item_size;
        let length_offset = type_info.length_offset;
        let base_size = type_info.size;

        // Trace fixed-part fields. The `is_managed_heap_object` guard
        // mirrors the `custom_trace` path above and `seed_major_root`
        // (line 814): a `Ptr(GcStruct)` field can transiently point at
        // memory outside the GC-managed heap during the L1/L2 stepping-
        // stone state (e.g. `W_TupleObject.wrappeditems` →
        // `std::alloc`'d ItemsBlock until Task #98 lands). In that
        // window calling `header_of` on the field would dereference
        // memory before the std::alloc'd block. Upstream RPython
        // `_collect_obj` (incminimark.py:2739-2752) does not need this
        // guard because RPython's type system guarantees every
        // `Ptr(GcStruct)` is GC-managed. PRE-EXISTING-ADAPTATION:
        // matches `mark_object`'s own custom_trace-path guard and
        // converges away once every gc_ptr_offsets target is a real
        // GC allocation.
        for &offset in &gc_ptr_offsets {
            let field_ref = unsafe { *((obj_addr + offset) as *const GcRef) };
            if !field_ref.is_null() && self.is_managed_heap_object(field_ref.0) {
                let hdr = unsafe { header_of(field_ref.0) };
                if unsafe { !(*hdr).has_flag(flags::VISITED) } {
                    unsafe { (*hdr).set_flag(flags::VISITED) };
                    self.incr_state.gray_stack.push(field_ref.0);
                }
            }
        }

        // Trace variable-part items. Same `is_managed_heap_object`
        // guard as the fixed-part loop — `items_have_gc_ptrs` blocks
        // (e.g. `ItemsBlock` once it migrates to GC varsize) may
        // transiently coexist with std::alloc'd siblings during the
        // L1/L2 stepping-stone window.
        if items_have_gc_ptrs && item_size > 0 {
            let length = unsafe { *((obj_addr + length_offset) as *const usize) };
            let items_start = obj_addr + base_size;
            for i in 0..length {
                let field_ref = unsafe { *((items_start + i * item_size) as *const GcRef) };
                if !field_ref.is_null() && self.is_managed_heap_object(field_ref.0) {
                    let hdr = unsafe { header_of(field_ref.0) };
                    if unsafe { !(*hdr).has_flag(flags::VISITED) } {
                        unsafe { (*hdr).set_flag(flags::VISITED) };
                        self.incr_state.gray_stack.push(field_ref.0);
                    }
                }
            }
        }
    }

    /// Complete the sweep phase after incremental marking finishes.
    fn finish_incremental_cycle(&mut self) {
        self.major_collections += 1;
        self.oldgen.sweep();
        self.last_major_bytes = self.oldgen.total_bytes();
        self.bytes_made_old_since_cycle = 0;
        self.threshold_bytes_made_old = 0;
    }

    /// Whether an incremental marking cycle is currently in progress.
    pub fn is_incremental_marking(&self) -> bool {
        self.incr_state.marking_in_progress
    }

    /// Number of objects marked so far in the current incremental cycle.
    pub fn incremental_objects_marked(&self) -> usize {
        self.incr_state.objects_marked
    }

    /// Set the per-step marking budget in bytes.
    pub fn set_mark_budget(&mut self, budget: usize) {
        self.incr_state.mark_budget_per_step = budget;
    }

    /// Perform a full (major) mark-sweep collection.
    ///
    /// 1. First do a minor collection to promote all live nursery objects.
    /// 2. Mark phase: trace all roots and transitively mark reachable objects.
    /// 3. Sweep phase: free all unmarked old-gen objects.
    pub fn do_collect_full(&mut self) {
        // Minor collection first to empty the nursery.
        // Note: do_collect_nursery may itself start/advance an incremental
        // cycle, but we need a complete mark-sweep here regardless.
        self.do_collect_nursery();

        if self.incr_state.marking_in_progress {
            // An incremental cycle is in progress. Finish it by draining
            // the gray stack without budget limits.
            while !self.incr_state.gray_stack.is_empty() {
                let obj_addr = self.incr_state.gray_stack.pop().unwrap();
                self.mark_object(obj_addr);
                self.incr_state.objects_marked += 1;
            }
            self.incr_state.marking_in_progress = false;
            self.finish_incremental_cycle();
        } else {
            // No incremental cycle in progress. Do a full stop-the-world
            // mark-sweep using the same mark_object infrastructure.
            self.incr_state.gray_stack.clear();

            self.seed_major_roots();

            // Drain gray stack completely.
            while let Some(obj_addr) = self.incr_state.gray_stack.pop() {
                self.mark_object(obj_addr);
            }

            self.finish_incremental_cycle();
        }
    }

    /// Write barrier: call before storing a GC reference into an old-gen object.
    ///
    /// If the object has TRACK_YOUNG_PTRS set, we clear it and add the object
    /// to the remembered set, because it might now point to a young object.
    ///
    /// The `is_managed_heap_object` guard is NOT in incminimark — it
    /// compensates for virtualizable objects (PyFrame) being Rust
    /// Box-allocated instead of GC-managed. The JIT's COND_CALL_GC_WB
    /// fires for these objects because the rewriter emits WB for all
    /// SETFIELD_GC on Ref-typed values, and the flag-test reads garbage
    /// RPython's write_barrier() only tests the header flag and does not
    /// special-case non-managed addresses here.
    pub fn do_write_barrier(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
            unsafe { (*hdr).clear_flag(flags::TRACK_YOUNG_PTRS) };
            self.remembered_set.push(obj.0);
        }
    }

    /// incminimark.py:1495-1501 write_barrier_from_array:
    /// called by non-JIT code. Checks TRACK_YOUNG_PTRS, then dispatches
    /// to remember_young_pointer_from_array2 (card path) or
    /// remember_young_pointer (generic).
    pub fn do_write_barrier_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        if obj.is_null() {
            return;
        }
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
            self.remember_young_pointer_from_array2(obj, index, card_page_shift);
        }
    }

    /// incminimark.py:1557-1600 remember_young_pointer_from_array2:
    /// Called when TRACK_YOUNG_PTRS is set. If HAS_CARDS, marks the
    /// card WITHOUT clearing TRACK_YOUNG_PTRS. Otherwise falls back
    /// to the generic _remember_young_pointer_inlined.
    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            // No cards — fall back to generic barrier.
            // _remember_young_pointer_inlined clears TRACK_YOUNG_PTRS.
            unsafe { (*hdr).clear_flag(flags::TRACK_YOUNG_PTRS) };
            self.remembered_set.push(obj.0);
            return;
        }
        // Card path: mark the specific card. TRACK_YOUNG_PTRS stays set
        // so that subsequent writes still enter the barrier.
        self.mark_card(obj, index, card_page_shift);
    }

    /// incminimark.py:1606-1617 jit_remember_young_pointer_from_array:
    /// Minimal version called by the JIT when TRACK_YOUNG_PTRS is set
    /// but CARDS_SET is not. Tries to set CARDS_SET; otherwise falls
    /// back to remember_young_pointer (generic barrier).
    pub fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { (*hdr).has_flag(flags::HAS_CARDS) } {
            // incminimark.py:1614-1615
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(flags::CARDS_SET) };
        } else {
            // No cards — generic barrier.
            self.do_write_barrier(obj);
        }
    }

    /// incminimark.py:1574-1598: set the card bit for a given array index.
    /// Card bits are stored inline before the GcHeader. If CARDS_SET not
    /// yet set, add obj to `old_objects_with_cards_set` and set CARDS_SET.
    fn mark_card(&mut self, obj: GcRef, index: usize, card_page_shift: u32) {
        // incminimark.py:1576-1578
        let bitindex = index >> card_page_shift;
        let byteindex = bitindex >> 3;
        let bitmask: u8 = 1 << (bitindex & 7);

        // incminimark.py:1581-1584: if bit already set, return.
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        let byte = unsafe { *addr_byte };
        if byte & bitmask != 0 {
            return;
        }

        // incminimark.py:1594: set the bit.
        unsafe {
            *addr_byte = byte | bitmask;
        }

        // incminimark.py:1596-1598: if CARDS_SET not set, track and set.
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::CARDS_SET) } {
            self.old_objects_with_cards_set.push(obj.0);
            unsafe { (*hdr).set_flag(flags::CARDS_SET) };
        }
    }

    /// Check whether a specific card of an object is dirty.
    /// Reads inline card bytes before the GcHeader.
    pub fn is_card_dirty(&self, obj: GcRef, card_index: usize) -> bool {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            return false;
        }
        let byteindex = card_index >> 3;
        let bitmask: u8 = 1 << (card_index & 7);
        let addr_byte = Self::get_card_ptr(obj.0, byteindex);
        unsafe { *addr_byte & bitmask != 0 }
    }

    /// Return all dirty card indices for the given object.
    /// Reads inline card bytes before the GcHeader.
    pub fn dirty_cards(&self, obj: GcRef) -> Vec<usize> {
        let hdr = unsafe { header_of(obj.0) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            return Vec::new();
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        let length = unsafe { *((obj.0 + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        let mut result = Vec::new();
        for bi in 0..bytes {
            let addr = Self::get_card_ptr(obj.0, bi);
            let cardbyte = unsafe { *addr };
            for bit in 0..8 {
                if cardbyte & (1 << bit) != 0 {
                    result.push(bi * 8 + bit);
                }
            }
        }
        result
    }

    /// incminimark.py:2009-2083: collect_cardrefs_to_nursery.
    /// Process `old_objects_with_cards_set` — for each object, scan
    /// dirty card ranges and copy nursery objects out.
    fn collect_cardrefs_to_nursery(&mut self) {
        while !self.old_objects_with_cards_set.is_empty() {
            let obj = self.old_objects_with_cards_set.pop().unwrap();

            // incminimark.py:2016-2020
            let hdr = unsafe { header_of(obj) };
            debug_assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });
            debug_assert!(unsafe { (*hdr).has_flag(flags::CARDS_SET) });
            unsafe { (*hdr).clear_flag(flags::CARDS_SET) };

            // incminimark.py:2023-2026
            let type_id = unsafe { (*hdr).type_id() };
            let type_info = self.types.get(type_id);
            let length_offset = type_info.length_offset;
            let length = unsafe { *((obj + length_offset) as *const usize) };
            let bytes = self.card_marking_bytes_for_length(length);

            // incminimark.py:2033-2039: if !TRACK_YOUNG_PTRS, object is also
            // in old_objects_pointing_to_young and will be fully traced.
            // Just clear card bytes.
            if unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) } {
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
                continue;
            }

            // incminimark.py:2041-2072: walk card bytes, trace dirty ranges.
            if type_info.items_have_gc_ptrs && type_info.item_size > 0 {
                let item_size = type_info.item_size;
                let base_size = type_info.size;
                let items_start = obj + base_size;
                let card_page_indices = 1usize << self.card_page_shift;

                let mut interval_start = 0usize;
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    let mut cardbyte = unsafe { *p };
                    unsafe {
                        *p = 0;
                    } // reset
                    let next_byte_start = interval_start + 8 * card_page_indices;

                    while cardbyte != 0 {
                        let mut interval_stop = interval_start + card_page_indices;
                        if cardbyte & 1 != 0 {
                            if interval_stop > length {
                                interval_stop = length;
                                if interval_stop <= interval_start {
                                    break;
                                }
                            }
                            // trace_and_drag_out_of_nursery_partial
                            for i in interval_start..interval_stop {
                                let slot = (items_start + i * item_size) as *mut GcRef;
                                let field_ref = unsafe { *slot };
                                if !field_ref.is_null() && self.is_in_nursery(field_ref.0) {
                                    let new_ref = self.copy_nursery_object(field_ref.0);
                                    unsafe {
                                        *slot = new_ref;
                                    }
                                }
                            }
                        }
                        interval_start = interval_stop;
                        cardbyte >>= 1;
                    }
                    interval_start = next_byte_start;
                }
            } else {
                // No GC ptrs in items — just clear card bytes.
                for bi in 0..bytes {
                    let p = Self::get_card_ptr(obj, bi);
                    unsafe {
                        *p = 0;
                    }
                }
            }

            // incminimark.py:2079-2083: if incremental marking, re-add.
            if self.incr_state.marking_in_progress {
                unsafe { (*hdr).clear_flag(flags::VISITED) };
                self.incr_state.gray_stack.push(obj);
            }
        }
    }

    /// Clear inline card bytes for a given object.
    pub fn clear_cards(&mut self, obj_addr: usize) {
        let hdr = unsafe { header_of(obj_addr) };
        if unsafe { !(*hdr).has_flag(flags::HAS_CARDS) } {
            unsafe { (*hdr).clear_flag(flags::CARDS_SET) };
            return;
        }
        let type_id = unsafe { (*hdr).type_id() };
        let type_info = self.types.get(type_id);
        let length_offset = type_info.length_offset;
        let length = unsafe { *((obj_addr + length_offset) as *const usize) };
        let bytes = self.card_marking_bytes_for_length(length);
        for bi in 0..bytes {
            let p = Self::get_card_ptr(obj_addr, bi);
            unsafe {
                *p = 0;
            }
        }
        unsafe { (*hdr).clear_flag(flags::CARDS_SET) };
    }

    // ── JIT integration hooks ──

    /// Fast-path write barrier for JIT-compiled code.
    ///
    /// Called from JIT-compiled code when a write barrier fires.
    /// Adds the object directly to the remembered set without the
    /// full flag-check logic of `do_write_barrier()`, because the
    /// JIT has already determined that the barrier is needed (via
    /// the inline flag test emitted by COND_CALL_GC_WB).
    ///
    /// Equivalent to incminimark's `jit_remember_young_pointer()`.
    ///
    /// See `do_write_barrier` for why `is_managed_heap_object` is needed.
    pub fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        if obj.is_null() || !self.is_managed_heap_object(obj.0) {
            return;
        }
        // Clear TRACK_YOUNG_PTRS so the object won't trigger the
        // barrier again until it is re-processed in a minor collection.
        let hdr = unsafe { header_of(obj.0) };
        unsafe { (*hdr).clear_flag(flags::TRACK_YOUNG_PTRS) };
        self.remembered_set.push(obj.0);
    }

    /// Returns true if the GC supports optimized conditional write barriers.
    ///
    /// When true, the JIT can emit COND_CALL_GC_WB (an inline flag test +
    /// conditional call) instead of a full write-barrier call. Nursery-based
    /// collectors always support this because the barrier check is a simple
    /// flag test on the object header.
    pub fn can_optimize_cond_call(&self) -> bool {
        true
    }

    /// Perform one incremental GC step. Called from JIT safepoints.
    ///
    /// If an incremental marking cycle should start, it is initiated.
    /// If a cycle is already in progress, one bounded marking step is
    /// performed. Returns true if any GC work was done.
    pub fn gc_step(&mut self) -> bool {
        if self.should_start_major_cycle() && !self.incr_state.marking_in_progress {
            self.start_incremental_cycle();
            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
            }
            true
        } else if self.incr_state.marking_in_progress {
            let done = self.incremental_mark_step();
            if done {
                self.finish_incremental_cycle();
            }
            true
        } else {
            false
        }
    }

    /// Reset the nursery while preserving pinned objects.
    ///
    /// Saves pinned object data, zeroes the nursery, restores pinned objects,
    /// and sets the free pointer past the highest pinned object.
    fn reset_nursery_with_pinned(&mut self) {
        let nursery_start = self.nursery.start_ptr() as usize;

        // Collect (header_start, total_size, data) for each pinned object.
        let mut saved: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        for &obj_addr in &self.pinned_objects {
            let type_id = unsafe { (*header_of(obj_addr)).type_id() };
            let type_info = self.types.get(type_id);
            let payload_size = if type_info.item_size > 0 {
                let length = unsafe { *((obj_addr + type_info.length_offset) as *const usize) };
                type_info.total_instance_size(length)
            } else {
                type_info.size
            };
            let total_size = (GcHeader::SIZE + payload_size).max(GcHeader::MIN_NURSERY_OBJ_SIZE);
            let total_size = (total_size + 7) & !7;
            let header_start = obj_addr - GcHeader::SIZE;
            let data = unsafe {
                std::slice::from_raw_parts(header_start as *const u8, total_size).to_vec()
            };
            saved.push((header_start, total_size, data));
        }

        // Zero-fill the entire nursery.
        self.nursery.reset();

        // Restore pinned objects and compute the highest end.
        let mut max_end = nursery_start;
        for (header_start, total_size, data) in &saved {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), *header_start as *mut u8, *total_size);
            }
            let end = header_start + total_size;
            if end > max_end {
                max_end = end;
            }
        }

        // Set free pointer past the highest pinned object so new allocations
        // don't overwrite it.
        if max_end > nursery_start {
            unsafe {
                self.nursery.set_free_ptr(max_end as *mut u8);
            }
        }
    }

    /// Pin a nursery object so it won't be moved during minor collection.
    /// Sets the PINNED flag in the object header and records the address.
    /// Returns true if pinning succeeded, false if the object is null or
    /// not in the nursery.
    pub fn pin(&mut self, obj: GcRef) -> bool {
        if obj.is_null() || !self.is_nursery_object_start(obj.0) {
            return false;
        }
        unsafe {
            (*header_of(obj.0)).set_flag(flags::PINNED);
        }
        self.pinned_objects.insert(obj.0);
        true
    }

    /// Unpin a previously pinned object.
    pub fn unpin(&mut self, obj: GcRef) {
        if obj.is_null() {
            return;
        }
        // Clear the header flag if the object is still in the nursery.
        if self.is_nursery_object_start(obj.0) {
            unsafe {
                (*header_of(obj.0)).clear_flag(flags::PINNED);
            }
        }
        self.pinned_objects.remove(&obj.0);
    }

    /// Check if an object is currently pinned.
    pub fn is_pinned(&self, obj: GcRef) -> bool {
        self.pinned_objects.contains(&obj.0)
    }

    /// Free memory associated with invalidated JIT compiled code.
    ///
    /// `code_ptr` and `size` identify the compiled code region to release.
    /// The region is looked up and removed from the compiled code registry
    /// so the GC no longer scans it for root references.
    pub fn jit_free(&mut self, code_ptr: usize, size: usize) {
        // Find and remove any compiled code region that matches the given range.
        self.compiled_code_registry
            .regions
            .retain(|r| !(r.code_start == code_ptr && r.code_size == size));
    }

    /// Number of objects in the remembered set (for testing / diagnostics).
    pub fn remembered_set_len(&self) -> usize {
        self.remembered_set.len()
    }
}

/// Safepoint GC map: records which frame slots contain GC references
/// at a specific program point (guard or call site).
///
/// The Cranelift backend builds these during compilation and stores them
/// alongside the compiled code. During collection, the GC uses them to
/// find live references on the stack.
#[derive(Debug, Clone)]
pub struct SafepointMap {
    /// Map from code offset to GcMap.
    pub entries: Vec<SafepointEntry>,
}

/// A single safepoint entry.
#[derive(Debug, Clone)]
pub struct SafepointEntry {
    /// Offset in the compiled code (bytes from function start).
    pub code_offset: u32,
    /// Bitmap of which frame slots contain GC references.
    pub gc_map: crate::GcMap,
}

impl SafepointMap {
    pub fn new() -> Self {
        SafepointMap {
            entries: Vec::new(),
        }
    }

    /// Add a safepoint entry.
    pub fn add(&mut self, code_offset: u32, gc_map: crate::GcMap) {
        self.entries.push(SafepointEntry {
            code_offset,
            gc_map,
        });
    }

    /// Look up the GcMap for a given code offset.
    pub fn lookup(&self, code_offset: u32) -> Option<&crate::GcMap> {
        self.entries
            .iter()
            .find(|e| e.code_offset == code_offset)
            .map(|e| &e.gc_map)
    }
}

impl Default for SafepointMap {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry of compiled code regions and their safepoint maps.
///
/// When the GC needs to scan the stack during collection, it uses the return
/// address to find which compiled code region is active, then looks up the
/// safepoint map to determine which frame slots contain GC references.
///
/// From rpython/jit/backend/llsupport/gc.py GcRootMap_asmgcc / GcRootMap_shadowstack.
pub struct CompiledCodeRegistry {
    /// Compiled code regions, sorted by start address for binary search.
    regions: Vec<CompiledCodeRegion>,
}

/// A single compiled code region with its safepoint map.
#[derive(Debug, Clone)]
pub struct CompiledCodeRegion {
    /// Start address of the compiled code.
    pub code_start: usize,
    /// Size of the compiled code in bytes.
    pub code_size: usize,
    /// Safepoint map for this region.
    pub safepoint_map: SafepointMap,
    /// Frame size in slots (each slot = 8 bytes).
    pub frame_size_slots: u32,
    /// JitCellToken number for identification.
    pub loop_token: u64,
}

impl CompiledCodeRegistry {
    pub fn new() -> Self {
        CompiledCodeRegistry {
            regions: Vec::new(),
        }
    }

    /// Register a compiled code region.
    pub fn register(&mut self, region: CompiledCodeRegion) {
        self.regions.push(region);
        // Keep sorted by code_start for binary search
        self.regions.sort_by_key(|r| r.code_start);
    }

    /// Unregister a compiled code region (e.g., when invalidating a loop).
    pub fn unregister(&mut self, loop_token: u64) {
        self.regions.retain(|r| r.loop_token != loop_token);
    }

    /// Look up a compiled code region containing the given return address.
    ///
    /// Returns the region and the offset within it.
    pub fn find_region(&self, return_addr: usize) -> Option<(&CompiledCodeRegion, u32)> {
        // Binary search for the region containing this address
        let idx = self
            .regions
            .binary_search_by(|r| {
                if return_addr < r.code_start {
                    std::cmp::Ordering::Greater
                } else if return_addr >= r.code_start + r.code_size {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Equal
                }
            })
            .ok()?;

        let region = &self.regions[idx];
        let offset = (return_addr - region.code_start) as u32;
        Some((region, offset))
    }

    /// Scan a compiled frame for GC references using the safepoint map.
    ///
    /// Given a return address (from the call stack) and the frame base pointer,
    /// enumerates all frame slots that contain GC references.
    ///
    /// # Safety
    /// `frame_base` must point to a valid JIT frame with at least
    /// `region.frame_size_slots` slots.
    pub unsafe fn scan_frame(
        &self,
        return_addr: usize,
        frame_base: *const usize,
    ) -> Vec<*mut GcRef> {
        let mut roots = Vec::new();

        let (region, offset) = match self.find_region(return_addr) {
            Some(r) => r,
            None => return roots,
        };

        let gc_map = match region.safepoint_map.lookup(offset) {
            Some(map) => map,
            None => return roots,
        };

        // Enumerate all slots marked as GC references
        for word_idx in 0..gc_map.ref_bitmap.len() {
            let mut bits = gc_map.ref_bitmap[word_idx];
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit;

                if slot_idx < region.frame_size_slots as usize {
                    let slot_ptr = unsafe { frame_base.add(slot_idx) } as *mut GcRef;
                    roots.push(slot_ptr);
                }

                bits &= bits - 1; // Clear lowest set bit
            }
        }

        roots
    }

    /// Number of registered regions.
    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }
}

impl Default for CompiledCodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for MiniMarkGC {
    fn default() -> Self {
        Self::new()
    }
}

impl GcAllocator for MiniMarkGC {
    fn alloc_nursery(&mut self, size: usize) -> GcRef {
        self.alloc_with_type(0, size)
    }

    fn alloc_nursery_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type(type_id, size)
    }

    fn alloc_nursery_no_collect(&mut self, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(0, size)
    }

    fn alloc_nursery_no_collect_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        self.alloc_with_type_no_collect(type_id, size)
    }

    fn alloc_varsize(&mut self, base_size: usize, item_size: usize, length: usize) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type(0, payload_size)
    }

    fn alloc_varsize_typed(
        &mut self,
        type_id: u32,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type(type_id, payload_size)
    }

    fn alloc_varsize_no_collect(
        &mut self,
        base_size: usize,
        item_size: usize,
        length: usize,
    ) -> GcRef {
        let payload_size = base_size + item_size * length;
        self.alloc_with_type_no_collect(0, payload_size)
    }

    fn alloc_oldgen_typed(&mut self, type_id: u32, size: usize) -> GcRef {
        let total_size = GcHeader::SIZE + size;
        self.alloc_in_oldgen(type_id, total_size)
    }

    fn is_managed_heap_object(&self, addr: usize) -> bool {
        self.nursery.contains(addr) || self.oldgen.contains(addr)
    }

    fn write_barrier(&mut self, obj: GcRef) {
        self.do_write_barrier(obj);
    }

    fn jit_remember_young_pointer_from_array(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer_from_array(obj);
    }

    fn remember_young_pointer_from_array2(
        &mut self,
        obj: GcRef,
        index: usize,
        card_page_shift: u32,
    ) {
        self.remember_young_pointer_from_array2(obj, index, card_page_shift);
    }

    fn collect_nursery(&mut self) {
        self.do_collect_nursery();
    }

    fn collect_full(&mut self) {
        self.do_collect_full();
    }

    unsafe fn add_root(&mut self, root: *mut GcRef) {
        unsafe { self.roots.add(root) };
    }

    fn remove_root(&mut self, root: *mut GcRef) {
        self.roots.remove(root);
    }

    fn nursery_free(&self) -> *mut u8 {
        self.nursery.free_ptr()
    }

    fn nursery_free_addr(&self) -> usize {
        self.nursery.free_addr()
    }

    fn nursery_top(&self) -> *const u8 {
        self.nursery.top_ptr()
    }

    fn nursery_top_addr(&self) -> usize {
        self.nursery.top_addr()
    }

    fn max_nursery_object_size(&self) -> usize {
        self.config.large_object_threshold
    }

    fn card_page_shift(&self) -> u32 {
        self.card_page_shift
    }

    fn jit_remember_young_pointer(&mut self, obj: GcRef) {
        self.jit_remember_young_pointer(obj);
    }

    fn can_optimize_cond_call(&self) -> bool {
        self.can_optimize_cond_call()
    }

    fn gc_step(&mut self) -> bool {
        self.gc_step()
    }

    fn jit_free(&mut self, code_ptr: usize, size: usize) {
        self.jit_free(code_ptr, size);
    }

    fn pin(&mut self, obj: GcRef) -> bool {
        self.pin(obj)
    }

    fn unpin(&mut self, obj: GcRef) {
        self.unpin(obj);
    }

    fn is_pinned(&self, obj: GcRef) -> bool {
        self.is_pinned(obj)
    }

    fn register_type(&mut self, info: TypeInfo) -> u32 {
        self.types.register(info)
    }

    fn type_count(&self) -> usize {
        self.types.len()
    }

    fn type_size(&self, type_id: u32) -> Option<usize> {
        if (type_id as usize) < self.types.len() {
            Some(self.types.get(type_id).size)
        } else {
            None
        }
    }

    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        // gc.py:563-590 GcLLDescr_framework
        //   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
        // RPython derives the typeid arithmetically:
        //   expected_typeid = classptr - sizeof_ti - type_info_group
        // pyre keeps an explicit vtable→type_id table populated via
        // register_vtable_for_type, mirroring the same contract.
        self.vtable_to_type_id.get(&classptr).copied()
    }

    fn register_vtable_for_type(&mut self, vtable: usize, type_id: u32) {
        self.vtable_to_type_id.insert(vtable, type_id);
    }

    /// gctypelayout.py:393-398 `encode_type_shapes_now` parity:
    /// closes the type-registration phase on the underlying
    /// `TypeRegistry`. The materialized `type_info_group` base
    /// address is stable from this point on, and every is_object
    /// type's preorder `subclassrange_{min,max}` is computed via
    /// `assign_inheritance_ids` (normalizecalls.py:373-389).
    fn freeze_types(&mut self) {
        self.types.freeze_types();
    }

    /// gc.py:318 `GcLLDescr_framework.supports_guard_gc_type = True`.
    /// MiniMarkGC owns a `TypeRegistry` that materializes a `TYPE_INFO`
    /// table on demand, so the framework-equivalent flag is always
    /// true here. (Boehm-equivalent stubs override the trait default
    /// and leave it `false`.)
    fn supports_guard_gc_type(&self) -> bool {
        true
    }

    /// gc.py:631-642 `check_is_object` parity.
    ///
    /// Implemented line-by-line as the RPython source:
    ///     typeid = self.get_actual_typeid(gcptr)
    ///     base_type_info, shift_by, sizeof_ti = (
    ///         self.get_translated_info_for_typeinfo())
    ///     infobits_offset, IS_OBJECT_FLAG = (
    ///         self.get_translated_info_for_guard_is_object())
    ///     p = base_type_info + (typeid << shift_by) + infobits_offset
    ///     p = rffi.cast(rffi.CCHARP, p)
    ///     return (ord(p[0]) & IS_OBJECT_FLAG) != 0
    ///
    /// `get_actual_typeid` is the only majit-specific seam: managed
    /// objects carry a `GcHeader` immediately before the payload, while
    /// pyre's host-allocated foreign objects keep their classptr at
    /// offset 0 and rely on the `vtable→type_id` table populated via
    /// `register_vtable_for_type`.
    fn check_is_object(&self, gcref: GcRef) -> bool {
        if gcref.is_null() {
            return false;
        }
        // gc.py:634: typeid = self.get_actual_typeid(gcptr).
        let Some(typeid) = self.get_actual_typeid(gcref) else {
            return false;
        };
        // gc.py:636-637: gc_ll_descr lookups.
        let (base_type_info, shift_by, _sizeof_ti) = self.get_translated_info_for_typeinfo();
        let (infobits_offset, is_object_flag) = self.get_translated_info_for_guard_is_object();
        // gc.py:640-642: addr arithmetic + byte test.
        let typeid = typeid as usize;
        if typeid >= self.types.len() {
            return false;
        }
        let p = base_type_info + (typeid << shift_by) + infobits_offset;
        let byte = unsafe { *(p as *const u8) };
        (byte & is_object_flag) != 0
    }

    /// gc.py:592 `get_translated_info_for_typeinfo` parity.
    /// Returns `(type_info_group_base, shift_by, sizeof_ti)` — the
    /// base address of the materialized `type_info_group` table, the
    /// SIB-style scale `genop_guard_guard_is_object` /
    /// `genop_guard_guard_subclass` apply to the typeid register
    /// (x86/assembler.py:1934, 1967), and `rffi.sizeof(TYPE_INFO)`.
    ///
    /// RPython's 64-bit port sets `shift_by = 0` because its typeid
    /// is already `GROUP_MEMBER_OFFSET`, i.e. the raw byte offset of
    /// the member inside the group struct
    /// (translator/c/src/llgroup.h:36). majit keeps typeid as a
    /// small-integer index (returned by `register_type`) and encodes
    /// the per-entry stride here: `shift_by = log2(TypeEntry::STRIDE)`
    /// so the backend formula
    /// `base + (typeid << shift_by) + offset` lands on the correct
    /// `TypeEntry[typeid]` field. `sizeof_ti` stays equal to
    /// `rffi.sizeof(TYPE_INFO)` because that is the distance to the
    /// paired `CLASSTYPE` entry the `genop_guard_guard_subclass`
    /// formula needs (`+ sizeof_ti + offset2`
    /// x86/assembler.py:1968-1969, gctypelayout.py:359-374
    /// `add_vtable_after_typeinfo`).
    fn get_translated_info_for_typeinfo(&self) -> (usize, u8, usize) {
        let table = self.types.type_info_table();
        let base = table.as_ptr() as usize;
        (base, TypeEntry::SHIFT_BY, TypeInfoLayout::SIZE_OF_TI)
    }

    /// gc.py:619-622 `get_translated_info_for_guard_is_object` parity.
    /// Reads the instance state populated by `_setup_guard_is_object`
    /// and returns `(infobits_offset + infobits_offset_plus,
    /// T_IS_RPYTHON_INSTANCE_BYTE)`. Mirrors:
    ///
    /// ```python
    /// def get_translated_info_for_guard_is_object(self):
    ///     infobits_offset = rffi.cast(lltype.Signed, self._infobits_offset)
    ///     infobits_offset += self._infobits_offset_plus
    ///     return (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    /// ```
    fn get_translated_info_for_guard_is_object(&self) -> (usize, u8) {
        let infobits_offset = self._infobits_offset + self._infobits_offset_plus;
        (infobits_offset, self._T_IS_RPYTHON_INSTANCE_BYTE)
    }

    /// x86/assembler.py:1951 `cpu.subclassrange_min_offset` parity.
    /// Byte offset of `subclassrange_min` inside `ClassTypeLayout`,
    /// the paired `rclass.CLASSTYPE` member that immediately follows
    /// the `TYPE_INFO` entry in the type_info_group
    /// (gctypelayout.py:359-374).
    fn subclassrange_min_offset(&self) -> usize {
        ClassTypeLayout::SUBCLASSRANGE_MIN_OFFSET
    }

    /// x86/assembler.py:1971-1974 codegen-time bounds lookup parity.
    /// Resolves `classptr` through the registered vtable→typeid map and
    /// returns the type's `(subclassrange_min, subclassrange_max)`.
    fn subclass_range(&self, classptr: usize) -> Option<(i64, i64)> {
        let type_id = self.vtable_to_type_id.get(&classptr).copied()?;
        let info = self.types.get(type_id);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// Companion to `subclass_range` keyed by typeid. Used by the
    /// executor's `GuardSubclass` arm after it resolves `value.typeptr`
    /// via `get_actual_typeid` (llgraph/runner.py:1271-1281).
    fn typeid_subclass_range(&self, typeid: u32) -> Option<(i64, i64)> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        let info = self.types.get(typeid);
        Some((info.subclassrange_min, info.subclassrange_max))
    }

    /// gc.py:624-629 `get_actual_typeid` parity.
    ///
    /// RPython reads the typeid from the GC header word
    /// (`llop.extract_ushort(llgroup.HALFWORD, hdr.tid)`). MiniMarkGC's
    /// managed objects use the same layout — `GcHeader::SIZE` bytes
    /// before the payload, type id in the lower `TYPE_ID_BITS`. pyre's
    /// host-allocated "foreign" objects do not carry a `GcHeader`; for
    /// those we recover the type id by reading the classptr (offset 0)
    /// and looking it up in `vtable_to_type_id`, which is what
    /// `register_vtable_for_type` populated.
    fn get_actual_typeid(&self, gcref: GcRef) -> Option<u32> {
        if gcref.is_null() {
            return None;
        }
        if self.is_managed_heap_object(gcref.0) {
            let header_addr = gcref.0.wrapping_sub(crate::header::GcHeader::SIZE);
            let header: crate::header::GcHeader = unsafe { *(header_addr as *const _) };
            return Some(header.type_id());
        }
        // Foreign object — read classptr at offset 0 and consult the
        // vtable→type_id table populated via register_vtable_for_type.
        let vtable = unsafe { *(gcref.0 as *const usize) };
        self.vtable_to_type_id.get(&vtable).copied()
    }

    /// Companion to `check_is_object` keyed by typeid. Reads the
    /// `T_IS_RPYTHON_INSTANCE` bit directly from the materialized
    /// TYPE_INFO entry (gctypelayout.py:642).
    fn typeid_is_object(&self, typeid: u32) -> Option<bool> {
        if (typeid as usize) >= self.types.len() {
            return None;
        }
        Some(self.types.get(typeid).is_object)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static SHADOW_STACK_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Helper: create a GC with a small nursery for testing.
    fn test_gc(nursery_size: usize) -> MiniMarkGC {
        MiniMarkGC::with_config(GcConfig {
            nursery_size,
            large_object_threshold: nursery_size / 2,
            ..GcConfig::default()
        })
    }

    #[test]
    fn test_basic_alloc() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));
        let obj = gc.alloc_with_type(0, 16);
        assert!(!obj.is_null());
        assert!(gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_multiple_allocs() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let mut refs = Vec::new();
        for _ in 0..10 {
            refs.push(gc.alloc_with_type(0, 16));
        }

        // All should be non-null and distinct.
        for i in 0..refs.len() {
            assert!(!refs[i].is_null());
            for j in (i + 1)..refs.len() {
                assert_ne!(refs[i], refs[j]);
            }
        }
    }

    #[test]
    fn test_large_object_goes_to_oldgen() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(1024));

        // 1024 > large_object_threshold (512), so goes to old gen.
        let obj = gc.alloc_with_type(0, 1024);
        assert!(!obj.is_null());
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_nursery_collection_basic() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an object and root it.
        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write something to the object payload.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEADBEEF;
        }

        // Root it.
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // The root should now point to old gen.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The data should be preserved.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEADBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_unrooted_object_dies() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate but don't root.
        let _obj = gc.alloc_with_type(0, 16);

        // Collection should run without issues.
        gc.collect_nursery();

        // Nursery is reset, the object is gone.
        assert_eq!(gc.nursery.used(), 0);
    }

    #[test]
    fn test_fill_nursery_triggers_collection() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Keep allocating until we must have triggered at least one collection.
        for _ in 0..100 {
            gc.alloc_with_type(0, 16);
        }
        assert!(gc.minor_collections > 0);
    }

    #[test]
    fn test_write_barrier() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(16));

        // Allocate a large object (goes to old gen).
        let old_obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(old_obj.0));

        // The old object should have TRACK_YOUNG_PTRS.
        let hdr = unsafe { header_of(old_obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Write barrier clears the flag and adds to remembered set.
        gc.do_write_barrier(old_obj);
        assert!(unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert_eq!(gc.remembered_set.len(), 1);

        // Second call: flag already cleared, should not add again.
        gc.do_write_barrier(old_obj);
        assert_eq!(gc.remembered_set.len(), 1);
    }

    #[test]
    fn test_nursery_collection_with_pointers() {
        // Object layout: one GcRef field at offset 0 (payload = 8 bytes).
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create two objects: parent -> child.
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let parent = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());

        // Write child's address into parent's first field.
        unsafe {
            *(parent.0 as *mut GcRef) = child;
        }

        // Root only the parent.
        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger collection.
        gc.collect_nursery();

        // Parent should have survived.
        assert!(!gc.is_in_nursery(root.0));
        assert!(!root.is_null());

        // The pointer field should now point to the child's new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_forwarding_dedup() {
        // Two roots pointing to the same nursery object should get the
        // same forwarded address.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let shared = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        let mut root1 = shared;
        let mut root2 = shared;

        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }

        gc.collect_nursery();

        // Both roots should point to the same old-gen location.
        assert_eq!(root1, root2);
        assert!(!gc.is_in_nursery(root1.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Allocate some objects and root one.
        let obj1 = gc.alloc_with_type(tid, 16);
        let _obj2 = gc.alloc_with_type(tid, 16); // unreachable

        let mut root = obj1;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Full collection: promotes to old gen and sweeps.
        gc.collect_full();

        assert!(!root.is_null());
        assert!(!gc.is_in_nursery(root.0));

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_frees_unreachable_old_objects() {
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote two objects to old gen.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));
        assert_eq!(gc.oldgen.object_count(), 2);

        // Now unroot obj2 and do a full collection.
        gc.roots.remove(&mut root2);
        gc.collect_full();

        // Only obj1 should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root1.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_repeated_collections() {
        let mut gc = test_gc(512);
        let tid = gc.register_type(TypeInfo::simple(16));

        let mut root = GcRef::NULL;
        unsafe {
            gc.roots.add(&mut root);
        }

        for i in 0..50 {
            let obj = gc.alloc_with_type(tid, 16);
            // Write a marker value.
            unsafe {
                *(obj.0 as *mut u64) = i as u64;
            }
            root = obj;

            if i % 10 == 0 {
                gc.collect_nursery();
                // Root should survive and preserve its value.
                if !root.is_null() {
                    let val = unsafe { *(root.0 as *const u64) };
                    assert_eq!(val, i as u64);
                }
            }
        }

        gc.roots.clear();
    }

    #[test]
    fn test_gc_allocator_trait() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_nursery(32);
        assert!(!obj.is_null());

        let varobj = gc.alloc_varsize(8, 4, 10);
        assert!(!varobj.is_null());

        gc.collect_nursery();
        gc.collect_full();
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// pyre's GC stores an explicit vtable→type_id table; verify that
    /// register_vtable_for_type / get_typeid_from_classptr_if_gcremovetypeptr
    /// round-trip via the GcAllocator trait.
    #[test]
    fn test_gcremovetypeptr_vtable_lookup() {
        let mut gc = test_gc(4096);
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let float_tid = gc.register_type(TypeInfo::simple(16));

        // Register two distinct vtables
        let int_vtable: usize = 0x1234_5670;
        let float_vtable: usize = 0x1234_5680;
        crate::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);
        crate::GcAllocator::register_vtable_for_type(&mut gc, float_vtable, float_tid);

        // Round-trip
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, int_vtable),
            Some(int_tid)
        );
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, float_vtable),
            Some(float_tid)
        );
        // Unknown classptr → None
        assert_eq!(
            crate::GcAllocator::get_typeid_from_classptr_if_gcremovetypeptr(&gc, 0xCAFEBABE),
            None
        );
    }

    #[test]
    fn test_alloc_nursery_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(24));

        let obj1 = gc.alloc_nursery_no_collect(24);
        let obj2 = gc.alloc_nursery_no_collect(24);

        assert!(!obj1.is_null());
        assert!(!obj2.is_null());
        assert_eq!(gc.minor_collections, 0);
        // The second allocation may have fallen back to old gen, but it must
        // still succeed without forcing a collection.
        assert_ne!(obj1, obj2);
    }

    #[test]
    fn test_alloc_varsize_no_collect_does_not_trigger_collection() {
        let mut gc = test_gc(64);
        gc.register_type(TypeInfo::simple(32));

        let obj = gc.alloc_varsize_no_collect(16, 8, 8);

        assert!(!obj.is_null());
        assert_eq!(gc.minor_collections, 0);
        // This request is too large for the tiny nursery, so no-collect mode
        // must have used the old generation instead.
        assert!(!gc.is_in_nursery(obj.0));
    }

    #[test]
    fn test_write_barrier_with_collection() {
        // Scenario: old object points to young object, write barrier ensures
        // the young object survives collection.
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen object.
        let old_obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());

        // Create a young object.
        let young_obj = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(young_obj.0 as *mut u64) = 0x42424242;
        }

        // Store young ref into old object's field.
        unsafe {
            *(old_obj.0 as *mut GcRef) = young_obj;
        }
        // Write barrier.
        gc.do_write_barrier(old_obj);

        // Root only the old object.
        let mut root = old_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collect.
        gc.collect_nursery();

        // The old object's field should be updated to the new location.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0x42424242);

        gc.roots.clear();
    }

    #[test]
    fn test_null_ref_safety() {
        let mut gc = test_gc(1024);
        gc.register_type(TypeInfo::simple(8));

        // Write barrier on null should be a no-op.
        gc.do_write_barrier(GcRef::NULL);
        assert!(gc.remembered_set.is_empty());
    }

    #[test]
    fn test_chain_of_pointers() {
        // Test a chain: root -> A -> B -> C, all in nursery.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        let c = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let b = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(b.0 as *mut GcRef) = c;
        }

        let a = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let mut root = a;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify the entire chain survived.
        assert!(!gc.is_in_nursery(root.0));
        let new_b = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_b.0));
        assert!(!new_b.is_null());
        let new_c = unsafe { *(new_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(new_c.0));
        assert!(!new_c.is_null());
        let tail = unsafe { *(new_c.0 as *const GcRef) };
        assert!(tail.is_null());

        gc.roots.clear();
    }

    #[test]
    fn test_major_collection_with_graph() {
        // Test major collection with a graph: root -> A -> B, root -> C (unreachable D).
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Type with one pointer field.
        let tid1 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));
        // Type with two pointer fields.
        let tid2 = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        let b = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(b.0 as *mut GcRef) = GcRef::NULL;
        }

        let a = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(a.0 as *mut GcRef) = b;
        }

        let c = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(c.0 as *mut GcRef) = GcRef::NULL;
        }

        let d = gc.alloc_with_type(tid1, ptr_size);
        unsafe {
            *(d.0 as *mut GcRef) = GcRef::NULL;
        }
        let _ = d; // unreachable

        // Root object points to both A and C.
        let root_obj = gc.alloc_with_type(tid2, ptr_size * 2);
        unsafe {
            *(root_obj.0 as *mut GcRef) = a;
            *((root_obj.0 + ptr_size) as *mut GcRef) = c;
        }

        let mut root = root_obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote to old gen.
        gc.collect_nursery();

        // Now do full collection. A, B, C should survive; D should be freed.
        // We have 5 objects in old gen (root, A, B, C, D).
        // Wait, D was unreachable from root, so it wouldn't have been promoted.
        // Actually it was in the nursery unreachable, so it was wiped on nursery reset.
        assert_eq!(gc.oldgen.object_count(), 4); // root, A, B, C

        gc.collect_full();

        // All 4 are reachable from the root, so all survive.
        assert_eq!(gc.oldgen.object_count(), 4);

        gc.roots.clear();
    }

    #[test]
    fn major_mark_skips_unmanaged_field_target() {
        // Stepping-stone parity: an old-gen object's `gc_ptr_offsets`
        // field may transiently point at a `std::alloc`-backed
        // (non-GC) block during the L1/L2 partial migration window.
        // The major-mark walker must skip such fields rather than
        // dereferencing memory before the unmanaged block as a
        // `GcHeader`. This test catches regression of the
        // `is_managed_heap_object` guard in `mark_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Allocate parent in old-gen. Field pointer will be set to a
        // raw `Box::into_raw` block — what `std::alloc` does for an
        // unmigrated `wrappeditems` / `items` slot today.
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
        let unmanaged = Box::into_raw(Box::new(0xCAFEBABEu64));
        unsafe {
            *(parent.0 as *mut GcRef) = GcRef(unmanaged as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Major collection — without the guard this would crash on
        // `header_of(unmanaged)` reading invalid memory.
        gc.collect_full();

        // Parent survived because it is rooted; the unmanaged target
        // was correctly skipped.
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged));
        }
    }

    #[test]
    fn major_mark_skips_unmanaged_varsize_item_target() {
        // Same stepping-stone scenario, varsize variant: an
        // `items_have_gc_ptrs = true` array can hold transient
        // unmanaged pointers during partial migration. The mark
        // walker's variable-part loop must also guard with
        // `is_managed_heap_object`.
        let ptr_size = std::mem::size_of::<GcRef>();
        let length_offset = 0;
        let base_size = 8; // length field (Signed)
        let length = 2usize;

        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::varsize(
            base_size,
            ptr_size,
            length_offset,
            true,
            Vec::new(),
        ));

        let total_payload = base_size + ptr_size * length;
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + total_payload);
        unsafe {
            *(parent.0 as *mut usize) = length;
        }
        let unmanaged_a = Box::into_raw(Box::new(0xDEADBEEFu64));
        let unmanaged_b = Box::into_raw(Box::new(0xFEEDFACEu64));
        unsafe {
            *((parent.0 + base_size) as *mut GcRef) = GcRef(unmanaged_a as usize);
            *((parent.0 + base_size + ptr_size) as *mut GcRef) = GcRef(unmanaged_b as usize);
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_full();
        assert_eq!(gc.oldgen.object_count(), 1);

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(unmanaged_a));
            drop(Box::from_raw(unmanaged_b));
        }
    }

    #[test]
    fn test_data_integrity_across_collections() {
        // Allocate objects with distinctive data, collect, verify data.
        let mut gc = test_gc(2048);
        // Type: 32 bytes payload, one GcRef at offset 0, then 24 bytes of data.
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(32, vec![0]));

        let child = gc.alloc_with_type(tid, 32);
        unsafe {
            *(child.0 as *mut GcRef) = GcRef::NULL;
            *((child.0 + 8) as *mut u64) = 0xAAAA_BBBB_CCCC_DDDD;
            *((child.0 + 16) as *mut u64) = 0x1111_2222_3333_4444;
            *((child.0 + 24) as *mut u64) = 0x5555_6666_7777_8888;
        }

        let parent = gc.alloc_with_type(tid, 32);
        unsafe {
            *(parent.0 as *mut GcRef) = child;
            *((parent.0 + 8) as *mut u64) = 0xCAFE_BABE_DEAD_BEEF;
        }

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // Verify parent data.
        let pdata = unsafe { *((root.0 + 8) as *const u64) };
        assert_eq!(pdata, 0xCAFE_BABE_DEAD_BEEF);

        // Verify child data.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!child_ref.is_null());
        let c1 = unsafe { *((child_ref.0 + 8) as *const u64) };
        let c2 = unsafe { *((child_ref.0 + 16) as *const u64) };
        let c3 = unsafe { *((child_ref.0 + 24) as *const u64) };
        assert_eq!(c1, 0xAAAA_BBBB_CCCC_DDDD);
        assert_eq!(c2, 0x1111_2222_3333_4444);
        assert_eq!(c3, 0x5555_6666_7777_8888);

        gc.roots.clear();
    }

    // ── Card marking tests ──
    // incminimark.py card bytes layout: [card_bytes...][GcHeader][payload]

    /// Helper: register a varsize array type and allocate in old gen with
    /// inline card bytes. Returns (obj, array_length).
    fn alloc_card_array(gc: &mut MiniMarkGC, array_length: usize) -> (GcRef, usize) {
        let ptr_size = std::mem::size_of::<GcRef>();
        let arr_tid = gc.register_type(TypeInfo::varsize(8, ptr_size, 0, true, Vec::new()));
        let total_size = GcHeader::SIZE + 8 + ptr_size * array_length;
        let obj = gc.alloc_in_oldgen_with_cards(arr_tid, total_size, array_length, true);
        // Write the length field.
        unsafe {
            *(obj.0 as *mut usize) = array_length;
        }
        (obj, array_length)
    }

    #[test]
    fn test_card_marking_basic() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // Card-marking write barrier: mark card for index 5.
        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // The card at index 5 >> 7 = 0 should be dirty.
        assert!(
            gc.is_card_dirty(obj, 0),
            "card 0 should be dirty after writing index 5"
        );
        assert!(
            unsafe { (*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET flag should be set"
        );

        // Mark another index in a different card.
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        let card_idx = 200 >> DEFAULT_CARD_PAGE_SHIFT;
        assert!(
            gc.is_card_dirty(obj, card_idx as usize),
            "card for index 200 should be dirty"
        );
    }

    #[test]
    fn test_card_marking_clear_after_collection() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // Mark some cards.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 200, DEFAULT_CARD_PAGE_SHIFT);
        assert!(
            unsafe { (*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET should be set before collection"
        );

        // Minor collection clears card bytes.
        gc.do_collect_nursery();

        let hdr = unsafe { header_of(obj.0) };
        assert!(
            unsafe { !(*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET flag should be cleared after collection"
        );
        assert!(
            !gc.is_card_dirty(obj, 0),
            "card 0 should be cleared after collection"
        );
    }

    #[test]
    fn test_card_marking_dirty_cards_list() {
        let mut gc = test_gc(4096);
        let (obj, _) = alloc_card_array(&mut gc, 512);

        // Mark cards for indices in different card pages.
        gc.do_write_barrier_card(obj, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 128, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj, 256, DEFAULT_CARD_PAGE_SHIFT);

        let dirty = gc.dirty_cards(obj);
        assert_eq!(dirty, vec![0, 1, 2], "should have cards 0, 1, 2 dirty");
    }

    #[test]
    fn test_card_marking_fallback_without_has_cards() {
        // Object without HAS_CARDS should fall back to remembered set.
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        // Do NOT set HAS_CARDS — no card header space allocated.

        gc.do_write_barrier_card(obj, 5, DEFAULT_CARD_PAGE_SHIFT);

        // Should fall back to full remembered set.
        assert_eq!(gc.remembered_set.len(), 1, "should add to remembered set");
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "should not add to old_objects_with_cards_set without HAS_CARDS"
        );
    }

    #[test]
    fn test_card_clear_individual() {
        let mut gc = test_gc(4096);
        let (obj1, _) = alloc_card_array(&mut gc, 512);
        let (obj2, _) = alloc_card_array(&mut gc, 512);

        gc.do_write_barrier_card(obj1, 0, DEFAULT_CARD_PAGE_SHIFT);
        gc.do_write_barrier_card(obj2, 0, DEFAULT_CARD_PAGE_SHIFT);

        // Clear only obj1's cards.
        gc.clear_cards(obj1.0);

        assert!(!gc.is_card_dirty(obj1, 0), "obj1 cards should be cleared");
        assert!(
            gc.is_card_dirty(obj2, 0),
            "obj2 cards should still be dirty"
        );
    }

    // ── SafepointMap tests ──

    #[test]
    fn test_safepoint_map_register_and_lookup() {
        let mut smap = SafepointMap::new();

        let mut gc_map_0 = crate::GcMap::new();
        gc_map_0.set_ref(0);
        gc_map_0.set_ref(3);

        let mut gc_map_1 = crate::GcMap::new();
        gc_map_1.set_ref(1);
        gc_map_1.set_ref(7);

        smap.add(100, gc_map_0);
        smap.add(200, gc_map_1);

        // Lookup existing entries.
        let found_0 = smap.lookup(100).unwrap();
        assert!(found_0.is_ref(0));
        assert!(found_0.is_ref(3));
        assert!(!found_0.is_ref(1));

        let found_1 = smap.lookup(200).unwrap();
        assert!(found_1.is_ref(1));
        assert!(found_1.is_ref(7));
        assert!(!found_1.is_ref(0));

        // Lookup non-existent offset returns None.
        assert!(smap.lookup(999).is_none());
    }

    #[test]
    fn test_safepoint_map_empty() {
        let smap = SafepointMap::new();
        assert!(smap.lookup(0).is_none());
        assert!(smap.entries.is_empty());
    }

    // ── CompiledCodeRegistry tests ──

    #[test]
    fn test_compiled_code_registry_register_and_find() {
        let mut registry = CompiledCodeRegistry::new();
        assert!(registry.is_empty());

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(16, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 42,
        });

        assert_eq!(registry.len(), 1);

        // Address inside the region.
        let (region, offset) = registry.find_region(0x1010).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0x10);

        // Address at the start.
        let (region, offset) = registry.find_region(0x1000).unwrap();
        assert_eq!(region.loop_token, 42);
        assert_eq!(offset, 0);

        // Address outside the region.
        assert!(registry.find_region(0x900).is_none());
        assert!(registry.find_region(0x1100).is_none());
    }

    #[test]
    fn test_compiled_code_registry_multiple_regions() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 1,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x3000,
            code_size: 0x200,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 8,
            loop_token: 2,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x80,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 2,
            loop_token: 3,
        });

        assert_eq!(registry.len(), 3);

        // Each region should be findable.
        assert_eq!(registry.find_region(0x1050).unwrap().0.loop_token, 1);
        assert_eq!(registry.find_region(0x2040).unwrap().0.loop_token, 3);
        assert_eq!(registry.find_region(0x3100).unwrap().0.loop_token, 2);

        // Gap between regions returns None.
        assert!(registry.find_region(0x1200).is_none());
    }

    #[test]
    fn test_compiled_code_registry_unregister() {
        let mut registry = CompiledCodeRegistry::new();

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 10,
        });
        registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 0x100,
            safepoint_map: SafepointMap::new(),
            frame_size_slots: 4,
            loop_token: 20,
        });

        assert_eq!(registry.len(), 2);

        registry.unregister(10);
        assert_eq!(registry.len(), 1);
        assert!(registry.find_region(0x1050).is_none());
        assert_eq!(registry.find_region(0x2050).unwrap().0.loop_token, 20);
    }

    #[test]
    fn test_compiled_code_registry_safepoint_lookup_for_root_scanning() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x20, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x5000,
            code_size: 0x200,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 99,
        });

        // Simulate finding a return address and looking up the safepoint map.
        let return_addr = 0x5020;
        let (region, offset) = registry.find_region(return_addr).unwrap();
        let gc_map = region.safepoint_map.lookup(offset).unwrap();

        // Verify the GC map identifies the correct slots.
        assert!(gc_map.is_ref(0), "slot 0 should be a GC ref");
        assert!(!gc_map.is_ref(1), "slot 1 should not be a GC ref");
        assert!(gc_map.is_ref(2), "slot 2 should be a GC ref");
        assert!(!gc_map.is_ref(3), "slot 3 should not be a GC ref");
    }

    #[test]
    fn test_scan_frame_enumerates_gc_ref_slots() {
        let mut registry = CompiledCodeRegistry::new();

        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x10, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0xA000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 77,
        });

        // Allocate a fake frame on the stack.
        let frame: [usize; 4] = [111, 222, 333, 444];
        let frame_base = frame.as_ptr();

        let return_addr = 0xA010;
        let roots = unsafe { registry.scan_frame(return_addr, frame_base) };

        // Should find slots 0 and 2.
        assert_eq!(roots.len(), 2);
        unsafe {
            assert_eq!(*(roots[0] as *const usize), 111);
            assert_eq!(*(roots[1] as *const usize), 333);
        }
    }

    // ── Incremental marking tests ──

    #[test]
    fn test_incremental_marking_basic() {
        // Start an incremental cycle, run steps, and verify completion.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote some objects to old gen via minor collection.
        let obj1 = gc.alloc_with_type(tid, 16);
        let obj2 = gc.alloc_with_type(tid, 16);
        let mut root1 = obj1;
        let mut root2 = obj2;
        unsafe {
            gc.roots.add(&mut root1);
            gc.roots.add(&mut root2);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root1.0));
        assert!(!gc.is_in_nursery(root2.0));

        // Manually start an incremental cycle.
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Run marking steps until complete.
        let mut steps = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            steps += 1;
            if steps > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Marking should have processed the 2 root objects.
        assert!(gc.incremental_objects_marked() >= 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_piggyback() {
        // Verify that incremental marking progresses during nursery collections.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Promote several objects to old gen by rooting each and collecting.
        let mut roots_storage = vec![GcRef::NULL; 10];
        for r in roots_storage.iter_mut() {
            unsafe {
                gc.roots.add(r);
            }
        }
        for r in roots_storage.iter_mut() {
            let obj = gc.alloc_with_type(tid, 16);
            *r = obj;
        }
        gc.do_collect_nursery();
        for r in &roots_storage {
            assert!(!gc.is_in_nursery(r.0));
        }

        // Start an incremental cycle with a tiny budget so it takes
        // multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Each nursery collection should advance the marking.
        let marked_before = gc.incremental_objects_marked();
        gc.do_collect_nursery();
        // After one nursery collection with budget=1, we should have
        // marked at least one more object (if any remained).
        assert!(
            gc.incremental_objects_marked() > marked_before,
            "marking should advance during nursery collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_minor_collection_can_take_multiple_major_steps_when_promotions_outpace_credit() {
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build an old-gen chain large enough that a single budget=1 marking
        // step cannot finish it.
        let mut prev = GcRef::NULL;
        for _ in 0..6 {
            let obj = gc.alloc_in_oldgen(tid, GcHeader::SIZE + ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }
        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.incr_state.gray_stack.len() >= 1);

        // Simulate a minor collection that promoted more than one step's worth
        // of objects so incminimark-style accounting demands extra progress.
        gc.bytes_made_old_since_cycle = gc.config.nursery_size;
        gc.threshold_bytes_made_old = 0;
        gc.run_major_progress_after_minor();

        assert!(
            gc.incremental_objects_marked() >= 2
                || gc.threshold_bytes_made_old > gc.config.nursery_size / 2,
            "major progress should take multiple steps when promoted bytes outpace credit"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_budget() {
        // Each step should process at most `mark_budget_per_step` objects.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create a chain of 10 objects so marking has plenty of work.
        let mut prev = GcRef::NULL;
        let mut roots = Vec::new();
        for _ in 0..10 {
            let obj = gc.alloc_with_type(tid, ptr_size);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
            roots.push(prev);
        }

        // Root the head of the chain.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start incremental cycle with a tiny byte budget so each step still
        // processes exactly one object.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // First step: marks at most 1 object.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after marking only 1 out of 10");
        assert_eq!(gc.incremental_objects_marked(), 1);

        // Second step: marks 1 more.
        let done = gc.incremental_mark_step();
        assert!(!done, "should not be done after 2 total");
        assert_eq!(gc.incremental_objects_marked(), 2);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_marking_completes() {
        // A full incremental cycle (start -> repeated steps -> sweep)
        // produces the same result as a stop-the-world full collection:
        // unreachable old-gen objects are freed.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create reachable and unreachable old-gen objects.
        let reachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(reachable.0 as *mut GcRef) = GcRef::NULL;
        }
        let unreachable = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(unreachable.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root = reachable;
        let mut root2 = unreachable;
        unsafe {
            gc.roots.add(&mut root);
            gc.roots.add(&mut root2);
        }

        // Promote both to old gen.
        gc.do_collect_nursery();
        assert_eq!(gc.oldgen.object_count(), 2);

        // Unroot the unreachable one.
        gc.roots.remove(&mut root2);

        // Run incremental cycle with budget=1 to force multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();

        // Drive the cycle to completion.
        let mut iterations = 0;
        while gc.is_incremental_marking() {
            gc.incremental_mark_step();
            iterations += 1;
            if iterations > 100 {
                panic!("incremental marking did not complete");
            }
        }

        // Finish: sweep unreachable objects.
        gc.finish_incremental_cycle();

        // Only the reachable object should survive.
        assert_eq!(gc.oldgen.object_count(), 1);
        assert!(!root.is_null());

        gc.roots.clear();
    }

    // ── GC stress tests ──

    #[test]
    #[cfg(debug_assertions)]
    fn test_gc_stress_with_safepoint_scanning() {
        // Register a compiled code region with a safepoint map, then
        // allocate objects under pressure so nursery collections fire.
        // After collection, verify that roots discovered via scan_frame
        // point to valid, promoted objects.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(512); // small nursery to force frequent collections
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size * 2, vec![0, ptr_size]));

        // Build a compiled code registry with a safepoint map marking
        // frame slots 0 and 2 as GC references.
        let mut registry = CompiledCodeRegistry::new();
        let mut smap = SafepointMap::new();
        let mut gc_map = crate::GcMap::new();
        gc_map.set_ref(0);
        gc_map.set_ref(2);
        smap.add(0x50, gc_map);

        registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 0x100,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        // Simulate a JIT frame: slots 0 and 2 hold GcRefs, slots 1 and 3
        // hold non-pointer data.
        let obj_a = gc.alloc_with_type(tid, ptr_size * 2);
        let obj_b = gc.alloc_with_type(tid, ptr_size * 2);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *((obj_a.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *((obj_b.0 + ptr_size) as *mut GcRef) = GcRef::NULL;
        }

        let frame: [usize; 4] = [obj_a.0, 0xDEAD, obj_b.0, 0xBEEF];

        // Register frame slots as GC roots (simulating what the backend does
        // at a safepoint).
        let roots_from_frame = unsafe { registry.scan_frame(0x1050, frame.as_ptr()) };
        assert_eq!(roots_from_frame.len(), 2);

        // Register the scanned slots as roots with the GC.
        for root_ptr in &roots_from_frame {
            unsafe {
                gc.roots.add(*root_ptr as *mut GcRef);
            }
        }

        // Allocate many objects to force multiple nursery collections.
        for i in 0..200 {
            let filler = gc.alloc_with_type(tid, ptr_size * 2);
            unsafe {
                *(filler.0 as *mut u64) = i as u64;
            }
        }
        assert!(
            gc.minor_collections > 0,
            "should have triggered nursery collections"
        );

        // Read back the GcRefs from the frame slots (the GC may have updated
        // them when it promoted the objects).
        let ref_a = GcRef(frame[0]);
        let ref_b = GcRef(frame[2]);

        // The original nursery objects should have been forwarded.
        // The frame slots must now point to valid (non-nursery) addresses.
        assert!(!ref_a.is_null());
        assert!(!ref_b.is_null());
        assert!(
            !gc.is_in_nursery(ref_a.0),
            "object A should have been promoted out of nursery"
        );
        assert!(
            !gc.is_in_nursery(ref_b.0),
            "object B should have been promoted out of nursery"
        );

        // Verify non-GC slots are untouched.
        assert_eq!(frame[1], 0xDEAD);
        assert_eq!(frame[3], 0xBEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_incremental_gc_under_allocation_pressure() {
        // Allocate many objects forming a linked list, promote to old gen,
        // then run an incremental major cycle with a tiny budget while
        // continuing to allocate. Verify data integrity throughout.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024);
        // Object layout: [next: GcRef][data: u64] = 16 bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Build a linked list of 20 objects.
        let mut prev = GcRef::NULL;
        let mut all_roots: Vec<GcRef> = Vec::new();
        for i in 0..20u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xA000 + i;
            }
            prev = obj;
            all_roots.push(obj);
        }

        // Root only the head of the list.
        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the whole list to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));

        // Start an incremental cycle with budget = 2 so it takes many steps.
        gc.set_mark_budget(2);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Interleave incremental marking steps with new allocations.
        let mut step_count = 0;
        while gc.is_incremental_marking() {
            // Allocate a few new (short-lived) objects to maintain pressure.
            for _ in 0..5 {
                let tmp = gc.alloc_with_type(tid, ptr_size + 8);
                unsafe {
                    *(tmp.0 as *mut GcRef) = GcRef::NULL;
                    *((tmp.0 + ptr_size) as *mut u64) = 0xFFFF;
                }
            }
            let done = gc.incremental_mark_step();
            step_count += 1;
            if done {
                break;
            }
            assert!(step_count < 200, "incremental marking should converge");
        }

        // Complete the cycle.
        gc.finish_incremental_cycle();
        assert!(gc.major_collections > 0);

        // Walk the list from the head and verify all data values.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            // data should be 0xA000 + (19 - count) because the list was
            // built in reverse.
            assert_eq!(
                data,
                0xA000 + (19 - count) as u64,
                "data corruption detected at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 20, "entire list should be reachable");

        gc.roots.clear();
    }

    #[test]
    #[cfg(debug_assertions)]
    fn test_card_marking_under_write_pressure() {
        // Allocate a large array in old gen, write GC refs into many slots,
        // and verify that card marking accurately tracks the dirty ranges
        // so that nursery objects stored in the array survive collection.

        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(2048);

        // Element type: simple 16-byte object.
        let elem_tid = gc.register_type(TypeInfo::simple(16));

        // Array type: varsize, base_size = 8 (length field at offset 0),
        // each item is a GcRef (item_size = ptr_size), items have GC ptrs.
        let arr_tid = gc.register_type(TypeInfo::varsize(
            8,          // base_size (length field)
            ptr_size,   // item_size
            0,          // length_offset
            true,       // items_have_gc_ptrs
            Vec::new(), // no fixed GC ptr fields
        ));

        let array_length = 512usize;
        let total_payload = 8 + ptr_size * array_length;

        // Allocate with inline card bytes (incminimark.py:1027 parity).
        let arr = gc.alloc_in_oldgen_with_cards(
            arr_tid,
            GcHeader::SIZE + total_payload,
            array_length,
            true,
        );

        // Write the length field.
        unsafe {
            *(arr.0 as *mut usize) = array_length;
        }

        let hdr = unsafe { header_of(arr.0) };
        assert!(unsafe { (*hdr).has_flag(flags::HAS_CARDS) });

        // Initialize all array slots to NULL.
        let items_start = arr.0 + 8;
        for i in 0..array_length {
            unsafe {
                *((items_start + i * ptr_size) as *mut GcRef) = GcRef::NULL;
            }
        }

        // Write nursery objects into scattered array positions and trigger
        // card-marking write barriers.
        let write_indices: Vec<usize> = vec![0, 1, 5, 64, 127, 128, 200, 255, 256, 400, 511];
        let mut expected_cards: HashSet<usize> = HashSet::new();
        let mut nursery_objs: Vec<(usize, GcRef)> = Vec::new();

        for &idx in &write_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            // Write a distinctive marker.
            unsafe {
                *(obj.0 as *mut u64) = 0xCAFE_0000 + idx as u64;
            }
            // Store into the array.
            unsafe {
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            // Write barrier with card marking.
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
            expected_cards.insert(idx >> DEFAULT_CARD_PAGE_SHIFT);
            nursery_objs.push((idx, obj));
        }

        // Verify the correct cards are dirty.
        let dirty = gc.dirty_cards(arr);
        let dirty_set: HashSet<usize> = dirty.into_iter().collect();
        assert_eq!(
            dirty_set, expected_cards,
            "dirty card set should match expected cards"
        );

        // Root the array so the collection traces it via card scanning.
        let mut root = arr;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger nursery collection — this should use card scanning
        // to find and promote the nursery objects stored in the array.
        gc.do_collect_nursery();

        // After collection, all stored objects should be promoted and
        // their data should be intact.
        for &(idx, _orig) in &nursery_objs {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(
                !slot_ref.is_null(),
                "array slot {idx} should not be null after collection"
            );
            assert!(
                !gc.is_in_nursery(slot_ref.0),
                "array slot {idx} should be promoted to old gen"
            );
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(
                marker,
                0xCAFE_0000 + idx as u64,
                "data in array slot {idx} should be preserved"
            );
        }

        // Cards should be cleared after collection.
        let hdr = unsafe { header_of(root.0) };
        assert!(
            unsafe { !(*hdr).has_flag(flags::CARDS_SET) },
            "CARDS_SET should be cleared after collection"
        );
        assert!(
            gc.old_objects_with_cards_set.is_empty(),
            "old_objects_with_cards_set should be empty after collection"
        );
        assert!(
            unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) },
            "TRACK_YOUNG_PTRS should be re-set after collection"
        );

        // Now do a second round of writes to verify card marking works
        // again after collection.
        let more_indices = vec![10, 300, 450];
        for &idx in &more_indices {
            let obj = gc.alloc_with_type(elem_tid, 16);
            unsafe {
                *(obj.0 as *mut u64) = 0xBEEF_0000 + idx as u64;
                *((items_start + idx * ptr_size) as *mut GcRef) = obj;
            }
            gc.do_write_barrier_card(arr, idx, DEFAULT_CARD_PAGE_SHIFT);
        }

        gc.do_collect_nursery();

        for &idx in &more_indices {
            let slot_ref = unsafe { *((items_start + idx * ptr_size) as *const GcRef) };
            assert!(!slot_ref.is_null());
            assert!(!gc.is_in_nursery(slot_ref.0));
            let marker = unsafe { *(slot_ref.0 as *const u64) };
            assert_eq!(marker, 0xBEEF_0000 + idx as u64);
        }

        gc.roots.clear();
    }

    // ── Write barrier + incremental marking interaction tests ──

    #[test]
    fn test_write_barrier_during_incremental_marking() {
        // Verify that write barriers fired during an active incremental
        // marking cycle correctly add old-gen objects to the remembered set.
        // The remembered set is processed at the next minor collection,
        // ensuring mutated old-gen objects are re-scanned.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [field: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Create 3 old-gen objects (A, B, C) with no pointers initially.
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = GcRef::NULL;
            *(obj_b.0 as *mut GcRef) = GcRef::NULL;
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }

        let mut root_a = obj_a;
        let mut root_b = obj_b;
        let mut root_c = obj_c;
        unsafe {
            gc.roots.add(&mut root_a);
            gc.roots.add(&mut root_b);
            gc.roots.add(&mut root_c);
        }

        // Promote all to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        assert!(!gc.is_in_nursery(root_b.0));
        assert!(!gc.is_in_nursery(root_c.0));

        // All old-gen objects should have TRACK_YOUNG_PTRS set.
        assert!(unsafe { (*header_of(root_a.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_b.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(unsafe { (*header_of(root_c.0)).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Start an incremental marking cycle with budget=1 so it stays
        // active across multiple steps.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // During marking, perform write barriers on A and B.
        gc.do_write_barrier(root_a);
        gc.do_write_barrier(root_b);

        // A and B should be in the remembered set.
        assert!(
            gc.remembered_set.contains(&root_a.0),
            "write barrier should add A to remembered set during marking"
        );
        assert!(
            gc.remembered_set.contains(&root_b.0),
            "write barrier should add B to remembered set during marking"
        );
        // C was not written to, so it shouldn't be in remembered set.
        assert!(
            !gc.remembered_set.contains(&root_c.0),
            "C should not be in remembered set"
        );

        // TRACK_YOUNG_PTRS should be cleared on A and B.
        assert!(!unsafe { (*header_of(root_a.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert!(!unsafe { (*header_of(root_b.0)).has_flag(flags::TRACK_YOUNG_PTRS) });
        // C still has it.
        assert!(unsafe { (*header_of(root_c.0)).has_flag(flags::TRACK_YOUNG_PTRS) });

        // Drive the incremental cycle to completion via nursery collections.
        for _ in 0..50 {
            if !gc.is_incremental_marking() {
                break;
            }
            gc.do_collect_nursery();
        }

        // No objects should be lost — all 3 are still rooted and should
        // survive the sweep.
        gc.do_collect_full();
        assert_eq!(
            gc.oldgen.object_count(),
            3,
            "all 3 rooted objects should survive full collection"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_mutation_during_incremental_preserves_reachability() {
        // During an incremental marking cycle, mutate an old-gen object to
        // point to a newly promoted object (D). The write barrier ensures D
        // is reachable through the remembered set, so D survives the sweep.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(4096);
        // Object layout: [next: GcRef] = ptr_size bytes, GC ptr at offset 0
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size, vec![0]));

        // Build a chain A→B→C in the nursery.
        let obj_c = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_c.0 as *mut GcRef) = GcRef::NULL;
        }
        let obj_b = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_b.0 as *mut GcRef) = obj_c;
        }
        let obj_a = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_a.0 as *mut GcRef) = obj_b;
        }

        let mut root_a = obj_a;
        unsafe {
            gc.roots.add(&mut root_a);
        }

        // Promote A→B→C to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_a.0));
        let promoted_b = unsafe { *(root_a.0 as *const GcRef) };
        let promoted_c = unsafe { *(promoted_b.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(promoted_b.0));
        assert!(!gc.is_in_nursery(promoted_c.0));

        // Start incremental marking with a small budget.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        // Do one marking step — only partially through the graph.
        gc.incremental_mark_step();

        // Now allocate D in the nursery.
        let obj_d = gc.alloc_with_type(tid, ptr_size);
        unsafe {
            *(obj_d.0 as *mut GcRef) = GcRef::NULL;
        }

        // Promote D to old gen by rooting it temporarily.
        let mut root_d = obj_d;
        unsafe {
            gc.roots.add(&mut root_d);
        }
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(root_d.0));

        // Mutate A to point to D instead of B (A→D).
        // The write barrier ensures A is in the remembered set.
        gc.do_write_barrier(root_a);
        unsafe {
            *(root_a.0 as *mut GcRef) = root_d;
        }

        // Remove D's direct root — D is now only reachable via A→D.
        gc.roots.remove(&mut root_d);

        // Complete the incremental cycle: drive marking to completion
        // and then sweep.
        gc.do_collect_full();

        // A and D should survive (A is rooted, D reachable via A→D).
        // B and C may or may not survive depending on marking order,
        // but D MUST survive.
        let d_addr = root_d.0;
        let a_field = unsafe { *(root_a.0 as *const GcRef) };
        assert_eq!(
            a_field.0, d_addr,
            "A should still point to D after collection"
        );

        // Verify D is actually alive by checking it's still in old gen.
        assert!(
            !a_field.is_null(),
            "D must survive collection — it's reachable via A"
        );

        gc.roots.clear();
    }

    #[test]
    fn test_nursery_alloc_during_incremental_marking() {
        // Allocate new nursery objects between incremental marking steps.
        // Trigger nursery collections that piggyback marking steps.
        // Verify that newly allocated objects are correctly promoted and
        // that the original old-gen object graph remains intact.
        let ptr_size = std::mem::size_of::<GcRef>();
        let mut gc = test_gc(1024); // small nursery to force frequent collections
        // Object layout: [next: GcRef][data: u64] = ptr_size + 8 bytes
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(ptr_size + 8, vec![0]));

        // Create initial old-gen objects (a small chain of 5).
        let mut prev = GcRef::NULL;
        for i in 0..5u64 {
            let obj = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
                *((obj.0 + ptr_size) as *mut u64) = 0xBB00 + i;
            }
            prev = obj;
        }

        let mut head = prev;
        unsafe {
            gc.roots.add(&mut head);
        }

        // Promote the chain to old gen.
        gc.do_collect_nursery();
        assert!(!gc.is_in_nursery(head.0));
        let old_count_after_promote = gc.oldgen.object_count();
        assert_eq!(old_count_after_promote, 5);

        // Start incremental marking with budget=1.
        gc.set_mark_budget(1);
        gc.start_incremental_cycle();
        assert!(gc.is_incremental_marking());

        let minor_before = gc.minor_collections;

        // Allocate many nursery objects between marking steps. The small
        // nursery (1024 bytes) forces nursery collections that piggyback
        // incremental marking steps.
        for i in 0..100u64 {
            let tmp = gc.alloc_with_type(tid, ptr_size + 8);
            unsafe {
                *(tmp.0 as *mut GcRef) = GcRef::NULL;
                *((tmp.0 + ptr_size) as *mut u64) = 0xDD00 + i;
            }
        }

        let minor_during = gc.minor_collections - minor_before;
        assert!(
            minor_during >= 2,
            "should have triggered multiple nursery collections, got {minor_during}"
        );

        // The incremental marking should have advanced via piggybacking.
        assert!(
            gc.incremental_objects_marked() > 0,
            "piggybacked marking should have processed some objects"
        );

        // Drive any remaining incremental marking to completion and sweep.
        // Use do_collect_full which correctly handles an in-progress cycle.
        gc.do_collect_full();
        assert!(gc.major_collections > 0);

        // Verify the original chain is intact — the 5 rooted objects
        // survived the incremental major cycle.
        let mut cursor = head;
        let mut count = 0;
        while !cursor.is_null() {
            let data = unsafe { *((cursor.0 + ptr_size) as *const u64) };
            assert_eq!(
                data,
                0xBB00 + (4 - count) as u64,
                "original chain data corrupted at node {count}"
            );
            cursor = unsafe { *(cursor.0 as *const GcRef) };
            count += 1;
        }
        assert_eq!(count, 5, "entire chain should be reachable after cycle");

        gc.roots.clear();
    }

    // ── JIT integration hook tests ──

    #[test]
    fn test_jit_remember_young_pointer() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // Allocate an old-gen object.
        let obj = gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        assert!(!gc.is_in_nursery(obj.0));

        // Initially TRACK_YOUNG_PTRS is set.
        let hdr = unsafe { header_of(obj.0) };
        assert!(unsafe { (*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });

        // JIT fast-path barrier: clears flag and adds to remembered set.
        gc.jit_remember_young_pointer(obj);

        assert!(unsafe { !(*hdr).has_flag(flags::TRACK_YOUNG_PTRS) });
        assert_eq!(gc.remembered_set_len(), 1);

        // Calling again adds a second entry (JIT fast-path does not
        // deduplicate; the collector handles this during minor collection).
        gc.jit_remember_young_pointer(obj);
        assert_eq!(gc.remembered_set_len(), 2);
    }

    #[test]
    fn test_jit_remember_young_pointer_null_is_noop() {
        let mut gc = test_gc(4096);
        gc.jit_remember_young_pointer(GcRef::NULL);
        assert_eq!(gc.remembered_set_len(), 0);
    }

    #[test]
    fn test_jit_remember_young_pointer_survives_collection() {
        // Verify that the remembered-set entry from jit_remember_young_pointer
        // causes a young object to survive minor collection.
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));

        // Create an old-gen parent and a young child.
        let parent = gc.alloc_in_oldgen(tid, GcHeader::SIZE + std::mem::size_of::<GcRef>());
        let child = gc.alloc_with_type(tid, std::mem::size_of::<GcRef>());
        unsafe {
            *(child.0 as *mut u64) = 0xABCD_1234;
            *(parent.0 as *mut GcRef) = child;
        }

        // Use the JIT hook instead of do_write_barrier.
        gc.jit_remember_young_pointer(parent);

        let mut root = parent;
        unsafe {
            gc.roots.add(&mut root);
        }

        gc.collect_nursery();

        // The child should have been promoted.
        let child_ref = unsafe { *(root.0 as *const GcRef) };
        assert!(!gc.is_in_nursery(child_ref.0));
        assert!(!child_ref.is_null());
        let val = unsafe { *(child_ref.0 as *const u64) };
        assert_eq!(val, 0xABCD_1234);

        gc.roots.clear();
    }

    #[test]
    fn test_can_optimize_cond_call() {
        let gc = test_gc(4096);
        assert!(gc.can_optimize_cond_call());
    }

    #[test]
    fn test_can_optimize_cond_call_via_trait() {
        let gc = test_gc(4096);
        let alloc: &dyn GcAllocator = &gc;
        assert!(alloc.can_optimize_cond_call());
    }

    #[test]
    fn test_gc_step_no_work_when_old_gen_small() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        // With an almost-empty old gen, gc_step should do nothing.
        assert!(!gc.gc_step());
        assert!(!gc.is_incremental_marking());
    }

    #[test]
    fn test_gc_step_triggers_incremental() {
        let mut gc = test_gc(256);
        gc.register_type(TypeInfo::simple(16));

        // Force a minor collection to set last_major_bytes baseline.
        let obj = gc.alloc_with_type(0, 16);
        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }
        gc.collect_full();

        // Fill old gen to trigger the major-cycle ratio threshold.
        // Allocate many objects directly in old gen.
        for _ in 0..200 {
            gc.alloc_in_oldgen(0, GcHeader::SIZE + 16);
        }

        // gc_step should now start an incremental cycle and do work.
        let did_work = gc.gc_step();
        assert!(did_work);

        gc.roots.clear();
    }

    #[test]
    fn test_gc_step_advances_marking() {
        // Nursery must be large enough to hold all 20 chain links without an
        // auto-collect mid-loop; otherwise intermediate links get dropped and
        // `prev` ends up chaining through stale nursery addresses (reset-zero
        // bytes parse as tid=0 / `simple(16)`, which causes `copy_nursery_object`
        // to over-read past the nursery end).
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        // Build a chain of old-gen objects so there's marking work to do.
        let mut prev = GcRef::NULL;
        let ptr_tid = gc.register_type(TypeInfo::with_gc_ptrs(
            std::mem::size_of::<GcRef>(),
            vec![0],
        ));
        for _ in 0..20 {
            let obj = gc.alloc_with_type(ptr_tid, std::mem::size_of::<GcRef>());
            unsafe {
                *(obj.0 as *mut GcRef) = prev;
            }
            prev = obj;
        }

        let mut root = prev;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Promote everything to old gen.
        gc.collect_nursery();

        // Add more old-gen objects to trigger the ratio threshold.
        for _ in 0..200 {
            gc.alloc_in_oldgen(tid, GcHeader::SIZE + 16);
        }

        // First step: start cycle.
        let work1 = gc.gc_step();
        assert!(work1);
        let marked_after_1 = gc.incremental_objects_marked();

        // Second step: advance marking further.
        if gc.is_incremental_marking() {
            let work2 = gc.gc_step();
            assert!(work2);
            assert!(gc.incremental_objects_marked() >= marked_after_1);
        }

        gc.roots.clear();
    }

    // ── Pin / Unpin / jit_free tests ──

    #[test]
    fn test_pin_prevents_nursery_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker value.
        unsafe {
            *(obj.0 as *mut u64) = 0xCAFE_BABE;
        }

        // Pin the object and root it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Trigger minor collection.
        gc.do_collect_nursery();

        // The root should still point to the same nursery address.
        assert_eq!(root.0, obj.0);
        assert!(gc.is_in_nursery(root.0));

        // Data should be intact.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xCAFE_BABE);

        gc.roots.clear();
    }

    #[test]
    fn test_unpin_allows_move() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);
        assert!(gc.is_in_nursery(obj.0));

        // Write a marker.
        unsafe {
            *(obj.0 as *mut u64) = 0xDEAD_BEEF;
        }

        // Pin, then unpin.
        assert!(gc.pin(obj));
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));

        let mut root = obj;
        unsafe {
            gc.roots.add(&mut root);
        }

        // Collection should now move the object to old gen.
        gc.do_collect_nursery();

        // The root should now point to old gen (different address).
        assert!(!gc.is_in_nursery(root.0));
        assert_ne!(root.0, obj.0);

        // Data should be preserved after the move.
        let val = unsafe { *(root.0 as *const u64) };
        assert_eq!(val, 0xDEAD_BEEF);

        gc.roots.clear();
    }

    #[test]
    fn test_is_pinned_query() {
        let mut gc = test_gc(4096);
        gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(0, 16);

        // Not pinned by default.
        assert!(!gc.is_pinned(obj));

        // Pin it.
        assert!(gc.pin(obj));
        assert!(gc.is_pinned(obj));

        // Null cannot be pinned.
        assert!(!gc.pin(GcRef(0)));
        assert!(!gc.is_pinned(GcRef(0)));

        // Unpin.
        gc.unpin(obj);
        assert!(!gc.is_pinned(obj));
    }

    #[test]
    fn test_jit_free_unregisters_code() {
        let mut gc = test_gc(4096);

        let smap = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x1000,
            code_size: 256,
            safepoint_map: smap,
            frame_size_slots: 4,
            loop_token: 1,
        });

        let smap2 = SafepointMap::new();
        gc.compiled_code_registry.register(CompiledCodeRegion {
            code_start: 0x2000,
            code_size: 512,
            safepoint_map: smap2,
            frame_size_slots: 8,
            loop_token: 2,
        });

        assert_eq!(gc.compiled_code_registry.len(), 2);

        // Free the first region.
        gc.jit_free(0x1000, 256);

        assert_eq!(gc.compiled_code_registry.len(), 1);
        assert!(gc.compiled_code_registry.find_region(0x1050).is_none());
        assert!(gc.compiled_code_registry.find_region(0x2050).is_some());

        // Free the second region.
        gc.jit_free(0x2000, 512);
        assert_eq!(gc.compiled_code_registry.len(), 0);
    }

    #[test]
    fn test_incremental_cycle_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xDEADBEEFu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.start_incremental_cycle();
        while !gc.gc_step() {}

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    #[test]
    fn test_full_collection_root_walk_skips_non_gc_roots() {
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let mut rooted = obj;
        let mut external = GcRef(Box::into_raw(Box::new(0xFEEDFACEu64)) as usize);
        unsafe {
            gc.roots.add(&mut rooted);
            gc.roots.add(&mut external);
        }

        gc.do_collect_nursery();
        gc.do_collect_full();

        assert!(!rooted.is_null());
        assert!(gc.oldgen.contains(rooted.0));

        gc.roots.clear();
        unsafe {
            drop(Box::from_raw(external.0 as *mut u64));
        }
    }

    #[test]
    fn test_incremental_cycle_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.start_incremental_cycle();
        gc.finish_incremental_cycle();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    #[test]
    fn test_full_collection_marks_jitframe_shadow_stack_roots() {
        let _guard = SHADOW_STACK_TEST_LOCK.lock().unwrap();
        crate::shadow_stack::clear();
        let mut gc = test_gc(4096);
        let tid = gc.register_type(TypeInfo::simple(16));

        let obj = gc.alloc_with_type(tid, 16);
        let depth = crate::shadow_stack::jf_depth();
        crate::shadow_stack::push_jf(obj);

        gc.do_collect_nursery();
        let rooted = crate::shadow_stack::jf_top_ptr();
        assert!(gc.oldgen.contains(rooted.0));

        gc.do_collect_full();

        assert!(gc.oldgen.contains(rooted.0));
        crate::shadow_stack::pop_jf_to(depth);
    }

    // Note: a `test_minor_root_walk_skips_interior_nursery_pointers`
    // test used to live here. It was predicated on a majit-local
    // extension where the nursery tracked exact object-start addresses
    // and `is_nursery_object_start` filtered out interior pointers.
    //
    // incminimark.py:1208 is_in_nursery is a pure range check
    //     return self.nursery <= addr < self.nursery + self.nursery_size
    // RPython guarantees that GC roots are exact object pointers (the
    // shadow stack only ever carries exact pointers produced by the
    // compiler), so interior-pointer filtering is not part of the GC
    // contract. The test disagreed with that contract and was removed
    // to keep majit-gc structurally aligned with RPython.
}
