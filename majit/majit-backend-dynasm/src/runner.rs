use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::Mutex;
/// runner.py: AbstractX86CPU — the Backend trait implementation.
///
/// This is the entry point for the dynasm backend, corresponding to
/// rpython/jit/backend/x86/runner.py AbstractX86CPU.
use std::sync::atomic::Ordering;

use majit_backend::{AsmInfo, Backend, BackendError, DeadFrame, ExitRecoveryLayout, JitCellToken};
use majit_ir::{FailDescr, GcRef, InputArg, Op, OpRef, Type, Value};

#[cfg(target_arch = "aarch64")]
use crate::aarch64::assembler::{AssemblerARM64 as Asm, CompiledCode};
#[cfg(target_arch = "aarch64")]
use crate::aarch64::cpu_ext::Aarch64CpuExt as ArchCpuExt;
use crate::arch;
use crate::codebuf;
use crate::frame::FrameData;
use crate::jitframe::JitFrame;
#[cfg(target_arch = "x86_64")]
use crate::x86::assembler::{Assembler386 as Asm, CompiledCode};
#[cfg(target_arch = "x86_64")]
use crate::x86::cpu_ext::X86CpuExt as ArchCpuExt;

/// Global CALL_ASSEMBLER target registry.
///
/// RPython stores `descr._ll_function_addr` on the target token
/// (x86/assembler.py:599) so `CALL_ASSEMBLER` can resolve the callee
/// address directly from the descriptor. pyre identifies callee tokens
/// by `u64 token_number` inside `MetaCallAssemblerDescr` (pyre PRE-
/// EXISTING-ADAPTATION — serializable descriptors), so a process-wide
/// `token_number -> DynasmCaTarget` index is required.
///
/// Each entry retains the callee's `Arc<CompiledLoopToken>` so
/// `handle_call_assembler` (rewrite.py:665-695) can sample
/// `_ll_initial_locs` and `frame_info` for the
/// `call_assembler_callee_locs` callback.  The Arc is created at
/// `register_pending_target` time and adopted onto
/// `JitCellToken.compiled_loop_token` at real-target registration so the
/// `frame_info` address baked into already-rewritten caller traces
/// stays valid across the pending → real transition (mirrors
/// `majit-backend-cranelift::compiler::register_call_assembler_target`,
/// `compiler.rs:2310-2326`).
struct DynasmCaTarget {
    /// 0 while pending; `compile_loop` overwrites with `_ll_function_addr`
    /// per `x86/assembler.py:599`.  Snapshot helpers filter out 0 entries
    /// so emitted CALL_ASSEMBLER code falls through to the force-fn
    /// trampoline until the callee compiles.
    code_addr: usize,
    /// `model.py:292-338` `CompiledLoopToken` — Arc-shared with the
    /// owning `JitCellToken` once the real target registers.
    compiled_loop_token: Arc<majit_backend::CompiledLoopToken>,
    /// `pyjitpl.py:3605` `outermost_jitdriver_sd.index_of_virtualizable`.
    /// Captured at pending-target registration since the Backend trait
    /// receives it there but `JitCellToken.virtualizable_arg_index` may
    /// not yet be wired at that moment.
    index_of_virtualizable: i32,
}

static CALL_ASSEMBLER_TARGETS: LazyLock<Mutex<HashMap<u64, DynasmCaTarget>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// `rewrite.py:665-695` `handle_call_assembler` per-callee metadata
/// lookup, sourced from the registered `DynasmCaTarget`'s CLT Arc.
/// Mirrors `majit-backend-cranelift::compiler.rs:6097-6121`.
pub(crate) fn lookup_call_assembler_callee_locs(
    token_number: u64,
) -> Option<majit_gc::rewrite::CallAssemblerCalleeLocs> {
    let guard = CALL_ASSEMBLER_TARGETS
        .lock()
        .expect("CALL_ASSEMBLER_TARGETS poisoned");
    let target = guard.get(&token_number)?;
    let clt = &target.compiled_loop_token;
    // `JitFrameInfo` is `#[repr(C)]` and the Arc keeps the allocation
    // pinned, matching cranelift's `compiler.rs:6107-6110` pattern.
    let frame_info_ptr = {
        let info = clt.frame_info.lock();
        &*info as *const majit_backend::JitFrameInfo as usize
    };
    let frame_depth = clt.frame_info.lock().jfi_frame_depth as usize;
    let ll_initial_locs = clt._ll_initial_locs.lock().clone();
    Some(majit_gc::rewrite::CallAssemblerCalleeLocs {
        _ll_initial_locs: ll_initial_locs,
        frame_depth,
        frame_info_ptr,
        index_of_virtualizable: target.index_of_virtualizable,
    })
}

thread_local! {
    /// llmodel.py self.gc_ll_descr — owned by the active dynasm
    /// backend on this thread. Stored as a thread-local so the
    /// backend-agnostic `majit_gc::ActiveGcGuardHooks` shims can
    /// reach the live allocator without taking a dynasm dependency.
    pub static DYNASM_ACTIVE_GC: RefCell<Option<Box<dyn majit_gc::GcAllocator>>> =
        const { RefCell::new(None) };
    static DYNASM_ACTIVE_GC_RAW: std::cell::Cell<Option<*mut dyn majit_gc::GcAllocator>> =
        const { std::cell::Cell::new(None) };
}

fn with_dynasm_active_gc<R>(f: impl FnOnce(&dyn majit_gc::GcAllocator) -> R) -> Option<R> {
    DYNASM_ACTIVE_GC.with(|cell| {
        let guard = cell.borrow();
        guard.as_deref().map(f)
    })
}

/// Clear both `DYNASM_ACTIVE_GC` and `DYNASM_ACTIVE_GC_RAW`. Callers
/// that want to drop the active dynasm GC must go through this helper
/// rather than mutating `DYNASM_ACTIVE_GC` directly, otherwise the raw
/// mirror used by `dynasm_gc_owns_object`'s reentrant fallback would be
/// left pointing at freed memory.
pub fn clear_gc_allocator() {
    // Drop the boxed allocator first so reentrant
    // `dynasm_gc_owns_object` queries from its drop body still resolve
    // old-heap addresses through the raw mirror, then clear the raw
    // mirror.
    DYNASM_ACTIVE_GC.with(|cell| {
        *cell.borrow_mut() = None;
    });
    DYNASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(None));
}

/// TYPE_INFO / CLASSTYPE constants read by the dynasm assemblers for
/// `GUARD_IS_OBJECT` and `GUARD_SUBCLASS`.
///
/// RPython reads these from `self.cpu.gc_ll_descr` at codegen time
/// (`x86/assembler.py:1934-1939`, `1946-1969`). The Rust assembler is a
/// transient emitter without a borrow of `DynasmBackend`, so the runner
/// pre-fetches the same values once per trace and passes them in.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuardGcTypeInfo {
    pub base_type_info: usize,
    pub shift_by: u8,
    pub sizeof_ti: usize,
    pub infobits_offset: usize,
    pub is_object_flag: u8,
    pub subclassrange_min_offset: usize,
}

fn dynasm_check_is_object(gcref: GcRef) -> bool {
    with_dynasm_active_gc(|gc| gc.check_is_object(gcref)).unwrap_or(false)
}

fn dynasm_get_actual_typeid(gcref: GcRef) -> Option<u32> {
    with_dynasm_active_gc(|gc| gc.get_actual_typeid(gcref)).flatten()
}

fn dynasm_subclass_range(classptr: usize) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.subclass_range(classptr)).flatten()
}

fn dynasm_typeid_subclass_range(typeid: u32) -> Option<(i64, i64)> {
    with_dynasm_active_gc(|gc| gc.typeid_subclass_range(typeid)).flatten()
}

/// gc.py:525-531 `get_nursery_free_addr` / `get_nursery_top_addr` parity:
/// the backend reads nursery slot addresses from the active GC descriptor,
/// NOT from a process-global singleton. Returns `(0, 0)` when no GC is
/// bound so the assembler falls back to the slow-path helper.
pub(crate) fn dynasm_nursery_addrs() -> (usize, usize) {
    with_dynasm_active_gc(|gc| (gc.nursery_free_addr(), gc.nursery_top_addr())).unwrap_or((0, 0))
}

/// Per-backend `CPU.load_supported_factors` (rewrite.py:1124 /
/// x86/runner.py:31 / llmodel.py:39). x86 addressing scales natively by
/// 1/2/4/8, aarch64 has no scaled store form and always expects factor 1.
#[cfg(target_arch = "x86_64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1, 2, 4, 8]
}

#[cfg(target_arch = "aarch64")]
fn gc_store_supported_factors() -> &'static [i64] {
    &[1]
}

/// Per-backend `CPU.supports_load_effective_address` — x86/runner.py:22
/// overrides model.py:22 base default `False` to `True`; aarch64/runner.py
/// inherits the base `False` (rewriter expands to INT_LSHIFT + INT_ADD +
/// INT_ADD per rewrite.py:1089-1098 instead of emitting LOAD_EFFECTIVE_ADDRESS).
#[cfg(target_arch = "x86_64")]
fn supports_load_effective_address() -> bool {
    true
}

#[cfg(target_arch = "aarch64")]
fn supports_load_effective_address() -> bool {
    false
}

fn dynasm_typeid_is_object(typeid: u32) -> Option<bool> {
    with_dynasm_active_gc(|gc| gc.typeid_is_object(typeid)).flatten()
}

/// Host-side nursery allocation trampoline. Published via
/// `majit_gc::set_active_alloc_nursery_typed` from `set_gc_allocator`
/// so backend-agnostic callers (e.g. pyre-object `w_int_new`) can
/// route through the live dynasm-owned GC without taking a backend
/// dependency.
fn dynasm_alloc_nursery_typed(type_id: u32, size: usize) -> GcRef {
    // NOTE host-side allocation must not trigger collection: the
    // caller holds a raw `*mut u8` on the Rust stack that is NOT
    // registered as a GC root. Collection here would move the
    // freshly-allocated nursery object, leaving the caller with a
    // dangling pointer. Routing through `alloc_nursery_no_collect_typed`
    // falls back to old-gen on nursery full — stable across minor
    // collections that fire between here and the caller's store into
    // a tracked slot.
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_nursery_no_collect_typed(type_id, size),
            None => GcRef(0),
        }
    })
}

/// Host-side old-gen allocation trampoline (Task #141). Used by
/// pyre-object allocators (`w_int_new`, `w_float_new`) whose
/// callers cannot register the returned pointer as a GC root before
/// subsequent allocations. MiniMark's old-gen is mark-sweep
/// (non-moving), so the returned pointer is stable across minor and
/// major collections.
fn dynasm_alloc_oldgen_typed(type_id: u32, size: usize) -> GcRef {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        match guard.as_deref_mut() {
            Some(gc) => gc.alloc_oldgen_typed(type_id, size),
            None => GcRef(0),
        }
    })
}

/// Host-side root-register trampoline (Task #141 option a). Bridges
/// `majit_gc::gc_add_root` to the active backend's `RootSet`.
///
/// # Safety
/// Caller must keep `slot` valid until [`dynasm_gc_remove_root`] is
/// called with the same pointer.
unsafe fn dynasm_gc_add_root(slot: *mut GcRef) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            unsafe { gc.add_root(slot) };
        }
    });
}

/// Companion to [`dynasm_gc_add_root`].
fn dynasm_gc_remove_root(slot: *mut GcRef) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            gc.remove_root(slot);
        }
    });
}

/// Host-side write-barrier trampoline for GC-managed objects updated
/// outside compiled code.
fn dynasm_gc_write_barrier(obj: GcRef) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_deref_mut() {
            gc.write_barrier(obj);
        }
    });
}

/// Host-side `is_managed_heap_object` trampoline. Lets host-side
/// allocators (`pyre_object::dealloc_items_block`) discriminate
/// `try_gc_alloc_stable`-allocated blocks from `std::alloc`-backed
/// fallback blocks during the L1/L2 stepping-stone window. Returns
/// `false` when no GC is installed (caller falls through to
/// `std::alloc::dealloc`).
fn dynasm_gc_owns_object(addr: usize) -> bool {
    DYNASM_ACTIVE_GC.with(|cell| {
        let guard = match cell.try_borrow() {
            Ok(guard) => guard,
            Err(_) => {
                // Structural adaptation: RPython's GC descriptor is a
                // normal object reference and `gc_current_object_address`
                // can query ownership while a collection is already in
                // progress. Pyre stores the active dynasm GC behind a
                // Rust `RefCell`; during `alloc_nursery` the mutable
                // borrow is held while extra-root walkers may ask whether
                // a mapdict key is GC-managed. Use the raw pointer only
                // for this immutable ownership query, matching the
                // read-only nature of RPython's descriptor call, instead
                // of panicking across the extern "C" slowpath.
                return DYNASM_ACTIVE_GC_RAW.with(|raw| match raw.get() {
                    Some(ptr) => unsafe { (&*ptr).is_managed_heap_object(addr) },
                    None => false,
                });
            }
        };
        match guard.as_deref() {
            Some(gc) => gc.is_managed_heap_object(addr),
            None => false,
        }
    })
}

/// `gc.py:51` malloc-helper OOM signaling.
///
/// `do_malloc_fixedsize_clear` raises `MemoryError` on failure;
/// translation lowers that to "store the singleton in
/// `cpu.pos_exc_value`, return NULL".  pyre malloc helpers return 0
/// directly, so this wrapper performs the equivalent thread-local
/// store before propagating the NULL up to the JIT-emitted
/// `CHECK_MEMORY_ERROR` (`x86/assembler.py:2334`,
/// `aarch64/assembler.py:1845`).  When no provider is registered
/// (typical for unit tests), `JIT_EXC_VALUE` stays 0 and Layer 4's
/// `cast_instance_to_gcref(memory_error)` fallback in
/// `compile.py:1095` will fill in the gap.
#[inline]
fn oom_signal_if_zero(result: u64) -> u64 {
    if result == 0 {
        let v = majit_backend::memory_error_singleton_ref();
        if v != 0 {
            // llmodel.py:194-199 `_store_exception` parity — `jit_exc_raise`
            // sets both `JIT_EXC_VALUE` and the typeptr-derived
            // `JIT_EXC_TYPE`, matching the translated `raise MemoryError`
            // sequence inside RPython's `do_malloc_fixedsize_clear`.
            crate::jit_exc_raise(v);
        }
    }
    result
}

/// _build_malloc_slowpath parity: nursery overflow slow path.
///
/// Called from JIT-compiled code when inline nursery bump allocation
/// fails (new_free > nursery_top). total_size includes GcHeader.
///
/// Returns payload pointer (after GcHeader), matching fast-path semantics.
pub extern "C" fn dynasm_nursery_slowpath(total_size: u64) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard
            .as_mut()
            .map(|gc| gc.alloc_nursery(total_size as usize - gc_hdr).0 as u64)
    });
    let ptr = result.unwrap_or_else(|| unsafe {
        // `libc::calloc` returns NULL on real host OOM; preserve that
        // NULL through to the trampoline's `TEST rax, rax; JZ propagate`
        // (assembler.py:300-302).  Adding `gc_hdr` unconditionally
        // masked OOM as a "valid" near-zero pointer and let the JIT
        // continue past the failure.
        let raw = libc::calloc(1, total_size as usize) as u64;
        if raw == 0 { 0 } else { raw + gc_hdr as u64 }
    });
    if crate::majit_log_enabled() {
        eprintln!("[dynasm][nursery-frame] total_size={total_size} payload=0x{ptr:x}");
    }
    ptr
}

/// malloc_cond_varsize_frame slow path for JITFRAME allocation.
///
/// `frame_size` is `jfi_frame_size`: bytes from the JITFRAME payload base
/// through the trailing array, i.e. it excludes the GC header that the
/// allocator prepends internally.
pub extern "C" fn dynasm_nursery_slowpath_jitframe(frame_size: u64) -> u64 {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_mut() {
            if let Some(type_id) = crate::jitframe_gc_type_id() {
                gc.alloc_nursery_typed(type_id, frame_size as usize).0 as u64
            } else {
                gc.alloc_nursery(frame_size as usize).0 as u64
            }
        } else {
            unsafe { libc::calloc(1, frame_size as usize) as u64 }
        }
    })
}

/// _build_malloc_slowpath(kind='var') parity: varsize nursery overflow.
/// Called with (base_size, item_size, length). Returns payload pointer.
pub extern "C" fn dynasm_nursery_slowpath_varsize(
    base_size: u64,
    item_size: u64,
    length: u64,
) -> u64 {
    let gc_hdr = majit_gc::header::GcHeader::SIZE;
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard.as_mut().map(|gc| {
            gc.alloc_varsize(base_size as usize, item_size as usize, length as usize)
                .0 as u64
        })
    });
    result.unwrap_or_else(|| {
        let total = base_size as usize + item_size as usize * length as usize + gc_hdr;
        unsafe {
            // `libc::calloc` returns NULL on real OOM; the previous
            // unconditional `raw + gc_hdr` masked failure as a tiny
            // non-zero "valid" payload pointer and let the JIT continue
            // past the failure.  Mirror `dynasm_nursery_slowpath`'s
            // OOM-null preservation so the caller's TEST/JZ propagate
            // path can fire on real OOM.
            let raw = libc::calloc(1, total) as u64;
            if raw == 0 { 0 } else { raw + gc_hdr as u64 }
        }
    })
}

fn dynasm_raw_varsize_alloc_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let Some(var_bytes) = item_size.checked_mul(length) else {
        return 0;
    };
    let Some(payload_size) = base_size.checked_add(var_bytes) else {
        return 0;
    };
    let Some(total_size) = majit_gc::header::GcHeader::SIZE.checked_add(payload_size) else {
        return 0;
    };
    unsafe {
        let raw = libc::calloc(1, total_size) as *mut u8;
        if raw.is_null() {
            return 0;
        }
        *(raw as *mut majit_gc::header::GcHeader) = majit_gc::header::GcHeader::new(type_id);
        let obj = raw.add(majit_gc::header::GcHeader::SIZE);
        *(obj.add(length_ofs) as *mut usize) = length;
        obj as u64
    }
}

fn dynasm_alloc_varsize_typed_and_set_len(
    type_id: u32,
    base_size: usize,
    item_size: usize,
    length_ofs: usize,
    length: usize,
) -> u64 {
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard.as_mut().map(|gc| {
            let obj = gc.alloc_varsize_typed(type_id, base_size, item_size, length);
            if obj.is_null() {
                0
            } else {
                unsafe {
                    *((obj.0 as *mut u8).add(length_ofs) as *mut usize) = length;
                }
                obj.0 as u64
            }
        })
    });
    result.unwrap_or_else(|| {
        dynasm_raw_varsize_alloc_typed_and_set_len(
            type_id, base_size, item_size, length_ofs, length,
        )
    })
}

pub extern "C" fn dynasm_malloc_array(item_size: u64, type_id: u64, num_elem: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        std::mem::size_of::<usize>(),
        item_size as usize,
        0,
        num_elem as usize,
    ))
}

pub extern "C" fn dynasm_malloc_array_nonstandard(
    base_size: u64,
    item_size: u64,
    length_ofs: u64,
    type_id: u64,
    num_elem: u64,
) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        base_size as usize,
        item_size as usize,
        length_ofs as usize,
        num_elem as usize,
    ))
}

fn dynasm_raw_fixedsize_alloc_typed(type_id: u32, size: usize) -> u64 {
    let Some(total_size) = majit_gc::header::GcHeader::SIZE.checked_add(size) else {
        return 0;
    };
    unsafe {
        let raw = libc::calloc(1, total_size) as *mut u8;
        if raw.is_null() {
            return 0;
        }
        *(raw as *mut majit_gc::header::GcHeader) = majit_gc::header::GcHeader::new(type_id);
        let obj = raw.add(majit_gc::header::GcHeader::SIZE);
        obj as u64
    }
}

fn dynasm_alloc_oldgen_typed_or_raw(type_id: u32, payload_size: usize) -> u64 {
    let result = DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        guard.as_mut().map(|gc| {
            let obj = gc.alloc_oldgen_typed(type_id, payload_size);
            if obj.is_null() { 0 } else { obj.0 as u64 }
        })
    });
    // gc.py:51 contract: helper returns NULL on OOM so
    // CHECK_MEMORY_ERROR can convert it into a MemoryError.  Only
    // `None` (no active runtime — typical for unit tests) falls back
    // to a raw alloc; `Some(0)` (a real OOM from the registered
    // allocator) propagates as 0 unchanged, mirroring
    // `dynasm_nursery_slowpath` above.
    match result {
        None => dynasm_raw_fixedsize_alloc_typed(type_id, payload_size),
        Some(v) => v,
    }
}

/// gc.py:481-490 `malloc_big_fixedsize(size, tid)` — fixed-size object
/// large enough to skip the nursery, allocated directly in the old gen
/// via `do_malloc_fixedsize_clear`.  Header is stamped with the type
/// id so callers MUST NOT emit a separate `gen_initialize_tid`.
pub extern "C" fn dynasm_malloc_big_fixedsize(size: u64, type_id: u64) -> u64 {
    // The CALL_R arg is `total = payload + GcHeader::SIZE` (built by
    // `handle_new` at rewrite.rs:857 to include the GC header).  The
    // runtime allocators (`alloc_oldgen_typed`, `alloc_nursery_typed`)
    // and the raw fallback both prepend the GC header themselves and
    // expect a payload-only size, so subtract `HDR` here once —
    // mirroring `dynasm_nursery_slowpath` above
    // (`alloc_nursery(total_size - gc_hdr)`).
    let payload = (size as usize).saturating_sub(majit_gc::header::GcHeader::SIZE);
    oom_signal_if_zero(dynasm_alloc_oldgen_typed_or_raw(type_id as u32, payload))
}

/// gc.py:460 `malloc_str(length)` — but the upstream closure captures
/// `str_type_id` from `self.str_descr.tid` at generate-time.  `extern
/// "C" fn` cannot capture, so the type id is threaded through the
/// CALL_R as an explicit Signed arg and the calldescr's first param is
/// it (see `make_malloc_str_calldescr`).
pub extern "C" fn dynasm_malloc_str(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_STR_TOKEN_BASE_SIZE,
        1,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// gc.py:469 `malloc_unicode(length)` — see `dynasm_malloc_str` for the
/// closure-vs-extern type-id threading rationale.
pub extern "C" fn dynasm_malloc_unicode(type_id: u64, length: u64) -> u64 {
    oom_signal_if_zero(dynasm_alloc_varsize_typed_and_set_len(
        type_id as u32,
        BUILTIN_UNICODE_TOKEN_BASE_SIZE,
        4,
        BUILTIN_STRING_LEN_OFFSET,
        length as usize,
    ))
}

/// opassembler.py:956-976: non-array write barrier slow path.
/// Calls gc.write_barrier(obj) which is the generic barrier.
pub extern "C" fn dynasm_write_barrier(obj_ptr: u64) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_mut() {
            gc.write_barrier(majit_ir::GcRef(obj_ptr as usize));
        }
    });
}

/// opassembler.py:953-960: array write barrier slow path.
/// Calls jit_remember_young_pointer_from_array(obj) which handles
/// the CARDS_SET transition for HAS_CARDS arrays.
pub extern "C" fn dynasm_write_barrier_from_array(obj_ptr: u64) {
    DYNASM_ACTIVE_GC.with(|cell| {
        let mut guard = cell.borrow_mut();
        if let Some(gc) = guard.as_mut() {
            gc.jit_remember_young_pointer_from_array(majit_ir::GcRef(obj_ptr as usize));
        }
    });
}

/// `_build_frame_realloc_slowpath` parity (assembler.py:143-189):
/// JIT-side helper invoked when `_check_frame_depth` detects that
/// `jf_frame.length < expected_depth`.  Allocates a wider JITFRAME,
/// copies the live slots, threads `jf_forward = new_frame`, and
/// returns the new pointer; the JIT-emitted slowpath body then writes
/// `rbp = rax` so subsequent frame-relative loads/stores land on the
/// reallocated frame.
///
/// `old_jf` and the returned pointer are both libc-allocated jitframes
/// registered with the shadow_stack tracker — matching the runner's
/// `execute_token` allocation strategy (see `register_libc_jitframe`).
///
/// # Safety
/// - `old_jf` must be a live, `register_libc_jitframe`-tracked
///   `*mut JitFrame` (i.e. the running loop/bridge's current frame).
/// - The caller (the JIT-emitted slowpath body) must have already
///   spilled all live registers into the old frame via
///   `_push_all_regs_to_jitframe`; this helper relies on the GC seeing
///   those slots through the gcmap pushed at the same site.
pub unsafe extern "C" fn dynasm_realloc_frame(
    old_jf: *mut JitFrame,
    expected_depth: isize,
) -> *mut JitFrame {
    let base_ofs = DynasmBackend::get_baseofs_of_frame_field() as isize;
    let new_jf = unsafe {
        majit_backend::jitframe::realloc_frame(
            old_jf,
            expected_depth,
            base_ofs,
            // `alloc`: libc::calloc to match the runner-allocated frame.
            |size_bytes| libc::calloc(1, size_bytes as usize) as *mut JitFrame,
            // `write_barrier`: register the new frame with the shadow
            // stack tracer.  The old frame stays registered for the
            // duration of the running call; `jf_forward` is followed
            // by the collector through `jitframe.py:111-118`.
            |new_jf| {
                majit_gc::shadow_stack::register_libc_jitframe(new_jf as usize);
            },
        )
    };
    if crate::majit_log_enabled() {
        eprintln!(
            "[dynasm][realloc-frame] old={old_jf:p} new={new_jf:p} expected_depth={expected_depth}"
        );
    }
    new_jf
}

/// runner.py:23 AbstractX86CPU — concrete Backend implementation.
pub struct DynasmBackend {
    /// Next unique trace ID.
    next_trace_id: u64,
    /// Next header PC (green key).
    next_header_pc: u64,
    /// Constants for the next compilation.
    constants: std::collections::HashMap<u32, i64>,
    /// Constant type annotations for GC rewriter.
    constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    /// llmodel.py:64-69 self.vtable_offset — byte offset of the typeptr
    /// field inside instance objects. None when gcremovetypeptr is enabled.
    vtable_offset: Option<usize>,
    /// `compile.py:665` `setattr(cpu, name, descr)` per-cpu attachments,
    /// held in a heap-pinned `Arc<RwLock<CpuDescrAttachments>>` so the
    /// pointer baked into the CALL_ASSEMBLER helper call site
    /// (`compile_loop` / `compile_bridge`) stays valid even when
    /// `DynasmBackend` is moved (the metainterp stores it by value; tests
    /// hold stack-local `DynasmBackend::new()`).  Compiled traces clone
    /// the `Arc` into `CompiledCode` so the attachments outlive the
    /// owning backend — matches the lifetime guarantee RPython gets from
    /// `cpu` being a long-lived Python object.
    descr_attachments: crate::guard::CpuDescrHandle,
    /// ptr → `DescrRef` registry enabling cross-token resolution of
    /// guard fail descriptors. RPython resolves
    /// `AbstractDescr.show(jf_descr)` via direct pointer dereference,
    /// so the lookup naturally crosses loop/bridge boundaries. Pyre
    /// keeps each descr as a `DescrRef` and stores them in per-token
    /// `asmmemmgr_blocks`, so a bridge that JUMPs into another
    /// compiled loop can leave the runtime holding a jf_descr whose
    /// owning token is not the one currently executing. This registry
    /// is the ptr-indexed view needed to complete that lookup.
    fail_descr_registry:
        Arc<std::sync::Mutex<std::collections::HashMap<usize, majit_ir::DescrRef>>>,
    /// Backend-internal side-table mapping a source guard descr's
    /// `Arc::as_ptr` address to the entry pointer of the compiled bridge
    /// patched in for that guard.  PyPy's `AbstractFailDescr._attrs_`
    /// (`history.py:132`) carries no `bridge_addr` slot; once a bridge is
    /// patched in, RPython relies on the in-place machine-code JMP and
    /// recovers structural state from `asmmemmgr_blocks`.  Pyre's
    /// metainterp queries `store_bridge_guard_hashes` /
    /// `compiled_bridge_fail_descr_layouts` need to walk back from a
    /// source descr to its bridge `CompiledCode`; this table is the
    /// indirection that lets us do that without polluting the descr.
    bridge_addr_by_descr: Arc<std::sync::Mutex<std::collections::HashMap<usize, usize>>>,
    /// Arch-specific per-CPU state PyPy keeps on `Assembler386` /
    /// `AssemblerARM64` (e.g. `self.malloc_slowpath`,
    /// `self.propagate_exception_path` at `assembler.py:63,344` and
    /// `aarch64/assembler.py:577`).  PyPy's assembler is one-per-CPU;
    /// pyre's `Asm` is per-`compile_loop`/`compile_bridge`, so the
    /// per-CPU stash lives here instead.  See
    /// `crate::x86::cpu_ext::X86CpuExt` /
    /// `crate::aarch64::cpu_ext::Aarch64CpuExt`.
    arch_cpu_ext: ArchCpuExt,
}

impl DynasmBackend {
    #[inline]
    fn raw_mem_ptr(addr: i64, offset: i64) -> usize {
        assert_ne!(
            addr, 0,
            "llmodel.py parity: raw memory helpers must not silently accept NULL addresses"
        );
        (addr as usize).wrapping_add(offset as usize)
    }

    /// llmodel.py:467-478 read_int_at_mem(gcref, ofs, size, sign).
    fn read_int_at_mem(&self, addr: i64, offset: i64, size: usize, sign: bool) -> i64 {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe {
            match (size, sign) {
                (1, true) => (ptr as *const i8).read_unaligned() as i64,
                (1, false) => (ptr as *const u8).read_unaligned() as i64,
                (2, true) => (ptr as *const i16).read_unaligned() as i64,
                (2, false) => (ptr as *const u16).read_unaligned() as i64,
                (4, true) => (ptr as *const i32).read_unaligned() as i64,
                (4, false) => (ptr as *const u32).read_unaligned() as i64,
                _ => (ptr as *const i64).read_unaligned(),
            }
        }
    }

    /// llmodel.py:481-488 write_int_at_mem(gcref, ofs, size, newvalue).
    fn write_int_at_mem(&self, addr: i64, offset: i64, size: usize, newvalue: i64) {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe {
            match size {
                1 => (ptr as *mut u8).write_unaligned(newvalue as u8),
                2 => (ptr as *mut u16).write_unaligned(newvalue as u16),
                4 => (ptr as *mut u32).write_unaligned(newvalue as u32),
                _ => (ptr as *mut i64).write_unaligned(newvalue),
            }
        }
    }

    /// llmodel.py:490-491 read_float_at_mem(gcref, ofs).
    fn read_float_at_mem(&self, addr: i64, offset: i64) -> f64 {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe { (ptr as *const f64).read_unaligned() }
    }

    /// llmodel.py:493-494 write_float_at_mem(gcref, ofs, newvalue).
    fn write_float_at_mem(&self, addr: i64, offset: i64, newvalue: f64) {
        let ptr = Self::raw_mem_ptr(addr, offset);
        unsafe { (ptr as *mut f64).write_unaligned(newvalue) }
    }

    pub fn new() -> Self {
        // `rpython/jit/backend/model.py` `AbstractCPU.__init__` parity:
        // the cpu is constructed with no attached descrs.  The
        // `DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`
        // singletons are attached later by
        // `compile.make_and_attach_done_descrs([self, cpu])` during
        // `MetaInterpStaticData.finish_setup` (pyjitpl.py:2222).
        DynasmBackend {
            next_trace_id: 1,
            next_header_pc: 0,
            constants: std::collections::HashMap::new(),
            constant_types: std::collections::HashMap::new(),
            vtable_offset: None,
            descr_attachments: Arc::new(std::sync::RwLock::new(
                crate::guard::CpuDescrAttachments::default(),
            )),
            fail_descr_registry: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            bridge_addr_by_descr: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            arch_cpu_ext: ArchCpuExt::new(),
        }
    }

    /// Bridge entry-pointer registration keyed on the source guard
    /// descr's `Arc::as_ptr` address.  Called from `compile_bridge`
    /// immediately after `patch_jump_for_descr` redirects the source
    /// guard at `runner.rs::compile_bridge` to point at the freshly
    /// compiled bridge.
    pub fn register_bridge_addr(&self, source_descr_ptr: usize, bridge_addr: usize) {
        self.bridge_addr_by_descr
            .lock()
            .expect("bridge_addr_by_descr mutex poisoned")
            .insert(source_descr_ptr, bridge_addr);
    }

    /// Bridge entry-pointer lookup by source descr `Arc::as_ptr` address.
    /// Returns `0` when no bridge has been registered for the source
    /// (PyPy parity: `assembler.py` treats `adr_jump_offset == 0`
    /// uniformly as "patched / no entry").
    pub fn lookup_bridge_addr(&self, source_descr_ptr: usize) -> usize {
        self.bridge_addr_by_descr
            .lock()
            .expect("bridge_addr_by_descr mutex poisoned")
            .get(&source_descr_ptr)
            .copied()
            .unwrap_or(0)
    }

    /// Test helper: attach synthetic per-cpu `DoneWithThisFrame*` +
    /// `ExitFrameWithExceptionDescrRef` descrs, mirroring the state
    /// `MetaInterpStaticData.attach_descrs_to_cpu(cpu)` leaves the
    /// backend in at `finish_setup` (pyjitpl.py:2222).  Production
    /// code reaches this state through `MetaInterp::new`; backend-
    /// only unit/integration tests that skip the metainterp call
    /// this to get a populated cpu before running `compile_loop`.
    pub fn attach_default_test_descrs(&mut self) {
        // `compile.py:665-674 make_and_attach_done_descrs` +
        // `pyjitpl.py:2283` `self.cpu.propagate_exception_descr = exc_descr`
        // parity: attach the class-distinct DoneWithThisFrameDescr* /
        // ExitFrameWithExceptionDescrRef plus a PropagateExceptionDescr
        // stand-in that the metainterp would mint through
        // `MetaInterp::new` / `MetaInterpStaticData.finish_setup`.
        // Backend-only tests that skip the metainterp call this to land
        // the same descrs the runtime classifier expects — and so that
        // `X86CpuExt::ensure_propagate_exception_path` can bake a
        // non-zero descr pointer into the propagate trampoline
        // (matching PyPy's `setup_once` ordering, which builds
        // trampolines after `finish_setup` has installed every CPU
        // descr).
        let void: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrVoid::new());
        let int: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrInt::new());
        let r: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrRef::new());
        let float: majit_ir::DescrRef = Arc::new(majit_backend::DoneWithThisFrameDescrFloat::new());
        let exit_exc: majit_ir::DescrRef =
            Arc::new(majit_backend::ExitFrameWithExceptionDescrRef::new());
        // `compile.py:712 PropagateExceptionDescr` parity: backend-only
        // tests still need the same descr class identity that production
        // `MetaInterpStaticData.finish_setup` installs.
        let propagate: majit_ir::DescrRef = Arc::new(majit_backend::PropagateExceptionDescr::new());
        <Self as Backend>::set_done_with_this_frame_descr_void(self, void);
        <Self as Backend>::set_done_with_this_frame_descr_int(self, int);
        <Self as Backend>::set_done_with_this_frame_descr_ref(self, r);
        <Self as Backend>::set_done_with_this_frame_descr_float(self, float);
        <Self as Backend>::set_exit_frame_with_exception_descr_ref(self, exit_exc);
        <Self as Backend>::set_propagate_exception_descr(self, propagate);
        // `pyjitpl.py:2297 self.cpu.setup_once()` parity — production
        // reaches `cpu.setup_once()` via `MetaInterpStaticData::_setup_once`
        // (`pyjitpl.py:2292-2303`) on first JIT entry, AFTER every descr
        // setter has run.  Backend-only tests bypass the metainterp gate,
        // so call `setup_once` here directly once the descrs are in place
        // — analogous to PyPy test helpers that explicitly call
        // `cpu.setup_once()` after manual descr attachment (e.g.
        // `rpython/jit/backend/ppc/test/test_regalloc_3.py:10`).
        <Self as Backend>::setup_once(self);
    }

    /// Active vtable_offset for the assembler to consume during codegen.
    pub fn vtable_offset(&self) -> Option<usize> {
        self.vtable_offset
    }

    /// `compile.py:665-674` `make_and_attach_done_descrs` parity: expose
    /// the six per-cpu-instance descrs as raw pointers for emission
    /// consumers (Assembler386 / AssemblerARM64 FINISH + CALL_ASSEMBLER
    /// sites).  The metainterp attaches the real descrs through
    /// `Backend::set_done_with_this_frame_descr_*` during
    /// `MetaInterpStaticData.finish_setup` (pyjitpl.py:2222); before that
    /// the per-cpu fallback descrs installed by `DynasmBackend::new()`
    /// answer, so backend-only integration tests see distinct, non-zero
    /// pointers per result type without ever consulting per-thread state.
    pub(crate) fn attached_descr_ptrs(&self) -> crate::guard::AttachedDescrPtrs {
        self.descr_attachments.read().unwrap().descr_ptrs()
    }

    /// `Arc` clone of the attachment handle, for compiled traces to
    /// keep alive alongside their executable buffer.  The `Arc`'s
    /// payload (the `RwLock<CpuDescrAttachments>`) lives at a heap-
    /// pinned address; `Arc::as_ptr(&clone)` is baked by emission into
    /// the CALL_ASSEMBLER helper call site as a compile-time immediate
    /// (same role as RPython's `self.cpu` closure capture in the
    /// translated code).  Cloning into `CompiledCode` keeps the pointee
    /// alive past any subsequent `DynasmBackend` drop — matches the
    /// lifetime guarantee RPython gets from `cpu` being a long-lived
    /// Python object.
    pub(crate) fn cpu_handle(&self) -> crate::guard::CpuDescrHandle {
        Arc::clone(&self.descr_attachments)
    }

    /// Add a newly-compiled loop/bridge's fail_descrs to the ptr-indexed
    /// registry. Called after every `compile_loop` / `compile_bridge`
    /// returns so subsequent `resolve_latest_descr` lookups can cross
    /// token boundaries — required when a bridge JUMPs into another
    /// compiled loop and that loop's guard fires before control returns
    /// to the bridge's owning token.
    /// Register `DescrRef` instances into the addr→`DescrRef` lookup
    /// table.  Called from `compile_loop` / `compile_bridge` to
    /// populate after codegen.  Re-running is idempotent: existing
    /// entries are preserved via `entry().or_insert_with`.
    pub fn register_fail_descrs(&self, descrs: &[majit_ir::DescrRef]) {
        let mut reg = self.fail_descr_registry.lock().unwrap();
        for descr_ref in descrs {
            let ptr = Arc::as_ptr(descr_ref) as *const () as usize;
            reg.entry(ptr).or_insert_with(|| descr_ref.clone());
            // Mirror into the trampoline-reachable global registry as a
            // `Weak<dyn Descr>` so the JIT helper trampoline can recover
            // the descr identity without going through a backend instance.
            crate::guard::register_fail_descr_global(ptr, descr_ref);
        }
    }

    // `set_constants`, `set_constant_types`, `set_next_trace_id`,
    // `set_next_header_pc` are provided via the `Backend` trait impl
    // below so `compile_tmp_callback` and other backend-agnostic
    // consumers can reach them through `&mut dyn Backend`.

    /// gc.py:525-531 parity: build a GcRewriterImpl from the active GC.
    fn gc_rewriter(
        &self,
        constant_types: &std::collections::HashMap<u32, majit_ir::Type>,
    ) -> Option<majit_gc::rewrite::GcRewriterImpl> {
        with_dynasm_active_gc(|gc| {
            let ct = constant_types.clone();
            majit_gc::rewrite::GcRewriterImpl {
                nursery_free_addr: gc.nursery_free_addr(),
                nursery_top_addr: gc.nursery_top_addr(),
                max_nursery_size: gc.max_nursery_object_size(),
                wb_descr: {
                    let mut descr = majit_gc::WriteBarrierDescr::for_current_gc();
                    let card_page_shift = gc.card_page_shift();
                    if card_page_shift > 0 {
                        descr.jit_wb_card_page_shift = card_page_shift;
                    } else {
                        descr.jit_wb_cards_set = 0;
                        descr.jit_wb_card_page_shift = 0;
                        descr.jit_wb_cards_set_byteofs = 0;
                        descr.jit_wb_cards_set_singlebyte = 0;
                    }
                    descr
                },
                jitframe_info: crate::jitframe_layout().and_then(|info| info.jitframe_descrs),
                constant_types: ct,
                // rewrite.py:673 — read compiled_loop_token._ll_initial_locs +
                // ptr2int(compiled_loop_token.frame_info), both sourced from the
                // CLT Arc on the registered DynasmCaTarget (model.py:292-338).
                call_assembler_callee_locs: Some(Box::new(|token_number| {
                    lookup_call_assembler_callee_locs(token_number)
                })),
                // x86/runner.py:31 `load_supported_factors = (1, 2, 4, 8)`
                // vs llmodel.py:39 default `(1,)` used by the aarch64
                // backend (which has no scaled store addressing mode and
                // asserts boxes[3].getint() == 1 in its regalloc — see
                // `consider_gc_store_indexed` cfg(target_arch = "aarch64")).
                load_supported_factors: gc_store_supported_factors(),
                // x86/runner.py:22 overrides model.py:22 base default
                // `False` to `True`.  aarch64/runner.py inherits the
                // base `False`, so the rewriter expands LEA to
                // INT_LSHIFT + INT_ADD + INT_ADD (rewrite.py:1089-1098)
                // even though the aarch64 backend has a native
                // `genop_load_effective_address` lowering.
                supports_load_effective_address: supports_load_effective_address(),
                // nursery.rs:68 `alloc_zeroed` + nursery.rs:105-110
                // `reset` memset-to-zero on recycle mean the nursery
                // payload is always zero-filled at allocation time;
                // `clear_gc_fields` short-circuits per rewrite.py:499-500.
                malloc_zero_filled: true,
                // gc.py:39 `self.memcpy_fn = memcpy_fn` cast through
                // `cast_ptr_to_adr` + `cast_adr_to_int` (rewrite.py:1046-1047).
                memcpy_fn: majit_ir::memcpy_fn_addr(),
                // gc.py:40-43 `self.memcpy_descr = get_call_descr(...)`.
                memcpy_descr: majit_ir::make_memcpy_calldescr(),
                // gc.py:46 `self.str_descr = get_array_descr(self, rstr.STR)`.
                str_descr: builtin_string_array_descr(majit_ir::OpCode::Newstr)
                    .expect("Newstr must produce a str ArrayDescr"),
                // gc.py:47 `self.unicode_descr = get_array_descr(self, rstr.UNICODE)`.
                unicode_descr: builtin_string_array_descr(majit_ir::OpCode::Newunicode)
                    .expect("Newunicode must produce a unicode ArrayDescr"),
                // gc.py:48 `self.str_hash_descr = get_field_descr(self, rstr.STR, 'hash')`.
                str_hash_descr: builtin_string_hash_field_descr(majit_ir::OpCode::Strhash)
                    .expect("Strhash must produce a str hash FieldDescr"),
                // gc.py:49 `self.unicode_hash_descr = get_field_descr(self, rstr.UNICODE, 'hash')`.
                unicode_hash_descr: builtin_string_hash_field_descr(majit_ir::OpCode::Unicodehash)
                    .expect("Unicodehash must produce a unicode hash FieldDescr"),
                // gc.py:33-37 `self.fielddescr_vtable = get_field_descr(
                // self, rclass.OBJECT, 'typeptr')`.  pyre always emits
                // a typeptr slot (no `gcremovetypeptr` build), so we
                // install Some unconditionally.
                fielddescr_vtable: Some(majit_ir::make_vtable_field_descr()),
                // gc.py:394 `self.fielddescr_tid = get_field_descr(self,
                // self.GCClass.HDR, 'tid')` — framework GC.  pyre's GC
                // is always framework-style; gen_initialize_tid translates
                // the descr's offset by `-HDR_SIZE` because pyre's HDR
                // sits before the object pointer.
                fielddescr_tid: Some(majit_ir::make_tid_field_descr()),
                malloc_array_fn: dynasm_malloc_array as *const () as i64,
                malloc_array_nonstandard_fn: dynasm_malloc_array_nonstandard as *const () as i64,
                malloc_str_fn: dynasm_malloc_str as *const () as i64,
                malloc_unicode_fn: dynasm_malloc_unicode as *const () as i64,
                malloc_big_fixedsize_fn: dynasm_malloc_big_fixedsize as *const () as i64,
                malloc_array_descr: majit_ir::make_malloc_array_calldescr(),
                malloc_array_nonstandard_descr: majit_ir::make_malloc_array_nonstandard_calldescr(),
                malloc_str_descr: majit_ir::make_malloc_str_calldescr(),
                malloc_unicode_descr: majit_ir::make_malloc_unicode_calldescr(),
                malloc_big_fixedsize_descr: majit_ir::make_malloc_big_fixedsize_calldescr(),
                standard_array_basesize: std::mem::size_of::<usize>(),
                standard_array_length_ofs: 0,
            }
        })
    }

    /// rewrite.py:345 parity: run GC rewriter on ops before assembly.
    fn prepare_ops_for_compile(&mut self, inputargs: &[InputArg], ops: &[Op]) -> Vec<Op> {
        let num_inputs = inputargs.len() as u32;
        let mut normalized: Vec<Op> = ops
            .iter()
            .enumerate()
            .map(|(op_idx, op)| {
                let mut n = op.clone();
                if n.result_type() != Type::Void && n.pos.is_none() {
                    let pos = num_inputs + op_idx as u32;
                    n.pos = match n.result_type() {
                        Type::Int => OpRef::int_op(pos),
                        Type::Float => OpRef::float_op(pos),
                        Type::Ref => OpRef::ref_op(pos),
                        Type::Void => unreachable!("filtered above"),
                    };
                }
                n
            })
            .collect();
        // rewrite.py:489 parity: inject str_descr/unicode_descr for NEWSTR/NEWUNICODE
        inject_builtin_string_descrs(&mut normalized);
        // Clone so self.constant_types remains populated for later
        // backend calls. rewrite.py:930 `v.type` — RPython Box carries
        // its type on the object itself, so InputArg / ResOperation /
        // Const all return the same `.type` attribute. Pyre's flat
        // `OpRef` stores types in side maps keyed by the raw u32;
        // `constant_types` already carries op-position and constant
        // types (disjoint key spaces via `OpRef::CONST_BIT = 1 << 31`).
        //
        // Each InputArg's raw OpRef value lives at `InputArg.index`:
        // top-level loops occupy `[0, num_inputs)`, while Phase E.2b
        // bridges occupy `[bridge_inputarg_base..)`. Key the merged
        // map by `ia.index` (NOT a dense `enumerate` index) so the
        // rewriter's `v.type` lookup matches the OpRef values that
        // ops reference, regardless of namespace — RPython relies on
        // Box identity here, pyre's line-by-line analog is `arg.0 ==
        // inputargs[i].index`. Mirrors the Cranelift backend's
        // `constant_types_with_inputargs` build at compiler.rs:7042.
        let mut constant_types = self.constant_types.clone();
        for ia in inputargs.iter() {
            constant_types.entry(ia.index).or_insert(ia.tp);
        }
        if let Some(rewriter) = self.gc_rewriter(&constant_types) {
            use majit_gc::GcRewriter;
            let constants = &self.constants;
            let (result, new_constants, new_constant_types) =
                rewriter.rewrite_for_gc_with_constants(&normalized, constants);
            for (k, v) in new_constants {
                self.constants.entry(k).or_insert(v);
            }
            for (k, tp) in new_constant_types {
                // rewrite.py creates fresh ConstInt boxes for sizes, offsets
                // and helper addresses. RPython stores the type on each
                // ConstInt object; pyre imports the rewriter's explicit
                // side-channel type entry instead of guessing from the raw
                // constant key.
                self.constant_types.entry(k).or_insert(tp);
            }
            result
        } else {
            normalized
        }
    }

    /// llmodel.py:53-54: store gc_ll_descr on the cpu instance.
    ///
    /// Dynasm does not have cranelift's runtime-id indirection, so it
    /// mirrors wasm: the live allocator is stored in a thread-local and
    /// exposed through backend-agnostic `majit_gc::ActiveGcGuardHooks`.
    pub fn set_gc_allocator(&mut self, mut gc: Box<dyn majit_gc::GcAllocator>) {
        gc.freeze_types();
        let supports_guard_gc_type = gc.supports_guard_gc_type();
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            // Drop the previous allocator first so reentrant
            // `dynasm_gc_owns_object` queries from its drop body still
            // resolve old-heap addresses through the raw mirror — it
            // keeps pointing at the live old box throughout the drop
            // body. Publishing the new raw pointer before the drop
            // would route those queries to the new allocator, which
            // does not know about old-heap addresses. After the drop
            // returns no further reentry is possible on this thread
            // before the raw mirror is republished synchronously below.
            *guard = Some(gc);
            let raw = guard
                .as_deref_mut()
                .map(|gc| gc as *mut dyn majit_gc::GcAllocator);
            DYNASM_ACTIVE_GC_RAW.with(|raw_cell| raw_cell.set(raw));
        });
        majit_gc::set_active_gc_guard_hooks(majit_gc::ActiveGcGuardHooks {
            check_is_object: Some(dynasm_check_is_object),
            get_actual_typeid: Some(dynasm_get_actual_typeid),
            subclass_range: Some(dynasm_subclass_range),
            typeid_subclass_range: Some(dynasm_typeid_subclass_range),
            typeid_is_object: Some(dynasm_typeid_is_object),
            supports_guard_gc_type,
        });
        majit_gc::set_active_alloc_nursery_typed(Some(dynasm_alloc_nursery_typed));
        majit_gc::set_active_alloc_oldgen_typed(Some(dynasm_alloc_oldgen_typed));
        majit_gc::set_active_root_hooks(Some(dynasm_gc_add_root), Some(dynasm_gc_remove_root));
        majit_gc::set_active_gc_owns_object(Some(dynasm_gc_owns_object));
        majit_gc::set_active_write_barrier(Some(dynasm_gc_write_barrier));
    }

    /// llmodel.py:64-69 self.vtable_offset configuration.
    pub fn set_vtable_offset(&mut self, offset: Option<usize>) {
        self.vtable_offset = offset;
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer to its registered GC type id via the
    /// installed gc_ll_descr (the GC backend supplied through
    /// set_gc_allocator).
    pub fn lookup_typeid_from_classptr(&self, classptr: usize) -> Option<u32> {
        with_dynasm_active_gc(|gc| gc.get_typeid_from_classptr_if_gcremovetypeptr(classptr))
            .flatten()
    }

    /// Pre-fetch the GC TYPE_INFO constants that RPython's assembler reads
    /// from `cpu.gc_ll_descr` while emitting `GUARD_IS_OBJECT` and
    /// `GUARD_SUBCLASS`.
    fn collect_guard_gc_type_info(&self) -> Option<GuardGcTypeInfo> {
        with_dynasm_active_gc(|gc| {
            if !gc.supports_guard_gc_type() {
                return None;
            }
            let (base_type_info, shift_by, sizeof_ti) = gc.get_translated_info_for_typeinfo();
            let (infobits_offset, is_object_flag) = gc.get_translated_info_for_guard_is_object();
            Some(GuardGcTypeInfo {
                base_type_info,
                shift_by,
                sizeof_ti,
                infobits_offset,
                is_object_flag,
                subclassrange_min_offset: gc.subclassrange_min_offset(),
            })
        })
        .flatten()
    }

    /// Pre-compute classptr → expected_typeid pairs for every GuardClass /
    /// GuardNonnullClass operand seen in `ops`. RPython resolves these on
    /// demand inside `_cmp_guard_class` (assembler.py:1887-1890); pyre's
    /// dynasm assembler runs without a borrow of `self`, so we materialize
    /// the resolver as a HashMap up front.
    fn collect_classptr_typeid_table(
        &self,
        ops: &[Op],
        constants: &std::collections::HashMap<u32, i64>,
    ) -> std::collections::HashMap<i64, u32> {
        let mut table = std::collections::HashMap::new();
        if self.vtable_offset.is_some() || DYNASM_ACTIVE_GC.with(|cell| cell.borrow().is_none()) {
            // vtable_offset path doesn't need typeid lookups; without a
            // gc_ll_descr there is nothing to resolve anyway.
            return table;
        }
        for op in ops {
            if matches!(
                op.opcode,
                majit_ir::OpCode::GuardClass | majit_ir::OpCode::GuardNonnullClass
            ) && op.args.len() >= 2
            {
                if let Some(&classptr) = constants.get(&op.args[1].raw()) {
                    if let Some(tid) = self.lookup_typeid_from_classptr(classptr as usize) {
                        table.insert(classptr, tid);
                    }
                }
            }
        }
        table
    }

    /// Pre-compute classptr → `(subclassrange_min, subclassrange_max)` for
    /// every constant `GuardSubclass` expected-class operand.
    ///
    /// RPython reads these fields from `loc_check_against_class.getint()` at
    /// codegen time (`x86/assembler.py:1971-1974`). This table is the
    /// smallest dynasm-side equivalent of that object-field read; it is not a
    /// per-box side table.
    fn collect_classptr_subclass_range_table(
        &self,
        ops: &[Op],
        constants: &std::collections::HashMap<u32, i64>,
    ) -> std::collections::HashMap<i64, (i64, i64)> {
        let mut table = std::collections::HashMap::new();
        if DYNASM_ACTIVE_GC.with(|cell| cell.borrow().is_none()) {
            return table;
        }
        for op in ops {
            if op.opcode == majit_ir::OpCode::GuardSubclass && op.args.len() >= 2 {
                if let Some(&classptr) = constants.get(&op.args[1].raw()) {
                    if let Some(range) =
                        with_dynasm_active_gc(|gc| gc.subclass_range(classptr as usize)).flatten()
                    {
                        table.insert(classptr, range);
                    }
                }
            }
        }
        table
    }

    fn get_compiled(token: &JitCellToken) -> &CompiledCode {
        token
            .compiled
            .as_ref()
            .expect("token has no compiled code")
            .downcast_ref::<CompiledCode>()
            .expect("compiled data is not CompiledCode")
    }

    fn input_slot(position: usize) -> usize {
        arch::JITFRAME_FIXED_SIZE + position
    }

    /// llsupport/regalloc.py:861-871 `_set_initial_bindings` parity:
    /// `_ll_initial_locs` stores `loc.value - base_ofs`, measured in bytes
    /// from `FIRST_ITEM_OFFSET`, not input-order slot numbers.
    fn input_initial_loc(position: usize) -> i32 {
        (Self::input_slot(position) * crate::jitframe::SIZEOFSIGNED) as i32
    }

    /// llmodel.py:412 get_latest_descr parity: resolve a raw jf_descr
    /// pointer to its `DescrRef`. Searches root loop fail_descrs
    /// first, then all bridge fail_descrs stored in asmmemmgr_blocks.
    /// RPython does this via AbstractDescr.show() which works for any
    /// descr from any loop/bridge.
    ///
    /// `compile.py:618-671` parity: the four `DoneWithThisFrame*`
    /// + `ExitFrameWithExceptionDescrRef` singletons attached to
    /// `self.cpu` are compared by pointer identity against the raw
    /// `jf_descr` value — same as RPython
    /// `llgraph/runner.py:1478-1484` (`faildescr == self.cpu.done_with_this_frame_descr_*`).
    ///
    /// Panics if not found — RPython uses object identity, so lookup
    /// failure is impossible in well-formed execution.
    ///
    /// `frame_ptr` is required so the `propagate_exception_descr` arm
    /// can run the equivalent of `compile.py:1092-1098`'s
    /// `cpu.grab_exc_value(deadframe)` — read+clear `jf_guard_exc` and
    /// stage the value into `jf_frame[0]` before synthesizing the
    /// exit-frame-with-exception descr the toplevel consumer expects.
    fn find_descr_by_ptr(
        &self,
        token: &JitCellToken,
        ptr: usize,
        frame_ptr: *mut JitFrame,
    ) -> majit_ir::DescrRef {
        let attached = self.attached_descr_ptrs();
        // compile.py:618-669 done_with_this_frame_descr — check all 4 variants.
        // Forward through `meta_descr` so the metainterp class hierarchy
        // (DoneWithThisFrameDescr{Void,Int,Ref,Float}) answers
        // `is_finish` / `fail_arg_types` etc. via `compile.py:624 final_descr=True`.
        if ptr != 0
            && (ptr == attached.done_with_this_frame_descr_void
                || ptr == attached.done_with_this_frame_descr_int
                || ptr == attached.done_with_this_frame_descr_ref
                || ptr == attached.done_with_this_frame_descr_float)
        {
            // Return the metainterp `DoneWithThisFrameDescr*` Arc directly.
            // `compile.py:618-672` class hierarchy answers
            // `is_finish`/`fail_arg_types` via its own FailDescr impl —
            // no backend wrapper needed (Phase C-1 cascade endpoint).
            let att = self.descr_attachments.read().unwrap();
            let meta = if ptr == attached.done_with_this_frame_descr_void {
                att.done_with_this_frame_descr_void.clone()
            } else if ptr == attached.done_with_this_frame_descr_float {
                att.done_with_this_frame_descr_float.clone()
            } else if ptr == attached.done_with_this_frame_descr_ref {
                att.done_with_this_frame_descr_ref.clone()
            } else {
                att.done_with_this_frame_descr_int.clone()
            };
            return meta.expect(
                "matched Done*WithThisFrameDescr ptr but slot is unattached — \
                 attach_default_test_descrs / MetaInterp::new must have installed it",
            );
        }

        // compile.py:658-662 ExitFrameWithExceptionDescrRef — return the
        // attached metainterp Arc directly; its class identity carries
        // is_exit_frame_with_exception()=true.
        if ptr != 0 && ptr == attached.exit_frame_with_exception_descr_ref {
            return self
                .descr_attachments
                .read()
                .unwrap()
                .exit_frame_with_exception_descr_ref
                .clone()
                .expect("matched exit_frame_with_exception ptr but slot is unattached");
        }

        // pyjitpl.py:2283 propagate_exception_descr — stamped into
        // jf_descr by the inline propagate path emitted at
        // OpCode::CheckMemoryError when a malloc helper returns NULL.
        //
        // compile.py:1092-1098 `PropagateExceptionDescr.handle_fail`:
        //     exception = cpu.grab_exc_value(deadframe)
        //     if not exception:
        //         exception = cast_instance_to_gcref(memory_error)
        //     raise jitexc.ExitFrameWithExceptionRef(exception)
        //
        // pyre routes this through the same FailDescr the toplevel
        // consumer (eval.rs:2564, 3166) already understands —
        // `is_exit_frame_with_exception=true` + `fail_arg_types=[Ref]`
        // — by transferring `jf_guard_exc` (set by Layer 3's emitted
        // _store_and_reset_exception, x86/assembler.rs:2304-...) into
        // `jf_frame[0]` so the existing slot-0 Ref reader picks it up.
        if ptr != 0 && ptr == attached.propagate_exception_descr {
            // grab_exc_value: read+clear jf_guard_exc.
            let exc_val = unsafe {
                let slot = &mut (*frame_ptr).jf_guard_exc;
                let v = *slot;
                *slot = 0;
                v
            };
            // memory_error fallback (compile.py:1095) for the unlikely
            // case where the propagate path fired without Layer 1's
            // singleton store winning the race (or the singleton
            // provider was never registered — unit tests).
            let exc_val = if exc_val != 0 {
                exc_val as i64
            } else {
                majit_backend::memory_error_singleton_ref()
            };
            // Stage into jf_frame[0] so EXIT_FRAME_WITH_EXCEPTION
            // dispatch reads the exc value through the standard
            // get_ref_value(0) path (compile.py:660).
            unsafe { crate::llmodel::set_int_value(frame_ptr, 0, exc_val as isize) };
            // `compile.py:1092-1098 PropagateExceptionDescr.handle_fail`
            // raises `jitexc.ExitFrameWithExceptionRef(exception)`.  pyre's
            // flat dispatcher reads `is_exit_frame_with_exception()`, so
            // return the ExitFrameWithExceptionDescrRef metainterp Arc
            // directly (its class identity answers the predicate true).
            return self
                .descr_attachments
                .read()
                .unwrap()
                .exit_frame_with_exception_descr_ref
                .clone()
                .expect(
                    "Propagate path requires exit_frame_with_exception_descr_ref to be attached",
                );
        }

        // Search root loop
        let compiled = Self::get_compiled(token);
        if let Some(found) = compiled
            .fail_descrs
            .iter()
            .find(|d| Arc::as_ptr(d) as *const () as usize == ptr)
        {
            return found.clone();
        }

        // Search bridge fail_descrs in asmmemmgr_blocks
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if let Some(found) = bridge
                    .fail_descrs
                    .iter()
                    .find(|d| Arc::as_ptr(d) as *const () as usize == ptr)
                {
                    return found.clone();
                }
            }
        }
        drop(blocks);

        // Cross-token fallback: a bridge attached to loop A may JUMP into
        // loop B's body. When B's guard fires, the jf_descr ptr identifies
        // a fail descr owned by B (or by a bridge attached to B), but the
        // currently-executing `token` is still A. RPython's
        // `AbstractDescr.show(jf_descr)` dereferences the pointer directly,
        // so the lookup is inherently global; pyre emulates that with the
        // per-backend ptr-indexed registry populated by `compile_loop` /
        // `compile_bridge`.
        if let Some(found) = self.fail_descr_registry.lock().unwrap().get(&ptr) {
            return found.clone();
        }

        panic!(
            "find_descr_by_ptr: jf_descr {:#x} not found in root loop, \
             bridges, or ptr registry — RPython equivalent \
             (AbstractDescr.show) never fails",
            ptr
        );
    }

    /// Find a descr by (trace_id, fail_index) across root loop + all
    /// bridges. Used by compile_bridge to locate the exact guard descr
    /// that failed — RPython passes the faildescr object directly.
    ///
    /// Panics if not found — in RPython, the faildescr is the exact
    /// object, so there is no lookup-miss path. Use `try_find_descr` for
    /// query-style callers (e.g. `bridge_was_compiled` /
    /// `compiled_bridge_fail_descr_layouts`) that legitimately probe
    /// "is this guard already compiled?" and must treat the miss as
    /// `None` (matching cranelift's `?`-on-miss semantics in
    /// `compiler.rs:11723`).
    fn find_descr(token: &JitCellToken, trace_id: u64, fail_index: u32) -> majit_ir::DescrRef {
        Self::try_find_descr(token, trace_id, fail_index).unwrap_or_else(|| {
            panic!(
                "find_descr: (trace_id={}, fail_index={}) not found in \
                 root loop or any bridge — RPython uses exact faildescr \
                 object identity, so this lookup must succeed",
                trace_id, fail_index
            )
        })
    }

    fn try_find_descr(
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<majit_ir::DescrRef> {
        // Query-style miss tolerance: when `token.compiled` is absent
        // (e.g. a freshly issued JitCellToken whose backend code has
        // not been attached, or one rotated into `previous_tokens`),
        // probing callers like `compiled_bridge_descr_arc` /
        // `compiled_bridge_fail_descr_layouts` must observe `None`
        // instead of a panic. Cranelift's counterpart at
        // `compiler.rs:13955-13958` (`token.compiled.as_ref().and_then
        // (|c| c.downcast_ref::<CompiledLoop>())?`) follows the same
        // contract; pyre's `bridge_source_descr`
        // (`pyjitpl/mod.rs:7743-7757`) walks `compiled.token` then
        // each `previous_tokens[*]`, depending on `None` to advance.
        let compiled = token
            .compiled
            .as_ref()?
            .downcast_ref::<CompiledCode>()
            .expect("compiled data is not CompiledCode");
        // RPython looks up the faildescr by object identity (the resume
        // descr stored on the guard op IS what `cpu.get_latest_descr()`
        // returns).  Pyre's lookup is `(trace_id, fail_index)`-keyed; the
        // `trace_id` is always a real allocated id (alloc_trace_id starts
        // at 1).  No `0 → root_trace_id` sentinel — that path was a
        // pyre-only deviation removed alongside `normalize_trace_id`.
        //
        // `fail_descrs[i].fail_index_per_trace() == i` is enforced at
        // codegen end (`x86/assembler.rs:1741` / `:1863`,
        // `aarch64/assembler.rs:1403` / `:1532`); the per-Compiled trace
        // identity is the `CompiledCode::trace_id` field rather than a
        // per-descr `trace_id()` read.  Look up by position so the
        // descr's internal per-emission state is not required — Session 7
        // singleton FINISH descrs share an Arc across emissions and
        // would otherwise answer `fail_index_per_trace()` / `trace_id()`
        // with the trait default `0`.
        if compiled.trace_id == trace_id {
            if let Some(found) = compiled.fail_descrs.get(fail_index as usize) {
                return Some(found.clone());
            }
        }
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if bridge.trace_id == trace_id {
                    if let Some(found) = bridge.fail_descrs.get(fail_index as usize) {
                        return Some(found.clone());
                    }
                }
            }
        }
        None
    }

    fn call_assembler_targets_snapshot() -> HashMap<u64, usize> {
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .iter()
            .filter_map(|(&k, t)| {
                if t.code_addr != 0 {
                    Some((k, t.code_addr))
                } else {
                    None
                }
            })
            .collect()
    }

    /// `rpython/jit/backend/x86/assembler.py:599` parity: store
    /// `_ll_function_addr` after the loop is fully assembled.  Adopts the
    /// pending CLT Arc onto `token.compiled_loop_token` so the
    /// `frame_info` address baked into already-rewritten caller traces
    /// stays valid (mirrors cranelift `compiler.rs:2310-2326`).  Falls
    /// back to the token's existing eager Arc when no pending entry was
    /// registered (e.g. tokens compiled directly without going through
    /// `compile_tmp_callback`).
    fn register_call_assembler_target(token: &mut JitCellToken, code_addr: usize) {
        let token_number = token.number;
        let index_of_virtualizable = token.virtualizable_arg_index.map_or(-1_i32, |i| i as i32);
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm][ca-target] register token={} addr=0x{:x}",
                token_number, code_addr
            );
        }
        let mut guard = CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned");
        match guard.get_mut(&token_number) {
            Some(existing) => {
                existing.code_addr = code_addr;
                // Adopt the pending CLT Arc onto the JitCellToken so the
                // pending-window allocation remains the live one.
                token.compiled_loop_token = Some(Arc::clone(&existing.compiled_loop_token));
            }
            None => {
                let clt = token
                    .compiled_loop_token
                    .as_ref()
                    .expect("JitCellToken missing compiled_loop_token")
                    .clone();
                guard.insert(
                    token_number,
                    DynasmCaTarget {
                        code_addr,
                        compiled_loop_token: clt,
                        index_of_virtualizable,
                    },
                );
            }
        }
    }

    fn redirect_call_assembler_target(old_number: u64, new_addr: usize) {
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm][ca-target] redirect token={} addr=0x{:x}",
                old_number, new_addr
            );
        }
        if let Some(existing) = CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .get_mut(&old_number)
        {
            existing.code_addr = new_addr;
        }
    }

    /// Static entry point for `lib.rs register_pending_call_assembler_target`.
    /// Creates a fresh `Arc<CompiledLoopToken>` (mirrors cranelift
    /// `compiler.rs:2427`) and inserts `code_addr = 0`; snapshot helpers
    /// filter the pending entry out so generated CALL_ASSEMBLER code
    /// falls through to the force-fn trampoline until `compile_loop`
    /// overwrites the address.
    pub fn register_pending_call_assembler_target_static(
        token_number: u64,
        num_inputs: usize,
        index_of_virtualizable: i32,
    ) {
        let pending_clt = Arc::new(majit_backend::CompiledLoopToken::new(token_number));
        // `regalloc.py:861-871` `_set_initial_bindings`: each input lands at
        // `loc.value - base_ofs = (JITFRAME_FIXED_SIZE + i) * SIZEOFSIGNED` —
        // the same formula `Self::input_initial_loc` uses when finalising
        // a real compile (see `compile_loop` ll_initial_locs assignment).
        // Self-recursive CALL_ASSEMBLER registers this stub before its own
        // trace finalises, so the rewriter's `handle_call_assembler` reads
        // these pending offsets to emit the callee inputarg `GcStore`s; if
        // they omit the JITFRAME_FIXED_SIZE shift the stores land in the
        // managed-register save area and the callee enters with NULL inputs.
        *pending_clt._ll_initial_locs.lock() =
            (0..num_inputs).map(Self::input_initial_loc).collect();
        CALL_ASSEMBLER_TARGETS
            .lock()
            .expect("CALL_ASSEMBLER_TARGETS poisoned")
            .entry(token_number)
            .or_insert_with(|| DynasmCaTarget {
                code_addr: 0,
                compiled_loop_token: pending_clt,
                index_of_virtualizable,
            });
    }

    /// `rpython/jit/backend/llsupport/llmodel.py:534-537`
    /// `get_baseofs_of_frame_field(self)` — offset from a `JITFRAME` base
    /// to the first frame-array item. Used by `_set_initial_bindings`
    /// (regalloc.py:865) and `update_frame_info` (model.py:316) for
    /// `jfi_frame_size` accounting (jitframe.py:19-22).
    fn get_baseofs_of_frame_field() -> i64 {
        crate::jitframe::FIRST_ITEM_OFFSET as i64
    }
}

impl Backend for DynasmBackend {
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError> {
        token.inputarg_types = inputargs.iter().map(|ia| ia.tp).collect();
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        let header_pc = self.next_header_pc;
        // gc.py:109 rewrite_assembler parity: run GC rewriter before regalloc.
        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops);
        let constants = std::mem::take(&mut self.constants);
        let constant_types = std::mem::take(&mut self.constant_types);
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &constants);
        let guard_gc_type_info = self.collect_guard_gc_type_info();
        let subclass_range_table =
            self.collect_classptr_subclass_range_table(&prepared_ops, &constants);
        let attached_descrs = self.attached_descr_ptrs();
        let cpu_handle = self.cpu_handle();
        // PyPy's `setup_once` (`llsupport/assembler.py:97`) is what
        // builds the per-CPU malloc / propagate trampolines, but the
        // pyre `Backend::setup_once` hook isn't yet wired into every
        // tracing entry (`force_start_tracing` builds the trace ctx
        // inline rather than going through `setup_tracing`).  Ensure
        // the trampoline is materialised lazily on the first
        // `compile_loop`/`compile_bridge` instead — idempotent and
        // cheap after the cache hit, matching PyPy's "build once per
        // CPU" semantics without requiring every trace-start path to
        // remember to call `_setup_once`.
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_fixed = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_fixed(&self.descr_attachments);
        let mut asm = Asm::new(
            trace_id,
            header_pc,
            constants,
            self.vtable_offset,
            typeid_table,
            guard_gc_type_info,
            subclass_range_table,
            attached_descrs,
            cpu_handle,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_fixed,
            inputargs,
            &prepared_ops,
        );
        asm.set_constant_types(constant_types);
        asm.set_call_assembler_targets(Self::call_assembler_targets_snapshot());
        let compiled = asm.assemble_loop()?;

        let code_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();
        let frame_depth = compiled.frame_depth.load(Ordering::Acquire) as i64;
        Self::register_call_assembler_target(token, code_addr);
        self.register_fail_descrs(&compiled.fail_descrs);

        // `compile.py:183-186 record_loop_or_bridge`: for each ResumeDescr
        // in the newly-compiled trace, stamp the owning CompiledLoopToken.
        // RPython predicates the stamp on `isinstance(descr, ResumeDescr)`
        // (`compile.py:185`); pyre uses the `is_resume_guard()` trait
        // method (descr.rs:779), implemented true on the
        // `ResumeGuardDescr` family (compile.rs:3117/3296/3434/3558/4125).
        //
        // `compile.py:183-186` walks `loop.operations` and writes
        // `op.descr.rd_loop_token` directly.  Pyre's `compiled.fail_descrs`
        // hold the same `DescrRef`s the metainterp stamped onto each
        // guard op (Session 6 unification), so the write lands on the
        // same `ResumeGuardDescr` Arc upstream targets.
        if let Some(clt) = token.compiled_loop_token.as_ref() {
            for descr in &compiled.fail_descrs {
                if !descr.is_resume_guard() {
                    continue;
                }
                if let Some(fd) = descr.as_fail_descr() {
                    fd.set_rd_loop_token_clt(std::sync::Arc::clone(clt)
                        as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                }
            }
        }

        // `rpython/jit/backend/x86/assembler.py:513-526` initializes the
        // per-loop `CompiledLoopToken` fields at assemble_loop entry:
        //   * frame_info is allocated and assigned (line 526-530)
        //   * looptoken.compiled_loop_token = clt (line 514)
        // pyre eagerly creates the CLT in `JitCellToken::new`, so the
        // equivalent here is populating its fields with the real values
        // computed during assembly.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let Some(clt) = token.compiled_loop_token.as_ref() {
            // `x86/assembler.py:526-530` frame_info = malloc_aligned + set
            // jfi_frame_depth/jfi_frame_size. pyre's frame_info lives on
            // the CLT already; just populate via update_frame_depth.
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, frame_depth);
            // `llsupport/regalloc.py:861-871` `_set_initial_bindings` —
            // each input lands at `loc.value - base_ofs =
            // (JITFRAME_FIXED_SIZE + i) * SIZEOFSIGNED` so the GcStores
            // synthesized by `handle_call_assembler` (rewrite.py:673)
            // hit the actual input slots, not the managed-register save
            // area at the head of `jf_frame`. The list length must match
            // `inputargs.len()` so `handle_call_assembler` can index it.
            let locs: Vec<i32> = (0..inputargs.len()).map(Self::input_initial_loc).collect();
            *clt._ll_initial_locs.lock() = locs;
        }
        // `x86/assembler.py:599` `looptoken._ll_function_addr =
        // rawstart + functionpos`. pyre stores the single entry point
        // so `_ll_function_addr` = compiled-code base.
        token._ll_function_addr = code_addr;
        token.compiled = Some(Box::new(compiled));

        Ok(AsmInfo {
            code_addr,
            code_size,
        })
    }

    fn set_constants(&mut self, constants: std::collections::HashMap<u32, i64>) {
        self.constants = constants;
    }

    fn set_constant_types(
        &mut self,
        constant_types: std::collections::HashMap<u32, majit_ir::Type>,
    ) {
        self.constant_types = constant_types;
    }

    fn set_next_trace_id(&mut self, trace_id: u64) {
        self.next_trace_id = trace_id;
    }

    fn set_next_header_pc(&mut self, header_pc: u64) {
        self.next_header_pc = header_pc;
    }

    fn set_done_with_this_frame_descr_void(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_void = Some(descr);
    }
    fn set_done_with_this_frame_descr_int(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_int = Some(descr);
    }
    fn set_done_with_this_frame_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_ref = Some(descr);
    }
    fn set_done_with_this_frame_descr_float(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .done_with_this_frame_descr_float = Some(descr);
    }
    fn set_exit_frame_with_exception_descr_ref(&mut self, descr: majit_ir::DescrRef) {
        self.descr_attachments
            .write()
            .unwrap()
            .exit_frame_with_exception_descr_ref = Some(descr);
    }
    fn set_propagate_exception_descr(&mut self, descr: majit_ir::DescrRef) {
        // x86/assembler.py:328 `_build_propagate_exception_path` parity:
        // PyPy bakes `propagate_exception_descr` into the per-CPU
        // propagate trampoline at setup time.  Pyre defers the bake to
        // `X86CpuExt::ensure_propagate_exception_path` (x86/cpu_ext.rs),
        // which reads the descr pointer directly from
        // `descr_attachments` and embeds it in the helper.
        //
        // The metainterp wiring re-installs the descr several times
        // during init (`attach_descrs_to_cpu`, `register_jitdriver_sd`,
        // and `attach_default_test_descrs` for backend-only tests).
        // The common case is the *same* `Arc<Descr>` arriving twice —
        // idempotent, no observable effect.  Tests that mint a fresh
        // `PropagateExceptionDescr` on every call also reach this
        // setter; allow the overwrite so the second install does not
        // panic.  The remaining risk — a *different* descr being
        // installed *after* the trampoline has baked the previous
        // pointer as an immediate — is a strictly post-`compile_loop`
        // hazard; in production the descr is set exactly once before
        // any compile fires, so this path never executes there.
        let mut attachments = self.descr_attachments.write().unwrap();
        if attachments
            .propagate_exception_descr
            .as_ref()
            .is_some_and(|existing| std::sync::Arc::ptr_eq(existing, &descr))
        {
            return;
        }
        attachments.propagate_exception_descr = Some(descr);
    }

    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
        _previous_tokens: &[std::sync::Arc<JitCellToken>],
        _caller_recovery_layout: Option<&majit_backend::ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError> {
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;

        let prepared_ops = self.prepare_ops_for_compile(inputargs, ops);
        let constants = std::mem::take(&mut self.constants);
        let constant_types = std::mem::take(&mut self.constant_types);
        if crate::majit_log_enabled() && trace_id == 2 {
            eprintln!(
                "--- dynasm bridge prepared ops (trace_id={}, fail_index={}) ---\n{}",
                trace_id,
                fail_descr.fail_index_per_trace(),
                majit_ir::format_trace(&prepared_ops, &constants)
            );
        }
        let typeid_table = self.collect_classptr_typeid_table(&prepared_ops, &constants);
        let guard_gc_type_info = self.collect_guard_gc_type_info();
        let subclass_range_table =
            self.collect_classptr_subclass_range_table(&prepared_ops, &constants);
        let attached_descrs = self.attached_descr_ptrs();
        let cpu_handle = self.cpu_handle();
        // PyPy's `setup_once` (`llsupport/assembler.py:97`) is what
        // builds the per-CPU malloc / propagate trampolines, but the
        // pyre `Backend::setup_once` hook isn't yet wired into every
        // tracing entry (`force_start_tracing` builds the trace ctx
        // inline rather than going through `setup_tracing`).  Ensure
        // the trampoline is materialised lazily on the first
        // `compile_loop`/`compile_bridge` instead — idempotent and
        // cheap after the cache hit, matching PyPy's "build once per
        // CPU" semantics without requiring every trace-start path to
        // remember to call `_setup_once`.
        #[cfg(target_arch = "x86_64")]
        let malloc_slowpath_fixed = self
            .arch_cpu_ext
            .ensure_malloc_slowpath_fixed(&self.descr_attachments);
        let mut asm = Asm::new(
            trace_id,
            0,
            constants,
            self.vtable_offset,
            typeid_table,
            guard_gc_type_info,
            subclass_range_table,
            attached_descrs,
            cpu_handle,
            #[cfg(target_arch = "x86_64")]
            malloc_slowpath_fixed,
            inputargs,
            &prepared_ops,
        );
        asm.set_constant_types(constant_types);
        asm.set_call_assembler_targets(Self::call_assembler_targets_snapshot());

        let _orig_compiled = Self::get_compiled(original_token);

        let guard_descr = Self::find_descr(
            original_token,
            fail_descr.trace_id(),
            fail_descr.fail_index_per_trace(),
        );
        let arglocs = Asm::rebuild_faillocs_from_descr(
            guard_descr
                .as_fail_descr()
                .expect("guard_descr is FailDescr"),
            inputargs,
        );
        let compiled = asm.assemble_bridge(fail_descr, &arglocs)?;

        let bridge_addr = codebuf::buffer_ptr(&compiled.buffer) as usize;
        let code_size = compiled.buffer.len();
        // `rpython/jit/backend/x86/assembler.py:691-693` `assemble_bridge`:
        // `frame_depth = max(current_clt.frame_info.jfi_frame_depth,
        //                    frame_depth_no_fixed_size + JITFRAME_FIXED_SIZE)`
        // → `self.update_frame_depth(frame_depth)` which calls
        // `self.current_clt.frame_info.update_frame_depth(baseofs, frame_depth)`.
        //
        // Without this, self-recursive CALL_ASSEMBLER allocates callee
        // jitframes with the loop's original (smaller) jfi_frame_depth,
        // so regalloc spill slots past that depth overrun into the next
        // nursery allocation (observed: jf_L[272] aliases the next
        // callee's jf_frame_info, so llfi ends up at input0).
        let bridge_frame_depth = compiled.frame_depth.load(Ordering::Acquire) as i64;
        let baseofs = Self::get_baseofs_of_frame_field();
        if let Some(clt) = original_token.compiled_loop_token.as_ref() {
            clt.frame_info
                .lock()
                .update_frame_depth(baseofs, bridge_frame_depth);
        }
        // Keep the existing loop CompiledCode.frame_depth in lockstep
        // (same rationale as `redirect_call_assembler` — dynasm codegen
        // reads `CompiledCode.frame_depth` in addition to
        // `frame_info.jfi_frame_depth`).
        Self::get_compiled(original_token)
            .frame_depth
            .fetch_max(bridge_frame_depth as usize, Ordering::Release);

        // assembler.py:987 patch_jump_for_descr — redirect guard to bridge.
        // Use the exact guard descr found above, not a fail_index search.
        let guard_fd = guard_descr
            .as_fail_descr()
            .expect("guard_descr is FailDescr");
        let ajo = guard_fd.adr_jump_offset();
        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm-bridge] patch: trace_id={} fail_index={} adr_jump_offset=0x{:x} bridge_addr=0x{:x}",
                guard_fd.trace_id(),
                guard_fd.fail_index_per_trace(),
                ajo,
                bridge_addr
            );
        }
        if ajo != 0 {
            Asm::patch_jump_for_descr(guard_fd, bridge_addr);
            self.register_bridge_addr(Arc::as_ptr(&guard_descr) as *const () as usize, bridge_addr);
        } else if crate::majit_log_enabled() {
            eprintln!("[dynasm-bridge] WARNING: adr_jump_offset=0, bridge NOT patched!");
        }

        // llmodel.py:252 asmmemmgr_blocks parity: store the entire
        // bridge CompiledCode on the owning loop token. This keeps
        // both the ExecutableBuffer (mapped code) AND the fail_descrs
        // (DescrRef) alive. Recovery stubs embed raw pointers to
        // these Arcs — dropping them would create dangling pointers
        // when a bridge-internal guard fires.  RPython's asmmemmgr
        // ties code blocks and their resume descriptors to the same
        // compiled_loop_token lifetime.
        self.register_fail_descrs(&compiled.fail_descrs);

        // `compile.py:183-186 record_loop_or_bridge`: a bridge's ResumeDescrs
        // inherit the original loop's CompiledLoopToken.  See the
        // sibling `compile_loop` site for the parity rationale on the
        // `is_resume_guard()` predicate (`compile.py:185`).
        if let Some(clt) = original_token.compiled_loop_token.as_ref() {
            for descr in &compiled.fail_descrs {
                if !descr.is_resume_guard() {
                    continue;
                }
                if let Some(fd) = descr.as_fail_descr() {
                    fd.set_rd_loop_token_clt(std::sync::Arc::clone(clt)
                        as std::sync::Arc<dyn std::any::Any + Send + Sync>);
                }
            }
        }

        original_token.asmmemmgr_blocks().push(Box::new(compiled));

        Ok(AsmInfo {
            code_addr: bridge_addr,
            code_size,
        })
    }

    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame {
        // assembler.py:1080 `_call_header_with_stack_check` emits the
        // inline probe at the top of every compiled loop, matching
        // cranelift's `jit_prologue_stack_check_shim` call in
        // `compiler.rs:5868`. The prior runner-level `jit_prologue_stack_check`
        // call only guarded top-level entry and missed compiled-to-
        // compiled CALL_ASSEMBLER recursion.
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_ptr = unsafe { libc::calloc(1, JitFrame::alloc_size(num_slots)) as *mut JitFrame };
        assert!(!jf_ptr.is_null(), "execute_token: calloc failed");
        unsafe { JitFrame::init(jf_ptr, std::ptr::null(), num_slots) };
        // Register this libc-allocated jitframe with the GC so its
        // interior Ref slots (pinned by gcmap bits) remain visible to
        // the collector during CallMallocNursery slow-path collections.
        majit_gc::shadow_stack::register_libc_jitframe(jf_ptr as usize);

        for (i, arg) in args.iter().enumerate() {
            let raw = match arg {
                Value::Int(v) => *v,
                Value::Ref(r) => r.0 as i64,
                Value::Float(f) => f.to_bits() as i64,
                Value::Void => 0,
            };
            unsafe { crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), raw as isize) };
        }

        if crate::majit_log_enabled() {
            for (i, arg) in args.iter().enumerate() {
                let raw = unsafe {
                    crate::llmodel::get_int_value_direct(jf_ptr, Self::input_slot(i)) as i64
                };
                eprintln!("[dynasm]   arg[{}] = {:#018x} ({:?})", i, raw as u64, arg);
            }
            eprintln!(
                "[dynasm] execute_token: entry={:?} jf_ptr={:?} num_args={} num_slots={} code_len={}",
                entry,
                jf_ptr,
                args.len(),
                num_slots,
                compiled.buffer.len()
            );
        }

        if crate::majit_dump_enabled() {
            let code = unsafe { std::slice::from_raw_parts(entry, compiled.buffer.len()) };
            eprintln!("[dynasm] CODE DUMP ({} bytes at {:?}):", code.len(), entry);
            for (i, chunk) in code.chunks(4).enumerate() {
                let word = u32::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ]);
                eprint!("{:08x} ", word);
                if (i + 1) % 8 == 0 {
                    eprintln!();
                }
            }
            eprintln!();
        }

        // Debug: verify bridge patches are visible
        if crate::majit_log_enabled() {
            for descr in &compiled.fail_descrs {
                let bridge_addr = self.lookup_bridge_addr(Arc::as_ptr(descr) as *const () as usize);
                if let Some(fd) = descr.as_fail_descr() {
                    if bridge_addr != 0 && fd.adr_jump_offset() == 0 {
                        eprintln!(
                            "[dynasm] bridge-patched guard fi={} bridge_addr={:#x} ajo=0 (patched)",
                            fd.fail_index_per_trace(),
                            bridge_addr
                        );
                    }
                }
            }
        }

        // llmodel.py:323: ll_frame = func(ll_frame). The compiled
        // prologue (gen_shadowstack_header) / epilogue
        // (gen_footer_shadowstack) push/pop the jf_ptr onto the shadow
        // stack inline, matching aarch64/assembler.py:1422/1438 — no
        // manual push_jf/pop_jf_to around the call.
        let func: unsafe extern "C" fn(*mut JitFrame) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr) };

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] execute_token returned: result_jf={:?} (expected={:?}) same={}",
                result_jf,
                jf_ptr,
                result_jf == jf_ptr
            );
        }

        // llmodel.py:412-420 get_latest_descr: read jf_descr from frame.
        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(result_jf) as i64 };
        let descr = self.find_descr_by_ptr(token, jf_descr_raw as usize, result_jf);
        let descr_fd = descr
            .as_fail_descr()
            .expect("find_descr_by_ptr result must implement FailDescr");

        if crate::majit_log_enabled() {
            eprintln!(
                "[dynasm] descr: fi={} finish={} types={} rd_locs={:?}",
                descr_fd.fail_index_per_trace(),
                descr_fd.is_finish(),
                descr_fd.fail_arg_types().len(),
                descr_fd.rd_locs()
            );
        }

        // PyPy `llmodel.py:422-424 _decode_pos` parity: remap jitframe
        // values via `descr.rd_locs[i]`.  Synthetic / out-of-range
        // descrs fall back to identity slot indexing.
        let rd_locs_len = descr_fd.rd_locs().len();
        let mut raw_values: Vec<i64> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            if i < rd_locs_len {
                match crate::guard::decode_rd_loc_slot(descr_fd, i) {
                    Some(slot) => {
                        let val =
                            unsafe { crate::llmodel::get_int_value_direct(result_jf, slot) as i64 };
                        if crate::majit_log_enabled() && i < 10 {
                            eprintln!(
                                "[dynasm] fail_arg[{}]: slot={} val={:#018x}",
                                i, slot, val as u64
                            );
                        }
                        raw_values.push(val);
                    }
                    None => raw_values.push(0),
                }
            } else {
                raw_values
                    .push(unsafe { crate::llmodel::get_int_value_direct(result_jf, i) as i64 });
            }
        }

        majit_gc::shadow_stack::unregister_libc_jitframe(jf_ptr as usize);
        unsafe { libc::free(jf_ptr as *mut std::ffi::c_void) };

        DeadFrame {
            data: Box::new(FrameData::new(raw_values, descr, None)),
        }
    }

    /// Override execute_token_ints_raw to return the FULL jitframe
    /// content (all slots), matching Cranelift's behavior.
    /// RPython: the deadframe IS the jitframe — all slots are accessible.
    fn execute_token_ints_raw(
        &self,
        token: &JitCellToken,
        args: &[i64],
    ) -> majit_backend::RawExecResult {
        // Same rationale as `execute_token`: the inline probe emitted by
        // `_call_header` (x86/aarch64 assembler.rs) is now the sole
        // stack-overflow detection site, so no runner-level probe is
        // needed here.
        let compiled = Self::get_compiled(token);
        let entry = codebuf::buffer_ptr(&compiled.buffer);

        let num_slots = args
            .len()
            .max(compiled.fail_descrs.len() * 4)
            .max(compiled.frame_depth.load(Ordering::Acquire))
            .max(64);
        let jf_ptr = unsafe { libc::calloc(1, JitFrame::alloc_size(num_slots)) as *mut JitFrame };
        assert!(!jf_ptr.is_null(), "execute_token_ints_raw: calloc failed");
        unsafe { JitFrame::init(jf_ptr, std::ptr::null(), num_slots) };
        // Same registration as `execute_token` above: the libc-allocated
        // jitframe must be visible to the minor-collection walker so its
        // `jf_gcmap`-marked Ref slots get traced during
        // CallMallocNursery-triggered collections. Without this the
        // inner-loop jitframe's live Refs go un-updated and later guard
        // deadframes read stale (freed-nursery) pointers. The matching
        // `unregister_libc_jitframe` below balances this registration.
        majit_gc::shadow_stack::register_libc_jitframe(jf_ptr as usize);

        for (i, &val) in args.iter().enumerate() {
            unsafe { crate::llmodel::set_int_value(jf_ptr, Self::input_slot(i), val as isize) };
        }

        let func: unsafe extern "C" fn(*mut JitFrame) -> *mut JitFrame =
            unsafe { std::mem::transmute(entry) };
        let result_jf = unsafe { func(jf_ptr) };

        let jf_descr_raw = unsafe { crate::llmodel::get_latest_descr(result_jf) as i64 };
        let descr = self.find_descr_by_ptr(token, jf_descr_raw as usize, result_jf);
        let descr_addr = Arc::as_ptr(&descr) as *const () as usize;
        let descr_fd = descr
            .as_fail_descr()
            .expect("find_descr_by_ptr result must implement FailDescr");

        let fail_arg_types = descr_fd.fail_arg_types();
        let num_fail_args = fail_arg_types.len();
        let mut outputs: Vec<i64> = Vec::with_capacity(num_slots);
        for i in 0..num_slots {
            outputs.push(unsafe { crate::llmodel::get_int_value_direct(result_jf, i) as i64 });
        }
        // PyPy `llmodel.py:422-424 _decode_pos` parity: read each
        // fail-arg slot from `descr.rd_locs[i]`.  Out-of-range index
        // (synthetic descrs without rd_locs) falls back to identity.
        let rd_locs_len = descr_fd.rd_locs().len();
        let mut typed_outputs = Vec::with_capacity(num_fail_args);
        for i in 0..num_fail_args {
            let raw = if i < rd_locs_len {
                match crate::guard::decode_rd_loc_slot(descr_fd, i) {
                    Some(slot) => outputs.get(slot).copied().unwrap_or(0),
                    None => 0,
                }
            } else {
                outputs.get(i).copied().unwrap_or(0)
            };
            typed_outputs.push(match fail_arg_types[i] {
                Type::Ref => Value::Ref(GcRef(raw as usize)),
                Type::Float => Value::Float(f64::from_bits(raw as u64)),
                // `Type::Void` is the resume.py:411-417 hole sentinel —
                // the slot's value is reconstructed from the resume
                // snapshot (TAGCONST/TAGVIRTUAL), not from the deadframe.
                // Surfacing `Value::Void` keeps `gc_ref_slots` and
                // downstream type-tag dispatchers from misclassifying
                // it as a live `Ref` and leaking a NULL `GcRef`.
                Type::Void => Value::Void,
                Type::Int => Value::Int(raw),
            });
        }
        let exit_layout = Some(crate::guard::layout_for_fail_descr(
            descr_fd,
            descr_addr,
            descr_fd.fail_index_per_trace(),
            descr_fd.trace_id(),
        ));

        majit_gc::shadow_stack::unregister_libc_jitframe(jf_ptr as usize);
        unsafe { libc::free(jf_ptr as *mut std::ffi::c_void) };

        let descr_arc: majit_ir::DescrRef = descr.clone();
        majit_backend::RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            force_token_slots: Vec::new(),
            savedata: None,
            exception_value: GcRef::NULL,
            fail_index: descr_fd.fail_index_per_trace(),
            trace_id: descr_fd.trace_id(),
            is_finish: descr_fd.is_finish(),
            is_exit_frame_with_exception: descr_fd.is_exit_frame_with_exception(),
            status: descr_fd.get_status(),
            descr_addr,
            descr_arc,
        }
    }

    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr {
        let data = frame.data.downcast_ref::<FrameData>().unwrap();
        data.fail_descr
            .as_fail_descr()
            .expect("FrameData::fail_descr must implement FailDescr")
    }

    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn majit_ir::Descr> {
        // `history.py:125` `cpu.get_latest_descr(deadframe)` returns the
        // metainterp `AbstractFailDescr` object op.descr stamped.  After
        // the FrameData::fail_descr type cascade to `DescrRef`, `fail_descr`
        // *is* the metainterp Arc (production codegen + synthetic exits
        // both stamp it via DescrRef directly), so we just clone the
        // shared identity.
        let data = frame.data.downcast_ref::<FrameData>().unwrap();
        Arc::clone(&data.fail_descr)
    }

    /// `fail_descr_registry` is the `addr → DescrRef` table populated by
    /// `register_fail_descrs` at compile time, keeping every emitted
    /// descr alive for the lifetime of its owning compiled code
    /// (mirroring RPython's `cpu` retaining descr objects).  The cloned
    /// `DescrRef` lets the CA bridge entry route descr identity into
    /// `_trace_and_compile_from_bridge` (compile.py:704-709) without
    /// going through the `(trace_id, fail_index)` surrogate key.
    fn free_loop(&mut self, token: &JitCellToken) {
        // memmgr.py:9 `MemoryManager.alive_loops` parity: when a JCT is
        // evicted, RPython's GC reclaims the token's compiled code and
        // every dependent FailDescr naturally because nothing keeps a
        // strong ref past `alive_loops`.  Pyre's `fail_descr_registry`
        // holds strong Arcs to every emitted descr for the C-ABI
        // recovery path (`fail_descr_arc_from_addr` parity with
        // `cpu.get_latest_descr`), so we must explicitly release the
        // entries belonging to this token here — otherwise the descrs
        // (and their `rd_loop_token_clt` chain) outlive their owning
        // loop and pyre keeps memory PyPy would have freed.
        //
        // Sweep entries whose owning-JCT upgrade points back at this
        // token. Ownerless entries (`descr_owning_jct == None`) are
        // PRESERVED: in PyPy these correspond to FINISH descrs
        // (`_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescr`,
        // `compile.py:185` skipped via `isinstance(descr, ResumeDescr)`)
        // which are module-level singletons not subject to per-loop GC.
        // Treating `None` as "delete" would over-shoot PyPy's lifetime
        // contract.
        let mut removed_bridge_addr_keys = Vec::new();
        let mut registry = self
            .fail_descr_registry
            .lock()
            .expect("fail_descr_registry mutex poisoned");
        registry.retain(|_, descr| {
            let dyn_descr = match descr.as_fail_descr() {
                Some(fd) => fd,
                None => return true,
            };
            match majit_backend::descr_owning_jct(dyn_descr) {
                Some(owner) if owner.number == token.number => {
                    removed_bridge_addr_keys.push(Arc::as_ptr(descr) as *const () as usize);
                    false
                }
                Some(_) => true,
                None => true,
            }
        });
        drop(registry);
        if !removed_bridge_addr_keys.is_empty() {
            let mut bridge_addrs = self
                .bridge_addr_by_descr
                .lock()
                .expect("bridge_addr_by_descr mutex poisoned");
            for key in removed_bridge_addr_keys {
                bridge_addrs.remove(&key);
            }
        }
    }

    fn fail_descr_arc_from_addr(&self, descr_addr: usize) -> majit_ir::DescrRef {
        // warmspot.py:1021 cpu.get_latest_descr(deadframe) parity: the
        // dynasm registry holds a strong DescrRef for every emitted
        // descr through its full lifetime, so a `descr_addr` arriving
        // from the C-ABI guard-fail path is always present in the
        // table.  A miss is an invariant violation, not a recoverable
        // runtime mode.
        let registry = self
            .fail_descr_registry
            .lock()
            .expect("fail_descr_registry mutex poisoned");
        let descr = registry.get(&descr_addr).cloned().unwrap_or_else(|| {
            panic!(
                "fail_descr_arc_from_addr: descr_addr {descr_addr:#x} not in \
                 fail_descr_registry — every emitted fail descr must be \
                 registered before its address reaches the C-ABI guard-fail boundary"
            )
        });
        descr as majit_ir::DescrRef
    }

    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_int(index)
    }

    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64 {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_float(index)
    }

    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> GcRef {
        frame
            .data
            .downcast_ref::<FrameData>()
            .unwrap()
            .get_ref(index)
    }

    fn invalidate_loop(&self, token: &JitCellToken) {
        token.invalidated.store(true, Ordering::Release);
    }

    // assembler.py:1138 redirect_call_assembler
    fn redirect_call_assembler(
        &self,
        old: &JitCellToken,
        new: &JitCellToken,
    ) -> Result<(), BackendError> {
        let old_compiled = Self::get_compiled(old);
        let new_compiled = Self::get_compiled(new);
        // x86/assembler.py:1146-1151 update_frame_info parity: propagate
        // new loop's frame depth onto the old token and every token in
        // its existing redirect chain, using the `baseofs` obtained from
        // `cpu.get_baseofs_of_frame_field()` so `jfi_frame_size` follows
        // jitframe.py:19-22 `base_ofs + new_depth * SIZEOFSIGNED`.
        let baseofs = Self::get_baseofs_of_frame_field();
        if let (Some(new_clt), Some(old_clt)) = (
            new.compiled_loop_token.as_ref(),
            old.compiled_loop_token.as_ref(),
        ) {
            // Seed new's CompiledLoopToken.frame_info.jfi_frame_depth
            // from the backend-specific compiled code depth so
            // update_frame_info has a non-zero value to propagate.
            let new_depth = new_compiled.frame_depth.load(Ordering::Acquire);
            new_clt
                .frame_info
                .lock()
                .update_frame_depth(baseofs, new_depth as i64);
            // model.py:316-329 update_frame_info — pass old CLT with a
            // weak ref for the "append self to chain" step (line 328
            // `new_loop_tokens.append(weakref.ref(oldlooptoken))`).
            let old_weak = Arc::downgrade(old_clt);
            new_clt.update_frame_info(old_clt, old_weak, baseofs);
            // Keep the backend-specific frame_depth in lockstep so bridge
            // codegen's existing readers (CompiledCode.frame_depth) also
            // see the propagated value. PRE-EXISTING-ADAPTATION: RPython
            // reads the depth back from `compiled_loop_token.frame_info`;
            // dynasm's codegen reads `CompiledCode.frame_depth`. Writing
            // both keeps the orthodox field authoritative while the
            // reader migration lands.
            old_compiled
                .frame_depth
                .fetch_max(new_depth, Ordering::Release);
        }
        let old_addr = codebuf::buffer_ptr(&old_compiled.buffer);
        let new_addr = codebuf::buffer_ptr(&new_compiled.buffer);
        Asm::redirect_call_assembler(old_addr, new_addr);
        Self::redirect_call_assembler_target(old.number, new_addr as usize);
        Ok(())
    }

    // No migrate_bridges — we patch in place.

    fn store_guard_hashes(&self, token: &JitCellToken, hashes: &[u64]) {
        let compiled = Self::get_compiled(token);
        for (i, &hash) in hashes.iter().enumerate() {
            if let Some(descr) = compiled.fail_descrs.get(i) {
                // `compile.py:826-829` `store_hash` only fires for non-final
                // `AbstractResumeGuardDescr` whose status is still 0 (no
                // counter yet stamped).  Route the predicate through the
                // FailDescr trait so the metainterp class hierarchy
                // (`final_descr=True` on Done*/Exit*/Propagate) answers.
                if let Some(fd) = descr.as_fail_descr() {
                    if !fd.is_finish() && fd.get_status() == 0 {
                        fd.store_hash(hash);
                    }
                }
            }
        }
    }

    fn get_guard_status(
        &self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> (u64, usize) {
        let descr = Self::find_descr(token, trace_id, fail_index);
        let fd = descr.as_fail_descr().expect("descr is FailDescr");
        (fd.get_status(), Arc::as_ptr(&descr) as *const () as usize)
    }

    fn store_bridge_guard_hashes(
        &self,
        token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
        hashes: &[u64],
    ) {
        let source_descr = Self::find_descr(token, source_trace_id, source_fail_index);
        let bridge_addr = self.lookup_bridge_addr(Arc::as_ptr(&source_descr) as *const () as usize);
        if bridge_addr == 0 {
            return;
        }
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                let addr = codebuf::buffer_ptr(&bridge.buffer) as usize;
                if addr == bridge_addr {
                    for (i, &hash) in hashes.iter().enumerate() {
                        if let Some(descr) = bridge.fail_descrs.get(i) {
                            if let Some(fd) = descr.as_fail_descr() {
                                if !fd.is_finish() && fd.get_status() == 0 {
                                    fd.store_hash(hash);
                                }
                            }
                        }
                    }
                    return;
                }
            }
        }
    }

    fn read_descr_status(&self, descr_addr: usize) -> u64 {
        // `compile.py:741 self.status` — route through the registry-backed
        // `Arc<dyn FailDescr>` so the dispatch is trait-based.  Once
        // `jf_descr` carries the meta Arc address (Unified-Descr identity
        // flip), only the registry value changes; this dispatch chain
        // stays the same.
        if descr_addr == 0 {
            return 0;
        }
        <Self as Backend>::fail_descr_arc_from_addr(self, descr_addr)
            .as_fail_descr()
            .map_or(0, |fd| fd.get_status())
    }

    fn start_compiling_descr(&self, descr_addr: usize) {
        // `compile.py:786-788 self.start_compiling()` — trait dispatch.
        if descr_addr == 0 {
            return;
        }
        if let Some(fd) =
            <Self as Backend>::fail_descr_arc_from_addr(self, descr_addr).as_fail_descr()
        {
            fd.start_compiling();
        }
    }

    fn done_compiling_descr(&self, descr_addr: usize) {
        // `compile.py:790-795 self.done_compiling()` — trait dispatch.
        if descr_addr == 0 {
            return;
        }
        if let Some(fd) =
            <Self as Backend>::fail_descr_arc_from_addr(self, descr_addr).as_fail_descr()
        {
            fd.done_compiling();
        }
    }

    fn bh_new(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe { libc::memset(ptr, 0, size) };
        }
        ptr as i64
    }

    fn bh_new_with_vtable(&self, sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let size = sizedescr.as_size();
        let vtable = sizedescr.get_vtable();
        let ptr = unsafe { libc::malloc(size) };
        if !ptr.is_null() {
            unsafe {
                libc::memset(ptr, 0, size);
                // llmodel.py:780-782: if self.vtable_offset is not None:
                //   self.write_int_at_mem(res, self.vtable_offset, WORD, sizedescr.get_vtable())
                if let Some(vt_off) = self.vtable_offset {
                    if vtable != 0 {
                        *((ptr as *mut u8).add(vt_off) as *mut usize) = vtable;
                    }
                }
            }
        }
        ptr as i64
    }

    /// llmodel.py:788-790 bh_new_array / bh_new_array_clear.
    fn bh_new_array(&self, length: i64, arraydescr: &majit_translate::jitcode::BhDescr) -> i64 {
        let length = usize::try_from(length).expect("bh_new_array length must be non-negative");
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let len_offset = arraydescr
            .array_len_offset()
            .expect("bh_new_array requires ArrayDescr.lendescr");
        // descr.py:340 `ArrayDescr.get_type_id(): assert self.tid` —
        // allocation requires a real GC type id; tid=0 means the descr
        // never went through `gc.py:548 set_type_id` and the GC tracer
        // would lack the per-item visit shape.
        // PRE-EXISTING-ADAPTATION: `BhDescr.get_type_id()` returns the
        // u64 `path_hash` cache key, but `dynasm_alloc_*` expects the
        // u32 GC tid.  Truncate `as u32` until gc_cache routing.
        let type_id = arraydescr.get_type_id() as u32;
        assert!(
            type_id != 0,
            "bh_new_array requires ArrayDescr.tid (descr.py:340) — got 0"
        );
        dynasm_alloc_varsize_typed_and_set_len(type_id, base_size, itemsize, len_offset, length)
            as i64
    }

    /// llmodel.py:790 bh_new_array_clear = bh_new_array.
    fn bh_new_array_clear(
        &self,
        length: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        self.bh_new_array(length, arraydescr)
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Resolves a vtable pointer through the installed gc_ll_descr.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, classptr: usize) -> Option<u32> {
        self.lookup_typeid_from_classptr(classptr)
    }

    /// llmodel.py:816 bh_call_i: ABI-correct dispatch via the shared call stub.
    ///
    /// On ARM64/x86-64 the C ABI assigns integer and float args to independent
    /// register files, so a typed `extern "C" fn(I × ints, F × floats) -> i64`
    /// transmute lands them correctly regardless of original interleaving.
    /// Routes through `majit_backend::call_stub::bh_call_i_dispatch` (Slice 0d
    /// of pyre-call-family-canonical-migration) which owns the arity table
    /// previously embedded in cranelift's `compiler.rs`.
    fn bh_call_i(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        if func == 0 {
            return 0;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args)
        }
    }

    /// llmodel.py:818 bh_call_r: GcRef-returning parallel of `bh_call_i`.
    /// `lltype.Ptr(lltype.GcStruct, ...)` lowers to a host pointer that
    /// matches `i64` on 64-bit, so we transmute via the shared int
    /// dispatcher and wrap the result. Without this override
    /// `bhimpl_residual_call_*_r` would silently no-op via the default
    /// trait impl at `majit-backend/lib.rs:1992`.
    fn bh_call_r(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> majit_ir::GcRef {
        if func == 0 {
            return majit_ir::GcRef::NULL;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        let raw = unsafe {
            majit_backend::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args)
        };
        majit_ir::GcRef(raw as usize)
    }

    /// llmodel.py:825 bh_call_f / descr.py:590-602 create_call_stub
    /// (`RESULT == lltype.Float`) parity: route through the f64-typed
    /// dispatcher so an f64-returning C callee delivers via xmm0 / d0
    /// instead of rax / x0. Without this override
    /// `bhimpl_residual_call_*_f` would silently no-op via the default
    /// trait impl at `majit-backend/lib.rs:2003`.
    fn bh_call_f(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        if func == 0 {
            return 0.0;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_f_dispatch(func as usize, &int_args, &float_args)
        }
    }

    /// llmodel.py:834 bh_call_v / descr.py:590-602 create_call_stub
    /// (`RESULT == lltype.Void`) parity: dispatch the funcptr through
    /// the void-typed `bh_call_v_dispatch` so a genuinely void C callee
    /// is called with the right C-ABI signature. Re-routing through
    /// `bh_call_i_dispatch` (a `extern "C" fn(...) -> i64` transmute)
    /// reads garbage from rax/x0 for true void returns.
    ///
    /// Without this override the canonical `residual_call_*_v` walker
    /// would silently no-op via the default trait impl
    /// (`majit-backend/lib.rs:2013`).
    fn bh_call_v(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        if func == 0 {
            return;
        }
        let (int_args, float_args) = majit_backend::call_stub::collect_call_args(
            &calldescr.arg_classes,
            args_i,
            args_r,
            args_f,
        );
        unsafe {
            majit_backend::call_stub::bh_call_v_dispatch(func as usize, &int_args, &float_args);
        }
    }

    /// llmodel.py:747-750 bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        addr: i64,
        offset: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        // llmodel.py:748-749: ofs, size, sign = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:749 assert)
        let size = descr.as_itemsize();
        let sign = descr.is_item_signed();
        // llmodel.py:750: return self.read_int_at_mem(addr, offset, size, sign)
        self.read_int_at_mem(addr, offset, size, sign)
    }

    /// llmodel.py:739-742 bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        addr: i64,
        offset: i64,
        newvalue: i64,
        descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:740-741: ofs, size, _ = self.unpack_arraydescr_size(descr)
        // ofs == 0 always for raw lengthless arrays (llmodel.py:741 assert)
        let size = descr.as_itemsize();
        // llmodel.py:742: self.write_int_at_mem(addr, offset, size, newvalue)
        self.write_int_at_mem(addr, offset, size, newvalue);
    }

    /// llmodel.py:752-753 bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        addr: i64,
        offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        // llmodel.py:753: return self.read_float_at_mem(addr, offset)
        self.read_float_at_mem(addr, offset)
    }

    /// llmodel.py:744-745 bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        addr: i64,
        offset: i64,
        newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
        // llmodel.py:745: self.write_float_at_mem(addr, offset, newvalue)
        self.write_float_at_mem(addr, offset, newvalue);
    }

    /// `llmodel.py:693-696 bh_getfield_gc_i` →
    /// `read_int_at_mem(struct, ofs, size, sign)`.  Threads the per-field
    /// `(offset, size, sign)` tuple from `BhDescr.unpack_fielddescr_size`
    /// to the size dispatch in `llmodel.py:467-478`.
    fn bh_getfield_gc_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let (offset, size, sign) = fielddescr.unpack_fielddescr_size();
        self.read_int_at_mem(struct_ptr, offset as i64, size, sign)
    }

    fn bh_getfield_gc_r(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        let offset = fielddescr.as_offset();
        GcRef(unsafe { *((struct_ptr as *const u8).add(offset) as *const usize) })
    }

    /// `llmodel.py:718-721 bh_setfield_gc_i` →
    /// `write_int_at_mem(struct, ofs, size, value)`.  Sign discarded by
    /// `unpack_fielddescr_size` consumer (`llmodel.py:651`); only
    /// `(offset, size)` reach the store.
    fn bh_setfield_gc_i(
        &self,
        struct_ptr: i64,
        value: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let (offset, size, _sign) = fielddescr.unpack_fielddescr_size();
        self.write_int_at_mem(struct_ptr, offset as i64, size, value);
    }

    fn bh_setfield_gc_r(
        &self,
        struct_ptr: i64,
        value: GcRef,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut usize) = value.0 };
    }

    /// llmodel.py:592-594 bh_getarrayitem_gc_i: ofs=base_size, size+sign
    /// from `unpack_arraydescr_size`; route through `read_int_at_mem`
    /// at `gcref + ofs + index*size`.
    fn bh_getarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let (base_size, itemsize, sign) = arraydescr.unpack_arraydescr_size();
        let offset = (base_size as i64) + index * (itemsize as i64);
        self.read_int_at_mem(array_ptr, offset, itemsize, sign)
    }

    /// model.py:254 / llmodel.py:585-588 bh_arraylen_gc.
    /// Read the length word from `arraydescr.lendescr.offset`.
    fn bh_arraylen_gc(
        &self,
        array_ptr: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let ofs = arraydescr
            .array_len_offset()
            .expect("bh_arraylen_gc requires ArrayDescr.lendescr");
        self.read_int_at_mem(array_ptr, ofs as i64, std::mem::size_of::<usize>(), true)
    }

    /// llmodel.py:597-599 bh_getarrayitem_gc_r: ofs=base_size, item width
    /// fixed at `WORD` (8 bytes).  Direct deref of `*const usize` mirrors
    /// `bh_getfield_gc_r`'s pattern so the GcRef carries the raw machine
    /// word from memory.
    fn bh_getarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> majit_ir::GcRef {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        let raw = unsafe { *((array_ptr as *const u8).offset(offset as isize) as *const usize) };
        majit_ir::GcRef(raw)
    }

    /// llmodel.py:603-606 bh_getarrayitem_gc_f: ofs=base_size, item
    /// width fixed at `sizeof(FLOATSTORAGE)` (8 bytes).  Routes through
    /// `read_float_at_mem` for the same `read_unaligned` safety as the
    /// field sibling.
    fn bh_getarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        self.read_float_at_mem(array_ptr, offset)
    }

    /// llmodel.py:609-611 bh_setarrayitem_gc_i.
    fn bh_setarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let offset = (base_size as i64) + index * (itemsize as i64);
        self.write_int_at_mem(array_ptr, offset, itemsize, newvalue);
    }

    /// llmodel.py:613-615 bh_setarrayitem_gc_r.
    fn bh_setarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: majit_ir::GcRef,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        unsafe {
            *((array_ptr as *mut u8).offset(offset as isize) as *mut usize) = newvalue.0;
        }
        // llmodel.py:495-497 `write_ref_at_mem`: raw_store + "write
        // barrier is implied above". `dynasm_write_barrier_from_array`
        // routes through `gc.jit_remember_young_pointer_from_array`
        // for the CARDS_SET transition matching
        // opassembler.py:953-960's array barrier path.
        if array_ptr != 0 {
            dynasm_write_barrier_from_array(array_ptr as u64);
        }
    }

    /// llmodel.py:618-621 bh_setarrayitem_gc_f.
    fn bh_setarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let offset = (base_size as i64) + index * 8;
        self.write_float_at_mem(array_ptr, offset, newvalue);
    }

    /// llmodel.py:705-707 bh_getfield_gc_f delegates to read_float_at_mem.
    /// `getfield_vable_f/rd>f` and the floating-point array reader rely
    /// on this — the trait default returns 0.0, which silently produces
    /// wrong results during blackhole resume on float vable fields.
    fn bh_getfield_gc_f(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        let offset = fielddescr.as_offset();
        // Route through `read_float_at_mem` (`runner.rs:611-615` —
        // `llmodel.py:490-491` parity) so misaligned struct fields use
        // `read_unaligned`. Direct `*const f64` deref was UB-prone on
        // misaligned vable float slots.
        self.read_float_at_mem(struct_ptr, offset as i64)
    }

    /// llmodel.py:728-730 bh_setfield_gc_f delegates to write_float_at_mem.
    /// Mirror of `bh_getfield_gc_f`; the trait default is a silent no-op
    /// which loses writes from `setfield_vable_f/rfd` during resume.
    fn bh_setfield_gc_f(
        &self,
        struct_ptr: i64,
        value: f64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        // Route through `write_float_at_mem` (`runner.rs:617-621` —
        // `llmodel.py:493-494` parity); see read sibling above for
        // alignment rationale.
        self.write_float_at_mem(struct_ptr, offset as i64, value);
    }

    /// compile_tmp_callback parity: register a placeholder for a pending
    /// CALL_ASSEMBLER target. The real code_addr is set by compile_loop.
    /// Until then, CALL_ASSEMBLER's generated code falls through to the
    /// helper trampoline which calls force_fn (interpreter re-execution).
    fn register_pending_target(
        &mut self,
        token_number: u64,
        _input_types: Vec<Type>,
        num_inputs: usize,
        _num_scalar_inputargs: usize,
        index_of_virtualizable: i32,
    ) {
        // Insert pending entry (code_addr = 0).  The CLT Arc + initial
        // `_ll_initial_locs` are populated here so `handle_call_assembler`
        // (rewrite.py:665-695) can look up callee metadata even before
        // the callee finishes compiling — mirrors cranelift
        // `compiler.rs:2427-2444`.
        Self::register_pending_call_assembler_target_static(
            token_number,
            num_inputs,
            index_of_virtualizable,
        );
    }

    fn compiled_fail_descr_layouts(
        &self,
        token: &JitCellToken,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        let trace_id = compiled.trace_id;
        Some(
            compiled
                .fail_descrs
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    crate::guard::layout_for_fail_descr(
                        d.as_fail_descr().expect("fail_descrs entry is FailDescr"),
                        Arc::as_ptr(d) as *const () as usize,
                        idx as u32,
                        trace_id,
                    )
                })
                .collect(),
        )
    }

    fn compiled_trace_fail_descr_layouts(
        &self,
        token: &JitCellToken,
        trace_id: u64,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        let compiled = Self::get_compiled(token);
        if compiled.trace_id == trace_id {
            return Some(
                compiled
                    .fail_descrs
                    .iter()
                    .enumerate()
                    .map(|(idx, d)| {
                        crate::guard::layout_for_fail_descr(
                            d.as_fail_descr().expect("fail_descrs entry is FailDescr"),
                            Arc::as_ptr(d) as *const () as usize,
                            idx as u32,
                            trace_id,
                        )
                    })
                    .collect(),
            );
        }
        // Search bridge fail_descrs in asmmemmgr_blocks.
        let blocks = token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                if bridge.trace_id == trace_id {
                    return Some(
                        bridge
                            .fail_descrs
                            .iter()
                            .enumerate()
                            .map(|(idx, d)| {
                                crate::guard::layout_for_fail_descr(
                                    d.as_fail_descr().expect("fail_descrs entry is FailDescr"),
                                    Arc::as_ptr(d) as *const () as usize,
                                    idx as u32,
                                    trace_id,
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        None
    }

    fn compiled_bridge_fail_descr_layouts(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Vec<majit_backend::FailDescrLayout>> {
        // RPython faildescr lookup is by object identity, never misses.
        // majit query-style callers (`bridge_was_compiled` etc.) probe
        // by (trace_id, fail_index) and must treat the miss as `None`
        // — match cranelift's `?` semantics in
        // `compiler.rs:11723 compiled_bridge_fail_descr_layouts`.
        let source_descr =
            Self::try_find_descr(original_token, source_trace_id, source_fail_index)?;
        let bridge_addr = self.lookup_bridge_addr(Arc::as_ptr(&source_descr) as *const () as usize);
        if bridge_addr == 0 {
            return None;
        }
        let blocks = original_token.asmmemmgr_blocks();
        for block in blocks.iter() {
            if let Some(bridge) = block.downcast_ref::<CompiledCode>() {
                let addr = codebuf::buffer_ptr(&bridge.buffer) as usize;
                if addr == bridge_addr {
                    let bridge_trace_id = bridge.trace_id;
                    return Some(
                        bridge
                            .fail_descrs
                            .iter()
                            .enumerate()
                            .map(|(idx, d)| {
                                crate::guard::layout_for_fail_descr(
                                    d.as_fail_descr().expect("fail_descrs entry is FailDescr"),
                                    Arc::as_ptr(d) as *const () as usize,
                                    idx as u32,
                                    bridge_trace_id,
                                )
                            })
                            .collect(),
                    );
                }
            }
        }
        None
    }

    fn compiled_bridge_descr_arc(
        &self,
        original_token: &JitCellToken,
        source_trace_id: u64,
        source_fail_index: u32,
    ) -> Option<Arc<dyn majit_ir::Descr>> {
        let source_descr =
            Self::try_find_descr(original_token, source_trace_id, source_fail_index)?;
        let bridge_addr = self.lookup_bridge_addr(Arc::as_ptr(&source_descr) as *const () as usize);
        if bridge_addr == 0 {
            return None;
        }
        Some(source_descr)
    }

    fn find_source_fail_descr(
        &self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
    ) -> Option<Arc<dyn majit_ir::Descr>> {
        Self::try_find_descr(token, trace_id, fail_index)
    }

    fn update_fail_descr_recovery_layout(
        &mut self,
        token: &JitCellToken,
        trace_id: u64,
        fail_index: u32,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        // Slice NN: backend no longer caches recovery_layout (PyPy parity —
        // `resume.py:450-488` decodes on-demand).  The metainterp's
        // `StoredExitLayout.recovery_layout` is the canonical store;
        // `patch_backend_guard_recovery_layouts_for_trace` updates it
        // directly without round-tripping through the backend.  Return
        // `true` for descrs we know about so the patch loop's "did patch"
        // tracking stays accurate.
        Self::try_find_descr(token, trace_id, fail_index).is_some()
    }

    /// `pyjitpl.py:2297 self.cpu.setup_once()` parity, dispatched by
    /// `MetaInterpStaticData::_setup_once` under the
    /// `globaldata.initialized` gate (`pyjitpl.py:2292-2303`).  All
    /// per-CPU descrs (notably `propagate_exception_descr` via
    /// `set_propagate_exception_descr`) must already be installed
    /// when this runs; the helpers we materialise here bake those
    /// descr pointers as immediates and assert non-zero on build.
    ///
    /// PyPy's `llsupport/assembler.py:97 setup_once` builds the
    /// propagate trampoline + every `_build_malloc_slowpath` variant
    /// (`fixed` / `varsize` / `str` / `unicode`).  Pyre's x86 path so
    /// far implements only `fixed`; varsize/str/unicode are inlined
    /// at the per-callsite emitter and remain to port.
    fn setup_once(&mut self) {
        #[cfg(target_arch = "x86_64")]
        {
            self.arch_cpu_ext
                .ensure_propagate_exception_path(&self.descr_attachments);
            self.arch_cpu_ext
                .ensure_malloc_slowpath_fixed(&self.descr_attachments);
        }
    }
    fn finish_once(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_backend::Backend;
    use majit_backend::jitframe::{
        FIRST_ITEM_OFFSET, JF_DESCR_OFS, JF_FORCE_DESCR_OFS, JF_FORWARD_OFS, JF_FRAME_INFO_OFS,
        JF_FRAME_OFS, JF_GUARD_EXC_OFS, JF_SAVEDATA_OFS, JITFRAME_FIXED_SIZE, LENGTHOFS, SIGN_SIZE,
    };
    use majit_gc::collector::{GcConfig, MiniMarkGC};
    use majit_gc::header::header_of;
    use majit_gc::trace::TypeInfo;
    use majit_ir::{
        CallDescr, DescrRef, EffectInfo, ExtraEffect, InputArg, OopSpecIndex, OpCode, Type, Value,
    };
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn install_test_libc_jitframe_tracer() {
        majit_gc::shadow_stack::register_libc_jitframe_tracer(
            majit_backend::jitframe::jitframe_custom_trace,
        );
    }

    #[derive(Debug)]
    struct TestCallAssemblerDescr {
        arg_types: Vec<Type>,
        result_type: Type,
        target_token: u64,
        vable_expansion: Option<majit_ir::VableExpansion>,
    }

    #[derive(Debug)]
    struct TestPlainCallDescr {
        arg_types: Vec<Type>,
        result_type: Type,
    }

    impl majit_ir::Descr for TestCallAssemblerDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }

        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }

        fn as_loop_token_descr(&self) -> Option<&dyn majit_ir::LoopTokenDescr> {
            Some(self)
        }
    }

    impl majit_ir::LoopTokenDescr for TestCallAssemblerDescr {
        fn loop_token_number(&self) -> u64 {
            self.target_token
        }
    }

    impl CallDescr for TestCallAssemblerDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }

        fn result_type(&self) -> Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn call_target_token(&self) -> Option<u64> {
            Some(self.target_token)
        }

        fn get_extra_info(&self) -> &EffectInfo {
            static INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
            &INFO
        }

        fn vable_expansion(&self) -> Option<&majit_ir::VableExpansion> {
            self.vable_expansion.as_ref()
        }
    }

    impl majit_ir::Descr for TestPlainCallDescr {
        fn index(&self) -> u32 {
            u32::MAX
        }

        fn as_call_descr(&self) -> Option<&dyn CallDescr> {
            Some(self)
        }
    }

    impl CallDescr for TestPlainCallDescr {
        fn arg_types(&self) -> &[Type] {
            &self.arg_types
        }

        fn result_type(&self) -> Type {
            self.result_type
        }

        fn result_size(&self) -> usize {
            8
        }

        fn get_extra_info(&self) -> &EffectInfo {
            static INFO: EffectInfo =
                EffectInfo::const_new(ExtraEffect::CanRaise, OopSpecIndex::None);
            &INFO
        }
    }

    fn mk_op(opcode: OpCode, args: &[OpRef], pos: u32) -> Op {
        let mut op = Op::new(opcode, args);
        op.pos = OpRef::op_typed(pos, opcode.result_type());
        op
    }

    fn make_call_assembler_descr(
        target: &JitCellToken,
        arg_types: Vec<Type>,
        result_type: Type,
    ) -> DescrRef {
        Arc::new(TestCallAssemblerDescr {
            arg_types,
            result_type,
            target_token: target.number,
            vable_expansion: None,
        })
    }

    fn make_call_assembler_descr_with_expansion(
        target: &JitCellToken,
        arg_types: Vec<Type>,
        result_type: Type,
        expansion: majit_ir::VableExpansion,
    ) -> DescrRef {
        Arc::new(TestCallAssemblerDescr {
            arg_types,
            result_type,
            target_token: target.number,
            vable_expansion: Some(expansion),
        })
    }

    fn make_plain_call_descr(arg_types: Vec<Type>, result_type: Type) -> DescrRef {
        Arc::new(TestPlainCallDescr {
            arg_types,
            result_type,
        })
    }

    extern "C" fn return_ref_passthrough(arg: i64) -> i64 {
        arg
    }

    static TEST_HELPER_ALLOC_TYPE_ID: AtomicU32 = AtomicU32::new(u32::MAX);

    const TEST_HELPER_MARKER: i64 = 0x5a5a5a5a_i64;

    extern "C" fn alloc_marked_ref() -> i64 {
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let gc = guard
                .as_mut()
                .expect("alloc_marked_ref requires an active dynasm GC");
            let type_id = TEST_HELPER_ALLOC_TYPE_ID.load(Ordering::Relaxed);
            assert_ne!(type_id, u32::MAX, "test helper type id not initialized");
            let obj = gc.alloc_nursery_typed(type_id, 16);
            unsafe {
                *(obj.0 as *mut i64) = TEST_HELPER_MARKER;
            }
            obj.0 as i64
        })
    }

    extern "C" fn alloc_marked_ref_collecting() -> i64 {
        DYNASM_ACTIVE_GC.with(|cell| {
            let mut guard = cell.borrow_mut();
            let gc = guard
                .as_mut()
                .expect("alloc_marked_ref_collecting requires an active dynasm GC");
            let type_id = TEST_HELPER_ALLOC_TYPE_ID.load(Ordering::Relaxed);
            assert_ne!(type_id, u32::MAX, "test helper type id not initialized");
            let obj = gc.alloc_nursery_typed(type_id, 16);
            unsafe {
                *(obj.0 as *mut i64) = TEST_HELPER_MARKER;
            }
            let root_depth = majit_gc::shadow_stack::depth();
            let ss_idx = majit_gc::shadow_stack::push(obj);
            let _bump = gc.alloc_nursery_typed(type_id, 80);
            let updated = majit_gc::shadow_stack::get(ss_idx);
            majit_gc::shadow_stack::pop_to(root_depth);
            updated.0 as i64
        })
    }

    fn install_call_assembler_test_layout() {
        crate::register_jitframe_layout(crate::JitFrameLayoutInfo {
            jitframe_descrs: Some(majit_gc::rewrite::JitFrameDescrs {
                jitframe_tid: crate::jitframe_gc_type_id().unwrap_or(u32::MAX),
                jitframe_fixed_size: JITFRAME_FIXED_SIZE,
                jf_frame_info_ofs: JF_FRAME_INFO_OFS,
                jf_descr_ofs: JF_DESCR_OFS,
                jf_force_descr_ofs: JF_FORCE_DESCR_OFS,
                jf_savedata_ofs: JF_SAVEDATA_OFS,
                jf_guard_exc_ofs: JF_GUARD_EXC_OFS,
                jf_forward_ofs: JF_FORWARD_OFS,
                jf_frame_ofs: JF_FRAME_OFS,
                jf_frame_baseitemofs: FIRST_ITEM_OFFSET,
                jf_frame_lengthofs: JF_FRAME_OFS + LENGTHOFS,
                sign_size: SIGN_SIZE,
            }),
        });
    }

    fn make_call_assembler_backend() -> DynasmBackend {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));
        backend
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr
    /// Verify the dynasm backend's gc_ll_descr round-trips a registered
    /// vtable→type_id mapping (the same contract Cranelift uses).
    #[test]
    fn test_backend_typeid_from_classptr_via_gc_ll_descr() {
        let mut gc = MiniMarkGC::new();
        let int_tid = gc.register_type(TypeInfo::simple(16));
        let int_vtable: usize = 0x2222_3300;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, int_vtable, int_tid);

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let resolved = backend.get_typeid_from_classptr_if_gcremovetypeptr(int_vtable);
        assert_eq!(resolved, Some(int_tid));
        let unknown = backend.get_typeid_from_classptr_if_gcremovetypeptr(0xCAFE_F00D);
        assert_eq!(unknown, None);
    }

    #[test]
    fn test_backend_installs_active_gc_guard_hooks() {
        let mut gc = MiniMarkGC::new();
        let obj_tid = gc.register_type(TypeInfo::object(16));
        let obj = gc.alloc_with_type(obj_tid, 16);

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        assert!(majit_gc::supports_guard_gc_type());
        assert!(majit_gc::check_is_object(obj));
        assert_eq!(majit_gc::get_actual_typeid(obj), Some(obj_tid));
        assert_eq!(majit_gc::typeid_is_object(obj_tid), Some(true));
    }

    #[test]
    fn test_input_initial_locs_match_frame_relative_entry_offsets() {
        assert_eq!(
            DynasmBackend::input_initial_loc(0),
            (DynasmBackend::input_slot(0) * crate::jitframe::SIZEOFSIGNED) as i32
        );
        assert_eq!(
            DynasmBackend::input_initial_loc(1),
            (DynasmBackend::input_slot(1) * crate::jitframe::SIZEOFSIGNED) as i32
        );
    }

    #[test]
    fn compile_loop_records_token_inputarg_types() {
        let mut backend = DynasmBackend::new();
        // `compile.py:665-674 make_and_attach_done_descrs` parity:
        // FINISH emission stamps the cpu-attached singleton Arc into
        // `compiled.fail_descrs`; without attachment the backend has
        // nothing to push.  Production reaches this state through
        // `MetaInterp::new`; backend-only tests must invoke this helper.
        backend.attach_default_test_descrs();
        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        // Match the typed `InputArg{Ref,Int}` boxes registered by the
        // backend regalloc — variant-aware Eq makes Untyped(N) and
        // InputArg{Ref,Int}(N) distinct keys.
        let ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
            OpRef::NONE.raw(),
        )];

        let mut token = JitCellToken::new(1499);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        assert_eq!(token.inputarg_types, vec![Type::Ref, Type::Int]);
    }

    #[test]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn compile_loop_accepts_nonzero_inputarg_indices() {
        let mut backend = DynasmBackend::new();
        let inputargs = vec![InputArg::new_int(10), InputArg::new_int(20)];
        let ops = vec![
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(10), OpRef::input_arg_int(20)],
                30,
            ),
            mk_op(OpCode::Finish, &[OpRef::int_op(30)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4), Value::Int(5)]);
        assert_eq!(backend.get_int_value(&frame, 0), 9);
    }

    #[test]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_gc_alloc_and_init_with_configured_runtime() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::new();
        gc.register_type(TypeInfo::simple(16));
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 0xDEAD_i64);
        consts.insert(10006, 16_i64);
        consts.insert(10007, 1_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10005),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10006),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1500);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 1);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 1);
    }

    #[test]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_collecting_alloc_preserves_initialized_header_and_payload() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 0xDEAD_i64);
        consts.insert(10006, 16_i64);
        consts.insert(10007, 1_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10005),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(0),
                    OpRef::int_op(10006),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 1),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 2),
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 3),
            mk_op(
                OpCode::Finish,
                &[
                    OpRef::ref_op(0),
                    OpRef::ref_op(1),
                    OpRef::ref_op(2),
                    OpRef::ref_op(3),
                ],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(1501);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { (*header_of(obj.0)).type_id() }, 1);
        assert_eq!(unsafe { *(obj.0 as *const u64) }, 0xDEAD);
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 1);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_varsize_frame_fastpath_does_not_overlap_previous_object_payload() {
        let mut gc = MiniMarkGC::new();
        gc.register_type(TypeInfo::simple(24));

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(10000, 32_i64);
        consts.insert(10001, -8_i64);
        consts.insert(10002, 1_i64);
        consts.insert(10003, 8_i64);
        consts.insert(10004, 0_i64);
        consts.insert(10005, 16_i64);
        consts.insert(10006, 111_i64);
        consts.insert(10007, 3_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_int(0)];
        let ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(10000)], 0),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::input_arg_int(0),
                    OpRef::int_op(10001),
                    OpRef::int_op(10002),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::input_arg_int(0),
                    OpRef::int_op(10004),
                    OpRef::int_op(10004),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::input_arg_int(0),
                    OpRef::int_op(10005),
                    OpRef::int_op(10006),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::CallMallocNurseryVarsizeFrame,
                &[OpRef::input_arg_int(0)],
                1,
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(1),
                    OpRef::int_op(10001),
                    OpRef::int_op(10007),
                    OpRef::int_op(10003),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];

        let mut token = JitCellToken::new(1502);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(64)]);
        let obj = backend.get_ref_value(&frame, 0);
        assert!(!obj.is_null());
        assert_eq!(unsafe { *((obj.0 + 16) as *const i64) }, 111);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_varsize_frame_gcstore_round_trips_first_user_slot() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(10000, 264_i64);
        consts.insert(10001, 256_i64);
        consts.insert(10002, 8_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let ops = vec![
            mk_op(
                OpCode::CallMallocNurseryVarsizeFrame,
                &[OpRef::int_op(10000)],
                1,
            ),
            mk_op(
                OpCode::GcStore,
                &[
                    OpRef::ref_op(1),
                    OpRef::int_op(10001),
                    OpRef::input_arg_ref(0),
                    OpRef::int_op(10002),
                ],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::GcLoadR,
                &[OpRef::ref_op(1), OpRef::int_op(10001), OpRef::int_op(10002)],
                2,
            ),
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];

        let mut token = JitCellToken::new(1503);
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_round_trips_ref_input_through_rewritten_jitframe() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));
        install_test_libc_jitframe_tracer();
        install_call_assembler_test_layout();

        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0)],
            OpRef::NONE.raw(),
        )];
        let mut callee_token = JitCellToken::new(1600);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();
        let direct = backend.execute_token(&callee_token, &[Value::Ref(payload)]);
        assert_eq!(backend.get_ref_value(&direct, 0), payload);

        let mut call = mk_op(OpCode::CallAssemblerR, &[OpRef::input_arg_ref(0)], 1);
        call.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1601);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload)]);
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_supports_direct_self_recursive_dispatch() {
        let mut backend = make_call_assembler_backend();

        let inputargs = vec![InputArg::new_int(0)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1602);
        backend.register_pending_target(token.number, vec![Type::Int], 1, 1, -1);
        // resoperation.py:719 InputArgInt — slot 0 is `InputArg::new_int(0)`,
        // referenced via `OpRef::input_arg_int(0)`. Variant-aware Eq/Hash
        // treats `IntOp(0)` and `InputArgInt(0)` as disjoint Box classes.
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::int_op(1)], OpRef::NONE.raw());
        guard.fail_args = Some(vec![OpRef::input_arg_int(0)].into());
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::IntGt,
                &[OpRef::input_arg_int(0), OpRef::int_op(101)],
                1,
            ),
            guard,
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(0), OpRef::int_op(100)],
                2,
            ),
            {
                let mut call = mk_op(OpCode::CallAssemblerI, &[OpRef::int_op(2)], 3);
                call.descr = Some(make_call_assembler_descr(
                    &token,
                    vec![Type::Int],
                    Type::Int,
                ));
                call
            },
            mk_op(OpCode::IntAdd, &[OpRef::int_op(3), OpRef::int_op(100)], 4),
            mk_op(OpCode::Finish, &[OpRef::int_op(4)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);
        backend.set_constants(HashMap::new());
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_int(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 4);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_supports_direct_self_recursive_ref_dispatch() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_int(0), InputArg::new_ref(1)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1603);
        backend.register_pending_target(token.number, vec![Type::Int, Type::Ref], 2, 2, -1);
        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::int_op(2)], OpRef::NONE.raw());
        guard.fail_args = Some(vec![OpRef::input_arg_int(0), OpRef::input_arg_ref(1)].into());
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::input_arg_int(0), OpRef::int_op(101)],
                2,
            ),
            guard,
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(0), OpRef::int_op(100)],
                3,
            ),
            {
                let mut call = mk_op(
                    OpCode::CallAssemblerR,
                    &[OpRef::int_op(3), OpRef::input_arg_ref(1)],
                    4,
                );
                call.descr = Some(make_call_assembler_descr(
                    &token,
                    vec![Type::Int, Type::Ref],
                    Type::Ref,
                ));
                call
            },
            mk_op(OpCode::Finish, &[OpRef::ref_op(4)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Int(0), Value::Ref(payload)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);
        backend.set_constants(HashMap::new());
        let bridge_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_int(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Int(4), Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_self_recursive_virtualizable_ref_arg_preserves_input0() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1614);
        token.virtualizable_arg_index = Some(0);
        backend.register_pending_target(token.number, vec![Type::Ref, Type::Int], 2, 2, 0);

        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::int_op(2)], OpRef::NONE.raw());
        guard.fail_args = Some(vec![OpRef::input_arg_ref(0)].into());
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::input_arg_int(1), OpRef::int_op(101)],
                2,
            ),
            guard,
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(1), OpRef::int_op(100)],
                3,
            ),
            {
                let mut call = mk_op(
                    OpCode::CallAssemblerR,
                    &[OpRef::input_arg_ref(0), OpRef::int_op(3)],
                    4,
                );
                call.descr = Some(make_call_assembler_descr(
                    &token,
                    vec![Type::Ref, Type::Int],
                    Type::Ref,
                ));
                call
            },
            mk_op(OpCode::Finish, &[OpRef::ref_op(4)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(0)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);
        backend.set_constants(HashMap::new());
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &[InputArg::new_ref(0)],
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(32)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_uses_gc_rewritten_vable_frame_without_double_materializing() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(24));
        let payload = gc.alloc_with_type(payload_tid, 24);
        const MARKER: i64 = 0x3456_789a_i64;
        unsafe {
            *((payload.0 as *mut u8).add(16) as *mut i64) = MARKER;
        }

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let field_descr: DescrRef =
            Arc::new(majit_ir::SimpleFieldDescr::new(0, 16, 8, Type::Int, false));
        let mut entry_getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 2);
        entry_getfield.descr = Some(field_descr);
        let callee_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            entry_getfield,
            mk_op(
                OpCode::IntAdd,
                &[OpRef::input_arg_int(1), OpRef::int_op(2)],
                3,
            ),
            mk_op(OpCode::Finish, &[OpRef::int_op(3)], OpRef::NONE.raw()),
        ];
        let mut callee_token = JitCellToken::new(1617);
        callee_token.virtualizable_arg_index = Some(0);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let expansion = majit_ir::VableExpansion {
            scalar_fields: vec![(16, Type::Int)],
            array_struct_offset: 0,
            array_ptr_offset: 0,
            num_array_items: 0,
            const_overrides: vec![],
            arg_overrides: vec![],
        };
        let mut call = mk_op(
            OpCode::CallAssemblerI,
            &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
            2,
        );
        call.descr = Some(make_call_assembler_descr_with_expansion(
            &callee_token,
            vec![Type::Ref, Type::Int],
            Type::Int,
            expansion,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let caller_ops = vec![
            call,
            mk_op(OpCode::Finish, &[OpRef::int_op(2)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1618);
        caller_token.virtualizable_arg_index = Some(0);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload), Value::Int(7)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), MARKER + 7);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_self_recursive_virtualizable_bridge_reads_input0_from_compiled_bridge() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(24));
        let payload = gc.alloc_with_type(payload_tid, 24);
        const MARKER: i64 = 0x5a5a_1234_7788_i64;
        unsafe {
            *((payload.0 as *mut u8).add(16) as *mut i64) = MARKER;
        }

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 0);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1615);
        token.virtualizable_arg_index = Some(0);
        backend.register_pending_target(token.number, vec![Type::Ref, Type::Int], 2, 2, 0);

        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::int_op(2)], OpRef::NONE.raw());
        guard.fail_args = Some(vec![OpRef::input_arg_ref(0)].into());
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            mk_op(
                OpCode::IntGt,
                &[OpRef::input_arg_int(1), OpRef::int_op(101)],
                2,
            ),
            guard,
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(1), OpRef::int_op(100)],
                3,
            ),
            {
                let mut call = mk_op(
                    OpCode::CallAssemblerR,
                    &[OpRef::input_arg_ref(0), OpRef::int_op(3)],
                    4,
                );
                call.descr = Some(make_call_assembler_descr(
                    &token,
                    vec![Type::Ref, Type::Int],
                    Type::Ref,
                ));
                call
            },
            mk_op(OpCode::Finish, &[OpRef::ref_op(4)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(0)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);
        backend.set_constants(HashMap::new());
        let field_descr: DescrRef =
            Arc::new(majit_ir::SimpleFieldDescr::new(0, 16, 8, Type::Int, false));
        let mut getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 1);
        getfield.descr = Some(field_descr);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            getfield,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &[InputArg::new_ref(0)],
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(32)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), MARKER);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_double_recursive_virtualizable_call_assembler_keeps_entry_input0_live() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(24));
        let payload = gc.alloc_with_type(payload_tid, 24);
        unsafe {
            *((payload.0 as *mut u8).add(16) as *mut i64) = 1;
        }

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_int(1)];
        let mut constants = HashMap::new();
        constants.insert(100, 1);
        constants.insert(101, 2);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1616);
        token.virtualizable_arg_index = Some(0);
        backend.register_pending_target(token.number, vec![Type::Ref, Type::Int], 2, 2, 0);

        let field_descr: DescrRef =
            Arc::new(majit_ir::SimpleFieldDescr::new(0, 16, 8, Type::Int, false));
        let mut entry_getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 2);
        entry_getfield.descr = Some(field_descr.clone());

        let mut guard = mk_op(OpCode::GuardTrue, &[OpRef::int_op(3)], OpRef::NONE.raw());
        guard.fail_args = Some(vec![OpRef::input_arg_ref(0)].into());

        let mut call1 = mk_op(
            OpCode::CallAssemblerI,
            &[OpRef::input_arg_ref(0), OpRef::int_op(4)],
            6,
        );
        call1.descr = Some(make_call_assembler_descr(
            &token,
            vec![Type::Ref, Type::Int],
            Type::Int,
        ));
        let mut call2 = mk_op(
            OpCode::CallAssemblerI,
            &[OpRef::input_arg_ref(0), OpRef::int_op(5)],
            7,
        );
        call2.descr = Some(make_call_assembler_descr(
            &token,
            vec![Type::Ref, Type::Int],
            Type::Int,
        ));

        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
            entry_getfield,
            mk_op(
                OpCode::IntGe,
                &[OpRef::input_arg_int(1), OpRef::int_op(101)],
                3,
            ),
            guard,
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(1), OpRef::int_op(100)],
                4,
            ),
            mk_op(
                OpCode::IntSub,
                &[OpRef::input_arg_int(1), OpRef::int_op(101)],
                5,
            ),
            call1,
            call2,
            mk_op(OpCode::IntAdd, &[OpRef::int_op(6), OpRef::int_op(7)], 8),
            mk_op(OpCode::Finish, &[OpRef::int_op(8)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(1)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);
        backend.set_constants(HashMap::new());
        let mut bridge_getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 1);
        bridge_getfield.descr = Some(field_descr);
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            bridge_getfield,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_int(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &[InputArg::new_ref(0)],
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload), Value::Int(10)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_int_value(&frame, 0), 89);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_bridge_materializes_register_ref_inputs_for_resolve_opref_ops() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let wrong_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        let wrong_vtable: usize = 0x2222_4400;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, wrong_vtable, wrong_tid);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let mut constants = HashMap::new();
        constants.insert(100, wrong_vtable as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1604);
        let mut guard = mk_op(
            OpCode::GuardClass,
            &[OpRef::input_arg_ref(0), OpRef::int_op(100)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(vec![OpRef::input_arg_ref(0)].into());
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(&token, &[Value::Ref(payload)]);
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);

        let mut bridge_constants = HashMap::new();
        bridge_constants.insert(200, return_ref_passthrough as *const () as usize as i64);
        backend.set_constants(bridge_constants);
        let mut bridge_value = mk_op(
            OpCode::CondCallValueR,
            &[OpRef::input_arg_ref(0), OpRef::int_op(200)],
            1,
        );
        bridge_value.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let bridge_ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            bridge_value,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_bridge_materializes_two_register_ref_inputs_before_unused_raw_load() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let frame_tid = gc.register_type(TypeInfo::simple(24));
        let other_tid = gc.register_type(TypeInfo::simple(16));
        let wrong_tid = gc.register_type(TypeInfo::simple(16));
        let frame_payload = gc.alloc_with_type(frame_tid, 24);
        let second_payload = gc.alloc_with_type(other_tid, 16);

        const MARKER: i64 = 0x1357_2468_1122_i64;
        unsafe {
            *((frame_payload.0 as *mut u8).add(16) as *mut i64) = MARKER;
        }

        let wrong_vtable: usize = 0x3333_5500;
        majit_gc::GcAllocator::register_vtable_for_type(&mut gc, wrong_vtable, wrong_tid);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let mut constants = HashMap::new();
        constants.insert(100, wrong_vtable as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1618);
        let mut guard = mk_op(
            OpCode::GuardClass,
            &[OpRef::input_arg_ref(1), OpRef::int_op(100)],
            OpRef::NONE.raw(),
        );
        guard.fail_args = Some(vec![OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)].into());
        let ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            guard,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let failed = backend.execute_token(
            &token,
            &[Value::Ref(frame_payload), Value::Ref(second_payload)],
        );
        let guard_fail_index = backend.get_latest_descr(&failed).fail_index();
        let guard_trace_id = backend.get_latest_descr(&failed).trace_id();
        let guard_descr = DynasmBackend::find_descr(&token, guard_trace_id, guard_fail_index);

        backend.set_constants(HashMap::new());
        let field_descr: DescrRef =
            Arc::new(majit_ir::SimpleFieldDescr::new(0, 16, 8, Type::Int, false));
        let mut getfield = mk_op(OpCode::GetfieldRawI, &[OpRef::input_arg_ref(0)], 2);
        getfield.descr = Some(field_descr);
        let bridge_ops = vec![
            mk_op(
                OpCode::Label,
                &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
            getfield,
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(1)],
                OpRef::NONE.raw(),
            ),
        ];
        backend
            .compile_bridge(
                guard_descr.as_fail_descr().unwrap(),
                &inputargs,
                &bridge_ops,
                &token,
                &[],
                None,
            )
            .unwrap();

        let frame = backend.execute_token(
            &token,
            &[Value::Ref(frame_payload), Value::Ref(second_payload)],
        );
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), second_payload);
    }

    #[test]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_label_uses_absolute_jitframe_input_slots_for_resolve_opref_ops() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let inputargs = vec![InputArg::new_ref(0)];
        let mut constants = HashMap::new();
        constants.insert(200, return_ref_passthrough as *const () as usize as i64);
        backend.set_constants(constants);

        let mut token = JitCellToken::new(1605);
        let mut passthrough = mk_op(
            OpCode::CondCallValueR,
            &[OpRef::input_arg_ref(0), OpRef::int_op(200)],
            1,
        );
        passthrough.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            mk_op(OpCode::Label, &[OpRef::input_arg_ref(0)], OpRef::NONE.raw()),
            passthrough,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        backend.compile_loop(&inputargs, &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_ref_from_immediately_preceding_callr() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0)],
            OpRef::NONE.raw(),
        )];
        let mut callee_token = JitCellToken::new(1604);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let mut constants = HashMap::new();
        constants.insert(200, return_ref_passthrough as usize as i64);
        backend.set_constants(constants);

        let mut plain_call = mk_op(
            OpCode::CallR,
            &[OpRef::int_op(200), OpRef::input_arg_ref(0)],
            1,
        );
        plain_call.descr = Some(make_plain_call_descr(vec![Type::Ref], Type::Ref));
        let mut call_asm = mk_op(OpCode::CallAssemblerR, &[OpRef::ref_op(1)], 2);
        call_asm.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            plain_call,
            call_asm,
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1605);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_helper_ref_when_rewritten_with_second_ref_arg() {
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 1 << 20,
            large_object_threshold: 1 << 20,
            ..GcConfig::default()
        });
        gc.register_type(TypeInfo::simple(16));
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);
        let ec = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        install_test_libc_jitframe_tracer();

        let mut backend = DynasmBackend::new();
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let callee_ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0)],
            OpRef::NONE.raw(),
        )];
        let mut callee_token = JitCellToken::new(1606);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let mut constants = HashMap::new();
        constants.insert(201, return_ref_passthrough as *const () as usize as i64);
        backend.set_constants(constants);

        let mut plain_call = mk_op(
            OpCode::CallR,
            &[OpRef::int_op(201), OpRef::input_arg_ref(0)],
            2,
        );
        plain_call.descr = Some(make_plain_call_descr(vec![Type::Ref], Type::Ref));
        let mut call_asm = mk_op(
            OpCode::CallAssemblerR,
            &[OpRef::ref_op(2), OpRef::input_arg_ref(1)],
            3,
        );
        call_asm.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref, Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let caller_ops = vec![
            plain_call,
            call_asm,
            mk_op(OpCode::Finish, &[OpRef::ref_op(3)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1607);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload), Value::Ref(ec)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_fresh_callr_ref_with_second_ref_arg_across_collecting_callee_alloc()
     {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let ec = gc.alloc_with_type(payload_tid, 16);

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(
            203,
            alloc_marked_ref_collecting as *const () as usize as i64,
        );
        consts.insert(202, 32_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let callee_ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(202)], 2),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut callee_token = JitCellToken::new(1610);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let mut plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(203)], 1);
        plain_call.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let mut call_asm = mk_op(
            OpCode::CallAssemblerR,
            &[OpRef::input_arg_ref(1), OpRef::input_arg_ref(0)],
            2,
        );
        call_asm.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref, Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            plain_call,
            call_asm,
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];
        let mut caller_consts = HashMap::new();
        caller_consts.insert(
            203,
            alloc_marked_ref_collecting as *const () as usize as i64,
        );
        backend.set_constants(caller_consts);
        let mut caller_token = JitCellToken::new(1611);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(ec)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_two_ref_inputs_across_collecting_callee_alloc() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(24));
        let ec_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 24);
        let ec = gc.alloc_with_type(ec_tid, 16);
        const MARKER: i64 = 0x1357_2468_1122_i64;
        unsafe {
            *((payload.0 as *mut u8).add(16) as *mut i64) = MARKER;
        }

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(202, 32_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let callee_ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(202)], 2),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut callee_token = JitCellToken::new(1612);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let mut call_asm = mk_op(
            OpCode::CallAssemblerR,
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            2,
        );
        call_asm.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref, Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0), InputArg::new_ref(1)];
        let caller_ops = vec![
            call_asm,
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1613);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload), Value::Ref(ec)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        unsafe {
            assert_eq!(*((result.0 as *const u8).add(16) as *const i64), MARKER);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_ref_input_across_collecting_callee_alloc() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));
        let payload = gc.alloc_with_type(payload_tid, 16);
        assert!(gc.pin(payload));

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(202, 32_i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![
            mk_op(OpCode::CallMallocNursery, &[OpRef::int_op(202)], 1),
            mk_op(
                OpCode::Finish,
                &[OpRef::input_arg_ref(0)],
                OpRef::NONE.raw(),
            ),
        ];
        let mut callee_token = JitCellToken::new(1608);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();
        let direct = backend.execute_token(&callee_token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&direct).is_finish());
        assert_eq!(backend.get_ref_value(&direct, 0), payload);

        let mut call = mk_op(OpCode::CallAssemblerR, &[OpRef::input_arg_ref(0)], 1);
        call.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref],
            Type::Ref,
        ));
        let caller_inputargs = vec![InputArg::new_ref(0)];
        let caller_ops = vec![
            call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(1)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1609);
        backend
            .compile_loop(&caller_inputargs, &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[Value::Ref(payload)]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        assert_eq!(backend.get_ref_value(&frame, 0), payload);
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_call_assembler_preserves_fresh_callr_ref_across_collecting_frame_alloc() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(203, alloc_marked_ref as *const () as usize as i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let callee_inputargs = vec![InputArg::new_ref(0)];
        let callee_ops = vec![mk_op(
            OpCode::Finish,
            &[OpRef::input_arg_ref(0)],
            OpRef::NONE.raw(),
        )];
        let mut callee_token = JitCellToken::new(1610);
        backend
            .compile_loop(&callee_inputargs, &callee_ops, &mut callee_token)
            .unwrap();

        let mut consts = HashMap::new();
        consts.insert(203, alloc_marked_ref as *const () as usize as i64);
        backend.set_constants(consts);

        let mut plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(203)], 1);
        plain_call.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let mut call = mk_op(OpCode::CallAssemblerR, &[OpRef::ref_op(1)], 2);
        call.descr = Some(make_call_assembler_descr(
            &callee_token,
            vec![Type::Ref],
            Type::Ref,
        ));
        let caller_ops = vec![
            plain_call,
            call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(2)], OpRef::NONE.raw()),
        ];
        let mut caller_token = JitCellToken::new(1611);
        backend
            .compile_loop(&[], &caller_ops, &mut caller_token)
            .unwrap();

        let frame = backend.execute_token(&caller_token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_plain_call_returns_fresh_gc_ref_without_call_assembler() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(205, alloc_marked_ref as *const () as usize as i64);
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let mut plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(205)], 0);
        plain_call.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            plain_call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];
        let mut token = JitCellToken::new(1612);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    #[test]
    #[cfg(target_arch = "aarch64")]
    #[ignore = "find_descr_by_ptr 0x0 — backend FINISH/CALL_ASSEMBLER/varsize-alloc paths do not yet write jf_descr in this test harness; tracked under Task #11/#21/#22 (CALL_ASSEMBLER red-only + recursive frame contract)"]
    fn test_plain_call_preserves_ref_result_across_collecting_helper_call() {
        install_test_libc_jitframe_tracer();
        let mut gc = MiniMarkGC::with_config(GcConfig {
            nursery_size: 96,
            large_object_threshold: 1024,
            ..GcConfig::default()
        });
        let jitframe_tid = gc.register_type(majit_backend::jitframe::jitframe_type_info());
        let payload_tid = gc.register_type(TypeInfo::simple(16));

        crate::set_jitframe_gc_type_id(jitframe_tid);
        install_call_assembler_test_layout();
        TEST_HELPER_ALLOC_TYPE_ID.store(payload_tid, Ordering::Relaxed);

        let mut backend = DynasmBackend::new();
        let mut consts = HashMap::new();
        consts.insert(
            206,
            alloc_marked_ref_collecting as *const () as usize as i64,
        );
        backend.set_constants(consts);
        backend.set_gc_allocator(Box::new(gc));

        let mut plain_call = mk_op(OpCode::CallR, &[OpRef::int_op(206)], 0);
        plain_call.descr = Some(make_plain_call_descr(vec![], Type::Ref));
        let ops = vec![
            plain_call,
            mk_op(OpCode::Finish, &[OpRef::ref_op(0)], OpRef::NONE.raw()),
        ];
        let mut token = JitCellToken::new(1613);
        backend.compile_loop(&[], &ops, &mut token).unwrap();

        let frame = backend.execute_token(&token, &[]);
        assert!(backend.get_latest_descr(&frame).is_finish());
        let result = backend.get_ref_value(&frame, 0);
        assert!(!result.is_null());
        unsafe {
            assert_eq!(*(result.0 as *const i64), TEST_HELPER_MARKER);
        }
    }

    #[test]
    fn test_jit_threadlocalref_base_round_trips_slot_contents() {
        crate::jit_threadlocalref_set(0, 0x1234);
        crate::jit_threadlocalref_set(8, 0x5678);
        let base = crate::jit_threadlocalref_base();
        assert!(!base.is_null());
        unsafe {
            assert_eq!(*base.add(0), 0x1234);
            assert_eq!(*base.add(1), 0x5678);
        }
    }
}

// ── rewrite.py:489 parity: inject str_descr/unicode_descr ──
//
// Token semantics come from `symbolic.get_array_token(rstr.STR/UNICODE, ...)`
// and `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` (see
// `rpython/jit/backend/llsupport/symbolic.py:7,29`). The layout encoded by
// `rstr.STR.become(GcStruct('rpy_string', ('hash', Signed), ('chars',
// Array(Char, hints={'extra_item_after_alloc': 1}))))`
// (`rpython/rtyper/lltypesystem/rstr.py:1226`) is:
//
//   [ hash (WORD) | chars.length (WORD) | chars[0..n] | +1 extra null ]
//
// `get_array_token` returns `basesize = before_array_part +
// carray.items.offset + extra_item_after_alloc`, so for STR the token
// `basesize` is 17 (not 16) — rewrite.py:295 then subtracts 1 for the
// extra null character when emitting STR{GET,SET}ITEM. UNICODE has no
// `extra_item_after_alloc` hint, so its token `basesize` is 16.
//
// Hash lives in its own field (the `hash` struct member), separate from
// the array tail. rewrite.py:283-294 reads it with
// `get_field_token(..., 'hash', ...)`, not `get_array_token(...)`.

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...).offset`.
const BUILTIN_STRING_HASH_OFFSET: usize = 0;
/// `symbolic.get_field_token(..., 'hash', ...).size` — assert == WORD at
/// rewrite.py:286,292.
const BUILTIN_STRING_HASH_SIZE: usize = std::mem::size_of::<usize>();
/// `symbolic.get_array_token(rstr.STR/UNICODE, ...).ofs_length` =
/// `before_array_part + carray.length.offset`.
const BUILTIN_STRING_LEN_OFFSET: usize = std::mem::size_of::<usize>();
/// STR token `basesize` — `before_array_part(8) + carray.items.offset(8) +
/// extra_item_after_alloc(1) = 17`.
const BUILTIN_STR_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>() + 1;
/// UNICODE token `basesize` — `before_array_part(8) + carray.items.offset(8)
/// = 16` (no extra_item_after_alloc).
const BUILTIN_UNICODE_TOKEN_BASE_SIZE: usize = 2 * std::mem::size_of::<usize>();

#[derive(Debug)]
struct BuiltinFieldDescr {
    offset: usize,
    field_size: usize,
    field_type: Type,
    signed: bool,
}

impl majit_ir::Descr for BuiltinFieldDescr {
    fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self)
    }
}

impl majit_ir::FieldDescr for BuiltinFieldDescr {
    fn offset(&self) -> usize {
        self.offset
    }
    fn field_size(&self) -> usize {
        self.field_size
    }
    fn field_type(&self) -> Type {
        self.field_type
    }
    fn is_field_signed(&self) -> bool {
        self.signed
    }
}

#[derive(Debug)]
struct BuiltinArrayDescr {
    base_size: usize,
    item_size: usize,
    type_id: u32,
    item_type: Type,
    signed: bool,
    len_descr: Arc<BuiltinFieldDescr>,
}

impl majit_ir::Descr for BuiltinArrayDescr {
    fn as_array_descr(&self) -> Option<&dyn majit_ir::ArrayDescr> {
        Some(self)
    }
}

impl majit_ir::ArrayDescr for BuiltinArrayDescr {
    fn base_size(&self) -> usize {
        self.base_size
    }
    fn item_size(&self) -> usize {
        self.item_size
    }
    fn type_id(&self) -> u32 {
        self.type_id
    }
    fn item_type(&self) -> Type {
        self.item_type
    }
    fn is_item_signed(&self) -> bool {
        self.signed
    }
    fn len_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
        Some(self.len_descr.as_ref())
    }
}

/// `symbolic.get_array_token(rstr.STR/UNICODE, ...)` token triple wrapped
/// as an `ArrayDescr`.  Fed to NEW{STR,UNICODE} / STR{LEN,GETITEM,SETITEM}
/// / UNICODE{LEN,GETITEM,SETITEM} / COPY{STR,UNICODE}CONTENT — every op
/// that upstream dispatches through `get_array_token` at
/// `rewrite.py:273-318`.  STR{,UNICODE}HASH takes a separate FieldDescr
/// (see `builtin_string_hash_field_descr` below).
fn builtin_string_array_descr(opcode: majit_ir::OpCode) -> Option<majit_ir::DescrRef> {
    use majit_ir::OpCode;
    let (base_size, item_size) = match opcode {
        OpCode::Newstr
        | OpCode::Strlen
        | OpCode::Strgetitem
        | OpCode::Strsetitem
        | OpCode::Copystrcontent => (BUILTIN_STR_TOKEN_BASE_SIZE, 1),
        OpCode::Newunicode
        | OpCode::Unicodelen
        | OpCode::Unicodegetitem
        | OpCode::Unicodesetitem
        | OpCode::Copyunicodecontent => (BUILTIN_UNICODE_TOKEN_BASE_SIZE, 4),
        _ => return None,
    };
    let len_descr = Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_LEN_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        signed: false,
    });
    Some(Arc::new(BuiltinArrayDescr {
        base_size,
        item_size,
        type_id: 0,
        item_type: Type::Int,
        signed: false,
        len_descr,
    }))
}

/// `symbolic.get_field_token(rstr.STR/UNICODE, 'hash', ...)` wrapped as a
/// FieldDescr.  rewrite.py:283-294 reads STRHASH/UNICODEHASH via
/// `get_field_token`, not `get_array_token`.  Kept separate so the two
/// upstream token helpers have independent pyre counterparts.
fn builtin_string_hash_field_descr(opcode: majit_ir::OpCode) -> Option<majit_ir::DescrRef> {
    use majit_ir::OpCode;
    if !matches!(opcode, OpCode::Strhash | OpCode::Unicodehash) {
        return None;
    }
    Some(Arc::new(BuiltinFieldDescr {
        offset: BUILTIN_STRING_HASH_OFFSET,
        field_size: BUILTIN_STRING_HASH_SIZE,
        field_type: Type::Int,
        // rewrite.py:288,293 pass `sign=True` for STR/UNICODE hash — the
        // `hash` struct field is `Signed`.
        signed: true,
    }))
}

fn inject_builtin_string_descrs(ops: &mut [Op]) {
    for op in ops {
        if op.descr.is_some() {
            continue;
        }
        if let Some(descr) = builtin_string_array_descr(op.opcode) {
            op.descr = Some(descr);
        } else if let Some(descr) = builtin_string_hash_field_descr(op.opcode) {
            op.descr = Some(descr);
        }
    }
}
